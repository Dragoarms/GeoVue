"""Build and import imagegen fragmentation-mask batches for grain training.

Workflow:
1. build-sheets: create contact sheets from labelled compartment images.
2. Ask imagegen to create a fragmentation mask for each sheet.
3. split-masks: split returned sheet masks into per-compartment truth masks.
4. vectorize: convert truth masks into annotation JSON, instance masks, class masks, and previews.

Source compartment images are read only.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

from grain_segmentation_labeler import GrainLabelerState, ImageRecord, safe_stem


IMAGE_KEY_PATTERN = re.compile(
    r"(?P<hole>[A-Za-z]{1,4}\d{3,5})_CC_(?P<depth>\d+(?:[pP]\d+|\.\d+)?)",
    re.IGNORECASE,
)


@dataclass
class SheetItem:
    batch_index: int
    item_index: int
    row: int
    col: int
    source_path: Path
    hole_id: str
    depth: float | None
    original_width: int
    original_height: int
    tile_x: int
    tile_y: int
    tile_width: int
    tile_height: int
    image_x: int
    image_y: int
    image_width: int
    image_height: int
    truth_mask_path: Path


def normalise_label(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def parse_image_key(path: Path) -> tuple[str, float | None]:
    match = IMAGE_KEY_PATTERN.search(path.name)
    if not match:
        return "", None
    depth_text = match.group("depth").replace("P", ".").replace("p", ".")
    try:
        depth = float(depth_text)
    except ValueError:
        depth = None
    return match.group("hole").upper(), depth


def select_records(args: argparse.Namespace) -> list[ImageRecord]:
    csv_path = Path(args.images_csv)
    df = pd.read_csv(csv_path)
    if args.target_label:
        target = normalise_label(args.target_label)
        df = df[df[args.label_col].map(normalise_label) == target].copy()
    if getattr(args, "selection", "first") == "diverse":
        df = select_diverse_rows(df, args)
    elif args.count:
        df = df.head(args.count)

    records: list[ImageRecord] = []
    for _, row in df.iterrows():
        path = Path(str(row[args.path_col]))
        if not path.exists():
            continue
        hole_id = str(row.get("hole_id") or "").strip()
        depth_value = row.get("depth")
        try:
            depth = float(depth_value)
        except Exception:
            depth = None
        parsed_hole, parsed_depth = parse_image_key(path)
        if not hole_id:
            hole_id = parsed_hole
        if depth is None:
            depth = parsed_depth
        records.append(
            ImageRecord(
                index=len(records),
                path=path.resolve(),
                name=path.name,
                hole_id=hole_id,
                depth=depth,
            )
        )
    if not records:
        raise ValueError("No source records selected.")
    return records


def image_feature(path: Path) -> np.ndarray:
    with Image.open(path).convert("RGB") as image:
        thumb = image.resize((48, 96), Image.Resampling.BILINEAR)
    arr = np.asarray(thumb, dtype=np.float32) / 255.0
    brightness = arr.mean(axis=2)
    material = arr[brightness > np.percentile(brightness, 15)]
    if material.size == 0:
        material = arr.reshape(-1, 3)
    mean = material.mean(axis=0)
    std = material.std(axis=0)
    q25 = np.percentile(material, 25, axis=0)
    q75 = np.percentile(material, 75, axis=0)
    return np.concatenate([mean, std, q25, q75])


def select_diverse_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    count = int(args.count or len(df))
    rows: list[dict] = []
    for source_idx, row in df.iterrows():
        path = Path(str(row[args.path_col]))
        if not path.exists():
            continue
        hole = str(row.get("hole_id") or "").strip()
        depth_value = row.get("depth")
        try:
            depth = float(depth_value)
        except Exception:
            _, parsed_depth = parse_image_key(path)
            depth = float(parsed_depth) if parsed_depth is not None else 0.0
        rows.append(
            {
                "source_idx": source_idx,
                "path": path,
                "hole": hole or parse_image_key(path)[0],
                "depth": depth,
                "feature": image_feature(path),
            }
        )
    if len(rows) <= count:
        return df.loc[[row["source_idx"] for row in rows]].copy()

    depths = np.array([row["depth"] for row in rows], dtype=np.float32)
    depth_span = float(depths.max() - depths.min()) or 1.0
    features = np.vstack([row["feature"] for row in rows])
    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0)
    feature_std[feature_std == 0] = 1.0
    norm_features = (features - feature_mean) / feature_std
    norm_depth = ((depths - depths.min()) / depth_span)[:, None]
    vectors = np.hstack([norm_features, norm_depth * float(args.depth_weight)])

    selected: list[int] = []
    hole_counts: dict[str, int] = {}
    max_per_hole = int(args.max_per_hole or 2)

    # Seed with extremes so colour/depth tails are represented before the
    # farthest-point pass fills the middle.
    seed_candidates = [
        int(np.argmin(depths)),
        int(np.argmax(depths)),
        int(np.argmin(features[:, 0] + features[:, 1] + features[:, 2])),
        int(np.argmax(features[:, 0] + features[:, 1] + features[:, 2])),
        int(np.argmax(features[:, 0] - features[:, 2])),
        int(np.argmax(features[:, 1] - features[:, 0])),
    ]
    for idx in seed_candidates:
        if len(selected) >= count:
            break
        hole = rows[idx]["hole"]
        if idx in selected or hole_counts.get(hole, 0) >= max_per_hole:
            continue
        selected.append(idx)
        hole_counts[hole] = hole_counts.get(hole, 0) + 1

    while len(selected) < count:
        best_idx = None
        best_score = -np.inf
        selected_vectors = vectors[selected]
        for idx, row in enumerate(rows):
            if idx in selected:
                continue
            hole = row["hole"]
            if hole_counts.get(hole, 0) >= max_per_hole:
                continue
            distances = np.linalg.norm(selected_vectors - vectors[idx], axis=1)
            score = float(distances.min())
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            max_per_hole += 1
            continue
        selected.append(best_idx)
        hole = rows[best_idx]["hole"]
        hole_counts[hole] = hole_counts.get(hole, 0) + 1

    chosen_indices = [rows[idx]["source_idx"] for idx in selected]
    return df.loc[chosen_indices].copy()


def write_sheet_prompt(path: Path) -> None:
    path.write_text(
        """Imagegen prompt for these contact sheets:

Create a pure fragmentation mask for the visible individual rock chips/grains in this contact sheet.

Preserve the sheet geometry, grid layout, gutters, and aspect ratio.
Output a pure mask image, not an annotated photo.
Use solid black for the background, gutters, tray/background, labels, empty space, dust smear, and anything that is not an individual grain.
Fill each accepted grain as one solid bright white island.
Keep touching grains separated by thin black cracks/gaps.
Do not add text, labels, arrows, legends, outlines-only drawings, shadows, gradients, texture, or photo detail.
Do not merge adjacent grains.
Do not invent grains outside the visible chip material.

GeoVue will split this returned mask back into the same grid cells, then fragment connected bright islands into training polygons.
""",
        encoding="utf-8",
    )


def build_sheets(args: argparse.Namespace) -> None:
    records = select_records(args)
    output_dir = Path(args.batch_dir)
    sheets_dir = output_dir / "source_sheets"
    masks_dir = output_dir / "returned_sheet_masks"
    split_dir = output_dir / "split_truth_masks"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    cols = args.cols
    rows = args.rows
    per_sheet = cols * rows
    tile_w = args.tile_width
    tile_h = args.tile_height
    gutter = args.gutter
    label_h = args.label_height
    sheet_w = cols * tile_w + (cols + 1) * gutter
    sheet_h = rows * (tile_h + label_h) + (rows + 1) * gutter
    manifest_rows: list[dict] = []

    for batch_index, start in enumerate(range(0, len(records), per_sheet)):
        batch = records[start : start + per_sheet]
        sheet = Image.new("RGB", (sheet_w, sheet_h), "black")
        draw = ImageDraw.Draw(sheet)
        items: list[SheetItem] = []
        for local_index, record in enumerate(batch):
            row = local_index // cols
            col = local_index % cols
            tile_x = gutter + col * (tile_w + gutter)
            tile_y = gutter + row * (tile_h + label_h + gutter)
            with Image.open(record.path).convert("RGB") as image:
                ow, oh = image.size
                scale = min(tile_w / ow, tile_h / oh)
                iw = max(1, round(ow * scale))
                ih = max(1, round(oh * scale))
                resized = image.resize((iw, ih), Image.Resampling.LANCZOS)
            image_x = tile_x + (tile_w - iw) // 2
            image_y = tile_y + (tile_h - ih) // 2
            sheet.paste(resized, (image_x, image_y))
            label = f"{start + local_index + 1:02d} {record.hole_id} {record.depth:g}" if record.depth is not None else f"{start + local_index + 1:02d} {record.hole_id}"
            draw.rectangle((tile_x, tile_y + tile_h, tile_x + tile_w, tile_y + tile_h + label_h), fill="black")
            draw.text((tile_x + 6, tile_y + tile_h + 4), label, fill=(180, 180, 180))
            truth_path = split_dir / f"{safe_stem(record)}.png"
            item = SheetItem(
                batch_index=batch_index,
                item_index=start + local_index,
                row=row,
                col=col,
                source_path=record.path,
                hole_id=record.hole_id,
                depth=record.depth,
                original_width=ow,
                original_height=oh,
                tile_x=tile_x,
                tile_y=tile_y,
                tile_width=tile_w,
                tile_height=tile_h,
                image_x=image_x,
                image_y=image_y,
                image_width=iw,
                image_height=ih,
                truth_mask_path=truth_path,
            )
            items.append(item)
            manifest_rows.append(
                {
                    "batch_index": item.batch_index,
                    "item_index": item.item_index,
                    "row": item.row,
                    "col": item.col,
                    "source_path": str(item.source_path),
                    "hole_id": item.hole_id,
                    "depth": item.depth,
                    "original_width": item.original_width,
                    "original_height": item.original_height,
                    "sheet_path": str(sheets_dir / f"batch_{batch_index:03d}_source.png"),
                    "returned_mask_path": str(masks_dir / f"batch_{batch_index:03d}_mask.png"),
                    "truth_mask_path": str(item.truth_mask_path),
                    "tile_x": item.tile_x,
                    "tile_y": item.tile_y,
                    "tile_width": item.tile_width,
                    "tile_height": item.tile_height,
                    "image_x": item.image_x,
                    "image_y": item.image_y,
                    "image_width": item.image_width,
                    "image_height": item.image_height,
                }
            )
        sheet_path = sheets_dir / f"batch_{batch_index:03d}_source.png"
        sheet.save(sheet_path)
        (sheets_dir / f"batch_{batch_index:03d}_items.json").write_text(
            json.dumps([row for row in manifest_rows if row["batch_index"] == batch_index], indent=2),
            encoding="utf-8",
        )

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    write_sheet_prompt(output_dir / "imagegen_sheet_prompt.txt")
    print(f"Selected records: {len(records)}")
    print(f"Sheets: {(len(records) + per_sheet - 1) // per_sheet}")
    print(f"Batch dir: {output_dir.resolve()}")
    print(f"Prompt: {(output_dir / 'imagegen_sheet_prompt.txt').resolve()}")


def split_masks(args: argparse.Namespace) -> None:
    batch_dir = Path(args.batch_dir)
    manifest = pd.read_csv(batch_dir / "manifest.csv")
    rejected_batches = read_rejected_batches(batch_dir)
    copied = 0
    for _, row in manifest.iterrows():
        if int(row["batch_index"]) in rejected_batches:
            continue
        returned = Path(str(row["returned_mask_path"]))
        if not returned.exists():
            continue
        with Image.open(returned).convert("RGB") as sheet_mask:
            mask_w, mask_h = sheet_mask.size
            with Image.open(str(row["sheet_path"])) as source_sheet:
                sheet_w, sheet_h = source_sheet.size
            sx = mask_w / sheet_w
            sy = mask_h / sheet_h
            left = round(float(row["image_x"]) * sx)
            top = round(float(row["image_y"]) * sy)
            right = round((float(row["image_x"]) + float(row["image_width"])) * sx)
            bottom = round((float(row["image_y"]) + float(row["image_height"])) * sy)
            crop = sheet_mask.crop((left, top, right, bottom))
            crop = crop.resize((int(row["original_width"]), int(row["original_height"])), Image.Resampling.LANCZOS)
            out = Path(str(row["truth_mask_path"]))
            out.parent.mkdir(parents=True, exist_ok=True)
            crop.save(out)
            copied += 1
    print(f"Split truth masks: {copied}")


def read_rejected_batches(batch_dir: Path) -> set[int]:
    rejected_path = batch_dir / "rejected_batches.txt"
    if not rejected_path.exists():
        return set()
    rejected: set[int] = set()
    for line in rejected_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rejected.add(int(line.split()[0].replace("batch_", "")))
        except ValueError:
            continue
    return rejected


def copy_truth_masks(args: argparse.Namespace) -> None:
    batch_dir = Path(args.batch_dir)
    output_dir = Path(args.output_dir)
    truth_dir = output_dir / "imagegen_truth_masks"
    truth_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(batch_dir / "manifest.csv")
    copied = 0
    for _, row in manifest.iterrows():
        src = Path(str(row["truth_mask_path"]))
        if not src.exists():
            continue
        dst = truth_dir / src.name
        Image.open(src).save(dst)
        copied += 1
    print(f"Copied truth masks: {copied} -> {truth_dir.resolve()}")


def vectorize(args: argparse.Namespace) -> None:
    batch_dir = Path(args.batch_dir)
    output_dir = Path(args.output_dir)
    manifest = pd.read_csv(batch_dir / "manifest.csv")
    rows = [row for _, row in manifest.iterrows() if Path(str(row["truth_mask_path"])).exists()]
    records: list[ImageRecord] = []
    truth_by_index: dict[int, Path] = {}
    for idx, row in enumerate(rows):
        source = Path(str(row["source_path"]))
        records.append(
            ImageRecord(
                index=idx,
                path=source,
                name=source.name,
                hole_id=str(row["hole_id"]),
                depth=float(row["depth"]) if pd.notna(row["depth"]) else None,
            )
        )
        truth_by_index[idx] = Path(str(row["truth_mask_path"]))

    state = GrainLabelerState(records, output_dir)
    saved = 0
    total_objects = 0
    for idx, record in enumerate(records):
        with Image.open(truth_by_index[idx]) as mask:
            result = state.vectorize_truth_mask(idx, mask, label=args.label, source=f"batch_truth:{truth_by_index[idx].name}")
        state.save_annotation(idx, result["objects"])
        saved += 1
        total_objects += int(result["object_count"])
    print(f"Vectorized samples: {saved}")
    print(f"Total objects: {total_objects}")
    print(f"Output dir: {output_dir.resolve()}")


def verify(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    batch_dir = Path(args.batch_dir)
    manifest = pd.read_csv(batch_dir / "manifest.csv")
    rows = manifest.head(args.count) if args.count else manifest
    missing: list[str] = []
    bad: list[str] = []
    for _, row in rows.iterrows():
        source = Path(str(row["source_path"]))
        record = ImageRecord(
            index=0,
            path=source,
            name=source.name,
            hole_id=str(row["hole_id"]),
            depth=float(row["depth"]) if pd.notna(row["depth"]) else None,
        )
        stem = safe_stem(record)
        expected = [
            output_dir / "imagegen_truth_masks" / f"{stem}.png",
            output_dir / "images" / f"{stem}.png",
            output_dir / "annotations" / f"{stem}.json",
            output_dir / "masks" / f"{stem}_masks.tif",
            output_dir / "class_masks" / f"{stem}_classes.png",
            output_dir / "previews" / f"{stem}_overlay.png",
        ]
        for path in expected:
            if not path.exists():
                missing.append(str(path))
        ann = output_dir / "annotations" / f"{stem}.json"
        if ann.exists():
            data = json.loads(ann.read_text(encoding="utf-8"))
            width = float(data.get("image_width", 0))
            height = float(data.get("image_height", 0))
            for obj in data.get("objects", []):
                for x, y in obj.get("points", []):
                    if x < 0 or y < 0 or x >= width or y >= height:
                        bad.append(str(ann))
                        break
            if data.get("objects"):
                clip_manifest = output_dir / "grain_clips" / stem / "clips.csv"
                if not clip_manifest.exists():
                    missing.append(str(clip_manifest))
    print(f"Checked rows: {len(rows)}")
    print(f"Missing files: {len(missing)}")
    print(f"Bad coordinate annotations: {len(set(bad))}")
    if missing:
        print("First missing:")
        for item in missing[:10]:
            print(item)
    if bad:
        print("Bad annotations:")
        for item in sorted(set(bad))[:10]:
            print(item)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--batch-dir", type=Path, default=Path("ml_detection/grain_segmentation_labels/biscuity_friable/imagegen_batch_50"))

    build = subparsers.add_parser("build-sheets", parents=[common])
    build.add_argument("--images-csv", type=Path, default=Path(r"C:\Users\gsymonds\Downloads\visual_labels.csv"))
    build.add_argument("--path-col", default="source_path")
    build.add_argument("--label-col", default="class_label")
    build.add_argument("--target-label", default="Biscuity_friable")
    build.add_argument("--count", type=int, default=50)
    build.add_argument("--selection", choices=["first", "diverse"], default="first")
    build.add_argument("--max-per-hole", type=int, default=2)
    build.add_argument("--depth-weight", type=float, default=2.0)
    build.add_argument("--cols", type=int, default=5)
    build.add_argument("--rows", type=int, default=2)
    build.add_argument("--tile-width", type=int, default=448)
    build.add_argument("--tile-height", type=int, default=1008)
    build.add_argument("--gutter", type=int, default=18)
    build.add_argument("--label-height", type=int, default=28)
    build.set_defaults(func=build_sheets)

    split = subparsers.add_parser("split-masks", parents=[common])
    split.set_defaults(func=split_masks)

    copy = subparsers.add_parser("copy-truth-masks", parents=[common])
    copy.add_argument("--output-dir", type=Path, default=Path("ml_detection/grain_segmentation_labels/biscuity_friable"))
    copy.set_defaults(func=copy_truth_masks)

    vec = subparsers.add_parser("vectorize", parents=[common])
    vec.add_argument("--output-dir", type=Path, default=Path("ml_detection/grain_segmentation_labels/biscuity_friable"))
    vec.add_argument("--label", default="grain")
    vec.set_defaults(func=vectorize)

    verify_parser = subparsers.add_parser("verify", parents=[common])
    verify_parser.add_argument("--output-dir", type=Path, default=Path("ml_detection/grain_segmentation_labels/biscuity_friable"))
    verify_parser.add_argument("--count", type=int, default=50)
    verify_parser.set_defaults(func=verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
