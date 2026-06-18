"""Create image contact sheets for BIFf population cluster review.

The script reads interval cluster CSVs produced by analyze_biff_population.py,
matches them to approved compartment images by HOLEID + depth_to, and writes
review contact sheets. It never modifies, moves, or copies source images.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


DEFAULT_REVIEW_DIR = Path("ml_detection/dataset_audit/biff_population_review")
DEFAULT_IMAGE_ROOT = Path(
    r"C:\Users\gsymonds\Fortescue Metals Group\Gabon - Belinga - Exploration Drilling\03 - Reverse Circulation\Chip Tray Photos\Extracted Compartment Images\Approved Compartment Images"
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
IMAGE_KEY_PATTERN = re.compile(
    r"(?P<hole>[A-Za-z]{1,4}\d{3,5})_CC_(?P<depth>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
LABEL_SUFFIX_PATTERN = re.compile(r"_(?P<label>Dry|Wet|Empty|Bad)(?:\.[^.]+)$", re.IGNORECASE)


def normalise_depth(value) -> str | None:
    try:
        depth = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(depth):
        return None
    return f"{depth:.3f}"


def folder_label(path: Path, image_root: Path) -> str | None:
    labels = {"dry": "Dry", "wet": "Wet", "empty": "Empty", "bad": "Bad"}
    try:
        parts = path.relative_to(image_root).parts[:-1]
    except ValueError:
        parts = path.parts[:-1]
    for part in reversed(parts):
        label = labels.get(part.lower())
        if label:
            return label
    match = LABEL_SUFFIX_PATTERN.search(path.name)
    return match.group("label").title() if match else None


def scan_images(image_root: Path, allowed_labels: set[str] | None) -> pd.DataFrame:
    rows: list[dict] = []
    for path in image_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        match = IMAGE_KEY_PATTERN.search(path.name)
        if not match:
            continue
        label = folder_label(path, image_root)
        if allowed_labels and label not in allowed_labels:
            continue
        depth_key = normalise_depth(match.group("depth"))
        if depth_key is None:
            continue
        rows.append(
            {
                "hole_key": match.group("hole").upper(),
                "depth_key": depth_key,
                "image_path": str(path.resolve()),
                "image_label": label,
                "image_name": path.name,
            }
        )
    return pd.DataFrame(rows)


def infer_cluster_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if c.endswith("_cluster_k5")]
    if len(candidates) != 1:
        raise ValueError(f"Expected one *_cluster_k5 column, found: {candidates}")
    return candidates[0]


def load_font(size: int) -> ImageFont.ImageFont:
    for font_name in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def make_contact_sheet(
    rows: pd.DataFrame,
    output_path: Path,
    title: str,
    tile_size: int = 240,
    cols: int = 4,
) -> None:
    font = load_font(13)
    title_font = load_font(18)
    label_h = 64
    pad = 10
    rows_count = int(np.ceil(len(rows) / cols))
    width = cols * tile_size + (cols + 1) * pad
    height = 44 + rows_count * (tile_size + label_h + pad) + pad
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, pad), title, fill="black", font=title_font)

    for i, (_, row) in enumerate(rows.iterrows()):
        x = pad + (i % cols) * (tile_size + pad)
        y = 44 + (i // cols) * (tile_size + label_h + pad)
        try:
            img = Image.open(row["image_path"]).convert("RGB")
            img.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (tile_size, tile_size), (245, 245, 245))
            canvas.paste(img, ((tile_size - img.width) // 2, (tile_size - img.height) // 2))
        except Exception:
            canvas = Image.new("RGB", (tile_size, tile_size), (230, 230, 230))
            ImageDraw.Draw(canvas).text((10, 10), "image error", fill="black", font=font)
        sheet.paste(canvas, (x, y))

        label = (
            f"{row['hole_id']} {row['depth_to']:.0f}m {row.get('image_label', '')}\n"
            f"Fe {row['fe']:.1f} Si {row['sio2']:.1f} Al {row['al2o3']:.1f}\n"
            f"P {row['p']:.3f} LOI {row['loi']:.1f}"
        )
        draw.multiline_text((x, y + tile_size + 3), label, fill="black", font=font, spacing=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def build_sheets_for_scope(
    scope_csv: Path,
    image_index: pd.DataFrame,
    output_dir: Path,
    max_per_cluster: int,
    random_state: int,
) -> dict:
    scope = pd.read_csv(scope_csv)
    cluster_col = infer_cluster_column(scope)
    scope = scope.copy()
    scope["hole_key"] = scope["hole_id"].astype(str).str.upper().str.strip()
    scope["depth_key"] = scope["depth_to"].map(normalise_depth)
    matched = scope.merge(image_index, on=["hole_key", "depth_key"], how="inner")

    scope_name = scope_csv.stem.replace("_interval_features_with_clusters", "")
    outputs = []
    for cluster, group in matched.groupby(cluster_col, sort=True):
        sample_n = min(max_per_cluster, len(group))
        if sample_n == 0:
            continue
        sample = group.sample(n=sample_n, random_state=random_state).sort_values(["hole_id", "depth_to"])
        output_path = output_dir / f"{scope_name}_cluster_{cluster}.jpg"
        make_contact_sheet(
            sample,
            output_path,
            title=f"{scope_name} cluster {cluster} ({len(group)} matched images)",
        )
        outputs.append(
            {
                "scope": scope_name,
                "cluster": int(cluster),
                "matched_images": int(len(group)),
                "sheet_samples": int(sample_n),
                "path": str(output_path),
            }
        )
    return {
        "scope": scope_name,
        "cluster_col": cluster_col,
        "intervals": int(len(scope)),
        "matched_images": int(len(matched)),
        "outputs": outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--scopes", default="biff_only,biff_s_only,combined_biff_biff_s")
    parser.add_argument("--allowed-image-labels", default="Dry")
    parser.add_argument("--max-per-cluster", type=int, default=16)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    allowed_labels = {label.strip().title() for label in args.allowed_image_labels.split(",") if label.strip()}
    image_index = scan_images(args.image_root, allowed_labels or None)
    if image_index.empty:
        raise ValueError(f"No matching images found under {args.image_root}")

    output_dir = args.review_dir / "contact_sheets"
    scope_results = []
    for scope_name in [name.strip() for name in args.scopes.split(",") if name.strip()]:
        scope_csv = args.review_dir / f"{scope_name}_interval_features_with_clusters.csv"
        if not scope_csv.exists():
            raise FileNotFoundError(scope_csv)
        scope_results.append(
            build_sheets_for_scope(
                scope_csv,
                image_index,
                output_dir,
                args.max_per_cluster,
                args.random_state,
            )
        )

    summary = {
        "image_root": str(args.image_root),
        "allowed_image_labels": sorted(allowed_labels),
        "indexed_images": int(len(image_index)),
        "scopes": scope_results,
    }
    summary_path = output_dir / "contact_sheet_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
