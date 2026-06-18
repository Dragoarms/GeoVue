"""Cluster non-BIF/non-mineralised intervals from image embeddings only.

The default pass is deliberately conservative:
- exclude manual BIF-family classes and strict mineralised intervals
- use only source images labelled Dry
- cap samples per hole
- optionally convert images to grayscale before embedding to reduce colour/weathering dominance

Outputs are review artifacts only. Source images are never modified or copied.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torchvision import transforms
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch/torchvision are required for image clustering.") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_pipeline.model import load_checkpoint


DEFAULT_CLASSIFICATION_CSV = Path(
    r"C:\Users\gsymonds\Downloads\classifications_export_20260617_151205.csv"
)
DEFAULT_IMAGE_ROOT = Path(
    r"C:\Users\gsymonds\Fortescue Metals Group\Gabon - Belinga - Exploration Drilling\03 - Reverse Circulation\Chip Tray Photos\Extracted Compartment Images\Approved Compartment Images"
)
DEFAULT_CHECKPOINT = Path("ml_output/wetdryempty_gate/checkpoints/best_model.pt")
DEFAULT_OUTPUT = Path("ml_detection/dataset_audit/non_bif_image_clustering")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
IMAGE_KEY_PATTERN = re.compile(
    r"(?P<hole>[A-Za-z]{1,4}\d{3,5})_CC_(?P<depth>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
LABEL_SUFFIX_PATTERN = re.compile(r"_(?P<label>Dry|Wet|Empty|Bad)(?:\.[^.]+)$", re.IGNORECASE)
BIF_FAMILY_CLASSES = {"biff", "biff_s", "bifhm", "compact"}


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


def scan_images(image_root: Path, allowed_label: str = "Dry") -> pd.DataFrame:
    rows = []
    for path in image_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        match = IMAGE_KEY_PATTERN.search(path.name)
        if not match:
            continue
        label = folder_label(path, image_root)
        if label != allowed_label:
            continue
        depth_key = normalise_depth(match.group("depth"))
        if depth_key is None:
            continue
        rows.append(
            {
                "HOLEID": match.group("hole").upper(),
                "DEPTH_KEY": depth_key,
                "image_path": str(path.resolve()),
                "image_label": label,
                "image_name": path.name,
            }
        )
    return pd.DataFrame(rows)


def weathering_band(value) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if text in {"Le", "Hc"}:
        return "lateritic_oxide"
    if text == "De":
        return "depleted_weathered"
    if text == "Hy":
        return "hypogene"
    if text == "Un":
        return "unclassified_or_unweathered"
    return "unknown"


def load_intervals(classification_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(classification_csv, low_memory=False)
    cls = df["consensus_classification"].astype(str).str.lower().str.strip()
    fe = pd.to_numeric(df["fe_pct (exassay)"], errors="coerce")
    sio2 = pd.to_numeric(df["sio2_pct (exassay)"], errors="coerce")
    al2o3 = pd.to_numeric(df["al2o3_pct (exassay)"], errors="coerce")
    mineralised = (fe > 50) & (sio2 < 10) & (al2o3 < 5)

    keep = ~cls.isin(BIF_FAMILY_CLASSES) & ~mineralised & cls.ne("not_confident")
    out = df[keep].copy()
    out["HOLEID"] = out["hole_id"].astype(str).str.upper().str.strip()
    out["DEPTH_KEY"] = out["depth_to"].map(normalise_depth)
    out["weathering_band"] = out.get("profilezonation (exgeologyRC)", pd.Series(index=out.index)).map(weathering_band)
    out["fe"] = fe[keep]
    out["sio2"] = sio2[keep]
    out["al2o3"] = al2o3[keep]
    out["mgo"] = pd.to_numeric(out.get("mgo_pct (exassay)"), errors="coerce")
    out["cao"] = pd.to_numeric(out.get("cao_pct (exassay)"), errors="coerce")
    out["k2o"] = pd.to_numeric(out.get("k2o_pct (exassay)"), errors="coerce")
    return out.dropna(subset=["HOLEID", "DEPTH_KEY"])


def sample_rows(df: pd.DataFrame, max_images: int, max_per_hole: int, random_state: int) -> pd.DataFrame:
    capped = []
    for _, group in df.groupby("HOLEID", sort=False):
        n = min(max_per_hole, len(group)) if max_per_hole > 0 else len(group)
        capped.append(group.sample(n=n, random_state=random_state))
    sampled = pd.concat(capped)
    if len(sampled) > max_images:
        # Preserve broad weathering coverage as much as possible.
        parts = []
        per_band = max(1, max_images // max(1, sampled["weathering_band"].nunique()))
        for _, group in sampled.groupby("weathering_band", sort=True):
            n = min(per_band, len(group))
            parts.append(group.sample(n=n, random_state=random_state))
        sampled = pd.concat(parts)
        if len(sampled) > max_images:
            sampled = sampled.sample(n=max_images, random_state=random_state)
        elif len(sampled) < max_images:
            remainder = df.drop(sampled.index, errors="ignore")
            if len(remainder):
                add = remainder.sample(n=min(max_images - len(sampled), len(remainder)), random_state=random_state)
                sampled = pd.concat([sampled, add])
    return sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)


def build_transform(grayscale: bool, image_size: int = 224):
    steps = [transforms.Resize((image_size, image_size))]
    if grayscale:
        steps.append(transforms.Grayscale(num_output_channels=3))
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(steps)


def gate_filter(
    df: pd.DataFrame,
    checkpoint: Path,
    batch_size: int,
    device_name: str,
    target_label: str,
    min_confidence: float,
) -> tuple[pd.DataFrame, dict]:
    model, checkpoint_data = load_checkpoint(checkpoint)
    class_names = checkpoint_data.get("class_names", ["Dry", "Wet", "Empty"])
    class_names = [str(name) for name in class_names]
    label_lookup = {name.lower(): name for name in class_names}
    resolved_target = label_lookup.get(target_label.lower())
    if resolved_target is None:
        raise ValueError(f"Gate target label {target_label!r} not found in checkpoint classes: {class_names}")

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    model = model.to(device)
    model.eval()
    transform = build_transform(grayscale=False)

    pred_labels: list[str] = []
    confidences: list[float] = []
    batch = []
    with torch.no_grad():
        for path in df["image_path"]:
            image = Image.open(path).convert("RGB")
            batch.append(transform(image))
            if len(batch) == batch_size:
                x = torch.stack(batch).to(device)
                probs = torch.softmax(model(x), dim=1).detach().cpu().numpy()
                idx = probs.argmax(axis=1)
                pred_labels.extend(class_names[i] for i in idx)
                confidences.extend(float(row[i]) for row, i in zip(probs, idx))
                batch = []
        if batch:
            x = torch.stack(batch).to(device)
            probs = torch.softmax(model(x), dim=1).detach().cpu().numpy()
            idx = probs.argmax(axis=1)
            pred_labels.extend(class_names[i] for i in idx)
            confidences.extend(float(row[i]) for row, i in zip(probs, idx))

    gated = df.copy()
    gated["gate_label"] = pred_labels
    gated["gate_confidence"] = confidences
    keep = gated["gate_label"].str.lower().eq(resolved_target.lower()) & (gated["gate_confidence"] >= min_confidence)
    kept = gated[keep].copy().reset_index(drop=True)
    stats = {
        "gate_class_names": class_names,
        "gate_target_label": resolved_target,
        "gate_min_confidence": min_confidence,
        "gate_input_images": int(len(gated)),
        "gate_kept_images": int(len(kept)),
        "gate_prediction_counts": gated["gate_label"].value_counts().to_dict(),
    }
    return kept, stats


def extract_embeddings(df: pd.DataFrame, checkpoint: Path, batch_size: int, grayscale: bool, device_name: str) -> np.ndarray:
    model, _ = load_checkpoint(checkpoint)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    model = model.to(device)
    model.eval()
    transform = build_transform(grayscale)

    embeddings = []
    batch = []
    with torch.no_grad():
        for path in df["image_path"]:
            image = Image.open(path).convert("RGB")
            batch.append(transform(image))
            if len(batch) == batch_size:
                x = torch.stack(batch).to(device)
                features = model.get_features(x).detach().cpu().numpy()
                embeddings.append(features)
                batch = []
        if batch:
            x = torch.stack(batch).to(device)
            features = model.get_features(x).detach().cpu().numpy()
            embeddings.append(features)
    return np.vstack(embeddings)


def top_counts(series: pd.Series, n: int = 5) -> str:
    counts = series.dropna().astype(str).value_counts().head(n)
    return "; ".join(f"{idx}:{val}" for idx, val in counts.items())


def summarize(df: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    rows = []
    for cluster, group in df.groupby(cluster_col, sort=True):
        rows.append(
            {
                "cluster": int(cluster),
                "count": int(len(group)),
                "holes": int(group["HOLEID"].nunique()),
                "top_weathering_band": top_counts(group["weathering_band"]),
                "top_strat": top_counts(group.get("strat (exgeologyRC)", pd.Series(dtype=object))),
                "fe_median": group["fe"].median(),
                "sio2_median": group["sio2"].median(),
                "al2o3_median": group["al2o3"].median(),
                "mgo_median": group["mgo"].median(),
                "cao_median": group["cao"].median(),
                "k2o_median": group["k2o"].median(),
            }
        )
    return pd.DataFrame(rows)


def load_font(size: int) -> ImageFont.ImageFont:
    for font_name in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def make_contact_sheet(rows: pd.DataFrame, output_path: Path, title: str, tile_size: int = 220, cols: int = 4) -> None:
    font = load_font(12)
    title_font = load_font(18)
    label_h = 72
    pad = 10
    n_rows = int(np.ceil(len(rows) / cols))
    width = cols * tile_size + (cols + 1) * pad
    height = 44 + n_rows * (tile_size + label_h + pad) + pad
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, pad), title, fill="black", font=title_font)

    for i, (_, row) in enumerate(rows.iterrows()):
        x = pad + (i % cols) * (tile_size + pad)
        y = 44 + (i // cols) * (tile_size + label_h + pad)
        img = Image.open(row["image_path"]).convert("RGB")
        img.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (tile_size, tile_size), (245, 245, 245))
        canvas.paste(img, ((tile_size - img.width) // 2, (tile_size - img.height) // 2))
        sheet.paste(canvas, (x, y))
        text = (
            f"{row['HOLEID']} {float(row['depth_to']):.0f}m {row['weathering_band']}\n"
            f"{row.get('strat (exgeologyRC)', '')}\n"
            f"Fe {row['fe']:.1f} Si {row['sio2']:.1f} Al {row['al2o3']:.1f}"
        )
        draw.multiline_text((x, y + tile_size + 3), text, fill="black", font=font, spacing=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-csv", type=Path, default=DEFAULT_CLASSIFICATION_CSV)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-images", type=int, default=6000)
    parser.add_argument("--max-per-hole", type=int, default=4)
    parser.add_argument("--chosen-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--color", action="store_true", help="Use colour images. Default is grayscale texture-biased pass.")
    parser.add_argument("--contact-samples-per-cluster", type=int, default=20)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    intervals = load_intervals(args.classification_csv)
    images = scan_images(args.image_root, allowed_label="Dry")
    matched = intervals.merge(images, on=["HOLEID", "DEPTH_KEY"], how="inner")
    if matched.empty:
        raise ValueError("No dry images matched non-BIF intervals by HOLEID + depth_to.")

    sampled = sample_rows(matched, args.max_images, args.max_per_hole, args.random_state)
    embeddings = extract_embeddings(
        sampled,
        args.checkpoint,
        args.batch_size,
        grayscale=not args.color,
        device_name=args.device,
    )

    scaler = StandardScaler()
    x = scaler.fit_transform(embeddings)
    pca50 = PCA(n_components=min(50, x.shape[1]), random_state=args.random_state)
    x_reduced = pca50.fit_transform(x)

    diagnostics = []
    for k in range(4, 13):
        labels = KMeans(n_clusters=k, random_state=args.random_state, n_init=30).fit_predict(x_reduced)
        diagnostics.append({"k": k, "silhouette": silhouette_score(x_reduced, labels)})

    labels = KMeans(n_clusters=args.chosen_k, random_state=args.random_state, n_init=50).fit_predict(x_reduced)
    sampled["image_cluster_k10"] = labels
    pca2 = PCA(n_components=2, random_state=args.random_state).fit_transform(x_reduced)
    sampled["pca1"] = pca2[:, 0]
    sampled["pca2"] = pca2[:, 1]

    summary = summarize(sampled, "image_cluster_k10")
    diagnostics_df = pd.DataFrame(diagnostics)

    sampled.to_csv(args.output_dir / "non_bif_image_cluster_assignments.csv", index=False)
    summary.to_csv(args.output_dir / "non_bif_image_cluster_k10_summary.csv", index=False)
    diagnostics_df.to_csv(args.output_dir / "non_bif_image_cluster_diagnostics.csv", index=False)
    np.save(args.output_dir / "non_bif_image_embeddings.npy", embeddings)

    sheet_outputs = []
    for cluster, group in sampled.groupby("image_cluster_k10", sort=True):
        n = min(args.contact_samples_per_cluster, len(group))
        sheet_sample = group.sample(n=n, random_state=args.random_state).sort_values(["HOLEID", "depth_to"])
        sheet_path = args.output_dir / "contact_sheets" / f"image_cluster_{cluster}.jpg"
        make_contact_sheet(sheet_sample, sheet_path, f"image cluster {cluster} ({len(group)} samples)")
        sheet_outputs.append(str(sheet_path))

    report = {
        "classification_csv": str(args.classification_csv),
        "image_root": str(args.image_root),
        "checkpoint": str(args.checkpoint),
        "mode": "color" if args.color else "grayscale_texture_biased",
        "matched_dry_non_bif_images": int(len(matched)),
        "sampled_images": int(len(sampled)),
        "holes": int(sampled["HOLEID"].nunique()),
        "chosen_k": args.chosen_k,
        "cluster_counts": sampled["image_cluster_k10"].value_counts().sort_index().to_dict(),
        "diagnostics": str(args.output_dir / "non_bif_image_cluster_diagnostics.csv"),
        "summary": str(args.output_dir / "non_bif_image_cluster_k10_summary.csv"),
        "assignments": str(args.output_dir / "non_bif_image_cluster_assignments.csv"),
        "contact_sheets": sheet_outputs,
    }
    with (args.output_dir / "non_bif_image_cluster_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print("\nCluster summary:")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
