"""Prepare or execute copy-only image samples from lithology cluster assignments.

The default behavior is a dry run: it writes a manifest but does not copy files.
Use --execute when the reviewed manifest is ready. This script never moves or
deletes source images.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ASSIGNMENTS = (
    Path("ml_detection")
    / "dataset_audit"
    / "lithology_clustering"
    / "rich_assay_norm_geophys_depth_weathering_cluster_assignments.parquet"
)
DEFAULT_OUTPUT_DIR = Path("ml_detection") / "lithology_classifier_images"
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


def infer_cluster_col(df: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Cluster column not found: {requested}")
        return requested
    cluster_cols = [c for c in df.columns if c.endswith("_cluster") and not c.endswith("_stage2_cluster")]
    if len(cluster_cols) == 1:
        return cluster_cols[0]
    if not cluster_cols:
        raise ValueError("No cluster column found. Pass --cluster-col.")
    raise ValueError(f"Multiple cluster columns found. Pass --cluster-col. Options: {cluster_cols}")


def folder_label(path: Path, image_root: Path) -> str | None:
    labels = {"dry": "Dry", "wet": "Wet", "empty": "Empty", "bad": "Bad"}
    for parent in [path.parent, *path.parents]:
        if parent == image_root.parent:
            break
        label = labels.get(parent.name.lower())
        if label:
            return label
    match = LABEL_SUFFIX_PATTERN.search(path.name)
    if match:
        return match.group("label").title()
    return None


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
                "HOLEID": match.group("hole").upper(),
                "DEPTH_KEY": depth_key,
                "image_path": str(path.resolve()),
                "image_label": label,
                "image_name": path.name,
            }
        )
    return pd.DataFrame(rows)


def safe_cluster_name(value) -> str:
    text = str(value)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._") or "cluster"


def unique_dest_path(dest_dir: Path, file_name: str, used: set[Path]) -> Path:
    candidate = dest_dir / file_name
    if candidate not in used:
        used.add(candidate)
        return candidate
    stem = Path(file_name).stem
    suffix = Path(file_name).suffix
    index = 2
    while True:
        candidate = dest_dir / f"{stem}_{index}{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        index += 1


def build_manifest(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    assignments = pd.read_parquet(args.assignments)
    cluster_col = infer_cluster_col(assignments, args.cluster_col)
    assignments = assignments.copy()
    assignments["HOLEID"] = assignments["HOLEID"].astype(str).str.upper().str.strip()
    assignments["DEPTH_KEY"] = assignments["SAMPTO"].map(normalise_depth)
    assignments = assignments.dropna(subset=["HOLEID", "DEPTH_KEY", cluster_col])

    allowed_labels = {label.strip().title() for label in args.allowed_image_labels.split(",") if label.strip()}
    image_index = scan_images(args.image_root, allowed_labels or None)
    if image_index.empty:
        raise ValueError(f"No matching images found under {args.image_root}")

    available = assignments.merge(image_index, on=["HOLEID", "DEPTH_KEY"], how="inner")
    if available.empty:
        raise ValueError("No assignment rows matched scanned image filenames by HOLEID + SAMPTO depth.")

    sampled_parts: list[pd.DataFrame] = []
    for cluster, group in available.groupby(cluster_col, sort=True):
        if len(group) < args.min_available_per_cluster:
            continue
        if args.max_per_hole_per_cluster > 0:
            hole_parts = []
            for _, hole_group in group.groupby("HOLEID", sort=False):
                sample_n = min(args.max_per_hole_per_cluster, len(hole_group))
                hole_parts.append(hole_group.sample(n=sample_n, random_state=args.random_state))
            group = pd.concat(hole_parts, ignore_index=True)
        sample_n = min(args.max_per_cluster, len(group))
        sampled_parts.append(group.sample(n=sample_n, random_state=args.random_state))
    if not sampled_parts:
        raise ValueError(
            f"No clusters had at least {args.min_available_per_cluster} matched images."
        )
    sampled = pd.concat(sampled_parts, ignore_index=True)

    rng = np.random.default_rng(args.random_state)
    sampled = sampled.iloc[rng.permutation(len(sampled))].reset_index(drop=True)
    used_destinations: set[Path] = set()
    manifest_rows: list[dict] = []
    for _, row in sampled.iterrows():
        cluster_name = safe_cluster_name(row[cluster_col])
        source = Path(row["image_path"])
        dest_dir = args.output_dir / cluster_name
        dest_name = f"{row['HOLEID']}_CC_{row['DEPTH_KEY'].replace('.', 'p')}_{source.name}"
        dest = unique_dest_path(dest_dir, dest_name, used_destinations)
        manifest_rows.append(
            {
                "cluster": cluster_name,
                "source_path": str(source),
                "dest_path": str(dest),
                "HOLEID": row["HOLEID"],
                "SAMPFROM": row["SAMPFROM"],
                "SAMPTO": row["SAMPTO"],
                "BESTSTRAT": row.get("BESTSTRAT"),
                "STRATUNIT": row.get("STRATUNIT"),
                "STRATSUM": row.get("STRATSUM"),
                "PROFILEZONATION": row.get("PROFILEZONATION"),
                "image_label": row.get("image_label"),
                "copy_status": "planned",
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    cluster_counts = manifest["cluster"].value_counts().sort_index().to_dict()
    per_cluster_holes = manifest.groupby("cluster")["HOLEID"].nunique().sort_index().to_dict()
    per_cluster_intervals = (
        manifest.groupby("cluster")[["HOLEID", "SAMPTO"]]
        .apply(lambda g: g.drop_duplicates().shape[0])
        .sort_index()
        .to_dict()
    )
    summary = {
        "assignments": str(args.assignments),
        "cluster_col": cluster_col,
        "image_root": str(args.image_root),
        "output_dir": str(args.output_dir),
        "matched_images": int(len(available)),
        "planned_copies": int(len(manifest)),
        "cluster_counts": {str(k): int(v) for k, v in cluster_counts.items()},
        "cluster_hole_counts": {str(k): int(v) for k, v in per_cluster_holes.items()},
        "cluster_unique_interval_counts": {
            str(k): int(v) for k, v in per_cluster_intervals.items()
        },
        "unique_source_images": int(manifest["source_path"].nunique()),
        "unique_holes": int(manifest["HOLEID"].nunique()),
        "execute": bool(args.execute),
    }
    return manifest, summary


def execute_copies(manifest: pd.DataFrame, overwrite: bool) -> pd.DataFrame:
    manifest = manifest.copy()
    for index, row in manifest.iterrows():
        source = Path(row["source_path"])
        dest = Path(row["dest_path"])
        if not source.exists():
            manifest.at[index, "copy_status"] = "missing_source"
            continue
        if dest.exists() and not overwrite:
            manifest.at[index, "copy_status"] = "exists_skipped"
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        manifest.at[index, "copy_status"] = "copied"
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--cluster-col", default=None)
    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help="Root folder containing source compartment images to copy from.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-per-cluster", type=int, default=120)
    parser.add_argument(
        "--max-per-hole-per-cluster",
        type=int,
        default=0,
        help="Optional cap on sampled images from one hole within each class/cluster.",
    )
    parser.add_argument("--min-available-per-cluster", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--allowed-image-labels",
        default="",
        help="Optional comma-separated source labels to allow, e.g. Dry.",
    )
    parser.add_argument("--execute", action="store_true", help="Actually copy files. Default is manifest only.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing destination files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest, summary = build_manifest(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.execute:
        manifest = execute_copies(manifest, args.overwrite)

    manifest_path = args.output_dir / "lithology_image_copy_manifest.csv"
    summary_path = args.output_dir / "lithology_image_copy_summary.json"
    manifest.to_csv(manifest_path, index=False)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote summary: {summary_path}")
    for cluster, count in summary["cluster_counts"].items():
        print(f"{cluster}: {count}")
    if not args.execute:
        print("Dry run only. Re-run with --execute to copy files.")


if __name__ == "__main__":
    main()
