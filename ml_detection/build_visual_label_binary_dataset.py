"""Build a copy-only binary image dataset from visual label CSVs.

The primary labels CSV is expected to contain:
- source_path: absolute image path
- hole_id
- depth
- class_label

This script creates class folders for one target visual class at a time:
target_label vs Other. It supports explicit negative feedback so intervals
marked "not this class" become hard negatives for that classifier.
Source images are only copied, never moved/deleted.

Optional feedback CSV columns:
- source_path
- hole_id
- depth
- target_label
- feedback: positive/negative, yes/no, correct/incorrect, 1/0, true/false

Recommended loop:
1. Train one binary model per visual tag.
2. Run predictions.
3. Mark false positives as negative feedback for that target_label.
4. Rebuild/retrain; those hard negatives are kept in Other.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
IMAGE_KEY_PATTERN = re.compile(
    r"(?P<hole>[A-Za-z]{1,4}\d{3,5})_CC_(?P<depth>\d+(?:[pP]\d+|\.\d+)?)",
    re.IGNORECASE,
)


def safe_name(value: str) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._") or "label"


def canonical_label(value: str) -> str:
    text = str(value).strip().replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def display_label(value: str) -> str:
    return safe_name(str(value).strip().replace("_", " "))


def normalise_path(value: str) -> str:
    return str(Path(str(value)).resolve()).lower()


def parse_image_key(path: Path) -> tuple[str | None, float | None]:
    match = IMAGE_KEY_PATTERN.search(path.name)
    if not match:
        return None, None
    hole_id = match.group("hole").upper()
    depth = float(match.group("depth").replace("p", ".").replace("P", "."))
    return hole_id, depth


def scan_negative_candidates(image_root: Path, positive_paths: set[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for path in image_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if normalise_path(str(path)) in positive_paths:
            continue
        if "_dry" not in path.stem.lower():
            continue
        hole_id, depth = parse_image_key(path)
        if not hole_id:
            continue
        rows.append(
            {
                "source_path": str(path.resolve()),
                "hole_id": hole_id,
                "depth": depth,
                "class_label": "Other",
            }
        )
    return pd.DataFrame(rows)


def load_feedback_rows(feedback_csv: Path | None, target_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not feedback_csv:
        return pd.DataFrame(), pd.DataFrame()
    if not feedback_csv.exists():
        raise FileNotFoundError(f"Feedback CSV not found: {feedback_csv}")

    feedback = pd.read_csv(feedback_csv)
    required = {"source_path", "target_label", "feedback"}
    missing = required - set(feedback.columns)
    if missing:
        raise ValueError(f"Feedback CSV missing required columns: {sorted(missing)}")

    target_key = canonical_label(target_label)
    feedback = feedback[
        feedback["target_label"].astype(str).map(canonical_label).eq(target_key)
    ].copy()
    if feedback.empty:
        return pd.DataFrame(), pd.DataFrame()

    positive_tokens = {"positive", "pos", "yes", "y", "true", "1", "correct", "right"}
    negative_tokens = {"negative", "neg", "no", "n", "false", "0", "incorrect", "wrong", "not"}
    token = feedback["feedback"].astype(str).str.strip().str.casefold()
    positives = feedback[token.isin(positive_tokens)].copy()
    negatives = feedback[token.isin(negative_tokens)].copy()
    unknown = feedback[~token.isin(positive_tokens | negative_tokens)]
    if not unknown.empty:
        values = sorted(unknown["feedback"].astype(str).unique().tolist())
        raise ValueError(f"Unknown feedback values: {values}")

    for frame in (positives, negatives):
        if frame.empty:
            continue
        if "hole_id" not in frame.columns:
            frame["hole_id"] = ""
        if "depth" not in frame.columns:
            frame["depth"] = np.nan
        frame["class_label"] = target_label
    return positives, negatives


def prepare_label_rows(labels: pd.DataFrame, target_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = labels.copy()
    labels["source_path"] = labels["source_path"].astype(str)
    labels["_label_key"] = labels["class_label"].astype(str).map(canonical_label)
    target_key = canonical_label(target_label)
    positives = labels[labels["_label_key"].eq(target_key)].copy()
    labelled_other = labels[~labels["_label_key"].eq(target_key)].copy()
    return positives, labelled_other


def sample_with_hole_cap(
    df: pd.DataFrame,
    n: int,
    per_hole_cap: int,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    shuffled = df.iloc[rng.permutation(len(df))].copy()
    if per_hole_cap > 0:
        shuffled = (
            shuffled.groupby("hole_id", group_keys=False)
            .head(per_hole_cap)
            .reset_index(drop=True)
        )
    if len(shuffled) < n:
        raise ValueError(
            f"Only {len(shuffled)} candidates available after hole cap; need {n}."
        )
    return shuffled.sample(n=n, random_state=random_state).reset_index(drop=True)


def unique_dest(dest_dir: Path, file_name: str, used: set[Path]) -> Path:
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


def copy_manifest(manifest: pd.DataFrame, overwrite: bool) -> pd.DataFrame:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--feedback-csv", type=Path)
    parser.add_argument("--target-label", required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--max-positive", type=int, default=0)
    parser.add_argument("--max-per-hole-other", type=int, default=4)
    parser.add_argument(
        "--include-labelled-other",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use rows labelled as other visual classes as hard negatives.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    labels = pd.read_csv(args.labels_csv)
    required = {"source_path", "hole_id", "depth", "class_label"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    target_label = args.target_label
    target_folder = display_label(target_label)
    other_folder = "Other"

    positives, labelled_other = prepare_label_rows(labels, target_label)
    feedback_positives, feedback_negatives = load_feedback_rows(args.feedback_csv, target_label)
    if not feedback_positives.empty:
        positives = pd.concat([positives, feedback_positives], ignore_index=True)
    positives["source_path"] = positives["source_path"].astype(str)
    positives = positives[positives["source_path"].map(lambda p: Path(p).exists())].copy()
    positives = positives.drop_duplicates("source_path").reset_index(drop=True)
    if args.max_positive > 0 and len(positives) > args.max_positive:
        positives = positives.sample(n=args.max_positive, random_state=args.random_state).reset_index(drop=True)
    if positives.empty:
        raise ValueError(f"No existing source images found for target label: {target_label}")

    positive_paths = {normalise_path(path) for path in positives["source_path"]}
    negative_parts: list[pd.DataFrame] = []

    if not feedback_negatives.empty:
        feedback_negatives["source_path"] = feedback_negatives["source_path"].astype(str)
        feedback_negatives = feedback_negatives[
            feedback_negatives["source_path"].map(lambda p: Path(p).exists())
        ].copy()
        negative_parts.append(feedback_negatives)

    if args.include_labelled_other and not labelled_other.empty:
        labelled_other = labelled_other[
            labelled_other["source_path"].map(lambda p: Path(p).exists())
        ].copy()
        negative_parts.append(labelled_other)

    explicit_negatives = (
        pd.concat(negative_parts, ignore_index=True)
        if negative_parts
        else pd.DataFrame(columns=["source_path", "hole_id", "depth", "class_label"])
    )
    if not explicit_negatives.empty:
        explicit_negatives = explicit_negatives[
            ~explicit_negatives["source_path"].map(normalise_path).isin(positive_paths)
        ]
        explicit_negatives = explicit_negatives.drop_duplicates("source_path")

    negative_n = int(round(len(positives) * args.negative_ratio))
    if len(explicit_negatives) >= negative_n:
        negatives = sample_with_hole_cap(
            explicit_negatives,
            n=negative_n,
            per_hole_cap=0,
            random_state=args.random_state,
        )
        random_negatives = pd.DataFrame()
    else:
        needed_random = negative_n - len(explicit_negatives)
        excluded_paths = positive_paths | {
            normalise_path(path) for path in explicit_negatives.get("source_path", [])
        }
        random_pool = scan_negative_candidates(args.image_root, excluded_paths)
        random_negatives = sample_with_hole_cap(
            random_pool,
            n=needed_random,
            per_hole_cap=args.max_per_hole_other,
            random_state=args.random_state,
        )
        negatives = pd.concat([explicit_negatives, random_negatives], ignore_index=True)

    positives["binary_class"] = target_folder
    negatives["binary_class"] = other_folder
    combined = pd.concat([positives, negatives], ignore_index=True)

    used: set[Path] = set()
    manifest_rows: list[dict] = []
    for _, row in combined.iterrows():
        source = Path(row["source_path"])
        class_name = row["binary_class"]
        hole_id = str(row["hole_id"]).upper().strip()
        try:
            depth = float(row["depth"])
        except (TypeError, ValueError):
            _, parsed_depth = parse_image_key(source)
            depth = parsed_depth if parsed_depth is not None else np.nan
        depth_token = f"{depth:.3f}".replace(".", "p") if np.isfinite(depth) else "unknown"
        dest_dir = args.output_dir / class_name
        dest = unique_dest(dest_dir, f"{hole_id}_CC_{depth_token}_{source.name}", used)
        manifest_rows.append(
            {
                "binary_class": class_name,
                "original_class_label": row.get("class_label", ""),
                "source_kind": (
                    "target_positive"
                    if class_name == target_folder
                    else (
                        "explicit_or_labelled_negative"
                        if str(row.get("source_path")) in set(explicit_negatives.get("source_path", []))
                        else "random_other"
                    )
                ),
                "source_path": str(source),
                "dest_path": str(dest),
                "hole_id": hole_id,
                "depth": depth,
                "copy_status": "planned",
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "visual_binary_copy_manifest.csv"
    summary_path = args.output_dir / "visual_binary_copy_summary.json"

    if args.execute:
        manifest = copy_manifest(manifest, overwrite=args.overwrite)

    manifest.to_csv(manifest_path, index=False)
    summary = {
        "labels_csv": str(args.labels_csv),
        "feedback_csv": str(args.feedback_csv) if args.feedback_csv else None,
        "target_label": target_label,
        "target_folder": target_folder,
        "image_root": str(args.image_root),
        "output_dir": str(args.output_dir),
        "execute": bool(args.execute),
        "class_counts": manifest["binary_class"].value_counts().sort_index().to_dict(),
        "hole_counts": manifest.groupby("binary_class")["hole_id"].nunique().sort_index().to_dict(),
        "source_kind_counts": manifest["source_kind"].value_counts().sort_index().to_dict(),
        "unique_source_images": int(manifest["source_path"].nunique()),
        "copy_status": manifest["copy_status"].value_counts().sort_index().to_dict(),
        "note": (
            "Other is built from explicit negative feedback, other labelled visual classes, "
            "and sampled dry approved images as needed. If labels are not exhaustive, sampled "
            "Other can still contain unlabelled positives; explicit negatives are the highest-value feedback."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
