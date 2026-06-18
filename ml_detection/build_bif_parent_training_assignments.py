"""Build parent BIF training assignments from consensus classifications.

Label priority:
1. Mineralised override: Fe > 50, SiO2 < 10, Al2O3 < 5.
2. Remaining consensus parent classes: BIFf, BIFf-S, BIFHm, Compact.

The output is an assignments parquet/csv suitable for lithology_sample_images.py.
No images are copied by this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_CLASSIFICATION_CSV = Path(
    r"C:\Users\gsymonds\Downloads\classifications_export_20260617_151205.csv"
)
DEFAULT_OUTPUT_DIR = Path("ml_detection/dataset_audit/bif_parent_training")

PARENT_MAP = {
    "biff": "C01_BIFf",
    "biff_s": "C02_BIFf-S",
    "bifhm": "C03_BIFHm",
    "compact": "C04_Compact",
}
MINERALISED_LABEL = "C00_Mineralised"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-csv", type=Path, default=DEFAULT_CLASSIFICATION_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.classification_csv, low_memory=False)
    cls = df["consensus_classification"].astype(str).str.lower().str.strip()

    parent_mask = cls.isin(PARENT_MAP)
    sub = df[parent_mask].copy()
    sub["_consensus_key"] = cls[parent_mask].values

    comments = sub.get("rev_gsymonds_comments", pd.Series(index=sub.index, dtype=object))
    comments = comments.astype(str).str.lower()
    bad_comment = comments.str.contains("no image|no photos|bad photo|bad photos|delete", na=False)
    sub = sub[~bad_comment].copy()

    sub["fe"] = pd.to_numeric(sub["fe_pct (exassay)"], errors="coerce")
    sub["sio2"] = pd.to_numeric(sub["sio2_pct (exassay)"], errors="coerce")
    sub["al2o3"] = pd.to_numeric(sub["al2o3_pct (exassay)"], errors="coerce")
    sub["mineralised_override"] = (
        (sub["fe"] > 50)
        & (sub["sio2"] < 10)
        & (sub["al2o3"] < 5)
    )

    sub["bif_parent_class"] = sub["_consensus_key"].map(PARENT_MAP)
    sub.loc[sub["mineralised_override"], "bif_parent_class"] = MINERALISED_LABEL

    sub["HOLEID"] = sub["hole_id"].astype(str).str.upper().str.strip()
    sub["SAMPFROM"] = pd.to_numeric(sub["depth_from"], errors="coerce")
    sub["SAMPTO"] = pd.to_numeric(sub["depth_to"], errors="coerce")
    sub["BESTSTRAT"] = sub["consensus_classification"]
    sub["STRATUNIT"] = sub["consensus_classification"]
    sub["STRATSUM"] = sub["consensus_classification"]
    sub["PROFILEZONATION"] = ""
    sub = sub.dropna(subset=["HOLEID", "SAMPFROM", "SAMPTO", "bif_parent_class"])

    output_cols = [
        "HOLEID",
        "SAMPFROM",
        "SAMPTO",
        "bif_parent_class",
        "BESTSTRAT",
        "STRATUNIT",
        "STRATSUM",
        "PROFILEZONATION",
        "consensus_classification",
        "agreement",
        "review_count",
        "rev_gsymonds_comments",
        "fe",
        "sio2",
        "al2o3",
        "mineralised_override",
    ]

    parquet_path = args.output_dir / "bif_parent_with_mineralised_assignments.parquet"
    csv_path = args.output_dir / "bif_parent_with_mineralised_assignments.csv"
    summary_path = args.output_dir / "bif_parent_with_mineralised_summary.json"

    sub[output_cols].to_parquet(parquet_path, index=False)
    sub[output_cols].to_csv(csv_path, index=False)

    summary = {
        "classification_csv": str(args.classification_csv),
        "rows": int(len(sub)),
        "class_counts": sub["bif_parent_class"].value_counts().sort_index().to_dict(),
        "class_hole_counts": sub.groupby("bif_parent_class")["HOLEID"].nunique().sort_index().to_dict(),
        "mineralised_override_by_original_consensus": (
            sub[sub["mineralised_override"]]
            ["consensus_classification"]
            .value_counts()
            .to_dict()
        ),
        "outputs": {
            "parquet": str(parquet_path),
            "csv": str(csv_path),
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
