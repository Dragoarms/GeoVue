"""Build reviewable grade/weathering axes for tagged BIFf intervals.

This is a second-pass script over analyze_biff_population.py outputs. It does
not copy images or modify source data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_REVIEW_DIR = Path("ml_detection/dataset_audit/biff_population_review")
DEFAULT_INTERVALS = DEFAULT_REVIEW_DIR / "combined_biff_biff_s_interval_features_with_clusters.csv"


def n(row: pd.Series, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    return float(value)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in [
        "fe",
        "sio2",
        "al2o3",
        "p",
        "loi",
        "tio2",
        "mn",
        "s",
        "GIBBSITE",
        "KAOLINITE",
        "GOETHITE",
        "HEMATITE",
        "CHERT",
        "PYRITE",
        "APATITE",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["gibbsite_kaolinite"] = df.get("GIBBSITE", 0).fillna(0) + df.get("KAOLINITE", 0).fillna(0)
    df["hematite_goethite"] = df.get("HEMATITE", 0).fillna(0) + df.get("GOETHITE", 0).fillna(0)
    df["clean_mineralised_rule"] = (
        (df["fe"] > 50)
        & (df["sio2"] < 10)
        & (df["al2o3"] < 5)
    )
    df["aluminous_high_grade_rule"] = (
        (df["fe"] > 50)
        & (df["sio2"] < 10)
        & (df["al2o3"] >= 5)
    )
    df["bifhm_like_rule_v2"] = (df["fe"] < 40) & (df["sio2"] > 35)
    return df


def grade_axis(row: pd.Series) -> str:
    fe = n(row, "fe", np.nan)
    si = n(row, "sio2", np.nan)
    al = n(row, "al2o3", np.nan)
    chert = n(row, "CHERT")

    if not np.isfinite(fe) or not np.isfinite(si) or not np.isfinite(al):
        return "unclassified_grade"
    if fe > 50 and si < 10 and al < 5:
        return "high_grade_clean_low_al"
    if fe > 50 and si < 10 and al >= 5:
        return "high_grade_aluminous_weathered"
    if si >= 30 or chert >= 25 or (fe < 45 and si >= 20):
        return "siliceous_low_al_biff"
    if fe >= 50 and si < 25:
        return "hematite_rich_intermediate"
    return "intermediate_biff"


def weathering_axis(row: pd.Series) -> str:
    al = n(row, "al2o3", np.nan)
    loi = n(row, "loi", np.nan)
    gk = n(row, "gibbsite_kaolinite")
    goethite = n(row, "GOETHITE")
    hematite = n(row, "HEMATITE")
    pyrite = n(row, "PYRITE")
    sulphur = n(row, "s")

    if pyrite >= 1.0 or sulphur >= 0.20:
        return "reduced_sulphidic"
    if al >= 10 or loi >= 8 or gk >= 20:
        return "clay_weathered"
    if goethite >= 25 or al >= 5 or loi >= 4 or gk >= 8:
        return "oxide_hydrated"
    if hematite >= 45 and al < 5 and loi < 4 and gk < 8:
        return "fresh_hypogene"
    return "transitional_weathering"


def recommended_image_class(row: pd.Series) -> str:
    if bool(row.get("clean_mineralised_rule", False)):
        return "Mineralised"

    grade = row["grade_axis"]
    weather = row["weathering_axis"]

    if grade in {"high_grade_clean_low_al", "high_grade_aluminous_weathered"}:
        return "biff_hg_weathered_or_goethitic"
    if grade == "siliceous_low_al_biff":
        return "biff_siliceous_low_al"
    if weather == "clay_weathered":
        return "biff_clay_weathered"
    if weather == "oxide_hydrated":
        return "biff_oxide_hydrated"
    return "biff_intermediate"


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    numeric = [
        "fe",
        "sio2",
        "al2o3",
        "p",
        "loi",
        "GIBBSITE",
        "KAOLINITE",
        "GOETHITE",
        "HEMATITE",
        "CHERT",
        "PYRITE",
        "gamma_mean_hole_z",
    ]
    aggregations = {
        "intervals": ("hole_id", "size"),
        "holes": ("hole_id", "nunique"),
        "biff_pct": ("source_classification", lambda s: s.eq("biff").mean() * 100),
        "biff_s_pct": ("source_classification", lambda s: s.eq("biff_s").mean() * 100),
        "clean_mineralised_count": ("clean_mineralised_rule", "sum"),
        "aluminous_high_grade_count": ("aluminous_high_grade_rule", "sum"),
        "bifhm_like_count": ("bifhm_like_rule_v2", "sum"),
    }
    for col in numeric:
        if col in df.columns:
            aggregations[f"{col}_median"] = (col, "median")
            aggregations[f"{col}_p10"] = (col, lambda s: s.quantile(0.10))
            aggregations[f"{col}_p90"] = (col, lambda s: s.quantile(0.90))
    return df.groupby(group_cols, dropna=False).agg(**aggregations).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intervals", type=Path, default=DEFAULT_INTERVALS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    args = parser.parse_args()

    df = pd.read_csv(args.intervals)
    df = add_derived_columns(df)
    df["grade_axis"] = df.apply(grade_axis, axis=1)
    df["weathering_axis"] = df.apply(weathering_axis, axis=1)
    df["recommended_image_class"] = df.apply(recommended_image_class, axis=1)
    df["resolved_parent_class"] = df["source_classification"]
    df.loc[df["clean_mineralised_rule"], "resolved_parent_class"] = "mineralised"
    df["axis_label"] = df["weathering_axis"] + "__" + df["grade_axis"]

    df["review_flag"] = ""
    df.loc[df["aluminous_high_grade_rule"], "review_flag"] += "aluminous_high_grade_rule;"
    df.loc[df["clean_mineralised_rule"], "review_flag"] += "mineralised_override;"
    df.loc[
        df["source_classification"].eq("biff_s")
        & df["weathering_axis"].eq("fresh_hypogene")
        & ~df["clean_mineralised_rule"],
        "review_flag",
    ] += "fresh_biff_s_check;"
    df.loc[df["source_classification"].eq("biff") & df["weathering_axis"].isin(["clay_weathered", "oxide_hydrated"]), "review_flag"] += "weathered_biff_check;"
    if "pca1" in df.columns:
        df.loc[df["pca1"] < -15, "review_flag"] += "pca_outlier;"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    interval_path = args.output_dir / "biff_axis_interval_labels.csv"
    image_class_summary_path = args.output_dir / "biff_recommended_image_class_summary.csv"
    axis_summary_path = args.output_dir / "biff_axis_summary.csv"
    parent_axis_path = args.output_dir / "biff_axis_by_parent_class.csv"
    cluster_axis_path = args.output_dir / "biff_axis_by_cluster.csv"
    outlier_path = args.output_dir / "biff_axis_review_flags.csv"

    export_cols = [
        "hole_id",
        "depth_from",
        "depth_to",
        "source_classification",
        "resolved_parent_class",
        "grade_axis",
        "weathering_axis",
        "recommended_image_class",
        "axis_label",
        "review_flag",
        "fe",
        "sio2",
        "al2o3",
        "p",
        "loi",
        "GIBBSITE",
        "KAOLINITE",
        "GOETHITE",
        "HEMATITE",
        "CHERT",
        "PYRITE",
        "combined_biff_biff_s_cluster_k5",
        "pca1",
        "pca2",
    ]
    export_cols = [col for col in export_cols if col in df.columns]
    df[export_cols].to_csv(interval_path, index=False)

    summarize(df, ["recommended_image_class"]).to_csv(image_class_summary_path, index=False)
    summarize(df, ["weathering_axis", "grade_axis"]).to_csv(axis_summary_path, index=False)
    summarize(df, ["source_classification", "weathering_axis", "grade_axis"]).to_csv(parent_axis_path, index=False)
    if "combined_biff_biff_s_cluster_k5" in df.columns:
        summarize(df, ["combined_biff_biff_s_cluster_k5", "weathering_axis", "grade_axis"]).to_csv(cluster_axis_path, index=False)
    df[df["review_flag"].ne("")][export_cols].to_csv(outlier_path, index=False)

    report = {
        "intervals": int(len(df)),
        "source": str(args.intervals),
        "outputs": {
            "interval_labels": str(interval_path),
            "recommended_image_class_summary": str(image_class_summary_path),
            "axis_summary": str(axis_summary_path),
            "axis_by_parent_class": str(parent_axis_path),
            "axis_by_cluster": str(cluster_axis_path),
            "review_flags": str(outlier_path),
        },
        "recommended_image_class_counts": df["recommended_image_class"].value_counts().to_dict(),
        "weathering_axis_counts": df["weathering_axis"].value_counts().to_dict(),
        "grade_axis_counts": df["grade_axis"].value_counts().to_dict(),
        "review_flagged_intervals": int(df["review_flag"].ne("").sum()),
    }
    report_path = args.output_dir / "biff_axis_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print("\nRecommended image class summary:")
    print(pd.read_csv(image_class_summary_path).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
