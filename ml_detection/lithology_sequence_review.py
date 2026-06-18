"""Build two-level lithology labels and review diagnostics from cluster outputs.

This consumes the outputs from lithology_clustering.py. It does not touch source
images. The generated labels are deliberately reviewable pseudo-labels, not a
claim that the geological class set is final.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


CLUSTER_DIR = Path("ml_detection") / "dataset_audit" / "lithology_clustering"
OUTPUT_DIR = Path("ml_detection") / "dataset_audit" / "lithology_sequence_review"
KEY_COLS = ["PROJECTCODE", "HOLEID", "SAMPFROM", "SAMPTO"]

WEATHERING_MAP = {
    "C04": "laterite_cap",
    "C03": "upper_weathered",
    "C07": "transitional_or_fresh",
    "C00": "fresh_deep",
    "C01": "fresh_deep",
    "C02": "fresh_deep",
    "C05": "fresh_deep",
    "C06": "fresh_deep",
}

FAMILY_MAP = {
    "C04.C01": "iron_formation_bif_bid",
    "C04.C03": "potassic_clastic_mafic",
    "C04.C07": "pyritic_aluminous_mafic",
    "C04.C04": "ferruginous_bid_pyritic",
    "C04.C02": "calc_magnesian_mafic",
    "C04.C06": "sodic_potassic_clastic",
    "C04.C05": "manganiferous_iron_formation",
    "C04.C00": "sodic_clastic",
    "C01": "gibbsitic_aluminous",
    "C07": "gibbsitic_laterite",
    "C03": "carbonate_magnesian_mixed",
    "C00": "carbonate_apatite_mafic",
    "C02": "calcic_sodic_mafic",
    "C06": "calcic_sodic_mafic",
    "C05": "extreme_calcic_mafic",
    "C09": "sodic_feldspathic_clastic",
    "C08": "pyritic_sulphidic",
}

DEPTH_FEATURES = {
    "INTERVAL_MID_M",
    "INTERVAL_THICKNESS_M",
    "DEPTH_FRACTION_OF_HOLE",
    "DEPTH_WITHIN_HOLE_Z",
    "DEPTH_WITHIN_PROJECT_Z",
    "WEATHERING_ALTERATION_INDEX",
}

CONTEXT_COLS = [
    "BESTSTRAT",
    "STRATUNIT",
    "STRATSUM",
    "PROFILEZONATION",
    "FE_PCT_BEST",
    "SIO2_PCT_BEST",
    "AL2O3_PCT_BEST",
    "LOI_PCT_BEST",
    "CAO_PCT_BEST",
    "MGO_PCT_BEST",
    "K2O_PCT_BEST",
    "NA2O_PCT_BEST",
    "S_PCT_BEST",
    "MN_PCT_BEST",
    "GIBBSITE",
    "GOETHITE",
    "HEMATITE",
    "KAOLINITE",
    "PYRITE",
    "GAMMA_CPS_HOLE_Z_mean",
    "INTERVAL_MID_M",
    "DEPTH_FRACTION_OF_HOLE",
    "WEATHERING_ALTERATION_INDEX",
]


def top_counts(series: pd.Series, limit: int = 5) -> dict[str, int]:
    counts = series.dropna().astype(str).value_counts().head(limit)
    return {str(k): int(v) for k, v in counts.items()}


def load_inputs(cluster_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    baseline = pd.read_parquet(cluster_dir / "rich_assay_norm_geophys_cluster_assignments.parquet")
    depth = pd.read_parquet(cluster_dir / "rich_assay_norm_geophys_depth_weathering_cluster_assignments.parquet")
    report = json.loads((cluster_dir / "lithology_clustering_report.json").read_text(encoding="utf-8"))
    return baseline, depth, report


def feature_cols_from_report(report: dict, dataset_name: str) -> list[str]:
    for result in report["results"]:
        if result["dataset_name"] == dataset_name:
            return result["feature_cols"]
    raise ValueError(f"Dataset not found in report: {dataset_name}")


def make_two_level_labels(baseline: pd.DataFrame, depth: pd.DataFrame) -> pd.DataFrame:
    depth_cols = KEY_COLS + [
        "rich_assay_norm_geophys_depth_weathering_cluster",
        "rich_assay_norm_geophys_depth_weathering_stage2_cluster",
        "INTERVAL_MID_M",
        "INTERVAL_THICKNESS_M",
        "DEPTH_FRACTION_OF_HOLE",
        "DEPTH_WITHIN_HOLE_Z",
        "DEPTH_WITHIN_PROJECT_Z",
        "WEATHERING_ALTERATION_INDEX",
    ]
    merged = baseline.merge(
        depth[[c for c in depth_cols if c in depth.columns]],
        on=KEY_COLS,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_DEPTH"),
    )
    family_key = merged["rich_assay_norm_geophys_stage2_cluster"].fillna(
        merged["rich_assay_norm_geophys_cluster"]
    )
    weather_key = merged["rich_assay_norm_geophys_depth_weathering_cluster"]
    merged["lithology_family_key"] = family_key
    merged["weathering_key"] = weather_key
    merged["lithology_family"] = family_key.map(FAMILY_MAP).fillna("review_" + family_key.astype(str))
    merged["weathering_state"] = weather_key.map(WEATHERING_MAP).fillna("review_" + weather_key.astype(str))
    merged["two_level_label"] = merged["weathering_state"] + "__" + merged["lithology_family"]
    merged["training_label_candidate"] = np.where(
        merged["weathering_state"].isin(["laterite_cap", "upper_weathered"]),
        merged["two_level_label"],
        merged["lithology_family"],
    )
    return merged


def numeric_feature_cols(df: pd.DataFrame, candidates: list[str], min_coverage: float = 0.70) -> list[str]:
    cols: list[str] = []
    n = max(1, len(df))
    for col in candidates:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() / n >= min_coverage and values.nunique(dropna=True) > 1:
            cols.append(col)
    return cols


def summarise_labels(df: pd.DataFrame, label_col: str) -> list[dict]:
    rows: list[dict] = []
    for label, group in df.groupby(label_col, sort=True):
        row = {
            "label": str(label),
            "rows": int(len(group)),
            "projects": int(group["PROJECTCODE"].nunique()),
            "holes": int(group["HOLEID"].nunique()),
            "median_from": float(group["SAMPFROM"].median()),
            "median_depth_fraction": float(group["DEPTH_FRACTION_OF_HOLE"].median()),
            "median_weathering_index": float(group["WEATHERING_ALTERATION_INDEX"].median()),
            "top_weathering_states": top_counts(group["weathering_state"]),
            "top_lithology_families": top_counts(group["lithology_family"]),
            "top_BESTSTRAT": top_counts(group["BESTSTRAT"]) if "BESTSTRAT" in group else {},
            "top_PROFILEZONATION": top_counts(group["PROFILEZONATION"]) if "PROFILEZONATION" in group else {},
        }
        rows.append(row)
    return rows


def add_gmm_soft_membership(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    random_state: int,
) -> tuple[pd.DataFrame, dict]:
    labels = df[label_col].astype(str)
    label_names = sorted(labels.unique())
    pipeline = make_pipeline(SimpleImputer(strategy="median"), RobustScaler())
    matrix = pipeline.fit_transform(df[feature_cols].apply(pd.to_numeric, errors="coerce"))
    gmm = GaussianMixture(
        n_components=len(label_names),
        covariance_type="diag",
        reg_covar=1.0e-5,
        max_iter=250,
        n_init=3,
        random_state=random_state,
    )
    components = gmm.fit_predict(matrix)
    probabilities = gmm.predict_proba(matrix)

    component_to_label: dict[int, str] = {}
    for component in range(len(label_names)):
        mask = components == component
        if mask.any():
            component_to_label[component] = labels[mask].value_counts().idxmax()
        else:
            component_to_label[component] = f"component_{component:02d}"

    sorted_probs = np.sort(probabilities, axis=1)
    df = df.copy()
    df["gmm_component"] = components
    df["gmm_label"] = [component_to_label[int(c)] for c in components]
    df["gmm_confidence"] = probabilities.max(axis=1)
    df["gmm_margin"] = sorted_probs[:, -1] - sorted_probs[:, -2] if probabilities.shape[1] > 1 else 1.0
    df["gmm_agrees_with_training_label"] = df["gmm_label"] == labels
    df["gmm_ambiguous"] = (df["gmm_confidence"] < 0.60) | (df["gmm_margin"] < 0.20)

    summary = {
        "n_components": len(label_names),
        "converged": bool(gmm.converged_),
        "iterations": int(gmm.n_iter_),
        "mean_confidence": float(df["gmm_confidence"].mean()),
        "median_confidence": float(df["gmm_confidence"].median()),
        "ambiguous_fraction": float(df["gmm_ambiguous"].mean()),
        "agreement_fraction": float(df["gmm_agrees_with_training_label"].mean()),
        "component_to_label": {str(k): v for k, v in component_to_label.items()},
    }
    return df, summary


def nonlinear_outlier_review(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    sample_per_label: int,
    random_state: int,
) -> tuple[pd.DataFrame, dict]:
    parts = []
    for _, group in df.groupby(label_col, sort=True):
        parts.append(group.sample(n=min(sample_per_label, len(group)), random_state=random_state))
    sample = pd.concat(parts, ignore_index=True)

    pipeline = make_pipeline(SimpleImputer(strategy="median"), RobustScaler())
    matrix = pipeline.fit_transform(sample[feature_cols].apply(pd.to_numeric, errors="coerce"))
    n_components = min(8, matrix.shape[1], max(2, matrix.shape[0] - 1))
    pca = PCA(n_components=n_components, random_state=random_state)
    embedding = pca.fit_transform(matrix)

    n_labels = sample[label_col].nunique()
    agg = AgglomerativeClustering(n_clusters=n_labels, linkage="ward")
    agg_labels = agg.fit_predict(embedding)
    dbscan = DBSCAN(eps=2.5, min_samples=10)
    dbscan_labels = dbscan.fit_predict(embedding)
    lof = LocalOutlierFactor(n_neighbors=35, contamination=0.03)
    outlier_labels = lof.fit_predict(embedding)

    sample = sample.copy()
    for i in range(n_components):
        sample[f"pca_{i + 1}"] = embedding[:, i]
    sample["agglomerative_cluster"] = agg_labels
    sample["dbscan_cluster"] = dbscan_labels
    sample["local_outlier"] = outlier_labels == -1

    label_codes = pd.Categorical(sample[label_col]).codes
    summary = {
        "sample_rows": int(len(sample)),
        "sample_per_label": int(sample_per_label),
        "pca_explained_variance": [float(v) for v in pca.explained_variance_ratio_],
        "agglomerative_adjusted_rand_vs_labels": float(adjusted_rand_score(label_codes, agg_labels)),
        "agglomerative_silhouette": float(silhouette_score(embedding, agg_labels))
        if len(np.unique(agg_labels)) > 1
        else None,
        "dbscan_clusters": int(len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)),
        "dbscan_noise_fraction": float((dbscan_labels == -1).mean()),
        "local_outlier_fraction": float(sample["local_outlier"].mean()),
    }
    return sample, summary


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if pd.isna(obj):
        return None
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-dir", type=Path, default=CLUSTER_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sample-per-label", type=int, default=350)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline, depth, report = load_inputs(args.cluster_dir)
    labels = make_two_level_labels(baseline, depth)
    rich_features = feature_cols_from_report(report, "rich_assay_norm_geophys")
    depth_features = list(DEPTH_FEATURES)
    feature_cols = numeric_feature_cols(labels, rich_features + depth_features)

    labels, gmm_summary = add_gmm_soft_membership(
        labels,
        feature_cols,
        "training_label_candidate",
        args.random_state,
    )
    nonlinear_sample, nonlinear_summary = nonlinear_outlier_review(
        labels,
        feature_cols,
        "training_label_candidate",
        args.sample_per_label,
        args.random_state,
    )

    labels_path = args.output_dir / "two_level_lithology_labels.parquet"
    labels_csv_path = args.output_dir / "two_level_lithology_labels_preview.csv"
    label_summary_path = args.output_dir / "two_level_label_summary.csv"
    nonlinear_path = args.output_dir / "nonlinear_outlier_review_sample.csv"
    report_path = args.output_dir / "sequence_review_report.json"

    labels.to_parquet(labels_path, index=False)
    preview_cols = KEY_COLS + [
        "training_label_candidate",
        "two_level_label",
        "weathering_state",
        "lithology_family",
        "gmm_confidence",
        "gmm_margin",
        "gmm_ambiguous",
    ] + [c for c in CONTEXT_COLS if c in labels.columns]
    labels[preview_cols].head(5000).to_csv(labels_csv_path, index=False)
    pd.DataFrame(summarise_labels(labels, "training_label_candidate")).to_csv(
        label_summary_path, index=False
    )
    nonlinear_sample[
        KEY_COLS
        + [
            "training_label_candidate",
            "two_level_label",
            "weathering_state",
            "lithology_family",
            "agglomerative_cluster",
            "dbscan_cluster",
            "local_outlier",
            "pca_1",
            "pca_2",
        ]
        + [c for c in ["BESTSTRAT", "PROFILEZONATION", "INTERVAL_MID_M"] if c in nonlinear_sample.columns]
    ].to_csv(nonlinear_path, index=False)

    final_report = {
        "rows": int(len(labels)),
        "feature_cols": feature_cols,
        "training_label_count": int(labels["training_label_candidate"].nunique()),
        "training_label_counts": {
            str(k): int(v) for k, v in labels["training_label_candidate"].value_counts().sort_index().items()
        },
        "two_level_label_count": int(labels["two_level_label"].nunique()),
        "gmm": gmm_summary,
        "nonlinear_outlier_review": nonlinear_summary,
        "outputs": {
            "labels": str(labels_path),
            "labels_preview": str(labels_csv_path),
            "label_summary": str(label_summary_path),
            "nonlinear_sample": str(nonlinear_path),
        },
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(final_report), f, indent=2)

    print(f"Wrote sequence review report to {report_path}")
    print(f"Training labels: {final_report['training_label_count']}")
    print(f"GMM ambiguous fraction: {gmm_summary['ambiguous_fraction']:.3f}")
    print(f"Agglomerative ARI vs labels: {nonlinear_summary['agglomerative_adjusted_rand_vs_labels']:.3f}")


if __name__ == "__main__":
    main()
