"""Cluster lithology-style interval data from assays, mineralogy, and geophysics.

This script is intentionally data-first. It does not copy images or change the
wet/dry/empty classifier dataset. The output is a set of interval-level cluster
assignments and review samples that can later drive a copy-only image manifest.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


DEFAULT_CACHE_DIR = Path.home() / "AppData" / "Roaming" / "GeoVue" / "sf_cache"
DEFAULT_OUTPUT_DIR = Path("ml_detection") / "dataset_audit" / "lithology_clustering"

GEOPHYS_COLS = [
    "GAMMA_CPS",
    "MAGSUSCEP_SIE3",
    "MAGSUSCEP_SIE5",
    "MAGSUSCPS",
    "MAGSUSCEP_SI5_D",
]

CLR_ASSAY_PARTS = [
    "FE_PCT_BEST",
    "SIO2_PCT_BEST",
    "AL2O3_PCT_BEST",
    "TIO2_PCT_BEST",
    "CAO_PCT_BEST",
    "MGO_PCT_BEST",
    "K2O_PCT_BEST",
    "NA2O_PCT_BEST",
    "P_PCT_BEST",
    "MN_PCT_BEST",
    "LOI_PCT_BEST",
]

CLR_RATIO_PAIRS = [
    ("FE_PCT_BEST", "SIO2_PCT_BEST", "FE_SIO2"),
    ("SIO2_PCT_BEST", "AL2O3_PCT_BEST", "SIO2_AL2O3"),
    ("AL2O3_PCT_BEST", "TIO2_PCT_BEST", "AL2O3_TIO2"),
    ("CAO_PCT_BEST", "MGO_PCT_BEST", "CAO_MGO"),
    ("P_PCT_BEST", "FE_PCT_BEST", "P_FE"),
    ("FE_PCT_BEST", "MN_PCT_BEST", "FE_MN"),
]

CONTEXT_LABEL_COLS = [
    "STRATSUM",
    "PROFILEZONATION",
    "STRATUNIT",
    "BESTSTRAT",
    "SAMPLETYPE",
    "PROSPECT",
]

KEY_COLS = ["PROJECTCODE", "HOLEID", "SAMPFROM", "SAMPTO"]


@dataclass(frozen=True)
class ClusteringResult:
    dataset_name: str
    rows: int
    feature_cols: list[str]
    selected_k: int
    k_scores: list[dict]
    cluster_col: str
    stage2: dict | None


def _read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path, columns=columns)


def _coerce_interval_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["PROJECTCODE"] = df["PROJECTCODE"].astype(str).str.strip()
    df["HOLEID"] = df["HOLEID"].astype(str).str.strip()
    for col in ["SAMPFROM", "SAMPTO"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(3)
    return df.dropna(subset=KEY_COLS)


def _first_non_null(values: pd.Series):
    non_null = values.dropna()
    return non_null.iloc[0] if not non_null.empty else np.nan


def aggregate_assays(assay: pd.DataFrame) -> pd.DataFrame:
    assay = _coerce_interval_keys(assay)
    assay_pct_cols = [c for c in assay.columns if c.upper().endswith("_PCT_BEST")]
    numeric_cols = assay_pct_cols + [
        c
        for c in [
            "MIN_80_PCT",
            "MIN_70_PCT",
            "MIN_60_PCT",
            "MIN_50_PCT",
            "CHIP_PCT",
            "HOLEDEPTH",
        ]
        if c in assay.columns
    ]
    for col in numeric_cols:
        assay[col] = pd.to_numeric(assay[col], errors="coerce")

    context_cols = [c for c in CONTEXT_LABEL_COLS if c in assay.columns]
    passthrough_cols = [c for c in ["SAMPLEID", "LITHCOLOUR", "LITHCOMMENTS"] if c in assay.columns]
    agg = {col: "mean" for col in numeric_cols}
    agg.update({col: _first_non_null for col in context_cols + passthrough_cols})
    return assay.groupby(KEY_COLS, as_index=False, sort=False).agg(agg)


def robust_z(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    median = values.median()
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    scale = (q3 - q1) / 1.349
    if not np.isfinite(scale) or scale <= 1.0e-9:
        scale = values.std()
    if not np.isfinite(scale) or scale <= 1.0e-9:
        scale = 1.0
    return (values - median) / scale


def grouped_robust_z(df: pd.DataFrame, value_col: str, group_cols: list[str]) -> pd.Series:
    return df.groupby(group_cols, sort=False)[value_col].transform(robust_z)


def derive_depth_weathering_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    df["INTERVAL_MID_M"] = (pd.to_numeric(df["SAMPFROM"], errors="coerce") + pd.to_numeric(df["SAMPTO"], errors="coerce")) / 2
    df["INTERVAL_THICKNESS_M"] = pd.to_numeric(df["SAMPTO"], errors="coerce") - pd.to_numeric(
        df["SAMPFROM"], errors="coerce"
    )
    df["DEPTH_WITHIN_HOLE_Z"] = grouped_robust_z(df, "INTERVAL_MID_M", ["PROJECTCODE", "HOLEID"])
    df["DEPTH_WITHIN_PROJECT_Z"] = grouped_robust_z(df, "INTERVAL_MID_M", ["PROJECTCODE"])
    if "HOLEDEPTH" in df.columns:
        hole_depth = pd.to_numeric(df["HOLEDEPTH"], errors="coerce")
        df["DEPTH_FRACTION_OF_HOLE"] = df["INTERVAL_MID_M"] / hole_depth.where(hole_depth > 0)
    else:
        df["DEPTH_FRACTION_OF_HOLE"] = np.nan

    weathering_terms: list[pd.Series] = []
    for col, weight in [
        ("LOI_PCT_BEST", 1.0),
        ("AL2O3_PCT_BEST", 1.0),
        ("GIBBSITE", 1.0),
        ("GOETHITE", 0.5),
        ("SIO2_PCT_BEST", -0.5),
    ]:
        if col in df.columns:
            weathering_terms.append(robust_z(df[col]) * weight)
    if weathering_terms:
        df["WEATHERING_ALTERATION_INDEX"] = pd.concat(weathering_terms, axis=1).mean(axis=1, skipna=True)
    else:
        df["WEATHERING_ALTERATION_INDEX"] = np.nan

    feature_cols = [
        "INTERVAL_MID_M",
        "INTERVAL_THICKNESS_M",
        "DEPTH_FRACTION_OF_HOLE",
        "DEPTH_WITHIN_HOLE_Z",
        "DEPTH_WITHIN_PROJECT_Z",
        "WEATHERING_ALTERATION_INDEX",
    ]
    return df, feature_cols


def aggregate_normative(normative: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    normative = _coerce_interval_keys(normative)
    mineral_cols = [
        c
        for c in normative.columns
        if c
        not in {
            "PROJECTCODE",
            "HOLEID",
            "SAMPFROM",
            "SAMPTO",
            "SAMPLEID",
            "SAMPLETYPE",
        }
    ]
    for col in mineral_cols:
        normative[col] = pd.to_numeric(normative[col], errors="coerce")
    agg = {col: "mean" for col in mineral_cols}
    if "SAMPLEID" in normative.columns:
        normative = normative.rename(columns={"SAMPLEID": "NORM_SAMPLEID"})
        agg["NORM_SAMPLEID"] = _first_non_null
    if "SAMPLETYPE" in normative.columns:
        normative = normative.rename(columns={"SAMPLETYPE": "NORM_SAMPLETYPE"})
        agg["NORM_SAMPLETYPE"] = _first_non_null
    return normative.groupby(KEY_COLS, as_index=False, sort=False).agg(agg), mineral_cols


def add_oxide_clr_features(
    df: pd.DataFrame,
    assay_feature_cols: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Add CLR assay features and diagnostic log-ratios from major oxide parts."""
    available_parts = [col for col in CLR_ASSAY_PARTS if col in assay_feature_cols and col in df.columns]
    if len(available_parts) < 3:
        return df, [], []

    df = df.copy()
    parts = df[available_parts].apply(pd.to_numeric, errors="coerce")
    parts = parts.mask(parts <= 0)
    replacements = {}
    for col in available_parts:
        positive = parts[col].dropna()
        replacements[col] = float(positive.min() / 2.0) if not positive.empty else 1.0e-6
    parts = parts.fillna(replacements)
    parts = parts.clip(lower=1.0e-9)
    row_sums = parts.sum(axis=1).replace(0, np.nan)
    closed = parts.div(row_sums, axis=0).fillna(parts)

    logs = np.log(closed)
    clr = logs.sub(logs.mean(axis=1), axis=0)
    clr_cols: list[str] = []
    part_to_clr: dict[str, str] = {}
    for col in available_parts:
        clr_col = f"CLR_{col.replace('_PCT_BEST', '')}"
        df[clr_col] = clr[col]
        clr_cols.append(clr_col)
        part_to_clr[col] = clr_col

    ratio_cols: list[str] = []
    for numerator, denominator, name in CLR_RATIO_PAIRS:
        if numerator in part_to_clr and denominator in part_to_clr:
            ratio_col = f"LOGRATIO_{name}"
            df[ratio_col] = df[part_to_clr[numerator]] - df[part_to_clr[denominator]]
            ratio_cols.append(ratio_col)

    return df, clr_cols, ratio_cols


def prepare_geophysics(geophys: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    required = ["PROJECTCODE", "HOLEID", "DEPTH"]
    geophys = geophys.dropna(subset=required).copy()
    geophys["PROJECTCODE"] = geophys["PROJECTCODE"].astype(str).str.strip()
    geophys["HOLEID"] = geophys["HOLEID"].astype(str).str.strip()
    geophys["DEPTH"] = pd.to_numeric(geophys["DEPTH"], errors="coerce").round(3)
    geophys = geophys.dropna(subset=["DEPTH"])

    available_cols = [c for c in GEOPHYS_COLS if c in geophys.columns]
    for col in available_cols:
        geophys[col] = pd.to_numeric(geophys[col], errors="coerce")

    geophys = (
        geophys[["PROJECTCODE", "HOLEID", "DEPTH"] + available_cols]
        .groupby(["PROJECTCODE", "HOLEID", "DEPTH"], as_index=False, sort=False)
        .median(numeric_only=True)
    )

    hole_key = geophys["PROJECTCODE"] + "|" + geophys["HOLEID"]
    geophys["_HOLE_KEY"] = hole_key
    z_cols: list[str] = []
    for col in available_cols:
        grouped = geophys.groupby("_HOLE_KEY")[col]
        median = grouped.transform("median")
        q1 = grouped.transform(lambda s: s.quantile(0.25))
        q3 = grouped.transform(lambda s: s.quantile(0.75))
        std = grouped.transform("std")
        scale = (q3 - q1) / 1.349
        scale = scale.where(scale > 1.0e-9, std)
        scale = scale.where(scale > 1.0e-9, 1.0)
        z_col = f"{col}_HOLE_Z"
        geophys[z_col] = (geophys[col] - median) / scale
        z_cols.append(z_col)

    return geophys.drop(columns=["_HOLE_KEY"]), z_cols


def summarize_geophysics_for_intervals(
    intervals: pd.DataFrame,
    geophys: pd.DataFrame,
    value_cols: list[str],
) -> pd.DataFrame:
    summary_index = intervals.index
    out = pd.DataFrame(index=summary_index)
    out["GEOPHYS_POINT_COUNT"] = 0

    stats = ["mean", "min", "max", "range", "std", "first", "last", "slope"]
    for col in value_cols:
        for stat in stats:
            out[f"{col}_{stat}"] = np.nan

    geophys_groups = {
        key: group.sort_values("DEPTH")
        for key, group in geophys.groupby(["PROJECTCODE", "HOLEID"], sort=False)
    }

    interval_groups = intervals.groupby(["PROJECTCODE", "HOLEID"], sort=False)
    for key, idx in interval_groups.groups.items():
        if key not in geophys_groups:
            continue
        g = geophys_groups[key]
        depths = g["DEPTH"].to_numpy(dtype=float)
        if depths.size == 0:
            continue

        interval_slice = intervals.loc[idx, ["SAMPFROM", "SAMPTO"]]
        starts = np.searchsorted(depths, interval_slice["SAMPFROM"].to_numpy(dtype=float), side="left")
        ends = np.searchsorted(depths, interval_slice["SAMPTO"].to_numpy(dtype=float), side="left")
        counts = ends - starts
        out.loc[idx, "GEOPHYS_POINT_COUNT"] = counts

        for col in value_cols:
            values = g[col].to_numpy(dtype=float)
            for row_index, start, end in zip(idx, starts, ends):
                if end <= start:
                    continue
                chunk = values[start:end]
                chunk = chunk[np.isfinite(chunk)]
                if chunk.size == 0:
                    continue
                out.at[row_index, f"{col}_mean"] = float(np.nanmean(chunk))
                out.at[row_index, f"{col}_min"] = float(np.nanmin(chunk))
                out.at[row_index, f"{col}_max"] = float(np.nanmax(chunk))
                out.at[row_index, f"{col}_range"] = float(np.nanmax(chunk) - np.nanmin(chunk))
                out.at[row_index, f"{col}_std"] = float(np.nanstd(chunk))
                out.at[row_index, f"{col}_first"] = float(chunk[0])
                out.at[row_index, f"{col}_last"] = float(chunk[-1])
                depth_span = depths[end - 1] - depths[start] if end - start > 1 else 0.0
                interval_span = intervals.at[row_index, "SAMPTO"] - intervals.at[row_index, "SAMPFROM"]
                denominator = depth_span if depth_span > 0 else interval_span
                if denominator and denominator > 0:
                    out.at[row_index, f"{col}_slope"] = float((chunk[-1] - chunk[0]) / denominator)

    return out.reset_index(drop=True)


def top_counts(series: pd.Series, limit: int = 6) -> dict[str, int]:
    counts = series.dropna().astype(str).value_counts().head(limit)
    return {str(index): int(value) for index, value in counts.items()}


def finite_feature_cols(
    df: pd.DataFrame,
    columns: Iterable[str],
    min_non_null: int = 50,
    min_coverage: float = 0.70,
) -> list[str]:
    selected: list[str] = []
    row_count = max(1, len(df))
    for col in columns:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        non_null = int(values.notna().sum())
        coverage = non_null / row_count
        if (
            non_null >= min_non_null
            and coverage >= min_coverage
            and values.nunique(dropna=True) > 1
        ):
            selected.append(col)
    return selected


def feature_coverage(df: pd.DataFrame, columns: Iterable[str]) -> list[dict]:
    row_count = max(1, len(df))
    records: list[dict] = []
    for col in columns:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        records.append(
            {
                "feature": col,
                "non_null": int(values.notna().sum()),
                "coverage": float(values.notna().sum() / row_count),
                "unique_values": int(values.nunique(dropna=True)),
            }
        )
    return records


def evaluate_kmeans(
    matrix: np.ndarray,
    k_values: Iterable[int],
    random_state: int,
    sample_size: int,
) -> list[dict]:
    scores: list[dict] = []
    n_rows = matrix.shape[0]
    if n_rows < 3:
        return scores
    rng = np.random.default_rng(random_state)
    sample_n = min(sample_size, n_rows)
    sample_idx = rng.choice(n_rows, size=sample_n, replace=False)
    sample_matrix = matrix[sample_idx]

    for k in k_values:
        if k >= n_rows:
            continue
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            batch_size=4096,
            n_init=30,
            reassignment_ratio=0.01,
        )
        labels = model.fit_predict(matrix)
        cluster_sizes = np.bincount(labels, minlength=k)
        sample_labels = labels[sample_idx]
        if len(np.unique(sample_labels)) < 2:
            silhouette = np.nan
            davies_bouldin = np.nan
            calinski_harabasz = np.nan
        else:
            silhouette = float(silhouette_score(sample_matrix, sample_labels))
            davies_bouldin = float(davies_bouldin_score(sample_matrix, sample_labels))
            calinski_harabasz = float(calinski_harabasz_score(sample_matrix, sample_labels))
        scores.append(
            {
                "k": int(k),
                "silhouette_sample": silhouette,
                "davies_bouldin_sample": davies_bouldin,
                "calinski_harabasz_sample": calinski_harabasz,
                "min_cluster_size": int(cluster_sizes.min()),
                "max_cluster_size": int(cluster_sizes.max()),
                "max_cluster_fraction": float(cluster_sizes.max() / n_rows),
            }
        )
    return scores


def choose_k(scores: list[dict], n_rows: int, min_cluster_size: int) -> int:
    viable = [
        score
        for score in scores
        if score["min_cluster_size"] >= min_cluster_size
        and score["max_cluster_size"] <= max(min_cluster_size, int(0.70 * n_rows))
        and np.isfinite(score["silhouette_sample"])
    ]
    candidates = viable or [s for s in scores if np.isfinite(s["silhouette_sample"])] or scores
    return int(max(candidates, key=lambda s: s["silhouette_sample"])["k"])


def fit_cluster_column(
    df: pd.DataFrame,
    feature_cols: list[str],
    cluster_col: str,
    k: int,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray, object]:
    feature_frame = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    pipeline = make_pipeline(SimpleImputer(strategy="median"), RobustScaler())
    matrix = pipeline.fit_transform(feature_frame)
    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=random_state,
        batch_size=4096,
        n_init=30,
        reassignment_ratio=0.01,
    )
    labels = model.fit_predict(matrix)
    clustered = df.copy()
    clustered[cluster_col] = [f"C{label:02d}" for label in labels]
    return clustered, matrix, pipeline


def summarize_clusters(
    df: pd.DataFrame,
    cluster_col: str,
    feature_cols: list[str],
) -> list[dict]:
    global_medians = df[feature_cols].apply(pd.to_numeric, errors="coerce").median()
    global_iqr = (
        df[feature_cols].apply(pd.to_numeric, errors="coerce").quantile(0.75)
        - df[feature_cols].apply(pd.to_numeric, errors="coerce").quantile(0.25)
    ).replace(0, np.nan)
    rows: list[dict] = []
    for cluster, group in df.groupby(cluster_col, sort=True):
        medians = group[feature_cols].apply(pd.to_numeric, errors="coerce").median()
        deltas = ((medians - global_medians) / global_iqr).replace([np.inf, -np.inf], np.nan)
        strongest = (
            deltas.dropna()
            .sort_values(key=lambda s: s.abs(), ascending=False)
            .head(10)
            .round(3)
            .to_dict()
        )
        row = {
            "cluster": str(cluster),
            "rows": int(len(group)),
            "projects": int(group["PROJECTCODE"].nunique()),
            "holes": int(group["HOLEID"].nunique()),
            "median_from": float(group["SAMPFROM"].median()),
            "median_to": float(group["SAMPTO"].median()),
            "top_feature_deltas": {str(k): float(v) for k, v in strongest.items()},
        }
        for label_col in CONTEXT_LABEL_COLS:
            if label_col in group.columns:
                row[f"top_{label_col}"] = top_counts(group[label_col])
        rows.append(row)
    return rows


def cluster_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    feature_cols: list[str],
    output_dir: Path,
    random_state: int,
    sample_size: int,
    min_cluster_size: int,
    min_feature_coverage: float,
    k_values: range,
    stage2: bool,
) -> tuple[pd.DataFrame, ClusteringResult, list[dict]]:
    df = df.reset_index(drop=True).copy()
    feature_cols = finite_feature_cols(df, feature_cols, min_coverage=min_feature_coverage)
    feature_frame = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    pipeline = make_pipeline(SimpleImputer(strategy="median"), RobustScaler())
    matrix = pipeline.fit_transform(feature_frame)
    scores = evaluate_kmeans(matrix, k_values, random_state, sample_size)
    selected_k = choose_k(scores, len(df), min_cluster_size)
    cluster_col = f"{dataset_name}_cluster"
    clustered, matrix, _ = fit_cluster_column(df, feature_cols, cluster_col, selected_k, random_state)

    stage2_info = None
    if stage2:
        largest_cluster = clustered[cluster_col].value_counts().idxmax()
        largest_count = int(clustered[cluster_col].value_counts().max())
        if largest_count >= max(5000, min_cluster_size * 5):
            subset_mask = clustered[cluster_col] == largest_cluster
            subset = clustered.loc[subset_mask].copy()
            subset_features = subset[feature_cols].apply(pd.to_numeric, errors="coerce")
            subset_matrix = pipeline.fit_transform(subset_features)
            stage2_scores = evaluate_kmeans(
                subset_matrix,
                range(3, min(11, max(4, len(subset) // min_cluster_size)) + 1),
                random_state + 1,
                sample_size,
            )
            if stage2_scores:
                stage2_k = choose_k(stage2_scores, len(subset), min_cluster_size)
                stage2_col = f"{dataset_name}_stage2_cluster"
                subset_clustered, _, _ = fit_cluster_column(
                    subset, feature_cols, stage2_col, stage2_k, random_state + 1
                )
                clustered[stage2_col] = clustered[cluster_col]
                clustered.loc[subset_mask, stage2_col] = [
                    f"{largest_cluster}.{label}" for label in subset_clustered[stage2_col]
                ]
                stage2_info = {
                    "parent_cluster": str(largest_cluster),
                    "parent_rows": largest_count,
                    "selected_k": int(stage2_k),
                    "scores": stage2_scores,
                    "cluster_col": stage2_col,
                    "summary": summarize_clusters(
                        clustered.loc[subset_mask].assign(
                            **{stage2_col: clustered.loc[subset_mask, stage2_col]}
                        ),
                        stage2_col,
                        feature_cols,
                    ),
                }

    cluster_summary = summarize_clusters(clustered, cluster_col, feature_cols)
    result = ClusteringResult(
        dataset_name=dataset_name,
        rows=len(clustered),
        feature_cols=feature_cols,
        selected_k=selected_k,
        k_scores=scores,
        cluster_col=cluster_col,
        stage2=stage2_info,
    )

    assignment_path = output_dir / f"{dataset_name}_cluster_assignments.parquet"
    clustered.to_parquet(assignment_path, index=False)
    pd.DataFrame(scores).to_csv(output_dir / f"{dataset_name}_k_scores.csv", index=False)
    pd.DataFrame(cluster_summary).to_csv(output_dir / f"{dataset_name}_cluster_summary.csv", index=False)
    review_samples = pd.concat(
        [
            group.sample(n=min(100, len(group)), random_state=random_state)
            for _, group in clustered.groupby(cluster_col, sort=True)
        ],
        ignore_index=True,
    )
    review_samples.to_csv(output_dir / f"{dataset_name}_review_samples_100_per_cluster.csv", index=False)
    return clustered, result, cluster_summary


def build_joined_intervals(
    geophys_path: Path,
    normative_path: Path,
    assay_path: Path,
    output_dir: Path,
    geophys_mode: str,
    composition_mode: str,
) -> tuple[pd.DataFrame, list[str], list[str], list[str], list[str], dict]:
    assay = _read_parquet(assay_path)
    normative = _read_parquet(normative_path)
    geophys = _read_parquet(geophys_path)

    assay_unique = aggregate_assays(assay)
    normative_unique, mineral_cols = aggregate_normative(normative)
    geophys_prepared, geophys_z_cols = prepare_geophysics(geophys)

    joined = assay_unique.merge(
        normative_unique,
        on=KEY_COLS,
        how="left",
        validate="one_to_one",
    )

    if geophys_mode == "gamma":
        geophys_raw_cols = [c for c in ["GAMMA_CPS"] if c in geophys_prepared.columns]
        clustering_z_cols = [c for c in geophys_z_cols if c == "GAMMA_CPS_HOLE_Z"]
    else:
        geophys_raw_cols = [c for c in GEOPHYS_COLS if c in geophys_prepared.columns]
        clustering_z_cols = geophys_z_cols

    geophys_value_cols = geophys_raw_cols + clustering_z_cols
    geophys_summary = summarize_geophysics_for_intervals(joined, geophys_prepared, geophys_value_cols)
    joined = pd.concat([joined.reset_index(drop=True), geophys_summary], axis=1)
    joined, depth_weathering_feature_cols = derive_depth_weathering_features(joined)
    joined["HAS_NORMATIVE"] = joined[mineral_cols].notna().any(axis=1) if mineral_cols else False
    joined["HAS_GEOPHYS"] = joined["GEOPHYS_POINT_COUNT"] > 0

    raw_assay_feature_cols = [c for c in assay_unique.columns if c.upper().endswith("_PCT_BEST")]
    joined, clr_assay_feature_cols, clr_ratio_cols = add_oxide_clr_features(
        joined,
        raw_assay_feature_cols,
    )
    if composition_mode == "oxide-clr":
        assay_feature_cols = clr_assay_feature_cols
    else:
        assay_feature_cols = raw_assay_feature_cols
    geophys_feature_cols = []
    for col in clustering_z_cols:
        for stat in ["mean", "range", "std", "slope"]:
            feature = f"{col}_{stat}"
            if feature in joined.columns:
                geophys_feature_cols.append(feature)

    coverage = {
        "assay_raw_rows": int(len(assay)),
        "assay_unique_intervals": int(len(assay_unique)),
        "assay_duplicate_rows_collapsed": int(len(assay) - len(assay_unique)),
        "normative_raw_rows": int(len(normative)),
        "normative_unique_intervals": int(len(normative_unique)),
        "joined_intervals": int(len(joined)),
        "joined_with_normative": int(joined["HAS_NORMATIVE"].sum()),
        "joined_with_geophys": int(joined["HAS_GEOPHYS"].sum()),
        "joined_with_normative_and_geophys": int((joined["HAS_NORMATIVE"] & joined["HAS_GEOPHYS"]).sum()),
        "geophys_raw_rows": int(len(geophys)),
        "geophys_unique_depth_rows": int(len(geophys_prepared)),
        "geophys_duplicate_depth_rows_collapsed": int(len(geophys) - len(geophys_prepared)),
        "composition_mode": composition_mode,
        "raw_assay_features": raw_assay_feature_cols,
        "assay_features": assay_feature_cols,
        "clr_assay_features": clr_assay_feature_cols,
        "diagnostic_logratio_features": clr_ratio_cols,
        "diagnostic_logratio_feature_coverage": feature_coverage(joined, clr_ratio_cols),
        "mineral_features": mineral_cols,
        "geophys_mode": geophys_mode,
        "geophys_features_for_clustering_candidates": geophys_feature_cols,
        "geophys_candidate_feature_coverage": feature_coverage(joined, geophys_feature_cols),
        "depth_weathering_features_for_sensitivity": depth_weathering_feature_cols,
        "depth_weathering_candidate_feature_coverage": feature_coverage(joined, depth_weathering_feature_cols),
    }

    joined.to_parquet(output_dir / "joined_lithology_intervals.parquet", index=False)
    return joined, assay_feature_cols, mineral_cols, geophys_feature_cols, depth_weathering_feature_cols, coverage


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
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
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Folder containing the GeoVue Snowflake parquet cache.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--min-cluster-size", type=int, default=250)
    parser.add_argument("--min-feature-coverage", type=float, default=0.70)
    parser.add_argument(
        "--geophys-mode",
        choices=["gamma", "gamma-plus-covered"],
        default="gamma",
        help="Use normalised gamma only by default; opt in to sparse channels separately.",
    )
    parser.add_argument(
        "--composition-mode",
        choices=["raw", "oxide-clr"],
        default="raw",
        help="Use raw assay percentages or CLR-transformed major oxide assay parts.",
    )
    parser.add_argument("--k-min", type=int, default=4)
    parser.add_argument("--k-max", type=int, default=14)
    parser.add_argument("--no-stage2", action="store_true")
    parser.add_argument(
        "--include-depth-variant",
        action="store_true",
        help="Also run depth/weathering-aware clustering variants for comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    geophys_path = args.cache_dir / "AA_EXPLORATION_AFR__SENS_GAB__GEOPHYSICSDETAILS.parquet"
    normative_path = args.cache_dir / "AA_EXPLORATION_AFR__SENS_GAB__NORMATIVE_MINERALOGY.parquet"
    assay_path = args.cache_dir / "AA_EXPLORATION_AFR__SENS_GAB__SUMMARY_LOGGING_ASSAYS.parquet"

    (
        joined,
        assay_features,
        mineral_features,
        geophys_features,
        depth_weathering_features,
        coverage,
    ) = build_joined_intervals(
        geophys_path,
        normative_path,
        assay_path,
        args.output_dir,
        args.geophys_mode,
        args.composition_mode,
    )

    results: list[ClusteringResult] = []
    summaries: dict[str, list[dict]] = {}
    k_values = range(args.k_min, args.k_max + 1)

    rich = joined[joined["HAS_NORMATIVE"] & joined["HAS_GEOPHYS"]].copy()
    rich_features = assay_features + mineral_features + geophys_features
    _, result, summary = cluster_dataset(
        rich,
        "rich_assay_norm_geophys",
        rich_features,
        args.output_dir,
        args.random_state,
        args.sample_size,
        args.min_cluster_size,
        args.min_feature_coverage,
        k_values,
        stage2=not args.no_stage2,
    )
    results.append(result)
    summaries[result.dataset_name] = summary

    if args.include_depth_variant:
        _, result, summary = cluster_dataset(
            rich,
            "rich_assay_norm_geophys_depth_weathering",
            rich_features + depth_weathering_features,
            args.output_dir,
            args.random_state,
            args.sample_size,
            args.min_cluster_size,
            args.min_feature_coverage,
            k_values,
            stage2=not args.no_stage2,
        )
        results.append(result)
        summaries[result.dataset_name] = summary

    broad = joined[joined["HAS_GEOPHYS"]].copy()
    broad_features = assay_features + geophys_features
    _, result, summary = cluster_dataset(
        broad,
        "broad_assay_geophys",
        broad_features,
        args.output_dir,
        args.random_state,
        args.sample_size,
        args.min_cluster_size,
        args.min_feature_coverage,
        k_values,
        stage2=not args.no_stage2,
    )
    results.append(result)
    summaries[result.dataset_name] = summary

    if args.include_depth_variant:
        _, result, summary = cluster_dataset(
            broad,
            "broad_assay_geophys_depth_weathering",
            broad_features + depth_weathering_features,
            args.output_dir,
            args.random_state,
            args.sample_size,
            args.min_cluster_size,
            args.min_feature_coverage,
            k_values,
            stage2=not args.no_stage2,
        )
        results.append(result)
        summaries[result.dataset_name] = summary

    report = {
        "inputs": {
            "geophys_path": str(geophys_path),
            "normative_path": str(normative_path),
            "assay_path": str(assay_path),
        },
        "coverage": coverage,
        "results": [
            {
                "dataset_name": result.dataset_name,
                "rows": result.rows,
                "feature_count": len(result.feature_cols),
                "feature_cols": result.feature_cols,
                "selected_k": result.selected_k,
                "cluster_col": result.cluster_col,
                "k_scores": result.k_scores,
                "stage2": result.stage2,
                "cluster_summary": summaries[result.dataset_name],
            }
            for result in results
        ],
    }

    with (args.output_dir / "lithology_clustering_report.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(report), f, indent=2)

    print(f"Wrote lithology clustering report to {args.output_dir}")
    for result in results:
        print(
            f"{result.dataset_name}: rows={result.rows:,}, "
            f"features={len(result.feature_cols)}, selected_k={result.selected_k}"
        )


if __name__ == "__main__":
    main()
