"""
Analyze manually tagged BIFf intervals and candidate sub-populations.

This script is intentionally read-only with respect to source datasets. It writes
review outputs under ml_detection/dataset_audit/biff_population_review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plots are optional
    plt = None


DEFAULT_CSV = Path(r"C:\Users\gsymonds\Downloads\classifications_export_20260617_151205.csv")
DEFAULT_NORMATIVE = Path(
    r"C:\Users\gsymonds\AppData\Roaming\GeoVue\sf_cache\AA_EXPLORATION_AFR__SENS_GAB__NORMATIVE_MINERALOGY.parquet"
)
DEFAULT_GEOPHYSICS = Path(
    r"C:\Users\gsymonds\AppData\Roaming\GeoVue\sf_cache\AA_EXPLORATION_AFR__SENS_GAB__GEOPHYSICSDETAILS.parquet"
)
DEFAULT_OUTPUT = Path("ml_detection/dataset_audit/biff_population_review")

ASSAY_EXPORT_COLUMNS = {
    "fe": "fe_pct (exassay)",
    "sio2": "sio2_pct (exassay)",
    "al2o3": "al2o3_pct (exassay)",
    "p": "p_pct (exassay)",
    "loi": "loi_pct (exassay)",
    "tio2": "tio2_pct (exassay)",
    "mgo": "mgo_pct (exassay)",
    "cao": "cao_pct (exassay)",
    "k2o": "k2o_pct (exassay)",
    "s": "s_pct (exassay)",
    "mn": "mn_pct (exassay)",
    "na2o": "na2o_pct (exassay)",
}

NORMATIVE_COLUMNS = [
    "ANATASE",
    "PYRITE",
    "APATITE",
    "PHOSPHOSIDERITE",
    "GIBBSITE",
    "KAOLINITE",
    "KFELDSP",
    "DOLOMITE",
    "CALCITE",
    "MAGNESITE",
    "MANJORITE",
    "MANGANESE",
    "CHERT",
    "GOETHITE",
    "HEMATITE",
    "XSLOI",
]

CORE_OXIDES = [
    "fe",
    "sio2",
    "al2o3",
    "p",
    "loi",
    "tio2",
    "mgo",
    "cao",
    "k2o",
    "s",
    "mn",
]

CLUSTER_RAW_FEATURES = [
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
    "gamma_mean_hole_z",
]


def _norm_key(df: pd.DataFrame, hole_col: str, from_col: str, to_col: str) -> pd.DataFrame:
    out = df.copy()
    out["_hole_key"] = out[hole_col].astype(str).str.upper().str.strip()
    out["_from_key"] = pd.to_numeric(out[from_col], errors="coerce").round(3)
    out["_to_key"] = pd.to_numeric(out[to_col], errors="coerce").round(3)
    return out


def _safe_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _clr_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    values = df[columns].astype(float).copy()
    positive = values.where(values > 0)
    min_positive = positive.min().min()
    eps = 1e-4 if pd.isna(min_positive) else max(float(min_positive) * 0.5, 1e-4)
    values = values.fillna(np.nan).clip(lower=eps)
    log_values = np.log(values)
    clr = log_values.sub(log_values.mean(axis=1), axis=0)
    return clr.add_prefix("clr_")


def _add_log_ratios(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-4
    ratios = {
        "lr_fe_sio2": ("fe", "sio2"),
        "lr_sio2_al2o3": ("sio2", "al2o3"),
        "lr_al2o3_tio2": ("al2o3", "tio2"),
        "lr_p_fe": ("p", "fe"),
        "lr_fe_mn": ("fe", "mn"),
        "lr_loi_fe": ("loi", "fe"),
    }
    for name, (num, den) in ratios.items():
        df[name] = np.log(df[num].clip(lower=eps) / df[den].clip(lower=eps))
    return df


def _gamma_interval_stats(biff: pd.DataFrame, geophysics_path: Path) -> pd.DataFrame:
    if not geophysics_path.exists():
        return biff

    geo = pd.read_parquet(geophysics_path, columns=["HOLEID", "DEPTH", "GAMMA_CPS"])
    geo["HOLEID"] = geo["HOLEID"].astype(str).str.upper().str.strip()
    geo["DEPTH"] = pd.to_numeric(geo["DEPTH"], errors="coerce")
    geo["GAMMA_CPS"] = pd.to_numeric(geo["GAMMA_CPS"], errors="coerce")
    geo = geo.dropna(subset=["HOLEID", "DEPTH", "GAMMA_CPS"]).sort_values(["HOLEID", "DEPTH"])

    hole_norm = geo.groupby("HOLEID")["GAMMA_CPS"].agg(["median"])
    hole_q = geo.groupby("HOLEID")["GAMMA_CPS"].quantile([0.25, 0.75]).unstack()
    hole_norm["iqr"] = (hole_q[0.75] - hole_q[0.25]).replace(0, np.nan)

    stats = []
    geo_groups = {hole: group for hole, group in geo.groupby("HOLEID", sort=False)}
    for idx, row in biff[["_hole_key", "depth_from", "depth_to"]].iterrows():
        group = geo_groups.get(row["_hole_key"])
        if group is None:
            stats.append((idx, np.nan, np.nan, np.nan, np.nan, 0))
            continue
        depths = group["DEPTH"].to_numpy()
        start = np.searchsorted(depths, row["depth_from"], side="left")
        end = np.searchsorted(depths, row["depth_to"], side="left")
        vals = group["GAMMA_CPS"].to_numpy()[start:end]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            stats.append((idx, np.nan, np.nan, np.nan, np.nan, 0))
        else:
            stats.append((idx, float(np.mean(vals)), float(np.nanmin(vals)), float(np.nanmax(vals)), float(np.nanstd(vals)), int(vals.size)))

    gamma = pd.DataFrame(
        stats,
        columns=["_row_index", "gamma_mean", "gamma_min", "gamma_max", "gamma_std", "gamma_n"],
    ).set_index("_row_index")
    out = biff.join(gamma)
    out["gamma_range"] = out["gamma_max"] - out["gamma_min"]
    med = out["_hole_key"].map(hole_norm["median"])
    iqr = out["_hole_key"].map(hole_norm["iqr"]).replace(0, np.nan)
    out["gamma_mean_hole_z"] = (out["gamma_mean"] - med) / iqr
    return out


def _fit_clusters(features: pd.DataFrame, k_min: int = 2, k_max: int = 8) -> tuple[pd.DataFrame, dict, np.ndarray]:
    imputed = features.copy()
    imputed = imputed.fillna(imputed.median(numeric_only=True))
    keep = imputed.columns[imputed.notna().any(axis=0)]
    imputed = imputed[keep]
    scaler = StandardScaler()
    x = scaler.fit_transform(imputed)

    rows = []
    labels_by_k = {}
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=30)
        labels = kmeans.fit_predict(x)
        labels_by_k[f"kmeans_{k}"] = labels
        sil = silhouette_score(x, labels) if len(np.unique(labels)) > 1 else np.nan
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42, n_init=5)
        gmm_labels = gmm.fit_predict(x)
        labels_by_k[f"gmm_{k}"] = gmm_labels
        rows.append(
            {
                "k": k,
                "kmeans_silhouette": sil,
                "gmm_bic": gmm.bic(x),
                "gmm_aic": gmm.aic(x),
            }
        )

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(x)
    diagnostics = pd.DataFrame(rows)
    diagnostics.attrs["pca_explained_variance"] = pca.explained_variance_ratio_.tolist()
    return diagnostics, labels_by_k, coords


def _cluster_summary(df: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    numeric_cols = [
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
        "gamma_mean",
        "gamma_mean_hole_z",
    ]
    rows = []
    for cluster, group in df.groupby(cluster_col, dropna=False):
        row = {
            "cluster": cluster,
            "count": len(group),
            "holes": group["hole_id"].nunique(),
            "biff_pct": (group["source_classification"].eq("biff").mean() * 100),
            "biff_s_pct": (group["source_classification"].eq("biff_s").mean() * 100),
            "mineral_rule_pct": (group["mineralised_rule"].mean() * 100),
            "bifhm_like_rule_pct": (group["bifhm_like_rule"].mean() * 100),
        }
        for col in numeric_cols:
            if col in group.columns:
                row[f"{col}_median"] = group[col].median()
                row[f"{col}_p10"] = group[col].quantile(0.10)
                row[f"{col}_p90"] = group[col].quantile(0.90)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["cluster"])


def _suggest_cluster_name(row: pd.Series) -> str:
    fe = row.get("fe_median", np.nan)
    si = row.get("sio2_median", np.nan)
    al = row.get("al2o3_median", np.nan)
    loi = row.get("loi_median", np.nan)
    py = row.get("PYRITE_median", np.nan)
    chert = row.get("CHERT_median", np.nan)
    gib = row.get("GIBBSITE_median", np.nan)
    kao = row.get("KAOLINITE_median", np.nan)
    goethite = row.get("GOETHITE_median", np.nan)
    hematite = row.get("HEMATITE_median", np.nan)
    mineral_pct = row.get("mineral_rule_pct", 0)
    bifhm_pct = row.get("bifhm_like_rule_pct", 0)

    if fe >= 55 and si <= 10 and al <= 5 and hematite >= 65:
        return "fresh_high_grade_hematite"
    if fe >= 50 and si <= 10 and (al >= 5 or loi >= 5 or gib + kao >= 10):
        return "aluminous_weathered_high_grade"
    if bifhm_pct >= 25 or si >= 30 or chert >= 25:
        return "fresh_cherty_itabirite"
    if al >= 8 or loi >= 7 or gib + kao >= 18:
        return "deep_clay_weathered_biff_s"
    if goethite >= 25 or loi >= 4:
        return "goethitic_hydrated_biff"
    if py >= 1:
        return "sulphidic_reduced_biff"
    return "intermediate_biff"


def _run_cluster_scope(
    valid: pd.DataFrame,
    scope_name: str,
    mask: pd.Series,
    feature_columns: list[str],
    output_dir: Path,
    chosen_k: int = 5,
) -> dict:
    """Run k-means/GMM diagnostics for one BIFf scope and write review files."""
    scope = valid[mask].copy()
    if len(scope) < max(100, chosen_k * 20):
        return {"scope": scope_name, "rows": int(len(scope)), "skipped": "too few rows"}

    diagnostics, labels_by_k, pca_coords = _fit_clusters(scope[feature_columns])
    best_sil_row = diagnostics.loc[diagnostics["kmeans_silhouette"].idxmax()]
    best_bic_row = diagnostics.loc[diagnostics["gmm_bic"].idxmin()]

    chosen_k = min(chosen_k, int(diagnostics["k"].max()))
    cluster_col = f"{scope_name}_cluster_k{chosen_k}"
    gmm_col = f"{scope_name}_gmm_k{chosen_k}"
    scope[cluster_col] = labels_by_k[f"kmeans_{chosen_k}"]
    scope[gmm_col] = labels_by_k[f"gmm_{chosen_k}"]
    scope["pca1"] = pca_coords[:, 0]
    scope["pca2"] = pca_coords[:, 1]

    summary = _cluster_summary(scope, cluster_col)
    summary["suggested_name"] = summary.apply(_suggest_cluster_name, axis=1)

    diagnostics_path = output_dir / f"{scope_name}_cluster_diagnostics.csv"
    summary_path = output_dir / f"{scope_name}_cluster_k{chosen_k}_summary.csv"
    intervals_path = output_dir / f"{scope_name}_interval_features_with_clusters.csv"

    diagnostics.to_csv(diagnostics_path, index=False)
    summary.to_csv(summary_path, index=False)

    export_columns = [
        "hole_id",
        "depth_from",
        "depth_to",
        "source_classification",
        "rev_gsymonds_comments",
        "fe",
        "sio2",
        "al2o3",
        "p",
        "loi",
        "tio2",
        "mn",
        "s",
        "mineralised_rule",
        "bifhm_like_rule",
        "high_al_weathered_rule",
        cluster_col,
        gmm_col,
        "pca1",
        "pca2",
        "gamma_mean",
        "gamma_mean_hole_z",
    ]
    export_columns += [col for col in NORMATIVE_COLUMNS if col in scope.columns]
    export_columns = [col for col in export_columns if col in scope.columns]
    scope[export_columns].to_csv(intervals_path, index=False)

    return {
        "scope": scope_name,
        "rows": int(len(scope)),
        "chosen_k": int(chosen_k),
        "best_k_by_silhouette": {
            "k": int(best_sil_row["k"]),
            "score": float(best_sil_row["kmeans_silhouette"]),
        },
        "best_k_by_gmm_bic": {
            "k": int(best_bic_row["k"]),
            "bic": float(best_bic_row["gmm_bic"]),
        },
        "pca_explained_variance": diagnostics.attrs.get("pca_explained_variance", []),
        "diagnostics": str(diagnostics_path),
        "summary": str(summary_path),
        "intervals": str(intervals_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze BIFf manual tags and sub-populations.")
    parser.add_argument("--classification-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--classification-column", default="consensus_classification")
    parser.add_argument("--normative", type=Path, default=DEFAULT_NORMATIVE)
    parser.add_argument("--geophysics", type=Path, default=DEFAULT_GEOPHYSICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.classification_csv, low_memory=False)
    if args.classification_column not in df.columns:
        raise ValueError(
            f"Classification column '{args.classification_column}' was not found. "
            f"Available columns include: {list(df.columns[:20])}"
        )
    df = _norm_key(df, "hole_id", "depth_from", "depth_to")
    df["source_classification"] = (
        df[args.classification_column].astype(str).str.lower().str.strip()
    )

    biff = df[df["source_classification"].isin(["biff", "biff_s"])].copy()
    for out_col, src_col in ASSAY_EXPORT_COLUMNS.items():
        biff[out_col] = pd.to_numeric(biff[src_col], errors="coerce") if src_col in biff.columns else np.nan

    if args.normative.exists():
        norm = pd.read_parquet(args.normative)
        norm = _norm_key(norm, "HOLEID", "SAMPFROM", "SAMPTO")
        norm = _safe_numeric(norm, NORMATIVE_COLUMNS)
        keep = ["_hole_key", "_from_key", "_to_key"] + NORMATIVE_COLUMNS
        biff = biff.merge(norm[keep], on=["_hole_key", "_from_key", "_to_key"], how="left")

    biff = _gamma_interval_stats(biff, args.geophysics)

    assay_mask = biff[["fe", "sio2", "al2o3"]].notna().all(axis=1)
    biff["mineralised_rule"] = assay_mask & (biff["fe"] > 50) & (biff["sio2"] < 10) & (biff["al2o3"] < 5)
    biff["bifhm_like_rule"] = assay_mask & (biff["fe"] < 40) & (biff["sio2"] > 35)
    biff["high_al_weathered_rule"] = assay_mask & ((biff["al2o3"] >= 5) | (biff["loi"] >= 5))
    biff = _add_log_ratios(biff)

    valid = biff[biff[CORE_OXIDES].notna().sum(axis=1) >= 8].copy()
    clr = _clr_frame(valid, CORE_OXIDES)
    valid = pd.concat([valid, clr], axis=1)

    clr_features = [f"clr_{col}" for col in CORE_OXIDES]
    ratio_features = ["lr_fe_sio2", "lr_sio2_al2o3", "lr_al2o3_tio2", "lr_p_fe", "lr_fe_mn", "lr_loi_fe"]
    raw_features = [col for col in CLUSTER_RAW_FEATURES if col in valid.columns]
    cluster_feature_columns = clr_features + ratio_features + raw_features
    cluster_features = valid[cluster_feature_columns]
    diagnostics, labels_by_k, pca_coords = _fit_clusters(cluster_features)

    best_sil_row = diagnostics.loc[diagnostics["kmeans_silhouette"].idxmax()]
    best_bic_row = diagnostics.loc[diagnostics["gmm_bic"].idxmin()]
    chosen_k = 5
    valid["cluster_k5"] = labels_by_k["kmeans_5"]
    valid["gmm_k5"] = labels_by_k["gmm_5"]
    valid["pca1"] = pca_coords[:, 0]
    valid["pca2"] = pca_coords[:, 1]

    summary = _cluster_summary(valid, "cluster_k5")
    summary["suggested_name"] = summary.apply(_suggest_cluster_name, axis=1)

    scope_outputs = [
        _run_cluster_scope(
            valid,
            "combined_biff_biff_s",
            pd.Series(True, index=valid.index),
            cluster_feature_columns,
            args.output_dir,
            chosen_k=5,
        ),
        _run_cluster_scope(
            valid,
            "biff_only",
            valid["source_classification"].eq("biff"),
            cluster_feature_columns,
            args.output_dir,
            chosen_k=5,
        ),
        _run_cluster_scope(
            valid,
            "biff_s_only",
            valid["source_classification"].eq("biff_s"),
            cluster_feature_columns,
            args.output_dir,
            chosen_k=5,
        ),
    ]

    class_summary = (
        biff.groupby("source_classification")
        .agg(
            intervals=("hole_id", "size"),
            holes=("hole_id", "nunique"),
            assay_coverage_pct=("fe", lambda s: s.notna().mean() * 100),
            fe_median=("fe", "median"),
            sio2_median=("sio2", "median"),
            al2o3_median=("al2o3", "median"),
            p_median=("p", "median"),
            loi_median=("loi", "median"),
            mineral_rule_count=("mineralised_rule", "sum"),
            bifhm_like_rule_count=("bifhm_like_rule", "sum"),
            high_al_weathered_rule_count=("high_al_weathered_rule", "sum"),
        )
        .reset_index()
    )

    rule_table = pd.crosstab(
        biff["source_classification"],
        [biff["mineralised_rule"], biff["bifhm_like_rule"], biff["high_al_weathered_rule"]],
    )

    diagnostics.to_csv(args.output_dir / "biff_cluster_diagnostics.csv", index=False)
    summary.to_csv(args.output_dir / "biff_cluster_k5_summary.csv", index=False)
    class_summary.to_csv(args.output_dir / "biff_manual_class_summary.csv", index=False)
    rule_table.to_csv(args.output_dir / "biff_rule_crosstab.csv")

    feature_columns = [
        "hole_id",
        "depth_from",
        "depth_to",
        "source_classification",
        "rev_gsymonds_comments",
        "fe",
        "sio2",
        "al2o3",
        "p",
        "loi",
        "tio2",
        "mn",
        "s",
        "mineralised_rule",
        "bifhm_like_rule",
        "high_al_weathered_rule",
        "cluster_k5",
        "gmm_k5",
        "pca1",
        "pca2",
        "gamma_mean",
        "gamma_mean_hole_z",
    ]
    feature_columns += [col for col in NORMATIVE_COLUMNS if col in valid.columns]
    feature_columns = [col for col in feature_columns if col in valid.columns]
    valid[feature_columns].to_csv(args.output_dir / "biff_interval_features_with_clusters.csv", index=False)

    plot_paths = {}
    if plt is not None:
        fig, ax = plt.subplots(figsize=(9, 6))
        scatter = ax.scatter(valid["sio2"], valid["fe"], c=valid["cluster_k5"], s=8, cmap="tab10", alpha=0.45)
        ax.axhline(50, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(10, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("SiO2 %")
        ax.set_ylabel("Fe %")
        ax.set_title("Tagged BIFf Population: Fe vs SiO2, coloured by k=5 cluster")
        fig.colorbar(scatter, ax=ax, label="cluster_k5")
        fig.tight_layout()
        path = args.output_dir / "biff_fe_sio2_clusters.png"
        fig.savefig(path, dpi=180)
        plot_paths["fe_sio2"] = str(path)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 6))
        scatter = ax.scatter(valid["pca1"], valid["pca2"], c=valid["cluster_k5"], s=8, cmap="tab10", alpha=0.45)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Tagged BIFf CLR/chemistry feature PCA")
        fig.colorbar(scatter, ax=ax, label="cluster_k5")
        fig.tight_layout()
        path = args.output_dir / "biff_pca_clusters.png"
        fig.savefig(path, dpi=180)
        plot_paths["pca"] = str(path)
        plt.close(fig)

    report = {
        "input_csv": str(args.classification_csv),
        "classification_column": args.classification_column,
        "biff_rows": int(len(biff)),
        "clustered_rows": int(len(valid)),
        "manual_class_counts": biff["source_classification"].value_counts().to_dict(),
        "assay_coverage_pct": float(biff["fe"].notna().mean() * 100),
        "normative_coverage_pct": float(biff["HEMATITE"].notna().mean() * 100) if "HEMATITE" in biff.columns else None,
        "gamma_coverage_pct": float(biff["gamma_mean"].notna().mean() * 100) if "gamma_mean" in biff.columns else None,
        "rules": {
            "mineralised_rule": "Fe > 50, SiO2 < 10, Al2O3 < 5",
            "bifhm_like_rule": "Fe < 40, SiO2 > 35",
            "high_al_weathered_rule": "Al2O3 >= 5 or LOI >= 5",
            "mineralised_rule_count": int(biff["mineralised_rule"].sum()),
            "bifhm_like_rule_count": int(biff["bifhm_like_rule"].sum()),
            "high_al_weathered_rule_count": int(biff["high_al_weathered_rule"].sum()),
        },
        "best_k_by_silhouette": {
            "k": int(best_sil_row["k"]),
            "score": float(best_sil_row["kmeans_silhouette"]),
        },
        "best_k_by_gmm_bic": {
            "k": int(best_bic_row["k"]),
            "bic": float(best_bic_row["gmm_bic"]),
        },
        "chosen_k": chosen_k,
        "scope_outputs": scope_outputs,
        "pca_explained_variance": diagnostics.attrs.get("pca_explained_variance", []),
        "outputs": {
            "cluster_diagnostics": str(args.output_dir / "biff_cluster_diagnostics.csv"),
            "cluster_summary": str(args.output_dir / "biff_cluster_k5_summary.csv"),
            "interval_features": str(args.output_dir / "biff_interval_features_with_clusters.csv"),
            "manual_class_summary": str(args.output_dir / "biff_manual_class_summary.csv"),
            "rule_crosstab": str(args.output_dir / "biff_rule_crosstab.csv"),
            **plot_paths,
        },
    }
    with open(args.output_dir / "biff_population_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print("\nCluster summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
