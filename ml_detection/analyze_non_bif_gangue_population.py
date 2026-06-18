"""Cluster non-BIF/non-mineralised intervals using composition and weathering evidence.

This is a review workflow for weakly labelled gangue/lithology populations.
It does not copy, move, or edit source images.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


DEFAULT_CLASSIFICATION_CSV = Path(
    r"C:\Users\gsymonds\Downloads\classifications_export_20260617_151205.csv"
)
DEFAULT_NORMATIVE = Path(
    r"C:\Users\gsymonds\AppData\Roaming\GeoVue\sf_cache\AA_EXPLORATION_AFR__SENS_GAB__NORMATIVE_MINERALOGY.parquet"
)
DEFAULT_GEOPHYSICS = Path(
    r"C:\Users\gsymonds\AppData\Roaming\GeoVue\sf_cache\AA_EXPLORATION_AFR__SENS_GAB__GEOPHYSICSDETAILS.parquet"
)
DEFAULT_OUTPUT = Path("ml_detection/dataset_audit/non_bif_gangue_clustering")

BIF_FAMILY_CLASSES = {"biff", "biff_s", "bifhm", "compact"}
OXIDE_COLUMNS = {
    "fe": "fe_pct (exassay)",
    "sio2": "sio2_pct (exassay)",
    "al2o3": "al2o3_pct (exassay)",
    "p": "p_pct (exassay)",
    "loi": "loi_pct (exassay)",
    "tio2": "tio2_pct (exassay)",
    "mgo": "mgo_pct (exassay)",
    "cao": "cao_pct (exassay)",
    "k2o": "k2o_pct (exassay)",
    "na2o": "na2o_pct (exassay)",
    "s": "s_pct (exassay)",
    "mn": "mn_pct (exassay)",
    "ni": "ni_pct (exassay)",
    "cr": "cr_pct (exassay)",
}
CLR_OXIDES = ["fe", "sio2", "al2o3", "p", "loi", "tio2", "mgo", "cao", "k2o", "na2o", "s", "mn"]
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
RC_COLUMNS = [
    "rc_total_gangue_pct",
    "rc_si_gangue_pct",
    "rc_al_gangue_pct",
    "rc_carbonate_gangue_pct",
    "rc_sulphide_gangue_pct",
    "rc_manganese_gangue_pct",
    "rc_mafics_gangue_pct",
    "rc_magnesium_gangue_pct",
    "rc_magnetite_pct",
    "rc_quartz_pct",
    "rc_chert_pct",
    "rc_weighted_zonation",
]


def normalise_key(df: pd.DataFrame, hole_col: str, from_col: str, to_col: str) -> pd.DataFrame:
    out = df.copy()
    out["_hole_key"] = out[hole_col].astype(str).str.upper().str.strip()
    out["_from_key"] = pd.to_numeric(out[from_col], errors="coerce").round(3)
    out["_to_key"] = pd.to_numeric(out[to_col], errors="coerce").round(3)
    return out


def add_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for out_col, src_col in OXIDE_COLUMNS.items():
        out[out_col] = pd.to_numeric(out[src_col], errors="coerce") if src_col in out.columns else np.nan
    for col in RC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def join_normative(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.exists():
        return df
    norm = pd.read_parquet(path)
    norm = normalise_key(norm, "HOLEID", "SAMPFROM", "SAMPTO")
    for col in NORMATIVE_COLUMNS:
        if col in norm.columns:
            norm[col] = pd.to_numeric(norm[col], errors="coerce")
    keep = ["_hole_key", "_from_key", "_to_key"] + [c for c in NORMATIVE_COLUMNS if c in norm.columns]
    return df.merge(norm[keep], on=["_hole_key", "_from_key", "_to_key"], how="left")


def add_gamma_mean_z(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.exists():
        df["gamma_mean"] = np.nan
        df["gamma_n"] = 0
        df["gamma_mean_hole_z"] = np.nan
        return df

    geo = pd.read_parquet(path, columns=["HOLEID", "DEPTH", "GAMMA_CPS"])
    geo["HOLEID"] = geo["HOLEID"].astype(str).str.upper().str.strip()
    geo["DEPTH"] = pd.to_numeric(geo["DEPTH"], errors="coerce")
    geo["GAMMA_CPS"] = pd.to_numeric(geo["GAMMA_CPS"], errors="coerce")
    geo = geo.dropna(subset=["HOLEID", "DEPTH", "GAMMA_CPS"]).sort_values(["HOLEID", "DEPTH"])

    hole_norm = geo.groupby("HOLEID")["GAMMA_CPS"].agg(["median"])
    q = geo.groupby("HOLEID")["GAMMA_CPS"].quantile([0.25, 0.75]).unstack()
    hole_norm["iqr"] = (q[0.75] - q[0.25]).replace(0, np.nan)

    out = df.copy()
    out["gamma_mean"] = np.nan
    out["gamma_n"] = 0

    for hole, idx in out.groupby("_hole_key").groups.items():
        g = geo[geo["HOLEID"].eq(hole)]
        if g.empty:
            continue
        depths = g["DEPTH"].to_numpy()
        values = g["GAMMA_CPS"].to_numpy()
        csum = np.concatenate([[0.0], np.cumsum(values)])
        starts = np.searchsorted(depths, out.loc[idx, "depth_from"].to_numpy(), side="left")
        ends = np.searchsorted(depths, out.loc[idx, "depth_to"].to_numpy(), side="left")
        counts = ends - starts
        valid = counts > 0
        means = np.full(len(idx), np.nan)
        means[valid] = (csum[ends[valid]] - csum[starts[valid]]) / counts[valid]
        out.loc[idx, "gamma_mean"] = means
        out.loc[idx, "gamma_n"] = counts

    med = out["_hole_key"].map(hole_norm["median"])
    iqr = out["_hole_key"].map(hole_norm["iqr"])
    out["gamma_mean_hole_z"] = (out["gamma_mean"] - med) / iqr
    return out


def clr_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    values = df[columns].astype(float)
    positives = values.where(values > 0)
    min_positive = positives.min().min()
    eps = 1e-4 if pd.isna(min_positive) else max(float(min_positive) * 0.5, 1e-4)
    values = values.clip(lower=eps)
    log_values = np.log(values)
    clr = log_values.sub(log_values.mean(axis=1), axis=0)
    return clr.add_prefix("clr_")


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    eps = 1e-4
    out["weathering_clay_index"] = (
        out.get("GIBBSITE", 0).fillna(0)
        + out.get("KAOLINITE", 0).fillna(0)
        + out["al2o3"].fillna(0)
        + out["loi"].fillna(0)
    )
    out["oxide_hydration_index"] = out.get("GOETHITE", 0).fillna(0) + out["loi"].fillna(0)
    out["carbonate_index"] = (
        out.get("DOLOMITE", 0).fillna(0)
        + out.get("CALCITE", 0).fillna(0)
        + out.get("MAGNESITE", 0).fillna(0)
        + out["cao"].fillna(0)
        + out["mgo"].fillna(0)
    )
    out["silica_index"] = out["sio2"].fillna(0) + out.get("CHERT", 0).fillna(0)
    out["sulphide_index"] = out["s"].fillna(0) + out.get("PYRITE", 0).fillna(0)
    out["mafic_index"] = out["mgo"].fillna(0) + out["cao"].fillna(0) + out["ni"].fillna(0) + out["cr"].fillna(0)
    out["lr_sio2_al2o3"] = np.log(out["sio2"].clip(lower=eps) / out["al2o3"].clip(lower=eps))
    out["lr_al2o3_tio2"] = np.log(out["al2o3"].clip(lower=eps) / out["tio2"].clip(lower=eps))
    out["lr_cao_mgo"] = np.log(out["cao"].clip(lower=eps) / out["mgo"].clip(lower=eps))
    out["lr_fe_sio2"] = np.log(out["fe"].clip(lower=eps) / out["sio2"].clip(lower=eps))
    return out


def choose_candidate_name(row: pd.Series) -> str:
    si = row.get("sio2_median", 0)
    al = row.get("al2o3_median", 0)
    k = row.get("k2o_median", 0)
    mg = row.get("mgo_median", 0)
    ca = row.get("cao_median", 0)
    fe = row.get("fe_median", 0)
    chert = row.get("CHERT_median", 0)
    gk = row.get("GIBBSITE_median", 0) + row.get("KAOLINITE_median", 0)
    carbonate = row.get("DOLOMITE_median", 0) + row.get("CALCITE_median", 0) + row.get("MAGNESITE_median", 0)
    pyrite = row.get("PYRITE_median", 0)

    if carbonate >= 10 or ca + mg >= 12:
        return "carbonate_magnesian"
    if pyrite >= 1.0:
        return "sulphidic"
    if si >= 55 or chert >= 30:
        return "siliceous_quartz_chert"
    if gk >= 20 or (al >= 12 and row.get("loi_median", 0) >= 5):
        return "weathered_aluminous_clay"
    if mg >= 5 or ca >= 5:
        return "mafic_ultramafic"
    if al >= 8 or k >= 1.5:
        return "phyllite_metapelite_clastic"
    if fe >= 35:
        return "ferruginous_mixed"
    return "mixed_gangue"


def top_counts(series: pd.Series, n: int = 5) -> str:
    counts = series.dropna().astype(str).value_counts().head(n)
    return "; ".join(f"{idx}:{val}" for idx, val in counts.items())


def summarize_clusters(df: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    numeric = [
        "fe",
        "sio2",
        "al2o3",
        "p",
        "loi",
        "tio2",
        "mgo",
        "cao",
        "k2o",
        "na2o",
        "s",
        "mn",
        "ni",
        "cr",
        "GIBBSITE",
        "KAOLINITE",
        "GOETHITE",
        "HEMATITE",
        "CHERT",
        "PYRITE",
        "APATITE",
        "DOLOMITE",
        "CALCITE",
        "MAGNESITE",
        "gamma_mean_hole_z",
        "weathering_clay_index",
        "carbonate_index",
        "silica_index",
        "mafic_index",
    ]
    rows = []
    for cluster, group in df.groupby(cluster_col, sort=True):
        row = {
            "cluster": int(cluster),
            "count": int(len(group)),
            "holes": int(group["hole_id"].nunique()),
            "top_beststrat": top_counts(group.get("BESTSTRAT", pd.Series(dtype=object))),
            "top_strat_exgeo": top_counts(group.get("strat (exgeologyRC)", pd.Series(dtype=object))),
            "top_profile": top_counts(group.get("profilezonation (exgeologyRC)", pd.Series(dtype=object))),
        }
        for col in numeric:
            if col in group.columns:
                row[f"{col}_median"] = group[col].median()
                row[f"{col}_p10"] = group[col].quantile(0.10)
                row[f"{col}_p90"] = group[col].quantile(0.90)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["candidate_name"] = summary.apply(choose_candidate_name, axis=1)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-csv", type=Path, default=DEFAULT_CLASSIFICATION_CSV)
    parser.add_argument("--normative", type=Path, default=DEFAULT_NORMATIVE)
    parser.add_argument("--geophysics", type=Path, default=DEFAULT_GEOPHYSICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chosen-k", type=int, default=10)
    parser.add_argument("--sample-size", type=int, default=30000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.classification_csv, low_memory=False)
    raw = normalise_key(raw, "hole_id", "depth_from", "depth_to")
    raw = add_numeric_columns(raw)
    cls = raw["consensus_classification"].astype(str).str.lower().str.strip()
    mineralised = (raw["fe"] > 50) & (raw["sio2"] < 10) & (raw["al2o3"] < 5)
    non_bif = raw[~cls.isin(BIF_FAMILY_CLASSES) & ~mineralised & cls.ne("not_confident")].copy()

    non_bif = join_normative(non_bif, args.normative)
    non_bif = add_gamma_mean_z(non_bif, args.geophysics)
    non_bif = add_derived_features(non_bif)

    valid = non_bif[non_bif[CLR_OXIDES].notna().sum(axis=1) >= 8].copy()
    clr = clr_frame(valid, CLR_OXIDES)
    valid = pd.concat([valid, clr], axis=1)

    feature_cols = [f"clr_{col}" for col in CLR_OXIDES]
    feature_cols += [
        "lr_sio2_al2o3",
        "lr_al2o3_tio2",
        "lr_cao_mgo",
        "lr_fe_sio2",
        "weathering_clay_index",
        "oxide_hydration_index",
        "carbonate_index",
        "silica_index",
        "sulphide_index",
        "mafic_index",
        "gamma_mean_hole_z",
    ]
    feature_cols += [col for col in NORMATIVE_COLUMNS if col in valid.columns]
    feature_cols += [col for col in RC_COLUMNS if col in valid.columns]
    feature_cols = [col for col in feature_cols if col in valid.columns]

    features = valid[feature_cols].copy()
    features = features.fillna(features.median(numeric_only=True))
    scaler = StandardScaler()
    x = scaler.fit_transform(features)

    rng = np.random.default_rng(42)
    sample_idx = np.arange(len(valid))
    if len(sample_idx) > args.sample_size:
        sample_idx = rng.choice(sample_idx, size=args.sample_size, replace=False)
    x_sample = x[sample_idx]

    diagnostics = []
    for k in range(4, 13):
        km = KMeans(n_clusters=k, random_state=42, n_init=30)
        sample_labels = km.fit_predict(x_sample)
        sil = silhouette_score(x_sample, sample_labels) if len(np.unique(sample_labels)) > 1 else np.nan
        gmm = GaussianMixture(n_components=k, covariance_type="diag", random_state=42, n_init=3)
        gmm.fit(x_sample)
        diagnostics.append({"k": k, "sample_kmeans_silhouette": sil, "sample_gmm_bic_diag": gmm.bic(x_sample)})

    chosen = KMeans(n_clusters=args.chosen_k, random_state=42, n_init=50)
    valid["non_bif_cluster_k10"] = chosen.fit_predict(x)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(x_sample)
    sample_plot = valid.iloc[sample_idx][["hole_id", "depth_from", "depth_to", "non_bif_cluster_k10"]].copy()
    sample_plot["pca1"] = coords[:, 0]
    sample_plot["pca2"] = coords[:, 1]

    summary = summarize_clusters(valid, "non_bif_cluster_k10")
    diagnostics_df = pd.DataFrame(diagnostics)

    assignment_cols = [
        "HOLEID",
        "SAMPFROM",
        "SAMPTO",
        "non_bif_cluster_k10",
        "BESTSTRAT",
        "STRATUNIT",
        "STRATSUM",
        "PROFILEZONATION",
    ]
    valid["HOLEID"] = valid["hole_id"].astype(str).str.upper().str.strip()
    valid["SAMPFROM"] = pd.to_numeric(valid["depth_from"], errors="coerce")
    valid["SAMPTO"] = pd.to_numeric(valid["depth_to"], errors="coerce")
    valid["BESTSTRAT"] = valid.get("strat (exgeologyRC)", pd.Series(index=valid.index))
    valid["STRATUNIT"] = valid.get("strat (exgeologyRC)", pd.Series(index=valid.index))
    valid["STRATSUM"] = valid.get("strat (exgeologyRC)", pd.Series(index=valid.index))
    valid["PROFILEZONATION"] = valid.get("profilezonation (exgeologyRC)", pd.Series(index=valid.index))

    export_cols = [
        "hole_id",
        "depth_from",
        "depth_to",
        "consensus_classification",
        "non_bif_cluster_k10",
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
        "GIBBSITE",
        "KAOLINITE",
        "GOETHITE",
        "HEMATITE",
        "CHERT",
        "PYRITE",
        "DOLOMITE",
        "CALCITE",
        "MAGNESITE",
        "gamma_mean_hole_z",
        "strat (exgeologyRC)",
        "profilezonation (exgeologyRC)",
        "rc_si_gangue_pct",
        "rc_al_gangue_pct",
        "rc_carbonate_gangue_pct",
        "rc_mafics_gangue_pct",
        "rc_quartz_pct",
        "rc_chert_pct",
    ]
    export_cols = [c for c in export_cols if c in valid.columns]

    diagnostics_df.to_csv(args.output_dir / "non_bif_cluster_diagnostics.csv", index=False)
    summary.to_csv(args.output_dir / "non_bif_cluster_k10_summary.csv", index=False)
    sample_plot.to_csv(args.output_dir / "non_bif_cluster_pca_sample.csv", index=False)
    valid[export_cols].to_csv(args.output_dir / "non_bif_interval_features_with_clusters.csv", index=False)
    valid[assignment_cols].to_parquet(args.output_dir / "non_bif_cluster_assignments.parquet", index=False)
    valid[assignment_cols].to_csv(args.output_dir / "non_bif_cluster_assignments.csv", index=False)

    plot_path = None
    if plt is not None:
        fig, ax = plt.subplots(figsize=(9, 6))
        sc = ax.scatter(sample_plot["pca1"], sample_plot["pca2"], c=sample_plot["non_bif_cluster_k10"], s=5, cmap="tab10", alpha=0.45)
        ax.set_title("Non-BIF gangue clustering PCA sample")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.colorbar(sc, ax=ax, label="non_bif_cluster_k10")
        fig.tight_layout()
        plot_path = args.output_dir / "non_bif_cluster_pca_sample.png"
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)

    report = {
        "classification_csv": str(args.classification_csv),
        "rows_after_exclusions": int(len(non_bif)),
        "clustered_rows": int(len(valid)),
        "excluded_bif_family_rows": int(cls.isin(BIF_FAMILY_CLASSES).sum()),
        "excluded_mineralised_rows": int((~cls.isin(BIF_FAMILY_CLASSES) & mineralised).sum()),
        "excluded_not_confident_rows": int(cls.eq("not_confident").sum()),
        "assay_coverage_pct": float(non_bif["fe"].notna().mean() * 100),
        "normative_coverage_pct": float(non_bif["HEMATITE"].notna().mean() * 100) if "HEMATITE" in non_bif.columns else None,
        "gamma_coverage_pct": float(non_bif["gamma_mean"].notna().mean() * 100),
        "chosen_k": args.chosen_k,
        "cluster_counts": valid["non_bif_cluster_k10"].value_counts().sort_index().to_dict(),
        "cluster_hole_counts": valid.groupby("non_bif_cluster_k10")["hole_id"].nunique().sort_index().to_dict(),
        "outputs": {
            "diagnostics": str(args.output_dir / "non_bif_cluster_diagnostics.csv"),
            "summary": str(args.output_dir / "non_bif_cluster_k10_summary.csv"),
            "interval_features": str(args.output_dir / "non_bif_interval_features_with_clusters.csv"),
            "assignments": str(args.output_dir / "non_bif_cluster_assignments.parquet"),
            "pca_plot": str(plot_path) if plot_path else None,
        },
    }
    with (args.output_dir / "non_bif_cluster_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print("\nCluster summary:")
    view_cols = [
        "cluster",
        "count",
        "holes",
        "candidate_name",
        "fe_median",
        "sio2_median",
        "al2o3_median",
        "mgo_median",
        "cao_median",
        "k2o_median",
        "loi_median",
        "GIBBSITE_median",
        "KAOLINITE_median",
        "CHERT_median",
        "PYRITE_median",
        "top_strat_exgeo",
        "top_profile",
    ]
    view_cols = [c for c in view_cols if c in summary.columns]
    print(summary[view_cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
