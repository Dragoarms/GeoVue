import io
import logging
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from processing.DataManager.column_aliases import ColumnResolver
from processing.DataManager.interval_merger import IntervalMerger, merge_intervals_by_range
from processing.DataManager.keys import ImageKey

logger = logging.getLogger(__name__)


LOGGER_COLUMN_PRIORITY = [
    "LoggedBy",       # Per-interval logger - most reliable
    "LoggedBy_D",     # Per-hole logger - fallback
    "LoggedByCode",
    "LoggedByCode_D",
    "Logger",
]

DRILLDATE_COLUMN_PRIORITY = [
    "drilldate",
    "drill_date",
    "drilldate_d",
    "drill_date_d",
]

MAJOR_ELEMENT_STANDARD = [
    "fe_pct",
    "sio2_pct",
    "al2o3_pct",
    "p_pct",
    "s_pct",
    "loi_pct",
    "mn_pct",
    "cao_pct",
    "mgo_pct",
    "k2o_pct",
    "na2o_pct",
    "tio2_pct",
]

def _find_column_case_insensitive(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    columns_lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        actual = columns_lower.get(candidate.lower())
        if actual:
            return actual
    return None


def resolve_logger_column(df: pd.DataFrame) -> Optional[str]:
    """Resolve the best logger column from a DataFrame."""
    return _find_column_case_insensitive(df, LOGGER_COLUMN_PRIORITY)


def resolve_drilldate_column(df: pd.DataFrame) -> Optional[str]:
    """Resolve the best drill date column from a DataFrame."""
    return _find_column_case_insensitive(df, DRILLDATE_COLUMN_PRIORITY)


def fill_empty_logger_values(df: pd.DataFrame, logger_col: str, hole_col: str) -> pd.DataFrame:
    """
    Fill empty loggedby values using loggedby_d or forward/backward fill within holes.

    Priority:
    1. Use loggedby_d if available for the row
    2. Forward fill from previous interval in same hole
    3. Backward fill from next interval in same hole
    """
    if logger_col not in df.columns:
        return df

    result = df.copy()

    # Find empty logger values
    empty_mask = result[logger_col].isna() | (result[logger_col].astype(str).str.strip() == '')
    empty_count = empty_mask.sum()

    if empty_count == 0:
        return result

    logger.info(f"Filling {empty_count} empty {logger_col} values")

    # Try to fill from loggedby_d column first
    loggedby_d_col = None
    for candidate in ['loggedby_d', 'LoggedBy_D', 'LOGGEDBY_D']:
        if candidate in result.columns and candidate.lower() != logger_col.lower():
            loggedby_d_col = candidate
            break

    if loggedby_d_col:
        # Fill empty loggedby from loggedby_d
        fill_mask = empty_mask & result[loggedby_d_col].notna() & (result[loggedby_d_col].astype(str).str.strip() != '')
        result.loc[fill_mask, logger_col] = result.loc[fill_mask, loggedby_d_col]
        filled_from_d = fill_mask.sum()
        logger.info(f"  Filled {filled_from_d} values from {loggedby_d_col}")
        empty_mask = result[logger_col].isna() | (result[logger_col].astype(str).str.strip() == '')

    # Forward/backward fill within each hole for any remaining empty values
    if hole_col in result.columns and empty_mask.any():
        remaining_before = empty_mask.sum()
        result[logger_col] = result.groupby(hole_col)[logger_col].transform(
            lambda x: x.ffill().bfill()
        )
        empty_mask = result[logger_col].isna() | (result[logger_col].astype(str).str.strip() == '')
        filled_from_adjacent = remaining_before - empty_mask.sum()
        logger.info(f"  Filled {filled_from_adjacent} values from adjacent intervals")

    remaining_empty = empty_mask.sum()
    if remaining_empty > 0:
        logger.warning(f"  {remaining_empty} intervals still have empty {logger_col}")

    return result


def _robust_scale(series: pd.Series) -> pd.Series:
    median = series.median()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        std = series.std(ddof=0)
        if std == 0 or np.isnan(std):
            return series * 0.0
        return (series - median) / std
    robust_sigma = iqr / 1.349
    return (series - median) / robust_sigma


def compute_hybrid_outlier_scores(
    df: pd.DataFrame,
    strat_col: str,
    chem_cols: List[str],
    min_group_size: int = 10,
) -> pd.DataFrame:
    """
    Compute hybrid outlier scores per strat using robust multivariate distance and IQR flags.

    Returns a DataFrame aligned to df with columns:
    - outlier_score
    - outlier_reason
    """
    result = pd.DataFrame(index=df.index)
    result["outlier_score"] = 0.0
    result["outlier_reason"] = ""
    result["outlier_elements"] = ""

    if strat_col not in df.columns:
        logger.warning("Strat column '%s' not found for outlier scoring.", strat_col)
        return result

    for strat_value, group in df.groupby(strat_col):
        if len(group) < min_group_size:
            continue

        available_cols = [c for c in chem_cols if c in group.columns]
        if not available_cols:
            continue

        scaled = group[available_cols].apply(_robust_scale)
        scaled = scaled.replace([np.inf, -np.inf], np.nan)

        z_means = scaled.mean(axis=0, skipna=True)
        z_centered = scaled - z_means

        # Compute robust covariance with simple shrinkage to avoid instability
        cov = np.cov(z_centered.fillna(0.0).T, bias=True)
        cov = np.atleast_2d(cov)
        dim = cov.shape[0]
        if dim == 0:
            continue

        if len(group) < 10:
            alpha = 0.5
        elif len(group) < 30:
            alpha = 0.2
        else:
            alpha = 0.05
        cov_shrink = (1 - alpha) * cov + alpha * np.eye(dim)
        inv_cov = np.linalg.pinv(cov_shrink)

        mahal = []
        for idx, row in z_centered.iterrows():
            vec = row.values.astype(float)
            if np.all(np.isnan(vec)):
                mahal.append(0.0)
                continue
            mask = ~np.isnan(vec)
            vec = vec[mask]
            if vec.size == 0:
                mahal.append(0.0)
                continue
            sub_cov = inv_cov[np.ix_(mask, mask)]
            mahal.append(float(np.sqrt(vec.T @ sub_cov @ vec)))

        mahal = pd.Series(mahal, index=group.index)
        mahal_iqr = mahal.quantile(0.75) - mahal.quantile(0.25)
        if mahal_iqr == 0:
            mahal_norm = mahal * 0.0
        else:
            mahal_norm = (mahal - mahal.median()) / (mahal_iqr / 1.349)

        # IQR outlier flags and per-element severity
        flag_score = pd.Series(0.0, index=group.index)
        reason_text = pd.Series("", index=group.index)
        top_elements = pd.Series("", index=group.index)
        for col in available_cols:
            series = group[col]
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            z = _robust_scale(series)
            is_outlier = (series < lower) | (series > upper)
            flag_score[is_outlier] += z[is_outlier].abs()
            for idx in series[is_outlier].index:
                current = reason_text.at[idx]
                direction = "high" if series.at[idx] > upper else "low"
                reason = f"{col} {direction} (value {series.at[idx]:.2f}, IQR {q1:.2f}-{q3:.2f})"
                reason_text.at[idx] = reason if current == "" else f"{current}; {reason}"

        # Track top elements by absolute robust z-score
        z_abs = scaled[available_cols].abs()
        for idx, row in z_abs.iterrows():
            top = row.sort_values(ascending=False).head(3).index.tolist()
            top_elements.at[idx] = ", ".join(top)

        result.loc[group.index, "outlier_score"] = (mahal_norm + flag_score).fillna(0.0)
        result.loc[group.index, "outlier_reason"] = reason_text.fillna("")
        result.loc[group.index, "outlier_elements"] = top_elements.fillna("")

    return result


def predict_most_likely_strat(
    df: pd.DataFrame,
    strat_col: str,
    chem_cols: List[str],
    min_group_size: int = 10,
) -> pd.Series:
    """
    Predict the most likely strat classification for each row based on multivariate distance.

    Uses Mahalanobis distance to all strat group centroids. The strat with the smallest
    distance is the "most likely" classification. This helps identify misclassifications
    where an interval's geochemistry better matches a different strat group.

    Args:
        df: DataFrame with strat and chemistry columns
        strat_col: Name of the strat/classification column
        chem_cols: List of chemistry column names to use for distance calculation
        min_group_size: Minimum samples in a strat group to include it as a candidate

    Returns:
        Series with predicted "most likely" strat for each row
    """
    result = pd.Series(index=df.index, dtype=object)
    result[:] = "-"

    if strat_col not in df.columns:
        logger.warning("Strat column '%s' not found for prediction.", strat_col)
        return result

    available_cols = [c for c in chem_cols if c in df.columns]
    if not available_cols:
        logger.warning("No chemistry columns available for strat prediction.")
        return result

    # Build strat group statistics (centroids and covariances)
    strat_stats = {}
    for strat_value, group in df.groupby(strat_col):
        if pd.isna(strat_value) or len(group) < min_group_size:
            continue

        data = group[available_cols].dropna(how="all")
        if len(data) < min_group_size:
            continue

        # Compute centroid (median for robustness)
        centroid = data.median()

        # Compute robust covariance with shrinkage
        z_data = data.apply(lambda s: _robust_scale(s) if s.std() > 0 else s * 0)
        z_data = z_data.replace([np.inf, -np.inf], np.nan).fillna(0)

        cov = np.cov(z_data.T, bias=True)
        cov = np.atleast_2d(cov)
        dim = cov.shape[0]

        if dim == 0:
            continue

        # Shrinkage for stability
        alpha = 0.3 if len(data) < 30 else 0.1
        cov_shrink = (1 - alpha) * cov + alpha * np.eye(dim)

        try:
            inv_cov = np.linalg.pinv(cov_shrink)
        except Exception:
            continue

        strat_stats[strat_value] = {
            "centroid": centroid,
            "inv_cov": inv_cov,
            "count": len(group),
        }

    if not strat_stats:
        return result

    # For each row, find the strat with minimum Mahalanobis distance
    for idx, row in df.iterrows():
        row_values = row[available_cols]
        if row_values.isna().all():
            continue

        current_strat = row.get(strat_col)
        min_dist = float("inf")
        best_strat = current_strat if pd.notna(current_strat) else "-"

        for strat_value, stats in strat_stats.items():
            centroid = stats["centroid"]
            inv_cov = stats["inv_cov"]

            # Compute distance to this strat's centroid
            diff = row_values - centroid
            mask = ~diff.isna()
            if mask.sum() == 0:
                continue

            diff_vec = diff[mask].values.astype(float)
            sub_cov = inv_cov[np.ix_(mask.values, mask.values)]

            try:
                dist = float(np.sqrt(diff_vec.T @ sub_cov @ diff_vec))
            except Exception:
                continue

            if dist < min_dist:
                min_dist = dist
                best_strat = strat_value

        result.at[idx] = best_strat if pd.notna(best_strat) else "-"

    return result


def filter_dataframe_by_logger_and_date(
    df: pd.DataFrame,
    logger_col: str,
    date_col: str,
    logger_values: Optional[List[str]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> pd.DataFrame:
    """Filter a DataFrame by logger list and drill date range."""
    filtered = df.copy()

    if logger_values:
        logger_values = [str(v) for v in logger_values]
        filtered = filtered[filtered[logger_col].astype(str).isin(logger_values)]

    if date_col and (date_from or date_to):
        dates = pd.to_datetime(filtered[date_col], errors="coerce")
        if date_from:
            start = pd.to_datetime(date_from, errors="coerce")
            filtered = filtered[dates >= start]
        if date_to:
            end = pd.to_datetime(date_to, errors="coerce")
            filtered = filtered[dates <= end]

    return filtered


def resolve_chemistry_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Resolve major chemistry columns from a DataFrame."""
    resolver = ColumnResolver(df)
    resolved = {}
    for standard in MAJOR_ELEMENT_STANDARD:
        actual = resolver.get(standard)
        if actual:
            resolved[standard] = actual
    return resolved


def get_collar_dataframe(data_coordinator) -> pd.DataFrame:
    if not data_coordinator or not data_coordinator.is_initialized:
        return pd.DataFrame()
    geo_store = data_coordinator.geological_store
    if not geo_store:
        return pd.DataFrame()
    for source_name in ["excollar", "collar", "collars"]:
        if source_name in geo_store.list_sources():
            src = geo_store.get_source(source_name)
            if src and src.df is not None and not src.df.empty:
                return src.df.copy()
    return pd.DataFrame()


def build_merged_qaqc_dataframe(
    data_coordinator,
    hole_ids: Optional[set] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    merger = IntervalMerger(data_coordinator.geological_store)
    merged_df, stats = merger.get_merged_dataframe_for_qaqc(
        primary_source="exassay",
        merge_geology=True,
        hole_ids=hole_ids,
    )
    return merged_df, stats.to_dict() if hasattr(stats, "to_dict") else {}


def _log_dataframe_overview(name: str, df: pd.DataFrame, extra_cols: Optional[List[str]] = None) -> None:
    if df is None:
        logger.info("%s: <none>", name)
        return
    columns = list(df.columns)
    preview = columns[:15]
    suffix = f" (+{len(columns) - 15} more)" if len(columns) > 15 else ""

    # Calculate memory usage
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    logger.info(
        f"DATAFRAME OVERVIEW: {name}\n"
        f"  Shape: {len(df):,} rows x {len(columns)} columns\n"
        f"  Memory: {memory_mb:.1f} MB\n"
        f"  Columns: {preview}{suffix}"
    )
    if extra_cols:
        present = [col for col in extra_cols if col in columns]
        if present:
            logger.info(f"  Matched columns of interest: {present}")


def _find_logger_source(geo_store) -> Optional[Tuple[str, pd.DataFrame, str, str, str]]:
    if not geo_store:
        return None
    for source_name in geo_store.list_sources():
        src = geo_store.get_source(source_name)
        if not src or src.df is None or src.df.empty:
            continue
        df = src.df
        logger_col = resolve_logger_column(df)
        if not logger_col:
            continue
        resolver = ColumnResolver(df)
        hole_col = resolver.get("hole_id")
        from_col = resolver.get("depth_from")
        to_col = resolver.get("depth_to")
        if hole_col and from_col and to_col:
            return source_name, df, hole_col, from_col, to_col
    return None


def _find_exgeology_rc_source(geo_store) -> Optional[Tuple[str, pd.DataFrame]]:
    if not geo_store:
        return None
    for source_name in geo_store.list_sources():
        if source_name.lower() == "exgeologyrc":
            src = geo_store.get_source(source_name)
            if src and src.df is not None and not src.df.empty:
                return source_name, src.df.copy()
    return None


def _resolve_hole_ids_by_date(
    data_coordinator,
    date_from: Optional[str],
    date_to: Optional[str],
) -> Optional[Set[str]]:
    if not date_from and not date_to:
        return None
    collar_df = get_collar_dataframe(data_coordinator)
    if collar_df.empty:
        return None
    drilldate_col = resolve_drilldate_column(collar_df)
    if not drilldate_col:
        return None
    resolver = ColumnResolver(collar_df)
    hole_col = resolver.get("hole_id")
    if not hole_col:
        return None
    dates = pd.to_datetime(collar_df[drilldate_col], errors="coerce")
    mask = pd.Series(True, index=collar_df.index)
    if date_from:
        start = pd.to_datetime(date_from, errors="coerce")
        if not pd.isna(start):
            mask &= dates >= start
    if date_to:
        end = pd.to_datetime(date_to, errors="coerce")
        if not pd.isna(end):
            mask &= dates <= end
    hole_ids = (
        collar_df.loc[mask, hole_col]
        .dropna()
        .astype(str)
        .str.upper()
        .str.strip()
        .tolist()
    )
    return set(hole_ids)


def _merge_logger_by_overlap(
    merged_df: pd.DataFrame,
    logger_df: pd.DataFrame,
    merged_hole_col: str,
    merged_from_col: str,
    merged_to_col: str,
    logger_hole_col: str,
    logger_from_col: str,
    logger_to_col: str,
    hole_ids: Optional[Set[str]] = None,
) -> pd.DataFrame:
    logger_candidates = [
        col
        for col in logger_df.columns
        if col.lower() in {c.lower() for c in LOGGER_COLUMN_PRIORITY}
    ]
    if not logger_candidates:
        return merged_df
    logger_df = logger_df.reset_index(drop=True)
    for col in ["_hole_upper", "_depth_int"]:
        if col in logger_df.columns:
            logger_df = logger_df.drop(columns=[col])
    merged_df, _ = merge_intervals_by_range(
        primary_df=merged_df,
        secondary_df=logger_df,
        primary_keys={"hole_id": merged_hole_col, "from": merged_from_col, "to": merged_to_col},
        secondary_keys={"hole_id": logger_hole_col, "from": logger_from_col, "to": logger_to_col},
        cols_to_merge=logger_candidates,
        hole_ids=hole_ids,
    )
    return merged_df


def _merge_rc_metrics_by_overlap(
    merged_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    hole_col: str,
    depth_from_col: str,
    depth_to_col: str,
    hole_ids: Optional[Set[str]] = None,
) -> pd.DataFrame:
    if merged_df.empty or metrics_df.empty:
        return merged_df

    metrics_resolver = ColumnResolver(metrics_df)
    metrics_hole_col = metrics_resolver.get("hole_id") or (hole_col if hole_col in metrics_df.columns else None)
    metrics_from_col = (
        metrics_resolver.get("depth_from")
        or (depth_from_col if depth_from_col in metrics_df.columns else None)
    )
    metrics_to_col = metrics_resolver.get("depth_to") or (depth_to_col if depth_to_col in metrics_df.columns else None)

    if not metrics_hole_col or not metrics_from_col or not metrics_to_col:
        logger.warning("RC metrics missing required hole/depth columns; skipping overlap merge.")
        return merged_df

    primary_cols_lower = {c.lower() for c in merged_df.columns}
    metrics_key_cols = {metrics_hole_col, metrics_from_col, metrics_to_col}
    cols_to_merge = [
        col for col in metrics_df.columns
        if col not in metrics_key_cols
        and col.lower() not in primary_cols_lower
        and not col.startswith("_")
    ]

    if not cols_to_merge:
        return merged_df

    metrics_df = metrics_df.reset_index(drop=True)
    for col in ["_hole_upper", "_depth_int"]:
        if col in metrics_df.columns:
            metrics_df = metrics_df.drop(columns=[col])
    merged_df, _ = merge_intervals_by_range(
        primary_df=merged_df,
        secondary_df=metrics_df,
        primary_keys={"hole_id": hole_col, "from": depth_from_col, "to": depth_to_col},
        secondary_keys={"hole_id": metrics_hole_col, "from": metrics_from_col, "to": metrics_to_col},
        cols_to_merge=cols_to_merge,
        hole_ids=hole_ids,
    )
    return merged_df


def _resolve_logging_interval_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    from_candidates = [
        "geolfrom",
        "geology_from",
        "logging_from",
        "rc_from",
        "depth_from_geol",
    ]
    to_candidates = [
        "geolto",
        "geology_to",
        "logging_to",
        "rc_to",
        "depth_to_geol",
    ]
    columns_lower = {c.lower(): c for c in df.columns}
    log_from = next((columns_lower[c] for c in from_candidates if c in columns_lower), None)
    log_to = next((columns_lower[c] for c in to_candidates if c in columns_lower), None)
    return log_from, log_to


def _build_logging_interval_dataframe(
    data_coordinator,
    logger_values: Optional[List[str]],
    date_from: Optional[str],
    date_to: Optional[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    geo_store = data_coordinator.geological_store if data_coordinator else None
    exgeo = _find_exgeology_rc_source(geo_store)
    if not exgeo:
        raise ValueError("exgeologyRC source not found for logging interval dataframe.")
    source_name, logging_df = exgeo

    resolver = ColumnResolver(logging_df)
    hole_col = resolver.get("hole_id")
    depth_from_col = resolver.get("depth_from")
    depth_to_col = resolver.get("depth_to")
    if not hole_col or not depth_from_col or not depth_to_col:
        raise ValueError("exgeologyRC missing required hole/from/to columns.")

    logger_source = _find_logger_source(geo_store)
    hole_ids = None
    date_hole_ids = _resolve_hole_ids_by_date(data_coordinator, date_from, date_to)
    if date_hole_ids is not None:
        hole_ids = date_hole_ids
        logger.info(
            "Drilldate prefilter: %d holes in range (%s to %s)",
            len(date_hole_ids),
            date_from or "...",
            date_to or "...",
        )
        logger.debug(
            "Drilldate hole_ids sample: %s",
            sorted(list(date_hole_ids))[:25],
        )

    if logger_source:
        _, logger_df, logger_hole_col, logger_from_col, logger_to_col = logger_source
        logger_col = resolve_logger_column(logger_df)
        if logger_col:
            if logger_values:
                logger_df = logger_df[
                    logger_df[logger_col].astype(str).isin([str(v) for v in logger_values])
                ].copy()
            if hole_ids is not None:
                logger_df = logger_df[
                    logger_df[logger_hole_col].astype(str).str.upper().str.strip().isin(hole_ids)
                ].copy()
            logging_df = _merge_logger_by_overlap(
                merged_df=logging_df,
                logger_df=logger_df,
                merged_hole_col=hole_col,
                merged_from_col=depth_from_col,
                merged_to_col=depth_to_col,
                logger_hole_col=logger_hole_col,
                logger_from_col=logger_from_col,
                logger_to_col=logger_to_col,
                hole_ids=hole_ids,
            )
    if hole_ids is not None:
        logging_df = logging_df[
            logging_df[hole_col].astype(str).str.upper().str.strip().isin(hole_ids)
        ]

    stats = {
        "primary_source": source_name,
        "total_primary_rows": len(logging_df),
        "holes_processed": logging_df[hole_col].nunique(),
    }
    return logging_df, stats


def _summarize_grouping_accuracy_by_interval(
    df: pd.DataFrame,
    group_cols: List[str],
    chem_cols: List[str],
) -> List[str]:
    if not all(col in df.columns for col in group_cols):
        return ["Missing grouping columns for interval accuracy summary."]

    available = [c for c in chem_cols if c in df.columns]
    if not available:
        return ["Missing Fe/SiO2/Al2O3 columns for grouping accuracy summary."]

    grouped = df.groupby(group_cols)
    total_groups = grouped.ngroups
    invalid_counts = {col: 0 for col in available}

    for _, group in grouped:
        for element in available:
            values = group[element].dropna()
            if len(values) < 2:
                continue
            mean = values.mean()
            if mean == 0:
                continue
            cv = values.std(ddof=1) / abs(mean) * 100
            if cv > 100:
                invalid_counts[element] += 1

    lines = [f"Groups evaluated: {total_groups:,}"]
    for element in available:
        lines.append(f"- {element}: {invalid_counts[element]} / {total_groups} groups exceed CV 100%")
    return lines


def _build_boxplot_image(series: pd.Series, outlier_value: float) -> ImageReader:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    values = series.dropna().values
    if values.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.boxplot(values, orientation="vertical", showfliers=True)
        if not np.isnan(outlier_value):
            ax.scatter([1], [outlier_value], color="red", zorder=3)
        ax.set_xticks([])
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    buffer.seek(0)
    return ImageReader(buffer)


def prepare_logging_review_data(
    data_coordinator,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Build merged and logging dataframes for the whole dataset (no date/logger filter).
    Use this to cache data so report generation only filters and renders.
    progress_callback(message: str, fraction: float) may be called with 0.0-1.0.
    """
    def _progress(msg: str, pct: float) -> None:
        if progress_callback:
            try:
                progress_callback(msg, pct)
            except Exception:
                pass

    _progress("Building logging intervals...", 0.05)
    geo_store = data_coordinator.geological_store if data_coordinator else None
    logging_df, logging_stats = _build_logging_interval_dataframe(
        data_coordinator,
        logger_values=None,
        date_from=None,
        date_to=None,
    )
    _log_dataframe_overview("Logging interval dataframe", logging_df)

    _progress("Merging assay and geology...", 0.15)
    logging_resolver = ColumnResolver(logging_df)
    logging_hole_col = logging_resolver.get("hole_id")
    hole_ids = (
        set(logging_df[logging_hole_col].astype(str).str.upper().str.strip().tolist())
        if logging_hole_col
        else None
    )
    merged_df, stats = build_merged_qaqc_dataframe(data_coordinator, hole_ids=hole_ids)
    _log_dataframe_overview("Assay interval dataframe", merged_df)

    resolver = ColumnResolver(merged_df)
    hole_col = resolver.get("hole_id")
    depth_from_col = resolver.get("depth_from")
    depth_to_col = resolver.get("depth_to")
    strat_col = resolver.get("strat")
    if not hole_col or not depth_to_col or not strat_col:
        raise ValueError(
            "Required columns (hole_id, depth_to, strat) are missing from merged data."
        )
    logger_col = resolve_logger_column(logging_df) or resolve_logger_column(merged_df)
    if not logger_col:
        raise ValueError("No logger column found in merged data.")

    # Fill empty logger values from loggedby_d or adjacent intervals
    if logger_col in logging_df.columns:
        logging_hole_col = logging_resolver.get("hole_id")
        logging_df = fill_empty_logger_values(logging_df, logger_col, logging_hole_col)
    if logger_col in merged_df.columns:
        merged_df = fill_empty_logger_values(merged_df, logger_col, hole_col)

    logging_from_col = logging_resolver.get("depth_from")
    logging_to_col = logging_resolver.get("depth_to")

    _progress("Merging logger and metrics...", 0.35)
    if logger_col not in merged_df.columns:
        logger_source = _find_logger_source(geo_store)
        if logger_source and depth_from_col:
            source_name, logger_df, logger_hole_col, logger_from_col, logger_to_col = logger_source
            logger_df_col = resolve_logger_column(logger_df)
            if logger_df_col:
                merged_df = _merge_logger_by_overlap(
                    merged_df=merged_df,
                    logger_df=logger_df,
                    merged_hole_col=hole_col,
                    merged_from_col=depth_from_col,
                    merged_to_col=depth_to_col,
                    logger_hole_col=logger_hole_col,
                    logger_from_col=logger_from_col,
                    logger_to_col=logger_to_col,
                    hole_ids=hole_ids,
                )
    if data_coordinator.has_rc_metrics:
        metrics_df = data_coordinator.rc_metrics_store.get_metrics_dataframe()
        if metrics_df is not None and not metrics_df.empty:
            if depth_from_col:
                merged_df = _merge_rc_metrics_by_overlap(
                    merged_df=merged_df,
                    metrics_df=metrics_df,
                    hole_col=hole_col,
                    depth_from_col=depth_from_col,
                    depth_to_col=depth_to_col,
                    hole_ids=hole_ids,
                )
            if logging_from_col and logging_to_col and logging_hole_col:
                logging_df = _merge_rc_metrics_by_overlap(
                    merged_df=logging_df,
                    metrics_df=metrics_df,
                    hole_col=logging_hole_col,
                    depth_from_col=logging_from_col,
                    depth_to_col=logging_to_col,
                    hole_ids=hole_ids,
                )

    _progress("Adding drill dates...", 0.55)
    collar_df = get_collar_dataframe(data_coordinator)
    collar_df_full = collar_df.copy() if not collar_df.empty else collar_df
    drilldate_col = resolve_drilldate_column(collar_df) if not collar_df.empty else None
    if drilldate_col and not collar_df.empty:
        collar_resolver = ColumnResolver(collar_df)
        collar_hole_col = collar_resolver.get("hole_id")
        if collar_hole_col:
            collar_df = collar_df[[collar_hole_col, drilldate_col]].copy()
            collar_df = collar_df.rename(
                columns={collar_hole_col: hole_col, drilldate_col: "drilldate"}
            )
            merged_df = merged_df.merge(collar_df, on=hole_col, how="left")
            drilldate_col = "drilldate"
            if logging_hole_col and logging_hole_col in logging_df.columns:
                logging_df = logging_df.merge(
                    collar_df.rename(columns={hole_col: logging_hole_col}),
                    on=logging_hole_col,
                    how="left",
                )

    _progress("Computing outlier scores...", 0.75)
    chem_cols = resolve_chemistry_columns(merged_df)
    chem_actual_cols = list(chem_cols.values())
    outlier_scores = compute_hybrid_outlier_scores(
        merged_df,
        strat_col=strat_col,
        chem_cols=chem_actual_cols,
        min_group_size=10,
    )
    merged_df = merged_df.join(outlier_scores)

    logger_values = sorted(
        pd.concat(
            [
                merged_df[logger_col].dropna().astype(str),
                logging_df[logger_col].dropna().astype(str),
            ]
        )
        .unique()
        .tolist()
    )
    _progress("Ready.", 1.0)
    return {
        "merged_df": merged_df,
        "logging_df": logging_df,
        "collar_df_full": collar_df_full,
        "stats": stats,
        "logger_col": logger_col,
        "hole_col": hole_col,
        "depth_from_col": depth_from_col,
        "depth_to_col": depth_to_col,
        "strat_col": strat_col,
        "chem_actual_cols": chem_actual_cols,
        "drilldate_col": drilldate_col or "",
        "logger_values": logger_values,
    }


def generate_logger_reports(
    data_coordinator,
    output_dir: str,
    date_from: Optional[str],
    date_to: Optional[str],
    logger_values: Optional[List[str]],
    top_n: int,
    page_options: Dict[str, bool],
    include_images: bool = True,
    logo_path: Optional[str] = None,
    output_format: str = "PDF",
    prepped_data: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Generate per-logger PDF or HTML reports.
    If prepped_data is provided (from prepare_logging_review_data), only filtering and HTML generation run.
    Returns list of file paths created.
    """
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.perf_counter()
    geo_store = data_coordinator.geological_store if data_coordinator else None

    if prepped_data is not None:
        # Keep full team data before filtering for team statistics
        full_team_df = prepped_data["merged_df"].copy()

        merged_df = filter_dataframe_by_logger_and_date(
            df=prepped_data["merged_df"],
            logger_col=prepped_data["logger_col"],
            date_col=prepped_data["drilldate_col"] or "",
            logger_values=logger_values,
            date_from=date_from,
            date_to=date_to,
        )
        logging_df = filter_dataframe_by_logger_and_date(
            df=prepped_data["logging_df"],
            logger_col=prepped_data["logger_col"],
            date_col=prepped_data["drilldate_col"] or "",
            logger_values=logger_values,
            date_from=date_from,
            date_to=date_to,
        )
        collar_df_full = prepped_data["collar_df_full"]
        stats = prepped_data["stats"]
        logger_col = prepped_data["logger_col"]
        hole_col = prepped_data["hole_col"]
        depth_from_col = prepped_data["depth_from_col"]
        depth_to_col = prepped_data["depth_to_col"]
        strat_col = prepped_data["strat_col"]
        chem_actual_cols = prepped_data["chem_actual_cols"]
        drilldate_col = prepped_data.get("drilldate_col")
        if not logger_values:
            logger_values = prepped_data.get("logger_values") or []
        logger.info(
            "Filtering complete: assay=%d rows, logging=%d rows (team data=%d rows)",
            len(merged_df),
            len(logging_df),
            len(full_team_df),
        )
        logger.info("Report generation from cache in %.2fs", time.perf_counter() - start_time)
    else:
        logging_df, logging_stats = _build_logging_interval_dataframe(
            data_coordinator,
            logger_values=logger_values,
            date_from=date_from,
            date_to=date_to,
        )
        _log_dataframe_overview("Logging interval dataframe", logging_df)

        logging_resolver = ColumnResolver(logging_df)
        logging_hole_col = logging_resolver.get("hole_id")
        hole_ids = (
            set(logging_df[logging_hole_col].astype(str).str.upper().str.strip().tolist())
            if logging_hole_col
            else None
        )

        merged_df, stats = build_merged_qaqc_dataframe(data_coordinator, hole_ids=hole_ids)
        _log_dataframe_overview("Assay interval dataframe", merged_df)

        resolver = ColumnResolver(merged_df)
        hole_col = resolver.get("hole_id")
        depth_from_col = resolver.get("depth_from")
        depth_to_col = resolver.get("depth_to")
        strat_col = resolver.get("strat")

        if not hole_col or not depth_to_col or not strat_col:
            raise ValueError(
                "Required columns (hole_id, depth_to, strat) are missing from merged data."
            )

        logger_col = resolve_logger_column(logging_df) or resolve_logger_column(merged_df)

        if not logger_col:
            available_sources = geo_store.list_sources() if geo_store else []
            raise ValueError(
                "No logger column found in merged data. "
                f"Available sources: {available_sources}"
            )
        logger.info("Logger column resolved: %s", logger_col)
        logging_resolver = ColumnResolver(logging_df)
        logging_from_col = logging_resolver.get("depth_from")
        logging_to_col = logging_resolver.get("depth_to")

        # Fill empty logger values from loggedby_d or adjacent intervals
        logging_hole_col = logging_resolver.get("hole_id")
        if logger_col in logging_df.columns:
            logging_df = fill_empty_logger_values(logging_df, logger_col, logging_hole_col)
        if logger_col in merged_df.columns:
            merged_df = fill_empty_logger_values(merged_df, logger_col, hole_col)

        if logger_col not in merged_df.columns:
            logger_source = _find_logger_source(geo_store)
            if logger_source and depth_from_col:
                source_name, logger_df, logger_hole_col, logger_from_col, logger_to_col = logger_source
                logger_df_col = resolve_logger_column(logger_df)
                if logger_df_col:
                    logger.info(
                        "Merging logger column '%s' into assay dataframe from '%s'.",
                        logger_df_col,
                        source_name,
                    )
                    if logger_values:
                        logger_df = logger_df[
                            logger_df[logger_df_col].astype(str).isin([str(v) for v in logger_values])
                        ].copy()
                    merged_df = _merge_logger_by_overlap(
                        merged_df=merged_df,
                        logger_df=logger_df,
                        merged_hole_col=hole_col,
                        merged_from_col=depth_from_col,
                        merged_to_col=depth_to_col,
                        logger_hole_col=logger_hole_col,
                        logger_from_col=logger_from_col,
                        logger_to_col=logger_to_col,
                        hole_ids=hole_ids,
                    )
            else:
                logger.warning(
                    "Logger column '%s' missing from assay dataframe; filtering may fail.",
                    logger_col,
                )

        if data_coordinator.has_rc_metrics:
            metrics_df = data_coordinator.rc_metrics_store.get_metrics_dataframe()
            if metrics_df is not None and not metrics_df.empty:
                if depth_from_col:
                    merged_df = _merge_rc_metrics_by_overlap(
                        merged_df=merged_df,
                        metrics_df=metrics_df,
                        hole_col=hole_col,
                        depth_from_col=depth_from_col,
                        depth_to_col=depth_to_col,
                        hole_ids=hole_ids,
                    )
                else:
                    logger.warning("Depth_from column missing; skipping RC metrics overlap merge.")
                if logging_from_col and logging_to_col and logging_hole_col:
                    logging_df = _merge_rc_metrics_by_overlap(
                        merged_df=logging_df,
                        metrics_df=metrics_df,
                        hole_col=logging_hole_col,
                        depth_from_col=logging_from_col,
                        depth_to_col=logging_to_col,
                        hole_ids=hole_ids,
                    )

        collar_df = get_collar_dataframe(data_coordinator)
        collar_df_full = collar_df.copy() if not collar_df.empty else collar_df
        drilldate_col = resolve_drilldate_column(collar_df) if not collar_df.empty else None

        if drilldate_col and not collar_df.empty:
            collar_resolver = ColumnResolver(collar_df)
            collar_hole_col = collar_resolver.get("hole_id")
            if collar_hole_col:
                collar_df = collar_df[[collar_hole_col, drilldate_col]].copy()
                collar_df = collar_df.rename(
                    columns={collar_hole_col: hole_col, drilldate_col: "drilldate"}
                )
                merged_df = merged_df.merge(collar_df, on=hole_col, how="left")
                drilldate_col = "drilldate"
                if logging_hole_col and logging_hole_col in logging_df.columns:
                    logging_df = logging_df.merge(
                        collar_df.rename(columns={hole_col: logging_hole_col}),
                        on=logging_hole_col,
                        how="left",
                    )

        # Keep full team data before filtering for team statistics
        full_team_df = merged_df.copy()

        merged_df = filter_dataframe_by_logger_and_date(
            df=merged_df,
            logger_col=logger_col,
            date_col=drilldate_col or "",
            logger_values=logger_values,
            date_from=date_from,
            date_to=date_to,
        )
        logging_df = filter_dataframe_by_logger_and_date(
            df=logging_df,
            logger_col=logger_col,
            date_col=drilldate_col or "",
            logger_values=logger_values,
            date_from=date_from,
            date_to=date_to,
        )
        logger.info(
            "Filtering complete: assay=%d rows, logging=%d rows (team data=%d rows)",
            len(merged_df),
            len(logging_df),
            len(full_team_df),
        )
        logger.info("Report data prep finished in %.2fs", time.perf_counter() - start_time)

        chem_cols = resolve_chemistry_columns(merged_df)
        chem_actual_cols = list(chem_cols.values())
        outlier_scores = compute_hybrid_outlier_scores(
            merged_df,
            strat_col=strat_col,
            chem_cols=chem_actual_cols,
            min_group_size=10,
        )
        merged_df = merged_df.join(outlier_scores)

        if not logger_values:
            logger_values = sorted(
                pd.concat(
                    [
                        merged_df[logger_col].dropna().astype(str),
                        logging_df[logger_col].dropna().astype(str),
                    ]
                )
                .unique()
                .tolist()
            )

    if logo_path is None:
        default_logo = Path(__file__).resolve().parents[1] / "resources" / "full_logo.png"
        logo_path = str(default_logo) if default_logo.exists() else None

    if output_format and output_format.strip().upper() == "HTML":
        from processing.logging_review_html_report import (
            generate_logger_html_reports_from_prepped_data,
        )

        return generate_logger_html_reports_from_prepped_data(
            data_coordinator=data_coordinator,
            output_dir=output_dir,
            merged_df=merged_df,
            logging_df=logging_df,
            collar_df=collar_df_full,
            stats=stats,
            logger_col=logger_col,
            hole_col=hole_col,
            depth_from_col=depth_from_col,
            depth_to_col=depth_to_col,
            strat_col=strat_col,
            chem_actual_cols=chem_actual_cols,
            logger_values=logger_values,
            date_from=date_from,
            date_to=date_to,
            top_n=top_n,
            page_options=page_options,
            include_images=include_images,
            logo_path=logo_path,
            full_team_df=full_team_df,
        )

    output_files = []
    for logger_value in logger_values:
        assay_logger_df = merged_df[merged_df[logger_col].astype(str) == str(logger_value)].copy()
        logging_logger_df = logging_df[logging_df[logger_col].astype(str) == str(logger_value)].copy()
        if assay_logger_df.empty and logging_logger_df.empty:
            continue

        filename = f"RC_Logging_Review_{logger_value}.pdf"
        output_path = os.path.join(output_dir, filename)
        c = canvas.Canvas(output_path, pagesize=letter)

        if page_options.get("cover", True):
            if logo_path and os.path.exists(logo_path):
                c.drawImage(logo_path, 420, 690, width=150, height=60, preserveAspectRatio=True)
            c.setFont("Helvetica-Bold", 18)
            c.drawString(60, 720, "RC Logging Review Report")
            c.setFont("Helvetica", 12)
            c.drawString(60, 690, f"Logger: {logger_value}")
            c.drawString(60, 675, f"Generated: {datetime.now().strftime('%Y-%m-%d')}")
            if date_from or date_to:
                c.drawString(60, 660, f"Date range: {date_from or '...'} to {date_to or '...'}")
            c.showPage()

        if page_options.get("summary_stats", True):
            c.setFont("Helvetica-Bold", 14)
            c.drawString(60, 720, "Summary Statistics")
            c.setFont("Helvetica", 11)
            c.drawString(60, 695, f"Assay intervals: {len(assay_logger_df):,}")
            c.drawString(60, 680, f"Logging intervals: {len(logging_logger_df):,}")
            c.drawString(60, 665, f"Unique holes: {assay_logger_df[hole_col].nunique():,}")
            c.drawString(60, 650, f"Strat codes: {assay_logger_df[strat_col].nunique():,}")
            c.drawString(60, 635, f"Merge match rate: {stats.get('match_rate_pct', 'n/a')}")
            c.showPage()

        if page_options.get("comment_stats", True):
            c.setFont("Helvetica-Bold", 14)
            c.drawString(60, 720, "Comment Statistics")
            comment_cols = [cname for cname in assay_logger_df.columns if "comment" in cname.lower()]
            total_comments = 0
            if comment_cols:
                total_comments = (
                    assay_logger_df[comment_cols].astype(str).replace("nan", "").ne("").sum().sum()
                )
            c.setFont("Helvetica", 11)
            c.drawString(60, 695, f"Comment columns: {', '.join(comment_cols) if comment_cols else 'None'}")
            c.drawString(60, 680, f"Non-empty comment fields: {int(total_comments):,}")
            c.showPage()

        if page_options.get("fines_accuracy", True):
            c.setFont("Helvetica-Bold", 14)
            c.drawString(60, 720, "Fines Accuracy")
            c.setFont("Helvetica", 11)
            fines_summary = _summarize_fines_accuracy(assay_logger_df, chem_actual_cols)
            y = 695
            for line in fines_summary:
                c.drawString(60, y, line)
                y -= 15
            c.showPage()

        if page_options.get("grouping_accuracy", True):
            c.setFont("Helvetica-Bold", 14)
            c.drawString(60, 720, "Grouping Accuracy")
            c.setFont("Helvetica", 11)
            log_from_col, log_to_col = _resolve_logging_interval_columns(assay_logger_df)
            group_cols = [hole_col, logger_col, strat_col]
            if log_from_col and log_to_col:
                group_cols.extend([log_from_col, log_to_col])
            grouping_summary = _summarize_grouping_accuracy_by_interval(
                assay_logger_df, group_cols, chem_actual_cols
            )
            y = 695
            for line in grouping_summary:
                c.drawString(60, y, line)
                y -= 15
            c.showPage()

        if page_options.get("outliers", True):
            top_outliers = assay_logger_df.sort_values("outlier_score", ascending=False).head(top_n)
            c.setFont("Helvetica-Bold", 12)
            c.drawString(60, 720, f"Top {len(top_outliers)} Outliers")
            c.setFont("Helvetica", 9)
            y = 700
            for _, row in top_outliers.iterrows():
                line = (
                    f"{row.get(hole_col)} @ {row.get(depth_to_col)}m | "
                    f"{row.get(strat_col)} | score {row.get('outlier_score'):.2f}"
                )
                c.drawString(60, y, line)
                y -= 12
                if y < 60:
                    c.showPage()
                    c.setFont("Helvetica", 9)
                    y = 720
            c.showPage()

            for _, row in top_outliers.iterrows():
                c.setFont("Helvetica-Bold", 12)
                c.drawString(60, 720, f"Outlier: {row.get(hole_col)} @ {row.get(depth_to_col)}m")
                c.setFont("Helvetica", 10)
                c.drawString(60, 700, f"Strat: {row.get(strat_col)}")
                c.drawString(60, 685, f"Score: {row.get('outlier_score'):.2f}")
                c.drawString(60, 670, f"Reasons: {row.get('outlier_reason')}")

                elements = str(row.get("outlier_elements", "")).split(",")
                elements = [e.strip() for e in elements if e.strip()]
                y_positions = [460, 320, 180]
                for idx, element in enumerate(elements[:3]):
                    if element in assay_logger_df.columns:
                        img = _build_boxplot_image(assay_logger_df[element], row.get(element))
                        c.setFont("Helvetica", 9)
                        c.drawString(60, y_positions[idx] + 130, f"{element} distribution")
                        c.drawImage(img, 60, y_positions[idx], width=180, height=120)

                if include_images:
                    try:
                        key = ImageKey(str(row[hole_col]), float(row[depth_to_col]))
                        img_path = data_coordinator.get_image_path(key)
                        if img_path and os.path.exists(img_path):
                            c.drawImage(img_path, 260, 360, width=300, height=200, preserveAspectRatio=True)
                    except Exception:
                        pass

                c.showPage()

        c.save()
        output_files.append(output_path)

    return output_files


def _summarize_fines_accuracy(df: pd.DataFrame, chem_cols: List[str]) -> List[str]:
    required_cols = {
        "fe": None,
        "sio2": None,
        "al2o3": None,
        "total_gangue_pct": "total_gangue_pct",
    }
    resolver = ColumnResolver(df)
    required_cols["fe"] = resolver.get("fe_pct")
    required_cols["sio2"] = resolver.get("sio2_pct")
    required_cols["al2o3"] = resolver.get("al2o3_pct")
    if not all([required_cols["fe"], required_cols["sio2"], required_cols["al2o3"], required_cols["total_gangue_pct"]]):
        return ["Missing required columns for fines accuracy summary."]

    fe_col = required_cols["fe"]
    sio2_col = required_cols["sio2"]
    al2o3_col = required_cols["al2o3"]
    gangue_col = required_cols["total_gangue_pct"]

    flags = []
    for _, row in df.iterrows():
        if pd.isna(row[fe_col]):
            flags.append("Pending Assays")
            continue
        if pd.isna(row[gangue_col]) or pd.isna(row[sio2_col]) or pd.isna(row[al2o3_col]):
            continue
        total_gangue = row[gangue_col]
        sio2 = row[sio2_col]
        al2o3 = row[al2o3_col]
        if total_gangue == 0 and sio2 > 5 and al2o3 < 5:
            flags.append("Friable Silica")
        elif total_gangue == 0 and sio2 > 5 and al2o3 > 5:
            flags.append("Friable Silica and Clays" if sio2 > al2o3 else "Clays / Shales")
        elif total_gangue > 15 and sio2 <= 10 and al2o3 <= 10:
            flags.append("Mineralisation Logged as Gangue")

    if not flags:
        return ["No fines accuracy issues detected in current selection."]

    counts = pd.Series(flags).value_counts()
    lines = ["Fines accuracy flags:"]
    for label, count in counts.items():
        lines.append(f"- {label}: {int(count)}")
    return lines


def _summarize_grouping_accuracy(df: pd.DataFrame, strat_col: str, chem_cols: List[str]) -> List[str]:
    if strat_col not in df.columns:
        return ["Missing strat column for grouping accuracy summary."]

    resolver = ColumnResolver(df)
    fe_col = resolver.get("fe_pct")
    sio2_col = resolver.get("sio2_pct")
    al2o3_col = resolver.get("al2o3_pct")
    element_cols = [c for c in [fe_col, sio2_col, al2o3_col] if c]
    if not element_cols:
        return ["Missing Fe/SiO2/Al2O3 columns for grouping accuracy summary."]

    lines = ["CV > 100% by Strat (proxy for grouping inconsistency):"]
    for element in element_cols:
        cv_exceeds = 0
        total_groups = 0
        for _, group in df.groupby(strat_col):
            values = group[element].dropna()
            if len(values) < 2:
                continue
            total_groups += 1
            mean = values.mean()
            if mean == 0:
                continue
            cv = values.std(ddof=1) / abs(mean) * 100
            if cv > 100:
                cv_exceeds += 1
        lines.append(f"- {element}: {cv_exceeds} / {total_groups} strat groups")
    return lines

