"""Data preparation for logging review report (merge, filter, collar, prep)."""
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from processing.DataManager.column_aliases import ColumnResolver
from processing.DataManager.interval_merger import IntervalMerger, merge_intervals_by_range

from .columns import LOGGER_COLUMN_PRIORITY, resolve_chemistry_columns, resolve_drilldate_column, resolve_logger_column
from .outliers import compute_hybrid_outlier_scores

logger = logging.getLogger(__name__)


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
        mask = pd.Series(True, index=filtered.index)
        if date_from:
            start = pd.to_datetime(date_from, errors="coerce")
            mask &= (dates >= start)
        if date_to:
            end = pd.to_datetime(date_to, errors="coerce")
            mask &= (dates <= end)
        filtered = filtered.loc[mask]

    return filtered


def _dataframe_has_collar_columns(df: pd.DataFrame) -> bool:
    """Return True if df has hole-id and east/north-like columns (for map)."""
    if df is None or df.empty or not hasattr(df, "columns"):
        return False
    resolver = ColumnResolver(df)
    hole_col = resolver.get("hole_id")
    cols_lower = {c.lower(): c for c in df.columns}
    east_aliases = ["easting", "east", "utm_e", "e", "x", "grid_e", "grid_easting"]
    north_aliases = ["northing", "north", "utm_n", "n", "y", "grid_n", "grid_northing"]
    has_east = any(a in cols_lower for a in east_aliases)
    has_north = any(a in cols_lower for a in north_aliases)
    return bool(hole_col and has_east and has_north)


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
    # Fallback: use any source that has collar-like columns (hole id + east/north)
    for source_name in geo_store.list_sources():
        src = geo_store.get_source(source_name)
        if src and src.df is not None and not src.df.empty and _dataframe_has_collar_columns(src.df):
            return src.df.copy()
    return pd.DataFrame()


def get_logger_list_and_date_options(data_coordinator) -> Dict[str, Any]:
    """
    Lightweight options for the logging review dialog: logger list and date range only.
    Does not build merged or logging dataframes. Use for dialog open; run
    prepare_logging_review_data when the user clicks Generate.
    """
    result = {
        "logger_values": [],
        "drilldate_col": "",
        "date_from": "",
        "date_to": "",
        "logger_col": "",
        "hole_col": "",
    }
    if not data_coordinator or not data_coordinator.is_initialized:
        return result
    geo_store = data_coordinator.geological_store
    if not geo_store:
        return result

    # Logger list from first source that has a logger column
    logger_source = _find_logger_source(geo_store)
    if logger_source:
        _src_name, logger_df, hole_col, _from_col, _to_col = logger_source
        logger_col = resolve_logger_column(logger_df)
        if logger_col:
            result["logger_col"] = logger_col
            result["hole_col"] = hole_col
            result["logger_values"] = sorted(
                logger_df[logger_col].dropna().astype(str).str.strip().unique().tolist()
            )
            result["logger_values"] = [v for v in result["logger_values"] if v]

    # Date range from collar
    collar_df = get_collar_dataframe(data_coordinator)
    if not collar_df.empty:
        drilldate_col = resolve_drilldate_column(collar_df)
        if drilldate_col:
            result["drilldate_col"] = drilldate_col
            dates = pd.to_datetime(collar_df[drilldate_col], errors="coerce").dropna()
            if not dates.empty:
                result["date_from"] = dates.min().date().isoformat()
                result["date_to"] = dates.max().date().isoformat()

    return result


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
    resolver = ColumnResolver(df)
    return resolver.get("depth_from"), resolver.get("depth_to")


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
