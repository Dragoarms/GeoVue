import base64
import html
import io
import json
import logging
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/headless use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

try:
    import pyproj
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False

from processing.DataManager.column_aliases import ColumnResolver
from processing.DataManager.keys import ImageKey
from processing.logging_review_report import (
    _build_logging_interval_dataframe,
    _log_dataframe_overview,
    _merge_logger_by_overlap,
    _merge_rc_metrics_by_overlap,
    _resolve_logging_interval_columns,
    _summarize_fines_accuracy,
    _summarize_grouping_accuracy_by_interval,
    build_merged_qaqc_dataframe,
    compute_hybrid_outlier_scores,
    filter_dataframe_by_logger_and_date,
    get_collar_dataframe,
    predict_most_likely_strat,
    resolve_chemistry_columns,
    resolve_drilldate_column,
    resolve_logger_column,
)

logger = logging.getLogger(__name__)


PROJECT_CODE_CANDIDATES = [
    "project_code",
    "projectcode",
    "project",
    "project_d",
    "projcode",
    "proj",
    "tenement",
    "prospect",
    "prospect_d",
]

EASTING_CANDIDATES = [
    "easting",
    "east",
    "utm_e",
    "e",
    "x",
    "grid_e",
    "grid_easting",
]

NORTHING_CANDIDATES = [
    "northing",
    "north",
    "utm_n",
    "n",
    "y",
    "grid_n",
    "grid_northing",
]


def generate_logger_html_reports_from_prepped_data(
    data_coordinator,
    output_dir: str,
    merged_df: pd.DataFrame,
    logging_df: pd.DataFrame,
    collar_df: pd.DataFrame,
    stats: Dict[str, Any],
    logger_col: str,
    hole_col: str,
    depth_from_col: Optional[str],
    depth_to_col: str,
    strat_col: str,
    chem_actual_cols: List[str],
    logger_values: List[str],
    date_from: Optional[str],
    date_to: Optional[str],
    top_n: int,
    page_options: Dict[str, bool],
    include_images: bool = True,
    logo_path: Optional[str] = None,
    full_team_df: Optional[pd.DataFrame] = None,
    skip_csv_export: bool = False,
) -> List[str]:
    """
    Generate per-logger HTML reports from already-prepped data.
    No merges or heavy data prep are done here; use when the caller has
    already run the full prep in generate_logger_reports().
    If skip_csv_export is True, do not write CSVs to output_dir/_datasets (e.g. when
    regenerating from existing _datasets in a read-only or preview flow).
    """
    os.makedirs(output_dir, exist_ok=True)
    if logo_path is None:
        default_logo = Path(__file__).resolve().parents[1] / "resources" / "full_logo.png"
        logo_path = str(default_logo) if default_logo.exists() else None

    if not skip_csv_export:
        # Export CSV datasets for debugging
        datasets_dir = os.path.join(output_dir, "_datasets")
        os.makedirs(datasets_dir, exist_ok=True)
        logging_csv_path = os.path.join(datasets_dir, "01_logging_intervals.csv")
        logging_df.to_csv(logging_csv_path, index=False)
        logger.info(f"Exported logging intervals: {logging_csv_path} ({len(logging_df):,} rows)")

        merged_csv_path = os.path.join(datasets_dir, "02_merged_assay_intervals.csv")
        merged_df.to_csv(merged_csv_path, index=False)
        logger.info(f"Exported merged assay intervals: {merged_csv_path} ({len(merged_df):,} rows)")

        if full_team_df is not None:
            team_csv_path = os.path.join(datasets_dir, "03_full_team_data.csv")
            full_team_df.to_csv(team_csv_path, index=False)
            logger.info(f"Exported full team data: {team_csv_path} ({len(full_team_df):,} rows)")

    output_files = []
    for logger_value in logger_values:
        assay_logger_df = merged_df[merged_df[logger_col].astype(str) == str(logger_value)].copy()
        logging_logger_df = logging_df[logging_df[logger_col].astype(str) == str(logger_value)].copy()
        if assay_logger_df.empty and logging_logger_df.empty:
            continue

        filename = f"RC_Logging_Review_{logger_value}.html"
        output_path = os.path.join(output_dir, filename)

        html_report = _build_html_report(
            data_coordinator=data_coordinator,
            logger_value=str(logger_value),
            assay_logger_df=assay_logger_df,
            logging_logger_df=logging_logger_df,
            merged_df=merged_df,
            logging_df=logging_df,
            collar_df=collar_df,
            logger_col=logger_col,
            hole_col=hole_col,
            depth_from_col=depth_from_col,
            depth_to_col=depth_to_col,
            strat_col=strat_col,
            chem_actual_cols=chem_actual_cols,
            page_options=page_options,
            date_from=date_from,
            date_to=date_to,
            top_n=top_n,
            include_images=include_images,
            logo_path=logo_path,
            stats=stats,
            output_dir=output_dir,
            full_team_df=full_team_df,
        )

        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(html_report)

        output_files.append(output_path)

    logger.info("HTML report generation complete: %d file(s)", len(output_files))
    return output_files


def generate_logger_html_reports(
    data_coordinator,
    output_dir: str,
    date_from: Optional[str],
    date_to: Optional[str],
    logger_values: Optional[List[str]],
    top_n: int,
    page_options: Dict[str, bool],
    include_images: bool = True,
    logo_path: Optional[str] = None,
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    start_time = datetime.now()
    geo_store = data_coordinator.geological_store if data_coordinator else None

    logging_df, _ = _build_logging_interval_dataframe(
        data_coordinator,
        logger_values=logger_values,
        date_from=date_from,
        date_to=date_to,
    )
    _log_dataframe_overview("Logging interval dataframe", logging_df)

    # Export logging datasource CSV for debugging
    datasets_dir = os.path.join(output_dir, "_datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    logging_csv_path = os.path.join(datasets_dir, "01_logging_intervals_raw.csv")
    logging_df.to_csv(logging_csv_path, index=False)
    logger.info(f"Exported logging intervals datasource: {logging_csv_path} ({len(logging_df):,} rows, {len(logging_df.columns)} columns)")

    logging_resolver = ColumnResolver(logging_df)
    logging_hole_col = logging_resolver.get("hole_id")
    hole_ids = (
        set(logging_df[logging_hole_col].astype(str).str.upper().str.strip().tolist())
        if logging_hole_col
        else None
    )

    merged_df, stats = build_merged_qaqc_dataframe(data_coordinator, hole_ids=hole_ids)
    _log_dataframe_overview("Assay interval dataframe", merged_df)

    # Export initial merged/assay datasource CSV for debugging
    assay_csv_path = os.path.join(datasets_dir, "02_assay_intervals_raw.csv")
    merged_df.to_csv(assay_csv_path, index=False)
    logger.info(f"Exported assay intervals datasource: {assay_csv_path} ({len(merged_df):,} rows, {len(merged_df.columns)} columns)")

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
            log_from_col, log_to_col = _resolve_logging_interval_columns(logging_df)
            if log_from_col and log_to_col and logging_hole_col:
                logging_df = _merge_rc_metrics_by_overlap(
                    merged_df=logging_df,
                    metrics_df=metrics_df,
                    hole_col=logging_hole_col,
                    depth_from_col=log_from_col,
                    depth_to_col=log_to_col,
                    hole_ids=hole_ids,
                )

    collar_df = get_collar_dataframe(data_coordinator)
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

    chem_cols = resolve_chemistry_columns(merged_df)
    chem_actual_cols = list(chem_cols.values())
    outlier_scores = compute_hybrid_outlier_scores(
        merged_df,
        strat_col=strat_col,
        chem_cols=chem_actual_cols,
        min_group_size=10,
    )
    merged_df = merged_df.join(outlier_scores)

    # Export final processed dataframe (after all merges, filtering, and outlier scoring)
    final_csv_path = os.path.join(datasets_dir, "03_merged_processed_final.csv")
    merged_df.to_csv(final_csv_path, index=False)
    logger.info(
        f"Exported final processed dataframe: {final_csv_path} "
        f"({len(merged_df):,} rows, {len(merged_df.columns)} columns, "
        f"filtered by logger={logger_values}, dates={date_from} to {date_to})"
    )

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

    output_files = []
    for logger_value in logger_values:
        assay_logger_df = merged_df[merged_df[logger_col].astype(str) == str(logger_value)].copy()
        logging_logger_df = logging_df[logging_df[logger_col].astype(str) == str(logger_value)].copy()
        if assay_logger_df.empty and logging_logger_df.empty:
            continue

        filename = f"RC_Logging_Review_{logger_value}.html"
        output_path = os.path.join(output_dir, filename)

        html_report = _build_html_report(
            data_coordinator=data_coordinator,
            logger_value=str(logger_value),
            assay_logger_df=assay_logger_df,
            logging_logger_df=logging_logger_df,
            merged_df=merged_df,
            logging_df=logging_df,
            collar_df=get_collar_dataframe(data_coordinator),
            logger_col=logger_col,
            hole_col=hole_col,
            depth_from_col=depth_from_col,
            depth_to_col=depth_to_col,
            strat_col=strat_col,
            chem_actual_cols=chem_actual_cols,
            page_options=page_options,
            date_from=date_from,
            date_to=date_to,
            top_n=top_n,
            include_images=include_images,
            logo_path=logo_path,
            stats=stats,
        )

        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(html_report)

        output_files.append(output_path)

    elapsed = datetime.now() - start_time
    logger.info("HTML report generation complete in %s", elapsed)
    return output_files


def _find_logger_source(geo_store) -> Optional[Tuple[str, pd.DataFrame, str, str, str]]:
    if not geo_store:
        return None
    for source_name in geo_store.list_sources():
        src = geo_store.get_source(source_name)
        if not src or src.df is None or src.df.empty:
            continue
        df = src.df
        resolver = ColumnResolver(df)
        hole_col = resolver.get("hole_id")
        depth_from_col = resolver.get("depth_from")
        depth_to_col = resolver.get("depth_to")
        if hole_col and depth_from_col and depth_to_col and resolve_logger_column(df):
            return source_name, df.copy(), hole_col, depth_from_col, depth_to_col
    return None


def _resolve_project_code_column(df: pd.DataFrame) -> Optional[str]:
    columns_lower = {c.lower(): c for c in df.columns}
    for candidate in PROJECT_CODE_CANDIDATES:
        actual = columns_lower.get(candidate.lower())
        if actual:
            return actual
    return None


def _resolve_coordinate_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    columns_lower = {c.lower(): c for c in df.columns}
    east = next((columns_lower[c] for c in EASTING_CANDIDATES if c in columns_lower), None)
    north = next((columns_lower[c] for c in NORTHING_CANDIDATES if c in columns_lower), None)
    return east, north


def _safe_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return float(value)
    except Exception:
        return None


def _encode_image_base64(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    try:
        ext = Path(path).suffix.lower().lstrip(".")
        if ext == "jpg":
            ext = "jpeg"
        with open(path, "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("ascii")
        return f"data:image/{ext};base64,{encoded}"
    except Exception:
        logger.exception("Failed to encode image: %s", path)
        return None


def _extract_comment_wordcloud(df: pd.DataFrame) -> Dict[str, int]:
    """
    Extract word frequencies from comment columns for word cloud generation.

    Returns a dictionary of {word: count} for use with WordCloud library.
    """
    comment_cols = [c for c in df.columns if "comment" in c.lower()]
    if not comment_cols:
        return {}
    parts = []
    for _, row in df[comment_cols].iterrows():
        for v in row:
            if pd.isna(v):
                continue
            s = str(v).strip()
            if s and s.lower() not in ("nan", "none"):
                parts.append(s)
    text = " ".join(parts)
    if not text:
        return {}

    stopwords = {
        "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
        "les", "des", "une", "pour", "avec", "dans", "sur", "ces", "est", "sont",
        "aux", "par", "pas", "plus", "moins", "have", "has", "had", "mais", "ou",
        "not", "non", "oui", "very", "tres", "trop", "sur", "into", "just",
    }
    words = []
    for raw in text.replace("/", " ").replace("-", " ").split():
        cleaned = "".join(ch for ch in raw if ch.isalnum()).lower()
        if len(cleaned) < 3 or cleaned in stopwords:
            continue
        words.append(cleaned)
    if not words:
        return {}
    # Return all word frequencies as a dict (no limit - WordCloud handles display)
    counts = pd.Series(words).value_counts()
    return dict(counts)


def _compute_comment_stats(df: pd.DataFrame) -> Dict[str, Any]:
    comment_cols = [c for c in df.columns if "comment" in c.lower()]
    total_comments = 0
    if comment_cols:
        total_comments = (
            df[comment_cols].astype(str).replace("nan", "").ne("").sum().sum()
        )
    return {
        "comment_columns": comment_cols,
        "non_empty_comment_fields": int(total_comments),
        "total_comment_fields": int(len(df) * len(comment_cols)) if comment_cols else 0,
    }


def _flag_fines_issue(row: pd.Series, resolver: ColumnResolver) -> Optional[str]:
    fe_col = resolver.get("fe_pct")
    sio2_col = resolver.get("sio2_pct")
    al2o3_col = resolver.get("al2o3_pct")
    gangue_col = resolver.get("total_gangue_pct")
    if not fe_col or not sio2_col or not al2o3_col or not gangue_col:
        return None
    fe = row.get(fe_col)
    if pd.isna(fe):
        return "Pending Assays"
    total_gangue = row.get(gangue_col)
    sio2 = row.get(sio2_col)
    al2o3 = row.get(al2o3_col)
    if pd.isna(total_gangue) or pd.isna(sio2) or pd.isna(al2o3):
        return None
    if total_gangue == 0 and sio2 > 5 and al2o3 < 5:
        return "Friable Silica"
    if total_gangue == 0 and sio2 > 5 and al2o3 > 5:
        return "Friable Silica and Clays" if sio2 > al2o3 else "Clays / Shales"
    if total_gangue > 15 and sio2 <= 10 and al2o3 <= 10:
        return "Mineralisation Logged as Gangue"
    return None


def _has_magnetite_logged(row: pd.Series, resolver: ColumnResolver) -> bool:
    """True if row has MT, MBH, or MBM in dominant/mineralisation columns."""
    magnetite_codes = {"MT", "MBH", "MBM"}
    for std in ("min_80_pct", "min_50_pct", "mineralisation"):
        col = resolver.get(std)
        if not col or col not in row.index:
            continue
        val = str(row.get(col, "")).upper().strip()
        if val in magnetite_codes:
            return True
    return False


def _flag_magnetite_issue(row: pd.Series, resolver: ColumnResolver) -> Optional[str]:
    """Flag: magnetite (MT/MBH/MBM) logged but LOI1000 not negative."""
    if not _has_magnetite_logged(row, resolver):
        return None
    loi_col = resolver.get("loi_1000_pct") or resolver.get("loi_pct")
    if not loi_col or loi_col not in row.index:
        return None
    loi = row.get(loi_col)
    if pd.isna(loi):
        return "Magnetite logged (MT/MBH/MBM) but LOI1000 missing"
    if float(loi) >= 0:
        return "Magnetite logged (MT/MBH/MBM) but LOI1000 not negative"
    return None


def _flag_goethite_issue(row: pd.Series, resolver: ColumnResolver) -> Optional[str]:
    """Flag: high zonation Hy (goethite) but LOI (e.g. 425) low or missing."""
    hy_col = resolver.get("zonation_hy_pct")
    if not hy_col or hy_col not in row.index:
        return None
    hy = row.get(hy_col)
    if pd.isna(hy) or float(hy) < 20:
        return None
    loi_col = resolver.get("loi_425_pct") or resolver.get("loi_pct")
    if not loi_col or loi_col not in row.index:
        return None
    loi = row.get(loi_col)
    if pd.isna(loi):
        return "High goethite (Hy) but LOI missing"
    if float(loi) < 2:
        return "High goethite (Hy) but LOI low"
    return None


def _flag_carbonate_issue(row: pd.Series, resolver: ColumnResolver) -> Optional[str]:
    """Flag: logged carbonate gangue % inconsistent with assay proxy (LOI/CaO)."""
    carb_col = resolver.get("carbonate_gangue_pct")
    if not carb_col or carb_col not in row.index:
        return None
    carb = row.get(carb_col)
    if pd.isna(carb):
        return None
    carb_val = float(carb)
    cao_col = resolver.get("cao_pct")
    loi_col = resolver.get("loi_pct")
    assay_proxy = None
    if cao_col and cao_col in row.index and not pd.isna(row.get(cao_col)):
        assay_proxy = float(row.get(cao_col))
    if assay_proxy is None and loi_col and loi_col in row.index and not pd.isna(row.get(loi_col)):
        assay_proxy = float(row.get(loi_col))
    if assay_proxy is None:
        return None
    if carb_val > 15 and assay_proxy < 2:
        return "Logged carbonate % high but assay (CaO/LOI) low"
    if carb_val < 2 and assay_proxy > 10:
        return "Logged carbonate % low but assay (CaO/LOI) high"
    return None


def _build_grouping_issue_intervals(
    df: pd.DataFrame,
    group_cols: List[str],
    chem_cols: List[str],
    hole_col: str,
    depth_from_col: Optional[str],
    depth_to_col: str,
) -> List[Dict[str, Any]]:
    if not all(col in df.columns for col in group_cols):
        return []
    available = [c for c in chem_cols if c in df.columns]
    if not available:
        return []
    intervals = []
    for _, group in df.groupby(group_cols):
        for element in available:
            values = group[element].dropna()
            if len(values) < 2:
                continue
            mean = values.mean()
            if mean == 0:
                continue
            cv = values.std(ddof=1) / abs(mean) * 100
            if cv > 100:
                row = group.iloc[0]
                intervals.append(
                    {
                        "hole_id": _safe_str(row.get(hole_col)),
                        "depth_from": _safe_float(row.get(depth_from_col)) if depth_from_col else None,
                        "depth_to": _safe_float(row.get(depth_to_col)),
                        "issue": f"High CV in {element} (>{100}%)",
                    }
                )
                break
    return intervals


# Default UTM zone for collar reprojection (e.g. 33N = Gabon/Belinga). Override via config if needed.
DEFAULT_UTM_ZONE = 33
DEFAULT_UTM_NORTH = True


def _build_map_points(
    collar_df: pd.DataFrame,
    hole_col: str,
    logger_holes: set,
    all_holes: set,
    utm_zone: Optional[int] = None,
    utm_north: bool = True,
) -> Dict[str, Any]:
    if collar_df.empty:
        return {"points": [], "has_coords": False}
    resolver = ColumnResolver(collar_df)
    collar_hole_col = resolver.get("hole_id") or hole_col
    east_col, north_col = _resolve_coordinate_columns(collar_df)
    if not east_col or not north_col:
        return {"points": [], "has_coords": False}

    df = collar_df[[collar_hole_col, east_col, north_col]].copy()
    df = df.dropna(subset=[east_col, north_col])
    if df.empty:
        return {"points": [], "has_coords": False}

    df = df[df[collar_hole_col].astype(str).str.upper().str.strip().isin(all_holes)]
    if df.empty:
        # No overlap between collar holes and report holes: show all collar points so map still displays
        df = collar_df[[collar_hole_col, east_col, north_col]].copy()
        df = df.dropna(subset=[east_col, north_col])
    if df.empty:
        return {"points": [], "has_coords": False}

    xs = df[east_col].astype(float)
    ys = df[north_col].astype(float)
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    # Heuristic: coords look like projected (e.g. UTM) if outside WGS84 or in typical UTM range
    outside_wgs84 = min_x < -180 or max_x > 180 or min_y < -90 or max_y > 90
    # Typical UTM: easting 10k–999k, northing 0–10M (or southern hemisphere negative)
    looks_like_utm = (10000 <= min_x <= 999000 and 0 <= min_y <= 10000000) or (
        10000 <= min_x <= 999000 and -10000000 <= max_y <= 0
    )
    looks_projected = outside_wgs84 or looks_like_utm

    # Reproject UTM (easting, northing) to WGS84 (lng, lat) when coords look projected
    if looks_projected and PYPROJ_AVAILABLE:

        def try_reproject(zone: int, north: bool):
            epsg_utm = 32600 + zone if north else 32700 + zone
            transformer = pyproj.Transformer.from_crs(
                f"EPSG:{epsg_utm}", "EPSG:4326", always_xy=True
            )
            rows_ll = []
            for _, row in df.iterrows():
                hole = str(row[collar_hole_col]).upper().strip()
                east_val = float(row[east_col])
                north_val = float(row[north_col])
                lng, lat = transformer.transform(east_val, north_val)
                rows_ll.append((hole, lng, lat))
            lngs = [r[1] for r in rows_ll]
            lats = [r[2] for r in rows_ll]
            min_lat, max_lat = min(lats), max(lats)
            min_lng, max_lng = min(lngs), max(lngs)
            span_lat = max(max_lat - min_lat, 1e-6)
            span_lng = max(max_lng - min_lng, 1e-6)
            points = []
            for hole, lng, lat in rows_ll:
                points.append(
                    {
                        "x": (lat - min_lat) / span_lat,
                        "y": (lng - min_lng) / span_lng,
                        "hole_id": hole,
                        "is_logger": hole in logger_holes,
                        "lat": lat,
                        "lng": lng,
                    }
                )
            return {"points": points, "has_coords": True, "bounds": [min_lat, min_lng, max_lat, max_lng], "warn_projected": False}

        # Infer hemisphere from northing: positive => north (326xx), negative => south (327xx)
        inferred_north = min_y >= 0 and max_y <= 10000000
        primary_zones = [utm_zone, DEFAULT_UTM_ZONE, 32, 31, 34] if utm_zone is None else [utm_zone, 32, 31, 34]
        primary_zones = [z for z in primary_zones if z is not None]
        primary_zones = list(dict.fromkeys(primary_zones))
        last_error = None
        # 1) Try primary zones with configured hemisphere, then inferred
        for zone in primary_zones:
            for north in (utm_north, inferred_north):
                try:
                    return try_reproject(zone, north)
                except Exception as e:
                    last_error = e
        # 2) Try all UTM zones 1-60 for inferred hemisphere, then the other
        for north in (inferred_north, not inferred_north):
            for zone in range(1, 61):
                try:
                    return try_reproject(zone, north)
                except Exception as e:
                    last_error = e
        if last_error is not None:
            logger.warning("UTM reprojection failed for all zones (%s), map may show projected coords warning", last_error)

    # No reprojection: pass through as-is (or already WGS84)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    points = []
    for _, row in df.iterrows():
        hole = str(row[collar_hole_col]).upper().strip()
        east_val = float(row[east_col])
        north_val = float(row[north_col])
        x = (east_val - min_x) / span_x
        y = (north_val - min_y) / span_y
        points.append(
            {
                "x": x,
                "y": y,
                "hole_id": hole,
                "is_logger": hole in logger_holes,
                "lat": north_val,
                "lng": east_val,
            }
        )
    bounds = [min_y, min_x, max_y, max_x]
    warn_projected = looks_projected
    return {"points": points, "has_coords": True, "bounds": bounds, "warn_projected": warn_projected}


def _compute_logger_median(
    df: pd.DataFrame,
    logger_col: str,
    metric_fn,
) -> Dict[str, float]:
    values = []
    for _, group in df.groupby(logger_col):
        try:
            value = metric_fn(group)
            if value is not None and not math.isnan(value):
                values.append(value)
        except Exception:
            continue
    if not values:
        return {"median": float("nan")}
    return {"median": float(np.median(values))}


def _within_median_band(value: Optional[float], median: Optional[float], tolerance: float = 0.2) -> Optional[bool]:
    if value is None or median is None:
        return None
    if median == 0:
        return None
    return abs(value - median) / abs(median) <= tolerance


def _calc_comment_coverage(df: pd.DataFrame) -> float:
    stats = _compute_comment_stats(df)
    total = stats["total_comment_fields"]
    if total == 0:
        return 0.0
    return stats["non_empty_comment_fields"] / total * 100


def _calc_mineral_assay_ratio(df: pd.DataFrame) -> float:
    resolver = ColumnResolver(df)
    mineral_col = resolver.get("mineralisation")
    chem_cols = resolve_chemistry_columns(df)
    chem_actual_cols = list(chem_cols.values())
    if not mineral_col or not chem_actual_cols:
        return 0.0
    mineralised = df[mineral_col].astype(str).replace("nan", "").ne("").sum()
    if mineralised == 0:
        return 0.0
    assays_available = df[chem_actual_cols[0]].notna().sum()
    return assays_available / mineralised * 100


def _calc_profile_coverage(df: pd.DataFrame, strat_col: str) -> float:
    if strat_col not in df.columns:
        return 0.0
    return df[strat_col].astype(str).replace("nan", "").ne("").mean() * 100


def _calc_fines_flag_rate(df: pd.DataFrame) -> float:
    resolver = ColumnResolver(df)
    if df.empty:
        return 0.0
    flags = 0
    for _, row in df.iterrows():
        if _flag_fines_issue(row, resolver):
            flags += 1
    return flags / len(df) * 100


def _calc_grouping_issue_rate(
    df: pd.DataFrame,
    group_cols: List[str],
    chem_cols: List[str],
    hole_col: str,
    depth_from_col: Optional[str],
    depth_to_col: str,
) -> float:
    intervals = _build_grouping_issue_intervals(
        df, group_cols, chem_cols, hole_col, depth_from_col, depth_to_col
    )
    if df.empty:
        return 0.0
    return len(intervals) / len(df) * 100


def _calc_outlier_rate(df: pd.DataFrame, top_n: int) -> float:
    if df.empty:
        return 0.0
    top_count = min(top_n, len(df))
    return top_count / len(df) * 100


def _intervals_overlap(
    a_from: Optional[float], a_to: Optional[float],
    b_from: Optional[float], b_to: Optional[float],
) -> bool:
    if a_from is None or a_to is None or b_from is None or b_to is None:
        return False
    return not (a_to <= b_from or b_to <= a_from)


def _compute_assay_received_outstanding(
    logging_df: pd.DataFrame,
    assay_df: pd.DataFrame,
    hole_col: str,
    depth_from_col: Optional[str],
    depth_to_col: str,
    log_from_col: Optional[str],
    log_to_col: Optional[str],
) -> Tuple[int, int]:
    """Assays received = assay interval count; outstanding = logging intervals with no overlapping assay."""
    assay_received = len(assay_df)
    if logging_df.empty or not log_from_col or not log_to_col:
        return assay_received, 0
    assay_ranges = []
    for _, row in assay_df.iterrows():
        f = _safe_float(row.get(depth_from_col)) if depth_from_col else None
        t = _safe_float(row.get(depth_to_col))
        if t is not None:
            assay_ranges.append((str(row.get(hole_col, "")).upper().strip(), f, t))
    outstanding = 0
    for _, row in logging_df.iterrows():
        hole = str(row.get(hole_col, "")).upper().strip()
        lf = _safe_float(row.get(log_from_col))
        lt = _safe_float(row.get(log_to_col))
        if lt is None:
            continue
        has_overlap = False
        for ah, af, at in assay_ranges:
            if ah == hole and _intervals_overlap(lf, lt, af, at):
                has_overlap = True
                break
        if not has_overlap:
            outstanding += 1
    return assay_received, outstanding


def _compute_avg_logging_interval(
    logging_df: pd.DataFrame,
    log_from_col: Optional[str],
    log_to_col: Optional[str],
) -> Optional[float]:
    if logging_df.empty or not log_to_col:
        return None
    from_vals = logging_df[log_from_col] if log_from_col and log_from_col in logging_df.columns else 0
    to_vals = logging_df[log_to_col]
    lengths = to_vals.astype(float) - (from_vals.astype(float) if log_from_col else 0)
    lengths = lengths[lengths > 0]
    if lengths.empty:
        return None
    return float(lengths.mean())


def _comment_stats_logging_intervals(logging_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comment statistics for logging intervals.

    Returns total_rows, rows_with_comment, rows_without_comment, comment_ratio_pct, avg_comment_length.
    """
    total = len(logging_df)
    if total == 0:
        return {"total_rows": 0, "rows_with_comment": 0, "rows_without_comment": 0, "comment_ratio_pct": 0.0, "avg_comment_length": 0.0}
    comment_cols = [c for c in logging_df.columns if "comment" in c.lower()]
    if not comment_cols:
        return {"total_rows": total, "rows_with_comment": 0, "rows_without_comment": total, "comment_ratio_pct": 0.0, "avg_comment_length": 0.0}

    # Check which rows have comments
    comment_df = logging_df[comment_cols].astype(str).replace("nan", "").replace("", np.nan)
    has_comment = comment_df.notna().any(axis=1)
    rows_with = int(has_comment.sum())
    rows_without = total - rows_with

    # Calculate average comment length
    avg_length = 0.0
    if rows_with > 0:
        all_comments = []
        for col in comment_cols:
            comments = comment_df[col].dropna()
            for c in comments:
                if c and str(c).strip():
                    all_comments.append(len(str(c).strip()))
        if all_comments:
            avg_length = sum(all_comments) / len(all_comments)

    return {
        "total_rows": total,
        "rows_with_comment": rows_with,
        "rows_without_comment": rows_without,
        "comment_ratio_pct": (rows_with / total * 100) if total else 0.0,
        "avg_comment_length": avg_length,
    }


def _add_mineralisation_accuracy_column(df: pd.DataFrame, resolver: ColumnResolver) -> pd.DataFrame:
    """Add Logging Accuracy: Match, Mismatch, Pending Assays. QAQC04 logic."""
    result = df.copy()
    fe_col = resolver.get("fe_pct")
    sio2_col = resolver.get("sio2_pct")
    al2o3_col = resolver.get("al2o3_pct")
    gangue_col = resolver.get("total_gangue_pct")
    for col in [fe_col, sio2_col, al2o3_col, gangue_col]:
        if col and col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    def determine_accuracy(row):
        if fe_col not in result.columns or pd.isna(row.get(fe_col)):
            return "Pending Assays"
        fe = row.get(fe_col)
        if pd.isna(fe):
            return "Pending Assays"
        sio2 = row.get(sio2_col) if sio2_col else np.nan
        al2o3 = row.get(al2o3_col) if al2o3_col else np.nan
        gangue = row.get(gangue_col) if gangue_col else np.nan
        is_mineralised_assay = (
            pd.notna(fe) and pd.notna(sio2) and pd.notna(al2o3)
            and float(fe) > 50 and float(sio2) < 10 and float(al2o3) < 5
        )
        is_mineralised_logging = pd.notna(gangue) and float(gangue) < 15
        if is_mineralised_assay == is_mineralised_logging:
            return "Match"
        return "Mismatch"

    result["Logging_Accuracy"] = result.apply(determine_accuracy, axis=1)
    return result


def _group_mineralisation_by_quarter(
    df: pd.DataFrame, date_col: Optional[str]
) -> pd.DataFrame:
    """Quarterly summary: Quarter, Match, Mismatch, Pending Assays."""
    if "Logging_Accuracy" not in df.columns or not date_col or date_col not in df.columns:
        return pd.DataFrame(columns=["Quarter", "Match", "Mismatch", "Pending Assays"])
    data = df[[date_col, "Logging_Accuracy"]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col])
    data["Quarter"] = pd.to_datetime(data[date_col]).dt.to_period("Q").dt.start_time
    summary = (
        data.groupby("Quarter")["Logging_Accuracy"]
        .value_counts()
        .unstack(fill_value=0)
    )
    for cat in ["Match", "Mismatch", "Pending Assays"]:
        if cat not in summary.columns:
            summary[cat] = 0
    summary = summary.reset_index()
    return summary


def _resolve_zonation_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Resolve BestProfileZonation_D / profilezonation, Total Gangue, De/Hy/Pr/Un % (including zonation_*_pct)."""
    out = {}
    cols_lower = {c.lower(): c for c in df.columns}
    for key, candidates in [
        ("zonation", ["bestprofilezonation_d", "bestprofilezonation", "profilezonation", "profile_zonation", "zonation"]),
        ("total_gangue", ["total gangue logged", "total_gangue_logged", "total_gangue_pct"]),
        ("de_pct", ["de % logged", "de_pct_logged", "de_logged", "zonation_de_pct"]),
        ("hy_pct", ["hy % logged", "hy_pct_logged", "hy_logged", "zonation_hy_pct"]),
        ("pr_pct", ["pr % logged", "pr_pct_logged", "pr_logged", "zonation_pr_pct"]),
        ("un_pct", ["un % logged", "un_pct_logged", "un_logged", "zonation_un_pct"]),
    ]:
        out[key] = next((cols_lower[c] for c in candidates if c in cols_lower), None)
    return out


def _derive_dominant_zonation_column(
    df: pd.DataFrame, zonation_cols: Dict[str, Optional[str]]
) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    """If no logged zonation column exists but zonation_*_pct columns do, add dominant zonation (Un/Pr/De/Hy) per row. Returns (df_with_column, updated_zonation_cols)."""
    if zonation_cols.get("zonation") is not None:
        return df, zonation_cols
    un_col = zonation_cols.get("un_pct")
    pr_col = zonation_cols.get("pr_pct")
    de_col = zonation_cols.get("de_pct")
    hy_col = zonation_cols.get("hy_pct")
    if not all(c and c in df.columns for c in (un_col, pr_col, de_col, hy_col)):
        return df, zonation_cols
    df_out = df.copy()
    un_vals = df_out[un_col].apply(lambda x: _safe_float(x) or 0)
    pr_vals = df_out[pr_col].apply(lambda x: _safe_float(x) or 0)
    de_vals = df_out[de_col].apply(lambda x: _safe_float(x) or 0)
    hy_vals = df_out[hy_col].apply(lambda x: _safe_float(x) or 0)
    # Dominant = argmax of Un, Pr, De, Hy; use "Un" if all zero
    def row_dominant(i):
        un, pr, de, hy = un_vals.iloc[i], pr_vals.iloc[i], de_vals.iloc[i], hy_vals.iloc[i]
        vals = {"Un": un, "Pr": pr, "De": de, "Hy": hy}
        best = max(vals, key=vals.get)
        return best if vals[best] > 0 else "Un"
    df_out["_dominant_zonation"] = [row_dominant(i) for i in range(len(df_out))]
    cols_updated = dict(zonation_cols)
    cols_updated["zonation"] = "_dominant_zonation"
    return df_out, cols_updated


def _check_profile_zonation_and_analyze(
    df: pd.DataFrame, zonation_cols: Dict[str, Optional[str]]
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, Dict[str, int]]]:
    """Returns correct_counts, mismatch_counts (by category), mismatch_attribution (category -> {should_be: count})."""
    zonation_categories = ["Un", "Le", "De", "Hy", "Pr"]
    correct_counts = {c: 0 for c in zonation_categories}
    mismatch_counts = {c: 0 for c in zonation_categories}
    mismatch_attribution = {c: {} for c in zonation_categories}
    z_col = zonation_cols.get("zonation")
    g_col = zonation_cols.get("total_gangue")
    de_col = zonation_cols.get("de_pct")
    hy_col = zonation_cols.get("hy_pct")
    pr_col = zonation_cols.get("pr_pct")
    if not z_col or z_col not in df.columns:
        return correct_counts, mismatch_counts, mismatch_attribution
    for _, row in df.iterrows():
        zonation = row.get(z_col)
        if pd.isna(zonation):
            continue
        zonation = str(zonation).strip()
        if zonation not in zonation_categories:
            continue
        total_gangue = _safe_float(row.get(g_col)) if g_col else None
        de_pct = _safe_float(row.get(de_col)) if de_col else 0
        hy_pct = _safe_float(row.get(hy_col)) if hy_col else 0
        pr_pct = _safe_float(row.get(pr_col)) if pr_col else 0
        rules = {
            "Un": (16, 100, None),
            "Le": (11, 15, None),
            "De": (0, 10, "De"),
            "Hy": (0, 10, "Hy"),
            "Pr": (0, 10, "Pr"),
        }
        min_g, max_g, dominant = rules.get(zonation, (0, 100, None))
        ok = True
        if total_gangue is not None and not (min_g <= total_gangue < max_g):
            ok = False
        if ok and dominant:
            minerals = {"De": de_pct or 0, "Hy": hy_pct or 0, "Pr": pr_pct or 0}
            actual_dom = max(minerals, key=minerals.get)
            if actual_dom != dominant:
                ok = False
        if ok:
            correct_counts[zonation] = correct_counts.get(zonation, 0) + 1
        else:
            mismatch_counts[zonation] = mismatch_counts.get(zonation, 0) + 1
            actual_dom = None
            if total_gangue is not None:
                if total_gangue > 15:
                    actual_dom = "Un"
                elif total_gangue > 10:
                    actual_dom = "Le"
                elif de_col or hy_col or pr_col:
                    minerals = {"De": de_pct or 0, "Hy": hy_pct or 0, "Pr": pr_pct or 0}
                    actual_dom = max(minerals, key=minerals.get)
            if actual_dom and actual_dom in zonation_categories:
                mismatch_attribution[zonation][actual_dom] = mismatch_attribution[zonation].get(actual_dom, 0) + 1
    return correct_counts, mismatch_counts, mismatch_attribution


def _collect_zonation_mismatch_rows(
    df: pd.DataFrame,
    zonation_cols: Dict[str, Optional[str]],
    hole_col: str,
    depth_from_col: Optional[str],
    depth_to_col: str,
) -> List[Dict[str, Any]]:
    """Collect row-level zonation mismatches for evidence table (errors only)."""
    zonation_categories = ["Un", "Le", "De", "Hy", "Pr"]
    z_col = zonation_cols.get("zonation")
    g_col = zonation_cols.get("total_gangue")
    de_col = zonation_cols.get("de_pct")
    hy_col = zonation_cols.get("hy_pct")
    pr_col = zonation_cols.get("pr_pct")
    if not z_col or z_col not in df.columns:
        return []
    rules = {
        "Un": (16, 100, None),
        "Le": (11, 15, None),
        "De": (0, 10, "De"),
        "Hy": (0, 10, "Hy"),
        "Pr": (0, 10, "Pr"),
    }
    rows_out = []
    for _, row in df.iterrows():
        zonation = row.get(z_col)
        if pd.isna(zonation):
            continue
        zonation = str(zonation).strip()
        if zonation not in zonation_categories:
            continue
        total_gangue = _safe_float(row.get(g_col)) if g_col else None
        de_pct = _safe_float(row.get(de_col)) if de_col else 0
        hy_pct = _safe_float(row.get(hy_col)) if hy_col else 0
        pr_pct = _safe_float(row.get(pr_col)) if pr_col else 0
        min_g, max_g, dominant = rules.get(zonation, (0, 100, None))
        ok = True
        if total_gangue is not None and not (min_g <= total_gangue < max_g):
            ok = False
        if ok and dominant:
            minerals = {"De": de_pct or 0, "Hy": hy_pct or 0, "Pr": pr_pct or 0}
            actual_dom = max(minerals, key=minerals.get)
            if actual_dom != dominant:
                ok = False
        if not ok:
            should_be = None
            if total_gangue is not None:
                if total_gangue > 15:
                    should_be = "Un"
                elif total_gangue > 10:
                    should_be = "Le"
                elif de_col or hy_col or pr_col:
                    minerals = {"De": de_pct or 0, "Hy": hy_pct or 0, "Pr": pr_pct or 0}
                    should_be = max(minerals, key=minerals.get)
            # Only include rows where logged zonation differs from what it should be
            # (skip rows where classification is actually correct)
            if should_be and should_be != zonation:
                rows_out.append({
                    "hole_id": _safe_str(row.get(hole_col)),
                    "depth_from": _safe_float(row.get(depth_from_col)) if depth_from_col else None,
                    "depth_to": _safe_float(row.get(depth_to_col)),
                    "logged_zonation": zonation,
                    "should_be": should_be,
                    "total_gangue_pct": total_gangue,
                    "de_pct": de_pct,
                    "hy_pct": hy_pct,
                    "pr_pct": pr_pct,
                })
    return rows_out


GROUPING_CV_ELEMENTS = ["fe_pct", "sio2_pct", "al2o3_pct", "mgo_pct", "cao_pct", "s_pct"]


def _build_resolved_group_cols(
    df: pd.DataFrame,
    hole_col: str,
    logger_col: str,
    strat_col: str,
    resolver: Optional[ColumnResolver] = None,
) -> Tuple[List[str], List[str]]:
    """
    Build grouping columns from resolved semantic names; include only columns that exist.
    Returns (list of column names, list of display names for rules box).
    Matches combine_intervals semantics when data has Prospect_D, StratSum, Min_*_pct, LithComments.
    """
    if resolver is None:
        resolver = ColumnResolver(df)
    group_cols = [hole_col, logger_col, strat_col]
    display_names = [hole_col, logger_col, strat_col]
    optional_std = [
        ("prospect_d", "prospect_d"),
        ("stratsum", "stratsum"),
        ("lithology", "lithology"),
    ]
    for std_name, display in optional_std:
        col = resolver.get(std_name)
        if col and col in df.columns and col not in group_cols:
            group_cols.append(col)
            display_names.append(display)
    for c in df.columns:
        if c in group_cols:
            continue
        c_lower = c.lower()
        if ("min_" in c_lower and "_pct" in c_lower) or c_lower in ("min_80_pct", "min_50_pct", "min_20_pct", "min_10_pct", "min_5_pct", "min_0_5_pct"):
            group_cols.append(c)
            display_names.append(c)
    return group_cols, display_names


def _grouping_avg_max_interval_and_groups(
    df: pd.DataFrame,
    group_cols: List[str],
    chem_actual_cols: List[str],
    resolver: ColumnResolver,
    hole_col: str,
    depth_from_col: Optional[str],
    depth_to_col: str,
    top_n_groups: int = 20,
) -> Tuple[Optional[float], Optional[float], List[Dict[str, Any]]]:
    """Returns (avg_group_interval_m, max_group_interval_m, list of groups with high CV)."""
    if not all(c in df.columns for c in group_cols) or not depth_to_col:
        return None, None, []
    available = []
    for std in GROUPING_CV_ELEMENTS:
        col = resolver.get(std)
        if col and col in df.columns and col in chem_actual_cols:
            available.append(col)
    if not available:
        available = [c for c in chem_actual_cols if c in df.columns][:6]
    intervals_per_group = []
    group_cv_list = []
    for key, group in df.groupby(group_cols):
        from_vals = group[depth_from_col] if depth_from_col and depth_from_col in group.columns else 0
        to_vals = group[depth_to_col]
        lengths = to_vals.astype(float) - (from_vals.astype(float) if depth_from_col else 0)
        lengths = lengths[lengths > 0]
        if lengths.empty:
            continue
        mean_len = float(lengths.mean())
        max_len = float(lengths.max())
        intervals_per_group.append((mean_len, max_len))
        max_cv = 0.0
        for col in available:
            vals = group[col].dropna()
            if len(vals) < 2:
                continue
            m = vals.mean()
            if m == 0:
                continue
            cv = vals.std(ddof=1) / abs(m) * 100
            if cv > max_cv:
                max_cv = cv
        if max_cv > 100:
            group_cv_list.append((key, max_cv, mean_len, max_len, group))
    group_cv_list.sort(key=lambda x: x[1], reverse=True)
    avg_all = float(np.mean([x[0] for x in intervals_per_group])) if intervals_per_group else None
    max_all = float(np.max([x[1] for x in intervals_per_group])) if intervals_per_group else None
    groups_for_review = []
    for key, cv, mean_len, max_len, group in group_cv_list[:top_n_groups]:
        intervals_in_group = []
        for _, row in group.iterrows():
            intervals_in_group.append({
                "hole_id": _safe_str(row.get(hole_col)),
                "depth_from": _safe_float(row.get(depth_from_col)) if depth_from_col else None,
                "depth_to": _safe_float(row.get(depth_to_col)),
                "issue": f"CV >100% in group",
                "geochem": {col: _safe_float(row.get(col)) for col in available if col in row.index},
            })
        groups_for_review.append({
            "group_key": str(key),
            "cv_max": cv,
            "mean_interval_m": mean_len,
            "max_interval_m": max_len,
            "intervals": intervals_in_group,
        })
    return avg_all, max_all, groups_for_review


def _build_html_report(
    data_coordinator,
    logger_value: str,
    assay_logger_df: pd.DataFrame,
    logging_logger_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    logging_df: pd.DataFrame,
    collar_df: pd.DataFrame,
    logger_col: str,
    hole_col: str,
    depth_from_col: Optional[str],
    depth_to_col: str,
    strat_col: str,
    chem_actual_cols: List[str],
    page_options: Dict[str, bool],
    date_from: Optional[str],
    date_to: Optional[str],
    top_n: int,
    include_images: bool,
    logo_path: Optional[str],
    stats: Dict[str, Any],
    output_dir: Optional[str] = None,
    full_team_df: Optional[pd.DataFrame] = None,
) -> str:
    # Set up charts output folder for this logger
    charts_dir = None
    if output_dir:
        charts_dir = os.path.join(output_dir, f"charts_{logger_value}")
        os.makedirs(charts_dir, exist_ok=True)
        logger.info(f"Charts will be saved to: {charts_dir}")

    logger_holes = set(
        assay_logger_df[hole_col].astype(str).str.upper().str.strip().tolist()
    )
    all_holes = set(merged_df[hole_col].astype(str).str.upper().str.strip().tolist())
    # For map: use team/project scope so "team holes" count is correct (not 0)
    if full_team_df is not None and hole_col in full_team_df.columns:
        all_holes_for_map = set(
            full_team_df[hole_col].astype(str).str.upper().str.strip().tolist()
        )
    elif collar_df is not None and not collar_df.empty:
        collar_resolver = ColumnResolver(collar_df)
        collar_hole_col = collar_resolver.get("hole_id") or hole_col
        if collar_hole_col in collar_df.columns:
            all_holes_for_map = set(
                collar_df[collar_hole_col].astype(str).str.upper().str.strip().tolist()
            )
        else:
            all_holes_for_map = all_holes
    else:
        all_holes_for_map = all_holes

    comment_stats = _compute_comment_stats(assay_logger_df)
    wordcloud = _extract_comment_wordcloud(assay_logger_df)

    log_from_col, log_to_col = _resolve_logging_interval_columns(logging_logger_df)
    log_from_assay, log_to_assay = _resolve_logging_interval_columns(assay_logger_df)

    assay_received_count, assay_outstanding_count = _compute_assay_received_outstanding(
        logging_logger_df, assay_logger_df, hole_col,
        depth_from_col, depth_to_col, log_from_col, log_to_col,
    )
    average_logging_interval_m = _compute_avg_logging_interval(
        logging_logger_df, log_from_col, log_to_col
    )
    comment_stats_logging = _comment_stats_logging_intervals(logging_logger_df)
    strat_code_counts = (
        assay_logger_df[strat_col].value_counts().head(30).to_dict()
        if strat_col in assay_logger_df.columns
        else {}
    )
    strat_code_list = [{"code": k, "count": int(v)} for k, v in strat_code_counts.items()]

    # Team strat code distribution from full_team_df (all loggers, unfiltered)
    team_data_source = full_team_df if full_team_df is not None else merged_df
    team_strat_code_counts = (
        team_data_source[strat_col].value_counts().head(30).to_dict()
        if strat_col in team_data_source.columns
        else {}
    )
    team_strat_code_list = [{"code": k, "count": int(v)} for k, v in team_strat_code_counts.items()]

    project_col = _resolve_project_code_column(merged_df)
    logger_project_codes = (
        assay_logger_df[project_col].dropna().astype(str).unique().tolist()
        if project_col and project_col in assay_logger_df.columns
        else []
    )
    has_project_scope = bool(project_col and logger_project_codes)
    # Use full_team_df for team statistics (not filtered by logger)
    team_base_df = full_team_df if full_team_df is not None else merged_df
    project_filtered_df = (
        team_base_df[team_base_df[project_col].astype(str).isin(logger_project_codes)].copy()
        if has_project_scope and project_col in team_base_df.columns
        else team_base_df.copy()
    )

    resolver_assay = ColumnResolver(assay_logger_df)
    group_cols, group_cols_display = _build_resolved_group_cols(
        assay_logger_df, hole_col, logger_col, strat_col, resolver_assay
    )
    assay_with_accuracy = _add_mineralisation_accuracy_column(assay_logger_df, resolver_assay)
    date_col = None
    for c in ["drilldate", "StartDate_D", "LoggedDate", "EffectiveDate"]:
        if resolver_assay.get(c) or (c in assay_logger_df.columns):
            date_col = resolver_assay.get(c) or c
            break
    mineral_accuracy_counts = (
        assay_with_accuracy["Logging_Accuracy"].value_counts().to_dict()
        if "Logging_Accuracy" in assay_with_accuracy.columns
        else {}
    )
    mineral_quarterly_user = _group_mineralisation_by_quarter(assay_with_accuracy, date_col)

    # Collect mineralisation intervals for evidence table (mismatches only)
    mineral_mismatch_intervals = []
    if "Logging_Accuracy" in assay_with_accuracy.columns:
        gangue_col = resolver_assay.get("total_gangue_pct")
        fe_col_min = resolver_assay.get("fe_pct")
        sio2_col_min = resolver_assay.get("sio2_pct")
        al2o3_col_min = resolver_assay.get("al2o3_pct")
        # Use same zonation resolution as Zonation tab (BestProfileZonation_D / profile_zonation or derived dominant)
        zonation_cols_min = _resolve_zonation_columns(assay_with_accuracy)
        assay_for_mineral_zonation, zonation_cols_derived = _derive_dominant_zonation_column(
            assay_with_accuracy, zonation_cols_min
        )
        zonation_col = zonation_cols_derived.get("zonation")

        for _, row in assay_for_mineral_zonation.iterrows():
            fe_val = _safe_float(row.get(fe_col_min)) if fe_col_min else None
            sio2_val = _safe_float(row.get(sio2_col_min)) if sio2_col_min else None
            al2o3_val = _safe_float(row.get(al2o3_col_min)) if al2o3_col_min else None
            gangue_val = _safe_float(row.get(gangue_col)) if gangue_col else None

            # Determine what assay suggests: Mineralised, Leached, or Unmineralised
            # Mineralised: Fe >50%, SiO2 <10%, Al2O3 <5%
            # Leached: Fe >50% but SiO2 is 10-15% or Al2O3 is 5-10% (almost mineralised)
            # Unmineralised: otherwise
            if fe_val is not None and fe_val > 50:
                sio2_high = sio2_val is not None and 10 <= sio2_val <= 15
                al2o3_high = al2o3_val is not None and 5 <= al2o3_val <= 10
                sio2_ok = sio2_val is not None and sio2_val < 10
                al2o3_ok = al2o3_val is not None and al2o3_val < 5

                if sio2_ok and al2o3_ok:
                    assay_suggests = "Mineralised"
                elif sio2_high or al2o3_high:
                    assay_suggests = "Leached"
                else:
                    assay_suggests = "Unmineralised"
            else:
                assay_suggests = "Unmineralised"

            # Get the strat code and zonation as logged (zonation from resolved or derived column)
            logged_strat = _safe_str(row.get(strat_col)) if strat_col else ""
            logged_zonation = _safe_str(row.get(zonation_col)) if zonation_col else ""
            validation = row.get("Logging_Accuracy", "")

            if validation == "Mismatch":
                # Low significance = borderline (assay suggests Leached); High = clear mismatch
                significance = "Low" if assay_suggests == "Leached" else "High"
                mineral_mismatch_intervals.append({
                    "hole_id": _safe_str(row.get(hole_col)),
                    "depth_from": _safe_float(row.get(depth_from_col)) if depth_from_col else None,
                    "depth_to": _safe_float(row.get(depth_to_col)),
                    "validation": validation,
                    "logged_as": logged_strat,
                    "logged_zonation": logged_zonation,
                    "assay_suggests": assay_suggests,
                    "significance": significance,
                    "gangue_pct": gangue_val,
                    "geochem": {"Fe": fe_val, "SiO2": sio2_val, "Al2O3": al2o3_val},
                })

    # Add compartment images for mineralisation evidence table
    if data_coordinator and include_images:
        for item in mineral_mismatch_intervals:
            item["image"] = _lookup_interval_image(
                data_coordinator,
                item.get("hole_id"),
                item.get("depth_to"),
            )

    project_with_accuracy = _add_mineralisation_accuracy_column(project_filtered_df, ColumnResolver(project_filtered_df))
    mineral_quarterly_team = _group_mineralisation_by_quarter(project_with_accuracy, date_col)
    mineral_accuracy_counts_team = (
        project_with_accuracy["Logging_Accuracy"].value_counts().to_dict()
        if "Logging_Accuracy" in project_with_accuracy.columns
        else {}
    )

    zonation_cols = _resolve_zonation_columns(assay_logger_df)
    assay_df_for_zonation, zonation_cols = _derive_dominant_zonation_column(assay_logger_df, zonation_cols)
    z_correct, z_mismatch, z_attribution = _check_profile_zonation_and_analyze(assay_df_for_zonation, zonation_cols)
    zonation_mismatch_rows = _collect_zonation_mismatch_rows(
        assay_df_for_zonation, zonation_cols, hole_col, depth_from_col, depth_to_col
    )
    if data_coordinator and include_images:
        for item in zonation_mismatch_rows:
            item["image"] = _lookup_interval_image(
                data_coordinator,
                item.get("hole_id"),
                item.get("depth_to"),
            )
    zonation_cols_team = _resolve_zonation_columns(project_filtered_df)
    team_df_for_zonation, zonation_cols_team = _derive_dominant_zonation_column(project_filtered_df, zonation_cols_team)
    z_correct_team, z_mismatch_team, z_attribution_team = _check_profile_zonation_and_analyze(team_df_for_zonation, zonation_cols_team)

    grouping_avg_m, grouping_max_m, grouping_groups_for_review = _grouping_avg_max_interval_and_groups(
        assay_logger_df, group_cols, chem_actual_cols, resolver_assay,
        hole_col, depth_from_col, depth_to_col, top_n_groups=15,
    )

    grouping_summary = _summarize_grouping_accuracy_by_interval(
        assay_logger_df, group_cols, chem_actual_cols
    )
    grouping_intervals = _build_grouping_issue_intervals(
        assay_logger_df, group_cols, chem_actual_cols, hole_col, depth_from_col, depth_to_col
    )

    fines_summary = _summarize_fines_accuracy(assay_logger_df, chem_actual_cols)
    fines_resolver = ColumnResolver(assay_logger_df)
    fe_col = fines_resolver.get("fe_pct")
    al2o3_col = fines_resolver.get("al2o3_pct")
    sio2_col = fines_resolver.get("sio2_pct")
    p_col = fines_resolver.get("p_pct")
    fines_intervals = []
    for _, row in assay_logger_df.iterrows():
        issue = _flag_fines_issue(row, fines_resolver)
        if issue:
            geochem = {}
            if fe_col and fe_col in row: geochem["Fe"] = _safe_float(row.get(fe_col))
            if al2o3_col and al2o3_col in row: geochem["Al2O3"] = _safe_float(row.get(al2o3_col))
            if sio2_col and sio2_col in row: geochem["SiO2"] = _safe_float(row.get(sio2_col))
            if p_col and p_col in row: geochem["P"] = _safe_float(row.get(p_col))
            fines_intervals.append(
                {
                    "hole_id": _safe_str(row.get(hole_col)),
                    "depth_from": _safe_float(row.get(depth_from_col)) if depth_from_col else None,
                    "depth_to": _safe_float(row.get(depth_to_col)),
                    "classified_as": _safe_str(row.get(strat_col)),  # Geologist's original classification
                    "strat": _safe_str(row.get(strat_col)),
                    "issue": issue,
                    "geochem": geochem,
                }
            )

    # Logging detail accuracy: magnetite, goethite, carbonate issue types
    magnetite_intervals = []
    goethite_intervals = []
    carbonate_intervals = []
    for _, row in assay_logger_df.iterrows():
        geochem = {}
        if fe_col and fe_col in row:
            geochem["Fe"] = _safe_float(row.get(fe_col))
        if al2o3_col and al2o3_col in row:
            geochem["Al2O3"] = _safe_float(row.get(al2o3_col))
        if sio2_col and sio2_col in row:
            geochem["SiO2"] = _safe_float(row.get(sio2_col))
        if p_col and p_col in row:
            geochem["P"] = _safe_float(row.get(p_col))
        base = {
            "hole_id": _safe_str(row.get(hole_col)),
            "depth_from": _safe_float(row.get(depth_from_col)) if depth_from_col else None,
            "depth_to": _safe_float(row.get(depth_to_col)),
            "classified_as": _safe_str(row.get(strat_col)),
            "strat": _safe_str(row.get(strat_col)),
            "geochem": geochem,
        }
        issue_m = _flag_magnetite_issue(row, fines_resolver)
        if issue_m:
            magnetite_intervals.append({**base, "issue": issue_m})
        issue_g = _flag_goethite_issue(row, fines_resolver)
        if issue_g:
            goethite_intervals.append({**base, "issue": issue_g})
        issue_c = _flag_carbonate_issue(row, fines_resolver)
        if issue_c:
            carbonate_intervals.append({**base, "issue": issue_c})

    # Compute "most likely" strat prediction using multivariate distance to all strat centroids
    most_likely_predictions = predict_most_likely_strat(
        assay_logger_df, strat_col, chem_actual_cols, min_group_size=10
    )
    assay_logger_df["most_likely_strat"] = most_likely_predictions

    # Get ALL outliers with positive scores (intervals that don't match expected chemistry)
    # Sort by score descending, but don't limit to top_n - show all misclassifications
    all_outliers = assay_logger_df[assay_logger_df["outlier_score"] > 0].copy()
    all_outliers = all_outliers.sort_values("outlier_score", ascending=False)

    # Total count of potentially misclassified intervals
    total_misclassified = len(all_outliers)

    # Build outlier rows - show top_n in the table but report total count
    display_outliers = all_outliers.head(top_n) if top_n > 0 else all_outliers
    outlier_rows = []
    for _, row in display_outliers.iterrows():
        strat_val = _safe_str(row.get(strat_col))
        most_likely = _safe_str(row.get("most_likely_strat", "-"))
        reason = _safe_str(row.get("outlier_reason"))

        # Build geochem dict for display
        geochem = {}
        if fe_col and fe_col in row:
            geochem["Fe"] = _safe_float(row.get(fe_col))
        if al2o3_col and al2o3_col in row:
            geochem["Al2O3"] = _safe_float(row.get(al2o3_col))
        if sio2_col and sio2_col in row:
            geochem["SiO2"] = _safe_float(row.get(sio2_col))
        if p_col and p_col in row:
            geochem["P"] = _safe_float(row.get(p_col))

        outlier_rows.append(
            {
                "hole_id": _safe_str(row.get(hole_col)),
                "depth_from": _safe_float(row.get(depth_from_col)) if depth_from_col else None,
                "depth_to": _safe_float(row.get(depth_to_col)),
                "strat": strat_val,
                "recorded_as": strat_val,
                "most_likely": most_likely if most_likely != strat_val else strat_val,
                "reason": reason or _safe_str(row.get("outlier_elements")),
                "outlier_score": _safe_float(row.get("outlier_score")) or 0.0,
                "outlier_reason": reason,
                "outlier_elements": _safe_str(row.get("outlier_elements")),
                "geochem": geochem,
            }
        )
    common_errors = []
    if outlier_rows:
        from collections import Counter
        strat_counts = Counter(r["recorded_as"] for r in outlier_rows if r["recorded_as"])
        common_errors = [{"recorded_as": s, "count": c} for s, c in strat_counts.most_common(10)]

    def _metric_total_depth(df: pd.DataFrame) -> float:
        if depth_from_col and depth_to_col:
            return float((df[depth_to_col] - df[depth_from_col]).sum())
        return float("nan")

    summary_metrics = {
        "assay_intervals": len(assay_logger_df),
        "logging_intervals": len(logging_logger_df),
        "unique_holes": assay_logger_df[hole_col].nunique(),
        "strat_codes": assay_logger_df[strat_col].nunique(),
        "total_depth_m": _metric_total_depth(assay_logger_df),
        "match_rate_pct": stats.get("match_rate_pct"),
    }

    summary_median_all = _compute_logger_median(
        merged_df, logger_col, lambda df: float(len(df))
    )
    summary_median_project = _compute_logger_median(
        project_filtered_df, logger_col, lambda df: float(len(df))
    )

    comment_coverage = _calc_comment_coverage(assay_logger_df)
    comment_median_all = _compute_logger_median(
        merged_df,
        logger_col,
        _calc_comment_coverage,
    )
    comment_median_project = _compute_logger_median(
        project_filtered_df,
        logger_col,
        _calc_comment_coverage,
    )

    mineral_col = ColumnResolver(assay_logger_df).get("mineralisation")
    mineralised_intervals = (
        assay_logger_df[mineral_col].astype(str).replace("nan", "").ne("").sum()
        if mineral_col
        else 0
    )
    assays_available = assay_logger_df[chem_actual_cols[0]].notna().sum() if chem_actual_cols else 0
    mineral_assay_ratio = (assays_available / mineralised_intervals * 100) if mineralised_intervals else 0.0

    profile_coverage = _calc_profile_coverage(assay_logger_df, strat_col)

    mineral_median_all = _compute_logger_median(
        merged_df, logger_col, _calc_mineral_assay_ratio
    )
    mineral_median_project = _compute_logger_median(
        project_filtered_df, logger_col, _calc_mineral_assay_ratio
    )
    profile_median_all = _compute_logger_median(
        merged_df, logger_col, lambda df: _calc_profile_coverage(df, strat_col)
    )
    profile_median_project = _compute_logger_median(
        project_filtered_df, logger_col, lambda df: _calc_profile_coverage(df, strat_col)
    )

    fines_rate = _calc_fines_flag_rate(assay_logger_df)
    fines_median_all = _compute_logger_median(
        merged_df, logger_col, _calc_fines_flag_rate
    )
    fines_median_project = _compute_logger_median(
        project_filtered_df, logger_col, _calc_fines_flag_rate
    )

    grouping_rate = _calc_grouping_issue_rate(
        assay_logger_df, group_cols, chem_actual_cols, hole_col, depth_from_col, depth_to_col
    )
    grouping_median_all = _compute_logger_median(
        merged_df,
        logger_col,
        lambda df: _calc_grouping_issue_rate(
            df, group_cols, chem_actual_cols, hole_col, depth_from_col, depth_to_col
        ),
    )
    grouping_median_project = _compute_logger_median(
        project_filtered_df,
        logger_col,
        lambda df: _calc_grouping_issue_rate(
            df, group_cols, chem_actual_cols, hole_col, depth_from_col, depth_to_col
        ),
    )

    outlier_rate = _calc_outlier_rate(assay_logger_df, top_n)
    outlier_median_all = _compute_logger_median(
        merged_df, logger_col, lambda df: _calc_outlier_rate(df, top_n)
    )
    outlier_median_project = _compute_logger_median(
        project_filtered_df, logger_col, lambda df: _calc_outlier_rate(df, top_n)
    )

    map_data = _build_map_points(
        collar_df=collar_df,
        hole_col=hole_col,
        logger_holes=logger_holes,
        all_holes=all_holes_for_map,
    )

    report_data = {
        "meta": {
            "logger": logger_value,
            "date_from": date_from or "",
            "date_to": date_to or "",
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        "overview": {
            "assay_received_count": assay_received_count,
            "assay_outstanding_count": assay_outstanding_count,
            "average_logging_interval_m": average_logging_interval_m,
            "strat_code_list": strat_code_list,
            "team_strat_code_list": team_strat_code_list,
            "comment_ratio_pct_logging": comment_stats_logging["comment_ratio_pct"],
            "comment_ratio_pct": comment_stats_logging["comment_ratio_pct"],
        },
        "summary": summary_metrics,
        "comment_stats": comment_stats,
        "comment_stats_logging": comment_stats_logging,
        "comment_coverage": comment_coverage,
        "comparisons": {
            "assay_intervals": {
                "value": float(summary_metrics["assay_intervals"]),
                "median_all": summary_median_all["median"],
                "median_project": summary_median_project["median"],
            },
            "comment_coverage": {
                "value": float(comment_coverage),
                "median_all": comment_median_all["median"],
                "median_project": comment_median_project["median"],
            },
            "mineral_assay_ratio": {
                "value": float(mineral_assay_ratio),
                "median_all": mineral_median_all["median"],
                "median_project": mineral_median_project["median"],
            },
            "profile_coverage": {
                "value": float(profile_coverage),
                "median_all": profile_median_all["median"],
                "median_project": profile_median_project["median"],
            },
            "fines_rate": {
                "value": float(fines_rate),
                "median_all": fines_median_all["median"],
                "median_project": fines_median_project["median"],
            },
            "grouping_rate": {
                "value": float(grouping_rate),
                "median_all": grouping_median_all["median"],
                "median_project": grouping_median_project["median"],
            },
            "outlier_rate": {
                "value": float(outlier_rate),
                "median_all": outlier_median_all["median"],
                "median_project": outlier_median_project["median"],
            },
        },
        "wordcloud": wordcloud,
        "fines_summary": fines_summary,
        "grouping_summary": grouping_summary,
        "outliers": outlier_rows,
        "mineralisation": {
            "mineralised_intervals": mineralised_intervals,
            "assays_available": assays_available,
            "assay_ratio_pct": mineral_assay_ratio,
            "accuracy_counts": mineral_accuracy_counts,
            "quarterly_user": mineral_quarterly_user,
            "accuracy_counts_team": mineral_accuracy_counts_team,
            "quarterly_team": mineral_quarterly_team,
            "mismatch_intervals": mineral_mismatch_intervals,  # Evidence for review
            "mismatch_low_count": sum(1 for m in mineral_mismatch_intervals if m.get("significance") == "Low"),
            "mismatch_high_count": sum(1 for m in mineral_mismatch_intervals if m.get("significance") == "High"),
        },
        "profile_zonation": {
            "strat_coverage_pct": profile_coverage,
            "unique_strats": assay_logger_df[strat_col].nunique() if strat_col in assay_logger_df.columns else 0,
            "correct_counts": z_correct,
            "mismatch_counts": z_mismatch,
            "mismatch_attribution": z_attribution,
            "mismatch_rows": zonation_mismatch_rows,
            "correct_counts_team": z_correct_team,
            "mismatch_counts_team": z_mismatch_team,
            "mismatch_attribution_team": z_attribution_team,
        },
        "grouping_kpis": {
            "avg_group_interval_m": grouping_avg_m,
            "max_group_interval_m": grouping_max_m,
            "groups_for_review": grouping_groups_for_review,
        },
        "grouping_columns_used": group_cols_display,
        "outlier_kpis": {
            "total_misclassified": total_misclassified,  # Total count of ALL outliers, not just displayed
            "displayed_count": len(outlier_rows),  # Number shown in table (may be limited by top_n)
            "common_errors": common_errors,
        },
        "map": map_data,
        "project_codes": logger_project_codes,
        "has_project_scope": has_project_scope,
        "logging_detail_issue_types": [
            {"key": "fines", "label_fr": "Fines / mineralisation vs gangue", "label_en": "Fines / mineralisation vs gangue", "rules_fr": "Proportion des intervalles ou la geochimie (Fe, SiO2, Al2O3, gangue) suggere un probleme: ex. mineralisation loggee comme gangue, silice friable, argiles.", "rules_en": "Proportion of intervals where geochemistry suggests an issue: e.g. mineralisation logged as gangue, friable silica, clays."},
            {"key": "magnetite", "label_fr": "Magnetite vs LOI1000", "label_en": "Magnetite vs LOI1000", "rules_fr": "LOI1000 negatif suggere magnetite; codes MT, MBH, MBM. Flag: magnetite loggee mais LOI1000 non negatif.", "rules_en": "Negative LOI1000 suggests magnetite; codes MT, MBH, MBM. Flag: magnetite logged but LOI1000 not negative."},
            {"key": "goethite", "label_fr": "Goethite (Hy) vs LOI", "label_en": "Goethite (Hy) vs LOI", "rules_fr": "Zonation Hy (goethite) elevee doit correspondre a un LOI (ex. LOI 425) eleve. Flag: Hy eleve mais LOI bas ou manquant.", "rules_en": "High zonation Hy (goethite) should align with LOI (e.g. LOI 425). Flag: high Hy but low or missing LOI."},
            {"key": "carbonate_gangue", "label_fr": "Gangue carbonate vs essai", "label_en": "Carbonate gangue vs assay", "rules_fr": "Gangue carbonate loggee % coherente avec l'essai (LOI/CaO). Flag: ecart important entre logge et essai.", "rules_en": "Logged carbonate gangue % should be consistent with assay (LOI/CaO). Flag: logged % inconsistent with assay."},
        ],
    }

    fines_with_images = _build_intervals_with_images(
        data_coordinator, fines_intervals, include_images
    )
    for i, item in enumerate(fines_with_images):
        if i < len(fines_intervals) and "geochem" in fines_intervals[i]:
            item["geochem"] = fines_intervals[i]["geochem"]

    magnetite_with_images = _build_intervals_with_images(
        data_coordinator, magnetite_intervals, include_images
    )
    goethite_with_images = _build_intervals_with_images(
        data_coordinator, goethite_intervals, include_images
    )
    carbonate_with_images = _build_intervals_with_images(
        data_coordinator, carbonate_intervals, include_images
    )
    for i, item in enumerate(magnetite_with_images):
        if i < len(magnetite_intervals) and "geochem" in magnetite_intervals[i]:
            item["geochem"] = magnetite_intervals[i]["geochem"]
    for i, item in enumerate(goethite_with_images):
        if i < len(goethite_intervals) and "geochem" in goethite_intervals[i]:
            item["geochem"] = goethite_intervals[i]["geochem"]
    for i, item in enumerate(carbonate_with_images):
        if i < len(carbonate_intervals) and "geochem" in carbonate_intervals[i]:
            item["geochem"] = carbonate_intervals[i]["geochem"]

    grouping_with_groups = _build_grouping_groups_with_images(
        data_coordinator, grouping_groups_for_review, include_images
    )

    intervals_for_review = {
        "fines": fines_with_images,
        "logging_detail": {
            "fines": fines_with_images,
            "magnetite": magnetite_with_images,
            "goethite": goethite_with_images,
            "carbonate_gangue": carbonate_with_images,
        },
        "grouping_flat": _build_intervals_with_images(
            data_coordinator, grouping_intervals, include_images
        ),
        "grouping": grouping_with_groups,
        "outliers": _build_outlier_intervals_with_images(
            data_coordinator, outlier_rows, include_images
        ),
    }

    return _render_html(report_data, intervals_for_review, logo_path, page_options, charts_dir=charts_dir)


def _build_intervals_with_images(
    data_coordinator,
    intervals: List[Dict[str, Any]],
    include_images: bool,
) -> List[Dict[str, Any]]:
    result = []
    for item in intervals:
        image_data = None
        if include_images:
            image_data = _lookup_interval_image(
                data_coordinator,
                item.get("hole_id"),
                item.get("depth_to"),
            )
        result.append({**item, "image": image_data})
    return result


def _build_grouping_groups_with_images(
    data_coordinator,
    groups: List[Dict[str, Any]],
    include_images: bool,
) -> List[Dict[str, Any]]:
    result = []
    for grp in groups:
        intervals_with_img = []
        for it in grp.get("intervals", []):
            image_data = None
            if include_images:
                image_data = _lookup_interval_image(
                    data_coordinator,
                    it.get("hole_id"),
                    it.get("depth_to"),
                )
            intervals_with_img.append({**it, "image": image_data})
        result.append({
            "group_key": grp.get("group_key"),
            "cv_max": grp.get("cv_max"),
            "mean_interval_m": grp.get("mean_interval_m"),
            "max_interval_m": grp.get("max_interval_m"),
            "intervals": intervals_with_img,
        })
    return result


def _build_outlier_intervals_with_images(
    data_coordinator,
    intervals: List[Dict[str, Any]],
    include_images: bool,
) -> List[Dict[str, Any]]:
    result = []
    for item in intervals:
        image_data = None
        if include_images:
            image_data = _lookup_interval_image(
                data_coordinator,
                item.get("hole_id"),
                item.get("depth_to"),
            )
        issue = item.get("reason") or item.get("outlier_reason") or "Geochemistry outlier vs strat expectation"
        result.append({**item, "issue": issue, "image": image_data})
    return result


def _lookup_interval_image(data_coordinator, hole_id: Optional[str], depth_to: Optional[float]) -> Optional[str]:
    """
    Look up and encode an interval image from the data coordinator.

    Tries multiple moisture statuses (Wet, Dry, None) to find a matching image,
    since the image index keys include moisture_status but the lookup may not know it.
    """
    if not data_coordinator or not hole_id or depth_to is None:
        return None
    try:
        hole_str = str(hole_id).upper()
        depth_float = float(depth_to)

        # Try different moisture statuses - images are typically indexed with "Wet" or "Dry"
        for moisture in ("Wet", "Dry", None):
            key = ImageKey(hole_str, depth_float, moisture)
            img_path = data_coordinator.get_image_path(key)
            if img_path:
                encoded = _encode_image_base64(img_path)
                if encoded:
                    return encoded

        # If no direct match, try to find any image for this hole/depth via the hole index
        # This handles cases where the exact moisture_status doesn't match
        try:
            hole_keys = data_coordinator.get_keys_for_hole(hole_str)
            depth_int = int(depth_float)
            for key in hole_keys:
                if key.depth_to_int == depth_int:
                    img_path = data_coordinator.get_image_path(key)
                    if img_path:
                        encoded = _encode_image_base64(img_path)
                        if encoded:
                            return encoded
        except Exception:
            pass

    except Exception as e:
        logger.debug(f"Image lookup failed for {hole_id}@{depth_to}: {e}")
        return None
    return None


def _render_wordcloud(word_frequencies: Dict[str, int], save_path: Optional[str] = None) -> str:
    """
    Render a word cloud as a base64-embedded image.

    Uses matplotlib WordCloud library to generate a professional visualization
    with brand colors, matching the style of the old QAQC scripts.

    Args:
        word_frequencies: Dictionary of {word: count}
        save_path: Optional file path to save the chart PNG (in addition to embedding)

    Returns:
        HTML string with embedded base64 image, or fallback text-based cloud
    """
    if not word_frequencies:
        logger.debug("No word frequencies provided for wordcloud")
        return (
            "<div class=\"empty\" data-i18n-fr=\"Aucun texte de commentaire disponible.\" "
            "data-i18n-en=\"No comment text available.\">"
            "Aucun texte de commentaire disponible.</div>"
        )

    logger.debug(f"Generating wordcloud with {len(word_frequencies)} words, WORDCLOUD_AVAILABLE={WORDCLOUD_AVAILABLE}")

    # Try to generate matplotlib WordCloud image
    if WORDCLOUD_AVAILABLE:
        try:
            # Brand colors matching old QAQC scripts
            brand_colors = ['#00AEC7', '#6BC643']

            def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
                return random.choice(brand_colors)

            # Generate word cloud
            wordcloud = WordCloud(
                width=1200,
                height=600,
                background_color='white',
                color_func=color_func,
                scale=2,
                max_words=100,
                min_font_size=10,
            ).generate_from_frequencies(word_frequencies)

            # Render to PNG in memory
            fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(
                'Wordmap showing your most popular comments',
                fontsize=12, fontstyle='italic', pad=10
            )

            # Save to file if path provided
            if save_path:
                try:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    fig.savefig(save_path, format='png', bbox_inches='tight', dpi=150)
                    logger.info(f"WordCloud chart saved to: {save_path}")
                except Exception as save_err:
                    logger.warning(f"Failed to save wordcloud to file: {save_err}")

            # Save to bytes buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            plt.close(fig)
            buf.seek(0)

            # Encode as base64
            img_base64 = base64.b64encode(buf.read()).decode('ascii')
            buf.close()

            logger.info(f"WordCloud image generated successfully ({len(img_base64)} bytes base64)")

            return (
                f'<div class="wordcloud-image">'
                f'<img src="data:image/png;base64,{img_base64}" '
                f'alt="Word Cloud" style="max-width:100%; height:auto; border-radius:8px;" />'
                f'</div>'
            )

        except Exception as e:
            logger.warning(f"WordCloud image generation failed, falling back to text: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    # Fallback: text-based word cloud (used if wordcloud library unavailable or error)
    logger.info(f"Using fallback text-based word cloud (WORDCLOUD_AVAILABLE={WORDCLOUD_AVAILABLE})")
    sorted_words = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)[:40]
    if not sorted_words:
        return (
            "<div class=\"empty\" data-i18n-fr=\"Aucun texte de commentaire disponible.\" "
            "data-i18n-en=\"No comment text available.\">"
            "Aucun texte de commentaire disponible.</div>"
        )

    counts = [c for _, c in sorted_words]
    min_count = min(counts)
    max_count = max(counts)
    span_html = []
    for word, count in sorted_words:
        escaped_word = html.escape(word)
        if max_count == min_count:
            size = 18
        else:
            size = 12 + (count - min_count) / (max_count - min_count) * 22
        span_html.append(
            f"<span class=\"word\" style=\"font-size:{size:.0f}px\">{escaped_word}</span>"
        )
    return "<div class=\"wordcloud\">" + " ".join(span_html) + "</div>"


def _render_comment_bar_chart(total_intervals: int, intervals_with_comments: int, avg_length: float = 0, save_path: Optional[str] = None) -> str:
    """
    Render a matplotlib bar chart for comment statistics as a base64-embedded image.

    Matches the style of the old QAQC scripts' comment statistics bar chart.

    Args:
        total_intervals: Total number of logging intervals
        intervals_with_comments: Number of intervals with non-empty comments
        avg_length: Average comment length in characters
        save_path: Optional file path to save the chart PNG (in addition to embedding)

    Returns:
        HTML string with embedded base64 image
    """
    if total_intervals == 0:
        return "<div class=\"empty\">No data available for chart.</div>"

    try:
        intervals_without = total_intervals - intervals_with_comments
        comment_pct = (intervals_with_comments / total_intervals * 100) if total_intervals > 0 else 0

        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

        categories = ['No Comment', 'With Comment']
        counts = [intervals_without, intervals_with_comments]
        colors = ['#c9382a', '#2f7d61']  # Red for no comment, green for comment

        bars = ax.bar(categories, counts, color=colors, width=0.5)

        # Annotate bars with counts
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{int(yval):,}',
                    ha='center', va='bottom', fontsize=9)

        title = f'Comment Statistics\nTotal: {total_intervals:,} intervals ({comment_pct:.1f}% with comments)'
        if avg_length > 0:
            title += f'\nAvg length: {avg_length:.0f} chars'
        ax.set_title(title, fontsize=11, fontstyle='italic', pad=10)
        ax.set_ylabel('Count', fontsize=9, fontweight='bold')

        # Save to file if path provided
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path, format='png', bbox_inches='tight', dpi=150)
                logger.info(f"Comment bar chart saved to: {save_path}")
            except Exception as save_err:
                logger.warning(f"Failed to save comment chart to file: {save_err}")

        # Save to bytes buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        buf.seek(0)

        img_base64 = base64.b64encode(buf.read()).decode('ascii')
        buf.close()

        return (
            f'<div class="comment-chart-image">'
            f'<img src="data:image/png;base64,{img_base64}" '
            f'alt="Comment Statistics" style="max-width:100%; height:auto;" />'
            f'</div>'
        )

    except Exception as e:
        logger.warning(f"Comment bar chart generation failed: {e}")
        return "<div class=\"empty\">Chart generation failed.</div>"


def _render_intervals_table(intervals: List[Dict[str, Any]], logger_id: str, tab_id: str) -> str:
    if not intervals:
        return (
            "<div class=\"empty\" data-i18n-fr=\"Aucun intervalle signale pour revue.\" "
            "data-i18n-en=\"No intervals flagged for review.\">"
            "Aucun intervalle signale pour revue.</div>"
        )
    rows = []
    for idx, item in enumerate(intervals):
        checkbox_id = f"{logger_id}::{tab_id}::{idx}"
        image_html = (
            f"<img src=\"{item['image']}\" alt=\"Interval image\" class=\"rotated-image\" />"
            if item.get("image")
            else "<div class=\"image-placeholder\" data-i18n-fr=\"Aucune image\" data-i18n-en=\"No image\">Aucune image</div>"
        )
        depth = (
            f"{item.get('depth_from', '')} - {item.get('depth_to', '')}"
            if item.get("depth_from") is not None
            else f"{item.get('depth_to', '')}"
        )
        rows.append(
            "<tr>"
            f"<td><input type=\"checkbox\" data-review-id=\"{checkbox_id}\"></td>"
            f"<td>{html.escape(_safe_str(item.get('hole_id')))}</td>"
            f"<td>{html.escape(str(depth))}</td>"
            f"<td>{html.escape(_safe_str(item.get('issue')))}</td>"
            f"<td class=\"image-cell rotated-image-cell\">{image_html}</td>"
            "</tr>"
        )
    return (
        "<table class=\"interval-table\">"
        "<thead><tr>"
        "<th></th>"
        "<th data-i18n-fr=\"Trou\" data-i18n-en=\"Hole\">Trou</th>"
        "<th data-i18n-fr=\"Profondeur\" data-i18n-en=\"Depth\">Profondeur</th>"
        "<th data-i18n-fr=\"Probleme probable\" data-i18n-en=\"Likely issue\">Probleme probable</th>"
        "<th data-i18n-fr=\"Image\" data-i18n-en=\"Image\">Image</th>"
        "</tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_mineralisation_evidence_table(intervals: List[Dict[str, Any]], logger_id: str) -> str:
    """Render mineralisation evidence table with sortable columns and conditional formatting."""
    if not intervals:
        return (
            "<div class=\"empty\" data-i18n-fr=\"Aucun intervalle de mineralisation.\" "
            "data-i18n-en=\"No mineralisation intervals.\">"
            "Aucun intervalle de mineralisation.</div>"
        )

    def format_depth(val):
        """Format depth as integer if whole number, otherwise as float."""
        if val is None:
            return ""
        if isinstance(val, float) and val == int(val):
            return str(int(val))
        return str(val)

    def get_assay_class(assay_suggests):
        """Get CSS class for conditional formatting based on assay suggestion."""
        if assay_suggests == "Mineralised":
            return "assay-mineralised"
        elif assay_suggests == "Leached":
            return "assay-leached"
        else:
            return "assay-unmineralised"

    def get_validation_class(validation):
        """Get CSS class for validation status."""
        if validation == "Match":
            return "validation-match"
        elif validation == "Mismatch":
            return "validation-mismatch"
        else:
            return "validation-pending"

    rows = []
    for idx, item in enumerate(intervals):
        checkbox_id = f"{logger_id}::mineral_evidence::{idx}"

        # Format depths as integers if whole numbers
        depth_from = format_depth(item.get('depth_from'))
        depth_to = format_depth(item.get('depth_to'))
        depth = f"{depth_from} - {depth_to}" if depth_from else depth_to

        # Build geochem display with conditional formatting
        geochem = item.get("geochem", {})
        fe_val = geochem.get('Fe')
        sio2_val = geochem.get('SiO2')
        al2o3_val = geochem.get('Al2O3')

        fe_display = f"{fe_val:.1f}" if fe_val is not None else "-"
        sio2_display = f"{sio2_val:.1f}" if sio2_val is not None else "-"
        al2o3_display = f"{al2o3_val:.1f}" if al2o3_val is not None else "-"

        assay_suggests = item.get("assay_suggests", "")
        assay_class = get_assay_class(assay_suggests)
        validation = item.get("validation", "")
        validation_class = get_validation_class(validation)
        significance = item.get("significance", "High")
        logged_as = item.get("logged_as", "")
        logged_zonation = item.get("logged_zonation", "")
        if item.get("image"):
            image_html = (
                f"<img src=\"{item['image']}\" alt=\"Interval\" class=\"rotated-image expandable-img img-small\" "
                f"onclick=\"handleImageExpand(this)\" title=\"Click to expand\" />"
            )
        else:
            image_html = "<div class=\"image-placeholder-small\">-</div>"

        rows.append(
            "<tr>"
            f"<td><input type=\"checkbox\" data-review-id=\"{checkbox_id}\"></td>"
            f"<td data-sort=\"{html.escape(_safe_str(item.get('hole_id')))}\">{html.escape(_safe_str(item.get('hole_id')))}</td>"
            f"<td data-sort=\"{item.get('depth_from', 0) or 0}\">{html.escape(depth)}</td>"
            f"<td class=\"{validation_class}\" data-sort=\"{validation}\">{html.escape(validation)}</td>"
            f"<td class=\"significance-{significance.lower()}\" data-sort=\"{significance}\">{html.escape(significance)}</td>"
            f"<td data-sort=\"{logged_as}\">{html.escape(logged_as)}</td>"
            f"<td data-sort=\"{logged_zonation}\">{html.escape(logged_zonation)}</td>"
            f"<td class=\"{assay_class}\" data-sort=\"{assay_suggests}\">{html.escape(assay_suggests)}</td>"
            f"<td class=\"geochem-cell {assay_class}\" data-sort=\"{fe_val or 0}\">{fe_display}</td>"
            f"<td class=\"geochem-cell {assay_class}\" data-sort=\"{sio2_val or 0}\">{sio2_display}</td>"
            f"<td class=\"geochem-cell {assay_class}\" data-sort=\"{al2o3_val or 0}\">{al2o3_display}</td>"
            f"<td class=\"image-cell-compact\">{image_html}</td>"
            "</tr>"
        )
    return (
        "<table class=\"interval-table evidence-table sortable-table\" id=\"mineral-evidence-table\">"
        "<thead><tr>"
        "<th></th>"
        "<th class=\"sortable\" data-i18n-fr=\"Trou\" data-i18n-en=\"Hole\">Hole ▼</th>"
        "<th class=\"sortable\" data-i18n-fr=\"Profondeur\" data-i18n-en=\"Depth\">Depth ▼</th>"
        "<th class=\"sortable\" data-i18n-fr=\"Validation\" data-i18n-en=\"Validation\">Validation ▼</th>"
        "<th class=\"sortable\" data-i18n-fr=\"Significance\" data-i18n-en=\"Significance\">Significance ▼</th>"
        "<th class=\"sortable\" data-i18n-fr=\"Logue comme\" data-i18n-en=\"Logged as\">Logged as ▼</th>"
        "<th class=\"sortable\" data-i18n-fr=\"Zonation loggee\" data-i18n-en=\"Logged Zonation\">Logged Zonation ▼</th>"
        "<th class=\"sortable\" data-i18n-fr=\"Essai suggere\" data-i18n-en=\"Assay suggests\">Assay suggests ▼</th>"
        "<th class=\"sortable\">Fe% ▼</th>"
        "<th class=\"sortable\">SiO2% ▼</th>"
        "<th class=\"sortable\">Al2O3% ▼</th>"
        "<th data-i18n-fr=\"Image\" data-i18n-en=\"Image\">Image</th>"
        "</tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_zonation_evidence_table(
    intervals: List[Dict[str, Any]], logger_id: str
) -> str:
    """Evidence table for zonation mismatches: Hole, Depth, Logged zonation, Should be, Total gangue %, De/Hy/Pr %, Image."""
    if not intervals:
        return (
            "<div class=\"empty\" data-i18n-fr=\"Aucune discordance de zonation.\" "
            "data-i18n-en=\"No zonation mismatches.\">No zonation mismatches.</div>"
        )
    rows = []
    for idx, item in enumerate(intervals):
        checkbox_id = f"{logger_id}::zonation_evidence::{idx}"
        depth = (
            f"{item.get('depth_from', '')} - {item.get('depth_to', '')}"
            if item.get("depth_from") is not None
            else f"{item.get('depth_to', '')}"
        )
        total_g = item.get("total_gangue_pct")
        total_g_display = f"{total_g:.1f}%" if total_g is not None else "-"
        de_display = f"{item.get('de_pct', 0) or 0:.1f}%" if item.get("de_pct") is not None else "-"
        hy_display = f"{item.get('hy_pct', 0) or 0:.1f}%" if item.get("hy_pct") is not None else "-"
        pr_display = f"{item.get('pr_pct', 0) or 0:.1f}%" if item.get("pr_pct") is not None else "-"
        if item.get("image"):
            image_html = (
                f"<img src=\"{item['image']}\" alt=\"Interval\" class=\"rotated-image expandable-img img-small\" "
                f"onclick=\"handleImageExpand(this)\" title=\"Click to expand\" />"
            )
        else:
            image_html = "<div class=\"image-placeholder-small\">-</div>"
        rows.append(
            "<tr>"
            f"<td><input type=\"checkbox\" data-review-id=\"{checkbox_id}\"></td>"
            f"<td class=\"sortable\" data-sort=\"{html.escape(_safe_str(item.get('hole_id')))}\">{html.escape(_safe_str(item.get('hole_id')))}</td>"
            f"<td class=\"sortable\" data-sort=\"{item.get('depth_from', 0) or 0}\">{html.escape(str(depth))}</td>"
            f"<td data-sort=\"{html.escape(_safe_str(item.get('logged_zonation')))}\">{html.escape(_safe_str(item.get('logged_zonation')))}</td>"
            f"<td data-sort=\"{html.escape(_safe_str(item.get('should_be')))}\">{html.escape(_safe_str(item.get('should_be')))}</td>"
            f"<td class=\"sortable\" data-sort=\"{total_g or 0}\">{total_g_display}</td>"
            f"<td class=\"sortable\" data-sort=\"{item.get('de_pct') or 0}\">{de_display}</td>"
            f"<td class=\"sortable\" data-sort=\"{item.get('hy_pct') or 0}\">{hy_display}</td>"
            f"<td class=\"sortable\" data-sort=\"{item.get('pr_pct') or 0}\">{pr_display}</td>"
            f"<td class=\"image-cell-compact\">{image_html}</td>"
            "</tr>"
        )
    return (
        "<table class=\"interval-table evidence-table sortable-table\" id=\"zonation-evidence-table\">"
        "<thead><tr>"
        "<th></th>"
        "<th class=\"sortable\" data-i18n-fr=\"Trou\" data-i18n-en=\"Hole\">Hole</th>"
        "<th class=\"sortable\" data-i18n-fr=\"Profondeur\" data-i18n-en=\"Depth\">Depth</th>"
        "<th class=\"sortable\" data-i18n-fr=\"Zonation loggee\" data-i18n-en=\"Logged zonation\">Logged zonation</th>"
        "<th class=\"sortable\" data-i18n-fr=\"Devrait etre\" data-i18n-en=\"Should be\">Should be</th>"
        "<th class=\"sortable\" data-i18n-fr=\"Gangue totale %\" data-i18n-en=\"Total gangue %\">Total gangue %</th>"
        "<th class=\"sortable\">De %</th>"
        "<th class=\"sortable\">Hy %</th>"
        "<th class=\"sortable\">Pr %</th>"
        "<th data-i18n-fr=\"Image\" data-i18n-en=\"Image\">Image</th>"
        "</tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_logging_detail_evidence_table(
    intervals: List[Dict[str, Any]], logger_id: str, table_id: str
) -> str:
    """One evidence table per issue type: Hole, Depth, Issue, Fe%, SiO2%, Al2O3%, Classified as, Image (small placeholder)."""
    if not intervals:
        return (
            "<div class=\"empty\" data-i18n-fr=\"Aucun intervalle signale pour revue.\" "
            "data-i18n-en=\"No intervals flagged for review.\">"
            "No intervals flagged for review.</div>"
        )
    rows = []
    for idx, item in enumerate(intervals):
        checkbox_id = f"{logger_id}::logging_detail::{table_id}::{idx}"
        if item.get("image"):
            image_html = (
                f"<img src=\"{item['image']}\" alt=\"Interval\" class=\"rotated-image expandable-img img-small\" "
                f"onclick=\"handleImageExpand(this)\" title=\"Click to expand\" />"
            )
        else:
            image_html = "<div class=\"image-placeholder-small\">-</div>"
        depth = (
            f"{item.get('depth_from', '')} - {item.get('depth_to', '')}"
            if item.get("depth_from") is not None
            else f"{item.get('depth_to', '')}"
        )
        classified_as = html.escape(_safe_str(item.get("classified_as", item.get("strat", "-"))))
        geochem = item.get("geochem", {})
        fe_val = geochem.get("Fe")
        sio2_val = geochem.get("SiO2")
        al2o3_val = geochem.get("Al2O3")
        fe_display = f"{fe_val:.1f}" if fe_val is not None else "-"
        sio2_display = f"{sio2_val:.1f}" if sio2_val is not None else "-"
        al2o3_display = f"{al2o3_val:.1f}" if al2o3_val is not None else "-"

        rows.append(
            "<tr class=\"compact-row\">"
            f"<td><input type=\"checkbox\" data-review-id=\"{checkbox_id}\"></td>"
            f"<td class=\"sortable\" data-sort=\"{html.escape(_safe_str(item.get('hole_id')))}\">{html.escape(_safe_str(item.get('hole_id')))}</td>"
            f"<td class=\"sortable\" data-sort=\"{item.get('depth_from', 0) or 0}\">{html.escape(str(depth))}</td>"
            f"<td class=\"issue-cell\">{html.escape(_safe_str(item.get('issue')))}</td>"
            f"<td class=\"geochem-cell\" data-sort=\"{fe_val or 0}\">{fe_display}</td>"
            f"<td class=\"geochem-cell\" data-sort=\"{sio2_val or 0}\">{sio2_display}</td>"
            f"<td class=\"geochem-cell\" data-sort=\"{al2o3_val or 0}\">{al2o3_display}</td>"
            f"<td class=\"classification-cell\">{classified_as}</td>"
            f"<td class=\"image-cell-compact\">{image_html}</td>"
            "</tr>"
        )
    return (
        "<table class=\"interval-table evidence-table sortable-table compact-table\" "
        f"id=\"logging-detail-table-{html.escape(table_id)}\">"
        "<thead><tr>"
        "<th></th>"
        "<th class=\"sortable\" data-i18n-fr=\"Trou\" data-i18n-en=\"Hole\">Hole</th>"
        "<th class=\"sortable\" data-i18n-fr=\"Profondeur\" data-i18n-en=\"Depth\">Depth</th>"
        "<th class=\"sortable\" data-i18n-fr=\"Probleme\" data-i18n-en=\"Issue\">Issue</th>"
        "<th class=\"sortable\">Fe%</th>"
        "<th class=\"sortable\">SiO2%</th>"
        "<th class=\"sortable\">Al2O3%</th>"
        "<th data-i18n-fr=\"Classe comme\" data-i18n-en=\"Classified as\">Classified as</th>"
        "<th data-i18n-fr=\"Image\" data-i18n-en=\"Image\">Image</th>"
        "</tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_fines_intervals_table(intervals: List[Dict[str, Any]], logger_id: str, tab_id: str) -> str:
    """Fines: delegate to logging detail evidence table (small placeholder, geochem columns)."""
    return _render_logging_detail_evidence_table(intervals, logger_id, tab_id)


def _render_outlier_table(intervals: List[Dict[str, Any]], logger_id: str) -> str:
    """Outlier table: Recorded as, Most likely, Reason, image."""
    if not intervals:
        return (
            "<div class=\"empty\" data-i18n-fr=\"Aucun intervalle signale.\" "
            "data-i18n-en=\"No intervals flagged.\">Aucun intervalle signale.</div>"
        )
    rows = []
    for idx, item in enumerate(intervals):
        checkbox_id = f"{logger_id}::outliers::{idx}"
        image_html = (
            f"<img src=\"{item['image']}\" alt=\"\" class=\"rotated-image\" />"
            if item.get("image")
            else "<div class=\"image-placeholder\">-</div>"
        )
        depth = f"{item.get('depth_from', '')} - {item.get('depth_to', '')}" if item.get("depth_from") is not None else str(item.get("depth_to", ""))

        # Build tabulated geochemistry display
        geochem = item.get("geochem", {})
        geochem_rows = []
        for k, v in geochem.items():
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                geochem_rows.append(f"<tr><td class=\"geochem-label\">{html.escape(k)}</td><td class=\"geochem-value\">{v:.2f}</td></tr>")
            else:
                geochem_rows.append(f"<tr><td class=\"geochem-label\">{html.escape(k)}</td><td class=\"geochem-value\">-</td></tr>")
        geochem_html = f"<table class=\"geochem-table\">{''.join(geochem_rows)}</table>" if geochem_rows else "<span class=\"geochem-na\">-</span>"

        rows.append(
            "<tr>"
            f"<td><input type=\"checkbox\" data-review-id=\"{checkbox_id}\"></td>"
            f"<td>{html.escape(_safe_str(item.get('hole_id')))}</td>"
            f"<td>{html.escape(depth)}</td>"
            f"<td>{html.escape(_safe_str(item.get('recorded_as', item.get('strat', ''))))}</td>"
            f"<td class=\"most-likely-cell\">{html.escape(_safe_str(item.get('most_likely', '-')))}</td>"
            f"<td class=\"geochem-cell\">{geochem_html}</td>"
            f"<td class=\"reason-cell\">{html.escape(_safe_str(item.get('reason', item.get('issue', ''))))}</td>"
            f"<td class=\"image-cell rotated-image-cell\">{image_html}</td>"
            "</tr>"
        )
    return (
        "<table class=\"interval-table outlier-table\">"
        "<thead><tr>"
        "<th></th>"
        "<th data-i18n-fr=\"Trou\" data-i18n-en=\"Hole\">Trou</th>"
        "<th data-i18n-fr=\"Profondeur\" data-i18n-en=\"Depth\">Profondeur</th>"
        "<th data-i18n-fr=\"Enregistre comme\" data-i18n-en=\"Recorded as\">Enregistre comme</th>"
        "<th data-i18n-fr=\"Plus probable\" data-i18n-en=\"Most likely\">Plus probable</th>"
        "<th data-i18n-fr=\"Geochimie\" data-i18n-en=\"Geochem\">Geochimie</th>"
        "<th data-i18n-fr=\"Raison\" data-i18n-en=\"Reason\">Raison</th>"
        "<th data-i18n-fr=\"Image\" data-i18n-en=\"Image\">Image</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_grouping_groups(groups: List[Dict[str, Any]], logger_id: str) -> str:
    """Render groups for review with visual CV indicators and compact layout."""
    if not groups:
        return (
            "<div class=\"empty\" data-i18n-fr=\"Aucun groupe a revoir.\" "
            "data-i18n-en=\"No groups for review.\">Aucun groupe a revoir.</div>"
        )
    out = []
    for gidx, grp in enumerate(groups):
        cv_max = grp.get('cv_max', 0)
        strat = html.escape(_safe_str(grp.get('strat', '')))
        num_intervals = len(grp.get("intervals", []))

        # CV severity indicator: green <30%, yellow 30-50%, red >50%
        if cv_max < 30:
            cv_class = "cv-low"
            cv_color = "#22c55e"
        elif cv_max < 50:
            cv_class = "cv-medium"
            cv_color = "#f59e0b"
        else:
            cv_class = "cv-high"
            cv_color = "#ef4444"

        # Visual group header with CV bar
        cv_bar_width = min(cv_max, 100)
        out.append(f'''<div class="grouping-group {cv_class}">
            <div class="group-header">
                <div class="group-info">
                    <span class="group-strat">{strat if strat else "Groupe " + str(gidx+1)}</span>
                    <span class="group-count">{num_intervals} intervalle(s)</span>
                </div>
                <div class="cv-indicator">
                    <span class="cv-label">CV max:</span>
                    <div class="cv-bar-container">
                        <div class="cv-bar" style="width:{cv_bar_width}%;background:{cv_color}"></div>
                    </div>
                    <span class="cv-value" style="color:{cv_color}">{cv_max:.0f}%</span>
                </div>
            </div>''')

        # Compact table
        out.append('<table class="interval-table compact-table grouping-table"><thead><tr>'
                   '<th>Trou</th><th>Prof.</th><th>Strat</th><th>Geochimie</th><th>Image</th>'
                   '</tr></thead><tbody>')
        for it in grp.get("intervals", []):
            depth = f"{it.get('depth_from', '')}-{it.get('depth_to', '')}" if it.get("depth_from") is not None else str(it.get("depth_to", ""))
            interval_strat = html.escape(_safe_str(it.get('strat', '')))

            # Inline geochem display
            geochem = it.get("geochem", {})
            geochem_parts = []
            for k, v in geochem.items():
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    geochem_parts.append(f"<span class='geochem-inline'><b>{k}:</b>{v:.1f}</span>")
            geochem_html = " ".join(geochem_parts) if geochem_parts else "-"

            # Expandable image
            if it.get("image"):
                img = f'<img src="{it["image"]}" alt="" class="rotated-image expandable-img" onclick="handleImageExpand(this)" />'
            else:
                img = '<div class="image-placeholder">-</div>'

            out.append(f'<tr class="compact-row">'
                       f'<td class="nowrap">{html.escape(_safe_str(it.get("hole_id")))}</td>'
                       f'<td class="nowrap">{html.escape(depth)}</td>'
                       f'<td class="strat-cell">{interval_strat}</td>'
                       f'<td class="geochem-inline-cell">{geochem_html}</td>'
                       f'<td class="image-cell-compact">{img}</td>'
                       '</tr>')
        out.append("</tbody></table></div>")
    return "".join(out)


def _plotly_pie_json(labels: List[str], values: List[float], title: str) -> Tuple[str, str]:
    """Return (data_json, layout_json) for a Plotly pie chart."""
    import json
    colors = {"Match": "#2f7d61", "Mismatch": "#c9382a", "Pending Assays": "#5d6672"}
    data = [{
        "type": "pie",
        "labels": labels,
        "values": values,
        "marker": {"colors": [colors.get(l, "#5d6672") for l in labels]},
        "textinfo": "label+percent",
        "hovertemplate": "%{label}: %{value} (%{percent})<extra></extra>",
    }]
    layout = {
        "title": {"text": title},
        "margin": {"t": 40, "b": 20, "l": 20, "r": 20},
        "showlegend": True,
        "height": 280,
        "autosize": True,
    }
    return json.dumps(data), json.dumps(layout)


def _plotly_stacked_bar_json(
    quarters: List[str], match: List[float], mismatch: List[float], pending: List[float], title: str
) -> Tuple[str, str]:
    """Return (data_json, layout_json) for stacked bar by quarter (absolute counts)."""
    import json
    data = [
        {"type": "bar", "x": quarters, "y": pending, "name": "Pending Assays", "marker": {"color": "#5d6672"}},
        {"type": "bar", "x": quarters, "y": mismatch, "name": "Mismatch", "marker": {"color": "#c9382a"}},
        {"type": "bar", "x": quarters, "y": match, "name": "Match", "marker": {"color": "#2f7d61"}},
    ]
    layout = {
        "title": {"text": title},
        "barmode": "stack",
        "margin": {"t": 40, "b": 60},
        "height": 280,
        "autosize": True,
        "xaxis": {"type": "category", "tickangle": -45},
    }
    return json.dumps(data), json.dumps(layout)


def _plotly_stacked_bar_pct_json(
    quarters: List[str],
    match: List[float],
    mismatch: List[float],
    pending: List[float],
    title: str,
) -> Tuple[str, str]:
    """Stacked bar by quarter as % of quarter (volume-neutral). Each bar sums to 100%."""
    import json
    n = len(quarters)
    match_pct = [0.0] * n
    mismatch_pct = [0.0] * n
    pending_pct = [0.0] * n
    for i in range(n):
        total = (match[i] if i < len(match) else 0) + (mismatch[i] if i < len(mismatch) else 0) + (pending[i] if i < len(pending) else 0)
        if total > 0:
            match_pct[i] = 100.0 * (match[i] if i < len(match) else 0) / total
            mismatch_pct[i] = 100.0 * (mismatch[i] if i < len(mismatch) else 0) / total
            pending_pct[i] = 100.0 * (pending[i] if i < len(pending) else 0) / total
    data = [
        {"type": "bar", "x": quarters, "y": pending_pct, "name": "Pending Assays", "marker": {"color": "#5d6672"}},
        {"type": "bar", "x": quarters, "y": mismatch_pct, "name": "Mismatch", "marker": {"color": "#c9382a"}},
        {"type": "bar", "x": quarters, "y": match_pct, "name": "Match", "marker": {"color": "#2f7d61"}},
    ]
    layout = {
        "title": {"text": title},
        "barmode": "stack",
        "margin": {"t": 40, "b": 60},
        "height": 280,
        "autosize": True,
        "xaxis": {"type": "category", "tickangle": -45},
        "yaxis": {"title": {"text": "% of quarter"}, "range": [0, 100], "ticksuffix": "%"},
    }
    return json.dumps(data), json.dumps(layout)


def _plotly_strat_grouped_bar_json(
    strat_codes: List[str],
    logger_counts: List[int],
    team_counts: List[int],
    title: str,
) -> Tuple[str, str]:
    """Grouped bar: Logger vs Team strat code counts (allows missing values – 0 when not used)."""
    import json
    data = [
        {"type": "bar", "x": strat_codes, "y": logger_counts, "name": "Logger", "marker": {"color": "#0d5b88"}},
        {"type": "bar", "x": strat_codes, "y": team_counts, "name": "Team", "marker": {"color": "#94a3b8"}},
    ]
    layout = {
        "title": {"text": title},
        "barmode": "group",
        "margin": {"t": 40, "b": 80},
        "height": 320,
        "autosize": True,
        "xaxis": {"tickangle": -45},
    }
    return json.dumps(data), json.dumps(layout)


def _plotly_strat_grouped_bar_pct_json(
    strat_codes: List[str],
    logger_counts: List[int],
    team_counts: List[int],
    title: str,
) -> Tuple[str, str]:
    """Grouped bar: Logger vs Team strat code as % of intervals (relative distribution)."""
    import json
    total_logger = max(sum(logger_counts), 1)
    total_team = max(sum(team_counts), 1)
    logger_pct = [x / total_logger * 100 for x in logger_counts]
    team_pct = [x / total_team * 100 for x in team_counts]
    data = [
        {"type": "bar", "x": strat_codes, "y": logger_pct, "name": "Logger", "marker": {"color": "#0d5b88"}},
        {"type": "bar", "x": strat_codes, "y": team_pct, "name": "Team", "marker": {"color": "#94a3b8"}},
    ]
    layout = {
        "title": {"text": title},
        "barmode": "group",
        "margin": {"t": 40, "b": 80},
        "height": 320,
        "autosize": True,
        "xaxis": {"tickangle": -45},
        "yaxis": {"title": {"text": "% of intervals"}},
    }
    return json.dumps(data), json.dumps(layout)


def _plotly_zonation_bar_json(
    categories: List[str], correct: List[int], incorrect: List[int], title: str
) -> Tuple[str, str]:
    """Clustered bar: Correctly vs Incorrectly logged by zonation category."""
    import json
    data = [
        {"type": "bar", "x": categories, "y": correct, "name": "Correct", "marker": {"color": "#2f7d61"}},
        {"type": "bar", "x": categories, "y": incorrect, "name": "Incorrect", "marker": {"color": "#c9382a"}},
    ]
    layout = {
        "title": {"text": title},
        "barmode": "group",
        "margin": {"t": 40, "b": 40},
        "height": 280,
        "autosize": True,
    }
    return json.dumps(data), json.dumps(layout)


def _render_map(map_data: Dict[str, Any]) -> str:
    """
    Render an interactive Leaflet map with drill hole locations.

    Uses OpenTopoMap (free, no API key). Team holes as grey circle markers,
    logger holes as accent-colored markers. Map fills the widget; bounds fit
    all collar points. Coordinates are passed as-is (lat/lng); if collar data
    is in a projected CRS, reprojection to WGS84 must be added when building
    map_data.
    """
    if not map_data.get("has_coords"):
        return (
            "<div class=\"empty\" data-i18n-fr=\"Aucune coordonnee de colliers disponible.\" "
            "data-i18n-en=\"No collar coordinates available for map.\">"
            "Aucune coordonnee de colliers disponible.</div>"
        )

    if map_data.get("warn_projected"):
        return (
            "<div class=\"warning-box\" data-i18n-fr=\"Les coordonnees des colliers semblent etre en CRS projete (ex. UTM). "
            "La carte exige WGS84 (lat/lng). Utilisez des coordonnees en degres decimaux ou reprojetez les donnees.\" "
            "data-i18n-en=\"Collar coordinates appear to be in a projected CRS (e.g. UTM). Map requires WGS84 (lat/lng). "
            "Use decimal-degree coordinates or reproject the collar data.\">"
            "Collar coordinates may be in a projected CRS (e.g. UTM). Map requires WGS84 (lat/lng).</div>"
        )

    points = map_data["points"]
    bounds = map_data.get("bounds")  # [min_lat, min_lng, max_lat, max_lng]
    logger_count = sum(1 for p in points if p.get("is_logger"))
    team_count = len(points) - logger_count

    # Serialize points and bounds for JS (escape for HTML attribute)
    points_json = json.dumps([{"lat": p["lat"], "lng": p["lng"], "hole_id": p.get("hole_id", ""), "is_logger": p.get("is_logger", False)} for p in points])
    bounds_json = json.dumps(bounds) if bounds else "null"

    return f'''
        <div class="map-container map-container-leaflet">
            <div id="leaflet-map" class="map-leaflet-viewport" data-map-points="{html.escape(points_json)}" data-map-bounds="{html.escape(bounds_json)}"></div>
            <div class="map-legend map-legend-leaflet">
                <span><span class="legend-dot logger-dot"></span> <span data-i18n-en="Logger holes" data-i18n-fr="Vos forages">Logger holes</span> ({logger_count})</span>
                <span><span class="legend-dot team-dot"></span> <span data-i18n-en="Team holes" data-i18n-fr="Forages equipe">Team holes</span> ({team_count})</span>
            </div>
        </div>
    '''


def _format_metric(value: Optional[float], unit: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if unit == "%":
        return f"{value:.1f}%"
    if unit == "m":
        return f"{value:,.0f}m"
    return f"{value:.0f}"


def _render_comparison_block(
    comp: Dict[str, Any],
    label_fr: str,
    label_en: str,
    unit: str = "",
) -> str:
    value = comp.get("value")
    median_project = comp.get("median_project")
    median_all = comp.get("median_all")
    project_ok = _within_median_band(value, median_project)
    all_ok = _within_median_band(value, median_all)

    def _render_scope(scope: str, median: Optional[float], ok: Optional[bool]) -> str:
        if median is None or (isinstance(median, float) and np.isnan(median)):
            return ""
        status_fr = "OK" if ok else "A revoir"
        status_en = "OK" if ok else "Review"
        status_class = "ok" if ok else "review"
        width_pct = 0.0
        if median and value is not None:
            width_pct = min(max(value / median * 100, 0), 200)
        return (
            f"<div class=\"comparison-row\" data-scope=\"{scope}\">"
            f"<div class=\"comparison-label\">{_format_metric(value, unit)} vs {_format_metric(median, unit)}</div>"
            f"<div class=\"comparison-bar\"><div class=\"comparison-fill\" style=\"width:{width_pct:.1f}%\"></div></div>"
            f"<div class=\"comparison-status {status_class}\" "
            f"data-i18n-fr=\"{status_fr}\" data-i18n-en=\"{status_en}\">{status_fr}</div>"
            "</div>"
        )

    return (
        "<div class=\"comparison-card\">"
        f"<div class=\"comparison-title\" data-i18n-fr=\"{html.escape(label_fr)}\" "
        f"data-i18n-en=\"{html.escape(label_en)}\">{html.escape(label_fr)}</div>"
        + _render_scope("project", median_project, project_ok)
        + _render_scope("all", median_all, all_ok)
        + "</div>"
    )


def _render_html(
    report: Dict[str, Any],
    intervals: Dict[str, List[Dict[str, Any]]],
    logo_path: Optional[str],
    page_options: Dict[str, bool],
    charts_dir: Optional[str] = None,
) -> str:
    logo_data = _encode_image_base64(logo_path) if logo_path else None
    logger_id = report["meta"]["logger"]
    comment_columns = report["comment_stats"]["comment_columns"]
    if comment_columns:
        comment_columns_html = html.escape(", ".join(comment_columns))
    else:
        comment_columns_html = "<span data-i18n-fr=\"Aucune\" data-i18n-en=\"None\">Aucune</span>"
    comparisons = report["comparisons"]

    overview = report.get("overview", {})
    assay_received = overview.get("assay_received_count", report["summary"].get("assay_intervals", 0))
    assay_outstanding = overview.get("assay_outstanding_count", 0)
    avg_log_m = overview.get("average_logging_interval_m")
    avg_log_str = f"{avg_log_m:.2f}" if avg_log_m is not None and not (isinstance(avg_log_m, float) and np.isnan(avg_log_m)) else "n/a"
    strat_list = overview.get("strat_code_list", [])
    strat_bar_data = strat_list[:20]
    comment_ratio_logging = overview.get("comment_ratio_pct_logging", report.get("comment_coverage", 0))
    if isinstance(comment_ratio_logging, (int, float)) and not np.isnan(comment_ratio_logging):
        comment_ratio_str = f"{comment_ratio_logging:.1f}%"
    else:
        comment_ratio_str = "n/a"

    # Strat codes: one grouped bar (Logger vs Team), allowing missing values (0 when not used)
    team_strat_list = overview.get("team_strat_code_list", [])
    logger_counts_by_code = {str(s["code"]): int(s["count"]) for s in strat_bar_data}
    team_counts_by_code = {str(s["code"]): int(s["count"]) for s in team_strat_list}
    all_codes_set = set(logger_counts_by_code.keys()) | set(team_counts_by_code.keys())
    all_codes_sorted = sorted(all_codes_set, key=lambda c: (logger_counts_by_code.get(c, 0) + team_counts_by_code.get(c, 0)), reverse=True)[:25]
    logger_counts_list = [logger_counts_by_code.get(c, 0) for c in all_codes_sorted]
    team_counts_list = [team_counts_by_code.get(c, 0) for c in all_codes_sorted]
    if all_codes_sorted:
        strat_grouped_data, strat_grouped_layout = _plotly_strat_grouped_bar_pct_json(
            all_codes_sorted, logger_counts_list, team_counts_list,
            "Strat codes: Logger vs Team (% of intervals)"
        )
    else:
        strat_grouped_data, strat_grouped_layout = "[]", "{}"

    date_from_str = report["meta"].get("date_from") or ""
    date_to_str = report["meta"].get("date_to") or ""
    overview_hero_date = f"From: {date_from_str or '-'} To: {date_to_str or '-'}"
    overview_section = f"""
        <section class="tab-panel" data-tab="overview">
            <div class="overview-hero">
                <h1 class="overview-hero-title" data-i18n-fr="Revue de logging" data-i18n-en="Logging Review">Logging Review</h1>
                <p class="overview-hero-logger">{html.escape(report["meta"]["logger"])}</p>
                <p class="overview-hero-date">{html.escape(overview_hero_date)}</p>
            </div>
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label" data-i18n-fr="Forages uniques" data-i18n-en="Unique holes">Forages uniques</div>
                    <div class="kpi-value">{report["summary"]["unique_holes"]}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label" data-i18n-fr="Codes strat" data-i18n-en="Strat codes">Codes strat</div>
                    <div class="kpi-value">{report["summary"]["strat_codes"]}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label" data-i18n-fr="Profondeur totale (m)" data-i18n-en="Total depth (m)">Profondeur totale (m)</div>
                    <div class="kpi-value">{_format_metric(report["summary"].get("total_depth_m"), "m")}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label" data-i18n-fr="Essais recus" data-i18n-en="Assays received">Essais recus</div>
                    <div class="kpi-value">{assay_received}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label" data-i18n-fr="Essais en attente" data-i18n-en="Pending Assays">Essais en attente</div>
                    <div class="kpi-value">{assay_outstanding}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label" data-i18n-fr="Intervalle moyen (m)" data-i18n-en="Avg logging interval (m)">Intervalle moyen (m)</div>
                    <div class="kpi-value">{avg_log_str}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label" data-i18n-fr="Intervalles avec commentaire (%)" data-i18n-en="Intervals with comment (%)">Intervalles avec commentaire (%)</div>
                    <div class="kpi-value">{comment_ratio_str}</div>
                </div>
            </div>
            <div class="two-panel overview-two-panel">
                <div class="panel-card">
                    <h3 data-i18n-fr="Carte des colliers" data-i18n-en="Collar map">Carte des colliers</h3>
                    {_render_map(report["map"])}
                </div>
                <div class="panel-card">
                    <h3 data-i18n-fr="Codes strat: vous vs equipe" data-i18n-en="Strat codes: you vs team">Codes strat: vous vs equipe</h3>
                    <div id="strat-grouped-bar" class="plotly-chart" data-plotly-data="{html.escape(strat_grouped_data)}" data-plotly-layout="{html.escape(strat_grouped_layout)}"></div>
                </div>
            </div>
        </section>
    """

    cl = report.get("comment_stats_logging", {})
    total_rows_log = cl.get("total_rows", 0)
    rows_with_comment = cl.get("rows_with_comment", 0)
    rows_without_comment = cl.get("rows_without_comment", total_rows_log)
    comment_ratio_pct_log = cl.get("comment_ratio_pct", 0) if cl else report.get("comment_coverage", 0)
    comment_bar_data = [rows_without_comment, rows_with_comment]
    comment_bar_max = max(comment_bar_data) or 1
    comment_bar_pcts = [x / comment_bar_max * 100 for x in comment_bar_data]

    # Generate matplotlib bar chart for comment statistics (like old QAQC scripts)
    avg_comment_length = cl.get("avg_comment_length", 0) if cl else 0
    comment_chart_save_path = os.path.join(charts_dir, "comment_statistics.png") if charts_dir else None
    comment_bar_chart_html = _render_comment_bar_chart(total_rows_log, rows_with_comment, avg_comment_length, save_path=comment_chart_save_path)

    comment_section = f"""
        <section class="tab-panel" data-tab="comments">
            <div class="panel-header">
                <h2 data-i18n-fr="Utilisation des commentaires" data-i18n-en="Comment usage">Utilisation des commentaires</h2>
            </div>
            <div class="two-panel">
                <div class="panel-card">
                    <h3 data-i18n-fr="Nuage de mots" data-i18n-en="Word cloud">Nuage de mots</h3>
                    {_render_wordcloud(report["wordcloud"], save_path=os.path.join(charts_dir, "wordcloud.png") if charts_dir else None)}
                </div>
                <div class="panel-card">
                    <h3 data-i18n-fr="Statistiques de commentaires" data-i18n-en="Comment Statistics">Statistiques de commentaires</h3>
                    {comment_bar_chart_html}
                </div>
            </div>
            <div class="info-box">
                <h4 data-i18n-fr="Comment cette mesure est calculee" data-i18n-en="How this is calculated">Comment cette mesure est calculee</h4>
                <ul>
                    <li data-i18n-fr="Total = tous les intervalles de logging. NULL/manquant = Sans commentaire."
                        data-i18n-en="Total = all logging intervals. NULL/missing = No Comment.">
                        Total = tous les intervalles de logging. NULL/manquant = Sans commentaire.
                    </li>
                    <li data-i18n-fr="Le nuage de mots utilise les mots les plus frequents apres suppression des mots courants."
                        data-i18n-en="The word cloud uses the most frequent words after removing common stopwords.">
                        Le nuage de mots utilise les mots les plus frequents apres suppression des mots courants.
                    </li>
                </ul>
            </div>
            <div class="notes-box">
                <label data-i18n-fr="Notes du reviseur" data-i18n-en="Reviewer notes">Notes du reviseur</label>
                <textarea data-note-id="{logger_id}::comments" placeholder="Ajouter des notes..."></textarea>
            </div>
        </section>
    """

    min_acc = report.get("mineralisation", {})
    acc_counts = min_acc.get("accuracy_counts", {})
    acc_labels = []
    acc_values = []
    for k in ["Match", "Mismatch", "Pending Assays"]:
        if acc_counts.get(k, 0) > 0:
            acc_labels.append(k)
            acc_values.append(int(acc_counts[k]))
    mineral_pie_user_data, mineral_pie_user_layout = _plotly_pie_json(
        acc_labels or ["N/A"], acc_values or [0],
        "Votre precision (Overall accuracy)"
    ) if (acc_labels or acc_values) else ("[]", "{}")
    q_user = min_acc.get("quarterly_user", pd.DataFrame())
    if not q_user.empty and "Quarter" in q_user.columns:
        # Format as categorical labels (e.g. Q1 2025) so single-bin x-axis is readable
        quarters_user = []
        for p in q_user["Quarter"]:
            try:
                per = getattr(p, "quarter", None) or ((pd.Timestamp(p).month - 1) // 3 + 1)
                yr = getattr(p, "year", None) or pd.Timestamp(p).year
                quarters_user.append(f"Q{per} {yr}")
            except Exception:
                quarters_user.append(str(p))
        match_user = q_user.get("Match", pd.Series([0] * len(q_user))).fillna(0).tolist()
        mismatch_user = q_user.get("Mismatch", pd.Series([0] * len(q_user))).fillna(0).tolist()
        pending_user = q_user.get("Pending Assays", pd.Series([0] * len(q_user))).fillna(0).tolist()
        mineral_bar_user_data, mineral_bar_user_layout = _plotly_stacked_bar_pct_json(
            quarters_user, match_user, mismatch_user, pending_user,
            "Votre precision par trimestre (% of quarter)"
        )
    else:
        mineral_bar_user_data, mineral_bar_user_layout = "[]", "{}"
    acc_team = min_acc.get("accuracy_counts_team", {})
    acc_labels_t = []
    acc_values_t = []
    for k in ["Match", "Mismatch", "Pending Assays"]:
        if acc_team.get(k, 0) > 0:
            acc_labels_t.append(k)
            acc_values_t.append(int(acc_team[k]))
    mineral_pie_team_data, mineral_pie_team_layout = _plotly_pie_json(
        acc_labels_t or ["N/A"], acc_values_t or [0],
        "Equipe (Overall accuracy)"
    ) if (acc_labels_t or acc_values_t) else ("[]", "{}")
    q_team = min_acc.get("quarterly_team", pd.DataFrame())
    if not q_team.empty and "Quarter" in q_team.columns:
        quarters_team = []
        for p in q_team["Quarter"]:
            try:
                per = getattr(p, "quarter", None) or ((pd.Timestamp(p).month - 1) // 3 + 1)
                yr = getattr(p, "year", None) or pd.Timestamp(p).year
                quarters_team.append(f"Q{per} {yr}")
            except Exception:
                quarters_team.append(str(p))
        match_team = q_team.get("Match", pd.Series([0] * len(q_team))).fillna(0).tolist()
        mismatch_team = q_team.get("Mismatch", pd.Series([0] * len(q_team))).fillna(0).tolist()
        pending_team = q_team.get("Pending Assays", pd.Series([0] * len(q_team))).fillna(0).tolist()
        mineral_bar_team_data, mineral_bar_team_layout = _plotly_stacked_bar_pct_json(
            quarters_team, match_team, mismatch_team, pending_team,
            "Equipe par trimestre (% of quarter)"
        )
    else:
        mineral_bar_team_data, mineral_bar_team_layout = "[]", "{}"

    mineral_section = f"""
        <section class="tab-panel" data-tab="mineralisation">
            <div class="panel-header">
                <h2 data-i18n-fr="Precision de la mineralisation" data-i18n-en="Mineralisation accuracy">Precision de la mineralisation</h2>
            </div>
            <div class="two-panel charts-panel">
                <div class="panel-card">
                    <h3 data-i18n-fr="Vos donnees" data-i18n-en="Your data">Vos donnees</h3>
                    <div id="mineral-pie-user" class="plotly-chart" data-plotly-data="{html.escape(mineral_pie_user_data)}" data-plotly-layout="{html.escape(mineral_pie_user_layout)}"></div>
                    <div id="mineral-bar-user" class="plotly-chart" data-plotly-data="{html.escape(mineral_bar_user_data)}" data-plotly-layout="{html.escape(mineral_bar_user_layout)}"></div>
                </div>
                <div class="panel-card">
                    <h3 data-i18n-fr="Tous les loggers" data-i18n-en="All loggers">Tous les loggers</h3>
                    <div id="mineral-pie-team" class="plotly-chart" data-plotly-data="{html.escape(mineral_pie_team_data)}" data-plotly-layout="{html.escape(mineral_pie_team_layout)}"></div>
                    <div id="mineral-bar-team" class="plotly-chart" data-plotly-data="{html.escape(mineral_bar_team_data)}" data-plotly-layout="{html.escape(mineral_bar_team_layout)}"></div>
                </div>
            </div>
            <div class="info-box">
                <h4 data-i18n-fr="Note methodologique" data-i18n-en="Methodology note">Note methodologique</h4>
                <ul>
                    <li data-i18n-html-fr="<strong>Match</strong> = Le logging est coherent avec les essais (gangue &lt;15% loggee ET essai montre mineralise, OU gangue &gt;=15% loggee ET essai montre non-mineralise)"
                        data-i18n-html-en="<strong>Match</strong> = Logging is consistent with assays (gangue &lt;15% logged AND assay shows mineralised, OR gangue &gt;=15% logged AND assay shows unmineralised)">
                        <strong>Match</strong> = Logging is consistent with assays (gangue &lt;15% logged AND assay shows mineralised, OR gangue &gt;=15% logged AND assay shows unmineralised)
                    </li>
                    <li data-i18n-html-fr="<strong>Mismatch</strong> = Le logging est incoherent avec les essais (ex. gangue &lt;15% loggee mais essai montre non-mineralise)"
                        data-i18n-html-en="<strong>Mismatch</strong> = Logging inconsistent with assays (e.g. gangue &lt;15% logged but assay shows unmineralised)">
                        <strong>Mismatch</strong> = Logging inconsistent with assays (e.g. gangue &lt;15% logged but assay shows unmineralised)
                    </li>
                    <li data-i18n-html-fr="<strong>Pending Assays</strong> = Pas de donnees d'essai disponibles pour cet intervalle"
                        data-i18n-html-en="<strong>Pending Assays</strong> = No assay data available for this interval">
                        <strong>Pending Assays</strong> = No assay data available for this interval
                    </li>
                    <li data-i18n-html-fr="<strong>Mineralise (essai)</strong> = Fe &gt;50%, SiO2 &lt;10%, Al2O3 &lt;5%"
                        data-i18n-html-en="<strong>Mineralised (assay)</strong> = Fe &gt;50%, SiO2 &lt;10%, Al2O3 &lt;5%">
                        <strong>Mineralised (assay)</strong> = Fe &gt;50%, SiO2 &lt;10%, Al2O3 &lt;5%
                    </li>
                    <li data-i18n-html-fr="<strong>Leached (essai)</strong> = Fe &gt;50% mais SiO2 10-15% ou Al2O3 5-10% (presque mineralise)"
                        data-i18n-html-en="<strong>Leached (assay)</strong> = Fe &gt;50% but SiO2 10-15% or Al2O3 5-10% (almost mineralised)">
                        <strong>Leached (assay)</strong> = Fe &gt;50% but SiO2 10-15% or Al2O3 5-10% (almost mineralised)
                    </li>
                    <li data-i18n-html-fr="<strong>Non-mineralise (essai)</strong> = Ne repond pas aux criteres de mineralisation"
                        data-i18n-html-en="<strong>Unmineralised (assay)</strong> = Does not meet mineralisation criteria">
                        <strong>Unmineralised (assay)</strong> = Does not meet mineralisation criteria
                    </li>
                    <li data-i18n-html-fr="<strong>Graphiques par trimestre</strong> = Chaque barre represente 100% du trimestre (Match + Mismatch + Pending). Volume neutre."
                        data-i18n-html-en="<strong>Quarterly charts</strong> = Each bar is 100% of that quarter (Match + Mismatch + Pending). Volume-neutral.">
                        <strong>Quarterly charts</strong> = Each bar is 100% of that quarter (Match + Mismatch + Pending). Volume-neutral.
                    </li>
                </ul>
            </div>
            <div class="evidence-section">
                <h3 data-i18n-fr="Tableau de preuves - Ecarts de mineralisation" data-i18n-en="Evidence Table - Mineralisation Mismatches">Evidence Table - Mineralisation Mismatches</h3>
                <p class="mineral-mismatch-stats">
                    <span data-i18n-fr="Mismatches:" data-i18n-en="Mismatches:">Mismatches:</span>
                    <strong>{report["mineralisation"].get("mismatch_low_count", 0)}</strong>
                    <span data-i18n-fr="Low (borderline)" data-i18n-en="Low (borderline)">Low (borderline)</span>,
                    <strong>{report["mineralisation"].get("mismatch_high_count", 0)}</strong>
                    <span data-i18n-fr="High" data-i18n-en="High">High</span>
                    <span class="stats-note" data-i18n-fr="(borderline = assay suggests Leached; ne pas penaliser les erreurs limite)" data-i18n-en="(borderline = assay suggests Leached; do not penalise borderline errors)">(borderline = assay suggests Leached; do not penalise borderline errors)</span>
                </p>
                {_render_mineralisation_evidence_table(sorted(report["mineralisation"].get("mismatch_intervals", []), key=lambda x: 0 if x.get("significance") == "High" else 1), logger_id)}
            </div>
            <div class="notes-box">
                <label data-i18n-fr="Notes du reviseur" data-i18n-en="Reviewer notes">Notes du reviseur</label>
                <textarea data-note-id="{logger_id}::mineralisation" placeholder="Ajouter des notes..."></textarea>
            </div>
        </section>
    """

    pz = report.get("profile_zonation", {})
    z_cats = ["Un", "Le", "De", "Hy", "Pr"]
    z_correct = [pz.get("correct_counts", {}).get(c, 0) for c in z_cats]
    z_mismatch = [pz.get("mismatch_counts", {}).get(c, 0) for c in z_cats]

    # Check if there's actually any zonation data
    total_zonation_data = sum(z_correct) + sum(z_mismatch)
    has_zonation_data = total_zonation_data > 0

    zonation_bar_data, zonation_bar_layout = _plotly_zonation_bar_json(
        z_cats, z_correct, z_mismatch,
        "Profile Zonation Logging Accuracy by Category"
    )
    zonation_mismatch_rows = pz.get("mismatch_rows", [])
    zonation_evidence_html = _render_zonation_evidence_table(zonation_mismatch_rows, logger_id)
    zonation_based_on_note = (
        "<p class=\"zonation-based-on-note\" data-i18n-en=\"Based on mineral logging codes only.\" "
        "data-i18n-fr=\"Basé sur les codes de minéraux uniquement.\">Based on mineral logging codes only.</p>"
    )

    # Show a warning if no zonation data found
    zonation_data_warning = ""
    if not has_zonation_data:
        zonation_data_warning = '''<div class="warning-box">
            <h4 data-i18n-fr="⚠️ Donnees de zonation non disponibles" data-i18n-en="⚠️ Zonation data not available">⚠️ Donnees de zonation non disponibles</h4>
            <p data-i18n-fr="Les colonnes de zonation (BestProfileZonation_D, Total Gangue Logged, De % Logged, Hy % Logged, Pr % Logged) n'ont pas ete trouvees ou sont vides dans les donnees."
               data-i18n-en="Zonation columns (BestProfileZonation_D, Total Gangue Logged, De % Logged, Hy % Logged, Pr % Logged) were not found or are empty in the data.">
                Les colonnes de zonation n'ont pas ete trouvees ou sont vides dans les donnees.
            </p>
        </div>'''

    profile_section = f"""
        <section class="tab-panel" data-tab="profile">
            <div class="panel-header">
                <h2 data-i18n-fr="Zonation de profil" data-i18n-en="Profile zonation">Zonation de profil</h2>
            </div>
            {zonation_data_warning}
            <div class="panel-card">
                <h3 data-i18n-fr="Precision par categorie (Correct vs Incorrect)" data-i18n-en="Accuracy by category">Precision par categorie</h3>
                <div id="zonation-bar" class="plotly-chart" data-plotly-data="{html.escape(zonation_bar_data)}" data-plotly-layout="{html.escape(zonation_bar_layout)}"></div>
            </div>
            <div class="evidence-section">
                <h3 data-i18n-fr="Tableau de preuves" data-i18n-en="Evidence Table">Evidence Table</h3>
                <p class="zonation-based-on-note" data-i18n-en="Based on mineral logging codes only." data-i18n-fr="Basé sur les codes de minéraux uniquement.">Based on mineral logging codes only.</p>
                <div class="intervals-section">
                    {zonation_evidence_html}
                </div>
            </div>
            <div class="info-box rules-box">
                <h4 data-i18n-fr="Regles de validation de la zonation" data-i18n-en="Zonation validation rules">Regles de validation</h4>
                <table class="rules-table">
                    <thead>
                        <tr>
                            <th data-i18n-fr="Zonation" data-i18n-en="Zonation">Zonation</th>
                            <th data-i18n-fr="Gangue totale attendue" data-i18n-en="Expected total gangue">Gangue totale attendue</th>
                            <th data-i18n-fr="Mineral dominant" data-i18n-en="Dominant mineral">Mineral dominant</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr><td><strong>Un</strong> (Unmineralised)</td><td>16-100%</td><td>-</td></tr>
                        <tr><td><strong>Le</strong> (Leached)</td><td>11-15%</td><td>-</td></tr>
                        <tr><td><strong>De</strong> (Detrital)</td><td>0-10%</td><td><span data-i18n-en="De minerals are dominant" data-i18n-fr="De doit etre dominant">De minerals are dominant</span></td></tr>
                        <tr><td><strong>Hy</strong> (Hydrated)</td><td>0-10%</td><td><span data-i18n-en="Hy minerals are dominant" data-i18n-fr="Hy doit etre dominant">Hy minerals are dominant</span></td></tr>
                        <tr><td><strong>Pr</strong> (Primary)</td><td>0-10%</td><td><span data-i18n-en="Pr minerals are dominant" data-i18n-fr="Pr doit etre dominant">Pr minerals are dominant</span></td></tr>
                    </tbody>
                </table>
                <p class="zonation-based-on-note" data-i18n-en="Based on mineral logging codes only." data-i18n-fr="Basé sur les codes de minéraux uniquement.">Based on mineral logging codes only.</p>
            </div>
            <div class="notes-box">
                <label data-i18n-fr="Notes du reviseur" data-i18n-en="Reviewer notes">Notes du reviseur</label>
                <textarea data-note-id="{logger_id}::profile" placeholder="Ajouter des notes..."></textarea>
            </div>
        </section>
    """

    logging_detail_issue_types = report.get("logging_detail_issue_types", [])
    logging_detail_tables_html = []
    for it in logging_detail_issue_types:
        key = it.get("key", "")
        label_fr = it.get("label_fr", key)
        label_en = it.get("label_en", key)
        rules_fr = it.get("rules_fr", "")
        rules_en = it.get("rules_en", "")
        issue_intervals = (intervals.get("logging_detail") or {}).get(key, [])
        logging_detail_tables_html.append(f"""
            <div class="panel-card">
                <h3 data-i18n-fr="{html.escape(label_fr)}" data-i18n-en="{html.escape(label_en)}">{html.escape(label_en)}</h3>
                <div class="rules-box">
                    <p data-i18n-fr="{html.escape(rules_fr)}" data-i18n-en="{html.escape(rules_en)}">{html.escape(rules_en)}</p>
                </div>
                <div class="intervals-section">
                    {_render_logging_detail_evidence_table(issue_intervals, logger_id, key)}
                </div>
            </div>
        """)

    logging_detail_section = f"""
        <section class="tab-panel" data-tab="logging-detail">
            <div class="panel-header">
                <h2 data-i18n-fr="Precision du detail de logging" data-i18n-en="Logging detail accuracy">Logging detail accuracy</h2>
            </div>
            <div class="panel-card">
                <h3 data-i18n-fr="Resume (fines)" data-i18n-en="Summary (fines)">Summary (fines)</h3>
                <ul class="summary-list">
                    {''.join(f"<li>{html.escape(line)}</li>" for line in report.get("fines_summary", []))}
                </ul>
            </div>
            {''.join(logging_detail_tables_html)}
            <div class="notes-box">
                <label data-i18n-fr="Notes du reviseur" data-i18n-en="Reviewer notes">Notes du reviseur</label>
                <textarea data-note-id="{logger_id}::logging-detail" placeholder="Ajouter des notes..."></textarea>
            </div>
        </section>
    """

    gkpis = report.get("grouping_kpis", {})
    avg_grp = gkpis.get("avg_group_interval_m")
    max_grp = gkpis.get("max_group_interval_m")
    avg_grp_str = f"{avg_grp:.2f}" if avg_grp is not None and not (isinstance(avg_grp, float) and np.isnan(avg_grp)) else "n/a"
    max_grp_str = f"{max_grp:.2f}" if max_grp is not None and not (isinstance(max_grp, float) and np.isnan(max_grp)) else "n/a"
    grouping_groups = intervals.get("grouping", [])

    grouping_section = f"""
        <section class="tab-panel" data-tab="grouping">
            <div class="panel-header">
                <h2 data-i18n-fr="Precision du regroupement" data-i18n-en="Grouping accuracy">Precision du regroupement</h2>
            </div>
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label" data-i18n-fr="Intervalle moyen de groupe (m)" data-i18n-en="Avg group interval (m)">Intervalle moyen (m)</div>
                    <div class="kpi-value">{avg_grp_str}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label" data-i18n-fr="Intervalle max de groupe (m)" data-i18n-en="Max group interval (m)">Intervalle max (m)</div>
                    <div class="kpi-value">{max_grp_str}</div>
                </div>
            </div>
            <div class="panel-card">
                <h3 data-i18n-fr="Resume simplifie" data-i18n-en="Simplified summary">Resume simplifie</h3>
                <ul class="summary-list">
                    {''.join(f"<li>{html.escape(line)}</li>" for line in report["grouping_summary"])}
                </ul>
            </div>
            <div class="info-box rules-box">
                <h4 data-i18n-fr="Regles de regroupement" data-i18n-en="Grouping rules">Regles de regroupement</h4>
                <p data-i18n-fr="Colonnes de regroupement utilisees:" data-i18n-en="Grouping columns used:">
                    Grouping columns used: <code>{html.escape(", ".join(report.get("grouping_columns_used", [])))}</code>
                </p>
                <p data-i18n-fr="Longueur d'intervalle = max(profondeur_to) − min(profondeur_from) par groupe. CV = std/mean × 100 pour chaque element; les groupes avec CV > 100% sont signales. Si les donnees incluent les memes colonnes que combine_intervals (ex. Prospect_D, StratSum, Min_*_pct, LithComments), la logique est alignee; sinon on utilise le sous-ensemble present."
                    data-i18n-en="Interval length = max(depth_to) − min(depth_from) per group. CV = std/mean × 100 for each assay element; groups with CV > 100% are flagged. When data includes the same grouping columns as Logging Review combine_intervals (e.g. Prospect_D, StratSum, Min_*_pct, LithComments), grouping matches that logic; otherwise the subset present is used.">
                    Interval length = max(depth_to) − min(depth_from) per group. CV = std/mean × 100 for each assay element; groups with CV &gt; 100% are flagged. When data includes the same grouping columns as combine_intervals, grouping matches that logic; otherwise the subset present is used.
                </p>
            </div>
            <div class="intervals-section">
                <h3 data-i18n-fr="Groupes a revoir (CV eleve Fe, SiO2, Al2O3, MgO, CaO, S)" data-i18n-en="Groups for review (high CV)">Groupes a revoir</h3>
                {_render_grouping_groups(grouping_groups, logger_id)}
            </div>
            <div class="notes-box">
                <label data-i18n-fr="Notes du reviseur" data-i18n-en="Reviewer notes">Notes du reviseur</label>
                <textarea data-note-id="{logger_id}::grouping" placeholder="Ajouter des notes..."></textarea>
            </div>
        </section>
    """

    okpis = report.get("outlier_kpis", {})
    total_misclassified = okpis.get("total_misclassified", len(report.get("outliers", [])))
    displayed_count = okpis.get("displayed_count", total_misclassified)
    common_errors = okpis.get("common_errors", [])
    common_errors_html = ""
    if common_errors:
        common_errors_html = "<ul class=\"common-errors-list\">"
        for e in common_errors[:10]:
            common_errors_html += f"<li>Classifie comme <strong>{html.escape(str(e.get('recorded_as', '')))}</strong>: {e.get('count', 0)} intervalle(s)</li>"
        common_errors_html += "</ul>"
    else:
        common_errors_html = "<div class=\"empty\" data-i18n-fr=\"Aucune erreur commune.\" data-i18n-en=\"No common errors.\">Aucune.</div>"

    # Show how many are displayed if limited
    display_note = ""
    if displayed_count < total_misclassified:
        display_note = f" <span class=\"display-note\" data-i18n-fr=\"(montrant {displayed_count})\" data-i18n-en=\"(showing {displayed_count})\">(montrant {displayed_count})</span>"

    outlier_section = f"""
        <section class="tab-panel" data-tab="outliers">
            <div class="panel-header">
                <h2 data-i18n-fr="Detection des anomalies" data-i18n-en="Outlier detection">Detection des anomalies</h2>
            </div>
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label" data-i18n-fr="Intervalles probablement mal classes" data-i18n-en="Intervals likely misclassified">Mal classes</div>
                    <div class="kpi-value">{total_misclassified}{display_note}</div>
                </div>
            </div>
            <div class="panel-card">
                <h3 data-i18n-fr="Erreurs les plus frequentes" data-i18n-en="Most common errors">Erreurs les plus frequentes</h3>
                {common_errors_html}
            </div>
            <div class="intervals-section">
                <h3 data-i18n-fr="Intervalles a revoir (Enregistre vs Plus probable / Raison)" data-i18n-en="Intervals for review (Recorded vs Most likely / Reason)">Intervalles a revoir</h3>
                {_render_outlier_table(intervals["outliers"], logger_id)}
            </div>
            <div class="notes-box">
                <label data-i18n-fr="Notes du reviseur" data-i18n-en="Reviewer notes">Notes du reviseur</label>
                <textarea data-note-id="{logger_id}::outliers" placeholder="Ajouter des notes..."></textarea>
            </div>
        </section>
    """

    tabs = [
        ("overview", "Vue d'ensemble", "Overview"),
        ("comments", "Commentaires", "Comments"),
        ("mineralisation", "Mineralisation", "Mineralisation"),
        ("profile", "Zonation", "Zonation"),
        ("logging-detail", "Precision du detail", "Logging detail accuracy"),
        ("grouping", "Regroupement", "Grouping"),
        ("outliers", "Anomalies", "Outliers"),
    ]

    include_overview = page_options.get("summary_stats", True) or page_options.get("cover", True)
    include_comments = page_options.get("comment_stats", True)
    include_logging_detail = page_options.get("fines_accuracy", True)
    include_grouping = page_options.get("grouping_accuracy", True)
    include_outliers = page_options.get("outliers", True)

    tab_visibility = {
        "overview": include_overview,
        "comments": include_comments,
        "mineralisation": True,
        "profile": True,
        "logging-detail": include_logging_detail,
        "grouping": include_grouping,
        "outliers": include_outliers,
    }

    tab_buttons = []
    for tab_id, fr, en in tabs:
        if not tab_visibility.get(tab_id, True):
            continue
        tab_buttons.append(
            f"<button class=\"tab-button\" data-tab-button=\"{tab_id}\" "
            f"data-i18n-fr=\"{html.escape(fr)}\" data-i18n-en=\"{html.escape(en)}\">{html.escape(fr)}</button>"
        )

    html_content = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Logging Review Report</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {{
            --bg: #f6f7f8;
            --panel: #ffffff;
            --ink: #1d2025;
            --muted: #5d6672;
            --accent: #0d5b88;
            --accent-warm: #c9802a;
            --border: #d8dde3;
            --success: #2f7d61;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            font-family: "Segoe UI", "Georgia", serif;
            color: var(--ink);
            background: var(--bg);
        }}
        .app {{
            display: flex;
            min-height: 100vh;
        }}
        .sidebar {{
            width: 240px;
            background: #101820;
            color: #ffffff;
            padding: 20px 16px;
            display: flex;
            flex-direction: column;
            gap: 14px;
        }}
        .logo {{
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            gap: 8px;
        }}
        .logo img {{
            max-width: 140px;
            height: auto;
        }}
        .lang-toggle {{
            display: flex;
            gap: 8px;
        }}
        .lang-toggle button {{
            background: transparent;
            border: 1px solid #ffffff;
            color: #ffffff;
            padding: 4px 8px;
            cursor: pointer;
            border-radius: 4px;
            font-size: 12px;
        }}
        .lang-toggle button.active {{
            background: #ffffff;
            color: #101820;
        }}
        .tab-button {{
            width: 100%;
            text-align: left;
            background: #1b2733;
            color: #ffffff;
            border: none;
            padding: 12px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
        }}
        .tab-button.active {{
            background: var(--accent);
        }}
        .content {{
            flex: 1;
            padding: 24px;
        }}
        .overview-hero {{
            text-align: center;
            margin-bottom: 28px;
            padding: 24px 16px;
            border-bottom: 2px solid var(--border);
        }}
        .overview-hero-title {{
            margin: 0 0 8px 0;
            font-size: 32px;
            font-weight: 700;
            letter-spacing: 0.02em;
        }}
        .overview-hero-logger {{
            margin: 0 0 4px 0;
            font-size: 20px;
            font-weight: 600;
            color: var(--accent);
        }}
        .overview-hero-date {{
            margin: 0;
            font-size: 16px;
            color: var(--muted);
        }}
        .panel-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }}
        .panel-header h2 {{
            margin: 0;
            font-size: 22px;
        }}
        .header-meta {{
            display: flex;
            gap: 16px;
            color: var(--muted);
            font-size: 13px;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            margin-bottom: 16px;
        }}
        .kpi-card {{
            background: var(--panel);
            padding: 14px;
            border-radius: 10px;
            border: 1px solid var(--border);
        }}
        .kpi-label {{
            color: var(--muted);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .kpi-value {{
            font-size: 20px;
            font-weight: 700;
            margin-top: 6px;
        }}
        .two-panel {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
        }}
        .overview-two-panel {{
            min-height: 65vh;
            align-items: stretch;
        }}
        .overview-two-panel .panel-card {{
            display: flex;
            flex-direction: column;
            min-height: 0;
        }}
        .overview-two-panel .map-container {{
            flex: 1;
            min-height: 400px;
        }}
        .overview-two-panel .map-container-leaflet {{
            min-height: 400px;
        }}
        .overview-two-panel .map-leaflet-viewport {{
            min-height: 380px;
        }}
        .overview-two-panel .plotly-chart {{
            flex: 1;
            min-height: 400px;
        }}
        .panel-card {{
            background: var(--panel);
            border-radius: 12px;
            border: 1px solid var(--border);
            padding: 16px;
        }}
        .panel-card h3 {{
            margin-top: 0;
            font-size: 16px;
        }}
        .comparison-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 12px;
            margin-bottom: 16px;
        }}
        .comparison-card {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 12px;
        }}
        .comparison-title {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--muted);
            margin-bottom: 8px;
        }}
        .comparison-row {{
            display: grid;
            grid-template-columns: 1fr 2fr auto;
            gap: 8px;
            align-items: center;
            margin-bottom: 6px;
        }}
        .comparison-bar {{
            background: #e5eaef;
            height: 8px;
            border-radius: 6px;
            overflow: hidden;
        }}
        .comparison-fill {{
            height: 8px;
            background: var(--accent);
        }}
        .comparison-status {{
            font-size: 11px;
            padding: 2px 6px;
            border-radius: 6px;
            text-transform: uppercase;
        }}
        .comparison-status.ok {{
            background: #e2f3ec;
            color: var(--success);
        }}
        .comparison-status.review {{
            background: #fce8d6;
            color: var(--accent-warm);
        }}
        .stat-line {{
            display: flex;
            justify-content: space-between;
            margin-top: 8px;
        }}
        .info-box {{
            margin-top: 16px;
            background: #eef4f8;
            border-left: 4px solid var(--accent);
            padding: 12px 16px;
            border-radius: 8px;
        }}
        .info-box ul {{
            margin: 8px 0 0 18px;
        }}
        .notes-box {{
            margin-top: 16px;
            background: var(--panel);
            border: 1px solid var(--border);
            padding: 12px;
            border-radius: 8px;
        }}
        .notes-box textarea {{
            width: 100%;
            min-height: 80px;
            border: 1px solid var(--border);
            padding: 8px;
            border-radius: 6px;
            font-family: inherit;
        }}
        .wordcloud-image {{
            text-align: center;
            margin: 16px 0;
        }}
        .wordcloud-image img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .wordcloud {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}
        .wordcloud .word {{
            background: #f0f4f7;
            padding: 4px 8px;
            border-radius: 6px;
        }}
        .bar-track {{
            background: #e5eaef;
            height: 10px;
            border-radius: 8px;
            overflow: hidden;
        }}
        .bar-fill {{
            height: 10px;
            background: var(--accent);
        }}
        .bar-block {{
            margin-top: 12px;
        }}
        .interval-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
            font-size: 13px;
        }}
        .interval-table th, .interval-table td {{
            padding: 8px;
            border-bottom: 1px solid var(--border);
            text-align: left;
        }}
        .interval-table th {{
            background: #f3f5f7;
        }}
        .image-cell img {{
            width: 120px;
            height: auto;
            border-radius: 6px;
            border: 1px solid var(--border);
        }}
        .fines-image-cell img.fines-image, .fines-image {{
            width: 280px;
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            border: 1px solid var(--border);
        }}
        .fines-placeholder {{
            width: 280px;
            min-height: 140px;
        }}
        .image-placeholder-small {{
            width: 80px;
            height: 50px;
            min-width: 80px;
            min-height: 50px;
            background: #f0f2f4;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--muted);
            font-size: 11px;
            border-radius: 4px;
        }}
        .image-placeholder {{
            width: 120px;
            height: 80px;
            background: #f0f2f4;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--muted);
            font-size: 12px;
            border-radius: 6px;
        }}
        /* Rotated images - 90 degrees to save horizontal space */
        .rotated-image-cell {{
            padding: 4px !important;
            text-align: center;
        }}
        .rotated-image-cell img.rotated-image {{
            transform: rotate(90deg);
            width: 100px;
            height: auto;
            margin: 20px 0;
            border-radius: 4px;
            border: 1px solid var(--border);
        }}
        /* Geochem table - compact tabulated display */
        .geochem-table {{
            border-collapse: collapse;
            font-size: 11px;
            margin: 0;
        }}
        .geochem-table tr {{
            border: none;
        }}
        .geochem-table td {{
            padding: 2px 6px;
            border: none;
        }}
        .geochem-label {{
            font-weight: 600;
            color: var(--muted);
            text-align: right;
        }}
        .geochem-value {{
            text-align: left;
            font-family: monospace;
        }}
        .geochem-na {{
            color: var(--muted);
        }}
        .most-likely-cell {{
            font-weight: 600;
            color: var(--accent);
        }}
        .strat-bar-chart {{ margin-top: 8px; }}
        .strat-bar-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 13px; }}
        .strat-code {{ min-width: 80px; }}
        .strat-bar-track {{ flex: 1; background: #e5eaef; height: 14px; border-radius: 8px; overflow: hidden; }}
        .strat-bar-fill {{ height: 14px; background: var(--accent); border-radius: 8px; }}
        .strat-bar-fill.team-fill {{ background: #94a3b8; }}
        .panel-card.full-width {{ width: 100%; margin-bottom: 16px; }}
        .strat-count {{ min-width: 36px; text-align: right; }}
        .comment-bar-chart {{ margin-top: 12px; }}
        .comment-bar-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }}
        .comment-bar-label {{ min-width: 140px; }}
        .comment-bar-value {{ min-width: 40px; }}
        .bar-fill-red {{ background: #c9382a !important; }}
        .bar-fill-green {{ background: var(--success) !important; }}
        .comment-chart-image {{ text-align: center; margin: 12px 0; }}
        .comment-chart-image img {{ max-width: 100%; height: auto; border-radius: 8px; }}
        .attribution-table {{ width: 100%; font-size: 13px; border-collapse: collapse; margin-top: 8px; }}
        .attribution-table th, .attribution-table td {{ padding: 6px 8px; border-bottom: 1px solid var(--border); text-align: left; }}
        .attribution-table th {{ background: #f3f5f7; }}
        .grouping-group {{ margin-bottom: 16px; border: 1px solid var(--border); border-radius: 8px; padding: 12px; background: var(--panel); }}
        .grouping-group.cv-low {{ border-left: 4px solid #22c55e; }}
        .grouping-group.cv-medium {{ border-left: 4px solid #f59e0b; }}
        .grouping-group.cv-high {{ border-left: 4px solid #ef4444; background: #fef2f2; }}
        .group-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px; }}
        .group-info {{ display: flex; align-items: center; gap: 12px; }}
        .group-strat {{ font-weight: 600; font-size: 14px; color: var(--text); }}
        .group-count {{ font-size: 12px; color: var(--muted); background: #f1f5f9; padding: 2px 8px; border-radius: 12px; }}
        .cv-indicator {{ display: flex; align-items: center; gap: 8px; }}
        .cv-label {{ font-size: 12px; color: var(--muted); }}
        .cv-bar-container {{ width: 80px; height: 8px; background: #e5eaef; border-radius: 4px; overflow: hidden; }}
        .cv-bar {{ height: 100%; border-radius: 4px; transition: width 0.3s ease; }}
        .cv-value {{ font-weight: 600; font-size: 13px; min-width: 40px; }}
        .grouping-table {{ margin-top: 0; }}
        .strat-cell {{ font-weight: 500; }}
        .geochem-cell {{ font-size: 12px; color: var(--muted); }}
        .reason-cell {{ max-width: 280px; font-size: 12px; }}
        .plotly-chart {{ min-height: 260px; }}
        .mineral-mismatch-stats {{ margin: 8px 0 12px 0; font-size: 14px; color: var(--muted); }}
        .mineral-mismatch-stats .stats-note {{ font-style: italic; margin-left: 8px; }}
        .significance-low {{ background: #e8f4ea; color: #1b5e20; }}
        .significance-high {{ background: #ffebee; color: #b71c1c; }}
        .charts-panel .panel-card {{ min-width: 320px; }}
        /* Compact table styles for fines and evidence tables */
        .compact-table th, .compact-table td {{
            padding: 4px 6px !important;
            font-size: 12px;
        }}
        .compact-row td {{
            vertical-align: middle;
        }}
        .nowrap {{
            white-space: nowrap;
        }}
        .classification-cell {{
            font-weight: 600;
            background: #fef3c7;
            border-radius: 4px;
            padding: 2px 6px !important;
        }}
        .classification-cell.logged {{
            background: #fee2e2;
            color: #991b1b;
        }}
        .classification-cell.assay {{
            background: #dcfce7;
            color: #166534;
        }}
        .issue-cell {{
            max-width: 180px;
            font-size: 11px;
        }}
        /* Inline geochem display */
        .geochem-inline {{
            display: inline-block;
            margin-right: 8px;
            font-size: 11px;
        }}
        .geochem-inline b {{
            color: var(--muted);
            margin-right: 2px;
        }}
        .geochem-inline-cell {{
            white-space: nowrap;
            font-size: 11px;
        }}
        /* Expandable images */
        .expandable-img {{
            cursor: zoom-in;
            transition: all 0.3s ease;
        }}
        .expandable-img.expanded {{
            position: fixed !important;
            top: 50% !important;
            left: 50% !important;
            transform: translate(-50%, -50%) rotate(90deg) !important;
            width: auto !important;
            max-width: 90vh !important;
            max-height: 90vw !important;
            z-index: 9999;
            cursor: zoom-out;
            box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
            border-radius: 8px;
        }}
        .image-cell-compact {{
            width: 80px;
            padding: 2px !important;
        }}
        .image-cell-compact img {{
            width: 80px;
            height: 50px;
            object-fit: cover;
            border-radius: 4px;
            border: 1px solid var(--border);
            transform: none;
        }}
        /* Evidence section */
        .evidence-section {{
            margin-top: 20px;
            padding: 16px;
            background: #fef9e7;
            border: 1px solid #f9e79f;
            border-radius: 8px;
        }}
        .evidence-section h3 {{
            margin-top: 0;
            color: #9a7b1a;
            font-size: 14px;
        }}
        /* Warning box for missing data */
        .warning-box {{
            padding: 16px;
            background: #fff7ed;
            border: 1px solid #fed7aa;
            border-left: 4px solid #f97316;
            border-radius: 8px;
            margin-bottom: 16px;
        }}
        .warning-box h4 {{
            margin: 0 0 8px 0;
            color: #c2410c;
            font-size: 14px;
        }}
        .warning-box p {{
            margin: 0;
            color: #9a3412;
            font-size: 13px;
        }}
        /* Rules table for zonation */
        .rules-box {{
            background: #f0f9ff;
            border: 1px solid #bae6fd;
        }}
        .rules-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 12px 0;
            font-size: 13px;
        }}
        .rules-table th, .rules-table td {{
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #e0f2fe;
        }}
        .rules-table th {{
            background: #e0f2fe;
            font-weight: 600;
        }}
        .rules-table tr:nth-child(even) {{
            background: #f0f9ff;
        }}
        .rules-note {{
            font-size: 11px;
            color: #0369a1;
            margin: 8px 0 0 0;
            font-style: italic;
        }}
        .evidence-table {{
            background: white;
        }}
        .evidence-table th {{
            background: #fef3c7 !important;
        }}
        /* Sortable table styles */
        .sortable-table th.sortable {{
            cursor: pointer;
            user-select: none;
        }}
        .sortable-table th.sortable:hover {{
            background: #fde68a !important;
        }}
        .sortable-table th.sort-asc::after {{ content: " ▲"; }}
        .sortable-table th.sort-desc::after {{ content: " ▼"; }}
        /* Assay suggestion conditional formatting */
        .assay-mineralised {{
            background-color: #dcfce7 !important;
            color: #166534;
        }}
        .assay-leached {{
            background-color: #fef9c3 !important;
            color: #854d0e;
        }}
        .assay-unmineralised {{
            background-color: #fee2e2 !important;
            color: #991b1b;
        }}
        /* Validation status formatting */
        .validation-match {{
            background-color: #dcfce7 !important;
            color: #166534;
            font-weight: 500;
        }}
        .validation-mismatch {{
            background-color: #fee2e2 !important;
            color: #991b1b;
            font-weight: 500;
        }}
        .validation-pending {{
            background-color: #f3f4f6 !important;
            color: #6b7280;
        }}
        .geochem-cell {{
            text-align: right;
            font-family: monospace;
        }}
        /* Credit text */
        .credit-text {{
            font-size: 10px;
            color: #64748b;
            text-align: center;
            margin-top: 4px;
        }}
        /* Print button and styles */
        .print-btn {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 12px 20px;
            background: var(--accent);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 100;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .print-btn:hover {{
            background: #1d6856;
        }}
        @media print {{
            .sidebar, .print-btn, .notes-box, .map-controls, .leaflet-control-container {{
                display: none !important;
            }}
            .app {{
                flex-direction: column;
            }}
            .content {{
                width: 100%;
                padding: 0;
            }}
            .tab-panel {{
                display: block !important;
                page-break-after: always;
                border-bottom: 2px solid #ccc;
                padding-bottom: 20px;
                margin-bottom: 20px;
            }}
            .tab-panel:last-child {{
                page-break-after: avoid;
            }}
            .expandable-img.expanded {{
                position: relative !important;
                transform: rotate(90deg) !important;
            }}
        }}
        /* Map container – Leaflet fills widget */
        .map-container {{
            position: relative;
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            background: #f0f4f8;
        }}
        .map-container-leaflet {{
            display: flex;
            flex-direction: column;
            min-height: 300px;
        }}
        .map-leaflet-viewport {{
            width: 100%;
            flex: 1;
            min-height: 300px;
            z-index: 0;
        }}
        .map-leaflet-viewport.leaflet-container {{
            font-family: inherit;
        }}
        .map-legend {{
            display: flex;
            gap: 16px;
            padding: 8px 12px;
            background: white;
            border-top: 1px solid var(--border);
            font-size: 12px;
            color: var(--muted);
        }}
        .map-legend-leaflet {{
            flex-shrink: 0;
        }}
        .legend-dot {{
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 6px;
        }}
        .legend-dot.logger-dot {{
            background: var(--accent);
            border: 1px solid white;
            box-shadow: 0 0 0 1px var(--accent);
        }}
        .legend-dot.team-dot {{
            background: #94a3b8;
            opacity: 0.6;
        }}
        .legend {{
            display: flex;
            gap: 16px;
            margin-top: 10px;
            font-size: 12px;
            color: var(--muted);
        }}
        .legend .dot {{
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 6px;
        }}
        .legend .dot.logger {{ background: var(--accent); }}
        .legend .dot.team {{ background: #cbd2d9; }}
        .empty {{
            color: var(--muted);
            font-size: 13px;
            padding: 12px;
        }}
        .tab-panel {{
            display: none;
        }}
        .tab-panel.active {{
            display: block;
        }}
        .summary-list {{
            margin: 8px 0 0 18px;
        }}
        .intervals-section {{
            margin-top: 16px;
        }}
        @media (max-width: 860px) {{
            .app {{
                flex-direction: column;
            }}
            .sidebar {{
                width: 100%;
                flex-direction: row;
                overflow-x: auto;
                padding: 12px;
            }}
            .tab-button {{
                white-space: nowrap;
            }}
        }}
    </style>
</head>
<body>
    <div class="app">
        <aside class="sidebar">
            <div class="logo">
                {"<img src=\"" + logo_data + "\" alt=\"Logo\">" if logo_data else "<div></div>"}
                <div class="credit-text">George Symonds 2025</div>
                <div class="lang-toggle">
                    <button data-lang="fr" class="active">FR</button>
                    <button data-lang="en">EN</button>
                </div>
            </div>
            {''.join(tab_buttons)}
        </aside>
        <main class="content">
            {overview_section if include_overview else ""}
            {comment_section if include_comments else ""}
            {mineral_section}
            {profile_section}
            {logging_detail_section if include_logging_detail else ""}
            {grouping_section if include_grouping else ""}
            {outlier_section if include_outliers else ""}
        </main>
    </div>
    <script>
        const defaultLang = 'fr';
        const langKey = 'loggingReviewLang';

        function applyLanguage(lang) {{
            document.querySelectorAll('[data-i18n-fr], [data-i18n-html-fr]').forEach(el => {{
                const htmlText = el.getAttribute(`data-i18n-html-${{lang}}`);
                const plainText = el.getAttribute(`data-i18n-${{lang}}`);
                if (htmlText) {{
                    el.innerHTML = htmlText;
                }} else if (plainText) {{
                    el.textContent = plainText;
                }}
            }});
            document.querySelectorAll('.lang-toggle button').forEach(btn => {{
                btn.classList.toggle('active', btn.dataset.lang === lang);
            }});
            localStorage.setItem(langKey, lang);
        }}

        document.querySelectorAll('.lang-toggle button').forEach(btn => {{
            btn.addEventListener('click', () => applyLanguage(btn.dataset.lang));
        }});

        const savedLang = localStorage.getItem(langKey) || defaultLang;
        applyLanguage(savedLang);

        function decodePlotlyAttr(str) {{
            if (!str) return str;
            return str.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"');
        }}
        function initPlotlyCharts() {{
            if (typeof Plotly === 'undefined') return;
            document.querySelectorAll('.plotly-chart').forEach(function(el) {{
                var dataStr = el.getAttribute('data-plotly-data');
                var layoutStr = el.getAttribute('data-plotly-layout');
                if (!dataStr || !layoutStr) return;
                try {{
                    var data = JSON.parse(decodePlotlyAttr(dataStr));
                    var layout = JSON.parse(decodePlotlyAttr(layoutStr));
                    if (data && (data.length || Array.isArray(data))) Plotly.react(el, data, layout);
                }} catch (e) {{ console.warn('Plotly init failed', e); }}
            }});
        }}
        function resizePlotlyInActivePanel() {{
            if (typeof Plotly === 'undefined') return;
            var active = document.querySelector('.tab-panel.active');
            if (!active) return;
            var charts = active.querySelectorAll('.plotly-chart');
            setTimeout(function() {{
                charts.forEach(function(el) {{
                    try {{ if (el._fullData) Plotly.Plots.resize(el); }} catch (e) {{}}
                }});
            }}, 50);
        }}

        const tabs = document.querySelectorAll('.tab-panel');
        const buttons = document.querySelectorAll('.tab-button');
        function activateTab(tabId) {{
            tabs.forEach(tab => tab.classList.toggle('active', tab.dataset.tab === tabId));
            buttons.forEach(btn => btn.classList.toggle('active', btn.dataset.tabButton === tabId));
            resizePlotlyInActivePanel();
        }}
        buttons.forEach(btn => {{
            btn.addEventListener('click', () => activateTab(btn.dataset.tabButton));
        }});
        const firstTab = buttons.length ? buttons[0].dataset.tabButton : 'overview';
        activateTab(firstTab);

        function setupNotes() {{
            document.querySelectorAll('[data-note-id]').forEach(area => {{
                const key = `note::${{area.dataset.noteId}}`;
                area.value = localStorage.getItem(key) || '';
                area.addEventListener('input', () => {{
                    localStorage.setItem(key, area.value);
                }});
            }});
        }}

        function setupCheckboxes() {{
            document.querySelectorAll('[data-review-id]').forEach(box => {{
                const key = `review::${{box.dataset.reviewId}}`;
                box.checked = localStorage.getItem(key) === '1';
                box.addEventListener('change', () => {{
                    localStorage.setItem(key, box.checked ? '1' : '0');
                }});
            }});
        }}

        setupNotes();
        setupCheckboxes();

        // Leaflet map: OpenTopoMap basemap, logger/team markers, fit bounds
        function setupLeafletMap() {{
            const el = document.getElementById('leaflet-map');
            if (!el || typeof L === 'undefined') return;
            const pointsJson = el.dataset.mapPoints;
            const boundsJson = el.dataset.mapBounds;
            if (!pointsJson || pointsJson === '[]') return;
            let points, bounds;
            try {{
                points = JSON.parse(pointsJson);
                bounds = boundsJson && boundsJson !== 'null' ? JSON.parse(boundsJson) : null;
            }} catch (e) {{
                return;
            }}
            if (!points.length) return;

            const map = L.map(el, {{ scrollWheelZoom: true }});
            L.tileLayer('https://{{s}}.tile.opentopomap.org/{{z}}/{{x}}/{{y}}.png', {{
                maxZoom: 17,
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, &copy; <a href="https://opentopomap.org">OpenTopoMap</a>'
            }}).addTo(map);

            const loggerColor = '#0d5b88';
            const teamColor = '#94a3b8';
            const loggerLayer = L.layerGroup();
            const teamLayer = L.layerGroup();
            points.forEach(function(p) {{
                const lat = p.lat;
                const lng = p.lng;
                const holeId = p.hole_id || '';
                const isLogger = p.is_logger;
                const marker = L.circleMarker([lat, lng], {{
                    radius: isLogger ? 8 : 6,
                    fillColor: isLogger ? loggerColor : teamColor,
                    color: '#fff',
                    weight: 1,
                    opacity: 1,
                    fillOpacity: isLogger ? 0.9 : 0.5
                }});
                marker.bindTooltip(holeId, {{ permanent: false, direction: 'top' }});
                if (isLogger) loggerLayer.addLayer(marker);
                else teamLayer.addLayer(marker);
            }});
            teamLayer.addTo(map);
            loggerLayer.addTo(map);

            if (bounds && bounds.length === 4) {{
                map.fitBounds([[bounds[0], bounds[1]], [bounds[2], bounds[3]]], {{ padding: [20, 20], maxZoom: 15 }});
            }} else {{
                const lats = points.map(p => p.lat);
                const lngs = points.map(p => p.lng);
                map.fitBounds([[Math.min.apply(null, lats), Math.min.apply(null, lngs)], [Math.max.apply(null, lats), Math.max.apply(null, lngs)]], {{ padding: [20, 20], maxZoom: 15 }});
            }}
        }}

        setupLeafletMap();

        // Print functionality - show all tabs for PDF export
        function printReport() {{
            // Show all tab panels for printing
            document.querySelectorAll('.tab-panel').forEach(panel => {{
                panel.style.display = 'block';
            }});
            // Initialize all Plotly charts (they may not have been rendered yet)
            initPlotlyCharts();
            // Small delay to ensure charts render
            setTimeout(() => {{
                window.print();
                // Restore tab panel display after printing
                setTimeout(() => {{
                    activateTab(document.querySelector('.tab-button.active')?.dataset.tabButton || 'overview');
                }}, 500);
            }}, 300);
        }}

        // Add print button click handler
        const printBtn = document.getElementById('print-report-btn');
        if (printBtn) {{
            printBtn.addEventListener('click', printReport);
        }}

        // Image expansion handler - ensures only one image is expanded at a time
        function handleImageExpand(img) {{
            const wasExpanded = img.classList.contains('expanded');
            // Close all expanded images first
            document.querySelectorAll('.expandable-img.expanded').forEach(function(other) {{
                other.classList.remove('expanded');
            }});
            // Toggle the clicked image (if it wasn't already expanded, expand it)
            if (!wasExpanded) {{
                img.classList.add('expanded');
            }}
        }}

        // Close expanded images when clicking outside
        document.addEventListener('click', function(e) {{
            // If clicking on an expandable image, let the onclick handler manage it
            if (e.target.classList.contains('expandable-img')) {{
                return;
            }}
            // Otherwise, close all expanded images
            document.querySelectorAll('.expandable-img.expanded').forEach(function(img) {{
                img.classList.remove('expanded');
            }});
        }});

        // Sortable table functionality
        function initSortableTables() {{
            document.querySelectorAll('.sortable-table').forEach(function(table) {{
                const headers = table.querySelectorAll('th.sortable');
                headers.forEach(function(header, index) {{
                    header.addEventListener('click', function() {{
                        const tbody = table.querySelector('tbody');
                        const rows = Array.from(tbody.querySelectorAll('tr'));
                        const isAsc = header.classList.contains('sort-asc');

                        // Remove sort classes from all headers
                        headers.forEach(h => h.classList.remove('sort-asc', 'sort-desc'));

                        // Sort rows
                        rows.sort(function(a, b) {{
                            const cellA = a.children[index + 1]; // +1 for checkbox column
                            const cellB = b.children[index + 1];
                            let valA = cellA ? (cellA.dataset.sort || cellA.textContent.trim()) : '';
                            let valB = cellB ? (cellB.dataset.sort || cellB.textContent.trim()) : '';

                            // Try numeric comparison
                            const numA = parseFloat(valA);
                            const numB = parseFloat(valB);
                            if (!isNaN(numA) && !isNaN(numB)) {{
                                return isAsc ? numB - numA : numA - numB;
                            }}

                            // String comparison
                            return isAsc ? valB.localeCompare(valA) : valA.localeCompare(valB);
                        }});

                        // Re-add sorted rows
                        rows.forEach(row => tbody.appendChild(row));
                        header.classList.add(isAsc ? 'sort-desc' : 'sort-asc');
                    }});
                }});
            }});
        }}

        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', function() {{
                initPlotlyCharts();
                resizePlotlyInActivePanel();
                initSortableTables();
            }});
        }} else {{
            initPlotlyCharts();
            resizePlotlyInActivePanel();
            initSortableTables();
        }}
    </script>
    <button id="print-report-btn" class="print-btn" title="Export to PDF">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M6 9V2h12v7M6 18H4a2 2 0 01-2-2v-5a2 2 0 012-2h16a2 2 0 012 2v5a2 2 0 01-2 2h-2"/>
            <rect x="6" y="14" width="12" height="8"/>
        </svg>
        Print / PDF
    </button>
</body>
</html>
"""
    return html_content
