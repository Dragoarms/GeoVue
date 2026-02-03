import html
import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pyproj
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False

from processing.DataManager.column_aliases import ColumnResolver
from processing.DataManager.keys import ImageKey
from reports.logging_review.html.assets.styles import CSS_STYLES
from reports.logging_review.html.assets.scripts import JS_SCRIPTS
from reports.logging_review.html.utils import _safe_str, _safe_float, _format_metric
from reports.logging_review.html.charts import (
    _plotly_pie_json,
    _plotly_stacked_bar_json,
    _plotly_stacked_bar_pct_json,
    _plotly_strat_grouped_bar_json,
    _plotly_strat_grouped_bar_pct_json,
    _plotly_zonation_bar_json,
    _plotly_outlier_box_json,
    _plotly_outlier_scatter_json,
)
from reports.logging_review.html.tables import (
    _render_intervals_table,
    _render_mineralisation_evidence_table,
    _render_zonation_evidence_table,
    _render_logging_detail_evidence_table,
    _render_fines_intervals_table,
    _render_outlier_table,
    _render_grouping_groups,
)
from reports.logging_review.html.images import _encode_image_base64
from reports.logging_review.html.collar_map import _build_map_points, _render_map
from reports.logging_review.html.tabs.comments import render_comments_section
from reports.logging_review.html.tabs.grouping import render_grouping_section
from reports.logging_review.html.tabs.logging_detail import render_logging_detail_section
from reports.logging_review.html.tabs.mineralisation import render_mineralisation_section
from reports.logging_review.html.tabs.outliers import render_outlier_section
from reports.logging_review.html.tabs.overview import build_overview_data, render_overview_section
from reports.logging_review.html.tabs.profile import render_profile_section
from reports.logging_review.html.report_builder import build_html_report
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
    """Thin wrapper: delegate to reports.logging_review.html.orchestration."""
    from reports.logging_review.html.orchestration import (
        generate_logger_html_reports_from_prepped_data as _impl,
    )
    return _impl(
        data_coordinator=data_coordinator,
        output_dir=output_dir,
        merged_df=merged_df,
        logging_df=logging_df,
        collar_df=collar_df,
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
        skip_csv_export=skip_csv_export,
    )


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
    """Thin wrapper: delegate to reports.logging_review.html.orchestration."""
    from reports.logging_review.html.orchestration import generate_logger_html_reports as _impl
    return _impl(
        data_coordinator=data_coordinator,
        output_dir=output_dir,
        date_from=date_from,
        date_to=date_to,
        logger_values=logger_values,
        top_n=top_n,
        page_options=page_options,
        include_images=include_images,
        logo_path=logo_path,
    )


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
    resolver = ColumnResolver(df)
    return resolver.get("project_code")


def _resolve_coordinate_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    resolver = ColumnResolver(df)
    return resolver.get("easting"), resolver.get("northing")


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


def _significance_for_logging_detail_issue(issue: str, issue_type: str) -> str:
    """Classify significance: High if major elements (Fe, SiO2, Al2O3, gangue) involved; Low if only minor (P, CaO, LOI, carbonate)."""
    if not issue:
        return "High"
    issue_lower = issue.lower()
    major = any(x in issue_lower for x in ("fe", "sio2", "al2o3", "al ", "gangue", "mineralisation", "silica", "iron"))
    minor_only = any(x in issue_lower for x in ("p ", "p%", "cao", "loi", "carbonate")) and not major
    if issue_type == "carbonate":
        return "Low"
    if minor_only:
        return "Low"
    return "High"


def _build_grouping_issue_intervals(
    df: pd.DataFrame,
    group_cols: List[str],
    chem_cols: List[str],
    hole_col: str,
    depth_from_col: Optional[str],
    depth_to_col: str,
) -> List[Dict[str, Any]]:
    """
    Build list of intervals with high CV issues.

    This function fills NaN values in grouping columns with 'UNKNOWN' placeholder to allow
    proper grouping (matches the approach in the original combine_intervals function).
    """
    # Use only grouping columns that exist
    group_cols_use = [c for c in group_cols if c in df.columns]
    if not group_cols_use:
        return []
    available = [c for c in chem_cols if c in df.columns]
    if not available:
        return []

    # Work on a copy to avoid modifying the original DataFrame
    df_work = df.copy()

    # Fill missing values in grouping columns with 'UNKNOWN' placeholder to allow grouping
    for col in group_cols_use:
        df_work[col] = df_work[col].fillna("UNKNOWN").astype(str).replace("", "UNKNOWN")

    intervals = []
    for _, group in df_work.groupby(group_cols_use):
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


def _zonation_adjacent(logged: str, should_be: str) -> bool:
    """True when logged and should_be are adjacent in Un, Le, De, Hy, Pr (e.g. Le vs De)."""
    order = ["Un", "Le", "De", "Hy", "Pr"]
    if logged not in order or should_be not in order:
        return False
    return abs(order.index(logged) - order.index(should_be)) == 1


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
                significance = "Low" if _zonation_adjacent(zonation, should_be) else "High"
                rows_out.append({
                    "hole_id": _safe_str(row.get(hole_col)),
                    "depth_from": _safe_float(row.get(depth_from_col)) if depth_from_col else None,
                    "depth_to": _safe_float(row.get(depth_to_col)),
                    "logged_zonation": zonation,
                    "should_be": should_be,
                    "validation": "Mismatch",
                    "significance": significance,
                    "total_gangue_pct": total_gangue,
                    "de_pct": de_pct,
                    "hy_pct": hy_pct,
                    "pr_pct": pr_pct,
                })
    return rows_out


GROUPING_CV_ELEMENTS = ["fe_pct", "sio2_pct", "al2o3_pct", "mgo_pct", "cao_pct", "s_pct"]
GROUPING_MAJOR_ELEMENTS = ["fe_pct", "sio2_pct", "al2o3_pct"]


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
    # Scan for Min_*_pct columns (mineral logging percentages) - matches combine_intervals grouping
    for c in df.columns:
        if c in group_cols:
            continue
        c_lower = c.lower()
        if "min_" in c_lower and "_pct" in c_lower:
            group_cols.append(c)
            display_names.append(c)
    # Add LithComments if present (important grouping column from original combine_intervals)
    for c in df.columns:
        if c in group_cols:
            continue
        c_lower = c.lower()
        if c_lower in ("lithcomments", "lith_comments", "lithology_comments"):
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
    strat_col: str,
    top_n_groups: int = 20,
) -> Tuple[Optional[float], Optional[float], List[Dict[str, Any]]]:
    """Returns (avg_group_interval_m, max_group_interval_m, list of groups with high CV).

    Interval length KPIs use combined span per group: max(depth_to) - min(depth_from),
    matching the old combine_intervals / QAQC_07 report semantics.

    This function fills NaN values in grouping columns with 'UNKNOWN' placeholder to allow
    proper grouping (matches the approach in the original combine_intervals function).
    """
    # Use only grouping columns that exist so we don't bail when optional cols are missing
    group_cols_use = [c for c in group_cols if c in df.columns]
    missing_group = [c for c in group_cols if c not in df.columns]
    logger.debug(
        "Grouping: df.shape=%s, group_cols_use=%s, missing_in_df=%s, depth_to_col=%s, depth_to_in_df=%s",
        df.shape,
        group_cols_use,
        missing_group,
        depth_to_col,
        depth_to_col in df.columns if depth_to_col else False,
    )
    if not group_cols_use or not depth_to_col or depth_to_col not in df.columns:
        logger.debug(
            "Grouping early return: group_cols_use empty=%s, no depth_to=%s, depth_to not in df=%s",
            not group_cols_use,
            not depth_to_col,
            depth_to_col not in df.columns if depth_to_col else "n/a",
        )
        return None, None, []

    # Work on a copy to avoid modifying the original DataFrame
    df_work = df.copy()

    # Fill missing values in grouping columns with 'UNKNOWN' placeholder to allow grouping
    # This matches the approach in the original combine_intervals function
    for col in group_cols_use:
        if col in df_work.columns:
            df_work[col] = df_work[col].fillna("UNKNOWN").astype(str).replace("", "UNKNOWN")

    # Drop rows where critical depth columns are null (can't calculate interval lengths)
    depth_cols_check = [depth_to_col]
    if depth_from_col and depth_from_col in df_work.columns:
        depth_cols_check.append(depth_from_col)
    df_work = df_work.dropna(subset=depth_cols_check)

    if df_work.empty:
        logger.debug("Grouping: DataFrame empty after dropping null depth rows")
        return None, None, []

    available = []
    for std in GROUPING_CV_ELEMENTS:
        col = resolver.get(std)
        if col and col in df_work.columns and col in chem_actual_cols:
            available.append(col)
    if not available:
        available = [c for c in chem_actual_cols if c in df_work.columns][:6]
    logger.debug(
        "Grouping: CV elements available=%s (count=%s)",
        available[:8],
        len(available),
    )
    # Combined interval length per group (max(to) - min(from)) for parity with old report
    combined_lengths_per_group: List[float] = []
    strat_idx = group_cols_use.index(strat_col) if strat_col in group_cols_use else None
    major_cols = {resolver.get(std) for std in GROUPING_MAJOR_ELEMENTS if resolver.get(std)}
    group_cv_list = []
    for key, group in df_work.groupby(group_cols_use):
        to_vals = group[depth_to_col].astype(float)
        from_vals = group[depth_from_col].astype(float) if depth_from_col and depth_from_col in group.columns else 0.0
        if depth_from_col and depth_from_col in group.columns:
            combined_len = float(to_vals.max() - from_vals.min())
        else:
            combined_len = float(to_vals.max() - to_vals.min())
        if combined_len <= 0:
            continue
        combined_lengths_per_group.append(combined_len)
        # Per-row lengths still used for display in each group (mean_interval_m, max_interval_m)
        from_vals = group[depth_from_col] if depth_from_col and depth_from_col in group.columns else 0
        lengths = to_vals - (from_vals.astype(float) if depth_from_col else 0)
        lengths = lengths[lengths > 0]
        mean_len = float(lengths.mean()) if len(lengths) else combined_len
        max_len = float(lengths.max()) if len(lengths) else combined_len
        max_cv = 0.0
        cols_over_100 = []
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
            if cv > 100:
                cols_over_100.append(col)
        if max_cv > 100:
            group_cv_list.append((key, max_cv, mean_len, max_len, group, cols_over_100))
    group_cv_list.sort(key=lambda x: x[1], reverse=True)
    logger.debug(
        "Grouping: num_groups=%s, num_with_high_cv=%s",
        len(combined_lengths_per_group),
        len(group_cv_list),
    )
    avg_all = float(np.mean(combined_lengths_per_group)) if combined_lengths_per_group else None
    max_all = float(np.max(combined_lengths_per_group)) if combined_lengths_per_group else None
    groups_for_review = []
    for item in group_cv_list[:top_n_groups]:
        key, cv, mean_len, max_len, group, cols_over_100 = item
        strat_value = str(key[strat_idx]) if strat_idx is not None and len(key) > strat_idx else ""
        has_major = any(col in major_cols for col in cols_over_100)
        significance = "High" if has_major else "Low"
        intervals_in_group = []
        for _, row in group.iterrows():
            intervals_in_group.append({
                "hole_id": _safe_str(row.get(hole_col)),
                "depth_from": _safe_float(row.get(depth_from_col)) if depth_from_col else None,
                "depth_to": _safe_float(row.get(depth_to_col)),
                "strat": _safe_str(row.get(strat_col)),
                "issue": f"CV >100% in group",
                "geochem": {col: _safe_float(row.get(col)) for col in available if col in row.index},
            })
        groups_for_review.append({
            "group_key": str(key),
            "strat": strat_value,
            "cv_max": cv,
            "mean_interval_m": mean_len,
            "max_interval_m": max_len,
            "significance": significance,
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
    """Thin wrapper: delegate to reports.logging_review.html.report_builder.build_html_report."""
    return build_html_report(
        data_coordinator=data_coordinator,
        logger_value=logger_value,
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
            "strat": grp.get("strat"),
            "cv_max": grp.get("cv_max"),
            "mean_interval_m": grp.get("mean_interval_m"),
            "max_interval_m": grp.get("max_interval_m"),
            "significance": grp.get("significance", "High"),
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
