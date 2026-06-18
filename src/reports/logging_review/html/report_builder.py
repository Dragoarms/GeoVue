"""Build logging review report data and delegate to HTML renderer."""
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from processing.DataManager.column_aliases import ColumnResolver
from .collar_map import _build_map_points
from .charts import _plotly_outlier_box_json, _plotly_outlier_scatter_grid_json, _plotly_pca_biplot_json
from .report_renderer import render_html
from .tabs.overview import build_overview_data
from .tables import _outlier_significance_from_reason, _parse_outlier_reason_to_flags
from .types import IntervalsForReview, ReportData
from .utils import _safe_float, _safe_str
from reports.logging_review.data.outliers import OUTLIER_DISPLAY_THRESHOLD

logger = logging.getLogger(__name__)


def _metric_total_depth(
    df: pd.DataFrame,
    hole_col: str,
    depth_from_col: Optional[str],
    depth_to_col: str,
) -> float:
    """Total metres from unique physical intervals, not merge-expanded rows."""
    if not depth_from_col or not depth_to_col or depth_from_col not in df.columns or depth_to_col not in df.columns:
        return float("nan")
    key_cols = [depth_from_col, depth_to_col]
    if hole_col in df.columns:
        key_cols.insert(0, hole_col)
    intervals = df[key_cols].copy()
    intervals[depth_from_col] = pd.to_numeric(intervals[depth_from_col], errors="coerce")
    intervals[depth_to_col] = pd.to_numeric(intervals[depth_to_col], errors="coerce")
    intervals = intervals.dropna(subset=[depth_from_col, depth_to_col]).drop_duplicates()
    lengths = intervals[depth_to_col] - intervals[depth_from_col]
    lengths = lengths[lengths > 0]
    return float(lengths.sum())


def _mineral_assay_coverage_counts(
    df: pd.DataFrame,
    mineral_col: Optional[str],
    chem_actual_cols: List[str],
) -> Dict[str, float]:
    """Assay coverage among mineralised intervals using any available assay column."""
    if not mineral_col or mineral_col not in df.columns or not chem_actual_cols:
        return {"mineralised_intervals": 0, "assays_available": 0, "assay_ratio_pct": 0.0}
    assay_cols = [c for c in chem_actual_cols if c in df.columns]
    if not assay_cols:
        return {"mineralised_intervals": 0, "assays_available": 0, "assay_ratio_pct": 0.0}
    mineralised_mask = df[mineral_col].astype(str).replace("nan", "").str.strip().ne("")
    has_assay_mask = df[assay_cols].notna().any(axis=1)
    mineralised_intervals = int(mineralised_mask.sum())
    assays_available = int((mineralised_mask & has_assay_mask).sum())
    ratio = (assays_available / mineralised_intervals * 100.0) if mineralised_intervals else 0.0
    return {
        "mineralised_intervals": mineralised_intervals,
        "assays_available": assays_available,
        "assay_ratio_pct": min(100.0, float(ratio)),
    }


def build_html_report(
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
    output_path: Optional[str] = None,
) -> str:
    """
    Build report data and render HTML.
    If output_path is set, stream HTML directly to that file (reduces memory for large reports with many images).
    Returns the output_path when streaming, or the HTML string when not.
    """
    from processing.logging_review_html_report import (
        _add_mineralisation_accuracy_column,
        _build_grouping_groups_with_images,
        _build_grouping_issue_intervals,
        _build_intervals_with_images,
        _build_outlier_intervals_with_images,
        _build_resolved_group_cols,
        _calc_comment_coverage,
        _calc_fines_flag_rate,
        _calc_grouping_issue_rate,
        _calc_mineral_assay_ratio,
        _calc_outlier_rate,
        _calc_profile_coverage,
        _check_profile_zonation_and_analyze,
        _collect_zonation_mismatch_rows,
        _comment_stats_logging_intervals,
        _compute_assay_received_outstanding,
        _compute_avg_logging_interval,
        _compute_comment_stats,
        _compute_logger_median,
        _derive_dominant_zonation_column,
        _extract_comment_wordcloud,
        _flag_carbonate_issue,
        _flag_fines_issue,
        _flag_goethite_issue,
        _flag_magnetite_issue,
        _flag_sulphide_issue,
        _flag_manganese_issue,
        _flag_mafics_issue,
        _flag_magnesium_issue,
        _flag_clay_issue,
        _grouping_avg_max_interval_and_groups,
        _group_mineralisation_by_quarter,
        _lookup_interval_image,
        _resolve_project_code_column,
        _resolve_zonation_columns,
        _significance_for_logging_detail_issue,
    )
    from processing.logging_review_report import (
        _resolve_logging_interval_columns,
        _summarize_fines_accuracy,
        _summarize_grouping_accuracy_by_interval,
    )

    # Compute population stats from full dataset (all loggers) for dynamic thresholds
    from reports.logging_review.data.population_stats import PopulationStats
    pop_stats = PopulationStats(merged_df)

    def _build_mineral_summary(row, mineral_cols_resolved):
        """Build compact mineral assemblage string from MIN_*_PCT columns."""
        parts = []
        for mcol, pct_val in mineral_cols_resolved:
            if mcol not in row.index:
                continue
            mval = row.get(mcol)
            if pd.isna(mval):
                continue
            s = str(mval).strip()
            if s and s.lower() not in ("nan", "none", ""):
                pct_str = f"{int(pct_val)}%" if pct_val == int(pct_val) else f"{pct_val}%"
                parts.append(f"{s} {pct_str}")
        return ", ".join(parts)

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
    team_data_source = full_team_df if full_team_df is not None else merged_df

    project_col = _resolve_project_code_column(merged_df)
    logger_project_codes = (
        assay_logger_df[project_col].dropna().astype(str).unique().tolist()
        if project_col and project_col in assay_logger_df.columns
        else []
    )
    has_project_scope = bool(project_col and logger_project_codes)
    team_base_df = full_team_df if full_team_df is not None else merged_df
    project_filtered_df = (
        team_base_df[team_base_df[project_col].astype(str).isin(logger_project_codes)].copy()
        if has_project_scope and project_col in team_base_df.columns
        else team_base_df.copy()
    )

    resolver_assay = ColumnResolver(assay_logger_df)
    group_cols_raw, group_cols_display_raw = _build_resolved_group_cols(
        assay_logger_df, hole_col, logger_col, strat_col, resolver_assay
    )
    # Use only columns present in assay data so grouping runs (avoid n/a and 0 groups when cols differ)
    group_cols = [c for c in group_cols_raw if c in assay_logger_df.columns]
    group_cols_display = [
        group_cols_display_raw[i] for i, c in enumerate(group_cols_raw) if c in assay_logger_df.columns
    ]
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

    mineral_mismatch_intervals = []
    if "Logging_Accuracy" in assay_with_accuracy.columns:
        gangue_col = resolver_assay.get("total_gangue_pct")
        fe_col_min = resolver_assay.get("fe_pct")
        sio2_col_min = resolver_assay.get("sio2_pct")
        al2o3_col_min = resolver_assay.get("al2o3_pct")
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

            logged_strat = _safe_str(row.get(strat_col)) if strat_col else ""
            logged_zonation = _safe_str(row.get(zonation_col)) if zonation_col else ""
            validation = row.get("Logging_Accuracy", "")

            if validation == "Mismatch":
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

    image_cache = {} if (data_coordinator and include_images) else None
    if data_coordinator and include_images:
        for item in mineral_mismatch_intervals:
            item["image"] = _lookup_interval_image(
                data_coordinator,
                item.get("hole_id"),
                item.get("depth_to"),
                cache=image_cache,
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
                cache=image_cache,
            )
    zonation_cols_team = _resolve_zonation_columns(project_filtered_df)
    team_df_for_zonation, zonation_cols_team = _derive_dominant_zonation_column(project_filtered_df, zonation_cols_team)
    z_correct_team, z_mismatch_team, z_attribution_team = _check_profile_zonation_and_analyze(team_df_for_zonation, zonation_cols_team)

    # Debug: grouping tab uses assay (expanded) frame; need group_cols and depth cols present
    logger.debug(
        "Grouping tab input: assay_logger_df shape=%s, group_cols=%s, depth_from=%s, depth_to=%s",
        assay_logger_df.shape,
        group_cols,
        depth_from_col,
        depth_to_col,
    )
    logger.debug(
        "Grouping columns in assay df: %s; depth_to in df: %s",
        [c for c in group_cols if c in assay_logger_df.columns],
        depth_to_col in assay_logger_df.columns if depth_to_col else False,
    )
    logger.debug(
        "Grouping chem columns (first 6): %s",
        (chem_actual_cols[:6] if chem_actual_cols else []),
    )
    grouping_avg_m, grouping_max_m, grouping_groups_for_review = _grouping_avg_max_interval_and_groups(
        assay_logger_df, group_cols, chem_actual_cols, resolver_assay,
        hole_col, depth_from_col, depth_to_col, strat_col, top_n_groups=15,
    )
    logger.debug(
        "Grouping result: avg_m=%s, max_m=%s, groups_for_review=%s",
        grouping_avg_m,
        grouping_max_m,
        len(grouping_groups_for_review),
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
    k2o_col = fines_resolver.get("k2o_pct")
    loi_1000_col = fines_resolver.get("loi_1000_pct") or fines_resolver.get("loi_pct")
    hy_col = fines_resolver.get("zonation_hy_pct")
    loi_425_col = fines_resolver.get("loi_425_pct") or fines_resolver.get("loi_pct")
    carbonate_col = fines_resolver.get("carbonate_gangue_pct")
    cao_col = fines_resolver.get("cao_pct")
    loi_col = fines_resolver.get("loi_pct")
    magnetite_pct_col = fines_resolver.get("magnetite_pct")
    s_col = fines_resolver.get("s_pct")
    mn_col = fines_resolver.get("mn_pct")
    mgo_col = fines_resolver.get("mgo_pct")
    sulphide_col = fines_resolver.get("sulphide_gangue_pct")
    manganese_col = fines_resolver.get("manganese_gangue_pct")
    mafics_col = fines_resolver.get("mafics_gangue_pct")
    magnesium_col = fines_resolver.get("magnesium_gangue_pct")
    fines_intervals = []
    magnetite_intervals = []
    goethite_intervals = []
    carbonate_intervals = []
    sulphide_intervals = []
    manganese_intervals = []
    mafics_intervals = []
    magnesium_intervals = []
    clay_intervals = []
    # Resolve ALL mineral percentage columns for evidence table display
    _mineral_pct_thresholds = [
        ("min_80_pct", 80), ("min_70_pct", 70), ("min_60_pct", 60),
        ("min_50_pct", 50), ("min_40_pct", 40), ("min_30_pct", 30),
        ("min_20_pct", 20), ("min_10_pct", 10), ("min_5_pct", 5),
        ("min_2_pct", 2), ("min_1_pct", 1), ("min_0_5_pct", 0.5),
    ]
    _mineral_cols_resolved = []
    for std_name, pct_val in _mineral_pct_thresholds:
        col = fines_resolver.get(std_name)
        if col:
            _mineral_cols_resolved.append((col, pct_val))
    cols = assay_logger_df.columns
    for tup in assay_logger_df.itertuples(index=False):
        row = pd.Series(dict(zip(cols, tup)))
        geochem = {}
        if fe_col and fe_col in row:
            geochem["Fe"] = _safe_float(row.get(fe_col))
        if al2o3_col and al2o3_col in row:
            geochem["Al2O3"] = _safe_float(row.get(al2o3_col))
        if sio2_col and sio2_col in row:
            geochem["SiO2"] = _safe_float(row.get(sio2_col))
        if p_col and p_col in row:
            geochem["P"] = _safe_float(row.get(p_col))
        issue_fines = _flag_fines_issue(row, fines_resolver)
        if issue_fines:
            fines_intervals.append({
                "hole_id": _safe_str(row.get(hole_col)),
                "depth_from": _safe_float(row.get(depth_from_col)) if depth_from_col else None,
                "depth_to": _safe_float(row.get(depth_to_col)),
                "classified_as": _safe_str(row.get(strat_col)),
                "strat": _safe_str(row.get(strat_col)),
                "issue": issue_fines,
                "geochem": geochem,
                "significance": _significance_for_logging_detail_issue(issue_fines, "fines"),
            })
        base = {
            "hole_id": _safe_str(row.get(hole_col)),
            "depth_from": _safe_float(row.get(depth_from_col)) if depth_from_col else None,
            "depth_to": _safe_float(row.get(depth_to_col)),
            "classified_as": _safe_str(row.get(strat_col)),
            "strat": _safe_str(row.get(strat_col)),
            "geochem": geochem,
            "logged_minerals": _build_mineral_summary(row, _mineral_cols_resolved),
        }
        assay = {}
        if loi_1000_col and loi_1000_col in row:
            assay["loi_1000_pct"] = _safe_float(row.get(loi_1000_col))
        if hy_col and hy_col in row:
            assay["hy_pct"] = _safe_float(row.get(hy_col))
        if loi_425_col and loi_425_col in row:
            assay["loi_pct"] = _safe_float(row.get(loi_425_col))
        if carbonate_col and carbonate_col in row:
            assay["carbonate_pct"] = _safe_float(row.get(carbonate_col))
        if cao_col and cao_col in row:
            assay["cao_pct"] = _safe_float(row.get(cao_col))
        if loi_col and loi_col in row and "loi_pct" not in assay:
            assay["loi_pct"] = _safe_float(row.get(loi_col))
        if magnetite_pct_col and magnetite_pct_col in row:
            assay["magnetite_pct"] = _safe_float(row.get(magnetite_pct_col))
        if sulphide_col and sulphide_col in row:
            assay["sulphide_pct"] = _safe_float(row.get(sulphide_col))
        if s_col and s_col in row:
            assay["s_pct"] = _safe_float(row.get(s_col))
        if manganese_col and manganese_col in row:
            assay["manganese_pct"] = _safe_float(row.get(manganese_col))
        if mn_col and mn_col in row:
            assay["mn_pct"] = _safe_float(row.get(mn_col))
        if mafics_col and mafics_col in row:
            assay["mafics_pct"] = _safe_float(row.get(mafics_col))
        if fe_col and fe_col in row and "fe_pct" not in assay:
            assay["fe_pct"] = _safe_float(row.get(fe_col))
        if magnesium_col and magnesium_col in row:
            assay["magnesium_pct"] = _safe_float(row.get(magnesium_col))
        if mgo_col and mgo_col in row:
            assay["mgo_pct"] = _safe_float(row.get(mgo_col))
        if k2o_col and k2o_col in row:
            assay["k2o_pct"] = _safe_float(row.get(k2o_col))
        issue_m, sig_m = _flag_magnetite_issue(row, fines_resolver, pop_stats)
        if issue_m:
            magnetite_intervals.append({
                **base, "issue": issue_m, "significance": sig_m,
                "assay": assay,
            })
        issue_g, sig_g = _flag_goethite_issue(row, fines_resolver, pop_stats)
        if issue_g:
            goethite_intervals.append({
                **base, "issue": issue_g, "significance": sig_g,
                "assay": assay,
            })
        issue_c, sig_c = _flag_carbonate_issue(row, fines_resolver, pop_stats)
        if issue_c:
            carbonate_intervals.append({
                **base, "issue": issue_c, "significance": sig_c,
                "assay": assay,
            })
        issue_s, sig_s = _flag_sulphide_issue(row, fines_resolver, pop_stats)
        if issue_s:
            sulphide_intervals.append({
                **base, "issue": issue_s, "significance": sig_s,
                "assay": assay,
            })
        issue_mn, sig_mn = _flag_manganese_issue(row, fines_resolver, pop_stats)
        if issue_mn:
            manganese_intervals.append({
                **base, "issue": issue_mn, "significance": sig_mn,
                "assay": assay,
            })
        issue_maf, sig_maf = _flag_mafics_issue(row, fines_resolver, pop_stats)
        if issue_maf:
            mafics_intervals.append({
                **base, "issue": issue_maf, "significance": sig_maf,
                "assay": assay,
            })
        issue_mg, sig_mg = _flag_magnesium_issue(row, fines_resolver, pop_stats)
        if issue_mg:
            magnesium_intervals.append({
                **base, "issue": issue_mg, "significance": sig_mg,
                "assay": assay,
            })
        issue_cl, sig_cl = _flag_clay_issue(row, fines_resolver, pop_stats)
        if issue_cl:
            clay_intervals.append({
                **base, "issue": issue_cl, "significance": sig_cl,
                "assay": assay,
            })

    if "most_likely_strat" not in assay_logger_df.columns:
        logger.warning(
            "most_likely_strat missing from prepared data; using recorded strat as defensive fallback."
        )
        assay_logger_df["most_likely_strat"] = assay_logger_df[strat_col] if strat_col in assay_logger_df.columns else "-"

    all_outliers = assay_logger_df[assay_logger_df["outlier_score"] > OUTLIER_DISPLAY_THRESHOLD].copy()
    all_outliers = all_outliers.sort_values("outlier_score", ascending=False)
    # Evidence table and KPI: only intervals where recorded strat != most likely (strat mismatch for review)
    recorded_strat = all_outliers[strat_col].astype(str).str.strip()
    likely_strat = all_outliers["most_likely_strat"].fillna("-").astype(str).str.strip()
    meaningful_likely = likely_strat.ne("") & likely_strat.ne("-")
    strat_mismatch_mask = meaningful_likely & (recorded_strat != likely_strat)
    strat_mismatch_outliers = all_outliers[strat_mismatch_mask].copy()
    total_misclassified = len(strat_mismatch_outliers)
    # Mark strat-mismatch outliers so PCA/scatter charts highlight only these (not in-strat chemistry noise)
    assay_logger_df["display_outlier"] = False
    assay_logger_df.loc[strat_mismatch_outliers.index, "display_outlier"] = True

    display_outliers = strat_mismatch_outliers.head(top_n) if top_n > 0 else strat_mismatch_outliers
    outlier_rows = []
    out_cols = display_outliers.columns
    for tup in display_outliers.itertuples(index=False):
        row = pd.Series(dict(zip(out_cols, tup)))
        strat_val = _safe_str(row.get(strat_col))
        most_likely = _safe_str(row.get("most_likely_strat", "-"))
        reason = _safe_str(row.get("outlier_reason"))

        geochem = {}
        if fe_col and fe_col in row:
            geochem["Fe"] = _safe_float(row.get(fe_col))
        if al2o3_col and al2o3_col in row:
            geochem["Al2O3"] = _safe_float(row.get(al2o3_col))
        if sio2_col and sio2_col in row:
            geochem["SiO2"] = _safe_float(row.get(sio2_col))
        if p_col and p_col in row:
            geochem["P"] = _safe_float(row.get(p_col))

        reason_str = reason or _safe_str(row.get("outlier_elements"))
        elements_str = _safe_str(row.get("outlier_elements"))
        flags_list = _parse_outlier_reason_to_flags(reason_str)
        significance = _outlier_significance_from_reason(reason_str, elements_str)
        outlier_rows.append(
            {
                "hole_id": _safe_str(row.get(hole_col)),
                "depth_from": _safe_float(row.get(depth_from_col)) if depth_from_col else None,
                "depth_to": _safe_float(row.get(depth_to_col)),
                "strat": strat_val,
                "recorded_as": strat_val,
                "most_likely": most_likely,
                "reason": reason_str,
                "outlier_score": _safe_float(row.get("outlier_score")) or 0.0,
                "outlier_reason": reason,
                "outlier_elements": elements_str,
                "geochem": geochem,
                "flags": flags_list,
                "significance": significance,
            }
        )
    common_errors = []
    if outlier_rows:
        from collections import Counter
        strat_counts = Counter(r["recorded_as"] for r in outlier_rows if r["recorded_as"])
        common_errors = [{"recorded_as": s, "count": c} for s, c in strat_counts.most_common(10)]

    summary_metrics = {
        "assay_intervals": len(assay_logger_df),
        "logging_intervals": len(logging_logger_df),
        "unique_holes": assay_logger_df[hole_col].nunique(),
        "strat_codes": assay_logger_df[strat_col].nunique(),
        "total_depth_m": _metric_total_depth(assay_logger_df, hole_col, depth_from_col, depth_to_col),
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
    mineral_coverage = _mineral_assay_coverage_counts(assay_logger_df, mineral_col, chem_actual_cols)
    mineralised_intervals = int(mineral_coverage["mineralised_intervals"])
    assays_available = int(mineral_coverage["assays_available"])
    mineral_assay_ratio = float(mineral_coverage["assay_ratio_pct"])

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

    outlier_box_data, outlier_box_layout = _plotly_outlier_box_json(
        project_filtered_df, strat_col, chem_actual_cols
    )
    # Use lower point caps for large reports to avoid MemoryError when embedding chart JSON
    n_assay = len(assay_logger_df)
    max_pts_pca = min(2000, max(600, n_assay // 5))
    max_pts_grid = 400 if n_assay > 5000 else 500
    outlier_scatter_data, outlier_scatter_layout = _plotly_outlier_scatter_grid_json(
        assay_logger_df, strat_col, chem_actual_cols,
        min_strat_count=10, max_points_per_plot=max_pts_grid
    )
    outlier_pca_data, outlier_pca_layout = _plotly_pca_biplot_json(
        assay_logger_df, strat_col, chem_actual_cols,
        min_strat_count=10, max_points=max_pts_pca
    )

    report_data: ReportData = {
        "meta": {
            "logger": logger_value,
            "date_from": date_from or "",
            "date_to": date_to or "",
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        "overview": build_overview_data(
            assay_logger_df,
            team_data_source,
            strat_col,
            assay_received_count,
            assay_outstanding_count,
            average_logging_interval_m,
            comment_stats_logging,
        ),
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
            "mismatch_intervals": mineral_mismatch_intervals,
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
            "total_misclassified": total_misclassified,
            "displayed_count": len(outlier_rows),
            "common_errors": common_errors,
        },
        "outlier_box_plot_data": outlier_box_data,
        "outlier_box_plot_layout": outlier_box_layout,
        "outlier_scatter_data": outlier_scatter_data,
        "outlier_scatter_layout": outlier_scatter_layout,
        "outlier_pca_data": outlier_pca_data,
        "outlier_pca_layout": outlier_pca_layout,
        "map": map_data,
        "project_codes": logger_project_codes,
        "has_project_scope": has_project_scope,
        "population_stats": pop_stats.summary_dict(),
        "logging_detail_issue_types": [
            # Major element checks first
            {"key": "fines", "label_fr": "Fines / mineralisation vs gangue", "label_en": "Fines / mineralisation vs gangue", "rules_fr": "Proportion des intervalles ou la geochimie (Fe, SiO2, Al2O3, gangue) suggere un probleme: ex. mineralisation loggee comme gangue, silice friable, argiles.", "rules_en": "Proportion of intervals where geochemistry suggests an issue: e.g. mineralisation logged as gangue, friable silica, clays."},
            {"key": "clay", "label_fr": "Argiles vs Al2O3%", "label_en": "Clay logging vs Al2O3%", "rules_fr": "Al2O3% eleve (>P75/P90) indique des mineraux argileux (kaolinite, gibbsite). Flag: Al2O3 eleve mais pas d'argiles ou shales logges.", "rules_en": "Elevated Al2O3% (above dataset P75/P90) indicates clay minerals (kaolinite, gibbsite). Flag: Al2O3 elevated but no clay or shale minerals logged."},
            {"key": "magnetite", "label_fr": "Magnetite vs LOI1000", "label_en": "Magnetite vs LOI1000", "rules_fr": "LOI1000 negatif suggere magnetite; codes MT, MBH, MBM. Flag: LOI1000 negatif mais pas de magnetite loggee.", "rules_en": "Negative LOI1000 suggests magnetite; codes MT, MBH, MBM. Flag: LOI1000 negative but no magnetite logged."},
            {"key": "goethite", "label_fr": "Goethite (Hy) vs LOI", "label_en": "Goethite (Hy) vs LOI", "rules_fr": "Zonation Hy (goethite) elevee doit correspondre a un LOI (ex. LOI 425) eleve. Flag: Hy eleve mais LOI bas ou manquant.", "rules_en": "High zonation Hy (goethite) should align with LOI (e.g. LOI 425). Flag: high Hy but low or missing LOI."},
            # Minor element checks
            {"key": "carbonate_gangue", "label_fr": "Gangue carbonate vs CaO/MgO%", "label_en": "Carbonate gangue vs CaO/MgO%", "rules_fr": "CaO/MgO eleves (>P75) indiquent des carbonates. Flag: CaO/MgO eleves mais pas de gangue carbonate loggee.", "rules_en": "Elevated CaO/MgO% (above dataset P75/P90) indicates carbonate minerals. Flag: assay shows elevated CaO/MgO but no carbonate gangue logged."},
            {"key": "sulphide_gangue", "label_fr": "Gangue sulfure vs S%", "label_en": "Sulphide gangue vs S%", "rules_fr": "S% eleve (>P75) indique des sulfures. Flag: S eleve mais pas de gangue sulfure loggee.", "rules_en": "Elevated S% (above dataset P75/P90) indicates sulphide minerals. Flag: assay shows elevated S but no sulphide gangue logged."},
            {"key": "manganese_gangue", "label_fr": "Gangue manganese vs Mn%", "label_en": "Manganese gangue vs Mn%", "rules_fr": "Mn% eleve (>P75) indique des mineraux de manganese. Flag: Mn eleve mais pas de gangue manganese loggee.", "rules_en": "Elevated Mn% (above dataset P75/P90) indicates manganese minerals. Flag: assay shows elevated Mn but no manganese gangue logged."},
            {"key": "mafics_gangue", "label_fr": "Mafiques vs MgO/CaO/Cr/Ni/K2O", "label_en": "Mafics vs MgO/CaO/Cr/Ni/K2O", "rules_fr": "Mineraux mafiques indiques par MgO, CaO, Cr2O3, NiO, K2O eleves. Flag: indicateurs eleves mais pas de mafiques logges.", "rules_en": "Mafic minerals indicated by elevated MgO, CaO, Cr2O3, NiO, K2O. Flag: assay shows elevated mafic indicators but no mafics gangue logged. Thresholds: dataset P75 (Low) / P90 (High)."},
            {"key": "magnesium_gangue", "label_fr": "Gangue magnesium vs MgO%", "label_en": "Magnesium gangue vs MgO%", "rules_fr": "MgO% eleve (>P75) suggere des mineraux magnesiens. Flag: MgO eleve mais pas de gangue magnesium ou carbonate loggee.", "rules_en": "Elevated MgO% (above dataset P75/P90) suggests magnesian minerals. Flag: assay shows elevated MgO but no magnesium or carbonate gangue logged."},
        ],
    }

    fines_with_images = _build_intervals_with_images(
        data_coordinator, fines_intervals, include_images, image_cache=image_cache
    )
    for i, item in enumerate(fines_with_images):
        if i < len(fines_intervals) and "geochem" in fines_intervals[i]:
            item["geochem"] = fines_intervals[i]["geochem"]

    magnetite_with_images = _build_intervals_with_images(
        data_coordinator, magnetite_intervals, include_images, image_cache=image_cache
    )
    goethite_with_images = _build_intervals_with_images(
        data_coordinator, goethite_intervals, include_images, image_cache=image_cache
    )
    carbonate_with_images = _build_intervals_with_images(
        data_coordinator, carbonate_intervals, include_images, image_cache=image_cache
    )
    sulphide_with_images = _build_intervals_with_images(
        data_coordinator, sulphide_intervals, include_images, image_cache=image_cache
    )
    manganese_with_images = _build_intervals_with_images(
        data_coordinator, manganese_intervals, include_images, image_cache=image_cache
    )
    mafics_with_images = _build_intervals_with_images(
        data_coordinator, mafics_intervals, include_images, image_cache=image_cache
    )
    magnesium_with_images = _build_intervals_with_images(
        data_coordinator, magnesium_intervals, include_images, image_cache=image_cache
    )
    for i, item in enumerate(magnetite_with_images):
        if i < len(magnetite_intervals):
            if "geochem" in magnetite_intervals[i]:
                item["geochem"] = magnetite_intervals[i]["geochem"]
            if "assay" in magnetite_intervals[i]:
                item["assay"] = magnetite_intervals[i]["assay"]
    for i, item in enumerate(goethite_with_images):
        if i < len(goethite_intervals):
            if "geochem" in goethite_intervals[i]:
                item["geochem"] = goethite_intervals[i]["geochem"]
            if "assay" in goethite_intervals[i]:
                item["assay"] = goethite_intervals[i]["assay"]
    for i, item in enumerate(carbonate_with_images):
        if i < len(carbonate_intervals):
            if "geochem" in carbonate_intervals[i]:
                item["geochem"] = carbonate_intervals[i]["geochem"]
            if "assay" in carbonate_intervals[i]:
                item["assay"] = carbonate_intervals[i]["assay"]
    for i, item in enumerate(sulphide_with_images):
        if i < len(sulphide_intervals) and "assay" in sulphide_intervals[i]:
            item["assay"] = sulphide_intervals[i]["assay"]
    for i, item in enumerate(manganese_with_images):
        if i < len(manganese_intervals) and "assay" in manganese_intervals[i]:
            item["assay"] = manganese_intervals[i]["assay"]
    for i, item in enumerate(mafics_with_images):
        if i < len(mafics_intervals) and "assay" in mafics_intervals[i]:
            item["assay"] = mafics_intervals[i]["assay"]
    for i, item in enumerate(magnesium_with_images):
        if i < len(magnesium_intervals) and "assay" in magnesium_intervals[i]:
            item["assay"] = magnesium_intervals[i]["assay"]

    clay_with_images = _build_intervals_with_images(
        data_coordinator, clay_intervals, include_images, image_cache=image_cache
    )
    for i, item in enumerate(clay_with_images):
        if i < len(clay_intervals) and "assay" in clay_intervals[i]:
            item["assay"] = clay_intervals[i]["assay"]

    grouping_with_groups = _build_grouping_groups_with_images(
        data_coordinator, grouping_groups_for_review, include_images, image_cache=image_cache
    )

    intervals_for_review: IntervalsForReview = {
        "fines": fines_with_images,
        "logging_detail": {
            "fines": fines_with_images,
            "clay": clay_with_images,
            "magnetite": magnetite_with_images,
            "goethite": goethite_with_images,
            "carbonate_gangue": carbonate_with_images,
            "sulphide_gangue": sulphide_with_images,
            "manganese_gangue": manganese_with_images,
            "mafics_gangue": mafics_with_images,
            "magnesium_gangue": magnesium_with_images,
        },
        "grouping_flat": _build_intervals_with_images(
            data_coordinator, grouping_intervals, include_images, image_cache=image_cache
        ),
        "grouping": grouping_with_groups,
        "outliers": _build_outlier_intervals_with_images(
            data_coordinator, outlier_rows, include_images, image_cache=image_cache
        ),
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            render_html(
                report_data, intervals_for_review, logo_path, page_options,
                charts_dir=charts_dir, stream=f
            )
        return output_path
    return render_html(
        report_data, intervals_for_review, logo_path, page_options, charts_dir=charts_dir
    )
