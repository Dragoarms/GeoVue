"""Orchestration for HTML report generation: public entry points."""
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from processing.DataManager.column_aliases import ColumnResolver
from reports.logging_review.data.columns import (
    resolve_chemistry_columns,
    resolve_drilldate_column,
    resolve_logger_column,
)
from reports.logging_review.data.outliers import compute_hybrid_outlier_scores, predict_most_likely_strat
from reports.logging_review.data.prep import (
    _build_logging_interval_dataframe,
    _log_dataframe_overview,
    _merge_logger_by_overlap,
    _merge_rc_metrics_by_overlap,
    _resolve_logging_interval_columns,
    build_merged_qaqc_dataframe,
    fill_empty_logger_values,
    filter_dataframe_by_logger_and_date,
    get_collar_dataframe,
)

from .report_builder import build_html_report

logger = logging.getLogger(__name__)


def _ensure_outlier_predictions(
    merged_df: pd.DataFrame,
    strat_col: str,
    chem_actual_cols: List[str],
) -> pd.DataFrame:
    """Ensure merged data carries report-ready outlier scores and population predictions."""
    result = merged_df.copy()
    if "outlier_score" not in result.columns:
        outlier_scores = compute_hybrid_outlier_scores(
            result,
            strat_col=strat_col,
            chem_cols=chem_actual_cols,
            min_group_size=10,
        )
        result = result.join(outlier_scores)
    if "most_likely_strat" not in result.columns:
        result["most_likely_strat"] = predict_most_likely_strat(
            result,
            strat_col=strat_col,
            chem_cols=chem_actual_cols,
            min_group_size=10,
        )
    return result


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
    image_mode: str = "thumbnail",
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
        # From src/reports/logging_review/html/ -> src/resources
        default_logo = Path(__file__).resolve().parents[4] / "resources" / "full_logo.png"
        logo_path = str(default_logo) if default_logo.exists() else None

    merged_df = _ensure_outlier_predictions(merged_df, strat_col, chem_actual_cols)

    if not skip_csv_export:
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

        # Stream report directly to file to avoid MemoryError (evidence tables with many base64 images)
        build_html_report(
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
            output_path=output_path,
            image_mode=image_mode,
        )

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
    image_mode: str = "thumbnail",
    logo_path: Optional[str] = None,
) -> List[str]:
    """Generate per-logger HTML reports (full pipeline: build data then render)."""
    from processing.logging_review_html_report import _find_logger_source

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

    chem_cols = resolve_chemistry_columns(merged_df)
    chem_actual_cols = list(chem_cols.values())
    merged_df = _ensure_outlier_predictions(
        merged_df,
        strat_col=strat_col,
        chem_actual_cols=chem_actual_cols,
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

    if logger_col in logging_df.columns:
        logging_df = fill_empty_logger_values(logging_df, logger_col, logging_hole_col)
    if logger_col in merged_df.columns:
        merged_df = fill_empty_logger_values(merged_df, logger_col, hole_col)

    if logo_path is None:
        default_logo = Path(__file__).resolve().parents[4] / "resources" / "full_logo.png"
        logo_path = str(default_logo) if default_logo.exists() else None

    output_files = []
    collar_df_for_report = get_collar_dataframe(data_coordinator)
    for logger_value in logger_values:
        assay_logger_df = merged_df[merged_df[logger_col].astype(str) == str(logger_value)].copy()
        logging_logger_df = logging_df[logging_df[logger_col].astype(str) == str(logger_value)].copy()
        if assay_logger_df.empty and logging_logger_df.empty:
            continue

        filename = f"RC_Logging_Review_{logger_value}.html"
        output_path = os.path.join(output_dir, filename)

        build_html_report(
            data_coordinator=data_coordinator,
            logger_value=str(logger_value),
            assay_logger_df=assay_logger_df,
            logging_logger_df=logging_logger_df,
            merged_df=merged_df,
            logging_df=logging_df,
            collar_df=collar_df_for_report,
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
            output_path=output_path,
            image_mode=image_mode,
        )

        output_files.append(output_path)

    elapsed = datetime.now() - start_time
    logger.info("HTML report generation complete in %s", elapsed)
    return output_files
