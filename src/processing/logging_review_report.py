import io
import logging
import os
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
from processing.DataManager.keys import ImageKey
from reports.logging_review.data.columns import (
    resolve_chemistry_columns,
    resolve_drilldate_column,
    resolve_logger_column,
)
from reports.logging_review.data.outliers import (
    _clr_transform,  # noqa: F401 - re-exported for existing tests/callers
    compute_hybrid_outlier_scores,
    predict_most_likely_strat,  # noqa: F401 - re-exported for logging_review_html_report
)
from reports.logging_review.data.prep import (
    _find_logger_source,
    _build_logging_interval_dataframe,
    _log_dataframe_overview,
    _merge_logger_by_overlap,
    _merge_rc_metrics_by_overlap,
    _resolve_logging_interval_columns,
    build_merged_qaqc_dataframe,
    fill_empty_logger_values,
    filter_dataframe_by_logger_and_date,
    get_collar_dataframe,
    get_logger_list_and_date_options,  # noqa: F401 - public facade for the GUI
    prepare_logging_review_data,  # noqa: F401 - public facade for the GUI
)

logger = logging.getLogger(__name__)


def _summarize_grouping_accuracy_by_interval(
    df: pd.DataFrame,
    group_cols: List[str],
    chem_cols: List[str],
) -> List[str]:
    """
    Summarize grouping accuracy by calculating CV for each assay element within groups.

    This function fills NaN values in grouping columns with 'UNKNOWN' placeholder to allow
    proper grouping (matches the approach in the original combine_intervals function).
    """
    # Use only grouping columns that exist
    group_cols_use = [c for c in group_cols if c in df.columns]
    if not group_cols_use:
        return ["Missing grouping columns for interval accuracy summary."]

    available = [c for c in chem_cols if c in df.columns]
    if not available:
        return ["Missing Fe/SiO2/Al2O3 columns for grouping accuracy summary."]

    # Work on a copy to avoid modifying the original DataFrame
    df_work = df.copy()

    # Fill missing values in grouping columns with 'UNKNOWN' placeholder to allow grouping
    # This matches the approach in the original combine_intervals function
    for col in group_cols_use:
        df_work[col] = df_work[col].fillna("UNKNOWN").astype(str).replace("", "UNKNOWN")

    grouped = df_work.groupby(group_cols_use)
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


def generate_logger_reports(
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
    output_format: str = "PDF",
    prepped_data: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Generate per-logger PDF or HTML reports.
    If prepped_data is provided (from prepare_logging_review_data), only filtering and HTML generation run.
    Returns (list of file paths created, list of logger names skipped because they had 0 rows in the filtered date range).
    """
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.perf_counter()
    geo_store = data_coordinator.geological_store if data_coordinator else None

    if prepped_data is not None:
        # Full team data for team statistics (read-only in HTML report; no copy needed)
        full_team_df = prepped_data["merged_df"]

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

    # Skip loggers with no data in the filtered date range; report them to the caller
    loggers_with_data: Set[str] = set()
    for v in logger_values:
        sv = str(v)
        if (merged_df[logger_col].astype(str) == sv).any() or (logging_df[logger_col].astype(str) == sv).any():
            loggers_with_data.add(v)
    skipped_loggers = [v for v in logger_values if v not in loggers_with_data]
    logger_values = [v for v in logger_values if v in loggers_with_data]
    if skipped_loggers:
        logger.info(
            "Skipping %d logger(s) with no data in selected date range: %s",
            len(skipped_loggers),
            ", ".join(str(s) for s in skipped_loggers),
        )

    if not logger_values:
        return ([], skipped_loggers)

    if logo_path is None:
        default_logo = Path(__file__).resolve().parents[1] / "resources" / "full_logo.png"
        logo_path = str(default_logo) if default_logo.exists() else None

    if output_format and output_format.strip().upper() == "HTML":
        from processing.logging_review_html_report import (
            generate_logger_html_reports_from_prepped_data,
        )

        output_files = generate_logger_html_reports_from_prepped_data(
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
            image_mode=image_mode,
            logo_path=logo_path,
            full_team_df=full_team_df,
        )
        return (output_files, skipped_loggers)

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

    return (output_files, skipped_loggers)


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
