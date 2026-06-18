"""Overview tab: build data and render HTML."""
import html
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..charts import _plotly_strat_grouped_bar_pct_json
from ..collar_map import _render_map
from ..utils import _format_metric


def build_overview_data(
    assay_logger_df: pd.DataFrame,
    team_data_source: pd.DataFrame,
    strat_col: str,
    assay_received_count: int,
    assay_outstanding_count: int,
    average_logging_interval_m: Optional[float],
    comment_stats_logging: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the overview tab data dict for report_data['overview']."""
    strat_code_counts = (
        assay_logger_df[strat_col].value_counts().head(30).to_dict()
        if strat_col in assay_logger_df.columns
        else {}
    )
    strat_code_list = [{"code": k, "count": int(v)} for k, v in strat_code_counts.items()]

    team_strat_code_counts = (
        team_data_source[strat_col].value_counts().head(30).to_dict()
        if strat_col in team_data_source.columns
        else {}
    )
    team_strat_code_list = [{"code": k, "count": int(v)} for k, v in team_strat_code_counts.items()]

    return {
        "assay_received_count": assay_received_count,
        "assay_outstanding_count": assay_outstanding_count,
        "average_logging_interval_m": average_logging_interval_m,
        "strat_code_list": strat_code_list,
        "team_strat_code_list": team_strat_code_list,
        "comment_ratio_pct_logging": comment_stats_logging["comment_ratio_pct"],
        "comment_ratio_pct": comment_stats_logging["comment_ratio_pct"],
    }


def render_overview_section(report: Dict[str, Any]) -> str:
    """Render the overview tab HTML section."""
    overview = report.get("overview", {})
    assay_received = overview.get("assay_received_count", report["summary"].get("assay_intervals", 0))
    assay_outstanding = overview.get("assay_outstanding_count", 0)
    avg_log_m = overview.get("average_logging_interval_m")
    avg_log_str = (
        f"{avg_log_m:.2f}"
        if avg_log_m is not None and not (isinstance(avg_log_m, float) and np.isnan(avg_log_m))
        else "n/a"
    )
    strat_list = overview.get("strat_code_list", [])
    strat_bar_data = strat_list[:20]
    comment_ratio_logging = overview.get("comment_ratio_pct_logging", report.get("comment_coverage", 0))
    if isinstance(comment_ratio_logging, (int, float)) and not np.isnan(comment_ratio_logging):
        comment_ratio_str = f"{comment_ratio_logging:.1f}%"
    else:
        comment_ratio_str = "n/a"

    team_strat_list = overview.get("team_strat_code_list", [])
    logger_counts_by_code = {str(s["code"]): int(s["count"]) for s in strat_bar_data}
    team_counts_by_code = {str(s["code"]): int(s["count"]) for s in team_strat_list}
    all_codes_set = set(logger_counts_by_code.keys()) | set(team_counts_by_code.keys())
    all_codes_sorted = sorted(
        all_codes_set,
        key=lambda c: (logger_counts_by_code.get(c, 0) + team_counts_by_code.get(c, 0)),
        reverse=True,
    )[:25]
    logger_counts_list = [logger_counts_by_code.get(c, 0) for c in all_codes_sorted]
    team_counts_list = [team_counts_by_code.get(c, 0) for c in all_codes_sorted]
    if all_codes_sorted:
        strat_grouped_data, strat_grouped_layout = _plotly_strat_grouped_bar_pct_json(
            all_codes_sorted,
            logger_counts_list,
            team_counts_list,
            "Strat codes: Logger vs Team (% of intervals)",
        )
    else:
        strat_grouped_data, strat_grouped_layout = "[]", "{}"

    date_from_str = report["meta"].get("date_from") or ""
    date_to_str = report["meta"].get("date_to") or ""
    overview_hero_date = f"From: {date_from_str or '-'} To: {date_to_str or '-'}"
    return f"""
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
