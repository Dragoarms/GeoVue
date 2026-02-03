"""Grouping tab: render HTML section."""
import html
from typing import Any, Dict

import numpy as np

from ..tables import _render_grouping_groups


def render_grouping_section(report: Dict[str, Any], intervals: Dict[str, Any], logger_id: str) -> str:
    """Render the grouping tab HTML section."""
    gkpis = report.get("grouping_kpis", {})
    avg_grp = gkpis.get("avg_group_interval_m")
    max_grp = gkpis.get("max_group_interval_m")
    avg_grp_str = (
        f"{avg_grp:.2f}"
        if avg_grp is not None and not (isinstance(avg_grp, float) and np.isnan(avg_grp))
        else "n/a"
    )
    max_grp_str = (
        f"{max_grp:.2f}"
        if max_grp is not None and not (isinstance(max_grp, float) and np.isnan(max_grp))
        else "n/a"
    )
    grouping_groups = intervals.get("grouping", [])
    grouping_summary = report.get("grouping_summary", [])
    grouping_columns_used = report.get("grouping_columns_used", [])

    groups_html = _render_grouping_groups(grouping_groups, logger_id)
    summary_list_html = "".join(f"<li>{html.escape(line)}</li>" for line in grouping_summary)
    columns_str = html.escape(", ".join(grouping_columns_used))

    return f"""
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
                    {summary_list_html}
                </ul>
            </div>
            <div class="info-box rules-box">
                <h4 data-i18n-fr="Regles de regroupement" data-i18n-en="Grouping rules">Regles de regroupement</h4>
                <p data-i18n-fr="Colonnes de regroupement utilisees:" data-i18n-en="Grouping columns used:">
                    Grouping columns used: <code>{columns_str}</code>
                </p>
                <p data-i18n-fr="Longueur d'intervalle = max(profondeur_to) − min(profondeur_from) par groupe. CV = std/mean × 100 pour chaque element; les groupes avec CV > 100% sont signales. Si les donnees incluent les memes colonnes que combine_intervals (ex. Prospect_D, StratSum, Min_*_pct, LithComments), la logique est alignee; sinon on utilise le sous-ensemble present."
                    data-i18n-en="Interval length = max(depth_to) − min(depth_from) per group. CV = std/mean × 100 for each assay element; groups with CV > 100% are flagged. When data includes the same grouping columns as Logging Review combine_intervals (e.g. Prospect_D, StratSum, Min_*_pct, LithComments), grouping matches that logic; otherwise the subset present is used.">
                    Interval length = max(depth_to) − min(depth_from) per group. CV = std/mean × 100 for each assay element; groups with CV &gt; 100% are flagged. When data includes the same grouping columns as combine_intervals, grouping matches that logic; otherwise the subset present is used.
                </p>
            </div>
            <div class="intervals-section">
                <h3 data-i18n-fr="Groupes a revoir (CV eleve Fe, SiO2, Al2O3, MgO, CaO, S)" data-i18n-en="Groups for review (high CV)">Groupes a revoir</h3>
                {groups_html}
            </div>
            <div class="notes-box">
                <label data-i18n-fr="Notes du reviseur" data-i18n-en="Reviewer notes">Notes du reviseur</label>
                <textarea data-note-id="{html.escape(logger_id)}::grouping" placeholder="Ajouter des notes..."></textarea>
            </div>
        </section>
    """
