"""Mineralisation tab: render HTML section."""
import html
from typing import Any, Dict

import pandas as pd

from ..charts import _plotly_pie_json, _plotly_stacked_bar_pct_json
from ..tables import _render_mineralisation_evidence_table


def render_mineralisation_section(report: Dict[str, Any], logger_id: str) -> str:
    """Render the mineralisation tab HTML section."""
    min_acc = report.get("mineralisation", {})
    acc_counts = min_acc.get("accuracy_counts", {})
    acc_labels = []
    acc_values = []
    for k in ["Match", "Mismatch", "Pending Assays"]:
        if acc_counts.get(k, 0) > 0:
            acc_labels.append(k)
            acc_values.append(int(acc_counts[k]))
    mineral_pie_user_data, mineral_pie_user_layout = (
        _plotly_pie_json(
            acc_labels or ["N/A"],
            acc_values or [0],
            "Votre precision (Overall accuracy)",
        )
        if (acc_labels or acc_values)
        else ("[]", "{}")
    )
    q_user = min_acc.get("quarterly_user", pd.DataFrame())
    if not q_user.empty and "Quarter" in q_user.columns:
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
            quarters_user,
            match_user,
            mismatch_user,
            pending_user,
            "Votre precision par trimestre (% of quarter)",
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
    mineral_pie_team_data, mineral_pie_team_layout = (
        _plotly_pie_json(
            acc_labels_t or ["N/A"],
            acc_values_t or [0],
            "Equipe (Overall accuracy)",
        )
        if (acc_labels_t or acc_values_t)
        else ("[]", "{}")
    )
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
            quarters_team,
            match_team,
            mismatch_team,
            pending_team,
            "Equipe par trimestre (% of quarter)",
        )
    else:
        mineral_bar_team_data, mineral_bar_team_layout = "[]", "{}"

    mismatch_intervals = sorted(
        report.get("mineralisation", {}).get("mismatch_intervals", []),
        key=lambda x: 0 if x.get("significance") == "High" else 1,
    )
    evidence_table_html = _render_mineralisation_evidence_table(mismatch_intervals, logger_id)
    min_data = report.get("mineralisation", {})

    return f"""
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
                    <li data-i18n-html-fr="<strong>Match</strong> = Le logging est coherent avec les essais (zonation Un/Le ou gangue &gt;=15% = non-mineralise; De/Hy/Pr ou gangue &lt;15% = mineralise; accord avec l'essai)"
                        data-i18n-html-en="<strong>Match</strong> = Logging is consistent with assays (zonation Un/Le or gangue &gt;=15% = unmineralised; De/Hy/Pr or gangue &lt;15% = mineralised; agrees with assay)">
                        <strong>Match</strong> = Logging is consistent with assays (zonation Un/Le or gangue &gt;=15% = unmineralised; De/Hy/Pr or gangue &lt;15% = mineralised; agrees with assay)
                    </li>
                    <li data-i18n-html-fr="<strong>Mismatch</strong> = Le logging est incoherent avec les essais (ex. zonation mineralise ou gangue &lt;15% mais essai montre non-mineralise)"
                        data-i18n-html-en="<strong>Mismatch</strong> = Logging inconsistent with assays (e.g. mineralised zonation or gangue &lt;15% but assay shows unmineralised)">
                        <strong>Mismatch</strong> = Logging inconsistent with assays (e.g. mineralised zonation or gangue &lt;15% but assay shows unmineralised)
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
                    <strong>{min_data.get('mismatch_low_count', 0)}</strong>
                    <span data-i18n-fr="Low (borderline)" data-i18n-en="Low (borderline)">Low (borderline)</span>,
                    <strong>{min_data.get('mismatch_high_count', 0)}</strong>
                    <span data-i18n-fr="High" data-i18n-en="High">High</span>
                    <span class="stats-note" data-i18n-fr="(borderline = assay suggests Leached; ne pas penaliser les erreurs limite)" data-i18n-en="(borderline = assay suggests Leached; do not penalise borderline errors)">(borderline = assay suggests Leached; do not penalise borderline errors)</span>
                </p>
                {evidence_table_html}
            </div>
            <div class="notes-box">
                <label data-i18n-fr="Notes du reviseur" data-i18n-en="Reviewer notes">Notes du reviseur</label>
                <textarea data-note-id="{html.escape(logger_id)}::mineralisation" placeholder="Ajouter des notes..."></textarea>
            </div>
        </section>
    """
