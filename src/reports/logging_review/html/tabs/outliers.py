"""Outliers tab: render HTML section."""
import html
from typing import Any, Dict

from ..tables import _render_outlier_table


def render_outlier_section(
    report: Dict[str, Any],
    intervals: Dict[str, Any],
    logger_id: str,
) -> str:
    """Render the outliers tab HTML section."""
    okpis = report.get("outlier_kpis", {})
    total_misclassified = okpis.get("total_misclassified", len(report.get("outliers", [])))
    displayed_count = okpis.get("displayed_count", total_misclassified)
    common_errors = okpis.get("common_errors", [])

    if common_errors:
        common_errors_html = '<ul class="common-errors-list">'
        for e in common_errors[:10]:
            common_errors_html += f"<li>Classifie comme <strong>{html.escape(str(e.get('recorded_as', '')))}</strong>: {e.get('count', 0)} intervalle(s)</li>"
        common_errors_html += "</ul>"
    else:
        common_errors_html = '<div class="empty" data-i18n-fr="Aucune erreur commune." data-i18n-en="No common errors.">Aucune.</div>'

    display_note = ""
    if displayed_count < total_misclassified:
        display_note = f' <span class="display-note" data-i18n-fr="(montrant {displayed_count})" data-i18n-en="(showing {displayed_count})">(montrant {displayed_count})</span>'

    outlier_box_data = report.get("outlier_box_plot_data", "[]")
    outlier_box_layout = report.get("outlier_box_plot_layout", "{}")
    outlier_scatter_data = report.get("outlier_scatter_data", "[]")
    outlier_scatter_layout = report.get("outlier_scatter_layout", "{}")
    outlier_pca_data = report.get("outlier_pca_data", "[]")
    outlier_pca_layout = report.get("outlier_pca_layout", "{}")

    outlier_intervals = intervals.get("outliers", [])
    outlier_table_html = _render_outlier_table(outlier_intervals, logger_id)

    return f"""
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
            <div class="panel-card">
                <h3 data-i18n-fr="Gammes geochimiques equipe par strat" data-i18n-en="Team geochem ranges by strat">Gammes geochimiques equipe par strat</h3>
                <div id="outlier-box-plot" class="plotly-chart" data-plotly-data="{html.escape(outlier_box_data)}" data-plotly-layout="{html.escape(outlier_box_layout)}"></div>
            </div>
            <div class="panel-card">
                <h3 data-i18n-fr="Clusters geochimiques (PCA multivarié)" data-i18n-en="Geochem clusters (multivariate PCA)">Clusters geochimiques (PCA multivarié)</h3>
                <p class="chart-description" data-i18n-fr="Projection PCA en espace CLR. Les vecteurs montrent quels éléments séparent les groupes. Les points rouges (X) sont les anomalies détectées." data-i18n-en="PCA projection in CLR space. Vectors show which elements separate groups. Red X markers are detected outliers.">PCA projection in CLR space. Vectors show which elements separate groups. Red X markers are detected outliers.</p>
                <div id="outlier-pca-plot" class="plotly-chart" data-plotly-data="{html.escape(outlier_pca_data)}" data-plotly-layout="{html.escape(outlier_pca_layout)}"></div>
            </div>
            <div class="panel-card">
                <h3 data-i18n-fr="Paires d'éléments (convex hulls par strat)" data-i18n-en="Element pairs (convex hulls per strat)">Element pairs (convex hulls per strat)</h3>
                <p class="chart-description" data-i18n-fr="Tous les strats sont affichés; les hulls sont tracés seulement pour les strats avec assez d'échantillons. Chaque graphique a ses propres axes." data-i18n-en="All strats are shown; hulls are drawn only for strats with enough samples. Each plot has its own axes.">All strats are shown; hulls are drawn only for strats with enough samples. Each plot has its own axes.</p>
                <div id="outlier-scatter-plot" class="plotly-chart" data-plotly-data="{html.escape(outlier_scatter_data)}" data-plotly-layout="{html.escape(outlier_scatter_layout)}"></div>
            </div>
            <div class="intervals-section">
                <h3 data-i18n-fr="Intervalles a revoir (Enregistre vs Plus probable / Drapeaux)" data-i18n-en="Intervals for review (Recorded vs Likely / Flags)">Intervalles a revoir</h3>
                {outlier_table_html}
            </div>
            <div id="outlier-details-modal" class="outlier-modal" role="dialog" aria-hidden="true">
                <div class="outlier-modal-backdrop"></div>
                <div class="outlier-modal-content">
                    <div class="outlier-modal-header">
                        <h3 data-i18n-en="Outlier details">Outlier details</h3>
                        <button type="button" class="outlier-modal-close" aria-label="Close">&times;</button>
                    </div>
                    <div class="outlier-modal-body">
                        <p class="outlier-detail-hole"></p>
                        <p class="outlier-detail-depth"></p>
                        <p class="outlier-detail-recorded"></p>
                        <p class="outlier-detail-likely"></p>
                        <p class="outlier-detail-reason"></p>
                        <div class="outlier-detail-geochem"></div>
                    </div>
                </div>
            </div>
            <div class="notes-box">
                <label data-i18n-fr="Notes du reviseur" data-i18n-en="Reviewer notes">Notes du reviseur</label>
                <textarea data-note-id="{html.escape(logger_id)}::outliers" placeholder="Ajouter des notes..."></textarea>
            </div>
        </section>
    """
