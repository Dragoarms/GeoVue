"""Profile (zonation) tab: render HTML section."""
import html
from typing import Any, Dict

from ..charts import _plotly_zonation_bar_pct_json
from ..tables import _render_zonation_evidence_table


def render_profile_section(report: Dict[str, Any], logger_id: str) -> str:
    """Render the profile/zonation tab HTML section."""
    pz = report.get("profile_zonation", {})
    z_cats = ["Un", "Le", "De", "Hy", "Pr"]
    z_correct = [pz.get("correct_counts", {}).get(c, 0) for c in z_cats]
    z_mismatch = [pz.get("mismatch_counts", {}).get(c, 0) for c in z_cats]
    z_correct_team = [pz.get("correct_counts_team", {}).get(c, 0) for c in z_cats]
    z_mismatch_team = [pz.get("mismatch_counts_team", {}).get(c, 0) for c in z_cats]
    total_zonation_data = sum(z_correct) + sum(z_mismatch)
    has_zonation_data = total_zonation_data > 0

    zonation_bar_data, zonation_bar_layout = _plotly_zonation_bar_pct_json(
        z_cats,
        z_correct,
        z_mismatch,
        "Logger: Accuracy by category (%)",
    )
    zonation_bar_data_team, zonation_bar_layout_team = _plotly_zonation_bar_pct_json(
        z_cats,
        z_correct_team,
        z_mismatch_team,
        "Team: Accuracy by category (%)",
    )
    zonation_mismatch_rows = pz.get("mismatch_rows", [])
    zonation_evidence_html = _render_zonation_evidence_table(zonation_mismatch_rows, logger_id)

    zonation_data_warning = ""
    if not has_zonation_data:
        zonation_data_warning = '''<div class="warning-box">
            <h4 data-i18n-fr="⚠️ Donnees de zonation non disponibles" data-i18n-en="⚠️ Zonation data not available">⚠️ Donnees de zonation non disponibles</h4>
            <p data-i18n-fr="Les colonnes de zonation (BestProfileZonation_D, Total Gangue Logged, De % Logged, Hy % Logged, Pr % Logged) n'ont pas ete trouvees ou sont vides dans les donnees."
               data-i18n-en="Zonation columns (BestProfileZonation_D, Total Gangue Logged, De % Logged, Hy % Logged, Pr % Logged) were not found or are empty in the data.">
                Les colonnes de zonation n'ont pas ete trouvees ou sont vides dans les donnees.
            </p>
        </div>'''

    return f"""
        <section class="tab-panel" data-tab="profile">
            <div class="panel-header">
                <h2 data-i18n-fr="Zonation de profil" data-i18n-en="Profile zonation">Zonation de profil</h2>
            </div>
            {zonation_data_warning}
            <div class="two-panel">
                <div class="panel-card">
                    <h3 data-i18n-fr="Precision par categorie (Logger)" data-i18n-en="Accuracy by category (Logger)">Precision par categorie (Logger)</h3>
                    <div id="zonation-bar" class="plotly-chart" data-plotly-data="{html.escape(zonation_bar_data)}" data-plotly-layout="{html.escape(zonation_bar_layout)}"></div>
                </div>
                <div class="panel-card">
                    <h3 data-i18n-fr="Precision par categorie (Equipe)" data-i18n-en="Accuracy by category (Team)">Precision par categorie (Equipe)</h3>
                    <div id="zonation-bar-team" class="plotly-chart" data-plotly-data="{html.escape(zonation_bar_data_team)}" data-plotly-layout="{html.escape(zonation_bar_layout_team)}"></div>
                </div>
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
                <textarea data-note-id="{html.escape(logger_id)}::profile" placeholder="Ajouter des notes..."></textarea>
            </div>
        </section>
    """
