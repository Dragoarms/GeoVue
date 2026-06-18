"""Comments tab: render HTML section."""
import html
from typing import Any, Dict


def render_comments_section(
    report: Dict[str, Any],
    logger_id: str,
    wordcloud_html: str,
    comment_bar_chart_html: str,
) -> str:
    """Render the comments tab HTML section. Wordcloud and bar chart HTML are pre-rendered by the caller."""
    return f"""
        <section class="tab-panel" data-tab="comments">
            <div class="panel-header">
                <h2 data-i18n-fr="Utilisation des commentaires" data-i18n-en="Comment usage">Utilisation des commentaires</h2>
            </div>
            <div class="two-panel">
                <div class="panel-card">
                    <h3 data-i18n-fr="Nuage de mots" data-i18n-en="Word cloud">Nuage de mots</h3>
                    {wordcloud_html}
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
                <textarea data-note-id="{html.escape(logger_id)}::comments" placeholder="Ajouter des notes..."></textarea>
            </div>
        </section>
    """
