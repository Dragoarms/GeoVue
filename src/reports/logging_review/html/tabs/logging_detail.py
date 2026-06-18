"""Logging detail (fines etc.) tab: render HTML section."""
import html
from typing import Any, Dict

from ..tables import _render_logging_detail_evidence_table


def render_logging_detail_section(
    report: Dict[str, Any],
    intervals: Dict[str, Any],
    logger_id: str,
) -> str:
    """Render the logging-detail tab HTML section."""
    logging_detail_issue_types = report.get("logging_detail_issue_types", [])
    subtab_buttons = [
        '<div class="logging-detail-subnav" role="tablist">',
        '<button type="button" class="logging-detail-subtab active" data-logging-detail-subtab="summary" '
        'data-i18n-fr="Resume" data-i18n-en="Summary">Summary</button>',
    ]
    for it in logging_detail_issue_types:
        key = it.get("key", "")
        label_en = it.get("label_en", key)
        subtab_buttons.append(
            f'<button type="button" class="logging-detail-subtab" data-logging-detail-subtab="{html.escape(key)}" '
            f'data-i18n-en="{html.escape(label_en)}">{html.escape(label_en)}</button>'
        )
    subtab_buttons.append("</div>")
    subnav_html = "".join(subtab_buttons)

    # Build aggregated summary across ALL sub-tabs
    logging_detail = intervals.get("logging_detail") or {}
    issue_summary_lines = []
    total_issues = 0
    total_high = 0
    for it in logging_detail_issue_types:
        key = it.get("key", "")
        label_en = it.get("label_en", key)
        issue_list = logging_detail.get(key, [])
        count = len(issue_list)
        high_count = sum(1 for i in issue_list if i.get("significance") == "High")
        total_issues += count
        total_high += high_count
        if count > 0:
            issue_summary_lines.append(
                f"{label_en}: {count} issue(s) ({high_count} High)"
            )

    fines_summary = report.get("fines_summary", [])

    if total_issues > 0:
        overview_line = f"Total: {total_issues} issues across all checks ({total_high} High significance)"
    else:
        overview_line = "No logging detail issues found."

    summary_panel_html = f"""
            <div class="panel-card">
                <h3 data-i18n-fr="Resume" data-i18n-en="Summary">Summary</h3>
                <p style="font-weight:600; margin-bottom:8px;">{html.escape(overview_line)}</p>
                <ul class="summary-list">
                    {''.join(f"<li>{html.escape(line)}</li>" for line in issue_summary_lines)}
                </ul>
                <h4 style="margin-top:12px;" data-i18n-fr="Detail (fines)" data-i18n-en="Fines detail">Fines detail</h4>
                <ul class="summary-list">
                    {''.join(f"<li>{html.escape(line)}</li>" for line in fines_summary)}
                </ul>
            </div>
            <div class="notes-box">
                <label data-i18n-fr="Notes du reviseur" data-i18n-en="Reviewer notes">Notes du reviseur</label>
                <textarea data-note-id="{html.escape(logger_id)}::logging-detail" placeholder="Ajouter des notes..."></textarea>
            </div>"""

    subpanels = [
        f'<div class="logging-detail-subpanel active" data-logging-detail-subpanel="summary" role="tabpanel">'
        f'{summary_panel_html}</div>'
    ]
    for it in logging_detail_issue_types:
        key = it.get("key", "")
        label_fr = it.get("label_fr", key)
        label_en = it.get("label_en", key)
        rules_fr = it.get("rules_fr", "")
        rules_en = it.get("rules_en", "")
        issue_intervals = logging_detail.get(key, [])
        subpanels.append(f"""
            <div class="logging-detail-subpanel" data-logging-detail-subpanel="{html.escape(key)}" role="tabpanel">
                <div class="panel-card">
                    <h3 data-i18n-fr="{html.escape(label_fr)}" data-i18n-en="{html.escape(label_en)}">{html.escape(label_en)}</h3>
                    <div class="rules-box">
                        <p data-i18n-fr="{html.escape(rules_fr)}" data-i18n-en="{html.escape(rules_en)}">{html.escape(rules_en)}</p>
                    </div>
                    <div class="intervals-section">
                        {_render_logging_detail_evidence_table(issue_intervals, logger_id, key)}
                    </div>
                </div>
                <div class="notes-box">
                    <label data-i18n-fr="Notes du reviseur" data-i18n-en="Reviewer notes">Notes du reviseur</label>
                    <textarea data-note-id="{html.escape(logger_id)}::logging-detail::{html.escape(key)}" placeholder="Ajouter des notes..."></textarea>
                </div>
            </div>""")

    return f"""
        <section class="tab-panel" data-tab="logging-detail">
            <div class="panel-header">
                <h2 data-i18n-fr="Precision du detail de logging" data-i18n-en="Logging detail accuracy">Precision du detail de logging</h2>
            </div>
            {subnav_html}
            <div class="logging-detail-subpanels">
                {''.join(subpanels)}
            </div>
        </section>
    """
