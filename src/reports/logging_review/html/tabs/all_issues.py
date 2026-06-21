"""All Issues tab: master list of every flagged interval across all sections."""
import html
from typing import Any, Dict, List

from ..utils import _safe_str

ALL_ISSUES_ROW_LIMIT = 500


def _truncation_note(total: int, limit: int = ALL_ISSUES_ROW_LIMIT) -> str:
    if total <= limit:
        return ""
    extra = total - limit
    return (
        '<div class="evidence-truncation-note">'
        f"Showing first {limit:,} of {total:,} issues. "
        f"+{extra:,} more in the CSV export."
        "</div>"
    )


def render_all_issues_section(
    report: Dict[str, Any],
    intervals: Dict[str, Any],
    logger_id: str,
) -> str:
    """
    Render a master 'All Issues' tab: every flagged interval from all sections,
    grouped by project_code → hole_id → depth.  Gives the geologist a single
    checklist to work through in Acquire.
    """
    # Collect all issues from every section into a flat list
    all_rows: List[Dict[str, Any]] = []

    # 1. Mineralisation mismatches
    for item in report.get("mineralisation", {}).get("mismatch_intervals", []):
        all_rows.append({
            "section": "Mineralisation",
            "hole_id": item.get("hole_id", ""),
            "depth_from": item.get("depth_from"),
            "depth_to": item.get("depth_to"),
            "issue": f"Mismatch: logged as {item.get('logged_as', '?')}, assay suggests {item.get('assay_suggests', '?')}",
            "significance": item.get("significance", "Low"),
            "strat": item.get("logged_as", ""),
            "image": item.get("image"),
        })

    # 2. Zonation mismatches
    for item in report.get("profile_zonation", {}).get("mismatch_rows", []):
        all_rows.append({
            "section": "Zonation",
            "hole_id": item.get("hole_id", ""),
            "depth_from": item.get("depth_from"),
            "depth_to": item.get("depth_to"),
            "issue": f"Zonation: logged {item.get('logged_zonation', '?')}, should be {item.get('should_be', '?')}",
            "significance": item.get("significance", "Low"),
            "strat": item.get("logged_zonation", ""),
            "image": item.get("image"),
        })

    # 3. All logging detail sub-tabs
    logging_detail = intervals.get("logging_detail") or {}
    issue_type_labels = {
        it.get("key", ""): it.get("label_en", it.get("key", ""))
        for it in report.get("logging_detail_issue_types", [])
    }
    for key, issue_list in logging_detail.items():
        label = issue_type_labels.get(key, key)
        for item in issue_list:
            all_rows.append({
                "section": f"Detail: {label}",
                "hole_id": item.get("hole_id", ""),
                "depth_from": item.get("depth_from"),
                "depth_to": item.get("depth_to"),
                "issue": item.get("issue", ""),
                "significance": item.get("significance", "Low"),
                "strat": item.get("classified_as", item.get("strat", "")),
                "image": item.get("image"),
            })

    # 4. Grouping issues (flat list)
    for item in intervals.get("grouping_flat", []):
        all_rows.append({
            "section": "Grouping",
            "hole_id": item.get("hole_id", ""),
            "depth_from": item.get("depth_from"),
            "depth_to": item.get("depth_to"),
            "issue": item.get("issue", ""),
            "significance": "Low",
            "strat": item.get("strat", ""),
            "image": item.get("image"),
        })

    # 5. Outlier mismatches
    for item in intervals.get("outliers", []):
        all_rows.append({
            "section": "Outlier",
            "hole_id": item.get("hole_id", ""),
            "depth_from": item.get("depth_from"),
            "depth_to": item.get("depth_to"),
            "issue": f"Recorded: {item.get('recorded_as', '?')}, likely: {item.get('most_likely', '?')}",
            "significance": item.get("significance", "Low"),
            "strat": item.get("recorded_as", item.get("strat", "")),
            "image": item.get("image"),
        })

    if not all_rows:
        return """
            <section class="tab-panel" data-tab="all-issues">
                <div class="panel-header">
                    <h2 data-i18n-fr="Tous les problemes" data-i18n-en="All Issues">All Issues</h2>
                </div>
                <div class="empty" data-i18n-fr="Aucun probleme detecte." data-i18n-en="No issues detected.">No issues detected.</div>
            </section>
        """

    # Sort: significance High first, then by hole_id, then depth_from
    all_rows.sort(key=lambda r: (
        0 if r.get("significance") == "High" else 1,
        r.get("hole_id", ""),
        r.get("depth_from") or 0,
    ))

    total = len(all_rows)
    display_rows = all_rows[:ALL_ISSUES_ROW_LIMIT]
    limit_note = _truncation_note(total)

    # Group by hole_id for display
    holes_seen = {}
    for row in display_rows:
        hid = row.get("hole_id", "Unknown")
        if hid not in holes_seen:
            holes_seen[hid] = []
        holes_seen[hid].append(row)

    # Summary counts
    high_count = sum(1 for r in all_rows if r.get("significance") == "High")
    total_holes_affected = len({r.get("hole_id", "Unknown") for r in all_rows})
    section_counts = {}
    for r in all_rows:
        s = r.get("section", "Other")
        section_counts[s] = section_counts.get(s, 0) + 1

    summary_parts = [f"{s}: {c}" for s, c in sorted(section_counts.items(), key=lambda x: -x[1])]

    # Build table rows grouped by hole
    table_rows = []
    for hole_id, items in holes_seen.items():
        # Hole header row
        table_rows.append(
            f'<tr class="all-issues-hole-header">'
            f'<td colspan="7"><strong>{html.escape(hole_id)}</strong> '
            f'<span class="all-issues-hole-count">({len(items)} issue{"s" if len(items) != 1 else ""})</span></td>'
            f'</tr>'
        )
        for idx, item in enumerate(items):
            checkbox_id = f"{logger_id}::all-issues::{hole_id}::{idx}"
            depth = (
                f"{item.get('depth_from', '')} - {item.get('depth_to', '')}"
                if item.get("depth_from") is not None
                else f"{item.get('depth_to', '')}"
            )
            significance = item.get("significance", "Low")
            section = html.escape(item.get("section", ""))
            issue = html.escape(_safe_str(item.get("issue", "")))
            strat = html.escape(_safe_str(item.get("strat", "")))

            if item.get("image"):
                image_html = (
                    f'<img src="{item["image"]}" alt="" class="rotated-image expandable-img img-small" '
                    f'onclick="handleImageExpand(this)" />'
                )
            else:
                image_html = '<div class="image-placeholder-small">-</div>'

            table_rows.append(
                f'<tr class="compact-row">'
                f'<td><input type="checkbox" data-review-id="{checkbox_id}"></td>'
                f'<td class="nowrap">{html.escape(depth)}</td>'
                f'<td><span class="section-badge">{section}</span></td>'
                f'<td class="issue-cell">{issue}</td>'
                f'<td class="significance-{significance.lower()}">{html.escape(significance)}</td>'
                f'<td class="strat-cell">{strat}</td>'
                f'<td class="image-cell-compact">{image_html}</td>'
                f'</tr>'
            )

    table_html = (
        '<table class="interval-table evidence-table sortable-table compact-table" id="all-issues-table">'
        '<thead><tr>'
        '<th></th>'
        '<th class="sortable" data-i18n-fr="Profondeur" data-i18n-en="Depth">Depth</th>'
        '<th class="sortable" data-i18n-fr="Section" data-i18n-en="Section">Section</th>'
        '<th class="sortable" data-i18n-fr="Probleme" data-i18n-en="Issue">Issue</th>'
        '<th class="sortable" data-i18n-fr="Significance" data-i18n-en="Significance">Significance</th>'
        '<th data-i18n-fr="Strat" data-i18n-en="Strat">Strat</th>'
        '<th data-i18n-fr="Image" data-i18n-en="Image">Image</th>'
        '</tr></thead><tbody>'
        + "".join(table_rows)
        + '</tbody></table>'
    )

    return f"""
        <section class="tab-panel" data-tab="all-issues">
            <div class="panel-header">
                <h2 data-i18n-fr="Tous les problemes" data-i18n-en="All Issues">All Issues</h2>
            </div>
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label" data-i18n-fr="Total problemes" data-i18n-en="Total issues">Total issues</div>
                    <div class="kpi-value">{total}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label" data-i18n-fr="Haute significance" data-i18n-en="High significance">High significance</div>
                    <div class="kpi-value">{high_count}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label" data-i18n-fr="Forages affectes" data-i18n-en="Holes affected">Holes affected</div>
                    <div class="kpi-value">{total_holes_affected}</div>
                </div>
            </div>
            <div class="panel-card">
                <h3 data-i18n-fr="Repartition par section" data-i18n-en="Breakdown by section">Breakdown by section</h3>
                <p>{html.escape(', '.join(summary_parts))}</p>
            </div>
            <div class="intervals-section">
                <h3 data-i18n-fr="Intervalles a corriger (groupes par forage)" data-i18n-en="Intervals to fix (grouped by hole)">Intervals to fix (grouped by hole)</h3>
                {limit_note}
                {table_html}
            </div>
            <div class="notes-box">
                <label data-i18n-fr="Notes du reviseur" data-i18n-en="Reviewer notes">Notes du reviseur</label>
                <textarea data-note-id="{html.escape(logger_id)}::all-issues" placeholder="Ajouter des notes..."></textarea>
            </div>
        </section>
    """
