"""Table renderers for logging review HTML report."""
import html
import json
from typing import Any, Dict, List, Tuple

import numpy as np

from .utils import _safe_str


def _render_intervals_table(intervals: List[Dict[str, Any]], logger_id: str, tab_id: str) -> str:
    if not intervals:
        return (
            "<div class=\"empty\" data-i18n-fr=\"Aucun intervalle signale pour revue.\" "
            "data-i18n-en=\"No intervals flagged for review.\">"
            "Aucun intervalle signale pour revue.</div>"
        )
    rows = []
    for idx, item in enumerate(intervals):
        checkbox_id = f"{logger_id}::{tab_id}::{idx}"
        image_html = (
            f"<img src=\"{item['image']}\" alt=\"Interval image\" class=\"rotated-image\" />"
            if item.get("image")
            else "<div class=\"image-placeholder\" data-i18n-fr=\"Aucune image\" data-i18n-en=\"No image\">Aucune image</div>"
        )
        depth = (
            f"{item.get('depth_from', '')} - {item.get('depth_to', '')}"
            if item.get("depth_from") is not None
            else f"{item.get('depth_to', '')}"
        )
        rows.append(
            "<tr>"
            f"<td><input type=\"checkbox\" data-review-id=\"{checkbox_id}\"></td>"
            f"<td>{html.escape(_safe_str(item.get('hole_id')))}</td>"
            f"<td>{html.escape(str(depth))}</td>"
            f"<td>{html.escape(_safe_str(item.get('issue')))}</td>"
            f"<td class=\"image-cell rotated-image-cell\">{image_html}</td>"
            "</tr>"
        )
    return (
        "<table class=\"interval-table\">"
        "<thead><tr>"
        "<th></th>"
        "<th data-i18n-fr=\"Trou\" data-i18n-en=\"Hole\">Trou</th>"
        "<th data-i18n-fr=\"Profondeur\" data-i18n-en=\"Depth\">Profondeur</th>"
        "<th data-i18n-fr=\"Probleme probable\" data-i18n-en=\"Likely issue\">Probleme probable</th>"
        "<th data-i18n-fr=\"Image\" data-i18n-en=\"Image\">Image</th>"
        "</tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_mineralisation_evidence_table(intervals: List[Dict[str, Any]], logger_id: str) -> str:
    """Render mineralisation evidence table with sortable columns and conditional formatting."""
    if not intervals:
        return (
            "<div class=\"empty\" data-i18n-fr=\"Aucun intervalle de mineralisation.\" "
            "data-i18n-en=\"No mineralisation intervals.\">"
            "Aucun intervalle de mineralisation.</div>"
        )

    def format_depth(val):
        """Format depth as integer if whole number, otherwise as float."""
        if val is None:
            return ""
        if isinstance(val, float) and val == int(val):
            return str(int(val))
        return str(val)

    def get_assay_class(assay_suggests):
        """Get CSS class for conditional formatting based on assay suggestion."""
        if assay_suggests == "Mineralised":
            return "assay-mineralised"
        elif assay_suggests == "Leached":
            return "assay-leached"
        else:
            return "assay-unmineralised"

    def get_validation_class(validation):
        """Get CSS class for validation status."""
        if validation == "Match":
            return "validation-match"
        elif validation == "Mismatch":
            return "validation-mismatch"
        else:
            return "validation-pending"

    rows = []
    for idx, item in enumerate(intervals):
        checkbox_id = f"{logger_id}::mineral_evidence::{idx}"

        # Format depths as integers if whole numbers
        depth_from = format_depth(item.get('depth_from'))
        depth_to = format_depth(item.get('depth_to'))
        depth = f"{depth_from} - {depth_to}" if depth_from else depth_to

        # Build geochem display with conditional formatting
        geochem = item.get("geochem", {})
        fe_val = geochem.get('Fe')
        sio2_val = geochem.get('SiO2')
        al2o3_val = geochem.get('Al2O3')

        fe_display = f"{fe_val:.1f}" if fe_val is not None else "-"
        sio2_display = f"{sio2_val:.1f}" if sio2_val is not None else "-"
        al2o3_display = f"{al2o3_val:.1f}" if al2o3_val is not None else "-"

        assay_suggests = item.get("assay_suggests", "")
        assay_class = get_assay_class(assay_suggests)
        validation = item.get("validation", "")
        validation_class = get_validation_class(validation)
        significance = item.get("significance", "High")
        logged_as = item.get("logged_as", "")
        logged_zonation = item.get("logged_zonation", "")
        if item.get("image"):
            image_html = (
                f"<img src=\"{item['image']}\" alt=\"Interval\" class=\"rotated-image expandable-img img-small\" "
                f"onclick=\"handleImageExpand(this)\" title=\"Click to expand\" />"
            )
        else:
            image_html = "<div class=\"image-placeholder-small\">-</div>"

        rows.append(
            "<tr>"
            f"<td><input type=\"checkbox\" data-review-id=\"{checkbox_id}\"></td>"
            f"<td data-sort=\"{html.escape(_safe_str(item.get('hole_id')))}\">{html.escape(_safe_str(item.get('hole_id')))}</td>"
            f"<td data-sort=\"{item.get('depth_from', 0) or 0}\">{html.escape(depth)}</td>"
            f"<td class=\"{validation_class}\" data-sort=\"{validation}\">{html.escape(validation)}</td>"
            f"<td class=\"significance-{significance.lower()}\" data-sort=\"{significance}\">{html.escape(significance)}</td>"
            f"<td data-sort=\"{logged_as}\">{html.escape(logged_as)}</td>"
            f"<td data-sort=\"{logged_zonation}\">{html.escape(logged_zonation)}</td>"
            f"<td class=\"{assay_class}\" data-sort=\"{assay_suggests}\">{html.escape(assay_suggests)}</td>"
            f"<td class=\"geochem-cell {assay_class}\" data-sort=\"{fe_val or 0}\">{fe_display}</td>"
            f"<td class=\"geochem-cell {assay_class}\" data-sort=\"{sio2_val or 0}\">{sio2_display}</td>"
            f"<td class=\"geochem-cell {assay_class}\" data-sort=\"{al2o3_val or 0}\">{al2o3_display}</td>"
            f"<td class=\"image-cell-compact\">{image_html}</td>"
            "</tr>"
        )
    return (
        "<table class=\"interval-table evidence-table sortable-table\" id=\"mineral-evidence-table\">"
        "<thead><tr>"
        "<th></th>"
        "<th class=\"sortable\" data-i18n-fr=\"Trou\" data-i18n-en=\"Hole\">Hole ▼</th>"
        "<th class=\"sortable\" data-i18n-fr=\"Profondeur\" data-i18n-en=\"Depth\">Depth ▼</th>"
        "<th class=\"sortable\" data-i18n-fr=\"Validation\" data-i18n-en=\"Validation\">Validation ▼</th>"
        "<th class=\"sortable\" data-i18n-fr=\"Significance\" data-i18n-en=\"Significance\">Significance ▼</th>"
        "<th class=\"sortable\" data-i18n-fr=\"Logue comme\" data-i18n-en=\"Logged as\">Logged as ▼</th>"
        "<th class=\"sortable\" data-i18n-fr=\"Zonation loggee\" data-i18n-en=\"Logged Zonation\">Logged Zonation ▼</th>"
        "<th class=\"sortable\" data-i18n-fr=\"Essai suggere\" data-i18n-en=\"Assay suggests\">Assay suggests ▼</th>"
        "<th class=\"sortable\">Fe% ▼</th>"
        "<th class=\"sortable\">SiO2% ▼</th>"
        "<th class=\"sortable\">Al2O3% ▼</th>"
        "<th data-i18n-fr=\"Image\" data-i18n-en=\"Image\">Image</th>"
        "</tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_zonation_evidence_table(
    intervals: List[Dict[str, Any]], logger_id: str
) -> str:
    """Evidence table for zonation mismatches: Hole, Depth, Logged zonation, Should be, Total gangue %, De/Hy/Pr %, Image."""
    if not intervals:
        return (
            "<div class=\"empty\" data-i18n-fr=\"Aucune discordance de zonation.\" "
            "data-i18n-en=\"No zonation mismatches.\">No zonation mismatches.</div>"
        )
    rows = []
    for idx, item in enumerate(intervals):
        checkbox_id = f"{logger_id}::zonation_evidence::{idx}"
        depth = (
            f"{item.get('depth_from', '')} - {item.get('depth_to', '')}"
            if item.get("depth_from") is not None
            else f"{item.get('depth_to', '')}"
        )
        total_g = item.get("total_gangue_pct")
        total_g_display = f"{total_g:.1f}%" if total_g is not None else "-"
        de_display = f"{item.get('de_pct', 0) or 0:.1f}%" if item.get("de_pct") is not None else "-"
        hy_display = f"{item.get('hy_pct', 0) or 0:.1f}%" if item.get("hy_pct") is not None else "-"
        pr_display = f"{item.get('pr_pct', 0) or 0:.1f}%" if item.get("pr_pct") is not None else "-"
        if item.get("image"):
            image_html = (
                f"<img src=\"{item['image']}\" alt=\"Interval\" class=\"rotated-image expandable-img img-small\" "
                f"onclick=\"handleImageExpand(this)\" title=\"Click to expand\" />"
            )
        else:
            image_html = "<div class=\"image-placeholder-small\">-</div>"
        rows.append(
            "<tr>"
            f"<td><input type=\"checkbox\" data-review-id=\"{checkbox_id}\"></td>"
            f"<td class=\"sortable\" data-sort=\"{html.escape(_safe_str(item.get('hole_id')))}\">{html.escape(_safe_str(item.get('hole_id')))}</td>"
            f"<td class=\"sortable\" data-sort=\"{item.get('depth_from', 0) or 0}\">{html.escape(str(depth))}</td>"
            f"<td data-sort=\"{html.escape(_safe_str(item.get('logged_zonation')))}\">{html.escape(_safe_str(item.get('logged_zonation')))}</td>"
            f"<td data-sort=\"{html.escape(_safe_str(item.get('should_be')))}\">{html.escape(_safe_str(item.get('should_be')))}</td>"
            f"<td class=\"sortable\" data-sort=\"{total_g or 0}\">{total_g_display}</td>"
            f"<td class=\"sortable\" data-sort=\"{item.get('de_pct') or 0}\">{de_display}</td>"
            f"<td class=\"sortable\" data-sort=\"{item.get('hy_pct') or 0}\">{hy_display}</td>"
            f"<td class=\"sortable\" data-sort=\"{item.get('pr_pct') or 0}\">{pr_display}</td>"
            f"<td class=\"image-cell-compact\">{image_html}</td>"
            "</tr>"
        )
    return (
        "<table class=\"interval-table evidence-table sortable-table\" id=\"zonation-evidence-table\">"
        "<thead><tr>"
        "<th></th>"
        "<th class=\"sortable\" data-i18n-fr=\"Trou\" data-i18n-en=\"Hole\">Hole</th>"
        "<th class=\"sortable\" data-i18n-fr=\"Profondeur\" data-i18n-en=\"Depth\">Depth</th>"
        "<th class=\"sortable\" data-i18n-fr=\"Zonation loggee\" data-i18n-en=\"Logged zonation\">Logged zonation</th>"
        "<th class=\"sortable\" data-i18n-fr=\"Devrait etre\" data-i18n-en=\"Should be\">Should be</th>"
        "<th class=\"sortable\" data-i18n-fr=\"Gangue totale %\" data-i18n-en=\"Total gangue %\">Total gangue %</th>"
        "<th class=\"sortable\">De %</th>"
        "<th class=\"sortable\">Hy %</th>"
        "<th class=\"sortable\">Pr %</th>"
        "<th data-i18n-fr=\"Image\" data-i18n-en=\"Image\">Image</th>"
        "</tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_logging_detail_evidence_table(
    intervals: List[Dict[str, Any]], logger_id: str, table_id: str
) -> str:
    """One evidence table per issue type: Hole, Depth, Issue, Fe%, SiO2%, Al2O3%, Classified as, Image (small placeholder)."""
    if not intervals:
        return (
            "<div class=\"empty\" data-i18n-fr=\"Aucun intervalle signale pour revue.\" "
            "data-i18n-en=\"No intervals flagged for review.\">"
            "No intervals flagged for review.</div>"
        )
    rows = []
    for idx, item in enumerate(intervals):
        checkbox_id = f"{logger_id}::logging_detail::{table_id}::{idx}"
        if item.get("image"):
            image_html = (
                f"<img src=\"{item['image']}\" alt=\"Interval\" class=\"rotated-image expandable-img img-small\" "
                f"onclick=\"handleImageExpand(this)\" title=\"Click to expand\" />"
            )
        else:
            image_html = "<div class=\"image-placeholder-small\">-</div>"
        depth = (
            f"{item.get('depth_from', '')} - {item.get('depth_to', '')}"
            if item.get("depth_from") is not None
            else f"{item.get('depth_to', '')}"
        )
        classified_as = html.escape(_safe_str(item.get("classified_as", item.get("strat", "-"))))
        significance = item.get("significance", "High")
        geochem = item.get("geochem", {})
        fe_val = geochem.get("Fe")
        sio2_val = geochem.get("SiO2")
        al2o3_val = geochem.get("Al2O3")
        fe_display = f"{fe_val:.1f}" if fe_val is not None else "-"
        sio2_display = f"{sio2_val:.1f}" if sio2_val is not None else "-"
        al2o3_display = f"{al2o3_val:.1f}" if al2o3_val is not None else "-"

        rows.append(
            "<tr class=\"compact-row\">"
            f"<td><input type=\"checkbox\" data-review-id=\"{checkbox_id}\"></td>"
            f"<td class=\"sortable\" data-sort=\"{html.escape(_safe_str(item.get('hole_id')))}\">{html.escape(_safe_str(item.get('hole_id')))}</td>"
            f"<td class=\"sortable\" data-sort=\"{item.get('depth_from', 0) or 0}\">{html.escape(str(depth))}</td>"
            f"<td class=\"issue-cell\">{html.escape(_safe_str(item.get('issue')))}</td>"
            f"<td class=\"significance-{significance.lower()}\" data-sort=\"{significance}\">{html.escape(significance)}</td>"
            f"<td class=\"geochem-cell\" data-sort=\"{fe_val or 0}\">{fe_display}</td>"
            f"<td class=\"geochem-cell\" data-sort=\"{sio2_val or 0}\">{sio2_display}</td>"
            f"<td class=\"geochem-cell\" data-sort=\"{al2o3_val or 0}\">{al2o3_display}</td>"
            f"<td class=\"classification-cell\">{classified_as}</td>"
            f"<td class=\"image-cell-compact\">{image_html}</td>"
            "</tr>"
        )
    return (
        "<table class=\"interval-table evidence-table sortable-table compact-table\" "
        f"id=\"logging-detail-table-{html.escape(table_id)}\">"
        "<thead><tr>"
        "<th></th>"
        "<th class=\"sortable\" data-i18n-fr=\"Trou\" data-i18n-en=\"Hole\">Hole</th>"
        "<th class=\"sortable\" data-i18n-fr=\"Profondeur\" data-i18n-en=\"Depth\">Depth</th>"
        "<th class=\"sortable\" data-i18n-fr=\"Probleme\" data-i18n-en=\"Issue\">Issue</th>"
        "<th class=\"sortable\" data-i18n-fr=\"Significance\" data-i18n-en=\"Significance\">Significance</th>"
        "<th class=\"sortable\">Fe%</th>"
        "<th class=\"sortable\">SiO2%</th>"
        "<th class=\"sortable\">Al2O3%</th>"
        "<th data-i18n-fr=\"Classe comme\" data-i18n-en=\"Classified as\">Classified as</th>"
        "<th data-i18n-fr=\"Image\" data-i18n-en=\"Image\">Image</th>"
        "</tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_fines_intervals_table(intervals: List[Dict[str, Any]], logger_id: str, tab_id: str) -> str:
    """Fines: delegate to logging detail evidence table (small placeholder, geochem columns)."""
    return _render_logging_detail_evidence_table(intervals, logger_id, tab_id)


def _parse_outlier_reason_to_flags(reason: str) -> List[Tuple[str, str]]:
    """Parse outlier_reason text into list of (element_short_name, direction). Direction is 'up' or 'down'."""
    if not reason or not isinstance(reason, str):
        return []
    flags = []
    for part in reason.split(";"):
        part = part.strip()
        if " low " in part or " low(" in part:
            direction = "down"
        elif " high " in part or " high(" in part:
            direction = "up"
        else:
            continue
        col_name = part.split()[0] if part else ""
        short = col_name.replace("_pct", "").replace("_", "")
        if short.upper() == "FE":
            short = "Fe"
        elif short.upper() == "SIO2":
            short = "SiO2"
        elif short.upper() == "AL2O3":
            short = "Al2O3"
        elif short.upper() == "P":
            short = "P"
        else:
            short = col_name[:6] if len(col_name) > 6 else col_name
        if short:
            flags.append((short, direction))
    return flags


def _outlier_significance_from_reason(reason: str, elements_str: str) -> str:
    """High if major elements (Fe, Al, Si) in reason/elements; Low if only minor (e.g. P)."""
    text = (reason or "") + " " + (elements_str or "")
    text_lower = text.lower()
    major = any(x in text_lower for x in ("fe_pct", "fe ", "sio2", "al2o3", "al "))
    minor_only = "p_pct" in text_lower or " p " in text_lower
    if major:
        return "High"
    if minor_only and not major:
        return "Low"
    return "High"


def _render_outlier_table(intervals: List[Dict[str, Any]], logger_id: str) -> str:
    """Outlier evidence table: Hole, Depth, Recorded, Likely, Flags (pills), Significance, Fe/SiO2/Al2O3/P, Actions (Details, Image)."""
    if not intervals:
        return (
            "<div class=\"empty\" data-i18n-fr=\"Aucun intervalle signale.\" "
            "data-i18n-en=\"No intervals flagged.\">Aucun intervalle signale.</div>"
        )
    max_flags_visible = 5
    rows = []
    for idx, item in enumerate(intervals):
        checkbox_id = f"{logger_id}::outliers::{idx}"
        image_html = (
            f"<img src=\"{item['image']}\" alt=\"\" class=\"rotated-image expandable-img img-small\" onclick=\"handleImageExpand(this)\" />"
            if item.get("image")
            else "<div class=\"image-placeholder-small\">-</div>"
        )
        depth = f"{item.get('depth_from', '')} - {item.get('depth_to', '')}" if item.get("depth_from") is not None else str(item.get("depth_to", ""))

        flags_list = item.get("flags", []) or _parse_outlier_reason_to_flags(item.get("reason", ""))
        arrow = "\u2191"  # up
        arrow_down = "\u2193"  # down
        flag_pills = []
        for (short, direction) in flags_list[:max_flags_visible]:
            sym = arrow if direction == "up" else arrow_down
            flag_pills.append(f'<span class="flag-pill">{html.escape(short)}{sym}</span>')
        extra = len(flags_list) - max_flags_visible
        if extra > 0:
            flag_pills.append(f'<span class="flag-pill-extra">+{extra}</span>')
        flags_html = " ".join(flag_pills) if flag_pills else "<span class=\"geochem-na\">-</span>"

        significance = item.get("significance", "High")
        geochem = item.get("geochem", {})
        fe_val = geochem.get("Fe")
        sio2_val = geochem.get("SiO2")
        al2o3_val = geochem.get("Al2O3")
        p_val = geochem.get("P")
        fe_display = f"{fe_val:.1f}" if fe_val is not None else "-"
        sio2_display = f"{sio2_val:.1f}" if sio2_val is not None else "-"
        al2o3_display = f"{al2o3_val:.1f}" if al2o3_val is not None else "-"
        p_display = f"{p_val:.1f}" if p_val is not None else "-"

        details_json = json.dumps({
            "hole_id": item.get("hole_id"),
            "depth": depth,
            "recorded_as": item.get("recorded_as"),
            "most_likely": item.get("most_likely"),
            "reason": item.get("reason"),
            "geochem": geochem,
        })
        details_payload = html.escape(details_json, quote=True)

        rows.append(
            "<tr data-outlier-idx=\"" + str(idx) + "\">"
            f"<td><input type=\"checkbox\" data-review-id=\"{checkbox_id}\"></td>"
            f"<td class=\"sortable\" data-sort=\"{html.escape(_safe_str(item.get('hole_id')))}\">{html.escape(_safe_str(item.get('hole_id')))}</td>"
            f"<td class=\"sortable\" data-sort=\"{item.get('depth_from') or 0}\">{html.escape(depth)}</td>"
            f"<td>{html.escape(_safe_str(item.get('recorded_as', item.get('strat', ''))))}</td>"
            f"<td class=\"most-likely-cell\">{html.escape(_safe_str(item.get('most_likely', '-')))}</td>"
            f"<td class=\"flags-cell\">{flags_html}</td>"
            f"<td class=\"significance-{significance.lower()}\" data-sort=\"{significance}\">{html.escape(significance)}</td>"
            f"<td class=\"geochem-cell\" data-sort=\"{fe_val or 0}\">{fe_display}</td>"
            f"<td class=\"geochem-cell\" data-sort=\"{sio2_val or 0}\">{sio2_display}</td>"
            f"<td class=\"geochem-cell\" data-sort=\"{al2o3_val or 0}\">{al2o3_display}</td>"
            f"<td class=\"geochem-cell\" data-sort=\"{p_val or 0}\">{p_display}</td>"
            f"<td class=\"actions-cell\"><a href=\"#\" class=\"outlier-details-link\" data-details=\"{details_payload}\" data-i18n-en=\"Details\">Details</a> | <a href=\"#\" class=\"outlier-image-link\" data-outlier-idx=\"{idx}\">Image</a></td>"
            f"<td class=\"image-cell-compact\">{image_html}</td>"
            "</tr>"
        )
    return (
        "<table class=\"interval-table evidence-table sortable-table outlier-table\" id=\"outlier-evidence-table\">"
        "<thead><tr>"
        "<th></th>"
        "<th class=\"sortable\" data-i18n-fr=\"Trou\" data-i18n-en=\"Hole\">Hole</th>"
        "<th class=\"sortable\" data-i18n-fr=\"Profondeur\" data-i18n-en=\"Depth\">Depth</th>"
        "<th data-i18n-fr=\"Enregistre\" data-i18n-en=\"Recorded\">Recorded</th>"
        "<th data-i18n-fr=\"Plus probable\" data-i18n-en=\"Likely\">Likely</th>"
        "<th data-i18n-fr=\"Drapeaux\" data-i18n-en=\"Flags\">Flags</th>"
        "<th class=\"sortable\" data-i18n-fr=\"Significance\" data-i18n-en=\"Significance\">Significance</th>"
        "<th class=\"sortable\">Fe%</th><th class=\"sortable\">SiO2%</th><th class=\"sortable\">Al2O3%</th><th class=\"sortable\">P%</th>"
        "<th data-i18n-fr=\"Actions\" data-i18n-en=\"Actions\">Actions</th>"
        "<th data-i18n-fr=\"Image\" data-i18n-en=\"Image\">Image</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_grouping_groups(groups: List[Dict[str, Any]], logger_id: str) -> str:
    """Render groups for review with visual CV indicators, significance, and evidence-table layout."""
    if not groups:
        return (
            "<div class=\"empty\" data-i18n-fr=\"Aucun groupe a revoir.\" "
            "data-i18n-en=\"No groups for review.\">Aucun groupe a revoir.</div>"
            "<p class=\"grouping-empty-note\" data-i18n-en=\"No groups had CV > 100% for Fe, SiO2, Al2O3, MgO, CaO, S.\" "
            "data-i18n-fr=\"Aucun groupe n'a un CV > 100% pour Fe, SiO2, Al2O3, MgO, CaO, S.\">"
            "No groups had CV &gt; 100%.</p>"
        )
    out = []
    for gidx, grp in enumerate(groups):
        cv_max = grp.get('cv_max', 0)
        strat = html.escape(_safe_str(grp.get('strat', '')))
        significance = grp.get("significance", "High")
        num_intervals = len(grp.get("intervals", []))

        # CV severity indicator: green <30%, yellow 30-50%, red >50%
        if cv_max < 30:
            cv_class = "cv-low"
            cv_color = "#22c55e"
        elif cv_max < 50:
            cv_class = "cv-medium"
            cv_color = "#f59e0b"
        else:
            cv_class = "cv-high"
            cv_color = "#ef4444"

        # Visual group header with CV bar
        cv_bar_width = min(cv_max, 100)
        out.append(f'''<div class="grouping-group {cv_class}">
            <div class="group-header">
                <div class="group-info">
                    <span class="group-strat">{strat if strat else "Groupe " + str(gidx+1)}</span>
                    <span class="group-count">{num_intervals} intervalle(s)</span>
                    <span class="significance-{significance.lower()}" data-sort="{significance}">{html.escape(significance)}</span>
                </div>
                <div class="cv-indicator">
                    <span class="cv-label">CV max:</span>
                    <div class="cv-bar-container">
                        <div class="cv-bar" style="width:{cv_bar_width}%;background:{cv_color}"></div>
                    </div>
                    <span class="cv-value" style="color:{cv_color}">{cv_max:.0f}%</span>
                </div>
            </div>''')

        # Evidence-table style: Hole, Depth, Strat, Geochem, Significance, Image
        out.append('<table class="interval-table evidence-table sortable-table compact-table grouping-table"><thead><tr>'
                   '<th data-i18n-en="Hole">Trou</th><th data-i18n-en="Depth">Prof.</th><th data-i18n-en="Strat">Strat</th>'
                   '<th data-i18n-en="Geochem">Geochimie</th><th data-i18n-en="Significance">Significance</th><th data-i18n-en="Image">Image</th>'
                   '</tr></thead><tbody>')
        for it in grp.get("intervals", []):
            depth = f"{it.get('depth_from', '')}-{it.get('depth_to', '')}" if it.get("depth_from") is not None else str(it.get("depth_to", ""))
            interval_strat = html.escape(_safe_str(it.get('strat', '')))
            row_sig = grp.get("significance", "High")

            # Inline geochem display
            geochem = it.get("geochem", {})
            geochem_parts = []
            for k, v in geochem.items():
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    geochem_parts.append(f"<span class='geochem-inline'><b>{k}:</b>{v:.1f}</span>")
            geochem_html = " ".join(geochem_parts) if geochem_parts else "-"

            # Expandable image
            if it.get("image"):
                img = f'<img src="{it["image"]}" alt="" class="rotated-image expandable-img" onclick="handleImageExpand(this)" />'
            else:
                img = '<div class="image-placeholder">-</div>'

            out.append(f'<tr class="compact-row">'
                       f'<td class="nowrap">{html.escape(_safe_str(it.get("hole_id")))}</td>'
                       f'<td class="nowrap">{html.escape(depth)}</td>'
                       f'<td class="strat-cell">{interval_strat}</td>'
                       f'<td class="geochem-inline-cell">{geochem_html}</td>'
                       f'<td class="significance-{row_sig.lower()}" data-sort="{row_sig}">{html.escape(row_sig)}</td>'
                       f'<td class="image-cell-compact">{img}</td>'
                       '</tr>')
        out.append("</tbody></table></div>")
    return "".join(out)


