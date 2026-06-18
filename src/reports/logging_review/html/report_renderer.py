"""Render logging review report data to HTML."""
import html
import os
from typing import Optional

from .assets.scripts import JS_SCRIPTS
from .assets.styles import CSS_STYLES
from .images import _encode_image_base64
from .tabs.comments import render_comments_section
from .tabs.grouping import render_grouping_section
from .tabs.logging_detail import render_logging_detail_section
from .tabs.mineralisation import render_mineralisation_section
from .tabs.all_issues import render_all_issues_section
from .tabs.outliers import render_outlier_section
from .tabs.overview import render_overview_section
from .tabs.profile import render_profile_section
from .types import IntervalsForReview, ReportData


def render_html(
    report: ReportData,
    intervals: IntervalsForReview,
    logo_path: Optional[str],
    page_options: dict,
    charts_dir: Optional[str] = None,
    stream: Optional[object] = None,
) -> Optional[str]:
    """
    Render report data and intervals to full HTML document.
    If stream is provided, write HTML directly to stream and return None (avoids holding full HTML in memory).
    If stream is None, build and return the full HTML string.
    """
    from .comment_charts import render_comment_bar_chart, render_wordcloud

    logo_data = _encode_image_base64(logo_path) if logo_path else None
    logger_id = report["meta"]["logger"]

    cl = report.get("comment_stats_logging", {})
    total_rows_log = cl.get("total_rows", 0)
    rows_with_comment = cl.get("rows_with_comment", 0)
    avg_comment_length = cl.get("avg_comment_length", 0) if cl else 0
    comment_chart_save_path = os.path.join(charts_dir, "comment_statistics.png") if charts_dir else None
    comment_bar_chart_html = render_comment_bar_chart(total_rows_log, rows_with_comment, avg_comment_length, save_path=comment_chart_save_path)
    wordcloud_html = render_wordcloud(report["wordcloud"], save_path=os.path.join(charts_dir, "wordcloud.png") if charts_dir else None)

    tabs = [
        ("overview", "Vue d'ensemble", "Overview"),
        ("comments", "Commentaires", "Comments"),
        ("mineralisation", "Mineralisation", "Mineralisation"),
        ("profile", "Zonation", "Zonation"),
        ("logging-detail", "Precision du detail", "Logging detail accuracy"),
        ("grouping", "Regroupement", "Grouping"),
        ("outliers", "Anomalies", "Outliers"),
        ("all-issues", "Tous les problemes", "All Issues"),
    ]

    include_overview = page_options.get("summary_stats", True) or page_options.get("cover", True)
    include_comments = page_options.get("comment_stats", True)
    include_logging_detail = page_options.get("fines_accuracy", True)
    include_grouping = page_options.get("grouping_accuracy", True)
    include_outliers = page_options.get("outliers", True)

    tab_visibility = {
        "overview": include_overview,
        "comments": include_comments,
        "mineralisation": True,
        "profile": True,
        "logging-detail": include_logging_detail,
        "grouping": include_grouping,
        "outliers": include_outliers,
        "all-issues": True,
    }

    tab_buttons = []
    for tab_id, fr, en in tabs:
        if not tab_visibility.get(tab_id, True):
            continue
        tab_buttons.append(
            f"<button class=\"tab-button\" data-tab-button=\"{tab_id}\" "
            f"data-i18n-fr=\"{html.escape(fr)}\" data-i18n-en=\"{html.escape(en)}\">{html.escape(fr)}</button>"
        )
    tab_buttons_html = "".join(tab_buttons)

    logo_img = f'<img src="{logo_data}" alt="Logo">' if logo_data else "<div></div>"

    # Stream to file: write header then each section one-by-one to avoid holding full HTML in memory
    if stream is not None:
        def _w(s: str) -> None:
            stream.write(s)
        _w("""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Logging Review Report</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
""")
        _w(CSS_STYLES)
        _w("""
    </style>
</head>
<body>
    <div class="app">
        <aside class="sidebar">
            <div class="logo">
                """)
        _w(logo_img)
        _w("""
                <div class="credit-text">George Symonds 2025</div>
                <div class="lang-toggle">
                    <button data-lang="fr" class="active">FR</button>
                    <button data-lang="en">EN</button>
                </div>
            </div>
            """)
        _w(tab_buttons_html)
        _w("""
        </aside>
        <main class="content">
            """)
        if include_overview:
            _w(render_overview_section(report))
        if include_comments:
            _w(render_comments_section(report, logger_id, wordcloud_html, comment_bar_chart_html))
        _w(render_mineralisation_section(report, logger_id))
        _w(render_profile_section(report, logger_id))
        if include_logging_detail:
            _w(render_logging_detail_section(report, intervals, logger_id))
        if include_grouping:
            _w(render_grouping_section(report, intervals, logger_id))
        if include_outliers:
            _w(render_outlier_section(report, intervals, logger_id))
        _w(render_all_issues_section(report, intervals, logger_id))
        _w("""
        </main>
    </div>
    <script>
""")
        _w(JS_SCRIPTS)
        _w("""
    </script>
    <button id="print-report-btn" class="print-btn" title="Export to PDF">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M6 9V2h12v7M6 18H4a2 2 0 01-2-2v-5a2 2 0 012-2h16a2 2 0 012 2v5a2 2 0 01-2 2h-2"/>
            <rect x="6" y="14" width="12" height="8"/>
        </svg>
        Print / PDF
    </button>
</body>
</html>
""")
        return None

    # In-memory path: build sections then join
    overview_section = render_overview_section(report)
    comment_section = render_comments_section(report, logger_id, wordcloud_html, comment_bar_chart_html)
    mineral_section = render_mineralisation_section(report, logger_id)
    profile_section = render_profile_section(report, logger_id)
    logging_detail_section = render_logging_detail_section(report, intervals, logger_id)
    grouping_section = render_grouping_section(report, intervals, logger_id)
    outlier_section = render_outlier_section(report, intervals, logger_id)
    all_issues_section = render_all_issues_section(report, intervals, logger_id)
    main_parts = [
        (overview_section if include_overview else ""),
        (comment_section if include_comments else ""),
        mineral_section,
        profile_section,
        (logging_detail_section if include_logging_detail else ""),
        (grouping_section if include_grouping else ""),
        (outlier_section if include_outliers else ""),
        all_issues_section,
    ]
    main_content = "".join(main_parts)

    html_parts = [
        """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Logging Review Report</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
""",
        CSS_STYLES,
        """
    </style>
</head>
<body>
    <div class="app">
        <aside class="sidebar">
            <div class="logo">
                """,
        logo_img,
        """
                <div class="credit-text">George Symonds 2025</div>
                <div class="lang-toggle">
                    <button data-lang="fr" class="active">FR</button>
                    <button data-lang="en">EN</button>
                </div>
            </div>
            """,
        tab_buttons_html,
        """
        </aside>
        <main class="content">
            """,
        main_content,
        """
        </main>
    </div>
    <script>
""",
        JS_SCRIPTS,
        """
    </script>
    <button id="print-report-btn" class="print-btn" title="Export to PDF">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M6 9V2h12v7M6 18H4a2 2 0 01-2-2v-5a2 2 0 012-2h16a2 2 0 012 2v5a2 2 0 01-2 2h-2"/>
            <rect x="6" y="14" width="12" height="8"/>
        </svg>
        Print / PDF
    </button>
</body>
</html>
""",
    ]
    return "".join(html_parts)
