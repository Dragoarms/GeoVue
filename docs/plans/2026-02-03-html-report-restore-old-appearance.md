# Plan: Restore Old Appearance and Functionality to Logging Review HTML Report

**Date:** 2026-02-03  
**Status:** Draft  
**Reference:** OLD = `reports/ouputs/RC_Logging_Review_AAB.html`, NEW = `reports/New folder/RC_Logging_Review_AAB.html`

## Problem Summary

After the migration of the logging review report into `src/reports/logging_review/`, the generated HTML report no longer matches the old appearance and behaviour. Inspection shows:

1. **Template not rendered (critical):** The NEW file contains **literal Python expression text** in the body (e.g. `{"<img src="" + logo_data + "" alt="Logo">" if logo_data else "<div></div>"}`, `{''.join(tab_buttons)}`, `{overview_section if include_overview else ""}`) instead of the actual HTML. So the sidebar has no tab buttons and the main area has no tab panels—the report is broken.

2. **Root cause:** In `src/reports/logging_review/html/report_renderer.py`, the HTML is built with an f-string that is **split by concatenation**. The first segment is an f-string (lines 100–109 up to `</style>`). The next segment is a **plain** string (`"""\n    </style>\n...\n"""`), so all `{...}` in the body and script block are output literally and never evaluated.

3. **Possible further drift:** Even after fixing rendering, CSS/JS or per-tab content might have diverged from the OLD report during the migration (e.g. different rules in `assets/styles.py`, different script behaviour, or different section structure). These need to be compared and aligned to the OLD report.

## Goals

- Restore correct **rendering** so the generated HTML contains real content (sidebar buttons, tab panels, logo, etc.).
- Restore **appearance** (CSS) and **behaviour** (JS) to match the OLD report.
- Restore **tab content and structure** (Overview, Comments, Mineralisation, Zonation, Logging detail, Grouping, Outliers) so each tab looks and behaves as in OLD.
- Add a **regression guard** so un-rendered template placeholders never reappear in output.

## Tasks

### 1. Fix template rendering in `report_renderer.py` (critical)

**Problem:** The body and script sections are inside a non–f-string literal, so `{''.join(tab_buttons)}`, `{overview_section}`, etc. are written literally.

**Required:**

- Use a **single** f-string for the entire HTML document so every `{...}` is evaluated.
- Interpolate `CSS_STYLES` and `JS_SCRIPTS` as variables inside that f-string (e.g. `{CSS_STYLES}` and `{JS_SCRIPTS}`) instead of concatenating with `+ CSS_STYLES +` and `+ JS_SCRIPTS +`.
- Keep all section variables (`overview_section`, `comment_section`, `mineral_section`, `profile_section`, `logging_detail_section`, `grouping_section`, `outlier_section`) and visibility flags in the same f-string.
- For the logo fragment, use a variable computed before the f-string (e.g. `logo_img = f'<img src="{logo_data}" alt="Logo">' if logo_data else '<div></div>'`) and then `{logo_img}` in the template to avoid nested quote/brace issues.

**Verification:** Generate a report; open the HTML. Sidebar must show tab buttons; main content must show `<section class="tab-panel" data-tab="overview">`, etc., with real content. No literal `{overview_section` or `{''.join(tab_buttons)}` in the file.

---

### 2. Regression test: no raw placeholders in generated HTML

**Required:**

- Add a test (e.g. in `src/processing/tests/test_logging_review_report.py` or a dedicated test under `src/reports/`) that:
  - Builds minimal report data and calls the code path that produces the full HTML string (e.g. `render_html(...)` or the public entry that returns HTML).
  - Asserts that the returned HTML string does **not** contain literal substrings such as `{overview_section}`, `{''.join(tab_buttons)}`, `{comment_section}`, `{mineral_section}`, `{profile_section}`, `{logging_detail_section}`, `{grouping_section}`, `{outlier_section}` (and optionally no unescaped `if include_overview`).
- This prevents reintroducing the same bug if the template is refactored again.

**Verification:** Run the new test; it must pass after task 1 and fail if the template is changed back to a non–f-string body.

---

### 3. Align CSS with OLD report

**Required:**

- Diff the **inline `<style>` block** in OLD (`ouputs/RC_Logging_Review_AAB.html`) against the CSS emitted by the current code (from `src/reports/logging_review/html/assets/styles.py` / `CSS_STYLES`).
- List any rules (or parts of rules) that exist in OLD but are missing or different in the new CSS.
- Restore missing rules and fix differences so that the generated report’s styles match OLD (e.g. `.credit-text`, `.print-btn`, `.expandable-img.expanded` min-width/min-height, `.validation-*`, `.geochem-cell`, `.fines-image-cell`, media queries, print styles).
- If OLD does **not** include `.logging-detail-subnav` / `.logging-detail-subtab` / `.logging-detail-subpanel` but the current code does, decide whether to keep them (if Logging detail subtabs are desired) or remove them to match OLD; document the choice.

**Verification:** Generate report; compare layout, typography, colours, and responsive/print behaviour with OLD in a browser.

---

### 4. Align JavaScript with OLD report

**Required:**

- Diff the **inline `<script>` block** in OLD against the JS emitted by the current code (`src/reports/logging_review/html/assets/scripts.py` / `JS_SCRIPTS`).
- Restore any behaviour from OLD that is missing or changed (e.g. tab switching, language toggle, Plotly init/resize, print button, expandable images, Leaflet map, logging-detail subtabs if present in OLD).
- Ensure event handlers and selectors match the HTML structure (e.g. `.tab-button`, `.tab-panel`, `data-tab`, `data-tab-button`).

**Verification:** In the generated report, test: tab navigation, FR/EN toggle, print/PDF, expandable images, map interaction, and any Plotly charts; behaviour should match OLD.

---

### 5. Align tab section structure and content with OLD

**Required:**

- For each tab (Overview, Comments, Mineralisation, Zonation, Logging detail, Grouping, Outliers), compare OLD’s `<section class="tab-panel" data-tab="...">` content with the output of the corresponding render function in `src/reports/logging_review/html/tabs/`.
- Ensure:
  - Each section has the same top-level structure (e.g. hero, KPI grid, two-panel layout, tables, charts).
  - Class names and data attributes match so CSS and JS still apply.
  - Logging detail: if OLD uses a single panel vs subtabs (e.g. `.logging-detail-subnav`), match that structure; if OLD has subtabs, ensure the new code emits the same subnav and subpanels.
- Fix any tab that produces different structure or missing blocks so it matches OLD.

**Verification:** Side-by-side comparison of OLD and NEW (after fix) HTML for each tab; visual comparison in browser.

---

### 6. End-to-end verification

**Required:**

- Generate a report from the app (same dataset/options as used for OLD if possible).
- Open OLD and NEW in the same browser; compare:
  - Sidebar: logo, credit, FR/EN, tab list and active state.
  - Overview: hero, KPIs, map, chart.
  - Comments: wordcloud, bar chart, attribution.
  - Mineralisation: tables/charts as in OLD.
  - Zonation: rules, coverage, tables.
  - Logging detail: layout and subnav (if any), tables, images.
  - Grouping: groups, CV indicators, tables.
  - Outliers: table, modal (if present), flags.
- Confirm print/PDF and expandable images behave as in OLD.

**Verification:** Sign-off that appearance and functionality match OLD.

---

## Order of execution

1. **Task 1** (fix template rendering) first—without it the report is unusable.
2. **Task 2** (regression test) immediately after task 1.
3. **Tasks 3–5** can be done in parallel or in any order once rendering is correct.
4. **Task 6** after 3–5 are done.

## Out of scope

- Changing the data pipeline or report data shape unless required to restore OLD content.
- Adding new features; this plan is limited to restoring OLD appearance and behaviour.

## References

- OLD report: `c:\Users\georg\Downloads\reports\ouputs\RC_Logging_Review_AAB.html`
- NEW report (current output): `c:\Users\georg\Downloads\reports\New folder\RC_Logging_Review_AAB.html`
- Renderer: `src/reports/logging_review/html/report_renderer.py`
- CSS: `src/reports/logging_review/html/assets/styles.py`
- JS: `src/reports/logging_review/html/assets/scripts.py`
- Tab renderers: `src/reports/logging_review/html/tabs/*.py`
