# Logging Review Report Refactor — Atomic Tasks (Complete)

Each task has an **exact before/after** (or a **mechanical step** with line numbers), a **grep command to verify completion**, and requires **zero judgment**. Do **one file at a time** in the order listed.

**Paths:** All paths are relative to repo root `efz` (e.g. `src/processing/...`). Use your workspace root if different.

---

## Phase A.5 — Deduplicate with DataManager

### File 1: `src/processing/DataManager/column_aliases.py`

#### Task A.5.1 — Add `project_code`, `easting`, `northing` to COLUMN_ALIASES

**Before (replace this exact block):**
```python
    "min_qz_pct": ["min_qz_pct", "min_quartz_pct", "quartz_pct"],
}
```

**After:**
```python
    "min_qz_pct": ["min_qz_pct", "min_quartz_pct", "quartz_pct"],

    # === Collar / project (logging review report) ===
    "project_code": ["project_code", "projectcode", "project", "project_d", "projcode", "proj", "tenement", "prospect", "prospect_d"],
    "easting": ["easting", "east", "utm_e", "e", "x", "grid_e", "grid_easting"],
    "northing": ["northing", "north", "utm_n", "n", "y", "grid_n", "grid_northing"],
}
```

**Verify:**
```bash
rg '"project_code"' src/processing/DataManager/column_aliases.py
```
Expected: one match.

---

#### Task A.5.5 — Extend `depth_from` / `depth_to` for logging columns

**Before (replace these two lines):**
```python
    "depth_from": ["sampfrom", "geolfrom", "from", "depth_from", "from_depth", "interval_from"],
    "depth_to": ["sampto", "geolto", "to", "depth_to", "to_depth", "interval_to"],
```

**After:**
```python
    "depth_from": ["sampfrom", "geolfrom", "geology_from", "logging_from", "rc_from", "depth_from_geol", "from", "depth_from", "from_depth", "interval_from"],
    "depth_to": ["sampto", "geolto", "geology_to", "logging_to", "rc_to", "depth_to_geol", "to", "depth_to", "to_depth", "interval_to"],
```

**Verify:**
```bash
rg 'logging_from|depth_from_geol' src/processing/DataManager/column_aliases.py
```
Expected: two matches.

---

### File 2: `src/processing/logging_review_html_report.py`

#### Task A.5.2 — Remove constants PROJECT_CODE_CANDIDATES, EASTING_CANDIDATES, NORTHING_CANDIDATES

**Before (delete this exact block):**
```python
PROJECT_CODE_CANDIDATES = [
    "project_code",
    "projectcode",
    "project",
    "project_d",
    "projcode",
    "proj",
    "tenement",
    "prospect",
    "prospect_d",
]

EASTING_CANDIDATES = [
    "easting",
    "east",
    "utm_e",
    "e",
    "x",
    "grid_e",
    "grid_easting",
]

NORTHING_CANDIDATES = [
    "northing",
    "north",
    "utm_n",
    "n",
    "y",
    "grid_n",
    "grid_northing",
]


```

**After:** Nothing. Next line after `logger = logging.getLogger(__name__)` must be `def generate_logger_html_reports_from_prepped_data`.

**Verify:**
```bash
rg 'PROJECT_CODE_CANDIDATES|EASTING_CANDIDATES|NORTHING_CANDIDATES' src/processing/logging_review_html_report.py
```
Expected: no matches.

---

#### Task A.5.3 — Replace `_resolve_project_code_column` with ColumnResolver

**Before (replace this exact function):**
```python
def _resolve_project_code_column(df: pd.DataFrame) -> Optional[str]:
    columns_lower = {c.lower(): c for c in df.columns}
    for candidate in PROJECT_CODE_CANDIDATES:
        actual = columns_lower.get(candidate.lower())
        if actual:
            return actual
    return None
```

**After:**
```python
def _resolve_project_code_column(df: pd.DataFrame) -> Optional[str]:
    resolver = ColumnResolver(df)
    return resolver.get("project_code")
```

**Verify:**
```bash
rg 'get\("project_code"\)' src/processing/logging_review_html_report.py
```
Expected: one match.

---

#### Task A.5.4 — Replace `_resolve_coordinate_columns` with ColumnResolver

**Before (replace this exact function):**
```python
def _resolve_coordinate_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    columns_lower = {c.lower(): c for c in df.columns}
    east = next((columns_lower[c] for c in EASTING_CANDIDATES if c in columns_lower), None)
    north = next((columns_lower[c] for c in NORTHING_CANDIDATES if c in columns_lower), None)
    return east, north
```

**After:**
```python
def _resolve_coordinate_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    resolver = ColumnResolver(df)
    return resolver.get("easting"), resolver.get("northing")
```

**Verify:**
```bash
rg 'get\("easting"\)|get\("northing"\)' src/processing/logging_review_html_report.py
```
Expected: two matches.

---

### File 3: `src/processing/logging_review_report.py`

#### Task A.5.6 — Replace `_resolve_logging_interval_columns` with ColumnResolver

**Before (replace this exact function):**
```python
def _resolve_logging_interval_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    from_candidates = [
        "geolfrom",
        "geology_from",
        "logging_from",
        "rc_from",
        "depth_from_geol",
    ]
    to_candidates = [
        "geolto",
        "geology_to",
        "logging_to",
        "rc_to",
        "depth_to_geol",
    ]
    columns_lower = {c.lower(): c for c in df.columns}
    log_from = next((columns_lower[c] for c in from_candidates if c in columns_lower), None)
    log_to = next((columns_lower[c] for c in to_candidates if c in columns_lower), None)
    return log_from, log_to
```

**After:**
```python
def _resolve_logging_interval_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    resolver = ColumnResolver(df)
    return resolver.get("depth_from"), resolver.get("depth_to")
```

**Verify:**
```bash
rg 'get\("depth_from"\)|get\("depth_to"\)' src/processing/logging_review_report.py
```
```bash
rg 'from_candidates|to_candidates' src/processing/logging_review_report.py
```
Expected: first has match(es); second has no matches.

---

## Setup — Create `src/reports` package structure (do once before Phase A/B/C)

Do these in order. One file per task.

### Task S.1 — Create `src/reports/__init__.py`

**Before:** N/A (new file).

**After (exact content):**
```python
# reports package - logging review report and future report types
```

**Verify:**
```bash
rg 'reports package' src/reports/__init__.py
```

---

### Task S.2 — Create `src/reports/logging_review/__init__.py`

**Before:** N/A (new file).

**After (exact content):**
```python
# logging_review report (HTML + PDF)
```

**Verify:**
```bash
rg 'logging_review' src/reports/logging_review/__init__.py
```

---

### Task S.3 — Create `src/reports/logging_review/html/__init__.py`

**Before:** N/A (new file).

**After (exact content):**
```python
# HTML report generation
```

**Verify:**
```bash
rg 'HTML report' src/reports/logging_review/html/__init__.py
```

---

### Task S.4 — Create `src/reports/logging_review/html/assets/__init__.py`

**Before:** N/A (new file).

**After (exact content):**
```python
# CSS and JS assets for HTML report
```

**Verify:**
```bash
rg 'CSS and JS' src/reports/logging_review/html/assets/__init__.py
```

---

## Phase A — Extract CSS/JS

### File: `src/processing/logging_review_html_report.py` (line numbers from current file; adjust if file changed)

**Reference:** CSS is in `_render_html`: lines 3757–4631 (`<style>` through `</style>`). JS is lines 4656–4965 (`<script>` through `</script>`).

### Task A.1 — Create `src/reports/logging_review/html/assets/styles.py` with CSS constant

**Mechanical steps (zero judgment):**
1. Extract lines **3758–4630** from `src/processing/logging_review_html_report.py` (the CSS content between `<style>` and `</style>`).
2. In the extracted text, replace every `{{` with `{` and every `}}` with `}` (undo f-string escaping).
3. Create file `src/reports/logging_review/html/assets/styles.py` with exactly:
   - Line 1: `"""CSS for logging review HTML report."""`
   - Line 2: (empty)
   - Line 3: `CSS_STYLES: str = """`
   - Lines 4–N: the extracted (and brace-fixed) CSS from step 1–2.
   - Line N+1: `"""`

**Verify:**
```bash
rg '^CSS_STYLES\s*=' src/reports/logging_review/html/assets/styles.py
```
```bash
wc -l src/reports/logging_review/html/assets/styles.py
```
Expected: first one match; second at least 10 lines.

---

### Task A.2 — In HTML report, add import for CSS_STYLES and use it in `_render_html`

**Before (in `src/processing/logging_review_html_report.py`):** At the top, after the existing `from processing.DataManager.column_aliases import ColumnResolver` line, there is no import from assets. And in `_render_html`, the HTML string contains the literal block from line 3757 `    <style>` through line 4631 `    </style>` (inclusive).

**After — Part 1 (add import):** Insert this line immediately after `from processing.DataManager.keys import ImageKey`:
```python
from processing.reports.logging_review.html.assets.styles import CSS_STYLES
```
If the package is under `src/reports` and run from repo root with `src` on path, use:
```python
from reports.logging_review.html.assets.styles import CSS_STYLES
```

**After — Part 2 (replace style block):** Replace the contiguous block that starts with the line containing exactly `    <style>` (four spaces + `<style>`) and ends with the line containing exactly `    </style>` (four spaces + `</style>`) with exactly these four lines:
```python
    <style>
""" + CSS_STYLES + """
    </style>
```

**Verify:**
```bash
rg 'CSS_STYLES' src/processing/logging_review_html_report.py
```
```bash
rg 'from.*styles import|from.*assets.styles import' src/processing/logging_review_html_report.py
```
Expected: at least two matches (import and usage).

---

### Task A.3 — Create `src/reports/logging_review/html/assets/scripts.py` with JS constant

**Mechanical steps:**
1. Extract lines **4657–4964** from `src/processing/logging_review_html_report.py` (the JS content between `<script>` and `</script>`).
2. In the extracted text, replace every `{{` with `{` and every `}}` with `}`.
3. Create file `src/reports/logging_review/html/assets/scripts.py` with:
   - Line 1: `"""JavaScript for logging review HTML report."""`
   - Line 2: (empty)
   - Line 3: `JS_SCRIPTS: str = """`
   - Lines 4–N: the extracted (and brace-fixed) JS.
   - Line N+1: `"""`

**Verify:**
```bash
rg '^JS_SCRIPTS\s*=' src/reports/logging_review/html/assets/scripts.py
```

---

### Task A.4 — In HTML report, add import for JS_SCRIPTS and use it in `_render_html`

**Before:** The HTML string contains the literal block from the line with `    <script>` (inline script) through the line with `    </script>` (same block).

**After — Part 1:** Add import (next to CSS_STYLES import):
```python
from reports.logging_review.html.assets.scripts import JS_SCRIPTS
```
(or `from processing.reports...` if your import path differs).

**After — Part 2:** Replace the contiguous block that starts with the line containing exactly `    <script>` (the one that starts the inline script, not the Leaflet/Plotly script tags) and ends with the line containing exactly `    </script>` (the one that closes the inline script) with:
```python
    <script>
""" + JS_SCRIPTS + """
    </script>
```

**Verify:**
```bash
rg 'JS_SCRIPTS' src/processing/logging_review_html_report.py
```
```bash
rg 'from.*scripts import|from.*assets.scripts import' src/processing/logging_review_html_report.py
```
Expected: at least two matches.

---

## Phase B — Extract utils (`_safe_str`, `_safe_float`, `_format_metric`)

### Task B.1 — Create `src/reports/logging_review/html/utils.py`

**Before:** N/A (new file).

**After (exact content of the file):**
```python
"""Shared helpers for HTML report (safe formatting, etc.)."""
from typing import Any, Optional
import numpy as np


def _safe_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return float(value)
    except Exception:
        return None


def _format_metric(value: Optional[float], unit: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if unit == "%":
        return f"{value:.1f}%"
    if unit == "m":
        return f"{value:,.0f}m"
    return f"{value:.0f}"
```

**Verify:**
```bash
rg 'def _safe_str|def _safe_float|def _format_metric' src/reports/logging_review/html/utils.py
```
Expected: three matches.

---

### Task B.2 — In `logging_review_html_report.py` remove `_safe_str`, `_safe_float`, `_format_metric` and add import

**Before (replace these three function definitions):**
```python
def _safe_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return float(value)
    except Exception:
        return None


def _format_metric(value: Optional[float], unit: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if unit == "%":
        return f"{value:.1f}%"
    if unit == "m":
        return f"{value:,.0f}m"
    return f"{value:.0f}"
```

**After:** Delete the block above. Add after the other `from processing.` / `from reports.` imports:
```python
from reports.logging_review.html.utils import _safe_str, _safe_float, _format_metric
```
(or `from processing.reports.logging_review.html.utils import ...` to match your layout).

**Verify:**
```bash
rg 'def _safe_str\(|def _safe_float\(|def _format_metric\(' src/processing/logging_review_html_report.py
```
Expected: no matches.
```bash
rg '_safe_str|_safe_float|_format_metric' src/processing/logging_review_html_report.py
```
Expected: one import line and usages only (no def).

---

## Phase C — Extract charts, tables, images, collar_map (one file at a time)

Phase C is “extract module X into new file and replace with import.” For each new file, the **before** is “N/A”; the **after** is “file exists with the exact functions moved from `logging_review_html_report.py`” and in the HTML file the **before** is “the definitions of those functions”; the **after** is “import from the new module; delete the definitions.”

Use the same pattern for each:

- **Task C.N.1:** Create `src/reports/logging_review/html/<module>.py` with exact content = (required imports + the function bodies copied from `logging_review_html_report.py` for the listed functions).  
- **Task C.N.2:** In `logging_review_html_report.py`: add `from reports.logging_review.html.<module> import ...` (list the moved functions); delete the block of lines that defined those functions.

**Function list by module (from plan):**
- **charts.py:** `_plotly_pie_json`, `_plotly_stacked_bar_json`, `_plotly_stacked_bar_pct_json`, `_plotly_strat_grouped_bar_json`, `_plotly_strat_grouped_bar_pct_json`, `_plotly_zonation_bar_json`, `_plotly_outlier_box_json`, `_plotly_outlier_scatter_json`
- **tables.py:** `_render_intervals_table`, `_render_mineralisation_evidence_table`, `_render_zonation_evidence_table`, `_render_logging_detail_evidence_table`, `_render_fines_intervals_table`, `_render_outlier_table`, `_render_grouping_groups`
- **images.py:** `_encode_image_base64` only (image lookup stays or uses DataCoordinator; see plan)
- **collar_map.py:** `_build_map_points`, `_render_map`

**Verification pattern for each:**
- New file: `rg 'def _plotly_pie_json|def _render_map' src/reports/logging_review/html/charts.py` (or the appropriate def names for that module).
- HTML file: `rg 'from reports.logging_review.html.charts import' src/processing/logging_review_html_report.py` and `rg 'def _plotly_pie_json' src/processing/logging_review_html_report.py` → first match, second no match.

Because the moved blocks are large, each C task is defined as:

- **C.1:** Create `charts.py` with the eight `_plotly_*` functions (copy lines 2794–3070 from current HTML file, add imports at top). Verify: `rg 'def _plotly_pie_json' src/reports/logging_review/html/charts.py` and `rg 'def _plotly_outlier_scatter_json' src/reports/logging_review/html/charts.py`.
- **C.2:** In HTML file, add `from reports.logging_review.html.charts import (_plotly_pie_json, _plotly_stacked_bar_json, _plotly_stacked_bar_pct_json, _plotly_strat_grouped_bar_json, _plotly_strat_grouped_bar_pct_json, _plotly_zonation_bar_json, _plotly_outlier_box_json, _plotly_outlier_scatter_json)` and remove the definitions of those eight functions. Verify: `rg 'from.*charts import' src/processing/logging_review_html_report.py` and `rg 'def _plotly_pie_json' src/processing/logging_review_html_report.py` → no match for second.

Repeat the same pattern for **tables.py**, **images.py**, **collar_map.py**. Line ranges: Charts 2794–3070; Tables 2293–2792; Images 468–480; Collar map 689–850 and 3072–3126. For C.3–C.5: create each module file with the listed line range(s), add the corresponding import in the HTML file, delete the moved lines. Verify with `rg 'from.*tables import'`, `rg 'from.*images import'`, `rg 'from.*collar_map import'` and that `def _render_intervals_table`, `def _encode_image_base64`, `def _build_map_points`/`def _render_map` no longer appear in the HTML file.: one task “create file with [list of functions]”, one task “import and remove from HTML file”, with the corresponding grep checks.

---

## Phase D — Extract data layer (when moving to `reports.logging_review.data`)

Same idea as Phase C: one new file per module (prep, columns, outliers), exact copy of the current code from `logging_review_report.py` (and any from HTML report that belongs in data), then replace with imports.

- **data/columns.py:** `resolve_logger_column`, `resolve_drilldate_column`, `resolve_chemistry_columns` (from `logging_review_report.py`).
- **data/outliers.py:** `compute_hybrid_outlier_scores`, `predict_most_likely_strat`.
- **data/prep.py:** `prepare_logging_review_data`, `get_collar_dataframe`, `filter_dataframe_by_logger_and_date`, and any merge/prep helpers that remain after A.5.

**Verification:** After each move, run:
```bash
rg 'from reports.logging_review.data.columns import' src/processing/logging_review_report.py
rg 'def resolve_logger_column' src/processing/logging_review_report.py
```
Expected: first match; second no match (def lives in data/columns.py).

---

## Phase E — Split by tab (one tab at a time, diff HTML after each)

For each tab (overview, comments, mineralisation, zonation, fines, grouping, outlier):

1. **Task E.T.1:** Create `src/reports/logging_review/html/tabs/<tab>.py` with the exact functions that build and render that tab (copy from current `_build_html_report` / `_render_html` the block that builds that tab’s data and the block that renders that tab’s HTML).
2. **Task E.T.2:** In `report_builder` / `report_renderer` (or still-monolithic HTML file), replace that block with a call to the new module and delete the inlined block.
3. **Verify:** Generate HTML; diff output before/after (same file path, compare with previous run).

**Verification grep examples:**
- `rg 'def build_overview_data|def render_overview_section' src/reports/logging_review/html/tabs/overview.py`
- `rg 'from.*tabs.overview import' src/processing/logging_review_html_report.py`

---

## Phase F — Thin entry files

- **Task F.1:** Move the body of `_build_html_report` into `src/reports/logging_review/html/report_builder.py` as the only content (or the main function). In `logging_review_html_report.py`, replace the body with `from reports.logging_review.html.report_builder import build_html_report; _build_html_report = build_html_report` (or re-export).
- **Task F.2:** Same for `_render_html` → `report_renderer.py`.
- **Verify:** `rg 'def _build_html_report|def build_html_report' src/processing/logging_review_html_report.py` (expect at most a one-line wrapper or re-export); `rg 'build_html_report|_render_html' src/reports/logging_review/html/report_builder.py`.

---

## Phase G — Backward-compat wrappers in `processing/`

- **Task G.1:** In `src/processing/logging_review_report.py`, replace the current implementation of each public function with a single line that imports from `reports.logging_review.data` (or `.pdf`) and returns the result. Example: `def prepare_logging_review_data(...): from reports.logging_review.data.prep import prepare_logging_review_data as _impl; return _impl(...)`.
- **Task G.2:** In `src/processing/logging_review_html_report.py`, replace the current implementation of `generate_logger_html_reports_from_prepped_data` (and any other public entry) with an import from `reports.logging_review.html.orchestration` and a call to the same function.
- **Verify:** `rg 'from reports.logging_review' src/processing/logging_review_report.py` and `rg 'from reports.logging_review' src/processing/logging_review_html_report.py`.

---

## Phase H — TypedDict (optional)

- **Task H.1:** Create `src/reports/logging_review/html/types.py` (or `logging_review/types.py`) with the exact TypedDict definitions from the plan (ReportMeta, ReportData, IntervalsForReview, and nested types). No placeholder `# ... etc`.
- **Task H.2:** In `report_builder.py` and `report_renderer.py`, add the type hints (return type and parameter type) using these TypedDicts.
- **Verify:** `rg 'class ReportData|class IntervalsForReview' src/reports/logging_review/html/types.py`.

---

## Docs cleanup

- **Task DOC.1:** Create `docs/LOGGING_REVIEW_REPORT_STATUS.md` with exact structure: `## Done`, `## Deferred`, `## Obsolete`, each with a short bullet list and pointers to other docs.
- **Verify:** `rg '## Done|## Deferred|## Obsolete' docs/LOGGING_REVIEW_REPORT_STATUS.md`
- **Task DOC.2:** Move `docs/Logging_Review_Integration_Review.md` to `docs/archive/Logging_Review_Integration_Review.md` (create `docs/archive` if needed).
- **Verify:** `test -f docs/archive/Logging_Review_Integration_Review.md` (or on Windows: `dir docs\archive\Logging_Review_Integration_Review.md`)
- **Task DOC.3:** At the top of `docs/plans/2026-02-01-html-report-verification.md`, insert exactly: `**Status: Implemented**` on the first line.
- **Verify:** `rg 'Status: Implemented' docs/plans/2026-02-01-html-report-verification.md`

---

## Grep verification cheat sheet

| Task   | File(s) | Verification |
|--------|--------|--------------|
| A.5.1  | column_aliases.py | `rg '"project_code"' src/processing/DataManager/column_aliases.py` → 1 |
| A.5.2  | logging_review_html_report.py | `rg 'PROJECT_CODE_CANDIDATES|EASTING_CANDIDATES|NORTHING_CANDIDATES' src/processing/logging_review_html_report.py` → 0 |
| A.5.3  | logging_review_html_report.py | `rg 'get\("project_code"\)' src/processing/logging_review_html_report.py` → 1 |
| A.5.4  | logging_review_html_report.py | `rg 'get\("easting"\)|get\("northing"\)' src/processing/logging_review_html_report.py` → 2 |
| A.5.5  | column_aliases.py | `rg 'logging_from|depth_from_geol' src/processing/DataManager/column_aliases.py` → 2 |
| A.5.6  | logging_review_report.py | `rg 'get\("depth_from"\)|get\("depth_to"\)' src/processing/logging_review_report.py`; `rg 'from_candidates|to_candidates' src/processing/logging_review_report.py` → 0 |
| S.1–S.4 | src/reports/... | `rg 'reports package|logging_review|HTML report|CSS and JS' src/reports/...` |
| A.1    | styles.py | `rg '^CSS_STYLES\s*=' src/reports/logging_review/html/assets/styles.py` |
| A.2    | logging_review_html_report.py | `rg 'CSS_STYLES' src/processing/logging_review_html_report.py` |
| A.3    | scripts.py | `rg '^JS_SCRIPTS\s*=' src/reports/logging_review/html/assets/scripts.py` |
| A.4    | logging_review_html_report.py | `rg 'JS_SCRIPTS' src/processing/logging_review_html_report.py` |
| B.1    | utils.py | `rg 'def _safe_str|def _safe_float|def _format_metric' src/reports/logging_review/html/utils.py` → 3 |
| B.2    | logging_review_html_report.py | `rg 'def _safe_str\(' src/processing/logging_review_html_report.py` → 0 |
| C.*    | charts/tables/images/collar_map | `rg 'from.*charts import|def _plotly_pie_json'` etc. per module |
| DOC.*  | docs/ | `rg 'Status: Implemented|## Done' docs/...` |

---

## Execution order summary

1. **Phase A.5** (File 1: column_aliases — A.5.1, A.5.5. File 2: logging_review_html_report — A.5.2, A.5.3, A.5.4. File 3: logging_review_report — A.5.6.)
2. **Setup** S.1–S.4 (create `src/reports` and subdirs + `__init__.py`).
3. **Phase A** A.1–A.4 (CSS/JS extract and use).
4. **Phase B** B.1–B.2 (utils extract and use).
5. **Phase C** (charts, tables, images, collar_map — one file at a time, create then import).
6. **Phase D** (data layer — same pattern).
7. **Phase E** (one tab at a time; diff HTML after each).
8. **Phase F** (thin entry files).
9. **Phase G** (wrappers in processing/).
10. **Phase H** (optional TypedDict).
11. **Docs** DOC.1–DOC.3.

Do one file at a time within each phase; run the grep verification after each task.
