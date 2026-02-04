# Tables Layout, Row Highlight, Zonation Headers & Charts Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task (this session).

**Goal:** Standardise evidence table layouts, add row highlight when an image is expanded, rename zonation evidence table headers, add Match/Mismatch and Significance columns to the zonation evidence table, and convert the zonation tab to percentage accuracy charts with Logger and Team side-by-side and evidence table below.

**Architecture:** (1) Tables: shared pattern for checkbox, sortable, image cell; zonation table gets new headers and two new columns with data from processing. (2) Scripts: in `handleImageExpand`, mark the source row (e.g. `tr`) with a highlight class and clear it when collapsing or switching image; CSS for highlight. (3) Processing: extend `_collect_zonation_mismatch_rows` to add `validation` ("Mismatch") and `significance` (Low when logged zonation is adjacent to should_be, else High). (4) Charts: new percentage bar chart and zonation tab layout with two charts (Logger % and Team %) then evidence table.

**Tech Stack:** Python (report_builder, processing), HTML/JS/CSS in `src/reports/logging_review/html/` (tables.py, charts.py, tabs/profile.py, assets/scripts.py, assets/styles.py).

---

## Task 1: Add row highlight when image is expanded

**Files:**
- Modify: `src/reports/logging_review/html/assets/scripts.py` (handleImageExpand and click-outside handler)
- Modify: `src/reports/logging_review/html/assets/styles.py` (add .source-row-highlight)

**Step 1: Add CSS for highlighted row**

In `styles.py`, after the `.expandable-img.expanded` block (around line 530), add:

```python
        .source-row-highlight {
            background: rgba(13, 91, 136, 0.12) !important;
            outline: 2px solid #0d5b88;
        }
```

**Step 2: Update handleImageExpand to set/clear row highlight**

In `scripts.py`, in `handleImageExpand`:
- Before closing other expanded images, remove class `source-row-highlight` from any `tr` that has it (e.g. `document.querySelectorAll('tr.source-row-highlight').forEach(...)`).
- After expanding the clicked image, get the row: `const row = img.closest('tr'); if (row) row.classList.add('source-row-highlight');`
- When toggling off (wasExpanded), remove `source-row-highlight` from that row.

In the click-outside handler that closes expanded images, also remove `source-row-highlight` from all `tr` elements.

**Step 3: Verify**

Run the app, open a logging review report, go to any tab with an evidence table (e.g. Mineralisation or Zonation), click an image to expand: the row containing that image should highlight. Click another image: previous row unhighlights, new row highlights. Click outside: image closes and row unhighlights.

**Step 4: Commit**

```bash
git add src/reports/logging_review/html/assets/scripts.py src/reports/logging_review/html/assets/styles.py
git commit -m "feat(report): highlight evidence table row when image is expanded"
```

---

## Task 2: Zonation evidence table – rename headers and add Match/Mismatch + Significance

**Files:**
- Modify: `src/processing/logging_review_html_report.py` (_collect_zonation_mismatch_rows: add validation and significance to each row)
- Modify: `src/reports/logging_review/html/tables.py` (_render_zonation_evidence_table: new headers, Validation and Significance columns)

**Step 1: Add validation and significance in _collect_zonation_mismatch_rows**

In `logging_review_html_report.py`, inside the loop where `rows_out.append({...})` (around line 801), add:
- `"validation": "Mismatch"` (all rows in this table are mismatches).
- `"significance":` use a helper: e.g. if logged zonation and should_be are adjacent in the list `["Un", "Le", "De", "Hy", "Pr"]` then `"Low"`, else `"High"`. Implement a small helper `_zonation_adjacent(logged, should_be)` that returns True when indices differ by 1.

Example helper (add near the function):

```python
def _zonation_adjacent(logged: str, should_be: str) -> bool:
    order = ["Un", "Le", "De", "Hy", "Pr"]
    if logged not in order or should_be not in order:
        return False
    return abs(order.index(logged) - order.index(should_be)) == 1
```

Then in the appended dict: `"significance": "Low" if _zonation_adjacent(zonation, should_be) else "High"`.

**Step 2: Update zonation evidence table headers and columns in tables.py**

In `_render_zonation_evidence_table`:
- Rename headers: "Total gangue %" → "Total gangue % Logged"; "De %" → "Logged De %"; "Hy %" → "Logged Hy %"; "Pr %" → "Logged Pr %".
- After "Should be", insert two columns: Validation (with class `validation-mismatch` and data-sort), then Significance (with class `significance-{significance.lower()}` and data-sort), using `item.get("validation", "Mismatch")` and `item.get("significance", "High")`.
- Keep column order: checkbox, Hole, Depth, Logged zonation, Should be, **Validation**, **Significance**, Total gangue % Logged, Logged De %, Logged Hy %, Logged Pr %, Image.

**Step 3: Run tests / smoke test**

If there are tests for report building or zonation, run them. Otherwise generate a report and open the Zonation tab: evidence table should show the new headers and Validation (Mismatch) and Significance (Low/High) columns.

**Step 4: Commit**

```bash
git add src/processing/logging_review_html_report.py src/reports/logging_review/html/tables.py
git commit -m "feat(report): zonation evidence table headers, validation and significance columns"
```

---

## Task 3: Zonation accuracy by category as percentage and add Team chart (two charts side-by-side)

**Files:**
- Modify: `src/reports/logging_review/html/charts.py` (add or change zonation bar to percentage; support optional title)
- Modify: `src/reports/logging_review/html/tabs/profile.py` (use percentage data, two charts in a two-panel layout, evidence table below)

**Step 1: Add percentage zonation bar chart in charts.py**

Add a new function (or extend existing) so the zonation bar chart shows percentages:

- Input: `categories: List[str]`, `correct: List[int]`, `incorrect: List[int]`, `title: str`.
- For each category i: total_i = correct[i] + incorrect[i]; correct_pct_i = (100.0 * correct[i] / total_i) if total_i > 0 else 0; incorrect_pct_i = (100.0 * incorrect[i] / total_i) if total_i > 0 else 0.
- Build two traces: Correct % and Incorrect % (or stacked bar so each bar sums to 100%).
- Layout: yaxis range [0, 100], ticksuffix "%", title as given.

Example signature and behaviour:

```python
def _plotly_zonation_bar_pct_json(
    categories: List[str], correct: List[int], incorrect: List[int], title: str
) -> Tuple[str, str]:
    """Clustered bar: Correct % vs Incorrect % by zonation category (each category sums to 100%)."""
    n = len(categories)
    correct_pct = []
    incorrect_pct = []
    for i in range(n):
        c, inc = correct[i] if i < len(correct) else 0, incorrect[i] if i < len(incorrect) else 0
        total = c + inc
        if total > 0:
            correct_pct.append(100.0 * c / total)
            incorrect_pct.append(100.0 * inc / total)
        else:
            correct_pct.append(0.0)
            incorrect_pct.append(0.0)
    data = [
        {"type": "bar", "x": categories, "y": correct_pct, "name": "Correct %", "marker": {"color": "#2f7d61"}},
        {"type": "bar", "x": categories, "y": incorrect_pct, "name": "Incorrect %", "marker": {"color": "#c9382a"}},
    ]
    layout = {
        "title": {"text": title},
        "barmode": "group",
        "margin": {"t": 40, "b": 40},
        "height": 280,
        "autosize": True,
        "yaxis": {"title": {"text": "%"}, "range": [0, 100], "ticksuffix": "%"},
    }
    return json.dumps(data), json.dumps(layout)
```

**Step 2: Profile tab – two charts (Logger, Team) and evidence table below**

In `tabs/profile.py`:
- Read from `pz`: `correct_counts`, `mismatch_counts` (logger), and `correct_counts_team`, `mismatch_counts_team` (team). Note: these are dicts keyed by category ("Un", "Le", "De", "Hy", "Pr"); build lists in that order: `z_correct = [pz.get("correct_counts", {}).get(c, 0) for c in z_cats]`, same for `z_mismatch`, and `z_correct_team`, `z_mismatch_team`.
- Build two chart JSONs: one with `_plotly_zonation_bar_pct_json(z_cats, z_correct, z_mismatch, "Logger: Accuracy by category (%)")` and one with team data and title "Team: Accuracy by category (%)".
- Render a two-panel layout at the top (e.g. `class="two-panel"` or reuse overview style): left panel card with Logger chart, right panel card with Team chart.
- Below that, keep the evidence section unchanged (same format as always): Evidence Table with the zonation evidence table.

**Step 3: Verify**

Generate report, open Zonation tab: two percentage charts side-by-side (Logger and Team), then evidence table below. Y-axis should show 0–100%.

**Step 4: Commit**

```bash
git add src/reports/logging_review/html/charts.py src/reports/logging_review/html/tabs/profile.py
git commit -m "feat(report): zonation accuracy by category as %, Logger and Team charts side-by-side"
```

---

## Task 4: Standardise evidence table layouts in tables.py

**Files:**
- Modify: `src/reports/logging_review/html/tables.py`

**Step 1: Align table structure and class usage**

- Ensure all evidence tables that have sortable columns use the same thead pattern: `<th></th>` for checkbox, then `<th class="sortable">` for sortable columns, with consistent `data-i18n-*` where used.
- Ensure image column is last and uses `class="image-cell-compact"` and the same expandable image markup (`expandable-img img-small`, `handleImageExpand(this)`).
- Where one table uses `data-sort` on cells and another doesn’t for the same logical column, add `data-sort` so sorting is consistent (e.g. zonation table already has sortable; intervals table does not use sortable – leave intervals table as is per existing behaviour).
- Apply small alignment only where it doesn’t change behaviour: e.g. ensure mineralisation, zonation, logging detail, and outlier tables all use `<table class=\"interval-table evidence-table sortable-table\" ...>` (or with extra classes like `compact-table` where already used); ensure each has checkbox, then content columns, then image column.

**Step 2: No new tests required**

Manual check: all evidence tables render and sort where expected; image expand still works with row highlight.

**Step 3: Commit**

```bash
git add src/reports/logging_review/html/tables.py
git commit -m "refactor(report): standardise evidence table layout and classes"
```

---

## Execution note

Run tasks in order (1 → 2 → 3 → 4). After each task: run spec compliance review, then code quality review, then mark task complete and proceed. After all tasks, run final code review and use superpowers:finishing-a-development-branch.
