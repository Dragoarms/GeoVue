# Logging Review Report Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a themed date picker and modern themed widgets to the logging review report dialog, and fix report data handling by switching RC metrics joins to interval overlap matching.

**Architecture:** Use `GUIManager` helpers and a new `tkcalendar` wrapper widget for the dialog, while refactoring RC metrics joining in `processing/logging_review_report.py` to use `interval_merger.merge_intervals_by_range` (or `DataJoiner`) for interval overlap. Keep logger filtering per-interval (not per-hole) and avoid imprecise `depth_to` exact joins.

**Tech Stack:** Tkinter/ttk, `GUIManager`, `DialogHelper`, pandas, `processing/DataManager/interval_merger.py`, `processing/DataManager/data_joiner.py`.
---

### Task 1: Add date picker widget wrapper

**Files:**
- Create: `src/gui/widgets/themed_date_entry.py`
- Modify: `src/gui/widgets/__init__.py`
- Modify: `src/requirements.txt`
- Test: `src/gui/tests/test_themed_date_entry.py` (if GUI tests exist) or `src/gui/tests/test_widget_smoke.py`

**Step 1: Write the failing test**
```python
def test_date_entry_fallback_without_tkcalendar():
    # Verify the factory returns a tk.Entry when tkcalendar is unavailable.
    # (Use monkeypatch to simulate ImportError.)
    assert isinstance(widget, tk.Entry)
```

**Step 2: Run test to verify it fails**
Run: `pytest src/gui/tests/test_themed_date_entry.py::test_date_entry_fallback_without_tkcalendar -v`  
Expected: FAIL (factory not implemented)

**Step 3: Write minimal implementation**
```python
# themed_date_entry.py
def create_themed_date_entry(...):
    try:
        from tkcalendar import DateEntry
        # return DateEntry styled with theme_colors
    except Exception:
        # fallback to create_entry_with_validation(...)
```

**Step 4: Run test to verify it passes**
Run: `pytest src/gui/tests/test_themed_date_entry.py::test_date_entry_fallback_without_tkcalendar -v`  
Expected: PASS

**Step 5: Add dependency**
Run: `python -m pip install tkcalendar`  
Update `src/requirements.txt` to include `tkcalendar` (un-pinned or pinned via `pip freeze`).

**Step 6: Commit**
```bash
git add src/gui/widgets/themed_date_entry.py src/gui/widgets/__init__.py src/requirements.txt src/gui/tests/test_themed_date_entry.py
git commit -m "feat: add themed date entry wrapper"
```

---

### Task 2: Update logging review report dialog to use themed widgets

**Files:**
- Modify: `src/gui/logging_review_report_dialog.py`
- Test: `src/gui/tests/test_logging_review_report_dialog.py` (if available)

**Step 1: Write the failing test**
```python
def test_dialog_builds_with_themed_widgets():
    # Instantiate dialog with dummy gui_manager and ensure build does not raise.
    # Optionally check that the date widgets are DateEntry when tkcalendar is available.
    assert dialog is not None
```

**Step 2: Run test to verify it fails**
Run: `pytest src/gui/tests/test_logging_review_report_dialog.py::test_dialog_builds_with_themed_widgets -v`  
Expected: FAIL (test missing or dialog uses old widgets)

**Step 3: Write minimal implementation**
```python
# Use DialogHelper.create_dialog
# Replace ttk.Entry with gui_manager.create_field_with_label or create_entry_with_validation
# Use create_themed_date_entry for date range inputs
# Replace ttk.Button with gui_manager.create_modern_button
# Replace ttk.Checkbutton with gui_manager.create_custom_checkbox
# Style listbox with gui_manager.theme_colors
```

**Step 4: Run test to verify it passes**
Run: `pytest src/gui/tests/test_logging_review_report_dialog.py::test_dialog_builds_with_themed_widgets -v`  
Expected: PASS

**Step 5: Commit**
```bash
git add src/gui/logging_review_report_dialog.py src/gui/tests/test_logging_review_report_dialog.py
git commit -m "feat: apply themed widgets to logging review report dialog"
```

---

### Task 3: Refactor RC metrics merge to interval overlap

**Files:**
- Modify: `src/processing/logging_review_report.py`
- Test: `src/processing/tests/test_logging_review_report.py`

**Step 1: Write the failing test**
```python
def test_rc_metrics_join_uses_interval_overlap():
    # Build a primary df with depth_from/to intervals and metrics df with overlapping interval
    # Expect metrics columns to be populated by overlap, not exact depth_to match.
    assert merged_df.loc[0, "weighted_hardness"] == expected_value
```

**Step 2: Run test to verify it fails**
Run: `pytest src/processing/tests/test_logging_review_report.py::test_rc_metrics_join_uses_interval_overlap -v`  
Expected: FAIL (exact merge misses the overlap)

**Step 3: Write minimal implementation**
```python
# Add helper merge_rc_metrics_by_interval(...)
# Use interval_merger.merge_intervals_by_range(...) with hole_id/from/to keys
# Avoid using _depth_int truncation
```

**Step 4: Run test to verify it passes**
Run: `pytest src/processing/tests/test_logging_review_report.py::test_rc_metrics_join_uses_interval_overlap -v`  
Expected: PASS

**Step 5: Commit**
```bash
git add src/processing/logging_review_report.py src/processing/tests/test_logging_review_report.py
git commit -m "fix: join rc metrics by interval overlap"
```

---

### Task 4: Optional performance filter by hole ids (without losing per-interval logger changes)

**Files:**
- Modify: `src/processing/logging_review_report.py`
- Test: `src/processing/tests/test_logging_review_report.py`

**Step 1: Write the failing test**
```python
def test_logger_filter_preserves_intervals_within_same_hole():
    # Same hole with different logger values; ensure filtering by logger keeps only matching intervals
    assert set(filtered["logger_col"]) == {"LoggerA"}
```

**Step 2: Run test to verify it fails**
Run: `pytest src/processing/tests/test_logging_review_report.py::test_logger_filter_preserves_intervals_within_same_hole -v`  
Expected: FAIL (if filtering is incorrect)

**Step 3: Write minimal implementation**
```python
# Ensure logger filtering is applied per interval, not per hole
# Use hole_ids only to reduce join scope, not to collapse logger data
```

**Step 4: Run test to verify it passes**
Run: `pytest src/processing/tests/test_logging_review_report.py::test_logger_filter_preserves_intervals_within_same_hole -v`  
Expected: PASS

**Step 5: Commit**
```bash
git add src/processing/logging_review_report.py src/processing/tests/test_logging_review_report.py
git commit -m "test: ensure logger filtering is per-interval"
```

