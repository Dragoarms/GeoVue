# Outlier Threshold and Scoring Fix — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task in this session.

**Goal:** Fix outlier scoring so that (1) only genuinely anomalous samples are flagged when plotting and in report outlier lists, and (2) the scoring logic uses a proper Mahalanobis normalization and max-based combination instead of median-centered and additive logic that flags almost everything.

**Architecture:** Apply a display threshold of 3.0 (approx 3-sigma equivalent) in charts and report builder; normalize Mahalanobis by 95th percentile (no median centering); use max severity across IQR violations (not sum); combine Mahal and IQR scores with max (not sum).

**Tech Stack:** Python, pandas, numpy. Files in `src/reports/logging_review/` (data/outliers.py, html/charts.py, html/report_builder.py).

---

## Summary of current bugs

| Issue | Effect |
|-------|--------|
| `outlier_score > 0` threshold | Flags ~100% of data |
| mahal_norm centered on median | Half of samples positive by design |
| Additive flag_score | Minor violations on multiple elements = high score |
| Sum of Mahal + IQR | Two different scales combined arbitrarily |

---

## Task 1: Use proper display threshold (3.0) in charts and report builder

**Files:**
- Modify: `src/reports/logging_review/data/outliers.py` — add constant at top (after imports).
- Modify: `src/reports/logging_review/html/charts.py` — use constant for scatter outlier flag.
- Modify: `src/reports/logging_review/html/report_builder.py` — use same constant for `all_outliers` filter.
- Test: `src/processing/tests/test_logging_review_report.py` (run existing tests; optional: add test for threshold).

**Step 1: Add display threshold constant and use in charts/report_builder**

In `src/reports/logging_review/data/outliers.py`, after the `logger = logging.getLogger(__name__)` line, add:

```python
# Threshold above which a sample is shown as outlier in scatter plots and report lists.
# ~3-sigma equivalent; only genuinely anomalous samples are highlighted.
OUTLIER_DISPLAY_THRESHOLD = 3.0
```

In `src/reports/logging_review/html/charts.py`:
- Add at top with other imports: `from reports.logging_review.data.outliers import OUTLIER_DISPLAY_THRESHOLD`
- Find:
  `is_outlier = df["outlier_score"] > 0 if has_outlier else pd.Series(False, index=df.index)`
- Replace with:
  `is_outlier = df["outlier_score"] > OUTLIER_DISPLAY_THRESHOLD if has_outlier else pd.Series(False, index=df.index)`

In `src/reports/logging_review/html/report_builder.py`:
- Add import for `OUTLIER_DISPLAY_THRESHOLD` from `reports.logging_review.data.outliers`.
- Find:
  `all_outliers = assay_logger_df[assay_logger_df["outlier_score"] > 0].copy()`
- Replace with:
  `all_outliers = assay_logger_df[assay_logger_df["outlier_score"] > OUTLIER_DISPLAY_THRESHOLD].copy()`

**Step 2: Run existing outlier tests**

From repo root:
`python -m pytest src/processing/tests/test_logging_review_report.py -v -k "outlier"`

Expected: existing outlier tests still pass (ranking and single-column behavior).

**Step 3: Commit**

```bash
git add src/reports/logging_review/data/outliers.py src/reports/logging_review/html/charts.py src/reports/logging_review/html/report_builder.py
git commit -m "fix(reports): use OUTLIER_DISPLAY_THRESHOLD 3.0 for scatter and outlier list"
```

---

## Task 2: Normalize Mahalanobis by 95th percentile (no median centering)

**Files:**
- Modify: `src/reports/logging_review/data/outliers.py` (in `compute_hybrid_outlier_scores`, Mahal normalization block).
- Test: `src/processing/tests/test_logging_review_report.py`.

**Step 1: Replace mahal_norm calculation**

In `src/reports/logging_review/data/outliers.py`, in `compute_hybrid_outlier_scores`, find:

```python
        mahal = pd.Series(mahal, index=group.index)
        mahal_iqr = mahal.quantile(0.75) - mahal.quantile(0.25)
        if mahal_iqr == 0:
            mahal_norm = mahal * 0.0
        else:
            mahal_norm = (mahal - mahal.median()) / (mahal_iqr / 1.349)
```

Replace with:

```python
        mahal = pd.Series(mahal, index=group.index)
        # Normalize by 95th percentile so scores are comparable across strats.
        # Score of 1.0 = 95th percentile, >1 = more extreme (no median centering).
        mahal_p95 = mahal.quantile(0.95)
        if mahal_p95 == 0 or np.isnan(mahal_p95):
            mahal_norm = mahal * 0.0
        else:
            mahal_norm = mahal / mahal_p95
```

**Step 2: Run tests**

`python -m pytest src/processing/tests/test_logging_review_report.py -v -k "outlier"`

Expected: PASS.

**Step 3: Commit**

```bash
git add src/reports/logging_review/data/outliers.py
git commit -m "fix(outliers): normalize Mahalanobis by 95th percentile, not median-centered IQR"
```

---

## Task 3: IQR flag scoring — max severity across elements (not sum)

**Files:**
- Modify: `src/reports/logging_review/data/outliers.py` (IQR loop in `compute_hybrid_outlier_scores`).
- Test: `src/processing/tests/test_logging_review_report.py`.

**Step 1: Replace additive flag_score with max**

In `src/reports/logging_review/data/outliers.py`, find the IQR block:

```python
        # IQR outlier flags and per-element severity
        flag_score = pd.Series(0.0, index=group.index)
        reason_text = pd.Series("", index=group.index)
        top_elements = pd.Series("", index=group.index)
        for col in available_cols:
            series = group[col]
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            z = _robust_scale(series)
            is_outlier = (series < lower) | (series > upper)
            flag_score[is_outlier] += z[is_outlier].abs()
            for idx in series[is_outlier].index:
```

Replace with:

```python
        # IQR outlier flags — use max severity across elements, not sum
        flag_score = pd.Series(0.0, index=group.index)
        reason_text = pd.Series("", index=group.index)
        top_elements = pd.Series("", index=group.index)
        for col in available_cols:
            series = group[col]
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            z = _robust_scale(series)
            is_outlier_mask = (series < lower) | (series > upper)
            flag_score = flag_score.combine(z.abs().where(is_outlier_mask, 0.0), max)
            for idx in series[is_outlier_mask].index:
```

Then fix the remaining line in that loop that still references `is_outlier` — change the next line from `series[is_outlier]` to `series[is_outlier_mask]` in the `for idx in ...` (the loop body already uses `reason_text.at[idx]` and `series.at[idx]`; only the iterator must use `is_outlier_mask`: `for idx in series[is_outlier_mask].index:` which is already updated above). Ensure the line that builds `reason` still uses `direction = "high" if series.at[idx] > upper else "low"` and the rest unchanged.

**Step 2: Run tests**

`python -m pytest src/processing/tests/test_logging_review_report.py -v -k "outlier"`

Expected: PASS.

**Step 3: Commit**

```bash
git add src/reports/logging_review/data/outliers.py
git commit -m "fix(outliers): IQR flag score = max severity across elements, not sum"
```

---

## Task 4: Final outlier score = max(mahal_norm, flag_score)

**Files:**
- Modify: `src/reports/logging_review/data/outliers.py` (single line in `compute_hybrid_outlier_scores`).
- Test: `src/processing/tests/test_logging_review_report.py`.

**Step 1: Replace sum with max**

In `src/reports/logging_review/data/outliers.py`, find:

```python
        result.loc[group.index, "outlier_score"] = (mahal_norm + flag_score).fillna(0.0)
```

Replace with:

```python
        result.loc[group.index, "outlier_score"] = np.maximum(mahal_norm, flag_score).fillna(0.0)
```

**Step 2: Run tests**

`python -m pytest src/processing/tests/test_logging_review_report.py -v -k "outlier"`

Expected: PASS.

**Step 3: Commit**

```bash
git add src/reports/logging_review/data/outliers.py
git commit -m "fix(outliers): combine Mahal and IQR scores with max, not sum"
```

---

## Execution order

Tasks 1–4 must be implemented in order. Task 1 is independent of 2–4; Tasks 2–4 are all in the same file and should be done in sequence to keep tests passing and commits small.

After all tasks: run full test suite for logging review and optionally manual smoke of HTML report scatter + outlier list.

---

## Verification (2026-02-04)

Implementation was completed and verified. Code-architect review: see **`docs/plans/2026-02-04-outlier-fix-verification-review.md`**. All four required changes are implemented and match the plan; no remaining `outlier_score > 0` in display/filter paths; recommendations stand.
