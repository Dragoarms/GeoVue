# Code-architect verification: Outlier threshold and scoring fix

**Plan reference:** `docs/plans/2026-02-04-outlier-threshold-and-scoring-fix.md`  
**Review date:** 2026-02-04  
**Scope:** Verify all required changes from the plan are implemented and that recommendations remain correct; note any drift or follow-up.

---

## 1. Required changes (from original spec)

| # | Requirement | Status |
|---|-------------|--------|
| 1 | **Display threshold:** Use threshold 3.0 (not `> 0`) for scatter plot and report outlier list | ✅ Verified |
| 2 | **Mahal norm:** Normalize by 95th percentile; no median centering | ✅ Verified |
| 3 | **IQR flag score:** Max severity across elements, not sum | ✅ Verified |
| 4 | **Final score:** `max(mahal_norm, flag_score)`, not sum | ✅ Verified |

---

## 2. Task 1 — Display threshold (3.0)

**Spec:** Single constant `OUTLIER_DISPLAY_THRESHOLD = 3.0` in `outliers.py`; used in `charts.py` (scatter) and `report_builder.py` (all_outliers).

**Verification:**

- **`src/reports/logging_review/data/outliers.py`**
  - Lines 10–12: `OUTLIER_DISPLAY_THRESHOLD = 3.0` present with comment. ✅

- **`src/reports/logging_review/html/charts.py`**
  - Line 7: `from ..data.outliers import OUTLIER_DISPLAY_THRESHOLD`. ✅
  - Line 231: `is_outlier = df["outlier_score"] > OUTLIER_DISPLAY_THRESHOLD if has_outlier else ...`. ✅
  - Plan suggested `from reports.logging_review.data.outliers`; implementation uses relative `..data.outliers`. Acceptable and consistent with package layout. ✅

- **`src/reports/logging_review/html/report_builder.py`**
  - Line 17: `from reports.logging_review.data.outliers import OUTLIER_DISPLAY_THRESHOLD`. ✅
  - Line 351: `all_outliers = assay_logger_df[assay_logger_df["outlier_score"] > OUTLIER_DISPLAY_THRESHOLD].copy()`. ✅

**Other usages:** Grep for `outlier_score > 0` in `src/` found no matches. No remaining hardcoded `> 0` for outlier display. ✅

**Recommendation:** No change. Implementation matches spec.

---

## 3. Task 2 — Mahalanobis normalization (95th percentile)

**Spec:** Replace median-centered IQR scaling with normalization by 95th percentile: `mahal_norm = mahal / mahal_p95` (with safe handling for zero/NaN).

**Verification:**

- **`src/reports/logging_review/data/outliers.py`** (lines 124–136)
  - `mahal = pd.Series(mahal, index=group.index)` ✅
  - `mahal_p95 = mahal.quantile(0.95)` ✅
  - Zero/NaN guard: `if mahal_p95 == 0 or np.isnan(mahal_p95): mahal_norm = mahal * 0.0` ✅
  - Else: `mahal_norm = mahal / mahal_p95` ✅
  - Comment documents “Score of 1.0 = 95th percentile, >1 = more extreme”. ✅

**Recommendation:** No change. Matches spec; interpretation of score is clear.

---

## 4. Task 3 — IQR flag score = max across elements

**Spec:** Use max of per-element severity instead of sum; variable renamed to `is_outlier_mask` where appropriate.

**Verification:**

- **`src/reports/logging_review/data/outliers.py`** (lines 138–159)
  - Comment: “IQR outlier flags — use max severity across elements, not sum”. ✅
  - `is_outlier_mask = (series < lower) | (series > upper)` ✅
  - `flag_score = flag_score.combine(z.abs().where(is_outlier_mask, 0.0), max)` ✅
  - Loop: `for idx in series[is_outlier_mask].index:` ✅
  - Loop body uses `reason_text.at[idx]`, `series.at[idx]`, `direction = "high" if series.at[idx] > upper else "low"` — correct and unchanged. ✅

**Recommendation:** No change. Additive logic removed; max semantics and reason text behavior match spec.

---

## 5. Task 4 — Final score = max(Mahal, IQR)

**Spec:** `result.loc[group.index, "outlier_score"] = np.maximum(mahal_norm, flag_score).fillna(0.0)`.

**Verification:**

- **`src/reports/logging_review/data/outliers.py`** (line 166)
  - `result.loc[group.index, "outlier_score"] = np.maximum(mahal_norm, flag_score).fillna(0.0)` ✅

**Recommendation:** No change.

---

## 6. Consistency and other call sites

- **Processing layer:** `src/processing/logging_review_report.py` and `src/processing/logging_review_html_report.py` import `compute_hybrid_outlier_scores` from `reports.logging_review.data.outliers`. They do not apply a display threshold; they only attach `outlier_score` to the dataframe. Display threshold is applied only in the HTML report (charts + report_builder). ✅
- **`report_builder.py`** “total_misclassified” (line 353) uses `len(all_outliers)` where `all_outliers` is already filtered by `OUTLIER_DISPLAY_THRESHOLD`. So the count is consistent with the displayed outlier list. ✅
- **No other files** in `src/` use `outlier_score > 0` for display or filtering. ✅

---

## 7. Summary and recommendations

| Item | Verdict |
|------|--------|
| All four required changes | Implemented and match the plan |
| Single source of truth for threshold | `OUTLIER_DISPLAY_THRESHOLD` in `outliers.py`, used in charts and report_builder |
| No leftover `outlier_score > 0` | Confirmed |
| Scoring logic | Mahal 95th-percentile norm + IQR max severity + final max combination as specified |

**Code-architect recommendation:** All required changes from the plan are correctly implemented. No code changes needed for spec compliance. Optional follow-ups (not required by the original spec):

- Consider adding a unit test that asserts scatter/report use `OUTLIER_DISPLAY_THRESHOLD` (e.g. that a sample with score 2.0 is not shown as outlier and one with 4.0 is), if you want regression coverage on the threshold.
- If the threshold ever becomes configurable, expose it via config and keep `OUTLIER_DISPLAY_THRESHOLD` as the default.

Plan and implementation are aligned; verification complete.
