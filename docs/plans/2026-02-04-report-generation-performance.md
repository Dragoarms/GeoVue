# Report Generation Performance — Implementation Plan

> **For Claude:** Use subagent-driven-development to implement task-by-task. Each task: TDD (failing test first, then minimal code), then spec review, then code quality review. All tests must pass.

**Goal:** Reduce logging review report generation time by removing inefficient practices: duplicate image I/O, slow DataFrame iteration (iterrows), and per-row loops that can be vectorized.

**Architecture:** (1) Cache image lookups by (hole_id, depth_to) so the same image is never read/encoded twice. (2) Replace iterrows() with itertuples() in report_builder and optionally merge two passes over assay_logger_df into one. (3) Vectorize Mahalanobis distance computation in data/outliers.py. No change to report output semantics or HTML structure.

**Tech Stack:** Python, pandas, numpy. Tests: pytest in `src/processing/tests/` and `src/reports/` if needed.

---

## Identified inefficiencies

| # | Issue | Location | Impact |
|---|--------|----------|--------|
| 1 | Image lookup has no cache; same (hole_id, depth_to) can be looked up many times across fines, logging_detail, grouping, outliers. Each call: get_image_path + open file + base64 encode. | `processing/logging_review_html_report.py`: `_lookup_interval_image`, `_build_*_with_images` | High when include_images=True and many intervals share same hole/depth |
| 2 | Two full passes over assay_logger_df using iterrows(): one for fines (report_builder ~316), one for logging_detail issues (~345). iterrows() is slow. | `reports/logging_review/html/report_builder.py` | Medium for large assay_logger_df |
| 3 | Outlier rows built with iterrows() over display_outliers (~448). | Same file | Lower (typically small display set) |
| 4 | Mahalanobis computed in a per-row loop (iterrows over z_centered) in compute_hybrid_outlier_scores. | `reports/logging_review/data/outliers.py` ~116–127 | Medium for large groups |

---

## Task 1: Add image lookup cache

**Files:**
- Modify: `src/processing/logging_review_html_report.py` — add an optional cache to `_lookup_interval_image` (or a wrapper that builds intervals with a cache). Callers `_build_intervals_with_images`, `_build_grouping_groups_with_images`, `_build_outlier_intervals_with_images` must use the same cache for the duration of one report build.
- Test: `src/processing/tests/test_logging_review_report.py` (or new `test_logging_review_html_report_performance.py`).

**TDD steps:**

1. **Write failing test:** Test that when building intervals with images for two interval lists that reference the same (hole_id, depth_to), the file is read at most once (e.g. mock `open` or data_coordinator.get_image_path and assert call count per key). Or test that a cache is used: e.g. provide a pre-populated cache and assert it was used (no new read for cached key).
2. **Run test:** `pytest src/processing/tests/test_logging_review_report.py -v -k "image"` (or new test file). Expect failure (no cache yet).
3. **Implement:** Add an optional `image_cache: Optional[Dict[Tuple[str, float], Optional[str]]] = None` to `_lookup_interval_image`. If provided, check cache before get_image_path; store result in cache after encode. In report_builder (or the single entry point that builds all intervals), create one cache dict and pass it into each _build_*_with_images call (those functions need to accept and forward the cache to _lookup_interval_image). Ensure cache key is (str(hole_id).upper(), float(depth_to)) to match _lookup_interval_image logic.
4. **Run tests:** All logging_review tests pass. New test passes.
5. **Commit:** e.g. "perf(report): cache image lookups by (hole_id, depth_to) to avoid duplicate I/O"

**Note:** If the report_builder builds intervals and calls processing functions that do the image lookup, the cache must be created in report_builder and passed into the processing layer. Check `_build_intervals_with_images` signature and where it is called from.

---

## Task 2: Replace iterrows() with itertuples() in report_builder (fines + logging_detail)

**Files:**
- Modify: `src/reports/logging_review/html/report_builder.py` — replace the two loops over `assay_logger_df.iterrows()` (fines loop and logging_detail loop) with itertuples(). Build each row dict from the tuple (use column index or name via itertuples(index=False, name='Row')). Ensure None/missing handling matches (row.get(col) vs getattr(row, col, None)).
- Test: `src/processing/tests/test_logging_review_report.py` — add a test that builds a minimal assay_logger_df and calls the same logic that produces fines_intervals and magnetite_intervals (e.g. via build_html_report with minimal data, or extract the interval-building into a testable helper). Assert counts and that a known row with a known issue appears in the expected list.

**TDD steps:**

1. **Write failing test:** Test that for a small assay_logger_df (e.g. 3 rows) with one row that has a fines issue and one that has a magnetite issue, the returned fines_intervals and magnetite_intervals have length 1 each and contain the expected hole_id/depth_to. (This may require refactoring the interval-building into a function that can be called from tests; if so, do that first with a test that the refactor preserves behavior.)
2. **Run test:** Expect failure or skip if refactor needed.
3. **Implement:** Replace iterrows() with itertuples() in both loops. Use getattr(tup, col, None) or column index. Preserve exact structure of appended dicts.
4. **Run tests:** All tests pass. Optionally benchmark with a 5k-row DataFrame to confirm speedup.
5. **Commit:** "perf(report): use itertuples() instead of iterrows() for interval building"

---

## Task 3: Single pass over assay_logger_df for fines + logging_detail

**Files:**
- Modify: `src/reports/logging_review/html/report_builder.py` — merge the fines loop and the logging_detail (magnetite, goethite, carbonate, sulphide, manganese, mafics, magnesium) loop into one loop over assay_logger_df. In each iteration, compute fines issue and all logging_detail issues; append to fines_intervals and to each of the 7 other lists as needed.
- Test: Same as Task 2 — assert fines_intervals and each logging_detail list match the previous (two-pass) output for a fixture.

**TDD steps:**

1. **Write failing test:** Test that combined pass produces identical (length and content) fines_intervals, magnetite_intervals, ... as the two-pass implementation. Use a small DataFrame and compare result lists (order may matter for display).
2. **Run test:** Pass (two-pass still there) or fail if test compares to "expected" from single-pass.
3. **Implement:** Single loop; compute all flags per row; append to each list as needed.
4. **Run tests:** All pass. Remove the duplicate loop.
5. **Commit:** "perf(report): single pass over assay_logger_df for fines and logging_detail intervals"

---

## Task 4: Vectorize Mahalanobis in outliers.py

**Files:**
- Modify: `src/reports/logging_review/data/outliers.py` — replace the loop `for idx, row in z_centered.iterrows(): ... mahal.append(...)` with a vectorized computation. For each group, z_centered is (n, d). Mahalanobis distance for each row: sqrt((x - mu) @ inv_cov @ (x - mu).T) for row x; with matrix X (n x d), diag(X @ inv_cov @ X.T) gives squared distances (when mean is 0). So: diff = z_centered.fillna(0) (or handle NaNs by row: only use valid dimensions). For full vectorization with NaN: either compute per-row in a list comp with numpy, or mask. Simple approach: mahal_sq = np.einsum('ij,jk,ik->i', z_centered.fillna(0), inv_cov, z_centered.fillna(0)); mahal = np.sqrt(np.maximum(mahal_sq, 0)). Handle NaN rows by setting their mahal to 0.
- Test: `src/processing/tests/test_logging_review_report.py` or `src/reports/logging_review/data/` test — add test that compute_hybrid_outlier_scores(assay_df, strat_col, chem_cols) gives the same outlier_score values (within float tolerance) as the current implementation. Compare on a small DataFrame (e.g. 50 rows, 2 strats, 3 chem cols).

**TDD steps:**

1. **Write failing test:** Call compute_hybrid_outlier_scores on a fixed small DataFrame; snapshot the outlier_score Series (or first few values). Then change implementation to vectorized; run test again to ensure scores match (e.g. np.allclose).
2. **Run test:** Pass with current implementation.
3. **Implement:** Vectorize the Mahalanobis block. Keep NaN handling: rows with all NaN should get 0. Use fillna(0) for the matrix multiply and then zero out mahal where row was all NaN.
4. **Run tests:** test_hybrid_outlier_scores_* and new test pass.
5. **Commit:** "perf(outliers): vectorize Mahalanobis distance in compute_hybrid_outlier_scores"

---

## Task 5: Replace outlier display_outliers iterrows() with itertuples()

**Files:**
- Modify: `src/reports/logging_review/html/report_builder.py` — replace the loop over display_outliers.iterrows() (~448) with itertuples(). Same dict structure for outlier_rows.
- Test: Existing test that checks outlier rows or report content; or add a small test that build_html_report with outlier data produces outlier_rows with expected keys/values.
**TDD steps:** Same pattern: test first (behavior unchanged), then replace iterrows with itertuples, run tests, commit.

---

## Verification (after all tasks)

- Run full test suite: `pytest src/processing/tests/test_logging_review_report.py -v`
- Run any report integration tests. Confirm no regression in HTML content (e.g. test_html_report_contains_no_raw_template_placeholders still passes).
- Optionally: time report generation on a large dataset (e.g. 9k assay rows, include_images=True) before/after to confirm improvement.

---

## Summary

| Task | Change | Test approach |
|------|--------|----------------|
| 1 | Image lookup cache | Mock/count or cache hit assertion |
| 2 | itertuples for fines + logging_detail | Fixture + compare interval lists |
| 3 | Single pass over assay_logger_df | Compare combined vs two-pass output |
| 4 | Vectorize Mahalanobis | Snapshot or allclose vs current scores |
| 5 | itertuples for outlier_rows | Existing or small behavior test |

All changes must preserve existing behavior; tests must be written first (TDD) and pass before and after each task.
