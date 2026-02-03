# Grouping Tab Investigation – Modular Report vs Standalone

**Date:** 2026-02-03  
**Scope:** Make the Grouping tab functional by aligning with the old standalone logic where needed.

---

## 1. Current Grouping Tab (Modular Report)

### 1.1 Entry point and data flow

- **Tab render:** `src/reports/logging_review/html/tabs/grouping.py`  
  `render_grouping_section(report, intervals, logger_id)` builds the HTML.
- **Data sources:**
  - `report["grouping_kpis"]` → `avg_group_interval_m`, `max_group_interval_m`
  - `report["grouping_summary"]` → list of summary lines (e.g. “Groups evaluated: N”, “- Fe_pct: X / N groups exceed CV 100%”)
  - `report["grouping_columns_used"]` → display names for the “Grouping columns used” box
  - `intervals["grouping"]` → list of **groups for review** (each with `strat`, `cv_max`, `intervals`, etc.) passed to `_render_grouping_groups()` in `html/tables.py`

### 1.2 Where the data is produced

All built in `src/reports/logging_review/html/report_builder.py`:

1. **Group columns**  
   `group_cols, group_cols_display = _build_resolved_group_cols(assay_logger_df, hole_col, logger_col, strat_col, resolver_assay)`  
   Implemented in `src/processing/logging_review_html_report.py` (`_build_resolved_group_cols`).

2. **KPIs + groups for review**  
   `grouping_avg_m, grouping_max_m, grouping_groups_for_review = _grouping_avg_max_interval_and_groups(assay_logger_df, group_cols, chem_actual_cols, resolver_assay, hole_col, depth_from_col, depth_to_col, strat_col, top_n_groups=15)`

3. **Summary text**  
   `grouping_summary = _summarize_grouping_accuracy_by_interval(assay_logger_df, group_cols, chem_actual_cols)`  
   Implemented in `src/processing/logging_review_report.py`.

4. **Intervals with images**  
   `grouping_with_groups = _build_grouping_groups_with_images(data_coordinator, grouping_groups_for_review, include_images)`  
   Then `intervals_for_review["grouping"] = grouping_with_groups`.

So the tab is wired correctly: if `_grouping_avg_max_interval_and_groups` and the rest return sensible data, the tab will show it.

### 1.3 Current grouping logic (in processing)

**`_build_resolved_group_cols`** (`logging_review_html_report.py`):

- Starts with `[hole_col, logger_col, strat_col]`.
- Adds optional: prospect_d, stratsum, lithology (via ColumnResolver).
- Adds any column matching `Min_*_pct` (or explicit min_80/50/20/10/5/0_5).
- Returns `(group_cols, display_names)`.

**`_grouping_avg_max_interval_and_groups`** (`logging_review_html_report.py`):

- **Input:** Assay DataFrame (raw intervals), `group_cols`, `chem_actual_cols`, resolver, hole/depth_from/depth_to/strat, `top_n_groups=20`.
- **Guard:** If any `group_cols` missing or `not depth_to_col` → return `(None, None, [])`.
- **CV elements:** From `GROUPING_CV_ELEMENTS = ["fe_pct", "sio2_pct", "al2o3_pct", "mgo_pct", "cao_pct", "s_pct"]`; only columns present in `chem_actual_cols` are used; fallback `chem_actual_cols[:6]` if none resolved.
- **Per group (df.groupby(group_cols)):**
  - Interval length = `depth_to - depth_from` per row; then **mean** and **max** of those lengths in the group.
  - For each chemistry column: CV = `std(ddof=1) / abs(mean) * 100` on raw assay values in the group.
  - If **max CV across elements > 100%**, group is added to “for review” list.
- **Sort** by CV descending, take top `top_n_groups`.
- **Return:** `(avg_all_mean_length, max_all_max_length, groups_for_review)` where each group has `group_key`, `strat`, `cv_max`, `mean_interval_m`, `max_interval_m`, `significance` (High if any of Fe/SiO2/Al2O3 over 100%, else Low), and `intervals` (list of hole_id, depth_from, depth_to, strat, issue, geochem).

So the **current** implementation:

- Works on **raw assay intervals** (no prior combine step).
- Uses **resolved** grouping columns (hole, logger, strat + optional prospect/strat/min_*).
- Uses **interval length** = per-row length, then mean/max **within group** (not the combined span).
- Flags groups with **any** of the CV elements > 100% and shows top N by max CV.

---

## 2. Old Standalone Approach

### 2.1 combine_intervals.py

- **Input:** Filtered dataset (one row per sample/interval).
- **Grouping columns (fixed):**  
  `['HOLEID', 'LoggedBy', 'Prospect_D', 'StratSum', 'Min_80_pct', 'Min_70_pct', 'Min_60_pct', 'Min_50_pct', 'Min_40_pct', 'Min_30_pct', 'Min_20_pct', 'Min_10_pct', 'Min_5_pct', 'Min_2_pct', 'Min_1_pct', 'Min_0_5_pct', 'LithComments']`
- **Aggregation:**
  - `SAMPFROM = min(SAMPFROM)`, `SAMPTO = max(SAMPTO)` → one combined interval per group.
  - `LoggingIntervalLength = SAMPTO - SAMPFROM` (length of the **combined** interval).
  - For each `*_pct_BEST` / `*_ppm_BEST`: `count`, `min`, `max`, `median`, `mean`, `std` (sample std).
  - ReturnDate_D completeness.
  - `categorize_intervals(LoggingIntervalLength)` → e.g. "0-1m", "1-2m", … "10+".

So the **old** pipeline first **combines** intervals into one row per group, with pre-aggregated mean/std per element.

### 2.2 QAQC_07_Grouping_Accuracy_No_charts.py

- **Input:** Combined intervals DataFrame (output of combine_intervals), i.e. **one row per group**.
- **CV:** For Fe_pct_BEST, SiO2_pct_BEST, Al2O3_pct_BEST:  
  `CV = (std / mean) * 100` using the **pre-aggregated** `*_mean` and `*_std` columns (same formula, same concept as current).
- **Validity:** CV ≤ 100% → Valid; CV > 100% → Invalid; group valid only if **all three** elements valid.
- **Invalid_Reasons:** Which element(s) failed and optional count/range.
- **Outputs:**  
  - Group validity counts (Valid/Invalid/Unknown).  
  - Summaries by interval size (e.g. `summarize_by_intervalsize`, `summarize_cv_by_intervalsize`).  
  - Extreme groups (e.g. top 5 by Fe CV, top 5 by Si CV) with HOLEID, From–To, length, element ranges.

So the **old** approach:

- Uses a **fixed**, wide set of grouping columns (including many Min_*_pct and LithComments).
- Works on **combined** intervals (one row per group); **interval length** = combined span (max to − min from).
- Restricts **validity** to Fe, SiO2, Al2O3 (same 100% threshold); reports validity and extreme groups.

---

## 3. Comparison and Gap Analysis

| Aspect | Current (modular) | Old (standalone) |
|--------|-------------------|-------------------|
| **Data shape** | Raw assay rows, groupby in code | Pre-combined one row per group (combine_intervals) |
| **Grouping columns** | Resolved: hole, logger, strat + optional prospect_d/stratsum/lithology + any Min_*_pct in DataFrame | Fixed list: HOLEID, LoggedBy, Prospect_D, StratSum, Min_80…Min_0_5_pct, LithComments |
| **Interval length (KPI)** | Mean/max of **per-row** lengths in group | **Combined** span (SAMPTO−SAMPFROM) per group |
| **CV formula** | Same: std/mean×100 within group | Same, on pre-aggregated mean/std |
| **Elements for “invalid”** | fe, sio2, al2o3, mgo, cao, s (any over 100% → flag) | Fe, SiO2, Al2O3 only; all three must pass for “Valid” |
| **Output** | Top N groups by max CV, with constituent intervals | Validity counts, by-interval-size summaries, extreme Fe/Si groups |

### 3.1 Why the tab might appear “not functional”

1. **Empty or wrong columns**
   - `depth_to_col` or `depth_from_col` not resolved → guard in `_grouping_avg_max_interval_and_groups` returns `(None, None, [])` → KPIs “n/a”, no groups.
   - `group_cols` or `chem_actual_cols` missing or wrong (e.g. different names in data) → no or wrong grouping/CV.

2. **No chemistry columns for CV**
   - If no element from GROUPING_CV_ELEMENTS is in `chem_actual_cols` and fallback is empty, no CV is computed → no group ever has max_cv > 100 → “No groups to review” even when data exists.

3. **Different grouping columns**
   - If the dataset uses the same columns as the old pipeline (Prospect_D, StratSum, Min_*_pct, LithComments), `_build_resolved_group_cols` can be extended to include LithComments and the same Min_* set so grouping matches. Otherwise groups (and thus which intervals are “for review”) can differ from the old report.

4. **KPI meaning**
   - Current “avg group interval” = mean of per-row interval lengths; “max” = max of those. Old report uses combined interval length. So numbers can differ even when logic is correct.

---

## 4. Recommended Approaches

### 4.1 Quick checks (why it might be broken)

- Ensure **depth columns** are resolved for the assay DataFrame (e.g. SAMPFROM/SAMPTO or GeoVue equivalents) and passed as `depth_from_col` / `depth_to_col` into the report builder.
- Ensure **chemistry columns** are resolved and passed as `chem_actual_cols` and that at least Fe, SiO2, Al2O3 (or their aliases) are present.
- Log or debug in `_grouping_avg_max_interval_and_groups`: `group_cols`, `depth_to_col`, `len(available)`, and first group’s CV so you can see if grouping and CV run at all.

### 4.2 Align with old behaviour (optional but useful)

- **Grouping columns:**  
  In `_build_resolved_group_cols`, optionally add the same set as combine_intervals where columns exist (e.g. Prospect_D, StratSum, Min_80_pct … Min_0_5_pct, LithComments). Use ColumnResolver for semantic names so both old (HOLEID, LoggedBy, Prospect_D, …) and GeoVue names work.
- **Interval length (optional):**  
  If you want KPIs to match the old report, add a path that computes “combined” length per group (max(depth_to) − min(depth_from)) and report **avg/max of those** as the two KPIs; keep current per-row mean/max as an alternative if desired.
- **Validity rule:**  
  Current “any of fe/sio2/al2o3/mgo/cao/s over 100%” is broader than old “all three Fe, SiO2, Al2O3 must be ≤100%”. You can restrict “groups for review” to only groups where at least one of Fe/SiO2/Al2O3 has CV > 100% (already partly done via `GROUPING_MAJOR_ELEMENTS` and significance); keep MgO/CaO/S as optional extra flags if desired.

### 4.3 Reuse combine_intervals in the pipeline (optional)

- If you want **identical** grouping and combined intervals to the old script, you could:
  - Add a step that runs the same logic as `combine_intervals` (same grouping list, same aggregations) on the assay data used for the report.
  - Then either:
    - Feed the **combined** DataFrame into a small adapter that produces `grouping_groups_for_review` (and optionally KPIs) from the combined rows with CV > 100%, or
    - Keep current in-memory grouping but change `_build_resolved_group_cols` to use the same column set as combine_intervals so that at least group membership matches.

### 4.4 Tests

- Existing tests in `src/processing/tests/test_logging_review_report.py` already cover:
  - `_grouping_avg_max_interval_and_groups` with missing `depth_to_col` → empty.
  - High CV in group → non-empty groups with `group_key`, `cv_max`, `intervals`.
  - Low CV → empty groups.
- Add an integration-style test that builds a minimal report (or calls the builder with a small DataFrame) and asserts that `report["grouping_kpis"]` and `intervals["grouping"]` are present and, for a known high-CV dataset, non-empty.

---

## 5. File Reference

| Purpose | Current (modular) | Old standalone |
|--------|--------------------|-----------------|
| Tab UI | `src/reports/logging_review/html/tabs/grouping.py` | — |
| Table render | `src/reports/logging_review/html/tables.py` (`_render_grouping_groups`) | — |
| Report data build | `src/reports/logging_review/html/report_builder.py` | — |
| Group columns | `src/processing/logging_review_html_report.py` (`_build_resolved_group_cols`) | `Tools/RC/combine_intervals.py` (fixed list) |
| Grouping + CV + groups for review | `src/processing/logging_review_html_report.py` (`_grouping_avg_max_interval_and_groups`) | `combine_intervals.py` + `QAQC_07_Grouping_Accuracy_No_charts.py` |
| Summary lines | `src/processing/logging_review_report.py` (`_summarize_grouping_accuracy_by_interval`) | QAQC_07 (different format) |

---

## 6. Next steps

1. **Verify data path:** Run the report with real data; confirm `depth_from_col`/`depth_to_col` and `chem_actual_cols` are non-empty and that `_grouping_avg_max_interval_and_groups` receives them (log or breakpoint).
2. **Fix guards/defaults:** If depth or chemistry is missing, fix resolution or defaults so grouping and CV run when the data supports it.
3. **Optionally align grouping columns:** Extend `_build_resolved_group_cols` to include the same columns as combine_intervals (where present) so the tab matches the old grouping semantics.
4. **Optionally align KPIs:** Add combined-interval length (max_to − min_from per group) and use it for avg/max group interval in the tab if you want parity with the old report.
5. **Add a simple integration test** that asserts grouping tab data is populated for a minimal high-CV dataset.

This gives a clear map from the current Grouping tab to the old standalone code and concrete steps to make the tab functional and, if desired, aligned with the old behaviour.
