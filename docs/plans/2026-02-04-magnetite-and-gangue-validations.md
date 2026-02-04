# Magnetite RC Metric and Gangue Phase Validations Implementation Plan

> **For Claude:** Use subagent-driven-development in this session: one task at a time, spec review then code quality review after each.

**Goal:** (1) Add an RC metric for magnetite/magnetic mineral % derived from mineral_codes.json (magnetic/magnetite codes only, excluding non-magnetic) and use it in the magnetite validation (LOI1000 negative but no magnetite logged). (2) Add logging-detail validations for sulphide_gangue_pct, manganese_gangue_pct, mafics_gangue_pct, magnesium_gangue_pct with assay comparison (same pattern as carbonate).

**Architecture:** MineralCodeManager gains a method to return magnetic/magnetite codes from mineral_codes.json (name contains "magnetic" or "magnetite", exclude "non-magnetic"/"non magnetic"). RCMetricsCalculator sums mineral % for those codes into a new IntervalMetrics field (e.g. magnetite_pct). Report prep already merges RC metrics into merged_df so assay_logger_df can have magnetite_pct and all gangue columns. Magnetite flag: LOI1000 < 0 and (magnetite_pct missing or < threshold) → flag. New gangue flags: same pattern as _flag_carbonate_issue for sulphide (vs S%), manganese (vs Mn%), mafics (vs no single assay proxy; optional or skip), magnesium (vs MgO%).

**Tech Stack:** Python; mineral_codes.json; processing/DataManager/rc_metrics_store.py, column_aliases.py; processing/logging_review_html_report.py; reports/logging_review/html/report_builder.py, tables.py.

---

## Task 1: Magnetite RC metric (mineral_codes + RCMetricsCalculator + IntervalMetrics)

**Files:**
- Modify: `src/resources/mineral_codes.json` (no change unless we add a "magnetic" flag; we use name matching)
- Modify: `src/processing/DataManager/rc_metrics_store.py` (MineralCodeManager, RCMetricsCalculator, IntervalMetrics)

**Steps:**

1. **MineralCodeManager: add get_magnetic_magnetite_codes()**  
   In `rc_metrics_store.py`, add a method that returns a set of code strings where the code's `name` (from loaded JSON) contains "magnetite" or "magnetic" (case-insensitive) and does NOT contain "non-magnetic" or "non magnetic". Use `self._codes` and `info.name`. Return `Set[str]` of uppercase codes.

2. **IntervalMetrics: add magnetite_pct**  
   In the `IntervalMetrics` dataclass add `magnetite_pct: float = 0.0`. In `to_dict()` add `"magnetite_pct": self.magnetite_pct`.

3. **RCMetricsCalculator.calculate_interval_metrics: accumulate magnetite %**  
   Before the minerals loop, get magnetic codes: `magnetic_codes = self.mineral_codes.get_magnetic_magnetite_codes()` (you need to add this to MineralCodeManager first). Initialise `magnetite_pct = 0.0`. In the loop over `(code, raw_pct, col)`, if `code in magnetic_codes`, add `raw_pct * 100` to magnetite_pct (so it's a percentage 0–100). After the loop, set `metrics.magnetite_pct = round(magnetite_pct, 2)`.

4. **Verify**  
   Run existing RC metrics tests if any; or manually load mineral codes, call `get_magnetic_magnetite_codes()` and assert MT, MBH, MBM, MBB, MI are in the set and NBB, NBH, NBM are not.

**Commit:** `feat(rc_metrics): add magnetite_pct from mineral_codes (magnetic/magnetite codes only)`

---

## Task 2: Magnetite validation (LOI1000 negative but no magnetite logged) + report wiring

**Files:**
- Modify: `src/processing/logging_review_html_report.py` (_flag_magnetite_issue, _has_magnetite_logged)
- Modify: `src/reports/logging_review/html/report_builder.py` (pass magnetite_pct into magnetite intervals; optional "Magnetite logged?" in table)
- Modify: `src/reports/logging_review/html/tables.py` (LOGGING_DETAIL_CHEM_COLUMNS for magnetite: include Magnetite% or "Magnetite logged?" if desired)

**Steps:**

1. **Change magnetite flag logic**  
   Per DATA_VALIDATIONS_AND_FLAGS.md: flag when **LOI1000 is negative but no magnetite logged**. So: if LOI1000 is missing or ≥ 0 → return None. If LOI1000 < 0, then if magnetite was logged (either via existing _has_magnetite_logged from strat/min columns OR from RC metric magnetite_pct when available) → return None; else return `"LOI1000 negative but no magnetite logged (MT/MBH/MBM)"`. Prefer using RC metric magnetite_pct when present (e.g. column resolver get "magnetite_pct") and threshold e.g. > 0.5% or > 0; if column not present, fall back to _has_magnetite_logged(row, resolver).

2. **Column alias for magnetite_pct**  
   In `column_aliases.py`, add alias for "magnetite_pct" so resolver.get("magnetite_pct") finds the merged RC metric column (e.g. ["magnetite_pct", "magnetic_pct"]).

3. **Report builder: magnetite intervals**  
   When building magnetite_intervals, include assay/geochem that has magnetite_pct (from row) so the table can show "Magnetite %" or "Magnetite logged?". Add to the assay dict passed to magnetite: `magnetite_pct` from resolver.get("magnetite_pct") and row.

4. **Tables: magnetite columns**  
   Ensure LOGGING_DETAIL_CHEM_COLUMNS for "magnetite" includes Fe%, LOI1000%, and optionally "Magnetite %" (assay key magnetite_pct) so the evidence table shows magnetite logged %.

**Commit:** `feat(report): magnetite validation LOI1000 negative and no magnetite logged; show Magnetite % in table`

---

## Task 3: Gangue validations (sulphide, manganese, mafics, magnesium)

**Files:**
- Modify: `src/processing/DataManager/column_aliases.py` (add aliases for sulphide_gangue_pct, manganese_gangue_pct, mafics_gangue_pct, magnesium_gangue_pct)
- Modify: `src/processing/logging_review_html_report.py` (add _flag_sulphide_issue, _flag_manganese_issue, _flag_mafics_issue, _flag_magnesium_issue; same pattern as carbonate: compare logged gangue % to assay proxy)
- Modify: `src/reports/logging_review/html/report_builder.py` (build interval lists for sulphide, manganese, mafics, magnesium; add to logging_detail_issue_types and intervals_for_review)
- Modify: `src/reports/logging_review/html/tables.py` (LOGGING_DETAIL_CHEM_COLUMNS for each new type; optional new table_ids)
- Modify: `src/reports/logging_review/html/types.py` (logging_detail: add sulphide, manganese, mafics, magnesium lists if using TypedDict)

**Steps:**

1. **Column aliases**  
   Add entries: `sulphide_gangue_pct`, `manganese_gangue_pct`, `mafics_gangue_pct`, `magnesium_gangue_pct` with plausible column name variants (e.g. sulphide_gangue, manganese_gangue, mafics_gangue, magnesium_gangue; match names that RC metrics output).

2. **Flag functions**  
   For each gangue type, implement _flag_X_issue(row, resolver):
   - **Sulphide:** sulphide_gangue_pct vs S% (s_pct). High logged / low assay or low logged / high assay (thresholds similar to carbonate: e.g. >15% vs <2%, <2% vs >10%).
   - **Manganese:** manganese_gangue_pct vs Mn% (mn_pct). Same pattern.
   - **Mafics:** mafics_gangue_pct vs assay proxy (no single standard; could use Fe+MgO or omit assay comparison and only flag e.g. very high mafics with no Fe; or defer to "logged vs expected" later). Simplest: compare to MgO or Fe as proxy; if no proxy, skip flag or return None.
   - **Magnesium:** magnesium_gangue_pct vs MgO% (mgo_pct). Same pattern as carbonate.

   Use _significance_for_logging_detail_issue(issue, "sulphide") etc.; add issue_type for each in the helper if needed (e.g. always Low for these gangue checks or by magnitude).

3. **Report builder**  
   In the loop where fines/magnetite/goethite/carbonate intervals are built, add four more: sulphide_intervals, manganese_intervals, mafics_intervals, magnesium_intervals. Resolve columns for each gangue_pct and assay proxy (s_pct, mn_pct, mgo_pct; mafics proxy if any). Append to logging_detail in intervals_for_review. Add four entries to logging_detail_issue_types (key, label_en, label_fr, rules_en, rules_fr).

4. **Tables and types**  
   Add table_id entries in LOGGING_DETAIL_CHEM_COLUMNS for sulphide, manganese, mafics, magnesium (columns: relevant gangue % and assay proxy %). Add new sub-tabs in logging detail tab (report_renderer / tabs use logging_detail_issue_types so they will appear automatically if keys are in the dict). Update types.py IntervalsForReview if the structure expects explicit keys.

5. **Data coordinator / prep**  
   Ensure merged_df gets sulphide_gangue_pct, manganese_gangue_pct, mafics_gangue_pct, magnesium_gangue_pct from RC metrics merge (they already come from IntervalMetrics.to_dict() and get_metrics_dataframe(); verify column names match).

**Commit:** `feat(report): add sulphide, manganese, mafics, magnesium gangue validations and evidence tables`

---

## Execution order

1. Task 1 (magnetite_pct in RC metrics).
2. Task 2 (magnetite validation and report).
3. Task 3 (four gangue validations and report/tables).

After all tasks: update DATA_VALIDATIONS_AND_FLAGS.md if any behaviour is refined (e.g. mafics proxy).
