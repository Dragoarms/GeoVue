# Logging Review ↔ GeoVue Integration Review

This document reviews how the standalone **Logging Review** codebase (`C:\Users\georg\OneDrive\Home Python\Logging Review`) can be integrated with **GeoVue** (`GeoVue 26_01_27\src`).

---

## 1. What Each Codebase Does

### 1.1 Standalone Logging Review (`Logging Review\`)

| Component | Purpose |
|-----------|---------|
| **NewMain.py** | Standalone Tk app: load path config → load lookup tables (ReadMe.xlsx) → prompt for .txt file → preprocess → filter (date range + loggers) → save filtered CSVs → combine intervals → apply metrics → run QAQC scripts. |
| **Configuration/** | `Folder_Structure_and_File_Path_Config.JSON` (paths), `FilterConfig.json` (date + loggers), `RequiredColumns.json`, `ReadMe.xlsx` (Lookups_LoggedBy_Codes, Lookups_Mineral_Logging_Codes, Lookups_StratSums_Logging). |
| **Tools/RC/** | `combine_intervals.py`, `filters.py`, `metrics_calculation.py`, `PDFUtilityFunctions.py`, etc. (some logic is also in NewMain.py, e.g. `calculate_numerical_metrics`, `open_filter_window`). |
| **QAQC Scripts/RC/** | QAQC_01 Cover Page, QAQC_02 Summary Statistics, QAQC_03 Comment Statistics, QAQC_04 Mineralisation Accuracy, QAQC_05 Profile Zonation, QAQC_06 Fines Accuracy, QAQC_07 Grouping Accuracy. Each has `main(**kwargs)` and produces FPDF reports. |

**Data flow:** `.txt` (tab-separated) → preprocess (required columns, dates, LoggedBy, SAMPLETYPE, StratSum, intervals) → filter by date + loggers → save CSVs per logger → combine intervals (grouping columns) → `calculate_numerical_metrics` (hardness, gangue, zonation from ReadMe.xlsx mineral table) → run QAQC_01–07 with `qaqc_dataframe`, lookups, paths.

### 1.2 GeoVue (`GeoVue 26_01_27\src`)

| Component | Purpose |
|-----------|---------|
| **gui/logging_review_dialog.py** | Lithology classification dialog (BIFf, Compact, etc.) with compartment images, CSV/register integration, scatter plot filtering, undo/redo. Uses `DrillholeDataManager`, `DrillholeDataVisualizer`, `ColorMapManager`, `ValidationRegister`. |
| **gui/logging_qaqc/** | In-app QAQC: `rule_engine`, `validation_register`, `qaqc_orchestrator` (RC metrics, Mahalanobis, Random Forest, logical rules, RC accuracy), `qaqc_pdf_reports` (ReportLab-based "RC Logging Review" PDF), `statistical_analyzer`, `mineral_code_manager` (reads `resources/mineral_codes.json`). |
| **processing/LoggingReviewStep/** | `drillhole_data_manager` (collar + multi-source logging data), `drillhole_data_visualizer`, `color_map_manager`. |
| **processing/DataManager/** | `rc_metrics_store` (RCMetricsCalculator, mineral_codes.json), `register_store` (ValidationRegister). |

**Data flow (Logging Review dialog):** User opens "Logging Review" from main GUI → `LoggingReviewDialog` loads data via `DrillholeDataManager` (CSV/Excel), shows compartment images and classifications, runs in-app QAQC (orchestrator) and can generate RC Logging Review PDF. No "load .txt → filter by date/loggers → combine intervals → run QAQC_01–07" pipeline.

---

## 2. Overlaps and Differences

| Aspect | Logging Review (standalone) | GeoVue |
|--------|----------------------------|--------|
| **RC metrics** | `calculate_numerical_metrics` in NewMain; lookups from ReadMe.xlsx (Mineral_Logging_Codes). | `RCMetricsStore` + `mineral_codes.json`; similar concepts (hardness, gangue, zonation). |
| **Combine intervals** | `combine_intervals.py` + logic in NewMain (grouping columns, completeness_pct, interval categories). | No equivalent in GeoVue; `DrillholeDataManager` does range-based merging for display, not this exact aggregation. |
| **Filter by date/loggers** | `open_filter_window` in NewMain; FilterConfig.json. | Logging Review dialog filters by hole/depth/classification etc., not by "date range + list of loggers" for RC pipeline. |
| **Preprocess .txt** | `preprocess_dataset` in NewMain (required columns, dates, LoggedBy, SAMPLETYPE, StratSum, zero-length intervals). | GeoVue does not load or preprocess this .txt format in the same way. |
| **QAQC reports** | QAQC_01–07: FPDF, per-logger and "All Loggers" (cover, summary stats, comment stats, mineralisation, zonation, fines, grouping). | `qaqc_pdf_reports.RCLoggingReviewReport`: ReportLab, 7-page report (executive summary, mineralisation, gangue, zonation, hardness, flag summary, detailed flags) from `UnifiedAnalysisResult` + metrics_df. |
| **Config** | Folder_Structure_and_File_Path_Config.JSON, RequiredColumns.json, ReadMe.xlsx. | config.json, resources/mineral_codes.json; no ReadMe.xlsx, no RC pipeline paths. |

So: **same domain (RC logging, metrics, QAQC)** but **different entry points and report formats**. The standalone pipeline is "file → filter → combine → metrics → QAQC_01–07"; GeoVue is "images + multi-source data → review + in-app QAQC + one consolidated PDF".

---

## 3. Integration Options

### Option A: "Run RC Logging Pipeline" from GeoVue (recommended for full parity)

**Goal:** User can run the full standalone pipeline (file selection → preprocess → filter → combine intervals → metrics → QAQC_01–07) from inside GeoVue.

**Approach:**

1. **Add a new menu item or dialog** in GeoVue (e.g. "Run RC Logging Pipeline" or "Export & Run Logging Review") that:
   - Prompts for the .txt logging file (or uses a configured path).
   - Uses or replicates the standalone "filter" step (date range + loggers). Either:
     - Embed the filter UI (date + logger list) from NewMain into a GeoVue dialog, or
     - Reuse `FilterConfig.json` and a small config UI in GeoVue.
   - Runs the same steps as NewMain: preprocess → save filtered datasets → combine intervals → apply metrics → run QAQC_01–07.

2. **Where to put the logic:**
   - **Option A1 – Subprocess:** GeoVue calls `NewMain.py` (or a thin CLI wrapper) as a subprocess, passing paths (e.g. path to .txt, output dir, filter config). Minimal code duplication; keeps standalone runnable on its own. GeoVue must know the path to "Logging Review" and its config (e.g. `Folder_Structure_and_File_Path_Config.JSON`).
   - **Option A2 – In-process:** Copy or move the pipeline into GeoVue (e.g. `processing/LoggingReviewStep/rc_pipeline.py` or `processing/RCPipeline/`). Implement or reuse: `preprocess_dataset`, `save_filtered_datasets`, `combine_intervals_for_all_loggers`, `apply_metrics_to_logging_data`, and the QAQC script runner. Point path config to GeoVue's config or a dedicated RC pipeline config.

3. **Config alignment:**
   - GeoVue could read an optional "RC pipeline config" path (e.g. `Folder_Structure_and_File_Path_Config.JSON`) so output directory, script directory, and paths stay consistent with the standalone.
   - Alternatively, add RC pipeline keys to GeoVue's `config.json` (e.g. `rc_pipeline_output_directory`, `rc_pipeline_script_directory`, `rc_lookup_table_path`).

4. **Lookups:** Pipeline expects ReadMe.xlsx (LoggedBy, Mineral_Logging, StratSums). Either:
   - Use the same ReadMe.xlsx (path in config), or
   - Add a small adapter that builds the same dicts from GeoVue's `mineral_codes.json` + a LoggedBy/StratSums source (Excel or JSON) so one source of truth lives in GeoVue.

**Pros:** Full feature parity with standalone; one place (GeoVue) to launch everything.  
**Cons:** Subprocess option requires two codebases/configs; in-process option requires refactoring and possibly duplicating some code.

---

### Option B: Reuse only the standalone QAQC scripts (QAQC_01–07) inside GeoVue

**Goal:** GeoVue prepares the same `qaqc_data` and kwargs that the standalone uses, then calls the existing QAQC_01–07 modules so FPDF reports are identical.

**Approach:**

1. **Data shape:** In GeoVue, after loading and processing logging data:
   - Build `qaqc_data = { "All Loggers": { "date_filtered": df1, "combined_intervals": df2 }, "Logger1": { ... }, ... }` in the same format as `get_qaqc_dataframes()` in NewMain (including "with_metrics" files if the scripts expect them).
2. **Paths and lookups:** Build `path_config`, load `loggedby_lookup_dict`, `mineral_logging_lookup_dict`, `stratsums_lookup_dict` (from ReadMe.xlsx or adapters).
3. **Invoke scripts:** Add the standalone `QAQC Scripts/RC` directory to `sys.path` (or copy it under `src`), then call `import_qaqc_scripts` / `run_qaqc_scripts` with the same `kwargs` the standalone uses.

**Pros:** Same FPDF reports as standalone; no rewrite of QAQC_01–07.  
**Cons:** GeoVue must implement or reuse the preprocessing, filtering, combine-intervals, and metrics steps so that the DataFrames and file names match what the scripts expect. Lookup source (ReadMe.xlsx vs GeoVue resources) must be aligned.

---

### Option C: Unify config and data formats only (no new UI)

**Goal:** Make it easy to run the standalone pipeline on data that came from or is shared with GeoVue (e.g. same column names, same output layout).

**Approach:**

1. **Required columns:** Use the same `RequiredColumns.json` (or a copy under GeoVue's resources) so any .txt or CSV exported/used by GeoVue conforms to what `preprocess_dataset` expects.
2. **Output layout:** If GeoVue ever exports "filtered" or "combined" datasets, use the same naming and folder structure as the standalone (`All Loggers`, per-logger folders, `Date_Filtered_*`, `combined_intervals_output_*`, `*_with_metrics.csv`) so the standalone can run on that output without changes.
3. **Path config:** Document or configure `Folder_Structure_and_File_Path_Config.JSON` so `output_directory` can point to a GeoVue project or shared folder.

**Pros:** Low effort; no new UI; standalone remains the single runner for the full pipeline.  
**Cons:** User still runs two apps (GeoVue for review, standalone for pipeline); no "one click" from GeoVue.

---

### Option D: Move pipeline code into GeoVue and keep one report path

**Goal:** All logic lives in GeoVue; optionally retire the standalone or keep it as a thin launcher.

**Approach:**

1. **Copy/refactor into GeoVue:**
   - `preprocess_dataset`, column remap, validation → e.g. `processing/LoggingReviewStep/rc_preprocess.py` or under `processing/RCPipeline/`.
   - `combine_intervals` (and categorize) → reuse or adapt `Logging Review/Tools/RC/combine_intervals.py` into `processing/LoggingReviewStep/` or `processing/DataManager/`.
   - `calculate_numerical_metrics` → either keep using GeoVue's `RCMetricsStore` and ensure it matches the standalone's output columns, or port the standalone logic into a single module that reads from the same lookup source (ReadMe.xlsx or mineral_codes.json + adapters).
   - Filter UI (date + loggers) → new small dialog in `gui/` that writes `FilterConfig.json` and/or in-memory filter state.
2. **QAQC reports:** Either:
   - Call the standalone QAQC_01–07 from GeoVue (Option B), or
   - Gradually replace them with GeoVue's existing `RCLoggingReviewReport` and extend it to cover the same content (cover page, summary stats, comment stats, etc.) so only ReportLab reports remain.
3. **Config:** RC pipeline paths and lookup paths in GeoVue `config.json` or a dedicated RC config file.

**Pros:** Single codebase; one config; can evolve toward one report style.  
**Cons:** Most work; need to keep behaviour and outputs consistent with current standalone.

---

## 4. Recommended Path (short term)

1. **Option A1 (subprocess)**  
   - Add a GeoVue menu item "Run RC Logging Pipeline" that:
     - Opens a dialog to choose the .txt file (and optionally "Logging Review" root or config path).
     - Launches `NewMain.py` (or a small CLI that runs the same steps) as a subprocess with the chosen file and config.
   - Requires: a stable way to pass "input file" and "config path" to the standalone (e.g. CLI args or a small wrapper that writes a one-off config and runs NewMain). No need to move code yet.

2. **Option C**  
   - Align `RequiredColumns.json` and document that GeoVue and the standalone can share the same column names and, if desired, the same output folder layout so the standalone can be run on GeoVue-related data.

3. **Later (if you want everything in-process)**  
   - Implement Option A2 or D: move preprocessing, combine intervals, and metrics into GeoVue, and either call QAQC_01–07 (Option B) or extend GeoVue's PDF (Option D).

---

## 5. File and Config Mapping

| Standalone (Logging Review) | GeoVue equivalent / action |
|----------------------------|----------------------------|
| `Configuration/Folder_Structure_and_File_Path_Config.JSON` | Add RC pipeline section to `config.json` or a separate rc_pipeline_config.json. |
| `Configuration/RequiredColumns.json` | Copy to `resources/` or reference from config so preprocessing can validate columns. |
| `Configuration/ReadMe.xlsx` (LoggedBy, Mineral, StratSums) | Keep using it for pipeline, or add adapter from `mineral_codes.json` + LoggedBy/StratSums JSON or Excel. |
| `Configuration/FilterConfig.json` | Generated by filter step; GeoVue can write it when running the pipeline (dialog or in-process). |
| `NewMain.py` | Entry point for subprocess; or logic moved into `processing/LoggingReviewStep/` or `processing/RCPipeline/`. |
| `Tools/RC/combine_intervals.py` | Reuse or port in `processing/LoggingReviewStep/` or `processing/DataManager/`. |
| `QAQC Scripts/RC/*.py` | Either keep on disk and call via `sys.path` + `import_qaqc_scripts` / `run_qaqc_scripts`, or copy under `src` (e.g. `processing/LoggingReviewStep/qaqc_scripts/`) and adapt imports. |

---

## 6. Summary

- **Logging Review** = full pipeline: .txt → preprocess → filter (date + loggers) → combine intervals → metrics (from ReadMe.xlsx) → QAQC_01–07 (FPDF).
- **GeoVue** = image/compartment review + in-app QAQC (orchestrator + ReportLab RC report); no equivalent "RC pipeline" yet.
- **Easiest integration:** Add "Run RC Logging Pipeline" in GeoVue that runs the standalone (subprocess) and align config/column names (Option A1 + C).
- **Tighter integration:** Move pipeline and optionally QAQC scripts into GeoVue (Options A2, B, D) and unify config and lookups.

If you tell me which option you prefer (e.g. "subprocess only" vs "move pipeline into GeoVue"), I can outline concrete steps and file changes next (e.g. new menu handler, dialog, and CLI for NewMain).
