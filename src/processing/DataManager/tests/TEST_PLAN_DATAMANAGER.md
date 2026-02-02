# DataManager Test & Validation Plan

This document identifies useful tests and validations that can be created for the DataManager package. It is organized by module and by test type (unit, integration, validation).

**Existing coverage:** `test_rc_metrics_store.py` (RCMetricsStore, MineralCodeManager, RCMetricsCalculator, IntervalMetrics).

---

## 1. keys.py — ImageKey & FilenameParser

### Unit tests

| Test | Purpose |
|------|--------|
| **ImageKey validation** | `hole_id` empty → ValueError; `depth_to` negative → ValueError |
| **ImageKey normalization** | `hole_id_upper`, `depth_to_int`, `to_tuple()`, `to_base_tuple()` return expected values |
| **ImageKey equality/hash** | Same (hole_id, depth_to, moisture) are equal; usable as dict/set keys |
| **ImageKey `matches_interval`** | Case-insensitive hole_id, int depth_to match; edge cases (float vs int) |
| **FilenameParser compartment** | `BA0001_CC_045_Wet.png` → hole_id=BA0001, depth_to=45, moisture=Wet; no moisture → None |
| **FilenameParser compartment invalid** | Wrong prefix, bad depth, unknown extension → None |
| **FilenameParser original** | `BA0001_20-40m.jpg` → hole_id, depth_from=20, depth_to=40 |
| **FilenameParser simple depth** | `BA0001_045.png` → ImageKey(BA0001, 45) via `create_key_from_filename` |
| **create_key** | Uppercases hole_id, capitalizes moisture_status |
| **get_parser()** | Returns same instance when called without args; new instance with custom `valid_prefixes` |

### Validation ideas

- ImageKey with non-"Wet"/"Dry" moisture_status logs warning but does not raise (document current behavior).

---

## 2. schema.py — DataSourceSchema, ColumnSchema, SchemaInferrer

### Unit tests

| Test | Purpose |
|------|--------|
| **ColumnSchema default display_name** | `display_name is None` → becomes `source_name` |
| **ColumnSchema apply_null_handling** | KEEP, FILL_ZERO, FILL_EMPTY, FILL_VALUE produce correct Series |
| **ColumnSchema validate_value** | Numeric min/max, categorical allowed values, invalid → False |
| **ColumnSchema to_dict / from_dict** | Round-trip preserves enums as strings |
| **DataSourceSchema key columns** | INTERVAL gets hole_id, from, to in `columns`; COLLAR only hole_id |
| **DataSourceSchema to_dict / from_dict** | Round-trip with dataset_type, columns |
| **DataSourceSchema get_column** | Case-insensitive lookup; missing → None |
| **SchemaInferrer HOLE_ID_PATTERNS** | Correct column chosen for holeid, hole_id, BHID, etc. |
| **SchemaInferrer FROM/TO_PATTERNS** | sampfrom/sampto, depth_from/depth_to, geolfrom/geolto detected |
| **SchemaInferrer dataset type** | From/to present → INTERVAL; collar indicators → COLLAR; azimuth+dip+depth → SURVEY |
| **SchemaInferrer column type** | Numeric ratio ≥ threshold → NUMERIC; low cardinality → CATEGORICAL; else TEXT |
| **SchemaInferrer numeric with blanks** | Column with many blanks but numeric non-blanks → still NUMERIC (non-blank ratio) |
| **infer_schema** | Full DataFrame → DataSourceSchema with expected dataset_type and column types |

### Validation ideas

- Schema with missing hole_id / from / to for INTERVAL → error or clear default.
- Inferrer with no hole column → fallback to "holeid" and log warning (assert behavior).

---

## 3. image_index.py — ImageIndex

### Unit tests (with temp dirs)

| Test | Purpose |
|------|--------|
| **add_compartment_folder** | Existing path added; non-existing path logs warning, not added |
| **build empty** | No folders → empty index, no error |
| **build with sample files** | Temp dir with `BA0001_CC_045_Wet.png`, `BA0001_CC_046.png` → 2 entries, 1 hole |
| **get / get_path** | Key present → ImageInfo/path; key missing → None |
| **get_keys_for_hole** | Sorted by depth; case-insensitive hole_id |
| **Duplicate key** | Same key in two files → later file wins (document current behavior) |
| **Original linking** | Compartment depth_to within original range → original_path set |
| **get_depth_range** | Returns (min_depth, max_depth) for hole; missing hole → None |
| **filter_by_hole** | Only listed hole_ids returned, sorted |
| **get_stats** | is_built, images_indexed, unique_holes, build_time populated after build |
| **clear** | After clear, len() == 0, is_built == False |

### Integration

- Build with real folder structure (e.g. project/test_data) if available; assert image_count and hole_count.

---

## 4. geological_store.py — IndexedDataSource, GeologicalStore

### Unit tests (IndexedDataSource)

| Test | Purpose |
|------|--------|
| **load missing file** | Returns False, is_loaded False |
| **load CSV encoding** | UTF-8; if needed latin-1/cp1252 fallback (or mock read_csv) |
| **load INTERVAL** | CSV with holeid, sampfrom, sampto → index on (hole_upper, depth_int); get_row(key) returns row |
| **load COLLAR** | CSV with holeid only → index on hole; get_row by hole (key.depth_to unused for collar) |
| **get_row** | Valid key → dict without leading-underscore cols; NaN → None |
| **get_row missing key** | KeyError path → None |
| **get_value** | get_value(key, col) returns scalar; missing col/key → None |
| **query no filters** | return_keys=True → set of (hole_id, depth_to); return_keys=False → full df copy |
| **query with filters** | operator =, !=, >, <, between, in, contains; mask applied correctly |
| **Column names** | Loaded df columns lowercased |

### Unit tests (GeologicalStore)

| Test | Purpose |
|------|--------|
| **add_source** | Path exists → source added; path missing → skipped or error (document) |
| **load_all** | Multiple sources; all loaded, list_sources returns names |
| **get_source** | By name → IndexedDataSource; missing → None |
| **get_row** | Delegates to correct source (e.g. by key); which source is used when multiple have same key (first? priority?) |
| **get_rows_for_hole** | Returns dict source_name -> DataFrame for that hole |
| **get_data_sources** | Returns dict name -> IndexedDataSource for GeologicalStore (used by sanitizer) |
| **get_available_columns** | Returns dict source_name -> list of (col_name, DataType) |
| **get_column_values** | Unique values for a column across sources (or per source—check API) |

### Integration

- Load a small CSV from `test_output/merged_sample.csv` (or fixture); assert row count, get_row(known key), get_rows_for_hole.

### Validation ideas

- CSV with no holeid column → load fails with clear error.
- INTERVAL CSV without to_column → load fails.

---

## 5. register_store.py — RegisterStore, ReviewMetadata, ImageProperties

### Unit tests (with mocked JSONRegisterManager)

| Test | Purpose |
|------|--------|
| **get_review_metadata no manager** | Returns empty ReviewMetadata (or default) when manager is None |
| **get_review_metadata with mock** | Mock manager returns review for key → classification, tags, consensus, agreement set |
| **get_image_properties** | Mock manager returns hex props → wet_hex, dry_hex, combined_hex |
| **refresh_key** | Clears cache for that key; next get refetches |
| **invalidate_cache** | All cache cleared |
| **ReviewMetadata.to_dict** | All fields present, empty strings for None where applicable |
| **ImageProperties.to_dict** | wet_hex, dry_hex, combined_hex, has_wet, has_dry |

### Integration

- With real JSON register (if test data exists): build_cache or lazy load, then get_review_metadata for known key.

---

## 6. data_coordinator.py — DataCoordinator, CompartmentData

### Unit tests (with mocks)

| Test | Purpose |
|------|--------|
| **create_coordinator** | Factory returns DataCoordinator; optional config_manager, file_manager |
| **initialize empty** | compartment_folders=[] and no csv/json → image index built, geological/register empty; is_initialized True |
| **initialize with folders/csv** | Mock or temp: add compartment folder + CSV path → after initialize, get_unique_holes, image_count, total_rows consistent |
| **get_image_data missing key** | Key not in image index → None |
| **get_image_data present** | Key in index; mock register + geological → CompartmentData has image_path, classification, csv_data |
| **get_image_data_dict** | Same as get_image_data but returns dict via to_dict(); missing key → {} |
| **get_image_path / get_original_path** | Delegate to image_index; None when key missing |
| **get_keys_for_hole / get_data_for_hole** | Delegate to image_index; get_data_for_hole returns list of CompartmentData |
| **get_filtered_keys** | hole_ids, classified_only, unclassified_only, depth_min/max, moisture_status, classification, tags; combine with mocked register |
| **count_by_classification** | With mocked register metadata → dict classification -> count |
| **build_dataframe** | List of keys → DataFrame with expected columns; include_tags True adds tag columns from tag_definitions |
| **build_dataframe_for_hole** | Same as build_dataframe for get_keys_for_hole(hole_id) |
| **get_available_columns / get_column_values** | Delegate to geological_store |
| **refresh_image / refresh_hole / invalidate_caches** | Delegate to register_store; no exception |
| **get_collar_data** | Geological store has excollar/collar/collars source → DataFrame with standardized cols (x, y, z, holeid, etc.) |
| **get_collar_data no source** | No collar source → empty DataFrame |
| **get_hole_intervals** | source_names None → use default list; returns first matching source DataFrame for hole |
| **get_stats** | is_initialized, image_index stats, geological_store stats, register_store stats |
| **CompartmentData.to_dict** | All fields including csv_data and tag_ columns for tags in .tags |

### RC metrics (DataCoordinator)

| Test | Purpose |
|------|--------|
| **has_rc_metrics / rc_metrics_store** | Before compute → False/None; after compute_from_dataframe → True and get_metrics works |
| **get_rc_metrics / get_rc_metrics_dict** | Existing interval → IntervalMetrics / dict; missing → None / {} |
| **get_rc_metrics_dataframe / get_rc_metrics_statistics** | After compute → non-empty df and stats |
| **clear_rc_metrics** | Clears store; has_rc_metrics False |

### Integration

- **Full init:** Real or fixture compartment folder + CSV + (optional) register → initialize() then get_image_data, get_filtered_keys, build_dataframe for one hole. Assert no crash and shapes/counts.
- **Sanitizer and RC in init:** Geological store with mineral columns → after initialize(), sanitization report and RC metrics computed (and optionally assert counts).

---

## 7. data_sanitizer.py — DataSanitizer, ValidationRule, Report

### Unit tests

| Test | Purpose |
|------|--------|
| **DataIssue / SanitizationReport** | add_issue, get_issues_by_severity, get_issues_by_source, summary(); has_critical, has_errors, error_count, warning_count |
| **DataSanitizer sanitize_sources** | GeologicalStore with 2 sources sharing a column name → duplicate issue; auto_fix → duplicates_resolved |
| **DataSanitizer duplicate resolution** | PREFIX_SOURCE / SUFFIX_SOURCE / RENAME_DUPLICATES → get_qualified_name returns expected |
| **DataSanitizer type mismatch** | Same column name in two sources, different inferred types → type_mismatch issue |
| **DataSanitizer missing key** | Source without hole_id_column → missing_key ERROR; INTERVAL without from/to → missing_key ERROR |
| **DataSanitizer hole consistency** | Two sources, one missing some hole IDs → missing_holes INFO (and sample list) |
| **DataSanitizer high null rate** | Column with ≥90% nulls → high_nulls INFO |
| **DepthOrderRule** | Interval df with from >= to → invalid_depth_order ERROR |
| **NegativeDepthRule** | from/to/depth column with negative values → negative_depth WARNING |
| **PercentageRangeRule** | Column name with _pct and values outside 0–100 → percentage_range WARNING |
| **run_validation_rules** | Default rules run; custom rules list used when provided |
| **sanitize_geological_store** | Convenience function returns SanitizationReport |
| **log_sanitization_report** | No exception; log level by severity |

### Integration

- Create GeologicalStore with 2–3 small CSVs (some duplicate column names, one with from>=to row); run sanitize_geological_store and assert report.issues and report.duplicates_found/type_mismatches.

---

## 8. interval_merger.py — IntervalMerger, MergeStats

### Unit tests

| Test | Purpose |
|------|--------|
| **_detect_key_columns** | DataFrame with holeid, sampfrom, sampto → hole_id, from, to detected |
| **merge_sources** | Primary and secondary with overlapping intervals → merged df; overlapping interval gets correct secondary value |
| **MergeStats** | total_primary_rows, rows_matched, rows_unmatched, match_rate_pct |
| **No overlap** | Primary interval with no overlapping secondary → NaN or default for merged cols |
| **Multiple secondary sources** | Columns from several secondaries; precedence or naming (e.g. suffix) as designed |

### Integration

- GeologicalStore with exassay + exgeologyRC (or fixture); IntervalMerger.merge_sources → assert row count and that key columns + merged columns present.

---

## 9. column_aliases.py — ColumnResolver, COLUMN_ALIASES

### Unit tests

| Test | Purpose |
|------|--------|
| **ColumnResolver get** | DataFrame with fe_pct_best → get("fe_pct") returns "fe_pct_best" (or first match) |
| **ColumnResolver get missing** | No matching column → None |
| **ColumnResolver case insensitivity** | HOLEID vs holeid both resolve |
| **COLUMN_ALIASES coverage** | hole_id, depth_from, depth_to, fe_pct, sio2_pct, lithology, etc. have expected aliases |

---

## 10. color_map_store.py & image_index (extra)

- **ColorMapStore:** If it loads from config or files, test with mock config; get color map by name returns expected structure.
- **ImageIndex:** Already listed above; add any edge case for original linking (e.g. compartment exactly on boundary of original range).

---

## 11. Cross-module integration tests

| Test | Purpose |
|------|--------|
| **Keys → Schema → GeologicalStore** | ImageKey.to_base_tuple() matches index built from schema (hole_id, depth_to); get_row(key) works |
| **Coordinator init → sanitizer** | After initialize(), sanitize_geological_store was run and report logged (or assert report.issues) |
| **Coordinator init → RC metrics** | Geological source with min_*_pct columns → RC metrics computed and get_rc_metrics works |
| **Filtering pipeline** | get_filtered_keys(classified_only=True) then build_dataframe(keys) → DataFrame rows consistent with filter |
| **Collar + intervals** | get_collar_data() and get_hole_intervals() for same hole_id; collar has one row, intervals have multiple |

---

## 12. Test data and fixtures

- **Fixtures:** Small CSVs in `tests/fixtures/` (e.g. interval CSV with holeid, sampfrom, sampto, fe_pct; collar CSV; one with from>=to for sanitizer).
- **Image fixtures:** Temp directory with 5–10 compartment filenames (e.g. BA0001_CC_045_Wet.png).
- **Register:** Optional JSON or mock for RegisterStore/DataCoordinator.
- **pytest markers:** Use `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow` for folder-based or full init tests.

---

## 13. Suggested implementation order

1. **keys.py** — Pure logic, no I/O; unblocks many other tests.
2. **schema.py** — ColumnSchema/DataSourceSchema/SchemaInferrer; used by geological_store and sanitizer.
3. **data_sanitizer.py** — Uses GeologicalStore.get_data_sources(); can use in-memory or small CSV fixtures.
4. **geological_store.py** — IndexedDataSource load/get_row/query; then GeologicalStore add_source/load_all/get_row/get_rows_for_hole/get_data_sources.
5. **image_index.py** — Temp dir with sample filenames.
6. **register_store.py** — Mock JSON manager.
7. **data_coordinator.py** — With mocks first; then one integration test with real fixtures.
8. **interval_merger.py** — With GeologicalStore or raw DataFrames.
9. **column_aliases.py** — Quick unit tests.

This order builds from low-level keys and schema up to coordinator and integration, and reuses existing `test_rc_metrics_store.py` style (unittest or pytest).
