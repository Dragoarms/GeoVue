# Logging Review Report Status

## Done

- Phase A.5: Deduplicated with DataManager (project_code, easting, northing, depth_from/depth_to aliases)
- Phase A: Extracted CSS (~875 lines) and JS (~310 lines) into `src/reports/logging_review/html/assets/`
- Phase B: Extracted utils (_safe_str, _safe_float, _format_metric)
- Phase C: Extracted charts, tables, images, collar_map into separate modules
- Setup: Created `src/reports/` package structure

## Deferred

- Phase D: Extract data layer into `reports/logging_review/data/`
- Phase E: Split by tab (one at a time with diff check)
- Phase F: Thin entry files (report_builder.py, report_renderer.py)
- Phase G: Backward-compat wrappers in `processing/`
- Phase H: TypedDict for ReportData (optional)

## Obsolete

- `docs/Logging_Review_Integration_Review.md` - Moved to `docs/archive/`

## Related Documents

- `docs/plans/logging-review-report-atomic-tasks.md` - Detailed atomic task list
- `docs/plans/2026-02-01-html-report-verification.md` - Verification checklist (Status: Implemented)
- `docs/Logging_Review_Report_QA_Checklist.md` - QA checklist (still valid)
