# Logging Review Report QA Checklist

## Automated
- `python -m unittest src/processing/tests/test_logging_review_report.py`
- `python -m unittest src/gui/tests/test_logging_review_report_dialog.py`

## Data Validation
- Verify `exassay` and `exgeologyRC` are loaded and non-empty.
- Confirm `Strat`, `LoggedBy`/`LoggedBy_D`, and `drilldate` columns resolve.
- Confirm `hole_id`, `depth_to` resolution for image lookups.
- Check RC metrics store is loaded (for gangue-related summaries).

## Report Generation
- Generate report with a small date range and 1–2 loggers.
- Confirm one PDF per logger in output folder.
- Validate cover page metadata (logger, date range, generation date).
- Confirm summary stats (interval count, unique holes, strat count).
- Confirm outlier index page lists top N entries.
- For 3–5 outliers, verify boxplots show outlier point.
- Confirm image embedding when a compartment image is available.

## Outlier Logic Spot Checks
- Pick a known strat with stable assays: confirm low outlier scores.
- Pick a known geochemically anomalous interval: confirm it appears in top N.
- Verify outlier reasons list expected elements and directions.

## Fines / Grouping Summaries
- If RC metrics are available, verify fines accuracy summary is populated.
- Check grouping CV summary lines for Fe/SiO2/Al2O3.

