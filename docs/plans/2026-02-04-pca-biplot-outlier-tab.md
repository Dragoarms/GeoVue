# PCA Biplot for Outlier Tab — Implementation Plan

> **For Claude:** Use subagent-driven-development to implement; outlier threshold/scoring (Tasks 1–6) are already done per 2026-02-04-outlier-fix-verification-review.

**Goal:** Add a PCA biplot (CLR space) to the logging review outlier tab so users get a multivariate view of geochemical clusters and which elements separate strat units; keep Fe–SiO2 scatter as secondary.

**Architecture:** New `_plotly_pca_biplot_json` in `charts.py` (CLR + SVD, loadings, strat-colored points, outliers as red X). Report builder generates and passes PCA data; outlier tab renders PCA panel first, then Fe vs SiO2 panel.

**Tech Stack:** Python, numpy, pandas, Plotly JSON. Files: `src/reports/logging_review/html/charts.py`, `report_builder.py`, `types.py`, `tabs/outliers.py`.

---

## Task 7: Add _plotly_pca_biplot_json and numpy import (charts.py)

**Files:** Modify `src/reports/logging_review/html/charts.py`.

- Add `import numpy as np` (required for CLR/SVD in PCA).
- Insert `_plotly_pca_biplot_json` **before** `_plotly_outlier_scatter_json` with full implementation: CLR transform, SVD, PC1/PC2, loadings, strat traces, outlier trace, loading arrows + labels, layout with variance explained.

---

## Task 8: Import PCA function (report_builder.py)

**Files:** Modify `src/reports/logging_review/html/report_builder.py`.

- Extend charts import: `from .charts import _plotly_outlier_box_json, _plotly_outlier_scatter_json, _plotly_pca_biplot_json`.

---

## Task 9: Generate PCA biplot data (report_builder.py)

**Files:** Modify `src/reports/logging_review/html/report_builder.py`.

- After `_plotly_outlier_scatter_json(...)` call, add:
  `outlier_pca_data, outlier_pca_layout = _plotly_pca_biplot_json(assay_logger_df, strat_col, chem_actual_cols)`.

---

## Task 10: Add PCA to report_data (report_builder.py)

**Files:** Modify `src/reports/logging_review/html/report_builder.py`.

- In `report_data` dict, after `outlier_scatter_data`/`outlier_scatter_layout`, add:
  `"outlier_pca_data": outlier_pca_data`, `"outlier_pca_layout": outlier_pca_layout`.

---

## Task 11: Add PCA fields to TypedDict (types.py)

**Files:** Modify `src/reports/logging_review/html/types.py`.

- In `ReportData`, after `outlier_scatter_data`/`outlier_scatter_layout`, add:
  `outlier_pca_data: Any`, `outlier_pca_layout: Any`.

---

## Task 12: Get PCA data in outlier tab (tabs/outliers.py)

**Files:** Modify `src/reports/logging_review/html/tabs/outliers.py`.

- After reading `outlier_scatter_data`/`outlier_scatter_layout`, add:
  `outlier_pca_data = report.get("outlier_pca_data", "[]")`, `outlier_pca_layout = report.get("outlier_pca_layout", "{}")`.

---

## Task 13: Render PCA biplot and Fe vs SiO2 in outlier tab HTML (tabs/outliers.py)

**Files:** Modify `src/reports/logging_review/html/tabs/outliers.py`.

- Replace the single “Clusters geochimiques et positions des anomalies” panel with:
  1. **PCA panel:** Title “Clusters geochimiques (PCA multivarié)”, description paragraph, `id="outlier-pca-plot"`, data/layout from `outlier_pca_data`/`outlier_pca_layout`.
  2. **Fe vs SiO2 panel:** Title “Fe vs SiO2 (vue simplifiée)”, existing `id="outlier-scatter-plot"` with scatter data/layout.

---

## Verification

- Build report; open outlier tab: PCA biplot appears first (strat colors, loadings, red X outliers), then Fe vs SiO2 scatter.
- No new linter errors; existing outlier tests still pass.
