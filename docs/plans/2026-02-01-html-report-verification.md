**Status: Implemented**

# HTML Logging Review Report – Verification Checklist

Verification against `2026-02-01-html-report-graph-migration.md`.  
**Status:** All requested changes implemented except optional chart (Outlier ternary/contour).

---

## 1. Overview Tab

| Requirement | Implementation | Status |
|-------------|----------------|--------|
| KPIs: Holes, Prospects, Total Depth, Date range | Unique holes, Strat codes, Total depth (m), Date range in header-meta | Done |
| Assays received (exassay row count) | `assay_received_count` from `_compute_assay_received_outstanding`; KPI "Essais recus" | Done |
| Outstanding (exgeologyRC intervals with no matching exassay) | `assay_outstanding_count`; KPI "En attente" | Done |
| Average logging interval (m) | `average_logging_interval_m` from `_compute_avg_logging_interval`; KPI "Intervalle moyen (m)" | Done |
| No "assay intervals vs assay intervals" loading bar | Not present; replaced by assay received/outstanding counts | Done |
| Strat code population bar (user's data only) | `strat_code_list`; bar chart in "Population des codes strat" panel | Done |
| Map: user + team holes | `_render_map(report["map"])`; user (logger) vs team points; legend | Done |
| No team comparison KPIs on Overview | No `comparison-grid` in Overview section | Done |
| Comment ratio: (intervals with comment) / (total logging intervals), NULL = no comment | `comment_stats_logging`; KPI "Intervalles avec commentaire (%)"; target >20% noted | Done |
| Comparison rule at bottom of tab | `info-box` with "Regle de comparaison" at end of Overview | Done |

---

## 2. Comments Tab

| Requirement | Implementation | Status |
|-------------|----------------|--------|
| Word cloud (same as QAQC03) | `_extract_comment_wordcloud`; `_render_wordcloud(report["wordcloud"])` | Done |
| Bar: "No Comment" vs "Comment" counts | `comment_stats_logging` (total_rows, rows_with_comment, rows_without_comment); bar rows "Sans commentaire" / "Avec commentaire" | Done |
| Total rows = logging intervals; NULL/missing = No Comment | `_comment_stats_logging_intervals(logging_logger_df)`; NULL/missing counted as no comment | Done |
| Ratio with comments high (>20%) | Text: "Taux avec commentaire: X% (objectif >20%)." | Done |

---

## 3. Mineralisation Tab

| Requirement | Implementation | Status |
|-------------|----------------|--------|
| Logging Accuracy (Match/Mismatch/Pending Assays) from Fe_pct_BEST, SiO2_pct_BEST, Al2O3_pct_BEST, Total Gangue Logged | `_add_mineralisation_accuracy_column` (QAQC04 logic); ColumnResolver for fe_pct, sio2_pct, al2o3_pct, total_gangue_pct | Done |
| Pie: Overall accuracy (Match/Mismatch/Pending) | `_plotly_pie_json`; user pie + team pie in "Vos donnees" / "Tous les loggers" | Done |
| Stacked bar by quarter (Match/Mismatch/Pending %) | `_group_mineralisation_by_quarter`; `_plotly_stacked_bar_json`; user + team quarterly bars | Done |
| User's charts + Team's charts side by side | Two panels: "Vos donnees" (pie + bar), "Tous les loggers" (pie + bar) | Done |
| Plotly | Plotly.js CDN; `initPlotlyCharts()`; data/layout in data-plotly-* attributes | Done |

---

## 4. Zonation Tab

| Requirement | Implementation | Status |
|-------------|----------------|--------|
| Zonation categories (Un, Le, De, Hy, Pr) | `_check_profile_zonation_and_analyze`; zonation_categories = ["Un", "Le", "De", "Hy", "Pr"] | Done |
| Clustered bar: Correctly vs Incorrectly logged by category | `_plotly_zonation_bar_json`; correct_counts, mismatch_counts; "Precision par categorie" | Done |
| Attribution: per category "Should be X" counts | `mismatch_attribution`; table "Logue comme | Devrait etre | Nombre" | Done |
| QAQC05 logic | `_resolve_zonation_columns` (BestProfileZonation_D, Total Gangue, De/Hy/Pr % Logged); gangue ranges and dominant checks | Done |

---

## 5. Fines Tab

| Requirement | Implementation | Status |
|-------------|----------------|--------|
| Keep likely-issue column and intervals list | `_flag_fines_issue`; fines_intervals with issue; `_render_fines_intervals_table` | Done |
| Larger, landscape images | `.fines-image`, `.fines-image-cell` (280px width); `.fines-placeholder` | Done |
| Geochem (Fe, Al2O3, SiO2, P) next to each image | `geochem` per fines_intervals item; column "Geochimie (Fe, Al2O3, SiO2, P)" in table | Done |
| Flag rate explanation | info-box "Taux de drapeaux" with FR/EN text (proportion where geochem suggests issue) | Done |
| Fix all-loggers comparison | `comparisons["fines_rate"]` with median_project, median_all; `_render_comparison_block` | Done |

---

## 6. Grouping Tab

| Requirement | Implementation | Status |
|-------------|----------------|--------|
| KPI: Average group interval (m) | `grouping_kpis.avg_group_interval_m`; KPI "Intervalle moyen (m)" | Done |
| KPI: Max group interval (m) | `grouping_kpis.max_group_interval_m`; KPI "Intervalle max (m)" | Done |
| CV analysis: Fe, SiO2, Al2O3, MgO, CaO, S only | `GROUPING_CV_ELEMENTS = ["fe_pct", "sio2_pct", "al2o3_pct", "mgo_pct", "cao_pct", "s_pct"]` in `_grouping_avg_max_interval_and_groups` | Done |
| Intervals for review = groups with highest CV (not single metre) | `grouping_groups_for_review` = list of groups (group_key, cv_max, intervals); `_build_grouping_groups_with_images` | Done |
| Each group: each image in group + geochem | `_render_grouping_groups`; per group table with Hole, Depth, Geochem, Image per interval | Done |

---

## 7. Outlier Tab

| Requirement | Implementation | Status |
|-------------|----------------|--------|
| KPI: Total intervals likely misclassified | `outlier_kpis.total_misclassified`; KPI "Intervalles probablement mal classes" | Done |
| Most common errors (e.g. "Classified as X but actually Y") | `common_errors` from Counter(recorded_as); "Erreurs les plus frequentes" list | Done |
| Top outliers table: "Recorded as" (strat) and "Most likely" | `_render_outlier_table`: columns Enregistre comme, Plus probable, Raison, Image | Done |
| Explanation "because of Z" (geochem/distance), not generic | `reason` = item.get("reason") or item.get("outlier_reason"); `_build_outlier_intervals_with_images` uses reason as issue | Done |
| Ternary/contour + outlier positions | Not implemented (optional per plan; table + common errors cover core requirement) | Deferred |

---

## 8. Data Contract

| Contract Key | Provided In report_data | Status |
|--------------|-------------------------|--------|
| Overview: assay_received_count, assay_outstanding_count, average_logging_interval_m | `overview` | Done |
| Overview: strat_code_counts (dict or list for bar) | `overview.strat_code_list` | Done |
| Overview: map_points (user + team) | `map` (points, has_coords) | Done |
| Overview: summary (holes, total_depth, etc.) | `summary` | Done |
| Overview: comment_ratio_pct (NULLs = no comment) | `overview.comment_ratio_pct_logging` | Done |
| Overview: No comparisons | No comparison block in Overview section | Done |
| Comments: wordcloud_data, total_rows, rows_with_comment, rows_without_comment | `wordcloud`, `comment_stats_logging` | Done |
| Mineralisation: accuracy_counts, quarterly_summary; same for team | `mineralisation.accuracy_counts`, `quarterly_user`, `accuracy_counts_team`, `quarterly_team` | Done |
| Zonation: correct_counts, mismatch_counts, mismatch_attribution | `profile_zonation` | Done |
| Fines: geochem per interval, flag rate explanation, team comparison | geochem in fines_intervals; info-box; comparisons.fines_rate | Done |
| Grouping: avg_group_interval_m, max_group_interval_m, groups (key, CV, list of image+geochem) | `grouping_kpis`; `intervals_for_review["grouping"]` | Done |
| Outlier: total_misclassified, common_errors, outlier_rows with recorded_as, likely, reason | `outlier_kpis`; outlier_rows with recorded_as, most_likely, reason | Done |

---

## 9. Implementation Choices

| Choice | Implementation | Status |
|--------|----------------|--------|
| Interactive charts: Plotly | plotly-2.27.0.min.js CDN; Plotly.react from data-plotly-data/layout | Done |
| Map: hillshade or simple gradient | SVG with points; simple background (plan allows gradient if no tiles) | Done |
| All-loggers comparison: team medians in main step | `_compute_logger_median(merged_df/project_filtered_df, logger_col, metric_fn)`; comparisons.*.median_all, median_project | Done |

---

## 10. Code / Test Check

- Helpers import and run: `_add_mineralisation_accuracy_column`, `_plotly_pie_json` exercised from `src` with minimal DataFrame.
- Linter: no errors on `logging_review_html_report.py`.

---

## Summary

- **Implemented:** All per-tab requirements from the plan (Overview, Comments, Mineralisation, Zonation, Fines, Grouping, Outlier) including KPIs, charts, tables, geochem, flag rate explanation, group-based review, recorded_as/most_likely/reason, and data contract keys.
- **Deferred:** Outlier ternary/contour chart (strat unit clusters + outlier positions); plan allows table + common errors as primary; chart can be added later.
- **Enhancement:** Date range added to Overview header-meta for alignment with "Date range" in the plan.
