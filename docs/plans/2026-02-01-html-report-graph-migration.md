# HTML Logging Review Report – Graph Migration & Per-Tab Spec

## 1. Graph Migration Table (QAQC Script → HTML Tab → Implementation)

| QAQC Script | Chart/Table | HTML Tab | Implementation | Notes |
|-------------|-------------|----------|----------------|-------|
| QAQC_02_Summary_Statistics | KPI rows (holes, prospects, total depth, date range) | Overview | KPI cards (existing) | Match PDF layout; add assay received/outstanding. |
| QAQC_02 | Table 1: Prospects Summary (Prospect, Holes, Total Depth) | Overview | HTML table | Optional if data available. |
| QAQC_02 | Table 2: Assay Status (Prospect, Total Samples, Pending Dispatch, Pending Assays, Assays Returned) | Overview | HTML table | Assay received = rows in exassay; outstanding = in exgeologyRC but not in exassay (interval lengths differ). |
| — | Assay received vs outstanding (counts) | Overview | KPI cards + short explanation | Not a bar; "Assays received" (exassay row count), "Outstanding" (exgeologyRC intervals with no matching exassay). |
| — | Average logging interval (m) | Overview | KPI card | Mean of (depth_to - depth_from) from logging intervals. |
| — | Strat code population bar | Overview | Plotly bar or Python PNG | Bar chart of strat code counts (user's data only on Overview). |
| — | Collar map with user + team holes | Overview | Map with world hillshade background | User holes highlighted; team in grey. No "team comparison" KPIs on Overview. |
| QAQC_03_Comment_Statistics | Word cloud | Comments | Pre-generated PNG or Plotly | Same as QAQC03 word map; words from comments. |
| QAQC_03 | Bar: "No Comment" vs "Comment" (counts) | Comments | Plotly bar or PNG | Total rows = logging intervals; NULL/missing = No Comment (bad). Ratio with comments should be high (>20%). |
| QAQC_04_Mineralisation_Accuracy_Checks | Pie: Match / Mismatch / Pending Assays | Mineralisation | Plotly pie or PNG (QAQC04 logic) | User's pie + Team's pie side by side. |
| QAQC_04 | Stacked bar by quarter (Match/Mismatch/Pending %) | Mineralisation | Plotly stacked bar or PNG | User's chart + Team's chart. |
| QAQC_05_Check_Profile_Zonation | Clustered bar: Correctly vs Incorrectly logged by zonation category | Zonation | Plotly or PNG | Match QAQC05 zonation_accuracy_chart. |
| QAQC_05 | Attribution chart: per zonation type, "Should be X" counts | Zonation | Plotly or PNG | Match QAQC05 zonation_attribution_chart. |
| QAQC_06_Fines_Accuracy | Fines flags summary + intervals for review | Fines | Existing + larger images, geochem | Landscape images; add Fe, Al2O3, SiO2, P next to each image; explain flag rate; fix all-loggers comparison. |
| QAQC_07_Grouping_Accuracy | CV analysis (Fe, SiO2, Al2O3, MgO, CaO, S only) | Grouping | KPI cards + table | Avg group interval card, max group interval card; intervals for review = groups with highest CV (not single metre); show each image in group + geochem. |
| Outlier (logging_review_report) | Top outliers list + "classified as X but likely Y" | Outlier | Redo: total misclassified, common errors (strat X → Y), ternary/contour + outlier position | "Recorded as" (strat column) vs "Most likely" (model/cluster); explanation "because of Z" (e.g. geochem). |

## 2. Per-Tab Requirements (Summary)

### Overview
- **KPIs (match QAQC02 PDF):** Holes, Prospects, Total Depth, Date range, **Assays received** (exassay row count), **Outstanding** (exgeologyRC intervals without matching exassay), **Average logging interval (m)**.
- **No** "assay intervals vs assay intervals" loading bar; replace with assays received count and outstanding count.
- **Map:** World hillshade background; user's drillholes + team's drillholes (no team comparison metrics on this tab).
- **Strat bar:** Population of strat codes (user's data).
- **Comparison rule:** Move to bottom of tab.
- **Comment stats on Overview:** Ratio = (intervals with at least one non-NULL comment) / (total logging intervals). Total = all logging intervals; NULL/missing counts as "no comment". Target >20%.

### Comments
- **Word cloud:** Pre-generated (Python wordcloud lib) or Plotly; same as QAQC03.
- **Bar chart:** Same as QAQC03 – "No Comment" vs "Comment" counts; **rows with NULL/missing must be counted as No Comment**.
- Pre-generated or Plotly; ensure NULLs are counted.

### Mineralisation
- **Fully redo** to match QAQC_04: add "Logging Accuracy" (Match/Mismatch/Pending Assays) from Fe_pct_BEST, SiO2_pct_BEST, Al2O3_pct_BEST, Total Gangue Logged.
- **Charts:** (1) Pie – Overall accuracy (Match/Mismatch/Pending); (2) Stacked bar – Accuracy by quarter. User's charts + Team's charts side by side.
- Use same logic as QAQC_04_Mineralisation_Accuracy_Checks (add_accuracy_column, group_and_summarize, create_summary_plots).

### Zonation
- **Redo** to match QAQC_05: zonation categories (Un, Le, De, Hy, Pr etc.), correct vs incorrect by category.
- **Charts:** (1) Clustered bar – Correctly vs Incorrectly logged by zonation category; (2) Attribution – per category "Should be X" horizontal bars. Plotly or Python PNG embed.

### Fines
- **Keep** likely-issue column and intervals list.
- **Images:** Larger, landscape; add **geochem (Fe, Al2O3, SiO2, P)** next to each image.
- **Flag rate:** Add short explanation (what flag rate means); fix **all-loggers** comparison so team stats are actually computed and shown (currently identical in every tab).

### Grouping
- **KPI cards:** Average group interval (m), Max group interval (m).
- **CV analysis:** Only major elements Fe, SiO2, Al2O3, MgO, CaO, S (not full chemistry).
- **Intervals for review:** Show **groups** with highest CV (not single metre); for each group show each image in the group if available + geochem; keep layout tight.

### Outlier
- **KPIs:** Total number of intervals likely misclassified; **most common errors** (e.g. "Classified as BIF but actually AMP").
- **Charts:** Strat unit clusters (e.g. ternary diagram with strat clusters as heatmap/contour) and outlier positions on same plot.
- **Top outliers table:** Show "Recorded as" (strat column) and "Most likely" (model/cluster suggestion); explanation "because of Z" (geochem/distance), not generic "geochemistry outlier vs strat expectation".

## 3. Implementation Choices

- **Interactive charts:** Plotly (embed as JSON in HTML, or `plotly.js` inline) so charts are interactive where useful.
- **Static charts:** Python (matplotlib/plotly) → PNG → base64 embed for compatibility or when interactivity not needed.
- **Map:** Use a world hillshade raster or tile URL (e.g. Stamen Terrain, or a static hillshade image) as background; plot collar points (user + team) on top. If no external tiles, use a simple gradient or pattern as "terrain" and plot points.
- **All-loggers comparison:** Compute team (or project-scoped) medians/aggregates in the main report data step and pass into each tab so "Your data" vs "All loggers" panels show different values.

## 4. Data Contract (for section renderers)

Each tab receives a `report_data` dict (and optionally `intervals_for_review`). Main generator must provide:

- **Overview:** `assay_received_count`, `assay_outstanding_count`, `average_logging_interval_m`, `strat_code_counts` (dict or list for bar), `map_points` (user + team), `summary` (holes, prospects, total_depth, date_range), `comment_ratio_pct` (with NULLs as no comment). No `comparisons` on Overview.
- **Comments:** `wordcloud_data` (word → count), `total_rows`, `rows_with_comment`, `rows_without_comment` (include NULLs), `comment_ratio_pct`; optional team equivalents for comparison.
- **Mineralisation:** `accuracy_counts` (Match/Mismatch/Pending), `quarterly_summary` (for stacked bar); same for team.
- **Zonation:** `correct_counts`, `mismatch_counts`, `mismatch_attribution` (per category "should be X"); from QAQC05 logic.
- **Fines:** Existing + `geochem_cols` (Fe, Al2O3, SiO2, P) per interval; flag rate explanation; team comparison values.
- **Grouping:** `avg_group_interval_m`, `max_group_interval_m`, CV stats for Fe/SiO2/Al2O3/MgO/CaO/S; intervals = list of **groups** (with group key, CV, list of (image, geochem) for each interval in group).
- **Outlier:** `total_misclassified`, `common_errors` (list of {recorded_as, likely, count}), `outlier_rows` with `recorded_as`, `likely`, `reason`; ternary/contour data for strat clusters + outlier positions.

## 5. Agent Task Boundaries (for parallel implementation)

1. **Overview** – KPIs (assay received/outstanding, avg logging interval), map (hillshade + points), strat bar, comment ratio (NULLs = no comment), comparison rule at bottom; no team comparison on this tab.
2. **Comments** – Word cloud (pre-gen or Plotly), bar chart (No Comment vs Comment, NULLs counted); QAQC03 logic.
3. **Mineralisation** – QAQC04 accuracy column + pie + quarterly stacked bar; user + team side by side; Plotly or PNG.
4. **Zonation** – QAQC05 mismatch chart + attribution chart; Plotly or PNG.
5. **Fines** – Larger landscape images, geochem (Fe/Al2O3/SiO2/P) per row, flag rate explanation, fix all-loggers comparison.
6. **Grouping** – Avg/max group interval cards, CV for Fe/SiO2/Al2O3/MgO/CaO/S, intervals = groups with highest CV, show all images in group + geochem.
7. **Outlier** – Total misclassified, common errors (X → Y), "recorded as" vs "most likely", ternary/contour + outlier position, specific explanation per row.

Each agent (or sequential pass) produces or updates one section of the report and validates that their part is functional.
