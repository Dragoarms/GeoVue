# Logging Review: Data Validations and Flags Reference

This document lists each validation used in the logging review report, where it runs, what it returns (flags/values), and how significance is set. Use it to correct logic or add new checks.

---

## 1. Mineralisation (Logging Accuracy)

**Where:** Mineralisation tab; applied per assay interval.

**Logic:** `_add_mineralisation_accuracy_column` in `processing/logging_review_html_report.py`. Compares assay chemistry (Fe, SiO2, Al2O3) vs logged "mineralised" status. **Logged status** is derived from zonation when available: Un/Le → unmineralised, De/Hy/Pr → mineralised; otherwise from total gangue (&lt;15% = mineralised, ≥15% = unmineralised). If logging agrees with assay → Match, else → Mismatch; if Fe missing → Pending Assays.

**Returns (column `Logging_Accuracy`):**

| Value             | Meaning |
|------------------|--------|
| `Match`          | Assay and logging agree (both mineralised or both not). |
| `Mismatch`       | Assay says mineralised but logging says unmineralised (or vice versa). |
| `Pending Assays` | Fe (or key assay) missing; cannot evaluate. |

**Evidence table only shows:** Mismatch rows.

**Significance (evidence table):**  
- `Low` if the logged percentages are within 10% of the assay-derived expected values (minor calibration issue).  
- `High` if the logged percentages deviate by more than 10% from assay-derived expected values (significant logging error).  

Set in `report_builder.py` when building `mineral_mismatch_intervals`:  
`significance = "Low" if deviation_within_tolerance else "High"`.
---

## 2. Zonation (Profile Zonation)

**Where:** Profile (zonation) tab; applied per interval with zonation + total gangue / De/Hy/Pr %.

**Logic:** `_check_profile_zonation_and_analyze` and `_collect_zonation_mismatch_rows` in `processing/logging_review_html_report.py`. Rules: Un 16–100% gangue; Le 11–15%; De/Hy/Pr 0–10% with matching dominant mineral (De/Hy/Pr). If rule fails → mismatch.

**Returns (evidence table rows):**

| Field           | Values / Meaning |
|----------------|------------------|
| `validation`   | Always `Mismatch` (table only lists mismatches). |
| `significance` | Based on magnitude of percentage deviation between logged and expected zonation category. |
| `logged_zonation` | Logged category (Un/Le/De/Hy/Pr). |
| `should_be`    | Category implied by gangue + mineral %. |

**Significance logic:**  
- `Low` if the logged mineral percentages are close to category boundaries (minor calibration issue, e.g. 12% gangue logged as De instead of Le).  
- `High` if the logged percentages are far from the expected range (significant logging error, e.g. 70% De logged as Hy).

The `_zonation_adjacent(logged, should_be)` helper approximates this by checking if categories are neighbours in the sequence [Un, Le, De, Hy, Pr] — adjacent categories imply the logger was near a boundary, non-adjacent implies a gross mismatch.

---
## 3. Mineralisation Logging Errors (Gangue Misclassification)

**Where:** Logging detail tab, sub-tab **Mineralisation Errors**.

**Logic:** `_flag_fines_issue` in `processing/logging_review_html_report.py`. Uses Fe, SiO2, Al2O3, total gangue to identify intervals where mineralisation logging contradicts assay chemistry.

**Purpose:** Flags two types of misclassification:
1. **Mineralisation logged as unmineralised/leached** — Assay shows ore-grade material but logger recorded high gangue.
2. **Unmineralised/leached logged as mineralisation** — Logger recorded no gangue but assay shows significant Si and/or Al phases.

The issue text guides the user on which gangue phases (silica, clays, or both) to focus on identifying.

**Returns (issue string or None):**

| Returned value                       | Condition |
|--------------------------------------|-----------|
| `Pending Assays`                     | Fe missing. |
| `Friable Silica`                     | Mineralisation logged (gangue = 0) but SiO2 > 5%, Al2O3 < 5% — silica phases present. |
| `Friable Silica and Clays`           | Mineralisation logged (gangue = 0) but SiO2 > 5%, Al2O3 > 5%, SiO2 > Al2O3 — both phases, silica dominant. |
| `Clays / Shales`                     | Mineralisation logged (gangue = 0) but SiO2 > 5%, Al2O3 > 5%, Al2O3 ≥ SiO2 — both phases, clay dominant. |
| `Mineralisation Logged as Gangue`    | Gangue logged (> 15%) but assay shows ore (SiO2 ≤ 10%, Al2O3 ≤ 10%). |
| `None`                               | No misclassification detected. |

**Significance:** `_significance_for_logging_detail_issue(issue, "fines")`:  
- **High** if issue text mentions major elements (fe, sio2, al2o3, gangue, mineralisation, silica, iron).  
- **Low** if only minor (p, cao, loi, carbonate) and not major.  
- Mineralisation errors are typically major → usually **High**.
---
## 4. Magnetite (vs LOI1000)

**Where:** Logging detail tab, sub-tab **Magnetite**.

**Logic:** `_flag_magnetite_issue`. LOI1000 is negative but no magnetite logged (MT, MBH, MBM) in min_*_pct columns.

**Rationale:** Negative LOI1000 indicates magnetite presence (mass gain on ignition). The reverse check (magnetite logged but LOI1000 ≥ 0) is not valid because other volatile components can increase LOI sufficiently to offset the magnetite effect.

**Returns (issue string or None):**

| Returned value                                              | Condition |
|-------------------------------------------------------------|-----------|
| `LOI1000 negative but no magnetite logged (MT/MBH/MBM)`    | LOI1000 < 0 and no magnetite codes present. |
| `None`                                                      | LOI1000 ≥ 0, LOI1000 missing/NaN, or magnetite logged. |

**Significance:** Same helper as fines; magnetite issue text mentions "LOI" → often **Low** (minor); if "magnetite"/"iron" counted as major could be **High**. Current logic: **Low** only when issue is minor-only; magnetite text has "Magnetite"/"LOI" so can be **Low**.
---

## 5. Goethite (Hy vs LOI425)

**Where:** Logging detail tab, sub-tab **Goethite**.

**Logic:** `_flag_goethite_issue`. Compares logged goethite (Hy %) against LOI425 to detect potential misclassification in either direction.

**Purpose:** LOI425 is a proxy for goethite content (water loss from FeOOH). A mismatch between logged Hy and LOI425 indicates:
1. **Low Hy logged but high LOI425** — Goethite present but logged as something else (e.g. hematite, detrital).
2. **High Hy logged but low LOI425** — Something else logged as goethite.

**Returns (issue string or None):**

| Returned value                              | Condition |
|--------------------------------------------|-----------|
| `Low goethite (Hy) logged but LOI425 high` | Hy < 20%, LOI425 ≥ 2% — goethite likely misclassified. |
| `High goethite (Hy) logged but LOI425 low` | Hy ≥ 20%, LOI425 < 2% — non-goethite likely misclassified as Hy. |
| `None`                                      | Hy and LOI425 are consistent. |

**Significance:** Based on the magnitude of mismatch between logged Hy % and LOI425-implied goethite content:
- **Low** if percentages are close to threshold boundaries (minor calibration issue).
- **High** if large deviation (e.g. 5% Hy logged with 8% LOI425, or 60% Hy logged with 0.5% LOI425).
---

## 6. Carbonate Gangue (vs Assay)

**Where:** Logging detail tab, sub-tab **Carbonate gangue**.

**Logic:** `_flag_carbonate_issue`. Compares logged carbonate gangue % against assay proxies for carbonate content.

**Geochemistry columns used:**
- **LOI 650–1000** — CO2 loss from carbonate decomposition (calcite decomposes ~650–900°C, dolomite ~500–750°C).
- **CaO** — Calcium oxide, proxy for calcite (CaCO3).
- **MgO** — Magnesium oxide, proxy for dolomite (CaMg(CO3)2) and magnesite (MgCO3).

**Purpose:** Flags mismatches between logged carbonate gangue and assay evidence. High CaO/MgO with elevated LOI 650–1000 indicates carbonate minerals; logged percentages should reflect this.

**Returns (issue string or None):**

| Returned value                                              | Condition |
|-------------------------------------------------------------|-----------|
| `Logged carbonate % high but assay (CaO/MgO/LOI) low`      | Carbonate > 15%, but CaO < 2%, MgO < 2%, LOI 650–1000 < 2%. |
| `Logged carbonate % low but assay (CaO/MgO/LOI) high`      | Carbonate < 2%, but CaO > 10% or MgO > 10% or LOI 650–1000 > 5%. |
| `None`                                                      | Consistent or insufficient data. |

**Significance:** Based on magnitude of mismatch between logged carbonate % and assay-derived carbonate content:
- **Low** if percentages are near threshold boundaries (minor calibration issue).
- **High** if large deviation (e.g. 30% carbonate logged with negligible CaO/MgO/LOI, or 0% logged with 15% CaO).

---
## 7. Outliers (geochemistry vs strat expectation)

**Where:** Outliers tab; rows with `outlier_score` above threshold.

**Logic:** `compute_hybrid_outlier_scores` in `reports/logging_review/data/outliers.py`. Per-strat Mahalanobis (CLR) + IQR flags. Reason text built from IQR: e.g. `"fe_pct high (value 62.00, IQR 45.00-52.00)"` or `"sio2_pct low (...)"`.

**Evidence table, KPI, and charts:** Only **strat-mismatch** intervals (recorded strat ≠ most likely strat, with high outlier score) are counted, listed in the table, and highlighted as outliers in the PCA and scatter charts. Chemistry-only extremes within the same unit are not shown as outliers; focusing on confidently misclassified intervals improves re-logging focus and, over iterations, improves the geochemical populations used for the next run.

**Returns (per row):**

| Field               | Meaning |
|---------------------|--------|
| `outlier_score`     | Max of Mahalanobis norm and IQR flag score. |
| `outlier_reason`    | Semicolon-separated list of `"{col} {high|low} (value ..., IQR ...)"`. |
| `outlier_elements`  | Comma-separated top 3 elements by robust z (for display). |

**Flags (for pills):** `_parse_outlier_reason_to_flags(reason)` parses reason and returns list of `(element_short_name, direction)` with `direction` in `"up"` | `"down"`.  
- “high” → up (↑), “low” → down (↓).  
- Element short names: Fe, SiO2, Al2O3, P, or truncated column name.

**Significance:** `_outlier_significance_from_reason(reason, elements_str)` in `tables.py`:  
- **High** if reason/elements mention major (fe_pct, fe, sio2, al2o3, al).  
- **Low** if only minor (p_pct, p) and not major.

---

## 8. Grouping (CV > 100% in group)

**Where:** Grouping tab; groups with high CV on Fe, SiO2, Al2O3, MgO, CaO, S.

**Logic:** `_grouping_avg_max_interval_and_groups` in `processing/logging_review_html_report.py`. CV computed per group; groups with max CV > 100% are flagged.

**Returns (per group):**

| Field          | Meaning |
|----------------|--------|
| `significance` | **High** if any of Fe, SiO2, Al2O3 (major) has CV > 100%; else **Low**. |
| `issue`        | `"CV >100% in group"` for each interval in the group. |

---

## Summary table

| Validation                    | Location        | Main flag/values returned                    | Significance rule |
|-------------------------------|-----------------|----------------------------------------------|-------------------|
| Mineralisation                | Mineralisation  | Match, Mismatch, Pending Assays              | Low if ≤10% deviation, else High |
| Zonation                      | Profile         | validation=Mismatch, should_be vs logged     | Low if near boundary, High if gross mismatch |
| Mineralisation Logging Errors | Logging detail  | 4 issue strings (Friable Silica, Clays, etc) | High (major element misclassification) |
| Magnetite                     | Logging detail  | LOI1000 negative but no magnetite logged     | High (missed magnetite) |
| Goethite                      | Logging detail  | 2 issue strings (Hy/LOI425 mismatch)         | Low if near threshold, High if large deviation |
| Carbonate gangue              | Logging detail  | 2 issue strings (high/low mismatch)          | Low if near threshold, High if large deviation |
| Outliers                      | Outliers        | outlier_reason (element high/low), flags     | High if major elements |
| Grouping                      | Grouping        | CV >100% in group                            | High if major element CV |

---

## Evidence table column summary

| Table / tab        | Columns (besides checkbox, Hole, Depth, Image) |
|--------------------|-------------------------------------------------|
| **Mineralisation** | Validation, Significance, Logged as, Logged Zonation, Assay suggests, Fe%, SiO2%, Al2O3% |
| **Zonation**       | Logged zonation, Should be, Validation, Significance, Total gangue % Logged, Logged De%, Logged Hy%, Logged Pr% |
| **Mineralisation errors**          | Issue, Significance, Classified as, Logged zonation, Fe%, SiO2%, Al2O3% |
| **Magnetite**      | Issue, Significance, Fe%, LOI1000%, Magnetite logged? {need a new part in the data prep for this!!}|
| **Goethite**       | Issue, Significance, Logged Hy%, LOI425%|
| **Carbonate gangue** | Issue, Significance, Carbonate%, CaO%, MgO%, LOI 650-1000% |
| **Outliers**       | Significance, Logged as, Could be, Flags (pills), , Fe%, SiO2%, Al2O3%, P%, Details |
| **Grouping**       | Strat, Significance (per group), Fe%, SiO2%, Al2O3%, P%, (geochem per interval not per group, each row coloured by the variance to the rest of the group to make it clear what shouldn't have been included.) |

---

**Files to change when fixing or adding validations:**

- **Mineralisation:** `processing/logging_review_html_report.py` (`_add_mineralisation_accuracy_column`); evidence rows in `report_builder.py`.
- **Zonation:** `processing/logging_review_html_report.py` (`_check_profile_zonation_and_analyze`, `_collect_zonation_mismatch_rows`, `_zonation_adjacent`).
- **Fines / Magnetite / Goethite / Carbonate:** `processing/logging_review_html_report.py` (`_flag_*_issue`, `_significance_for_logging_detail_issue`); intervals built in `report_builder.py`.
- **Outliers:** `reports/logging_review/data/outliers.py` (scores/reason); `report_builder.py` (outlier_rows); `tables.py` (`_parse_outlier_reason_to_flags`, `_outlier_significance_from_reason`).
- **Grouping:** `processing/logging_review_html_report.py` (`_grouping_avg_max_interval_and_groups`).
