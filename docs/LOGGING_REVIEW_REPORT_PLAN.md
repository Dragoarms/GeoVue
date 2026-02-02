# Logging Review Report – Implementation Plan

This plan addresses fixes and enhancements for the HTML logging review report (e.g. `RC_Logging_Review_CNE.html`), based on user feedback.

---

## 1. Overview Page – Map

**Issue:** Points plot in a single line or wrong positions; map does not work correctly.

**Cause:** Collar coordinates are resolved as easting/northing (e.g. UTM) and passed to Leaflet as `lat`/`lng` without reprojection. Leaflet expects WGS84 (lat/lng). UTM values (e.g. 500000, 7000000) interpreted as lat/lng produce a line or invalid positions.

**Implementation:**

- **File:** `src/processing/logging_review_html_report.py`
- **Function:** `_build_map_points` (uses `_resolve_coordinate_columns`), and any caller that assumes WGS84.
- **Options:**
  1. **Reproject to WGS84:** If collar CSV or config indicates a projected CRS (e.g. UTM zone), reproject easting/northing to lat/lng (e.g. with `pyproj`) before building `map_data["points"]` and `bounds`. Ensure `bounds` is `[min_lat, min_lng, max_lat, max_lng]` for Leaflet.
  2. **Document and validate:** If reprojection is not feasible in v1, document that collar coordinates must be in WGS84 (lat/lng) for the map to work, and add a simple heuristic (e.g. if all “lng” values are in 0–360 and “lat” in -90–90, treat as WGS84; else show a message “Collar coordinates may be in a projected CRS; map requires WGS84”).
- **JS:** No change needed once `points` and `bounds` are correct WGS84.

---

## 2. Overview Page – Strat Codes Chart (Logger vs Team)

**Issue:** Team dataset is much larger, so the grouped strat bar chart is dominated by team counts and logger distribution is hard to see.

**Goal:** Show **relative** logging distributions (logger vs team), not raw counts.

**Implementation:**

- **File:** `src/processing/logging_review_html_report.py`
- **Data:** Overview build already has `strat_code_list` (logger) and `team_strat_code_list` (team) as `[{"code": k, "count": v}, ...]`.
- **Changes:**
  1. **Normalize to proportions:** For the same set of strat codes (union of both), compute:
     - Logger: `logger_pct[code] = count / total_logger_intervals * 100`
     - Team: `team_pct[code] = count / total_team_intervals * 100`
  2. **Chart:** Keep or add a grouped bar chart that uses **percentages** (or proportions) on the y-axis, with two series “Logger” and “Team”, so both bars are on the same 0–100% scale.
  3. **Optional:** Keep a second chart with raw counts, or a toggle; main default view should be relative (%).  
- **Functions:** `_plotly_strat_grouped_bar_json` (or a new variant) should accept counts and totals and convert to percentages; layout `yaxis` title e.g. “% of intervals”.

---

## 3. Mineralisation Tab – “Your Data” Bar Chart (Quarter)

**Issue:** With only one quarter/bin, the x-axis shows weird time-like labels (e.g. `-59:59.9996 30, 2025`, `00:00:00 1, 2025`) and the chart looks wrong.

**Cause:** `Quarter` is a pandas Period/Timestamp; when converted to string it becomes a full datetime string. Plotly then treats x as continuous time and formats ticks as time. With one bin this produces odd labels.

**Implementation:**

- **File:** `src/processing/logging_review_html_report.py`
- **Data:** `_group_mineralisation_by_quarter` returns a DataFrame with column `Quarter` (from `pd.to_datetime(...).dt.to_period("Q").dt.start_time`).
- **Changes:**
  1. **Format quarter as categorical label:** When building the list of quarters for the chart (e.g. in the mineralisation section where `quarters_user = q_user["Quarter"].astype(str).tolist()`), format each quarter as a short categorical label, e.g. `"Q1 2025"` (e.g. `quarter.dt.strftime("%q %Y")` or Period’s `strftime` / manual "Q{n} {year}").
  2. **Force categorical x-axis:** In `_plotly_stacked_bar_json`, set layout `xaxis: { type: "category" }` so Plotly does not treat x as time. Optionally set `tickvals`/`ticktext` to the formatted quarter strings.
- **Single-bin case:** With one category, the chart will show one stacked bar with a single, readable x-tick (e.g. “Q1 2025”).

---

## 4. Mineralisation Tab – Evidence Table: Mismatches Only + Compartment Images

**Issue:** Evidence table shows both Match and Mismatch rows; it should show **mismatches only**. Also add compartment images as in the logging detail accuracy tab.

**Implementation:**

- **File:** `src/processing/logging_review_html_report.py`
- **Data:** Currently `mineral_mismatch_intervals` is populated with **all** rows (comment: “all statuses, not just mismatches”). The report key is `"mismatch_intervals"`.
- **Changes:**
  1. **Filter to mismatches only:** When building the list passed to the report, keep only rows where `validation == "Mismatch"`. Either:
     - Filter after the loop: `mineral_mismatch_intervals = [r for r in mineral_mismatch_intervals if r.get("validation") == "Mismatch"]`, or
     - Only append when `validation == "Mismatch"`.
  2. **Compartment images:** Reuse the same pattern as logging detail evidence table:
     - When building each interval dict, add an `image` field: base64 data URL from `_encode_image_for_interval(data_coordinator, hole_id, depth_to)` (or equivalent) using `hole_id` and `depth_to` from the row.
     - In `_render_mineralisation_evidence_table`, add a column “Image” (or “Compartment”) and for each row output the same small expandable image HTML as in `_render_logging_detail_evidence_table` (using `item.get("image")`). Pass `data_coordinator` into the renderer if needed for image lookup, or pass pre-resolved image URLs/base64 in the interval dicts.

---

## 5. Zonation Tab – Rules Table and “Required columns” Text

**Issues:**
- Rules table shows French-only text for dominant mineral: “De doit etre dominant”, “Hy doit etre dominant”, “Pr doit etre dominant”. Should be bilingual and EN text should read e.g. “De minerals are dominant”.
- The info paragraph “Required columns: BestProfileZonation_D, Total Gangue Logged, De % Logged, Hy % Logged, Pr % Logged” should not be shown.

**Implementation:**

- **File:** `src/processing/logging_review_html_report.py`
- **Rules table (profile zonation section):**
  - Replace the hardcoded `<td>De doit etre dominant</td>` (and Hy, Pr) with cells that support i18n, e.g.:
    - De: `data-i18n-en="De minerals are dominant"` `data-i18n-fr="De doit etre dominant"` (and ensure the JS or backend uses the correct locale for “Attribution (should be X)” and these cells).
  - Hy: “Hy minerals are dominant” / “Hy doit etre dominant”.
  - Pr: “Pr minerals are dominant” / “Pr doit etre dominant”.
  - Un and Le stay “-” (no dominant).
- **Required columns paragraph:** Remove the `<p class="rules-note">` that contains “Required columns: BestProfileZonation_D, …” (or the entire block that renders that text). Do not show this info text.

---

## 6. Zonation Tab – “Attribution (should be X)” as Evidence Table

**Issue:** The “Attribution” card is currently a summary table (Logged as, Should be, Count). User wants it to be an **evidence table** like the others: filtered for **rows with errors**, with compartment images, and showing **each logged mineral (Min_x_pct columns)**, **what it was logged as**, and **what it should be**. Also make it clear that zonation is based on **mineral logging codes only**.

**Implementation:**

- **File:** `src/processing/logging_review_html_report.py`
- **Data:** `_check_profile_zonation_and_analyze` currently returns counts and `mismatch_attribution` (category → {should_be: count}). We need **row-level** zonation mismatch data.
- **Changes:**
  1. **Row-level zonation mismatches:** Extend the zonation analysis (or add a new pass) to produce a list of interval-level rows where zonation validation failed. Each row should include: hole_id, depth_from, depth_to, logged_zonation, expected_zonation (or “should be”), total_gangue, De/Hy/Pr/Un % (or the actual Min_*_pct columns used for zonation), and optionally other Min_x_pct columns for display. Use the same zonation rules (Un 16–100%, Le 11–15%, De/Hy/Pr 0–10% with dominant mineral).
  2. **Evidence table columns:** Hole, Depth, Logged zonation, Should be, Total gangue (%), and for each mineral logging code column (e.g. min_80_pct … min_0_5_pct or the resolved zonation De/Hy/Pr/Un % and any other Min_x_pct you expose): “Logged value” and “Expected / rule”. Add an “Image” column with compartment image (same pattern as logging detail and mineralisation evidence tables).
  3. **Replace attribution card:** In the zonation section, replace the current attribution summary table with this new evidence table (filtered to rows with errors only). Title can stay “Attribution (should be X)” or “Evidence – Zonation mismatches”.
  4. **Clarification text:** Add a short note above or below the table: “Based on mineral logging codes only” (EN) / “Basé sur les codes de minéraux uniquement” (FR), with `data-i18n-en` / `data-i18n-fr` so it switches with the rest of the UI.

---

## 7. Zonation Tab – “Based on mineral logging codes only”

**Implementation:** Add the clarification in two places:
- Near the zonation validation rules table (or the new evidence table): one line of text with i18n.
- Optionally in the methodology note for the zonation section so it’s clear that validation uses only the mineral logging columns (BestProfileZonation_D, total gangue, De/Hy/Pr/Un % or equivalent Min_*_pct logic).

---

## 8. Compartment Images in All Evidence Tables

**Requirement:** Show compartment images in **all** evidence tables (mineralisation, zonation attribution, and any others that list intervals), in the same way as the logging detail accuracy tab.

**Implementation:**

- **Pattern:** Reuse the logging detail approach:
  - When building interval lists for report data, add an `image` key (base64 data URL or path) per row using `data_coordinator.get_image_path(ImageKey(hole, depth_to, moisture))` and `_encode_image_base64`.
- **Tables to update:**
  1. **Mineralisation evidence table** – see §4.
  2. **Zonation attribution evidence table** – see §6.
  3. **Any other evidence tables** that list hole/depth intervals (e.g. fines, grouping) – add the same Image column and populate `image` when building the interval dicts.
- **Render:** Each evidence table renderer should accept an optional “image” per row and render the same small expandable image cell as in `_render_logging_detail_evidence_table`.

---

## 9. Summary Checklist

| # | Item | File(s) | Priority |
|---|------|--------|----------|
| 1 | Map: reproject or document WGS84 | `logging_review_html_report.py` | High |
| 2 | Strat chart: relative (%). Logger vs Team | `logging_review_html_report.py` | High |
| 3 | Mineralisation bar: categorical quarter, single-bin fix | `logging_review_html_report.py` | High |
| 4 | Mineralisation evidence: mismatches only + images | `logging_review_html_report.py` | High |
| 5 | Zonation rules: bilingual dominant text; remove “Required columns” | `logging_review_html_report.py` | Medium |
| 6 | Zonation: Attribution → evidence table (errors only, Min_x_pct, images) | `logging_review_html_report.py` | High |
| 7 | Zonation: “Based on mineral logging codes only” text | `logging_review_html_report.py` | Medium |
| 8 | Compartment images in all evidence tables | `logging_review_html_report.py` | High |

---

## 10. Suggested Order of Work

1. **Quick wins:** Zonation rules text (bilingual + remove “Required columns”) (§5).
2. **Mineralisation:** Filter evidence table to mismatches only (§4); fix quarter bar chart (§3).
3. **Strat chart:** Relative distribution (§2).
4. **Images:** Add compartment images to mineralisation evidence table (§4), then zonation evidence table (§6), then others (§8).
5. **Zonation evidence:** Row-level data and new evidence table (§6) + “mineral logging codes only” (§7).
6. **Map:** Reproject or document WGS84 (§1).

This plan can be implemented incrementally; each section can be done and tested independently.
