# Logging Review Report – Removing Volume Bias

## Problem

Absolute count bar charts hide the real signal. With varying sample sizes (e.g. more meters logged in one quarter), you cannot tell whether a quarter’s high mismatch count means:

- **Performance dropped**, or  
- **You simply logged more meters** (so raw counts went up).

To avoid penalising people for volume and to see real performance trends, we need to **remove volume bias** wherever counts are shown over time or by segment.

---

## Scope (“everything”)

Apply volume-bias removal to all places where **counts** are shown and **sample size varies**:

1. **Mineralisation tab – quarterly bar charts (Your data / All loggers)**  
   Currently: stacked bar by quarter with **absolute** Match / Mismatch / Pending Assays counts.  
   High mismatch in Q3 could mean worse performance or just more intervals in Q3.

2. **Any other time-based or segment-based count charts** in the report (e.g. zonation by period, comments by period, etc.) – same principle: show proportions or rates with uncertainty, not raw counts alone.

---

## Normalisation approaches (in order of use)

### 1. Percentage stacked bars (simplest fix)

**Idea:** Convert each quarter (or segment) to 100% and show Match / Mismatch / Pending as **proportions**. Volume is removed because each bar sums to 100%.

**Implementation:**

- For each quarter: `total = match + mismatch + pending`; if `total > 0`:  
  `match_pct = 100 * match / total`, same for mismatch and pending.
- Stack these percentages (so each bar is 100%).
- Y-axis: “% of quarter” (or “% of intervals”), range 0–100.
- Optional: show total count in the quarter as a label or tooltip so users still see volume.

**Pros:** Very easy to read; no volume bias.  
**Cons:** Does not show uncertainty (small quarters look as precise as large ones).

---

### 2. Control charts with confidence intervals (statistically rigorous)

**Idea:** For binomial-style outcomes (Match vs not), treat each quarter as an estimate of “match rate” and show **uncertainty**. Smaller sample size → wider interval → you can see when a “drop” is likely noise vs a real trend.

**Formulas (per quarter):**

- Match rate: `p = matches / n` (n = total intervals in that quarter).
- Standard error: `SE = sqrt(p * (1 - p) / n)`.
- 95% CI: `p ± 1.96 * SE` (or use `scipy.stats` for exact binomial CI if preferred).

**Chart:**

- X-axis: quarter (or segment).
- Y-axis: match rate (0–1 or 0–100%).
- Point: `p`; vertical interval: 95% CI.
- Optionally a horizontal line at overall match rate (e.g. pool all quarters) as a “process average”.

**Pros:** Correct for varying n; distinguishes noise from real change.  
**Cons:** Slightly more complex; need to explain “wide interval = small sample”.

---

### 3. Funnel plot

**Idea:** Plot **accuracy rate** (y-axis) vs **sample size** (x-axis), with control limits that **widen as n decreases**. Points outside the funnel are statistically different from the mean.

**Implementation:**

- Each point = one quarter (or segment): x = n (sample size), y = match rate (or accuracy rate).
- Overall rate: e.g. `p0 = total_matches / total_n` over all quarters.
- Control limits: for each n, compute acceptable range for p (e.g. 95% CI under p0); plot as a funnel (narrow at high n, wide at low n).
- Points outside the funnel → that quarter is “out of control” (genuinely different from average).

**Pros:** Single plot for “which periods are genuinely different”; good for auditing.  
**Cons:** Less intuitive for non-statistical users; best as a second view (e.g. “Funnel” tab or below the main chart).

---

## Implementation plan (phased)

### Phase 1 – Percentage stacked bars (immediate)

- **Mineralisation tab – quarterly charts (user + team):**
  - Keep same data source: `_group_mineralisation_by_quarter` (Quarter, Match, Mismatch, Pending Assays).
  - When building the chart inputs: for each quarter convert counts to **percentages** (each quarter sums to 100%).
  - Use a single chart function that accepts either counts or percentages; for mineralisation, pass **percentages** and set y-axis to “% of quarter”, range 0–100.
  - Optionally: add a small note under the chart or in tooltip: “Bars show % of intervals in each quarter (volume-neutral).”
- **File:** `src/processing/logging_review_html_report.py`
- **Functions:** `_plotly_stacked_bar_json` (extend or add `_plotly_stacked_bar_pct_json`), and the mineralisation section that builds `match_user`, `mismatch_user`, `pending_user` (and team) from `q_user` / `q_team`.

### Phase 2 – Control chart with 95% CI (optional)

- Add a **second** chart (or replace the stacked bar) for “Match rate by quarter with 95% CI”:
  - Per quarter: n = match + mismatch (+ pending if you treat pending as “not match”); p = match / n; SE; 95% CI.
  - Plot: line or points for p, with error bars or shaded band for 95% CI.
- Good for power users who want to see “is this quarter’s drop real?”

### Phase 3 – Funnel plot (optional)

- Add an optional “Funnel” view: one scatter plot, x = n per quarter, y = match rate, with funnel limits.
- Could live in a collapsible “Advanced” section or a separate small panel.

---

## Consistency across the report

- **Strat codes (overview):** Already normalised to % (logger vs team as % of intervals). Keep as is.
- **Mineralisation quarterly:** Phase 1 = percentage stacked bars; Phases 2–3 = optional CI/funnel.
- **Any other count-by-period or count-by-segment chart:** Apply the same principle (prefer % or rate + uncertainty over raw counts when comparing across different sample sizes).

---

## Summary

| Approach              | Removes volume bias? | Shows uncertainty? | Complexity |
|-----------------------|----------------------|--------------------|------------|
| Percentage stacked bar| Yes                  | No                 | Low        |
| Control chart (95% CI)| Yes                  | Yes                | Medium     |
| Funnel plot           | Yes                  | Yes (implicit)     | Medium     |

**Recommendation:** Implement **Phase 1 (percentage stacked bars)** for mineralisation quarterly charts first. Add Phase 2 (and optionally Phase 3) if you want to distinguish “noise” from “real trend” in a statistically rigorous way.
