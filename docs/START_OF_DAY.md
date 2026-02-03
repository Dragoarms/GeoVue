# Start of day – where we left off

**Last updated:** 3 Feb 2026 (end of day)

Use this when you say **"start of the day - where were we"** to get a quick rundown.

---

## Git status at end of day

- **Branch:** `main`
- **Working tree:** Clean (nothing to commit)
- **Recent commits:** Dependencies/plans/preview script; logging review report updates (zonation, mineralisation, map CRS warning)

Everything we did today is committed and in the repo.

---

## Completed today (3 Feb 2026)

### Logging review dialog fixes (all done)

1. **Zonation evidence table** – Rows where logged zonation = “should be” are no longer shown (correct classifications filtered out).
2. **Mineralisation mismatches** – Table is sorted so **High** significance appears before Low.
3. **Evidence table images** – Clicking an image expands it; clicking another **replaces** it (no stacking). Clicking outside closes the expanded image.
4. **Mineralisation thumbnails** – Small images in evidence tables use landscape-style sizing (80×50px, `object-fit: cover`, no rotation).
5. **Zonation tab layout** – Evidence table is full width **below** the “Accuracy by category” chart (same pattern as mineralisation tab). Title is **“Evidence Table”** with description “Based on mineral logging codes only.” underneath.

### Map / collar coordinates

6. **UTM reprojection** – When collar coordinates look like UTM, the code now tries primary zones (33, 32, 31, 34) then **all UTM zones 1–60** for both hemispheres so the “Collar coordinates appear to be in a projected CRS…” warning goes away when reprojection succeeds. **Requires `pyproj`** (`pip install pyproj`).

---

## Not started / optional follow-ups

- **Volume bias (LOGGING_REVIEW_VOLUME_BIAS_PLAN.md)** – Phase 1 (percentage stacked bars for mineralisation quarterly charts) is documented but not implemented. Optional.
- **Manual testing** – If you want to re-check the dialog: mineralisation significance order, zonation layout, image expand/replace, thumbnails landscape, and map with collar CSV in UTM.

---

## How to resume tomorrow

1. Say: **“Start of the day - where were we”** (or open this file).
2. You’ll get: this rundown + anything left to do.
3. If you have new tasks, we can add them here or in a plan and track them.

No blocking work left from today; repo is in a good state to continue.
