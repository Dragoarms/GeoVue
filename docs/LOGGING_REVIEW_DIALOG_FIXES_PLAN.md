# Logging Review Dialog Fixes Plan

## Overview

This plan addresses several UI/UX issues in the Logging Review HTML Report (`src/processing/logging_review_html_report.py`).

---

## Issues to Fix

### 1. Mineralisation Mismatches Tab - Image Orientation

**Problem:** The small thumbnail images in the mismatches table display in portrait orientation but should be landscape.

**Current State:**
- Images have `transform: rotate(90deg)` applied via `.rotated-image` CSS class
- This rotates the image but the container doesn't adjust properly for small thumbnails

**Solution:**
- Update CSS for `.image-cell-compact img` to remove or adjust the rotation for small thumbnails
- The small images (60px width) should display naturally in landscape orientation
- Only apply rotation when expanded to full-screen view

**Files to Modify:**
- `src/processing/logging_review_html_report.py` (CSS section, ~line 3874-3883)

**Changes:**
```css
/* Before */
.image-cell-compact img {
    width: 60px;
    height: auto;
    border-radius: 4px;
    border: 1px solid var(--border);
}

/* After - add explicit landscape sizing */
.image-cell-compact img {
    width: 80px;
    height: 50px;
    object-fit: cover;
    border-radius: 4px;
    border: 1px solid var(--border);
    transform: none;  /* Override rotated-image transform for thumbnails */
}
```

---

### 2. Mineralisation Mismatches Tab - Significance Column Sorting

**Problem:** The 'Significance' column should default to showing 'High' first.

**Current State:**
- Table has sortable columns but no default sort applied
- Significance values are "High" or "Low"

**Solution:**
- Sort the `mineral_mismatch_intervals` list by significance before rendering
- High significance items first, then Low

**Files to Modify:**
- `src/processing/logging_review_html_report.py` (~line 3121)

**Changes:**
- Before rendering the evidence table, sort intervals:
```python
# Sort mismatches by significance (High first)
sorted_intervals = sorted(
    report["mineralisation"].get("mismatch_intervals", []),
    key=lambda x: 0 if x.get("significance") == "High" else 1
)
```

---

### 3. Zonation Tab - Layout Restructure

**Problem:** The Zonation tab layout should match the Mineralisation tab with a full-width evidence table underneath the accuracy charts.

**Current State (lines 3167-3179):**
```html
<div class="two-panel">
    <div class="panel-card">
        <!-- Accuracy by category chart -->
    </div>
    <div class="panel-card">
        <!-- Attribution (should be X) - evidence table side by side -->
    </div>
</div>
```

**Desired Layout:**
```html
<div class="panel-card">
    <!-- Accuracy by category chart (full width or in charts panel) -->
</div>
<div class="evidence-section">
    <!-- Evidence Table (full width, below charts) -->
</div>
```

**Files to Modify:**
- `src/processing/logging_review_html_report.py` (~lines 3161-3204)

**Changes:**
- Remove the `two-panel` wrapper
- Put the chart in its own full-width panel or keep charts together
- Move the evidence table into an `evidence-section` div below the chart
- Update the title from "Attribution (should be X)" to "Evidence Table"
- Move "Based on mineral logging codes only." to be the description under the title

---

### 4. Zonation Tab - Title Rename

**Problem:** 
- Current title: "Attribution (should be X)"
- Should be: "Evidence Table"
- The description "Based on mineral logging codes only." should appear under the title

**Files to Modify:**
- `src/processing/logging_review_html_report.py` (~lines 3173-3174)

**Changes:**
```python
# Before
<h3 data-i18n-fr="Attribution (devrait etre X)" data-i18n-en="Attribution (should be X)">Attribution (should be X)</h3>
{zonation_based_on_note}

# After
<h3 data-i18n-fr="Tableau de preuves" data-i18n-en="Evidence Table">Evidence Table</h3>
<p class="zonation-based-on-note">Based on mineral logging codes only.</p>
```

---

### 5. Evidence Tables - Image Click Stacking Issue (IMPORTANT)

**Problem:** When clicking multiple images in evidence tables, they stack on top of each other instead of replacing the previous expanded image.

**Current State:**
```html
<img ... onclick="this.classList.toggle('expanded')" />
```
- Each image toggles its own `expanded` class independently
- Multiple images can have `expanded` class simultaneously

**Solution:**
- Add JavaScript to close any previously expanded image before expanding a new one
- Create a global click handler or modify the onclick to first remove `expanded` from all other images

**Files to Modify:**
- `src/processing/logging_review_html_report.py` (JavaScript section at end of file)

**Changes:**
Add to the `<script>` section:
```javascript
// Close any open expanded images when clicking a new one
document.addEventListener('click', function(e) {
    if (e.target.classList.contains('expandable-img')) {
        // Close all other expanded images first
        document.querySelectorAll('.expandable-img.expanded').forEach(function(img) {
            if (img !== e.target) {
                img.classList.remove('expanded');
            }
        });
    }
});

// Also close expanded image when clicking outside (on overlay area)
document.addEventListener('click', function(e) {
    if (e.target.classList.contains('expanded')) {
        // Clicking the expanded image itself closes it (already handled by toggle)
        return;
    }
    // If clicking elsewhere, close all expanded images
    if (!e.target.closest('.expandable-img')) {
        document.querySelectorAll('.expandable-img.expanded').forEach(function(img) {
            img.classList.remove('expanded');
        });
    }
});
```

Also update the onclick handlers to not use toggle:
```python
# Before
onclick="this.classList.toggle('expanded')"

# After (handled by the global event listener)
onclick="handleImageExpand(this)"
```

And add the handler function:
```javascript
function handleImageExpand(img) {
    const wasExpanded = img.classList.contains('expanded');
    // Close all expanded images
    document.querySelectorAll('.expandable-img.expanded').forEach(function(other) {
        other.classList.remove('expanded');
    });
    // Toggle the clicked one (if it wasn't already expanded, expand it)
    if (!wasExpanded) {
        img.classList.add('expanded');
    }
}
```

---

### 6. Zonation Evidence Table - Filter Out Correct Classifications (IMPORTANT)

**Problem:** If the logged zonation is the same as the 'should be' column, the row should not appear in the evidence table since the classification is correct.

**Current State:**
- `_collect_zonation_mismatch_rows()` function collects rows where the zonation doesn't match rules
- These rows include `logged_zonation` and `should_be` fields
- Currently all mismatches are shown, even if logged_zonation == should_be

**Analysis:**
Looking at `_collect_zonation_mismatch_rows()` (~line 1136-1200):
- It checks if the row passes validation rules (gangue %, dominant mineral)
- If not ok, it determines `should_be` based on the actual dominant mineral
- However, if `logged_zonation == should_be`, it means the zonation code is correct but perhaps the data triggered a false positive

**Solution:**
- In `_collect_zonation_mismatch_rows()`, add a filter to exclude rows where `logged_zonation == should_be`

**Files to Modify:**
- `src/processing/logging_review_html_report.py` (~line 1185-1200)

**Changes:**
```python
# Before (in _collect_zonation_mismatch_rows)
if not ok:
    rows_out.append({
        "hole_id": ...,
        "logged_zonation": zonation,
        "should_be": should_be or "-",
        ...
    })

# After - only add if logged != should_be
if not ok:
    if should_be and should_be != zonation:  # Only include actual mismatches
        rows_out.append({
            "hole_id": ...,
            "logged_zonation": zonation,
            "should_be": should_be,
            ...
        })
```

---

## Implementation Order

1. **Fix 6 (Filter correct classifications)** - Data-level fix, ensures clean data
2. **Fix 2 (Significance sorting)** - Simple sort before render
3. **Fix 5 (Image stacking)** - JavaScript fix for UX
4. **Fix 1 (Image orientation)** - CSS adjustment
5. **Fixes 3 & 4 (Zonation layout + title)** - HTML structure changes

---

## Testing Checklist

- [ ] Mineralisation tab: Images display in landscape orientation (wider than tall)
- [ ] Mineralisation tab: High significance rows appear before Low
- [ ] Zonation tab: Evidence table appears full-width below the chart
- [ ] Zonation tab: Title reads "Evidence Table" with description below
- [ ] All tabs: Clicking an image expands it; clicking another replaces it (no stacking)
- [ ] All tabs: Clicking outside an expanded image closes it
- [ ] Zonation tab: Rows where logged_zonation == should_be do not appear

---

## Summary of File Changes

**File:** `src/processing/logging_review_html_report.py`

| Section | Line Range (approx) | Change |
|---------|---------------------|--------|
| `_collect_zonation_mismatch_rows()` | 1185-1200 | Filter out rows where logged == should_be |
| `_render_mineralisation_evidence_table()` call | 3121 | Pre-sort intervals by significance |
| Profile section HTML | 3161-3204 | Restructure layout, rename title |
| CSS `.image-cell-compact img` | 3874-3883 | Adjust for landscape thumbnails |
| JavaScript section | end of file | Add image expand handler |
| Image onclick handlers | multiple | Update to use new handler |
