# GeoVue ↔ Section Tool Integration Proposal

**Audience:** Section Tool (PyQt6) project — what GeoVue provides and what the Section Tool should be aware of for integration.  
**Date:** 2026-02-04

This document describes GeoVue’s features, paths, data formats, and conventions so the Section Tool can be migrated to PyQt6 and prepared for integration (launch from GeoVue, shared folder path, and export format for section interpretations).

---

## 1. How GeoVue will launch the Section Tool

- GeoVue will start the Section Tool as a **subprocess** (separate process). The Section Tool can use **PyQt6** (or any GUI toolkit) without affecting GeoVue (Tkinter).
- Launch: GeoVue will typically run something like  
  `subprocess.Popen([sys.executable, str(launcher_path)], cwd=str(section_tool_root))`  
  or `python -m geotools.main_gui` if the package is installed.
- **Optional:** GeoVue may pass the Cross Sections folder path via command-line argument or a small config file (path written by GeoVue, Section Tool reads on startup). This avoids the user having to point the Section Tool at the same folder again.

**Section Tool should:**

- Be runnable as a standalone app (e.g. `python -m geotools.main_gui` or `launcher.py`).
- Optionally accept a single argument: path to the folder of section PDFs (GeoVue’s Cross Sections folder).
- Optionally support “open this folder on startup” when that path is provided (so when launched from GeoVue, the correct project folder is already loaded).

---

## 2. Cross Sections folder path (GeoVue side)

- GeoVue will add a **user-configurable shared path** for the Cross Sections folder.
- **Config key:** `shared_folder_cross_sections`
- **UI:** Settings → “Shared Folder Settings” collapsible → “Cross Sections Folder:” with Browse (same pattern as “Drillhole Datasets Folder”).
- **First run:** Default folder under the shared base path will be created: `{shared_base}/Cross Sections`. User can change it later via Browse.
- The Section Tool does **not** read GeoVue’s config directly. When launched from GeoVue, GeoVue can pass this path as an argument (or write it to a small file the Section Tool reads). When run standalone, the Section Tool uses its own config/paths.

**Section Tool should:**

- Use the same folder for “section PDFs” when launched from GeoVue (path passed or config file). When run standalone, use Section Tool’s own path config.

---

## 3. Coordinate system and units (shared convention)

Both GeoVue and the Section Tool must use the **same coordinate convention** so section interpretations and drillhole data align.

| Item | Convention |
|------|------------|
| **Axes** | **E** = Easting, **N** = Northing, **Z** = RL (reduced level / elevation). Right-handed. |
| **Units** | Metres (m). |
| **Vertical section** | Section plane = constant Northing (N). Horizontal axis = Easting (E); vertical axis = RL. So each section is (E, RL) at one N. |
| **Drillhole collar** | GeoVue collar data: columns `HOLEID`, `BEST_X` (Easting), `BEST_Y` (Northing), `BEST_Z` (RL). |

**Section Tool should:**

- Export all interpretation geometry in **(E, N, RL)** in metres.
- Use the same E/N/RL convention in any export that GeoVue will import (e.g. “Export for GeoVue” JSON).

---

## 4. Export format for GeoVue import (“Export for GeoVue”)

GeoVue will import section interpretations from a **single JSON file** produced by the Section Tool (e.g. “Export for GeoVue” or “Export section interpretations”). GeoVue will merge this into its `section_interpretations.json` (or equivalent store).

### 4.1 Suggested JSON schema (Section Tool output)

The Section Tool should be able to export a JSON file in this shape (or equivalent so GeoVue can map fields):

```json
{
  "version": 1,
  "source": "section_tool",
  "sections": [
    {
      "section_id": "KM_122800_p0",
      "pdf_path": "path/to/section.pdf",
      "page_num": 0,
      "northing": 122800,
      "easting_min": 294000,
      "easting_max": 296000,
      "rl_min": 80,
      "rl_max": 350,
      "features": [
        {
          "type": "Polygon",
          "formation": "BIF",
          "name": "Unit 1",
          "vertices": [[294100, 122800, 120], [294500, 122800, 115], ...],
          "color": "#808080",
          "source": "pdf_import",
          "author": "Optional"
        },
        {
          "type": "PolyLine",
          "formation": "Contact",
          "name": "Contact 1",
          "vertices": [[294200, 122800, 110], [294400, 122800, 105], ...],
          "color": "#000000",
          "source": "pdf_import"
        },
        {
          "type": "Fault",
          "formation": "FAULT",
          "name": "Fault A",
          "vertices": [[294300, 122800, 200], [294600, 122800, 180], ...],
          "color": "#ff0000",
          "source": "pdf_import"
        }
      ]
    }
  ]
}
```

- **vertices:** List of `[easting, northing, rl]` in metres. For a vertical section, northing is constant for the whole section.
- **type:** `"Polygon"` | `"PolyLine"` | `"Fault"` (or equivalents GeoVue can map).
- **section_id:** Stable id (e.g. `{pdf_stem}_p{page_num}`).
- **pdf_path:** Relative to project or absolute; used for reference and re-import.
- **color:** Hex string preferred for display in GeoVue.

**Section Tool should:**

- Implement an export action (e.g. “Export for GeoVue”) that writes the above (or a directly compatible) JSON from current section data, with all vertices in (E, N, RL) from the existing georeferencing transform.

---

## 5. GeoVue data the Section Tool might use (optional / future)

If the Section Tool ever needs to **read** drillhole or project data from GeoVue (e.g. to show hole locations on sections, or to use the same stratigraphy), below is what GeoVue has.

### 5.1 Collar / hole locations

- **Source:** Collar file (CSV or Excel) path is derived from GeoVue’s “Drillhole Datasets” folder or a designated collar file.
- **Columns:** `HOLEID`, `BEST_X` (Easting), `BEST_Y` (Northing), `BEST_Z` (RL). Optional: `PROJECT`, `PLANNED_HOLEID`.
- **Units:** Metres. Same E, N, RL as sections.

If GeoVue ever passes a collar path to the Section Tool (e.g. when launching), the Section Tool could load it and plot hole collars on the section (E, RL at section northing).

### 5.2 Drillhole datasets folder

- **Config key in GeoVue:** `shared_folder_datasets`
- **Contents:** Folder containing one or more CSV files (interval data: HoleID, From, To, assays, lithology, etc.). One of these may be or reference the collar file.
- The Section Tool does not need to read this for the first phase of integration; it is listed here for future “drillholes on section” or stratigraphy alignment.

### 5.3 Lithology / stratigraphy (future)

- GeoVue uses **color presets** (e.g. `lithology.json`, `fe_grade.json`) for visualisation. Section Tool uses `configs.json` / strat column.
- Deeper integration could map Section Tool units to GeoVue lithology or share a single stratigraphy config; not required for subprocess launch + export.

---

## 6. File locations and paths

| Purpose | GeoVue | Section Tool (to be aware of) |
|--------|--------|--------------------------------|
| Section PDFs folder | `shared_folder_cross_sections` (user-configurable) | When launched from GeoVue, receive this path (arg or small config). When standalone, use own config. |
| Imported interpretations in GeoVue | GeoVue will store in e.g. `section_interpretations.json` under project/shared folder. | Section Tool only **produces** the export JSON; it does not need to know where GeoVue saves it. |
| Collar / datasets | `shared_folder_datasets`, collar file inside or alongside. | Optional: if GeoVue passes collar path, Section Tool can use it for hole locations. |

---

## 7. Summary: Section Tool checklist for GeoVue integration

1. **Runnable as subprocess**  
   Entry point (e.g. `python -m geotools.main_gui` or `launcher.py`) so GeoVue can start it without importing its code.

2. **Optional: accept Cross Sections folder path**  
   Command-line argument or small config file so GeoVue can pass `shared_folder_cross_sections` and Section Tool opens that folder when launched from GeoVue.

3. **Same coordinate convention**  
   E, N, RL in metres; vertical section = constant Northing; export all vertices as (E, N, RL).

4. **“Export for GeoVue” JSON**  
   Export format matching (or easily mappable to) the schema in §4: sections with metadata and features (Polygon/PolyLine/Fault) with `vertices` as `[[easting, northing, rl], ...]`, plus type, formation, name, color, source.

5. **Optional / later**  
   Read collar CSV (HOLEID, BEST_X, BEST_Y, BEST_Z) if GeoVue passes a path; use for hole locations on section. Align stratigraphy with GeoVue in a later phase.

This document can live in the GeoVue repo and be copied or linked into the Section Tool (PyQt6) repo as the integration contract.
