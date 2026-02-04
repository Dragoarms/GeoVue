# Section Tool Integration – Feasibility and Design

**Date:** 2026-02-04  
**Goal:** Integrate the Section Tool’s PDF section interpretations into GeoVue so you can point at a folder of section PDFs, import 3D georeferenced interpretations, and view them with drill holes (and optionally add your own geometry), reusing GeoVue’s path config, dataset access, and trace/coordinate systems.

---

## 1. What We Have

### 1.1 Section Tool (Cross-Section PDF Suite)

- **Location:** `Home Python/section tool/` (sibling to GeoVue).
- **Role:** Extract and georeference geological features from PDF cross-sections.
- **Data flow:**
  - **Input:** PDFs (single or folder); each page has coordinate labels (Easting, Northing, RL).
  - **Detection:** `GeoReferencer.detect_coordinates(page, pdf_path)` → `coord_system` (northing, easting_labels, rl_labels, linear fit `ax,bx,cy,dy`).
  - **Transform:** `build_transformation()` → `(pdf_x, pdf_y) → (easting, northing, rl)` (vertical section: northing constant per page, E vs RL).
  - **Features:** `FeatureExtractor.extract_annotations(page)` → polygons, polylines, faults (with author tag); vertices in PDF coords.
  - **Real-world:** Each vertex `(x,y)` → `transform(x,y)` → `(E, N, RL)`. Same transform used for DXF/CSV export.
- **Outputs:** GeoTIFF, DXF (3D), CSV (e.g. `unified_sections.csv`), and in-memory structures (units, contacts, faults with vertices).
- **Batch:** `BatchProcessor` can scan folders, optionally use `northing_overrides`, and export unified DXF/CSV.

So the Section Tool already produces **3D georeferenced objects** (E, N, RL) suitable for import into GeoVue.

### 1.2 GeoVue Systems to Reuse

| System | Purpose | Relevance to sections |
|--------|---------|------------------------|
| **Config paths** | `local_folder_*`, `shared_folder_*`, `shared_folder_datasets` | Add e.g. `shared_folder_section_pdfs` (or `local_folder_section_pdfs`) for “folder/subfolders of section PDFs”. |
| **Dataset access** | `shared_folder_datasets` = folder of drillhole CSVs; collar file (HOLEID, BEST_X, BEST_Y, BEST_Z) | Same project = same coordinate system (E, N, RL). Section interpretations and collar/traces share (E,N,RL). |
| **DataCoordinator** | Single API: compartment_folders, csv_files, GeologicalStore, ImageIndex, etc. | Add a **SectionInterpretationStore** (or “section store”) that holds section metadata + list of features with 3D vertices. Optional `initialize(..., section_pdf_folder=...)` and lazy-load section data. |
| **DrillholeDataManager** | Collar (X,Y,Z), multi-CSV intervals, harmonization | Collar used for trace 3D positions; section plane (e.g. fixed northing) can be used to filter “holes on this section” and to project traces onto section view. |
| **survey_trace** | `build_trace()`, `xyz_at_depth()` → (E,N,RL) along hole | Project trace (E,N,RL) onto section plane (E vs RL at fixed N) for overlay. |
| **Correlation dialog** | Drillhole columns, tie lines, depth transforms | Natural place to **overlay section interpretations** when viewing holes that lie on a given section (northing band). |
| **File manager** | Path resolution, shared vs local | Section PDF folder can live on shared path so all users see same sections. |

No need to duplicate path resolution or dataset discovery; we add one more “resource type” (section PDFs) and one more store (section interpretations).

---

## 2. Data Model for “Interpretations in GeoVue”

### 2.1 Section metadata (per PDF page)

- `section_id`: e.g. `pdf_path_stem + "_p" + page_num` or a stable ID.
- `pdf_path`: path to PDF (relative to project or absolute).
- `page_num`: 0-based.
- `northing`: constant for that section.
- `easting_min`, `easting_max`, `rl_min`, `rl_max`: extent in real-world (for quick spatial filter).

### 2.2 Interpreted features (per section)

Each feature (from PDF or from “our own” drawing):

- `type`: `"Polygon"` | `"PolyLine"` | `"Fault"`.
- `formation` / `name`: e.g. unit name, fault name.
- `vertices`: list of `(easting, northing, rl)` (3D).
- `color`: optional (hex or RGB) for display.
- `source`: `"pdf_import"` | `"geovue_drawing"`.
- Optional: `author`, `metadata`.

For a vertical section, northing is constant so vertices could be stored as (easting, rl) and northing on the section; storing full 3D keeps one format for all views and future 3D.

### 2.3 Where to store

- **Option A – JSON in project:** e.g. `section_interpretations.json` under shared (or local) folder, similar to register. Single file: list of sections, each with metadata + list of features. Good for versioning and sharing.
- **Option B – Alongside datasets:** e.g. `section_interpretations/` folder with one JSON per section or one unified file. Fits “dataset” metaphor.
- **Option C – SQLite/DB:** Overkill for first version; can migrate later if needed.

**Recommendation:** Option A – one `section_interpretations.json` (or path in config) so it respects “shared folder” and stays consistent with GeoVue’s existing pattern.

---

## 3. Import Pipeline: “Point at folder of PDFs”

1. **Config:** User sets “Section PDF folder” (e.g. `shared_folder_section_pdfs`) in First Run or Settings; can be same as Section Tool’s folder.
2. **Scan:** Recursively (or one level) find `*.pdf` under that folder.
3. **Per PDF, per page:**
   - Open PDF (PyMuPDF); get page.
   - Run coordinate detection (Section Tool’s `GeoReferencer.detect_coordinates`). If missing northing, use filename/page heuristic or prompt once (or northing override map).
   - Build transform (`build_transformation()`).
   - Extract annotations (Section Tool’s `FeatureExtractor.extract_annotations`).
   - For each polygon/polyline: transform vertices (pdf → E,N,RL), attach formation/type/color, append to section’s feature list.
4. **Save:** Write/update `section_interpretations.json` (merge with existing so user-drawn items are kept).

**Dependency:** Either (a) call Section Tool as a **subprocess** and have it export a JSON/CSV that GeoVue reads (no PyMuPDF in GeoVue), or (b) add **optional** PyMuPDF (and optionally shapely) to GeoVue and reuse/copy the Section Tool’s `georeferencing` + `feature_extraction` logic (or import from section_tool package). (a) is zero dependency in GeoVue; (b) gives a single “Import sections” button and no external UI.

---

## 4. Display: “Pick holes and see interpretations on the section”

- **Correlation dialog:** Already has hole list and collar-based data. For a “current section” (e.g. chosen northing or “section line”):
  - Filter holes whose collar (N) lies within a small band of that northing (or “section line” if you define section by polyline later).
  - Load section interpretations for that northing (from SectionInterpretationStore).
  - Project: section is (E, RL) at fixed N; trace points (E,N,RL) → use (E, RL) and optionally clip by northing band. Draw interpretation polygons/lines in (E, RL) space alongside the hole columns (e.g. as a background layer or side panel “section strip”).
- **Alternative / addition:** Dedicated “Section viewer” tab or dialog: list of sections (by northing or by PDF name); pick one → show section extent (E vs RL), overlay drillhole traces (projected) and interpretation geometry; same store, same coordinates.

Both use the same store and same (E,N,RL) convention; no duplication of path or dataset logic.

---

## 5. Adding your own drawings

- In the Section Viewer or in the Correlation view (section layer), add tools: “Draw polyline”, “Draw polygon”.
- Input: user clicks in (E, RL) space (or PDF space if you keep inverse transform). Convert to (E, N, RL) using section northing.
- Append to the section’s feature list with `source: "geovue_drawing"`.
- Save to `section_interpretations.json`. No new system; same data model and store.

---

## 6. Feasibility Summary

| Item | Feasibility | Notes |
|------|-------------|--------|
| Point system at folder/subfolders of section PDFs | **High** | One new config key; reuse FileManager/path patterns. |
| Import interpretations with spatial (E,N,RL) info | **High** | Section Tool already does PDF → 3D; need import path (subprocess export vs in-process PyMuPDF). |
| Use shared path config / dataset access / coordinates / traces | **High** | Add one store and one optional init arg; rest is existing config + DataCoordinator + survey_trace. |
| Pick holes and see interpretations on section | **High** | Filter holes by northing; load section features; project traces to (E,RL); draw in Correlation or Section Viewer. |
| Add our own drawings and geometry | **High** | Draw in (E,RL), store as same feature type with `source: "geovue_drawing"`. |

Overall: **feasible** without breaking existing behaviour, and making full use of GeoVue’s path config, dataset access, shared folder, coordinates, and traces.

---

## 7. Phased Approach

### Phase 1 – Config + import (no UI display yet)

- Add config key(s): e.g. `shared_folder_section_pdfs`, `local_folder_section_pdfs` (optional).
- Implement **import** only:
  - **Option 1:** Section Tool subprocess: “export to GeoVue JSON” (new export in Section Tool) → GeoVue reads JSON and writes `section_interpretations.json`.
  - **Option 2:** In GeoVue, add optional dependency PyMuPDF (and minimal code): scan folder, run detection + extraction (copy or import from section tool), write `section_interpretations.json`.
- **Subprocess + toolkit independence:** If Option 1 is used, the Section Tool runs in a **separate process**. It can use a different GUI toolkit (e.g. migrate from Tkinter to **PyQt6**) with no impact on GeoVue. GeoVue only starts the process and (optionally) reads the exported JSON; no shared UI or imports, so Tkinter (GeoVue) and PyQt6 (Section Tool) never run in the same process.
- Implement **SectionInterpretationStore**: load/save JSON, query by northing or section_id. No UI yet; just “Import section PDFs” menu that runs import and saves.

### Phase 2 – Show interpretations with holes

- In Correlation dialog (or new “Section viewer”): when a section (northing) is selected, load section features from store; get holes for that northing from collar data; project traces to (E, RL); draw section layer (polygons/lines) + trace overlays. Read-only.

### Phase 3 – Drawing

- Drawing tools in section view: polyline/polygon; convert to (E,N,RL); append to store and save.

This keeps risk low and each step testable: Phase 1 doesn’t touch existing dialogs; Phase 2 is display-only; Phase 3 adds editing.

---

## 8. Technical Notes

- **Subprocess and GUI toolkit:** Launching the Section Tool via subprocess means it runs in its own process. You can migrate the Section Tool to PyQt6 (or any other toolkit) without causing issues when opening it from GeoVue; the two apps never share a process or GUI stack.
- **Coordinate convention:** Section Tool and GeoVue both use East, North, RL (Z). Section = (E, RL) at constant N. Align units (m) and axis orientation when projecting.
- **Northing matching:** Sections are keyed by northing (and possibly pdf_path+page). Collar “on section” = northing within a tolerance of section northing; same for trace projection.
- **Performance:** Large PDF folders: run import in background thread; show progress; cache store in memory and reload when JSON changes.
- **Breaking changes:** None if the new store and UI are additive and the new config key is optional with empty default.

---

## 9. Next Steps

1. Decide Phase 1 import strategy: subprocess + “GeoVue JSON” export from Section Tool vs in-process PyMuPDF in GeoVue.
2. Define exact JSON schema for `section_interpretations.json` (and optionally a small reader/writer in GeoVue).
3. Add `shared_folder_section_pdfs` (and optional local) to config and First Run / Settings UI.
4. Implement SectionInterpretationStore and “Import section PDFs” action that populates it.
5. Then Phase 2: section layer in Correlation (or Section Viewer) and Phase 3: drawing tools.

If you want, next we can (a) sketch the JSON schema and store API, or (b) outline the subprocess vs in-process trade-offs in more detail for Phase 1.
