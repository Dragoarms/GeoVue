# Section Tool Integration Analysis

## Executive Summary

The **Section Tool** (Geological Cross-Section Tool Suite) and **GeoVue** are complementary geological tools that can be integrated effectively. Both use Tkinter, Python, and deal with geological data. This document analyzes integration options and recommends a pragmatic approach.

---

## Tool Overviews

### Section Tool (Cross-Section PDF Suite)
- **Purpose**: Extract, georeference, and analyze geological features from PDF cross-sections
- **Key features**: PDF feature extraction, coordinate detection (E/N/RL), formation identification, contact detection, stratigraphic correlation, 3D visualization, DXF/GeoTIFF/CSV export
- **Entry point**: `geotools.main_gui:main()` or `python -m geotools.main_gui`
- **Dependencies**: PyMuPDF, numpy, matplotlib, shapely, rasterio, scipy (optional)
- **Config**: `configs.json` (prospects, units, faults), `~/.geo_cross_section/strat_column.json`

### GeoVue
- **Purpose**: Chip tray photo processing, RC drilling sample management, QA/QC, geological logging
- **Key features**: ArUco marker detection, compartment extraction, drillhole trace generation, logging review, drillhole correlation, visualization
- **Dependencies**: opencv, pillow, pandas, openpyxl, numpy (no PyMuPDF, matplotlib, shapely, rasterio by default)

---

## Integration Approaches

### Option A: Menu Launcher (Recommended – Fastest)
Add a menu item that launches the Section Tool as a **standalone subprocess**. Minimal coupling, no new dependencies in GeoVue core.

**Pros**: Quick to implement, no dependency conflicts, Section Tool runs in its own process
**Cons**: No shared state (stratigraphy, paths); two separate windows

### Option B: Embedded Window (Medium Effort)
Launch the Section Tool GUI in a `tk.Toplevel` within GeoVue (similar to Bulk Renamer / Logging Review). Requires installing Section Tool dependencies into GeoVue's environment.

**Pros**: Single application, shared window management, can pre-load paths
**Cons**: Dependency merge (PyMuPDF, matplotlib, shapely, rasterio); potential theme/conflict issues

### Option C: Shared Stratigraphy & Data (Deep Integration)
Full integration: shared lithology/stratigraphic configs between GeoVue and Section Tool, drillhole data feeding cross-section correlation.

**Pros**: Unified geological workflow, consistent units across chip trays and cross-sections
**Cons**: Significant development; schema alignment; config sync logic

---

## Recommended Path: Option A + Future Option C

1. **Phase 1**: Add a "Cross Section Tool" menu item that launches the Section Tool via subprocess (like an external tool).
2. **Phase 2** (optional): Add optional dependency installation and embedded window if user prefers in-app access.
3. **Phase 3** (optional): Shared stratigraphy config and drillhole → section correlation.

---

## Implementation Details

### Phase 1: Subprocess Launcher

1. **Add to GeoVue menu** (`main_gui.py`):
   - Add `"Tools"` menu with "Cross Section Tool" command
   - Or add under `Settings` as "Cross Section Tool..."

2. **Launch logic**:
   ```python
   def _open_cross_section_tool(self):
       """Launch the Geological Cross-Section Tool as external application."""
       try:
           import subprocess
           import sys
           section_tool_path = Path(__file__).resolve().parents[2] / "section tool"
           if not section_tool_path.exists():
               section_tool_path = Path.home() / "OneDrive" / "Home Python" / "section tool"
           launcher = section_tool_path / "launcher.py"
           if launcher.exists():
               subprocess.Popen([sys.executable, str(launcher)], cwd=str(section_tool_path))
           else:
               # Fallback: try geotools if installed
               subprocess.Popen([sys.executable, "-m", "geotools.main_gui"])
       except Exception as e:
           DialogHelper.show_message(self.root, "Error", f"Could not launch Cross Section Tool: {e}")
   ```

3. **Configurable path**: Store Section Tool path in `config.json` so users can point to their install location.

### Phase 2: Optional Embedded Window

- Add optional dependencies to `pyproject.toml`: `PyMuPDF`, `matplotlib`, `shapely`, `rasterio`
- Import `GeologicalCrossSectionGUI` and instantiate in a `Toplevel` (like `PhotoRenamerGUI`)
- Use `gui_manager.apply_theme(dialog)` for consistency
- Guard with try/except; fall back to subprocess if imports fail

### Phase 3: Stratigraphy Sharing

- **Section Tool** uses `configs.json` with units (name, color, prospect, thickness, etc.)
- **GeoVue** uses `lithology.json` and other color presets (categorical color mapping)
- Possible mapping:
  - Export Section Tool `configs.json` units → GeoVue-compatible lithology preset
  - Or: shared config file both tools read (e.g. `shared_stratigraphy.json` in project folder)
- **Drillhole data**: GeoVue's `drillhole_data.csv` / `drillhole_data_manager` could feed collar locations (E, N, RL) into Section Tool's correlation workflow for tie-line placement.

---

## Dependency Considerations

| Package     | Section Tool | GeoVue | Notes                                  |
|------------|--------------|--------|----------------------------------------|
| numpy      | ✓            | ✓      | Compatible                             |
| PyMuPDF    | ✓            | ✗      | Add only if embedding                   |
| matplotlib | ✓            | ✗      | Add only if embedding                   |
| shapely    | ✓            | ✗      | Add only if embedding                   |
| rasterio   | ✓            | ✗      | Add only if embedding                   |
| opencv     | ✗            | ✓      | No conflict                            |
| tkinter    | ✓            | ✓      | Both use TkAgg (matplotlib) if embedded |

**Recommendation**: For Phase 1, keep dependencies separate. Section Tool runs in its own process with its own venv. GeoVue only needs `subprocess` and `pathlib`.

---

## File Structure Suggestion

```
Home Python/
├── GeoVue 26_01_27/
│   ├── src/
│   │   └── gui/
│   │       └── main_gui.py      # Add _open_cross_section_tool, menu entry
│   └── config.json              # Optional: "section_tool_path": "..."
└── section tool/
    ├── launcher.py              # Existing - used by subprocess
    ├── geotools/
    └── configs.json
```

---

## Quick Implementation Checklist (Phase 1)

- [ ] Add `_open_cross_section_tool()` to `MainGUI`
- [ ] Add "Tools" menu with "Cross Section Tool" (or under Settings)
- [ ] Resolve Section Tool path: sibling folder `section tool` or config
- [ ] Add translations for "Cross Section Tool", "Tools" (if using translations.csv)
- [ ] Test launch on Windows (PowerShell/CMD)
- [ ] Document in CLAUDE.md / user docs

---

## Risks & Mitigations

| Risk                      | Mitigation                                              |
|---------------------------|---------------------------------------------------------|
| Section Tool path varies  | Configurable path in GeoVue config; fallback to `-m geotools.main_gui` if installed |
| Section Tool not installed| Clear error message; link to install instructions       |
| Two separate apps feel disjointed | Phase 2 embedded option; or document as "companion tool" |
| Different stratigraphy schemes | Phase 3: shared config or export/import utility        |

---

## Conclusion

**Phase 1 (subprocess launcher)** is low-risk, quick to implement, and provides immediate value by giving GeoVue users one-click access to the Cross Section Tool from within the same workflow. Deeper integration (stratigraphy sharing, drillhole ↔ section correlation) can follow based on user needs.
