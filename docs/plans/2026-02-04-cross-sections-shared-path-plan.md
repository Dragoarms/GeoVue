# Plan: Add Cross Sections Shared Path to GeoVue

**Date:** 2026-02-04  
**Goal:** Add a user-configurable shared folder path for Cross Section PDFs (Section Tool integration), following the same patterns as Drill Traces and Drillhole Datasets.

---

## 1. Config key and default

- **Config key:** `shared_folder_cross_sections`
- **Default:** Empty string `""` (or, when shared base is set, default to `{shared_base_dir}/Cross Sections` in FileManager – see below).
- **Type:** String path (folder).

Add to `ConfigManager`:

- **USER_SETTINGS_KEYS:** append `"shared_folder_cross_sections"`.
- **DEFAULT_CONFIG:** add `"shared_folder_cross_sections": ""`.

**Files:** `src/core/config_manager.py`

---

## 2. FileManager

- **Path key:** `"cross_sections"` (used in `get_shared_path`, `update_shared_path`, browse callback).
- **Config key mapping:** In `update_shared_path`’s `config_key_map`, add  
  `"cross_sections": "shared_folder_cross_sections"`.
- **FOLDER_NAMES:** Add `"cross_sections": "Cross Sections"` (for default under shared base).
- **initialize_shared_paths:**  
  - Add to `shared_paths` dict:  
    `"cross_sections": Path(config.get("shared_folder_cross_sections")) if config.get("shared_folder_cross_sections") else (self.shared_base_dir / self.FOLDER_NAMES["cross_sections"])`  
    (only when `self.shared_base_dir` is set; otherwise leave unset or from config only).  
  - In the block that writes back to config, add:  
    `self.config_manager.set("shared_folder_cross_sections", str(self.shared_paths["cross_sections"]))`  
    for the case where we have a default (so config stays in sync).

**Files:** `src/core/file_manager.py`

---

## 3. Main GUI – Shared folder collapsible

- **StringVar:** e.g. `self.cross_sections_path_var = tk.StringVar()` (with other shared path vars).
- **Initial value:** Same pattern as datasets:  
  `file_manager.get_shared_path("cross_sections", create_if_missing=False)` or `config.get("shared_folder_cross_sections", "")`.
- **Path field:** Call `_create_shared_path_field(..., "Cross Sections Folder:", self.cross_sections_path_var, cross_sections_exists, is_file=False, path_key="cross_sections")` after the Drillhole Datasets field.
- **Validity for coloring:** `cross_sections_exists = bool(cross_sections_path and Path(cross_sections_path).exists())` where `cross_sections_path` comes from get_shared_path or config (same as display value).

**Files:** `src/gui/main_gui.py`

---

## 4. Main GUI – setup_shared_paths

In `setup_shared_paths`, after the Register Data folder block:

- Get path: `cross_sections_path = self.file_manager.get_shared_path("cross_sections", create_if_missing=False)`.
- If path exists:  
  - `self.app.config_manager.set("shared_folder_cross_sections", str(cross_sections_path))`  
  - `self.cross_sections_path_var.set(str(cross_sections_path))`  
  - `paths_found = True` (if using that flag).

**Files:** `src/gui/main_gui.py`

---

## 5. Main GUI – translation / label update

In the block that updates shared folder path labels when language changes (search for “Approved” / “Processed” / “Drill” / “Excel” in `shared_collapsible.content_frame`), add a condition for the Cross Sections label, e.g.:

- If label text contains “Cross Sections”, set `subchild.config(text=self.t("Cross Sections Folder:"))`.

**Files:** `src/gui/main_gui.py`

---

## 6. First Run dialog (optional)

- **Option A – No First Run change:** Cross Sections path is config-only; user sets it in Settings (Shared Folder Settings) after first run. No folder created by first run.
- **Option B – Default folder under shared base:** Add `"cross_sections": "Cross Sections"` to `REQUIRED_FOLDERS` in first_run_dialog and in `_get_folder_paths` add:
  - `"shared_folder_cross_sections": str(backup_path / self.REQUIRED_FOLDERS["cross_sections"])`  
  so that when user chooses a shared/backup path, the default Cross Sections folder is created and stored in config.

Recommendation: **Option B** so new users get a consistent default under the shared folder; existing users can still change it via Browse in main GUI.

**Files:** `src/gui/first_run_dialog.py`  
- Add `"cross_sections": "Cross Sections"` to both `REQUIRED_FOLDERS` dicts (and ensure `create_folder_structure` creates it if it uses REQUIRED_FOLDERS).
- In `_get_folder_paths`, inside the `if backup_path:` block, add  
  `"shared_folder_cross_sections": str(backup_path / self.REQUIRED_FOLDERS["cross_sections"])`.

GeoVue’s `file_manager.create_folder_structure(base_path)` creates one folder per entry in `FOLDER_NAMES`. So once `FOLDER_NAMES["cross_sections"] = "Cross Sections"` is added in file_manager (step 2), the shared Cross Sections folder will be created automatically when first run calls `create_folder_structure(backup_path)`. No separate creation list is needed.

---

## 7. Translations (optional)

If GeoVue uses `translations.csv` for UI strings, add a row for “Cross Sections Folder:” so it can be translated.

**Files:** `src/resources/translations.csv` (if applicable)

---

## 8. Checklist summary

| # | Task | File(s) |
|---|------|--------|
| 1 | Add `shared_folder_cross_sections` to USER_SETTINGS_KEYS and DEFAULT_CONFIG | config_manager.py |
| 2 | Add FOLDER_NAMES["cross_sections"], shared_paths["cross_sections"], config_key_map, config set in initialize_shared_paths | file_manager.py |
| 3 | Add cross_sections_path_var, initial value, _create_shared_path_field for Cross Sections | main_gui.py |
| 4 | Add cross_sections block in setup_shared_paths | main_gui.py |
| 5 | Add Cross Sections label translation update | main_gui.py |
| 6 | Add REQUIRED_FOLDERS["cross_sections"] and shared_folder_cross_sections in _get_folder_paths; ensure folder is created in create_folder_structure | first_run_dialog.py, file_manager.py |
| 7 | (Optional) Add translation for “Cross Sections Folder:” | translations.csv |

---

## 9. Testing

- First run: Choose shared path → confirm “Cross Sections” folder is created and config has `shared_folder_cross_sections`.
- Settings: Open Shared Folder Settings → change Cross Sections path via Browse → save/restart → path persists.
- FileManager: After init, `get_shared_path("cross_sections")` returns the configured or default path; `update_shared_path("cross_sections", new_path)` updates config and shared_paths.

After this, the Section Tool (or any importer) can read `config_manager.get("shared_folder_cross_sections")` or `file_manager.get_shared_path("cross_sections")` to get the folder of section PDFs.
