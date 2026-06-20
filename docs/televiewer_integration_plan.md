# Televiewer Integration Plan

This note captures the current OTV prototype direction and the first-pass GeoVue integration shape.

## Recommended Workflow

Use a dedicated Televiewer workflow, with hooks into existing GeoVue areas:

- `Process Images`: `Process Televiewer Data (.tfd)` ingests raw TFD files and builds reusable processed assets.
- `Image Analysis and Review`: `Televiewer Viewer` opens processed holes in the preserved web viewer.
- `Drillhole Correlation`: later, add a launcher/context action so selected correlation holes can open in the Televiewer viewer with the same hole/depth context.

The Televiewer UI is close to drillhole correlation, but it is not just another strip column. It needs 3D trace context, wrapped/flat switching, structural plane interpretation, chip photos, graph tracks, and eventually multi-hole spatial relationships. Keeping it as its own workflow is cleaner, while still reusing correlation data services, colour maps, trace/survey data, and selected-hole context.

## Shared Folder Structure

Under the configured GeoVue shared folder:

```text
Televiewer Datasets/
  <PROJECT_CODE>/
    <HOLE_ID>/
      raw/
        <source>.tfd                 # optional copy only
        source_metadata.json
      processed/
        manifest.json                # authoritative processed dataset index
        viewer_data.json             # web viewer bundle
        chip_tray_manifest.json      # references existing approved chip photos
        records.csv
        meter_segments_manifest.csv
        extraction_summary.json
        raw_by_record/
        slices_1m/
        thumbnails/
        full_strip/
      qc/
        stretched_zone_report.csv
        tfd_vs_survey_comparison.csv # future QC output
      annotations/
        planes.json
        observations.json
```

The code should treat `processed/manifest.json` as the authoritative index. The TFD only needs to be decoded once unless the raw file, processing version, depth calibration, or orientation assumptions change.

## Processing Outputs

The first-pass processor now produces or prepares:

- depth-referenced 1 m image slices with a configurable pixels-per-metre resampling rate
- raw-by-record image strips for QC and fallback display
- `viewer_data.json` for the web viewer
- a compact manifest with coverage, telemetry field names, output locations, and source metadata
- accumulated `raw/source_metadata.json` records for every TFD selected for the hole, including non-image tool files such as ATV/caliper/e-log/magsus
- extracted APS544-style downhole telemetry where present: roll, magnetic roll, tilt, azimuth, gravity, magnetic field, and taps
- ABI40 ATV prototype extraction from numeric amplitude scanlines where JPEG OTV strips are not present
- mechanical caliper TFD extraction to `caliper_diameter.csv`, using the tool calibration points and `ToolOD + 2 * arm_mm` as the borehole diameter estimate
- QC reporting for stretched/pixelated depth zones
- optional raw TFD copy into shared storage, disabled by default
- enrichment hooks for collar, survey, trace, geophysics, assays, normative mineralogy, and approved chip tray photos

## Code Shape

Current modules:

```text
src/processing/Televiewer/
  __init__.py
  enrichment.py
  manifest.py
  paths.py
  atv_decoder.py
  caliper_decoder.py
  tfd_decoder.py
  tfd_processor.py

src/gui/Televiewer/
  __init__.py
  local_server.py
  televiewer_process_dialog.py
  televiewer_viewer_dialog.py
  web/
    index.html
    app.js
    styles.css
    README.md
```

`FileManager` owns the shared path key:

- folder name: `Televiewer Datasets`
- internal key: `televiewer_datasets`
- config key: `shared_folder_televiewer_datasets`

## Viewer Architecture

Keep the current WebGL viewer as a web surface for now. GeoVue is Tkinter-based, and porting this interaction to Tk Canvas would be expensive and lower quality. A small local server mounts:

- the static viewer source
- the selected processed dataset folder
- external chip tray photo roots at launch time
- virtual JSON payloads for launch-time rewritten chip manifests

The viewer should support two launch modes:

- single-hole interpretation: strongest mode for plane picking and OTV review
- multi-hole spatial comparison: show traces together, allow selecting a trace/depth to drive the main detailed viewer

## Current Integration Checkpoint

Implemented in GeoVue so far:

- `FileManager` and `ConfigManager` know about the shared `Televiewer Datasets` folder.
- The Process Images tab has a `Process Televiewer Data (.tfd)` entry point.
- The processor dialog is now a compact file/folder picker: selecting a folder recursively finds `.tfd` files, registers every source file, decodes compatible OTV image records, and leaves unsupported/non-image TFDs as provenance without aborting the batch.
- The Image Analysis and Review tab has a `Televiewer Viewer` launcher.
- Compact toolbar entries exist for processing and viewing Televiewer datasets.
- Drillhole Correlation has an `Open Televiewer` bridge plus a drillhole-column right-click `Open Televiewer Here...` action that hands hole/depth context to the Televiewer launcher.
- `src/processing/Televiewer` prepares project/hole folders and can decode BA0007-style TFD JPEG strip records into records CSV, 1 m raw/resampled slices, QC reports, viewer data, and manifests without copying large raw files unless requested.
- ABI40 ATV files can now fall back to a numeric-image extractor when no OTV JPEG strips are present. NB0264 ATV decodes as an amplitude candidate from 24-69 m at 0.004 m row spacing.
- Caliper 7006 TFD files are parsed into `caliper_diameter.csv` and injected into `viewer_data.json` as a `CALIPER_MM` geophysics curve when viewer data exists.
- The processor can enrich decoded viewer data from GeoVue's DataCoordinator when it is initialized.
- Televiewer plane/observation annotations are persisted through `JSONRegisterManager` in the shared `televiewer_annotations.json` register and served to the WebGL viewer over a local JSON route.
- Approved chip tray images are referenced by manifest instead of copied; the launcher rewrites those file references to localhost URLs for the browser.
- `src/gui/Televiewer/web` preserves the WebGL prototype and accepts launch-time dataset URLs.
- A small local HTTP server mounts the viewer source plus selected dataset folders on localhost for browser display.

Still to build:

- a richer Drillhole Correlation bridge that can pass multiple selected holes and viewport extents into the Televiewer overview
- multi-hole overview/comparison mode with colour-coordinated viewport extents
- richer per-plane metadata editing, export, and review/audit flows for saved annotations
- production QC for TFD telemetry versus collar survey trace agreement
- broader decoder validation on additional vendor TFD variants beyond the BA0007 file structure; NB0264 ATV `Uplog@68.45m.tfd` currently has no BA0007-style JPEG strip records and needs separate decoder support
- integration with GeoVue colour-map/config managers for graph and assay styling

## Preservation Notes

The prototype source has been preserved separately from generated images/data. Generated OTV slices, screenshots, JSON exports, CSV comparisons, parquets, and chip-tray images should stay out of git and live in shared/local data storage.
