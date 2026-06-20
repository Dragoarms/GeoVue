# Televiewer Web Viewer

This folder preserves the source-only WebGL televiewer prototype for integration into GeoVue.

The original scratch prototype lives under ignored `tfd_extract_preview/` and includes generated BA0007 images, JSON, CSV, and screenshots. Those generated artifacts should stay out of git. This folder should only contain reusable viewer source and, later, the GeoVue-facing data bridge.

Current source files:

- `index.html`
- `app.js`
- `styles.css`

Expected future data contract:

- `viewer_data.json`: hole trace, OTV slice manifest, geophysics, assays, mineralogy, collar/survey context
- `chip_tray_manifest.json`: optional RC chip tray interval images

The production integration should generate those manifests from `Televiewer Datasets/<PROJECT>/<HOLE>/processed/` and existing GeoVue data stores, not from checked-in sample data.


## Launcher URL Contract

GeoVue can launch the viewer with query parameters so the same WebGL source works for any processed hole:

- `holeId`: display id and segment filename prefix.
- `dataUrl`: JSON bundle with collar, survey, trace, geophysics, assays, mineralogy, and comparison rows.
- `chipTrayManifestUrl`: optional chip tray manifest JSON.
- `rawDir`: folder containing `<HOLE>_<START>_<END>m_raw.jpg` files.
- `resampledDir`: folder containing `<HOLE>_<START>_<END>m_resampled.jpg` files.
- `firstMeter` / `maxDepthMeter`: optional depth bounds.
- `start`, `range`, `geometry`, and `chip`: initial viewport state.
- `annotationsUrl`: optional JSON endpoint for existing plane/observation annotations.
- `annotationsSaveUrl`: optional writable JSON endpoint; when present, viewer edits are saved back through GeoVue's register manager.