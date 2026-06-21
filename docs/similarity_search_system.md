# Logging Review Similarity Search

This document describes the GeoVue similarity system used by the Logging Review dialog. The goal is to let a reviewer choose one chip-tray compartment image and reload the grid with visually, chemically, spatially, or geologically continuous matches while preserving the normal Logging Review filters.

## User Workflow

1. Build or refresh the visual embedding database with `python -m ml_pipeline.build_similarity_index build --image-root <compartment-image-root> --db ml_output\image_similarity\geovue_embeddings.sqlite`.
2. Open Logging Review and load the normal review image set.
3. Optionally select exactly one grid image first. If one image is selected, that image is used immediately after the options dialog.
4. Click `Search Similar...` in the `Similarity Search` panel.
5. Choose the search mode and options in the compact dialog. If no single image is selected, press `OK` and then click the grid image that should seed the search.
6. The similarity ranking runs across all loaded review images, then the existing Logging Review filters are applied to the ranked result list.
7. Click `Clear Similarity` to cancel a pending query pick or remove the active similarity layer while leaving the rest of the filter state intact.

If no filters are active, the similarity layer can rank all loaded review images that have the required signal. Visual search can query the full SQLite embedding index, then the dialog displays the ranked hits that are currently part of the Logging Review image set.

## Advanced Filter Embedding Projection

The Advanced Filter Window has an `embedding projection` plot type when the visual embedding database is configured. It loads the existing SQLite embeddings, builds or reuses a cached two-dimensional projection, merges `umap_x` and `umap_y` back onto the interval DataFrame, and then uses the same lasso-selection callback as the normal scatter plot.

The projection uses UMAP when `umap-learn` is available and falls back to deterministic PCA otherwise. Projection cache files are generated beside the embedding database under `ml_output/` by default and must remain local/generated artifacts rather than git-tracked source.

## Modes

### Visual Only

Visual similarity uses image embeddings stored in `ml_output/image_similarity/geovue_embeddings.sqlite`. The current builder writes MobileNetV3-small embeddings, and the viewer ranks candidates by cosine similarity to the selected or browsed query image.

This answers: "Which compartments look like this photo?" It does not know assay values, drillhole position, or geological continuity unless those are included by another mode.

### Chemical Only

Chemical similarity uses assay/geochemistry vectors from the merged interval row already available to Logging Review. Column names are resolved through `ColumnResolver`, so standard names such as `fe_pct`, `sio2_pct`, and `al2o3_pct` can match Snowflake or CSV columns such as `Fe_pct_BEST` and `SiO2_pct_BEST`.

The current implementation uses:

- centered log-ratio transform for compositional assay data;
- robust median/IQR scaling so high-variance elements do not dominate by accident;
- weighted masked distance so rows with missing elements can still participate when enough requested chemistry is present;
- a minimum coverage threshold to avoid ranking a sample from one shared element only.

This answers: "Which compartments assay like this interval?" It deliberately remains separate from visual similarity so lookalike chips and chemically similar intervals can be compared rather than conflated.

### Spatial Only

Spatial similarity prefers surveyed 3D proximity when collar and survey data are loaded. `DataCoordinator.get_trace_for_hole()` builds a trace from `collar_sens_gab`/legacy collar sources and `collar_survey`/legacy survey sources, then `survey_trace.xyz_at_depth()` interpolates XYZ at the compartment midpoint.

If XYZ is unavailable, the system falls back to same-hole measured-depth proximity. Other-hole depth proximity is not treated as spatial similarity unless both intervals have XYZ.

This answers: "Which compartments are physically close to this interval?"

### Hybrid

Hybrid combines visual, chemical, and spatial signals. The default weights are intentionally conservative:

- visual: 0.45
- chemical: 0.45
- spatial: 0.10

Spatial is treated as a plausibility prior rather than a hard gate. This keeps visually and chemically strong matches from being hidden simply because collar/survey data is missing or still loading.

### Continuity

Continuity adds a same-hole local-neighbourhood score. For each candidate, the ranker looks at nearby intervals in the same hole and averages their query-relative visual and chemical similarity. This rewards matches that form a short coherent run rather than a single isolated compartment.

This answers: "Where does this visual/chemical signature continue downhole or recur as a short local trend?"

## Data Sources And Reality Checks

The implementation is tied to real GeoVue data hooks rather than assumed tables:

- Visual data: `ml_pipeline.embedding_store.SQLiteEmbeddingStore` and `ml_pipeline.build_similarity_index`.
- Logging Review candidates: the currently loaded `CompartmentImage` objects in `src/gui/logging_review_dialog.py`.
- Assay/geochemistry values: `CompartmentImage.csv_data`, hydrated through Logging Review's existing `_get_csv_data_cached()` path when needed.
- Column aliases: `src/processing/DataManager/column_aliases.py`.
- Collar data: `DataCoordinator.get_collar_data()` prefers `collar_sens_gab`, which maps to Snowflake `AA_EXPLORATION_AFR.SENS_GAB.COLLAR`.
- Survey data: `DataCoordinator.get_survey_data()` prefers `collar_survey`, which maps to Snowflake `AA_EXPLORATION_AFR.SENS_GAB.COLLAR_SURVEY`.
- Trace math: `src/processing/DataManager/survey_trace.py` uses straight segments or minimum curvature depending on the available survey stations.

The ranker degrades cleanly when data is missing. For example, hybrid search still works with chemical and spatial scores if visual embeddings are unavailable, and spatial search falls back to same-hole depth only when XYZ cannot be built.

## Why There Is No 3D Viewer In Logging Review

A 3D component would be useful later for QA and explanation, but it is not required for first-class spatial similarity. The current Logging Review dialog is already a dense classification and filtering workspace. Adding a 3D viewport there would increase interaction cost and layout risk.

The implemented design keeps 3D in the model layer:

- collar/survey traces provide XYZ coordinates;
- XYZ distance contributes to ranking;
- the grid remains the familiar review surface;
- a later drillhole/correlation view can visualize the same ranked intervals in 3D without changing the ranking logic.

A future 3D view should be launched from a selected similarity result or from Drillhole Correlation, not embedded permanently in the Logging Review filter panel.

## Main Files

- `ml_pipeline/similarity_search.py`: backend ranking engine and explainable result model.
- `src/gui/similarity_search_options_dialog.py`: compact Tk options dialog.
- `src/gui/logging_review_dialog.py`: button, worker-thread orchestration, filter integration, and grid reload.
- `tests/test_similarity_search.py`: focused tests for CLR chemistry, aliasing, missing-data coverage, spatial fallback, and continuity.

## Assumptions

- A selected grid image represents a real interval with `hole_id`, `depth_from`, `depth_to`, and image path metadata.
- Chemical similarity is meaningful only across columns chosen for the current search and only when enough shared chemistry is present.
- Surveyed XYZ is preferred over measured depth, but missing collar/survey data should not make the rest of the search unusable.
- The similarity layer must run after the existing Logging Review filters so the user's current project, hole, classification, moisture, chemistry, or tag filters remain authoritative.
- Generated embedding databases and image outputs are local/shared data products and should not be committed to git.

## Future Work

- Add saved presets for chemistry columns and weights by commodity or review task.
- Surface per-result explanation text in the grid card or a side detail panel.
- Add an export of ranked results with component scores for audit and reporting.
- Add cross-section or 3D visualization from Drillhole Correlation for ranked spatial hits.
- Validate the default weights against reviewed examples and known logged domains.
