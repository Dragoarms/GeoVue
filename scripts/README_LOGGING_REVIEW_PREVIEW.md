# Logging Review Report Preview

Regenerate the HTML Logging Review report from exported `_datasets` CSVs **without launching GeoVue**.

## Why use this?

After you generate a report once from GeoVue, the output folder contains `_datasets/` with:

- `01_logging_intervals.csv`
- `02_merged_assay_intervals.csv`
- `03_full_team_data.csv` (optional)

When you change report layout (e.g. overview header, date format, map reprojection), you can **regenerate the HTML from these CSVs** and open it in your browser instead of:

1. Launching GeoVue  
2. Loading project and datasets  
3. Running “Generate report” for a logger  
4. Opening the HTML

## Usage

From the **repo root** with your venv active:

```bash
# Regenerate report and open in browser (default)
python scripts/preview_logging_review_report.py "C:\path\to\report\output\folder"

# With collar CSV for map (easting/northing → WGS84)
python scripts/preview_logging_review_report.py "C:\path\to\report\output" --collar "C:\path\to\collar.csv"

# Generate only, do not open browser
python scripts/preview_logging_review_report.py "C:\path\to\report\output" --no-open
```

- **output_dir**: Folder that contains `_datasets/` (e.g. the folder where `RC_Logging_Review_AAB.html` was generated).
- **--collar**: Optional path to a collar CSV with hole id and easting/northing columns. Used for: (1) overview map coordinates (reprojection to WGS84 if needed), (2) report title date range (From: ... To: ...) when merged data has no drilldate column. Without it you may see "No collar coordinates available for map" and "From: - To: -" in the title.
- **--open** / **--no-open**: Whether to open the first generated HTML in the default browser (default: open).

## Requirements

- Same Python env as GeoVue (e.g. `pip install -e .` from repo root).
- `pyproj` installed for map reprojection (UTM → WGS84): `pip install pyproj`.

## Tests

To run report-related tests (including HTML report logic) without the full app:

```bash
pytest src/processing/tests/test_logging_review_report.py -v
```
