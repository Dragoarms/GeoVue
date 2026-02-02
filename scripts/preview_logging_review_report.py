# python "c:/Users/georg/OneDrive/Home Python/GeoVue 26_01_27/scripts/preview_logging_review_report.py" "C:/Users/georg/Downloads/reports/New folder (4)"

"""
Regenerate the Logging Review HTML report from exported _datasets CSVs and optionally open it.

Use this to quickly see report layout changes without launching GeoVue or re-running
full data prep. After generating a report once from GeoVue (which creates output_dir/_datasets/),
run this script to regenerate HTML from those CSVs and open it in your browser.

Usage:
  # From repo root, with venv active:
  python scripts/preview_logging_review_report.py <output_dir> [--collar <collar.csv>] [--open]

  output_dir   Path to folder containing _datasets/ (e.g. C:/Users/.../reports/New folder (4))
  --collar     Optional path to collar CSV (hole id + easting/northing) for map
  --open       Open the generated HTML in the default browser (default: True)
  --no-open    Do not open the browser

Example:
  python scripts/preview_logging_review_report.py "C:/Users/georg/Downloads/reports/New folder (4)" --collar collars.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import webbrowser

# Add src so that "processing" package is importable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pandas as pd

from processing.DataManager.column_aliases import ColumnResolver
from processing.logging_review_report import (
    compute_hybrid_outlier_scores,
    resolve_chemistry_columns,
    resolve_drilldate_column,
    resolve_logger_column,
)
from processing.logging_review_html_report import generate_logger_html_reports_from_prepped_data


def _find_csv(path: str, pattern: str) -> str | None:
    """Return path to first file under path that matches pattern (e.g. 01_*.csv)."""
    if not os.path.isdir(path):
        return None
    for name in os.listdir(path):
        if pattern in name and name.endswith(".csv"):
            return os.path.join(path, name)
    return None


def load_datasets(datasets_dir: str):
    """Load logging, merged, and optional full_team CSVs from _datasets dir."""
    logging_path = _find_csv(datasets_dir, "01_")
    merged_path = _find_csv(datasets_dir, "02_")
    team_path = _find_csv(datasets_dir, "03_")
    if not logging_path or not merged_path:
        raise FileNotFoundError(
            f"Expected 01_*.csv and 02_*.csv in {datasets_dir}. "
            "Generate a report once from GeoVue to create _datasets."
        )
    logging_df = pd.read_csv(logging_path)
    merged_df = pd.read_csv(merged_path)
    full_team_df = pd.read_csv(team_path) if team_path else merged_df.copy()
    return logging_df, merged_df, full_team_df


def load_collar(collar_path: str) -> pd.DataFrame:
    """Load collar CSV (must have hole id and easting/northing columns)."""
    return pd.read_csv(collar_path)


def resolve_columns(merged_df: pd.DataFrame, logging_df: pd.DataFrame, collar_df: pd.DataFrame | None):
    """Resolve column names and return dict for report generation."""
    resolver = ColumnResolver(merged_df)
    hole_col = resolver.get("hole_id")
    depth_from_col = resolver.get("depth_from")
    depth_to_col = resolver.get("depth_to")
    strat_col = resolver.get("strat")
    if not hole_col or not depth_to_col or not strat_col:
        raise ValueError(
            "Merged CSV must have hole_id, depth_to, and strat-like columns. "
            f"Found columns: {list(merged_df.columns)}"
        )
    logger_col = resolve_logger_column(logging_df) or resolve_logger_column(merged_df)
    if not logger_col:
        raise ValueError("No logger column found in logging or merged CSV.")
    chem_cols = resolve_chemistry_columns(merged_df)
    chem_actual_cols = list(chem_cols.values())
    drilldate_col = resolve_drilldate_column(collar_df) if collar_df is not None and not collar_df.empty else None
    return {
        "logger_col": logger_col,
        "hole_col": hole_col,
        "depth_from_col": depth_from_col,
        "depth_to_col": depth_to_col,
        "strat_col": strat_col,
        "chem_actual_cols": chem_actual_cols,
        "drilldate_col": drilldate_col,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate Logging Review HTML from _datasets CSVs and optionally open it."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to folder containing _datasets/ (e.g. report output folder)",
    )
    parser.add_argument(
        "--collar",
        type=str,
        default=None,
        help="Optional path to collar CSV for map (hole id + easting/northing)",
    )
    parser.add_argument("--open", dest="open_browser", action="store_true", default=True)
    parser.add_argument("--no-open", dest="open_browser", action="store_false")
    args = parser.parse_args()

    input_path = os.path.abspath(args.output_dir)
    
    # Allow user to provide either the _datasets folder or its parent
    if os.path.basename(input_path) == "_datasets":
        datasets_dir = input_path
        output_dir = os.path.dirname(input_path)
    elif os.path.isdir(os.path.join(input_path, "_datasets")):
        datasets_dir = os.path.join(input_path, "_datasets")
        output_dir = input_path
    else:
        print(f"Error: _datasets not found at {input_path} or {os.path.join(input_path, '_datasets')}", file=sys.stderr)
        return 1

    try:
        logging_df, merged_df, full_team_df = load_datasets(datasets_dir)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    collar_df = load_collar(args.collar) if args.collar and os.path.isfile(args.collar) else pd.DataFrame()
    if collar_df.empty and (args.collar is None or not os.path.isfile(args.collar)):
        print("Tip: use --collar <path_to_collar.csv> for map coordinates and date range in the report title.")
    try:
        cols = resolve_columns(merged_df, logging_df, collar_df if not collar_df.empty else None)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1

    # Ensure outlier columns exist (report expects them); add if missing
    if "outlier_score" not in merged_df.columns and cols["strat_col"] in merged_df.columns:
        try:
            outlier_scores = compute_hybrid_outlier_scores(
                merged_df,
                strat_col=cols["strat_col"],
                chem_cols=cols["chem_actual_cols"],
                min_group_size=2,
            )
            merged_df = merged_df.join(outlier_scores)
        except Exception:
            merged_df["outlier_score"] = 0.0
            merged_df["outlier_reason"] = ""

    logger_col = cols["logger_col"]
    logger_values = merged_df[logger_col].dropna().astype(str).unique().tolist()
    if not logger_values:
        print("Error: No logger values in merged data.", file=sys.stderr)
        return 1
    # Preview: only first logger, open report as soon as it's ready
    logger_values = logger_values[:1]

    # Infer date range for report title (From: ... To: ...)
    # Prefer merged_df if it has a drilldate column; otherwise use collar if provided
    date_from = date_to = None
    date_col_merged = resolve_drilldate_column(merged_df)
    if date_col_merged and date_col_merged in merged_df.columns:
        try:
            dates = pd.to_datetime(merged_df[date_col_merged], errors="coerce").dropna()
            if not dates.empty:
                date_from = dates.min().strftime("%Y-%m-%d")
                date_to = dates.max().strftime("%Y-%m-%d")
        except Exception:
            pass
    if (date_from is None or date_to is None) and collar_df is not None and not collar_df.empty:
        date_col_collar = cols.get("drilldate_col") or resolve_drilldate_column(collar_df)
        if date_col_collar and date_col_collar in collar_df.columns:
            try:
                dates = pd.to_datetime(collar_df[date_col_collar], errors="coerce").dropna()
                if not dates.empty:
                    if date_from is None:
                        date_from = dates.min().strftime("%Y-%m-%d")
                    if date_to is None:
                        date_to = dates.max().strftime("%Y-%m-%d")
            except Exception:
                pass

    # Minimal stats (report uses match_rate_pct if present)
    stats = {}

    try:
        output_files = generate_logger_html_reports_from_prepped_data(
            data_coordinator=None,
            output_dir=output_dir,
            merged_df=merged_df,
            logging_df=logging_df,
            collar_df=collar_df,
            stats=stats,
            logger_col=cols["logger_col"],
            hole_col=cols["hole_col"],
            depth_from_col=cols["depth_from_col"],
            depth_to_col=cols["depth_to_col"],
            strat_col=cols["strat_col"],
            chem_actual_cols=cols["chem_actual_cols"],
            logger_values=logger_values,
            date_from=date_from,
            date_to=date_to,
            top_n=15,
            page_options={
                "summary_stats": True,
                "cover": True,
                "comment_stats": True,
                "fines_accuracy": True,
                "grouping_accuracy": True,
                "outliers": True,
            },
            include_images=False,
            logo_path=None,
            full_team_df=full_team_df,
            skip_csv_export=True,
        )
    except Exception as e:
        print(f"Report generation failed: {e}", file=sys.stderr)
        raise

    if not output_files:
        print("No report files generated.", file=sys.stderr)
        return 1

    first_html = output_files[0]
    print(f"Generated: {first_html}")
    if args.open_browser:
        webbrowser.open(f"file:///{first_html.replace(os.sep, '/')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
