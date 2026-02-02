"""
test_data_joining.py - Standalone test for debugging data joining between sources.

This script:
1. Initializes DataCoordinator without launching GeoVue
2. Lists all available data sources and their columns
3. Compares schema column names vs actual DataFrame column names
4. Tests joining columns from different sources (e.g., exgeologyRC with exassay)
5. Handles interval matching (different from/to depths between sources)
6. Provides extensive debugging output

Run directly from terminal:
    python -m tests.test_data_joining

Or:
    cd src
    python tests/test_data_joining.py

Author: George Symonds / Claude
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set

# Add src to path for imports
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION - EDIT THESE PATHS TO MATCH YOUR DATA LOCATION
# =============================================================================

# Default datasets directory - update this path to your actual data location
# This path was auto-detected from GeoVue settings.json
DEFAULT_DATASETS_DIR = r"C:\Users\georg\Pictures\Shared folder EX\Drillhole Datasets"

# Environment variable override
DATASETS_DIR = os.environ.get('GEOVUE_DATASETS_DIR', DEFAULT_DATASETS_DIR)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_separator(title: str = "", char: str = "=", width: int = 80):
    """Print a separator line with optional title."""
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"\n{char * padding} {title} {char * padding}")
    else:
        print(char * width)


def print_dataframe_info(df: pd.DataFrame, name: str, max_rows: int = 5):
    """Print detailed info about a DataFrame."""
    print(f"\n  DataFrame: {name}")
    print(f"  Shape: {df.shape} (rows, columns)")
    print(f"  Columns ({len(df.columns)}): {list(df.columns)[:15]}{'...' if len(df.columns) > 15 else ''}")

    if not df.empty:
        print(f"  Sample data (first {min(max_rows, len(df))} rows):")
        print(df.head(max_rows).to_string(index=False, max_cols=10))


def detect_key_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Auto-detect hole_id, from, and to column names in a DataFrame.

    Returns dict with keys: 'hole_id', 'from', 'to'
    """
    cols_lower = {c.lower(): c for c in df.columns}

    result = {'hole_id': None, 'from': None, 'to': None}

    # Hole ID patterns
    for pattern in ['holeid', 'hole_id', 'bhid', 'drillhole_id']:
        if pattern in cols_lower:
            result['hole_id'] = cols_lower[pattern]
            break

    # From depth patterns
    for pattern in ['sampfrom', 'geolfrom', 'from', 'depth_from', 'from_depth']:
        if pattern in cols_lower:
            result['from'] = cols_lower[pattern]
            break

    # To depth patterns
    for pattern in ['sampto', 'geolto', 'to', 'depth_to', 'to_depth']:
        if pattern in cols_lower:
            result['to'] = cols_lower[pattern]
            break

    return result


# =============================================================================
# DATA COORDINATOR INITIALIZATION
# =============================================================================

def initialize_data_coordinator(datasets_dir: str) -> Optional['DataCoordinator']:
    """
    Initialize DataCoordinator with CSV files from the specified directory.

    Args:
        datasets_dir: Path to directory containing CSV files

    Returns:
        Initialized DataCoordinator or None if failed
    """
    from processing.DataManager.data_coordinator import DataCoordinator

    print_separator("INITIALIZING DATA COORDINATOR")
    print(f"Datasets directory: {datasets_dir}")

    if not os.path.exists(datasets_dir):
        print(f"ERROR: Datasets directory does not exist: {datasets_dir}")
        print("\nPlease update DATASETS_DIR in this script or set GEOVUE_DATASETS_DIR environment variable")
        return None

    # Find all CSV files
    csv_files = []
    print(f"\nScanning for CSV files...")
    for file in os.listdir(datasets_dir):
        if file.lower().endswith('.csv'):
            file_path = os.path.join(datasets_dir, file)
            csv_files.append(file_path)
            print(f"  Found: {file}")

    if not csv_files:
        print(f"ERROR: No CSV files found in {datasets_dir}")
        return None

    print(f"\nTotal CSV files found: {len(csv_files)}")

    # Create and initialize DataCoordinator
    coordinator = DataCoordinator(config_manager=None, file_manager=None)

    print("\nInitializing DataCoordinator...")
    try:
        coordinator.initialize(
            compartment_folders=[],  # No image folders for this test
            original_folder=None,
            csv_files=csv_files,
            json_manager=None
        )
        print("DataCoordinator initialized successfully!")
        return coordinator

    except Exception as e:
        print(f"ERROR initializing DataCoordinator: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# DEBUG: LIST ALL DATA SOURCES
# =============================================================================

def debug_data_sources(coordinator: 'DataCoordinator'):
    """List all loaded data sources with their properties."""
    print_separator("LOADED DATA SOURCES")

    geological_store = coordinator.geological_store
    sources = geological_store.list_sources()

    print(f"\nTotal sources loaded: {len(sources)}")
    print(f"Source names: {sources}")

    for source_name in sources:
        print_separator(f"Source: {source_name}", char="-")

        source = geological_store.get_source(source_name)
        if source is None:
            print(f"  ERROR: Could not get source")
            continue

        print(f"  Is loaded: {source.is_loaded}")
        print(f"  Row count: {source.row_count:,}")

        # Get schema info
        schema = geological_store.get_schema(source_name)
        if schema:
            print(f"  Schema info:")
            print(f"    Dataset type: {getattr(schema, 'dataset_type', 'N/A')}")
            print(f"    Hole ID column (schema): {schema.hole_id_column}")
            print(f"    From column (schema): {schema.from_column}")
            print(f"    To column (schema): {schema.to_column}")

        # Get actual DataFrame columns
        if source.is_loaded and source.df is not None:
            df = source.df
            print(f"  Actual DataFrame columns ({len(df.columns)}):")

            # Show all columns grouped by likely type
            all_cols = list(df.columns)
            print(f"    All: {all_cols[:30]}{'...' if len(all_cols) > 30 else ''}")

            # Detect key columns from actual data
            detected = detect_key_columns(df)
            print(f"  Auto-detected key columns:")
            print(f"    hole_id: {detected['hole_id']}")
            print(f"    from: {detected['from']}")
            print(f"    to: {detected['to']}")

            # Show unique holes count
            hole_col = detected['hole_id']
            if hole_col:
                unique_holes = df[hole_col].nunique()
                print(f"  Unique holes: {unique_holes:,}")
                print(f"  Sample holes: {list(df[hole_col].dropna().unique()[:5])}")


# =============================================================================
# DEBUG: COMPARE SCHEMA VS ACTUAL COLUMNS
# =============================================================================

def debug_schema_vs_actual(coordinator: 'DataCoordinator'):
    """Compare schema column names with actual DataFrame column names."""
    print_separator("SCHEMA VS ACTUAL COLUMN NAMES")

    geological_store = coordinator.geological_store

    # Get available columns from schema (what get_available_columns returns)
    all_schema_columns = geological_store.get_available_columns()

    print(f"\nColumns from schema (via get_available_columns):")
    for source_name, columns in all_schema_columns.items():
        print(f"\n  Source '{source_name}':")
        col_names = [name for name, dtype in columns]
        print(f"    Schema columns ({len(col_names)}): {col_names[:10]}{'...' if len(col_names) > 10 else ''}")

        # Get actual columns from DataFrame
        source = geological_store.get_source(source_name)
        if source and source.is_loaded and source.df is not None:
            actual_cols = list(source.df.columns)
            print(f"    Actual columns ({len(actual_cols)}): {actual_cols[:10]}{'...' if len(actual_cols) > 10 else ''}")

            # Find mismatches
            schema_set = set(c.lower() for c in col_names)
            actual_set = set(c.lower() for c in actual_cols)

            in_schema_not_actual = schema_set - actual_set
            in_actual_not_schema = actual_set - schema_set

            if in_schema_not_actual:
                print(f"    IN SCHEMA BUT NOT IN ACTUAL: {in_schema_not_actual}")
            if in_actual_not_schema:
                internal = {c for c in in_actual_not_schema if c.startswith('_')}
                external = in_actual_not_schema - internal
                if external:
                    print(f"    IN ACTUAL BUT NOT IN SCHEMA: {external}")


# =============================================================================
# DEBUG: NUMERIC COLUMNS FOR QAQC
# =============================================================================

def debug_numeric_columns(coordinator: 'DataCoordinator'):
    """List numeric columns available for statistical analysis."""
    from processing.DataManager.schema import DataType

    print_separator("NUMERIC COLUMNS FOR QAQC ANALYSIS")

    geological_store = coordinator.geological_store
    all_columns = geological_store.get_available_columns()

    print("\nNumeric columns by source:")

    all_numeric = []
    for source_name, columns in all_columns.items():
        numeric_cols = [(name, dtype) for name, dtype in columns if dtype == DataType.NUMERIC]
        print(f"\n  Source '{source_name}' ({len(numeric_cols)} numeric columns):")

        for col_name, dtype in numeric_cols[:10]:
            print(f"    - {col_name}")
            all_numeric.append((source_name, col_name))

        if len(numeric_cols) > 10:
            print(f"    ... and {len(numeric_cols) - 10} more")

    print(f"\nTotal numeric columns across all sources: {len(all_numeric)}")

    # Now check if these columns actually exist in DataFrames
    print("\n" + "-" * 60)
    print("CHECKING IF SCHEMA COLUMNS EXIST IN ACTUAL DATAFRAMES:")

    for source_name, col_name in all_numeric[:20]:
        source = geological_store.get_source(source_name)
        if source and source.is_loaded and source.df is not None:
            col_lower = col_name.lower()
            exists = col_lower in [c.lower() for c in source.df.columns]
            status = "OK" if exists else "NOT FOUND"
            if not exists:
                print(f"  {source_name}.{col_name}: {status}")


# =============================================================================
# TEST: JOIN DATA FROM MULTIPLE SOURCES
# =============================================================================

def test_data_joining(coordinator: 'DataCoordinator'):
    """Test joining data from multiple sources on common intervals."""
    print_separator("TESTING DATA JOINING")

    geological_store = coordinator.geological_store
    sources = geological_store.list_sources()

    # Find which sources have interval data (from/to columns)
    interval_sources = {}
    for source_name in sources:
        source = geological_store.get_source(source_name)
        if source and source.is_loaded and source.df is not None:
            detected = detect_key_columns(source.df)
            if detected['hole_id'] and detected['from'] and detected['to']:
                df = source.df
                from_col = detected['from']
                to_col = detected['to']

                # Calculate interval sizes
                intervals = pd.to_numeric(df[to_col], errors='coerce') - pd.to_numeric(df[from_col], errors='coerce')
                intervals = intervals.dropna()

                interval_sources[source_name] = {
                    'df': df,
                    'hole_col': detected['hole_id'],
                    'from_col': from_col,
                    'to_col': to_col,
                    'rows': len(df),
                    'interval_mean': intervals.mean() if len(intervals) > 0 else 0,
                    'interval_min': intervals.min() if len(intervals) > 0 else 0,
                    'interval_max': intervals.max() if len(intervals) > 0 else 0,
                }

    print(f"\nInterval-based sources found: {list(interval_sources.keys())}")
    print("\n  INTERVAL SIZE SUMMARY:")
    for name, info in interval_sources.items():
        print(f"    {name}: {info['rows']:,} rows, interval size: mean={info['interval_mean']:.2f}m, min={info['interval_min']:.2f}m, max={info['interval_max']:.2f}m")

    if len(interval_sources) < 2:
        print("\nNeed at least 2 interval sources to test joining")
        return

    # Test joining first two sources
    source_names = list(interval_sources.keys())
    src1_name, src2_name = source_names[0], source_names[1]
    src1, src2 = interval_sources[src1_name], interval_sources[src2_name]

    print_separator(f"JOINING: {src1_name} + {src2_name}", char="-")

    # Get DataFrames
    df1, df2 = src1['df'].copy(), src2['df'].copy()

    # Normalize hole IDs to uppercase
    df1['_hole_upper'] = df1[src1['hole_col']].astype(str).str.strip().str.upper()
    df2['_hole_upper'] = df2[src2['hole_col']].astype(str).str.strip().str.upper()

    # Find common holes
    holes1 = set(df1['_hole_upper'].unique())
    holes2 = set(df2['_hole_upper'].unique())
    common_holes = holes1 & holes2

    print(f"\n  Holes in {src1_name}: {len(holes1):,}")
    print(f"  Holes in {src2_name}: {len(holes2):,}")
    print(f"  Common holes: {len(common_holes):,}")

    if not common_holes:
        print("  ERROR: No common holes found!")
        return

    # Take a sample hole for detailed analysis
    sample_hole = list(common_holes)[0]
    print(f"\n  Sample hole for detailed analysis: {sample_hole}")

    # Get intervals for sample hole from both sources
    df1_hole = df1[df1['_hole_upper'] == sample_hole].copy()
    df2_hole = df2[df2['_hole_upper'] == sample_hole].copy()

    print(f"\n  {src1_name} intervals for {sample_hole}: {len(df1_hole)}")
    if not df1_hole.empty:
        df1_hole_sorted = df1_hole.sort_values(by=src1['from_col'])
        print(f"    Depth range: {df1_hole_sorted[src1['from_col']].min()} to {df1_hole_sorted[src1['to_col']].max()}")
        print(f"    Sample intervals (from, to):")
        for _, row in df1_hole_sorted.head(5).iterrows():
            print(f"      ({row[src1['from_col']]}, {row[src1['to_col']]})")

    print(f"\n  {src2_name} intervals for {sample_hole}: {len(df2_hole)}")
    if not df2_hole.empty:
        df2_hole_sorted = df2_hole.sort_values(by=src2['from_col'])
        print(f"    Depth range: {df2_hole_sorted[src2['from_col']].min()} to {df2_hole_sorted[src2['to_col']].max()}")
        print(f"    Sample intervals (from, to):")
        for _, row in df2_hole_sorted.head(5).iterrows():
            print(f"      ({row[src2['from_col']]}, {row[src2['to_col']]})")

    # Check interval sizes
    print(f"\n  Interval size analysis:")
    if not df1_hole.empty:
        df1_hole['_interval'] = pd.to_numeric(df1_hole[src1['to_col']], errors='coerce') - pd.to_numeric(df1_hole[src1['from_col']], errors='coerce')
        print(f"    {src1_name}: mean={df1_hole['_interval'].mean():.2f}m, min={df1_hole['_interval'].min():.2f}m, max={df1_hole['_interval'].max():.2f}m")

    if not df2_hole.empty:
        df2_hole['_interval'] = pd.to_numeric(df2_hole[src2['to_col']], errors='coerce') - pd.to_numeric(df2_hole[src2['from_col']], errors='coerce')
        print(f"    {src2_name}: mean={df2_hole['_interval'].mean():.2f}m, min={df2_hole['_interval'].min():.2f}m, max={df2_hole['_interval'].max():.2f}m")

    # Test different joining strategies
    print_separator("JOIN STRATEGY TESTS", char="-")

    # Strategy 1: Exact match on (hole_id, depth_to)
    print("\n  1. EXACT MATCH on (hole_id, depth_to):")
    df1_keyed = df1.copy()
    df2_keyed = df2.copy()
    df1_keyed['_key'] = df1_keyed['_hole_upper'] + '_' + pd.to_numeric(df1_keyed[src1['to_col']], errors='coerce').astype(int).astype(str)
    df2_keyed['_key'] = df2_keyed['_hole_upper'] + '_' + pd.to_numeric(df2_keyed[src2['to_col']], errors='coerce').astype(int).astype(str)

    keys1 = set(df1_keyed['_key'].dropna())
    keys2 = set(df2_keyed['_key'].dropna())
    exact_matches = keys1 & keys2
    print(f"     {src1_name} keys: {len(keys1):,}")
    print(f"     {src2_name} keys: {len(keys2):,}")
    print(f"     Exact matches: {len(exact_matches):,}")
    print(f"     Match rate: {len(exact_matches) / max(len(keys1), 1) * 100:.1f}%")

    # Strategy 2: Range overlap (src2 interval overlaps with src1 interval)
    print("\n  2. RANGE OVERLAP (for different interval sizes):")
    print("     This finds src2 intervals that overlap with src1 intervals")

    # For sample hole, demonstrate range matching
    if not df1_hole.empty and not df2_hole.empty:
        overlap_count = 0
        for _, row1 in df1_hole.head(10).iterrows():
            from1 = float(row1[src1['from_col']])
            to1 = float(row1[src1['to_col']])

            # Find overlapping intervals in src2
            overlaps = df2_hole[
                (pd.to_numeric(df2_hole[src2['from_col']], errors='coerce') < to1) &
                (pd.to_numeric(df2_hole[src2['to_col']], errors='coerce') > from1)
            ]

            if not overlaps.empty:
                overlap_count += 1

        print(f"     Sample: {overlap_count}/10 {src1_name} intervals have overlapping {src2_name} data")


# =============================================================================
# RANGE-BASED INTERVAL MERGING
# =============================================================================

def merge_intervals_by_range(
    primary_df: pd.DataFrame,
    secondary_df: pd.DataFrame,
    primary_keys: Dict[str, str],
    secondary_keys: Dict[str, str],
    cols_to_merge: List[str],
    sample_holes: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Merge secondary source columns into primary using range overlap.

    For each primary interval, finds the secondary interval(s) that overlap
    and assigns values based on the dominant (longest overlap) secondary interval.

    Args:
        primary_df: Primary DataFrame (e.g., exassay)
        secondary_df: Secondary DataFrame (e.g., exgeologyRC)
        primary_keys: Dict with 'hole_id', 'from', 'to' column names
        secondary_keys: Dict with 'hole_id', 'from', 'to' column names
        cols_to_merge: Columns from secondary to bring into primary
        sample_holes: Optional list of holes to process (for testing)

    Returns:
        Tuple of (merged DataFrame, stats dict)
    """
    # Normalize hole IDs
    primary_df = primary_df.copy()
    secondary_df = secondary_df.copy()

    primary_df['_hole_upper'] = primary_df[primary_keys['hole_id']].astype(str).str.upper()
    secondary_df['_hole_upper'] = secondary_df[secondary_keys['hole_id']].astype(str).str.upper()

    # Convert depths to numeric
    primary_df['_from'] = pd.to_numeric(primary_df[primary_keys['from']], errors='coerce')
    primary_df['_to'] = pd.to_numeric(primary_df[primary_keys['to']], errors='coerce')
    secondary_df['_from'] = pd.to_numeric(secondary_df[secondary_keys['from']], errors='coerce')
    secondary_df['_to'] = pd.to_numeric(secondary_df[secondary_keys['to']], errors='coerce')

    # Initialize merged columns with proper types from secondary
    for col in cols_to_merge:
        if col in secondary_df.columns:
            # Use object dtype to handle any type
            primary_df[col] = pd.Series([None] * len(primary_df), dtype='object')

    # Get holes to process
    if sample_holes:
        holes_to_process = sample_holes
    else:
        holes_to_process = primary_df['_hole_upper'].unique()

    stats = {
        'total_primary_rows': len(primary_df),
        'holes_processed': 0,
        'rows_matched': 0,
        'rows_unmatched': 0,
        'match_details': []
    }

    for hole in holes_to_process:
        primary_hole = primary_df[primary_df['_hole_upper'] == hole]
        secondary_hole = secondary_df[secondary_df['_hole_upper'] == hole]

        if secondary_hole.empty:
            stats['rows_unmatched'] += len(primary_hole)
            continue

        stats['holes_processed'] += 1

        # For each primary interval, find overlapping secondary intervals
        for idx, prow in primary_hole.iterrows():
            p_from, p_to = prow['_from'], prow['_to']

            if pd.isna(p_from) or pd.isna(p_to):
                stats['rows_unmatched'] += 1
                continue

            # Find overlapping secondary intervals
            overlaps = secondary_hole[
                (secondary_hole['_from'] < p_to) &
                (secondary_hole['_to'] > p_from)
            ]

            if overlaps.empty:
                stats['rows_unmatched'] += 1
                continue

            stats['rows_matched'] += 1

            # If multiple overlaps, find the one with maximum overlap
            if len(overlaps) > 1:
                overlaps = overlaps.copy()
                overlaps['_overlap'] = overlaps.apply(
                    lambda r: min(r['_to'], p_to) - max(r['_from'], p_from),
                    axis=1
                )
                best_match = overlaps.loc[overlaps['_overlap'].idxmax()]
            else:
                best_match = overlaps.iloc[0]

            # Copy values from secondary to primary
            for col in cols_to_merge:
                if col in best_match.index:
                    primary_df.at[idx, col] = best_match[col]

    return primary_df, stats


# =============================================================================
# TEST: MERGED ANALYSIS TABLE WITH EXPORT
# =============================================================================

def test_merged_analysis_table(coordinator: 'DataCoordinator'):
    """
    Test creating a merged table suitable for statistical analysis.

    This demonstrates the proper way to merge data from multiple sources
    and exports the result to CSV for examination.
    """
    print_separator("MERGED ANALYSIS TABLE (WITH RANGE-BASED MERGING)")

    geological_store = coordinator.geological_store
    sources = geological_store.list_sources()

    # Find sources
    primary_name = None
    secondary_name = None

    for prefer in ['exassay', 'drillhole_data']:
        if prefer.lower() in [s.lower() for s in sources]:
            primary_name = next(s for s in sources if s.lower() == prefer.lower())
            break

    for prefer in ['exgeologyrc', 'exgeologydiamond']:
        if prefer.lower() in [s.lower() for s in sources]:
            secondary_name = next(s for s in sources if s.lower() == prefer.lower())
            break

    if not primary_name or not secondary_name:
        print("ERROR: Need both exassay and exgeologyRC for this test")
        return

    print(f"\nPrimary source: {primary_name}")
    print(f"Secondary source: {secondary_name}")

    primary_source = geological_store.get_source(primary_name)
    secondary_source = geological_store.get_source(secondary_name)

    primary_df = primary_source.df.copy()
    secondary_df = secondary_source.df.copy()

    primary_keys = detect_key_columns(primary_df)
    secondary_keys = detect_key_columns(secondary_df)

    print(f"\nPrimary ({primary_name}):")
    print(f"  Rows: {len(primary_df):,}")
    print(f"  Key columns: {primary_keys}")

    print(f"\nSecondary ({secondary_name}):")
    print(f"  Rows: {len(secondary_df):,}")
    print(f"  Key columns: {secondary_keys}")

    # Show interval size comparison
    p_intervals = pd.to_numeric(primary_df[primary_keys['to']], errors='coerce') - pd.to_numeric(primary_df[primary_keys['from']], errors='coerce')
    s_intervals = pd.to_numeric(secondary_df[secondary_keys['to']], errors='coerce') - pd.to_numeric(secondary_df[secondary_keys['from']], errors='coerce')

    print(f"\n  {primary_name} interval sizes: mean={p_intervals.mean():.2f}m, min={p_intervals.min():.2f}m, max={p_intervals.max():.2f}m")
    print(f"  {secondary_name} interval sizes: mean={s_intervals.mean():.2f}m, min={s_intervals.min():.2f}m, max={s_intervals.max():.2f}m")

    # Identify columns to merge from secondary
    secondary_cols = [c for c in secondary_df.columns if not c.startswith('_')]
    primary_cols_lower = {c.lower() for c in primary_df.columns}

    # Columns unique to secondary (not in primary)
    cols_to_merge = []
    for col in secondary_cols:
        if col.lower() not in primary_cols_lower:
            cols_to_merge.append(col)

    print(f"\n  Columns unique to {secondary_name} (to merge): {cols_to_merge}")

    # Get sample holes (common between both sources)
    p_holes = set(primary_df[primary_keys['hole_id']].astype(str).str.upper().unique())
    s_holes = set(secondary_df[secondary_keys['hole_id']].astype(str).str.upper().unique())
    common_holes = list(p_holes & s_holes)[:10]  # First 10 common holes for sample

    print(f"\n  Common holes (sampling first 10): {common_holes}")

    # Perform range-based merge on sample holes
    print_separator("RANGE-BASED MERGE (sample holes)", char="-")

    merged_df, stats = merge_intervals_by_range(
        primary_df=primary_df,
        secondary_df=secondary_df,
        primary_keys=primary_keys,
        secondary_keys=secondary_keys,
        cols_to_merge=cols_to_merge,
        sample_holes=common_holes
    )

    print(f"\n  Merge Statistics:")
    print(f"    Holes processed: {stats['holes_processed']}")
    print(f"    Rows matched: {stats['rows_matched']:,}")
    print(f"    Rows unmatched: {stats['rows_unmatched']:,}")
    match_rate = stats['rows_matched'] / max(stats['rows_matched'] + stats['rows_unmatched'], 1) * 100
    print(f"    Match rate: {match_rate:.1f}%")

    # Check fill rates for merged columns
    print(f"\n  Fill rates for merged columns:")
    sample_merged = merged_df[merged_df['_hole_upper'].isin(common_holes)]
    for col in cols_to_merge[:10]:
        filled = sample_merged[col].notna().sum()
        total = len(sample_merged)
        print(f"    {col}: {filled:,}/{total:,} ({filled/max(total,1)*100:.1f}%)")

    # Export sample to CSV
    output_dir = Path(__file__).parent.parent.parent / "test_output"
    output_dir.mkdir(exist_ok=True)

    # Select columns for export
    export_cols = [
        primary_keys['hole_id'],
        primary_keys['from'],
        primary_keys['to']
    ]

    # Add some numeric columns from primary
    numeric_from_primary = []
    for col in primary_df.columns:
        if col.startswith('_'):
            continue
        try:
            if pd.api.types.is_numeric_dtype(primary_df[col]) and col not in export_cols:
                numeric_from_primary.append(col)
        except:
            pass
    export_cols.extend(numeric_from_primary[:5])  # First 5 numeric

    # Add merged columns
    export_cols.extend(cols_to_merge[:10])

    # Filter to available columns
    export_cols = [c for c in export_cols if c in sample_merged.columns]

    # Export
    export_df = sample_merged[export_cols].copy()
    export_path = output_dir / "merged_sample.csv"
    export_df.to_csv(export_path, index=False)
    print(f"\n  Exported sample to: {export_path}")
    print(f"  Rows: {len(export_df):,}, Columns: {len(export_df.columns)}")

    # Show sample of merged data
    print("\n  Sample merged data (first 20 rows):")
    display_cols = export_cols[:8]  # First 8 columns for display
    print(export_df[display_cols].head(20).to_string(index=False))

    # Skip full merge (too slow for test) - sample data exported above is sufficient
    print("\n  (Full merge skipped - see sample CSV for interval analysis)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point for the test script."""
    print_separator("DATA JOINING DEBUG TEST")
    print(f"Script location: {__file__}")
    print(f"Working directory: {os.getcwd()}")

    # Check if we should use a custom path from command line
    datasets_dir = DATASETS_DIR
    if len(sys.argv) > 1:
        datasets_dir = sys.argv[1]
        print(f"Using datasets directory from command line: {datasets_dir}")
    else:
        print(f"Using default datasets directory: {datasets_dir}")
        print("  (Set GEOVUE_DATASETS_DIR env var or pass path as argument to override)")

    # Initialize DataCoordinator
    coordinator = initialize_data_coordinator(datasets_dir)
    if coordinator is None:
        print("\n" + "=" * 80)
        print("FAILED TO INITIALIZE - Cannot proceed with tests")
        print("=" * 80)
        return 1

    # Run debug functions
    debug_data_sources(coordinator)
    debug_schema_vs_actual(coordinator)
    debug_numeric_columns(coordinator)

    # Run join tests
    test_data_joining(coordinator)
    test_merged_analysis_table(coordinator)

    print_separator("TEST COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
