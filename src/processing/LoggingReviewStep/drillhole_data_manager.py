"""
Manages multi-source data loading, validation, and harmonization for drillhole traces.
Handles different interval scales and data types across multiple CSV files.
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import re

logger = logging.getLogger(__name__)


class DataType(Enum):
    """Enumeration of supported data types for columns."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    UNKNOWN = "unknown"


# TODO - REMOVE THIS IntervalScale class? i don't think this is necessary...
class IntervalScale(Enum):
    """Common interval scales in meters."""

    ULTRA_FINE = 0.1  # 10cm intervals
    FINE = 0.5  # 50cm intervals
    STANDARD = 1.0  # 1m intervals
    COARSE = 2.0  # 2m intervals
    COMPOSITE = 5.0  # 5m intervals
    BULK = 10.0  # 10m intervals


class DrillholeDataManager:
    """
    Manages loading, validation, and harmonization of drillhole data from multiple sources.

    This class handles:
    - Loading multiple CSV files with different formats
    - Validating required columns (HoleID, From/To or Depth)
    - Detecting data types and interval scales
    - Harmonizing data across different interval scales
    - Providing unified access to multi-source data
    """

    def __init__(self, register_manager=None):
        """Initialize the data manager.
        
        Args:
            register_manager: Optional JSONRegisterManager for accessing image properties (hex colors)
        """
        self.logger = logging.getLogger(__name__)
        self.data_sources: Dict[str, pd.DataFrame] = {}
        self.data_info: Dict[str, Dict[str, Any]] = {}
        self.harmonized_data: Optional[pd.DataFrame] = None
        self.hole_ids: List[str] = []
        self.collar_data_path: Optional[str] = None
        self.register_manager = register_manager
        
        self.logger.info(f"DrillholeDataManager initialized with register_manager={'Yes' if register_manager else 'No'}")

    def set_collar_data_path(self, file_path: str) -> None:
        """Set the path to the collar data file (CSV or Excel).
        
        Args:
            file_path: Path to collar file with columns HoleID, BEST_X, BEST_Y, BEST_Z
                      Supports .csv, .xlsx, .xls formats
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Collar data file not found: {file_path}")
        
        # Validate file extension
        valid_extensions = ['.csv', '.xlsx', '.xls']
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in valid_extensions:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported: {valid_extensions}")
        
        self.collar_data_path = file_path
        self.logger.info(f"Collar data path set to: {file_path}")

    def get_collar_data(self) -> pd.DataFrame:
        """Get collar location data from the configured collar file (CSV or Excel).
        
        Returns:
            DataFrame with columns: HOLEID, X, Y, Z, PROJECT, PLANNED_HOLEID
            Returns empty DataFrame if no collar data is available
        
        Raises:
            ValueError: If collar file is missing required columns
        """
        if self.collar_data_path is None:
            self.logger.warning("No collar data path configured")
            return pd.DataFrame(columns=['HOLEID', 'X', 'Y', 'Z', 'PROJECT', 'PLANNED_HOLEID'])
        
        if not os.path.exists(self.collar_data_path):
            self.logger.error(f"Collar file not found: {self.collar_data_path}")
            return pd.DataFrame(columns=['HOLEID', 'X', 'Y', 'Z', 'PROJECT', 'PLANNED_HOLEID'])
        
        try:
            # Determine file type and load accordingly
            file_ext = os.path.splitext(self.collar_data_path)[1].lower()
            
            if file_ext == '.csv':
                df = pd.read_csv(self.collar_data_path)
            elif file_ext in ['.xlsx', '.xls']:
                # Read Excel file - assume first sheet
                df = pd.read_excel(self.collar_data_path, sheet_name=0)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            self.logger.info(f"Loaded {len(df)} rows from {os.path.basename(self.collar_data_path)}")
            
            # Normalize column names (remove spaces, uppercase)
            df.columns = [col.strip().upper().replace(' ', '_') for col in df.columns]
            
            # Check required columns exist in source
            # Required: HoleID, BEST_X, BEST_Y, BEST_Z
            # Optional but useful: Project, Planned HoleID
            required = ['HOLEID', 'BEST_X', 'BEST_Y', 'BEST_Z']
            missing = [col for col in required if col not in df.columns]
            
            if missing:
                raise ValueError(
                    f"Collar file missing required columns: {missing}. "
                    f"Expected: HoleID, BEST_X, BEST_Y, BEST_Z. "
                    f"Found columns: {list(df.columns)}"
                )
            
            # Create result DataFrame with required + optional columns
            result = pd.DataFrame({
                'HOLEID': df['HOLEID'].astype(str),
                'X': pd.to_numeric(df['BEST_X'], errors='coerce'),
                'Y': pd.to_numeric(df['BEST_Y'], errors='coerce'),
                'Z': pd.to_numeric(df['BEST_Z'], errors='coerce')
            })
            
            # Add optional columns if they exist
            if 'PROJECT' in df.columns:
                result['PROJECT'] = df['PROJECT'].astype(str)
            else:
                result['PROJECT'] = ''
            
            if 'PLANNED_HOLEID' in df.columns:
                result['PLANNED_HOLEID'] = df['PLANNED_HOLEID'].astype(str)
            else:
                result['PLANNED_HOLEID'] = ''
            
            # Remove rows with null coordinates (but keep if only PROJECT/PLANNED_HOLEID is missing)
            initial_count = len(result)
            result = result.dropna(subset=['X', 'Y', 'Z'])
            removed_count = initial_count - len(result)
            
            if removed_count > 0:
                self.logger.warning(f"Removed {removed_count} rows with missing coordinates")
            
            self.logger.info(f"Loaded {len(result)} valid collar records")
            return result
            
        except Exception as e:
            self.logger.error(f"Error loading collar data: {str(e)}")
            raise

    def find_collar_file(self, datasets_dir: str) -> Optional[str]:
        """Find Collar_Dashboard_PowerBi_Export file in datasets directory.
        
        Args:
            datasets_dir: Path to Drillhole Datasets directory
        
        Returns:
            Full path to collar file if found, None otherwise
        """
        base_name = 'Collar_Dashboard_PowerBi_Export'
        
        # Try extensions in order of preference
        for ext in ['.xlsx', '.csv', '.xls']:
            file_path = os.path.join(datasets_dir, base_name + ext)
            if os.path.exists(file_path):
                self.logger.info(f"Found collar file: {file_path}")
                return file_path
        
        self.logger.warning(f"Collar file not found in {datasets_dir}")
        return None

    def load_csv_files(self, file_paths: List[str]) -> Dict[str, str]:
        """
        Load multiple CSV files and validate their structure.

        Args:
            file_paths: List of paths to CSV files

        Returns:
            Dictionary of {filename: status_message} for each file
        """
        self.logger.info(f"=" * 80)
        self.logger.info(f"LOADING CSV FILES: {len(file_paths)} files to process")
        self.logger.info(f"=" * 80)
        
        results = {}

        for file_path in file_paths:
            filename = os.path.basename(file_path)
            try:
                # Load CSV
                df = pd.read_csv(file_path)

                # Standardize column names to lowercase
                df.columns = [col.lower().strip() for col in df.columns]

                # Validate required columns
                validation_result = self._validate_columns(df, filename)
                if not validation_result["valid"]:
                    results[filename] = validation_result["message"]
                    continue

                # Standardize depth columns
                df = self._standardize_depth_columns(df)

                # Detect data types for each column
                column_info = self._analyze_columns(df)

                # Detect interval scale
                interval_scale = self._detect_interval_scale(df)

                # Store the data and metadata
                self.data_sources[filename] = df
                self.data_info[filename] = {
                    "columns": column_info,
                    "interval_scale": interval_scale,
                    "row_count": len(df),
                    "hole_ids": df["holeid"].unique().tolist(),
                }

                # Update global hole IDs list
                self.hole_ids = list(
                    set(self.hole_ids + self.data_info[filename]["hole_ids"])
                )

                results[filename] = (
                    f"Successfully loaded: {len(df)} rows, {interval_scale.value}m intervals"
                )
                self.logger.info(f"✓ Loaded {filename}: {results[filename]}")
                self.logger.info(f"  - Columns: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}")
                self.logger.info(f"  - Hole IDs in file: {len(self.data_info[filename]['hole_ids'])}")
                self.logger.info(f"  - Total hole IDs now: {len(self.hole_ids)}")

            except Exception as e:
                error_msg = f"Error loading file: {str(e)}"
                results[filename] = error_msg
                self.logger.error(f"Error loading {filename}: {str(e)}")

        # Skip hex color enrichment - it's slow and not needed
        # Hex colors are already cached in LoggingReviewDialog for display
        # and are available on-demand via register_manager.get_hex_colors_for_interval()
        self.logger.info("Skipping hex color enrichment (not needed for filtering)")
        
        # Summary
        self.logger.info(f"=" * 80)
        self.logger.info(f"CSV LOADING COMPLETE")
        self.logger.info(f"  - Files loaded: {len(self.data_sources)}")
        self.logger.info(f"  - Total hole IDs: {len(self.hole_ids)}")
        self.logger.info(f"  - Hole IDs: {self.hole_ids[:10]}{'...' if len(self.hole_ids) > 10 else ''}")
        for source_name, df in self.data_sources.items():
            self.logger.info(f"  - {source_name}: {len(df)} rows, {len(df.columns)} columns")
        self.logger.info(f"=" * 80)

        return results

    def enrich_with_hex_colors(self) -> None:
        """
        Enrich loaded data sources with hex colors from image properties register.
        
        Adds three columns to each DataFrame:
        - wet_hex: Hex color for wet images
        - dry_hex: Hex color for dry images  
        - hex_color: Combined hex color (priority: dry > wet > white)
        
        This is called automatically after load_csv_files if register_manager is available.
        """
        if not self.register_manager:
            self.logger.debug("No register_manager available - skipping hex color enrichment")
            return
        
        if not hasattr(self.register_manager, 'get_hex_colors_for_interval'):
            self.logger.warning("register_manager doesn't have get_hex_colors_for_interval method")
            return
        
        self.logger.info("Enriching data sources with hex colors from image properties register")
        
        for source_name, df in self.data_sources.items():
            # Check if we have required columns
            if 'holeid' not in df.columns or 'from' not in df.columns or 'to' not in df.columns:
                self.logger.warning(f"Skipping {source_name} - missing required columns for hex color enrichment")
                continue
            
            # Initialize hex color columns
            df['wet_hex'] = '#FFFFFF'
            df['dry_hex'] = '#FFFFFF'
            df['hex_color'] = '#FFFFFF'
            
            enriched_count = 0
            
            # Get hex colors for each interval
            for idx, row in df.iterrows():
                try:
                    hole_id = row['holeid']
                    depth_from = row['from']
                    depth_to = row['to']
                    
                    hex_colors = self.register_manager.get_hex_colors_for_interval(
                        hole_id, depth_from, depth_to
                    )
                    
                    if hex_colors:
                        wet = hex_colors.get('wet_hex', '')
                        dry = hex_colors.get('dry_hex', '')
                        combined = hex_colors.get('combined_hex', '')
                        
                        # Update DataFrame
                        if wet:
                            df.at[idx, 'wet_hex'] = wet
                        if dry:
                            df.at[idx, 'dry_hex'] = dry
                        if combined:
                            df.at[idx, 'hex_color'] = combined
                            enriched_count += 1
                        
                except Exception as e:
                    self.logger.debug(f"Could not get hex color for {hole_id} {depth_from}-{depth_to}: {e}")
                    continue
            
            # Update the stored DataFrame
            self.data_sources[source_name] = df
            
            self.logger.info(f"Enriched {source_name}: {enriched_count} of {len(df)} intervals have hex colors")

    def _validate_columns(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """
        Validate that required columns exist in the dataframe.

        Args:
            df: DataFrame to validate
            filename: Source filename for error messages

        Returns:
            Dictionary with 'valid' boolean and 'message' string
        """
        columns = df.columns.tolist()

        # Check for HoleID column (various possible names)
        holeid_variants = ["holeid", "hole_id", "hole", "bhid", "drillhole"]
        holeid_col = None
        for variant in holeid_variants:
            if variant in columns:
                holeid_col = variant
                break

        if not holeid_col:
            return {
                "valid": False,
                "message": f"Missing HoleID column. Expected one of: {', '.join(holeid_variants)}",
            }

        # Rename to standard 'holeid'
        if holeid_col != "holeid":
            df.rename(columns={holeid_col: "holeid"}, inplace=True)

        # Check for depth columns - either From/To or single Depth
        has_from_to = "from" in columns and "to" in columns
        has_depth = "depth" in columns or "depth_m" in columns

        if not has_from_to and not has_depth:
            return {
                "valid": False,
                "message": "Missing depth columns. Expected either 'from'/'to' or 'depth'",
            }

        return {"valid": True, "message": "Valid structure"}

    def _standardize_depth_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize depth column names and ensure from/to columns exist.

        Args:
            df: DataFrame to standardize

        Returns:
            DataFrame with standardized depth columns
        """
        # If single depth column, create from/to based on assumed interval
        if "depth" in df.columns or "depth_m" in df.columns:
            depth_col = "depth" if "depth" in df.columns else "depth_m"

            # Detect interval by looking at depth differences
            depths = df[depth_col].sort_values().unique()
            if len(depths) > 1:
                intervals = np.diff(depths)
                common_interval = np.median(intervals)
            else:
                common_interval = 1.0  # Default to 1m

            # Create from/to columns
            df["from"] = df[depth_col] - (common_interval / 2)
            df["to"] = df[depth_col] + (common_interval / 2)

        # Ensure numeric
        df["from"] = pd.to_numeric(df["from"], errors="coerce")
        df["to"] = pd.to_numeric(df["to"], errors="coerce")

        return df

    def _analyze_columns(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Analyze each column to determine data type and characteristics.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary of column information
        """
        column_info = {}

        for col in df.columns:
            if col in ["holeid", "from", "to"]:
                # Skip standard columns
                continue

            # Try to convert to numeric
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            numeric_ratio = numeric_series.notna().sum() / len(df)

            if numeric_ratio > 0.9:  # 90% or more are numeric
                # It's a numeric column
                column_info[col] = {
                    "type": DataType.NUMERIC,
                    "min": numeric_series.min(),
                    "max": numeric_series.max(),
                    "mean": numeric_series.mean(),
                    "nulls": numeric_series.isna().sum(),
                }
            else:
                # Check if categorical (limited unique values)
                unique_values = df[col].dropna().unique()
                if len(unique_values) < 50:  # Arbitrary threshold
                    column_info[col] = {
                        "type": DataType.CATEGORICAL,
                        "categories": unique_values.tolist(),
                        "nulls": df[col].isna().sum(),
                    }
                else:
                    column_info[col] = {
                        "type": DataType.TEXT,
                        "nulls": df[col].isna().sum(),
                    }

        return column_info

    def _detect_interval_scale(self, df: pd.DataFrame) -> IntervalScale:
        """
        Detect the most common interval scale in the data.

        Args:
            df: DataFrame with from/to columns

        Returns:
            Detected IntervalScale
        """
        # Calculate interval lengths
        intervals = (df["to"] - df["from"]).round(1)

        # Find most common interval
        common_interval = intervals.mode()[0] if len(intervals.mode()) > 0 else 1.0

        # Match to closest standard scale
        for scale in IntervalScale:
            if abs(common_interval - scale.value) < 0.05:
                return scale

        # Default to custom scale (round to nearest 0.1)
        return IntervalScale.STANDARD

    def get_data_for_hole(
        self, hole_id: str, source: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get data for a specific hole from one or all sources.

        Args:
            hole_id: Hole ID to retrieve
            source: Optional specific source filename

        Returns:
            DataFrame with data for the specified hole
        """
        if source:
            if source in self.data_sources:
                df = self.data_sources[source]
                return df[df["holeid"].str.upper() == hole_id.upper()].copy()
            else:
                return pd.DataFrame()
        else:
            # Combine from all sources
            all_data = []
            for source_name, df in self.data_sources.items():
                hole_data = df[df["holeid"].str.upper() == hole_id.upper()].copy()
                hole_data["_source"] = source_name
                all_data.append(hole_data)

            if all_data:
                return pd.concat(all_data, ignore_index=True)
            else:
                return pd.DataFrame()
    
    def get_data_for_interval(
        self, hole_id: str, depth_to: float, source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get data for a specific depth interval.
        
        Args:
            hole_id: Hole ID to retrieve
            depth_to: Depth (to) value to match
            source: Optional specific source filename
            
        Returns:
            Dictionary of column values for the matching interval, empty dict if not found
        """
        # Normalize hole ID for comparison
        hole_id_upper = str(hole_id).strip().upper()
        
        # Get all data for this hole
        hole_data = self.get_data_for_hole(hole_id, source)
        
        if hole_data.empty:
            return {}
        
        # Find matching row by 'to' depth
        # Try exact match first
        matching_rows = hole_data[hole_data['to'] == depth_to]
        
        if matching_rows.empty:
            # Try with small tolerance for floating point comparison
            tolerance = 0.01
            matching_rows = hole_data[
                (hole_data['to'] >= depth_to - tolerance) & 
                (hole_data['to'] <= depth_to + tolerance)
            ]
        
        if matching_rows.empty:
            return {}
        
        # Return first match as dictionary
        return matching_rows.iloc[0].to_dict()

    def harmonize_intervals(
        self, hole_id: str, target_scale: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Harmonize data from different sources to a common interval scale.

        Args:
            hole_id: Hole ID to harmonize
            target_scale: Target interval scale (auto-detect if None)

        Returns:
            Harmonized DataFrame with consistent intervals
        """
        # Get all data for this hole
        all_data = self.get_data_for_hole(hole_id)

        if all_data.empty:
            return pd.DataFrame()

        # Determine target scale
        if target_scale is None:
            # Use the finest scale available
            scales = []
            for source in all_data["_source"].unique():
                source_scale = self.data_info[source]["interval_scale"]
                scales.append(source_scale.value)
            target_scale = min(scales)

        # Create regular interval grid
        min_depth = all_data["from"].min()
        max_depth = all_data["to"].max()

        # Generate intervals
        depth_grid = np.arange(min_depth, max_depth, target_scale)
        harmonized = pd.DataFrame(
            {"holeid": hole_id, "from": depth_grid, "to": depth_grid + target_scale}
        )

        # Merge data from each source
        for source in all_data["_source"].unique():
            source_data = all_data[all_data["_source"] == source]
            source_info = self.data_info[source]

            # Get data columns (excluding standard columns)
            data_cols = [
                col
                for col in source_data.columns
                if col not in ["holeid", "from", "to", "_source"]
            ]

            for col in data_cols:
                col_type = source_info["columns"][col]["type"]

                # Map each interval in harmonized to source data
                if col_type == DataType.NUMERIC:
                    # For numeric data, use weighted average
                    harmonized[f"{col}_{source}"] = self._interpolate_numeric(
                        harmonized, source_data, col
                    )
                else:
                    # For categorical/text, use most common value
                    harmonized[f"{col}_{source}"] = self._interpolate_categorical(
                        harmonized, source_data, col
                    )

        return harmonized

    def harmonize_intervals_fast(
        self, hole_id: str, target_scale: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Faster harmonization using vectorized operations.
        """
        # Get all data for this hole
        all_data = self.get_data_for_hole(hole_id)

        if all_data.empty:
            return pd.DataFrame()

        # Determine target scale
        if target_scale is None:
            scales = []
            for source in all_data["_source"].unique():
                source_scale = self.data_info[source]["interval_scale"]
                scales.append(source_scale.value)
            target_scale = min(scales)

        # Create regular interval grid
        min_depth = all_data["from"].min()
        max_depth = all_data["to"].max()

        # Generate intervals
        depth_grid = np.arange(min_depth, max_depth, target_scale)
        harmonized = pd.DataFrame(
            {"holeid": hole_id, "from": depth_grid, "to": depth_grid + target_scale}
        )

        # Group by source for batch processing
        for source in all_data["_source"].unique():
            source_data = all_data[all_data["_source"] == source]
            source_info = self.data_info[source]

            # Get data columns
            data_cols = [
                col
                for col in source_data.columns
                if col not in ["holeid", "from", "to", "_source"]
            ]

            # Process numeric columns together using vectorized operations
            numeric_cols = [
                col
                for col in data_cols
                if source_info["columns"][col]["type"] == DataType.NUMERIC
            ]

            if numeric_cols:
                # Ensure dtypes match for merge_asof
                source_sorted = source_data.sort_values("from").copy()
                source_sorted["from"] = source_sorted["from"].astype(float)
                source_sorted["to"] = source_sorted["to"].astype(float)

                harmonized_sorted = harmonized.sort_values("from").copy()
                harmonized_sorted["from"] = harmonized_sorted["from"].astype(float)

                # Use pandas merge_asof for efficient interval joining
                harmonized_temp = pd.merge_asof(
                    harmonized_sorted,
                    source_sorted[["from", "to"] + numeric_cols],
                    left_on="from",
                    right_on="from",
                    direction="nearest",
                    tolerance=target_scale,
                )

                # Add columns to main dataframe
                for col in numeric_cols:
                    harmonized[f"{col}_{source}"] = harmonized_temp[col]

            # Process categorical columns (still need iteration but can batch)
            cat_cols = [
                col
                for col in data_cols
                if source_info["columns"][col]["type"] != DataType.NUMERIC
            ]

            for col in cat_cols:
                harmonized[f"{col}_{source}"] = self._interpolate_categorical(
                    harmonized, source_data, col
                )

        return harmonized

    def _interpolate_numeric(
        self, target_df: pd.DataFrame, source_df: pd.DataFrame, column: str
    ) -> pd.Series:
        """
        Interpolate numeric values to target intervals using weighted averaging
        based on interval overlaps.
        """
        result = pd.Series(index=target_df.index, dtype=float)

        # Pre-filter source data to remove NaN values
        valid_source = source_df[source_df[column].notna()].copy()

        if valid_source.empty:
            return result

        # Sort source data once for efficiency
        valid_source = valid_source.sort_values("from")

        # Get bounds for early termination
        source_min = valid_source["from"].min()
        source_max = valid_source["to"].max()

        # Process all target intervals
        for idx, row in target_df.iterrows():
            # Skip if target is completely outside source range
            if row["to"] <= source_min or row["from"] >= source_max:
                continue

            # Find overlapping intervals
            mask = (valid_source["from"] < row["to"]) & (
                valid_source["to"] > row["from"]
            )
            overlaps = valid_source[mask]

            if not overlaps.empty:
                # Calculate overlap weights
                overlap_starts = np.maximum(overlaps["from"].values, row["from"])
                overlap_ends = np.minimum(overlaps["to"].values, row["to"])
                weights = overlap_ends - overlap_starts

                # Weighted average
                if weights.sum() > 0:
                    result[idx] = np.average(overlaps[column].values, weights=weights)

        return result

    def _interpolate_categorical(
        self, target_df: pd.DataFrame, source_df: pd.DataFrame, column: str
    ) -> pd.Series:
        """
        Interpolate categorical values to target intervals.

        Args:
            target_df: DataFrame with target intervals
            source_df: DataFrame with source data
            column: Column name to interpolate

        Returns:
            Series with interpolated values
        """
        result = pd.Series(index=target_df.index, dtype=object)

        for idx, row in target_df.iterrows():
            # Find overlapping intervals in source
            overlap = source_df[
                (source_df["from"] < row["to"]) & (source_df["to"] > row["from"])
            ]

            if not overlap.empty:
                # Use the value with maximum overlap
                max_overlap = 0
                best_value = None

                for _, source_row in overlap.iterrows():
                    overlap_start = max(row["from"], source_row["from"])
                    overlap_end = min(row["to"], source_row["to"])
                    overlap_length = overlap_end - overlap_start

                    if overlap_length > max_overlap and pd.notna(source_row[column]):
                        max_overlap = overlap_length
                        best_value = source_row[column]

                result[idx] = best_value

        return result

    def get_available_columns(self) -> Dict[str, List[Tuple[str, DataType]]]:
        """
        Get all available columns across all data sources.

        Returns:
            Dictionary of {source: [(column_name, data_type), ...]}
        """
        available = {}

        for source, info in self.data_info.items():
            columns = []
            for col_name, col_info in info["columns"].items():
                columns.append((col_name, col_info["type"]))
            available[source] = columns

        return available


    def get_data_for_color_map(self, column_name: str) -> Optional[List[float]]:
        """
        Get all values for a specific column across all data sources.
        Used for histogram preview in color map editor.
        
        Args:
            column_name: Name of column to extract (case-insensitive)
            
        Returns:
            List of numeric values, or None if column not found/not numeric
        """
        self.logger.debug(f"Getting data for color map: {column_name}")
        
        column_name_lower = column_name.lower().strip()
        all_values = []
        
        # Collect values from all data sources
        for source_name, df in self.data_sources.items():
            if column_name_lower in df.columns:
                # Get column data
                col_data = df[column_name_lower]
                
                # Only include numeric data
                numeric_data = pd.to_numeric(col_data, errors='coerce')
                valid_values = numeric_data.dropna().tolist()
                
                all_values.extend(valid_values)
                self.logger.debug(f"  {source_name}: {len(valid_values)} values")
        
        if not all_values:
            self.logger.warning(f"No numeric data found for column: {column_name}")
            return None
        
        self.logger.info(f"Collected {len(all_values)} values for {column_name}")
        return all_values