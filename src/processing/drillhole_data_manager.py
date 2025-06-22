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


class IntervalScale(Enum):
    """Common interval scales in meters."""
    ULTRA_FINE = 0.1    # 10cm intervals
    FINE = 0.5          # 50cm intervals  
    STANDARD = 1.0      # 1m intervals
    COARSE = 2.0        # 2m intervals
    COMPOSITE = 5.0     # 5m intervals
    BULK = 10.0         # 10m intervals


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
    
    def __init__(self):
        """Initialize the data manager."""
        self.logger = logging.getLogger(__name__)
        self.data_sources: Dict[str, pd.DataFrame] = {}
        self.data_info: Dict[str, Dict[str, Any]] = {}
        self.harmonized_data: Optional[pd.DataFrame] = None
        self.hole_ids: List[str] = []
        
    def load_csv_files(self, file_paths: List[str]) -> Dict[str, str]:
        """
        Load multiple CSV files and validate their structure.
        
        Args:
            file_paths: List of paths to CSV files
            
        Returns:
            Dictionary of {filename: status_message} for each file
        """
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
                if not validation_result['valid']:
                    results[filename] = validation_result['message']
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
                    'columns': column_info,
                    'interval_scale': interval_scale,
                    'row_count': len(df),
                    'hole_ids': df['holeid'].unique().tolist()
                }
                
                # Update global hole IDs list
                self.hole_ids = list(set(self.hole_ids + self.data_info[filename]['hole_ids']))
                
                results[filename] = f"Successfully loaded: {len(df)} rows, {interval_scale.value}m intervals"
                self.logger.info(f"Loaded {filename}: {results[filename]}")
                
            except Exception as e:
                error_msg = f"Error loading file: {str(e)}"
                results[filename] = error_msg
                self.logger.error(f"Error loading {filename}: {str(e)}")
                
        return results
    
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
        holeid_variants = ['holeid', 'hole_id', 'hole', 'bhid', 'drillhole']
        holeid_col = None
        for variant in holeid_variants:
            if variant in columns:
                holeid_col = variant
                break
                
        if not holeid_col:
            return {
                'valid': False,
                'message': f"Missing HoleID column. Expected one of: {', '.join(holeid_variants)}"
            }
            
        # Rename to standard 'holeid'
        if holeid_col != 'holeid':
            df.rename(columns={holeid_col: 'holeid'}, inplace=True)
            
        # Check for depth columns - either From/To or single Depth
        has_from_to = 'from' in columns and 'to' in columns
        has_depth = 'depth' in columns or 'depth_m' in columns
        
        if not has_from_to and not has_depth:
            return {
                'valid': False,
                'message': "Missing depth columns. Expected either 'from'/'to' or 'depth'"
            }
            
        return {'valid': True, 'message': 'Valid structure'}
    
    def _standardize_depth_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize depth column names and ensure from/to columns exist.
        
        Args:
            df: DataFrame to standardize
            
        Returns:
            DataFrame with standardized depth columns
        """
        # If single depth column, create from/to based on assumed interval
        if 'depth' in df.columns or 'depth_m' in df.columns:
            depth_col = 'depth' if 'depth' in df.columns else 'depth_m'
            
            # Detect interval by looking at depth differences
            depths = df[depth_col].sort_values().unique()
            if len(depths) > 1:
                intervals = np.diff(depths)
                common_interval = np.median(intervals)
            else:
                common_interval = 1.0  # Default to 1m
                
            # Create from/to columns
            df['from'] = df[depth_col] - (common_interval / 2)
            df['to'] = df[depth_col] + (common_interval / 2)
            
        # Ensure numeric
        df['from'] = pd.to_numeric(df['from'], errors='coerce')
        df['to'] = pd.to_numeric(df['to'], errors='coerce')
        
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
            if col in ['holeid', 'from', 'to']:
                # Skip standard columns
                continue
                
            # Try to convert to numeric
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            numeric_ratio = numeric_series.notna().sum() / len(df)
            
            if numeric_ratio > 0.9:  # 90% or more are numeric
                # It's a numeric column
                column_info[col] = {
                    'type': DataType.NUMERIC,
                    'min': numeric_series.min(),
                    'max': numeric_series.max(),
                    'mean': numeric_series.mean(),
                    'nulls': numeric_series.isna().sum()
                }
            else:
                # Check if categorical (limited unique values)
                unique_values = df[col].dropna().unique()
                if len(unique_values) < 50:  # Arbitrary threshold
                    column_info[col] = {
                        'type': DataType.CATEGORICAL,
                        'categories': unique_values.tolist(),
                        'nulls': df[col].isna().sum()
                    }
                else:
                    column_info[col] = {
                        'type': DataType.TEXT,
                        'nulls': df[col].isna().sum()
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
        intervals = (df['to'] - df['from']).round(1)
        
        # Find most common interval
        common_interval = intervals.mode()[0] if len(intervals.mode()) > 0 else 1.0
        
        # Match to closest standard scale
        for scale in IntervalScale:
            if abs(common_interval - scale.value) < 0.05:
                return scale
                
        # Default to custom scale (round to nearest 0.1)
        return IntervalScale.STANDARD
    
    def get_data_for_hole(self, hole_id: str, 
                         source: Optional[str] = None) -> pd.DataFrame:
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
                return df[df['holeid'].str.upper() == hole_id.upper()].copy()
            else:
                return pd.DataFrame()
        else:
            # Combine from all sources
            all_data = []
            for source_name, df in self.data_sources.items():
                hole_data = df[df['holeid'].str.upper() == hole_id.upper()].copy()
                hole_data['_source'] = source_name
                all_data.append(hole_data)
                
            if all_data:
                return pd.concat(all_data, ignore_index=True)
            else:
                return pd.DataFrame()
    
    def harmonize_intervals(self, hole_id: str, 
                          target_scale: Optional[float] = None) -> pd.DataFrame:
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
            for source in all_data['_source'].unique():
                source_scale = self.data_info[source]['interval_scale']
                scales.append(source_scale.value)
            target_scale = min(scales)
            
        # Create regular interval grid
        min_depth = all_data['from'].min()
        max_depth = all_data['to'].max()
        
        # Generate intervals
        depth_grid = np.arange(min_depth, max_depth, target_scale)
        harmonized = pd.DataFrame({
            'holeid': hole_id,
            'from': depth_grid,
            'to': depth_grid + target_scale
        })
        
        # Merge data from each source
        for source in all_data['_source'].unique():
            source_data = all_data[all_data['_source'] == source]
            source_info = self.data_info[source]
            
            # Get data columns (excluding standard columns)
            data_cols = [col for col in source_data.columns 
                        if col not in ['holeid', 'from', 'to', '_source']]
            
            for col in data_cols:
                col_type = source_info['columns'][col]['type']
                
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
    
    def _interpolate_numeric(self, target_df: pd.DataFrame, 
                           source_df: pd.DataFrame, 
                           column: str) -> pd.Series:
        """
        Interpolate numeric values to target intervals.
        
        Args:
            target_df: DataFrame with target intervals
            source_df: DataFrame with source data
            column: Column name to interpolate
            
        Returns:
            Series with interpolated values
        """
        result = pd.Series(index=target_df.index, dtype=float)
        
        for idx, row in target_df.iterrows():
            # Find overlapping intervals in source
            overlap = source_df[
                (source_df['from'] < row['to']) & 
                (source_df['to'] > row['from'])
            ]
            
            if not overlap.empty:
                # Calculate weighted average based on overlap
                weights = []
                values = []
                
                for _, source_row in overlap.iterrows():
                    # Calculate overlap length
                    overlap_start = max(row['from'], source_row['from'])
                    overlap_end = min(row['to'], source_row['to'])
                    overlap_length = overlap_end - overlap_start
                    
                    if overlap_length > 0 and pd.notna(source_row[column]):
                        weights.append(overlap_length)
                        values.append(source_row[column])
                        
                if weights:
                    result[idx] = np.average(values, weights=weights)
                    
        return result
    
    def _interpolate_categorical(self, target_df: pd.DataFrame,
                               source_df: pd.DataFrame,
                               column: str) -> pd.Series:
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
                (source_df['from'] < row['to']) & 
                (source_df['to'] > row['from'])
            ]
            
            if not overlap.empty:
                # Use the value with maximum overlap
                max_overlap = 0
                best_value = None
                
                for _, source_row in overlap.iterrows():
                    overlap_start = max(row['from'], source_row['from'])
                    overlap_end = min(row['to'], source_row['to'])
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
            for col_name, col_info in info['columns'].items():
                columns.append((col_name, col_info['type']))
            available[source] = columns
            
        return available