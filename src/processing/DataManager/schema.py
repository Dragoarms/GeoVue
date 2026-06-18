"""
schema.py - Schema definitions for data sources and columns.

This module provides:
- DataType enum for column type classification
- ColumnSchema: Definition for a single column with type, mapping, and display settings
- DataSourceSchema: Complete schema for a CSV data source
- Schema validation and type inference utilities

Schemas are persisted to ConfigManager and can be edited via the settings dialog.

Author: George Symonds
"""

import re
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Set, Callable
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataType(Enum):
    """
    Supported data types for columns.
    
    Used for:
    - Filter UI (determines operators available)
    - Color map compatibility
    - Data validation and conversion
    """
    NUMERIC = "numeric"       # Float/int values (Fe_pct, SiO2_pct, etc.)
    CATEGORICAL = "categorical"  # Limited set of values (STRATSUM, lithology codes)
    TEXT = "text"             # Free-form text (comments, descriptions)
    BOOLEAN = "boolean"       # True/False values
    DATE = "date"             # Date/datetime values
    
    @classmethod
    def from_string(cls, value) -> "DataType":
        """Convert string to DataType, with fallback to TEXT.
        
        Handles both string values and DataType objects for robustness.
        """
        # If already a DataType, return as-is
        if isinstance(value, cls):
            return value
        
        # Handle string conversion
        try:
            return cls(value.lower())
        except (ValueError, AttributeError):
            logger.warning(f"Unknown data type '{value}', defaulting to TEXT")
            return cls.TEXT


class DatasetType(Enum):
    """
    Type of geological dataset based on structure.
    
    Different dataset types require different indexing strategies:
    - INTERVAL: Has from/to depths, keyed by (hole_id, depth_to)
    - COLLAR: One row per hole, keyed by hole_id only
    - SURVEY: Single depth points (azimuth/dip), keyed by (hole_id, depth)
    - POINT: Single depth measurements, keyed by (hole_id, depth)
    """
    INTERVAL = "interval"     # Has from/to columns (assay, geology, drillhole_data)
    COLLAR = "collar"         # One row per hole (collar coordinates)
    SURVEY = "survey"         # Single depth with azimuth/dip
    POINT = "point"           # Single depth measurements
    
    @classmethod
    def from_string(cls, value) -> "DatasetType":
        """Convert string to DatasetType, with fallback to INTERVAL.
        
        Handles both string values and DatasetType objects for robustness.
        """
        # If already a DatasetType, return as-is
        if isinstance(value, cls):
            return value
        
        # Handle string conversion
        try:
            return cls(value.lower())
        except (ValueError, AttributeError):
            logger.warning(f"Unknown dataset type '{value}', defaulting to INTERVAL")
            return cls.INTERVAL


class NullHandling(Enum):
    """
    Strategy for handling null/missing values.
    
    Applied during data loading to ensure consistent data for filtering.
    """
    KEEP = "keep"              # Keep as NaN/None (default)
    FILL_ZERO = "fill_zero"    # Replace with 0 (for numeric)
    FILL_EMPTY = "fill_empty"  # Replace with empty string (for text)
    FILL_VALUE = "fill_value"  # Replace with custom value
    DROP = "drop"              # Mark rows for exclusion

    @classmethod
    def from_string(cls, value) -> "NullHandling":
        """Convert string to NullHandling, with fallback to KEEP.
        
        Handles both string values and NullHandling objects for robustness.
        """
        # If already a NullHandling, return as-is
        if isinstance(value, cls):
            return value
        
        # Handle string conversion
        try:
            return cls(value.lower())
        except (ValueError, AttributeError):
            logger.warning(f"Unknown null handling '{value}', defaulting to KEEP")
            return cls.KEEP


@dataclass
class ColumnSchema:
    """
    Schema definition for a single column in a data source.
    
    Defines how a column should be loaded, typed, displayed, and colored.
    
    Attributes:
        source_name: Original column name in the CSV file
        display_name: User-friendly name for UI display
        data_type: Type classification for the column
        null_handling: How to handle missing values
        fill_value: Custom fill value (when null_handling=FILL_VALUE)
        color_map: Name of color map preset to use (for visualization)
        is_key_column: Whether this is part of the primary key (holeid, from, to)
        is_visible: Whether to show in filter dropdowns
        min_value: Optional minimum valid value (for numeric)
        max_value: Optional maximum valid value (for numeric)
        categories: Optional list of valid categories (for categorical)
        
    Example:
        >>> col = ColumnSchema(
        ...     source_name="Fe_pct_BEST",
        ...     display_name="Fe %",
        ...     data_type=DataType.NUMERIC,
        ...     color_map="fe_grade"
        ... )
    """
    source_name: str
    display_name: Optional[str] = None
    data_type: DataType = DataType.TEXT
    null_handling: NullHandling = NullHandling.KEEP
    fill_value: Any = None
    color_map: Optional[str] = None
    is_key_column: bool = False
    is_visible: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    categories: Optional[List[str]] = None
    decimals: int = 2
    order: int = 0
    
    def __post_init__(self):
        """Set display_name to source_name if not provided."""
        if self.display_name is None:
            self.display_name = self.source_name
    
    @property
    def source_name_lower(self) -> str:
        """Get lowercase source name for case-insensitive matching."""
        return self.source_name.lower()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert enums to strings
        data["data_type"] = self.data_type.value
        data["null_handling"] = self.null_handling.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ColumnSchema":
        """Create from dictionary (JSON deserialization).
        
        Note: Works on a copy to avoid mutating the input dict.
        """
        # Work on a copy to avoid mutating the input dict (which may be from config cache)
        data = data.copy()
        
        # Convert strings back to enums
        if "data_type" in data:
            data["data_type"] = DataType.from_string(data["data_type"])
        if "null_handling" in data:
            data["null_handling"] = NullHandling.from_string(data["null_handling"])
            
        return cls(**data)
    
    def apply_null_handling(self, series: pd.Series) -> pd.Series:
        """
        Apply null handling strategy to a pandas Series.
        
        Args:
            series: Pandas Series to process
            
        Returns:
            Processed Series with nulls handled according to strategy
        """
        if self.null_handling == NullHandling.KEEP:
            return series
        
        if self.null_handling == NullHandling.FILL_ZERO:
            return series.fillna(0)
        
        if self.null_handling == NullHandling.FILL_EMPTY:
            return series.fillna("")
        
        if self.null_handling == NullHandling.FILL_VALUE:
            return series.fillna(self.fill_value)
        
        # DROP is handled at DataFrame level
        return series
    
    def validate_value(self, value: Any) -> bool:
        """
        Check if a value is valid according to this schema.
        
        Args:
            value: Value to validate
            
        Returns:
            True if valid, False otherwise
        """
        if pd.isna(value):
            return True  # Nulls are allowed (handled separately)
        
        if self.data_type == DataType.NUMERIC:
            try:
                num_val = float(value)
                if self.min_value is not None and num_val < self.min_value:
                    return False
                if self.max_value is not None and num_val > self.max_value:
                    return False
                return True
            except (ValueError, TypeError):
                return False
        
        if self.data_type == DataType.CATEGORICAL:
            if self.categories and str(value) not in self.categories:
                return False
        
        return True


@dataclass
class DataSourceSchema:
    """
    Complete schema definition for a CSV data source.
    
    Defines:
    - File path and identification
    - Dataset type (interval, collar, survey, point)
    - Key column mappings (hole_id, from, to, or depth)
    - All column schemas
    - Load/validation settings
    
    Attributes:
        name: Unique identifier for this data source (e.g., "drillhole_data")
        file_path: Path to the CSV file
        description: Human-readable description
        dataset_type: Type of dataset (INTERVAL, COLLAR, SURVEY, POINT)
        hole_id_column: Name of column containing hole IDs
        from_column: Name of column containing interval start depths (for INTERVAL type)
        to_column: Name of column containing interval end depths (for INTERVAL type)
        depth_column: Name of column containing single depth (for SURVEY/POINT type)
        columns: Dictionary of column schemas by source_name
        is_enabled: Whether to load this data source
        last_modified: Timestamp of last schema modification
        row_count: Cached row count from last load
        
    Example:
        >>> schema = DataSourceSchema(
        ...     name="drillhole_data",
        ...     file_path="/path/to/drillhole_data.csv",
        ...     dataset_type=DatasetType.INTERVAL,
        ...     hole_id_column="HOLEID",
        ...     from_column="SAMPFROM",
        ...     to_column="SAMPTO"
        ... )
    """
    name: str
    file_path: str
    description: str = ""
    dataset_type: DatasetType = DatasetType.INTERVAL
    hole_id_column: str = "HOLEID"
    from_column: Optional[str] = "from"
    to_column: Optional[str] = "to"
    depth_column: Optional[str] = None  # For SURVEY/POINT types
    columns: Dict[str, ColumnSchema] = field(default_factory=dict)
    is_enabled: bool = True
    last_modified: Optional[str] = None
    row_count: int = 0
    
    def __post_init__(self):
        """Initialize key columns based on dataset type."""
        # For COLLAR type, we don't need from/to columns
        if self.dataset_type == DatasetType.COLLAR:
            # Collar data only needs hole_id
            if self.hole_id_column.lower() not in {c.lower() for c in self.columns}:
                self.columns[self.hole_id_column] = ColumnSchema(
                    source_name=self.hole_id_column,
                    display_name="Hole ID",
                    data_type=DataType.TEXT,
                    is_key_column=True,
                    is_visible=False
                )
            return
        
        # For SURVEY/POINT type, we need hole_id and depth
        if self.dataset_type in (DatasetType.SURVEY, DatasetType.POINT):
            key_columns = [
                (self.hole_id_column, "Hole ID", DataType.TEXT),
            ]
            if self.depth_column:
                key_columns.append((self.depth_column, "Depth", DataType.NUMERIC))
            
            for col_name, display, dtype in key_columns:
                if col_name.lower() not in {c.lower() for c in self.columns}:
                    self.columns[col_name] = ColumnSchema(
                        source_name=col_name,
                        display_name=display,
                        data_type=dtype,
                        is_key_column=True,
                        is_visible=False
                    )
            return
        
        # For INTERVAL type (default), we need hole_id, from, to
        key_columns = [
            (self.hole_id_column, "Hole ID", DataType.TEXT),
        ]
        if self.from_column:
            key_columns.append((self.from_column, "From", DataType.NUMERIC))
        if self.to_column:
            key_columns.append((self.to_column, "To", DataType.NUMERIC))
        
        for col_name, display, dtype in key_columns:
            if col_name and col_name.lower() not in {c.lower() for c in self.columns}:
                self.columns[col_name] = ColumnSchema(
                    source_name=col_name,
                    display_name=display,
                    data_type=dtype,
                    is_key_column=True,
                    is_visible=False
                )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "file_path": self.file_path,
            "description": self.description,
            "dataset_type": self.dataset_type.value,
            "hole_id_column": self.hole_id_column,
            "from_column": self.from_column,
            "to_column": self.to_column,
            "depth_column": self.depth_column,
            "columns": {k: v.to_dict() for k, v in self.columns.items()},
            "is_enabled": self.is_enabled,
            "last_modified": self.last_modified,
            "row_count": self.row_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataSourceSchema":
        """Create from dictionary (JSON deserialization)."""
        columns = {}
        for name, col_data in data.get("columns", {}).items():
            columns[name] = ColumnSchema.from_dict(col_data)
        
        # Handle dataset_type conversion
        dataset_type = DatasetType.INTERVAL
        if "dataset_type" in data:
            dataset_type = DatasetType.from_string(data["dataset_type"])
        
        return cls(
            name=data["name"],
            file_path=data["file_path"],
            description=data.get("description", ""),
            dataset_type=dataset_type,
            hole_id_column=data.get("hole_id_column", "HOLEID"),
            from_column=data.get("from_column"),
            to_column=data.get("to_column"),
            depth_column=data.get("depth_column"),
            columns=columns,
            is_enabled=data.get("is_enabled", True),
            last_modified=data.get("last_modified"),
            row_count=data.get("row_count", 0),
        )
    
    def get_visible_columns(self) -> List[ColumnSchema]:
        """Get columns that should be shown in filter dropdowns."""
        return sorted(
            [c for c in self.columns.values() if c.is_visible],
            key=lambda c: c.order
        )
    
    def get_column(self, name: str) -> Optional[ColumnSchema]:
        """
        Get column schema by name (case-insensitive).
        
        Args:
            name: Column name to find
            
        Returns:
            ColumnSchema if found, None otherwise
        """
        name_lower = name.lower()
        for col_name, schema in self.columns.items():
            if col_name.lower() == name_lower:
                return schema
        return None


class SchemaInferrer:
    """
    Infers column schemas from actual CSV data.
    
    Used when loading a new CSV file to auto-detect:
    - Dataset type (INTERVAL, COLLAR, SURVEY, POINT)
    - Column data types
    - Categorical values
    - Numeric ranges
    - Null patterns
    """
    
    # Expanded patterns for common column name mappings (case-insensitive)
    HOLE_ID_PATTERNS = [
        "holeid", "hole_id", "hole", "bhid", "drillhole", "dhid",
        "boreholeid", "borehole_id", "dh_id", "dhname", "hole_name"
    ]
    
    # Expanded FROM patterns - covers many naming conventions
    FROM_PATTERNS = [
        "sampfrom", "samp_from", "from", "depth_from", "depthfrom",
        "from_m", "from_depth", "start_depth", "startdepth",
        "geolfrom", "geol_from", "assayfrom", "assay_from",
        "interval_from", "intervalfrom", "begindepth", "begin_depth",
        "top", "top_depth", "from_d"
    ]
    
    # Expanded TO patterns
    TO_PATTERNS = [
        "sampto", "samp_to", "to", "depth_to", "depthto",
        "to_m", "to_depth", "end_depth", "enddepth",
        "geolto", "geol_to", "assayto", "assay_to",
        "interval_to", "intervalto", "finishdepth", "finish_depth",
        "bottom", "bottom_depth", "to_d"
    ]
    
    # Single depth column patterns (for SURVEY/POINT datasets)
    DEPTH_PATTERNS = [
        "depth", "survey_depth", "surveydepth", "dip_depth",
        "measurement_depth", "point_depth", "md", "measured_depth"
    ]
    
    # Collar-specific columns that indicate COLLAR dataset type
    COLLAR_INDICATORS = [
        "east", "easting", "north", "northing", "rl", "elevation",
        "collar_rl", "collar_east", "collar_north", "x", "y", "z",
        "longitude", "latitude", "max_depth", "total_depth", "end_of_hole"
    ]
    
    # Survey-specific columns that indicate SURVEY dataset type
    SURVEY_INDICATORS = [
        "azimuth", "dip", "inclination", "bearing", "azim", "incl"
    ]
    
    # Threshold for categorical vs text classification
    CATEGORICAL_THRESHOLD = 50  # Max unique values for categorical
    
    # Threshold for numeric detection
    NUMERIC_THRESHOLD = 0.9  # 90% must parse as numeric
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def infer_schema(self, df: pd.DataFrame, name: str, file_path: str) -> DataSourceSchema:
        """
        Infer a complete schema from a DataFrame.
        
        Args:
            df: DataFrame to analyze
            name: Name for the data source
            file_path: Path to the source file
            
        Returns:
            DataSourceSchema with inferred column types and mappings
        """
        self.logger.info(f"Inferring schema for '{name}' with {len(df)} rows, {len(df.columns)} columns")
        
        # Lowercase column names for pattern matching
        columns_lower = {c.lower().strip(): c for c in df.columns}
        
        # Find hole ID column (required for all dataset types)
        hole_id_col = self._find_column(df.columns, self.HOLE_ID_PATTERNS)
        if not hole_id_col:
            self.logger.warning(f"Could not find hole ID column in {name}")
            hole_id_col = "holeid"  # Default
        
        # Detect dataset type and find appropriate key columns
        dataset_type, from_col, to_col, depth_col = self._detect_dataset_type(
            df.columns, columns_lower, name
        )
        
        self.logger.info(f"Detected dataset type: {dataset_type.value}")
        
        # Determine which columns are key columns based on dataset type
        key_col_names = {hole_id_col.lower()}
        if dataset_type == DatasetType.INTERVAL:
            if from_col:
                key_col_names.add(from_col.lower())
            if to_col:
                key_col_names.add(to_col.lower())
        elif dataset_type in (DatasetType.SURVEY, DatasetType.POINT):
            if depth_col:
                key_col_names.add(depth_col.lower())
        
        # Infer schemas for all columns
        columns = {}
        for idx, col in enumerate(df.columns):
            col_lower = col.lower().strip()
            is_key = col_lower in key_col_names
            
            col_schema = self._infer_column_schema(
                df[col], 
                col,
                is_key=is_key,
                order=idx
            )
            columns[col] = col_schema
        
        schema = DataSourceSchema(
            name=name,
            file_path=file_path,
            dataset_type=dataset_type,
            hole_id_column=hole_id_col,
            from_column=from_col,
            to_column=to_col,
            depth_column=depth_col,
            columns=columns,
            row_count=len(df)
        )
        
        self.logger.info(
            f"Inferred schema for '{name}' ({dataset_type.value}): "
            f"{sum(1 for c in columns.values() if c.data_type == DataType.NUMERIC)} numeric, "
            f"{sum(1 for c in columns.values() if c.data_type == DataType.CATEGORICAL)} categorical, "
            f"{sum(1 for c in columns.values() if c.data_type == DataType.TEXT)} text columns"
        )
        
        return schema
    
    def _detect_dataset_type(
        self, 
        columns: pd.Index, 
        columns_lower: Dict[str, str],
        name: str
    ) -> tuple:
        """
        Detect the dataset type based on available columns.
        
        Returns:
            Tuple of (DatasetType, from_col, to_col, depth_col)
        """
        # Check for from/to columns first (INTERVAL type)
        from_col = self._find_column(columns, self.FROM_PATTERNS)
        to_col = self._find_column(columns, self.TO_PATTERNS)
        depth_col = self._find_column(columns, self.DEPTH_PATTERNS)
        
        # Count indicator columns
        collar_count = sum(1 for p in self.COLLAR_INDICATORS if p in columns_lower)
        survey_count = sum(1 for p in self.SURVEY_INDICATORS if p in columns_lower)
        
        # Determine dataset type
        if from_col and to_col:
            # Has from/to columns - this is interval data
            return (DatasetType.INTERVAL, from_col, to_col, None)
        
        if survey_count >= 2 and depth_col:
            # Has azimuth/dip and depth - this is survey data
            self.logger.info(f"Detected SURVEY dataset (has azimuth/dip columns)")
            return (DatasetType.SURVEY, None, None, depth_col)
        
        if collar_count >= 3:
            # Has multiple collar indicators (coords, elevation) - this is collar data
            self.logger.info(f"Detected COLLAR dataset (has coordinate columns)")
            return (DatasetType.COLLAR, None, None, None)
        
        if depth_col and not from_col and not to_col:
            # Has single depth column but no from/to - point data
            self.logger.info(f"Detected POINT dataset (has depth but no from/to)")
            return (DatasetType.POINT, None, None, depth_col)
        
        # Default to INTERVAL but log warning about missing columns
        if not from_col:
            self.logger.warning(f"Could not find 'from' column in {name} - dataset may not load correctly")
        if not to_col:
            self.logger.warning(f"Could not find 'to' column in {name} - dataset may not load correctly")
        
        return (DatasetType.INTERVAL, from_col, to_col, None)
        
        return schema
    
    def _find_column(self, columns: pd.Index, patterns: List[str]) -> Optional[str]:
        """Find a column matching any of the given patterns."""
        columns_lower = {c.lower().strip(): c for c in columns}
        
        for pattern in patterns:
            if pattern in columns_lower:
                return columns_lower[pattern]
        
        return None
    
    def _infer_column_schema(
        self, 
        series: pd.Series, 
        column_name: str,
        is_key: bool = False,
        order: int = 0
    ) -> ColumnSchema:
        """
        Infer schema for a single column.
        
        Args:
            series: Pandas Series to analyze
            column_name: Name of the column
            is_key: Whether this is a key column
            order: Display order
            
        Returns:
            Inferred ColumnSchema
        """
        # Check for numeric
        # First, filter out blank/empty/whitespace-only values before checking
        non_blank_mask = series.notna() & (series.astype(str).str.strip() != "")
        non_blank_series = series[non_blank_mask]
        non_blank_count = len(non_blank_series)

        if non_blank_count > 0:
            numeric_series = pd.to_numeric(non_blank_series, errors="coerce")
            # Calculate ratio based on NON-BLANK values only
            # This prevents blank rows from causing numeric columns to be classed as text
            numeric_ratio = numeric_series.notna().sum() / non_blank_count
        else:
            numeric_series = pd.to_numeric(series, errors="coerce")
            numeric_ratio = 0

        if numeric_ratio >= self.NUMERIC_THRESHOLD:
            # Numeric column
            return ColumnSchema(
                source_name=column_name,
                display_name=self._prettify_name(column_name),
                data_type=DataType.NUMERIC,
                is_key_column=is_key,
                is_visible=not is_key,
                min_value=float(numeric_series.min()) if numeric_series.notna().any() else None,
                max_value=float(numeric_series.max()) if numeric_series.notna().any() else None,
                order=order
            )
        
        # Check for categorical (limited unique values)
        unique_values = series.dropna().unique()
        if len(unique_values) <= self.CATEGORICAL_THRESHOLD:
            return ColumnSchema(
                source_name=column_name,
                display_name=self._prettify_name(column_name),
                data_type=DataType.CATEGORICAL,
                is_key_column=is_key,
                is_visible=not is_key,
                categories=sorted([str(v) for v in unique_values]),
                order=order
            )
        
        # Default to text
        return ColumnSchema(
            source_name=column_name,
            display_name=self._prettify_name(column_name),
            data_type=DataType.TEXT,
            is_key_column=is_key,
            is_visible=not is_key,
            order=order
        )
    
    def _prettify_name(self, column_name: str) -> str:
        """
        Convert column name to user-friendly display name.
        
        Examples:
            Fe_pct_BEST -> Fe % Best
            HOLEID -> Hole ID
        """
        # Common replacements
        name = column_name
        name = re.sub(r'_pct_?', ' % ', name, flags=re.IGNORECASE)
        name = re.sub(r'_', ' ', name)
        
        # Title case with some exceptions
        words = name.split()
        result = []
        for word in words:
            if word.upper() in {"ID", "UID", "CSV", "JSON"}:
                result.append(word.upper())
            elif word.lower() in {"pct", "%"}:
                result.append("%")
            else:
                result.append(word.capitalize())
        
        return " ".join(result)


# Module-level inferrer instance
_inferrer = None

def get_inferrer() -> SchemaInferrer:
    """Get the singleton SchemaInferrer instance."""
    global _inferrer
    if _inferrer is None:
        _inferrer = SchemaInferrer()
    return _inferrer


def infer_schema(df: pd.DataFrame, name: str, file_path: str) -> DataSourceSchema:
    """Convenience function to infer schema from DataFrame."""
    return get_inferrer().infer_schema(df, name, file_path)
