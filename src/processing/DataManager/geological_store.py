"""
geological_store.py - Geological CSV data management with efficient lookups.

This module provides:
- GeologicalStore: Loads CSV files with MultiIndex for O(1) interval lookups
- Multiple data source support with schema-driven loading
- Value retrieval by ImageKey

This replaces the slow row-by-row approach in the original DrillholeDataManager.

Key Performance Feature:
- Uses pandas MultiIndex on (hole_id, depth_to) for O(1) lookups
- No more iterating through DataFrames to find matching rows

Author: George Symonds
"""

import os
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from collections import defaultdict
import pandas as pd
import numpy as np

from processing.DataManager.keys import ImageKey
from processing.DataManager.schema import DataSourceSchema, ColumnSchema, DataType, SchemaInferrer, infer_schema

logger = logging.getLogger(__name__)


class IndexedDataSource:
    """
    A single CSV data source with MultiIndex for fast lookups.
    
    Wraps a pandas DataFrame with a (hole_id, depth_to) MultiIndex,
    enabling O(1) lookups by ImageKey.
    
    Attributes:
        name: Identifier for this data source
        schema: DataSourceSchema defining column types and mappings
        df: The underlying pandas DataFrame
        is_loaded: Whether data has been loaded successfully
    """
    
    def __init__(self, schema: DataSourceSchema):
        """
        Initialize with a schema.
        
        Args:
            schema: DataSourceSchema defining the data source
        """
        self.schema = schema
        self.name = schema.name
        self._df: Optional[pd.DataFrame] = None
        self._indexed_df: Optional[pd.DataFrame] = None
        self._is_loaded = False
        self._load_time: float = 0
        self._row_count: int = 0
        
        logger.debug(f"IndexedDataSource '{self.name}' initialized with schema")
    
    @property
    def is_loaded(self) -> bool:
        """Whether data has been loaded."""
        return self._is_loaded
    
    @property
    def df(self) -> Optional[pd.DataFrame]:
        """The raw (non-indexed) DataFrame."""
        return self._df
    
    @property
    def row_count(self) -> int:
        """Number of rows in the data source."""
        return self._row_count
    
    def load(self) -> bool:
        """
        Load the CSV file and build the index.
        
        Handles:
        - Multiple encoding fallbacks (utf-8, latin-1, cp1252)
        - Different dataset types (INTERVAL, COLLAR, SURVEY, POINT)
        
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(self.schema.file_path):
            logger.error(f"File not found: {self.schema.file_path}")
            return False
        
        start_time = time.time()
        
        try:
            logger.info(f"Loading data source '{self.name}' from {self.schema.file_path}")
            
            # Try multiple encodings
            self._df = self._read_csv_with_fallback(self.schema.file_path)
            if self._df is None:
                return False
            
            self._row_count = len(self._df)
            logger.debug(f"  Loaded {self._row_count:,} rows, {len(self._df.columns)} columns")
            
            # Standardize column names to lowercase
            self._df.columns = [c.lower().strip() for c in self._df.columns]
            
            # Get hole ID column (required for all types)
            hole_col = self.schema.hole_id_column.lower()
            if hole_col not in self._df.columns:
                logger.error(f"Hole ID column '{hole_col}' not found. Available: {list(self._df.columns)[:10]}...")
                return False
            
            # Create normalized hole column
            self._df["_hole_upper"] = self._df[hole_col].astype(str).str.strip().str.upper()
            
            # Build index based on dataset type
            success = self._build_index_for_type()
            if not success:
                return False
            
            self._is_loaded = True
            self._load_time = time.time() - start_time
            
            index_desc = self._get_index_description()
            logger.info(f"  Loaded '{self.name}': {self._row_count:,} rows in {self._load_time:.2f}s. {index_desc}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data source '{self.name}': {e}")
            import traceback
            logger.debug(traceback.format_exc())
            self._is_loaded = False
            return False
    
    def _read_csv_with_fallback(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Read CSV with encoding fallback.
        
        Tries: utf-8 → latin-1 → cp1252 → utf-8 with errors='replace'
        """
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, low_memory=False, encoding=encoding)
                logger.debug(f"  Successfully read with encoding: {encoding}")
                return df
            except UnicodeDecodeError:
                logger.debug(f"  Encoding {encoding} failed, trying next...")
                continue
            except Exception as e:
                logger.error(f"  Error reading CSV: {e}")
                return None
        
        # Last resort: read with error replacement
        try:
            logger.warning(f"  All encodings failed, using utf-8 with error replacement")
            df = pd.read_csv(file_path, low_memory=False, encoding='utf-8', errors='replace')
            return df
        except Exception as e:
            logger.error(f"  Failed to read CSV even with error replacement: {e}")
            return None
    
    def _build_index_for_type(self) -> bool:
        """
        Build the appropriate index based on dataset type.
        
        Returns:
            True if successful, False otherwise
        """
        from .schema import DatasetType
        
        dataset_type = getattr(self.schema, 'dataset_type', DatasetType.INTERVAL)
        
        if dataset_type == DatasetType.COLLAR:
            # COLLAR: Index by hole_id only
            self._indexed_df = self._df.set_index("_hole_upper", drop=False)
            self._indexed_df = self._indexed_df.sort_index()
            return True
        
        elif dataset_type == DatasetType.SURVEY:
            # SURVEY: Index by (hole_id, depth)
            depth_col = self.schema.depth_column
            if depth_col:
                depth_col = depth_col.lower()
                if depth_col in self._df.columns:
                    self._df["_depth_int"] = pd.to_numeric(self._df[depth_col], errors="coerce").fillna(0).astype(int)
                    self._indexed_df = self._df.set_index(["_hole_upper", "_depth_int"], drop=False)
                    self._indexed_df = self._indexed_df.sort_index()
                    return True
            # Fallback: index by hole only
            logger.warning(f"  SURVEY dataset missing depth column, indexing by hole only")
            self._indexed_df = self._df.set_index("_hole_upper", drop=False)
            self._indexed_df = self._indexed_df.sort_index()
            return True
        
        elif dataset_type == DatasetType.POINT:
            # POINT: Index by (hole_id, depth)
            depth_col = self.schema.depth_column
            if depth_col:
                depth_col = depth_col.lower()
                if depth_col in self._df.columns:
                    self._df["_depth_int"] = pd.to_numeric(self._df[depth_col], errors="coerce").fillna(0).astype(int)
                    self._indexed_df = self._df.set_index(["_hole_upper", "_depth_int"], drop=False)
                    self._indexed_df = self._indexed_df.sort_index()
                    return True
            # Fallback: index by hole only
            logger.warning(f"  POINT dataset missing depth column, indexing by hole only")
            self._indexed_df = self._df.set_index("_hole_upper", drop=False)
            self._indexed_df = self._indexed_df.sort_index()
            return True
        
        else:
            # INTERVAL (default): Index by (hole_id, depth_to)
            to_col = self.schema.to_column
            if not to_col:
                logger.error(f"INTERVAL dataset requires 'to' column")
                return False
            
            to_col = to_col.lower()
            if to_col not in self._df.columns:
                logger.error(f"To column '{to_col}' not found. Available: {list(self._df.columns)[:10]}...")
                return False
            
            self._df["_depth_int"] = pd.to_numeric(self._df[to_col], errors="coerce").fillna(0).astype(int)
            self._indexed_df = self._df.set_index(["_hole_upper", "_depth_int"], drop=False)
            self._indexed_df = self._indexed_df.sort_index()
            return True
    
    def _get_index_description(self) -> str:
        """Get a description of the index that was built."""
        from .schema import DatasetType
        
        dataset_type = getattr(self.schema, 'dataset_type', DatasetType.INTERVAL)
        
        if dataset_type == DatasetType.COLLAR:
            return f"Index built on ({self.schema.hole_id_column})"
        elif dataset_type in (DatasetType.SURVEY, DatasetType.POINT):
            depth_col = self.schema.depth_column or "depth"
            return f"Index built on ({self.schema.hole_id_column}, {depth_col})"
        else:
            return f"Index built on ({self.schema.hole_id_column}, {self.schema.to_column})"
    
    def get_row(self, key: ImageKey) -> Optional[Dict[str, Any]]:
        """
        Get a row by ImageKey (O(1) lookup).
        
        Args:
            key: ImageKey to look up
            
        Returns:
            Dictionary of column values, or None if not found
        """
        if not self._is_loaded or self._indexed_df is None:
            return None
        
        lookup_key = key.to_base_tuple()  # (hole_id_upper, depth_int)
        
        try:
            # O(1) lookup via MultiIndex
            row = self._indexed_df.loc[lookup_key]
            
            # Handle case where multiple rows match (take first)
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            
            # Convert to dict, excluding internal columns
            result = {}
            for col, value in row.items():
                if not col.startswith("_"):
                    # Handle NaN
                    if pd.isna(value):
                        result[col] = None
                    else:
                        result[col] = value
            
            return result
            
        except KeyError:
            # Key not found in index
            return None
        except Exception as e:
            logger.debug(f"Error looking up {key}: {e}")
            return None
    
    def query(
        self,
        filters: List[Dict[str, Any]],
        return_keys: bool = True
    ) -> Union[Set[Tuple[str, float]], pd.DataFrame]:
        """
        Query this data source with filter criteria.
        
        Args:
            filters: List of filter dicts, each with:
                - column: str - Column name to filter
                - operator: str - Operator ('=', '>', '<', '>=', '<=', '!=', 'between', 'contains', 'in', etc.)
                - value: Any - Filter value
                - value2: Any - Second value for 'between' operator
            return_keys: If True, return Set of (hole_id, depth_to) tuples
                        If False, return filtered DataFrame
        
        Returns:
            Set of (hole_id, depth_to) tuples matching all filters, or DataFrame
        """
        if not self._is_loaded or self._df is None:
            logger.warning(f"[QUERY DEBUG] Source '{self.name}' not loaded or no data")
            return set() if return_keys else pd.DataFrame()
        
        df = self._df
        logger.info(f"[QUERY DEBUG] Source '{self.name}': {len(df)} rows, filters: {filters}")
        
        if not filters:
            # No filters - return all keys
            if return_keys:
                hole_col = self.schema.hole_id_column.lower() if self.schema.hole_id_column else 'holeid'
                depth_col = self.schema.to_column.lower() if self.schema.to_column else 'to'
                
                all_keys = set()
                if hole_col in df.columns and depth_col in df.columns:
                    for _, row in df[[hole_col, depth_col]].iterrows():
                        try:
                            all_keys.add((str(row[hole_col]).upper(), float(row[depth_col])))
                        except (ValueError, TypeError):
                            continue
                return all_keys
            else:
                return df.copy()
        
        # Build mask for all filters
        mask = pd.Series([True] * len(df), index=df.index)
        
        for flt in filters:
            col = flt.get('column', '').lower()
            op = flt.get('operator', '')
            val = flt.get('value')
            val2 = flt.get('value2')
            
            if not col or not op:
                continue
            
            # Find column (case-insensitive)
            col_match = None
            for c in df.columns:
                if c.lower() == col:
                    col_match = c
                    break
            
            if not col_match:
                # Column not in this source - return empty result
                # (can't match a filter on a column that doesn't exist)
                logger.info(f"[QUERY DEBUG] Column '{col}' NOT FOUND in source '{self.name}' - returning empty result")
                return set() if return_keys else pd.DataFrame()
            
            col_data = df[col_match]
            
            # Debug: show column data info
            logger.info(f"[QUERY DEBUG] Filtering '{col_match}': dtype={col_data.dtype}, non-null={col_data.notna().sum()}")
            unique_sample = col_data.dropna().unique()[:10]
            logger.info(f"[QUERY DEBUG] Sample unique values in '{col_match}': {list(unique_sample)}")
            
            # Apply operator
            try:
                if op in ('=', 'equals', '=='):
                    mask &= (col_data == val)
                elif op in ('!=', '≠', 'not equals'):
                    mask &= (col_data != val)
                elif op in ('>', 'greater than'):
                    mask &= (pd.to_numeric(col_data, errors='coerce') > float(val))
                elif op in ('>=', '≥', 'greater than or equal'):
                    mask &= (pd.to_numeric(col_data, errors='coerce') >= float(val))
                elif op in ('<', 'less than'):
                    mask &= (pd.to_numeric(col_data, errors='coerce') < float(val))
                elif op in ('<=', '≤', 'less than or equal'):
                    mask &= (pd.to_numeric(col_data, errors='coerce') <= float(val))
                elif op == 'between':
                    numeric_col = pd.to_numeric(col_data, errors='coerce')
                    mask &= (numeric_col >= float(val)) & (numeric_col <= float(val2))
                elif op == 'contains':
                    mask &= col_data.astype(str).str.contains(str(val), case=False, na=False)
                elif op == 'not contains':
                    mask &= ~col_data.astype(str).str.contains(str(val), case=False, na=False)
                elif op == 'starts with':
                    mask &= col_data.astype(str).str.startswith(str(val), na=False)
                elif op == 'ends with':
                    mask &= col_data.astype(str).str.endswith(str(val), na=False)
                elif op in ('in', 'in list'):
                    if isinstance(val, str):
                        val_list = [v.strip().lower() for v in val.split(',')]
                    else:
                        val_list = [str(v).lower() for v in val]
                    
                    # Debug the 'in' operation
                    col_normalized = col_data.astype(str).str.strip().str.lower()
                    logger.info(f"[QUERY DEBUG] 'in' filter: looking for {val_list}")
                    logger.info(f"[QUERY DEBUG] Sample normalized col values: {list(col_normalized.dropna().unique()[:10])}")
                    
                    in_mask = col_normalized.isin(val_list)
                    match_count = in_mask.sum()
                    logger.info(f"[QUERY DEBUG] 'in' match count: {match_count} / {len(df)}")
                    
                    # Case-insensitive comparison
                    mask &= in_mask
                elif op in ('not in', 'not in list'):
                    if isinstance(val, str):
                        val_list = [v.strip().lower() for v in val.split(',')]
                    else:
                        val_list = [str(v).lower() for v in val]
                    # Case-insensitive comparison
                    mask &= ~col_data.astype(str).str.strip().str.lower().isin(val_list)
                elif op in ('is null', 'is empty'):
                    mask &= col_data.isna() | (col_data == '')
                elif op in ('not null', 'is not empty'):
                    mask &= col_data.notna() & (col_data != '')
            except Exception as e:
                logger.warning(f"Filter error on {col_match} {op} {val}: {e}")
                continue
        
        # Get matching rows
        filtered_df = df[mask]
        logger.info(f"[QUERY DEBUG] After all filters: {len(filtered_df)} / {len(df)} rows match (mask sum: {mask.sum()})")
        
        if return_keys:
            # Extract (hole_id, depth_to) tuples
            hole_col = self.schema.hole_id_column.lower() if self.schema.hole_id_column else 'holeid'
            depth_col = self.schema.to_column.lower() if self.schema.to_column else 'to'
            
            result_keys = set()
            if hole_col in filtered_df.columns and depth_col in filtered_df.columns:
                for _, row in filtered_df[[hole_col, depth_col]].iterrows():
                    try:
                        result_keys.add((str(row[hole_col]).upper(), float(row[depth_col])))
                    except (ValueError, TypeError):
                        continue
            
            return result_keys
        else:
            return filtered_df


    def get_value(self, key: ImageKey, column: str) -> Optional[Any]:
        """
        Get a single column value by ImageKey.
        
        Args:
            key: ImageKey to look up
            column: Column name to retrieve
            
        Returns:
            Column value, or None if not found
        """
        row = self.get_row(key)
        if row is None:
            return None
        
        col_lower = column.lower()
        return row.get(col_lower)
    
    def get_rows_for_hole(self, hole_id: str) -> pd.DataFrame:
        """
        Get all rows for a specific hole.
        
        Args:
            hole_id: Hole identifier (case-insensitive)
            
        Returns:
            DataFrame with all rows for the hole
        """
        if not self._is_loaded or self._indexed_df is None:
            return pd.DataFrame()
        
        hole_upper = hole_id.upper()
        
        try:
            # Slice on first level of MultiIndex
            result = self._indexed_df.loc[hole_upper]
            
            # If single row returned, convert to DataFrame
            if isinstance(result, pd.Series):
                result = result.to_frame().T
            
            return result
            
        except KeyError:
            return pd.DataFrame()
    
    def get_unique_holes(self) -> Set[str]:
        """Get set of unique hole IDs in this data source."""
        if not self._is_loaded or self._df is None:
            return set()
        return set(self._df["_hole_upper"].unique())
    
    def get_column_names(self) -> List[str]:
        """Get list of available column names."""
        if not self._is_loaded or self._df is None:
            return []
        return [c for c in self._df.columns if not c.startswith("_")]
    
    def get_column_values(self, column: str) -> List[Any]:
        """
        Get all values for a column (for histograms, etc.).
        
        Args:
            column: Column name
            
        Returns:
            List of values (excluding NaN)
        """
        if not self._is_loaded or self._df is None:
            return []
        
        col_lower = column.lower()
        if col_lower not in self._df.columns:
            return []
        
        return self._df[col_lower].dropna().tolist()
    
    def get_row_by_key(self, hole_id_upper: str, depth_int: int) -> Optional[pd.Series]:
        """
        Get a row by hole_id and depth_to (for internal use by GeologicalStore).
        
        Args:
            hole_id_upper: Uppercase hole ID
            depth_int: Integer depth_to value
            
        Returns:
            pandas Series if found, None otherwise
        """
        if not self._is_loaded or self._indexed_df is None:
            return None
        
        lookup_key = (hole_id_upper, depth_int)
        
        try:
            row = self._indexed_df.loc[lookup_key]
            
            # Handle case where multiple rows match (take first)
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            
            return row
            
        except KeyError:
            return None
        except Exception as e:
            logger.debug(f"Error looking up ({hole_id_upper}, {depth_int}): {e}")
            return None


class GeologicalStore:
    """
    Manages multiple geological data sources with efficient lookups.
    
    Provides a unified interface over multiple IndexedDataSource instances,
    allowing queries across all loaded CSV files.
    
    Usage:
        >>> store = GeologicalStore(config_manager)
        >>> store.add_source("/path/to/drillhole_data.csv")
        >>> store.load_all()
        >>> 
        >>> # Get data for a specific interval
        >>> data = store.get_row(ImageKey("BA0001", 45.0))
        >>> fe_value = data.get("fe_pct_best")
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the geological store.
        
        Args:
            config_manager: Optional ConfigManager for schema persistence
        """
        self.config_manager = config_manager
        self._sources: Dict[str, IndexedDataSource] = {}
        self._schemas: Dict[str, DataSourceSchema] = {}
        self._is_loaded = False
        
        # Cache for column metadata
        self._all_columns: Dict[str, Tuple[str, DataType]] = {}  # col_name -> (source_name, type)
        
        logger.debug("GeologicalStore initialized")
        
        # Load schemas from config if available
        if config_manager:
            self._load_schemas_from_config()
    
    def _load_schemas_from_config(self):
        """Load data source schemas from config manager."""
        saved_schemas = self.config_manager.get("geological_data_sources", [])
        
        for schema_data in saved_schemas:
            try:
                schema = DataSourceSchema.from_dict(schema_data)
                self._schemas[schema.name] = schema
                logger.debug(f"Loaded schema '{schema.name}' from config")
            except Exception as e:
                logger.error(f"Error loading schema from config: {e}")
    
    def _save_schemas_to_config(self):
        """Save data source schemas to config manager."""
        if not self.config_manager:
            return

        schemas_data = [schema.to_dict() for schema in self._schemas.values()]
        self.config_manager.set("geological_data_sources", schemas_data)
        logger.debug(f"Saved {len(schemas_data)} schemas to config")

    def save_schemas_to_config(self) -> bool:
        """
        Save all source schemas to ConfigManager for persistence.

        Serializes column settings (display_name, color_map, decimals,
        is_visible, data_type, null_handling) so they survive application restart.

        Returns:
            True if successful, False if no config_manager
        """
        if not self.config_manager:
            logger.warning("Cannot save schemas: no ConfigManager configured")
            return False

        schemas_dict = {}
        for source_name, indexed_source in self._sources.items():
            schema = indexed_source.schema
            schemas_dict[source_name] = schema.to_dict()

        self.config_manager.set("column_schemas", schemas_dict)
        logger.info(f"Saved column schemas for {len(schemas_dict)} sources to config")
        return True

    def _load_saved_schemas(self) -> Dict[str, dict]:
        """
        Load previously saved schemas from ConfigManager.

        Returns:
            Dictionary of {source_name: schema_dict}
        """
        if not self.config_manager:
            return {}

        saved = self.config_manager.get("column_schemas", {})
        
        if saved:
            logger.info(f"Found saved schemas for {len(saved)} sources")
            # Debug: print exactly what's saved for each source
            for source_name, source_data in saved.items():
                columns = source_data.get("columns", {})
                hidden_cols = [c for c, d in columns.items() if d.get("is_visible") is False]
                logger.debug(f"  [{source_name}] {len(columns)} columns saved, {len(hidden_cols)} hidden")
                
                # Show data_type values and their Python types for first few columns
                sample_cols = list(columns.items())[:3]
                for col_name, col_data in sample_cols:
                    dt = col_data.get("data_type")
                    vis = col_data.get("is_visible", True)
                    logger.debug(f"    - {col_name}: data_type={dt!r} (type={type(dt).__name__}), visible={vis}")
                    
        return saved

    def _apply_saved_schema(self, source_name: str, schema: 'DataSourceSchema', saved_schemas: Dict[str, dict]) -> None:
        """
        Apply saved column settings to a schema.

        Preserves user customizations (display_name, color_map, decimals,
        is_visible, data_type override) while keeping inferred structure.

        Args:
            source_name: Name of the data source
            schema: The freshly inferred schema to modify
            saved_schemas: Dictionary of saved schema data
        """
        if source_name not in saved_schemas:
            logger.debug(f"No saved schema for '{source_name}'")
            return

        saved = saved_schemas[source_name]
        saved_columns = saved.get("columns", {})
        
        logger.debug(f"Applying saved schema for '{source_name}' ({len(saved_columns)} saved columns)")

        applied_count = 0
        hidden_count = 0
        for col_name, col_schema in schema.columns.items():
            col_lower = col_name.lower()

            # Find matching saved column (case-insensitive)
            saved_col = None
            for saved_name, saved_data in saved_columns.items():
                if saved_name.lower() == col_lower:
                    saved_col = saved_data
                    break

            if not saved_col:
                continue

            # Apply saved settings
            if "display_name" in saved_col and saved_col["display_name"]:
                col_schema.display_name = saved_col["display_name"]

            if "color_map" in saved_col and saved_col["color_map"]:
                col_schema.color_map = saved_col["color_map"]

            if "decimals" in saved_col:
                col_schema.decimals = saved_col["decimals"]

            if "is_visible" in saved_col:
                old_visible = col_schema.is_visible
                col_schema.is_visible = saved_col["is_visible"]
                if not col_schema.is_visible:
                    hidden_count += 1
                if old_visible != col_schema.is_visible:
                    logger.debug(f"    {col_name}: visibility {old_visible} -> {col_schema.is_visible}")

            if "data_type" in saved_col:
                from .schema import DataType
                raw_dt = saved_col["data_type"]
                logger.debug(f"    {col_name}: data_type raw value = {raw_dt!r} (type={type(raw_dt).__name__})")
                col_schema.data_type = DataType.from_string(raw_dt)

            if "null_handling" in saved_col:
                from .schema import NullHandling
                col_schema.null_handling = NullHandling.from_string(saved_col["null_handling"])

            applied_count += 1

        if applied_count > 0:
            logger.info(f"Applied saved settings to {applied_count} columns in '{source_name}' ({hidden_count} hidden)")

    # =========================================================================
    # Source Management
    # =========================================================================
    
    def add_source(
        self,
        file_path: str,
        name: Optional[str] = None,
        schema: Optional[DataSourceSchema] = None
    ) -> bool:
        """
        Add a CSV data source.

        If no schema is provided, checks for saved schema first.
        Only infers schema if no saved schema exists or columns have changed.

        Args:
            file_path: Path to CSV file
            name: Optional name for the source (defaults to filename)
            schema: Optional predefined schema

        Returns:
            True if added successfully, False otherwise
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False

        # Generate name from filename if not provided
        if name is None:
            name = Path(file_path).stem

        # Check for duplicates
        if name in self._sources:
            logger.warning(f"Data source '{name}' already exists, replacing")

        # Create or use provided schema
        if schema is None:
            # Check for saved schema first (optimization: skip inference if schema exists)
            saved_schemas = self._load_saved_schemas()
            saved_schema_data = saved_schemas.get(name)

            if saved_schema_data:
                # We have a saved schema - check if file columns still match
                try:
                    # Read just the header to get column names
                    df_header = pd.read_csv(file_path, nrows=0, low_memory=False)
                    file_columns = set(c.lower().strip() for c in df_header.columns)

                    saved_columns = set(c.lower() for c in saved_schema_data.get("columns", {}).keys())

                    # If columns match (or saved is subset), use saved schema
                    if saved_columns and saved_columns.issubset(file_columns):
                        logger.info(f"Using saved schema for '{name}' (skipping inference)")
                        try:
                            schema = DataSourceSchema.from_dict(saved_schema_data)
                            # Ensure file_path is current
                            schema.file_path = file_path
                        except Exception as e:
                            logger.warning(f"Failed to restore saved schema for '{name}': {e}")
                            schema = None
                    else:
                        logger.info(f"Columns changed in '{name}' - will infer new schema")
                        logger.debug(f"  File columns: {len(file_columns)}, Saved columns: {len(saved_columns)}")
                except Exception as e:
                    logger.debug(f"Could not check columns for '{name}': {e}")

            # Infer schema if we don't have one yet
            if schema is None:
                try:
                    logger.info(f"Inferring schema for '{name}'...")
                    df_sample = pd.read_csv(file_path, nrows=1000, low_memory=False)
                    schema = infer_schema(df_sample, name, file_path)
                except Exception as e:
                    logger.error(f"Error inferring schema for {file_path}: {e}")
                    return False

        # Store schema
        self._schemas[name] = schema

        # Create data source
        source = IndexedDataSource(schema)
        self._sources[name] = source

        logger.info(f"Added data source '{name}' from {file_path}")

        # Save to config
        self._save_schemas_to_config()

        return True
    
    def remove_source(self, name: str) -> bool:
        """
        Remove a data source.
        
        Args:
            name: Name of the source to remove
            
        Returns:
            True if removed, False if not found
        """
        if name not in self._sources:
            logger.warning(f"Data source '{name}' not found")
            return False
        
        del self._sources[name]
        if name in self._schemas:
            del self._schemas[name]
        
        # Update column cache
        self._rebuild_column_cache()
        
        # Save to config
        self._save_schemas_to_config()
        
        logger.info(f"Removed data source '{name}'")
        return True
    
    def get_source(self, name: str) -> Optional[IndexedDataSource]:
        """Get a specific data source by name."""
        return self._sources.get(name)
    
    def get_schema(self, name: str) -> Optional[DataSourceSchema]:
        """Get the schema for a data source."""
        return self._schemas.get(name)
    
    def list_sources(self) -> List[str]:
        """Get names of all data sources."""
        return list(self._sources.keys())
    
    # =========================================================================
    # Loading
    # =========================================================================
    
    def load_all(self, progress_callback=None) -> Dict[str, bool]:
        """
        Load all data sources.

        Args:
            progress_callback: Optional callback(source_name, success) for progress

        Returns:
            Dictionary of {source_name: success}
        """
        logger.info("=" * 60)
        logger.info("LOADING GEOLOGICAL DATA SOURCES")
        logger.info("=" * 60)

        # Load saved schemas BEFORE loading sources
        saved_schemas = self._load_saved_schemas()

        results = {}
        total = len(self._sources)

        for idx, (name, source) in enumerate(self._sources.items()):
            logger.info(f"[{idx+1}/{total}] Loading '{name}'...")

            # Apply saved customizations to the inferred schema BEFORE loading
            self._apply_saved_schema(name, source.schema, saved_schemas)

            success = source.load()
            results[name] = success

            if progress_callback:
                progress_callback(name, success)
        
        # Rebuild column cache
        self._rebuild_column_cache()
        
        self._is_loaded = True
        
        # Summary
        success_count = sum(1 for v in results.values() if v)
        logger.info("=" * 60)
        logger.info(f"LOADING COMPLETE: {success_count}/{total} sources loaded successfully")
        logger.info("=" * 60)
        
        return results
    
    def load_source(self, name: str) -> bool:
        """
        Load a specific data source.
        
        Args:
            name: Name of the source to load
            
        Returns:
            True if successful
        """
        source = self._sources.get(name)
        if not source:
            logger.error(f"Data source '{name}' not found")
            return False
        
        success = source.load()
        
        if success:
            self._rebuild_column_cache()
        
        return success
    
    def _rebuild_column_cache(self):
        """Rebuild the aggregated column metadata cache."""
        self._all_columns.clear()
        
        for name, source in self._sources.items():
            if not source.is_loaded:
                continue
            
            for col in source.get_column_names():
                # Get type from schema if available
                schema = self._schemas.get(name)
                col_type = DataType.TEXT
                
                if schema:
                    col_schema = schema.get_column(col)
                    if col_schema:
                        col_type = col_schema.data_type
                
                # Store with source reference
                self._all_columns[col] = (name, col_type)
        
        logger.debug(f"Column cache rebuilt: {len(self._all_columns)} columns from {len(self._sources)} sources")
    
    # =========================================================================
    # Data Access
    # =========================================================================
    
    def get_row(self, key: ImageKey, source_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all data for an interval by ImageKey.
        
        If source_name is not specified, queries all sources and merges results.
        
        Args:
            key: ImageKey to look up
            source_name: Optional specific source to query
            
        Returns:
            Dictionary of column -> value. Empty dict if not found.
        """
        if source_name:
            source = self._sources.get(source_name)
            if source and source.is_loaded:
                return source.get_row(key) or {}
            return {}
        
        # Query all sources and merge
        result = {}
        col_counts = {}  # Track how many times we've seen each column
        for name, source in self._sources.items():
            if not source.is_loaded:
                continue

            row = source.get_row(key)
            if row:
                # Handle columns with numeric suffix for duplicates (_1, _2, etc.)
                for col, value in row.items():
                    if col in result:
                        # Duplicate - add numeric suffix
                        col_counts[col] = col_counts.get(col, 1) + 1
                        result[f"{col}_{col_counts[col]}"] = value
                    else:
                        result[col] = value

        return result
    
    def get_value(
        self, 
        key: ImageKey, 
        column: str, 
        source_name: Optional[str] = None
    ) -> Optional[Any]:
        """
        Get a single column value for an interval.
        
        Args:
            key: ImageKey to look up
            column: Column name
            source_name: Optional specific source
            
        Returns:
            Column value, or None if not found
        """
        if source_name:
            source = self._sources.get(source_name)
            if source and source.is_loaded:
                return source.get_value(key, column)
            return None
        
        # Search all sources
        for name, source in self._sources.items():
            if not source.is_loaded:
                continue
            
            value = source.get_value(key, column)
            if value is not None:
                return value
        
        return None
    
    def get_rows_for_hole(self, hole_id: str) -> Dict[str, pd.DataFrame]:
        """
        Get all data for a hole from all sources.
        
        Args:
            hole_id: Hole identifier
            
        Returns:
            Dictionary of {source_name: DataFrame}
        """
        result = {}
        
        for name, source in self._sources.items():
            if not source.is_loaded:
                continue
            
            df = source.get_rows_for_hole(hole_id)
            if not df.empty:
                result[name] = df
        
        return result
    
    def get_depth_from(self, hole_id: str, depth_to: float) -> Optional[float]:
        """
        Get the depth_from value for a given hole_id and depth_to.
        
        Looks up the actual interval from the CSV data rather than assuming 1m.
        
        Args:
            hole_id: Hole ID (case-insensitive)
            depth_to: End depth
            
        Returns:
            depth_from value if found, None otherwise
        """
        hole_id_upper = hole_id.upper()
        depth_int = int(depth_to)
        
        # Try each source until we find a match
        for source_name, source in self._sources.items():
            if not source.is_loaded:
                continue
            
            row = source.get_row_by_key(hole_id_upper, depth_int)
            if row is not None:
                # Look for 'from' column (schema should have identified it)
                schema = self._schemas.get(source_name)
                if schema and schema.from_column:
                    from_col = schema.from_column.lower()
                    if from_col in row.index:
                        try:
                            return float(row[from_col])
                        except (ValueError, TypeError):
                            pass
                
                # Fallback: try common column names
                for col_name in ['from', 'depth_from', 'from_m', 'start_depth']:
                    if col_name in row.index:
                        try:
                            return float(row[col_name])
                        except (ValueError, TypeError):
                            pass
        
        # Not found - return None (caller can decide to use default)
        return None
    
    def get_interval_size(self, hole_id: str, depth_to: float) -> float:
        """
        Get the interval size for a given depth.
        
        Args:
            hole_id: Hole ID
            depth_to: End depth
            
        Returns:
            Interval size in meters (defaults to 1.0 if not found)
        """
        depth_from = self.get_depth_from(hole_id, depth_to)
        if depth_from is not None:
            return depth_to - depth_from
        return 1.0  # Default fallback

    def get_available_columns(self) -> Dict[str, List[Tuple[str, DataType]]]:
        """
        Get all available columns across all sources.
        
        Returns:
            Dictionary of {source_name: [(column_name, data_type), ...]}
        """
        result = {}
        
        for name, source in self._sources.items():
            if not source.is_loaded:
                continue
            
            schema = self._schemas.get(name)
            columns = []
            
            for col in source.get_column_names():
                col_type = DataType.TEXT
                if schema:
                    col_schema = schema.get_column(col)
                    if col_schema:
                        col_type = col_schema.data_type
                
                columns.append((col, col_type))
            
            result[name] = columns
        
        return result
    
    def get_column_values(self, column: str, source_name: Optional[str] = None) -> List[Any]:
        """
        Get all values for a column (for histograms, etc.).
        
        Args:
            column: Column name
            source_name: Optional specific source
            
        Returns:
            List of values
        """
        if source_name:
            source = self._sources.get(source_name)
            if source and source.is_loaded:
                return source.get_column_values(column)
            return []
        
        # Collect from all sources
        values = []
        for name, source in self._sources.items():
            if source.is_loaded:
                values.extend(source.get_column_values(column))
        
        return values
    
    def query(
        self,
        filters: List[Dict[str, Any]],
        source_name: Optional[str] = None
    ) -> Set[Tuple[str, float]]:
        """
        Query data with filter criteria across all sources.
        
        Delegates to IndexedDataSource.query() and aggregates results.
        
        Args:
            filters: List of filter dicts, each with:
                - column: str - Column name to filter
                - operator: str - Operator ('=', '>', '<', '>=', '<=', '!=', 'between', 'contains', 'in', etc.)
                - value: Any - Filter value
                - value2: Any - Second value for 'between' operator
            source_name: Optional specific source to query (queries all if None)
            
        Returns:
            Set of (hole_id, depth_to) tuples matching all filters
            
        Example:
            keys = store.query([
                {'column': 'fe_pct_best', 'operator': '>', 'value': 50},
                {'column': 'sio2_pct_best', 'operator': '<', 'value': 10}
            ])
        """
        if not filters:
            # No filters - return all keys
            all_keys = set()
            sources_to_query = [self._sources[source_name]] if source_name else list(self._sources.values())
            
            for source in sources_to_query:
                if source.is_loaded and source.df is not None:
                    df = source.df
                    hole_col = source.schema.hole_id_column.lower() if source.schema.hole_id_column else 'holeid'
                    depth_col = source.schema.to_column.lower() if source.schema.to_column else 'to'
                    
                    if hole_col in df.columns and depth_col in df.columns:
                        for _, row in df[[hole_col, depth_col]].iterrows():
                            try:
                                all_keys.add((str(row[hole_col]).upper(), float(row[depth_col])))
                            except (ValueError, TypeError):
                                continue
            return all_keys
        
        # Query specific source or all sources
        if source_name:
            source = self._sources.get(source_name)
            if source and source.is_loaded:
                return source.query(filters, return_keys=True)
            return set()
        
        # Query all sources and union results (OR across sources)
        all_matching_keys = set()
        
        for source in self._sources.values():
            if not source.is_loaded:
                continue
            
            # Get keys from this source
            source_keys = source.query(filters, return_keys=True)
            all_matching_keys.update(source_keys)
        
        return all_matching_keys
    
    # =========================================================================
    # Metadata
    # =========================================================================
    
    def get_unique_holes(self) -> Set[str]:
        """Get all unique hole IDs across all sources."""
        holes = set()
        for source in self._sources.values():
            if source.is_loaded:
                holes.update(source.get_unique_holes())
        return holes
    
    def get_data_sources(self) -> Dict[str, 'IndexedDataSource']:
        """
        Get all data sources.
        
        Returns:
            Dictionary of {source_name: IndexedDataSource}
        """
        return self._sources.copy()
    
    def get_schemas(self) -> Dict[str, DataSourceSchema]:
        """
        Get all schemas.
        
        Returns:
            Dictionary of {source_name: DataSourceSchema}
        """
        return self._schemas.copy()
    
    @property
    def is_loaded(self) -> bool:
        """Whether any sources are loaded."""
        return any(s.is_loaded for s in self._sources.values())
    
    @property
    def total_rows(self) -> int:
        """Total rows across all sources."""
        return sum(s.row_count for s in self._sources.values() if s.is_loaded)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded data sources."""
        return {
            "sources_count": len(self._sources),
            "sources_loaded": sum(1 for s in self._sources.values() if s.is_loaded),
            "total_rows": self.total_rows,
            "total_columns": len(self._all_columns),
            "unique_holes": len(self.get_unique_holes()),
            "sources": {
                name: {
                    "is_loaded": source.is_loaded,
                    "row_count": source.row_count,
                    "columns": len(source.get_column_names()) if source.is_loaded else 0
                }
                for name, source in self._sources.items()
            }
        }
