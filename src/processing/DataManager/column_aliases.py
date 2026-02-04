"""
column_aliases.py - Standard column name aliases for geological data.

Provides a mapping system that allows analysis code to use standard column names
(like "fe_pct", "total_gangue") while the data layer resolves them to actual
column names in the DataFrames.

This decouples analysis code from specific CSV column naming conventions.

Usage:
    >>> resolver = ColumnResolver(df)
    >>> fe_col = resolver.get("fe_pct")  # Returns actual column name or None
    >>> fe_data = df[fe_col] if fe_col else None

Author: George Symonds / Claude
"""

import logging
from typing import Any, Dict, List, Optional, Set
import pandas as pd

logger = logging.getLogger(__name__)


# Standard column aliases grouped by semantic meaning
# Each key is a standard name, value is list of possible actual column names (case-insensitive)
COLUMN_ALIASES: Dict[str, List[str]] = {
    # === Key columns ===
    "hole_id": ["holeid", "hole_id", "bhid", "drillhole_id", "dhid", "hole"],
    "depth_from": ["sampfrom", "geolfrom", "geology_from", "logging_from", "rc_from", "depth_from_geol", "from", "depth_from", "from_depth", "interval_from"],
    "depth_to": ["sampto", "geolto", "geology_to", "logging_to", "rc_to", "depth_to_geol", "to", "depth_to", "to_depth", "interval_to"],

    # === Survey columns (drillhole deviation) ===
    "survey_depth": [
        "depth", "survey_depth", "surveydepth", "dip_depth",
        "measurement_depth", "point_depth", "md", "measured_depth",
    ],
    "azimuth": ["azimuth", "azim", "bearing", "direction", "hole_azimuth", "azi"],
    "dip": ["dip", "inclination", "incl", "dip_angle", "hole_dip", "deviation"],

    # === Assay columns (Iron ore) ===
    "fe_pct": ["fe_pct", "fe_pct_best", "fe%", "fe_percent", "fe_grade", "fe"],
    "sio2_pct": ["sio2_pct", "sio2_pct_best", "sio2%", "sio2_percent", "sio2", "silica_pct"],
    "al2o3_pct": ["al2o3_pct", "al2o3_pct_best", "al2o3%", "al2o3_percent", "al2o3", "alumina_pct"],
    "p_pct": ["p_pct", "p_pct_best", "p%", "p_percent", "phosphorus_pct", "p"],
    "s_pct": ["s_pct", "s_pct_best", "s%", "s_percent", "sulphur_pct", "sulfur_pct", "s"],
    "loi_pct": ["loi_pct", "loi_pct_best", "loi%", "loi", "loss_on_ignition"],
    "loi_1000_pct": ["loi_1000_pct", "loi1000", "loi_1000", "loi1000_pct", "loi_pct_1000"],
    "loi_425_pct": ["loi_425_pct", "loi425", "loi_425", "loi425_pct", "loi_pct_425"],
    "carbonate_gangue_pct": ["carbonate_gangue_pct", "carbonate_gangue", "carbonate_gangue_logged", "carbonate_pct"],
    "sulphide_gangue_pct": ["sulphide_gangue_pct", "sulphide_gangue", "sulphide_gangue_logged"],
    "manganese_gangue_pct": ["manganese_gangue_pct", "manganese_gangue", "manganese_gangue_logged"],
    "mafics_gangue_pct": ["mafics_gangue_pct", "mafics_gangue", "mafics_gangue_logged"],
    "magnesium_gangue_pct": ["magnesium_gangue_pct", "magnesium_gangue", "magnesium_gangue_logged"],
    "zonation_hy_pct": ["zonation_hy_pct", "zonation_hy", "hy_pct", "goethite_pct"],
    "mn_pct": ["mn_pct", "mn_pct_best", "mn%", "mn_percent", "manganese_pct", "mn"],
    "cao_pct": ["cao_pct", "cao_pct_best", "cao%", "cao_percent", "cao"],
    "mgo_pct": ["mgo_pct", "mgo_pct_best", "mgo%", "mgo_percent", "mgo"],
    "k2o_pct": ["k2o_pct", "k2o_pct_best", "k2o%", "k2o_percent", "k2o"],
    "na2o_pct": ["na2o_pct", "na2o_pct_best", "na2o%", "na2o_percent", "na2o"],
    "tio2_pct": ["tio2_pct", "tio2_pct_best", "tio2%", "tio2_percent", "tio2"],

    # === RC Logging columns ===
    "total_gangue_pct": [
        "total_gangue_pct", "total gangue logged", "total_gangue",
        "gangue_pct", "gangue_logged", "totalgangue"
    ],
    "loggedby": [
        "loggedby", "loggedby_d", "logged_by", "logged_by_d",
        "logger", "loggedbycode", "loggedbycode_d"
    ],
    "drilldate": [
        "drilldate", "drill_date", "drilldate_d", "drill_date_d",
        "drilldate_dt", "drill_date_dt"
    ],
    "hardness": ["hardness", "hardness_code", "hard", "hardness_logged"],
    "oxidation": ["oxidation", "oxidation_code", "ox", "oxidation_state"],
    "moisture": ["moisture", "moisture_status", "moist", "moisture_state"],

    # === Geological columns ===
    "lithology": ["lithology", "lith", "lith_code", "rock_type", "rocktype"],
    "strat": ["strat", "strat_code", "stratigraphy", "stratcode"],
    "stratsum": ["stratsum", "strat_summary", "stratsummary", "stratsum_code"],
    "prospect_d": ["prospect_d", "prospect", "prospect_d_code", "prospect_code", "prospectid"],
    "alteration": ["alteration", "alt", "alt_code", "alteration_code"],
    "mineralisation": ["mineralisation", "mineralization", "min", "min_code"],
    "weathering": ["weathering", "weather", "weathering_code", "weather_code"],
    "min_80_pct": ["min_80_pct", "min80pct", "min_80", "dominant_mineral"],
    "min_50_pct": ["min_50_pct", "min50pct", "min_50"],

    # === Mineral percentages (from RC logging) ===
    "min_he_pct": ["min_he_pct", "min_hematite_pct", "hematite_pct"],
    "min_go_pct": ["min_go_pct", "min_goethite_pct", "goethite_pct"],
    "min_li_pct": ["min_li_pct", "min_limonite_pct", "limonite_pct"],
    "min_ma_pct": ["min_ma_pct", "min_magnetite_pct", "magnetite_pct"],
    "magnetite_pct": ["magnetite_pct", "min_ma_pct", "min_magnetite_pct", "magnetic_pct"],
    "min_oc_pct": ["min_oc_pct", "min_ocite_pct", "ocite_pct", "ochre_pct"],
    "min_cl_pct": ["min_cl_pct", "min_clay_pct", "clay_pct"],
    "min_sh_pct": ["min_sh_pct", "min_shale_pct", "shale_pct"],
    "min_si_pct": ["min_si_pct", "min_silica_pct", "silica_logged_pct"],
    "min_qz_pct": ["min_qz_pct", "min_quartz_pct", "quartz_pct"],

    # === Collar / project (logging review report) ===
    "project_code": ["project_code", "projectcode", "project", "project_d", "projcode", "proj", "tenement", "prospect", "prospect_d"],
    "easting": ["easting", "east", "utm_e", "e", "x", "grid_e", "grid_easting"],
    "northing": ["northing", "north", "utm_n", "n", "y", "grid_n", "grid_northing"],
}


class ColumnResolver:
    """
    Resolves standard column names to actual DataFrame column names.

    Uses the COLUMN_ALIASES mapping to find the actual column name in a DataFrame
    given a standard/semantic column name.

    Example:
        >>> resolver = ColumnResolver(df)
        >>> fe_col = resolver.get("fe_pct")  # Returns "fe_pct_best" if that exists
        >>> if fe_col:
        ...     fe_data = df[fe_col]
    """

    def __init__(self, df: pd.DataFrame, custom_aliases: Optional[Dict[str, List[str]]] = None):
        """
        Initialize resolver with a DataFrame.

        Args:
            df: DataFrame to resolve columns against
            custom_aliases: Optional additional aliases to merge with defaults
        """
        self._df = df
        self._columns_lower = {c.lower(): c for c in df.columns}

        # Merge custom aliases with defaults
        self._aliases = dict(COLUMN_ALIASES)
        if custom_aliases:
            for key, values in custom_aliases.items():
                if key in self._aliases:
                    # Prepend custom values (higher priority)
                    self._aliases[key] = values + self._aliases[key]
                else:
                    self._aliases[key] = values

        # Cache resolved columns
        self._cache: Dict[str, Optional[str]] = {}

    def get(self, standard_name: str) -> Optional[str]:
        """
        Get the actual column name for a standard column name.

        Args:
            standard_name: Standard column name (e.g., "fe_pct")

        Returns:
            Actual column name in DataFrame, or None if not found
        """
        # Check cache first
        if standard_name in self._cache:
            return self._cache[standard_name]

        standard_lower = standard_name.lower()

        # First check if the standard name itself exists
        if standard_lower in self._columns_lower:
            actual = self._columns_lower[standard_lower]
            self._cache[standard_name] = actual
            return actual

        # Check aliases
        if standard_lower in self._aliases:
            for alias in self._aliases[standard_lower]:
                alias_lower = alias.lower()
                if alias_lower in self._columns_lower:
                    actual = self._columns_lower[alias_lower]
                    self._cache[standard_name] = actual
                    return actual

        # Not found
        self._cache[standard_name] = None
        return None

    def get_multiple(self, standard_names: List[str]) -> Dict[str, Optional[str]]:
        """
        Resolve multiple standard column names at once.

        Args:
            standard_names: List of standard column names

        Returns:
            Dictionary of {standard_name: actual_column_name_or_None}
        """
        return {name: self.get(name) for name in standard_names}

    def has(self, standard_name: str) -> bool:
        """Check if a standard column exists in the DataFrame."""
        return self.get(standard_name) is not None

    def has_all(self, standard_names: List[str]) -> bool:
        """Check if all standard columns exist in the DataFrame."""
        return all(self.has(name) for name in standard_names)

    def has_any(self, standard_names: List[str]) -> bool:
        """Check if any of the standard columns exist in the DataFrame."""
        return any(self.has(name) for name in standard_names)

    def get_series(self, standard_name: str) -> Optional[pd.Series]:
        """
        Get a Series for a standard column name.

        Args:
            standard_name: Standard column name

        Returns:
            pandas Series or None if column not found
        """
        col = self.get(standard_name)
        if col:
            return self._df[col]
        return None

    def available_columns(self) -> Dict[str, str]:
        """
        Get all available columns with their resolved names.

        Returns:
            Dictionary of {standard_name: actual_name} for columns that exist
        """
        result = {}
        for standard_name in self._aliases.keys():
            actual = self.get(standard_name)
            if actual:
                result[standard_name] = actual
        return result

    def missing_columns(self, required: List[str]) -> List[str]:
        """
        Get list of required columns that are missing.

        Args:
            required: List of standard column names that are required

        Returns:
            List of standard names that couldn't be resolved
        """
        return [name for name in required if not self.has(name)]


def resolve_column(df: pd.DataFrame, standard_name: str) -> Optional[str]:
    """
    Convenience function to resolve a single column name.

    Args:
        df: DataFrame to search
        standard_name: Standard column name

    Returns:
        Actual column name or None
    """
    resolver = ColumnResolver(df)
    return resolver.get(standard_name)


def resolve_columns(df: pd.DataFrame, standard_names: List[str]) -> Dict[str, Optional[str]]:
    """
    Convenience function to resolve multiple column names.

    Args:
        df: DataFrame to search
        standard_names: List of standard column names

    Returns:
        Dictionary of {standard_name: actual_name_or_None}
    """
    resolver = ColumnResolver(df)
    return resolver.get_multiple(standard_names)


def get_available_standard_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Get all standard columns that exist in a DataFrame.

    Args:
        df: DataFrame to search

    Returns:
        Dictionary of {standard_name: actual_name}
    """
    resolver = ColumnResolver(df)
    return resolver.available_columns()
