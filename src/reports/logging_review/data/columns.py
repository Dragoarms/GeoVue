"""Column resolution for logging review report (logger, drilldate, chemistry)."""
from typing import Dict, List, Optional

import pandas as pd

from processing.DataManager.column_aliases import ColumnResolver

LOGGER_COLUMN_PRIORITY = [
    "LoggedBy",       # Per-interval logger - most reliable
    "LoggedBy_D",     # Per-hole logger - fallback
    "LoggedByCode",
    "LoggedByCode_D",
    "Logger",
]

DRILLDATE_COLUMN_PRIORITY = [
    "drilldate",
    "drill_date",
    "drilldate_d",
    "drill_date_d",
]

MAJOR_ELEMENT_STANDARD = [
    "fe_pct",
    "sio2_pct",
    "al2o3_pct",
    "p_pct",
    "s_pct",
    "loi_pct",
    "mn_pct",
    "cao_pct",
    "mgo_pct",
    "k2o_pct",
    "na2o_pct",
    "tio2_pct",
]


def _find_column_case_insensitive(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    columns_lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        actual = columns_lower.get(candidate.lower())
        if actual:
            return actual
    return None


def resolve_logger_column(df: pd.DataFrame) -> Optional[str]:
    """Resolve the best logger column from a DataFrame."""
    return _find_column_case_insensitive(df, LOGGER_COLUMN_PRIORITY)


def resolve_drilldate_column(df: pd.DataFrame) -> Optional[str]:
    """Resolve the best drill date column from a DataFrame."""
    return _find_column_case_insensitive(df, DRILLDATE_COLUMN_PRIORITY)


def resolve_chemistry_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Resolve major chemistry columns from a DataFrame."""
    resolver = ColumnResolver(df)
    resolved = {}
    for standard in MAJOR_ELEMENT_STANDARD:
        actual = resolver.get(standard)
        if actual:
            resolved[standard] = actual
    return resolved
