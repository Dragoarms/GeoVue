"""
interval_merger.py - Range-based interval merging for geological data sources.

This module provides efficient range-based merging of geological data from sources
with different interval sizes (e.g., 1m assay intervals vs 2.63m geology intervals).

Unlike exact depth matching which fails when intervals don't align, range-based
merging finds overlapping intervals and assigns values from the best-matching
(longest overlap) secondary interval.

Key Features:
- Vectorized operations where possible for performance
- Handles case-insensitive hole ID matching
- Supports multiple secondary sources
- Provides detailed match statistics

Usage:
    >>> merger = IntervalMerger(geological_store)
    >>> merged_df = merger.merge_sources(
    ...     primary_source="exassay",
    ...     secondary_sources=["exgeologyRC"],
    ...     columns_to_merge=["total_gangue_pct", "lithology"]
    ... )

Author: George Symonds / Claude
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MergeStats:
    """Statistics from an interval merge operation."""
    primary_source: str = ""
    secondary_sources: List[str] = field(default_factory=list)
    total_primary_rows: int = 0
    rows_matched: int = 0
    rows_unmatched: int = 0
    holes_processed: int = 0
    columns_merged: List[str] = field(default_factory=list)

    @property
    def match_rate(self) -> float:
        """Percentage of rows that were matched."""
        if self.total_primary_rows == 0:
            return 0.0
        return (self.rows_matched / self.total_primary_rows) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/display."""
        return {
            "primary_source": self.primary_source,
            "secondary_sources": self.secondary_sources,
            "total_primary_rows": self.total_primary_rows,
            "rows_matched": self.rows_matched,
            "rows_unmatched": self.rows_unmatched,
            "match_rate_pct": round(self.match_rate, 1),
            "holes_processed": self.holes_processed,
            "columns_merged": self.columns_merged,
        }


def _detect_key_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
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


def merge_intervals_by_range(
    primary_df: pd.DataFrame,
    secondary_df: pd.DataFrame,
    primary_keys: Optional[Dict[str, str]] = None,
    secondary_keys: Optional[Dict[str, str]] = None,
    cols_to_merge: Optional[List[str]] = None,
    hole_ids: Optional[Set[str]] = None
) -> Tuple[pd.DataFrame, MergeStats]:
    """
    Merge secondary source columns into primary using range overlap.

    For each primary interval, finds the secondary interval(s) that overlap
    and assigns values based on the dominant (longest overlap) secondary interval.

    This handles the common case where data sources have different interval sizes
    (e.g., 1m assay intervals vs 2.63m geology intervals).

    Args:
        primary_df: Primary DataFrame (e.g., exassay)
        secondary_df: Secondary DataFrame (e.g., exgeologyRC)
        primary_keys: Dict with 'hole_id', 'from', 'to' column names (auto-detected if None)
        secondary_keys: Dict with 'hole_id', 'from', 'to' column names (auto-detected if None)
        cols_to_merge: Columns from secondary to bring into primary (all unique if None)
        hole_ids: Optional set of specific holes to process (all if None)

    Returns:
        Tuple of (merged DataFrame, MergeStats)
    """
    stats = MergeStats()

    # Initial logging of merge inputs
    logger.debug(
        f"Starting range merge:\n"
        f"  Primary: {len(primary_df):,} rows x {len(primary_df.columns)} cols\n"
        f"  Secondary: {len(secondary_df):,} rows x {len(secondary_df.columns)} cols\n"
        f"  Hole filter: {len(hole_ids) if hole_ids else 'None (all holes)'}"
    )

    if primary_df.empty:
        logger.warning("Primary DataFrame is empty, nothing to merge")
        return primary_df.copy(), stats

    if secondary_df.empty:
        logger.warning("Secondary DataFrame is empty, nothing to merge")
        return primary_df.copy(), stats

    # Auto-detect key columns if not provided
    if primary_keys is None:
        primary_keys = _detect_key_columns(primary_df)
    if secondary_keys is None:
        secondary_keys = _detect_key_columns(secondary_df)

    # Validate key columns exist
    for key_type in ['hole_id', 'from', 'to']:
        if not primary_keys.get(key_type):
            raise ValueError(f"Could not detect '{key_type}' column in primary DataFrame")
        if not secondary_keys.get(key_type):
            raise ValueError(f"Could not detect '{key_type}' column in secondary DataFrame")

    # Auto-detect columns to merge if not specified
    if cols_to_merge is None:
        primary_cols_lower = {c.lower() for c in primary_df.columns}
        cols_to_merge = [
            c for c in secondary_df.columns
            if c.lower() not in primary_cols_lower and not c.startswith('_')
        ]

    if not cols_to_merge:
        logger.warning("No columns to merge - all secondary columns already in primary")
        return primary_df.copy(), stats

    stats.columns_merged = cols_to_merge

    # Create working copies
    primary = primary_df.copy()
    secondary = secondary_df.copy()

    # Normalize hole IDs for matching
    primary['_hole_upper'] = primary[primary_keys['hole_id']].astype(str).str.upper().str.strip()
    secondary['_hole_upper'] = secondary[secondary_keys['hole_id']].astype(str).str.upper().str.strip()

    # Convert depths to numeric
    primary['_from'] = pd.to_numeric(primary[primary_keys['from']], errors='coerce')
    primary['_to'] = pd.to_numeric(primary[primary_keys['to']], errors='coerce')
    secondary['_from'] = pd.to_numeric(secondary[secondary_keys['from']], errors='coerce')
    secondary['_to'] = pd.to_numeric(secondary[secondary_keys['to']], errors='coerce')

    # Get holes to process and filter dataframes when requested
    if hole_ids:
        holes_to_process = [h.upper() for h in hole_ids]
        primary = primary[primary['_hole_upper'].isin(holes_to_process)].copy()
        secondary = secondary[secondary['_hole_upper'].isin(holes_to_process)].copy()
    else:
        holes_to_process = primary['_hole_upper'].unique()

    stats.total_primary_rows = len(primary)

    # Build best-match mapping using chunked processing to avoid memory explosion
    # Process holes individually to limit cross-product size
    primary["_pidx"] = primary.index
    secondary["_sidx"] = secondary.index

    # Pre-extract numpy arrays for faster access
    sec_hole = secondary["_hole_upper"].values
    sec_from = secondary["_from"].values
    sec_to = secondary["_to"].values
    sec_idx = secondary["_sidx"].values

    # Build secondary lookup by hole (numpy arrays for speed)
    secondary_by_hole: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for hole in secondary["_hole_upper"].unique():
        mask = sec_hole == hole
        secondary_by_hole[hole] = (
            sec_from[mask],
            sec_to[mask],
            sec_idx[mask],
        )

    # Process all primary rows vectorized per hole
    best_match_sidx = pd.Series(index=primary.index, dtype='object')
    matched_count = 0

    for hole in primary["_hole_upper"].unique():
        p_mask = primary["_hole_upper"] == hole
        p_indices = primary.index[p_mask]
        p_from = primary.loc[p_indices, "_from"].values
        p_to = primary.loc[p_indices, "_to"].values

        if hole not in secondary_by_hole:
            continue

        s_from, s_to, s_sidx = secondary_by_hole[hole]

        # Vectorized overlap detection using broadcasting
        # Shape: (n_primary, n_secondary)
        overlap_start = np.maximum(p_from[:, None], s_from[None, :])
        overlap_end = np.minimum(p_to[:, None], s_to[None, :])
        overlap_len = overlap_end - overlap_start
        overlap_len = np.where(overlap_len > 0, overlap_len, -np.inf)

        # Best match per primary (argmax along secondary axis)
        has_overlap = (overlap_len > 0).any(axis=1)
        best_local_idx = np.argmax(overlap_len, axis=1)

        # Assign best secondary index where overlap exists (vectorized)
        matched_pidx = p_indices[has_overlap]
        matched_sidx = s_sidx[best_local_idx[has_overlap]]
        best_match_sidx.loc[matched_pidx] = matched_sidx
        matched_count += len(matched_pidx)

    stats.rows_matched = matched_count
    stats.rows_unmatched = stats.total_primary_rows - matched_count
    stats.holes_processed = primary["_hole_upper"].nunique()

    # Vectorized column assignment
    valid_matches = best_match_sidx.dropna()
    if not valid_matches.empty:
        for col in cols_to_merge:
            if col in secondary.columns:
                primary.loc[valid_matches.index, col] = secondary.loc[
                    valid_matches.values, col
                ].values

    # Clean up temporary columns
    primary = primary.drop(
        columns=['_hole_upper', '_from', '_to', '_pidx'], errors='ignore'
    )

    # Detailed logging of merge operation
    cols_display = stats.columns_merged[:5]
    cols_suffix = f" (+{len(stats.columns_merged) - 5} more)" if len(stats.columns_merged) > 5 else ""

    logger.info(
        f"RANGE MERGE COMPLETE:\n"
        f"  Primary rows: {stats.total_primary_rows:,} | Matched: {stats.rows_matched:,} ({stats.match_rate:.1f}%)\n"
        f"  Holes processed: {stats.holes_processed:,}\n"
        f"  Columns merged ({len(stats.columns_merged)}): {', '.join(cols_display)}{cols_suffix}"
    )

    if stats.rows_unmatched > 0:
        logger.debug(
            f"  Unmatched rows: {stats.rows_unmatched:,} "
            f"(may be intervals outside secondary source coverage)"
        )

    return primary, stats


class IntervalMerger:
    """
    High-level interface for merging geological data sources with different intervals.

    Works with GeologicalStore to merge data from multiple sources into a single
    DataFrame suitable for QAQC analysis.

    Example:
        >>> merger = IntervalMerger(geological_store)
        >>> df, stats = merger.merge_sources(
        ...     primary_source="exassay",
        ...     secondary_sources=["exgeologyRC"],
        ... )
    """

    def __init__(self, geological_store):
        """
        Initialize the merger.

        Args:
            geological_store: GeologicalStore instance with loaded data sources
        """
        self._geological_store = geological_store

    def merge_sources(
        self,
        primary_source: str,
        secondary_sources: List[str],
        columns_to_merge: Optional[Dict[str, List[str]]] = None,
        hole_ids: Optional[Set[str]] = None,
    ) -> Tuple[pd.DataFrame, MergeStats]:
        """
        Merge data from multiple sources using range-based interval matching.

        Args:
            primary_source: Name of the primary data source (e.g., "exassay")
            secondary_sources: List of secondary source names (e.g., ["exgeologyRC"])
            columns_to_merge: Optional dict of {source_name: [columns]} to merge.
                             If None, merges all unique columns from each secondary.
            hole_ids: Optional set of specific holes to process

        Returns:
            Tuple of (merged DataFrame, combined MergeStats)
        """
        # Get primary DataFrame
        primary_src = self._geological_store.get_source(primary_source)
        if primary_src is None or primary_src.df is None or primary_src.df.empty:
            raise ValueError(f"Primary source '{primary_source}' not found or empty")

        result_df = primary_src.df.copy()
        combined_stats = MergeStats(
            primary_source=primary_source,
            secondary_sources=list(secondary_sources),
            total_primary_rows=len(result_df),
        )

        # Merge each secondary source
        for sec_name in secondary_sources:
            sec_src = self._geological_store.get_source(sec_name)
            if sec_src is None or sec_src.df is None or sec_src.df.empty:
                logger.warning(f"Secondary source '{sec_name}' not found or empty, skipping")
                continue

            # Get columns to merge for this source
            cols = None
            if columns_to_merge and sec_name in columns_to_merge:
                cols = columns_to_merge[sec_name]

            sec_rows = len(sec_src.df)
            sec_cols_count = len(sec_src.df.columns)
            cols_desc = f"specific columns: {cols}" if cols else f"all unique columns ({sec_cols_count} available)"
            logger.info(
                f"MERGING SOURCE: '{sec_name}' ({sec_rows:,} rows) -> '{primary_source}'\n"
                f"  Merge mode: {cols_desc}"
            )

            result_df, stats = merge_intervals_by_range(
                primary_df=result_df,
                secondary_df=sec_src.df,
                cols_to_merge=cols,
                hole_ids=hole_ids,
            )

            # Accumulate stats
            combined_stats.rows_matched = stats.rows_matched
            combined_stats.rows_unmatched = stats.rows_unmatched
            combined_stats.holes_processed = max(combined_stats.holes_processed, stats.holes_processed)
            combined_stats.columns_merged.extend(stats.columns_merged)

        return result_df, combined_stats

    def get_merged_dataframe_for_qaqc(
        self,
        primary_source: str = "exassay",
        merge_geology: bool = True,
        hole_ids: Optional[Set[str]] = None,
    ) -> Tuple[pd.DataFrame, MergeStats]:
        """
        Get a merged DataFrame optimized for QAQC analysis.

        This is a convenience method that automatically detects and merges
        relevant geology data (e.g., exgeologyRC, exgeologyDiamond) into
        the primary assay data.

        IMPORTANT: For QAQC purposes, the 'strat' column from geology sources
        (exgeologyRC) takes priority over any existing 'strat' in the primary.
        If the primary has a 'strat' column, it will be renamed to 'strat_assay'
        to preserve it while allowing geology strat to be the main classification.

        Args:
            primary_source: Primary data source (default: exassay)
            merge_geology: Whether to merge geology sources (default: True)
            hole_ids: Optional set of specific holes to process

        Returns:
            Tuple of (merged DataFrame, MergeStats)
        """
        secondary_sources = []

        if merge_geology:
            # Auto-detect geology sources - exgeologyRC has priority
            available = self._geological_store.list_sources()
            for src_name in ['exgeologyRC', 'exgeologyDiamond', 'exgeology']:
                if src_name.lower() in [s.lower() for s in available]:
                    actual_name = next(s for s in available if s.lower() == src_name.lower())
                    secondary_sources.append(actual_name)

        if not secondary_sources:
            # No secondary sources - return primary as-is
            primary_src = self._geological_store.get_source(primary_source)
            if primary_src is None or primary_src.df is None:
                raise ValueError(f"Primary source '{primary_source}' not found")
            return primary_src.df.copy(), MergeStats(
                primary_source=primary_source,
                total_primary_rows=len(primary_src.df),
            )

        # Get the primary DataFrame and rename 'strat' if it exists
        # This ensures geology strat (from exgeologyRC) takes priority
        primary_src = self._geological_store.get_source(primary_source)
        if primary_src is None or primary_src.df is None:
            raise ValueError(f"Primary source '{primary_source}' not found")

        result_df = primary_src.df.copy()

        # Check for existing 'strat' column in primary (case-insensitive)
        strat_cols = [c for c in result_df.columns if c.lower() == 'strat']
        if strat_cols:
            # Rename primary's strat to preserve it, allowing geology strat to be merged
            for col in strat_cols:
                new_name = f"{col}_assay"
                if new_name not in result_df.columns:
                    result_df = result_df.rename(columns={col: new_name})
                    logger.info(f"Renamed '{col}' to '{new_name}' to prioritize geology strat")

        # Now merge geology sources - exgeologyRC's strat will be the main one
        combined_stats = MergeStats(
            primary_source=primary_source,
            secondary_sources=list(secondary_sources),
            total_primary_rows=len(result_df),
        )

        for sec_name in secondary_sources:
            sec_src = self._geological_store.get_source(sec_name)
            if sec_src is None or sec_src.df is None or sec_src.df.empty:
                logger.warning(f"Secondary source '{sec_name}' not found or empty, skipping")
                continue

            sec_rows = len(sec_src.df)
            sec_cols = list(sec_src.df.columns)
            logger.info(
                f"MERGING GEOLOGY SOURCE: '{sec_name}' ({sec_rows:,} rows) -> '{primary_source}'\n"
                f"  Available columns: {sec_cols[:8]}{'...' if len(sec_cols) > 8 else ''}"
            )

            result_df, stats = merge_intervals_by_range(
                primary_df=result_df,
                secondary_df=sec_src.df,
                cols_to_merge=None,  # Auto-detect columns
                hole_ids=hole_ids,
            )

            combined_stats.rows_matched = stats.rows_matched
            combined_stats.rows_unmatched = stats.rows_unmatched
            combined_stats.holes_processed = max(combined_stats.holes_processed, stats.holes_processed)
            combined_stats.columns_merged.extend(stats.columns_merged)

        return result_df, combined_stats

    def _get_merged_dataframe_for_qaqc_legacy(
        self,
        primary_source: str = "exassay",
        merge_geology: bool = True,
        hole_ids: Optional[Set[str]] = None,
    ) -> Tuple[pd.DataFrame, MergeStats]:
        """Legacy version without strat priority handling - kept for reference."""
        secondary_sources = []

        if merge_geology:
            available = self._geological_store.list_sources()
            for src_name in ['exgeologyRC', 'exgeologyDiamond', 'exgeology']:
                if src_name.lower() in [s.lower() for s in available]:
                    actual_name = next(s for s in available if s.lower() == src_name.lower())
                    secondary_sources.append(actual_name)

        if not secondary_sources:
            primary_src = self._geological_store.get_source(primary_source)
            if primary_src is None or primary_src.df is None:
                raise ValueError(f"Primary source '{primary_source}' not found")
            return primary_src.df.copy(), MergeStats(
                primary_source=primary_source,
                total_primary_rows=len(primary_src.df),
            )

        return self.merge_sources(
            primary_source=primary_source,
            secondary_sources=secondary_sources,
            hole_ids=hole_ids,
        )
