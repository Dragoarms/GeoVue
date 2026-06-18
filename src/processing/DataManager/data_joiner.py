"""
data_joiner.py - Cross-table data joining with interval overlap detection.

Provides LEFT JOIN semantics where:
- All primary (left) rows are preserved
- Secondary (right) rows are matched by hole_id and interval overlap
- Missing secondary data results in NaN (not row loss)
- Overlap percentage is tracked for each match

Uses vectorized operations for O(n) complexity per hole instead of O(n²) iterrows.

Author: George Symonds
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IntervalMatch:
    """
    Represents a match between primary and secondary intervals.

    Attributes:
        primary_idx: Index in primary DataFrame
        secondary_idx: Index in secondary DataFrame (None if no match)
        hole_id: Hole identifier
        primary_from: Primary interval start
        primary_to: Primary interval end
        secondary_from: Secondary interval start (if matched)
        secondary_to: Secondary interval end (if matched)
        overlap_from: Start of overlap region
        overlap_to: End of overlap region
        overlap_pct: Percentage of primary interval covered by this match
    """
    primary_idx: int
    secondary_idx: Optional[int]
    hole_id: str
    primary_from: float
    primary_to: float
    secondary_from: Optional[float] = None
    secondary_to: Optional[float] = None
    overlap_from: Optional[float] = None
    overlap_to: Optional[float] = None
    overlap_pct: float = 0.0

    @property
    def has_match(self) -> bool:
        return self.secondary_idx is not None

    @property
    def overlap_length(self) -> float:
        if self.overlap_from is None or self.overlap_to is None:
            return 0.0
        return self.overlap_to - self.overlap_from


@dataclass
class JoinResult:
    """
    Result of a join operation with metadata about matches.

    Attributes:
        joined_df: The joined DataFrame (LEFT JOIN result)
        primary_source: Name of primary data source
        secondary_source: Name of secondary data source
        matches: List of all interval matches found
        unmatched_primary_count: Number of primary rows with no secondary match
        total_primary_rows: Total primary rows
        total_secondary_rows: Total secondary rows
    """
    joined_df: pd.DataFrame
    primary_source: str = ""
    secondary_source: str = ""
    matches: List[IntervalMatch] = field(default_factory=list)
    unmatched_primary_count: int = 0
    total_primary_rows: int = 0
    total_secondary_rows: int = 0

    def get_matches_for_interval(
        self,
        hole_id: str,
        from_depth: float,
        to_depth: float
    ) -> List[IntervalMatch]:
        """Get all secondary matches for a specific primary interval."""
        return [
            m for m in self.matches
            if m.hole_id.upper() == hole_id.upper()
            and m.primary_from == from_depth
            and m.primary_to == to_depth
            and m.has_match
        ]

    def get_coverage_for_hole(self, hole_id: str) -> float:
        """
        Calculate what percentage of primary intervals have secondary coverage.

        Returns:
            Float 0.0-1.0 representing coverage percentage
        """
        hole_matches = [m for m in self.matches if m.hole_id.upper() == hole_id.upper()]
        if not hole_matches:
            return 0.0

        total_primary_length = sum(m.primary_to - m.primary_from for m in hole_matches)
        if total_primary_length == 0:
            return 0.0

        # Sum overlap lengths (accounting for multiple matches per primary)
        primary_intervals = {}
        for m in hole_matches:
            key = (m.primary_from, m.primary_to)
            if key not in primary_intervals:
                primary_intervals[key] = []
            if m.has_match:
                primary_intervals[key].append(m.overlap_length)

        total_covered = 0.0
        for (p_from, p_to), overlaps in primary_intervals.items():
            interval_length = p_to - p_from
            if overlaps:
                covered = min(sum(overlaps), interval_length)
                total_covered += covered

        return total_covered / total_primary_length

    def summary(self) -> str:
        """Generate human-readable summary of the join."""
        matched = len([m for m in self.matches if m.has_match])
        return (
            f"Join Result: {self.primary_source} <- {self.secondary_source}\n"
            f"  Primary rows: {self.total_primary_rows}\n"
            f"  Secondary rows: {self.total_secondary_rows}\n"
            f"  Matched intervals: {matched}\n"
            f"  Unmatched primary: {self.unmatched_primary_count}\n"
            f"  Result rows: {len(self.joined_df)}"
        )


class DataJoiner:
    """
    Joins DataFrames using interval overlap matching.

    Implements LEFT JOIN semantics where all primary rows are preserved
    and secondary rows are matched based on hole_id and depth interval overlap.

    Uses vectorized merge + overlap filter instead of nested iterrows.
    """

    def __init__(self):
        self._debug = True  # Enable verbose logging

    def join(
        self,
        primary_df: pd.DataFrame,
        secondary_df: pd.DataFrame,
        primary_key_cols: Dict[str, str],
        secondary_key_cols: Dict[str, str],
        aggregate_secondary: bool = False,
        primary_source: str = "primary",
        secondary_source: str = "secondary",
    ) -> JoinResult:
        """
        Join two DataFrames using interval overlap matching.

        Vectorized approach: merge on hole_id, filter by overlap, then build result.
        """
        logger.info(f"Starting join: {primary_source} <- {secondary_source}")
        logger.debug(f"  Primary rows: {len(primary_df)}, columns: {list(primary_df.columns)}")
        logger.debug(f"  Secondary rows: {len(secondary_df)}, columns: {list(secondary_df.columns)}")

        self._validate_key_cols(primary_df, primary_key_cols, "primary")
        self._validate_key_cols(secondary_df, secondary_key_cols, "secondary")

        p_hole = primary_key_cols["hole"]
        p_from = primary_key_cols["from"]
        p_to = primary_key_cols["to"]
        s_hole = secondary_key_cols["hole"]
        s_from = secondary_key_cols["from"]
        s_to = secondary_key_cols["to"]

        primary = primary_df.copy()
        secondary = secondary_df.copy()
        primary["_hole_upper"] = primary[p_hole].astype(str).str.upper().str.strip()
        secondary["_hole_upper"] = secondary[s_hole].astype(str).str.upper().str.strip()

        p_f = pd.to_numeric(primary[p_from], errors="coerce")
        p_t = pd.to_numeric(primary[p_to], errors="coerce")
        s_f = pd.to_numeric(secondary[s_from], errors="coerce")
        s_t = pd.to_numeric(secondary[s_to], errors="coerce")

        primary["_pf"] = p_f
        primary["_pt"] = p_t
        primary["_pidx"] = primary.index
        secondary["_sf"] = s_f
        secondary["_st"] = s_t
        secondary["_sidx"] = secondary.index

        # Vectorized: merge on hole (creates cross product per hole)
        cross = pd.merge(
            primary[["_hole_upper", "_pidx", "_pf", "_pt"]],
            secondary[["_hole_upper", "_sidx", "_sf", "_st"]],
            on="_hole_upper",
            how="left",
        )

        # Filter by overlap: (primary_from < secondary_to) AND (primary_to > secondary_from)
        overlap_mask = (cross["_pf"] < cross["_st"]) & (cross["_pt"] > cross["_sf"])
        overlaps = cross[overlap_mask].copy()

        # Compute overlap metrics
        overlaps["_overlap_from"] = np.maximum(overlaps["_pf"], overlaps["_sf"])
        overlaps["_overlap_to"] = np.minimum(overlaps["_pt"], overlaps["_st"])
        overlaps["_overlap_len"] = overlaps["_overlap_to"] - overlaps["_overlap_from"]
        primary_len = overlaps["_pt"] - overlaps["_pf"]
        overlaps["_overlap_pct"] = np.where(
            primary_len > 0,
            overlaps["_overlap_len"] / primary_len,
            0.0,
        )

        # Build matches list (for JoinResult API compatibility)
        matches = []
        matched_primary_set = set(overlaps["_pidx"].unique())

        # Build from overlaps (avoid iterrows)
        for i in range(len(overlaps)):
            row = overlaps.iloc[i]
            matches.append(
                IntervalMatch(
                    primary_idx=row["_pidx"],
                    secondary_idx=row["_sidx"],
                    hole_id=row["_hole_upper"],
                    primary_from=float(row["_pf"]),
                    primary_to=float(row["_pt"]),
                    secondary_from=float(row["_sf"]),
                    secondary_to=float(row["_st"]),
                    overlap_from=float(row["_overlap_from"]),
                    overlap_to=float(row["_overlap_to"]),
                    overlap_pct=float(row["_overlap_pct"]),
                )
            )
            if self._debug:
                logger.debug(
                    f"  Match: {row['_hole_upper']} primary[{row['_pf']}-{row['_pt']}] <-> "
                    f"secondary[{row['_sf']}-{row['_st']}] overlap={row['_overlap_pct']:.1%}"
                )

        # Add "no match" entries for primary rows with no overlaps
        unmatched_idx = primary.index.difference(matched_primary_set)
        if len(unmatched_idx) > 0:
            unmatched_df = primary.loc[unmatched_idx]
            for p_idx in unmatched_idx:
                p_row = unmatched_df.loc[p_idx]
                matches.append(
                    IntervalMatch(
                        primary_idx=p_idx,
                        secondary_idx=None,
                        hole_id=p_row["_hole_upper"],
                        primary_from=float(p_row["_pf"]) if pd.notna(p_row["_pf"]) else 0.0,
                        primary_to=float(p_row["_pt"]) if pd.notna(p_row["_pt"]) else 0.0,
                    )
                )

        matched_secondary_cols = [
            c for c in secondary_df.columns
            if c not in [s_hole, s_from, s_to, "_hole_upper"]
        ]

        if aggregate_secondary:
            joined_df = self._build_aggregated_join_vectorized(
                primary_df, secondary_df, primary, overlaps,
                matched_secondary_cols, p_hole, p_from, p_to,
            )
        else:
            joined_df = self._build_expanded_join_vectorized(
                primary_df, secondary_df, primary, overlaps,
                matched_secondary_cols,
            )

        if "_hole_upper" in joined_df.columns:
            joined_df = joined_df.drop(columns=["_hole_upper"], errors="ignore")

        unmatched = sum(1 for m in matches if not m.has_match)

        logger.info(
            f"Join complete: {len(joined_df)} result rows, "
            f"{len(matches) - unmatched} matches, {unmatched} unmatched primary"
        )

        return JoinResult(
            joined_df=joined_df,
            primary_source=primary_source,
            secondary_source=secondary_source,
            matches=matches,
            unmatched_primary_count=unmatched,
            total_primary_rows=len(primary_df),
            total_secondary_rows=len(secondary_df),
        )

    def _build_expanded_join_vectorized(
        self,
        primary_df: pd.DataFrame,
        secondary_df: pd.DataFrame,
        primary: pd.DataFrame,
        overlaps: pd.DataFrame,
        secondary_cols: List[str],
    ) -> pd.DataFrame:
        """Build expanded join (one row per match) using vectorized merge."""
        if overlaps.empty:
            result = primary_df.copy()
            for col in secondary_cols:
                result[col] = np.nan
            result["_overlap_pct"] = np.nan
            return result

        # Merge primary rows with overlaps (one row per match)
        expanded = overlaps[["_pidx", "_sidx", "_overlap_pct"]].merge(
            primary_df, left_on="_pidx", right_index=True, how="left"
        )
        for col in secondary_cols:
            expanded[col] = secondary_df.loc[overlaps["_sidx"].values, col].values
        expanded = expanded.drop(columns=["_pidx", "_sidx"], errors="ignore")

        # Add primary rows with no match
        matched_primary = overlaps["_pidx"].unique()
        unmatched_mask = ~primary_df.index.isin(matched_primary)
        if unmatched_mask.any():
            unmatched_rows = primary_df.loc[unmatched_mask].copy()
            for col in secondary_cols:
                unmatched_rows[col] = np.nan
            unmatched_rows["_overlap_pct"] = np.nan
            expanded = pd.concat([expanded, unmatched_rows], ignore_index=True)

        return expanded

    def _build_aggregated_join_vectorized(
        self,
        primary_df: pd.DataFrame,
        secondary_df: pd.DataFrame,
        primary: pd.DataFrame,
        overlaps: pd.DataFrame,
        secondary_cols: List[str],
        p_hole: str,
        p_from: str,
        p_to: str,
    ) -> pd.DataFrame:
        """Build aggregated join (one row per primary) using vectorized groupby."""
        result = primary_df.copy()

        agg_suffixes = ["_mean", "_min", "_max"]
        for col in secondary_cols:
            for suf in agg_suffixes:
                result[f"{col}{suf}"] = np.nan
        result["_match_count"] = 0
        result["_total_overlap_pct"] = 0.0

        if overlaps.empty:
            return result

        overlaps_with_sec = overlaps.merge(
            secondary_df[secondary_cols],
            left_on="_sidx",
            right_index=True,
            how="left",
        )

        for col in secondary_cols:
            if col not in overlaps_with_sec.columns:
                continue
            vals = pd.to_numeric(overlaps_with_sec[col], errors="coerce")
            grp = overlaps_with_sec.assign(_val=vals).groupby("_pidx")["_val"]
            result[f"{col}_mean"] = grp.mean().reindex(result.index).values
            result[f"{col}_min"] = grp.min().reindex(result.index).values
            result[f"{col}_max"] = grp.max().reindex(result.index).values

        match_count = overlaps.groupby("_pidx").size().reindex(result.index).fillna(0).astype(int)
        result["_match_count"] = match_count.values
        total_overlap = overlaps.groupby("_pidx")["_overlap_pct"].sum().reindex(result.index).fillna(0)
        result["_total_overlap_pct"] = np.minimum(total_overlap.values, 1.0)

        return result

    def _validate_key_cols(self, df: pd.DataFrame, key_cols: Dict[str, str], name: str):
        """Validate that key columns exist in DataFrame."""
        required = {"hole", "from", "to"}
        if not required.issubset(key_cols.keys()):
            raise ValueError(f"{name}_key_cols must have 'hole', 'from', 'to' keys")
        for key, col in key_cols.items():
            if col not in df.columns:
                raise ValueError(f"{name} DataFrame missing column '{col}' (for {key})")
