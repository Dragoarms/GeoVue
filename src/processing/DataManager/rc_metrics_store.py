"""
rc_metrics_store.py - Pre-computed RC metrics storage with O(1) lookups.

This module provides:
- MineralCodeManager: Loads and provides access to mineral code properties
- RCMetricsCalculator: Calculates hardness, gangue, zonation metrics from logging data
- RCMetricsStore: Stores pre-computed metrics with efficient lookups

The RC metrics include:
- Weighted Hardness (based on mineral proportions and hardness values)
- Total Gangue Logged (sum of all gangue mineral percentages)
- Gangue breakdown by type (Silica, Aluminium, Carbonate, etc.)
- Si Gangue friability split (Friable vs Non-Friable based on hardness)
- Profile zonation percentages (Pr, Hy, De, Un, Hc)
- Specific mineral tracking (Quartz, Chert)

Usage:
    >>> from processing.DataManager.rc_metrics_store import RCMetricsStore
    >>> store = RCMetricsStore()
    >>> store.compute_from_dataframe(df)
    >>> metrics = store.get_metrics("BA0001", 45.0)

Author: George Symonds / Claude
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Path to mineral codes JSON (relative to src/)
DEFAULT_MINERAL_CODES_PATH = Path(__file__).parent.parent.parent / "resources" / "mineral_codes.json"


@dataclass
class MineralCodeInfo:
    """Information about a single mineral code."""
    code: str
    name: str
    hardness: int  # 0, 1, 3, or 5
    gangue_type: int  # 0=Mineralised, 1=Silica, 2=Aluminium, etc.
    gangue_name: str
    profile_zonation: str  # "Un", "Pr", "De", "Hy", "Hc", or mixed like "PrHy"
    numeric_zonation: Union[int, str]  # 1-4 or "Mix"


class MineralCodeManager:
    """
    Manages mineral code lookup data.

    Loads mineral codes from JSON and provides fast lookup of
    hardness, gangue type, and zonation information.
    """

    def __init__(self, json_path: Optional[Path] = None):
        """
        Initialize the mineral code manager.

        Args:
            json_path: Path to mineral_codes.json. Defaults to resources/mineral_codes.json
        """
        self.json_path = json_path or DEFAULT_MINERAL_CODES_PATH
        self._codes: Dict[str, MineralCodeInfo] = {}
        self._gangue_types: Dict[int, str] = {}
        self._zonation_mix_rules: Dict[str, Dict[str, float]] = {}
        self._quartz_variants: Set[str] = set()
        self._chert_variants: Set[str] = set()
        self._is_loaded = False

    def load(self) -> bool:
        """
        Load mineral codes from JSON file.

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.json_path.exists():
                logger.error(f"Mineral codes file not found: {self.json_path}")
                return False

            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Load codes
            for code, info in data.get("codes", {}).items():
                self._codes[code.upper()] = MineralCodeInfo(
                    code=code.upper(),
                    name=info.get("name", ""),
                    hardness=int(info.get("hardness", 0)),
                    gangue_type=int(info.get("gangue_type", 0)),
                    gangue_name=info.get("gangue_name", ""),
                    profile_zonation=info.get("profile_zonation", "Un"),
                    numeric_zonation=info.get("numeric_zonation", 1),
                )

            # Load gangue type mapping
            self._gangue_types = {
                int(k): v for k, v in data.get("gangue_types", {}).items()
            }

            # Load zonation mix rules
            self._zonation_mix_rules = data.get("zonation_mix_rules", {})

            # Load special mineral variants
            self._quartz_variants = set(v.upper() for v in data.get("quartz_variants", []))
            self._chert_variants = set(v.upper() for v in data.get("chert_variants", []))

            self._is_loaded = True
            logger.info(f"Loaded {len(self._codes)} mineral codes from {self.json_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading mineral codes: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        """Whether mineral codes have been loaded."""
        return self._is_loaded

    def get_code_info(self, code: str) -> Optional[MineralCodeInfo]:
        """
        Get information for a mineral code.

        Args:
            code: Mineral code (case-insensitive)

        Returns:
            MineralCodeInfo or None if not found
        """
        return self._codes.get(code.upper())

    def get_hardness(self, code: str) -> int:
        """Get hardness value for a mineral code (0 if not found)."""
        info = self.get_code_info(code)
        return info.hardness if info else 0

    def get_gangue_type(self, code: str) -> int:
        """Get gangue type for a mineral code (0=Mineralised if not found)."""
        info = self.get_code_info(code)
        return info.gangue_type if info else 0

    def get_gangue_name(self, code: str) -> str:
        """Get gangue name for a mineral code."""
        info = self.get_code_info(code)
        return info.gangue_name if info else "Unknown"

    def get_zonation(self, code: str) -> str:
        """Get profile zonation for a mineral code."""
        info = self.get_code_info(code)
        return info.profile_zonation if info else "Un"

    def get_zonation_split(self, zonation: str) -> Dict[str, float]:
        """
        Get zonation percentages for a zonation code.

        For simple zonations (Pr, Hy, De, Un, Hc), returns {zonation: 1.0}
        For mixed zonations (PrHy, HyDe, DeHy), uses mix rules (e.g., {Pr: 0.6, Hy: 0.4})

        Args:
            zonation: Zonation code

        Returns:
            Dictionary of zonation -> percentage
        """
        if zonation in self._zonation_mix_rules:
            return self._zonation_mix_rules[zonation]
        return {zonation: 1.0}

    def is_quartz_variant(self, code: str) -> bool:
        """Check if code is a quartz variant."""
        return code.upper() in self._quartz_variants

    def is_chert_variant(self, code: str) -> bool:
        """Check if code is a chert variant."""
        return code.upper() in self._chert_variants

    def get_all_codes(self) -> List[str]:
        """Get list of all known mineral codes."""
        return list(self._codes.keys())

    def get_magnetic_magnetite_codes(self) -> Set[str]:
        """
        Return mineral codes that are magnetic/magnetite (NOT non-magnetic variants).
        Uses name: contains 'magnetite' or 'magnetic', excludes 'non-magnetic' and 'non magnetic'.
        """
        out = set()
        for code, info in self._codes.items():
            name = (info.name or "").lower()
            if "non-magnetic" in name or "non magnetic" in name:
                continue
            if "magnetite" in name or "magnetic" in name:
                out.add(code)
        return out


@dataclass
class IntervalMetrics:
    """Computed metrics for a single interval."""
    hole_id: str
    depth_from: float
    depth_to: float

    # Hardness
    weighted_hardness: Optional[float] = None
    no_hardness_pct: float = 0.0

    # Total logged and normalisation
    total_logged_pct: float = 0.0
    normalisation_factor: float = 1.0

    # Gangue breakdown (normalised)
    total_gangue_pct: float = 0.0
    si_gangue_pct: float = 0.0
    al_gangue_pct: float = 0.0
    carbonate_gangue_pct: float = 0.0
    sulphide_gangue_pct: float = 0.0
    manganese_gangue_pct: float = 0.0
    mafics_gangue_pct: float = 0.0
    undefined_gangue_pct: float = 0.0
    magnesium_gangue_pct: float = 0.0
    magnetite_pct: float = 0.0

    # Si gangue friability
    si_friable_pct: float = 0.0
    si_non_friable_pct: float = 0.0

    # Zonation breakdown (normalised)
    zonation_un_pct: float = 0.0
    zonation_pr_pct: float = 0.0
    zonation_de_pct: float = 0.0
    zonation_hy_pct: float = 0.0
    zonation_hc_pct: float = 0.0
    weighted_zonation: Optional[float] = None

    # Specific minerals
    quartz_pct: float = 0.0
    chert_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame row."""
        return {
            "hole_id": self.hole_id,
            "depth_from": self.depth_from,
            "depth_to": self.depth_to,
            "weighted_hardness": self.weighted_hardness,
            "no_hardness_pct": self.no_hardness_pct,
            "total_logged_pct": self.total_logged_pct,
            "normalisation_factor": self.normalisation_factor,
            "total_gangue_pct": self.total_gangue_pct,
            "si_gangue_pct": self.si_gangue_pct,
            "al_gangue_pct": self.al_gangue_pct,
            "carbonate_gangue_pct": self.carbonate_gangue_pct,
            "sulphide_gangue_pct": self.sulphide_gangue_pct,
            "manganese_gangue_pct": self.manganese_gangue_pct,
            "mafics_gangue_pct": self.mafics_gangue_pct,
            "undefined_gangue_pct": self.undefined_gangue_pct,
            "magnesium_gangue_pct": self.magnesium_gangue_pct,
            "magnetite_pct": self.magnetite_pct,
            "si_friable_pct": self.si_friable_pct,
            "si_non_friable_pct": self.si_non_friable_pct,
            "zonation_un_pct": self.zonation_un_pct,
            "zonation_pr_pct": self.zonation_pr_pct,
            "zonation_de_pct": self.zonation_de_pct,
            "zonation_hy_pct": self.zonation_hy_pct,
            "zonation_hc_pct": self.zonation_hc_pct,
            "weighted_zonation": self.weighted_zonation,
            "quartz_pct": self.quartz_pct,
            "chert_pct": self.chert_pct,
        }


class RCMetricsCalculator:
    """
    Calculates RC metrics from mineral logging data.

    This implements the same calculation logic as the standalone
    RC hardness and logging calculations script.
    """

    # Gangue type IDs
    GANGUE_MINERALISED = 0
    GANGUE_SILICA = 1
    GANGUE_ALUMINIUM = 2
    GANGUE_CARBONATE = 3
    GANGUE_SULPHIDE = 4
    GANGUE_MANGANESE = 5
    GANGUE_MAFICS = 6
    GANGUE_UNDEFINED = 7
    GANGUE_MAGNESIUM = 8

    # Zonation numeric values
    ZONATION_VALUES = {
        "Un": 1,
        "Pr": 2,
        "De": 3,
        "Hy": 4,
        "Hc": 5,
    }

    def __init__(self, mineral_codes: MineralCodeManager):
        """
        Initialize the calculator.

        Args:
            mineral_codes: MineralCodeManager instance with loaded codes
        """
        self.mineral_codes = mineral_codes

        if not mineral_codes.is_loaded:
            raise ValueError("MineralCodeManager must have codes loaded")

    def calculate_interval_metrics(
        self,
        row: pd.Series,
        mineral_columns: List[str],
        hole_id_col: str = "holeid",
        from_col: str = "sampfrom",
        to_col: str = "sampto"
    ) -> IntervalMetrics:
        """
        Calculate metrics for a single interval.

        Args:
            row: DataFrame row with mineral logging data
            mineral_columns: List of Min_XX_pct column names
            hole_id_col: Column name for hole ID
            from_col: Column name for from depth
            to_col: Column name for to depth

        Returns:
            IntervalMetrics with calculated values
        """
        # Get interval identifiers
        hole_id = str(row.get(hole_id_col, ""))
        depth_from = float(row.get(from_col, 0))
        depth_to = float(row.get(to_col, 0))

        metrics = IntervalMetrics(
            hole_id=hole_id,
            depth_from=depth_from,
            depth_to=depth_to,
        )

        # Extract mineral codes and their percentages from column names
        minerals = []  # List of (code, percentage, column_name)
        for col in mineral_columns:
            value = row.get(col)
            if pd.isna(value) or str(value).upper() in ('', 'NULL', 'NAN', 'NONE'):
                continue

            # Extract percentage from column name (e.g., "Min_80_pct" -> 80%)
            try:
                # Column format: Min_XX_pct where XX is the percentage
                parts = col.split('_')
                if len(parts) >= 2:
                    pct_str = parts[1].replace('pct', '')
                    percentage = float(pct_str) / 100.0
                else:
                    continue
            except (ValueError, IndexError):
                logger.debug(f"Could not parse percentage from column {col}")
                continue

            code = str(value).strip().upper()
            if code:
                minerals.append((code, percentage, col))

        if not minerals:
            return metrics

        # Calculate total logged percentage
        total_pct = sum(pct for _, pct, _ in minerals)
        metrics.total_logged_pct = total_pct * 100

        # Calculate normalisation factor
        if total_pct > 0 and abs(total_pct - 1.0) > 0.001:
            metrics.normalisation_factor = 1.0 / total_pct
        else:
            metrics.normalisation_factor = 1.0

        # Accumulators
        hardness_weighted_sum = 0.0
        hardness_pct_sum = 0.0
        no_hardness_pct = 0.0

        gangue_by_type = {i: 0.0 for i in range(9)}
        si_friable = 0.0
        si_non_friable = 0.0

        zonation_pcts = {"Un": 0.0, "Pr": 0.0, "De": 0.0, "Hy": 0.0, "Hc": 0.0}
        zonation_weighted_sum = 0.0
        zonation_pct_sum = 0.0

        quartz_pct = 0.0
        chert_pct = 0.0
        magnetic_codes = self.mineral_codes.get_magnetic_magnetite_codes()
        magnetite_pct_sum = 0.0

        for code, raw_pct, col in minerals:
            normalised_pct = raw_pct * metrics.normalisation_factor

            # Get mineral info
            info = self.mineral_codes.get_code_info(code)
            if not info:
                logger.debug(f"Unknown mineral code: {code}")
                continue

            # Hardness calculation
            if info.hardness == 0:
                no_hardness_pct += normalised_pct
            else:
                hardness_weighted_sum += info.hardness * normalised_pct
                hardness_pct_sum += normalised_pct

            # Gangue calculation
            if info.gangue_type > 0:
                gangue_by_type[info.gangue_type] += normalised_pct

                # Si gangue friability split
                if info.gangue_type == self.GANGUE_SILICA:
                    if info.hardness < 3:
                        si_friable += normalised_pct
                    else:
                        si_non_friable += normalised_pct

            # Zonation calculation
            zonation_split = self.mineral_codes.get_zonation_split(info.profile_zonation)
            for zon, ratio in zonation_split.items():
                if zon in zonation_pcts:
                    zonation_pcts[zon] += normalised_pct * ratio

                    # Weighted zonation
                    zon_value = self.ZONATION_VALUES.get(zon, 1)
                    zonation_weighted_sum += zon_value * normalised_pct * ratio
                    zonation_pct_sum += normalised_pct * ratio

            # Special minerals (track raw percentages, not normalised)
            if self.mineral_codes.is_quartz_variant(code):
                quartz_pct += raw_pct
            if self.mineral_codes.is_chert_variant(code):
                chert_pct += raw_pct
            if code in magnetic_codes:
                magnetite_pct_sum += raw_pct * 100

        # Calculate final metrics
        if hardness_pct_sum > 0:
            metrics.weighted_hardness = round(hardness_weighted_sum / hardness_pct_sum, 2)
        metrics.no_hardness_pct = round(no_hardness_pct * 100, 2)

        # Gangue totals (as percentages)
        metrics.total_gangue_pct = round(sum(gangue_by_type.values()) * 100, 2)
        metrics.si_gangue_pct = round(gangue_by_type[self.GANGUE_SILICA] * 100, 2)
        metrics.al_gangue_pct = round(gangue_by_type[self.GANGUE_ALUMINIUM] * 100, 2)
        metrics.carbonate_gangue_pct = round(gangue_by_type[self.GANGUE_CARBONATE] * 100, 2)
        metrics.sulphide_gangue_pct = round(gangue_by_type[self.GANGUE_SULPHIDE] * 100, 2)
        metrics.manganese_gangue_pct = round(gangue_by_type[self.GANGUE_MANGANESE] * 100, 2)
        metrics.mafics_gangue_pct = round(gangue_by_type[self.GANGUE_MAFICS] * 100, 2)
        metrics.undefined_gangue_pct = round(gangue_by_type[self.GANGUE_UNDEFINED] * 100, 2)
        metrics.magnesium_gangue_pct = round(gangue_by_type[self.GANGUE_MAGNESIUM] * 100, 2)
        metrics.magnetite_pct = round(magnetite_pct_sum, 2)

        # Si gangue friability (as percentage of Si gangue)
        total_si = si_friable + si_non_friable
        if total_si > 0:
            metrics.si_friable_pct = round((si_friable / total_si) * 100, 2)
            metrics.si_non_friable_pct = round((si_non_friable / total_si) * 100, 2)

        # Zonation percentages
        metrics.zonation_un_pct = round(zonation_pcts["Un"] * 100, 2)
        metrics.zonation_pr_pct = round(zonation_pcts["Pr"] * 100, 2)
        metrics.zonation_de_pct = round(zonation_pcts["De"] * 100, 2)
        metrics.zonation_hy_pct = round(zonation_pcts["Hy"] * 100, 2)
        metrics.zonation_hc_pct = round(zonation_pcts["Hc"] * 100, 2)

        if zonation_pct_sum > 0:
            metrics.weighted_zonation = round(zonation_weighted_sum / zonation_pct_sum, 2)

        # Specific minerals (as percentages)
        metrics.quartz_pct = round(quartz_pct * 100, 2)
        metrics.chert_pct = round(chert_pct * 100, 2)

        return metrics


class RCMetricsStore:
    """
    Stores pre-computed RC metrics with O(1) lookup by (hole_id, depth_to).

    This integrates with the DataCoordinator to provide persistent
    access to calculated metrics.
    """

    def __init__(self, mineral_codes_path: Optional[Path] = None):
        """
        Initialize the RC metrics store.

        Args:
            mineral_codes_path: Path to mineral_codes.json
        """
        self._mineral_codes = MineralCodeManager(mineral_codes_path)
        self._calculator: Optional[RCMetricsCalculator] = None

        # Index: (hole_id_upper, depth_to_int) -> IntervalMetrics
        self._metrics_index: Dict[Tuple[str, int], IntervalMetrics] = {}

        # DataFrame view of all metrics
        self._metrics_df: Optional[pd.DataFrame] = None

        # State
        self._is_loaded = False
        self._compute_time: float = 0
        self._source_name: str = ""

    def initialize(self) -> bool:
        """
        Initialize by loading mineral codes.

        Returns:
            True if successful
        """
        if not self._mineral_codes.load():
            return False

        self._calculator = RCMetricsCalculator(self._mineral_codes)
        return True

    @property
    def is_loaded(self) -> bool:
        """Whether metrics have been computed and loaded."""
        return self._is_loaded

    @property
    def mineral_codes(self) -> MineralCodeManager:
        """Get the mineral code manager."""
        return self._mineral_codes

    @property
    def metrics_count(self) -> int:
        """Number of intervals with computed metrics."""
        return len(self._metrics_index)

    def compute_from_dataframe(
        self,
        df: pd.DataFrame,
        source_name: str = "unknown",
        hole_id_col: Optional[str] = None,
        from_col: Optional[str] = None,
        to_col: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> bool:
        """
        Compute metrics from a DataFrame of logging data.

        Args:
            df: DataFrame with mineral logging columns (Min_XX_pct)
            source_name: Name of the data source (for logging)
            hole_id_col: Column name for hole ID (auto-detected if None)
            from_col: Column name for from depth (auto-detected if None)
            to_col: Column name for to depth (auto-detected if None)
            progress_callback: Optional callback(message, percentage)

        Returns:
            True if successful
        """
        if self._calculator is None:
            if not self.initialize():
                return False

        start_time = time.time()

        # Find mineral columns
        mineral_columns = self._find_mineral_columns(df)
        if not mineral_columns:
            logger.warning(f"No mineral columns (Min_XX_pct) found in DataFrame")
            return False

        # Auto-detect column names if not provided
        hole_id_col, from_col, to_col = self._detect_key_columns(df, hole_id_col, from_col, to_col)
        if not hole_id_col or not to_col:
            logger.warning("Could not detect hole_id or depth_to columns")
            return False

        logger.info(f"Computing RC metrics for {len(df)} rows using {len(mineral_columns)} mineral columns")
        logger.info(f"  Key columns: hole_id={hole_id_col}, from={from_col}, to={to_col}")

        # Standardize column names to lowercase for lookup
        df_lower = df.copy()
        df_lower.columns = [c.lower() for c in df_lower.columns]
        hole_id_col_lower = hole_id_col.lower()
        from_col_lower = from_col.lower() if from_col else None
        to_col_lower = to_col.lower()
        mineral_columns_lower = [c.lower() for c in mineral_columns]

        # Clear existing metrics
        self._metrics_index.clear()

        # Process each row
        total_rows = len(df_lower)
        metrics_list = []

        for idx, (_, row) in enumerate(df_lower.iterrows()):
            if progress_callback and idx % 100 == 0:
                pct = (idx / total_rows) * 100
                progress_callback(f"Computing RC metrics... ({idx}/{total_rows})", pct)

            try:
                metrics = self._calculator.calculate_interval_metrics(
                    row,
                    mineral_columns_lower,
                    hole_id_col_lower,
                    from_col_lower,
                    to_col_lower
                )

                # Index by (hole_id_upper, depth_to_int)
                key = (metrics.hole_id.upper(), int(metrics.depth_to))
                self._metrics_index[key] = metrics
                metrics_list.append(metrics.to_dict())

            except Exception as e:
                logger.debug(f"Error computing metrics for row {idx}: {e}")
                continue

        # Build DataFrame view
        if metrics_list:
            self._metrics_df = pd.DataFrame(metrics_list)

            # Create MultiIndex for efficient lookups
            self._metrics_df['_hole_upper'] = self._metrics_df['hole_id'].str.upper()
            self._metrics_df['_depth_int'] = self._metrics_df['depth_to'].astype(int)
            self._metrics_df.set_index(['_hole_upper', '_depth_int'], inplace=True, drop=False)

        self._is_loaded = True
        self._compute_time = time.time() - start_time
        self._source_name = source_name

        if progress_callback:
            progress_callback("RC metrics computed", 100)

        logger.info(
            f"Computed {len(self._metrics_index)} interval metrics "
            f"in {self._compute_time:.2f}s from source '{source_name}'"
        )

        return True

    def _find_mineral_columns(self, df: pd.DataFrame) -> List[str]:
        """Find mineral percentage columns (Min_XX_pct pattern)."""
        mineral_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if col_lower.startswith('min_') and col_lower.endswith('_pct'):
                # Extract the percentage value to verify it's numeric
                try:
                    parts = col_lower.split('_')
                    if len(parts) >= 2:
                        pct_str = parts[1].replace('pct', '')
                        float(pct_str)  # Validate it's numeric
                        mineral_cols.append(col)
                except (ValueError, IndexError):
                    continue

        return sorted(mineral_cols, key=lambda c: int(c.lower().split('_')[1].replace('pct', '')), reverse=True)

    def _detect_key_columns(
        self,
        df: pd.DataFrame,
        hole_id_col: Optional[str],
        from_col: Optional[str],
        to_col: Optional[str]
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Auto-detect hole_id, from, and to column names.

        Args:
            df: DataFrame to inspect
            hole_id_col: Explicit hole_id column or None to detect
            from_col: Explicit from column or None to detect
            to_col: Explicit to column or None to detect

        Returns:
            Tuple of (hole_id_col, from_col, to_col)
        """
        cols_lower = {c.lower(): c for c in df.columns}

        # Hole ID detection
        if hole_id_col is None:
            hole_patterns = ['holeid', 'hole_id', 'bhid', 'drillhole_id']
            for pattern in hole_patterns:
                if pattern in cols_lower:
                    hole_id_col = cols_lower[pattern]
                    break

        # From depth detection
        if from_col is None:
            from_patterns = ['sampfrom', 'geolfrom', 'from', 'depth_from', 'from_depth', 'from_m']
            for pattern in from_patterns:
                if pattern in cols_lower:
                    from_col = cols_lower[pattern]
                    break

        # To depth detection
        if to_col is None:
            to_patterns = ['sampto', 'geolto', 'to', 'depth_to', 'to_depth', 'to_m']
            for pattern in to_patterns:
                if pattern in cols_lower:
                    to_col = cols_lower[pattern]
                    break

        return hole_id_col, from_col, to_col

    def get_metrics(self, hole_id: str, depth_to: float) -> Optional[IntervalMetrics]:
        """
        Get computed metrics for an interval (O(1) lookup).

        Args:
            hole_id: Hole identifier
            depth_to: End depth

        Returns:
            IntervalMetrics or None if not found
        """
        key = (hole_id.upper(), int(depth_to))
        return self._metrics_index.get(key)

    def get_metrics_dict(self, hole_id: str, depth_to: float) -> Dict[str, Any]:
        """
        Get computed metrics as a dictionary.

        Args:
            hole_id: Hole identifier
            depth_to: End depth

        Returns:
            Dictionary of metrics, empty dict if not found
        """
        metrics = self.get_metrics(hole_id, depth_to)
        return metrics.to_dict() if metrics else {}

    def get_metrics_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Get all computed metrics as a DataFrame.

        Returns:
            DataFrame with all metrics, or None if not computed
        """
        return self._metrics_df.copy() if self._metrics_df is not None else None

    def get_metrics_for_hole(self, hole_id: str) -> List[IntervalMetrics]:
        """
        Get all metrics for a specific hole.

        Args:
            hole_id: Hole identifier

        Returns:
            List of IntervalMetrics sorted by depth
        """
        hole_upper = hole_id.upper()
        metrics = [
            m for (h, _), m in self._metrics_index.items()
            if h == hole_upper
        ]
        return sorted(metrics, key=lambda m: m.depth_to)

    def clear(self) -> None:
        """Clear all computed metrics."""
        self._metrics_index.clear()
        self._metrics_df = None
        self._is_loaded = False
        self._compute_time = 0
        self._source_name = ""
        logger.debug("RC metrics store cleared")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics about computed metrics.

        Returns:
            Dictionary with stats
        """
        if not self._is_loaded or self._metrics_df is None:
            return {}

        df = self._metrics_df
        return {
            "total_intervals": len(df),
            "unique_holes": df['hole_id'].nunique(),
            "compute_time_seconds": self._compute_time,
            "source_name": self._source_name,
            "avg_weighted_hardness": df['weighted_hardness'].mean() if 'weighted_hardness' in df else None,
            "avg_total_gangue_pct": df['total_gangue_pct'].mean() if 'total_gangue_pct' in df else None,
            "intervals_with_quartz": (df['quartz_pct'] > 0).sum() if 'quartz_pct' in df else 0,
            "intervals_with_chert": (df['chert_pct'] > 0).sum() if 'chert_pct' in df else 0,
        }
