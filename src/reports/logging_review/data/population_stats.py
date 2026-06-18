"""Population statistics for dynamic threshold-based flagging.

Pre-computes percentile statistics from the full dataset ONCE at report build
time so that flag functions use data-driven thresholds rather than hardcoded
numbers.  The full merged_df (all loggers, all holes) is used so thresholds
represent the true dataset population, not just one logger.

Usage in report_builder.py:
    pop_stats = PopulationStats(merged_df)
    issue, significance = _flag_magnesium_issue(row, resolver, pop_stats)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from processing.DataManager.column_aliases import ColumnResolver

logger = logging.getLogger(__name__)

# Elements to compute stats for — covers all gangue flag checks plus
# mafic indicators and trace elements used in logging detail accuracy.
POPULATION_ELEMENTS: List[Tuple[str, str]] = [
    # (standard_name, display_name)
    ("fe_pct",       "Fe%"),
    ("sio2_pct",     "SiO2%"),
    ("al2o3_pct",    "Al2O3%"),
    ("p_pct",        "P%"),
    ("s_pct",        "S%"),
    ("loi_pct",      "LOI%"),
    ("loi_1000_pct", "LOI1000%"),
    ("loi_425_pct",  "LOI425%"),
    ("mn_pct",       "Mn%"),
    ("cao_pct",      "CaO%"),
    ("mgo_pct",      "MgO%"),
    ("k2o_pct",      "K2O%"),
    ("na2o_pct",     "Na2O%"),
    ("tio2_pct",     "TiO2%"),
    ("cr2o3_pct",    "Cr2O3%"),
    ("nio_pct",      "NiO%"),
]


@dataclass
class ElementStats:
    """Pre-computed population statistics for one element."""

    column: str           # Actual column name in the DataFrame
    standard_name: str    # e.g. "mgo_pct"
    display_name: str     # e.g. "MgO%"
    count: int
    p10: float
    p25: float
    p50: float            # median
    p75: float
    p90: float
    p95: float
    mean: float
    std: float


class PopulationStats:
    """
    Pre-computed percentile statistics for all available assay elements.

    Built once from the full merged_df (all loggers, all holes) so that
    thresholds represent the true dataset population.

    Elevation / depletion checks:
        is_elevated("mgo_pct", 3.2)  ->  "High" (> P90) / "Low" (> P75) / None
        is_depleted("loi_pct", 0.5)  ->  "High" (< P10) / "Low" (< P25) / None

    The severity string doubles as the significance value for the evidence
    table, eliminating the old keyword-matching significance function.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        resolver: Optional[ColumnResolver] = None,
    ):
        if resolver is None:
            resolver = ColumnResolver(df)
        self._stats: Dict[str, ElementStats] = {}
        self._row_count = len(df)
        self._build(df, resolver)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self, df: pd.DataFrame, resolver: ColumnResolver) -> None:
        """Compute percentiles for all available elements."""
        for std_name, display_name in POPULATION_ELEMENTS:
            col = resolver.get(std_name)
            if not col or col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) < 20:
                # Need a reasonable sample for meaningful percentiles
                logger.debug(
                    "PopulationStats: skipping %s (%s) — only %d non-null values",
                    std_name, col, len(series),
                )
                continue
            self._stats[std_name] = ElementStats(
                column=col,
                standard_name=std_name,
                display_name=display_name,
                count=len(series),
                p10=float(series.quantile(0.10)),
                p25=float(series.quantile(0.25)),
                p50=float(series.quantile(0.50)),
                p75=float(series.quantile(0.75)),
                p90=float(series.quantile(0.90)),
                p95=float(series.quantile(0.95)),
                mean=float(series.mean()),
                std=float(series.std()),
            )
        logger.info(
            "PopulationStats built: %d elements from %d rows",
            len(self._stats),
            self._row_count,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, std_name: str) -> Optional[ElementStats]:
        """Get stats for a standard element name, or None if unavailable."""
        return self._stats.get(std_name)

    def is_elevated(self, std_name: str, value: float) -> Optional[str]:
        """
        Check if a value is elevated relative to the population.

        Returns:
            "High" if value > P90
            "Low"  if value > P75 (but <= P90)
            None   if not elevated (or element not available)
        """
        s = self.get(std_name)
        if s is None:
            return None
        if value > s.p90:
            return "High"
        if value > s.p75:
            return "Low"
        return None

    def is_depleted(self, std_name: str, value: float) -> Optional[str]:
        """
        Check if a value is depleted relative to the population.

        Returns:
            "High" if value < P10
            "Low"  if value < P25 (but >= P10)
            None   if not depleted (or element not available)
        """
        s = self.get(std_name)
        if s is None:
            return None
        if value < s.p10:
            return "High"
        if value < s.p25:
            return "Low"
        return None

    def threshold_str(self, std_name: str, level: str = "High", direction: str = "elevated") -> str:
        """
        Human-readable threshold for display in evidence tables / rules text.

        Examples:
            threshold_str("mgo_pct", "High")  ->  "P90=1.83%"
            threshold_str("s_pct", "Low")      ->  "P75=0.08%"
            threshold_str("loi_pct", "High", direction="depleted")  ->  "P10=0.42%"
        """
        s = self.get(std_name)
        if s is None:
            return "n/a"
        if direction == "depleted":
            if level == "High":
                return f"P10={s.p10:.2f}%"
            return f"P25={s.p25:.2f}%"
        if level == "High":
            return f"P90={s.p90:.2f}%"
        return f"P75={s.p75:.2f}%"

    @property
    def row_count(self) -> int:
        """Number of rows used to build these stats."""
        return self._row_count

    @property
    def element_count(self) -> int:
        """Number of elements with stats computed."""
        return len(self._stats)

    def summary_dict(self) -> Dict[str, Dict[str, float]]:
        """
        Return all stats as a serialisable dict for report metadata.

        Included in report_data so the HTML report can display the actual
        thresholds used (making the report self-documenting).
        """
        out = {}
        for std_name, s in self._stats.items():
            out[std_name] = {
                "column": s.column,
                "display": s.display_name,
                "n": s.count,
                "P10": round(s.p10, 3),
                "P25": round(s.p25, 3),
                "P50": round(s.p50, 3),
                "P75": round(s.p75, 3),
                "P90": round(s.p90, 3),
                "P95": round(s.p95, 3),
                "mean": round(s.mean, 3),
                "std": round(s.std, 3),
            }
        return out
