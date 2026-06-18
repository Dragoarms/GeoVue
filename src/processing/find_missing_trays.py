"""
find_missing_trays.py — Missing Tray & Compartment Analyzer for GeoVue.

Identifies missing chip tray intervals and missing compartment images
by comparing expected data (from Snowflake COLLAR + GeologicalStore intervals)
against actual images (from ImageIndex).

No file dialogs or folder scanning needed — all data is already loaded.

Usage (from main_gui.py):
    from processing.find_missing_trays import MissingTrayAnalyzer, MissingTrayStatus, export_missing_trays_pdf
    
    analyzer = MissingTrayAnalyzer(data_coordinator)
    results = analyzer.run()
    
    # Optional: filter by acknowledgement status
    status = MissingTrayStatus(json_register_manager)
    filtered = status.filter_results(results, mode="unresolved")
    export_missing_trays_pdf(filtered, output_path)

Author: George Symonds / Claude
"""

import logging
import math
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.json_register_manager import JSONRegisterManager

logger = logging.getLogger(__name__)


# ── Result dataclasses ────────────────────────────────────────────────

@dataclass
class MissingTray:
    """A tray interval that should exist but has no original image."""
    hole_id: str
    depth_from: int
    depth_to: int
    max_depth: float
    drilling_method: str = ""
    project_code: str = ""
    start_date: str = ""


@dataclass
class MissingCompartment:
    """A compartment that should exist within a tray but has no image."""
    hole_id: str
    tray_from: int
    tray_to: int
    missing_depth_to: int


@dataclass
class TrayWithMissing:
    """A tray that exists but is missing some compartment images."""
    hole_id: str
    tray_from: int
    tray_to: int
    expected_count: int
    found_count: int
    missing_depths: List[int] = field(default_factory=list)
    project_code: str = ""
    start_date: str = ""

    @property
    def missing_count(self) -> int:
        return len(self.missing_depths)

    @property
    def missing_pct(self) -> float:
        return (self.missing_count / self.expected_count * 100) if self.expected_count > 0 else 0.0


@dataclass
class AnalysisResults:
    """Complete results from the missing tray/compartment analysis."""
    # Input stats
    total_holes_analysed: int = 0
    rc_holes: int = 0
    diamond_holes: int = 0

    # Missing trays
    total_expected_trays: int = 0
    total_existing_trays: int = 0
    missing_trays: List[MissingTray] = field(default_factory=list)

    # Missing compartments
    total_trays_checked: int = 0
    total_expected_compartments: int = 0
    total_existing_compartments: int = 0
    trays_with_missing: List[TrayWithMissing] = field(default_factory=list)

    # Holes with no images at all
    holes_with_no_images: List[str] = field(default_factory=list)

    # Trays that exist (original image) but have zero extracted compartments
    unprocessed_trays: List[MissingTray] = field(default_factory=list)

    @property
    def total_missing_trays(self) -> int:
        return len(self.missing_trays)

    @property
    def total_missing_compartments(self) -> int:
        return sum(t.missing_count for t in self.trays_with_missing)

    @property
    def tray_coverage_pct(self) -> float:
        if self.total_expected_trays == 0:
            return 0.0
        return (self.total_existing_trays / self.total_expected_trays) * 100

    @property
    def compartment_coverage_pct(self) -> float:
        if self.total_expected_compartments == 0:
            return 0.0
        return (self.total_existing_compartments / self.total_expected_compartments) * 100


# ── Status tracking ──────────────────────────────────────────────────

class MissingTrayStatus:
    """
    Tracks acknowledged/resolved missing tray entries via a shared JSON file
    managed by JSONRegisterManager.
    
    Status file lives in Register Data alongside other shared JSON files.
    File locking and backup handled by JSONRegisterManager.
    
    Valid statuses: acknowledged, not_recoverable, re_photograph, different_interval
    """

    VALID_STATUSES = {
        "lost",
    }

    def __init__(self, register_manager: 'JSONRegisterManager'):
        """
        Args:
            register_manager: Initialized JSONRegisterManager instance.
        """
        self._rm = register_manager
        self._entries: Dict[str, dict] = {}
        self._load()

    def _load(self):
        """Load status entries via register manager."""
        self._entries = self._rm.read_missing_tray_status()
        logger.info(f"Loaded {len(self._entries)} missing tray status entries")

    def _save(self) -> bool:
        """Save status entries via register manager."""
        return self._rm.write_missing_tray_status(self._entries)

    def reload(self):
        """Reload from disk (e.g. after another user updates the file)."""
        self._load()

    @staticmethod
    def make_key(hole_id: str, depth_from: int, depth_to: int) -> str:
        """Generate a status key from tray identifiers."""
        return f"{hole_id.upper()}_{depth_from}-{depth_to}"

    def get_status(self, hole_id: str, depth_from: int, depth_to: int) -> Optional[dict]:
        """Get status entry for a tray interval, or None if unresolved."""
        return self._entries.get(self.make_key(hole_id, depth_from, depth_to))

    def is_resolved(self, hole_id: str, depth_from: int, depth_to: int) -> bool:
        """Check if a tray interval has been acknowledged/resolved."""
        return self.make_key(hole_id, depth_from, depth_to) in self._entries

    def set_status(
        self,
        hole_id: str,
        depth_from: int,
        depth_to: int,
        status: str,
        reason: str = "",
        user: str = "",
    ) -> bool:
        """
        Set status for a tray interval.
        
        Args:
            hole_id: Hole identifier
            depth_from: Tray start depth
            depth_to: Tray end depth
            status: One of VALID_STATUSES
            reason: Free-text reason
            user: Username who acknowledged
            
        Returns:
            True if saved successfully
        """
        if status not in self.VALID_STATUSES:
            logger.warning(f"Invalid status '{status}', must be one of {self.VALID_STATUSES}")
            return False

        key = self.make_key(hole_id, depth_from, depth_to)
        self._entries[key] = {
            "status": status,
            "reason": reason,
            "by": user,
            "date": datetime.now().strftime("%Y-%m-%d"),
        }
        return self._save()

    def set_status_batch(
        self,
        items: List[dict],
    ) -> bool:
        """
        Set status for multiple tray intervals in one write.
        
        Args:
            items: List of dicts with keys: hole_id, depth_from, depth_to, status, reason, user
            
        Returns:
            True if saved successfully
        """
        for item in items:
            status = item.get("status", "")
            if status not in self.VALID_STATUSES:
                logger.warning(f"Skipping invalid status '{status}' for {item}")
                continue
            key = self.make_key(
                item["hole_id"], item["depth_from"], item["depth_to"]
            )
            self._entries[key] = {
                "status": status,
                "reason": item.get("reason", ""),
                "by": item.get("user", ""),
                "date": datetime.now().strftime("%Y-%m-%d"),
            }
        return self._save()

    def remove_status(self, hole_id: str, depth_from: int, depth_to: int) -> bool:
        """Remove a status entry (un-resolve a tray)."""
        key = self.make_key(hole_id, depth_from, depth_to)
        if key in self._entries:
            del self._entries[key]
            return self._save()
        return False

    def get_all_entries(self) -> Dict[str, dict]:
        """Get all status entries (copy)."""
        return self._entries.copy()

    @property
    def resolved_count(self) -> int:
        return len(self._entries)

    def filter_results(self, results: 'AnalysisResults', mode: str = "unresolved") -> 'AnalysisResults':
        """
        Filter analysis results based on resolution status.
        
        Args:
            results: Raw AnalysisResults from analyzer
            mode: "all" | "unresolved" | "resolved"
            
        Returns:
            New AnalysisResults with filtered lists (counts remain from original)
        """
        if mode == "all" or not self._entries:
            return results

        from copy import deepcopy
        filtered = deepcopy(results)

        if mode == "unresolved":
            filtered.missing_trays = [
                t for t in filtered.missing_trays
                if not self.is_resolved(t.hole_id, t.depth_from, t.depth_to)
            ]
            filtered.trays_with_missing = [
                t for t in filtered.trays_with_missing
                if not self.is_resolved(t.hole_id, t.tray_from, t.tray_to)
            ]
            filtered.unprocessed_trays = [
                t for t in filtered.unprocessed_trays
                if not self.is_resolved(t.hole_id, t.depth_from, t.depth_to)
            ]
        elif mode == "resolved":
            filtered.missing_trays = [
                t for t in filtered.missing_trays
                if self.is_resolved(t.hole_id, t.depth_from, t.depth_to)
            ]
            filtered.trays_with_missing = [
                t for t in filtered.trays_with_missing
                if self.is_resolved(t.hole_id, t.tray_from, t.tray_to)
            ]
            filtered.unprocessed_trays = [
                t for t in filtered.unprocessed_trays
                if self.is_resolved(t.hole_id, t.depth_from, t.depth_to)
            ]

        return filtered


# ── Analyzer ──────────────────────────────────────────────────────────

class MissingTrayAnalyzer:
    """
    Compares expected tray intervals and compartments against what
    ImageIndex actually has indexed. Uses Snowflake COLLAR for hole
    depths and GeologicalStore for expected interval data.
    """

    TRAY_INTERVAL = 20  # metres per tray

    def __init__(self, data_coordinator):
        """
        Args:
            data_coordinator: Initialized DataCoordinator with image_index,
                              geological_store, and optionally collar data.
        """
        self._dc = data_coordinator
        self._image_index = data_coordinator.image_index
        self._geo_store = data_coordinator.geological_store

    def run(
        self,
        collar_depths: Optional[Dict[str, Tuple[float, float]]] = None,
        hole_filter: Optional[Set[str]] = None,
        rc_only: bool = True,
    ) -> AnalysisResults:
        """
        Run the full analysis.

        Args:
            collar_depths: {HOLE_ID: (min_depth, max_depth)} from Snowflake.
                          If None, derives from GeologicalStore.
            hole_filter: Optional set of HOLEIDs to analyse (None = all).
            rc_only: If True (default), only analyse RC holes.

        Returns:
            AnalysisResults with all findings.
        """
        results = AnalysisResults()

        # 1. Build collar metadata (drilling method, project code, start date)
        collar_meta = self._build_collar_metadata()
        logger.info(f"Collar metadata loaded for {len(collar_meta)} holes")

        # 2. Build hole depth map
        hole_depths = self._get_hole_depths(collar_depths)
        if hole_filter:
            hole_depths = {h: d for h, d in hole_depths.items() if h in hole_filter}

        # 3. Filter to RC only if requested
        if rc_only:
            import re
            # Standard RC hole ID: 2 uppercase letters + 4 digits (e.g., BA0001, KM0137)
            rc_pattern = re.compile(r'^[A-Z]{2}\d{4}$')

            before = len(hole_depths)
            filtered = {}
            for h, d in hole_depths.items():
                method = collar_meta.get(h, {}).get("method", "").upper()
                if method == "RC":
                    filtered[h] = d
                elif method in ("DIAMOND", "CHANNEL", "DD", "ADIT"):
                    continue  # Explicitly non-RC
                elif method == "":
                    # No collar metadata — infer from hole ID pattern
                    if rc_pattern.match(h):
                        filtered[h] = d
                    else:
                        logger.debug(f"Excluding '{h}' — no drilling method and non-standard hole ID")
                else:
                    continue  # Unknown method, skip

            hole_depths = filtered
            filtered_out = before - len(hole_depths)
            if filtered_out:
                logger.info(f"Filtered to RC only: {len(hole_depths)} holes ({filtered_out} non-RC excluded)")

        results.total_holes_analysed = len(hole_depths)
        logger.info(f"Analysing {len(hole_depths)} holes for missing trays/compartments")

        if not hole_depths:
            logger.warning("No hole depths available — cannot analyse")
            return results

        # 4. Build existing tray index from ImageIndex._original_index
        existing_trays = self._build_existing_tray_set()
        logger.info(f"Found existing trays for {len(existing_trays)} holes")

        # 5. Build existing compartment index from ImageIndex
        existing_compartments = self._build_existing_compartment_set()
        logger.info(f"Found existing compartments for {len(existing_compartments)} holes")

        # 6. Get expected intervals from GeologicalStore
        expected_intervals = self._get_expected_intervals()
        logger.info(f"Found expected intervals for {len(expected_intervals)} holes")

        # 7. Identify missing trays
        for hole_id, max_depth in sorted(hole_depths.items()):
            meta = collar_meta.get(hole_id, {})
            method = meta.get("method", "")
            project = meta.get("project", "")
            start = meta.get("start_date", "")

            if method.upper() == "RC":
                results.rc_holes += 1
            elif method.upper() == "DIAMOND":
                results.diamond_holes += 1

            hole_trays = existing_trays.get(hole_id, set())

            # Check if hole has any images at all
            if not hole_trays and hole_id not in existing_compartments:
                results.holes_with_no_images.append(hole_id)

            # Generate expected tray intervals
            depth = 0
            while depth < max_depth:
                from_depth = depth
                to_depth = min(depth + self.TRAY_INTERVAL, int(max_depth))
                results.total_expected_trays += 1

                # Check overlap with any existing tray
                has_coverage = any(
                    from_depth < act_to and act_from < to_depth
                    for act_from, act_to in hole_trays
                )

                if has_coverage:
                    results.total_existing_trays += 1
                else:
                    results.missing_trays.append(MissingTray(
                        hole_id=hole_id,
                        depth_from=from_depth,
                        depth_to=to_depth,
                        max_depth=max_depth,
                        drilling_method=method,
                        project_code=project,
                        start_date=start,
                    ))

                depth += self.TRAY_INTERVAL

        # 8. Identify missing compartments within existing trays
        for hole_id, tray_intervals in existing_trays.items():
            hole_compartments = existing_compartments.get(hole_id, set())
            hole_expected = expected_intervals.get(hole_id, set())

            for tray_from, tray_to in tray_intervals:
                results.total_trays_checked += 1

                # Expected compartments in this tray range
                expected_in_tray = {
                    d for d in hole_expected
                    if tray_from < d <= tray_to
                }

                # If no expected interval data, generate from tray range (1m intervals)
                if not expected_in_tray and tray_to > tray_from:
                    expected_in_tray = set(range(int(tray_from) + 1, int(tray_to) + 1))

                results.total_expected_compartments += len(expected_in_tray)

                found = expected_in_tray & hole_compartments
                missing = expected_in_tray - hole_compartments
                results.total_existing_compartments += len(found)

                if missing:
                    meta = collar_meta.get(hole_id, {})
                    results.trays_with_missing.append(TrayWithMissing(
                        hole_id=hole_id,
                        tray_from=int(tray_from),
                        tray_to=int(tray_to),
                        expected_count=len(expected_in_tray),
                        found_count=len(found),
                        missing_depths=sorted(missing),
                        project_code=meta.get("project", ""),
                        start_date=meta.get("start_date", ""),
                    ))

        # 9. Identify unprocessed trays (original exists, zero compartments extracted)
        for hole_id, tray_intervals in existing_trays.items():
            hole_compartments = existing_compartments.get(hole_id, set())

            for tray_from, tray_to in tray_intervals:
                # Check if ANY compartment exists in this tray range
                has_compartment = any(
                    tray_from < d <= tray_to
                    for d in hole_compartments
                )
                if not has_compartment:
                    meta = collar_meta.get(hole_id, {})
                    results.unprocessed_trays.append(MissingTray(
                        hole_id=hole_id,
                        depth_from=int(tray_from),
                        depth_to=int(tray_to),
                        max_depth=hole_depths.get(hole_id, 0),
                        drilling_method=meta.get("method", ""),
                        project_code=meta.get("project", ""),
                        start_date=meta.get("start_date", ""),
                    ))

        results.unprocessed_trays.sort(
            key=lambda t: (t.project_code, t.start_date, t.hole_id, t.depth_from)
        )

        # Sort by project → start date → hole → depth
        results.missing_trays.sort(
            key=lambda t: (t.project_code, t.start_date, t.hole_id, t.depth_from)
        )
        results.trays_with_missing.sort(
            key=lambda t: (t.project_code, t.start_date, t.hole_id, t.tray_from)
        )

        # Log summary
        logger.info("=" * 60)
        logger.info("MISSING TRAYS/COMPARTMENTS ANALYSIS COMPLETE")
        logger.info(f"  Holes analysed: {results.total_holes_analysed}")
        logger.info(f"  RC holes: {results.rc_holes}")
        logger.info(f"  Expected trays: {results.total_expected_trays}")
        logger.info(f"  Missing trays: {results.total_missing_trays}")
        logger.info(f"  Tray coverage: {results.tray_coverage_pct:.1f}%")
        logger.info(f"  Expected compartments: {results.total_expected_compartments}")
        logger.info(f"  Missing compartments: {results.total_missing_compartments}")
        logger.info(f"  Compartment coverage: {results.compartment_coverage_pct:.1f}%")
        logger.info(f"  Unprocessed trays (original exists, no compartments): {len(results.unprocessed_trays)}")
        logger.info(f"  Holes with no images: {len(results.holes_with_no_images)}")
        logger.info("=" * 60)

        return results

    # ── Internal data extraction ──────────────────────────────────────

    def _build_collar_metadata(self) -> Dict[str, dict]:
        """
        Build {HOLE_ID: {method, project, start_date}} from GeologicalStore.
        Checks all sources for collar-like columns. Snowflake sources override CSV
        because they load after CSV and overwrite the same keys.
        """
        meta: Dict[str, dict] = {}

        if not self._geo_store or not self._geo_store.is_loaded:
            return meta

        for source_name in self._geo_store.list_sources():
            source = self._geo_store.get_source(source_name)
            if source is None or source.df is None:
                continue

            cols_lower = {c.lower(): c for c in source.df.columns}

            hole_col = next((cols_lower[p] for p in ['holeid', 'hole_id'] if p in cols_lower), None)
            method_col = next((cols_lower[p] for p in ['drillingmethod', 'drilling_method'] if p in cols_lower), None)

            if not (hole_col and method_col):
                continue

            # This source has collar data
            project_col = next((cols_lower[p] for p in ['projectcode', 'project_code', 'project'] if p in cols_lower), None)
            date_col = next((cols_lower[p] for p in ['startdate', 'start_date', 'drilldate'] if p in cols_lower), None)

            logger.info(f"Loading collar metadata from '{source_name}' ({len(source.df)} rows)")

            # Build a working subset with normalized hole IDs
            work_cols = [hole_col, method_col]
            if project_col:
                work_cols.append(project_col)
            if date_col:
                work_cols.append(date_col)

            subset = source.df[work_cols].copy()
            subset['_hid'] = subset[hole_col].astype(str).str.strip().str.upper()
            subset = subset[subset['_hid'].ne('') & subset['_hid'].ne('NAN')]

            # De-duplicate: keep last occurrence per hole (Snowflake overrides CSV)
            subset = subset.drop_duplicates(subset='_hid', keep='last')

            for _, row in subset.iterrows():
                hid = row['_hid']
                entry = meta.get(hid, {})

                method_val = str(row[method_col]).strip() if pd.notna(row[method_col]) else ""
                if method_val:
                    entry["method"] = method_val

                if project_col and pd.notna(row.get(project_col)):
                    entry["project"] = str(row[project_col]).strip()

                if date_col and pd.notna(row.get(date_col)):
                    raw = row[date_col]
                    if hasattr(raw, 'strftime'):
                        entry["start_date"] = raw.strftime("%Y-%m-%d")
                    else:
                        entry["start_date"] = str(raw).strip()

                meta[hid] = entry

        logger.info(f"Collar metadata: {len(meta)} holes, "
                     f"RC={sum(1 for m in meta.values() if m.get('method','').upper()=='RC')}, "
                     f"Diamond={sum(1 for m in meta.values() if m.get('method','').upper()=='DIAMOND')}")

        return meta

    def _get_hole_depths(
        self, collar_depths: Optional[Dict[str, Tuple[float, float]]]
    ) -> Dict[str, float]:
        """Get max depth per hole. Prefers Snowflake collar, falls back to GeologicalStore."""
        hole_depths: Dict[str, float] = {}

        # Snowflake collar data (already rounded to 20m)
        if collar_depths:
            for hole_id, (_, max_d) in collar_depths.items():
                if max_d > 0:
                    hole_depths[hole_id.upper()] = max_d
            logger.info(f"Got depths for {len(hole_depths)} holes from Snowflake COLLAR")

        # Fall back to GeologicalStore if no collar data
        if not hole_depths and self._geo_store and self._geo_store.is_loaded:
            for source_name in self._geo_store.list_sources():
                source = self._geo_store.get_source(source_name)
                if source is None or source.df is None:
                    continue

                # Find hole_id and to columns
                cols_lower = {c.lower(): c for c in source.df.columns}
                hole_col = next((cols_lower[p] for p in ['holeid', 'hole_id'] if p in cols_lower), None)
                to_col = next((cols_lower[p] for p in ['sampto', 'geolto', 'to', 'depth_to'] if p in cols_lower), None)

                if hole_col and to_col:
                    grouped = source.df.groupby(hole_col)[to_col].max()
                    for hole_id, max_d in grouped.items():
                        hid = str(hole_id).upper()
                        d = float(max_d)
                        rounded = math.ceil(d / 20) * 20
                        if hid not in hole_depths or rounded > hole_depths[hid]:
                            hole_depths[hid] = rounded

            logger.info(f"Derived depths for {len(hole_depths)} holes from GeologicalStore")

        return hole_depths

    def _build_existing_tray_set(self) -> Dict[str, Set[Tuple[int, int]]]:
        """Build {hole_id: set((from, to))} from ImageIndex original images."""
        trays: Dict[str, Set[Tuple[int, int]]] = {}
        for (hole_id, from_d, to_d) in self._image_index._original_index.keys():
            hid = hole_id.upper()
            if hid not in trays:
                trays[hid] = set()
            trays[hid].add((int(from_d), int(to_d)))
        return trays

    def _build_existing_compartment_set(self) -> Dict[str, Set[int]]:
        """Build {hole_id: set(depth_to)} from ImageIndex compartment images."""
        compartments: Dict[str, Set[int]] = {}
        for key_tuple in self._image_index._compartment_index.keys():
            # key_tuple is (hole_id_upper, depth_to_int, moisture_status)
            hole_id = str(key_tuple[0]).upper()
            depth_to = int(key_tuple[1])
            if hole_id not in compartments:
                compartments[hole_id] = set()
            compartments[hole_id].add(depth_to)
        return compartments

    def _get_expected_intervals(self) -> Dict[str, Set[int]]:
        """
        Get expected compartment depth_to values from GeologicalStore.
        Returns {hole_id: set(depth_to_int)} for all interval sources.
        
        Uses vectorized pandas operations instead of row iteration for performance.
        """
        intervals: Dict[str, Set[int]] = {}

        if not self._geo_store or not self._geo_store.is_loaded:
            return intervals

        for source_name in self._geo_store.list_sources():
            source = self._geo_store.get_source(source_name)
            if source is None or source.df is None:
                continue

            # Only use interval-type sources (have FROM and TO columns)
            cols_lower = {c.lower(): c for c in source.df.columns}
            hole_col = next((cols_lower[p] for p in ['holeid', 'hole_id'] if p in cols_lower), None)
            to_col = next((cols_lower[p] for p in ['sampto', 'geolto', 'to', 'depth_to'] if p in cols_lower), None)
            from_col = next((cols_lower[p] for p in ['sampfrom', 'geolfrom', 'from', 'depth_from'] if p in cols_lower), None)

            if not (hole_col and to_col and from_col):
                continue

            # Check it's 1m intervals (RC data)
            df = source.df
            sample_interval = (df[to_col] - df[from_col]).mode()
            if len(sample_interval) == 0 or abs(sample_interval.iloc[0] - 1.0) >= 0.1:
                continue

            # Vectorized extraction — no row iteration
            subset = df[[hole_col, to_col]].copy()
            subset[hole_col] = subset[hole_col].astype(str).str.strip().str.upper()
            subset[to_col] = pd.to_numeric(subset[to_col], errors='coerce')
            subset = subset.dropna(subset=[to_col])
            subset[to_col] = subset[to_col].astype(int)

            # Filter out invalid hole IDs
            subset = subset[subset[hole_col].ne('') & subset[hole_col].ne('NAN')]

            # Group by hole and collect sets
            for hid, group in subset.groupby(hole_col)[to_col]:
                if hid in intervals:
                    intervals[hid].update(group.values)
                else:
                    intervals[hid] = set(group.values)

        logger.info(f"Found expected 1m intervals for {len(intervals)} holes from GeologicalStore")
        return intervals


# ── PDF Export ────────────────────────────────────────────────────────

def export_missing_trays_pdf(results: AnalysisResults, output_path: str) -> bool:
    """
    Export analysis results to a PDF report.

    Args:
        results: AnalysisResults from MissingTrayAnalyzer.run()
        output_path: Path for the output PDF file

    Returns:
        True if export successful
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.platypus import (
            SimpleDocTemplate, Table, TableStyle, Paragraph,
            Spacer, PageBreak,
        )
    except ImportError:
        logger.error("ReportLab not installed — cannot export PDF")
        return False

    logger.info(f"Exporting missing trays report to: {output_path}")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Heading1'],
        fontSize=16, textColor=colors.HexColor('#2c3e50'),
        spaceAfter=10, alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        'CustomSubtitle', parent=styles['Normal'],
        fontSize=10, textColor=colors.HexColor('#7f8c8d'),
        spaceAfter=15, alignment=TA_CENTER,
    )
    section_style = ParagraphStyle(
        'SectionHeading', parent=styles['Heading2'],
        fontSize=13, textColor=colors.HexColor('#2c3e50'),
        spaceBefore=15, spaceAfter=8,
    )

    elements = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Title page / summary ──────────────────────────────────────
    elements.append(Paragraph("Missing Chip Trays Report (RC Only)", title_style))
    elements.append(Paragraph(f"Generated: {timestamp}", subtitle_style))

    # Summary table
    summary_data = [
        ["Metric", "Value"],
        ["RC holes analysed", f"{results.rc_holes:,}"],
        ["Non-RC excluded", f"{results.diamond_holes:,}"],
        ["", ""],
        ["Expected tray intervals (20m)", f"{results.total_expected_trays:,}"],
        ["Existing tray images", f"{results.total_existing_trays:,}"],
        ["Missing tray intervals", f"{results.total_missing_trays:,}"],
        ["Tray coverage", f"{results.tray_coverage_pct:.1f}%"],
        ["", ""],
        ["Expected compartments", f"{results.total_expected_compartments:,}"],
        ["Existing compartment images", f"{results.total_existing_compartments:,}"],
        ["Missing compartments", f"{results.total_missing_compartments:,}"],
        ["Compartment coverage", f"{results.compartment_coverage_pct:.1f}%"],
        ["", ""],
        ["Unprocessed trays (no compartments)", f"{len(results.unprocessed_trays):,}"],
        ["Holes with zero images", f"{len(results.holes_with_no_images):,}"],
    ]

    summary_table = Table(summary_data, colWidths=[3 * inch, 2 * inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
    ]))
    elements.append(summary_table)

    # ── Missing trays section ─────────────────────────────────────
    if results.missing_trays:
        elements.append(PageBreak())
        elements.append(Paragraph(
            f"Missing Tray Intervals ({results.total_missing_trays:,})",
            section_style,
        ))

        # Data is already sorted by project → start_date → hole → depth
        sorted_trays = results.missing_trays

        # Build table in 3 columns
        cols_per_page = 3
        rows_per_col = 40

        for page_start in range(0, len(sorted_trays), cols_per_page * rows_per_col):
            page_chunk = sorted_trays[page_start:page_start + cols_per_page * rows_per_col]

            # Split into columns
            col_chunks = []
            for i in range(cols_per_page):
                start = i * rows_per_col
                end = min(start + rows_per_col, len(page_chunk))
                if start < len(page_chunk):
                    col_chunks.append(page_chunk[start:end])

            if not col_chunks:
                continue

            # Build header
            header = []
            for _ in col_chunks:
                header.extend(["Project", "HoleID", "From", "To", " "])
            header = header[:-1]  # Remove trailing spacer

            table_data = [header]

            # Build rows
            max_rows = max(len(c) for c in col_chunks)
            for row_idx in range(max_rows):
                row = []
                for col_data in col_chunks:
                    if row_idx < len(col_data):
                        t = col_data[row_idx]
                        row.extend([t.project_code, t.hole_id, str(t.depth_from), str(t.depth_to), ""])
                    else:
                        row.extend(["", "", "", "", ""])
                row = row[:-1]
                table_data.append(row)

            col_widths = []
            for _ in col_chunks:
                col_widths.extend([0.5 * inch, 0.9 * inch, 0.4 * inch, 0.4 * inch, 0.1 * inch])
            col_widths = col_widths[:-1]

            table = Table(table_data, colWidths=col_widths)

            style_cmds = [
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 7),
                ('GRID', (0, 0), (-1, -1), 0.25, colors.HexColor('#ddd')),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
            ]

            # Colour-code by hole ID (alternating shades per hole)
            if len(table_data) > 1:
                prev_hole = None
                shade_idx = 0
                shades = [colors.white, colors.HexColor('#e8f4fd')]
                for row_i in range(1, len(table_data)):
                    # Column index 1 is HoleID in the first data column
                    current_hole = table_data[row_i][1] if table_data[row_i][1] else prev_hole
                    if current_hole != prev_hole and current_hole:
                        shade_idx = 1 - shade_idx
                        prev_hole = current_hole
                    style_cmds.append(
                        ('BACKGROUND', (0, row_i), (3, row_i), shades[shade_idx])
                    )

            table.setStyle(TableStyle(style_cmds))
            elements.append(table)
            elements.append(Spacer(1, 10))

            if page_start + cols_per_page * rows_per_col < len(sorted_trays):
                elements.append(PageBreak())

    # ── Missing compartments section ──────────────────────────────
    if results.trays_with_missing:
        elements.append(PageBreak())
        elements.append(Paragraph(
            f"Trays with Missing Compartments ({len(results.trays_with_missing):,} trays)",
            section_style,
        ))

        table_data = [["Project", "HoleID", "Tray", "Expected", "Found", "Missing", "Missing %", "Missing Depths"]]

        for tray in results.trays_with_missing[:200]:  # Cap at 200 rows for PDF size
            depths_str = ", ".join(str(d) for d in tray.missing_depths[:10])
            if len(tray.missing_depths) > 10:
                depths_str += f" (+{len(tray.missing_depths) - 10} more)"

            table_data.append([
                tray.project_code,
                tray.hole_id,
                f"{tray.tray_from}-{tray.tray_to}",
                str(tray.expected_count),
                str(tray.found_count),
                str(tray.missing_count),
                f"{tray.missing_pct:.0f}%",
                depths_str,
            ])

        table = Table(table_data, colWidths=[
            0.5 * inch, 0.8 * inch, 0.7 * inch, 0.55 * inch, 0.45 * inch,
            0.55 * inch, 0.55 * inch, 2.85 * inch,
        ])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#c0392b')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fdf2f2')]),
        ]))
        elements.append(table)

    # ── Unprocessed trays ─────────────────────────────────────────
    if results.unprocessed_trays:
        elements.append(PageBreak())
        elements.append(Paragraph(
            f"Unprocessed Trays — Original Exists, No Compartments ({len(results.unprocessed_trays):,})",
            section_style,
        ))
        elements.append(Paragraph(
            "These tray images exist in Approved Originals but have zero extracted compartment images.",
            styles['Normal'],
        ))
        elements.append(Spacer(1, 10))

        table_data = [["Project", "HoleID", "From", "To"]]
        for t in results.unprocessed_trays:
            table_data.append([t.project_code, t.hole_id, str(t.depth_from), str(t.depth_to)])

        table = Table(table_data, colWidths=[0.8 * inch, 1.2 * inch, 0.8 * inch, 0.8 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e67e22')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fef5e7')]),
        ]))
        elements.append(table)

    # ── Holes with no images ──────────────────────────────────────
    if results.holes_with_no_images:
        elements.append(PageBreak())
        elements.append(Paragraph(
            f"Holes with No Images ({len(results.holes_with_no_images):,})",
            section_style,
        ))
        elements.append(Paragraph(
            "These RC holes have collar data but no tray or compartment images found.",
            styles['Normal'],
        ))
        elements.append(Spacer(1, 10))

        # Multi-column list
        holes_sorted = sorted(results.holes_with_no_images)
        cols = 5
        rows_needed = math.ceil(len(holes_sorted) / cols)
        table_data = []
        for r in range(rows_needed):
            row = []
            for c in range(cols):
                idx = r + c * rows_needed
                row.append(holes_sorted[idx] if idx < len(holes_sorted) else "")
            table_data.append(row)

        table = Table(table_data, colWidths=[1.4 * inch] * cols)
        table.setStyle(TableStyle([
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.HexColor('#ddd')),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
        ]))
        elements.append(table)

    # Build
    doc.build(elements)
    logger.info(f"PDF exported: {output_path}")
    return True
