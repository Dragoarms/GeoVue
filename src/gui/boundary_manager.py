# gui/boundary_manager.py

"""
Boundary Manager - Single source of truth for compartment boundary management.
Handles all boundary operations with marker ID as the primary key.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from dataclasses import dataclass, field


@dataclass
class BoundaryData:
    """Single boundary entry with all metadata."""

    marker_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    boundary_type: str = "detected"  # "detected", "manual", "interpolated"
    depth: Optional[int] = None

    def __post_init__(self):
        # Ensure all coordinates are regular Python ints
        self.marker_id = int(self.marker_id)
        self.x1 = int(self.x1)
        self.y1 = int(self.y1)
        self.x2 = int(self.x2)
        self.y2 = int(self.y2)

    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """Get boundary as tuple (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center_x(self) -> int:
        """Get horizontal center of boundary."""
        return (self.x1 + self.x2) // 2

    @property
    def width(self) -> int:
        """Get width of boundary."""
        return self.x2 - self.x1


class BoundaryManager:
    """
    Manages all compartment boundaries with marker ID as the key.
    Provides a single source of truth for boundary data.
    """

    def __init__(
        self,
        marker_to_compartment: Dict[int, int],
        expected_compartment_count: int = 20,
        compartment_interval: int = 1,
    ):
        """
        Initialize boundary manager.

        Args:
            marker_to_compartment: Mapping of marker IDs to depth values
            expected_compartment_count: Total expected compartments
            compartment_interval: Interval between compartments (1 or 2 meters)
        """
        self.logger = logging.getLogger(__name__)
        self.boundaries: Dict[int, BoundaryData] = {}
        self.marker_to_compartment = self._normalize_dict_keys(marker_to_compartment)
        self.expected_compartment_count = expected_compartment_count
        self.compartment_interval = compartment_interval

        # Cache for performance
        self._sorted_boundaries_cache = None
        self._cache_valid = False

    def _normalize_dict_keys(self, d: Dict[Any, Any]) -> Dict[int, Any]:
        """Convert all dict keys to regular Python ints."""
        return {int(k): v for k, v in d.items()}

    def _normalize_marker_id(self, marker_id: Union[int, np.integer]) -> int:
        """Ensure marker ID is a regular Python int."""
        return int(marker_id)

    def _invalidate_cache(self):
        """Invalidate sorted boundaries cache."""
        self._cache_valid = False
        self._sorted_boundaries_cache = None

    def add_boundary(
        self,
        marker_id: Union[int, np.integer],
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        boundary_type: str = "detected",
    ) -> bool:
        """
        Add or update a boundary.

        Args:
            marker_id: Marker ID for this boundary
            x1, y1, x2, y2: Boundary coordinates
            boundary_type: Type of boundary ("detected", "manual", "interpolated")

        Returns:
            True if boundary was added/updated, False if invalid
        """
        marker_id = self._normalize_marker_id(marker_id)

        # Validate marker ID is in expected range
        if marker_id < 4 or marker_id > 23:
            self.logger.warning(
                f"Invalid marker ID {marker_id} - outside compartment range"
            )
            return False

        # Get depth from marker mapping
        depth = self.marker_to_compartment.get(marker_id)
        if depth is None:
            self.logger.warning(f"No depth mapping for marker {marker_id}")
            depth = marker_id - 3  # Fallback calculation

        # Create boundary data
        boundary = BoundaryData(
            marker_id=marker_id,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            boundary_type=boundary_type,
            depth=depth,
        )

        # Check if we're overwriting an existing boundary
        if marker_id in self.boundaries:
            old_type = self.boundaries[marker_id].boundary_type
            if old_type == "manual" and boundary_type != "manual":
                self.logger.warning(
                    f"Attempting to overwrite manual boundary {marker_id} with {boundary_type}"
                )
                return False

        self.boundaries[marker_id] = boundary
        self._invalidate_cache()

        self.logger.debug(
            f"Added {boundary_type} boundary for marker {marker_id} at depth {depth}m"
        )
        return True

    def remove_boundary(self, marker_id: Union[int, np.integer]) -> bool:
        """
        Remove a boundary by marker ID.

        Args:
            marker_id: Marker ID to remove

        Returns:
            True if removed, False if not found
        """
        marker_id = self._normalize_marker_id(marker_id)

        if marker_id in self.boundaries:
            boundary = self.boundaries.pop(marker_id)
            self._invalidate_cache()
            self.logger.debug(
                f"Removed {boundary.boundary_type} boundary for marker {marker_id}"
            )
            return True
        return False

    def get_boundary(self, marker_id: Union[int, np.integer]) -> Optional[BoundaryData]:
        """Get boundary data for a specific marker."""
        marker_id = self._normalize_marker_id(marker_id)
        return self.boundaries.get(marker_id)

    def has_boundary(self, marker_id: Union[int, np.integer]) -> bool:
        """Check if a marker has a boundary."""
        marker_id = self._normalize_marker_id(marker_id)
        return marker_id in self.boundaries

    def get_sorted_boundaries(self) -> List[Tuple[int, int, int, int]]:
        """
        Get boundaries sorted by marker ID (depth order).

        Returns:
            List of (x1, y1, x2, y2) tuples in depth order
        """
        if self._cache_valid and self._sorted_boundaries_cache is not None:
            return self._sorted_boundaries_cache

        # Sort by marker ID (which corresponds to depth order)
        sorted_markers = sorted(self.boundaries.keys())
        self._sorted_boundaries_cache = [
            self.boundaries[mid].bounds for mid in sorted_markers
        ]
        self._cache_valid = True

        return self._sorted_boundaries_cache

    def get_boundary_to_marker_mapping(self) -> Dict[int, int]:
        """
        Get mapping of boundary index to marker ID.

        Returns:
            Dict mapping boundary index to marker ID
        """
        sorted_markers = sorted(self.boundaries.keys())
        return {i: mid for i, mid in enumerate(sorted_markers)}

    def get_missing_markers(
        self, start_marker: int = 4, count: Optional[int] = None
    ) -> List[int]:
        """
        Get list of markers that don't have boundaries.

        Args:
            start_marker: First marker ID (default 4)
            count: Number of expected markers (default: expected_compartment_count)

        Returns:
            List of missing marker IDs
        """
        if count is None:
            count = self.expected_compartment_count

        expected_markers = set(range(start_marker, start_marker + count))
        existing_markers = set(self.boundaries.keys())
        missing = sorted(expected_markers - existing_markers)

        self.logger.debug(
            f"Missing markers: {missing} "
            f"(expected {len(expected_markers)}, found {len(existing_markers)})"
        )
        return missing

    def get_boundaries_by_type(self, boundary_type: str) -> Dict[int, BoundaryData]:
        """Get all boundaries of a specific type."""
        return {
            mid: boundary
            for mid, boundary in self.boundaries.items()
            if boundary.boundary_type == boundary_type
        }

    def clear_interpolated_boundaries(self) -> int:
        """
        Remove all interpolated boundaries.

        Returns:
            Number of boundaries removed
        """
        interpolated = self.get_boundaries_by_type("interpolated")
        count = len(interpolated)

        for marker_id in interpolated:
            self.boundaries.pop(marker_id)

        if count > 0:
            self._invalidate_cache()
            self.logger.info(f"Removed {count} interpolated boundaries")

        return count

    def find_gaps(self) -> List[Dict[str, Any]]:
        """
        Find gaps in the boundary sequence.

        Returns:
            List of gap dictionaries with start/end markers and missing count
        """
        sorted_markers = sorted(self.boundaries.keys())
        gaps = []

        for i in range(len(sorted_markers) - 1):
            current_marker = sorted_markers[i]
            next_marker = sorted_markers[i + 1]

            # Check if there's a gap
            expected_next = current_marker + 1
            if next_marker > expected_next:
                missing_count = next_marker - current_marker - 1
                missing_markers = list(range(expected_next, next_marker))

                gap_info = {
                    "start_marker": current_marker,
                    "end_marker": next_marker,
                    "missing_count": missing_count,
                    "missing_markers": missing_markers,
                    "start_depth": self.marker_to_compartment.get(
                        current_marker, current_marker - 3
                    ),
                    "end_depth": self.marker_to_compartment.get(
                        next_marker, next_marker - 3
                    ),
                }
                gaps.append(gap_info)

                self.logger.debug(
                    f"Gap found: markers {current_marker}-{next_marker}, "
                    f"missing {missing_count} compartments"
                )

        return gaps

    def get_boundary_at_position(
        self, x: int, y: int
    ) -> Optional[Tuple[int, BoundaryData]]:
        """
        Find boundary at given position.

        Args:
            x, y: Position to check

        Returns:
            Tuple of (marker_id, BoundaryData) if found, None otherwise
        """
        self.logger.debug(
            f"Checking position ({x}, {y}) against {len(self.boundaries)} boundaries"
        )

        # Sort by marker ID to ensure consistent checking order
        sorted_items = sorted(self.boundaries.items(), key=lambda x: x[0])

        for marker_id, boundary in sorted_items:
            # Check both X and Y bounds
            x_match = boundary.x1 <= x <= boundary.x2
            y_match = boundary.y1 <= y <= boundary.y2

            if x_match and y_match:
                self.logger.debug(
                    f"Position ({x}, {y}) matched boundary {marker_id}: ({boundary.x1}-{boundary.x2}, {boundary.y1}-{boundary.y2})"
                )
                return (marker_id, boundary)
            elif x_match and not y_match:
                self.logger.debug(
                    f"Position ({x}, {y}) matched X but not Y for boundary {marker_id}"
                )
            elif y_match and not x_match:
                self.logger.debug(
                    f"Position ({x}, {y}) matched Y but not X for boundary {marker_id}"
                )

        self.logger.debug(f"No boundary found at position ({x}, {y})")
        return None

    def import_boundaries(
        self,
        boundaries: List[Tuple[int, int, int, int]],
        boundary_to_marker: Dict[int, int],
        boundary_types: Optional[List[str]] = None,
    ):
        """
        Import boundaries from old format.

        Args:
            boundaries: List of (x1, y1, x2, y2) tuples
            boundary_to_marker: Mapping of boundary index to marker ID
            boundary_types: Optional list of boundary types
        """
        self.boundaries.clear()
        self._invalidate_cache()

        # Normalize the mapping
        boundary_to_marker = self._normalize_dict_keys(boundary_to_marker)

        for idx, bounds in enumerate(boundaries):
            marker_id = boundary_to_marker.get(idx)
            if marker_id is None:
                self.logger.warning(f"No marker ID for boundary at index {idx}")
                continue

            boundary_type = "detected"
            if boundary_types and idx < len(boundary_types):
                boundary_type = boundary_types[idx]

            x1, y1, x2, y2 = bounds
            self.add_boundary(marker_id, x1, y1, x2, y2, boundary_type)

        self.logger.info(f"Imported {len(self.boundaries)} boundaries")

    def export_to_legacy_format(
        self,
    ) -> Tuple[List[Tuple[int, int, int, int]], Dict[int, int], List[str]]:
        """
        Export boundaries to legacy format for compatibility.

        Returns:
            Tuple of (boundaries list, boundary_to_marker dict, boundary_types list)
        """
        sorted_markers = sorted(self.boundaries.keys())

        boundaries = []
        boundary_to_marker = {}
        boundary_types = []

        for idx, marker_id in enumerate(sorted_markers):
            boundary = self.boundaries[marker_id]
            boundaries.append(boundary.bounds)
            boundary_to_marker[idx] = marker_id
            boundary_types.append(boundary.boundary_type)

        return boundaries, boundary_to_marker, boundary_types

    def validate_consistency(self) -> List[str]:
        """
        Validate boundary consistency and return any issues found.

        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []

        # Check for duplicate depths
        depth_to_markers = {}
        for marker_id, boundary in self.boundaries.items():
            depth = boundary.depth
            if depth in depth_to_markers:
                issues.append(
                    f"Duplicate depth {depth}m: markers {depth_to_markers[depth]} and {marker_id}"
                )
            else:
                depth_to_markers[depth] = marker_id

        # Check for overlapping boundaries
        sorted_boundaries = sorted(self.boundaries.items(), key=lambda x: x[1].x1)

        for i in range(len(sorted_boundaries) - 1):
            curr_id, curr_boundary = sorted_boundaries[i]
            next_id, next_boundary = sorted_boundaries[i + 1]

            if curr_boundary.x2 > next_boundary.x1:
                issues.append(
                    f"Overlap between markers {curr_id} and {next_id}: "
                    f"x2={curr_boundary.x2} > x1={next_boundary.x1}"
                )

        return issues

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about current boundaries."""
        total = len(self.boundaries)
        by_type = {
            "detected": len(self.get_boundaries_by_type("detected")),
            "manual": len(self.get_boundaries_by_type("manual")),
            "interpolated": len(self.get_boundaries_by_type("interpolated")),
        }

        gaps = self.find_gaps()
        missing = self.get_missing_markers()

        return {
            "total_boundaries": total,
            "boundaries_by_type": by_type,
            "gaps_count": len(gaps),
            "missing_markers_count": len(missing),
            "missing_markers": missing,
            "completion_percentage": (total / self.expected_compartment_count * 100),
        }
