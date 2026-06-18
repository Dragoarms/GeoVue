"""
survey_trace.py - Build 3D drillhole trace from collar + survey (azimuth/dip at depth).

Converts collar coordinates and survey stations (depth, azimuth, dip) into a
list of (depth, x, y, z) points representing the hole path. Supports single
station (straight line) and multiple stations (minimum curvature).

Convention: azimuth = degrees from North, clockwise; dip = inclination from
horizontal, typically negative downward. Collar (x, y, z) in East, North, RL.

Author: George Symonds
"""

import logging
import math
from typing import List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Dip for "vertical" fallback (straight down): -90 degrees
VERTICAL_DIP_DEG = -90.0


def _deg2rad(deg: float) -> float:
    return math.radians(deg)


def _straight_segment(
    x0: float, y0: float, z0: float,
    depth0: float, depth1: float,
    azimuth_deg: float, dip_deg: float,
) -> Tuple[float, float, float]:
    """
    Compute (x, y, z) at depth1 given start (x0,y0,z0) at depth0 and constant
    azimuth/dip. Convention: x=East, y=North, z=RL; down is negative.
    """
    md = depth1 - depth0
    if md <= 0:
        return (x0, y0, z0)
    azi = _deg2rad(azimuth_deg)
    dip = _deg2rad(dip_deg)
    # Horizontal displacement = md * cos(dip); vertical = md * sin(dip)
    # North = horizontal * cos(azi), East = horizontal * sin(azi)
    horiz = md * math.cos(dip)
    vert = md * math.sin(dip)
    north = horiz * math.cos(azi)
    east = horiz * math.sin(azi)
    return (x0 + east, y0 + north, z0 + vert)


def _minimum_curvature_segment(
    x0: float, y0: float, z0: float,
    depth0: float, depth1: float,
    azi1_deg: float, dip1_deg: float,
    azi2_deg: float, dip2_deg: float,
) -> Tuple[float, float, float]:
    """
    Compute (x, y, z) at depth1 using minimum curvature between two stations.
    Inclination I from vertical: I = 90 + dip (so dip=-90 => I=0, dip=0 => I=90).
    TVD (vertical) is positive down; our z is RL so delta_z = -TVD.
    """
    md = depth1 - depth0
    if md <= 0:
        return (x0, y0, z0)
    I1 = _deg2rad(90 + dip1_deg)
    I2 = _deg2rad(90 + dip2_deg)
    A1 = _deg2rad(azi1_deg)
    A2 = _deg2rad(azi2_deg)
    # Dogleg angle (radians)
    cos_beta = (
        math.cos(I2 - I1)
        - math.sin(I1) * math.sin(I2) * (1 - math.cos(A2 - A1))
    )
    cos_beta = max(-1.0, min(1.0, cos_beta))
    beta = math.acos(cos_beta)
    # Ratio factor
    if beta < 1e-10:
        rf = 1.0
    else:
        rf = (2.0 / beta) * math.tan(beta / 2.0)
    half_md = md / 2.0
    delta_n = half_md * (math.sin(I1) * math.cos(A1) + math.sin(I2) * math.cos(A2)) * rf
    delta_e = half_md * (math.sin(I1) * math.sin(A1) + math.sin(I2) * math.sin(A2)) * rf
    tvd = half_md * (math.cos(I1) + math.cos(I2)) * rf  # positive = down
    # Our z is RL (elevation), so going down = negative delta_z
    return (x0 + delta_e, y0 + delta_n, z0 - tvd)


def build_trace(
    collar_x: float,
    collar_y: float,
    collar_z: float,
    survey_rows: List[Tuple[float, float, float]],
    vertical_if_missing: bool = True,
    max_depth: Optional[float] = None,
) -> List[Tuple[float, float, float, float]]:
    """
    Build 3D trace from collar and survey rows.

    Args:
        collar_x, collar_y, collar_z: Collar coordinates (East, North, RL).
        survey_rows: List of (depth, azimuth_deg, dip_deg), should be sorted by depth.
        vertical_if_missing: If True, use vertical hole when no valid survey rows.
        max_depth: Optional max depth for vertical fallback (second point).

    Returns:
        List of (depth, x, y, z) with depth 0 = collar. Sorted by depth.
    """
    # Normalize and validate survey rows: (depth, azimuth, dip)
    rows: List[Tuple[float, float, float]] = []
    for t in survey_rows:
        if len(t) != 3:
            continue
        depth, azi, dip = t[0], t[1], t[2]
        try:
            depth = float(depth)
            azi = float(azi) % 360.0
            dip = max(-90.0, min(90.0, float(dip)))
        except (TypeError, ValueError):
            continue
        if depth < 0 or (math.isnan(depth) or math.isnan(azi) or math.isnan(dip)):
            continue
        rows.append((depth, azi, dip))
    # Deduplicate by depth, keep first
    seen_depths: set = set()
    unique: List[Tuple[float, float, float]] = []
    for r in rows:
        if r[0] not in seen_depths:
            seen_depths.add(r[0])
            unique.append(r)
    rows = sorted(unique, key=lambda r: r[0])

    trace: List[Tuple[float, float, float, float]] = [(0.0, collar_x, collar_y, collar_z)]

    if not rows:
        if vertical_if_missing and max_depth is not None and max_depth > 0:
            # Vertical hole: one segment from collar
            x1, y1, z1 = _straight_segment(
                collar_x, collar_y, collar_z,
                0.0, max_depth,
                0.0, VERTICAL_DIP_DEG,
            )
            trace.append((max_depth, x1, y1, z1))
        return trace

    # Single station: straight line from collar to "infinite" depth with that azi/dip
    if len(rows) == 1:
        depth1, azi1, dip1 = rows[0]
        if depth1 <= 0:
            if max_depth and max_depth > 0:
                x1, y1, z1 = _straight_segment(
                    collar_x, collar_y, collar_z,
                    0.0, max_depth, azi1, dip1,
                )
                trace.append((max_depth, x1, y1, z1))
            return trace
        x1, y1, z1 = _straight_segment(
            collar_x, collar_y, collar_z,
            0.0, depth1, azi1, dip1,
        )
        trace.append((depth1, x1, y1, z1))
        if max_depth is not None and max_depth > depth1:
            x2, y2, z2 = _straight_segment(
                x1, y1, z1, depth1, max_depth, azi1, dip1,
            )
            trace.append((max_depth, x2, y2, z2))
        return trace

    # Multiple stations: minimum curvature between consecutive stations
    depth0, azi0, dip0 = 0.0, rows[0][1], rows[0][2]
    x0, y0, z0 = collar_x, collar_y, collar_z
    for i in range(len(rows)):
        depth1, azi1, dip1 = rows[i][0], rows[i][1], rows[i][2]
        if depth1 <= depth0:
            continue
        if i == 0:
            # Segment from collar to first station: use first station's azi/dip
            x1, y1, z1 = _straight_segment(
                x0, y0, z0, depth0, depth1, azi1, dip1,
            )
        else:
            depth_prev, azi_prev, dip_prev = rows[i - 1][0], rows[i - 1][1], rows[i - 1][2]
            x1, y1, z1 = _minimum_curvature_segment(
                x0, y0, z0, depth0, depth1,
                azi_prev, dip_prev, azi1, dip1,
            )
        trace.append((depth1, x1, y1, z1))
        depth0, azi0, dip0 = depth1, azi1, dip1
        x0, y0, z0 = x1, y1, z1

    return trace


def xyz_at_depth(
    trace: List[Tuple[float, float, float, float]],
    depth: float,
) -> Optional[Tuple[float, float, float]]:
    """
    Interpolate (x, y, z) at given depth from a trace.

    Linear interpolation between bracketing trace points. If depth is outside
    [0, max_depth], returns None (no extrapolation).

    Args:
        trace: List of (depth, x, y, z) sorted by depth.
        depth: Requested depth.

    Returns:
        (x, y, z) or None if depth out of range or trace too short.
    """
    if not trace or len(trace) < 2:
        if trace and len(trace) == 1:
            d0, x0, y0, z0 = trace[0]
            if abs(depth - d0) < 1e-9:
                return (x0, y0, z0)
        return None
    depths = [t[0] for t in trace]
    if depth < depths[0] - 1e-9 or depth > depths[-1] + 1e-9:
        return None
    # Find bracket
    for i in range(len(trace) - 1):
        d0, x0, y0, z0 = trace[i]
        d1, x1, y1, z1 = trace[i + 1]
        if d0 - 1e-9 <= depth <= d1 + 1e-9:
            if abs(d1 - d0) < 1e-9:
                return (x0, y0, z0)
            t = (depth - d0) / (d1 - d0)
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            z = z0 + t * (z1 - z0)
            return (x, y, z)
    return None
