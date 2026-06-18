"""
Trace scaling and similarity helpers for drillhole correlation.

The canvas widgets use these functions for drawing, and future similarity
search can use the same transformations so visual correlation and algorithmic
correlation stay aligned.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import mean
from typing import Iterable, List, Optional, Sequence, Tuple


SCALE_MODE_LABELS: Tuple[Tuple[str, str], ...] = (
    ("raw", "Raw"),
    ("global", "Global"),
    ("per_hole_minmax", "Hole Min/Max"),
    ("per_hole_percentile", "Hole P2-P98"),
    ("zscore", "Z-score"),
)

_LABEL_TO_MODE = {label.lower(): mode for mode, label in SCALE_MODE_LABELS}
_MODE_TO_LABEL = dict(SCALE_MODE_LABELS)


@dataclass(frozen=True)
class TraceScaleResult:
    """Scaled trace values and the plotting range to use for them."""

    values: List[float]
    plot_min: float
    plot_max: float
    label: str
    mode: str


def canonical_scale_mode(mode: Optional[str]) -> str:
    """Return the persisted scale-mode key for a stored key or display label."""
    if not mode:
        return "raw"

    text = str(mode).strip().lower()
    if text in _LABEL_TO_MODE:
        return _LABEL_TO_MODE[text]

    key = text.replace("-", "_").replace(" ", "_").replace("/", "_")
    aliases = {
        "none": "raw",
        "auto": "raw",
        "hole_min_max": "per_hole_minmax",
        "per_hole_min_max": "per_hole_minmax",
        "hole_minmax": "per_hole_minmax",
        "minmax": "per_hole_minmax",
        "hole_p2_p98": "per_hole_percentile",
        "p2_p98": "per_hole_percentile",
        "percentile": "per_hole_percentile",
        "per_hole_p2_p98": "per_hole_percentile",
        "z_score": "zscore",
        "standard": "zscore",
        "standard_score": "zscore",
    }
    key = aliases.get(key, key)
    return key if key in _MODE_TO_LABEL else "raw"


def scale_mode_display_name(mode: Optional[str]) -> str:
    """Return the user-facing label for a scale-mode key."""
    return _MODE_TO_LABEL[canonical_scale_mode(mode)]


def finite_float_values(values: Iterable[object]) -> List[float]:
    """Convert an iterable to finite floats, dropping None/NaN/inf values."""
    result: List[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if isfinite(numeric):
            result.append(numeric)
    return result


def percentile(values: Sequence[float], percentile_value: float) -> float:
    """Linear-interpolated percentile without requiring numpy."""
    if not values:
        raise ValueError("percentile requires at least one value")

    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]

    bounded = max(0.0, min(100.0, float(percentile_value)))
    rank = (bounded / 100.0) * (len(ordered) - 1)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = rank - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction


def scale_trace_values(
    values: Sequence[object],
    scale_mode: Optional[str] = "raw",
    *,
    auto_scale: bool = True,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    global_min: Optional[float] = None,
    global_max: Optional[float] = None,
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
) -> TraceScaleResult:
    """
    Scale trace values for plotting or similarity prep.

    Raw and global modes preserve original units. Per-hole modes normalise to
    0..1. Z-score mode standardises to mean 0 and standard deviation 1.
    """
    numeric = finite_float_values(values)
    if not numeric:
        return TraceScaleResult([], 0.0, 1.0, "No data", canonical_scale_mode(scale_mode))

    mode = canonical_scale_mode(scale_mode)

    if mode == "per_hole_minmax":
        source_min = min(numeric)
        source_max = max(numeric)
        return _scale_to_unit_interval(numeric, source_min, source_max, "Hole Min/Max", mode)

    if mode == "per_hole_percentile":
        source_min = percentile(numeric, percentile_low)
        source_max = percentile(numeric, percentile_high)
        return _scale_to_unit_interval(
            numeric,
            source_min,
            source_max,
            f"P{percentile_low:g}-P{percentile_high:g}",
            mode,
        )

    if mode == "zscore":
        centre = mean(numeric)
        variance = sum((value - centre) ** 2 for value in numeric) / len(numeric)
        std_dev = sqrt(variance)
        if std_dev == 0.0:
            scaled = [0.0 for _ in numeric]
        else:
            scaled = [(value - centre) / std_dev for value in numeric]
        plot_min = min(scaled)
        plot_max = max(scaled)
        if plot_min == plot_max:
            plot_min -= 1.0
            plot_max += 1.0
        return TraceScaleResult(scaled, plot_min, plot_max, "Z-score", mode)

    if mode == "global" and global_min is not None and global_max is not None:
        plot_min = float(global_min)
        plot_max = float(global_max)
        label = f"Global {plot_min:g}-{plot_max:g}"
    elif not auto_scale and min_value is not None and max_value is not None:
        plot_min = float(min_value)
        plot_max = float(max_value)
        label = f"{plot_min:g}-{plot_max:g}"
    else:
        plot_min = min(numeric)
        plot_max = max(numeric)
        label = f"{plot_min:g}-{plot_max:g}"

    if plot_min == plot_max:
        plot_min -= 0.5
        plot_max += 0.5
    return TraceScaleResult(numeric, plot_min, plot_max, label, mode)


def _scale_to_unit_interval(
    values: Sequence[float],
    source_min: float,
    source_max: float,
    label: str,
    mode: str,
) -> TraceScaleResult:
    """Scale values into 0..1, clipping outliers."""
    if source_min == source_max:
        return TraceScaleResult([0.5 for _ in values], 0.0, 1.0, label, mode)

    value_range = source_max - source_min
    scaled = [
        max(0.0, min(1.0, (float(value) - source_min) / value_range))
        for value in values
    ]
    return TraceScaleResult(scaled, 0.0, 1.0, label, mode)


def pearson_similarity(values_a: Sequence[object], values_b: Sequence[object]) -> Optional[float]:
    """
    Return Pearson correlation for aligned trace values.

    Returns None when fewer than two paired finite samples are available or
    either trace has no variance.
    """
    paired: List[Tuple[float, float]] = []
    for left, right in zip(values_a, values_b):
        try:
            left_value = float(left)
            right_value = float(right)
        except (TypeError, ValueError):
            continue
        if isfinite(left_value) and isfinite(right_value):
            paired.append((left_value, right_value))

    if len(paired) < 2:
        return None

    left_values = [left for left, _right in paired]
    right_values = [right for _left, right in paired]
    left_mean = mean(left_values)
    right_mean = mean(right_values)
    left_var = sum((value - left_mean) ** 2 for value in left_values)
    right_var = sum((value - right_mean) ** 2 for value in right_values)
    denominator = sqrt(left_var * right_var)
    if denominator == 0.0:
        return None

    numerator = sum(
        (left - left_mean) * (right - right_mean)
        for left, right in paired
    )
    return numerator / denominator


def resample_depth_points(
    points: Sequence[Tuple[object, object]],
    depth_from: float,
    depth_to: float,
    *,
    step: float = 0.1,
) -> Tuple[List[float], List[Optional[float]]]:
    """
    Resample depth/value points onto a regular depth grid.

    This gives the future similarity matcher a stable 0.1 m grid for comparing
    traces sampled at slightly different depths. Values outside the supplied
    point range are returned as None.
    """
    if step <= 0:
        raise ValueError("step must be positive")

    finite_points = []
    for depth, value in points:
        try:
            depth_value = float(depth)
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if isfinite(depth_value) and isfinite(numeric_value):
            finite_points.append((depth_value, numeric_value))

    # Deduplicate depths with the last value winning, then sort.
    by_depth = {depth: value for depth, value in finite_points}
    ordered = sorted(by_depth.items())

    depths: List[float] = []
    values: List[Optional[float]] = []
    current = float(depth_from)
    end = float(depth_to)
    epsilon = step / 1000.0
    right_index = 0

    while current <= end + epsilon:
        depth = round(current, 6)
        depths.append(depth)
        values.append(_interpolate_at_depth(ordered, depth, right_index))
        while right_index < len(ordered) and ordered[right_index][0] < depth:
            right_index += 1
        current += step

    return depths, values


def _interpolate_at_depth(
    ordered_points: Sequence[Tuple[float, float]],
    depth: float,
    start_index: int,
) -> Optional[float]:
    """Linearly interpolate an ordered depth series at one depth."""
    if not ordered_points:
        return None

    right_index = max(0, min(start_index, len(ordered_points) - 1))
    while right_index < len(ordered_points) and ordered_points[right_index][0] < depth:
        right_index += 1

    if right_index >= len(ordered_points):
        return None

    right_depth, right_value = ordered_points[right_index]
    if right_depth == depth:
        return right_value

    left_index = right_index - 1
    if left_index < 0:
        return None

    left_depth, left_value = ordered_points[left_index]
    if left_depth == right_depth:
        return right_value

    fraction = (depth - left_depth) / (right_depth - left_depth)
    return left_value + (right_value - left_value) * fraction
