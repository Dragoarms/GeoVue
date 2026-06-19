"""Depth helpers for register-based compartment re-extraction."""

from typing import Any, Dict, Iterable, Optional, Tuple


def coerce_optional_int(value: Any) -> Optional[int]:
    """Return ``value`` as an int, or None when it cannot be interpreted."""
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def infer_register_compartment_interval(
    original_record: Dict[str, Any],
    corner_records: Iterable[Dict[str, Any]],
    fallback_interval: int = 1,
) -> int:
    """Infer the interval represented by each compartment in stored register data."""
    for key in ("Compartment_Interval", "compartment_interval", "Interval"):
        explicit = coerce_optional_int(original_record.get(key))
        if explicit and explicit > 0:
            return explicit

    depth_from = coerce_optional_int(original_record.get("Depth_From"))
    depth_to = coerce_optional_int(original_record.get("Depth_To"))
    corners = list(corner_records or [])
    max_compartment = max(
        (coerce_optional_int(record.get("Compartment_Number")) or 0 for record in corners),
        default=0,
    )
    compartment_count = max(max_compartment, len(corners))

    if (
        depth_from is not None
        and depth_to is not None
        and depth_to > depth_from
        and compartment_count > 0
    ):
        span = depth_to - depth_from
        if span % compartment_count == 0:
            interval = span // compartment_count
        else:
            interval = round(span / compartment_count)
        if interval > 0:
            return interval

    fallback = coerce_optional_int(fallback_interval)
    return fallback if fallback and fallback > 0 else 1


def resolve_register_compartment_depths(
    original_record: Dict[str, Any],
    corner_record: Dict[str, Any],
    corner_records: Iterable[Dict[str, Any]],
    fallback_interval: int = 1,
) -> Tuple[int, int]:
    """Return the interval ``(from_depth, to_depth)`` for a stored corner record."""
    interval = infer_register_compartment_interval(
        original_record,
        corner_records,
        fallback_interval=fallback_interval,
    )

    tray_depth_from = coerce_optional_int(original_record.get("Depth_From"))
    if tray_depth_from is None:
        tray_depth_from = coerce_optional_int(corner_record.get("Depth_From")) or 0

    compartment_num = coerce_optional_int(corner_record.get("Compartment_Number"))
    if compartment_num and compartment_num > 0:
        depth_from = tray_depth_from + ((compartment_num - 1) * interval)
        return depth_from, depth_from + interval

    legacy_depth_from = coerce_optional_int(corner_record.get("Depth_From"))
    legacy_depth_to = coerce_optional_int(corner_record.get("Depth_To"))
    if (
        legacy_depth_from is not None
        and legacy_depth_to is not None
        and legacy_depth_to > legacy_depth_from
    ):
        return legacy_depth_from, legacy_depth_to

    return tray_depth_from, tray_depth_from + interval
