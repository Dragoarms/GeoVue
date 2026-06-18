"""Shared helpers for HTML report (safe formatting, etc.)."""
from typing import Any, Optional
import numpy as np


def _safe_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return float(value)
    except Exception:
        return None


def _format_metric(value: Optional[float], unit: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if unit == "%":
        return f"{value:.1f}%"
    if unit == "m":
        return f"{value:,.0f}m"
    return f"{value:.0f}"
