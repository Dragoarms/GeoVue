"""True/apparent thickness helpers for correlation displays."""

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List

def _normalise_dip_degrees(dip_degrees: float) -> float:
    """Return absolute bed dip from horizontal, validating the hyperbolic limit."""
    dip = abs(float(dip_degrees))
    if dip >= 90.0:
        raise ValueError("Dip must be less than 90 degrees for finite apparent thickness")
    return dip


def apparent_factor_from_dip(dip_degrees: float) -> float:
    """
    Return apparent/true thickness factor for a vertical hole.

    Dip is bed dip from horizontal. 60 degrees returns 2.0 because
    apparent thickness = true thickness / cos(60).
    """
    dip = _normalise_dip_degrees(dip_degrees)
    return 1.0 / math.cos(math.radians(dip))


def apparent_thickness_from_true(true_thickness: float, dip_degrees: float) -> float:
    """Convert true thickness to apparent thickness for a vertical hole."""
    return float(true_thickness) * apparent_factor_from_dip(dip_degrees)


def true_thickness_from_apparent(apparent_thickness: float, dip_degrees: float) -> float:
    """Convert apparent thickness to true thickness for a vertical hole."""
    return float(apparent_thickness) / apparent_factor_from_dip(dip_degrees)


def stretch_scale_for_true_thickness(dip_degrees: float) -> float:
    """
    Return display scale that compresses apparent downhole thickness to true thickness.

    At 60 degrees this is 0.5, matching the doubling of apparent thickness.
    """
    return 1.0 / apparent_factor_from_dip(dip_degrees)


@dataclass
class ThicknessBand:
    """Manual depth interval with a local bed dip from horizontal."""

    depth_from: float
    depth_to: float
    dip_degrees: float

    def __post_init__(self) -> None:
        self.depth_from = float(self.depth_from)
        self.depth_to = float(self.depth_to)
        self.dip_degrees = _normalise_dip_degrees(self.dip_degrees)
        if self.depth_from == self.depth_to:
            raise ValueError("Thickness band must have non-zero depth length")
        if self.depth_from > self.depth_to:
            self.depth_from, self.depth_to = self.depth_to, self.depth_from

    @property
    def apparent_thickness(self) -> float:
        """Measured downhole interval length in metres."""
        return abs(self.depth_to - self.depth_from)

    @property
    def apparent_factor(self) -> float:
        """Apparent/true thickness factor for this band."""
        return apparent_factor_from_dip(self.dip_degrees)

    @property
    def stretch_scale(self) -> float:
        """Visual scale that compresses this apparent interval to true thickness."""
        return stretch_scale_for_true_thickness(self.dip_degrees)

    @property
    def true_thickness(self) -> float:
        """True thickness represented by this apparent interval."""
        return true_thickness_from_apparent(self.apparent_thickness, self.dip_degrees)

    def to_dict(self) -> Dict[str, float]:
        """Return a serializable band representation."""
        return {
            "depth_from": self.depth_from,
            "depth_to": self.depth_to,
            "dip_degrees": self.dip_degrees,
            "apparent_thickness": self.apparent_thickness,
            "true_thickness": self.true_thickness,
            "apparent_factor": self.apparent_factor,
            "stretch_scale": self.stretch_scale,
        }


def summarise_thickness_bands(bands: Iterable[ThicknessBand]) -> Dict[str, float]:
    """Summarise a manual downhole thickness profile."""
    band_list: List[ThicknessBand] = list(bands)
    total_apparent = sum(b.apparent_thickness for b in band_list)
    total_true = sum(b.true_thickness for b in band_list)
    if total_apparent <= 0 or total_true <= 0:
        return {
            "total_apparent": total_apparent,
            "total_true": total_true,
            "apparent_factor": 0.0,
            "stretch_scale": 0.0,
        }

    return {
        "total_apparent": total_apparent,
        "total_true": total_true,
        "apparent_factor": total_apparent / total_true,
        "stretch_scale": total_true / total_apparent,
    }
