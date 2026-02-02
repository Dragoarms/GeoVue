"""Unit tests for survey_trace module (build_trace, xyz_at_depth)."""
import math
import pytest

from processing.DataManager.survey_trace import (
    build_trace,
    xyz_at_depth,
    VERTICAL_DIP_DEG,
)


class TestBuildTrace:
    """Tests for build_trace."""

    def test_empty_survey_vertical_fallback_single_point(self):
        """No survey rows, no max_depth: single point at collar."""
        trace = build_trace(100.0, 200.0, 50.0, [], vertical_if_missing=True, max_depth=None)
        assert len(trace) == 1
        assert trace[0] == (0.0, 100.0, 200.0, 50.0)

    def test_empty_survey_vertical_fallback_with_max_depth(self):
        """No survey rows, max_depth given: collar + point at max_depth (vertical)."""
        trace = build_trace(100.0, 200.0, 50.0, [], vertical_if_missing=True, max_depth=10.0)
        assert len(trace) == 2
        assert trace[0] == (0.0, 100.0, 200.0, 50.0)
        d1, x1, y1, z1 = trace[1]
        assert d1 == 10.0
        assert x1 == 100.0 and y1 == 200.0  # no horizontal
        assert z1 == 50.0 + 10.0 * math.sin(math.radians(VERTICAL_DIP_DEG))  # z - 10

    def test_single_station_straight_line(self):
        """One survey row: straight line from collar to that depth."""
        # Azimuth 0 = North, dip -45 = 45° down from horizontal
        survey_rows = [(10.0, 0.0, -45.0)]  # depth 10, North, 45 down
        trace = build_trace(0.0, 0.0, 0.0, survey_rows)
        assert len(trace) == 2
        assert trace[0] == (0.0, 0.0, 0.0, 0.0)
        d1, x1, y1, z1 = trace[1]
        assert d1 == 10.0
        # North direction: y increases. Horizontal = 10*cos(45°), vertical = 10*sin(-45°)
        horiz = 10.0 * math.cos(math.radians(-45))
        vert = 10.0 * math.sin(math.radians(-45))
        north = horiz * math.cos(0)
        east = horiz * math.sin(0)
        assert abs(x1 - east) < 1e-9 and abs(y1 - north) < 1e-9 and abs(z1 - vert) < 1e-9

    def test_two_stations_minimum_curvature(self):
        """Two survey rows: minimum curvature between stations."""
        survey_rows = [(0.0, 0.0, -30.0), (20.0, 90.0, -60.0)]  # shallow then E, steeper
        trace = build_trace(0.0, 0.0, 0.0, survey_rows)
        # Trace: collar (0) + point at depth 20 (no separate point at depth 0)
        assert len(trace) >= 2
        assert trace[0] == (0.0, 0.0, 0.0, 0.0)
        d2, x2, y2, z2 = trace[-1]
        assert d2 == 20.0
        # Path should have moved (curved from N to E, down)
        assert abs(x2) > 1e-9 or abs(y2) > 1e-9 or abs(z2) < -1e-9

    def test_duplicate_depths_deduped(self):
        """Duplicate depths in survey: keep first, sorted by depth."""
        survey_rows = [(5.0, 0.0, -45.0), (5.0, 90.0, -45.0), (10.0, 0.0, -45.0)]
        trace = build_trace(0.0, 0.0, 0.0, survey_rows)
        depths = [t[0] for t in trace]
        assert depths == sorted(set(depths))
        assert len([d for d in depths if d == 5.0]) == 1

    def test_invalid_rows_dropped(self):
        """Invalid survey rows (NaN, negative depth) are dropped."""
        survey_rows = [
            (5.0, float("nan"), -45.0),
            (10.0, 0.0, -45.0),
            (-1.0, 0.0, -45.0),
        ]
        trace = build_trace(0.0, 0.0, 0.0, survey_rows)
        # Only valid row is (10, 0, -45)
        assert len(trace) == 2
        assert trace[1][0] == 10.0

    def test_dip_clamped(self):
        """Dip outside [-90, 90] is clamped."""
        survey_rows = [(10.0, 0.0, -120.0)]  # clamp to -90
        trace = build_trace(0.0, 0.0, 0.0, survey_rows)
        assert len(trace) == 2
        # Should behave like vertical at -90 (no horizontal, z down)
        d1, x1, y1, z1 = trace[1]
        assert abs(x1) < 1e-9 and abs(y1) < 1e-9
        assert abs(z1 - (0.0 + 10.0 * math.sin(math.radians(-90)))) < 1e-9


class TestXyzAtDepth:
    """Tests for xyz_at_depth."""

    def test_empty_trace_returns_none(self):
        assert xyz_at_depth([], 0.0) is None

    def test_single_point_exact_match(self):
        trace = [(0.0, 1.0, 2.0, 3.0)]
        assert xyz_at_depth(trace, 0.0) == (1.0, 2.0, 3.0)

    def test_single_point_no_match_returns_none(self):
        trace = [(0.0, 1.0, 2.0, 3.0)]
        assert xyz_at_depth(trace, 5.0) is None  # no extrapolation

    def test_interpolate_between_two_points(self):
        trace = [(0.0, 0.0, 0.0, 0.0), (10.0, 10.0, 0.0, -5.0)]
        xyz = xyz_at_depth(trace, 5.0)
        assert xyz is not None
        x, y, z = xyz
        assert abs(x - 5.0) < 1e-9 and abs(y - 0.0) < 1e-9 and abs(z - (-2.5)) < 1e-9

    def test_out_of_range_returns_none(self):
        trace = [(0.0, 0.0, 0.0, 0.0), (10.0, 10.0, 0.0, -5.0)]
        assert xyz_at_depth(trace, -1.0) is None
        assert xyz_at_depth(trace, 11.0) is None

    def test_exact_trace_point_returns_point(self):
        trace = [(0.0, 0.0, 0.0, 0.0), (10.0, 10.0, 0.0, -5.0)]
        assert xyz_at_depth(trace, 10.0) == (10.0, 0.0, -5.0)
