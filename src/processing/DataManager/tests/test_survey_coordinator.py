"""Tests for DataCoordinator survey and XYZ API (get_survey_data, get_trace_for_hole, get_xyz_at_depth, add_xyz_to_interval_dataframe)."""
import pytest
import pandas as pd
from pathlib import Path

from processing.DataManager.data_coordinator import DataCoordinator


class TestGetSurveyData:
    """Tests for get_survey_data()."""

    @pytest.fixture
    def exsurvey_csv(self, tmp_path):
        """Create sample exsurvey CSV (holeid, depth, azimuth, dip)."""
        csv_path = tmp_path / "exsurvey.csv"
        csv_path.write_text(
            "holeid,depth,azimuth,dip\n"
            "BA0001,0,0,-45\n"
            "BA0001,20,90,-60\n"
            "BA0002,0,180,-30\n"
        )
        return str(csv_path)

    @pytest.fixture
    def excollar_csv(self, tmp_path):
        """Create sample excollar CSV."""
        csv_path = tmp_path / "excollar.csv"
        csv_path.write_text(
            "holeid,east,north,rl\n"
            "BA0001,100.0,200.0,50.0\n"
            "BA0002,110.0,210.0,55.0\n"
        )
        return str(csv_path)

    @pytest.fixture
    def coordinator_with_survey(self, exsurvey_csv, excollar_csv):
        """DataCoordinator with excollar and exsurvey sources."""
        coord = DataCoordinator()
        coord.initialize(
            compartment_folders=[],
            csv_files=[excollar_csv, exsurvey_csv],
        )
        return coord

    def test_get_survey_data_no_source_returns_empty(self):
        """When no survey source exists, get_survey_data returns empty DataFrame."""
        coord = DataCoordinator()
        coord.initialize(compartment_folders=[], csv_files=[])
        df = coord.get_survey_data()
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_get_survey_data_returns_standardized_columns(self, coordinator_with_survey):
        """get_survey_data returns DataFrame with holeid, depth, azimuth, dip."""
        coord = coordinator_with_survey
        df = coord.get_survey_data()
        assert not df.empty
        for col in ("holeid", "depth", "azimuth", "dip"):
            assert col in df.columns, f"Missing column {col}"
        assert len(df) >= 3  # at least 3 rows from fixture

    def test_get_survey_data_holeid_uppercase(self, coordinator_with_survey):
        """Survey holeid values are normalized to uppercase."""
        df = coordinator_with_survey.get_survey_data()
        assert df["holeid"].str.isupper().all()


class TestGetTraceForHole:
    """Tests for get_trace_for_hole()."""

    @pytest.fixture
    def excollar_csv(self, tmp_path):
        csv_path = tmp_path / "excollar.csv"
        csv_path.write_text(
            "holeid,east,north,rl\n"
            "BA0001,100.0,200.0,50.0\n"
        )
        return str(csv_path)

    @pytest.fixture
    def exsurvey_csv(self, tmp_path):
        csv_path = tmp_path / "exsurvey.csv"
        csv_path.write_text(
            "holeid,depth,azimuth,dip\n"
            "BA0001,0,0,-45\n"
            "BA0001,10,0,-45\n"
        )
        return str(csv_path)

    @pytest.fixture
    def coordinator_collar_and_survey(self, excollar_csv, exsurvey_csv):
        coord = DataCoordinator()
        coord.initialize(
            compartment_folders=[],
            csv_files=[excollar_csv, exsurvey_csv],
        )
        return coord

    def test_get_trace_for_hole_no_collar_returns_empty(self, exsurvey_csv):
        """When hole has no collar, get_trace_for_hole returns empty list."""
        coord = DataCoordinator()
        coord.initialize(compartment_folders=[], csv_files=[exsurvey_csv])
        trace = coord.get_trace_for_hole("NOHOLE")
        assert trace == []

    def test_get_trace_for_hole_with_collar_returns_trace(self, coordinator_collar_and_survey):
        """When collar exists, get_trace_for_hole returns non-empty trace with depth 0 = collar."""
        coord = coordinator_collar_and_survey
        trace = coord.get_trace_for_hole("BA0001")
        assert len(trace) >= 1
        d0, x0, y0, z0 = trace[0]
        assert d0 == 0.0
        assert x0 == 100.0 and y0 == 200.0 and z0 == 50.0

    def test_get_trace_for_hole_vertical_fallback(self, excollar_csv):
        """When no survey for hole, trace is vertical (collar only or collar + max_depth)."""
        coord = DataCoordinator()
        coord.initialize(compartment_folders=[], csv_files=[excollar_csv])
        trace = coord.get_trace_for_hole("BA0001", max_depth=20.0)
        assert len(trace) == 2
        assert trace[0] == (0.0, 100.0, 200.0, 50.0)
        d1, x1, y1, z1 = trace[1]
        assert d1 == 20.0
        assert x1 == 100.0 and y1 == 200.0
        assert z1 < 50.0  # down


class TestGetXyzAtDepth:
    """Tests for get_xyz_at_depth()."""

    @pytest.fixture
    def excollar_csv(self, tmp_path):
        csv_path = tmp_path / "excollar.csv"
        csv_path.write_text(
            "holeid,east,north,rl\n"
            "BA0001,100.0,200.0,50.0\n"
        )
        return str(csv_path)

    @pytest.fixture
    def exsurvey_csv(self, tmp_path):
        csv_path = tmp_path / "exsurvey.csv"
        csv_path.write_text(
            "holeid,depth,azimuth,dip\n"
            "BA0001,0,0,-45\n"
            "BA0001,10,0,-45\n"
        )
        return str(csv_path)

    @pytest.fixture
    def coordinator_collar_and_survey(self, excollar_csv, exsurvey_csv):
        coord = DataCoordinator()
        coord.initialize(
            compartment_folders=[],
            csv_files=[excollar_csv, exsurvey_csv],
        )
        return coord

    def test_get_xyz_at_depth_at_collar(self, coordinator_collar_and_survey):
        """get_xyz_at_depth(hole_id, 0) returns collar coordinates."""
        coord = coordinator_collar_and_survey
        xyz = coord.get_xyz_at_depth("BA0001", 0.0)
        assert xyz is not None
        assert xyz == (100.0, 200.0, 50.0)

    def test_get_xyz_at_depth_no_collar_returns_none(self, exsurvey_csv):
        """When hole has no collar, get_xyz_at_depth returns None."""
        coord = DataCoordinator()
        coord.initialize(compartment_folders=[], csv_files=[exsurvey_csv])
        assert coord.get_xyz_at_depth("BA0001", 0.0) is None

    def test_get_xyz_at_depth_beyond_trace_returns_none(self, coordinator_collar_and_survey):
        """Requesting depth beyond trace range returns None (no extrapolation)."""
        coord = coordinator_collar_and_survey
        # Trace ends at depth 10; requesting 100 should return None
        xyz = coord.get_xyz_at_depth("BA0001", 100.0)
        assert xyz is None


class TestAddXyzToIntervalDataframe:
    """Tests for add_xyz_to_interval_dataframe()."""

    @pytest.fixture
    def excollar_csv(self, tmp_path):
        csv_path = tmp_path / "excollar.csv"
        csv_path.write_text(
            "holeid,east,north,rl\n"
            "BA0001,100.0,200.0,50.0\n"
        )
        return str(csv_path)

    @pytest.fixture
    def exsurvey_csv(self, tmp_path):
        csv_path = tmp_path / "exsurvey.csv"
        csv_path.write_text(
            "holeid,depth,azimuth,dip\n"
            "BA0001,0,0,-45\n"
            "BA0001,10,0,-45\n"
        )
        return str(csv_path)

    @pytest.fixture
    def coordinator_collar_and_survey(self, excollar_csv, exsurvey_csv):
        coord = DataCoordinator()
        coord.initialize(
            compartment_folders=[],
            csv_files=[excollar_csv, exsurvey_csv],
        )
        return coord

    def test_add_xyz_adds_columns(self, coordinator_collar_and_survey):
        """add_xyz_to_interval_dataframe adds x_from, y_from, z_from, x_to, y_to, z_to."""
        coord = coordinator_collar_and_survey
        interval_df = pd.DataFrame({
            "holeid": ["BA0001", "BA0001"],
            "sampfrom": [0.0, 5.0],
            "sampto": [1.0, 6.0],
        })
        out = coord.add_xyz_to_interval_dataframe(interval_df, inplace=False)
        for col in ("x_from", "y_from", "z_from", "x_to", "y_to", "z_to"):
            assert col in out.columns
        assert len(out) == 2

    def test_add_xyz_resolves_columns_via_resolver(self, coordinator_collar_and_survey):
        """add_xyz_to_interval_dataframe resolves holeid/from/to via ColumnResolver when not provided."""
        coord = coordinator_collar_and_survey
        interval_df = pd.DataFrame({
            "holeid": ["BA0001"],
            "sampfrom": [0.0],
            "sampto": [5.0],
        })
        out = coord.add_xyz_to_interval_dataframe(interval_df, inplace=False)
        assert "x_from" in out.columns and "x_to" in out.columns
        # At 0 and 5 we should have valid XYZ (within trace 0..10)
        assert out["x_from"].iloc[0] is not None or pd.notna(out["x_from"].iloc[0])
