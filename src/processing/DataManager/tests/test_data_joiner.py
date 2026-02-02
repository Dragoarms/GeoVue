"""Tests for cross-table data joining with interval overlap detection."""
import pytest
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Will be implemented
# from processing.DataManager.data_joiner import DataJoiner, JoinResult, IntervalMatch


class TestDataJoiner:
    """Test cross-table joining with interval overlap semantics."""

    @pytest.fixture
    def geology_df(self):
        """1m interval geology data."""
        return pd.DataFrame({
            "holeid": ["BA0001", "BA0001", "BA0001", "BA0002", "BA0002"],
            "geolfrom": [0, 1, 2, 0, 1],
            "geolto": [1, 2, 3, 1, 2],
            "lith": ["BIF", "BIF", "SHALE", "BIF", "BIF"],
            "hardness": [3, 4, 2, 3, 3],
        })

    @pytest.fixture
    def assay_df(self):
        """0.5m interval assay data (finer resolution)."""
        return pd.DataFrame({
            "holeid": ["BA0001", "BA0001", "BA0001", "BA0001", "BA0001", "BA0001"],
            "sampfrom": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
            "sampto": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            "fe_pct": [45.0, 46.5, 48.0, 49.2, 35.0, 32.0],
            "sio2_pct": [12.0, 11.5, 10.0, 9.5, 25.0, 28.0],
        })

    @pytest.fixture
    def sparse_assay_df(self):
        """Assay data with gaps (missing intervals)."""
        return pd.DataFrame({
            "holeid": ["BA0001", "BA0001", "BA0001"],  # Missing 1-2m
            "sampfrom": [0.0, 0.5, 2.0],
            "sampto": [0.5, 1.0, 2.5],
            "fe_pct": [45.0, 46.5, 35.0],
        })

    def test_join_preserves_all_primary_rows(self, geology_df, sparse_assay_df):
        """LEFT JOIN: all geology rows preserved even without matching assay."""
        from processing.DataManager.data_joiner import DataJoiner

        joiner = DataJoiner()
        result = joiner.join(
            primary_df=geology_df,
            secondary_df=sparse_assay_df,
            primary_key_cols={"hole": "holeid", "from": "geolfrom", "to": "geolto"},
            secondary_key_cols={"hole": "holeid", "from": "sampfrom", "to": "sampto"},
            aggregate_secondary=True,  # Use aggregation to get one row per primary
        )

        # All 5 geology rows should be present (with aggregation, one row per primary)
        assert len(result.joined_df) == 5, "All primary rows must be preserved"

        # BA0001 1-2m should have NaN for assay (no matching assay data)
        ba0001_1_2 = result.joined_df[
            (result.joined_df["holeid"] == "BA0001") &
            (result.joined_df["geolfrom"] == 1)
        ]
        assert len(ba0001_1_2) == 1
        assert pd.isna(ba0001_1_2["fe_pct_mean"].iloc[0]), "Missing assay should be NaN"

    def test_join_detects_interval_overlaps(self, geology_df, assay_df):
        """Overlapping intervals should be detected and matched."""
        from processing.DataManager.data_joiner import DataJoiner

        joiner = DataJoiner()
        result = joiner.join(
            primary_df=geology_df,
            secondary_df=assay_df,
            primary_key_cols={"hole": "holeid", "from": "geolfrom", "to": "geolto"},
            secondary_key_cols={"hole": "holeid", "from": "sampfrom", "to": "sampto"},
        )

        # Geology 0-1m should match assay 0-0.5m and 0.5-1m
        matches = result.get_matches_for_interval("BA0001", 0, 1)
        assert len(matches) == 2, "1m geology should match 2x 0.5m assay intervals"

        # Check overlap percentages
        assert matches[0].overlap_pct > 0.49  # ~50% overlap
        assert matches[1].overlap_pct > 0.49

    def test_join_reports_coverage_percentage(self, geology_df, sparse_assay_df):
        """Result should report how much of primary is covered by secondary."""
        from processing.DataManager.data_joiner import DataJoiner

        joiner = DataJoiner()
        result = joiner.join(
            primary_df=geology_df,
            secondary_df=sparse_assay_df,
            primary_key_cols={"hole": "holeid", "from": "geolfrom", "to": "geolto"},
            secondary_key_cols={"hole": "holeid", "from": "sampfrom", "to": "sampto"},
        )

        # BA0001: 0-1m has assay, 1-2m missing, 2-3m partial
        coverage = result.get_coverage_for_hole("BA0001")
        assert 0 < coverage < 1.0, "Partial coverage expected"

        # BA0002: no assay data at all
        coverage_ba2 = result.get_coverage_for_hole("BA0002")
        assert coverage_ba2 == 0.0, "No coverage for BA0002"

    def test_join_aggregates_secondary_to_primary_intervals(self, geology_df, assay_df):
        """When secondary has finer intervals, aggregate to primary."""
        from processing.DataManager.data_joiner import DataJoiner

        joiner = DataJoiner()
        result = joiner.join(
            primary_df=geology_df,
            secondary_df=assay_df,
            primary_key_cols={"hole": "holeid", "from": "geolfrom", "to": "geolto"},
            secondary_key_cols={"hole": "holeid", "from": "sampfrom", "to": "sampto"},
            aggregate_secondary=True,  # Average overlapping assay values
        )

        # Geology 0-1m with aggregated assay (average of 45.0 and 46.5)
        row = result.joined_df[
            (result.joined_df["holeid"] == "BA0001") &
            (result.joined_df["geolfrom"] == 0)
        ].iloc[0]

        expected_fe = (45.0 + 46.5) / 2  # 45.75
        assert abs(row["fe_pct_mean"] - expected_fe) < 0.01

    def test_join_logs_informative_debug_info(self, geology_df, assay_df, caplog):
        """Join operation should produce clear debug logs."""
        import logging
        from processing.DataManager.data_joiner import DataJoiner

        with caplog.at_level(logging.DEBUG):
            joiner = DataJoiner()
            result = joiner.join(
                primary_df=geology_df,
                secondary_df=assay_df,
                primary_key_cols={"hole": "holeid", "from": "geolfrom", "to": "geolto"},
                secondary_key_cols={"hole": "holeid", "from": "sampfrom", "to": "sampto"},
            )

        # Should log structure info
        assert "primary rows: 5" in caplog.text.lower() or "5 primary" in caplog.text.lower()
        assert "secondary rows: 6" in caplog.text.lower() or "6 secondary" in caplog.text.lower()
        assert "overlap" in caplog.text.lower() or "match" in caplog.text.lower()

    def test_join_handles_different_hole_id_columns(self, geology_df):
        """Should work when hole column has different names."""
        from processing.DataManager.data_joiner import DataJoiner

        # Create assay with different column name
        assay_df = pd.DataFrame({
            "bhid": ["BA0001", "BA0001"],  # Different name
            "from_m": [0.0, 0.5],
            "to_m": [0.5, 1.0],
            "fe": [45.0, 46.5],
        })

        joiner = DataJoiner()
        result = joiner.join(
            primary_df=geology_df,
            secondary_df=assay_df,
            primary_key_cols={"hole": "holeid", "from": "geolfrom", "to": "geolto"},
            secondary_key_cols={"hole": "bhid", "from": "from_m", "to": "to_m"},
            aggregate_secondary=True,  # Use aggregation to get one row per primary
        )

        # Should still match BA0001 rows (with aggregation, one row per primary)
        assert len(result.joined_df) == 5
        ba0001_matches = result.joined_df[
            (result.joined_df["holeid"] == "BA0001") &
            (result.joined_df["geolfrom"] == 0)
        ]
        assert not pd.isna(ba0001_matches["fe_mean"].iloc[0])


class TestDataCoordinatorJoin:
    """Test DataCoordinator join integration."""

    @pytest.fixture
    def temp_csvs(self, tmp_path):
        """Create temp CSV files for testing."""
        geology_csv = tmp_path / "exgeologyRC.csv"
        geology_csv.write_text(
            "holeid,geolfrom,geolto,lith\n"
            "BA0001,0,1,BIF\n"
            "BA0001,1,2,BIF\n"
        )

        assay_csv = tmp_path / "exassay.csv"
        assay_csv.write_text(
            "holeid,sampfrom,sampto,fe_pct\n"
            "BA0001,0.0,0.5,45.0\n"
            "BA0001,0.5,1.0,46.5\n"
            "BA0001,1.0,1.5,48.0\n"
            "BA0001,1.5,2.0,49.2\n"
        )

        return {"geology": str(geology_csv), "assay": str(assay_csv)}

    def test_coordinator_join_sources(self, temp_csvs):
        """DataCoordinator should provide join API."""
        from processing.DataManager.data_coordinator import DataCoordinator

        coordinator = DataCoordinator()
        coordinator.initialize(
            compartment_folders=[],
            csv_files=[temp_csvs["geology"], temp_csvs["assay"]],
        )

        result = coordinator.join_sources(
            primary="exgeologyRC",
            secondary="exassay",
            aggregate=True,
        )

        assert result is not None
        assert len(result.joined_df) == 2  # 2 geology rows
        assert "fe_pct_mean" in result.joined_df.columns
