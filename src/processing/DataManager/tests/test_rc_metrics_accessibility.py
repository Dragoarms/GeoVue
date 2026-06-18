"""Tests for RC metrics accessibility in DataCoordinator."""
import pytest
import pandas as pd
from pathlib import Path


class TestRCMetricsAccessibility:
    """Test that RC metrics are accessible as columns."""

    @pytest.fixture
    def mineral_csv(self, tmp_path):
        """Create CSV with mineral percentage columns."""
        csv_path = tmp_path / "exgeologyRC.csv"
        csv_path.write_text(
            "holeid,sampfrom,sampto,min_80_pct,min_10_pct,min_5_pct,min_5_pct2\n"
            "BA0001,0,1,HEM,GOE,QZ,\n"
            "BA0001,1,2,HEM,HEM,MAR,QZ\n"
        )
        return str(csv_path)

    def test_rc_metrics_in_available_columns(self, mineral_csv):
        """RC metrics should appear in available columns list."""
        from processing.DataManager.data_coordinator import DataCoordinator

        coordinator = DataCoordinator()
        coordinator.initialize(
            compartment_folders=[],
            csv_files=[mineral_csv],
        )

        columns = coordinator.get_available_columns()
        column_names = [c[0] for c in columns]

        # RC metric columns should be available
        assert "weighted_hardness" in column_names or any("hardness" in c.lower() for c in column_names)
        assert "total_gangue_pct" in column_names or any("gangue" in c.lower() for c in column_names)

    def test_rc_metrics_in_hole_data(self, mineral_csv):
        """RC metrics should be included when getting hole data."""
        from processing.DataManager.data_coordinator import DataCoordinator

        coordinator = DataCoordinator()
        coordinator.initialize(
            compartment_folders=[],
            csv_files=[mineral_csv],
        )

        # Get metrics for specific interval
        metrics = coordinator.get_rc_metrics("BA0001", 1.0)

        assert metrics is not None
        assert hasattr(metrics, "weighted_hardness")
        assert hasattr(metrics, "total_gangue_pct")

    def test_rc_metrics_columns_have_correct_type(self, mineral_csv):
        """RC metric columns should be marked as NUMERIC type."""
        from processing.DataManager.data_coordinator import DataCoordinator
        from processing.DataManager.schema import DataType

        coordinator = DataCoordinator()
        coordinator.initialize(
            compartment_folders=[],
            csv_files=[mineral_csv],
        )

        columns = coordinator.get_available_columns()
        columns_dict = {name: dtype for name, dtype in columns}

        # Check that RC metric columns are NUMERIC
        if "weighted_hardness" in columns_dict:
            assert columns_dict["weighted_hardness"] == DataType.NUMERIC
        if "total_gangue_pct" in columns_dict:
            assert columns_dict["total_gangue_pct"] == DataType.NUMERIC

    def test_no_duplicate_columns(self, mineral_csv):
        """Available columns should not have duplicates."""
        from processing.DataManager.data_coordinator import DataCoordinator

        coordinator = DataCoordinator()
        coordinator.initialize(
            compartment_folders=[],
            csv_files=[mineral_csv],
        )

        columns = coordinator.get_available_columns()
        column_names = [c[0] for c in columns]

        # Check for duplicates
        assert len(column_names) == len(set(column_names)), "Duplicate columns found"

    def test_has_rc_metrics_property(self, mineral_csv):
        """DataCoordinator should report has_rc_metrics correctly."""
        from processing.DataManager.data_coordinator import DataCoordinator

        coordinator = DataCoordinator()
        coordinator.initialize(
            compartment_folders=[],
            csv_files=[mineral_csv],
        )

        # Should have RC metrics after loading CSV with mineral columns
        assert coordinator.has_rc_metrics is True

    def test_no_rc_metrics_without_mineral_columns(self, tmp_path):
        """DataCoordinator should not have RC metrics without mineral columns."""
        from processing.DataManager.data_coordinator import DataCoordinator

        # Create CSV without mineral columns
        csv_path = tmp_path / "simple.csv"
        csv_path.write_text(
            "holeid,sampfrom,sampto,lithology\n"
            "BA0001,0,1,BIF\n"
            "BA0001,1,2,SHALE\n"
        )

        coordinator = DataCoordinator()
        coordinator.initialize(
            compartment_folders=[],
            csv_files=[str(csv_path)],
        )

        # Should not have RC metrics
        assert coordinator.has_rc_metrics is False

        # RC metric columns should not appear
        columns = coordinator.get_available_columns()
        column_names = [c[0] for c in columns]
        assert "weighted_hardness" not in column_names
        assert "total_gangue_pct" not in column_names
