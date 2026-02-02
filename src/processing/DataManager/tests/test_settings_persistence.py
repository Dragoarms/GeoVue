"""Tests for column settings persistence across sessions."""
import pytest
import tempfile
import os
from pathlib import Path

from processing.DataManager.geological_store import GeologicalStore
from processing.DataManager.schema import DataType, NullHandling


class MockConfigManager:
    """Mock ConfigManager for testing persistence."""

    USER_SETTINGS_KEYS = ["column_schemas", "register_column_settings"]

    def __init__(self):
        self._data = {}

    def get(self, key, default=None):
        return self._data.get(key, default)

    def set(self, key, value):
        self._data[key] = value


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV for testing."""
    csv_path = tmp_path / "test_data.csv"
    csv_path.write_text(
        "holeid,sampfrom,sampto,fe_pct,sio2_pct\n"
        "BA0001,0,1,45.5,12.3\n"
        "BA0001,1,2,48.2,10.1\n"
        "BA0002,0,1,52.1,8.5\n"
    )
    return str(csv_path)


class TestColumnSettingsPersistence:
    """Test that column settings persist across GeologicalStore reloads."""

    def test_color_map_persists_across_reload(self, sample_csv):
        """Color map assignment should survive store reload."""
        config = MockConfigManager()

        # First session: load and customize
        store1 = GeologicalStore(config)
        store1.add_source(sample_csv)
        store1.load_all()

        # Modify column settings
        source = store1.get_source("test_data")
        assert source is not None
        schema = source.schema
        schema.columns["fe_pct"].color_map = "fe_grade"
        schema.columns["fe_pct"].display_name = "Iron %"
        schema.columns["fe_pct"].decimals = 1

        # Save to config
        store1.save_schemas_to_config()

        # Second session: reload and verify
        store2 = GeologicalStore(config)
        store2.add_source(sample_csv)
        store2.load_all()

        source2 = store2.get_source("test_data")
        assert source2 is not None

        # Verify settings persisted
        fe_col = source2.schema.columns["fe_pct"]
        assert fe_col.color_map == "fe_grade", "Color map should persist"
        assert fe_col.display_name == "Iron %", "Display name should persist"
        assert fe_col.decimals == 1, "Decimals should persist"

    def test_visibility_persists_across_reload(self, sample_csv):
        """Column visibility should survive store reload."""
        config = MockConfigManager()

        # First session
        store1 = GeologicalStore(config)
        store1.add_source(sample_csv)
        store1.load_all()

        source = store1.get_source("test_data")
        schema = source.schema
        schema.columns["sio2_pct"].is_visible = False

        store1.save_schemas_to_config()

        # Second session
        store2 = GeologicalStore(config)
        store2.add_source(sample_csv)
        store2.load_all()

        source2 = store2.get_source("test_data")
        assert source2.schema.columns["sio2_pct"].is_visible == False

    def test_data_type_override_persists(self, sample_csv):
        """User data type overrides should persist."""
        config = MockConfigManager()

        # First session - change inferred type
        store1 = GeologicalStore(config)
        store1.add_source(sample_csv)
        store1.load_all()

        source = store1.get_source("test_data")
        # holeid might be inferred as text, user changes to categorical
        schema = source.schema
        schema.columns["holeid"].data_type = DataType.CATEGORICAL

        store1.save_schemas_to_config()

        # Second session
        store2 = GeologicalStore(config)
        store2.add_source(sample_csv)
        store2.load_all()

        source2 = store2.get_source("test_data")
        assert source2.schema.columns["holeid"].data_type == DataType.CATEGORICAL
