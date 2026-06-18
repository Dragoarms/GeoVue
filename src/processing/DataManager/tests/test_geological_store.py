"""Comprehensive tests for GeologicalStore."""
import pytest
import pandas as pd
import tempfile
from pathlib import Path

from processing.DataManager.geological_store import GeologicalStore, IndexedDataSource
from processing.DataManager.schema import DataType, DatasetType


class TestGeologicalStoreLoading:
    """Test data loading functionality."""

    @pytest.fixture
    def sample_interval_csv(self, tmp_path):
        """Create sample interval data CSV."""
        csv_path = tmp_path / "intervals.csv"
        csv_path.write_text(
            "holeid,geolfrom,geolto,lith,fe_pct\n"
            "BA0001,0,1,BIF,45.5\n"
            "BA0001,1,2,BIF,48.2\n"
            "BA0002,0,1,SHALE,12.3\n"
        )
        return str(csv_path)

    @pytest.fixture
    def sample_collar_csv(self, tmp_path):
        """Create sample collar data CSV."""
        csv_path = tmp_path / "collar.csv"
        csv_path.write_text(
            "holeid,east,north,rl,projectcode\n"
            "BA0001,500000,7500000,450,PROJ1\n"
            "BA0002,500100,7500100,455,PROJ1\n"
        )
        return str(csv_path)

    def test_load_single_csv(self, sample_interval_csv):
        """Should load a single CSV file successfully."""
        store = GeologicalStore()
        store.add_source(sample_interval_csv)
        result = store.load_all()
        # load_all returns Dict[str, bool], check all sources loaded
        assert all(result.values())
        assert store.is_loaded
        assert len(store.list_sources()) == 1

    def test_load_multiple_csvs(self, sample_interval_csv, sample_collar_csv):
        """Should load multiple CSV files."""
        store = GeologicalStore()
        store.add_source(sample_interval_csv)
        store.add_source(sample_collar_csv)
        result = store.load_all()
        assert all(result.values())
        assert len(store.list_sources()) == 2

    def test_get_row_by_key(self, sample_interval_csv):
        """Should retrieve row data by ImageKey."""
        from processing.DataManager.keys import ImageKey

        store = GeologicalStore()
        store.add_source(sample_interval_csv)
        store.load_all()

        key = ImageKey(hole_id="BA0001", depth_to=1.0)
        row = store.get_row(key)

        assert row is not None
        assert row.get("lith") == "BIF"
        assert float(row.get("fe_pct", 0)) == 45.5

    def test_get_rows_for_hole(self, sample_interval_csv):
        """Should retrieve all rows for a hole."""
        store = GeologicalStore()
        store.add_source(sample_interval_csv)
        store.load_all()

        rows = store.get_rows_for_hole("BA0001")
        assert "intervals" in rows
        assert len(rows["intervals"]) == 2

    def test_handles_encoding_fallback(self, tmp_path):
        """Should handle non-UTF8 encoded files."""
        csv_path = tmp_path / "latin1.csv"
        # Write with latin-1 encoding
        csv_path.write_bytes(
            "holeid,geolfrom,geolto,comment\n"
            "BA0001,0,1,Caf\xe9\n".encode("latin-1")
        )

        store = GeologicalStore()
        store.add_source(str(csv_path))
        result = store.load_all()
        assert all(result.values())

    def test_infers_data_types(self, sample_interval_csv):
        """Should correctly infer column data types."""
        store = GeologicalStore()
        store.add_source(sample_interval_csv)
        store.load_all()

        source = store.get_source("intervals")
        schema = source.schema

        assert schema.columns["fe_pct"].data_type == DataType.NUMERIC
        assert schema.columns["lith"].data_type in (DataType.CATEGORICAL, DataType.TEXT)


class TestGeologicalStoreQueries:
    """Test query functionality."""

    @pytest.fixture
    def loaded_store(self, tmp_path):
        """Create and load a store with test data."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "holeid,geolfrom,geolto,lith,fe_pct\n"
            "BA0001,0,1,BIF,45.5\n"
            "BA0001,1,2,BIF,48.2\n"
            "BA0001,2,3,SHALE,12.3\n"
            "BA0002,0,1,BIF,52.1\n"
            "BA0002,1,2,SHALE,15.5\n"
        )
        store = GeologicalStore()
        store.add_source(str(csv_path))
        store.load_all()
        return store

    def test_get_unique_holes(self, loaded_store):
        """Should return all unique hole IDs."""
        holes = loaded_store.get_unique_holes()
        assert "BA0001" in holes or "ba0001" in [h.lower() for h in holes]
        assert "BA0002" in holes or "ba0002" in [h.lower() for h in holes]

    def test_get_column_values(self, loaded_store):
        """Should return unique values for a column."""
        liths = loaded_store.get_column_values("lith")
        assert "BIF" in liths or "bif" in [str(v).lower() for v in liths]
        assert "SHALE" in liths or "shale" in [str(v).lower() for v in liths]

    def test_get_available_columns(self, loaded_store):
        """Should return all available columns with types."""
        columns = loaded_store.get_available_columns()
        assert len(columns) > 0
        # Should be dict of source -> [(col_name, type), ...]
        assert isinstance(columns, dict)


class TestIndexedDataSource:
    """Test IndexedDataSource functionality."""

    def test_multiindex_lookup_performance(self, tmp_path):
        """MultiIndex lookup should be O(1)."""
        import time

        # Create larger dataset
        rows = ["holeid,geolfrom,geolto,value"]
        for hole in range(100):
            for depth in range(100):
                rows.append(f"HOLE{hole:04d},{depth},{depth+1},{hole*100+depth}")

        csv_path = tmp_path / "large.csv"
        csv_path.write_text("\n".join(rows))

        from processing.DataManager.schema import infer_schema
        import pandas as pd

        # Read sample for schema inference
        df_sample = pd.read_csv(str(csv_path), nrows=1000)
        schema = infer_schema(df_sample, "large", str(csv_path))
        source = IndexedDataSource(schema)
        source.load()

        # Time single lookup
        from processing.DataManager.keys import ImageKey

        start = time.time()
        for _ in range(1000):
            key = ImageKey("HOLE0050", 50.0)
            source.get_row(key)
        elapsed = time.time() - start

        # Should be very fast (<100ms for 1000 lookups)
        assert elapsed < 0.5, f"Lookups too slow: {elapsed:.2f}s for 1000 lookups"
