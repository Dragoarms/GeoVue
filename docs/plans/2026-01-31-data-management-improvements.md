# Data Management Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a robust, testable data management layer with persistent column settings, cross-table joining with interval overlap detection, accessible RC metrics, and proper UI feedback for data quality issues.

**Architecture:**
- DataCoordinator remains the single entry point for all data operations
- New `DataJoiner` class handles cross-table operations with LEFT JOIN semantics and interval overlap detection
- GeologicalStore enhanced to load/save column schemas from ConfigManager on initialization
- ColumnSettingsDialog fixed to use grid layout for proper alignment and surface sanitization warnings

**Tech Stack:** Python 3.11+, pandas, tkinter, pytest

---

## Phase 1: Settings Persistence Fix

### Task 1.1: Write Failing Test for Schema Persistence Round-Trip

**Files:**
- Create: `src/processing/DataManager/tests/test_settings_persistence.py`

**Step 1: Write the failing test**

```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest src/processing/DataManager/tests/test_settings_persistence.py -v`
Expected: FAIL with `AttributeError: 'GeologicalStore' object has no attribute 'save_schemas_to_config'`

**Step 3: Commit the failing test**

```bash
git add src/processing/DataManager/tests/test_settings_persistence.py
git commit -m "test: add failing tests for column settings persistence"
```

---

### Task 1.2: Implement Schema Save Method in GeologicalStore

**Files:**
- Modify: `src/processing/DataManager/geological_store.py`

**Step 1: Add save_schemas_to_config method**

Add after line ~400 (after `get_data_sources` method):

```python
def save_schemas_to_config(self) -> bool:
    """
    Save all source schemas to ConfigManager for persistence.

    Serializes column settings (display_name, color_map, decimals,
    is_visible, data_type, null_handling) so they survive application restart.

    Returns:
        True if successful, False if no config_manager
    """
    if not self._config_manager:
        logger.warning("Cannot save schemas: no ConfigManager configured")
        return False

    schemas_dict = {}
    for source_name, indexed_source in self._sources.items():
        schema = indexed_source.schema
        schemas_dict[source_name] = schema.to_dict()

    self._config_manager.set("column_schemas", schemas_dict)
    logger.info(f"Saved column schemas for {len(schemas_dict)} sources to config")
    return True
```

**Step 2: Run test - should still fail**

Run: `pytest src/processing/DataManager/tests/test_settings_persistence.py::TestColumnSettingsPersistence::test_color_map_persists_across_reload -v`
Expected: FAIL - save works but load doesn't restore settings

**Step 3: Commit partial implementation**

```bash
git add src/processing/DataManager/geological_store.py
git commit -m "feat: add save_schemas_to_config to GeologicalStore"
```

---

### Task 1.3: Implement Schema Load on Initialization

**Files:**
- Modify: `src/processing/DataManager/geological_store.py`

**Step 1: Add _load_saved_schemas method and call it during load_all**

Add method after `save_schemas_to_config`:

```python
def _load_saved_schemas(self) -> Dict[str, dict]:
    """
    Load previously saved schemas from ConfigManager.

    Returns:
        Dictionary of {source_name: schema_dict}
    """
    if not self._config_manager:
        return {}

    saved = self._config_manager.get("column_schemas", {})
    if saved:
        logger.info(f"Found saved schemas for {len(saved)} sources")
    return saved

def _apply_saved_schema(self, source_name: str, schema: 'DataSourceSchema', saved_schemas: Dict[str, dict]) -> None:
    """
    Apply saved column settings to a schema.

    Preserves user customizations (display_name, color_map, decimals,
    is_visible, data_type override) while keeping inferred structure.

    Args:
        source_name: Name of the data source
        schema: The freshly inferred schema to modify
        saved_schemas: Dictionary of saved schema data
    """
    if source_name not in saved_schemas:
        logger.debug(f"No saved schema for '{source_name}'")
        return

    saved = saved_schemas[source_name]
    saved_columns = saved.get("columns", {})

    applied_count = 0
    for col_name, col_schema in schema.columns.items():
        col_lower = col_name.lower()

        # Find matching saved column (case-insensitive)
        saved_col = None
        for saved_name, saved_data in saved_columns.items():
            if saved_name.lower() == col_lower:
                saved_col = saved_data
                break

        if not saved_col:
            continue

        # Apply saved settings
        if "display_name" in saved_col and saved_col["display_name"]:
            col_schema.display_name = saved_col["display_name"]

        if "color_map" in saved_col and saved_col["color_map"]:
            col_schema.color_map = saved_col["color_map"]

        if "decimals" in saved_col:
            col_schema.decimals = saved_col["decimals"]

        if "is_visible" in saved_col:
            col_schema.is_visible = saved_col["is_visible"]

        if "data_type" in saved_col:
            from .schema import DataType
            col_schema.data_type = DataType.from_string(saved_col["data_type"])

        if "null_handling" in saved_col:
            from .schema import NullHandling
            try:
                col_schema.null_handling = NullHandling(saved_col["null_handling"])
            except ValueError:
                pass  # Keep default if invalid

        applied_count += 1

    if applied_count > 0:
        logger.info(f"Applied saved settings to {applied_count} columns in '{source_name}'")
```

**Step 2: Modify load_all to use saved schemas**

Find the `load_all` method and add schema restoration after loading each source. Modify around line ~320:

```python
def load_all(self) -> bool:
    """
    Load all registered data sources.

    Returns:
        True if all sources loaded successfully
    """
    if not self._pending_sources:
        logger.warning("No data sources registered to load")
        return True

    # Load saved schemas BEFORE loading sources
    saved_schemas = self._load_saved_schemas()

    start_time = time.time()
    success_count = 0

    for file_path in self._pending_sources:
        source_name = Path(file_path).stem

        try:
            # Infer schema
            schema = infer_schema(file_path)
            if schema is None:
                logger.error(f"Could not infer schema for {file_path}")
                continue

            # Apply saved customizations BEFORE creating IndexedDataSource
            self._apply_saved_schema(source_name, schema, saved_schemas)

            # Create and load source
            source = IndexedDataSource(schema)
            if source.load():
                self._sources[source_name] = source
                success_count += 1
            else:
                logger.error(f"Failed to load source: {source_name}")

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    self._is_loaded = success_count > 0
    load_time = time.time() - start_time

    logger.info("=" * 60)
    logger.info(f"LOADING COMPLETE: {success_count}/{len(self._pending_sources)} sources loaded successfully")
    logger.info("=" * 60)

    return self._is_loaded
```

**Step 3: Run tests to verify they pass**

Run: `pytest src/processing/DataManager/tests/test_settings_persistence.py -v`
Expected: All 3 tests PASS

**Step 4: Commit implementation**

```bash
git add src/processing/DataManager/geological_store.py
git commit -m "feat: load saved column schemas on GeologicalStore initialization"
```

---

## Phase 2: Cross-Table Data Joining with Interval Overlap

### Task 2.1: Write Failing Tests for Data Joiner

**Files:**
- Create: `src/processing/DataManager/tests/test_data_joiner.py`

**Step 1: Write the failing tests**

```python
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
        )

        # All 5 geology rows should be present
        assert len(result.joined_df) == 5, "All primary rows must be preserved"

        # BA0001 1-2m should have NaN for assay (no matching assay data)
        ba0001_1_2 = result.joined_df[
            (result.joined_df["holeid"] == "BA0001") &
            (result.joined_df["geolfrom"] == 1)
        ]
        assert len(ba0001_1_2) == 1
        assert pd.isna(ba0001_1_2["fe_pct"].iloc[0]), "Missing assay should be NaN"

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
        )

        # Should still match BA0001 rows
        assert len(result.joined_df) == 5
        ba0001_matches = result.joined_df[
            (result.joined_df["holeid"] == "BA0001") &
            (result.joined_df["geolfrom"] == 0)
        ]
        assert not pd.isna(ba0001_matches["fe"].iloc[0])
```

**Step 2: Run test to verify it fails**

Run: `pytest src/processing/DataManager/tests/test_data_joiner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'processing.DataManager.data_joiner'`

**Step 3: Commit the failing tests**

```bash
git add src/processing/DataManager/tests/test_data_joiner.py
git commit -m "test: add failing tests for cross-table data joining"
```

---

### Task 2.2: Implement DataJoiner Core Classes

**Files:**
- Create: `src/processing/DataManager/data_joiner.py`

**Step 1: Create the data_joiner module**

```python
"""
data_joiner.py - Cross-table data joining with interval overlap detection.

Provides LEFT JOIN semantics where:
- All primary (left) rows are preserved
- Secondary (right) rows are matched by hole_id and interval overlap
- Missing secondary data results in NaN (not row loss)
- Overlap percentage is tracked for each match

Author: George Symonds
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IntervalMatch:
    """
    Represents a match between primary and secondary intervals.

    Attributes:
        primary_idx: Index in primary DataFrame
        secondary_idx: Index in secondary DataFrame (None if no match)
        hole_id: Hole identifier
        primary_from: Primary interval start
        primary_to: Primary interval end
        secondary_from: Secondary interval start (if matched)
        secondary_to: Secondary interval end (if matched)
        overlap_from: Start of overlap region
        overlap_to: End of overlap region
        overlap_pct: Percentage of primary interval covered by this match
    """
    primary_idx: int
    secondary_idx: Optional[int]
    hole_id: str
    primary_from: float
    primary_to: float
    secondary_from: Optional[float] = None
    secondary_to: Optional[float] = None
    overlap_from: Optional[float] = None
    overlap_to: Optional[float] = None
    overlap_pct: float = 0.0

    @property
    def has_match(self) -> bool:
        return self.secondary_idx is not None

    @property
    def overlap_length(self) -> float:
        if self.overlap_from is None or self.overlap_to is None:
            return 0.0
        return self.overlap_to - self.overlap_from


@dataclass
class JoinResult:
    """
    Result of a join operation with metadata about matches.

    Attributes:
        joined_df: The joined DataFrame (LEFT JOIN result)
        primary_source: Name of primary data source
        secondary_source: Name of secondary data source
        matches: List of all interval matches found
        unmatched_primary_count: Number of primary rows with no secondary match
        total_primary_rows: Total rows in primary
        total_secondary_rows: Total rows in secondary
    """
    joined_df: pd.DataFrame
    primary_source: str = ""
    secondary_source: str = ""
    matches: List[IntervalMatch] = field(default_factory=list)
    unmatched_primary_count: int = 0
    total_primary_rows: int = 0
    total_secondary_rows: int = 0

    def get_matches_for_interval(
        self,
        hole_id: str,
        from_depth: float,
        to_depth: float
    ) -> List[IntervalMatch]:
        """Get all secondary matches for a specific primary interval."""
        return [
            m for m in self.matches
            if m.hole_id.upper() == hole_id.upper()
            and m.primary_from == from_depth
            and m.primary_to == to_depth
            and m.has_match
        ]

    def get_coverage_for_hole(self, hole_id: str) -> float:
        """
        Calculate what percentage of primary intervals have secondary coverage.

        Returns:
            Float 0.0-1.0 representing coverage percentage
        """
        hole_matches = [m for m in self.matches if m.hole_id.upper() == hole_id.upper()]
        if not hole_matches:
            return 0.0

        total_primary_length = sum(m.primary_to - m.primary_from for m in hole_matches)
        if total_primary_length == 0:
            return 0.0

        # Sum overlap lengths (accounting for multiple matches per primary)
        # Group by primary interval to avoid double-counting
        primary_intervals = {}
        for m in hole_matches:
            key = (m.primary_from, m.primary_to)
            if key not in primary_intervals:
                primary_intervals[key] = []
            if m.has_match:
                primary_intervals[key].append(m.overlap_length)

        # For each primary interval, take the union of overlaps
        total_covered = 0.0
        for (p_from, p_to), overlaps in primary_intervals.items():
            interval_length = p_to - p_from
            if overlaps:
                # Simple: sum overlaps (may overcount if secondary intervals overlap each other)
                # More accurate: merge overlapping secondary intervals
                covered = min(sum(overlaps), interval_length)
                total_covered += covered

        return total_covered / total_primary_length

    def summary(self) -> str:
        """Generate human-readable summary of the join."""
        matched = len([m for m in self.matches if m.has_match])
        return (
            f"Join Result: {self.primary_source} <- {self.secondary_source}\n"
            f"  Primary rows: {self.total_primary_rows}\n"
            f"  Secondary rows: {self.total_secondary_rows}\n"
            f"  Matched intervals: {matched}\n"
            f"  Unmatched primary: {self.unmatched_primary_count}\n"
            f"  Result rows: {len(self.joined_df)}"
        )


class DataJoiner:
    """
    Joins DataFrames using interval overlap matching.

    Implements LEFT JOIN semantics where all primary rows are preserved
    and secondary rows are matched based on hole_id and depth interval overlap.

    Key behaviors:
    - Primary rows never lost (LEFT JOIN)
    - Multiple secondary matches per primary allowed
    - Overlap percentage tracked for each match
    - Informative debug logging
    """

    def __init__(self):
        self._debug = True  # Enable verbose logging

    def join(
        self,
        primary_df: pd.DataFrame,
        secondary_df: pd.DataFrame,
        primary_key_cols: Dict[str, str],
        secondary_key_cols: Dict[str, str],
        aggregate_secondary: bool = False,
        primary_source: str = "primary",
        secondary_source: str = "secondary",
    ) -> JoinResult:
        """
        Join two DataFrames using interval overlap matching.

        Args:
            primary_df: Left DataFrame (all rows preserved)
            secondary_df: Right DataFrame (matched by overlap)
            primary_key_cols: Dict with keys 'hole', 'from', 'to' -> column names
            secondary_key_cols: Dict with keys 'hole', 'from', 'to' -> column names
            aggregate_secondary: If True, aggregate multiple secondary matches
            primary_source: Name for logging
            secondary_source: Name for logging

        Returns:
            JoinResult with joined DataFrame and match metadata
        """
        logger.info(f"Starting join: {primary_source} <- {secondary_source}")
        logger.debug(f"  Primary rows: {len(primary_df)}, columns: {list(primary_df.columns)}")
        logger.debug(f"  Secondary rows: {len(secondary_df)}, columns: {list(secondary_df.columns)}")

        # Validate key columns
        self._validate_key_cols(primary_df, primary_key_cols, "primary")
        self._validate_key_cols(secondary_df, secondary_key_cols, "secondary")

        # Extract column names
        p_hole = primary_key_cols["hole"]
        p_from = primary_key_cols["from"]
        p_to = primary_key_cols["to"]
        s_hole = secondary_key_cols["hole"]
        s_from = secondary_key_cols["from"]
        s_to = secondary_key_cols["to"]

        # Normalize hole IDs for matching
        primary_df = primary_df.copy()
        secondary_df = secondary_df.copy()
        primary_df["_hole_upper"] = primary_df[p_hole].astype(str).str.upper().str.strip()
        secondary_df["_hole_upper"] = secondary_df[s_hole].astype(str).str.upper().str.strip()

        # Find all matches
        matches = []
        matched_secondary_cols = [c for c in secondary_df.columns if c not in [s_hole, s_from, s_to, "_hole_upper"]]

        # Build secondary index by hole for faster lookup
        secondary_by_hole = secondary_df.groupby("_hole_upper")

        for p_idx, p_row in primary_df.iterrows():
            hole_upper = p_row["_hole_upper"]
            p_from_val = float(p_row[p_from])
            p_to_val = float(p_row[p_to])

            # Find overlapping secondary intervals
            if hole_upper in secondary_by_hole.groups:
                hole_secondary = secondary_by_hole.get_group(hole_upper)

                for s_idx, s_row in hole_secondary.iterrows():
                    s_from_val = float(s_row[s_from])
                    s_to_val = float(s_row[s_to])

                    # Check for overlap
                    overlap_from = max(p_from_val, s_from_val)
                    overlap_to = min(p_to_val, s_to_val)

                    if overlap_from < overlap_to:  # There is overlap
                        primary_length = p_to_val - p_from_val
                        overlap_pct = (overlap_to - overlap_from) / primary_length if primary_length > 0 else 0

                        match = IntervalMatch(
                            primary_idx=p_idx,
                            secondary_idx=s_idx,
                            hole_id=hole_upper,
                            primary_from=p_from_val,
                            primary_to=p_to_val,
                            secondary_from=s_from_val,
                            secondary_to=s_to_val,
                            overlap_from=overlap_from,
                            overlap_to=overlap_to,
                            overlap_pct=overlap_pct,
                        )
                        matches.append(match)

                        logger.debug(
                            f"  Match: {hole_upper} primary[{p_from_val}-{p_to_val}] <-> "
                            f"secondary[{s_from_val}-{s_to_val}] overlap={overlap_pct:.1%}"
                        )

            # Add "no match" entry if no overlaps found for this primary
            if not any(m.primary_idx == p_idx and m.has_match for m in matches):
                matches.append(IntervalMatch(
                    primary_idx=p_idx,
                    secondary_idx=None,
                    hole_id=hole_upper,
                    primary_from=p_from_val,
                    primary_to=p_to_val,
                ))

        # Build joined DataFrame
        if aggregate_secondary:
            joined_df = self._build_aggregated_join(
                primary_df, secondary_df, matches,
                matched_secondary_cols, p_hole, p_from, p_to
            )
        else:
            joined_df = self._build_expanded_join(
                primary_df, secondary_df, matches,
                matched_secondary_cols
            )

        # Clean up temp columns
        if "_hole_upper" in joined_df.columns:
            joined_df = joined_df.drop(columns=["_hole_upper"])

        unmatched = sum(1 for m in matches if not m.has_match)

        logger.info(
            f"Join complete: {len(joined_df)} result rows, "
            f"{len(matches) - unmatched} matches, {unmatched} unmatched primary"
        )

        return JoinResult(
            joined_df=joined_df,
            primary_source=primary_source,
            secondary_source=secondary_source,
            matches=matches,
            unmatched_primary_count=unmatched,
            total_primary_rows=len(primary_df),
            total_secondary_rows=len(secondary_df),
        )

    def _validate_key_cols(self, df: pd.DataFrame, key_cols: Dict[str, str], name: str):
        """Validate that key columns exist in DataFrame."""
        required = {"hole", "from", "to"}
        if not required.issubset(key_cols.keys()):
            raise ValueError(f"{name}_key_cols must have 'hole', 'from', 'to' keys")

        for key, col in key_cols.items():
            if col not in df.columns:
                raise ValueError(f"{name} DataFrame missing column '{col}' (for {key})")

    def _build_expanded_join(
        self,
        primary_df: pd.DataFrame,
        secondary_df: pd.DataFrame,
        matches: List[IntervalMatch],
        secondary_cols: List[str],
    ) -> pd.DataFrame:
        """Build join where each match creates a row (may have multiple rows per primary)."""
        rows = []

        # Group matches by primary index
        matches_by_primary = {}
        for m in matches:
            if m.primary_idx not in matches_by_primary:
                matches_by_primary[m.primary_idx] = []
            matches_by_primary[m.primary_idx].append(m)

        for p_idx, p_matches in matches_by_primary.items():
            p_row = primary_df.loc[p_idx].to_dict()

            # If no matches, add primary row with NaN for secondary
            if not any(m.has_match for m in p_matches):
                for col in secondary_cols:
                    p_row[col] = np.nan
                rows.append(p_row)
            else:
                # Add one row per match
                for m in p_matches:
                    if m.has_match:
                        row = p_row.copy()
                        s_row = secondary_df.loc[m.secondary_idx]
                        for col in secondary_cols:
                            row[col] = s_row[col]
                        row["_overlap_pct"] = m.overlap_pct
                        rows.append(row)

        return pd.DataFrame(rows)

    def _build_aggregated_join(
        self,
        primary_df: pd.DataFrame,
        secondary_df: pd.DataFrame,
        matches: List[IntervalMatch],
        secondary_cols: List[str],
        p_hole: str,
        p_from: str,
        p_to: str,
    ) -> pd.DataFrame:
        """Build join where secondary values are aggregated per primary row."""
        rows = []

        # Group matches by primary index
        matches_by_primary = {}
        for m in matches:
            if m.primary_idx not in matches_by_primary:
                matches_by_primary[m.primary_idx] = []
            matches_by_primary[m.primary_idx].append(m)

        for p_idx in primary_df.index:
            p_row = primary_df.loc[p_idx].to_dict()
            p_matches = matches_by_primary.get(p_idx, [])

            matched = [m for m in p_matches if m.has_match]

            if not matched:
                # No matches - NaN for all aggregates
                for col in secondary_cols:
                    p_row[f"{col}_mean"] = np.nan
                    p_row[f"{col}_min"] = np.nan
                    p_row[f"{col}_max"] = np.nan
                p_row["_match_count"] = 0
                p_row["_total_overlap_pct"] = 0.0
            else:
                # Aggregate secondary values
                secondary_values = {col: [] for col in secondary_cols}
                total_overlap = 0.0

                for m in matched:
                    s_row = secondary_df.loc[m.secondary_idx]
                    for col in secondary_cols:
                        val = s_row[col]
                        if pd.notna(val):
                            try:
                                secondary_values[col].append(float(val))
                            except (ValueError, TypeError):
                                pass  # Skip non-numeric
                    total_overlap += m.overlap_pct

                for col in secondary_cols:
                    vals = secondary_values[col]
                    if vals:
                        p_row[f"{col}_mean"] = np.mean(vals)
                        p_row[f"{col}_min"] = np.min(vals)
                        p_row[f"{col}_max"] = np.max(vals)
                    else:
                        p_row[f"{col}_mean"] = np.nan
                        p_row[f"{col}_min"] = np.nan
                        p_row[f"{col}_max"] = np.nan

                p_row["_match_count"] = len(matched)
                p_row["_total_overlap_pct"] = min(total_overlap, 1.0)

            rows.append(p_row)

        return pd.DataFrame(rows)
```

**Step 2: Run tests**

Run: `pytest src/processing/DataManager/tests/test_data_joiner.py -v`
Expected: All tests PASS

**Step 3: Commit implementation**

```bash
git add src/processing/DataManager/data_joiner.py
git commit -m "feat: add DataJoiner for cross-table interval overlap joining"
```

---

### Task 2.3: Integrate DataJoiner into DataCoordinator

**Files:**
- Modify: `src/processing/DataManager/data_coordinator.py`

**Step 1: Write failing test for DataCoordinator.join_sources**

Add to `test_data_joiner.py`:

```python
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
```

**Step 2: Run test - should fail**

Run: `pytest src/processing/DataManager/tests/test_data_joiner.py::TestDataCoordinatorJoin -v`
Expected: FAIL with `AttributeError: 'DataCoordinator' object has no attribute 'join_sources'`

**Step 3: Implement join_sources in DataCoordinator**

Add to `data_coordinator.py` after line ~900 (after `get_column_values`):

```python
def join_sources(
    self,
    primary: str,
    secondary: str,
    aggregate: bool = False,
    primary_key_cols: Optional[Dict[str, str]] = None,
    secondary_key_cols: Optional[Dict[str, str]] = None,
) -> Optional["JoinResult"]:
    """
    Join two data sources using interval overlap matching.

    Uses LEFT JOIN semantics - all primary rows preserved,
    secondary matched by hole_id and depth interval overlap.

    Args:
        primary: Name of primary data source (all rows kept)
        secondary: Name of secondary data source (matched by overlap)
        aggregate: If True, aggregate secondary values per primary row
        primary_key_cols: Override key column detection for primary
        secondary_key_cols: Override key column detection for secondary

    Returns:
        JoinResult with joined DataFrame and match metadata, or None if sources not found
    """
    from processing.DataManager.data_joiner import DataJoiner

    # Get source DataFrames
    primary_source = self._geological_store.get_source(primary)
    secondary_source = self._geological_store.get_source(secondary)

    if not primary_source or primary_source.df is None:
        logger.error(f"Primary source '{primary}' not found or empty")
        return None

    if not secondary_source or secondary_source.df is None:
        logger.error(f"Secondary source '{secondary}' not found or empty")
        return None

    # Auto-detect key columns if not provided
    if primary_key_cols is None:
        primary_key_cols = self._detect_interval_columns(primary_source)

    if secondary_key_cols is None:
        secondary_key_cols = self._detect_interval_columns(secondary_source)

    logger.info(f"Joining {primary} <- {secondary}")
    logger.debug(f"  Primary keys: {primary_key_cols}")
    logger.debug(f"  Secondary keys: {secondary_key_cols}")

    joiner = DataJoiner()
    result = joiner.join(
        primary_df=primary_source.df,
        secondary_df=secondary_source.df,
        primary_key_cols=primary_key_cols,
        secondary_key_cols=secondary_key_cols,
        aggregate_secondary=aggregate,
        primary_source=primary,
        secondary_source=secondary,
    )

    logger.info(result.summary())
    return result

def _detect_interval_columns(self, source: "IndexedDataSource") -> Dict[str, str]:
    """
    Auto-detect hole_id, from, and to columns for a data source.

    Args:
        source: IndexedDataSource to inspect

    Returns:
        Dict with 'hole', 'from', 'to' keys mapped to column names
    """
    schema = source.schema
    cols = {c.lower() for c in source.df.columns}

    # Hole ID
    hole_col = schema.hole_id_column
    if not hole_col:
        for candidate in ["holeid", "hole_id", "bhid"]:
            if candidate in cols:
                hole_col = candidate
                break

    # From depth
    from_col = schema.from_column
    if not from_col:
        for candidate in ["geolfrom", "sampfrom", "from", "depth_from"]:
            if candidate in cols:
                from_col = candidate
                break

    # To depth
    to_col = schema.to_column
    if not to_col:
        for candidate in ["geolto", "sampto", "to", "depth_to"]:
            if candidate in cols:
                to_col = candidate
                break

    if not all([hole_col, from_col, to_col]):
        raise ValueError(
            f"Could not detect interval columns for {source.name}. "
            f"Found: hole={hole_col}, from={from_col}, to={to_col}"
        )

    return {"hole": hole_col, "from": from_col, "to": to_col}
```

**Step 4: Run tests**

Run: `pytest src/processing/DataManager/tests/test_data_joiner.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/processing/DataManager/data_coordinator.py
git commit -m "feat: add join_sources API to DataCoordinator"
```

---

## Phase 3: RC Metrics Accessibility

### Task 3.1: Expose RC Metrics as Virtual Columns

**Files:**
- Modify: `src/processing/DataManager/data_coordinator.py`
- Create: `src/processing/DataManager/tests/test_rc_metrics_accessibility.py`

**Step 1: Write failing test**

```python
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
```

**Step 2: Run test**

Run: `pytest src/processing/DataManager/tests/test_rc_metrics_accessibility.py -v`
Expected: Some tests may pass (get_rc_metrics exists), some may fail (available_columns)

**Step 3: Modify get_available_columns to include RC metrics**

In `data_coordinator.py`, modify `get_available_columns`:

```python
def get_available_columns(self) -> List[Tuple[str, DataType]]:
    """
    Get list of available columns with their data types.

    Includes:
    - CSV columns from all geological sources
    - Computed RC metrics (if available)

    Returns:
        List of (column_name, DataType) tuples
    """
    all_columns = self._geological_store.get_available_columns()

    result = []
    seen = set()

    # Add geological columns
    for source_name, columns in all_columns.items():
        for col_name, col_type in columns:
            if col_name not in seen:
                result.append((col_name, col_type))
                seen.add(col_name)

    # Add RC metric columns if computed
    if self.has_rc_metrics:
        rc_columns = [
            ("weighted_hardness", DataType.NUMERIC),
            ("total_gangue_pct", DataType.NUMERIC),
            ("si_gangue_pct", DataType.NUMERIC),
            ("al_gangue_pct", DataType.NUMERIC),
            ("carbonate_gangue_pct", DataType.NUMERIC),
            ("zonation_pr_pct", DataType.NUMERIC),
            ("zonation_hy_pct", DataType.NUMERIC),
            ("zonation_de_pct", DataType.NUMERIC),
            ("zonation_un_pct", DataType.NUMERIC),
            ("quartz_pct", DataType.NUMERIC),
            ("chert_pct", DataType.NUMERIC),
        ]
        for col_name, col_type in rc_columns:
            if col_name not in seen:
                result.append((col_name, col_type))
                seen.add(col_name)

        logger.debug(f"Added {len(rc_columns)} RC metric columns to available columns")

    return result
```

**Step 4: Run tests**

Run: `pytest src/processing/DataManager/tests/test_rc_metrics_accessibility.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/processing/DataManager/data_coordinator.py src/processing/DataManager/tests/test_rc_metrics_accessibility.py
git commit -m "feat: expose RC metrics in available columns list"
```

---

## Phase 4: Column Settings Dialog Fixes

### Task 4.1: Fix Column Header Alignment Using Grid Layout

**Files:**
- Modify: `src/gui/column_settings_dialog.py`

**Step 1: Identify the alignment issue**

Current code uses `pack(side=tk.LEFT)` with fixed `width` on Labels, but Entry/OptionMenu/Spinbox don't respect the same widths.

**Step 2: Refactor to use grid layout**

Replace `_create_column_headers` and modify `_create_column_row` to use grid:

```python
# Define column widths as class constants (add near top of class)
COLUMN_WIDTHS = {
    "visible": 50,      # Checkbox
    "name": 150,        # Column name
    "alias": 150,       # Alias entry
    "type": 100,        # Data type dropdown
    "decimals": 60,     # Decimals spinbox
    "nulls": 100,       # Null handling dropdown
    "colormap": 120,    # Color map dropdown + button
    "unique": 60,       # Unique count
    "nullcount": 100,   # Null count
    "key": 40,          # Key indicator
}

def _create_column_headers(self, parent):
    """Create the header row for columns table using grid layout."""
    header_frame = tk.Frame(parent, bg=self.theme["field_bg"])
    header_frame.pack(fill=tk.X, padx=5, pady=(5, 0))

    headers = [
        ("Visible", self.COLUMN_WIDTHS["visible"]),
        ("Column Name", self.COLUMN_WIDTHS["name"]),
        ("Alias", self.COLUMN_WIDTHS["alias"]),
        ("Type", self.COLUMN_WIDTHS["type"]),
        ("Dec", self.COLUMN_WIDTHS["decimals"]),
        ("Null Handling", self.COLUMN_WIDTHS["nulls"]),
        ("Color Map", self.COLUMN_WIDTHS["colormap"]),
        ("Uniq", self.COLUMN_WIDTHS["unique"]),
        ("Nulls", self.COLUMN_WIDTHS["nullcount"]),
        ("Key", self.COLUMN_WIDTHS["key"]),
    ]

    for col_idx, (text, width) in enumerate(headers):
        label = tk.Label(
            header_frame,
            text=text,
            font=("Arial", 9, "bold"),
            bg=self.theme["field_bg"],
            fg=self.theme["text"],
            anchor="w",
        )
        label.grid(row=0, column=col_idx, sticky="w", padx=2, pady=3)
        header_frame.columnconfigure(col_idx, minsize=width)

def _create_column_row(self, parent, source_name: str, config: ColumnDisplayConfig):
    """Create a row for a single column using grid layout."""
    row_idx = len([w for w in parent.winfo_children() if isinstance(w, tk.Frame)]) - 1
    row_bg = self.theme["background"] if row_idx % 2 == 0 else self.theme["secondary_bg"]

    if config.is_key:
        row_bg = self.theme.get("highlight_bg", "#3a5a7a")

    row_frame = tk.Frame(parent, bg=row_bg)
    row_frame.pack(fill=tk.X, padx=5, pady=1)
    row_frame.column_config = config

    col = 0

    # Visible checkbox
    config.visible_var = tk.BooleanVar(value=config.is_visible)
    visible_cb = tk.Checkbutton(
        row_frame,
        variable=config.visible_var,
        bg=row_bg,
        fg=self.theme["text"],
        selectcolor=self.theme.get("accent_green", "#47b881"),
        activebackground=row_bg,
    )
    visible_cb.grid(row=0, column=col, sticky="w", padx=2)
    row_frame.columnconfigure(col, minsize=self.COLUMN_WIDTHS["visible"])
    col += 1

    # Column name (read-only label)
    name_label = tk.Label(
        row_frame,
        text=config.source_name,
        font=("Arial", 9),
        bg=row_bg,
        fg=self.theme["text"],
        anchor="w",
    )
    name_label.grid(row=0, column=col, sticky="w", padx=2)
    row_frame.columnconfigure(col, minsize=self.COLUMN_WIDTHS["name"])
    col += 1

    # Alias entry
    config.alias_var = tk.StringVar(value=config.display_name)
    alias_entry = tk.Entry(
        row_frame,
        textvariable=config.alias_var,
        font=("Arial", 9),
        bg=self.theme["field_bg"],
        fg=self.theme["text"],
        width=18,
    )
    alias_entry.grid(row=0, column=col, sticky="w", padx=2)
    row_frame.columnconfigure(col, minsize=self.COLUMN_WIDTHS["alias"])
    col += 1

    # Data type dropdown
    config.type_var = tk.StringVar(value=config.data_type)
    type_menu = ttk.Combobox(
        row_frame,
        textvariable=config.type_var,
        values=self.DATA_TYPES,
        state="readonly",
        width=10,
        font=("Arial", 9),
    )
    type_menu.grid(row=0, column=col, sticky="w", padx=2)
    row_frame.columnconfigure(col, minsize=self.COLUMN_WIDTHS["type"])
    col += 1

    # Decimals spinbox
    config.decimals_var = tk.IntVar(value=config.decimals)
    decimals_spin = tk.Spinbox(
        row_frame,
        from_=0,
        to=6,
        textvariable=config.decimals_var,
        width=4,
        font=("Arial", 9),
        bg=self.theme["field_bg"],
        fg=self.theme["text"],
    )
    decimals_spin.grid(row=0, column=col, sticky="w", padx=2)
    row_frame.columnconfigure(col, minsize=self.COLUMN_WIDTHS["decimals"])
    col += 1

    # Null handling dropdown
    config.nulls_var = tk.StringVar(value=config.null_handling)
    nulls_menu = ttk.Combobox(
        row_frame,
        textvariable=config.nulls_var,
        values=self.NULL_HANDLING,
        state="readonly",
        width=10,
        font=("Arial", 9),
    )
    nulls_menu.grid(row=0, column=col, sticky="w", padx=2)
    row_frame.columnconfigure(col, minsize=self.COLUMN_WIDTHS["nulls"])
    col += 1

    # Color map dropdown + edit button frame
    colormap_frame = tk.Frame(row_frame, bg=row_bg)
    colormap_frame.grid(row=0, column=col, sticky="w", padx=2)
    row_frame.columnconfigure(col, minsize=self.COLUMN_WIDTHS["colormap"])
    col += 1

    available_maps = ["(None)"]
    if self.data_coordinator and self.data_coordinator.color_maps:
        try:
            if config.data_type == "numeric":
                available_maps.extend(self.data_coordinator.color_maps.list_numeric())
            elif config.data_type == "categorical":
                available_maps.extend(self.data_coordinator.color_maps.list_categorical())
            else:
                available_maps.extend(self.data_coordinator.color_maps.list_all())
        except Exception:
            pass

    config.colormap_var = tk.StringVar(value=config.color_map or "(None)")
    colormap_combo = ttk.Combobox(
        colormap_frame,
        textvariable=config.colormap_var,
        values=available_maps,
        state="readonly",
        width=10,
        font=("Arial", 9),
    )
    colormap_combo.pack(side=tk.LEFT)
    config.colormap_menu = colormap_combo

    edit_btn = tk.Button(
        colormap_frame,
        text="...",
        font=("Arial", 8),
        width=2,
        bg=self.theme["field_bg"],
        command=lambda c=config, s=source_name: self._open_color_map_editor(c, s),
    )
    edit_btn.pack(side=tk.LEFT, padx=2)

    # Unique count
    unique_label = tk.Label(
        row_frame,
        text=f"{config.unique_count:,}" if config.unique_count > 0 else "-",
        font=("Arial", 9),
        bg=row_bg,
        fg=self.theme["text"],
        anchor="e",
    )
    unique_label.grid(row=0, column=col, sticky="e", padx=2)
    row_frame.columnconfigure(col, minsize=self.COLUMN_WIDTHS["unique"])
    col += 1

    # Null count with percentage
    null_pct = (config.null_count / config.total_count * 100) if config.total_count > 0 else 0
    null_text = f"{config.null_count:,} ({null_pct:.0f}%)" if config.total_count > 0 else "-"
    null_label = tk.Label(
        row_frame,
        text=null_text,
        font=("Arial", 9),
        bg=row_bg,
        fg=self.theme["accent_red"] if null_pct > 50 else self.theme["text"],
        anchor="e",
    )
    null_label.grid(row=0, column=col, sticky="e", padx=2)
    row_frame.columnconfigure(col, minsize=self.COLUMN_WIDTHS["nullcount"])
    col += 1

    # Key indicator
    key_text = "K" if config.is_key else ""
    key_label = tk.Label(
        row_frame,
        text=key_text,
        font=("Arial", 10, "bold"),
        bg=row_bg,
        fg=self.theme["accent_blue"] if config.is_key else self.theme.get("subtext", "#888"),
    )
    key_label.grid(row=0, column=col, sticky="w", padx=2)
    row_frame.columnconfigure(col, minsize=self.COLUMN_WIDTHS["key"])
```

**Step 3: Commit**

```bash
git add src/gui/column_settings_dialog.py
git commit -m "fix: use grid layout for column alignment in settings dialog"
```

---

### Task 4.2: Add Sanitization Warnings Panel

**Files:**
- Modify: `src/gui/column_settings_dialog.py`

**Step 1: Add warnings panel to _build_ui**

After `_build_header(main_frame)`, add:

```python
def _build_ui(self):
    """Build the dialog UI layout."""
    main_frame = tk.Frame(self.dialog, bg=self.theme["background"])
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # TOP: Title and filter bar
    self._build_header(main_frame)

    # WARNINGS: Sanitization issues panel (collapsible)
    self._build_warnings_panel(main_frame)

    # MIDDLE: Scrollable content area with data sources
    self._build_content_area(main_frame)

    # BOTTOM: Action buttons
    self._build_bottom_buttons(main_frame)

def _build_warnings_panel(self, parent):
    """Build collapsible panel showing data sanitization warnings."""
    # Get sanitization report from coordinator
    warnings = []
    if self.data_coordinator and hasattr(self.data_coordinator, '_geological_store'):
        try:
            from processing.DataManager.data_sanitizer import sanitize_geological_store
            report = sanitize_geological_store(self.data_coordinator._geological_store)
            warnings = [issue for issue in report.issues if issue.severity.value in ("warning", "error")]
        except Exception as e:
            logger.debug(f"Could not get sanitization report: {e}")

    if not warnings:
        return  # No warnings, don't show panel

    # Create collapsible warnings frame
    self.warnings_expanded = tk.BooleanVar(value=True)

    warnings_container = tk.Frame(
        parent,
        bg=self.theme.get("accent_orange", "#d97a4a"),
        highlightbackground=self.theme.get("accent_red", "#c74440"),
        highlightthickness=2,
    )
    warnings_container.pack(fill=tk.X, pady=(0, 10))

    # Header
    header_frame = tk.Frame(warnings_container, bg=self.theme.get("accent_orange", "#d97a4a"))
    header_frame.pack(fill=tk.X)

    expand_label = tk.Label(
        header_frame,
        text="!",
        font=("Arial", 12, "bold"),
        bg=self.theme.get("accent_orange", "#d97a4a"),
        fg="white",
    )
    expand_label.pack(side=tk.LEFT, padx=10, pady=5)

    title_label = tk.Label(
        header_frame,
        text=f"Data Warnings ({len(warnings)})",
        font=("Arial", 11, "bold"),
        bg=self.theme.get("accent_orange", "#d97a4a"),
        fg="white",
        cursor="hand2",
    )
    title_label.pack(side=tk.LEFT, pady=5)

    # Warnings content
    self.warnings_content = tk.Frame(warnings_container, bg=self.theme["background"])
    self.warnings_content.pack(fill=tk.X, padx=5, pady=5)

    for issue in warnings[:10]:  # Show first 10
        issue_frame = tk.Frame(self.warnings_content, bg=self.theme["background"])
        issue_frame.pack(fill=tk.X, pady=1)

        severity_color = self.theme.get("accent_red") if issue.severity.value == "error" else self.theme.get("accent_orange")

        tk.Label(
            issue_frame,
            text=f"[{issue.severity.value.upper()}]",
            font=("Arial", 8, "bold"),
            bg=self.theme["background"],
            fg=severity_color,
        ).pack(side=tk.LEFT, padx=(0, 5))

        tk.Label(
            issue_frame,
            text=f"{issue.source}.{issue.column or ''}: {issue.message}",
            font=("Arial", 9),
            bg=self.theme["background"],
            fg=self.theme["text"],
            anchor="w",
        ).pack(side=tk.LEFT, fill=tk.X)

    if len(warnings) > 10:
        tk.Label(
            self.warnings_content,
            text=f"... and {len(warnings) - 10} more warnings",
            font=("Arial", 9, "italic"),
            bg=self.theme["background"],
            fg=self.theme.get("subtext", "#888"),
        ).pack(anchor="w", padx=5)

    # Toggle expand/collapse
    def toggle_warnings(event=None):
        if self.warnings_expanded.get():
            self.warnings_content.pack_forget()
            self.warnings_expanded.set(False)
        else:
            self.warnings_content.pack(fill=tk.X, padx=5, pady=5)
            self.warnings_expanded.set(True)

    title_label.bind("<Button-1>", toggle_warnings)
```

**Step 2: Commit**

```bash
git add src/gui/column_settings_dialog.py
git commit -m "feat: add collapsible warnings panel to column settings dialog"
```

---

## Phase 5: Comprehensive Unit Tests

### Task 5.1: Create Test Suite for GeologicalStore

**Files:**
- Create: `src/processing/DataManager/tests/test_geological_store.py`

**Step 1: Write comprehensive tests**

```python
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
        assert store.load_all()
        assert store.is_loaded
        assert len(store.list_sources()) == 1

    def test_load_multiple_csvs(self, sample_interval_csv, sample_collar_csv):
        """Should load multiple CSV files."""
        store = GeologicalStore()
        store.add_source(sample_interval_csv)
        store.add_source(sample_collar_csv)
        assert store.load_all()
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
        assert store.load_all()

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
        schema = infer_schema(str(csv_path))
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
```

**Step 2: Run tests**

Run: `pytest src/processing/DataManager/tests/test_geological_store.py -v`

**Step 3: Commit**

```bash
git add src/processing/DataManager/tests/test_geological_store.py
git commit -m "test: add comprehensive unit tests for GeologicalStore"
```

---

## Summary

**Phases completed:**
1. Settings Persistence - GeologicalStore now saves/loads column schemas
2. Cross-Table Joining - DataJoiner with interval overlap detection
3. RC Metrics Accessibility - Metrics exposed in available columns
4. Dialog Layout - Grid layout for alignment, warnings panel
5. Unit Tests - Comprehensive test coverage

**Key architectural decisions:**
- LEFT JOIN semantics for data joining (preserve all primary rows)
- Interval overlap detection with percentage tracking
- Saved schemas applied before IndexedDataSource creation
- RC metrics as virtual columns in available columns list

**Files created:**
- `src/processing/DataManager/data_joiner.py`
- `src/processing/DataManager/tests/test_settings_persistence.py`
- `src/processing/DataManager/tests/test_data_joiner.py`
- `src/processing/DataManager/tests/test_rc_metrics_accessibility.py`
- `src/processing/DataManager/tests/test_geological_store.py`

**Files modified:**
- `src/processing/DataManager/geological_store.py`
- `src/processing/DataManager/data_coordinator.py`
- `src/gui/column_settings_dialog.py`
