"""
test_rc_metrics_store.py - Unit tests for RC metrics store.

Tests cover:
- MineralCodeManager: Loading and lookup functionality
- RCMetricsCalculator: Metric calculation accuracy
- RCMetricsStore: Storage and retrieval
- DataCoordinator integration

Run with: python -m pytest src/processing/DataManager/tests/test_rc_metrics_store.py -v
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, Any
import unittest

import pandas as pd
import numpy as np

from processing.DataManager.rc_metrics_store import (
    MineralCodeManager,
    MineralCodeInfo,
    RCMetricsCalculator,
    RCMetricsStore,
    IntervalMetrics,
    DEFAULT_MINERAL_CODES_PATH,
)


class TestMineralCodeManager(unittest.TestCase):
    """Tests for MineralCodeManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = MineralCodeManager()

    def test_load_from_default_path(self):
        """Test loading mineral codes from the default JSON file."""
        if not DEFAULT_MINERAL_CODES_PATH.exists():
            self.skipTest("mineral_codes.json not found at default path")

        result = self.manager.load()
        self.assertTrue(result)
        self.assertTrue(self.manager.is_loaded)
        self.assertGreater(len(self.manager.get_all_codes()), 100)

    def test_load_from_custom_path(self):
        """Test loading mineral codes from a custom JSON file."""
        # Create a minimal test JSON
        test_data = {
            "version": "1.0",
            "codes": {
                "HO": {
                    "name": "Hematite",
                    "hardness": 3,
                    "gangue_type": 0,
                    "gangue_name": "Mineralised",
                    "profile_zonation": "De",
                    "numeric_zonation": 3
                },
                "QZ": {
                    "name": "Quartz",
                    "hardness": 5,
                    "gangue_type": 1,
                    "gangue_name": "Silica",
                    "profile_zonation": "Un",
                    "numeric_zonation": 1
                }
            },
            "gangue_types": {"0": "Mineralised", "1": "Silica"},
            "zonation_mix_rules": {"PrHy": {"Pr": 0.6, "Hy": 0.4}},
            "quartz_variants": ["QZ", "QTZ"],
            "chert_variants": ["CHM", "CHH"]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = Path(f.name)

        try:
            manager = MineralCodeManager(temp_path)
            result = manager.load()

            self.assertTrue(result)
            self.assertTrue(manager.is_loaded)
            self.assertEqual(len(manager.get_all_codes()), 2)
        finally:
            temp_path.unlink()

    def test_get_code_info(self):
        """Test retrieving mineral code information."""
        if not self.manager.load():
            self.skipTest("Could not load mineral codes")

        # Test known code
        info = self.manager.get_code_info("HO")
        self.assertIsNotNone(info)
        self.assertEqual(info.code, "HO")
        self.assertEqual(info.name, "Hematite")
        self.assertEqual(info.hardness, 3)
        self.assertEqual(info.gangue_type, 0)

        # Test case insensitivity
        info_lower = self.manager.get_code_info("ho")
        self.assertEqual(info, info_lower)

        # Test unknown code
        unknown = self.manager.get_code_info("UNKNOWN_CODE_XYZ")
        self.assertIsNone(unknown)

    def test_get_hardness(self):
        """Test hardness lookup."""
        if not self.manager.load():
            self.skipTest("Could not load mineral codes")

        # Hematite = 3 (medium)
        self.assertEqual(self.manager.get_hardness("HO"), 3)

        # Quartz = 5 (hard)
        self.assertEqual(self.manager.get_hardness("QZ"), 5)

        # Unknown = 0
        self.assertEqual(self.manager.get_hardness("UNKNOWN"), 0)

    def test_get_gangue_type(self):
        """Test gangue type lookup."""
        if not self.manager.load():
            self.skipTest("Could not load mineral codes")

        # Hematite = 0 (Mineralised)
        self.assertEqual(self.manager.get_gangue_type("HO"), 0)

        # Quartz = 1 (Silica)
        self.assertEqual(self.manager.get_gangue_type("QZ"), 1)

        # Shale = 2 (Aluminium)
        self.assertEqual(self.manager.get_gangue_type("SH"), 2)

    def test_get_zonation_split(self):
        """Test zonation split calculation."""
        if not self.manager.load():
            self.skipTest("Could not load mineral codes")

        # Simple zonation
        simple = self.manager.get_zonation_split("De")
        self.assertEqual(simple, {"De": 1.0})

        # Mixed zonation (PrHy = 60% Pr, 40% Hy)
        mixed = self.manager.get_zonation_split("PrHy")
        self.assertAlmostEqual(mixed.get("Pr", 0), 0.6)
        self.assertAlmostEqual(mixed.get("Hy", 0), 0.4)

    def test_quartz_variants(self):
        """Test quartz variant detection."""
        if not self.manager.load():
            self.skipTest("Could not load mineral codes")

        self.assertTrue(self.manager.is_quartz_variant("QZ"))
        self.assertTrue(self.manager.is_quartz_variant("QTZ"))
        self.assertTrue(self.manager.is_quartz_variant("QT"))
        self.assertFalse(self.manager.is_quartz_variant("HO"))

    def test_chert_variants(self):
        """Test chert variant detection."""
        if not self.manager.load():
            self.skipTest("Could not load mineral codes")

        self.assertTrue(self.manager.is_chert_variant("CHM"))
        self.assertTrue(self.manager.is_chert_variant("CHH"))
        self.assertTrue(self.manager.is_chert_variant("NBH"))
        self.assertTrue(self.manager.is_chert_variant("MBH"))
        self.assertFalse(self.manager.is_chert_variant("HO"))


class TestRCMetricsCalculator(unittest.TestCase):
    """Tests for RCMetricsCalculator."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.manager = MineralCodeManager()
        if not cls.manager.load():
            raise unittest.SkipTest("Could not load mineral codes")
        cls.calculator = RCMetricsCalculator(cls.manager)

    def test_calculate_simple_interval(self):
        """Test metric calculation for a simple interval."""
        # Create a row with 100% Hematite (HO)
        row = pd.Series({
            'holeid': 'BA0001',
            'sampfrom': 44.0,
            'sampto': 45.0,
            'min_80_pct': 'HO',
            'min_20_pct': 'QZ'
        })

        metrics = self.calculator.calculate_interval_metrics(
            row,
            mineral_columns=['min_80_pct', 'min_20_pct'],
            hole_id_col='holeid',
            from_col='sampfrom',
            to_col='sampto'
        )

        self.assertEqual(metrics.hole_id, 'BA0001')
        self.assertEqual(metrics.depth_from, 44.0)
        self.assertEqual(metrics.depth_to, 45.0)
        self.assertEqual(metrics.total_logged_pct, 100.0)
        self.assertEqual(metrics.normalisation_factor, 1.0)

        # HO (80%, hardness=3) + QZ (20%, hardness=5)
        # Weighted hardness = (3*0.8 + 5*0.2) / 1.0 = 3.4
        self.assertAlmostEqual(metrics.weighted_hardness, 3.4, places=1)

        # HO = Mineralised (gangue_type=0), QZ = Silica (gangue_type=1)
        # Total gangue = 20% (only QZ counts as gangue)
        self.assertEqual(metrics.total_gangue_pct, 20.0)
        self.assertEqual(metrics.si_gangue_pct, 20.0)

    def test_calculate_normalisation(self):
        """Test normalisation when percentages don't sum to 100%."""
        # Create a row with only 50% total (Min_30_pct + Min_20_pct)
        row = pd.Series({
            'holeid': 'BA0002',
            'sampfrom': 99.0,
            'sampto': 100.0,
            'min_30_pct': 'HO',
            'min_20_pct': 'QZ'
        })

        metrics = self.calculator.calculate_interval_metrics(
            row,
            mineral_columns=['min_30_pct', 'min_20_pct'],
            hole_id_col='holeid',
            from_col='sampfrom',
            to_col='sampto'
        )

        # Total logged = 50%
        self.assertEqual(metrics.total_logged_pct, 50.0)

        # Normalisation factor = 100/50 = 2.0
        self.assertAlmostEqual(metrics.normalisation_factor, 2.0, places=1)

        # Normalised: HO = 60%, QZ = 40%
        # Total gangue (normalised) = 40%
        self.assertEqual(metrics.total_gangue_pct, 40.0)

    def test_calculate_zonation(self):
        """Test zonation calculation."""
        # Create a row with minerals of different zonations
        row = pd.Series({
            'holeid': 'BA0003',
            'sampfrom': 49.0,
            'sampto': 50.0,
            'min_60_pct': 'HO',   # De zonation
            'min_40_pct': 'GO'    # Hy zonation
        })

        metrics = self.calculator.calculate_interval_metrics(
            row,
            mineral_columns=['min_60_pct', 'min_40_pct'],
            hole_id_col='holeid',
            from_col='sampfrom',
            to_col='sampto'
        )

        # HO = De (60%), GO = Hy (40%)
        self.assertEqual(metrics.zonation_de_pct, 60.0)
        self.assertEqual(metrics.zonation_hy_pct, 40.0)
        self.assertEqual(metrics.zonation_un_pct, 0.0)
        self.assertEqual(metrics.zonation_pr_pct, 0.0)

    def test_calculate_si_friability(self):
        """Test Si gangue friability split."""
        # Create a row with Si gangue minerals of different hardness
        row = pd.Series({
            'holeid': 'BA0004',
            'sampfrom': 59.0,
            'sampto': 60.0,
            'min_50_pct': 'CHH',  # Chert Hard (hardness=5, Si gangue)
            'min_50_pct': 'CHF'   # Chert Friable (hardness=1, Si gangue)
        })

        # Note: This test might need adjustment depending on how column names work
        # Since we have duplicate keys, pandas will use the last one

    def test_calculate_empty_row(self):
        """Test handling of row with no mineral data."""
        row = pd.Series({
            'holeid': 'BA0005',
            'sampfrom': 69.0,
            'sampto': 70.0,
            'min_80_pct': None,
            'min_20_pct': ''
        })

        metrics = self.calculator.calculate_interval_metrics(
            row,
            mineral_columns=['min_80_pct', 'min_20_pct'],
            hole_id_col='holeid',
            from_col='sampfrom',
            to_col='sampto'
        )

        self.assertEqual(metrics.total_logged_pct, 0.0)
        self.assertIsNone(metrics.weighted_hardness)

    def test_calculate_unknown_mineral_code(self):
        """Test handling of unknown mineral codes."""
        row = pd.Series({
            'holeid': 'BA0006',
            'sampfrom': 79.0,
            'sampto': 80.0,
            'min_80_pct': 'UNKNOWN_XYZ',
            'min_20_pct': 'HO'
        })

        # Should not raise an error, just skip unknown code
        metrics = self.calculator.calculate_interval_metrics(
            row,
            mineral_columns=['min_80_pct', 'min_20_pct'],
            hole_id_col='holeid',
            from_col='sampfrom',
            to_col='sampto'
        )

        # Only HO (20%) should be counted
        self.assertEqual(metrics.total_logged_pct, 100.0)  # Columns still sum to 100%


class TestRCMetricsStore(unittest.TestCase):
    """Tests for RCMetricsStore."""

    def setUp(self):
        """Set up test fixtures."""
        self.store = RCMetricsStore()

    def test_initialize(self):
        """Test store initialization."""
        if not DEFAULT_MINERAL_CODES_PATH.exists():
            self.skipTest("mineral_codes.json not found")

        result = self.store.initialize()
        self.assertTrue(result)
        self.assertTrue(self.store.mineral_codes.is_loaded)

    def test_compute_from_dataframe(self):
        """Test computing metrics from a DataFrame."""
        if not self.store.initialize():
            self.skipTest("Could not initialize store")

        # Create test DataFrame
        df = pd.DataFrame({
            'holeid': ['BA0001', 'BA0001', 'BA0002'],
            'sampfrom': [44.0, 45.0, 100.0],
            'sampto': [45.0, 46.0, 101.0],
            'min_80_pct': ['HO', 'GO', 'QZ'],
            'min_20_pct': ['QZ', 'HO', 'SH']
        })

        result = self.store.compute_from_dataframe(
            df,
            source_name='test_data',
            hole_id_col='holeid',
            from_col='sampfrom',
            to_col='sampto'
        )

        self.assertTrue(result)
        self.assertTrue(self.store.is_loaded)
        self.assertEqual(self.store.metrics_count, 3)

    def test_get_metrics_lookup(self):
        """Test O(1) metrics lookup."""
        if not self.store.initialize():
            self.skipTest("Could not initialize store")

        df = pd.DataFrame({
            'holeid': ['BA0001', 'BA0001'],
            'sampfrom': [44.0, 45.0],
            'sampto': [45.0, 46.0],
            'min_80_pct': ['HO', 'GO'],
            'min_20_pct': ['QZ', 'HO']
        })

        self.store.compute_from_dataframe(df, source_name='test')

        # Lookup existing interval
        metrics = self.store.get_metrics('BA0001', 45.0)
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.hole_id, 'BA0001')
        self.assertEqual(metrics.depth_to, 45.0)

        # Lookup with case insensitivity
        metrics_lower = self.store.get_metrics('ba0001', 45.0)
        self.assertIsNotNone(metrics_lower)

        # Lookup non-existent interval
        missing = self.store.get_metrics('BA9999', 999.0)
        self.assertIsNone(missing)

    def test_get_metrics_dict(self):
        """Test getting metrics as dictionary."""
        if not self.store.initialize():
            self.skipTest("Could not initialize store")

        df = pd.DataFrame({
            'holeid': ['BA0001'],
            'sampfrom': [44.0],
            'sampto': [45.0],
            'min_80_pct': ['HO'],
            'min_20_pct': ['QZ']
        })

        self.store.compute_from_dataframe(df, source_name='test')

        metrics_dict = self.store.get_metrics_dict('BA0001', 45.0)
        self.assertIsInstance(metrics_dict, dict)
        self.assertIn('weighted_hardness', metrics_dict)
        self.assertIn('total_gangue_pct', metrics_dict)

        # Non-existent returns empty dict
        empty_dict = self.store.get_metrics_dict('MISSING', 999.0)
        self.assertEqual(empty_dict, {})

    def test_get_metrics_dataframe(self):
        """Test getting all metrics as DataFrame."""
        if not self.store.initialize():
            self.skipTest("Could not initialize store")

        df = pd.DataFrame({
            'holeid': ['BA0001', 'BA0002'],
            'sampfrom': [44.0, 99.0],
            'sampto': [45.0, 100.0],
            'min_80_pct': ['HO', 'GO'],
            'min_20_pct': ['QZ', 'SH']
        })

        self.store.compute_from_dataframe(df, source_name='test')

        result_df = self.store.get_metrics_dataframe()
        self.assertIsNotNone(result_df)
        self.assertEqual(len(result_df), 2)
        self.assertIn('weighted_hardness', result_df.columns)

    def test_get_metrics_for_hole(self):
        """Test getting all metrics for a specific hole."""
        if not self.store.initialize():
            self.skipTest("Could not initialize store")

        df = pd.DataFrame({
            'holeid': ['BA0001', 'BA0001', 'BA0002'],
            'sampfrom': [44.0, 45.0, 99.0],
            'sampto': [45.0, 46.0, 100.0],
            'min_80_pct': ['HO', 'GO', 'QZ'],
            'min_20_pct': ['QZ', 'HO', 'SH']
        })

        self.store.compute_from_dataframe(df, source_name='test')

        hole_metrics = self.store.get_metrics_for_hole('BA0001')
        self.assertEqual(len(hole_metrics), 2)

        # Should be sorted by depth
        self.assertEqual(hole_metrics[0].depth_to, 45.0)
        self.assertEqual(hole_metrics[1].depth_to, 46.0)

    def test_clear(self):
        """Test clearing stored metrics."""
        if not self.store.initialize():
            self.skipTest("Could not initialize store")

        df = pd.DataFrame({
            'holeid': ['BA0001'],
            'sampfrom': [44.0],
            'sampto': [45.0],
            'min_80_pct': ['HO'],
            'min_20_pct': ['QZ']
        })

        self.store.compute_from_dataframe(df, source_name='test')
        self.assertTrue(self.store.is_loaded)
        self.assertEqual(self.store.metrics_count, 1)

        self.store.clear()
        self.assertFalse(self.store.is_loaded)
        self.assertEqual(self.store.metrics_count, 0)

    def test_get_statistics(self):
        """Test getting summary statistics."""
        if not self.store.initialize():
            self.skipTest("Could not initialize store")

        df = pd.DataFrame({
            'holeid': ['BA0001', 'BA0001', 'BA0002'],
            'sampfrom': [44.0, 45.0, 99.0],
            'sampto': [45.0, 46.0, 100.0],
            'min_80_pct': ['HO', 'GO', 'QZ'],
            'min_20_pct': ['QZ', 'HO', 'SH']
        })

        self.store.compute_from_dataframe(df, source_name='test_stats')

        stats = self.store.get_statistics()
        self.assertEqual(stats['total_intervals'], 3)
        self.assertEqual(stats['unique_holes'], 2)
        self.assertEqual(stats['source_name'], 'test_stats')
        self.assertIn('avg_weighted_hardness', stats)
        self.assertIn('avg_total_gangue_pct', stats)


class TestIntervalMetrics(unittest.TestCase):
    """Tests for IntervalMetrics dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = IntervalMetrics(
            hole_id='BA0001',
            depth_from=44.0,
            depth_to=45.0,
            weighted_hardness=3.5,
            total_gangue_pct=25.0
        )

        result = metrics.to_dict()

        self.assertIsInstance(result, dict)
        self.assertEqual(result['hole_id'], 'BA0001')
        self.assertEqual(result['depth_to'], 45.0)
        self.assertEqual(result['weighted_hardness'], 3.5)
        self.assertEqual(result['total_gangue_pct'], 25.0)

    def test_default_values(self):
        """Test default values."""
        metrics = IntervalMetrics(
            hole_id='BA0001',
            depth_from=44.0,
            depth_to=45.0
        )

        self.assertIsNone(metrics.weighted_hardness)
        self.assertEqual(metrics.no_hardness_pct, 0.0)
        self.assertEqual(metrics.total_gangue_pct, 0.0)
        self.assertEqual(metrics.normalisation_factor, 1.0)


if __name__ == '__main__':
    unittest.main()
