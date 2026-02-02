import sys
import os
import unittest
import warnings
import pandas as pd
import numpy as np

processing_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.dirname(processing_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, processing_dir)


try:
    import logging_review_report as report
    _import_error = None
except ImportError as exc:
    report = None
    _import_error = exc


class TestLoggingReviewReport(unittest.TestCase):
    def setUp(self):
        if report is None and self._testMethodName != "test_module_exists":
            self.skipTest(f"logging_review_report import failed: {_import_error}")

    def test_module_exists(self):
        """Module should be available for report generation."""
        self.assertIsNotNone(report)

    def test_resolve_logger_column_prefers_loggedby_d(self):
        df = pd.DataFrame(columns=["LoggedBy_D", "LoggedBy"])
        self.assertEqual(report.resolve_logger_column(df), "LoggedBy_D")

    def test_resolve_drilldate_column(self):
        df = pd.DataFrame(columns=["DRILLDATE", "drill_date"])
        self.assertEqual(report.resolve_drilldate_column(df), "DRILLDATE")

    def test_hybrid_outlier_scores_ranks_extreme_interval(self):
        df = pd.DataFrame(
            {
                "Strat": ["A"] * 5 + ["B"] * 5,
                "Fe_pct": [60, 61, 59, 60, 90, 55, 56, 55, 54, 53],
                "SiO2_pct": [5, 6, 5, 5, 40, 10, 11, 10, 9, 8],
            }
        )
        scores = report.compute_hybrid_outlier_scores(
            df,
            strat_col="Strat",
            chem_cols=["Fe_pct", "SiO2_pct"],
            min_group_size=3,
        )
        self.assertGreater(scores.loc[4, "outlier_score"], scores.loc[0, "outlier_score"])
        self.assertIn("Fe_pct", scores.loc[4, "outlier_reason"])

    def test_hybrid_outlier_scores_single_column(self):
        df = pd.DataFrame(
            {
                "Strat": ["A"] * 6,
                "Fe_pct": [60, 61, 59, 60, 62, 58],
            }
        )
        scores = report.compute_hybrid_outlier_scores(
            df,
            strat_col="Strat",
            chem_cols=["Fe_pct"],
            min_group_size=3,
        )
        self.assertEqual(len(scores), len(df))
        self.assertTrue(scores["outlier_score"].notna().all())

    def test_filter_dataframe_by_logger_and_date(self):
        df = pd.DataFrame(
            {
                "LoggedBy": ["A", "B", "A", "C"],
                "drilldate": ["2024-01-05", "2024-01-10", "2024-02-01", "2024-01-20"],
                "value": [1, 2, 3, 4],
            }
        )
        filtered = report.filter_dataframe_by_logger_and_date(
            df=df,
            logger_col="LoggedBy",
            date_col="drilldate",
            logger_values=["A"],
            date_from="2024-01-01",
            date_to="2024-01-31",
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["value"], 1)

    def test_filter_dataframe_by_logger_filters_per_interval(self):
        df = pd.DataFrame(
            {
                "hole_id": ["H1", "H1", "H2", "H3"],
                "depth_from": [0.0, 1.0, 0.0, 0.0],
                "depth_to": [1.0, 2.0, 1.0, 1.0],
                "LoggedBy": ["A", "B", "A", "B"],
                "value": [10, 20, 30, 40],
            }
        )
        filtered = report.filter_dataframe_by_logger_and_date(
            df=df,
            logger_col="LoggedBy",
            date_col="",
            logger_values=["A"],
        )
        self.assertEqual(len(filtered), 2)
        self.assertListEqual(filtered["hole_id"].tolist(), ["H1", "H2"])
        self.assertListEqual(filtered["depth_from"].tolist(), [0.0, 0.0])

    def test_merge_rc_metrics_by_overlap_matches_misaligned_depth_to(self):
        merged_df = pd.DataFrame(
            {
                "hole_id": ["H1"],
                "depth_from": [0.0],
                "depth_to": [1.0],
                "LoggedBy": ["A"],
                "Strat": ["S1"],
            }
        )
        metrics_df = pd.DataFrame(
            {
                "hole_id": ["H1"],
                "depth_from": [0.0],
                "depth_to": [1.5],
                "total_gangue_pct": [12.5],
            }
        )

        result = report._merge_rc_metrics_by_overlap(
            merged_df=merged_df,
            metrics_df=metrics_df,
            hole_col="hole_id",
            depth_from_col="depth_from",
            depth_to_col="depth_to",
            hole_ids={"H1"},
        )

        self.assertEqual(result.loc[0, "total_gangue_pct"], 12.5)

    def test_merge_rc_metrics_handles_hole_upper_column(self):
        merged_df = pd.DataFrame(
            {
                "hole_id": ["H1"],
                "depth_from": [0.0],
                "depth_to": [1.0],
            }
        )
        metrics_df = pd.DataFrame(
            {
                "hole_id": ["H1"],
                "depth_from": [0.0],
                "depth_to": [1.5],
                "_hole_upper": ["H1"],
                "total_gangue_pct": [12.5],
            }
        )
        metrics_df = metrics_df.set_index(["_hole_upper"])

        result = report._merge_rc_metrics_by_overlap(
            merged_df=merged_df,
            metrics_df=metrics_df,
            hole_col="hole_id",
            depth_from_col="depth_from",
            depth_to_col="depth_to",
        )

        self.assertEqual(result.loc[0, "total_gangue_pct"], 12.5)

    def test_merge_logger_by_overlap_adds_loggedby(self):
        merged_df = pd.DataFrame(
            {
                "hole_id": ["H1", "H2"],
                "depth_from": [0.0, 0.0],
                "depth_to": [1.0, 1.0],
            }
        )
        logger_df = pd.DataFrame(
            {
                "hole_id": ["H1"],
                "depth_from": [0.0],
                "depth_to": [1.5],
                "LoggedBy": ["LoggerA"],
            }
        )
        result = report._merge_logger_by_overlap(
            merged_df=merged_df,
            logger_df=logger_df,
            merged_hole_col="hole_id",
            merged_from_col="depth_from",
            merged_to_col="depth_to",
            logger_hole_col="hole_id",
            logger_from_col="depth_from",
            logger_to_col="depth_to",
            hole_ids={"H1"},
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "LoggedBy"], "LoggerA")

    def test_build_boxplot_image_handles_all_nan(self):
        series = pd.Series([np.nan, np.nan, np.nan])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            image = report._build_boxplot_image(series, outlier_value=1.0)
        self.assertIsNotNone(image)

    def test_grouping_accuracy_by_interval_summary(self):
        df = pd.DataFrame(
            {
                "hole_id": ["H1", "H1", "H1"],
                "LoggedBy": ["A", "A", "A"],
                "strat": ["S1", "S1", "S1"],
                "geolfrom": [0.0, 0.0, 0.0],
                "geolto": [1.0, 1.0, 1.0],
                "Fe_pct": [50, 55, 60],
                "SiO2_pct": [5, 5, 5],
                "Al2O3_pct": [2, 2, 2],
            }
        )
        lines = report._summarize_grouping_accuracy_by_interval(
            df,
            group_cols=["hole_id", "LoggedBy", "strat", "geolfrom", "geolto"],
            chem_cols=["Fe_pct", "SiO2_pct", "Al2O3_pct"],
        )
        self.assertTrue(lines and lines[0].startswith("Groups evaluated:"))

    def test_merge_intervals_filters_by_hole_ids(self):
        from processing.DataManager.interval_merger import merge_intervals_by_range

        primary = pd.DataFrame(
            {
                "hole_id": ["H1", "H2"],
                "depth_from": [0.0, 0.0],
                "depth_to": [1.0, 1.0],
            }
        )
        secondary = pd.DataFrame(
            {
                "hole_id": ["H1", "H2"],
                "depth_from": [0.0, 0.0],
                "depth_to": [1.0, 1.0],
                "LoggedBy": ["A", "B"],
            }
        )
        result, _ = merge_intervals_by_range(
            primary_df=primary,
            secondary_df=secondary,
            primary_keys={"hole_id": "hole_id", "from": "depth_from", "to": "depth_to"},
            secondary_keys={"hole_id": "hole_id", "from": "depth_from", "to": "depth_to"},
            cols_to_merge=["LoggedBy"],
            hole_ids={"H1"},
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["hole_id"], "H1")

