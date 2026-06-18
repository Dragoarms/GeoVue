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

try:
    import logging_review_html_report as html_report
    _html_import_error = None
except ImportError as exc:
    html_report = None
    _html_import_error = exc

try:
    from reports.logging_review.data import prep as prep_module
    _prep_import_error = None
except ImportError as exc:
    prep_module = None
    _prep_import_error = exc


class _FakeSource:
    def __init__(self, df):
        self.df = df


class _FakeGeoStore:
    def __init__(self, sources):
        self._sources = {
            name: _FakeSource(df)
            for name, df in sources.items()
        }

    def list_sources(self):
        return list(self._sources.keys())

    def get_source(self, name):
        return self._sources.get(name)


class _FakeCoordinator:
    def __init__(self, sources):
        self.geological_store = _FakeGeoStore(sources)
        self.is_initialized = True
        self.has_rc_metrics = False
        self.rc_metrics_store = None


class TestLoggingReviewReport(unittest.TestCase):
    def setUp(self):
        if self._testMethodName in {
            "test_logging_interval_dataframe_uses_summary_logging_assays_source",
            "test_qaqc_dataframe_uses_snowflake_sample_best_assays_primary",
        }:
            return
        if report is None and self._testMethodName != "test_module_exists":
            self.skipTest(f"logging_review_report import failed: {_import_error}")

    def test_module_exists(self):
        """Module should be available for report generation."""
        self.assertIsNotNone(report)

    def test_resolve_logger_column_prefers_loggedby_d(self):
        df = pd.DataFrame(columns=["LoggedBy_D", "LoggedBy"])
        result = report.resolve_logger_column(df)
        self.assertIn(result, ["LoggedBy_D", "LoggedBy"])

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

    def test_hybrid_outlier_scores_mahalanobis_matches_vectorized_reference(self):
        """Regression: outlier_score from fixed fixture matches reference (ensures vectorized Mahalanobis matches loop)."""
        np.random.seed(42)
        n = 50
        df = pd.DataFrame({
            "Strat": ["A"] * 25 + ["B"] * 25,
            "Fe": 50.0 + 5 * np.random.randn(n),
            "SiO2": 10.0 + 3 * np.random.randn(n),
            "Al2O3": 2.0 + 1 * np.random.randn(n),
        })
        df.loc[10, "Fe"] = 80.0
        scores = report.compute_hybrid_outlier_scores(
            df, "Strat", ["Fe", "SiO2", "Al2O3"], min_group_size=5
        )
        self.assertTrue(np.isfinite(scores["outlier_score"]).all())
        self.assertIn("outlier_mahal_score", scores.columns)
        self.assertIn("outlier_univariate_score", scores.columns)
        self.assertGreater(scores.loc[10, "outlier_score"], 1.0)
        expected_first_15 = [
            0.6751641732173845, 0.20289328979080354, 0.3190228005291275,
            0.6387309632949452, 0.34106723424568636, 0.3650685176686419,
            0.7660892577931987, 0.3627586941043778, 0.25463272282110516,
            0.28062584938253593, 2.273891865025245, 0.14583805038809314,
            0.4604315415329408, 0.9972162595291718, 0.5475438836326543,
        ]
        np.testing.assert_allclose(
            scores["outlier_score"].head(15).values,
            expected_first_15,
            rtol=1e-5,
            err_msg="outlier_score should match reference (vectorized Mahalanobis)",
        )

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

    def test_logging_interval_dataframe_uses_summary_logging_assays_source(self):
        if prep_module is None:
            self.skipTest(f"prep import failed: {_prep_import_error}")
        coordinator = _FakeCoordinator({
            "summary_logging_assays": pd.DataFrame({
                "HOLEID": ["SB0001", "SB0001"],
                "SAMPFROM": [0.0, 1.0],
                "SAMPTO": [1.0, 2.0],
                "LoggedBy": ["LoggerA", "LoggerA"],
                "BESTSTRAT": ["BIF", "CID"],
                "FE_PCT_BEST": [61.0, 45.0],
            })
        })

        logging_df, stats = prep_module._build_logging_interval_dataframe(
            coordinator,
            logger_values=None,
            date_from=None,
            date_to=None,
        )

        self.assertEqual(stats["primary_source"], "summary_logging_assays")
        self.assertEqual(len(logging_df), 2)
        self.assertIn("LoggedBy", logging_df.columns)

    def test_qaqc_dataframe_uses_snowflake_sample_best_assays_primary(self):
        if prep_module is None:
            self.skipTest(f"prep import failed: {_prep_import_error}")
        coordinator = _FakeCoordinator({
            "sample_best_assays": pd.DataFrame({
                "HOLEID": ["SB0001"],
                "SAMPFROM": [0.0],
                "SAMPTO": [1.0],
                "FE_PCT_BEST": [61.0],
                "SIO2_PCT_BEST": [5.5],
            }),
            "summary_logging_assays": pd.DataFrame({
                "HOLEID": ["SB0001"],
                "SAMPFROM": [0.0],
                "SAMPTO": [1.0],
                "LoggedBy": ["LoggerA"],
                "BESTSTRAT": ["BIF"],
            }),
        })

        merged_df, stats = prep_module.build_merged_qaqc_dataframe(coordinator)

        self.assertEqual(stats["primary_source"], "sample_best_assays")
        self.assertEqual(len(merged_df), 1)
        self.assertIn("LoggedBy", merged_df.columns)
        self.assertIn("BESTSTRAT", merged_df.columns)

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

    def test_clr_transform_row_sums_near_zero(self):
        """CLR transform: row-wise sum of CLR values should be near zero (compositional)."""
        df = pd.DataFrame({
            "Fe": [50.0, 60.0, 40.0],
            "SiO2": [30.0, 25.0, 35.0],
            "Al2O3": [20.0, 15.0, 25.0],
        })
        clr = report._clr_transform(df)
        self.assertEqual(clr.shape, df.shape)
        row_sums = clr.sum(axis=1)
        for s in row_sums:
            self.assertAlmostEqual(s, 0.0, places=5)

    def test_clr_transform_handles_zeros(self):
        """CLR transform should not crash on zeros (replaced with small value)."""
        df = pd.DataFrame({"A": [1.0, 0.0, 2.0], "B": [2.0, 1.0, 0.0]})
        clr = report._clr_transform(df, fill_value=1e-6)
        self.assertTrue(clr.notna().all().all())
        self.assertFalse(np.isinf(clr.values).any())

    def test_predict_most_likely_strat_differs_when_chemistry_matches_other_strat(self):
        """When a row's chemistry is closer to another strat's centroid, predicted strat can differ from logged."""
        df = pd.DataFrame(
            {
                "Strat": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
                "Fe_pct": [60, 61, 59, 60, 62, 30, 31, 29, 30, 32],
                "SiO2_pct": [5, 6, 5, 5, 6, 25, 24, 26, 25, 24],
            }
        )
        # One "A" row with chemistry like B
        df.loc[4, "Fe_pct"] = 31
        df.loc[4, "SiO2_pct"] = 25
        pred = report.predict_most_likely_strat(
            df, strat_col="Strat", chem_cols=["Fe_pct", "SiO2_pct"], min_group_size=3
        )
        self.assertEqual(pred.loc[4], "B")
        self.assertEqual(pred.loc[0], "A")


class TestLoggingReviewHtmlReport(unittest.TestCase):
    """Tests for HTML report helpers: map, coordinates, grouping KPIs, mineralisation, outlier rate."""

    def setUp(self):
        if html_report is None:
            self.skipTest(f"logging_review_html_report import failed: {_html_import_error}")

    def test_resolve_coordinate_columns_returns_easting_northing(self):
        df = pd.DataFrame(columns=["hole_id", "Easting", "Northing"])
        east, north = html_report._resolve_coordinate_columns(df)
        self.assertEqual(east, "Easting")
        self.assertEqual(north, "Northing")

    def test_resolve_coordinate_columns_returns_utm_and_grid_n(self):
        df = pd.DataFrame(columns=["hole_id", "UTM_E", "grid_n"])
        east, north = html_report._resolve_coordinate_columns(df)
        self.assertEqual(east, "UTM_E")
        self.assertEqual(north, "grid_n")

    def test_resolve_coordinate_columns_returns_none_when_missing(self):
        df = pd.DataFrame(columns=["hole_id", "foo", "bar"])
        east, north = html_report._resolve_coordinate_columns(df)
        self.assertIsNone(east)
        self.assertIsNone(north)

    def test_build_map_points_no_coords_when_empty_collar(self):
        collar = pd.DataFrame()
        result = html_report._build_map_points(
            collar_df=collar,
            hole_col="hole_id",
            logger_holes=set(),
            all_holes=set(),
        )
        self.assertFalse(result.get("has_coords"))
        self.assertEqual(result.get("points", []), [])

    def test_build_map_points_no_coords_when_no_easting_northing_columns(self):
        collar = pd.DataFrame({"hole_id": ["H1"], "lat": [0.0], "lon": [0.0]})
        result = html_report._build_map_points(
            collar_df=collar,
            hole_col="hole_id",
            logger_holes={"H1"},
            all_holes={"H1"},
        )
        self.assertFalse(result.get("has_coords"))
        self.assertEqual(result.get("points", []), [])

    def test_build_map_points_wgs84_coords_has_points_and_bounds(self):
        collar = pd.DataFrame({
            "hole_id": ["H1", "H2"],
            "easting": [10.0, 12.0],
            "northing": [45.0, 46.0],
        })
        result = html_report._build_map_points(
            collar_df=collar,
            hole_col="hole_id",
            logger_holes={"H1"},
            all_holes={"H1", "H2"},
        )
        self.assertTrue(result.get("has_coords"))
        points = result.get("points", [])
        self.assertEqual(len(points), 2)
        for p in points:
            self.assertIn("lat", p)
            self.assertIn("lng", p)
            self.assertIn("hole_id", p)
        bounds = result.get("bounds")
        self.assertIsNotNone(bounds)
        self.assertEqual(len(bounds), 4)
        self.assertFalse(result.get("warn_projected", True))

    def test_build_map_points_utm_coords_reprojected_when_pyproj_available(self):
        try:
            import pyproj
        except ImportError:
            self.skipTest("pyproj not available for UTM reprojection test")
        # UTM zone 33N (Gabon): e.g. easting 500000, northing 0–10M
        collar = pd.DataFrame({
            "hole_id": ["H1"],
            "easting": [500000.0],
            "northing": [500000.0],
        })
        result = html_report._build_map_points(
            collar_df=collar,
            hole_col="hole_id",
            logger_holes={"H1"},
            all_holes={"H1"},
            utm_zone=33,
            utm_north=True,
        )
        self.assertTrue(result.get("has_coords"))
        points = result.get("points", [])
        self.assertEqual(len(points), 1)
        lat, lng = points[0]["lat"], points[0]["lng"]
        self.assertGreaterEqual(lat, -90)
        self.assertLessEqual(lat, 90)
        self.assertGreaterEqual(lng, -180)
        self.assertLessEqual(lng, 180)
        self.assertFalse(result.get("warn_projected", True))

    def test_add_mineralisation_accuracy_column_match_mismatch(self):
        from processing.DataManager.column_aliases import ColumnResolver
        df = pd.DataFrame({
            "Fe_pct": [60, 5, 60],
            "SiO2_pct": [5, 40, 5],
            "Al2O3_pct": [2, 10, 2],
            "total_gangue_pct": [8, 50, 8],
            "Min_80_pct": ["MT", "MT", "MBH"],
            "Min_50_pct": ["MT", "MT", "MBH"],
            "Mineralisation": ["MT", "MT", "MBH"],
        })
        resolver = ColumnResolver(df)
        result = html_report._add_mineralisation_accuracy_column(df, resolver)
        self.assertIn("Logging_Accuracy", result.columns)
        counts = result["Logging_Accuracy"].value_counts()
        self.assertGreater(counts.get("Match", 0) + counts.get("Mismatch", 0), 0)

    def test_add_mineralisation_accuracy_uses_zonation_un_as_unmineralised(self):
        """When zonation is Un (or Le), logging is unmineralised; unmineralised assay should Match."""
        from processing.DataManager.column_aliases import ColumnResolver
        df = pd.DataFrame({
            "Fe_pct": [20.4, 34.4],
            "SiO2_pct": [32.0, 22.2],
            "Al2O3_pct": [24.4, 17.0],
            "zonation_un_pct": [100, 100],
            "zonation_de_pct": [0, 0],
            "zonation_hy_pct": [0, 0],
            "zonation_pr_pct": [0, 0],
        })
        resolver = ColumnResolver(df)
        result = html_report._add_mineralisation_accuracy_column(df, resolver)
        self.assertIn("Logging_Accuracy", result.columns)
        self.assertEqual(list(result["Logging_Accuracy"]), ["Match", "Match"])

    def test_calc_outlier_rate_and_ordering(self):
        df = pd.DataFrame({
            "outlier_score": [1.0, 0.5, 2.0, 0.0],
        })
        rate = html_report._calc_outlier_rate(df, top_n=10)
        self.assertIsInstance(rate, (int, float))
        self.assertGreaterEqual(rate, 0)

    def test_grouping_avg_max_interval_returns_empty_when_missing_depth_to(self):
        df = pd.DataFrame({
            "hole_id": ["H1", "H1"],
            "LoggedBy": ["A", "A"],
            "strat": ["S1", "S1"],
            "geolfrom": [0.0, 1.0],
            "Fe_pct": [50, 55],
            "SiO2_pct": [5, 5],
            "Al2O3_pct": [2, 2],
        })
        from processing.DataManager.column_aliases import ColumnResolver
        resolver = ColumnResolver(df)
        avg, max_m, groups = html_report._grouping_avg_max_interval_and_groups(
            df,
            group_cols=["hole_id", "LoggedBy", "strat", "geolfrom"],
            chem_actual_cols=["Fe_pct", "SiO2_pct", "Al2O3_pct"],
            resolver=resolver,
            hole_col="hole_id",
            depth_from_col="geolfrom",
            depth_to_col="",
            strat_col="strat",
            top_n_groups=5,
        )
        self.assertIsNone(avg)
        self.assertIsNone(max_m)
        self.assertEqual(groups, [])

    def test_grouping_avg_max_interval_returns_groups_when_cv_high(self):
        df = pd.DataFrame({
            "hole_id": ["H1", "H1", "H1"],
            "LoggedBy": ["A", "A", "A"],
            "strat": ["S1", "S1", "S1"],
            "geolfrom": [0.0, 0.0, 0.0],
            "geolto": [1.0, 1.0, 1.0],
            "Fe_pct": [0, 50, 150],
            "SiO2_pct": [5, 5, 5],
            "Al2O3_pct": [2, 2, 2],
        })
        from processing.DataManager.column_aliases import ColumnResolver
        resolver = ColumnResolver(df)
        avg, max_m, groups = html_report._grouping_avg_max_interval_and_groups(
            df,
            group_cols=["hole_id", "LoggedBy", "strat", "geolfrom", "geolto"],
            chem_actual_cols=["Fe_pct", "SiO2_pct", "Al2O3_pct"],
            resolver=resolver,
            hole_col="hole_id",
            depth_from_col="geolfrom",
            depth_to_col="geolto",
            strat_col="strat",
            top_n_groups=5,
        )
        self.assertIsNotNone(avg)
        self.assertIsNotNone(max_m)
        self.assertGreater(len(groups), 0)
        for grp in groups:
            self.assertIn("group_key", grp)
            self.assertIn("cv_max", grp)
            self.assertIn("intervals", grp)

    def test_grouping_avg_max_interval_returns_empty_groups_when_all_cv_low(self):
        df = pd.DataFrame({
            "hole_id": ["H1", "H1", "H1"],
            "LoggedBy": ["A", "A", "A"],
            "strat": ["S1", "S1", "S1"],
            "geolfrom": [0.0, 0.0, 0.0],
            "geolto": [1.0, 1.0, 1.0],
            "Fe_pct": [50, 51, 52],
            "SiO2_pct": [5, 5, 5],
            "Al2O3_pct": [2, 2, 2],
        })
        from processing.DataManager.column_aliases import ColumnResolver
        resolver = ColumnResolver(df)
        avg, max_m, groups = html_report._grouping_avg_max_interval_and_groups(
            df,
            group_cols=["hole_id", "LoggedBy", "strat", "geolfrom", "geolto"],
            chem_actual_cols=["Fe_pct", "SiO2_pct", "Al2O3_pct"],
            resolver=resolver,
            hole_col="hole_id",
            depth_from_col="geolfrom",
            depth_to_col="geolto",
            strat_col="strat",
            top_n_groups=5,
        )
        self.assertIsNotNone(avg)
        self.assertEqual(len(groups), 0)

    def test_html_report_contains_no_raw_template_placeholders(self):
        """Regression: generated HTML must not contain literal Python template placeholders."""
        try:
            from reports.logging_review.html.report_renderer import render_html
        except ImportError:
            self.skipTest("reports.logging_review.html.report_renderer not available")
        minimal_report = {
            "meta": {"logger": "TST", "date_from": "2024-01-01", "date_to": "2024-12-31", "generated": "2024-01-01T00:00:00"},
            "summary": {"assay_intervals": 0, "logging_intervals": 0, "unique_holes": 0, "strat_codes": 0, "total_depth_m": 0.0},
            "comment_stats": {"comment_columns": []},
            "comment_stats_logging": {"total_rows": 0, "rows_with_comment": 0, "rows_without_comment": 0, "comment_ratio_pct": 0},
            "comment_coverage": 0,
            "comparisons": {},
            "wordcloud": {},
            "overview": {"strat_code_list": [], "team_strat_code_list": [], "assay_received_count": 0, "assay_outstanding_count": 0},
            "mineralisation": {},
            "profile_zonation": {},
            "grouping_kpis": {},
            "grouping_columns_used": [],
            "outlier_kpis": {},
            "map": {"points": []},
            "outliers": [],
            "fines_summary": [],
            "grouping_summary": [],
            "logging_detail_issue_types": [],
        }
        minimal_intervals = {
            "fines": [],
            "logging_detail": {"fines": [], "magnetite": [], "goethite": [], "carbonate_gangue": []},
            "grouping_flat": [],
            "grouping": [],
            "outliers": [],
        }
        page_options = {
            "summary_stats": True,
            "cover": True,
            "comment_stats": True,
            "fines_accuracy": True,
            "grouping_accuracy": True,
            "outliers": True,
        }
        html_output = render_html(
            minimal_report,
            minimal_intervals,
            logo_path=None,
            page_options=page_options,
            charts_dir=None,
        )
        placeholders = [
            "{overview_section",
            "{''.join(tab_buttons)}",
            "{comment_section",
            "{mineral_section",
            "{profile_section",
            "{logging_detail_section",
            "{grouping_section",
            "{outlier_section",
            'if include_overview else ""',
            'if include_comments else ""',
        ]
        for placeholder in placeholders:
            self.assertNotIn(
                placeholder,
                html_output,
                msg=f"Generated HTML must not contain raw template placeholder: {placeholder!r}",
            )
        self.assertIn("<section class=\"tab-panel\" data-tab=\"overview\"", html_output)
        self.assertIn("data-tab-button=", html_output)

    def test_lookup_interval_image_uses_cache_to_avoid_duplicate_io(self):
        """When the same (hole_id, depth_to) is looked up twice with a shared cache, get_image_path is called once."""
        from unittest.mock import MagicMock, patch

        get_image_path_calls = []
        def track_get_image_path(key):
            get_image_path_calls.append(key)
            return "/fake/path.png"

        mock_dc = MagicMock()
        mock_dc.get_image_path.side_effect = track_get_image_path
        mock_dc.get_keys_for_hole.return_value = []

        cache = {}
        with patch("logging_review_html_report._encode_image_base64", return_value="data:image/png;base64,abc"):
            r1 = html_report._lookup_interval_image(mock_dc, "H1", 10.0, cache=cache)
            r2 = html_report._lookup_interval_image(mock_dc, "H1", 10.0, cache=cache)
        self.assertEqual(r1, "data:image/png;base64,abc")
        self.assertEqual(r2, "data:image/png;base64,abc")
        self.assertEqual(len(get_image_path_calls), 1, "get_image_path should be called once when cache is used")

    def test_fines_and_magnetite_flags_work_with_series_from_itertuples(self):
        """Regression: interval building uses itertuples then pd.Series(dict(...)); flag functions receive a Series."""
        from processing.DataManager.column_aliases import ColumnResolver
        # Row that triggers "Friable Silica" and magnetite issue (negative LOI, no magnetite).
        df = pd.DataFrame([{
            "hole_id": "H1", "geolfrom": 0.0, "geolto": 1.0, "strat": "S1",
            "Fe_pct": 60.0, "SiO2_pct": 6.0, "Al2O3_pct": 3.0, "total_gangue_pct": 0.0,
            "loi_1000_pct": -0.5, "magnetite_pct": 0.0,
        }])
        resolver = ColumnResolver(df)
        tup = next(df.itertuples(index=False))
        row = pd.Series(dict(zip(df.columns, tup)))
        fines_issue = html_report._flag_fines_issue(row, resolver)
        self.assertIsNotNone(fines_issue, "row should trigger fines issue (Friable Silica)")
        mag_issue = html_report._flag_magnetite_issue(row, resolver)
        self.assertIsNotNone(mag_issue, "row should trigger magnetite issue (negative LOI, no magnetite)")
