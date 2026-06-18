import math
import time

import numpy as np
import pandas as pd
import pytest

from processing.logging_review_html_report import _calc_mineral_assay_ratio
from reports.logging_review.data.outliers import (
    OUTLIER_DISPLAY_THRESHOLD,
    _clr_transform,
    _mahalanobis_distance_masked,
    compute_hybrid_outlier_scores,
    predict_most_likely_strat,
)
from reports.logging_review.data.population_stats import PopulationStats
from reports.logging_review.data.prep import fill_empty_logger_values
from reports.logging_review.html.collar_map import _build_map_points
from reports.logging_review.html.report_builder import (
    _metric_total_depth,
    _mineral_assay_coverage_counts,
)
from reports.logging_review.html.tables import _outlier_significance_from_reason


CHEM = ["Fe_pct", "SiO2_pct", "Al2O3_pct", "P_pct"]


def _strat_population_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    centers = {
        "A": np.array([62.0, 5.0, 2.0, 0.05]),
        "B": np.array([35.0, 28.0, 9.0, 0.10]),
        "C": np.array([45.0, 12.0, 24.0, 0.20]),
    }
    rows = []
    for strat, center in centers.items():
        for i in range(36):
            values = center + rng.normal(0, [0.8, 0.5, 0.4, 0.01])
            rows.append({"Strat": strat, "LoggedBy": f"good-{strat}", **dict(zip(CHEM, values))})
    for i in range(12):
        values = centers["A"] + rng.normal(0, [0.5, 0.3, 0.3, 0.01])
        rows.append({"Strat": "B", "LoggedBy": "bad-logger", **dict(zip(CHEM, values))})
    return pd.DataFrame(rows)


def test_population_prediction_flags_bad_logger_rows_but_subset_prediction_does_not():
    df = _strat_population_frame()
    bad_idx = df.index[df["LoggedBy"] == "bad-logger"]

    population_pred = predict_most_likely_strat(df, "Strat", CHEM, min_group_size=10)
    assert set(population_pred.loc[bad_idx]) == {"A"}

    old_subset_pred = predict_most_likely_strat(
        df.loc[bad_idx].copy(),
        "Strat",
        CHEM,
        min_group_size=10,
    )
    assert set(old_subset_pred) == {"B"}


def test_hybrid_score_flags_multivariate_and_univariate_outliers_on_same_scale():
    rng = np.random.default_rng(11)
    cols = [f"E{i}" for i in range(6)]
    base = rng.normal(10.0, 0.30, size=(600, len(cols)))
    df = pd.DataFrame(base, columns=cols)
    df["Strat"] = "A"

    multi_idx = len(df)
    df.loc[multi_idx, cols] = np.array([10.55, 10.55, 10.55, 9.45, 9.45, 9.45])
    df.loc[multi_idx, "Strat"] = "A"

    uni_idx = len(df)
    df.loc[uni_idx, cols] = np.full(len(cols), 10.0)
    df.loc[uni_idx, "E0"] = 13.0
    df.loc[uni_idx, "Strat"] = "A"

    scores = compute_hybrid_outlier_scores(df, "Strat", cols, min_group_size=20)
    assert scores.loc[multi_idx, "outlier_score"] > OUTLIER_DISPLAY_THRESHOLD
    assert scores.loc[multi_idx, "outlier_mahal_score"] > scores.loc[multi_idx, "outlier_univariate_score"]
    assert scores.loc[uni_idx, "outlier_score"] > OUTLIER_DISPLAY_THRESHOLD
    assert scores.loc[uni_idx, "outlier_univariate_score"] > scores.loc[uni_idx, "outlier_mahal_score"]

    inlier_components = scores.iloc[:600][["outlier_mahal_score", "outlier_univariate_score"]]
    med_diff = abs(inlier_components["outlier_mahal_score"].median() - inlier_components["outlier_univariate_score"].median())
    iqr_mahal = inlier_components["outlier_mahal_score"].quantile(0.75) - inlier_components["outlier_mahal_score"].quantile(0.25)
    iqr_uni = inlier_components["outlier_univariate_score"].quantile(0.75) - inlier_components["outlier_univariate_score"].quantile(0.25)
    print(
        "component medians/IQRs",
        inlier_components["outlier_mahal_score"].median(),
        inlier_components["outlier_univariate_score"].median(),
        iqr_mahal,
        iqr_uni,
    )
    assert med_diff < 0.45
    assert abs(iqr_mahal - iqr_uni) < 0.45


def _old_clr_transform(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy().astype(float)
    positive = out[out > 0].min().min()
    fill_value = min(positive / 2.0, 1e-6) if pd.notna(positive) and positive > 0 else 1e-6
    out = out.replace(0, np.nan).fillna(fill_value).clip(lower=1e-12)
    log_data = np.log(out.values)
    gm = np.nanmean(log_data, axis=1, keepdims=True)
    return pd.DataFrame(log_data - gm, index=data.index, columns=data.columns)


def _old_mahal_ratio_for_last_row(df: pd.DataFrame, cols) -> float:
    clr = _old_clr_transform(df[cols].fillna(0.0)).replace([np.inf, -np.inf], np.nan)
    centered = clr - clr.mean(axis=0, skipna=True)
    cov = np.cov(centered.fillna(0.0).T, bias=True)
    cov = np.atleast_2d(cov)
    inv_cov = np.linalg.pinv((1 - 0.05) * cov + 0.05 * np.eye(cov.shape[0]))
    x = centered.fillna(0.0).to_numpy()
    mahal = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", x, inv_cov, x), 0.0))
    p95 = pd.Series(mahal).quantile(0.95)
    return float(mahal[-1] / p95)


def test_zero_below_detection_no_longer_creates_artificial_high_outlier():
    n = 100
    df = pd.DataFrame(
        {
            "Strat": ["A"] * (n + 1),
            "Fe": [60.0] * (n + 1),
            "SiO2": np.geomspace(0.01, 1.0, n).tolist() + [0.0],
            "Al2O3": [40.0] * (n + 1),
        }
    )
    scores = compute_hybrid_outlier_scores(df, "Strat", ["Fe", "SiO2", "Al2O3"], min_group_size=10)
    assert scores.iloc[-1]["outlier_score"] < OUTLIER_DISPLAY_THRESHOLD
    assert _old_mahal_ratio_for_last_row(df, ["Fe", "SiO2", "Al2O3"]) > 3.0


def test_clr_zero_replacement_is_finite_and_sane_for_percent_scale_data():
    df = pd.DataFrame(
        {
            "A": [10.0, 10.0, 10.0, 0.0],
            "B": [20.0, 20.0, 20.0, 20.0],
            "C": [70.0, 70.0, 70.0, 70.0],
        }
    )
    clr = _clr_transform(df)
    assert np.isfinite(clr.to_numpy()).all()
    assert clr.min().min() > -8.0


def test_masked_mahalanobis_uses_covariance_submatrix_not_precision_block():
    covariance = np.array(
        [
            [4.0, 3.5, 2.5],
            [3.5, 4.0, 3.0],
            [2.5, 3.0, 4.0],
        ]
    )
    diff = np.array([1.2, np.nan, -0.7])
    mask = np.array([True, False, True])

    actual = _mahalanobis_distance_masked(np.nan_to_num(diff), covariance, mask)
    brute = math.sqrt(
        np.array([1.2, -0.7]).T
        @ np.linalg.pinv(covariance[np.ix_(mask, mask)])
        @ np.array([1.2, -0.7])
    )
    old_precision_block = np.linalg.pinv(covariance)[np.ix_(mask, mask)]
    old_wrong = math.sqrt(np.array([1.2, -0.7]).T @ old_precision_block @ np.array([1.2, -0.7]))

    assert actual == pytest.approx(brute, abs=1e-6)
    assert abs(old_wrong - brute) > 1e-2


def _loop_predict_reference(df: pd.DataFrame, strat_col: str, chem_cols, min_group_size: int = 10) -> pd.Series:
    raw_all = df[chem_cols].apply(pd.to_numeric, errors="coerce")
    clr_all = _clr_transform(raw_all)
    stats = {}
    for strat_value, group in df.groupby(strat_col):
        if len(group) < min_group_size:
            continue
        clr_data = clr_all.loc[group.index]
        clr_data = clr_data.loc[clr_data.notna().any(axis=1)]
        centroid = clr_data.median()
        centered = clr_data - centroid
        cov = np.cov(centered.fillna(0.0).T, bias=True)
        cov = np.atleast_2d(cov)
        diag_mean = np.nanmean(np.diag(cov))
        cov = 0.9 * cov + 0.1 * (diag_mean if diag_mean > 0 else 1.0) * np.eye(cov.shape[0])
        stats[strat_value] = (centroid, cov)

    out = pd.Series("-", index=df.index, dtype=object)
    for idx in df.index:
        row = clr_all.loc[idx]
        best = "-"
        best_dist = float("inf")
        for strat_value, (centroid, cov) in stats.items():
            diff = row - centroid
            mask = diff.notna().to_numpy()
            if not mask.any():
                continue
            dist = _mahalanobis_distance_masked(diff.fillna(0.0).to_numpy(), cov, mask)
            if dist < best_dist:
                best_dist = dist
                best = strat_value
        out.loc[idx] = best
    return out


def test_vectorized_prediction_matches_loop_reference_and_is_fast():
    rng = np.random.default_rng(19)
    centers = {
        "A": np.array([60.0, 5.0, 2.0]),
        "B": np.array([35.0, 25.0, 8.0]),
        "C": np.array([45.0, 12.0, 20.0]),
    }
    rows = []
    for strat, center in centers.items():
        values = center + rng.normal(0, [1.0, 0.8, 0.6], size=(220, 3))
        for row in values:
            rows.append({"Strat": strat, "Fe": row[0], "SiO2": row[1], "Al2O3": row[2]})
    small = pd.DataFrame(rows)

    vector_pred = predict_most_likely_strat(small, "Strat", ["Fe", "SiO2", "Al2O3"], min_group_size=20)
    loop_pred = _loop_predict_reference(small, "Strat", ["Fe", "SiO2", "Al2O3"], min_group_size=20)
    pd.testing.assert_series_equal(vector_pred, loop_pred)

    big_parts = []
    for strat, center in centers.items():
        values = center + rng.normal(0, [1.0, 0.8, 0.6], size=(17000, 3))
        part = pd.DataFrame(values, columns=["Fe", "SiO2", "Al2O3"])
        part["Strat"] = strat
        big_parts.append(part)
    big = pd.concat(big_parts, ignore_index=True)

    t0 = time.perf_counter()
    _ = predict_most_likely_strat(big, "Strat", ["Fe", "SiO2", "Al2O3"], min_group_size=20)
    vector_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    _ = _loop_predict_reference(small, "Strat", ["Fe", "SiO2", "Al2O3"], min_group_size=20)
    loop_seconds = time.perf_counter() - t1

    vector_per_row = vector_seconds / len(big)
    loop_per_row = loop_seconds / len(small)
    print("prediction timings", {"vector_51k": vector_seconds, "loop_660": loop_seconds})
    assert vector_seconds < 5.0
    assert loop_per_row / vector_per_row > 10.0


def test_outlier_significance_defaults_are_not_inflated():
    assert _outlier_significance_from_reason("Fe_pct high (value 65)", "") == "High"
    assert _outlier_significance_from_reason("p_pct high (value 0.2)", "P_pct") == "Low"
    assert _outlier_significance_from_reason("", "") == "Low"
    assert _outlier_significance_from_reason("multivariate chemistry distance high", "") == "Low"


def test_utm_reprojection_requires_supplied_zone_and_uses_that_zone():
    pyproj = pytest.importorskip("pyproj")
    lon, lat = 9.0, 1.0
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    easting, northing = transformer.transform(lon, lat)
    collar = pd.DataFrame({"hole_id": ["H1"], "easting": [easting], "northing": [northing]})

    resolved = _build_map_points(collar, "hole_id", {"H1"}, {"H1"}, utm_zone=32, utm_north=True)
    assert resolved["warn_projected"] is False
    assert resolved["points"][0]["lat"] == pytest.approx(lat, abs=1e-5)
    assert resolved["points"][0]["lng"] == pytest.approx(lon, abs=1e-5)

    no_zone = _build_map_points(collar, "hole_id", {"H1"}, {"H1"})
    assert no_zone["warn_projected"] is True
    assert no_zone["points"] == []


def test_mineral_assay_coverage_counts_only_assayed_mineralised_intervals_and_clamps():
    df = pd.DataFrame(
        {
            "Mineralisation": ["MT", "MBH", "", "MT"],
            "Fe_pct": [60.0, np.nan, 12.0, np.nan],
            "SiO2_pct": [np.nan, 8.0, np.nan, np.nan],
        }
    )
    counts = _mineral_assay_coverage_counts(df, "Mineralisation", ["Fe_pct", "SiO2_pct"])
    assert counts["mineralised_intervals"] == 3
    assert counts["assays_available"] == 2
    assert counts["assay_ratio_pct"] == pytest.approx(66.6666667)
    assert counts["assay_ratio_pct"] <= 100.0
    assert _calc_mineral_assay_ratio(df) == pytest.approx(66.6666667)


def test_total_depth_deduplicates_merge_expanded_physical_intervals():
    df = pd.DataFrame(
        {
            "hole_id": ["H1", "H1", "H1", "H2"],
            "depth_from": [0.0, 0.0, 10.0, 0.0],
            "depth_to": [10.0, 10.0, 20.0, 5.0],
            "assay": [1, 2, 3, 4],
        }
    )
    assert _metric_total_depth(df, "hole_id", "depth_from", "depth_to") == pytest.approx(25.0)


def test_population_threshold_string_is_direction_aware_for_depletion():
    df = pd.DataFrame({"loi_pct": np.linspace(0.0, 9.9, 100)})
    stats = PopulationStats(df)
    assert stats.threshold_str("loi_pct", "High", direction="depleted").startswith("P10=")
    assert stats.threshold_str("loi_pct", "Low", direction="depleted").startswith("P25=")
    assert stats.threshold_str("loi_pct", "High").startswith("P90=")


def test_fill_empty_logger_values_uses_depth_adjacent_order_not_input_order():
    df = pd.DataFrame(
        {
            "hole_id": ["H1", "H1", "H1"],
            "depth_from": [20.0, 10.0, 0.0],
            "LoggedBy": ["LoggerB", "", "LoggerA"],
        }
    )
    filled = fill_empty_logger_values(df, "LoggedBy", "hole_id")
    assert filled.loc[1, "LoggedBy"] == "LoggerA"


def test_shrinkage_target_tracks_covariance_scale():
    from reports.logging_review.data.outliers import _shrink_covariance

    cov = np.diag([100.0, 200.0])
    shrunk = _shrink_covariance(cov, alpha=0.2)
    assert np.diag(shrunk).tolist() == pytest.approx([110.0, 190.0])
