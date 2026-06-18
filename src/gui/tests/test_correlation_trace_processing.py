import pytest

from gui.DrillholeCorrelation.trace_processing import (
    canonical_scale_mode,
    pearson_similarity,
    resample_depth_points,
    scale_mode_display_name,
    scale_trace_values,
)


def test_raw_scale_preserves_values_and_manual_range():
    result = scale_trace_values(
        [10, 20, 30],
        "raw",
        auto_scale=False,
        min_value=0,
        max_value=100,
    )

    assert result.values == [10.0, 20.0, 30.0]
    assert result.plot_min == 0.0
    assert result.plot_max == 100.0


def test_global_scale_preserves_values_and_uses_global_range():
    result = scale_trace_values(
        [10, 20, 30],
        "global",
        global_min=0,
        global_max=200,
    )

    assert result.values == [10.0, 20.0, 30.0]
    assert result.plot_min == 0.0
    assert result.plot_max == 200.0
    assert result.label == "Global 0-200"


def test_per_hole_percentile_scale_clips_outliers_to_unit_range():
    result = scale_trace_values(
        [0, 10, 20, 1000],
        "per_hole_percentile",
        percentile_low=25,
        percentile_high=75,
    )

    assert result.plot_min == 0.0
    assert result.plot_max == 1.0
    assert result.values[0] == 0.0
    assert result.values[-1] == 1.0


def test_zscore_scale_standardises_trace_shape():
    result = scale_trace_values([10, 20, 30], "zscore")

    assert sum(result.values) == pytest.approx(0.0)
    assert result.values[0] < result.values[1] < result.values[2]
    assert result.label == "Z-score"


def test_scale_mode_label_aliases_round_trip():
    assert canonical_scale_mode("Hole P2-P98") == "per_hole_percentile"
    assert scale_mode_display_name("z_score") == "Z-score"


def test_pearson_similarity_supports_future_trace_matching():
    assert pearson_similarity([0, 1, 2], [10, 11, 12]) == pytest.approx(1.0)
    assert pearson_similarity([0, 1, 2], [12, 11, 10]) == pytest.approx(-1.0)
    assert pearson_similarity([1, 1, 1], [1, 2, 3]) is None


def test_resample_depth_points_builds_uniform_grid_for_similarity():
    depths, values = resample_depth_points(
        [(10.0, 0.0), (10.2, 20.0), (10.4, 40.0)],
        10.0,
        10.4,
        step=0.1,
    )

    assert depths == [10.0, 10.1, 10.2, 10.3, 10.4]
    assert values == pytest.approx([0.0, 10.0, 20.0, 30.0, 40.0])


def test_resample_depth_points_marks_outside_range_missing():
    depths, values = resample_depth_points(
        [(10.1, 10.0), (10.2, 20.0)],
        10.0,
        10.3,
        step=0.1,
    )

    assert depths == [10.0, 10.1, 10.2, 10.3]
    assert values[0] is None
    assert values[1:3] == pytest.approx([10.0, 20.0])
    assert values[3] is None
