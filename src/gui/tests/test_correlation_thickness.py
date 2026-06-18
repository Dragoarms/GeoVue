import math

import pytest

from gui.DrillholeCorrelation.thickness import (
    ThicknessBand,
    apparent_factor_from_dip,
    apparent_thickness_from_true,
    stretch_scale_for_true_thickness,
    summarise_thickness_bands,
    true_thickness_from_apparent,
)


def test_sixty_degree_dip_doubles_apparent_thickness():
    assert apparent_factor_from_dip(60.0) == pytest.approx(2.0)
    assert apparent_thickness_from_true(5.0, 60.0) == pytest.approx(10.0)
    assert true_thickness_from_apparent(10.0, 60.0) == pytest.approx(5.0)
    assert stretch_scale_for_true_thickness(60.0) == pytest.approx(0.5)


def test_zero_dip_leaves_thickness_unchanged():
    assert apparent_factor_from_dip(0.0) == pytest.approx(1.0)
    assert apparent_thickness_from_true(7.5, 0.0) == pytest.approx(7.5)
    assert true_thickness_from_apparent(7.5, 0.0) == pytest.approx(7.5)


def test_apparent_factor_grows_towards_vertical():
    assert apparent_factor_from_dip(80.0) > apparent_factor_from_dip(60.0)
    assert math.isfinite(apparent_factor_from_dip(89.0))


def test_vertical_dip_has_no_finite_apparent_thickness():
    with pytest.raises(ValueError):
        apparent_factor_from_dip(90.0)


def test_manual_thickness_bands_sum_true_thickness_by_local_dip():
    bands = [
        ThicknessBand(10.0, 14.0, 0.0),
        ThicknessBand(14.0, 18.0, 60.0),
    ]

    summary = summarise_thickness_bands(bands)

    assert bands[0].true_thickness == pytest.approx(4.0)
    assert bands[1].true_thickness == pytest.approx(2.0)
    assert summary["total_apparent"] == pytest.approx(8.0)
    assert summary["total_true"] == pytest.approx(6.0)
    assert summary["stretch_scale"] == pytest.approx(0.75)


def test_thickness_band_normalises_reversed_depths():
    band = ThicknessBand(20.0, 10.0, 30.0)

    assert band.depth_from == pytest.approx(10.0)
    assert band.depth_to == pytest.approx(20.0)
    assert band.apparent_thickness == pytest.approx(10.0)
