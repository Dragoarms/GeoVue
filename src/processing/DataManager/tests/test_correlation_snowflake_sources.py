"""Tests for correlation-facing Snowflake source routing."""

import pandas as pd

from processing.DataManager.data_coordinator import DataCoordinator


def test_snowflake_sources_feed_correlation_collar_and_survey_data():
    coord = DataCoordinator()
    coord.add_source_from_dataframe(
        pd.DataFrame(
            {
                "HOLEID": ["ba001", "ba002"],
                "DEPTH": [100.0, 200.0],
                "BEST_X": [101.0, 200.0],
                "BEST_Y": [1001.0, 2000.0],
                "BEST_Z": [51.0, 60.0],
                "PROJECTCODE": ["BA", "BA"],
                "PlannedHoleID": ["plan-1", "plan-2"],
            }
        ),
        name="collar_sens_gab",
    )
    coord.add_source_from_dataframe(
        pd.DataFrame(
            {
                "HOLEID": ["ba001", "ba001", "ba002"],
                "DEPTH": [10.0, 0.0, 0.0],
                "AZIMUTH": [90.0, 90.0, 180.0],
                "DIP": [-60.0, -60.0, -55.0],
            }
        ),
        name="collar_survey",
    )

    collars = coord.get_collar_data()
    assert list(collars["holeid"]) == ["BA001", "BA002"]
    assert collars.loc[collars["holeid"] == "BA001", "x"].iloc[0] == 101.0
    assert collars.loc[collars["holeid"] == "BA001", "z"].iloc[0] == 51.0
    assert collars.loc[collars["holeid"] == "BA001", "project"].iloc[0] == "BA"
    assert collars.loc[collars["holeid"] == "BA001", "planned_holeid"].iloc[0] == "plan-1"

    survey = coord.get_survey_data()
    assert len(survey) == 3
    assert set(survey.columns) == {"holeid", "depth", "azimuth", "dip"}


def test_snowflake_interval_sources_are_prioritized_for_correlation_intervals():
    coord = DataCoordinator()
    coord.add_source_from_dataframe(
        pd.DataFrame(
            {
                "HOLEID": ["BA001", "BA001"],
                "DEPTH_FROM": [0.0, 1.0],
                "DEPTH_TO": [1.0, 2.0],
                "LITHOLOGY": ["BIF", "SHALE"],
            }
        ),
        name="lithology_diamond",
    )

    intervals = coord.get_hole_intervals("ba001")
    assert len(intervals) == 2
    assert list(intervals["lithology"]) == ["BIF", "SHALE"]


def test_collar_drilling_method_filters_correlation_sources_for_diamond_holes():
    coord = DataCoordinator()
    coord.add_source_from_dataframe(
        pd.DataFrame(
            {
                "HOLEID": ["NBD0001", "NB0001"],
                "DEPTH": [20.0, 20.0],
                "BEST_X": [1.0, 2.0],
                "BEST_Y": [1.0, 2.0],
                "BEST_Z": [1.0, 2.0],
                "DrillingMethod": ["Diamond", "RC"],
            }
        ),
        name="collar_sens_gab",
    )
    coord.add_source_from_dataframe(
        pd.DataFrame(
            {
                "HOLEID": ["NBD0001"],
                "GEOLFROM": [0.0],
                "GEOLTO": [1.0],
                "LITH1": ["BIF"],
            }
        ),
        name="lithology_diamond",
    )
    coord.add_source_from_dataframe(
        pd.DataFrame(
            {
                "HOLEID": ["NB0001"],
                "SAMPFROM": [0.0],
                "SAMPTO": [1.0],
                "FE_PCT_BEST": [58.0],
            }
        ),
        name="summary_logging_assays",
    )
    coord.add_source_from_dataframe(
        pd.DataFrame(
            {
                "HOLEID": ["NBD0001", "NB0001"],
                "DEPTH": [0.5, 0.5],
                "GAMMA_CPS": [100.0, 50.0],
            }
        ),
        name="geophysicsdetails",
    )

    diamond_sources = coord.get_correlation_source_names_for_holes(["NBD0001"])
    rc_sources = coord.get_correlation_source_names_for_holes(["NB0001"])

    assert "lithology_diamond" in diamond_sources
    assert "geophysicsdetails" in diamond_sources
    assert "summary_logging_assays" not in diamond_sources
    assert "summary_logging_assays" in rc_sources
    assert "geophysicsdetails" in rc_sources
    assert "lithology_diamond" not in rc_sources


def test_common_geophysics_source_does_not_make_rc_assays_look_diamond():
    coord = DataCoordinator()
    coord.add_source_from_dataframe(
        pd.DataFrame(
            {
                "HOLEID": ["NB0001"],
                "SAMPFROM": [0.0],
                "SAMPTO": [1.0],
                "FE_PCT_BEST": [58.0],
            }
        ),
        name="summary_logging_assays",
    )
    coord.add_source_from_dataframe(
        pd.DataFrame(
            {
                "HOLEID": ["NB0001"],
                "DEPTH": [0.5],
                "GAMMA_CPS": [50.0],
            }
        ),
        name="geophysicsdetails",
    )

    sources = coord.get_correlation_source_names_for_holes(["NB0001"])

    assert "summary_logging_assays" in sources
    assert "geophysicsdetails" in sources
    assert "lithology_diamond" not in sources


def test_correlation_source_names_cache_collar_method_lookup(monkeypatch):
    coord = DataCoordinator()
    coord.add_source_from_dataframe(
        pd.DataFrame(
            {
                "HOLEID": ["OK0119"],
                "DEPTH": [247.0],
                "BEST_X": [1.0],
                "BEST_Y": [1.0],
                "BEST_Z": [1.0],
                "DrillingMethod": ["RC"],
            }
        ),
        name="collar_sens_gab",
    )
    coord.add_source_from_dataframe(
        pd.DataFrame(
            {
                "HOLEID": ["OK0119"],
                "SAMPFROM": [0.0],
                "SAMPTO": [1.0],
                "FE_PCT_BEST": [55.29],
            }
        ),
        name="summary_logging_assays",
    )

    calls = {"count": 0}
    original_get_collar_data = coord.get_collar_data

    def counted_get_collar_data():
        calls["count"] += 1
        return original_get_collar_data()

    monkeypatch.setattr(coord, "get_collar_data", counted_get_collar_data)

    assert coord.get_correlation_source_names_for_holes(["OK0119"]) == [
        "summary_logging_assays"
    ]
    assert coord.get_correlation_source_names_for_holes(["OK0119"]) == [
        "summary_logging_assays"
    ]
    assert calls["count"] == 1


def test_geophysics_point_values_are_returned_without_aggregation():
    coord = DataCoordinator()
    coord.add_source_from_dataframe(
        pd.DataFrame(
            {
                "HOLEID": ["NB0001", "NB0001", "NB0001"],
                "DEPTH": [10.1, 10.4, 10.9],
                "GAMMA_CPS": [5.0, 120.0, 6.0],
            }
        ),
        name="geophysicsdetails",
    )

    values = coord.get_point_values_for_interval(
        "NB0001",
        10.0,
        11.0,
        "gamma_cps",
        "geophysicsdetails",
    )

    assert values == [(10.1, 5.0), (10.4, 120.0), (10.9, 6.0)]


def test_holes_with_assays_uses_populated_assay_sources():
    coord = DataCoordinator()
    coord.add_source_from_dataframe(
        pd.DataFrame(
            {
                "HOLEID": ["NB0001", "NB0002", "NB0003"],
                "SAMPFROM": [0.0, 0.0, 0.0],
                "SAMPTO": [1.0, 1.0, 1.0],
                "FE_PCT_BEST": [58.0, None, ""],
                "SAMPLETYPE": ["RC Chip", "RC Chip", "RC Chip"],
            }
        ),
        name="summary_logging_assays",
    )
    coord.add_source_from_dataframe(
        pd.DataFrame(
            {
                "HOLEID": ["NBD0001"],
                "GEOLFROM": [0.0],
                "GEOLTO": [1.0],
                "LITH1": ["BIF"],
            }
        ),
        name="lithology_diamond",
    )

    assert coord.get_holes_with_assays() == {"NB0001"}


def test_sens_gab_collar_is_registered_for_phase2_loading():
    from processing.DataManager.snowflake_session import TABLE_REGISTRY

    collar_tables = [
        table for table in TABLE_REGISTRY
        if table.geovue_name == "collar_sens_gab"
    ]
    assert len(collar_tables) == 1
    assert collar_tables[0].phase == 2
    assert collar_tables[0].is_collar
