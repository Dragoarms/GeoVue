from gui.DrillholeCorrelation.correlation_viz_settings_dialog import (
    CorrelationVizRow,
    CorrelationVizSettingsDialog,
)


def test_legacy_column_source_spec_is_split_for_settings_dialog():
    assert CorrelationVizRow._split_column_source(
        "fe_pct_best (summary_logging_assays)"
    ) == ("fe_pct_best", "summary_logging_assays")


def test_plain_column_spec_keeps_empty_source():
    assert CorrelationVizRow._split_column_source("sio2_pct_best") == (
        "sio2_pct_best",
        None,
    )


def test_legacy_bare_assay_column_promotes_to_best_column():
    assert CorrelationVizRow._canonical_column_for_source(
        "sio2_pct",
        ["Fe_pct_BEST", "SiO2_pct_BEST", "Al2O3_pct_BEST"],
    ) == "SiO2_pct_BEST"


def test_geophysics_defaults_to_line_and_percentile_scale():
    assert CorrelationVizSettingsDialog._default_viz_type_for_source(
        "geophysicsdetails"
    ) == "line"
    assert CorrelationVizSettingsDialog._default_scale_mode_for_source(
        "geophysicsdetails"
    ) == "per_hole_percentile"
