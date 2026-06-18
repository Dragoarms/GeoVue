from gui.DrillholeCorrelation.correlation_models import DrillholeInterval
from gui.DrillholeCorrelation.synchronized_zoom_window import SynchronizedZoomWindow


class FakeColumnWidget:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def _get_value_for_interval(self, interval, column_name, source_name=None):
        self.calls.append((interval.depth_to, column_name, source_name))
        return self.value


def make_interval(csv_data=None):
    return DrillholeInterval(
        hole_id="NB0242",
        depth_from=10.0,
        depth_to=11.0,
        csv_data=csv_data or {},
    )


def test_zoom_window_delegates_value_resolution_to_column_widget():
    zoom = object.__new__(SynchronizedZoomWindow)
    column_widget = FakeColumnWidget(17.93)
    interval = make_interval()

    value = zoom._get_interval_value(interval, "sio2_pct_best", column_widget)

    assert value == 17.93
    assert column_widget.calls == [(11.0, "sio2_pct_best", None)]


def test_zoom_window_passes_separate_source_to_column_widget():
    zoom = object.__new__(SynchronizedZoomWindow)
    column_widget = FakeColumnWidget(61.4)
    interval = make_interval()

    value = zoom._get_interval_value(
        interval,
        "fe_pct_best",
        column_widget,
        "sample_best_assays",
    )

    assert value == 61.4
    assert column_widget.calls == [(11.0, "fe_pct_best", "sample_best_assays")]


def test_zoom_window_matches_source_suffixed_csv_fallback():
    zoom = object.__new__(SynchronizedZoomWindow)
    interval = make_interval({"sio2_pct_best (summary_logging_assays)": "9.75"})

    value = zoom._get_interval_value(interval, "sio2_pct_best")

    assert value == 9.75
