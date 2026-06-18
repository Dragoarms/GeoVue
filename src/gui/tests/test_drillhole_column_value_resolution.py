from gui.DrillholeCorrelation.correlation_models import DrillholeInterval
from gui.widgets.drillhole_column_widget import DrillholeColumnWidget


class FakeStore:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def get_value(self, key, column, source_name=None):
        self.calls.append((key.hole_id, key.depth_to, column, source_name))
        return self.values.get((column.lower(), source_name))


class FakeDataManager:
    def __init__(self, store):
        self.geological_store = store


def make_widget(values=None):
    widget = object.__new__(DrillholeColumnWidget)
    widget.hole_id = "NB0242"
    widget.data_manager = FakeDataManager(FakeStore(values or {}))
    return widget


def make_interval(csv_data=None):
    return DrillholeInterval(
        hole_id="NB0242",
        depth_from=10.0,
        depth_to=11.0,
        csv_data=csv_data or {},
    )


def test_unqualified_viz_column_searches_geological_store():
    widget = make_widget({("sio2_pct_best", None): "12.5"})

    value = widget._get_value_for_interval(make_interval(), "sio2_pct_best")

    assert value == 12.5
    assert widget.data_manager.geological_store.calls == [
        ("NB0242", 11.0, "sio2_pct_best", None)
    ]


def test_source_qualified_viz_column_keeps_requested_source():
    widget = make_widget({("fe_pct_best", "summary_logging_assays"): "58.2"})

    value = widget._get_value_for_interval(
        make_interval(),
        "fe_pct_best (summary_logging_assays)",
    )

    assert value == 58.2
    assert widget.data_manager.geological_store.calls == [
        ("NB0242", 11.0, "fe_pct_best", "summary_logging_assays")
    ]


def test_separate_viz_source_field_is_used_for_lookup():
    widget = make_widget({
        ("fe_pct_best", None): "10.0",
        ("fe_pct_best", "sample_best_assays"): "61.4",
    })

    value = widget._get_value_for_interval(
        make_interval(),
        "fe_pct_best",
        "sample_best_assays",
    )

    assert value == 61.4
    assert widget.data_manager.geological_store.calls == [
        ("NB0242", 11.0, "fe_pct_best", "sample_best_assays")
    ]


def test_unqualified_viz_column_matches_source_suffixed_csv_key():
    widget = make_widget()

    value = widget._get_value_for_interval(
        make_interval({"sio2_pct_best (summary_logging_assays)": "9.75"}),
        "sio2_pct_best",
    )

    assert value == 9.75


def test_widget_exposes_viz_update_and_collapse_methods():
    assert callable(getattr(DrillholeColumnWidget, "update_viz_columns", None))
    assert callable(getattr(DrillholeColumnWidget, "set_data_viz_visible", None))


def test_data_columns_x_start_includes_visible_image_column():
    widget = object.__new__(DrillholeColumnWidget)
    widget.show_depth_ruler = True
    widget.ruler_width = 40
    widget.image_mode = "thumbnail"
    widget.thumbnail_width = 160
    widget.color_bar_width = 20

    assert widget._get_data_columns_x_start() == 200


def test_legacy_data_source_label_searches_all_sources():
    widget = make_widget({("fe_pct_best", None): "61.2"})

    value = widget._get_value_for_interval(
        make_interval(),
        "fe_pct_best",
        "Data",
    )

    assert value == 61.2
    assert widget.data_manager.geological_store.calls == [
        ("NB0242", 11.0, "fe_pct_best", None)
    ]


def test_legacy_pct_column_falls_back_to_best_column():
    widget = make_widget({("sio2_pct_best", None): "8.4"})

    value = widget._get_value_for_interval(
        make_interval(),
        "sio2_pct",
        "Data",
    )

    assert value == 8.4
    assert widget.data_manager.geological_store.calls == [
        ("NB0242", 11.0, "sio2_pct", None),
        ("NB0242", 11.0, "sio2_pct_best", None),
    ]


def test_legacy_pct_column_matches_best_csv_key():
    widget = make_widget()

    value = widget._get_value_for_interval(
        make_interval({"al2o3_pct_best": "2.1"}),
        "al2o3_pct",
        "Data",
    )

    assert value == 2.1


class FakeConfigManager:
    def __init__(self, values=None):
        self.values = values or {}

    def get(self, key, default=None):
        return self.values.get(key, default)


class SourceAwareFakeDataManager:
    def __init__(self):
        self.geological_store = FakeStore({})
        self.interval_calls = []
        self.point_calls = []
        self.source_list_calls = 0

    def get_correlation_source_names_for_holes(self, hole_ids):
        self.source_list_calls += 1
        return ["summary_logging_assays", "geophysicsdetails"]

    def get_interval_source_value(self, hole_id, depth_from, depth_to, column, source_name):
        self.interval_calls.append((hole_id, depth_from, depth_to, column, source_name))
        if column == "sio2_pct_best" and source_name == "summary_logging_assays":
            return "8.4"
        return None

    def get_point_values_for_interval(self, hole_id, depth_from, depth_to, column, source_name):
        self.point_calls.append((hole_id, depth_from, depth_to, column, source_name))
        if column == "gamma_cps" and source_name == "geophysicsdetails":
            return [(10.1, 5.0), (10.2, 8.0)]
        return []


def test_generic_data_source_tries_correlation_sources_for_interval_values():
    widget = object.__new__(DrillholeColumnWidget)
    widget.hole_id = "NB0093"
    widget.data_manager = SourceAwareFakeDataManager()

    value = widget._get_value_for_interval(
        make_interval(),
        "sio2_pct",
        "Data",
    )

    assert value == 8.4
    assert (
        "NB0093",
        10.0,
        11.0,
        "sio2_pct_best",
        "summary_logging_assays",
    ) in widget.data_manager.interval_calls


def test_generic_data_source_uses_loaded_interval_row_before_source_lookup():
    widget = object.__new__(DrillholeColumnWidget)
    widget.hole_id = "OK0119"
    widget.data_manager = SourceAwareFakeDataManager()

    value = widget._get_value_for_interval(
        make_interval({"sio2_pct_best": "7.3"}),
        "sio2_pct",
        "Data",
    )

    assert value == 7.3
    assert widget.data_manager.interval_calls == []


def test_generic_data_source_tries_point_sources_for_raw_traces():
    widget = object.__new__(DrillholeColumnWidget)
    widget.hole_id = "NB0093"
    widget.data_manager = SourceAwareFakeDataManager()
    widget.intervals = [make_interval()]

    values = widget._get_point_values_for_viz_range("gamma_cps", "Data")

    assert values == [(10.1, 5.0), (10.2, 8.0)]


def test_viz_source_candidates_are_cached_per_widget():
    widget = object.__new__(DrillholeColumnWidget)
    widget.hole_id = "NB0093"
    widget.data_manager = SourceAwareFakeDataManager()

    assert widget._viz_source_candidates("Data") == [
        "summary_logging_assays",
        "geophysicsdetails",
    ]
    assert widget.data_manager.source_list_calls == 1
    assert widget._viz_source_candidates("Data") == [
        "summary_logging_assays",
        "geophysicsdetails",
    ]
    assert widget.data_manager.source_list_calls == 1


def test_legacy_graph_width_expands_to_content_minimum():
    widget = object.__new__(DrillholeColumnWidget)
    widget.config_manager = FakeConfigManager()
    widget.viz_columns = [{"column": "sio2_pct", "display_name": "SiO2 %"}]
    widget.data_column_min_width = 20
    widget.data_column_max_width = 260

    assert widget._normalise_data_column_width(30) >= widget.MIN_GRAPH_COLUMN_CONTENT_WIDTH


def test_resize_handle_detection_uses_data_column_dividers():
    widget = object.__new__(DrillholeColumnWidget)
    widget.show_data_viz = True
    widget.viz_columns = [{"column": "fe_pct_best"}]
    widget.show_depth_ruler = True
    widget.ruler_width = 40
    widget.image_mode = "thumbnail"
    widget.thumbnail_width = 70
    widget.color_bar_width = 20
    widget.data_column_width = 70
    widget.header_height = 45

    assert widget.is_on_data_column_resize_handle(180, 50)
    assert not widget.is_on_data_column_resize_handle(150, 50)
    assert not widget.is_on_data_column_resize_handle(180, 20)


def test_thumbnail_cover_resize_preserves_source_aspect_ratio():
    new_w, new_h = DrillholeColumnWidget._cover_resize_dimensions(100, 50, 70, 50)

    assert new_w >= 70
    assert new_h >= 50
    assert new_w / new_h == 2.0


def test_visible_thumbnail_indices_include_only_viewport_band_with_margin():
    widget = object.__new__(DrillholeColumnWidget)
    widget.header_height = 45
    widget.cell_height = 10
    widget.thumbnail_lazy_margin_px = 20
    widget.intervals = [
        DrillholeInterval(hole_id="OK0119", depth_from=float(i), depth_to=float(i + 1))
        for i in range(30)
    ]

    assert widget._thumbnail_indices_for_visible_range(95, 135) == list(range(3, 12))
