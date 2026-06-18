from gui.widgets.minimap_canvas import MinimapCanvas


def make_canvas():
    canvas = object.__new__(MinimapCanvas)
    canvas.collar_color = "#3498DB"
    canvas.assay_collar_color = "#2ECC71"
    canvas.no_assay_collar_color = "#F1C40F"
    canvas.selected_collar_color = "#E74C3C"
    return canvas


def test_minimap_uses_assay_status_colors_for_unselected_collars():
    canvas = make_canvas()

    assert canvas._get_collar_fill_color(False, True) == "#2ECC71"
    assert canvas._get_collar_fill_color(False, False) == "#F1C40F"
    assert canvas._get_collar_fill_color(False, None) == "#3498DB"


def test_minimap_selection_color_overrides_assay_status():
    canvas = make_canvas()

    assert canvas._get_collar_fill_color(True, True) == "#E74C3C"
    assert canvas._get_collar_fill_color(True, False) == "#E74C3C"


def test_minimap_coerces_dataframe_bool_values():
    assert MinimapCanvas._coerce_optional_bool("true") is True
    assert MinimapCanvas._coerce_optional_bool("0") is False
    assert MinimapCanvas._coerce_optional_bool("") is False
    assert MinimapCanvas._coerce_optional_bool(None) is None
