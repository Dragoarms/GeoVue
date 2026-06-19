from utils.reprocess_depths import (
    infer_register_compartment_interval,
    resolve_register_compartment_depths,
)


def test_register_reextract_depths_use_compartment_number_with_tray_range():
    original = {"Depth_From": 100, "Depth_To": 120}
    corners = [
        {"Compartment_Number": i, "Depth_From": 100, "Depth_To": 120}
        for i in range(1, 21)
    ]

    assert infer_register_compartment_interval(original, corners) == 1
    assert resolve_register_compartment_depths(original, corners[0], corners) == (
        100,
        101,
    )
    assert resolve_register_compartment_depths(original, corners[19], corners) == (
        119,
        120,
    )


def test_register_reextract_depths_respect_explicit_interval():
    original = {"Depth_From": 200, "Depth_To": 240, "Compartment_Interval": 2}
    corners = [{"Compartment_Number": 3, "Depth_From": 200, "Depth_To": 240}]

    assert infer_register_compartment_interval(original, corners) == 2
    assert resolve_register_compartment_depths(original, corners[0], corners) == (
        204,
        206,
    )
