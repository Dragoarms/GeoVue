from gui.boundary_manager import BoundaryManager


def test_interpolate_missing_boundary_uses_interior_gap_without_scale():
    manager = BoundaryManager(
        marker_to_compartment={marker_id: marker_id - 3 for marker_id in range(4, 9)},
        expected_compartment_count=5,
        compartment_interval=1,
    )
    manager.add_boundary(4, 0, 10, 20, 60)
    manager.add_boundary(5, 25, 10, 45, 60)
    manager.add_boundary(7, 75, 10, 95, 60)
    manager.add_boundary(8, 100, 10, 120, 60)

    result = manager.interpolate_missing_boundaries(
        vertical_constraints=(10, 60),
        scale_px_per_cm=None,
        config={"compartment_count": 5},
        image_width=200,
    )

    assert result["interpolated_marker_ids"] == [6]
    assert result["interpolated_boundaries"] == [(50, 10, 70, 60)]
