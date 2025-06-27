import cv2
import numpy as np
from typing import Dict, Tuple, Any


def generate_test_tray_image(config: Dict[str, Any], width: int = 800, height: int = 400) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """Generate a synthetic chip tray image with ArUco markers for testing.

    This helper creates a blank white image and draws a subset of ArUco markers
    for the corner and compartment IDs defined in ``config``.

    Args:
        config: Configuration dictionary containing ``aruco_dict_type``,
            ``corner_marker_ids`` and ``compartment_marker_ids``.
        width: Width of the generated image in pixels.
        height: Height of the generated image in pixels.

    Returns:
        Tuple of ``(image, markers)`` where ``image`` is the generated BGR image
        and ``markers`` is a dictionary mapping marker IDs to their corner
        coordinates.
    """
    dictionary_name = config.get("aruco_dict_type", "DICT_4X4_1000")
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))

    # Start with a white background
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    markers: Dict[int, np.ndarray] = {}

    marker_size = 50
    margin = 20

    # Place corner markers in each corner
    corner_ids = config.get("corner_marker_ids", [0, 1, 2, 3])
    corner_positions = [
        (margin, margin),
        (width - margin - marker_size, margin),
        (width - margin - marker_size, height - margin - marker_size),
        (margin, height - margin - marker_size),
    ]
    for idx, (x, y) in enumerate(corner_positions):
        marker_id = corner_ids[idx % len(corner_ids)]
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
        image[y : y + marker_size, x : x + marker_size] = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        corners = np.array(
            [[x, y], [x + marker_size, y], [x + marker_size, y + marker_size], [x, y + marker_size]],
            dtype=np.float32,
        )
        markers[marker_id] = corners

    # Place a few compartment markers along the middle
    comp_ids = config.get("compartment_marker_ids", [])
    if comp_ids:
        step = (width - 2 * margin - marker_size) // len(comp_ids[:4])
        y = height // 2 - marker_size // 2
        for i, marker_id in enumerate(comp_ids[:4]):
            x = margin + i * step
            marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
            image[y : y + marker_size, x : x + marker_size] = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
            corners = np.array(
                [[x, y], [x + marker_size, y], [x + marker_size, y + marker_size], [x, y + marker_size]],
                dtype=np.float32,
            )
            markers[marker_id] = corners

    return image, markers
