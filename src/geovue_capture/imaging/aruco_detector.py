"""
ArUco marker detection for chip tray alignment.

Detects ArUco markers in images for:
  - Confirming chip tray is properly positioned
  - Image alignment reference points
"""

import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np
    from PIL import Image
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available - ArUco detection disabled")


class ArucoDetector:
    """Detect and annotate ArUco markers in images."""

    DICT_TYPE = cv2.aruco.DICT_4X4_50 if CV2_AVAILABLE else None

    def __init__(self):
        if not CV2_AVAILABLE:
            return
        self._dict = cv2.aruco.getPredefinedDictionary(self.DICT_TYPE)
        self._params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(self._dict, self._params)

    def detect(self, image) -> Tuple[Optional[list], Optional[list]]:
        """
        Detect ArUco markers in an image.

        Args:
            image: PIL Image or numpy array (BGR).

        Returns:
            (corners, ids) tuple. Both None if no markers found or CV2 unavailable.
        """
        if not CV2_AVAILABLE:
            return None, None

        if isinstance(image, Image.Image):
            img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img_array = image

        corners, ids, _ = self._detector.detectMarkers(img_array)

        if ids is not None:
            return corners, ids.flatten().tolist()
        return None, None

    def detect_and_annotate(self, image) -> Tuple[Optional[object], int]:
        """
        Detect markers and draw them on the image.

        Returns:
            (annotated_pil_image, marker_count)
        """
        if not CV2_AVAILABLE:
            return image, 0

        if isinstance(image, Image.Image):
            img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img_array = image.copy()

        corners, ids, _ = self._detector.detectMarkers(img_array)
        count = len(ids) if ids is not None else 0

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(img_array, corners, ids)

        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb), count
