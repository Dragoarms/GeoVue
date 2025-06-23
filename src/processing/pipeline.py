"""processing/pipeline.py

Utility functions for image loading and transformations used across GeoVue.
"""

import logging
from typing import Tuple
import numpy as np
import cv2

try:
    from PIL import Image as PILImage
except Exception:  # pragma: no cover - PIL optional
    PILImage = None

logger = logging.getLogger(__name__)


def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk using PIL with OpenCV fallback.

    Args:
        image_path: Path to the image file.

    Returns:
        The loaded image as a ``numpy`` array in BGR colour order.
    """
    if PILImage:
        try:
            pil_img = PILImage.open(image_path)
            img = np.array(pil_img)
            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
        except Exception as exc:  # pragma: no cover - PIL may fail
            logger.debug("PIL failed to load %s: %s", image_path, exc)
    img = cv2.imread(image_path)
    if img is None:
        raise IOError(f"Unable to load image: {image_path}")
    return img


def resize_for_processing(image: np.ndarray, target_pixels: int = 2_000_000) -> np.ndarray:
    """Return a down-sampled copy for analysis.

    The image is scaled so that the total pixel count is roughly ``target_pixels``.

    Args:
        image: Source image array.
        target_pixels: Desired pixel count for the result.

    Returns:
        Resized image array suitable for processing.
    """
    h, w = image.shape[:2]
    original_pixels = h * w
    if original_pixels <= target_pixels:
        return image.copy()
    scale = (target_pixels / original_pixels) ** 0.5
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def apply_transform(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply an affine transform to ``image``.

    Args:
        image: Input image array.
        matrix: 2x3 affine transformation matrix.

    Returns:
        The transformed image array.
    """
    h, w = image.shape[:2]
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
