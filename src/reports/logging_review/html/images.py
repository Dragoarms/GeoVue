"""Image encoding for logging review HTML report."""
import base64
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def _encode_image_base64(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    try:
        ext = Path(path).suffix.lower().lstrip(".")
        if ext == "jpg":
            ext = "jpeg"
        with open(path, "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("ascii")
        return f"data:image/{ext};base64,{encoded}"
    except Exception:
        logger.exception("Failed to encode image: %s", path)
        return None


