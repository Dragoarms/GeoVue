"""Image encoding for logging review HTML report."""
import base64
import hashlib
import logging
import os
import posixpath
import re
from pathlib import Path
from typing import Dict, Optional

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


ImageAssetCache = Optional[Dict[str, Optional[str]]]
THUMBNAIL_MAX_PIXELS = 900
THUMBNAIL_QUALITY = 72


def _normalize_image_mode(include_images: bool, image_mode: Optional[str]) -> str:
    """Return one of: none, thumbnail, embedded."""
    if not include_images:
        return "none"
    mode = (image_mode or "thumbnail").strip().lower()
    if mode in {"none", "off", "false", "no"}:
        return "none"
    if mode in {"embedded", "embed", "inline", "base64", "original", "full"}:
        return "embedded"
    if mode in {"thumbnail", "thumbnails", "thumb", "external", "small"}:
        return "thumbnail"
    logger.warning("Unknown logging review image mode %r; using thumbnail mode.", image_mode)
    return "thumbnail"


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


def _safe_asset_stem(path: str) -> str:
    stem = Path(path).stem or "interval"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem[:80] or "interval"


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB")


def _write_thumbnail_asset(
    path: str,
    output_dir: Optional[str],
    relative_dir: Optional[str],
    asset_cache: ImageAssetCache = None,
    max_pixels: int = THUMBNAIL_MAX_PIXELS,
    quality: int = THUMBNAIL_QUALITY,
) -> Optional[str]:
    """Write a deduplicated JPEG thumbnail and return its relative report URL."""
    if not path or not os.path.exists(path) or not output_dir or not relative_dir:
        return None

    abs_path = os.path.abspath(path)
    if asset_cache is not None and abs_path in asset_cache:
        return asset_cache[abs_path]

    try:
        os.makedirs(output_dir, exist_ok=True)
        digest = hashlib.sha1(abs_path.encode("utf-8")).hexdigest()[:12]
        file_name = f"{_safe_asset_stem(abs_path)}_{digest}.jpg"
        output_path = os.path.join(output_dir, file_name)

        if not os.path.exists(output_path):
            with Image.open(abs_path) as image:
                thumb = ImageOps.exif_transpose(image)
                thumb.thumbnail((max_pixels, max_pixels), Image.Resampling.LANCZOS)
                thumb = _to_rgb(thumb)
                thumb.save(
                    output_path,
                    format="JPEG",
                    quality=quality,
                    optimize=True,
                    progressive=True,
                )

        relative_url = posixpath.join(relative_dir.replace("\\", "/"), file_name)
        if asset_cache is not None:
            asset_cache[abs_path] = relative_url
        return relative_url
    except Exception:
        logger.exception("Failed to create thumbnail asset: %s", path)
        if asset_cache is not None:
            asset_cache[abs_path] = None
        return None


