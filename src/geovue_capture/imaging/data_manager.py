"""
Session and image data management for GeoVue Capture.

Handles:
  - Session state persistence (current hole, depth, etc.)
  - Image file saving with structured naming
  - Soft-delete to recovery folder
  - Daily processing log
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DataManager:
    """Manages capture session state and image files."""

    def __init__(self, storage_dir: Path, images_dir: Path, deleted_dir: Path):
        self._storage_dir = storage_dir
        self._images_dir = images_dir
        self._deleted_dir = deleted_dir
        self._session_file = storage_dir / "session_state.json"

        # Ensure directories exist
        for d in [storage_dir, images_dir, deleted_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Session data
        self.session = self._load_session()

        # Today's processed images
        self.processed_today: list[dict] = []

    def _load_session(self) -> dict:
        """Load persisted session state."""
        if self._session_file.exists():
            try:
                with open(self._session_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load session: {e}")
        return {
            "hole_id": "",
            "depth_from": 0,
            "depth_to": 20,
            "depth_increment": 20,
            "moisture": "Dry",
        }

    def save_session(self):
        """Persist current session state."""
        try:
            with open(self._session_file, "w") as f:
                json.dump(self.session, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save session: {e}")

    def save_image(self, source_path: str, hole_id: str,
                   depth_from: int, depth_to: int, moisture: str) -> Optional[str]:
        """
        Save an image with structured filename and directory.

        Directory structure: images_dir / project_code / hole_id / filename.jpg
        Filename format: {hole_id}_{from}-{to}m_{moisture}_{timestamp}.jpg
        """
        source = Path(source_path)
        if not source.exists():
            logger.error(f"Source image not found: {source_path}")
            return None

        try:
            # Project code = first 2 chars of hole ID
            project_code = hole_id[:2] if len(hole_id) >= 2 else "XX"
            save_dir = self._images_dir / project_code / hole_id
            save_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = source.suffix or ".jpg"
            filename = f"{hole_id}_{depth_from}-{depth_to}m_{moisture}_{timestamp}{ext}"
            dest = save_dir / filename

            shutil.move(str(source), str(dest))

            entry = {
                "hole_id": hole_id,
                "depth": f"{depth_from}-{depth_to}m",
                "moisture": moisture,
                "time": datetime.now().strftime("%H:%M:%S"),
                "path": str(dest),
            }
            self.processed_today.append(entry)

            logger.info(f"Saved image: {dest}")
            return str(dest)

        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            return None

    def delete_image(self, image_path: str) -> bool:
        """Soft-delete: move image to deleted folder (organized by date)."""
        try:
            today_dir = self._deleted_dir / datetime.now().strftime("%Y%m%d")
            today_dir.mkdir(parents=True, exist_ok=True)

            dest = today_dir / Path(image_path).name
            shutil.move(image_path, str(dest))

            self.processed_today = [
                p for p in self.processed_today if p.get("path") != image_path
            ]

            logger.info(f"Moved to deleted: {dest}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete image: {e}")
            return False
