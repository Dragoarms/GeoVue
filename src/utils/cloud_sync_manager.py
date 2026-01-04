"""
Cloud Sync Manager for moving local files to cloud storage.

This module handles synchronization of local files to cloud storage,
particularly for compartment images to free up local disk space.
"""

import os
import shutil
import logging
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import traceback


class CloudSyncManager:
    """Manages synchronization of local files to cloud storage."""

    def __init__(self, file_manager, logger=None):
        """Initialize the cloud sync manager."""
        self.file_manager = file_manager
        self.logger = logger or logging.getLogger(__name__)
        self.stats = {}

    def sync_compartments_to_cloud(self, progress_callback=None) -> Dict[str, Any]:
        """
        Sync local compartment images to cloud storage.
        Focus on temp/review compartments to free local space.

        Returns:
            Dictionary with sync statistics
        """
        self.stats = {
            "temp_moved": 0,
            "temp_failed": 0,
            "approved_moved": 0,
            "approved_failed": 0,
            "already_in_cloud": 0,
            "bytes_freed": 0,
            "errors": [],
        }

        try:
            # Check if cloud storage is configured
            if not self.file_manager.shared_paths:
                return {"success": False, "error": "No cloud storage configured"}

            # Sync temp/review compartments first (highest priority)
            if progress_callback:
                progress_callback("Syncing review compartments...", 10)
            self._sync_review_compartments()

            # Sync approved compartments that are marked as uploaded
            if progress_callback:
                progress_callback("Syncing approved compartments...", 50)
            self._sync_approved_compartments()

            # Calculate space freed
            self.stats["mb_freed"] = round(self.stats["bytes_freed"] / (1024 * 1024), 2)

            if progress_callback:
                progress_callback("Sync complete", 100)

            return {"success": True, "stats": self.stats}

        except Exception as e:
            self.logger.error(f"Error in cloud sync: {str(e)}")
            self.stats["errors"].append(str(e))
            return {"success": False, "error": str(e), "stats": self.stats}

    def _sync_review_compartments(self):
        """Sync temp/review compartments to cloud."""
        local_review = self.file_manager.dir_structure.get("temp_review")
        cloud_review = self.file_manager.get_shared_path("review_compartments")

        if not local_review or not cloud_review:
            return

        self._sync_folder(
            local_review,
            cloud_review,
            "temp",
            delete_after_sync=True,  # Remove local files after successful sync
        )

    def _sync_approved_compartments(self):
        """Sync approved compartments that are already uploaded."""
        local_approved = self.file_manager.dir_structure.get("approved_compartments")
        cloud_approved = self.file_manager.get_shared_path("approved_compartments")

        if not local_approved or not cloud_approved:
            return

        # Only sync files marked as UPLOADED
        self._sync_folder(
            local_approved,
            cloud_approved,
            "approved",
            file_pattern="*_UPLOADED.*",
            delete_after_sync=True,
        )

    def _sync_folder(
        self,
        local_base: Path,
        cloud_base: Path,
        category: str,
        file_pattern: str = "*",
        delete_after_sync: bool = False,
    ):
        """Sync files from local folder to cloud."""
        if not os.path.exists(local_base):
            return

        # Walk through project/hole structure
        for project_dir in os.listdir(local_base):
            project_path = Path(local_base) / project_dir
            if not project_path.is_dir():
                continue

            for hole_dir in os.listdir(project_path):
                hole_path = project_path / hole_dir
                if not hole_path.is_dir():
                    continue

                # Create cloud directory
                cloud_hole_path = cloud_base / project_dir / hole_dir
                cloud_hole_path.mkdir(parents=True, exist_ok=True)

                # Find files to sync
                pattern_path = str(hole_path / file_pattern)
                for file_path in glob.glob(pattern_path):
                    self._sync_file(
                        file_path, cloud_hole_path, category, delete_after_sync
                    )

    def _sync_file(
        self, local_file: str, cloud_dir: Path, category: str, delete_after_sync: bool
    ):
        """Sync a single file to cloud."""
        try:
            filename = os.path.basename(local_file)
            cloud_file = cloud_dir / filename

            # Check if already in cloud
            if cloud_file.exists():
                # Skip files that already exist in cloud - leave for manual review
                self.stats["already_in_cloud"] += 1
                self.logger.info(
                    f"Skipped {filename} - already exists in cloud (kept locally for manual review)"
                )
                return

            # Copy to cloud
            shutil.copy2(local_file, str(cloud_file))

            # Verify copy
            if cloud_file.exists() and cloud_file.stat().st_size == os.path.getsize(
                local_file
            ):
                self.stats[f"{category}_moved"] += 1
                file_size = os.path.getsize(local_file)

                if delete_after_sync:
                    os.remove(local_file)
                    self.stats["bytes_freed"] += file_size
                    self.logger.info(f"Moved to cloud and deleted: {filename}")
                else:
                    self.logger.info(f"Copied to cloud: {filename}")
            else:
                self.stats[f"{category}_failed"] += 1
                self.logger.error(f"Failed to verify cloud copy: {filename}")

        except Exception as e:
            self.stats[f"{category}_failed"] += 1
            self.logger.error(f"Error syncing file {local_file}: {str(e)}")
            self.stats["errors"].append(f"{os.path.basename(local_file)}: {str(e)}")

    def get_sync_summary(self) -> Dict[str, Any]:
        """Get summary of what would be synced without actually syncing."""
        summary = {
            "temp_review_files": 0,
            "temp_review_size_mb": 0,
            "temp_review_would_skip": 0,
            "approved_uploaded_files": 0,
            "approved_uploaded_size_mb": 0,
            "already_synced_files": 0,
        }

        # Count temp/review files
        local_review = self.file_manager.dir_structure.get("temp_review")
        cloud_review = self.file_manager.get_shared_path("review_compartments")

        if local_review and os.path.exists(local_review):
            total_count, total_size = self._count_files_in_folder(local_review)
            skip_count = 0

            # Check how many would be skipped
            if cloud_review:
                skip_count = self._count_existing_in_cloud(local_review, cloud_review)

            summary["temp_review_files"] = total_count
            summary["temp_review_would_skip"] = skip_count
            summary["temp_review_size_mb"] = round(total_size / (1024 * 1024), 2)

        # Count approved uploaded files
        local_approved = self.file_manager.dir_structure.get("approved_compartments")
        if local_approved and os.path.exists(local_approved):
            count, size = self._count_files_in_folder(local_approved, "*_UPLOADED.*")
            summary["approved_uploaded_files"] = count
            summary["approved_uploaded_size_mb"] = round(size / (1024 * 1024), 2)

        return summary

    def _count_existing_in_cloud(self, local_base: Path, cloud_base: Path) -> int:
        """Count how many files already exist in cloud."""
        skip_count = 0

        if not cloud_base.exists():
            return 0

        for project_dir in os.listdir(local_base):
            project_path = Path(local_base) / project_dir
            if not project_path.is_dir():
                continue

            for hole_dir in os.listdir(project_path):
                hole_path = project_path / hole_dir
                if not hole_path.is_dir():
                    continue

                cloud_hole_path = cloud_base / project_dir / hole_dir
                if not cloud_hole_path.exists():
                    continue

                for file in os.listdir(hole_path):
                    if (cloud_hole_path / file).exists():
                        skip_count += 1

        return skip_count

    def _count_files_in_folder(
        self, folder: Path, pattern: str = "*"
    ) -> Tuple[int, int]:
        """Count files and total size in folder."""
        count = 0
        total_size = 0

        import glob

        for root, dirs, files in os.walk(folder):
            for file in files:
                if pattern == "*" or glob.fnmatch.fnmatch(file, pattern):
                    count += 1
                    total_size += os.path.getsize(os.path.join(root, file))

        return count, total_size
