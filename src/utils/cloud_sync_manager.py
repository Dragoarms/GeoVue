"""
Cloud Sync Manager for moving local files to cloud storage.

This module handles synchronization of local files to cloud storage,
particularly for compartment images to free up local disk space.

Key features:
- Non-blocking operations with progress callbacks
- Transparent status updates
- Confirmation before any destructive operations
- Leverages cached data where possible
"""

import os
import shutil
import logging
import glob
import threading
import queue
import signal
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
import traceback


class CloudSyncManager:
    """Manages synchronization of local files to cloud storage."""

    def __init__(self, file_manager, logger=None):
        """Initialize the cloud sync manager."""
        self.file_manager = file_manager
        self.logger = logger or logging.getLogger(__name__)
        self.stats = {}
        self._cancel_requested = False

    def request_cancel(self):
        """Request cancellation of current operation."""
        self._cancel_requested = True
        self.logger.info("Cancellation requested")

    def _check_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._cancel_requested

    def _safe_listdir(self, path: Path) -> List[str]:
        """Safely list directory contents, returning empty list on error."""
        try:
            if path and path.exists() and path.is_dir():
                return os.listdir(str(path))
        except (OSError, PermissionError) as e:
            self.logger.warning(f"Could not list directory {path}: {e}")
        except Exception as e:
            self.logger.debug(f"Error listing directory {path}: {e}")
        return []

    def _cleanup_empty_directories(self, base_path: Path) -> int:
        """
        Recursively clean up empty directories under base_path.
        
        Returns:
            Number of directories removed
        """
        if not base_path or not base_path.exists():
            return 0
        
        removed_count = 0
        
        try:
            # Walk bottom-up to clean nested empty directories
            for project_dir in self._safe_listdir(base_path):
                project_path = base_path / project_dir
                if not project_path.is_dir():
                    continue
                
                for hole_dir in self._safe_listdir(project_path):
                    hole_path = project_path / hole_dir
                    if not hole_path.is_dir():
                        continue
                    
                    # Remove empty hole directories
                    try:
                        if hole_path.exists() and not any(hole_path.iterdir()):
                            hole_path.rmdir()
                            removed_count += 1
                            self.logger.debug(f"Removed empty hole directory: {hole_path}")
                    except (OSError, PermissionError) as e:
                        self.logger.debug(f"Could not remove {hole_path}: {e}")
                
                # Remove empty project directories
                try:
                    if project_path.exists() and not any(project_path.iterdir()):
                        project_path.rmdir()
                        removed_count += 1
                        self.logger.debug(f"Removed empty project directory: {project_path}")
                except (OSError, PermissionError) as e:
                    self.logger.debug(f"Could not remove {project_path}: {e}")
        
        except Exception as e:
            self.logger.warning(f"Error during empty directory cleanup: {e}")
        
        return removed_count

    def get_sync_summary(
        self, 
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Get summary of what would be synced without actually syncing.
        
        This method provides progress updates and can be cancelled.
        
        Args:
            progress_callback: Optional callback(message, percentage) for progress updates
            
        Returns:
            Dictionary with sync summary statistics
        """
        self._cancel_requested = False
        
        summary = {
            "temp_review_files": 0,
            "temp_review_size_mb": 0,
            "temp_review_would_skip": 0,
            "approved_uploaded_files": 0,
            "approved_uploaded_size_mb": 0,
            "approved_originals_uploaded_files": 0,
            "approved_originals_uploaded_size_mb": 0,
            "rejected_originals_uploaded_files": 0,
            "rejected_originals_uploaded_size_mb": 0,
            "already_synced_files": 0,
            "cloud_configured": bool(self.file_manager.shared_paths),
            "scan_completed": False,
        }
        
        def safe_progress(msg: str, pct: int):
            if progress_callback:
                try:
                    progress_callback(msg, pct)
                except Exception as e:
                    self.logger.warning(f"Progress callback error: {e}")

        try:
            # Check if cloud storage is configured
            if not self.file_manager.shared_paths:
                safe_progress("No cloud storage configured", 100)
                return summary

            safe_progress("Checking review compartments...", 10)
            
            # Count temp/review files
            local_review = self.file_manager.dir_structure.get("temp_review")
            cloud_review = self.file_manager.get_shared_path("review_compartments")

            if local_review and os.path.exists(local_review):
                if self._check_cancelled():
                    return summary
                    
                total_count, total_size = self._count_files_in_folder(
                    local_review, 
                    progress_callback=lambda msg: safe_progress(f"Review: {msg}", 15)
                )
                skip_count = 0

                # Check how many would be skipped
                if cloud_review and Path(cloud_review).exists():
                    safe_progress("Checking existing cloud files...", 20)
                    skip_count = self._count_existing_in_cloud(local_review, cloud_review)

                summary["temp_review_files"] = total_count
                summary["temp_review_would_skip"] = skip_count
                summary["temp_review_size_mb"] = round(total_size / (1024 * 1024), 2)

            if self._check_cancelled():
                return summary

            safe_progress("Checking approved compartments...", 40)
            
            # Count approved compartment uploaded files
            local_approved = self.file_manager.dir_structure.get("approved_compartments")
            if local_approved and os.path.exists(local_approved):
                count, size = self._count_files_in_folder(
                    local_approved, 
                    "*_UPLOADED.*",
                    progress_callback=lambda msg: safe_progress(f"Approved: {msg}", 50)
                )
                summary["approved_uploaded_files"] = count
                summary["approved_uploaded_size_mb"] = round(size / (1024 * 1024), 2)

            if self._check_cancelled():
                return summary

            safe_progress("Checking approved originals...", 60)
            
            # Count approved originals uploaded files
            local_approved_originals = self.file_manager.dir_structure.get(
                "approved_originals"
            )
            if local_approved_originals and os.path.exists(local_approved_originals):
                count, size = self._count_files_in_folder(
                    local_approved_originals, 
                    "*_UPLOADED.*",
                    progress_callback=lambda msg: safe_progress(f"Approved originals: {msg}", 70)
                )
                summary["approved_originals_uploaded_files"] = count
                summary["approved_originals_uploaded_size_mb"] = round(
                    size / (1024 * 1024), 2
                )

            if self._check_cancelled():
                return summary

            safe_progress("Checking rejected originals...", 80)
            
            # Count rejected originals uploaded files
            local_rejected_originals = self.file_manager.dir_structure.get(
                "rejected_originals"
            )
            if local_rejected_originals and os.path.exists(local_rejected_originals):
                count, size = self._count_files_in_folder(
                    local_rejected_originals, 
                    "*_UPLOADED.*",
                    progress_callback=lambda msg: safe_progress(f"Rejected originals: {msg}", 90)
                )
                summary["rejected_originals_uploaded_files"] = count
                summary["rejected_originals_uploaded_size_mb"] = round(
                    size / (1024 * 1024), 2
                )

            summary["scan_completed"] = True
            safe_progress("Scan complete", 100)

        except Exception as e:
            self.logger.error(f"Error getting sync summary: {e}")
            summary["error"] = str(e)

        return summary

    def _count_files_in_folder(
        self, 
        folder: Path, 
        pattern: str = "*",
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Tuple[int, int]:
        """
        Count files and total size in folder.
        
        Args:
            folder: Path to scan
            pattern: File pattern to match
            progress_callback: Optional callback for status updates
            
        Returns:
            Tuple of (count, total_size_bytes)
        """
        count = 0
        total_size = 0

        import fnmatch

        self.logger.debug(f"Counting files in {folder} with pattern {pattern}")
        
        try:
            folder_path = Path(folder)
            
            # Quick check - if folder doesn't exist or is empty, return early
            if not folder_path.exists():
                return 0, 0
            
            # Use os.scandir for better performance than os.walk on network drives
            for root, dirs, files in os.walk(folder):
                if self._check_cancelled():
                    break
                    
                # Report which folder we're scanning
                if progress_callback:
                    rel_path = Path(root).relative_to(folder_path) if folder_path in Path(root).parents or folder_path == Path(root) else Path(root).name
                    progress_callback(f"Scanning {rel_path}...")
                
                for file in files:
                    if pattern == "*" or fnmatch.fnmatch(file, pattern):
                        file_path = os.path.join(root, file)
                        try:
                            count += 1
                            total_size += os.path.getsize(file_path)
                        except OSError:
                            pass  # Skip files we can't access
                        
                        if "_UPLOADED" in file:
                            self.logger.debug(f"Found uploaded file: {file_path}")

        except Exception as e:
            self.logger.warning(f"Error counting files in {folder}: {e}")

        self.logger.debug(
            f"Found {count} files matching {pattern} in {folder}, total size: {total_size / (1024*1024):.2f} MB"
        )
        return count, total_size

    def _count_existing_in_cloud(self, local_base: Path, cloud_base: Path) -> int:
        """Count how many files already exist in cloud."""
        skip_count = 0

        cloud_path = Path(cloud_base)
        if not cloud_path.exists():
            return 0

        try:
            for project_dir in self._safe_listdir(Path(local_base)):
                if self._check_cancelled():
                    break
                    
                project_path = Path(local_base) / project_dir
                if not project_path.is_dir():
                    continue

                for hole_dir in self._safe_listdir(project_path):
                    if self._check_cancelled():
                        break
                        
                    hole_path = project_path / hole_dir
                    if not hole_path.is_dir():
                        continue

                    cloud_hole_path = cloud_path / project_dir / hole_dir
                    if not cloud_hole_path.exists():
                        continue

                    for file in self._safe_listdir(hole_path):
                        if (cloud_hole_path / file).exists():
                            skip_count += 1
        except Exception as e:
            self.logger.warning(f"Error counting existing cloud files: {e}")

        return skip_count

    def sync_compartments_to_cloud(
        self, 
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Sync local compartment images to cloud storage.
        Focus on temp/review compartments to free local space.

        Args:
            progress_callback: Optional callback(message, percentage) for progress updates

        Returns:
            Dictionary with sync statistics
        """
        self._cancel_requested = False
        
        self.stats = {
            "temp_moved": 0,
            "temp_failed": 0,
            "approved_moved": 0,
            "approved_failed": 0,
            "approved_cleaned": 0,
            "approved_originals_moved": 0,
            "approved_originals_failed": 0,
            "approved_originals_cleaned": 0,
            "rejected_originals_moved": 0,
            "rejected_originals_failed": 0,
            "rejected_originals_cleaned": 0,
            "already_in_cloud": 0,
            "missing_cloud_files": 0,
            "bytes_freed": 0,
            "errors": [],
        }

        try:
            # Check if cloud storage is configured
            if not self.file_manager.shared_paths:
                return {"success": False, "error": "No cloud storage configured"}

            # Thread-safe progress callback wrapper
            def safe_progress(msg, pct):
                if progress_callback:
                    try:
                        progress_callback(msg, pct)
                    except Exception as e:
                        self.logger.warning(f"Progress callback error: {e}")

            if self._check_cancelled():
                return {"success": False, "error": "Cancelled", "stats": self.stats}

            # Sync temp/review compartments first (highest priority)
            safe_progress("Syncing review compartments...", 10)
            self._sync_review_compartments()

            if self._check_cancelled():
                return {"success": False, "error": "Cancelled", "stats": self.stats}

            # Sync approved compartments that are marked as uploaded
            safe_progress("Syncing approved compartments...", 40)
            self._sync_approved_compartments()

            if self._check_cancelled():
                return {"success": False, "error": "Cancelled", "stats": self.stats}

            # Sync processed original images
            safe_progress("Syncing processed original images...", 60)
            self._sync_processed_originals()

            if self._check_cancelled():
                return {"success": False, "error": "Cancelled", "stats": self.stats}

            # Final cleanup pass - remove any empty directories
            safe_progress("Cleaning up empty directories...", 85)
            dirs_removed = self._final_directory_cleanup()
            self.stats["directories_cleaned"] = dirs_removed

            # Calculate space freed
            self.stats["mb_freed"] = round(self.stats["bytes_freed"] / (1024 * 1024), 2)

            # Log summary
            total_moved = (
                self.stats.get("temp_moved", 0) + 
                self.stats.get("approved_moved", 0) +
                self.stats.get("approved_originals_moved", 0) +
                self.stats.get("rejected_originals_moved", 0)
            )
            total_cleaned = (
                self.stats.get("approved_cleaned", 0) +
                self.stats.get("approved_originals_cleaned", 0) +
                self.stats.get("rejected_originals_cleaned", 0)
            )
            
            if total_moved == 0 and total_cleaned == 0 and self.stats.get("already_in_cloud", 0) == 0:
                self.logger.info("Sync complete - nothing to sync")
            else:
                self.logger.info(f"Sync complete - Stats: {self.stats}")

            safe_progress("Sync complete", 100)

            return {"success": True, "stats": self.stats}

        except Exception as e:
            self.logger.error(f"Error in cloud sync: {str(e)}\n{traceback.format_exc()}")
            if not hasattr(self, "stats") or self.stats is None:
                self.stats = {"errors": []}
            self.stats["errors"].append(str(e))
            return {"success": False, "error": str(e), "stats": self.stats}

    def _sync_review_compartments(self):
        """Sync temp/review compartments to cloud."""
        local_review = self.file_manager.dir_structure.get("temp_review")
        cloud_review = self.file_manager.get_shared_path("review_compartments")

        if not local_review or not cloud_review:
            self.logger.debug(
                f"Skipping review sync - local: {local_review}, cloud: {cloud_review}"
            )
            return

        self.logger.info(
            f"Syncing review compartments from {local_review} to {cloud_review}"
        )
        self._sync_folder(
            local_review,
            cloud_review,
            "temp",
            delete_after_sync=True,  # Remove local files after successful sync
        )

    def _sync_approved_compartments(self):
        """Sync approved compartments that failed upload or haven't been uploaded yet."""
        local_approved = self.file_manager.dir_structure.get("approved_compartments")
        cloud_approved = self.file_manager.get_shared_path("approved_compartments")

        if not local_approved or not cloud_approved:
            self.logger.debug(
                f"Skipping approved sync - local: {local_approved}, cloud: {cloud_approved}"
            )
            return

        self.logger.info(
            f"Syncing approved compartments from {local_approved} to {cloud_approved}"
        )

        # Process files with UPLOAD_FAILED status
        self._sync_failed_uploads(local_approved, cloud_approved, "approved")

        # Process files without upload status (must have moisture suffix)
        self._sync_pending_uploads(local_approved, cloud_approved, "approved")

        # Clean up successfully uploaded files
        self._cleanup_uploaded_files(local_approved, cloud_approved, "approved")

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
            self.logger.debug(f"Local base does not exist: {local_base}")
            return

        self.logger.info(
            f"Syncing folder: {local_base} -> {cloud_base} (pattern: {file_pattern}, delete: {delete_after_sync})"
        )

        # Walk through project/hole structure
        for project_dir in self._safe_listdir(Path(local_base)):
            if self._check_cancelled():
                return
                
            project_path = Path(local_base) / project_dir
            if not project_path.is_dir():
                continue

            for hole_dir in self._safe_listdir(project_path):
                if self._check_cancelled():
                    return
                    
                hole_path = project_path / hole_dir
                if not hole_path.is_dir():
                    continue

                # Create cloud directory
                cloud_hole_path = cloud_base / project_dir / hole_dir
                cloud_hole_path.mkdir(parents=True, exist_ok=True)

                # Find files to sync
                pattern_path = str(hole_path / file_pattern)
                files = glob.glob(pattern_path)
                if files:
                    self.logger.debug(
                        f"Found {len(files)} files in {hole_path} matching {file_pattern}"
                    )
                for file_path in files:
                    if self._check_cancelled():
                        return
                    self._sync_file(
                        file_path, cloud_hole_path, category, delete_after_sync
                    )

                # Clean up empty hole directory after sync with better error handling
                if delete_after_sync and hole_path.exists():
                    try:
                        if not any(hole_path.iterdir()):
                            hole_path.rmdir()
                            self.logger.debug(
                                f"Removed empty hole directory: {hole_path}"
                            )
                    except OSError as e:
                        if e.errno == 5:  # Access denied
                            self.logger.warning(
                                f"Permission denied removing directory {hole_path} - directory may contain hidden files"
                            )
                        else:
                            self.logger.warning(
                                f"Could not remove hole directory {hole_path}: {e}"
                            )
                    except Exception as e:
                        self.logger.debug(
                            f"Could not remove hole directory {hole_path}: {e}"
                        )

            # Clean up empty project directory after sync with better error handling
            if delete_after_sync and project_path.exists():
                try:
                    if not any(project_path.iterdir()):
                        project_path.rmdir()
                        self.logger.debug(
                            f"Removed empty project directory: {project_path}"
                        )
                except OSError as e:
                    if e.errno == 5:  # Access denied
                        self.logger.warning(
                            f"Permission denied removing directory {project_path} - directory may contain hidden files"
                        )
                    else:
                        self.logger.warning(
                            f"Could not remove project directory {project_path}: {e}"
                        )
                except Exception as e:
                    self.logger.debug(
                        f"Could not remove project directory {project_path}: {e}"
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
                # Verify sizes match before deciding cleanup action
                local_size = os.path.getsize(local_file)
                cloud_size = cloud_file.stat().st_size

                if local_size == cloud_size:
                    # Also verify UID consistency for image files
                    uid_match = self._verify_uid_consistency(
                        local_file, str(cloud_file)
                    )

                    if uid_match:
                        self.stats["already_in_cloud"] += 1
                        # If delete_after_sync is True and files match exactly, clean up local duplicate
                        if delete_after_sync:
                            os.remove(local_file)
                            self.stats["bytes_freed"] += local_size
                            self.logger.info(
                                f"Removed local duplicate (UID verified): {filename}"
                            )
                        else:
                            self.logger.info(
                                f"Skipped {filename} - already exists in cloud with matching UID"
                            )
                    else:
                        # UID mismatch - this is serious, don't delete local copy
                        self.logger.error(
                            f"UID mismatch between local and cloud file: {filename}"
                        )
                        self.stats["errors"].append(f"UID mismatch: {filename}")
                        self.stats[f"{category}_failed"] += 1
                else:
                    # Size mismatch - keep both and log warning
                    self.logger.warning(
                        f"Size mismatch for {filename}: local={local_size}, cloud={cloud_size} - keeping both copies"
                    )
                    self.stats[f"{category}_failed"] += 1
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

    def _sync_failed_uploads(self, local_base: Path, cloud_base: Path, category: str):
        """Retry upload for files marked as UPLOAD_FAILED."""
        if not os.path.exists(local_base):
            return

        for project_dir in self._safe_listdir(Path(local_base)):
            if self._check_cancelled():
                return
                
            project_path = Path(local_base) / project_dir
            if not project_path.is_dir():
                continue

            for hole_dir in self._safe_listdir(project_path):
                if self._check_cancelled():
                    return
                    
                hole_path = project_path / hole_dir
                if not hole_path.is_dir():
                    continue

                # Find UPLOAD_FAILED files
                for file_path in hole_path.glob("*_UPLOAD_FAILED.*"):
                    if self._check_cancelled():
                        return
                    self._retry_failed_upload(
                        file_path, cloud_base / project_dir / hole_dir, category
                    )

    def _sync_pending_uploads(self, local_base: Path, cloud_base: Path, category: str):
        """Upload approved files without upload status (must have moisture suffix)."""
        if not os.path.exists(local_base):
            return

        self.logger.debug(f"Checking for pending uploads in {local_base}")
        pending_count = 0

        for project_dir in self._safe_listdir(Path(local_base)):
            if self._check_cancelled():
                return
                
            project_path = Path(local_base) / project_dir
            if not project_path.is_dir():
                continue

            for hole_dir in self._safe_listdir(project_path):
                if self._check_cancelled():
                    return
                    
                hole_path = project_path / hole_dir
                if not hole_path.is_dir():
                    continue

                # Find files without upload status
                for file_path in hole_path.iterdir():
                    if self._check_cancelled():
                        return
                        
                    if file_path.is_file():
                        filename = file_path.name
                        # Check if it's a valid approved compartment (has moisture suffix but no upload status)
                        if (
                            ("_Dry." in filename or "_Wet." in filename)
                            and "_UPLOADED." not in filename
                            and "_UPLOAD_FAILED." not in filename
                        ):
                            self.logger.debug(f"Found pending upload: {filename}")
                            pending_count += 1
                            self._upload_and_mark(
                                file_path, cloud_base / project_dir / hole_dir, category
                            )

        if pending_count > 0:
            self.logger.info(f"Processed {pending_count} pending uploads")

    def _retry_failed_upload(self, local_file: Path, cloud_dir: Path, category: str):
        """Retry a failed upload by removing suffix, uploading, and marking as uploaded."""
        try:
            cloud_dir.mkdir(parents=True, exist_ok=True)

            # Remove _UPLOAD_FAILED suffix to get original name
            filename = local_file.name
            clean_name = filename.replace("_UPLOAD_FAILED", "")
            cloud_file = cloud_dir / clean_name

            # Check if already exists in cloud - skip for manual review
            if cloud_file.exists():
                self.stats["already_in_cloud"] += 1
                self.logger.info(
                    f"Skipped {filename} - {clean_name} already exists in cloud (kept locally for manual review)"
                )
                return

            # Copy to cloud
            shutil.copy2(str(local_file), str(cloud_file))

            # Verify copy
            if (
                cloud_file.exists()
                and cloud_file.stat().st_size == local_file.stat().st_size
            ):
                # Rename local file to mark as uploaded
                new_name = filename.replace("_UPLOAD_FAILED", "_UPLOADED")
                new_path = local_file.parent / new_name
                local_file.rename(new_path)

                self.stats[f"{category}_moved"] += 1
                self.logger.info(f"Successfully retried upload: {clean_name}")
            else:
                self.stats[f"{category}_failed"] += 1
                self.logger.error(f"Failed to verify cloud copy on retry: {filename}")

        except Exception as e:
            self.stats[f"{category}_failed"] += 1
            self.logger.error(f"Error retrying upload {local_file}: {str(e)}")
            self.stats["errors"].append(f"{local_file.name}: {str(e)}")

    def _upload_and_mark(self, local_file: Path, cloud_dir: Path, category: str):
        """Upload a file and mark it as uploaded if successful."""
        try:
            cloud_dir.mkdir(parents=True, exist_ok=True)

            filename = local_file.name

            # Check if already marked as uploaded (prevent double-marking)
            if "_UPLOADED." in filename:
                self.logger.debug(
                    f"File already marked as uploaded, skipping: {filename}"
                )
                return

            cloud_file = cloud_dir / filename

            # Build the _UPLOADED filename for later use
            name_parts = filename.rsplit(".", 1)
            if len(name_parts) == 2:
                uploaded_name = f"{name_parts[0]}_UPLOADED.{name_parts[1]}"
            else:
                uploaded_name = f"{filename}_UPLOADED"
            uploaded_path = local_file.parent / uploaded_name

            # Check if already in cloud
            if cloud_file.exists():
                local_size = local_file.stat().st_size
                cloud_size = cloud_file.stat().st_size
                
                if local_size == cloud_size:
                    # File already in cloud with matching size - clean up local
                    self.stats["already_in_cloud"] += 1
                    
                    # Check if _UPLOADED version also exists (duplicate situation)
                    if uploaded_path.exists():
                        # Both file.png and file_UPLOADED.png exist locally
                        # Cloud has the file, so delete both local copies
                        self.logger.info(
                            f"Cleaning up duplicate: {filename} (already in cloud, "
                            f"_UPLOADED version also exists locally)"
                        )
                        try:
                            uploaded_size = uploaded_path.stat().st_size
                            local_file.unlink()
                            uploaded_path.unlink()
                            self.stats["bytes_freed"] += local_size + uploaded_size
                            self.stats[f"{category}_moved"] += 1
                        except Exception as e:
                            self.logger.warning(f"Could not clean up duplicates: {e}")
                    else:
                        # Just the regular file exists, delete it since it's in cloud
                        self.logger.debug(f"Deleting local copy (already in cloud): {filename}")
                        try:
                            local_file.unlink()
                            self.stats["bytes_freed"] += local_size
                        except Exception as e:
                            self.logger.warning(f"Could not delete local file: {e}")
                    return
                else:
                    # Size mismatch - log warning and continue to re-upload
                    self.logger.warning(
                        f"Size mismatch for {filename}: local={local_size}, cloud={cloud_size}"
                    )

            # Copy to cloud
            shutil.copy2(str(local_file), str(cloud_file))

            # Verify copy
            if (
                cloud_file.exists()
                and cloud_file.stat().st_size == local_file.stat().st_size
            ):
                # Mark local file as uploaded
                # Note: uploaded_name and uploaded_path were defined earlier
                
                # Handle case where _UPLOADED version already exists (duplicate)
                if uploaded_path.exists():
                    uploaded_size = uploaded_path.stat().st_size
                    local_size = local_file.stat().st_size
                    
                    if uploaded_size == local_size:
                        # Same content, just delete the non-UPLOADED version
                        self.logger.info(
                            f"Duplicate found, removing non-UPLOADED version: {filename}"
                        )
                        local_file.unlink()
                    else:
                        # Different sizes - keep the newer one (just uploaded)
                        self.logger.warning(
                            f"Duplicate with different size: {filename} ({local_size}) vs "
                            f"{uploaded_name} ({uploaded_size}), keeping newer upload"
                        )
                        uploaded_path.unlink()
                        local_file.rename(uploaded_path)
                else:
                    local_file.rename(uploaded_path)
                
                self.stats[f"{category}_moved"] += 1
                self.logger.info(f"Successfully uploaded: {filename}")
            else:
                # Mark as failed
                name_parts = filename.rsplit(".", 1)
                if len(name_parts) == 2:
                    failed_name = f"{name_parts[0]}_UPLOAD_FAILED.{name_parts[1]}"
                else:
                    failed_name = f"{filename}_UPLOAD_FAILED"
                failed_path = local_file.parent / failed_name
                
                # Handle case where _UPLOAD_FAILED version already exists
                if failed_path.exists():
                    self.logger.warning(f"Failed upload marker already exists: {failed_name}")
                    local_file.unlink()
                else:
                    local_file.rename(failed_path)

                self.stats[f"{category}_failed"] += 1
                self.logger.error(f"Failed to verify cloud copy: {filename}")

        except Exception as e:
            self.stats[f"{category}_failed"] += 1
            self.logger.error(f"Error uploading {local_file}: {str(e)}")
            self.stats["errors"].append(f"{local_file.name}: {str(e)}")

    def _verify_uid_consistency(self, local_file: str, cloud_file: str) -> bool:
        """Verify that local and cloud files have the same UID."""
        try:
            # Only check for image files that should have UIDs
            if not any(
                local_file.lower().endswith(ext)
                for ext in [".png", ".jpg", ".jpeg", ".tiff", ".tif"]
            ):
                return True  # Non-image files don't have UIDs

            local_uid = self.file_manager.extract_uid_from_any_image(local_file)
            cloud_uid = self.file_manager.extract_uid_from_any_image(cloud_file)

            # Both should have UIDs
            if local_uid is None and cloud_uid is None:
                self.logger.warning(
                    f"Neither file has UID: {os.path.basename(local_file)}"
                )
                return True  # No UIDs to compare
            elif local_uid is None or cloud_uid is None:
                self.logger.warning(
                    f"UID missing from one file: {os.path.basename(local_file)}"
                )
                return False  # One has UID, other doesn't
            else:
                return local_uid == cloud_uid

        except Exception as e:
            self.logger.error(f"Error verifying UID consistency: {e}")
            return False  # Assume mismatch on error

    def _cleanup_uploaded_files(
        self, local_base: Path, cloud_base: Path, category: str
    ):
        """Clean up files that have been successfully uploaded to cloud."""
        if not os.path.exists(local_base):
            self.logger.debug(
                f"Cleanup skipped - local base does not exist: {local_base}"
            )
            return

        self.logger.info(
            f"Starting cleanup of uploaded files in {local_base} (category: {category})"
        )
        cleaned_count = 0
        cleaned_bytes = 0

        for project_dir in self._safe_listdir(Path(local_base)):
            if self._check_cancelled():
                return
                
            project_path = Path(local_base) / project_dir
            if not project_path.is_dir():
                continue

            for hole_dir in self._safe_listdir(project_path):
                if self._check_cancelled():
                    return
                    
                hole_path = project_path / hole_dir
                if not hole_path.is_dir():
                    continue

                cloud_hole_path = cloud_base / project_dir / hole_dir

                # Find uploaded files
                for file_path in hole_path.glob("*_UPLOADED.*"):
                    if self._check_cancelled():
                        return
                        
                    # Get the original filename without _UPLOADED suffix
                    filename = file_path.name
                    clean_name = filename.replace("_UPLOADED", "")
                    cloud_file = cloud_hole_path / clean_name

                    # Verify file exists in cloud before deletion
                    if cloud_file.exists():
                        local_size = file_path.stat().st_size
                        cloud_size = cloud_file.stat().st_size

                        if local_size == cloud_size:
                            # Safe to delete local file
                            self.logger.debug(f"Deleting uploaded file: {file_path}")
                            file_size = local_size
                            os.remove(str(file_path))
                            cleaned_count += 1
                            cleaned_bytes += file_size
                            self.stats["bytes_freed"] += file_size

                            # Track specific cleanup stats
                            if "approved_originals" in category:
                                self.stats["approved_originals_cleaned"] += 1
                            elif "rejected_originals" in category:
                                self.stats["rejected_originals_cleaned"] += 1
                            elif "approved" in category:
                                self.stats["approved_cleaned"] += 1
                        else:
                            self.logger.warning(
                                f"Size mismatch for {filename}: local={local_size}, cloud={cloud_size}"
                            )
                            self.stats["missing_cloud_files"] += 1
                            self.stats["errors"].append(f"Size mismatch: {filename}")
                    else:
                        self.logger.warning(
                            f"Cloud file missing for uploaded file: {filename} -> {cloud_file}"
                        )
                        self.stats["missing_cloud_files"] += 1
                        self.stats["errors"].append(f"Missing in cloud: {filename}")

                # Clean up empty hole directory
                if hole_path.exists() and not any(hole_path.iterdir()):
                    self.logger.debug(f"Removing empty hole directory: {hole_path}")
                    hole_path.rmdir()

            # Clean up empty project directory
            if project_path.exists() and not any(project_path.iterdir()):
                self.logger.debug(f"Removing empty project directory: {project_path}")
                project_path.rmdir()

        self.logger.info(
            f"Cleaned up {cleaned_count} uploaded files, freed {cleaned_bytes / (1024*1024):.2f} MB"
        )

    def _sync_processed_originals(self):
        """Sync processed original images (approved and rejected) to cloud."""
        self.logger.info("Starting sync of processed original images")

        # Sync approved originals - use correct keys from file_manager
        local_approved = self.file_manager.dir_structure.get("approved_originals")
        cloud_approved = self.file_manager.get_shared_path("approved_originals")

        if local_approved and cloud_approved:
            self.logger.info(
                f"Syncing approved originals from {local_approved} to {cloud_approved}"
            )
            self._sync_original_images(
                local_approved, cloud_approved, "approved_originals"
            )
            # Clean up uploaded files
            self._cleanup_uploaded_files(
                local_approved, cloud_approved, "approved_originals"
            )
        else:
            self.logger.debug(
                f"Skipping approved originals sync - local: {local_approved}, cloud: {cloud_approved}"
            )

        # Sync rejected originals - use correct keys from file_manager
        local_rejected = self.file_manager.dir_structure.get("rejected_originals")
        cloud_rejected = self.file_manager.get_shared_path("rejected_originals")

        if local_rejected and cloud_rejected:
            self.logger.info(
                f"Syncing rejected originals from {local_rejected} to {cloud_rejected}"
            )
            self._sync_original_images(
                local_rejected, cloud_rejected, "rejected_originals"
            )
            # Clean up uploaded files
            self._cleanup_uploaded_files(
                local_rejected, cloud_rejected, "rejected_originals"
            )
        else:
            self.logger.debug(
                f"Skipping rejected originals sync - local: {local_rejected}, cloud: {cloud_rejected}"
            )

    def _final_directory_cleanup(self) -> int:
        """
        Final pass to clean up empty directories across all local sync folders.
        
        Returns:
            Total number of directories removed
        """
        total_removed = 0
        
        # List of local directories to check for empty folders
        dir_keys = [
            "temp_review",
            "approved_compartments", 
            "approved_originals",
            "rejected_originals",
            "pending_originals",
        ]
        
        for key in dir_keys:
            if self._check_cancelled():
                break
                
            local_path = self.file_manager.dir_structure.get(key)
            if local_path:
                path = Path(local_path)
                removed = self._cleanup_empty_directories(path)
                if removed > 0:
                    self.logger.info(f"Cleaned up {removed} empty directories in {key}")
                total_removed += removed
        
        return total_removed

    def _sync_original_images(self, local_base: Path, cloud_base: Path, category: str):
        """Sync original images folder to cloud."""
        if not os.path.exists(local_base):
            return

        for project_dir in self._safe_listdir(Path(local_base)):
            if self._check_cancelled():
                return
                
            project_path = Path(local_base) / project_dir
            if not project_path.is_dir():
                continue

            for hole_dir in self._safe_listdir(project_path):
                if self._check_cancelled():
                    return
                    
                hole_path = project_path / hole_dir
                if not hole_path.is_dir():
                    continue

                cloud_hole_path = cloud_base / project_dir / hole_dir
                cloud_hole_path.mkdir(parents=True, exist_ok=True)

                # Sync all images in the folder
                for file_path in hole_path.iterdir():
                    if self._check_cancelled():
                        return
                        
                    if file_path.is_file():
                        filename = file_path.name
                        # Check for upload status
                        if (
                            "_UPLOADED." not in filename
                            and "_UPLOAD_FAILED." not in filename
                        ):
                            self._upload_and_mark(file_path, cloud_hole_path, category)
                        elif "_UPLOAD_FAILED." in filename:
                            self._retry_failed_upload(
                                file_path, cloud_hole_path, category
                            )