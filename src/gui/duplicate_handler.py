# gui/duplicate_handler.py

import os
import re
import logging
import shutil
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from typing import Dict, List, Optional, Union, Any
import traceback

from gui.dialog_helper import DialogHelper

logger = logging.getLogger(__name__)


class DuplicateHandler:
    """
    Manages detection and handling of duplicate image processing entries.

    Tracks processed entries to prevent unintentional duplicate processing.
    Provides interactive dialogs for resolving duplicates with options to:
    - Skip and keep existing images
    - Replace all existing images
    - Selectively replace specific compartments
    - Modify metadata for the current image
    """

    def __init__(self, file_manager=None):
        """
        Initialize the duplicate handler.

        Args:
            file_manager: FileManager instance for handling file operations
        """
        self.file_manager = file_manager
        self.logger = logging.getLogger(__name__)

        # ===================================================
        # PERFORMANCE: Directory scan caching
        # ===================================================
        self._scan_cache = {}  # {directory_path: (timestamp, entries_dict)}
        self._cache_timeout = 300  # 5 minutes in seconds
        self._scan_shared_folders = True  # Can be disabled for performance

        self.processed_entries = self._load_existing_entries()
        self.parent = None  # Will be set by the parent component
        self.root = None  # Will be set based on parent

    # ===================================================
    # NEW METHOD: Extract project code helper
    # ===================================================
    def _get_project_code(self, hole_id: str) -> str:
        """Extract project code from hole ID."""
        return hole_id[:2].upper() if len(hole_id) >= 2 else ""

    # ===================================================
    # PERFORMANCE: Cache management methods
    # ===================================================
    def clear_scan_cache(self):
        """
        Clear the directory scan cache completely.

        Only use this when:
        - External processes modified files
        - You need to force a full refresh
        - Cache might be corrupt

        For normal file additions, use incremental updates instead.
        """
        self._scan_cache.clear()
        self.logger.info("Cleared directory scan cache (full refresh on next scan)")

    def refresh_cache_for_directory(self, directory: str):
        """
        Force refresh cache for a specific directory.

        Args:
            directory: The directory path to refresh
        """
        if directory in self._scan_cache:
            del self._scan_cache[directory]
            self.logger.debug(f"Cleared cache for directory: {directory}")

        # Trigger a re-scan by loading entries
        temp_entries = {}
        self._scan_directory_for_entries(directory, temp_entries)
        self._cache_scan_results(directory, temp_entries)
        self.logger.info(f"Refreshed cache for directory: {directory}")

    def set_scan_shared_folders(self, enabled: bool):
        """Enable/disable scanning of shared OneDrive folders."""
        if self._scan_shared_folders != enabled:
            self._scan_shared_folders = enabled
            self.clear_scan_cache()  # Clear cache when settings change
            self.logger.info(
                f"Shared folder scanning {'enabled' if enabled else 'disabled'}"
            )

    def ensure_loaded(self):
        """
        Trigger the folder scan if it hasn't happened yet (lazy init).
        Call this before any duplicate check during image processing.
        """
        if not self.processed_entries and not self._scan_shared_folders:
            self.logger.info("DuplicateHandler: triggering deferred folder scan")
            self._scan_shared_folders = True
            self.processed_entries = self._load_existing_entries()
            self.logger.info(
                f"Shared folder scanning {'enabled' if enabled else 'disabled'}"
            )

    def _get_cached_scan(self, directory: str):
        """
        Get cached scan results if available and not expired.

        Returns:
            Cached entries dict or None if cache miss/expired
        """
        if directory not in self._scan_cache:
            return None

        import time

        timestamp, entries = self._scan_cache[directory]
        age = time.time() - timestamp

        if age > self._cache_timeout:
            # Cache expired
            del self._scan_cache[directory]
            return None

        self.logger.debug(f"Cache HIT for {directory} (age: {age:.1f}s)")
        return entries

    def _cache_scan_results(self, directory: str, entries: dict):
        """Cache the scan results for a directory."""
        import time

        self._scan_cache[directory] = (time.time(), entries.copy())
        self.logger.debug(f"Cached scan results for {directory}")

    def _should_refresh_directory(self, directory: str) -> bool:
        """
        Check if a directory needs to be rescanned based on modification time.

        Args:
            directory: Directory path to check

        Returns:
            True if directory should be rescanned, False if cache is still valid
        """
        if directory not in self._scan_cache:
            return True

        import os
        import time

        try:
            cache_timestamp, _ = self._scan_cache[directory]
            dir_mtime = os.path.getmtime(directory)

            # If directory was modified after we cached it, refresh
            if dir_mtime > cache_timestamp:
                self.logger.debug(
                    f"Directory {directory} modified since cache, needs refresh"
                )
                return True

            # Check cache age
            age = time.time() - cache_timestamp
            if age > self._cache_timeout:
                self.logger.debug(f"Cache for {directory} expired (age: {age:.1f}s)")
                return True

            return False

        except OSError as e:
            self.logger.debug(f"Error checking directory mtime: {e}")
            return True  # Refresh on error

    def _update_cache_with_new_entry(self, entry_key: str, output_files: List[str]):
        """
        Incrementally update cache when new files are added.
        This avoids expensive re-scans during batch processing.

        Args:
            entry_key: The entry key (e.g., "BA0074_60-80")
            output_files: List of filenames that were just created
        """
        if not self._scan_cache:
            # No cache to update
            return

        # Update all cached directories that might contain these files
        updated_count = 0
        for directory, (timestamp, entries) in list(self._scan_cache.items()):
            # Add this entry to the cached results
            if entry_key not in entries:
                entries[entry_key] = []

            # Add the new files (avoiding duplicates)
            for filename in output_files:
                if filename not in entries[entry_key]:
                    entries[entry_key].append(filename)

            # Update the cache with modified entries (keep same timestamp)
            self._scan_cache[directory] = (timestamp, entries)
            updated_count += 1

        if updated_count > 0:
            self.logger.debug(
                f"Incrementally updated {updated_count} cached directories with new entry {entry_key}"
            )

    def _invalidate_cache_for_entry(self, entry_key: str):
        """
        Invalidate cached entries for a specific entry_key.
        Use this when files are deleted or moved.

        Args:
            entry_key: The entry key to remove from all caches
        """
        if not self._scan_cache:
            return

        invalidated_count = 0
        for directory, (timestamp, entries) in list(self._scan_cache.items()):
            if entry_key in entries:
                del entries[entry_key]
                self._scan_cache[directory] = (timestamp, entries)
                invalidated_count += 1

        if invalidated_count > 0:
            self.logger.debug(
                f"Invalidated cache for entry {entry_key} in {invalidated_count} directories"
            )

    def _generate_entry_key(self, hole_id: str, depth_from: int, depth_to: int) -> str:
        """
        Generate a unique key for a processed entry.

        Args:
            hole_id: Unique hole identifier
            depth_from: Starting depth
            depth_to: Ending depth

        Returns:
            Standardized key representing the entry
        """
        return f"{hole_id}_{depth_from}-{depth_to}"

    def is_duplicate(self, hole_id: str, depth_from: int, depth_to: int) -> bool:
        """
        Check if an entry is a duplicate without showing any dialog.
        Used for automated batch processing to skip duplicates.

        Args:
            hole_id: Unique hole identifier
            depth_from: Starting depth
            depth_to: Ending depth

        Returns:
            True if duplicate exists, False otherwise
        """
        # First check in-memory processed entries
        entry_key = self._generate_entry_key(hole_id, depth_from, depth_to)
        if entry_key in self.processed_entries:
            return True

        # Build list of expected depths (depth_from + 1 to depth_to inclusive)
        expected_depths = list(range(depth_from + 1, depth_to + 1))

        # Then check existing compartments on disk
        existing_compartments = self._scan_for_existing_compartments(
            hole_id, expected_depths
        )

        if existing_compartments:
            self.logger.debug(
                f"Found {len(existing_compartments)} existing compartments for "
                f"{hole_id} {depth_from}-{depth_to}m"
            )
            return True

        return False

    def _load_existing_entries(self) -> Dict[str, List[str]]:
        """
        Load existing processed entries from saved files across multiple directories.
        Checks Temp_Review/, Approved Compartment Images/, and shared folder storage.
        Handles both with and without project code folder structures.

        PERFORMANCE: Uses caching to avoid repeated slow OneDrive scans.

        Returns:
            Dictionary of processed entries where:
            - key: "{hole_id}_{depth_from}-{depth_to}"
            - value: List of filenames for that tray
        """
        entries = {}

        # ===================================================
        # FIXED: Use correct directory names from FileManager
        # ===================================================
        directories_to_check = []

        # Local directories - get from parent's FileManager if available
        if hasattr(self, "parent") and hasattr(self.parent, "file_manager"):
            file_manager = self.parent.file_manager

            # Add Compartment Images for Review directory
            local_qaqc_dir = file_manager.dir_structure.get("review_compartments")
            if local_qaqc_dir and os.path.exists(local_qaqc_dir):
                directories_to_check.append(("local_review", str(local_qaqc_dir)))

            # Add Extracted Compartment Images/Approved Compartment Images directory
            approved_compartments_dir = file_manager.dir_structure.get(
                "approved_compartments"
            )
            if approved_compartments_dir and os.path.exists(approved_compartments_dir):
                directories_to_check.append(
                    ("local_approved", str(approved_compartments_dir))
                )
        else:
            # Fallback if parent not set but file_manager available
            if self.file_manager:
                # Use FileManager's dir_structure
                local_qaqc_dir = self.file_manager.dir_structure.get(
                    "review_compartments"
                )
                if local_qaqc_dir and os.path.exists(local_qaqc_dir):
                    directories_to_check.append(("local_review", str(local_qaqc_dir)))

                approved_compartments_dir = self.file_manager.dir_structure.get(
                    "approved_compartments"
                )
                if approved_compartments_dir and os.path.exists(
                    approved_compartments_dir
                ):
                    directories_to_check.append(
                        ("local_approved", str(approved_compartments_dir))
                    )

        # ===================================================
        # PERFORMANCE: Optionally skip shared folders for speed
        # ===================================================
        if (
            self._scan_shared_folders
            and self.file_manager
            and self.file_manager.shared_paths
        ):
            try:
                # Get approved compartments folder from shared paths
                approved_folder = self.file_manager.get_shared_path(
                    "approved_compartments", create_if_missing=False
                )
                if approved_folder and os.path.exists(approved_folder):
                    directories_to_check.append(
                        ("shared_approved", str(approved_folder))
                    )

                # Also check review compartments folder
                review_folder = self.file_manager.get_shared_path(
                    "review_compartments", create_if_missing=False
                )
                if review_folder and os.path.exists(review_folder):
                    directories_to_check.append(("shared_review", str(review_folder)))
            except Exception as e:
                self.logger.debug(f"Could not access shared directories: {e}")
        elif not self._scan_shared_folders:
            self.logger.info("Skipping shared folder scan (disabled for performance)")

        # ===================================================
        # PERFORMANCE: Use caching to avoid repeated scans
        # ===================================================
        self.logger.info(
            f"Scanning {len(directories_to_check)} directories for unreviewed compartments:"
        )

        import time

        for dir_type, directory in directories_to_check:
            # Check if directory needs refresh based on modification time and age
            needs_refresh = self._should_refresh_directory(directory)

            if not needs_refresh:
                # Use cached results
                cached_entries = self._get_cached_scan(directory)
                if cached_entries is not None:
                    self.logger.info(f"  - Using CACHED results for: {directory}")
                    entries.update(cached_entries)
                    continue

            # Perform fresh scan (either cache expired, directory modified, or no cache)
            start_time = time.time()
            self.logger.info(f"  - Scanning: {directory}")

            dir_entries = {}
            self._scan_directory_for_entries(directory, dir_entries)

            scan_time = time.time() - start_time
            self.logger.info(
                f"  - Scanned in {scan_time:.2f}s ({len(dir_entries)} entries)"
            )

            # Cache the results
            self._cache_scan_results(directory, dir_entries)

            # Merge into main entries
            entries.update(dir_entries)

        return entries

    def _scan_directory_for_entries(
        self, directory: str, entries: Dict[str, List[str]]
    ):
        """
        Scan a directory for compartment images, handling both with and without project code folders.

        Args:
            directory: Directory to scan
            entries: Dictionary to update with found entries
        """
        if not os.path.exists(directory):
            return

        # Helper function to process hole ID directories
        def process_hole_id_directory(hole_id_dir_path: str, hole_id: str):
            """Process a single hole ID directory and extract compartment files."""
            compartment_files = {}  # Group files by depth

            try:
                for filename in os.listdir(hole_id_dir_path):
                    filepath = os.path.join(hole_id_dir_path, filename)

                    # Skip directories
                    if os.path.isdir(filepath):
                        continue

                    # Skip files without extensions
                    if "." not in filename:
                        continue

                    # Extract depth from compartment filename
                    # Pattern: XX1234_CC_123{anything}.{anything}
                    depth_match = re.search(r"_CC_(\d+)", filename)
                    if depth_match:
                        depth = int(depth_match.group(1))
                        if depth not in compartment_files:
                            compartment_files[depth] = []
                        compartment_files[depth].append(filename)

                # Group compartments into trays
                if compartment_files:
                    sorted_depths = sorted(compartment_files.keys())

                    # Find contiguous ranges of depths
                    current_start = sorted_depths[0]
                    current_end = sorted_depths[0]

                    for i in range(1, len(sorted_depths)):
                        if sorted_depths[i] == current_end + 1:
                            # Contiguous, extend the range
                            current_end = sorted_depths[i]
                        else:
                            # Gap found, record current range
                            entry_key = self._generate_entry_key(
                                hole_id, current_start - 1, current_end
                            )
                            if entry_key not in entries:
                                entries[entry_key] = []
                            entries[entry_key].extend(
                                [
                                    compartment_files[d][0]
                                    for d in range(current_start, current_end + 1)
                                    if d in compartment_files
                                ]
                            )

                            # Start new range
                            current_start = sorted_depths[i]
                            current_end = sorted_depths[i]

                    # Record final range
                    entry_key = self._generate_entry_key(
                        hole_id, current_start - 1, current_end
                    )
                    if entry_key not in entries:
                        entries[entry_key] = []
                    entries[entry_key].extend(
                        [
                            compartment_files[d][0]
                            for d in range(current_start, current_end + 1)
                            if d in compartment_files
                        ]
                    )

            except Exception as e:
                self.logger.debug(
                    f"Error processing hole directory {hole_id_dir_path}: {e}"
                )

        # Check if directory contains project code folders
        has_project_folders = False
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path) and re.match(r"^[A-Z]{2}$", item):
                has_project_folders = True
                break

        if has_project_folders:
            # Directory structure with project code folders
            for project_code in os.listdir(directory):
                project_path = os.path.join(directory, project_code)

                # Skip if not a directory or not a valid project code
                if not os.path.isdir(project_path) or not re.match(
                    r"^[A-Z]{2}$", project_code
                ):
                    continue

                # Look for hole ID directories within project folder
                for hole_id_dir in os.listdir(project_path):
                    hole_id_path = os.path.join(project_path, hole_id_dir)

                    # Skip if not a directory or not a valid hole ID
                    if not os.path.isdir(hole_id_path) or not re.match(
                        r"^[A-Z]{2}\d{4}$", hole_id_dir
                    ):
                        continue

                    process_hole_id_directory(hole_id_path, hole_id_dir)
        else:
            # Directory structure without project code folders
            for hole_id_dir in os.listdir(directory):
                hole_id_path = os.path.join(directory, hole_id_dir)

                # Skip if not a directory
                if not os.path.isdir(hole_id_path):
                    continue

                # Skip special directories
                if hole_id_dir.startswith(".") or hole_id_dir in ["Blur Analysis"]:
                    continue

                # Check if directory name matches hole ID pattern
                if not re.match(r"^[A-Z]{2}\d{4}$", hole_id_dir):
                    continue

                process_hole_id_directory(hole_id_path, hole_id_dir)

    # ===================================================
    # Scan for existing compartments with ANY suffix/extension
    # ===================================================
    def _scan_for_existing_compartments(
        self, hole_id: str, expected_depths: List[int]
    ) -> Dict[int, List[str]]:
        """
        Scan all relevant directories for existing compartment files.
        Matches ANY file with pattern: {HOLE_ID}_CC_{DEPTH}{ANY_SUFFIX}.{ANY_EXT}

        PERFORMANCE: Uses cached results from _load_existing_entries when available.

        Args:
            hole_id: Hole ID to search for
            expected_depths: List of depths to check

        Returns:
            Dict mapping depth to list of found file paths
        """
        existing = {}  # depth -> [file_paths]
        project_code = self._get_project_code(hole_id)

        # Build regex pattern that matches hole_id + depth, ignoring suffix and extension
        # Pattern: SB1234_CC_001{anything}
        pattern = re.compile(rf"^{re.escape(hole_id)}_CC_(\d{{3}})", re.IGNORECASE)

        # ===================================================
        # PERFORMANCE: Try to use cached results first
        # ===================================================
        # Check if we already have entries for this hole from cache
        cached_entries_found = False
        for entry_key, filenames in self.processed_entries.items():
            # Extract hole_id from entry_key (format: "BA0003_60-80")
            if entry_key.startswith(f"{hole_id}_"):
                for filename in filenames:
                    match = pattern.match(filename)
                    if match:
                        depth = int(match.group(1))
                        if depth in expected_depths:
                            # We need to find the full path - check cached directories
                            for directory, (
                                timestamp,
                                dir_entries,
                            ) in self._scan_cache.items():
                                if (
                                    entry_key in dir_entries
                                    and filename in dir_entries[entry_key]
                                ):
                                    # Found in cache - construct full path
                                    filepath = os.path.join(
                                        directory, project_code, hole_id, filename
                                    )
                                    if os.path.exists(filepath):
                                        if depth not in existing:
                                            existing[depth] = []
                                        if filepath not in existing[depth]:
                                            existing[depth].append(filepath)
                                            cached_entries_found = True

        if cached_entries_found:
            self.logger.debug(
                f"Found {sum(len(v) for v in existing.values())} existing compartments "
                f"for {hole_id} using cache"
            )
            # Still need to scan for any files not in processed_entries
            # but we can skip directories that are fully cached

        # ===================================================
        # FIXED: Define search locations using FileManager methods
        # ===================================================
        search_locations = []

        # Local directories
        if self.file_manager:
            # Get local approved compartments directory
            local_approved = self.file_manager.get_hole_dir(
                "approved_compartments", hole_id
            )
            if os.path.exists(local_approved):
                search_locations.append(("local_approved", local_approved))

            # Get local temp review directory
            local_temp = self.file_manager.get_hole_dir("temp_review", hole_id)
            if os.path.exists(local_temp):
                search_locations.append(("local_temp", local_temp))

            # Get shared directories
            if self.file_manager.shared_paths:
                # Shared approved compartments
                shared_approved_base = self.file_manager.get_shared_path(
                    "approved_compartments", create_if_missing=False
                )
                if shared_approved_base:
                    shared_approved = os.path.join(
                        shared_approved_base, project_code, hole_id
                    )
                    if os.path.exists(shared_approved):
                        search_locations.append(("shared_approved", shared_approved))

                # Shared review compartments
                shared_review_base = self.file_manager.get_shared_path(
                    "review_compartments", create_if_missing=False
                )
                if shared_review_base:
                    shared_review = os.path.join(
                        shared_review_base, project_code, hole_id
                    )
                    if os.path.exists(shared_review):
                        search_locations.append(("shared_review", shared_review))

        # Scan each location
        for location_name, path in search_locations:
            if path and os.path.exists(path):
                try:
                    for filename in os.listdir(path):
                        filepath = os.path.join(path, filename)

                        # Skip directories
                        if os.path.isdir(filepath):
                            continue

                        # Must have an extension
                        if "." not in filename:
                            continue

                        match = pattern.match(filename)
                        if match:
                            depth = int(match.group(1))
                            if depth in expected_depths:
                                if depth not in existing:
                                    existing[depth] = []
                                existing[depth].append(filepath)
                                self.logger.debug(
                                    f"Found existing compartment at {location_name}: {filename}"
                                )
                except Exception as e:
                    self.logger.debug(f"Error scanning {location_name}: {e}")

        return existing

    def _scan_all_compartment_locations(
        self, hole_id: str, expected_depths: List[int]
    ) -> Dict[int, List[str]]:
        """
        Scan ALL locations for existing compartment files.
        Checks both approved and review folders, both local and shared.
        """
        existing = {}
        project_code = self._get_project_code(hole_id)

        # Pattern matches compartment files with any suffix
        pattern = re.compile(rf"^{re.escape(hole_id)}_CC_(\d{{3}})", re.IGNORECASE)

        search_locations = []

        if self.file_manager:
            # Local locations
            for location_type in [
                "approved_compartments",
                "review_compartments",
                "temp_review",
            ]:
                local_dir = self.file_manager.get_hole_dir(location_type, hole_id)
                if local_dir and os.path.exists(local_dir):
                    search_locations.append((f"local_{location_type}", local_dir))

            # Shared locations
            if self.file_manager.shared_paths:
                for location_type in ["approved_compartments", "review_compartments"]:
                    shared_base = self.file_manager.get_shared_path(
                        location_type, create_if_missing=False
                    )
                    if shared_base:
                        shared_dir = os.path.join(shared_base, project_code, hole_id)
                        if os.path.exists(shared_dir):
                            search_locations.append(
                                (f"shared_{location_type}", shared_dir)
                            )

        # Scan each location
        for location_name, path in search_locations:
            try:
                for filename in os.listdir(path):
                    filepath = os.path.join(path, filename)
                    if os.path.isdir(filepath):
                        continue

                    match = pattern.match(filename)
                    if match:
                        depth = int(match.group(1))
                        if depth in expected_depths:
                            if depth not in existing:
                                existing[depth] = []
                            existing[depth].append(filepath)
                            self.logger.debug(
                                f"Found compartment at {location_name}: {filename}"
                            )

            except Exception as e:
                self.logger.debug(f"Error scanning {location_name}: {e}")

        return existing

    def _scan_all_original_locations(
        self, hole_id: str, depth_from: int, depth_to: int
    ) -> List[str]:
        """
        Scan ALL locations for existing original images.
        Checks approved and rejected originals in both local and shared locations.
        """
        existing_originals = []
        project_code = self._get_project_code(hole_id)

        # Pattern for original files
        pattern = re.compile(
            rf"^{re.escape(hole_id)}_{depth_from}-{depth_to}_.*\.(png|jpg|jpeg|tif|tiff)$",
            re.IGNORECASE,
        )

        search_locations = []

        if self.file_manager:
            # Local approved originals
            local_approved = self.file_manager.get_hole_dir(
                "approved_originals", hole_id
            )
            if local_approved and os.path.exists(local_approved):
                search_locations.append(("local_approved", local_approved))

            # Local rejected originals
            local_rejected = self.file_manager.get_hole_dir(
                "rejected_originals", hole_id
            )
            if local_rejected and os.path.exists(local_rejected):
                search_locations.append(("local_rejected", local_rejected))

            # Shared locations
            if self.file_manager.shared_paths:
                shared_processed = self.file_manager.get_shared_path(
                    "processed_originals", create_if_missing=False
                )
                if shared_processed:
                    # Check approved subfolder
                    shared_approved = os.path.join(
                        shared_processed, "Approved Originals", project_code, hole_id
                    )
                    if os.path.exists(shared_approved):
                        search_locations.append(("shared_approved", shared_approved))

                    # Check rejected subfolder
                    shared_rejected = os.path.join(
                        shared_processed, "Rejected Originals", project_code, hole_id
                    )
                    if os.path.exists(shared_rejected):
                        search_locations.append(("shared_rejected", shared_rejected))

        # Scan each location
        for location_name, path in search_locations:
            try:
                for filename in os.listdir(path):
                    if pattern.match(filename):
                        filepath = os.path.join(path, filename)
                        existing_originals.append(filepath)
                        self.logger.info(
                            f"Found original in {location_name}: {filename}"
                        )

            except Exception as e:
                self.logger.debug(f"Error scanning {location_name}: {e}")

        return existing_originals

    # REFACTORED METHOD: Now returns consistent action-based results
    def check_duplicate(
        self,
        hole_id: str,
        depth_from: int,
        depth_to: int,
        small_image: np.ndarray,
        full_filename: str,
        extracted_compartments: Optional[List[np.ndarray]] = None,
        register_manager=None,
        allow_interactive: bool = True,
    ) -> Dict[str, Any]:
        """
        Check if an entry is a potential duplicate and prompt user.

        Args:
            hole_id: Unique hole identifier
            depth_from: Starting depth
            depth_to: Ending depth
            small_image: Downsampled image for comparison
            full_filename: Full path to the original image file
            extracted_compartments: Optional list of already extracted compartment images
            register_manager: Optional register manager for checking entries
            allow_interactive: If False, skip for manual review instead of showing dialog

        Returns:
            Dict with action and parameters:
            - {"action": "continue"} - No duplicates, continue processing
            - {"action": "skip"} - Skip this image, keep existing
            - {"action": "replace_all"} - Replace all existing compartments
            - {"action": "selective_replacement", "selected_depths": [...]} - Replace selected
            - {"action": "keep_with_gaps", "missing_depths": [...]} - Keep existing, fill gaps
            - {"action": "modify_metadata", "hole_id": ..., "depth_from": ..., "depth_to": ...} - New metadata
            - {"action": "quit"} - User wants to stop processing
            - {"action": "needs_review"} - Duplicate requires manual review (batch mode only)
        """
        if self.parent is None:
            self.logger.warning(
                "DuplicateHandler parent not set, unable to show interactive dialogs"
            )
            return {"action": "continue"}  # No duplicates

        # Ensure we have a reference to the root window
        if self.root is None and hasattr(self.parent, "root"):
            self.root = self.parent.root

        # Store original path for potential qaqc processing
        self._current_image_path = full_filename

        # ===================================================
        # PERFORMANCE: Only reload if cache is stale
        # Instead of reloading every time, we rely on cache
        # Cache auto-expires after 5 minutes
        # ===================================================
        # NOTE: processed_entries is already loaded in __init__ with caching
        # No need to reload unless specifically requested

        # Build list of expected depths
        expected_depths = list(range(depth_from + 1, depth_to + 1))

        # Check for existing compartments in ALL locations
        existing_compartments = self._scan_all_compartment_locations(
            hole_id, expected_depths
        )

        # Check for existing original images
        existing_originals = self._scan_all_original_locations(
            hole_id, depth_from, depth_to
        )

        # Check register entries if manager provided
        register_entries = {}
        if register_manager:
            try:
                # Check compartment register
                comp_register_entries = (
                    register_manager.find_compartments_by_depth_range(
                        hole_id, depth_from, depth_to
                    )
                )
                if comp_register_entries:
                    register_entries["compartments"] = comp_register_entries
                    self.logger.info(
                        f"Found {len(comp_register_entries)} compartment register entries"
                    )

                # Check original images register
                orig_register_entry = register_manager.find_original_by_depth_range(
                    hole_id, depth_from, depth_to
                )
                if orig_register_entry:
                    register_entries["original"] = orig_register_entry
                    self.logger.info(f"Found original image register entry")
            except Exception as e:
                self.logger.debug(f"Error checking register entries: {e}")

        # Analyze what we found
        found_depths = set(existing_compartments.keys())
        expected_depths_set = set(expected_depths)

        # Calculate gaps and duplicates
        missing_depths = sorted(expected_depths_set - found_depths)
        duplicate_depths = sorted(found_depths & expected_depths_set)

        # Determine if we have ANY duplicates
        has_duplicates = bool(
            duplicate_depths or existing_originals or register_entries
        )

        # Log comprehensive findings
        self.logger.info(f"Duplicate check for {hole_id} {depth_from}-{depth_to}m:")
        self.logger.info(f"  - Expected depths: {expected_depths}")
        self.logger.info(
            f"  - Compartments found: {len(duplicate_depths)} of {len(expected_depths)}"
        )
        self.logger.info(f"  - Original images found: {len(existing_originals)}")
        self.logger.info(f"  - Register entries found: {bool(register_entries)}")
        self.logger.info(f"  - Missing depths: {missing_depths}")

        if not has_duplicates:
            self.logger.info("No duplicates found anywhere")
            return {"action": "continue"}

        # ===================================================
        # AUTO-HANDLE: All compartments missing
        # ===================================================
        # If NO compartment files exist (only register/originals), and we're in auto mode,
        # automatically replace. This handles the case where previous processing failed
        # but left register entries behind.
        if not duplicate_depths and len(missing_depths) == len(expected_depths):
            # Check if we're in auto mode (batch processing)
            in_auto_mode = False
            if hasattr(self.parent, "auto_mode"):
                in_auto_mode = self.parent.auto_mode

            if in_auto_mode:
                self.logger.info(
                    f"Auto-mode: All {len(expected_depths)} compartments missing, "
                    f"automatically replacing stale entries"
                )
                return {
                    "action": "replace_all",
                    "files_to_delete": [],  # No compartment files to delete
                    "existing_originals": existing_originals,
                    "register_entries": register_entries,
                    "existing_compartments": existing_compartments,
                }

        # Categorize the situation
        if not missing_depths:
            # All compartments already exist
            situation = "complete"
        elif len(duplicate_depths) < len(expected_depths_set) / 2:
            # Only a few exist
            situation = "partial_few"
        else:
            # Most exist but some are missing
            situation = "partial_most"

        # Check if interactive dialogs are allowed (for batch processing control)
        if not allow_interactive:
            self.logger.info(
                f"Batch mode: Duplicate found for {hole_id} {depth_from}-{depth_to}m, "
                f"flagging for manual review"
            )
            return {
                "action": "needs_review",
                "reason": f"Duplicate: {len(duplicate_depths)} existing, {len(missing_depths)} missing",
                "duplicate_depths": duplicate_depths,
                "missing_depths": missing_depths,
                "existing_compartments": existing_compartments,
            }

        # Show dialog and get user choice
        try:
            dialog_result = self._show_duplicate_dialog_enhanced(
                hole_id=hole_id,
                depth_from=depth_from,
                depth_to=depth_to,
                existing_compartments=existing_compartments,
                existing_originals=existing_originals,
                register_entries=register_entries,
                missing_depths=missing_depths,
                duplicate_depths=duplicate_depths,
                situation=situation,
                small_image=small_image,
                extracted_compartments=extracted_compartments,
                register_manager=register_manager,
            )

            # Ensure we always return a dict with action key
            if isinstance(dialog_result, dict):
                if "action" not in dialog_result:
                    # Legacy format conversion
                    if dialog_result.get("quit", False):
                        return {"action": "quit"}
                    elif dialog_result.get("skipped", False):
                        return {"action": "skip"}
                    elif dialog_result.get("selective_replacement", False):
                        return {
                            "action": "selective_replacement",
                            "selected_depths": dialog_result.get("selected_depths", []),
                        }
                    elif "hole_id" in dialog_result:
                        # Metadata modification
                        return {"action": "modify_metadata", **dialog_result}
                    else:
                        return {"action": "replace_all"}
                return dialog_result
            elif dialog_result == True:
                return {"action": "replace_all"}
            else:
                return {"action": "skip"}

        except Exception as e:
            self.logger.error(f"Error showing duplicate dialog: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {"action": "continue"}  # Continue on error

    def register_processed_entry(
        self, hole_id: str, depth_from: int, depth_to: int, output_files: List[str]
    ) -> None:
        """
        Register a successfully processed entry.

        Args:
            hole_id: Unique hole identifier
            depth_from: Starting depth
            depth_to: Ending depth
            output_files: List of output filenames created
        """
        entry_key = self._generate_entry_key(hole_id, depth_from, depth_to)
        self.processed_entries[entry_key] = output_files
        self.logger.info(
            f"Registered processed entry: {entry_key} with {len(output_files)} files"
        )

        # ===================================================
        # PERFORMANCE: Incrementally update cache instead of clearing
        # This is MUCH faster for batch processing workflows
        # ===================================================
        self._update_cache_with_new_entry(entry_key, output_files)

    def _move_to_rejected(
        self,
        files_to_move: List[str],
        hole_id: str,
        depth_from: int,
        depth_to: int,
        suffix: str = "Replaced",
    ) -> List[str]:
        """Move files to rejected folder with new suffix."""
        moved_files = []

        if not self.file_manager:
            return moved_files

        rejected_dir = self.file_manager.get_hole_dir("rejected_originals", hole_id)
        if not rejected_dir:
            self.logger.error(f"Could not get rejected folder for {hole_id}")
            return moved_files

        os.makedirs(rejected_dir, exist_ok=True)

        for file_path in files_to_move:
            try:
                if not os.path.exists(file_path):
                    continue

                # Get extension
                _, ext = os.path.splitext(file_path)

                # Build new filename with suffix
                base_name = f"{hole_id}_{depth_from}-{depth_to}_{suffix}"
                new_filename = f"{base_name}{ext}"
                new_path = os.path.join(rejected_dir, new_filename)

                # Handle duplicates
                counter = 1
                while os.path.exists(new_path):
                    new_filename = f"{base_name}_{counter}{ext}"
                    new_path = os.path.join(rejected_dir, new_filename)
                    counter += 1

                # Move the file
                shutil.move(file_path, new_path)
                moved_files.append(new_path)
                self.logger.info(
                    f"Moved {os.path.basename(file_path)} to rejected as {new_filename}"
                )

            except Exception as e:
                self.logger.error(f"Error moving file to rejected: {e}")

        return moved_files

    def _find_existing_originals(
        self, hole_id: str, depth_from: int, depth_to: int
    ) -> List[str]:
        """Find existing original images for this depth range."""
        existing_originals = []

        if not self.file_manager:
            return existing_originals

        # Check both approved and rejected originals
        for location in ["approved_originals", "rejected_originals"]:
            try:
                base_dir = self.file_manager.get_hole_dir(location, hole_id)
                if not base_dir or not os.path.exists(base_dir):
                    continue

                # Pattern for original files: HoleID_From-To_[Original|Rejected|Skipped|Selected_Compartments][_N].ext
                pattern = re.compile(
                    rf"^{re.escape(hole_id)}_{depth_from}-{depth_to}_"
                    r"(Original|Rejected|Skipped|Selected_Compartments)"
                    r"(_\d+)?\..*$",
                    re.IGNORECASE,
                )

                for filename in os.listdir(base_dir):
                    if pattern.match(filename):
                        full_path = os.path.join(base_dir, filename)
                        existing_originals.append(full_path)
                        self.logger.info(
                            f"Found existing original: {filename} in {location}"
                        )

            except Exception as e:
                self.logger.debug(f"Error searching for originals in {location}: {e}")

        return existing_originals

    def _get_theme_colors(self) -> Dict[str, str]:
        """
        Get current theme colors from GUI manager.

        Returns:
            Dictionary of theme colors
        """
        # Try to get theme colors from various sources
        if hasattr(self, "parent") and hasattr(self.parent, "gui_manager"):
            return self.parent.gui_manager.theme_colors
        elif hasattr(self, "root") and hasattr(self.root, "gui_manager"):
            return self.root.gui_manager.theme_colors
        elif (
            hasattr(self, "parent")
            and hasattr(self.parent, "app")
            and hasattr(self.parent.app, "gui_manager")
        ):
            return self.parent.app.gui_manager.theme_colors
        else:
            # Return default dark theme colors as fallback
            return {
                "background": "#1e1e1e",
                "secondary_bg": "#252526",
                "text": "#e0e0e0",
                "accent_blue": "#3a7ca5",
                "accent_green": "#4a8259",
                "accent_red": "#a54242",
                "field_bg": "#2d2d2d",
                "field_border": "#3f3f3f",
                "hover_highlight": "#3a3a3a",
                "accent_error": "#5c3a3a",
                "accent_valid": "#3a5c3a",
                "checkbox_bg": "#2d2d2d",
                "checkbox_fg": "#4a8259",
                "progress_bg": "#252526",
                "progress_fg": "#4a8259",
                "menu_bg": "#252526",
                "menu_fg": "#e0e0e0",
                "menu_active_bg": "#3a7ca5",
                "menu_active_fg": "#e0e0e0",
                "border": "#3f3f3f",
                "separator": "#3f3f3f",
            }

    def _show_duplicate_dialog(
        self,
        hole_id: str,
        depth_from: int,
        depth_to: int,
        existing_compartments: Dict[int, List[str]],
        missing_depths: List[int],
        duplicate_depths: List[int],
        situation: str,
        small_image: np.ndarray,
        extracted_compartments: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Show dialog for duplicate resolution with improved layout."""
        if self.root is None:
            self.logger.error(
                "Cannot show duplicate dialog: root window reference is missing"
            )
            return {"action": "quit"}

        # Create dialog - only basic parameters
        dialog = DialogHelper.create_dialog(
            parent=self.root,
            title=DialogHelper.t("Duplicate Compartments Found"),
            modal=False,
            topmost=False,
        )

        # Get DPI scale from dialog for use in content
        dpi_scale = getattr(dialog, "dpi_scale", 1.0)

        # Store result with default action as "quit"
        dialog.result = {"action": "quit"}

        # Get theme colors
        theme_colors = self._get_theme_colors()

        # ===================================================
        # FIXED: Create proper layout structure
        # ===================================================
        # Main container frame that holds everything
        main_container = ttk.Frame(dialog)
        main_container.pack(fill=tk.BOTH, expand=True)

        # ===================================================
        # FIXED: Create a frame for the scrollable content that takes full width
        # ===================================================
        scroll_container = ttk.Frame(main_container)
        scroll_container.pack(fill=tk.BOTH, expand=True)

        # Create scrollable area inside scroll container
        main_canvas = tk.Canvas(
            scroll_container, bg=theme_colors["background"], highlightthickness=0
        )
        v_scrollbar = ttk.Scrollbar(
            scroll_container, orient="vertical", command=main_canvas.yview
        )
        main_canvas.configure(yscrollcommand=v_scrollbar.set)

        # Pack scrollbar and canvas properly
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        content_frame = ttk.Frame(main_canvas)
        content_window = main_canvas.create_window(
            (0, 0), window=content_frame, anchor="nw"
        )

        # ===================================================
        # FIXED: Configure canvas to expand content frame to full width
        # ===================================================
        def configure_canvas_width(event=None):
            # Update the width of the content frame to match canvas
            canvas_width = event.width if event else main_canvas.winfo_width()
            main_canvas.itemconfig(content_window, width=canvas_width)

        # Bind to canvas resize
        main_canvas.bind("<Configure>", configure_canvas_width)

        # ===================================================
        # Scale fonts with DPI
        # ===================================================
        title_font_size = int(16 * dpi_scale)
        subtitle_font_size = int(14 * dpi_scale)
        normal_font_size = int(12 * dpi_scale)
        small_font_size = int(8 * dpi_scale)
        button_font_size = int(10 * dpi_scale)

        # ===================================================
        # Title Section
        # ===================================================
        title_frame = ttk.Frame(content_frame, padding="10 10 10 5")
        title_frame.pack(fill=tk.X)

        title_label = ttk.Label(
            title_frame,
            text=DialogHelper.t("Duplicate Compartments Found"),
            font=("Arial", title_font_size, "bold"),
        )
        title_label.pack()

        subtitle_label = ttk.Label(
            title_frame,
            text=f"{hole_id} - {depth_from}-{depth_to}m",
            font=("Arial", subtitle_font_size),
        )
        subtitle_label.pack(pady=(5, 0))

        # Status text
        status_text = f"{len(duplicate_depths)} existing compartment(s) found"
        if missing_depths:
            status_text += f", {len(missing_depths)} missing"

        status_label = ttk.Label(
            title_frame, text=status_text, font=("Arial", normal_font_size)
        )
        status_label.pack(pady=(5, 0))

        # ===================================================
        # Standardized depth range handling
        # ===================================================
        # Define the complete depth range consistently
        all_depths = list(range(depth_from + 1, depth_to + 1))
        num_compartments = len(all_depths)

        # Create a mapping for consistent depth handling
        depth_to_index = {depth: i for i, depth in enumerate(all_depths)}

        self.logger.info(
            f"Depth range: {depth_from}-{depth_to}m, Compartments: {all_depths}"
        )

        # ===================================================
        # FIXED: Calculate optimal compartment size to fit without scrolling
        # ===================================================
        # Get screen width to calculate dialog width
        screen_width = dialog.winfo_screenwidth()
        dialog_width = int(screen_width * 0.95)  # Match the landscape sizing

        # FIXED: Account for all padding and margins more accurately
        # - Dialog padding: 20 (10 each side)
        # - Content frame padding: 10 (5 each side)
        # - LabelFrame padding: 4 (2 each side)
        # - Container margins: 20 (for safety)
        # - Scrollbar: 20
        total_margins = int(80 * dpi_scale)  # Increased to account for all padding
        available_width = dialog_width - total_margins

        # FIXED: Include padding between compartments in calculation
        min_spacing = max(2, int(4 * dpi_scale))  # Increased spacing
        total_spacing = min_spacing * (
            num_compartments + 1
        )  # Add extra spacing for edges

        # Calculate width per compartment
        comp_width = (available_width - total_spacing) // max(num_compartments, 1)

        # FIXED: Limit maximum compartment width to prevent oversizing
        max_comp_width = int(150 * dpi_scale)  # Maximum reasonable size
        comp_width = min(comp_width, max_comp_width)

        # Get aspect ratio from actual compartments
        aspect_ratio = 2.0
        if extracted_compartments and len(extracted_compartments) > 0:
            for comp in extracted_compartments:
                if comp is not None:
                    h, w = comp.shape[:2]
                    if w > 0:
                        aspect_ratio = h / w
                        break

        # Calculate height based on width and aspect ratio
        comp_height = int(comp_width * aspect_ratio)

        # Apply reasonable height constraint
        max_height = int(300 * dpi_scale)
        comp_height = min(comp_height, max_height)

        self.logger.info(
            f"Dialog width: {dialog_width}, Available: {available_width}, Comp size: {comp_width}x{comp_height}"
        )

        # ===================================================
        # FIXED: Shared compartment rendering function with better layout
        # ===================================================
        def render_compartments(
            parent_frame, compartments_data, section_title, is_existing=True
        ):
            """
            Render a row of compartment images with consistent styling.

            Args:
                parent_frame: Parent frame to add compartments to
                compartments_data: Dict of depth -> image path (for existing) or list of images (for current)
                section_title: Title for the section
                is_existing: True for existing compartments, False for current
            """
            section = ttk.LabelFrame(parent_frame, text=section_title, padding="5")
            section.pack(fill=tk.X, pady=(0, 5) if is_existing else 0)

            # FIXED: Create a scrollable container for compartments if needed
            comp_canvas = tk.Canvas(
                section, bg=theme_colors["background"], highlightthickness=0
            )
            comp_canvas.pack(fill=tk.BOTH, expand=True)

            # Calculate total width needed
            total_width_needed = (comp_width * num_compartments) + (
                min_spacing * (num_compartments + 1)
            )

            # Set canvas height to accommodate compartments
            canvas_height = comp_height + int(30 * dpi_scale)  # Extra space for labels
            comp_canvas.configure(height=canvas_height)

            # Create frame inside canvas
            compartments_frame = ttk.Frame(comp_canvas)
            comp_canvas.create_window(0, 0, window=compartments_frame, anchor="nw")

            # If compartments would overflow, add horizontal scrollbar
            if total_width_needed > available_width:
                h_scrollbar = ttk.Scrollbar(
                    section, orient="horizontal", command=comp_canvas.xview
                )
                h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
                comp_canvas.configure(xscrollcommand=h_scrollbar.set)

            # Render compartments with proper spacing
            for i, depth in enumerate(all_depths):
                comp_container = ttk.Frame(compartments_frame)
                comp_container.grid(row=0, column=i, padx=min_spacing // 2, pady=5)

                # Depth label
                depth_label = ttk.Label(
                    comp_container, text=f"{depth}m", font=("Arial", small_font_size)
                )
                depth_label.pack()

                # Determine if we have an image for this depth
                has_image = False
                image_data = None

                if is_existing:
                    # For existing compartments, check the dict
                    if depth in compartments_data:
                        has_image = True
                        image_data = compartments_data[depth][0]  # First file path
                else:
                    # For current compartments, check the list
                    if i < len(compartments_data) and compartments_data[i] is not None:
                        has_image = True
                        image_data = compartments_data[i]

                if has_image:
                    try:
                        # Load and resize image
                        if is_existing:
                            # Load from file path
                            comp_img = cv2.imread(image_data)
                        else:
                            # Use numpy array directly
                            comp_img = image_data

                        if comp_img is not None:
                            comp_img = cv2.resize(
                                comp_img,
                                (comp_width, comp_height),
                                interpolation=cv2.INTER_AREA,
                            )
                            comp_rgb = cv2.cvtColor(comp_img, cv2.COLOR_BGR2RGB)
                            pil_img = Image.fromarray(comp_rgb)
                            tk_img = ImageTk.PhotoImage(image=pil_img)

                            # For current compartments, highlight if filling a gap
                            if not is_existing and depth in missing_depths:
                                img_frame = tk.Frame(
                                    comp_container,
                                    bg="green",
                                    highlightbackground="green",
                                    highlightthickness=2,
                                )
                                img_frame.pack()
                                img_label = ttk.Label(img_frame, image=tk_img)
                                img_label.pack()
                            else:
                                img_label = ttk.Label(comp_container, image=tk_img)
                                img_label.pack()

                            img_label.image = tk_img  # Keep reference
                        else:
                            raise ValueError("Failed to load image")

                    except Exception as e:
                        self.logger.error(
                            f"Error loading {'existing' if is_existing else 'current'} compartment at depth {depth}: {e}"
                        )
                        error_label = ttk.Label(
                            comp_container,
                            text="Error",
                            foreground="red",
                            font=("Arial", small_font_size),
                        )
                        error_label.pack()
                else:
                    # Missing compartment placeholder (only for existing)
                    if is_existing:
                        placeholder = tk.Frame(
                            comp_container,
                            bg="orange",
                            width=comp_width,
                            height=comp_height,
                            highlightbackground="orange",
                            highlightthickness=2,
                        )
                        placeholder.pack()
                        placeholder.pack_propagate(False)

                        missing_label = ttk.Label(
                            placeholder,
                            text=DialogHelper.t("Missing"),
                            background="orange",
                            font=("Arial", small_font_size),
                        )
                        missing_label.place(relx=0.5, rely=0.5, anchor="center")

            # Update scroll region after rendering
            compartments_frame.update_idletasks()
            comp_canvas.configure(scrollregion=comp_canvas.bbox("all"))

        # ===================================================
        # Compartment Comparison Section
        # ===================================================
        comp_frame = ttk.Frame(content_frame, padding="10")
        comp_frame.pack(fill=tk.X)

        # Render existing compartments
        render_compartments(
            comp_frame,
            existing_compartments,
            DialogHelper.t("Existing Compartments"),
            is_existing=True,
        )

        # Render current compartments
        render_compartments(
            comp_frame,
            extracted_compartments or [],
            DialogHelper.t("Current Compartments"),
            is_existing=False,
        )

        # ===================================================
        # Load existing original image if available
        # ===================================================
        existing_original_image = None
        existing_original_path = None

        if self.file_manager:
            for location in ["approved", "rejected"]:
                try:
                    if location == "approved":
                        base_dir = self.file_manager.dir_structure.get(
                            "processed_originals"
                        )
                        if base_dir:
                            approved_dir = os.path.join(base_dir, "Approved Originals")
                    else:
                        rejected_dir = self.file_manager.get_hole_dir(
                            "rejected_originals", hole_id
                        )
                        approved_dir = rejected_dir

                    if os.path.exists(approved_dir):
                        project_code = self._get_project_code(hole_id)
                        hole_dir = os.path.join(approved_dir, project_code, hole_id)

                        if os.path.exists(hole_dir):
                            pattern = re.compile(
                                rf"^{re.escape(hole_id)}_{depth_from}-{depth_to}.*_Original\..*$",
                                re.IGNORECASE,
                            )

                            for filename in os.listdir(hole_dir):
                                if pattern.match(filename):
                                    existing_original_path = os.path.join(
                                        hole_dir, filename
                                    )
                                    existing_original_image = cv2.imread(
                                        existing_original_path
                                    )
                                    if existing_original_image is not None:
                                        self.logger.info(
                                            f"Found existing original image: {filename}"
                                        )
                                        break
                except Exception as e:
                    self.logger.debug(
                        f"Error searching for original in {location}: {e}"
                    )

        # ===================================================
        # FULL IMAGE COMPARISON SECTION
        # ===================================================
        image_frame = ttk.LabelFrame(
            content_frame, text=DialogHelper.t("Full Image Comparison"), padding="10"
        )
        image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        image_display_frame = ttk.Frame(image_frame)
        image_display_frame.pack(fill=tk.BOTH, expand=True)

        # Store both images
        current_image_data = {
            "image": small_image,
            "label": "Current Image",
            "path": None,
        }

        existing_image_data = {
            "image": existing_original_image,
            "label": "Existing Original",
            "path": existing_original_path,
        }

        showing_current = tk.BooleanVar(value=True)

        # Image display label
        image_label = ttk.Label(image_display_frame)
        image_label.pack(expand=True)

        # Store current PhotoImage reference to properly release it
        current_photo_image = None

        def update_image_display():
            """Update the displayed image based on toggle."""
            nonlocal current_photo_image

            # Release previous image reference properly
            if current_photo_image:
                # Remove widget reference first
                if hasattr(image_label, "image"):
                    image_label.image = None
                # Then delete our reference
                del current_photo_image
                current_photo_image = None
                # Force immediate cleanup
                import gc

                gc.collect()

            if showing_current.get():
                img_data = current_image_data
            else:
                img_data = existing_image_data

            if img_data["image"] is None:
                image_label.configure(
                    image="", text=f"No {img_data['label']} Available"
                )
                return

            try:
                img = img_data["image"]
                h, w = img.shape[:2]

                # Use reasonable image display size
                image_display_frame.update_idletasks()
                available_width = int(dialog_width * 0.8)
                available_height = int(400 * dpi_scale)

                scale_w = available_width / w
                scale_h = available_height / h
                scale = min(scale_w, scale_h)

                new_width = int(w * scale)
                new_height = int(h * scale)

                resized = cv2.resize(
                    img, (new_width, new_height), interpolation=cv2.INTER_AREA
                )

                if len(resized.shape) == 3:
                    img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                else:
                    img_rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

                pil_img = Image.fromarray(img_rgb)
                current_photo_image = ImageTk.PhotoImage(image=pil_img)

                image_label.configure(image=current_photo_image, text="")
                image_label.image = current_photo_image  # Keep reference

            except Exception as e:
                self.logger.error(f"Error updating image display: {e}")
                image_label.configure(
                    image="", text=f"Error displaying {img_data['label']}"
                )

        # Toggle buttons
        toggle_frame = ttk.Frame(image_frame)
        toggle_frame.pack(pady=(0, 5))

        if existing_original_image is not None:
            ttk.Radiobutton(
                toggle_frame,
                text=DialogHelper.t("Show Current Image"),
                variable=showing_current,
                value=True,
                command=update_image_display,
            ).pack(side=tk.LEFT, padx=10)

            ttk.Radiobutton(
                toggle_frame,
                text=DialogHelper.t("Show Existing Original"),
                variable=showing_current,
                value=False,
                command=update_image_display,
            ).pack(side=tk.LEFT, padx=10)
        else:
            ttk.Label(
                toggle_frame,
                text=DialogHelper.t("Existing original image not found"),
                foreground="gray",
            ).pack()

        # Initial image display
        update_image_display()

        # ===================================================
        # METADATA EDITOR (Collapsible)
        # ===================================================
        metadata_frame = ttk.LabelFrame(
            content_frame, text=DialogHelper.t("Metadata Editor"), padding="10"
        )

        metadata_visible = tk.BooleanVar(value=False)

        def toggle_metadata():
            if metadata_visible.get():
                metadata_frame.pack(fill=tk.X, padx=10, pady=5)
            else:
                metadata_frame.pack_forget()

            content_frame.update_idletasks()
            main_canvas.configure(scrollregion=main_canvas.bbox("all"))

            # FIXED: Update dialog height after toggling metadata
            dialog.after(100, lambda: size_and_center_dialog(resize_only=True))

        # Metadata fields
        modified_metadata = {
            "hole_id": tk.StringVar(value=hole_id),
            "depth_from": tk.StringVar(value=str(depth_from)),
            "depth_to": tk.StringVar(value=str(depth_to)),
        }

        # Create metadata fields with proper theming
        fields_frame = ttk.Frame(metadata_frame)
        fields_frame.pack()

        # Get fonts for fields
        fonts = {"normal": ("Arial", normal_font_size)}
        if hasattr(self, "parent") and hasattr(self.parent, "gui_manager"):
            fonts = self.parent.gui_manager.fonts

        # Import the themed field creation
        from gui.widgets.field_with_label import create_field_with_label

        # Hole ID field
        hole_id_field, hole_id_entry = create_field_with_label(
            fields_frame,
            "Hole ID:",
            modified_metadata["hole_id"],
            theme_colors,
            fonts,
            translator=DialogHelper.t,
            field_type="entry",
            width=12,
        )

        # Depth range fields in a custom frame
        depth_frame = ttk.Frame(fields_frame)
        depth_frame.pack(fill=tk.X, pady=5)

        # Label
        depth_label = ttk.Label(
            depth_frame, text=DialogHelper.t("Depth Range:"), width=12, anchor="w"
        )
        depth_label.pack(side=tk.LEFT)

        # From field
        from gui.widgets.entry_with_validation import create_entry_with_validation

        depth_from_entry = create_entry_with_validation(
            depth_frame,
            modified_metadata["depth_from"],
            theme_colors,
            fonts["normal"],
            width=8,
        )
        depth_from_entry.pack(side=tk.LEFT, padx=(0, 5))

        # Separator
        ttk.Label(depth_frame, text="-").pack(side=tk.LEFT)

        # To field
        depth_to_entry = create_entry_with_validation(
            depth_frame,
            modified_metadata["depth_to"],
            theme_colors,
            fonts["normal"],
            width=8,
        )
        depth_to_entry.pack(side=tk.LEFT, padx=(5, 5))

        # Unit label
        ttk.Label(depth_frame, text="m").pack(side=tk.LEFT)

        def apply_metadata():
            # Ensure metadata editor is visible
            if not metadata_visible.get():
                DialogHelper.show_message(
                    dialog,
                    DialogHelper.t("Metadata Hidden"),
                    DialogHelper.t(
                        "Please make the metadata editor visible before applying changes"
                    ),
                    message_type="warning",
                )
                return

            try:
                new_hole_id = modified_metadata["hole_id"].get().strip().upper()
                new_depth_from = int(modified_metadata["depth_from"].get())
                new_depth_to = int(modified_metadata["depth_to"].get())

                if not re.match(r"^[A-Z]{2}\d{4}$", new_hole_id):
                    raise ValueError("Invalid hole ID format")
                if new_depth_to <= new_depth_from:
                    raise ValueError("Invalid depth range")

                dialog.result = {
                    "action": "modify_metadata",
                    "hole_id": new_hole_id,
                    "depth_from": new_depth_from,
                    "depth_to": new_depth_to,
                }
                dialog.destroy()

            except Exception as e:
                DialogHelper.show_message(
                    dialog,
                    DialogHelper.t("Validation Error"),
                    str(e),
                    message_type="error",
                )

        # Use modern button for Apply
        gui_manager = None
        if hasattr(self, "parent") and hasattr(self.parent, "gui_manager"):
            gui_manager = self.parent.gui_manager

        if gui_manager:
            apply_btn = gui_manager.create_modern_button(
                metadata_frame,
                text=DialogHelper.t("Apply Changes"),
                color=theme_colors["accent_blue"],
                command=apply_metadata,
            )
            apply_btn.pack(pady=10)
        else:
            ttk.Button(
                metadata_frame,
                text=DialogHelper.t("Apply Changes"),
                command=apply_metadata,
            ).pack(pady=10)

        # ===================================================
        # ACTION BUTTONS (Fixed at bottom)
        # ===================================================
        button_frame = ttk.Frame(main_container, padding="10")
        button_frame.pack(side=tk.BOTTOM, fill=tk.X)

        help_text = ttk.Label(
            button_frame,
            text=DialogHelper.t("Choose an action:"),
            font=("Arial", button_font_size),
        )
        help_text.pack(side=tk.LEFT, padx=10)

        buttons_container = ttk.Frame(button_frame)
        buttons_container.pack(side=tk.RIGHT)

        # Define button actions with consistent depth handling
        def on_keep_original():
            if missing_depths:
                dialog.result = {
                    "action": "keep_with_gaps",
                    "missing_depths": missing_depths,
                }
            else:
                dialog.result = {"action": "skip"}
            dialog.destroy()

        def on_replace_all():
            """Replace all existing compartments and original images with new ones FOR THIS SPECIFIC TRAY ONLY."""
            # Find existing original images too
            existing_originals = self._find_existing_originals(
                hole_id, depth_from, depth_to
            )

            confirm_message = (
                f"This will move existing images to rejected folder for:\n\n"
                f"{hole_id} {depth_from}-{depth_to}m\n\n"
                f"Found {len(existing_compartments)} compartment location(s)\n"
            )

            if existing_originals:
                confirm_message += (
                    f"Found {len(existing_originals)} original image(s)\n"
                )

            confirm_message += (
                "\nExisting files will be moved to rejected with 'Replaced' suffix.\n"
            )
            confirm_message += "Are you sure you want to replace them?"

            if DialogHelper.confirm_dialog(
                dialog, DialogHelper.t("Replace Compartments"), confirm_message
            ):
                files_to_move = []
                for depth, paths in existing_compartments.items():
                    # Use the standardized depth range check
                    if depth in all_depths:
                        files_to_move.extend(paths)

                self.logger.info(
                    f"Replace All: Will move {len(files_to_move)} existing compartment files to rejected for {hole_id} {depth_from}-{depth_to}m"
                )

                # Include original files to move
                if existing_originals:
                    files_to_move.extend(existing_originals)

                dialog.result = {
                    "action": "replace_all",
                    "files_to_delete": files_to_move,  # Keep this key name for backward compatibility with main.py
                    "existing_compartments": existing_compartments,
                    "existing_originals": existing_originals,
                    "hole_id": hole_id,
                    "depth_from": depth_from,
                    "depth_to": depth_to,
                    "depth_range": all_depths,  # Include standardized range
                }
                dialog.destroy()

        def on_selective_replace():
            dialog.result = {"action": "selective_replacement"}
            dialog.destroy()

        def on_modify_metadata():
            metadata_visible.set(not metadata_visible.get())
            toggle_metadata()
            if metadata_visible.get():
                modify_btn.configure(text=DialogHelper.t("Hide Metadata"))
            else:
                modify_btn.configure(text=DialogHelper.t("Modify Metadata"))

        def on_reject():
            """Handle rejection using DialogHelper's standard rejection handler."""

            # Create metadata callback to get current values
            def get_metadata():
                return {
                    "hole_id": hole_id,
                    "depth_from": depth_from,
                    "depth_to": depth_to,
                    "compartment_interval": 1,  # Default compartment interval
                }

            # Call the standard rejection handler
            rejection_result = DialogHelper.handle_rejection(
                dialog,
                small_image,  # Pass the image for rejection
                metadata_callback=get_metadata,
                cleanup_callback=None,
            )

            if rejection_result:
                # User confirmed rejection
                dialog.result = {
                    "action": "reject",
                    **rejection_result,  # Include all rejection details
                }
                dialog.destroy()

        def on_quit():
            if DialogHelper.confirm_dialog(
                dialog,
                DialogHelper.t("Stop Processing"),
                DialogHelper.t("Are you sure you want to stop processing?"),
            ):
                dialog.result = {"action": "quit"}
                dialog.destroy()

        def on_window_close():
            """Handle window close button - default to quit action."""
            dialog.result = {"action": "quit"}
            dialog.destroy()

        dialog.protocol("WM_DELETE_WINDOW", on_window_close)

        # Create buttons with proper styling
        if gui_manager:
            keep_btn = gui_manager.create_modern_button(
                buttons_container,
                text=DialogHelper.t("Keep Original"),
                color=theme_colors["accent_green"],
                command=on_keep_original,
            )
            keep_btn.pack(side=tk.LEFT, padx=3)

            replace_btn = gui_manager.create_modern_button(
                buttons_container,
                text=DialogHelper.t("Replace All"),
                color=theme_colors["accent_red"],
                command=on_replace_all,
            )
            replace_btn.pack(side=tk.LEFT, padx=3)

            selective_btn = gui_manager.create_modern_button(
                buttons_container,
                text=DialogHelper.t("Selective Replace"),
                color=theme_colors["accent_blue"],
                command=on_selective_replace,
            )
            selective_btn.pack(side=tk.LEFT, padx=3)

            modify_btn = gui_manager.create_modern_button(
                buttons_container,
                text=DialogHelper.t("Modify Metadata"),
                color=theme_colors["accent_blue"],
                command=on_modify_metadata,
            )
            modify_btn.pack(side=tk.LEFT, padx=3)

            reject_btn = gui_manager.create_modern_button(
                buttons_container,
                text=DialogHelper.t("Reject"),
                color=theme_colors["accent_red"],
                command=on_reject,
            )
            reject_btn.pack(side=tk.LEFT, padx=3)

            quit_btn = gui_manager.create_modern_button(
                buttons_container,
                text=DialogHelper.t("Quit"),
                color=theme_colors["accent_red"],
                command=on_quit,
            )
            quit_btn.pack(side=tk.LEFT, padx=3)
        else:
            # Fallback buttons
            ttk.Button(
                buttons_container,
                text=DialogHelper.t("Keep Original"),
                command=on_keep_original,
            ).pack(side=tk.LEFT, padx=3)
            ttk.Button(
                buttons_container,
                text=DialogHelper.t("Replace All"),
                command=on_replace_all,
            ).pack(side=tk.LEFT, padx=3)
            ttk.Button(
                buttons_container,
                text=DialogHelper.t("Selective Replace"),
                command=on_selective_replace,
            ).pack(side=tk.LEFT, padx=3)
            modify_btn = ttk.Button(
                buttons_container,
                text=DialogHelper.t("Modify Metadata"),
                command=on_modify_metadata,
            )
            modify_btn.pack(side=tk.LEFT, padx=3)
            ttk.Button(
                buttons_container, text=DialogHelper.t("Reject"), command=on_reject
            ).pack(side=tk.LEFT, padx=3)
            ttk.Button(
                buttons_container, text=DialogHelper.t("Quit"), command=on_quit
            ).pack(side=tk.LEFT, padx=3)

        # Configure canvas scrolling
        def configure_scroll(event=None):
            main_canvas.configure(scrollregion=main_canvas.bbox("all"))
            # Also update width when content changes
            if main_canvas.winfo_width() > 1:
                main_canvas.itemconfig(content_window, width=main_canvas.winfo_width())

        content_frame.bind("<Configure>", configure_scroll)

        # Enable mouse wheel scrolling
        def on_mousewheel(event):
            main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        main_canvas.bind_all("<MouseWheel>", on_mousewheel)

        # Update scroll region
        content_frame.update_idletasks()
        main_canvas.configure(scrollregion=main_canvas.bbox("all"))

        # FIXED: Improved sizing and centering function
        def size_and_center_dialog(resize_only=False):
            """Size the dialog for landscape orientation and center it."""
            # Ensure all content is rendered
            dialog.update_idletasks()
            content_frame.update_idletasks()

            # Get screen dimensions
            screen_width = dialog.winfo_screenwidth()
            screen_height = dialog.winfo_screenheight()

            # Use nearly full screen width for landscape
            desired_width = int(screen_width * 0.95)

            # FIXED: Calculate natural height based on actual content
            # Force canvas to update its scroll region
            main_canvas.configure(scrollregion=main_canvas.bbox("all"))
            dialog.update_idletasks()

            # Get the actual content height
            canvas_bbox = main_canvas.bbox("all")
            if canvas_bbox:
                content_height = canvas_bbox[3] - canvas_bbox[1]
            else:
                content_height = content_frame.winfo_reqheight()

            # Add height for title bar, button frame, and padding
            button_height = button_frame.winfo_reqheight()
            padding_height = int(100 * dpi_scale)  # Title bar and various paddings

            natural_height = content_height + button_height + padding_height

            # Constrain height to screen limits
            max_height = int(screen_height * 0.9)
            min_height = int(600 * dpi_scale)

            natural_height = max(min_height, min(natural_height, max_height))

            # Ensure minimum width
            desired_width = max(desired_width, int(1200 * dpi_scale))

            # Calculate center position
            if not resize_only:
                if (
                    self.root
                    and self.root.winfo_exists()
                    and self.root.winfo_viewable()
                ):
                    # Center on parent
                    parent_x = self.root.winfo_rootx()
                    parent_y = self.root.winfo_rooty()
                    parent_width = self.root.winfo_width()
                    parent_height = self.root.winfo_height()

                    x = parent_x + (parent_width - desired_width) // 2
                    y = parent_y + (parent_height - natural_height) // 2
                else:
                    # Center on screen
                    x = (screen_width - desired_width) // 2
                    y = (screen_height - natural_height) // 2

                # Ensure dialog stays on screen
                screen_margin = 50
                taskbar_margin = 100
                x = max(
                    screen_margin, min(x, screen_width - desired_width - screen_margin)
                )
                y = max(
                    screen_margin,
                    min(y, screen_height - natural_height - taskbar_margin),
                )

                # Apply the geometry with position
                dialog.geometry(f"{desired_width}x{natural_height}+{x}+{y}")
            else:
                # Just resize without repositioning
                current_x = dialog.winfo_x()
                current_y = dialog.winfo_y()
                dialog.geometry(
                    f"{desired_width}x{natural_height}+{current_x}+{current_y}"
                )

            # Ensure dialog is visible and on top
            dialog.deiconify()
            dialog.lift()
            dialog.focus_force()

            self.logger.info(
                f"Dialog sized to: {desired_width}x{natural_height}, content height: {content_height}"
            )

        # FIXED: Delay initial sizing to ensure all content is rendered
        dialog.after(200, size_and_center_dialog)

        # Wait for dialog
        dialog.wait_window()

        # Unbind mousewheel
        main_canvas.unbind_all("<MouseWheel>")

        # Log the final user action
        self.logger.info(
            f"Duplicate dialog result for {hole_id} {depth_from}-{depth_to}m: {dialog.result}"
        )

        # Clean up image references
        if current_photo_image:
            del current_photo_image

        return dialog.result

    def _show_duplicate_dialog_enhanced(
        self,
        hole_id: str,
        depth_from: int,
        depth_to: int,
        existing_compartments: Dict[int, List[str]],
        existing_originals: List[str],
        register_entries: Dict[str, Any],
        missing_depths: List[int],
        duplicate_depths: List[int],
        situation: str,
        small_image: np.ndarray,
        extracted_compartments: Optional[List[np.ndarray]] = None,
        register_manager=None,
    ) -> Dict[str, Any]:
        """Show enhanced duplicate dialog with register info."""
        # For now, call the regular dialog but include extra info in result
        result = self._show_duplicate_dialog(
            hole_id=hole_id,
            depth_from=depth_from,
            depth_to=depth_to,
            existing_compartments=existing_compartments,
            missing_depths=missing_depths,
            duplicate_depths=duplicate_depths,
            situation=situation,
            small_image=small_image,
            extracted_compartments=extracted_compartments,
        )

        # Add the additional information to the result
        if result.get("action") == "replace_all":
            result["existing_originals"] = existing_originals
            result["register_entries"] = register_entries
            result["register_manager"] = register_manager

        return result

    def handle_replace_all(self, replacement_info: Dict[str, Any]) -> bool:
        """
        Handle complete replacement of existing files and register entries.

        Args:
            replacement_info: Dict containing all replacement information

        Returns:
            True if successful, False otherwise
        """
        hole_id = replacement_info.get("hole_id")
        depth_from = replacement_info.get("depth_from")
        depth_to = replacement_info.get("depth_to")
        register_manager = replacement_info.get("register_manager")

        if not hole_id:
            return False

        success = True

        try:

            # 1. Move existing original images to rejected
            if "existing_originals" in replacement_info:
                for original_path in replacement_info["existing_originals"]:
                    if os.path.exists(original_path):
                        if self.file_manager:
                            rejected_dir = self.file_manager.get_hole_dir(
                                "rejected_originals", hole_id
                            )
                            if rejected_dir:
                                os.makedirs(rejected_dir, exist_ok=True)
                                _, ext = os.path.splitext(original_path)
                                base_name = (
                                    f"{hole_id}_{depth_from}-{depth_to}_Replaced"
                                )
                                new_filename = f"{base_name}{ext}"
                                new_path = os.path.join(rejected_dir, new_filename)

                                # Handle duplicates
                                counter = 1
                                while os.path.exists(new_path):
                                    new_filename = f"{base_name}_{counter}{ext}"
                                    new_path = os.path.join(rejected_dir, new_filename)
                                    counter += 1

                                shutil.move(original_path, new_path)
                                self.logger.info(
                                    f"Moved original to rejected: {os.path.basename(original_path)}"
                                )

            # 2. Remove existing compartments
            if "existing_compartments" in replacement_info:
                # Invalidate cache entries for files we're about to remove
                entry_key = self._generate_entry_key(hole_id, depth_from, depth_to)
                self._invalidate_cache_for_entry(entry_key)

                for depth, file_paths in replacement_info[
                    "existing_compartments"
                ].items():
                    for file_path in file_paths:
                        if os.path.exists(file_path):
                            try:
                                os.remove(file_path)
                                self.logger.info(
                                    f"Removed existing compartment: {os.path.basename(file_path)}"
                                )
                            except Exception as e:
                                self.logger.error(f"Failed to remove compartment: {e}")
                                success = False

            # 3. Update registers if manager provided
            if register_manager and "register_entries" in replacement_info:
                # Remove old compartment entries
                if "compartments" in replacement_info["register_entries"]:
                    for entry in replacement_info["register_entries"]["compartments"]:
                        uid = entry.get("uid")
                        if uid:
                            try:
                                register_manager.remove_compartment_by_uid(uid)
                                self.logger.info(
                                    f"Removed compartment register entry: {uid}"
                                )
                            except Exception as e:
                                self.logger.error(
                                    f"Failed to remove compartment register entry: {e}"
                                )

                # Remove old original entry
                if "original" in replacement_info["register_entries"]:
                    orig_uid = replacement_info["register_entries"]["original"].get(
                        "uid"
                    )
                    if orig_uid:
                        try:
                            register_manager.remove_original_by_uid(orig_uid)
                            self.logger.info(
                                f"Removed original register entry: {orig_uid}"
                            )
                        except Exception as e:
                            self.logger.error(
                                f"Failed to remove original register entry: {e}"
                            )

            return success

        except Exception as e:
            self.logger.error(f"Error during replace all operation: {e}")
            self.logger.error(traceback.format_exc())
            return False
