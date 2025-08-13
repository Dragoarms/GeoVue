"""
Register Synchronizer for updating registers based on folder contents.

This module scans approved compartment images and original images folders
to ensure the registers are up-to-date with actual files.

Author: George Symonds
Created: 2025
"""

import os
import re
import logging
import time
import cv2
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple, Any, Set
from datetime import datetime
from collections import defaultdict, Counter

from utils.json_register_manager import JSONRegisterManager


class CheckpointLogger:
    """Logger that provides checkpoint-based progress logging."""

    def __init__(self, logger: logging.Logger, checkpoint_interval: int = 100) -> None:
        """
        Initialize checkpoint logger.

        Args:
            logger: Base logger instance
            checkpoint_interval: Number of operations between checkpoints
        """
        self.logger = logger
        self.checkpoint_interval = checkpoint_interval
        self.counters: Dict[str, Dict[str, Any]] = {}
        self.start_time = time.time()

    def log_action(self, action_type: str, current_item: Optional[str] = None) -> None:
        """Log action with checkpoint intervals."""
        if action_type not in self.counters:
            self.counters[action_type] = {
                "count": 0,
                "last_logged": 0,
                "first_item": current_item,
                "last_item": current_item,
            }

        counter = self.counters[action_type]
        counter["count"] += 1
        counter["last_item"] = current_item

        # Log at checkpoints
        if counter["count"] - counter["last_logged"] >= self.checkpoint_interval:
            elapsed = time.time() - self.start_time
            rate = counter["count"] / elapsed if elapsed > 0 else 0

            self.logger.info(
                f"[CHECKPOINT] {action_type}: {counter['count']} processed "
                f"({counter['first_item']} to {counter['last_item']}) "
                f"Rate: {rate:.1f}/sec"
            )
            counter["last_logged"] = counter["count"]

    def final_summary(self) -> None:
        """Log final summary."""
        elapsed = time.time() - self.start_time
        self.logger.info(f"=== OPERATION SUMMARY ({elapsed:.1f}s) ===")
        for action_type, counter in self.counters.items():
            self.logger.info(f"  {action_type}: {counter['count']} total")


class RegisterSynchronizer:
    """Synchronizes register data with actual files in folders."""

    def __init__(
        self,
        file_manager: Any,
        config: Dict[str, Any],
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> None:
        """
        Initialize the register synchronizer.

        Args:
            file_manager: FileManager instance for file operations
            config: Configuration dictionary
            progress_callback: Optional callback for progress updates (message, percentage)
        """
        self.file_manager = file_manager
        self.config = config
        self.progress_callback = progress_callback
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.json_manager: Optional[JSONRegisterManager] = None
        self.checkpoint_logger = CheckpointLogger(self.logger)

        # Batch update queues
        self.compartment_updates: List[Dict[str, Any]] = []
        self.original_updates: List[Dict[str, Any]] = []

        # Path cache
        self._cached_paths: Dict[str, Optional[Path]] = {}
        self._cache_paths()

    def _cache_paths(self) -> None:
        """Cache all required paths at initialization."""
        try:
            # Shared paths
            path_mappings = {
                "approved_folder": "approved_originals",
                "processed_originals": "processed_originals",
                "rejected_folder": "rejected_originals",
                "register_data": "register_data",
                "approved_compartments": "approved_compartments",
                "review_compartments": "review_compartments",
            }

            for cache_key, path_key in path_mappings.items():
                try:
                    self._cached_paths[cache_key] = self.file_manager.get_shared_path(
                        path_key
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to cache shared path {path_key}: {e}")
                    self._cached_paths[cache_key] = None

            # Local paths
            local_mappings = {
                "local_approved_folder": "approved_originals",
                "local_processed_originals": "processed_originals",
                "local_rejected_folder": "rejected_originals",
                "local_approved_compartments": "approved_compartments",
                "local_review_compartments": "temp_review",
            }

            for cache_key, struct_key in local_mappings.items():
                try:
                    path = self.file_manager.dir_structure.get(struct_key)
                    self._cached_paths[cache_key] = Path(path) if path else None
                except Exception as e:
                    self.logger.warning(f"Failed to cache local path {struct_key}: {e}")
                    self._cached_paths[cache_key] = None

            self.logger.info("Path caching complete")

        except Exception as e:
            self.logger.error(f"Error caching paths: {e}")

    def set_json_manager(self, base_path: str) -> None:
        """
        Set up JSON manager for register operations.

        Args:
            base_path: Base directory path for register files
        """
        try:
            self.json_manager = JSONRegisterManager(base_path, self.logger)
            self.logger.info("JSON manager initialized for register synchronization")
        except Exception as e:
            self.logger.error(f"Failed to initialize JSON manager: {e}")
            self.json_manager = None

    def _report_progress(self, message: str, percentage: float) -> None:
        """Report progress if callback is available."""
        if self.progress_callback:
            try:
                self.progress_callback(message, percentage)
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {e}")

    # ===== MERGE OPERATIONS (Most should be in json_register_manager) =====

    def check_and_merge_user_registers(self) -> Dict[str, Any]:
        """
        Check for multiple user registers and merge if needed.

        Returns:
            Dictionary with merge results or None if no merge performed
        """
        if not self.json_manager:
            return {"success": False, "error": "No JSON manager available"}

        # Check if there are multiple user files
        has_multiple_users = False
        base_path = str(self.json_manager.base_path)

        for file_type in ["compartment", "original", "review", "corners"]:
            user_files = JSONRegisterManager.get_all_user_files_static(
                base_path, file_type
            )
            if len(user_files) > 1:
                has_multiple_users = True
                break

        if not has_multiple_users:
            return {"success": True, "merge_performed": False}

        # Get summary of all user data
        summary = JSONRegisterManager.get_data_summary_static(base_path)

        # Return summary for GUI to display
        return {
            "success": True,
            "merge_needed": True,
            "summary": summary,
            "has_multiple_users": has_multiple_users,
        }

    def perform_user_merge(self, progress_callback=None) -> Dict[str, Any]:
        """
        Perform the actual merge of user registers.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with merge statistics
        """
        if not self.json_manager:
            return {"success": False, "error": "No JSON manager available"}

        try:
            if progress_callback:
                progress_callback("Starting merge...", 10)

            # Use the JSON manager's efficient merge method
            merge_stats = self.json_manager.merge_all_user_registers_efficient(
                file_manager=self.file_manager,
                progress_callback=progress_callback,
                confirm_callback=None,  # No confirmation needed here, already done in GUI
            )

            if progress_callback:
                progress_callback("Merge complete", 100)

            return {"success": True, "stats": merge_stats}

        except Exception as e:
            self.logger.error(f"Error during merge: {str(e)}")
            return {"success": False, "error": str(e)}

    # ===== SYNCHRONIZATION OPERATIONS =====

    def synchronize_all(self) -> Dict[str, Any]:
        """
        Synchronize all registers with folders using transactions for safety.

        Returns:
            Dictionary with synchronization results
        """
        # Log path configuration for debugging
        self.log_path_configuration()

        results = {
            "success": True,
            "compartments_added": 0,
            "missing_compartments": 0,
            "originals_added": 0,
            "originals_updated": 0,
            "missing_files_compartments": 0,
            "missing_files_originals": 0,
            "hex_colors_calculated": 0,
            "hex_colors_failed": 0,
            "files_renamed": 0,
            "rename_failed": 0,
            "errors": [],
        }

        transaction = None

        try:
            # Clear batch queues
            self.compartment_updates.clear()
            self.original_updates.clear()

            # Clear batch queues
            self.compartment_updates.clear()
            self.original_updates.clear()

            # Step 1: Scan compartment images
            self._report_progress("Scanning compartment images...", 10)
            comp_results = self._sync_compartment_images()
            results["compartments_added"] = comp_results["added"]

            # Step 2: Scan original images
            self._report_progress("Scanning original images...", 30)
            orig_results = self._sync_original_images()
            results["originals_added"] = orig_results["added"]
            results["originals_updated"] = orig_results.get("updated", 0)

            # Step 3: Validate existing entries
            self._report_progress("Validating existing entries...", 50)
            validation_results = self._validate_existing_entries()
            results["missing_files_compartments"] = validation_results[
                "missing_compartments"
            ]
            results["missing_files_originals"] = validation_results["missing_originals"]

            # Step 4: Check for missing compartments
            self._report_progress("Checking for missing compartments...", 60)
            missing_results = self._check_missing_compartments()
            results["missing_compartments"] = missing_results["missing"]

            # Step 5: Calculate missing hex colors
            self._report_progress("Calculating missing hex colors...", 70)
            hex_results = self._calculate_missing_hex_colors()
            results["hex_colors_calculated"] = hex_results["calculated"]
            results["hex_colors_failed"] = hex_results["failed"]

            # Step 6: Sync wet/dry filenames
            self._report_progress("Syncing wet/dry filenames...", 80)
            wet_dry_results = self._sync_wet_dry_filenames()
            results["files_renamed"] = wet_dry_results["renamed"]
            results["rename_failed"] = wet_dry_results["failed"]
            if wet_dry_results["errors"]:
                results["errors"].extend(wet_dry_results["errors"])

            # Step 7: Perform batch updates
            self._report_progress("Writing updates to register...", 90)
            self._perform_batch_updates()

            # Log summary
            self.checkpoint_logger.final_summary()
            self._report_progress("Synchronization complete", 100)

        except Exception as e:
            self.logger.error(f"Synchronization error: {e}")
            results["success"] = False
            results["errors"].append(str(e))

        return results

    def _sync_compartment_images(self) -> Dict[str, int]:
        """
        Synchronize compartment images with register.

        Returns:
            Dictionary with results
        """
        results = {"added": 0}

        try:
            # Get existing entries
            existing_keys: Set[Tuple[str, int, int]] = set()
            if self.json_manager:
                existing_df = self.json_manager.get_compartment_data()
                if not existing_df.empty:
                    existing_keys = set(
                        (row["HoleID"], row["From"], row["To"])
                        for _, row in existing_df.iterrows()
                    )

            # Pattern for compartment files
            pattern = re.compile(
                r"^([A-Z]{2}\d{4})_CC_(\d{3,4})(?:_(Wet|Dry))?\.(?:png|jpg|jpeg|tiff|tif)$",
                re.IGNORECASE,
            )

            # Locations to scan
            locations = [
                ("Found", self._cached_paths.get("approved_compartments")),
                ("For Review", self._cached_paths.get("review_compartments")),
                ("Found", self._cached_paths.get("local_approved_compartments")),
                ("For Review", self._cached_paths.get("local_review_compartments")),
            ]

            # Track found compartments
            found_compartments: Set[Tuple[str, int]] = set()

            for photo_status, base_path in locations:
                if not base_path or not base_path.exists():
                    continue

                self.logger.info(f"Scanning {photo_status} folder: {base_path}")

                try:
                    for project_folder in os.listdir(str(base_path)):
                        project_path = base_path / project_folder
                        if not project_path.is_dir():
                            continue

                        for hole_folder in os.listdir(str(project_path)):
                            hole_path = project_path / hole_folder
                            if not hole_path.is_dir():
                                continue

                            for filename in os.listdir(str(hole_path)):
                                match = pattern.match(filename)
                                if match:
                                    hole_id = match.group(1)
                                    depth_to = int(match.group(2))
                                    wet_dry = match.group(3)

                                    # Skip if already found
                                    comp_key = (hole_id, depth_to)
                                    if comp_key in found_compartments:
                                        continue

                                    # Determine interval
                                    interval = self._detect_interval_for_hole(
                                        hole_id, existing_keys
                                    )
                                    depth_from = depth_to - interval

                                    # Check if already in register
                                    key = (hole_id, depth_from, depth_to)
                                    if key not in existing_keys:
                                        # Determine status
                                        if photo_status == "For Review":
                                            final_status = "For Review"
                                        elif wet_dry:
                                            final_status = f"OK_{wet_dry}"
                                        else:
                                            final_status = photo_status

                                        # Add to batch
                                        self.compartment_updates.append(
                                            {
                                                "hole_id": hole_id,
                                                "depth_from": depth_from,
                                                "depth_to": depth_to,
                                                "photo_status": final_status,
                                                "processed_by": "System Sync",
                                                "comments": f"Auto-imported from {filename}",
                                            }
                                        )

                                        results["added"] += 1
                                        existing_keys.add(key)
                                        found_compartments.add(comp_key)

                                        self.checkpoint_logger.log_action(
                                            "Compartments added",
                                            f"{hole_id} {depth_to}m",
                                        )

                except Exception as e:
                    self.logger.error(f"Error scanning {base_path}: {e}")

        except Exception as e:
            self.logger.error(f"Error in compartment sync: {e}")

        return results

    def _sync_original_images(self) -> Dict[str, int]:
        """
        Synchronize original images with register.

        Returns:
            Dictionary with results
        """
        results = {"added": 0, "updated": 0}

        try:
            # Get existing entries
            existing_entries: Dict[Tuple[str, int, int], Dict] = {}
            if self.json_manager:
                existing_df = self.json_manager.get_original_image_data()
                if not existing_df.empty:
                    for _, row in existing_df.iterrows():
                        if pd.notna(row.get("Depth_From")) and pd.notna(
                            row.get("Depth_To")
                        ):
                            key = (
                                row["HoleID"],
                                int(row["Depth_From"]),
                                int(row["Depth_To"]),
                            )
                            existing_entries[key] = row.to_dict()

            # Pattern for original files
            pattern = re.compile(
                r"^([A-Z]{2}\d{4})_(\d+)-(\d+)_Original(?:_(\d+))?\.(?:png|jpg|jpeg|tiff|tif)$",
                re.IGNORECASE,
            )

            # Dictionary to collect files by key
            files_by_key: Dict[Tuple[str, int, int], List[str]] = defaultdict(list)

            # Scan approved folder
            processed_path = self._cached_paths.get("processed_originals")
            if processed_path and processed_path.exists():
                self._scan_original_folder(
                    processed_path, pattern, files_by_key, "approved"
                )

            # Scan rejected folder
            rejected_path = self._cached_paths.get("rejected_folder")
            if rejected_path and rejected_path.exists():
                self._scan_original_folder(
                    rejected_path, pattern, files_by_key, "rejected"
                )

            # Process collected files
            for key, filenames in files_by_key.items():
                hole_id, depth_from, depth_to = key

                # Sort filenames
                filenames.sort()
                primary_filename = filenames[0]
                file_count = len(filenames)
                all_filenames = (
                    ", ".join(filenames) if file_count > 1 else primary_filename
                )

                if key in existing_entries:
                    # Check if update needed
                    existing = existing_entries[key]
                    existing_count = existing.get("File_Count", 1)
                    existing_filenames = existing.get("All_Filenames", "")

                    if (
                        existing_count != file_count
                        or existing_filenames != all_filenames
                        or "File_Count" not in existing
                    ):

                        # Update needed
                        self.original_updates.append(
                            {
                                "hole_id": hole_id,
                                "depth_from": depth_from,
                                "depth_to": depth_to,
                                "original_filename": primary_filename,
                                "is_approved": "approved" in filenames[0].lower(),
                                "upload_success": True,
                                "uploaded_by": existing.get(
                                    "Uploaded_By", "System Sync"
                                ),
                                "comments": f"Updated file count: {existing_count} -> {file_count}",
                                "scale_px_per_cm": existing.get("Scale_PxPerCm"),
                                "scale_confidence": existing.get("Scale_Confidence"),
                            }
                        )

                        results["updated"] += 1
                        self.checkpoint_logger.log_action(
                            "Originals updated", f"{hole_id} {depth_from}-{depth_to}m"
                        )
                else:
                    # New entry
                    self.original_updates.append(
                        {
                            "hole_id": hole_id,
                            "depth_from": depth_from,
                            "depth_to": depth_to,
                            "original_filename": primary_filename,
                            "is_approved": "approved" in filenames[0].lower(),
                            "upload_success": True,
                            "uploaded_by": "System Sync",
                            "comments": f"Auto-imported. File count: {file_count}",
                        }
                    )

                    results["added"] += 1
                    self.checkpoint_logger.log_action(
                        "Originals added", f"{hole_id} {depth_from}-{depth_to}m"
                    )

        except Exception as e:
            self.logger.error(f"Error in original sync: {e}")

        return results

    def _scan_original_folder(
        self,
        base_path: Path,
        pattern: re.Pattern,
        files_dict: Dict[Tuple[str, int, int], List[str]],
        folder_type: str,
    ) -> None:
        """Helper to scan an original images folder."""
        try:
            self.logger.info(f"Scanning {folder_type} folder: {base_path}")
            self.logger.debug(f"Pattern: {pattern.pattern}")

            project_count = 0
            hole_count = 0
            file_count = 0

            for project_folder in os.listdir(str(base_path)):
                project_path = base_path / project_folder
                if not project_path.is_dir():
                    continue

                project_count += 1

                for hole_folder in os.listdir(str(project_path)):
                    hole_path = project_path / hole_folder
                    if not hole_path.is_dir():
                        continue

                    hole_count += 1
                    hole_files = list(os.listdir(str(hole_path)))

                    if hole_files:
                        self.logger.debug(
                            f"\n  Scanning hole {hole_folder} ({len(hole_files)} files)"
                        )

                    for filename in hole_files:
                        match = pattern.match(filename)
                        if match:
                            hole_id = match.group(1)
                            depth_from = int(match.group(2))
                            depth_to = int(match.group(3))

                            key = (hole_id, depth_from, depth_to)
                            files_dict[key].append(filename)
                            file_count += 1

                            self.logger.debug(f"    - Matched: {filename} -> {key}")

                            # Log based on folder type
                            if folder_type == "approved":
                                self.checkpoint_logger.log_action(
                                    "Approved originals scanned",
                                    f"{hole_id} {depth_from}-{depth_to}m",
                                )
                            elif folder_type == "rejected":
                                self.checkpoint_logger.log_action(
                                    "Rejected originals scanned",
                                    f"{hole_id} {depth_from}-{depth_to}m",
                                )

        except Exception as e:
            self.logger.error(f"Error scanning {folder_type} folder: {e}")

    def _validate_existing_entries(self) -> Dict[str, int]:
        """Validate that files referenced in register still exist."""
        results = {"missing_compartments": 0, "missing_originals": 0}

        try:
            # Validate compartments
            comp_results = self._validate_compartment_images()
            results["missing_compartments"] = comp_results["missing"]

            # Validate originals
            orig_results = self._validate_original_images()
            results["missing_originals"] = orig_results["missing"]

        except Exception as e:
            self.logger.error(f"Validation error: {e}")

        return results

    def _validate_compartment_images(self) -> Dict[str, int]:
        """Check if compartment images still exist."""
        results = {"missing": 0}

        if not self.json_manager:
            return results

        try:
            compartments_df = self.json_manager.get_compartment_data()
            if compartments_df.empty:
                return results

            for _, row in compartments_df.iterrows():
                # Skip if already marked as missing
                if row.get("Photo_Status") == "MISSING_FILE":
                    continue

                hole_id = row["HoleID"]
                depth_to = int(row["To"])

                # Check if file exists
                if not self._find_compartment_image(hole_id, depth_to):
                    # File is missing
                    self.compartment_updates.append(
                        {
                            "hole_id": hole_id,
                            "depth_from": int(row["From"]),
                            "depth_to": depth_to,
                            "photo_status": "MISSING_FILE",
                            "processed_by": "System Validation",
                            "comments": f"File not found on {datetime.now().strftime('%Y-%m-%d')}",
                        }
                    )

                    results["missing"] += 1
                    self.checkpoint_logger.log_action(
                        "Missing compartments", f"{hole_id} {depth_to}m"
                    )

        except Exception as e:
            self.logger.error(f"Error validating compartments: {e}")

        return results

    def _validate_original_images(self) -> Dict[str, int]:
        """Check if original images still exist."""
        results = {"missing": 0}

        if not self.json_manager:
            return results

        try:
            originals_df = self.json_manager.get_original_image_data()
            if originals_df.empty:
                return results

            self.logger.debug(
                f"Validating {len(originals_df)} original image register entries"
            )
            updates_to_mark_missing = []

            for idx, row in originals_df.iterrows():
                # Refresh locks every 100 records
                if idx > 0 and idx % 100 == 0:
                    transaction = getattr(self, "_current_transaction", None)
                    if transaction and hasattr(
                        self.json_manager, "refresh_transaction_locks"
                    ):
                        if not self.json_manager.refresh_transaction_locks(transaction):
                            self.logger.warning("Failed to refresh transaction locks")

                hole_id = row["HoleID"]
                depth_from = int(row["Depth_From"])
                depth_to = int(row["Depth_To"])
                original_filename = row.get("Original_Filename", "Unknown")
                is_approved = row.get("Approved_Upload_Status") == "Uploaded"

                self.logger.debug(
                    f"\n--- Validating original image register entry {idx + 1}/{len(originals_df)} ---"
                )
                self.logger.debug(f"HoleID: {hole_id}, Depth: {depth_from}-{depth_to}m")
                self.logger.debug(f"Original filename: {original_filename}")
                self.logger.debug(
                    f"Status: {'Approved' if is_approved else 'Rejected'}"
                )

                # Check if file exists with renamed pattern
                file_found = self._check_original_file_exists(
                    hole_id,
                    depth_from,
                    depth_to,
                    is_approved,
                )

                self.logger.debug(f"File found: {file_found}")

                if not file_found:
                    # Queue for batch update
                    updates_to_mark_missing.append(
                        {
                            "hole_id": hole_id,
                            "depth_from": depth_from,
                            "depth_to": depth_to,
                            "original_filename": row.get(
                                "Original_Filename", "Unknown"
                            ),
                            "is_approved": row.get("Approved_Upload_Status")
                            == "Uploaded",
                            "upload_success": False,
                            "uploaded_by": "System Validation",
                            "comments": f"File not found on {datetime.now().strftime('%Y-%m-%d')}",
                        }
                    )

                    results["missing"] += 1
                    self.checkpoint_logger.log_action(
                        "Missing originals", f"{hole_id} {depth_from}-{depth_to}m"
                    )

            # Add to batch updates
            self.original_updates.extend(updates_to_mark_missing)

        except Exception as e:
            self.logger.error(f"Error validating originals: {e}")

        return results

    def _check_missing_compartments(self) -> Dict[str, int]:
        """Check for missing compartments based on original image ranges."""
        results = {"missing": 0}

        if not self.json_manager:
            return results

        try:
            # Get data
            originals_df = self.json_manager.get_original_image_data()
            if originals_df.empty:
                return results

            compartments_df = self.json_manager.get_compartment_data()

            # Build set of existing compartments
            existing_compartments: Set[Tuple[str, int]] = set()
            if not compartments_df.empty:
                for _, row in compartments_df.iterrows():
                    existing_compartments.add((row["HoleID"], row["To"]))

            # Check each original image range
            interval = self.config.get("compartment_interval", 1)

            for _, orig_row in originals_df.iterrows():
                hole_id = orig_row["HoleID"]
                depth_from = int(orig_row["Depth_From"])
                depth_to = int(orig_row["Depth_To"])

                # Calculate expected compartments
                current_depth = depth_from
                while current_depth < depth_to:
                    comp_depth_to = current_depth + interval

                    if (hole_id, comp_depth_to) not in existing_compartments:
                        # Missing compartment
                        self.compartment_updates.append(
                            {
                                "hole_id": hole_id,
                                "depth_from": current_depth,
                                "depth_to": comp_depth_to,
                                "photo_status": "Missing",
                                "processed_by": "System Check",
                                "comments": f"Missing from range {depth_from}-{depth_to}m",
                            }
                        )

                        results["missing"] += 1
                        existing_compartments.add((hole_id, comp_depth_to))

                        self.checkpoint_logger.log_action(
                            "Missing compartments detected",
                            f"{hole_id} {comp_depth_to}m",
                        )

                    current_depth = comp_depth_to

        except Exception as e:
            self.logger.error(f"Error checking missing compartments: {e}")

        return results

    def _calculate_missing_hex_colors(self) -> Dict[str, int]:
        """Calculate average hex colors for compartments missing this data."""
        results = {"calculated": 0, "failed": 0}

        if not self.json_manager:
            return results

        try:
            # Get compartment data
            compartments_df = self.json_manager.get_compartment_data()
            if compartments_df.empty:
                return results

            # Ensure column exists with proper handling
            if "Average_Hex_Color" not in compartments_df.columns:
                compartments_df["Average_Hex_Color"] = None
                self.logger.info("Added Average_Hex_Color column to compartment data")

            # Find entries needing hex color
            valid_statuses = ["OK_Wet", "OK_Dry", "Found", "Wet", "Dry"]

            # Create mask for valid statuses
            status_mask = compartments_df["Photo_Status"].isin(valid_statuses)

            # Create mask for missing hex colors
            # Handle None, empty string, NaN
            hex_mask = (
                compartments_df["Average_Hex_Color"].isna()
                | (compartments_df["Average_Hex_Color"] == "")
                | (compartments_df["Average_Hex_Color"].isnull())
            )

            # Combine masks
            needs_color = compartments_df[status_mask & hex_mask]

            if needs_color.empty:
                self.logger.info("No compartments need hex color calculation")
                return results

            self.logger.info(
                f"Found {len(needs_color)} compartments needing hex color calculation"
            )

            # Calculate colors in batches for better performance
            hex_updates = []
            batch_size = 50

            for batch_start in range(0, len(needs_color), batch_size):
                batch_end = min(batch_start + batch_size, len(needs_color))
                batch = needs_color.iloc[batch_start:batch_end]

                for _, row in batch.iterrows():
                    hole_id = row["HoleID"]
                    depth_to = int(row["To"])

                    image_path = self._find_compartment_image(hole_id, depth_to)
                    if image_path:
                        try:
                            hex_color = self._calculate_average_hex_color(
                                str(image_path)
                            )

                            if hex_color:
                                hex_updates.append(
                                    {
                                        "hole_id": hole_id,
                                        "depth_from": int(row["From"]),
                                        "depth_to": depth_to,
                                        "average_hex_color": hex_color,
                                    }
                                )
                                results["calculated"] += 1
                            else:
                                results["failed"] += 1
                                self.logger.warning(
                                    f"Failed to calculate hex for {hole_id} {depth_to}m"
                                )
                        except Exception as e:
                            self.logger.error(
                                f"Error calculating hex for {hole_id} {depth_to}m: {e}"
                            )
                            results["failed"] += 1
                    else:
                        results["failed"] += 1
                        self.logger.warning(
                            f"Image not found for {hole_id} {depth_to}m"
                        )

                    self.checkpoint_logger.log_action(
                        "Hex colors processed", f"{hole_id} {depth_to}m"
                    )

            # Batch update all hex colors at once
            if hex_updates:
                successful = self.json_manager.batch_update_compartment_colors(
                    hex_updates
                )
                self.logger.info(f"Updated {successful} hex colors in batch")

        except Exception as e:
            self.logger.error(f"Error calculating hex colors: {e}")
            import traceback

            self.logger.error(traceback.format_exc())

        return results

    def _sync_wet_dry_filenames(self) -> Dict[str, Any]:
        """Rename files to include _Wet or _Dry suffix based on register data."""
        results = {"renamed": 0, "failed": 0, "errors": []}

        approved_path = self._cached_paths.get("approved_compartments")
        if not approved_path or not approved_path.exists():
            self.logger.warning("Approved compartments folder not found")
            return results

        if not self.json_manager:
            return results

        try:
            # Get wet/dry entries
            compartments_df = self.json_manager.get_compartment_data()
            if compartments_df.empty:
                return results

            wet_dry_entries = compartments_df[
                compartments_df["Photo_Status"].isin(["OK_Wet", "OK_Dry"])
            ]

            if wet_dry_entries.empty:
                return results

            # Process each entry
            for _, row in wet_dry_entries.iterrows():
                hole_id = row["HoleID"]
                depth_to = int(row["To"])
                status = row["Photo_Status"]

                expected_suffix = "_Wet" if status == "OK_Wet" else "_Dry"

                # Find and rename if needed
                project_code = hole_id[:2].upper()
                hole_path = approved_path / project_code / hole_id

                if not hole_path.exists():
                    continue

                # Look for file
                for filename in os.listdir(str(hole_path)):
                    if f"{hole_id}_CC_{depth_to:03d}" in filename:
                        # Check if file already has the correct suffix
                        if expected_suffix in filename:
                            self.logger.debug(
                                f"File already has correct suffix: {filename}"
                            )
                            continue

                        # Check if file has the wrong suffix
                        wrong_suffix = "_Dry" if expected_suffix == "_Wet" else "_Wet"
                        if wrong_suffix in filename:
                            # File has wrong suffix, needs correction
                            old_path = str(hole_path / filename)

                            # Replace wrong suffix with correct one
                            new_filename = filename.replace(
                                wrong_suffix, expected_suffix
                            )
                            new_path = str(hole_path / new_filename)

                            if self.file_manager.rename_file_safely(old_path, new_path):
                                results["renamed"] += 1
                                self.checkpoint_logger.log_action(
                                    "Files renamed", f"{filename} -> {new_filename}"
                                )
                            else:
                                results["failed"] += 1
                                results["errors"].append(f"Failed to rename {filename}")
                        elif "_Wet" not in filename and "_Dry" not in filename:
                            # File has no suffix, needs one added
                            old_path = str(hole_path / filename)

                            # Build new name
                            name_parts = filename.split(".")
                            if len(name_parts) > 1:
                                extension = name_parts[-1]
                                base_name = ".".join(name_parts[:-1])
                                new_filename = (
                                    f"{base_name}{expected_suffix}.{extension}"
                                )
                                new_path = str(hole_path / new_filename)

                                if self.file_manager.rename_file_safely(
                                    old_path, new_path
                                ):
                                    results["renamed"] += 1
                                    self.checkpoint_logger.log_action(
                                        "Files renamed", new_filename
                                    )
                                else:
                                    results["failed"] += 1
                                    results["errors"].append(
                                        f"Failed to rename {filename}"
                                    )

        except Exception as e:
            self.logger.error(f"Error syncing wet/dry filenames: {e}")
            results["errors"].append(str(e))

        return results

    def refresh_transaction_locks(self, transaction: Dict[str, Any]) -> bool:
        """
        Refresh all locks held by a transaction if needed.
        Should be called periodically during long operations.

        Returns:
            True if all locks were refreshed successfully
        """
        try:
            current_time = time.time()
            # Refresh if more than 30 seconds since last refresh
            if current_time - transaction.get("last_refresh", 0) < 30:
                return True

            all_refreshed = True
            for name, lock in transaction.get("locks_held", []):
                if not self._refresh_file_lock(lock):
                    all_refreshed = False
                    self.logger.error(f"Failed to refresh lock for {name}")

            if all_refreshed:
                transaction["last_refresh"] = current_time

            return all_refreshed

        except Exception as e:
            self.logger.error(f"Error refreshing transaction locks: {e}")
            return False

    def _perform_batch_updates(self) -> None:
        """Perform all batch updates at once."""
        if not self.json_manager:
            self.logger.warning("No JSON manager available for batch updates")
            return

        try:
            # Store transaction reference if we have one
            transaction = getattr(self, "_current_transaction", None)

            # Update compartments
            if self.compartment_updates:
                self.logger.info(
                    f"Performing {len(self.compartment_updates)} compartment updates"
                )

                # Refresh locks if in transaction
                if transaction and hasattr(
                    self.json_manager, "refresh_transaction_locks"
                ):
                    self.json_manager.refresh_transaction_locks(transaction)

                successful = self.json_manager.batch_update_compartments(
                    self.compartment_updates
                )
                self.logger.info(
                    f"Successfully updated {successful} compartment entries"
                )

            # Batch update originals
            if self.original_updates:
                self.logger.info(
                    f"Performing {len(self.original_updates)} original updates in batch"
                )

                # Use transaction for batch updates
                transaction = None
                try:
                    if hasattr(self.json_manager, "begin_transaction"):
                        transaction = self.json_manager.begin_transaction()

                    # Read current data once
                    original_data = self.json_manager._read_json_file(
                        self.json_manager.original_json_path,
                        self.json_manager.original_lock,
                    )

                    # Create lookup for faster updates
                    originals_map = {}
                    for record in original_data:
                        key = (
                            record["HoleID"],
                            record["Depth_From"],
                            record["Depth_To"],
                            record["Original_Filename"],
                        )
                        originals_map[key] = record

                    # Apply all updates
                    successful = 0
                    for update in self.original_updates:
                        key = (
                            update["hole_id"],
                            update["depth_from"],
                            update["depth_to"],
                            update["original_filename"],
                        )

                        if key in originals_map:
                            # Update existing record
                            record = originals_map[key]
                            record["Uploaded_By"] = update.get(
                                "uploaded_by", "System Sync"
                            )
                            record["Comments"] = update.get("comments")

                            if update["is_approved"]:
                                record["Approved_Upload_Status"] = (
                                    "Uploaded" if update["upload_success"] else "Failed"
                                )
                            else:
                                record["Rejected_Upload_Status"] = "Failed"

                            successful += 1
                        else:
                            # Add new record
                            new_record = {
                                "HoleID": update["hole_id"],
                                "Depth_From": update["depth_from"],
                                "Depth_To": update["depth_to"],
                                "Original_Filename": update["original_filename"],
                                "File_Count": 1,
                                "All_Filenames": update["original_filename"],
                                "Approved_Upload_Date": (
                                    datetime.now().isoformat()
                                    if update["is_approved"]
                                    else None
                                ),
                                "Approved_Upload_Status": (
                                    (
                                        "Uploaded"
                                        if update["upload_success"]
                                        else "Failed"
                                    )
                                    if update["is_approved"]
                                    else None
                                ),
                                "Rejected_Upload_Date": (
                                    datetime.now().isoformat()
                                    if not update["is_approved"]
                                    else None
                                ),
                                "Rejected_Upload_Status": (
                                    "Failed" if not update["is_approved"] else None
                                ),
                                "Uploaded_By": update.get("uploaded_by", "System Sync"),
                                "Comments": update.get("comments"),
                                "Scale_PxPerCm": update.get("scale_px_per_cm"),
                                "Scale_Confidence": update.get("scale_confidence"),
                            }
                            original_data.append(new_record)
                            successful += 1

                    # Write once
                    self.json_manager._write_json_file(
                        self.json_manager.original_json_path,
                        self.json_manager.original_lock,
                        original_data,
                    )

                    if transaction:
                        self.json_manager.commit_transaction(transaction)

                    self.logger.info(
                        f"Successfully batch updated {successful} original entries"
                    )

                except Exception as e:
                    if transaction:
                        self.json_manager.rollback_transaction(transaction)
                    raise e

        except Exception as e:
            self.logger.error(f"Error in batch updates: {e}")

    # ===== HELPER METHODS =====

    def _detect_interval_for_hole(
        self, hole_id: str, existing_keys: Set[Tuple[str, int, int]]
    ) -> int:
        """Detect the compartment interval for a specific hole."""
        # Find existing compartments for this hole
        hole_compartments = [
            (from_depth, to_depth)
            for h_id, from_depth, to_depth in existing_keys
            if h_id == hole_id
        ]

        if len(hole_compartments) < 2:
            return self.config.get("compartment_interval", 1)

        # Sort by to_depth
        hole_compartments.sort(key=lambda x: x[1])

        # Calculate intervals
        intervals = []
        for i in range(1, len(hole_compartments)):
            interval = hole_compartments[i][1] - hole_compartments[i - 1][1]
            if interval > 0:
                intervals.append(interval)

        if not intervals:
            return self.config.get("compartment_interval", 1)

        # Return most common interval
        interval_counts = Counter(intervals)
        most_common = interval_counts.most_common(1)[0][0]

        self.logger.debug(f"Detected interval for {hole_id}: {most_common}m")
        return int(most_common)

    def _find_compartment_image(self, hole_id: str, depth_to: int) -> Optional[Path]:
        """Find the compartment image file path."""
        project_code = hole_id[:2].upper()

        self.logger.debug(
            f"\nSearching for compartment image: {hole_id} at {depth_to}m"
        )
        self.logger.debug(f"Project code: {project_code}")

        patterns = [
            f"{hole_id}_CC_{depth_to:03d}_Wet",
            f"{hole_id}_CC_{depth_to:03d}_Dry",
            f"{hole_id}_CC_{depth_to:03d}",
        ]

        self.logger.debug(f"Looking for patterns: {patterns}")

        locations = [
            (
                "shared approved_compartments",
                self._cached_paths.get("approved_compartments"),
            ),
            (
                "shared review_compartments",
                self._cached_paths.get("review_compartments"),
            ),
            (
                "local approved_compartments",
                self._cached_paths.get("local_approved_compartments"),
            ),
            (
                "local review_compartments",
                self._cached_paths.get("local_review_compartments"),
            ),
        ]

        for location_name, base_path in locations:
            self.logger.debug(f"\nChecking {location_name}: {base_path}")

            if not base_path:
                self.logger.debug(f"  - Path is None")
                continue

            if not base_path.exists():
                self.logger.debug(f"  - Path does not exist")
                continue

            hole_path = base_path / project_code / hole_id
            self.logger.debug(f"  - Full hole path: {hole_path}")

            if not hole_path.exists():
                self.logger.debug(f"  - Hole directory does not exist")
                continue

            # List files in directory for debugging
            try:
                files = list(os.listdir(str(hole_path)))
                self.logger.debug(f"  - Found {len(files)} files in directory")

                for pattern in patterns:
                    for ext in [".png", ".jpg", ".jpeg", ".tiff", ".tif"]:
                        filename = f"{pattern}{ext}"
                        file_path = hole_path / filename
                        self.logger.debug(f"  - Checking: {filename}")

                        if file_path.exists():
                            self.logger.debug(f"  - FOUND: {file_path}")
                            return file_path

                if files:
                    self.logger.debug(f"  - Sample files in directory: {files[:5]}")

            except Exception as e:
                self.logger.debug(f"  - Error listing directory: {e}")

        self.logger.debug(f"Compartment image not found in any location")
        return None

    def _check_original_file_exists(
        self, hole_id: str, depth_from: int, depth_to: int, is_approved: bool
    ) -> bool:
        """Check if an original file exists using proper naming convention."""
        try:
            project_code = hole_id[:2].upper()

            self.logger.debug(
                f"Searching for original file: {hole_id} {depth_from}-{depth_to}m"
            )
            self.logger.debug(f"Project code: {project_code}")

            if is_approved:
                base_paths = [
                    (
                        "shared processed_originals",
                        self._cached_paths.get("processed_originals"),
                    ),
                    (
                        "local processed_originals",
                        self._cached_paths.get("local_processed_originals"),
                    ),
                ]
            else:
                base_paths = [
                    (
                        "shared rejected_folder",
                        self._cached_paths.get("rejected_folder"),
                    ),
                    (
                        "local rejected_folder",
                        self._cached_paths.get("local_rejected_folder"),
                    ),
                ]

            # Build the expected filename pattern
            # Files are renamed to: HoleID_From-To_Original.ext (or _Rejected, _Skipped, etc.)
            pattern = f"{hole_id}_{depth_from}-{depth_to}_"
            self.logger.debug(f"Looking for files matching pattern: {pattern}*")

            for path_type, base_path in base_paths:
                self.logger.debug(f"\nChecking {path_type}: {base_path}")

                if not base_path:
                    self.logger.debug(f"  - Path is None (not configured)")
                    continue

                if not base_path.exists():
                    self.logger.debug(f"  - Path does not exist")
                    continue

                hole_path = base_path / project_code / hole_id
                self.logger.debug(f"  - Full hole path: {hole_path}")

                if not hole_path.exists():
                    self.logger.debug(f"  - Hole directory does not exist")
                    continue

                # List all files in the directory
                try:
                    files = list(os.listdir(str(hole_path)))
                    self.logger.debug(f"  - Found {len(files)} files in directory")

                    # Look for any file matching the pattern
                    matching_files = []
                    for filename in files:
                        if filename.startswith(pattern):
                            matching_files.append(filename)

                    if matching_files:
                        self.logger.debug(f"  - Found matching files: {matching_files}")
                        return True
                    else:
                        self.logger.debug(f"  - No files matching pattern '{pattern}*'")
                        if files:
                            self.logger.debug(
                                f"  - Sample files in directory: {files[:5]}"
                            )

                except Exception as e:
                    self.logger.debug(f"  - Error listing directory: {e}")

            self.logger.debug(f"File not found in any location")
            return False

        except Exception as e:
            self.logger.error(f"Error checking original file existence: {e}")
            return False

    def log_path_configuration(self) -> None:
        """Log all configured paths for debugging."""
        self.logger.info("\n=== PATH CONFIGURATION ===")

        for cache_key, path in self._cached_paths.items():
            if path:
                exists = path.exists() if path else False
                self.logger.info(f"{cache_key}: {path} (exists: {exists})")
            else:
                self.logger.info(f"{cache_key}: Not configured")

        self.logger.info("=========================\n")

    def _calculate_average_hex_color(self, image_path: str) -> Optional[str]:
        """Calculate the average hex color of an image."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None

            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Calculate mean color
            mean_color = img_rgb.mean(axis=(0, 1))
            r, g, b = [int(c) for c in mean_color]

            # Convert to hex
            hex_color = f"#{r:02x}{g:02x}{b:02x}".upper()

            return hex_color

        except Exception as e:
            self.logger.error(f"Error calculating hex color for {image_path}: {e}")
            return None
