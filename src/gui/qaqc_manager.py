"""
Consolidated QAQC (Quality Assurance/Quality Control) system for compartment image review.
Combines all QAQC functionality into a single organized module.
"""

import os
import re
import time
import shutil
import threading
import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Set, Any
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk


from gui.dialog_helper import DialogHelper
from gui.progress_dialog import ProgressDialog
from gui.widgets.multiselect_review_dialog import MultiSelectReviewDialog

# ML Pipeline integration - optional
try:
    from ml_pipeline.predictor import get_predictor, ensure_model_loaded
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class ReviewItem:
    """Represents a compartment image to be reviewed."""

    filename: str
    hole_id: str
    depth_from: int
    depth_to: int
    compartment_depth: int  # The "To" value used in naming
    image_path: str
    duplicate_count: int = 0
    duplicate_paths: List[str] = field(default_factory=list)
    all_files_for_interval: Dict[str, List[str]] = field(default_factory=dict)
    moisture: str = "unknown"  # wet/dry/unknown
    quality: str = "unreviewed"  # OK/Blurry/Damaged/Missing/unreviewed
    average_hex_color: str = ""
    is_reviewed: bool = False
    register_status: Optional[str] = None
    _image: Optional[np.ndarray] = None
    is_conflict_resolution: bool = False
    conflicting_file_path: Optional[str] = None
    chosen_action: Optional[str] = None
    is_placeholder: bool = False
    source_uid: Optional[str] = None
    bad_image: bool = False  # Flag for poor quality images
    file_type: str = (
        "original"  # "original", "temp", "new", "pending", "review", "approved", "upload_failed"
    )
    # ML prediction fields
    ml_predicted: bool = False  # True if moisture was set by ML model
    ml_confidence: float = 0.0  # Confidence score from ML model (0.0 to 1.0)

    @property
    def image(self) -> Optional[np.ndarray]:
        """Lazy load image only when accessed."""
        if self._image is None and self.image_path and not self.is_placeholder:
            if os.path.exists(self.image_path):
                self._image = cv2.imread(self.image_path)
        return self._image

    def unload_image(self):
        """Free memory by unloading the image."""
        self._image = None


class QAQCConstants:
    """Constants used throughout the QAQC system."""

    STATUS_OK = "OK"
    STATUS_DAMAGED = "Damaged"
    STATUS_EMPTY = "Empty"
    STATUS_WET = "Wet"
    STATUS_DRY = "Dry"

    VALID_QAQC_STATUSES = ["OK_Wet", "OK_Dry", "Damaged_Wet", "Damaged_Dry", "Empty"]

    NEEDS_REVIEW_STATUSES = ["Found", None, "", "Not Set"]

    SKIP_STATUSES = ["MISSING_FILE", "Not Found", "MISSING"]


# ============================================================================
# SCANNER COMPONENT
# ============================================================================


class QAQCScanner:
    """Handles scanning and discovery of compartment files for QAQC."""

    def __init__(self, file_manager, register_manager, compartment_interval, logger):
        self.file_manager = file_manager
        self.register_manager = register_manager
        self.compartment_interval = compartment_interval
        self.logger = logger

        # Regex patterns
        self.HOLE_ID_PATTERN = re.compile(r"^[A-Z]{2}\d{4}$")
        self.PROJECT_CODE_PATTERN = re.compile(r"^[A-Z]{2}$")
        self.COMPARTMENT_FILE_PATTERN = re.compile(
            r"([A-Z]{2}\d{4})_CC_(\d+)(?:_temp|_new|_review|_pending_[a-f0-9]+|_Wet|_Dry)?\.(?:png|tiff|jpg)$",
            re.IGNORECASE,
        )

        self.compartment_register = {}

        # Cache for compartment scans to avoid re-scanning
        self._compartment_scan_cache = None
        self._cache_timestamp = None

    def _get_cached_compartments(
        self, hole_id: str, expected_depths: List[int]
    ) -> Dict[int, List[str]]:
        """
        Get compartments using cached scan results when possible.
        Cache is invalidated after 5 minutes or can be manually cleared.
        """

        # Check if we need to refresh the cache (5-minute timeout)
        current_time = time.time()
        cache_expired = (
            self._cache_timestamp is None
            or (current_time - self._cache_timestamp) > 300
        )

        if cache_expired or self._compartment_scan_cache is None:
            self.logger.info(
                "Performing comprehensive compartment scan (will cache results)..."
            )

            # Perform a full scan of all directories once
            self._compartment_scan_cache = {}

            # Get all compartment locations to scan
            directories_to_scan = []

            # Add local directories
            if self.file_manager:
                local_review = self.file_manager.dir_structure.get(
                    "review_compartments"
                )
                if local_review and os.path.exists(local_review):
                    directories_to_scan.append(("local_review", local_review))

                local_approved = self.file_manager.dir_structure.get(
                    "approved_compartments"
                )
                if local_approved and os.path.exists(local_approved):
                    directories_to_scan.append(("local_approved", local_approved))

                # Add shared directories
                shared_review = self.file_manager.get_shared_path(
                    "review_compartments", create_if_missing=False
                )
                if shared_review and os.path.exists(shared_review):
                    directories_to_scan.append(("shared_review", shared_review))

                shared_approved = self.file_manager.get_shared_path(
                    "approved_compartments", create_if_missing=False
                )
                if shared_approved and os.path.exists(shared_approved):
                    directories_to_scan.append(("shared_approved", shared_approved))

            # Scan all directories once and cache by hole_id
            # Pattern excludes _UPLOADED files but includes _UPLOAD_FAILED
            pattern = re.compile(
                r"([A-Z]{2}\d{4})_CC_(\d+)(.*?)(?<!_UPLOADED)\.(?:png|tiff|jpg)$",
                re.IGNORECASE,
            )

            for location_name, directory in directories_to_scan:
                self.logger.debug(f"Scanning {location_name}: {directory}")

                # Walk through directory structure
                for root, dirs, files in os.walk(directory):
                    for filename in files:
                        match = pattern.match(filename)
                        if match:
                            file_hole_id = match.group(1)
                            depth = int(match.group(2))
                            filepath = os.path.join(root, filename)

                            # Store in cache by hole_id and depth
                            if file_hole_id not in self._compartment_scan_cache:
                                self._compartment_scan_cache[file_hole_id] = {}
                            if depth not in self._compartment_scan_cache[file_hole_id]:
                                self._compartment_scan_cache[file_hole_id][depth] = []

                            self._compartment_scan_cache[file_hole_id][depth].append(
                                filepath
                            )

            self._cache_timestamp = current_time
            self.logger.info(
                f"Scan complete. Cached data for {len(self._compartment_scan_cache)} holes"
            )
        else:
            self.logger.debug(
                f"Using cached scan results (age: {current_time - self._cache_timestamp:.1f}s)"
            )

        # Extract results for this specific hole from cache
        existing_compartments = {}
        if hole_id in self._compartment_scan_cache:
            hole_data = self._compartment_scan_cache[hole_id]
            for depth in expected_depths:
                if depth in hole_data:
                    existing_compartments[depth] = hole_data[depth]

        return existing_compartments

    def clear_scan_cache(self):
        """Clear the compartment scan cache to force a fresh scan."""
        self._compartment_scan_cache = None
        self._cache_timestamp = None
        self.logger.info("Compartment scan cache cleared")

    def load_register_into_memory(self) -> Dict[str, pd.DataFrame]:
        """Load compartment register efficiently."""
        df = self.register_manager.get_compartment_data()

        if df.empty:
            return {}

        relevant_columns = ["HoleID", "From", "To", "Photo_Status"]
        if all(col in df.columns for col in relevant_columns):
            df = df[relevant_columns]
        else:
            self.logger.warning("Register missing expected columns")
            return {}

        self.compartment_register = {
            hole_id: group for hole_id, group in df.groupby("HoleID")
        }
        return self.compartment_register

    def analyze_compartment_files(self, hole_id: str) -> Dict[str, Any]:
        """
        Comprehensively analyze all compartment files for a hole across all locations.
        Uses the existing compartment scan cache to avoid re-scanning.

        Returns a detailed analysis including duplicates, misplaced files, and review status.
        """
        analysis = {
            "hole_id": hole_id,
            "compartments_by_depth": {},  # depth -> list of file info
            "duplicates": {},  # depth -> list of duplicate files
            "misplaced_files": {
                "reviewed_in_review_folder": [],  # Files with _Wet/_Dry in review folder
                "unreviewed_in_approved_folder": [],  # Files without _Wet/_Dry in approved folder
            },
            "needs_review": [],  # Depths that need review
            "register_status": {},  # depth -> current register status
            "recommendations": [],
        }

        # Get expected depths based on original images in register
        expected_depths = self._get_expected_depths_from_register(hole_id)

        # PERFORMANCE FIX: Use the compartment scan cache directly instead of
        # calling duplicate_handler which triggers a full directory re-scan
        existing_compartments = {}
        if self._compartment_scan_cache and hole_id in self._compartment_scan_cache:
            # Extract from cache
            hole_data = self._compartment_scan_cache[hole_id]
            for depth in expected_depths:
                if depth in hole_data:
                    existing_compartments[depth] = hole_data[depth]
            self.logger.debug(
                f"Using compartment scan cache for analysis ({len(existing_compartments)} depths)"
            )
        else:
            # Fallback: trigger a cache refresh if needed
            self.logger.warning(
                f"Cache miss in analyze_compartment_files for {hole_id}, triggering scan"
            )
            # Don't call duplicate_handler - just use our own cache method
            existing_compartments = self._get_cached_compartments(
                hole_id, expected_depths
            )

        # Also explicitly scan approved folder to ensure all files are found
        if self.file_manager:
            approved_path = self.file_manager.get_shared_path(
                "approved_compartments", create_if_missing=False
            )
            if approved_path:
                from pathlib import Path

                approved_path = (
                    Path(approved_path)
                    if not isinstance(approved_path, Path)
                    else approved_path
                )
                project_code = hole_id[:2].upper() if len(hole_id) >= 2 else ""
                hole_approved_path = approved_path / project_code / hole_id
                if hole_approved_path.exists():
                    for file in hole_approved_path.iterdir():
                        if file.is_file():
                            match = self.COMPARTMENT_FILE_PATTERN.match(file.name)
                            if match:
                                depth = int(match.group(2))
                                if depth in expected_depths:
                                    filepath = str(file)
                                    if depth not in existing_compartments:
                                        existing_compartments[depth] = []
                                    # Only add if not already in list
                                    if filepath not in existing_compartments[depth]:
                                        existing_compartments[depth].append(filepath)
                                        self.logger.debug(
                                            f"Added approved file: {file.name}"
                                        )

        # Sort file paths for consistent ordering
        for depth in existing_compartments:
            existing_compartments[depth] = sorted(existing_compartments[depth])

        # Analyze each found compartment
        for depth, file_paths in existing_compartments.items():
            analysis["compartments_by_depth"][depth] = []

            for file_path in file_paths:
                file_info = self._analyze_compartment_file(file_path, hole_id, depth)
                analysis["compartments_by_depth"][depth].append(file_info)

                # Check for misplaced files
                if file_info["is_reviewed"] and file_info["location_type"] == "review":
                    analysis["misplaced_files"]["reviewed_in_review_folder"].append(
                        file_info
                    )
                elif (
                    not file_info["is_reviewed"]
                    and file_info["location_type"] == "approved"
                ):
                    analysis["misplaced_files"]["unreviewed_in_approved_folder"].append(
                        file_info
                    )

            # Check for duplicates at this depth
            if len(file_paths) > 1:
                analysis["duplicates"][depth] = file_paths

            # Get register status
            analysis["register_status"][depth] = self.get_register_status(
                hole_id, depth
            )

            # Determine if needs review
            if self._needs_review_check(
                analysis["compartments_by_depth"][depth],
                analysis["register_status"][depth],
            ):
                analysis["needs_review"].append(depth)

        # Add recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)

        return analysis

    def scan_local_review_folder(
        self, temp_review_path: str
    ) -> Dict[str, List[ReviewItem]]:
        """Scan local temp_review folder for items needing review."""
        review_items_by_hole = {}

        if not os.path.exists(temp_review_path):
            return review_items_by_hole

        # Find all compartment files
        for root, dirs, files in os.walk(temp_review_path):
            for file in files:
                match = self.COMPARTMENT_FILE_PATTERN.match(file)
                if match:
                    hole_id = match.group(1)
                    depth = int(match.group(2))
                    full_path = os.path.join(root, file)

                    # ===================================================
                    # INSERT: Pre-load moisture status from filename
                    moisture = "unknown"
                    if "_Wet" in file:
                        moisture = "Wet"
                    elif "_Dry" in file:
                        moisture = "Dry"
                    # ===================================================

                    # Get register status
                    register_status = self.get_register_status(hole_id, depth)

                    # Pre-load moisture from register if available and not in filename
                    if moisture == "unknown" and register_status:
                        if "Wet" in register_status:
                            moisture = "Wet"
                        elif "Dry" in register_status:
                            moisture = "Dry"

                    # Extract UID from image if available
                    source_uid = None
                    if hasattr(self, "file_manager") and self.file_manager:
                        source_uid = self.file_manager.extract_uid_from_any_image(
                            full_path
                        )

                    # Create review item
                    review_item = ReviewItem(
                        filename=file,
                        hole_id=hole_id,
                        depth_from=depth - self.compartment_interval,
                        depth_to=depth,
                        compartment_depth=depth,
                        image_path=full_path,
                        register_status=register_status,
                        moisture=moisture,
                        quality="OK" if moisture != "unknown" else "unreviewed",
                        source_uid=source_uid,
                    )

                    # Mark if this is a temp or new file
                    if "_temp" in file.lower():
                        review_item.file_type = "temp"
                    elif "_new" in file.lower():
                        review_item.file_type = "new"
                    elif "_pending_" in file.lower():
                        review_item.file_type = "pending"
                    else:
                        review_item.file_type = "original"

                    # Add to hole's list
                    if hole_id not in review_items_by_hole:
                        review_items_by_hole[hole_id] = []
                    review_items_by_hole[hole_id].append(review_item)

        return review_items_by_hole

    def scan_local_approved_duplicates(self) -> Dict[str, List[ReviewItem]]:
        """
        Scan local approved_compartments for files that also exist in shared approved.
        These duplicates need QAQC review to decide which to keep.
        
        Returns:
            Dict mapping hole_id -> list of ReviewItems needing review
        """
        review_items_by_hole = {}
        
        if not self.file_manager:
            return review_items_by_hole
        
        local_approved = self.file_manager.dir_structure.get("approved_compartments")
        shared_approved = self.file_manager.get_shared_path("approved_compartments", create_if_missing=False)
        
        if not local_approved or not shared_approved:
            self.logger.debug("Local or shared approved paths not configured")
            return review_items_by_hole
        
        if not os.path.exists(local_approved) or not os.path.exists(shared_approved):
            return review_items_by_hole
        
        self.logger.info("Scanning local approved for duplicates with shared...")
        
        local_path = Path(local_approved)
        shared_path = Path(shared_approved)
        
        # Pattern for compartment files (exclude _UPLOADED)
        pattern = re.compile(
            r"([A-Z]{2}\d{4})_CC_(\d+)_(Wet|Dry)\.(?:png|jpg|tiff)$",
            re.IGNORECASE
        )
        
        duplicate_count = 0
        
        # Walk through local approved
        for project_dir in os.listdir(local_path):
            project_local = local_path / project_dir
            project_shared = shared_path / project_dir
            
            if not project_local.is_dir():
                continue
                
            for hole_dir in os.listdir(project_local):
                hole_local = project_local / hole_dir
                hole_shared = project_shared / hole_dir
                
                if not hole_local.is_dir():
                    continue
                
                for filename in os.listdir(hole_local):
                    match = pattern.match(filename)
                    if not match:
                        continue
                    
                    local_file = hole_local / filename
                    shared_file = hole_shared / filename
                    
                    # Check if same file exists in shared
                    if shared_file.exists():
                        hole_id = match.group(1)
                        depth = int(match.group(2))
                        moisture = match.group(3)
                        
                        # Get register status
                        register_status = self.get_register_status(hole_id, depth)
                        
                        # Extract UID
                        source_uid = None
                        if self.file_manager:
                            source_uid = self.file_manager.extract_uid_from_any_image(str(local_file))
                        
                        # Create review item for the LOCAL file (it's the duplicate)
                        review_item = ReviewItem(
                            filename=filename,
                            hole_id=hole_id,
                            depth_from=depth - self.compartment_interval,
                            depth_to=depth,
                            compartment_depth=depth,
                            image_path=str(local_file),
                            register_status=register_status,
                            moisture=moisture,
                            quality="OK",
                            source_uid=source_uid,
                        )
                        review_item.file_type = "local_approved_duplicate"
                        review_item.is_reviewed = True  # Already has Wet/Dry
                        
                        # Store shared file info for comparison
                        review_item.shared_path = str(shared_file)
                        review_item.local_size = local_file.stat().st_size
                        review_item.shared_size = shared_file.stat().st_size
                        
                        if hole_id not in review_items_by_hole:
                            review_items_by_hole[hole_id] = []
                        review_items_by_hole[hole_id].append(review_item)
                        duplicate_count += 1
        
        self.logger.info(f"Found {duplicate_count} local/shared duplicates across {len(review_items_by_hole)} holes")
        return review_items_by_hole

    def scan_all_compartments_for_hole(self, hole_id: str) -> List[ReviewItem]:
        """Scan ALL compartment files for a hole including approved ones for comparison."""
        all_items = []
        seen_paths = set()

        # Check if we have cached results first
        current_time = time.time()
        cache_expired = (
            self._cache_timestamp is None
            or (current_time - self._cache_timestamp) > 300
        )

        # If cache is valid and contains this hole, use cached paths
        if (
            not cache_expired
            and self._compartment_scan_cache
            and hole_id in self._compartment_scan_cache
        ):
            self.logger.debug(
                f"Using cached scan results for {hole_id} (age: {current_time - self._cache_timestamp:.1f}s)"
            )
            cached_data = self._compartment_scan_cache[hole_id]

            # Build ReviewItems from cached paths
            for depth, file_paths in cached_data.items():
                for filepath in file_paths:
                    if filepath in seen_paths:
                        continue

                    filename = os.path.basename(filepath)

                    # Determine moisture status from filename
                    moisture = "unknown"
                    quality = "unreviewed"
                    is_reviewed = False

                    if "_Wet" in filename and "_UPLOAD" not in filename:
                        moisture = "Wet"
                        quality = "OK"
                        is_reviewed = True
                    elif "_Dry" in filename and "_UPLOAD" not in filename:
                        moisture = "Dry"
                        quality = "OK"
                        is_reviewed = True
                    elif "_UPLOAD_FAILED" in filename:
                        if "_Wet_UPLOAD_FAILED" in filename:
                            moisture = "Wet"
                            quality = "OK"
                            is_reviewed = True
                        elif "_Dry_UPLOAD_FAILED" in filename:
                            moisture = "Dry"
                            quality = "OK"
                            is_reviewed = True

                    # Determine file type
                    file_type = "original"
                    if "_temp" in filename.lower():
                        file_type = "temp"
                    elif "_new" in filename.lower():
                        file_type = "new"
                    elif "_pending_" in filename.lower():
                        file_type = "pending"
                    elif "_review" in filename.lower():
                        file_type = "review"
                    elif "_UPLOAD_FAILED" in filename:
                        file_type = "upload_failed"
                    elif "_Wet" in filename or "_Dry" in filename:
                        file_type = "approved"
                    elif "Approved Compartment Images" in filepath:
                        file_type = "in_approved_unclassified"

                    # Get register status
                    register_status = self.get_register_status(hole_id, depth)

                    # Extract UID if available
                    source_uid = None
                    if self.file_manager:
                        source_uid = self.file_manager.extract_uid_from_any_image(
                            filepath
                        )

                    review_item = ReviewItem(
                        filename=filename,
                        hole_id=hole_id,
                        depth_from=depth - self.compartment_interval,
                        depth_to=depth,
                        compartment_depth=depth,
                        image_path=filepath,
                        register_status=register_status,
                        moisture=moisture,
                        quality=quality,
                        is_reviewed=is_reviewed,
                        source_uid=source_uid,
                        file_type=file_type,
                    )

                    all_items.append(review_item)
                    seen_paths.add(filepath)

            # Sort results
            all_items.sort(
                key=lambda x: (
                    x.depth_to,
                    0 if x.file_type == "approved" else 1,
                    x.filename,
                )
            )

            self.logger.info(
                f"Found {len(all_items)} total compartments for {hole_id} (from cache): "
                f"{len([i for i in all_items if i.file_type == 'approved'])} approved, "
                f"{len([i for i in all_items if 'new' in i.file_type])} new, "
                f"{len([i for i in all_items if 'temp' in i.file_type])} temp, "
                f"{len([i for i in all_items if i.file_type == 'upload_failed'])} upload_failed"
            )

            # Apply ML predictions to unclassified items (from cache)
            self._apply_ml_predictions(all_items)

            return all_items

        # Cache miss or expired - perform full scan
        self.logger.info(f"Cache miss for {hole_id}, performing full directory scan...")

        # Comprehensive pattern to catch all compartment files EXCEPT _UPLOADED
        pattern = re.compile(
            rf"^{re.escape(hole_id)}_CC_(\d{{3}})(.*?)(?<!_UPLOADED)\.(?:png|tiff|jpg)$",
            re.IGNORECASE,
        )

        locations_to_scan = []

        if self.file_manager:
            project_code = hole_id[:2].upper() if len(hole_id) >= 2 else ""

            # Local temp review
            local_temp = self.file_manager.get_hole_dir("temp_review", hole_id)
            if local_temp and os.path.exists(local_temp):
                locations_to_scan.append(("local_review", local_temp, "review"))

            # Local approved
            local_approved = self.file_manager.get_hole_dir(
                "approved_compartments", hole_id
            )
            if local_approved and os.path.exists(local_approved):
                locations_to_scan.append(("local_approved", local_approved, "approved"))

            # Shared review
            shared_review = self.file_manager.get_shared_path(
                "review_compartments", create_if_missing=False
            )
            if shared_review:
                shared_review_hole = os.path.join(shared_review, project_code, hole_id)
                if os.path.exists(shared_review_hole):
                    locations_to_scan.append(
                        ("shared_review", shared_review_hole, "review")
                    )

            # Shared approved - IMPORTANT for seeing existing classifications
            shared_approved = self.file_manager.get_shared_path(
                "approved_compartments", create_if_missing=False
            )
            if shared_approved:
                shared_approved_hole = os.path.join(
                    shared_approved, project_code, hole_id
                )
                if os.path.exists(shared_approved_hole):
                    locations_to_scan.append(
                        ("shared_approved", shared_approved_hole, "approved")
                    )

        # Scan each location
        for location_name, location_path, location_type in locations_to_scan:
            self.logger.debug(f"Scanning {location_name}: {location_path}")

            try:
                for filename in os.listdir(location_path):
                    # Skip _UPLOADED files explicitly
                    if "_UPLOADED" in filename and not "_UPLOAD_FAILED" in filename:
                        self.logger.debug(f"Skipping backup file: {filename}")
                        continue

                    filepath = os.path.join(location_path, filename)

                    # Skip if already processed
                    if filepath in seen_paths:
                        continue

                    # Skip directories
                    if os.path.isdir(filepath):
                        continue

                    match = pattern.match(filename)
                    if match:
                        depth = int(match.group(1))
                        suffix = match.group(2) or ""

                        # Determine moisture status from filename
                        moisture = "unknown"
                        quality = "unreviewed"
                        is_reviewed = False

                        if "_Wet" in suffix and "_UPLOAD" not in suffix:
                            moisture = "Wet"
                            quality = "OK"
                            is_reviewed = True
                        elif "_Dry" in suffix and "_UPLOAD" not in suffix:
                            moisture = "Dry"
                            quality = "OK"
                            is_reviewed = True
                        elif "_UPLOAD_FAILED" in suffix:
                            # Handle upload failed files - check for wet/dry before _UPLOAD_FAILED
                            if "_Wet_UPLOAD_FAILED" in suffix:
                                moisture = "Wet"
                                quality = "OK"
                                is_reviewed = True
                            elif "_Dry_UPLOAD_FAILED" in suffix:
                                moisture = "Dry"
                                quality = "OK"
                                is_reviewed = True

                        # Get register status
                        register_status = self.get_register_status(hole_id, depth)

                        # Extract UID if available
                        source_uid = None
                        if self.file_manager:
                            source_uid = self.file_manager.extract_uid_from_any_image(
                                filepath
                            )

                        # Create review item
                        review_item = ReviewItem(
                            filename=filename,
                            hole_id=hole_id,
                            depth_from=depth - self.compartment_interval,
                            depth_to=depth,
                            compartment_depth=depth,
                            image_path=filepath,
                            register_status=register_status,
                            moisture=moisture,
                            quality=quality,
                            is_reviewed=is_reviewed,
                            source_uid=source_uid,
                        )

                        # Set file type based on suffix and location
                        if "_temp" in suffix.lower():
                            review_item.file_type = "temp"
                        elif "_new" in suffix.lower():
                            review_item.file_type = "new"
                        elif "_pending_" in suffix.lower():
                            review_item.file_type = "pending"
                        elif "_review" in suffix.lower():
                            review_item.file_type = "review"
                        elif "_UPLOAD_FAILED" in suffix:
                            review_item.file_type = "upload_failed"
                        elif "_Wet" in suffix or "_Dry" in suffix:
                            review_item.file_type = "approved"  # Already approved
                        elif location_type == "approved":
                            review_item.file_type = "in_approved_unclassified"
                        elif location_type == "review":
                            review_item.file_type = "unclassified"
                        else:
                            review_item.file_type = "unknown"

                        all_items.append(review_item)
                        seen_paths.add(filepath)

                        self.logger.debug(
                            f"Added {filename}: depth={depth}, moisture={moisture}, "
                            f"type={review_item.file_type}, location={location_name}"
                        )

            except Exception as e:
                self.logger.error(f"Error scanning {location_name}: {e}")

        # Sort by depth then by file type (approved items first for reference)
        all_items.sort(
            key=lambda x: (
                x.depth_to,
                0 if x.file_type == "approved" else 1,
                x.filename,
            )
        )

        self.logger.info(
            f"Found {len(all_items)} total compartments for {hole_id}: "
            f"{len([i for i in all_items if i.file_type == 'approved'])} approved, "
            f"{len([i for i in all_items if 'new' in i.file_type])} new, "
            f"{len([i for i in all_items if 'temp' in i.file_type])} temp, "
            f"{len([i for i in all_items if i.file_type == 'upload_failed'])} upload_failed"
        )

        # Apply ML predictions to unclassified items
        self._apply_ml_predictions(all_items)

        return all_items

    def _apply_ml_predictions(self, items: List[ReviewItem]) -> None:
        """
        Apply ML wet/dry predictions to unclassified items.

        Modifies items in-place, setting moisture, ml_predicted, and ml_confidence
        for items that are currently unclassified.
        """
        if not ML_AVAILABLE:
            return

        # Get unclassified items
        unclassified = [
            item for item in items
            if item.moisture == "unknown"
            and item.image_path
            and os.path.exists(item.image_path)
        ]

        if not unclassified:
            return

        try:
            # Ensure model is loaded
            if not ensure_model_loaded():
                self.logger.warning("ML model not available for predictions")
                return

            predictor = get_predictor()

            self.logger.info(f"Running ML predictions on {len(unclassified)} unclassified items...")

            image_paths = [item.image_path for item in unclassified]
            batch_results = predictor.predict_batch(image_paths)

            for item, (label, confidence) in zip(unclassified, batch_results):
                if label in ("Wet", "Dry"):
                    item.moisture = label
                    item.ml_predicted = True
                    item.ml_confidence = confidence
                    item.quality = "OK"

            # Log summary
            ml_predicted = [i for i in items if i.ml_predicted]
            wet_count = sum(1 for i in ml_predicted if i.moisture == "Wet")
            dry_count = sum(1 for i in ml_predicted if i.moisture == "Dry")
            high_conf = sum(1 for i in ml_predicted if i.ml_confidence >= 0.90)

            self.logger.info(
                f"ML Predictions applied: {wet_count} Wet, {dry_count} Dry "
                f"({high_conf} high confidence, {len(ml_predicted) - high_conf} need review)"
            )

        except Exception as e:
            self.logger.error(f"Error applying ML predictions: {e}")

    def scan_shared_folders_for_review(self) -> Dict[str, List[str]]:
        """Scan shared folders for compartments needing review."""
        items_by_hole = {}

        # Check review folder
        review_path = self.file_manager.get_shared_path(
            "review_compartments", create_if_missing=False
        )
        self.logger.info(f"Checking shared review path: {review_path}")
        if review_path and os.path.exists(review_path):
            self.logger.info(f"Scanning review folder: {review_path}")
            self._scan_shared_folder(review_path, items_by_hole, "review")
            self.logger.info(f"Items after review scan: {len(items_by_hole)}")
        else:
            self.logger.info("Review path doesn't exist or not configured")

        # Check approved folder
        approved_path = self.file_manager.get_shared_path(
            "approved_compartments", create_if_missing=False
        )
        self.logger.info(f"Checking shared approved path: {approved_path}")
        if approved_path and os.path.exists(approved_path):
            self.logger.info(f"Scanning approved folder: {approved_path}")
            initial_count = len(items_by_hole)
            self._scan_shared_folder(approved_path, items_by_hole, "approved")
            self.logger.info(
                f"Found {len(items_by_hole) - initial_count} additional items in approved"
            )
        else:
            self.logger.info("Approved path doesn't exist or not configured")

        self.logger.info(f"Total items found: {len(items_by_hole)}")
        return items_by_hole

    def _scan_shared_folder(
        self, base_path: str, items_by_hole: Dict[str, List[str]], folder_type: str
    ):
        """Scan a shared folder for compartments needing review."""
        for project_code in os.listdir(base_path):
            if not self.PROJECT_CODE_PATTERN.match(project_code):
                continue

            project_path = os.path.join(base_path, project_code)
            if not os.path.isdir(project_path):
                continue

            for hole_id in os.listdir(project_path):
                if not self.HOLE_ID_PATTERN.match(hole_id):
                    continue

                hole_path = os.path.join(project_path, hole_id)
                if not os.path.isdir(hole_path):
                    continue

                for file in os.listdir(hole_path):
                    match = self.COMPARTMENT_FILE_PATTERN.match(file)
                    if match:
                        depth = int(match.group(2))
                        register_status = self.get_register_status(hole_id, depth)

                        if (
                            folder_type == "review"
                            or register_status in QAQCConstants.NEEDS_REVIEW_STATUSES
                        ):
                            full_path = os.path.join(hole_path, file)
                            if hole_id not in items_by_hole:
                                items_by_hole[hole_id] = []
                            items_by_hole[hole_id].append(full_path)

    def get_register_status(self, hole_id: str, depth_to: int) -> Optional[str]:
        """Get the current Photo_Status from register for a compartment."""
        if hole_id in self.compartment_register:
            df = self.compartment_register[hole_id]
            matching = df[df["To"] == depth_to]
            if not matching.empty:
                return matching.iloc[0].get("Photo_Status")
        return None

    def scan_specific_files(self, file_paths: List[str]) -> List[ReviewItem]:
        """Create ReviewItems from specific file paths."""
        review_items = []

        for file_path in file_paths:
            if not os.path.exists(file_path):
                continue

            filename = os.path.basename(file_path)
            match = self.COMPARTMENT_FILE_PATTERN.match(filename)

            if match:
                hole_id = match.group(1)
                depth = int(match.group(2))

                review_item = ReviewItem(
                    filename=filename,
                    hole_id=hole_id,
                    depth_from=depth - self.compartment_interval,
                    depth_to=depth,
                    compartment_depth=depth,
                    image_path=file_path,
                    register_status=self.get_register_status(hole_id, depth),
                )

                review_items.append(review_item)

        return review_items

    def _get_expected_depths_from_register(self, hole_id: str) -> List[int]:
        """Get expected compartment depths from original image register."""
        if not self.register_manager:
            return []

        # Get original images for this hole
        orig_df = self.register_manager.get_original_image_data(hole_id)
        if orig_df.empty:
            return []

        expected_depths = []
        for _, row in orig_df.iterrows():
            depth_from = int(row["Depth_From"])
            depth_to = int(row["Depth_To"])

            # Calculate expected compartments
            current = depth_from + self.compartment_interval
            while current <= depth_to:
                expected_depths.append(current)
                current += self.compartment_interval

        return sorted(list(set(expected_depths)))

    def _analyze_compartment_file(
        self, file_path: str, hole_id: str, depth: int
    ) -> Dict[str, Any]:
        """Analyze a single compartment file."""
        filename = os.path.basename(file_path)

        # Determine location type using file manager paths
        location_type = "unknown"
        location_name = "Unknown"

        # Check against known paths
        if self.file_manager.get_shared_path("approved_compartments"):
            if (
                str(self.file_manager.get_shared_path("approved_compartments"))
                in file_path
            ):
                location_type = "approved"
                location_name = "Shared Approved"

        if self.file_manager.get_shared_path("review_compartments"):
            if (
                str(self.file_manager.get_shared_path("review_compartments"))
                in file_path
            ):
                location_type = "review"
                location_name = "Shared Review"

        if "temp_review" in file_path:
            location_type = "temp"
            location_name = "Local Temp"
        elif "Approved Compartment Images" in file_path:
            location_type = "approved"
            location_name = "Local Approved"

        # Check if reviewed based on filename
        is_reviewed = "_Wet" in filename or "_Dry" in filename
        moisture = None
        if "_Wet" in filename:
            moisture = "Wet"
        elif "_Dry" in filename:
            moisture = "Dry"

        return {
            "file_path": file_path,
            "filename": filename,
            "location_type": location_type,
            "location_name": location_name,
            "is_reviewed": is_reviewed,
            "moisture": moisture,
            "depth": depth,
            "file_exists": os.path.exists(file_path),
        }

    def _needs_review_check(self, file_infos: List[Dict], register_status: str) -> bool:
        """Check if compartment needs review based on files and register."""
        # PEP-8: keep logic simple and aligned with UI duplicate rules.
        # Only flag when there is at least one UNREVIEWED file at this depth.
        # Do NOT flag when only approved Wet/Dry exist (even if both are present).
        if not file_infos:
            return False
        has_unreviewed = any(not f.get("is_reviewed", False) for f in file_infos)
        return bool(has_unreviewed)

    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Check for duplicates
        if analysis["duplicates"]:
            recommendations.append(
                f"Found {len(analysis['duplicates'])} depths with duplicate files. "
                "Duplicate resolution will be required."
            )

        # Check for misplaced files
        if analysis["misplaced_files"]["reviewed_in_review_folder"]:
            count = len(analysis["misplaced_files"]["reviewed_in_review_folder"])
            recommendations.append(
                f"Found {count} reviewed files in review folder. "
                "These should be moved to approved folder."
            )

        if analysis["misplaced_files"]["unreviewed_in_approved_folder"]:
            count = len(analysis["misplaced_files"]["unreviewed_in_approved_folder"])
            recommendations.append(
                f"Found {count} unreviewed files in approved folder. "
                "These need review or should be moved to review folder."
            )

        # Check register sync
        out_of_sync = 0
        for depth, files in analysis["compartments_by_depth"].items():
            register_status = analysis["register_status"].get(depth)
            if files and not register_status:
                out_of_sync += 1
            elif register_status and not files:
                out_of_sync += 1

        if out_of_sync > 0:
            recommendations.append(
                f"Found {out_of_sync} compartments out of sync with register. "
                "Register synchronization recommended."
            )

        return recommendations


# ============================================================================
# PROCESSOR COMPONENT
# ============================================================================


class QAQCProcessor:
    """Handles batch processing, saving, and register updates for QAQC."""

    def __init__(self, file_manager, register_manager, config, logger):
        self.file_manager = file_manager
        self.register_manager = register_manager
        self.config = config
        self.logger = logger

        self.COMPARTMENT_FILE_PATTERN = re.compile(
            r"([A-Z]{2}\d{4})_CC_(\d+)(?:_temp|_new|_review|_pending_[a-f0-9]+|_Wet|_Dry)?(?<!_UPLOADED)\.(?:png|tiff|jpg)$",
            re.IGNORECASE,
        )

        self.stats = {
            "processed": 0,
            "uploaded": 0,
            "upload_failed": 0,
            "saved_locally": 0,
            "register_updated": 0,
            "register_failed": 0,
        }

    def reset_stats(self):
        """Reset processing statistics."""
        self.stats = {k: 0 for k in self.stats}

    def batch_process_reviewed_items(self, review_items: List[ReviewItem]):
        """Batch process all reviewed items for efficiency."""
        to_save = []
        to_update_register = []
        to_delete = []

        for item in review_items:
            if not item.is_reviewed:
                continue

            if item.quality in ["OK", "Damaged"]:
                to_save.append(item)
                to_update_register.append(item)
            elif item.quality == "Missing":
                to_delete.append(item.image_path)

        # Batch operations
        if to_save:
            self._batch_save_compartments(to_save)
        if to_update_register:
            self._batch_update_register(to_update_register)
        if to_delete:
            self._batch_delete_files(to_delete)

    def _batch_save_compartments(self, items: List[ReviewItem]):
        """Batch save reviewed compartments."""
        for item in items:
            try:
                # Check current state
                current_suffix = None
                is_pending = "_pending_" in item.filename.lower()
                if "_Wet" in item.filename:
                    current_suffix = "Wet"
                elif "_Dry" in item.filename:
                    current_suffix = "Dry"
                
                # Pending files from auto-loop always need processing
                if is_pending:
                    self.logger.info(f"Processing pending file from auto-loop: {item.filename}")

                # Check if this is an approved file being changed or deleted
                is_approved_file = (
                    hasattr(item, "file_type") and item.file_type == "approved"
                )
                is_being_deleted = item.quality == "Missing"
                is_being_changed = current_suffix and current_suffix != item.moisture

                # Handle deletion of approved files
                if is_approved_file and is_being_deleted:
                    try:
                        os.remove(item.image_path)
                        self.logger.info(f"Deleted approved file: {item.filename}")
                        self.stats["processed"] += 1
                        continue
                    except Exception as e:
                        self.logger.error(
                            f"Failed to delete approved file {item.filename}: {e}"
                        )
                        continue

                # Handle change of classification for approved files
                if is_approved_file and is_being_changed:
                    # Delete the old file
                    try:
                        os.remove(item.image_path)
                        self.logger.info(
                            f"Removed old approved file for reclassification: {item.filename}"
                        )
                    except Exception as e:
                        self.logger.warning(f"Could not remove old file: {e}")

                    # Continue to save with new classification
                    self.logger.info(
                        f"Reclassifying from {current_suffix} to {item.moisture}"
                    )

                # Check if file is in temp_review folder or is a pending auto-loop file
                is_in_temp_review = (
                    "temp_review" in item.image_path
                    or "Compartment Images for Review" in item.image_path
                    or is_pending  # Pending files always need processing
                )

                # Skip ONLY if already has correct suffix AND is NOT in temp review AND not being changed
                if (
                    current_suffix == item.moisture
                    and not is_in_temp_review
                    and not is_being_changed
                ):
                    self.logger.info(
                        f"File already has correct suffix and is in correct location: {item.filename}"
                    )
                    self.stats["processed"] += 1
                    continue

                # If in temp_review, always process (move to approved and update register)
                if is_in_temp_review:
                    self.logger.info(
                        f"Processing file from temp_review: {item.filename}"
                    )

                # Continue with normal processing...
                # Determine final status
                if item.moisture in ["Wet", "Dry"]:
                    status = item.moisture
                else:
                    status = "unknown"

                # Save using FileManager
                result = self.file_manager.save_reviewed_compartment(
                    image=item.image,
                    hole_id=item.hole_id,
                    compartment_depth=item.compartment_depth,
                    status=status,
                    output_format=self.config.get("output_format", "png"),
                    source_uid=item.source_uid,
                )

                # Clean up source file if it was successfully saved and is in temp_review
                if result.get("local_path") and is_in_temp_review:
                    try:
                        if os.path.exists(item.image_path) and os.path.abspath(
                            item.image_path
                        ) != os.path.abspath(result["local_path"]):
                            os.remove(item.image_path)
                            self.logger.info(
                                f"Cleaned up temp source file: {item.image_path}"
                            )
                    except Exception as e:
                        self.logger.warning(
                            f"Could not clean up temp source file {item.image_path}: {e}"
                        )

                # Update stats
                if result.get("local_path"):
                    self.stats["saved_locally"] += 1
                    self.stats["processed"] += 1

                if result.get("upload_success"):
                    self.stats["uploaded"] += 1
                elif result.get("local_path") and not result.get("upload_success"):
                    self.stats["upload_failed"] += 1

            except Exception as e:
                self.logger.error(
                    f"Error saving compartment {item.hole_id}_{item.compartment_depth}: {e}"
                )

    def _batch_update_register(self, items: List[ReviewItem]):
        """Batch update register entries."""
        updates = []

        for item in items:
            # Determine photo status
            if item.quality == "OK" and item.moisture in ["Wet", "Dry"]:
                photo_status = f"OK_{item.moisture}"
            elif item.quality == "Damaged" and item.moisture in ["Wet", "Dry"]:
                photo_status = f"Damaged_{item.moisture}"
            else:
                photo_status = item.quality

            # Append "_Bad" suffix if image quality is poor
            if item.bad_image and photo_status not in ["Missing", "Empty"]:
                photo_status += "_Bad"

            update = {
                "hole_id": item.hole_id,
                "depth_from": int(item.depth_from),
                "depth_to": int(item.depth_to),
                "photo_status": photo_status,
                "comments": f"QAQC reviewed on {datetime.now().strftime('%Y-%m-%d')}",
                "qaqc_by": os.getenv("USERNAME", "Unknown"),
                "qaqc_date": datetime.now().strftime("%Y-%m-%d"),
            }

            if item.average_hex_color:
                update["average_hex_color"] = item.average_hex_color

            updates.append(update)

        if updates:
            try:
                updated = self.register_manager.batch_update_compartments(updates)
                self.stats["register_updated"] += updated
                self.stats["register_failed"] += len(updates) - updated
            except Exception as e:
                self.logger.error(f"Error batch updating register: {e}")
                self.stats["register_failed"] += len(updates)

    def _extract_pending_uid(self, filename: str) -> Optional[str]:
        """Extract UID from pending filename like 'KM0335_CC_021_pending_45ecbec5.png'."""
        import re
        match = re.search(r"_pending_([a-f0-9]+)\.", filename.lower())
        return match.group(1) if match else None

    def _batch_delete_files(self, file_paths: List[str]):
        """Batch delete files."""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    self.logger.info(f"Deleted file: {file_path}")
            except Exception as e:
                self.logger.error(f"Error deleting file {file_path}: {e}")

    # NOTE : Removed this method to ensure that all files are processed in the shared folder path at this stage
    # NOTE : users should sync the image files for smoother use.
    # def move_files_to_local(self, hole_id: str, file_paths: List[str]) -> List[str]:
    #     """Move files from shared location to local temp_review folder."""
    #     local_paths = []

    #     temp_review_path = self.file_manager.get_local_path("temp_review")
    #     project_code = hole_id[:2]
    #     local_hole_path = os.path.join(temp_review_path, project_code, hole_id)

    #     os.makedirs(local_hole_path, exist_ok=True)

    #     for src_path in file_paths:
    #         try:
    #             filename = os.path.basename(src_path)
    #             dst_path = os.path.join(local_hole_path, filename)

    #             shutil.copy2(src_path, dst_path)
    #             local_paths.append(dst_path)

    #             try:
    #                 os.remove(src_path)
    #             except Exception as e:
    #                 self.logger.warning(f"Could not remove source file {src_path}: {e}")

    #         except Exception as e:
    #             self.logger.error(f"Error moving file {src_path}: {e}")

    #     return local_paths

    def move_unreviewed_items_to_shared(
        self, unreviewed_items: List[ReviewItem], progress_callback=None
    ) -> tuple[int, int]:
        """Move unreviewed items to shared review folder."""
        moved_count = 0
        failed_count = 0

        shared_review_path = self.file_manager.get_shared_path(
            "review_compartments", create_if_missing=True
        )
        if not shared_review_path:
            self.logger.error("Shared review folder is not configured")
            return 0, len(unreviewed_items)

        for idx, item in enumerate(unreviewed_items):
            if progress_callback:
                progress_callback(idx + 1, f"Moving {item.filename}...")

            try:
                project_code = item.hole_id[:2].upper()
                shared_hole_path = (
                    Path(shared_review_path) / project_code / item.hole_id
                )
                shared_hole_path.mkdir(parents=True, exist_ok=True)

                base_name = f"{item.hole_id}_CC_{item.compartment_depth:03d}"
                if "_review" not in item.filename:
                    dest_filename = f"{base_name}_review.png"
                else:
                    dest_filename = item.filename

                dest_path = shared_hole_path / dest_filename

                if self.file_manager.copy_with_metadata(
                    item.image_path, str(dest_path)
                ):
                    if dest_path.exists():
                        try:
                            os.remove(item.image_path)
                            moved_count += 1
                            self.logger.info(
                                f"Moved unreviewed file to shared: {dest_filename}"
                            )
                        except Exception as e:
                            self.logger.warning(
                                f"Could not delete local file after copy: {e}"
                            )
                            moved_count += 1
                    else:
                        failed_count += 1
                        self.logger.error(
                            f"Copy verification failed for {item.filename}"
                        )
                else:
                    failed_count += 1
                    self.logger.error(
                        f"Failed to copy {item.filename} to shared folder"
                    )

            except Exception as e:
                failed_count += 1
                self.logger.error(f"Error moving {item.filename}: {e}")

        return moved_count, failed_count

    def check_and_move_remaining_temp_files(self, hole_id: str) -> int:
        """Check for any remaining files in temp_review folder and move them."""
        try:
            temp_review_path = Path(
                self.file_manager.get_hole_dir("temp_review", hole_id)
            )

            if not temp_review_path.exists():
                return 0

            remaining_files = []
            for file in temp_review_path.iterdir():
                if file.is_file() and self.COMPARTMENT_FILE_PATTERN.match(file.name):
                    remaining_files.append(file)

            if not remaining_files:
                return 0

            shared_review_path = self.file_manager.get_shared_path(
                "review_compartments", create_if_missing=True
            )
            if not shared_review_path:
                return 0

            project_code = hole_id[:2].upper()
            shared_hole_path = Path(shared_review_path) / project_code / hole_id
            shared_hole_path.mkdir(parents=True, exist_ok=True)

            moved_count = 0
            for file in remaining_files:
                try:
                    dest_path = shared_hole_path / file.name
                    if self.file_manager.copy_with_metadata(str(file), str(dest_path)):
                        file.unlink()
                        self.logger.info(f"Moved additional file: {file.name}")
                        moved_count += 1
                except Exception as e:
                    self.logger.error(
                        f"Failed to move additional file {file.name}: {e}"
                    )

            # Clean up empty directories
            try:
                if not any(temp_review_path.iterdir()):
                    temp_review_path.rmdir()

                    project_path = temp_review_path.parent
                    if not any(project_path.iterdir()):
                        project_path.rmdir()
            except Exception as e:
                self.logger.debug(f"Could not remove empty directories: {e}")

            return moved_count

        except Exception as e:
            self.logger.error(f"Error checking for remaining temp files: {e}")
            return 0

    def get_summary_message(self) -> str:
        """Get processing summary message."""
        summary_lines = [
            "QAQC Processing Complete:",
            f"- Compartments processed: {self.stats['processed']}",
            f"- Successfully uploaded to OneDrive: {self.stats['uploaded']}",
            f"- Failed OneDrive uploads: {self.stats['upload_failed']}",
            f"- Saved locally: {self.stats['saved_locally']}",
            f"- Register entries updated: {self.stats['register_updated']}",
            f"- Register update failures: {self.stats['register_failed']}",
        ]
        return "\n".join(summary_lines)

    def cleanup_processed_files(
        self, hole_id: str, reviewed_items: List[ReviewItem]
    ) -> int:
        """
        Clean up temporary files after successful processing.

        Args:
            hole_id: Hole ID
            reviewed_items: List of reviewed items

        Returns:
            Number of files cleaned up
        """
        temp_paths = []

        # Collect temp file paths from reviewed items
        for item in reviewed_items:
            if item.is_reviewed and item.image_path:
                # Only add if it's in temp_review folder
                if (
                    "Compartment Images for Review" in item.image_path
                    or "_temp" in item.image_path
                    or "_new" in item.image_path
                ):
                    temp_paths.append(item.image_path)

        # Use file manager to clean up
        if temp_paths:
            self.file_manager.cleanup_temp_compartments(hole_id, temp_paths)
            return len(temp_paths)

        return 0


# ============================================================================
# GRID CANVAS UI
# ============================================================================


class QAQCGridCanvas:
    """Grid-based canvas for reviewing compartment images."""

    def __init__(self, parent, gui_manager):
        self.parent = parent
        self.gui_manager = gui_manager

        self.logger = logging.getLogger(__name__)

        # Use theme colors from GUI manager
        if gui_manager:
            self.theme_colors = gui_manager.theme_colors
        else:
            # Fallback colors
            self.theme_colors = {
                "background": "#1e1e1e",
                "text": "#e0e0e0",
                "border": "#3f3f3f",
                "accent_blue": "#1e88e5",
                "accent_green": "#43a047",
                "accent_red": "#e53935",
                "accent_yellow": "#fdd835",
                "row_invalid": "#3a2222",
                "row_valid": "#223a22",
                "row_neutral": "#1e1e1e",
            }

        # Create main container
        self.container = ttk.Frame(parent)
        self.container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create canvas with scrollbars
        self.canvas = tk.Canvas(
            self.container, bg=self.theme_colors["background"], highlightthickness=0
        )
        self.v_scrollbar = ttk.Scrollbar(
            self.container, orient="vertical", command=self.canvas.yview
        )
        self.h_scrollbar = ttk.Scrollbar(
            self.container, orient="horizontal", command=self.canvas.xview
        )

        self.canvas.configure(
            yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set
        )

        # Grid layout for scrollbars
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        # Grid tracking structures
        self.cells = {}  # {(row, col): cell_data}
        self.cell_ids = {}  # {(row, col): canvas_item_ids}
        self.image_refs = {}  # Keep PhotoImage references
        self.depth_to_row = {}  # Map depth to row for easy lookup
        self.row_backgrounds = {}  # Track row background rectangles

        # Initialize all grid settings and caches
        # Grid settings - will be calculated from images
        self.base_cell_width = 300  # Default fallback
        self.base_cell_height = 200  # Default fallback
        self.scale_factor = 1.0
        self.cell_width = self.base_cell_width
        self.cell_height = self.base_cell_height
        self.padding = 5
        self.cols_per_row = 6
        self.target_cols = 6  # Target number of columns

        # Image dimensions cache - MUST be initialized here
        self.image_dimensions = {}  # {idx: (width, height)}
        self.optimal_cell_size = None

        # Current state
        self.hole_id = ""
        self.review_items = []
        self.current_mode = "select"
        self.selected_cells = set()
        self.item_states = {}  # idx: {"moisture": "wet/dry", "delete": bool}
        self.show_classified = True  # Flag to show/hide classified items
        self.scanner = None  # Reference to scanner for register status lookup

        self.layout_mode = "grid"  # "grid", "vertical", or "duplicate_resolution"

        # Add new attributes for layout and rotation
        self.image_rotation = 0  # Current rotation angle
        self.duplicates_resolved = False  # Track if duplicates have been resolved
        self.has_duplicates = False  # Track if there are duplicates
        self.duplicate_depths = {}  # Track which depths have duplicates

        # Selection handling
        self.drag_start = None
        self.selection_box = None

        # Add these to the existing __init__ method
        self.clicked_cell = None
        self.cells_in_drag = set()

        # Auto-scroll while dragging
        self.auto_scroll_timer = None
        self.auto_scroll_speed = 0  # Starts at 0, accelerates
        self.auto_scroll_direction = None  # 'up' or 'down'
        self.scroll_edge_threshold = 50  # pixels from edge to trigger scroll

        # Bind events
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<Button-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_release)
        self.canvas.bind("<Control-MouseWheel>", self._on_ctrl_mousewheel)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def load_items(
        self, review_items: List[ReviewItem], hole_id: str, reset_states: bool = False
    ):
        """
        Load review items with duplicate-first workflow.

        Args:
            review_items: List of items to display
            hole_id: The hole ID being displayed
            reset_states: If True, clear all classifications. If False, preserve existing states.
        """
        # Check if this is a new hole (different hole_id)
        is_new_hole = hole_id != self.hole_id

        self.hole_id = hole_id
        self.review_items = sorted(review_items, key=lambda x: x.depth_to)

        # Calculate optimal cell size before loading
        self._calculate_optimal_cell_size()

        # Reset item_states only if:
        # 1. Explicitly requested (reset_states=True), OR
        # 2. Loading a new hole (different hole_id), OR
        # 3. Number of items changed (can't preserve old states)
        should_reset = (
            reset_states
            or is_new_hole
            or len(self.item_states) != len(self.review_items)
        )

        if should_reset:
            self.logger.debug(
                f"Resetting item_states (reset_states={reset_states}, is_new_hole={is_new_hole})"
            )
            self.item_states = {}
            for i, item in enumerate(self.review_items):
                # Pre-load state from item's moisture status
                moisture = None
                if item.moisture in ["Wet", "Dry"]:
                    moisture = item.moisture.lower()

                self.item_states[i] = {
                    "moisture": moisture,
                    "delete": False,
                    "bad_image": False,
                    "original_moisture": moisture,  # Track original state for change detection
                    "file_type": getattr(item, "file_type", "unknown"),
                    "ml_predicted": getattr(item, "ml_predicted", False),
                    "ml_confidence": getattr(item, "ml_confidence", 0.0),
                }

                self.logger.debug(
                    f"Item {i}: {item.filename} - moisture={moisture}, "
                    f"type={getattr(item, 'file_type', 'unknown')}"
                )
        else:
            self.logger.debug("Preserving existing item_states (UI change only)")

        # Clear existing display only
        self.canvas.delete("all")
        self.cells.clear()
        self.cell_ids.clear()
        self.image_refs.clear()
        self.depth_to_row.clear()
        self.row_backgrounds.clear()
        self.selected_cells.clear()

        # Group items by depth to find duplicates - ensure all files are included
        self.depth_groups = {}
        seen_files = {}  # Track files we've seen to avoid missing any

        for idx, item in enumerate(self.review_items):
            depth = item.depth_to

            # Create unique key for this file to ensure we don't miss any
            file_key = f"{item.hole_id}_{depth}_{item.filename}"

            if depth not in self.depth_groups:
                self.depth_groups[depth] = []
                seen_files[depth] = set()

            # Add this item if we haven't seen this exact file
            if file_key not in seen_files[depth]:
                self.depth_groups[depth].append(idx)
                seen_files[depth].add(file_key)

        # Check if we have duplicates (redefined):
        # Only flag as duplicates when there is at least one UNREVIEWED item at that depth.
        # - If only approved Wet/Dry exist (and no unreviewed), do NOT flag as duplicate.
        # - If approved Wet and Dry exist AND there is a new unreviewed, flag and show all.
        # - If one approved exists and one unreviewed exists, flag and show both.
        self.duplicate_depths = {}
        for depth, indices in self.depth_groups.items():
            approved_idxs = []
            unreviewed_idxs = []
            for idx in indices:
                item = self.review_items[idx]
                # Treat 'approved' by either explicit file_type or is_reviewed flag
                is_approved = (
                    getattr(item, "is_reviewed", False)
                    or self.item_states[idx].get("file_type") == "approved"
                )
                if is_approved:
                    approved_idxs.append(idx)
                else:
                    unreviewed_idxs.append(idx)
            # Only mark duplicate when at least one unreviewed is present AND there is something else to compare to
            if unreviewed_idxs and (approved_idxs or len(unreviewed_idxs) > 1):
                self.duplicate_depths[depth] = indices
        self.has_duplicates = bool(self.duplicate_depths)

        # Determine which layout to use
        if self.has_duplicates and not getattr(self, "duplicates_resolved", False):
            # Force duplicate resolution first
            self._load_duplicate_resolution_layout()
        elif self.layout_mode == "vertical":
            # Vertical layout with duplicates side-by-side
            self._load_vertical_layout_with_duplicates()
        else:
            # Standard grid layout
            # Hide approved-only rows (no unreviewed present at that depth)
            depths_with_unreviewed = set()
            for depth, indices in self.depth_groups.items():
                if any(
                    not getattr(self.review_items[i], "is_reviewed", False)
                    and self.item_states[i].get("file_type") != "approved"
                    for i in indices
                ):
                    depths_with_unreviewed.add(depth)
            self._depths_with_unreviewed = (
                depths_with_unreviewed  # cache for _load_grid_layout
            )
            self.cols_per_row = max(
                3, (self.parent.winfo_width() - 100) // (self.cell_width + self.padding)
            )
            self._load_grid_layout()

        # Update scroll region
        self._update_scroll_region()

        # Update status table if available
        if hasattr(self.parent.master, "update_status_table"):
            self.parent.master.update_status_table()

    def _calculate_optimal_cell_size(self):
        """Calculate optimal cell size based on all loaded images."""
        if not hasattr(self, "image_dimensions"):
            self.image_dimensions = {}

        if not hasattr(self, "logger"):
            self.logger = logging.getLogger(__name__)

        if not self.review_items:
            return

        # Sample a subset of images to get dimensions
        sample_size = min(20, len(self.review_items))
        sample_indices = range(
            0, len(self.review_items), max(1, len(self.review_items) // sample_size)
        )

        widths = []
        heights = []
        aspect_ratios = []
        portrait_count = 0
        landscape_count = 0

        for idx in sample_indices:
            item = self.review_items[idx]
            if item.image_path and os.path.exists(item.image_path):
                try:
                    # Read image to get dimensions
                    img = cv2.imread(item.image_path)
                    if img is not None:
                        h, w = img.shape[:2]
                        self.image_dimensions[idx] = (w, h)

                        # Consider rotation
                        if self.image_rotation in [90, 270]:
                            w, h = h, w  # Swap for rotation

                        widths.append(w)
                        heights.append(h)
                        aspect_ratio = w / h if h > 0 else 1
                        aspect_ratios.append(aspect_ratio)

                        # Count orientations
                        if aspect_ratio < 1:
                            portrait_count += 1
                        else:
                            landscape_count += 1

                except Exception as e:
                    self.logger.debug(f"Could not read image dimensions: {e}")

        if not aspect_ratios:
            return

        # Determine dominant orientation
        is_mostly_portrait = portrait_count > landscape_count

        # Calculate median aspect ratio
        median_aspect = sorted(aspect_ratios)[len(aspect_ratios) // 2]

        # Get available canvas width
        canvas_width = self.parent.winfo_width()
        if canvas_width <= 1:  # Not yet rendered
            canvas_width = 1200  # Default assumption

        # Adjust target columns based on orientation
        if is_mostly_portrait:
            # Use more columns for portrait images
            self.target_cols = 8  # More columns for portrait
        else:
            # Fewer columns for landscape
            self.target_cols = 5  # Fewer columns for landscape

        # Calculate cell width based on target columns
        available_width = canvas_width - (self.target_cols + 1) * self.padding
        target_cell_width = available_width // self.target_cols

        # Calculate corresponding height from median aspect ratio
        target_cell_height = int(target_cell_width / median_aspect)

        # Apply reasonable limits based on orientation
        if is_mostly_portrait:
            # Portrait limits
            min_width, max_width = 150, 350
            min_height, max_height = 200, 500
        else:
            # Landscape limits
            min_width, max_width = 200, 600
            min_height, max_height = 150, 400

        self.base_cell_width = max(min_width, min(max_width, target_cell_width))
        self.base_cell_height = max(min_height, min(max_height, target_cell_height))

        # Update current dimensions
        self.cell_width = int(self.base_cell_width * self.scale_factor)
        self.cell_height = int(self.base_cell_height * self.scale_factor)

        # Recalculate columns that fit
        self.cols_per_row = max(
            1, (canvas_width - 2 * self.padding) // (self.cell_width + self.padding)
        )

        self.logger.info(
            f"Calculated optimal cell size: {self.cell_width}x{self.cell_height} "
            f"(aspect ratio: {median_aspect:.2f}, orientation: {'portrait' if is_mostly_portrait else 'landscape'}, "
            f"cols: {self.cols_per_row})"
        )

    def validate_duplicate_resolution(self) -> Tuple[bool, List[str]]:
        """Validate that all duplicates have been properly resolved."""
        # ===================================================
        # INSERT: New validation method specifically for duplicates
        errors = []

        # Check each depth with duplicates
        for depth, indices in self.duplicate_depths.items():
            # Count classifications at this depth
            wet_count = 0
            dry_count = 0
            delete_count = 0
            unclassified_count = 0

            for idx in indices:
                state = self.item_states[idx]
                if state["delete"]:
                    delete_count += 1
                elif state["moisture"] == "wet":
                    wet_count += 1
                elif state["moisture"] == "dry":
                    dry_count += 1
                else:
                    unclassified_count += 1

            # Validation rules for duplicates
            if unclassified_count > 0:
                errors.append(
                    f"Depth {depth}m: {unclassified_count} image(s) not classified"
                )

            # Allow at most 1 wet and 1 dry
            if wet_count > 1:
                errors.append(f"Depth {depth}m: Multiple wet images ({wet_count})")
            if dry_count > 1:
                errors.append(f"Depth {depth}m: Multiple dry images ({dry_count})")

            # At least one image should be kept (wet, dry, or both)
            if wet_count == 0 and dry_count == 0:
                errors.append(
                    f"Depth {depth}m: No images kept (all marked for deletion)"
                )

        return len(errors) == 0, errors
        # ===================================================

    def _load_vertical_layout_with_duplicates(self):
        """Load items vertically by depth, with duplicates shown horizontally."""
        row = 0

        for depth in sorted(self.depth_groups.keys()):
            indices = self.depth_groups[depth]

            # Skip if all items at this depth are already classified (optional filter)
            if self.duplicates_resolved:
                all_classified = all(
                    self.item_states[idx]["moisture"] or self.item_states[idx]["delete"]
                    for idx in indices
                )
                if all_classified:
                    continue

            # Create row background
            row_width = (
                self.padding + (len(indices) * (self.cell_width + self.padding)) + 60
            )  # 60 for depth label
            bg_rect = self.canvas.create_rectangle(
                0,
                row * (self.cell_height + self.padding),
                row_width,
                (row + 1) * (self.cell_height + self.padding),
                fill=self.theme_colors["row_neutral"],
                outline="",
                tags=f"row_bg_{depth}",
            )
            self.row_backgrounds[depth] = bg_rect

            # Add depth label
            self.canvas.create_text(
                30,
                row * (self.cell_height + self.padding) + self.cell_height // 2,
                text=f"{depth}m",
                fill=self.theme_colors["text"],
                font=("Arial", 14, "bold"),
                anchor="center",
                tags=f"depth_label_{depth}",
            )

            # Add items for this depth horizontally
            for col, idx in enumerate(indices):
                item = self.review_items[idx]
                # Offset by 60 pixels for depth label
                self._create_grid_cell_with_offset(row, col, idx, item, x_offset=60)

            # Validate this row
            self._validate_depth_row(depth)

            row += 1

    def _load_duplicate_resolution_layout(self):
        """Load items in vertical layout for duplicate resolution."""
        row = 0

        # Only show depths with duplicates
        for depth, indices in sorted(self.duplicate_depths.items()):
            # Log what we're displaying
            self.logger.info(f"Displaying {len(indices)} items at depth {depth}m:")
            for idx in indices:
                item = self.review_items[idx]
                self.logger.info(
                    f"  - {item.filename} from {os.path.dirname(item.image_path)}"
                )

            # Create row background
            bg_rect = self.canvas.create_rectangle(
                0,
                row * (self.cell_height + self.padding),
                self.padding + (len(indices) * (self.cell_width + self.padding)),
                (row + 1) * (self.cell_height + self.padding),
                fill=self.theme_colors["accent_error"],  # Red background for duplicates
                outline="",
                tags=f"row_bg_{depth}",
            )
            self.row_backgrounds[depth] = bg_rect

            # Add depth label
            self.canvas.create_text(
                10,
                row * (self.cell_height + self.padding) + self.cell_height // 2,
                text=f"{depth}m",
                fill="white",
                font=("Arial", 16, "bold"),
                anchor="w",
                tags=f"depth_label_{depth}",
            )

            # Add duplicate images side by side
            for col, idx in enumerate(indices):
                item = self.review_items[idx]
                self._create_grid_cell(row, col, idx, item)

            row += 1

    def _create_grid_cell_with_offset(
        self, row: int, col: int, idx: int, item: ReviewItem, x_offset: int = 0
    ):
        """Create a grid cell with custom x offset."""
        x = x_offset + col * (self.cell_width + self.padding) + self.padding
        y = row * (self.cell_height + self.padding) + self.padding

        # Create background if needed
        bg_rect = self.canvas.create_rectangle(
            x,
            y,
            x + self.cell_width,
            y + self.cell_height,
            fill="#404040",
            outline="",
            tags=f"bg_{row}_{col}",
        )

        # Rest is same as _create_grid_cell but using the idx parameter
        cell_data = {"idx": idx, "item": item, "selected": False}

        # Load and display image
        img, photo = self._prepare_image(item)
        if photo:
            self.image_refs[(row, col)] = photo

            img_id = self.canvas.create_image(
                x + self.cell_width // 2,
                y + self.cell_height // 2,
                image=photo,
                tags=f"img_{row}_{col}",
            )
        else:
            img_id = self.canvas.create_text(
                x + self.cell_width // 2,
                y + self.cell_height // 2,
                text="No Image",
                fill="white",
                font=("Arial", 12),
            )

        # Create border
        border_id = self.canvas.create_rectangle(
            x,
            y,
            x + self.cell_width,
            y + self.cell_height,
            outline=self.theme_colors["border"],
            width=3,
            tags=f"border_{row}_{col}",
        )

        # Add depth label in corner
        self.canvas.create_text(
            x + 5,
            y + self.cell_height - 5,
            text=f"{item.compartment_depth}m",
            fill="white",
            font=("Arial", 10, "bold"),
            anchor="sw",
            tags=f"depth_label_{row}_{col}",
        )

        # Store cell data
        self.cells[(row, col)] = cell_data
        self.cell_ids[(row, col)] = {
            "bg": bg_rect,
            "image": img_id,
            "border": border_id,
            "x": x,
            "y": y,
        }

        # Apply visual state
        self._update_cell_visual(row, col)

        # Update status table
        if hasattr(self.parent.master, "update_status_table"):
            self.parent.master.update_status_table()

    def _load_grid_layout(self):
        """Load items in standard grid layout."""
        # Filter out items based on current state
        items_to_show = []
        for idx, item in enumerate(self.review_items):
            state = self.item_states[idx]

            # Skip if hiding classified items and this item is classified
            if not self.show_classified:
                if state["moisture"] or state["delete"]:
                    continue

            items_to_show.append((idx, item))

        # Create grid
        grid_i = 0
        for idx, item in items_to_show:
            depth = getattr(item, "depth_to", None)
            # Skip approved-only items when the depth has no unreviewed images
            if depth is not None and hasattr(self, "_depths_with_unreviewed"):
                if (
                    self.item_states[idx].get("file_type") == "approved"
                    and depth not in self._depths_with_unreviewed
                ):
                    continue
            row = grid_i // self.cols_per_row
            col = grid_i % self.cols_per_row
            self._create_grid_cell(row, col, idx, item)
            grid_i += 1

    def _create_grid_cell(self, row: int, col: int, idx: int, item: ReviewItem):
        """Create a single grid cell."""
        x = col * (self.cell_width + self.padding) + self.padding
        y = row * (self.cell_height + self.padding) + self.padding

        # Create cell background
        bg_rect = self.canvas.create_rectangle(
            x,
            y,
            x + self.cell_width,
            y + self.cell_height,
            fill="#404040",
            outline="",
            tags=f"bg_{row}_{col}",
        )

        # Load and display image
        img, photo = self._prepare_image(item)
        if photo:
            self.image_refs[(row, col)] = photo

            img_id = self.canvas.create_image(
                x + self.cell_width // 2,
                y + self.cell_height // 2,
                image=photo,
                tags=f"img_{row}_{col}",
            )
        else:
            # Placeholder for missing image
            img_id = self.canvas.create_text(
                x + self.cell_width // 2,
                y + self.cell_height // 2,
                text="No Image",
                fill="white",
                font=("Arial", 12),
            )

        # Create border
        border_id = self.canvas.create_rectangle(
            x,
            y,
            x + self.cell_width,
            y + self.cell_height,
            outline=self.theme_colors["border"],
            width=3,
            tags=f"border_{row}_{col}",
        )

        # Add depth label
        self.canvas.create_text(
            x + 5,
            y + self.cell_height - 5,
            text=f"{item.compartment_depth}m",
            fill="white",
            font=("Arial", 10, "bold"),
            anchor="sw",
            tags=f"depth_label_{row}_{col}",
        )

        # Store cell data
        self.cells[(row, col)] = {"idx": idx, "item": item, "selected": False}

        self.cell_ids[(row, col)] = {
            "bg": bg_rect,
            "image": img_id,
            "border": border_id,
            "x": x,
            "y": y,
        }

        # Apply initial visual state
        self._update_cell_visual(row, col)

    def _prepare_image(
        self, item: ReviewItem
    ) -> Tuple[Optional[Image.Image], Optional[ImageTk.PhotoImage]]:
        """Prepare image for display with rotation support."""
        try:
            img = item.image
            if img is None:
                return None, None

            # Calculate average hex color if not already done
            if not item.average_hex_color and img is not None:
                item.average_hex_color = self._calculate_average_hex_color(img)

            # Convert to PIL
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            pil_img = Image.fromarray(img)

            # ===================================================
            # INSERT: Apply rotation if set
            if hasattr(self, "image_rotation") and self.image_rotation > 0:
                pil_img = pil_img.rotate(-self.image_rotation, expand=True)
            # ===================================================

            # Calculate scaled dimensions
            target_w = int(self.cell_width * 0.99)
            target_h = int(self.cell_height * 0.99)

            # Resize maintaining aspect ratio
            pil_img.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)

            # Create PhotoImage
            photo = ImageTk.PhotoImage(pil_img)

            return pil_img, photo

        except Exception as e:
            self.logger.error(f"Failed to prepare image: {e}")
            return None, None

    def _calculate_average_hex_color(self, img: np.ndarray) -> str:
        """Calculate the average color of an image and return as hex string."""
        try:
            # Convert BGR to RGB if needed
            if len(img.shape) == 3 and img.shape[2] == 3:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                rgb_img = img

            # Calculate mean color
            mean_color = np.mean(rgb_img.reshape(-1, 3), axis=0).astype(int)

            # Convert to hex
            hex_color = "#{:02x}{:02x}{:02x}".format(
                mean_color[0], mean_color[1], mean_color[2]
            )

            return hex_color
        except Exception as e:
            self.logger.error(f"Error calculating average hex color: {e}")
            return ""

    def _update_cell_visual(self, row: int, col: int):
        """Update visual appearance of a cell with classification text."""
        if (row, col) not in self.cells or (row, col) not in self.cell_ids:
            return

        cell = self.cells[(row, col)]
        ids = self.cell_ids[(row, col)]
        idx = cell["idx"]
        state = self.item_states[idx]

        # Check if this is an approved item (for visual distinction only)
        is_approved = state.get("file_type") == "approved"

        # Determine border appearance
        selected = cell.get("selected", False)

        # Add subtle visual indicator for approved items
        if is_approved:
            # Slightly different background to show it's already approved
            bg_color = "#2a3a2a"  # Subtle difference
            self.canvas.itemconfig(ids.get("bg"), fill=bg_color)

        # Priority: selection > classification > default
        if selected:
            border_color = self.theme_colors["accent_yellow"]
            border_width = 6
        elif state["delete"]:
            border_color = self.theme_colors["accent_red"]
            border_width = 6
        elif state["moisture"] == "wet":
            border_color = self.theme_colors["accent_blue"]
            border_width = 4
        elif state["moisture"] == "dry":
            border_color = self.theme_colors["accent_green"]
            border_width = 4
        else:
            border_color = self.theme_colors["border"]
            border_width = 3

        # Update border
        self.canvas.itemconfig(ids["border"], outline=border_color, width=border_width)

        # ===================================================
        # INSERT: Add classification text overlay
        # Remove old text
        self.canvas.delete(f"class_text_{row}_{col}")

        # Build classification text
        text_parts = []
        is_ml_prediction = state.get("ml_predicted", False)
        ml_confidence = state.get("ml_confidence", 0.0)

        if state["moisture"]:
            moisture_text = state["moisture"].upper()
            # Add confidence percentage for ML predictions
            if is_ml_prediction and ml_confidence > 0:
                confidence_pct = int(ml_confidence * 100)
                moisture_text = f"{moisture_text} {confidence_pct}%"
            text_parts.append(moisture_text)
        if state["delete"]:
            text_parts.append("DELETE")
        if state["bad_image"]:
            text_parts.append("BAD")

        if text_parts:
            text = " ".join(text_parts)
            x = ids["x"] + self.cell_width - 10
            y = ids["y"] + 10

            # Choose text color based on ML prediction and confidence
            if is_ml_prediction:
                if ml_confidence >= 0.90:
                    text_color = "#90EE90"  # Light green for high confidence
                elif ml_confidence >= 0.70:
                    text_color = "#FFD700"  # Gold for medium confidence
                else:
                    text_color = "#FFA500"  # Orange for low confidence
            else:
                text_color = "white"  # White for user-confirmed

            # Create text with background
            text_id = self.canvas.create_text(
                x,
                y,
                text=text,
                fill=text_color,
                font=("Arial", 10, "bold"),
                anchor="ne",
                tags=f"class_text_{row}_{col}",
            )

            # Add semi-transparent background
            bbox = self.canvas.bbox(text_id)
            if bbox:
                bg_id = self.canvas.create_rectangle(
                    bbox[0] - 2,
                    bbox[1] - 2,
                    bbox[2] + 2,
                    bbox[3] + 2,
                    fill="#000000",
                    stipple="gray50",
                    outline="",
                    tags=f"class_text_{row}_{col}",
                )
                self.canvas.tag_raise(text_id)
        # ===================================================

    def _preview_cells_in_box(self, x1: float, y1: float, x2: float, y2: float):
        """Preview cells that would be affected by the drag box."""
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)

        # Clear previous preview
        for row, col in getattr(self, "cells_in_drag", set()):
            if (row, col) in self.cell_ids:
                self._update_cell_visual(row, col)

        # Find cells in box
        new_cells_in_drag = set()
        for (row, col), ids in self.cell_ids.items():
            cell_x = ids["x"]
            cell_y = ids["y"]

            # Check if cell is within the box
            if (
                cell_x < max_x
                and cell_x + self.cell_width > min_x
                and cell_y < max_y
                and cell_y + self.cell_height > min_y
            ):
                new_cells_in_drag.add((row, col))

        # Update preview for cells in drag
        self.cells_in_drag = new_cells_in_drag
        for row, col in self.cells_in_drag:
            if (row, col) in self.cell_ids:
                # Show preview highlight
                ids = self.cell_ids[(row, col)]
                self.canvas.itemconfig(
                    ids["border"],
                    outline=self.theme_colors["accent_yellow"],
                    width=4,
                    dash=(3, 3),
                )

    def _on_mouse_down(self, event):
        """Handle mouse down for selection/action."""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # Store the clicked cell and drag start
        self.clicked_cell = self._get_cell_at_position(canvas_x, canvas_y)
        self.drag_start = (canvas_x, canvas_y)
        self.cells_in_drag = set()  # Track cells in current drag

        # Always allow starting a drag for box actions
        self.drag_start = (canvas_x, canvas_y)

    def _on_mouse_drag(self, event):
        """Handle mouse drag for box selection."""
        if not self.drag_start:
            return

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # Check if near edge and start auto-scrolling
        self._check_edge_scroll(event.y)

        # Draw the action box
        if self.selection_box:
            self.canvas.delete(self.selection_box)

        # Use different colors for different modes
        box_colors = {
            "wet": self.theme_colors["accent_blue"],
            "dry": self.theme_colors["accent_green"],
            "delete": self.theme_colors["accent_red"],
            "bad_image": self.theme_colors["accent_yellow"],
        }

        box_color = box_colors.get(
            self.current_mode, self.theme_colors["accent_yellow"]
        )

        self.selection_box = self.canvas.create_rectangle(
            self.drag_start[0],
            self.drag_start[1],
            canvas_x,
            canvas_y,
            outline=box_color,
            width=3,
            dash=(5, 5),
            tags="selection_box",
        )

        # Preview cells in box without applying
        self._preview_cells_in_box(
            self.drag_start[0], self.drag_start[1], canvas_x, canvas_y
        )

    def _check_edge_scroll(self, mouse_y):
        """Check if mouse is near edge and trigger auto-scrolling."""
        canvas_height = self.canvas.winfo_height()

        # Check if near top edge
        if mouse_y < self.scroll_edge_threshold:
            if self.auto_scroll_direction != "up":
                self.auto_scroll_direction = "up"
                self.auto_scroll_speed = 0  # Reset speed when changing direction
            self._start_auto_scroll()
        # Check if near bottom edge
        elif mouse_y > (canvas_height - self.scroll_edge_threshold):
            if self.auto_scroll_direction != "down":
                self.auto_scroll_direction = "down"
                self.auto_scroll_speed = 0  # Reset speed when changing direction
            self._start_auto_scroll()
        else:
            # Not near edge, stop scrolling
            self._stop_auto_scroll()

    def _start_auto_scroll(self):
        """Start or continue auto-scrolling."""
        if self.auto_scroll_timer is None:
            self._perform_auto_scroll()

    def _stop_auto_scroll(self):
        """Stop auto-scrolling."""
        if self.auto_scroll_timer:
            self.canvas.after_cancel(self.auto_scroll_timer)
            self.auto_scroll_timer = None
        self.auto_scroll_speed = 0
        self.auto_scroll_direction = None

    def _perform_auto_scroll(self):
        """Perform one step of auto-scrolling with acceleration."""
        if self.auto_scroll_direction is None:
            return

        # Accelerate: start slow, increase to max
        max_speed = 5  # Maximum scroll units per step
        acceleration = 0.05  # How fast to accelerate

        self.auto_scroll_speed = min(self.auto_scroll_speed + acceleration, max_speed)

        # Calculate scroll amount (minimum 1 unit)
        scroll_amount = max(1, int(self.auto_scroll_speed))

        # Scroll in the appropriate direction
        if self.auto_scroll_direction == "up":
            self.canvas.yview_scroll(-scroll_amount, "units")
        elif self.auto_scroll_direction == "down":
            self.canvas.yview_scroll(scroll_amount, "units")

        # Schedule next scroll with faster interval as speed increases
        # Start at 50ms, decrease to 10ms as speed increases
        interval = max(10, int(50 - (self.auto_scroll_speed * 2)))
        self.auto_scroll_timer = self.canvas.after(interval, self._perform_auto_scroll)

    def _on_mouse_release(self, event):
        """Handle mouse release."""
        # Stop auto-scrolling
        self._stop_auto_scroll()

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # Check if this was a click or a drag
        if self.drag_start:
            start_x, start_y = self.drag_start
            distance = ((canvas_x - start_x) ** 2 + (canvas_y - start_y) ** 2) ** 0.5

            if distance < 5 and self.clicked_cell:
                # This was a click, not a drag
                row, col = self.clicked_cell
                self._apply_action_to_cell(row, col)
            elif self.cells_in_drag:
                # This was a drag - apply to all previewed cells
                for row, col in self.cells_in_drag:
                    cell = self.cells[(row, col)]
                    idx = cell["idx"]
                    state = self.item_states[idx]

                    # Toggle behavior for drag operations
                    if self.current_mode == "wet":
                        # Toggle wet: if already wet, remove it; otherwise set wet
                        if state["moisture"] == "wet":
                            state["moisture"] = None
                        else:
                            state["moisture"] = "wet"
                            state["delete"] = False
                        # Clear ML prediction flag - user has manually classified
                        state["ml_predicted"] = False
                    elif self.current_mode == "dry":
                        # Toggle dry: if already dry, remove it; otherwise set dry
                        if state["moisture"] == "dry":
                            state["moisture"] = None
                        else:
                            state["moisture"] = "dry"
                            state["delete"] = False
                        # Clear ML prediction flag - user has manually classified
                        state["ml_predicted"] = False
                    elif self.current_mode == "delete":
                        # Toggle delete
                        state["delete"] = not state["delete"]
                        if state["delete"]:
                            state["moisture"] = None
                        state["ml_predicted"] = False
                    elif self.current_mode == "bad_image":
                        # Toggle bad image flag
                        state["bad_image"] = not state["bad_image"]

                    self._update_cell_visual(row, col)

                # Validate row if in vertical layout
                if self.layout_mode == "vertical":
                    # Get unique depths that were modified
                    modified_depths = set()
                    for r, c in self.cells_in_drag:
                        if (r, c) in self.cells:
                            item = self.cells[(r, c)]["item"]
                            modified_depths.add(item.depth_to)

                    # Validate each modified depth
                    for depth in modified_depths:
                        self._validate_depth_row(depth)

        # Clear preview
        if self.selection_box:
            self.canvas.delete(self.selection_box)
            self.selection_box = None

        # Clear preview highlights
        for row, col in getattr(self, "cells_in_drag", set()):
            self._update_cell_visual(row, col)

        # Reset state
        self.drag_start = None
        self.clicked_cell = None
        self.cells_in_drag = set()

        # Update status table after action
        if hasattr(self.parent.master, "update_status_table"):
            try:
                self.parent.master.update_status_table()
            except Exception as e:
                self.logger.error(f"Error updating status table: {e}")

    def _get_cell_at_position(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """Get cell coordinates at canvas position."""
        for (row, col), ids in self.cell_ids.items():
            cell_x = ids["x"]
            cell_y = ids["y"]

            if (
                cell_x <= x <= cell_x + self.cell_width
                and cell_y <= y <= cell_y + self.cell_height
            ):
                return (row, col)

        return None

    def _select_cell(self, row: int, col: int):
        """Select a single cell."""
        if (row, col) in self.cells:
            self.selected_cells.add((row, col))
            self.cells[(row, col)]["selected"] = True
            self._update_cell_visual(row, col)

    def _toggle_cell_selection(self, row: int, col: int):
        """Toggle selection state of a cell."""
        if (row, col) in self.selected_cells:
            self.selected_cells.remove((row, col))
            self.cells[(row, col)]["selected"] = False
        else:
            self._select_cell(row, col)
        self._update_cell_visual(row, col)

    def _select_range_to(self, row: int, col: int):
        """Select range from last selected to this cell."""
        if not self.selected_cells:
            self._select_cell(row, col)
            return

        # Get last selected
        last_row, last_col = max(self.selected_cells)

        # Calculate range
        min_row, max_row = min(row, last_row), max(row, last_row)
        min_col, max_col = min(col, last_col), max(col, last_col)

        # Select all cells in range
        for r in range(min_row, max_row + 1):
            for c in range(min_col, max_col + 1):
                if (r, c) in self.cells:
                    self._select_cell(r, c)

    def _select_cells_in_box(self, x1: float, y1: float, x2: float, y2: float):
        """Select all cells within the box."""
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)

        self._clear_selection()

        for (row, col), ids in self.cell_ids.items():
            cell_x = ids["x"]
            cell_y = ids["y"]

            if (
                cell_x < max_x
                and cell_x + self.cell_width > min_x
                and cell_y < max_y
                and cell_y + self.cell_height > min_y
            ):
                self._select_cell(row, col)

    def _clear_selection(self):
        """Clear all selected cells."""
        for row, col in self.selected_cells:
            self.cells[(row, col)]["selected"] = False
            self._update_cell_visual(row, col)
        self.selected_cells.clear()

    def validate_classifications(self) -> Tuple[bool, List[str]]:
        """Validate all classifications according to rules."""
        errors = []

        # Group items by depth
        depth_groups = {}
        for idx, item in enumerate(self.review_items):
            if item.is_placeholder:
                continue
            state = self.item_states[idx]
            depth = item.depth_to

            if depth not in depth_groups:
                depth_groups[depth] = []

            depth_groups[depth].append({"idx": idx, "state": state, "item": item})

        # Check each depth
        for depth, items in depth_groups.items():
            # Check if all items are classified
            unclassified = [
                i
                for i in items
                if not i["state"]["moisture"] and not i["state"]["delete"]
            ]
            if unclassified:
                errors.append(
                    f"Depth {depth}m has {len(unclassified)} unclassified image(s)"
                )

            # Check for multiple wet/dry of same type
            wet_items = [i for i in items if i["state"]["moisture"] == "wet"]
            dry_items = [i for i in items if i["state"]["moisture"] == "dry"]

            if len(wet_items) > 1:
                errors.append(
                    f"Depth {depth}m has {len(wet_items)} wet images (max 1 allowed)"
                )
            if len(dry_items) > 1:
                errors.append(
                    f"Depth {depth}m has {len(dry_items)} dry images (max 1 allowed)"
                )

        return len(errors) == 0, errors

    def _apply_action_to_cell(self, row: int, col: int):
        """Apply current mode action to a cell with toggle behavior."""
        if (row, col) not in self.cells:
            return

        cell = self.cells[(row, col)]
        idx = cell["idx"]
        state = self.item_states[idx]

        # Handle click behavior based on mode
        if self.current_mode == "wet":
            # Toggle wet classification
            if state["moisture"] == "wet":
                state["moisture"] = None
            else:
                state["moisture"] = "wet"
                state["delete"] = False
            state["ml_predicted"] = False  # User manually classified
        elif self.current_mode == "dry":
            # Toggle dry classification
            if state["moisture"] == "dry":
                state["moisture"] = None
            else:
                state["moisture"] = "dry"
                state["delete"] = False
            state["ml_predicted"] = False  # User manually classified
        elif self.current_mode == "delete":
            # Toggle delete
            state["delete"] = not state["delete"]
            if state["delete"]:
                state["moisture"] = None
            state["ml_predicted"] = False
        elif self.current_mode == "bad_image":
            # Toggle bad image flag (doesn't affect moisture)
            state["bad_image"] = not state["bad_image"]

        self._update_cell_visual(row, col)

        # Validate row if in vertical layout
        if self.layout_mode == "vertical" and hasattr(self, "_validate_depth_row"):
            item = self.cells[(row, col)]["item"]
            self._validate_depth_row(item.depth_to)

        # Update status table
        if hasattr(self.parent.master, "update_status_table"):
            self.parent.master.update_status_table()

    def apply_action_to_selected(self, action: str):
        """Apply an action to all selected cells."""
        for row, col in self.selected_cells:
            cell = self.cells[(row, col)]
            idx = cell["idx"]
            state = self.item_states[idx]

            if action == "wet":
                state["moisture"] = "wet"
                state["delete"] = False
                state["ml_predicted"] = False  # User manually classified
            elif action == "dry":
                state["moisture"] = "dry"
                state["delete"] = False
                state["ml_predicted"] = False  # User manually classified
            elif action == "delete":
                state["delete"] = True
                state["moisture"] = None
                state["ml_predicted"] = False

            self._update_cell_visual(row, col)

        self._clear_selection()

    def set_mode(self, mode: str):
        """Set the current interaction mode."""
        self.current_mode = mode

        # Update cursor
        if mode == "select":
            self.canvas.configure(cursor="arrow")
        elif mode in ["wet", "dry"]:
            self.canvas.configure(cursor="hand2")
        elif mode == "delete":
            self.canvas.configure(cursor="X_cursor")

    def _validate_depth_row(self, depth: int):
        """Validate and update visual feedback for a depth row."""
        if depth not in self.depth_groups:
            return

        indices = self.depth_groups[depth]

        # Count classifications
        wet_count = 0
        dry_count = 0
        delete_count = 0
        unclassified_count = 0

        for idx in indices:
            state = self.item_states[idx]
            if state["delete"]:
                delete_count += 1
            elif state["moisture"] == "wet":
                wet_count += 1
            elif state["moisture"] == "dry":
                dry_count += 1
            else:
                unclassified_count += 1

        # Determine row color based on validation
        if unclassified_count > 0:
            # Has unclassified items
            bg_color = self.theme_colors.get("row_invalid", "#3a2222")
        elif wet_count > 1 or dry_count > 1:
            # Multiple of same type
            bg_color = self.theme_colors.get("row_invalid", "#3a2222")
        elif wet_count == 0 and dry_count == 0 and delete_count == len(indices):
            # All deleted
            bg_color = self.theme_colors.get("row_invalid", "#3a2222")
        else:
            # Valid configuration
            bg_color = self.theme_colors.get("row_valid", "#223a22")

        # Update row background if it exists
        if depth in self.row_backgrounds:
            self.canvas.itemconfig(self.row_backgrounds[depth], fill=bg_color)

    def toggle_show_classified(self):
        """Toggle showing/hiding classified items."""
        self.show_classified = not self.show_classified
        # Reload the display
        self.load_items(self.review_items, self.hole_id)

    def get_results(self) -> Tuple[List[ReviewItem], List[ReviewItem]]:
        """Get reviewed and unreviewed items."""
        reviewed = []
        unreviewed = []

        for idx, item in enumerate(self.review_items):
            state = self.item_states[idx]

            # Transfer bad_image state to the item
            item.bad_image = state.get("bad_image", False)

            if state["delete"]:
                item.quality = "Missing"
                item.is_reviewed = True
                reviewed.append(item)
            elif state["moisture"]:
                item.moisture = state["moisture"].capitalize()
                item.quality = "OK"
                item.is_reviewed = True
                reviewed.append(item)
            else:
                item.is_reviewed = False
                unreviewed.append(item)

        return reviewed, unreviewed

    def get_status_counts(self) -> Dict[str, int]:
        """Get counts for status bar."""
        counts = {
            "total": len(self.review_items),
            "wet": 0,
            "dry": 0,
            "delete": 0,
            "unclassified": 0,
            "selected": len(self.selected_cells),
        }

        for state in self.item_states.values():
            if state["delete"]:
                counts["delete"] += 1
            elif state["moisture"] == "wet":
                counts["wet"] += 1
            elif state["moisture"] == "dry":
                counts["dry"] += 1
            else:
                counts["unclassified"] += 1

        return counts

    def _on_mousewheel(self, event):
        # Ignore if canvas has been destroyed
        if not hasattr(self, "canvas") or not str(self.canvas):
            return
        if not (event.state & 0x0004):  # Not Ctrl
            try:
                self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except tk.TclError:
                return


    def _on_ctrl_mousewheel(self, event):
        """Handle Ctrl+MouseWheel for zooming."""
        if event.delta > 0:
            zoom = 1.1
        else:
            zoom = 0.9

        new_scale = self.scale_factor * zoom
        new_scale = max(0.5, min(2.0, new_scale))

        if new_scale != self.scale_factor:
            self.scale_factor = new_scale
            self.cell_width = int(self.base_cell_width * self.scale_factor)
            self.cell_height = int(self.base_cell_height * self.scale_factor)

            # Reload with new scale
            self.load_items(
                self.review_items, self.hole_id
            )

    def _update_scroll_region(self):
        """Update canvas scroll region."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        """Handle canvas resize."""
        # ===================================================
        # REPLACE: Recalculate columns on resize
        new_width = event.width
        if new_width > 100:  # Meaningful resize
            new_cols = max(
                1, (new_width - 2 * self.padding) // (self.cell_width + self.padding)
            )
            if new_cols != self.cols_per_row and self.layout_mode == "grid":
                self.cols_per_row = new_cols
                # Reload grid layout
                self.canvas.after_idle(
                    lambda: self.load_items(self.review_items, self.hole_id)
                )
        # ===================================================


# ============================================================================
# GRID REVIEW DIALOG
# ============================================================================


class QAQCGridReviewDialog:
    """Grid-based review dialog for rapid QAQC classification."""

    def __init__(self, parent, app, translator_func, logger):
        self.parent = parent
        self.app = app
        self.t = translator_func
        self.logger = logger
        self.gui_manager = app.gui_manager if hasattr(app, "gui_manager") else None

        self.review_window = None
        self.close_callback = None
        self.grid_canvas = None
        self.status_label = None
        self.mode_buttons = {}
        self.save_btn = None  # Store reference to update text dynamically
        self.close_btn = None  # Store reference for Close button

        # Queue support for multiple holes
        self.hole_queue = []  # List of hole_ids to process
        self.current_hole_index = -1
        self.items_by_hole = {}  # Store the original dict reference
        self.hole_processor = None  # Function to lazy-load hole data

    def _reload_hole_data(self, hole_id: str, review_items: List[ReviewItem]):
        """Reload new hole data into existing window."""
        # Update window title with queue position
        queue_info = ""
        if self.hole_queue:
            queue_info = f" ({self.current_hole_index + 1}/{len(self.hole_queue)})"
        self.review_window.title(f"QAQC Grid Review - {hole_id}{queue_info}")

        # Clear and reload grid canvas with reset_states=True (new hole)
        if self.grid_canvas:
            self.grid_canvas.load_items(review_items, hole_id, reset_states=True)
            # Scroll to top of canvas for new hole
            self.grid_canvas.canvas.yview_moveto(0)

        # Update status
        self._update_status()

        # Update button labels
        self._update_button_labels()

        self.logger.info(
            f"Reloaded window with {len(review_items)} items for {hole_id}"
        )

    def _cleanup_previous_hole_memory(self):
        """Clear memory from previously loaded hole to prevent accumulation."""
        self.logger.debug("Cleaning up previous hole memory...")

        # Clear grid canvas image references
        if self.grid_canvas:
            # Clear PhotoImage references
            if hasattr(self.grid_canvas, "image_refs"):
                self.grid_canvas.image_refs.clear()

            # Unload images from ReviewItems
            if hasattr(self.grid_canvas, "review_items"):
                for item in self.grid_canvas.review_items:
                    if hasattr(item, "unload_image"):
                        item.unload_image()

        # Force garbage collection
        import gc

        gc.collect()

        self.logger.debug("Memory cleanup complete")

    def _update_button_labels(self):
        """Update Save button text based on queue position."""
        if not self.save_btn:
            return

        # Check if there are more holes after current
        has_next = (
            self.hole_queue and self.current_hole_index < len(self.hole_queue) - 1
        )

        if has_next:
            new_text = (
                f"Save & Next ({self.current_hole_index + 1}/{len(self.hole_queue)})"
            )
        else:
            new_text = "Save & Close (Last Hole)"

        self.save_btn.configure(text=new_text)

    def show_review_window(self, hole_id: str, review_items: List[ReviewItem]):
        """Show the grid review window."""
        # If window doesn't exist, create it
        if not self.review_window or not self.review_window.winfo_exists():
            self._create_window(hole_id, review_items)
        else:
            # Window exists, just reload data
            self._reload_hole_data(hole_id, review_items)

    def load_hole_queue(self, hole_ids: List[str], processor_callback):
        """
        Queue multiple holes for review.

        Args:
            hole_ids: List of hole IDs to review
            processor_callback: Function that takes hole_id and returns processed review_items
        """
        self.hole_queue = hole_ids
        self.hole_processor = processor_callback
        self.current_hole_index = 0

        # Load first hole with progress dialog
        if self.hole_queue:
            first_hole = self.hole_queue[0]
            self._load_first_hole_with_progress(first_hole)

    def _load_first_hole_with_progress(self, hole_id: str):
        """Load first hole data with progress feedback."""
        from gui.progress_dialog import ProgressDialog

        # Create progress dialog
        progress = ProgressDialog(
            self.parent, "Loading Review", f"Loading {hole_id}..."
        )

        def load_task():
            """Task to run in background thread."""
            # Update progress - scanning
            progress.update_progress(f"Scanning compartments for {hole_id}...", 30)

            # Process the hole (this is the slow part)
            review_items = self.hole_processor(hole_id)

            # Update progress - analyzing
            progress.update_progress(f"Analyzing files for {hole_id}...", 70)

            # Small delay to show final progress
            import time

            time.sleep(0.2)

            # Update progress - complete
            progress.update_progress(f"Loading complete", 100)

            return review_items

        try:
            # Run with progress dialog (blocks until complete)
            review_items = progress.run_with_progress(load_task)

            # Show window with loaded data
            if review_items:
                self.show_review_window(hole_id, review_items)
            else:
                self.logger.info(f"No items to review for {hole_id}")
        except Exception as e:
            self.logger.error(f"Error loading hole data: {e}")
            DialogHelper.show_message(
                self.parent,
                "Error",
                f"Failed to load {hole_id}: {str(e)}",
                message_type="error",
            )

    def _create_window(self, hole_id: str, review_items: List[ReviewItem]):
        """Create the main window."""
        if self.review_window and self.review_window.winfo_exists():
            self.review_window.destroy()

        self.review_window = tk.Toplevel(self.parent)
        self.review_window.title(f"QAQC Grid Review - {hole_id}")

        # Apply theme
        if self.gui_manager:
            self.gui_manager.configure_ttk_styles(self.review_window)

        # Make it nearly fullscreen
        screen_width = self.review_window.winfo_screenwidth()
        screen_height = self.review_window.winfo_screenheight()
        window_width = int(screen_width * 0.95)
        window_height = int(screen_height * 0.95)
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.review_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Create main container frame for horizontal layout
        main_container = ttk.Frame(self.review_window)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left side container for toolbar and canvas
        left_container = ttk.Frame(main_container)
        left_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create toolbar in left container
        self._create_toolbar(left_container)

        # Create canvas area in left container
        canvas_frame = ttk.Frame(left_container)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self._create_canvas_area(hole_id, review_items, parent=canvas_frame)

        # Create status bar at bottom of left container
        self._create_status_bar(left_container)

        # Setup keyboard shortcuts
        self._setup_keyboard_shortcuts()

        self.review_window.protocol("WM_DELETE_WINDOW", self._on_close)

    def _create_toolbar(self, parent=None):
        """Create the toolbar with mode buttons and actions."""
        # Use provided parent or default to review_window
        toolbar_parent = parent if parent else self.review_window

        toolbar = ttk.Frame(toolbar_parent)
        toolbar.pack(fill=tk.X, padx=5, pady=5)

        # Mode section
        mode_frame = ttk.LabelFrame(toolbar, text="Classification Mode", padding=5)
        mode_frame.pack(side=tk.LEFT, padx=5)

        # Mode buttons with keyboard shortcuts
        modes = [
            (
                "Wet (1)",
                "wet",
                (
                    self.gui_manager.theme_colors["accent_blue"]
                    if self.gui_manager
                    else "#1e88e5"
                ),
            ),
            (
                "Dry (2)",
                "dry",
                (
                    self.gui_manager.theme_colors["accent_green"]
                    if self.gui_manager
                    else "#43a047"
                ),
            ),
            (
                "Delete (3)",
                "delete",
                (
                    self.gui_manager.theme_colors["accent_red"]
                    if self.gui_manager
                    else "#e53935"
                ),
            ),
            (
                "Bad Image (B)",
                "bad_image",
                (
                    self.gui_manager.theme_colors["accent_yellow"]
                    if self.gui_manager
                    else "#fdd835"
                ),
            ),
        ]

        for text, mode, color in modes:
            if self.gui_manager:
                btn = self.gui_manager.create_modern_button(
                    mode_frame, text, color, lambda m=mode: self._set_mode(m)
                )
            else:
                btn = ttk.Button(
                    mode_frame, text=text, command=lambda m=mode: self._set_mode(m)
                )

            self.mode_buttons[mode] = btn
            btn.pack(side=tk.LEFT, padx=2)

        # Set initial mode to wet
        self.current_mode = "wet"

        # Duplicate resolution controls
        self.duplicate_frame = ttk.LabelFrame(
            toolbar, text="Duplicate Resolution", padding=5
        )
        # Don't pack yet - will show only when needed

        if self.gui_manager:
            self.apply_dup_button = self.gui_manager.create_modern_button(
                self.duplicate_frame,
                "Apply & Continue",
                self.gui_manager.theme_colors["accent_green"],
                self._apply_duplicate_resolution,
            )
        else:
            self.apply_dup_button = ttk.Button(
                self.duplicate_frame,
                text="Apply & Continue",
                command=self._apply_duplicate_resolution,
            )
        self.apply_dup_button.pack(side=tk.LEFT, padx=2)

        # Actions section with themed buttons
        action_frame = ttk.LabelFrame(toolbar, text="Actions", padding=5)
        action_frame.pack(side=tk.LEFT, padx=20)

        if self.gui_manager:
            # Image display controls
            display_frame = ttk.Frame(action_frame)
            display_frame.pack(side=tk.LEFT, padx=5)

            ttk.Label(display_frame, text="Display:").pack(side=tk.LEFT, padx=2)

            # Toggle classified button
            self.toggle_classified_btn = self.gui_manager.create_modern_button(
                display_frame,
                "Hide Classified",
                self.gui_manager.theme_colors["secondary_bg"],
                self._toggle_show_classified,
            )
            self.toggle_classified_btn.pack(side=tk.LEFT, padx=2)

            self.gui_manager.create_modern_button(
                display_frame,
                "Rotate",
                self.gui_manager.theme_colors["secondary_bg"],
                self._rotate_images,
            ).pack(side=tk.LEFT, padx=2)

            self.gui_manager.create_modern_button(
                display_frame,
                "Grid Layout",
                self.gui_manager.theme_colors["secondary_bg"],
                lambda: self._set_layout("grid"),
            ).pack(side=tk.LEFT, padx=2)

            self.gui_manager.create_modern_button(
                display_frame,
                "Column Layout",
                self.gui_manager.theme_colors["secondary_bg"],
                lambda: self._set_layout("column"),
            ).pack(side=tk.LEFT, padx=2)

            # Size controls
            size_frame = ttk.Frame(action_frame)
            size_frame.pack(side=tk.LEFT, padx=10)

            ttk.Label(size_frame, text="Size:").pack(side=tk.LEFT, padx=2)

            self.gui_manager.create_modern_button(
                size_frame,
                "Smaller",
                self.gui_manager.theme_colors["secondary_bg"],
                lambda: self._scale_images(0.8),
            ).pack(side=tk.LEFT, padx=2)

            self.gui_manager.create_modern_button(
                size_frame,
                "Bigger",
                self.gui_manager.theme_colors["secondary_bg"],
                lambda: self._scale_images(1.2),
            ).pack(side=tk.LEFT, padx=2)

            # Reset button
            self.gui_manager.create_modern_button(
                action_frame,
                "Reset All",
                self.gui_manager.theme_colors["accent_red"],
                self._reset_all,
            ).pack(side=tk.LEFT, padx=10)
        else:
            # Fallback non-themed buttons
            ttk.Button(action_frame, text="Rotate", command=self._rotate_images).pack(
                side=tk.LEFT, padx=2
            )
            ttk.Button(
                action_frame,
                text="Grid Layout",
                command=lambda: self._set_layout("grid"),
            ).pack(side=tk.LEFT, padx=2)
            ttk.Button(
                action_frame,
                text="Column Layout",
                command=lambda: self._set_layout("column"),
            ).pack(side=tk.LEFT, padx=2)
            ttk.Button(
                action_frame, text="Smaller", command=lambda: self._scale_images(0.8)
            ).pack(side=tk.LEFT, padx=2)
            ttk.Button(
                action_frame, text="Bigger", command=lambda: self._scale_images(1.2)
            ).pack(side=tk.LEFT, padx=2)
            ttk.Button(action_frame, text="Reset All", command=self._reset_all).pack(
                side=tk.LEFT, padx=2
            )

        # Save/Close buttons on the right
        button_frame = ttk.Frame(toolbar)
        button_frame.pack(side=tk.RIGHT, padx=5)

        if self.gui_manager:
            self.save_btn = self.gui_manager.create_modern_button(
                button_frame,
                "Save & Next",  # Will update dynamically
                self.gui_manager.theme_colors["accent_green"],
                self._save_and_next,
            )
            self.close_btn = self.gui_manager.create_modern_button(
                button_frame,
                "Close",
                self.gui_manager.theme_colors["accent_red"],
                self._close_without_save,
            )
            cancel_btn = self.gui_manager.create_modern_button(
                button_frame,
                "Cancel",
                self.gui_manager.theme_colors["secondary_bg"],
                self._on_close,
            )
        else:
            self.save_btn = ttk.Button(
                button_frame, text="Save & Next", command=self._save_and_next
            )
            self.close_btn = ttk.Button(
                button_frame, text="Close", command=self._close_without_save
            )
            cancel_btn = ttk.Button(button_frame, text="Cancel", command=self._on_close)

        self.save_btn.pack(side=tk.LEFT, padx=5)
        self.close_btn.pack(side=tk.LEFT, padx=5)
        cancel_btn.pack(side=tk.LEFT, padx=5)

        # Update button text based on queue status
        self._update_button_labels()

        # Set initial mode after UI is created
        self.review_window.after(100, lambda: self._set_mode("dry"))

    def _create_status_bar(self, parent=None):
        """Create status bar at bottom."""
        status_parent = parent if parent else self.review_window

        status_frame = ttk.Frame(status_parent)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_label = ttk.Label(status_frame, text="", relief=tk.SUNKEN)
        self.status_label.pack(fill=tk.X, padx=5, pady=2)

        self._update_status()

    def _create_vertical_status_table(self, parent):
        """Create vertical status table on the right side."""
        # Create a frame for the status table
        self.table_frame = ttk.LabelFrame(
            parent, text="Register Update Preview", padding=5
        )
        self.table_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))

        # Create a frame to hold both treeview and scrollbars
        self.tree_container = ttk.Frame(self.table_frame)
        self.tree_container.pack(fill=tk.BOTH, expand=True)

        # Create treeview for status table - taller for vertical layout
        self.status_tree = ttk.Treeview(
            self.tree_container,
            columns=(
                "file_location",
                "current_status",
                "new_status",
                "action",
                "validation",
            ),
            height=30,  # Increased height for vertical layout
            show="tree headings",
        )

        # Configure columns for narrower width
        self.status_tree.heading("#0", text="Depth")
        self.status_tree.heading("file_location", text="Location")
        self.status_tree.heading("current_status", text="Current Status")
        self.status_tree.heading("new_status", text="New Status")
        self.status_tree.heading("action", text="Action")
        self.status_tree.heading("validation", text="Validation")

        # Narrower columns to minimize width
        self.status_tree.column("#0", width=60, stretch=False)
        self.status_tree.column("file_location", width=150, stretch=False)
        self.status_tree.column("current_status", width=120, stretch=False)
        self.status_tree.column("new_status", width=120, stretch=False)
        self.status_tree.column("action", width=80, stretch=False)
        self.status_tree.column("validation", width=140, stretch=False)

        # Add vertical scrollbar
        v_scrollbar = ttk.Scrollbar(
            self.tree_container, orient="vertical", command=self.status_tree.yview
        )
        # Add horizontal scrollbar
        h_scrollbar = ttk.Scrollbar(
            self.tree_container, orient="horizontal", command=self.status_tree.xview
        )

        self.status_tree.configure(
            yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set
        )

        # Grid layout for proper scrollbar placement
        self.status_tree.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        self.tree_container.grid_rowconfigure(0, weight=1)
        self.tree_container.grid_columnconfigure(0, weight=1)

        # Initial update
        self.update_status_table()

    def update_status_table(self):
        """Update the status table with current classifications."""
        # Debug logging
        self.logger.info("Updating status table...")

        # Clear existing items
        for item in self.status_tree.get_children():
            self.status_tree.delete(item)

        # Get current results from grid
        reviewed, unreviewed = self.grid_canvas.get_results()
        all_items = reviewed + unreviewed

        self.logger.info(
            f"Total items: {len(all_items)}, Reviewed: {len(reviewed)}, Unreviewed: {len(unreviewed)}"
        )

        # Group by depth
        depth_groups = {}
        for item in all_items:
            depth = item.depth_to
            if depth not in depth_groups:
                depth_groups[depth] = []
            depth_groups[depth].append(item)

        # Add rows for each depth
        for depth in sorted(depth_groups.keys()):
            items = depth_groups[depth]

            # Determine file location and source color
            file_location = "Unknown"
            row_tag = "neutral"

            # Check file locations for items at this depth
            for item in items:
                # Get file location info
                location_info = self._get_file_location(item)
                if location_info:
                    file_location = location_info["display_name"]
                    row_tag = location_info["tag"]
                    break  # Use first valid location

            # Get current status from register
            current_status = "OK_unknown"
            if items:
                # Check register status from first item (should be same for all at same depth)
                register_status = items[0].register_status
                if register_status:
                    current_status = register_status
                elif self.grid_canvas.scanner and hasattr(
                    self.grid_canvas.scanner, "get_register_status"
                ):
                    current_status = (
                        self.grid_canvas.scanner.get_register_status(
                            items[0].hole_id, depth
                        )
                        or "OK_unknown"
                    )

            # Determine new status from classifications
            new_statuses = []
            has_changes = False

            for item in items:
                idx = self.grid_canvas.review_items.index(item)
                state = self.grid_canvas.item_states[idx]

                if state["delete"]:
                    new_statuses.append("Delete")
                    has_changes = True
                elif state["moisture"]:
                    status = f"OK_{state['moisture'].capitalize()}"
                    if state.get("bad_image"):
                        status += "_Bad"
                    new_statuses.append(status)
                    has_changes = True
                else:
                    new_statuses.append("Unclassified")

            # Determine new status string
            if "Unclassified" in new_statuses:
                new_status = "Pending..."
            else:
                new_status = ", ".join(new_statuses)

            # Determine action and validation
            if not has_changes:
                action = "—"
                validation = "No changes"
            elif "Unclassified" in new_statuses:
                action = "Pending"
                validation = "⚠️ Needs classification"
            elif len([s for s in new_statuses if s.startswith("OK_Wet")]) > 1:
                action = "Error"
                validation = "❌ Multiple wet files"
            elif len([s for s in new_statuses if s.startswith("OK_Dry")]) > 1:
                action = "Error"
                validation = "❌ Multiple dry files"
            else:
                action = "Update"
                validation = "✅ Ready"

            # Insert row with tag for coloring
            tree_item = self.status_tree.insert(
                "",
                "end",
                text=f"{depth}m",
                values=(file_location, current_status, new_status, action, validation),
                tags=(row_tag,),
            )

        # Configure tag colors
        self.status_tree.tag_configure("shared_approved", background="#2d5a2d")  # Green
        self.status_tree.tag_configure("shared_review", background="#5a5a2d")  # Yellow
        self.status_tree.tag_configure("local_temp", background="#5a3a2d")  # Orange
        self.status_tree.tag_configure("neutral", background="")  # Default

        self.logger.info("Status table update complete")

        # Force visual update
        self.status_tree.update_idletasks()

    def _get_file_location(self, item: ReviewItem) -> Optional[Dict[str, str]]:
        """Determine where a compartment file is located."""
        try:
            # Check if file_manager is available through app
            if not hasattr(self.app, "file_manager"):
                return None

            file_manager = self.app.file_manager
            hole_id = item.hole_id
            depth = item.compartment_depth

            # Check local temp_review first
            temp_review_path = file_manager.get_hole_dir("temp_review", hole_id)
            if os.path.exists(item.image_path) and temp_review_path in item.image_path:
                return {
                    "display_name": "Local Temp Review",
                    "tag": "local_temp",
                    "path": temp_review_path,
                }

            # Check shared review folder
            shared_review = file_manager.get_shared_path(
                "review_compartments", create_if_missing=False
            )
            if shared_review:
                project_code = hole_id[:2].upper()
                shared_review_hole = shared_review / project_code / hole_id
                if shared_review_hole.exists():
                    # Check for compartment files
                    for file in shared_review_hole.iterdir():
                        if f"CC_{depth:03d}" in file.name:
                            return {
                                "display_name": "Shared Review",
                                "tag": "shared_review",
                                "path": str(shared_review_hole),
                            }

            # Check shared approved folder
            shared_approved = file_manager.get_shared_path(
                "approved_compartments", create_if_missing=False
            )
            if shared_approved:
                project_code = hole_id[:2].upper()
                shared_approved_hole = shared_approved / project_code / hole_id
                if shared_approved_hole.exists():
                    # Check for compartment files
                    for file in shared_approved_hole.iterdir():
                        if f"CC_{depth:03d}" in file.name:
                            return {
                                "display_name": "Shared Approved",
                                "tag": "shared_approved",
                                "path": str(shared_approved_hole),
                            }

            return {"display_name": "Unknown", "tag": "neutral", "path": ""}

        except Exception as e:
            self.logger.error(f"Error determining file location: {e}")
            return None

    def _create_canvas_area(
        self, hole_id: str, review_items: List[ReviewItem], parent=None
    ):
        """Create the grid canvas area."""
        # Use provided parent or create frame
        canvas_frame = parent if parent else ttk.Frame(self.review_window)
        if not parent:
            canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create grid canvas
        self.grid_canvas = QAQCGridCanvas(canvas_frame, self.gui_manager)
        self.grid_canvas.logger = self.logger  # Add logger reference

        # Pass scanner reference if available for register status lookup
        if hasattr(self.app, "qaqc_manager") and hasattr(
            self.app.qaqc_manager, "scanner"
        ):
            self.grid_canvas.scanner = self.app.qaqc_manager.scanner

        # Load items
        self.grid_canvas.load_items(review_items, hole_id)

        # Ensure the canvas updates the status table when changes occur
        # The grid canvas already calls self.parent.master.update_status_table()
        # after actions, which will find our update_status_table method

    def _create_status_table(self, parent):
        """Create status table on the right side."""
        # ===================================================
        # INSERT: New method for vertical status table
        # Create a frame for the status table
        self.table_frame = ttk.LabelFrame(
            parent, text="Register Update Preview", padding=5
        )
        self.table_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))

        # Create treeview for status table
        self.status_tree = ttk.Treeview(
            self.table_frame,
            columns=("status", "action"),
            height=25,  # Taller for vertical layout
            show="tree headings",
        )

        # Configure columns for narrower width
        self.status_tree.heading("#0", text="Depth")
        self.status_tree.heading("status", text="Status")
        self.status_tree.heading("action", text="Action")

        self.status_tree.column("#0", width=80)
        self.status_tree.column("status", width=150)
        self.status_tree.column("action", width=80)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.table_frame, command=self.status_tree.yview)
        self.status_tree.configure(yscrollcommand=scrollbar.set)

        self.status_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        # ===================================================

    def _apply_duplicate_resolution(self):
        """Apply duplicate resolution and move to normal review."""
        # ===================================================
        # Validate duplicates are resolved
        is_valid, errors = self.grid_canvas.validate_duplicate_resolution()

        if not is_valid:
            DialogHelper.show_message(
                self.review_window,
                "Validation Error",
                "Please resolve all duplicates:\n\n" + "\n".join(errors),
                message_type="error",
            )
            return

        # Mark duplicates as resolved
        self.grid_canvas.duplicates_resolved = True

        # Hide duplicate frame, show normal actions
        self.duplicate_frame.pack_forget()

        # Reload in standard grid layout
        self.grid_canvas.layout_mode = "grid"
        self.grid_canvas.load_items(
            self.grid_canvas.review_items, self.grid_canvas.hole_id
        )

        # Update status
        self._update_status()
        # ===================================================

    # Update load_items to show/hide duplicate button
    def _update_duplicate_controls(self):
        """Show/hide duplicate resolution controls based on state."""
        # ===================================================
        # INSERT: New method
        if hasattr(self, "duplicate_frame") and hasattr(
            self.grid_canvas, "has_duplicates"
        ):
            if (
                self.grid_canvas.has_duplicates
                and not self.grid_canvas.duplicates_resolved
            ):
                self.duplicate_frame.pack(
                    side=tk.LEFT, padx=20, after=self.mode_buttons["bad_image"].master
                )
            else:
                self.duplicate_frame.pack_forget()
        # ===================================================

    def _set_mode(self, mode: str):
        """Set the current interaction mode."""
        self.grid_canvas.set_mode(mode)

        # Update button states
        for m, btn in self.mode_buttons.items():
            if m == mode:
                btn.configure(state="disabled")
            else:
                btn.configure(state="normal")

        self._update_status()

    def _clear_selection(self):
        """Clear current selection."""
        self.grid_canvas._clear_selection()
        self._update_status()

    def _update_status(self):
        """Update status bar."""
        counts = self.grid_canvas.get_status_counts()

        status = (
            f"Total: {counts['total']} | "
            f"Wet: {counts['wet']} | "
            f"Dry: {counts['dry']} | "
            f"Delete: {counts['delete']} | "
            f"Unclassified: {counts['unclassified']}"
        )

        if counts["selected"] > 0:
            status += f" | Selected: {counts['selected']}"

        status += f" | Mode: {self.grid_canvas.current_mode.upper()}"

        self.status_label.configure(text=status)

    def _setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts."""
        self.review_window.bind("s", lambda e: self._set_mode("select"))
        self.review_window.bind("w", lambda e: self._set_mode("wet"))
        self.review_window.bind("d", lambda e: self._set_mode("dry"))
        self.review_window.bind("x", lambda e: self._set_mode("delete"))
        self.review_window.bind("<Control-a>", lambda e: self._select_all())
        self.review_window.bind("<Escape>", lambda e: self._clear_selection())

    def _select_all(self):
        """Select all items."""
        for row, col in self.grid_canvas.cells:
            self.grid_canvas._select_cell(row, col)
        self._update_status()

    def _toggle_show_classified(self):
        """Toggle showing/hiding classified items."""
        self.grid_canvas.toggle_show_classified()

        # Update button text
        if self.grid_canvas.show_classified:
            self.toggle_classified_btn.configure(text="Hide Classified")
        else:
            self.toggle_classified_btn.configure(text="Show Classified")

        self._update_status()

    def _save_and_next(self):
        """Save classifications and proceed to next hole or close."""
        # Get current hole ID before moving forward
        current_hole = (
            self.hole_queue[self.current_hole_index] if self.hole_queue else None
        )

        # Check if this is the LAST hole in the queue
        is_last_hole = not self.hole_queue or self.current_hole_index >= len(self.hole_queue) - 1
        
        # Log unclassified count for reference (no prompt - user can see status bar)
        counts = self.grid_canvas.get_status_counts()
        if counts["unclassified"] > 0:
            self.logger.info(
                f"Saving {current_hole} with {counts['unclassified']} unclassified items "
                f"(will remain in review folder)"
            )

        # Trigger callback to save current hole
        if self.close_callback and current_hole:
            self.close_callback(hole_id=current_hole, cancelled=False)

        # Clean up memory from current hole before loading next
        self._cleanup_previous_hole_memory()

        # Check if there are more holes in queue
        if not is_last_hole:
            # Move to next hole
            self.current_hole_index += 1
            next_hole_id = self.hole_queue[self.current_hole_index]

            self.logger.info(
                f"Loading next hole: {next_hole_id} "
                f"({self.current_hole_index + 1}/{len(self.hole_queue)})"
            )

            # Lazy load next hole's data using processor
            if self.hole_processor:
                review_items = self.hole_processor(next_hole_id)
            else:
                self.logger.error("No hole processor available for lazy loading!")
                review_items = []

            # Reload window with next hole
            self._reload_hole_data(next_hole_id, review_items)

            # Update button labels for new position
            self._update_button_labels()
        else:
            # No more holes, close window
            self.logger.info("All holes reviewed, closing window")
            self.review_window.destroy()

    def _close_without_save(self):
        """Close the review window without saving current hole, but keep previous work."""
        # Don't call callback - we're stopping early
        self.logger.info("User closed review early - previous holes are saved")

        # Clean up memory
        self._cleanup_previous_hole_memory()

        # Close window
        self.review_window.destroy()

    def _on_close(self):
        """Handle window close."""
        response = DialogHelper.confirm_dialog(
            self.review_window,
            "Cancel Review",
            "Are you sure you want to cancel? Unsaved changes will be lost.",
        )
        if response:
            if self.close_callback:
                current_hole = (
                    self.hole_queue[self.current_hole_index]
                    if self.hole_queue
                    else None
                )
                self.close_callback(hole_id=current_hole, cancelled=True)
            self.review_window.destroy()

    def set_close_callback(
        self, callback
    ):  # TODO - check that closing will actually close!
        """Set callback for window close."""
        self.close_callback = callback

    def get_reviewed_items(self) -> List[ReviewItem]:
        """Get reviewed items."""
        reviewed, _ = self.grid_canvas.get_results()
        return reviewed

    def get_unreviewed_items(self) -> List[ReviewItem]:
        """Get unreviewed items."""
        _, unreviewed = self.grid_canvas.get_results()
        return unreviewed

    def _rotate_images(self):
        """Rotate all images 90 degrees clockwise."""
        # ===================================================
        # REPLACE: Better rotation handling with aspect ratio preservation
        # Rotate and recalculate optimal dimensions
        current_rotation = getattr(self.grid_canvas, "image_rotation", 0)
        new_rotation = (current_rotation + 90) % 360
        self.grid_canvas.image_rotation = new_rotation

        # Swap base dimensions for 90/270 degree rotations
        if (current_rotation in [0, 180] and new_rotation in [90, 270]) or (
            current_rotation in [90, 270] and new_rotation in [0, 180]
        ):
            # Swap base dimensions
            self.grid_canvas.base_cell_width, self.grid_canvas.base_cell_height = (
                self.grid_canvas.base_cell_height,
                self.grid_canvas.base_cell_width,
            )

        # Apply scale factor
        self.grid_canvas.cell_width = int(
            self.grid_canvas.base_cell_width * self.grid_canvas.scale_factor
        )
        self.grid_canvas.cell_height = int(
            self.grid_canvas.base_cell_height * self.grid_canvas.scale_factor
        )

        # Recalculate columns
        canvas_width = self.grid_canvas.parent.winfo_width()
        self.grid_canvas.cols_per_row = max(
            1,
            (canvas_width - 2 * self.grid_canvas.padding)
            // (self.grid_canvas.cell_width + self.grid_canvas.padding),
        )

        # Reload with current classifications preserved
        self.grid_canvas.load_items(
            self.grid_canvas.review_items, self.grid_canvas.hole_id
        )
        self._update_status()

    def _set_layout(self, layout_type: str):
        """Change the grid layout type."""
        if layout_type == "grid":
            self.grid_canvas.cols_per_row = 5  # Default grid columns
        elif layout_type == "column":
            self.grid_canvas.layout_mode = "vertical"
            # self.grid_canvas.cols_per_row = 1  # Single column

        # Reload with current classifications preserved
        self.grid_canvas.load_items(
            self.grid_canvas.review_items, self.grid_canvas.hole_id
        )
        self._update_status()

    def _scale_images(self, scale_factor: float):
        """Scale images by the given factor."""
        # ===================================================
        # REPLACE: Better scaling with aspect ratio preservation
        # Update scale
        new_scale = self.grid_canvas.scale_factor * scale_factor
        new_scale = max(0.3, min(2.0, new_scale))

        if new_scale == self.grid_canvas.scale_factor:
            return  # No change

        self.grid_canvas.scale_factor = new_scale

        # Update cell dimensions maintaining aspect ratio
        self.grid_canvas.cell_width = int(self.grid_canvas.base_cell_width * new_scale)
        self.grid_canvas.cell_height = int(
            self.grid_canvas.base_cell_height * new_scale
        )

        # Recalculate columns
        canvas_width = self.grid_canvas.parent.winfo_width()
        self.grid_canvas.cols_per_row = max(
            1,
            (canvas_width - 2 * self.grid_canvas.padding)
            // (self.grid_canvas.cell_width + self.grid_canvas.padding),
        )

        # Reload with current classifications preserved
        self.grid_canvas.load_items(
            self.grid_canvas.review_items, self.grid_canvas.hole_id
        )
        self._update_status()
        # ===================================================

    def _reset_all(self):
        """Reset all classifications after confirmation."""
        response = DialogHelper.confirm_dialog(
            self.review_window,
            "Reset All Classifications",
            "This will clear all classifications and bring back all images. Continue?",
        )
        if response:
            # Reset all states
            for state in self.grid_canvas.item_states.values():
                state["moisture"] = None
                state["delete"] = False
                state["bad_image"] = False

            # Reset duplicate resolution flag if any
            if hasattr(self.grid_canvas, "duplicates_resolved"):
                self.grid_canvas.duplicates_resolved = False

            # Reload the grid with reset_states=True (full reset)
            self.grid_canvas.load_items(
                self.grid_canvas.review_items,
                self.grid_canvas.hole_id,
                reset_states=True,
            )

            # Update status
            self._update_status()
            self.update_status_table()


# ============================================================================
# MAIN QAQC MANAGER
# ============================================================================


class QAQCManager:
    """Main coordinator for quality assurance and quality control."""

    def __init__(
        self,
        file_manager,
        translator_func,
        config_manager,
        app,
        logger,
        register_manager,
        gui_manager=None,
    ):
        """Initialize the QAQC Manager."""
        if threading.current_thread() != threading.main_thread():
            raise RuntimeError("QAQCManager must be created on the main thread!")

        self.root = app.root
        self.file_manager = file_manager
        self.t = translator_func
        self.config_manager = config_manager
        self.app = app
        self.logger = logger
        self.register_manager = register_manager
        self.gui_manager = gui_manager
        self.stop_review_process = False

        # Initialize components
        self.scanner = QAQCScanner(
            file_manager=file_manager,
            register_manager=register_manager,
            compartment_interval=app.config["compartment_interval"],
            logger=logger,
        )

        self.processor = QAQCProcessor(
            file_manager=file_manager,
            register_manager=register_manager,
            config=app.config,
            logger=logger,
        )

        self.dialog = QAQCGridReviewDialog(
            parent=self.root, app=app, translator_func=translator_func, logger=logger
        )

        self.main_gui = app.main_gui if hasattr(app, "main_gui") else None

    def set_main_gui(self, main_gui):
        """Set reference to main GUI for status updates."""
        self.main_gui = main_gui

    def _move_shared_items_to_local(self, shared_items_by_hole: Dict[str, List[str]]):
        """Use MultiSelectReviewDialog to move selected holes from shared to local."""

        # Show progress in main GUI
        if self.main_gui:
            total_files = sum(len(files) for files in shared_items_by_hole.values())
            self.main_gui.direct_status_update(
                f"Loading review dialog for {len(shared_items_by_hole)} holes ({total_files} files)...",
                status_type="info",
            )

        # Convert string paths to Path objects
        items_by_hole_paths = {}
        for hole_id, file_paths in shared_items_by_hole.items():
            items_by_hole_paths[hole_id] = [Path(fp) for fp in file_paths]

        # Track if any items were moved
        items_moved = False

        def on_confirm(selected_items: Dict[str, List[Path]]):
            """Callback when user confirms selection and files are moved."""
            nonlocal items_moved
            if selected_items:
                items_moved = True
                self.logger.info(
                    f"Moved {len(selected_items)} holes to local review folder"
                )

        # Create and show the dialog
        dialog = MultiSelectReviewDialog(
            parent=self.root,
            items_by_hole=items_by_hole_paths,
            on_confirm=on_confirm,
            max_selection=10,
            gui_manager=self.gui_manager,
        )

        # Show dialog and wait for completion
        dialog.show()

        # If items were moved, rescan local folder and process
        if items_moved:
            # Give filesystem a moment to settle
            import time

            time.sleep(0.5)

            # Rescan local folder
            temp_review_path = self.file_manager.dir_structure["temp_review"]
            local_review_items = self.scanner.scan_local_review_folder(temp_review_path)

            if local_review_items:
                # Process the newly moved local files
                self._process_review_queue(local_review_items)
            else:
                self.logger.warning(
                    "Files were moved but could not be found in local folder"
                )

    def start_review_process(self):
        """Entry point for the QAQC review process."""
        # Reset statistics
        self.processor.reset_stats()

        # Step 1: Load register data
        self.scanner.load_register_into_memory()

        # Step 2: Build review queue from local files
        temp_review_path = self.file_manager.dir_structure["temp_review"]
        self.logger.info(f"Scanning temp_review path: {temp_review_path}")
        self.logger.info(f"Path exists: {os.path.exists(temp_review_path)}")
        if os.path.exists(temp_review_path):
            items = list(os.listdir(temp_review_path))
            self.logger.info(f"Items in temp_review: {len(items)}")
            if items:
                self.logger.info(f"First few items: {items[:5]}")
        local_review_items = self.scanner.scan_local_review_folder(temp_review_path)
        self.logger.info(
            f"Found {len(local_review_items) if local_review_items else 0} local review items"
        )
        
        # Also scan for local approved files that duplicate shared approved
        local_approved_duplicates = self.scanner.scan_local_approved_duplicates()
        if local_approved_duplicates:
            self.logger.info(
                f"Found {sum(len(v) for v in local_approved_duplicates.values())} "
                f"local approved duplicates across {len(local_approved_duplicates)} holes"
            )
            # Merge into local_review_items
            for hole_id, items in local_approved_duplicates.items():
                if hole_id not in local_review_items:
                    local_review_items[hole_id] = []
                local_review_items[hole_id].extend(items)

        if local_review_items:
            # Process local files by hole
            self._process_review_queue(local_review_items)
        else:
            # Step 3: Check shared folders for items needing review
            self.logger.info("No local items found, scanning shared folders...")
            shared_review_items = self.scanner.scan_shared_folders_for_review()
            self.logger.info(
                f"Found {len(shared_review_items)} holes in shared folders"
            )

            if shared_review_items:
                # Log what was found
                for hole_id, paths in list(shared_review_items.items())[:3]:
                    self.logger.info(f"  - {hole_id}: {len(paths)} files")
                if len(shared_review_items) > 3:
                    self.logger.info(
                        f"  ... and {len(shared_review_items) - 3} more holes"
                    )

                # Use MultiSelectReviewDialog to move files to local
                self._move_shared_items_to_local(shared_review_items)
            else:
                self.logger.info("No shared items found either")
                self._show_nothing_to_review_message()

    def _select_holes_for_processing(
        self, shared_items_by_hole: Dict[str, List[str]]
    ) -> List[str]:
        """Let user select which holes to process using MultiSelectReviewDialog."""

        # Convert string paths to Path objects for the dialog
        items_by_hole_paths = {}
        for hole_id, file_paths in shared_items_by_hole.items():
            items_by_hole_paths[hole_id] = [Path(fp) for fp in file_paths]

        # Track which holes were selected
        selected_holes = []

        def on_confirm(selected_items: Dict[str, List[Path]]):
            """Callback when user confirms selection."""
            selected_holes.extend(selected_items.keys())

        # Create and show the dialog
        dialog = MultiSelectReviewDialog(
            parent=self.root,
            items_by_hole=items_by_hole_paths,
            on_confirm=on_confirm,
            max_selection=10,  # Adjust as needed
            gui_manager=self.gui_manager,
        )

        dialog.show()

        return selected_holes

    def _show_pre_review_analysis(self, hole_id: str, analysis: Dict) -> Dict[str, Any]:
        """
        Log analysis results and auto-continue with smart defaults.
        No user interaction - just logging and status updates.

        Returns:
            Dict with user choices:
            - continue: bool - whether to continue with review
            - move_reviewed: bool - whether to move reviewed files to approved
            - move_unreviewed: bool - whether to move unreviewed files to review
        """
        # Summary
        total_depths = len(analysis["compartments_by_depth"])
        duplicate_count = len(analysis["duplicates"])
        needs_review_count = len(analysis["needs_review"])

        # Misplaced files
        reviewed_misplaced = len(
            analysis["misplaced_files"]["reviewed_in_review_folder"]
        )
        unreviewed_misplaced = len(
            analysis["misplaced_files"]["unreviewed_in_approved_folder"]
        )

        # Log the analysis
        self.logger.info(f"Analysis for {hole_id}:")
        self.logger.info(f"  • Found {total_depths} depths with compartments")
        self.logger.info(f"  • {duplicate_count} depths have duplicates")
        self.logger.info(f"  • {needs_review_count} depths need review")

        if reviewed_misplaced > 0:
            self.logger.info(
                f"  • {reviewed_misplaced} reviewed files will be auto-moved to approved"
            )
        if unreviewed_misplaced > 0:
            self.logger.info(
                f"  • {unreviewed_misplaced} unreviewed files will be auto-moved to review"
            )

        # Always continue with smart defaults
        result = {
            "continue": True,  # Always continue
            "move_reviewed": reviewed_misplaced > 0,  # Auto-move if found
            "move_unreviewed": unreviewed_misplaced > 0,  # Auto-move if found
        }

        return result

    def _move_misplaced_files(
        self, file_infos: List[Dict], target_type: str
    ) -> Dict[str, int]:
        """
        Move misplaced files to correct locations.

        Args:
            file_infos: List of file info dicts from analysis
            target_type: "approved" or "review"

        Returns:
            Dict with counts of moved and failed
        """
        moved = 0
        failed = 0

        # Determine target path
        if target_type == "approved":
            target_base = self.file_manager.get_shared_path("approved_compartments")
        else:
            target_base = self.file_manager.get_shared_path("review_compartments")

        if not target_base:
            self.logger.error(f"Target path not configured for {target_type}")
            return {"moved": 0, "failed": len(file_infos)}

        # Process each file
        for file_info in file_infos:
            try:
                source_path = file_info["file_path"]
                filename = file_info["filename"]

                # Extract hole_id from filename
                match = self.scanner.COMPARTMENT_FILE_PATTERN.match(filename)
                if not match:
                    self.logger.error(f"Could not parse filename: {filename}")
                    failed += 1
                    continue

                hole_id = match.group(1)
                project_code = hole_id[:2].upper()

                # Build target path
                target_dir = target_base / project_code / hole_id
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path = target_dir / filename

                # Use file manager's copy method to preserve metadata
                if self.file_manager.copy_with_metadata(source_path, str(target_path)):
                    # Verify copy before deleting
                    if target_path.exists():
                        try:
                            os.remove(source_path)
                            moved += 1
                            self.logger.info(
                                f"Moved {filename} to {target_type} folder"
                            )
                        except Exception as e:
                            self.logger.error(
                                f"Could not delete source after copy: {e}"
                            )
                            # Still count as moved since copy succeeded
                            moved += 1
                    else:
                        failed += 1
                        self.logger.error(
                            f"Target file not found after copy: {target_path}"
                        )
                else:
                    failed += 1
                    self.logger.error(f"Failed to copy {filename}")

            except Exception as e:
                self.logger.error(f"Error moving file {file_info.get('filename')}: {e}")
                failed += 1

        return {"moved": moved, "failed": failed}

    def _process_review_queue(self, items_by_hole: Dict[str, List[ReviewItem]]):
        """Process review items with all holes queued in a single window."""
        total_holes = len(items_by_hole)
        self.stop_review_process = False

        # Store hole IDs to process
        hole_ids = list(items_by_hole.keys())

        # Create processor function for lazy loading
        def process_hole(hole_id: str) -> List[ReviewItem]:
            """Process a single hole and return review items."""
            if self.stop_review_process:
                return []

            self.logger.info(f"Processing hole: {hole_id}")

            # Get initial review items
            review_items = items_by_hole.get(hole_id, [])

            # Perform comprehensive scan
            all_compartments = self.scanner.scan_all_compartments_for_hole(hole_id)
            if all_compartments:
                review_items = all_compartments
                self.logger.info(
                    f"Using {len(all_compartments)} items from comprehensive scan"
                )

            # Perform analysis
            analysis = self.scanner.analyze_compartment_files(hole_id)

            # Get user choices for misplaced files
            user_choices = self._show_pre_review_analysis(hole_id, analysis)

            if not user_choices["continue"]:
                self.logger.info(f"User skipped review for {hole_id}")
                return []

            # Handle file movements
            if (
                user_choices["move_reviewed"]
                and analysis["misplaced_files"]["reviewed_in_review_folder"]
            ):
                self.logger.info("Moving reviewed files to approved folder...")
                move_results = self._move_misplaced_files(
                    analysis["misplaced_files"]["reviewed_in_review_folder"], "approved"
                )
                if move_results["moved"] > 0 and self.main_gui:
                    self.main_gui.direct_status_update(
                        f"Moved {move_results['moved']} files to approved",
                        status_type="success",
                    )

            if (
                user_choices["move_unreviewed"]
                and analysis["misplaced_files"]["unreviewed_in_approved_folder"]
            ):
                self.logger.info("Moving unreviewed files to review folder...")
                move_results = self._move_misplaced_files(
                    analysis["misplaced_files"]["unreviewed_in_approved_folder"],
                    "review",
                )
                if move_results["moved"] > 0 and self.main_gui:
                    self.main_gui.direct_status_update(
                        f"Moved {move_results['moved']} files to review",
                        status_type="success",
                    )

            # Return processed items
            if not review_items:
                review_items.sort(key=lambda x: x.depth_to)

            return review_items

        # Set callback for when each hole completes
        self.dialog.set_close_callback(self._on_hole_review_complete)

        # Queue all holes into dialog (lazy loading via processor)
        self.dialog.load_hole_queue(hole_ids, process_hole)

        # Wait for window to close (only once for all holes)
        self.root.wait_window(self.dialog.review_window)

    def _on_hole_review_complete(self, hole_id: str = None, cancelled: bool = False):
        """Handle completion of review for a single hole."""
        if cancelled:
            self.stop_review_process = True
            self.logger.info(f"Review cancelled for hole {hole_id or 'unknown'}")
            return

        if not hole_id:
            self.logger.warning("No hole_id provided to completion callback")
            return

        # Get reviewed and unreviewed items
        reviewed_items = self.dialog.get_reviewed_items()
        unreviewed_items = self.dialog.get_unreviewed_items()

        # Process reviewed items
        if reviewed_items:
            self.processor.batch_process_reviewed_items(reviewed_items)

            # Clean up successfully processed temp files
            temp_paths = []
            for item in reviewed_items:
                if item.is_reviewed and item.image_path:
                    # Only add if it's in temp_review folder
                    if (
                        "Compartment Images for Review" in item.image_path
                        or "temp_review" in item.image_path
                        or "_temp" in item.image_path
                        or "_new" in item.image_path
                    ):
                        temp_paths.append(item.image_path)

            # Use file manager to clean up
            if temp_paths:
                self.file_manager.cleanup_temp_compartments(hole_id, temp_paths)
                self.logger.info(f"Cleaned up {len(temp_paths)} processed temp files")
            
            # Track deleted pending UIDs and cleanup orphan pending originals
            self._cleanup_orphan_pending_originals(hole_id, reviewed_items, unreviewed_items)

        # Free memory
        for item in reviewed_items + unreviewed_items:
            item.unload_image()

        # Update status with completion info
        if self.main_gui:
            if reviewed_items:
                self.main_gui.direct_status_update(
                    f"Successfully moved {len(reviewed_items)} files to approved folder",
                    status_type="success",
                )

            # Note about unreviewed items (they'll be handled by cloud sync)
            if unreviewed_items:
                self.logger.info(
                    f"{len(unreviewed_items)} unreviewed items will be handled by cloud sync"
                )

    def _cleanup_orphan_pending_originals(
        self, 
        hole_id: str, 
        reviewed_items: List[ReviewItem], 
        unreviewed_items: List[ReviewItem]
    ):
        """
        Clean up pending originals when all their compartments have been deleted.
        
        For each pending UID:
        - If ANY compartments were approved -> keep original (will be finalized later)
        - If ALL compartments were deleted -> delete the pending original
        """
        import re
        from collections import defaultdict
        
        # Pattern to extract UID from pending filename
        pending_pattern = re.compile(r"_pending_([a-f0-9]+)\.", re.IGNORECASE)
        
        # Safety check: Don't cleanup if there are unreviewed pending items
        # (they haven't been classified yet - we don't know if user will keep or delete them)
        unreviewed_pending_uids = set()
        for item in unreviewed_items:
            if item.filename:
                match = pending_pattern.search(item.filename)
                if match:
                    unreviewed_pending_uids.add(match.group(1).lower())
        
        if unreviewed_pending_uids:
            self.logger.info(
                f"Skipping pending original cleanup - {len(unreviewed_pending_uids)} UIDs "
                f"still have unreviewed compartments: {unreviewed_pending_uids}"
            )
            return
        
        # Track UIDs and their fate (only from reviewed items now)
        uid_approved = set()  # UIDs with at least one approved compartment
        uid_deleted = set()   # UIDs where compartments were deleted
        
        # Scan ONLY reviewed items to track UID status
        for item in reviewed_items:
            if not item.filename:
                continue
                
            match = pending_pattern.search(item.filename)
            if not match:
                continue
                
            uid = match.group(1).lower()
            # Check if this pending compartment was approved (has Wet/Dry suffix)
            if item.is_reviewed and item.moisture in ["wet", "dry", "Wet", "Dry"]:
                uid_approved.add(uid)
            elif item.is_reviewed and item.quality == "Missing":
                # Explicitly deleted by user
                uid_deleted.add(uid)
        
        # Find UIDs where ALL compartments were deleted (none approved)
        # uid_deleted is now a set, not a dict
        orphan_uids = uid_deleted - uid_approved
        
        for uid in orphan_uids:
            self.logger.debug(f"UID {uid}: deleted, none approved - orphan original")
        
        if not orphan_uids:
            return
        
        # Clean up orphan pending originals
        pending_originals_dir = self.file_manager.dir_structure.get("pending_originals")
        if not pending_originals_dir:
            self.logger.warning("pending_originals directory not configured")
            return
        
        project_code = hole_id[:2].upper()
        hole_pending_dir = Path(pending_originals_dir) / project_code / hole_id
        
        if not hole_pending_dir.exists():
            self.logger.debug(f"No pending originals directory for {hole_id}")
            return
        
        # Find and delete orphan originals
        deleted_count = 0
        for file_path in hole_pending_dir.iterdir():
            if not file_path.is_file():
                continue
            
            filename = file_path.name
            # Original pattern: KM0335_0-20_Original_45ecbec5.JPG
            orig_match = re.search(r"_Original_([a-f0-9]+)\.", filename, re.IGNORECASE)
            if not orig_match:
                continue
            
            orig_uid = orig_match.group(1).lower()
            if orig_uid in orphan_uids:
                try:
                    file_path.unlink()
                    deleted_count += 1
                    self.logger.info(f"Deleted orphan pending original: {filename}")
                except Exception as e:
                    self.logger.error(f"Failed to delete orphan original {filename}: {e}")
        
        if deleted_count > 0:
            self.logger.info(f"Cleaned up {deleted_count} orphan pending originals for {hole_id}")
            
            # Clean up empty directories
            try:
                if hole_pending_dir.exists() and not any(hole_pending_dir.iterdir()):
                    hole_pending_dir.rmdir()
                    project_dir = hole_pending_dir.parent
                    if project_dir.exists() and not any(project_dir.iterdir()):
                        project_dir.rmdir()
            except Exception as e:
                self.logger.debug(f"Could not clean up empty pending directories: {e}")

    def _show_nothing_to_review_message(self):
        """Show message when there's nothing to review."""
        DialogHelper.show_message(
            self.root,
            self.t("Review Complete"),
            self.t("No images to review and all files are synchronized."),
            message_type="info",
        )

    def _show_final_summary(self):
        """Show final processing summary and update main GUI."""
        summary_message = self.processor.get_summary_message()

        # Log summary
        self.logger.info(summary_message)

        # Update main GUI with final summary
        if self.main_gui:
            # Parse the summary to show success/error counts appropriately
            if (
                self.processor.stats.get("upload_failed", 0) > 0
                or self.processor.stats.get("register_failed", 0) > 0
            ):
                self.main_gui.direct_status_update(
                    f"QAQC Complete with errors - Check log for details",
                    status_type="warning",
                )
            else:
                self.main_gui.direct_status_update(
                    f"QAQC Complete - {self.processor.stats['processed']} compartments processed",
                    status_type="success",
                )


# Export main classes
__all__ = ["QAQCManager", "ReviewItem", "QAQCConstants"]
