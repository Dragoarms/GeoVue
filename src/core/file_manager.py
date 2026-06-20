# core/file_manager.py

"""
Manages file operations for GeoVue.

Handles directory creation, file naming conventions, and saving operations.

Directory Structure:


The local and shared folder paths are configured on first run in the GeoVue/settings.json file (appdata folder).
==================================================================================

LOCAL DIRECTORY FOR LOCAL PROCESSING AND BACKUPS:


[Local Path folder]/
├── 📁 Images to Process/
├── 📁 Processed Original Images/
│   ├── 📁 Approved Originals/
│   │   └── 📁 [PROJECT CODE]/
│   │       └── 📁 [HOLE_ID]/
│   │           └── [HOLE_ID]_[FROM]-[TO]_Original.jpg
│   └── 📁 Rejected Originals/
│      └── 📁 [PROJECT CODE]/
│           └── 📁 [HOLE_ID]/
│               └── [HOLE_ID]_[FROM]-[TO]_Original.jpg
├── 📁 Extracted Compartment Images/
│   ├── 📁 [Approved Compartment Images]/
│   |   └── 📁 [PROJECT CODE]/
|   |       └── 📁 [HOLE_ID]/
|   |            └──  [HOLE_ID]_CC_[Depth padded to three digits]_[Wet/Dry suffix].[ext]
|   └── 📁 [Compartment Images for Review]/
│      └── 📁 [PROJECT CODE]/
|           └── 📁 [HOLE_ID]/
|                └──  [HOLE_ID]_CC_[Depth padded to three digits]_{temp/new / or no suffix}.[ext]
├── 📁 Drillhole Traces/
│      └── 📁 [PROJECT CODE]/
|           └── 📁 [HOLE_ID]/
|               └── [HOLE_ID]_trace.[ext]

==================================================================================
SHARED DIRECTORY STRUCTURE:
[Shared Folder]/
├── 📁 Chip Tray Register/
│   ├── Chip_Tray_Register.xlsx
│   └── 📁 Register Data (Do not edit)/
│       ├── compartment_register.json
│       ├── original_images_register.json
│       └── compartment_reviews.json
├── 📁 Images to Process/
├── 📁 Processed Original Images/
│   ├── 📁 Approved Originals/
│   │   └── 📁 [PROJECT CODE]/
│   │       └── 📁 [HOLE_ID]/
│   │           └── [HOLE_ID]_[FROM]-[TO]_Original.jpg [HOLE_ID]_[FROM]-[TO]_Original.jpg
│   └── 📁 Rejected Originals/
│       └── 📁 [PROJECT CODE]/
│           └── 📁 [HOLE_ID]/
│               └── [HOLE_ID]_[FROM]-[TO]_Original.jpg
├── 📁 Extracted Compartment Images/
│   ├── 📁 [Approved Compartment Images]/
│   |   └── 📁 [PROJECT CODE]/
|   |       └── 📁 [HOLE_ID]/
|   |           | ├── [HOLE_ID]_CC_001.[ext]
|   |           | └── [HOLE_ID]_CC_002.[ext]
│   |           └── 📁 With_Data/
│   |                └── [HOLE_ID]_CC_001_Data.[ext]
|   └── 📁 [Compartment Images for Review]/
│       └── 📁 [PROJECT CODE]/
|           └── 📁 [HOLE_ID]/
|                ├── [HOLE_ID]_CC_001_temp.[ext]
|                └── [HOLE_ID]_CC_002_temp.[ext]
├── 📁 Drillhole Traces/
    └── 📁 [PROJECT CODE]/
        └── 📁 [HOLE_ID]/
            └── [HOLE_ID]_trace.[ext]


"""

import os
import re
import cv2
import numpy as np
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import traceback
import uuid
import piexif
from pillow_heif import register_heif_opener

register_heif_opener()
from PIL import Image as PILImage
from PIL.PngImagePlugin import PngInfo


class FileManager:
    """
    Manages file operations for the Chip Tray Extractor.

    Handles directory creation, file naming conventions, and saving operations
    for both local and shared (OneDrive) folders.
    """

    # Define folder structure constants
    FOLDER_NAMES = {
        "register": "Chip Tray Register",
        "images": "Images to Process",
        "processed": "Processed Original Images",
        "compartments": "Extracted Compartment Images",
        "traces": "Drillhole Traces",
        "datasets": "Drillhole Datasets",
        "televiewer_datasets": "Televiewer Datasets",
        "cross_sections": "Cross Sections",
        "debugging": "Debugging",
    }

    SUBFOLDER_NAMES = {
        "approved_originals": "Approved Originals",
        "rejected_originals": "Rejected Originals",
        "pending_originals": "Pending Originals",
        "approved_compartments": "Approved Compartment Images",
        "review_compartments": "Compartment Images for Review",
        "blur": "Blur Analysis",
        "debug": "Debug Images",
        "register_data": "Register Data (Do not edit)",  # folder containing the register datasets and automatic excel file.
        "with_data": "With_Data",  # Individual chip compartments with plots / visualisations / datasets
    }

    EXCEL_REGISTER_NAME = "Chip_Tray_Register.xlsx"

    def __init__(self, base_dir: str = None, config_manager=None):
        """
        Initialize the file manager with a base directory.

        Args:
            base_dir: Base directory for all outputs
            config_manager: ConfigManager instance for accessing configuration
        """
        # Logger setup first
        self.logger = logging.getLogger(__name__)

        # Store config manager reference
        self.config_manager = config_manager

        # Set up local base directory
        self.base_dir = (
            Path(base_dir) if base_dir else Path("C:/GeoVue Chip Tray Photos")
        )

        # NEW: Initialize shared folder paths

        self.shared_base_dir = None
        self.shared_paths = {}

        # Use constants for folder names

        self.dir_structure = {
            "local_output_folder": self.base_dir,
            "images_to_process": self.base_dir / self.FOLDER_NAMES["images"],
            "processed_originals": self.base_dir / self.FOLDER_NAMES["processed"],
            "approved_originals": self.base_dir
            / self.FOLDER_NAMES["processed"]
            / self.SUBFOLDER_NAMES["approved_originals"],
            "rejected_originals": self.base_dir
            / self.FOLDER_NAMES["processed"]
            / self.SUBFOLDER_NAMES["rejected_originals"],
            "pending_originals": self.base_dir
            / self.FOLDER_NAMES["processed"]
            / self.SUBFOLDER_NAMES["pending_originals"],
            "chip_compartments": self.base_dir / self.FOLDER_NAMES["compartments"],
            "approved_compartments": self.base_dir
            / self.FOLDER_NAMES["compartments"]
            / self.SUBFOLDER_NAMES["approved_compartments"],
            "temp_review": self.base_dir
            / self.FOLDER_NAMES["compartments"]
            / self.SUBFOLDER_NAMES["review_compartments"],
            # Add alias for compatibility with duplicate_handler
            "review_compartments": self.base_dir
            / self.FOLDER_NAMES["compartments"]
            / self.SUBFOLDER_NAMES["review_compartments"],
            "drill_traces": self.base_dir / self.FOLDER_NAMES["traces"],
            "datasets": self.base_dir / self.FOLDER_NAMES["datasets"],
            "televiewer_datasets": self.base_dir
            / self.FOLDER_NAMES["televiewer_datasets"],
            "debugging": self.base_dir / self.FOLDER_NAMES["debugging"],
            "blur_analysis": self.base_dir
            / self.FOLDER_NAMES["debugging"]
            / self.SUBFOLDER_NAMES["blur"],
            "debug_images": self.base_dir
            / self.FOLDER_NAMES["debugging"]
            / self.SUBFOLDER_NAMES["debug"],
        }

        # Initialize shared paths if configured

        self.initialize_shared_paths()

    def initialize_shared_paths(self, shared_base_dir: str = None) -> None:
        """
        Initialize or update shared folder paths.

        Args:
            shared_base_dir: Base directory for shared folders (OneDrive)
                        If None, attempts to load from config
        """
        # Get shared base directory from parameter or config
        if shared_base_dir:
            self.shared_base_dir = Path(shared_base_dir)
        elif self.config_manager:
            shared_path = self.config_manager.get("shared_folder_path")
            if shared_path and os.path.exists(shared_path):
                self.shared_base_dir = Path(shared_path)

        # Build shared paths structure if we have a base directory
        if self.shared_base_dir:
            self.shared_paths = {
                "register": self.shared_base_dir / self.FOLDER_NAMES["register"],
                "register_excel": self.shared_base_dir
                / self.FOLDER_NAMES["register"]
                / self.EXCEL_REGISTER_NAME,
                "register_data": self.shared_base_dir
                / self.FOLDER_NAMES["register"]
                / self.SUBFOLDER_NAMES["register_data"],
                # Default drillhole_data.csv is now stored under 'Drillhole Datasets'
                "drillhole_data_csv": self.shared_base_dir
                / self.FOLDER_NAMES["datasets"]
                / "drillhole_data.csv",
                "images_to_process": self.shared_base_dir / self.FOLDER_NAMES["images"],
                "processed_originals": self.shared_base_dir
                / self.FOLDER_NAMES["processed"],
                "approved_originals": self.shared_base_dir
                / self.FOLDER_NAMES["processed"]
                / self.SUBFOLDER_NAMES["approved_originals"],
                "rejected_originals": self.shared_base_dir
                / self.FOLDER_NAMES["processed"]
                / self.SUBFOLDER_NAMES["rejected_originals"],
                "compartments": self.shared_base_dir
                / self.FOLDER_NAMES["compartments"],
                "approved_compartments": self.shared_base_dir
                / self.FOLDER_NAMES["compartments"]
                / self.SUBFOLDER_NAMES["approved_compartments"],
                "review_compartments": self.shared_base_dir
                / self.FOLDER_NAMES["compartments"]
                / self.SUBFOLDER_NAMES["review_compartments"],
                "drill_traces": self.shared_base_dir / self.FOLDER_NAMES["traces"],
                "datasets": self.shared_base_dir / self.FOLDER_NAMES["datasets"],
                "televiewer_datasets": self.shared_base_dir
                / self.FOLDER_NAMES["televiewer_datasets"],
                "cross_sections": (
                    Path(self.config_manager.get("shared_folder_cross_sections"))
                    if (
                        self.config_manager
                        and self.config_manager.get("shared_folder_cross_sections")
                    )
                    else (self.shared_base_dir / self.FOLDER_NAMES["cross_sections"])
                ),
            }

            # Save individual paths to config if available
            if self.config_manager:
                self.config_manager.set(
                    "shared_folder_register_path", str(self.shared_paths["register"])
                )
                self.config_manager.set(
                    "shared_folder_register_excel_path",
                    str(self.shared_paths["register_excel"]),
                )
                self.config_manager.set(
                    "shared_folder_datasets",
                    str(self.shared_paths["datasets"]),
                )
                self.config_manager.set(
                    "shared_folder_register_data_folder",
                    str(self.shared_paths["register_data"]),
                )
                self.config_manager.set(
                    "shared_folder_processed_originals",
                    str(self.shared_paths["processed_originals"]),
                )
                self.config_manager.set(
                    "shared_folder_approved_folder",
                    str(self.shared_paths["approved_originals"]),
                )
                self.config_manager.set(
                    "shared_folder_rejected_folder",
                    str(self.shared_paths["rejected_originals"]),
                )
                self.config_manager.set(
                    "shared_folder_extracted_compartments_folder",
                    str(self.shared_paths["compartments"]),
                )
                self.config_manager.set(
                    "shared_folder_approved_compartments_folder",
                    str(self.shared_paths["approved_compartments"]),
                )
                self.config_manager.set(
                    "shared_folder_review_compartments_folder",
                    str(self.shared_paths["review_compartments"]),
                )
                self.config_manager.set(
                    "shared_folder_drill_traces", str(self.shared_paths["drill_traces"])
                )
                self.config_manager.set(
                    "shared_folder_televiewer_datasets",
                    str(self.shared_paths["televiewer_datasets"]),
                )
                # Only write default cross_sections path when user has not set one
                if not self.config_manager.get("shared_folder_cross_sections"):
                    self.config_manager.set(
                        "shared_folder_cross_sections",
                        str(self.shared_paths["cross_sections"]),
                    )

            # When no shared base dir, still allow cross_sections from config (user may have set it)
            if (
                self.config_manager
                and self.config_manager.get("shared_folder_cross_sections")
                and "cross_sections" not in self.shared_paths
            ):
                if not self.shared_paths:
                    self.shared_paths = {}
                self.shared_paths["cross_sections"] = Path(
                    self.config_manager.get("shared_folder_cross_sections")
                )

            self.logger.debug(
                f"Initialized shared paths with base: {self.shared_base_dir}"
            )

    def get_shared_path(
        self, path_key: str, create_if_missing: bool = True
    ) -> Optional[Path]:
        """
        Get a shared folder path by key.

        Args:
            path_key: Key from shared_paths dictionary
            create_if_missing: Whether to create directory if it doesn't exist

        Returns:
            Path object or None if not configured
        """
        if not self.shared_paths or path_key not in self.shared_paths:
            return None

        path = self.shared_paths[path_key]

        # Create directory if requested (not for files)
        if create_if_missing and not path_key.endswith("_excel") and not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created shared directory: {path}")
            except Exception as e:
                self.logger.error(f"Failed to create shared directory {path}: {e}")
                return None

        return path if path.exists() else None

    def get_cross_sections_path(
        self, create_if_missing: bool = False
    ) -> Optional[Path]:
        """
        Get the shared Cross Sections folder path (section PDFs for Section Tool integration).

        Returns:
            Path to the Cross Sections folder, or None if not configured.
        """
        return self.get_shared_path("cross_sections", create_if_missing=create_if_missing)

    def update_shared_path(self, path_key: str, new_path: str) -> bool:
        """
        Update a specific shared path.

        Args:
            path_key: Key to update in shared_paths
            new_path: New path value

        Returns:
            True if successful, False otherwise
        """
        try:
            if not new_path:
                self.logger.warning(f"Attempted to set empty path for {path_key}")
                return False

            new_path_obj = Path(new_path)

            # Update the path in our dictionary
            if path_key in self.shared_paths:
                self.shared_paths[path_key] = new_path_obj

                # Also update config if available
                if self.config_manager:
                    # Map internal keys to config keys
                    config_key_map = {
                        "register": "shared_folder_register_path",
                        "register_excel": "shared_folder_register_excel_path",
                        "register_data": "shared_folder_register_data_folder",
                        "processed_originals": "shared_folder_processed_originals",
                        "approved_originals": "shared_folder_approved_folder",
                        "rejected_originals": "shared_folder_rejected_folder",
                        "compartments": "shared_folder_extracted_compartments_folder",
                        "approved_compartments": "shared_folder_approved_compartments_folder",
                        "review_compartments": "shared_folder_review_compartments_folder",
                        "drill_traces": "shared_folder_drill_traces",
                        "datasets": "shared_folder_datasets",
                        "televiewer_datasets": "shared_folder_televiewer_datasets",
                        "cross_sections": "shared_folder_cross_sections",
                    }

                    if path_key in config_key_map:
                        self.config_manager.set(config_key_map[path_key], new_path)

                self.logger.debug(f"Updated shared path {path_key}: {new_path}")
                return True
            else:
                self.logger.warning(f"Unknown path key: {path_key}")
                return False

        except Exception as e:
            self.logger.error(f"Error updating shared path {path_key}: {e}")
            return False

    def prompt_for_shared_path(
        self,
        path_key: str,
        dialog_title: str,
        missing_message: str,
        is_file: bool = False,
    ) -> Optional[str]:
        """
        Prompt user to select a shared folder/file path.

        Args:
            path_key: Key for the path in shared_paths
            dialog_title: Title for the file dialog
            missing_message: Message to show if path doesn't exist
            is_file: Whether selecting a file (True) or folder (False)

        Returns:
            Selected path or None if cancelled
        """
        from gui.dialog_helper import DialogHelper
        import tkinter as tk
        from tkinter import filedialog

        # Get current path if exists
        current_path = self.shared_paths.get(path_key) if self.shared_paths else None
        initial_dir = (
            str(current_path.parent) if current_path and current_path.exists() else None
        )

        # Create root if needed
        root = tk.Tk()
        root.withdraw()

        try:
            # Show message if path missing
            if missing_message and (not current_path or not current_path.exists()):
                result = DialogHelper.confirm_dialog(
                    root,
                    DialogHelper.t("Path Not Found"),
                    DialogHelper.t(missing_message),
                )
                if not result:
                    return None

            # Show file/folder dialog
            if is_file:
                path = filedialog.askopenfilename(
                    title=DialogHelper.t(dialog_title),
                    initialdir=initial_dir,
                    filetypes=(
                        [("Excel files", "*.xlsx"), ("All files", "*.*")]
                        if path_key == "register_excel"
                        else []
                    ),
                )
            else:
                path = filedialog.askdirectory(
                    title=DialogHelper.t(dialog_title), initialdir=initial_dir
                )

            if path:
                # Update the path
                self.update_shared_path(path_key, path)
                return path

        finally:
            root.destroy()

        return None

    def _get_local_path(
        self,
        hole_id: str,
        depth_from: float,
        depth_to: float,
        is_processed: bool = True,
        is_rejected: bool = False,
    ) -> str:
        """
        Get local path for original files based on processing status.

        Args:
            hole_id: Hole identifier
            depth_from: Starting depth (unused but kept for signature compatibility)
            depth_to: Ending depth (unused but kept for signature compatibility)
            is_processed: Whether file was processed (kept for compatibility, always True)
            is_rejected: Whether file was rejected during processing

        Returns:
            Path to appropriate local directory (approved or rejected)

        """
        # Since all handled files are "processed", use is_rejected to determine path
        if is_rejected:
            return self.get_hole_dir("rejected_originals", hole_id)
        else:
            return self.get_hole_dir("approved_originals", hole_id)

    def _verify_upload(
        self, upload_path: Union[str, Path], max_retries: int = 3
    ) -> bool:
        """
        Verify file upload completed successfully with retries.

        Args:
            upload_path: Path to verify
            max_retries: Maximum verification attempts

        Returns:
            True if file exists and accessible

        Technical rationale: Handles OneDrive sync delays
        """
        import time

        path = Path(upload_path) if isinstance(upload_path, str) else upload_path

        for attempt in range(max_retries):
            try:
                if path.exists() and path.stat().st_size > 0:
                    return True
                time.sleep(1)  # Wait for sync
            except Exception as e:
                self.logger.warning(
                    f"Upload verification attempt {attempt + 1} failed: {e}"
                )

        return False

    def rename_file_safely(self, old_path: str, new_path: str) -> bool:
        """
        Safely rename a file with error handling and backup.

        Args:
            old_path: Current file path
            new_path: New file path

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if source exists
            if not os.path.exists(old_path):
                self.logger.error(f"Source file does not exist: {old_path}")
                return False

            # Check if target already exists
            if os.path.exists(new_path):
                self.logger.error(f"Target file already exists: {new_path}")
                return False

            # Ensure target directory exists
            os.makedirs(os.path.dirname(new_path), exist_ok=True)

            # Try to rename
            try:
                os.rename(old_path, new_path)
                self.logger.info(f"Renamed file: {old_path} -> {new_path}")
                return True
            except OSError as e:
                # If rename fails, try copy and delete
                self.logger.warning(f"Direct rename failed, trying copy method: {e}")
                try:
                    # ===================================================
                    # FIXED: Use copy_with_metadata to preserve EXIF data
                    # ===================================================
                    if self.copy_with_metadata(old_path, new_path):
                        # Verify copy succeeded
                        if os.path.exists(new_path) and os.path.getsize(
                            new_path
                        ) == os.path.getsize(old_path):
                            os.remove(old_path)
                            self.logger.info(
                                f"Renamed file via copy: {old_path} -> {new_path}"
                            )
                            return True
                        else:
                            self.logger.error("Copy verification failed")
                            if os.path.exists(new_path):
                                os.remove(new_path)
                            return False
                    else:
                        self.logger.error("copy_with_metadata failed")
                        return False
                except Exception as copy_error:
                    self.logger.error(f"Copy method also failed: {copy_error}")
                    return False

        except Exception as e:
            self.logger.error(f"Error renaming file: {str(e)}")
            return False

    def extract_metadata_from_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from a filename, looking for patterns like 'BB1234_40-60_Original'.

        Args:
            filename: Path to the file or just the filename

        Returns:
            Dictionary with extracted metadata if found, None otherwise
        """
        try:
            # Get the base filename without path and extension
            base_name = os.path.splitext(os.path.basename(filename))[0]

            # Define a pattern to extract metadata from filename
            # Pattern matches formats like "BB1234_40-60_Original" or "BB1234_40-60_Original_Skipped"
            pattern = r"([A-Z]{2}\d{4})_(\d+)-(\d+)(?:_Original)?(?:_Skipped)?"

            # Try to extract metadata directly from filename
            match = re.search(pattern, base_name)
            if match:
                hole_id = match.group(1)
                depth_from = int(match.group(2))
                depth_to = int(match.group(3))

                # Check against valid prefixes if the configuration exists and prefix validation is enabled
                valid_prefixes = (
                    self.config_manager.get("valid_hole_prefixes", [])
                    if self.config_manager
                    else []
                )
                enable_prefix_validation = (
                    self.config_manager.get("enable_prefix_validation", False)
                    if self.config_manager
                    else False
                )

                # If prefix validation is enabled, check against valid prefixes
                if enable_prefix_validation and valid_prefixes:
                    prefix = hole_id[:2].upper()
                    if prefix not in valid_prefixes:
                        self.logger.warning(
                            f"Hole ID prefix {prefix} not in valid prefixes: {valid_prefixes}"
                        )
                        return None

                self.logger.info(
                    f"Extracted metadata from filename: {hole_id}, {depth_from}-{depth_to}"
                )
                return {
                    "hole_id": hole_id,
                    "depth_from": depth_from,
                    "depth_to": depth_to,
                    "confidence": 100.0,  # High confidence for filename-derived metadata
                    "from_filename": True,
                }

            return None

        except Exception as e:
            self.logger.error(
                f"Error extracting metadata from filename {filename}: {str(e)}"
            )
            return None

    def check_original_file_processed(
        self, original_filename: str
    ) -> Optional[Dict[str, Any]]:
        """
        Check if an original file has already been processed by extracting metadata from filename.

        Args:
            original_filename: Original input filename

        Returns:
            Dictionary with extracted metadata if found, None otherwise
        """
        try:
            # Get the base filename without path and extension
            base_name = os.path.splitext(os.path.basename(original_filename))[0]

            # Define a pattern to extract metadata from filename
            # Pattern matches formats like "BB1234_40-60_Original" or "BB1234_40-60_Original_Skipped"
            pattern = r"([A-Z]{2}\d{4})_(\d+)-(\d+)_Original(?:_Skipped)?"

            # Try to extract metadata directly from filename
            match = re.search(pattern, base_name)
            if match:
                hole_id = match.group(1)
                depth_from = int(match.group(2))
                depth_to = int(match.group(3))

                # Check against valid prefixes if the configuration exists and prefix validation is enabled
                valid_prefixes = getattr(self, "config", {}).get(
                    "valid_hole_prefixes", []
                )
                enable_prefix_validation = getattr(self, "config", {}).get(
                    "enable_prefix_validation", False
                )

                # If prefix validation is enabled, check against valid prefixes
                if enable_prefix_validation and valid_prefixes:
                    prefix = hole_id[:2].upper()
                    if prefix not in valid_prefixes:
                        self.logger.warning(
                            f"Hole ID prefix {prefix} not in valid prefixes: {valid_prefixes}"
                        )
                        return None

                # Determine if the file was previously skipped
                is_skipped = "_Skipped" in base_name

                self.logger.info(
                    f"Found metadata in filename: {hole_id}, {depth_from}-{depth_to}"
                )
                return {
                    "hole_id": hole_id,
                    "depth_from": depth_from,
                    "depth_to": depth_to,
                    "confidence": (
                        100.0 if not is_skipped else 90.0
                    ),  # Slightly lower confidence for skipped files
                    "from_filename": True,
                    "previously_skipped": is_skipped,
                }

            # No match found
            return None

        except Exception as e:
            self.logger.error(f"Error checking for previously processed file: {str(e)}")
            return None

    def convert_heif_to_jpeg(
        self, heif_path: str, delete_original: bool = True
    ) -> Optional[str]:
        """
        Convert HEIF/HEIC file to JPEG format.

        Args:
            heif_path: Path to HEIF/HEIC file
            delete_original: Whether to delete original after successful conversion

        Returns:
            Path to converted JPEG file or None if conversion failed
        """
        try:
            # Check if file is HEIF/HEIC
            if not heif_path.lower().endswith((".heic", ".heif")):
                self.logger.warning(f"File is not HEIF/HEIC: {heif_path}")
                return None

            # Load with PIL (pillow-heif registered)
            img = PILImage.open(heif_path)

            # Create output path with .jpg extension
            base_path = os.path.splitext(heif_path)[0]
            jpeg_path = f"{base_path}.jpg"

            # Handle existing files
            counter = 1
            while os.path.exists(jpeg_path):
                jpeg_path = f"{base_path}_{counter}.jpg"
                counter += 1

            # Convert to RGB if necessary
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Save as JPEG with high quality
            img.save(jpeg_path, "JPEG", quality=95, optimize=True)

            # Verify conversion
            if os.path.exists(jpeg_path) and os.path.getsize(jpeg_path) > 0:
                self.logger.info(f"Converted HEIF to JPEG: {heif_path} -> {jpeg_path}")

                # Copy metadata
                try:
                    self.copy_with_metadata(heif_path, jpeg_path)
                except Exception as e:
                    self.logger.warning(f"Could not copy all metadata: {e}")

                # Delete original if requested
                if delete_original:
                    try:
                        os.remove(heif_path)
                        self.logger.info(f"Deleted original HEIF file: {heif_path}")
                    except Exception as e:
                        self.logger.error(f"Could not delete original HEIF: {e}")

                return jpeg_path
            else:
                self.logger.error("JPEG file not created or empty")
                return None

        except Exception as e:
            self.logger.error(f"Error converting HEIF to JPEG: {e}")
            return None

    def save_compartment(
        self,
        image: np.ndarray,
        hole_id: str,
        compartment_num: int,
        has_data: bool = False,
        output_format: str = "png",
        source_uid: str = None,
    ) -> str:
        """
        Save a compartment image.

        Args:
            image: Image to save
            hole_id: Hole ID
            compartment_num: Compartment number
            has_data: Whether the image has data columns
            output_format: Output image format
            source_uid: UID from process image original file

        Returns:
            Path to the saved file
        """
        try:
            save_dir = self.get_hole_dir("approved_compartments", hole_id)

            if has_data:
                filename = f"{hole_id}_CC_{compartment_num:03d}_Data.{output_format}"
            else:
                filename = f"{hole_id}_CC_{compartment_num:03d}.{output_format}"

            file_path = os.path.join(save_dir, filename)

            # Handle PNG with UID
            if output_format.lower() == "png" and source_uid:
                success = self.save_png_with_uid(image, file_path, source_uid)
                if not success:
                    cv2.imwrite(file_path, image)
            elif output_format.lower() == "jpg":
                cv2.imwrite(file_path, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
                # Embed UID in JPEG if provided
                if source_uid:
                    self.embed_uid_in_image(file_path, source_uid)
            else:
                cv2.imwrite(file_path, image)

            self.logger.info(f"Saved compartment image: {file_path}")
            return file_path
        except Exception as e:
            self.logger.error(f"Error saving compartment image: {str(e)}")
            return None

    def save_reviewed_compartment(
        self,
        image: np.ndarray,
        hole_id: str,
        compartment_depth: int,
        status: str,
        output_format: str = "png",
        source_uid: str = None,
    ) -> Dict[str, Any]:
        """
        Save a reviewed compartment with appropriate suffix and upload to shared folder.

        Args:
            image: Compartment image
            hole_id: Hole ID
            compartment_depth: Compartment depth number
            status: Review status
            output_format: Output format
            source_uid: unique ID from original image

        Returns:
            Dictionary with save results
        """
        result = {"local_path": None, "shared_path": None, "upload_success": False}

        try:
            # Skip if keeping original or missing
            if status in ["KEEP_ORIGINAL", "MISSING"]:
                return result

            # Guard against invalid/unknown status — never write an unclassified file
            # to the approved folder
            if status not in ["Wet", "Dry"]:
                self.logger.error(
                    f"save_reviewed_compartment called with invalid status '{status}' "
                    f"for {hole_id}_CC_{compartment_depth:03d} — skipping to prevent "
                    f"unclassified file in approved folder"
                )
                return result

            # Build the classified filename upfront — no intermediate unclassified file
            save_dir = self.get_hole_dir("approved_compartments", hole_id)
            filename = f"{hole_id}_CC_{compartment_depth:03d}_{status}.{output_format}"
            local_path = os.path.join(save_dir, filename)

            # Save directly to final classified path
            if output_format.lower() == "png" and source_uid:
                success = self.save_png_with_uid(image, local_path, source_uid)
                if not success:
                    cv2.imwrite(local_path, image)
            elif output_format.lower() == "jpg":
                cv2.imwrite(local_path, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
                if source_uid:
                    self.embed_uid_in_image(local_path, source_uid)
            else:
                cv2.imwrite(local_path, image)

            if not os.path.exists(local_path):
                self.logger.error(f"Failed to write reviewed compartment: {local_path}")
                return result

            self.logger.info(f"Saved reviewed compartment: {local_path}")
            result["local_path"] = local_path

            # Upload to shared folder if configured
            shared_path = self.get_shared_path(
                "approved_compartments", create_if_missing=True
            )
            if shared_path:
                project_code = hole_id[:2].upper() if len(hole_id) >= 2 else ""
                shared_hole_folder = shared_path / project_code / hole_id
                shared_hole_folder.mkdir(parents=True, exist_ok=True)

                # Create filename
                filename = os.path.basename(local_path)
                shared_file_path = shared_hole_folder / filename

                # Copy to shared
                if self.copy_with_metadata(local_path, str(shared_file_path)):
                    if self._verify_upload(shared_file_path):
                        # Mark as uploaded
                        base, ext = os.path.splitext(local_path)
                        uploaded_path = f"{base}_UPLOADED{ext}"
                        os.rename(local_path, uploaded_path)

                        result["local_path"] = uploaded_path
                        result["shared_path"] = str(shared_file_path)
                        result["upload_success"] = True

        except Exception as e:
            self.logger.error(f"Error saving reviewed compartment: {e}")

        return result

    def save_compartment_with_data(
        self,
        image: np.ndarray,
        hole_id: str,
        compartment_num: int,
        output_format: str = "tiff",
        source_uid: str = None,
    ) -> str:
        """
        Save a compartment image with data columns to a 'With_Data' subfolder.

        Args:
            image: Image to save
            hole_id: Hole ID
            compartment_num: Compartment number
            output_format: Output image format
            source_uid: UID from source image to preserve

        Returns:
            Path to the saved file
        """
        try:
            # Get the appropriate directory and create a "With_Data" subfolder
            base_dir = self.get_hole_dir("approved_compartments", hole_id)
            with_data_dir = os.path.join(base_dir, "With_Data")
            os.makedirs(with_data_dir, exist_ok=True)

            # Create filename with 3-digit compartment number (001, 002, etc.)
            filename = f"{hole_id}_CC_{compartment_num:03d}_Data.{output_format}"

            # Full path
            file_path = os.path.join(with_data_dir, filename)

            # Save the image based on format
            if output_format.lower() == "png" and source_uid:
                success = self.save_png_with_uid(image, file_path, source_uid)
                if not success:
                    cv2.imwrite(file_path, image)
            elif output_format.lower() == "jpg":
                cv2.imwrite(file_path, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
                # Embed UID in JPEG if provided
                if source_uid:
                    self.embed_uid_in_image(file_path, source_uid)
            elif output_format.lower() in ["tif", "tiff"]:
                cv2.imwrite(file_path, image)
                # Embed UID in TIFF if provided
                if source_uid:
                    self.embed_uid_in_any_image(file_path, source_uid)
            else:
                cv2.imwrite(file_path, image)

            self.logger.info(f"Saved compartment with data image: {file_path}")
            return file_path

        except Exception as e:
            self.logger.error(f"Error saving compartment with data image: {str(e)}")
            return None

    def save_temp_compartment(
        self,
        image: np.ndarray,
        hole_id: str,
        compartment_depth: int,
        suffix: str = "temp",
        source_uid: str = None,
    ) -> str:
        """Save a temporary compartment image for review."""
        try:
            save_dir = self.get_hole_dir("temp_review", hole_id)
            filename = f"{hole_id}_CC_{compartment_depth:03d}_{suffix}.png"
            file_path = os.path.join(save_dir, filename)

            # Save with UID if provided
            if source_uid:
                success = self.save_png_with_uid(image, file_path, source_uid)
                if not success:
                    cv2.imwrite(file_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            else:
                cv2.imwrite(file_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            self.logger.info(f"Saved temp compartment: {file_path}")
            return file_path
        except Exception as e:
            self.logger.error(f"Error saving temp compartment: {str(e)}")
            return None

    def save_approved_compartment_direct(
        self,
        image: np.ndarray,
        hole_id: str,
        compartment_depth: int,
        moisture_status: str,
        source_uid: str = None,
        upload_to_shared: bool = True,
        overwrite_existing: bool = False,
    ) -> Optional[str]:
        """
        Save a compartment directly to approved folder with known wet/dry status.

        Used for re-extraction when wet/dry classification is already known from register.
        Skips the temp/review folder entirely.

        Args:
            image: Compartment image as numpy array
            hole_id: Hole identifier
            compartment_depth: Depth of compartment
            moisture_status: "Wet" or "Dry"
            source_uid: Source image UID to embed in PNG
            upload_to_shared: Whether to also upload to shared folder
            overwrite_existing: Whether to overwrite an existing file with the same UID

        Returns:
            Path to saved file or None if failed
        """
        try:
            # Validate moisture status
            if moisture_status not in ("Wet", "Dry"):
                self.logger.warning(
                    f"Invalid moisture status '{moisture_status}', defaulting to temp save"
                )
                return self.save_temp_compartment(
                    image, hole_id, compartment_depth, "temp", source_uid
                )

            # Get approved compartments directory
            save_dir = self.get_hole_dir("approved_compartments", hole_id)
            filename = f"{hole_id}_CC_{compartment_depth:03d}_{moisture_status}.png"
            file_path = os.path.join(save_dir, filename)

            # Check if file already exists with same UID - preserve it
            if os.path.exists(file_path) and source_uid and not overwrite_existing:
                existing_uid = self.extract_uid_from_any_image(file_path)
                if existing_uid == source_uid:
                    self.logger.info(
                        f"Compartment already exists with same UID, skipping: {file_path}"
                    )
                    return file_path

            # Save with UID if provided
            if source_uid:
                success = self.save_png_with_uid(image, file_path, source_uid)
                if not success:
                    cv2.imwrite(file_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            else:
                cv2.imwrite(file_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            self.logger.info(f"Saved approved compartment directly: {file_path}")

            # Upload to shared folder if configured
            if upload_to_shared:
                shared_path = self.get_shared_path(
                    "approved_compartments", create_if_missing=True
                )
                if shared_path:
                    project_code = hole_id[:2].upper() if len(hole_id) >= 2 else ""
                    shared_hole_dir = os.path.join(shared_path, project_code, hole_id)
                    os.makedirs(shared_hole_dir, exist_ok=True)
                    shared_file_path = os.path.join(shared_hole_dir, filename)

                    # Copy to shared with UID preserved
                    if source_uid:
                        self.save_png_with_uid(image, shared_file_path, source_uid)
                    else:
                        cv2.imwrite(
                            shared_file_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0]
                        )

                    self.logger.info(f"Uploaded to shared folder: {shared_file_path}")

            return file_path

        except Exception as e:
            self.logger.error(f"Error saving approved compartment direct: {str(e)}")
            return None

    def cleanup_temp_compartments(self, hole_id: str, temp_paths: List[str]) -> None:
        """
        Clean up temporary compartment files after processing.
        Only removes files if we can confirm they were successfully uploaded.

        Args:
            hole_id: Hole ID to clean up
            temp_paths: List of temporary file paths to remove
        """
        try:
            # Get approved compartments directory to check for uploads
            approved_dir = self.get_hole_dir("approved_compartments", hole_id)

            # Remove individual temp files only if uploaded
            for temp_path in temp_paths:
                if temp_path and os.path.exists(temp_path):
                    try:
                        # Extract compartment number from temp filename
                        filename = os.path.basename(temp_path)
                        match = re.search(r"CC_(\d{3})", filename)

                        if match:
                            comp_num = match.group(1)
                            base_name = f"{hole_id}_CC_{comp_num}"

                            # Check if uploaded file exists
                            uploaded_patterns = [
                                f"{base_name}_Wet_UPLOADED.png",
                                f"{base_name}_Dry_UPLOADED.png",
                                f"{base_name}_UPLOADED.png",
                                f"{base_name}_Wet_UPLOAD_FAILED.png",
                                f"{base_name}_Dry_UPLOAD_FAILED.png",
                                f"{base_name}_UPLOAD_FAILED.png",
                            ]

                            # Debug logging
                            self.logger.debug(f"Checking for uploads of {base_name}:")
                            self.logger.debug(f"  Approved dir: {approved_dir}")
                            self.logger.debug(
                                f"  Dir exists: {os.path.exists(approved_dir)}"
                            )
                            if os.path.exists(approved_dir):
                                actual_files = os.listdir(approved_dir)
                                self.logger.debug(f"  Files in dir: {actual_files}")

                            upload_exists = False
                            for pattern in uploaded_patterns:
                                full_path = os.path.join(approved_dir, pattern)
                                exists = os.path.exists(full_path)
                                self.logger.debug(f"  Checking {pattern}: {exists}")
                                if exists:
                                    upload_exists = True
                                    break

                            if upload_exists:
                                os.remove(temp_path)
                                self.logger.debug(
                                    f"Removed temp file after confirming upload: {temp_path}"
                                )
                            else:
                                self.logger.warning(
                                    f"Keeping temp file - no upload confirmation: {temp_path}"
                                )
                        else:
                            # Can't determine comp number, keep the file
                            self.logger.warning(
                                f"Could not extract compartment number from: {filename}"
                            )

                    except Exception as e:
                        self.logger.warning(
                            f"Could not remove temp file {temp_path}: {e}"
                        )

            # Try to remove empty directory structure
            temp_review_dir = self.get_hole_dir("temp_review", hole_id)
            if os.path.exists(temp_review_dir):
                try:
                    # Remove hole directory if empty
                    if not os.listdir(temp_review_dir):
                        os.rmdir(temp_review_dir)

                        # Try to remove project code directory if empty
                        project_dir = os.path.dirname(temp_review_dir)
                        if os.path.exists(project_dir) and not os.listdir(project_dir):
                            os.rmdir(project_dir)

                except Exception as e:
                    self.logger.debug(f"Could not remove empty directories: {e}")

        except Exception as e:
            self.logger.error(f"Error cleaning up temp files: {e}")

    def rename_debug_files(
        self,
        original_filename: str,
        hole_id: Optional[str],
        depth_from: Optional[float],
        depth_to: Optional[float],
    ) -> None:
        """
        Rename debug files for a specific image after successful metadata extraction.

        Args:
            original_filename: Original input image filename
            hole_id: Extracted hole identifier
            depth_from: Starting depth
            depth_to: Ending depth
        """
        try:
            if not (hole_id and depth_from is not None and depth_to is not None):
                return

            # Get the debug directory for this hole ID
            debug_dir = self.get_hole_dir("debug_images", hole_id)

            # Find all debug files for this image
            base_name = os.path.splitext(os.path.basename(original_filename))[0]

            try:
                debug_files = [
                    f
                    for f in os.listdir(debug_dir)
                    if f.startswith(base_name) and f.endswith(".jpg")
                ]
            except FileNotFoundError:
                # Directory might not exist yet
                os.makedirs(debug_dir, exist_ok=True)
                debug_files = []
                self.logger.info(f"Created debug directory for {hole_id}")

            # Also look for temp debug images in the Unidentified folder
            temp_debug_dir = os.path.join(
                self.dir_structure["debug_images"], "Unidentified"
            )
            if os.path.exists(temp_debug_dir):
                try:
                    temp_files = [
                        f
                        for f in os.listdir(temp_debug_dir)
                        if f.startswith(base_name) and f.endswith(".jpg")
                    ]

                    # Move and rename temp files
                    for old_filename in temp_files:
                        # Extract the step name from the old filename
                        if "_" in old_filename:
                            step_name = old_filename.split("_", 1)[1].replace(
                                ".jpg", ""
                            )

                            # Generate new filename with metadata
                            new_filename = f"{hole_id}_{int(depth_from)}-{int(depth_to)}_Debug_{step_name}.jpg"

                            old_path = os.path.join(temp_debug_dir, old_filename)
                            new_path = os.path.join(debug_dir, new_filename)

                            try:
                                # Ensure debug directory exists
                                os.makedirs(debug_dir, exist_ok=True)

                                if self.copy_with_metadata(old_path, new_path):
                                    # Only remove original after successful copy
                                    os.remove(old_path)
                                    self.logger.info(
                                        f"Moved debug file from temp location: {old_filename} -> {new_filename}"
                                    )
                                else:
                                    self.logger.error(
                                        f"Failed to copy debug file with metadata: {old_filename}"
                                    )

                                # Only remove original after successful copy
                                os.remove(old_path)

                                self.logger.info(
                                    f"Moved debug file from temp location: {old_filename} -> {new_filename}"
                                )
                            except Exception as e:
                                self.logger.error(
                                    f"Error moving temp debug file {old_filename}: {e}"
                                )
                except Exception as e:
                    self.logger.error(f"Error processing temp debug files: {e}")

            # Rename files in the debug directory
            for old_filename in debug_files:
                try:
                    # Extract the step name from the old filename
                    step_parts = old_filename.split("_")
                    if len(step_parts) >= 2:
                        step_name = step_parts[-1].replace(".jpg", "")

                        # Generate new filename with metadata
                        new_filename = f"{hole_id}_{int(depth_from)}-{int(depth_to)}_Debug_{step_name}.jpg"

                        old_path = os.path.join(debug_dir, old_filename)
                        new_path = os.path.join(debug_dir, new_filename)

                        # Rename the file
                        os.rename(old_path, new_path)
                        self.logger.info(
                            f"Renamed debug file: {old_filename} -> {new_filename}"
                        )
                except Exception as e:
                    self.logger.error(f"Error renaming debug file {old_filename}: {e}")
        except Exception as e:
            self.logger.error(f"Error in rename_debug_files: {e}")

    def save_temp_debug_image(
        self, image: np.ndarray, original_filename: str, debug_type: str
    ) -> str:
        """
        Save a debug image without proper hole/depth metadata.

        Args:
            image: Image to save
            original_filename: Original input filename used to generate base name
            debug_type: Type of debug image

        Returns:
            Full path to the saved image
        """
        try:
            # Use the centralized directory structure
            temp_dir = os.path.join(self.dir_structure["debug_images"], "Unidentified")
            os.makedirs(temp_dir, exist_ok=True)

            # Generate filename from original file
            base_name = os.path.splitext(os.path.basename(original_filename))[0]
            filename = f"{base_name}_{debug_type}.jpg"
            full_path = os.path.join(temp_dir, filename)

            # Save the image with moderate compression
            cv2.imwrite(full_path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])

            self.logger.info(f"Saved temporary debug image: {full_path}")
            return full_path

        except Exception as e:
            self.logger.error(f"Error saving temporary debug image: {str(e)}")

            # Create a fallback directory if needed
            fallback_dir = os.path.join(self.base_dir, "Temp_Debug")
            os.makedirs(fallback_dir, exist_ok=True)

            # Generate a unique fallback filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fallback_filename = f"debug_{timestamp}_{debug_type}.jpg"
            fallback_path = os.path.join(fallback_dir, fallback_filename)

            # Try to save to fallback location
            try:
                cv2.imwrite(fallback_path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                self.logger.info(
                    f"Saved debug image to fallback location: {fallback_path}"
                )
                return fallback_path
            except Exception as fallback_error:
                self.logger.error(
                    f"Failed to save debug image to fallback location: {str(fallback_error)}"
                )
                return ""

    def create_base_directories(self) -> None:
        """Create the base directory structure, and log only if something is newly created."""
        try:
            created_any = False

            # Create the base directory if needed
            if not os.path.exists(self.base_dir):
                os.makedirs(self.base_dir)
                created_any = True

            # Create each subdirectory if it doesn't exist
            for dir_path in self.dir_structure.values():
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                    created_any = True

            if created_any:
                self.logger.info(
                    f"✅ Base directory structure created at: {self.base_dir}"
                )
            else:
                self.logger.debug(
                    f"📁 All directories already existed at: {self.base_dir}"
                )

        except Exception as e:
            self.logger.error(f"❌ Error creating directory structure: {e}")
            raise

    def create_folder_structure(self, base_path: str) -> None:
        """Create the full folder structure at a specified path (includes register folder for OneDrive)."""
        try:
            base = Path(base_path)
            created_any = False

            # Create base directory
            if not base.exists():
                os.makedirs(base)
                created_any = True

            # Create all main folders including register
            for folder_name in self.FOLDER_NAMES.values():
                folder_path = base / folder_name
                if not folder_path.exists():
                    os.makedirs(folder_path)
                    created_any = True

            # Create subfolders under Processed Original Images
            processed_path = base / self.FOLDER_NAMES["processed"]
            for subfolder in ["approved_originals", "rejected_originals", "pending_originals"]:
                subfolder_path = processed_path / self.SUBFOLDER_NAMES[subfolder]
                if not subfolder_path.exists():
                    os.makedirs(subfolder_path)
                    created_any = True

            # Create subfolders under Extracted Compartment Images
            compartments_path = base / self.FOLDER_NAMES["compartments"]
            for subfolder in ["approved_compartments", "review_compartments"]:
                subfolder_path = compartments_path / self.SUBFOLDER_NAMES[subfolder]
                if not subfolder_path.exists():
                    os.makedirs(subfolder_path)
                    created_any = True

            # Create subfolders under Debugging
            debugging_path = base / self.FOLDER_NAMES["debugging"]
            for subfolder in ["blur", "debug"]:
                subfolder_path = debugging_path / self.SUBFOLDER_NAMES[subfolder]
                if not subfolder_path.exists():
                    os.makedirs(subfolder_path)
                    created_any = True

            if created_any:
                self.logger.info(f"✅ Folder structure created at: {base_path}")
            else:
                self.logger.debug(f"📁 All folders already existed at: {base_path}")

        except Exception as e:
            self.logger.error(f"❌ Error creating folder structure: {e}")
            raise

    def create_local_folder_structure(self, base_path: str) -> None:
        """Create folder structure at a specified path (excludes register folder for local storage)."""
        try:
            base = Path(base_path)
            created_any = False

            # Create base directory
            if not base.exists():
                os.makedirs(base)
                created_any = True

            # Create all main folders EXCEPT register
            for key, folder_name in self.FOLDER_NAMES.items():
                if key == "register":
                    continue  # Skip register folder for local storage
                folder_path = base / folder_name
                if not folder_path.exists():
                    os.makedirs(folder_path)
                    created_any = True

            # Create subfolders under Processed Original Images
            processed_path = base / self.FOLDER_NAMES["processed"]
            for subfolder in ["approved_originals", "rejected_originals", "pending_originals"]:
                subfolder_path = processed_path / self.SUBFOLDER_NAMES[subfolder]
                if not subfolder_path.exists():
                    os.makedirs(subfolder_path)
                    created_any = True

            # Create subfolders under Extracted Compartment Images
            compartments_path = base / self.FOLDER_NAMES["compartments"]
            for subfolder in ["approved_compartments", "review_compartments"]:
                subfolder_path = compartments_path / self.SUBFOLDER_NAMES[subfolder]
                if not subfolder_path.exists():
                    os.makedirs(subfolder_path)
                    created_any = True

            # Create subfolders under Debugging
            debugging_path = base / self.FOLDER_NAMES["debugging"]
            for subfolder in ["blur", "debug"]:
                subfolder_path = debugging_path / self.SUBFOLDER_NAMES[subfolder]
                if not subfolder_path.exists():
                    os.makedirs(subfolder_path)
                    created_any = True

            if created_any:
                self.logger.info(f"✅ Local folder structure created at: {base_path}")
            else:
                self.logger.debug(f"📁 All local folders already existed at: {base_path}")

        except Exception as e:
            self.logger.error(f"❌ Error creating local folder structure: {e}")
            raise

    def get_hole_dir(self, dir_type: str, hole_id: str) -> str:
        """Get the directory path for a specific hole ID with project code."""
        if dir_type not in self.dir_structure:
            raise ValueError(f"Invalid directory type: {dir_type}")

        # Extract project code (first 2 characters)
        project_code = hole_id[:2].upper() if len(hole_id) >= 2 else "XX"

        # Create project/hole directory structure
        hole_dir = os.path.join(self.dir_structure[dir_type], project_code, hole_id)
        os.makedirs(hole_dir, exist_ok=True)

        return hole_dir

    def save_blur_analysis(
        self, image: np.ndarray, hole_id: str, compartment_num: int
    ) -> str:
        """
        Save a blur analysis image.

        Args:
            image: Image to save
            hole_id: Hole ID
            compartment_num: Compartment number

        Returns:
            Path to the saved file
        """
        try:
            # Get the appropriate directory
            save_dir = self.get_hole_dir("blur_analysis", hole_id)

            # Create filename
            filename = f"{hole_id}_{compartment_num}_blur_analysis.jpg"

            # Full path
            file_path = os.path.join(save_dir, filename)

            # Save the image
            cv2.imwrite(file_path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])

            self.logger.info(f"Saved blur analysis image: {file_path}")
            return file_path

        except Exception as e:
            self.logger.error(f"Error saving blur analysis image: {str(e)}")
            return None

    def save_debug_image(
        self,
        image: np.ndarray,
        hole_id: str,
        depth_from: float,
        depth_to: float,
        image_type: str,
    ) -> str:
        """
        Save a debug image.

        Args:
            image: Image to save
            hole_id: Hole ID
            depth_from: Starting depth
            depth_to: Ending depth
            image_type: Type of debug image

        Returns:
            Path to the saved file
        """
        try:
            # Get the appropriate directory
            save_dir = self.get_hole_dir("debug_images", hole_id)

            # Create filename
            filename = (
                f"{hole_id}_{int(depth_from)}-{int(depth_to)}_Debug_{image_type}.jpg"
            )

            # Full path
            file_path = os.path.join(save_dir, filename)

            # Save the image
            cv2.imwrite(file_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])

            self.logger.info(f"Saved debug image: {file_path}")
            return file_path

        except Exception as e:
            self.logger.error(f"Error saving debug image: {str(e)}")
            return None

    def save_drill_trace(self, image: np.ndarray, hole_id: str) -> str:
        """
        Save a drill trace image.

        Args:
            image: Image to save
            hole_id: Hole ID

        Returns:
            Path to the saved file
        """
        try:
            # Get the appropriate directory
            save_dir = self.dir_structure["drill_traces"]

            # Create filename
            filename = f"{hole_id}_Trace.tiff"

            # Full path
            file_path = os.path.join(save_dir, filename)

            # Save the image
            cv2.imwrite(file_path, image)

            self.logger.info(f"Saved drill trace image: {file_path}")
            return file_path

        except Exception as e:
            self.logger.error(f"Error saving drill trace image: {str(e)}")
            return None

    def save_embedding_plot(
        self, image_path: str, filename: str = "embedding_plot.png"
    ) -> Optional[str]:
        """
        Save an embedding plot image to the debugging directory.

        Args:
            image_path: Path to the image file to copy.
            filename: Desired filename for the saved plot.

        Returns:
            Path to the saved plot or None if an error occurs.
        """
        try:
            save_dir = self.dir_structure["debugging"]
            os.makedirs(save_dir, exist_ok=True)
            dest = os.path.join(save_dir, filename)
            shutil.copy(image_path, dest)
            self.logger.info(f"Saved embedding plot: {dest}")
            return dest
        except Exception as e:
            self.logger.error(f"Error saving embedding plot: {str(e)}")
            return None

    def save_original_file(
        self,
        source_path: str,
        hole_id: str,
        depth_from: int,
        depth_to: int,
        is_processed: bool = True,
        is_rejected: bool = False,
        is_selective: bool = False,
        is_skipped: bool = False,
        is_pending: bool = False,
        delete_source: bool = True,
        image_uid: str = None,
    ) -> Tuple[Optional[str], bool]:
        """
        Save original file by uploading to shared folder (if available) and moving to local storage.
        This method handles both operations atomically to avoid file loss and preserves metadata.

        Args:
            source_path: Path to the source file
            hole_id: Hole ID
            depth_from: Starting depth
            depth_to: Ending depth
            is_processed: Whether the file was processed (True for all processed files)
            is_rejected: Whether the file was rejected during processing
            is_selective: Whether this is a selective compartment replacement
            is_skipped: Whether the file was skipped during duplicate check
            delete_source: Whether to delete source file after successful copy

        Returns:
            Tuple of (local_path, upload_success) where:
                local_path: Path to the final local file location (None if failed)
                upload_success: Whether shared folder upload was successful
        """
        try:
            # Check if source file exists
            if not os.path.exists(source_path):
                self.logger.warning(f"Source file does not exist: {source_path}")
                return None, False

            # Get original file extension
            _, ext = os.path.splitext(source_path)

            # Determine base suffix based on status
            if is_rejected:
                base_suffix = "_Rejected"
            elif is_skipped:
                base_suffix = "_Skipped"
            elif is_selective:
                base_suffix = "_Selected_Compartments"
            elif is_pending and image_uid:
                # Pending originals include UID for QAQC traceability
                base_suffix = f"_Original_{image_uid[:8]}"
            else:
                base_suffix = "_Original"

            # Variables to track uploads
            shared_upload_path = None
            upload_attempted = False

            # Start shared folder upload first (using internal shared_paths)
            # Use internal shared paths configuration
            # Skip shared upload for pending originals - they stay local until QAQC finalization
            if self.shared_paths and is_processed and not is_pending:
                try:
                    # Use get_shared_path() method
                    if is_rejected:
                        shared_base = self.get_shared_path("rejected_originals")
                    else:
                        # For approved/processed originals
                        shared_base = self.get_shared_path("approved_originals")

                    # Only attempt upload if we have a valid base path
                    if shared_base:
                        upload_attempted = True
                        self.logger.info("Starting shared folder upload...")

                        # Extract project code from hole_id
                        project_code = hole_id[:2].upper() if len(hole_id) >= 2 else ""

                        # Use Path for better path handling
                        if project_code:
                            hole_folder = shared_base / project_code / hole_id
                        else:
                            hole_folder = shared_base / hole_id

                        hole_folder.mkdir(parents=True, exist_ok=True)

                        # Create shared filename
                        base_shared_filename = (
                            f"{hole_id}_{int(depth_from)}-{int(depth_to)}{base_suffix}"
                        )
                        shared_filename = f"{base_shared_filename}{ext}"
                        shared_upload_path = hole_folder / shared_filename

                        # If file exists, add incrementing number
                        counter = 1
                        while shared_upload_path.exists():
                            shared_filename = f"{base_shared_filename}_{counter}{ext}"
                            shared_upload_path = hole_folder / shared_filename
                            counter += 1

                        # Copy to shared folder
                        if self.copy_with_metadata(
                            source_path, str(shared_upload_path)
                        ):
                            self.logger.info(
                                f"Started shared folder upload: {shared_upload_path}"
                            )
                        else:
                            self.logger.error(f"Failed to start shared folder copy")
                            shared_upload_path = None
                    else:
                        self.logger.info(
                            "Shared folder path not configured for this file type"
                        )
                        upload_attempted = False

                except Exception as e:
                    self.logger.error(
                        f"Shared folder upload initiation failed: {str(e)}"
                    )
                    shared_upload_path = None

            # Now handle local storage
            # Determine local storage directory
            if is_rejected:
                target_dir = self.get_hole_dir("rejected_originals", hole_id)
            elif is_skipped:
                # Skipped files go to rejected folder but with different suffix
                target_dir = self.get_hole_dir("rejected_originals", hole_id)
            elif is_pending:
                # Pending originals go to staging folder until QAQC finalization
                target_dir = self.get_hole_dir("pending_originals", hole_id)
            else:
                target_dir = self.get_hole_dir("approved_originals", hole_id)

            # Create local filename (without suffix for now)
            base_local_filename = (
                f"{hole_id}_{int(depth_from)}-{int(depth_to)}{base_suffix}"
            )
            local_filename = f"{base_local_filename}{ext}"
            target_path = os.path.join(target_dir, local_filename)

            # Handle existing files - but don't save if we're skipping
            if is_skipped and not is_rejected:
                # If skipping (keep original), check if original already exists
                if os.path.exists(target_path):
                    self.logger.info(
                        f"Skipping save - keeping existing original at {target_path}"
                    )
                    return (
                        target_path,
                        upload_success,
                    )  # Return existing path without saving

            # Only increment counter for actual new files
            counter = 1
            while os.path.exists(target_path):
                local_filename = f"{base_local_filename}_{counter}{ext}"
                target_path = os.path.join(target_dir, local_filename)
                counter += 1

            # If UID provided, embed it in the source image
            if image_uid:
                embedded_uid = self.embed_uid_in_any_image(source_path, image_uid)
                self.logger.info(
                    f"Embedded UID {embedded_uid} in source image before saving"
                )

            # Move to local storage preserving metadata
            local_success = False
            try:
                os.makedirs(os.path.dirname(target_path), exist_ok=True)

                # Copy with metadata then delete source
                if self.copy_with_metadata(source_path, target_path):
                    # Verify local copy - but give it a moment to complete
                    import time

                    time.sleep(0.1)  # Small delay to ensure file write is complete

                    if os.path.exists(target_path):
                        # For verification, check file exists and has reasonable size (not just exact match)
                        source_size = os.path.getsize(source_path)
                        target_size = os.path.getsize(target_path)

                        # Allow small size differences (metadata might differ slightly)
                        if (
                            abs(target_size - source_size) < 1000
                        ):  # Within 1KB difference
                            local_success = True
                            self.logger.info(f"Local copy successful: {target_path}")
                        else:
                            self.logger.error(
                                f"Local copy size mismatch: source={source_size}, target={target_size}"
                            )
                    else:
                        self.logger.error("Local copy not found after copy operation")
                else:
                    self.logger.error("Failed to copy file locally")

            except Exception as e:
                self.logger.error(f"Failed to save file locally: {str(e)}")

            # verify shared folder upload if it was attempted
            upload_success = False
            if shared_upload_path and upload_attempted:
                try:
                    # Give shared folder a moment to sync (helps with cloud/network drives)
                    import time

                    time.sleep(0.5)  # 500ms delay for sync

                    # Verify the upload (more tolerant of small size differences)
                    if shared_upload_path.exists():
                        local_size = os.path.getsize(target_path)
                        shared_size = shared_upload_path.stat().st_size

                        # Allow small size differences (metadata, compression, etc.)
                        size_diff = abs(shared_size - local_size)
                        max_diff = max(
                            1024, local_size * 0.01
                        )  # 1KB or 1% of file size

                        if size_diff <= max_diff:
                            upload_success = True
                            self.logger.info(
                                f"Shared folder upload verified: {shared_upload_path} "
                                f"(local: {local_size}, shared: {shared_size})"
                            )
                        else:
                            self.logger.warning(
                                f"Shared folder upload size mismatch: local={local_size}, "
                                f"shared={shared_size}, diff={size_diff}"
                            )
                    else:
                        self.logger.warning(
                            f"Shared folder upload verification failed - file not found: {shared_upload_path}"
                        )

                except Exception as e:
                    self.logger.error(f"Error verifying shared folder upload: {str(e)}")

            # Determine final suffix and rename local file if needed
            if upload_success:
                final_suffix = base_suffix + "_UPLOADED"
            elif upload_attempted:
                final_suffix = base_suffix + "_UPLOAD_FAILED"
            else:
                final_suffix = base_suffix

            # Rename local file if suffix changed
            if final_suffix != base_suffix and local_success:
                try:
                    new_base_filename = (
                        f"{hole_id}_{int(depth_from)}-{int(depth_to)}{final_suffix}"
                    )
                    new_filename = f"{new_base_filename}{ext}"
                    new_target_path = os.path.join(
                        os.path.dirname(target_path), new_filename
                    )

                    # Handle existing files with new name
                    counter = 1
                    while os.path.exists(new_target_path):
                        new_filename = f"{new_base_filename}_{counter}{ext}"
                        new_target_path = os.path.join(
                            os.path.dirname(target_path), new_filename
                        )
                        counter += 1

                    os.rename(target_path, new_target_path)
                    target_path = new_target_path
                    self.logger.info(f"Renamed local file to: {new_target_path}")

                except Exception as e:
                    self.logger.error(f"Failed to rename local file: {str(e)}")
                    # Continue with original path if rename fails

            # SAFETY CHECK: Delete source file if we have at least one successful copy
            if delete_source and os.path.exists(source_path):
                # Check if we have at least one successful save (local or shared)
                has_local = local_success and os.path.exists(target_path)
                has_shared = (
                    upload_success
                    and shared_upload_path
                    and shared_upload_path.exists()
                )

                if has_local or has_shared:
                    try:
                        os.remove(source_path)
                        self.logger.info(
                            f"Source file deleted after successful save: {source_path}"
                        )
                    except Exception as e:
                        self.logger.error(f"Failed to delete source file: {str(e)}")
                        # Try alternative deletion method
                        try:
                            import shutil

                            shutil.move(source_path, source_path + ".processed")
                            self.logger.info(
                                f"Renamed source file to .processed: {source_path}"
                            )
                        except Exception as e:
                            self.logger.debug(f"Could not rename source file: {e}")
                else:
                    self.logger.warning(
                        f"Source file not deleted - no successful saves confirmed"
                    )
                    self.logger.warning(
                        f"Local save: {has_local}, Shared save: {has_shared}"
                    )

            if local_success or upload_success:
                return target_path, upload_success
            else:
                # Both saves failed - try emergency move
                self.logger.warning("Both saves failed, attempting emergency move...")
                emergency_path = self.move_original_file(
                    source_path,
                    hole_id,
                    depth_from,
                    depth_to,
                    is_processed,
                    is_rejected,
                    is_selective,
                    is_skipped,
                    suffix="EMERGENCY_SAVE",
                    image_uid=image_uid,
                )
                if emergency_path:
                    return emergency_path, False
                else:
                    return None, False

        except Exception as e:
            self.logger.error(f"Error in save_original_file: {str(e)}")
            return None, False

    def move_original_file(
        self,
        source_path: str,
        hole_id: str,
        depth_from: float,
        depth_to: float,
        is_processed: bool = True,
        is_rejected: bool = False,
        is_selective: bool = False,
        is_skipped: bool = False,
        suffix: str = None,
        image_uid: str = None,
    ) -> Optional[str]:
        """
        Emergency method to move original file to organized local storage without OneDrive upload.
        Used when the main save_original_file method fails.

        Args:
            source_path: Path to the source file
            hole_id: Hole identifier
            depth_from: Starting depth
            depth_to: Ending depth
            is_processed: Whether the file was successfully processed
            is_rejected: Whether the file was rejected by user
            is_selective: Whether this is a selective replacement
            suffix: Optional suffix to add to filename (e.g., 'UPLOAD_FAILED')

        Returns:
            Path to the saved file or None if save failed
        """
        try:
            # Get the appropriate organized path - UPDATED
            if is_rejected or is_skipped:
                target_dir = self.get_hole_dir("rejected_originals", hole_id)
            else:
                target_dir = self.get_hole_dir("approved_originals", hole_id)

            # Create the target directory
            os.makedirs(target_dir, exist_ok=True)

            # Get file extension
            _, ext = os.path.splitext(source_path)

            # Determine base suffix based on status - UPDATED
            if is_rejected:
                base_suffix = "_Rejected"
            elif is_skipped:
                base_suffix = "_Skipped"
            elif is_selective:
                base_suffix = "_Selected_Compartments"
            else:
                base_suffix = "_Original"

            # Add additional suffix if provided (e.g., "_UPLOAD_FAILED")
            full_suffix = f"{base_suffix}_{suffix}" if suffix else base_suffix

            # Build the new filename: HoleID_From-To_Original_UPLOAD_FAILED
            new_filename = (
                f"{hole_id}_{int(depth_from)}-{int(depth_to)}{full_suffix}{ext}"
            )

            target_path = os.path.join(target_dir, new_filename)

            # Handle existing files
            counter = 1
            while os.path.exists(target_path):
                new_filename = f"{hole_id}_{int(depth_from)}-{int(depth_to)}{full_suffix}_{counter}{ext}"
                target_path = os.path.join(target_dir, new_filename)
                counter += 1

            # Before copying, embed UID if provided
            if image_uid:
                embedded_uid = self.embed_uid_in_any_image(source_path, image_uid)
                self.logger.info(f"Embedded UID {embedded_uid} in emergency save")

            # Copy with metadata preservation
            if self.copy_with_metadata(source_path, target_path):
                # Verify the copy was successful
                if os.path.exists(target_path) and os.path.getsize(
                    target_path
                ) == os.path.getsize(source_path):
                    # Delete source file after successful copy
                    try:
                        os.remove(source_path)
                        self.logger.info(f"Emergency moved file to: {target_path}")
                        return target_path
                    except Exception as e:
                        self.logger.error(
                            f"Failed to delete source after emergency copy: {e}"
                        )
                        # Still return target path - we have the file saved
                        return target_path
                else:
                    self.logger.error("Emergency copy verification failed")
                    # Clean up failed copy
                    if os.path.exists(target_path):
                        try:
                            os.remove(target_path)
                        except Exception as e:
                            self.logger.debug(f"Could not clean up failed copy: {e}")
                    return None
            else:
                self.logger.error("Emergency copy_with_metadata failed")
                return None

        except Exception as e:
            self.logger.error(f"Emergency move failed: {str(e)}")
            return None

    def finalize_pending_original(
        self,
        pending_path: str,
        hole_id: str,
    ) -> Tuple[Optional[str], bool]:
        """
        Finalize a pending original by uploading to shared folder and moving to local approved.
        
        Uses standardized naming: HoleID_From-To_Original_N.ext where N is sequential.
        
        Flow:
        1. Scan shared folder to find next available number
        2. Copy to shared approved_originals as _Original_N
        3. Move to local approved_originals as _Original_N_UPLOADED
        4. Delete the pending file
        
        Args:
            pending_path: Path to the pending original file
            hole_id: Hole identifier
            
        Returns:
            Tuple of (local_path, upload_success)
        """
        import shutil
        import re
        
        try:
            if not os.path.exists(pending_path):
                self.logger.warning(f"Pending original does not exist: {pending_path}")
                return None, False
            
            filename = os.path.basename(pending_path)
            _, ext = os.path.splitext(pending_path)
            
            # Extract depth info from filename
            # Pattern: OK0171_120-140_Original_a1532568.JPG
            match = re.match(
                r"([A-Z]{2}\d{4})_(\d+)-(\d+)_Original_([a-f0-9]+)",
                filename,
                re.IGNORECASE
            )
            if not match:
                self.logger.warning(f"Could not parse pending original filename: {filename}")
                return None, False
            
            parsed_hole_id = match.group(1)
            depth_from = int(match.group(2))
            depth_to = int(match.group(3))
            uid = match.group(4)
            
            project_code = hole_id[:2].upper()
            
            # Find next available number by scanning shared folder
            next_number = self._get_next_original_number(hole_id, depth_from, depth_to)
            
            # Build standardized filename
            base_filename = f"{hole_id}_{depth_from}-{depth_to}_Original_{next_number}"
            
            upload_success = False
            
            # Step 1: Copy to shared approved_originals
            if self.shared_paths:
                shared_base = self.get_shared_path("approved_originals")
                if shared_base:
                    try:
                        shared_hole_folder = shared_base / project_code / hole_id
                        shared_hole_folder.mkdir(parents=True, exist_ok=True)
                        
                        shared_filename = f"{base_filename}{ext}"
                        shared_upload_path = shared_hole_folder / shared_filename
                        
                        # Check if already exists (shouldn't happen with proper numbering)
                        if shared_upload_path.exists():
                            self.logger.warning(f"Shared file already exists: {shared_filename}")
                        else:
                            # Copy to shared
                            if self.copy_with_metadata(pending_path, str(shared_upload_path)):
                                # Verify upload
                                if shared_upload_path.exists():
                                    pending_size = os.path.getsize(pending_path)
                                    shared_size = shared_upload_path.stat().st_size
                                    if abs(shared_size - pending_size) < 1000:
                                        upload_success = True
                                        self.logger.info(
                                            f"Uploaded pending original to shared: {shared_filename}"
                                        )
                                    else:
                                        self.logger.warning(
                                            f"Shared upload size mismatch: {shared_filename}"
                                        )
                    except Exception as e:
                        self.logger.error(f"Failed to upload pending original to shared: {e}")
            
            # Step 2: Move to local approved_originals
            local_approved_dir = self.get_hole_dir("approved_originals", hole_id)
            os.makedirs(local_approved_dir, exist_ok=True)
            
            # Add _UPLOADED suffix if upload succeeded
            if upload_success:
                local_filename = f"{base_filename}_UPLOADED{ext}"
            else:
                local_filename = f"{base_filename}{ext}"
            
            local_path = os.path.join(local_approved_dir, local_filename)
            
            # Move the pending file to approved
            try:
                shutil.move(pending_path, local_path)
                self.logger.info(f"Finalized pending original: {filename} -> {local_filename}")
                
                # Clean up empty pending directories
                pending_dir = os.path.dirname(pending_path)
                if os.path.exists(pending_dir) and not os.listdir(pending_dir):
                    os.rmdir(pending_dir)
                    project_dir = os.path.dirname(pending_dir)
                    if os.path.exists(project_dir) and not os.listdir(project_dir):
                        os.rmdir(project_dir)
                
                return local_path, upload_success
                
            except Exception as e:
                self.logger.error(f"Failed to move pending original to approved: {e}")
                return None, upload_success
            
        except Exception as e:
            self.logger.error(f"Error finalizing pending original: {e}")
            return None, False

    def _get_next_original_number(self, hole_id: str, depth_from: int, depth_to: int) -> int:
        """
        Find the next available number for an original file.
        
        Scans both shared and local approved_originals folders.
        Handles legacy files without numbers (treats as _1).
        
        Returns:
            Next available number (1, 2, 3, etc.)
        """
        import re
        
        existing_numbers = set()
        project_code = hole_id[:2].upper()
        base_pattern = f"{hole_id}_{depth_from}-{depth_to}_Original"
        
        # Pattern to match: _Original.ext, _Original_1.ext, _Original_2.ext, etc.
        # Also handles _UPLOADED suffix
        pattern = re.compile(
            rf"{re.escape(base_pattern)}(?:_(\d+))?(?:_UPLOADED)?(?:_UPLOAD_FAILED)?\.[a-zA-Z]+$",
            re.IGNORECASE
        )
        
        # Scan shared folder
        if self.shared_paths:
            shared_base = self.get_shared_path("approved_originals")
            if shared_base:
                shared_hole_dir = shared_base / project_code / hole_id
                if shared_hole_dir.exists():
                    for f in shared_hole_dir.iterdir():
                        if f.is_file():
                            match = pattern.match(f.name)
                            if match:
                                num = match.group(1)
                                if num:
                                    existing_numbers.add(int(num))
                                else:
                                    # Legacy file without number = 1
                                    existing_numbers.add(1)
        
        # Scan local folder
        local_dir = self.get_hole_dir("approved_originals", hole_id)
        if os.path.exists(local_dir):
            for f in os.listdir(local_dir):
                match = pattern.match(f)
                if match:
                    num = match.group(1)
                    if num:
                        existing_numbers.add(int(num))
                    else:
                        existing_numbers.add(1)
        
        # Find next available number
        if not existing_numbers:
            return 1
        
        next_num = 1
        while next_num in existing_numbers:
            next_num += 1
        
        return next_num

    def delete_approved_original_by_uid(self, hole_id: str, uid: str) -> bool:
        """
        Delete an approved original file by its UID.
        
        Searches both local and shared approved_originals folders.
        The UID is stored in EXIF metadata, not the filename.
        
        Args:
            hole_id: Hole identifier
            uid: UID to search for
            
        Returns:
            True if file was found and deleted
        """
        project_code = hole_id[:2].upper()
        deleted = False
        
        # Search and delete from local
        local_dir = self.get_hole_dir("approved_originals", hole_id)
        if os.path.exists(local_dir):
            for filename in os.listdir(local_dir):
                filepath = os.path.join(local_dir, filename)
                if os.path.isfile(filepath):
                    file_uid = self.extract_uid_from_any_image(filepath)
                    if file_uid and file_uid.lower() == uid.lower():
                        try:
                            os.remove(filepath)
                            self.logger.info(f"Deleted local approved original: {filename}")
                            deleted = True
                        except Exception as e:
                            self.logger.error(f"Failed to delete local original {filename}: {e}")
        
        # Search and delete from shared
        if self.shared_paths:
            shared_base = self.get_shared_path("approved_originals")
            if shared_base:
                shared_hole_dir = shared_base / project_code / hole_id
                if shared_hole_dir.exists():
                    for f in shared_hole_dir.iterdir():
                        if f.is_file():
                            file_uid = self.extract_uid_from_any_image(str(f))
                            if file_uid and file_uid.lower() == uid.lower():
                                try:
                                    f.unlink()
                                    self.logger.info(f"Deleted shared approved original: {f.name}")
                                    deleted = True
                                except Exception as e:
                                    self.logger.error(f"Failed to delete shared original {f.name}: {e}")
        
        return deleted

    # ===== UID methods =====

    def embed_uid_in_image(self, image_path: str, uid: str = None) -> str:
        """
        Embed a UUID into the EXIF ImageUniqueID tag of a JPEG image.

        Args:
            image_path: Path to the JPEG image
            uid: UUID to embed (if None, generates a new one)

        Returns:
            The UID that was embedded
        """
        try:
            # Generate UID if not provided
            if uid is None:
                uid = str(uuid.uuid4())

            # Only process JPEG files
            if not image_path.lower().endswith((".jpg", ".jpeg")):
                self.logger.warning(f"Cannot embed UID in non-JPEG file: {image_path}")
                return uid

            # Load existing EXIF data
            try:
                exif_dict = piexif.load(image_path)
            except Exception as e:
                # Create new EXIF dict if none exists
                self.logger.debug(f"No existing EXIF data in {image_path}: {e}")
                exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}

            # Embed the UID in the ImageUniqueID field (0xA420)
            exif_dict["Exif"][piexif.ExifIFD.ImageUniqueID] = uid.encode("ascii")

            # Also store in ImageDescription for redundancy
            exif_dict["0th"][piexif.ImageIFD.ImageDescription] = f"UID:{uid}".encode(
                "utf-8"
            )

            # Dump EXIF data and insert into image
            exif_bytes = piexif.dump(exif_dict)
            piexif.insert(exif_bytes, image_path)

            self.logger.info(f"Embedded UID {uid} into {os.path.basename(image_path)}")
            return uid

        except Exception as e:
            self.logger.error(f"Error embedding UID in image: {e}")
            # Return the UID anyway so processing can continue
            return uid if uid else str(uuid.uuid4())

    def extract_uid_from_image(self, image_path: str) -> Optional[str]:
        """
        Extract the UID from an image's EXIF data.

        Args:
            image_path: Path to the image

        Returns:
            The UID if found, None otherwise
        """
        try:
            exif_dict = piexif.load(image_path)

            # Try to get from ImageUniqueID first
            if piexif.ExifIFD.ImageUniqueID in exif_dict.get("Exif", {}):
                uid_bytes = exif_dict["Exif"][piexif.ExifIFD.ImageUniqueID]
                return uid_bytes.decode("ascii") if uid_bytes else None

            # Fallback to ImageDescription
            if piexif.ImageIFD.ImageDescription in exif_dict.get("0th", {}):
                desc = exif_dict["0th"][piexif.ImageIFD.ImageDescription].decode(
                    "utf-8"
                )
                if desc.startswith("UID:"):
                    return desc[4:]

        except Exception as e:
            self.logger.debug(f"Could not extract UID from {image_path}: {e}")

        return None

    def embed_uid_in_png(self, image_array: np.ndarray, uid: str) -> bytes:
        """
        Embed UID into PNG metadata and return PNG bytes.

        Args:
            image_array: OpenCV image array (BGR)
            uid: UID to embed

        Returns:
            PNG image bytes with embedded UID
        """
        try:
            # Convert BGR to RGB for PIL
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                rgb_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            else:
                rgb_array = image_array

            # Convert to PIL Image
            pil_image = PILImage.fromarray(rgb_array)

            # Create PNG metadata
            png_info = PngInfo()
            png_info.add_text("SourceUID", uid)
            png_info.add_text("Description", f"Source Image UID: {uid}")

            # Save to bytes buffer
            from io import BytesIO

            buffer = BytesIO()
            pil_image.save(buffer, format="PNG", pnginfo=png_info)
            return buffer.getvalue()

        except Exception as e:
            self.logger.error(f"Error embedding UID in PNG: {e}")
            return None

    def save_png_with_uid(
        self, image_array: np.ndarray, file_path: str, uid: str
    ) -> bool:
        """
        Save PNG image with embedded UID.

        Args:
            image_array: OpenCV image array
            file_path: Where to save the PNG
            uid: UID to embed

        Returns:
            True if successful
        """
        try:
            png_bytes = self.embed_uid_in_png(image_array, uid)
            if png_bytes:
                with open(file_path, "wb") as f:
                    f.write(png_bytes)
                return True
            else:
                # Fallback to regular save
                cv2.imwrite(file_path, image_array)
                return True
        except Exception as e:
            self.logger.error(f"Error saving PNG with UID: {e}")
            return False

    def extract_uid_from_png(self, png_path: str) -> Optional[str]:
        """Extract UID from PNG metadata."""
        try:
            img = PILImage.open(png_path)
            if "SourceUID" in img.info:
                return img.info["SourceUID"]
            elif "Description" in img.info:
                desc = img.info["Description"]
                if desc.startswith("Source Image UID: "):
                    return desc[18:]
        except Exception as e:
            self.logger.debug(f"Could not extract UID from PNG: {e}")
        return None

    def embed_uid_in_any_image(self, image_path: str, uid: str = None) -> str:
        """
        Embed a UUID into any image format using appropriate metadata method.

        Args:
            image_path: Path to the image
            uid: UUID to embed (if None, generates a new one)

        Returns:
            The UID that was embedded
        """
        try:
            # Generate UID if not provided
            if uid is None:
                uid = str(uuid.uuid4())

            ext = os.path.splitext(image_path.lower())[1]

            if ext in [".jpg", ".jpeg"]:
                # Use EXIF for JPEG
                return self.embed_uid_in_image(image_path, uid)

            elif ext == ".png":
                # Use PNG text chunks
                img = PILImage.open(image_path)
                png_info = PngInfo()
                png_info.add_text("SourceUID", uid)
                png_info.add_text("Description", f"Source Image UID: {uid}")

                # Save with metadata
                img.save(image_path, pnginfo=png_info)
                self.logger.info(
                    f"Embedded UID {uid} in PNG: {os.path.basename(image_path)}"
                )
                return uid

            elif ext in [".tif", ".tiff"]:
                # TIFF can use EXIF similar to JPEG
                try:
                    exif_dict = piexif.load(image_path)
                except Exception as e:
                    self.logger.debug(
                        f"No existing EXIF data in TIFF {image_path}: {e}"
                    )
                    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}

                # Embed UID in EXIF
                exif_dict["Exif"][piexif.ExifIFD.ImageUniqueID] = uid.encode("ascii")
                exif_dict["0th"][piexif.ImageIFD.ImageDescription] = (
                    f"UID:{uid}".encode("utf-8")
                )

                exif_bytes = piexif.dump(exif_dict)
                piexif.insert(exif_bytes, image_path)

                self.logger.info(
                    f"Embedded UID {uid} in TIFF: {os.path.basename(image_path)}"
                )
                return uid

            elif ext in [".heic", ".heif"]:
                # For HEIC/HEIF, we can use EXIF after conversion or store in a sidecar file
                self.logger.warning(
                    f"HEIC/HEIF UID embedding not fully implemented, storing UID: {uid}"
                )
                # Could implement sidecar file or convert to JPEG first
                return uid

            else:
                self.logger.warning(
                    f"Unknown image format {ext}, cannot embed UID directly: {uid}"
                )
                return uid

        except Exception as e:
            self.logger.error(f"Error embedding UID in image: {e}")
            return uid if uid else str(uuid.uuid4())

    def extract_uid_from_any_image(self, image_path: str) -> Optional[str]:
        """
        Extract UID from any image format.

        Args:
            image_path: Path to the image

        Returns:
            The UID if found, None otherwise
        """
        try:
            ext = os.path.splitext(image_path.lower())[1]

            if ext in [".jpg", ".jpeg"]:
                # Use existing EXIF extraction
                return self.extract_uid_from_image(image_path)

            elif ext == ".png":
                # Extract from PNG text chunks
                return self.extract_uid_from_png(image_path)

            elif ext in [".tif", ".tiff"]:
                # Extract from TIFF EXIF
                try:
                    exif_dict = piexif.load(image_path)

                    # Try ImageUniqueID first
                    if piexif.ExifIFD.ImageUniqueID in exif_dict.get("Exif", {}):
                        uid_bytes = exif_dict["Exif"][piexif.ExifIFD.ImageUniqueID]
                        return uid_bytes.decode("ascii") if uid_bytes else None

                    # Fallback to ImageDescription
                    if piexif.ImageIFD.ImageDescription in exif_dict.get("0th", {}):
                        desc = exif_dict["0th"][
                            piexif.ImageIFD.ImageDescription
                        ].decode("utf-8")
                        if desc.startswith("UID:"):
                            return desc[4:]
                except Exception as e:
                    self.logger.debug(
                        f"Could not extract UID from TIFF {image_path}: {e}"
                    )

            elif ext in [".heic", ".heif"]:
                # Could check sidecar file or other metadata
                self.logger.debug(f"HEIC/HEIF UID extraction not implemented")
                return None

            else:
                self.logger.debug(f"Unknown format {ext} for UID extraction")
                return None

        except Exception as e:
            self.logger.debug(f"Error extracting UID from {image_path}: {e}")
            return None

    def calculate_robust_hex_color(
        self, image_path: str, method: str = "LAB_shadow_compensated"
    ) -> Dict[str, Any]:
        """
        Calculate robust hex color for chip tray image using LAB color space
        to reduce shadow and highlight sensitivity.

        This method uses LAB color space which separates luminance (L*) from
        chromatic content (a*, b*). By masking out dark shadows and bright
        highlights, we get a more consistent color representation of the actual
        chips, not the lighting conditions.

        Args:
            image_path: Path to image file
            method: Calculation method:
                - "LAB_shadow_compensated" (recommended): Uses LAB color space,
                  filters shadows/highlights, more robust for lithology
                - "simple_average": Global RGB average (legacy method)

        Returns:
            Dictionary with:
                - hex_color: Hex color string (e.g., "#8B4513")
                - method: Method used
                - valid: Whether calculation was successful
                - notes: Any warnings or notes about the calculation
        """
        try:
            import cv2
            import numpy as np

            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                return {
                    "hex_color": "",
                    "method": method,
                    "valid": False,
                    "notes": "Failed to read image",
                }

            if method == "LAB_shadow_compensated":
                # Convert to LAB color space
                # LAB separates luminance (L*) from chromatic content (a*, b*)
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                L, A, B = cv2.split(lab)

                # Mask out dark shadows (L < 60) and bright highlights (L > 240)
                # L* ranges from 0 (black) to 255 (white) in OpenCV's representation
                # This keeps only mid-range luminance pixels representing actual chips
                mask = (L > 60) & (L < 240)

                if mask.sum() < 100:  # Too few valid pixels
                    return {
                        "hex_color": "",
                        "method": method,
                        "valid": False,
                        "notes": "Insufficient mid-range pixels (image too dark/bright)",
                    }

                # Calculate mean LAB values from masked region
                mean_L = L[mask].mean()
                mean_A = A[mask].mean()
                mean_B = B[mask].mean()

                # Convert back to RGB
                lab_avg = np.uint8([[[mean_L, mean_A, mean_B]]])
                rgb_avg = cv2.cvtColor(lab_avg, cv2.COLOR_LAB2BGR)[0, 0]

                # Convert BGR to RGB (OpenCV uses BGR ordering)
                rgb_avg_corrected = rgb_avg[::-1]
                hex_color = "#%02X%02X%02X" % tuple(int(x) for x in rgb_avg_corrected)

                return {
                    "hex_color": hex_color,
                    "method": method,
                    "valid": True,
                    "notes": f"Calculated from {mask.sum()} pixels",
                }

            elif method == "simple_average":
                # Simple global average (current/legacy method)
                mean_rgb = img.mean(axis=(0, 1))
                # OpenCV uses BGR, convert to RGB
                mean_rgb_corrected = mean_rgb[::-1]
                hex_color = "#%02X%02X%02X" % tuple(mean_rgb_corrected.astype(int))

                return {
                    "hex_color": hex_color,
                    "method": method,
                    "valid": True,
                    "notes": "Global average of all pixels (legacy method)",
                }

            else:
                return {
                    "hex_color": "",
                    "method": method,
                    "valid": False,
                    "notes": f"Unknown method: {method}",
                }

        except Exception as e:
            self.logger.error(f"Error calculating hex color for {image_path}: {e}")
            return {
                "hex_color": "",
                "method": method,
                "valid": False,
                "notes": f"Error: {str(e)}",
            }

    def copy_with_metadata(self, source: str, destination: str) -> bool:
        """
        Copy a file preserving all metadata including EXIF data for images.

        Args:
            source: Source file path
            destination: Destination file path

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # First, copy the file content and basic metadata
            shutil.copy2(source, destination)

            # For image files, ensure EXIF data is preserved
            image_extensions = {".jpg", ".jpeg", ".tiff", ".tif", ".png"}
            _, ext = os.path.splitext(source.lower())

            if ext in image_extensions:
                try:
                    if ext.lower() in [".jpg", ".jpeg", ".tif", ".tiff"]:
                        # Read EXIF data from source
                        exif_dict = piexif.load(source)

                        # Write EXIF data to destination if available
                        if exif_dict:
                            piexif.insert(piexif.dump(exif_dict), destination)
                            self.logger.debug(
                                f"Preserved EXIF metadata for {destination}"
                            )

                    elif ext.lower() == ".png":
                        # Extract and preserve PNG metadata
                        src_uid = self.extract_uid_from_png(source)
                        if src_uid:
                            # Re-save with UID
                            img = cv2.imread(destination)
                            if img is not None:
                                self.save_png_with_uid(img, destination, src_uid)
                                self.logger.debug(
                                    f"Preserved PNG UID for {destination}"
                                )

                except Exception as exif_error:
                    self.logger.warning(
                        f"Could not preserve image metadata: {exif_error}"
                    )

            # Copy additional file attributes
            try:
                shutil.copystat(source, destination)
            except Exception as stat_error:
                self.logger.warning(f"Could not copy all file stats: {stat_error}")

            return True

        except Exception as e:
            self.logger.error(f"Error copying file with metadata: {e}")
            return False
