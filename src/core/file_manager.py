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
|   |           | ├── [HOLE_ID]_CC_001.[ext]
|   |           | └── [HOLE_ID]_CC_002.[ext]
│   |           └── 📁 With_Data/
│   |                └── [HOLE_ID]_CC_001_Data.[ext]
|   └── 📁 [Compartment Images for Review]/
│      └── 📁 [PROJECT CODE]/
|           └── 📁 [HOLE_ID]/
|                ├── [HOLE_ID]_CC_001_{suffix}.[ext]
|                └── [HOLE_ID]_CC_002_{suffix}.[ext]
├── 📁 Drillhole Traces/
│      └── 📁 [PROJECT CODE]/
|           └── 📁 [HOLE_ID]/
|               └── [HOLE_ID]_trace.[ext]
└── 📁 Debugging/
    ├── 📁 Blur Analysis/
    │      └── 📁 [PROJECT CODE]/
               └── 📁 [HOLE_ID]/
    │               └── [HOLE_ID]_[COMPARTMENT]_blur_analysis.jpg
    └── 📁 Debug Images/
        ├── 📁 Unidentified/
        │   └── [ORIGINAL_NAME]_[DEBUG_TYPE].jpg
        └── 📁 [PROJECT CODE]/
            └── 📁 [HOLE_ID]/
                └── [HOLE_ID]_[FROM]-[TO]_[DEBUG_TYPE].jpg

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
import piexif



class FileManager:
    """
    Manages file operations for the Chip Tray Extractor.
    
    Handles directory creation, file naming conventions, and saving operations
    for both local and shared (OneDrive) folders.
    """
    
    # Define folder structure constants
    FOLDER_NAMES = {
        'register': 'Chip Tray Register',
        'images': 'Images to Process',
        'processed': 'Processed Original Images',
        'compartments': 'Extracted Compartment Images',
        'traces': 'Drillhole Traces',
        'debugging': 'Debugging'
    }
    
    SUBFOLDER_NAMES = {
        'approved_originals': 'Approved Originals',
        'rejected_originals': 'Rejected Originals',
        'approved_compartments': 'Approved Compartment Images',
        'review_compartments': 'Compartment Images for Review',
        'blur': 'Blur Analysis',
        'debug': 'Debug Images',
        'register_data': 'Register Data (Do not edit)', # folder containing the register datasets and automatic excel file.
        'with_data': 'With_Data' # Individual chip compartments with plots / visualisations / datasets
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
        self.base_dir = Path(base_dir) if base_dir else Path("C:/GeoVue Chip Tray Photos")
        
    
        # NEW: Initialize shared folder paths
    
        self.shared_base_dir = None
        self.shared_paths = {}
        
    
        # Use constants for folder names
    
        self.dir_structure = {
            "local_output_folder": self.base_dir,
            "images_to_process": self.base_dir / self.FOLDER_NAMES['images'],
            "processed_originals": self.base_dir / self.FOLDER_NAMES['processed'],
            "approved_originals": self.base_dir / self.FOLDER_NAMES['processed'] / self.SUBFOLDER_NAMES['approved_originals'],
            "rejected_originals": self.base_dir / self.FOLDER_NAMES['processed'] / self.SUBFOLDER_NAMES['rejected_originals'],
            "chip_compartments": self.base_dir / self.FOLDER_NAMES['compartments'],
            "approved_compartments": self.base_dir / self.FOLDER_NAMES['compartments'] / self.SUBFOLDER_NAMES['approved_compartments'],
            "temp_review": self.base_dir / self.FOLDER_NAMES['compartments'] / self.SUBFOLDER_NAMES['review_compartments'],
            "drill_traces": self.base_dir / self.FOLDER_NAMES['traces'],
            "debugging": self.base_dir / self.FOLDER_NAMES['debugging'],
            "blur_analysis": self.base_dir / self.FOLDER_NAMES['debugging'] / self.SUBFOLDER_NAMES['blur'],
            "debug_images": self.base_dir / self.FOLDER_NAMES['debugging'] / self.SUBFOLDER_NAMES['debug']
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
            shared_path = self.config_manager.get('shared_folder_path')
            if shared_path and os.path.exists(shared_path):
                self.shared_base_dir = Path(shared_path)
        
        # Build shared paths structure if we have a base directory
        if self.shared_base_dir:
            self.shared_paths = {
                "register": self.shared_base_dir / self.FOLDER_NAMES['register'],
                "register_excel": self.shared_base_dir / self.FOLDER_NAMES['register'] / self.EXCEL_REGISTER_NAME,
                "register_data": self.shared_base_dir / self.FOLDER_NAMES['register'] / self.SUBFOLDER_NAMES['register_data'],
                "images_to_process": self.shared_base_dir / self.FOLDER_NAMES['images'],
                "processed_originals": self.shared_base_dir / self.FOLDER_NAMES['processed'],
                "approved_originals": self.shared_base_dir / self.FOLDER_NAMES['processed'] / self.SUBFOLDER_NAMES['approved_originals'],
                "rejected_originals": self.shared_base_dir / self.FOLDER_NAMES['processed'] / self.SUBFOLDER_NAMES['rejected_originals'],
                "compartments": self.shared_base_dir / self.FOLDER_NAMES['compartments'],
                "approved_compartments": self.shared_base_dir / self.FOLDER_NAMES['compartments'] / self.SUBFOLDER_NAMES['approved_compartments'],
                "review_compartments": self.shared_base_dir / self.FOLDER_NAMES['compartments'] / self.SUBFOLDER_NAMES['review_compartments'],
                "drill_traces": self.shared_base_dir / self.FOLDER_NAMES['traces']
            }
            
            # Save individual paths to config if available
            if self.config_manager:
                self.config_manager.set('shared_folder_register_path', str(self.shared_paths['register']))
                self.config_manager.set('shared_folder_register_excel_path', str(self.shared_paths['register_excel']))
                self.config_manager.set('shared_folder_register_data_folder', str(self.shared_paths['register_data']))
                self.config_manager.set('shared_folder_processed_originals', str(self.shared_paths['processed_originals']))
                self.config_manager.set('shared_folder_approved_folder', str(self.shared_paths['approved_originals']))
                self.config_manager.set('shared_folder_rejected_folder', str(self.shared_paths['rejected_originals']))
                self.config_manager.set('shared_folder_extracted_compartments_folder', str(self.shared_paths['compartments']))
                self.config_manager.set('shared_folder_approved_compartments_folder', str(self.shared_paths['approved_compartments']))
                self.config_manager.set('shared_folder_review_compartments_folder', str(self.shared_paths['review_compartments']))
                self.config_manager.set('shared_folder_drill_traces', str(self.shared_paths['drill_traces']))
            
            self.logger.info(f"Initialized shared paths with base: {self.shared_base_dir}")

    def get_shared_path(self, path_key: str, create_if_missing: bool = True) -> Optional[Path]:
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
        if create_if_missing and not path_key.endswith('_excel') and not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created shared directory: {path}")
            except Exception as e:
                self.logger.error(f"Failed to create shared directory {path}: {e}")
                return None
                
        return path if path.exists() else None

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
                        'register': 'shared_folder_register_path',
                        'register_excel': 'shared_folder_register_excel_path',
                        'register_data': 'shared_folder_register_data_folder',
                        'processed_originals': 'shared_folder_processed_originals',
                        'approved_originals': 'shared_folder_approved_folder',
                        'rejected_originals': 'shared_folder_rejected_folder',
                        'compartments': 'shared_folder_extracted_compartments_folder',
                        'approved_compartments': 'shared_folder_approved_compartments_folder',
                        'review_compartments': 'shared_folder_review_compartments_folder',
                        'drill_traces': 'shared_folder_drill_traces'
                    }
                    
                    if path_key in config_key_map:
                        self.config_manager.set(config_key_map[path_key], new_path)
                        
                self.logger.info(f"Updated shared path {path_key}: {new_path}")
                return True
            else:
                self.logger.warning(f"Unknown path key: {path_key}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error updating shared path {path_key}: {e}")
            return False

    def prompt_for_shared_path(self, path_key: str, dialog_title: str, 
                             missing_message: str, is_file: bool = False) -> Optional[str]:
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
        initial_dir = str(current_path.parent) if current_path and current_path.exists() else None
        
        # Create root if needed
        root = tk.Tk()
        root.withdraw()
        
        try:
            # Show message if path missing
            if missing_message and (not current_path or not current_path.exists()):
                result = DialogHelper.ask_yes_no(
                    root,
                    DialogHelper.t("Path Not Found"),
                    DialogHelper.t(missing_message)
                )
                if not result:
                    return None
            
            # Show file/folder dialog
            if is_file:
                path = filedialog.askopenfilename(
                    title=DialogHelper.t(dialog_title),
                    initialdir=initial_dir,
                    filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")] if path_key == 'register_excel' else []
                )
            else:
                path = filedialog.askdirectory(
                    title=DialogHelper.t(dialog_title),
                    initialdir=initial_dir
                )
            
            if path:
                # Update the path
                self.update_shared_path(path_key, path)
                return path
                
        finally:
            root.destroy()
            
        return None

    def _get_local_path(self, hole_id: str, depth_from: float, depth_to: float, 
                            is_processed: bool = True, is_rejected: bool = False) -> str:
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
    
    def _verify_upload(self, upload_path: Union[str, Path], max_retries: int = 3) -> bool:
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
                self.logger.warning(f"Upload verification attempt {attempt + 1} failed: {e}")
                
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
                            if os.path.exists(new_path) and os.path.getsize(new_path) == os.path.getsize(old_path):
                                os.remove(old_path)
                                self.logger.info(f"Renamed file via copy: {old_path} -> {new_path}")
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
            pattern = r'([A-Z]{2}\d{4})_(\d+)-(\d+)(?:_Original)?(?:_Skipped)?'
            
            # Try to extract metadata directly from filename
            match = re.search(pattern, base_name)
            if match:
                hole_id = match.group(1)
                depth_from = float(match.group(2))
                depth_to = float(match.group(3))
                
                # Check against valid prefixes if the configuration exists and prefix validation is enabled
                valid_prefixes = self.config_manager.get('valid_hole_prefixes', []) if self.config_manager else []
                enable_prefix_validation = self.config_manager.get('enable_prefix_validation', False) if self.config_manager else False
                
                # If prefix validation is enabled, check against valid prefixes
                if enable_prefix_validation and valid_prefixes:
                    prefix = hole_id[:2].upper()
                    if prefix not in valid_prefixes:
                        self.logger.warning(f"Hole ID prefix {prefix} not in valid prefixes: {valid_prefixes}")
                        return None
                
                self.logger.info(f"Extracted metadata from filename: {hole_id}, {depth_from}-{depth_to}")
                return {
                    'hole_id': hole_id,
                    'depth_from': depth_from,
                    'depth_to': depth_to,
                    'confidence': 100.0,  # High confidence for filename-derived metadata
                    'from_filename': True
                }
            
            return None
                
        except Exception as e:
            self.logger.error(f"Error extracting metadata from filename {filename}: {str(e)}")
            return None

    def check_original_file_processed(self, original_filename: str) -> Optional[Dict[str, Any]]:
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
            pattern = r'([A-Z]{2}\d{4})_(\d+)-(\d+)_Original(?:_Skipped)?'
            
            # Try to extract metadata directly from filename
            match = re.search(pattern, base_name)
            if match:
                hole_id = match.group(1)
                depth_from = int(match.group(2))
                depth_to = int(match.group(3))
                
                # Check against valid prefixes if the configuration exists and prefix validation is enabled
                valid_prefixes = getattr(self, 'config', {}).get('valid_hole_prefixes', [])
                enable_prefix_validation = getattr(self, 'config', {}).get('enable_prefix_validation', False)
                
                # If prefix validation is enabled, check against valid prefixes
                if enable_prefix_validation and valid_prefixes:
                    prefix = hole_id[:2].upper()
                    if prefix not in valid_prefixes:
                        self.logger.warning(f"Hole ID prefix {prefix} not in valid prefixes: {valid_prefixes}")
                        return None
                
                # Determine if the file was previously skipped
                is_skipped = '_Skipped' in base_name
                
                self.logger.info(f"Found metadata in filename: {hole_id}, {depth_from}-{depth_to}")
                return {
                    'hole_id': hole_id,
                    'depth_from': depth_from,
                    'depth_to': depth_to,
                    'confidence': 100.0 if not is_skipped else 90.0,  # Slightly lower confidence for skipped files
                    'from_filename': True,
                    'previously_skipped': is_skipped
                }
            
            # No match found
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking for previously processed file: {str(e)}")
            return None
    

    def save_compartment(self, 
                        image: np.ndarray, 
                        hole_id: str, 
                        compartment_num: int,
                        has_data: bool = False,
                        output_format: str = "png") -> str:
        """
        Save a compartment image.
        
        Args:
            image: Image to save
            hole_id: Hole ID
            compartment_num: Compartment number
            has_data: Whether the image has data columns
            output_format: Output image format
            
        Returns:
            Path to the saved file
        """
        try:
            # Get the appropriate directory
            save_dir = self.get_hole_dir("approved_compartments", hole_id)
            
            # Create filename with 3-digit compartment number (001, 002, etc.)
            if has_data:
                filename = f"{hole_id}_CC_{compartment_num:03d}_Data.{output_format}"
            else:
                filename = f"{hole_id}_CC_{compartment_num:03d}.{output_format}"
            
            # Full path
            file_path = os.path.join(save_dir, filename)
            
            # Save the image
            if output_format.lower() == "jpg":
                cv2.imwrite(file_path, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
            else:
                cv2.imwrite(file_path, image)
            
            self.logger.info(f"Saved compartment image: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error saving compartment image: {str(e)}")
            return None

    def save_reviewed_compartment(self, image: np.ndarray, hole_id: str, 
                                compartment_depth: int, status: str,
                                output_format: str = "png") -> Dict[str, Any]:
        """
        Save a reviewed compartment with appropriate suffix and upload to shared folder.
        
        Args:
            image: Compartment image
            hole_id: Hole ID
            compartment_depth: Compartment depth number
            status: Review status ("Wet", "Dry", "Blurry", etc.)
            output_format: Output format
            
        Returns:
            Dictionary with save results
        """
        result = {
            'local_path': None,
            'shared_path': None,
            'upload_success': False
        }
        
        try:
            # Skip if keeping original or missing
            if status in ["KEEP_ORIGINAL", "MISSING"]:
                return result
            
            # Save locally first
            local_path = self.save_compartment(
                image, hole_id, compartment_depth,
                has_data=False, output_format=output_format
            )
            
            if local_path:
                # Add suffix for wet/dry
                if status in ["Wet", "Dry"]:
                    suffix = f"_{status}"
                    base, ext = os.path.splitext(local_path)
                    new_local_path = f"{base}{suffix}{ext}"
                    os.rename(local_path, new_local_path)
                    local_path = new_local_path
                
                result['local_path'] = local_path
                
                # Upload to shared folder if configured
                shared_path = self.get_shared_path('approved_compartments', create_if_missing=True)
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
                            
                            result['local_path'] = uploaded_path
                            result['shared_path'] = str(shared_file_path)
                            result['upload_success'] = True
                            
        except Exception as e:
            self.logger.error(f"Error saving reviewed compartment: {e}")
            
        return result

    def save_compartment_with_data(self, 
                                image: np.ndarray, 
                                hole_id: str, 
                                compartment_num: int,
                                output_format: str = "tiff") -> str:
        """
        Save a compartment image with data columns to a 'With_Data' subfolder.
        
        Args:
            image: Image to save
            hole_id: Hole ID
            compartment_num: Compartment number
            output_format: Output image format
            
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
            
            # Save the image
            if output_format.lower() == "jpg":
                cv2.imwrite(file_path, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
            else:
                cv2.imwrite(file_path, image)
            
            self.logger.info(f"Saved compartment with data image: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error saving compartment with data image: {str(e)}")
            return None
    
    def save_temp_compartment(self, image: np.ndarray, hole_id: str, 
                            compartment_depth: int, suffix: str = "temp") -> str:
        """Save a temporary compartment image for review."""
        try:
            # Get the temp review directory
            save_dir = self.get_hole_dir("temp_review", hole_id)
            
            # Create filename
            filename = f"{hole_id}_CC_{compartment_depth:03d}_{suffix}.png"
            file_path = os.path.join(save_dir, filename)
            
            # Save the image
            cv2.imwrite(file_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
            self.logger.info(f"Saved temp compartment: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error saving temp compartment: {str(e)}")
            return None

    def cleanup_temp_compartments(self, hole_id: str, temp_paths: List[str]) -> None:
        """
        Clean up temporary compartment files after processing.
        
        Args:
            hole_id: Hole ID to clean up
            temp_paths: List of temporary file paths to remove
        """
        try:
            # Remove individual temp files
            for temp_path in temp_paths:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                        self.logger.debug(f"Removed temp file: {temp_path}")
                    except Exception as e:
                        self.logger.warning(f"Could not remove temp file {temp_path}: {e}")
            
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


    def rename_debug_files(self, 
                        original_filename: str, 
                        hole_id: Optional[str], 
                        depth_from: Optional[float], 
                        depth_to: Optional[float]) -> None:
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
                    f for f in os.listdir(debug_dir) 
                    if f.startswith(base_name) and f.endswith('.jpg')
                ]
            except FileNotFoundError:
                # Directory might not exist yet
                os.makedirs(debug_dir, exist_ok=True)
                debug_files = []
                self.logger.info(f"Created debug directory for {hole_id}")
            
            # Also look for temp debug images in the Unidentified folder
            temp_debug_dir = os.path.join(self.dir_structure["debug_images"], "Unidentified")
            if os.path.exists(temp_debug_dir):
                try:
                    temp_files = [
                        f for f in os.listdir(temp_debug_dir)
                        if f.startswith(base_name) and f.endswith('.jpg')
                    ]
                    
                    # Move and rename temp files
                    for old_filename in temp_files:
                        # Extract the step name from the old filename
                        if '_' in old_filename:
                            step_name = old_filename.split('_', 1)[1].replace('.jpg', '')
                            
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
                                    self.logger.info(f"Moved debug file from temp location: {old_filename} -> {new_filename}")
                                else:
                                    self.logger.error(f"Failed to copy debug file with metadata: {old_filename}")
                                
                                # Only remove original after successful copy
                                os.remove(old_path)
                                
                                self.logger.info(f"Moved debug file from temp location: {old_filename} -> {new_filename}")
                            except Exception as e:
                                self.logger.error(f"Error moving temp debug file {old_filename}: {e}")
                except Exception as e:
                    self.logger.error(f"Error processing temp debug files: {e}")
            
            # Rename files in the debug directory
            for old_filename in debug_files:
                try:
                    # Extract the step name from the old filename
                    step_parts = old_filename.split('_')
                    if len(step_parts) >= 2:
                        step_name = step_parts[-1].replace('.jpg', '')
                        
                        # Generate new filename with metadata
                        new_filename = f"{hole_id}_{int(depth_from)}-{int(depth_to)}_Debug_{step_name}.jpg"
                        
                        old_path = os.path.join(debug_dir, old_filename)
                        new_path = os.path.join(debug_dir, new_filename)
                        
                        # Rename the file
                        os.rename(old_path, new_path)
                        self.logger.info(f"Renamed debug file: {old_filename} -> {new_filename}")
                except Exception as e:
                    self.logger.error(f"Error renaming debug file {old_filename}: {e}")
        except Exception as e:
            self.logger.error(f"Error in rename_debug_files: {e}")

    def save_temp_debug_image(self, image: np.ndarray, original_filename: str, debug_type: str) -> str:
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
                self.logger.info(f"Saved debug image to fallback location: {fallback_path}")
                return fallback_path
            except Exception as fallback_error:
                self.logger.error(f"Failed to save debug image to fallback location: {str(fallback_error)}")
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
                self.logger.info(f"✅ Base directory structure created at: {self.base_dir}")
            else:
                self.logger.debug(f"📁 All directories already existed at: {self.base_dir}")

        except Exception as e:
            self.logger.error(f"❌ Error creating directory structure: {e}")
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
    
    def save_blur_analysis(self, 
                        image: np.ndarray, 
                        hole_id: str, 
                        compartment_num: int) -> str:
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

    def save_debug_image(self, 
                       image: np.ndarray, 
                       hole_id: str, 
                       depth_from: float,
                       depth_to: float,
                       image_type: str) -> str:
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
            filename = f"{hole_id}_{int(depth_from)}-{int(depth_to)}_Debug_{image_type}.jpg"
            
            # Full path
            file_path = os.path.join(save_dir, filename)
            
            # Save the image
            cv2.imwrite(file_path, image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            self.logger.info(f"Saved debug image: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error saving debug image: {str(e)}")
            return None
    
    def save_drill_trace(self, 
                       image: np.ndarray, 
                       hole_id: str) -> str:
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
        
    def save_original_file(self, 
                    source_path: str, 
                    hole_id: str, 
                    depth_from: int,
                    depth_to: int,
                    is_processed: bool = True,
                    is_rejected: bool = False,
                    is_selective: bool = False,
                    is_skipped: bool = False,
                    delete_source: bool = True) -> Tuple[Optional[str], bool]:
        """
        Save original file by uploading to shared folder (if available) and moving to local storage.
        This method handles both operations atomically to avoid file loss and preserves metadata.
        
        Args:
            source_path: Path to the source file
            hole_id: Hole ID
            depth_from: Starting depth
            depth_to: Ending depth
            onedrive_manager: DEPRECATED - no longer used, paths handled internally
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
            else:
                base_suffix = "_Original"
            
            # Variables to track uploads
            shared_upload_path = None
            upload_attempted = False
            
            # Start shared folder upload first (using internal shared_paths)
            # Use internal shared paths configuration
            if self.shared_paths and is_processed:
                try:
                    # Use get_shared_path() method
                    if is_rejected:
                        shared_base = self.get_shared_path('rejected_originals')
                    else:
                        # For approved/processed originals
                        shared_base = self.get_shared_path('approved_originals')
                    
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
                        base_shared_filename = f"{hole_id}_{int(depth_from)}-{int(depth_to)}{base_suffix}"
                        shared_filename = f"{base_shared_filename}{ext}"
                        shared_upload_path = hole_folder / shared_filename
                        
                        # If file exists, add incrementing number
                        counter = 1
                        while shared_upload_path.exists():
                            shared_filename = f"{base_shared_filename}_{counter}{ext}"
                            shared_upload_path = hole_folder / shared_filename
                            counter += 1
                        
                        # Copy to shared folder
                        if self.copy_with_metadata(source_path, str(shared_upload_path)):
                            self.logger.info(f"Started shared folder upload: {shared_upload_path}")
                        else:
                            self.logger.error(f"Failed to start shared folder copy")
                            shared_upload_path = None
                    else:
                        self.logger.info("Shared folder path not configured for this file type")
                        upload_attempted = False
                            
                except Exception as e:
                    self.logger.error(f"Shared folder upload initiation failed: {str(e)}")
                    shared_upload_path = None
            
            # Now handle local storage
            # Determine local storage directory
            if is_rejected:
                target_dir = self.get_hole_dir("rejected_originals", hole_id)
            elif is_skipped:
                # Skipped files go to rejected folder but with different suffix
                target_dir = self.get_hole_dir("rejected_originals", hole_id)
            else:
                target_dir = self.get_hole_dir("approved_originals", hole_id)
            
            # Create local filename (without suffix for now)
            base_local_filename = f"{hole_id}_{int(depth_from)}-{int(depth_to)}{base_suffix}"
            local_filename = f"{base_local_filename}{ext}"
            target_path = os.path.join(target_dir, local_filename)
            
            # Handle existing files
            counter = 1
            while os.path.exists(target_path):
                local_filename = f"{base_local_filename}_{counter}{ext}"
                target_path = os.path.join(target_dir, local_filename)
                counter += 1
            
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
                        if abs(target_size - source_size) < 1000:  # Within 1KB difference
                            local_success = True
                            self.logger.info(f"Local copy successful: {target_path}")
                        else:
                            self.logger.error(f"Local copy size mismatch: source={source_size}, target={target_size}")
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
                    # Optional: Give shared folder a moment to sync
                    # import time
                    # time.sleep(1)  # 1 second delay
                    
                    # Verify the upload
                    if shared_upload_path.exists() and shared_upload_path.stat().st_size == os.path.getsize(target_path):
                        upload_success = True
                        self.logger.info(f"Shared folder upload verified: {shared_upload_path}")
                    else:
                        self.logger.warning(f"Shared folder upload verification failed - file may still be syncing")
                        
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
                    new_base_filename = f"{hole_id}_{int(depth_from)}-{int(depth_to)}{final_suffix}"
                    new_filename = f"{new_base_filename}{ext}"
                    new_target_path = os.path.join(os.path.dirname(target_path), new_filename)
                    
                    # Handle existing files with new name
                    counter = 1
                    while os.path.exists(new_target_path):
                        new_filename = f"{new_base_filename}_{counter}{ext}"
                        new_target_path = os.path.join(os.path.dirname(target_path), new_filename)
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
                has_shared = upload_success and shared_upload_path and shared_upload_path.exists()
                
                if has_local or has_shared:
                    try:
                        os.remove(source_path)
                        self.logger.info(f"Source file deleted after successful save: {source_path}")
                    except Exception as e:
                        self.logger.error(f"Failed to delete source file: {str(e)}")
                        # Try alternative deletion method
                        try:
                            import shutil
                            shutil.move(source_path, source_path + ".processed")
                            self.logger.info(f"Renamed source file to .processed: {source_path}")
                        except:
                            pass
                else:
                    self.logger.warning(f"Source file not deleted - no successful saves confirmed")
                    self.logger.warning(f"Local save: {has_local}, Shared save: {has_shared}")

            if local_success or upload_success:
                return target_path, upload_success
            else:
                # Both saves failed - try emergency move
                self.logger.warning("Both saves failed, attempting emergency move...")
                emergency_path = self.move_original_file(
                    source_path, hole_id, depth_from, depth_to,
                    is_processed, is_rejected, is_selective, is_skipped,
                    suffix="EMERGENCY_SAVE"
                )
                if emergency_path:
                    return emergency_path, False
                else:
                    return None, False

                
        except Exception as e:
            self.logger.error(f"Error in save_original_file: {str(e)}")
            return None, False

    def move_original_file(self, source_path: str, hole_id: str, depth_from: float, depth_to: float,
                        is_processed: bool = True, is_rejected: bool = False, is_selective: bool = False, is_skipped: bool = False,
                        suffix: str = None) -> Optional[str]:
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
            new_filename = f"{hole_id}_{int(depth_from)}-{int(depth_to)}{full_suffix}{ext}"
            
            target_path = os.path.join(target_dir, new_filename)
            
            # Handle existing files
            counter = 1
            while os.path.exists(target_path):
                new_filename = f"{hole_id}_{int(depth_from)}-{int(depth_to)}{full_suffix}_{counter}{ext}"
                target_path = os.path.join(target_dir, new_filename)
                counter += 1
            
            # Copy with metadata preservation
            if self.copy_with_metadata(source_path, target_path):
                # Verify the copy was successful
                if os.path.exists(target_path) and os.path.getsize(target_path) == os.path.getsize(source_path):
                    # Delete source file after successful copy
                    try:
                        os.remove(source_path)
                        self.logger.info(f"Emergency moved file to: {target_path}")
                        return target_path
                    except Exception as e:
                        self.logger.error(f"Failed to delete source after emergency copy: {e}")
                        # Still return target path - we have the file saved
                        return target_path
                else:
                    self.logger.error("Emergency copy verification failed")
                    # Clean up failed copy
                    if os.path.exists(target_path):
                        try:
                            os.remove(target_path)
                        except:
                            pass
                    return None
            else:
                self.logger.error("Emergency copy_with_metadata failed")
                return None
            
        except Exception as e:
            self.logger.error(f"Emergency move failed: {str(e)}")
            return None
               
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
            image_extensions = {'.jpg', '.jpeg', '.tiff', '.tif', '.png'}
            _, ext = os.path.splitext(source.lower())
            
            if ext in image_extensions:
                try:
                    # Read EXIF data from source
                    exif_dict = piexif.load(source)
                    
                    # Write EXIF data to destination
                    if exif_dict:
                        piexif.insert(piexif.dump(exif_dict), destination)
                        self.logger.debug(f"Preserved EXIF metadata for {destination}")
                        
                except Exception as exif_error:
                    # If EXIF preservation fails, log but don't fail the copy
                    self.logger.warning(f"Could not preserve EXIF data: {exif_error}")
                    # The file is still copied, just without EXIF
            
            # Copy additional file attributes
            try:
                # Copy all stat info (permissions, timestamps, etc.)
                shutil.copystat(source, destination)
            except Exception as stat_error:
                self.logger.warning(f"Could not copy all file stats: {stat_error}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error copying file with metadata: {e}")
            return False
