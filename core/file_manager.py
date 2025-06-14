# core/file_manager.py

"""
Manages file operations for GeoVue.

Handles directory creation, file naming conventions, and saving operations.
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
    
    Handles directory creation, file naming conventions, and saving operations.
    """
    
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
        
        self.dir_structure = {
            
            "register": self.base_dir / "Chip Tray Register",
            "images_to_process": self.base_dir / "Images to Process",
            "processed_originals": self.base_dir / "Processed Original Images",
            "approved_originals": self.base_dir / "Processed Original Images" / "Approved Originals",
            "rejected_originals": self.base_dir / "Processed Original Images" / "Rejected Originals",
            "chip_compartments": self.base_dir / "Extracted Compartment Images",
            "temp_review": self.base_dir / "Compartments for Review",
            "drill_traces": self.base_dir / "Drillhole Traces",
            "debugging": self.base_dir / "Debugging",
            "blur_analysis": self.base_dir / "Debugging" / "Blur Analysis",
            "debug_images": self.base_dir / "Debugging" / "Debug Images"
        }

        # Create base directories after everything is set up
        self.create_base_directories()

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
                        shutil.copy2(old_path, new_path)
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
            save_dir = self.get_hole_dir("chip_compartments", hole_id)
            
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
            base_dir = self.get_hole_dir("chip_compartments", hole_id)
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
                                
                                # Copy the file (use copy instead of move for safety)
                                shutil.copy2(old_path, new_path)
                                
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
        """Create the base directory structure."""
        try:
            # Create the base directory
            os.makedirs(self.base_dir, exist_ok=True)
            
            # Create each subdirectory
            for dir_path in self.dir_structure.values():
                os.makedirs(dir_path, exist_ok=True)
                
            self.logger.info(f"Base directory structure created at: {self.base_dir}")
        except Exception as e:
            self.logger.error(f"Error creating directory structure: {str(e)}")
            raise
    
    def get_hole_dir(self, dir_type: str, hole_id: str) -> str:
        """
        Get the directory path for a specific hole ID.
        
        Args:
            dir_type: Directory type (must be one of the keys in dir_structure)
            hole_id: Hole ID
            
        Returns:
            Full path to the hole directory
        """
        if dir_type not in self.dir_structure:
            raise ValueError(f"Invalid directory type: {dir_type}")
        
        # Create hole-specific directory
        hole_dir = os.path.join(self.dir_structure[dir_type], hole_id)
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
                      depth_from: float,
                      depth_to: float,
                      onedrive_manager=None,
                      is_processed: bool = True,
                      is_rejected: bool = False,
                      is_selective: bool = False,
                      is_skipped: bool = False,
                      delete_source: bool = True) -> Tuple[str, bool]:
        """
        Save original file by uploading to OneDrive (if available) and moving to local storage.
        This method handles both operations atomically to avoid file loss and preserves metadata.
        
        Args:
            source_path: Path to the source file
            hole_id: Hole ID
            depth_from: Starting depth
            depth_to: Ending depth
            onedrive_manager: OneDrivePathManager instance (optional)
            is_processed: Whether the file was processed (True for all processed files including rejected/skipped)
            is_rejected: Whether the file was rejected during processing
            is_selective: Whether this is a selective compartment replacement
            
        Returns:
            Tuple of (local_path, upload_success) where:
                local_path: Path to the final local file location (None if failed)
                upload_success: Whether OneDrive upload was successful
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
            elif is_skipped:  # Fixed: Use is_skipped instead of not is_processed
                base_suffix = "_Skipped"
            elif is_selective:
                base_suffix = "_Selected_Compartments"
            else:
                base_suffix = "_Original"
            
            # Variables to track uploads
            onedrive_path = None
            upload_attempted = False
            
            # Start OneDrive upload first (if manager provided and paths configured)
            if onedrive_manager and is_processed:
                try:
                    # Get OneDrive base path from config
                    if is_rejected:
                        onedrive_base = self.config_manager.get('onedrive_processed_originals', '') if self.config_manager else ''
                        if onedrive_base:
                            onedrive_base = os.path.join(onedrive_base, "Rejected")
                    else:
                        onedrive_base = self.config_manager.get('onedrive_processed_originals', '') if self.config_manager else ''

                    
                    # Only attempt upload if we have a valid base path
                    if onedrive_base and os.path.exists(onedrive_base):
                        upload_attempted = True
                        self.logger.info("Starting OneDrive upload...")
                        
                        # Extract project code from hole_id
                        project_code = hole_id[:2].upper() if len(hole_id) >= 2 else ""
                        
                        # Create OneDrive path structure
                        if project_code:
                            hole_folder = os.path.join(onedrive_base, project_code, hole_id)
                        else:
                            hole_folder = os.path.join(onedrive_base, hole_id)
                        
                        os.makedirs(hole_folder, exist_ok=True)
                        
                        # Create OneDrive filename
                        base_onedrive_filename = f"{hole_id}_{int(depth_from)}-{int(depth_to)}{base_suffix}"
                        onedrive_filename = f"{base_onedrive_filename}{ext}"
                        onedrive_path = os.path.join(hole_folder, onedrive_filename)
                        
                        # If file exists, add incrementing number
                        counter = 1
                        while os.path.exists(onedrive_path):
                            onedrive_filename = f"{base_onedrive_filename}_{counter}{ext}"
                            onedrive_path = os.path.join(hole_folder, onedrive_filename)
                            counter += 1
                        
                        # Start copy to OneDrive
                        if self.copy_with_metadata(source_path, onedrive_path):
                            self.logger.info(f"Started OneDrive upload: {onedrive_path}")
                        else:
                            self.logger.error(f"Failed to start OneDrive copy")
                            onedrive_path = None
                    else:
                        if not onedrive_base:
                            self.logger.info("OneDrive path not configured - skipping upload")
                        else:
                            self.logger.warning(f"OneDrive path does not exist: {onedrive_base}")
                        upload_attempted = False
                            
                except Exception as e:
                    self.logger.error(f"OneDrive upload initiation failed: {str(e)}")
                    onedrive_path = None
            
            # Now handle local storage
            # Determine local storage directory
            if is_rejected or not is_processed:
                target_dir = self.get_hole_dir("failed_originals", hole_id)
            else:
                target_dir = self.get_hole_dir("processed_originals", hole_id)
            
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
                    # Verify local copy
                    if os.path.exists(target_path) and os.path.getsize(target_path) == os.path.getsize(source_path):
                        if delete_source:  # Only delete if explicitly requested
                            os.remove(source_path)
                        local_success = True
                        self.logger.info(f"Local copy successful: {target_path}")
                    else:
                        self.logger.error("Local copy verification failed")
                else:
                    self.logger.error("Failed to copy file locally")
                    
            except Exception as e:
                self.logger.error(f"Failed to save file locally: {str(e)}")
            
            # Now verify OneDrive upload if it was attempted
            upload_success = False
            if onedrive_path and upload_attempted:
                try:
                    # Optional: Give OneDrive a moment to sync
                    # import time
                    # time.sleep(1)  # 1 second delay
                    
                    # Verify the upload
                    if os.path.exists(onedrive_path) and os.path.getsize(onedrive_path) == os.path.getsize(target_path):
                        upload_success = True
                        self.logger.info(f"OneDrive upload verified: {onedrive_path}")
                    else:
                        self.logger.warning(f"OneDrive upload verification failed - file may still be syncing")
                        
                except Exception as e:
                    self.logger.error(f"Error verifying OneDrive upload: {str(e)}")
            
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
            
            # SAFETY CHECK: Only delete source if we're absolutely sure we have a valid copy
            final_file_verified = False
            if local_success:
                # Verify the final file exists and has the correct size
                if os.path.exists(target_path) and os.path.getsize(target_path) == os.path.getsize(source_path):
                    final_file_verified = True
                    
                    # Only delete source if requested and verification passed
                    if delete_source and os.path.exists(source_path):
                        try:
                            os.remove(source_path)
                            self.logger.info(f"Source file deleted after successful save: {source_path}")
                        except Exception as e:
                            self.logger.error(f"Failed to delete source file: {str(e)}")
                            # Not critical - we have the file saved
                else:
                    self.logger.error(f"Final file verification failed!")
                    self.logger.error(f"Expected file at: {target_path}")
                    self.logger.error(f"Source file preserved at: {source_path}")
                    
                    # Try to clean up the failed copy if it exists
                    if os.path.exists(target_path):
                        try:
                            os.remove(target_path)
                            self.logger.info(f"Cleaned up failed copy: {target_path}")
                        except Exception as e:
                            self.logger.error(f"Failed to clean up bad copy: {str(e)}")
                    
                    # Mark as failed
                    local_success = False
            
            # Return results based on final verification
            if local_success and final_file_verified:
                return target_path, upload_success
            else:
                # Return None for local_path to indicate failure
                return None, False
                
        except Exception as e:
            self.logger.error(f"Error in save_original_file: {str(e)}")
            return None, False

    def move_original_file(self, source_path: str, hole_id: str, depth_from: float, depth_to: float,
                          is_processed: bool = True, is_rejected: bool = False, is_selective: bool = False,
                          suffix: str = None) -> Optional[str]:
        """
        Emergency method to move original file to organized storage without OneDrive upload.
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
            # Get the appropriate organized path
            target_dir = self._get_organized_path(hole_id, depth_from, depth_to, is_processed)
            
            # Create the target directory
            os.makedirs(target_dir, exist_ok=True)
            
            # Build the filename with optional suffix
            original_filename = os.path.basename(source_path)
            name_parts = os.path.splitext(original_filename)
            
            if suffix:
                new_filename = f"{name_parts[0]}_{suffix}{name_parts[1]}"
            else:
                new_filename = original_filename
                
            target_path = os.path.join(target_dir, new_filename)
            
            # Move the file
            shutil.move(source_path, target_path)
            
            self.logger.info(f"Emergency moved file to: {target_path}")
            return target_path
            
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