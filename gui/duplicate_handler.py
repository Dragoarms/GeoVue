# gui/duplicate_handler.py

import os
import re
import logging
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from typing import Dict, List, Optional, Union, Any
import traceback

from gui.dialog_helper import DialogHelper
from utils.onedrive_path_manager import OneDrivePathManager

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
    
    def __init__(self, output_dir: str, onedrive_manager=None):
        """
        Initialize the duplicate handler.
        
        Args:
            output_dir: Directory where processed images are saved
            onedrive_manager: Optional OneDrivePathManager instance to use
        """
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        self.onedrive_manager = onedrive_manager  # Use passed instance
        self.processed_entries = self._load_existing_entries()
        self.parent = None  # Will be set by the parent component
        self.root = None    # Will be set based on parent
        
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
    
    def _load_existing_entries(self) -> Dict[str, List[str]]:
        """
        Load existing processed entries from saved files across multiple directories.
        Checks Temp_Review/, Chip Compartments/, and OneDrive storage.
        Handles both with and without project code folder structures.
        
        Returns:
            Dictionary of processed entries where:
            - key: "{hole_id}_{depth_from}-{depth_to}"
            - value: List of filenames for that tray
        """
        entries = {}
        
        # ===================================================
        # NEW CODE: Define all directories to check
        # ===================================================
        directories_to_check = []
        
        # Local directories (relative to base_dir)
        base_dir = os.path.dirname(self.output_dir)  # Go up from Compartments to base
        
        # TODO - only have a temp review directory locally not uploaded to onedrive currently... but might as well??
        # Add Temp_Review directory
        temp_review_dir = os.path.join(base_dir, "Temp_Review")
        if os.path.exists(temp_review_dir):
            directories_to_check.append(temp_review_dir)
        
        # Add Chip Compartments directory (the original output_dir)
        if os.path.exists(self.output_dir):
            directories_to_check.append(self.output_dir)
        
        # Add OneDrive directory if available
        # Use the instance OneDrivePathManager if available
        try:
            if self.onedrive_manager:
                approved_folder = self.onedrive_manager.get_approved_folder_path()
                if approved_folder and os.path.exists(approved_folder):
                    directories_to_check.append(approved_folder)
                    self.logger.info(f"Including OneDrive approved folder in duplicate check: {approved_folder}")
            else:
                self.logger.debug("No OneDrivePathManager instance available for duplicate check")
        except Exception as e:
            self.logger.debug(f"Could not access OneDrive directory: {e}")
        
        # Process each directory
        for directory in directories_to_check:
            self._scan_directory_for_entries(directory, entries)
        
        return entries

    def _scan_directory_for_entries(self, directory: str, entries: Dict[str, List[str]]):
        """
        Scan a directory for compartment images, handling both with and without project code folders.
        
        Args:
            directory: Directory to scan
            entries: Dictionary to update with found entries
        """
        if not os.path.exists(directory):
            return
        
        # ===================================================
        # NEW CODE: Helper function to process hole ID directories
        # ===================================================
        def process_hole_id_directory(hole_id_dir_path: str, hole_id: str):
            """Process a single hole ID directory and extract compartment files."""
            compartment_files = {}  # Group files by depth
            
            try:
                for filename in os.listdir(hole_id_dir_path):
                    # Skip directories and non-image files
                    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                        continue
                    
                    # Extract depth from compartment filename
                    # Pattern: XX1234_CC_123 or XX1234_CC_123_suffix
                    depth_match = re.search(r'_CC_(\d+)(?:_[^.]+)?', filename)
                    if depth_match:
                        depth = int(depth_match.group(1))
                        
                        # Store filename - we keep track of all variants
                        if depth not in compartment_files:
                            compartment_files[depth] = []
                        compartment_files[depth].append(filename)
            except Exception as e:
                self.logger.debug(f"Error scanning {hole_id_dir_path}: {e}")
                return
            
            # Group compartments into trays
            if compartment_files:
                # Sort depths to group them properly
                sorted_depths = sorted(compartment_files.keys())
                
                # Determine interval (1m or 2m) based on depth spacing
                interval = 1  # Default
                if len(sorted_depths) > 1:
                    # Check spacing between consecutive depths
                    spacing = sorted_depths[1] - sorted_depths[0]
                    if spacing == 2:
                        interval = 2
                
                # Group into trays based on interval
                current_tray_start = None
                current_tray_files = []
                
                for depth in sorted_depths:
                    # Determine which tray this depth belongs to
                    # For interval=1: depths 1-20 are tray 0-20, 21-40 are tray 20-40, etc.
                    # For interval=2: depths 2,4,6...40 are tray 0-40, 42,44...80 are tray 40-80, etc.
                    
                    if interval == 1:
                        tray_start = ((depth - 1) // 20) * 20
                        tray_end = tray_start + 20
                    else:  # interval == 2
                        tray_start = ((depth - 2) // 40) * 40
                        tray_end = tray_start + 40
                    
                    # Check if we've moved to a new tray
                    if current_tray_start is None or tray_start != current_tray_start:
                        # Save previous tray if exists
                        if current_tray_start is not None and current_tray_files:
                            key = self._generate_entry_key(hole_id, current_tray_start, 
                                                        current_tray_start + (20 * interval))
                            if key not in entries:
                                entries[key] = []
                            entries[key].extend(current_tray_files)
                        
                        # Start new tray
                        current_tray_start = tray_start
                        current_tray_files = []
                    
                    # Add files for this depth to current tray
                    current_tray_files.extend(compartment_files[depth])
                
                # Don't forget the last tray
                if current_tray_start is not None and current_tray_files:
                    key = self._generate_entry_key(hole_id, current_tray_start, 
                                                current_tray_start + (20 * interval))
                    if key not in entries:
                        entries[key] = []
                    entries[key].extend(current_tray_files)
        
        # ===================================================
        # Check directory structure
        # ===================================================
        # First, check if we have project code folders (XX, YY, etc.)
        has_project_folders = False
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path) and re.match(r'^[A-Z]{2}$', item):
                has_project_folders = True
                break
        
        if has_project_folders:
            # Directory structure with project code folders
            for project_code in os.listdir(directory):
                project_path = os.path.join(directory, project_code)
                
                # Skip if not a directory or not a valid project code
                if not os.path.isdir(project_path) or not re.match(r'^[A-Z]{2}$', project_code):
                    continue
                
                # Look for hole ID directories within project folder
                for hole_id_dir in os.listdir(project_path):
                    hole_id_path = os.path.join(project_path, hole_id_dir)
                    
                    # Skip if not a directory or not a valid hole ID
                    if not os.path.isdir(hole_id_path) or not re.match(r'^[A-Z]{2}\d{4}$', hole_id_dir):
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
                if hole_id_dir.startswith('.') or hole_id_dir in ['Blur Analysis']:
                    continue
                
                # Check if directory name matches hole ID pattern
                if not re.match(r'^[A-Z]{2}\d{4}$', hole_id_dir):
                    continue
                
                process_hole_id_directory(hole_id_path, hole_id_dir)


    def check_debug_image_exists(self, 
                               hole_id: str, 
                               depth_from: int, 
                               depth_to: int) -> Optional[str]:
        """
        Check if a debug small image already exists for this hole ID and depth range.
        
        Args:
            hole_id: Hole ID
            depth_from: Starting depth
            depth_to: Ending depth
            
        Returns:
            Path to existing debug image, or None if not found
        """
        # Build the path to where debug images would be stored
        debug_dir = os.path.join(self.output_dir, "Debug Images", hole_id)
        
        if not os.path.exists(debug_dir):
            return None
            
        # Look for debug small image files
        for filename in os.listdir(debug_dir):
            if f"{hole_id}_{int(depth_from)}-{int(depth_to)}_Debug_small_image" in filename:
                return os.path.join(debug_dir, filename)
        
        return None
    
    def check_duplicate(self, 
                    hole_id: str, 
                    depth_from: int, 
                    depth_to: int,
                    small_image: np.ndarray,
                    full_filename: str,
                    extracted_compartments: Optional[List[np.ndarray]] = None) -> Union[bool, Dict[str, Any]]:
        """
        Check if an entry is a potential duplicate and prompt user.
        
        Args:
            hole_id: Unique hole identifier
            depth_from: Starting depth
            depth_to: Ending depth
            small_image: Downsampled image for comparison
            full_filename: Full path to the original image file
            extracted_compartments: Optional list of already extracted compartment images
        
        Returns:
            bool or Dict: True if processing should continue (replace),
                        False if processing should be skipped,
                        or a dictionary with modified metadata or selective replacement flag
        """
        if self.parent is None:
            self.logger.warning("DuplicateHandler parent not set, unable to show interactive dialogs")
            return True  # No duplicates
            
        # Ensure we have a reference to the root window
        if self.root is None and hasattr(self.parent, 'root'):
            self.root = self.parent.root
        
        # Store original path for potential QAQC processing
        self._current_image_path = full_filename
        
        # Reload existing entries to ensure we have the latest data
        self.processed_entries = self._load_existing_entries()
        
        # Generate key for current entry
        entry_key = self._generate_entry_key(hole_id, depth_from, depth_to)
        
        # Log for debugging
        self.logger.info(f"Checking for duplicates with key: {entry_key}")
        self.logger.info(f"Known entries: {list(self.processed_entries.keys())}")
        
        # Check if this exact entry exists
        is_duplicate = False
        self.duplicate_files = []  # Store duplicate files as instance variable
        
        # Check BOTH Chip Compartment folder AND Temp_Review folder for existing compartments
        try:
            # First check the Chip Compartments folder
            compartment_dir = os.path.join(self.output_dir, "Chip Compartments", hole_id)
            temp_review_dir = os.path.join(self.output_dir, "Temp_Review", hole_id)
            onedrive_approved_dir = None
            try:             
                # Get the approved folder path from OneDrive
                if self.onedrive_manager:
                    approved_path = self.onedrive_manager.get_approved_folder_path()
                    if approved_path and os.path.exists(approved_path):

                        # Check for both direct and project-grouped folder structures
                        # Extract project code from hole ID (first 2 characters)
                        project_code = hole_id[:2] if len(hole_id) >= 2 else ""
                        
                        # Try project-grouped structure first (e.g., KM/KM0001)
                        onedrive_project_hole_dir = os.path.join(approved_path, project_code, hole_id)
                        
                        # Try direct hole ID structure (e.g., KM0001)
                        onedrive_hole_dir = os.path.join(approved_path, hole_id)
                        
                        # Check which structure exists
                        if os.path.exists(onedrive_project_hole_dir) and os.path.isdir(onedrive_project_hole_dir):
                            onedrive_approved_dir = onedrive_project_hole_dir
                            self.logger.info(f"Found OneDrive approved folder for {hole_id} in project structure: {onedrive_project_hole_dir}")
                        elif os.path.exists(onedrive_hole_dir) and os.path.isdir(onedrive_hole_dir):
                            onedrive_approved_dir = onedrive_hole_dir
                            self.logger.info(f"Found OneDrive approved folder for {hole_id}: {onedrive_hole_dir}")
                        else:
                            self.logger.debug(f"No OneDrive subfolder found for {hole_id} in either direct or project-grouped structure")
                            self.logger.debug(f"Checked paths: {onedrive_hole_dir} and {onedrive_project_hole_dir}")
                    else:
                        self.logger.debug("OneDrive approved folder not configured or not accessible")
            except Exception as e:
                self.logger.warning(f"Could not check OneDrive approved folder: {str(e)}")
                # Continue without OneDrive checking - don't break the duplicate check
            
            # Calculate depth range covered by this image
            # Convert to integers for comparison
            start_depth = int(depth_from)
            end_depth = int(depth_to)
            
            # Calculate the compartment depths for this range
            # For a range like 40-60, compartments would be at 41, 42, ..., 60
            # Each compartment is labeled with its end depth
            compartment_depths = list(range(start_depth + 1, end_depth + 1))
            
            self.logger.info(f"Image covers compartments at depths: {compartment_depths}")
            
            # Initialize our tracking variables
            missing_compartments = set(compartment_depths)  # Start with all compartments as missing
            existing_compartments = []
            
            # Check Chip Compartments folder if it exists
            if os.path.exists(compartment_dir) and os.path.isdir(compartment_dir):
                self.logger.info(f"Checking compartment directory: {compartment_dir}")
                
                # Get all compartment files in the directory
                for file in os.listdir(compartment_dir):
                    if file.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                        # Look for the pattern HoleID_CC_XXX
                        match = re.search(rf'{hole_id}_CC_(\d+)', file)
                        if match:
                            compartment_depth = int(match.group(1))
                            
                            # Check if this depth is in our compartment depths
                            if compartment_depth in compartment_depths:
                                compartment_path = os.path.join(compartment_dir, file)
                                existing_compartments.append({
                                    'depth': compartment_depth,
                                    'path': compartment_path,
                                    'source': 'Chip Compartments'
                                })
                                missing_compartments.discard(compartment_depth)
                                self.logger.info(f"Found existing compartment at depth {compartment_depth} in Chip Compartments: {file}")
            
            # Check Temp_Review folder if it exists
            if os.path.exists(temp_review_dir) and os.path.isdir(temp_review_dir):
                self.logger.info(f"Checking Temp_Review directory: {temp_review_dir}")
                
                # Get all compartment files in the Temp_Review directory
                for file in os.listdir(temp_review_dir):
                    if file.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                        # Look for the pattern HoleID_CC_XXX
                        match = re.search(rf'{hole_id}_CC_(\d+)', file)
                        if match:
                            compartment_depth = int(match.group(1))
                            
                            # Check if this depth is in our compartment depths and not already found
                            if compartment_depth in compartment_depths and compartment_depth in missing_compartments:
                                compartment_path = os.path.join(temp_review_dir, file)
                                existing_compartments.append({
                                    'depth': compartment_depth,
                                    'path': compartment_path,
                                    'source': 'Temp_Review'
                                })
                                missing_compartments.discard(compartment_depth)
                                self.logger.info(f"Found existing compartment at depth {compartment_depth} in Temp_Review: {file}")
                                
            if onedrive_approved_dir and os.path.exists(onedrive_approved_dir):
                try:
                    self.logger.info(f"Checking OneDrive approved directory: {onedrive_approved_dir}")
                    
                    # Get all compartment files in the OneDrive directory
                    for file in os.listdir(onedrive_approved_dir):
                        if file.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                            # Look for the pattern HoleID_CC_XXX
                            match = re.search(rf'{hole_id}_CC_(\d+)', file)
                            if match:
                                compartment_depth = int(match.group(1))
                                
                                # Check if this depth is in our compartment depths and not already found
                                if compartment_depth in compartment_depths and compartment_depth in missing_compartments:
                                    compartment_path = os.path.join(onedrive_approved_dir, file)
                                    existing_compartments.append({
                                        'depth': compartment_depth,
                                        'path': compartment_path,
                                        'source': 'OneDrive Approved'
                                    })
                                    missing_compartments.discard(compartment_depth)
                                    self.logger.info(f"Found existing compartment at depth {compartment_depth} in OneDrive Approved: {file}")
                                    
                except PermissionError:
                    self.logger.warning(f"Permission denied accessing OneDrive folder: {onedrive_approved_dir}")
                except Exception as e:
                    self.logger.warning(f"Error checking OneDrive approved folder: {str(e)}")



            # If we found existing compartments, mark as duplicate but show only if we have all of them
            if existing_compartments:
                if not missing_compartments:
                    # We have all compartments - mark as full duplicate
                    is_duplicate = True
                    self.duplicate_files.extend([comp['path'] for comp in existing_compartments])
                    self.logger.info(f"Found all {len(existing_compartments)} compartments for {hole_id} between {start_depth}-{end_depth}m")
                    # Also store where they were found for better user feedback
                    self.duplicate_sources = [comp['source'] for comp in existing_compartments]
                else:
                    # We have some compartments but not all
                    # Mark as partial duplicate
                    self.is_partial_duplicate = True
                    self.existing_compartments = existing_compartments
                    self.missing_compartments = missing_compartments
                    self.logger.info(f"Found partial match: {len(existing_compartments)} existing compartments, {len(missing_compartments)} missing")
                    
                    # Store existing compartment data as duplicate_files for the dialog
                    self.duplicate_files = [comp['path'] for comp in existing_compartments]
                    self.duplicate_sources = [comp['source'] for comp in existing_compartments]
                    
                    # Store partial info for visualization
                    if hasattr(self, 'parent') and hasattr(self.parent, 'visualization_cache'):
                        self.parent.visualization_cache['partial_duplicate_info'] = {
                            'existing_compartments': existing_compartments,
                            'missing_compartments': missing_compartments
                        }
                    
                    # Mark as duplicate to show dialog
                    is_duplicate = True
                    self.logger.info("Marking as duplicate to show dialog for user decision")
        except Exception as e:
            self.logger.error(f"Error checking compartment directories: {str(e)}")
            self.logger.error(traceback.format_exc())
        
        # Log additional info about existing files
        if hasattr(self, 'duplicate_files') and self.duplicate_files:
            self.logger.info(f"Found {len(self.duplicate_files)} duplicate files")
            # Limit the number of files logged to avoid overwhelming the log
            for file in self.duplicate_files[:5]:  # Log up to 5 files
                self.logger.info(f"Existing file: {file}")
            if len(self.duplicate_files) > 5:
                self.logger.info(f"...and {len(self.duplicate_files) - 5} more files")
        
        if is_duplicate:
            # Create duplicate resolution dialog
            self.logger.info(f"Duplicate detected for {entry_key}, showing dialog")
            result = self._show_duplicate_dialog(
                hole_id, 
                depth_from, 
                depth_to, 
                small_image,
                extracted_compartments
            )
            
            # Handle selective replacement
            if isinstance(result, dict) and result.get('selective_replacement', False):
                # If user chose to selectively replace compartments, return the selective replacement flag
                return {
                    'selective_replacement': True,
                    'hole_id': hole_id,
                    'depth_from': depth_from,
                    'depth_to': depth_to,
                    'original_path': full_filename
                }
            
            # Return the result directly - it will be either:
            # - False to skip processing
            # - True to continue processing (replacing existing)
            # - A dictionary with new metadata
            return result
        
        return True  # No duplicate found, continue processing

    def _calculate_optimal_compartment_size(self, dialog_width, num_compartments, 
                                            sample_compartment_image=None):
            """
            Calculate optimal display size for compartments based on available space.
            
            Args:
                dialog_width: Width of the dialog window
                num_compartments: Number of compartments to display
                sample_compartment_image: Optional sample image to determine aspect ratio
                
            Returns:
                Tuple of (width, height) for compartment display
            """
            # Calculate available width
            margin_space = 150  # Increased from 100 to account for labels and padding
            available_width = max(400, dialog_width - margin_space)
            
            # Space between compartments
            spacing = 2
            total_spacing = spacing * (num_compartments + 1)
            
            # Maximum width per compartment
            max_width = (available_width - total_spacing) // max(num_compartments, 1)
            
            # Determine aspect ratio from sample image or use default
            if sample_compartment_image is not None and len(sample_compartment_image.shape) >= 2:
                h, w = sample_compartment_image.shape[:2]
                aspect_ratio = h / w if w > 0 else 2.0
            else:
                aspect_ratio = 2.0  # Default 1:2 width:height ratio
            
            # Calculate dimensions with bounds
            width = max(60, min(max_width, 160))  # Increased minimum from 50 to 60, max from 150 to 160
            height = int(width * aspect_ratio)
            
            # Cap height to prevent overly tall compartments
            if height > 300:  # Increased from 250
                height = 300
                width = int(height / aspect_ratio)
            
            return width, height


    def _show_duplicate_dialog(self, 
                              hole_id: str, 
                              depth_from: int, 
                              depth_to: int, 
                              current_image: np.ndarray,
                              extracted_compartments: Optional[List[np.ndarray]] = None) -> Union[bool, Dict[str, Any]]:
        """
        Show dialog for duplicate resolution with improved UI and maximized layout.
        
        Args:
            hole_id: Unique hole identifier
            depth_from: Starting depth
            depth_to: Ending depth
            current_image: Current image being processed
            extracted_compartments: Optional list of already extracted compartment images
            
        Returns:
            Either a boolean (False to skip, True to continue) or a dict with new metadata
        """
        if self.root is None:
            self.logger.error("Cannot show duplicate dialog: root window reference is missing")
            return False  # Default to skipping
        
        # Create a Tkinter dialog using DialogHelper
        dialog = DialogHelper.create_dialog(
            parent=self.root,
            title=DialogHelper.t("Duplicate Compartments Found"),
            modal=True,
            topmost=True,
            size_ratio=0.9,  # Use most of screen but leave some margin
            min_width=1200,
            min_height=800,
            max_width=None,  # Prevent dialog from being too wide
            max_height=None   # Prevent dialog from being too tall
        )

        # Ensure the dialog is properly sized before we start laying out content
        dialog.update()  # Process all pending events
        dialog.update_idletasks()  # Update geometry
        
        # Get the actual dialog dimensions after it's been properly sized
        actual_dialog_width = dialog.winfo_width()
        actual_dialog_height = dialog.winfo_height()
        
        # If we still get 1 for width, force a minimum width
        if actual_dialog_width <= 1:
            # Get screen dimensions
            screen_width = dialog.winfo_screenwidth()
            screen_height = dialog.winfo_screenheight()
            
            # Calculate desired width based on size_ratio and constraints
            desired_width = min(max(1200, int(screen_width * 0.9)), 1600)
            desired_width = min(desired_width, screen_width - 100)
            
            # Calculate height similarly
            desired_height = min(max(800, int(screen_height * 0.9)), 1000)
            desired_height = min(desired_height, screen_height - 100)
            
            # Center the dialog
            x = (screen_width - desired_width) // 2
            y = (screen_height - desired_height) // 2
            
            # Set the geometry explicitly
            dialog.geometry(f"{desired_width}x{desired_height}+{x}+{y}")
            
            # Update again after setting geometry
            dialog.update()
            dialog.update_idletasks()
            
            # Now get the actual dimensions
            actual_dialog_width = desired_width
            actual_dialog_height = desired_height
        
        self.logger.info(f"Dialog dimensions: {actual_dialog_width}x{actual_dialog_height}")        
        
        # Store the result as a class attribute for retrieval later
        dialog.result = None
        
        # Get theme colors if GUI manager is available
        theme_colors = None
        gui_manager = None
        if hasattr(self, 'parent') and hasattr(self.parent, 'gui_manager'):
            gui_manager = self.parent.gui_manager
            theme_colors = gui_manager.theme_colors
        else:
            # Default dark theme colors as fallback
            theme_colors = {
                "background": "#1e1e1e",
                "secondary_bg": "#252526",
                "text": "#e0e0e0",
                "field_bg": "#2d2d2d",
                "field_border": "#3f3f3f",
                "accent_blue": "#3a7ca5",
                "accent_green": "#4a8259"
            }
        
        # Create modified_metadata dict to store any changes
        modified_metadata = {
            'hole_id': tk.StringVar(value=hole_id),
            'depth_from': tk.StringVar(value=str(depth_from)),
            'depth_to': tk.StringVar(value=str(depth_to))
        }
        
        # Main frame with padding - apply theme colors
        main_frame = ttk.Frame(dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Calculate compartment depths (integer values)
        start_depth = int(depth_from)
        end_depth = int(depth_to)
        all_depths = list(range(start_depth + 1, end_depth + 1))

        # First, determine optimal compartment display size based on screen space and number of compartments
        screen_width = dialog.winfo_screenwidth()
        screen_height = dialog.winfo_screenheight()
        
        # Reserve space for UI elements and margins
        ui_margin = 200  # Space for labels, padding, scrollbar if needed
        available_screen_width = screen_width - ui_margin
        
        # Calculate minimum compartment width to fit all on screen
        num_compartments = len(all_depths)
        spacing_between = 2
        total_spacing = spacing_between * (num_compartments + 1)
        
        # Start with a reasonable compartment width
        ideal_comp_width = 120  # Ideal width for good visibility
        min_comp_width = 80     # Minimum acceptable width
        max_comp_width = 200    # Maximum to prevent too large
        
        # Calculate what width we need to fit everything
        total_width_needed = (ideal_comp_width * num_compartments) + total_spacing + ui_margin
        
        # If it doesn't fit, scale down the compartments
        if total_width_needed > screen_width:
            # Calculate maximum possible width per compartment
            max_possible_width = (available_screen_width - total_spacing) // max(num_compartments, 1)
            comp_display_width = max(min_comp_width, min(max_possible_width, ideal_comp_width))
        else:
            comp_display_width = ideal_comp_width
        
        # Get aspect ratio for height calculation
        actual_aspect_ratio = 2.0  # Default aspect ratio (height/width)

        # Collect information about duplicates and organize by depth
        existing_compartment_data = {}
        missing_depths = set(all_depths)  # Track which depths are missing
        
        if hasattr(self, 'duplicate_files') and self.duplicate_files:
            for path in self.duplicate_files:
                if path and os.path.exists(path):
                    # Extract depth from filename
                    match = re.search(r'_CC_(\d+)', os.path.basename(path))
                    if match:
                        depth = int(match.group(1))
                        existing_compartment_data[depth] = path
                        missing_depths.discard(depth)  # Remove from missing set
        
        # Get a sample compartment image for aspect ratio calculation
        sample_compartment = None
        actual_aspect_ratio = 2.0  # Default aspect ratio (height/width)
        
        # Try to get aspect ratio from existing compartments first
        if existing_compartment_data:
            try:
                first_path = next(iter(existing_compartment_data.values()))
                sample_compartment = cv2.imread(first_path)
                if sample_compartment is not None:
                    h, w = sample_compartment.shape[:2]
                    if w > 0:
                        actual_aspect_ratio = h / w
                        self.logger.info(f"Got aspect ratio {actual_aspect_ratio:.2f} from existing compartment")
            except Exception as e:
                self.logger.warning(f"Could not load existing compartment for aspect ratio: {str(e)}")
        
        # If no existing compartments, try extracted compartments
        if sample_compartment is None and extracted_compartments and len(extracted_compartments) > 0:
            for comp in extracted_compartments:
                if comp is not None:
                    h, w = comp.shape[:2]
                    if w > 0:
                        actual_aspect_ratio = h / w
                        self.logger.info(f"Got aspect ratio {actual_aspect_ratio:.2f} from extracted compartment")
                        break
        
        # Calculate compartment height based on width and aspect ratio
        comp_display_height = int(comp_display_width * actual_aspect_ratio)
        
        # Cap height if it's too tall
        max_comp_height = 400
        if comp_display_height > max_comp_height:
            comp_display_height = max_comp_height
            comp_display_width = int(comp_display_height / actual_aspect_ratio)
        
        self.logger.info(f"Using compartment display size: {comp_display_width}x{comp_display_height} "
                        f"for {num_compartments} compartments (aspect ratio: {actual_aspect_ratio:.2f})")
        
        # Title section with improved styling
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Main title with proper styling
        title_label = ttk.Label(
            title_frame,
            text=DialogHelper.t(f"Duplicate Compartments Found"),
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(0, 5))
        
        # Subtitle with hole ID and depth range
        subtitle_label = ttk.Label(
            title_frame,
            text=DialogHelper.t(f"{hole_id} - {int(depth_from)}-{int(depth_to)}m"),
            font=("Arial", 14)
        )
        subtitle_label.pack(pady=(0, 5))
        
        # Duplicate count with details
        duplicate_count = len(existing_compartment_data)
        missing_count = len(missing_depths)
        
        # Count duplicates by source
        source_counts = {}
        if hasattr(self, 'existing_compartments'):
            for comp in self.existing_compartments:
                source = comp.get('source', 'Unknown')
                source_counts[source] = source_counts.get(source, 0) + 1
        
        status_text = f"{duplicate_count} existing compartment{'s' if duplicate_count != 1 else ''} found"
        
        # Add source breakdown if available
        if source_counts:
            source_details = []
            for source, count in source_counts.items():
                source_details.append(f"{count} in {source}")
            status_text += f" ({', '.join(source_details)})"
        
        if missing_count > 0:
            status_text += f", {missing_count} missing"
            missing_text = ", ".join([f"{d}m" for d in sorted(missing_depths)])
            status_text += f"\nMissing: {missing_text}"
        
        
        duplicate_label = ttk.Label(
            title_frame,
            text=DialogHelper.t(status_text),
            font=("Arial", 12)
        )
        duplicate_label.pack(pady=(0, 10))
        
        # Comparison frame to show both existing and current compartments
        comparison_frame = ttk.Frame(
            main_frame,
            padding=5
        )
        comparison_frame.pack(fill=tk.X, expand=False, pady=(0, 10))
        
        # Create a non-scrollable frame for compartments
        canvas_frame = ttk.Frame(comparison_frame)
        canvas_frame.pack(fill=tk.X, expand=False)
        
        # Calculate required canvas width
        header_width = 100  # Width for row labels
        canvas_width = header_width + (comp_display_width * num_compartments) + (spacing_between * (num_compartments + 1))
        
        # Calculate canvas height based on compartment display size
        header_height = 50  # Height for depth labels
        separator_height = 15
        bottom_padding = 30
        num_rows = 2  # Existing and Current rows
        canvas_height = header_height + (comp_display_height * num_rows) + separator_height + bottom_padding
        
        self.logger.info(f"Canvas dimensions: {canvas_width}x{canvas_height}")
        
        # Create canvas with calculated dimensions
        canvas = tk.Canvas(
            canvas_frame, 
            bg=theme_colors["background"],
            highlightthickness=0,
            width=canvas_width,   # Set explicit width
            height=canvas_height  # Set explicit height
        )
        canvas.pack(expand=False)  # Don't expand - use natural size
        
        # Container for all compartments
        compartments_container = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=compartments_container, anchor="nw", tags="compartments")
        
        # Section for existing compartments
        existing_header = ttk.Label(
            compartments_container,
            text=DialogHelper.t("Existing:"),
            font=("Arial", 11, "bold")
        )
        existing_header.grid(row=0, column=0, sticky="w", pady=(0, 5), padx=(5, 10))
        
        # Create headers for each depth with smaller font
        for i, depth in enumerate(all_depths):
            depth_label = ttk.Label(
                compartments_container,
                text=f"{depth}m",
                font=("Arial", 9),
                width=max(6, comp_display_width // 10),
                anchor="center"
            )
            depth_label.grid(row=0, column=i+1, padx=1)
            
            # Highlight missing compartments
            if depth in missing_depths:
                depth_label.configure(foreground="orange")
        
        # Row for existing compartments
        existing_row = 1
        
        # Display existing compartments with placeholders for missing ones
        for i, depth in enumerate(all_depths):
            if depth in existing_compartment_data:
                path = existing_compartment_data[depth]
                try:
                    # Load image with OpenCV
                    comp_img = cv2.imread(path)
                    if comp_img is not None:
                        # Convert to RGB for PIL
                        comp_rgb = cv2.cvtColor(comp_img, cv2.COLOR_BGR2RGB)
                        
                        # Get actual dimensions
                        h, w = comp_rgb.shape[:2]
                        
                        # Calculate scale to fit display size while maintaining aspect ratio
                        scale_w = comp_display_width / w
                        scale_h = comp_display_height / h
                        scale = min(scale_w, scale_h)  # Use smaller scale to ensure it fits
                        
                        new_width = int(w * scale)
                        new_height = int(h * scale)
                        
                        # Resize maintaining aspect ratio
                        comp_rgb = cv2.resize(comp_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
                        
                        # Convert to PIL and then to ImageTk
                        pil_img = Image.fromarray(comp_rgb)
                        tk_img = ImageTk.PhotoImage(image=pil_img)
                        
                        # Create frame with border
                        img_frame = ttk.Frame(
                            compartments_container,
                            padding=1,
                            borderwidth=1,
                            relief="solid"
                        )
                        img_frame.grid(row=existing_row, column=i+1, padx=1, pady=2)
                        
                        # Display image
                        img_label = ttk.Label(img_frame, image=tk_img)
                        img_label.image = tk_img  # Keep reference
                        img_label.pack()

                        if hasattr(self, 'existing_compartments'):
                            for comp in self.existing_compartments:
                                if comp['depth'] == depth and comp.get('source') == 'OneDrive Approved':
                                    source_label = ttk.Label(
                                        img_frame,
                                        text="OneDrive",
                                        font=("Arial", 8),
                                        foreground="blue"
                                    )
                                    source_label.pack()
                                    break
                except Exception as e:
                    self.logger.error(f"Error loading compartment image: {str(e)}")
                    error_frame = ttk.Frame(
                        compartments_container,
                        padding=1,
                        borderwidth=1,
                        relief="solid"
                    )
                    error_frame.grid(row=existing_row, column=i+1, padx=1, pady=2)
                    
                    error_label = ttk.Label(
                        error_frame,
                        text=DialogHelper.t("Error"),
                        foreground="red"
                    )
                    error_label.pack(fill=tk.BOTH, expand=True)
                    error_label.configure(width=comp_display_width//10, height=comp_display_height//20)
            else:
                # Create placeholder for missing compartment with orange border
                placeholder_frame = tk.Frame(
                    compartments_container,
                    bg="orange",
                    highlightbackground="orange",
                    highlightthickness=2,
                    width=comp_display_width,
                    height=comp_display_height
                )
                placeholder_frame.grid(row=existing_row, column=i+1, padx=1, pady=2)
                placeholder_frame.grid_propagate(False)
                
                # Add a placeholder label
                placeholder_label = ttk.Label(
                    placeholder_frame,
                    text=DialogHelper.t("Missing"),
                    anchor="center"
                )
                placeholder_label.place(relx=0.5, rely=0.5, anchor="center")
        
        # Add separator between existing and current
        ttk.Separator(compartments_container, orient="horizontal").grid(
            row=existing_row+1, column=0, columnspan=len(all_depths)+1, sticky="ew", pady=5
        )
        
        # Section for current compartments
        current_header = ttk.Label(
            compartments_container,
            text=DialogHelper.t("Current:"),
            font=("Arial", 11, "bold")
        )
        current_header.grid(row=existing_row+2, column=0, sticky="w", pady=(0, 5), padx=(5, 10))
        
        # Process current image to extract compartments if not provided
        current_compartments = {}
        
        # Use pre-extracted compartments if available
        if extracted_compartments and len(extracted_compartments) > 0:
            self.logger.info(f"Using {len(extracted_compartments)} pre-extracted compartments for duplicate dialog")
            
            # Map compartments to depths
            for i, comp in enumerate(extracted_compartments):
                if i < len(all_depths) and comp is not None:
                    current_compartments[all_depths[i]] = comp
        else:
            self.logger.warning("No pre-extracted compartments provided, using basic slicing")
            # Fallback to basic extraction if no compartments provided
            h, w = current_image.shape[:2]
            num_depths = len(all_depths)
            
            if num_depths > 0:
                slice_width = w // num_depths
                for i, depth in enumerate(all_depths):
                    start_x = i * slice_width
                    end_x = (i+1) * slice_width if i < num_depths-1 else w
                    
                    # Extract slice
                    if len(current_image.shape) == 3:
                        compartment = current_image[:, start_x:end_x].copy()
                    else:
                        compartment = cv2.cvtColor(current_image[:, start_x:end_x], cv2.COLOR_GRAY2BGR)
                    
                    current_compartments[depth] = compartment
        
        # Display current compartments
        current_row = existing_row + 3
        for i, depth in enumerate(all_depths):
            if depth in current_compartments:
                comp_img = current_compartments[depth]
                try:
                    # Get actual dimensions
                    h, w = comp_img.shape[:2]
                    
                    # Calculate scale to fit display size while maintaining aspect ratio
                    scale_w = comp_display_width / w
                    scale_h = comp_display_height / h
                    scale = min(scale_w, scale_h)  # Use smaller scale to ensure it fits
                    
                    new_width = int(w * scale)
                    new_height = int(h * scale)
                    
                    # Resize maintaining aspect ratio
                    comp_img = cv2.resize(comp_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    
                    # Convert to RGB for PIL
                    if len(comp_img.shape) == 3:
                        comp_rgb = cv2.cvtColor(comp_img, cv2.COLOR_BGR2RGB)
                    else:
                        comp_rgb = cv2.cvtColor(comp_img, cv2.COLOR_GRAY2RGB)
                    
                    # Convert to PIL and then to ImageTk
                    pil_img = Image.fromarray(comp_rgb)
                    tk_img = ImageTk.PhotoImage(image=pil_img)
                    
                    # Create frame with border - highlight if this is a missing compartment
                    border_color = "green" if depth in missing_depths else None
                    if border_color and depth in missing_depths:
                        img_frame = tk.Frame(
                            compartments_container,
                            bg=border_color,
                            highlightbackground=border_color,
                            highlightthickness=2
                        )
                    else:
                        img_frame = ttk.Frame(
                            compartments_container,
                            padding=1,
                            borderwidth=1,
                            relief="solid"
                        )
                    img_frame.grid(row=current_row, column=i+1, padx=1, pady=2)
                    
                    # Display image
                    img_label = ttk.Label(img_frame, image=tk_img)
                    img_label.image = tk_img  # Keep reference
                    img_label.pack()
                except Exception as e:
                    self.logger.error(f"Error displaying current compartment: {str(e)}")
                    ttk.Label(
                        compartments_container,
                        text=DialogHelper.t("Error"),
                        foreground="red"
                    ).grid(row=current_row, column=i+1, padx=1, pady=2)
            else:
                # Create placeholder for missing compartment
                placeholder_frame = ttk.Frame(
                    compartments_container,
                    padding=1,
                    borderwidth=1,
                    relief="solid"
                )
                placeholder_frame.grid(row=current_row, column=i+1, padx=1, pady=2)
                placeholder_frame.configure(width=comp_display_width, height=comp_display_height)
                placeholder_frame.grid_propagate(False)
                
                placeholder_label = ttk.Label(
                    placeholder_frame,
                    text=DialogHelper.t("No image"),
                    anchor="center"
                )
                placeholder_label.place(relx=0.5, rely=0.5, anchor="center")
        
        # Center compartments if they don't fill the width
        compartments_container.update_idletasks()
        bbox = canvas.bbox("all")
        if bbox:
            content_width = bbox[2] - bbox[0]
            canvas_width = canvas.winfo_reqwidth()
            if content_width < canvas_width:
                # Center the content horizontally
                x_offset = (canvas_width - content_width) // 2
                canvas.coords("compartments", x_offset, 0)
        
        # Add Full Image Section - fills remaining space
        full_image_frame = ttk.Frame(main_frame, padding=5)
        full_image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        # Add a label for the full image section
        full_image_label = ttk.Label(
            full_image_frame,
            text=DialogHelper.t("Full Current Image:"),
            font=("Arial", 11, "bold")
        )
        full_image_label.pack(anchor="w", pady=(0, 5))

        # Create frame for the full image (no scroll)
        full_image_display_frame = ttk.Frame(full_image_frame)
        full_image_display_frame.pack(fill=tk.BOTH, expand=True)

        # Process and display the full image
        try:
            # Update dialog to get actual dimensions
            dialog.update_idletasks()
            
            # Calculate available space for full image
            # Account for: title + status + compartments + buttons + padding
            reserved_height = 150 + canvas_height + 150  # Rough estimate
            available_height = max(200, actual_dialog_height - reserved_height)
            
            # Cap at reasonable maximum to prevent image being too large
            available_height = min(400, available_height)
            
            # Get dialog width for horizontal constraint
            available_width = actual_dialog_width - 60  # Account for padding
  
            # Calculate dimensions maintaining aspect ratio
            h, w = current_image.shape[:2]
            aspect_ratio = w / h
            
            # Try to fit by height first
            new_height = available_height
            new_width = int(new_height * aspect_ratio)
            
            # Check if it fits horizontally
            if new_width > available_width:
                # Scale by width instead
                new_width = available_width
                new_height = int(new_width / aspect_ratio)
            
            # Resize image
            resized_image = cv2.resize(current_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Convert to RGB for PIL
            if len(resized_image.shape) == 3:
                img_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)
            
            # Convert to PIL and then to ImageTk
            pil_img = Image.fromarray(img_rgb)
            tk_img = ImageTk.PhotoImage(image=pil_img)
            
            # Create label to display image (centered)
            full_img_label = ttk.Label(full_image_display_frame, image=tk_img)
            full_img_label.image = tk_img  # Keep reference
            full_img_label.pack(expand=True)  # Center in frame
            
            # Add image info
            info_text = f"Image: {w}x{h} pixels (displayed at {new_width}x{new_height})"
            info_label = ttk.Label(
                full_image_display_frame,
                text=info_text,
                font=("Arial", 9),
                foreground="gray"
            )
            info_label.pack(pady=(5, 0))
            
        except Exception as e:
            self.logger.error(f"Error displaying full image: {str(e)}")
            ttk.Label(
                full_image_display_frame,
                text=DialogHelper.t(f"Error displaying full image: {str(e)}"),
                foreground="red"
            ).pack(pady=20)

        # Create a frame for action buttons at the bottom
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10, side=tk.BOTTOM)  # Pack at bottom
        
        # Add a container to right-align the buttons
        buttons_container = ttk.Frame(button_frame)
        buttons_container.pack(side=tk.RIGHT, padx=10)
        
        # Add a frame for metadata editing (initially hidden - don't pack yet)
        metadata_frame = ttk.Frame(main_frame, padding=10)
        
        # Track metadata visibility
        metadata_visible = [False]  # Use list to make it mutable in nested functions
        
        # Function to toggle visibility of metadata editor
        def toggle_metadata_editor():
            metadata_visible[0] = not metadata_visible[0]
            
            if metadata_visible[0]:
                # Show metadata editor
                metadata_frame.pack(anchor="center", pady=10, before=button_frame)
                # Update button text
                if hasattr(modify_button, "set_text"):
                    modify_button.set_text(DialogHelper.t("Hide Metadata"))
            else:
                # Hide metadata editor
                metadata_frame.pack_forget()
                # Update button text
                if hasattr(modify_button, "set_text"):
                    modify_button.set_text(DialogHelper.t("Modify Metadata"))
        
        # Button actions - define these before creating buttons
        
        
        def on_keep_images():
            # Check if there are missing compartments
            if missing_depths:
                self.logger.info(f"Keep Original selected, but extracting {len(missing_depths)} missing compartments to fill gaps")
                # Return a special flag to extract only missing compartments
                dialog.result = {
                    'action': 'keep_with_gaps',
                    'missing_depths': list(missing_depths),
                    'hole_id': hole_id,
                    'depth_from': depth_from,
                    'depth_to': depth_to,
                    'original_path': getattr(self, '_current_image_path', None)
                }
            else:
                # No missing compartments, just skip
                dialog.result = False  # Skip processing (keep existing)
            dialog.destroy()
        
        def on_selective_replace():
            # Return selective replacement flag
            dialog.result = {
                'selective_replacement': True,
                'hole_id': hole_id,
                'depth_from': depth_from,
                'depth_to': depth_to,
                'original_path': getattr(self, '_current_image_path', None)
            }
            dialog.destroy()
        
        def on_quit():
            """Handle quit button - stop all processing and return to main GUI."""
            # Confirm quitting
            if DialogHelper.confirm_dialog(
                dialog,
                DialogHelper.t("Stop Processing"),
                DialogHelper.t("Are you sure you want to stop processing?\n\nNo changes will be made to existing files and processing will stop."),
                yes_text=DialogHelper.t("Stop"),
                no_text=DialogHelper.t("Continue")
            ):
                # Set a special quit result
                dialog.result = {
                    'quit': True,
                    'message': 'User stopped processing'
                }
                dialog.destroy()

        def on_apply_metadata():
            # Validate input
            try:
                hole_id = modified_metadata['hole_id'].get().strip()
                depth_from_str = modified_metadata['depth_from'].get().strip()
                depth_to_str = modified_metadata['depth_to'].get().strip()
                
                # Validate hole ID - must be 2 letters followed by 4 digits
                if not hole_id:
                    DialogHelper.show_message(
                        dialog, 
                        DialogHelper.t("Validation Error"), 
                        DialogHelper.t("Hole ID is required"),
                        message_type="error"
                    )
                    return
                
                # Validate hole ID format using regex - 2 UPPERCASE letters followed by 4 digits
                if not re.match(r'^[A-Z]{2}\d{4}$', hole_id):
                    # If provided in lowercase, try to convert
                    if re.match(r'^[a-z]{2}\d{4}$', hole_id):
                        hole_id = hole_id.upper()
                        modified_metadata['hole_id'].set(hole_id)
                    else:
                        DialogHelper.show_message(
                            dialog, 
                            DialogHelper.t("Validation Error"), 
                            DialogHelper.t("Hole ID must be 2 uppercase letters followed by 4 digits (e.g., AB1234)"),
                            message_type="error"
                        )
                        return
                
                # Check if the hole ID prefix is in the list of valid prefixes
                if hasattr(self, 'parent') and hasattr(self.parent, 'tesseract_manager') and \
                   hasattr(self.parent.tesseract_manager, 'config'):
                    config = self.parent.tesseract_manager.config
                    if config.get('enable_prefix_validation', False):
                        valid_prefixes = config.get('valid_hole_prefixes', [])
                        if valid_prefixes:
                            prefix = hole_id[:2].upper()
                            if prefix not in valid_prefixes:
                                if not DialogHelper.confirm_dialog(
                                    dialog,
                                    DialogHelper.t("Prefix Validation Warning"),
                                    DialogHelper.t(f"The prefix '{prefix}' is not in the list of valid prefixes: {', '.join(valid_prefixes)}.\n\nDo you want to continue anyway?")
                                ):
                                    return
                
                # Validate depth range if provided - must be whole numbers
                depth_from = None
                depth_to = None
                
                if depth_from_str:
                    try:
                        depth_from = float(depth_from_str)
                        # Validate as a whole number
                        if depth_from != int(depth_from):
                            DialogHelper.show_message(
                                dialog, 
                                DialogHelper.t("Validation Error"), 
                                DialogHelper.t("Depth From must be a whole number"),
                                message_type="error"
                            )
                            return
                        # Convert to integer
                        depth_from = int(depth_from)
                    except ValueError:
                        DialogHelper.show_message(
                            dialog, 
                            DialogHelper.t("Validation Error"), 
                            DialogHelper.t("Depth From must be a number"),
                            message_type="error"
                        )
                        return
                
                if depth_to_str:
                    try:
                        depth_to = float(depth_to_str)
                        # Validate as a whole number
                        if depth_to != int(depth_to):
                            DialogHelper.show_message(
                                dialog, 
                                DialogHelper.t("Validation Error"), 
                                DialogHelper.t("Depth To must be a whole number"),
                                message_type="error"
                            )
                            return
                        # Convert to integer
                        depth_to = int(depth_to)
                    except ValueError:
                        DialogHelper.show_message(
                            dialog, 
                            DialogHelper.t("Validation Error"), 
                            DialogHelper.t("Depth To must be a number"),
                            message_type="error"
                        )
                        return
                
                # Validate that depth_to is greater than depth_from
                if depth_from is not None and depth_to is not None:
                    if depth_to <= depth_from:
                        DialogHelper.show_message(
                            dialog, 
                            DialogHelper.t("Validation Error"), 
                            DialogHelper.t("Depth To must be greater than Depth From"),
                            message_type="error"
                        )
                        return
                    
                    # Validate that depth intervals are sensible
                    if depth_to - depth_from > 40:
                        if not DialogHelper.confirm_dialog(
                            dialog,
                            DialogHelper.t("Validation Warning"),
                            DialogHelper.t(f"Depth range ({depth_from}-{depth_to}) seems unusually large. Continue anyway?")
                        ):
                            return
                
                # Set result
                dialog.result = {
                    'hole_id': hole_id,
                    'depth_from': depth_from,
                    'depth_to': depth_to
                }
                
                # Close dialog
                dialog.destroy()
                
            except Exception as e:
                self.logger.error(f"Error validating metadata input: {str(e)}")
                DialogHelper.show_message(
                    dialog,
                    DialogHelper.t("Error"),
                    DialogHelper.t(f"An error occurred: {str(e)}"),
                    message_type="error"
                )
        def show_help():
            # Create a help dialog
            help_text = "Duplicate Handling Options\n\n"
            
            for button_name, explanation in button_explanations.items():
                help_text += f"{button_name}:\n{explanation}\n\n"
            
            DialogHelper.show_message(
                dialog,
                DialogHelper.t("Help - Duplicate Handling Options"),
                DialogHelper.t(help_text),
                message_type="info"
            )
        
        # Create explanation texts for help - updated for gap filling and quit
        button_explanations = {
            "Keep Original": "Keep the existing compartment images. If there are gaps (missing compartments), those will be extracted from the current image to complete the set. The current image will be moved to 'Failed and Skipped Originals'.",
            "Selective Replace": "Extract all compartments to the review queue. You'll be able to choose which ones to keep during review.",
            "Modify Metadata": "Change the hole ID or depth range on the current image - this is likely required if the label was not changed between tray photos.",
            "Quit": "Stop processing immediately and return to the main window. No changes will be made to existing files."
        }
        
        # Set up metadata input fields - properly themed
        # Create metadata editor fields with proper styling
        ttk.Label(
            metadata_frame,
            text=DialogHelper.t("Edit Metadata:"),
            font=("Arial", 12, "bold")
        ).pack(anchor="center", pady=(10, 15))
        
        # Create a container for the metadata fields to keep them centered
        metadata_fields_container = ttk.Frame(metadata_frame)
        metadata_fields_container.pack(anchor="center")
        
        # Hole ID field with fixed width, not expanding
        hole_id_frame = ttk.Frame(metadata_fields_container)
        hole_id_frame.pack(pady=5)
        
        ttk.Label(hole_id_frame, text=DialogHelper.t("Hole ID:"), width=15).pack(side=tk.LEFT)
        
        # Use themed entry if available
        if gui_manager:
            hole_id_entry = gui_manager.create_entry_with_validation(
                hole_id_frame,
                textvariable=modified_metadata['hole_id'],
                width=20
            )
        else:
            hole_id_entry = ttk.Entry(
                hole_id_frame,
                textvariable=modified_metadata['hole_id'],
                width=20
            )
        hole_id_entry.pack(side=tk.LEFT, padx=5)
        
        # Depth fields with fixed width, not expanding
        depth_frame = ttk.Frame(metadata_fields_container)
        depth_frame.pack(pady=5)
        
        ttk.Label(depth_frame, text=DialogHelper.t("Depth Range:"), width=15).pack(side=tk.LEFT)
        
        # Use themed entries if available
        if gui_manager:
            depth_from_entry = gui_manager.create_entry_with_validation(
                depth_frame,
                textvariable=modified_metadata['depth_from'],
                width=10
            )
        else:
            depth_from_entry = ttk.Entry(
                depth_frame,
                textvariable=modified_metadata['depth_from'],
                width=10
            )
        depth_from_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(depth_frame, text="-").pack(side=tk.LEFT)
        
        if gui_manager:
            depth_to_entry = gui_manager.create_entry_with_validation(
                depth_frame,
                textvariable=modified_metadata['depth_to'],
                width=10
            )
        else:
            depth_to_entry = ttk.Entry(
                depth_frame,
                textvariable=modified_metadata['depth_to'],
                width=10
            )
        depth_to_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(depth_frame, text="m").pack(side=tk.LEFT)
        
        # Apply button centered at the bottom of the metadata section
        apply_button_frame = ttk.Frame(metadata_frame)
        apply_button_frame.pack(pady=10)
        
        if gui_manager:
            apply_button = gui_manager.create_modern_button(
                apply_button_frame,
                text=DialogHelper.t("Apply Metadata Changes"),
                color=gui_manager.theme_colors["accent_green"],
                command=on_apply_metadata
            )
            apply_button.pack()
        else:
            apply_button = ttk.Button(
                apply_button_frame,
                text=DialogHelper.t("Apply Metadata Changes"),
                command=on_apply_metadata
            )
            apply_button.pack()
        
        # Create buttons - now that all functions are defined
        if gui_manager:
            # Keep Original button
            keep_button = gui_manager.create_modern_button(
                buttons_container,
                text=DialogHelper.t("Keep Original"),
                color=gui_manager.theme_colors["accent_green"],
                command=on_keep_images
            )
            keep_button.pack(side=tk.LEFT, padx=5)
            
            # Selective Replace button
            selective_button = gui_manager.create_modern_button(
                buttons_container,
                text=DialogHelper.t("Selective Replace"),
                color=gui_manager.theme_colors["accent_blue"],
                command=on_selective_replace
            )
            selective_button.pack(side=tk.LEFT, padx=5)
            
            # Modify Metadata button
            modify_button = gui_manager.create_modern_button(
                buttons_container,
                text=DialogHelper.t("Modify Metadata"),
                color=gui_manager.theme_colors["accent_blue"],
                command=toggle_metadata_editor
            )
            modify_button.pack(side=tk.LEFT, padx=5)

            # Quit button
            quit_button = gui_manager.create_modern_button(
                buttons_container,
                text=DialogHelper.t("Quit"),
                color=gui_manager.theme_colors["accent_red"],
                command=lambda: on_quit()
            )
            quit_button.pack(side=tk.LEFT, padx=5)
            
            # Help button
            help_button = gui_manager.create_modern_button(
                buttons_container,
                text="?",
                color=gui_manager.theme_colors["accent_blue"],
                command=show_help
            )
            help_button.pack(side=tk.LEFT, padx=5)
        else:
            # Fallback to regular buttons if GUI manager not available
            keep_button = tk.Button(
                buttons_container,
                text=DialogHelper.t("Keep Original"),
                background="#4a8259",  # Green color
                foreground="white",
                font=("Arial", 12, "bold"),
                command=on_keep_images
            )
            keep_button.pack(side=tk.LEFT, padx=5)
            
            selective_button = tk.Button(
                buttons_container,
                text=DialogHelper.t("Selective Replace"),
                background="#3a7ca5",  # Blue color
                foreground="white",
                font=("Arial", 12, "bold"),
                command=on_selective_replace
            )
            selective_button.pack(side=tk.LEFT, padx=5)
            
            modify_button = tk.Button(
                buttons_container,
                text=DialogHelper.t("Modify Metadata"),
                background="#3a7ca5",  # Blue color
                foreground="white",
                font=("Arial", 12, "bold"),
                command=toggle_metadata_editor
            )
            modify_button.pack(side=tk.LEFT, padx=5)
            
            quit_button = tk.Button(
                buttons_container,
                text=DialogHelper.t("Quit"),
                background="#c05a5a",  # Red color
                foreground="white",
                font=("Arial", 12, "bold"),
                command=lambda: on_quit()
            )
            quit_button.pack(side=tk.LEFT, padx=5)
            

            help_button = tk.Button(
                buttons_container,
                text="?",
                background="#3a7ca5",  # Blue color
                foreground="white",
                font=("Arial", 12, "bold"),
                command=show_help
            )
            help_button.pack(side=tk.LEFT, padx=5)
        
        # Store the current image path for potential QAQC processing
        self._current_image_path = getattr(self, '_current_image_path', None)
        
        # Wait for dialog
        # Force update to calculate actual content size
        dialog.update_idletasks()
        
        # Get the required size for all content
        req_width = dialog.winfo_reqwidth()
        req_height = dialog.winfo_reqheight()
        
        # Make sure it fits on screen with some margin
        screen_width = dialog.winfo_screenwidth()
        screen_height = dialog.winfo_screenheight()
        margin = 50  # Margin from screen edges
        
        final_width = min(req_width, screen_width - margin * 2)
        final_height = min(req_height, screen_height - margin * 2)
        
        # Center the dialog
        x = (screen_width - final_width) // 2
        y = (screen_height - final_height) // 2
        
        # Apply the calculated geometry
        dialog.geometry(f"{final_width}x{final_height}+{x}+{y}")
        
        self.logger.info(f"Final dialog size: {final_width}x{final_height} (requested: {req_width}x{req_height})")

        dialog.wait_window()
        
        # Force process any remaining GUI events after dialog closes
        if self.root:
            self.root.update_idletasks()
        
        # Return the result
        return dialog.result


    def register_processed_entry(self, 
                               hole_id: str, 
                               depth_from: int, 
                               depth_to: int, 
                               output_files: List[str]) -> None:
        """
        Register a successfully processed entry.
        
        Args:
            hole_id: Unique hole identifier
            depth_from: Starting depth
            depth_to: Ending depth
            output_files: List of files generated for this entry
        """
        entry_key = self._generate_entry_key(hole_id, depth_from, depth_to)
        self.processed_entries[entry_key] = output_files