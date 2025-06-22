import os
import re
import cv2
import shutil
import logging
import traceback
import json
import time
from datetime import datetime
from tkinter import ttk
import tkinter as tk
from typing import List, Optional, Dict, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import threading


# Local imports
from utils.json_register_manager import JSONRegisterManager
from gui.dialog_helper import DialogHelper

if threading.current_thread() != threading.main_thread():
    raise RuntimeError("❌ QAQCManager called from a background thread!")


class QAQCManager:
    """
    Manages quality assurance and quality control of extracted compartment images.
    Provides a GUI for reviewing images and approving/rejecting them.
    Handles synchronization between local storage, OneDrive, and register.
    """
    
    def __init__(self, file_manager, translator_func, config_manager, 
                app, logger, register_manager):
        """
        Initialize the QAQC Manager.
        
        Args:
            file_manager: FileManager instance
            translator_func: Translation function
            config_manager: ConfigManager instance
            app: Main application instance (ChipTrayApp)
            logger: Logger instance
            register_manager: JSONRegisterManager instance
        """
        self.root = app.root # TODO - app.root? or main_app.root??
        self.file_manager = file_manager
        self.t = translator_func
        self.config_manager = config_manager
        self.app = app
        self.logger = logger
        self.register_manager = register_manager
        
        # Queue for pending trays to review
        self.pending_trays = []
        
        # Current tray being reviewed
        self.current_tray = None
        
        # Current compartment being reviewed
        self.current_compartment_index = 0
        
        # Status data for compartments
        self.compartment_statuses = {}
        
        # Review window
        self.review_window = None
        
        # Constants for status values
        self.STATUS_OK = "OK"
        self.STATUS_BLURRY = "Blurry"
        self.STATUS_DAMAGED = "Damaged"
        self.STATUS_MISSING = "MISSING"
        self.STATUS_WET = "Wet"
        self.STATUS_DRY = "Dry"

        # Valid QAQC statuses that indicate review is complete
        self.VALID_QAQC_STATUSES = [
            "OK_Wet",       # OK + wet sample
            "OK_Dry",       # OK + dry sample  
            "Blurry",
            "Damaged",
            "Missing",
            "MISSING"       # Alternative format for missing
        ]
        
        # Statuses that need QAQC review
        self.NEEDS_REVIEW_STATUSES = [
            "Found",        # Synchronizer found the file
            None,           # No status set
            "",             # Empty status
            "Not Set"       # Placeholder
        ]
        
        # Skip these - can't review
        self.SKIP_STATUSES = [
            "MISSING_FILE", # File deleted from shared folder
            "Not Found"     # Never had an image
        ]
        

        # Dry/Wet state tracking

        self.is_wet = False  # Default to Dry TODO - the toggle is a three state toggle from a custom widget - we need a third 'null' state.
        

        # Processing statistics

        self.stats = {
            'processed': 0,
            'uploaded': 0,
            'upload_failed': 0,
            'saved_locally': 0,
            'register_updated': 0,
            'register_failed': 0
        }
        

        # Main GUI reference for status updates
        self.main_gui = app.main_gui if hasattr(app, 'main_gui') else None


    def t(self, text):
        """Translate text using DialogHelper."""
        return DialogHelper.t(text)
    
    def set_main_gui(self, main_gui):
        """Set reference to main GUI for status updates."""
        self.main_gui = main_gui
        
    def add_tray_for_review(self, hole_id: str, depth_from: int, depth_to: int, 
                        original_path: str, compartments: List[np.ndarray],
                        is_selective_replacement: bool = False,
                        is_full_replacement: bool = False,
                        corners_list: Optional[List[Dict]] = None):
        """
        Add a tray to the review queue.
        
        Args:
            hole_id: The hole ID
            depth_from: Starting depth
            depth_to: Ending depth
            original_path: Path to the original image file
            compartments: List of extracted compartment images
            is_selective_replacement: Whether this is a selective replacement
            is_full_replacement: Whether this is a full replacement (replace_all action)
            corners_list: List of corner dictionaries for each compartment
        """
        # Save compartments with appropriate suffix
        temp_paths = []
        temp_review_dir = self.file_manager.get_hole_dir("temp_review", hole_id)
        
        for i, compartment in enumerate(compartments):
            try:
                depth_increment = self.app.config['compartment_interval']
                comp_depth_from = depth_from + (i * depth_increment)
                comp_depth_to = comp_depth_from + depth_increment
                compartment_depth = int(comp_depth_to)
                
                # Determine suffix based on replacement type
                if is_selective_replacement:
                    suffix = "new"  # For side-by-side comparison
                elif is_full_replacement:
                    suffix = "temp"  # Normal processing (conflicts already deleted)
                else:
                    suffix = "temp"  # Normal processing
                
                file_path = self.file_manager.save_temp_compartment(
                    compartment, hole_id, compartment_depth, suffix=suffix
                )
                temp_paths.append(file_path)
                
            except Exception as e:
                self.logger.error(f"Error saving temporary compartment {i+1}: {str(e)}")
                temp_paths.append(None)
        
        # Add to pending trays
        self.pending_trays.append({
            'hole_id': hole_id,
            'depth_from': depth_from,
            'depth_to': depth_to,
            'original_path': original_path,
            'compartments': compartments.copy(),
            'temp_paths': temp_paths,
            'compartment_statuses': {},
            'is_selective_replacement': is_selective_replacement,
            'is_full_replacement': is_full_replacement,
            'corners_list': corners_list
        })
        
        # Log appropriate message
        if is_full_replacement:
            message = f"Added tray for review (full replacement): {hole_id} {depth_from}-{depth_to}m"
        elif is_selective_replacement:
            message = f"Added tray for review (selective replacement): {hole_id} {depth_from}-{depth_to}m"
        else:
            message = f"Added tray for review: {hole_id} {depth_from}-{depth_to}m"
        
        self.logger.info(f"{message} with {len(compartments)} compartments")
        
        if hasattr(self, 'main_gui') and self.main_gui:
            self.main_gui.direct_status_update(message, status_type="info")

    def start_review_process(self):
        """Start the review process for all pending trays and sync with register."""
        # CHANGED: Reset statistics at start
        self.stats = {
            'processed': 0,
            'uploaded': 0,
            'upload_failed': 0,
            'saved_locally': 0,
            'register_updated': 0,
            'register_failed': 0
        }
        
        # First check if we already have pending trays from previous operation
        if self.pending_trays:
            # Process the first tray in the queue
            self._review_next_tray()
            return
        
        # Check if there are any trays in the Compartments for Review folder
        temp_review_dir = self.file_manager.dir_structure["temp_review"]
        if os.path.exists(temp_review_dir):
            # Get all hole IDs from project code subdirectories
            hole_dirs = []
            # First look for project code directories
            for project_code in os.listdir(temp_review_dir):
                project_path = os.path.join(temp_review_dir, project_code)
                if os.path.isdir(project_path) and re.match(r'^[A-Z]{2}$', project_code):
                    # Look for hole IDs within project folder
                    for hole_id in os.listdir(project_path):
                        hole_path = os.path.join(project_path, hole_id)
                        if os.path.isdir(hole_path) and re.match(r'^[A-Z]{2}\d{4}$', hole_id):
                            hole_dirs.append((hole_id, hole_path))
            
            if hole_dirs:
                # Process temp review files
                self._load_temp_review_files(temp_review_dir, hole_dirs)
                
        # Check for unregistered compartments in approved folder
        if not self.pending_trays:
            self._check_approved_vs_register()

                # Update status if we found entries needing QAQC
        if self.pending_trays:
            total_compartments = sum(len(tray['compartments']) for tray in self.pending_trays)
            needs_qaqc = sum(1 for tray in self.pending_trays if tray.get('from_register_review', False))
            
            if needs_qaqc > 0:
                if hasattr(self, 'main_gui') and self.main_gui:
                    self.main_gui.direct_status_update(
                        f"Found {needs_qaqc} entries with 'Found' status needing QAQC review", 
                        status_type="info"
                    )
        
        if not self.pending_trays:
            DialogHelper.show_message(
                self.root, 
                self.t("Review Complete"), 
                self.t("No images to review and all files are synchronized."), 
                message_type="info"
            )
            return
        
        # Process the first tray
        self._review_next_tray()
    
    def _load_temp_review_files(self, temp_review_dir: str, hole_dirs: List[str]):
        """Load compartments from temp review directory."""

        # Update main GUI with loading status

        if hasattr(self, 'main_gui') and self.main_gui:
            self.main_gui.direct_status_update(f"Loading temporary review files from {len(hole_dirs)} holes...", status_type="info")
        
        total_compartments_loaded = 0
        
        for hole_id, hole_dir_path in hole_dirs:
            
            # Find all temporary compartment images
            temp_files = [f for f in os.listdir(hole_dir_path) 
                        if (f.endswith('_temp.png') or f.endswith('_new.png')) and hole_id in f]
            
            if not temp_files:
                continue
            
    
            # Update status for each hole being loaded
    
            if hasattr(self, 'main_gui') and self.main_gui:
                self.main_gui.direct_status_update(f"Loading {len(temp_files)} compartments for hole {hole_id}...", status_type="info")
                
            # Extract depth range from filenames
            depths = []
            depth_increment = self.app.config['compartment_interval']
            for filename in temp_files:
                match = re.search(r'([A-Za-z]{2}\d{4})_CC_(\d{3})_(?:temp|new)\.png', filename)
                if match:
                    depth = int(match.group(2))
                    depths.append(depth)
            
            if depths:
                # Sort depths to find min and max
                depths.sort()
                min_depth = depths[0] - depth_increment
                max_depth = depths[-1]
                
                # Load the compartment images
                compartments = []
                temp_paths = []
                
        
                # FIXED: Use actual depths from files instead of range
        
                for depth in depths:
                    # Look for both temp and new suffixes
                    for suffix in ['temp', 'new']:
                        filename = f"{hole_id}_CC_{depth:03d}_{suffix}.png"
                        file_path = os.path.join(hole_dir_path, filename)
                        
                        if os.path.exists(file_path):
                            # Load image
                            img = cv2.imread(file_path)
                            if img is not None:
                                compartments.append(img)
                                temp_paths.append(file_path)
                                break
                
        
                # Track actual number of compartments loaded
        
                total_compartments_loaded += len(compartments)
                
                # Create a tray entry and add to pending trays
                tray_entry = {
                    'hole_id': hole_id,
                    'depth_from': min_depth,
                    'depth_to': max_depth,
                    'original_path': "From Temp_Review folder",
                    'compartments': compartments,
                    'temp_paths': temp_paths,
                    'compartment_statuses': {}
                }
                
                self.pending_trays.append(tray_entry)
                self.logger.info(f"Added tray for review from Temp_Review folder: {hole_id} {min_depth}-{max_depth}m with {len(compartments)} compartments")
        

        # Final status update

        if hasattr(self, 'main_gui') and self.main_gui:
            self.main_gui.direct_status_update(
                f"Loaded {total_compartments_loaded} compartments from {len(self.pending_trays)} trays for review", 
                status_type="success"
            )
    

    def _load_register_data(self) -> pd.DataFrame:
        """Load compartment data from JSON register."""
        return self.register_manager.get_compartment_data()


    def _review_next_tray(self):
        """Display the next tray for review."""
        if not self.pending_trays:
    
            # CHANGED: Show summary and update main GUI
    
            self._show_final_summary()
            return
        
        # Get the next tray
        self.current_tray = self.pending_trays.pop(0)
        self.current_compartment_index = 0
        

        # Check for auto-approve flag

        if self.current_tray.get('auto_approve', False):
            # Auto-approve all compartments
            for i in range(len(self.current_tray['compartments'])):
                self.current_tray['compartment_statuses'][i] = self.STATUS_OK
            
            # Save and move to next
            self._save_approved_compartments()
            self._review_next_tray()
            return
        
        # Save compartment images to temporary location if needed
        if not self.current_tray.get('temp_paths'):
            self._save_temp_compartments()
        

        # CHANGED: Don't initialize statuses - wait for user input

        self.current_tray['compartment_statuses'] = {}
        
        # Create review window if it doesn't exist
        if not self.review_window or not self.review_window.winfo_exists():
            self._create_review_window()
        
        # Show first compartment
        self._show_current_compartment()
    
    def _save_temp_compartments(self):
        """Save compartment images to a temporary location for review."""
        if not self.current_tray:
            return
        
        # Clear any previous temp paths
        self.current_tray['temp_paths'] = []
        
        # Save each compartment
        for i, compartment in enumerate(self.current_tray['compartments']):
            try:
                # Calculate compartment depth
                depth_from = self.current_tray['depth_from']
                depth_increment = self.app.config['compartment_interval']
                comp_depth_from = depth_from + (i * depth_increment)
                comp_depth_to = comp_depth_from + depth_increment
                compartment_depth = int(comp_depth_to)
                
        
                # Use FileManager to save
                # Check if this is a selective replacement tray
                is_selective = self.current_tray.get('is_selective_replacement', False)
                suffix = "new" if is_selective else "temp"

                # Use FileManager to save
                file_path = self.file_manager.save_temp_compartment(
                    compartment,
                    self.current_tray['hole_id'],
                    compartment_depth,
                    suffix=suffix
                )
                
                if file_path:
                    self.current_tray['temp_paths'].append(file_path)
                else:
                    self.logger.error(f"Failed to save temporary compartment {i+1}")
                    
            except Exception as e:
                self.logger.error(f"Error saving temporary compartment {i+1}: {str(e)}")

    def _create_review_window(self):
        """Create a window for reviewing compartments with the specified layout."""
        # Close existing window if open
        if self.review_window and self.review_window.winfo_exists():
            self.review_window.destroy()
        
        # Create new window
        self.review_window = tk.Toplevel(self.root)
        
        # Set title based on current tray
        hole_id = self.current_tray['hole_id']
        depth_from = self.current_tray['depth_from']
        depth_to = self.current_tray['depth_to']
        
        self.review_window.title(f"QAQC - {hole_id} {depth_from}-{depth_to}")
        
        # Set window to appear on top and maximize
        self.review_window.lift()  # Bring to front explicitly
        # self.review_window.state('zoomed')  # This maximizes the window
        
        # Protocol for window close
        self.review_window.protocol("WM_DELETE_WINDOW", self._on_review_window_close)
        

        # Bind keyboard shortcuts

        self.review_window.bind('1', lambda e: self._set_dry_from_keyboard())
        self.review_window.bind('2', lambda e: self._set_wet_from_keyboard())
        self.review_window.bind('<Left>', lambda e: self._on_previous())
        self.review_window.bind('<Right>', lambda e: self._on_next())
        
        # Main container frame with padding
        main_frame = ttk.Frame(self.review_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title and progress at the top
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X)
        
        title_label = ttk.Label(
            title_frame,
            text=DialogHelper.t(f"QAQC - {hole_id} {depth_from}-{depth_to}"),
            font=("Arial", 18, "bold")
        )
        title_label.pack(pady=(0, 5))
        
        # Progress label (Compartment x/x)
        self.progress_label = ttk.Label(
            title_frame,
            text=DialogHelper.t(f"Compartment {self.current_compartment_index + 1}/{len(self.current_tray['compartments'])}"),
            font=("Arial", 14)
        )
        self.progress_label.pack(pady=(0, 10))
        
        
        # Main content frame for the four panels
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Frame 1: Existing compartment image (left side)
        self.existing_frame = ttk.LabelFrame(content_frame, text=DialogHelper.t("Existing Image"), padding=10)
        self.existing_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Frame 2: Keep Original button
        self.keep_original_frame = ttk.LabelFrame(content_frame, text=DialogHelper.t("Keep Original"), padding=10)
        self.keep_original_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # Use ModernButton instead of tk.Button
        keep_original_btn = self.app.gui_manager.create_modern_button(
            self.keep_original_frame,
            text=DialogHelper.t("Keep Original"),
            color="#4a8259",  # Green color from theme
            command=self._set_keep_original_and_next
        )
        keep_original_btn.pack(fill=tk.X, expand=True, padx=5, pady=20)
        
        # Frame 3: New compartment image
        self.new_frame = ttk.LabelFrame(content_frame, text=DialogHelper.t("New Image"), padding=10)
        self.new_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Frame 4: Status buttons with keyboard shortcuts and summary
        status_container = ttk.Frame(content_frame)
        status_container.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        

        # Sample Moisture frame

        moisture_frame = ttk.LabelFrame(status_container, text=DialogHelper.t("Sample Moisture"), padding=10)
        moisture_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.wet_dry_var = tk.StringVar(value="Dry" if not self.is_wet else "Wet")
        self.wet_dry_button = self.app.gui_manager.create_modern_button(
            moisture_frame,
            text=self.t(f"Sample Type: {self.wet_dry_var.get()}"),
            color="#4a8259" if not self.is_wet else "#3a7ca5",  # Green for dry, blue for wet
            command=self._toggle_wet_dry
        )
        self.wet_dry_button.pack(fill=tk.X)
        
        # Status buttons frame
        self.status_frame = ttk.LabelFrame(status_container, text=DialogHelper.t("Compartment Status"), padding=10)
        self.status_frame.pack(fill=tk.X)
        
        # Add status buttons using ModernButton
        ok_button = self.app.gui_manager.create_modern_button(
            self.status_frame,
            text=DialogHelper.t("OK"),
            color="#4a8259",  # Green
            command=lambda: self._set_status_and_next(self.STATUS_OK)
        )
        ok_button.pack(fill=tk.X, pady=5)
        
        blurry_button = self.app.gui_manager.create_modern_button(
            self.status_frame,
            text=DialogHelper.t("BLURRY"),
            color="#9e4a4a",  # Red
            command=lambda: self._set_status_and_next(self.STATUS_BLURRY)
        )
        blurry_button.pack(fill=tk.X, pady=5)
        
        damaged_button = self.app.gui_manager.create_modern_button(
            self.status_frame,
            text=DialogHelper.t("DAMAGED"),
            color="#d68c23",  # Orange
            command=lambda: self._set_status_and_next(self.STATUS_DAMAGED)
        )
        damaged_button.pack(fill=tk.X, pady=5)
        
        missing_button = self.app.gui_manager.create_modern_button(
            self.status_frame,
            text=DialogHelper.t("MISSING"),
            color="#333333",  # Dark gray/black
            command=lambda: self._set_status_and_next(self.STATUS_MISSING)
        )
        missing_button.pack(fill=tk.X, pady=5)
        

        # Keyboard shortcuts info

        shortcuts_frame = ttk.LabelFrame(status_container, text=DialogHelper.t("Keyboard Shortcuts"), padding=5)
        shortcuts_frame.pack(fill=tk.X, pady=(10, 0))
        
        shortcuts_text = ttk.Label(
            shortcuts_frame,
            text=DialogHelper.t("1: Dry | 2: Wet\n← → Navigate (after status set)"),
            font=("Arial", 9),
            justify=tk.LEFT
        )
        shortcuts_text.pack()
        

        # Register summary box

        self.summary_frame = ttk.LabelFrame(status_container, text=DialogHelper.t("Register Data"), padding=5)
        self.summary_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.summary_text = tk.Text(
            self.summary_frame,
            height=6,
            width=25,
            wrap=tk.WORD,
            bg=self.app.gui_manager.theme_colors["field_bg"],
            fg=self.app.gui_manager.theme_colors["text"],
            font=("Arial", 9),
            state=tk.DISABLED
        )
        self.summary_text.pack(fill=tk.X)
        
        # Frame for the previous and next buttons at the bottom
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=tk.X, pady=10)
        
        # Previous button using ModernButton
        self.prev_button = self.app.gui_manager.create_modern_button(
            nav_frame,
            text=DialogHelper.t("← Previous"),
            color="#3a7ca5",  # Blue
            command=self._on_previous
        )
        self.prev_button.pack(side=tk.LEFT, padx=5)
        

        # Next button

        self.next_button = self.app.gui_manager.create_modern_button(
            nav_frame,
            text=DialogHelper.t("Next →"),
            color="#3a7ca5",  # Blue
            command=self._on_next
        )
        self.next_button.pack(side=tk.LEFT, padx=5)
        
        # Initialize the compartment frames to be ready for showing the current compartment
        self.compartment_frame = self.new_frame  # Set this for compatibility with _show_current_compartment
        
        # Ensure the "Keep Original" button frame is hidden initially
        self.keep_original_frame.pack_forget()
        self.existing_frame.pack_forget()

    def _toggle_wet_dry(self):
        """Toggle between Wet and Dry sample types."""
        self.is_wet = not self.is_wet
        new_state = "Wet" if self.is_wet else "Dry"
        self.wet_dry_var.set(new_state)

        # Update both text and color using helper
        self.wet_dry_button.update_button(
            text=self.t(f"Sample Type: {new_state}"),
            color="#3a7ca5" if self.is_wet else "#4a8259"
        )

        

        # Update register summary

        self._update_register_summary()

    def _set_dry_from_keyboard(self):
        """Set sample type to Dry from keyboard shortcut."""
        if self.is_wet:
            self._toggle_wet_dry()

    def _set_wet_from_keyboard(self):
        """Set sample type to Wet from keyboard shortcut."""
        if not self.is_wet:
            self._toggle_wet_dry()

    def _on_previous(self):
        """Show the previous compartment."""

        # CHANGED: Check if current has status before allowing navigation

        current_status = self.current_tray['compartment_statuses'].get(self.current_compartment_index, None)
        
        # Can always go back if we're not at the first image
        if self.current_compartment_index > 0:
            self.current_compartment_index -= 1
            self._show_current_compartment()

    def _on_next(self):
        """Move to next compartment without setting status."""

        # CHANGED: Only allow navigation if status is set

        current_status = self.current_tray['compartment_statuses'].get(self.current_compartment_index, None)
        
        if not current_status:
            # No status set - can't navigate forward
            DialogHelper.show_message(
                self.review_window,
                self.t("Status Required"),
                self.t("Please select a status for this compartment before proceeding."),
                message_type="warning"
            )
            return
        
        if self.current_compartment_index < len(self.current_tray['compartments']) - 1:
            self.current_compartment_index += 1
            self._show_current_compartment()

    def _update_register_summary(self):
        """Update the register summary display for current compartment."""
        try:
            self.summary_text.config(state=tk.NORMAL)
            self.summary_text.delete(1.0, tk.END)
            
            # Get current compartment info
            depth_from = self.current_tray['depth_from']
            depth_increment = self.app.config['compartment_interval']
            comp_depth_from = depth_from + (self.current_compartment_index * depth_increment)
            comp_depth_to = comp_depth_from + depth_increment
            
            # Get status
            status = self.current_tray['compartment_statuses'].get(self.current_compartment_index, None)
            
            # Build summary text
            summary_lines = [
                f"HoleID: {self.current_tray['hole_id']}",
                f"From: {int(comp_depth_from)}",
                f"To: {int(comp_depth_to)}"
            ]
            
    
            # FIXED: Only show photo status if one has been set
    
            if status:
                # Map status to Photo Status
                if status == "KEEP_ORIGINAL":
                    photo_status = "[No Change]"
                elif status == self.STATUS_MISSING:
                    photo_status = "MISSING"
                elif status in ["Wet", "Dry"]:
                    photo_status = f"OK_{status}"
                else:
                    photo_status = status
                    
                summary_lines.append(f"Photo Status: {photo_status}")
                summary_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
                summary_lines.append(f"By: {os.getenv('USERNAME', 'Unknown')}")
            else:
                # No status set yet - just show basic info
                summary_lines.append(f"Photo Status: [Not Set]")
            
            # Write summary
            self.summary_text.insert(1.0, "\n".join(summary_lines))
            self.summary_text.config(state=tk.DISABLED)
            
        except Exception as e:
            self.logger.error(f"Error updating register summary: {str(e)}")

    def _show_current_compartment(self) -> None:
        """Display the current compartment with the specified layout."""
        # Clear previous content from image frames
        for widget in self.new_frame.winfo_children():
            widget.destroy()
        
        for widget in self.existing_frame.winfo_children():
            widget.destroy()
        
        # Update progress label
        total_compartments = len(self.current_tray['compartments'])
        self.progress_label.config(
            text=DialogHelper.t(f"Compartment {self.current_compartment_index + 1}/{total_compartments}")
        )
        
        # Update navigation button states
        if hasattr(self, 'prev_button'):
            self.prev_button.set_state("normal" if self.current_compartment_index > 0 else "disabled")
        if hasattr(self, 'next_button'):
            self.next_button.set_state("normal" if self.current_compartment_index < total_compartments - 1 else "disabled")
        

        # Restore wet/dry state if compartment was already reviewed

        current_status = self.current_tray['compartment_statuses'].get(self.current_compartment_index, None)
        if current_status == "Wet":
            self.is_wet = True
            self.wet_dry_var.set("Wet")
            self.wet_dry_button.set_text(self.t("Sample Type: Wet"))
            self.wet_dry_button.color = "#3a7ca5"
            self.wet_dry_button.set_state("normal")
        elif current_status == "Dry":
            self.is_wet = False
            self.wet_dry_var.set("Dry")
            self.wet_dry_button.set_text(self.t("Sample Type: Dry"))
            self.wet_dry_button.color = "#4a8259"
            self.wet_dry_button.set_state("normal")
        

        # Update register summary

        self._update_register_summary()
        
        # Calculate depth for this compartment
        depth_from = self.current_tray['depth_from']
        depth_increment = self.app.config['compartment_interval']
        comp_depth_from = depth_from + (self.current_compartment_index * depth_increment)
        comp_depth_to = comp_depth_from + depth_increment
        compartment_depth = int(comp_depth_to)
        
        # Check if this is a selective replacement and find existing compartment image
        is_selective_replacement = self.current_tray.get('is_selective_replacement', False)
        existing_image_path = None
        
        if is_selective_replacement:
            existing_image_path = self._find_existing_compartment(
                self.current_tray['hole_id'], 
                compartment_depth
            )
        
        # Depth label for new image
        ttk.Label(
            self.new_frame,
            text=DialogHelper.t(f"Depth: {int(comp_depth_from)}-{int(comp_depth_to)}m"),
            font=("Arial", 14, "bold")
        ).pack(pady=(0, 10))
        
        # --- NEW IMAGE (current compartment) ---
        if self.current_compartment_index < len(self.current_tray['compartments']):
            current_img_data = self.current_tray['compartments'][self.current_compartment_index]
            
            self.logger.info(f"Loading compartment image for index {self.current_compartment_index}")
            self.logger.info(f"Image shape: {current_img_data.shape if hasattr(current_img_data, 'shape') else 'Not a numpy array'}")
            
            if current_img_data is not None and hasattr(current_img_data, 'shape'):
                from PIL import Image, ImageTk
                if len(current_img_data.shape) == 2:
                    display_img = cv2.cvtColor(current_img_data, cv2.COLOR_GRAY2RGB)
                else:
                    display_img = cv2.cvtColor(current_img_data, cv2.COLOR_BGR2RGB)
                
                h, w = display_img.shape[:2]
                screen_height = self.review_window.winfo_screenheight()
                max_height = int(screen_height * 0.8)
                if h > max_height:
                    scale = max_height / h
                    new_width = int(w * scale)
                    display_img = cv2.resize(display_img, (new_width, max_height), interpolation=cv2.INTER_AREA)
                
                pil_img = Image.fromarray(display_img)
                tk_img = ImageTk.PhotoImage(image=pil_img)
                
                img_label = ttk.Label(self.new_frame, image=tk_img)
                img_label.image = tk_img  # Keep reference
                img_label.pack(padx=10, pady=10)
            else:
                ttk.Label(
                    self.new_frame,
                    text=DialogHelper.t("No image available for this compartment"),
                    foreground="red",
                    font=("Arial", 12)
                ).pack(pady=50)
                self.logger.error(f"No valid image data for compartment index {self.current_compartment_index}")
        else:
            ttk.Label(
                self.new_frame,
                text=DialogHelper.t("Compartment index out of range"),
                foreground="red",
                font=("Arial", 12)
            ).pack(pady=50)
            self.logger.error(f"Compartment index {self.current_compartment_index} out of range (total: {len(self.current_tray['compartments'])})")
        
        # --- EXISTING IMAGE (optional) ---
        if existing_image_path:
            self.existing_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
            self.keep_original_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
            
            ttk.Label(
                self.existing_frame,
                text=DialogHelper.t(f"Existing Depth: {int(comp_depth_from)}-{int(comp_depth_to)}m"),
                font=("Arial", 14, "bold")
            ).pack(pady=(0, 10))
            
            try:
                existing_img = cv2.imread(existing_image_path)
                if existing_img is not None:
                    if len(existing_img.shape) == 2:
                        display_img = cv2.cvtColor(existing_img, cv2.COLOR_GRAY2RGB)
                    else:
                        display_img = cv2.cvtColor(existing_img, cv2.COLOR_BGR2RGB)
                    
                    h, w = display_img.shape[:2]
                    screen_height = self.review_window.winfo_screenheight()
                    max_height = int(screen_height * 0.8)
                    if h > max_height:
                        scale = max_height / h
                        new_width = int(w * scale)
                        display_img = cv2.resize(display_img, (new_width, max_height), interpolation=cv2.INTER_AREA)
                    
                    from PIL import Image, ImageTk
                    pil_img = Image.fromarray(display_img)
                    tk_img = ImageTk.PhotoImage(image=pil_img)
                    
                    img_label = ttk.Label(self.existing_frame, image=tk_img)
                    img_label.image = tk_img
                    img_label.pack(padx=10, pady=10)
                    
                    self.current_tray['existing_paths'] = self.current_tray.get('existing_paths', {})
                    self.current_tray['existing_paths'][self.current_compartment_index] = existing_image_path
                    
                    self.logger.info(f"Loaded existing image for comparison: {existing_image_path}")
                else:
                    ttk.Label(
                        self.existing_frame,
                        text=DialogHelper.t("Could not load existing image"),
                        foreground="red",
                        font=("Arial", 12)
                    ).pack(pady=50)
                    self.logger.warning(f"Could not load existing image: {existing_image_path}")
            except Exception as e:
                self.logger.error(f"Error loading existing compartment: {str(e)}")
                ttk.Label(
                    self.existing_frame,
                    text=DialogHelper.t(f"Error: {str(e)}"),
                    foreground="red",
                    font=("Arial", 12)
                ).pack(pady=50)
        else:
            self.existing_frame.pack_forget()
            self.keep_original_frame.pack_forget()
        
        # Update window title
        self.review_window.title(
            f"QAQC - {self.current_tray['hole_id']} {self.current_tray['depth_from']}-{self.current_tray['depth_to']} - "
            f"Compartment {self.current_compartment_index+1}/{total_compartments}"
        )

    def _find_existing_compartment(self, hole_id: str, compartment_depth: int) -> Optional[str]:
        """
        Find an existing compartment image for selective replacement comparison.
        
        Args:
            hole_id: The hole ID
            compartment_depth: The compartment depth
            
        Returns:
            Path to existing image if found, None otherwise
        """
        project_code = hole_id[:2].upper() if len(hole_id) >= 2 else ""
        
        # ===================================================
        # Check shared approved folder first
        # ===================================================
        approved_path = self.file_manager.get_shared_path('approved_compartments', create_if_missing=False)
        if approved_path:
            hole_path = os.path.join(approved_path, project_code, hole_id)
            if os.path.exists(hole_path):
                for file in os.listdir(hole_path):
                    if f"{hole_id}_CC_{compartment_depth:03d}" in file:
                        return os.path.join(hole_path, file)
        
        # ===================================================
        # Check shared review folder
        # ===================================================
        review_path = self.file_manager.get_shared_path('review_compartments', create_if_missing=False)
        if review_path:
            hole_path = os.path.join(review_path, project_code, hole_id)
            if os.path.exists(hole_path):
                for file in os.listdir(hole_path):
                    if f"{hole_id}_CC_{compartment_depth:03d}" in file:
                        return os.path.join(hole_path, file)
        
        # ===================================================
        # Check local approved compartments folder
        # ===================================================
        compartment_dir = self.file_manager.get_hole_dir("approved_compartments", hole_id)
        if os.path.exists(compartment_dir):
            for ext in ['.png', '.jpg', '.tiff', '.tif']:
                for suffix in ['_Dry', '_Wet', '_UPLOADED', '']:
                    comp_path = os.path.join(compartment_dir, f"{hole_id}_CC_{compartment_depth:03d}{suffix}{ext}")
                    if os.path.exists(comp_path):
                        return comp_path
        
        # ===================================================
        # Check local review folder
        # ===================================================
        review_dir = self.file_manager.get_hole_dir("temp_review", hole_id)
        if os.path.exists(review_dir):
            for ext in ['.png', '.jpg', '.tiff', '.tif']:
                for suffix in ['_temp', '_new', '_review', '']:
                    comp_path = os.path.join(review_dir, f"{hole_id}_CC_{compartment_depth:03d}{suffix}{ext}")
                    if os.path.exists(comp_path):
                        return comp_path
        
        return None
    
    def _set_status_and_next(self, status):
        """Set the status for the current compartment and move to the next one."""

        # CHANGED: Include wet/dry in status if OK

        if status == self.STATUS_OK:
            status = "Wet" if self.is_wet else "Dry"
        
        # Save current status
        self.current_tray['compartment_statuses'][self.current_compartment_index] = status
        
        # Check if we have more compartments to review
        if self.current_compartment_index < len(self.current_tray['compartments']) - 1:
            # Move to next compartment
            self.current_compartment_index += 1
            self._show_current_compartment()
        else:
    
            # CHANGED: Save and move to next tray silently
    
            self._complete_current_tray()
    
    def _set_keep_original_and_next(self) -> None:
        """Mark the current compartment to keep the original and move to next."""
        # Store keep_original flag in the status dictionary
        self.current_tray['compartment_statuses'][self.current_compartment_index] = "KEEP_ORIGINAL"
        
        # Also store the existing image path if we have it
        if hasattr(self.current_tray, 'existing_paths') and self.current_compartment_index in self.current_tray.get('existing_paths', {}):
            self.current_tray['kept_original_paths'] = self.current_tray.get('kept_original_paths', {})
            self.current_tray['kept_original_paths'][self.current_compartment_index] = self.current_tray['existing_paths'][self.current_compartment_index]
            self.logger.info(f"Marked compartment {self.current_compartment_index+1} to keep original image")
        
        # Move to the next compartment
        if self.current_compartment_index < len(self.current_tray['compartments']) - 1:
            self.current_compartment_index += 1
            self._show_current_compartment()
        else:
            # Save and move to next tray silently
            self._complete_current_tray()
    
    def _complete_current_tray(self):
        """Complete processing of current tray and move to next."""
        # Save approved compartments
        self._save_approved_compartments()
        
        # ===================================================
        # Copy original to shared folder if applicable
        # ===================================================
        if self.current_tray['original_path'] not in ["From Temp_Review folder", "From Approved folder"]:
            self._copy_original_to_shared( #TODO - FIX - this doesn't exist and shouldn't exist here - the original image will be moved at the end of processing in the main app?
                self.current_tray['original_path'],
                self.current_tray['hole_id'],
                self.current_tray['depth_from'],
                self.current_tray['depth_to']
            )
        
        # Process next tray
        self._review_next_tray()
    
    def _on_review_window_close(self):
        """Handle review window close event."""
        # ===================================================
        # CHANGED: Simplified close handling
        # ===================================================
        if DialogHelper.confirm_dialog(
            self.review_window,
            DialogHelper.t("Unsaved Changes"),
            DialogHelper.t("There are unsaved changes. Save before closing?"),
            yes_text=DialogHelper.t("Save & Close"),
            no_text=DialogHelper.t("Close Without Saving")
        ):
            # Save current progress
            if self.current_compartment_index > 0:
                # Only save compartments that have been reviewed
                reviewed_compartments = self.current_tray['compartments'][:self.current_compartment_index]
                reviewed_statuses = {i: self.current_tray['compartment_statuses'].get(i, self.STATUS_OK) 
                                for i in range(self.current_compartment_index)}
                
                # Update current tray to only include reviewed items
                original_compartments = self.current_tray['compartments']
                original_statuses = self.current_tray['compartment_statuses']
                
                self.current_tray['compartments'] = reviewed_compartments
                self.current_tray['compartment_statuses'] = reviewed_statuses
                
                # Save the reviewed compartments
                self._save_approved_compartments()

                
                # Restore original data for unreviewed compartments
                self.current_tray['compartments'] = original_compartments[self.current_compartment_index:]
                self.current_tray['compartment_statuses'] = {
                    i-self.current_compartment_index: original_statuses.get(i, self.STATUS_OK) 
                    for i in range(self.current_compartment_index, len(original_compartments))
                }
                
                # Check if there are unreviewed compartments
                if self.current_tray['compartments']:
                    # ===================================================
                    # Ask user if they want to move unreviewed compartments to shared folder
                    # ===================================================
                    if DialogHelper.confirm_dialog(
                        self.review_window,
                        DialogHelper.t("Unreviewed Compartments"),
                        DialogHelper.t("There are unreviewed compartments. Move them to shared folder for later review?").format(
                            len(self.current_tray['compartments'])
                        ),
                        yes_text=DialogHelper.t("Move to Shared Folder"),
                        no_text=DialogHelper.t("Keep Local Only")
                    ):
                        # Move unreviewed compartments to shared folder
                        self._move_unreviewed_to_shared()
                    else:
                        # Re-add current tray to pending if keeping local
                        self.pending_trays.insert(0, self.current_tray)
        
        # Show summary
        self._show_final_summary()
        
        # Destroy window
        self.review_window.destroy()
        self.review_window = None


    def _check_approved_vs_register(self):
        """Check for discrepancies between approved folder and register."""
        try:
            # ===================================================
            # Get shared paths using FileManager
            # ===================================================
            approved_path = self.file_manager.get_shared_path('approved_compartments', create_if_missing=False)
            register_path = self.file_manager.get_shared_path('register_excel', create_if_missing=False)
            
            if not approved_path or not register_path:
                self.logger.warning("Cannot sync - shared folder paths not configured")
                return
                
            # Get compartment data from JSON register
            register_df = self.register_manager.get_compartment_data()
            
            if register_df.empty:
                self.logger.info("Register is empty - nothing to check")

            else:
                # Find entries that need QAQC review
                needs_qaqc = register_df[
                    (register_df['Photo_Status'].isin(self.NEEDS_REVIEW_STATUSES) | 
                    register_df['Photo_Status'].isna()) &
                    ~register_df['Photo_Status'].isin(self.SKIP_STATUSES)
                ]
                
                if not needs_qaqc.empty:
                    self.logger.info(f"Found {len(needs_qaqc)} register entries needing QAQC review")
                    
                    # Group by hole
                    holes_needing_review = {}
                    for _, row in needs_qaqc.iterrows():
                        hole_id = row['HoleID']
                        depth_to = int(row['To'])
                        
                        if hole_id not in holes_needing_review:
                            holes_needing_review[hole_id] = []
                        holes_needing_review[hole_id].append((depth_to, row))
                    
                    # Create review entries for each hole
                    for hole_id, depth_data in holes_needing_review.items():
                        # Sort by depth
                        depth_data.sort(key=lambda x: x[0])
                        
                        # Load compartment images from approved folder
                        compartments = []
                        temp_paths = []
                        depths = []
                        
                        project_code = hole_id[:2].upper() if len(hole_id) >= 2 else ""
                        hole_path = os.path.join(approved_path, project_code, hole_id)
                        if os.path.exists(hole_path):
                            for depth_to, row in depth_data:
                                # Look for compartment image
                                found = False
                                for file in os.listdir(hole_path):
                                    if f"{hole_id}_CC_{depth_to:03d}" in file:
                                        file_path = os.path.join(hole_path, file)
                                        img = cv2.imread(file_path)
                                        if img is not None:
                                            compartments.append(img)
                                            temp_paths.append(file_path)
                                            depths.append(depth_to)
                                            found = True
                                            break
                                
                                if not found:
                                    self.logger.warning(f"Could not find image for {hole_id} depth {depth_to}")
                        
                        if compartments:
                            # Create tray entry
                            depth_increment = self.app.config['compartment_interval']
                            min_depth = min(depths) - depth_increment
                            max_depth = max(depths)
                            
                            tray_entry = {
                                'hole_id': hole_id,
                                'depth_from': min_depth,
                                'depth_to': max_depth,
                                'original_path': "From Register - Needs QAQC",
                                'compartments': compartments,
                                'temp_paths': temp_paths,
                                'compartment_statuses': {},
                                'from_register_review': True  # Flag to identify these
                            }
                            
                            self.pending_trays.append(tray_entry)
                            self.logger.info(f"Added {len(compartments)} compartments from {hole_id} for QAQC review")

            
            # Get all compartments from approved folder
            approved_compartments = set()
            if os.path.exists(approved_path):
                # Look for project code directories first
                for project_code in os.listdir(approved_path):
                    project_path = os.path.join(approved_path, project_code)
                    if os.path.isdir(project_path) and re.match(r'^[A-Z]{2}$', project_code):
                        # Then look for hole IDs within project folder
                        for hole_id in os.listdir(project_path):
                            hole_path = os.path.join(project_path, hole_id)
                            if os.path.isdir(hole_path) and re.match(r'^[A-Z]{2}\d{4}$', hole_id):
                                # Now look for compartment files within the hole directory
                                for file in os.listdir(hole_path):
                                    match = re.search(r'([A-Za-z]{2}\d{4})_CC_(\d{3})(?:_Dry|_Wet)?\.(?:png|tiff|jpg)', file)
                                    if match:
                                        hole_id = match.group(1)
                                        depth = int(match.group(2))
                                        approved_compartments.add((hole_id, depth))
            
            # ===================================================
            # Check for compartments in review folder
            # ===================================================
            review_path = self.file_manager.get_shared_path('review_compartments', create_if_missing=False)
            if review_path and os.path.exists(review_path):
                review_compartments = []
                # Look for project code directories
                for project_code in os.listdir(review_path):
                    project_path = os.path.join(review_path, project_code)
                    if os.path.isdir(project_path) and re.match(r'^[A-Z]{2}$', project_code):
                        # Look for hole IDs
                        for hole_id in os.listdir(project_path):
                            hole_path = os.path.join(project_path, hole_id)
                            if os.path.isdir(hole_path) and re.match(r'^[A-Z]{2}\d{4}$', hole_id):
                                # Count review files
                                review_files = [f for f in os.listdir(hole_path) 
                                            if f.endswith('_review.png')]
                                if review_files:
                                    review_compartments.append((hole_id, len(review_files)))
                
                if review_compartments:
                    self.logger.info(f"Found compartments awaiting review in shared folder:")
                    for hole_id, count in review_compartments:
                        self.logger.info(f"  {hole_id}: {count} compartments")
            
            # Get all compartments from register
            register_compartments = set()
            for _, row in register_df.iterrows():
                if pd.notna(row.get('HoleID')) and pd.notna(row.get('To')):
                    register_compartments.add((row['HoleID'], int(row['To'])))
            
            # Find compartments in approved but not in register
            unregistered = approved_compartments - register_compartments
            
            # Find compartments in register but not in approved
            missing = register_compartments - approved_compartments
            
            # Update register for missing compartments
            if missing:
                self.logger.info(f"Found {len(missing)} compartments in register but not in approved folder")
                for hole_id, depth in missing:
                    # ===================================================
                    # Use register_manager to update status instead of direct DataFrame manipulation
                    # ===================================================
                    self.register_manager.update_compartment(
                        hole_id=hole_id,
                        depth_from=depth - self.app.config['compartment_interval'],
                        depth_to=depth,
                        photo_status='MISSING',
                        comments='Not found in approved folder during sync check'
                    )
                    self.stats['register_updated'] += 1
            
            # Create review entries for unregistered compartments
            if unregistered:
                self.logger.info(f"Found {len(unregistered)} compartments in approved folder but not in register")
                
                # Group by hole
                holes = {}
                for hole_id, depth in unregistered:
                    if hole_id not in holes:
                        holes[hole_id] = []
                    holes[hole_id].append(depth)
                
                # Create review entries
                for hole_id, depths in holes.items():
                    depths.sort()
                    
                    # Load compartment images
                    compartments = []
                    temp_paths = []
                    
                    for depth in depths:
                        # Find the file with project code organization
                        project_code = hole_id[:2].upper() if len(hole_id) >= 2 else ""
                        hole_path = os.path.join(approved_path, project_code, hole_id)
                        
                        if os.path.exists(hole_path):
                            for file in os.listdir(hole_path):
                                if f"{hole_id}_CC_{depth:03d}" in file:
                                    file_path = os.path.join(hole_path, file)
                                    img = cv2.imread(file_path)
                                    if img is not None:
                                        compartments.append(img)
                                        temp_paths.append(file_path)
                                        break
                    
                    if compartments:
                        # Create tray entry
                        depth_increment = self.app.config['compartment_interval']
                        min_depth = min(depths) - depth_increment
                        max_depth = max(depths)
                        
                        tray_entry = {
                            'hole_id': hole_id,
                            'depth_from': min_depth,
                            'depth_to': max_depth,
                            'original_path': "From Approved folder",
                            'compartments': compartments,
                            'temp_paths': temp_paths,
                            'compartment_statuses': {},
                            'auto_approve': True  # Flag to auto-approve these
                        }
                        
                        self.pending_trays.append(tray_entry)
            
            # ===================================================
            # REMOVED: _save_register_with_lock call (method doesn't exist)
            # Register updates are now handled through register_manager
            # ===================================================
                
        except Exception as e:
            self.logger.error(f"Error checking approved vs register: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _move_unreviewed_to_shared(self):
        """Move unreviewed compartments to shared Compartment Images for Review folder."""
        try:
            # ===================================================
            # Get the shared review compartments folder
            # ===================================================
            review_path = self.file_manager.get_shared_path('review_compartments', create_if_missing=True)
            if not review_path:
                self.logger.warning("Shared review compartments path not configured")
                return
            
            hole_id = self.current_tray['hole_id']
            project_code = hole_id[:2].upper() if len(hole_id) >= 2 else ""
            
            # Create project code and hole ID folders
            shared_hole_path = os.path.join(review_path, project_code, hole_id)
            os.makedirs(shared_hole_path, exist_ok=True)
            
            # Get starting index for unreviewed compartments
            start_index = self.current_compartment_index
            depth_from = self.current_tray['depth_from']
            depth_increment = self.app.config['compartment_interval']
            
            moved_count = 0
            for i, compartment in enumerate(self.current_tray['compartments']):
                try:
                    # Calculate compartment depth
                    actual_index = start_index + i
                    comp_depth_from = depth_from + (actual_index * depth_increment)
                    comp_depth_to = comp_depth_from + depth_increment
                    compartment_depth = int(comp_depth_to)
                    
                    # Create filename
                    filename = f"{hole_id}_CC_{compartment_depth:03d}_review.png"
                    file_path = os.path.join(shared_hole_path, filename)
                    
                    # Save the compartment image
                    cv2.imwrite(file_path, compartment)
                    moved_count += 1
                    
                    self.logger.info(f"Moved unreviewed compartment to shared folder: {filename}")
                    
                except Exception as e:
                    self.logger.error(f"Error moving compartment {actual_index+1} to shared folder: {str(e)}")
            
            # Clean up local temp files
            if 'temp_paths' in self.current_tray:
                for temp_path in self.current_tray['temp_paths'][start_index:]:
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except Exception as e:
                            self.logger.warning(f"Could not remove temp file {temp_path}: {str(e)}")
            
            # Update stats
            self.stats['saved_locally'] += moved_count
            
            # Show message to user
            DialogHelper.show_message(
                self.review_window,
                DialogHelper.t("Compartments Moved"),
                DialogHelper.t("{} unreviewed compartments moved to shared folder for later review.").format(moved_count),
                message_type="info"
            )
            
            self.logger.info(f"Moved {moved_count} unreviewed compartments to shared review folder")
            
        except Exception as e:
            self.logger.error(f"Error moving unreviewed compartments to shared folder: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _save_approved_compartments(self):
        """Save approved compartments using FileManager."""
        if not self.current_tray:
            return
        
        hole_id = self.current_tray['hole_id']
        depth_from = self.current_tray['depth_from']
        interval = self.app.config['compartment_interval']
        
        # Save each compartment
        for i, compartment in enumerate(self.current_tray['compartments']):
            # Calculate depth
            comp_depth_from = depth_from + (i * interval)
            comp_depth_to = comp_depth_from + interval
            compartment_depth = int(comp_depth_to)
            
            # Get status
            status = self.current_tray['compartment_statuses'].get(i, self.STATUS_OK)
            
            # Save using FileManager
            result = self.file_manager.save_reviewed_compartment(
                image=compartment,
                hole_id=hole_id,
                compartment_depth=compartment_depth,
                status=status,
                output_format=self.app.config['output_format']
            )
            
            # Update stats
            if result['local_path']:
                self.stats['saved_locally'] += 1
                self.stats['processed'] += 1
                
            if result['upload_success']:
                self.stats['uploaded'] += 1
            elif result['local_path'] and not result['upload_success']:
                self.stats['upload_failed'] += 1
        
        # Update register with corners
        self._update_register()
        
        # Cleanup temp files using FileManager
        self.file_manager.cleanup_temp_compartments(
            hole_id=self.current_tray['hole_id'],
            temp_paths=self.current_tray.get('temp_paths', [])
        )

    def _update_register(self):
        """Update register with compartment statuses and corners."""
        if not self.current_tray:
            return
        
        try:
            # Prepare batch updates
            updates = []
            
            hole_id = self.current_tray['hole_id']
            depth_from = self.current_tray['depth_from']
            interval = self.app.config['compartment_interval']
            corners_list = self.current_tray.get('corners_list', [])
            
            # Build updates for each compartment
            for i, compartment in enumerate(self.current_tray['compartments']):
                # Get status
                status = self.current_tray['compartment_statuses'].get(i, self.STATUS_OK)
                
                # Skip if keeping original
                if status == "KEEP_ORIGINAL":
                    continue
                
                # Calculate depth
                comp_depth_from = depth_from + (i * interval)
                comp_depth_to = comp_depth_from + interval
                
                # Map status to photo status
                if status == self.STATUS_MISSING:
                    photo_status = "MISSING"
                elif status in ["Wet", "Dry"]:
                    photo_status = f"OK_{status}"
                else:
                    photo_status = status
                
                # Build update
                update = {
                    'hole_id': hole_id,
                    'depth_from': int(comp_depth_from),
                    'depth_to': int(comp_depth_to),
                    'photo_status': photo_status
                }
                
                if corners_list and i < len(corners_list):
                    update['corners'] = corners_list[i]
                
                updates.append(update)
            
            # Batch update compartments
            if updates:
                updated = self.register_manager.batch_update_compartments(updates)
                self.stats['register_updated'] += updated
                self.stats['register_failed'] += len(updates) - updated
                
                if updated < len(updates):
                    self.logger.warning(f"Only {updated}/{len(updates)} register updates succeeded")
            
            # Update original image entry in register
            if self.current_tray['original_path'] not in ["From Temp_Review folder", "From Approved folder"]:
                success = self.register_manager.update_original_image(
                    hole_id=hole_id,
                    depth_from=self.current_tray['depth_from'],
                    depth_to=self.current_tray['depth_to'],
                    original_filename=os.path.basename(self.current_tray['original_path']),
                    is_approved=True,
                    upload_success=True
                )
                if not success:
                    self.logger.error("Failed to update original image register entry")
                    
        except Exception as e:
            self.logger.error(f"Error updating register: {e}")
            self.logger.error(traceback.format_exc())
            self.stats['register_failed'] += len(self.current_tray['compartments'])

    def _update_register(self):
            """Update the register with the approved compartments."""
            try:
                if not self.current_tray:
                    return
                    
                # Prepare data
                hole_id = self.current_tray['hole_id']
                depth_from = self.current_tray['depth_from']
                compartment_interval = self.app.config['compartment_interval']
                
                # Prepare batch updates
                updates = []
                
                # Update each compartment
                for i, compartment in enumerate(self.current_tray['compartments']):
                    # Calculate compartment depth
                    comp_depth_from = depth_from + (i * compartment_interval)
                    comp_depth_to = comp_depth_from + compartment_interval
                    
                    # Get status
                    status = self.current_tray['compartment_statuses'].get(i, self.STATUS_OK)
                    
                    # Skip if keeping original
                    if status == "KEEP_ORIGINAL":
                        continue
                    
                    # Map status to Photo Status
                    if status == self.STATUS_MISSING:
                        photo_status = "MISSING"
                    elif status in ["Wet", "Dry"]:
                        photo_status = f"OK_{status}"
                    else:
                        photo_status = status
                    
                    # Add to batch
                    updates.append({
                        'hole_id': hole_id,
                        'depth_from': int(comp_depth_from),
                        'depth_to': int(comp_depth_to),
                        'photo_status': photo_status
                    })
                
                # Batch update
                if updates:
                    updated = self.register_manager.batch_update_compartments(updates)
                    self.stats['register_updated'] += updated
                    self.stats['register_failed'] += len(updates) - updated
                    
                    if updated < len(updates):
                        self.logger.warning(f"Only {updated}/{len(updates)} register updates succeeded")
                
                # Update original image entry if this was from a real image
                if self.current_tray['original_path'] not in ["From Temp_Review folder", "From Approved folder"]:
                    original_filename = os.path.basename(self.current_tray['original_path'])
                    success = self.register_manager.update_original_image(
                        hole_id,
                        self.current_tray['depth_from'],
                        self.current_tray['depth_to'],
                        original_filename,
                        is_approved=True,
                        upload_success=True
                    )
                    if not success:
                        self.logger.error("Failed to update original image register entry")
                        
            except Exception as e:
                self.logger.error(f"Error updating register: {str(e)}")
                self.logger.error(traceback.format_exc())
                self.stats['register_failed'] += len(self.current_tray['compartments']) if self.current_tray else 0
    
    def _show_final_summary(self):
        """Show final processing summary and update main GUI."""
        # Create summary message
        summary_lines = [
            f"QAQC Processing Complete:",
            f"- Compartments processed: {self.stats['processed']}",
            f"- Successfully uploaded to OneDrive: {self.stats['uploaded']}",
            f"- Failed OneDrive uploads: {self.stats['upload_failed']}",
            f"- Saved locally: {self.stats['saved_locally']}",
            f"- Register entries updated: {self.stats['register_updated']}",
            f"- Register update failures: {self.stats['register_failed']}"
        ]
        
        summary_message = "\n".join(summary_lines)
        
        # Update main GUI if available
        if hasattr(self, 'main_gui') and self.main_gui:
            self.main_gui.direct_status_update(summary_message, status_type="info")
            
            # Log summary
            self.logger.info(summary_message)