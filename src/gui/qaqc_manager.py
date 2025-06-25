import os
import re
from pathlib import Path
from datetime import datetime
from tkinter import ttk
import tkinter as tk
from typing import List, Dict
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageTk

# Local imports
from utils.json_register_manager import JSONRegisterManager
from gui.dialog_helper import DialogHelper

if threading.current_thread() != threading.main_thread():
    raise RuntimeError("❌ QAQCManager called from a background thread!")


@dataclass
class ReviewItem:
    filename: str
    hole_id: str
    depth_from: int
    depth_to: int
    compartment_depth: int  # The "To" value used in naming
    image_path: str
    duplicate_count: int = 0
    duplicate_paths: List[str] = field(default_factory=list)
    moisture: str = "unknown"  # wet/dry/unknown
    quality: str = "unreviewed"  # OK/Blurry/Damaged/Missing/unreviewed
    average_hex_color: str = ""
    is_reviewed: bool = False
    register_status: Optional[str] = None  # Current status from register
    _image: Optional[np.ndarray] = None  # Lazy loaded

    @property
    def image(self) -> Optional[np.ndarray]:
        """Lazy load image only when accessed."""
        if self._image is None and self.image_path:
            self._image = cv2.imread(self.image_path)
        return self._image

    def unload_image(self):
        """Free memory by unloading the image."""
        self._image = None


class QAQCManager:
    """
    Manages quality assurance and quality control of extracted compartment images.
    Provides a GUI for reviewing images and approving/rejecting them.
    Handles synchronization between local storage, OneDrive, and register.
    """

    # 🔧 Setup & Utilities
    def __init__(self, file_manager, translator_func, config_manager,
                app, logger, register_manager):
        """Initialize the QAQC Manager."""
        self.root = app.root
        self.file_manager = file_manager
        self.t = translator_func
        self.config_manager = config_manager
        self.app = app
        self.logger = logger
        self.register_manager = register_manager

        # New attributes for refactored approach
        self.compartment_register: Dict[str, pd.DataFrame] = {}
        self.current_hole_items: List[ReviewItem] = []
        self.current_item_index: int = 0

        # Review window
        self.review_window = None

        # Constants for status values
        self.STATUS_OK = "OK"
        self.STATUS_DAMAGED = "Damaged"
        self.STATUS_EMPTY = "Empty"
        self.STATUS_WET = "Wet"
        self.STATUS_DRY = "Dry"

        # Valid QAQC statuses
        self.VALID_QAQC_STATUSES = [
            "OK_Wet", "OK_Dry", "Damaged_Wet", "Damaged_Dry", "Empty"
        ]

        # Statuses needing review
        self.NEEDS_REVIEW_STATUSES = [
            "Found", None, "", "Not Set"
        ]

        # Skip statuses
        self.SKIP_STATUSES = [
            "MISSING_FILE", "Not Found", "MISSING"
        ]

        # Processing statistics
        self.stats = {
            'processed': 0,
            'uploaded': 0,
            'upload_failed': 0,
            'saved_locally': 0,
            'register_updated': 0,
            'register_failed': 0
        }

        # Main GUI reference
        self.main_gui = app.main_gui if hasattr(app, 'main_gui') else None

        # Regex patterns
        self.HOLE_ID_PATTERN = re.compile(r'^[A-Z]{2}\d{4}$')
        self.PROJECT_CODE_PATTERN = re.compile(r'^[A-Z]{2}$')
        self.COMPARTMENT_FILE_PATTERN = re.compile(
            r'([A-Z]{2}\d{4})_CC_(\d{3})(?:_temp|_new|_review|_Wet|_Dry)?\.(?:png|tiff|jpg)$',
            re.IGNORECASE
        )

    def set_main_gui(self, main_gui):
        """Set reference to main GUI for status updates."""
        self.main_gui = main_gui

    def t(self, text):
        """Translate text using DialogHelper."""
        return DialogHelper.t(text)

    # 🚀 Review Entry Point
    def start_review_process(self):
        """Refactored review process with better memory management."""
        # Reset statistics
        self.stats = {k: 0 for k in self.stats}

        # Step 1: Load register data
        self.compartment_register = self._load_register_into_memory()

        # Step 2: Build review queue from local files
        local_review_items = self._scan_local_review_folder()

        if local_review_items:
            # Process local files by hole
            self._process_review_queue(local_review_items)
        else:
            # Step 3: Check shared folders for items needing review
            shared_review_items = self._scan_shared_folders_for_review()

            if shared_review_items:
                # Move one hole at a time to local and process
                self._process_shared_items(shared_review_items)
            else:
                self._show_nothing_to_review_message()

    # 📥 Data Loading & Scanning
    def _load_register_into_memory(self) -> Dict[str, pd.DataFrame]:
        """Load compartment register efficiently."""
        df = self.register_manager.get_compartment_data()

        if df.empty:
            return {}

        # Only keep relevant columns
        relevant_columns = ["HoleID", "From", "To", "Photo_Status"]
        if all(col in df.columns for col in relevant_columns):
            df = df[relevant_columns]
        else:
            self.logger.warning("Register missing expected columns")
            return {}

        # Index by HoleID for fast lookups
        return {hole_id: group for hole_id, group in df.groupby('HoleID')}

    def _scan_local_review_folder(self) -> Dict[str, List[ReviewItem]]:
        """Scan local temp_review folder and identify duplicates upfront."""
        review_items_by_hole = {}
        
        # Use dir_structure to get the temp_review path
        temp_review_path = self.file_manager.dir_structure["temp_review"]
        
        if not os.path.exists(temp_review_path):
            return review_items_by_hole
        
        # Update status
        if self.main_gui:
            self.main_gui.direct_status_update("Scanning local review folder...", status_type="info")
        
        # Scan for all compartment files
        all_files = self._find_all_compartment_files(temp_review_path)

        # Group by hole and interval
        files_by_interval = self._group_files_by_interval(all_files)

        # Check for duplicates across all locations
        for (hole_id, depth_to), file_paths in files_by_interval.items():
            # Find existing files in other locations
            duplicate_paths = self._find_duplicates_in_all_locations(
                hole_id, depth_to, exclude_paths=file_paths
            )

            # Create ReviewItem
            review_item = ReviewItem(
                filename=os.path.basename(file_paths[0]),
                hole_id=hole_id,
                depth_from=depth_to - self.app.config['compartment_interval'],
                depth_to=depth_to,
                compartment_depth=depth_to,
                image_path=file_paths[0],  # Primary file
                duplicate_count=len(duplicate_paths),
                duplicate_paths=duplicate_paths,
                register_status=self._get_register_status(hole_id, depth_to)
            )

            if hole_id not in review_items_by_hole:
                review_items_by_hole[hole_id] = []
            review_items_by_hole[hole_id].append(review_item)

        # Log findings
        total_items = sum(len(items) for items in review_items_by_hole.values())
        self.logger.info(f"Found {total_items} compartments in {len(review_items_by_hole)} holes for review")

        return review_items_by_hole

    def _find_all_compartment_files(self, base_path: str) -> List[Tuple[str, str]]:
        """Find all compartment files in a directory tree."""
        compartment_files = []

        for root, dirs, files in os.walk(base_path):
            for file in files:
                match = self.COMPARTMENT_FILE_PATTERN.match(file)
                if match:
                    full_path = os.path.join(root, file)
                    compartment_files.append((full_path, file))

        return compartment_files

    def _group_files_by_interval(self, files: List[Tuple[str, str]]) -> Dict[Tuple[str, int], List[str]]:
        """Group files by hole_id and depth interval."""
        files_by_interval = {}

        for full_path, filename in files:
            match = self.COMPARTMENT_FILE_PATTERN.match(filename)
            if match:
                hole_id = match.group(1)
                depth = int(match.group(2))
                key = (hole_id, depth)

                if key not in files_by_interval:
                    files_by_interval[key] = []
                files_by_interval[key].append(full_path)

        return files_by_interval

    def _get_register_status(self, hole_id: str, depth_to: int) -> Optional[str]:
        """Get the current Photo_Status from register for a compartment."""
        if hole_id in self.compartment_register:
            df = self.compartment_register[hole_id]
            matching = df[df['To'] == depth_to]
            if not matching.empty:
                return matching.iloc[0].get('Photo_Status')
        return None

    def _find_duplicates_in_all_locations(self, hole_id: str, depth: int,
                                        exclude_paths: List[str]) -> List[str]:
        """Find all duplicate files for a compartment across all locations."""
        duplicates = []

        # Define search locations using correct path access methods
        locations = [
            ('temp_review', self.file_manager.dir_structure["temp_review"]),
            ('approved_local', self.file_manager.dir_structure["approved_compartments"]),
            ('review_shared', self.file_manager.get_shared_path('review_compartments')),
            ('approved_shared', self.file_manager.get_shared_path('approved_compartments'))
        ]

        for location_name, base_path in locations:
            if not base_path or not os.path.exists(base_path):
                continue

            # Build expected path
            project_code = hole_id[:2]
            search_path = os.path.join(base_path, project_code, hole_id)

            if os.path.exists(search_path):
                # Look for files matching this interval
                pattern = f"{hole_id}_CC_{depth:03d}"
                try:
                    for file in os.listdir(search_path):
                        if pattern in file and file.endswith(('.png', '.tiff', '.jpg')):
                            full_path = os.path.join(search_path, file)
                            if full_path not in exclude_paths:
                                duplicates.append(full_path)
                except Exception as e:
                    self.logger.warning(f"Error scanning {search_path}: {e}")

        return duplicates

    def _scan_shared_folders_for_review(self) -> Dict[str, List[str]]:
        """Scan shared folders for compartments needing review."""
        items_by_hole = {}

        # Check review folder first
        review_path = self.file_manager.get_shared_path('review_compartments', create_if_missing=False)
        if review_path and os.path.exists(review_path):
            self._scan_shared_folder(review_path, items_by_hole, "review")

        # Check approved folder for items needing QAQC
        approved_path = self.file_manager.get_shared_path('approved_compartments', create_if_missing=False)
        if approved_path and os.path.exists(approved_path):
            self._scan_shared_folder(approved_path, items_by_hole, "approved")

        return items_by_hole

    def _scan_shared_folder(self, base_path: str, items_by_hole: Dict[str, List[str]], folder_type: str):
        """Scan a shared folder for compartments needing review."""
        # Look for project code directories
        for project_code in os.listdir(base_path):
            if not self.PROJECT_CODE_PATTERN.match(project_code):
                continue

            project_path = os.path.join(base_path, project_code)
            if not os.path.isdir(project_path):
                continue

            # Look for hole IDs
            for hole_id in os.listdir(project_path):
                if not self.HOLE_ID_PATTERN.match(hole_id):
                    continue

                hole_path = os.path.join(project_path, hole_id)
                if not os.path.isdir(hole_path):
                    continue

                # Find compartment files
                for file in os.listdir(hole_path):
                    match = self.COMPARTMENT_FILE_PATTERN.match(file)
                    if match:
                        depth = int(match.group(2))

                        # Check if this needs review
                        register_status = self._get_register_status(hole_id, depth)

                        if folder_type == "review" or register_status in self.NEEDS_REVIEW_STATUSES:
                            full_path = os.path.join(hole_path, file)

                            if hole_id not in items_by_hole:
                                items_by_hole[hole_id] = []
                            items_by_hole[hole_id].append(full_path)

    def _process_review_queue(self, items_by_hole: Dict[str, List[ReviewItem]]):
        """Process review items one hole at a time."""
        total_holes = len(items_by_hole)

        for idx, (hole_id, review_items) in enumerate(items_by_hole.items()):
            # Update status
            if self.main_gui:
                self.main_gui.direct_status_update(
                    f"Processing hole {idx + 1}/{total_holes}: {hole_id}",
                    status_type="info"
                )

            # Sort by depth
            review_items.sort(key=lambda x: x.depth_to)

            # Set current hole items
            self.current_hole_items = review_items
            self.current_item_index = 0

            # Show review GUI
            self._show_review_window_for_hole(hole_id)

            # After hole is complete, batch process
            self._batch_process_reviewed_items(review_items)

            # Free memory
            for item in review_items:
                item.unload_image()

    def _show_review_window_for_hole(self, hole_id: str):
        """Show the review window for a specific hole."""
        # Build a compatibility structure for the existing GUI
        if not self.current_hole_items:
            return

        # Calculate depth range from items
        min_depth = min(item.depth_from for item in self.current_hole_items)
        max_depth = max(item.depth_to for item in self.current_hole_items)

        # Create a "tray" structure that the GUI expects
        self.current_tray = {
            'hole_id': hole_id,
            'depth_from': min_depth,
            'depth_to': max_depth,
            'original_path': "From QAQC Review",
            'compartments': [],  # Will be populated on demand
            'temp_paths': [item.image_path for item in self.current_hole_items],
            'compartment_statuses': {},
            'review_items': self.current_hole_items  # Keep reference to ReviewItems
        }

        # Reset index
        self.current_compartment_index = 0

        # Create or update review window
        if not self.review_window or not self.review_window.winfo_exists():
            self._create_review_window()

        # Show first compartment
        self._show_current_compartment_new()


    def _process_shared_items(self, shared_items_by_hole: Dict[str, List[str]]):
        """Move shared items to local one hole at a time and process."""
        for hole_id, file_paths in shared_items_by_hole.items():
            # Update status
            if self.main_gui:
                self.main_gui.direct_status_update(
                    f"Moving {len(file_paths)} files for {hole_id} to local review...",
                    status_type="info"
                )

            # Move files to local temp_review
            local_paths = self._move_files_to_local(hole_id, file_paths)

            if local_paths:
                # Now scan and process as local files
                local_items = self._scan_specific_files(local_paths)

                # Process this hole
                self._process_review_queue({hole_id: local_items})

    def _move_files_to_local(self, hole_id: str, file_paths: List[str]) -> List[str]:
        """Move files from shared location to local temp_review folder."""
        local_paths = []

        # Get local temp_review path
        temp_review_path = self.file_manager.get_local_path('temp_review')
        project_code = hole_id[:2]
        local_hole_path = os.path.join(temp_review_path, project_code, hole_id)

        # Create directory
        os.makedirs(local_hole_path, exist_ok=True)

        for src_path in file_paths:
            try:
                filename = os.path.basename(src_path)
                dst_path = os.path.join(local_hole_path, filename)

                # Copy file
                import shutil
                shutil.copy2(src_path, dst_path)
                local_paths.append(dst_path)

                # Remove from source
                try:
                    os.remove(src_path)
                except Exception as e:
                    self.logger.warning(f"Could not remove source file {src_path}: {e}")

            except Exception as e:
                self.logger.error(f"Error moving file {src_path}: {e}")

        return local_paths

    def _scan_specific_files(self, file_paths: List[str]) -> List[ReviewItem]:
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

                # Find duplicates
                duplicate_paths = self._find_duplicates_in_all_locations(
                    hole_id, depth, exclude_paths=[file_path]
                )

                review_item = ReviewItem(
                    filename=filename,
                    hole_id=hole_id,
                    depth_from=depth - self.app.config['compartment_interval'],
                    depth_to=depth,
                    compartment_depth=depth,
                    image_path=file_path,
                    duplicate_count=len(duplicate_paths),
                    duplicate_paths=duplicate_paths,
                    register_status=self._get_register_status(hole_id, depth)
                )

                review_items.append(review_item)

        return review_items

    def _show_nothing_to_review_message(self):
        """Show message when there's nothing to review."""
        DialogHelper.show_message(
            self.root,
            self.t("Review Complete"),
            self.t("No images to review and all files are synchronized."),
            message_type="info"
        )

    # 💾 Saving & Batch Processing
    def _batch_process_reviewed_items(self, review_items: List[ReviewItem]):
        """Batch process all reviewed items for efficiency."""
        # Separate by action needed
        to_save = []
        to_update_register = []
        to_delete = []

        for item in review_items:
            if not item.is_reviewed:
                continue

            if item.quality in ["OK", "Damaged"]:
                to_save.append(item)
                to_update_register.append(item)
            elif item.quality == "Empty":
                to_update_register.append(item)
                to_delete.append(item.image_path)

        # Batch save files
        if to_save:
            self._batch_save_compartments(to_save)

        # Batch update register
        if to_update_register:
            self._batch_update_register(to_update_register)

        # Clean up files
        if to_delete:
            self._batch_delete_files(to_delete)

    def _batch_save_compartments(self, items: List[ReviewItem]):
        """Batch save reviewed compartments."""
        for item in items:
            try:
                # Determine final status
                if item.quality == "OK" and item.moisture in ["Wet", "Dry"]:
                    status = f"OK_{item.moisture}"
                elif item.quality == "Damaged" and item.moisture in ["Wet", "Dry"]:
                    status = f"Damaged_{item.moisture}"
                else:
                    status = item.quality

                # Save using FileManager
                result = self.file_manager.save_reviewed_compartment(
                    image=item.image,  # This will lazy-load if needed
                    hole_id=item.hole_id,
                    compartment_depth=item.compartment_depth,
                    status=status,
                    output_format=self.app.config.get('output_format', 'png')
                )

                # Update stats
                if result.get('local_path'):
                    self.stats['saved_locally'] += 1
                    self.stats['processed'] += 1

                if result.get('upload_success'):
                    self.stats['uploaded'] += 1
                elif result.get('local_path') and not result.get('upload_success'):
                    self.stats['upload_failed'] += 1

            except Exception as e:
                self.logger.error(f"Error saving compartment {item.hole_id}_{item.compartment_depth}: {e}")

    def _batch_update_register(self, items: List[ReviewItem]):
        """Batch update register entries."""
        updates = []

        for item in items:
            # Determine photo status
            if item.quality == "OK" and item.moisture in ["Wet", "Dry"]:
                photo_status = f"OK_{item.moisture}"
            elif item.quality == "Damaged" and item.moisture in ["Wet", "Dry"]:
                photo_status = f"Damaged_{item.moisture}"
            elif item.quality == "Empty":
                photo_status = "Empty"
            else:
                photo_status = item.quality

            update = {
                'hole_id': item.hole_id,
                'depth_from': int(item.depth_from),
                'depth_to': int(item.depth_to),
                'photo_status': photo_status,
                'comments': f"QAQC reviewed on {datetime.now().strftime('%Y-%m-%d')}",
                'qaqc_by': os.getenv('USERNAME', 'Unknown'),
                'qaqc_date': datetime.now().strftime('%Y-%m-%d')
            }

            # Add average color if available
            if item.average_hex_color:
                update['average_hex_color'] = item.average_hex_color

            updates.append(update)

        if updates:
            try:
                updated = self.register_manager.batch_update_compartments(updates)
                self.stats['register_updated'] += updated
                self.stats['register_failed'] += len(updates) - updated

                if updated < len(updates):
                    self.logger.warning(f"Only {updated}/{len(updates)} register updates succeeded")
            except Exception as e:
                self.logger.error(f"Error batch updating register: {e}")
                self.stats['register_failed'] += len(updates)

    def _batch_delete_files(self, file_paths: List[str]):
        """Batch delete files."""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    self.logger.info(f"Deleted file: {file_path}")
            except Exception as e:
                self.logger.error(f"Error deleting file {file_path}: {e}")

    # 📝 Register Updates
    def _update_register(self):
        """Placeholder for register updates - handled through batch operations."""
        # This is now handled through _batch_update_register
        pass

    # Legacy methods that need updating for new approach
    def _save_approved_compartments(self):
        """Legacy method - redirects to batch processing."""
        # Convert current hole items to batch process
        if hasattr(self, 'current_hole_items'):
            self._batch_process_reviewed_items(self.current_hole_items)

    def _complete_current_tray(self):
        """Complete processing of current items."""
        if hasattr(self, 'current_hole_items'):
            self._batch_process_reviewed_items(self.current_hole_items)

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
        if self.main_gui:
            self.main_gui.direct_status_update(summary_message, status_type="info")

        # Log summary
        self.logger.info(summary_message)

        # Show dialog
        DialogHelper.show_message(
            self.root,
            self.t("QAQC Complete"),
            self.t(summary_message),
            message_type="info"
        )
    # ===================================================
    # NEW LAYOUTS AND METHODS BELOW
    # ===================================================


    def _create_review_window(self):
        """Create a window for reviewing compartments with 3-frame layout."""
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

        # Set window to appear on top
        self.review_window.lift()

        # Protocol for window close
        self.review_window.protocol("WM_DELETE_WINDOW", self._on_review_window_close)

        # Bind keyboard shortcuts
        self.review_window.bind('<Left>', lambda e: self._on_previous())
        self.review_window.bind('<Right>', lambda e: self._on_next())
        self.review_window.bind('1', lambda e: self._set_current_status('OK'))
        self.review_window.bind('2', lambda e: self._set_current_status('Blurry'))
        self.review_window.bind('3', lambda e: self._set_current_status('Damaged'))
        self.review_window.bind('4', lambda e: self._set_current_status('Missing'))
        self.review_window.bind('w', lambda e: self._quick_set_moisture('Wet'))
        self.review_window.bind('d', lambda e: self._quick_set_moisture('Dry'))

        # Configure window grid
        self.review_window.grid_rowconfigure(0, weight=1)
        self.review_window.grid_columnconfigure(0, weight=1)

        # Main container using grid
        main_frame = ttk.Frame(self.review_window, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Configure main frame grid - 3 rows
        main_frame.grid_rowconfigure(0, weight=0)  # Title
        main_frame.grid_rowconfigure(1, weight=1)  # Images
        main_frame.grid_rowconfigure(2, weight=0)  # Controls
        main_frame.grid_columnconfigure(0, weight=1)

        # 1. Title and progress section
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        # Title label
        title_label = ttk.Label(
            title_frame,
            text=DialogHelper.t(f"QAQC Review - {hole_id}"),
            font=("Arial", 18, "bold")
        )
        title_label.pack()

        # Progress label
        self.progress_label = ttk.Label(
            title_frame,
            text="",
            font=("Arial", 14)
        )
        self.progress_label.pack()

        # 2. Main content area for 3 image frames
        self.images_frame = ttk.Frame(main_frame)
        self.images_frame.grid(row=1, column=0, sticky="nsew", pady=10)
        
        # Configure for 3 equal columns
        self.images_frame.grid_columnconfigure(0, weight=1, uniform="frame")
        self.images_frame.grid_columnconfigure(1, weight=1, uniform="frame")
        self.images_frame.grid_columnconfigure(2, weight=1, uniform="frame")
        self.images_frame.grid_rowconfigure(0, weight=1)

        # Create the three image frames
        self.frames = {
            'wet': self._create_image_frame(self.images_frame, "Wet", 0),
            'current': self._create_image_frame(self.images_frame, "Current Image", 1),
            'dry': self._create_image_frame(self.images_frame, "Dry", 2)
        }

        # 3. Bottom controls container
        controls_container = ttk.Frame(main_frame)
        controls_container.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        
        # Create sub-frames in controls container
        controls_container.grid_columnconfigure(0, weight=1)
        controls_container.grid_columnconfigure(1, weight=1)
        controls_container.grid_columnconfigure(2, weight=1)
        
        # Left section - Keyboard shortcuts
        left_section = ttk.Frame(controls_container)
        left_section.grid(row=0, column=0, sticky="w", padx=(0, 10))
        
        shortcuts_frame = ttk.LabelFrame(left_section, text=DialogHelper.t("Keyboard Shortcuts"), padding=5)
        shortcuts_frame.pack(side=tk.TOP, fill=tk.X)
        
        shortcuts_text = ttk.Label(
            shortcuts_frame,
            text=DialogHelper.t("1: OK | 2: Blurry | 3: Damaged | 4: Missing\nW: Set Wet | D: Set Dry | ← → Navigate"),
            font=("Arial", 9),
            justify=tk.LEFT
        )
        shortcuts_text.pack()
        
        # Middle section - Moisture and Quality controls
        middle_section = ttk.Frame(controls_container)
        middle_section.grid(row=0, column=1, sticky="ew")
        
        # Moisture frame
        moisture_frame = ttk.LabelFrame(middle_section, text=DialogHelper.t("Moisture"), padding=5)
        moisture_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))
        
        self.current_moisture_var = tk.StringVar(value="Dry")
        self.moisture_toggle = self.app.gui_manager.create_modern_button(
            moisture_frame,
            text=DialogHelper.t("Dry"),
            color="#4a8259",
            command=self._toggle_moisture
        )
        self.moisture_toggle.pack(fill=tk.X, padx=5)
        
        # Quality frame
        quality_frame = ttk.LabelFrame(middle_section, text=DialogHelper.t("Quality Status"), padding=5)
        quality_frame.pack(side=tk.TOP, fill=tk.X)
        
        button_grid = ttk.Frame(quality_frame)
        button_grid.pack(fill=tk.BOTH, expand=True)
        
        # Configure 2x2 grid
        for i in range(2):
            button_grid.grid_columnconfigure(i, weight=1)
            button_grid.grid_rowconfigure(i, weight=1)
        
        # Create status buttons
        self.status_buttons = {}
        
        self.status_buttons['ok'] = self.app.gui_manager.create_modern_button(
            button_grid,
            text=DialogHelper.t("OK (1)"),
            color="#4a8259",
            command=lambda: self._set_current_status("OK")
        )
        self.status_buttons['ok'].grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        
        self.status_buttons['blurry'] = self.app.gui_manager.create_modern_button(
            button_grid,
            text=DialogHelper.t("Blurry (2)"),
            color="#9e4a4a",
            command=lambda: self._set_current_status("Blurry")
        )
        self.status_buttons['blurry'].grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        
        self.status_buttons['damaged'] = self.app.gui_manager.create_modern_button(
            button_grid,
            text=DialogHelper.t("Damaged (3)"),
            color="#d68c23",
            command=lambda: self._set_current_status("Damaged")
        )
        self.status_buttons['damaged'].grid(row=1, column=0, padx=2, pady=2, sticky="ew")
        
        self.status_buttons['missing'] = self.app.gui_manager.create_modern_button(
            button_grid,
            text=DialogHelper.t("Missing (4)"),
            color="#333333",
            command=lambda: self._set_current_status("Missing")
        )
        self.status_buttons['missing'].grid(row=1, column=1, padx=2, pady=2, sticky="ew")
        
        # Right section - Register data
        right_section = ttk.Frame(controls_container)
        right_section.grid(row=0, column=2, sticky="e", padx=(10, 0))
        
        self.summary_frame = ttk.LabelFrame(right_section, text=DialogHelper.t("Register Data"), padding=5)
        self.summary_frame.pack(side=tk.TOP)
        
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
        self.summary_text.pack()
        
        # Navigation buttons row
        nav_section = ttk.Frame(controls_container)
        nav_section.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        
        # Left navigation
        nav_left = ttk.Frame(nav_section)
        nav_left.pack(side=tk.LEFT)
        
        self.prev_button = self.app.gui_manager.create_modern_button(
            nav_left,
            text=DialogHelper.t("← Previous"),
            color="#3a7ca5",
            command=self._on_previous
        )
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        self.next_button = self.app.gui_manager.create_modern_button(
            nav_left,
            text=DialogHelper.t("Next →"),
            color="#3a7ca5",
            command=self._on_next
        )
        self.next_button.pack(side=tk.LEFT, padx=5)
        
        # Right actions
        nav_right = ttk.Frame(nav_section)
        nav_right.pack(side=tk.RIGHT)
        
        # self.save_continue_btn = self.app.gui_manager.create_modern_button(
        #     nav_right,
        #     text=DialogHelper.t("Save & Continue"),
        #     color="#4a8259",
        #     command=self._save_and_continue
        # )
        # self.save_continue_btn.pack(side=tk.LEFT, padx=5)
        
        cancel_btn = self.app.gui_manager.create_modern_button(
            nav_right,
            text=DialogHelper.t("Cancel"),
            color="#9e4a4a",
            command=self._on_review_window_close
        )
        cancel_btn.pack(side=tk.LEFT, padx=5)
        
        # Initialize compartment state tracking
        self.compartment_states = {}
        
        # Size and position window
        self._size_and_center_window()

    def _create_image_frame(self, parent, title, column):
        """Create a frame for displaying an image with proper scaling and centering."""
        frame_data = {
            'current_image': None,
            'image_path': None
        }
        
        # Main frame - reduce padding to 2
        main_frame = ttk.LabelFrame(parent, text=DialogHelper.t(title), padding=2)
        main_frame.grid(row=0, column=column, sticky="nsew", padx=2)  # Reduced padx
        frame_data['main'] = main_frame
        
        # Configure frame grid
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Canvas for image - no border
        canvas = tk.Canvas(
            main_frame,
            bg=self.app.gui_manager.theme_colors["field_bg"],
            highlightthickness=0,  # Remove border
            borderwidth=0,  # Remove border
            width=300,
            height=400
        )
        canvas.grid(row=0, column=0, sticky="nsew")
        
        # Image label
        image_label = ttk.Label(canvas, anchor="center")
        canvas_window = canvas.create_window(
            0, 0,  # Will be updated when image loads
            window=image_label,
            anchor="center"
        )
        
        frame_data['canvas'] = canvas
        frame_data['image_label'] = image_label
        frame_data['canvas_window'] = canvas_window
        
        # Depth info label
        depth_label = ttk.Label(
            main_frame,
            text="",
            font=("Arial", 10, "bold")
        )
        depth_label.grid(row=1, column=0, pady=(2, 0))  # Reduced padding
        frame_data['depth_label'] = depth_label
        
        # Status label for empty frames
        status_label = ttk.Label(
            canvas,
            text=DialogHelper.t(f"No {title.lower()} image"),
            font=("Arial", 10),
            foreground="#999999"
        )
        status_window = canvas.create_window(
            150, 200,
            window=status_label,
            anchor="center"
        )
        frame_data['status_label'] = status_label
        frame_data['status_window'] = status_window
        
        # Initially show status label
        canvas.itemconfig(status_window, state='normal')
        canvas.itemconfig(canvas_window, state='hidden')
        
        return frame_data
    
    def _toggle_moisture(self):
        """Toggle moisture setting for current image."""
        current = self.current_moisture_var.get()
        new_state = "Wet" if current == "Dry" else "Dry"
        
        self.current_moisture_var.set(new_state)
        self.moisture_toggle.update_button(
            text=DialogHelper.t(new_state),
            color="#3a7ca5" if new_state == "Wet" else "#4a8259"
        )

    def _quick_set_moisture(self, moisture):
        """Quickly set moisture type via keyboard."""
        if self.current_moisture_var.get() != moisture:
            self._toggle_moisture()

    # Remove the old _toggle_frame_moisture method and replace with above _toggle_moisture
    # Also remove _create_control_buttons as it's now integrated into _create_review_window

    def update_image_in_frame(self, frame, image_path):
        """Update the image in a frame with proper scaling and centering."""
        if not image_path or not os.path.exists(image_path):
            # Clear the image if no path or file doesn't exist
            frame.image_label.configure(image="")
            frame.image_label.image = None
            return
        
        try:
            # Open and get image dimensions
            img = Image.open(image_path)
            
            # Get the available space in the frame
            frame.update_idletasks()
            available_width = frame.winfo_width() - 20  # Subtract padding
            available_height = frame.winfo_height() - 50  # Subtract padding and title
            
            # Calculate scaling to fit while maintaining aspect ratio
            img_width, img_height = img.size
            scale_x = available_width / img_width
            scale_y = available_height / img_height
            scale = min(scale_x, scale_y, 1.0)  # Don't upscale beyond original size
            
            # Resize image
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Update label
            frame.image_label.configure(image=photo)
            frame.image_label.image = photo  # Keep a reference
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            frame.image_label.configure(image="")
            frame.image_label.image = None

    def _create_control_buttons(self, parent_frame):
        """Create control buttons for the current image frame."""
        # Moisture selection frame
        moisture_frame = ttk.LabelFrame(parent_frame, text=DialogHelper.t("Moisture"), padding=5)
        moisture_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        
        self.current_moisture_var = tk.StringVar(value="Dry")
        
        # Moisture toggle button
        self.moisture_toggle = self.app.gui_manager.create_modern_button(
            moisture_frame,
            text=DialogHelper.t("Dry"),
            color="#4a8259",
            command=lambda: self._toggle_frame_moisture('current')  # Change this line
        )
        self.moisture_toggle.grid(row=0, column=0, sticky="ew", padx=5, pady=2)
        moisture_frame.grid_columnconfigure(0, weight=1)
        
        # Quality status buttons frame
        quality_frame = ttk.LabelFrame(parent_frame, text=DialogHelper.t("Quality Status"), padding=5)
        quality_frame.grid(row=1, column=0, sticky="ew", pady=(0, 5))
        
        button_container = ttk.Frame(quality_frame)
        button_container.grid(row=0, column=0, sticky="ew")
        quality_frame.grid_columnconfigure(0, weight=1)
        
        # Configure grid for 2x2 button layout
        for i in range(2):
            button_container.grid_columnconfigure(i, weight=1)
        
        # Create status buttons
        self.status_buttons = {}
        
        # OK button
        self.status_buttons['ok'] = self.app.gui_manager.create_modern_button(
            button_container,
            text=DialogHelper.t("OK (1)"),
            color="#4a8259",
            command=lambda: self._set_current_status("OK")
        )
        self.status_buttons['ok'].grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        
        # Blurry button
        self.status_buttons['blurry'] = self.app.gui_manager.create_modern_button(
            button_container,
            text=DialogHelper.t("Blurry (2)"),
            color="#9e4a4a",
            command=lambda: self._set_current_status("Blurry")
        )
        self.status_buttons['blurry'].grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        
        # Damaged button
        self.status_buttons['damaged'] = self.app.gui_manager.create_modern_button(
            button_container,
            text=DialogHelper.t("Damaged (3)"),
            color="#d68c23",
            command=lambda: self._set_current_status("Damaged")
        )
        self.status_buttons['damaged'].grid(row=1, column=0, padx=2, pady=2, sticky="ew")
        
        # Missing button
        self.status_buttons['missing'] = self.app.gui_manager.create_modern_button(
            button_container,
            text=DialogHelper.t("Missing (4)"),
            color="#333333",
            command=lambda: self._set_current_status("Missing")
        )
        self.status_buttons['missing'].grid(row=1, column=1, padx=2, pady=2, sticky="ew")

    def _set_current_status(self, status):
        """Set status for the current image."""
        if not hasattr(self, 'compartment_states'):
            return
            
        state = self.compartment_states.get(self.current_compartment_index, {})
        
        if status == 'OK':
            moisture = self.current_moisture_var.get()
            
            # Update the compartment status
            self.current_tray['compartment_statuses'][self.current_compartment_index] = moisture
            
            # Mark as reviewed
            if self.current_compartment_index < len(self.current_hole_items):
                item = self.current_hole_items[self.current_compartment_index]
                item.moisture = moisture
                item.quality = "OK"
                item.is_reviewed = True
        
        elif status in ['Blurry', 'Damaged', 'Missing']:
            # Update the compartment status
            self.current_tray['compartment_statuses'][self.current_compartment_index] = status
            
            # Mark as reviewed
            if self.current_compartment_index < len(self.current_hole_items):
                item = self.current_hole_items[self.current_compartment_index]
                item.quality = status
                item.is_reviewed = True

    def _on_review_window_close(self):
        """Handle review window close event with proper file management."""
        try:
            # First, update all reviewed items
            reviewed_count = 0
            unreviewed_items = []
            
            for idx, item in enumerate(self.current_hole_items):
                if idx in self.current_tray['compartment_statuses']:
                    status = self.current_tray['compartment_statuses'][idx]
                    self._update_review_item_from_status(item, status)
                    reviewed_count += 1
                else:
                    # This item was not reviewed
                    unreviewed_items.append(item)
            
            # Check if there are unreviewed items
            if unreviewed_items:
                # Ask user if they want to move unreviewed files to shared folder
                message = self.t(
                    f"There are {len(unreviewed_items)} unreviewed compartments.\n\n"
                    "Would you like to move them to the shared review folder for later processing?\n\n"
                    "Yes - Move to shared folder for team review\n"
                    "No - Keep in local folder (may be lost if not backed up)"
                )
                
                move_to_shared = DialogHelper.ask_yes_no(
                    self.review_window,
                    self.t("Unreviewed Compartments"),
                    message,
                    yes_text=self.t("Move to Shared"),
                    no_text=self.t("Keep Local")
                )
                
                if move_to_shared:
                    self._move_unreviewed_items_to_shared(unreviewed_items)
            
            # Also check for any other files in temp_review directory for this hole
            self._check_and_move_remaining_temp_files()
            
            # Process any reviewed items
            if reviewed_count > 0:
                # Only process the reviewed items
                reviewed_items = [
                    item for idx, item in enumerate(self.current_hole_items)
                    if idx in self.current_tray['compartment_statuses']
                ]
                self._batch_process_reviewed_items(reviewed_items)
            
            # Show summary if we processed anything
            if reviewed_count > 0 or unreviewed_items:
                summary_message = []
                if reviewed_count > 0:
                    summary_message.append(self.t(f"✓ Reviewed and saved: {reviewed_count} compartments"))
                if unreviewed_items and move_to_shared:
                    summary_message.append(self.t(f"↗ Moved to shared review: {len(unreviewed_items)} compartments"))
                elif unreviewed_items:
                    summary_message.append(self.t(f"⚠ Left in local folder: {len(unreviewed_items)} compartments"))
                
                DialogHelper.show_message(
                    self.review_window,
                    self.t("Review Session Complete"),
                    "\n".join(summary_message),
                    message_type="info"
                )
        
        finally:
            # Always close window
            if self.review_window:
                self.review_window.destroy()
                self.review_window = None

    def _move_unreviewed_items_to_shared(self, unreviewed_items: List[ReviewItem]):
        """Move unreviewed items to shared review folder."""
        moved_count = 0
        failed_count = 0
        
        # Get shared review path
        shared_review_path = self.file_manager.get_shared_path('review_compartments', create_if_missing=True)
        if not shared_review_path:
            DialogHelper.show_message(
                self.review_window,
                self.t("Error"),
                self.t("Shared review folder is not configured. Files will remain in local folder."),
                message_type="error"
            )
            return
        
        # Progress dialog
        progress_dialog = None
        if len(unreviewed_items) > 3:
            from gui.dialog_helper import ProgressDialog
            progress_dialog = ProgressDialog(
                self.review_window,
                self.t("Moving Files"),
                self.t("Moving unreviewed files to shared folder..."),
                maximum=len(unreviewed_items)
            )
        
        try:
            for idx, item in enumerate(unreviewed_items):
                if progress_dialog:
                    progress_dialog.update(
                        idx + 1,
                        self.t(f"Moving {item.filename}...")
                    )
                
                try:
                    # Build destination path
                    project_code = item.hole_id[:2].upper()
                    shared_hole_path = shared_review_path / project_code / item.hole_id
                    shared_hole_path.mkdir(parents=True, exist_ok=True)
                    
                    # Create destination filename with _review suffix if not already present
                    base_name = f"{item.hole_id}_CC_{item.compartment_depth:03d}"
                    if "_review" not in item.filename:
                        dest_filename = f"{base_name}_review.png"
                    else:
                        dest_filename = item.filename
                    
                    dest_path = shared_hole_path / dest_filename
                    
                    # Copy file to shared location
                    if self.file_manager.copy_with_metadata(item.image_path, str(dest_path)):
                        # Verify the copy
                        if dest_path.exists():
                            # Delete local file after successful copy
                            try:
                                os.remove(item.image_path)
                                moved_count += 1
                                self.logger.info(f"Moved unreviewed file to shared: {dest_filename}")
                            except Exception as e:
                                self.logger.warning(f"Could not delete local file after copy: {e}")
                                moved_count += 1  # Still count as success
                        else:
                            failed_count += 1
                            self.logger.error(f"Copy verification failed for {item.filename}")
                    else:
                        failed_count += 1
                        self.logger.error(f"Failed to copy {item.filename} to shared folder")
                        
                except Exception as e:
                    failed_count += 1
                    self.logger.error(f"Error moving {item.filename}: {e}")
        
        finally:
            if progress_dialog:
                progress_dialog.close()
        
        # Report results
        if failed_count > 0:
            DialogHelper.show_message(
                self.review_window,
                self.t("Partial Success"),
                self.t(f"Moved {moved_count} files successfully.\n{failed_count} files failed to move."),
                message_type="warning"
            )
        
        self.logger.info(f"Moved {moved_count}/{len(unreviewed_items)} unreviewed files to shared folder")

    def _check_and_move_remaining_temp_files(self):
        """Check for any remaining files in temp_review folder and offer to move them."""
        try:
            # Get local temp_review path for current hole
            hole_id = self.current_hole_items[0].hole_id if self.current_hole_items else None
            if not hole_id:
                return
            
            temp_review_path = Path(self.file_manager.get_hole_dir("temp_review", hole_id))
            
            if not temp_review_path.exists():
                return
            
            # Find any remaining compartment files
            remaining_files = []
            for file in temp_review_path.iterdir():
                if file.is_file() and self.file_manager.COMPARTMENT_FILE_PATTERN.match(file.name):
                    # Check if this file is already in our current items
                    is_current = any(
                        Path(item.image_path).name == file.name 
                        for item in self.current_hole_items
                    )
                    if not is_current:
                        remaining_files.append(file)
            
            if remaining_files:
                message = self.t(
                    f"Found {len(remaining_files)} additional compartment files in the temp review folder.\n\n"
                    "These may be from a previous session. Move them to the shared review folder?"
                )
                
                if DialogHelper.ask_yes_no(
                    self.review_window,
                    self.t("Additional Files Found"),
                    message
                ):
                    # Move these files too
                    shared_review_path = self.file_manager.get_shared_path('review_compartments', create_if_missing=True)
                    if shared_review_path:
                        project_code = hole_id[:2].upper()
                        shared_hole_path = shared_review_path / project_code / hole_id
                        shared_hole_path.mkdir(parents=True, exist_ok=True)
                        
                        for file in remaining_files:
                            try:
                                dest_path = shared_hole_path / file.name
                                if self.file_manager.copy_with_metadata(str(file), str(dest_path)):
                                    file.unlink()  # Delete after successful copy
                                    self.logger.info(f"Moved additional file: {file.name}")
                            except Exception as e:
                                self.logger.error(f"Failed to move additional file {file.name}: {e}")
            
            # Try to clean up empty directories
            try:
                if not any(temp_review_path.iterdir()):  # Directory is empty
                    temp_review_path.rmdir()
                    
                    # Try to remove project directory if empty
                    project_path = temp_review_path.parent
                    if not any(project_path.iterdir()):
                        project_path.rmdir()
            except Exception as e:
                self.logger.debug(f"Could not remove empty directories: {e}")
                
        except Exception as e:
            self.logger.error(f"Error checking for remaining temp files: {e}")

    def _display_image_in_frame(self, frame_data, image_path=None, image_array=None, depth_text=""):
        """Display an image in a frame with proper scaling and centering."""
        try:
            # Clear any existing image
            frame_data['image_label'].configure(image="")
            frame_data['image_label'].image = None
            
            # Get canvas dimensions
            canvas = frame_data['canvas']
            canvas.update_idletasks()
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            
            # Use minimum dimensions if canvas not yet sized
            if canvas_width <= 1:
                canvas_width = 300
            if canvas_height <= 1:
                canvas_height = 400
            
            # Load image
            if image_array is not None:
                # Use numpy array directly
                img = image_array
            elif image_path and os.path.exists(image_path):
                # Load from file
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError(f"Could not load image from {image_path}")
            else:
                # No image - show status label
                canvas.itemconfig(frame_data['status_window'], state='normal')
                canvas.itemconfig(frame_data['canvas_window'], state='hidden')
                frame_data['depth_label'].configure(text="")
                return
            
            # Convert color if needed
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Calculate scaling to fit while maintaining aspect ratio
            img_height, img_width = img.shape[:2]
            
            # Use full canvas space - minimal padding
            max_width = canvas_width - 4  # Just 2px padding each side
            max_height = canvas_height - 4
            
            # Calculate scale factor
            scale_x = max_width / img_width
            scale_y = max_height / img_height
            scale = min(scale_x, scale_y)  # Don't upscale beyond 1.5x
            
            # Resize image
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Convert to PhotoImage
            pil_img = Image.fromarray(img)
            photo = ImageTk.PhotoImage(pil_img)
            
            # Update image label
            frame_data['image_label'].configure(image=photo)
            frame_data['image_label'].image = photo  # Keep reference
            
            # Center the image in canvas
            center_x = canvas_width // 2
            center_y = canvas_height // 2
            canvas.coords(frame_data['canvas_window'], center_x, center_y)
            
            # Show image, hide status
            canvas.itemconfig(frame_data['canvas_window'], state='normal')
            canvas.itemconfig(frame_data['status_window'], state='hidden')
            
            # Update depth label
            frame_data['depth_label'].configure(text=depth_text)
            
        except Exception as e:
            self.logger.error(f"Error displaying image: {e}")
            # Show error in status label
            frame_data['status_label'].configure(
                text=DialogHelper.t("Error loading image"),
                foreground="#cc0000"
            )
            canvas.itemconfig(frame_data['status_window'], state='normal')
            canvas.itemconfig(frame_data['canvas_window'], state='hidden')
            
    def _show_current_compartment_new(self):
        """Display current compartment using ReviewItem structure."""
        if self.current_compartment_index >= len(self.current_hole_items):
            return

        # Get current ReviewItem
        current_item = self.current_hole_items[self.current_compartment_index]

        # Load the compartment image into the tray structure for GUI compatibility
        if 'compartments' not in self.current_tray:
            self.current_tray['compartments'] = []

        # Ensure we have enough slots
        while len(self.current_tray['compartments']) <= self.current_compartment_index:
            self.current_tray['compartments'].append(None)

        # Load image if not already loaded
        if self.current_tray['compartments'][self.current_compartment_index] is None:
            self.current_tray['compartments'][self.current_compartment_index] = current_item.image

        # Update item status from GUI selections
        if self.current_compartment_index in self.current_tray['compartment_statuses']:
            status = self.current_tray['compartment_statuses'][self.current_compartment_index]

            # Update ReviewItem based on GUI status
            if status in ["Wet", "Dry"]:
                current_item.moisture = status
                current_item.quality = "OK"
            elif status == "Empty":
                current_item.quality = "Empty"
            elif status == "Damaged":
                current_item.quality = "Damaged"

            current_item.is_reviewed = True

        # Call the existing show method
        self._show_current_compartment()
    
    def _show_current_compartment(self):
        """Display compartment in 3-frame layout."""
        # Initialize state for this compartment if not exists
        if self.current_compartment_index not in self.compartment_states:
            self.compartment_states[self.current_compartment_index] = {
                'wet': None,
                'dry': None,
                'current': None,
                'actions': {}  # Track what to do with each image
            }

        state = self.compartment_states[self.current_compartment_index]

        # Calculate depth
        depth_from = self.current_tray['depth_from']
        interval = self.app.config['compartment_interval']
        comp_depth_from = depth_from + (self.current_compartment_index * interval)
        comp_depth_to = comp_depth_from + interval
        compartment_depth = int(comp_depth_to)

        # Get all existing images
        existing_images = self._find_all_existing_compartments(
            self.current_tray['hole_id'],
            compartment_depth
        )

        # Initialize state with existing images
        if not state['wet'] and existing_images.get('wet'):
            state['wet'] = existing_images['wet'][0]
        if not state['dry'] and existing_images.get('dry'):
            state['dry'] = existing_images['dry'][0]

        # Always have current image
        state['current'] = self.current_tray['compartments'][self.current_compartment_index]

        # Update progress
        self.progress_label.configure(
            text=f"Compartment {self.current_compartment_index + 1}/{len(self.current_tray['compartments'])}"
                f" ({compartment_depth}m)"
        )

        # Display images in frames
        self._display_state_images(state, compartment_depth)
    
    def _update_frame_layout(self, state):
        """Frame layout is static in 3-frame design - nothing to update."""
        # All 3 frames are always visible in the new layout
        # This method is kept for compatibility but doesn't need to do anything
        return

    def _size_and_center_window(self):
        """Size and center the review window using DialogHelper."""
        # Get screen dimensions for max constraints
        screen_width = self.review_window.winfo_screenwidth()
        screen_height = self.review_window.winfo_screenheight()

        # Use 90% of screen as maximum constraints
        max_width = int(screen_width * 0.9)
        max_height = int(screen_height * 0.9)

        # Use DialogHelper to center the window
        # size_ratio=1.2 gives a bit more space than the natural content size
        DialogHelper.center_dialog(
            dialog=self.review_window,
            parent=self.root,
            size_ratio=1.2,  # 120% of natural size for comfortable spacing
            max_width=max_width,
            max_height=max_height
        )

    def _set_frame_status(self, frame_type, status):
        """Set status for a frame and handle image movement."""
        state = self.compartment_states[self.current_compartment_index]

        if frame_type == 'current':
            if status == 'REMOVE':
                state['actions']['current'] = 'skip'
                self._check_and_advance()
            elif status == 'OK':
                # Use the main moisture variable, not frame-specific
                moisture = self.current_moisture_var.get()

                # Check if this will replace existing
                if moisture == 'Wet' and state['wet']:
                    state['actions'][state['wet']] = 'replace'
                    state['wet'] = 'current'  # Mark current as new wet
                elif moisture == 'Dry' and state['dry']:
                    state['actions'][state['dry']] = 'replace'
                    state['dry'] = 'current'  # Mark current as new dry
                else:
                    # No conflict
                    if moisture == 'Wet':
                        state['wet'] = 'current'
                    else:
                        state['dry'] = 'current'

                state['actions']['current'] = f'save_as_{moisture.lower()}'
                self._check_and_advance()
            else:
                # Blurry, Damaged, Missing
                state['actions']['current'] = f'save_{status.lower()}'
                self._check_and_advance()

        elif frame_type == 'unclassified':
            current_unclassified = state['unclassified_queue'][state['unclassified_index']]

            if status == 'REMOVE':
                state['actions'][current_unclassified] = 'remove'
                self._advance_unclassified_queue()
            elif status == 'OK':
                # Use the main moisture variable
                moisture = self.current_moisture_var.get()

                # Move to appropriate frame
                if moisture == 'Wet':
                    if state['wet'] and state['wet'] != 'current':
                        state['actions'][state['wet']] = 'replace'
                    state['wet'] = current_unclassified
                    state['actions'][current_unclassified] = f'classify_as_wet'
                else:
                    if state['dry'] and state['dry'] != 'current':
                        state['actions'][state['dry']] = 'replace'
                    state['dry'] = current_unclassified
                    state['actions'][current_unclassified] = f'classify_as_dry'

                self._advance_unclassified_queue()
            else:
                # Blurry, Damaged, Missing
                state['actions'][current_unclassified] = f'classify_{status.lower()}'
                self._advance_unclassified_queue()
    
    def _advance_unclassified_queue(self):
        """Move to next unclassified image or complete if done."""
        state = self.compartment_states[self.current_compartment_index]
        state['unclassified_index'] += 1

        if state['unclassified_index'] >= len(state['unclassified_queue']):
            # All unclassified processed
            self._check_and_advance()
        else:
            # Show next unclassified
            self._update_frame_layout(state)
            self._display_state_images(state, self._get_current_depth())

    def _check_and_advance(self):
        """Check if all images are processed and advance if ready."""
        state = self.compartment_states[self.current_compartment_index]

        # Check if current is processed
        if 'current' not in state['actions']:
            return

        # Check if all unclassified are processed
        for i, img_path in enumerate(state['unclassified_queue']):
            if img_path not in state['actions']:
                return

        # All processed - save state and move to next
        self._save_compartment_state()
        self._on_next()

    def _on_previous(self):
        """Show the previous compartment and restore its state."""
        if self.current_compartment_index > 0:
            # Save current status to ReviewItem before moving
            if self.current_compartment_index < len(self.current_hole_items):
                current_item = self.current_hole_items[self.current_compartment_index]
                current_status = self.current_tray['compartment_statuses'].get(self.current_compartment_index)
                if current_status:
                    self._update_review_item_from_status(current_item, current_status)
            
            # Move to previous
            self.current_compartment_index -= 1
            
            # Restore the previous item's state
            prev_status = self.current_tray['compartment_statuses'].get(self.current_compartment_index)
            if prev_status and prev_status in ["Wet", "Dry"]:
                # Restore moisture setting
                self.current_moisture_var.set(prev_status)
                self._update_moisture_button()
            elif prev_status in ["Blurry", "Damaged", "Missing", "Empty"]:
                # Quality was set to something other than OK
                # Could highlight the corresponding button here if needed
                pass
            
            # Show the compartment
            self._show_current_compartment_new()

    def _on_next(self):
        """Move to next compartment without setting status."""
        # Get current item
        if self.current_compartment_index < len(self.current_hole_items):
            current_item = self.current_hole_items[self.current_compartment_index]

            # Check if current has status
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

            # Update ReviewItem from status
            self._update_review_item_from_status(current_item, current_status)

        if self.current_compartment_index < len(self.current_hole_items) - 1:
            self.current_compartment_index += 1
            self._show_current_compartment_new()
        else:
            # All items reviewed - close window
            self._on_review_complete()

    def _update_review_item_from_status(self, item: ReviewItem, status: str):
        """Update ReviewItem based on GUI status selection."""
        if status in ["Wet", "Dry"]:
            item.moisture = status
            item.quality = "OK"
        elif status == "Empty":
            item.quality = "Empty"
        elif status == "Damaged":
            item.quality = "Damaged"
        elif status == "KEEP_ORIGINAL":
            item.quality = "OK"  # Keep existing TODO - CHECK THIS if we've chosen to keep original then this image should be flagged for removal.

        item.is_reviewed = True

    def _on_review_complete(self):
        """Handle completion of review for current hole."""
        # Mark all reviewed items
        for idx, item in enumerate(self.current_hole_items):
            if idx in self.current_tray['compartment_statuses']:
                status = self.current_tray['compartment_statuses'][idx]
                self._update_review_item_from_status(item, status)

        # Close window
        if self.review_window:
            self.review_window.destroy()
            self.review_window = None

    def _display_state_images(self, state, compartment_depth):
        """Display images in the 3 frames based on current state."""
        # Display wet image or empty state
        if state['wet']:
            if state['wet'] == 'current':
                # Current image will be shown as wet
                self._display_image_in_frame(
                    self.frames['wet'],
                    image_array=state['current'],
                    depth_text=f"Wet - {compartment_depth}m"
                )
            else:
                # Load and display existing wet image
                self._display_image_in_frame(
                    self.frames['wet'],
                    image_path=state['wet'],
                    depth_text=f"Wet - {compartment_depth}m"
                )
        else:
            # Show empty wet frame
            self._display_image_in_frame(
                self.frames['wet'],
                image_path=None,
                depth_text=""
            )

        # Display current image
        if state['current'] is not None and state['actions'].get('current') != 'skip':
            self._display_image_in_frame(
                self.frames['current'],
                image_array=state['current'],
                depth_text=f"{compartment_depth}m"
            )
        else:
            # Show empty current frame
            self._display_image_in_frame(
                self.frames['current'],
                image_path=None,
                depth_text=""
            )

        # Display dry image or empty state
        if state['dry']:
            if state['dry'] == 'current':
                # Current image will be shown as dry
                self._display_image_in_frame(
                    self.frames['dry'],
                    image_array=state['current'],
                    depth_text=f"Dry - {compartment_depth}m"
                )
            else:
                # Load and display existing dry image
                self._display_image_in_frame(
                    self.frames['dry'],
                    image_path=state['dry'],
                    depth_text=f"Dry - {compartment_depth}m"
                )
        else:
            # Show empty dry frame
            self._display_image_in_frame(
                self.frames['dry'],
                image_path=None,
                depth_text=""
            )

        # Update visual indicators
        self._update_frame_indicators(state)

    def _find_all_existing_compartments(self, hole_id: str, compartment_depth: int) -> Dict[str, List[str]]:
        """
        Find all existing compartment images organized by moisture classification.
        ONLY for the specified hole_id.

        Returns:
            Dict with keys 'wet', 'dry', 'unknown' containing lists of file paths
        """
        existing = {'wet': [], 'dry': [], 'unknown': []}
        project_code = hole_id[:2].upper() if len(hole_id) >= 2 else ""

        # Define search locations - ONLY for this specific hole
        search_locations = []

        # Shared folders - specific to this hole
        for folder_key in ['approved_compartments', 'review_compartments']:
            path = self.file_manager.get_shared_path(folder_key, create_if_missing=False)
            if path:
                # Add the specific hole path, not just project path
                hole_specific_path = os.path.join(path, project_code, hole_id)
                search_locations.append(hole_specific_path)

        # Local folders - specific to this hole
        for folder_type in ['approved_compartments', 'temp_review']:
            hole_specific_path = self.file_manager.get_hole_dir(folder_type, hole_id)
            search_locations.append(hole_specific_path)

        # Search all locations
        for location in search_locations:
            if os.path.exists(location):
                try:
                    # The location is already hole-specific, so files here should be for this hole
                    for file in os.listdir(location):
                        # Double-check the file is for this hole AND depth
                        if file.startswith(f"{hole_id}_CC_{compartment_depth:03d}"):
                            full_path = os.path.join(location, file)

                            # Skip directories
                            if os.path.isdir(full_path):
                                continue

                            # Classify by moisture type based on filename
                            if '_Wet' in file or '_wet' in file:
                                existing['wet'].append(full_path)
                            elif '_Dry' in file or '_dry' in file:
                                existing['dry'].append(full_path)
                            else:
                                # No clear classification - could be _temp, _new, or no suffix
                                existing['unknown'].append(full_path)
                except Exception as e:
                    self.logger.warning(f"Error searching location {location}: {e}")

        # Log findings - now should only be for this specific hole
        self.logger.info(f"Found compartments for {hole_id} at {compartment_depth}m: "
                        f"Wet: {len(existing['wet'])}, Dry: {len(existing['dry'])}, "
                        f"Unknown: {len(existing['unknown'])}")

        return existing

    def _display_compartment_image(self, frame_data, image_data, depth_text, is_numpy=True):
        """Display a compartment image in a frame."""
        try:
            # Clear existing image
            frame_data['image_label'].configure(image="")

            if image_data is None:
                frame_data['image_label'].configure(
                    text=DialogHelper.t("No image available"),
                    foreground="red"
                )
                return

            # Process image
            if is_numpy:
                # It's a numpy array from the compartments list
                display_img = image_data
            else:
                # It's a file path - load it
                display_img = cv2.imread(image_data)
                if display_img is None:
                    frame_data['image_label'].configure(
                        text=DialogHelper.t("Error loading image"),
                        foreground="red"
                    )
                    return

            # Convert color if needed
            if len(display_img.shape) == 2:
                display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)
            else:
                display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

            # Calculate resize to fit frame
            h, w = display_img.shape[:2]
            max_height = 400
            max_width = 300

            scale_h = max_height / h
            scale_w = max_width / w
            scale = min(scale_h, scale_w)

            if scale < 1:
                new_width = int(w * scale)
                new_height = int(h * scale)
                display_img = cv2.resize(display_img, (new_width, new_height),
                                    interpolation=cv2.INTER_AREA)

            # Convert to PhotoImage
            from PIL import Image, ImageTk
            pil_img = Image.fromarray(display_img)
            tk_img = ImageTk.PhotoImage(image=pil_img)

            # Update label
            frame_data['image_label'].configure(image=tk_img, text="")
            frame_data['image_label'].image = tk_img  # Keep reference

            # Update depth label
            frame_data['depth_label'].configure(text=depth_text)

            # Update canvas scroll region
            frame_data['canvas'].configure(scrollregion=frame_data['canvas'].bbox("all"))

        except Exception as e:
            self.logger.error(f"Error displaying image: {e}")
            frame_data['image_label'].configure(
                text=DialogHelper.t("Error displaying image"),
                foreground="red"
            )

    def _display_existing_image(self, frame_data, image_path, depth_text):
        """Display an existing image from file path."""
        self._display_compartment_image(frame_data, image_path, depth_text, is_numpy=False)

    def _update_frame_indicators(self, state):
        """Update visual indicators on frames based on actions."""
        # Reset all frame styles first
        for frame_key, frame_data in self.frames.items():
            frame_data['main'].configure(relief=tk.RAISED)

            # Reset opacity if it was dimmed
            if 'image_label' in frame_data and frame_data['image_label'].winfo_exists():
                frame_data['image_label'].configure(style="")

        # Apply indicators based on actions
        for item, action in state['actions'].items():
            if action == 'replace':
                # Find which frame contains this item and dim it
                if item == state.get('wet'):
                    self.frames['wet']['main'].configure(relief=tk.SUNKEN)
                    self._dim_frame(self.frames['wet'])
                elif item == state.get('dry'):
                    self.frames['dry']['main'].configure(relief=tk.SUNKEN)
                    self._dim_frame(self.frames['dry'])
            elif action == 'skip' and item == 'current':
                self._dim_frame(self.frames['current'])

    def _dim_frame(self, frame_data):
        """Apply dimming effect to a frame to show it will be replaced/removed."""
        frame_data['main'].configure(relief=tk.SUNKEN)
        # Could also reduce opacity or add overlay here if desired

    def _get_current_depth(self):
        """Get the current compartment depth."""
        depth_from = self.current_tray['depth_from']
        interval = self.app.config['compartment_interval']
        comp_depth_from = depth_from + (self.current_compartment_index * interval)
        comp_depth_to = comp_depth_from + interval
        return int(comp_depth_to)

    def _keyboard_shortcut(self, frame_type, action):
        """Handle keyboard shortcuts for status buttons."""
        # Keyboard shortcuts only work on current image
        if frame_type == 'current':
            if action == 'ok':
                self._set_current_status("OK")
            elif action == 'blurry':
                self._set_current_status("Blurry")
            elif action == 'damaged':
                self._set_current_status("Damaged")
            elif action == 'missing':
                self._set_current_status("Missing")
                
    # def _save_and_continue(self):
    #     """Save current state and continue to next tray."""
    #     # Force save current compartment state
    #     self._save_compartment_state()

    #     # Move to next tray
    #     self._save_approved_compartments()
    #     self._review_next_tray()

    def _save_compartment_state(self):
        """Convert the current compartment state to the format expected by save methods."""
        state = self.compartment_states.get(self.current_compartment_index)
        if not state:
            return

        # Build the legacy format status
        final_status = {}

        # Handle current image
        if 'current' in state['actions']:
            action = state['actions']['current']
            if action == 'skip':
                final_status = "SKIP"
            elif action.startswith('save_as_'):
                # Extract moisture (wet/dry)
                moisture = action.replace('save_as_', '').title()
                final_status = moisture
            elif action.startswith('save_'):
                # Extract status (blurry/damaged/missing)
                status = action.replace('save_', '').title()
                final_status = status

        # Store in the format expected by save methods
        self.current_tray['compartment_statuses'][self.current_compartment_index] = final_status

