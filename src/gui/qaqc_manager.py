"""
Consolidated QAQC (Quality Assurance/Quality Control) system for compartment image review.
Combines all QAQC functionality into a single organized module.
"""

import os
import re
import shutil
import threading
import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Set
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk


from gui.dialog_helper import DialogHelper
from gui.progress_dialog import ProgressDialog


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
    
    VALID_QAQC_STATUSES = [
        "OK_Wet", "OK_Dry", "Damaged_Wet", "Damaged_Dry", "Empty"
    ]
    
    NEEDS_REVIEW_STATUSES = [
        "Found", None, "", "Not Set"
    ]
    
    SKIP_STATUSES = [
        "MISSING_FILE", "Not Found", "MISSING"
    ]


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
        self.HOLE_ID_PATTERN = re.compile(r'^[A-Z]{2}\d{4}$')
        self.PROJECT_CODE_PATTERN = re.compile(r'^[A-Z]{2}$')
        self.COMPARTMENT_FILE_PATTERN = re.compile(
            r'([A-Z]{2}\d{4})_CC_(\d{3})(?:_temp|_new|_review|_Wet|_Dry)?\.(?:png|tiff|jpg)$',
            re.IGNORECASE
        )
        
        self.compartment_register = {}
    
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
        
        self.compartment_register = {hole_id: group for hole_id, group in df.groupby('HoleID')}
        return self.compartment_register
    
    # In qaqc_manager.py, QAQCScanner class
    def scan_local_review_folder(self, temp_review_path: str) -> Dict[str, List[ReviewItem]]:
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
                    
                    # Create review item
                    review_item = ReviewItem(
                        filename=file,
                        hole_id=hole_id,
                        depth_from=depth - self.compartment_interval,
                        depth_to=depth,
                        compartment_depth=depth,
                        image_path=full_path,
                        register_status=self.get_register_status(hole_id, depth),
                        # ===================================================
                        # INSERT: Set moisture from filename
                        moisture=moisture,
                        quality="OK" if moisture != "unknown" else "unreviewed"
                        # ===================================================
                    )
                    
                    # Add to hole's list
                    if hole_id not in review_items_by_hole:
                        review_items_by_hole[hole_id] = []
                    review_items_by_hole[hole_id].append(review_item)
        
        return review_items_by_hole
    
    def scan_shared_folders_for_review(self) -> Dict[str, List[str]]:
        """Scan shared folders for compartments needing review."""
        items_by_hole = {}
        
        # Check review folder
        review_path = self.file_manager.get_shared_path('review_compartments', create_if_missing=False)
        if review_path and os.path.exists(review_path):
            self._scan_shared_folder(review_path, items_by_hole, "review")
        
        # Check approved folder
        approved_path = self.file_manager.get_shared_path('approved_compartments', create_if_missing=False)
        if approved_path and os.path.exists(approved_path):
            self._scan_shared_folder(approved_path, items_by_hole, "approved")
        
        return items_by_hole
    
    def _scan_shared_folder(self, base_path: str, items_by_hole: Dict[str, List[str]], folder_type: str):
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
                        
                        if folder_type == "review" or register_status in QAQCConstants.NEEDS_REVIEW_STATUSES:
                            full_path = os.path.join(hole_path, file)
                            if hole_id not in items_by_hole:
                                items_by_hole[hole_id] = []
                            items_by_hole[hole_id].append(full_path)
    
    def get_register_status(self, hole_id: str, depth_to: int) -> Optional[str]:
        """Get the current Photo_Status from register for a compartment."""
        if hole_id in self.compartment_register:
            df = self.compartment_register[hole_id]
            matching = df[df['To'] == depth_to]
            if not matching.empty:
                return matching.iloc[0].get('Photo_Status')
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
                    register_status=self.get_register_status(hole_id, depth)
                )
                
                review_items.append(review_item)
        
        return review_items


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
            r'([A-Z]{2}\d{4})_CC_(\d{3})(?:_temp|_new|_review|_Wet|_Dry)?\.(?:png|tiff|jpg)$',
            re.IGNORECASE
        )
        
        self.stats = {
            'processed': 0,
            'uploaded': 0,
            'upload_failed': 0,
            'saved_locally': 0,
            'register_updated': 0,
            'register_failed': 0
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
                # ===================================================
                # INSERT: Check if file already has the correct suffix
                current_suffix = None
                if "_Wet" in item.filename:
                    current_suffix = "Wet"
                elif "_Dry" in item.filename:
                    current_suffix = "Dry"
                
                # Skip if already has correct suffix
                if current_suffix == item.moisture:
                    self.logger.info(f"File already has correct suffix: {item.filename}")
                    self.stats['processed'] += 1
                    continue
                # ===================================================
                
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
                    output_format=self.config.get('output_format', 'png')
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
            
            if item.average_hex_color:
                update['average_hex_color'] = item.average_hex_color
            
            updates.append(update)
        
        if updates:
            try:
                updated = self.register_manager.batch_update_compartments(updates)
                self.stats['register_updated'] += updated
                self.stats['register_failed'] += len(updates) - updated
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
    
    def move_files_to_local(self, hole_id: str, file_paths: List[str]) -> List[str]:
        """Move files from shared location to local temp_review folder."""
        local_paths = []
        
        temp_review_path = self.file_manager.get_local_path('temp_review')
        project_code = hole_id[:2]
        local_hole_path = os.path.join(temp_review_path, project_code, hole_id)
        
        os.makedirs(local_hole_path, exist_ok=True)
        
        for src_path in file_paths:
            try:
                filename = os.path.basename(src_path)
                dst_path = os.path.join(local_hole_path, filename)
                
                shutil.copy2(src_path, dst_path)
                local_paths.append(dst_path)
                
                try:
                    os.remove(src_path)
                except Exception as e:
                    self.logger.warning(f"Could not remove source file {src_path}: {e}")
                    
            except Exception as e:
                self.logger.error(f"Error moving file {src_path}: {e}")
        
        return local_paths
    
    def move_unreviewed_items_to_shared(self, unreviewed_items: List[ReviewItem], 
                                       progress_callback=None) -> tuple[int, int]:
        """Move unreviewed items to shared review folder."""
        moved_count = 0
        failed_count = 0
        
        shared_review_path = self.file_manager.get_shared_path('review_compartments', create_if_missing=True)
        if not shared_review_path:
            self.logger.error("Shared review folder is not configured")
            return 0, len(unreviewed_items)
        
        for idx, item in enumerate(unreviewed_items):
            if progress_callback:
                progress_callback(idx + 1, f"Moving {item.filename}...")
            
            try:
                project_code = item.hole_id[:2].upper()
                shared_hole_path = Path(shared_review_path) / project_code / item.hole_id
                shared_hole_path.mkdir(parents=True, exist_ok=True)
                
                base_name = f"{item.hole_id}_CC_{item.compartment_depth:03d}"
                if "_review" not in item.filename:
                    dest_filename = f"{base_name}_review.png"
                else:
                    dest_filename = item.filename
                
                dest_path = shared_hole_path / dest_filename
                
                if self.file_manager.copy_with_metadata(item.image_path, str(dest_path)):
                    if dest_path.exists():
                        try:
                            os.remove(item.image_path)
                            moved_count += 1
                            self.logger.info(f"Moved unreviewed file to shared: {dest_filename}")
                        except Exception as e:
                            self.logger.warning(f"Could not delete local file after copy: {e}")
                            moved_count += 1
                    else:
                        failed_count += 1
                        self.logger.error(f"Copy verification failed for {item.filename}")
                else:
                    failed_count += 1
                    self.logger.error(f"Failed to copy {item.filename} to shared folder")
                    
            except Exception as e:
                failed_count += 1
                self.logger.error(f"Error moving {item.filename}: {e}")
        
        return moved_count, failed_count
    
    def check_and_move_remaining_temp_files(self, hole_id: str) -> int:
        """Check for any remaining files in temp_review folder and move them."""
        try:
            temp_review_path = Path(self.file_manager.get_hole_dir("temp_review", hole_id))
            
            if not temp_review_path.exists():
                return 0
            
            remaining_files = []
            for file in temp_review_path.iterdir():
                if file.is_file() and self.COMPARTMENT_FILE_PATTERN.match(file.name):
                    remaining_files.append(file)
            
            if not remaining_files:
                return 0
            
            shared_review_path = self.file_manager.get_shared_path('review_compartments', create_if_missing=True)
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
                    self.logger.error(f"Failed to move additional file {file.name}: {e}")
            
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
            f"- Register update failures: {self.stats['register_failed']}"
        ]
        return "\n".join(summary_lines)


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
                "row_neutral": "#1e1e1e"
            }
        
        # Create main container
        self.container = ttk.Frame(parent)
        self.container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create canvas with scrollbars
        self.canvas = tk.Canvas(
            self.container,
            bg=self.theme_colors["background"],
            highlightthickness=0
        )
        self.v_scrollbar = ttk.Scrollbar(
            self.container,
            orient="vertical",
            command=self.canvas.yview
        )
        self.h_scrollbar = ttk.Scrollbar(
            self.container,
            orient="horizontal",
            command=self.canvas.xview
        )
        
        self.canvas.configure(
            yscrollcommand=self.v_scrollbar.set,
            xscrollcommand=self.h_scrollbar.set
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
        
        self.layout_mode = "grid"  # "grid", "vertical", or "duplicate_resolution"

        # Add new attributes for layout and rotation
        self.image_rotation = 0  # Current rotation angle
        self.duplicates_resolved = False  # Track if duplicates have been resolved
        self.has_duplicates = False  # Track if there are duplicates
        self.duplicate_depths = {}  # Track which depths have duplicates

        # Selection handling
        self.drag_start = None
        self.selection_box = None
        
        # Bind events
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<Button-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_release)
        self.canvas.bind("<Control-MouseWheel>", self._on_ctrl_mousewheel)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
    


    def load_items(self, review_items: List[ReviewItem], hole_id: str):
        """Load review items with duplicate-first workflow."""
        self.hole_id = hole_id
        self.review_items = sorted(review_items, key=lambda x: x.depth_to)
        

        # Calculate optimal cell size before loading
        self._calculate_optimal_cell_size()
    
        
        # Initialize states (preserve existing if reloading)
        if not hasattr(self, 'item_states') or len(self.item_states) != len(self.review_items):
            self.item_states = {}
            for i, item in enumerate(self.review_items):
                # Pre-load state from item's moisture status
                moisture = None
                if item.moisture in ["Wet", "Dry"]:
                    moisture = item.moisture.lower()
                
                self.item_states[i] = {
                    "moisture": moisture,
                    "delete": False,
                    "bad_image": False
                }
        
        # Clear existing display only
        self.canvas.delete("all")
        self.cells.clear()
        self.cell_ids.clear()
        self.image_refs.clear()
        self.depth_to_row.clear()
        self.row_backgrounds.clear()
        self.selected_cells.clear()
        
        # Group items by depth to find duplicates
        self.depth_groups = {}
        for idx, item in enumerate(self.review_items):
            depth = item.depth_to
            if depth not in self.depth_groups:
                self.depth_groups[depth] = []
            self.depth_groups[depth].append(idx)
        
        # Check if we have duplicates
        self.has_duplicates = any(len(indices) > 1 for indices in self.depth_groups.values())
        self.duplicate_depths = {depth: indices for depth, indices in self.depth_groups.items() if len(indices) > 1}
        
        # Determine which layout to use
        if self.has_duplicates and not getattr(self, 'duplicates_resolved', False):
            # Force duplicate resolution first
            self._load_duplicate_resolution_layout()
        elif self.layout_mode == "vertical":
            # Vertical layout with duplicates side-by-side
            self._load_vertical_layout_with_duplicates()
        else:
            # Standard grid layout
            self.cols_per_row = max(3, (self.parent.winfo_width() - 100) // (self.cell_width + self.padding))
            self._load_grid_layout()
        
        # Update scroll region
        self._update_scroll_region()
        
        # Update status table if available
        if hasattr(self.parent.master, 'update_status_table'):
            self.parent.master.update_status_table()
    

    def _calculate_optimal_cell_size(self):
        """Calculate optimal cell size based on all loaded images."""
        
        if not hasattr(self, 'image_dimensions'):
            self.image_dimensions = {}
            
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(__name__)
        # ===================================================
        
        if not self.review_items:
            return
        
        # method to calculate dynamic cell size
        if not self.review_items:
            return
        
        # Sample a subset of images to get dimensions
        sample_size = min(20, len(self.review_items))  # Sample up to 20 images
        sample_indices = range(0, len(self.review_items), max(1, len(self.review_items) // sample_size))
        
        max_width = 0
        max_height = 0
        aspect_ratios = []
        
        for idx in sample_indices:
            item = self.review_items[idx]
            if item.image_path and os.path.exists(item.image_path):
                try:
                    # Read image to get dimensions without loading full data
                    img = cv2.imread(item.image_path)
                    if img is not None:
                        h, w = img.shape[:2]
                        self.image_dimensions[idx] = (w, h)
                        
                        # Consider rotation
                        if self.image_rotation in [90, 270]:
                            w, h = h, w  # Swap for rotation
                        
                        max_width = max(max_width, w)
                        max_height = max(max_height, h)
                        aspect_ratios.append(w / h if h > 0 else 1)
                except Exception as e:
                    self.logger.debug(f"Could not read image dimensions: {e}")
        
        if not aspect_ratios:
            return
        
        # Calculate median aspect ratio
        median_aspect = sorted(aspect_ratios)[len(aspect_ratios) // 2]
        
        # Get available canvas width
        canvas_width = self.parent.winfo_width()
        if canvas_width <= 1:  # Not yet rendered
            canvas_width = 1200  # Default assumption
        
        # Calculate cell width based on target columns
        available_width = canvas_width - (self.target_cols + 1) * self.padding
        target_cell_width = available_width // self.target_cols
        
        # Calculate corresponding height from aspect ratio
        target_cell_height = int(target_cell_width / median_aspect)
        
        # Apply reasonable limits
        min_width, max_width = 150, 600
        min_height, max_height = 100, 400
        
        self.base_cell_width = max(min_width, min(max_width, target_cell_width))
        self.base_cell_height = max(min_height, min(max_height, target_cell_height))
        
        # Update current dimensions
        self.cell_width = int(self.base_cell_width * self.scale_factor)
        self.cell_height = int(self.base_cell_height * self.scale_factor)
        
        # Recalculate columns that fit
        self.cols_per_row = max(1, (canvas_width - 2 * self.padding) // (self.cell_width + self.padding))
        
        self.logger.info(f"Calculated optimal cell size: {self.cell_width}x{self.cell_height} "
                        f"(aspect ratio: {median_aspect:.2f}, cols: {self.cols_per_row})")
        # ===================================================

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
                errors.append(f"Depth {depth}m: {unclassified_count} image(s) not classified")
            
            # Allow at most 1 wet and 1 dry
            if wet_count > 1:
                errors.append(f"Depth {depth}m: Multiple wet images ({wet_count})")
            if dry_count > 1:
                errors.append(f"Depth {depth}m: Multiple dry images ({dry_count})")
            
            # At least one image should be kept (wet, dry, or both)
            if wet_count == 0 and dry_count == 0:
                errors.append(f"Depth {depth}m: No images kept (all marked for deletion)")
        
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
            row_width = self.padding + (len(indices) * (self.cell_width + self.padding)) + 60  # 60 for depth label
            bg_rect = self.canvas.create_rectangle(
                0, row * (self.cell_height + self.padding),
                row_width, (row + 1) * (self.cell_height + self.padding),
                fill=self.theme_colors["row_neutral"],
                outline="",
                tags=f"row_bg_{depth}"
            )
            self.row_backgrounds[depth] = bg_rect
            
            # Add depth label
            self.canvas.create_text(
                30, row * (self.cell_height + self.padding) + self.cell_height // 2,
                text=f"{depth}m",
                fill=self.theme_colors["text"],
                font=("Arial", 14, "bold"),
                anchor="center",
                tags=f"depth_label_{depth}"
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
            # Create row background
            bg_rect = self.canvas.create_rectangle(
                0, row * (self.cell_height + self.padding),
                self.padding + (len(indices) * (self.cell_width + self.padding)),
                (row + 1) * (self.cell_height + self.padding),
                fill=self.theme_colors["accent_error"],  # Red background for duplicates
                outline="",
                tags=f"row_bg_{depth}"
            )
            self.row_backgrounds[depth] = bg_rect
            
            # Add depth label
            self.canvas.create_text(
                10, row * (self.cell_height + self.padding) + self.cell_height // 2,
                text=f"{depth}m",
                fill="white",
                font=("Arial", 16, "bold"),
                anchor="w",
                tags=f"depth_label_{depth}"
            )
            
            # Add duplicate images side by side
            for col, idx in enumerate(indices):
                item = self.review_items[idx]
                self._create_grid_cell(row, col, idx, item)
            
            row += 1

    def _create_grid_cell_with_offset(self, row: int, col: int, idx: int, item: ReviewItem, x_offset: int = 0):
        """Create a grid cell with custom x offset."""
        x = x_offset + col * (self.cell_width + self.padding) + self.padding
        y = row * (self.cell_height + self.padding) + self.padding
        
        # Create background if needed
        bg_rect = self.canvas.create_rectangle(
            x, y,
            x + self.cell_width, y + self.cell_height,
            fill="#404040",
            outline="",
            tags=f"bg_{row}_{col}"
        )
        
        # Rest is same as _create_grid_cell but using the idx parameter
        cell_data = {
            'idx': idx,
            'item': item,
            'selected': False
        }
        
        # Load and display image
        img, photo = self._prepare_image(item)
        if photo:
            self.image_refs[(row, col)] = photo
            
            img_id = self.canvas.create_image(
                x + self.cell_width // 2,
                y + self.cell_height // 2,
                image=photo,
                tags=f"img_{row}_{col}"
            )
        else:
            img_id = self.canvas.create_text(
                x + self.cell_width // 2,
                y + self.cell_height // 2,
                text="No Image",
                fill="white",
                font=("Arial", 12)
            )
        
        # Create border
        border_id = self.canvas.create_rectangle(
            x, y,
            x + self.cell_width, y + self.cell_height,
            outline=self.theme_colors["border"],
            width=3,
            tags=f"border_{row}_{col}"
        )
        
        # Add depth label in corner
        self.canvas.create_text(
            x + 5, y + self.cell_height - 5,
            text=f"{item.compartment_depth}m",
            fill="white",
            font=("Arial", 10, "bold"),
            anchor="sw",
            tags=f"depth_label_{row}_{col}"
        )
        
        # Store cell data
        self.cells[(row, col)] = cell_data
        self.cell_ids[(row, col)] = {
            'bg': bg_rect,
            'image': img_id,
            'border': border_id,
            'x': x,
            'y': y
        }
        
        # Apply visual state
        self._update_cell_visual(row, col)

    def _load_grid_layout(self):
        """Load items in standard grid layout."""
        # Filter out items based on current state
        items_to_show = []
        for idx, item in enumerate(self.review_items):
            state = self.item_states[idx]
            
            # Skip if already classified and we're past duplicate resolution
            if self.duplicates_resolved and state["moisture"]:
                continue
                
            items_to_show.append((idx, item))
        
        # Create grid
        for i, (idx, item) in enumerate(items_to_show):
            row = i // self.cols_per_row
            col = i % self.cols_per_row
            self._create_grid_cell(row, col, idx, item)

    def _create_grid_cell(self, row: int, col: int, idx: int, item: ReviewItem):
        """Create a single grid cell."""
        x = col * (self.cell_width + self.padding) + self.padding
        y = row * (self.cell_height + self.padding) + self.padding
        
        # Create cell background
        bg_rect = self.canvas.create_rectangle(
            x, y,
            x + self.cell_width, y + self.cell_height,
            fill="#404040",
            outline="",
            tags=f"bg_{row}_{col}"
        )
        
        # Load and display image
        img, photo = self._prepare_image(item)
        if photo:
            self.image_refs[(row, col)] = photo
            
            img_id = self.canvas.create_image(
                x + self.cell_width // 2,
                y + self.cell_height // 2,
                image=photo,
                tags=f"img_{row}_{col}"
            )
        else:
            # Placeholder for missing image
            img_id = self.canvas.create_text(
                x + self.cell_width // 2,
                y + self.cell_height // 2,
                text="No Image",
                fill="white",
                font=("Arial", 12)
            )
        
        # Create border
        border_id = self.canvas.create_rectangle(
            x, y,
            x + self.cell_width, y + self.cell_height,
            outline=self.theme_colors["border"],
            width=3,
            tags=f"border_{row}_{col}"
        )
        
        # Add depth label
        self.canvas.create_text(
            x + 5, y + self.cell_height - 5,
            text=f"{item.compartment_depth}m",
            fill="white",
            font=("Arial", 10, "bold"),
            anchor="sw",
            tags=f"depth_label_{row}_{col}"
        )
        
        # Store cell data
        self.cells[(row, col)] = {
            'idx': idx,
            'item': item,
            'selected': False
        }
        
        self.cell_ids[(row, col)] = {
            'bg': bg_rect,
            'image': img_id,
            'border': border_id,
            'x': x,
            'y': y
        }
        
        # Apply initial visual state
        self._update_cell_visual(row, col)
    
    def _prepare_image(self, item: ReviewItem) -> Tuple[Optional[Image.Image], Optional[ImageTk.PhotoImage]]:
        """Prepare image for display with rotation support."""
        try:
            img = item.image
            if img is None:
                return None, None
            
            # Convert to PIL
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            pil_img = Image.fromarray(img)
            
            # ===================================================
            # INSERT: Apply rotation if set
            if hasattr(self, 'image_rotation') and self.image_rotation > 0:
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

    def _update_cell_visual(self, row: int, col: int):
        """Update visual appearance of a cell with classification text."""
        if (row, col) not in self.cells or (row, col) not in self.cell_ids:
            return
        
        cell = self.cells[(row, col)]
        ids = self.cell_ids[(row, col)]
        idx = cell['idx']
        state = self.item_states[idx]
        
        # Determine border appearance
        selected = cell.get('selected', False)
        
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
        self.canvas.itemconfig(
            ids['border'],
            outline=border_color,
            width=border_width
        )
        
        # ===================================================
        # INSERT: Add classification text overlay
        # Remove old text
        self.canvas.delete(f"class_text_{row}_{col}")
        
        # Build classification text
        text_parts = []
        if state["moisture"]:
            text_parts.append(state["moisture"].upper())
        if state["delete"]:
            text_parts.append("DELETE")
        if state["bad_image"]:
            text_parts.append("BAD")
        
        if text_parts:
            text = " ".join(text_parts)
            x = ids['x'] + self.cell_width - 10
            y = ids['y'] + 10
            
            # Create text with background
            text_id = self.canvas.create_text(
                x, y,
                text=text,
                fill="white",
                font=("Arial", 10, "bold"),
                anchor="ne",
                tags=f"class_text_{row}_{col}"
            )
            
            # Add semi-transparent background
            bbox = self.canvas.bbox(text_id)
            if bbox:
                bg_id = self.canvas.create_rectangle(
                    bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2,
                    fill="#000000",
                    stipple="gray50",
                    outline="",
                    tags=f"class_text_{row}_{col}"
                )
                self.canvas.tag_raise(text_id)
        # ===================================================
    
    def _on_mouse_down(self, event):
        """Handle mouse down for selection/action."""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        clicked_cell = self._get_cell_at_position(canvas_x, canvas_y)
        
        if clicked_cell:
            row, col = clicked_cell
            
            if self.current_mode == "select":
                if event.state & 0x0004:  # Ctrl held
                    self._toggle_cell_selection(row, col)
                elif event.state & 0x0001:  # Shift held
                    self._select_range_to(row, col)
                else:
                    self._clear_selection()
                    self._select_cell(row, col)
                    self.drag_start = (canvas_x, canvas_y)
            else:
                # Apply action based on mode
                self._apply_action_to_cell(row, col)
        else:
            if self.current_mode == "select":
                self._clear_selection()
                self.drag_start = (canvas_x, canvas_y)
    
    def _on_mouse_drag(self, event):
        """Handle mouse drag for box selection."""
        if self.current_mode != "select" or not self.drag_start:
            return
        
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        if self.selection_box:
            self.canvas.delete(self.selection_box)
        
        self.selection_box = self.canvas.create_rectangle(
            self.drag_start[0], self.drag_start[1],
            canvas_x, canvas_y,
            outline=self.theme_colors["accent_yellow"],
            width=2,
            dash=(5, 5),
            tags="selection_box"
        )
        
        self._select_cells_in_box(self.drag_start[0], self.drag_start[1], canvas_x, canvas_y)
    
    def _on_mouse_release(self, event):
        """Handle mouse release."""
        if self.selection_box:
            self.canvas.delete(self.selection_box)
            self.selection_box = None
        self.drag_start = None
    
    def _get_cell_at_position(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """Get cell coordinates at canvas position."""
        for (row, col), ids in self.cell_ids.items():
            cell_x = ids['x']
            cell_y = ids['y']
            
            if (cell_x <= x <= cell_x + self.cell_width and 
                cell_y <= y <= cell_y + self.cell_height):
                return (row, col)
        
        return None
    
    def _select_cell(self, row: int, col: int):
        """Select a single cell."""
        if (row, col) in self.cells:
            self.selected_cells.add((row, col))
            self.cells[(row, col)]['selected'] = True
            self._update_cell_visual(row, col)
    
    def _toggle_cell_selection(self, row: int, col: int):
        """Toggle selection state of a cell."""
        if (row, col) in self.selected_cells:
            self.selected_cells.remove((row, col))
            self.cells[(row, col)]['selected'] = False
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
            cell_x = ids['x']
            cell_y = ids['y']
            
            if (cell_x < max_x and cell_x + self.cell_width > min_x and
                cell_y < max_y and cell_y + self.cell_height > min_y):
                self._select_cell(row, col)
    
    def _clear_selection(self):
        """Clear all selected cells."""
        for (row, col) in self.selected_cells:
            self.cells[(row, col)]['selected'] = False
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
            
            depth_groups[depth].append({
                'idx': idx,
                'state': state,
                'item': item
            })
        
        # Check each depth
        for depth, items in depth_groups.items():
            # Check if all items are classified
            unclassified = [i for i in items if not i['state']['moisture'] and not i['state']['delete']]
            if unclassified:
                errors.append(f"Depth {depth}m has {len(unclassified)} unclassified image(s)")
            
            # Check for multiple wet/dry of same type
            wet_items = [i for i in items if i['state']['moisture'] == 'wet']
            dry_items = [i for i in items if i['state']['moisture'] == 'dry']
            
            if len(wet_items) > 1:
                errors.append(f"Depth {depth}m has {len(wet_items)} wet images (max 1 allowed)")
            if len(dry_items) > 1:
                errors.append(f"Depth {depth}m has {len(dry_items)} dry images (max 1 allowed)")
        
        return len(errors) == 0, errors

    def _apply_action_to_cell(self, row: int, col: int):
        """Apply current mode action to a cell with toggle behavior."""
        if (row, col) not in self.cells:
            return
        
        cell = self.cells[(row, col)]
        idx = cell['idx']
        state = self.item_states[idx]
        
        # ===================================================
        # REPLACE: Handle click behavior based on mode
        if self.current_mode == "wet":
            # Toggle wet classification
            if state["moisture"] == "wet":
                state["moisture"] = None
            else:
                state["moisture"] = "wet"
                state["delete"] = False
        elif self.current_mode == "dry":
            # Toggle dry classification
            if state["moisture"] == "dry":
                state["moisture"] = None
            else:
                state["moisture"] = "dry"
                state["delete"] = False
        elif self.current_mode == "delete":
            # Toggle delete
            state["delete"] = not state["delete"]
            if state["delete"]:
                state["moisture"] = None
        elif self.current_mode == "bad_image":
            # Toggle bad image flag (doesn't affect moisture)
            state["bad_image"] = not state["bad_image"]
        # ===================================================
        
        self._update_cell_visual(row, col)
        
        # Update status table
        if hasattr(self.parent.master, 'update_status_table'):
            self.parent.master.update_status_table()


    def apply_action_to_selected(self, action: str):
        """Apply an action to all selected cells."""
        for (row, col) in self.selected_cells:
            cell = self.cells[(row, col)]
            idx = cell['idx']
            state = self.item_states[idx]
            
            if action == "wet":
                state["moisture"] = "wet"
                state["delete"] = False
            elif action == "dry":
                state["moisture"] = "dry"
                state["delete"] = False
            elif action == "delete":
                state["delete"] = True
                state["moisture"] = None
            
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
    
    def get_results(self) -> Tuple[List[ReviewItem], List[ReviewItem]]:
        """Get reviewed and unreviewed items."""
        reviewed = []
        unreviewed = []
        
        for idx, item in enumerate(self.review_items):
            state = self.item_states[idx]
            
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
            'total': len(self.review_items),
            'wet': 0,
            'dry': 0,
            'delete': 0,
            'unclassified': 0,
            'selected': len(self.selected_cells)
        }
        
        for state in self.item_states.values():
            if state["delete"]:
                counts['delete'] += 1
            elif state["moisture"] == "wet":
                counts['wet'] += 1
            elif state["moisture"] == "dry":
                counts['dry'] += 1
            else:
                counts['unclassified'] += 1
        
        return counts
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling."""
        if not (event.state & 0x0004):  # Not Ctrl
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
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
            self.load_items(self.review_items, self.hole_id) # TODO - make sure this doesn't wipe out the selections.
    
    def _update_scroll_region(self):
        """Update canvas scroll region."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def _on_canvas_configure(self, event):
        """Handle canvas resize."""
        # ===================================================
        # REPLACE: Recalculate columns on resize
        new_width = event.width
        if new_width > 100:  # Meaningful resize
            new_cols = max(1, (new_width - 2 * self.padding) // (self.cell_width + self.padding))
            if new_cols != self.cols_per_row and self.layout_mode == "grid":
                self.cols_per_row = new_cols
                # Reload grid layout
                self.canvas.after_idle(lambda: self.load_items(self.review_items, self.hole_id))
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
        self.gui_manager = app.gui_manager if hasattr(app, 'gui_manager') else None
        
        self.review_window = None
        self.close_callback = None
        self.grid_canvas = None
        self.status_label = None
        self.mode_buttons = {}
    
    def show_review_window(self, hole_id: str, review_items: List[ReviewItem]):
        """Show the grid review window."""
        self._create_window(hole_id, review_items)
    
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
        
        # Create UI components
        self._create_toolbar()

        # ===================================================
        # INSERT: Add status table between toolbar and canvas
        # Create a frame for the status table
        self.table_frame = ttk.LabelFrame(self.review_window, text="Register Update Preview", padding=5)
        self.table_frame.pack(fill=tk.X, padx=5, pady=5)

        # Create treeview for status table
        self.status_tree = ttk.Treeview(
            self.table_frame,
            columns=('depth', 'status', 'action', 'validation'),
            height=6,
            show='tree headings'
        )

        # Configure columns
        self.status_tree.heading('#0', text='#')
        self.status_tree.heading('depth', text='Depth (m)')
        self.status_tree.heading('status', text='Photo Status')
        self.status_tree.heading('action', text='Action')
        self.status_tree.heading('validation', text='Validation')

        self.status_tree.column('#0', width=40)
        self.status_tree.column('depth', width=100)
        self.status_tree.column('status', width=150)
        self.status_tree.column('action', width=100)
        self.status_tree.column('validation', width=200)

        self.status_tree.pack(fill=tk.X, expand=True)
        # ===================================================

        self._create_canvas_area(hole_id, review_items)
        self._create_status_bar()
        
        # Setup keyboard shortcuts
        self._setup_keyboard_shortcuts()
        
        self.review_window.protocol("WM_DELETE_WINDOW", self._on_close)
    

    def _create_canvas_area(self, hole_id: str, review_items: List[ReviewItem], parent=None):
        """Create the grid canvas area."""
        # ===================================================
        # REPLACE: Use provided parent or create frame
        # Create canvas frame
        canvas_frame = parent if parent else ttk.Frame(self.review_window)
        if not parent:
            canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        # ===================================================
        
        # Create grid canvas
        self.grid_canvas = QAQCGridCanvas(canvas_frame, self.gui_manager)
        self.grid_canvas.logger = self.logger  # Add logger reference
        
        # Load items
        self.grid_canvas.load_items(review_items, hole_id)
    

    def _create_status_table(self, parent):
        """Create status table on the right side."""
        # ===================================================
        # INSERT: New method for vertical status table
        # Create a frame for the status table
        self.table_frame = ttk.LabelFrame(parent, text="Register Update Preview", padding=5)
        self.table_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        # Create treeview for status table
        self.status_tree = ttk.Treeview(
            self.table_frame,
            columns=('status', 'action'),
            height=25,  # Taller for vertical layout
            show='tree headings'
        )
        
        # Configure columns for narrower width
        self.status_tree.heading('#0', text='Depth')
        self.status_tree.heading('status', text='Status')
        self.status_tree.heading('action', text='Action')
        
        self.status_tree.column('#0', width=80)
        self.status_tree.column('status', width=150)
        self.status_tree.column('action', width=80)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.table_frame, command=self.status_tree.yview)
        self.status_tree.configure(yscrollcommand=scrollbar.set)
        
        self.status_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        # ===================================================

    def _create_toolbar(self):
        """Create the toolbar with mode buttons and actions."""
        toolbar = ttk.Frame(self.review_window)
        toolbar.pack(fill=tk.X, padx=5, pady=5)
        
        # Mode section
        mode_frame = ttk.LabelFrame(toolbar, text="Classification Mode", padding=5)
        mode_frame.pack(side=tk.LEFT, padx=5)

        # ===================================================
        # Mode buttons - removed Select, add Bad Image
        modes = [
            ("Wet", "wet", self.gui_manager.theme_colors["accent_blue"] if self.gui_manager else "#1e88e5"),
            ("Dry", "dry", self.gui_manager.theme_colors["accent_green"] if self.gui_manager else "#43a047"),
            ("Delete", "delete", self.gui_manager.theme_colors["accent_red"] if self.gui_manager else "#e53935"),
            ("Bad Image", "bad_image", self.gui_manager.theme_colors["accent_yellow"] if self.gui_manager else "#fdd835")
        ]

        for text, mode, color in modes:
            if self.gui_manager:
                btn = self.gui_manager.create_modern_button(
                    mode_frame, text, color, lambda m=mode: self._set_mode(m)
                )
            else:
                btn = ttk.Button(mode_frame, text=text, command=lambda m=mode: self._set_mode(m))
            
            self.mode_buttons[mode] = btn
            btn.pack(side=tk.LEFT, padx=2)

        # Set initial mode to wet
        self.current_mode = "wet"
        # ===================================================
        
        # Duplicate resolution controls
        self.duplicate_frame = ttk.LabelFrame(toolbar, text="Duplicate Resolution", padding=5)
        # Don't pack yet - will show only when needed
        
        if self.gui_manager:
            self.apply_dup_button = self.gui_manager.create_modern_button(
                self.duplicate_frame,
                "Apply & Continue",
                self.gui_manager.theme_colors["accent_green"],
                self._apply_duplicate_resolution
            )
        else:
            self.apply_dup_button = ttk.Button(
                self.duplicate_frame,
                text="Apply & Continue",
                command=self._apply_duplicate_resolution
            )
        self.apply_dup_button.pack(side=tk.LEFT, padx=2)
        # ===================================================


        # Actions section with themed buttons
        action_frame = ttk.LabelFrame(toolbar, text="Actions", padding=5)
        action_frame.pack(side=tk.LEFT, padx=20)

        # ===================================================
        # REPLACE: Use themed buttons for actions
        if self.gui_manager:
            # Image display controls
            display_frame = ttk.Frame(action_frame)
            display_frame.pack(side=tk.LEFT, padx=5)
            
            ttk.Label(display_frame, text="Display:").pack(side=tk.LEFT, padx=2)
            
            self.gui_manager.create_modern_button(
                display_frame, "Rotate", 
                self.gui_manager.theme_colors["secondary_bg"],
                self._rotate_images
            ).pack(side=tk.LEFT, padx=2)
            
            self.gui_manager.create_modern_button(
                display_frame, "Grid Layout",
                self.gui_manager.theme_colors["secondary_bg"],
                lambda: self._set_layout("grid")
            ).pack(side=tk.LEFT, padx=2)
            
            self.gui_manager.create_modern_button(
                display_frame, "Column Layout",
                self.gui_manager.theme_colors["secondary_bg"],
                lambda: self._set_layout("column")
            ).pack(side=tk.LEFT, padx=2)
            
            # Size controls
            size_frame = ttk.Frame(action_frame)
            size_frame.pack(side=tk.LEFT, padx=10)
            
            ttk.Label(size_frame, text="Size:").pack(side=tk.LEFT, padx=2)
            
            self.gui_manager.create_modern_button(
                size_frame, "Smaller",
                self.gui_manager.theme_colors["secondary_bg"],
                lambda: self._scale_images(0.8)
            ).pack(side=tk.LEFT, padx=2)
            
            self.gui_manager.create_modern_button(
                size_frame, "Bigger",
                self.gui_manager.theme_colors["secondary_bg"],
                lambda: self._scale_images(1.2)
            ).pack(side=tk.LEFT, padx=2)
            
            # Reset button
            self.gui_manager.create_modern_button(
                action_frame, "Reset All",
                self.gui_manager.theme_colors["accent_red"],
                self._reset_all
            ).pack(side=tk.LEFT, padx=10)
        else:
            # Fallback non-themed buttons
            ttk.Button(action_frame, text="Rotate", command=self._rotate_images).pack(side=tk.LEFT, padx=2)
            ttk.Button(action_frame, text="Grid Layout", command=lambda: self._set_layout("grid")).pack(side=tk.LEFT, padx=2)
            ttk.Button(action_frame, text="Column Layout", command=lambda: self._set_layout("column")).pack(side=tk.LEFT, padx=2)
            ttk.Button(action_frame, text="Smaller", command=lambda: self._scale_images(0.8)).pack(side=tk.LEFT, padx=2)
            ttk.Button(action_frame, text="Bigger", command=lambda: self._scale_images(1.2)).pack(side=tk.LEFT, padx=2)
            ttk.Button(action_frame, text="Reset All", command=self._reset_all).pack(side=tk.LEFT, padx=2)
        # ===================================================
        # Save/Cancel on the right
        button_frame = ttk.Frame(toolbar)
        button_frame.pack(side=tk.RIGHT, padx=5)
        
        if self.gui_manager:
            save_btn = self.gui_manager.create_modern_button(
                button_frame, "Save & Close", 
                self.gui_manager.theme_colors["accent_green"], 
                self._save_and_close
            )
            cancel_btn = self.gui_manager.create_modern_button(
                button_frame, "Cancel", 
                self.gui_manager.theme_colors["accent_red"], 
                self._on_close
            )
        else:
            save_btn = ttk.Button(button_frame, text="Save & Close", command=self._save_and_close)
            cancel_btn = ttk.Button(button_frame, text="Cancel", command=self._on_close)
        
        save_btn.pack(side=tk.LEFT, padx=5)
        cancel_btn.pack(side=tk.LEFT, padx=5)
        
        # Set initial mode after UI is created
        self.review_window.after(100, lambda: self._set_mode("dry"))
    
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
                message_type="error"
            )
            return
        
        # Mark duplicates as resolved
        self.grid_canvas.duplicates_resolved = True
        
        # Hide duplicate frame, show normal actions
        self.duplicate_frame.pack_forget()
        
        # Reload in standard grid layout
        self.grid_canvas.layout_mode = "grid"
        self.grid_canvas.load_items(self.grid_canvas.review_items, self.grid_canvas.hole_id)
        
        # Update status
        self._update_status()
        self.update_status_table()
        # ===================================================

    # Update load_items to show/hide duplicate button
    def _update_duplicate_controls(self):
        """Show/hide duplicate resolution controls based on state."""
        # ===================================================
        # INSERT: New method
        if hasattr(self, 'duplicate_frame') and hasattr(self.grid_canvas, 'has_duplicates'):
            if self.grid_canvas.has_duplicates and not self.grid_canvas.duplicates_resolved:
                self.duplicate_frame.pack(side=tk.LEFT, padx=20, after=self.mode_buttons['bad_image'].master)
            else:
                self.duplicate_frame.pack_forget()
        # ===================================================

    def update_status_table(self):
        """Update the status table with current classifications."""
        # Clear existing items
        for item in self.status_tree.get_children():
            self.status_tree.delete(item)
        
        # Get current results from grid
        reviewed, unreviewed = self.grid_canvas.get_results()
        all_items = reviewed + unreviewed
        
        # Group by depth
        depth_groups = {}
        for item in all_items:
            depth = item.depth_to
            if depth not in depth_groups:
                depth_groups[depth] = []
            depth_groups[depth].append(item)
        
        # Add rows for each depth
        row_num = 1
        for depth in sorted(depth_groups.keys()):
            items = depth_groups[depth]
            
            # Determine status
            statuses = []
            for item in items:
                if item.quality == "Missing":
                    statuses.append("Delete")
                elif item.moisture:
                    status = f"OK_{item.moisture}"
                    # Check for bad image flag
                    idx = self.grid_canvas.review_items.index(item)
                    if self.grid_canvas.item_states[idx].get("bad_image"):
                        status += "_Bad"
                    statuses.append(status)
                else:
                    statuses.append("Unclassified")
            
            # Determine action
            if "Unclassified" in statuses:
                action = "Pending"
                validation = "⚠️ Needs classification"
            elif len([s for s in statuses if s.startswith("OK_")]) > 1:
                action = "Error"
                validation = "❌ Multiple files for same moisture"
            else:
                action = "Update"
                validation = "✅ Ready"
            
            # Insert row
            self.status_tree.insert(
                '', 'end',
                text=str(row_num),
                values=(
                    f"{depth}m",
                    ", ".join(statuses),
                    action,
                    validation
                )
            )
            row_num += 1

    def _create_status_bar(self):
        """Create status bar at bottom."""
        status_frame = ttk.Frame(self.review_window)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = ttk.Label(status_frame, text="", relief=tk.SUNKEN)
        self.status_label.pack(fill=tk.X, padx=5, pady=2)
        
        self._update_status()
    
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
    
    def _reset_all(self):
        """Reset all classifications."""
        response = DialogHelper.confirm_dialog(
            self.review_window,
            "Reset All",
            "This will clear all classifications. Continue?"
        )
        if response:
            # Reset all states
            for state in self.grid_canvas.item_states.values():
                state["moisture"] = None
                state["delete"] = False
            
            # Update all visuals
            for (row, col) in self.grid_canvas.cells:
                self.grid_canvas._update_cell_visual(row, col)
            
            self._update_status()
    
    def _update_status(self):
        """Update status bar."""
        counts = self.grid_canvas.get_status_counts()
        
        status = (f"Total: {counts['total']} | "
                 f"Wet: {counts['wet']} | "
                 f"Dry: {counts['dry']} | "
                 f"Delete: {counts['delete']} | "
                 f"Unclassified: {counts['unclassified']}")
        
        if counts['selected'] > 0:
            status += f" | Selected: {counts['selected']}"
        
        status += f" | Mode: {self.grid_canvas.current_mode.upper()}"
        
        self.status_label.configure(text=status)
    
    def _setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts."""
        self.review_window.bind('s', lambda e: self._set_mode("select"))
        self.review_window.bind('w', lambda e: self._set_mode("wet"))
        self.review_window.bind('d', lambda e: self._set_mode("dry"))
        self.review_window.bind('x', lambda e: self._set_mode("delete"))
        self.review_window.bind('<Control-a>', lambda e: self._select_all())
        self.review_window.bind('<Escape>', lambda e: self._clear_selection())
    
    def _select_all(self):
        """Select all items."""
        for (row, col) in self.grid_canvas.cells:
            self.grid_canvas._select_cell(row, col)
        self._update_status()
    
    def _save_and_close(self):
        """Save classifications and close."""
        if self.close_callback:
            self.close_callback(cancelled=False)
        self.review_window.destroy()
    
    def _on_close(self):
        """Handle window close."""
        response = DialogHelper.confirm_dialog(
            self.review_window,
            "Cancel Review",
            "Are you sure you want to cancel? Unsaved changes will be lost." # TODO - Don't say it will be lost just prompt them to save.
        )
        if response:
            if self.close_callback:
                self.close_callback(cancelled=True)
            self.review_window.destroy()
    
    def set_close_callback(self, callback): # TODO - check that closing will actually close!
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
        current_rotation = getattr(self.grid_canvas, 'image_rotation', 0)
        new_rotation = (current_rotation + 90) % 360
        self.grid_canvas.image_rotation = new_rotation
        
        # Swap base dimensions for 90/270 degree rotations
        if (current_rotation in [0, 180] and new_rotation in [90, 270]) or \
        (current_rotation in [90, 270] and new_rotation in [0, 180]):
            # Swap base dimensions
            self.grid_canvas.base_cell_width, self.grid_canvas.base_cell_height = \
                self.grid_canvas.base_cell_height, self.grid_canvas.base_cell_width
        
        # Apply scale factor
        self.grid_canvas.cell_width = int(self.grid_canvas.base_cell_width * self.grid_canvas.scale_factor)
        self.grid_canvas.cell_height = int(self.grid_canvas.base_cell_height * self.grid_canvas.scale_factor)
        
        # Recalculate columns
        canvas_width = self.grid_canvas.parent.winfo_width()
        self.grid_canvas.cols_per_row = max(1, (canvas_width - 2 * self.grid_canvas.padding) // 
                                            (self.grid_canvas.cell_width + self.grid_canvas.padding))
        
        # Reload with current classifications preserved
        self.grid_canvas.load_items(self.grid_canvas.review_items, self.grid_canvas.hole_id)
        self._update_status()

    def _set_layout(self, layout_type: str):
        """Change the grid layout type."""
        if layout_type == "grid":
            self.grid_canvas.cols_per_row = 5  # Default grid columns
        elif layout_type == "column":
            self.grid_canvas.layout_mode = "vertical"
            # self.grid_canvas.cols_per_row = 1  # Single column
        
        # Reload with current classifications preserved
        self.grid_canvas.load_items(self.grid_canvas.review_items, self.grid_canvas.hole_id)
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
        self.grid_canvas.cell_height = int(self.grid_canvas.base_cell_height * new_scale)
        
        # Recalculate columns
        canvas_width = self.grid_canvas.parent.winfo_width()
        self.grid_canvas.cols_per_row = max(1, (canvas_width - 2 * self.grid_canvas.padding) // 
                                            (self.grid_canvas.cell_width + self.grid_canvas.padding))
        
        # Reload with current classifications preserved
        self.grid_canvas.load_items(self.grid_canvas.review_items, self.grid_canvas.hole_id)
        self._update_status()
        # ===================================================

    def _reset_all(self):
        """Reset all classifications after confirmation."""
        response = DialogHelper.confirm_dialog(
            self.review_window,
            "Reset All Classifications",
            "This will clear all classifications and bring back all images. Continue?"
        )
        if response:
            # Reset all states
            for state in self.grid_canvas.item_states.values():
                state["moisture"] = None
                state["delete"] = False
                state["bad_image"] = False
            
            # Reset duplicate resolution flag if any
            if hasattr(self.grid_canvas, 'duplicates_resolved'):
                self.grid_canvas.duplicates_resolved = False
            
            # Reload the grid
            self.grid_canvas.load_items(self.grid_canvas.review_items, self.grid_canvas.hole_id)
            
            # Update status
            self._update_status()
            self.update_status_table()

# ============================================================================
# MAIN QAQC MANAGER
# ============================================================================

class QAQCManager:
    """Main coordinator for quality assurance and quality control."""
    
    def __init__(self, file_manager, translator_func, config_manager,
                 app, logger, register_manager, gui_manager=None):
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
            compartment_interval=app.config['compartment_interval'],
            logger=logger
        )
        
        self.processor = QAQCProcessor(
            file_manager=file_manager,
            register_manager=register_manager,
            config=app.config,
            logger=logger
        )
        
        self.dialog = QAQCGridReviewDialog(
            parent=self.root,
            app=app,
            translator_func=translator_func,
            logger=logger
        )
        
        self.main_gui = app.main_gui if hasattr(app, 'main_gui') else None
    
    def set_main_gui(self, main_gui):
        """Set reference to main GUI for status updates."""
        self.main_gui = main_gui
    
    def start_review_process(self):
        """Entry point for the QAQC review process."""
        # Reset statistics
        self.processor.reset_stats()
        
        # Step 1: Load register data
        self.scanner.load_register_into_memory()
        
        # Step 2: Build review queue from local files
        temp_review_path = self.file_manager.dir_structure["temp_review"]
        local_review_items = self.scanner.scan_local_review_folder(temp_review_path)
        
        if local_review_items:
            # Process local files by hole
            self._process_review_queue(local_review_items)
        else:
            # Step 3: Check shared folders for items needing review
            shared_review_items = self.scanner.scan_shared_folders_for_review()
            
            if shared_review_items:
                # Move one hole at a time to local and process
                self._process_shared_items(shared_review_items)
            else:
                self._show_nothing_to_review_message()
    
    def _process_review_queue(self, items_by_hole: Dict[str, List[ReviewItem]]):
        """Process review items one hole at a time."""
        total_holes = len(items_by_hole)
        self.stop_review_process = False
        
        for idx, (hole_id, review_items) in enumerate(items_by_hole.items()):
            # Check if user wants to stop
            if self.stop_review_process:
                self.logger.info("Review process stopped by user")
                break
            
            # Update status
            if self.main_gui:
                self.main_gui.direct_status_update(
                    f"Processing hole {idx + 1}/{total_holes}: {hole_id}",
                    status_type="info"
                )
            
            # Sort by depth
            review_items.sort(key=lambda x: x.depth_to)
            
            # Set callback
            self.dialog.set_close_callback(
                lambda cancelled=False: self._on_hole_review_complete(hole_id, cancelled)
            )
            
            # Show review GUI
            self.dialog.show_review_window(hole_id, review_items)
            
            # Wait for dialog to close
            self.root.wait_window(self.dialog.review_window)
    
    def _on_hole_review_complete(self, hole_id: str, cancelled: bool = False):
        """Handle completion of review for a single hole."""
        if cancelled:
            self.stop_review_process = True
            self.logger.info(f"Review cancelled for hole {hole_id}")
            return
        
        # Get reviewed and unreviewed items
        reviewed_items = self.dialog.get_reviewed_items()
        unreviewed_items = self.dialog.get_unreviewed_items()
        

        # Handle unreviewed items
        move_to_shared = False  # Track user choice
        if unreviewed_items:
            message = self.t(
                f"There are {len(unreviewed_items)} unreviewed compartments.\n\n"
                "Would you like to move them to the shared review folder for later processing?\n\n"
                "Yes - Move to shared folder for team review\n"
                "No - Keep in local folder (may be lost if not backed up)"
            )
            
            move_to_shared = DialogHelper.confirm_dialog(
                self.dialog.review_window,
                self.t("Unreviewed Compartments"),
                message,
                yes_text=self.t("Move to Shared"),
                no_text=self.t("Keep Local")
            )
            
            if move_to_shared:
                # Create progress dialog if many items
                progress_dialog = None
                if len(unreviewed_items) > 3:
                    progress_dialog = ProgressDialog(
                        self.dialog.review_window,
                        self.t("Moving Files"),
                        self.t("Moving unreviewed files to shared folder..."),
                        maximum=len(unreviewed_items)
                    )
                
                try:
                    moved, failed = self.processor.move_unreviewed_items_to_shared(
                        unreviewed_items,
                        progress_callback=progress_dialog.update if progress_dialog else None
                    )
                    
                    if failed > 0:
                        DialogHelper.show_message(
                            self.dialog.review_window,
                            self.t("Partial Success"),
                            self.t(f"Moved {moved} files successfully.\n{failed} files failed to move."),
                            message_type="warning"
                        )
                finally:
                    if progress_dialog:
                        progress_dialog.close()

        # Check for any other remaining files
        if move_to_shared:
            remaining_moved = self.processor.check_and_move_remaining_temp_files(hole_id)
            if remaining_moved > 0:
                self.logger.info(f"Moved {remaining_moved} additional files from temp folder")
        
        # ===================================================
        # Process reviewed items
        if reviewed_items:
            self.processor.batch_process_reviewed_items(reviewed_items)
        
        # Free memory
        for item in reviewed_items + unreviewed_items:
            item.unload_image()
        
        # Show summary if we processed anything
        if reviewed_items or unreviewed_items:
            self._show_hole_summary(len(reviewed_items), len(unreviewed_items), 
                                   move_to_shared if unreviewed_items else False)
    
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
            local_paths = self.processor.move_files_to_local(hole_id, file_paths)
            
            if local_paths:
                # Now scan and process as local files
                local_items = self.scanner.scan_specific_files(local_paths)
                
                # Process this hole
                self._process_review_queue({hole_id: local_items})
    
    def _show_hole_summary(self, reviewed_count: int, unreviewed_count: int, moved_to_shared: bool):
        """Show summary for a single hole."""
        summary_message = []
        if reviewed_count > 0:
            summary_message.append(self.t(f"✓ Reviewed and saved: {reviewed_count} compartments"))
        if unreviewed_count > 0 and moved_to_shared:
            summary_message.append(self.t(f"↗ Moved to shared review: {unreviewed_count} compartments"))
        elif unreviewed_count > 0:
            summary_message.append(self.t(f"⚠ Left in local folder: {unreviewed_count} compartments"))
        
        DialogHelper.show_message(
            self.root,
            self.t("Hole Review Complete"),
            "\n".join(summary_message),
            message_type="info"
        )
    
    def _show_nothing_to_review_message(self):
        """Show message when there's nothing to review."""
        DialogHelper.show_message(
            self.root,
            self.t("Review Complete"),
            self.t("No images to review and all files are synchronized."),
            message_type="info"
        )
    
    def _show_final_summary(self):
        """Show final processing summary and update main GUI."""
        summary_message = self.processor.get_summary_message()
        
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


# Export main classes
__all__ = [
    'QAQCManager',
    'ReviewItem',
    'QAQCConstants'
]