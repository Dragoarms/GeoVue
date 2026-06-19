"""
Standalone Photo Bulk Renamer with Depth Validation
A simple GUI tool for rapidly renaming geological chip tray photos with depth validation.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import csv
import re
import logging
import tempfile
from collections import Counter
from PIL import Image, ImageTk, ImageOps, ImageDraw
import numpy as np
import cv2
import os
from gui.widgets.modern_button import ModernButton
from gui.widgets.entry_with_validation import create_entry_with_validation

try:
    from gui.dialog_helper import DialogHelper
except Exception:
    DialogHelper = None


class DepthValidator:
    """Handles CSV loading and depth validation"""

    def __init__(self):
        self.depth_ranges: Dict[str, tuple] = (
            {}
        )  # {hole_id: (min_depth, max_depth_rounded)}
        self.csv_path: Optional[Path] = None

    def load_csv(self, csv_path: str) -> bool:
        """Load depth validation data from CSV and construct valid ranges"""
        try:
            import math

            self.csv_path = Path(csv_path)
            self.depth_ranges.clear()

            # Collect all samples per hole
            hole_samples = {}  # {hole_id: [(from, to), ...]}

            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    hole_id = None
                    from_depth = None
                    to_depth = None

                    # Try to find columns - exact match first, then variations
                    for key in row.keys():
                        key_normalized = key.strip().upper()

                        # Check for hole ID
                        if hole_id is None:
                            if key_normalized == "HOLEID":
                                hole_id = row[key].strip().upper()
                            elif key_normalized in [
                                "HOLE_ID",
                                "DRILLHOLE",
                                "DRILL_HOLE",
                            ]:
                                hole_id = row[key].strip().upper()

                        # Check for from depth
                        if from_depth is None:
                            if key_normalized in ["SAMPFROM", "SAMP_FROM", "FROM"]:
                                try:
                                    from_depth = float(row[key])
                                except (ValueError, TypeError):
                                    pass

                        # Check for to depth
                        if to_depth is None:
                            if key_normalized in ["SAMPTO", "SAMP_TO", "TO"]:
                                try:
                                    to_depth = float(row[key])
                                except (ValueError, TypeError):
                                    pass

                    if hole_id and from_depth is not None and to_depth is not None:
                        if hole_id not in hole_samples:
                            hole_samples[hole_id] = []
                        hole_samples[hole_id].append((from_depth, to_depth))

            # Now construct valid ranges: min to rounded-up max (20m intervals)
            for hole_id, samples in hole_samples.items():
                if samples:
                    min_depth = min(s[0] for s in samples)
                    max_depth = max(s[1] for s in samples)
                    # Round max depth up to nearest 20m
                    rounded_max = math.ceil(max_depth / 20) * 20
                    self.depth_ranges[hole_id] = (min_depth, rounded_max)

            return len(self.depth_ranges) > 0

        except Exception as e:
            messagebox.showerror("CSV Load Error", f"Failed to load CSV:\n{str(e)}")
            return False

    def load_from_dict(self, depth_ranges: Dict[str, tuple]) -> bool:
        """Load depth ranges from a pre-built dict (e.g. from Snowflake).
        
        Args:
            depth_ranges: {hole_id_upper: (min_depth, max_depth_rounded)}
            
        Returns:
            True if any data was loaded
        """
        self.depth_ranges.clear()
        self.depth_ranges.update(depth_ranges)
        self.csv_path = None
        return len(self.depth_ranges) > 0

    def validate_depth(
        self, hole_id: str, from_depth: float, to_depth: float
    ) -> tuple[bool, str]:
        """Validate if depth range is within valid range for hole ID"""
        if not self.depth_ranges:
            return True, "No validation data loaded"

        # Normalize hole ID for case-insensitive lookup
        hole_id_upper = hole_id.strip().upper()

        if hole_id_upper not in self.depth_ranges:
            return False, f"Hole ID '{hole_id}' not found in validation data"

        # Get valid range for this hole
        valid_min, valid_max = self.depth_ranges[hole_id_upper]

        # Check if the requested range is within the valid range
        if from_depth < valid_min:
            return False, f"Depth {from_depth}m is before valid start ({valid_min}m)"

        if to_depth > valid_max:
            return False, f"Depth {to_depth}m exceeds valid end ({valid_max}m)"

        return True, f"✓ Valid (range: {valid_min}-{valid_max}m)"


class ExistingCompartmentChecker:
    """Checks for existing compartment images in output folders"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.output_folders: List[Path] = []
        self.cache: Dict[str, Dict] = {}
        self.compartment_pattern = re.compile(
            r"([A-Z]{2,4}\d+)_CC_0*(\d+)(?:_(Wet|Dry))?(?:_.*)?\.(?:png|tiff?|jpe?g)$",
            re.IGNORECASE,
        )
    
    def set_output_folders(self, folders: List[str]):
        """Set folders to scan for existing compartments"""
        self.output_folders = [Path(f) for f in folders if f and Path(f).exists()]
        self.cache.clear()
    
    def check_existing(self, hole_id: str, depth_from: int, depth_to: int) -> Dict[str, Any]:
        """Check if compartments already exist for this tray."""
        cache_key = f"{hole_id}_{depth_from}-{depth_to}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        result = {"exists": False, "count": 0, "files": [], "wet_count": 0, "dry_count": 0}
        
        if not self.output_folders:
            return result
        
        hole_id_upper = hole_id.upper()
        found_files, wet_files, dry_files = [], [], []
        
        for folder in self.output_folders:
            if not folder.exists():
                continue
            try:
                for file_path in folder.iterdir():
                    if not file_path.is_file():
                        continue
                    match = self.compartment_pattern.match(file_path.name)
                    if (
                        match
                        and match.group(1).upper() == hole_id_upper
                        and int(match.group(2)) == int(depth_to)
                    ):
                        found_files.append(file_path)
                        moisture = (match.group(3) or "").lower()
                        if moisture == "wet" or "_Wet" in file_path.name:
                            wet_files.append(file_path)
                        elif moisture == "dry" or "_Dry" in file_path.name:
                            dry_files.append(file_path)
            except PermissionError:
                continue
        
        if found_files:
            result = {
                "exists": True, "count": len(found_files), "files": found_files,
                "wet_count": len(wet_files), "dry_count": len(dry_files),
                "has_wet": len(wet_files) > 0, "has_dry": len(dry_files) > 0,
            }
        
        self.cache[cache_key] = result
        return result
    
    def clear_cache(self):
        self.cache.clear()



class ProcessingPreviewCache:
    """Caches processing preview results for quick navigation"""
    
    def __init__(self, max_size: int = 50):
        self.cache: Dict[str, Dict] = {}
        self.max_size = max_size
    
    def get(self, file_path: str, mtime: float) -> Optional[Dict]:
        key = str(file_path)
        if key in self.cache and self.cache[key].get("mtime") == mtime:
            return self.cache[key]
        return None
    
    def set(self, file_path: str, mtime: float, result: Dict):
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        result["mtime"] = mtime
        self.cache[str(file_path)] = result
    
    def clear(self):
        self.cache.clear()


class PhotoRenamerGUI:
    """Main GUI application for photo renaming"""

    def __init__(
        self,
        root,
        gui_manager=None,
        initial_photo_folder: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.root = root
        self.gui_manager = gui_manager
        self.initial_photo_folder = Path(initial_photo_folder) if initial_photo_folder else None

        # Don't set title/geometry here - parent dialog handles it
        # self.root.title("Photo Bulk Renamer")
        # self.root.geometry("1200x800")

        # Get theme colors if gui_manager available
        if self.gui_manager and hasattr(self.gui_manager, "theme_colors"):
            self.theme_colors = self.gui_manager.theme_colors
            self.fonts = self.gui_manager.fonts
        else:
            # Fallback theme colors
            self.theme_colors = {
                "background": "#2b2b2b",
                "secondary_bg": "#3c3c3c",
                "text": "#ffffff",
                "field_bg": "#1e1e1e",
                "field_border": "#555555",
                "accent_green": "#5aa06c",
                "accent_red": "#ff6b6b",
                "accent_blue": "#4a90c0",
            }
            self.fonts = {
                "normal": ("Arial", 10),
                "heading": ("Arial", 12, "bold"),
                "small": ("Arial", 9),
            }

        # Data
        self.validator = DepthValidator()
        self.photo_folder: Optional[Path] = None
        self.photo_files: List[Path] = []
        self.current_index: int = 0
        self.current_image: Optional[ImageTk.PhotoImage] = None

        # Last values for increment
        self.last_hole_id: str = ""
        self.last_to: str = ""

        # Rejected photos tracking
        self.rejected_files: List[Dict[str, str]] = []  # [{original_name, reason, timestamp}]
        self.rejected_folder: Optional[Path] = None

        # Fixed interval length (always 20m)
        self.interval_length: int = 20
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Config from caller/gui_manager
        self.config = config or {}
        if not self.config and self.gui_manager and hasattr(self.gui_manager, "config"):
            self.config = self.gui_manager.config
        
        # Existing compartment checker
        self.existing_checker = ExistingCompartmentChecker()
        self._setup_output_folders()
        
        # Processing preview
        self.preview_cache = ProcessingPreviewCache()
        self.preview_enabled = tk.BooleanVar(value=True)
        self.current_preview_data: Optional[Dict] = None
        self._aruco_manager = None
        self._viz_manager = None
        self._moisture_predictor = None
        self._moisture_predictor_checked = False
        self.current_moisture: Dict[str, Any] = {
            "label": "unknown",
            "confidence": 0.0,
            "source": "not run",
        }
        self.current_compartment_moisture: List[Dict[str, Any]] = []
        self.duplicate_state: Dict[str, Any] = {"requires_decision": False}
        self.rejected_exported = False
        self.processing_finished = False
        self._pulse_job = None

        self._build_gui()

        if self.initial_photo_folder and self.initial_photo_folder.exists():
            self._load_photo_folder(self.initial_photo_folder, show_empty_warning=False)

        # Auto-load depth validation from Snowflake in background — safe, non-blocking
        self.root.after(200, self._load_from_snowflake)
    
    def _setup_output_folders(self):
        """Configure output folders for existing compartment checking"""
        folders = []
        if self.config:
            for key in ["local_extracted_compartments_folder", "shared_folder_extracted_compartments_folder",
                        "local_approved_compartments_folder", "shared_folder_approved_compartments_folder"]:
                folder = self.config.get(key, "")
                if folder:
                    folders.append(folder)
        self.existing_checker.set_output_folders(folders)
    
    @property
    def aruco_manager(self):
        """Lazy load ArUco manager"""
        if self._aruco_manager is None:
            try:
                from processing.ArucoMarkersAndBlurDetectionStep.aruco_manager import ArucoManager
                self._aruco_manager = ArucoManager(self.config)
            except Exception as e:
                self.logger.error(f"Failed to init ArUco manager: {e}")
        return self._aruco_manager
    
    @property
    def viz_manager(self):
        """Lazy load Visualization manager"""
        if self._viz_manager is None:
            try:
                from core.visualization_manager import VisualizationManager
                self._viz_manager = VisualizationManager(self.config)
            except Exception as e:
                self.logger.error(f"Failed to init Visualization manager: {e}")
        return self._viz_manager

    def _build_gui(self):
        """Build the GUI layout"""

        # Apply theme to root if gui_manager available
        if self.gui_manager:
            self.gui_manager.apply_theme(self.root)
            self.root.configure(bg=self.theme_colors["background"])

        # Top frame - File selection
        top_frame = tk.Frame(self.root, bg=self.theme_colors["background"])
        top_frame.pack(fill=tk.X, padx=10, pady=10)

        ModernButton(
            top_frame,
            text="Select CSV",
            color=self.theme_colors["accent_blue"],
            command=self._select_csv,
            theme_colors=self.theme_colors,
            fonts=self.fonts,
        ).pack(side=tk.LEFT, padx=5)
        self.csv_label = tk.Label(
            top_frame,
            text="No CSV selected",
            bg=self.theme_colors["background"],
            fg=self.theme_colors["text"],
            font=self.fonts["normal"],
        )
        self.csv_label.pack(side=tk.LEFT, padx=5)

        # Snowflake depth load — uses session manager singleton (no re-auth)
        self._sf_button = ModernButton(
            top_frame,
            text="Load Depths (Snowflake)",
            color=self.theme_colors.get("accent_blue", "#4a90c0"),
            command=self._load_from_snowflake,
            theme_colors=self.theme_colors,
            fonts=self.fonts,
        )
        self._sf_button.pack(side=tk.LEFT, padx=(20, 5))
        self._sf_status_label = tk.Label(
            top_frame,
            text="",
            bg=self.theme_colors["background"],
            fg=self.theme_colors["text"],
            font=self.fonts["small"],
        )
        self._sf_status_label.pack(side=tk.LEFT, padx=5)

        ModernButton(
            top_frame,
            text="Select Photo Folder",
            color=self.theme_colors["accent_blue"],
            command=self._select_folder,
            theme_colors=self.theme_colors,
            fonts=self.fonts,
        ).pack(side=tk.LEFT, padx=20)
        self.folder_label = tk.Label(
            top_frame,
            text="No folder selected",
            bg=self.theme_colors["background"],
            fg=self.theme_colors["text"],
            font=self.fonts["normal"],
        )
        self.folder_label.pack(side=tk.LEFT, padx=5)

        # Middle frame - Image display
        img_frame = tk.Frame(self.root, bg=self.theme_colors["background"])
        img_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(
            img_frame, bg=self.theme_colors["secondary_bg"], highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bottom frame - Controls
        control_frame = tk.LabelFrame(
            self.root,
            text="Rename Controls",
            bg=self.theme_colors["background"],
            fg=self.theme_colors["text"],
            font=self.fonts["heading"],
        )
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        # Current filename
        filename_frame = tk.Frame(control_frame, bg=self.theme_colors["background"])
        filename_frame.pack(fill=tk.X, pady=5)
        tk.Label(
            filename_frame,
            text="Current:",
            bg=self.theme_colors["background"],
            fg=self.theme_colors["text"],
            font=self.fonts["normal"],
        ).pack(side=tk.LEFT)
        self.current_filename_label = tk.Label(
            filename_frame,
            text="",
            bg=self.theme_colors["background"],
            fg=self.theme_colors["text"],
            font=self.fonts["heading"],
        )
        self.current_filename_label.pack(side=tk.LEFT, padx=10)
        
        # Preview toggle checkbox
        self.preview_checkbox = ttk.Checkbutton(
            filename_frame,
            text="Preview Processing",
            variable=self.preview_enabled,
            command=self._on_preview_toggle,
        )
        self.preview_checkbox.pack(side=tk.RIGHT, padx=20)

        self.safety_label = tk.Label(
            control_frame,
            text="Select a photo folder to begin.",
            bg=self.theme_colors.get("secondary_bg", "#3c3c3c"),
            fg=self.theme_colors["text"],
            font=self.fonts["heading"],
            anchor="w",
            padx=10,
            pady=6,
        )
        self.safety_label.pack(fill=tk.X, padx=6, pady=(0, 8))

        # Input fields
        input_frame = tk.Frame(control_frame, bg=self.theme_colors["background"])
        input_frame.pack(fill=tk.X, pady=10)

        hole_id_label = tk.Label(
            input_frame,
            text="Hole ID:",
            bg=self.theme_colors["background"],
            fg=self.theme_colors["text"],
            font=self.fonts["normal"],
        )
        hole_id_label.grid(row=0, column=0, padx=5, sticky=tk.E)

        self.hole_id_var = tk.StringVar()
        self.hole_id_var.trace_add("write", lambda *args: self._on_hole_id_changed())
        self.hole_id_entry = create_entry_with_validation(
            input_frame,
            textvariable=self.hole_id_var,
            theme_colors=self.theme_colors,
            font=self.fonts["normal"],
            width=20,
        )
        self.hole_id_entry.grid(row=0, column=1, padx=5)

        # Hole ID format feedback label
        self.hole_id_status_label = tk.Label(
            input_frame,
            text="",
            bg=self.theme_colors["background"],
            fg="#999999",
            font=self.fonts["small"],
        )
        self.hole_id_status_label.grid(row=1, column=0, columnspan=2, padx=5, sticky=tk.W)

        # Single "To Depth" field (from is calculated as to-20)
        to_label = tk.Label(
            input_frame,
            text="To Depth (m):",
            bg=self.theme_colors["background"],
            fg=self.theme_colors["text"],
            font=self.fonts["normal"],
        )
        to_label.grid(row=0, column=2, padx=5, sticky=tk.E)

        self.to_var = tk.StringVar()
        self.to_var.trace_add("write", lambda *args: self._update_from_display())
        self.to_entry = create_entry_with_validation(
            input_frame,
            textvariable=self.to_var,
            theme_colors=self.theme_colors,
            font=self.fonts["normal"],
            width=15,
        )
        self.to_entry.grid(row=0, column=3, padx=5)

        # Up/Down buttons for incrementing depth
        depth_button_frame = tk.Frame(input_frame, bg=self.theme_colors["background"])
        depth_button_frame.grid(row=0, column=4, padx=2)

        ModernButton(
            depth_button_frame,
            text="▲",
            color=self.theme_colors["accent_blue"],
            command=self._increment_depth_up,
            theme_colors=self.theme_colors,
            fonts=self.fonts,
            width=3,
        ).pack(side=tk.TOP, pady=1)

        ModernButton(
            depth_button_frame,
            text="▼",
            color=self.theme_colors["accent_blue"],
            command=self._increment_depth_down,
            theme_colors=self.theme_colors,
            fonts=self.fonts,
            width=3,
        ).pack(side=tk.TOP, pady=1)

        for child, label in zip(depth_button_frame.winfo_children(), ["+", "-"]):
            for grandchild in child.winfo_children():
                if isinstance(grandchild, tk.Label):
                    grandchild.config(text=label)
                    break

        # Display calculated from depth
        self.from_display_label = tk.Label(
            input_frame,
            text="",
            bg=self.theme_colors["background"],
            fg=self.theme_colors["accent_green"],
            font=self.fonts["normal"],
        )
        self.from_display_label.grid(row=0, column=5, padx=5, sticky=tk.W)

        ModernButton(
            input_frame,
            text="Increment from Last",
            color=self.theme_colors["accent_green"],
            command=self._increment_from_last,
            theme_colors=self.theme_colors,
            fonts=self.fonts,
        ).grid(row=0, column=6, padx=10)
        ModernButton(
            input_frame,
            text="Increment Hole",
            color=self.theme_colors["accent_green"],
            command=self._increment_hole,  # PEP-8: new helper method
            theme_colors=self.theme_colors,
            fonts=self.fonts,
        ).grid(row=0, column=7, padx=10)

        # Validation status
        self.validation_label = tk.Label(
            control_frame,
            text="",
            bg=self.theme_colors["background"],
            fg="#999999",
            font=self.fonts["small"],
        )
        self.validation_label.pack(pady=5)
        
        # Status row for existing check and preview info
        status_row = tk.Frame(control_frame, bg=self.theme_colors["background"])
        status_row.pack(fill=tk.X, pady=2)
        
        self.existing_label = tk.Label(
            status_row, text="", bg=self.theme_colors["background"],
            fg="#999999", font=self.fonts["small"],
        )
        self.existing_label.pack(side=tk.LEFT, padx=10)

        self.moisture_label = tk.Label(
            status_row,
            text="Moisture: waiting for preview",
            bg=self.theme_colors["background"],
            fg="#999999",
            font=self.fonts["small"],
        )
        self.moisture_label.pack(side=tk.LEFT, padx=10)
        
        self.preview_status_label = tk.Label(
            status_row, text="", bg=self.theme_colors["background"],
            fg="#999999", font=self.fonts["small"],
        )
        self.preview_status_label.pack(side=tk.RIGHT, padx=10)

        # Action buttons
        button_frame = tk.Frame(control_frame, bg=self.theme_colors["background"])
        button_frame.pack(pady=10)
        button_frame.pack_configure(fill=tk.X, padx=6, pady=(8, 4))

        ModernButton(
            button_frame,
            text="← Previous",
            color=self.theme_colors["accent_blue"],
            command=self._previous_photo,
            theme_colors=self.theme_colors,
            fonts=self.fonts,
        ).pack(side=tk.LEFT, padx=5)
        ModernButton(
            button_frame,
            text="Rename & Next →",
            color=self.theme_colors["accent_green"],
            command=self._rename_and_next,
            theme_colors=self.theme_colors,
            fonts=self.fonts,
        ).pack(side=tk.LEFT, padx=5)
        ModernButton(
            button_frame,
            text="Reject & Next →",
            color=self.theme_colors["accent_red"],
            command=self._reject_and_next,
            theme_colors=self.theme_colors,
            fonts=self.fonts,
        ).pack(side=tk.LEFT, padx=5)
        ModernButton(
            button_frame,
            text="Skip →",
            color=self.theme_colors["accent_blue"],
            command=self._next_photo,
            theme_colors=self.theme_colors,
            fonts=self.fonts,
        ).pack(side=tk.LEFT, padx=5)

        # Spacer
        tk.Frame(button_frame, bg=self.theme_colors["background"], width=30).pack(side=tk.LEFT)

        self.export_rejected_btn = ModernButton(
            button_frame,
            text="Export Rejected List",
            color="#8B6914",
            command=self._export_rejected_list,
            theme_colors=self.theme_colors,
            fonts=self.fonts,
        )
        self.export_rejected_btn.pack(side=tk.LEFT, padx=5)
        self.export_rejected_btn.set_state("disabled")
        self.rejected_count_label = tk.Label(
            button_frame,
            text="",
            bg=self.theme_colors["background"],
            fg=self.theme_colors["accent_red"],
            font=self.fonts["small"],
        )
        self.rejected_count_label.pack(side=tk.LEFT, padx=5)
        action_children = list(button_frame.winfo_children())
        if len(action_children) >= 6:
            previous_frame, rename_frame, reject_frame, skip_frame, _spacer_frame, export_frame = action_children[:6]
            for child in action_children:
                child.pack_forget()

            def set_button_label(frame, text):
                for grandchild in frame.winfo_children():
                    if isinstance(grandchild, tk.Label):
                        grandchild.config(text=text)
                        break

            set_button_label(previous_frame, "< Previous")
            set_button_label(rename_frame, "Rename & Next >")
            set_button_label(skip_frame, "Skip >")
            set_button_label(reject_frame, "Reject & Next >")
            set_button_label(export_frame, "Export Rejected List")

            previous_frame.pack(side=tk.LEFT, padx=(0, 5))
            rename_frame.pack(side=tk.LEFT, padx=5)
            skip_frame.pack(side=tk.LEFT, padx=5)
            reject_frame.pack(side=tk.RIGHT, padx=(20, 0))
            self.rejected_count_label.pack(side=tk.RIGHT, padx=5)
            export_frame.pack(side=tk.RIGHT, padx=5)

        self.export_note_label = tk.Label(
            control_frame,
            text="Rejected list is for retaking photos that are missing or unsafe to process.",
            bg=self.theme_colors["background"],
            fg="#999999",
            font=self.fonts["small"],
        )
        self.export_note_label.pack(pady=(0, 4))
        self._attach_tooltip(
            self.export_rejected_btn.frame,
            "Export a CSV list of rejected photos so they can be retaken if no usable copy exists.",
        )

        # Progress
        self.progress_label = tk.Label(
            control_frame,
            text="0 / 0",
            bg=self.theme_colors["background"],
            fg=self.theme_colors["text"],
            font=self.fonts["normal"],
        )
        self.progress_label.pack(pady=5)

        # Bind keyboard shortcuts — Left/Right/Delete suppress when an Entry has focus
        self.root.bind("<Return>", lambda e: self._rename_and_next())
        self.root.bind("<Right>", self._on_key_right)
        self.root.bind("<Left>", self._on_key_left)
        self.root.bind("<Delete>", self._on_key_delete)

        # Up/Down context-sensitive: depth on to_entry, hole on hole_id_entry
        self.to_entry.bind("<Up>", lambda e: (self._increment_depth_up(), "break")[1])
        self.to_entry.bind("<Down>", lambda e: (self._increment_depth_down(), "break")[1])
        self.hole_id_entry.bind("<Up>", lambda e: (self._increment_hole(), "break")[1])
        self.hole_id_entry.bind("<Down>", lambda e: (self._decrement_hole(), "break")[1])

        # Bind validation check on depth entry change
        # (hole_id validation fires via StringVar trace in _on_hole_id_changed)
        self.to_entry.bind("<KeyRelease>", lambda e: self._check_validation())

    def _attach_tooltip(self, widget, text: str):
        """Attach a small hover tooltip to a widget."""
        tooltip = {"window": None}

        def show(event):
            if tooltip["window"] is not None:
                return
            tip = tk.Toplevel(widget)
            tip.wm_overrideredirect(True)
            tip.wm_geometry(f"+{event.x_root + 12}+{event.y_root + 12}")
            label = tk.Label(
                tip,
                text=text,
                bg=self.theme_colors.get("secondary_bg", "#3c3c3c"),
                fg=self.theme_colors["text"],
                relief=tk.SOLID,
                borderwidth=1,
                padx=8,
                pady=5,
                font=self.fonts["small"],
                wraplength=320,
                justify=tk.LEFT,
            )
            label.pack()
            tooltip["window"] = tip

        def hide(_event=None):
            if tooltip["window"] is not None:
                tooltip["window"].destroy()
                tooltip["window"] = None

        widget.bind("<Enter>", show, add="+")
        widget.bind("<Leave>", hide, add="+")

    def _is_entry_focused(self, event) -> bool:
        """Check if the event came from a text entry widget"""
        widget = event.widget
        return isinstance(widget, (tk.Entry, ttk.Entry))

    def _on_key_right(self, event):
        """Right arrow: next photo unless typing in an entry"""
        if self._is_entry_focused(event):
            return  # let default cursor-move behaviour happen
        self._next_photo()

    def _on_key_left(self, event):
        """Left arrow: previous photo unless typing in an entry"""
        if self._is_entry_focused(event):
            return
        self._previous_photo()

    def _on_key_delete(self, event):
        """Delete key: reject photo unless typing in an entry"""
        if self._is_entry_focused(event):
            return
        self._reject_and_next()

    def _decrement_hole(self):
        """Decrement alphanumeric hole IDs like 'XY0001' -> 'XY0000'"""
        current = self.hole_id_var.get().strip()
        if not current:
            if self.last_hole_id:
                current = self.last_hole_id
            else:
                return

        m = re.match(r"^([A-Za-z]+)(\d+)$", current)
        if not m:
            return

        prefix, digits = m.groups()
        num = int(digits)
        if num <= 0:
            return

        prev_num = str(num - 1).zfill(len(digits))
        new_hole_id = f"{prefix.upper()}{prev_num}"

        # Reset to first interval (0-20)
        self.hole_id_var.set(new_hole_id)
        self.to_var.set("20")

    def _select_csv(self):
        """Select CSV file for validation"""
        filename = filedialog.askopenfilename(
            title="Select Depth Validation CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )

        if filename:
            if self.validator.load_csv(filename):
                self.csv_label.config(text=f"✓ {Path(filename).name}")
                self.csv_label.config(text=f"OK {Path(filename).name}")
                if hasattr(DialogHelper, "show_message") and self.gui_manager:
                    DialogHelper.show_message(
                        self.root,
                        "Success",
                        f"Loaded validation data for {len(self.validator.depth_ranges)} holes",
                        "info",
                    )
            else:
                self.csv_label.config(text="✗ Failed to load")

    def _load_from_snowflake(self):
        """
        Load collar depth ranges from the shared SnowflakeSessionManager.
        If Phase 1 is already complete (loaded at startup), this is instant.
        Otherwise waits up to 30s for Phase 1, then starts it if not running.
        """
        import threading

        def _run():
            try:
                self._sf_button.configure(state="disabled")
            except Exception:
                pass

            self.root.after(0, lambda: self._sf_status_label.config(
                text="⏳ Waiting for Snowflake collar data…",
                fg=self.theme_colors.get("accent_blue", "#4a90c0"),
            ))

            try:
                from processing.DataManager.snowflake_session import (
                    get_session_manager, SessionState,
                )
                sm = get_session_manager()

                # If phase 1 hasn't started yet, start it now
                if sm.state in (SessionState.IDLE, SessionState.FAILED):
                    sm.start_phase1()

                # Wait for phase 1 (up to 45s — SSO can be slow)
                got_data = sm.wait_for_phase1(timeout=45.0)

            except ImportError:
                got_data = False
                sm = None

            def _apply():
                try:
                    self._sf_button.configure(state="normal")
                except Exception:
                    pass

                if got_data and sm and sm.collar_depth_ranges:
                    self.validator.load_from_dict(sm.collar_depth_ranges)
                    self.csv_label.config(text="(Snowflake active)")
                    msg = f"✓ {len(sm.collar_depth_ranges):,} holes"
                    msg = f"OK {len(sm.collar_depth_ranges):,} holes"
                    self._sf_status_label.config(
                        text=msg,
                        fg=self.theme_colors.get("accent_green", "#5aa06c"),
                    )
                    self.logger.info(f"BulkRenamer: depth data from Snowflake ({len(sm.collar_depth_ranges):,} holes)")
                else:
                    err = getattr(sm, "error_message", "unavailable") if sm else "unavailable"
                    self._sf_status_label.config(
                        text=f"✗ Snowflake: {err}",
                        fg=self.theme_colors.get("accent_red", "#ff6b6b"),
                    )
                    self._sf_status_label.config(
                        text=f"Snowflake: {err}",
                        fg=self.theme_colors.get("accent_red", "#ff6b6b"),
                    )
                    self.logger.warning("BulkRenamer: Snowflake depth data not available")

                self._check_validation()

            self.root.after(0, _apply)

        threading.Thread(target=_run, daemon=True, name="sf-bulk-depth").start()

    def _select_folder(self):
        """Select folder containing photos"""
        folder = filedialog.askdirectory(title="Select Photo Folder")

        if folder:
            self._load_photo_folder(Path(folder), show_empty_warning=True)
            return

            self.photo_folder = Path(folder)

            # Load all image files
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
            self.photo_files = [
                f
                for f in sorted(self.photo_folder.iterdir())
                if f.suffix.lower() in image_extensions
            ]

            if self.photo_files:
                self.folder_label.config(text=f"✓ {len(self.photo_files)} photos")
                self.current_index = 0
                self._load_current_photo()
            else:
                self.folder_label.config(text="✗ No photos found")
                if hasattr(DialogHelper, "show_message") and self.gui_manager:
                    DialogHelper.show_message(
                        self.root,
                        "No Photos",
                        "No image files found in selected folder",
                        "warning",
                    )

    def _load_photo_folder(self, folder: Path, show_empty_warning: bool = True):
        """Load images from a photo folder and display the first one."""
        self.photo_folder = Path(folder)
        self.rejected_folder = None
        self.rejected_files.clear()
        self.rejected_exported = False
        self.processing_finished = False
        self._stop_export_pulse()

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        self.photo_files = [
            f
            for f in sorted(self.photo_folder.iterdir())
            if f.suffix.lower() in image_extensions
            and f.parent.name.lower() != "rejected"
        ]

        if self.photo_files:
            self.folder_label.config(text=f"✓ {len(self.photo_files)} photos")
            self.current_index = 0
            self.folder_label.config(text=f"OK {len(self.photo_files)} photos")
            self._update_rejected_count()
            self._load_current_photo()
        else:
            self.folder_label.config(text="No photos found")
            self.canvas.delete("all")
            self.current_filename_label.config(text="")
            self.safety_label.config(
                text="No photos found in the selected folder.",
                bg=self.theme_colors.get("accent_error", "#3d2222"),
            )
            self._update_rejected_count()
            if show_empty_warning and hasattr(DialogHelper, "show_message") and self.gui_manager:
                DialogHelper.show_message(
                    self.root,
                    "No Photos",
                    "No image files found in selected folder",
                    "warning",
                )

    def _on_preview_toggle(self):
        """Handle preview checkbox toggle"""
        self._load_current_photo()
    
    def _generate_processing_preview(self, image_path: Path) -> Optional[Dict]:
        """Generate processing preview data for an image."""
        if not self.aruco_manager:
            return None
        
        try:
            mtime = image_path.stat().st_mtime
            cached = self.preview_cache.get(str(image_path), mtime)
            if cached:
                return cached
        except:
            mtime = 0
        
        try:
            pil_img = Image.open(image_path)
            pil_img = ImageOps.exif_transpose(pil_img)
            
            img_array = np.array(pil_img)
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            elif img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            else:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Downscale
            max_dim = self.config.get("working_image_max_dimension", 2560)
            h, w = img_array.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img_array = cv2.resize(img_array, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            
            markers = self.aruco_manager.improve_marker_detection(img_array)
            
            result = {
                "markers": markers, "marker_count": len(markers),
                "compartment_marker_count": len([m for m in markers if 4 <= m <= 23]),
                "rotation_angle": 0.0, "scale_px_per_cm": None,
                "boundaries": [], "image_array": img_array,
                "preview_size": (int(img_array.shape[1]), int(img_array.shape[0])),
            }
            
            # Skew correction
            if self.viz_manager and markers:
                try:
                    self.viz_manager.set_working_image(img_array)
                    correction = self.viz_manager.correct_image_skew(markers)
                    
                    # Always capture the calculated rotation angle
                    result["rotation_angle"] = correction.get("rotation_angle", 0.0)
                    
                    if correction.get("needs_redetection"):
                        img_array = correction["image"]
                        result["image_array"] = img_array
                        result["preview_size"] = (int(img_array.shape[1]), int(img_array.shape[0]))
                        # Re-detect markers on rotated image
                        markers = self.aruco_manager.improve_marker_detection(img_array)
                        result["markers"] = markers
                        result["marker_count"] = len(markers)
                        result["compartment_marker_count"] = len([m for m in markers if 4 <= m <= 23])
                        self.logger.debug(f"Applied rotation: {result['rotation_angle']:.2f}°, re-detected {len(markers)} markers")
                    else:
                        self.logger.debug(f"Rotation {result['rotation_angle']:.2f}° below threshold, not applied")
                except Exception as e:
                    self.logger.warning(f"Skew correction failed: {e}")
            
            # Scale estimation
            if markers and len(markers) >= 4 and self.viz_manager:
                try:
                    marker_config = {
                        "corner_marker_size_cm": self.config.get("corner_marker_size_cm", 1.0),
                        "compartment_marker_size_cm": self.config.get("compartment_marker_size_cm", 2.0),
                        "corner_ids": self.config.get("corner_marker_ids", [0, 1, 2, 3]),
                        "compartment_ids": self.config.get("compartment_marker_ids", list(range(4, 24))),
                    }
                    scale_data = self.viz_manager.estimate_scale_from_markers(markers, marker_config)
                    if scale_data:
                        result["scale_px_per_cm"] = scale_data.get("scale_px_per_cm")
                except Exception as e:
                    self.logger.warning(f"Scale estimation failed: {e}")
            
            # Boundary analysis
            if markers:
                try:
                    analysis = self.aruco_manager.analyze_compartment_boundaries(
                        img_array, markers,
                        compartment_count=self.config.get("compartment_count", 20),
                        smart_cropping=True,
                        metadata={"scale_px_per_cm": result.get("scale_px_per_cm")},
                    )
                    result["boundaries"] = analysis.get("boundaries", [])
                    result["missing_markers"] = analysis.get("missing_marker_ids", [])
                    result["vertical_constraints"] = analysis.get("vertical_constraints")
                except Exception as e:
                    self.logger.warning(f"Boundary analysis failed: {e}")

            result["marker_quality"] = self._analyse_marker_quality(result)
            
            corrected_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            result["corrected_image"] = Image.fromarray(corrected_rgb)
            
            self.preview_cache.set(str(image_path), mtime, result)
            return result
            
        except Exception as e:
            self.logger.error(f"Preview generation failed: {e}")
            return None

    def _analyse_marker_quality(self, preview_data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize marker count, order, and spacing quality."""
        expected_count = int(self.config.get("compartment_count", 20))
        expected_ids = list(range(4, 4 + expected_count))
        markers = preview_data.get("markers") or {}
        detected_ids = sorted(marker_id for marker_id in markers if marker_id in expected_ids)
        missing_ids = [marker_id for marker_id in expected_ids if marker_id not in markers]

        centers = []
        for marker_id in detected_ids:
            corners = markers.get(marker_id)
            if corners is None:
                continue
            centers.append((marker_id, float(np.mean(corners[:, 0]))))

        order_ok = True
        if len(centers) >= 2:
            order_by_x = [marker_id for marker_id, _x in sorted(centers, key=lambda item: item[1])]
            order_ok = order_by_x == sorted(order_by_x)

        gap_warning = False
        gap_ratio = None
        if len(centers) >= 4:
            xs = [x for _marker_id, x in sorted(centers, key=lambda item: item[1])]
            gaps = [b - a for a, b in zip(xs, xs[1:]) if b > a]
            if gaps:
                median_gap = float(np.median(gaps))
                if median_gap > 0:
                    max_gap = max(gaps)
                    min_gap = min(gaps)
                    gap_ratio = max_gap / median_gap
                    gap_warning = max_gap > median_gap * 2.2 or min_gap < median_gap * 0.35

        boundary_count = len(preview_data.get("boundaries", []))
        ok = (
            len(missing_ids) == 0
            and order_ok
            and not gap_warning
            and boundary_count >= expected_count
        )

        return {
            "ok": ok,
            "expected_count": expected_count,
            "detected_count": len(detected_ids),
            "missing_ids": missing_ids,
            "order_ok": order_ok,
            "gap_warning": gap_warning,
            "gap_ratio": gap_ratio,
            "boundary_count": boundary_count,
        }
    
    def _draw_preview_overlay(
        self,
        pil_img: Image.Image,
        preview_data: Dict,
        scale_x: float,
        scale_y: float,
    ) -> Image.Image:
        """Draw boundary boxes, markers, and vertical constraints on the image"""
        img = pil_img.copy()
        draw = ImageDraw.Draw(img)
        
        # Draw vertical constraint lines (top/bottom of tray area)
        vert = preview_data.get("vertical_constraints")
        if vert:
            top_y, bottom_y = vert
            img_width = img.size[0]
            scaled_top = int(top_y * scale_y)
            scaled_bottom = int(bottom_y * scale_y)
            draw.line([(0, scaled_top), (img_width, scaled_top)], fill=(255, 100, 100), width=2)
            draw.line([(0, scaled_bottom), (img_width, scaled_bottom)], fill=(255, 100, 100), width=2)
        
        # Draw boundaries (green boxes)
        moisture_by_index = {
            record.get("index"): record
            for record in self.current_compartment_moisture
            if isinstance(record, dict)
        }
        for index, (x1, y1, x2, y2) in enumerate(preview_data.get("boundaries", []), start=1):
            sx1, sy1 = int(x1 * scale_x), int(y1 * scale_y)
            sx2, sy2 = int(x2 * scale_x), int(y2 * scale_y)
            draw.rectangle([sx1, sy1, sx2, sy2], outline=(0, 255, 0), width=2)
            moisture = moisture_by_index.get(index)
            if moisture:
                label = str(moisture.get("label", "?"))[:1].upper()
                if label == "W":
                    fill = (0, 190, 255)
                elif label == "D":
                    fill = (255, 185, 65)
                elif label == "E":
                    fill = (165, 165, 165)
                else:
                    fill = (230, 230, 230)
                draw.text((sx1 + 3, max(0, sy1 - 13)), label, fill=fill)
        
        # Draw markers
        for marker_id, corners in preview_data.get("markers", {}).items():
            cx = int(np.mean(corners[:, 0]) * scale_x)
            cy = int(np.mean(corners[:, 1]) * scale_y)
            
            if marker_id in (0, 1, 2, 3):
                # Corner markers - yellow/orange, larger
                draw.rectangle([cx - 6, cy - 6, cx + 6, cy + 6], outline=(255, 200, 0), width=2)
                draw.text((cx + 8, cy - 8), str(marker_id), fill=(255, 200, 0))
            elif 4 <= marker_id <= 23:
                # Compartment markers - cyan dots
                draw.rectangle([cx - 3, cy - 3, cx + 3, cy + 3], fill=(0, 200, 255))
        
        return img
    
    def _update_preview_status(self, preview_data: Optional[Dict]):
        """Update the preview status label"""
        if not preview_data:
            self.preview_status_label.config(text="Preview: Analysis failed", fg="red")
            self.safety_label.config(
                text="Preview failed. Reject or inspect this photo before processing.",
                bg=self.theme_colors.get("accent_error", "#3d2222"),
            )
            return

        quality = preview_data.get("marker_quality") or {}
        if quality:
            if quality.get("ok"):
                self.safety_label.config(
                    text="Preview checks passed: markers are complete, ordered, and ready.",
                    bg=self.theme_colors.get("accent_valid", "#1f3d2a"),
                )
            else:
                issues = []
                missing = quality.get("missing_ids") or []
                if missing:
                    issues.append(f"missing markers {missing}")
                if not quality.get("order_ok", True):
                    issues.append("markers out of order")
                if quality.get("gap_warning"):
                    issues.append("unusual marker spacing")
                if quality.get("boundary_count", 0) < quality.get("expected_count", 20):
                    issues.append(
                        f"{quality.get('boundary_count', 0)}/{quality.get('expected_count', 20)} bounds"
                    )
                issue_text = ", ".join(issues) or "preview checks need attention"
                self.safety_label.config(
                    text=f"Needs decision: {issue_text}.",
                    bg=self.theme_colors.get("accent_error", "#3d2222"),
                )
        
        parts = []
        comp_count = preview_data.get("compartment_marker_count", 0)
        expected = self.config.get("compartment_count", 20)

        if comp_count >= expected:
            parts.append(f"OK {comp_count}/{expected} markers")
            color = "green"
        elif comp_count >= expected - 2:
            parts.append(f"WARN {comp_count}/{expected} markers")
            color = "orange"
        else:
            parts.append(f"FAIL {comp_count}/{expected} markers")
            color = "red"

        rotation = preview_data.get("rotation_angle", 0)
        if rotation is not None:
            parts.append(f"Rot: {rotation:.1f} deg")

        scale = preview_data.get("scale_px_per_cm")
        if scale:
            parts.append(f"{scale:.1f} px/cm")

        parts.append(f"{len(preview_data.get('boundaries', []))} bounds")
        self.preview_status_label.config(text=" | ".join(parts), fg=color)
        return
        
        if comp_count >= expected:
            parts.append(f"✓ {comp_count}/{expected} markers")
            color = "green"
        elif comp_count >= expected - 2:
            parts.append(f"⚠ {comp_count}/{expected} markers")
            color = "orange"
        else:
            parts.append(f"✗ {comp_count}/{expected} markers")
            color = "red"
        
        rotation = preview_data.get("rotation_angle", 0)
        if rotation is not None:
            parts.append(f"Rot: {rotation:.1f}°")
        
        scale = preview_data.get("scale_px_per_cm")
        if scale:
            parts.append(f"{scale:.1f} px/cm")
        
        parts.append(f"{len(preview_data.get('boundaries', []))} bounds")
        
        self.preview_status_label.config(text=" | ".join(parts), fg=color)

    def _get_moisture_predictor(self):
        """Load the wet/dry classifier once, preferring the new gate model."""
        if self._moisture_predictor_checked:
            return self._moisture_predictor

        self._moisture_predictor_checked = True
        try:
            from ml_pipeline.predictor import get_predictor

            predictor = get_predictor()
            candidate_paths = [
                Path("ml_output/wetdryempty_gate/checkpoints/best_model.pt"),
                Path(__file__).resolve().parents[2]
                / "ml_output"
                / "wetdryempty_gate"
                / "checkpoints"
                / "best_model.pt",
                Path("ml_output/checkpoints/best_model.pt"),
            ]
            for model_path in candidate_paths:
                if model_path.exists() and predictor.load_model(model_path):
                    self._moisture_predictor = predictor
                    return predictor
        except Exception as e:
            self.logger.warning("Moisture classifier unavailable: %s", e)

        self._moisture_predictor = None
        return None

    def _normalise_moisture_label(self, label: str) -> str:
        normalized = str(label or "unknown").strip().capitalize()
        if normalized not in {"Wet", "Dry", "Empty"}:
            return "unknown"
        return normalized

    def _set_moisture_result(self, label: str, confidence: float, source: str):
        """Store and display a fallback moisture classification."""
        normalized = self._normalise_moisture_label(label)
        self.current_compartment_moisture = []

        self.current_moisture = {
            "label": normalized,
            "confidence": float(confidence or 0.0),
            "source": source,
        }

        if normalized in {"Wet", "Dry"}:
            color = self.theme_colors.get("accent_green", "#5aa06c")
            text = f"Moisture by compartment: {normalized} fallback ({confidence * 100:.0f}% ML)"
        elif normalized == "Empty":
            color = self.theme_colors.get("accent_yellow", "#f0b429")
            text = f"Moisture by compartment: Empty/uncertain ({confidence * 100:.0f}% ML)"
        else:
            color = "#999999"
            text = f"Moisture by compartment: unknown ({source})"
        self.moisture_label.config(text=text, fg=color)

    def _set_compartment_moisture_results(
        self,
        records: List[Dict[str, Any]],
        source: str,
    ):
        """Store and display wet/dry/empty predictions per chip compartment."""
        self.current_compartment_moisture = records
        counts = Counter(record.get("label", "unknown") for record in records)
        wet_count = counts.get("Wet", 0)
        dry_count = counts.get("Dry", 0)
        empty_count = counts.get("Empty", 0)
        unknown_count = counts.get("unknown", 0)
        usable = [record for record in records if record.get("label") in {"Wet", "Dry"}]

        if not records:
            summary_label = "unknown"
            summary_confidence = 0.0
            color = "#999999"
            text = f"Moisture by compartment: unknown ({source})"
        elif usable and len({record["label"] for record in usable}) == 1:
            summary_label = usable[0]["label"]
            summary_confidence = sum(
                float(record.get("confidence", 0.0)) for record in usable
            ) / max(1, len(usable))
            color = self.theme_colors.get("accent_green", "#5aa06c")
            text = (
                f"Moisture by compartment: {wet_count} Wet, {dry_count} Dry"
                f" ({source})"
            )
        elif wet_count and dry_count:
            summary_label = "mixed"
            summary_confidence = sum(
                float(record.get("confidence", 0.0)) for record in usable
            ) / max(1, len(usable))
            color = self.theme_colors.get("accent_yellow", "#f0b429")
            text = (
                f"Moisture by compartment: {wet_count} Wet, {dry_count} Dry"
                f" ({source})"
            )
        elif empty_count and not usable:
            summary_label = "Empty"
            summary_confidence = max(
                (float(record.get("confidence", 0.0)) for record in records),
                default=0.0,
            )
            color = self.theme_colors.get("accent_yellow", "#f0b429")
            text = f"Moisture by compartment: {empty_count} Empty/uncertain ({source})"
        else:
            summary_label = "unknown"
            summary_confidence = 0.0
            color = "#999999"
            text = f"Moisture by compartment: {unknown_count or len(records)} unknown ({source})"

        detail_parts = []
        if empty_count and (wet_count or dry_count):
            detail_parts.append(f"{empty_count} Empty")
        if unknown_count:
            detail_parts.append(f"{unknown_count} unknown")
        if detail_parts:
            text = f"{text}; " + ", ".join(detail_parts)

        self.current_moisture = {
            "label": summary_label,
            "confidence": summary_confidence,
            "source": source,
        }
        self.moisture_label.config(text=text, fg=color)

    def _update_moisture_prediction(self, image_path: Path, preview_data: Dict[str, Any]):
        """Predict wet/dry/empty state for each detected compartment crop."""
        predictor = self._get_moisture_predictor()
        if not predictor or not getattr(predictor, "is_available", False):
            self._set_moisture_result("unknown", 0.0, "ML unavailable")
            return

        corrected = preview_data.get("corrected_image")
        boundaries = preview_data.get("boundaries") or []
        if not corrected or not boundaries:
            self._set_moisture_result("unknown", 0.0, "no preview crops")
            return

        try:
            with tempfile.TemporaryDirectory(prefix="geovue_moisture_") as tmp_dir:
                crop_paths = []
                crop_records = []
                for idx, (x1, y1, x2, y2) in enumerate(boundaries[:20]):
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = corrected.crop((int(x1), int(y1), int(x2), int(y2)))
                    if crop.size[0] < 8 or crop.size[1] < 8:
                        continue
                    crop_path = Path(tmp_dir) / f"crop_{idx:02d}.jpg"
                    crop.convert("RGB").save(crop_path, quality=92)
                    crop_paths.append(str(crop_path))
                    crop_records.append(
                        {
                            "index": idx + 1,
                            "box": (int(x1), int(y1), int(x2), int(y2)),
                        }
                    )

                if not crop_paths:
                    self._set_moisture_result("unknown", 0.0, "no valid crops")
                    return

                predictions = predictor.predict_batch(crop_paths, batch_size=16)

            records = []
            for crop_record, prediction in zip(crop_records, predictions):
                label, confidence = prediction
                normalized = self._normalise_moisture_label(label)
                records.append(
                    {
                        **crop_record,
                        "label": normalized,
                        "confidence": float(confidence or 0.0),
                    }
                )

            if not records:
                self._set_moisture_result("unknown", 0.0, "ML no results")
                return

            self._set_compartment_moisture_results(records, f"ML {len(records)}/20")
        except Exception as e:
            self.logger.warning("Moisture prediction failed for %s: %s", image_path, e)
            self._set_moisture_result("unknown", 0.0, "ML error")
    
    def _check_existing_compartments(self):
        """Check if compartments already exist for current hole/depth"""
        hole_id = self.hole_id_var.get().strip()
        to_str = self.to_var.get().strip()
        
        if not hole_id or not to_str:
            self.existing_label.config(text="")
            self.duplicate_state = {"requires_decision": False}
            return
        
        try:
            to_depth = int(float(to_str))
            from_depth = to_depth - self.interval_length
            result = self.existing_checker.check_existing(hole_id, from_depth, to_depth)
            moisture = self.current_moisture.get("label", "unknown")
            self.duplicate_state = {
                "requires_decision": False,
                "message": "",
                "existing": result,
                "moisture": moisture,
                "compartment_moisture": self.current_compartment_moisture,
            }

            if result["exists"]:
                existing_states = []
                if result.get("has_wet"):
                    existing_states.append("Wet")
                if result.get("has_dry"):
                    existing_states.append("Dry")
                state_text = "/".join(existing_states) if existing_states else "unknown state"

                same_state_conflict = moisture in existing_states
                both_states_exist = result.get("has_wet") and result.get("has_dry")
                mixed_current = moisture == "mixed"
                unknown_current = moisture not in {"Wet", "Dry"}
                unknown_existing = not existing_states

                if both_states_exist or same_state_conflict or unknown_current or unknown_existing:
                    self.duplicate_state["requires_decision"] = True
                    if both_states_exist:
                        message = "Conflict: wet and dry already exist for this interval"
                        color = self.theme_colors.get("accent_red", "red")
                    elif same_state_conflict:
                        message = f"Conflict: existing {moisture} copy for this interval"
                        color = self.theme_colors.get("accent_red", "red")
                    elif unknown_existing:
                        message = "Existing interval has unknown moisture state"
                        color = "orange"
                    elif mixed_current:
                        message = f"Existing {state_text}; current tray has mixed wet/dry compartments"
                        color = "orange"
                    else:
                        message = f"Existing {state_text}; current moisture unknown"
                        color = "orange"
                    self.duplicate_state["message"] = message
                    self.existing_label.config(text=message, fg=color)
                else:
                    message = f"OK: existing {state_text}; current {moisture} makes a wet/dry pair"
                    self.duplicate_state["message"] = message
                    self.existing_label.config(
                        text=message,
                        fg=self.theme_colors.get("accent_green", "green"),
                    )
            else:
                self.existing_label.config(text="New tray interval", fg="green")
            return
            
            if result["exists"]:
                wet_dry = ""
                if result.get("has_wet") and result.get("has_dry"):
                    wet_dry = f" (W:{result['wet_count']}, D:{result['dry_count']})"
                elif result.get("has_wet"):
                    wet_dry = " (Wet)"
                elif result.get("has_dry"):
                    wet_dry = " (Dry)"
                self.existing_label.config(text=f"⚠ Existing: {result['count']} files{wet_dry}", fg="orange")
            else:
                self.existing_label.config(text="✓ New tray", fg="green")
        except ValueError:
            self.existing_label.config(text="")

    def _load_current_photo(self):
        """Load and display current photo"""
        if not self.photo_files or self.current_index >= len(self.photo_files):
            return

        current_file = self.photo_files[self.current_index]
        self.current_filename_label.config(text=current_file.name)

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1:
            canvas_width = 800
        if canvas_height <= 1:
            canvas_height = 500

        try:
            if self.preview_enabled.get():
                # Show processing preview
                self.root.config(cursor="wait")
                self.root.update()
                
                preview_data = self._generate_processing_preview(current_file)
                self.current_preview_data = preview_data
                
                if preview_data and preview_data.get("corrected_image"):
                    img = preview_data["corrected_image"].copy()
                    source_size = preview_data.get("preview_size") or img.size
                    img.thumbnail((canvas_width - 20, canvas_height - 20), Image.Resampling.LANCZOS)
                    self._update_preview_status(preview_data)
                    self._update_moisture_prediction(current_file, preview_data)
                    source_w = max(1, int(source_size[0]))
                    source_h = max(1, int(source_size[1]))
                    scale_x = img.size[0] / source_w
                    scale_y = img.size[1] / source_h
                    img = self._draw_preview_overlay(img, preview_data, scale_x, scale_y)
                else:
                    img = Image.open(current_file)
                    img = ImageOps.exif_transpose(img)
                    img.thumbnail((canvas_width - 20, canvas_height - 20), Image.Resampling.LANCZOS)
                    self._update_preview_status(None)
                    self._set_moisture_result("unknown", 0.0, "preview failed")
                
                self.root.config(cursor="")
            else:
                # Fast display without preview
                img = Image.open(current_file)
                img = ImageOps.exif_transpose(img)
                img.thumbnail((canvas_width - 20, canvas_height - 20), Image.Resampling.LANCZOS)
                self.current_preview_data = None
                self.preview_status_label.config(text="")
                self._set_moisture_result("unknown", 0.0, "preview off")
                self.safety_label.config(
                    text="Preview processing is off. Rename requires confirmation.",
                    bg=self.theme_colors.get("accent_yellow", "#8B6914"),
                )

            self.current_image = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=self.current_image)

        except Exception as e:
            self.logger.error(f"Image load error: {e}")
            messagebox.showerror("Image Load Error", f"Failed to load image:\n{str(e)}")
            self.root.config(cursor="")

        self._update_progress()
        self._check_validation()
        self._check_existing_compartments()

    # Standard RC hole ID: 2 uppercase letters + 4 digits (BA0001, KM0137, etc.)
    _HOLE_ID_PATTERN = re.compile(r"^[A-Z]{2}\d{4}$")

    def _on_hole_id_changed(self):
        """Auto-uppercase hole ID input and show format feedback"""
        raw = self.hole_id_var.get()
        upper = raw.upper()
        # Auto-uppercase without re-triggering trace infinitely
        if upper != raw:
            # Temporarily suppress trace to avoid recursion
            self.hole_id_var.set(upper)
            return  # The set() will re-trigger this method with the uppercased value

        if not upper:
            self.hole_id_status_label.config(text="Format: XX0000", fg="#999999")
            return

        if self._HOLE_ID_PATTERN.match(upper):
            self.hole_id_status_label.config(text="OK", fg=self.theme_colors["accent_green"])
        elif len(upper) < 6:
            self.hole_id_status_label.config(text="Too short - need XX0000", fg=self.theme_colors["accent_red"])
        elif len(upper) > 6:
            self.hole_id_status_label.config(text="Too long - need XX0000", fg=self.theme_colors["accent_red"])
        elif not upper[:2].isalpha():
            self.hole_id_status_label.config(text="Must start with 2 letters", fg=self.theme_colors["accent_red"])
        elif not upper[2:].isdigit():
            self.hole_id_status_label.config(text="Must end with 4 digits", fg=self.theme_colors["accent_red"])
        else:
            self.hole_id_status_label.config(text="Invalid format - need XX0000", fg=self.theme_colors["accent_red"])

        self._check_validation()
        self._check_existing_compartments()
        return

        if self._HOLE_ID_PATTERN.match(upper):
            self.hole_id_status_label.config(text="✓", fg=self.theme_colors["accent_green"])
        else:
            # Show what's wrong
            if len(upper) < 6:
                self.hole_id_status_label.config(text="Too short — need XX0000", fg=self.theme_colors["accent_red"])
            elif len(upper) > 6:
                self.hole_id_status_label.config(text="Too long — need XX0000", fg=self.theme_colors["accent_red"])
            elif not upper[:2].isalpha():
                self.hole_id_status_label.config(text="Must start with 2 letters", fg=self.theme_colors["accent_red"])
            elif not upper[2:].isdigit():
                self.hole_id_status_label.config(text="Must end with 4 digits", fg=self.theme_colors["accent_red"])
            else:
                self.hole_id_status_label.config(text="Invalid format — need XX0000", fg=self.theme_colors["accent_red"])

        self._check_validation()
        self._check_existing_compartments()

    def _validate_hole_id(self, hole_id: str) -> tuple:
        """Validate hole ID format. Returns (is_valid, error_message)."""
        if not hole_id:
            return False, "Hole ID is required"
        if not self._HOLE_ID_PATTERN.match(hole_id):
            return False, f"'{hole_id}' is not a valid Hole ID.\nExpected format: XX0000 (e.g. BA0001, KM0137)"
        return True, ""

    def _check_validation(self):
        """Check if current entries are valid"""
        hole_id = self.hole_id_var.get().strip()
        to_str = self.to_var.get().strip()

        if not hole_id or not to_str:
            self.validation_label.config(
                text="Enter hole ID and to depth", foreground="gray"
            )
            self.existing_label.config(text="")
            return

        try:
            to_depth = float(to_str)
            from_depth = to_depth - self.interval_length

            is_valid, message = self.validator.validate_depth(
                hole_id, from_depth, to_depth
            )

            if is_valid:
                self.validation_label.config(text=message, foreground="green")
            else:
                self.validation_label.config(text=message, foreground="red")

        except ValueError:
            self.validation_label.config(text="Invalid depth value", foreground="red")

    def _update_from_display(self):
        """Update the display showing the calculated 'from' depth"""
        to_str = self.to_var.get().strip()

        if to_str:
            try:
                to_depth = float(to_str)
                from_depth = to_depth - self.interval_length
                self.from_display_label.config(
                    text=f"(From: {from_depth:.1f}m)",
                    foreground=self.theme_colors["accent_green"],
                )
                self._check_validation()
                self._check_existing_compartments()
            except ValueError:
                self.from_display_label.config(
                    text="(Invalid)", foreground=self.theme_colors["accent_red"]
                )
                self.duplicate_state = {"requires_decision": False}
        else:
            self.from_display_label.config(text="")
            self.duplicate_state = {"requires_decision": False}

    def _increment_depth_up(self):
        """Increment the to depth by interval_length"""
        to_str = self.to_var.get().strip()

        if to_str:
            try:
                to_depth = float(to_str)
                new_to = to_depth + self.interval_length
                self.to_var.set(f"{new_to:.1f}" if new_to % 1 else f"{int(new_to)}")
            except ValueError:
                messagebox.showwarning(
                    "Invalid Depth", "Current depth value is invalid"
                )
        else:
            # Start at 20
            self.to_var.set("20")

    def _increment_depth_down(self):
        """Decrement the to depth by interval_length"""
        to_str = self.to_var.get().strip()

        if to_str:
            try:
                to_depth = float(to_str)
                new_to = to_depth - self.interval_length

                # Don't go below interval_length (from would be negative)
                if new_to < self.interval_length:
                    messagebox.showwarning(
                        "Invalid Range",
                        f"Cannot go below {self.interval_length}m (from would be negative)",
                    )
                    return

                self.to_var.set(f"{new_to:.1f}" if new_to % 1 else f"{int(new_to)}")
            except ValueError:
                messagebox.showwarning(
                    "Invalid Depth", "Current depth value is invalid"
                )

    def _increment_from_last(self):
        """Increment depth values from last entry"""
        if not self.last_hole_id or not self.last_to:
            messagebox.showinfo(
                "No Last Values", "No previous values to increment from"
            )
            return

        try:
            last_to_depth = float(self.last_to)
            new_to = last_to_depth + self.interval_length

            self.hole_id_var.set(self.last_hole_id)
            self.to_var.set(f"{new_to:.1f}" if new_to % 1 else f"{int(new_to)}")

        except ValueError:
            messagebox.showerror("Error", "Invalid last values")

    def _increment_hole(self):
        """
        Increment alphanumeric hole IDs like 'XY0000' -> 'XY0001' and reset depths
        to 20m (first interval). Falls back to last_hole_id if field is empty.
        """
        current = self.hole_id_var.get().strip()
        if not current:
            if self.last_hole_id:
                current = self.last_hole_id
            else:
                messagebox.showwarning("Missing Hole ID", "Enter a Hole ID to increment")
                return

        m = re.match(r"^([A-Za-z]+)(\d+)$", current)
        if not m:
            messagebox.showwarning(
                "Invalid Hole ID",
                "Hole ID must be letters followed by digits, e.g., 'XY0000'",
            )
            return

        prefix, digits = m.groups()
        next_num = str(int(digits) + 1).zfill(len(digits))
        new_hole_id = f"{prefix.upper()}{next_num}"

        # Reset to first interval (0-20)
        self.hole_id_var.set(new_hole_id)
        self.to_var.set("20")

    def _auto_increment_next(self):
        """Automatically increment depth values for next photo.
        
        If depth validation data is loaded and the next interval would exceed
        the known max depth for this hole, clear the hole ID and reset depth
        to 20 so the user is prompted to enter the next hole.
        """
        if not self.last_hole_id or not self.last_to:
            return

        try:
            last_to_depth = float(self.last_to)
            new_to = last_to_depth + self.interval_length
            new_from = new_to - self.interval_length

            # Check if we've exceeded the known depth range for this hole
            if self.validator.depth_ranges:
                hole_upper = self.last_hole_id.strip().upper()
                if hole_upper in self.validator.depth_ranges:
                    valid_min, valid_max = self.validator.depth_ranges[hole_upper]
                    if new_to > valid_max:
                        # Hole is complete — clear fields for next hole
                        self.hole_id_var.set("")
                        self.to_var.set("20")
                        self.validation_label.config(
                            text=f"✓ {self.last_hole_id} complete (max depth {valid_max}m reached)",
                            foreground=self.theme_colors["accent_green"],
                        )
                        self.hole_id_entry.focus_set()
                        return

            self.hole_id_var.set(self.last_hole_id)
            self.to_var.set(f"{new_to:.1f}" if new_to % 1 else f"{int(new_to)}")

        except ValueError:
            # Silently fail - don't interrupt the workflow
            pass

    def _preview_requires_decision(self) -> Tuple[bool, str]:
        """Return whether the current preview needs a human decision."""
        if not self.preview_enabled.get():
            return True, "Preview processing is turned off."
        if not self.current_preview_data:
            return True, "Preview processing has not completed for this photo."
        quality = self.current_preview_data.get("marker_quality") or {}
        if quality and not quality.get("ok"):
            issues = []
            missing = quality.get("missing_ids") or []
            if missing:
                issues.append(f"missing markers {missing}")
            if not quality.get("order_ok", True):
                issues.append("markers out of order")
            if quality.get("gap_warning"):
                issues.append("unusual marker spacing")
            if quality.get("boundary_count", 0) < quality.get("expected_count", 20):
                issues.append(
                    f"{quality.get('boundary_count', 0)}/{quality.get('expected_count', 20)} boundaries"
                )
            return True, ", ".join(issues) or "Preview checks failed."
        return False, ""

    def _confirm_risky_rename(self) -> bool:
        """Ask for confirmation when preview or duplicate checks are risky."""
        messages = []
        preview_risky, preview_message = self._preview_requires_decision()
        if preview_risky:
            messages.append(f"Preview: {preview_message}")

        if self.duplicate_state.get("requires_decision"):
            messages.append(f"Duplicate check: {self.duplicate_state.get('message')}")

        if not messages:
            return True

        return messagebox.askyesno(
            "Human Decision Required",
            "\n".join(messages)
            + "\n\nReject the photo if it needs retaking. Rename anyway?",
        )

    def _rename_and_next(self):
        """Rename current file and move to next"""
        if not self.photo_files or self.current_index >= len(self.photo_files):
            return

        hole_id = self.hole_id_var.get().strip()
        to_str = self.to_var.get().strip()

        if not hole_id:
            messagebox.showwarning("Missing Hole ID", "Enter a Hole ID before renaming.")
            self.hole_id_entry.focus_set()
            return

        # Validate hole ID format
        id_valid, id_error = self._validate_hole_id(hole_id)
        if not id_valid:
            messagebox.showwarning("Invalid Hole ID", id_error)
            self.hole_id_entry.focus_set()
            return

        if not to_str:
            messagebox.showwarning("Missing Depth", "Enter a To Depth before renaming.")
            self.to_entry.focus_set()
            return

        try:
            to_depth = float(to_str)
            from_depth = to_depth - self.interval_length

            # --- Depth range confirmation ---
            if self.validator.depth_ranges:
                is_valid, val_message = self.validator.validate_depth(
                    hole_id, from_depth, to_depth
                )
                if not is_valid:
                    proceed = messagebox.askyesno(
                        "Depth Outside Known Range",
                        f"{val_message}\n\n"
                        f"Rename as {hole_id} {from_depth:.0f}-{to_depth:.0f}m anyway?",
                    )
                    if not proceed:
                        return

            if not self._confirm_risky_rename():
                return

            current_file = self.photo_files[self.current_index]
            extension = current_file.suffix

            # Create new filename: HoleID_FromDepth-ToDepthm.ext
            from_str_fmt = (
                f"{int(from_depth)}"
                if from_depth == int(from_depth)
                else f"{from_depth:.1f}"
            )
            to_str_fmt = (
                f"{int(to_depth)}" if to_depth == int(to_depth) else f"{to_depth:.1f}"
            )
            new_name = f"{hole_id}_{from_str_fmt}-{to_str_fmt}m{extension}"
            new_path = current_file.parent / new_name

            # Check if file already exists
            if new_path.exists() and new_path != current_file:
                if not messagebox.askyesno(
                    "File Exists", f"File {new_name} already exists. Overwrite?"
                ):
                    return

            # Rename file
            current_file.rename(new_path)

            # Update list
            self.photo_files[self.current_index] = new_path

            # Store last values
            self.last_hole_id = hole_id
            self.last_to = to_str

            # Move to next
            self._next_photo()

            # Auto-increment for next photo
            self._auto_increment_next()

        except ValueError:
            messagebox.showerror("Invalid Input", "To depth must be a valid number")
        except Exception as e:
            messagebox.showerror("Rename Error", f"Failed to rename file:\n{str(e)}")

    def _reject_and_next(self):
        """Move current photo to Rejected folder and advance"""
        if not self.photo_files or self.current_index >= len(self.photo_files):
            return

        current_file = self.photo_files[self.current_index]
        if not messagebox.askyesno(
            "Reject Photo?",
            f"Move this photo to the Rejected folder?\n\n{current_file.name}",
        ):
            return

        # Create Rejected subfolder in the photo folder
        if self.rejected_folder is None and self.photo_folder:
            self.rejected_folder = self.photo_folder / "Rejected"
        if self.rejected_folder is None:
            self.rejected_folder = current_file.parent / "Rejected"

        try:
            self.rejected_folder.mkdir(exist_ok=True)
            dest = self.rejected_folder / current_file.name

            # Handle name collision in rejected folder
            if dest.exists():
                stem = current_file.stem
                suffix = current_file.suffix
                counter = 1
                while dest.exists():
                    dest = self.rejected_folder / f"{stem}_{counter}{suffix}"
                    counter += 1

            current_file.rename(dest)

            # Track rejection
            from datetime import datetime
            self.rejected_files.append({
                "original_name": current_file.name,
                "rejected_name": dest.name,
                "source_folder": str(current_file.parent),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            self.rejected_exported = False
            self._update_rejected_count()

            # Remove from file list and stay at same index (which now points to the next file)
            self.photo_files.pop(self.current_index)

            if not self.photo_files:
                self.canvas.delete("all")
                self.current_filename_label.config(text="")
                self._handle_all_processed()
                return

            # Clamp index if we popped the last item
            if self.current_index >= len(self.photo_files):
                self.current_index = len(self.photo_files) - 1

            # Clear fields — next photo may be a different tray entirely
            self.hole_id_var.set("")
            self.to_var.set("20")

            self._load_current_photo()
            self.hole_id_entry.focus_set()

        except Exception as e:
            messagebox.showerror("Reject Error", f"Failed to move file to Rejected:\n{str(e)}")

    def _update_rejected_count(self):
        """Update the rejected count label"""
        count = len(self.rejected_files)
        if count > 0:
            self.rejected_count_label.config(text=f"({count} rejected)")
            self.export_rejected_btn.set_state("normal")
        else:
            self.rejected_count_label.config(text="")
            self.export_rejected_btn.set_state("disabled")

    def _export_rejected_list(self):
        """Export list of rejected photos as CSV for retake requests"""
        if not self.rejected_files:
            messagebox.showinfo("No Rejections", "No photos have been rejected yet.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Export Rejected Photo List",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="rejected_photos.csv",
        )
        if not save_path:
            return

        try:
            with open(save_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["original_name", "rejected_name", "source_folder", "timestamp"],
                )
                writer.writeheader()
                writer.writerows(self.rejected_files)

            messagebox.showinfo(
                "Export Complete",
                f"Rejected list saved:\n{save_path}\n\n{len(self.rejected_files)} photo(s) listed for retake.",
            )
            self.rejected_exported = True
            self._stop_export_pulse()
            if self.processing_finished:
                self._close_after_completion()
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export:\n{str(e)}")

    def _stop_export_pulse(self):
        """Stop pulsing the export button."""
        if self._pulse_job is not None:
            try:
                self.root.after_cancel(self._pulse_job)
            except Exception:
                pass
            self._pulse_job = None
        if hasattr(self, "export_rejected_btn"):
            self.export_rejected_btn.configure_color("#8B6914")

    def _pulse_export_button(self):
        """Pulse the export button while rejected photos need exporting."""
        if not self.rejected_files or self.rejected_exported:
            self._stop_export_pulse()
            return

        current = getattr(self.export_rejected_btn, "base_color", "#8B6914")
        next_color = (
            self.theme_colors.get("accent_yellow", "#f0b429")
            if current == "#8B6914"
            else "#8B6914"
        )
        self.export_rejected_btn.configure_color(next_color)
        self._pulse_job = self.root.after(550, self._pulse_export_button)

    def _close_after_completion(self):
        """Close the renamer after the final required action is complete."""
        self._stop_export_pulse()
        self.root.after(350, self.root.destroy)

    def _handle_all_processed(self):
        """Handle completion according to rejected-photo state."""
        self.processing_finished = True
        self.photo_files = []
        self.current_index = 0
        self.canvas.delete("all")
        self.current_filename_label.config(text="")
        self.progress_label.config(text=f"{len(self.rejected_files)} rejected")

        if self.rejected_files and not self.rejected_exported:
            self.safety_label.config(
                text="All photos processed. Export the rejected list before closing.",
                bg=self.theme_colors.get("accent_yellow", "#8B6914"),
            )
            self._update_rejected_count()
            self._pulse_export_button()
            return

        self.safety_label.config(
            text="All photos processed. Closing bulk renamer.",
            bg=self.theme_colors.get("accent_valid", "#1f3d2a"),
        )
        self._close_after_completion()

    def _next_photo(self):
        """Move to next photo"""
        if self.current_index < len(self.photo_files) - 1:
            self.current_index += 1
            self._load_current_photo()
        else:
            self._handle_all_processed()

    def _previous_photo(self):
        """Move to previous photo"""
        if self.current_index > 0:
            self.current_index -= 1
            self._load_current_photo()

    def _update_progress(self):
        """Update progress label"""
        if self.photo_files:
            self.progress_label.config(
                text=f"{self.current_index + 1} / {len(self.photo_files)}"
            )
        else:
            self.progress_label.config(text="0 / 0")


def main():
    """Main entry point"""
    root = tk.Tk()

    # Note: When used standalone, no gui_manager is available
    # Theming will use fallback colors
    app = PhotoRenamerGUI(root, gui_manager=None)
    root.mainloop()


if __name__ == "__main__":
    main()
