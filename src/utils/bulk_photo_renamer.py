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
from PIL import Image, ImageTk, ImageOps, ImageDraw
import numpy as np
import cv2
import os
from gui.widgets.modern_button import ModernButton
from gui.widgets.entry_with_validation import create_entry_with_validation


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
            r"([A-Z]{2,4}\d+)_CC_(\d+)(?:_temp|_new|_review|_pending_\d+|_Wet|_Dry)?\.(?:png|tiff|jpg)$",
            re.IGNORECASE
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
                    if match and match.group(1).upper() == hole_id_upper:
                        found_files.append(file_path)
                        if "_Wet" in file_path.name:
                            wet_files.append(file_path)
                        elif "_Dry" in file_path.name:
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

    def __init__(self, root, gui_manager=None):
        self.root = root
        self.gui_manager = gui_manager

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
        
        # Config from gui_manager
        self.config = {}
        if self.gui_manager and hasattr(self.gui_manager, "config"):
            self.config = self.gui_manager.config
        
        # Existing compartment checker
        self.existing_checker = ExistingCompartmentChecker()
        self._setup_output_folders()
        
        # Processing preview
        self.preview_cache = ProcessingPreviewCache()
        self.preview_enabled = tk.BooleanVar(value=False)
        self.current_preview_data: Optional[Dict] = None
        self._aruco_manager = None
        self._viz_manager = None

        self._build_gui()

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
        
        self.preview_status_label = tk.Label(
            status_row, text="", bg=self.theme_colors["background"],
            fg="#999999", font=self.fonts["small"],
        )
        self.preview_status_label.pack(side=tk.RIGHT, padx=10)

        # Action buttons
        button_frame = tk.Frame(control_frame, bg=self.theme_colors["background"])
        button_frame.pack(pady=10)

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
        self.rejected_count_label = tk.Label(
            button_frame,
            text="",
            bg=self.theme_colors["background"],
            fg=self.theme_colors["accent_red"],
            font=self.fonts["small"],
        )
        self.rejected_count_label.pack(side=tk.LEFT, padx=5)

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
                if hasattr(DialogHelper, "show_message") and self.gui_manager:
                    DialogHelper.show_message(
                        self.root,
                        "Success",
                        f"Loaded validation data for {len(self.validator.validation_data)} holes",
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
                    self.logger.warning("BulkRenamer: Snowflake depth data not available")

                self._check_validation()

            self.root.after(0, _apply)

        threading.Thread(target=_run, daemon=True, name="sf-bulk-depth").start()

    def _select_folder(self):
        """Select folder containing photos"""
        folder = filedialog.askdirectory(title="Select Photo Folder")

        if folder:
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
            
            corrected_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            result["corrected_image"] = Image.fromarray(corrected_rgb)
            
            self.preview_cache.set(str(image_path), mtime, result)
            return result
            
        except Exception as e:
            self.logger.error(f"Preview generation failed: {e}")
            return None
    
    def _draw_preview_overlay(self, pil_img: Image.Image, preview_data: Dict, display_scale: float) -> Image.Image:
        """Draw boundary boxes, markers, and vertical constraints on the image"""
        img = pil_img.copy()
        draw = ImageDraw.Draw(img)
        
        # Draw vertical constraint lines (top/bottom of tray area)
        vert = preview_data.get("vertical_constraints")
        if vert:
            top_y, bottom_y = vert
            img_width = img.size[0]
            scaled_top = int(top_y * display_scale)
            scaled_bottom = int(bottom_y * display_scale)
            draw.line([(0, scaled_top), (img_width, scaled_top)], fill=(255, 100, 100), width=2)
            draw.line([(0, scaled_bottom), (img_width, scaled_bottom)], fill=(255, 100, 100), width=2)
        
        # Draw boundaries (green boxes)
        for x1, y1, x2, y2 in preview_data.get("boundaries", []):
            sx1, sy1 = int(x1 * display_scale), int(y1 * display_scale)
            sx2, sy2 = int(x2 * display_scale), int(y2 * display_scale)
            draw.rectangle([sx1, sy1, sx2, sy2], outline=(0, 255, 0), width=2)
        
        # Draw markers
        for marker_id, corners in preview_data.get("markers", {}).items():
            cx = int(np.mean(corners[:, 0]) * display_scale)
            cy = int(np.mean(corners[:, 1]) * display_scale)
            
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
            return
        
        parts = []
        comp_count = preview_data.get("compartment_marker_count", 0)
        expected = self.config.get("compartment_count", 20)
        
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
    
    def _check_existing_compartments(self):
        """Check if compartments already exist for current hole/depth"""
        hole_id = self.hole_id_var.get().strip()
        to_str = self.to_var.get().strip()
        
        if not hole_id or not to_str:
            self.existing_label.config(text="")
            return
        
        try:
            to_depth = int(float(to_str))
            from_depth = to_depth - self.interval_length
            result = self.existing_checker.check_existing(hole_id, from_depth, to_depth)
            
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
                    original_size = img.size
                    img.thumbnail((canvas_width - 20, canvas_height - 20), Image.Resampling.LANCZOS)
                    display_scale = img.size[0] / original_size[0]
                    img = self._draw_preview_overlay(img, preview_data, display_scale)
                    self._update_preview_status(preview_data)
                else:
                    img = Image.open(current_file)
                    img = ImageOps.exif_transpose(img)
                    img.thumbnail((canvas_width - 20, canvas_height - 20), Image.Resampling.LANCZOS)
                    self._update_preview_status(None)
                
                self.root.config(cursor="")
            else:
                # Fast display without preview
                img = Image.open(current_file)
                img = ImageOps.exif_transpose(img)
                img.thumbnail((canvas_width - 20, canvas_height - 20), Image.Resampling.LANCZOS)
                self.current_preview_data = None
                self.preview_status_label.config(text="")

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
            except ValueError:
                self.from_display_label.config(
                    text="(Invalid)", foreground=self.theme_colors["accent_red"]
                )
        else:
            self.from_display_label.config(text="")

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
            self._update_rejected_count()

            # Remove from file list and stay at same index (which now points to the next file)
            self.photo_files.pop(self.current_index)

            if not self.photo_files:
                self.canvas.delete("all")
                self.current_filename_label.config(text="")
                messagebox.showinfo("Complete", "All photos processed!")
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
        else:
            self.rejected_count_label.config(text="")

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
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export:\n{str(e)}")

    def _next_photo(self):
        """Move to next photo"""
        if self.current_index < len(self.photo_files) - 1:
            self.current_index += 1
            self._load_current_photo()
        else:
            messagebox.showinfo("Complete", "All photos processed!")

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
