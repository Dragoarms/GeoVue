# gui/main_gui.py
import os
import re
import cv2
import numpy as np
from datetime import datetime
import logging
import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path
import queue
import shutil
import platform
import subprocess
import time
from typing import Dict, List, Optional, Any, Tuple, Union
import traceback

# Bulk Renamer integration (PEP-8: imports near top; safe optional import)
try:
    from utils.bulk_photo_renamer import (
        PhotoRenamerGUI,
    )  # assuming module on PYTHONPATH
except Exception:
    PhotoRenamerGUI = None

# Use DialogHelper for consistent dialogs
from gui.dialog_helper import DialogHelper

# from processing.drillhole_trace_generator import DrillholeTraceGenerator
from gui.widgets import *

# Note: batch_processor is imported dynamically in _start_auto_batch_processing

# from processing.QAQCStep.qaqc_manager import QAQCManager
from gui.logging_review_dialog import LoggingReviewDialog
from gui.embedding_training_dialog import EmbeddingTrainingDialog
from gui.DrillholeCorrelation.drillhole_selection_dialog import DrillholeSelectionDialog
from gui.DrillholeCorrelation.correlation_dialog import CorrelationDialog
import os
from utils.json_register_manager import JSONRegisterManager
from utils.register_synchronizer import RegisterSynchronizer
from utils.file_deduplication_manager import UIDDeduplicationManager
from gui.progress_dialog import ProgressDialog
from utils.image_processing_depth_validation import DepthValidator


import threading

if threading.current_thread() != threading.main_thread():
    raise RuntimeError("❌ Main GUI called from a background thread!")


logger = logging.getLogger(__name__)


class MainGUI:
    def __init__(self, app):
        """Initialize with reference to main application."""
        self.app = app  # Reference to main app for accessing components
        self.root = app.root  # Use the existing root window
        self.root.app = app  # Also store on root for dialog access
        self.progress_queue = queue.Queue()
        self.processing_complete = False
        self.active_threads = []  # Track active threads
        self.after_id = None  # Initialize after_id to None

        # Add a logger instance
        self.logger = logging.getLogger(__name__)

        # Add custom handler to route cloud sync logs to status box
        self._setup_sync_logger()
        self.root = app.root  # Use the existing root window
        self.progress_queue = queue.Queue()
        self.processing_complete = False
        self.active_threads = []  # Track active threads
        self.after_id = None  # Initialize after_id to None

        # Get references to app components for easier access
        self.file_manager = app.file_manager
        self.gui_manager = app.gui_manager
        self.config_manager = (
            app.config_manager
        )  # ADD THIS LINE - store the actual ConfigManager instance
        self.translator = app.translator
        self.blur_detector = app.blur_detector
        self.aruco_manager = app.aruco_manager
        self.qaqc_manager = app.qaqc_manager
        self.depth_validator = app.depth_validator
        self.trace_generator = app.trace_generator  # Yes, get this from app too
        self.drillhole_data_manager = app.drillhole_data_manager  # For collar data

        # Set a flag to check if processing is active
        self.is_processing_image = False

        # Initialize language variable
        self.language_var = tk.StringVar(value=self.translator.get_current_language())

        # Setup configuration
        self.config = app.config_manager.as_dict()

        # Create GUI
        self.create_gui()

        # Close any splash screen from launcher
        self._close_launcher_splash()

        # NOW you can update the status after GUI is created
        self._update_depth_csv_status()

        # Set up the shared paths if available
        self.setup_shared_paths()

        # Initialize visualization cache for dialogs
        self.visualization_cache = {}

        # Delay the first check_progress call
        self.root.after(1000, self.check_progress)

    def _close_launcher_splash(self):
        """Close any splash screen from launcher."""
        try:
            # Find and destroy any overridden toplevel windows (splash screens)
            for widget in self.root.winfo_children():
                if (
                    isinstance(widget, tk.Toplevel)
                    and widget.winfo_exists()
                    and hasattr(widget, "overrideredirect")
                    and widget.overrideredirect()
                ):
                    widget.destroy()
                    self.logger.debug("Closed launcher splash screen")
        except Exception as e:
            self.logger.debug(f"Splash cleanup: {e}")

    def t(self, text):
        """Translate text using the translator."""
        return self.translator.translate(text)

    def create_gui(self):
        """Create a GUI for chip tray extraction using the GUIManager."""

        # Initialize language variable
        self.language_var = tk.StringVar(value=self.translator.get_current_language())

        # Initialize window layout with GUIManager
        window = self.gui_manager.create_main_window(
            self.root,
            self.t("GeoVue"),
            # width=1200,
            # height=900
        )

        # Store window components
        self.main_container = window["main_container"]
        self.header_frame = window["header_frame"]
        self.title_label = window["title_label"]
        self.content_outer_frame = window["content_outer_frame"]
        self.canvas = window["canvas"]
        self.content_frame = window["content_frame"]
        self.footer_frame = window["footer_frame"]

        # Hide the default title label from create_main_window
        self.title_label.pack_forget()

        # Create centered container for logo and title (like first run dialog)
        center_container = ttk.Frame(self.header_frame, style="Header.TFrame")
        center_container.place(relx=0.5, rely=0.5, anchor="center")

        # Add logo and title side by side
        try:
            import PIL.Image
            import PIL.ImageTk

            # Get logo path
            logo_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "resources", "full_logo.png"
            )

            if os.path.exists(logo_path):
                # Load and resize logo to larger size
                logo_image = PIL.Image.open(logo_path)

                # Resize to larger height (e.g., 80 pixels for better visibility)
                logo_height = 80
                aspect_ratio = logo_image.width / logo_image.height
                logo_width = int(logo_height * aspect_ratio)
                logo_image = logo_image.resize(
                    (logo_width, logo_height), PIL.Image.Resampling.LANCZOS
                )

                # Convert to PhotoImage
                self.logo_photo = PIL.ImageTk.PhotoImage(logo_image)

                # Create label for logo
                logo_label = tk.Label(
                    center_container,
                    image=self.logo_photo,
                    bg=self.gui_manager.theme_colors["secondary_bg"],
                )
                logo_label.pack(side=tk.LEFT, padx=(0, 20))

                # Add GeoVue title next to logo with refined typography
                title_label = tk.Label(
                    center_container,
                    text="GeoVue",
                    font=("Segoe UI Light", 28),
                    bg=self.gui_manager.theme_colors["secondary_bg"],
                    fg=self.gui_manager.theme_colors["text"],
                )
                title_label.pack(side=tk.LEFT)

                self.logger.debug(f"Logo and title added to main GUI")

        except Exception as e:
            self.logger.debug(f"Could not load logo: {e}")
            # If logo fails, just show the title
            title_label = tk.Label(
                center_container,
                text="GeoVue",
                font=("Segoe UI Light", 28),
                bg=self.gui_manager.theme_colors["secondary_bg"],
                fg=self.gui_manager.theme_colors["text"],
            )
            title_label.pack()

        # Set minimum height for header to prevent overlap
        self.header_frame.configure(height=100)
        self.header_frame.pack_propagate(False)

        # =====================================================
        # IMAGE PROCESSING - Collapsible Frame
        # =====================================================
        processing_collapsible = self.gui_manager.create_collapsible_frame(
            self.content_frame,
            title=self.t("Image Processing"),
            expanded=False,
        )
        self.processing_collapsible = processing_collapsible

        # Input folder field
        input_frame = ttk.Frame(processing_collapsible.content_frame, style="Content.TFrame")
        input_frame.pack(fill=tk.X, pady=(0, 5))

        self.folder_var = tk.StringVar()
        folder_frame, self.folder_entry = self.gui_manager.create_field_with_label(
            input_frame,
            self.t("Input Folder:"),
            self.folder_var,
            field_type="entry",
            validate_func=self._update_input_folder_color,
            width=None,
        )

        # Browse button in folder_frame
        browse_button = self.gui_manager.create_modern_button(
            folder_frame,
            text=self.t("Browse"),
            color="#5aa06c",
            command=self.browse_input_folder,
        )
        browse_button.pack(side=tk.RIGHT, padx=(5, 0))

        # Output settings
        output_frame = ttk.Frame(processing_collapsible.content_frame, style="Content.TFrame")
        output_frame.pack(fill=tk.X, pady=2)

        # Local output
        self.output_folder_var = tk.StringVar(
            value=self.file_manager.dir_structure["local_output_folder"]
        )
        output_folder_frame, output_entry = self.gui_manager.create_field_with_label(
            output_frame,
            self.t("Local Output:"),
            self.output_folder_var,
            readonly=True,
            width=None,
        )

        # Info button
        info_button = self.gui_manager.create_modern_button(
            output_folder_frame,
            text="[?]",
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._show_file_structure_info,
        )
        info_button.pack(side=tk.RIGHT, padx=(5, 0))

        # Output format
        format_frame = ttk.Frame(processing_collapsible.content_frame, style="Content.TFrame")
        format_frame.pack(fill=tk.X, pady=2)

        format_label = ttk.Label(
            format_frame,
            text=self.t("Output Format:"),
            anchor="w",
            style="Content.TLabel",
        )
        format_label.pack(side=tk.LEFT)

        format_combo_frame = tk.Frame(
            format_frame,
            bg=self.gui_manager.theme_colors["field_bg"],
            highlightbackground=self.gui_manager.theme_colors["field_border"],
            highlightthickness=1,
            bd=0,
        )
        format_combo_frame.pack(side=tk.LEFT, padx=(5, 0))

        self.format_var = tk.StringVar(value=self.config["output_format"])
        format_options = ["tiff", "png"]

        format_dropdown = tk.OptionMenu(
            format_combo_frame, self.format_var, *format_options
        )
        self.gui_manager.style_dropdown(format_dropdown, width=6)
        format_dropdown.pack()

        # Debug images checkbox
        debug_frame = ttk.Frame(processing_collapsible.content_frame, style="Content.TFrame")
        debug_frame.pack(fill=tk.X, pady=2)

        self.debug_var = tk.BooleanVar(value=self.config["save_debug_images"])

        debug_check = self.gui_manager.create_custom_checkbox(
            debug_frame, text=self.t("Save Debug Images"), variable=self.debug_var
        )
        debug_check.pack(anchor="w")

        # Auto-loop batch processing checkbox
        auto_loop_frame = ttk.Frame(processing_collapsible.content_frame, style="Content.TFrame")
        auto_loop_frame.pack(fill=tk.X, pady=2)

        self.auto_loop_var = tk.BooleanVar(value=False)

        auto_loop_check = self.gui_manager.create_custom_checkbox(
            auto_loop_frame,
            text=self.t("Auto-Loop Mode (Skip manual steps)"),
            variable=self.auto_loop_var,
            command=self._toggle_auto_loop_mode,
        )
        auto_loop_check.pack(anchor="w")

        # Add info label for auto-loop mode
        auto_info_label = ttk.Label(
            auto_loop_frame,
            text=self.t(
                "   Automatically processes images with valid filename metadata"
            ),
            font=("Segoe UI", 8),
            foreground=self.gui_manager.theme_colors["field_border"],
            style="Content.TLabel",
        )
        auto_info_label.pack(anchor="w", padx=(20, 0))

        # Auto-loop duplicate handling option
        self.auto_skip_duplicates_var = tk.BooleanVar(value=False)

        auto_skip_dup_check = self.gui_manager.create_custom_checkbox(
            auto_loop_frame,
            text=self.t("   Skip existing/duplicate intervals"),
            variable=self.auto_skip_duplicates_var,
        )
        auto_skip_dup_check.pack(anchor="w", padx=(20, 0))

        skip_dup_info_label = ttk.Label(
            auto_loop_frame,
            text=self.t(
                "      Uncheck to extract duplicates (e.g., wet/dry pairs) for manual selection"
            ),
            font=("Segoe UI", 8),
            foreground=self.gui_manager.theme_colors["field_border"],
            style="Content.TLabel",
        )
        skip_dup_info_label.pack(anchor="w", padx=(40, 0))

        # Processing buttons row
        processing_buttons_frame = ttk.Frame(
            processing_collapsible.content_frame, style="Content.TFrame"
        )
        processing_buttons_frame.pack(fill=tk.X, pady=(10, 5))

        # Process Photos button (primary action)
        self.process_button = self.gui_manager.create_modern_button(
            processing_buttons_frame,
            text=self.t("Process Photos"),
            color=self.gui_manager.theme_colors["accent_green"],
            command=self.start_processing,
            icon="▶",
        )
        self.process_button.pack(side=tk.LEFT, padx=(0, 5))

        # Bulk Renamer button
        bulk_renamer_button = self.gui_manager.create_modern_button(
            processing_buttons_frame,
            text=self.t("Bulk Renamer"),
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._open_bulk_renamer,
            icon="✏",
        )
        bulk_renamer_button.pack(side=tk.LEFT, padx=5)

        # Review Extracted Images button
        review_button = self.gui_manager.create_modern_button(
            processing_buttons_frame,
            text=self.t("Review Extracted Images"),
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._start_image_review,
            icon="🔍",
        )
        review_button.pack(side=tk.LEFT, padx=5)

        # Calculate Image Properties button
        validate_button = self.gui_manager.create_modern_button(
            processing_buttons_frame,
            text=self.t("Calculate Image Properties"),
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._generate_image_properties,
            icon="🎨",
        )
        validate_button.pack(side=tk.LEFT, padx=5)

        # Sync to Cloud button
        sync_button = self.gui_manager.create_modern_button(
            processing_buttons_frame,
            text=self.t("Sync to Cloud"),
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._sync_to_cloud,
            icon="☁",
        )
        sync_button.pack(side=tk.LEFT, padx=5)

        # Create collapsible sections with the GUIManager
        # Shared Folder Path Settings

        # Check if paths exist using FileManager
        approved_path_exists = bool(
            self.file_manager.get_shared_path(
                "approved_compartments", create_if_missing=False
            )
        )
        processed_originals_exists = bool(
            self.file_manager.get_shared_path(
                "processed_originals", create_if_missing=False
            )
        )
        drill_traces_exists = bool(
            self.file_manager.get_shared_path("drill_traces", create_if_missing=False)
        )
        register_path_exists = bool(
            self.file_manager.get_shared_path("register_excel", create_if_missing=False)
        )
        # Check for datasets folder (contains CSV files for data manager)
        datasets_dir = self.file_manager.get_shared_path("datasets", create_if_missing=False)
        datasets_folder_exists = bool(datasets_dir and datasets_dir.exists())

        # Expand if any paths are missing
        should_expand = not (
            approved_path_exists
            and processed_originals_exists
            and drill_traces_exists
            and register_path_exists
            and datasets_folder_exists
        )

        shared_collapsible = self.gui_manager.create_collapsible_frame(
            self.content_frame,
            title=self.t("Shared Folder Settings"),
            expanded=should_expand,
        )
        self.shared_collapsible = shared_collapsible
        # Create variables for shared paths
        self.approved_path_var = tk.StringVar()
        self.processed_originals_path_var = tk.StringVar()
        self.drill_traces_path_var = tk.StringVar()
        self.register_path_var = tk.StringVar()
        self.datasets_folder_var = tk.StringVar()

        # Set display values using FileManager's paths or config
        config = self.app.config_manager.as_dict()

        # Approved folder
        approved_path = self.file_manager.get_shared_path(
            "approved_compartments", create_if_missing=False
        )
        self.approved_path_var.set(
            str(approved_path)
            if approved_path
            else config.get("shared_folder_approved_compartments_folder", "")
        )

        # Processed originals
        processed_path = self.file_manager.get_shared_path(
            "processed_originals", create_if_missing=False
        )
        self.processed_originals_path_var.set(
            str(processed_path)
            if processed_path
            else config.get("shared_folder_processed_originals", "")
        )

        # Drill traces
        drill_traces_path = self.file_manager.get_shared_path(
            "drill_traces", create_if_missing=False
        )
        self.drill_traces_path_var.set(
            str(drill_traces_path)
            if drill_traces_path
            else config.get("shared_folder_drill_traces", "")
        )

        # Register
        register_path = self.file_manager.get_shared_path(
            "register_excel", create_if_missing=False
        )
        self.register_path_var.set(
            str(register_path)
            if register_path
            else config.get("shared_folder_register_excel_path", "")
        )

        # Drillhole Datasets folder (contains CSV files loaded by data manager)
        datasets_path = self.file_manager.get_shared_path(
            "datasets", create_if_missing=False
        )

        self.datasets_folder_var.set(
            str(datasets_path)
            if datasets_path
            else config.get("shared_folder_datasets", "")
        )

        # Create path input fields
        self._create_shared_path_field(
            shared_collapsible.content_frame,
            self.t("Approved Folder:"),
            self.approved_path_var,
            approved_path_exists,
            is_file=False,
            path_key="approved_compartments",
        )
        self._create_shared_path_field(
            shared_collapsible.content_frame,
            self.t("Processed Originals Folder:"),
            self.processed_originals_path_var,
            processed_originals_exists,
            is_file=False,
            path_key="processed_originals",
        )
        self._create_shared_path_field(
            shared_collapsible.content_frame,
            self.t("Drill Traces Folder:"),
            self.drill_traces_path_var,
            drill_traces_exists,
            is_file=False,
            path_key="drill_traces",
        )
        self._create_shared_path_field(
            shared_collapsible.content_frame,
            self.t("Excel Register:"),
            self.register_path_var,
            register_path_exists,
            is_file=True,  # This is a file, not a folder
            path_key="register_excel",
        )
        self._create_shared_path_field(
            shared_collapsible.content_frame,
            self.t("Drillhole Datasets Folder:"),
            self.datasets_folder_var,
            datasets_folder_exists,
            is_file=False,  # This is a folder containing CSV files
            path_key="datasets",
        )
        # Data Settings button (replaces obsolete Create New Register)
        button_container = ttk.Frame(
            shared_collapsible.content_frame, style="Content.TFrame"
        )
        button_container.pack(fill=tk.X, pady=(5, 0))

        data_settings_button = self.gui_manager.create_modern_button(
            button_container,
            text=self.t("Data Settings"),
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._open_data_settings,
        )
        data_settings_button.pack(side=tk.RIGHT)

        # =====================================================
        # IMAGE ANALYSIS AND REVIEW - Collapsible Frame
        # =====================================================
        analysis_collapsible = self.gui_manager.create_collapsible_frame(
            self.content_frame,
            title=self.t("Image Analysis and Review"),
            expanded=False,
        )
        self.analysis_collapsible = analysis_collapsible

        # Analysis buttons row
        analysis_buttons_frame = ttk.Frame(
            analysis_collapsible.content_frame, style="Content.TFrame"
        )
        analysis_buttons_frame.pack(fill=tk.X, pady=5)

        # Review All Images button (prominent - renamed from Logging Review)
        review_all_button = self.gui_manager.create_modern_button(
            analysis_buttons_frame,
            text=self.t("Review All Images"),
            color=self.gui_manager.theme_colors["accent_green"],
            command=self._start_logging_review,
            icon="📋",
        )
        review_all_button.pack(side=tk.LEFT, padx=(0, 5))

        # Original Image Viewer button
        original_viewer_button = self.gui_manager.create_modern_button(
            analysis_buttons_frame,
            text=self.t("Original Image Viewer"),
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._open_original_image_viewer,
            icon="🖼",
        )
        original_viewer_button.pack(side=tk.LEFT, padx=5)

        # Drillhole Correlation button
        correlation_button = self.gui_manager.create_modern_button(
            analysis_buttons_frame,
            text=self.t("Drillhole Correlation"),
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._open_drillhole_correlation,
            icon="📊",
        )
        correlation_button.pack(side=tk.LEFT, padx=5)

        report_button = self.gui_manager.create_modern_button(
            analysis_buttons_frame,
            text=self.t("Logging Review Report"),
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._open_logging_review_report,
            icon="📄",
        )
        report_button.pack(side=tk.LEFT, padx=5)

        # Initialize default values for removed settings (for backwards compatibility)
        self.blur_enable_var = tk.BooleanVar(value=self.config.get("enable_blur_detection", False))
        self.blur_threshold_var = tk.DoubleVar(value=self.config.get("blur_threshold", 100.0))
        self.blur_roi_var = tk.DoubleVar(value=self.config.get("blur_roi_ratio", 0.5))
        self.flag_blurry_var = tk.BooleanVar(value=self.config.get("flag_blurry_images", False))
        self.save_blur_viz_var = tk.BooleanVar(value=self.config.get("save_blur_visualizations", False))
        self.blur_threshold_pct_var = tk.DoubleVar(value=self.config.get("blurry_threshold_percentage", 25.0))
        self.ocr_enable_var = tk.BooleanVar(value=self.config.get("enable_ocr", False))
        self.prefix_validation_var = tk.BooleanVar(value=self.config.get("enable_prefix_validation", True))
        prefix_str = ", ".join(self.config.get("valid_hole_prefixes", ["BA", "NB", "SB", "KM"]))
        self.prefix_var = tk.StringVar(value=prefix_str)

        # Create status section (no progress bar)
        status_components = self.gui_manager.create_status_section(self.content_frame)
        self.progress_var = status_components["progress_var"]
        self.progress_bar = status_components["progress_bar"]
        self.status_text = status_components["status_text"]

        # Create footer buttons
        button_configs = [
            {
                "name": "start_processing_button",
                "text": self.t("Start Image Processing"),
                "color": self.gui_manager.theme_colors["accent_green"],
                "command": self._open_processing_section,
                "icon": "▶",
            },
            {
                "name": "start_analysis_button",
                "text": self.t("Start Image Analysis"),
                "color": self.gui_manager.theme_colors["accent_blue"],
                "command": self._open_analysis_section,
                "icon": "📊",
            },
            {
                "name": "quit_button",
                "text": self.t("Quit"),
                "color": self.gui_manager.theme_colors["accent_red"],
                "command": self.quit_app,
                "icon": "✖",
            },
        ]

        self.button_row_frame, self.buttons = self.gui_manager.create_button_row(
            self.footer_frame,
            button_configs,
            side="bottom",
            anchor="se",
            padx=10,
            pady=10,
        )

        # Store button references for backwards compatibility
        self.buttons["process_button"] = self.process_button

        # Set up toggle callbacks for collapsible frames to update footer buttons
        self.processing_collapsible.set_on_toggle(self._on_collapsible_toggle)
        self.shared_collapsible.set_on_toggle(self._on_collapsible_toggle)
        self.analysis_collapsible.set_on_toggle(self._on_collapsible_toggle)

        # Create menu definitions
        menu_defs = {
            self.t("File"): [
                {
                    "type": "command",
                    "label": self.gui_manager.get_theme_label(),
                    "command": self._toggle_theme,
                },
                {"type": "separator"},
                {"type": "command", "label": self.t("Exit"), "command": self.quit_app},
            ],
            self.t("Settings"): [
                {
                    "type": "command",
                    "label": self.t("Data Settings"),
                    "command": self._open_data_settings,
                },
                {
                    "type": "command",
                    "label": self.t("Color Maps"),
                    "command": self._open_color_map_settings,
                },
                {"type": "separator"},
                {
                    "type": "command",
                    "label": self.t("Classification Manager"),
                    "command": self._open_classification_manager,
                },
                {
                    "type": "command",
                    "label": self.t("Tag Manager"),
                    "command": self._open_tag_manager,
                },
            ],
            self.t("Help"): [
                {
                    "type": "command",
                    "label": self.t("Check for Updates"),
                    "command": self.on_check_for_updates,
                },
                {
                    "type": "command",
                    "label": self.t("About"),
                    "command": self._show_about_dialog,
                },
            ],
        }

        # Add language options to menu definitions
        languages = self.translator.get_available_languages()
        language_items = []
        for lang_code in languages:
            lang_name = self.translator.get_language_name(lang_code)
            language_items.append(
                {
                    "type": "radiobutton",
                    "label": lang_name,
                    "value": lang_code,
                    "variable": self.language_var,
                    "command": lambda lc=lang_code: self.change_language(lc),
                }
            )

        menu_defs[self.t("Language")] = language_items

        # Create menubar
        self.menubar = self.gui_manager.setup_menubar(self.root, menu_defs)

        # Size and center the main window after everything is created
        def size_and_center_window():
            # First, temporarily set a large size so everything can lay out
            self.root.geometry("1200x800")
            self.root.update_idletasks()

            # Now get the actual required sizes
            header_width = self.header_frame.winfo_reqwidth()
            footer_width = self.footer_frame.winfo_reqwidth()
            canvas_width = self.canvas.winfo_reqwidth()

            # Use the widest component
            required_width = max(
                header_width, footer_width, canvas_width, 1000
            )

            window_height = 800

            # Get screen dimensions
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()

            # Apply limits
            window_width = min(required_width + 40, int(screen_width * 0.9))
            window_height = min(window_height, int(screen_height * 0.85))

            # Center
            x = (screen_width - window_width) // 2
            y = (screen_height - window_height) // 2 - 30

            # Apply final geometry
            self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

            # Update scroll region
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        # Use after_idle to ensure all widgets are laid out
        self.root.after_idle(size_and_center_window)

        # Add initial status message
        self.update_status(
            self.t("Ready. Select a folder and click 'Process Photos'."), "info"
        )

        # Check for updates at startup if enabled
        if self.config.get("check_for_updates", True):
            self.root.after(2000, self._check_updates_at_startup)

    def _toggle_blur_settings(self):
        """Backwards compatibility stub - blur settings removed from UI."""
        pass

    def _toggle_ocr_settings(self):
        """Backwards compatibility stub - OCR settings removed from UI."""
        pass

    def _toggle_prefix_settings(self):
        """Backwards compatibility stub - prefix settings removed from UI."""
        pass

    def _show_blur_calibration_dialog(self):
        """Backwards compatibility stub - blur calibration removed from UI."""
        DialogHelper.show_message(
            self.root,
            self.t("Feature Removed"),
            self.t("Blur calibration has been removed from the interface."),
            message_type="info"
        )

    def _show_blur_help(self):
        """Backwards compatibility stub - blur help removed from UI."""
        pass

    def _on_collapsible_toggle(self, frame, expanded):
        """Handle collapsible frame toggle events to update footer buttons."""
        # When a frame is expanded, collapse other frames (accordion behavior)
        if expanded:
            collapsibles = [
                self.processing_collapsible,
                self.shared_collapsible,
                self.analysis_collapsible,
            ]
            for collapsible in collapsibles:
                if collapsible != frame and collapsible.expanded:
                    # Temporarily disable callback to avoid recursion
                    old_callback = collapsible.on_toggle
                    collapsible.on_toggle = None
                    collapsible.toggle()
                    collapsible.on_toggle = old_callback

        # Update footer buttons based on current state
        self._update_footer_buttons()

    def _update_footer_buttons(self):
        """Update the footer button bar based on which collapsible frame is open."""
        # Destroy existing button row
        if hasattr(self, 'button_row_frame') and self.button_row_frame:
            self.button_row_frame.destroy()

        # Determine which buttons to show
        accent_green = self.gui_manager.theme_colors["accent_green"]
        accent_blue = self.gui_manager.theme_colors["accent_blue"]
        accent_red = self.gui_manager.theme_colors["accent_red"]

        # Check which frame is expanded
        processing_open = hasattr(self, 'processing_collapsible') and self.processing_collapsible.expanded
        shared_open = hasattr(self, 'shared_collapsible') and self.shared_collapsible.expanded
        analysis_open = hasattr(self, 'analysis_collapsible') and self.analysis_collapsible.expanded

        if processing_open:
            # Show processing buttons
            button_configs = [
                {
                    "name": "process_button",
                    "text": self.t("Process Photos"),
                    "color": accent_green,
                    "command": self.start_processing,
                    "icon": "▶",
                },
                {
                    "name": "bulk_renamer_button",
                    "text": self.t("Bulk Renamer"),
                    "color": accent_blue,
                    "command": self._open_bulk_renamer,
                    "icon": "✏",
                },
                {
                    "name": "review_extracted_button",
                    "text": self.t("Review Extracted"),
                    "color": accent_blue,
                    "command": self._start_image_review,
                    "icon": "🔍",
                },
                {
                    "name": "quit_button",
                    "text": self.t("Quit"),
                    "color": accent_red,
                    "command": self.quit_app,
                    "icon": "✖",
                },
            ]
        elif analysis_open:
            # Show analysis buttons
            button_configs = [
                {
                    "name": "review_all_button",
                    "text": self.t("Review All Images"),
                    "color": accent_green,
                    "command": self._start_logging_review,
                    "icon": "📋",
                },
                {
                    "name": "original_viewer_button",
                    "text": self.t("Original Viewer"),
                    "color": accent_blue,
                    "command": self._open_original_image_viewer,
                    "icon": "🖼",
                },
                {
                    "name": "correlation_button",
                    "text": self.t("Drillhole Correlation"),
                    "color": accent_blue,
                    "command": self._open_drillhole_correlation,
                    "icon": "📊",
                },
                {
                    "name": "quit_button",
                    "text": self.t("Quit"),
                    "color": accent_red,
                    "command": self.quit_app,
                    "icon": "✖",
                },
            ]
        elif shared_open:
            # Show shared settings buttons
            button_configs = [
                {
                    "name": "data_settings_button",
                    "text": self.t("Data Settings"),
                    "color": accent_blue,
                    "command": self._open_data_settings,
                    "icon": "⚙",
                },
                {
                    "name": "quit_button",
                    "text": self.t("Quit"),
                    "color": accent_red,
                    "command": self.quit_app,
                    "icon": "✖",
                },
            ]
        else:
            # Show intro buttons (all frames collapsed)
            button_configs = [
                {
                    "name": "start_processing_button",
                    "text": self.t("Start Image Processing"),
                    "color": accent_green,
                    "command": self._open_processing_section,
                    "icon": "▶",
                },
                {
                    "name": "start_analysis_button",
                    "text": self.t("Start Image Analysis"),
                    "color": accent_blue,
                    "command": self._open_analysis_section,
                    "icon": "📊",
                },
                {
                    "name": "quit_button",
                    "text": self.t("Quit"),
                    "color": accent_red,
                    "command": self.quit_app,
                    "icon": "✖",
                },
            ]

        # Create new button row
        self.button_row_frame, self.buttons = self.gui_manager.create_button_row(
            self.footer_frame,
            button_configs,
            side="bottom",
            anchor="se",
            padx=10,
            pady=10,
        )

        # Store process button reference for backwards compatibility
        if "process_button" in self.buttons:
            pass  # Already stored with correct name
        elif hasattr(self, 'process_button'):
            self.buttons["process_button"] = self.process_button

    def _open_processing_section(self):
        """Expand the Image Processing collapsible frame and scroll to it."""
        if hasattr(self, 'processing_collapsible'):
            # Expand if not already expanded
            if not self.processing_collapsible.expanded:
                self.processing_collapsible.toggle()
            # Scroll to make it visible
            self.root.after(100, lambda: self._scroll_to_widget(self.processing_collapsible))

    def _open_analysis_section(self):
        """Expand the Image Analysis and Review collapsible frame and scroll to it."""
        if hasattr(self, 'analysis_collapsible'):
            # Expand if not already expanded
            if not self.analysis_collapsible.expanded:
                self.analysis_collapsible.toggle()
            # Scroll to make it visible
            self.root.after(100, lambda: self._scroll_to_widget(self.analysis_collapsible))

    def _scroll_to_widget(self, widget):
        """Scroll the canvas to make the given widget visible."""
        try:
            # Update the canvas to ensure geometry is calculated
            self.canvas.update_idletasks()
            # Get widget position relative to canvas
            widget_y = widget.winfo_y()
            canvas_height = self.canvas.winfo_height()
            # Scroll to position
            self.canvas.yview_moveto(widget_y / max(1, self.canvas.bbox("all")[3]))
        except Exception as e:
            self.logger.debug(f"Could not scroll to widget: {e}")

    def _open_data_settings(self):
        """Open the Data Settings dialog for column configuration."""
        try:
            from gui.column_settings_dialog import ColumnSettingsDialog
            
            # Get data coordinator from app
            data_coordinator = getattr(self.app, 'data_coordinator', None)
            
            if not data_coordinator or not data_coordinator.is_initialized:
                from gui.dialog_helper import DialogHelper
                DialogHelper.show_message(
                    self.root,
                    self.t("Data Not Loaded"),
                    self.t("Please wait for data to finish loading, or check that data sources are configured."),
                    message_type="warning"
                )
                return
            
            dialog = ColumnSettingsDialog(
                parent=self.root,
                gui_manager=self.gui_manager,
                data_coordinator=data_coordinator,
                config_manager=self.app.config_manager,
                on_save_callback=self._on_data_settings_saved,
            )
            
            dialog.show()
            
        except ImportError as e:
            self.logger.error(f"Could not import ColumnSettingsDialog: {e}")
            from gui.dialog_helper import DialogHelper
            DialogHelper.show_message(
                self.root,
                self.t("Error"),
                self.t("Column Settings dialog not available."),
                message_type="error"
            )
        except Exception as e:
            self.logger.error(f"Error opening Data Settings: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _on_data_settings_saved(self):
        """Callback when data settings are saved."""
        self.logger.info("Data settings saved - refreshing data coordinator")
        # Optionally refresh UI or data coordinator here
        self.update_status(self.t("Data settings updated."), "info")
    
    def _open_color_map_settings(self):
        """Open the Color Map Editor dialog."""
        try:
            from gui.color_map_editor_dialog import ColorMapEditorDialog
            
            # Get data coordinator for color maps and data
            data_coordinator = getattr(self.app, 'data_coordinator', None)
            
            dialog = ColorMapEditorDialog(
                parent=self.root,
                gui_manager=self.gui_manager,
                data_coordinator=data_coordinator,
                on_save_callback=lambda: self.update_status(self.t("Color map saved."), "info"),
            )
            
            dialog.show()
            
        except ImportError as e:
            self.logger.error(f"Could not import ColorMapEditorDialog: {e}")
        except Exception as e:
            self.logger.error(f"Error opening Color Map settings: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _open_classification_manager(self):
        """Open the Classification Manager dialog."""
        # If you have an existing classification manager dialog, call it here
        # For now, show a placeholder message
        from gui.dialog_helper import DialogHelper
        DialogHelper.show_message(
            self.root,
            self.t("Classification Manager"),
            self.t("Classification Manager dialog - coming soon."),
            message_type="info"
        )
    
    def _open_tag_manager(self):
        """Open the Tag Manager dialog."""
        # If you have an existing tag manager dialog, call it here
        # For now, show a placeholder message
        from gui.dialog_helper import DialogHelper
        DialogHelper.show_message(
            self.root,
            self.t("Tag Manager"),
            self.t("Tag Manager dialog - coming soon."),
            message_type="info"
        )

    def _open_original_image_viewer(self):
        """Open the Original Image Viewer dialog."""
        try:
            from gui.original_image_viewer_dialog import open_original_image_viewer
            
            # Create a new instance of the viewer (allows multiple windows)
            viewer = open_original_image_viewer(
                parent=self.root,
                file_manager=self.file_manager,
                gui_manager=self.gui_manager,
                json_register_manager=self.app.register_manager
            )
            
            logger.info("Opened Original Image Viewer dialog")
            
        except Exception as e:
            logger.error(f"Error opening Original Image Viewer: {e}")
            import traceback
            traceback.print_exc()
            DialogHelper.show_message(
                self.root,
                self.t("Error"),
                self.t(f"Failed to open Original Image Viewer:\n\n{str(e)}"),
                message_type="error"
            )
    
    def _open_bulk_renamer(self):
        """
        Open the Bulk Photo Renamer using DialogHelper (PEP-8: docstring).
        Assumes PhotoRenamerGUI and DialogHelper are importable.
        """
        try:
            # Guard missing modules with themed message dialog
            if PhotoRenamerGUI is None or DialogHelper is None:
                if DialogHelper:
                    DialogHelper.show_message(
                        self.root,
                        self.t("Error"),
                        self.t("Bulk Renamer or DialogHelper module not found."),
                        message_type="error",
                    )
                else:
                    self.update_status(
                        self.t("Bulk Renamer or DialogHelper module not found."),
                        "error",
                    )
                return

            # Create a normal window (not transient) that can be maximized
            dialog = tk.Toplevel(self.root)
            dialog.title(self.t("Photo Bulk Renamer"))

            # Apply theme if gui_manager available
            if self.gui_manager:
                self.gui_manager.apply_theme(dialog)
                dialog.configure(bg=self.gui_manager.theme_colors["background"])

            # Make it maximizable (not transient)
            dialog.transient(None)  # Explicitly set to not be transient
            dialog.resizable(True, True)

            # Start maximized
            dialog.state("zoomed")  # Windows/Linux
            # For cross-platform compatibility, also try:
            try:
                dialog.attributes("-zoomed", True)  # Alternative for some systems
            except:
                pass

            # Instantiate the renamer inside the dialog (consistent theming/focus)
            # Pass gui_manager for consistent theming
            renamer = PhotoRenamerGUI(dialog, gui_manager=self.gui_manager)

            # Try to preload CSV via existing paths (silent fallback)
            # Look for drillhole_data.csv in the datasets folder
            try:
                csv_path = None
                datasets_folder = self.file_manager.get_shared_path(
                    "datasets", create_if_missing=False
                )
                if datasets_folder and datasets_folder.exists():
                    # Look for drillhole_data.csv in the datasets folder
                    potential_csv = datasets_folder / "drillhole_data.csv"
                    if potential_csv.exists():
                        csv_path = str(potential_csv)
                if csv_path and os.path.exists(csv_path):
                    # assuming renamer.validator.load_csv works
                    if getattr(
                        renamer, "validator", None
                    ) and renamer.validator.load_csv(csv_path):
                        if hasattr(renamer, "csv_label"):
                            renamer.csv_label.config(
                                text=f"✓ {os.path.basename(csv_path)}",
                                foreground="green",
                            )
            except Exception:
                pass  # non-fatal preload

            # Let layout calculate natural size then center with constraints
            dialog.update_idletasks()
            DialogHelper.center_dialog(
                dialog,
                parent=self.root,
                size_ratio=0.9,
                min_width=900,
                min_height=600,
                max_width=None,
                max_height=None,
            )

            # Block until closed (DialogHelper already grabs/focuses)
            dialog.wait_window()

        except Exception as e:
            self.logger.error(f"Failed to open Bulk Renamer: {e}")
            if DialogHelper:
                DialogHelper.show_message(
                    self.root,
                    self.t("Error"),
                    self.t("Failed to open Bulk Renamer."),
                    message_type="error",
                )
            else:
                self.update_status(self.t("Failed to open Bulk Renamer."), "error")

    def _generate_image_properties(self):
        """Generate image properties register (hex colors, chip size, etc) by scanning approved compartment folders."""
        try:
            # Check if we have a register path (we need this to know where to save the properties file)
            register_path = self.file_manager.get_shared_path(
                "register_excel", create_if_missing=False
            )
            if not register_path:
                DialogHelper.show_message(
                    self.root,
                    self.t("Error"),
                    self.t(
                        "No register path configured. Please check shared folder settings."
                    ),
                    message_type="error",
                )
                return

            # Get base path for properties file
            base_path = str(register_path.parent)

            # Create JSON manager
            json_manager = JSONRegisterManager(base_path, self.logger)

            # Get ONLY approved compartments folder (not review - those don't have wet/dry yet)
            approved_path = self.file_manager.get_shared_path("approved_compartments")

            if not approved_path or not approved_path.exists():
                DialogHelper.show_message(
                    self.root,
                    self.t("Error"),
                    self.t("Approved compartments folder not found or not configured."),
                    message_type="error",
                )
                return

            # Create progress dialog
            progress_dialog = ProgressDialog(
                self.root,
                self.t("Generating Image Properties"),
                self.t("Scanning folders..."),
            )

            try:
                # Recursively scan for all compartment image files in approved folder
                # Structure: approved_compartments/PROJECT_CODE/HOLE_ID/*.png
                image_files = []

                # Use rglob for recursive search through project/hole subfolders
                for file_path in approved_path.rglob("*.png"):
                    image_files.append(str(file_path))
                for file_path in approved_path.rglob("*.jpg"):
                    image_files.append(str(file_path))
                for file_path in approved_path.rglob("*.jpeg"):
                    image_files.append(str(file_path))

                if not image_files:
                    DialogHelper.show_message(
                        self.root,
                        self.t("No Images"),
                        self.t("No compartment images found in approved folder."),
                        message_type="info",
                    )
                    return

                progress_dialog.update_progress(
                    self.t("Found {} images. Loading existing properties...").format(
                        len(image_files)
                    ),
                    3,
                )
                if progress_dialog.dialog and progress_dialog.dialog.winfo_exists():
                    progress_dialog.dialog.update()

                # Load existing properties once for batch processing using JSONRegisterManager
                all_props = json_manager.get_all_image_properties()
                self.logger.info(f"Loaded {len(all_props)} existing image properties from register")

                progress_dialog.update_progress(
                    self.t("Processing {} images...").format(len(image_files)), 5
                )
                if progress_dialog.dialog and progress_dialog.dialog.winfo_exists():
                    progress_dialog.dialog.update()

                # Track results
                calculated = 0
                cached = 0
                failed = 0
                skipped_no_classification = 0
                errors = []

                # Build an interval-based dictionary to consolidate wet/dry
                # Key: (hole_id, depth_from, depth_to), Value: {wet_hex, dry_hex, etc}
                interval_props = {}
                
                # Build lookup dictionary from existing properties for fast skip logic
                # Key: (hole_id, depth_from, depth_to), Value: {Wet_Hex, Dry_Hex, etc}
                existing_intervals = {}
                for prop in all_props:
                    key = (
                        prop.get("HoleID"),
                        prop.get("Depth_From"),
                        prop.get("Depth_To"),
                    )
                    existing_intervals[key] = prop
                
                self.logger.info(f"Found {len(existing_intervals)} existing intervals in register")

                # Process each image file
                for idx, file_path in enumerate(image_files):
                    try:
                        filename = os.path.basename(file_path)

                        # Check if image has wet/dry classification in filename
                        filename_lower = filename.lower()
                        if (
                            "_wet" not in filename_lower
                            and "_dry" not in filename_lower
                        ):
                            skipped_no_classification += 1
                            self.logger.debug(
                                f"Skipping {filename} - no wet/dry classification"
                            )
                            continue

                        # Parse filename for metadata (HoleID_DepthFrom-DepthTo_Wet/Dry format)
                        # Example: KM0002_CC_001_Dry.png or BELD001_100-120_Wet.png
                        hole_id = "Unknown"
                        depth_from = 0
                        depth_to = 0

                        try:
                            # Remove extension
                            name_no_ext = os.path.splitext(filename)[0]
                            parts = name_no_ext.split("_")

                            if len(parts) >= 2:
                                hole_id = parts[0]

                                # Look for depth pattern (FROM-TO or just TO)
                                for part in parts[1:]:
                                    if "-" in part:
                                        # Found depth range: 100-120
                                        depth_parts = part.split("-")
                                        if len(depth_parts) == 2:
                                            depth_from = int(float(depth_parts[0]))
                                            depth_to = int(float(depth_parts[1]))
                                            break
                                    elif (
                                        part.isdigit()
                                        or part.replace(".", "").isdigit()
                                    ):
                                        # Single depth value is the TO depth, calculate FROM
                                        # Example: BA0131_CC_071_Dry.png means TO=71m, so FROM=70m (1m interval)
                                        depth_to = int(float(part))
                                        # Always assume 1m interval for single depth values
                                        depth_from = depth_to - 1
                                        break
                        except Exception as parse_error:
                            self.logger.debug(
                                f"Could not parse filename {filename}: {parse_error}"
                            )
                            # Continue with Unknown/0 values

                        # Determine moisture status
                        moisture = "Unknown"
                        if "_Wet" in filename or "_wet" in filename:
                            moisture = "Wet"
                        elif "_Dry" in filename or "_dry" in filename:
                            moisture = "Dry"

                        # Create interval key
                        interval_key = (hole_id, depth_from, depth_to)

                        # Check if we can skip this image (hex already exists in register)
                        skip_calculation = False
                        if interval_key in existing_intervals:
                            existing_prop = existing_intervals[interval_key]
                            if moisture == "Wet" and existing_prop.get("Wet_Hex"):
                                # Wet hex already calculated, skip
                                skip_calculation = True
                                cached += 1
                                # Copy existing to interval_props if not already there
                                if interval_key not in interval_props:
                                    interval_props[interval_key] = existing_prop.copy()
                                    # Ensure Combined_Hex is calculated (for legacy data)
                                    if "Combined_Hex" not in interval_props[interval_key]:
                                        dry_hex = interval_props[interval_key].get("Dry_Hex", "")
                                        wet_hex = interval_props[interval_key].get("Wet_Hex", "")
                                        interval_props[interval_key]["Combined_Hex"] = dry_hex if dry_hex else wet_hex
                            elif moisture == "Dry" and existing_prop.get("Dry_Hex"):
                                # Dry hex already calculated, skip
                                skip_calculation = True
                                cached += 1
                                # Copy existing to interval_props if not already there
                                if interval_key not in interval_props:
                                    interval_props[interval_key] = existing_prop.copy()
                                    # Ensure Combined_Hex is calculated (for legacy data)
                                    if "Combined_Hex" not in interval_props[interval_key]:
                                        dry_hex = interval_props[interval_key].get("Dry_Hex", "")
                                        wet_hex = interval_props[interval_key].get("Wet_Hex", "")
                                        interval_props[interval_key]["Combined_Hex"] = dry_hex if dry_hex else wet_hex

                        if skip_calculation:
                            # Skip this image, hex color already exists
                            continue

                        # Calculate hex color (only if not skipped)
                        color_result = self.file_manager.calculate_robust_hex_color(
                            file_path, method="LAB_shadow_compensated"
                        )

                        # Get or create interval entry
                        if interval_key not in interval_props:
                            interval_props[interval_key] = {
                                "HoleID": hole_id,
                                "Depth_From": depth_from,
                                "Depth_To": depth_to,
                                "Wet_Hex": "",
                                "Dry_Hex": "",
                                "Combined_Hex": "",
                                "Calculation_Method": color_result["method"],
                                "Calculated_Date": datetime.now().isoformat(),
                            }
                            calculated += 1

                        # Update hex color for this moisture type
                        if moisture == "Wet" and color_result["valid"]:
                            interval_props[interval_key]["Wet_Hex"] = color_result[
                                "hex_color"
                            ]
                        elif moisture == "Dry" and color_result["valid"]:
                            interval_props[interval_key]["Dry_Hex"] = color_result[
                                "hex_color"
                            ]
                        
                        # Update Combined_Hex (prefer Dry, fallback to Wet)
                        dry_hex = interval_props[interval_key].get("Dry_Hex", "")
                        wet_hex = interval_props[interval_key].get("Wet_Hex", "")
                        interval_props[interval_key]["Combined_Hex"] = dry_hex if dry_hex else wet_hex

                        # Checkpoint: Save every 1,000 processed files
                        if (idx + 1) % 1000 == 0:
                            # Convert interval_props to list
                            checkpoint_props = list(interval_props.values())
                            # Merge with existing all_props (update or append)
                            existing_keys = {
                                (
                                    p.get("HoleID"),
                                    p.get("Depth_From"),
                                    p.get("Depth_To"),
                                ): i
                                for i, p in enumerate(all_props)
                            }
                            for prop in checkpoint_props:
                                key = (
                                    prop["HoleID"],
                                    prop["Depth_From"],
                                    prop["Depth_To"],
                                )
                                if key in existing_keys:
                                    all_props[existing_keys[key]] = prop
                                else:
                                    all_props.append(prop)

                            # Write checkpoint to disk
                            if json_manager.save_image_properties_batch(all_props):
                                self.logger.info(
                                    f"Checkpoint saved at {idx + 1} images"
                                )

                        # Update progress every 100 files or at milestones
                        if (idx + 1) % 100 == 0 or (idx + 1) == len(image_files):
                            progress = 5 + int((idx + 1) / len(image_files) * 90)
                            progress_dialog.update_progress(
                                self.t("Processed {}/{}: {}...").format(
                                    idx + 1, len(image_files), filename[:30]
                                ),
                                progress,
                            )
                            # Force UI update
                            if (
                                progress_dialog.dialog
                                and progress_dialog.dialog.winfo_exists()
                            ):
                                progress_dialog.dialog.update()

                    except Exception as e:
                        failed += 1
                        error_msg = f"{os.path.basename(file_path)}: {str(e)}"
                        errors.append(error_msg)
                        self.logger.error(f"Error processing image: {error_msg}")

                # Final merge and write to disk
                progress_dialog.update_progress(
                    self.t("Finalizing and saving all properties..."), 95
                )
                if progress_dialog.dialog and progress_dialog.dialog.winfo_exists():
                    progress_dialog.dialog.update()

                # Convert final interval_props to list
                final_props = list(interval_props.values())

                # Merge with existing all_props
                existing_keys = {
                    (p.get("HoleID"), p.get("Depth_From"), p.get("Depth_To")): i
                    for i, p in enumerate(all_props)
                }
                for prop in final_props:
                    key = (prop["HoleID"], prop["Depth_From"], prop["Depth_To"])
                    if key in existing_keys:
                        all_props[existing_keys[key]] = prop
                    else:
                        all_props.append(prop)

                if not json_manager.save_image_properties_batch(all_props):
                    self.logger.error("Failed to save final image properties batch")

                progress_dialog.update_progress(self.t("Complete!"), 100)
                # Force final UI update
                if progress_dialog.dialog and progress_dialog.dialog.winfo_exists():
                    progress_dialog.dialog.update()

                # Build result message
                message_lines = [
                    self.t("Image Properties Generation Complete"),
                    "",
                    self.t("Total images scanned: {}").format(len(image_files)),
                    self.t("Newly calculated: {}").format(calculated),
                    self.t("Retrieved from cache: {}").format(cached),
                ]

                if skipped_no_classification > 0:
                    message_lines.append(
                        self.t("Skipped (no wet/dry): {}").format(
                            skipped_no_classification
                        )
                    )

                if failed > 0:
                    message_lines.append("")
                    message_lines.append(self.t("Failed: {}").format(failed))

                    if len(errors) <= 10:
                        message_lines.append("")
                        message_lines.append(self.t("Errors:"))
                        for error in errors[:10]:
                            message_lines.append(f"  • {error}")
                    else:
                        message_lines.append(
                            f"  (showing first 10 of {len(errors)} errors)"
                        )
                        for error in errors[:10]:
                            message_lines.append(f"  • {error}")

                message_lines.append("")
                message_lines.append(
                    self.t("Properties saved to: {}/Register_Data/{}").format(
                        base_path, json_manager.IMAGE_PROPERTIES_JSON
                    )
                )

                # Show appropriate message type
                if failed == 0:
                    msg_type = "info"
                elif failed < len(image_files) / 2:
                    msg_type = "warning"
                else:
                    msg_type = "error"

                DialogHelper.show_message(
                    self.root,
                    self.t("Generation Results"),
                    "\n".join(message_lines),
                    message_type=msg_type,
                )

            finally:
                if progress_dialog.dialog and progress_dialog.dialog.winfo_exists():
                    progress_dialog.close()

        except Exception as e:
            self.logger.error(f"Error in _generate_image_properties: {str(e)}")
            self.logger.error(traceback.format_exc())
            DialogHelper.show_message(
                self.root,
                self.t("Error"),
                self.t("An error occurred:") + f"\n{str(e)}",
                message_type="error",
            )

    def _setup_sync_logger(self):
        """Setup a custom logger handler for cloud sync operations."""
        import logging

        class StatusBoxHandler(logging.Handler):
            """Custom handler that routes logs to the GUI status box."""

            def __init__(self, gui):
                super().__init__()
                self.gui = gui

            def emit(self, record):
                try:
                    msg = self.format(record)
                    # Determine status type based on log level
                    if record.levelno >= logging.ERROR:
                        status_type = "error"
                    elif record.levelno >= logging.WARNING:
                        status_type = "warning"
                    elif record.levelno >= logging.INFO:
                        status_type = "info"
                    else:
                        status_type = "info"

                    # Use after to ensure it runs on the main thread
                    if hasattr(self.gui, "root") and self.gui.root.winfo_exists():
                        self.gui.root.after(
                            0, lambda: self.gui.update_status(msg, status_type)
                        )
                except:
                    pass

        # Add handler to cloud sync logger
        sync_logger = logging.getLogger("utils.cloud_sync_manager")
        sync_logger.setLevel(logging.DEBUG)  # Show debug messages
        handler = StatusBoxHandler(self)
        handler.setFormatter(logging.Formatter("%(message)s"))
        sync_logger.addHandler(handler)

    # Add a new direct update method for other scripts to call
    def direct_status_update(self, message, status_type="info", progress=None):
        """
        Directly update status without using the queue.
        Other modules can call this method to update the UI directly.

        Args:
            message: Status message to display
            status_type: Type of status message (error, warning, success, info)
            progress: Optional progress value to update the progress bar
        """
        # Use the root's after method to ensure this runs on the UI thread
        if hasattr(self, "root") and self.root.winfo_exists():
            self.root.after(
                0, lambda: self._perform_direct_update(message, status_type, progress)
            )

    def _perform_direct_update(self, message, status_type, progress):
        """Helper method to actually perform the update on the UI thread."""
        # Update progress if provided
        if progress is not None:
            self.progress_var.set(progress)

        # Update status text
        self.update_status(message, status_type)

    def _on_depth_csv_changed(self, new_path):
        """Handle Drilling intervals CSV path change."""
        if new_path and os.path.exists(new_path):
            # Reload the depth validator
            self._initialize_depth_validator()
            self.logger.info(f"Drilling intervals updated: {new_path}")

    def _update_input_folder_color(self, *args):
        """Update input folder entry background color based on content validity."""
        folder_path = self.folder_var.get()
        if folder_path and os.path.isdir(folder_path):
            # Check if directory contains image files
            has_images = any(
                f.lower().endswith((".jpg", ".jpeg", ".HEIC", ".png", ".tif", ".tiff"))
                for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f))
            )

            if has_images:
                self.folder_entry.config(
                    bg=self.gui_manager.theme_colors["accent_valid"]
                )
                # Make browse button less prominent once valid folder selected
                self.buttons["process_button"].set_state("normal")
            else:
                self.folder_entry.config(
                    bg=self.gui_manager.theme_colors["accent_error"]
                )
        else:
            self.folder_entry.config(bg=self.gui_manager.theme_colors["accent_error"])

    def setup_shared_paths(self):
        """
        Setup all shared folder paths and save them to config.
        This should be called after FileManager is initialized.
        """
        if not hasattr(self, "file_manager") or not self.file_manager:
            self.logger.error("FileManager not available for setting up shared paths")
            return False

        try:
            # Get all the paths
            paths_found = False

            # Approved folder
            approved_path = self.file_manager.get_shared_path(
                "approved_compartments", create_if_missing=False
            )
            if approved_path and approved_path.exists():
                self.app.config_manager.set(
                    "shared_folder_approved_compartments_folder", str(approved_path)
                )
                self.approved_path_var.set(str(approved_path))
                self.logger.info(f"Saved shared approved folder: {approved_path}")
                paths_found = True

            # Processed originals
            processed_path = self.file_manager.get_shared_path(
                "processed_originals", create_if_missing=False
            )
            if processed_path and processed_path.exists():
                self.app.config_manager.set(
                    "shared_folder_processed_originals", str(processed_path)
                )
                self.processed_originals_path_var.set(str(processed_path))
                self.logger.info(f"Saved shared processed originals: {processed_path}")
                paths_found = True

            # Rejected/Failed folder
            rejected_path = self.file_manager.get_shared_path(
                "rejected_originals", create_if_missing=False
            )
            if rejected_path and rejected_path.exists():
                self.app.config_manager.set(
                    "shared_folder_rejected_folder", str(rejected_path)
                )
                self.logger.info(f"Saved shared rejected folder: {rejected_path}")
                paths_found = True

            # Drill traces
            drill_traces_path = self.file_manager.get_shared_path(
                "drill_traces", create_if_missing=False
            )
            if drill_traces_path and drill_traces_path.exists():
                self.app.config_manager.set(
                    "shared_folder_drill_traces", str(drill_traces_path)
                )
                self.drill_traces_path_var.set(str(drill_traces_path))
                self.logger.info(f"Saved shared drill traces: {drill_traces_path}")
                paths_found = True

            # Excel register
            register_path = self.file_manager.get_shared_path(
                "register_excel", create_if_missing=False
            )
            if register_path and register_path.exists():
                self.app.config_manager.set(
                    "shared_folder_register_excel_path", str(register_path)
                )
                self.register_path_var.set(str(register_path))
                self.logger.info(f"Saved shared register path: {register_path}")
                paths_found = True

            # Register Data folder
            register_data_path = self.file_manager.get_shared_path(
                "register_data", create_if_missing=False
            )
            if register_data_path and register_data_path.exists():
                self.app.config_manager.set(
                    "shared_folder_register_data_folder", str(register_data_path)
                )
                self.logger.info(
                    f"Saved shared register data folder: {register_data_path}"
                )
                paths_found = True

            # Update entry widget colors based on path validity
            self._update_path_entry_colors()

            return paths_found

        except Exception as e:
            self.logger.error(f"Error setting up shared paths: {str(e)}")
            return False

    def _update_path_entry_colors(self):
        """Update the colors of path entry widgets based on validity."""
        # Find all entry widgets in the shared folder collapsible section
        if hasattr(self, "shared_collapsible"):
            for frame in self.shared_collapsible.content_frame.winfo_children():
                if isinstance(frame, ttk.Frame):
                    for child in frame.winfo_children():
                        if isinstance(child, tk.Entry):
                            # Check which variable this entry is bound to
                            try:
                                var_str = str(child.cget("textvariable"))
                                path_value = child.get()

                                # Update color based on path validity
                                if path_value and os.path.exists(path_value):
                                    child.config(
                                        bg=self.gui_manager.theme_colors["accent_valid"]
                                    )
                                else:
                                    child.config(
                                        bg=self.gui_manager.theme_colors["accent_error"]
                                    )
                            except:
                                pass

    def _create_shared_path_field(
        self, parent, label_text, string_var, valid=False, is_file=False, path_key=None
    ):
        """Create a field for shared folder path input with browse button.

        Args:
            parent: Parent widget
            label_text: Label text for the field
            string_var: StringVar to bind to the entry
            valid: Whether the current path is valid (for coloring)
            is_file: Whether browsing for a file vs folder
            path_key: FileManager path key (e.g., 'approved_compartments', 'register_excel')
        """
        frame = ttk.Frame(parent, style="Content.TFrame")
        frame.pack(fill=tk.X, pady=5)

        # Label with wider fixed width
        label = ttk.Label(
            frame, text=self.t(label_text), width=25, anchor="w", style="Content.TLabel"
        )
        label.pack(side=tk.LEFT)

        # Themed entry field with validation coloring
        entry = tk.Entry(
            frame,
            textvariable=string_var,
            font=self.gui_manager.fonts["entry"],
            bg=(
                self.gui_manager.theme_colors["accent_valid"]
                if valid
                else self.gui_manager.theme_colors["accent_error"]
            ),
            fg=self.gui_manager.theme_colors["text"],
            insertbackground=self.gui_manager.theme_colors["text"],
            relief=tk.FLAT,
            bd=1,
            highlightbackground=self.gui_manager.theme_colors["field_border"],
            highlightthickness=1,
        )
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        # Browse button with path_key passed to browse method
        browse_button = self.gui_manager.create_modern_button(
            frame,
            text=self.t("Browse"),
            color=self.gui_manager.theme_colors["accent_blue"],
            command=lambda: self._browse_shared_path(
                string_var, entry, is_file=is_file, path_key=path_key
            ),
        )
        browse_button.pack(side=tk.RIGHT)

    def _load_language_preference(self):
        """Load language preference from config file."""
        try:
            # Access language preference directly from config_manager
            language = self.app.config_manager.get("language")
            if language:
                self.translator.set_language(language)
                self.logger.info(f"Loaded language preference: {language}")
        except Exception as e:
            self.logger.error(f"Error loading language preference: {str(e)}")

    def _save_language_preference(self, language_code):
        """Save language preference to config file."""
        try:
            # Use the app's config_manager to save the preference
            self.app.config_manager.set("language", language_code)
            self.logger.info(f"Saved language preference: {language_code}")
        except Exception as e:
            self.logger.error(f"Error saving language preference: {str(e)}")

    def _toggle_theme(self):
        """Toggle between light and dark themes."""
        self.gui_manager.toggle_theme()

        # Update menubar theme toggle label
        theme_label = self.gui_manager.get_theme_label()
        file_menu = self.menubar.children.get(self.t("File").lower())
        if file_menu:
            file_menu.entryconfigure(0, label=theme_label)

        # Apply theme to menubar directly
        self.menubar.config(
            bg=self.gui_manager.theme_colors["menu_bg"],
            fg=self.gui_manager.theme_colors["menu_fg"],
            activebackground=self.gui_manager.theme_colors["menu_active_bg"],
            activeforeground=self.gui_manager.theme_colors["menu_active_fg"],
        )

        # Apply theme to all cascade menus
        for menu_name in self.menubar.winfo_children():
            try:
                menu = self.menubar.nametowidget(menu_name)
                menu.config(
                    bg=self.gui_manager.theme_colors["menu_bg"],
                    fg=self.gui_manager.theme_colors["menu_fg"],
                    activebackground=self.gui_manager.theme_colors["menu_active_bg"],
                    activeforeground=self.gui_manager.theme_colors["menu_active_fg"],
                )
            except:
                pass

        # Update all widgets with the new theme
        self.gui_manager.update_widget_theme(self.root)

        # Update custom widget themes explicitly
        self.gui_manager.update_custom_widget_theme(self.main_container)

        # Add a status message
        self.update_status(
            self.t(f"Switched to {self.gui_manager.current_theme} theme"), "info"
        )

    def update_status(self, message: str, status_type) -> None:
        """
        Add a message to the status text widget with optional formatting.

        Args:
            message: The message to display
            status_type: Optional status type for formatting (error, warning, success, info)
        """
        try:
            # Check if status_text exists
            if not hasattr(self, "status_text") or self.status_text is None:
                # Just log the message
                self.logger.info(message)
                return

            self.status_text.config(state=tk.NORMAL)

            # Add timestamp
            import datetime

            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            self.status_text.insert(tk.END, f"[{timestamp}] ", "info")

            # Add message with appropriate tag if specified
            if status_type and status_type in ["error", "warning", "success", "info"]:
                self.status_text.insert(tk.END, f"{message}\n", status_type)
            else:
                self.status_text.insert(tk.END, f"{message}\n")

            self.status_text.see(tk.END)
            self.status_text.config(state=tk.DISABLED)
        except Exception as e:
            # Log error but don't raise it to avoid UI crashes
            self.logger.error(f"Error updating status: {str(e)}")

    def _create_modern_button(
        self, parent, text, color, command, icon=None, grid_pos=None
    ):
        """
        Create a modern styled button with hover effects.

        Args:
            parent: Parent widget
            text: Button text
            color: Base button color
            command: Button command
            icon: Optional text icon
            grid_pos: Optional grid position tuple (row, col)

        Returns:
            The created button frame
        """
        # Create a frame with the button's background color
        button_frame = tk.Frame(
            parent,
            background=color,
            highlightbackground=color,
            highlightthickness=1,
            bd=0,
            cursor="hand2",
        )

        # Minimum width based on text length
        min_width = max(120, len(text) * 10)

        # Button content with icon and text
        prefix = f"{icon} " if icon else ""
        button_text = tk.Label(
            button_frame,
            text=prefix + self.t(text),
            background=color,
            foreground="white",
            font=(self.gui_manager.fonts["button"]),
            padx=15,
            pady=8,
            cursor="hand2",
            width=min_width // 10,  # Approximate character width
        )
        button_text.pack(fill=tk.BOTH, expand=True)

        # Hover effects
        def on_enter(e):
            # Safety check - make sure the widget still exists
            if button_frame.winfo_exists() and button_text.winfo_exists():
                # Lighten the color slightly on hover
                button_frame.config(background=self._lighten_color(color, 0.15))
                button_text.config(background=self._lighten_color(color, 0.15))

        def on_leave(e):
            # Safety check - make sure the widget still exists
            if button_frame.winfo_exists() and button_text.winfo_exists():
                button_frame.config(background=color)
                button_text.config(background=color)

        def on_click(e):
            if button_frame.winfo_exists() and button_text.winfo_exists():
                try:
                    button_frame.config(background=self._darken_color(color, 0.15))
                    button_text.config(background=self._darken_color(color, 0.15))
                    command()
                except Exception as err:
                    import traceback

                    self.logger.error(
                        "Error executing button command:\n" + traceback.format_exc()
                    )

        # Bind events
        button_frame.bind("<Enter>", on_enter)
        button_text.bind("<Enter>", on_enter)
        button_frame.bind("<Leave>", on_leave)
        button_text.bind("<Leave>", on_leave)
        button_frame.bind("<Button-1>", on_click)
        button_text.bind("<Button-1>", on_click)

        return button_frame

    def _create_custom_checkbox(self, parent, text, variable, command=None):
        """
        Create a custom checkbox with better visibility when checked/unchecked.

        Args:
            parent: Parent widget
            text: Checkbox text
            variable: BooleanVar to track state
            command: Optional command to execute on toggle

        Returns:
            Frame containing the custom checkbox
        """
        # Create frame for checkbox
        frame = ttk.Frame(parent, style="Content.TFrame")

        # Custom checkbox appearance
        checkbox_size = 18
        checkbox_frame = tk.Frame(
            frame,
            width=checkbox_size,
            height=checkbox_size,
            bg=self.theme_colors["checkbox_bg"],
            highlightbackground=self.theme_colors["field_border"],
            highlightthickness=1,
        )
        checkbox_frame.pack(side=tk.LEFT, padx=(0, 5))
        checkbox_frame.pack_propagate(False)  # Maintain size

        # Checkmark that appears when checked
        checkmark = tk.Label(
            checkbox_frame,
            text="✓",
            bg=self.theme_colors["checkbox_bg"],
            fg=self.theme_colors["checkbox_fg"],
            font=(self.gui_manager.fonts["button"]),
        )

        # Show/hide checkmark based on variable state
        def update_state(*args):
            if variable.get():
                checkmark.pack(fill=tk.BOTH, expand=True)
            else:
                checkmark.pack_forget()

        # Initial state
        update_state()

        # Bind variable changes
        variable.trace_add("write", update_state)

        # Toggle function
        def toggle(event=None):
            variable.set(not variable.get())
            if command:
                command()

        # Bind click events
        checkbox_frame.bind("<Button-1>", toggle)
        checkmark.bind("<Button-1>", toggle)

        # Add text label
        label = ttk.Label(frame, text=self.t(text), style="Content.TLabel")
        label.pack(side=tk.LEFT)
        label.bind("<Button-1>", toggle)

        return frame

    def _create_collapsible_frame(self, parent, title, expanded=False):
        """
        Create a themed collapsible frame.

        Args:
            parent: Parent widget
            title: Frame title
            expanded: Whether frame is initially expanded

        Returns:
            CollapsibleFrame instance
        """
        frame = CollapsibleFrame(
            parent,
            text=title,
            expanded=expanded,
            bg=self.theme_colors["secondary_bg"],
            fg=self.theme_colors["text"],
            title_bg=self.theme_colors["secondary_bg"],
            title_fg=self.theme_colors["text"],
            content_bg=self.theme_colors["background"],
            border_color=self.theme_colors["border"],
            arrow_color=self.theme_colors["accent_blue"],
        )
        frame.pack(fill=tk.X, pady=(0, 10))
        return frame

    def _update_gui_translations(self):
        """Update all GUI text elements with current language."""
        try:
            if not hasattr(self, "root") or not self.root:
                return

            # Update window title
            self.root.title(self.t("GeoVue"))

            # Update header and section titles
            if hasattr(self, "title_label"):
                self.title_label.config(text=self.t("GeoVue"))

            # Update input section labels
            if hasattr(self, "folder_entry"):
                # Find the label in the folder_entry's parent (folder_frame)
                for child in self.folder_entry.master.winfo_children():
                    if isinstance(child, ttk.Label):
                        child.config(text=self.t("Input Folder:"))
                        break

            # Update interval label
            for frame in self.content_frame.winfo_children():
                if isinstance(frame, ttk.Frame):
                    for child in frame.winfo_children():
                        if isinstance(child, ttk.Label) and child.cget(
                            "text"
                        ).startswith("Compartment Interval"):
                            child.config(text=self.t("Compartment Interval (m):"))
                        elif isinstance(child, ttk.Label) and child.cget(
                            "text"
                        ).startswith("Output Format"):
                            child.config(text=self.t("Output Format:"))
                        elif isinstance(child, ttk.Label) and child.cget(
                            "text"
                        ).startswith("Local Output"):
                            child.config(text=self.t("Local Output:"))

            # Update all custom widgets with translation methods
            for container in [
                self.content_frame,
                self.main_container,
                self.footer_frame,
            ]:
                self._update_translations_recursive(container)

            # Update collapsible frame titles
            if hasattr(self, "processing_collapsible"):
                self.processing_collapsible.set_text(self.t("Image Processing"))
            if hasattr(self, "shared_collapsible"):
                self.shared_collapsible.set_text(self.t("Shared Folder Settings"))
            if hasattr(self, "analysis_collapsible"):
                self.analysis_collapsible.set_text(self.t("Image Analysis and Review"))

            # Update shared folder path labels
            if hasattr(self, "shared_collapsible"):
                shared_frame = self.shared_collapsible.content_frame
                for child in shared_frame.winfo_children():
                    if isinstance(child, ttk.Frame):
                        for subchild in child.winfo_children():
                            if isinstance(subchild, ttk.Label):
                                if "Approved" in subchild.cget("text"):
                                    subchild.config(text=self.t("Approved Folder:"))
                                elif "Processed" in subchild.cget("text"):
                                    subchild.config(
                                        text=self.t("Processed Originals Folder:")
                                    )
                                elif "Drill" in subchild.cget("text"):
                                    subchild.config(text=self.t("Drill Traces Folder:"))
                                elif "Excel" in subchild.cget("text"):
                                    subchild.config(text=self.t("Excel Register:"))

            # Update button texts - include all buttons
            if hasattr(self, "buttons"):
                button_texts = {
                    "process_button": "Process Photos",
                    "review_button": "Review Extracted Images",
                    "validate_button": "Calculate Image Properties",
                    "logging_button": "Logging Review",
                    "trace_button": "Generate Drillhole Trace",
                    "quit_button": "Quit",
                }

                for btn_name, text in button_texts.items():
                    if btn_name in self.buttons:
                        button = self.buttons[btn_name]
                        try:
                            # ModernButton has a set_text method
                            if hasattr(button, "set_text"):
                                button.set_text(self.t(text))
                            elif hasattr(button, "config"):
                                button.config(text=self.t(text))
                            elif hasattr(button, "configure"):
                                button.configure(text=self.t(text))
                        except Exception as btn_error:
                            self.logger.warning(
                                f"Could not update text for button {btn_name}: {str(btn_error)}"
                            )

            # Rebuild menu with new translations
            self._rebuild_menu()

            # Update status text
            self.update_status(self.t("Language changed successfully"), "info")

        except Exception as e:
            self.logger.error(f"Error updating translations: {str(e)}")
            import traceback

            self.logger.error(traceback.format_exc())

    def _update_translations_recursive(self, container):
        """
        Recursively update translations for all widgets that support it.

        Args:
            container: The container widget to search through
        """
        for child in container.winfo_children():
            # Check if this widget has a translation update method
            if hasattr(child, "update_translation"):
                try:
                    child.update_translation(self.t)
                except Exception as e:
                    self.logger.debug(f"Error updating translation for widget: {e}")

            # For custom frames from field_with_label that return a tuple (frame, field)
            # The frame itself has the update_translation method
            if hasattr(child, "winfo_children"):
                self._update_translations_recursive(child)

    def change_language(self, language_code):
        """
        Change the application language and update all GUI text.

        Args:
            language_code (str): The language code to switch to

        Returns:
            bool: True if language was changed successfully, False otherwise
        """
        try:
            if language_code == self.translator.get_current_language():
                # Already using this language, no need to change
                return True

            if self.translator.set_language(language_code):
                # Save preference
                self._save_language_preference(language_code)

                # Update all GUI text elements
                self._update_gui_translations()

                self.logger.info(
                    f"Language changed to {self.translator.get_language_name(language_code)}"
                )

                # Show confirmation to the user
                DialogHelper.show_message(
                    self.root,
                    self.t("Language Changed"),
                    self.t(
                        f"Application language has been changed to {self.translator.get_language_name(language_code)}."
                    ),
                    message_type="info",
                )

                return True
            return False
        except Exception as e:
            self.logger.error(f"Error changing language: {str(e)}")
            import traceback

            self.logger.error(traceback.format_exc())

            # Show error message to user
            DialogHelper.show_message(
                self.root,
                "Error",
                f"Error changing language: {str(e)}",
                message_type="error",
            )
            return False

    def _rebuild_menu(self):
        """
        Rebuild the menu with current translations.

        This method recreates the entire menu with translated text.
        """
        try:
            # Create menu definitions with updated translations
            menu_defs = {
                self.t("File"): [
                    {
                        "type": "command",
                        "label": self.gui_manager.get_theme_label(),
                        "command": self._toggle_theme,
                    },
                    {"type": "separator"},
                    {
                        "type": "command",
                        "label": self.t("Exit"),
                        "command": self.quit_app,
                    },
                ],
                self.t("Settings"): [
                    {
                        "type": "command",
                        "label": self.t("Data Settings"),
                        "command": self._open_data_settings,
                    },
                    {
                        "type": "command",
                        "label": self.t("Color Maps"),
                        "command": self._open_color_map_settings,
                    },
                    {"type": "separator"},
                    {
                        "type": "command",
                        "label": self.t("Classification Manager"),
                        "command": self._open_classification_manager,
                    },
                    {
                        "type": "command",
                        "label": self.t("Tag Manager"),
                        "command": self._open_tag_manager,
                    },
                ],
                self.t("Help"): [
                    {
                        "type": "command",
                        "label": self.t("Check for Updates"),
                        "command": self.on_check_for_updates,
                    },
                    {
                        "type": "command",
                        "label": self.t("About"),
                        "command": self._show_about_dialog,
                    },
                ],
            }

            # Add language options to menu definitions
            languages = self.translator.get_available_languages()
            language_items = []

            # Current language for highlighting
            current_lang = self.translator.get_current_language()

            # Create language menu items
            for lang_code in languages:
                lang_name = self.translator.get_language_name(lang_code)

                # Ensure each language gets its own command function to avoid closure issues
                # By using a function factory pattern
                def make_lang_command(code):
                    return lambda: self.change_language(code)

                language_items.append(
                    {
                        "type": "radiobutton",
                        "label": lang_name,
                        "value": lang_code,
                        "variable": self.language_var,
                        "command": make_lang_command(lang_code),
                    }
                )

            menu_defs[self.t("Language")] = language_items

            # Remove old menubar if it exists
            if hasattr(self, "menubar") and self.menubar:
                self.root.config(menu="")

            # Create new menubar
            self.menubar = self.gui_manager.setup_menubar(self.root, menu_defs)

            # Update language variable to match current language
            self.language_var.set(current_lang)
        except Exception as e:
            self.logger.error(f"Error rebuilding menu: {str(e)}")
            import traceback

            self.logger.error(traceback.format_exc())

    def on_check_for_updates(self):
        """Handle check for updates menu option."""
        if hasattr(self.app, "update_checker"):
            # Use check_and_update which shows appropriate dialogs
            self.app.update_checker.check_and_update(parent_window=self.root)

    def _show_about_dialog(self):
        """Show information about the application."""
        version = (
            self.update_checker.get_local_version()
            if hasattr(self, "update_checker")
            else "Unknown"
        )

        about_text = (
            f"GeoVue v{version}\n\n"
            "A tool to extract individual compartment images from\n"
            "panoramic chip tray photos using ArUco markers"
        )

        DialogHelper.show_message(
            self.root, "About GeoVue", about_text, message_type="info"
        )

    def _check_updates_at_startup(self):
        """Check for updates at startup without showing dialogs for up-to-date case."""
        if not hasattr(self, "update_checker"):
            return

        try:
            result = self.update_checker.compare_versions()

            if result["update_available"]:
                if DialogHelper.confirm_dialog(
                    self.root,
                    "Update Available",
                    f"A new version is available:\n{result['github_version']}.\n\nDownload and restart?",
                ):
                    self.update_checker.download_and_replace_script(self.file_manager)

        except Exception as e:
            self.logger.error(f"Error checking for updates on startup: {e}")

    def _toggle_blur_settings(self):
        """Enable or disable blur detection settings based on checkbox state."""
        if not hasattr(self, "blur_settings_controls"):
            return

        state = tk.NORMAL if self.blur_enable_var.get() else tk.DISABLED

        for widget in self.blur_settings_controls:
            self.gui_manager._apply_theme_to_widget(widget)
            for child in widget.winfo_children():
                try:
                    # Only configure widgets that accept 'state'
                    if hasattr(child, "configure"):
                        if (
                            "state" in child.configure()
                        ):  # Check if 'state' is a valid config option
                            child.configure(state=state)
                except Exception:
                    pass

    def _toggle_ocr_settings(self):
        """Enable/disable OCR settings based on checkbox state."""
        if not hasattr(self, "ocr_controls"):
            return

        state = tk.NORMAL if self.ocr_enable_var.get() else tk.DISABLED

        for widget in self.ocr_controls:
            self.gui_manager._apply_theme_to_widget(widget)
            for child in widget.winfo_children():
                try:
                    if hasattr(child, "configure") and "state" in child.configure():
                        child.configure(state=state)
                except Exception:
                    pass

    def _toggle_prefix_settings(self):
        """Enable/disable prefix settings based on checkbox state."""
        if not hasattr(self, "prefix_controls"):
            return

        # CHANGED: Only depend on prefix validation state, not OCR state
        state = tk.NORMAL if self.prefix_validation_var.get() else tk.DISABLED

        for widget in self.prefix_controls:
            self.gui_manager._apply_theme_to_widget(widget)
            for child in widget.winfo_children():
                try:
                    if hasattr(child, "configure"):
                        child.configure(state=state)
                except:
                    pass

    def _set_widget_state(self, widget, state):
        """
        Set the state of a widget and its children, handling different widget types.

        Args:
            widget: The widget to set state for
            state: 'normal' or 'disabled'
        """
        try:
            # Some widgets have direct state configuration
            if isinstance(
                widget,
                (ttk.Entry, ttk.Button, ttk.Scale, ttk.Combobox, ttk.Checkbutton),
            ):
                widget.configure(state=state)
            elif isinstance(widget, tk.Entry):
                if state == "disabled":
                    widget.configure(state="readonly")
                else:
                    widget.configure(state="normal")
            elif isinstance(widget, (tk.Button, tk.OptionMenu)):
                widget.configure(state=state)
            elif isinstance(widget, (tk.Text, tk.Canvas)):
                widget.configure(state=state)

            # Handle containers - for frames, apply to all children
            if isinstance(widget, (ttk.Frame, tk.Frame)):
                for child in widget.winfo_children():
                    self._set_widget_state(child, state)
        except Exception as e:
            # If widget doesn't support state config, log but don't crash
            self.logger.debug(f"Could not set state to {state} for {widget}: {str(e)}")

    def _browse_shared_path(
        self, string_var, entry_widget=None, is_file=False, path_key=None
    ):
        """Open browser and update path variable.

        Args:
            string_var: StringVar to update with selected path
            entry_widget: Optional entry widget to update color
            is_file: Whether browsing for file (True) or folder (False)
            path_key: The FileManager path key (e.g., 'approved_compartments', 'register_excel')
        """
        if is_file:
            # Determine file types based on path_key
            if path_key == "register_excel":
                title = self.t("Select Excel Register")
                filetypes = [("Excel files", "*.xlsx"), ("All files", "*.*")]
                initialfile = None
            else:
                # Generic file browser
                title = self.t("Select File")
                filetypes = [("All files", "*.*")]
                initialfile = None

            # Browse for file
            file_path = filedialog.askopenfilename(
                title=title,
                filetypes=filetypes,
                initialfile=initialfile if "initialfile" in locals() else None,
            )
            if file_path:
                string_var.set(file_path)
                # Update the entry background color if widget was provided
                if entry_widget and os.path.exists(file_path):
                    entry_widget.config(
                        bg=self.gui_manager.theme_colors["accent_valid"]
                    )
                # Update FileManager immediately if path_key provided
                if path_key and self.file_manager:
                    self.file_manager.update_shared_path(path_key, file_path)

        else:
            # Browse for folder
            folder_path = filedialog.askdirectory(title=self.t("Select shared folder"))
            if folder_path:
                string_var.set(folder_path)
                # Update the entry background color if widget was provided
                if entry_widget and os.path.exists(folder_path):
                    entry_widget.config(
                        bg=self.gui_manager.theme_colors["accent_valid"]
                    )
                # Update FileManager immediately if path_key provided
                if path_key and self.file_manager:
                    self.file_manager.update_shared_path(path_key, folder_path)

    def _create_new_register(self):
        """Create a new Excel register using JSONRegisterManager."""
        try:
            # Ask user where to save the register
            default_filename = "Chip_Tray_Register.xlsx"

            # Try to suggest the standard location
            suggested_dir = None
            if self.file_manager.shared_base_dir:
                register_folder = (
                    self.file_manager.shared_base_dir / "Chip Tray Register"
                )
                if register_folder.exists():
                    suggested_dir = str(register_folder)

            if suggested_dir:
                # Check if register already exists in suggested location
                if JSONRegisterManager.has_existing_data_static(suggested_dir):
                    summary = JSONRegisterManager.get_data_summary_static(suggested_dir)

                    existing_msg = self.t(
                        "A register already exists in this location with:"
                    )
                    if summary["has_excel"]:
                        existing_msg += f"\n- {self.t('Excel file')}"
                    if summary["compartment_count"] > 0:
                        existing_msg += f"\n- {summary['compartment_count']} {self.t('compartment records')}"
                    if summary["original_count"] > 0:
                        existing_msg += f"\n- {summary['original_count']} {self.t('original image records')}"

                    existing_msg += f"\n\n{self.t('Do you want to create a new register in a different location?')}"

                    if not DialogHelper.confirm_dialog(
                        self.root, self.t("Existing Register Found"), existing_msg
                    ):
                        return

                    # Clear suggested dir to force user to choose new location
                    suggested_dir = None

            file_path = filedialog.asksaveasfilename(
                title=self.t("Create New Excel Register"),
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                initialdir=suggested_dir,
                initialfile=default_filename,
            )

            if not file_path:
                return

            # Get the directory for JSONRegisterManager
            register_dir = os.path.dirname(file_path)

            # Check if any register files exist in this location
            existing_files = JSONRegisterManager.check_existing_files_static(
                register_dir
            )

            if any(existing_files.values()):
                # Build detailed message about what exists
                existing_items = []
                if existing_files["excel"]:
                    existing_items.append(self.t("Excel register file"))
                if existing_files["compartment_json"]:
                    existing_items.append(self.t("Compartment data"))
                if existing_files["original_json"]:
                    existing_items.append(self.t("Original image data"))
                if existing_files["data_folder"]:
                    existing_items.append(self.t("Register data folder"))

                message = (
                    self.t("The following items already exist in this location:") + "\n"
                )
                message += "\n".join(f"• {item}" for item in existing_items)
                message += f"\n\n{self.t('Creating a new register will overwrite these files.')}"
                message += f"\n\n{self.t('Do you want to continue?')}"

                if not DialogHelper.confirm_dialog(
                    self.root,
                    self.t("Overwrite Existing Register?"),
                    message,
                    yes_text=self.t("Overwrite"),
                    no_text=self.t("Cancel"),
                ):
                    return

                # Ask if user wants to create a backup
                if DialogHelper.confirm_dialog(
                    self.root,
                    self.t("Create Backup?"),
                    self.t(
                        "Would you like to create a backup of the existing register before overwriting?"
                    ),
                ):
                    self._create_register_backup(register_dir)

            # Create the register manager - this will create the Excel file from template
            register_manager = JSONRegisterManager(register_dir, self.logger)

            # ===================================================
            # FIXED: The JSONRegisterManager creates the Excel file with the correct name
            # We only need to handle the case where user chose a different filename
            # ===================================================
            expected_excel = os.path.join(register_dir, register_manager.EXCEL_FILE)

            # Check if the file was created successfully
            if not os.path.exists(expected_excel):
                DialogHelper.show_message(
                    self.root,
                    self.t("Error"),
                    self.t("Failed to create Excel register file."),
                    message_type="error",
                )
                return

            # If user chose a different filename than the default, rename it
            if os.path.basename(file_path) != register_manager.EXCEL_FILE:
                try:
                    # Remove the user's chosen path if it exists (from the save dialog)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    # Rename the created file to match user's choice
                    shutil.move(expected_excel, file_path)
                    self.logger.info(f"Renamed register file to: {file_path}")
                except Exception as e:
                    self.logger.error(f"Error renaming register file: {str(e)}")
                    DialogHelper.show_message(
                        self.root,
                        self.t("Error"),
                        self.t("Could not rename register file:") + f" {str(e)}",
                        message_type="error",
                    )
                    return
            else:
                # User chose the default name, use it as is
                file_path = expected_excel

            # Update the path in FileManager
            self.file_manager.update_shared_path("register_excel", file_path)

            # Update the GUI field
            self.register_path_var.set(file_path)

            # Find and update the entry widget color
            if hasattr(self, "shared_collapsible"):
                for frame in self.shared_collapsible.content_frame.winfo_children():
                    if isinstance(frame, ttk.Frame):
                        for child in frame.winfo_children():
                            if isinstance(child, tk.Entry) and child.cget(
                                "textvariable"
                            ) == str(self.register_path_var):
                                child.config(
                                    bg=self.gui_manager.theme_colors["accent_valid"]
                                )
                                break

            # Show success message
            DialogHelper.show_message(
                self.root,
                self.t("Success"),
                self.t(
                    "Excel register created successfully!\n\nThe register includes:\n- Compartment Register sheet\n- Original Images Register sheet\n- Power Query setup instructions\n\nData files are stored in the 'Register Data (Do not edit)' subfolder."
                ),
                message_type="info",
            )

            # Ask if user wants to open the file
            if DialogHelper.confirm_dialog(
                self.root,
                self.t("Open Register"),
                self.t("Would you like to open the Excel register now?"),
            ):
                try:
                    if platform.system() == "Windows":
                        os.startfile(file_path)
                    elif platform.system() == "Darwin":  # macOS
                        subprocess.Popen(["open", file_path])
                    else:  # Linux
                        subprocess.Popen(["xdg-open", file_path])
                except Exception as e:
                    self.logger.error(f"Could not open Excel file: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error creating new register: {str(e)}")
            DialogHelper.show_message(
                self.root,
                self.t("Error"),
                self.t("An error occurred creating the register:") + f" {str(e)}",
                message_type="error",
            )

    def _toggle_auto_loop_mode(self):
        """Handle auto-loop mode toggle"""
        if self.auto_loop_var.get():
            # Show information dialog
            message = (
                "Auto-Loop Mode will:\n\n"
                "✔ Process images with metadata in filename (e.g., BA001_0-20m.jpg)\n"
                "✔ Skip images missing compartment markers\n"
                "✔ Skip images with markers out of sequence\n"
                "✔ Use filename metadata instead of manual input\n\n"
                "Duplicate Handling:\n"
                "• Check 'Skip existing/duplicate intervals' to only extract NEW images\n"
                "• Uncheck to extract duplicates (wet/dry pairs) for manual selection\n\n"
                "Images that fail quality checks will be logged for manual review.\n\n"
                "Continue with Auto-Loop Mode?"
            )

            if not DialogHelper.confirm_dialog(self.root, "Auto-Loop Mode", message):
                self.auto_loop_var.set(False)
                return

            self.logger.info("Auto-Loop Mode enabled")
        else:
            self.logger.info("Auto-Loop Mode disabled")

    def start_processing(self):
        """Start processing in a scheduled manner on the main thread."""
        self.logger.debug("Entered start_processing()")

        # Check if auto-loop mode is enabled
        auto_loop_mode = (
            self.auto_loop_var.get() if hasattr(self, "auto_loop_var") else False
        )

        folder_path = self.folder_var.get()
        if not folder_path:
            DialogHelper.show_message(
                self.root, "Error", "Please select a folder", message_type="error"
            )
            return

        if not os.path.isdir(folder_path):
            DialogHelper.show_message(
                self.root,
                "Error",
                "Selected path is not a valid folder",
                message_type="error",
            )
            return

        # Update config with current GUI settings
        self.app.config["output_format"] = self.format_var.get()
        self.app.config["save_debug_images"] = self.debug_var.get()
        # self.app.config['compartment_interval'] = self.interval_var.get()

        # Update blur detection settings
        self.app.config["enable_blur_detection"] = self.blur_enable_var.get()
        self.app.config["blur_threshold"] = self.blur_threshold_var.get()
        self.app.config["blur_roi_ratio"] = self.blur_roi_var.get()
        self.app.config["flag_blurry_images"] = self.flag_blurry_var.get()
        self.app.config["save_blur_visualizations"] = self.save_blur_viz_var.get()
        self.app.config["blurry_threshold_percentage"] = (
            self.blur_threshold_pct_var.get()
        )

        # Update OCR settings
        self.app.config["enable_ocr"] = self.ocr_enable_var.get()
        self.app.config["enable_prefix_validation"] = self.prefix_validation_var.get()

        # Parse the prefix string into a list
        prefix_str = self.prefix_var.get()
        if prefix_str:
            # Split by comma and strip whitespace
            prefixes = [p.strip().upper() for p in prefix_str.split(",")]
            # Filter out any empty or invalid entries
            self.app.config["valid_hole_prefixes"] = [
                p for p in prefixes if p and len(p) == 2 and p.isalpha()
            ]

        # Update blur detector with new settings
        self.app.blur_detector.threshold = self.app.config["blur_threshold"]
        self.app.blur_detector.roi_ratio = self.app.config["blur_roi_ratio"]

        # Ensure TesseractManager has the updated config
        # self.app.tesseract_manager.config = self.app.config

        self.progress_var.set(0)
        self.app.processing_complete = False

        # Disable process button while processing
        if "process_button" in self.buttons:
            self.buttons["process_button"].set_state("disabled")
        else:
            self.logger.warning("Process button not found in buttons dictionary")

        # Create a processing indicator in the status text
        self.update_status(f"Started processing folder: {folder_path}", "info")

        # Process the first image immediately on the main thread, then schedule the rest with after()
        image_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
            and f.lower().endswith(
                (".jpg", ".HEIC", ".heic", ".jpeg", ".HEIC", ".png", ".tif", ".tiff")
            )
        ]

        if not image_files:
            self.update_status("No image files found in the selected folder", "warning")
            if "process_button" in self.buttons:
                self.buttons["process_button"].set_state("normal")
            return

        # Store the list of files and initialize counters
        self.files_to_process = image_files
        self.current_file_index = 0
        self.successful_count = 0
        self.failed_count = 0
        self.processing_complete = False

        # Start processing based on mode
        if auto_loop_mode:
            # Use batch processor for automatic processing
            self._start_auto_batch_processing()
        else:
            # Start the processing cycle with a single timer
            self.root.after(100, self._process_cycle)

    def _process_cycle(self):
        """Process cycle that handles one image at a time and only continues after completion."""
        current_thread = threading.current_thread()
        self.logger.debug(
            f"Process cycle executing in thread: {current_thread.name}, is main: {current_thread is threading.main_thread()}"
        )

        try:
            # Check if processing should continue
            if self.processing_complete:
                return

            # Check if there are more files to process
            if self.current_file_index < len(self.files_to_process):
                current_file = self.files_to_process[self.current_file_index]
                self.current_file_index += 1

                # Update the progress
                progress = (self.current_file_index / len(self.files_to_process)) * 100
                self.progress_queue.put(
                    ("Processing image: " + os.path.basename(current_file), progress)
                )

                # CHANGED: Process the current image with a callback
                # This is key - we'll use a callback to schedule the next image
                # only after this one is completely done
                self._process_current_image(current_file)
            else:
                # All images processed
                self._finish_processing()
        except Exception as e:
            self.logger.error(f"Error in process cycle: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.processing_complete = True

            # Re-enable process button
            if "process_button" in self.buttons:
                self.buttons["process_button"].set_state("normal")

    def _process_current_image(self, file_path):
        """Process a single image and schedule the next one only after completion."""
        try:
            # Process the image
            result = self.app.process_image(file_path)

            if result:
                # Successfully processed this image
                self.successful_count += 1
            else:
                # Processing failed or was canceled
                self.failed_count += 1
                self.update_status(
                    "Processing stopped or failed for file: "
                    + os.path.basename(file_path),
                    "error",
                )
                self.processing_complete = True

                # Re-enable process button
                if "process_button" in self.buttons:
                    self.buttons["process_button"].set_state("normal")
                return

            # Check progress
            self.check_progress(schedule_next=False)

            # CHANGED: Only schedule the next cycle after the current image is fully processed
            # This happens naturally at the end of this method
            if not self.processing_complete:
                self.root.after(100, self._process_cycle)
        except Exception as e:
            self.logger.error(f"Error processing image {file_path}: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.failed_count += 1
            self.processing_complete = True

            # Re-enable process button
            if "process_button" in self.buttons:
                self.buttons["process_button"].set_state("normal")

    def _finish_processing(self):
        """Clean up after all files have been processed."""
        # Update progress display
        summary_msg = f"Processing complete: {self.successful_count} successful, {self.failed_count} failed"
        self.logger.info(summary_msg)
        self.update_status(summary_msg, "success")
        self.progress_var.set(100)

        # Re-enable process button
        if "process_button" in self.buttons:
            self.buttons["process_button"].set_state("normal")

        # Prompt to start QAQC review if there are successfully processed images
        if self.successful_count > 0:

            def show_qaqc_prompt():
                start_review = DialogHelper.confirm_dialog(
                    self.root,
                    DialogHelper.t("Processing Complete"),
                    DialogHelper.t(
                        f"Successfully processed {self.successful_count} images.\n\nWould you like to start the QAQC review process now?"
                    ),
                )
                if start_review and hasattr(self.app, "qaqc_manager"):
                    self.app.qaqc_manager.start_review_process()

            # Use after() to ensure the messagebox appears after GUI updates
            self.root.after(500, show_qaqc_prompt)

    def _start_image_review(self):
        """Start the image review process for pending trays."""
        try:
            # QAQC manager should already be initialized in app
            if not hasattr(self.app, "qaqc_manager") or self.app.qaqc_manager is None:
                self.logger.error("QAQC manager not initialized in app")
                DialogHelper.show_message(
                    self.root,
                    self.t("Error"),
                    self.t("QAQC manager not properly initialized"),
                    message_type="error",
                )
                return

            # Use the QAQC manager from app
            qaqc_manager = self.app.qaqc_manager

            # Set main GUI reference for status updates if needed
            if not hasattr(qaqc_manager, "main_gui") or qaqc_manager.main_gui is None:
                qaqc_manager.set_main_gui(self)

            # Create a simple "Loading..." dialog that shows immediately
            loading_dialog = tk.Toplevel(self.root)
            loading_dialog.title("Loading Review")
            loading_dialog.transient(self.root)
            loading_dialog.grab_set()

            # Center it
            window_width = 300
            window_height = 100
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            x = (screen_width - window_width) // 2
            y = (screen_height - window_height) // 2
            loading_dialog.geometry(f"{window_width}x{window_height}+{x}+{y}")

            # Add message
            frame = ttk.Frame(loading_dialog, padding="20")
            frame.pack(fill=tk.BOTH, expand=True)
            ttk.Label(
                frame,
                text="Scanning review folders...\nPlease wait...",
                justify=tk.CENTER,
            ).pack(expand=True)

            # Prevent closing
            loading_dialog.protocol("WM_DELETE_WINDOW", lambda: None)

            # Force display
            loading_dialog.update()

            def run_review_process():
                """Run on main thread after dialog shows."""
                try:
                    # Start the review process - this will check Temp_Review folder automatically
                    qaqc_manager.start_review_process()
                finally:
                    # Close loading dialog
                    if loading_dialog.winfo_exists():
                        loading_dialog.destroy()

            # Schedule on main thread after 50ms (lets dialog appear first)
            self.root.after(50, run_review_process)

        except Exception as e:
            self.logger.error(f"Error starting image review: {str(e)}")
            DialogHelper.show_message(
                self.root,
                self.t("Error"),
                self.t("An error occurred starting the review:") + f" {str(e)}",
                message_type="error",
            )

    def _start_logging_review(self):
        """Open the logging review dialog with register synchronization."""
        try:
            # Set wait cursor
            self.root.config(cursor="wait")
            self.root.update_idletasks()

            # Set cursor on all child widgets
            def set_cursor_recursive(widget):
                try:
                    widget.config(cursor="wait")
                except:
                    pass
                for child in widget.winfo_children():
                    set_cursor_recursive(child)

            set_cursor_recursive(self.root)

            # Continue with opening logging review dialog
            self.logger.info("Opening logging review dialog...")

            # Check register path
            register_path = self.file_manager.get_shared_path(
                "register_excel", create_if_missing=False
            )
            if not register_path:
                DialogHelper.show_message(
                    self.root,
                    DialogHelper.t("Error"),
                    DialogHelper.t("Could not find register path"),
                    message_type="error",
                )
                return

            # Now open the logging review dialog
            from gui.logging_review_dialog import LoggingReviewDialog

            dialog = LoggingReviewDialog(
                self.root, self.file_manager, self.gui_manager, self.config_manager
            )

            # Set JSON manager if needed
            base_path = str(register_path.parent)
            if JSONRegisterManager.has_existing_data_static(base_path):
                dialog.json_manager = JSONRegisterManager(base_path, self.logger)

            dialog.show()

        except Exception as e:
            self.logger.error(f"Error in _start_logging_review: {str(e)}")
            self.logger.error(traceback.format_exc())
            DialogHelper.show_message(
                self.root,
                DialogHelper.t("Error"),
                DialogHelper.t("Failed to open logging review:") + f"\n{str(e)}",
                message_type="error",
            )
        finally:
            # Reset cursor on all widgets
            def reset_cursor_recursive(widget):
                try:
                    widget.config(cursor="")
                except:
                    pass
                for child in widget.winfo_children():
                    reset_cursor_recursive(child)

            reset_cursor_recursive(self.root)

    def _open_logging_review_report(self):
        """Open the logging review report dialog."""
        try:
            data_coordinator = getattr(self.app, "data_coordinator", None)
            if not data_coordinator or not data_coordinator.is_initialized:
                DialogHelper.show_message(
                    self.root,
                    self.t("Data Not Loaded"),
                    self.t("Data is not loaded. Please load datasets before generating reports."),
                    message_type="warning",
                )
                return

            from gui.logging_review_report_dialog import LoggingReviewReportDialog

            LoggingReviewReportDialog(
                parent=self.root,
                gui_manager=self.gui_manager,
                data_coordinator=data_coordinator,
                translator=self.t,
            )
        except Exception as e:
            self.logger.error(f"Error opening logging review report dialog: {str(e)}")
            DialogHelper.show_message(
                self.root,
                self.t("Error"),
                self.t("Failed to open logging review report dialog.") + f"\n{str(e)}",
                message_type="error",
            )

    def _check_shared_paths(self):
        """Check if shared folder paths are available and accessible."""
        # Check register path using FileManager
        register_path = self.file_manager.get_shared_path(
            "register_excel", create_if_missing=False
        )
        if not register_path or not register_path.exists():
            DialogHelper.show_message(
                self.root,
                self.t("Shared Folder Path Error"),
                self.t(
                    "Excel register path is not available. Please check shared folder settings."
                ),
                message_type="error",
            )
            return False

        # Check approved folder path using FileManager
        approved_path = self.file_manager.get_shared_path(
            "approved_compartments", create_if_missing=False
        )
        if not approved_path or not approved_path.exists():
            DialogHelper.show_message(
                self.root,
                self.t("Shared Folder Path Error"),
                self.t(
                    "Approved folder path is not available. Please check shared folder settings."
                ),
                message_type="error",
            )
            return False

        return True

    def _show_file_structure_info(self):
        """Show information about the new file structure."""
        info_message = (
            "Files are saved in a centralized location with the following structure:\n\n"
            f"{self.file_manager.dir_structure['processed_originals']}\n"
            "├── Blur Analysis\\\n"
            "├── Chip Compartments\\\n"
            "├── Debug Images\\\n"
            "├── Drill Traces\\\n"
            "├── Processed Originals\\\n"
            "└── Failed and Skipped Originals\\\n\n"
            "Each folder (except Drill Traces) contains subfolders for each project code [first two characters of the HoleIDs] and each sorted Hole ID.\n\n"
            "File naming follows the pattern:\n"
            "- Debug: HoleID_From-To_Debug_[type].jpg\n"
            "- Compartments: HoleID_CC_[number].png\n"
            "- Blur Analysis: HoleID_[number]_blur_analysis.jpg\n"
            "- Originals: HoleID_From-To_Original.ext"
        )
        DialogHelper.show_message(
            self.root, "File Structure Information", info_message, message_type="info"
        )

    def _open_embedding_dialog(self):
        """Open the embedding training dialog."""
        try:
            dialog = EmbeddingTrainingDialog(
                self.root, self.gui_manager, self.file_manager
            )
        except Exception as e:
            self.logger.error(f"Error opening embedding dialog: {str(e)}")
            DialogHelper.show_message(
                self.root,
                self.t("Error"),
                self.t("Failed to open embedding tool") + f"\n{str(e)}",
                message_type="error",
            )
    
    def _open_drillhole_correlation(self):
        """Open the drillhole correlation dialog."""
        try:
            import pandas as pd
            
            # Get collar data from DataCoordinator (already loaded at startup)
            self.logger.info("Loading collar data...")
            
            collar_data = pd.DataFrame()
            data_coordinator = getattr(self.app, 'data_coordinator', None)
            self.logger.debug(f"[COLLAR] data_coordinator from app: {data_coordinator is not None}")
            
            if data_coordinator and data_coordinator.is_initialized:
                # Get collar data from geological_store - look for 'excollar' or similar
                geo_store = data_coordinator.geological_store
                self.logger.debug(f"[COLLAR] geological_store: {geo_store is not None}")
                
                if geo_store:
                    # Try to find collar dataset
                    available_sources = geo_store.list_sources()
                    self.logger.debug(f"[COLLAR] Available data sources: {available_sources}")
                    
                    for source_name in ['excollar', 'collar', 'collars']:
                        if source_name in available_sources:
                            source = geo_store.get_source(source_name)
                            if source and source.df is not None:
                                collar_data = source.df.copy()
                                self.logger.info(f"[COLLAR] Got collar data from '{source_name}': {len(collar_data)} rows")
                                break
                    
                    if collar_data.empty:
                        self.logger.warning(f"[COLLAR] No collar source found in: {available_sources}")
            else:
                self.logger.warning(f"[COLLAR] DataCoordinator not available or not initialized")
            
            if collar_data.empty:
                # No collar data available - show error
                from gui.dialog_helper import DialogHelper
                self.logger.error("[COLLAR] No collar data found - cannot open correlation dialog")
                DialogHelper.show_message(
                    self.root,
                    self.t("No Collar Data"),
                    self.t("No collar data available.\n\n"
                           "Please ensure 'excollar.csv' (or a collar dataset with HOLEID, X, Y columns) "
                           "is in the 'Drillhole Datasets' folder."),
                    message_type="error",
                )
                return
            
            self.logger.info(f"Loaded {len(collar_data)} collars from file")
            
            # Create dialog helper instance
            from gui.dialog_helper import DialogHelper
            dialog_helper = DialogHelper(self.root, self.translator)
            
            # Get previous selection if it exists (for remembering state)
            initial_selection = getattr(self, '_last_drillhole_selection', None)
            if initial_selection:
                # Extract internal data from previous result
                initial_selection = initial_selection.get('_internal')
            
            # Get initial holes from last selection if available
            initial_holes = None
            if hasattr(self, '_last_drillhole_selection') and self._last_drillhole_selection:
                initial_holes = self._last_drillhole_selection.get('hole_ids', [])
                self.logger.info(f"Using {len(initial_holes)} holes from last selection")
            
            # Initialize color map manager if not already available
            color_map_manager = None
            if hasattr(self, 'color_map_manager'):
                color_map_manager = self.color_map_manager
            else:
                try:
                    from processing.LoggingReviewStep.color_map_manager import ColorMapManager
                    color_map_manager = ColorMapManager(self.config_manager)
                    self.logger.info("Created ColorMapManager for correlation")
                except ImportError:
                    self.logger.warning("ColorMapManager not available - data visualization will be limited")
            
            # Data is already loaded in data_coordinator at startup
            # Just verify it's available
            if data_coordinator:
                stats = data_coordinator.get_stats()
                self.logger.info(f"[COLLAR] DataCoordinator has {stats.get('geological_store', {}).get('total_rows', 0)} rows loaded")
            else:
                self.logger.warning("[COLLAR] DataCoordinator not available - some features may be limited")
            
            # Open correlation dialog with all required managers
            self.logger.info("Creating CorrelationDialog...")
            self.logger.debug(f"[COLLAR] Passing data_coordinator as data_manager: {data_coordinator is not None}")
            
            dialog = CorrelationDialog(
                parent=self.root,
                gui_manager=self.gui_manager,
                data_manager=data_coordinator,
                file_manager=self.file_manager,
                config_manager=self.config_manager,
                color_map_manager=color_map_manager,
                translator=self.translator,
                dialog_helper=dialog_helper,
                initial_holes=initial_holes
            )
            
            self.logger.info("CorrelationDialog opened successfully")
                
        except Exception as e:
            self.logger.error(f"Error opening drillhole selector: {str(e)}")
            import traceback
            traceback.print_exc()
            
            from gui.dialog_helper import DialogHelper
            DialogHelper.show_message(
                self.root,
                self.t("Error"),
                self.t("Failed to open drillhole selector") + f"\n{str(e)}",
                message_type="error",
            )

    def _start_auto_batch_processing(self):
        """Start automatic batch processing with quality checks"""
        import threading

        try:
            # Import batch processor
            from processing.batch_processor import BatchProcessor

            # Get duplicate handling preference
            skip_duplicates = (
                self.auto_skip_duplicates_var.get()
                if hasattr(self, "auto_skip_duplicates_var")
                else False
            )

            # Create batch processor instance
            batch_processor = BatchProcessor(
                self.app, self.config, skip_duplicates=skip_duplicates
            )

            # Create progress dialog (non-modal to allow duplicate dialogs to interrupt)
            progress_dialog = ProgressDialog(
                self.root,
                "Auto Batch Processing",
                f"Processing {len(self.files_to_process)} images...",
                modal=False,  # Allow duplicate dialogs to show over this
            )

            # Store reference so we can check cancellation
            self._batch_processor = batch_processor
            self._batch_progress_dialog = progress_dialog
            self._batch_stats = None
            self._batch_complete = False

            def update_progress(message, percent, status=""):
                """Update progress dialog - thread-safe"""
                # Check for cancellation from dialog
                if progress_dialog.is_cancelled():
                    batch_processor.request_cancel()

                # Update UI via after() for thread safety
                self.root.after(0, lambda: progress_dialog.update_progress(message, percent, status))

            def run_batch():
                """Run batch processing in background thread"""
                try:
                    self._batch_stats = batch_processor.process_batch(
                        self.files_to_process, progress_callback=update_progress
                    )
                except Exception as e:
                    self.logger.error(f"Batch processing error: {e}")
                    self._batch_stats = {"error": str(e), "processed": 0, "failed": 0, "skipped": 0, "total": 0}
                finally:
                    self._batch_complete = True
                    # Schedule completion handler on main thread
                    self.root.after(0, self._on_batch_complete)

            # Start batch processing in background thread
            self.logger.info(
                f"Starting auto batch processing of {len(self.files_to_process)} images"
            )

            batch_thread = threading.Thread(target=run_batch, daemon=True)
            batch_thread.start()

        except Exception as e:
            self.logger.error(f"Error starting auto batch processing: {str(e)}")
            self.logger.error(traceback.format_exc())

            DialogHelper.show_message(
                self.root,
                "Error",
                f"Failed to start batch processing:\n{str(e)}",
                message_type="error",
            )

            # Re-enable process button
            if "process_button" in self.buttons:
                self.buttons["process_button"].set_state("normal")

    def _on_batch_complete(self):
        """Handle batch processing completion on main thread"""
        try:
            # Close progress dialog
            if hasattr(self, '_batch_progress_dialog') and self._batch_progress_dialog:
                self._batch_progress_dialog.close()

            stats = getattr(self, '_batch_stats', None) or {}

            # Check for error
            if "error" in stats:
                DialogHelper.show_message(
                    self.root,
                    "Error",
                    f"Batch processing failed:\n{stats['error']}",
                    message_type="error",
                )
                if "process_button" in self.buttons:
                    self.buttons["process_button"].set_state("normal")
                return

            # Update counts
            self.successful_count = stats.get("processed", 0)
            self.failed_count = stats.get("failed", 0)

            # Generate and show report
            batch_processor = getattr(self, '_batch_processor', None)
            if batch_processor:
                report = batch_processor.get_processing_report()

                # Save report to file
                report_path = os.path.join(
                    self.config.get("local_folder_path", "."), "batch_processing_report.txt"
                )
                with open(report_path, "w") as f:
                    f.write(report)

            # Build summary message
            was_cancelled = stats.get("cancelled", False)
            if was_cancelled:
                summary_message = (
                    f"Batch Processing Cancelled\n\n"
                    f"Completed before cancellation:\n"
                    f"Successfully Processed: {stats.get('processed', 0)}\n"
                    f"Skipped (Quality Issues): {stats.get('skipped', 0)}\n"
                    f"Failed: {stats.get('failed', 0)}\n"
                )
            else:
                summary_message = (
                    f"Auto Batch Processing Complete!\n\n"
                    f"Total Images: {stats.get('total', 0)}\n"
                    f"Successfully Processed: {stats.get('processed', 0)}\n"
                    f"Skipped (Quality Issues): {stats.get('skipped', 0)}\n"
                    f"Failed: {stats.get('failed', 0)}\n\n"
                )

            if stats.get("skipped", 0) > 0:
                summary_message += (
                    f"Skipped files require manual processing.\n"
                    f"See report for details: {report_path}\n\n"
                )

            if not was_cancelled:
                summary_message += "Would you like to review skipped images now?"

            if was_cancelled or stats.get("skipped", 0) == 0:
                DialogHelper.show_message(
                    self.root, "Batch Processing", summary_message, message_type="info"
                )
                if "process_button" in self.buttons:
                    self.buttons["process_button"].set_state("normal")
                if self.successful_count > 0:
                    self.root.after(500, lambda: self._prompt_qaqc_review())
            else:
                review_skipped = DialogHelper.confirm_dialog(
                    self.root, "Batch Processing Complete", summary_message
                )

                if review_skipped and batch_processor and batch_processor.skipped_files:
                    # Switch to manual mode for skipped files
                    self.auto_loop_var.set(False)
                    self.files_to_process = batch_processor.skipped_files
                    self.current_file_index = 0
                    self.successful_count = 0
                    self.failed_count = 0
                    self.processing_complete = False

                    # Start normal processing for skipped files
                    self._process_cycle()
                else:
                    # Re-enable process button
                    if "process_button" in self.buttons:
                        self.buttons["process_button"].set_state("normal")

                    # Prompt for QAQC if successful
                    if self.successful_count > 0:
                        self.root.after(500, lambda: self._prompt_qaqc_review())

        except Exception as e:
            self.logger.error(f"Error in batch completion handler: {str(e)}")
            self.logger.error(traceback.format_exc())

            DialogHelper.show_message(
                self.root,
                "Error",
                f"Error completing batch processing:\n{str(e)}",
                message_type="error",
            )

            if "process_button" in self.buttons:
                self.buttons["process_button"].set_state("normal")

    def _prompt_qaqc_review(self):
        """Prompt to start QAQC review after processing"""
        start_review = DialogHelper.confirm_dialog(
            self.root,
            DialogHelper.t("Processing Complete"),
            DialogHelper.t(
                f"Successfully processed {self.successful_count} images.\n\n"
                f"Would you like to start the QAQC review process now?"
            ),
        )
        if start_review and hasattr(self.app, "qaqc_manager"):
            self.app.qaqc_manager.start_review_process()

    
    def _sync_to_cloud(self):
        """Sync local files to cloud storage with UID verification."""
        self.logger.info("=== SYNC TO CLOUD STARTED ===")
        
        try:
            # Check if cloud storage is configured FIRST (this is fast)
            if not self.file_manager.shared_paths:
                DialogHelper.show_message(
                    self.root,
                    self.t("Error"),
                    self.t(
                        "No cloud storage configured. Please configure shared folders first."
                    ),
                    message_type="error",
                )
                return

            # Show progress dialog IMMEDIATELY - before any blocking operations
            self.update_status("Preparing cloud sync...", "info")
            
            checking_dialog = ProgressDialog(
                self.root,
                self.t("Sync to Cloud"),
                self.t("Preparing..."),
                modal=False  # Don't use modal to avoid grab_set() issues
            )
            
            # Force the dialog to appear NOW
            checking_dialog.update_progress(self.t("Initializing..."), 2)
            self.root.update_idletasks()
            self.root.update()
            
            self.logger.debug("Progress dialog shown")
            
            # Container for background thread results
            result_container = {
                "summary": None, 
                "error": None, 
                "completed": False,
                "dedup_results": None
            }
            
            def progress_callback(msg, pct):
                """Thread-safe progress callback."""
                def update_ui():
                    try:
                        if checking_dialog and checking_dialog.dialog and checking_dialog.dialog.winfo_exists():
                            checking_dialog.update_progress(msg, pct)
                    except Exception:
                        pass
                self.root.after_idle(update_ui)
            
            def background_prepare():
                """Background task: run deduplication and gather summary."""
                self.logger.info("Background: Starting preparation...")

                try:
                    # Step 1: Run deduplication check (optimized - only checks potential duplicates)
                    progress_callback(self.t("Running deduplication check..."), 5)
                    try:
                        dedup_manager = UIDDeduplicationManager(self.file_manager, self.logger)
                        dedup_results = dedup_manager.find_and_remove_duplicates(dry_run=False)
                        result_container["dedup_results"] = dedup_results
                        self.logger.info(f"Background: Deduplication complete - {dedup_results}")
                    except Exception as e:
                        self.logger.warning(f"Background: Deduplication failed - {e}")

                    progress_callback(self.t("Preparation complete"), 20)
                    
                    # Step 2: Create sync manager and get summary (20% - 100%)
                    progress_callback(self.t("Scanning files..."), 25)
                    
                    from utils.cloud_sync_manager import CloudSyncManager
                    sync_manager = CloudSyncManager(self.file_manager, self.logger)
                    
                    # Get summary with progress updates
                    def summary_progress(msg, pct):
                        # Scale from 25% to 95%
                        scaled_pct = 25 + int(pct * 0.70)
                        progress_callback(msg, scaled_pct)
                    
                    summary = sync_manager.get_sync_summary(progress_callback=summary_progress)
                    result_container["summary"] = summary
                    result_container["sync_manager"] = sync_manager
                    
                    self.logger.info(f"Background: Summary complete - {summary}")
                    progress_callback(self.t("Scan complete"), 100)
                    
                except Exception as e:
                    self.logger.error(f"Background: Error - {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    result_container["error"] = str(e)
                finally:
                    result_container["completed"] = True
                    self.logger.info("Background: Finished")
            
            # Start background thread
            self.logger.debug("Starting background preparation thread...")
            prep_thread = threading.Thread(target=background_prepare, daemon=True)
            prep_thread.start()
            
            # Poll for completion using after()
            def check_preparation_complete():
                if result_container["completed"]:
                    self.logger.debug("Preparation complete, closing dialog...")
                    try:
                        checking_dialog.close()
                    except Exception:
                        pass
                    
                    # Process results on main thread
                    self._process_sync_preparation(result_container)
                else:
                    # Check again in 100ms
                    self.root.after(100, check_preparation_complete)
            
            # Start polling
            self.root.after(100, check_preparation_complete)
            
        except Exception as e:
            self.logger.error(f"Error in cloud sync: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            DialogHelper.show_message(
                self.root,
                self.t("Error"),
                self.t("An error occurred during sync:") + f"\n{str(e)}",
                message_type="error",
            )

    def _process_sync_preparation(self, result_container):
        """Process the sync preparation results and show confirmation dialog."""
        self.logger.debug("Processing sync preparation results...")
        
        try:
            # Handle dedup results
            dedup_results = result_container.get("dedup_results")
            if dedup_results:
                if dedup_results.get("files_removed", 0) > 0:
                    self.logger.info(
                        f"Deduplication removed {dedup_results['files_removed']} files, "
                        f"saved {dedup_results.get('bytes_saved', 0) / (1024*1024):.2f} MB"
                    )
                if dedup_results.get("uid_conflicts", 0) > 0:
                    self.update_status(
                        f"⚠ Found {dedup_results['uid_conflicts']} UID conflicts requiring manual review",
                        "warning",
                    )
            
            # Handle errors
            if result_container.get("error"):
                DialogHelper.show_message(
                    self.root,
                    self.t("Error"),
                    self.t("Error checking sync status:") + f"\n{result_container['error']}",
                    message_type="error",
                )
                return
            
            summary = result_container.get("summary")
            sync_manager = result_container.get("sync_manager")
            
            if not summary:
                DialogHelper.show_message(
                    self.root,
                    self.t("Error"),
                    self.t("Failed to get sync summary"),
                    message_type="error",
                )
                return
            
            # Check if cloud is configured
            if not summary.get("cloud_configured", False):
                DialogHelper.show_message(
                    self.root,
                    self.t("Error"),
                    self.t("No cloud storage configured. Please configure shared folders first."),
                    message_type="error",
                )
                return

            # Build confirmation message
            message_lines = [self.t("Files to sync to cloud:"), ""]
            total_files = 0
            total_mb = 0

            if summary["temp_review_files"] > 0:
                message_lines.append(
                    f"• {self.t('Review compartments')}: {summary['temp_review_files']} files "
                    f"({summary['temp_review_size_mb']} MB)"
                )
                total_files += summary["temp_review_files"]
                total_mb += summary["temp_review_size_mb"]

            if summary.get("temp_review_would_skip", 0) > 0:
                message_lines.append(
                    f"  ({summary['temp_review_would_skip']} will be skipped - already in cloud)"
                )

            if summary["approved_uploaded_files"] > 0:
                message_lines.append(
                    f"• {self.t('Approved compartments (already in cloud)')}: "
                    f"{summary['approved_uploaded_files']} files "
                    f"({summary['approved_uploaded_size_mb']} MB)"
                )
                message_lines.append(
                    f"  {self.t('(will be deleted locally to free space)')}"
                )
                total_files += summary["approved_uploaded_files"]
                total_mb += summary["approved_uploaded_size_mb"]

            if summary.get("approved_originals_uploaded_files", 0) > 0:
                message_lines.append(
                    f"• {self.t('Approved original images (already in cloud)')}: "
                    f"{summary['approved_originals_uploaded_files']} files "
                    f"({summary['approved_originals_uploaded_size_mb']} MB)"
                )
                message_lines.append(
                    f"  {self.t('(will be deleted locally to free space)')}"
                )
                total_files += summary["approved_originals_uploaded_files"]
                total_mb += summary["approved_originals_uploaded_size_mb"]

            if summary.get("rejected_originals_uploaded_files", 0) > 0:
                message_lines.append(
                    f"• {self.t('Rejected original images (already in cloud)')}: "
                    f"{summary['rejected_originals_uploaded_files']} files "
                    f"({summary['rejected_originals_uploaded_size_mb']} MB)"
                )
                message_lines.append(
                    f"  {self.t('(will be deleted locally to free space)')}"
                )
                total_files += summary["rejected_originals_uploaded_files"]
                total_mb += summary["rejected_originals_uploaded_size_mb"]

            if total_files == 0:
                DialogHelper.show_message(
                    self.root,
                    self.t("Nothing to Sync"),
                    self.t("No files need to be synced to cloud storage."),
                    message_type="info",
                )
                return

            action_descriptions = []
            if summary["temp_review_files"] > 0:
                action_descriptions.append(
                    self.t("Review compartments will be moved to cloud and deleted locally.")
                )
            if summary["approved_uploaded_files"] > 0:
                action_descriptions.append(
                    self.t("Already-uploaded files will be deleted locally to free space.")
                )

            message_lines.extend([
                "",
                f"{self.t('Total')}: {total_files} files ({total_mb:.1f} MB)",
                "",
            ])
            message_lines.extend(action_descriptions)
            message_lines.append(self.t("Continue?"))

            if not DialogHelper.confirm_dialog(
                self.root, self.t("Sync to Cloud"), "\n".join(message_lines)
            ):
                self.update_status("Cloud sync cancelled", "info")
                return

            # Run the actual sync
            self._run_cloud_sync(sync_manager)
            
        except Exception as e:
            self.logger.error(f"Error processing sync preparation: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            DialogHelper.show_message(
                self.root,
                self.t("Error"),
                self.t("An error occurred:") + f"\n{str(e)}",
                message_type="error",
            )

    def _run_cloud_sync(self, sync_manager):
        """Run the actual cloud sync operation in background."""
        self.logger.info("Starting cloud sync operation...")
        
        progress_dialog = ProgressDialog(
            self.root,
            self.t("Syncing to Cloud"),
            self.t("Starting sync..."),
            modal=False
        )
        
        result_container = {"result": None, "completed": False}

        def sync_progress_callback(msg, pct):
            def update_gui():
                try:
                    if progress_dialog and progress_dialog.dialog and progress_dialog.dialog.winfo_exists():
                        progress_dialog.update_progress(msg, pct)
                    if msg and self.root.winfo_exists():
                        self.update_status(msg, "info")
                except Exception as e:
                    self.logger.warning(f"GUI update error: {e}")
            self.root.after_idle(update_gui)

        def background_sync():
            try:
                result = sync_manager.sync_compartments_to_cloud(
                    progress_callback=sync_progress_callback
                )
                result_container["result"] = result
            except Exception as e:
                self.logger.error(f"Background sync error: {e}")
                result_container["result"] = {"success": False, "error": str(e)}
            finally:
                result_container["completed"] = True

        sync_thread = threading.Thread(target=background_sync, daemon=True)
        sync_thread.start()

        def check_sync_complete():
            if result_container["completed"]:
                try:
                    progress_dialog.close()
                except Exception:
                    pass
                self._handle_sync_completion(result_container["result"])
            else:
                self.root.after(100, check_sync_complete)

        self.root.after(100, check_sync_complete)

    def _handle_sync_completion(self, result):
        """Handle sync completion on main thread."""
        try:
            if result["success"]:
                stats = result["stats"]

                # First, update the status box with detailed results
                self.update_status("=" * 50, "info")
                self.update_status(self.t("CLOUD SYNC DETAILED RESULTS:"), "success")
                self.update_status("=" * 50, "info")

                # Show what was moved
                if stats["temp_moved"] > 0:
                    self.update_status(
                        f"✓ Moved {stats['temp_moved']} review compartments to cloud",
                        "success",
                    )
                if stats["approved_moved"] > 0:
                    self.update_status(
                        f"✓ Uploaded {stats['approved_moved']} approved compartments",
                        "success",
                    )
                if stats.get("approved_cleaned", 0) > 0:
                    self.update_status(
                        f"✓ Cleaned up {stats['approved_cleaned']} uploaded compartments locally",
                        "success",
                    )
                if stats.get("approved_originals_cleaned", 0) > 0:
                    self.update_status(
                        f"✓ Cleaned up {stats['approved_originals_cleaned']} uploaded original images",
                        "success",
                    )
                if stats.get("rejected_originals_cleaned", 0) > 0:
                    self.update_status(
                        f"✓ Cleaned up {stats['rejected_originals_cleaned']} rejected originals",
                        "success",
                    )
                if stats["already_in_cloud"] > 0:
                    self.update_status(
                        f"• Skipped {stats['already_in_cloud']} files (already in cloud)",
                        "info",
                    )
                if stats.get("mb_freed", 0) > 0:
                    self.update_status(
                        f"💾 Total space freed: {stats['mb_freed']:.2f} MB",
                        "success",
                    )

                # Show missing cloud files as warnings
                if stats.get("missing_cloud_files", 0) > 0:
                    self.update_status(
                        f"⚠ Files marked as uploaded but missing from cloud: {stats['missing_cloud_files']}",
                        "warning",
                    )
                    self.update_status(
                        "  These files need manual review - they may have been deleted from cloud storage",
                        "warning",
                    )

                # Show any errors
                if stats.get("errors"):
                    self.update_status("⚠ Errors encountered:", "warning")
                    for error in stats["errors"][:5]:  # Show first 5 errors
                        self.update_status(f"  - {error}", "warning")
                    if len(stats["errors"]) > 5:
                        self.update_status(
                            f"  ... and {len(stats['errors']) - 5} more errors",
                            "warning",
                        )

                self.update_status("=" * 50, "info")

                # Build result message for dialog
                result_lines = [self.t("Cloud sync completed:"), ""]

                if stats["temp_moved"] > 0:
                    result_lines.append(
                        f"✓ {self.t('Review compartments moved')}: {stats['temp_moved']}"
                    )
                if stats["approved_moved"] > 0:
                    result_lines.append(
                        f"✓ {self.t('Approved compartments moved')}: {stats['approved_moved']}"
                    )
                if stats.get("approved_cleaned", 0) > 0:
                    result_lines.append(
                        f"✓ {self.t('Uploaded files cleaned up locally')}: {stats['approved_cleaned']}"
                    )
                if stats["already_in_cloud"] > 0:
                    result_lines.append(
                        f"• {self.t('Skipped (already in cloud)')}: {stats['already_in_cloud']}"
                    )
                if stats.get("mb_freed", 0) > 0:
                    result_lines.append(
                        f"\n{self.t('Space freed')}: {stats['mb_freed']:.1f} MB"
                    )

                # Show failures and warnings
                total_failures = (
                    stats.get("temp_failed", 0)
                    + stats.get("approved_failed", 0)
                    + stats.get("approved_originals_failed", 0)
                    + stats.get("rejected_originals_failed", 0)
                )

                if total_failures > 0:
                    result_lines.append(
                        f"\n⚠ {self.t('Failed uploads')}: {total_failures} files"
                    )

                if stats.get("missing_cloud_files", 0) > 0:
                    result_lines.append(
                        f"⚠ {self.t('Missing from cloud')}: {stats['missing_cloud_files']} files"
                    )
                    result_lines.append(
                        f"  {self.t('(marked as uploaded but not found in cloud)')}"
                    )

                if stats.get("errors"):
                    result_lines.append(
                        f"\n{self.t('See status box for error details')}"
                    )

                DialogHelper.show_message(
                    self.root,
                    self.t("Sync Complete"),
                    "\n".join(result_lines),
                    message_type="info",
                )
            else:
                DialogHelper.show_message(
                    self.root,
                    self.t("Sync Error"),
                    result.get("error", "Unknown error"),
                    message_type="error",
                )

        except Exception as e:
            self.logger.error(f"Error handling sync completion: {str(e)}")
            DialogHelper.show_message(
                self.root,
                self.t("Error"),
                self.t("Error processing sync results: ") + str(e),
                message_type="error",
            )

    def _show_blur_help(self):
        """Show help information about blur detection."""
        help_text = """
    Blur Detection Help

    The blur detector identifies blurry images using the Laplacian variance method:

    - Blur Threshold: Lower values make the detector more sensitive (detecting more images as blurry).
    Typical values range from 50-200 depending on image content.

    - ROI Ratio: Percentage of the central image to analyze. Use higher values for more complete analysis,
    lower values to focus on the center where subjects are typically sharpest.

    - Quality Alert Threshold: Percentage of blurry compartments that will trigger a quality alert.

    - Calibration: Use the calibration tool to automatically set an optimal threshold based on example
    sharp and blurry images.
    """
        DialogHelper.show_message(
            self.root, "Blur Detection Help", help_text, message_type="info"
        )

    def _show_blur_calibration_dialog(self):
        """Show a dialog for calibrating blur detection."""
        # Create a dialog
        dialog = tk.Toplevel(self.root)
        dialog.title(DialogHelper.t("Calibrate Blur Detection"))
        dialog.geometry("600x500")
        dialog.grab_set()  # Make dialog modal

        # Main frame with padding
        main_frame = ttk.Frame(dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Instructions
        instructions = ttk.Label(
            main_frame,
            text=DialogHelper.t(
                "Select example sharp and blurry images to calibrate the blur detection threshold."
            ),
            wraplength=580,
            justify=tk.LEFT,
        )
        instructions.pack(fill=tk.X, pady=(0, 10))

        # Frame for sharp images
        sharp_frame = ttk.LabelFrame(
            main_frame, text=DialogHelper.t("Sharp (Good) Images"), padding=10
        )
        sharp_frame.pack(fill=tk.X, pady=(0, 10))

        sharp_path_var = tk.StringVar()
        sharp_entry = ttk.Entry(sharp_frame, textvariable=sharp_path_var)
        sharp_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        sharp_button = ttk.Button(
            sharp_frame,
            text=DialogHelper.t("Browse"),
            command=lambda: self._select_calibration_images(sharp_path_var),
        )
        sharp_button.pack(side=tk.RIGHT)

        # Frame for blurry images
        blurry_frame = ttk.LabelFrame(
            main_frame, text=DialogHelper.t("Blurry (Poor) Images"), padding=10
        )
        blurry_frame.pack(fill=tk.X, pady=(0, 10))

        blurry_path_var = tk.StringVar()
        blurry_entry = ttk.Entry(blurry_frame, textvariable=blurry_path_var)
        blurry_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        blurry_button = ttk.Button(
            blurry_frame,
            text=DialogHelper.t("Browse"),
            command=lambda: self._select_calibration_images(blurry_path_var),
        )
        blurry_button.pack(side=tk.RIGHT)

        # Results frame
        results_frame = ttk.LabelFrame(
            main_frame, text=DialogHelper.t("Calibration Results"), padding=10
        )
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Results text widget
        results_text = tk.Text(results_frame, height=8, wrap=tk.WORD)
        results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar for results
        scrollbar = ttk.Scrollbar(results_frame, command=results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        results_text.config(yscrollcommand=scrollbar.set)

        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)

        # Calibrate button
        def calibrate():
            # Get paths
            sharp_paths = sharp_path_var.get().split(";")
            blurry_paths = blurry_path_var.get().split(";")

            # Validate
            if (
                not sharp_paths
                or not sharp_paths[0]
                or not blurry_paths
                or not blurry_paths[0]
            ):
                DialogHelper.show_message(
                    dialog,
                    "Error",
                    "Please select both sharp and blurry images",
                    message_type="error",
                )
                return

            # Load images
            sharp_images = []
            blurry_images = []

            results_text.delete(1.0, tk.END)
            results_text.insert(tk.END, "Loading images...\n")

            try:
                # Load sharp images
                for path in sharp_paths:
                    if path and os.path.exists(path):
                        img = cv2.imread(path)
                        if img is not None:
                            sharp_images.append(img)
                            results_text.insert(
                                tk.END,
                                f"Loaded sharp image: {os.path.basename(path)}\n",
                            )

                # Load blurry images
                for path in blurry_paths:
                    if path and os.path.exists(path):
                        img = cv2.imread(path)
                        if img is not None:
                            blurry_images.append(img)
                            results_text.insert(
                                tk.END,
                                f"Loaded blurry image: {os.path.basename(path)}\n",
                            )

                if not sharp_images or not blurry_images:
                    results_text.insert(tk.END, "Error: Failed to load images\n")
                    return

                # Calibrate threshold
                results_text.insert(tk.END, "\nCalibrating...\n")

                old_threshold = self.blur_detector.threshold
                new_threshold = self.blur_detector.calibrate_threshold(
                    sharp_images, blurry_images
                )

                # Update UI
                self.blur_threshold_var.set(new_threshold)

                # Log results
                results_text.insert(tk.END, f"\nResults:\n")
                results_text.insert(tk.END, f"Old threshold: {old_threshold:.2f}\n")
                results_text.insert(tk.END, f"New threshold: {new_threshold:.2f}\n")

                # Show variances
                results_text.insert(tk.END, "\nImage Variances:\n")

                for i, img in enumerate(sharp_images):
                    variance = self.blur_detector.get_laplacian_variance(img)
                    results_text.insert(tk.END, f"Sharp image {i+1}: {variance:.2f}\n")

                for i, img in enumerate(blurry_images):
                    variance = self.blur_detector.get_laplacian_variance(img)
                    results_text.insert(tk.END, f"Blurry image {i+1}: {variance:.2f}\n")

                # Remind user to save settings
                results_text.insert(
                    tk.END, "\nRemember to click 'OK' to apply the new threshold.\n"
                )

            except Exception as e:
                results_text.insert(tk.END, f"Error during calibration: {str(e)}\n")
                logger.error(f"Calibration error: {str(e)}")
                logger.error(traceback.format_exc())

        calibrate_button = ttk.Button(
            button_frame, text=DialogHelper.t("Calibrate"), command=calibrate
        )
        calibrate_button.pack(side=tk.LEFT, padx=(0, 10))

        # OK button
        ok_button = ttk.Button(
            button_frame, text=DialogHelper.t("OK"), command=dialog.destroy
        )
        ok_button.pack(side=tk.RIGHT)

    def _select_calibration_images(self, path_var):
        """
        Show file dialog to select images for calibration.

        Args:
            path_var: StringVar to store selected paths
        """
        file_paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )

        if file_paths:
            # Join paths with semicolons for display
            path_var.set(";".join(file_paths))

    def _update_depth_csv_status(self, folder_path=None):
        """Update the status indicator for datasets folder.

        DEPRECATED: This method is kept for backwards compatibility but
        now operates on the datasets folder instead of a single CSV file.
        The folder status is handled by the path field's visual indicator.
        """
        # This method is now a no-op since we use folder-based status display
        # via the _create_shared_path_field mechanism
        pass

    def on_generate_trace(self):
        """Handle the 'Generate Drillhole Trace' button click."""
        try:
            # Ask user which method to use
            result = DialogHelper.confirm_dialog(
                self.root,
                DialogHelper.t("Trace Generation Method"),
                DialogHelper.t(
                    "Use the new Trace Designer for customizable visualizations?\n\nYes - Open Trace Designer (recommended)\nNo - Use classic method"
                ),
                yes_text=DialogHelper.t("Use Designer"),
                no_text=DialogHelper.t("Classic Method"),
            )

            if result:  # User chose Designer
                # Use the FileManager's directory structure
                compartment_dir = self.file_manager.dir_structure["chip_compartments"]

                # Check if the directory exists
                if not os.path.exists(compartment_dir):
                    DialogHelper.show_message(
                        self.root,
                        DialogHelper.t("Error"),
                        DialogHelper.t(
                            f"Compartment directory not found: {compartment_dir}"
                        ),
                        message_type="error",
                    )
                    return

                # Initialize trace generator
                trace_generator = self.DrillholeTraceGenerator(
                    config=self.config,
                    progress_queue=self.progress_queue,
                    root=self.root,
                    file_manager=self.file_manager,
                )

                # Set app reference for GUI manager access
                trace_generator.app = self.app

                # Show designer and get configuration
                config = trace_generator.show_trace_designer()

                if config:
                    # Get list of holes to process
                    holes_to_process = []
                    if os.path.exists(compartment_dir):
                        holes_to_process = [
                            d
                            for d in os.listdir(compartment_dir)
                            if os.path.isdir(os.path.join(compartment_dir, d))
                        ]

                    if not holes_to_process:
                        DialogHelper.show_message(
                            self.root,
                            DialogHelper.t("Info"),
                            DialogHelper.t("No holes found to process."),
                            message_type="info",
                        )
                        return

                    # Generate traces with configuration
                    generated_paths = trace_generator.generate_configured_traces(
                        compartment_dir=compartment_dir, hole_ids=holes_to_process
                    )

                    if generated_paths:
                        DialogHelper.show_message(
                            self.root,
                            DialogHelper.t("Success"),
                            DialogHelper.t(
                                f"Generated {len(generated_paths)} trace images."
                            ),
                            message_type="info",
                        )

                        # Ask if user wants to open the directory
                        if DialogHelper.confirm_dialog(
                            self.root,
                            DialogHelper.t("Open Directory"),
                            DialogHelper.t(
                                "Would you like to open the directory containing the trace images?"
                            ),
                        ):
                            trace_dir = os.path.dirname(generated_paths[0])
                            try:
                                if platform.system() == "Windows":
                                    os.startfile(trace_dir)
                                elif platform.system() == "Darwin":  # macOS
                                    subprocess.Popen(["open", trace_dir])
                                else:  # Linux
                                    subprocess.Popen(["xdg-open", trace_dir])
                            except Exception as e:
                                DialogHelper.show_message(
                                    self.root,
                                    DialogHelper.t("Error"),
                                    DialogHelper.t(
                                        f"Could not open directory: {str(e)}"
                                    ),
                                    message_type="warning",
                                )
                    else:
                        DialogHelper.show_message(
                            self.root,
                            DialogHelper.t("No Output"),
                            DialogHelper.t("No trace images were generated."),
                            message_type="warning",
                        )

            else:  # User chose classic method
                # Original implementation
                compartment_dir = self.file_manager.dir_structure["chip_compartments"]

                if not os.path.exists(compartment_dir):
                    DialogHelper.show_message(
                        self.root,
                        DialogHelper.t("Error"),
                        DialogHelper.t(
                            f"Compartment directory not found: {compartment_dir}"
                        ),
                        message_type="error",
                    )
                    return

                # Select CSV file
                csv_path = filedialog.askopenfilename(
                    title=DialogHelper.t("Select CSV File"),
                    filetypes=[("CSV Files", "*.csv")],
                )
                if not csv_path:
                    return

                # Rest of original implementation...
                trace_generator = self.DrillholeTraceGenerator(
                    config=self.config,
                    progress_queue=self.progress_queue,
                    root=self.root,
                    file_manager=self.file_manager,
                )

                # Let user select optional columns
                csv_columns = trace_generator.get_csv_columns(csv_path)
                if not csv_columns:
                    DialogHelper.show_message(
                        self.root,
                        DialogHelper.t("CSV Error"),
                        DialogHelper.t("Could not read columns from CSV."),
                        message_type="error",
                    )
                    return

                selected_columns = trace_generator.select_csv_columns(csv_columns)

                # Get the drill traces directory
                traces_dir = self.file_manager.dir_structure["drill_traces"]

                # Get list of existing trace files
                existing_traces = set()
                if os.path.exists(traces_dir):
                    existing_traces = {
                        os.path.splitext(f)[0].split("_")[0]
                        for f in os.listdir(traces_dir)
                        if f.lower().endswith((".tif", ".tiff", ".png", ".jpg"))
                        and "_Trace" in f
                    }

                # Get list of holes from compartment directories
                hole_dirs = [
                    d
                    for d in os.listdir(compartment_dir)
                    if os.path.isdir(os.path.join(compartment_dir, d))
                ]

                # Filter to holes that don't have traces
                holes_to_process = [
                    hole for hole in hole_dirs if hole not in existing_traces
                ]

                if not holes_to_process:
                    DialogHelper.show_message(
                        self.root,
                        DialogHelper.t("Info"),
                        DialogHelper.t("All holes already have trace images."),
                        message_type="info",
                    )
                    return

                # Ask user for confirmation
                if not DialogHelper.confirm_dialog(
                    self.root,
                    DialogHelper.t("Confirm"),
                    DialogHelper.t(
                        f"Found {len(holes_to_process)} holes without trace images. Process them all?"
                    ),
                ):
                    return

                # Run the trace generation
                generated_paths = trace_generator.process_selected_holes(
                    compartment_dir=compartment_dir,
                    csv_path=csv_path,
                    selected_columns=selected_columns,
                    hole_ids=holes_to_process,
                )

                if generated_paths:
                    DialogHelper.show_message(
                        self.root,
                        DialogHelper.t("Success"),
                        DialogHelper.t(
                            f"Generated {len(generated_paths)} drillhole trace images."
                        ),
                        message_type="info",
                    )

                    # Ask if the user wants to open the directory
                    if DialogHelper.confirm_dialog(
                        self.root,
                        DialogHelper.t("Open Directory"),
                        DialogHelper.t(
                            "Would you like to open the directory containing the trace images?"
                        ),
                    ):
                        if os.path.isfile(generated_paths[0]):
                            trace_dir = os.path.dirname(generated_paths[0])
                        else:
                            trace_dir = generated_paths[0]

                        try:
                            if platform.system() == "Windows":
                                os.startfile(trace_dir)
                            elif platform.system() == "Darwin":  # macOS
                                subprocess.Popen(["open", trace_dir])
                            else:  # Linux
                                subprocess.Popen(["xdg-open", trace_dir])
                        except Exception as e:
                            DialogHelper.show_message(
                                self.root,
                                DialogHelper.t("Error"),
                                DialogHelper.t(f"Could not open directory: {str(e)}"),
                                message_type="warning",
                            )
                else:
                    DialogHelper.show_message(
                        self.root,
                        DialogHelper.t("No Output"),
                        DialogHelper.t("No drillhole trace images were generated."),
                        message_type="warning",
                    )

        except Exception as e:
            DialogHelper.show_message(
                self.root,
                DialogHelper.t("Error"),
                DialogHelper.t(f"An error occurred: {str(e)}"),
                message_type="error",
            )
            logger.error(f"Error in on_generate_trace: {str(e)}")
            logger.error(traceback.format_exc())

    def check_progress(self, schedule_next=False):
        """
        Check for progress updates from the processing thread with enhanced status handling.

        Args:
            schedule_next: Whether to schedule another check automatically
        """
        # Only proceed if the root window still exists
        if not hasattr(self, "root") or not self.root.winfo_exists():
            return

        try:
            # Process all queued messages
            while not self.progress_queue.empty():
                try:
                    message, progress = self.progress_queue.get_nowait()

                    # Update progress bar if progress value provided
                    if progress is not None:
                        self.progress_var.set(progress)

                    # Update status message with appropriate status type
                    if message:
                        # Determine message type based on content
                        if any(
                            error_term in message.lower()
                            for error_term in ["error", "failed", "not enough"]
                        ):
                            self.update_status(message, "error")
                        elif any(
                            warning_term in message.lower()
                            for warning_term in ["warning", "missing"]
                        ):
                            self.update_status(message, "warning")
                        elif any(
                            success_term in message.lower()
                            for success_term in ["success", "complete", "saved"]
                        ):
                            self.update_status(message, "success")
                        else:
                            self.update_status(message, "info")

                        # If processing is complete, re-enable the process button
                        if self.processing_complete:
                            if "process_button" in self.buttons:
                                self.buttons["process_button"].set_state("normal")

                except Exception as e:
                    # Log the error but continue processing
                    self.logger.error(f"Error processing queue item: {str(e)}")

        except Exception as e:
            # Log the outer error
            self.logger.error(f"Error in progress check: {str(e)}")

        # Always reschedule if requested, even if exceptions occurred above
        if schedule_next and hasattr(self, "root") and self.root.winfo_exists():
            try:
                self.after_id = self.root.after(
                    500, lambda: self.check_progress(schedule_next=True)
                )
            except Exception as e:
                # Last resort logging
                self.logger.critical(f"Failed to reschedule progress check: {str(e)}")

    def browse_input_folder(self):
        """Open folder browser dialog and update the folder entry."""
        folder_path = filedialog.askdirectory(
            title="Select folder with chip tray photos"
        )
        if folder_path:
            self.folder_var.set(folder_path)

            # Count image files in the selected folder
            image_extensions = (".jpg", ".jpeg", ".HEIC", ".png", ".tif", ".tiff")
            image_files = [
                f
                for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f))
                and f.lower().endswith(image_extensions)
            ]

            # Create message about found files
            message = f"{len(image_files)} image files found in the selected folder"

            # Log to the GUI
            # self.update_status(message, "info")

            # Also add to progress queue if it exists
            if hasattr(self, "progress_queue"):
                self.progress_queue.put((message, None))
                self.check_progress(schedule_next=False)

    def quit_app(self):
        """Close the application after properly terminating threads."""
        try:
            # Cancel any pending after callbacks
            if hasattr(self, "after_id") and self.after_id:
                self.root.after_cancel(self.after_id)
                self.after_id = None

            # Wait for active threads to complete
            for thread in self.active_threads:
                if thread.is_alive():
                    try:
                        thread.join(0.1)  # Wait with timeout
                    except Exception:
                        pass

            # Clear thread references
            self.active_threads = []

            # Close root window
            if self.root:
                self.root.destroy()
        except Exception as e:
            self.logger.error(f"Error during application shutdown: {str(e)}")
            # Force exit if normal shutdown fails
            if self.root:
                self.root.destroy()
