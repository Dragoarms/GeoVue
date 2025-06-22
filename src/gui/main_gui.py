# gui/main_gui.py
import os
import re
import cv2
import numpy as np
import logging
import tkinter as tk
from tkinter import ttk, filedialog
import queue
import shutil
import platform
import subprocess
from typing import Dict, List, Optional, Any, Tuple, Union
import traceback
from gui.dialog_helper import DialogHelper
from processing.drillhole_trace_generator import DrillholeTraceGenerator
from gui.widgets import *
from gui.qaqc_manager import QAQCManager
from gui.logging_review_dialog import LoggingReviewDialog
from utils.json_register_manager import JSONRegisterManager
from utils.register_synchronizer import RegisterSynchronizer
from gui.progress_dialog import ProgressDialog


import threading

if threading.current_thread() != threading.main_thread():
    raise RuntimeError("❌ Main GUI called from a background thread!")


logger = logging.getLogger(__name__)

class MainGUI:
    
    def __init__(self, app):
        """Initialize with reference to main application."""
        self.app = app  # Reference to main app for accessing components
        self.root = app.root  # Use the existing root window
        self.progress_queue = queue.Queue()
        self.processing_complete = False
        self.active_threads = []  # Track active threads
        self.after_id = None  # Initialize after_id to None
        
        # Add a logger instance
        self.logger = logging.getLogger(__name__)

        # Get references to app components for easier access
        self.file_manager = app.file_manager
        self.gui_manager = app.gui_manager
        self.translator = app.translator
        self.tesseract_manager = app.tesseract_manager
        self.blur_detector = app.blur_detector
        self.aruco_manager = app.aruco_manager

        # Set a flag to check if processing is active
        self.is_processing_image = False  # Flag to indicate when an image is being processed
        
        # Initialize language variable
        self.language_var = tk.StringVar(value=self.translator.get_current_language())
        
        # Setup configuration
        self.config = app.config_manager.as_dict()
        
        # ===================================================
        # Removed OneDrivePathManager - now using FileManager's shared paths
        # ===================================================
        
        # Create GUI
        self.create_gui()

        # Set up the shared paths if available
        self.setup_shared_paths()
        
        # Initialize visualization cache for dialogs
        self.visualization_cache = {}
        
        # Delay the first check_progress call until all initialization is complete
        # Use a longer delay for the initial call to ensure everything is ready
        self.root.after(1000, self.check_progress)
    
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
            #width=1200,
            #height=900
        )
        
        # Store window components
        self.main_container = window['main_container']
        self.header_frame = window['header_frame']
        self.title_label = window['title_label']
        self.content_outer_frame = window['content_outer_frame']
        self.canvas = window['canvas']
        self.content_frame = window['content_frame']
        self.footer_frame = window['footer_frame']

        # Hide the default title label from create_main_window
        self.title_label.pack_forget()
        
        # Create centered container for logo and title (like first run dialog)
        center_container = ttk.Frame(self.header_frame, style='Header.TFrame')
        center_container.place(relx=0.5, rely=0.5, anchor="center")
        
        # Add logo and title side by side
        try:
            import PIL.Image
            import PIL.ImageTk
            
            # Get logo path
            logo_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                "resources", 
                "full_logo.png"
            )
            
            if os.path.exists(logo_path):
                # Load and resize logo to larger size
                logo_image = PIL.Image.open(logo_path)
                
                # Resize to larger height (e.g., 80 pixels for better visibility)
                logo_height = 80
                aspect_ratio = logo_image.width / logo_image.height
                logo_width = int(logo_height * aspect_ratio)
                logo_image = logo_image.resize(
                    (logo_width, logo_height), 
                    PIL.Image.Resampling.LANCZOS
                )
                
                # Convert to PhotoImage
                self.logo_photo = PIL.ImageTk.PhotoImage(logo_image)
                
                # Create label for logo
                logo_label = tk.Label(
                    center_container,
                    image=self.logo_photo,
                    bg=self.gui_manager.theme_colors["secondary_bg"]
                )
                logo_label.pack(side=tk.LEFT, padx=(0, 20))
                
                # Add GeoVue title next to logo
                title_label = tk.Label(
                    center_container,
                    text="GeoVue",
                    font=("Arial", 24, "bold"),
                    bg=self.gui_manager.theme_colors["secondary_bg"],
                    fg=self.gui_manager.theme_colors["text"]
                )
                title_label.pack(side=tk.LEFT)
                
                self.logger.debug(f"Logo and title added to main GUI")
                
        except Exception as e:
            self.logger.debug(f"Could not load logo: {e}")
            # If logo fails, just show the title
            title_label = tk.Label(
                center_container,
                text="GeoVue",
                font=("Arial", 24, "bold"),
                bg=self.gui_manager.theme_colors["secondary_bg"],
                fg=self.gui_manager.theme_colors["text"]
            )
            title_label.pack()
            
        # Set minimum height for header to prevent overlap
        self.header_frame.configure(height=100)
        self.header_frame.pack_propagate(False)
        
        
        # Create input section with tighter padding
        input_frame = self.gui_manager.create_section_frame(self.content_frame, padding=5)
        
        # Input folder field
        self.folder_var = tk.StringVar()
        folder_frame, self.folder_entry = self.gui_manager.create_field_with_label(
            input_frame,
            self.t("Input Folder:"), 
            self.folder_var,
            field_type="entry",
            validate_func=self._update_folder_color,
            width=None  # Let it expand naturally
        )

        # Browse button in folder_frame
        browse_button = self.gui_manager.create_modern_button(
            folder_frame,
            text=self.t("Browse"),
            color="#5aa06c",
            command=self.browse_folder
        )
        browse_button.pack(side=tk.RIGHT, padx=(5, 0))

        # # Compartment interval setting
        # interval_frame = ttk.Frame(input_frame, style='Content.TFrame')
        # interval_frame.pack(fill=tk.X, pady=2)
        
        # interval_label = ttk.Label(
        #     interval_frame, 
        #     text=self.t("Compartment Interval (m):"), 
        #     anchor='w',
        #     style='Content.TLabel'
        # )
        # interval_label.pack(side=tk.LEFT)
        
        # # Custom styled combobox frame
        # interval_combo_frame = tk.Frame(
        #     interval_frame,
        #     bg=self.gui_manager.theme_colors["field_bg"],
        #     highlightbackground=self.gui_manager.theme_colors["field_border"],
        #     highlightthickness=1,
        #     bd=0
        # )
        # interval_combo_frame.pack(side=tk.LEFT, padx=(5, 0))
        
        # # Interval dropdown
        # self.interval_var = tk.DoubleVar(value=self.config['compartment_interval'])
        # interval_choices = [1, 2]
        
        # interval_dropdown = tk.OptionMenu(
        #     interval_combo_frame, 
        #     self.interval_var, 
        #     *interval_choices
        # )
        # self.gui_manager.style_dropdown(interval_dropdown, width=3)
        # interval_dropdown.pack()
        
        # Output settings section
        output_frame = self.gui_manager.create_section_frame(self.content_frame, padding=5)
        
        # Local output
        self.output_folder_var = tk.StringVar(value=self.file_manager.dir_structure['local_output_folder'])
        output_folder_frame, output_entry = self.gui_manager.create_field_with_label(
            output_frame, 
            self.t("Local Output:"), 
            self.output_folder_var, 
            readonly=True,
            width=None
        )
        
        # Info button 
        info_button = self.gui_manager.create_modern_button(
            output_folder_frame,
            text="[?]",
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._show_file_structure_info
        )
        info_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Output format
        format_frame = ttk.Frame(output_frame, style='Content.TFrame')
        format_frame.pack(fill=tk.X, pady=2)
        
        format_label = ttk.Label(
            format_frame,
            text=self.t("Output Format:"),
            anchor='w',
            style='Content.TLabel'
        )
        format_label.pack(side=tk.LEFT)
        
        format_combo_frame = tk.Frame(
            format_frame,
            bg=self.gui_manager.theme_colors["field_bg"],
            highlightbackground=self.gui_manager.theme_colors["field_border"],
            highlightthickness=1,
            bd=0
        )
        format_combo_frame.pack(side=tk.LEFT, padx=(5, 0))
        
        self.format_var = tk.StringVar(value=self.config['output_format'])
        format_options = ['tiff','png']
        
        format_dropdown = tk.OptionMenu(
            format_combo_frame,
            self.format_var,
            *format_options
        )
        self.gui_manager.style_dropdown(format_dropdown, width=6)
        format_dropdown.pack()
        
        # Debug images checkbox
        debug_frame = ttk.Frame(output_frame, style='Content.TFrame')
        debug_frame.pack(fill=tk.X, pady=2)
        
        self.debug_var = tk.BooleanVar(value=self.config['save_debug_images'])
        
        # Custom styled checkbox
        debug_check = self.gui_manager.create_custom_checkbox(
            debug_frame,
            text=self.t("Save Debug Images"),
            variable=self.debug_var
        )
        debug_check.pack(anchor='w')
        
        # Create collapsible sections with the GUIManager
        # Shared Folder Path Settings
        
        # Check if paths exist using FileManager
        approved_path_exists = bool(self.file_manager.get_shared_path('approved_compartments', create_if_missing=False))
        processed_originals_exists = bool(self.file_manager.get_shared_path('processed_originals', create_if_missing=False))
        drill_traces_exists = bool(self.file_manager.get_shared_path('drill_traces', create_if_missing=False))
        register_path_exists = bool(self.file_manager.get_shared_path('register_excel', create_if_missing=False))
        
        # Expand if any paths are missing
        should_expand = not (approved_path_exists and processed_originals_exists and 
                        drill_traces_exists and register_path_exists)
        
        shared_collapsible = self.gui_manager.create_collapsible_frame(
            self.content_frame,
            title=self.t("Shared Folder Settings"),
            expanded=should_expand
        )
        self.shared_collapsible = shared_collapsible
        
        # Create variables for shared paths
        self.approved_path_var = tk.StringVar()
        self.processed_originals_path_var = tk.StringVar()
        self.drill_traces_path_var = tk.StringVar()
        self.register_path_var = tk.StringVar()
        
        # Set display values using FileManager's paths or config
        config = self.app.config_manager.as_dict()
        
        # Approved folder
        approved_path = self.file_manager.get_shared_path('approved_compartments', create_if_missing=False)
        self.approved_path_var.set(str(approved_path) if approved_path else config.get('shared_folder_approved_compartments_folder', ""))
        
        # Processed originals
        processed_path = self.file_manager.get_shared_path('processed_originals', create_if_missing=False)
        self.processed_originals_path_var.set(str(processed_path) if processed_path else config.get('shared_folder_processed_originals', ""))
        
        # Drill traces
        drill_traces_path = self.file_manager.get_shared_path('drill_traces', create_if_missing=False)
        self.drill_traces_path_var.set(str(drill_traces_path) if drill_traces_path else config.get('shared_folder_drill_traces', ""))
        
        # Register
        register_path = self.file_manager.get_shared_path('register_excel', create_if_missing=False)
        self.register_path_var.set(str(register_path) if register_path else config.get('shared_folder_register_excel_path', ""))
        
        # Create path input fields
        self._create_shared_path_field(
            shared_collapsible.content_frame, 
            self.t("Approved Folder:"), 
            self.approved_path_var, 
            approved_path_exists,
            is_file=False,
            path_key='approved_compartments'
        )
        self._create_shared_path_field(
            shared_collapsible.content_frame, 
            self.t("Processed Originals Folder:"), 
            self.processed_originals_path_var,
            processed_originals_exists,
            is_file=False,
            path_key='processed_originals'
        )
        self._create_shared_path_field(
            shared_collapsible.content_frame, 
            self.t("Drill Traces Folder:"), 
            self.drill_traces_path_var,
            drill_traces_exists,
            is_file=False,
            path_key='drill_traces'
        )
        self._create_shared_path_field(
            shared_collapsible.content_frame, 
            self.t("Excel Register:"), 
            self.register_path_var,
            register_path_exists,
            is_file=True,  # This is a file, not a folder
            path_key='register_excel'
        )
        
        # Create New Register button
        button_container = ttk.Frame(shared_collapsible.content_frame, style='Content.TFrame')
        button_container.pack(fill=tk.X, pady=(5, 0))
        
        create_register_button = self.gui_manager.create_modern_button(
            button_container,
            text=self.t("Create New Register"),
            color=self.gui_manager.theme_colors["accent_green"],
            command=self._create_new_register
        )
        create_register_button.pack(side=tk.RIGHT)
        
        # Blur Detection - Collapsible
        blur_collapsible = self.gui_manager.create_collapsible_frame(
            self.content_frame,
            title=self.t("Blur Detection"),
            expanded=False
        )
        self.blur_collapsible = blur_collapsible
        
        # Enable blur detection checkbox
        blur_enable_frame = ttk.Frame(blur_collapsible.content_frame, style='Content.TFrame')
        blur_enable_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.blur_enable_var = tk.BooleanVar(value=self.config['enable_blur_detection'])
        blur_enable_check = self.gui_manager.create_custom_checkbox(
            blur_enable_frame,
            text=self.t("Enable Blur Detection"),
            variable=self.blur_enable_var,
            command=self._toggle_blur_settings
        )
        blur_enable_check.pack(anchor='w')
        
        # Blur settings container
        blur_settings_frame = ttk.Frame(blur_collapsible.content_frame, style='Content.TFrame')
        blur_settings_frame.pack(fill=tk.X, padx=(20, 5))
        
        # Blur threshold
        threshold_frame = ttk.Frame(blur_settings_frame, style='Content.TFrame')
        threshold_frame.pack(fill=tk.X, pady=2)
        
        threshold_label = ttk.Label(
            threshold_frame,
            text=self.t("Blur Threshold:"),
            anchor='w',
            style='Content.TLabel'
        )
        threshold_label.pack(side=tk.LEFT)
        
        self.blur_threshold_var = tk.DoubleVar(value=self.config['blur_threshold'])
        
        threshold_slider_frame = ttk.Frame(threshold_frame, style='Content.TFrame')
        threshold_slider_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        
        threshold_slider = ttk.Scale(
            threshold_slider_frame,
            from_=10.0,
            to=500.0,
            orient=tk.HORIZONTAL,
            variable=self.blur_threshold_var,
            style='Horizontal.TScale'
        )
        threshold_slider.pack(fill=tk.X)
        
        threshold_value = ttk.Label(
            threshold_frame,
            width=5,
            style='Value.TLabel'
        )
        threshold_value.pack(side=tk.RIGHT)
        
        # Update threshold value label when slider changes
        def update_threshold_label(*args):
            threshold_value.config(text=f"{self.blur_threshold_var.get():.1f}")
        
        self.blur_threshold_var.trace_add("write", update_threshold_label)
        update_threshold_label()  # Initial update
        
        # ROI ratio
        roi_frame = ttk.Frame(blur_settings_frame, style='Content.TFrame')
        roi_frame.pack(fill=tk.X, pady=2)
        
        roi_label = ttk.Label(
            roi_frame,
            text=self.t("ROI Ratio:"),
            anchor='w',
            style='Content.TLabel'
        )
        roi_label.pack(side=tk.LEFT)
        
        self.blur_roi_var = tk.DoubleVar(value=self.config['blur_roi_ratio'])
        
        roi_slider_frame = ttk.Frame(roi_frame, style='Content.TFrame')
        roi_slider_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        
        roi_slider = ttk.Scale(
            roi_slider_frame,
            from_=0.1,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.blur_roi_var,
            style='Horizontal.TScale'
        )
        roi_slider.pack(fill=tk.X)
        
        roi_value = ttk.Label(
            roi_frame,
            width=5,
            style='Value.TLabel'
        )
        roi_value.pack(side=tk.RIGHT)
        
        # Update ROI value label when slider changes
        def update_roi_label(*args):
            roi_value.config(text=f"{self.blur_roi_var.get():.2f}")
        
        self.blur_roi_var.trace_add("write", update_roi_label)
        update_roi_label()  # Initial update
        
        # Flag blurry images
        flag_blurry_frame = ttk.Frame(blur_settings_frame, style='Content.TFrame')
        flag_blurry_frame.pack(fill=tk.X, pady=2)
        
        self.flag_blurry_var = tk.BooleanVar(value=self.config['flag_blurry_images'])
        flag_blurry_check = self.gui_manager.create_custom_checkbox(
            flag_blurry_frame,
            text=self.t("Flag Blurry Images"),
            variable=self.flag_blurry_var
        )
        flag_blurry_check.pack(anchor='w')
        
        # Save blur visualizations
        save_viz_frame = ttk.Frame(blur_settings_frame, style='Content.TFrame')
        save_viz_frame.pack(fill=tk.X, pady=2)
        
        self.save_blur_viz_var = tk.BooleanVar(value=self.config['save_blur_visualizations'])
        save_viz_check = self.gui_manager.create_custom_checkbox(
            save_viz_frame,
            text=self.t("Save Blur Analysis Visualizations"),
            variable=self.save_blur_viz_var
        )
        save_viz_check.pack(anchor='w')
        
        # Blurry threshold percentage
        threshold_pct_frame = ttk.Frame(blur_settings_frame, style='Content.TFrame')
        threshold_pct_frame.pack(fill=tk.X, pady=2)
        
        threshold_pct_label = ttk.Label(
            threshold_pct_frame,
            text=self.t("Quality Alert Threshold:"),
            anchor='w',
            style='Content.TLabel'
        )
        threshold_pct_label.pack(side=tk.LEFT)
        
        self.blur_threshold_pct_var = tk.DoubleVar(value=self.config['blurry_threshold_percentage'])
        
        threshold_pct_slider_frame = ttk.Frame(threshold_pct_frame, style='Content.TFrame')
        threshold_pct_slider_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        
        threshold_pct_slider = ttk.Scale(
            threshold_pct_slider_frame,
            from_=5.0,
            to=100.0,
            orient=tk.HORIZONTAL,
            variable=self.blur_threshold_pct_var,
            style='Horizontal.TScale'
        )
        threshold_pct_slider.pack(fill=tk.X)
        
        threshold_pct_value = ttk.Label(
            threshold_pct_frame,
            width=5,
            style='Value.TLabel'
        )
        threshold_pct_value.pack(side=tk.RIGHT)
        
        # Update threshold percentage value label when slider changes
        def update_threshold_pct_label(*args):
            threshold_pct_value.config(text=f"{self.blur_threshold_pct_var.get():.1f}%")
        
        self.blur_threshold_pct_var.trace_add("write", update_threshold_pct_label)
        update_threshold_pct_label()  # Initial update
        
        # Calibration and help buttons
        button_frame = ttk.Frame(blur_settings_frame, style='Content.TFrame')
        button_frame.pack(fill=tk.X, pady=(5, 0))
        
        calibrate_button = self.gui_manager.create_modern_button(
            button_frame,
            text=self.t("Calibrate Blur Detection"),
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._show_blur_calibration_dialog
        )
        calibrate_button.pack(side=tk.LEFT, padx=(0, 5))

        help_button = self.gui_manager.create_modern_button(
            button_frame,
            text="?",
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._show_blur_help
        )
        help_button.pack(side=tk.RIGHT)

        # Store blur settings widgets for toggling
        self.blur_settings_controls = [blur_settings_frame]
        
        # OCR Settings - Collapsible
        ocr_collapsible = self.gui_manager.create_collapsible_frame(
            self.content_frame,
            title=self.t("OCR Settings"),
            expanded=False
        )
        self.ocr_collapsible = ocr_collapsible
        
        # Enable OCR checkbox
        ocr_enable_frame = ttk.Frame(ocr_collapsible.content_frame, style='Content.TFrame')
        ocr_enable_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.ocr_enable_var = tk.BooleanVar(value=self.config['enable_ocr'])
        ocr_enable_check = self.gui_manager.create_custom_checkbox(
            ocr_enable_frame,
            text=self.t("Enable OCR"),
            variable=self.ocr_enable_var,
            command=self._toggle_ocr_settings
        )
        ocr_enable_check.pack(anchor='w')
        
        # OCR settings container
        ocr_settings_frame = ttk.Frame(ocr_collapsible.content_frame, style='Content.TFrame')
        ocr_settings_frame.pack(fill=tk.X, padx=(20, 5))
        
        # Prefix validation
        prefix_validation_frame = ttk.Frame(ocr_settings_frame, style='Content.TFrame')
        prefix_validation_frame.pack(fill=tk.X, pady=2)
        
        self.prefix_validation_var = tk.BooleanVar(value=self.config.get('enable_prefix_validation', True))
        prefix_validation_check = self.gui_manager.create_custom_checkbox(
            prefix_validation_frame,
            text=self.t("Validate Hole ID Prefixes"),
            variable=self.prefix_validation_var,
            command=self._toggle_prefix_settings
        )
        prefix_validation_check.pack(anchor='w')

        # Prefix list
        prefix_frame = ttk.Frame(ocr_settings_frame, style='Content.TFrame')
        prefix_frame.pack(fill=tk.X, pady=2, padx=(20, 0))
        
        prefix_label = ttk.Label(
            prefix_frame,
            text=self.t("Valid Prefixes (comma separated):"),
            anchor='w',
            style='Content.TLabel'
        )
        prefix_label.pack(side=tk.LEFT)
        
        # Convert list to comma-separated string for display
        prefix_str = ", ".join(self.config.get('valid_hole_prefixes', ['BA', 'NB', 'SB', 'KM']))
        self.prefix_var = tk.StringVar(value=prefix_str)
        
        prefix_entry = self.gui_manager.create_entry_with_validation(
            prefix_frame, 
            self.prefix_var
        )
        prefix_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Store reference to OCR control widgets for toggling
        self.prefix_controls = [prefix_frame]
        self.ocr_controls = [prefix_validation_frame] + self.prefix_controls
        
        # Initialize visibility based on checkbox states
        self._toggle_blur_settings()
        self._toggle_ocr_settings()
        self._toggle_prefix_settings()
        
        # Create status and progress section
        status_components = self.gui_manager.create_status_section(self.content_frame)
        self.progress_var = status_components['progress_var']
        self.progress_bar = status_components['progress_bar']
        self.status_text = status_components['status_text']
        
        # Create footer buttons
        button_configs = [
            {
                'name': 'process_button',
                'text': self.t("Process Photos"),
                'color': self.gui_manager.theme_colors["accent_green"],
                'command': self.start_processing,
                'icon': "▶"
            },
            {
                'name': 'review_button',
                'text': self.t("Review Extracted Images"),
                'color': self.gui_manager.theme_colors["accent_blue"],
                'command': self._start_image_review,
                'icon': "🔍"
            },
            {
                'name': 'validate_button',
                'text': self.t("Validate Register"),
                'color': self.gui_manager.theme_colors["accent_blue"],
                'command': self._validate_register_entries,
                'icon': "✓"
            },
            {
                'name': 'logging_button',
                'text': self.t("Logging Review"),
                'color': self.gui_manager.theme_colors["accent_blue"],
                'command': self._start_logging_review,
                'icon': "📋"
            },
            {
                'name': 'trace_button',
                'text': self.t("Generate Drillhole Trace"),
                'color': self.gui_manager.theme_colors["accent_blue"],
                'command': self.on_generate_trace,
                'icon': "📊"
            },
            {
                'name': 'quit_button',
                'text': self.t("Quit"),
                'color': self.gui_manager.theme_colors["accent_red"],
                'command': self.quit_app,
                'icon': "✖"
            }
        ]
        
        _, self.buttons = self.gui_manager.create_button_row(
            self.footer_frame, 
            button_configs,
            side="bottom", 
            anchor="se", 
            padx=10, 
            pady=10
        )

        # Create menu definitions
        menu_defs = {
            self.t("File"): [
                {
                    'type': 'command',
                    'label': self.gui_manager.get_theme_label(),
                    'command': self._toggle_theme
                },
                {
                    'type': 'separator'
                },
                {
                    'type': 'command',
                    'label': self.t("Exit"),
                    'command': self.quit_app
                }
            ],
            self.t("Help"): [
                {
                    'type': 'command',
                    'label': self.t("Check for Updates"),
                    'command': self.on_check_for_updates
                },
                {
                    'type': 'command',
                    'label': self.t("About"),
                    'command': self._show_about_dialog
                }
            ]
        }

        # Add language options to menu definitions
        languages = self.translator.get_available_languages()
        language_items = []
        for lang_code in languages:
            lang_name = self.translator.get_language_name(lang_code)
            language_items.append({
                'type': 'radiobutton',
                'label': lang_name,
                'value': lang_code,
                'variable': self.language_var,
                'command': lambda lc=lang_code: self.change_language(lc)
            })

        menu_defs[self.t("Language")] = language_items

        # Create menubar
        self.menubar = self.gui_manager.setup_menubar(self.root, menu_defs)

        # Size and center the main window after everything is created
        def size_and_center_window():
            # First, temporarily set a large size so everything can lay out
            self.root.geometry("1200x900")
            self.root.update_idletasks()
            
            # Now get the actual required sizes
            # For width, check the widest element
            header_width = self.header_frame.winfo_reqwidth()
            footer_width = self.footer_frame.winfo_reqwidth()
            
            # For the content, get the canvas width (not the content frame inside it)
            canvas_width = self.canvas.winfo_reqwidth()
            
            # Use the widest component
            required_width = max(header_width, footer_width, canvas_width, 1000)  # Min 1000 for buttons
            
            # For height, be more conservative
            window_height = 900  # Fixed reasonable height
            
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
        self.update_status(self.t("Ready. Select a folder and click 'Process Photos'."), "info")

        # Check for updates at startup if enabled
        if self.config.get('check_for_updates', True):
            self.root.after(2000, self._check_updates_at_startup)

       
    def _validate_register_entries(self):
        """Validate that all register entries have existing files."""
        try:
            # Check if we have a register path
            register_path = self.file_manager.get_shared_path('register_excel', create_if_missing=False)
            if not register_path:
                DialogHelper.show_message(
                    self.root,
                    self.t("Error"),
                    self.t("No register path configured. Please check shared folder settings."),
                    message_type="error"
                )
                return
            
            # Initialize JSON manager if needed
            # register_path is a Path object, so use .parent instead of os.path.dirname
            base_path = str(register_path.parent)
            if not JSONRegisterManager.has_existing_data_static(base_path):
                DialogHelper.show_message(
                    self.root,
                    self.t("Error"),
                    self.t("No register data found at the configured location."),
                    message_type="error"
                )
                return
            
            # Create progress dialog
            progress_dialog = ProgressDialog(
                self.root,
                self.t("Validating Register"),
                self.t("Initializing validation...")
            )
            
            # Force the progress dialog to appear
            progress_dialog.dialog.update()
            self.root.update_idletasks()
            
            try:
                # Create synchronizer with progress callback
                synchronizer = RegisterSynchronizer(
                    self.file_manager,
                    self.config,
                    progress_callback=lambda msg, pct: progress_dialog.update_progress(msg, pct)
                )
                
                # Set JSON manager
                synchronizer.set_json_manager(base_path)
                
                # Run full synchronization which includes validation
                results = progress_dialog.run_with_progress(
                    synchronizer.synchronize_all  # This includes validation + hex colors
                )
                
                # Process results
                if results and results.get('success', False):
                    message_lines = [
                        self.t("Validation Complete"),
                        "",
                        self.t("Files checked and register updated:"),
                        self.t("Missing compartment images:") + f" {results.get('missing_files_compartments', 0)}",
                        self.t("Missing original images:") + f" {results.get('missing_files_originals', 0)}"
                    ]
                    
                    # Add sync results
                    if results.get('compartments_added', 0) > 0:
                        message_lines.append(self.t("New compartments found:") + f" {results['compartments_added']}")
                    if results.get('originals_added', 0) > 0:
                        message_lines.append(self.t("New originals found:") + f" {results['originals_added']}")
                    if results.get('originals_updated', 0) > 0:
                        message_lines.append(self.t("Originals updated:") + f" {results['originals_updated']}")
                    
                    # Add hex color results
                    if results.get('hex_colors_calculated', 0) > 0 or results.get('hex_colors_failed', 0) > 0:
                        message_lines.extend([
                            "",
                            self.t("Hex colors calculated:") + f" {results.get('hex_colors_calculated', 0)}",
                            self.t("Hex color failures:") + f" {results.get('hex_colors_failed', 0)}"
                        ])
                    
                    # Add missing files note if any
                    total_missing = results.get('missing_files_compartments', 0) + results.get('missing_files_originals', 0)
                    if total_missing > 0:
                        message_lines.extend([
                            "",
                            self.t("Entries have been marked as MISSING_FILE in the register."),
                            self.t("These files may have been deleted from the shared folder.")
                        ])
                        
                    DialogHelper.show_message(
                        self.root,
                        self.t("Validation Results"),
                        "\n".join(message_lines),
                        message_type="info"
                    )
                else:
                    error_msg = results.get('error', 'Unknown error') if results else 'No results returned'
                    DialogHelper.show_message(
                        self.root,
                        self.t("Error"),
                        self.t("Validation failed:") + f"\n{error_msg}",
                        message_type="error"
                    )
                    
            finally:
                # Ensure dialog is closed
                if progress_dialog.dialog and progress_dialog.dialog.winfo_exists():
                    progress_dialog.close()
                    
        except Exception as e:
            self.logger.error(f"Error in _validate_register_entries: {str(e)}")
            self.logger.error(traceback.format_exc())
            DialogHelper.show_message(
                self.root,
                self.t("Error"),
                self.t("An error occurred:") + f"\n{str(e)}",
                message_type="error"
            )

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
        if hasattr(self, 'root') and self.root.winfo_exists():
            self.root.after(0, lambda: self._perform_direct_update(message, status_type, progress))

    def _perform_direct_update(self, message, status_type, progress):
        """Helper method to actually perform the update on the UI thread."""
        # Update progress if provided
        if progress is not None:
            self.progress_var.set(progress)
        
        # Update status text
        self.update_status(message, status_type)


    def _update_folder_color(self, *args):
        """Update input folder entry background color based on content validity."""
        folder_path = self.folder_var.get()
        if folder_path and os.path.isdir(folder_path):
            # Check if directory contains image files
            has_images = any(f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')) 
                            for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)))
            
            if has_images:
                self.folder_entry.config(bg=self.gui_manager.theme_colors["accent_valid"])
                # Make browse button less prominent once valid folder selected
                self.buttons['process_button'].set_state("normal")
            else:
                self.folder_entry.config(bg=self.gui_manager.theme_colors["accent_error"])
        else:
            self.folder_entry.config(bg=self.gui_manager.theme_colors["accent_error"])
    
    def setup_shared_paths(self):
        """
        Setup all shared folder paths and save them to config.
        This should be called after FileManager is initialized.
        """
        if not hasattr(self, 'file_manager') or not self.file_manager:
            self.logger.error("FileManager not available for setting up shared paths")
            return False
        
        try:
            # Get all the paths
            paths_found = False
            
            # Approved folder
            approved_path = self.file_manager.get_shared_path('approved_compartments', create_if_missing=False)
            if approved_path and approved_path.exists():
                self.app.config_manager.set('shared_folder_approved_compartments_folder', str(approved_path))
                self.approved_path_var.set(str(approved_path))
                self.logger.info(f"Saved shared approved folder: {approved_path}")
                paths_found = True
            
            # Processed originals
            processed_path = self.file_manager.get_shared_path('processed_originals', create_if_missing=False)
            if processed_path and processed_path.exists():
                self.app.config_manager.set('shared_folder_processed_originals', str(processed_path))
                self.processed_originals_path_var.set(str(processed_path))
                self.logger.info(f"Saved shared processed originals: {processed_path}")
                paths_found = True
            
            # Rejected/Failed folder
            rejected_path = self.file_manager.get_shared_path('rejected_originals', create_if_missing=False)
            if rejected_path and rejected_path.exists():
                self.app.config_manager.set('shared_folder_rejected_folder', str(rejected_path))
                self.logger.info(f"Saved shared rejected folder: {rejected_path}")
                paths_found = True
            
            # Drill traces
            drill_traces_path = self.file_manager.get_shared_path('drill_traces', create_if_missing=False)
            if drill_traces_path and drill_traces_path.exists():
                self.app.config_manager.set('shared_folder_drill_traces', str(drill_traces_path))
                self.drill_traces_path_var.set(str(drill_traces_path))
                self.logger.info(f"Saved shared drill traces: {drill_traces_path}")
                paths_found = True
            
            # Excel register
            register_path = self.file_manager.get_shared_path('register_excel', create_if_missing=False)
            if register_path and register_path.exists():
                self.app.config_manager.set('shared_folder_register_excel_path', str(register_path))
                self.register_path_var.set(str(register_path))
                self.logger.info(f"Saved shared register path: {register_path}")
                paths_found = True
            
            # Register Data folder
            register_data_path = self.file_manager.get_shared_path('register_data', create_if_missing=False)
            if register_data_path and register_data_path.exists():
                self.app.config_manager.set('shared_folder_register_data_folder', str(register_data_path))
                self.logger.info(f"Saved shared register data folder: {register_data_path}")
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
        if hasattr(self, 'shared_collapsible'):
            for frame in self.shared_collapsible.content_frame.winfo_children():
                if isinstance(frame, ttk.Frame):
                    for child in frame.winfo_children():
                        if isinstance(child, tk.Entry):
                            # Check which variable this entry is bound to
                            try:
                                var_str = str(child.cget('textvariable'))
                                path_value = child.get()
                                
                                # Update color based on path validity
                                if path_value and os.path.exists(path_value):
                                    child.config(bg=self.gui_manager.theme_colors["accent_valid"])
                                else:
                                    child.config(bg=self.gui_manager.theme_colors["accent_error"])
                            except:
                                pass

    def _create_shared_path_field(self, parent, label_text, string_var, valid=False, is_file=False, path_key=None):
        """Create a field for shared folder path input with browse button.
        
        Args:
            parent: Parent widget
            label_text: Label text for the field
            string_var: StringVar to bind to the entry
            valid: Whether the current path is valid (for coloring)
            is_file: Whether browsing for a file vs folder
            path_key: FileManager path key (e.g., 'approved_compartments', 'register_excel')
        """
        frame = ttk.Frame(parent, style='Content.TFrame')
        frame.pack(fill=tk.X, pady=5)
        
        # Label with wider fixed width
        label = ttk.Label(
            frame, 
            text=self.t(label_text), 
            width=25, 
            anchor='w',
            style='Content.TLabel'
        )
        label.pack(side=tk.LEFT)
        
        # Themed entry field with validation coloring
        entry = tk.Entry(
            frame, 
            textvariable=string_var,
            font=('Arial', 10),
            bg=self.gui_manager.theme_colors["accent_valid"] if valid else self.gui_manager.theme_colors["accent_error"],
            fg=self.gui_manager.theme_colors["text"],
            insertbackground=self.gui_manager.theme_colors["text"],
            relief=tk.FLAT,
            bd=1,
            highlightbackground=self.gui_manager.theme_colors["field_border"],
            highlightthickness=1
        )
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        # Browse button with path_key passed to browse method
        browse_button = self.gui_manager.create_modern_button(
            frame, 
            text=self.t("Browse"), 
            color=self.gui_manager.theme_colors["accent_blue"],
            command=lambda: self._browse_shared_path(string_var, entry, is_file=is_file, path_key=path_key)
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
            activeforeground=self.gui_manager.theme_colors["menu_active_fg"]
        )
        
        # Apply theme to all cascade menus
        for menu_name in self.menubar.winfo_children():
            try:
                menu = self.menubar.nametowidget(menu_name)
                menu.config(
                    bg=self.gui_manager.theme_colors["menu_bg"],
                    fg=self.gui_manager.theme_colors["menu_fg"],
                    activebackground=self.gui_manager.theme_colors["menu_active_bg"],
                    activeforeground=self.gui_manager.theme_colors["menu_active_fg"]
                )
            except:
                pass

        # Update all widgets with the new theme
        self.gui_manager.update_widget_theme(self.root)

        # Update custom widget themes explicitly
        self.gui_manager.update_custom_widget_theme(self.main_container)

        # Add a status message
        self.update_status(self.t(f"Switched to {self.gui_manager.current_theme} theme"), "info")

    def update_status(self, message: str, status_type: str = None) -> None:
        """
        Add a message to the status text widget with optional formatting.
        
        Args:
            message: The message to display
            status_type: Optional status type for formatting (error, warning, success, info)
        """
        try:
            # Check if status_text exists
            if not hasattr(self, 'status_text') or self.status_text is None:
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

    def _create_modern_button(self, parent, text, color, command, icon=None, grid_pos=None):
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
            cursor="hand2"
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
            font=("Arial", 11),
            padx=15,
            pady=8,
            cursor="hand2",
            width=min_width // 10  # Approximate character width
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
                    self.logger.error("Error executing button command:\n" + traceback.format_exc())

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
        frame = ttk.Frame(parent, style='Content.TFrame')
        
        # Custom checkbox appearance
        checkbox_size = 18
        checkbox_frame = tk.Frame(
            frame,
            width=checkbox_size,
            height=checkbox_size,
            bg=self.theme_colors["checkbox_bg"],
            highlightbackground=self.theme_colors["field_border"],
            highlightthickness=1
        )
        checkbox_frame.pack(side=tk.LEFT, padx=(0, 5))
        checkbox_frame.pack_propagate(False)  # Maintain size
        
        # Checkmark that appears when checked
        checkmark = tk.Label(
            checkbox_frame,
            text="✓",
            bg=self.theme_colors["checkbox_bg"],
            fg=self.theme_colors["checkbox_fg"],
            font=("Arial", 12, "bold")
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
        label = ttk.Label(
            frame,
            text=self.t(text),
            style='Content.TLabel'
        )
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
            arrow_color=self.theme_colors["accent_blue"]
        )
        frame.pack(fill=tk.X, pady=(0, 10))
        return frame

    def _update_gui_translations(self):
        """Update all GUI text elements with current language."""
        try:
            if not hasattr(self, 'root') or not self.root:
                return
                
            # Update window title
            self.root.title(self.t("GeoVue"))
            
            # Update header and section titles
            if hasattr(self, 'title_label'):
                self.title_label.config(text=self.t("GeoVue"))
            
            # Update input section labels
            if hasattr(self, 'folder_entry'):
                # Find the label in the folder_entry's parent (folder_frame)
                for child in self.folder_entry.master.winfo_children():
                    if isinstance(child, ttk.Label):
                        child.config(text=self.t("Input Folder:"))
                        break
            
            # Update interval label
            for frame in self.content_frame.winfo_children():
                if isinstance(frame, ttk.Frame):
                    for child in frame.winfo_children():
                        if isinstance(child, ttk.Label) and child.cget("text").startswith("Compartment Interval"):
                            child.config(text=self.t("Compartment Interval (m):"))
                        elif isinstance(child, ttk.Label) and child.cget("text").startswith("Output Format"):
                            child.config(text=self.t("Output Format:"))
                        elif isinstance(child, ttk.Label) and child.cget("text").startswith("Local Output"):
                            child.config(text=self.t("Local Output:"))
            
            # Update all custom widgets with translation methods
            for container in [self.content_frame, self.main_container, self.footer_frame]:
                self._update_translations_recursive(container)
            
            # Update collapsible frame titles
            if hasattr(self, 'shared_collapsible'):
                self.shared_collapsible.set_text(self.t("Shared Folder Settings"))
            if hasattr(self, 'blur_collapsible'):
                self.blur_collapsible.set_text(self.t("Blur Detection"))
            if hasattr(self, 'ocr_collapsible'):
                self.ocr_collapsible.set_text(self.t("OCR Settings"))
            
            # Update shared folder path labels
            if hasattr(self, 'shared_collapsible'):
                shared_frame = self.shared_collapsible.content_frame
                for child in shared_frame.winfo_children():
                    if isinstance(child, ttk.Frame):
                        for subchild in child.winfo_children():
                            if isinstance(subchild, ttk.Label):
                                if "Approved" in subchild.cget("text"):
                                    subchild.config(text=self.t("Approved Folder:"))
                                elif "Processed" in subchild.cget("text"):
                                    subchild.config(text=self.t("Processed Originals Folder:"))
                                elif "Drill" in subchild.cget("text"):
                                    subchild.config(text=self.t("Drill Traces Folder:"))
                                elif "Excel" in subchild.cget("text"):
                                    subchild.config(text=self.t("Excel Register:"))
            
            # Update button texts - include all buttons
            if hasattr(self, 'buttons'):
                button_texts = {
                    'process_button': "Process Photos",
                    'review_button': "Review Extracted Images",
                    'validate_button': "Validate Register",
                    'logging_button': "Logging Review",
                    'trace_button': "Generate Drillhole Trace",
                    'quit_button': "Quit"
                }
                
                for btn_name, text in button_texts.items():
                    if btn_name in self.buttons:
                        button = self.buttons[btn_name]
                        try:
                            # ModernButton has a set_text method
                            if hasattr(button, 'set_text'):
                                button.set_text(self.t(text))
                            elif hasattr(button, 'config'):
                                button.config(text=self.t(text))
                            elif hasattr(button, 'configure'):
                                button.configure(text=self.t(text))
                        except Exception as btn_error:
                            self.logger.warning(f"Could not update text for button {btn_name}: {str(btn_error)}")
            
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
            if hasattr(child, 'update_translation'):
                try:
                    child.update_translation(self.t)
                except Exception as e:
                    self.logger.debug(f"Error updating translation for widget: {e}")
            
            # For custom frames from field_with_label that return a tuple (frame, field)
            # The frame itself has the update_translation method
            if hasattr(child, 'winfo_children'):
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
                
                self.logger.info(f"Language changed to {self.translator.get_language_name(language_code)}")
                
                # Show confirmation to the user
                DialogHelper.show_message(
                    self.root,
                    self.t("Language Changed"),
                    self.t(f"Application language has been changed to {self.translator.get_language_name(language_code)}."),
                    message_type="info"
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
                message_type="error"
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
                        'type': 'command',
                        'label': self.gui_manager.get_theme_label(),
                        'command': self._toggle_theme
                    },
                    {
                        'type': 'separator'
                    },
                    {
                        'type': 'command',
                        'label': self.t("Exit"),
                        'command': self.quit_app
                    }
                ],
                self.t("Help"): [
                    {
                        'type': 'command',
                        'label': self.t("Check for Updates"),
                        'command': self.on_check_for_updates
                    },
                    {
                        'type': 'command',
                        'label': self.t("About"),
                        'command': self._show_about_dialog
                    }
                ]
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
                    
                language_items.append({
                    'type': 'radiobutton',
                    'label': lang_name,
                    'value': lang_code,
                    'variable': self.language_var,
                    'command': make_lang_command(lang_code)
                })

            menu_defs[self.t("Language")] = language_items
            
            # Remove old menubar if it exists
            if hasattr(self, 'menubar') and self.menubar:
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
        if hasattr(self.app, 'update_checker'):
            # Use check_and_update which shows appropriate dialogs
            self.app.update_checker.check_and_update(parent_window=self.root)

    def _show_about_dialog(self):
        """Show information about the application."""
        version = self.update_checker.get_local_version() if hasattr(self, 'update_checker') else "Unknown"
        
        about_text = (
            f"GeoVue v{version}\n\n"
            "A tool to extract individual compartment images from\n"
            "panoramic chip tray photos using ArUco markers"
        )
        
        DialogHelper.show_message(self.root, "About GeoVue", about_text, message_type="info")

    def _check_updates_at_startup(self):
        """Check for updates at startup without showing dialogs for up-to-date case."""
        if not hasattr(self, 'update_checker'):
            return

        try:
            result = self.update_checker.compare_versions()

            if result["update_available"]:
                if DialogHelper.confirm_dialog(
                    self.root,
                    "Update Available",
                    f"A new version is available:\n{result['github_version']}.\n\nDownload and restart?"
                ):
                    self.update_checker.download_and_replace_script(self.file_manager)

        except Exception as e:
            self.logger.error(f"Error checking for updates on startup: {e}")
    
    def _toggle_blur_settings(self):
        """Enable or disable blur detection settings based on checkbox state."""
        if not hasattr(self, 'blur_settings_controls'):
            return

        state = tk.NORMAL if self.blur_enable_var.get() else tk.DISABLED

        for widget in self.blur_settings_controls:
            self.gui_manager._apply_theme_to_widget(widget)
            for child in widget.winfo_children():
                try:
                    # Only configure widgets that accept 'state'
                    if hasattr(child, 'configure'):
                        if 'state' in child.configure():  # Check if 'state' is a valid config option
                            child.configure(state=state)
                except Exception:
                    pass
    
    def _toggle_ocr_settings(self):
        """Enable/disable OCR settings based on checkbox state."""
        if not hasattr(self, 'ocr_controls'):
            return

        state = tk.NORMAL if self.ocr_enable_var.get() else tk.DISABLED

        for widget in self.ocr_controls:
            self.gui_manager._apply_theme_to_widget(widget)
            for child in widget.winfo_children():
                try:
                    if hasattr(child, 'configure') and 'state' in child.configure():
                        child.configure(state=state)
                except Exception:
                    pass


    def _toggle_prefix_settings(self):
        """Enable/disable prefix settings based on checkbox state."""
        if not hasattr(self, 'prefix_controls'):
            return
        
        # CHANGED: Only depend on prefix validation state, not OCR state
        state = tk.NORMAL if self.prefix_validation_var.get() else tk.DISABLED
    
            
        for widget in self.prefix_controls:
            self.gui_manager._apply_theme_to_widget(widget)
            for child in widget.winfo_children():
                try:
                    if hasattr(child, 'configure'):
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
            if isinstance(widget, (ttk.Entry, ttk.Button, ttk.Scale, ttk.Combobox, ttk.Checkbutton)):
                widget.configure(state=state)
            elif isinstance(widget, tk.Entry):
                if state == 'disabled':
                    widget.configure(state='readonly')
                else:
                    widget.configure(state='normal')
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
    
    def _browse_shared_path(self, string_var, entry_widget=None, is_file=False, path_key=None):
        """Open browser and update path variable.
        
        Args:
            string_var: StringVar to update with selected path
            entry_widget: Optional entry widget to update color
            is_file: Whether browsing for file (True) or folder (False)
            path_key: The FileManager path key (e.g., 'approved_compartments', 'register_excel')
        """
        if is_file:
            # Browse for file
            file_path = filedialog.askopenfilename(
                title=self.t("Select Excel Register"),
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
            )
            if file_path:
                string_var.set(file_path)
                # Update the entry background color if widget was provided
                if entry_widget and os.path.exists(file_path):
                    entry_widget.config(bg=self.gui_manager.theme_colors["accent_valid"])
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
                    entry_widget.config(bg=self.gui_manager.theme_colors["accent_valid"])
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
                register_folder = self.file_manager.shared_base_dir / "Chip Tray Register"
                if register_folder.exists():
                    suggested_dir = str(register_folder)
            
            if suggested_dir:
                # Check if register already exists in suggested location
                if JSONRegisterManager.has_existing_data_static(suggested_dir):
                    summary = JSONRegisterManager.get_data_summary_static(suggested_dir)
                    
                    existing_msg = self.t("A register already exists in this location with:")
                    if summary['has_excel']:
                        existing_msg += f"\n- {self.t('Excel file')}"
                    if summary['compartment_count'] > 0:
                        existing_msg += f"\n- {summary['compartment_count']} {self.t('compartment records')}"
                    if summary['original_count'] > 0:
                        existing_msg += f"\n- {summary['original_count']} {self.t('original image records')}"
                    
                    existing_msg += f"\n\n{self.t('Do you want to create a new register in a different location?')}"
                    
                    if not DialogHelper.confirm_dialog(
                        self.root,
                        self.t("Existing Register Found"),
                        existing_msg
                    ):
                        return
                    
                    # Clear suggested dir to force user to choose new location
                    suggested_dir = None

            file_path = filedialog.asksaveasfilename(
                title=self.t("Create New Excel Register"),
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                initialdir=suggested_dir,
                initialfile=default_filename
            )
            
            if not file_path:
                return
            
            # Get the directory for JSONRegisterManager
            register_dir = os.path.dirname(file_path)
            
            # Check if any register files exist in this location
            existing_files = JSONRegisterManager.check_existing_files_static(register_dir)
            
            if any(existing_files.values()):
                # Build detailed message about what exists
                existing_items = []
                if existing_files['excel']:
                    existing_items.append(self.t("Excel register file"))
                if existing_files['compartment_json']:
                    existing_items.append(self.t("Compartment data"))
                if existing_files['original_json']:
                    existing_items.append(self.t("Original image data"))
                if existing_files['data_folder']:
                    existing_items.append(self.t("Register data folder"))
                
                message = self.t("The following items already exist in this location:") + "\n"
                message += "\n".join(f"• {item}" for item in existing_items)
                message += f"\n\n{self.t('Creating a new register will overwrite these files.')}"
                message += f"\n\n{self.t('Do you want to continue?')}"
                
                if not DialogHelper.confirm_dialog(
                    self.root,
                    self.t("Overwrite Existing Register?"),
                    message,
                    yes_text=self.t("Overwrite"),
                    no_text=self.t("Cancel")
                ):
                    return
                
                # Ask if user wants to create a backup
                if DialogHelper.confirm_dialog(
                    self.root,
                    self.t("Create Backup?"),
                    self.t("Would you like to create a backup of the existing register before overwriting?")
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
                    message_type="error"
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
                        message_type="error"
                    )
                    return
            else:
                # User chose the default name, use it as is
                file_path = expected_excel
            
            # Update the path in FileManager
            self.file_manager.update_shared_path('register_excel', file_path)
            
            # Update the GUI field
            self.register_path_var.set(file_path)
            
            # Find and update the entry widget color
            if hasattr(self, 'shared_collapsible'):
                for frame in self.shared_collapsible.content_frame.winfo_children():
                    if isinstance(frame, ttk.Frame):
                        for child in frame.winfo_children():
                            if isinstance(child, tk.Entry) and child.cget('textvariable') == str(self.register_path_var):
                                child.config(bg=self.gui_manager.theme_colors["accent_valid"])
                                break
            
            # Show success message
            DialogHelper.show_message(
                self.root,
                self.t("Success"),
                self.t("Excel register created successfully!\n\nThe register includes:\n- Compartment Register sheet\n- Original Images Register sheet\n- Power Query setup instructions\n\nData files are stored in the 'Register Data (Do not edit)' subfolder."),
                message_type="info"
            )
            
            # Ask if user wants to open the file
            if DialogHelper.confirm_dialog(
                self.root,
                self.t("Open Register"),
                self.t("Would you like to open the Excel register now?")
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
                message_type="error"
            )
    
    
    def start_processing(self):
        """Start processing in a scheduled manner on the main thread."""
        self.logger.debug("Entered start_processing()")
        folder_path = self.folder_var.get()
        if not folder_path:
            DialogHelper.show_message(self.root, "Error", "Please select a folder", message_type="error")
            return
        
        if not os.path.isdir(folder_path):
            DialogHelper.show_message(self.root, "Error", "Selected path is not a valid folder", message_type="error")
            return
        
        # Update config with current GUI settings
        self.app.config['output_format'] = self.format_var.get()
        self.app.config['save_debug_images'] = self.debug_var.get()
        # self.app.config['compartment_interval'] = self.interval_var.get()
        
        # Update blur detection settings
        self.app.config['enable_blur_detection'] = self.blur_enable_var.get()
        self.app.config['blur_threshold'] = self.blur_threshold_var.get()
        self.app.config['blur_roi_ratio'] = self.blur_roi_var.get()
        self.app.config['flag_blurry_images'] = self.flag_blurry_var.get()
        self.app.config['save_blur_visualizations'] = self.save_blur_viz_var.get()
        self.app.config['blurry_threshold_percentage'] = self.blur_threshold_pct_var.get()
        
        # Update OCR settings
        self.app.config['enable_ocr'] = self.ocr_enable_var.get()
        self.app.config['enable_prefix_validation'] = self.prefix_validation_var.get()
        
        # Parse the prefix string into a list
        prefix_str = self.prefix_var.get()
        if prefix_str:
            # Split by comma and strip whitespace
            prefixes = [p.strip().upper() for p in prefix_str.split(',')]
            # Filter out any empty or invalid entries
            self.app.config['valid_hole_prefixes'] = [p for p in prefixes if p and len(p) == 2 and p.isalpha()]
        
        # Update blur detector with new settings
        self.app.blur_detector.threshold = self.app.config['blur_threshold']
        self.app.blur_detector.roi_ratio = self.app.config['blur_roi_ratio']
        
        # Ensure TesseractManager has the updated config
        self.app.tesseract_manager.config = self.app.config

        self.progress_var.set(0)
        self.app.processing_complete = False
        
        # Disable process button while processing
        if 'process_button' in self.buttons:
            self.buttons['process_button'].set_state("disabled")
        else:
            self.logger.warning("Process button not found in buttons dictionary")

        # Create a processing indicator in the status text
        self.update_status(f"Started processing folder: {folder_path}", "info")
        
        # Process the first image immediately on the main thread, then schedule the rest with after()
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                    if os.path.isfile(os.path.join(folder_path, f)) and 
                    f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
        
        if not image_files:
            self.update_status("No image files found in the selected folder", "warning")
            if 'process_button' in self.buttons:
                self.buttons['process_button'].set_state("normal")
            return
        
        # Store the list of files and initialize counters
        self.files_to_process = image_files
        self.current_file_index = 0
        self.successful_count = 0
        self.failed_count = 0
        self.processing_complete = False
        
        # Start the processing cycle with a single timer
        self.root.after(100, self._process_cycle)

    def _process_cycle(self):
        """Process cycle that handles one image at a time and only continues after completion."""
        current_thread = threading.current_thread()
        self.logger.debug(f"Process cycle executing in thread: {current_thread.name}, is main: {current_thread is threading.main_thread()}")
        
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
                self.progress_queue.put(("Processing image: " + os.path.basename(current_file), progress))
                
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
            if 'process_button' in self.buttons:
                self.buttons['process_button'].set_state("normal")

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
                self.update_status("Processing stopped or failed for file: " + os.path.basename(file_path), "error")
                self.processing_complete = True
                
                # Re-enable process button
                if 'process_button' in self.buttons:
                    self.buttons['process_button'].set_state("normal")
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
            if 'process_button' in self.buttons:
                self.buttons['process_button'].set_state("normal")
    
    def _finish_processing(self):
        """Clean up after all files have been processed."""
        # Update progress display
        summary_msg = f"Processing complete: {self.successful_count} successful, {self.failed_count} failed"
        self.logger.info(summary_msg)
        self.update_status(summary_msg, "success")
        self.progress_var.set(100)
        
        # Re-enable process button
        if 'process_button' in self.buttons:
            self.buttons['process_button'].set_state("normal")
        
        # Prompt to start QAQC review if there are successfully processed images
        if self.successful_count > 0:
            def show_qaqc_prompt():
                start_review = DialogHelper.confirm_dialog(
                    self.root,
                    DialogHelper.t("Processing Complete"),
                    DialogHelper.t(f"Successfully processed {self.successful_count} images.\n\nWould you like to start the QAQC review process now?")
                )
                if start_review and hasattr(self.app, 'qaqc_manager'):
                    self.app.qaqc_manager.start_review_process()
            
            # Use after() to ensure the messagebox appears after GUI updates
            self.root.after(500, show_qaqc_prompt)

    def _start_image_review(self):
        """Start the image review process for pending trays."""
        try:
            # QAQC manager should already be initialized in app
            if not hasattr(self.app, 'qaqc_manager') or self.app.qaqc_manager is None:
                self.logger.error("QAQC manager not initialized in app")
                DialogHelper.show_message(
                    self.root, 
                    self.t("Error"), 
                    self.t("QAQC manager not properly initialized"),
                    message_type="error"
                )
                return
                
            # Use the QAQC manager from app
            qaqc_manager = self.app.qaqc_manager
            
            # Set main GUI reference for status updates if needed
            if not hasattr(qaqc_manager, 'main_gui') or qaqc_manager.main_gui is None:
                qaqc_manager.set_main_gui(self)
                
            # Start the review process - this will check Temp_Review folder automatically
            qaqc_manager.start_review_process()
            
        except Exception as e:
            self.logger.error(f"Error starting image review: {str(e)}")
            DialogHelper.show_message(
                self.root, 
                self.t("Error"), 
                self.t("An error occurred starting the review:") + f" {str(e)}", 
                message_type="error"
            )

    def _start_logging_review(self):
        """Open the logging review dialog with register synchronization."""
        try:
            # ===================================================
            # Set wait cursor on all widgets
            # ===================================================
            self.root.config(cursor="wait")
            self.root.update_idletasks()  # Force immediate update
            
            # Set cursor on all child widgets too
            def set_cursor_recursive(widget):
                try:
                    widget.config(cursor="wait")
                except:
                    pass
                for child in widget.winfo_children():
                    set_cursor_recursive(child)
            
            set_cursor_recursive(self.root)

            # ===================================================
            # ADD: Prompt to validate register before continuing
            # ===================================================
            response = DialogHelper.confirm_dialog(
                self.root,
                DialogHelper.t("Validate Register?"),
                DialogHelper.t("Would you like to validate the register before continuing?") + "\n\n" +
                DialogHelper.t("This will check for missing files and mark any entries with missing images."),
                yes_text=DialogHelper.t("Yes"),
                no_text=DialogHelper.t("No")
            )
            
            if response:
                # User wants to validate - run validation
                self._validate_register_entries()

            # ===================================================
            # Continue with synchronization
            # ===================================================
            
            # First, synchronize registers with shared folders
            self.logger.info("Starting register synchronization...")

            # Create synchronizer without progress callback since we're not using progress dialog
            synchronizer = RegisterSynchronizer(
                self.file_manager,
                self.config
            )
            
            # Get register path for JSON manager using FileManager
            register_path = self.file_manager.get_shared_path('register_excel', create_if_missing=False)
            if register_path:
                base_path = str(register_path.parent)
                synchronizer.set_json_manager(base_path)
            else:
                DialogHelper.show_message(
                    self.root,
                    DialogHelper.t("Error"),
                    DialogHelper.t("Could not find register path"),
                    message_type="error"
                )
                return
            
            try:
                # Run synchronization directly without progress dialog
                sync_results = synchronizer.synchronize_all()
                
            except Exception as sync_error:
                self.logger.error(f"Synchronization failed: {str(sync_error)}")
                self.logger.error(traceback.format_exc())
                
                DialogHelper.show_message(
                    self.root,
                    DialogHelper.t("Synchronization Error"),
                    DialogHelper.t("Synchronization failed:") + f"\n{str(sync_error)}",
                    message_type="error"
                )
                return
            
            # ===================================================
            # Show results
            # ===================================================
            if sync_results['success']:
                # Only show dialog if there were actual changes
                if (sync_results['compartments_added'] > 0 or 
                    sync_results['missing_compartments'] > 0 or 
                    sync_results['originals_added'] > 0 or
                    sync_results.get('originals_updated', 0) > 0):
                    
                    message_lines = [
                        DialogHelper.t("Register synchronization completed:"), 
                        "",
                        DialogHelper.t("Compartments added:") + f" {sync_results['compartments_added']}",
                        DialogHelper.t("Missing compartments:") + f" {sync_results['missing_compartments']}",
                        DialogHelper.t("Original images added:") + f" {sync_results['originals_added']}"
                    ]
                    
                    if sync_results.get('originals_updated', 0) > 0:
                        message_lines.append(DialogHelper.t("Original images updated:") + f" {sync_results['originals_updated']}")
                    
                    DialogHelper.show_message(
                        self.root,
                        DialogHelper.t("Synchronization Complete"),
                        "\n".join(message_lines),
                        message_type="info"
                    )
                else:
                    self.logger.info("No changes needed during synchronization")
            else:
                DialogHelper.show_message(
                    self.root,
                    DialogHelper.t("Synchronization Error"),
                    DialogHelper.t("Error during synchronization:") + f"\n{sync_results['error']}",
                    message_type="error"
                )
                return
            
            # ===================================================
            # Open logging review dialog
            # ===================================================
            self.logger.info("Opening logging review dialog...")
            
            # Now open the logging review dialog
            from gui.logging_review_dialog import LoggingReviewDialog
            
            dialog = LoggingReviewDialog(
                self.root,
                self.file_manager,
                self.gui_manager,
                self.config
            )
            
            # Set JSON manager if available
            if hasattr(synchronizer, 'json_manager') and synchronizer.json_manager:
                dialog.json_manager = synchronizer.json_manager
            
            dialog.show()
            
        except Exception as e:
            self.logger.error(f"Error in _start_logging_review: {str(e)}")
            self.logger.error(traceback.format_exc())
            DialogHelper.show_message(
                self.root,
                DialogHelper.t("Error"),
                DialogHelper.t("Failed to open logging review:") + f"\n{str(e)}",
                message_type="error"
            )
        finally:
            # ===================================================
            # Reset cursor on all widgets
            # ===================================================
            def reset_cursor_recursive(widget):
                try:
                    widget.config(cursor="")
                except:
                    pass
                for child in widget.winfo_children():
                    reset_cursor_recursive(child)
            
            reset_cursor_recursive(self.root)

    def _check_shared_paths(self):
        """Check if shared folder paths are available and accessible."""
        # Check register path using FileManager
        register_path = self.file_manager.get_shared_path('register_excel', create_if_missing=False)
        if not register_path or not register_path.exists():
            DialogHelper.show_message(
                self.root,
                self.t("Shared Folder Path Error"),
                self.t("Excel register path is not available. Please check shared folder settings."),
                message_type="error"
            )
            return False
            
        # Check approved folder path using FileManager
        approved_path = self.file_manager.get_shared_path('approved_compartments', create_if_missing=False)
        if not approved_path or not approved_path.exists():
            DialogHelper.show_message(
                self.root,
                self.t("Shared Folder Path Error"),
                self.t("Approved folder path is not available. Please check shared folder settings."),
                message_type="error"
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
        DialogHelper.show_message(self.root, "File Structure Information", info_message, message_type="info")

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
        DialogHelper.show_message(self.root, "Blur Detection Help", help_text, message_type="info")

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
            text=DialogHelper.t("Select example sharp and blurry images to calibrate the blur detection threshold."),
            wraplength=580,
            justify=tk.LEFT
        )
        instructions.pack(fill=tk.X, pady=(0, 10))
        
        # Frame for sharp images
        sharp_frame = ttk.LabelFrame(main_frame, text=DialogHelper.t("Sharp (Good) Images"), padding=10)
        sharp_frame.pack(fill=tk.X, pady=(0, 10))
        
        sharp_path_var = tk.StringVar()
        sharp_entry = ttk.Entry(sharp_frame, textvariable=sharp_path_var)
        sharp_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        sharp_button = ttk.Button(
            sharp_frame,
            text=DialogHelper.t("Browse"),
            command=lambda: self._select_calibration_images(sharp_path_var)
        )
        sharp_button.pack(side=tk.RIGHT)
        
        # Frame for blurry images
        blurry_frame = ttk.LabelFrame(main_frame, text=DialogHelper.t("Blurry (Poor) Images"), padding=10)
        blurry_frame.pack(fill=tk.X, pady=(0, 10))
        
        blurry_path_var = tk.StringVar()
        blurry_entry = ttk.Entry(blurry_frame, textvariable=blurry_path_var)
        blurry_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        blurry_button = ttk.Button(
            blurry_frame,
            text=DialogHelper.t("Browse"),
            command=lambda: self._select_calibration_images(blurry_path_var)
        )
        blurry_button.pack(side=tk.RIGHT)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text=DialogHelper.t("Calibration Results"), padding=10)
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
            sharp_paths = sharp_path_var.get().split(';')
            blurry_paths = blurry_path_var.get().split(';')
            
            # Validate
            if not sharp_paths or not sharp_paths[0] or not blurry_paths or not blurry_paths[0]:
                DialogHelper.show_message(dialog, "Error", "Please select both sharp and blurry images", message_type="error")
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
                            results_text.insert(tk.END, f"Loaded sharp image: {os.path.basename(path)}\n")
                
                # Load blurry images
                for path in blurry_paths:
                    if path and os.path.exists(path):
                        img = cv2.imread(path)
                        if img is not None:
                            blurry_images.append(img)
                            results_text.insert(tk.END, f"Loaded blurry image: {os.path.basename(path)}\n")
                
                if not sharp_images or not blurry_images:
                    results_text.insert(tk.END, "Error: Failed to load images\n")
                    return
                
                # Calibrate threshold
                results_text.insert(tk.END, "\nCalibrating...\n")
                
                old_threshold = self.blur_detector.threshold
                new_threshold = self.blur_detector.calibrate_threshold(sharp_images, blurry_images)
                
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
                results_text.insert(tk.END, "\nRemember to click 'OK' to apply the new threshold.\n")
                
            except Exception as e:
                results_text.insert(tk.END, f"Error during calibration: {str(e)}\n")
                logger.error(f"Calibration error: {str(e)}")
                logger.error(traceback.format_exc())
        
        calibrate_button = ttk.Button(
            button_frame,
            text=DialogHelper.t("Calibrate"),
            command=calibrate
        )
        calibrate_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # OK button
        ok_button = ttk.Button(
            button_frame,
            text=DialogHelper.t("OK"),
            command=dialog.destroy
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
                ("All files", "*.*")
            ]
        )
        
        if file_paths:
            # Join paths with semicolons for display
            path_var.set(';'.join(file_paths))
    
    def on_generate_trace(self):
        """Handle the 'Generate Drillhole Trace' button click."""
        try:
            # Ask user which method to use
            result = DialogHelper.confirm_dialog(
                self.root,
                DialogHelper.t("Trace Generation Method"),
                DialogHelper.t("Use the new Trace Designer for customizable visualizations?\n\nYes - Open Trace Designer (recommended)\nNo - Use classic method"),
                yes_text=DialogHelper.t("Use Designer"),
                no_text=DialogHelper.t("Classic Method")
            )
            
            if result:  # User chose Designer
                # Use the FileManager's directory structure
                compartment_dir = self.file_manager.dir_structure["chip_compartments"]
                
                # Check if the directory exists
                if not os.path.exists(compartment_dir):
                    DialogHelper.show_message(
                        self.root, 
                        DialogHelper.t("Error"), 
                        DialogHelper.t(f"Compartment directory not found: {compartment_dir}"),
                        message_type="error"
                    )
                    return
                    
                # Initialize trace generator
                trace_generator = DrillholeTraceGenerator(
                    config=self.config,
                    progress_queue=self.progress_queue,
                    root=self.root,
                    file_manager=self.file_manager
                )
                
                # Set app reference for GUI manager access
                trace_generator.app = self.app
                
                # Show designer and get configuration
                config = trace_generator.show_trace_designer()
                
                if config:
                    # Get list of holes to process
                    holes_to_process = []
                    if os.path.exists(compartment_dir):
                        holes_to_process = [d for d in os.listdir(compartment_dir) 
                                          if os.path.isdir(os.path.join(compartment_dir, d))]
                    
                    if not holes_to_process:
                        DialogHelper.show_message(
                            self.root,
                            DialogHelper.t("Info"),
                            DialogHelper.t("No holes found to process."),
                            message_type="info"
                        )
                        return
                        
                    # Generate traces with configuration
                    generated_paths = trace_generator.generate_configured_traces(
                        compartment_dir=compartment_dir,
                        hole_ids=holes_to_process
                    )
                    
                    if generated_paths:
                        DialogHelper.show_message(
                            self.root,
                            DialogHelper.t("Success"),
                            DialogHelper.t(f"Generated {len(generated_paths)} trace images."),
                            message_type="info"
                        )
                        
                        # Ask if user wants to open the directory
                        if DialogHelper.confirm_dialog(
                            self.root,
                            DialogHelper.t("Open Directory"),
                            DialogHelper.t("Would you like to open the directory containing the trace images?")
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
                                    DialogHelper.t(f"Could not open directory: {str(e)}"),
                                    message_type="warning"
                                )
                    else:
                        DialogHelper.show_message(
                            self.root,
                            DialogHelper.t("No Output"),
                            DialogHelper.t("No trace images were generated."),
                            message_type="warning"
                        )
                        
            else:  # User chose classic method
                # Original implementation
                compartment_dir = self.file_manager.dir_structure["chip_compartments"]
                
                if not os.path.exists(compartment_dir):
                    DialogHelper.show_message(
                        self.root,
                        DialogHelper.t("Error"),
                        DialogHelper.t(f"Compartment directory not found: {compartment_dir}"),
                        message_type="error"
                    )
                    return

                # Select CSV file
                csv_path = filedialog.askopenfilename(
                    title=DialogHelper.t("Select CSV File"),
                    filetypes=[("CSV Files", "*.csv")]
                )
                if not csv_path:
                    return

                # Rest of original implementation...
                trace_generator = DrillholeTraceGenerator(
                    config=self.config, 
                    progress_queue=self.progress_queue,
                    root=self.root,
                    file_manager=self.file_manager
                )

                # Let user select optional columns 
                csv_columns = trace_generator.get_csv_columns(csv_path)
                if not csv_columns:
                    DialogHelper.show_message(
                        self.root,
                        DialogHelper.t("CSV Error"),
                        DialogHelper.t("Could not read columns from CSV."),
                        message_type="error"
                    )
                    return

                selected_columns = trace_generator.select_csv_columns(csv_columns)
                
                # Get the drill traces directory
                traces_dir = self.file_manager.dir_structure["drill_traces"]
                
                # Get list of existing trace files
                existing_traces = set()
                if os.path.exists(traces_dir):
                    existing_traces = {os.path.splitext(f)[0].split('_')[0] for f in os.listdir(traces_dir) 
                                    if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg')) and '_Trace' in f}
                
                # Get list of holes from compartment directories
                hole_dirs = [d for d in os.listdir(compartment_dir) if os.path.isdir(os.path.join(compartment_dir, d))]
                
                # Filter to holes that don't have traces
                holes_to_process = [hole for hole in hole_dirs if hole not in existing_traces]
                
                if not holes_to_process:
                    DialogHelper.show_message(
                        self.root,
                        DialogHelper.t("Info"),
                        DialogHelper.t("All holes already have trace images."),
                        message_type="info"
                    )
                    return
                
                # Ask user for confirmation
                if not DialogHelper.confirm_dialog(
                    self.root,
                    DialogHelper.t("Confirm"),
                    DialogHelper.t(f"Found {len(holes_to_process)} holes without trace images. Process them all?")
                ):
                    return
                
                # Run the trace generation
                generated_paths = trace_generator.process_selected_holes(
                    compartment_dir=compartment_dir,
                    csv_path=csv_path,
                    selected_columns=selected_columns,
                    hole_ids=holes_to_process
                )

                if generated_paths:
                    DialogHelper.show_message(
                        self.root,
                        DialogHelper.t("Success"),
                        DialogHelper.t(f"Generated {len(generated_paths)} drillhole trace images."),
                        message_type="info"
                    )
                    
                    # Ask if the user wants to open the directory
                    if DialogHelper.confirm_dialog(
                        self.root,
                        DialogHelper.t("Open Directory"),
                        DialogHelper.t("Would you like to open the directory containing the trace images?")
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
                                message_type="warning"
                            )
                else:
                    DialogHelper.show_message(
                        self.root,
                        DialogHelper.t("No Output"),
                        DialogHelper.t("No drillhole trace images were generated."),
                        message_type="warning"
                    )
                    
        except Exception as e:
            DialogHelper.show_message(
                self.root,
                DialogHelper.t("Error"),
                DialogHelper.t(f"An error occurred: {str(e)}"),
                message_type="error"
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
        if not hasattr(self, 'root') or not self.root.winfo_exists():
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
                        if any(error_term in message.lower() for error_term in ["error", "failed", "not enough"]):
                            self.update_status(message, "error")
                        elif any(warning_term in message.lower() for warning_term in ["warning", "missing"]):
                            self.update_status(message, "warning")
                        elif any(success_term in message.lower() for success_term in ["success", "complete", "saved"]):
                            self.update_status(message, "success")
                        else:
                            self.update_status(message, "info")
                            
                        # If processing is complete, re-enable the process button
                        if self.processing_complete:
                            if 'process_button' in self.buttons:
                                self.buttons['process_button'].set_state("normal")
                                
                except Exception as e:
                    # Log the error but continue processing
                    self.logger.error(f"Error processing queue item: {str(e)}")
                    
        except Exception as e:
            # Log the outer error
            self.logger.error(f"Error in progress check: {str(e)}")
        
        # Always reschedule if requested, even if exceptions occurred above
        if schedule_next and hasattr(self, 'root') and self.root.winfo_exists():
            try:
                self.after_id = self.root.after(500, lambda: self.check_progress(schedule_next=True))
            except Exception as e:
                # Last resort logging
                self.logger.critical(f"Failed to reschedule progress check: {str(e)}")
    
    def browse_folder(self):
        """Open folder browser dialog and update the folder entry."""
        folder_path = filedialog.askdirectory(title="Select folder with chip tray photos")
        if folder_path:
            self.folder_var.set(folder_path)
            
            # Count image files in the selected folder
            image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
            image_files = [f for f in os.listdir(folder_path) 
                        if os.path.isfile(os.path.join(folder_path, f)) and 
                        f.lower().endswith(image_extensions)]
            
            # Create message about found files
            message = f"{len(image_files)} image files found in the selected folder"
            
            # Log to the GUI
            # self.update_status(message, "info")
            
            # Also add to progress queue if it exists
            if hasattr(self, 'progress_queue'):
                self.progress_queue.put((message, None))
                self.check_progress(schedule_next=False)


    def quit_app(self):
        """Close the application after properly terminating threads."""
        try:
            # Cancel any pending after callbacks
            if hasattr(self, 'after_id') and self.after_id:
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