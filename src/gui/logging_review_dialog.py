"""
Logging Review Dialog for reviewing and logging compartment images.
Handles data entry with JSON register integration.
"""

import os
import json
import logging
import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import cv2
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import threading
import traceback
from tkinter import filedialog
from gui.dialog_helper import DialogHelper
from gui.widgets.modern_button import ModernButton
from processing.LoggingReviewStep.color_map_manager import ColorMapManager, ColorMapType
from utils.image_pan_zoom_handler import ImagePanZoomHandler
from PIL import Image, ImageTk


class LoggingReviewDialog:
    """
    Dialog for reviewing compartment images and logging geological data.
    Uses JSON register for data storage.
    """
    
    def __init__(self, parent, file_manager, gui_manager, config, json_manager=None):
        """Initialize the logging review dialog."""
        self.parent = parent
        self.file_manager = file_manager
        self.gui_manager = gui_manager
        self.config = config
        self.json_manager = json_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize color map manager
        self.color_map_manager = ColorMapManager()
        self.color_map_manager.create_default_presets()
        
        # Data management
        self.register_data = None
        self.filtered_data = None
        self.current_index = 0
        self.current_image = None
        self.previous_toggle_states = {}
        self.external_data = None
        self.external_data_path = None
        self.current_review_info = {}  # Track current review status

        # UI references
        self.dialog = None
        self.image_label = None
        self.data_displays = []
        self.comment_text = None
        self.toggle_vars = {}
        self.review_status_label = None
        
        # Session management
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.changes_made = False
        self.unsaved_changes = False  # Track unsaved changes for current image
        self.user_interacted = {}  # Track which compartments user actually interacted with
        self.session_review_numbers = {}  # Track review numbers assigned in this session
        self.toggle_interaction_state = {}  # Track if user has interacted with each toggle
        
        # Auto-save timer
        self.auto_save_timer = None
        self.auto_save_interval = 30000  # 30 seconds in milliseconds
        self.last_save_time = datetime.now()
        self.save_in_progress = False
        
    def show(self):
        """Show the logging review dialog."""
        # Load register data first
        if not self._load_register_data():
            return
            
        # Check for existing images
        self._check_existing_images()
        
        # ===================================================
        # CHANGED: Load external data BEFORE creating dialog
        # ===================================================
        if not self._load_external_data():
            return
        
        # NOW create dialog after all data is loaded
        self._create_dialog()
        
        # Apply initial filters
        self._apply_filters()
        
        # Load first image
        if not self.filtered_data.empty:
            self._load_current_image()
        
        # Start auto-save timer
        self._start_auto_save()
        
        # Start the dialog
        self.dialog.wait_window()


    def _load_external_data(self):
            """Load external data file for category displays."""
            
            
            def reset_cursor_recursive(widget):
                try:
                    widget.config(cursor="")
                except:
                    pass
                for child in widget.winfo_children():
                    reset_cursor_recursive(child)
            
            if hasattr(self.parent, 'winfo_children'):
                reset_cursor_recursive(self.parent)
                self.parent.update_idletasks()

            # ===================================================
            # CHANGED: Skip confirmation dialog and directly prompt for file
            # ===================================================
            # Ask user to select file
            file_path = filedialog.askopenfilename(
                parent=self.parent,
                title=DialogHelper.t("Select Data File for Display"),
                filetypes=[
                    ("All Supported", "*.csv;*.txt;*.xlsx;*.xls"),
                    ("CSV files", "*.csv"),
                    ("Text files", "*.txt"),
                    ("Excel files", "*.xlsx;*.xls"),
                    ("All files", "*.*")
                ]
            )
            
            if not file_path:
                # User cancelled - we need external data, so return False
                DialogHelper.show_message(
                    self.parent,
                    DialogHelper.t("External Data Required"),
                    DialogHelper.t("An external data file is required for logging review. Please select a file."),
                    message_type="warning"
                )
                return False
                
            try:
                # Load file based on extension
                file_ext = os.path.splitext(file_path)[1].lower()
                
                if file_ext == '.csv':
                    self.external_data = pd.read_csv(file_path)
                elif file_ext == '.txt':
                    # Try tab-delimited first
                    self.external_data = pd.read_csv(file_path, sep='\t')
                elif file_ext in ['.xlsx', '.xls']:
                    self.external_data = pd.read_excel(file_path)
                else:
                    DialogHelper.show_message(
                        self.parent,
                        DialogHelper.t("Error"),
                        DialogHelper.t("Unsupported file format. Please select a CSV, TXT, or Excel file."),
                        message_type="error"
                    )
                    return False
                    
                self.external_data_path = file_path
                
                # Merge with register data on HoleID, From, To
                if all(col in self.external_data.columns for col in ['HoleID', 'From', 'To']):
                    # Ensure data types match for merge
                    self.external_data['From'] = pd.to_numeric(self.external_data['From'], errors='coerce')
                    self.external_data['To'] = pd.to_numeric(self.external_data['To'], errors='coerce')
                    
                    # Merge on these columns
                    self.register_data = pd.merge(
                        self.register_data,
                        self.external_data,
                        on=['HoleID', 'From', 'To'],
                        how='left',
                        suffixes=('', '_external')
                    )
                    self.logger.info(f"Merged external data with register: {len(self.external_data.columns)} new columns")
                else:
                    DialogHelper.show_message(
                        self.parent,
                        DialogHelper.t("Warning"),
                        DialogHelper.t("External file must contain HoleID, From, and To columns for merging."),
                        message_type="warning"
                    )
                    return False
                    
                return True
                
            except Exception as e:
                self.logger.error(f"Error loading external data: {str(e)}")
                self.logger.error(traceback.format_exc())
                DialogHelper.show_message(
                    self.parent,
                    DialogHelper.t("Error"),
                    DialogHelper.t("Failed to load file:") + f" {str(e)}",
                    message_type="error"
                )
                return False


    def _load_register_data(self):
        """Load data from the JSON register."""
        try:
            if self.json_manager:
                # Load compartment data from JSON
                self.register_data = self.json_manager.get_compartment_data()
                
                if self.register_data.empty:
                    self.logger.info("No compartment data found in register")
                    DialogHelper.show_message(
                        self.parent,
                        DialogHelper.t("No Data"),
                        DialogHelper.t("No compartment data found in register. Process some images first."),
                        message_type="info"
                    )
                    return False
                    
                # Ensure numeric types for From/To columns
                self.register_data['From'] = pd.to_numeric(self.register_data['From'], errors='coerce')
                self.register_data['To'] = pd.to_numeric(self.register_data['To'], errors='coerce')
                    
                self.logger.info(f"Loaded {len(self.register_data)} compartments from register")
                return True
                
            else:
                self.logger.error("No JSON manager available")
                DialogHelper.show_message(
                    self.parent,
                    DialogHelper.t("Error"),
                    DialogHelper.t("Register manager not available. Cannot load data."),
                    message_type="error"
                )
                return False
                
        except Exception as e:
            self.logger.error(f"Unexpected error loading register data: {e}")
            self.logger.error(traceback.format_exc())
            
            DialogHelper.show_message(
                self.parent,
                DialogHelper.t("Error"),
                DialogHelper.t("Unexpected error loading register."),
                message_type="error"
            )
            return False
    
    def _check_existing_images(self):
        """Check which intervals have existing images in the approved folder."""

        approved_path = self.file_manager.get_shared_path('approved_compartments', create_if_missing=False)
        if not approved_path or not approved_path.exists():
            self.logger.warning("Could not find approved folder path")
            return
            
        # Initialize ImagePath column if needed
        if 'ImagePath' not in self.register_data.columns:
            self.register_data['ImagePath'] = None
            
        # Look for images and match to register
        found_count = 0
        for idx, row in self.register_data.iterrows():
            hole_id = row['HoleID']
            to_depth = int(row['To'])
            
            # Find the image
            image_path = self._find_compartment_image(hole_id, to_depth)
            if image_path:
                self.register_data.at[idx, 'ImagePath'] = image_path
                found_count += 1
                
                # Try to extract wet/dry status if present
                if '_Wet.' in image_path:
                    self.register_data.at[idx, 'WetDryStatus'] = 'Wet'
                elif '_Dry.' in image_path:
                    self.register_data.at[idx, 'WetDryStatus'] = 'Dry'
                    
        self.logger.info(f"Found {found_count} images for {len(self.register_data)} compartments")
                    
    def _create_dialog(self):
        """Create the main dialog window."""
        # Create dialog using DialogHelper
        self.dialog = DialogHelper.create_dialog(
            self.parent,
            DialogHelper.t("Logging Review"),
            modal=True,
            size_ratio=0.9,
            min_width=1200,
            min_height=800
        )
        
        # Apply theme
        self.dialog.configure(bg=self.gui_manager.theme_colors["background"])

        # Handle window close event
        self.dialog.protocol("WM_DELETE_WINDOW", self._quit_dialog)

        # NEW CODE: Bind arrow keys for navigation
        self.dialog.bind('<Left>', lambda e: self._navigate_with_save('previous'))
        self.dialog.bind('<Right>', lambda e: self._navigate_with_save('next'))
        self.dialog.bind('<Control-s>', lambda e: self._save_changes())
        
        # ===================================================
        # ADD: Keyboard shortcuts for zoom
        # ===================================================
        self.dialog.bind('<Control-plus>', lambda e: self._zoom_in())
        self.dialog.bind('<Control-equal>', lambda e: self._zoom_in())  # For keyboards without numpad
        self.dialog.bind('<Control-minus>', lambda e: self._zoom_out())
        self.dialog.bind('<Control-0>', lambda e: self._reset_image_view())
        
        # Focus on dialog to capture key events
        self.dialog.focus_set()
        
        # Start auto-save timer
        self._start_auto_save_timer()
        
        # Start auto-save timer
        self._start_auto_save_timer()
        
        
        # Create main container
        main_frame = ttk.Frame(self.dialog, style='Main.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create content area
        content_frame = ttk.Frame(main_frame, style='Content.TFrame')
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Configure grid for narrower image display (chip trays are narrow)
        content_frame.grid_columnconfigure(0, weight=2)  # Image side gets 2/5 (narrower)
        content_frame.grid_columnconfigure(1, weight=3)  # Data side gets 3/5 (wider)
        content_frame.grid_rowconfigure(0, weight=1)

        # Left side - Image display
        image_frame = ttk.Frame(content_frame, style='Content.TFrame')
        image_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        self._create_image_display(image_frame)

        # Right side - Data display and controls
        data_frame = ttk.Frame(content_frame, style='Content.TFrame')
        data_frame.grid(row=0, column=1, sticky="nsew")

        # ===================================================
        # ADDED: Create toolbar (filters) ABOVE data display
        # ===================================================
        self._create_toolbar(data_frame)
        
        # Then create data display below filters
        self._create_data_display(data_frame)
        
        # Bottom controls
        self._create_bottom_controls(main_frame)

        # Bind resize event
        self.dialog.bind("<Configure>", self._on_dialog_resize)
    
    
    def _on_dialog_resize(self, event=None):
        """Handle dialog resize to update image display."""
        if hasattr(self, 'current_pil_image') and self.current_pil_image:
            # ===================================================
            # MODIFIED: Let pan/zoom handler handle resize
            # ===================================================
            # The pan/zoom handler will automatically handle resize events
            pass

    def _navigate_with_save(self, direction):
        """Navigate with automatic saving if changes exist."""
        if direction == 'previous' and self.current_index > 0:
            if self.changes_made:
                self._save_current_data()
            self._previous_image()
        elif direction == 'next' and self.current_index < len(self.filtered_data) - 1:
            if self.changes_made:
                self._save_current_data()
            self._next_image()

    def _create_toolbar(self, parent):
        """Create the toolbar with filtering options."""
        # ===================================================
        # REPLACE THE EXISTING TOOLBAR CODE WITH THIS:
        # ===================================================
        
        # Create collapsible frame for filters
        self.filter_frame = self.gui_manager.create_collapsible_frame(
            parent,
            title="Filters - Showing 0 of 0 intervals",
            expanded=True
        )
        
        # Pack it tightly
        self.filter_frame.pack_configure(fill=tk.X, pady=(0, 5))
        
        # Get the content frame from collapsible frame
        filter_content = self.filter_frame.content_frame
        
        # Container for filter rows
        self.filter_rows_container = ttk.Frame(filter_content)
        self.filter_rows_container.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # List to track filter row objects
        self.filter_rows = []
        
        # Buttons frame
        buttons_frame = ttk.Frame(filter_content)
        buttons_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Add filter button
        add_filter_btn = ModernButton(
            buttons_frame,
            text=DialogHelper.t("Add Filter"),
            color=self.gui_manager.theme_colors["accent_green"],
            command=self._add_filter_row,
            theme_colors=self.gui_manager.theme_colors
        )
        add_filter_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Apply filters button
        apply_filters_btn = ModernButton(
            buttons_frame,
            text=DialogHelper.t("Apply Filters"),
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._apply_filters,
            theme_colors=self.gui_manager.theme_colors
        )
        apply_filters_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Clear filters button
        clear_filters_btn = ModernButton(
            buttons_frame,
            text=DialogHelper.t("Clear All"),
            color=self.gui_manager.theme_colors["secondary_bg"],
            command=self._clear_all_filters,
            theme_colors=self.gui_manager.theme_colors
        )
        clear_filters_btn.pack(side=tk.LEFT)
        
        # Toggles frame
        toggles_frame = ttk.Frame(filter_content)
        toggles_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Only with images checkbox
        self.only_with_images_var = tk.BooleanVar(value=True)
        images_check = self.gui_manager.create_custom_checkbox(
            toggles_frame,
            DialogHelper.t("Only intervals with images"),
            self.only_with_images_var,
            command=self._apply_filters
        )
        images_check.pack(side=tk.LEFT, padx=(0, 20))
        
        # Hide reviewed by me checkbox
        self.hide_reviewed_var = tk.BooleanVar(value=False)
        hide_reviewed_check = self.gui_manager.create_custom_checkbox(
            toggles_frame,
            DialogHelper.t("Hide reviewed by me"),
            self.hide_reviewed_var,
            command=self._apply_filters
        )
        hide_reviewed_check.pack(side=tk.LEFT)
        
        # Initialize column info
        self._update_columns_info()
        
        # Add default filters for Project and HoleID
        self._add_default_filters()
    
    def _update_columns_info(self):
        """Update information about columns and their data types."""
        self.columns_info = {}
        
        for col in self.register_data.columns:
            if col in ['ImagePath', 'ReviewedBy', 'Review_Date']:  # Skip system columns
                continue
                
            if pd.api.types.is_numeric_dtype(self.register_data[col]):
                self.columns_info[col] = 'numeric'
            else:
                self.columns_info[col] = 'text'
                
    def _add_default_filters(self):
        """Add default filter rows for Project and HoleID."""
        # Add HoleID filter
        row = self._add_filter_row()
        if row:
            row.column_var.set('HoleID')
            row.operator_var.set('is')
            
    def _add_filter_row(self):
        """Add a new filter row."""
        # Import here to avoid circular imports
        from gui.widgets.dynamic_filter_row import DynamicFilterRow
        
        index = len(self.filter_rows)
        filter_row = DynamicFilterRow(
            self.filter_rows_container,
            self.gui_manager,
            self.columns_info,
            self.register_data,  # Pass the register data
            self._remove_filter_row,
            index
        )
        
        self.filter_rows.append(filter_row)
        return filter_row

    def _remove_filter_row(self, index):
        """Remove a filter row."""
        if index < len(self.filter_rows):
            self.filter_rows[index].destroy()
            self.filter_rows.pop(index)
            
            # Re-index remaining rows
            for i, row in enumerate(self.filter_rows):
                row.index = i
                
    def _clear_all_filters(self):
        """Clear all filter rows."""
        for row in self.filter_rows[:]:  # Copy list to avoid modification during iteration
            row.destroy()
        self.filter_rows.clear()
        
        # Re-add default filters
        self._add_default_filters()
        
        # Apply to show all results
        self._apply_filters()

    def _update_hole_filter(self):
        """Update hole ID filter based on selected project code."""
        project = self.project_var.get()
        
        if project == 'All':
            hole_ids = sorted(self.register_data['HoleID'].unique().tolist())
        else:
            # Filter hole IDs by project code
            hole_ids = sorted([
                h for h in self.register_data['HoleID'].unique()
                if isinstance(h, str) and h.startswith(project)
            ])
        
        # Update combo values
        self.hole_combo['values'] = ['All'] + hole_ids
        
        # Reset selection if current selection is invalid
        if self.hole_var.get() not in ['All'] + hole_ids:
            self.hole_var.set('All')
    
    def _start_auto_save_timer(self):
        """Start the auto-save timer."""
        if self.auto_save_timer:
            self.dialog.after_cancel(self.auto_save_timer)
        
        self.auto_save_timer = self.dialog.after(
            self.auto_save_interval, 
            self._auto_save_callback
        )
    
    def _auto_save_callback(self):
        """Auto-save callback function."""
        if self.changes_made and not self.save_in_progress:
            self.logger.info("Auto-saving changes...")
            self._perform_auto_save()
        
        # Restart timer
        self._start_auto_save_timer()
    
    def _perform_auto_save(self):
        """Perform auto-save with JSON register."""
        if self.save_in_progress:
            return
            
        self.save_in_progress = True
        self.auto_save_label.config(
            text=DialogHelper.t("Auto-saving..."),
            foreground=self.gui_manager.theme_colors["accent_blue"]
        )
        self.dialog.update_idletasks()
        
        try:
            # Save current data first
            self._save_current_data()
            
            if hasattr(self, 'json_manager') and self.json_manager:
                # Get current row for context
                if not self.filtered_data.empty and self.current_index < len(self.filtered_data):
                    current_row = self.filtered_data.iloc[self.current_index]
                    
                    # Use JSON manager to update the current compartment review
                    # Build kwargs for all toggle fields
                    toggle_kwargs = {}
                    for name, var in self.toggle_vars.items():
                        # Convert toggle name to valid field name (replace spaces and special chars)
                        field_name = name.replace(' ', '_').replace('/', '_').replace('+', 'plus')
                        toggle_kwargs[field_name] = var.get()
                    
                    # Update review
                    success = self.json_manager.update_compartment_review(
                        hole_id=current_row['HoleID'],
                        depth_from=int(current_row['From']),
                        depth_to=int(current_row['To']),
                        comments=self.comment_text.get(1.0, tk.END).strip() or None,
                        **toggle_kwargs
                    )
                    
                    if success:
                        self.changes_made = False
                        self.last_save_time = datetime.now()
                        self.auto_save_label.config(
                            text=DialogHelper.t("Auto-saved") + f" {self.last_save_time.strftime('%H:%M:%S')}",
                            foreground=self.gui_manager.theme_colors["accent_green"]
                        )
                    else:
                        raise Exception("Failed to update compartment review")
                        
            else:
                # Fallback to temp file method
                self._save_to_temp_file()
                
        except Exception as e:
            self.logger.error(f"Auto-save failed: {str(e)}")
            self.auto_save_label.config(
                text=DialogHelper.t("Auto-save failed"),
                foreground=self.gui_manager.theme_colors["accent_red"]
            )
            # Don't show dialog for auto-save failures
            
        finally:
            self.save_in_progress = False
    
    def _save_to_temp_file(self):
        """Save to temporary file (fallback method)."""
        temp_dir = os.path.join(self.file_manager.base_dir, "Temp_Logging")
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_filename = f"logging_session_{self.session_id}.json"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        # Convert to records format
        changes = self.register_data.to_dict('records')
        
        with open(temp_path, 'w') as f:
            json.dump({
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'user': os.environ.get('USERNAME', 'unknown'),
                'changes': changes
            }, f, indent=2)
        
        self.changes_made = False
        self.last_save_time = datetime.now()
        self.auto_save_label.config(
            text=DialogHelper.t("Auto-saved") + f" {self.last_save_time.strftime('%H:%M:%S')}",
            foreground=self.gui_manager.theme_colors["accent_green"]
        )

    def _reset_image_view(self):
        """Reset the image zoom and center it."""
        if hasattr(self, 'pan_zoom_handler'):
            self.pan_zoom_handler.reset_view()

    def _create_image_display(self, parent):
        """Create the image display area."""
        # ===================================================
        # MODIFIED: Add container for info label and reset button
        # ===================================================
        info_container = ttk.Frame(parent)
        info_container.pack(fill=tk.X, pady=(0, 10))
        
        # Image info label
        self.image_info_label = ttk.Label(
            info_container,
            text="",
            style='Content.TLabel',
            font=('Arial', 12, 'bold')
        )
        self.image_info_label.pack(side=tk.LEFT, expand=True)
        
        # Reset zoom button
        self.reset_zoom_btn = ModernButton(
            info_container,
            text=DialogHelper.t("Reset Zoom"),
            color=self.gui_manager.theme_colors["secondary_bg"],
            command=self._reset_image_view,
            theme_colors=self.gui_manager.theme_colors
        )
        self.reset_zoom_btn.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Image frame with border
        image_container = tk.Frame(
            parent,
            bg=self.gui_manager.theme_colors["field_border"],
            relief=tk.SUNKEN,
            bd=2
        )
        image_container.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas for image display with pan/zoom
        self.image_canvas = tk.Canvas(
            image_container,
            bg=self.gui_manager.theme_colors["secondary_bg"],
            highlightthickness=0,
            cursor="hand2"
        )
        self.image_canvas.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Initialize pan/zoom handler
        self.pan_zoom_handler = ImagePanZoomHandler(
            self.image_canvas,
            self.gui_manager.theme_colors
        )
        
        # Initialize image reference
        self.current_pil_image = None
        
    def _create_data_display(self, parent):
        """Create the data display panels."""
        # Header info
        header_frame = ttk.Frame(parent, style='Content.TFrame')
        header_frame.pack(fill=tk.X, pady=(5, 10))
        
        self.interval_label = ttk.Label(
            header_frame,
            text="",
            style='Title.TLabel'
        )
        self.interval_label.pack()
        
        # Review status indicator
        self.review_status_label = ttk.Label(
            header_frame,
            text="",
            style='Content.TLabel',
            font=('Arial', 10, 'italic')
        )
        self.review_status_label.pack(pady=(5, 0))
        
        # Auto-save status label
        self.auto_save_label = ttk.Label(
            header_frame,
            text="",
            style='Content.TLabel',
            font=('Arial', 9),
            foreground=self.gui_manager.theme_colors["accent_green"]
        )
        self.auto_save_label.pack(pady=(2, 0))
        
        # ===================================================
        # Main Data Container
        # ===================================================
        data_container = ttk.LabelFrame(
            parent,
            text=DialogHelper.t("Data"),
            style='TLabelframe'
        )
        data_container.pack(fill=tk.X, pady=(0, 5))

        # Create a header frame inside the data container for better control
        data_header_frame = ttk.Frame(data_container)
        data_header_frame.pack(fill=tk.X, padx=5, pady=(5, 0))

        # Edit button - positioned in the header
        self.edit_mode = tk.BooleanVar(value=False)
        self.edit_btn = ModernButton(
            data_header_frame,
            text=DialogHelper.t("Edit"),
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._toggle_edit_mode,
            theme_colors=self.gui_manager.theme_colors
        )
        self.edit_btn.pack(side=tk.RIGHT)

        # Categories Section
        if hasattr(self, 'external_data') and self.external_data is not None:
            categories_frame = ttk.Frame(data_container)
            categories_frame.pack(fill=tk.X, padx=5, pady=(5, 3))
            
            ttk.Label(
                categories_frame,
                text=DialogHelper.t("Categories:"),
                style='Content.TLabel',
                font=('Arial', 10, 'bold')
            ).pack(side=tk.LEFT, padx=(0, 10))
            
            # Container for category displays
            self.category_container = ttk.Frame(categories_frame)
            self.category_container.pack(side=tk.LEFT, fill=tk.X, expand=True, anchor='center')
            
            # Add button (initially hidden)
            self.add_cat_btn = tk.Button(
                categories_frame,
                text="+",
                command=self._add_category_display,
                bg=self.gui_manager.theme_colors["accent_green"],
                fg="white",
                font=("Arial", 12, "bold"),
                bd=0,
                padx=8,
                pady=2,
                cursor="hand2"
            )
            # Don't pack yet - will be shown in edit mode
            
            # Create initial category displays
            self.category_displays = []
            for i in range(2):
                self._add_category_display()
            
            # ===================================================
            # Add separator line between categories and numeric
            # ===================================================
            separator = ttk.Separator(data_container, orient='horizontal')
            separator.pack(fill=tk.X, padx=10, pady=5)


        # Numeric Section
        numeric_frame = ttk.Frame(data_container)
        numeric_frame.pack(fill=tk.X, padx=5, pady=(3, 5))

        
        ttk.Label(
            numeric_frame,
            text=DialogHelper.t("Numeric:"),
            style='Content.TLabel',
            font=('Arial', 10, 'bold')
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        # Container for numeric displays
        self.numeric_container = ttk.Frame(numeric_frame)
        self.numeric_container.pack(side=tk.LEFT, fill=tk.X, expand=True, anchor='center')
        
        # Add button (initially hidden)
        self.add_num_btn = tk.Button(
            numeric_frame,
            text="+",
            command=self._add_numeric_display,
            bg=self.gui_manager.theme_colors["accent_green"],
            fg="white",
            font=("Arial", 12, "bold"),
            bd=0,
            padx=8,
            pady=2,
            cursor="hand2"
        )
        # Don't pack yet - will be shown in edit mode
        
        # Create initial numeric displays
        self.numeric_displays = []
        self._create_default_numeric_displays()
        
        # Hide remove buttons initially
        self._update_edit_controls()
        
        # Comments section (rest remains the same)
        comments_frame = ttk.LabelFrame(
            parent,
            text=DialogHelper.t("Comments"),
            style='TLabelframe'
        )
        comments_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.comment_text = tk.Text(
            comments_frame,
            height=3,
            wrap=tk.WORD,
            bg=self.gui_manager.theme_colors["field_bg"],
            fg=self.gui_manager.theme_colors["text"],
            insertbackground=self.gui_manager.theme_colors["text"]
        )
        self.comment_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Track changes in comment text
        self.comment_text.bind('<KeyRelease>', lambda e: self._mark_unsaved_changes())
        self.comment_text.bind('<Key>', self._on_comment_interaction)
        
        # Toggles section
        toggles_frame = ttk.LabelFrame(
            parent,
            text=DialogHelper.t("Toggles"),
            style='TLabelframe'
        )
        toggles_frame.pack(fill=tk.X)
        
        # Same as Before button
        button_frame = ttk.Frame(toggles_frame, style='Content.TFrame')
        button_frame.pack(fill=tk.X, padx=5, pady=(5, 0))
        
        self.same_as_before_button = ModernButton(
            button_frame,
            text=DialogHelper.t("Same as Before"),
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._copy_previous_toggles,
            icon="📋",
            theme_colors=self.gui_manager.theme_colors
        )
        self.same_as_before_button.pack(side=tk.LEFT)
        self.same_as_before_button.set_state('disabled')  # Initially disabled
        
        # Toggles container
        toggles_container = ttk.Frame(toggles_frame, style='Content.TFrame')
        toggles_container.pack(fill=tk.X, padx=5, pady=5)
        
        # Get toggle names from config
        toggle_names = self.config.get('review_toggles', [
            'Bad Image', 'Banded', 'Porous', 'Biscuity', '+ QZ', '+ CHH/M'
        ])
        
        # Create a grid frame for toggles
        toggle_grid = ttk.Frame(toggles_container, style='Content.TFrame')
        toggle_grid.pack(fill=tk.X)
        
        # Determine columns based on number of toggles
        columns = 4 if len(toggle_names) > 6 else 3

        # ===================================================
        # MODIFIED: Import three-state toggle at the top of the method
        # ===================================================
        from gui.widgets.three_state_toggle import ThreeStateToggle

        # Add Wet/Dry status as a special toggle
        self.wet_dry_var = tk.StringVar(value="Unknown")

        wet_dry_frame = ttk.Frame(toggles_container, style='Content.TFrame')
        wet_dry_frame.pack(fill=tk.X, pady=(0, 10))

        wet_dry_label = ttk.Label(
            wet_dry_frame,
            text=DialogHelper.t("Sample Status:"),
            style='Content.TLabel',
            font=('Arial', 11, 'bold')
        )
        wet_dry_label.pack(side=tk.LEFT, padx=(0, 10))

        # Create button container
        wet_dry_button_frame = ttk.Frame(wet_dry_frame)
        wet_dry_button_frame.pack(side=tk.LEFT)

        # Create modern toggle buttons
        self.wet_dry_buttons = {}
        for i, status in enumerate(["Wet", "Dry", "Unknown"]):
            btn = ModernButton(
                wet_dry_button_frame,
                text=DialogHelper.t(status),
                color=self.gui_manager.theme_colors["secondary_bg"],
                command=lambda s=status: self._set_wet_dry_status(s),
                theme_colors=self.gui_manager.theme_colors
            )
            btn.pack(side=tk.LEFT, padx=(0, 5) if i < 2 else 0)
            self.wet_dry_buttons[status] = btn

        # Set initial state
        self._update_wet_dry_buttons()
        
        # ===================================================
        # REPLACED: Regular toggles now use ThreeStateToggle
        # ===================================================
        # Regular toggles below
        for i, name in enumerate(toggle_names):
            row = i // columns
            col = i % columns
            
            # Create three-state toggle with callback to mark interaction
            toggle = ThreeStateToggle(
                toggle_grid,
                DialogHelper.t(name),
                self.gui_manager.theme_colors,
                initial_value=None,
                on_change=lambda n=name: self._on_toggle_interaction(n)
            )
            toggle.grid(row=row, column=col, padx=3, pady=3, sticky='ew')
            
            # Store reference to the toggle widget (not a BooleanVar)
            self.toggle_vars[name] = toggle
            
        # Configure grid columns to expand evenly
        for col in range(columns):
            toggle_grid.columnconfigure(col, weight=1)
    

    def _set_wet_dry_status(self, status):
        """Set wet/dry status and update button appearance."""
        self.wet_dry_var.set(status)
        self._update_wet_dry_buttons()
        self._on_wet_dry_change()
        
    def _on_wet_dry_change(self):
        """Handle wet/dry status change."""
        # Mark that user has interacted with this compartment
        if not self.filtered_data.empty and self.current_index < len(self.filtered_data):
            current_row = self.filtered_data.iloc[self.current_index]
            key = (current_row['HoleID'], int(current_row['From']), int(current_row['To']))
            
            if key not in self.user_interacted:
                self.user_interacted[key] = {
                    'toggles_touched': set(),
                    'comments_touched': False,
                    'wet_dry_touched': False,
                    'first_interaction_time': datetime.now()
                }
            
            self.user_interacted[key]['wet_dry_touched'] = True
            
        # Mark unsaved changes
        self._mark_unsaved_changes()
    def _update_wet_dry_buttons(self):
        """Update wet/dry button appearances based on selection."""
        current_status = self.wet_dry_var.get()
        
        for status, btn in self.wet_dry_buttons.items():
            if status == current_status:
                # Selected button
                if status == "Wet":
                    btn.configure_color(self.gui_manager.theme_colors["accent_blue"])
                elif status == "Dry":
                    btn.configure_color(self.gui_manager.theme_colors["accent_green"])
                else:  # Unknown
                    btn.configure_color(self.gui_manager.theme_colors["accent_error"])
            else:
                # Not selected - grayed out
                btn.configure_color(self.gui_manager.theme_colors["secondary_bg"])

    def _toggle_edit_mode(self):
        """Toggle edit mode for data displays."""
        self.edit_mode.set(not self.edit_mode.get())
        
        # Update button appearance
        if self.edit_mode.get():
            self.edit_btn.set_text(DialogHelper.t("Done"))
            self.edit_btn.configure_color(self.gui_manager.theme_colors["accent_green"])
        else:
            self.edit_btn.set_text(DialogHelper.t("Edit"))
            self.edit_btn.configure_color(self.gui_manager.theme_colors["accent_blue"])
        
        self._update_edit_controls()

    def _update_edit_controls(self):
        """Show/hide edit controls based on edit mode."""
        edit_mode = self.edit_mode.get()
        
        # Show/hide add buttons
        if hasattr(self, 'add_cat_btn'):
            if edit_mode:
                self.add_cat_btn.pack(side=tk.RIGHT, padx=(10, 0))
            else:
                self.add_cat_btn.pack_forget()
                
        if hasattr(self, 'add_num_btn'):
            if edit_mode:
                self.add_num_btn.pack(side=tk.RIGHT, padx=(10, 0))
            else:
                self.add_num_btn.pack_forget()
        
        # Show/hide remove buttons and color pickers for categories
        for display in getattr(self, 'category_displays', []):
            if 'remove_btn' in display and 'control_frame' in display:
                if edit_mode:
                    if 'color_btn' in display:  # ADD color button for categories
                        display['color_btn'].pack(side=tk.LEFT, padx=(2, 0))
                    display['remove_btn'].pack(side=tk.LEFT, padx=(2, 0))
                else:
                    display['remove_btn'].pack_forget()
                    if 'color_btn' in display:
                        display['color_btn'].pack_forget()
                    
        # Show/hide remove buttons and color pickers for numeric
        for display in getattr(self, 'numeric_displays', []):
            if 'remove_btn' in display and 'control_frame' in display:
                if edit_mode:
                    display['color_btn'].pack(side=tk.LEFT, padx=(2, 0))
                    display['remove_btn'].pack(side=tk.LEFT, padx=(2, 0))
                else:
                    display['remove_btn'].pack_forget()
                    display['color_btn'].pack_forget()



    def _get_color_map_for_column(self, column_name):
        """
        Get appropriate color map for a column using fuzzy matching.
        
        Args:
            column_name: The column name to match
            
        Returns:
            ColorMap or None
        """
        if not column_name:
            return None
        
        # Normalize the column name: lowercase, remove special chars, spaces
        normalized = column_name.lower()
        normalized = normalized.replace('_', '').replace(' ', '').replace('%', '').replace('-', '')
        
        # Define patterns to match (element/compound patterns with their preset names)
        patterns = {
            # Iron patterns
            'fe': 'fe_grade',
            'iron': 'fe_grade',
            
            # Silica patterns
            'sio2': 'sio2_grade',
            'silica': 'sio2_grade',
            'si': 'sio2_grade',
            
            # Alumina patterns
            'al2o3': 'al2o3_grade',
            'alumina': 'al2o3_grade',
            'al': 'al2o3_grade',
            
            # Phosphorus patterns
            'p': 'p_grade',
            'phos': 'p_grade',
            'phosphorus': 'p_grade',
            
            # Lithology patterns
            'lith': 'lithology',
            'lithology': 'lithology',
            'rock': 'lithology',
            'rocktype': 'lithology',
            
            # BIFf patterns
            'bif': 'biff_categories',
            'biff': 'biff_categories',
            'facies': 'biff_categories',
        }
        
        # Check each pattern
        for pattern, preset_name in patterns.items():
            if pattern in normalized:
                # Additional checks for specific elements to avoid false matches
                if pattern in ['p', 'al', 'si'] and len(normalized) > len(pattern) + 3:
                    # Skip if it's part of a longer word (e.g., 'slip' containing 'p')
                    continue
                    
                preset = self.color_map_manager.get_preset(preset_name)
                if preset:
                    return preset
        
        # If no match found, try exact preset name match
        return self.color_map_manager.get_preset(normalized)

    def _update_numeric_display_column(self, index):
        """Update numeric display when column selection changes."""
        if index < len(self.numeric_displays):
            display = self.numeric_displays[index]
            col = display['column_var'].get()
            display['column'] = col if col != 'None' else None
            
            # Update color map using fuzzy matching
            if col and col != 'None':
                # Check user config first
                color_map_key = f"color_maps.{col.lower()}"
                if color_map_key in self.config:
                    display['color_map'] = self._load_color_map_from_config(col)
                else:
                    # Use fuzzy matching
                    display['color_map'] = self._get_color_map_for_column(col)
            else:
                display['color_map'] = None
            
            # Update the display
            self._update_data_displays()

    def _add_numeric_display(self, preset_column=None, skip_update=False):
        """Add a new numeric display."""
        index = len(self.numeric_displays)
        
        # Create frame for this numeric display
        num_frame = ttk.Frame(self.numeric_container)
        num_frame.pack(side=tk.LEFT, padx=2, fill=tk.Y)
        
        # Top row with dropdown and buttons
        control_frame = ttk.Frame(num_frame)
        control_frame.pack(fill=tk.X)
        
        # Get numeric columns
        numeric_cols = [col for col in self.register_data.columns 
                    if pd.api.types.is_numeric_dtype(self.register_data[col])
                    and col not in ['From', 'To']]
        
        # Column selector with themed dropdown
        col_var = tk.StringVar()
        
        # Dropdown frame
        dropdown_frame = tk.Frame(
            control_frame,
            bg=self.gui_manager.theme_colors["field_bg"],
            highlightbackground=self.gui_manager.theme_colors["field_border"],
            highlightthickness=1,
            bd=0
        )
        dropdown_frame.pack(side=tk.LEFT)
        
        col_dropdown = tk.OptionMenu(
            dropdown_frame,
            col_var,
            preset_column or 'None',
            *(['None'] + sorted(numeric_cols))
        )
        self.gui_manager.style_dropdown(col_dropdown, width=10)
        col_dropdown.config(font=('Arial', 10))
        col_dropdown.pack()
        
        # Set preset column if provided
        if preset_column and preset_column in numeric_cols:
            col_var.set(preset_column)
        else:
            col_var.set('None')
        
        # Color picker button
        color_btn = tk.Button(
            control_frame,
            text="🎨",
            command=lambda: self._edit_color_map(index, 'numeric'),
            bg=self.gui_manager.theme_colors["secondary_bg"],
            fg=self.gui_manager.theme_colors["text"],
            font=("Arial", 10),
            bd=0,
            padx=4,
            pady=1,
            cursor="hand2",
            relief=tk.FLAT
        )
        
        # Remove button
        remove_btn = tk.Button(
            control_frame,
            text="✕",
            command=lambda: self._remove_numeric_display(index),
            bg=self.gui_manager.theme_colors["accent_red"],
            fg="white",
            font=("Arial", 10, "bold"),
            bd=0,
            padx=4,
            pady=1,
            cursor="hand2",
            relief=tk.FLAT
        )
        
        # Value display
        value_label = tk.Label(
            num_frame,
            text="--",
            bg=self.gui_manager.theme_colors["field_bg"],
            fg=self.gui_manager.theme_colors["text"],
            font=('Arial', 14, 'bold'),
            width=8,
            relief=tk.SUNKEN,
            bd=1,
            pady=2
        )
        value_label.pack(pady=(2, 0))
        
        # ===================================================
        # Fuzzy match column names to preset names
        # ===================================================
        color_map = None
        if preset_column:
            # Try to load saved color map from user config first
            color_map_key = f"color_maps.{preset_column.lower()}"
            if color_map_key in self.config:
                color_map = self._load_color_map_from_config(preset_column)
            else:
                # Try fuzzy matching to find appropriate preset
                color_map = self._get_color_map_for_column(preset_column)
        
        # Store reference
        display_info = {
            'type': 'numeric',
            'frame': num_frame,
            'control_frame': control_frame,
            'column': preset_column,
            'column_var': col_var,
            'value_label': value_label,
            'remove_btn': remove_btn,
            'color_btn': color_btn,
            'color_map': color_map,
            'index': index
        }
        
        self.numeric_displays.append(display_info)
        self.data_displays.append(display_info)
        
        # Update display when column changes
        col_var.trace_add('write', lambda *args: self._update_numeric_display_column(index))
        
        # Update edit controls visibility
        self._update_edit_controls()
        
        # Only update if not skipping
        if not skip_update:
            self._update_data_displays()

    def _edit_color_map(self, index, display_type):
        """Open color map editor for a display."""
        if display_type == 'numeric' and index < len(self.numeric_displays):
            display = self.numeric_displays[index]
            column = display['column_var'].get()
        elif display_type == 'category' and index < len(self.category_displays):
            display = self.category_displays[index]
            column = display['column_var'].get()
        else:
            return
            
        if column and column != 'None':
            try:
                # Import the color map config dialog from gui
                from gui.color_map_config_dialog import ColorMapConfigDialog
                
                current_map = display.get('color_map')
                
                # Get the data for this column
                column_data = None
                if column in self.register_data.columns:
                    column_data = self.register_data[column].dropna().tolist()
                
                # Debug logging
                self.logger.info(f"Opening color map dialog for column: {column}")
                self.logger.info(f"Data values count: {len(column_data) if column_data else 0}")
                
                # Create and show the dialog
                dialog = ColorMapConfigDialog(
                    self.dialog,                    # parent
                    self.gui_manager,              # gui_manager
                    self.color_map_manager,        # color_map_manager
                    column_data,                   # data_values
                    current_map                    # current_map
                )
                
                # Show dialog and get result
                new_map = dialog.show()
                
                if new_map:
                    display['color_map'] = new_map
                    # Save to config
                    self._save_color_map_to_config(column, new_map)
                    # Update display
                    self._update_data_displays()
                    self.logger.info(f"Color map updated for column: {column}")
                    
            except Exception as e:
                self.logger.error(f"Error opening color map editor: {e}")
                self.logger.error(traceback.format_exc())
                DialogHelper.show_message(
                    self.dialog,
                    DialogHelper.t("Error"),
                    DialogHelper.t("Failed to open color map editor:") + f"\n{str(e)}",
                    message_type="error"
                )

    def _save_color_map_to_config(self, column, color_map):
        """Save color map to config."""
        # Convert color map to serializable format based on type
        map_data = {
            'type': color_map.type.value,
            'default_color': color_map.default_color if hasattr(color_map, 'default_color') else (200, 200, 200),
            'null_color': color_map.null_color if hasattr(color_map, 'null_color') else (255, 255, 255)
        }
        
        # Add type-specific data
        if color_map.type == ColorMapType.CATEGORICAL:
            # For categorical, save the categories dictionary
            if hasattr(color_map, 'categories'):
                map_data['categories'] = {
                    cat: color for cat, color in color_map.categories.items()
                }
        elif color_map.type == ColorMapType.NUMERIC:
            # For numeric, save the ranges
            if hasattr(color_map, 'ranges'):
                map_data['ranges'] = [
                    {
                        'min': r.min,
                        'max': r.max,
                        'color': r.color,
                        'label': r.label if hasattr(r, 'label') else ''
                    }
                    for r in color_map.ranges
                ]
        elif color_map.type == ColorMapType.GRADIENT:
            # For gradient, we need to extract colors from the gradient
            # This depends on how gradient is stored in your ColorMap
            if hasattr(color_map, 'gradient_colors'):
                map_data['gradient_colors'] = color_map.gradient_colors
            elif hasattr(color_map, 'colors'):
                map_data['colors'] = [(c[0], c[1], c[2]) for c in color_map.colors]
        
        # Save to config
        config_key = f"color_maps.{column.lower()}"
        self.config[config_key] = map_data
        
        # Save config file
        if hasattr(self, 'config_manager'):
            self.config_manager.save_config()

    def _load_color_map_from_config(self, column):
        """Load color map from config."""
        config_key = f"color_maps.{column.lower()}"
        if config_key in self.config:
            map_data = self.config[config_key]
            
            # Determine the type
            map_type = ColorMapType(map_data.get('type', 'categorical'))
            
            # Create color map based on type
            if hasattr(self.color_map_manager, 'create_from_config'):
                return self.color_map_manager.create_from_config(map_data)
            else:
                # Manual recreation if create_from_config doesn't exist
                from processing.LoggingReviewStep.color_map_manager import ColorMap
                
                color_map = ColorMap(
                    name=f"{column}_custom",
                    map_type=map_type
                )
                
                # Set default colors
                if 'default_color' in map_data:
                    color_map.default_color = tuple(map_data['default_color'])
                if 'null_color' in map_data:
                    color_map.null_color = tuple(map_data['null_color'])
                
                # Load type-specific data
                if map_type == ColorMapType.CATEGORICAL and 'categories' in map_data:
                    for cat, color in map_data['categories'].items():
                        color_map.add_category(cat, tuple(color))
                elif map_type == ColorMapType.NUMERIC and 'ranges' in map_data:
                    from processing.LoggingReviewStep.color_map_manager import ColorRange
                    for r in map_data['ranges']:
                        color_range = ColorRange(
                            r['min'],
                            r['max'],
                            tuple(r['color']),
                            r.get('label', '')
                        )
                        color_map.add_range(color_range)
                elif map_type == ColorMapType.GRADIENT:
                    if 'gradient_colors' in map_data:
                        color_map.gradient_colors = map_data['gradient_colors']
                    elif 'colors' in map_data:
                        color_map.colors = [tuple(c) for c in map_data['colors']]
                
                return color_map
        
        return None

    def _add_category_display(self):
        """Add a new category display."""
        index = len(self.category_displays)
        
        # Create frame for this category display
        cat_frame = ttk.Frame(self.category_container)
        cat_frame.pack(side=tk.LEFT, padx=2, fill=tk.Y)
        
        # Top row with dropdown and buttons
        control_frame = ttk.Frame(cat_frame)
        control_frame.pack(fill=tk.X)
        
        # Column selector with themed dropdown
        col_var = tk.StringVar()
        
        # Get categorical columns
        exclude_cols = ['HoleID', 'From', 'To', 'ImagePath', 'Comments', 'ReviewedBy', 'WetDryStatus']
        exclude_cols.extend(self.toggle_vars.keys() if hasattr(self, 'toggle_vars') else [])
        
        categorical_cols = [col for col in self.register_data.columns 
                        if col not in exclude_cols
                        and (pd.api.types.is_object_dtype(self.register_data[col]) 
                                or pd.api.types.is_categorical_dtype(self.register_data[col]))]
        
        # Dropdown frame
        dropdown_frame = tk.Frame(
            control_frame,
            bg=self.gui_manager.theme_colors["field_bg"],
            highlightbackground=self.gui_manager.theme_colors["field_border"],
            highlightthickness=1,
            bd=0
        )
        dropdown_frame.pack(side=tk.LEFT)
        
        col_dropdown = tk.OptionMenu(
            dropdown_frame,
            col_var,
            'None',
            *(['None'] + sorted(categorical_cols))
        )
        self.gui_manager.style_dropdown(col_dropdown, width=12)
        # Make font bigger and bold
        col_dropdown.config(font=('Arial', 12, 'bold'))
        col_dropdown.pack()
        
        # Color picker button - ADD THIS
        color_btn = tk.Button(
            control_frame,
            text="🎨", 
            command=lambda: self._edit_color_map(index, 'category'),
            bg=self.gui_manager.theme_colors["secondary_bg"],
            fg=self.gui_manager.theme_colors["text"],
            font=("Arial", 12),
            bd=0,
            padx=4,
            pady=1,
            cursor="hand2",
            relief=tk.FLAT
        )
        # Don't pack yet - will be shown/hidden based on edit mode
        
        # Remove button
        remove_btn = tk.Button(
            control_frame,
            text="✕",
            command=lambda: self._remove_category_display(index),
            bg=self.gui_manager.theme_colors["accent_red"],
            fg="white",
            font=("Arial", 10, "bold"),
            bd=0,
            padx=4,
            pady=1,
            cursor="hand2",
            relief=tk.FLAT
        )
        
        # Value display with better sizing
        value_label = tk.Label(
            cat_frame,
            text="[Select]",
            bg=self.gui_manager.theme_colors["field_bg"],
            fg=self.gui_manager.theme_colors["text"],
            font=('Arial', 13),  # Slightly bigger font
            relief=tk.SUNKEN,
            bd=1,
            padx=8,  # More padding
            pady=6,  # More padding
            width=14  # Fixed width
        )
        value_label.pack(pady=(2, 0), fill=tk.X)
        
        # Store reference
        display_info = {
            'type': 'category',
            'frame': cat_frame,
            'control_frame': control_frame,
            'column_var': col_var,
            'value_label': value_label,
            'remove_btn': remove_btn,
            'color_btn': color_btn,  # ADD THIS
            'color_map': None,
            'index': index
        }
        
        self.category_displays.append(display_info)
        self.data_displays.append(display_info)
        
        # Update display when column changes
        col_var.trace_add('write', lambda *args: self._update_category_color_map(index))
        
        # Update edit controls visibility
        self._update_edit_controls()

    def _update_category_color_map(self, index):
        """Update category display color map when column changes."""
        if index < len(self.category_displays):
            display = self.category_displays[index]
            col = display['column_var'].get()
            
            if col and col != 'None':
                # Check user config first
                color_map_key = f"color_maps.{col.lower()}"
                if color_map_key in self.config:
                    display['color_map'] = self._load_color_map_from_config(col)
                else:
                    # Use fuzzy matching for categories too
                    display['color_map'] = self._get_color_map_for_column(col)
            else:
                display['color_map'] = None
            
            # Update the display
            self._update_data_displays()

    def _remove_category_display(self, index):
        """Remove a category display."""
        if index < len(self.category_displays):
            display = self.category_displays[index]
            display['frame'].destroy()
            self.category_displays.pop(index)
            
            # Remove from data_displays
            self.data_displays = [d for d in self.data_displays if d.get('frame') != display['frame']]
            
            # Re-index remaining displays
            for i, disp in enumerate(self.category_displays):
                disp['index'] = i

    def _create_default_numeric_displays(self):
        """Create default numeric displays for common columns."""
        # Default columns to look for (case-insensitive)
        default_cols = ['fe_pct', 'sio2_pct', 'al2o3_pct', 'p_pct']
        
        # Get numeric columns
        numeric_cols = [col for col in self.register_data.columns 
                    if pd.api.types.is_numeric_dtype(self.register_data[col])
                    and col not in ['From', 'To']]
        
        # Find matching columns (case-insensitive)
        added_count = 0
        for default in default_cols:
            for col in numeric_cols:
                if col.lower() == default.lower():
                    # ===================================================
                    # Skip update during initial creation
                    # ===================================================
                    self._add_numeric_display(preset_column=col, skip_update=True)
                    added_count += 1
                    break
        
        # If we didn't find all 4 defaults, add empty ones
        while added_count < 4:
            # ===================================================
            # Skip update during initial creation
            # ===================================================
            self._add_numeric_display(skip_update=True)
            added_count += 1

    def _add_numeric_display(self, preset_column=None, skip_update=False):
        """Add a new numeric display."""
        index = len(self.numeric_displays)
        
        # Create frame for this numeric display
        num_frame = ttk.Frame(self.numeric_container)
        num_frame.pack(side=tk.LEFT, padx=2, fill=tk.Y)
        
        # Top row with dropdown and buttons
        control_frame = ttk.Frame(num_frame)
        control_frame.pack(fill=tk.X)
        
        # Get numeric columns
        numeric_cols = [col for col in self.register_data.columns 
                    if pd.api.types.is_numeric_dtype(self.register_data[col])
                    and col not in ['From', 'To']]
        
        # Column selector with themed dropdown
        col_var = tk.StringVar()
        
        # Dropdown frame
        dropdown_frame = tk.Frame(
            control_frame,
            bg=self.gui_manager.theme_colors["field_bg"],
            highlightbackground=self.gui_manager.theme_colors["field_border"],
            highlightthickness=1,
            bd=0
        )
        dropdown_frame.pack(side=tk.LEFT)
        
        col_dropdown = tk.OptionMenu(
            dropdown_frame,
            col_var,
            preset_column or 'None',
            *(['None'] + sorted(numeric_cols))
        )
        self.gui_manager.style_dropdown(col_dropdown, width=10)
        # Make font bigger and bold
        col_dropdown.config(font=('Arial', 12, 'bold'))
        col_dropdown.pack()
        
        # Set preset column if provided
        if preset_column and preset_column in numeric_cols:
            col_var.set(preset_column)
        else:
            col_var.set('None')
        
        # Color picker button
        color_btn = tk.Button(
            control_frame,
            text="🎨",  
            command=lambda: self._edit_color_map(index, 'numeric'),
            bg=self.gui_manager.theme_colors["secondary_bg"],
            fg=self.gui_manager.theme_colors["text"],
            font=("Arial", 12),
            bd=0,
            padx=4,
            pady=1,
            cursor="hand2",
            relief=tk.FLAT
        )
        
        # Remove button
        remove_btn = tk.Button(
            control_frame,
            text="✕",
            command=lambda: self._remove_numeric_display(index),
            bg=self.gui_manager.theme_colors["accent_red"],
            fg="white",
            font=("Arial", 10, "bold"),
            bd=0,
            padx=4,
            pady=1,
            cursor="hand2",
            relief=tk.FLAT
        )
        
        # Value display with better sizing
        value_label = tk.Label(
            num_frame,
            text="--",
            bg=self.gui_manager.theme_colors["field_bg"],
            fg=self.gui_manager.theme_colors["text"],
            font=('Arial', 16, 'bold'),  # Bigger, bold font
            width=10,  # Wider
            relief=tk.SUNKEN,
            bd=1,
            pady=4  # Taller
        )
        value_label.pack(pady=(2, 0))
        
        # Get or create color map
        color_map = None
        if preset_column:
            # Try to load saved color map from config
            color_map_key = f"color_maps.{preset_column.lower()}"
            if color_map_key in self.config:
                # Load from config
                color_map = self._load_color_map_from_config(preset_column)
            else:
                # Use default from color map manager
                color_map = self.color_map_manager.get_preset(preset_column.lower().replace(' ', '_'))
        
        # Store reference
        display_info = {
            'type': 'numeric',
            'frame': num_frame,
            'control_frame': control_frame,
            'column': preset_column,
            'column_var': col_var,
            'value_label': value_label,
            'remove_btn': remove_btn,
            'color_btn': color_btn,
            'color_map': color_map,
            'index': index
        }
        
        self.numeric_displays.append(display_info)
        self.data_displays.append(display_info)
        
        # Update display when column changes
        col_var.trace_add('write', lambda *args: self._update_numeric_display_column(index))
        
        # Update edit controls visibility
        self._update_edit_controls()
        
        # Only update if not skipping
        if not skip_update:
            self._update_data_displays()


    def _remove_numeric_display(self, index):
        """Remove a numeric display."""
        if index < len(self.numeric_displays):
            display = self.numeric_displays[index]
            display['frame'].destroy()
            self.numeric_displays.pop(index)
            
            # Remove from data_displays
            self.data_displays = [d for d in self.data_displays if d.get('frame') != display['frame']]
            
            # Re-index remaining displays
            for i, disp in enumerate(self.numeric_displays):
                disp['index'] = i

    def _on_toggle_interaction(self, toggle_name: str):
            """Handle toggle interaction."""
            # Mark that user has interacted with this compartment
            if not self.filtered_data.empty and self.current_index < len(self.filtered_data):
                current_row = self.filtered_data.iloc[self.current_index]
                key = (current_row['HoleID'], int(current_row['From']), int(current_row['To']))
                
                if key not in self.user_interacted:
                    self.user_interacted[key] = {
                        'toggles_touched': set(),
                        'comments_touched': False,
                        'first_interaction_time': datetime.now()
                    }
                
                self.user_interacted[key]['toggles_touched'].add(toggle_name)
                
            # Mark unsaved changes
            self._mark_unsaved_changes()
        
    def _on_comment_interaction(self, event=None):
        """Handle comment text interaction."""
        # Mark that user has interacted with this compartment
        if not self.filtered_data.empty and self.current_index < len(self.filtered_data):
            current_row = self.filtered_data.iloc[self.current_index]
            key = (current_row['HoleID'], int(current_row['From']), int(current_row['To']))
            
            if key not in self.user_interacted:
                self.user_interacted[key] = {
                    'toggles_touched': set(),
                    'comments_touched': False,
                    'first_interaction_time': datetime.now()
                }
            
            self.user_interacted[key]['comments_touched'] = True
            
        # Mark unsaved changes
        self._mark_unsaved_changes()


    def _on_wet_dry_change(self):
        """Handle wet/dry status change."""
        new_status = self.wet_dry_var.get()
        
        if new_status == "Unknown":
            return
            
        # Mark as changed
        self.changes_made = True
        
        # Update register data
        if not self.filtered_data.empty and self.current_index < len(self.filtered_data):
            current_row_idx = self.filtered_data.index[self.current_index]
            self.register_data.at[current_row_idx, 'WetDryStatus'] = new_status
            
            # Note: File renaming would require:
            # 1. Closing the current image display
            # 2. Renaming the file
            # 3. Updating the path in register
            # 4. Reloading the image
            # This is complex and might fail if file is locked
            
            # Instead, we'll just track the status change for now
            self.logger.info(f"Wet/Dry status changed to: {new_status}")

    def _create_category_display(self, parent, index):
        """Create a single category display panel."""
        frame = ttk.Frame(parent, style='Content.TFrame')
        
        # Column selector
        col_var = tk.StringVar()
        
        # Get categorical columns from merged data
        exclude_cols = ['HoleID', 'From', 'To', 'ImagePath', 'Comments', 'ReviewedBy', 'WetDryStatus']
        exclude_cols.extend(self.toggle_vars.keys())  # Exclude toggle columns
        
        categorical_cols = [col for col in self.register_data.columns 
                           if col not in exclude_cols
                           and (pd.api.types.is_object_dtype(self.register_data[col]) 
                                or pd.api.types.is_categorical_dtype(self.register_data[col]))]
        
        col_combo = self.gui_manager.create_combobox(
            frame,
            col_var,
            values=['None'] + sorted(categorical_cols),
            width=15
        )
        col_combo.pack(pady=(0, 5))
        col_combo.set('None')
        
        # Value display
        value_label = tk.Label(
            frame,
            text=DialogHelper.t("[Select Column]"),
            bg=self.gui_manager.theme_colors["field_bg"],
            fg=self.gui_manager.theme_colors["text"],
            font=('Arial', 14, 'bold'),
            width=15,
            height=2,
            relief=tk.SUNKEN,
            bd=2
        )
        value_label.pack(fill=tk.BOTH, expand=True)
        
        # Store references
        self.data_displays.append({
            'type': 'category',
            'column_var': col_var,
            'value_label': value_label,
            'color_map': None
        })
        
        # Update display when column changes
        col_var.trace_add('write', lambda *args: self._update_data_displays())
        
        return frame
        
    def _create_numeric_displays(self, parent):
        """Create numeric data displays."""
        # Get numeric columns
        numeric_cols = [col for col in self.register_data.columns 
                       if pd.api.types.is_numeric_dtype(self.register_data[col])
                       and col not in ['From', 'To']]
        
        # Create horizontal displays for key numeric values
        display_frame = ttk.Frame(parent, style='Content.TFrame')
        display_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create 4 numeric displays
        for i in range(4):
            if i < len(numeric_cols):
                col_name = numeric_cols[i]
                
                # Create display
                num_frame = ttk.Frame(display_frame, style='Content.TFrame')
                num_frame.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
                
                # Label
                label = ttk.Label(
                    num_frame,
                    text=col_name,
                    style='Content.TLabel'
                )
                label.pack()
                
                # Value display
                value_label = tk.Label(
                    num_frame,
                    text="--",
                    bg=self.gui_manager.theme_colors["field_bg"],
                    fg=self.gui_manager.theme_colors["text"],
                    font=('Arial', 16, 'bold'),
                    width=8,
                    height=2,
                    relief=tk.SUNKEN,
                    bd=2
                )
                value_label.pack()
                
                # Store reference
                self.data_displays.append({
                    'type': 'numeric',
                    'column': col_name,
                    'value_label': value_label,
                    'color_map': self.color_map_manager.get_preset(col_name.lower().replace(' ', '_'))
                })
                
    def _create_bottom_controls(self, parent):
        """Create bottom navigation controls."""
        controls_frame = ttk.Frame(parent, style='Content.TFrame')
        controls_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Navigation info
        self.nav_label = ttk.Label(
            controls_frame,
            text="",
            style='Content.TLabel'
        )
        self.nav_label.pack(side=tk.LEFT)
        
        # Unsaved changes indicator
        self.unsaved_label = ttk.Label(
            controls_frame,
            text="",
            style='Content.TLabel',
            font=('Arial', 10, 'italic'),
            foreground='red'
        )
        self.unsaved_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Spacer
        ttk.Frame(controls_frame).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Navigation buttons
        self.prev_button = ModernButton(
            controls_frame,
            text=DialogHelper.t("Previous"),
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._previous_image,
            theme_colors=self.gui_manager.theme_colors
        )
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        self.next_button = ModernButton(
            controls_frame,
            text=DialogHelper.t("Next"),
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._next_image,
            theme_colors=self.gui_manager.theme_colors
        )
        self.next_button.pack(side=tk.LEFT, padx=5)
        
        # Save button
        save_button = ModernButton(
            controls_frame,
            text=DialogHelper.t("Save Changes"),
            color=self.gui_manager.theme_colors["accent_green"],
            command=self._save_changes,
            theme_colors=self.gui_manager.theme_colors
        )
        save_button.pack(side=tk.LEFT, padx=20)
        
        # Quit button
        quit_button = ModernButton(
            controls_frame,
            text=DialogHelper.t("Quit"),
            color=self.gui_manager.theme_colors["accent_red"],
            command=self._quit_dialog,
            theme_colors=self.gui_manager.theme_colors
        )
        quit_button.pack(side=tk.LEFT, padx=5)
        
    def _mark_unsaved_changes(self):
        """Mark that there are unsaved changes for the current image."""
        self.unsaved_changes = True
        self.unsaved_label.config(text=DialogHelper.t("(unsaved changes)"))
        
    def _clear_unsaved_indicator(self):
        """Clear the unsaved changes indicator."""
        self.unsaved_changes = False
        self.unsaved_label.config(text="")
        
    def _start_auto_save(self):
        """Start the auto-save timer."""
        if self.auto_save_timer:
            self.dialog.after_cancel(self.auto_save_timer)
        self.auto_save_timer = self.dialog.after(self.auto_save_interval, self._auto_save)
        
    def _auto_save(self):
        """Auto-save current data if there are unsaved changes."""
        if self.unsaved_changes:
            self._save_current_data()
            self.logger.info("Auto-saved current review data")
        
        # Restart timer
        self._start_auto_save()
        
    def _apply_filters(self):
        """Apply the selected filters to the data."""
        # Save current data before filtering
        if self.unsaved_changes:
            self._save_current_data()
        
        # Start with all data
        filtered = self.register_data.copy()
        
        # ===================================================
        # Apply dynamic filters
        # ===================================================
        for filter_row in self.filter_rows:
            config = filter_row.get_filter_config()
            column = config['column']
            operator = config['operator']
            value = config['value']
            value2 = config['value2']
            
            if not column or not operator or (not value and operator != 'between'):
                continue
                
            try:
                col_type = self.columns_info.get(column, 'text')
                
                if col_type == 'numeric':
                    # Numeric filters
                    if operator == '<':
                        filtered = filtered[filtered[column] < float(value)]
                    elif operator == '<=':
                        filtered = filtered[filtered[column] <= float(value)]
                    elif operator == '=':
                        filtered = filtered[filtered[column] == float(value)]
                    elif operator == '>=':
                        filtered = filtered[filtered[column] >= float(value)]
                    elif operator == '>':
                        filtered = filtered[filtered[column] > float(value)]
                    elif operator == '!=':
                        filtered = filtered[filtered[column] != float(value)]
                    elif operator == 'between' and value and value2:
                        filtered = filtered[(filtered[column] >= float(value)) & 
                                        (filtered[column] <= float(value2))]
                else:
                    # Text filters
                    if operator == 'is':
                        filtered = filtered[filtered[column] == value]
                    elif operator == 'is not':
                        filtered = filtered[filtered[column] != value]
                    elif operator == 'contains':
                        filtered = filtered[filtered[column].str.contains(value, case=False, na=False)]
                    elif operator == 'starts with':
                        filtered = filtered[filtered[column].str.startswith(value, na=False)]
                    elif operator == 'ends with':
                        filtered = filtered[filtered[column].str.endswith(value, na=False)]
                    elif operator == 'in':
                        values = [v.strip() for v in value.split(',')]
                        filtered = filtered[filtered[column].isin(values)]
                        
            except Exception as e:
                self.logger.error(f"Error applying filter: {e}")
                continue
        
        # Apply only with images filter
        if self.only_with_images_var.get():
            filtered = filtered[filtered['ImagePath'].notna()]
        
        # Hide reviewed by current user
        if self.hide_reviewed_var.get():
            current_user = os.environ.get('USERNAME', 'unknown')
            
            if hasattr(self, 'json_manager') and self.json_manager:
                # Filter out rows where current user has reviewed
                reviewed_indices = []
                for idx, row in filtered.iterrows():
                    review = self.json_manager.get_user_review(
                        row['HoleID'],
                        int(row['From']),
                        int(row['To']),
                        current_user
                    )
                    if review is not None:
                        reviewed_indices.append(idx)
                
                filtered = filtered.drop(reviewed_indices)
            else:
                filtered = filtered[filtered['ReviewedBy'] != current_user]
        
        # Update filtered data
        self.filtered_data = filtered
        
        # Update results counter in collapsible frame title
        total_count = len(self.register_data)
        filtered_count = len(self.filtered_data)
        self.filter_frame.set_text(f"Filters - Showing {filtered_count} of {total_count} intervals")
        
        # Reset to first image if current index is out of bounds
        if self.current_index >= len(self.filtered_data):
            self.current_index = 0
        
        # Load the current image - FIXED METHOD NAME
        self._load_current_image()


    def _load_current_image(self):
        """Load and display the current image."""
        if self.filtered_data.empty or self.current_index >= len(self.filtered_data):
            # ===================================================
            # MODIFIED: Clear canvas when no data
            # ===================================================
            self.image_canvas.delete("all")
            self.image_canvas.update_idletasks()
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            if canvas_width > 1 and canvas_height > 1:
                self.image_canvas.create_text(
                    canvas_width // 2,
                    canvas_height // 2,
                    text=DialogHelper.t("No image available"),
                    fill=self.gui_manager.theme_colors["text"],
                    font=('Arial', 14)
                )
            self.current_pil_image = None
            self._update_data_displays()
            return
            
        # Get current row
        current_row = self.filtered_data.iloc[self.current_index]
        
        # Update info labels
        self.image_info_label.config(
            text=f"{current_row['HoleID']} - {int(current_row['From'])}-{int(current_row['To'])}m"
        )
        self.interval_label.config(
            text=f"{current_row['HoleID']} - {int(current_row['From'])}-{int(current_row['To'])}m"
        )
        
        # Try to find the image if ImagePath is not set
        if pd.isna(current_row.get('ImagePath', None)):
            image_path = self._find_compartment_image(current_row['HoleID'], int(current_row['To']))
            if image_path:
                self.filtered_data.at[self.current_index, 'ImagePath'] = image_path
                current_row = self.filtered_data.iloc[self.current_index]
        
        # Load image if available
        if pd.isna(current_row.get('ImagePath', None)) or not os.path.exists(current_row['ImagePath']):
            # ===================================================
            # MODIFIED: Update canvas when no image
            # ===================================================
            self.image_canvas.delete("all")
            self.image_canvas.update_idletasks()
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            if canvas_width > 1 and canvas_height > 1:
                # Check if this is a missing file vs never had an image
                if pd.notna(current_row.get('ImagePath', None)):
                    # Had a path but file is missing
                    message = DialogHelper.t("Image file missing from OneDrive")
                    color = self.gui_manager.theme_colors["accent_error"]
                    
                    # Check Photo_Status for more info
                    if current_row.get('Photo_Status') == 'MISSING_FILE':
                        message += f"\n{DialogHelper.t('(Marked as missing during validation)')}"
                else:
                    # Never had an image
                    message = DialogHelper.t("No image available")
                    color = self.gui_manager.theme_colors["text"]
                    
                self.image_canvas.create_text(
                    canvas_width // 2,
                    canvas_height // 2,
                    text=message,
                    fill=color,
                    font=('Arial', 14),
                    width=canvas_width * 0.8,  # Allow text wrapping
                    justify='center'
                )
            self.current_pil_image = None
        else:
            try:
                # Load image with PIL
                self.current_pil_image = Image.open(current_row['ImagePath'])
                
                # ===================================================
                # MODIFIED: Use pan/zoom handler
                # ===================================================
                self.pan_zoom_handler.set_image(self.current_pil_image)
                
            except Exception as e:
                self.logger.error(f"Error loading image: {str(e)}")
                # ===================================================
                # MODIFIED: Update canvas for error
                # ===================================================
                self.image_canvas.delete("all")
                self.image_canvas.update_idletasks()
                canvas_width = self.image_canvas.winfo_width()
                canvas_height = self.image_canvas.winfo_height()
                if canvas_width > 1 and canvas_height > 1:
                    self.image_canvas.create_text(
                        canvas_width // 2,
                        canvas_height // 2,
                        text=DialogHelper.t("Error loading image"),
                        fill=self.gui_manager.theme_colors["accent_red"],
                        font=('Arial', 14)
                    )
                self.current_pil_image = None
            
        # Update data displays and review status
        self._update_data_displays()
        
        # Update navigation
        self._update_navigation()


    def _zoom_in(self):
        """Zoom in on the image."""
        if hasattr(self, 'pan_zoom_handler') and self.current_pil_image:
            # Simulate mouse wheel event at center
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            event = type('Event', (), {
                'x': canvas_width // 2,
                'y': canvas_height // 2,
                'delta': 120,
                'num': 4
            })()
            self.pan_zoom_handler._on_mouse_wheel(event)
            
    def _zoom_out(self):
        """Zoom out on the image."""
        if hasattr(self, 'pan_zoom_handler') and self.current_pil_image:
            # Simulate mouse wheel event at center
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            event = type('Event', (), {
                'x': canvas_width // 2,
                'y': canvas_height // 2,
                'delta': -120,
                'num': 5
            })()
            self.pan_zoom_handler._on_mouse_wheel(event)

    def _find_compartment_image(self, hole_id: str, compartment_depth: int) -> Optional[str]:
        """
        Find a compartment image in the approved folder structure.
        
        Args:
            hole_id: The hole ID
            compartment_depth: The compartment depth (To value)
            
        Returns:
            Path to image if found, None otherwise
        """
        approved_path = self.file_manager.get_shared_path('approved_compartments', create_if_missing=False)
        if not approved_path or not approved_path.exists():
            return None
        
        # Extract project code from first two letters of hole ID
        if len(hole_id) >= 2:
            project_code = hole_id[:2].upper()
        else:
            self.logger.warning(f"Invalid hole ID format: {hole_id}")
            return None
        
        # Build the path: approved → project code → hole ID
        project_folder = approved_path / project_code
        hole_path = project_folder / hole_id
        
        if not hole_path.exists():
            return None
        
        try:
            # Look for any file matching the pattern
            expected_prefix = f"{hole_id}_CC_{compartment_depth:03d}"
            
            for filename in os.listdir(str(hole_path)):
                if filename.startswith(expected_prefix):
                    # Check if it's an image file (case-insensitive)
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
                        return str(hole_path / filename)
                        
        except Exception as e:
            self.logger.error(f"Error finding compartment image: {str(e)}")
        
        return None

    def _convert_toggle_name_to_field(self, toggle_name: str) -> str:
        """
        Convert UI toggle name to database field name.
        
        Examples:
            "Bad Image" -> "Bad_Image"
            "+ QZ" -> "Contains_Qz"
            "+ CHH/M" -> "Contains_CHHM"
        """
        # Handle special cases
        if toggle_name.startswith('+'):
            # Convert "+ XX" to "Contains_XX"
            suffix = toggle_name[1:].strip().replace('/', '').replace(' ', '')
            return f"Contains_{suffix}"
        else:
            # Replace spaces and special characters with underscores
            return toggle_name.replace(' ', '_').replace('/', '_')

    def _update_data_displays(self):
        """Update all data display panels with current row data."""
        if self.filtered_data is None or self.filtered_data.empty or self.current_index >= len(self.filtered_data):
            # Clear all displays if no data
            if hasattr(self, 'category_displays'):
                for display in self.category_displays:
                    display['value_label'].config(text="[No Data]", bg=self.gui_manager.theme_colors["field_bg"])
            
            if hasattr(self, 'numeric_displays'):
                for display in self.numeric_displays:
                    display['value_label'].config(text="--", bg=self.gui_manager.theme_colors["field_bg"])
            return
            
        current_row = self.filtered_data.iloc[self.current_index]
        
        # Clear unsaved indicator when loading new data
        self._clear_unsaved_indicator()
        
        # Check review status and load user's previous review if it exists
        username = os.getenv("USERNAME", "Unknown")
        if self.json_manager:
            # Get the current user's review for this compartment
            previous_review = self.json_manager.get_user_review(
                hole_id=current_row['HoleID'],
                depth_from=int(current_row['From']),
                depth_to=int(current_row['To'])
            )
            
            if previous_review:
                # Update review status indicator
                review_number = previous_review.get('Review_Number', 1)
                review_date = previous_review.get('Review_Date', '')
                if review_date:
                    try:
                        date_obj = datetime.fromisoformat(review_date)
                        date_str = date_obj.strftime('%Y-%m-%d %H:%M')
                    except:
                        date_str = review_date[:16]  # Fallback to first 16 chars
                else:
                    date_str = "unknown date"
                    
                self.review_status_label.config(
                    text=DialogHelper.t(f"Modify Entry by {username} (Edit #{review_number}, Last: {date_str})"),
                    foreground=self.gui_manager.theme_colors["accent_blue"]
                )
                
                # Load previous comments
                if previous_review.get('Comments'):
                    self.comment_text.delete(1.0, tk.END)
                    self.comment_text.insert(1.0, previous_review['Comments'])
                else:
                    self.comment_text.delete(1.0, tk.END)
                
                # Load toggle states - handle dynamic field mapping
                for toggle_name, toggle in self.toggle_vars.items():
                    # Convert UI name to DB field name
                    db_field_name = self._convert_toggle_name_to_field(toggle_name)
                    
                    # Reset interaction state when loading new compartment
                    toggle.reset_interaction()
                    
                    # Check if field exists in the review
                    if db_field_name in previous_review:
                        toggle.set(bool(previous_review[db_field_name]), mark_as_interacted=False)
                    else:
                        toggle.set(None, mark_as_interacted=False)
                        
                # Store current review info
                self.current_review_info = {
                    'exists': True,
                    'review_number': review_number,
                    'previous_data': previous_review
                }
            else:
                # No previous review - clear everything
                self.review_status_label.config(
                    text=DialogHelper.t(f"New Entry by {username}"),
                    foreground=self.gui_manager.theme_colors["accent_green"]
                )
                
                self.comment_text.delete(1.0, tk.END)
                for var in self.toggle_vars.values():
                    var.set(False)
                    
                # Store current review info
                self.current_review_info = {
                    'exists': False,
                    'review_number': 0
                }
        
            # Update category displays
            for display in self.category_displays:
                col = display['column_var'].get()
                if col != 'None' and col in current_row.index:
                    value = current_row[col]
                    value_text = str(value) if pd.notna(value) else "[None]"
                    
                    # Auto-size the label
                    display['value_label'].config(text=value_text)
                    
                    # Apply color if color map exists
                    if display.get('color_map'):
                        color = display['color_map'].get_color(value)
                        hex_color = '#{:02x}{:02x}{:02x}'.format(color[2], color[1], color[0])
                        display['value_label'].config(bg=hex_color)
                        # Adjust text color for contrast
                        brightness = (color[0] * 299 + color[1] * 587 + color[2] * 114) / 1000
                        text_color = 'black' if brightness > 128 else 'white'
                        display['value_label'].config(fg=text_color)
                else:
                    display['value_label'].config(
                        text="[Select]", 
                        bg=self.gui_manager.theme_colors["field_bg"],
                        fg=self.gui_manager.theme_colors["text"]
                    )
            
            # Update numeric displays
            for display in self.numeric_displays:
                col = display.get('column') or display['column_var'].get()
                if col and col != 'None' and col in current_row.index:
                    value = current_row[col]
                    if pd.notna(value):
                        display['value_label'].config(text=f"{value:.2f}")
                        
                        # Apply color if color map exists
                        if display.get('color_map'):
                            color = display['color_map'].get_color(value)
                            hex_color = '#{:02x}{:02x}{:02x}'.format(color[2], color[1], color[0])
                            display['value_label'].config(bg=hex_color)
                            # Adjust text color for contrast
                            brightness = (color[0] * 299 + color[1] * 587 + color[2] * 114) / 1000
                            text_color = 'black' if brightness > 128 else 'white'
                            display['value_label'].config(fg=text_color)
                        else:
                            display['value_label'].config(
                                bg=self.gui_manager.theme_colors["field_bg"],
                                fg=self.gui_manager.theme_colors["text"]
                            )
                    else:
                        display['value_label'].config(
                            text="--", 
                            bg=self.gui_manager.theme_colors["field_bg"],
                            fg=self.gui_manager.theme_colors["text"]
                        )
                else:
                    display['value_label'].config(
                        text="--", 
                        bg=self.gui_manager.theme_colors["field_bg"],
                        fg=self.gui_manager.theme_colors["text"]
                    )
        # ===================================================
        # NEW CODE: Update wet/dry status
        # ===================================================
        if hasattr(self, 'wet_dry_var'):
            # ===================================================
            # FIXED: Read from Photo_Status, not WetDryStatus
            # ===================================================
            photo_status = current_row.get('Photo_Status', '')
            
            if 'OK_Wet' in photo_status or '_Wet' in photo_status:
                self.wet_dry_var.set("Wet")
            elif 'OK_Dry' in photo_status or '_Dry' in photo_status:
                self.wet_dry_var.set("Dry")
            else:
                # Try to infer from filename if available
                if 'ImagePath' in current_row.index and pd.notna(current_row['ImagePath']):
                    filename = os.path.basename(current_row['ImagePath'])
                    if '_Wet.' in filename:
                        self.wet_dry_var.set("Wet")
                    elif '_Dry.' in filename:
                        self.wet_dry_var.set("Dry")
                    else:
                        self.wet_dry_var.set("Unknown")
                else:
                    self.wet_dry_var.set("Unknown")

        # Update wet/dry button states
        if hasattr(self, 'wet_dry_buttons'):
            self._update_wet_dry_buttons()

                        
    def _update_navigation(self):
        """Update navigation controls."""
        total = len(self.filtered_data)
        current = self.current_index + 1 if total > 0 else 0
        
        self.nav_label.config(text=f"{current} / {total}")
        
        # Enable/disable buttons
        self.prev_button.set_state('normal' if self.current_index > 0 else 'disabled')
        self.next_button.set_state('normal' if self.current_index < total - 1 else 'disabled')
        self.same_as_before_button.set_state('normal' if self.previous_toggle_states else 'disabled')
        
    def _previous_image(self):
        """Navigate to previous image."""
        if self.current_index > 0:
            self._save_current_data()
            self.current_index -= 1
            self._load_current_image()
            self._update_navigation()
            
    def _next_image(self):
        """Navigate to next image."""
        if self.current_index < len(self.filtered_data) - 1:
            self._save_current_data()
            self.current_index += 1
            self._load_current_image()
            self._update_navigation()
    
    def _copy_previous_toggles(self):
        """Copy toggle states from the previously processed image."""
        if not self.previous_toggle_states:
            return
            
        # Apply previous states to current toggles
        for name, state in self.previous_toggle_states.items():
            if name in self.toggle_vars:
                # ===================================================
                # MODIFIED: Mark as interacted when copying
                # ===================================================
                self.toggle_vars[name].set(state, mark_as_interacted=True)
                
        # Mark interaction for this compartment
        if not self.filtered_data.empty and self.current_index < len(self.filtered_data):
            current_row = self.filtered_data.iloc[self.current_index]
            key = (current_row['HoleID'], int(current_row['From']), int(current_row['To']))
            
            if key not in self.user_interacted:
                self.user_interacted[key] = {
                    'toggles_touched': set(),
                    'comments_touched': False,
                    'first_interaction_time': datetime.now()
                }
            
            # Mark all toggles as touched since we copied them
            self.user_interacted[key]['toggles_touched'].update(self.previous_toggle_states.keys())
                
        # Mark as having unsaved changes
        self._mark_unsaved_changes()
                
        # Show feedback to user
        DialogHelper.show_message(
            self.dialog,
            DialogHelper.t("Success"),
            DialogHelper.t("Copied toggle states from previous image."),
            message_type="info"
        )
            
    def _save_current_data(self):
            """Save current review data to JSON register."""
            if self.filtered_data.empty or self.current_index >= len(self.filtered_data):
                return
                
            current_row = self.filtered_data.iloc[self.current_index]
            key = (current_row['HoleID'], int(current_row['From']), int(current_row['To']))
            
            # ===================================================
            # NEW: Check if user actually interacted with this compartment
            # ===================================================
            if key not in self.user_interacted:
                # No interaction - don't save
                self.logger.debug(f"No interaction detected for {key}, skipping save")
                return
            
            interaction_info = self.user_interacted[key]
            if not interaction_info['toggles_touched'] and not interaction_info['comments_touched'] and not interaction_info.get('wet_dry_touched', False):
                # No actual interaction - don't save
                self.logger.debug(f"No meaningful interaction for {key}, skipping save")
                return
            
            # Save current toggle states as "previous" for next image
            self.previous_toggle_states = {}
            for name, toggle in self.toggle_vars.items():
                if toggle.has_interacted():
                    self.previous_toggle_states[name] = toggle.get()
            
            # Get comments
            comments = self.comment_text.get(1.0, tk.END).strip()
            
            if self.json_manager:
                # ===================================================
                # NEW: Determine review number for this session
                # ===================================================
                if key in self.session_review_numbers:
                    # Use the review number already assigned in this session
                    review_number = self.session_review_numbers[key]
                else:
                    # Get existing review to determine next number
                    existing_review = self.json_manager.get_user_review(
                        current_row['HoleID'],
                        int(current_row['From']),
                        int(current_row['To'])
                    )
                    
                    if existing_review:
                        # This is an edit - increment review number
                        review_number = existing_review.get('Review_Number', 0) + 1
                    else:
                        # This is a new review
                        review_number = 1
                        
                    # Store for this session
                    self.session_review_numbers[key] = review_number
                
                # Prepare review data from interacted toggles only
                review_data = {}
                for toggle_name, toggle in self.toggle_vars.items():
                    if toggle.has_interacted():
                        # Only save toggles that user actually touched
                        db_field_name = self._convert_toggle_name_to_field(toggle_name)
                        review_data[db_field_name] = toggle.get()
                
                # ===================================================
                # MODIFIED: Pass review number to prevent auto-increment
                # ===================================================
                # Note: This requires updating json_register_manager to accept review_number parameter
                self.json_manager.update_compartment_review(
                    hole_id=current_row['HoleID'],
                    depth_from=int(current_row['From']),
                    depth_to=int(current_row['To']),
                    comments=comments if comments else None,
                    review_number=review_number,  # Pass explicit review number
                    **review_data
                )
                
                self.logger.info(f"Saved review #{review_number} for {current_row['HoleID']} {current_row['From']}-{current_row['To']}")
                self.changes_made = True
                
                # ===================================================
                # ADD: Update main compartment register if wet/dry changed
                # ===================================================
                if interaction_info.get('wet_dry_touched', False):
                    new_wet_dry_status = self.wet_dry_var.get()
                    
                    if new_wet_dry_status != "Unknown":
                        # Get current photo status
                        current_photo_status = current_row.get('Photo_Status', '')
                        
                        # Determine if there's a change
                        status_changed = False
                        if new_wet_dry_status == "Wet" and 'Wet' not in current_photo_status:
                            status_changed = True
                            new_photo_status = "OK_Wet"
                        elif new_wet_dry_status == "Dry" and 'Dry' not in current_photo_status:
                            status_changed = True
                            new_photo_status = "OK_Dry"
                        
                        if status_changed:
                            # Update the main compartment register
                            self.json_manager.update_compartment(
                                hole_id=current_row['HoleID'],
                                depth_from=int(current_row['From']),
                                depth_to=int(current_row['To']),
                                photo_status=new_photo_status,
                                approved_by=os.getenv("USERNAME", "Unknown"),
                                comments=f"Wet/Dry status updated via logging review (was: {current_photo_status})"
                            )
                            
                            self.logger.info(f"Updated wet/dry status to {new_wet_dry_status}")
                            
                            # Note: Files are NOT renamed - they keep original suffixes
                    
            # Clear unsaved indicator
            self._clear_unsaved_indicator()

    def _save_changes(self):
        """Save all changes."""
        # Save current data first
        self._save_current_data()
        
        if self.changes_made:
            DialogHelper.show_message(
                self.dialog,
                DialogHelper.t("Success"),
                DialogHelper.t("All changes saved successfully."),
                message_type="info"
            )
            self.changes_made = False
        else:
            DialogHelper.show_message(
                self.dialog,
                DialogHelper.t("Info"),
                DialogHelper.t("No changes to save."),
                message_type="info"
            )
            
    def _quit_dialog(self):
            """Close the dialog with confirmation if changes pending."""
            # ===================================================
            # NEW CODE: Cancel auto-save timer first
            # ===================================================
            if self.auto_save_timer:
                self.dialog.after_cancel(self.auto_save_timer)
            
            # Check for unsaved changes in current image
            if self.unsaved_changes:
                if DialogHelper.confirm_dialog(
                    self.dialog,
                    DialogHelper.t("Unsaved Changes"),
                    DialogHelper.t("You have unsaved changes for the current image. Save before closing?")
                ):
                    self._save_current_data()
            
            # Check if there are any pending changes that need final save
            if self.changes_made:
                # Perform final save to ensure everything is persisted
                try:
                    self._perform_auto_save()
                    DialogHelper.show_message(
                        self.dialog,
                        DialogHelper.t("Info"),
                        DialogHelper.t("All changes have been saved."),
                        message_type="info"
                    )
                except Exception as e:
                    self.logger.error(f"Error during final save: {str(e)}")
                    if DialogHelper.confirm_dialog(
                        self.dialog,
                        DialogHelper.t("Save Error"),
                        DialogHelper.t("Failed to save some changes. Close anyway?")
                    ):
                        pass  # User chose to close anyway
                    else:
                        return  # Don't close the dialog
                    
            self.dialog.destroy()