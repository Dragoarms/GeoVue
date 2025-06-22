"""
GUI for designing and configuring drillhole trace visualizations.
Allows users to load multiple data sources, configure plot columns, and preview results.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
import cv2
import numpy as np
from PIL import Image, ImageTk
from typing import Dict, List, Optional, Tuple, Any
import logging
import traceback

from gui.dialog_helper import DialogHelper
from processing.drillhole_data_manager import DrillholeDataManager, DataType
from processing.drillhole_data_visualizer import (
    DrillholeDataVisualizer, VisualizationMode, PlotType, PlotConfig
)
from processing.color_map_manager import ColorMapManager, ColorMap, ColorMapType
from gui.color_map_config_dialog import ColorMapConfigDialog

logger = logging.getLogger(__name__)


class DrillholeTraceDesigner:
    """
    GUI for designing drillhole trace visualizations.
    
    Features:
    - Load multiple CSV data sources
    - Configure visualization columns
    - Preview with sample data
    - Save configuration for reuse
    """
    
    def __init__(self, parent, file_manager, gui_manager, config=None):
        """
        Initialize the trace designer dialog.
        
        Args:
            parent: Parent window
            file_manager: FileManager instance
            gui_manager: GUIManager instance for theming
            config: Optional configuration dictionary
        """
        self.parent = parent
        self.file_manager = file_manager
        self.gui_manager = gui_manager
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize data manager and visualizer
        self.data_manager = DrillholeDataManager()
        self.visualizer = DrillholeDataVisualizer(VisualizationMode.STITCHABLE)

        # Initialize color map manager
        resources_dir = os.path.join(os.path.dirname(__file__), '..', 'resources')
        self.color_map_manager = ColorMapManager(resources_dir)
        self.color_map_manager.create_default_presets()

        # Storage for UI elements
        self.data_source_widgets = []
        self.plot_column_widgets = []
        self.preview_image = None
        
        # Create the dialog
        self._create_dialog()
        
    def _create_dialog(self):
        """Create the main designer dialog."""
        self.dialog = DialogHelper.create_dialog(
            self.parent,
            title="Drillhole Trace Designer",
            modal=True,
            size_ratio=0.9,
            min_width=1400,
            min_height=900
        )
        
        # Apply theme
        theme_colors = self.gui_manager.theme_colors
        self.dialog.configure(bg=theme_colors["background"])
        
        # Main container with padding
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create three-panel layout
        self._create_left_panel(main_frame)   # Data sources
        self._create_center_panel(main_frame) # Plot configuration
        self._create_right_panel(main_frame)  # Preview
        
        # Bottom button panel
        self._create_button_panel(main_frame)
        
        # Load sample data for preview
        self._load_sample_data()
        
    def _create_left_panel(self, parent):
        """Create the data sources panel."""
        # Container frame
        left_frame = ttk.Frame(parent, width=350)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        left_frame.pack_propagate(False)
        
        # Title
        title_label = ttk.Label(
            left_frame,
            text=DialogHelper.t("Data Sources"),
            font=self.gui_manager.fonts["subtitle"]
        )
        title_label.pack(pady=(0, 10))
        
        # Buttons frame
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Add CSV button
        self.add_csv_button = self.gui_manager.create_modern_button(
            button_frame,
            text=DialogHelper.t("Add CSV"),
            color=self.gui_manager.theme_colors["accent_green"],
            command=self._add_csv_file,
            icon="+"
        )
        self.add_csv_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Clear all button
        self.clear_button = self.gui_manager.create_modern_button(
            button_frame,
            text=DialogHelper.t("Clear All"),
            color=self.gui_manager.theme_colors["accent_red"],
            command=self._clear_data_sources,
            icon="✖"
        )
        self.clear_button.pack(side=tk.LEFT)
        
        # Data sources list frame with scrollbar
        list_container = ttk.Frame(left_frame)
        list_container.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for scrolling
        canvas = tk.Canvas(
            list_container,
            bg=self.gui_manager.theme_colors["field_bg"],
            highlightthickness=0
        )
        scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=canvas.yview)
        
        self.data_sources_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=self.data_sources_frame, anchor="nw")
        
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure canvas scrolling
        def configure_scroll(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
            
        self.data_sources_frame.bind("<Configure>", configure_scroll)
        
        # Status label
        self.data_status_label = ttk.Label(
            left_frame,
            text=DialogHelper.t("No data sources loaded"),
            font=self.gui_manager.fonts["small"]
        )
        self.data_status_label.pack(pady=(10, 0))
        
    def _create_center_panel(self, parent):
        """Create the plot configuration panel."""
        # Container frame
        center_frame = ttk.Frame(parent, width=500)
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Title
        title_label = ttk.Label(
            center_frame,
            text=DialogHelper.t("Plot Configuration"),
            font=self.gui_manager.fonts["subtitle"]
        )
        title_label.pack(pady=(0, 10))
        
        # Orientation selector
        orientation_frame = ttk.Frame(center_frame)
        orientation_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(
            orientation_frame,
            text=DialogHelper.t("Compartment Orientation:")
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        self.orientation_var = tk.StringVar(value="horizontal")
        
        # Custom styled dropdown frame
        orientation_combo_frame = tk.Frame(
            orientation_frame,
            bg=self.gui_manager.theme_colors["field_bg"],
            highlightbackground=self.gui_manager.theme_colors["field_border"],
            highlightthickness=1,
            bd=0
        )
        orientation_combo_frame.pack(side=tk.LEFT)
        
        # Orientation dropdown
        orientation_choices = ["horizontal", "vertical"]
        orientation_dropdown = tk.OptionMenu(
            orientation_combo_frame, 
            self.orientation_var, 
            *orientation_choices
        )
        self.gui_manager.style_dropdown(orientation_dropdown, width=15)
        orientation_dropdown.pack()
        self.orientation_var.trace_add("write", lambda *args: self._update_preview())        
        
        # Buttons frame
        button_frame = ttk.Frame(center_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Add plot column button
        self.add_plot_button = self.gui_manager.create_modern_button(
            button_frame,
            text=DialogHelper.t("Add Plot Column"),
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._add_plot_column,
            icon="+"
        )
        self.add_plot_button.pack(side=tk.LEFT)
        
        # Plot columns frame with scrollbar
        list_container = ttk.Frame(center_frame)
        list_container.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for scrolling
        canvas = tk.Canvas(
            list_container,
            bg=self.gui_manager.theme_colors["field_bg"],
            highlightthickness=0
        )
        scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=canvas.yview)
        
        self.plot_columns_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=self.plot_columns_frame, anchor="nw")
        
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure canvas scrolling
        def configure_scroll(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
            
        self.plot_columns_frame.bind("<Configure>", configure_scroll)
        
    def _create_right_panel(self, parent):
        """Create the preview panel."""
        # Container frame
        right_frame = ttk.Frame(parent, width=400)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH)
        right_frame.pack_propagate(False)
        
        # Title
        title_label = ttk.Label(
            right_frame,
            text=DialogHelper.t("Preview"),
            font=self.gui_manager.fonts["subtitle"]
        )
        title_label.pack(pady=(0, 10))
        
        # Preview controls
        controls_frame = ttk.Frame(right_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Update preview button
        self.update_preview_button = self.gui_manager.create_modern_button(
            controls_frame,
            text=DialogHelper.t("Update Preview"),
            color=self.gui_manager.theme_colors["accent_green"],
            command=self._update_preview,
            icon="↻"
        )
        self.update_preview_button.pack(side=tk.LEFT)
        
        # Preview canvas
        self.preview_canvas = tk.Canvas(
            right_frame,
            bg=self.gui_manager.theme_colors["secondary_bg"],
            highlightthickness=1,
            highlightbackground=self.gui_manager.theme_colors["border"]
        )
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Preview info label
        self.preview_info_label = ttk.Label(
            right_frame,
            text=DialogHelper.t("Using sample data for preview"),
            font=self.gui_manager.fonts["small"]
        )
        self.preview_info_label.pack(pady=(10, 0))
        
    def _create_button_panel(self, parent):
        """Create the bottom button panel."""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        # Right-align buttons
        button_container = ttk.Frame(button_frame)
        button_container.pack(side=tk.RIGHT)
        
        # Save configuration button
        save_button = self.gui_manager.create_modern_button(
            button_container,
            text=DialogHelper.t("Save Configuration"),
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._save_configuration
        )
        save_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Load configuration button
        load_button = self.gui_manager.create_modern_button(
            button_container,
            text=DialogHelper.t("Load Configuration"),
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._load_configuration
        )
        load_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Generate button
        self.generate_button = self.gui_manager.create_modern_button(
            button_container,
            text=DialogHelper.t("Generate Traces"),
            color=self.gui_manager.theme_colors["accent_green"],
            command=self._generate_traces
        )
        self.generate_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Cancel button
        cancel_button = self.gui_manager.create_modern_button(
            button_container,
            text=DialogHelper.t("Cancel"),
            color=self.gui_manager.theme_colors["accent_red"],
            command=self.dialog.destroy
        )
        cancel_button.pack(side=tk.LEFT)
        
    def _add_csv_file(self):
        """Add a CSV file to the data sources."""
        # Make sure dialog is on top
        self.dialog.lift()
        self.dialog.focus_force()
        
        file_paths = filedialog.askopenfilenames(
            parent=self.dialog,  # Add parent parameter
            title=DialogHelper.t("Select CSV Files"),
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        # Restore focus to dialog
        self.dialog.lift()
        
        if not file_paths:
            return
            
        # Load files
        results = self.data_manager.load_csv_files(list(file_paths))
        
        # Update UI for each file
        for filepath, status in results.items():
            self._add_data_source_widget(filepath, status)
            
        # Update status
        self._update_data_status()
        
        # Update available columns for plot configuration
        self._update_available_columns()
        
    def _add_data_source_widget(self, filepath: str, status: str):
        """Add a widget for a data source."""
        # Container frame
        source_frame = ttk.Frame(
            self.data_sources_frame,
            relief=tk.RIDGE,
            borderwidth=1
        )
        source_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Filename
        filename = os.path.basename(filepath)
        name_label = ttk.Label(
            source_frame,
            text=filename,
            font=self.gui_manager.fonts["normal"]
        )
        name_label.pack(anchor=tk.W, padx=5, pady=(5, 0))
        
        # Status
        is_error = "Error" in status or "Missing" in status
        status_color = "red" if is_error else "green"
        
        status_label = ttk.Label(
            source_frame,
            text=status,
            font=self.gui_manager.fonts["small"],
            foreground=status_color
        )
        status_label.pack(anchor=tk.W, padx=5, pady=(0, 5))
        
        # Remove button
        remove_button = self.gui_manager.create_modern_button(
            source_frame,
            text="✖",
            color=self.gui_manager.theme_colors["accent_red"],
            command=lambda: self._remove_data_source(filename, source_frame)
        )
        remove_button.place(relx=1.0, rely=0.5, anchor=tk.E, x=-5)
        
        # Store widget reference
        self.data_source_widgets.append({
            'filename': filename,
            'frame': source_frame,
            'status': status
        })
        
    def _remove_data_source(self, filename: str, frame: ttk.Frame):
        """Remove a data source."""
        # Remove from data manager
        if filename in self.data_manager.data_sources:
            del self.data_manager.data_sources[filename]
            del self.data_manager.data_info[filename]
            
        # Remove widget
        frame.destroy()
        
        # Update widget list
        self.data_source_widgets = [
            w for w in self.data_source_widgets 
            if w['filename'] != filename
        ]
        
        # Update status
        self._update_data_status()
        self._update_available_columns()
        
    def _clear_data_sources(self):
        """Clear all data sources."""
        # Clear data manager
        self.data_manager.data_sources.clear()
        self.data_manager.data_info.clear()
        self.data_manager.hole_ids.clear()
        
        # Clear widgets
        for widget_info in self.data_source_widgets:
            widget_info['frame'].destroy()
            
        self.data_source_widgets.clear()
        
        # Update status
        self._update_data_status()
        self._update_available_columns()
        
    def _update_data_status(self):
        """Update the data status label."""
        num_sources = len(self.data_manager.data_sources)
        num_holes = len(self.data_manager.hole_ids)
        
        if num_sources == 0:
            status = DialogHelper.t("No data sources loaded")
        else:
            status = DialogHelper.t(f"{num_sources} sources, {num_holes} holes")
            
        self.data_status_label.config(text=status)
        
    def _update_available_columns(self):
        """Update available columns in plot configuration widgets."""
        # Get all available columns
        available = self.data_manager.get_available_columns()
        
        # Build column list with source info
        columns = []
        for source, source_columns in available.items():
            for col_name, col_type in source_columns:
                columns.append(f"{col_name} ({source})")
        
        # Update each plot widget
        for widget_info in self.plot_column_widgets:
            if 'column_combos' in widget_info:
                for combo_info in widget_info['column_combos']:
                    dropdown = combo_info['widget']
                    column_var = combo_info['var']
                    combo_frame = combo_info['combo_frame']
                    
                    # Store current value
                    current_value = column_var.get()
                    
                    # Destroy old dropdown
                    dropdown.destroy()
                    
                    # Create new dropdown with updated values
                    new_dropdown = tk.OptionMenu(
                        combo_frame,
                        column_var,
                        *columns if columns else [""]
                    )
                    self.gui_manager.style_dropdown(new_dropdown, width=30)
                    new_dropdown.pack()
                    
                    # Update reference
                    combo_info['widget'] = new_dropdown
                    
                    # Restore value if it still exists
                    if current_value in columns:
                        column_var.set(current_value)
                    elif columns:
                        column_var.set(columns[0])
                    
    def _add_plot_column(self):
            """Add a new plot column configuration."""
            # Container frame
            plot_frame = ttk.Frame(
                self.plot_columns_frame,
                relief=tk.RIDGE,
                borderwidth=1
            )
            plot_frame.pack(fill=tk.X, pady=(0, 10))
            
            # Header with title and remove button
            header_frame = ttk.Frame(plot_frame)
            header_frame.pack(fill=tk.X, padx=5, pady=5)
            
            plot_num = len(self.plot_column_widgets) + 1
            title_label = ttk.Label(
                header_frame,
                text=DialogHelper.t(f"Plot Column {plot_num}"),
                font=self.gui_manager.fonts["heading"]
            )
            title_label.pack(side=tk.LEFT)
            
            # Remove button
            remove_button = self.gui_manager.create_modern_button(
                header_frame,
                text="✖",
                color=self.gui_manager.theme_colors["accent_red"],
                command=lambda: self._remove_plot_column(plot_frame)
            )
            remove_button.pack(side=tk.RIGHT)
            
            # Plot type selector
            type_frame = ttk.Frame(plot_frame)
            type_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Label(type_frame, text=DialogHelper.t("Plot Type:")).pack(side=tk.LEFT, padx=(0, 10))
            
            type_var = tk.StringVar(value="solid_column")
            plot_types = ["solid_column", "stacked_bar", "line_graph", "ternary", "scatter", "histogram"]
            
            # Custom styled dropdown frame
            type_combo_frame = tk.Frame(
                type_frame,
                bg=self.gui_manager.theme_colors["field_bg"],
                highlightbackground=self.gui_manager.theme_colors["field_border"],
                highlightthickness=1,
                bd=0
            )
            type_combo_frame.pack(side=tk.LEFT)
            
            # Type dropdown
            type_dropdown = tk.OptionMenu(
                type_combo_frame, 
                type_var, 
                *plot_types
            )
            self.gui_manager.style_dropdown(type_dropdown, width=20)
            type_dropdown.pack()
            
            # Width setting
            width_frame = ttk.Frame(plot_frame)
            width_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Label(width_frame, text=DialogHelper.t("Width (pixels):")).pack(side=tk.LEFT, padx=(0, 10))
            
            width_var = tk.IntVar(value=100)
            width_entry = self.gui_manager.create_entry_with_validation(
                width_frame,
                width_var,
                width=10
            )
            width_entry.pack(side=tk.LEFT)
            
            # Color configuration
            color_frame = ttk.Frame(plot_frame)
            color_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Label(color_frame, text=DialogHelper.t("Color Map:")).pack(side=tk.LEFT, padx=(0, 10))
            
            color_label = ttk.Label(
                color_frame,
                text=DialogHelper.t("Default"),
                font=self.gui_manager.fonts["normal"],
                foreground=self.gui_manager.theme_colors["accent_blue"]
            )
            color_label.pack(side=tk.LEFT, padx=(0, 10))
            
            # ===================================================
            # Create columns_frame BEFORE creating widget_info
            # ===================================================
            # Column selection frame
            columns_frame = ttk.Frame(plot_frame)
            columns_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Label(
                columns_frame,
                text=DialogHelper.t("Data Columns:"),
                font=self.gui_manager.fonts["normal"]
            ).pack(anchor=tk.W)
            
            # Store widget info - now columns_frame exists
            widget_info = {
                'frame': plot_frame,
                'type_var': type_var,
                'type_dropdown': type_dropdown,
                'width_var': width_var,
                'width_entry': width_entry,  # Add this reference
                'column_combos': [],
                'columns_frame': columns_frame,
                'color_map': None,
                'color_label': color_label
            }
            
            # Configure color button
            color_map = None  # Store in closure
            
            def configure_colors():
                nonlocal color_map
                # Get selected columns to determine data values
                data_values = []
                if widget_info['column_combos']:
                    col_str = widget_info['column_combos'][0]['var'].get()
                    if col_str:
                        col_name = col_str.split(' (')[0]
                        # Get sample values from data
                        for source_name, df in self.data_manager.data_sources.items():
                            if col_name in df.columns:
                                data_values.extend(df[col_name].dropna().unique().tolist())
                                break
                                
                # Show color configuration dialog
                dialog = ColorMapConfigDialog(
                    self.dialog,
                    self.gui_manager,
                    self.color_map_manager,
                    data_values=data_values,
                    current_map=color_map
                )
                
                result = dialog.show()
                if result:
                    color_map = result
                    color_label.config(text=color_map.name or DialogHelper.t("Custom"))
                    widget_info['color_map'] = color_map
                    
            color_button = self.gui_manager.create_modern_button(
                color_frame,
                text=DialogHelper.t("Configure"),
                color=self.gui_manager.theme_colors["accent_blue"],
                command=configure_colors
            )
            color_button.pack(side=tk.LEFT)
            
            # Add initial column selector
            self._add_column_selector(widget_info)
            
            # Bind type change to update column selectors
            type_var.trace_add("write", lambda *args: self._on_plot_type_change(widget_info))
            
            # Store widget reference
            self.plot_column_widgets.append(widget_info)
            
            # ===================================================
            # Check if ternary is selected initially and disable width
            # ===================================================
            if type_var.get() == "ternary":
                width_entry.config(state='disabled')


    def _add_column_selector(self, widget_info: dict):
        """Add a column selector to a plot widget."""
        columns_frame = widget_info['columns_frame']
        
        # Column selector frame
        selector_frame = ttk.Frame(columns_frame)
        selector_frame.pack(fill=tk.X, pady=2)
        
        # Get available columns
        available = self.data_manager.get_available_columns()
        columns = []
        
        for source, source_columns in available.items():
            for col_name, col_type in source_columns:
                columns.append(f"{col_name} ({source})")
                
        # Column variable
        column_var = tk.StringVar()
        if columns:
            column_var.set(columns[0])  # Set default value
        
        # Custom styled dropdown frame
        column_combo_frame = tk.Frame(
            selector_frame,
            bg=self.gui_manager.theme_colors["field_bg"],
            highlightbackground=self.gui_manager.theme_colors["field_border"],
            highlightthickness=1,
            bd=0
        )
        column_combo_frame.pack(side=tk.LEFT, padx=(20, 5))
        
        # Column dropdown
        column_dropdown = tk.OptionMenu(
            column_combo_frame, 
            column_var, 
            *columns if columns else [""]
        )
        self.gui_manager.style_dropdown(column_dropdown, width=30)
        column_dropdown.pack()
        
        # Remove button for this column
        if len(widget_info['column_combos']) > 0:  # Keep at least one
            remove_btn = self.gui_manager.create_modern_button(
                selector_frame,
                text="−",
                color=self.gui_manager.theme_colors["accent_red"],
                command=lambda: self._remove_column_selector(widget_info, selector_frame, column_var)
            )
            remove_btn.pack(side=tk.LEFT)
            
        # Store reference
        widget_info['column_combos'].append({
            'frame': selector_frame,
            'var': column_var,
            'widget': column_dropdown,
            'combo_frame': column_combo_frame
        })
        
    def _remove_column_selector(self, widget_info: dict, frame: ttk.Frame, var: tk.StringVar):
        """Remove a column selector."""
        frame.destroy()
        widget_info['column_combos'] = [
            c for c in widget_info['column_combos'] 
            if c['var'] != var
        ]
        
    def _on_plot_type_change(self, widget_info: dict):
            """Handle plot type change."""
            plot_type = widget_info['type_var'].get()
            
            # ===================================================
            # Disable width entry for ternary plots
            # ===================================================
            if plot_type == "ternary":
                # Find the width entry widget and disable it
                if 'width_entry' in widget_info:
                    widget_info['width_entry'].config(state='disabled')
                    # Set width to match expected height based on orientation
                    if hasattr(self, 'orientation_var'):
                        if self.orientation_var.get() == "vertical":
                            widget_info['width_var'].set(300)  # Taller compartments
                        else:
                            widget_info['width_var'].set(200)  # Wider compartments
                    else:
                        widget_info['width_var'].set(300)  # Default square size
            else:
                # Re-enable width entry for other plot types
                if 'width_entry' in widget_info:
                    widget_info['width_entry'].config(state='normal')
            
            # Determine required number of columns
            if plot_type == "ternary":
                required_columns = 3
            elif plot_type in ["scatter"]:
                required_columns = 2
            elif plot_type == "stacked_bar":
                required_columns = 2  # At least 2 for stacking
            else:
                required_columns = 1
                
            current_columns = len(widget_info['column_combos'])
            
            # Add more column selectors if needed
            while current_columns < required_columns:
                self._add_column_selector(widget_info)
                current_columns += 1
                
            # Add "Add Column" button for stacked bar
            if plot_type == "stacked_bar" and not hasattr(widget_info, 'add_column_button'):
                add_btn = self.gui_manager.create_modern_button(
                    widget_info['columns_frame'],
                    text=DialogHelper.t("Add Column"),
                    color=self.gui_manager.theme_colors["accent_green"],
                    command=lambda: self._add_column_selector(widget_info)
                )
                add_btn.pack(pady=5)
                widget_info['add_column_button'] = add_btn
            elif hasattr(widget_info, 'add_column_button') and plot_type != "stacked_bar":
                widget_info['add_column_button'].destroy()
                del widget_info['add_column_button']

    def _remove_plot_column(self, frame: ttk.Frame):
        """Remove a plot column configuration."""
        # Find and remove from list
        self.plot_column_widgets = [
            w for w in self.plot_column_widgets 
            if w['frame'] != frame
        ]
        
        # Destroy frame
        frame.destroy()
        
        # Renumber remaining plots
        for i, widget_info in enumerate(self.plot_column_widgets):
            title_label = widget_info['frame'].winfo_children()[0].winfo_children()[0]
            if isinstance(title_label, ttk.Label):
                title_label.config(text=DialogHelper.t(f"Plot Column {i+1}"))
                
    def _load_sample_data(self):
        """Load sample data for preview."""
        # Create sample data
        sample_data = pd.DataFrame({
            'holeid': ['SAMPLE'] * 10,
            'from': range(0, 10),
            'to': range(1, 11),
            'lithology': ['Sandstone', 'Shale', 'Sandstone', 'Limestone', 'Shale',
                         'Sandstone', 'Limestone', 'Shale', 'Sandstone', 'Limestone'],
            'fe_pct': [45.2, 52.1, 48.3, 35.6, 55.8, 46.9, 38.2, 54.3, 47.5, 36.8],
            'sio2_pct': [15.3, 8.2, 12.5, 25.8, 6.9, 14.1, 22.3, 7.5, 13.8, 24.1],
            'al2o3_pct': [5.2, 12.1, 6.8, 3.2, 14.5, 5.9, 3.8, 13.2, 6.1, 3.5]
        })
        
        # Add to data manager
        self.data_manager.data_sources['sample.csv'] = sample_data
        self.data_manager.data_info['sample.csv'] = {
            'columns': {
                'lithology': {'type': DataType.CATEGORICAL},
                'fe_pct': {'type': DataType.NUMERIC},
                'sio2_pct': {'type': DataType.NUMERIC},
                'al2o3_pct': {'type': DataType.NUMERIC}
            },
            'interval_scale': 1.0,
            'row_count': len(sample_data),
            'hole_ids': ['SAMPLE']
        }
        
    def _update_preview(self):
            """Update the preview visualization."""
            try:
                # Clear existing visualizer configuration
                self.visualizer.plot_configs.clear()
                
                # Build configuration from UI
                for widget_info in self.plot_column_widgets:
                    plot_type = widget_info['type_var'].get()
                    width = widget_info['width_var'].get()
                    
                    # Create plot config
                    config = PlotConfig(
                        plot_type=PlotType[plot_type.upper()],
                        width=width
                    )
                    
                    # Get selected columns
                    for combo_info in widget_info['column_combos']:
                        col_str = combo_info['var'].get()
                        if col_str:
                            # Extract column name (remove source info)
                            col_name = col_str.split(' (')[0]
                            config.columns.append(col_name)
                            
                    # Apply color map if configured
                    if widget_info.get('color_map'):
                        color_map = widget_info['color_map']
                        config.color_map = {}
                        
                        # Convert ColorMap to plot config format
                        if color_map.type == ColorMapType.CATEGORICAL:
                            config.color_map.update(color_map.categories)
                        elif color_map.type == ColorMapType.NUMERIC:
                            # For numeric, we'll handle it in the visualizer
                            config.custom_params['color_map_obj'] = color_map
                            
                    # Add to visualizer
                    self.visualizer.add_plot_column(config)
                    
                # Generate preview using sample data
                if 'sample.csv' in self.data_manager.data_sources:
                    sample_data = self.data_manager.data_sources['sample.csv']
                    
                    # Create preview for middle compartment
                    compartment_data = sample_data.iloc[5].to_dict()
                    
                    # ===================================================
                    # Try to use a real compartment image if available
                    # ===================================================
                    compartment_img = self._get_sample_compartment_image()
                    
                    if compartment_img is None:
                        # Fallback to generated sample
                        if self.orientation_var.get() == "vertical":
                            # Vertical orientation - taller than wide
                            compartment_img = np.ones((300, 200, 3), dtype=np.uint8) * 180
                            cv2.putText(compartment_img, "Sample", (50, 140),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                            cv2.putText(compartment_img, "Compartment", (30, 170),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                            viz_height = 300
                        else:
                            # Horizontal orientation - wider than tall
                            compartment_img = np.ones((200, 300, 3), dtype=np.uint8) * 180
                            cv2.putText(compartment_img, "Sample Compartment", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                            viz_height = 200
                    else:
                        # Use actual image dimensions
                        viz_height = compartment_img.shape[0]
                    
                    # Generate visualization with appropriate height
                    preview_image = self.visualizer.generate_compartment_visualization(
                        compartment_data=compartment_data,
                        full_hole_data=sample_data,
                        depth_from=5,
                        depth_to=6,
                        height=viz_height
                    )
                    
                    # Combine compartment and data
                    if preview_image.shape[1] > 0:
                        combined = np.hstack([compartment_img, preview_image])
                    else:
                        combined = compartment_img
                        
                    # Display in canvas
                    self._display_preview_image(combined)
                    
            except Exception as e:
                self.logger.error(f"Error updating preview: {str(e)}")
                self.logger.error(traceback.format_exc())
                
                # Show error in preview
                error_img = np.ones((200, 400, 3), dtype=np.uint8) * 255
                cv2.putText(error_img, "Preview Error", (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                self._display_preview_image(error_img)
        
    def _get_sample_compartment_image(self) -> Optional[np.ndarray]:
        """
        Try to load a real compartment image for the preview.
        
        Returns:
            Compartment image array or None if not found
        """
        try:
            # Check if we have access to the file manager
            if not self.file_manager:
                return None
                
            # Look for a compartment image in the processed directory
            compartments_dir = self.file_manager.dir_structure.get("chip_compartments")
            if not compartments_dir or not os.path.exists(compartments_dir):
                return None
                
            # Find the first available compartment image
            for hole_dir in os.listdir(compartments_dir):
                hole_path = os.path.join(compartments_dir, hole_dir)
                if os.path.isdir(hole_path):
                    # Look for compartment images
                    for img_file in os.listdir(hole_path):
                        if img_file.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
                            img_path = os.path.join(hole_path, img_file)
                            img = cv2.imread(img_path)
                            if img is not None:
                                # Apply orientation if needed
                                if self.orientation_var.get() == "vertical":
                                    # Keep vertical orientation
                                    return img
                                else:
                                    # Rotate to horizontal
                                    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                                    
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading sample compartment image: {str(e)}")
            return None

    def _display_preview_image(self, image: np.ndarray):
        """Display an image in the preview canvas."""
        # Get canvas size
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            self.preview_canvas.update_idletasks()
            canvas_width = self.preview_canvas.winfo_width()
            canvas_height = self.preview_canvas.winfo_height()
            
        # Scale image to fit canvas
        h, w = image.shape[:2]
        scale = min(canvas_width / w, canvas_height / h) * 0.9  # 90% to leave margin
        
        new_width = int(w * scale)
        new_height = int(h * scale)
        
        scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Convert to PIL
        image_rgb = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Convert to PhotoImage
        self.preview_image = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and display
        self.preview_canvas.delete("all")
        
        # Center image
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        
        self.preview_canvas.create_image(x, y, anchor=tk.NW, image=self.preview_image)
        
    def _save_configuration(self):
        """Save the current configuration to a file."""
        # Make sure dialog is on top
        self.dialog.lift()
        self.dialog.focus_force()
        
        filepath = filedialog.asksaveasfilename(
            parent=self.dialog,
            title=DialogHelper.t("Save Configuration"),
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        # Restore focus to dialog
        self.dialog.lift()
        
        if not filepath:
            return
            
        try:
            # Build configuration
            config = {
                'orientation': self.orientation_var.get(),
                'data_sources': list(self.data_manager.data_sources.keys()),
                'plot_columns': []
            }
            
            for widget_info in self.plot_column_widgets:
                plot_config = {
                    'type': widget_info['type_var'].get(),
                    'width': widget_info['width_var'].get(),
                    'columns': [c['var'].get() for c in widget_info['column_combos'] if c['var'].get()]
                }
                
                # Save color map if present
                if widget_info.get('color_map'):
                    plot_config['color_map'] = widget_info['color_map'].to_dict()
                    
                config['plot_columns'].append(plot_config)
                
            # Save to file
            import json
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
                
            DialogHelper.show_message(
                self.dialog,
                DialogHelper.t("Success"),
                DialogHelper.t("Configuration saved successfully"),
                message_type="info"
            )
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
            DialogHelper.show_message(
                self.dialog,
                DialogHelper.t("Error"),
                DialogHelper.t(f"Error saving configuration: {str(e)}"),
                message_type="error"
            )

    def _load_configuration(self):
        """Load a configuration from file."""
        # Make sure dialog is on top
        self.dialog.lift()
        self.dialog.focus_force()
        
        filepath = filedialog.askopenfilename(
            parent=self.dialog,  # Add parent parameter
            title=DialogHelper.t("Load Configuration"),
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        # Restore focus to dialog
        self.dialog.lift()
        
        if not filepath:
            return
            
        try:
            import json
            with open(filepath, 'r') as f:
                config = json.load(f)
                
            # Clear existing configuration
            self._clear_data_sources()
            for widget_info in self.plot_column_widgets[:]:
                self._remove_plot_column(widget_info['frame'])
                
            # Set orientation
            self.orientation_var.set(config.get('orientation', 'horizontal'))
            
            # Note: Data sources would need to be re-loaded manually
            # as we only store filenames, not full paths
            
            # Create plot columns
            for plot_config in config.get('plot_columns', []):
                self._add_plot_column()
                
                # Configure last added plot
                widget_info = self.plot_column_widgets[-1]
                widget_info['type_var'].set(plot_config['type'])
                widget_info['width_var'].set(plot_config['width'])
                
                # Set columns
                for i, col in enumerate(plot_config['columns']):
                    if i >= len(widget_info['column_combos']):
                        self._add_column_selector(widget_info)
                    widget_info['column_combos'][i]['var'].set(col)
                    
            DialogHelper.show_message(
                self.dialog,
                DialogHelper.t("Success"),
                DialogHelper.t("Configuration loaded. Please re-load data sources."),
                message_type="info"
            )
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            DialogHelper.show_message(
                self.dialog,
                DialogHelper.t("Error"),
                DialogHelper.t(f"Error loading configuration: {str(e)}"),
                message_type="error"
            )

    def _generate_traces(self):
        """Generate traces with the current configuration."""
        # Validate configuration
        if not self.data_manager.data_sources:
            DialogHelper.show_message(
                self.dialog,
                DialogHelper.t("No Data"),
                DialogHelper.t("Please load data sources first"),
                message_type="warning"
            )
            return
            
        if not self.plot_column_widgets:
            DialogHelper.show_message(
                self.dialog,
                DialogHelper.t("No Plots"),
                DialogHelper.t("Please configure at least one plot column"),
                message_type="warning"
            )
            return
            
        # Store the configuration
        self.result = {
            'data_manager': self.data_manager,
            'visualizer': self.visualizer,
            'orientation': self.orientation_var.get()
        }
        
        # Close dialog
        self.dialog.destroy()
        
    def show(self) -> Optional[dict]:
        """
        Show the dialog and return the configuration.
        
        Returns:
            Dictionary with data_manager, visualizer, and orientation, or None if cancelled
        """
        self.result = None
        self.dialog.wait_window()
        return self.result