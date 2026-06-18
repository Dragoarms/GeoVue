# gui/widgets/advanced_filter_window.py
"""
Advanced Filter Window - Reusable scatter plot filtering widget for geological data.

This widget provides a comprehensive interface for:
- Dynamic filter creation with any column
- Interactive scatter plot visualization
- Lasso selection tool for data exploration
- Color mapping by geological attributes
- Export and analysis capabilities

Can be used standalone or integrated with existing managers.
"""

import os
import logging
import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Dict, Optional, Callable, Any, Union
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np

# Import custom widgets
from gui.widgets.modern_button import ModernButton
from gui.widgets.dynamic_filter_row import DynamicFilterRow
from gui.widgets.select_on_scatterplot_widget import ScatterSelectionWidget


@dataclass
class FilterWindowConfig:
    """Configuration for AdvancedFilterWindow"""
    
    # Window settings
    window_title: str = "Geological Data Filter & Exploration"
    window_width: int = 1400
    window_height: int = 900
    
    # Default axes
    default_x_axis: str = "Fe_pct_BEST"
    default_y_axis: str = "SiO2_pct_BEST"
    default_color_by: str = "classification"
    
    # Plot settings
    default_point_size: int = 20
    default_point_alpha: float = 0.6
    show_legend: bool = True
    
    # Filter settings
    default_filter_logic: str = "AND"  # "AND" or "OR"
    
    # Feature toggles
    enable_color_map_editor: bool = True
    enable_export: bool = True
    enable_statistics: bool = True
    
    # Lasso selection
    lasso_color: str = "#00ff00"
    lasso_alpha: float = 0.3
    selection_outline_color: str = "#ffff00"
    selection_outline_width: int = 2


class AdvancedFilterWindow:
    """
    Reusable advanced filtering window with scatter plot and lasso selection.
    
    Can operate in multiple modes:
    1. Direct data mode: Pass DataFrame directly
    2. Manager mode: Pull data from DrillholeDataManager
    3. Image mode: Extract data from image list with CSV data
    
    Usage Examples:
    
    # Mode 1: Direct data
    window = AdvancedFilterWindow(
        parent=root,
        gui_manager=gui_mgr,
        data=my_dataframe,
        on_selection=handle_selection
    )
    window.show()
    
    # Mode 2: With managers
    window = AdvancedFilterWindow(
        parent=root,
        gui_manager=gui_mgr,
        data_manager=drillhole_mgr,
        color_map_manager=color_mgr,
        item_manager=classification_mgr,
        on_selection=handle_selection
    )
    window.show()
    
    # Mode 3: From images
    window = AdvancedFilterWindow(
        parent=root,
        gui_manager=gui_mgr,
        images=image_list,
        on_selection=handle_selection
    )
    window.show()
    """
    
    def __init__(
        self,
        parent: tk.Widget,
        gui_manager: Any,
        data: Optional[pd.DataFrame] = None,
        images: Optional[List] = None,
        data_manager: Optional[Any] = None,
        color_map_manager: Optional[Any] = None,
        item_manager: Optional[Any] = None,
        config: Optional[FilterWindowConfig] = None,
        on_selection: Optional[Callable[[pd.DataFrame], None]] = None,
        on_filter_change: Optional[Callable[[pd.DataFrame], None]] = None,
        on_close: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize Advanced Filter Window
        
        Args:
            parent: Parent widget
            gui_manager: GUI manager for theming (required)
            data: Optional DataFrame to filter/visualize
            images: Optional list of images with csv_data attribute
            data_manager: Optional DrillholeDataManager
            color_map_manager: Optional ColorMapManager for geological attributes
            item_manager: Optional ImageClassificationAndTagManager
            config: Optional FilterWindowConfig for customization
            on_selection: Callback when lasso selection made: fn(selected_df)
            on_filter_change: Callback when filters applied: fn(filtered_df)
            on_close: Callback when window closes
        """
        self.parent = parent
        self.gui_manager = gui_manager
        self.theme_colors = gui_manager.theme_colors
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = config or FilterWindowConfig()
        
        # Data sources
        self.raw_data = data
        self.images = images
        self.data_manager = data_manager
        self.color_map_manager = color_map_manager
        self.item_manager = item_manager
        
        # Callbacks
        self.on_selection = on_selection
        self.on_filter_change = on_filter_change
        self.on_close = on_close
        
        # State
        self.window = None
        self.is_open = False
        self.current_data = None  # Currently displayed data
        self.filtered_data = None  # Data after filters applied
        self.plot_mode = "filtered"  # "filtered" or "full"
        
        # UI Components
        self.scatter_widget = None
        self.filter_rows = []
        self.filter_container = None
        self.filter_canvas = None
        self.status_label = None
        self.selection_info_label = None
        self.point_size_var = None
        self.point_alpha_var = None
        
        # Filter state
        self.filter_logic_var = None
        
        # Axis variables
        self.x_var = None
        self.y_var = None
        self.color_var = None
        
        # Initialize data
        self._initialize_data()
    
    def _initialize_data(self):
        """Initialize data from available sources"""
        self.logger.info("Initializing AdvancedFilterWindow data...")
        
        # Priority: direct data > images > data_manager
        # Images are preferred because they include classifications and tags
        if self.raw_data is not None:
            self.logger.info(f"  Using direct DataFrame: {len(self.raw_data)} rows")
            self.current_data = self.raw_data.copy()
            
        elif self.images:
            self.logger.info(f"  Building data from {len(self.images)} images")
            self.current_data = self._build_from_images()
            
        elif self.data_manager and self._check_data_manager():
            manager_type = "DataCoordinator" if hasattr(self.data_manager, 'is_initialized') else "DrillholeDataManager"
            self.logger.info(f"  Building data from {manager_type}")
            self.current_data = self._build_from_data_manager()
            
        else:
            self.logger.warning("  No data source provided - creating empty DataFrame")
            self.current_data = pd.DataFrame()
        
        # Set filtered data to current data initially
        self.filtered_data = self.current_data.copy() if self.current_data is not None else pd.DataFrame()
        
        self.logger.info(f"  Final data shape: {len(self.current_data)} rows, {len(self.current_data.columns) if not self.current_data.empty else 0} columns")
    
    def _check_data_manager(self) -> bool:
        """Check if data manager has data loaded (supports DataCoordinator or DrillholeDataManager)"""
        if not self.data_manager:
            return False
        
        # Check for DataCoordinator (new architecture)
        if hasattr(self.data_manager, 'is_initialized'):
            if self.data_manager.is_initialized and self.data_manager.geological_store.is_loaded:
                self.logger.debug("DataCoordinator detected with loaded data")
                return True
            return False
        
        # Check for DrillholeDataManager (deprecated)
        if hasattr(self.data_manager, 'hole_ids'):
            if self.data_manager.hole_ids:
                self.logger.debug(f"DrillholeDataManager detected with {len(self.data_manager.hole_ids)} holes (deprecated)")
                return len(self.data_manager.hole_ids) > 0
        
        return False
    
    def _build_from_data_manager(self) -> pd.DataFrame:
        """Build DataFrame from DataCoordinator or DrillholeDataManager"""
        try:
            # Check for DataCoordinator (new architecture)
            if hasattr(self.data_manager, 'is_initialized') and self.data_manager.is_initialized:
                return self._build_from_data_coordinator()
            
            # Fall back to DrillholeDataManager (deprecated)
            return self._build_from_drillhole_data_manager()
            
        except Exception as e:
            self.logger.error(f"Error building data from manager: {e}")
            return pd.DataFrame()
    
    def _build_from_data_coordinator(self) -> pd.DataFrame:
        """Build DataFrame from DataCoordinator (new architecture)"""
        try:
            geo_store = self.data_manager.geological_store
            
            # Get all unique holes
            unique_holes = geo_store.get_unique_holes()
            if not unique_holes:
                self.logger.warning("DataCoordinator has no holes")
                return pd.DataFrame()
            
            self.logger.info(f"Building DataFrame from DataCoordinator ({len(unique_holes)} holes)...")
            
            # Get all data by iterating through sources
            all_data = []
            sources = geo_store.get_data_sources()
            
            for source_name, source in sources.items():
                if source.is_loaded and source.df is not None:
                    df = source.df.copy()
                    # Add source identifier
                    df['_source'] = source_name
                    all_data.append(df)
            
            if all_data:
                df = pd.concat(all_data, ignore_index=True)
                self.logger.info(f"Built {len(df)} rows from DataCoordinator")
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error building data from DataCoordinator: {e}")
            return pd.DataFrame()
    
    def _build_from_drillhole_data_manager(self) -> pd.DataFrame:
        """Build DataFrame from DrillholeDataManager (deprecated)"""
        try:
            # Get all available holes
            hole_ids = self.data_manager.hole_ids
            if not hole_ids:
                return pd.DataFrame()
            
            self.logger.info(f"Building DataFrame from DrillholeDataManager ({len(hole_ids)} holes, deprecated)...")
            
            # Get data for all holes
            all_data = []
            for hole_id in hole_ids:
                try:
                    hole_data = self.data_manager.get_data_for_hole(hole_id)
                    if hole_data is not None and not hole_data.empty:
                        # Ensure hole_id column exists
                        if 'holeid' not in hole_data.columns and 'hole_id' not in hole_data.columns:
                            hole_data['hole_id'] = hole_id
                        all_data.append(hole_data)
                except Exception as e:
                    self.logger.debug(f"Could not get data for {hole_id}: {e}")
            
            if all_data:
                df = pd.concat(all_data, ignore_index=True)
                self.logger.info(f"Built {len(df)} rows from DrillholeDataManager (deprecated)")
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error building data from DrillholeDataManager: {e}")
            return pd.DataFrame()
    
    def _build_from_images(self) -> pd.DataFrame:
        """Build DataFrame from image list with on-demand csv_data population"""
        data_rows = []
        
        # If we have a data manager, use it to efficiently populate csv_data
        if self.data_manager:
            self._populate_image_csv_data()
        
        for img in self.images:
            # Skip images with no csv data after population attempt
            if not hasattr(img, 'csv_data') or not img.csv_data:
                continue
            
            # Base row with image metadata
            row = {
                'hole_id': img.hole_id,
                'depth_from': img.depth_from,
                'depth_to': img.depth_to,
                'filename': img.filename,
                'image_path': img.image_path,
            }
            
            # Add classification
            if hasattr(img, 'classification'):
                row['classification'] = self._get_classification_text(img.classification)
            
            # Add moisture status
            if hasattr(img, 'moisture_status'):
                row['moisture_status'] = img.moisture_status or ""
            
            # Add tags
            if hasattr(img, 'tags') and img.tags:
                row['tags'] = ','.join(img.tags)
            
            # Add all CSV data
            row.update(img.csv_data)
            
            data_rows.append(row)
        
        if data_rows:
            df = pd.DataFrame(data_rows)
            self.logger.info(f"Built {len(df)} rows from {len(self.images)} images")
            return df
        
        return pd.DataFrame()
    
    def _populate_image_csv_data(self):
        """Populate csv_data for images using DataCoordinator or DrillholeDataManager"""
        from processing.DataManager.keys import ImageKey
        
        images_needing_data = [img for img in self.images 
                              if not hasattr(img, 'csv_data') or not img.csv_data]
        
        if not images_needing_data:
            return
        
        # Check for DataCoordinator (new architecture)
        if hasattr(self.data_manager, 'is_initialized') and self.data_manager.is_initialized:
            self.logger.info(f"Populating csv_data for {len(images_needing_data)} images via DataCoordinator...")
            
            geo_store = self.data_manager.geological_store
            loaded_count = 0
            
            for img in images_needing_data:
                try:
                    key = ImageKey(hole_id=img.hole_id, depth_to=float(img.depth_to))
                    row_data = geo_store.get_row(key)
                    if row_data:
                        img.csv_data = row_data
                        loaded_count += 1
                    else:
                        img.csv_data = {}
                except Exception as e:
                    self.logger.debug(f"CSV lookup failed for {img.hole_id}_{img.depth_to}: {e}")
                    img.csv_data = {}
            
            self.logger.info(f"Loaded CSV data for {loaded_count}/{len(images_needing_data)} images via DataCoordinator")
            return
        
        # Fall back to DrillholeDataManager (deprecated)
        if hasattr(self.data_manager, 'get_data_for_hole'):
            self.logger.info(f"Populating csv_data for {len(images_needing_data)} images via DrillholeDataManager (deprecated)...")
            
            # Build hole cache
            hole_data_cache = {}
            unique_holes = set(img.hole_id for img in images_needing_data)
            
            for hole_id in unique_holes:
                hole_data_cache[hole_id] = self.data_manager.get_data_for_hole(hole_id)
            
            self.logger.info(f"Cached data for {len(unique_holes)} unique holes")
            
            # Populate csv_data for all images
            for img in images_needing_data:
                hole_data = hole_data_cache.get(img.hole_id)
                if hole_data is not None and not hole_data.empty:
                    matching_rows = hole_data[hole_data["to"] == img.depth_to]
                    if not matching_rows.empty:
                        img.csv_data = matching_rows.iloc[0].to_dict()
                    else:
                        img.csv_data = {}
                else:
                    img.csv_data = {}

    def _get_classification_text(self, classification) -> str:
        """Convert classification to text"""
        if isinstance(classification, str):
            return classification
        elif hasattr(classification, "value"):
            return classification.value
        else:
            class_str = str(classification)
            # Remove enum prefix if present
            if "ClassificationCategory." in class_str:
                class_str = class_str.replace("ClassificationCategory.", "")
            return class_str
    
    def show(self):
        """Show the window"""
        if not self.window:
            self._create_window()
        
        self.window.deiconify()
        self.is_open = True
        
        # Position relative to parent
        if hasattr(self.parent, "winfo_x") and self.parent.winfo_exists():
            try:
                x = self.parent.winfo_x() + 50
                y = self.parent.winfo_y() + 50
                self.window.geometry(f"{self.config.window_width}x{self.config.window_height}+{x}+{y}")
            except:
                self.window.geometry(f"{self.config.window_width}x{self.config.window_height}+50+50")
        else:
            self.window.geometry(f"{self.config.window_width}x{self.config.window_height}+50+50")
    
    def hide(self):
        """Hide the window"""
        if self.window:
            self.window.withdraw()
            self.is_open = False
    
    def close(self):
        """Close and destroy the window"""
        if self.window:
            self.is_open = False
            if self.on_close:
                try:
                    self.on_close()
                except Exception as e:
                    self.logger.error(f"Error in on_close callback: {e}")
            self.window.destroy()
            self.window = None
    
    def _create_window(self):
        """Create the main window"""
        self.window = tk.Toplevel(self.parent)
        self.window.title(self.config.window_title)
        self.window.configure(bg=self.theme_colors["background"])
        
        # Apply theme
        if hasattr(self.gui_manager, "configure_ttk_styles"):
            self.gui_manager.configure_ttk_styles(self.window)
        
        # Main container
        main_container = ttk.Frame(self.window)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title bar
        self._create_title_bar(main_container)
        
        # Content area with paned window
        paned_window = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Left panel: Filters
        filter_panel = ttk.Frame(paned_window)
        paned_window.add(filter_panel, weight=1)
        self._create_filter_panel(filter_panel)
        
        # Right panel: Scatter plot
        scatter_panel = ttk.Frame(paned_window)
        paned_window.add(scatter_panel, weight=3)
        self._create_scatter_panel(scatter_panel)
        
        # Status bar
        self._create_status_bar(main_container)
        
        # Handle window close
        self.window.protocol("WM_DELETE_WINDOW", self.hide)
        
        # Initially hidden
        self.window.withdraw()
    
    def _create_title_bar(self, parent):
        """Create title bar with instructions"""
        title_frame = ttk.Frame(parent)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(
            title_frame,
            text="Dataset Filter & Exploration",
            font=("Arial", 14, "bold"),
            style="Content.TLabel"
        )
        title_label.pack(side=tk.LEFT)
        
        instructions = ttk.Label(
            title_frame,
            text="Filter geological data, then use lasso to select intervals",
            font=("Arial", 9, "italic"),
            style="Content.TLabel"
        )
        instructions.pack(side=tk.RIGHT, padx=(10, 0))

    def _create_filter_panel(self, parent):
        """Create filter panel with dynamic filter rows"""
        # Title
        title_label = ttk.Label(
            parent,
            text="Data Filters",
            font=("Arial", 11, "bold"),
            style="Content.TLabel"
        )
        title_label.pack(fill=tk.X, pady=(0, 5))

        # Instructions
        instructions = ttk.Label(
            parent,
            text="Add filters to narrow the data plotted.\nUse lasso tool to select points.",
            style="Content.TLabel",
            wraplength=350,
            justify=tk.LEFT
        )
        instructions.pack(fill=tk.X, pady=(0, 10))

        # Filter rows container with scrolling
        filter_container_frame = ttk.LabelFrame(parent, text="Filters", padding=5)
        filter_container_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Canvas for scrolling filters
        filter_canvas = tk.Canvas(
            filter_container_frame,
            bg=self.theme_colors.get("secondary_bg", "#252526"),
            highlightthickness=0,
            height=200
        )
        filter_scrollbar = ttk.Scrollbar(filter_container_frame, orient="vertical", command=filter_canvas.yview)
        self.filter_container = ttk.Frame(filter_canvas)

        self.filter_container.bind(
            "<Configure>",
            lambda e: filter_canvas.configure(scrollregion=filter_canvas.bbox("all"))
        )

        filter_canvas.create_window((0, 0), window=self.filter_container, anchor="nw")
        filter_canvas.configure(yscrollcommand=filter_scrollbar.set)

        filter_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        filter_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Store canvas reference
        self.filter_canvas = filter_canvas

        # Filter control buttons
        filter_btn_frame = ttk.Frame(parent)
        filter_btn_frame.pack(fill=tk.X, pady=5)

        add_filter_btn = ModernButton(
            filter_btn_frame,
            text="➕ Add Filter",
            command=self._add_filter_row,
            color=self.theme_colors.get("accent_green", "#4a8259"),
            theme_colors=self.theme_colors,
        )
        add_filter_btn.pack(side=tk.LEFT, padx=2)

        clear_filters_btn = ModernButton(
            filter_btn_frame,
            text="Clear All",
            command=self._clear_all_filters,
            color=self.theme_colors.get("accent_orange", "#ff9800"),
            theme_colors=self.theme_colors,
        )
        clear_filters_btn.pack(side=tk.LEFT, padx=2)

        apply_filters_btn = ModernButton(
            filter_btn_frame,
            text="Apply Filters",
            command=self._apply_all_filters,
            color=self.theme_colors.get("accent_blue", "#3a7ca5"),
            theme_colors=self.theme_colors,
        )
        apply_filters_btn.pack(side=tk.RIGHT, padx=2)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Filter summary label
        self.filter_summary_label = ttk.Label(
            parent,
            text="No filters active",
            style="Content.TLabel",
            wraplength=350
        )
        self.filter_summary_label.pack(fill=tk.X, pady=5)

        # Initialize filter rows list
        self.filter_rows = []

        # Set plot mode to always use filtered data (no toggle needed)
        self.plot_mode_var = tk.StringVar(value="filtered")
        self.plot_mode = "filtered"

    def _create_scatter_panel(self, parent):
        """Create scatter plot panel"""
        # Axis controls
        control_frame = ttk.LabelFrame(parent, text="Scatter Plot Axes", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Get available columns
        numeric_cols = self._get_numeric_columns()
        all_cols = self._get_all_columns()
        categorical_cols = self._get_categorical_columns()
        
        # X Axis
        ttk.Label(control_frame, text="X-Axis:", style="Content.TLabel").grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )
        self.x_var = tk.StringVar(value=self._get_default_x_axis(numeric_cols))
        x_menu = self.gui_manager.create_searchable_optionmenu(
            parent=control_frame,
            items=numeric_cols,
            variable=self.x_var,
            width=30,
            placeholder="Select X-Axis...",
            on_change=lambda val: self._update_scatter()
        )
        x_menu.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        # Y Axis
        ttk.Label(control_frame, text="Y-Axis:", style="Content.TLabel").grid(
            row=0, column=2, padx=5, pady=5, sticky="w"
        )
        self.y_var = tk.StringVar(value=self._get_default_y_axis(numeric_cols))
        y_menu = self.gui_manager.create_searchable_optionmenu(
            parent=control_frame,
            items=numeric_cols,
            variable=self.y_var,
            width=30,
            placeholder="Select Y-Axis...",
            on_change=lambda val: self._update_scatter()
        )
        y_menu.grid(row=0, column=3, padx=5, pady=5, sticky="ew")
        
        # Color By
        ttk.Label(control_frame, text="Color By:", style="Content.TLabel").grid(
            row=1, column=0, padx=5, pady=5, sticky="w"
        )
        self.color_var = tk.StringVar(value=self._get_default_color_by(all_cols))
        color_menu = self.gui_manager.create_searchable_optionmenu(
            parent=control_frame,
            items=all_cols,  # Allow any column for coloring
            variable=self.color_var,
            width=30,
            placeholder="Select color grouping...",
            on_change=lambda val: self._update_scatter()
        )
        color_menu.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        # Plot Type selector (Scatter or Ternary)
        ttk.Label(control_frame, text="Plot Type:", style="Content.TLabel").grid(
            row=2, column=0, padx=5, pady=5, sticky="w"
        )
        self.plot_type_var = tk.StringVar(value="scatter")
        plot_type_menu = self.gui_manager.create_searchable_optionmenu(
            parent=control_frame,
            items=["scatter", "ternary"],
            variable=self.plot_type_var,
            width=30,
            placeholder="Select plot type...",
            on_change=lambda val: self._on_plot_type_change()
        )
        plot_type_menu.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        
        # Z-Axis (for ternary plots - initially hidden)
        self.z_label = ttk.Label(control_frame, text="Z-Axis:", style="Content.TLabel")
        self.z_var = tk.StringVar(value=numeric_cols[2] if len(numeric_cols) > 2 else "")
        self.z_menu = self.gui_manager.create_searchable_optionmenu(
            parent=control_frame,
            items=numeric_cols,
            variable=self.z_var,
            width=30,
            placeholder="Select Z-Axis...",
            on_change=lambda val: self._update_scatter()
        )
        # Don't grid them yet - only show when ternary is selected
        
        # Plot customization

        
        
        # Plot customization
        ttk.Label(control_frame, text="Point Size:", style="Content.TLabel").grid(
            row=1, column=2, padx=5, pady=5, sticky="w"
        )
        self.point_size_var = tk.IntVar(value=self.config.default_point_size)
        point_size_spin = ttk.Spinbox(
            control_frame,
            from_=5,
            to=100,
            textvariable=self.point_size_var,
            width=10,
            command=self._update_scatter
        )
        point_size_spin.grid(row=1, column=3, padx=5, pady=5, sticky="w")
        
        # Update button
        update_btn = ModernButton(
            control_frame,
            text="🔄 Update Plot",
            command=self._update_scatter,
            color=self.theme_colors["accent_blue"],
            theme_colors=self.theme_colors,
        )
        update_btn.grid(row=0, column=4, rowspan=2, padx=10, pady=5, sticky="nsew")
        
        # Manage color maps button (if color_map_manager available)
        if self.color_map_manager:
            manage_colors_btn = ModernButton(
                control_frame,
                text="🎨 Manage Colors",
                command=self._open_color_map_manager,
                color=self.theme_colors.get("accent_green", "#4a8259"),
                theme_colors=self.theme_colors,
            )
            manage_colors_btn.grid(row=0, column=5, rowspan=2, padx=10, pady=5, sticky="nsew")
        
        control_frame.columnconfigure(1, weight=1)
        control_frame.columnconfigure(3, weight=1)

        # Scatter plot container (filter selectable is handled within the scatter widget itself)
        self.scatter_container = ttk.Frame(parent)
        self.scatter_container.pack(fill=tk.BOTH, expand=True)
        
        # Initial plot
        self._update_scatter()
    
    def _create_status_bar(self, parent):
        """Create status bar with info labels"""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Main status label
        self.status_label = ttk.Label(
            status_frame,
            text="Ready - Add filters or draw lasso to select data",
            relief=tk.SUNKEN,
            style="Content.TLabel"
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        # Selection info label
        self.selection_info_label = ttk.Label(
            status_frame,
            text="0 selected",
            relief=tk.SUNKEN,
            style="Content.TLabel",
            width=20
        )
        self.selection_info_label.pack(side=tk.RIGHT)
    
    def _get_numeric_columns(self) -> List[str]:
        """Get list of numeric columns from current data"""
        if self.current_data is None or self.current_data.empty:
            return []
        
        numeric_cols = []
        for col in self.current_data.columns:
            if pd.api.types.is_numeric_dtype(self.current_data[col]):
                numeric_cols.append(col)
        
        return sorted(numeric_cols)
    
    def _get_all_columns(self) -> List[str]:
        """Get all column names including register metadata"""
        if self.current_data is None or self.current_data.empty:
            return []
        
        columns = self.current_data.columns.tolist()
        
        # Ensure register columns are included prominently
        register_columns = [
            'consensus_classification', 'agreement', 'combined_hex', 'wet_hex', 'dry_hex'
        ]
        
        # Add register columns that exist in data
        for col in register_columns:
            if col in columns and col not in columns[:len(register_columns)]:
                # Move to front if not already there
                columns.remove(col)
                columns.insert(0, col)
        
        return columns
    
    def _get_categorical_columns(self) -> List[str]:
        """Get categorical/string columns"""
        if self.current_data is None or self.current_data.empty:
            return []
        
        cat_cols = []
        for col in self.current_data.columns:
            if pd.api.types.is_string_dtype(self.current_data[col]) or \
               pd.api.types.is_categorical_dtype(self.current_data[col]) or \
               self.current_data[col].dtype == 'object':
                cat_cols.append(col)
        
        return sorted(cat_cols)
    
    def _get_default_x_axis(self, numeric_cols: List[str]) -> str:
        """Get default X axis column - prioritizes Fe columns for geological data"""
        self.logger.debug(f"_get_default_x_axis: {len(numeric_cols)} numeric columns available")
        
        # Try configured default first
        if self.config.default_x_axis in numeric_cols:
            self.logger.debug(f"Using configured default X axis: {self.config.default_x_axis}")
            return self.config.default_x_axis
        
        # Build lowercase lookup for case-insensitive matching
        col_lower_map = {col.lower(): col for col in numeric_cols}
        
        # Try common Fe columns (case-insensitive) - geological standard
        fe_patterns = ['fe_pct_best', 'fe_pct', 'fe%', 'fe', 'fepct', 'fe_best']
        for pattern in fe_patterns:
            if pattern in col_lower_map:
                result = col_lower_map[pattern]
                self.logger.debug(f"Found Fe column for X axis: {result} (matched pattern: {pattern})")
                return result
        
        # Also try partial matching for Fe columns
        for col_lower, col in col_lower_map.items():
            if 'fe' in col_lower and 'pct' in col_lower:
                self.logger.debug(f"Found Fe percentage column for X axis: {col}")
                return col
        
        # Try depth columns as fallback
        for col in ['from', 'depth_from', 'sampfrom', 'geolfrom']:
            if col in numeric_cols:
                self.logger.debug(f"Using depth column for X axis: {col}")
                return col
        
        # Fallback to first numeric column
        result = numeric_cols[0] if numeric_cols else ""
        self.logger.debug(f"Fallback X axis: {result}")
        return result
    
    def _get_default_y_axis(self, numeric_cols: List[str]) -> str:
        """Get default Y axis column - prioritizes SiO2 columns for geological data"""
        self.logger.debug(f"_get_default_y_axis: {len(numeric_cols)} numeric columns available")
        
        # Try configured default first
        if self.config.default_y_axis in numeric_cols:
            self.logger.debug(f"Using configured default Y axis: {self.config.default_y_axis}")
            return self.config.default_y_axis
        
        # Build lowercase lookup for case-insensitive matching
        col_lower_map = {col.lower(): col for col in numeric_cols}
        
        # Try common SiO2 columns (case-insensitive) - geological standard
        sio2_patterns = ['sio2_pct_best', 'sio2_pct', 'sio2%', 'sio2', 'sio2pct', 'sio2_best']
        for pattern in sio2_patterns:
            if pattern in col_lower_map:
                result = col_lower_map[pattern]
                self.logger.debug(f"Found SiO2 column for Y axis: {result} (matched pattern: {pattern})")
                return result
        
        # Also try partial matching for SiO2 columns
        for col_lower, col in col_lower_map.items():
            if 'sio2' in col_lower and 'pct' in col_lower:
                self.logger.debug(f"Found SiO2 percentage column for Y axis: {col}")
                return col
        
        # Try depth columns as fallback
        for col in ['to', 'depth_to', 'sampto', 'geolto']:
            if col in numeric_cols:
                self.logger.debug(f"Using depth column for Y axis: {col}")
                return col
        
        # Fallback to second numeric column (or first if only one)
        if len(numeric_cols) > 1:
            result = numeric_cols[1]
        elif numeric_cols:
            result = numeric_cols[0]
        else:
            result = ""
        self.logger.debug(f"Fallback Y axis: {result}")
        return result
    
    def _get_default_color_by(self, all_cols: List[str]) -> str:
        """Get default color-by column - prefers hex color columns for geological data"""
        self.logger.debug(f"_get_default_color_by: {len(all_cols)} columns available")
        
        # Build lowercase lookup for case-insensitive matching
        col_lower_map = {col.lower(): col for col in all_cols}
        
        # Priority 1: Hex color columns (best for geological visualization)
        hex_preferred = ['combined_hex', 'hex_color', 'wet_hex', 'dry_hex']
        for pref in hex_preferred:
            if pref in col_lower_map:
                result = col_lower_map[pref]
                self.logger.debug(f"Found hex color column for color-by: {result}")
                return result
        
        # Priority 2: Try configured default
        if self.config.default_color_by in all_cols:
            self.logger.debug(f"Using configured default color-by: {self.config.default_color_by}")
            return self.config.default_color_by
        
        # Priority 3: Classification columns
        classification_preferred = ['classification', 'consensus_classification', 'lithology', 'rock_type']
        for pref in classification_preferred:
            if pref in col_lower_map:
                result = col_lower_map[pref]
                self.logger.debug(f"Found classification column for color-by: {result}")
                return result
        
        # Priority 4: Moisture status
        if 'moisture_status' in col_lower_map:
            result = col_lower_map['moisture_status']
            self.logger.debug(f"Found moisture_status for color-by: {result}")
            return result
        
        # Fallback to first column
        result = all_cols[0] if all_cols else ""
        self.logger.debug(f"Fallback color-by: {result}")
        return result

    def _open_color_map_manager(self):
        """Open color map editor dialog"""
        if not self.color_map_manager:
            messagebox.showinfo(
                "Not Available",
                "Color map manager is not available."
            )
            return
        
        from gui.color_map_editor_dialog import ColorMapEditorDialog
        
        # Get current color column
        color_col = self.color_var.get() if self.color_var else None
        
        # Get sample data for preview
        data_values = []
        if color_col and color_col in self.current_data.columns:
            data_values = self.current_data[color_col].dropna().unique().tolist()[:1000]
        
        self.logger.info(f"Opening color map editor for column: {color_col}")
        
        # Create editor dialog
        dialog = ColorMapEditorDialog(
            parent=self.window,
            gui_manager=self.gui_manager,
            color_map_manager=self.color_map_manager,
            data_column=color_col,
            data_values=data_values,
            initial_color_map=None
        )
        
        # Show modal dialog
        result = dialog.show()
        
        if result:
            # Refresh the plot with updated color map
            self.logger.info("Color map updated, refreshing plot")
            self._update_scatter()

    def _on_plot_mode_change(self):
        """Handle plot mode toggle - now always uses filtered data"""
        self.plot_mode = "filtered"
        self._update_scatter()

    def _add_filter_row(self):
        """Add a new dynamic filter row"""
        if self.current_data is None or self.current_data.empty:
            return

        # Build columns_info dict from current data
        columns_info = {}
        for col in self.current_data.columns:
            # Detect column type
            if pd.api.types.is_numeric_dtype(self.current_data[col]):
                col_type = "numeric"
            else:
                col_type = "text"
            columns_info[col] = {"type": col_type}

        # Get filter row index
        row_index = len(self.filter_rows)

        # Create filter row with correct interface
        filter_row = DynamicFilterRow(
            parent=self.filter_container,
            gui_manager=self.gui_manager,
            columns_info=columns_info,
            register_data=self.current_data,
            on_remove_callback=lambda idx=row_index: self._remove_filter_row_by_index(idx),
            index=row_index,
            on_column_selected_callback=self._on_filter_change,
        )

        self.filter_rows.append(filter_row)
        self.logger.info(f"Added filter row, total: {len(self.filter_rows)}")

    def _remove_filter_row_by_index(self, index: int):
        """Remove a filter row by its index"""
        # Find and remove the filter row
        for i, row in enumerate(self.filter_rows):
            if hasattr(row, 'index') and row.index == index:
                self.filter_rows.pop(i)
                if hasattr(row, 'frame'):
                    row.frame.destroy()
                elif hasattr(row, 'destroy'):
                    row.destroy()
                break

        self.logger.info(f"Removed filter row {index}, remaining: {len(self.filter_rows)}")
        # Re-apply filters after removal
        self._apply_all_filters()

    def _remove_filter_row(self, filter_row):
        """Remove a filter row"""
        if filter_row in self.filter_rows:
            self.filter_rows.remove(filter_row)
            filter_row.destroy()
            self.logger.info(f"Removed filter row, remaining: {len(self.filter_rows)}")
            # Re-apply filters after removal
            self._apply_all_filters()

    def _clear_all_filters(self):
        """Clear all filter rows"""
        for row in self.filter_rows[:]:  # Copy list to avoid modification during iteration
            row.destroy()
        self.filter_rows.clear()

        # Reset to full data
        self.filtered_data = self.current_data.copy() if self.current_data is not None else pd.DataFrame()
        self._update_filter_summary()
        self._update_scatter()
        self.logger.info("Cleared all filters")

    def _on_filter_change(self, *args):
        """Handle filter row value change - auto-apply filters"""
        self._apply_all_filters()

    def _apply_all_filters(self):
        """Apply all active filters to the data"""
        if self.current_data is None or self.current_data.empty:
            return

        # Start with full data
        filtered_df = self.current_data.copy()
        active_filters = []

        for filter_row in self.filter_rows:
            if hasattr(filter_row, 'get_filter_config'):
                try:
                    config = filter_row.get_filter_config()
                    column = config.get("column", "")
                    operator = config.get("operator", "")
                    value = config.get("value", "")
                    value2 = config.get("value2", "")
                    data_type = config.get("data_type", "text")

                    if not column or column not in filtered_df.columns:
                        continue

                    # Build filter description
                    if operator == "between" and value2:
                        description = f"{column} {operator} {value} and {value2}"
                    elif operator in ("is null", "not null"):
                        description = f"{column} {operator}"
                    else:
                        description = f"{column} {operator} {value}"
                    active_filters.append(description)

                    # Apply filter based on operator
                    col_data = filtered_df[column]

                    if operator == "is null":
                        mask = col_data.isna()
                    elif operator == "not null":
                        mask = col_data.notna()
                    elif data_type == "numeric":
                        mask = self._apply_numeric_mask(col_data, operator, value, value2)
                    else:
                        mask = self._apply_text_mask(col_data, operator, value)

                    filtered_df = filtered_df[mask]

                except Exception as e:
                    self.logger.error(f"Error applying filter: {e}")

        self.filtered_data = filtered_df
        self._update_filter_summary(active_filters)
        self._update_scatter()

        self.logger.info(f"Applied {len(active_filters)} filters: {len(self.current_data)} -> {len(filtered_df)} rows")

        # Callback for external integration
        if self.on_filter_change:
            try:
                self.on_filter_change(filtered_df)
            except Exception as e:
                self.logger.error(f"Error in on_filter_change callback: {e}")

    def _apply_numeric_mask(self, col_data: pd.Series, operator: str, value: str, value2: str = None) -> pd.Series:
        """Apply numeric filter and return boolean mask"""
        try:
            filter_val = float(value) if value else 0
            numeric_col = pd.to_numeric(col_data, errors='coerce')

            if operator == "=":
                return numeric_col == filter_val
            elif operator == "≠":
                return numeric_col != filter_val
            elif operator == "<":
                return numeric_col < filter_val
            elif operator == "≤":
                return numeric_col <= filter_val
            elif operator == ">":
                return numeric_col > filter_val
            elif operator == "≥":
                return numeric_col >= filter_val
            elif operator == "between" and value2:
                filter_val2 = float(value2)
                return (numeric_col >= filter_val) & (numeric_col <= filter_val2)
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Numeric filter error: {e}")

        return pd.Series([True] * len(col_data), index=col_data.index)

    def _apply_text_mask(self, col_data: pd.Series, operator: str, value: str) -> pd.Series:
        """Apply text filter and return boolean mask"""
        str_col = col_data.astype(str).str.lower()
        filter_val = str(value).lower() if value else ""

        if operator == "equals":
            return str_col == filter_val
        elif operator == "not equals":
            return str_col != filter_val
        elif operator == "contains":
            return str_col.str.contains(filter_val, na=False)
        elif operator == "not contains":
            return ~str_col.str.contains(filter_val, na=False)
        elif operator == "starts with":
            return str_col.str.startswith(filter_val, na=False)
        elif operator == "ends with":
            return str_col.str.endswith(filter_val, na=False)
        elif operator == "in":
            values = [v.strip().lower() for v in value.split(",")]
            return str_col.isin(values)
        elif operator == "not in":
            values = [v.strip().lower() for v in value.split(",")]
            return ~str_col.isin(values)

        return pd.Series([True] * len(col_data), index=col_data.index)

    def _update_filter_summary(self, active_filters: List[str] = None):
        """Update the filter summary label"""
        if not hasattr(self, 'filter_summary_label') or not self.filter_summary_label:
            return

        if active_filters and len(active_filters) > 0:
            summary = f"{len(active_filters)} filter(s) active\n"
            summary += f"Showing {len(self.filtered_data)} of {len(self.current_data)} rows"
            self.filter_summary_label.config(text=summary)
        else:
            total = len(self.current_data) if self.current_data is not None else 0
            self.filter_summary_label.config(text=f"No filters active ({total} rows)")
    
    def get_active_filter_descriptions(self) -> List[str]:
        """
        Get list of active filter descriptions.
        
        Note: Filter management moved to main dialog.
        This window now only handles scatter plot visualization.
        
        Returns:
            Empty list (filters managed in main dialog)
        """
        return []

    def _update_scatter(self, *args):
        """Update scatter plot with current settings"""
        self.logger.debug("=== _update_scatter called ===")
        
        x_col = self.x_var.get() if self.x_var else None
        y_col = self.y_var.get() if self.y_var else None
        color_col = self.color_var.get() if self.color_var else None
        
        self.logger.debug(f"Axis selection: X={x_col}, Y={y_col}, Color={color_col}")
        
        if not x_col or not y_col:
            self.logger.warning("Missing axis selection - cannot create plot")
            return
        
        # Choose dataset based on plot mode
        if self.plot_mode == "filtered":
            df = self.filtered_data
            self.logger.debug(f"Using filtered data: {len(df) if df is not None else 0} rows")
        else:
            df = self.current_data
            self.logger.debug(f"Using full data: {len(df) if df is not None else 0} rows")
        
        if df is None or df.empty:
            self.logger.warning("No data available for plotting")
            if self.status_label:
                self.status_label.config(text="No data to plot")
            return
        
        # Check columns exist
        if x_col not in df.columns or y_col not in df.columns:
            self.logger.warning(f"Columns not found in data: x={x_col} (exists={x_col in df.columns}), y={y_col} (exists={y_col in df.columns})")
            self.logger.debug(f"Available columns: {list(df.columns)[:20]}...")
            return
        
        # Filter for non-null values in x and y
        # CRITICAL: Always include hole_id and depth_to for selection matching!
        key_columns = []
        for col in df.columns:
            if col.lower() in ('hole_id', 'holeid'):
                key_columns.append(col)
            elif col.lower() in ('depth_to', 'to', 'sampto', 'geolto'):
                key_columns.append(col)
        
        self.logger.debug(f"Key columns for selection matching: {key_columns}")
        
        # Build column list: key columns + x + y + optional color
        columns_to_keep = list(set(key_columns + [x_col, y_col]))
        if color_col and color_col in df.columns and color_col not in columns_to_keep:
            columns_to_keep.append(color_col)
            self.logger.debug(f"Color column '{color_col}' found and added")
        elif color_col and color_col in columns_to_keep:
            self.logger.debug(f"Color column '{color_col}' already in columns_to_keep")
        else:
            self.logger.debug(f"Color column '{color_col}' not found or not specified - no coloring")
            color_col = None  # Don't use color if not available
        
        # Only keep columns that exist
        columns_to_keep = [c for c in columns_to_keep if c in df.columns]
        self.logger.debug(f"Columns to keep for plot: {columns_to_keep}")
        
        plot_df = df[columns_to_keep].dropna(subset=[x_col, y_col])
        
        if key_columns:
            self.logger.debug(f"Scatter plot includes key columns: {key_columns}")
        else:
            self.logger.warning("No hole_id/depth_to columns found - selection matching will fail!")
        
        if plot_df.empty:
            self.logger.warning(f"No valid (non-null) data for {x_col} vs {y_col}")
            if self.status_label:
                self.status_label.config(text=f"No valid data for {x_col} vs {y_col}")
            return
        
        self.logger.debug(f"Plot DataFrame: {len(plot_df)} rows after dropping nulls")
        
        # Build color mapping
        self.logger.debug(f"Building color map for column: {color_col}")
        color_map = self._build_color_map(plot_df, color_col)
        if color_map:
            if isinstance(color_map, dict):
                self.logger.debug(f"Color map built as dict with {len(color_map)} entries")
            else:
                self.logger.debug(f"Color map built as ColorMap object: {type(color_map).__name__}")
        else:
            self.logger.debug("No color map built - will use default coloring")
        
        # Get point size
        point_size = self.point_size_var.get() if self.point_size_var else self.config.default_point_size
        self.logger.debug(f"Point size: {point_size}")
        
        # Destroy old scatter widget
        if self.scatter_widget:
            self.logger.debug("Destroying existing scatter widget")
            self.scatter_widget.destroy()
        
        # Determine plot type
        plot_type = self.plot_type_var.get() if hasattr(self, 'plot_type_var') else "scatter"
        self.logger.debug(f"Plot type selected: {plot_type}")
        
        if plot_type == "ternary":
            # Get Z-axis column for ternary
            z_col = self.z_var.get() if hasattr(self, 'z_var') else None
            self.logger.debug(f"Ternary mode: z_col={z_col}")
            
            if not z_col or z_col not in df.columns:
                self.logger.warning(f"Ternary plot requires valid Z-axis column (got: {z_col})")
                if self.status_label:
                    self.status_label.config(text=f"Select a valid Z-Axis column for ternary plot")
                return
            
            # Ensure z_col is in plot_df
            if z_col not in plot_df.columns:
                # Re-build plot_df with z_col included
                columns_with_z = list(set(columns_to_keep + [z_col]))
                plot_df = df[columns_with_z].dropna(subset=[x_col, y_col, z_col])
                self.logger.debug(f"Rebuilt plot_df with z_col: {len(plot_df)} rows")
            
            if plot_df.empty:
                self.logger.warning("No valid data after dropping nulls for ternary")
                if self.status_label:
                    self.status_label.config(text=f"No valid data for ternary: {x_col}, {y_col}, {z_col}")
                return
            
            # Import and create TernarySelectionWidget
            try:
                from gui.widgets.ternary_selection_widget import TernarySelectionWidget
                
                self.logger.info(f"Creating TernarySelectionWidget: {len(plot_df)} points")
                self.logger.debug(f"  Axes: bottom={x_col}, right={y_col}, left={z_col}")
                self.logger.debug(f"  Color by: {color_col}")
                
                self.scatter_widget = TernarySelectionWidget(
                    parent=self.scatter_container,
                    gui_manager=self.gui_manager,
                    data=plot_df,
                    x_col=x_col,      # Bottom axis
                    y_col=y_col,      # Right axis
                    z_col=z_col,      # Left axis
                    color_by=color_col,
                    color_map=color_map,
                    point_size=point_size,
                    point_alpha=self.config.default_point_alpha,
                    on_selection=self._handle_selection,
                    lasso_color=self.config.lasso_color,
                    lasso_alpha=self.config.lasso_alpha,
                    selection_color=self.config.selection_outline_color,
                    selection_linewidth=self.config.selection_outline_width,
                )
                self.scatter_widget.pack(fill=tk.BOTH, expand=True)
                
                status_msg = f"Ternary: {len(plot_df)} points - {x_col} / {y_col} / {z_col}"
                if color_col:
                    status_msg += f", colored by {color_col}"
                
                self.logger.info(status_msg)
                if self.status_label:
                    self.status_label.config(text=status_msg)
                return
                
            except ImportError as e:
                self.logger.error(f"Could not import TernarySelectionWidget: {e}")
                self.logger.warning("Falling back to scatter plot")
                # Fall through to scatter plot
            except Exception as e:
                self.logger.error(f"Error creating ternary plot: {e}", exc_info=True)
                if self.status_label:
                    self.status_label.config(text=f"Ternary plot error: {e}")
                return
        
        # Create scatter widget (default or fallback from ternary)
        self.logger.info(f"Creating ScatterSelectionWidget: {len(plot_df)} points, x={x_col}, y={y_col}, color_by={color_col}")
        
        self.scatter_widget = ScatterSelectionWidget(
            parent=self.scatter_container,
            gui_manager=self.gui_manager,
            data=plot_df,
            x_col=x_col,
            y_col=y_col,
            color_by=color_col,
            color_map=color_map,
            point_size=point_size,
            point_alpha=self.config.default_point_alpha,
            on_selection=self._handle_selection,
            lasso_color=self.config.lasso_color,
            lasso_alpha=self.config.lasso_alpha,
            selection_color=self.config.selection_outline_color,
            selection_linewidth=self.config.selection_outline_width,
        )
        self.scatter_widget.pack(fill=tk.BOTH, expand=True)
        
        status_msg = f"Plotting {len(plot_df)} points: {x_col} vs {y_col}"
        if color_col:
            status_msg += f", colored by {color_col}"
        
        self.logger.info(status_msg)
        if self.status_label:
            self.status_label.config(text=status_msg)

    def _on_plot_type_change(self):
        """Handle plot type change between scatter and ternary"""
        plot_type = self.plot_type_var.get()
        self.logger.debug(f"Plot type changed to: {plot_type}")
        
        if plot_type == "ternary":
            # Show Z-axis selector for ternary
            self.z_label.grid(row=2, column=2, padx=5, pady=5, sticky="w")
            self.z_menu.grid(row=2, column=3, padx=5, pady=5, sticky="ew")
            self.logger.info("Ternary mode selected - Z-axis selector shown")
            # Note: Actual ternary plotting not yet implemented
        else:
            # Hide Z-axis selector for scatter
            self.z_label.grid_forget()
            self.z_menu.grid_forget()
            self.logger.debug("Scatter mode selected - Z-axis selector hidden")
        
        # Update the plot
        self._update_scatter()

    def _build_color_map(self, df: pd.DataFrame, color_col: str):
        """
        Build color mapping using ColorMapManager infrastructure.
        Returns either a ColorMap object or a dict, both work with ScatterSelectionWidget.
        """
        if not color_col or color_col not in df.columns:
            return None
        
        # Special handling for hex color columns - values ARE the colors
        # Supports: hex_color, combined_hex, wet_hex, dry_hex
        hex_column_names = {'hex_color', 'combined_hex', 'wet_hex', 'dry_hex'}
        if color_col in hex_column_names or color_col.lower() in hex_column_names:
            # DEBUG: Log raw column state before filtering
            self.logger.debug(f"Hex column '{color_col}' raw stats: dtype={df[color_col].dtype}, "
                            f"total={len(df[color_col])}, null={df[color_col].isna().sum()}, "
                            f"sample={df[color_col].head(5).tolist()}")
            
            # Filter out empty/null hex colors for unique count
            valid_hex = df[color_col].dropna()
            # Convert to string and strip whitespace to handle type mismatches
            valid_hex = valid_hex.astype(str).str.strip()
            valid_hex = valid_hex[valid_hex != '']
            valid_hex = valid_hex[valid_hex.str.lower() != 'nan']  # Filter string 'nan' from astype conversion
            unique_values = valid_hex.unique()
            
            color_dict = {}
            
            for val in unique_values:
                # Values in hex_color column are already hex color strings
                if isinstance(val, str) and val:
                    # Ensure it starts with #
                    hex_str = val if val.startswith('#') else '#' + val
                    # Validate it's a proper hex color (6 or 8 characters after #)
                    if len(hex_str) in [7, 9]:
                        color_dict[val] = hex_str
                    else:
                        color_dict[val] = '#808080'  # Gray for invalid
                else:
                    color_dict[val] = '#808080'  # Gray for empty
            
            # Add empty string mapping for null values
            color_dict[''] = '#808080'
            
            self.logger.info(f"Using hex_color values directly as colors ({len(unique_values)} unique colors, {len(color_dict)} total with fallbacks)")
            return color_dict
        
        # First, try to get a ColorMap preset from ColorMapManager
        if self.color_map_manager:
            # Try exact match first
            preset_name = color_col.lower().replace(" ", "_").replace("_", "")
            color_map_obj = self.color_map_manager.get_preset(preset_name)
            
            if color_map_obj:
                self.logger.info(f"Using ColorMap preset '{preset_name}' for column '{color_col}'")
                return color_map_obj
            
            # Try common name mappings
            name_mappings = {
                'classification': 'lithology',  # Often same categories
                'lithology': 'lithology',
                'rock_type': 'lithology',
                'moisture_status': None,  # No preset, will use item_manager
            }
            
            mapped_name = name_mappings.get(color_col.lower())
            if mapped_name:
                color_map_obj = self.color_map_manager.get_preset(mapped_name)
                if color_map_obj:
                    self.logger.info(f"Using mapped ColorMap preset '{mapped_name}' for column '{color_col}'")
                    return color_map_obj
        
        # Second, try item_manager for classifications/tags
        if self.item_manager and color_col in ['classification', 'lithology', 'rock_type']:
            unique_values = df[color_col].dropna().unique()
            color_dict = {}
            
            for val in unique_values:
                color = self.item_manager.get_color_for_item(str(val))
                if color:
                    color_dict[val] = color
            
            if color_dict:
                self.logger.info(f"Using ItemManager colors for '{color_col}'")
                return color_dict
        
        # Fallback: return None, let ScatterSelectionWidget use its default colormap
        self.logger.warning(f"No color map found for '{color_col}', using default visualization")
        return None

    def _handle_selection(self, selected_df: pd.DataFrame):
        """Handle lasso selection"""
        self.logger.info(f"Lasso selection: {len(selected_df)} points selected")
        
        # Update selection info
        if self.selection_info_label:
            self.selection_info_label.config(text=f"{len(selected_df)} selected")
        
        # Call user callback
        if self.on_selection:
            try:
                self.on_selection(selected_df)
            except Exception as e:
                self.logger.error(f"Error in on_selection callback: {e}")
    
    def _clear_selection(self):
        """Clear current selection"""
        if self.scatter_widget and hasattr(self.scatter_widget, 'clear_selection'):
            self.scatter_widget.clear_selection()
        
        if self.selection_info_label:
            self.selection_info_label.config(text="0 selected")
    
    def refresh_data(self, new_data: Optional[pd.DataFrame] = None):
        """Refresh window with new data"""
        if new_data is not None:
            self.raw_data = new_data
            self.current_data = new_data.copy()
            self.filtered_data = new_data.copy()
        
        # Reapply filters if any
        if self.filter_rows:
            self._apply_all_filters()
        else:
            self._update_scatter()
    
    def set_filtered_data(self, filtered_df: pd.DataFrame):
        """
        Set filtered data from external source (e.g., main dialog's filter rows).
        
        Args:
            filtered_df: DataFrame after applying external filters
        """
        self.filtered_data = filtered_df.copy() if filtered_df is not None else self.current_data.copy()
        self.logger.info(f"External filtered data set: {len(self.filtered_data)} rows")
        
        # Update plot if in filtered mode
        if self.plot_mode == "filtered":
            self._update_scatter()
    
    def get_current_data(self) -> pd.DataFrame:
        """Get currently displayed/filtered data"""
        if self.plot_mode == "filtered":
            return self.filtered_data.copy() if self.filtered_data is not None else pd.DataFrame()
        else:
            return self.current_data.copy() if self.current_data is not None else pd.DataFrame()
    
    def get_selected_data(self) -> pd.DataFrame:
        """Get currently selected data from lasso"""
        if self.scatter_widget and hasattr(self.scatter_widget, 'get_selected_data'):
            return self.scatter_widget.get_selected_data()
        return pd.DataFrame()
