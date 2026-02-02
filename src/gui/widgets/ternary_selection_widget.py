"""
TernarySelectionWidget - Interactive ternary diagram with lasso selection.

Provides a ternary (triangular) plot for visualizing three-component compositional data
with interactive lasso selection for filtering geological data.

Key Features:
- Renders ternary diagram using python-ternary library
- Supports lasso/polygon selection in ternary coordinate space
- Color-by categorical or numeric columns using ColorMap
- Selection callback for integration with filter workflows
- Theming via GUIManager

Coordinate System:
- Ternary coordinates (a, b, c) where a + b + c = scale (default 100)
- X-axis (bottom): Component A
- Y-axis (right): Component B  
- Z-axis (left): Component C

Usage:
    >>> widget = TernarySelectionWidget(
    ...     parent=frame,
    ...     gui_manager=gui_manager,
    ...     data=df,
    ...     x_col="Fe_pct",      # Bottom axis
    ...     y_col="SiO2_pct",    # Right axis
    ...     z_col="Al2O3_pct",   # Left axis
    ...     color_by="classification",
    ...     color_map=lithology_color_map,
    ...     on_selection=handle_selection,
    ... )

Author: George Symonds
"""

import tkinter as tk
from tkinter import ttk
import logging
from typing import Optional, Callable, List, Dict, Any, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from matplotlib.patches import Rectangle

from gui.gui_manager import GUIManager

# Try to import ColorMap types - handle if not available
try:
    from processing.LoggingReviewStep.color_map_manager import ColorMap, ColorMapType
except ImportError:
    ColorMap = None
    ColorMapType = None

logger = logging.getLogger(__name__)


class TernarySelectionWidget(tk.Frame):
    """
    Interactive ternary diagram widget with lasso selection support.
    
    Displays three-component compositional data on a ternary (triangular) plot
    with support for:
    - Lasso/polygon selection
    - Color-by categorical or numeric columns
    - Selection callbacks for filter integration
    - Theming via GUIManager
    
    The widget normalizes input data so components sum to 100% before plotting.
    """
    
    # Standard ternary scale (components sum to this value)
    SCALE = 100
    
    def __init__(
        self,
        parent,
        gui_manager: GUIManager,
        data: pd.DataFrame,
        x_col: str,              # Bottom axis (component A)
        y_col: str,              # Right axis (component B)
        z_col: str,              # Left axis (component C)
        color_by: Optional[str] = None,
        color_map: Optional[Union["ColorMap", Dict[str, str]]] = None,
        filter_values: Optional[List[str]] = None,
        on_selection: Optional[Callable[[pd.DataFrame], None]] = None,
        point_size: float = 20.0,
        point_alpha: float = 0.6,
        accumulate_mode: bool = False,
        lasso_color: str = "#00ff00",
        lasso_alpha: float = 0.3,
        selection_color: str = "#ffff00",
        selection_linewidth: int = 2,
        normalize: bool = True,
        show_gridlines: bool = True,
        **kwargs
    ):
        """
        Initialize the ternary selection widget.
        
        Args:
            parent: Parent tkinter widget
            gui_manager: GUIManager for theming
            data: DataFrame containing the data to plot
            x_col: Column name for bottom axis (component A)
            y_col: Column name for right axis (component B)
            z_col: Column name for left axis (component C)
            color_by: Optional column for point coloring
            color_map: ColorMap object or dict for color mapping
            filter_values: Optional list of values for filter dropdown
            on_selection: Callback function receiving selected DataFrame
            point_size: Size of scatter points
            point_alpha: Alpha transparency of points
            accumulate_mode: If True, selections add to existing
            lasso_color: Color of lasso selection line
            lasso_alpha: Alpha of lasso fill
            selection_color: Color of selection highlight
            selection_linewidth: Width of selection outline
            normalize: If True, normalize data to sum to SCALE
            show_gridlines: If True, show ternary gridlines
            **kwargs: Additional kwargs (ignored to prevent tkinter errors)
        """
        # Filter out any kwargs that tkinter Frame doesn't understand
        # This prevents errors like "unknown option -plot_type"
        tkinter_safe_kwargs = {}
        for key, value in kwargs.items():
            if key not in ('plot_type', 'z_col_extra', 'ternary_mode'):
                tkinter_safe_kwargs[key] = value
        
        super().__init__(parent, **tkinter_safe_kwargs)
        
        logger.info(f"=== TernarySelectionWidget.__init__ ===")
        logger.debug(f"  x_col={x_col}, y_col={y_col}, z_col={z_col}")
        logger.debug(f"  data shape: {data.shape}")
        logger.debug(f"  color_by={color_by}, point_size={point_size}")
        
        # Store references
        self.gui_manager = gui_manager
        self.theme = gui_manager.theme_colors
        self.fonts = gui_manager.fonts
        self.data = data.reset_index(drop=True)
        self.x_col = x_col  # Bottom
        self.y_col = y_col  # Right
        self.z_col = z_col  # Left
        self.color_by = color_by
        self.color_map = color_map
        self.on_selection = on_selection
        self.point_size = point_size
        self.point_alpha = point_alpha
        self.accumulate_mode = accumulate_mode
        self.lasso_color = lasso_color
        self.lasso_alpha = lasso_alpha
        self.selection_color = selection_color
        self.selection_linewidth = selection_linewidth
        self.normalize = normalize
        self.show_gridlines = show_gridlines
        
        # Validate columns exist
        missing_cols = []
        for col in [x_col, y_col, z_col]:
            if col not in self.data.columns:
                missing_cols.append(col)
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            logger.debug(f"Available columns: {list(self.data.columns)[:20]}...")
            raise ValueError(f"Missing columns for ternary plot: {missing_cols}")
        
        # Extract and normalize ternary coordinates
        self._prepare_ternary_data()
        
        # Selection state
        self._selected_mask = np.zeros(len(self._valid_data), dtype=bool)
        
        # Category data for coloring
        self._categories = None
        if self.color_by and self.color_by in self._valid_data.columns:
            self._categories = self._valid_data[self.color_by].astype(str).to_numpy()
            logger.debug(f"Categories extracted: {len(np.unique(self._categories))} unique values")
        else:
            self._categories = np.array([""] * len(self._valid_data), dtype=object)
        
        # Filter values for dropdown
        self.filter_var = tk.StringVar(value="")
        if filter_values is not None:
            self._filter_values = list(filter_values)
        elif self.color_by and self.color_by in self._valid_data.columns:
            uniques = (
                self._valid_data[self.color_by]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            self._filter_values = sorted(uniques)
        else:
            self._filter_values = []
        
        logger.debug(f"Filter values: {len(self._filter_values)} items")
        
        # Build UI
        self._build_ui()
        
        # Apply theme
        self.gui_manager.apply_theme(self)
        
        # Create the ternary plot
        self._create_plot()
        
        # Bind resize handler
        self.plot_frame.bind("<Configure>", self._on_configure)
        
        logger.info(f"TernarySelectionWidget initialized with {len(self._valid_data)} valid points")
    
    def _prepare_ternary_data(self):
        """
        Prepare data for ternary plotting.
        
        - Drops rows with null values in any of the three components
        - Optionally normalizes so components sum to SCALE
        - Computes Cartesian coordinates for plotting
        """
        logger.debug("Preparing ternary data...")
        
        # Get the three component columns
        cols = [self.x_col, self.y_col, self.z_col]
        
        # Drop rows with any nulls in the three columns
        self._valid_data = self.data.dropna(subset=cols).copy()
        logger.debug(f"After dropping nulls: {len(self._valid_data)} rows (from {len(self.data)})")
        
        if len(self._valid_data) == 0:
            logger.warning("No valid data after dropping nulls!")
            self._ternary_coords = np.zeros((0, 3))
            self._cartesian_coords = np.zeros((0, 2))
            return
        
        # Extract raw values
        raw = self._valid_data[cols].values.astype(float)
        logger.debug(f"Raw data shape: {raw.shape}")
        logger.debug(f"Raw data range: min={raw.min():.2f}, max={raw.max():.2f}")
        
        # Normalize if requested
        if self.normalize:
            row_sums = raw.sum(axis=1, keepdims=True)
            # Avoid division by zero
            row_sums[row_sums == 0] = 1
            self._ternary_coords = (raw / row_sums) * self.SCALE
            logger.debug(f"Normalized to scale={self.SCALE}")
        else:
            self._ternary_coords = raw
        
        # Convert to Cartesian for lasso selection
        # Standard ternary transformation:
        # x = 0.5 * (2*b + c) / (a + b + c)
        # y = (sqrt(3)/2) * c / (a + b + c)
        a = self._ternary_coords[:, 0]  # Bottom (x_col)
        b = self._ternary_coords[:, 1]  # Right (y_col)
        c = self._ternary_coords[:, 2]  # Left (z_col)
        total = a + b + c
        
        # Compute Cartesian coordinates (scaled to SCALE)
        cart_x = 0.5 * (2 * b + c) / total * self.SCALE
        cart_y = (np.sqrt(3) / 2) * c / total * self.SCALE
        
        self._cartesian_coords = np.column_stack([cart_x, cart_y])
        logger.debug(f"Cartesian coords computed: shape={self._cartesian_coords.shape}")
        logger.debug(f"Cartesian X range: [{cart_x.min():.2f}, {cart_x.max():.2f}]")
        logger.debug(f"Cartesian Y range: [{cart_y.min():.2f}, {cart_y.max():.2f}]")
    
    def _build_ui(self):
        """Build the control bar and plot frame."""
        logger.debug("Building UI...")
        
        # Top control frame
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(fill=tk.X, side=tk.TOP, padx=5, pady=5)
        
        # Filter label
        lbl = ttk.Label(
            ctrl_frame,
            text="Filter selectable:",
            style="Content.TLabel",
        )
        lbl.pack(side=tk.LEFT, padx=(0, 5))
        
        # Filter dropdown
        if self._filter_values:
            try:
                self.filter_widget = self.gui_manager.create_searchable_optionmenu(
                    parent=ctrl_frame,
                    items=[""] + self._filter_values,
                    variable=self.filter_var,
                    width=20,
                    placeholder="(none - all selectable)",
                    on_change=self._on_filter_change,
                )
                self.filter_widget.pack(side=tk.LEFT, fill=tk.X, expand=True)
            except Exception as e:
                logger.warning(f"Could not create searchable optionmenu: {e}")
                # Fallback to plain OptionMenu
                self.filter_widget = tk.OptionMenu(
                    ctrl_frame,
                    self.filter_var,
                    "",
                    *self._filter_values,
                    command=lambda *_: self._on_filter_change(self.filter_var.get()),
                )
                self.gui_manager.style_dropdown(self.filter_widget, width=25)
                self.filter_widget.pack(side=tk.LEFT, fill=tk.X, expand=True)
        else:
            no_filter_lbl = ttk.Label(
                ctrl_frame,
                text="(no categorical filter - all points selectable)",
                style="Content.TLabel",
            )
            no_filter_lbl.pack(side=tk.LEFT, padx=(0, 5))
        
        # Clear selection button
        clear_btn = self.gui_manager.create_modern_button(
            parent=ctrl_frame,
            text="Clear selection",
            color=self.theme.get("accent_blue", "#3a7ca5"),
            command=self.clear_selection,
        )
        clear_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Accumulation mode toggle
        self.accum_var = tk.BooleanVar(value=self.accumulate_mode)
        accum_check = ttk.Checkbutton(
            ctrl_frame,
            text="Add to selection",
            variable=self.accum_var,
            command=self._toggle_accumulate,
            style="Content.TCheckbutton"
        )
        accum_check.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Separator
        sep = ttk.Separator(self, orient=tk.HORIZONTAL)
        sep.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        # Plot frame
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)
        
        logger.debug("UI built successfully")
    
    def _create_plot(self):
        """Create the ternary plot using python-ternary."""
        logger.debug("Creating ternary plot...")
        
        # Check if python-ternary is available
        try:
            import ternary
            logger.debug("python-ternary imported successfully")
        except ImportError:
            logger.error("python-ternary package not installed!")
            self._create_fallback_message()
            return
        
        if len(self._valid_data) == 0:
            logger.warning("No valid data to plot")
            self._create_fallback_message("No valid data to plot")
            return
        
        # Create figure
        self.fig, self.tax = ternary.figure(scale=self.SCALE)
        self.ax = self.tax.get_axes()
        
        # Theme the figure
        bg = self.theme.get("background", "#1e1e1e")
        fg = self.theme.get("text", "#e0e0e0")
        secondary_bg = self.theme.get("secondary_bg", "#252526")
        
        self.fig.patch.set_facecolor(bg)
        self.ax.set_facecolor(secondary_bg)
        
        # Configure ternary appearance
        # Note: boundary() uses axes_colors internally, don't pass 'color' directly
        axes_colors = {'l': fg, 'r': fg, 'b': fg}
        self.tax.boundary(linewidth=2.0, axes_colors=axes_colors)
        
        if self.show_gridlines:
            self.tax.gridlines(multiple=10, linewidth=0.5, alpha=0.3, color=fg)
        
        # Add ticks with themed colors
        self.tax.ticks(
            axis='lbr',
            linewidth=1,
            multiple=20,
            offset=0.02,
            tick_formats="%.0f",
            fontsize=8,
            colors={'l': fg, 'r': fg, 'b': fg},  # Tick mark colors
        )

        # Fix tick label colors (python-ternary renders labels as text with default black)
        # We need to update all text elements in the axes to use theme color
        for text in self.ax.texts:
            text.set_color(fg)
        
        # Add axis labels
        fontsize = 10
        offset = 0.14
        self.tax.bottom_axis_label(self.x_col, fontsize=fontsize, offset=offset, color=fg)
        self.tax.right_axis_label(self.y_col, fontsize=fontsize, offset=offset, color=fg)
        self.tax.left_axis_label(self.z_col, fontsize=fontsize, offset=offset, color=fg)
        
        # Compute colors
        colors = self._compute_point_colors()
        logger.debug(f"Computed colors for {len(colors)} points")
        
        # Plot points using ternary coordinates
        # python-ternary expects list of (a, b, c) tuples
        points = [tuple(row) for row in self._ternary_coords]
        
        # We need to use ax.scatter with transformed coordinates for lasso to work
        # python-ternary's scatter doesn't return a PathCollection we can use
        
        # Transform ternary to the axes coordinates that python-ternary uses
        # This is different from our Cartesian transform - we need to match what ternary does
        self._scatter = self.tax.scatter(
            points,
            c=colors,
            s=self.point_size,
            alpha=self.point_alpha,
            zorder=2
        )
        
        # Store base colors for selection highlighting
        if hasattr(self._scatter, 'get_facecolors'):
            self._base_facecolors = self._scatter.get_facecolors().copy()
        else:
            self._base_facecolors = colors
        
        # Add legend
        self._add_legend()
        
        # Remove default matplotlib axes decorations (ternary handles this)
        self.ax.axis('off')
        
        # Tight layout
        self.fig.tight_layout(pad=1.5)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # Set up lasso selector on the underlying axes
        # The lasso will work in the transformed coordinate space
        self._lasso = LassoSelector(self.ax, onselect=self._on_lasso_select)
        
        # Right-click to clear
        self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        
        logger.info(f"Ternary plot created with {len(points)} points")
    
    def _create_fallback_message(self, message: str = "python-ternary package not installed"):
        """Show a fallback message when ternary plot can't be created."""
        logger.debug(f"Creating fallback message: {message}")
        
        msg_frame = ttk.Frame(self.plot_frame)
        msg_frame.pack(fill=tk.BOTH, expand=True)
        
        msg_label = ttk.Label(
            msg_frame,
            text=f"⚠️ {message}\n\nInstall with: pip install python-ternary",
            style="Content.TLabel",
            justify=tk.CENTER
        )
        msg_label.pack(expand=True)
        
        # Set dummy attributes so other methods don't crash
        self.fig = None
        self.ax = None
        self.tax = None
        self.canvas = None
        self._scatter = None
        self._lasso = None
    
    def _compute_point_colors(self) -> np.ndarray:
        """
        Compute matplotlib colors for each point.

        Handles:
        - Direct hex columns (combined_hex, wet_hex, dry_hex, hex_color)
        - ColorMap objects (categorical and numeric)
        - Dict color maps (hex strings)
        - Fallback to colormap gradient

        Returns:
            numpy array of RGB colors shape (n_points, 3)
        """
        logger.debug(f"Computing point colors:")
        logger.debug(f"  color_by: {self.color_by}")
        logger.debug(f"  color_map type: {type(self.color_map)}")
        logger.debug(f"  ColorMapType available: {ColorMapType is not None}")
        logger.debug(f"  categories: {len(self._categories) if self._categories is not None else 'None'} items")

        n = len(self._valid_data)
        colors = np.zeros((n, 3), dtype=float)

        # Check for direct hex color column (values ARE the colors)
        # This is the preferred mode for combined_hex, wet_hex, dry_hex columns
        hex_column_names = {'hex_color', 'combined_hex', 'wet_hex', 'dry_hex'}
        if self.color_by and self.color_by.lower() in hex_column_names and self.color_by in self._valid_data.columns:
            logger.info(f"Using direct hex values from column '{self.color_by}'")
            valid_count = 0
            for i, hex_val in enumerate(self._valid_data[self.color_by].values):
                if isinstance(hex_val, str) and hex_val:
                    hex_str = hex_val if hex_val.startswith('#') else '#' + hex_val
                    try:
                        if len(hex_str) in [7, 9]:  # Valid hex length
                            rgb = mcolors.hex2color(hex_str)
                            colors[i] = rgb[:3]
                            valid_count += 1
                        else:
                            colors[i] = (0.5, 0.5, 0.5)
                    except Exception:
                        colors[i] = (0.5, 0.5, 0.5)
                else:
                    colors[i] = (0.5, 0.5, 0.5)
            logger.debug(f"  Applied {valid_count}/{n} valid hex colors")
            return colors

        # Check for ColorMap object with categorical type
        # Handle both imported ColorMapType and fallback string comparison
        is_categorical_colormap = False
        if self.color_map is not None and hasattr(self.color_map, 'type'):
            if ColorMapType is not None:
                is_categorical_colormap = self.color_map.type == ColorMapType.CATEGORICAL
            elif hasattr(self.color_map.type, 'value'):
                # Fallback: compare string values if enum import failed
                is_categorical_colormap = str(self.color_map.type.value).lower() == 'categorical'
            elif hasattr(self.color_map.type, 'name'):
                is_categorical_colormap = str(self.color_map.type.name).lower() == 'categorical'

        if is_categorical_colormap and self._categories is not None:
            logger.info(f"Using categorical ColorMap for {n} points")
            for i, cat in enumerate(self._categories):
                b, g, r = self.color_map.get_color(cat)  # BGR
                colors[i, 0] = r / 255.0
                colors[i, 1] = g / 255.0
                colors[i, 2] = b / 255.0
            return colors

        # Check for dict color map - this is the most common path from AdvancedFilterWindow
        if isinstance(self.color_map, dict) and self._categories is not None:
            logger.info(f"Using dict color map with {len(self.color_map)} entries for {n} points")
            unique_cats = set()
            matched_count = 0

            for i, cat in enumerate(self._categories):
                unique_cats.add(cat)

                # Handle empty/null categories
                if cat == '' or cat is None or (isinstance(cat, float) and np.isnan(cat)):
                    colors[i] = (0.5, 0.5, 0.5)  # Gray for missing
                    continue

                color_str = self.color_map.get(cat, '#808080')  # Gray fallback
                if isinstance(color_str, str):
                    if not color_str.startswith('#'):
                        color_str = '#' + color_str
                    try:
                        rgb = mcolors.hex2color(color_str)
                        colors[i, 0] = rgb[0]
                        colors[i, 1] = rgb[1]
                        colors[i, 2] = rgb[2]
                        matched_count += 1
                    except Exception as e:
                        logger.debug(f"Failed to parse color '{color_str}' for category '{cat}': {e}")
                        colors[i] = (0.5, 0.5, 0.5)
                elif isinstance(color_str, (tuple, list)) and len(color_str) >= 3:
                    # Assume BGR tuple from OpenCV
                    colors[i, 0] = color_str[2] / 255.0 if color_str[2] > 1 else color_str[2]
                    colors[i, 1] = color_str[1] / 255.0 if color_str[1] > 1 else color_str[1]
                    colors[i, 2] = color_str[0] / 255.0 if color_str[0] > 1 else color_str[0]
                    matched_count += 1
                else:
                    colors[i] = (0.5, 0.5, 0.5)

            logger.debug(f"  {len(unique_cats)} unique categories, {matched_count}/{n} colors matched")
            return colors

        # Check for numeric ColorMap
        is_numeric_colormap = False
        if self.color_map is not None and hasattr(self.color_map, 'type'):
            if ColorMapType is not None:
                is_numeric_colormap = self.color_map.type == ColorMapType.NUMERIC
            elif hasattr(self.color_map.type, 'value'):
                is_numeric_colormap = str(self.color_map.type.value).lower() == 'numeric'
            elif hasattr(self.color_map.type, 'name'):
                is_numeric_colormap = str(self.color_map.type.name).lower() == 'numeric'

        if is_numeric_colormap:
            logger.info("Using numeric ColorMap")
            # Use first component (x_col) for coloring
            values = self._valid_data[self.x_col].values
            for i, val in enumerate(values):
                b, g, r = self.color_map.get_color(val)
                colors[i, 0] = r / 255.0
                colors[i, 1] = g / 255.0
                colors[i, 2] = b / 255.0
            return colors

        # Fallback: use viridis colormap based on first component
        logger.info(f"Using fallback viridis colormap (no color_map matched)")
        logger.debug(f"  color_map is None: {self.color_map is None}")
        logger.debug(f"  color_map is dict: {isinstance(self.color_map, dict)}")
        if self.color_map is not None:
            logger.debug(f"  color_map has type attr: {hasattr(self.color_map, 'type')}")

        values = self._valid_data[self.x_col].values
        norm = matplotlib.colors.Normalize(
            vmin=np.nanmin(values),
            vmax=np.nanmax(values)
        )
        cmap = matplotlib.cm.get_cmap("viridis")
        colors = cmap(norm(values))[:, :3]

        return colors
    
    def _add_legend(self):
        """Add a legend to the ternary plot."""
        # Skip legend for hex_color (too many unique values)
        if self.color_by == 'hex_color':
            logger.debug("Skipping legend for hex_color column")
            return
        
        if self.color_map is None:
            return
        
        # Handle categorical ColorMap
        if (hasattr(self.color_map, 'type') and 
            ColorMapType is not None and 
            self.color_map.type == ColorMapType.CATEGORICAL):
            self._add_categorical_legend_colormap()
        
        # Handle dict color maps
        elif isinstance(self.color_map, dict):
            self._add_categorical_legend_dict()
        
        # Handle numeric ColorMap
        elif (hasattr(self.color_map, 'type') and 
              ColorMapType is not None and 
              self.color_map.type == ColorMapType.NUMERIC):
            self._add_numeric_legend()
    
    def _add_categorical_legend_colormap(self):
        """Add legend for ColorMap categorical data."""
        if self._categories is None or self.color_by not in self._valid_data.columns:
            return
        
        categories = self._valid_data[self.color_by].dropna().astype(str)
        value_counts = categories.value_counts().to_dict()
        
        handles = []
        labels = []
        
        sorted_items = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
        
        for cat, count in sorted_items[:10]:  # Limit to top 10
            b, g, r = self.color_map.get_color(cat)
            color_rgb = (r/255.0, g/255.0, b/255.0)
            handles.append(Rectangle((0, 0), 1, 1, fc=color_rgb, edgecolor='none'))
            labels.append(f"{cat} ({count})")
        
        if handles:
            legend = self.ax.legend(
                handles, labels,
                loc='upper right',
                fontsize=7,
                framealpha=0.9,
                edgecolor=self.theme.get("text", "#e0e0e0"),
                facecolor=self.theme.get("secondary_bg", "#252526"),
                labelcolor=self.theme.get("text", "#e0e0e0"),
                ncol=1,
                columnspacing=0.5,
                handlelength=1.0,
                handletextpad=0.5,
            )
            legend.set_zorder(10)
    
    def _add_categorical_legend_dict(self):
        """Add legend for dict-based color mapping."""
        if self._categories is None or self.color_by not in self._valid_data.columns:
            return
        
        categories = self._valid_data[self.color_by].dropna().astype(str)
        value_counts = categories.value_counts().to_dict()
        
        handles = []
        labels = []
        
        sorted_items = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
        
        for cat, count in sorted_items[:10]:  # Limit to top 10
            color_str = self.color_map.get(cat, '#808080')
            
            if isinstance(color_str, str) and color_str.startswith('#'):
                try:
                    color_rgb = mcolors.hex2color(color_str)
                except Exception:
                    color_rgb = (0.5, 0.5, 0.5)
            else:
                color_rgb = (0.5, 0.5, 0.5)
            
            handles.append(Rectangle((0, 0), 1, 1, fc=color_rgb, edgecolor='none'))
            labels.append(f"{cat} ({count})")
        
        if handles:
            legend = self.ax.legend(
                handles, labels,
                loc='upper right',
                fontsize=7,
                framealpha=0.9,
                edgecolor=self.theme.get("text", "#e0e0e0"),
                facecolor=self.theme.get("secondary_bg", "#252526"),
                labelcolor=self.theme.get("text", "#e0e0e0"),
                ncol=1,
            )
            legend.set_zorder(10)
    
    def _add_numeric_legend(self):
        """Add colorbar for numeric ColorMap."""
        # For numeric, we'd add a colorbar - skip for now as it's complex with ternary
        logger.debug("Numeric legend (colorbar) not yet implemented for ternary")
        pass
    
    def _on_lasso_select(self, verts):
        """
        Handle lasso selection completion.
        
        The lasso vertices are in the axes coordinate space.
        We need to check which of our data points fall inside.
        """
        logger.debug(f"Lasso selection with {len(verts)} vertices")
        
        if len(verts) < 3:
            logger.debug("Lasso too small, ignoring")
            return
        
        # Create path from lasso vertices
        path = Path(verts)
        
        # Get the scatter plot's transformed coordinates
        # For python-ternary, we need to get the actual screen coordinates of our points
        if self._scatter is None:
            logger.warning("No scatter plot available for selection")
            return
        
        # Use pre-computed Cartesian coordinates for lasso selection
        # (self._scatter from python-ternary returns Axes, not PathCollection)
        if self._cartesian_coords is None or len(self._cartesian_coords) == 0:
            logger.warning("No Cartesian coordinates available for selection")
            return
        
        offsets = self._cartesian_coords
        logger.debug(f"Using Cartesian coords for selection: shape={offsets.shape}")
        
        # Check which points are inside the lasso path
        inside = path.contains_points(offsets)
        logger.debug(f"Points inside lasso: {np.sum(inside)}")
        
        # Apply filter if set
        filter_cat = self.filter_var.get()
        if filter_cat and self._categories is not None:
            # Only select points matching the filter
            cat_mask = self._categories == filter_cat
            inside = inside & cat_mask
            logger.debug(f"After category filter '{filter_cat}': {np.sum(inside)} points")
        
        # Update selection
        if self.accumulate_mode or self.accum_var.get():
            self._selected_mask = self._selected_mask | inside
            logger.debug("Accumulated selection")
        else:
            self._selected_mask = inside
            logger.debug("Replaced selection")
        
        # Update visual
        self._update_selection_visual()
        
        # Fire callback
        self._fire_selection_callback()
    
    def _update_selection_visual(self):
        """Update the scatter plot to show selected points."""
        if self._scatter is None or self._base_facecolors is None:
            return
        
        try:
            # Get current colors
            facecolors = self._base_facecolors.copy()
            
            # Get current sizes
            n = len(self._selected_mask)
            sizes = np.full(n, self.point_size, dtype=float)
            
            # Highlight selected points
            selected_idx = np.where(self._selected_mask)[0]
            if len(selected_idx) > 0:
                # Make selected points larger
                sizes[selected_idx] = self.point_size * 1.5
                
                # Add yellow tint to selected points
                sel_color = mcolors.hex2color(self.selection_color)
                for idx in selected_idx:
                    # Blend with selection color
                    facecolors[idx, :3] = 0.5 * facecolors[idx, :3] + 0.5 * np.array(sel_color)
            
            self._scatter.set_facecolors(facecolors)
            self._scatter.set_sizes(sizes)
            self.canvas.draw_idle()
            
            logger.debug(f"Selection visual updated: {len(selected_idx)} points highlighted")
            
        except Exception as e:
            logger.error(f"Error updating selection visual: {e}")
    
    def _fire_selection_callback(self):
        """Call the on_selection callback with selected data."""
        if self.on_selection is None:
            logger.debug("No on_selection callback registered")
            return
        
        # Get selected rows from the VALID data (post null-drop)
        selected_df = self._valid_data.loc[self._selected_mask].copy()
        logger.info(f"Firing selection callback with {len(selected_df)} selected rows")
        
        try:
            self.on_selection(selected_df)
        except Exception as e:
            logger.error(f"Error in selection callback: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_mouse_press(self, event):
        """Handle mouse press events."""
        # Right-click to clear selection
        if event.button == 3:
            self.clear_selection()
    
    def _on_filter_change(self, value):
        """Handle filter dropdown change."""
        logger.debug(f"Filter changed to: '{value}'")
        # Selection will be filtered on next lasso
    
    def _toggle_accumulate(self):
        """Toggle accumulation mode."""
        self.accumulate_mode = self.accum_var.get()
        logger.debug(f"Accumulate mode: {self.accumulate_mode}")
    
    def _on_configure(self, event):
        """Handle resize events."""
        if self.fig is not None and event.width > 1 and event.height > 1:
            # Resize figure to match frame
            try:
                dpi = self.fig.get_dpi()
                self.fig.set_size_inches(event.width / dpi, event.height / dpi)
                self.canvas.draw_idle()
            except Exception as e:
                logger.debug(f"Resize handling: {e}")
    
    # =========================================================================
    # Public API
    # =========================================================================
    
    def clear_selection(self):
        """Clear all selected points."""
        logger.debug("Clearing selection")
        self._selected_mask = np.zeros(len(self._valid_data), dtype=bool)
        self._update_selection_visual()
        self._fire_selection_callback()
    
    def get_selected_indices(self) -> np.ndarray:
        """Get indices of selected points."""
        return np.where(self._selected_mask)[0]
    
    def get_selected_data(self) -> pd.DataFrame:
        """Get DataFrame of selected rows."""
        return self._valid_data.loc[self._selected_mask].copy()
    
    def get_selection_count(self) -> int:
        """Get number of selected points."""
        return int(np.sum(self._selected_mask))
    
    def set_point_size(self, size: float):
        """Update point size and redraw."""
        self.point_size = max(5.0, min(100.0, float(size)))
        logger.debug(f"Setting point size to {self.point_size}")
        
        if self._scatter is not None:
            sizes = np.full(len(self._valid_data), self.point_size, dtype=float)
            selected_idx = np.where(self._selected_mask)[0]
            if len(selected_idx) > 0:
                sizes[selected_idx] = self.point_size * 1.5
            self._scatter.set_sizes(sizes)
            self.canvas.draw_idle()
