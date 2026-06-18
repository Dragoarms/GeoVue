import tkinter as tk
from tkinter import ttk

from typing import Optional, Callable, List, Any

import numpy as np
import pandas as pd
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

from gui.gui_manager import GUIManager
from processing.LoggingReviewStep.color_map_manager import ColorMap, ColorMapType  # adjust import path if needed


class ScatterSelectionWidget(tk.Frame):
    """
    Themed, reusable scatter-plot widget with:
      - X/Y from a DataFrame
      - Color-by categorical column using ColorMap
      - Optional 'Filter selectable' category
      - Lasso/polygon selection
      - Selection callback returning the selected DataFrame slice
    """

    def __init__(
        self,
        parent,
        gui_manager: GUIManager,
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        color_by: Optional[str] = None,
        color_map: Optional[ColorMap] = None,
        filter_values: Optional[List[str]] = None,
        on_selection: Optional[Callable[[pd.DataFrame], None]] = None,
        point_size: float = 20.0,
        point_alpha: float = 0.6,
        accumulate_mode: bool = False,  # Allow additive selection
        lasso_color: str = "#00ff00",
        lasso_alpha: float = 0.3,
        selection_color: str = "#ffff00",
        selection_linewidth: int = 2,
        **kwargs,
    ):
        super().__init__(parent, **kwargs)

        self.gui_manager = gui_manager
        self.theme = gui_manager.theme_colors
        self.fonts = gui_manager.fonts
        
        # Reorder data so empty hex values are drawn first (appear behind)
        hex_column_names = {'hex_color', 'combined_hex', 'wet_hex', 'dry_hex'}
        if color_by and color_by.lower() in hex_column_names and color_by in data.columns:
            # Sort so empty/invalid hex values come first (drawn first = behind)
            data = data.copy()
            has_hex = data[color_by].apply(lambda x: bool(x) and isinstance(x, str) and len(x) >= 6)
            data['_hex_sort'] = has_hex.astype(int)  # 0 for invalid, 1 for valid
            data = data.sort_values('_hex_sort', ascending=True).drop(columns=['_hex_sort'])
        
        self.data = data.reset_index(drop=True)
        self.x_col = x_col
        self.y_col = y_col
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

        # Internal arrays for fast lasso work
        self._x = self.data[self.x_col].to_numpy(dtype=float)
        self._y = self.data[self.y_col].to_numpy(dtype=float)
        self._categories = None
        if self.color_by and self.color_by in self.data.columns:
            self._categories = self.data[self.color_by].astype(str).to_numpy()
        else:
            self._categories = np.array([""] * len(self.data), dtype=object)

        # Selection state
        self._selected_mask = np.zeros(len(self.data), dtype=bool)
        self._lasso_patch = None  # For visualizing active lasso

        # Filter selectable
        self.filter_var = tk.StringVar(value="")  # empty == all selectable

        # If caller didn't provide filter values, compute from color_by
        if filter_values is not None:
            self._filter_values = list(filter_values)
        elif self.color_by and self.color_by in self.data.columns:
            uniques = (
                self.data[self.color_by]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            self._filter_values = sorted(uniques)
        else:
            self._filter_values = []

        # Build UI (controls + canvas)
        self._build_ui()

        # Apply GUIManager theme to our frame and children
        self.gui_manager.apply_theme(self)

        # Initialize scatter plot + lasso
        self._create_plot()

        # Respond to resizing
        self.plot_frame.bind("<Configure>", self._on_configure)

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        """
        Build top control row (Filter selectable) and matplotlib canvas area.
        """
        # Top control frame
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(fill=tk.X, side=tk.TOP, padx=5, pady=5)

        # Label
        lbl = ttk.Label(
            ctrl_frame,
            text="Filter selectable:",
            style="Content.TLabel",
        )
        lbl.pack(side=tk.LEFT, padx=(0, 5))

        # If there are filterable values, use a searchable option menu
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
            except Exception:
                # Fallback to a plain OptionMenu if anything goes wrong
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
            # No filter values - show a simple label so user understands
            no_filter_lbl = ttk.Label(
                ctrl_frame,
                text="(no categorical filter - all points selectable)",
                style="Content.TLabel",
            )
            no_filter_lbl.pack(side=tk.LEFT, padx=(0, 5))

        # Add a small "Clear selection" button
        clear_btn = self.gui_manager.create_modern_button(
            parent=ctrl_frame,
            text="Clear selection",
            color=self.theme.get("accent_blue", "#3a7ca5"),
            command=self.clear_selection,
        )
        clear_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Add accumulation mode toggle
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

    # ------------------------------------------------------------------ Plot creation

    def _create_plot(self):
        """
        Create the matplotlib figure, scatter plot and lasso selector.
        """
        # Figure and axis
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # Theming for figure / axes
        bg = self.theme.get("background", "#1e1e1e")
        fg = self.theme.get("text", "#e0e0e0")
        self.fig.patch.set_facecolor(bg)
        self.ax.set_facecolor(self.theme.get("secondary_bg", "#252526"))
        self.ax.tick_params(colors=fg, labelsize=8)
        self.ax.xaxis.label.set_color(fg)
        self.ax.yaxis.label.set_color(fg)

        self.ax.set_xlabel(self.x_col)
        self.ax.set_ylabel(self.y_col)

        # Colors
        colors = self._compute_point_colors()

        # Check if using hex colors - need variable alpha for invalid points
        hex_column_names = {'hex_color', 'combined_hex', 'wet_hex', 'dry_hex'}
        is_hex_coloring = self.color_by and self.color_by.lower() in hex_column_names
        
        if is_hex_coloring and hasattr(self, '_hex_invalid_mask'):
            # Create RGBA colors with reduced alpha for invalid hex values
            n = len(colors)
            rgba_colors = np.zeros((n, 4))
            rgba_colors[:, :3] = colors[:, :3] if colors.ndim == 2 else colors
            rgba_colors[:, 3] = self.point_alpha  # Default alpha
            # Reduce alpha for invalid hex values
            rgba_colors[self._hex_invalid_mask, 3] = self.point_alpha * 0.3
            
            # Also reduce size for invalid points
            sizes = np.full(n, self.point_size, dtype=float)
            sizes[self._hex_invalid_mask] = self.point_size * 0.5
            
            self._scatter = self.ax.scatter(
                self._x, self._y, c=rgba_colors, s=sizes,
                edgecolors='none', picker=False, zorder=2
            )
        else:
            self._scatter = self.ax.scatter(
                self._x, self._y, c=colors, s=self.point_size, alpha=self.point_alpha, 
                edgecolors='none', picker=False, zorder=2
            )

        # Store original facecolors for later highlighting
        self._base_facecolors = self._scatter.get_facecolors()
        
        # Add compact legend BEFORE creating canvas (so it's included in tight_layout)
        self._add_legend()
        
        # Apply tight layout to prevent legend overlap
        self.fig.tight_layout(pad=1.5)

        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Lasso selector
        self._lasso = LassoSelector(self.ax, onselect=self._on_lasso_select)

        # Optional: click to clear selection (right-click)
        self.canvas.mpl_connect("button_press_event", self._on_mouse_press)

    def _compute_point_colors(self):
        """
        Compute matplotlib colors for each point, using ColorMap if provided.
        ColorMap stores BGR tuples suitable for OpenCV; convert to RGB in [0, 1].
        Supports both ColorMap objects and dict color maps.
        Also supports direct hex columns where values ARE the colors.
        """
        import matplotlib.colors as mcolors
        
        n = len(self.data)
        colors = np.zeros((n, 3), dtype=float)

        # Check for direct hex color column (values ARE the colors)
        # This is the preferred mode for combined_hex, wet_hex, dry_hex columns
        hex_column_names = {'hex_color', 'combined_hex', 'wet_hex', 'dry_hex'}
        if self.color_by and self.color_by.lower() in hex_column_names and self.color_by in self.data.columns:
            # Track invalid hex for alpha adjustment
            self._hex_invalid_mask = np.zeros(n, dtype=bool)
            
            for i, hex_val in enumerate(self.data[self.color_by].values):
                if isinstance(hex_val, str) and hex_val:
                    hex_str = hex_val if hex_val.startswith('#') else '#' + hex_val
                    try:
                        if len(hex_str) in [7, 9]:  # Valid hex length
                            rgb = mcolors.hex2color(hex_str)
                            colors[i] = rgb[:3]
                        else:
                            colors[i] = (0.25, 0.25, 0.25)  # Dark gray for invalid
                            self._hex_invalid_mask[i] = True
                    except Exception:
                        colors[i] = (0.25, 0.25, 0.25)
                        self._hex_invalid_mask[i] = True
                else:
                    colors[i] = (0.25, 0.25, 0.25)  # Dark gray for empty
                    self._hex_invalid_mask[i] = True
            return colors

        # Check for ColorMap object (has .type attribute)
        if self.color_map is not None and hasattr(self.color_map, 'type') and self.color_map.type == ColorMapType.CATEGORICAL and self._categories is not None:
            for i, cat in enumerate(self._categories):
                b, g, r = self.color_map.get_color(cat)  # BGR
                colors[i, 0] = r / 255.0
                colors[i, 1] = g / 255.0
                colors[i, 2] = b / 255.0
        
        # Check for dict color map
        elif isinstance(self.color_map, dict) and self._categories is not None:
            for i, cat in enumerate(self._categories):
                # Handle empty/null categories
                if cat == '' or cat is None or (isinstance(cat, float) and np.isnan(cat)):
                    colors[i] = (0.5, 0.5, 0.5)  # Gray for missing
                    continue
                    
                color_str = self.color_map.get(cat, '#808080')  # Gray fallback
                if isinstance(color_str, str):
                    # Ensure it has # prefix
                    if not color_str.startswith('#'):
                        color_str = '#' + color_str
                    try:
                        rgb = mcolors.hex2color(color_str)
                        colors[i, 0] = rgb[0]
                        colors[i, 1] = rgb[1]
                        colors[i, 2] = rgb[2]
                    except:
                        colors[i] = (0.5, 0.5, 0.5)  # Gray fallback for invalid hex
                elif isinstance(color_str, (tuple, list)) and len(color_str) >= 3:
                    # Assume BGR tuple from OpenCV convention, convert to RGB
                    colors[i, 0] = color_str[2] / 255.0 if color_str[2] > 1 else color_str[2]
                    colors[i, 1] = color_str[1] / 255.0 if color_str[1] > 1 else color_str[1]
                    colors[i, 2] = color_str[0] / 255.0 if color_str[0] > 1 else color_str[0]
                else:
                    colors[i] = (0.5, 0.5, 0.5)  # Gray fallback
        
        # Check for numeric ColorMap
        elif self.color_map is not None and hasattr(self.color_map, 'type') and self.color_map.type == ColorMapType.NUMERIC:
            # Use numeric values from x column for coloring
            for i in range(n):
                b, g, r = self.color_map.get_color(self._x[i])
                colors[i, 0] = r / 255.0
                colors[i, 1] = g / 255.0
                colors[i, 2] = b / 255.0
        
        else:
            # Fallback: use viridis colormap on the color_by column (not X axis)
            if self.color_by and self.color_by in self.data.columns:
                color_data = self.data[self.color_by].values
                
                # Check if data is numeric
                if pd.api.types.is_numeric_dtype(self.data[self.color_by]):
                    # Numeric: use viridis gradient
                    color_values = pd.to_numeric(color_data, errors='coerce')
                    norm = matplotlib.colors.Normalize(
                        vmin=np.nanmin(color_values), vmax=np.nanmax(color_values)
                    )
                    cmap = matplotlib.cm.get_cmap("viridis")
                    colors = cmap(norm(color_values))[:, :3]
                else:
                    # Categorical: assign colors from a qualitative colormap
                    unique_cats = pd.unique(color_data)
                    # Use tab20 for more distinct colors
                    cmap = matplotlib.cm.get_cmap("tab20")
                    cat_to_idx = {cat: i for i, cat in enumerate(unique_cats)}
                    for i, cat in enumerate(color_data):
                        idx = cat_to_idx.get(cat, 0)
                        colors[i] = cmap(idx % 20)[:3]
            else:
                # No color_by column, fall back to X axis
                norm = matplotlib.colors.Normalize(
                    vmin=np.nanmin(self._x), vmax=np.nanmax(self._x)
                )
                cmap = matplotlib.cm.get_cmap("viridis")
                colors = cmap(norm(self._x))[:, :3]

        return colors
    
    def _add_legend(self):
        """
        Add a compact legend to the plot if color mapping is categorical and not hex_color.
        Shows color swatches with category names and point counts.
        """
        # Don't show legend for hex color columns - too many unique colors
        hex_column_names = {'hex_color', 'combined_hex', 'wet_hex', 'dry_hex'}
        if self.color_by and self.color_by.lower() in hex_column_names:
            return
        
        # Only show legend for categorical ColorMap or dict color maps
        if self.color_map is None:
            return
        
        # Handle ColorMap objects
        if hasattr(self.color_map, 'type') and self.color_map.type == ColorMapType.CATEGORICAL:
            self._add_categorical_legend_colormap()
        # Handle dict color maps (from item_manager or simple dicts)
        elif isinstance(self.color_map, dict):
            self._add_categorical_legend_dict()
        # Handle numeric ColorMap objects
        elif hasattr(self.color_map, 'type') and self.color_map.type == ColorMapType.NUMERIC:
            self._add_numeric_legend()
    
    def _add_categorical_legend_colormap(self):
        """Add legend for ColorMap categorical data with counts"""
        if self._categories is None or self.color_by not in self.data.columns:
            return
        
        # Get unique categories and their counts
        categories = self.data[self.color_by].dropna().astype(str)
        value_counts = categories.value_counts().to_dict()
        
        # Create legend handles
        from matplotlib.patches import Rectangle
        handles = []
        labels = []
        
        # Sort by count (descending)
        sorted_items = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
        
        for cat, count in sorted_items:
            # Get color from ColorMap (returns BGR, convert to RGB normalized)
            b, g, r = self.color_map.get_color(cat)
            color_rgb = (r/255.0, g/255.0, b/255.0)
            
            # Create color swatch
            handles.append(Rectangle((0, 0), 1, 1, fc=color_rgb, edgecolor='none'))
            labels.append(f"{cat} ({count})")
        
        # Add compact legend in upper right with explicit z-order
        legend = self.ax.legend(
            handles, labels,
            loc='upper right',
            fontsize=7,
            framealpha=0.9,  # Increased for better visibility
            edgecolor=self.theme.get("text", "#e0e0e0"),
            facecolor=self.theme.get("secondary_bg", "#252526"),
            labelcolor=self.theme.get("text", "#e0e0e0"),
            ncol=1 if len(handles) <= 10 else 2,  # Two columns if many items
            columnspacing=0.5,
            handlelength=1.0,
            handletextpad=0.5,
        )
        legend.set_zorder(10)  # Ensure legend is on top
        
        # Set legend title color
        if legend.get_title():
            legend.get_title().set_color(self.theme.get("text", "#e0e0e0"))
    
    def _add_categorical_legend_dict(self):
        """Add legend for dict-based color mapping with counts"""
        if self._categories is None or self.color_by not in self.data.columns:
            return
        
        # Get unique categories and their counts
        categories = self.data[self.color_by].dropna().astype(str)
        value_counts = categories.value_counts().to_dict()
        
        # Create legend handles
        from matplotlib.patches import Rectangle
        import matplotlib.colors as mcolors
        handles = []
        labels = []
        
        # Sort by count (descending)
        sorted_items = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
        
        for cat, count in sorted_items:
            # Get color from dict
            color_str = self.color_map.get(cat, '#808080')
            
            # Convert hex to RGB if needed
            if isinstance(color_str, str) and color_str.startswith('#'):
                color_rgb = mcolors.hex2color(color_str)
            else:
                color_rgb = (0.5, 0.5, 0.5)  # Gray fallback
            
            # Create color swatch
            handles.append(Rectangle((0, 0), 1, 1, fc=color_rgb, edgecolor='none'))
            labels.append(f"{cat} ({count})")
        
        # Add compact legend in upper right with explicit z-order
        legend = self.ax.legend(
            handles, labels,
            loc='upper right',
            fontsize=7,
            framealpha=0.9,  # Increased for better visibility
            edgecolor=self.theme.get("text", "#e0e0e0"),
            facecolor=self.theme.get("secondary_bg", "#252526"),
            labelcolor=self.theme.get("text", "#e0e0e0"),
            ncol=1 if len(handles) <= 10 else 2,  # Two columns if many items
            columnspacing=0.5,
            handlelength=1.0,
            handletextpad=0.5,
        )
        legend.set_zorder(10)  # Ensure legend is on top
        
        # Set legend title color
        if legend.get_title():
            legend.get_title().set_color(self.theme.get("text", "#e0e0e0"))
    
    def _add_numeric_legend(self):
        """Add color bar legend for numeric data"""
        if not hasattr(self.color_map, 'ranges') or not self.color_map.ranges:
            return
        
        # Create legend showing value ranges with color swatches
        from matplotlib.patches import Rectangle
        handles = []
        labels = []
        
        for color_range in self.color_map.ranges:
            # Get color (BGR from ColorMap, convert to RGB normalized)
            b, g, r = color_range.color
            color_rgb = (r/255.0, g/255.0, b/255.0)
            
            # Create color swatch
            handles.append(Rectangle((0, 0), 1, 1, fc=color_rgb, edgecolor='none'))
            
            # Format label
            if color_range.label:
                label = color_range.label
            else:
                label = f"{color_range.min:.1f} - {color_range.max:.1f}"
            labels.append(label)
        
        # Add compact legend in upper right
        legend = self.ax.legend(
            handles, labels,
            loc='upper right',
            fontsize=7,
            framealpha=0.85,
            edgecolor=self.theme.get("text", "#e0e0e0"),
            facecolor=self.theme.get("secondary_bg", "#252526"),
            labelcolor=self.theme.get("text", "#e0e0e0"),
            title=self.color_by,
            ncol=1,
            columnspacing=0.5,
            handlelength=1.0,
            handletextpad=0.5,
        )
        
        # Set legend title color
        if legend.get_title():
            legend.get_title().set_color(self.theme.get("text", "#e0e0e0"))

    def _draw_lasso_fill(self, verts):
        """Draw filled polygon for lasso selection visualization"""
        if not verts or len(verts) < 3:
            return
        
        from matplotlib.patches import Polygon
        
        # Remove old lasso fill if exists
        if self._lasso_patch is not None:
            try:
                self._lasso_patch.remove()
            except:
                pass
            self._lasso_patch = None
        
        # Draw filled polygon
        self._lasso_patch = Polygon(
            verts,
            facecolor=self.lasso_color,
            alpha=self.lasso_alpha,
            edgecolor=self.lasso_color,
            linewidth=2,
            zorder=10  # Draw above scatter points
        )
        self.ax.add_patch(self._lasso_patch)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ Events / selection

    def _on_filter_change(self, *_):
        """
        React to changes in the 'Filter selectable' control.
        Keep existing selection but filter out points not matching new category.
        """
        filter_value = self.filter_var.get().strip()
        
        # If there's a filter and selection, keep only selected points matching filter
        if filter_value and self._selected_mask.any() and self._categories is not None:
            eligible = self._categories == filter_value
            self._selected_mask = np.logical_and(self._selected_mask, eligible)
            self._update_selection_visuals()
            self._fire_selection_callback()

    def _on_lasso_select(self, verts: List[Any]):
        """
        Called by matplotlib's LassoSelector when a polygon is finished.
        Verts is a list of (x,y) tuples in data coordinates.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"Lasso selection triggered with {len(verts) if verts else 0} vertices")
        
        if verts is None or len(verts) < 3:
            logger.warning("Lasso selection aborted - not enough vertices")
            return

        # Draw lasso visualization
        self._draw_lasso_fill(verts)

        path = Path(verts)
        points = np.column_stack((self._x, self._y))
        inside = path.contains_points(points)
        
        num_selected = np.sum(inside)
        logger.info(f"Lasso contains {num_selected} points out of {len(points)} total")

        # Filter selectable logic
        filter_value = self.filter_var.get().strip()
        if filter_value and self._categories is not None and len(self._categories) == len(inside):
            eligible = self._categories == filter_value
            inside = np.logical_and(inside, eligible)

        # Accumulative or replace mode
        if self.accumulate_mode:
            # Add to existing selection (OR operation)
            self._selected_mask = np.logical_or(self._selected_mask, inside)
        else:
            # Replace selection
            self._selected_mask = inside

        self._update_selection_visuals()
        self._fire_selection_callback()

    def _on_mouse_press(self, event):
        """
        Optional: right-click anywhere clears selection.
        """
        if event.button == 3:  # right-click
            self.clear_selection()

    def _update_selection_visuals(self):
        """
        Update scatter facecolors to show which points are selected.
        Selected points get a bright colored outline.
        """
        if self._scatter is None:
            return

        # Remove lasso fill patch if it exists
        if self._lasso_patch is not None:
            try:
                self._lasso_patch.remove()
            except:
                pass
            self._lasso_patch = None

        # Use base colors but set edgecolor and linewidth on selected points
        self._scatter.set_facecolors(self._base_facecolors)

        # Parse selection color (convert hex to RGB tuple)
        import matplotlib.colors as mcolors
        try:
            selection_rgb = mcolors.to_rgba(self.selection_color)
        except:
            selection_rgb = (1.0, 1.0, 0.0, 1.0)  # fallback to yellow

        # Edgecolors - all points default to none
        edgecolors = np.zeros_like(self._base_facecolors)
        edgecolors[:] = (0, 0, 0, 0)  # fully transparent

        # Selected: bright colored edge
        selected_idx = np.where(self._selected_mask)[0]
        if len(selected_idx) > 0:
            edgecolors[selected_idx] = selection_rgb

        self._scatter.set_edgecolors(edgecolors)

        # Set linewidths - selected points get thicker edges
        linewidths = np.zeros(len(self._x), dtype=float)
        linewidths[selected_idx] = self.selection_linewidth
        self._scatter.set_linewidths(linewidths)

        # Optionally bump size for selected points
        sizes = np.full(len(self._x), self.point_size, dtype=float)
        sizes[selected_idx] = self.point_size * 1.5  # Slightly larger
        self._scatter.set_sizes(sizes)

        self.canvas.draw_idle()

    def clear_selection(self):
        """Clear the current selection and reset view"""
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Clearing scatter plot selection")
        
        # Clear selection mask
        self._selected_mask = np.zeros(len(self.data), dtype=bool)
        
        # Remove lasso patch if it exists
        if self._lasso_patch is not None:
            try:
                self._lasso_patch.remove()
            except:
                pass
            self._lasso_patch = None
        
        # Update visuals and fire callback
        self._update_selection_visuals()
        self._fire_selection_callback()

    def _fire_selection_callback(self):
        """
        Call the on_selection callback with the selected slice of the data.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if self.on_selection is None:
            logger.debug("No on_selection callback registered")
            return

        selected_df = self.data.loc[self._selected_mask].copy()
        logger.info(f"Firing selection callback with {len(selected_df)} selected rows")
        
        try:
            self.on_selection(selected_df)
        except Exception as e:
            # Log but don't crash UI
            import traceback
            logger.error(f"Error in ScatterSelectionWidget on_selection callback: {e}")
            traceback.print_exc()

    # ------------------------------------------------------------------ Public API

    def _toggle_accumulate(self):
        """Toggle accumulation mode on/off."""
        self.accumulate_mode = self.accum_var.get()

    def get_selected_indices(self) -> np.ndarray:
        """
        Return indices (relative to the original data passed in) of selected points.
        """
        return np.where(self._selected_mask)[0]

    def get_selected_data(self) -> pd.DataFrame:
        """
        Return a copy of the selected rows from the original DataFrame.
        """
        return self.data.loc[self._selected_mask].copy()
    
    def get_selection_count(self) -> int:
        """Get number of currently selected points"""
        return int(np.sum(self._selected_mask))

    def set_point_size(self, size: float):
        """
        Update point size and redraw
        
        Args:
            size: New point size (5-100)
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Clamp size to reasonable range
        self.point_size = max(5.0, min(100.0, float(size)))
        logger.debug(f"Setting point size to {self.point_size}")
        
        # Update scatter plot sizes
        if self._scatter is not None:
            # Compute sizes: selected points are slightly larger
            sizes = np.full(len(self._x), self.point_size, dtype=float)
            selected_idx = np.where(self._selected_mask)[0]
            if len(selected_idx) > 0:
                sizes[selected_idx] = self.point_size * 1.5
            
            self._scatter.set_sizes(sizes)
            self.canvas.draw_idle()
    # ------------------------------------------------------------------ Resizing

    def _on_configure(self, event):
        """
        Called when the plot frame is resized - update figure size to match.
        """
        # Only respond if the event is for the plot_frame, not parent frame
        if event.widget != self.plot_frame:
            return
            
        if not hasattr(self, "fig") or not hasattr(self, "canvas"):
            return

        # Convert pixels to inches (matplotlib uses inches)
        dpi = self.fig.get_dpi()
        # Account for padding/margins
        width_in = max((event.width - 10) / dpi, 1.0)
        height_in = max((event.height - 10) / dpi, 1.0)

        self.fig.set_size_inches(width_in, height_in, forward=True)
        self.canvas.draw_idle()
