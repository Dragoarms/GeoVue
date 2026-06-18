"""
Color Map Editor Dialog - New implementation based on ArcGIS-style UI.

Provides a modal dialog for creating and editing color maps with:
- Discrete mode: Table of intervals with min/max/color
- Continuous mode: Gradient with draggable curve points
- Live histogram preview with actual data distribution
- Integration with ColorMapManager for saving/loading presets
"""

import tkinter as tk
from tkinter import ttk, colorchooser
import logging
import json
import os
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Callable

from gui.dialog_helper import DialogHelper
from gui.widgets.modern_button import ModernButton

# Will import from actual location:
# from processing.LoggingReviewStep.color_map_manager import ColorMap, ColorRange, ColorMapType, ColorMapManager

logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions - Color Conversion
# ============================================================================

def bgr_to_hex(bgr: Tuple[int, int, int]) -> str:
    """
    Convert BGR tuple (OpenCV format) to hex color string.
    
    Args:
        bgr: (B, G, R) tuple, values 0-255
        
    Returns:
        Hex color string like '#RRGGBB'
    """
    b, g, r = bgr
    hex_color = f"#{r:02x}{g:02x}{b:02x}"
    logger.debug(f"BGR {bgr} -> Hex {hex_color}")
    return hex_color


def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """
    Convert hex color string to BGR tuple (OpenCV format).
    
    Args:
        hex_color: Hex color string like '#RRGGBB'
        
    Returns:
        (B, G, R) tuple, values 0-255
    """
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    bgr = (b, g, r)
    logger.debug(f"Hex {hex_color} -> BGR {bgr}")
    return bgr


# ============================================================================
# Histogram Preview Widget
# ============================================================================

class HistogramPreviewWidget(tk.Frame):
    """
    Widget that displays:
    - Histogram of data distribution in main area
    - Horizontal gradient bar below histogram (X-axis aligned)
    - Draggable handles for gradient stops in continuous mode
    - Discrete color blocks in discrete mode
    """

    def __init__(self, parent, gui_manager, width: int = 600, height: int = 400):
        """
        Initialize histogram preview widget.

        Args:
            parent: Parent widget
            gui_manager: GUIManager for theming
            width: Canvas width in pixels
            height: Canvas height in pixels
        """
        super().__init__(parent, bg=gui_manager.theme_colors["background"])

        self.gui_manager = gui_manager
        self.theme = gui_manager.theme_colors
        self.canvas_width = width
        self.canvas_height = height

        # Data state
        self.data_values: List[float] = []
        self.data_min = 0.0
        self.data_max = 1.0
        self.hist_bins: List[float] = []
        self.hist_counts: List[int] = []

        # Display mode
        self.mode = "discrete"  # "discrete" or "continuous"

        # Discrete mode: List of (min, max, hex_color) tuples
        self.discrete_ranges: List[Tuple[float, float, str]] = []

        # Continuous mode: List of (value, hex_color) gradient stops
        self.gradient_stops: List[Tuple[float, str]] = []
        self.gradient_min = 0.0
        self.gradient_max = 1.0

        # Drag/drop state for interactive gradient editing
        self.gradient_stop_handles: Dict[int, Tuple[float, str]] = {}  # {canvas_id: (value, color)}
        self.dragging_stop = None  # Canvas ID of stop being dragged
        self.drag_start_x = None  # Starting x coordinate of drag (now horizontal)
        self.on_gradient_changed_callback = None  # Callback when gradient is modified by dragging

        # Debounce state for redraw during drag (prevents performance bottleneck)
        self._redraw_pending = False
        self._redraw_timer = None
        self._redraw_debounce_ms = 50  # Increased to 50ms for smoother drag
        self._callback_pending = False
        self._callback_timer = None
        self._is_dragging = False  # Track if we're in a drag operation

        # Gradient bar position (set during draw) - now horizontal
        self.gradient_bar_x = 0
        self.gradient_bar_y = 0
        self.gradient_bar_width = 0
        self.gradient_bar_height = 25  # Thinner horizontal bar

        # Canvas item IDs for efficient updates during drag
        self._histogram_bar_ids: List[int] = []
        self._gradient_segment_ids: List[int] = []

        self._build_ui()
        
    def _build_ui(self):
        """Build the canvas UI."""
        # Main canvas
        self.canvas = tk.Canvas(
            self,
            width=self.canvas_width,
            height=self.canvas_height,
            bg=self.theme["field_bg"],
            highlightbackground=self.theme["border"],
            highlightthickness=1,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Bind resize event
        self.canvas.bind('<Configure>', self._on_resize)

        # Bind mouse events for drag/drop in continuous mode
        self.canvas.bind('<Button-1>', self._on_mouse_down)
        self.canvas.bind('<B1-Motion>', self._on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_mouse_up)
        
    def _on_resize(self, event):
        """Handle canvas resize."""
        new_width = event.width
        new_height = event.height

        if new_width != self.canvas_width or new_height != self.canvas_height:
            self.canvas_width = new_width
            self.canvas_height = new_height
            if not self._is_dragging:
                self.redraw()
     
    def set_data(self, data_values: List, is_categorical: bool = False):
        """
        Set the data for histogram display.

        Args:
            data_values: List of numeric or categorical values to display
            is_categorical: Whether the data is categorical (text) or numeric
        """
        self.data_values = data_values
        self.is_categorical = is_categorical

        if self.data_values:
            if is_categorical:
                # For categorical data, we don't compute min/max or histogram
                self.data_min = 0.0
                self.data_max = float(len(data_values) - 1)
                self.hist_counts = []
                self.hist_bins = []
            else:
                # For numeric data - filter out NaN values
                numeric_values = [v for v in self.data_values if isinstance(v, (int, float)) and not np.isnan(v)]

                if numeric_values:
                    self.data_min = float(np.min(numeric_values))
                    self.data_max = float(np.max(numeric_values))

                    # Generate histogram
                    histogram, bin_edges = np.histogram(numeric_values, bins=50)
                    self.hist_counts = histogram.tolist()
                    self.hist_bins = bin_edges.tolist()
                else:
                    self.data_min = 0.0
                    self.data_max = 1.0
                    self.hist_counts = []
                    self.hist_bins = []
        else:
            self.data_min = 0.0
            self.data_max = 1.0
            self.hist_counts = []
            self.hist_bins = []
            self.is_categorical = False

        # Redraw
        self.redraw()
        
    def set_discrete_preview(self, ranges: List[Tuple[float, float, str]]):
        """
        Set discrete mode and update ranges to display.

        Args:
            ranges: List of (min_val, max_val, hex_color) tuples
        """
        self.mode = "discrete"
        self.discrete_ranges = ranges[:]
        self.redraw()

    def set_continuous_preview(self, stops: List[Tuple[float, str]], gradient_min: float, gradient_max: float):
        """
        Set continuous mode and update gradient stops.

        Args:
            stops: List of (value, hex_color) tuples
            gradient_min: Minimum value for gradient mapping
            gradient_max: Maximum value for gradient mapping
        """
        self.mode = "continuous"
        self.gradient_stops = sorted(stops, key=lambda x: x[0])  # Sort by value
        self.gradient_min = gradient_min
        self.gradient_max = gradient_max
        self.redraw()
        
    def redraw(self):
        """Redraw the entire canvas."""
        self.canvas.delete("all")
        self._histogram_bar_ids.clear()
        self._gradient_segment_ids.clear()

        if not self.hist_counts:
            self._draw_empty_canvas()
            return

        # Define layout dimensions - horizontal gradient bar at bottom
        left_margin = 50
        right_margin = 20
        top_margin = 20
        gradient_height = 25  # Height of horizontal gradient bar
        axis_height = 25  # Space for axis labels
        bottom_margin = gradient_height + axis_height + 15

        hist_left = left_margin
        hist_right = self.canvas_width - right_margin
        hist_top = top_margin
        hist_bottom = self.canvas_height - bottom_margin

        # Store gradient bar position for drag handling
        self.gradient_bar_x = hist_left
        self.gradient_bar_y = hist_bottom + 5
        self.gradient_bar_width = hist_right - hist_left
        self.gradient_bar_height = gradient_height

        # Draw histogram bars first (they get colored by gradient)
        self._draw_histogram(hist_left, hist_top, hist_right, hist_bottom)

        # Draw horizontal gradient bar below histogram
        self._draw_gradient_bar(hist_left, self.gradient_bar_y, hist_right - hist_left, gradient_height)

        # Draw axis and labels
        self._draw_axis(hist_left, self.gradient_bar_y + gradient_height + 5, hist_right)
        
    def _draw_empty_canvas(self):
        """Draw empty canvas with message."""
        self.canvas.create_text(
            self.canvas_width / 2,
            self.canvas_height / 2,
            text="No data to display",
            fill=self.theme["text"],
            font=("Arial", 12),
        )

    def _draw_gradient_bar(self, x: float, y: float, width: float, height: float):
        """
        Draw horizontal gradient bar below the histogram.

        Args:
            x: Left edge x coordinate
            y: Top edge y coordinate
            width: Bar width
            height: Bar height
        """
        if self.mode == "discrete":
            self._draw_discrete_gradient_bar(x, y, width, height)
        else:
            self._draw_continuous_gradient_bar(x, y, width, height)

    def _draw_discrete_gradient_bar(self, x: float, y: float, width: float, height: float):
        """Draw discrete color blocks in horizontal gradient bar."""
        if not self.discrete_ranges:
            # Draw gray bar if no ranges
            self.canvas.create_rectangle(
                x, y, x + width, y + height,
                fill="#808080",
                outline=self.theme["border"],
            )
            return

        # Map each range to a horizontal position
        data_range = self.data_max - self.data_min
        if data_range == 0:
            data_range = 1.0

        for min_val, max_val, hex_color in self.discrete_ranges:
            # Calculate x positions based on data values
            x1_frac = (min_val - self.data_min) / data_range
            x2_frac = (max_val - self.data_min) / data_range

            x1 = x + (x1_frac * width)
            x2 = x + (x2_frac * width)

            self.canvas.create_rectangle(
                x1, y, x2, y + height,
                fill=hex_color,
                outline="",
            )

        # Draw border
        self.canvas.create_rectangle(
            x, y, x + width, y + height,
            fill="",
            outline=self.theme["border"],
            width=1,
        )

    def _draw_continuous_gradient_bar(self, x: float, y: float, width: float, height: float):
        """Draw continuous horizontal gradient with interpolated colors."""
        if len(self.gradient_stops) < 2:
            # Need at least 2 stops for gradient, draw gray
            self.canvas.create_rectangle(
                x, y, x + width, y + height,
                fill="#808080",
                outline=self.theme["border"],
            )
            return

        # Draw gradient by creating thin vertical rectangles
        num_segments = min(100, int(width))  # Use fewer segments for performance
        segment_width = width / num_segments

        gradient_range = self.gradient_max - self.gradient_min
        if gradient_range == 0:
            gradient_range = 1.0

        self._gradient_segment_ids.clear()
        for i in range(num_segments):
            # Calculate value at this segment
            t = i / num_segments
            value = self.gradient_min + t * gradient_range

            # Find interpolated color
            color = self._interpolate_gradient_color(value)

            # Draw segment (horizontal)
            x1 = x + (i * segment_width)
            x2 = x + ((i + 1) * segment_width)

            seg_id = self.canvas.create_rectangle(
                x1, y, x2, y + height,
                fill=color,
                outline="",
            )
            self._gradient_segment_ids.append(seg_id)

        # Draw interactive handles for each gradient stop
        self.gradient_stop_handles.clear()
        for stop_value, stop_color in self.gradient_stops:
            # Calculate x position for this stop (normalized position 0-1)
            if gradient_range > 0:
                normalized_pos = (stop_value - self.gradient_min) / gradient_range
            else:
                normalized_pos = 0.5

            # Clamp to [0, 1]
            normalized_pos = max(0.0, min(1.0, normalized_pos))

            handle_x = x + (normalized_pos * width)
            handle_y = y + height / 2

            # Draw triangular handle pointing up from bottom
            handle_size = 8
            handle_id = self.canvas.create_polygon(
                handle_x, y - 2,  # Top point
                handle_x - handle_size, y + height + 4,  # Bottom left
                handle_x + handle_size, y + height + 4,  # Bottom right
                fill=stop_color,
                outline="#ffffff",
                width=2,
                tags="gradient_handle",
            )

            # Store handle mapping
            self.gradient_stop_handles[handle_id] = (stop_value, stop_color)

        # Draw border
        self.canvas.create_rectangle(
            x, y, x + width, y + height,
            fill="",
            outline=self.theme["border"],
            width=1,
        )
            
    def _interpolate_gradient_color(self, value: float) -> str:
        """
        Interpolate color from gradient stops for a given value.
        
        Args:
            value: Data value to get color for
            
        Returns:
            Hex color string
        """
        if not self.gradient_stops:
            return "#808080"
        
        # Find surrounding stops
        stops = self.gradient_stops
        
        # Handle edge cases
        if value <= stops[0][0]:
            return stops[0][1]
        if value >= stops[-1][0]:
            return stops[-1][1]
        
        # Find surrounding stops
        for i in range(len(stops) - 1):
            v1, c1 = stops[i]
            v2, c2 = stops[i + 1]
            
            if v1 <= value <= v2:
                # Interpolate between c1 and c2
                if v2 == v1:
                    t = 0.0
                else:
                    t = (value - v1) / (v2 - v1)
                
                # Convert hex to RGB, interpolate, convert back
                r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
                r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
                
                r = int(r1 + t * (r2 - r1))
                g = int(g1 + t * (g2 - g1))
                b = int(b1 + t * (b2 - b1))
                
                return f"#{r:02x}{g:02x}{b:02x}"
        
        return "#808080"
        
    def _draw_histogram(self, left: float, top: float, right: float, bottom: float):
        """
        Draw histogram bars.

        Args:
            left: Left edge x coordinate
            top: Top edge y coordinate
            right: Right edge x coordinate
            bottom: Bottom edge y coordinate
        """
        if not self.hist_counts:
            return

        width = right - left
        height = bottom - top
        max_count = max(self.hist_counts) if self.hist_counts else 1
        data_range = self.data_max - self.data_min
        if data_range == 0:
            data_range = 1.0

        self._histogram_bar_ids.clear()

        # Draw each bar
        for i, count in enumerate(self.hist_counts):
            # Calculate x position
            bin_min = self.hist_bins[i]
            bin_max = self.hist_bins[i + 1]

            x1 = left + ((bin_min - self.data_min) / data_range) * width
            x2 = left + ((bin_max - self.data_min) / data_range) * width

            # Calculate bar height
            bar_height = (count / max_count) * (height * 0.9)  # Use 90% of height
            y1 = bottom - bar_height
            y2 = bottom

            # Determine bar color based on mode
            bin_center = (bin_min + bin_max) / 2
            if self.mode == "discrete":
                bar_color = self._get_color_for_value_discrete(bin_center)
            else:
                bar_color = self._get_color_for_value_continuous(bin_center)

            bar_id = self.canvas.create_rectangle(
                x1, y1, x2, y2,
                fill=bar_color,
                outline="",
            )
            self._histogram_bar_ids.append((bar_id, bin_center))

    def _get_color_for_value_discrete(self, value: float) -> str:
        """Get color for a value in discrete mode."""
        for min_val, max_val, hex_color in self.discrete_ranges:
            if min_val <= value < max_val:
                return hex_color
        return self.theme.get("accent_blue", "#4a6fae")
    
    def _get_color_for_value_continuous(self, value: float) -> str:
        """
        Get interpolated color for a value in continuous mode based on gradient stops.
        
        Args:
            value: Data value to get color for
            
        Returns:
            Hex color string
        """
        if not self.gradient_stops:
            return self.theme.get("accent_blue", "#4a6fae")
        
        # Normalize value to 0-1 range
        if self.gradient_max <= self.gradient_min:
            normalized = 0.5
        else:
            normalized = (value - self.gradient_min) / (self.gradient_max - self.gradient_min)
            normalized = max(0.0, min(1.0, normalized))  # Clamp to [0, 1]
        
        # Find the two stops to interpolate between
        sorted_stops = sorted(self.gradient_stops, key=lambda x: x[0])
        
        # If value is before first stop or after last stop
        if normalized <= sorted_stops[0][0]:
            return sorted_stops[0][1]
        if normalized >= sorted_stops[-1][0]:
            return sorted_stops[-1][1]
        
        # Find surrounding stops
        for i in range(len(sorted_stops) - 1):
            stop1_pos, stop1_color = sorted_stops[i]
            stop2_pos, stop2_color = sorted_stops[i + 1]
            
            if stop1_pos <= normalized <= stop2_pos:
                # Interpolate between these two stops
                if stop2_pos == stop1_pos:
                    return stop1_color
                
                t = (normalized - stop1_pos) / (stop2_pos - stop1_pos)
                
                # Convert hex to RGB
                color1_rgb = tuple(int(stop1_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                color2_rgb = tuple(int(stop2_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                
                # Interpolate
                r = int(color1_rgb[0] + (color2_rgb[0] - color1_rgb[0]) * t)
                g = int(color1_rgb[1] + (color2_rgb[1] - color1_rgb[1]) * t)
                b = int(color1_rgb[2] + (color2_rgb[2] - color1_rgb[2]) * t)
                
                return f"#{r:02x}{g:02x}{b:02x}"
        
        return self.theme.get("accent_blue", "#4a6fae")

    def _on_mouse_down(self, event):
        """Handle mouse button press for starting drag."""
        if self.mode != "continuous":
            return

        # Check if clicked on a gradient handle
        items = self.canvas.find_withtag("gradient_handle")
        for item in items:
            coords = self.canvas.coords(item)
            if len(coords) >= 2:
                # For polygon (triangle), get center x from first point
                center_x = coords[0]
                center_y = coords[1] if len(coords) == 6 else (coords[1] + coords[3]) / 2

                # Check if click is near this handle
                dist = ((event.x - center_x)**2 + (event.y - center_y)**2)**0.5
                if dist < 15:  # 15 pixel tolerance for triangle
                    self.dragging_stop = item
                    self.drag_start_x = event.x
                    self._is_dragging = True
                    self.canvas.config(cursor="sb_h_double_arrow")
                    return

    def _on_mouse_drag(self, event):
        """Handle mouse drag to reposition gradient stop with overlap prevention."""
        if self.mode != "continuous" or not self.dragging_stop:
            return

        # Get the gradient stop info
        if self.dragging_stop not in self.gradient_stop_handles:
            return

        old_value, stop_color = self.gradient_stop_handles[self.dragging_stop]

        # Calculate new value based on x position (horizontal)
        # Constrain to gradient bar area
        rel_x = event.x - self.gradient_bar_x
        rel_x = max(0, min(self.gradient_bar_width, rel_x))

        # Calculate normalized position (0-1)
        normalized_pos = rel_x / self.gradient_bar_width if self.gradient_bar_width > 0 else 0.5

        # Calculate new value
        gradient_range = self.gradient_max - self.gradient_min
        new_value = self.gradient_min + normalized_pos * gradient_range

        # Find the index of the stop being dragged
        drag_index = -1
        for i, (val, col) in enumerate(self.gradient_stops):
            if abs(val - old_value) < 0.0001 and col == stop_color:
                drag_index = i
                break

        if drag_index == -1:
            return

        # Sort stops by value to find neighbors
        sorted_stops = sorted(enumerate(self.gradient_stops), key=lambda x: x[1][0])
        sorted_index = next((i for i, (idx, _) in enumerate(sorted_stops) if idx == drag_index), -1)

        if sorted_index == -1:
            return

        # Get constraints from neighbors
        min_constraint = self.gradient_min
        max_constraint = self.gradient_max

        # Check stop before (lower value)
        if sorted_index > 0:
            prev_idx, (prev_val, _) = sorted_stops[sorted_index - 1]
            min_constraint = prev_val + 0.001

        # Check stop after (higher value)
        if sorted_index < len(sorted_stops) - 1:
            next_idx, (next_val, _) = sorted_stops[sorted_index + 1]
            max_constraint = next_val - 0.001

        # Clamp new value to constraints
        new_value = max(min_constraint, min(max_constraint, new_value))

        # Update the gradient stop value
        self.gradient_stops[drag_index] = (new_value, stop_color)
        self.gradient_stop_handles[self.dragging_stop] = (new_value, stop_color)

        # Update handle position directly (fast path - no full redraw)
        self._update_handle_position(self.dragging_stop, new_value)

        # Update histogram bar colors efficiently
        self._update_histogram_colors()

        # Debounced callback notification
        self._schedule_debounced_callback()

    def _update_handle_position(self, handle_id: int, new_value: float):
        """Move a gradient handle to a new position without full redraw."""
        gradient_range = self.gradient_max - self.gradient_min
        if gradient_range <= 0:
            return

        normalized_pos = (new_value - self.gradient_min) / gradient_range
        normalized_pos = max(0.0, min(1.0, normalized_pos))

        new_x = self.gradient_bar_x + (normalized_pos * self.gradient_bar_width)
        new_y = self.gradient_bar_y + self.gradient_bar_height / 2

        # Update triangle coordinates
        handle_size = 8
        new_coords = [
            new_x, self.gradient_bar_y - 2,  # Top point
            new_x - handle_size, self.gradient_bar_y + self.gradient_bar_height + 4,  # Bottom left
            new_x + handle_size, self.gradient_bar_y + self.gradient_bar_height + 4,  # Bottom right
        ]
        self.canvas.coords(handle_id, *new_coords)

    def _update_histogram_colors(self):
        """Update histogram bar colors based on current gradient stops."""
        for bar_id, bin_center in self._histogram_bar_ids:
            new_color = self._get_color_for_value_continuous(bin_center)
            self.canvas.itemconfig(bar_id, fill=new_color)

        # Also update gradient segments
        if self._gradient_segment_ids:
            gradient_range = self.gradient_max - self.gradient_min
            if gradient_range > 0:
                num_segments = len(self._gradient_segment_ids)
                for i, seg_id in enumerate(self._gradient_segment_ids):
                    t = i / num_segments
                    value = self.gradient_min + t * gradient_range
                    color = self._interpolate_gradient_color(value)
                    self.canvas.itemconfig(seg_id, fill=color)

    def _on_mouse_up(self, event):
        """Handle mouse button release to end drag."""
        if self.dragging_stop:
            self.dragging_stop = None
            self.drag_start_x = None
            self._is_dragging = False
            self.canvas.config(cursor="")

            # Cancel pending timers
            if self._redraw_timer:
                self.after_cancel(self._redraw_timer)
                self._redraw_timer = None
            if self._callback_timer:
                self.after_cancel(self._callback_timer)
                self._callback_timer = None

            self._redraw_pending = False
            self._callback_pending = False

            # Final callback and redraw
            if self.on_gradient_changed_callback:
                self.on_gradient_changed_callback()
            self.redraw()

    def _schedule_debounced_redraw(self):
        """
        Schedule a debounced redraw to prevent performance bottleneck during drag.
        """
        if self._redraw_pending:
            return

        self._redraw_pending = True

        def do_redraw():
            self._redraw_pending = False
            self._redraw_timer = None
            if self._is_dragging:
                # During drag, just update colors (already done in _on_mouse_drag)
                pass
            else:
                self.redraw()

        if self._redraw_timer:
            self.after_cancel(self._redraw_timer)

        self._redraw_timer = self.after(self._redraw_debounce_ms, do_redraw)

    def _schedule_debounced_callback(self):
        """Schedule a debounced callback notification."""
        if self._callback_pending:
            return

        self._callback_pending = True

        def do_callback():
            self._callback_pending = False
            self._callback_timer = None
            if self.on_gradient_changed_callback:
                self.on_gradient_changed_callback()

        if self._callback_timer:
            self.after_cancel(self._callback_timer)

        self._callback_timer = self.after(100, do_callback)  # 100ms debounce for callback

    def _draw_axis(self, left: float, y: float, right: float):
        """
        Draw axis line and min/max labels below the gradient bar.

        Args:
            left: Left edge x coordinate
            y: Y coordinate for axis labels
            right: Right edge x coordinate
        """
        # Draw min label
        self.canvas.create_text(
            left,
            y,
            text=f"{self.data_min:.2f}",
            fill=self.theme["text"],
            anchor="nw",
            font=("Arial", 9),
        )

        # Draw mid label
        mid_value = (self.data_min + self.data_max) / 2
        self.canvas.create_text(
            (left + right) / 2,
            y,
            text=f"{mid_value:.2f}",
            fill=self.theme["text"],
            anchor="n",
            font=("Arial", 9),
        )

        # Draw max label
        self.canvas.create_text(
            right,
            y,
            text=f"{self.data_max:.2f}",
            fill=self.theme["text"],
            anchor="ne",
            font=("Arial", 9),
        )


# ============================================================================
# Color Map Editor Dialog
# ============================================================================

class ColorMapEditorDialog:
    """
    Modal dialog for editing color maps.
    
    Supports two modes:
    - Discrete: Table of numeric ranges with min/max/color
    - Continuous: Gradient with curve points
    
    Features:
    - Live histogram preview with actual data
    - Generate intervals/curve from data
    - Integration with ColorMapManager
    - Proper theming via gui_manager
    """
    
    def __init__(
        self,
        parent,
        gui_manager,
        color_map_manager,
        data_column: Optional[str] = None,
        data_values: Optional[List[float]] = None,
        initial_color_map=None,
        on_save_callback: Optional[Callable] = None,
        data_coordinator=None,  # NEW: DataCoordinator for fetching column data
    ):
        """
        Initialize the color map editor dialog.
        
        Args:
            parent: Parent window
            gui_manager: GUIManager for theming
            color_map_manager: ColorMapManager instance
            data_column: Optional column name for context
            data_values: Optional list of data values for histogram
            initial_color_map: Optional ColorMap to edit
            on_save_callback: Optional callback when map is saved
        """
        print(f"\n{'='*70}")
        print(f"DEBUG: [ColorMapEditorDialog] Initializing")
        print(f"DEBUG: [ColorMapEditorDialog]   data_column: {data_column}")
        print(f"DEBUG: [ColorMapEditorDialog]   data_values: {len(data_values) if data_values else 0} values")
        print(f"DEBUG: [ColorMapEditorDialog]   initial_color_map: {initial_color_map.name if initial_color_map else 'None'}")
        print(f"{'='*70}\n")
        
        self.parent = parent
        self.gui_manager = gui_manager
        self.theme = gui_manager.theme_colors
        self.color_map_manager = color_map_manager
        self.data_column = data_column
        
        # Fetch data from DataCoordinator if available and no data_values provided
        if data_values is None and data_coordinator is not None and data_column:
            try:
                # Get column values from geological store
                data_values = data_coordinator.get_column_values(data_column)
                print(f"DEBUG: [ColorMapEditorDialog] Fetched {len(data_values) if data_values else 0} values from DataCoordinator for '{data_column}'")
            except Exception as e:
                print(f"DEBUG: [ColorMapEditorDialog] Could not fetch data from DataCoordinator: {e}")
                data_values = None
        
        self.data_coordinator = data_coordinator
        
        # Detect if data is categorical (text) or numeric
        self.is_categorical = False
        if data_values:
            # Check if data contains non-numeric values
            sample = data_values[:100] if len(data_values) > 100 else data_values
            numeric_count = sum(1 for v in sample if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)))
            text_count = sum(1 for v in sample if isinstance(v, str))
            
            if text_count > numeric_count:
                self.is_categorical = True
                # For categorical, just keep unique values
                self.data_values = list(set(str(v) for v in data_values if v is not None and not (isinstance(v, float) and np.isnan(v))))
                print(f"DEBUG: [ColorMapEditorDialog] Detected CATEGORICAL data with {len(self.data_values)} unique values")
            else:
                # For numeric, FILTER OUT NaN values (don't convert to 0 - that corrupts ranges)
                self.data_values = []
                nan_count = 0
                for v in data_values:
                    if isinstance(v, float) and np.isnan(v):
                        nan_count += 1
                        # Skip NaN values - don't include in range calculations
                        continue
                    elif isinstance(v, (int, float)):
                        self.data_values.append(float(v))
                print(f"DEBUG: [ColorMapEditorDialog] Detected NUMERIC data with {len(self.data_values)} valid values ({nan_count} NaN values filtered out)")
        else:
            self.data_values = []
            
        self.initial_color_map = initial_color_map
        self.on_save_callback = on_save_callback
        
        # Dialog state
        self.dialog = None
        self.result = None
        
        # UI components
        self.preview_widget = None
        self.config_container = None
        
        # Mode state
        self.mode_var = None
        self.name_var = None
        self.desc_var = None
        
        # Discrete mode state
        self.discrete_rows: List[Dict[str, Any]] = []
        self.discrete_list_frame = None
        
        # Continuous mode state
        self.gradient_preset_var = None
        self.invert_var = None
        self.dynamic_var = None
        self.min_var = None
        self.max_var = None
        self.gradient_stops: List[Tuple[float, str]] = []  # (value, hex_color)
        
        print(f"DEBUG: [ColorMapEditorDialog] State initialized")
        
        self._create_dialog()
    
    def _normalize_column_name(self, column_name: str) -> str:
        """
        Normalize column name for use as color map preset name.
        Converts to lowercase, replaces spaces/special chars with underscores.
        
        Args:
            column_name: Original column name
            
        Returns:
            Normalized name suitable for file/preset naming
        """
        if not column_name:
            return "new_color_map"
        
        import re
        # Convert to lowercase
        normalized = column_name.lower()
        # Replace spaces and special characters with underscores
        normalized = re.sub(r'[^\w]+', '_', normalized)
        # Remove leading/trailing underscores
        normalized = normalized.strip('_')
        # Replace multiple underscores with single
        normalized = re.sub(r'_+', '_', normalized)
        
        return normalized if normalized else "new_color_map"
        
    def _create_dialog(self):
        """Create the main dialog window."""
        print(f"DEBUG: [ColorMapEditorDialog] Creating dialog window")
        
        # Create dialog using DialogHelper
        self.dialog = DialogHelper.create_dialog(
            self.parent,
            "Edit Color Map",
            modal=True,
            topmost=True,
        )
        
        # Apply theme
        self.dialog.configure(bg=self.theme["background"])
        
        print(f"DEBUG: [ColorMapEditorDialog] Dialog created")
        
        # Build UI
        self._build_ui()
        
        # Load initial data
        self._load_initial_color_map()
        
        # Center dialog
        self.dialog.update_idletasks()
        DialogHelper.center_dialog(
            self.dialog,
            self.parent,
            min_width=900,
            min_height=600,
        )
        
        print(f"DEBUG: [ColorMapEditorDialog] Dialog ready")
        
    def _build_ui(self):
        """Build the dialog UI layout."""
        print(f"DEBUG: [ColorMapEditorDialog] Building UI layout")
        
        # Main container with left/right panels
        main_container = tk.Frame(self.dialog, bg=self.theme["background"])
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configure grid
        main_container.columnconfigure(0, weight=0, minsize=300)  # Left panel (fixed width)
        main_container.columnconfigure(1, weight=1, minsize=600)  # Right panel (expands)
        main_container.rowconfigure(0, weight=1)
        
        # LEFT PANEL: Configuration controls
        self._build_left_panel(main_container)
        
        # RIGHT PANEL: Histogram preview
        self._build_right_panel(main_container)
        
        # BOTTOM: Dialog buttons
        self._build_bottom_buttons()
        
        print(f"DEBUG: [ColorMapEditorDialog] UI layout complete")
        
    def _build_left_panel(self, parent):
        """Build the left configuration panel."""
        print(f"DEBUG: [ColorMapEditorDialog] Building left panel")
        
        left_panel = tk.Frame(
            parent,
            bg=self.theme["secondary_bg"],
            highlightbackground=self.theme["border"],
            highlightthickness=1,
        )
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        # Title section
        title_frame = tk.Frame(left_panel, bg=self.theme["secondary_bg"])
        title_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        ttk.Label(
            title_frame,
            text="Color Map Configuration",
            font=("Arial", 11, "bold"),
        ).pack(anchor=tk.W)
        
        # Name field
        name_frame = tk.Frame(left_panel, bg=self.theme["secondary_bg"])
        name_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(name_frame, text="Name:", width=12).pack(side=tk.LEFT)
        
        # Use normalized column name if available, otherwise generic name
        default_name = self._normalize_column_name(self.data_column) if self.data_column else "new_color_map"
        self.name_var = tk.StringVar(value=default_name)
        name_entry = tk.Entry(
            name_frame,
            textvariable=self.name_var,
            bg=self.theme["field_bg"],
            fg=self.theme["text"],
            font=("Arial", 10),
        )
        name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Description field
        desc_frame = tk.Frame(left_panel, bg=self.theme["secondary_bg"])
        desc_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(desc_frame, text="Description:", width=12).pack(side=tk.LEFT)
        
        self.desc_var = tk.StringVar(value="")
        desc_entry = tk.Entry(
            desc_frame,
            textvariable=self.desc_var,
            bg=self.theme["field_bg"],
            fg=self.theme["text"],
            font=("Arial", 10),
        )
        desc_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Mode selector
        mode_frame = tk.Frame(left_panel, bg=self.theme["secondary_bg"])
        mode_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(mode_frame, text="Type:", width=12).pack(side=tk.LEFT)
        
        self.mode_var = tk.StringVar(value="discrete")
        
        # Use styled dropdown
        mode_combo_frame = tk.Frame(
            mode_frame,
            bg=self.theme["field_bg"],
            highlightbackground=self.theme["field_border"],
            highlightthickness=1,
            bd=0,
        )
        mode_combo_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        mode_dropdown = tk.OptionMenu(
            mode_combo_frame,
            self.mode_var,
            "discrete",
            "continuous",
            command=self._on_mode_change,
        )
        self.gui_manager.style_dropdown(mode_dropdown, width=15)
        mode_dropdown.pack()
        
        # Container for mode-specific controls
        self.config_container = tk.Frame(left_panel, bg=self.theme["secondary_bg"])
        self.config_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Build both mode panels
        self._build_discrete_panel()
        self._build_continuous_panel()
        
        # Show initial mode
        self._show_mode_panel()
        
        print(f"DEBUG: [ColorMapEditorDialog] Left panel complete")
        
    def _build_discrete_panel(self):
        """Build the discrete mode configuration panel."""
        print(f"DEBUG: [ColorMapEditorDialog] Building discrete panel")
        
        self.discrete_panel = tk.Frame(
            self.config_container,
            bg=self.theme["secondary_bg"],
        )
        
        # Header
        header_frame = tk.Frame(self.discrete_panel, bg=self.theme["secondary_bg"])
        header_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(header_frame, text="Intervals", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        # Table header - store reference for dynamic rebuilding
        self.table_header_frame = tk.Frame(self.discrete_panel, bg=self.theme["secondary_bg"])
        self.table_header_frame.pack(fill=tk.X, pady=(0, 2))
        
        # Build initial header
        self._rebuild_table_header()
        
        # Scrollable list of ranges
        list_container = tk.Frame(self.discrete_panel, bg=self.theme["secondary_bg"])
        list_container.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = tk.Scrollbar(list_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        canvas = tk.Canvas(
            list_container,
            bg=self.theme["secondary_bg"],
            highlightthickness=0,
            yscrollcommand=scrollbar.set,
        )
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=canvas.yview)
        
        self.discrete_list_frame = tk.Frame(canvas, bg=self.theme["secondary_bg"])
        canvas.create_window((0, 0), window=self.discrete_list_frame, anchor="nw")
        
        def _configure_scroll(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        self.discrete_list_frame.bind("<Configure>", _configure_scroll)
        
        # Buttons
        button_frame = tk.Frame(self.discrete_panel, bg=self.theme["secondary_bg"])
        button_frame.pack(fill=tk.X, pady=(5, 0))
        
        ModernButton(
            button_frame,
            text="+ Add Range",
            color=self.theme["accent_blue"],
            command=self._add_discrete_row,
            theme_colors=self.theme,
        ).pack(side=tk.LEFT, padx=2)
        
        ModernButton(
            button_frame,
            text="Generate Intervals",
            color=self.theme["accent_green"],
            command=self._generate_intervals,
            theme_colors=self.theme,
        ).pack(side=tk.LEFT, padx=2)
        
        print(f"DEBUG: [ColorMapEditorDialog] Discrete panel built")
        
    def _build_continuous_panel(self):
        """Build the continuous mode configuration panel."""
        print(f"DEBUG: [ColorMapEditorDialog] Building continuous panel")
        
        self.continuous_panel = tk.Frame(
            self.config_container,
            bg=self.theme["secondary_bg"],
        )
        
        # Gradient preset selector
        preset_frame = tk.Frame(self.continuous_panel, bg=self.theme["secondary_bg"])
        preset_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(preset_frame, text="Gradient:", width=12).pack(side=tk.LEFT, padx=(0, 5))
        
        self.gradient_preset_var = tk.StringVar(value="Rainbow (bgyr)")
        
        gradient_combo_frame = tk.Frame(
            preset_frame,
            bg=self.theme["field_bg"],
            highlightbackground=self.theme["field_border"],
            highlightthickness=1,
            bd=0,
        )
        gradient_combo_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        gradient_dropdown = tk.OptionMenu(
            gradient_combo_frame,
            self.gradient_preset_var,
            "Rainbow (bgyr)",
            "Viridis",
            "Plasma",
            "Linear (gow)",
            "Diverging (bwr)",
        )
        self.gui_manager.style_dropdown(gradient_dropdown, width=20)
        gradient_dropdown.pack()
        
        # Invert checkbox
        self.invert_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            preset_frame,
            text="Invert",
            variable=self.invert_var,
            command=self._on_gradient_changed,
        ).pack(side=tk.LEFT, padx=(5, 0))
        
        # Limits section
        limits_frame = tk.Frame(self.continuous_panel, bg=self.theme["secondary_bg"])
        limits_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(limits_frame, text="Limits", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 3))
        
        # Dynamic checkbox
        self.dynamic_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            limits_frame,
            text="Dynamic",
            variable=self.dynamic_var,
            command=self._on_dynamic_toggle,
        ).pack(anchor=tk.W)
        
        # Min/Max entries
        minmax_frame = tk.Frame(limits_frame, bg=self.theme["secondary_bg"])
        minmax_frame.pack(fill=tk.X, pady=(3, 0))
        
        ttk.Label(minmax_frame, text="Minimum:", width=10).pack(side=tk.LEFT)
        
        self.min_var = tk.DoubleVar(value=0.0)
        self.min_entry = tk.Entry(
            minmax_frame,
            textvariable=self.min_var,
            width=10,
            bg=self.theme["field_bg"],
            fg=self.theme["text"],
            state="disabled",
        )
        self.min_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(minmax_frame, text="Maximum:", width=10).pack(side=tk.LEFT, padx=(10, 0))
        
        self.max_var = tk.DoubleVar(value=1.0)
        self.max_entry = tk.Entry(
            minmax_frame,
            textvariable=self.max_var,
            width=10,
            bg=self.theme["field_bg"],
            fg=self.theme["text"],
            state="disabled",
        )
        self.max_entry.pack(side=tk.LEFT, padx=5)
        
        # Set limits button
        ModernButton(
            limits_frame,
            text="Set Limits From Data",
            color=self.theme["accent_blue"],
            command=self._set_limits_from_data,
            theme_colors=self.theme,
        ).pack(anchor=tk.W, pady=(5, 0))
        
        # Curve points section
        curve_frame = tk.Frame(self.continuous_panel, bg=self.theme["secondary_bg"])
        curve_frame.pack(fill=tk.X, pady=(10, 5))
        
        ttk.Label(curve_frame, text="Curve Points", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 3))
        
        # Curve points list (simplified for now)
        self.curve_list_label = ttk.Label(
            curve_frame,
            text="No curve points defined",
            font=("Arial", 9),
        )
        self.curve_list_label.pack(anchor=tk.W, pady=3)
        
        # Curve buttons
        curve_btn_frame = tk.Frame(curve_frame, bg=self.theme["secondary_bg"])
        curve_btn_frame.pack(fill=tk.X, pady=(3, 0))
        
        ModernButton(
            curve_btn_frame,
            text="Add Point",
            color=self.theme["accent_blue"],
            command=self._add_gradient_stop,
            theme_colors=self.theme,
        ).pack(side=tk.LEFT, padx=2)
        
        ModernButton(
            curve_btn_frame,
            text="Generate Curve",
            color=self.theme["accent_green"],
            command=self._generate_curve,
            theme_colors=self.theme,
        ).pack(side=tk.LEFT, padx=2)
        
        print(f"DEBUG: [ColorMapEditorDialog] Continuous panel built")
        
    def _build_right_panel(self, parent):
        """Build the right preview panel."""
        print(f"DEBUG: [ColorMapEditorDialog] Building right panel (preview)")
        
        right_panel = tk.Frame(
            parent,
            bg=self.theme["background"],
            highlightbackground=self.theme["border"],
            highlightthickness=1,
        )
        right_panel.grid(row=0, column=1, sticky="nsew")
        
        # Title
        title_frame = tk.Frame(right_panel, bg=self.theme["background"])
        title_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        ttk.Label(
            title_frame,
            text="Preview",
            font=("Arial", 11, "bold"),
        ).pack(anchor=tk.W)
        
        # Column selector row (if data_coordinator available)
        if self.data_coordinator:
            column_frame = tk.Frame(title_frame, bg=self.theme["background"])
            column_frame.pack(fill=tk.X, pady=(5, 0))
            
            ttk.Label(column_frame, text="Data Column:").pack(side=tk.LEFT, padx=(0, 5))
            
            # Get available columns
            available_columns = self._get_available_columns()
            
            self.column_selector_var = tk.StringVar(value=self.data_column or "")
            
            column_combo_frame = tk.Frame(
                column_frame,
                bg=self.theme["field_bg"],
                highlightbackground=self.theme["field_border"],
                highlightthickness=1,
                bd=0,
            )
            column_combo_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            column_dropdown = tk.OptionMenu(
                column_combo_frame,
                self.column_selector_var,
                *available_columns if available_columns else ["No columns available"],
                command=self._on_column_changed,
            )
            self.gui_manager.style_dropdown(column_dropdown, width=25)
            column_dropdown.pack()
        else:
            # Just show current column as label
            if self.data_column:
                ttk.Label(
                    title_frame,
                    text=f"Data: {self.data_column}",
                    font=("Arial", 9),
                ).pack(anchor=tk.W)
        
        # Preview widget
        self.preview_widget = HistogramPreviewWidget(
            right_panel,
            self.gui_manager,
            width=600,
            height=400,
        )
        self.preview_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Set callback for when gradient is changed by dragging
        self.preview_widget.on_gradient_changed_callback = self._on_gradient_dragged
        
        # Set data
        if self.data_values:
            self.preview_widget.set_data(self.data_values, is_categorical=self.is_categorical)
        
        print(f"DEBUG: [ColorMapEditorDialog] Right panel complete")
    
    def _get_available_columns(self) -> List[str]:
        """Get list of available data columns from DataCoordinator."""
        if not self.data_coordinator:
            return []
        
        try:
            columns = self.data_coordinator.get_available_columns()
            # Return just the column names (not the data types)
            return [col_name for col_name, col_type in columns]
        except Exception as e:
            print(f"DEBUG: [ColorMapEditorDialog] Error getting columns: {e}")
            return []
    
    def _on_column_changed(self, *args):
        """Handle column selection change."""
        if not self.data_coordinator:
            return
        
        new_column = self.column_selector_var.get()
        if not new_column or new_column == "No columns available":
            return
        
        print(f"DEBUG: [ColorMapEditorDialog] Column changed to: {new_column}")
        
        try:
            # Fetch new column data
            new_data = self.data_coordinator.get_column_values(new_column)
            
            if new_data:
                self.data_column = new_column
                
                # Re-detect if categorical
                sample = new_data[:100] if len(new_data) > 100 else new_data
                numeric_count = sum(1 for v in sample if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)))
                text_count = sum(1 for v in sample if isinstance(v, str))
                
                if text_count > numeric_count:
                    self.is_categorical = True
                    self.data_values = list(set(str(v) for v in new_data if v is not None and not (isinstance(v, float) and np.isnan(v))))
                else:
                    self.is_categorical = False
                    self.data_values = [float(v) for v in new_data if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v))]
                
                print(f"DEBUG: [ColorMapEditorDialog] Loaded {len(self.data_values)} values, categorical={self.is_categorical}")
                
                # Update preview
                self.preview_widget.set_data(self.data_values, is_categorical=self.is_categorical)
                
                # Update name suggestion
                self.name_var.set(self._normalize_column_name(new_column))
                
                # Regenerate intervals if in discrete mode
                if self.mode_var.get() == "discrete":
                    self._generate_intervals()
                else:
                    self._set_limits_from_data()
                    
        except Exception as e:
            print(f"DEBUG: [ColorMapEditorDialog] Error loading column data: {e}")
        
    def _build_bottom_buttons(self):
        """Build bottom dialog buttons."""
        print(f"DEBUG: [ColorMapEditorDialog] Building bottom buttons")
        
        button_frame = tk.Frame(self.dialog, bg=self.theme["background"])
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(5, 10))
        
        # Right-aligned buttons
        right_buttons = tk.Frame(button_frame, bg=self.theme["background"])
        right_buttons.pack(side=tk.RIGHT)
        
        ModernButton(
            right_buttons,
            text="Apply",
            color=self.theme["accent_green"],
            command=self._on_apply,
            theme_colors=self.theme,
        ).pack(side=tk.LEFT, padx=5)
        
        ModernButton(
            right_buttons,
            text="Close",
            color=self.theme["secondary_bg"],
            command=self._on_close,
            theme_colors=self.theme,
        ).pack(side=tk.LEFT, padx=5)
        
        print(f"DEBUG: [ColorMapEditorDialog] Bottom buttons complete")
        
    def _rebuild_table_header(self):
        """Rebuild table header based on data type."""
        # Clear existing header
        for widget in self.table_header_frame.winfo_children():
            widget.destroy()
        
        # Dynamic header based on data type
        if self.is_categorical:
            header_cols = [
                ("Show", 5),
                ("Category", 25),
                ("Color", 8),
                ("Data %", 8),
            ]
        else:
            header_cols = [
                ("Show", 5),
                ("Min", 10),
                ("Max", 10),
                ("Color", 8),
                ("Data %", 8),
            ]
        
        for text, width in header_cols:
            ttk.Label(self.table_header_frame, text=text, width=width).pack(side=tk.LEFT, padx=2)

    # ========================================================================
    # Mode Management
    # ========================================================================
    
    def _show_mode_panel(self):
        """Show the appropriate panel for current mode."""
        current_mode = self.mode_var.get()
        print(f"DEBUG: [ColorMapEditorDialog] _show_mode_panel: mode={current_mode}")
        
        # Hide both panels
        self.discrete_panel.pack_forget()
        self.continuous_panel.pack_forget()
        
        # Show selected panel
        if current_mode == "discrete":
            self.discrete_panel.pack(fill=tk.BOTH, expand=True)
        else:
            self.continuous_panel.pack(fill=tk.BOTH, expand=True)
        
        print(f"DEBUG: [ColorMapEditorDialog] Mode panel switched to {current_mode}")
        
    def _on_mode_change(self, *args):
        """Handle mode change event."""
        print(f"DEBUG: [ColorMapEditorDialog] _on_mode_change triggered: {self.mode_var.get()}")
        self._show_mode_panel()
        self._refresh_preview()
        
    # ========================================================================
    # Discrete Mode Methods
    # ========================================================================
    
    def _add_discrete_row(self):
        """Add a new discrete range row."""
        print(f"DEBUG: [ColorMapEditorDialog] _add_discrete_row called")
        
        # For categorical data, prompt for category name or use unused category
        if self.is_categorical:
            # Find categories not yet used
            used_categories = set(row.get("category") for row in self.discrete_rows if row.get("category"))
            available_categories = [c for c in self.data_values if c not in used_categories]
            
            if available_categories:
                # Use next available category
                category = available_categories[0]
            else:
                # All categories used, create new placeholder
                category = f"Category_{len(self.discrete_rows) + 1}"
            
            next_idx = len(self.discrete_rows)
            
            # Generate a distinct color using HSV
            import colorsys
            hue = (next_idx * 0.618033988749895) % 1.0  # Golden ratio for good distribution
            saturation = 0.7
            value = 0.9
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            
            row_data = {
                "enabled": tk.BooleanVar(value=True),
                "min": tk.DoubleVar(value=float(next_idx)),
                "max": tk.DoubleVar(value=float(next_idx + 1)),
                "color_bgr": (int(b * 255), int(g * 255), int(r * 255)),
                "category": category,  # Store the category name
            }
            
            print(f"DEBUG: [ColorMapEditorDialog] Added categorical row {next_idx}: {category}")
        else:
            # For numeric data
            if self.data_values:
                # Smart default: start where last range ended
                if self.discrete_rows:
                    max_vals = [row["max"].get() for row in self.discrete_rows if row["enabled"].get()]
                    if max_vals:
                        data_min = round(max(max_vals), 3)
                        data_max = round(data_min + 10.0, 3)
                    else:
                        data_min = round(float(min(self.data_values)), 3)
                        data_max = round(float(max(self.data_values)), 3)
                else:
                    data_min = round(float(min(self.data_values)), 3)
                    data_max = round(float(max(self.data_values)), 3)
            else:
                data_min, data_max = 0.0, 1.0
            
            row_data = {
                "enabled": tk.BooleanVar(value=True),
                "min": tk.DoubleVar(value=data_min),
                "max": tk.DoubleVar(value=data_max),
                "color_bgr": (128, 128, 128),  # Store as BGR
            }
            
            print(f"DEBUG: [ColorMapEditorDialog] Added numeric row: {data_min:.3f}-{data_max:.3f}")
        
        self.discrete_rows.append(row_data)
        
        self._rebuild_discrete_rows()
        self._refresh_preview()
        
    def _rebuild_discrete_rows(self):
        """Rebuild the discrete range rows UI."""
        print(f"DEBUG: [ColorMapEditorDialog] _rebuild_discrete_rows: {len(self.discrete_rows)} rows")
        
        # Rebuild header in case data type changed
        if hasattr(self, '_rebuild_table_header'):
            self._rebuild_table_header()
        
        # Clear existing rows
        for widget in self.discrete_list_frame.winfo_children():
            widget.destroy()
        
        # Rebuild each row
        for idx, row_data in enumerate(self.discrete_rows):
            self._create_discrete_row_widget(idx, row_data)
        
        print(f"DEBUG: [ColorMapEditorDialog] Discrete rows rebuilt")


    def _create_discrete_row_widget(self, idx: int, row_data: Dict[str, Any]):
        """Create UI widget for a discrete range row."""
        row_frame = tk.Frame(
            self.discrete_list_frame,
            bg=self.theme["background"],
            highlightbackground=self.theme["border"],
            highlightthickness=1,
        )
        row_frame.pack(fill=tk.X, pady=2)
        
        # Enabled checkbox
        tk.Checkbutton(
            row_frame,
            variable=row_data["enabled"],
            bg=self.theme["background"],
            activebackground=self.theme["background"],
            selectcolor=self.theme.get("checkbox_bg", "#4a4a4a"),
            command=self._refresh_preview,
        ).pack(side=tk.LEFT, padx=2)
        
        # Check if this is a categorical row (has category name)
        is_categorical = "category" in row_data and row_data["category"] is not None
        
        if is_categorical:
            # Display category name as a label
            category_label = ttk.Label(
                row_frame,
                text=row_data["category"],
                width=25,
                background=self.theme["field_bg"],
                foreground=self.theme["text"],
            )
            category_label.pack(side=tk.LEFT, padx=2)
        else:
            # Display min/max entries for numeric data
            # Min entry
            min_entry = tk.Entry(
                row_frame,
                textvariable=row_data["min"],
                width=10,
                bg=self.theme["field_bg"],
                fg=self.theme["text"],
            )
            min_entry.pack(side=tk.LEFT, padx=2)
            min_entry.bind("<Return>", lambda e: self._refresh_preview())
            
            # Max entry
            max_entry = tk.Entry(
                row_frame,
                textvariable=row_data["max"],
                width=10,
                bg=self.theme["field_bg"],
                fg=self.theme["text"],
            )
            max_entry.pack(side=tk.LEFT, padx=2)
            max_entry.bind("<Return>", lambda e: self._refresh_preview())
        
        # Color swatch
        hex_color = bgr_to_hex(row_data["color_bgr"])
        swatch = tk.Canvas(
            row_frame,
            width=30,
            height=16,
            bg=hex_color,
            highlightbackground=self.theme["border"],
            highlightthickness=1,
        )
        swatch.pack(side=tk.LEFT, padx=2)
        swatch.bind("<Button-1>", lambda e, i=idx: self._pick_row_color(i))
        swatch.config(cursor="hand2")
        
        # Store swatch reference for updates
        row_data["swatch"] = swatch
        
        # Data percentage label
        if is_categorical:
            # For categorical, count exact matches
            data_pct = self._calculate_category_percentage(row_data["category"])
        else:
            # For numeric, use range
            data_pct = self._calculate_data_percentage(
                row_data["min"].get(),
                row_data["max"].get(),
            )
        
        pct_label = ttk.Label(row_frame, text=f"{data_pct:.1f}%", width=6)
        pct_label.pack(side=tk.LEFT, padx=2)
        row_data["pct_label"] = pct_label
        
        # Remove button
        ModernButton(
            row_frame,
            text="✖",
            color=self.theme["accent_red"],
            command=lambda i=idx: self._remove_discrete_row(i),
            theme_colors=self.theme,
        ).pack(side=tk.LEFT, padx=2)

    def _calculate_category_percentage(self, category: str) -> float:
        """Calculate what percentage of data matches a category."""
        if not self.data_values:
            return 0.0
        
        count = sum(1 for v in self.data_values if str(v) == category)
        return (count / len(self.data_values)) * 100.0

    def _remove_discrete_row(self, idx: int):
        """Remove a discrete range row."""
        print(f"DEBUG: [ColorMapEditorDialog] _remove_discrete_row: idx={idx}")
        
        if 0 <= idx < len(self.discrete_rows):
            self.discrete_rows.pop(idx)
            self._rebuild_discrete_rows()
            self._refresh_preview()
        
    def _pick_row_color(self, idx: int):
        """Open color picker for a discrete row."""
        print(f"DEBUG: [ColorMapEditorDialog] _pick_row_color: idx={idx}")
        
        if idx >= len(self.discrete_rows):
            return
        
        row_data = self.discrete_rows[idx]
        
        # Get current color
        current_hex = bgr_to_hex(row_data["color_bgr"])
        
        # Open color chooser
        color_result = colorchooser.askcolor(color=current_hex, parent=self.dialog)
        
        if color_result[1]:  # If user didn't cancel
            new_hex = color_result[1]
            new_bgr = hex_to_bgr(new_hex)
            
            print(f"DEBUG: [ColorMapEditorDialog] Color picked: {new_hex} (BGR: {new_bgr})")
            
            # Update row data
            row_data["color_bgr"] = new_bgr
            
            # Update swatch
            if "swatch" in row_data:
                row_data["swatch"].config(bg=new_hex)
            
            self._refresh_preview()
        
    def _calculate_data_percentage(self, min_val: float, max_val: float) -> float:
        """Calculate what percentage of data falls in a range."""
        if not self.data_values:
            return 0.0
        
        # For categorical data, match by index
        if self.is_categorical:
            # min_val and max_val are indices (e.g., 0-1, 1-2, etc.)
            idx = int(min_val)
            if 0 <= idx < len(self.data_values):
                # Each category gets equal representation
                return (1.0 / len(self.data_values)) * 100.0
            return 0.0
        
        # For numeric data, count values in range
        count = sum(1 for v in self.data_values if min_val <= v < max_val)
        return (count / len(self.data_values)) * 100.0
        
    def _generate_intervals(self):
        """Auto-generate discrete intervals from data."""
        print(f"DEBUG: [ColorMapEditorDialog] _generate_intervals called")
        
        if not self.data_values:
            print(f"DEBUG: [ColorMapEditorDialog] No data values, cannot generate")
            DialogHelper.show_message(
                self.dialog,
                "No Data",
                "No data available to generate intervals from.",
                message_type="warning",
            )
            return
        
        # Check unique value count
        unique_values = len(set(self.data_values))
        print(f"DEBUG: [ColorMapEditorDialog] Data has {unique_values} unique values")

        # HARD LIMIT: Refuse to create discrete maps with > 100 categories for categorical data
        MAX_DISCRETE_CATEGORIES = 100

        if self.is_categorical and unique_values > MAX_DISCRETE_CATEGORIES:
            # For categorical data with too many values, refuse with a clear message
            DialogHelper.show_message(
                self.dialog,
                "Too Many Categories",
                f"This categorical data has {unique_values} unique values, which exceeds "
                f"the maximum of {MAX_DISCRETE_CATEGORIES} for discrete color mapping.\n\n"
                f"Discrete mode is not suitable for this data. Please consider:\n"
                f"• Using a different data column with fewer categories\n"
                f"• Grouping or consolidating your category values",
                message_type="warning",
            )
            print(f"DEBUG: [ColorMapEditorDialog] Refused generation: categorical with {unique_values} > {MAX_DISCRETE_CATEGORIES}")
            return

        if not self.is_categorical and unique_values > MAX_DISCRETE_CATEGORIES:
            # For numeric data with many unique values, offer to use ranges instead
            result = DialogHelper.confirm_dialog(
                self.dialog,
                "Many Unique Values",
                f"This numeric data has {unique_values} unique values.\n\n"
                f"Discrete mode will generate quantile-based RANGES (not one row per value).\n"
                f"For smoother visualization, consider using Continuous mode instead.\n\n"
                f"Generate quantile ranges?",
                yes_text="Generate Ranges",
                no_text="Cancel"
            )

            if not result:
                print(f"DEBUG: [ColorMapEditorDialog] User cancelled generation")
                return

        # Handle categorical vs numeric data differently
        if self.is_categorical:
            # For categorical data, create one interval per category
            print(f"DEBUG: [ColorMapEditorDialog] Generating categorical intervals")

            # Sort categories alphabetically and enforce limit
            sorted_categories = sorted(self.data_values)
            if len(sorted_categories) > MAX_DISCRETE_CATEGORIES:
                # Safety limit - should not reach here due to earlier check
                sorted_categories = sorted_categories[:MAX_DISCRETE_CATEGORIES]
                print(f"DEBUG: [ColorMapEditorDialog] Truncated to {MAX_DISCRETE_CATEGORIES} categories")
            num_intervals = len(sorted_categories)
            
            print(f"DEBUG: [ColorMapEditorDialog] Generating {num_intervals} categorical intervals")
            
            # Clear existing rows
            self.discrete_rows.clear()
            
            # Generate distinct colors using HSV
            import colorsys
            for i, category in enumerate(sorted_categories):
                # Use HSV for better color distribution
                hue = i / num_intervals
                saturation = 0.7
                value = 0.9
                r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
                
                row_data = {
                    "enabled": tk.BooleanVar(value=True),
                    "min": tk.DoubleVar(value=float(i)),
                    "max": tk.DoubleVar(value=float(i + 1)),
                    "color_bgr": (int(b * 255), int(g * 255), int(r * 255)),
                    "category": category,  # Store the actual category text
                }
                
                self.discrete_rows.append(row_data)
                print(f"DEBUG: [ColorMapEditorDialog]   Category {i}: {category}")
            
            self._rebuild_discrete_rows()
            self._refresh_preview()
            print(f"DEBUG: [ColorMapEditorDialog] Generated {num_intervals} categorical intervals")
            return
            
        # For numeric data:
        data_min = min(self.data_values)
        data_max = max(self.data_values)
        
        # Decide number of intervals based on unique values
        if unique_values <= 10:
            num_intervals = min(unique_values, 10)
        elif unique_values <= 50:
            num_intervals = 10
        else:
            num_intervals = 20
        
        print(f"DEBUG: [ColorMapEditorDialog] Generating {num_intervals} numeric intervals from {data_min:.3f} to {data_max:.3f}")
        
        # Clear existing rows
        self.discrete_rows.clear()
        
        # Generate quantile-based intervals for better distribution
        quantiles = np.linspace(0, 100, num_intervals + 1)
        edges = np.percentile(self.data_values, quantiles)
        
        # Use a simple color gradient (blue to red)
        for i in range(num_intervals):
            min_val = float(edges[i])
            max_val = float(edges[i + 1])
            
            # Interpolate color
            t = i / (num_intervals - 1) if num_intervals > 1 else 0
            r = int(255 * t)
            b = int(255 * (1 - t))
            g = 0
            
            row_data = {
                "enabled": tk.BooleanVar(value=True),
                "min": tk.DoubleVar(value=min_val),
                "max": tk.DoubleVar(value=max_val),
                "color_bgr": (b, g, r),
            }
            
            self.discrete_rows.append(row_data)
            
            # Calculate percentage
            pct = self._calculate_data_percentage(min_val, max_val)
            print(f"DEBUG: [ColorMapEditorDialog]   Interval {i}: {min_val:.3f}-{max_val:.3f} = BGR({b},{g},{r}) [{pct:.1f}%]")
        
        self._rebuild_discrete_rows()
        self._refresh_preview()
        
        print(f"DEBUG: [ColorMapEditorDialog] Generated {num_intervals} intervals")
        
    # ========================================================================
    # Continuous Mode Methods
    # ========================================================================
    
    def _on_gradient_changed(self):
        """Handle gradient preset or invert change."""
        print(f"DEBUG: [ColorMapEditorDialog] _on_gradient_changed")
        # TODO: Implement gradient preset loading
        self._refresh_preview()
        
    def _on_dynamic_toggle(self):
        """Handle dynamic limits toggle."""
        is_dynamic = self.dynamic_var.get()
        print(f"DEBUG: [ColorMapEditorDialog] _on_dynamic_toggle: dynamic={is_dynamic}")
        
        # Enable/disable manual entry
        state = "disabled" if is_dynamic else "normal"
        self.min_entry.config(state=state)
        self.max_entry.config(state=state)
        
        if is_dynamic and self.data_values:
            self.min_var.set(min(self.data_values))
            self.max_var.set(max(self.data_values))
            print(f"DEBUG: [ColorMapEditorDialog] Set dynamic limits: {self.min_var.get():.3f} to {self.max_var.get():.3f}")
        
        self._refresh_preview()
        
    def _set_limits_from_data(self):
        """Set gradient limits from data range."""
        print(f"DEBUG: [ColorMapEditorDialog] _set_limits_from_data called")
        
        if not self.data_values:
            DialogHelper.show_message(
                self.dialog,
                "No Data",
                "No data available to set limits from.",
                message_type="warning",
            )
            return
        
        data_min = min(self.data_values)
        data_max = max(self.data_values)
        
        self.min_var.set(data_min)
        self.max_var.set(data_max)
        
        print(f"DEBUG: [ColorMapEditorDialog] Set limits from data: {data_min:.3f} to {data_max:.3f}")
        
        self._refresh_preview()
        
    def _add_gradient_stop(self):
        """Add a gradient stop point."""
        print(f"DEBUG: [ColorMapEditorDialog] _add_gradient_stop called")
        
        # Pick a color
        color_result = colorchooser.askcolor(parent=self.dialog)
        
        if color_result[1]:
            hex_color = color_result[1]
            
            # Place at midpoint of current range
            if self.data_values:
                value = np.median(self.data_values)
            else:
                value = (self.min_var.get() + self.max_var.get()) / 2
            
            self.gradient_stops.append((value, hex_color))
            self.gradient_stops.sort(key=lambda x: x[0])
            
            print(f"DEBUG: [ColorMapEditorDialog] Added gradient stop: value={value:.3f}, color={hex_color}")
            
            self._update_curve_list_label()
            self._refresh_preview()
        
    def _generate_curve(self):
        """Auto-generate gradient curve points."""
        print(f"DEBUG: [ColorMapEditorDialog] _generate_curve called")
        
        # Create 2 stops (min and max) with blue to red gradient
        if self.data_values:
            data_min = min(self.data_values)
            data_max = max(self.data_values)
        else:
            data_min = self.min_var.get()
            data_max = self.max_var.get()
        
        self.gradient_stops = [
            (data_min, "#0000FF"),  # Blue at min
            (data_max, "#FF0000"),  # Red at max
        ]
        
        print(f"DEBUG: [ColorMapEditorDialog] Generated gradient curve: {data_min:.3f} (blue) to {data_max:.3f} (red)")
        
        self._update_curve_list_label()
        self._refresh_preview()
        
    def _update_curve_list_label(self):
        """Update the curve points list label."""
        if not self.gradient_stops:
            text = "No curve points defined"
        else:
            text = f"{len(self.gradient_stops)} curve point(s) defined"
        
        self.curve_list_label.config(text=text)
        print(f"DEBUG: [ColorMapEditorDialog] Curve list label updated: {text}")
        
    # ========================================================================
    # Preview Management
    # ========================================================================
    
    def _refresh_preview(self):
        """Refresh the histogram preview."""
        print(f"DEBUG: [ColorMapEditorDialog] _refresh_preview called, mode={self.mode_var.get()}")
        
        if not self.preview_widget:
            print(f"DEBUG: [ColorMapEditorDialog] No preview widget, skipping")
            return
        
        if self.mode_var.get() == "discrete":
            self._refresh_discrete_preview()
        else:
            self._refresh_continuous_preview()
            
    def _on_gradient_dragged(self):
        """
        Handle gradient stops being changed by dragging in preview widget.
        Sync the stops back to our list and update curve display.
        """
        if not self.preview_widget:
            return

        # Get updated stops from preview widget
        self.gradient_stops = list(self.preview_widget.gradient_stops)

        # Update the curve point list label
        count = len(self.gradient_stops)
        if hasattr(self, 'curve_list_label'):
            self.curve_list_label.config(text=f"{count} curve point(s) defined")

    def _refresh_discrete_preview(self):
        """Refresh preview in discrete mode."""
        print(f"DEBUG: [ColorMapEditorDialog] _refresh_discrete_preview")
        
        # Build list of ranges
        ranges = []
        for idx, row_data in enumerate(self.discrete_rows):
            if row_data["enabled"].get():
                try:
                    # For categorical data, we still need numeric ranges for the preview
                    # but we're mapping categories to sequential indices
                    min_val = row_data["min"].get()
                    max_val = row_data["max"].get()
                    
                    # Validate values are not NaN
                    if np.isnan(min_val) or np.isnan(max_val):
                        print(f"DEBUG: [ColorMapEditorDialog] Skipping row with NaN values: min={min_val}, max={max_val}")
                        continue
                        
                    hex_color = bgr_to_hex(row_data["color_bgr"])
                    
                    # If this is categorical, log the category name too
                    if "category" in row_data and row_data["category"] is not None:
                        print(f"DEBUG: [ColorMapEditorDialog]   Category '{row_data['category']}' -> range {min_val:.1f}-{max_val:.1f} = {hex_color}")
                    
                    ranges.append((min_val, max_val, hex_color))
                except (tk.TclError, ValueError) as e:
                    print(f"DEBUG: [ColorMapEditorDialog] Error getting row values: {e}")
                    continue
        
        print(f"DEBUG: [ColorMapEditorDialog] Sending {len(ranges)} valid ranges to preview")
        
        self.preview_widget.set_discrete_preview(ranges)
        
    def _refresh_continuous_preview(self):
        """Refresh preview in continuous mode."""
        print(f"DEBUG: [ColorMapEditorDialog] _refresh_continuous_preview")
        
        gradient_min = self.min_var.get()
        gradient_max = self.max_var.get()
        
        print(f"DEBUG: [ColorMapEditorDialog] Gradient range: {gradient_min:.3f} to {gradient_max:.3f}")
        print(f"DEBUG: [ColorMapEditorDialog] {len(self.gradient_stops)} gradient stops")
        
        self.preview_widget.set_continuous_preview(
            self.gradient_stops,
            gradient_min,
            gradient_max,
        )
        
    # ========================================================================
    # Data Loading/Saving
    # ========================================================================
    
    def _load_initial_color_map(self):
        """Load initial color map if provided."""
        print(f"DEBUG: [ColorMapEditorDialog] _load_initial_color_map")
        
        if not self.initial_color_map:
            print(f"DEBUG: [ColorMapEditorDialog] No initial color map, using defaults")
            # Auto-detect appropriate mode based on data
            if self.data_values:
                unique_count = len(set(self.data_values))
                print(f"DEBUG: [ColorMapEditorDialog] Data has {unique_count} unique values from {len(self.data_values)} total")
                
                # Heuristic: If unique values > 20 and continuous numeric, use continuous mode
                if unique_count > 20:
                    # Check if data is numeric and continuous
                    try:
                        # Check if values are numeric
                        numeric_values = [float(v) for v in self.data_values]
                        # Check if data looks continuous (has decimal values or large range)
                        value_range = max(numeric_values) - min(numeric_values)
                        has_decimals = any(v != int(v) for v in numeric_values[:min(100, len(numeric_values))])
                        
                        # Use continuous mode if:
                        # - Range > 10 (e.g., 0-100), OR
                        # - Has decimal values (e.g., 0.5, 1.3, etc.), OR
                        # - Many unique values relative to range (dense data)
                        if value_range > 10 or has_decimals or (value_range > 0 and unique_count > value_range * 2):
                            # Looks like continuous numeric data
                            print(f"DEBUG: [ColorMapEditorDialog] Auto-detected continuous numeric data")
                            print(f"DEBUG: [ColorMapEditorDialog]   Range: {value_range:.3f}, Has decimals: {has_decimals}, Unique: {unique_count}")
                            self.mode_var.set("continuous")
                            self._show_mode_panel()
                            self._set_limits_from_data()
                            return
                    except (ValueError, TypeError):
                        pass  # Not numeric, use discrete
                
                # Default to discrete for categorical or small datasets
                print(f"DEBUG: [ColorMapEditorDialog] Using discrete mode")
                self.mode_var.set("discrete")
                self._show_mode_panel()
                self._generate_intervals()
            return
        
        # Load from ColorMap object
        print(f"DEBUG: [ColorMapEditorDialog] Loading color map: {self.initial_color_map.name}")
        print(f"DEBUG: [ColorMapEditorDialog] Type: {self.initial_color_map.type}")
        
        # Set name and description
        self.name_var.set(self.initial_color_map.name)
        self.desc_var.set(self.initial_color_map.description)
        
        # Import ColorMapType to check type
        from processing.LoggingReviewStep.color_map_manager import ColorMapType
        
        # Set mode based on type
        if self.initial_color_map.type == ColorMapType.CATEGORICAL:
            self.mode_var.set("discrete")
            print(f"DEBUG: [ColorMapEditorDialog] Loading categorical color map")
            
            # Load categories as discrete ranges
            self.discrete_rows.clear()
            
            for idx, (category, color_bgr) in enumerate(sorted(self.initial_color_map.categories.items())):
                row_data = {
                    "enabled": tk.BooleanVar(value=True),
                    "min": tk.DoubleVar(value=float(idx)),
                    "max": tk.DoubleVar(value=float(idx + 1)),
                    "color_bgr": color_bgr,
                    "category": category,  # Store the category name
                }
                self.discrete_rows.append(row_data)
                print(f"DEBUG: [ColorMapEditorDialog] Loaded category: {category} -> {color_bgr}")
            
            self._rebuild_discrete_rows()
            
        elif self.initial_color_map.type in (ColorMapType.NUMERIC, ColorMapType.GRADIENT):
            self.mode_var.set("discrete")  # Load numeric as discrete ranges
            print(f"DEBUG: [ColorMapEditorDialog] Loading numeric color map")
            
            # Load ranges
            self.discrete_rows.clear()
            
            for range_obj in self.initial_color_map.ranges:
                row_data = {
                    "enabled": tk.BooleanVar(value=True),
                    "min": tk.DoubleVar(value=range_obj.min),
                    "max": tk.DoubleVar(value=range_obj.max),
                    "color_bgr": range_obj.color,
                }
                self.discrete_rows.append(row_data)
                print(f"DEBUG: [ColorMapEditorDialog] Loaded range: {range_obj.min}-{range_obj.max} -> {range_obj.color}")
            
            self._rebuild_discrete_rows()
        
        # Trigger mode display
        self._show_mode_panel()
        self._refresh_preview()
        
        print(f"DEBUG: [ColorMapEditorDialog] Initial color map loaded successfully")
        
    # ========================================================================
    # Dialog Actions
    # ========================================================================
    
    def _on_apply(self):
        """Handle Apply button click."""
        print(f"DEBUG: [ColorMapEditorDialog] _on_apply clicked")
        
        # Import ColorMap classes
        from processing.LoggingReviewStep.color_map_manager import ColorMap, ColorRange, ColorMapType
        
        # Get name and description
        name = self.name_var.get().strip()
        description = self.desc_var.get().strip()
        
        if not name:
            DialogHelper.show_message(
                self.dialog,
                "Invalid Name",
                "Please provide a name for the color map.",
                message_type="error"
            )
            return
        
        # Determine type based on mode and whether this is categorical data
        mode = self.mode_var.get()
        
        if mode == "discrete":
            # Check if this is categorical data (has category names)
            has_categories = any("category" in row_data and row_data["category"] is not None 
                                for row_data in self.discrete_rows)
            
            if has_categories or self.is_categorical:
                # Build CATEGORICAL color map
                color_map = ColorMap(name, ColorMapType.CATEGORICAL, description)
                
                print(f"DEBUG: [ColorMapEditorDialog] Creating CATEGORICAL color map")
                
                # Add categories from discrete rows
                for row_data in self.discrete_rows:
                    if row_data["enabled"].get():
                        category = row_data.get("category")
                        if category:
                            color_bgr = row_data["color_bgr"]
                            color_map.add_category(category, color_bgr)
                            print(f"{category} = {color_bgr}")
                
                if not color_map.categories:
                    DialogHelper.show_message(
                        self.dialog,
                        "No Categories",
                        "Please add at least one enabled category.",
                        message_type="error"
                    )
                    return
            else:
                # Build NUMERIC color map with ranges
                color_map = ColorMap(name, ColorMapType.NUMERIC, description)
                
                print(f"DEBUG: [ColorMapEditorDialog] Creating NUMERIC color map")
                
                # Add ranges from discrete rows
                for row_data in self.discrete_rows:
                    if row_data["enabled"].get():
                        try:
                            min_val = row_data["min"].get()
                            max_val = row_data["max"].get()
                            
                            # Validate values are not NaN
                            if np.isnan(min_val) or np.isnan(max_val):
                                print(f"DEBUG: [ColorMapEditorDialog] Skipping row with NaN values during save")
                                continue
                                
                            color_bgr = row_data["color_bgr"]
                            
                            # Create label
                            label = f"{min_val:.2f}-{max_val:.2f}"
                            
                            color_range = ColorRange(min_val, max_val, color_bgr, label)
                            color_map.add_range(color_range)
                            
                            print(f"DEBUG: [ColorMapEditorDialog] Added range: {min_val:.2f}-{max_val:.2f}")
                        except (tk.TclError, ValueError) as e:
                            print(f"DEBUG: [ColorMapEditorDialog] Error reading row during save: {e}")
                            continue
                
                if not color_map.ranges:
                    DialogHelper.show_message(
                        self.dialog,
                        "No Ranges",
                        "Please add at least one enabled range.",
                        message_type="error"
                    )
                    return
                
        else:  # continuous mode
            # Build gradient color map (stored as NUMERIC with ranges)
            color_map = ColorMap(name, ColorMapType.GRADIENT, description)
            
            # For continuous/gradient mode, we could store gradient stops as metadata
            # For now, we'll create a discrete approximation with many small ranges
            
            if not self.gradient_stops or len(self.gradient_stops) < 2:
                DialogHelper.show_message(
                    self.dialog,
                    "Invalid Gradient",
                    "Please add at least 2 gradient stops.",
                    message_type="error"
                )
                return
            
            gradient_min = self.min_var.get()
            gradient_max = self.max_var.get()
            
            # Create 50 discrete ranges to approximate the gradient
            num_steps = 50
            step_size = (gradient_max - gradient_min) / num_steps
            
            for i in range(num_steps):
                range_min = gradient_min + (i * step_size)
                range_max = gradient_min + ((i + 1) * step_size)
                
                # Interpolate color for this range midpoint
                midpoint = (range_min + range_max) / 2
                hex_color = self._interpolate_gradient_color(midpoint)
                color_bgr = hex_to_bgr(hex_color)
                
                label = f"{range_min:.2f}-{range_max:.2f}"
                color_range = ColorRange(range_min, range_max, color_bgr, label)
                color_map.add_range(color_range)
            
            print(f"DEBUG: [ColorMapEditorDialog] Generated {num_steps} gradient ranges")
        
        # Save via ColorMapManager (in-memory)
        if self.color_map_manager:
            self.color_map_manager.save_preset(name, color_map)
            print(f"DEBUG: [ColorMapEditorDialog] Saved color map to manager: {name}")
        
        # Also save to JSON file in AppData/GeoVue/color_maps/
        json_path = self._save_color_map_to_json(name, color_map)
        
        if json_path:
            DialogHelper.show_message(
                self.dialog,
                "Success",
                f"Color map '{name}' saved successfully!\n\nFile: {json_path}",
                message_type="info"
            )
        else:
            DialogHelper.show_message(
                self.dialog,
                "Partial Success",
                f"Color map '{name}' saved to session, but could not save to file.",
                message_type="warning"
            )
        
        # Set result
        self.result = color_map
        
        # Call callback if provided
        if self.on_save_callback:
            print(f"DEBUG: [ColorMapEditorDialog] Calling save callback")
            self.on_save_callback()
        
        self._on_close()
    
    def _interpolate_gradient_color(self, value: float) -> str:
        """
        Interpolate color from gradient stops for a given value.
        
        Args:
            value: The value to get color for
            
        Returns:
            Hex color string
        """
        # Sort stops by value
        sorted_stops = sorted(self.gradient_stops, key=lambda x: x[0])
        
        # Find the two stops to interpolate between
        if value <= sorted_stops[0][0]:
            return sorted_stops[0][1]
        if value >= sorted_stops[-1][0]:
            return sorted_stops[-1][1]
        
        # Find bracketing stops
        for i in range(len(sorted_stops) - 1):
            stop1_val, stop1_color = sorted_stops[i]
            stop2_val, stop2_color = sorted_stops[i + 1]
            
            if stop1_val <= value <= stop2_val:
                # Linear interpolation
                t = (value - stop1_val) / (stop2_val - stop1_val)
                
                # Convert hex to RGB
                rgb1 = tuple(int(stop1_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                rgb2 = tuple(int(stop2_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                
                # Interpolate
                r = int(rgb1[0] + t * (rgb2[0] - rgb1[0]))
                g = int(rgb1[1] + t * (rgb2[1] - rgb1[1]))
                b = int(rgb1[2] + t * (rgb2[2] - rgb1[2]))
                
                return f"#{r:02x}{g:02x}{b:02x}"
        
        # Fallback
        return sorted_stops[0][1]

    def _save_color_map_to_json(self, name: str, color_map) -> Optional[str]:
        """
        Save color map to a JSON file in AppData/GeoVue/color_maps/.
        
        Args:
            name: Color map name (used for filename)
            color_map: ColorMap object to save
            
        Returns:
            Path to saved file, or None if failed
        """
        try:
            # Get AppData path
            appdata = os.getenv("APPDATA")
            if not appdata:
                appdata = os.path.expanduser("~/.config")
            
            # Create color_maps directory
            color_maps_dir = Path(appdata) / "GeoVue" / "color_maps"
            color_maps_dir.mkdir(parents=True, exist_ok=True)
            
            # Create filename from name
            import re
            safe_name = re.sub(r'[^\w\-]', '_', name.lower())
            json_path = color_maps_dir / f"{safe_name}.json"
            
            # Build JSON structure
            from processing.LoggingReviewStep.color_map_manager import ColorMapType
            
            json_data = {
                "name": color_map.name,
                "description": color_map.description,
                "type": color_map.type.value if hasattr(color_map.type, 'value') else str(color_map.type),
            }
            
            if color_map.type == ColorMapType.CATEGORICAL:
                # Save categories
                json_data["categories"] = {}
                for category, bgr in color_map.categories.items():
                    b, g, r = bgr
                    json_data["categories"][category] = {
                        "color": f"#{r:02x}{g:02x}{b:02x}",
                        "bgr": list(bgr),
                    }
            else:
                # Save ranges
                json_data["ranges"] = []
                for range_obj in color_map.ranges:
                    b, g, r = range_obj.color
                    json_data["ranges"].append({
                        "min": range_obj.min,
                        "max": range_obj.max,
                        "label": range_obj.label,
                        "color": f"#{r:02x}{g:02x}{b:02x}",
                        "bgr": list(range_obj.color),
                    })
            
            # Write file
            import json
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)
            
            print(f"DEBUG: [ColorMapEditorDialog] Saved color map to: {json_path}")
            return str(json_path)
            
        except Exception as e:
            print(f"DEBUG: [ColorMapEditorDialog] Error saving color map to JSON: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _on_close(self):
        """Handle Close button click."""
        print(f"DEBUG: [ColorMapEditorDialog] _on_close clicked")
        self.dialog.destroy()
        
    def show(self):
        """Show the dialog and wait for it to close."""
        print(f"DEBUG: [ColorMapEditorDialog] show() called, entering wait_window")
        self.dialog.wait_window()
        return self.result


# ============================================================================
# Test Function
# ============================================================================

def test_color_map_editor():
    """Test function to run the dialog standalone."""
    print("Starting ColorMapEditorDialog test...")
    
    # Create dummy root
    root = tk.Tk()
    root.withdraw()
    
    # Create dummy gui_manager
    class DummyGUIManager:
        theme_colors = {
            "background": "#2b2b2b",
            "secondary_bg": "#3c3c3c",
            "field_bg": "#4a4a4a",
            "field_border": "#5a5a5a",
            "text": "#ffffff",
            "border": "#5a5a5a",
            "accent_blue": "#4a6fae",
            "accent_green": "#5a9f5a",
            "accent_red": "#c74440",
            "checkbox_bg": "#4a4a4a",
        }
        
        def style_dropdown(self, dropdown, width=None):
            pass
    
    gui_manager = DummyGUIManager()
    
    # Generate test data
    np.random.seed(42)
    test_data = np.random.normal(50, 10, 1000).tolist()
    
    # Create dialog
    dialog = ColorMapEditorDialog(
        parent=root,
        gui_manager=gui_manager,
        color_map_manager=None,  # Dummy
        data_column="Fe_pct",
        data_values=test_data,
        initial_color_map=None,
    )
    
    dialog.show()
    
    print("Test complete")


if __name__ == "__main__":
    test_color_map_editor()
