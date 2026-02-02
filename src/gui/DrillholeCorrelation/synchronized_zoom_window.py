"""
SynchronizedZoomWindow - Floating detail view showing enlarged images and data
for all active drillholes at the current cursor position.
"""

import tkinter as tk
from tkinter import ttk
import logging
from typing import Dict, List, Optional, Any, Tuple
import os
import cv2
import numpy as np
from PIL import Image, ImageTk

logger = logging.getLogger(__name__)


class SynchronizedZoomWindow(tk.Toplevel):
    """
    Floating window showing enlarged view of multiple drillholes at cursor position.
    
    Features:
    - Shows N intervals above/below cursor for each hole
    - Displays large images (300-400px) + data viz columns
    - Stratigraphically aligned across all holes
    - Can be pinned open or follow cursor
    - Resizable and draggable
    """
    
    def __init__(
        self,
        parent: tk.Widget,
        gui_manager: Any,
        config_manager: Any,
        color_map_manager: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize SynchronizedZoomWindow.
        
        Args:
            parent: Parent widget
            gui_manager: GUIManager for theming
            config_manager: ConfigManager for settings
            color_map_manager: Optional ColorMapManager for data colors
        """
        super().__init__(parent, **kwargs)
        
        self.gui_manager = gui_manager
        self.config_manager = config_manager
        self.color_map_manager = color_map_manager
        self.theme_colors = gui_manager.theme_colors
        self.fonts = gui_manager.fonts
        
        # Window settings
        self.title("Synchronized Zoom View")
        self.configure(bg=self.theme_colors["background"])
        
        # Configuration
        self.image_width = config_manager.get("zoom_window_image_width", 200)  # Narrower cells
        self.intervals_before = config_manager.get("zoom_window_intervals_before", 3)
        self.intervals_after = config_manager.get("zoom_window_intervals_after", 3)
        self.cell_height = config_manager.get("correlation_cell_height", 150)  # Taller cells for larger images
        self.data_column_width = config_manager.get("correlation_data_column_width", 50)
        
        # Active drillholes
        self.active_holes: Dict[str, Any] = {}  # {hole_id: column_widget}
        self.hole_canvases: Dict[str, tk.Canvas] = {}  # {hole_id: canvas}
        self.image_refs: Dict[str, Dict[int, ImageTk.PhotoImage]] = {}  # {hole_id: {idx: photo}}
        
        # Current position
        self.current_depth_by_hole: Dict[str, float] = {}  # {hole_id: depth}
        
        # Pinned state
        self.is_pinned = False
        
        # Moisture preference for image display (Wet/Dry)
        self.moisture_preference = "Dry"  # Default to Dry images
        
        # Build UI
        self._create_ui()
        
        # Start hidden
        self.withdraw()
        
        logger.info("SynchronizedZoomWindow initialized")
    
    def _create_ui(self) -> None:
        """Create the window UI."""
        # Header with controls
        header = tk.Frame(self, bg=self.theme_colors["secondary_bg"], height=40)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)
        
        # Title
        title_label = tk.Label(
            header,
            text="🔍 Zoom View",
            font=self.fonts["heading"],
            bg=self.theme_colors["secondary_bg"],
            fg=self.theme_colors["text"]
        )
        title_label.pack(side="left", padx=10)
        
        # Pin button
        self.pin_btn = tk.Button(
            header,
            text="📌 Pin",
            command=self._toggle_pin,
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["text"],
            relief="flat",
            padx=10
        )
        self.pin_btn.pack(side="right", padx=5)
        
        # Moisture indicator label (shows current state, synced from parent)
        self.moisture_label = tk.Label(
            header,
            text="🔆 Dry",
            font=self.fonts["small"],
            bg=self.theme_colors["secondary_bg"],
            fg=self.theme_colors["accent_blue"]
        )
        self.moisture_label.pack(side="right", padx=10)
        
        # Main content area with horizontal scrollbar
        content_frame = tk.Frame(self, bg=self.theme_colors["background"])
        content_frame.pack(fill="both", expand=True)
        
        # Horizontal scrollable canvas
        self.h_scrollbar = ttk.Scrollbar(content_frame, orient="horizontal")
        self.h_scrollbar.pack(side="bottom", fill="x")
        
        self.main_canvas = tk.Canvas(
            content_frame,
            bg=self.theme_colors["field_bg"],
            highlightthickness=0,
            xscrollcommand=self.h_scrollbar.set
        )
        self.main_canvas.pack(side="top", fill="both", expand=True)
        
        self.h_scrollbar.config(command=self.main_canvas.xview)
        
        # Container for drillhole columns
        self.columns_container = tk.Frame(
            self.main_canvas,
            bg=self.theme_colors["field_bg"]
        )
        self.columns_window = self.main_canvas.create_window(
            (0, 0),
            window=self.columns_container,
            anchor="nw"
        )
    
    def add_drillhole(self, hole_id: str, column_widget: Any) -> None:
        """Add a drillhole to the zoom window."""
        if hole_id in self.active_holes:
            return
        
        logger.info(f"Adding {hole_id} to zoom window")
        
        self.active_holes[hole_id] = column_widget
        self.image_refs[hole_id] = {}
        
        # Create column frame
        column_frame = tk.Frame(
            self.columns_container,
            bg=self.theme_colors["secondary_bg"],
            highlightbackground=self.theme_colors["border"],
            highlightthickness=1
        )
        column_frame.pack(side="left", padx=5, pady=5, fill="both")
        
        # Header with hole ID
        header = tk.Label(
            column_frame,
            text=hole_id,
            font=self.fonts["heading"],
            bg=self.theme_colors["secondary_bg"],
            fg=self.theme_colors["accent_blue"],
            height=2
        )
        header.pack(fill="x")
        
        # Canvas for images and data
        total_height = (self.intervals_before + self.intervals_after + 1) * self.cell_height
        canvas_width = self.image_width + (len(column_widget.viz_columns) * self.data_column_width if column_widget.viz_columns else 0)
        
        canvas = tk.Canvas(
            column_frame,
            bg=self.theme_colors["field_bg"],
            width=canvas_width,
            height=total_height,
            highlightthickness=0
        )
        canvas.pack()
        
        self.hole_canvases[hole_id] = canvas
        
        # Update scroll region
        self.main_canvas.update_idletasks()
        self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
    
    def remove_drillhole(self, hole_id: str) -> None:
        """Remove a drillhole from zoom window."""
        if hole_id not in self.active_holes:
            return
        
        logger.info(f"Removing {hole_id} from zoom window")
        
        # Clean up
        if hole_id in self.image_refs:
            del self.image_refs[hole_id]
        if hole_id in self.hole_canvases:
            del self.hole_canvases[hole_id]
        if hole_id in self.active_holes:
            del self.active_holes[hole_id]
        
        # Rebuild UI
        for widget in self.columns_container.winfo_children():
            widget.destroy()
        
        for hid, col_widget in self.active_holes.items():
            self.add_drillhole(hid, col_widget)
    
    def update_position(self, depths_by_hole: Dict[str, float]) -> None:
        """
        Update zoom window to show intervals around specified depths.
        
        Args:
            depths_by_hole: Dict of {hole_id: depth_at_cursor}
        """
        self.current_depth_by_hole = depths_by_hole
        
        logger.debug(f"Updating zoom window: {depths_by_hole}")
        
        for hole_id, depth in depths_by_hole.items():
            if hole_id not in self.active_holes:
                continue
            
            self._update_hole_view(hole_id, depth)
        
        # Show window if hidden and not pinned
        if not self.winfo_viewable() and not self.is_pinned:
            self.deiconify()
    
    def _update_hole_view(self, hole_id: str, center_depth: float) -> None:
        """Update view for a specific drillhole."""
        if hole_id not in self.active_holes or hole_id not in self.hole_canvases:
            return
        
        column_widget = self.active_holes[hole_id]
        canvas = self.hole_canvases[hole_id]
        
        # Check if canvas still exists
        try:
            if not canvas.winfo_exists():
                return
        except:
            return
        
        # Find center interval
        center_idx = None
        for i, interval in enumerate(column_widget.intervals):
            if interval.depth_from <= center_depth <= interval.depth_to:
                center_idx = i
                break
        
        if center_idx is None:
            return
        
        # Clear canvas
        canvas.delete("all")
        
        # Calculate range
        start_idx = max(0, center_idx - self.intervals_before)
        end_idx = min(len(column_widget.intervals), center_idx + self.intervals_after + 1)
        
        # Draw intervals
        for i, idx in enumerate(range(start_idx, end_idx)):
            interval = column_widget.intervals[idx]
            y_pos = i * self.cell_height
            
            # Draw image
            self._draw_enlarged_image(canvas, hole_id, interval, 0, y_pos, idx == center_idx)
            
            # Draw data viz if available
            if column_widget.viz_columns:
                self._draw_data_viz(canvas, hole_id, interval, self.image_width, y_pos, column_widget.viz_columns)
    
    def _draw_enlarged_image(self, canvas: tk.Canvas, hole_id: str, interval: Any, x: int, y: int, is_center: bool) -> None:
        """Draw enlarged image for interval."""
        image_path = self._get_preferred_image_path(interval)
        
        if not image_path or not os.path.exists(image_path):
            # Placeholder - narrower, centered
            placeholder_width = 100
            placeholder_x = x + (self.image_width - placeholder_width) // 2
            
            canvas.create_rectangle(
                placeholder_x, y + 10, 
                placeholder_x + placeholder_width, y + self.cell_height - 10,
                fill="#2a2a2a",
                outline=self.theme_colors["accent_yellow"] if is_center else self.theme_colors["border"],
                width=3 if is_center else 1
            )
            canvas.create_text(
                x + self.image_width // 2, y + self.cell_height // 2,
                text=f"{interval.depth_from:.0f}-\n{interval.depth_to:.0f}m\nNo Image",
                fill="#666666",
                font=self.fonts["small"]
            )
            return
        
        try:
            # Load and rotate image
            img = cv2.imread(image_path)
            if img is None:
                return
            
            # Rotate 90 degrees to landscape
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            h, w = img.shape[:2]
            
            # Scale to fill cell height (make images as large as possible)
            scale = self.cell_height / h
            new_h = self.cell_height
            new_w = int(w * scale)
            
            # Constrain width if too wide
            if new_w > self.image_width:
                new_w = self.image_width
                new_h = int(h * (self.image_width / w))
            
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Convert to PhotoImage
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(pil_img)
            
            # Store reference
            self.image_refs[hole_id][interval.interval_id] = photo
            
            # Draw image
            img_x = x + (self.image_width - new_w) // 2
            img_y = y + (self.cell_height - new_h) // 2
            
            canvas.create_image(
                img_x, img_y,
                image=photo,
                anchor="nw"
            )
            
            # Draw border (highlight center)
            canvas.create_rectangle(
                x, y, x + self.image_width, y + self.cell_height,
                outline=self.theme_colors["accent_yellow"] if is_center else self.theme_colors["border"],
                width=3 if is_center else 1
            )
            
        except Exception as e:
            logger.error(f"Error drawing enlarged image: {e}")
    
    def _draw_data_viz(self, canvas: tk.Canvas, hole_id: str, interval: Any, x: int, y: int, viz_columns: List[Dict]) -> None:
        """Draw data visualization columns next to image."""
        if not viz_columns or not interval.csv_data:
            return
        
        logger.debug(f"Drawing {len(viz_columns)} data viz columns for {hole_id} at {interval.depth_from}-{interval.depth_to}m")
        
        # Draw each column
        for col_idx, viz_config in enumerate(viz_columns):
            column_name = viz_config.get("column", "")
            color_map_name = viz_config.get("color_map", "")
            viz_type = viz_config.get("type", "bar")
            
            col_x = x + (col_idx * self.data_column_width)
            
            # Draw column background
            canvas.create_rectangle(
                col_x, y,
                col_x + self.data_column_width, y + self.cell_height,
                fill=self.theme_colors["secondary_bg"],
                outline=self.theme_colors["border"],
                width=1
            )
            
            # Get value from CSV data (case-insensitive)
            value = None
            actual_column = None
            
            if column_name in interval.csv_data:
                actual_column = column_name
            else:
                # Try case-insensitive match
                column_name_lower = column_name.lower()
                for col in interval.csv_data.keys():
                    if col.lower() == column_name_lower:
                        actual_column = col
                        break
            
            if actual_column is None:
                # Draw "N/A" indicator
                canvas.create_text(
                    col_x + self.data_column_width // 2,
                    y + self.cell_height // 2,
                    text="N/A",
                    fill="#666666",
                    font=self.fonts["tiny"] if hasattr(self.fonts, "tiny") else self.fonts["small"]
                )
                continue
            
            try:
                value = float(interval.csv_data[actual_column])
                if np.isnan(value):
                    raise ValueError("NaN value")
            except (ValueError, TypeError):
                canvas.create_text(
                    col_x + self.data_column_width // 2,
                    y + self.cell_height // 2,
                    text="--",
                    fill="#666666",
                    font=self.fonts["tiny"] if hasattr(self.fonts, "tiny") else self.fonts["small"]
                )
                continue
            
            # Get color from color map
            color = self._get_color_for_value(value, color_map_name, viz_config)
            
            # Draw bar
            bar_width = int((self.data_column_width - 10))
            canvas.create_rectangle(
                col_x + 5, y + 5,
                col_x + 5 + bar_width, y + self.cell_height - 5,
                fill=color,
                outline=self.theme_colors["border"],
                width=1
            )
            
            # Draw value text
            canvas.create_text(
                col_x + self.data_column_width // 2,
                y + self.cell_height // 2,
                text=f"{value:.1f}",
                fill=self.theme_colors["text"],
                font=self.fonts["tiny"] if hasattr(self.fonts, "tiny") else self.fonts["small"]
            )
    
    def _get_color_for_value(self, value: float, color_map_name: str, viz_config: Dict[str, Any]) -> str:
        """Get color for a value using color map."""
        if not self.color_map_manager or not color_map_name:
            return self.theme_colors["accent_blue"]
        
        try:
            # Get color map preset
            color_map = self.color_map_manager.get_preset(color_map_name)
            
            if not color_map:
                return self.theme_colors["accent_blue"]
            
            # Get color from map (returns BGR tuple)
            bgr_color = color_map.get_color(value)
            
            # Convert BGR to hex
            hex_color = f"#{bgr_color[2]:02x}{bgr_color[1]:02x}{bgr_color[0]:02x}"
            
            return hex_color
            
        except Exception as e:
            logger.debug(f"Error getting color for value: {e}")
            return self.theme_colors["accent_blue"]
    
    def _toggle_pin(self) -> None:
        """Toggle pinned state."""
        self.is_pinned = not self.is_pinned
        
        if self.is_pinned:
            self.pin_btn.config(text="📍 Pinned", relief="sunken")
        else:
            self.pin_btn.config(text="📌 Pin", relief="flat")
    
    def set_moisture_preference(self, preference: str) -> None:
        """
        Set the moisture preference for image display.
        
        Called by parent CorrelationDialog when user toggles wet/dry.
        
        Args:
            preference: "Wet" or "Dry"
        """
        if preference not in ("Wet", "Dry"):
            logger.warning(f"Invalid moisture preference: {preference}, using 'Dry'")
            preference = "Dry"
        
        if self.moisture_preference != preference:
            self.moisture_preference = preference
            logger.info(f"Zoom window moisture preference set to {preference}")
            
            # Update moisture label
            if hasattr(self, 'moisture_label'):
                if preference == "Dry":
                    self.moisture_label.config(text="🔆 Dry")
                else:
                    self.moisture_label.config(text="💧 Wet")
            
            # Refresh all displayed images
            self._refresh_all_views()
    
    def _refresh_all_views(self) -> None:
        """Refresh all hole views with current settings."""
        for hole_id, depth in self.current_depth_by_hole.items():
            self._update_hole_view(hole_id, depth)
    
    def _get_preferred_image_path(self, interval: Any) -> Optional[str]:
        """
        Get image path based on current moisture preference.
        
        Falls back to available image if preferred isn't available.
        
        Args:
            interval: DrillholeInterval object
            
        Returns:
            Path to image file, or None if no image available
        """
        if self.moisture_preference == "Dry":
            # Prefer dry, fall back to wet
            if hasattr(interval, 'image_path_dry') and interval.image_path_dry:
                return interval.image_path_dry
            elif hasattr(interval, 'image_path_wet') and interval.image_path_wet:
                return interval.image_path_wet
        else:
            # Prefer wet, fall back to dry
            if hasattr(interval, 'image_path_wet') and interval.image_path_wet:
                return interval.image_path_wet
            elif hasattr(interval, 'image_path_dry') and interval.image_path_dry:
                return interval.image_path_dry
        
        # Fall back to generic or best available
        if hasattr(interval, 'image_path') and interval.image_path:
            return interval.image_path
        
        # Last resort: try get_best_image_path if it exists
        if hasattr(interval, 'get_best_image_path'):
            return interval.get_best_image_path()
        
        return None
    
    def hide(self) -> None:
        """Hide zoom window."""
        if not self.is_pinned:
            self.withdraw()