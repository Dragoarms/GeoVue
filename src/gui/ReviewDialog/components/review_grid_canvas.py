# src\gui\ReviewDialog\components\review_grid_canvas.py
"""
ReviewGridCanvas - Modular UI component for image grid display

Responsible for:
- Grid layout and rendering
- Image cell display
- Selection handling (click, drag, multi-select)
- Lazy loading and viewport management
- Zoom preview integration
- Data visualization column display

Phase 2: UI Component Layer
"""

import tkinter as tk
from tkinter import ttk
from typing import List, Dict, Optional, Set, Callable, Tuple
import logging
import time
import queue
import threading
import os
from pathlib import Path
from PIL import Image as PILImage, ImageTk
import cv2


class ReviewGridCanvas:
    """
    Manages the image grid canvas display.
    Delegates data management to ReviewDataManager (Phase 1).
    """

    def __init__(
        self,
        parent: tk.Widget,
        theme_colors: Dict[str, str],
        on_selection_change: Callable[[List[int]], None],
        on_cell_click: Callable[[int], None],
        on_classification_visual_update: Optional[Callable[[int], None]] = None,
        show_peer_info: bool = False,
    ):
        """
        Initialize grid canvas.

        Args:
            parent: Parent widget
            theme_colors: Theme color dictionary
            on_selection_change: Callback when selection changes
            on_cell_click: Callback when cell is clicked
            on_classification_visual_update: Optional callback for visual updates
        """
        self.parent = parent
        self.theme_colors = theme_colors
        self.logger = logging.getLogger(__name__)

        # Callbacks
        self._on_selection_change = on_selection_change
        self._on_cell_click = on_cell_click
        self._on_classification_visual_update = on_classification_visual_update

        # Peer review display settings
        self.show_peer_info = show_peer_info
        self.peer_review_manager = None  # Will be set if needed

        # Grid settings - larger default size for better visibility
        self.base_cell_width = 300  # Increased from 160
        self.base_cell_height = 150  # Increased from 80
        self.scale_factor = 1.0
        self.cell_width = self.base_cell_width
        self.cell_height = self.base_cell_height
        self.padding = 0  # No padding between cells
        self.cols_per_row = 5
        self.rotation = 0

        # Data visualization settings
        self.show_data_visualizations = True
        self.viz_column_width = 60  # Narrower visualization columns
        self.viz_columns = [
            "Fe_pct_BEST",
            "SiO2_pct_BEST",
            "Al2O3_pct_BEST",
            "Logged_pct_CHHM",
        ]
        self.viz_column_configs = {}  # Color map configurations

        self.viz_column_labels = {
            "Fe_pct_BEST": "Fe%",
            "SiO2_pct_BEST": "SiO2%",
            "Al2O3_pct_BEST": "Al2O3%",
            "Logged_pct_CHHM": "CHHM%",
            "BIFf_2": "Type",
        }

        # Initialize color map manager
        self._init_color_maps()

        # State
        self.displayed_images = []  # Images to display
        self.cells = {}  # (row, col): cell_data
        self.cell_ids = {}  # Canvas item IDs
        self.image_refs = {}  # PhotoImage references

        # Lazy loading state
        self.visible_range = (0, 0)  # (start_idx, end_idx)
        self.loaded_cells = set()  # Set of loaded indices
        self.pending_loads = set()  # Set of pending load indices
        self.load_buffer = 100  # Rows to preload

        # Threading for lazy loading
        import threading
        from queue import Queue, Empty

        self.load_queue = Queue()
        self.loading_thread = None
        self.stop_loading = threading.Event()
        self.dialog_ref = None  # Reference to parent dialog for CSV data

        # Start loading thread
        self._start_loading_thread()

        # Selection and interaction
        self.selected_indices = set()
        self.last_selected_index = None
        self.persistent_selection = None
        self.last_selected_indices = set()

        # Drag selection
        self.drag_start = None
        self.selection_box = None
        self.dragging = False
        self.current_event = None  # Track event for modifier keys
        self._drag_last_ts = 0.0
        self._drag_pending = False

        # Zoom preview
        self.zoom_preview = None
        self.hover_cell = None

        # Drag-to-classify callback
        self._on_drag_classify_callback = None

        # UI references
        self.container = None
        self.canvas = None

        # Build UI
        self._build_ui()

        self.logger.info("ReviewGridCanvas initialized")

    def _init_color_maps(self):
        """Initialize color maps for visualization columns"""
        try:
            from processing.LoggingReviewStep.color_map_manager import ColorMapManager

            self.color_map_manager = ColorMapManager()

            # Configure default color maps for common columns
            color_map_config = {
                "Fe_pct_BEST": "fe_grade",
                "SiO2_pct_BEST": "sio2_grade",
                "Al2O3_pct_BEST": "al2o3_grade",
                "Logged_pct_CHHM": "chhm_percentage",
                "BIFf_2": "lithology",
            }

            for column, map_name in color_map_config.items():
                if self.color_map_manager.has_preset(map_name):
                    self.viz_column_configs[column] = self.color_map_manager.get_preset(
                        map_name
                    )
                    self.logger.debug(
                        f"Loaded color map '{map_name}' for column '{column}'"
                    )

        except Exception as e:
            self.logger.warning(f"Could not initialize color maps: {e}")
            self.color_map_manager = None

    def _rgb_to_hex(self, rgb):
        """Convert RGB tuple or string to hex color for Tkinter."""
        if isinstance(rgb, str):
            # If it's already a hex string, return it
            if rgb.startswith("#"):
                return rgb
            # If it's space-separated RGB like "255 0 255"
            if " " in rgb:
                try:
                    parts = rgb.split()
                    r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
                    return f"#{r:02x}{g:02x}{b:02x}"
                except (ValueError, IndexError):
                    return "#808080"  # Default gray
        elif isinstance(rgb, (list, tuple)) and len(rgb) >= 3:
            # Convert RGB tuple to hex
            r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
            return f"#{r:02x}{g:02x}{b:02x}"
        # Default fallback
        return "#808080"

    def _get_border_color(self, img) -> str:
        """
        [NEW] Get border color based on classification state.

        Returns classification-specific colors for visual feedback:
        - BIFf: Green (#4CAF50)
        - BIFhm: Blue (#21B8F3)
        - Other: Orange (#FF9800)
        - Not Confident: Red (#CE3030)
        - Unassigned: Default border color

        Handles both string and enum classification types.
        """
        # Color mapping for classifications
        mode_colors = {
            "biff": "#4CAF50",  # Green
            "bifhm": "#21B8F3",  # Blue (was #f44336 red, changed to match old dialog)
            "other": "#FF9800",  # Orange
            "not_confident": "#CE3030",  # Red
        }

        # Handle both string and enum classifications
        classification_str = str(img.classification)

        # Check for string representation of enum
        if "ClassificationCategory." in classification_str:
            classification_str = classification_str.replace(
                "ClassificationCategory.", ""
            )

        # Map to proper colors - handle multiple string variations
        if classification_str in ["BIFF", "BIFf"]:
            return mode_colors["biff"]
        elif classification_str in ["BIFHM", "BIF HM"]:
            return mode_colors["bifhm"]
        elif classification_str in ["OTHER", "Other"]:
            return mode_colors["other"]
        elif classification_str in ["NOT_CONFIDENT", "Not Confident"]:
            return mode_colors["not_confident"]
        else:
            return self.theme_colors.get("border", "#3a3a3a")

    def set_peer_review_manager(self, peer_review_manager):
        """Set the peer review manager for displaying peer info"""
        self.peer_review_manager = peer_review_manager

    def _build_ui(self):
        """Build the grid canvas UI"""
        # Create container
        self.container = ttk.Frame(self.parent)
        self.container.pack(fill=tk.BOTH, expand=True)

        # Create canvas (no horizontal scrollbar)
        self.canvas = tk.Canvas(
            self.container, bg=self.theme_colors["background"], highlightthickness=0
        )

        # Bind hover events for tooltips
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Leave>", self._on_canvas_leave)
        self.current_tooltip = None
        self.tooltip_timer = None

        v_scrollbar = ttk.Scrollbar(
            self.container, orient="vertical", command=self.canvas.yview
        )

        self.canvas.configure(yscrollcommand=v_scrollbar.set)

        # Grid layout
        self.canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        # Bind events
        self._bind_events()

    def _start_loading_thread(self):
        """Start background thread for loading images"""
        if self.loading_thread and self.loading_thread.is_alive():
            return

        self.stop_loading.clear()
        self.loading_thread = threading.Thread(
            target=self._load_images_worker, daemon=True
        )
        self.loading_thread.start()
        self.logger.debug("Image loading thread started")

    def _load_images_worker(self):
        """Worker thread to load images from queue"""
        print(
            f"DEBUG: Worker thread STARTED (thread id: {threading.current_thread().ident})"
        )

        while not self.stop_loading.is_set():
            try:
                idx = self.load_queue.get(timeout=0.1)
                # print(f"DEBUG: Worker thread got idx={idx} from queue")

                # Skip if this index is no longer pending (was cleared during force reload)
                if idx not in self.pending_loads:
                    print(
                        f"DEBUG: Worker thread skipping idx={idx} (no longer pending)"
                    )
                    continue

                if 0 <= idx < len(self.displayed_images):
                    # print(f"DEBUG: Worker thread calling _load_cell_impl({idx})")
                    self._load_cell_impl(idx)
                else:
                    print(
                        f"DEBUG: Worker thread skipping idx={idx} (out of range, len={len(self.displayed_images)})"
                    )
            except queue.Empty:
                # This is normal - just means no items in queue
                pass
            except Exception as e:
                print(f"DEBUG: Worker thread error: {e}")
                self.logger.error(f"Error in loading thread: {e}")

        print(f"DEBUG: Worker thread STOPPED")

    def set_dialog_ref(self, dialog_ref):
        """Set reference to parent dialog for CSV data access"""
        self.dialog_ref = dialog_ref

    def rotate_images(self, angle: int = 90):
        """Rotate images by specified angle (always clockwise)"""
        self.rotation = (self.rotation + angle) % 360

        # Store a COPY of the current images to reload them
        images_to_reload = list(self.displayed_images)

        # Force reload with new rotation
        self.load_images(images_to_reload, preserve_selection=True, force_reload=True)
        self.logger.debug(f"Rotated to {self.rotation}°")

    def set_scale(self, scale: float):
        """Set the scale factor and reload"""
        old_scale = self.scale_factor
        self.scale_factor = max(0.3, min(5.0, scale))  # Clamp between 0.3x and 5x

        if self.scale_factor != old_scale:
            # Store a COPY of the current images to reload them
            images_to_reload = list(self.displayed_images)

            # Force reload with new scale
            self.load_images(
                images_to_reload, preserve_selection=True, force_reload=True
            )
            self.logger.debug(f"Scaled to {self.scale_factor:.2f}x")

    def _bind_events(self):
        """Bind mouse and keyboard events"""
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<Motion>", self._on_mouse_motion)
        self.canvas.bind("<Leave>", self._on_mouse_leave)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Control-MouseWheel>", self._on_ctrl_mousewheel)

        # CRITICAL: Configure event for lazy loading when canvas is resized
        self.canvas.bind("<Configure>", self._on_canvas_configure)

    def _refresh_cell_border(self, idx: int):
        """Refresh just the border of a cell based on current classification"""
        if idx >= len(self.displayed_images):
            return

        row = idx // self.cols_per_row
        col = idx % self.cols_per_row

        if (row, col) not in self.cell_ids:
            return

        cell_ids = self.cell_ids.get((row, col), {})
        border_id = cell_ids.get("border")

        if not border_id:
            return

        img = self.displayed_images[idx]

        # Update border styling
        border_color = self._get_classification_border_color(img)
        border_width = self._get_classification_border_width(img)
        dash_pattern = self._get_classification_dash_pattern(img)

        try:
            self.canvas.itemconfig(
                border_id,
                outline=border_color,
                width=border_width,
                dash=dash_pattern if dash_pattern else (),
            )
        except Exception as e:
            self.logger.error(f"Failed to refresh border for cell {idx}: {e}")

    # Event handlers
    def _on_left_click(self, event):
        """Handle left mouse click"""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # Always set drag_start for potential drag operation
        self.drag_start = (canvas_x, canvas_y)
        self.current_event = event  # Track modifiers for Ctrl+drag
        self.dragging = False  # Will be set to True if mouse moves

        cell = self._get_cell_at(canvas_x, canvas_y)

        if cell:
            idx = self.cells[cell]["idx"]

            # Ctrl+Click for multi-select
            if event.state & 0x0004:  # Ctrl key
                if idx in self.selected_indices:
                    self.selected_indices.remove(idx)
                else:
                    self.selected_indices.add(idx)
            # Shift+Click for range select
            elif event.state & 0x0001:  # Shift key
                if self.last_selected_index is not None:
                    start = min(self.last_selected_index, idx)
                    end = max(self.last_selected_index, idx)
                    self.selected_indices = set(range(start, end + 1))
            # Normal click
            else:
                self.selected_indices = {idx}

            self.last_selected_index = idx
            self._update_selection_visual()
            self._on_selection_change(list(self.selected_indices))
            self._on_cell_click(idx)
        else:
            # Clicked on empty space - start drag select
            self.drag_start = (canvas_x, canvas_y)
            self.dragging = True
            self.current_event = event  # Track modifiers for Ctrl+drag
            # Don't clear selection yet - will be cleared in _on_drag unless Ctrl is held
            self._update_selection_visual()

    def _on_left_release(self, event):
        """Handle left mouse release - apply classification if in drag mode"""
        if self.dragging:
            self.dragging = False
            if self.selection_box:
                self.canvas.delete(self.selection_box)
                self.selection_box = None
            self.drag_start = None

            # If we have selected indices from drag and a callback is set, trigger it immediately
            if (
                self.selected_indices
                and hasattr(self, "_on_drag_classify_callback")
                and self._on_drag_classify_callback
            ):
                selected_list = list(self.selected_indices)
                self._on_drag_classify_callback(selected_list)

                # Refresh the cells visually to show updated borders
                for idx in selected_list:
                    self._refresh_cell_border(idx)
        else:
            # Normal click (not drag) - keep selection but don't classify
            # This allows clicking to select without classifying
            self.drag_start = None

    def _on_drag(self, event):
        """Handle mouse drag for selection"""
        if not self.drag_start:
            return

        # Start dragging if mouse has moved enough
        if not self.dragging:
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)
            # Check if moved at least 3 pixels to avoid accidental drags
            dx = abs(canvas_x - self.drag_start[0])
            dy = abs(canvas_y - self.drag_start[1])
            if dx > 3 or dy > 3:
                self.dragging = True
                self.current_event = event  # Update modifier state
                # Clear selection unless Ctrl is held
                if not (event.state & 0x0004):
                    self.selected_indices.clear()
            else:
                return  # Not enough movement yet

        # Auto-scroll if near edges
        self._handle_auto_scroll(event)
        self.current_event = event  # Keep tracking for continued scrolling

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # ALWAYS update selection box immediately (no throttling for visual feedback)
        if self.selection_box:
            self.canvas.delete(self.selection_box)

        x1, y1 = self.drag_start
        self.selection_box = self.canvas.create_rectangle(
            x1,
            y1,
            canvas_x,
            canvas_y,
            outline=self.theme_colors.get("accent_blue", "#2196F3"),
            width=2,
            dash=(5, 5),
        )

        # Raise the selection box above all other items
        try:
            self.canvas.tag_raise(self.selection_box)
        except:
            pass

        # Only throttle cell selection updates (not the box drawing)
        now = time.time()
        if now - self._drag_last_ts >= 0.05:  # 50ms throttle for cell selection only
            self._drag_last_ts = now
            # Find cells in selection
            self._select_cells_in_box(x1, y1, canvas_x, canvas_y)

    def _handle_auto_scroll(self, event):
        """Auto-scroll canvas if mouse is near edges during drag"""
        widget_height = self.canvas.winfo_height()
        edge_zone = 80  # Pixels from edge to trigger scroll

        # Calculate scroll speed based on distance from edge
        scroll_amount = 0

        if event.y < edge_zone:
            # Near top - scroll up with GRADUAL speed
            distance_from_edge = edge_zone - event.y
            # More gradual speed calculation: 1 unit per 30px, max 3
            scroll_amount = -min(3, max(1, distance_from_edge // 30))
            self.logger.debug(
                f"Auto-scroll UP: distance={distance_from_edge}px, speed={abs(scroll_amount)}"
            )

        elif event.y > widget_height - edge_zone:
            # Near bottom - scroll down with GRADUAL speed
            distance_from_edge = event.y - (widget_height - edge_zone)
            # More gradual speed calculation: 1 unit per 30px, max 3
            scroll_amount = min(3, max(1, distance_from_edge // 30))
            self.logger.debug(
                f"Auto-scroll DOWN: distance={distance_from_edge}px, speed={scroll_amount}"
            )

        if scroll_amount != 0:
            self.canvas.yview_scroll(scroll_amount, "units")
            if self.dragging:
                # Store scroll params for continued scrolling
                self._auto_scroll_amount = scroll_amount
                # Increased delay for smoother scrolling
                self.canvas.after(100, self._check_continued_scroll)
        else:
            self._auto_scroll_amount = 0

    def _check_continued_scroll(self):
        """Continue scrolling if still dragging in edge zone"""
        if not self.dragging or not self.current_event:
            self._auto_scroll_amount = 0
            self.logger.debug("Stopped auto-scroll - no longer dragging")
            return

        # Continue with stored scroll amount for smoother scrolling
        if hasattr(self, "_auto_scroll_amount") and self._auto_scroll_amount != 0:
            self.canvas.yview_scroll(self._auto_scroll_amount, "units")
            self.logger.debug(
                f"Continuing auto-scroll: amount={self._auto_scroll_amount}"
            )

        # Re-trigger auto-scroll check
        self._handle_auto_scroll(self.current_event)

    def _on_mouse_motion(self, event):
        """Handle mouse motion for hover effects"""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        cell = self._get_cell_at(canvas_x, canvas_y)

        if cell != self.hover_cell:
            self.hover_cell = cell

            # Update zoom preview if open
            if cell and self.zoom_preview and self.zoom_preview.is_open:
                idx = self.cells[cell]["idx"]
                if idx < len(self.displayed_images):
                    img = self.displayed_images[idx]
                    if not (hasattr(img, "is_placeholder") and img.is_placeholder):
                        self.zoom_preview.update_image(img)

    def _on_mouse_leave(self, event):
        """Handle mouse leaving canvas"""
        self.hover_cell = None

    def _on_right_click(self, event):
        """Handle right click - toggle zoom preview"""
        if not self.zoom_preview:
            # This will be set by parent
            self.logger.debug("Zoom preview not available")
        else:
            if self.zoom_preview.is_open:
                self.zoom_preview.hide()
            else:
                self.zoom_preview.show()

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling with improved throttling"""
        # Standard scrolling
        scroll_amount = int(-1 * (event.delta / 120))
        self.canvas.yview_scroll(scroll_amount, "units")

        # Initialize throttle attributes if needed
        if not hasattr(self, "_scroll_timer"):
            self._scroll_timer = None
        if not hasattr(self, "_last_scroll_time"):
            self._last_scroll_time = 0
        if not hasattr(self, "_scroll_event_count"):
            self._scroll_event_count = 0

        # Track scroll events
        self._scroll_event_count += 1

        # Cancel any pending scroll timer
        if self._scroll_timer:
            self.canvas.after_cancel(self._scroll_timer)
            self._scroll_timer = None

        current_time = time.time()

        # More intelligent throttling based on scroll frequency
        # If many scroll events, increase throttle time
        throttle_time = 0.1 if self._scroll_event_count < 5 else 0.2

        # Only update immediately if enough time has passed
        if current_time - self._last_scroll_time > throttle_time:
            self.logger.debug(
                f"Mousewheel: immediate update (delta={event.delta}, events={self._scroll_event_count})"
            )
            self._update_lazy_load()
            self._last_scroll_time = current_time
            self._scroll_event_count = 0  # Reset event count

        # Always schedule a final update after scrolling stops
        self._scroll_timer = self.canvas.after(300, self._final_scroll_update)

    def _final_scroll_update(self):
        """Final update after scrolling stops"""
        self.logger.info("Scroll stopped - final lazy load update")
        self._scroll_timer = None
        self._update_lazy_load()

    def _on_ctrl_mousewheel(self, event):
        """Handle Ctrl+MouseWheel for zoom"""
        if event.delta > 0:
            factor = 1.1
        else:
            factor = 0.9

        new_scale = self.scale_factor * factor
        new_scale = max(0.3, min(5.0, new_scale))  # 0.3x to 5x zoom

        if new_scale != self.scale_factor:
            self.set_scale(new_scale)

    def _on_canvas_configure(self, event):
        """Handle canvas configuration changes - triggers lazy loading"""
        # Trigger lazy load update when canvas is resized or first rendered
        if self.displayed_images:
            self._update_lazy_load()

    # Helper methods
    def _get_cell_at(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """Get cell at canvas coordinates"""
        # Account for visualization columns
        viz_width = (
            len(self.viz_columns) * self.viz_column_width
            if self.show_data_visualizations
            else 0
        )

        total_cell_width = self.cell_width + viz_width

        col = int(x // total_cell_width)
        row = int(y // self.cell_height)

        if (row, col) in self.cells:
            return (row, col)
        return None

    def _select_cells_in_box(self, x1: float, y1: float, x2: float, y2: float):
        """Select all cells within a box"""
        # Normalize coordinates
        min_x, max_x = (x1, x2) if x1 < x2 else (x2, x1)
        min_y, max_y = (y1, y2) if y1 < y2 else (y2, y1)

        # Only clear if not holding Ctrl (for additive selection)
        if not (
            hasattr(self, "current_event")
            and self.current_event
            and self.current_event.state & 0x0004
        ):
            self.selected_indices.clear()

        viz_width = (
            len(self.viz_columns) * self.viz_column_width
            if self.show_data_visualizations
            else 0
        )
        total_cell_width = self.cell_width + viz_width

        for (row, col), cell_data in self.cells.items():
            cell_x = col * total_cell_width
            cell_y = row * self.cell_height

            # Check if cell overlaps with selection box
            if (
                cell_x < max_x
                and cell_x + self.cell_width > min_x
                and cell_y < max_y
                and cell_y + self.cell_height > min_y
            ):
                idx = cell_data["idx"]
                # Skip placeholder images
                if idx < len(self.displayed_images):
                    img = self.displayed_images[idx]
                    if not (hasattr(img, "is_placeholder") and img.is_placeholder):
                        self.selected_indices.add(idx)

        self._update_selection_visual()
        self._on_selection_change(list(self.selected_indices))

    def _update_selection_visual(self):
        """Update visual state of selected cells"""
        for (row, col), cell_data in self.cells.items():
            idx = cell_data["idx"]
            is_selected = idx in self.selected_indices

            # Update cell background or border
            cell_id_dict = self.cell_ids.get((row, col), {})
            # Try border first, fall back to placeholder
            border_id = cell_id_dict.get("border") or cell_id_dict.get("placeholder")

            if border_id:
                if is_selected:
                    self.canvas.itemconfig(
                        border_id, outline=self.theme_colors["accent_blue"], width=3
                    )
                else:
                    # Get proper color based on classification if cell is loaded
                    if idx in self.loaded_cells and idx < len(self.displayed_images):
                        img = self.displayed_images[idx]
                        color = self._get_classification_border_color(img)
                        width = self._get_classification_border_width(img)
                    else:
                        color = self.theme_colors["secondary_bg"]
                        width = 1
                    self.canvas.itemconfig(border_id, outline=color, width=width)

    def _calculate_visible_range(self):
        """Calculate which images should be visible/loaded"""
        canvas_height = self.canvas.winfo_height()
        view_top = self.canvas.canvasy(0)
        view_bottom = self.canvas.canvasy(canvas_height)

        print(f"DEBUG _calculate_visible_range():")
        print(f"  canvas_height: {canvas_height}")
        print(f"  view_top: {view_top}, view_bottom: {view_bottom}")

        if canvas_height <= 1:
            result = (0, min(40, len(self.displayed_images)))
            print(f"  Canvas height <= 1, returning: {result}")
            return result

        row_height = self.cell_height + self.padding
        if row_height <= 0:
            return (0, 0)

        start_row = max(0, int(view_top // row_height) - self.load_buffer)
        end_row = int(view_bottom // row_height) + self.load_buffer + 1

        start_idx = start_row * self.cols_per_row
        end_idx = min(end_row * self.cols_per_row, len(self.displayed_images))

        return (start_idx, end_idx)

    def _calculate_columns(self):
        """Calculate number of columns that fit in canvas width"""
        canvas_width = self.canvas.winfo_width()
        if canvas_width <= 1:
            canvas_width = 800  # Default width

        # Calculate effective cell width including viz columns if shown
        viz_width = (
            self.viz_column_width
            if self.show_data_visualizations and self.viz_columns
            else 0
        )
        total_cell_width = self.cell_width + viz_width

    def _update_lazy_load(self):
        """Update which cells are loaded based on viewport"""
        new_range = self._calculate_visible_range()

        self.logger.debug(
            f"_update_lazy_load: current={self.visible_range}, new={new_range}, "
            f"queue_size={self.load_queue.qsize()}, pending={len(self.pending_loads)}"
        )

        if new_range != self.visible_range:
            old_start = self.visible_range[0] if self.visible_range else 0
            new_start = new_range[0]

            # Check for large jumps (lower threshold for better responsiveness)
            jump_distance = abs(new_start - old_start)
            if jump_distance > 50:  # Lowered threshold for better queue management
                self.logger.warning(
                    f"Large jump detected: {jump_distance} items - clearing queue and pending loads"
                )

                # Clear the entire load queue
                cleared = 0
                while not self.load_queue.empty():
                    try:
                        self.load_queue.get_nowait()
                        cleared += 1
                    except:
                        break

                # Clear ALL pending loads on large jumps
                old_pending = len(self.pending_loads)
                self.pending_loads.clear()

                self.logger.info(
                    f"Cleared {cleared} queued items, removed {old_pending} pending loads"
                )
            else:
                # For smaller jumps, only clear items that are now far away
                self.pending_loads = {
                    idx
                    for idx in self.pending_loads
                    if new_range[0] - 50 <= idx <= new_range[1] + 50
                }

            self.visible_range = new_range
            self._load_visible_cells()
        else:
            self.logger.debug("Range unchanged, skipping load")

    def _load_visible_cells(self):
        """Queue visible cells for loading"""
        if not self.visible_range or not self.displayed_images:
            return

        start_idx, end_idx = self.visible_range

        queued_count = 0
        already_pending = 0
        already_loaded = 0

        # Queue cells for loading in the background
        for idx in range(start_idx, end_idx):
            if idx >= len(self.displayed_images):
                break

            if idx in self.loaded_cells:
                already_loaded += 1
            elif idx in self.pending_loads:
                already_pending += 1
            else:
                self.pending_loads.add(idx)
                self.load_queue.put(idx)
                queued_count += 1

        self.logger.debug(
            f"_load_visible_cells: queued={queued_count}, pending={already_pending}, loaded={already_loaded}"
        )
        if queued_count > 0:
            self.logger.info(
                f"Queued {queued_count} new cells for loading (range: {start_idx}-{end_idx})"
            )

    def _load_cell(self, idx: int):
        """Queue a cell for loading"""
        if idx not in self.loaded_cells and idx not in self.pending_loads:
            print(f"DEBUG _load_cell({idx}): Adding to queue")
            self.pending_loads.add(idx)
            self.load_queue.put(idx)

    def _load_cell_impl(self, idx: int):
        """Load single cell - runs in background thread"""
        if idx >= len(self.displayed_images):
            return

        try:
            img = self.displayed_images[idx]

            # Check placeholder
            if hasattr(img, "is_placeholder") and img.is_placeholder:
                self.canvas.after(0, lambda: self._display_placeholder_cell(idx))
                return

            # Populate CSV data if needed
            if self.show_data_visualizations and not img.csv_data:
                if self.dialog_ref and hasattr(self.dialog_ref, "_get_csv_data_cached"):
                    csv_data, in_csv = self.dialog_ref._get_csv_data_cached(
                        img.hole_id, img.depth_to
                    )
                    if isinstance(csv_data, dict) and isinstance(
                        csv_data.get("review_data"), dict
                    ):
                        csv_data = csv_data["review_data"]
                    img.csv_data = csv_data
                    if hasattr(img, "in_csv"):
                        img.in_csv = in_csv

            # Load image using old code's _prepare_thumbnail approach
            photo = self._prepare_thumbnail_old_style(
                img, self.cell_width - 10, self.cell_height - 10
            )

            if photo:
                # Cache reference
                self.image_refs[idx] = photo
                # Schedule display
                self.canvas.after_idle(
                    lambda i=idx, p=photo, im=img: self._display_cell(i, p, im)
                )
            else:
                self.canvas.after(0, lambda: self._display_no_image_cell(idx))

        except Exception as e:
            self.logger.error(f"Failed to load cell {idx}: {e}")
            self.canvas.after(0, lambda: self._display_error_cell(idx))

    def _create_cell_ui(self, row: int, col: int, idx: int, img):
        """Create cell UI elements (must be called on main thread)"""
        if (row, col) not in self.cells:
            return

        cell_data = self.cells[(row, col)]
        x, y = cell_data["x"], cell_data["y"]

        # Remove placeholder if exists
        cell_id_dict = self.cell_ids.get((row, col), {})
        placeholder_id = cell_id_dict.get("placeholder")
        if placeholder_id:
            try:
                self.canvas.delete(placeholder_id)
            except:
                pass

        # Create image
        if img._image is not None:
            try:
                # Apply rotation
                rotated = img._image
                if self.rotation == 90:
                    rotated = cv2.rotate(img._image, cv2.ROTATE_90_CLOCKWISE)
                elif self.rotation == 180:
                    rotated = cv2.rotate(img._image, cv2.ROTATE_180)
                elif self.rotation == 270:
                    rotated = cv2.rotate(img._image, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # Resize to fit cell
                h, w = rotated.shape[:2]
                scale = min(self.cell_width / w, self.cell_height / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                resized = cv2.resize(rotated, (new_w, new_h))

                # Convert to PhotoImage
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                pil_img = PILImage.fromarray(rgb)
                photo = ImageTk.PhotoImage(pil_img)

                # Center image in cell
                img_x = x + (self.cell_width - new_w) // 2
                img_y = y + (self.cell_height - new_h) // 2

                img_id = self.canvas.create_image(
                    img_x,
                    img_y,
                    image=photo,
                    anchor=tk.NW,
                    tags=(f"image_{idx}", f"cell_{idx}"),
                )

                # Keep reference
                self.image_refs[idx] = photo

            except Exception as e:
                self.logger.error(f"Failed to create cell image: {e}")
                img_id = None
        else:
            img_id = None

        # Create border with classification styling
        border_color = self._get_classification_border_color(img)
        border_width = self._get_classification_border_width(img)
        dash_pattern = self._get_classification_dash_pattern(img)

        border_id = self.canvas.create_rectangle(
            x,
            y,
            x + self.cell_width,
            y + self.cell_height,
            outline=border_color,
            width=border_width,
            dash=dash_pattern if dash_pattern else (),
            tags=(f"border_{idx}", f"cell_{idx}"),
        )

        # Create label
        label_text = f"{img.hole_id}\n{int(img.depth_to)}m"
        if hasattr(img, "moisture_status") and img.moisture_status:
            label_text += f"\n{img.moisture_status}"

        label_id = self.canvas.create_text(
            x + 5,
            y + 5,
            text=label_text,
            anchor=tk.NW,
            fill=self.theme_colors.get("text", "#ffffff"),
            font=("Arial", 9, "bold"),
            tags=(f"label_{idx}", f"cell_{idx}"),
        )

        # Store cell IDs
        self.cell_ids[(row, col)] = {
            "border": border_id,
            "image": img_id,
            "label": label_id,
        }

        # Mark as loaded
        self.loaded_cells.add(idx)
        self.pending_loads.discard(idx)

    def _get_classification_border_color(self, img) -> str:
        """Get border color based on classification (case-insensitive, PEP-8)"""
        if not hasattr(img, "classification") or not img.classification:
            return self.theme_colors.get("border", "#666666")

        raw = str(img.classification).strip()
        if raw.startswith("ClassificationCategory."):
            raw = raw.split(".", 1)[1]

        # Normalize to a canonical key
        key = raw.replace("-", " ").replace("_", " ").strip().lower()

        # Map common variants; keep colors identical to previous behavior
        color_map = {
            "biff": "#4CAF50",  # BIFf
            "bif f": "#4CAF50",  # 'BIF F' variant
            "bifhm": "#2196F3",  # BIFhm
            "bif hm": "#2196F3",  # 'BIF HM' variant
            "other": "#FF9800",
            "not confident": "#FFC107",
            "unassigned": self.theme_colors.get("border", "#666666"),
        }

        return color_map.get(key, self.theme_colors.get("border", "#666666"))

    def _get_classification_border_width(self, img) -> int:
        """Border width based on classification and cell size (PEP-8)"""
        base_width = max(1, min(4, self.cell_width // 80))

        if not hasattr(img, "classification") or not img.classification:
            return base_width

        s = str(img.classification).strip()
        s_norm = s.split(".", 1)[1] if s.startswith("ClassificationCategory.") else s
        if s_norm.strip().lower() in {"", "unassigned"}:
            return base_width

        return min(8, base_width * 2)

    def _get_classification_dash_pattern(self, img) -> tuple:
        """Dash pattern: dotted for unsaved classified, solid otherwise (PEP-8)"""
        if hasattr(img, "classified_date") and img.classified_date:
            return None

        if hasattr(img, "classification") and img.classification:
            s = str(img.classification).strip()
            s_norm = (
                s.split(".", 1)[1] if s.startswith("ClassificationCategory.") else s
            )
            if s_norm.strip().lower() not in {"", "unassigned"}:
                return (5, 3)

        return None

    def _display_placeholder(self, idx: int):
        """Display a placeholder cell for pending images"""
        row = idx // self.cols_per_row
        col = idx % self.cols_per_row

        if (row, col) not in self.cells:
            return

        cell_data = self.cells[(row, col)]
        x = cell_data["x"]
        y = cell_data["y"]

        # Update placeholder appearance
        cell_id_dict = self.cell_ids.get((row, col), {})
        placeholder_id = cell_id_dict.get("placeholder")
        if placeholder_id:
            self.canvas.itemconfig(
                placeholder_id,
                fill=self.theme_colors["secondary_bg"],
                outline=self.theme_colors["field_border"],
            )

        # Add "Image Pending" text
        text_id = self.canvas.create_text(
            x + self.cell_width // 2,
            y + self.cell_height // 2,
            text="Image Pending",
            fill=self.theme_colors["subtext"],
            font=("Arial", 9),
            tags=(f"text_{idx}", f"cell_{idx}"),
        )

        self.loaded_cells.add(idx)
        self.pending_loads.discard(idx)

    def _display_error_cell(self, idx: int):
        """Display an error indicator for failed loads"""
        row = idx // self.cols_per_row
        col = idx % self.cols_per_row

        if (row, col) not in self.cells:
            return

        cell_data = self.cells[(row, col)]
        x = cell_data["x"]
        y = cell_data["y"]

        # Update cell appearance
        cell_id = self.cell_ids.get((row, col))
        if cell_id:
            self.canvas.itemconfig(
                cell_id,
                fill="#3a1a1a",  # Dark red
                outline=self.theme_colors["accent_red"],
            )

        # Add error text
        text_id = self.canvas.create_text(
            x + self.cell_width // 2,
            y + self.cell_height // 2,
            text="Load Error",
            fill=self.theme_colors["accent_red"],
            font=("Arial", 9, "bold"),
            tags=(f"text_{idx}", f"cell_{idx}"),
        )

        self.loaded_cells.add(idx)
        self.pending_loads.discard(idx)

    def _display_loaded_image(self, idx: int, photo: ImageTk.PhotoImage, img):
        """Display loaded image, replacing placeholder"""
        row = idx // self.cols_per_row
        col = idx % self.cols_per_row

        if (row, col) not in self.cells:
            return

        cell_data = self.cells[(row, col)]
        x = cell_data["x"]
        y = cell_data["y"]

        # DELETE placeholder and loading text
        cell_ids = self.cell_ids.get((row, col), {})
        if cell_ids.get("placeholder"):
            self.canvas.delete(cell_ids["placeholder"])
        if cell_ids.get("loading_text"):
            self.canvas.delete(cell_ids["loading_text"])

        # Create EMPTY background (so image shows through)
        bg_id = self.canvas.create_rectangle(
            x,
            y,
            x + self.cell_width,
            y + self.cell_height,
            fill="",
            outline="",
            tags=(f"bg_{idx}", f"cell_{idx}"),
        )

        # Create image
        center_x = x + self.cell_width // 2
        center_y = y + self.cell_height // 2
        img_id = self.canvas.create_image(
            center_x,
            center_y,
            image=photo,
            anchor="center",
            tags=(f"image_{idx}", f"cell_{idx}"),
        )

        # Create border
        border_color = self._get_classification_border_color(img)
        border_width = self._get_classification_border_width(img)
        dash_pattern = self._get_classification_dash_pattern(img)
        border_id = self.canvas.create_rectangle(
            x,
            y,
            x + self.cell_width,
            y + self.cell_height,
            outline=border_color,
            width=border_width,
            dash=dash_pattern if dash_pattern else (),
            tags=(f"border_{idx}", f"cell_{idx}"),
        )

        # Add label
        label_text = f"{img.hole_id}\n{int(img.depth_to)}m"
        if hasattr(img, "moisture_status") and img.moisture_status:
            label_text += f"\n{img.moisture_status}"
        label_id = self.canvas.create_text(
            x + 5,
            y + 5,
            text=label_text,
            anchor="nw",
            fill=self.theme_colors.get("text", "#ffffff"),
            font=("Arial", 9, "bold"),
            tags=(f"label_{idx}", f"cell_{idx}"),
        )

        # Add peer review info if enabled
        if self.show_peer_info:
            self._add_peer_review_overlay(idx, x, y, img)

        # Add comment indicator if there are comments
        self._add_comment_indicator(idx, x, y, img)

        # Update cell_ids
        self.cell_ids[(row, col)] = {
            "bg": bg_id,
            "image": img_id,
            "border": border_id,
            "label": label_id,
        }

        # Mark as loaded
        self.loaded_cells.add(idx)
        self.pending_loads.discard(idx)

    def _add_peer_review_overlay(self, idx: int, x: int, y: int, img):
        """Add peer review information overlay to a cell"""
        if not self.peer_review_manager:
            return

        # Get peer review info
        peer_reviews = self.peer_review_manager.get_peer_reviews(img)
        if not peer_reviews:
            return

        # Count reviewers and check for conflicts
        reviewer_count = len(
            set(r.get("Reviewed_By") for r in peer_reviews if r.get("Reviewed_By"))
        )

        # Check if current user also reviewed
        if (
            hasattr(img, "classification")
            and img.classification
            and img.classification != "Unassigned"
        ):
            reviewer_count += 1

        # Check for conflicts
        has_conflicts = self.peer_review_manager.has_review_conflicts(img)

        # Create peer info indicator in top-right corner
        indicator_size = 20
        indicator_x = x + self.cell_width - indicator_size - 5
        indicator_y = y + 5

        # Determine indicator color
        if has_conflicts:
            indicator_color = self.theme_colors.get("accent_red", "#f44336")
            indicator_text = "!"
        else:
            indicator_color = self.theme_colors.get("accent_green", "#4CAF50")
            indicator_text = str(reviewer_count)

        # Draw circular indicator background
        indicator_bg_id = self.canvas.create_oval(
            indicator_x,
            indicator_y,
            indicator_x + indicator_size,
            indicator_y + indicator_size,
            fill=indicator_color,
            outline=self.theme_colors.get("text", "#ffffff"),
            width=2,
            tags=(f"peer_indicator_{idx}", f"cell_{idx}"),
        )

        # Draw indicator text
        indicator_text_id = self.canvas.create_text(
            indicator_x + indicator_size // 2,
            indicator_y + indicator_size // 2,
            text=indicator_text,
            fill=self.theme_colors.get("text", "#ffffff"),
            font=("Arial", 10, "bold"),
            tags=(f"peer_text_{idx}", f"cell_{idx}"),
        )

        # Add to cell_ids tracking
        row = idx // self.cols_per_row
        col = idx % self.cols_per_row
        if (row, col) in self.cell_ids:
            self.cell_ids[(row, col)]["peer_indicator"] = indicator_bg_id
            self.cell_ids[(row, col)]["peer_text"] = indicator_text_id

        # Create tooltip text for hover
        classifications = {}
        for review in peer_reviews:
            reviewer = review.get("Reviewed_By", "Unknown")
            classification = review.get("Classification", "Unknown")
            classifications[reviewer] = classification

        # Add current user's classification if exists
        if (
            hasattr(img, "classification")
            and img.classification
            and img.classification != "Unassigned"
        ):
            current_user = getattr(img, "classified_by", "Current User")
            classifications[current_user] = str(img.classification)

        # Store tooltip info for hover display
        if not hasattr(self, "peer_tooltips"):
            self.peer_tooltips = {}
        self.peer_tooltips[idx] = classifications

    def _add_comment_indicator(self, idx: int, x: int, y: int, img):
        """Add comment indicator to a cell if comments exist"""
        comments = []

        # Check current user's comment
        if hasattr(img, "comments") and img.comments and img.comments.strip():
            comments.append(
                {
                    "reviewer": getattr(img, "classified_by", "Current User"),
                    "comment": img.comments.strip(),
                    "date": getattr(img, "classified_date", ""),
                }
            )

        # Check peer reviews for comments (if peer review manager is available)
        if self.peer_review_manager:
            peer_reviews = self.peer_review_manager.get_peer_reviews(img)
            for review in peer_reviews:
                if review.get("Comments") and review["Comments"].strip():
                    comments.append(
                        {
                            "reviewer": review.get("Reviewed_By", "Unknown"),
                            "comment": review["Comments"].strip(),
                            "date": review.get("Review_Date", ""),
                        }
                    )

        # If no comments, return
        if not comments:
            return

        # Create comment indicator in bottom-right corner
        indicator_size = 20
        indicator_x = x + self.cell_width - indicator_size - 5
        indicator_y = y + self.cell_height - indicator_size - 5

        # If peer review indicator exists, offset the comment indicator
        if (
            self.show_peer_info
            and hasattr(self, "peer_tooltips")
            and idx in self.peer_tooltips
        ):
            indicator_x -= 25  # Move left to avoid overlap with peer indicator

        # Draw comment bubble icon
        bubble_id = self.canvas.create_oval(
            indicator_x,
            indicator_y,
            indicator_x + indicator_size,
            indicator_y + indicator_size,
            fill=self.theme_colors.get("accent_yellow", "#FFC107"),
            outline=self.theme_colors.get("text", "#ffffff"),
            width=1,
            tags=(f"comment_indicator_{idx}", f"cell_{idx}"),
        )

        # Draw comment count
        comment_count = len(comments)
        count_text = str(comment_count) if comment_count < 10 else "9+"

        text_id = self.canvas.create_text(
            indicator_x + indicator_size // 2,
            indicator_y + indicator_size // 2,
            text=count_text,
            fill=self.theme_colors.get("background", "#1e1e1e"),
            font=("Arial", 9, "bold"),
            tags=(f"comment_text_{idx}", f"cell_{idx}"),
        )

        # Add small tail to make it look like a speech bubble
        tail_points = [
            indicator_x + 3,
            indicator_y + indicator_size - 2,
            indicator_x - 2,
            indicator_y + indicator_size + 3,
            indicator_x + 6,
            indicator_y + indicator_size,
        ]

        tail_id = self.canvas.create_polygon(
            tail_points,
            fill=self.theme_colors.get("accent_yellow", "#FFC107"),
            outline="",
            tags=(f"comment_tail_{idx}", f"cell_{idx}"),
        )

        # Store comment data for tooltip
        if not hasattr(self, "comment_tooltips"):
            self.comment_tooltips = {}
        self.comment_tooltips[idx] = comments

        # Add to cell_ids tracking
        row = idx // self.cols_per_row
        col = idx % self.cols_per_row
        if (row, col) in self.cell_ids:
            self.cell_ids[(row, col)]["comment_indicator"] = bubble_id
            self.cell_ids[(row, col)]["comment_text"] = text_id
            self.cell_ids[(row, col)]["comment_tail"] = tail_id

    def _on_canvas_motion(self, event):
        """Handle mouse motion for showing tooltips"""
        # Convert canvas coordinates to cell index
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # Find which cell we're hovering over
        col = int(canvas_x // (self.cell_width + self.padding))
        row = int(canvas_y // (self.cell_height + self.padding))

        if col >= self.cols_per_row or row < 0:
            self._hide_tooltip()
            return

        idx = row * self.cols_per_row + col

        # Check if this is a valid cell with an image
        if idx >= len(self.displayed_images):
            self._hide_tooltip()
            return

        # Check if we're hovering over a different cell
        if hasattr(self, "tooltip_cell_idx") and self.tooltip_cell_idx == idx:
            return  # Still on same cell

        self.tooltip_cell_idx = idx

        # Hide existing tooltip
        self._hide_tooltip()

        # Set timer to show tooltip after delay
        if self.tooltip_timer:
            self.canvas.after_cancel(self.tooltip_timer)

        self.tooltip_timer = self.canvas.after(
            500,  # 500ms delay before showing tooltip
            lambda: self._show_tooltip_for_cell(idx, event.x_root, event.y_root),
        )

    def _on_canvas_leave(self, event):
        """Handle mouse leaving canvas"""
        self._hide_tooltip()
        if self.tooltip_timer:
            self.canvas.after_cancel(self.tooltip_timer)
            self.tooltip_timer = None

    def _show_tooltip_for_cell(self, idx: int, x: int, y: int):
        """Show tooltip for a specific cell"""
        # Check if we have comments or peer reviews for this cell
        has_comments = (
            hasattr(self, "comment_tooltips") and idx in self.comment_tooltips
        )
        has_peer_info = hasattr(self, "peer_tooltips") and idx in self.peer_tooltips

        if not has_comments and not has_peer_info:
            return

        # Create tooltip window
        self.current_tooltip = tk.Toplevel(self.canvas)
        self.current_tooltip.overrideredirect(True)
        self.current_tooltip.configure(
            bg=self.theme_colors.get("secondary_bg", "#3e3e3e")
        )

        # Create main frame with padding
        main_frame = tk.Frame(
            self.current_tooltip,
            bg=self.theme_colors.get("secondary_bg", "#3e3e3e"),
            padx=12,
            pady=8,
            highlightbackground=self.theme_colors.get("border", "#666666"),
            highlightthickness=1,
        )
        main_frame.pack()

        # Get image info
        if idx < len(self.displayed_images):
            img = self.displayed_images[idx]

            # Add header with image info
            header_text = f"{img.hole_id} - {int(img.depth_to)}m"
            tk.Label(
                main_frame,
                text=header_text,
                bg=self.theme_colors.get("secondary_bg", "#3e3e3e"),
                fg=self.theme_colors.get("text", "#ffffff"),
                font=("Arial", 11, "bold"),
            ).pack(anchor="w", pady=(0, 5))

            # Add separator
            ttk.Separator(main_frame, orient="horizontal").pack(fill="x", pady=2)

        # Add peer review info if available
        if has_peer_info:
            classifications = self.peer_tooltips[idx]

            tk.Label(
                main_frame,
                text="Classifications:",
                bg=self.theme_colors.get("secondary_bg", "#3e3e3e"),
                fg=self.theme_colors.get("accent_blue", "#2196F3"),
                font=("Arial", 10, "bold"),
            ).pack(anchor="w", pady=(5, 2))

            for reviewer, classification in classifications.items():
                # Clean up classification text
                if "ClassificationCategory." in str(classification):
                    classification = str(classification).replace(
                        "ClassificationCategory.", ""
                    )

                reviewer_frame = tk.Frame(
                    main_frame, bg=self.theme_colors.get("secondary_bg", "#3e3e3e")
                )
                reviewer_frame.pack(anchor="w", padx=(10, 0))

                tk.Label(
                    reviewer_frame,
                    text="• ",
                    bg=self.theme_colors.get("secondary_bg", "#3e3e3e"),
                    fg=self.theme_colors.get("subtext", "#aaaaaa"),
                    font=("Arial", 9),
                ).pack(side="left")

                tk.Label(
                    reviewer_frame,
                    text=f"{reviewer}:",
                    bg=self.theme_colors.get("secondary_bg", "#3e3e3e"),
                    fg=self.theme_colors.get("subtext", "#aaaaaa"),
                    font=("Arial", 9, "italic"),
                ).pack(side="left")

                tk.Label(
                    reviewer_frame,
                    text=f" {classification}",
                    bg=self.theme_colors.get("secondary_bg", "#3e3e3e"),
                    fg=self._get_classification_color_for_tooltip(classification),
                    font=("Arial", 9, "bold"),
                ).pack(side="left")

        # Add comments if available
        if has_comments:
            comments = self.comment_tooltips[idx]

            # Add separator if we also have peer info
            if has_peer_info:
                ttk.Separator(main_frame, orient="horizontal").pack(fill="x", pady=5)

            tk.Label(
                main_frame,
                text="Comments:",
                bg=self.theme_colors.get("secondary_bg", "#3e3e3e"),
                fg=self.theme_colors.get("accent_yellow", "#FFC107"),
                font=("Arial", 10, "bold"),
            ).pack(anchor="w", pady=(5, 2))

            for comment_data in comments:
                # Comment frame
                comment_frame = tk.Frame(
                    main_frame,
                    bg=self.theme_colors.get("field_bg", "#2e2e2e"),
                    relief=tk.FLAT,
                    bd=1,
                )
                comment_frame.pack(fill="x", padx=(10, 0), pady=2)

                # Reviewer and date header
                header_frame = tk.Frame(
                    comment_frame, bg=self.theme_colors.get("field_bg", "#2e2e2e")
                )
                header_frame.pack(fill="x", padx=5, pady=(3, 0))

                tk.Label(
                    header_frame,
                    text=comment_data["reviewer"],
                    bg=self.theme_colors.get("field_bg", "#2e2e2e"),
                    fg=self.theme_colors.get("text", "#ffffff"),
                    font=("Arial", 9, "italic"),
                ).pack(side="left")

                if comment_data.get("date"):
                    tk.Label(
                        header_frame,
                        text=f" - {comment_data['date']}",
                        bg=self.theme_colors.get("field_bg", "#2e2e2e"),
                        fg=self.theme_colors.get("subtext", "#888888"),
                        font=("Arial", 8),
                    ).pack(side="left")

                # Comment text (with word wrap)
                comment_text = tk.Text(
                    comment_frame,
                    bg=self.theme_colors.get("field_bg", "#2e2e2e"),
                    fg=self.theme_colors.get("text", "#ffffff"),
                    font=("Arial", 9),
                    width=40,
                    height=self._calculate_text_height(comment_data["comment"], 40),
                    wrap="word",
                    relief=tk.FLAT,
                    padx=5,
                    pady=2,
                )
                comment_text.pack(fill="x")
                comment_text.insert("1.0", comment_data["comment"])
                comment_text.configure(state="disabled")  # Make read-only

        # Position tooltip near cursor but ensure it stays on screen
        self.current_tooltip.update_idletasks()
        tooltip_width = self.current_tooltip.winfo_reqwidth()
        tooltip_height = self.current_tooltip.winfo_reqheight()

        screen_width = self.current_tooltip.winfo_screenwidth()
        screen_height = self.current_tooltip.winfo_screenheight()

        # Adjust position to keep tooltip on screen
        tooltip_x = x + 10
        tooltip_y = y + 10

        if tooltip_x + tooltip_width > screen_width:
            tooltip_x = x - tooltip_width - 10

        if tooltip_y + tooltip_height > screen_height:
            tooltip_y = y - tooltip_height - 10

        self.current_tooltip.geometry(f"+{tooltip_x}+{tooltip_y}")

    def _hide_tooltip(self):
        """Hide the current tooltip"""
        if self.current_tooltip:
            self.current_tooltip.destroy()
            self.current_tooltip = None

        if hasattr(self, "tooltip_cell_idx"):
            delattr(self, "tooltip_cell_idx")

    def _calculate_text_height(self, text: str, width: int) -> int:
        """Calculate required height for text widget based on content"""
        import textwrap

        lines = text.split("\n")
        total_lines = 0
        for line in lines:
            wrapped = textwrap.wrap(line, width=width)
            total_lines += len(wrapped) if wrapped else 1
        return min(max(total_lines, 1), 6)  # Minimum 1, maximum 6 lines

    def _get_classification_color_for_tooltip(self, classification: str) -> str:
        """Get color for classification text in tooltip"""
        color_map = {
            "BIFF": "#4CAF50",
            "BIFf": "#4CAF50",
            "BIFHM": "#2196F3",
            "BIF HM": "#2196F3",
            "OTHER": "#FF9800",
            "Other": "#FF9800",
            "NOT_CONFIDENT": "#FFC107",
            "Not Confident": "#FFC107",
            "UNASSIGNED": "#888888",
            "Unassigned": "#888888",
        }
        return color_map.get(classification, self.theme_colors.get("text", "#ffffff"))

    def _show_peer_tooltip(self, idx: int, x: int, y: int):
        """Show tooltip with peer review details"""
        if not hasattr(self, "peer_tooltips") or idx not in self.peer_tooltips:
            return

        classifications = self.peer_tooltips[idx]

        # Create tooltip window
        tooltip = tk.Toplevel(self.canvas)
        tooltip.overrideredirect(True)
        tooltip.configure(bg=self.theme_colors.get("secondary_bg", "#3e3e3e"))

        # Create content
        frame = tk.Frame(
            tooltip,
            bg=self.theme_colors.get("secondary_bg", "#3e3e3e"),
            padx=10,
            pady=5,
        )
        frame.pack()

        tk.Label(
            frame,
            text="Peer Reviews:",
            bg=self.theme_colors.get("secondary_bg", "#3e3e3e"),
            fg=self.theme_colors.get("text", "#ffffff"),
            font=("Arial", 10, "bold"),
        ).pack(anchor="w")

        for reviewer, classification in classifications.items():
            tk.Label(
                frame,
                text=f"• {reviewer}: {classification}",
                bg=self.theme_colors.get("secondary_bg", "#3e3e3e"),
                fg=self.theme_colors.get("subtext", "#aaaaaa"),
                font=("Arial", 9),
            ).pack(anchor="w")

        # Position near cursor
        tooltip.geometry(f"+{x}+{y}")

        # Store reference to destroy later
        self.peer_tooltip_window = tooltip

    def _hide_peer_tooltip(self):
        """Hide the peer review tooltip"""
        if hasattr(self, "peer_tooltip_window") and self.peer_tooltip_window:
            self.peer_tooltip_window.destroy()
            self.peer_tooltip_window = None

    def set_rotation(self, angle: int):
        """Set rotation angle and reload visible cells"""
        self.rotation = angle % 360

        # Unload all loaded cells
        for idx in list(self.loaded_cells):
            self._unload_cell(idx)

        # Reload visible range
        self._update_lazy_load()

    def _get_classification_color(self, classification: str) -> str:
        """Get color for classification indicator"""
        color_map = {
            "Waste": "#808080",
            "Ore": "#4CAF50",
            "Transitional": "#FFC107",
            "BIF": "#2196F3",
        }
        return color_map.get(classification, self.theme_colors["accent_blue"])

    def _draw_visualization_columns(self, idx: int, x: int, y: int, img):
        """Draw data visualization columns for a cell"""
        if not self.viz_columns or not img.csv_data:
            return

        viz_x = x + self.cell_width

        for i, col_name in enumerate(self.viz_columns):
            col_config = self.viz_column_configs.get(col_name, {})
            value = img.csv_data.get(col_name)

            if value is None:
                continue

            # Get color from color map
            color = self._get_viz_color(col_name, value, col_config)

            # Draw colored bar
            bar_id = self.canvas.create_rectangle(
                viz_x + (i * self.viz_column_width),
                y,
                viz_x + ((i + 1) * self.viz_column_width),
                y + self.cell_height,
                fill=color,
                outline="",
                tags=(f"viz_{idx}_{i}", f"cell_{idx}"),
            )

    def _add_data_visualizations(
        self, viz_x, viz_y, viz_width, viz_height, img, orientation, idx=0
    ):
        """
        [REPLACED] Add data visualization bars aligned to actual image dimensions.

        CRITICAL CHANGES:
        - Now respects rotation/orientation (vertical vs horizontal layout)
        - Properly stacks bars vertically for vertical orientation
        - Properly arranges bars horizontally for horizontal orientation
        - Uses BGR to hex conversion matching old dialog implementation
        - Calculates text color based on brightness for readability

        Args:
            viz_x: X coordinate for visualization area
            viz_y: Y coordinate for visualization area
            viz_width: Width of visualization area
            viz_height: Height of visualization area
            img: CompartmentImage with csv_data
            orientation: 'vertical' or 'horizontal' based on rotation
            idx: Image index for debugging
        """
        if not self.viz_columns or not hasattr(img, "csv_data"):
            return

        # Column name mappings for display
        column_display_names = getattr(
            self,
            "viz_column_labels",
            {
                "Fe_pct_BEST": "Fe%",
                "SiO2_pct_BEST": "SiO2%",
                "Al2O3_pct_BEST": "Al2O3%",
                "Logged_pct_CHHM": "CHHM%",
                "BIFf_2": "Type",
            },
        )

        num_columns = len(self.viz_columns)
        if num_columns == 0:
            return

        for i, column_name in enumerate(self.viz_columns):
            # Get value from csv_data
            # Strip source suffix if present (e.g., "fe_pct_best (drillhole_data)" -> "fe_pct_best")
            col_key = column_name.split(" (")[0].strip().lower() if " (" in column_name else column_name.lower()
            value = img.csv_data.get(col_key) if img.csv_data else None
            # Also try the original column name (case variations)
            if value is None and img.csv_data:
                # Try exact match and case variations
                for csv_key in img.csv_data.keys():
                    if csv_key.lower() == col_key:
                        value = img.csv_data[csv_key]
                        break
            display_name = column_display_names.get(column_name, column_name[:4])
            # Get color from color map - CRITICAL: use BGR format then convert
            color_hex = "#292828"  # Default gray
            if column_name in self.viz_column_configs and value is not None:
                color_map = self.viz_column_configs[column_name]
                color_bgr = color_map.get_color(value)
                # Convert BGR to hex - THIS IS THE KEY FIX
                color_hex = f"#{color_bgr[2]:02x}{color_bgr[1]:02x}{color_bgr[0]:02x}"

            # Calculate text color based on brightness
            r = int(color_hex[1:3], 16)
            g = int(color_hex[3:5], 16)
            b = int(color_hex[5:7], 16)
            brightness = (r * 299 + g * 587 + b * 114) / 1000
            text_color = "white" if brightness < 128 else "black"

            # Draw bar based on orientation - CRITICAL FIX
            if orientation == "vertical":
                # Vertical bars - single column, stacked vertically
                bar_width = viz_width  # Full width of viz area
                bar_height = (
                    viz_height / num_columns
                )  # Divide height by number of columns
                bar_x = viz_x
                bar_y = viz_y + (i * bar_height)

                # Draw bar in single column
                self.canvas.create_rectangle(
                    bar_x,
                    bar_y,
                    bar_x + bar_width,
                    bar_y + bar_height - 1,
                    fill=color_hex,
                    outline="",
                    tags=("viz_bar", f"viz_{idx}"),
                )

                # Vertical text
                if value is not None:
                    value_text = (
                        f"{value:.1f}" if isinstance(value, float) else str(value)[:4]
                    )
                else:
                    value_text = "-"

                # Draw text
                self.canvas.create_text(
                    bar_x + bar_width / 2,
                    bar_y + bar_height / 2,
                    text=f"{display_name}\n{value_text}",
                    fill=text_color,
                    font=("Arial", 7),
                    angle=0,
                    anchor="center",
                    tags=("viz_text", f"viz_{idx}"),
                )

            else:  # horizontal orientation
                # Horizontal bars - single row, arranged horizontally
                bar_height = viz_height  # Full height of viz area
                bar_width = viz_width / num_columns  # Divide width by number of columns
                bar_x = viz_x + (i * bar_width)
                bar_y = viz_y

                # Draw bar in single row
                self.canvas.create_rectangle(
                    bar_x,
                    bar_y,
                    bar_x + bar_width - 1,
                    bar_y + bar_height,
                    fill=color_hex,
                    outline="",
                    tags=("viz_bar", f"viz_{idx}"),
                )

                # Horizontal text
                if value is not None:
                    value_text = (
                        f"{value:.1f}" if isinstance(value, float) else str(value)[:4]
                    )
                else:
                    value_text = "-"

                # Compact text
                self.canvas.create_text(
                    bar_x + bar_width / 2,
                    bar_y + bar_height / 2,
                    text=f"{display_name[:2]}\n{value_text}",
                    fill=text_color,
                    font=("Arial", 6),
                    anchor="center",
                    tags=("viz_text", f"viz_{idx}"),
                )

    def _get_viz_color(self, column: str, value, config) -> str:
        """Get visualization color for a value using color map configuration"""
        try:
            # If we have a color map config for this column, use it
            if config and hasattr(config, "get_color"):
                # ColorMap object with get_color method
                color = config.get_color(value)
                # Convert to hex if needed
                return self._ensure_hex_color(color)
            elif config and hasattr(config, "ranges"):
                # Direct ColorMap usage
                numeric_value = float(value)
                for range_obj in config.ranges:
                    if range_obj.min_value <= numeric_value < range_obj.max_value:
                        # Handle both tuple and list color formats
                        if hasattr(range_obj, "color"):
                            if isinstance(range_obj.color, (list, tuple)):
                                r, g, b = range_obj.color
                                return f"#{r:02x}{g:02x}{b:02x}"
                            else:
                                # Color might already be a hex string
                                return self._ensure_hex_color(range_obj.color)

            # Fallback to simple gradient
            numeric_value = float(value)
            if numeric_value < 30:
                return "#e0e0e0"  # Light gray
            elif numeric_value < 50:
                return "#ff9800"  # Orange
            elif numeric_value < 60:
                return "#4caf50"  # Green
            else:
                return "#2196f3"  # Blue

        except (ValueError, TypeError):
            return self.theme_colors.get("secondary_bg", "#3e3e3e")

    def _ensure_hex_color(self, color) -> str:
        """Ensure color is in hex format for Tkinter"""
        if isinstance(color, str):
            # Already hex format
            if color.startswith("#"):
                return color
            # Space-separated RGB like "255 0 255"
            if " " in color:
                try:
                    parts = color.split()
                    r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
                    return f"#{r:02x}{g:02x}{b:02x}"
                except (ValueError, IndexError):
                    return "#808080"  # Default gray
            # Comma-separated RGB like "255,0,255"
            if "," in color:
                try:
                    parts = color.split(",")
                    r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
                    return f"#{r:02x}{g:02x}{b:02x}"
                except (ValueError, IndexError):
                    return "#808080"  # Default gray
        elif isinstance(color, (list, tuple)) and len(color) >= 3:
            # Convert RGB tuple/list to hex
            try:
                r, g, b = int(color[0]), int(color[1]), int(color[2])
                return f"#{r:02x}{g:02x}{b:02x}"
            except (ValueError, TypeError):
                return "#808080"  # Default gray
        # Default fallback
        return "#808080"

    def _unload_cell(self, idx: int):
        """Unload a single cell to free memory"""
        # Remove image reference
        if idx in self.image_refs:
            del self.image_refs[idx]

        # Clean up tooltip data
        if hasattr(self, "comment_tooltips") and idx in self.comment_tooltips:
            del self.comment_tooltips[idx]
        if hasattr(self, "peer_tooltips") and idx in self.peer_tooltips:
            del self.peer_tooltips[idx]

    # Public methods

    def load_images(
        self, images: List, preserve_selection: bool = False, force_reload: bool = False
    ):
        """Load images with lazy loading - creates placeholders, loads visible range"""
        if not images:
            return

        # Don't reload same images unless forced (for rotation/scale)
        if not force_reload:
            if self.displayed_images and len(images) == len(self.displayed_images):
                if all(a is b for a, b in zip(images, self.displayed_images)):
                    return

        # CRITICAL FIX: Clear queue and caches when force reloading
        if force_reload:
            # Clear the load queue completely
            while not self.load_queue.empty():
                try:
                    self.load_queue.get_nowait()
                except:
                    break

            # Clear all loading state
            self.loaded_cells.clear()
            self.pending_loads.clear()

            # Clear image references to force garbage collection
            for ref in self.image_refs.values():
                del ref
            self.image_refs.clear()

            # Delete all existing cell canvas items to ensure clean slate
            for cell_id_dict in self.cell_ids.values():
                if isinstance(cell_id_dict, dict):
                    for item_id in cell_id_dict.values():
                        if item_id:
                            try:
                                self.canvas.delete(item_id)
                            except:
                                pass
            self.cell_ids.clear()

            # Clear the canvas completely for force reload
            self.canvas.delete("all")

            # Clear cells dictionary too - CRITICAL
            self.cells.clear()

            # Reset visible range to force recalculation
            self.visible_range = (0, 0)

            self.logger.info(
                f"Force reload: cleared queue, caches, cells, and canvas items"
            )

        # Store images
        old_selection = list(self.selected_indices) if preserve_selection else []
        self.displayed_images = images
        self.selected_indices.clear()

        # Calculate optimal cell dimensions based on actual image aspect ratios
        if images:
            aspect_ratios = []

            # Sample first 20 images for performance
            for img in images[:20]:
                # Skip placeholders
                if hasattr(img, "is_placeholder") and img.is_placeholder:
                    continue

                # Load image to get dimensions
                if not hasattr(img, "image") or img.image is None:
                    img.load_image()

                if hasattr(img, "image") and img.image is not None:
                    h, w = img.image.shape[:2]

                    # Account for rotation
                    if self.rotation in [90, 270]:
                        w, h = h, w

                    aspect_ratios.append(w / h)

            if aspect_ratios:
                # Use median aspect ratio for consistent layout
                median_aspect = sorted(aspect_ratios)[len(aspect_ratios) // 2]

                # Calculate optimal cell dimensions
                canvas_width = self.canvas.winfo_width()
                if canvas_width > 100:
                    # Determine target columns based on aspect ratio
                    if median_aspect < 0.8:  # Portrait
                        target_cols = 5
                    elif median_aspect > 1.5:  # Landscape
                        target_cols = 3
                    else:  # Square-ish
                        target_cols = 4

                    # Calculate base dimensions
                    self.base_cell_width = (canvas_width - 10) // target_cols
                    self.base_cell_height = int(self.base_cell_width / median_aspect)

                    # Apply reasonable limits
                    max_height = int(canvas_width * 0.6)  # Max 60% of width for height
                    if self.base_cell_height > max_height:
                        self.base_cell_height = max_height
                        self.base_cell_width = int(
                            self.base_cell_height * median_aspect
                        )

                    # Apply scale factor
                    self.cell_width = int(self.base_cell_width * self.scale_factor)
                    self.cell_height = int(self.base_cell_height * self.scale_factor)

                    # Recalculate columns to fit
                    self.cols_per_row = max(1, (canvas_width - 10) // self.cell_width)

                    self.logger.debug(
                        f"Optimized cell dimensions: {self.cell_width}x{self.cell_height} "
                        f"(aspect: {median_aspect:.2f}, cols: {self.cols_per_row})"
                    )

        # Clear canvas
        self.canvas.delete("all")
        self.cells.clear()
        self.cell_ids.clear()
        self.image_refs.clear()
        self.loaded_cells.clear()
        self.pending_loads.clear()

        # Calculate grid layout
        viz_width = (
            len(self.viz_columns) * self.viz_column_width
            if self.show_data_visualizations
            else 0
        )
        total_cell_width = self.cell_width + viz_width

        # Create placeholder for each image
        for idx, img in enumerate(images):
            row = idx // self.cols_per_row
            col = idx % self.cols_per_row
            x = col * total_cell_width
            y = row * self.cell_height

            # Store cell info
            self.cells[(row, col)] = {"idx": idx, "x": x, "y": y}

            # Create placeholder rect
            placeholder_id = self.canvas.create_rectangle(
                x,
                y,
                x + self.cell_width,
                y + self.cell_height,
                fill=self.theme_colors["secondary_bg"],
                outline=self.theme_colors["field_border"],
                width=1,
                tags=(f"placeholder_{idx}", f"cell_{idx}"),
            )

            self.cell_ids[(row, col)] = {"placeholder": placeholder_id}

        # Update scroll region
        max_row = (len(images) - 1) // self.cols_per_row + 1
        scroll_height = max_row * self.cell_height
        self.canvas.configure(
            scrollregion=(0, 0, self.cols_per_row * total_cell_width, scroll_height)
        )

        # Load visible range
        self._update_lazy_load()

        # Restore selection
        if old_selection:
            self.selected_indices = set(old_selection)
            self._update_selection_visual()

        self.logger.info(f"Loaded {len(images)} images into grid")

    def _display_cell(self, idx: int, photo: ImageTk.PhotoImage, img):
        """Display loaded cell - replaces placeholder (runs on main thread)"""
        row = idx // self.cols_per_row
        col = idx % self.cols_per_row

        if (row, col) not in self.cells:
            return

        cell_data = self.cells[(row, col)]
        x = cell_data["x"]
        y = cell_data["y"]

        # Delete placeholder
        cell_ids = self.cell_ids.get((row, col), {})
        if cell_ids.get("placeholder"):
            self.canvas.delete(cell_ids["placeholder"])

        # Calculate viz dimensions
        viz_bar_width = 40
        viz_bar_height = 20
        viz_total_width = 0
        viz_total_height = 0

        if self.show_data_visualizations and self.viz_columns:
            if self.rotation in [0, 180]:
                viz_total_width = viz_bar_width
            else:
                viz_total_height = viz_bar_height

        # Calculate image allocation
        if self.rotation in [0, 180]:
            image_alloc_width = self.cell_width - viz_total_width
            image_alloc_height = self.cell_height
        else:
            image_alloc_width = self.cell_width
            image_alloc_height = self.cell_height - viz_total_height

        # Empty background
        bg_rect = self.canvas.create_rectangle(
            x,
            y,
            x + self.cell_width,
            y + self.cell_height,
            fill="",
            outline="",
            tags=(f"bg_{idx}", f"cell_{idx}"),
        )

        # Create image
        img_id = self.canvas.create_image(
            x,
            y,
            image=photo,
            anchor="nw",
            tags=(f"image_{idx}", f"cell_{idx}"),
        )

        # Get actual photo dimensions for viz alignment
        actual_img_width = photo.width()
        actual_img_height = photo.height()
        actual_img_x = x
        actual_img_y = y

        # Add data visualizations if enabled
        if self.show_data_visualizations and (
            img.csv_data or not (hasattr(img, "is_placeholder") and img.is_placeholder)
        ):
            if self.rotation in [0, 180]:
                viz_x = actual_img_x + actual_img_width
                viz_y = actual_img_y
                viz_width = viz_total_width
                viz_height = actual_img_height
                viz_orientation = "vertical"
            else:
                viz_x = actual_img_x
                viz_y = actual_img_y + actual_img_height
                viz_width = actual_img_width
                viz_height = viz_total_height
                viz_orientation = "horizontal"

            self._add_data_visualizations(
                viz_x, viz_y, viz_width, viz_height, img, viz_orientation, idx
            )

        # Create border
        border_color = self._get_classification_border_color(img)
        border_width = self._get_classification_border_width(img)
        dash_pattern = self._get_classification_dash_pattern(img)

        border_id = self.canvas.create_rectangle(
            x,
            y,
            x + self.cell_width,
            y + self.cell_height,
            outline=border_color,
            width=border_width,
            dash=dash_pattern if dash_pattern else (),
            tags=(f"border_{idx}", f"cell_{idx}"),
        )

        # Add label
        label_text = f"{img.hole_id}\n{int(img.depth_to)}m"
        if hasattr(img, "moisture_status") and img.moisture_status:
            label_text += f"\n{img.moisture_status}"

        label_id = self.canvas.create_text(
            x + 5,
            y + 5,
            text=label_text,
            anchor="nw",
            fill=self.theme_colors.get("text", "#ffffff"),
            font=("Arial", 9, "bold"),
            tags=(f"label_{idx}", f"cell_{idx}"),
        )

        # Update cell_ids
        self.cell_ids[(row, col)] = {
            "bg": bg_rect,
            "image": img_id,
            "border": border_id,
            "label": label_id,
        }

        self.loaded_cells.add(idx)
        self.pending_loads.discard(idx)

    def _prepare_thumbnail_old_style(self, img, target_width: int, target_height: int):
        """Prepare thumbnail using old code's proven approach with cv2"""
        # Check placeholder
        if hasattr(img, "is_placeholder") and img.is_placeholder:
            return None

        # Load image (returns OpenCV array in BGR format)
        if img.image is None:
            img.load_image()
            if img.image is None:
                return None

        # Get image and apply rotation with cv2
        image = img.image.copy()
        h, w = image.shape[:2]

        if self.rotation == 90:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            h, w = w, h
        elif self.rotation == 180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif self.rotation == 270:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            h, w = w, h

        # Calculate scale to fit within
        scale_x = target_width / w
        scale_y = target_height / h
        scale = min(scale_x, scale_y)

        new_width = int(w * scale)
        new_height = int(h * scale)

        # Resize
        resized = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_AREA
        )

        # Convert to PhotoImage
        pil_image = PILImage.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        return ImageTk.PhotoImage(pil_image)

    def _create_all_placeholders(self):
        """Create placeholder rectangles for all images (fast, no image loading)"""
        viz_width = (
            len(self.viz_columns) * self.viz_column_width
            if self.show_data_visualizations
            else 0
        )
        total_cell_width = self.cell_width + viz_width

        for idx, img in enumerate(self.displayed_images):
            row = idx // self.cols_per_row
            col = idx % self.cols_per_row

            x = col * total_cell_width
            y = row * self.cell_height

            # Store cell data
            self.cells[(row, col)] = {"idx": idx, "x": x, "y": y}

            # Create placeholder rectangle with FILLED background
            placeholder_id = self.canvas.create_rectangle(
                x,
                y,
                x + self.cell_width,
                y + self.cell_height,
                fill=self.theme_colors["secondary_bg"],
                outline=self.theme_colors["field_border"],
                width=1,
                tags=(f"placeholder_{idx}", f"cell_{idx}"),
            )

            # Add "Loading..." text
            text_id = self.canvas.create_text(
                x + self.cell_width // 2,
                y + self.cell_height // 2,
                text="Loading...",
                fill=self.theme_colors["subtext"],
                font=("Arial", 9),
                tags=(f"loading_text_{idx}", f"cell_{idx}"),
            )

            # Store IDs
            self.cell_ids[(row, col)] = {
                "placeholder": placeholder_id,
                "loading_text": text_id,
                "border": None,
                "image": None,
                "label": None,
            }

        # Update scroll region
        max_row = (len(self.displayed_images) - 1) // self.cols_per_row + 1
        scroll_height = max_row * self.cell_height
        self.canvas.configure(
            scrollregion=(0, 0, self.cols_per_row * total_cell_width, scroll_height)
        )

    def _create_cell_sync(self, idx: int, img):
        """Create and load a cell synchronously (like old code)"""
        row = idx // self.cols_per_row
        col = idx % self.cols_per_row

        viz_width = (
            len(self.viz_columns) * self.viz_column_width
            if self.show_data_visualizations
            else 0
        )
        total_cell_width = self.cell_width + viz_width

        x = col * total_cell_width
        y = row * self.cell_height

        # Store cell data
        self.cells[(row, col)] = {"idx": idx, "x": x, "y": y}

        # Create EMPTY background rectangle (no fill!)
        bg_rect = self.canvas.create_rectangle(
            x, y, x + self.cell_width, y + self.cell_height, fill="", outline=""
        )

        # Load image synchronously
        photo = self._prepare_thumbnail_sync(
            img, self.cell_width - 10, self.cell_height - 10
        )

        if photo:
            # Store reference
            self.image_refs[idx] = photo

            # Create image at top-left corner (like old code)
            img_id = self.canvas.create_image(
                x,
                y,
                image=photo,
                anchor="nw",
                tags=(f"image_{idx}", f"cell_{idx}"),
            )
        else:
            # Placeholder for missing image
            img_id = self.canvas.create_text(
                x + self.cell_width // 2,
                y + self.cell_height // 2,
                text="No Image",
                fill="white",
                tags=(f"text_{idx}", f"cell_{idx}"),
            )

        # Create border with classification styling
        border_color = self._get_classification_border_color(img)
        border_width = self._get_classification_border_width(img)
        dash_pattern = self._get_classification_dash_pattern(img)

        border_id = self.canvas.create_rectangle(
            x,
            y,
            x + self.cell_width,
            y + self.cell_height,
            outline=border_color,
            width=border_width,
            dash=dash_pattern if dash_pattern else (),
            tags=(f"border_{idx}", f"cell_{idx}"),
        )

        # Add label
        label_text = f"{img.hole_id}\n{int(img.depth_to)}m"
        if hasattr(img, "moisture_status") and img.moisture_status:
            label_text += f"\n{img.moisture_status}"

        label_id = self.canvas.create_text(
            x + 5,
            y + 5,
            text=label_text,
            anchor="nw",
            fill=self.theme_colors.get("text", "#ffffff"),
            font=("Arial", 9, "bold"),
            tags=(f"label_{idx}", f"cell_{idx}"),
        )

        # Store cell IDs
        self.cell_ids[(row, col)] = {
            "border": border_id,
            "image": img_id,
            "label": label_id,
        }

        self.loaded_cells.add(idx)

        # Draw visualization bars if enabled
        if self.show_data_visualizations:
            self._draw_visualization_bars(x, y, img, self.cell_width, self.cell_height)

    def _prepare_thumbnail_sync(self, img, target_width: int, target_height: int):
        """Prepare thumbnail synchronously (like old code)"""
        # Check if placeholder
        if hasattr(img, "is_placeholder") and img.is_placeholder:
            return None

        try:
            # Try to load PIL image
            pil_image = None

            if hasattr(img, "image_path") and img.image_path:
                image_path = Path(img.image_path)
                if image_path.exists():
                    pil_image = PILImage.open(image_path)

            if pil_image is None:
                return None

            # Apply rotation if set
            if self.rotation != 0:
                pil_image = pil_image.rotate(-self.rotation, expand=True)

            # Resize to fit (preserve aspect ratio)
            pil_image.thumbnail(
                (target_width, target_height), PILImage.Resampling.LANCZOS
            )

            # Convert to PhotoImage
            return ImageTk.PhotoImage(pil_image)

        except Exception as e:
            self.logger.error(f"Failed to load thumbnail: {e}")
            return None

    def _draw_visualization_bars(self, x, y, img, cell_width, cell_height):
        """Draw data visualization bars for an image"""
        if not self.show_data_visualizations or not self.viz_columns:
            return

        # Get CSV data for this image
        csv_data = getattr(img, "csv_data", {})
        if not csv_data:
            return

        # Determine orientation based on cell aspect ratio
        # Wide cells = horizontal bars, tall cells = vertical bars
        vertical_orientation = cell_height > cell_width * 1.2

        # Calculate viz area dimensions
        viz_width = self.viz_column_width
        viz_height = cell_height
        viz_x = x + cell_width  # Position to the right of image
        viz_y = y

        # Draw bars based on orientation
        num_columns = len(self.viz_columns)
        if num_columns == 0:
            return

        if vertical_orientation:
            # Vertical: Stack bars vertically
            bar_width = viz_width
            bar_height = viz_height / num_columns

            for i, column_name in enumerate(self.viz_columns):
                if column_name not in csv_data:
                    continue

                value = csv_data.get(column_name)
                if value is None or value == "":
                    continue

                # Get color from color map
                color = "#808080"  # Default gray
                if column_name in self.viz_column_configs and self.color_map_manager:
                    color_map = self.viz_column_configs[column_name]
                    color_rgb = self.color_map_manager.get_color_for_value(
                        value, color_map
                    )
                    if color_rgb:
                        color = self._rgb_to_hex(color_rgb)

                # Draw bar
                bar_y = viz_y + (i * bar_height)
                self.canvas.create_rectangle(
                    viz_x,
                    bar_y,
                    viz_x + bar_width,
                    bar_y + bar_height,
                    fill=color,
                    outline="",
                    tags="viz",
                )

                # Draw text (display name + value)
                display_name = self.viz_column_labels.get(column_name, column_name)
                value_text = (
                    f"{value:.1f}" if isinstance(value, (int, float)) else str(value)
                )

                # Determine text color based on background brightness
                text_color = self._get_contrast_text_color(color)

                # Draw text in center of bar
                text_x = viz_x + bar_width / 2
                text_y = bar_y + bar_height / 2

                self.canvas.create_text(
                    text_x,
                    text_y,
                    text=f"{display_name}\n{value_text}",
                    fill=text_color,
                    font=("Arial", 8),
                    anchor="center",
                    tags="viz",
                )

        else:
            # Horizontal: Bars side by side
            bar_width = viz_width / num_columns
            bar_height = viz_height

            for i, column_name in enumerate(self.viz_columns):
                if column_name not in csv_data:
                    continue

                value = csv_data.get(column_name)
                if value is None or value == "":
                    continue

                # Get color from color map
                color = "#808080"  # Default gray
                if column_name in self.viz_column_configs and self.color_map_manager:
                    color_map = self.viz_column_configs[column_name]
                    color_rgb = self.color_map_manager.get_color_for_value(
                        value, color_map
                    )
                    if color_rgb:
                        color = self._rgb_to_hex(color_rgb)

                # Draw bar
                bar_x = viz_x + (i * bar_width)
                self.canvas.create_rectangle(
                    bar_x,
                    viz_y,
                    bar_x + bar_width,
                    viz_y + bar_height,
                    fill=color,
                    outline="",
                    tags="viz",
                )

                # Draw compact text (first 2 chars + value)
                display_name = self.viz_column_labels.get(column_name, column_name)[:2]
                value_text = (
                    f"{value:.0f}"
                    if isinstance(value, (int, float))
                    else str(value)[:3]
                )

                # Determine text color
                text_color = self._get_contrast_text_color(color)

                # Draw text in center
                text_x = bar_x + bar_width / 2
                text_y = viz_y + bar_height / 2

                self.canvas.create_text(
                    text_x,
                    text_y,
                    text=f"{display_name}\n{value_text}",
                    fill=text_color,
                    font=("Arial", 7),
                    anchor="center",
                    tags="viz",
                )

    def _get_contrast_text_color(self, bg_color):
        """Get contrasting text color (black or white) based on background"""
        # Convert hex to RGB
        if bg_color.startswith("#"):
            r = int(bg_color[1:3], 16)
            g = int(bg_color[3:5], 16)
            b = int(bg_color[5:7], 16)
        else:
            return "black"  # Default

        # Calculate luminance
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255

        # Return white for dark backgrounds, black for light
        return "white" if luminance < 0.5 else "black"

    def _layout_grid(self):
        """Calculate grid layout and create cell placeholders"""
        print(f"DEBUG _layout_grid() called - {len(self.displayed_images)} images")

        if not self.displayed_images:
            print("DEBUG _layout_grid() - NO IMAGES, returning early")
            return

        viz_width = (
            len(self.viz_columns) * self.viz_column_width
            if self.show_data_visualizations
            else 0
        )

        total_cell_width = self.cell_width + viz_width

        for idx, img in enumerate(self.displayed_images):
            row = idx // self.cols_per_row
            col = idx % self.cols_per_row

            x = col * total_cell_width
            y = row * self.cell_height

            # Create cell placeholder - NO FILL so it doesn't cover images!
            cell_id = self.canvas.create_rectangle(
                x,
                y,
                x + total_cell_width,
                y + self.cell_height,
                outline=self.theme_colors["secondary_bg"],
                fill="",  # Empty fill so placeholder doesn't hide images
                width=1,
            )

            self.cells[(row, col)] = {"idx": idx, "x": x, "y": y}
            # Store as dict from the start for consistency
            self.cell_ids[(row, col)] = {
                "placeholder": cell_id,
                "border": None,
                "image": None,
                "label": None,
            }

        # Update scroll region
        max_row = (len(self.displayed_images) - 1) // self.cols_per_row + 1
        scroll_height = max_row * self.cell_height
        self.canvas.configure(
            scrollregion=(0, 0, self.cols_per_row * total_cell_width, scroll_height)
        )

    def clear(self):
        """Clear the grid"""
        # Stop loading thread
        self.stop_loading.set()

        # Clear queue
        while not self.load_queue.empty():
            try:
                self.load_queue.get_nowait()
            except:
                break

        # Clear data
        self.displayed_images.clear()
        self.selected_indices.clear()
        self.canvas.delete("all")
        self.cells.clear()
        self.cell_ids.clear()
        self.image_refs.clear()
        self.loaded_cells.clear()
        self.pending_loads.clear()

        # Restart loading thread for next load
        self.stop_loading.clear()
        if not self.loading_thread or not self.loading_thread.is_alive():
            self._start_loading_thread()

    def get_selected_indices(self) -> List[int]:
        """Get list of selected image indices"""
        return list(self.selected_indices)

    def select_index(self, idx: int):
        """Select a specific index"""
        if idx < len(self.displayed_images):
            self.selected_indices = {idx}
            self.last_selected_index = idx
            self._update_selection_visual()
            self._on_selection_change([idx])

    def clear_selection(self):
        """Clear all selection"""
        self.selected_indices.clear()
        self.last_selected_index = None
        self._update_selection_visual()
        self._on_selection_change([])

    def refresh_cell(self, idx: int):
        """Refresh a specific cell's visual state"""
        if idx in self.loaded_cells:
            self._load_cell(idx)

    def refresh_images(self, images: List):
        """
        Refresh visual state for a list of images.

        Args:
            images: List of CompartmentImage objects to refresh
        """
        # Find indices of these images in displayed_images
        for img in images:
            try:
                idx = self.displayed_images.index(img)
                self.refresh_cell(idx)
            except (ValueError, AttributeError):
                # Image not in current display, skip
                continue

    def set_zoom_preview(self, zoom_preview):
        """Set zoom preview window"""
        self.zoom_preview = zoom_preview

    def set_drag_classify_callback(
        self, callback: Optional[Callable[[List[int]], None]]
    ):
        """
        Set callback for drag-to-classify functionality.

        Args:
            callback: Function to call with list of selected indices when drag completes
        """
        self._on_drag_classify_callback = callback
        self.logger.debug("Drag-classify callback set")

    def toggle_data_visualizations(self):
        """Toggle display of data visualizations"""
        self.show_data_visualizations = not self.show_data_visualizations
        self.load_images(self.displayed_images, preserve_selection=True)

    def set_visualization_columns(self, columns, labels=None):
        """Set which columns to visualize with optional custom labels"""
        self.viz_columns = columns  # No limit on visualization columns
        if labels:
            self.viz_column_labels = labels
        self.load_images(self.displayed_images, preserve_selection=True)

    def configure_visualization_column(self, column_name, color_map_name):
        """Configure color map for a specific visualization column"""
        if self.color_map_manager and self.color_map_manager.has_preset(color_map_name):
            self.viz_column_configs[column_name] = self.color_map_manager.get_preset(
                color_map_name
            )
            self.logger.info(
                f"Set color map '{color_map_name}' for column '{column_name}'"
            )
            self.load_images(self.displayed_images, preserve_selection=True)

    def cleanup(self):
        """Clean up resources before destruction"""
        self.logger.debug("Cleaning up ReviewGridCanvas...")

        # Stop loading thread
        self.stop_loading.set()
        if self.loading_thread and self.loading_thread.is_alive():
            self.loading_thread.join(timeout=1.0)

        # Clear all data
        self.image_refs.clear()
        self.cells.clear()
        self.loaded_cells.clear()
        self.pending_loads.clear()

        self.logger.debug("ReviewGridCanvas cleanup complete")
