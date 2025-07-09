# Standard library
import logging
import threading
import traceback
import time
import re
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Dict, List, Optional, Tuple, Any
import tkinter as tk
from tkinter import ttk

# Third-party
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk

# Local application
from gui.dialog_helper import DialogHelper
from gui.widgets.entry_with_validation import create_entry_with_validation
from gui.gui_manager import GUIManager


logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

class DialogConstants:
    """All magic numbers and configuration constants."""
    COMPARTMENT_COUNT = 20
    ZOOM_WIDTH = 250
    ZOOM_HEIGHT = 350
    ZOOM_SCALE = 2
    CANVAS_UPDATE_DELAY_MS = 300
    COMPARTMENT_HEIGHT_CM = 4.5
    MIN_MARKER_SIZE = 20
    ADJUSTMENT_STEP_PX = 5
    DEBOUNCE_INTERVAL_S = 1.0
    MARKER_PREVIEW_SIZE = 40
    
    # Marker ID ranges
    CORNER_MARKER_IDS = [0, 1, 2, 3]
    COMPARTMENT_MARKER_IDS = list(range(4, 24))
    METADATA_MARKER_ID = 24
    
    # Zoom settings for adjustment mode
    STATIC_ZOOM_WIDTH = 200
    STATIC_ZOOM_HEIGHT = 200
    STATIC_ZOOM_REGION_SIZE = 100  # Size of region to extract


# ============================================================================
# Decorators
# ============================================================================

def ensure_main_thread(func):
    """Decorator to ensure function runs on main thread."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError(f"{func.__name__} must be called from main thread")
        return func(self, *args, **kwargs)
    return wrapper



# ============================================================================
# Data Models
# ============================================================================

@dataclass
class CompartmentMetadata:
    """Data model for compartment metadata."""
    hole_id: str
    depth_from: int
    depth_to: int
    compartment_interval: int


@dataclass
class BoundaryState:
    """State for boundary positions and adjustments."""
    top_y: int
    bottom_y: int
    left_height_offset: int = 0
    right_height_offset: int = 0
    

@dataclass
class VisualizationState:
    """State for visualization parameters."""
    scale_ratio: float = 1.0
    canvas_offset_x: int = 0
    canvas_offset_y: int = 0


@dataclass
class CompartmentBoundary:
    """Represents a single compartment boundary."""
    x1: int
    y1: int
    x2: int
    y2: int
    marker_id: int
    compartment_number: int
    center_x: int
    is_manual: bool = False
    is_interpolated: bool = False




class CompartmentRegistrationDialog:
    """Unified dialog for compartment registration workflow."""
    
    MODE_METADATA = 0
    MODE_MISSING_BOUNDARIES = 1
    MODE_ADJUST_BOUNDARIES = 2
    
    gui_manager: Optional[GUIManager]
    
    def __init__(self, parent: tk.Widget, image: np.ndarray, 
                 detected_boundaries: List[Tuple[int, int, int, int]],
                 missing_marker_ids: Optional[List[int]] = None,
                 **kwargs):
        """Initialize the dialog with clear parameter names."""
        self.parent = parent
        self.working_image = image.copy() if image is not None else None
        self.detected_boundaries = detected_boundaries.copy() if detected_boundaries else []
        self.missing_marker_ids = missing_marker_ids if missing_marker_ids else []
        
        # Extract optional parameters
        self.original_image = kwargs.get('original_image')
        self.theme_colors = kwargs.get('theme_colors', self._get_default_theme())
        self.gui_manager = kwargs.get('gui_manager')
        self.depth_validator = kwargs.get('depth_validator')
        self.file_manager = kwargs.get('file_manager')
        self.scale_data = kwargs.get('scale_data')
        self.image_path = kwargs.get('image_path')
        
        # Configuration
        self.config = kwargs.get('config') or {}
        self.markers = kwargs.get('markers') or {}
        self.corner_markers = kwargs.get('corner_markers') or {}
        self.marker_to_compartment = kwargs.get('marker_to_compartment', {})
        self.rotation_angle = kwargs.get('rotation_angle', 0.0)
        
        # Store last successful metadata if provided
        metadata = kwargs.get('metadata', {})
        self.last_successful_metadata = metadata.get('last_successful_metadata')
        
        # Callbacks
        self.on_apply_adjustments = kwargs.get('on_apply_adjustments')
        
        # Initialize logger
        self.logger = logger
        
        # Initialize state
        self.current_mode = self.MODE_METADATA
        self.boundary_state = self._calculate_initial_boundaries()
        self.mouse_hovering = False
        self.temp_point = None
        
        # Get average compartment width
        boundary_analysis = kwargs.get('boundary_analysis', {})
        self.avg_compartment_width = boundary_analysis.get(
            'avg_compartment_width', 
            self._calculate_avg_width()
        )
        
        # Create the dialog and components
        self.dialog = self._create_dialog()
        self._create_components()
        self._create_ui()
        
        # Initialize with metadata if provided
        if metadata:
            self.metadata_panel.set_metadata(metadata)
            
        # Initialize visualization manager
        self.canvas_viz_manager = DialogCanvasRenderer(self.working_image, self.theme_colors, self.gui_manager)
        self.canvas_viz_manager.set_canvas(self.canvas)

        self.show_wall_detection = False
        # Bind keyboard event
        self.dialog.bind("<KeyPress>", self._on_key_press)
        
    def _get_default_theme(self) -> Dict[str, str]:
        """Get default theme colors."""
        return {
            "background": "#1e1e1e",
            "text": "#e0e0e0",
            "field_bg": "#2d2d2d",
            "field_border": "#3f3f3f",
            "accent_green": "#4CAF50",
            "accent_blue": "#2196F3",
            "accent_red": "#F44336",
            "accent_yellow": "#FFEB3B",
            "hover_highlight": "#3a3a3a",
            "success_bg": "#ccffcc",
            "error_bg": "#ffcccc"
        }
        
    def _calculate_initial_boundaries(self) -> BoundaryState:
        """Calculate initial boundary positions from corner markers or detected boundaries."""
        # Try to get from corner markers first
        top_y_coords = []
        bottom_y_coords = []
        
        for marker_id in [0, 1]:  # Top corners
            if marker_id in self.markers:
                corners = self.markers[marker_id]
                top_y_coords.append(np.min(corners[:, 1]))  # Use top edge
            elif marker_id in self.corner_markers:
                corners = self.corner_markers[marker_id]
                top_y_coords.append(np.min(corners[:, 1]))
                
        for marker_id in [2, 3]:  # Bottom corners
            if marker_id in self.markers:
                corners = self.markers[marker_id]
                bottom_y_coords.append(np.max(corners[:, 1]))  # Use bottom edge
            elif marker_id in self.corner_markers:
                corners = self.corner_markers[marker_id]
                bottom_y_coords.append(np.max(corners[:, 1]))
                
        if top_y_coords and bottom_y_coords:
            top_y = int(np.mean(top_y_coords))
            bottom_y = int(np.mean(bottom_y_coords))
        elif self.detected_boundaries:
            # Fall back to detected boundaries
            tops = [y1 for _, y1, _, _ in self.detected_boundaries]
            bottoms = [y2 for _, _, _, y2 in self.detected_boundaries]
            top_y = int(np.mean(tops)) if tops else 100
            bottom_y = int(np.mean(bottoms)) if bottoms else 700
        else:
            # Default values
            h = self.working_image.shape[0] if self.working_image is not None else 800
            top_y = int(h * 0.1)
            bottom_y = int(h * 0.9)
            
        return BoundaryState(top_y=top_y, bottom_y=bottom_y)
        
    def _calculate_avg_width(self) -> int:
        """Calculate average compartment width from detected boundaries."""
        if not self.detected_boundaries:
            return 50  # Default
            
        widths = [x2 - x1 for x1, _, x2, _ in self.detected_boundaries]
        return int(np.mean(widths)) if widths else 50
        
    def _create_dialog(self) -> tk.Toplevel:
        """Create the dialog window."""
        dialog = DialogHelper.create_dialog(
            parent=self.parent,
            title=DialogHelper.t("Compartment Registration"),
            modal=False,
            topmost=False
        )
        
        dialog.configure(bg=self.theme_colors["background"])
        dialog.protocol("WM_DELETE_WINDOW", self._on_cancel)
        
        return dialog
        

    def _on_key_press(self, event: tk.Event) -> None:
        """Handle key press events."""
        if event.char == "`":  # Backtick key
            # Toggle wall detection visualization
            self.show_wall_detection = not self.show_wall_detection
            self.status_var.set(
                f"Wall detection visualization: {'ON' if self.show_wall_detection else 'OFF'}"
            )
            # Force cache invalidation and update
            self.canvas_viz_manager.invalidate_cache()
            self._update_visualization()

    def _create_components(self):
        """Create specialized components."""
        # Create metadata panel with last successful metadata
        self.metadata_panel = MetadataPanel(
            self.dialog,
            self.theme_colors,
            depth_validator=self.depth_validator,
            on_metadata_change=self._on_metadata_change,
            config=self.config,
            last_successful_metadata=self.last_successful_metadata
        )
        
        # Components that need canvas will be created after UI
        self.boundary_annotator = None
        self.boundary_adjuster = None
        self.zoom_lens = ZoomLens(self.dialog, self.theme_colors, self.gui_manager)
        
    def _create_ui(self):
        """Create the main UI layout."""
        # Configure ttk styles
        if self.gui_manager:
            self.gui_manager.configure_ttk_styles(self.dialog)
            
        # Main container
        main_frame = ttk.Frame(self.dialog, padding=10, style='Content.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Mode selector buttons
        self._create_mode_selector(main_frame)
        
        # Canvas for image display
        canvas_frame = ttk.Frame(main_frame, style='Content.TFrame')
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Calculate initial canvas size
        screen_width = self.dialog.winfo_screenwidth()
        screen_height = self.dialog.winfo_screenheight()
        canvas_width = int(screen_width * 0.8)
        canvas_height = int(screen_height * 0.6)
        
        self.canvas = tk.Canvas(
            canvas_frame,
            bg=self.theme_colors["background"],
            highlightthickness=0,
            width=canvas_width,
            height=canvas_height
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Create components that need canvas
        self.boundary_annotator = BoundaryAnnotator(
            self.canvas,
            self.working_image,
            self.missing_marker_ids,
            self.marker_to_compartment,
            self.boundary_state,
            self.avg_compartment_width,
            self.scale_data,
            self.config,
            self.detected_boundaries,
            on_annotation_complete=self._on_annotation_complete,
            gui_manager=self.gui_manager
        )
        
        self.boundary_adjuster = BoundaryAdjuster(
            self.canvas,
            self.boundary_state,
            on_adjustment=self._on_boundary_adjustment
        )
        
        # Status display
        self.status_frame = ttk.Frame(main_frame, style='Content.TFrame')
        self.status_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.status_var = tk.StringVar()
        self.status_label = ttk.Label(
            self.status_frame,
            textvariable=self.status_var,
            style='Instructions.TLabel'
        )
        self.status_label.pack()
        
        # Bottom container for mode-specific UI
        self.bottom_container = ttk.Frame(main_frame, style='Content.TFrame')
        self.bottom_container.pack(fill=tk.X, pady=(5, 0))
        
        # Create mode-specific UI elements
        self.metadata_ui = self.metadata_panel.create_ui(self.bottom_container, self.gui_manager)
        self.adjustment_ui = self.boundary_adjuster.create_controls(
            self.bottom_container, self.theme_colors, self.gui_manager
        )
        
        # Buttons
        self._create_buttons(main_frame)
        
        # Bind canvas events
        self._bind_canvas_events()
        
        # Initial update
        self.dialog.after_idle(self._initial_update)
        
    def _create_mode_selector(self, parent: tk.Widget):
        """Create mode selection buttons."""
        mode_frame = ttk.Frame(parent, style='Content.TFrame')
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        btn_container = ttk.Frame(mode_frame, style='Content.TFrame')
        btn_container.pack(anchor=tk.CENTER)
        
        self.mode_buttons = {}
        
        # Create mode buttons
        modes = [
            (self.MODE_METADATA, "Metadata Registration"),
            (self.MODE_MISSING_BOUNDARIES, "Add Missing Boundaries"),
            (self.MODE_ADJUST_BOUNDARIES, "Adjust Boundaries")
        ]
        
        for col, (mode, text) in enumerate(modes):
            if self.gui_manager:
                btn = self.gui_manager.create_modern_button(
                    btn_container,
                    text=DialogHelper.t(text),
                    color=self.theme_colors["accent_green"] if mode == self.current_mode 
                          else self.theme_colors["field_bg"],
                    command=lambda m=mode: self._switch_mode(m)
                )
            else:
                btn = tk.Button(
                    btn_container,
                    text=DialogHelper.t(text),
                    command=lambda m=mode: self._switch_mode(m)
                )
            btn.grid(row=0, column=col, padx=5)
            self.mode_buttons[mode] = btn
            
    def _create_buttons(self, parent: tk.Widget):
        """Create action buttons."""
        button_frame = ttk.Frame(parent, style='Content.TFrame')
        button_frame.pack(fill=tk.X, pady=(10, 5))
        
        button_container = ttk.Frame(button_frame, style='Content.TFrame')
        button_container.pack(expand=True)
        
        # Button definitions
        buttons = [
            ("quit", DialogHelper.t("Quit"), self.theme_colors["accent_red"], self._on_quit),
            ("undo_last", DialogHelper.t("Undo Last"), self.theme_colors["accent_blue"], self._undo_last),
            ("continue", DialogHelper.t("Continue"), self.theme_colors["accent_green"], self._on_continue),
            ("cancel", DialogHelper.t("Cancel"), self.theme_colors["accent_blue"], self._on_cancel),
            ("reject", DialogHelper.t("Reject"), self.theme_colors["accent_red"], self._on_reject)
        ]
        
        self.buttons = {}
        for i, (key, text, color, command) in enumerate(buttons):
            if self.gui_manager:
                btn = self.gui_manager.create_modern_button(
                    button_container,
                    text=text,
                    color=color,
                    command=command
                )
            else:
                btn = tk.Button(
                    button_container,
                    text=text,
                    command=command
                )
            btn.pack(side=tk.LEFT, padx=5)
            self.buttons[key] = btn
            
    def _bind_canvas_events(self):
        """Bind canvas mouse events."""
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Motion>", self._on_canvas_move)
        self.canvas.bind("<Leave>", self._on_canvas_leave)
        self.canvas.bind("<Enter>", self._on_canvas_enter)
        self.canvas.bind("<Button-3>", self._on_canvas_right_click)
        self.canvas.bind("<B3-Motion>", self._on_canvas_move)
        
    def _initial_update(self):
        """Initial update after UI is created."""
        self.canvas.update_idletasks()
        self._update_visualization()
        self._update_mode_display()
        self._update_status_message()
        
    def _switch_mode(self, new_mode: int):
        """Switch to a different mode."""
        if new_mode == self.current_mode:
            return
            
        # Validate before leaving metadata mode
        if self.current_mode == self.MODE_METADATA:
            is_valid, error_msg = self.metadata_panel.validate()
            if not is_valid:
                DialogHelper.show_message(
                    self.dialog,
                    DialogHelper.t("Validation Error"),
                    error_msg,
                    message_type="error"
                )
                return
                
        # Update boundary annotator when entering missing boundaries mode
        if new_mode == self.MODE_MISSING_BOUNDARIES:
            # Update marker to compartment mapping based on metadata
            self._update_marker_to_compartment_mapping()
            self.boundary_annotator.marker_to_compartment = self.marker_to_compartment
            
        # Setup boundary adjuster when entering adjustment mode
        if new_mode == self.MODE_ADJUST_BOUNDARIES:
            # Set all compartments for adjustment
            self.boundary_adjuster.set_compartments(
                self.detected_boundaries,
                self.boundary_annotator.result_boundaries
            )
            # Show static zoom windows
            self._update_static_zoom_views()
                
        self.current_mode = new_mode
        self._update_mode_display()
        self._update_visualization(force_full_update=True)
        self._update_status_message()
        
    def _update_mode_display(self):
        """Update UI elements for current mode."""
        # Update button styles
        for mode, btn in self.mode_buttons.items():
            if hasattr(btn, 'configure'):
                if mode == self.current_mode:
                    btn.configure(background=self.theme_colors["accent_green"])
                    if hasattr(btn, 'base_color'):
                        btn.base_color = self.theme_colors["accent_green"]
                        if hasattr(btn, '_reset_color'):
                            btn._reset_color()
                else:
                    btn.configure(background=self.theme_colors["field_bg"])
                    if hasattr(btn, 'base_color'):
                        btn.base_color = self.theme_colors["field_bg"]
                        if hasattr(btn, '_reset_color'):
                            btn._reset_color()
                        
        # Show/hide mode-specific UI
        if self.current_mode == self.MODE_METADATA:
            self.metadata_ui.pack(expand=True)
            self.adjustment_ui.pack_forget()
            self.zoom_lens.hide_static_zooms()
        elif self.current_mode == self.MODE_MISSING_BOUNDARIES:
            self.metadata_ui.pack_forget()
            self.adjustment_ui.pack_forget()
            self.zoom_lens.hide_static_zooms()
        elif self.current_mode == self.MODE_ADJUST_BOUNDARIES:
            self.metadata_ui.pack_forget()
            self.adjustment_ui.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
        # Update continue button text
        continue_btn = self.buttons.get('continue')
        if continue_btn:
            if self.current_mode == self.MODE_METADATA:
                text = DialogHelper.t("Continue to Boundaries")
            elif self.current_mode == self.MODE_MISSING_BOUNDARIES:
                if self.boundary_annotator.annotation_complete:
                    text = DialogHelper.t("Continue to Adjustment")
                else:
                    text = DialogHelper.t("Place Markers")
            else:
                text = DialogHelper.t("Finish")
            
            if hasattr(continue_btn, 'set_text'):
                continue_btn.set_text(text)
            elif hasattr(continue_btn, 'config'):
                continue_btn.config(text=text)
                
    def _update_status_message(self):
        """Update status message for current mode."""
        if self.current_mode == self.MODE_METADATA:
            self.status_var.set(DialogHelper.t(
                "Enter hole ID and depth information, then click Continue to Boundaries"
            ))
        elif self.current_mode == self.MODE_MISSING_BOUNDARIES:
            self.status_var.set(self.boundary_annotator.get_status_message())
        elif self.current_mode == self.MODE_ADJUST_BOUNDARIES:
            self.status_var.set(DialogHelper.t(
                "Adjust top and bottom boundaries or side heights, then click 'Finish'"
            ))
                
    def _update_visualization(self, fast_mode: bool = False, force_full_update: bool = False) -> None:
        """Update the canvas visualization using static/dynamic layers.
        
        Args:
            fast_mode: If True, skip static layer update for performance (e.g., during mouse motion)
            force_full_update: If True, force recreation of static elements even if cached
        """
        if self.working_image is None:
            return
        
        # Force cache invalidation if requested
        if force_full_update:
            self.canvas_viz_manager.invalidate_cache()
        
        # Only update static layer if not in fast mode
        if not fast_mode:
            # Create static visualization (wall detection flag no longer needed here)
            static_viz = self.canvas_viz_manager.create_static_visualization(
                self.markers,
                self.detected_boundaries,
                self.scale_data,
                show_scale_bar=True,
                mouse_hovering=self.mouse_hovering,
                boundary_analysis=getattr(self, 'boundary_analysis', None)
            )
            
            # Display static layer
            self.canvas_viz_manager.display_image_on_canvas(static_viz, self.canvas)
        
        # Always draw dynamic overlays (these are lightweight)
        # Pass wall detection flag to dynamic overlays
        self.canvas_viz_manager.draw_dynamic_overlays(
            self.canvas,
            self.current_mode,
            self.boundary_annotator,
            self.boundary_adjuster,
            self.boundary_state,
            self.temp_point,
            show_wall_detection=getattr(self, 'show_wall_detection', False)
        )
        
    def _update_static_zoom_views(self):
        """Update static zoom windows for adjustment mode."""
        if self.current_mode == self.MODE_ADJUST_BOUNDARIES:
            self.zoom_lens.update_static_zooms(
                self.working_image,
                self.boundary_state,
                self.corner_markers,
                self.markers,
                self.detected_boundaries,
                self.canvas
            )
        else:
            self.zoom_lens.hide_static_zooms()
            
    def _on_canvas_click(self, event):
        """Handle canvas click."""
        # Convert to image coordinates
        img_x, img_y = self.canvas_viz_manager.canvas_to_image_coords(event.x, event.y)
        
        # Check bounds
        if self.working_image is not None:
            h, w = self.working_image.shape[:2]
            if not (0 <= img_x < w and 0 <= img_y < h):
                return
        
        if self.current_mode == self.MODE_MISSING_BOUNDARIES:
            if self.boundary_annotator.needs_vertical_boundary_placement():
                # Place vertical boundaries
                if self.boundary_annotator.place_vertical_boundaries(img_y):
                    self.canvas_viz_manager.invalidate_cache()  # Force cache update
                    self._update_visualization()
                    self._update_status_message()
            else:
                # Handle compartment placement or removal
                result = self.boundary_annotator.handle_click(img_x, img_y)
                if result:
                    if 'message' in result:
                        self.status_var.set(result['message'])
                    self.canvas_viz_manager.invalidate_cache()
                    self._update_visualization(force_full_update=True)
                    self._update_status_message()
                    
        elif self.current_mode == self.MODE_ADJUST_BOUNDARIES:
            # Handle compartment selection
            if self.boundary_adjuster.handle_compartment_click(
                img_x, img_y, self.working_image.shape
            ):
                self._update_visualization()
                
    def _on_canvas_move(self, event):
        """Handle mouse movement on canvas."""
        # Convert to image coordinates
        img_x, img_y = self.canvas_viz_manager.canvas_to_image_coords(event.x, event.y)
        
        # Check bounds
        if self.working_image is not None:
            h, w = self.working_image.shape[:2]
            if 0 <= img_x < w and 0 <= img_y < h:
                self.temp_point = (img_x, img_y)
                
                if self.current_mode == self.MODE_MISSING_BOUNDARIES:
                    # Use fast mode for mouse movement
                    self._update_visualization(fast_mode=True)
                    
                    # Draw boundary preview (this is dynamic)
                    self.boundary_annotator.draw_preview(
                        event.x, event.y, self.canvas_viz_manager.viz_state
                    )
                elif self.current_mode == self.MODE_METADATA:
                    # Show hover zoom
                    if event.state & 0x400:  # Right button pressed
                        self._show_hover_zoom(event, flipped=True)
                    else:
                        self._show_hover_zoom(event, flipped=False)
            else:
                self.temp_point = None
                self.canvas.delete("boundary_preview")
                self.zoom_lens.hide_hover_zoom()
                
    def _on_canvas_leave(self, event):
        """Handle mouse leaving canvas."""
        self.temp_point = None
        self.mouse_hovering = False
        self.canvas.delete("boundary_preview")
        self.zoom_lens.hide_hover_zoom()
        
        # Update visualization if scale bar visibility changes
        if self.scale_data:
            self._update_visualization()
            
    def _on_canvas_enter(self, event):
        """Handle mouse entering canvas."""
        self.mouse_hovering = True
        # Update visualization if scale bar visibility changes
        if self.scale_data:
            self._update_visualization()
            
    def _on_canvas_right_click(self, event):
        """Handle right click for flipped zoom."""
        self._show_hover_zoom(event, flipped=True)
        
    def _show_hover_zoom(self, event, flipped=False):
        """Show hover zoom at mouse position."""
        if not self.temp_point:
            return
            
        img_x, img_y = self.temp_point
        
        # Get zoom region
        if self.current_mode == self.MODE_MISSING_BOUNDARIES:
            # Special handling for boundary placement
            region = self._get_boundary_preview_region(img_x, img_y)
        else:
            # Normal zoom region
            size = DialogConstants.ZOOM_WIDTH // DialogConstants.ZOOM_SCALE
            region = self.canvas_viz_manager.get_zoom_region(img_x, img_y, size)
            
        if region is not None:
            # Calculate screen position
            if flipped:
                screen_x = event.x_root - DialogConstants.ZOOM_WIDTH - 10
            else:
                screen_x = event.x_root + 10
            screen_y = event.y_root + 10
            
            self.zoom_lens.show_hover_zoom(screen_x, screen_y, region, flipped)
            
    def _get_boundary_preview_region(self, center_x: int, center_y: int) -> Optional[np.ndarray]:
        """Get zoom region for boundary placement preview."""
        if self.working_image is None:
            return None
            
        h, w = self.working_image.shape[:2]
        
        # For compartment placement, show wider horizontal view
        if (self.boundary_annotator.current_index < len(self.boundary_annotator.missing_marker_ids) and
            self.boundary_annotator.missing_marker_ids[self.boundary_annotator.current_index] in 
            DialogConstants.COMPARTMENT_MARKER_IDS):
            
            # Show compartment area
            zoom_w = DialogConstants.ZOOM_WIDTH // 2
            zoom_h = DialogConstants.ZOOM_HEIGHT // 2
            
            x1 = max(0, center_x - zoom_w)
            x2 = min(w, center_x + zoom_w)
            
            # Show full compartment height
            y1 = max(0, self.boundary_state.top_y - 20)
            y2 = min(h, self.boundary_state.bottom_y + 20)
            
        else:
            # Normal zoom for other cases
            size = DialogConstants.ZOOM_WIDTH // DialogConstants.ZOOM_SCALE
            x1 = max(0, center_x - size // 2)
            y1 = max(0, center_y - size // 2)
            x2 = min(w, x1 + size)
            y2 = min(h, y1 + size)
            
        if x2 <= x1 or y2 <= y1:
            return None
            
        return self.working_image[y1:y2, x1:x2].copy()
        
    def _on_metadata_change(self):
        """Handle metadata change to update compartment labels."""
        self._update_marker_to_compartment_mapping()
        
    def _update_marker_to_compartment_mapping(self):
        """Update marker to compartment mapping based on metadata."""
        metadata = self.metadata_panel.get_metadata()
        
        if metadata.depth_from and metadata.depth_to and metadata.compartment_interval:
            # Recalculate compartment depths
            new_mapping = {}
            for i in range(DialogConstants.COMPARTMENT_COUNT):
                marker_id = 4 + i
                depth = metadata.depth_from + ((i + 1) * metadata.compartment_interval)
                if depth <= metadata.depth_to:
                    new_mapping[marker_id] = depth
                    
            self.marker_to_compartment = new_mapping
            
            # Update annotator if it exists
            if self.boundary_annotator:
                self.boundary_annotator.marker_to_compartment = new_mapping
                
            # Invalidate cache and update visualization
            self.canvas_viz_manager.invalidate_cache()
            self._update_visualization()
            
    def _on_boundary_adjustment(self, adjustment_data: Dict):
        """Handle boundary adjustment from BoundaryAdjuster."""
        # Update visualization
        self._update_visualization()
        
        # Update static zoom views
        self._update_static_zoom_views()
        
        # Call external callback if provided
        if self.on_apply_adjustments:
            self.on_apply_adjustments(adjustment_data)
            
    def _on_annotation_complete(self):
        """Handle completion of annotation phase 1."""
        # Enable continue button
        continue_btn = self.buttons.get('continue')
        if continue_btn and hasattr(continue_btn, 'configure'):
            continue_btn.configure(state='normal')
            if hasattr(continue_btn, 'set_text'):
                continue_btn.set_text(DialogHelper.t("Continue to Adjustment"))
                
        # Update status
        self._update_status_message()
        
    def _undo_last(self):
        """Undo last action."""
        if self.current_mode == self.MODE_MISSING_BOUNDARIES:
            if self.boundary_annotator.undo_last():
                self.canvas_viz_manager.invalidate_cache()
                self._update_visualization(force_full_update=True)
                self._update_status_message()
        elif self.current_mode == self.MODE_ADJUST_BOUNDARIES:
            # Reset adjustments
            self.boundary_adjuster.reset_adjustments()
            self._update_visualization(force_full_update=True)
            self._update_static_zoom_views()
            self.status_var.set(DialogHelper.t("Reset all adjustments"))
            
    def _on_continue(self):
        """Handle continue button."""
        if self.current_mode == self.MODE_METADATA:
            # Validate and move to next mode
            is_valid, error_msg = self.metadata_panel.validate()
            if is_valid:
                if self.missing_marker_ids:
                    self._switch_mode(self.MODE_MISSING_BOUNDARIES)
                else:
                    self._switch_mode(self.MODE_ADJUST_BOUNDARIES)
            else:
                DialogHelper.show_message(
                    self.dialog, 
                    DialogHelper.t("Validation Error"), 
                    error_msg, 
                    message_type="error"
                )
                
        elif self.current_mode == self.MODE_MISSING_BOUNDARIES:
            # Check if all markers are placed
            if not self.missing_marker_ids or self.boundary_annotator.annotation_complete:
                self._switch_mode(self.MODE_ADJUST_BOUNDARIES)
            else:
                # Ask for confirmation
                missing_count = len(self.boundary_annotator.missing_marker_ids) - self.boundary_annotator.current_index
                if missing_count > 0:
                    if DialogHelper.confirm_dialog(
                        self.dialog,
                        DialogHelper.t("Incomplete Annotations"),
                        DialogHelper.t(
                            "You have %(count)s compartments left to annotate. "
                            "Do you want to proceed to boundary adjustment without placing all markers?",
                            count=missing_count
                        ),
                        yes_text=DialogHelper.t("Proceed"),
                        no_text=DialogHelper.t("Stay Here")
                    ):
                        self._switch_mode(self.MODE_ADJUST_BOUNDARIES)
                else:
                    self._switch_mode(self.MODE_ADJUST_BOUNDARIES)
            
        elif self.current_mode == self.MODE_ADJUST_BOUNDARIES:
            # Finish and close
            self._compile_and_close()
            
    def _on_quit(self):
        """Handle quit button."""
        if DialogHelper.confirm_dialog(
            self.dialog,
            DialogHelper.t("Stop Processing"),
            DialogHelper.t("Are you sure you want to stop processing?\n\n"
                         "No modifications will be made to the current image, "
                         "and processing of remaining images will be canceled."),
            yes_text=DialogHelper.t("Stop Processing"),
            no_text=DialogHelper.t("Continue")
        ):
            self._quit_flag = True
            self._cleanup_and_close()
            
    def _on_cancel(self):
        """Handle cancel button."""
        if DialogHelper.confirm_dialog(
            self.dialog,
            DialogHelper.t("Cancel Registration"),
            DialogHelper.t("Are you sure you want to cancel? All manual annotations will be lost."),
            yes_text=DialogHelper.t("Yes"),
            no_text=DialogHelper.t("No")
        ):
            self._cleanup_and_close()
            
    def _on_reject(self):
        """Handle reject button."""
        # Use DialogHelper's rejection handler
        result = DialogHelper.handle_rejection(
            self.dialog,
            self.image_path,
            metadata_callback=lambda: self.metadata_panel.get_metadata().__dict__,
            cleanup_callback=lambda: self.zoom_lens.cleanup()
        )
        
        if result:
            self._rejected_flag = True
            self._rejection_data = result
            self._cleanup_and_close()
            
    def _compile_and_close(self):
        """Compile results and close dialog."""
        # Store final results
        self.final_results = self._compile_results()
        
        # Close dialog
        self._cleanup_and_close()
        
    def _cleanup_and_close(self):
        """Clean up resources and close dialog."""
        # Cleanup zoom lens
        self.zoom_lens.cleanup()
        
        # Close dialog
        self.dialog.destroy()
        
    def _compile_results(self) -> Dict:
        """Compile all results."""
        metadata = self.metadata_panel.get_metadata()
        
        # Get all adjusted boundaries
        adjusted_boundaries = []
        if self.boundary_adjuster.all_compartments:
            adjusted_boundaries = self.boundary_adjuster.get_adjusted_boundaries()
        else:
            # Use original boundaries
            adjusted_boundaries = self.detected_boundaries.copy()
            
        # Add manually placed boundaries
        for boundary in self.boundary_annotator.result_boundaries:
            adjusted_boundaries.append((
                boundary['x1'], boundary['y1'], 
                boundary['x2'], boundary['y2']
            ))
        
        return {
            'hole_id': metadata.hole_id,
            'depth_from': metadata.depth_from,
            'depth_to': metadata.depth_to,
            'compartment_interval': metadata.compartment_interval,
            'result_boundaries': self.boundary_annotator.placed_positions,
            'detected_boundaries': adjusted_boundaries,
            'top_boundary': self.boundary_state.top_y,
            'bottom_boundary': self.boundary_state.bottom_y,
            'left_height_offset': self.boundary_state.left_height_offset,
            'right_height_offset': self.boundary_state.right_height_offset,
            'rotation_angle': self.rotation_angle,
            'avg_width': self.avg_compartment_width,
            'final_visualization': self.canvas_viz_manager.static_cache,
            'quit': getattr(self, '_quit_flag', False),
            'rejected': getattr(self, '_rejected_flag', False),
            'rejection_data': getattr(self, '_rejection_data', None)
        }
        
    @ensure_main_thread
    def show(self) -> Dict:
        """Show dialog and return results."""
        try:
            # Position dialog
            screen_width = self.dialog.winfo_screenwidth()
            screen_height = self.dialog.winfo_screenheight()
            
            # Landscape orientation
            width = int(screen_width * 0.95)
            height = int(screen_height * 0.9)
            
            x = (screen_width - width) // 2
            y = (screen_height - height) // 2
            
            self.dialog.geometry(f"{width}x{height}+{x}+{y}")
            
            if self.parent:
                self.dialog.transient(self.parent)
                
            # Show dialog
            self.dialog.deiconify()
            self.dialog.lift()
            self.dialog.focus_force()
            
            # Wait for completion
            self.dialog.wait_window()
            
            # Return results
            return getattr(self, 'final_results', self._compile_results())
            
        except Exception as e:
            self.logger.error(f"Error showing dialog: {e}")
            self.logger.error(traceback.format_exc())
            return {'error': str(e)}



# MetadataPanel Component

class MetadataPanel:
    """Handles metadata entry UI and validation."""
    
    def __init__(self, parent: tk.Widget, theme_colors: Dict[str, str], 
                 depth_validator: Optional[object] = None,
                 on_metadata_change: Optional[Callable] = None,
                 config: Optional[Dict] = None,
                 last_successful_metadata: Optional[Dict] = None):
        self.parent = parent
        self.theme_colors = theme_colors
        self.depth_validator = depth_validator
        self.on_metadata_change = on_metadata_change
        self.config = config or {}
        self.last_successful_metadata = last_successful_metadata
        self.logger = logger
        
        # Create UI variables
        self.hole_id = tk.StringVar()
        self.depth_from = tk.StringVar()
        self.depth_to = tk.StringVar()
        self.interval_var = tk.IntVar(value=self.config.get('compartment_interval', 1))
        
        # Manual override tracking
        self.depth_to_manual_override = False
        
        # Create the UI
        self.frame = None
        self.increment_button = None
        
    def create_ui(self, parent: tk.Widget, gui_manager: GUIManager) -> tk.Widget:
        """Create and return the metadata entry UI panel."""
        self.gui_manager = gui_manager
        self.frame = ttk.Frame(parent, style='Content.TFrame')
        
        # Create bordered container
        border_frame = ttk.Frame(self.frame, style='Content.TFrame', 
                               relief=tk.RIDGE, borderwidth=2)
        border_frame.pack(expand=True, pady=10, padx=10)
        
        container = ttk.Frame(border_frame, style='Content.TFrame', padding=15)
        container.pack(expand=True)
        
        # Get interval options from config
        interval_options = self.config.get('compartment_intervals_allowed', [1, 2, 3, 5])
        
        row = 0
        
        # Check if we should show increment button
        # if self._has_valid_last_metadata():
        # Determine button color based on validation
        button_color = self._get_increment_button_color()
        
        # Create increment button
        if gui_manager:
            self.increment_button = gui_manager.create_modern_button(
                container,
                text=DialogHelper.t("Increment From Last"),
                color=button_color,
                command=self._on_increment_from_last
            )
        else:
            self.increment_button = tk.Button(
                container,
                text=DialogHelper.t("Increment From Last"),
                command=self._on_increment_from_last
            )
        self.increment_button.grid(row=row, column=0, columnspan=4, pady=(0, 10), sticky=tk.EW)
        row += 1
        
        # Single row for all fields
        fields_frame = ttk.Frame(container, style='Content.TFrame')
        fields_frame.grid(row=row, column=0, sticky=tk.EW)
        
        # Configure column weights
        fields_frame.columnconfigure(1, weight=1)  # Hole ID entry
        fields_frame.columnconfigure(5, weight=1)  # From entry
        fields_frame.columnconfigure(7, weight=1)  # To entry
        
        col = 0
        
        # Hole ID
        ttk.Label(fields_frame, text=DialogHelper.t("Hole ID:"),
                 font=self.gui_manager.fonts["heading"]).grid(row=0, column=col, padx=(0,5))
        col += 1
        
        self.hole_id_entry = create_entry_with_validation(
            fields_frame,
            textvariable=self.hole_id,
            theme_colors=self.theme_colors,
            font=self.gui_manager.fonts["code"],
            validate_func=lambda *args: self._update_entry_color(self.hole_id_entry, self._validate_hole_id()),
            width=10,
            placeholder="AB1234"
        )
        self.hole_id_entry.grid(row=0, column=col, padx=(0, 15))
        self.hole_id_entry.bind('<FocusOut>', self._on_metadata_change_event)
        self.hole_id_entry.bind('<KeyRelease>', self._delayed_metadata_change)
        col += 1
        
        # Separator
        ttk.Label(fields_frame, text="|").grid(row=0, column=col, padx=10)
        col += 1
        
        # Interval dropdown
        ttk.Label(fields_frame, text=DialogHelper.t("Interval:"),
                 font=self.gui_manager.fonts["heading"]).grid(row=0, column=col, padx=(0,5))
        col += 1
                 
        interval_dropdown = tk.OptionMenu(
            fields_frame,
            self.interval_var,
            *interval_options,
            command=self._on_interval_change
        )
        interval_dropdown.config(
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["text"],
            activebackground=self.theme_colors.get("hover_highlight", "#3a3a3a")
        )
        interval_dropdown.grid(row=0, column=col, padx=(0, 5))
        col += 1
        
        ttk.Label(fields_frame, text="m").grid(row=0, column=col, padx=(0, 15))
        col += 1
        
        # Depth From
        ttk.Label(fields_frame, text=DialogHelper.t("From:"),
                 font=self.gui_manager.fonts["heading"]).grid(row=0, column=col, padx=(0,5))
        col += 1
                 
        self.depth_from_entry = create_entry_with_validation(
            fields_frame,
            textvariable=self.depth_from,
            theme_colors=self.theme_colors,
            font=self.gui_manager.fonts["code"],
            validate_func=lambda *args: self._update_entry_color(self.depth_from_entry, self._validate_depth_from()),
            width=8,
            placeholder="0"
        )
        self.depth_from_entry.grid(row=0, column=col, padx=(0, 10))
        self.depth_from_entry.bind('<FocusOut>', self._on_metadata_change_event)
        self.depth_from_entry.bind('<KeyRelease>', self._delayed_metadata_change)
        col += 1
        
        # Depth To
        ttk.Label(fields_frame, text=DialogHelper.t("To:"),
                 font=self.gui_manager.fonts["heading"]).grid(row=0, column=col, padx=(0,5))
        col += 1
                 
        self.depth_to_entry = create_entry_with_validation(
            fields_frame,
            textvariable=self.depth_to,
            theme_colors=self.theme_colors,
            font=self.gui_manager.fonts["code"],
            validate_func=lambda *args: self._update_entry_color(self.depth_to_entry, self._validate_depth_to()),
            width=8,
            placeholder="20"
        )
        self.depth_to_entry.grid(row=0, column=col)
        
        # Bind events
        self.depth_to_entry.bind('<KeyRelease>', self._on_depth_to_edit)
        self.depth_from.trace_add("write", self._on_depth_from_change)
        self.interval_var.trace_add("write", self._on_interval_change)
        
        return self.frame
        
    def _has_valid_last_metadata(self) -> bool:
        """Check if we have valid metadata from last processing."""
        return (self.last_successful_metadata and 
                self.last_successful_metadata.get('hole_id'))
                
    def _get_increment_button_color(self) -> str:
        """Determine increment button color based on validation."""
        if not self._has_valid_last_metadata():
            return self.theme_colors["accent_green"]
            
        # Get incremented data
        incremented = self.get_incremented_metadata()
        if not incremented:
            return self.theme_colors["accent_red"]
            
        # Check if it would be valid
        if self.depth_validator and hasattr(self.depth_validator, 'is_loaded'):
            if self.depth_validator.is_loaded:
                hole_id = incremented.get('hole_id', '')
                depth_from = incremented.get('depth_from', 0)
                # Calculate depth_to based on current interval
                current_interval = self.interval_var.get()
                depth_to = depth_from + (DialogConstants.COMPARTMENT_COUNT * current_interval)
                
                is_valid, _ = self.depth_validator.validate_depth_range(hole_id, depth_from, depth_to)
                return self.theme_colors["accent_green"] if is_valid else self.theme_colors["accent_red"]
                
        return self.theme_colors["accent_green"]
        
    def get_incremented_metadata(self) -> Optional[Dict]:
        """Get incremented metadata based on last successful processing."""
        if not self._has_valid_last_metadata():
            return None
            
        # Calculate increment based on last interval
        last_interval = self.last_successful_metadata.get("compartment_interval", 1)
        increment = DialogConstants.COMPARTMENT_COUNT * last_interval
        
        return {
            "hole_id": self.last_successful_metadata["hole_id"],
            "depth_from": self.last_successful_metadata["depth_to"],
            "depth_to": None,  # Will be calculated based on selected interval
            "compartment_interval": last_interval
        }
        
    def _on_increment_from_last(self):
        """Handle Increment From Last button click."""
        incremented = self.get_incremented_metadata()
        
        if incremented and incremented.get('hole_id'):
            # Reset manual override flag
            self.depth_to_manual_override = False
            
            # Set values
            self.hole_id.set(incremented.get('hole_id', ''))
            self.depth_from.set(str(incremented.get('depth_from', '')))
            
            # Get current interval
            current_interval = self.interval_var.get()
            
            # Calculate depth_to
            if incremented.get('depth_from') is not None:
                depth_to = incremented.get('depth_from') + (DialogConstants.COMPARTMENT_COUNT * current_interval)
                self.depth_to.set(str(depth_to))
            
            # Trigger change callback
            if self.on_metadata_change:
                self.on_metadata_change()
                
            # Update status (if we have access to status_var)
            if hasattr(self.parent, 'status_var'):
                self.parent.status_var.set(DialogHelper.t("Auto-filled with incremented metadata from last core"))
        else:
            DialogHelper.show_message(
                self.parent,
                DialogHelper.t("No Previous Data"),
                DialogHelper.t("No previous metadata available to increment from."),
                message_type="info"
            )
        
    def _validate_hole_id(self) -> str:
        """Validate hole ID format."""
        value = self.hole_id.get().strip()
        if not value:
            return "empty"
        elif not re.match(r'^[A-Za-z]{2}\d{4}$', value):
            return "invalid"
        return "valid"
            
    def _update_entry_color(self, entry, validation_result):
        """Update entry background color based on validation."""
        if validation_result == "empty":
            entry.config(bg=self.theme_colors.get("error_bg", "#ffcccc"))
        elif validation_result == "invalid":
            entry.config(bg=self.theme_colors.get("error_bg", "#ffcccc"))
        elif validation_result == "valid":
            entry.config(bg=self.theme_colors.get("success_bg", "#ccffcc"))
        else:
            entry.config(bg=self.theme_colors["field_bg"])
            
    def _on_interval_change(self, *args):
        """Handle interval change."""
        if not self.depth_to_manual_override:
            self._update_expected_depth_to()
        if self.on_metadata_change:
            self.on_metadata_change()
            
    def _on_depth_from_change(self, *args):
        """Handle depth from change."""
        if not self.depth_to_manual_override:
            self._update_expected_depth_to()
        if self.on_metadata_change:
            self.on_metadata_change()
            
    def _on_depth_to_edit(self, event):
        """Track manual edits to depth_to."""
        self.depth_to_manual_override = True
        self._delayed_metadata_change()
        
    def _on_metadata_change_event(self, event):
        """Handle metadata change from UI event."""
        if self.on_metadata_change:
            self.on_metadata_change()
            
    def _delayed_metadata_change(self, *args):
        """Delayed metadata change notification."""
        if hasattr(self, '_update_timer'):
            self.parent.after_cancel(self._update_timer)
        self._update_timer = self.parent.after(DialogConstants.CANVAS_UPDATE_DELAY_MS, 
                                              lambda: self.on_metadata_change() if self.on_metadata_change else None)
            
    def _update_expected_depth_to(self):
        """Auto-calculate depth_to based on depth_from and interval."""
        try:
            depth_from_str = self.depth_from.get().strip()
            if depth_from_str and depth_from_str.isdigit():
                depth_from = int(depth_from_str)
                interval = self.interval_var.get()
                depth_to = depth_from + (DialogConstants.COMPARTMENT_COUNT * interval)
                self.depth_to.set(str(depth_to))
        except Exception as e:
            self.logger.debug(f"Error updating depth_to: {e}")
            
    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate all metadata fields with non-standard range confirmation.
        
        Returns:
            (is_valid, error_message)
        """
        # Check hole ID
        hole_id = self.hole_id.get().strip()
        if not hole_id:
            return False, DialogHelper.t("Hole ID is required")
        if not re.match(r'^[A-Za-z]{2}\d{4}$', hole_id):
            return False, DialogHelper.t("Hole ID must be 2 letters followed by 4 digits (e.g., %(example)s)", 
                                       example="AB1234")
            
        # Check depths
        try:
            depth_from_str = self.depth_from.get().strip()
            depth_to_str = self.depth_to.get().strip()
            
            # Check for None strings
            if depth_from_str.lower() == 'none' or not depth_from_str:
                return False, DialogHelper.t("Depth From is required")
            if depth_to_str.lower() == 'none' or not depth_to_str:
                return False, DialogHelper.t("Depth To is required")
            
            depth_from = int(depth_from_str)
            depth_to = int(depth_to_str)
            if depth_from < 0 or depth_to < 0:
                return False, DialogHelper.t("Depth values cannot be negative")
            if depth_to <= depth_from:
                return False, DialogHelper.t("Depth To must be greater than Depth From")
                
            # Get compartment interval
            interval = self.interval_var.get()
            
            # Check for non-standard depth range
            compartment_count = self.config.get('compartment_count', DialogConstants.COMPARTMENT_COUNT)
            expected_depth_range = compartment_count * interval
            actual_depth_range = depth_to - depth_from
            
            if actual_depth_range != expected_depth_range:
                # Show confirmation dialog for non-standard range
                if not DialogHelper.confirm_dialog(
                    self.parent,
                    DialogHelper.t("Depth Range Warning"),
                    DialogHelper.t(
                        "The expected depth range for %(interval)sm interval with %(compartment_count)s "
                        "compartments is %(expected_depth_range)sm, but you entered %(actual_depth_range)sm.\n\n"
                        "Do you want to continue with this non-standard depth range?",
                        interval=interval,
                        compartment_count=compartment_count,
                        expected_depth_range=expected_depth_range,
                        actual_depth_range=actual_depth_range
                    ),
                    yes_text=DialogHelper.t("Continue"),
                    no_text=DialogHelper.t("Cancel")
                ):
                    return False, None  # User cancelled
                
            # Use depth validator if available
            if self.depth_validator and hasattr(self.depth_validator, 'is_loaded'):
                if self.depth_validator.is_loaded:
                    is_valid, error_msg = self.depth_validator.validate_depth_range(
                        hole_id, depth_from, depth_to
                    )
                    if not is_valid:
                        # Get valid range for additional info
                        valid_range = self.depth_validator.get_valid_range(hole_id)
                        if valid_range:
                            valid_from, valid_to = valid_range
                            error_msg += DialogHelper.t(
                                "\n\nValid range for %(hole_id)s: %(valid_from)sm - %(valid_to)sm",
                                hole_id=hole_id, valid_from=valid_from, valid_to=valid_to
                            )
                        return False, error_msg
                        
        except ValueError:
            return False, DialogHelper.t("Depth values must be whole numbers")
            
        return True, None
      
    def set_metadata(self, metadata: Dict):
        """Set metadata values from dict."""
        if 'hole_id' in metadata and metadata['hole_id'] is not None:
            self.hole_id.set(metadata['hole_id'])
        if 'depth_from' in metadata and metadata['depth_from'] is not None:
            self.depth_from.set(str(metadata['depth_from']))
        if 'depth_to' in metadata and metadata['depth_to'] is not None:
            self.depth_to.set(str(metadata['depth_to']))
        if 'compartment_interval' in metadata and metadata['compartment_interval'] is not None:
            self.interval_var.set(metadata['compartment_interval'])

    def get_metadata(self) -> CompartmentMetadata:
        """Get current metadata values."""
        # Helper function to safely convert to int
        def safe_int(value_str: str, default: int = 0) -> int:
            value_str = value_str.strip()
            if not value_str or value_str.lower() == 'none':
                return default
            try:
                return int(value_str)
            except ValueError:
                return default
        
        return CompartmentMetadata(
            hole_id=self.hole_id.get().strip(),
            depth_from=safe_int(self.depth_from.get()),
            depth_to=safe_int(self.depth_to.get()),
            compartment_interval=self.interval_var.get()
        )

    def _validate_depth_from(self) -> str:
        """Validate depth from value."""
        value = self.depth_from.get().strip()
        if not value or value.lower() == 'none':
            return "empty"
        try:
            depth = int(value)
            if depth < 0:
                return "invalid"
            return "valid"
        except ValueError:
            return "invalid"

    def _validate_depth_to(self) -> str:
        """Validate depth to value."""
        value = self.depth_to.get().strip()
        if not value or value.lower() == 'none':
            return "empty"
        try:
            depth_to = int(value)
            if depth_to < 0:
                return "invalid"
            depth_from_str = self.depth_from.get().strip()
            if depth_from_str and depth_from_str.isdigit():
                if depth_to <= int(depth_from_str):
                    return "invalid"
            return "valid"
        except ValueError:
            return "invalid"

    def _update_expected_depth_to(self):
        """Auto-calculate depth_to based on depth_from and interval."""
        try:
            depth_from_str = self.depth_from.get().strip()
            if depth_from_str and depth_from_str.isdigit() and depth_from_str.lower() != 'none':
                depth_from = int(depth_from_str)
                interval = self.interval_var.get()
                depth_to = depth_from + (DialogConstants.COMPARTMENT_COUNT * interval)
                self.depth_to.set(str(depth_to))
        except Exception as e:
            self.logger.debug(f"Error updating depth_to: {e}")


from typing import Dict, List, Optional, Tuple, Any, Union
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Import from your existing modules
from gui.dialog_helper import DialogHelper
from gui.gui_manager import GUIManager


class DialogCanvasRenderer:
    """Manages static and dynamic visualization layers for performance optimization."""
    
    def __init__(self, working_image: np.ndarray, theme_colors: Dict[str, str], 
                 gui_manager: Optional[GUIManager] = None) -> None:
        """Initialize the canvas renderer with image and theme settings."""
        self.working_image = working_image
        self.theme_colors = theme_colors
        self.gui_manager = gui_manager
        self.logger = logger
        
        # Static layer cache
        self.static_cache: Optional[np.ndarray] = None
        self.static_cache_params: Optional[Tuple] = None
        
        # Canvas reference and state
        self.canvas: Optional[tk.Canvas] = None
        self.photo_image: Optional[ImageTk.PhotoImage] = None
        self.viz_state = VisualizationState()
        
        # Store data for use in drawing methods
        self.scale_data: Optional[Dict[str, Any]] = None
        self.boundary_analysis: Optional[Dict[str, Any]] = None
        
    def set_canvas(self, canvas: tk.Canvas) -> None:
        """Set the canvas for visualization."""
        self.canvas = canvas
        
    def create_static_visualization(self, 
                                  markers: Dict[int, np.ndarray], 
                                  detected_boundaries: List[Any],
                                  scale_data: Optional[Dict[str, Any]] = None,
                                  show_scale_bar: bool = True,
                                  mouse_hovering: bool = False,
                                  show_wall_detection: bool = False,
                                  boundary_analysis: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Create or retrieve cached static visualization.
        
        Static elements include:
        - Base image
        - Detected markers with scale validation measurements
        - Scale bar and legend (hidden when mouse hovering)
        - Detected boundaries
        
        Note: Wall detection is now drawn in dynamic layers
        """
        # Store data for use in other methods
        self.scale_data = scale_data
        self.boundary_analysis = boundary_analysis
        
        # Create cache key from parameters (excluding wall detection as it's now dynamic)
        cache_key = (
            len(markers),
            len(detected_boundaries),
            show_scale_bar and not mouse_hovering,
            id(scale_data)
        )
        
        # Check if cache is valid
        if self.static_cache is not None and self.static_cache_params == cache_key:
            return self.static_cache.copy()
        
        # Create new static visualization
        self.logger.debug("Creating new static visualization")
        viz_image = self.working_image.copy()
        
        # Draw markers with scale validation
        self._draw_markers(viz_image, markers)
        
        # Draw detected boundaries
        self._draw_boundaries(viz_image, detected_boundaries)
        
        # Draw scale elements (only if not hovering)
        if scale_data and show_scale_bar and not mouse_hovering:
            self._draw_scale_elements(viz_image, scale_data, markers)
        
        # Note: Wall detection is now drawn in dynamic overlays
        
        # Cache the result
        self.static_cache = viz_image.copy()
        self.static_cache_params = cache_key
        
        return viz_image
    
    def draw_dynamic_overlays(self, 
                            canvas: tk.Canvas, 
                            mode: int, 
                            boundary_annotator: Optional['BoundaryAnnotator'] = None,
                            boundary_adjuster: Optional['BoundaryAdjuster'] = None,
                            boundary_state: Optional[BoundaryState] = None,
                            temp_point: Optional[Tuple[int, int]] = None,
                            show_wall_detection: bool = False) -> None:
        """Draw dynamic overlays based on current mode.
        
        Dynamic elements include:
        - Boundary previews during placement
        - Adjustment guides
        - Selection highlights
        - Wall detection visualization (when enabled)
        """
        # Clear previous dynamic elements
        canvas.delete("dynamic")
        canvas.delete("boundary_preview")
        canvas.delete("selection")
        canvas.delete("wall_detection")
        
        if mode == 1:  # MODE_MISSING_BOUNDARIES
            # Boundary preview is handled by BoundaryAnnotator.draw_preview()
            pass
            
        elif mode == 2:  # MODE_ADJUST_BOUNDARIES
            # Draw adjustment guides
            if boundary_state:
                self._draw_adjustment_guides(canvas, boundary_state)
                
            # Draw compartment selection
            if boundary_adjuster:
                boundary_adjuster.draw_selection_overlay(self.viz_state)
                
        # Draw wall detection visualization if enabled (now in dynamic layer)
        if show_wall_detection and self.boundary_analysis:
            self._draw_wall_detection_on_canvas(canvas, self.boundary_analysis)
    
    def _draw_markers(self, image: np.ndarray, markers: Dict[int, np.ndarray]) -> None:
        """Draw ArUco markers with scale validation visualization on the image.
        
        This includes:
        - Edge length measurements with validity coloring
        - Diagonal measurements
        - Marker IDs with descriptive labels for corners
        """
        # Get scale data if available
        scale_px_per_cm: Optional[float] = None
        if self.scale_data and 'scale_px_per_cm' in self.scale_data:
            scale_px_per_cm = self.scale_data['scale_px_per_cm']
        
        for marker_id, corners in markers.items():
            # Convert corners to int for drawing
            corners_int = corners.astype(np.int32)
            
            # Find this marker's measurements in scale data if available
            marker_measurements: Optional[Dict[str, Any]] = None
            if self.scale_data and "marker_measurements" in self.scale_data:
                for measurement in self.scale_data["marker_measurements"]:
                    if measurement.get("marker_id") == marker_id:
                        marker_measurements = measurement
                        break
            
            if marker_measurements and scale_px_per_cm:
                # We have scale data - draw edges with individual validity
                self._draw_marker_with_scale_validation(
                    image, corners_int, marker_measurements, scale_px_per_cm
                )
            else:
                # No scale data - just draw the marker outline
                cv2.polylines(
                    image,
                    [corners_int.reshape(-1, 1, 2)],
                    True,
                    (0, 255, 255),  # Cyan for markers without scale data
                    2
                )
                
                # Draw corner points
                for corner in corners_int:
                    cv2.circle(image, tuple(corner), 3, (0, 255, 255), -1)
            
            # Draw marker label
            self._draw_marker_label(image, marker_id, corners)
    
    def _draw_marker_with_scale_validation(self, 
                                         image: np.ndarray, 
                                         corners: np.ndarray, 
                                         measurements: Dict[str, Any], 
                                         scale_px_per_cm: float) -> None:
        """Draw a marker with edge measurements and validation colors.
        
        Green edges are within tolerance, red edges are outside tolerance.
        """
        edge_lengths = measurements.get("edge_lengths", [])
        diagonal_lengths = measurements.get("diagonal_lengths", [])
        scales = measurements.get("scales", [])
        
        # Determine expected physical size
        physical_size = measurements.get("physical_size_cm", 2.0)
        
        # Draw each edge with its validity color
        for i in range(4):  # 4 edges
            if i < len(edge_lengths) and i < len(scales):
                # Calculate expected length in pixels
                expected_px = physical_size * scale_px_per_cm
                actual_px = edge_lengths[i]
                
                # Check if this edge is within tolerance
                tolerance_pixels = 5.0
                is_valid = abs(actual_px - expected_px) <= tolerance_pixels
                
                # Choose color based on validity
                edge_color = (0, 255, 0) if is_valid else (0, 0, 255)  # Green if valid, red if invalid
                
                # Draw the edge
                start_corner = i
                end_corner = (i + 1) % 4
                cv2.line(
                    image,
                    tuple(corners[start_corner]),
                    tuple(corners[end_corner]),
                    edge_color,
                    2
                )
                
                # Draw measurement text on edge
                self._draw_edge_measurement(
                    image, corners[start_corner], corners[end_corner], 
                    actual_px, i
                )
        
        # Draw diagonals if available
        if len(diagonal_lengths) >= 2 and len(scales) >= 6:
            self._draw_diagonal_measurements(
                image, corners, diagonal_lengths, physical_size, scale_px_per_cm
            )
        
        # Draw corner points
        for corner in corners:
            cv2.circle(image, tuple(corner), 3, (255, 255, 255), -1)
    
    def _draw_edge_measurement(self, 
                             image: np.ndarray, 
                             start_corner: np.ndarray, 
                             end_corner: np.ndarray, 
                             actual_px: float, 
                             edge_index: int) -> None:
        """Draw measurement text on a marker edge."""
        # Calculate edge center
        edge_center_x = (start_corner[0] + end_corner[0]) // 2
        edge_center_y = (start_corner[1] + end_corner[1]) // 2
        
        # Offset text based on edge position to avoid overlap
        text_offset_x = 0
        text_offset_y = -10 if edge_index == 0 else 10 if edge_index == 2 else 0
        text_offset_x = -40 if edge_index == 3 else 40 if edge_index == 1 else 0
        
        # Format measurement text
        measurement_text = f"{actual_px:.0f}px"
        
        # Draw text with background for visibility
        text_size = cv2.getTextSize(
            measurement_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
        )[0]
        
        # Background rectangle
        cv2.rectangle(
            image,
            (
                edge_center_x + text_offset_x - text_size[0] // 2 - 2,
                edge_center_y + text_offset_y - text_size[1] - 2,
            ),
            (
                edge_center_x + text_offset_x + text_size[0] // 2 + 2,
                edge_center_y + text_offset_y + 2,
            ),
            (0, 0, 0),
            -1,
        )
        
        # Draw text
        cv2.putText(
            image,
            measurement_text,
            (
                edge_center_x + text_offset_x - text_size[0] // 2,
                edge_center_y + text_offset_y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
    
    def _draw_diagonal_measurements(self, 
                                  image: np.ndarray, 
                                  corners: np.ndarray, 
                                  diagonal_lengths: List[float], 
                                  physical_size: float, 
                                  scale_px_per_cm: float) -> None:
        """Draw diagonal lines with dashed pattern and validity coloring."""
        diagonals = [(0, 2), (1, 3)]  # Corner pairs for diagonals
        
        for i, (start, end) in enumerate(diagonals):
            if i < len(diagonal_lengths):
                # Calculate expected diagonal length
                expected_diag = physical_size * np.sqrt(2) * scale_px_per_cm
                actual_diag = diagonal_lengths[i]
                
                # Check validity with tolerance
                tolerance_ratio = 0.05  # 5% tolerance
                is_valid = abs(actual_diag - expected_diag) / expected_diag < tolerance_ratio
                
                # Choose color based on validity
                diag_color = (0, 255, 0) if is_valid else (0, 0, 255)
                
                # Draw diagonal with dashed line pattern
                start_pt = corners[start]
                end_pt = corners[end]
                
                # Create dashed line by drawing segments
                num_dashes = 10
                for j in range(0, num_dashes, 2):
                    t1 = j / num_dashes
                    t2 = min((j + 1) / num_dashes, 1.0)
                    pt1 = (
                        int(start_pt[0] + t1 * (end_pt[0] - start_pt[0])),
                        int(start_pt[1] + t1 * (end_pt[1] - start_pt[1])),
                    )
                    pt2 = (
                        int(start_pt[0] + t2 * (end_pt[0] - start_pt[0])),
                        int(start_pt[1] + t2 * (end_pt[1] - start_pt[1])),
                    )
                    cv2.line(image, pt1, pt2, diag_color, 1)
    
    def _draw_marker_label(self, 
                         image: np.ndarray, 
                         marker_id: int, 
                         corners: np.ndarray) -> None:
        """Draw marker ID label with appropriate styling."""
        # Calculate center position
        center = np.mean(corners, axis=0).astype(int)
        
        # Determine marker type and styling
        if marker_id in DialogConstants.CORNER_MARKER_IDS:
            color = (255, 255, 0)  # Yellow for corners
            # Descriptive labels for corner markers
            corner_labels = {0: "0 (TL)", 1: "1 (TR)", 2: "2 (BR)", 3: "3 (BL)"}
            label = corner_labels.get(marker_id, f"C{marker_id}")
            font_scale = 0.6
        elif marker_id in DialogConstants.COMPARTMENT_MARKER_IDS:
            color = (0, 255, 0)  # Green for compartments
            label = str(marker_id)
            font_scale = 0.5
        elif marker_id == DialogConstants.METADATA_MARKER_ID:
            color = (255, 0, 255)  # Magenta for metadata
            label = "24"
            font_scale = 0.5
        else:
            color = (255, 0, 255)  # Magenta for others
            label = str(marker_id)
            font_scale = 0.5
        
        # Draw label with background
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        
        # Background rectangle for text visibility
        cv2.rectangle(
            image,
            (center[0] - text_size[0]//2 - 3, center[1] - text_size[1]//2 - 3),
            (center[0] + text_size[0]//2 + 3, center[1] + text_size[1]//2 + 3),
            (0, 0, 0), 
            -1
        )
        
        # Draw text
        cv2.putText(
            image, 
            label, 
            (center[0] - text_size[0]//2, center[1] + text_size[1]//2),
            font, 
            font_scale, 
            color, 
            thickness
        )
    
    def _draw_boundaries(self, image: np.ndarray, boundaries: List[Any]) -> None:
        """Draw compartment boundaries on the image."""
        for boundary in boundaries:
            if isinstance(boundary, (list, tuple)) and len(boundary) >= 4:
                x1, y1, x2, y2 = boundary[:4]
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            elif isinstance(boundary, dict) and all(k in boundary for k in ['x1', 'y1', 'x2', 'y2']):
                # Manual boundaries drawn in different color
                cv2.rectangle(
                    image,
                    (int(boundary['x1']), int(boundary['y1'])),
                    (int(boundary['x2']), int(boundary['y2'])),
                    (255, 0, 255), 2  # Magenta for manual
                )
    
    def _draw_scale_elements(self, 
                           image: np.ndarray, 
                           scale_data: Dict[str, Any], 
                           markers: Dict[int, np.ndarray]) -> None:
        """Draw scale bar with legend and distance measurements on markers."""
        if 'scale_px_per_cm' not in scale_data:
            return
        
        scale_px_per_cm = scale_data['scale_px_per_cm']
        
        # Draw enhanced scale bar with legend
        self._draw_scale_bar(image, scale_px_per_cm)
        
        # Draw distance measurements between adjacent compartment markers
        self._draw_marker_distances(image, markers, scale_px_per_cm)
    
    def _draw_scale_bar(self, image: np.ndarray, scale_px_per_cm: float) -> None:
        """Draw a checkered scale bar with confidence display and legend."""
        # Scale bar properties
        bar_length_cm = 10  # 10cm scale bar
        bar_length_px = int(bar_length_cm * scale_px_per_cm)
        bar_height = 20
        margin = 20
        
        # Position at bottom right
        img_h, img_w = image.shape[:2]
        bar_x = margin
        bar_y = img_h - margin - bar_height - 60  # Extra space for confidence text and legend
        
        # Draw white background for entire scale bar area
        bg_padding = 10
        cv2.rectangle(
            image,
            (bar_x - bg_padding, bar_y - 30 - bg_padding),  # Extended up for confidence text
            (bar_x + bar_length_px + bg_padding, bar_y + bar_height + 40 + bg_padding),  # Extended down for legend
            (255, 255, 255),
            -1
        )
        
        # Draw checkered scale bar (1cm segments)
        for i in range(bar_length_cm):
            segment_start = bar_x + int(i * scale_px_per_cm)
            segment_end = bar_x + int((i + 1) * scale_px_per_cm)
            
            # Alternate black and white segments
            color = (0, 0, 0) if i % 2 == 0 else (200, 200, 200)
            cv2.rectangle(
                image,
                (segment_start, bar_y),
                (segment_end, bar_y + bar_height),
                color,
                -1
            )
        
        # Draw border around scale bar
        cv2.rectangle(
            image,
            (bar_x, bar_y),
            (bar_x + bar_length_px, bar_y + bar_height),
            (0, 0, 0),
            2
        )
        
        # Add labels at ends
        cv2.putText(
            image,
            "0cm",
            (bar_x, bar_y + bar_height + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
        
        text_size = cv2.getTextSize("10cm", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.putText(
            image,
            "10cm",
            (bar_x + bar_length_px - text_size[0], bar_y + bar_height + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
        
        # Add confidence text above the bar
        if self.scale_data:
            confidence = self.scale_data.get('confidence', 0.0)
            n_markers = self.scale_data.get('n_markers_used', 0)
            scale_text = f"{scale_px_per_cm:.1f} px/cm ({confidence:.0%} confidence, n={n_markers})"
        else:
            scale_text = f"{scale_px_per_cm:.1f} px/cm"
        
        text_size = cv2.getTextSize(scale_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = bar_x + (bar_length_px - text_size[0]) // 2
        cv2.putText(
            image,
            scale_text,
            (text_x, bar_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )
        
        # Draw legend below scale bar
        self._draw_scale_legend(image, bar_x, bar_y + bar_height + 35, bar_length_px)
    
    def _draw_scale_legend(self, image: np.ndarray, x: int, y: int, width: int) -> None:
        """Draw legend explaining green/red edge colors."""
        # Draw green line with label
        line_length = 30
        cv2.line(image, (x, y), (x + line_length, y), (0, 255, 0), 2)
        cv2.putText(
            image,
            "Valid edge",
            (x + line_length + 10, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )
        
        # Draw red line with label
        red_x = x + width // 2
        cv2.line(image, (red_x, y), (red_x + line_length, y), (0, 0, 255), 2)
        cv2.putText(
            image,
            "Invalid edge",
            (red_x + line_length + 10, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )
    
    def _draw_marker_distances(self, 
                             image: np.ndarray, 
                             markers: Dict[int, np.ndarray], 
                             scale_px_per_cm: float) -> None:
        """Draw distance measurements between adjacent compartment markers."""
        compartment_markers: Dict[int, np.ndarray] = {}
        
        # Collect compartment markers
        for marker_id, corners in markers.items():
            if marker_id in DialogConstants.COMPARTMENT_MARKER_IDS:
                center = np.mean(corners, axis=0)
                compartment_markers[marker_id] = center
        
        # Sort by marker ID
        sorted_markers = sorted(compartment_markers.items())
        
        # Draw distances between adjacent markers
        for i in range(len(sorted_markers) - 1):
            id1, pos1 = sorted_markers[i]
            id2, pos2 = sorted_markers[i + 1]
            
            # Calculate distance
            distance_px = np.linalg.norm(pos2 - pos1)
            distance_cm = distance_px / scale_px_per_cm
            
            # Determine if distance is valid (should be ~5cm for adjacent compartments)
            expected_distance_cm = DialogConstants.COMPARTMENT_HEIGHT_CM
            tolerance_cm = 1.0
            is_valid = abs(distance_cm - expected_distance_cm) <= tolerance_cm
            color = (0, 255, 0) if is_valid else (0, 0, 255)  # Green if valid, red if not
            
            # Draw line between markers
            cv2.line(image, tuple(pos1.astype(int)), tuple(pos2.astype(int)), color, 1)
            
            # Draw distance text at midpoint
            mid_point = ((pos1 + pos2) / 2).astype(int)
            text = f"{distance_cm:.1f}cm"
            
            # Background for text visibility
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.rectangle(
                image,
                (mid_point[0] - text_size[0]//2 - 2, mid_point[1] - text_size[1]//2 - 2),
                (mid_point[0] + text_size[0]//2 + 2, mid_point[1] + text_size[1]//2 + 2),
                (0, 0, 0), 
                -1
            )
            
            cv2.putText(
                image, 
                text, 
                (mid_point[0] - text_size[0]//2, mid_point[1] + text_size[1]//2),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.4, 
                color, 
                1
            )
    
    def _draw_wall_detection_on_canvas(self, 
                                     canvas: tk.Canvas, 
                                     boundary_analysis: Dict[str, Any]) -> None:
        """Draw wall detection visualization on canvas as dynamic elements.
        
        This draws search regions and detected edges directly on the canvas
        rather than the image, making them part of the dynamic layer.
        """
        wall_viz = boundary_analysis.get("wall_detection_results", {})
        if not wall_viz:
            return
        
        # Draw search regions
        search_regions = wall_viz.get("search_regions", [])
        for region_data in search_regions:
            if len(region_data) == 6:
                x1, y1, x2, y2, color, thickness = region_data
                
                # Convert image coordinates to canvas coordinates
                x1_canvas = int(x1 * self.viz_state.scale_ratio + self.viz_state.canvas_offset_x)
                y1_canvas = int(y1 * self.viz_state.scale_ratio + self.viz_state.canvas_offset_y)
                x2_canvas = int(x2 * self.viz_state.scale_ratio + self.viz_state.canvas_offset_x)
                y2_canvas = int(y2 * self.viz_state.scale_ratio + self.viz_state.canvas_offset_y)
                
                # Convert BGR color to hex for tkinter
                hex_color = f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"
                
                # Draw rectangle
                canvas.create_rectangle(
                    x1_canvas, y1_canvas, x2_canvas, y2_canvas,
                    outline=hex_color, 
                    width=thickness,
                    tags=("wall_detection", "dynamic")
                )
                
                # Add label
                canvas.create_text(
                    x1_canvas + 5, y1_canvas + 15,
                    text="Search",
                    fill=hex_color,
                    anchor=tk.W,
                    font=("Arial", 8),
                    tags=("wall_detection", "dynamic")
                )
        
        # Draw detected edges
        detected_edges = wall_viz.get("detected_edges", [])
        for edge_data in detected_edges:
            if len(edge_data) == 6:
                x1, y1, x2, y2, color, thickness = edge_data
                
                # Convert to canvas coordinates
                x1_canvas = int(x1 * self.viz_state.scale_ratio + self.viz_state.canvas_offset_x)
                y1_canvas = int(y1 * self.viz_state.scale_ratio + self.viz_state.canvas_offset_y)
                x2_canvas = int(x2 * self.viz_state.scale_ratio + self.viz_state.canvas_offset_x)
                y2_canvas = int(y2 * self.viz_state.scale_ratio + self.viz_state.canvas_offset_y)
                
                # Convert color
                hex_color = f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"
                
                # Draw line
                canvas.create_line(
                    x1_canvas, y1_canvas, x2_canvas, y2_canvas,
                    fill=hex_color,
                    width=thickness,
                    tags=("wall_detection", "dynamic")
                )
                
                # Add small labels for edge detection markers
                if abs(y2 - y1) < 50:  # This is a short line
                    # Determine position type
                    img_height = self.working_image.shape[0] if self.working_image is not None else 800
                    if y1 < img_height * 0.2:  # Top marker
                        label = "T"
                    elif y2 > img_height * 0.8:  # Bottom marker
                        label = "B"
                    else:  # Center marker
                        # Draw a small circle instead
                        center_x = (x1_canvas + x2_canvas) // 2
                        center_y = (y1_canvas + y2_canvas) // 2
                        canvas.create_oval(
                            center_x - 3, center_y - 3,
                            center_x + 3, center_y + 3,
                            outline=hex_color,
                            width=1,
                            tags=("wall_detection", "dynamic")
                        )
                        continue
                    
                    # Draw text label
                    canvas.create_text(
                        x1_canvas + 3, y1_canvas + 10 if label == "T" else y2_canvas - 3,
                        text=label,
                        fill=hex_color,
                        anchor=tk.W if label == "T" else tk.SW,
                        font=("Arial", 7),
                        tags=("wall_detection", "dynamic")
                    )
    
    def _draw_adjustment_guides(self, 
                              canvas: tk.Canvas, 
                              boundary_state: BoundaryState) -> None:
        """Draw boundary adjustment guide lines on canvas."""
        if not self.working_image:
            return
        
        img_width = self.working_image.shape[1]
        
        # Calculate boundary positions with offsets
        left_top_y = boundary_state.top_y + boundary_state.left_height_offset
        right_top_y = boundary_state.top_y + boundary_state.right_height_offset
        left_bottom_y = boundary_state.bottom_y + boundary_state.left_height_offset
        right_bottom_y = boundary_state.bottom_y + boundary_state.right_height_offset
        
        # Convert to canvas coordinates
        left_x = self.viz_state.canvas_offset_x
        right_x = img_width * self.viz_state.scale_ratio + self.viz_state.canvas_offset_x
        
        # Top boundary line
        left_top_canvas = left_top_y * self.viz_state.scale_ratio + self.viz_state.canvas_offset_y
        right_top_canvas = right_top_y * self.viz_state.scale_ratio + self.viz_state.canvas_offset_y
        
        canvas.create_line(
            left_x, left_top_canvas, right_x, right_top_canvas,
            fill="green", 
            width=2, 
            tags="dynamic"
        )
        
        # Bottom boundary line
        left_bottom_canvas = left_bottom_y * self.viz_state.scale_ratio + self.viz_state.canvas_offset_y
        right_bottom_canvas = right_bottom_y * self.viz_state.scale_ratio + self.viz_state.canvas_offset_y
        
        canvas.create_line(
            left_x, left_bottom_canvas, right_x, right_bottom_canvas,
            fill="green", 
            width=2, 
            tags="dynamic"
        )
        
        # Add text labels at ends
        if self.gui_manager and hasattr(self.gui_manager, 'fonts'):
            font = self.gui_manager.fonts["heading"]
        else:
            font = ("Arial", 10, "bold")
            
        canvas.create_text(
            left_x + 10, (left_top_canvas + left_bottom_canvas) / 2,
            text="LEFT", 
            fill="green", 
            anchor=tk.W,
            font=font, 
            tags="dynamic"
        )
        
        canvas.create_text(
            right_x - 10, (right_top_canvas + right_bottom_canvas) / 2,
            text="RIGHT", 
            fill="green", 
            anchor=tk.E,
            font=font, 
            tags="dynamic"
        )
    
    def display_image_on_canvas(self, 
                              image: np.ndarray, 
                              canvas: Optional[tk.Canvas] = None) -> None:
        """Display image on canvas with proper scaling and store transformation parameters."""
        if canvas is None:
            canvas = self.canvas
        if canvas is None:
            return
        
        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Get canvas dimensions
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 800
            canvas_height = 600
        
        # Calculate scale
        img_width, img_height = pil_image.size
        scale_x = (canvas_width - 20) / img_width
        scale_y = (canvas_height - 20) / img_height
        scale = min(scale_x, scale_y, 2.0)
        
        # Resize image
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        resized = pil_image.resize((new_width, new_height), Image.LANCZOS)
        
        # Create PhotoImage
        self.photo_image = ImageTk.PhotoImage(resized)
        
        # Calculate offsets for centering
        x_offset = (canvas_width - new_width) // 2
        y_offset = (canvas_height - new_height) // 2
        
        # Update visualization state
        self.viz_state.scale_ratio = scale
        self.viz_state.canvas_offset_x = x_offset
        self.viz_state.canvas_offset_y = y_offset
        
        # Clear canvas and display image
        canvas.delete("all")
        canvas.create_image(
            x_offset, y_offset, 
            anchor=tk.NW,
            image=self.photo_image, 
            tags="base_image"
        )
    
    def invalidate_cache(self) -> None:
        """Invalidate the static cache, forcing recreation on next update."""
        self.static_cache = None
        self.static_cache_params = None
        self.logger.debug("Static cache invalidated")
    
    def get_zoom_region(self, 
                       center_x: int, 
                       center_y: int, 
                       size: int) -> Optional[np.ndarray]:
        """Extract a region from the working image for zoom display."""
        if self.working_image is None:
            return None
        
        h, w = self.working_image.shape[:2]
        
        # Calculate extraction bounds
        x1 = max(0, center_x - size // 2)
        y1 = max(0, center_y - size // 2)
        x2 = min(w, x1 + size)
        y2 = min(h, y1 + size)
        
        # Validate bounds
        if x2 <= x1 or y2 <= y1:
            return None
        
        return self.working_image[y1:y2, x1:x2].copy()
    
    def canvas_to_image_coords(self, canvas_x: int, canvas_y: int) -> Tuple[int, int]:
        """Convert canvas coordinates to image coordinates."""
        img_x = int((canvas_x - self.viz_state.canvas_offset_x) / self.viz_state.scale_ratio)
        img_y = int((canvas_y - self.viz_state.canvas_offset_y) / self.viz_state.scale_ratio)
        return img_x, img_y
    
    def image_to_canvas_coords(self, img_x: int, img_y: int) -> Tuple[int, int]:
        """Convert image coordinates to canvas coordinates."""
        canvas_x = int(img_x * self.viz_state.scale_ratio + self.viz_state.canvas_offset_x)
        canvas_y = int(img_y * self.viz_state.scale_ratio + self.viz_state.canvas_offset_y)
        return canvas_x, canvas_y



# BoundaryAnnotator Component

class BoundaryAnnotator:
    """Handles placement of missing compartment boundaries with two-phase interaction."""
    
    def __init__(self, canvas: tk.Canvas, working_image: np.ndarray,
                 missing_marker_ids: List[int], 
                 marker_to_compartment: Dict[int, int],
                 boundary_state: BoundaryState,
                 avg_compartment_width: int,
                 scale_data: Optional[Dict] = None,
                 config: Optional[Dict] = None,
                 existing_boundaries: Optional[List] = None,
                 on_annotation_complete: Optional[Callable] = None,
                 gui_manager: Optional[GUIManager] = None):
        
        self.gui_manager = gui_manager
        self.canvas = canvas
        self.working_image = working_image
        self.missing_marker_ids = missing_marker_ids.copy()
        self.marker_to_compartment = marker_to_compartment
        self.boundary_state = boundary_state
        self.avg_width = avg_compartment_width
        self.scale_data = scale_data
        self.config = config or {}
        self.existing_boundaries = existing_boundaries or []
        self.on_annotation_complete = on_annotation_complete
        
        self.current_index = 0
        self.placed_positions = {}
        self.result_boundaries = []
        self.annotation_complete = False
        self.phase = 1  # Phase 1: placing, Phase 2: removing/replacing
        self.logger = logger
        
        # Get metadata marker IDs from config
        self.metadata_marker_ids = self.config.get('metadata_marker_ids', [DialogConstants.METADATA_MARKER_ID])
        
    def needs_vertical_boundary_placement(self) -> bool:
        """Check if we need to place vertical boundaries first."""
        if not self.missing_marker_ids or self.phase == 2:
            return False
        if self.current_index >= len(self.missing_marker_ids):
            return False
        current_id = self.missing_marker_ids[self.current_index]
        return current_id in DialogConstants.CORNER_MARKER_IDS
        
    def place_vertical_boundaries(self, center_y: int) -> bool:
        """Place top and bottom boundaries from center click."""
        if not self.scale_data or 'scale_px_per_cm' not in self.scale_data:
            self.logger.error("Cannot place boundaries - no scale data")
            return False
            
        scale_px_per_cm = self.scale_data['scale_px_per_cm']
        compartment_height_px = int(DialogConstants.COMPARTMENT_HEIGHT_CM * scale_px_per_cm)
        
        half_height = compartment_height_px // 2
        self.boundary_state.top_y = max(0, center_y - half_height)
        self.boundary_state.bottom_y = min(self.working_image.shape[0], center_y + half_height)
        
        # Remove corner markers from missing list
        self.missing_marker_ids = [mid for mid in self.missing_marker_ids 
                                  if mid not in DialogConstants.CORNER_MARKER_IDS]
        self.current_index = 0
        
        self.logger.info(f"Placed vertical boundaries: top={self.boundary_state.top_y}, "
                        f"bottom={self.boundary_state.bottom_y}")
        
        return True
        
    def handle_click(self, image_x: int, image_y: int) -> Optional[Dict]:
        """Handle click for both phase 1 (placement) and phase 2 (removal/replacement)."""
        
        # Phase 1: Initial placement
        if self.phase == 1 and self.current_index < len(self.missing_marker_ids):
            return self._handle_placement_click(image_x, image_y)
            
        # Phase 2: Click to remove/re-add
        elif self.phase == 2 or self.annotation_complete:
            return self._handle_removal_click(image_x, image_y)
            
        return None
        
    def _handle_placement_click(self, image_x: int, image_y: int) -> Optional[Dict]:
        """Handle click during initial placement phase."""
        current_id = self.missing_marker_ids[self.current_index]
        
        # Skip metadata markers
        if current_id in self.metadata_marker_ids:
            self.logger.info(f"Skipping metadata marker {current_id}")
            self.current_index += 1
            if self.current_index >= len(self.missing_marker_ids):
                self._complete_phase_1()
            return None
        
        # Check for overlap (only for compartment markers)
        if current_id in DialogConstants.COMPARTMENT_MARKER_IDS:
            if self._would_overlap(image_x):
                return None
                
        # Place the boundary
        boundary_info = self._place_boundary(current_id, image_x, image_y)
        
        # Move to next
        self.current_index += 1
        if self.current_index >= len(self.missing_marker_ids):
            self._complete_phase_1()
            
        return boundary_info
        
    def _handle_removal_click(self, image_x: int, image_y: int) -> Optional[Dict]:
        """Handle click to remove/re-add a compartment in phase 2."""
        # Find which compartment was clicked
        clicked_boundary = None
        clicked_index = None
        
        for i, boundary in enumerate(self.result_boundaries):
            if boundary['x1'] <= image_x <= boundary['x2'] and \
               boundary['y1'] <= image_y <= boundary['y2']:
                clicked_boundary = boundary
                clicked_index = i
                break
                
        if clicked_boundary:
            marker_id = clicked_boundary['marker_id']
            
            # Remove from result boundaries
            self.result_boundaries.pop(clicked_index)
            
            # Remove from placed positions
            if marker_id in self.placed_positions:
                del self.placed_positions[marker_id]
                
            # Add back to missing markers list for re-placement
            self.missing_marker_ids.append(marker_id)
            self.current_index = len(self.missing_marker_ids) - 1
            
            # Switch back to phase 1 for this marker
            self.phase = 1
            self.annotation_complete = False
            
            # Return info about removal
            compartment_num = self.marker_to_compartment.get(marker_id, marker_id - 3)
            return {
                'action': 'removed',
                'marker_id': marker_id,
                'compartment_number': compartment_num,
                'message': DialogHelper.t(
                    "Compartment %(num)s removed. Move mouse to position and click to place.",
                    num=compartment_num
                )
            }
            
        return None
        
    def _complete_phase_1(self):
        """Complete phase 1 and move to phase 2."""
        self.annotation_complete = True
        self.phase = 2
        
        # Call completion callback
        if self.on_annotation_complete:
            self.on_annotation_complete()
            
        self.logger.info("Phase 1 complete - all markers placed. Entering phase 2.")
        
    def _would_overlap(self, x_pos: int) -> bool:
        """Check if placement would overlap existing boundaries."""
        half_width = self.avg_width // 2
        new_x1 = max(0, x_pos - half_width)
        new_x2 = min(self.working_image.shape[1] - 1, x_pos + half_width)
        
        # Check against manually placed boundaries
        for boundary in self.result_boundaries:
            if 'x1' in boundary and 'x2' in boundary:
                if new_x1 < boundary['x2'] and new_x2 > boundary['x1']:
                    return True
                    
        # Check against existing detected boundaries
        for boundary in self.existing_boundaries:
            if isinstance(boundary, (list, tuple)) and len(boundary) >= 4:
                x1, _, x2, _ = boundary[:4]
                if new_x1 < x2 and new_x2 > x1:
                    return True
                    
        return False
        
    def _place_boundary(self, marker_id: int, x: int, y: int) -> Dict:
        """Place a boundary at given position."""
        if marker_id in DialogConstants.CORNER_MARKER_IDS:
            # Corner marker placement
            self.placed_positions[marker_id] = (x, y)
            return {'marker_id': marker_id, 'position': (x, y), 'type': 'corner'}
            
        else:
            # Compartment boundary
            half_width = self.avg_width // 2
            x1 = max(0, x - half_width)
            x2 = min(self.working_image.shape[1] - 1, x + half_width)
            
            # Calculate y-coordinates using current boundary state
            img_width = self.working_image.shape[1]
            left_top_y = self.boundary_state.top_y + self.boundary_state.left_height_offset
            right_top_y = self.boundary_state.top_y + self.boundary_state.right_height_offset
            left_bottom_y = self.boundary_state.bottom_y + self.boundary_state.left_height_offset
            right_bottom_y = self.boundary_state.bottom_y + self.boundary_state.right_height_offset
            
            # Calculate slopes
            if img_width > 0:
                top_slope = (right_top_y - left_top_y) / img_width
                bottom_slope = (right_bottom_y - left_bottom_y) / img_width
                
                # Calculate y values at this x-position
                mid_x = (x1 + x2) / 2
                y1 = int(left_top_y + (top_slope * mid_x))
                y2 = int(left_bottom_y + (bottom_slope * mid_x))
            else:
                y1 = self.boundary_state.top_y
                y2 = self.boundary_state.bottom_y
            
            boundary = {
                'x1': x1, 
                'y1': y1,
                'x2': x2, 
                'y2': y2,
                'marker_id': marker_id,
                'compartment_number': self.marker_to_compartment.get(marker_id, marker_id - 3),
                'center_x': x,
                'is_manual': True
            }
            
            self.result_boundaries.append(boundary)
            self.placed_positions[marker_id] = (x, y)
            
            return boundary
            
    def draw_preview(self, canvas_x: int, canvas_y: int, viz_state: VisualizationState):
        """Draw preview overlay on canvas."""
        
        # Don't show preview in phase 2 or if complete
        if self.phase == 2 or (self.phase == 1 and self.current_index >= len(self.missing_marker_ids)):
            return
        
        self.canvas.delete("boundary_preview")
        current_id = self.missing_marker_ids[self.current_index]
        
        # Skip metadata markers
        if current_id in self.metadata_marker_ids:
            return
        
        # Convert canvas to image coordinates
        image_x = int((canvas_x - viz_state.canvas_offset_x) / viz_state.scale_ratio)
        image_y = int((canvas_y - viz_state.canvas_offset_y) / viz_state.scale_ratio)
        
        if current_id in DialogConstants.CORNER_MARKER_IDS:
            # Draw horizontal line for corner marker
            self.canvas.create_line(
                0, canvas_y, self.canvas.winfo_width(), canvas_y,
                fill="green" if current_id in [0, 1] else "red",
                width=2, tags="boundary_preview"
            )
            
            # Add text label
            text = DialogHelper.t("TOP CONSTRAINT") if current_id in [0, 1] else DialogHelper.t("BOTTOM CONSTRAINT")
            text_x = self.canvas.winfo_width() // 2
            
            self.canvas.create_rectangle(
                text_x - 100, canvas_y - 20,
                text_x + 100, canvas_y + 5,
                fill="black", outline="", tags="boundary_preview"
            )
            self.canvas.create_text(
                text_x, canvas_y - 10,
                text=text, fill="green" if current_id in [0, 1] else "red",
                font=self.gui_manager.fonts["subtitle"],
                tags="boundary_preview"
            )
            
        elif current_id in DialogConstants.COMPARTMENT_MARKER_IDS:
            # Draw compartment preview
            half_width = self.avg_width // 2
            x1 = max(0, image_x - half_width)
            x2 = min(self.working_image.shape[1] - 1, image_x + half_width)
            
            # Calculate y-coordinates
            img_width = self.working_image.shape[1]
            left_top_y = self.boundary_state.top_y + self.boundary_state.left_height_offset
            right_top_y = self.boundary_state.top_y + self.boundary_state.right_height_offset
            left_bottom_y = self.boundary_state.bottom_y + self.boundary_state.left_height_offset
            right_bottom_y = self.boundary_state.bottom_y + self.boundary_state.right_height_offset
            
            if img_width > 0:
                top_slope = (right_top_y - left_top_y) / img_width
                bottom_slope = (right_bottom_y - left_bottom_y) / img_width
                mid_x = (x1 + x2) / 2
                y1 = int(left_top_y + (top_slope * mid_x))
                y2 = int(left_bottom_y + (bottom_slope * mid_x))
            else:
                y1 = self.boundary_state.top_y
                y2 = self.boundary_state.bottom_y
            
            # Convert to canvas coordinates
            x1_canvas = x1 * viz_state.scale_ratio + viz_state.canvas_offset_x
            x2_canvas = x2 * viz_state.scale_ratio + viz_state.canvas_offset_x
            y1_canvas = y1 * viz_state.scale_ratio + viz_state.canvas_offset_y
            y2_canvas = y2 * viz_state.scale_ratio + viz_state.canvas_offset_y
            
            # Check for overlap
            would_overlap = self._would_overlap(image_x)
            color = "red" if would_overlap else "magenta"
            
            self.canvas.create_rectangle(
                x1_canvas, y1_canvas, x2_canvas, y2_canvas,
                outline=color, width=2, tags="boundary_preview"
            )
            
            # Add depth label
            depth = self.marker_to_compartment.get(current_id, current_id - 3)
            mid_x = (x1_canvas + x2_canvas) / 2
            mid_y = (y1_canvas + y2_canvas) / 2
            
            # Background for text
            self.canvas.create_rectangle(
                mid_x - 30, mid_y - 15,
                mid_x + 30, mid_y + 15,
                fill="black", outline="",
                tags="boundary_preview"
            )
            
            self.canvas.create_text(
                mid_x, mid_y, text=f"{depth}m",
                fill=color, font=self.gui_manager.fonts["subtitle"],
                tags="boundary_preview"
            )
            
            if would_overlap:
                # Add overlap warning
                self.canvas.create_text(
                    mid_x, mid_y + 25,
                    text=DialogHelper.t("OVERLAP!"),
                    fill="red", font=self.gui_manager.fonts["heading"],
                    tags="boundary_preview"
                )
            
    def undo_last(self) -> bool:
        """Undo last placement."""
        if self.phase == 1 and self.current_index > 0:
            self.current_index -= 1
            current_id = self.missing_marker_ids[self.current_index]
            
            # Skip metadata markers when undoing
            while current_id in self.metadata_marker_ids and self.current_index > 0:
                self.current_index -= 1
                current_id = self.missing_marker_ids[self.current_index]
            
            # Remove from placed positions
            if current_id in self.placed_positions:
                del self.placed_positions[current_id]
                
            # Remove from result boundaries if it's there
            self.result_boundaries = [b for b in self.result_boundaries 
                                    if b.get('marker_id') != current_id]
            
            self.annotation_complete = False
            return True
        return False
        
    def get_status_message(self) -> str:
        """Get appropriate status message for current state."""
        if self.phase == 1 and self.current_index < len(self.missing_marker_ids):
            current_id = self.missing_marker_ids[self.current_index]
            
            if current_id in DialogConstants.CORNER_MARKER_IDS:
                return DialogHelper.t("Click in the center of where compartments should be placed")
            elif current_id in self.metadata_marker_ids:
                return DialogHelper.t("Metadata marker detected - skipping")
            else:
                depth = self.marker_to_compartment.get(current_id, current_id - 3)
                return DialogHelper.t(
                    "Click to place compartment at depth %(depth)sm (marker %(marker)s)",
                    depth=depth, marker=current_id
                )
        elif self.phase == 2:
            return DialogHelper.t(
                "All markers placed! Click any compartment to remove and re-place it.\n"
                "Click 'Continue' when done."
            )
        else:
            return DialogHelper.t("All markers placed. Click 'Continue' to proceed.")
        






class BoundaryAdjuster:
    """Handles fine-tuning of boundary positions with per-compartment selection."""
    
    def __init__(self, canvas: tk.Canvas, boundary_state: BoundaryState,
                 on_adjustment: Optional[Callable] = None):
        self.canvas = canvas
        self.boundary_state = boundary_state
        self.on_adjustment = on_adjustment
        self._last_adjustment_time = 0
        self.logger = logger
        
        # Selection state
        self.selected_compartment_index = None
        self.selected_compartment = None
        self.all_compartments = []  # Will store all compartments (detected + manual)
        
        # Per-compartment adjustments
        self.compartment_adjustments = {}  # compartment_index -> {'top_offset': 0, 'bottom_offset': 0}
        
    def create_controls(self, parent_frame: tk.Widget, theme_colors: Dict, 
                       gui_manager=None) -> tk.Widget:
        """Create adjustment control UI."""
        frame = ttk.Frame(parent_frame, style='Content.TFrame')
        
        # Title
        ttk.Label(frame, text=DialogHelper.t("Boundary Adjustment"),
                 font=gui_manager.fonts["subtitle"] if gui_manager else ("Arial", 12, "bold")).pack(pady=(5, 10))
        
        # Selection info
        self.selection_label = ttk.Label(
            frame, 
            text=DialogHelper.t("Click a compartment to select it for individual adjustment"),
            style='Content.TLabel'
        )
        self.selection_label.pack(pady=(0, 10))
        
        # Three columns for global adjustments
        columns_frame = ttk.Frame(frame, style='Content.TFrame')
        columns_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Left column - Move Left Side
        left_col = ttk.Frame(columns_frame)
        left_col.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        ttk.Label(left_col, text=DialogHelper.t("Move Left Side"),
                 font=gui_manager.fonts["heading"] if gui_manager else ("Arial", 10, "bold")).pack(pady=(0, 5))
                 
        if gui_manager:
            left_up = gui_manager.create_modern_button(
                left_col, text="▲", color=theme_colors["accent_blue"],
                command=lambda: self.adjust_side_height("left", -DialogConstants.ADJUSTMENT_STEP_PX)
            )
            left_up.pack(fill=tk.X, pady=2)
            
            left_down = gui_manager.create_modern_button(
                left_col, text="▼", color=theme_colors["accent_blue"],
                command=lambda: self.adjust_side_height("left", DialogConstants.ADJUSTMENT_STEP_PX)
            )
            left_down.pack(fill=tk.X, pady=2)
        
        # Center column - Move All
        center_col = ttk.Frame(columns_frame)
        center_col.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        ttk.Label(center_col, text=DialogHelper.t("Move All"),
                 font=gui_manager.fonts["heading"] if gui_manager else ("Arial", 10, "bold")).pack(pady=(0, 5))
                 
        if gui_manager:
            center_up = gui_manager.create_modern_button(
                center_col, text="▲", color=theme_colors["accent_blue"],
                command=lambda: self.adjust_height(-DialogConstants.ADJUSTMENT_STEP_PX)
            )
            center_up.pack(fill=tk.X, pady=2)
            
            center_down = gui_manager.create_modern_button(
                center_col, text="▼", color=theme_colors["accent_blue"],
                command=lambda: self.adjust_height(DialogConstants.ADJUSTMENT_STEP_PX)
            )
            center_down.pack(fill=tk.X, pady=2)
        
        # Right column - Move Right Side
        right_col = ttk.Frame(columns_frame)
        right_col.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        ttk.Label(right_col, text=DialogHelper.t("Move Right Side"),
                 font=gui_manager.fonts["heading"] if gui_manager else ("Arial", 10, "bold")).pack(pady=(0, 5))
                 
        if gui_manager:
            right_up = gui_manager.create_modern_button(
                right_col, text="▲", color=theme_colors["accent_blue"],
                command=lambda: self.adjust_side_height("right", -DialogConstants.ADJUSTMENT_STEP_PX)
            )
            right_up.pack(fill=tk.X, pady=2)
            
            right_down = gui_manager.create_modern_button(
                right_col, text="▼", color=theme_colors["accent_blue"],
                command=lambda: self.adjust_side_height("right", DialogConstants.ADJUSTMENT_STEP_PX)
            )
            right_down.pack(fill=tk.X, pady=2)
            
        # Separator
        ttk.Separator(frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Per-compartment adjustment controls
        self.compartment_frame = ttk.Frame(frame, style='Content.TFrame')
        self.compartment_frame.pack(fill=tk.X)
        
        ttk.Label(
            self.compartment_frame,
            text=DialogHelper.t("Selected Compartment Adjustment"),
            font=gui_manager.fonts["heading"] if gui_manager else ("Arial", 10, "bold")
        ).pack(pady=(0, 5))
        
        # Two columns for selected compartment
        comp_cols_frame = ttk.Frame(self.compartment_frame, style='Content.TFrame')
        comp_cols_frame.pack(fill=tk.X)
        
        # Top adjustment
        top_col = ttk.Frame(comp_cols_frame)
        top_col.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=10)
        
        ttk.Label(top_col, text=DialogHelper.t("Adjust Top"),
                 font=gui_manager.fonts["small"] if gui_manager else ("Arial", 9)).pack()
                 
        if gui_manager:
            self.comp_top_up = gui_manager.create_modern_button(
                top_col, text="▲", color=theme_colors["accent_yellow"],
                command=lambda: self.adjust_compartment_boundary("top", -DialogConstants.ADJUSTMENT_STEP_PX)
            )
            self.comp_top_up.pack(fill=tk.X, pady=2)
            self.comp_top_up.configure(state='disabled')
            
            self.comp_top_down = gui_manager.create_modern_button(
                top_col, text="▼", color=theme_colors["accent_yellow"],
                command=lambda: self.adjust_compartment_boundary("top", DialogConstants.ADJUSTMENT_STEP_PX)
            )
            self.comp_top_down.pack(fill=tk.X, pady=2)
            self.comp_top_down.configure(state='disabled')
        
        # Bottom adjustment
        bottom_col = ttk.Frame(comp_cols_frame)
        bottom_col.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=10)
        
        ttk.Label(bottom_col, text=DialogHelper.t("Adjust Bottom"),
                 font=gui_manager.fonts["small"] if gui_manager else ("Arial", 9)).pack()
                 
        if gui_manager:
            self.comp_bottom_up = gui_manager.create_modern_button(
                bottom_col, text="▲", color=theme_colors["accent_yellow"],
                command=lambda: self.adjust_compartment_boundary("bottom", -DialogConstants.ADJUSTMENT_STEP_PX)
            )
            self.comp_bottom_up.pack(fill=tk.X, pady=2)
            self.comp_bottom_up.configure(state='disabled')
            
            self.comp_bottom_down = gui_manager.create_modern_button(
                bottom_col, text="▼", color=theme_colors["accent_yellow"],
                command=lambda: self.adjust_compartment_boundary("bottom", DialogConstants.ADJUSTMENT_STEP_PX)
            )
            self.comp_bottom_down.pack(fill=tk.X, pady=2)
            self.comp_bottom_down.configure(state='disabled')
        
        # Initially hide compartment controls
        self.compartment_frame.pack_forget()
        
        return frame
        
    def set_compartments(self, detected_boundaries: List, manual_boundaries: List,
                        interpolated_boundaries: Optional[List] = None):
        """Set all compartments for adjustment (detected, manual, and interpolated)."""
        self.all_compartments = []
        
        # Add detected boundaries
        for i, boundary in enumerate(detected_boundaries):
            if isinstance(boundary, (list, tuple)) and len(boundary) >= 4:
                x1, y1, x2, y2 = boundary[:4]
                self.all_compartments.append(CompartmentBoundary(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    marker_id=-1,  # Detected boundaries don't have marker IDs
                    compartment_number=i + 1,
                    center_x=(x1 + x2) // 2,
                    is_manual=False,
                    is_interpolated=False
                ))
                
        # Add manual boundaries
        for boundary in manual_boundaries:
            if isinstance(boundary, dict) and all(k in boundary for k in ['x1', 'y1', 'x2', 'y2']):
                self.all_compartments.append(CompartmentBoundary(
                    x1=boundary['x1'],
                    y1=boundary['y1'],
                    x2=boundary['x2'],
                    y2=boundary['y2'],
                    marker_id=boundary.get('marker_id', -1),
                    compartment_number=boundary.get('compartment_number', -1),
                    center_x=boundary.get('center_x', (boundary['x1'] + boundary['x2']) // 2),
                    is_manual=True,
                    is_interpolated=False
                ))
                
        # Add interpolated boundaries if provided
        if interpolated_boundaries:
            for boundary in interpolated_boundaries:
                if isinstance(boundary, dict):
                    self.all_compartments.append(CompartmentBoundary(
                        x1=boundary['x1'],
                        y1=boundary['y1'],
                        x2=boundary['x2'],
                        y2=boundary['y2'],
                        marker_id=boundary.get('marker_id', -1),
                        compartment_number=boundary.get('compartment_number', -1),
                        center_x=boundary.get('center_x', (boundary['x1'] + boundary['x2']) // 2),
                        is_manual=False,
                        is_interpolated=True
                    ))
                    
        # Sort compartments by x-position
        self.all_compartments.sort(key=lambda c: c.center_x)
        
        # Initialize adjustments for each compartment
        for i in range(len(self.all_compartments)):
            if i not in self.compartment_adjustments:
                self.compartment_adjustments[i] = {'top_offset': 0, 'bottom_offset': 0}
                
    def handle_compartment_click(self, image_x: int, image_y: int, 
                                working_image_shape: Tuple[int, int]) -> bool:
        """Handle click on compartment for selection."""
        # Apply current boundary adjustments to get actual positions
        img_height, img_width = working_image_shape
        
        # Find clicked compartment
        for i, compartment in enumerate(self.all_compartments):
            # Get adjusted boundaries for this compartment
            x1, y1, x2, y2 = self._get_adjusted_compartment_bounds(i, img_width)
            
            if x1 <= image_x <= x2 and y1 <= image_y <= y2:
                self.selected_compartment_index = i
                self.selected_compartment = compartment
                
                # Enable compartment controls
                self.compartment_frame.pack(fill=tk.X)
                for btn in [self.comp_top_up, self.comp_top_down, 
                          self.comp_bottom_up, self.comp_bottom_down]:
                    if hasattr(btn, 'configure'):
                        btn.configure(state='normal')
                
                # Update selection label
                comp_type = "manual" if compartment.is_manual else \
                           "interpolated" if compartment.is_interpolated else "detected"
                self.selection_label.config(
                    text=DialogHelper.t(
                        "Selected: Compartment %(num)s (%(type)s)",
                        num=compartment.compartment_number,
                        type=comp_type
                    )
                )
                
                return True
                
        # No compartment clicked - deselect
        self.selected_compartment_index = None
        self.selected_compartment = None
        self.compartment_frame.pack_forget()
        self.selection_label.config(
            text=DialogHelper.t("Click a compartment to select it for individual adjustment")
        )
        
        return False
        
    def _get_adjusted_compartment_bounds(self, index: int, img_width: int) -> Tuple[int, int, int, int]:
        """Get adjusted bounds for a compartment including global and per-compartment adjustments."""
        compartment = self.all_compartments[index]
        
        # Start with original bounds
        x1, x2 = compartment.x1, compartment.x2
        
        # Calculate global adjustments at this x-position
        left_top_y = self.boundary_state.top_y + self.boundary_state.left_height_offset
        right_top_y = self.boundary_state.top_y + self.boundary_state.right_height_offset
        left_bottom_y = self.boundary_state.bottom_y + self.boundary_state.left_height_offset
        right_bottom_y = self.boundary_state.bottom_y + self.boundary_state.right_height_offset
        
        # Calculate slopes
        if img_width > 0:
            top_slope = (right_top_y - left_top_y) / img_width
            bottom_slope = (right_bottom_y - left_bottom_y) / img_width
            
            # Calculate y values at this x-position
            mid_x = compartment.center_x
            y1 = int(left_top_y + (top_slope * mid_x))
            y2 = int(left_bottom_y + (bottom_slope * mid_x))
        else:
            y1 = self.boundary_state.top_y
            y2 = self.boundary_state.bottom_y
            
        # Apply per-compartment adjustments
        if index in self.compartment_adjustments:
            adj = self.compartment_adjustments[index]
            y1 += adj['top_offset']
            y2 += adj['bottom_offset']
            
        return x1, y1, x2, y2
        
    def adjust_height(self, delta: int):
        """Adjust overall height for all compartments."""
        img_height = 800  # Default, should be passed from working image
        distance = self.boundary_state.bottom_y - self.boundary_state.top_y
        
        new_top = max(0, min(img_height - distance - 1, self.boundary_state.top_y + delta))
        new_bottom = new_top + distance
        
        if new_bottom >= img_height:
            new_bottom = img_height - 1
            new_top = new_bottom - distance
            
        self.boundary_state.top_y = new_top
        self.boundary_state.bottom_y = new_bottom
        
        self._notify_adjustment()
        
    def adjust_side_height(self, side: str, delta: int):
        """Adjust left or right side height."""
        if side == "left":
            self.boundary_state.left_height_offset += delta
        elif side == "right":
            self.boundary_state.right_height_offset += delta
            
        self._notify_adjustment()
        
    def adjust_compartment_boundary(self, boundary: str, delta: int):
        """Adjust top or bottom boundary of selected compartment."""
        if self.selected_compartment_index is None:
            return
            
        if boundary == "top":
            self.compartment_adjustments[self.selected_compartment_index]['top_offset'] += delta
        elif boundary == "bottom":
            self.compartment_adjustments[self.selected_compartment_index]['bottom_offset'] += delta
            
        self._notify_adjustment()
        
    def draw_selection_overlay(self, viz_state: VisualizationState):
        """Draw selection highlight for selected compartment."""
        if self.selected_compartment_index is None:
            return
            
        # Get adjusted bounds
        img_width = 800  # Should be passed from working image
        x1, y1, x2, y2 = self._get_adjusted_compartment_bounds(
            self.selected_compartment_index, img_width
        )
        
        # Convert to canvas coordinates
        x1_canvas = x1 * viz_state.scale_ratio + viz_state.canvas_offset_x
        y1_canvas = y1 * viz_state.scale_ratio + viz_state.canvas_offset_y
        x2_canvas = x2 * viz_state.scale_ratio + viz_state.canvas_offset_x
        y2_canvas = y2 * viz_state.scale_ratio + viz_state.canvas_offset_y
        
        # Draw selection highlight
        self.canvas.create_rectangle(
            x1_canvas - 2, y1_canvas - 2,
            x2_canvas + 2, y2_canvas + 2,
            outline="#00FF00", width=3,
            tags=("selection", "dynamic"),
            dash=(5, 5)
        )
        
        # Add handles at corners
        handle_size = 6
        for (cx, cy) in [(x1_canvas, y1_canvas), (x2_canvas, y1_canvas),
                        (x1_canvas, y2_canvas), (x2_canvas, y2_canvas)]:
            self.canvas.create_rectangle(
                cx - handle_size, cy - handle_size,
                cx + handle_size, cy + handle_size,
                fill="#00FF00", outline="white", width=2,
                tags=("selection", "dynamic")
            )
            
    def get_adjusted_boundaries(self) -> List[Tuple[int, int, int, int]]:
        """Get all compartment boundaries with adjustments applied."""
        adjusted = []
        img_width = 800  # Should be passed from working image
        
        for i in range(len(self.all_compartments)):
            x1, y1, x2, y2 = self._get_adjusted_compartment_bounds(i, img_width)
            adjusted.append((x1, y1, x2, y2))
            
        return adjusted
        
    def reset_adjustments(self):
        """Reset all adjustments to zero."""
        self.boundary_state.left_height_offset = 0
        self.boundary_state.right_height_offset = 0
        self.compartment_adjustments.clear()
        
        for i in range(len(self.all_compartments)):
            self.compartment_adjustments[i] = {'top_offset': 0, 'bottom_offset': 0}
            
        self._notify_adjustment()
        
    def _notify_adjustment(self):
        """Notify callback with debouncing."""
        current_time = time.time()
        if (self.on_adjustment and 
            current_time - self._last_adjustment_time > DialogConstants.DEBOUNCE_INTERVAL_S):
            self._last_adjustment_time = current_time
            
            # Include per-compartment adjustments in callback
            adjustment_data = {
                'boundary_state': self.boundary_state,
                'compartment_adjustments': self.compartment_adjustments,
                'all_boundaries': self.get_adjusted_boundaries()
            }
            self.on_adjustment(adjustment_data)








class ZoomLens:
    """Manages zoom lens visualization for both hover and static modes."""
    
    def __init__(self, parent: tk.Widget, theme_colors: Dict[str, str], 
                 gui_manager: Optional[GUIManager] = None):
        self.parent = parent
        self.theme_colors = theme_colors
        self.gui_manager = gui_manager
        self.logger = logger
        
        # Hover zoom windows
        self.hover_zoom = None
        self.hover_zoom_canvas = None
        self.hover_zoom_flipped = None
        self.hover_zoom_canvas_flipped = None
        
        # Static zoom windows for adjustment mode
        self.left_zoom_window = None
        self.right_zoom_window = None
        self.left_zoom_canvas = None
        self.right_zoom_canvas = None
        self.left_zoom_photo = None
        self.right_zoom_photo = None
        
        # State
        self.static_zooms_visible = False
        
        # Create hover zoom windows
        self._create_hover_zooms()
        
    def _create_hover_zooms(self):
        """Create the hover zoom windows (normal and flipped)."""
        # Normal hover zoom
        self.hover_zoom = tk.Toplevel(self.parent)
        self.hover_zoom.withdraw()
        self.hover_zoom.overrideredirect(True)
        self.hover_zoom.attributes('-topmost', True)
        
        self.hover_zoom_canvas = tk.Canvas(
            self.hover_zoom,
            width=DialogConstants.ZOOM_WIDTH,
            height=DialogConstants.ZOOM_HEIGHT,
            bg=self.theme_colors["background"],
            highlightthickness=1,
            highlightbackground=self.theme_colors["field_border"]
        )
        self.hover_zoom_canvas.pack()
        
        # Add crosshairs
        self._add_crosshairs(self.hover_zoom_canvas, DialogConstants.ZOOM_WIDTH, 
                           DialogConstants.ZOOM_HEIGHT)
        
        # Flipped hover zoom
        self.hover_zoom_flipped = tk.Toplevel(self.parent)
        self.hover_zoom_flipped.withdraw()
        self.hover_zoom_flipped.overrideredirect(True)
        self.hover_zoom_flipped.attributes('-topmost', True)
        
        self.hover_zoom_canvas_flipped = tk.Canvas(
            self.hover_zoom_flipped,
            width=DialogConstants.ZOOM_WIDTH,
            height=DialogConstants.ZOOM_HEIGHT,
            bg=self.theme_colors["background"],
            highlightthickness=1,
            highlightbackground=self.theme_colors["field_border"]
        )
        self.hover_zoom_canvas_flipped.pack()
        
        # Add crosshairs
        self._add_crosshairs(self.hover_zoom_canvas_flipped, DialogConstants.ZOOM_WIDTH,
                           DialogConstants.ZOOM_HEIGHT)
        
    def _add_crosshairs(self, canvas: tk.Canvas, width: int, height: int):
        """Add crosshairs to a zoom canvas."""
        center_x = width // 2
        center_y = height // 2
        
        canvas.create_line(
            0, center_y, width, center_y,
            fill='red', width=1, tags='crosshair'
        )
        canvas.create_line(
            center_x, 0, center_x, height,
            fill='red', width=1, tags='crosshair'
        )
        canvas.create_oval(
            center_x - 3, center_y - 3,
            center_x + 3, center_y + 3,
            fill='red', outline='red', tags='crosshair'
        )
        
    def create_static_zoom_windows(self):
        """Create static zoom windows for adjustment mode."""
        # Left zoom window
        self.left_zoom_window = tk.Toplevel(self.parent)
        self.left_zoom_window.withdraw()
        self.left_zoom_window.overrideredirect(True)
        self.left_zoom_window.attributes('-topmost', True)
        
        self.left_zoom_canvas = tk.Canvas(
            self.left_zoom_window,
            width=DialogConstants.STATIC_ZOOM_WIDTH,
            height=DialogConstants.STATIC_ZOOM_HEIGHT,
            bg=self.theme_colors["background"],
            highlightthickness=1,
            highlightbackground=self.theme_colors["field_border"]
        )
        self.left_zoom_canvas.pack()
        
        # Right zoom window
        self.right_zoom_window = tk.Toplevel(self.parent)
        self.right_zoom_window.withdraw()
        self.right_zoom_window.overrideredirect(True)
        self.right_zoom_window.attributes('-topmost', True)
        
        self.right_zoom_canvas = tk.Canvas(
            self.right_zoom_window,
            width=DialogConstants.STATIC_ZOOM_WIDTH,
            height=DialogConstants.STATIC_ZOOM_HEIGHT,
            bg=self.theme_colors["background"],
            highlightthickness=1,
            highlightbackground=self.theme_colors["field_border"]
        )
        self.right_zoom_canvas.pack()
        
    def show_hover_zoom(self, screen_x: int, screen_y: int, 
                       image_region: np.ndarray, flipped: bool = False):
        """Show hover zoom at screen position with given image region."""
        try:
            # Convert to RGB
            if len(image_region.shape) == 2:
                region_rgb = cv2.cvtColor(image_region, cv2.COLOR_GRAY2RGB)
            else:
                region_rgb = cv2.cvtColor(image_region, cv2.COLOR_BGR2RGB)
                
            # Convert to PIL
            pil_region = Image.fromarray(region_rgb)
            
            if flipped:
                pil_region = ImageOps.flip(pil_region)
                
            # Resize
            zoomed = pil_region.resize(
                (DialogConstants.ZOOM_WIDTH, DialogConstants.ZOOM_HEIGHT), 
                Image.LANCZOS
            )
            
            # Display
            if flipped:
                self._tk_image_flipped = ImageTk.PhotoImage(zoomed)
                self.hover_zoom_canvas_flipped.delete("image")
                self.hover_zoom_canvas_flipped.create_image(
                    0, 0, anchor=tk.NW, image=self._tk_image_flipped, tags="image"
                )
                
                # Position and show
                self.hover_zoom_flipped.geometry(
                    f"{DialogConstants.ZOOM_WIDTH}x{DialogConstants.ZOOM_HEIGHT}+"
                    f"{screen_x}+{screen_y}"
                )
                self.hover_zoom_flipped.deiconify()
            else:
                self._tk_image = ImageTk.PhotoImage(zoomed)
                self.hover_zoom_canvas.delete("image")
                self.hover_zoom_canvas.create_image(
                    0, 0, anchor=tk.NW, image=self._tk_image, tags="image"
                )
                
                # Position and show
                self.hover_zoom.geometry(
                    f"{DialogConstants.ZOOM_WIDTH}x{DialogConstants.ZOOM_HEIGHT}+"
                    f"{screen_x}+{screen_y}"
                )
                self.hover_zoom.deiconify()
            
        except Exception as e:
            self.logger.error(f"Error showing hover zoom: {e}")
            
    def hide_hover_zoom(self):
        """Hide all hover zoom windows."""
        if self.hover_zoom:
            self.hover_zoom.withdraw()
        if self.hover_zoom_flipped:
            self.hover_zoom_flipped.withdraw()
            
    def update_static_zooms(self, working_image: np.ndarray, boundary_state: BoundaryState,
                          corner_markers: Dict, markers: Dict, detected_boundaries: List,
                          canvas_widget: tk.Canvas):
        """Update static zoom windows for adjustment mode."""
        if not self.left_zoom_window or not self.right_zoom_window:
            self.create_static_zoom_windows()
            
        if working_image is None:
            return
            
        h, w = working_image.shape[:2]
        
        # Determine zoom positions
        left_x, right_x = self._determine_zoom_positions(w, corner_markers, detected_boundaries, markers)
        
        # Get Y position from boundary centers
        center_y = (boundary_state.top_y + boundary_state.bottom_y) // 2
        
        # Extract regions
        left_region = self._extract_zoom_region(working_image, left_x, center_y, h, w)
        right_region = self._extract_zoom_region(working_image, right_x, center_y, h, w)
        
        if left_region is None or right_region is None:
            return
            
        # Draw boundary lines on regions
        self._draw_boundary_lines_on_region(left_region, boundary_state, left_x, center_y, "left")
        self._draw_boundary_lines_on_region(right_region, boundary_state, right_x, center_y, "right")
        
        # Draw markers if present
        self._draw_markers_on_region(left_region, corner_markers, [0, 3], left_x, center_y)
        self._draw_markers_on_region(right_region, corner_markers, [1, 2], right_x, center_y)
        
        # Convert and display
        self._display_zoom_region(left_region, self.left_zoom_canvas, "left")
        self._display_zoom_region(right_region, self.right_zoom_canvas, "right")
        
        # Position windows
        self._position_static_zoom_windows(canvas_widget)
        
        # Show windows
        self.left_zoom_window.deiconify()
        self.right_zoom_window.deiconify()
        self.static_zooms_visible = True
        
    def _determine_zoom_positions(self, img_width: int, corner_markers: Dict, 
                                 detected_boundaries: List, markers: Dict) -> Tuple[int, int]:
        """Determine left and right zoom positions."""
        left_x = None
        right_x = None
        
        # Try corner markers first
        if corner_markers:
            left_markers = []
            if 0 in corner_markers:
                left_markers.append(np.mean(corner_markers[0][:, 0]))
            if 3 in corner_markers:
                left_markers.append(np.mean(corner_markers[3][:, 0]))
            
            if left_markers:
                left_x = int(np.mean(left_markers))
                
            right_markers = []
            if 1 in corner_markers:
                right_markers.append(np.mean(corner_markers[1][:, 0]))
            if 2 in corner_markers:
                right_markers.append(np.mean(corner_markers[2][:, 0]))
                
            if right_markers:
                right_x = int(np.mean(right_markers))
                
        # Fallback to compartment boundaries
        if left_x is None or right_x is None:
            x_positions = []
            
            if detected_boundaries:
                for x1, _, x2, _ in detected_boundaries:
                    x_positions.extend([x1, x2])
                    
            if markers:
                compartment_ids = DialogConstants.COMPARTMENT_MARKER_IDS
                for marker_id, corners in markers.items():
                    if marker_id in compartment_ids:
                        center_x = np.mean(corners[:, 0])
                        x_positions.append(center_x)
                        
            if x_positions:
                if left_x is None:
                    left_x = int(min(x_positions))
                if right_x is None:
                    right_x = int(max(x_positions))
                    
        # Final fallback
        if left_x is None:
            left_x = img_width // 4
        if right_x is None:
            right_x = (img_width * 3) // 4
            
        return left_x, right_x
        
    def _extract_zoom_region(self, image: np.ndarray, center_x: int, center_y: int,
                           img_h: int, img_w: int) -> Optional[np.ndarray]:
        """Extract a zoom region from the image."""
        size = DialogConstants.STATIC_ZOOM_REGION_SIZE
        
        x1 = max(0, center_x - size)
        y1 = max(0, center_y - size)
        x2 = min(img_w, center_x + size)
        y2 = min(img_h, center_y + size)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        return image[y1:y2, x1:x2].copy()
        
    def _draw_boundary_lines_on_region(self, region: np.ndarray, boundary_state: BoundaryState,
                                      region_center_x: int, region_center_y: int, side: str):
        """Draw boundary lines on zoom region."""
        region_h, region_w = region.shape[:2]
        zoom_size = DialogConstants.STATIC_ZOOM_REGION_SIZE
        
        # Calculate offsets
        if side == "left":
            height_offset = boundary_state.left_height_offset
        else:
            height_offset = boundary_state.right_height_offset
            
        # Calculate boundary positions relative to region
        top_y_in_region = (boundary_state.top_y + height_offset) - (region_center_y - zoom_size)
        bottom_y_in_region = (boundary_state.bottom_y + height_offset) - (region_center_y - zoom_size)
        
        # Draw lines if visible
        if 0 <= top_y_in_region < region_h:
            cv2.line(region, (0, int(top_y_in_region)), (region_w, int(top_y_in_region)),
                    (0, 255, 0), 2)
            
        if 0 <= bottom_y_in_region < region_h:
            cv2.line(region, (0, int(bottom_y_in_region)), (region_w, int(bottom_y_in_region)),
                    (0, 255, 0), 2)
            
    def _draw_markers_on_region(self, region: np.ndarray, corner_markers: Dict,
                               marker_ids: List[int], region_center_x: int, 
                               region_center_y: int):
        """Draw corner markers on zoom region if visible."""
        zoom_size = DialogConstants.STATIC_ZOOM_REGION_SIZE
        
        for marker_id in marker_ids:
            if marker_id in corner_markers:
                marker_x = int(np.mean(corner_markers[marker_id][:, 0]))
                marker_y = int(np.mean(corner_markers[marker_id][:, 1]))
                
                # Convert to region coordinates
                zoom_x = marker_x - (region_center_x - zoom_size)
                zoom_y = marker_y - (region_center_y - zoom_size)
                
                # Draw if visible
                region_h, region_w = region.shape[:2]
                if 0 <= zoom_x < region_w and 0 <= zoom_y < region_h:
                    cv2.circle(region, (zoom_x, zoom_y), 5, (255, 0, 0), -1)
                    cv2.putText(region, str(marker_id), (zoom_x - 10, zoom_y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                              
    def _display_zoom_region(self, region: np.ndarray, canvas: tk.Canvas, side: str):
        """Display zoom region on canvas."""
        try:
            # Convert to RGB
            region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
            pil_region = Image.fromarray(region_rgb)
            
            # Resize to fit canvas
            pil_region = pil_region.resize(
                (DialogConstants.STATIC_ZOOM_WIDTH, DialogConstants.STATIC_ZOOM_HEIGHT),
                Image.LANCZOS
            )
            
            # Create PhotoImage
            if side == "left":
                self.left_zoom_photo = ImageTk.PhotoImage(pil_region)
                photo = self.left_zoom_photo
            else:
                self.right_zoom_photo = ImageTk.PhotoImage(pil_region)
                photo = self.right_zoom_photo
                
            # Display
            canvas.delete("all")
            canvas.create_image(
                DialogConstants.STATIC_ZOOM_WIDTH // 2,
                DialogConstants.STATIC_ZOOM_HEIGHT // 2,
                image=photo
            )
            
            # Add title
            canvas.create_text(
                DialogConstants.STATIC_ZOOM_WIDTH // 2, 15,
                text=DialogHelper.t("Left Side" if side == "left" else "Right Side"),
                fill=self.theme_colors["text"],
                font=self.gui_manager.fonts["heading"]
            )
            
            # Add crosshairs
            self._add_crosshairs(canvas, DialogConstants.STATIC_ZOOM_WIDTH,
                               DialogConstants.STATIC_ZOOM_HEIGHT)
                               
        except Exception as e:
            self.logger.error(f"Error displaying zoom region: {e}")
            
    def _position_static_zoom_windows(self, canvas_widget: tk.Canvas):
        """Position static zoom windows relative to canvas."""
        if not canvas_widget.winfo_exists():
            return
            
        # Get canvas position
        canvas_x = canvas_widget.winfo_rootx()
        canvas_y = canvas_widget.winfo_rooty()
        canvas_width = canvas_widget.winfo_width()
        
        # Position windows above canvas
        zoom_y = canvas_y - DialogConstants.STATIC_ZOOM_HEIGHT - 20
        
        # Left window on left side
        left_zoom_x = canvas_x + 20
        
        # Right window on right side
        right_zoom_x = canvas_x + canvas_width - DialogConstants.STATIC_ZOOM_WIDTH - 20
        
        # Ensure windows stay on screen
        screen_width = self.parent.winfo_screenwidth()
        screen_height = self.parent.winfo_screenheight()
        
        left_zoom_x = max(10, min(screen_width - DialogConstants.STATIC_ZOOM_WIDTH - 10, left_zoom_x))
        right_zoom_x = max(10, min(screen_width - DialogConstants.STATIC_ZOOM_WIDTH - 10, right_zoom_x))
        zoom_y = max(10, min(screen_height - DialogConstants.STATIC_ZOOM_HEIGHT - 10, zoom_y))
        
        # Set positions
        self.left_zoom_window.geometry(
            f"{DialogConstants.STATIC_ZOOM_WIDTH}x{DialogConstants.STATIC_ZOOM_HEIGHT}+"
            f"{left_zoom_x}+{zoom_y}"
        )
        self.right_zoom_window.geometry(
            f"{DialogConstants.STATIC_ZOOM_WIDTH}x{DialogConstants.STATIC_ZOOM_HEIGHT}+"
            f"{right_zoom_x}+{zoom_y}"
        )
        
    def hide_static_zooms(self):
        """Hide static zoom windows."""
        if self.left_zoom_window:
            self.left_zoom_window.withdraw()
        if self.right_zoom_window:
            self.right_zoom_window.withdraw()
        self.static_zooms_visible = False
        
    def cleanup(self):
        """Clean up all zoom windows."""
        # Destroy hover zooms
        if self.hover_zoom:
            self.hover_zoom.destroy()
            self.hover_zoom = None
        if self.hover_zoom_flipped:
            self.hover_zoom_flipped.destroy()
            self.hover_zoom_flipped = None
            
        # Destroy static zooms
        if self.left_zoom_window:
            self.left_zoom_window.destroy()
            self.left_zoom_window = None
        if self.right_zoom_window:
            self.right_zoom_window.destroy()
            self.right_zoom_window = None