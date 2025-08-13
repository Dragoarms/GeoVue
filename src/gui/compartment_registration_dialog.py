# gui/compartment_registration_dialog.py

"""
Unified dialog for complete compartment registration workflow.
Combines metadata input, boundary annotation, and boundary adjustment in one interface.
"""

import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import logging
from PIL import Image, ImageTk, ImageOps
import re
import os
from typing import Dict, List, Optional, Tuple, Any, Union
import threading
import traceback
import math
import time

from gui.dialog_helper import DialogHelper
from gui.boundary_manager import BoundaryManager

# if threading.current_thread() != threading.main_thread():
#     raise RuntimeError("❌ CompartmentRegistrationDialog called from a background thread!")

logger = logging.getLogger(__name__)


class CompartmentRegistrationDialog:
    """
    Unified dialog for complete compartment registration workflow.
    Combines metadata input, boundary annotation, and boundary adjustment in a
    single interface with three distinct modes:

    1. Metadata Registration - Enter hole ID and depth information
    2. Add Missing Boundaries - Annotate missing compartment markers
    3. Adjust Boundaries - Fine-tune boundary positions

    Each mode offers specialized tools and visualization for its specific task.
    """

    # Define mode constants for clarity
    MODE_METADATA = 0
    MODE_MISSING_BOUNDARIES = 1
    MODE_ADJUST_BOUNDARIES = 2

    def __init__(
        self,
        parent,
        image,
        detected_boundaries,
        missing_marker_ids=None,
        theme_colors=None,
        gui_manager=None,
        boundaries_viz=None,
        original_image=None,
        output_format="png",
        file_manager=None,
        metadata=None,
        vertical_constraints=None,
        marker_to_compartment=None,
        rotation_angle=0.0,
        corner_markers=None,
        markers=None,
        config=None,
        on_apply_adjustments=None,
        show_adjustment_controls=True,
        image_path=None,
        scale_data=None,
        boundary_analysis=None,
        app=None,
    ):
        """Initialize the unified compartment registration dialog."""

        # 1. FIRST - Basic initialization
        self.parent = parent
        self.logger = logging.getLogger(__name__)

        # 2. Store all the input parameters
        self.detected_boundaries = detected_boundaries or []
        self.missing_marker_ids = list(missing_marker_ids) if missing_marker_ids else []
        self.theme_colors = theme_colors
        self.gui_manager = gui_manager
        self.output_format = output_format
        self.file_manager = file_manager
        self.metadata = metadata or {}
        self.rotation_angle = rotation_angle
        self.on_apply_adjustments = on_apply_adjustments
        self.adjustment_controls_visible = show_adjustment_controls
        self.image_path = image_path
        self.scale_data = scale_data
        self.corner_markers = corner_markers or {}
        self.markers = markers or {}

        # 3. Initialize configuration EARLY
        self.config = config or {
            "compartment_marker_ids": list(range(4, 24)),
            "corner_marker_ids": [0, 1, 2, 3],
            "metadata_marker_ids": [24],
            "compartment_count": 20,
        }

        # 4. Get compartment interval from metadata
        self.compartment_interval = int(self.metadata.get("compartment_interval", 1))
        self.interval_var = tk.IntVar(value=self.compartment_interval)

        # 5. Initialize marker_to_compartment mapping - FIXED positions
        depth_from = self.metadata.get("depth_from", 0)
        self.marker_to_compartment = marker_to_compartment or {
            marker_id: int(depth_from + ((marker_id - 3) * self.compartment_interval))
            for marker_id in range(4, 24)
        }

        # 6. Store boundary analysis data
        self.boundary_analysis = boundary_analysis
        self.boundary_to_marker = {}
        self.boundary_types = []  # Initialize empty

        if boundary_analysis:
            if "avg_compartment_width" in boundary_analysis:
                self.avg_width = boundary_analysis["avg_compartment_width"]
            if "boundary_to_marker" in boundary_analysis:
                self.boundary_to_marker = boundary_analysis["boundary_to_marker"]
            if "boundary_types" in boundary_analysis:
                self.boundary_types = boundary_analysis["boundary_types"]

        # 7. NOW create BoundaryManager with all dependencies ready
        self.boundary_manager = BoundaryManager(
            marker_to_compartment=self.marker_to_compartment,
            expected_compartment_count=self.config.get("compartment_count", 20),
            compartment_interval=self.compartment_interval,
        )

        # 8. Import existing boundaries if available
        if self.detected_boundaries and self.boundary_to_marker:
            self.boundary_manager.import_boundaries(
                self.detected_boundaries,
                self.boundary_to_marker,
                self.boundary_types if self.boundary_types else None,
            )

        # 9. Clean up missing markers list
        if self.boundary_to_marker:
            already_placed = set(self.boundary_to_marker.values())
            self.missing_marker_ids = [
                mid for mid in self.missing_marker_ids if mid not in already_placed
            ]
            self.logger.debug(
                f"Cleaned missing_marker_ids: removed already placed markers"
            )

        # Filter out corner markers
        corner_ids = self.config.get("corner_marker_ids", [0, 1, 2, 3])
        self.missing_marker_ids = [
            mid for mid in self.missing_marker_ids if mid not in corner_ids
        ]
        self.logger.info(
            f"Filtered corner markers from missing list. Remaining: {self.missing_marker_ids}"
        )

        # 10. Try to get app reference through parent chain
        self.logger.debug("=== APP REFERENCE DEBUG ===")
        self.logger.debug(f"app parameter received: {app}")
        self.logger.debug(
            f"kwargs keys: {list(kwargs.keys()) if 'kwargs' in locals() else 'N/A'}"
        )
        self.logger.debug(f"parent type: {type(parent)}")
        self.logger.debug(f"parent has app: {hasattr(parent, 'app')}")

        self.app = app  # Use provided app if given
        self.logger.debug(f"self.app after assignment: {self.app}")

        if not self.app:
            self.logger.debug("App not provided directly, checking parent chain...")

            # Check parent.master.app
            if hasattr(parent, "master"):
                self.logger.debug(f"parent has master: {type(parent.master)}")
                if hasattr(parent.master, "app"):
                    self.app = parent.master.app
                    self.logger.debug(f"Got app from parent.master.app: {self.app}")

            # Check parent.app
            elif hasattr(parent, "app"):
                self.app = parent.app
                self.logger.debug(f"Got app from parent.app: {self.app}")

            # Check toplevel
            elif hasattr(parent, "winfo_toplevel"):
                toplevel = parent.winfo_toplevel()
                self.logger.debug(f"toplevel type: {type(toplevel)}")
                self.logger.debug(f"toplevel has app: {hasattr(toplevel, 'app')}")
                if hasattr(toplevel, "app"):
                    self.app = toplevel.app
                    self.logger.debug(f"Got app from toplevel.app: {self.app}")

        if not self.app:
            self.logger.warning(
                "Could not get app reference - wall detection and depth validation will be disabled"
            )
        else:
            self.logger.debug(f"App reference obtained: {self.app}")
            self.logger.debug(
                f"App has depth_validator: {hasattr(self.app, 'depth_validator')}"
            )
            if hasattr(self.app, "depth_validator"):
                dv = self.app.depth_validator
                self.logger.debug(
                    f"Depth validator loaded: {dv.is_loaded if dv else 'None'}"
                )
                if dv and dv.is_loaded:
                    self.logger.debug(f"Depth validator CSV path: {dv.csv_path}")
                    self.logger.debug(f"Number of holes loaded: {len(dv.depth_ranges)}")

        self.logger.debug("=== END APP REFERENCE DEBUG ===")

        if self.app:
            self.app.boundary_manager = self.boundary_manager

        # 11. Image management with clear naming
        # Handle input parameters to determine our source image
        if image is not None:
            self.source_image = image.copy()
        elif boundaries_viz is not None:
            self.source_image = boundaries_viz.copy()
        else:
            self.source_image = None
            self.logger.error("No image provided to CompartmentRegistrationDialog")

        # Initialize display image as a copy of source
        self.display_image = (
            self.source_image.copy() if self.source_image is not None else None
        )

        # Store high-res image if provided (for extraction)
        self.high_res_image = (
            original_image.copy() if original_image is not None else None
        )

        # 12. Initialize UI state variables
        self.current_mode = self.MODE_METADATA  # Start in metadata mode
        self.current_index = 0
        self.result_boundaries = {}
        self.annotation_complete = False
        self.temp_point = None
        self.show_walls = False

        # 13. Initialize UI string variables
        self.hole_id = tk.StringVar(value=self.metadata.get("hole_id", ""))
        self.depth_from = tk.StringVar(value=str(self.metadata.get("depth_from", "")))
        self.depth_to = tk.StringVar(value=str(self.metadata.get("depth_to", "")))

        # Also update increment button state when interval changes
        self.interval_var.trace_add(
            "write", lambda *args: self._update_increment_button_state()
        )

        # 14. Initialize canvas and visualization parameters
        self.scale_ratio = 1.0
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
        self.static_viz_cache = None
        self.static_viz_params = None

        # 15. Initialize adjustment parameters
        self.left_height_offset = 0
        self.right_height_offset = 0
        self._last_apply_time = 0
        self._apply_debounce_interval = 1.0

        # Adjustment mode flags
        self.adjusting_top = False
        self.adjusting_bottom = False
        self.adjusting_left_side = False
        self.adjusting_right_side = False

        # 16. Handle vertical constraints
        if vertical_constraints:
            self.top_y, self.bottom_y = vertical_constraints
        else:
            # Fallback to sensible defaults if not provided
            h = self.source_image.shape[0] if self.source_image is not None else 800
            self.top_y = int(h * 0.1)
            self.bottom_y = int(h * 0.9)
            self.logger.warning(
                f"No vertical constraints provided, using defaults: "
                f"top_y={self.top_y}, bottom_y={self.bottom_y}"
            )

        # 17. Calculate average compartment width if not from boundary analysis
        if not hasattr(self, "avg_width"):
            self._calculate_average_compartment_width()

        # 18. Handle theme colors
        if self.gui_manager and hasattr(self.gui_manager, "theme_colors"):
            self.theme_colors = self.gui_manager.theme_colors
        else:
            # Fallback theme colors if gui_manager is unavailable
            self.theme_colors = self.theme_colors or {
                "background": "#1e1e1e",
                "text": "#e0e0e0",
                "field_bg": "#2d2d2d",
                "field_border": "#3f3f3f",
                "accent_green": "#4CAF50",
                "accent_blue": "#2196F3",
                "accent_red": "#F44336",
                "accent_yellow": "#FFEB3B",
                "hover_highlight": "#3a3a3a",
            }

        # 19. Debug logging
        self.logger.debug("=== Dialog Initialization Debug ===")
        self.logger.debug(
            f"Received image shape: {self.source_image.shape if self.source_image is not None else 'None'}"
        )
        self.logger.debug(f"Received boundaries: {len(self.detected_boundaries)}")
        self.logger.debug(f"Missing markers: {self.missing_marker_ids}")
        self.logger.debug(f"Metadata: {self.metadata}")
        self.logger.debug(f"Vertical constraints: ({self.top_y}, {self.bottom_y})")
        self.logger.debug(f"Scale data provided: {self.scale_data is not None}")
        self.logger.debug(
            f"Boundary analysis provided: {self.boundary_analysis is not None}"
        )
        self.logger.debug(
            f"Boundary manager initialized with {len(self.boundary_manager.boundaries)} boundaries"
        )

        # 20. Create the dialog UI
        self.dialog = self._create_dialog()

        # Apply gui_manager ttk styles
        if self.gui_manager:
            self.gui_manager.configure_ttk_styles(self.dialog)

        # Register trace on depth variables for auto-updating compartment labels
        self.dialog.after_idle(self._register_traces)

        # Bind resizing event to dialog
        self.dialog.bind("<Configure>", self._on_dialog_resize)

        # Setup the dialog content
        self._create_widgets()

        # Use after_idle to ensure canvas is properly sized
        self.dialog.after_idle(self._initial_visualization_update)

        # Initialize zoom lens after creating widgets
        self._init_zoom_lens()

        # Create visualization with existing boundaries
        self._update_visualization()

        # Update mode display
        self._update_mode_indicator()

    def _register_traces(self):
        """Register variable traces after GUI is initialized - enhanced version."""
        # Original traces
        self.hole_id.trace_add("write", self._validate_hole_id_realtime)
        self.depth_from.trace_add("write", self._validate_depth_from_realtime)
        self.depth_to.trace_add("write", self._validate_depth_to_realtime)
        self.interval_var.trace_add("write", self._update_expected_depth)
        self.interval_var.trace_add(
            "write", lambda *args: self._update_increment_button_state()
        )

        # Add traces for updating the display
        self.hole_id.trace_add("write", lambda *args: self._update_metadata_display())
        self.hole_id.trace_add(
            "write", lambda *args: self._update_existing_data_visualization()
        )
        self.depth_from.trace_add(
            "write", lambda *args: self._update_metadata_display()
        )
        self.depth_to.trace_add("write", lambda *args: self._update_metadata_display())

        # Force initial update
        self._update_compartment_labels()
        self._update_metadata_display()
        self._update_existing_data_visualization()

    def _calculate_average_compartment_width(self):
        """Calculate the average width of detected compartments."""
        if not self.detected_boundaries:
            # Default value if no boundaries are detected
            w = 0
            if self.source_image is not None:
                w = self.source_image.shape[1]
            else:
                w = 1000  # Fallback default width
            self.avg_width = int(w * 0.04)  # 4% of image width as default
            return

        # Calculate width of each detected compartment
        widths = [x2 - x1 for x1, _, x2, _ in self.detected_boundaries]

        if widths:
            self.avg_width = int(sum(widths) / len(widths))
        else:
            w = self.source_image.shape[1] if self.source_image is not None else 1000
            self.avg_width = int(w * 0.04)  # 4% of image width as default

    def _would_overlap_existing(self, x_pos):
        """
        Check if a compartment placed at the given x position would overlap with existing compartments.
        Checks against both manually annotated boundaries and automatically detected ones.

        Args:
            x_pos: X-coordinate for potential compartment placement

        Returns:
            bool: True if overlap would occur, False otherwise
        """
        try:
            # Calculate the boundaries for the potential new compartment
            half_width = self.avg_width // 2
            new_x1 = max(0, x_pos - half_width)
            new_x2 = min(self.source_image.shape[1] - 1, x_pos + half_width)

            # Check against all manually annotated compartments (excluding metadata marker)
            for comp_id, boundary in self.result_boundaries.items():
                if comp_id == 24:  # Skip metadata marker
                    continue

                # Ensure boundary is a tuple with 4 values (compartment)
                if isinstance(boundary, tuple) and len(boundary) == 4:
                    x1, y1, x2, y2 = boundary
                    # Check for horizontal overlap
                    if new_x1 < x2 and new_x2 > x1:
                        return True

            # Also check against automatically detected compartments
            for x1, y1, x2, y2 in self.detected_boundaries:
                # Check for horizontal overlap
                if new_x1 < x2 and new_x2 > x1:
                    return True

            # No overlap found
            return False
        except Exception as e:
            self.logger.error(f"Error checking for overlap: {str(e)}")
            return False

    def _bind_window_tracking(self):
        """Bind events to track dialog movement and update zoom positions."""
        # Track when dialog is moved or resized
        self.dialog.bind("<Configure>", self._on_dialog_configure)

        # Store last known dialog position
        self._last_dialog_x = self.dialog.winfo_x()
        self._last_dialog_y = self.dialog.winfo_y()

    def _on_dialog_configure(self, event):
        """Handle dialog movement/resize to update zoom positions."""
        if event.widget != self.dialog:
            return

        # Get current dialog position
        current_x = self.dialog.winfo_x()
        current_y = self.dialog.winfo_y()

        # Check if dialog moved
        if current_x != self._last_dialog_x or current_y != self._last_dialog_y:
            # Update zoom window positions if they're visible
            if (
                hasattr(self, "_zoom_lens")
                and self._zoom_lens
                and self._zoom_lens.winfo_viewable()
            ):
                # Force re-render at current mouse position
                self.canvas.event_generate("<Motion>")

            # Update static zoom windows in adjustment mode
            if self.current_mode == self.MODE_ADJUST_BOUNDARIES:
                if hasattr(self, "left_zoom_window") and self.left_zoom_visible:
                    self._update_static_zoom_views()

            # Update last known position
            self._last_dialog_x = current_x
            self._last_dialog_y = current_y

    def _validate_hole_id_realtime(self, *args):
        """Real-time validation for hole ID as user types."""
        hole_id = self.hole_id.get().upper()  # Auto-uppercase

        # Remove any invalid characters and limit length
        cleaned = ""
        for i, char in enumerate(hole_id):
            if i < 2 and char.isalpha():  # First 2 chars must be letters
                cleaned += char
            elif i >= 2 and i < 6 and char.isdigit():  # Next 4 must be digits
                cleaned += char

        # Update if cleaned
        if cleaned != hole_id:
            self.hole_id.set(cleaned)

        # Visual feedback via entry widget styling
        if hasattr(self, "hole_id_entry"):
            if len(cleaned) == 6 and re.match(r"^[A-Z]{2}\d{4}$", cleaned):
                # Valid complete format
                self.hole_id_entry.config(
                    highlightbackground=self.theme_colors.get(
                        "accent_green", "#4a8259"
                    ),
                    highlightthickness=2,
                )
            elif len(cleaned) < 6:
                # Incomplete but valid so far
                self.hole_id_entry.config(
                    highlightbackground=self.theme_colors.get("field_border"),
                    highlightthickness=1,
                )
            else:
                # Invalid format
                self.hole_id_entry.config(
                    highlightbackground=self.theme_colors.get("accent_red", "#9e4a4a"),
                    highlightthickness=2,
                )

    def _validate_depth_from_realtime(self, *args):
        """Real-time validation for depth_from and auto-update depth_to."""
        depth_str = self.depth_from.get().strip()

        # Remove non-digits
        cleaned = "".join(c for c in depth_str if c.isdigit())

        if cleaned != depth_str:
            self.depth_from.set(cleaned)

        # Auto-update depth_to if we have a valid number
        if cleaned and cleaned.isdigit():
            depth_from = int(cleaned)
            interval = self.interval_var.get()
            depth_to = depth_from + (20 * interval)
            self.depth_to.set(str(depth_to))

            # Update compartment labels
            self._update_compartment_labels()

        # Visual feedback
        if hasattr(self, "depth_from_entry"):
            if cleaned and int(cleaned) >= 0:
                self.depth_from_entry.config(
                    highlightbackground=self.theme_colors.get(
                        "accent_green", "#4a8259"
                    ),
                    highlightthickness=1,
                )
            else:
                self.depth_from_entry.config(
                    highlightbackground=self.theme_colors.get("field_border"),
                    highlightthickness=1,
                )

    def _validate_depth_to_realtime(self, *args):
        """Real-time validation for depth_to."""
        depth_str = self.depth_to.get().strip()

        # Remove non-digits
        cleaned = "".join(c for c in depth_str if c.isdigit())

        if cleaned != depth_str:
            self.depth_to.set(cleaned)

        # Check validity against depth_from
        if hasattr(self, "depth_to_entry"):
            depth_from_str = self.depth_from.get().strip()
            if cleaned and depth_from_str and depth_from_str.isdigit():
                depth_from = int(depth_from_str)
                depth_to = int(cleaned)
                interval = self.interval_var.get()
                expected_range = 20 * interval
                actual_range = depth_to - depth_from

                if depth_to > depth_from and actual_range == expected_range:
                    # Perfect match
                    self.depth_to_entry.config(
                        highlightbackground=self.theme_colors.get(
                            "accent_green", "#4a8259"
                        ),
                        highlightthickness=2,
                    )
                elif depth_to > depth_from:
                    # Valid but non-standard range
                    self.depth_to_entry.config(
                        highlightbackground=self.theme_colors.get(
                            "accent_yellow", "#e5c07b"
                        ),
                        highlightthickness=2,
                    )
                else:
                    # Invalid
                    self.depth_to_entry.config(
                        highlightbackground=self.theme_colors.get(
                            "accent_red", "#9e4a4a"
                        ),
                        highlightthickness=2,
                    )
            else:
                self.depth_to_entry.config(
                    highlightbackground=self.theme_colors.get("field_border"),
                    highlightthickness=1,
                )

    def _create_dialog(self):
        """Create the dialog window with proper styling and parent relationship."""
        try:
            # Use DialogHelper to create a properly configured dialog
            # create_dialog only takes: parent, title, modal, topmost
            dialog = DialogHelper.create_dialog(
                parent=self.parent,
                title=DialogHelper.t("Compartment Registration"),
                modal=False,
                topmost=False,
            )

            # Apply theme colors from gui_manager
            dialog.configure(bg=self.theme_colors["background"])
            dialog.state("normal")
            dialog.protocol("WM_DELETE_WINDOW", self._on_cancel)
            return dialog
        except Exception as e:
            self.logger.error(f"Error creating registration dialog: {e}")
            self.logger.error(traceback.format_exc())
            raise  # Let the caller handle fatal GUI failures

    def _create_widgets(self):
        """Create all widgets for the unified dialog."""
        # Main container with padding
        main_frame = ttk.Frame(self.dialog, padding=10, style="Content.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Configure ttk styles if gui_manager is available
        if self.gui_manager:
            self.gui_manager.configure_ttk_styles(self.dialog)
        else:
            # Create basic styles with theme colors
            style = ttk.Style(self.dialog)
            style.configure(
                "Content.TFrame", background=self.theme_colors["background"]
            )
            style.configure(
                "Content.TLabel",
                background=self.theme_colors["background"],
                foreground=self.theme_colors["text"],
            )
            # Define a larger style for instructions
            style.configure(
                "Instructions.TLabel",
                font=("Arial", 18, "bold"),
                background=self.theme_colors["background"],
                foreground=self.theme_colors["text"],
                padding=10,
            )

        # Create mode selector at top
        self._create_mode_selector(main_frame)

        # Initialize status_var BEFORE creating enhanced display
        self.status_var = tk.StringVar(value="")

        # Enhanced status display with metadata and existing data visualization
        self._create_enhanced_status_display(main_frame)

        # Main image canvas frame
        self.canvas_frame = ttk.Frame(main_frame, style="Content.TFrame")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Calculate canvas dimensions based on image and screen size
        canvas_width = 800  # Default minimum
        canvas_height = 600  # Default minimum

        if hasattr(self, "source_image") and self.source_image is not None:
            # Get image dimensions
            img_height, img_width = self.source_image.shape[:2]

            # Get available screen space
            screen_width, screen_height = self._get_monitor_size()

            # Calculate vertical UI space (buttons, status, metadata panels, etc.)
            vertical_ui_space = 300  # Approximate space for UI elements

            # Calculate available space with margins
            available_width = screen_width - 50  # Reduced margin from 100 to 50
            available_height = screen_height - vertical_ui_space - 50  # Reduced margin

            # Calculate scale factor to fit image
            scale_width = available_width / img_width
            scale_height = available_height / img_height
            scale = min(scale_width, scale_height, 1.5)  # Don't upscale beyond 150%

            # Calculate scaled dimensions as canvas size
            canvas_width = max(int(img_width * scale), 800)
            canvas_height = max(int(img_height * scale), 400)

            self.logger.debug(
                f"Canvas sizing - Image: {img_width}x{img_height}, "
                f"Scale: {scale:.2f}, Canvas: {canvas_width}x{canvas_height}"
            )

        # Create canvas with calculated size
        self.canvas = tk.Canvas(
            self.canvas_frame,
            width=canvas_width,
            height=canvas_height,
            bg=self.theme_colors["background"],
            highlightthickness=1,
            highlightbackground=self.theme_colors["field_border"],
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas_frame.pack_propagate(
            False
        )  # prevent the frame from resizing to the (shrinking) canvas height

        # Bind events
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Motion>", self._on_canvas_move)
        self.canvas.bind("<Leave>", self._on_canvas_leave)
        self.canvas.bind("<Button-3>", self._on_canvas_right_click)  # Right click
        self.canvas.bind("<B3-Motion>", self._on_canvas_move)  # Right button drag

        # Create bottom container for metadata and adjustment controls

        self.bottom_container = ttk.Frame(main_frame, style="Content.TFrame")
        self.bottom_container.pack(fill=tk.X, pady=(5, 0))

        # Create metadata panel in bottom container (will be shown/hidden based on mode)
        self.metadata_frame = ttk.Frame(
            self.bottom_container, style="Content.TFrame", padding=10
        )
        # Initially pack to the right side if in metadata mode
        if self.current_mode == self.MODE_METADATA:
            self.metadata_frame.pack(side=tk.RIGHT, anchor=tk.SE, padx=(10, 0))
        self._create_metadata_panel(self.metadata_frame)

        # Create boundary adjustment controls in bottom container (will be shown/hidden based on mode)
        self.adjustment_frame = ttk.Frame(self.bottom_container, style="Content.TFrame")
        self._create_adjustment_controls(self.adjustment_frame)

        # Only display adjustment frame if in adjustment mode
        if (
            self.current_mode == self.MODE_ADJUST_BOUNDARIES
            and self.adjustment_controls_visible
        ):
            self.adjustment_frame.pack(
                side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10)
            )

        # Button row
        button_frame = ttk.Frame(main_frame, style="Content.TFrame")
        button_frame.pack(fill=tk.X, pady=(10, 5))

        # Create buttons using gui_manager or fallback to ttk buttons
        if self.gui_manager:

            # Quit button - leftmost position
            self.quit_button = self.gui_manager.create_modern_button(
                button_frame,
                text=DialogHelper.t("Quit"),
                color=self.theme_colors["accent_red"],
                command=self._on_quit,
            )
            self.quit_button.pack(side=tk.LEFT, padx=5)

            # Reject button
            self.reject_button = self.gui_manager.create_modern_button(
                button_frame,
                text=DialogHelper.t("Reject"),
                color=self.theme_colors["accent_red"],
                command=self._on_reject,
            )
            self.reject_button.pack(side=tk.RIGHT, padx=5)

            # Cancel button
            self.cancel_button = self.gui_manager.create_modern_button(
                button_frame,
                text=DialogHelper.t("Cancel"),
                color=self.theme_colors["accent_blue"],
                command=self._on_cancel,
            )
            self.cancel_button.pack(side=tk.RIGHT, padx=5)

            # Continue button - text will be updated based on mode
            self.continue_button = self.gui_manager.create_modern_button(
                button_frame,
                text=self._get_continue_button_text(),
                color=self.theme_colors["accent_green"],
                command=self._on_continue,
            )
            self.continue_button.pack(side=tk.RIGHT, padx=5)

            # Undo button
            self.undo_button = self.gui_manager.create_modern_button(
                button_frame,
                text=DialogHelper.t("Undo Last"),
                color=self.theme_colors["accent_blue"],
                command=self._undo_last,
            )
            self.undo_button.pack(side=tk.RIGHT, padx=5)

            # Interpolate missing boundaries button
            self.interpolate_button = self.gui_manager.create_modern_button(
                button_frame,
                text=DialogHelper.t("Interpolate"),
                color=self.theme_colors["accent_yellow"],
                command=self._on_interpolate_clicked,
            )
            self.interpolate_button.pack(side=tk.RIGHT, padx=5)
            self.interpolate_button.pack_forget()  # Initially hidden

            # Show Walls toggle
            self.toggle_walls_btn = self.gui_manager.create_modern_button(
                button_frame,
                text="Show Walls",
                color=self.theme_colors["accent_blue"],
                command=self._on_toggle_walls,
            )
            self.toggle_walls_btn.pack(side=tk.LEFT, padx=5)

        else:
            # Use ttk buttons as fallback with better styling
            style = ttk.Style(self.dialog)
            style.configure("TButton", font=("Arial", 12))

            self.quit_button = ttk.Button(
                button_frame,
                text=DialogHelper.t("Quit"),
                command=self._on_quit,
                style="TButton",
                padding=10,
            )
            self.quit_button.pack(side=tk.LEFT, padx=5, pady=5)

            self.reject_button = ttk.Button(
                button_frame,
                text=DialogHelper.t("Reject"),
                command=self._on_reject,
                style="TButton",
                padding=10,
            )
            self.reject_button.pack(side=tk.RIGHT, padx=5, pady=5)

            self.cancel_button = ttk.Button(
                button_frame,
                text=DialogHelper.t("Cancel"),
                command=self._on_cancel,
                style="TButton",
                padding=10,
            )
            self.cancel_button.pack(side=tk.RIGHT, padx=5, pady=5)

            self.continue_button = ttk.Button(
                button_frame,
                text=self._get_continue_button_text(),
                command=self._on_continue,
                style="TButton",
                padding=10,
            )
            self.continue_button.pack(side=tk.RIGHT, padx=5, pady=5)

            self.undo_button = ttk.Button(
                button_frame,
                text=DialogHelper.t("Undo Last"),
                command=self._undo_last,
                style="TButton",
                padding=10,
            )
            self.undo_button.pack(side=tk.RIGHT, padx=5, pady=5)

            self.interpolate_button = ttk.Button(
                button_frame,
                text=DialogHelper.t("Interpolate"),
                command=self._on_interpolate_clicked,
                style="TButton",
                padding=10,
            )
            self.interpolate_button.pack(side=tk.RIGHT, padx=5, pady=5)
            self.interpolate_button.pack_forget()

            # Show Walls toggle (fallback)
            self.toggle_walls_btn = ttk.Button(
                button_frame,
                text="Show Walls",
                command=self._on_toggle_walls,
                style="TButton",
                padding=5,
            )
            self.toggle_walls_btn.pack(side=tk.LEFT, padx=5)

        # Update the status message based on the current mode
        self._update_status_message()

        # Bind window tracking for zoom lens positioning
        self._bind_window_tracking()

    def _create_mode_selector(self, parent_frame):
        """Create mode selector tabs at the top of the dialog."""
        # Create frame for mode buttons
        mode_frame = ttk.Frame(parent_frame, style="Content.TFrame")
        mode_frame.pack(fill=tk.X, pady=(0, 10))

        # Button width
        btn_width = 15

        # Define mode button styles for fallbacks
        active_style = {
            "background": self.theme_colors.get("accent_green", "#4CAF50"),
            "foreground": "white",
            "font": ("Arial", 12, "bold"),
            "relief": tk.RAISED,
            "borderwidth": 2,
            "padx": 10,
            "pady": 5,
        }

        inactive_style = {
            "background": self.theme_colors.get("field_bg", "#2d2d2d"),
            "foreground": self.theme_colors.get("text", "#e0e0e0"),
            "font": ("Arial", 12),
            "relief": tk.FLAT,
            "borderwidth": 1,
            "padx": 10,
            "pady": 5,
        }

        # Create mode buttons
        mode_btn_container = ttk.Frame(mode_frame, style="Content.TFrame")
        mode_btn_container.pack(anchor=tk.CENTER)

        # Method to create a mode button
        def create_mode_button(text, mode, position, container):
            if self.gui_manager:
                # Use gui_manager to create modern button
                button = self.gui_manager.create_modern_button(
                    container,
                    text=DialogHelper.t(text),
                    color=(
                        self.theme_colors["accent_green"]
                        if self.current_mode == mode
                        else self.theme_colors["field_bg"]
                    ),
                    command=lambda: self._switch_mode(mode),
                )
                button.grid(row=0, column=position, padx=5, pady=5)
                return button
            else:
                # Use standard Tkinter button
                button = tk.Button(
                    container,
                    text=DialogHelper.t(text),
                    width=btn_width,
                    command=lambda: self._switch_mode(mode),
                    cursor="hand2",
                )
                # Apply styles based on active/inactive state
                if self.current_mode == mode:
                    for k, v in active_style.items():
                        button.config(**{k: v})
                else:
                    for k, v in inactive_style.items():
                        button.config(**{k: v})

                button.grid(row=0, column=position, padx=5, pady=5)
                return button

        # Create the three mode buttons
        self.metadata_button = create_mode_button(
            "Metadata Registration", self.MODE_METADATA, 0, mode_btn_container
        )

        self.boundaries_button = create_mode_button(
            "Add Missing Boundaries",
            self.MODE_MISSING_BOUNDARIES,
            1,
            mode_btn_container,
        )

        self.adjust_button = create_mode_button(
            "Adjust Boundaries", self.MODE_ADJUST_BOUNDARIES, 2, mode_btn_container
        )

        # Store buttons for later style updates
        self.mode_buttons = {
            self.MODE_METADATA: self.metadata_button,
            self.MODE_MISSING_BOUNDARIES: self.boundaries_button,
            self.MODE_ADJUST_BOUNDARIES: self.adjust_button,
        }

    def _create_metadata_panel(self, parent_frame):
        """Create metadata input panel with hole ID and depth fields - single row version."""
        # Main container for metadata input
        fields_frame = ttk.Frame(parent_frame, style="Content.TFrame", padding=5)
        fields_frame.pack(fill=tk.X, padx=5, pady=5)

        # fonts for entries
        label_font = self.gui_manager.fonts["label"]
        entry_font = self.gui_manager.fonts["entry"]

        # Create a single row frame
        row_frame = ttk.Frame(fields_frame, style="Content.TFrame")
        row_frame.pack(fill=tk.X)

        # --- Hole ID ---
        hole_id_label = ttk.Label(
            row_frame,
            text=DialogHelper.t("Hole ID:"),
            font=label_font,
            style="Content.TLabel",
        )
        hole_id_label.pack(side=tk.LEFT, padx=(0, 5))

        # Use themed entry with validation
        self.hole_id_entry = self.gui_manager.create_entry_with_validation(
            row_frame,
            textvariable=self.hole_id,
            validate_func=self._validate_hole_id_realtime,
            width=8,
            placeholder="AB1234",
        )
        self.hole_id_entry.pack(side=tk.LEFT, padx=(0, 15))

        # --- Depth Range ---
        depth_label = ttk.Label(
            row_frame,
            text=DialogHelper.t("Depth:"),
            font=label_font,
            style="Content.TLabel",
        )
        depth_label.pack(side=tk.LEFT, padx=(0, 5))

        # Use themed entry for depth_from with validation
        self.depth_from_entry = self.gui_manager.create_entry_with_validation(
            row_frame,
            textvariable=self.depth_from,
            validate_func=self._validate_depth_from_realtime,
            width=5,
            placeholder="0",
        )
        self.depth_from_entry.pack(side=tk.LEFT)

        depth_separator = ttk.Label(
            row_frame,
            text=DialogHelper.t("-"),
            font=entry_font,
            style="Content.TLabel",
        )
        depth_separator.pack(side=tk.LEFT, padx=3)

        # Use themed entry for depth_to with validation
        self.depth_to_entry = self.gui_manager.create_entry_with_validation(
            row_frame,
            textvariable=self.depth_to,
            validate_func=self._validate_depth_to_realtime,
            width=5,
            placeholder="20",
        )
        self.depth_to_entry.pack(side=tk.LEFT, padx=(0, 15))

        # --- Interval ---
        interval_label = ttk.Label(
            row_frame,
            text=DialogHelper.t("Interval:"),
            font=label_font,
            style="Content.TLabel",
        )
        interval_label.pack(side=tk.LEFT, padx=(0, 5))

        # Create a frame for the dropdown with theme colors
        interval_combo_frame = tk.Frame(
            row_frame,
            bg=self.theme_colors["field_bg"],
            highlightbackground=self.theme_colors["field_border"],
            highlightthickness=1,
        )
        interval_combo_frame.pack(side=tk.LEFT, padx=(0, 5))

        # Create the dropdown options
        interval_options = [1, 2]

        # Create the OptionMenu widget
        interval_dropdown = tk.OptionMenu(
            interval_combo_frame,
            self.interval_var,
            *interval_options,
            command=self._update_expected_depth,
        )

        # Style the dropdown using gui_manager
        self.gui_manager.style_dropdown(interval_dropdown, width=2)
        interval_dropdown.pack()

        interval_help = ttk.Label(
            row_frame,
            text=DialogHelper.t("m"),
            font=label_font,
            foreground="gray",
            style="Content.TLabel",
        )
        interval_help.pack(side=tk.LEFT, padx=(0, 20))

        # --- Increment button ---
        self.increment_button = self.gui_manager.create_modern_button(
            row_frame,
            text=DialogHelper.t("Increment From Last"),
            color=self.theme_colors["accent_blue"],
            command=self._on_increment_from_last,
        )
        self.increment_button.pack(side=tk.LEFT, padx=5)

        # Set initial button state
        self._update_increment_button_state()

        # Trigger initial validation to set visual states
        self._validate_hole_id_realtime()
        self._validate_depth_from_realtime()
        self._validate_depth_to_realtime()

    def _create_adjustment_controls(self, frame):
        """Create boundary adjustment controls with improved layout."""
        # Boundary adjustment frame with title
        title_label = ttk.Label(
            frame,
            text=DialogHelper.t("Boundary Adjustment"),
            style="Content.TLabel",
            font=("Arial", 12, "bold"),
        )
        title_label.pack(anchor="w", pady=(5, 10))

        # Create a container for three columns
        columns_frame = ttk.Frame(frame, style="Content.TFrame", padding=5)
        columns_frame.pack(fill=tk.X)

        # Create three equal columns
        left_column = ttk.Frame(columns_frame, style="Content.TFrame")
        left_column.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        center_column = ttk.Frame(columns_frame, style="Content.TFrame")
        center_column.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        right_column = ttk.Frame(columns_frame, style="Content.TFrame")
        right_column.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        # LEFT COLUMN - Move Left Side
        ttk.Label(
            left_column,
            text=DialogHelper.t("Move Left Side"),
            style="Content.TLabel",
            font=("Arial", 10, "bold"),
            anchor="center",
        ).pack(fill=tk.X, pady=(0, 5))

        # Up button for left side
        if self.gui_manager:
            left_up_button = self.gui_manager.create_modern_button(
                left_column,
                text="▲",
                color=self.theme_colors["accent_blue"],
                command=lambda: self._adjust_side_height("left", -5),
            )
            left_up_button.pack(fill=tk.X, expand=True, pady=2)

            # Down button for left side
            left_down_button = self.gui_manager.create_modern_button(
                left_column,
                text="▼",
                color=self.theme_colors["accent_blue"],
                command=lambda: self._adjust_side_height("left", 5),
            )
            left_down_button.pack(fill=tk.X, expand=True, pady=2)
        else:
            left_up_button = ttk.Button(
                left_column,
                text="▲",
                command=lambda: self._adjust_side_height("left", -5),
            )
            left_up_button.pack(fill=tk.X, expand=True, pady=2)

            left_down_button = ttk.Button(
                left_column,
                text="▼",
                command=lambda: self._adjust_side_height("left", 5),
            )
            left_down_button.pack(fill=tk.X, expand=True, pady=2)

        # CENTER COLUMN - Move All
        ttk.Label(
            center_column,
            text=DialogHelper.t("Move All"),
            style="Content.TLabel",
            font=("Arial", 10, "bold"),
            anchor="center",
        ).pack(fill=tk.X, pady=(0, 5))

        # Up button for all
        if self.gui_manager:
            center_up_button = self.gui_manager.create_modern_button(
                center_column,
                text="▲",
                color=self.theme_colors["accent_blue"],
                command=lambda: self._adjust_height(-5),
            )
            center_up_button.pack(fill=tk.X, expand=True, pady=2)

            # Down button for all
            center_down_button = self.gui_manager.create_modern_button(
                center_column,
                text="▼",
                color=self.theme_colors["accent_blue"],
                command=lambda: self._adjust_height(5),
            )
            center_down_button.pack(fill=tk.X, expand=True, pady=2)
        else:
            center_up_button = ttk.Button(
                center_column, text="▲", command=lambda: self._adjust_height(-5)
            )
            center_up_button.pack(fill=tk.X, expand=True, pady=2)

            center_down_button = ttk.Button(
                center_column, text="▼", command=lambda: self._adjust_height(5)
            )
            center_down_button.pack(fill=tk.X, expand=True, pady=2)

        # RIGHT COLUMN - Move Right Side
        ttk.Label(
            right_column,
            text=DialogHelper.t("Move Right Side"),
            style="Content.TLabel",
            font=("Arial", 10, "bold"),
            anchor="center",
        ).pack(fill=tk.X, pady=(0, 5))

        # Up button for right side
        if self.gui_manager:
            right_up_button = self.gui_manager.create_modern_button(
                right_column,
                text="▲",
                color=self.theme_colors["accent_blue"],
                command=lambda: self._adjust_side_height("right", -5),
            )
            right_up_button.pack(fill=tk.X, expand=True, pady=2)

            # Down button for right side
            right_down_button = self.gui_manager.create_modern_button(
                right_column,
                text="▼",
                color=self.theme_colors["accent_blue"],
                command=lambda: self._adjust_side_height("right", 5),
            )
            right_down_button.pack(fill=tk.X, expand=True, pady=2)
        else:
            right_up_button = ttk.Button(
                right_column,
                text="▲",
                command=lambda: self._adjust_side_height("right", -5),
            )
            right_up_button.pack(fill=tk.X, expand=True, pady=2)

            right_down_button = ttk.Button(
                right_column,
                text="▼",
                command=lambda: self._adjust_side_height("right", 5),
            )
            right_down_button.pack(fill=tk.X, expand=True, pady=2)

        # We're removing the static zoom frame and will use popup windows instead
        # Store references to track left/right zoom window states
        self.left_zoom_visible = False
        self.right_zoom_visible = False

    def _init_zoom_lens(self):
        """Initialize zoom lens windows for different modes."""
        try:
            # Only create zoom lens if dialog exists and is visible
            if (
                not hasattr(self, "dialog")
                or not self.dialog
                or not self.dialog.winfo_exists()
            ):
                self.logger.warning(
                    "Cannot create zoom lens - dialog doesn't exist or isn't visible"
                )
                return

            # Create a toplevel window for the zoom lens - ensure it's a child of the dialog
            self._zoom_lens = tk.Toplevel(self.dialog)
            self._zoom_lens.withdraw()  # Initially hidden
            self._zoom_lens.overrideredirect(True)  # No window decorations
            self._zoom_lens.attributes("-topmost", True)

            # Set up the zoom canvas for hovering
            zoom_width = 250
            zoom_height = 350
            self._zoom_canvas = tk.Canvas(
                self._zoom_lens,
                width=zoom_width,
                height=zoom_height,
                bg=self.theme_colors["background"],
                highlightthickness=1,
                highlightbackground=self.theme_colors["field_border"],
            )
            self._zoom_canvas.pack()

            # Create the flipped zoom lens with the same parent dialog
            self._zoom_lens_flipped = tk.Toplevel(self.dialog)
            self._zoom_lens_flipped.withdraw()
            self._zoom_lens_flipped.overrideredirect(True)
            self._zoom_lens_flipped.attributes("-topmost", True)

            # Set up the flipped zoom canvas - twice as wide as tall
            zoom_width_flipped = zoom_height * 2  # Make it twice as wide
            self._zoom_canvas_flipped = tk.Canvas(
                self._zoom_lens_flipped,
                width=zoom_width_flipped,
                height=zoom_height,
                bg=self.theme_colors["background"],
                highlightthickness=1,
                highlightbackground=self.theme_colors["field_border"],
            )
            self._zoom_canvas_flipped.pack()

            # Add crosshairs to both canvases
            center_x = zoom_width // 2
            center_y = zoom_height // 2

            # Add crosshair to normal lens
            self._zoom_canvas.create_line(
                0, center_y, zoom_width, center_y, fill="red", width=1, tags="crosshair"
            )
            self._zoom_canvas.create_line(
                center_x,
                0,
                center_x,
                zoom_height,
                fill="red",
                width=1,
                tags="crosshair",
            )
            self._zoom_canvas.create_oval(
                center_x - 3,
                center_y - 3,
                center_x + 3,
                center_y + 3,
                fill="red",
                outline="red",
                tags="crosshair",
            )

            # Add crosshair to flipped lens
            center_x_flipped = zoom_width_flipped // 2
            self._zoom_canvas_flipped.create_line(
                0,
                center_y,
                zoom_width_flipped,
                center_y,
                fill="red",
                width=1,
                tags="crosshair",
            )
            self._zoom_canvas_flipped.create_line(
                center_x_flipped,
                0,
                center_x_flipped,
                zoom_height,
                fill="red",
                width=1,
                tags="crosshair",
            )
            self._zoom_canvas_flipped.create_oval(
                center_x_flipped - 3,
                center_y - 3,
                center_x_flipped + 3,
                center_y + 3,
                fill="red",
                outline="red",
                tags="crosshair",
            )

            # Store dimensions for later use
            self._zoom_width = zoom_width
            self._zoom_height = zoom_height
            self._zoom_width_flipped = zoom_width_flipped

        except Exception as e:
            self.logger.error(f"Error initializing zoom lens: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Proceed without zoom lens functionality - don't let this stop the dialog

    def _bind_adjustment_keys(self):
        """Enable WASD for quick boundary adjustments (only in adjust mode)."""
        PIXEL_STEP = 5

        self.logger.info("Binding WASD keys for adjustment mode")

        # Bind each key with a little debug around it
        def bind_key(seq, fn):
            def _handler(event):
                fn()
                # Prevent event from propagating
                return "break"

            self.dialog.bind_all(seq, _handler)

        # bind keyboard for adjustments
        bind_key("<Down>", lambda: self._adjust_height(-PIXEL_STEP))
        bind_key("<Up>", lambda: self._adjust_height(PIXEL_STEP))
        bind_key("<Left>", lambda: self._adjust_side_height("left", -PIXEL_STEP))
        bind_key("<Right>", lambda: self._adjust_side_height("right", -PIXEL_STEP))

        # Make this dialog modal so it always gets key events
        self.dialog.grab_set()

        # Ensure the dialog (or the canvas) actually has focus
        # Try multiple approaches to ensure we get focus
        self.dialog.after(100, lambda: self.dialog.focus_force())
        self.canvas.focus_set()

    def _unbind_adjustment_keys(self):
        """Remove WASD bindings when leaving adjust mode."""
        # Unbind both lowercase and uppercase
        for seq in ("<w>", "<s>", "<a>", "<d>", "<W>", "<S>", "<A>", "<D>"):
            self.dialog.unbind_all(seq)
        self.dialog.grab_release()

    def _on_dialog_resize(self, event=None):
        """Handle dialog resize events to optimize canvas usage."""
        try:
            # Only process if the dialog is the source of the event
            if event and event.widget != self.dialog:
                return

            # Get current dialog size
            dialog_width = self.dialog.winfo_width()
            dialog_height = self.dialog.winfo_height()

            # Skip if dialog hasn't been properly sized yet
            if dialog_width <= 1 or dialog_height <= 1:
                return

            # Update the visualization to fit the new size
            if hasattr(self, "current_viz") and self.current_viz is not None:
                self._display_visualization(self.current_viz)

        except Exception as e:
            self.logger.debug(f"Error handling resize: {str(e)}")

    def _update_mode_indicator(self):
        """Update the mode buttons to highlight current mode."""
        # Define styles for active and inactive buttons (for fallback)
        active_style = {
            "background": self.theme_colors.get("accent_green", "#4CAF50"),
            "foreground": "white",
            "font": ("Arial", 12, "bold"),
            "relief": tk.RAISED,
            "borderwidth": 2,
        }

        inactive_style = {
            "background": self.theme_colors.get("field_bg", "#2d2d2d"),
            "foreground": self.theme_colors.get("text", "#e0e0e0"),
            "font": ("Arial", 12),
            "relief": tk.FLAT,
            "borderwidth": 1,
        }

        # Update button styles based on current mode
        for mode, button in self.mode_buttons.items():
            if hasattr(button, "configure") and callable(getattr(button, "configure")):
                # For ModernButton, use configure method to update appearance
                if mode == self.current_mode:
                    # Active button - green background
                    button.configure(background=self.theme_colors["accent_green"])

                    # Update base_color and reset for ModernButton
                    if hasattr(button, "base_color"):
                        button.base_color = self.theme_colors["accent_green"]
                        button._reset_color()

                    # Update text weight if possible
                    if hasattr(button, "label") and button.label:
                        current_text = button.label.cget("text")
                        button.label.configure(
                            font=("Arial", 12, "bold"), foreground="white"
                        )
                else:
                    # Inactive button - field background
                    button.configure(background=self.theme_colors["field_bg"])

                    # Update base_color and reset for ModernButton
                    if hasattr(button, "base_color"):
                        button.base_color = self.theme_colors["field_bg"]
                        button._reset_color()

                    # Update text weight if possible
                    if hasattr(button, "label") and button.label:
                        current_text = button.label.cget("text")
                        button.label.configure(
                            font=("Arial", 12), foreground=self.theme_colors["text"]
                        )
            elif isinstance(button, tk.Button):
                # For standard Tkinter buttons
                if mode == self.current_mode:
                    for k, v in active_style.items():
                        button.config(**{k: v})
                else:
                    for k, v in inactive_style.items():
                        button.config(**{k: v})
        # Update interface based on current mode
        self._update_interface_for_mode()

        # Bind or unbind WASD keys based on mode
        if self.current_mode == self.MODE_ADJUST_BOUNDARIES:
            self._bind_adjustment_keys()
        else:
            self._unbind_adjustment_keys()

    def _update_interface_for_mode(self):
        """Update interface elements based on the current mode."""

        # Handle metadata frame visibility based on mode
        if self.current_mode == self.MODE_METADATA:
            # Show metadata only in metadata mode
            self.metadata_frame.pack(side=tk.RIGHT, anchor=tk.SE, padx=(10, 0))

            # Hide adjustment controls in metadata mode
            if self.adjustment_frame.winfo_ismapped():
                self.adjustment_frame.pack_forget()

            # Hide zoom windows if visible
            self._hide_zoom_windows()

        elif self.current_mode == self.MODE_MISSING_BOUNDARIES:
            # Hide metadata in missing boundaries mode
            if self.metadata_frame.winfo_ismapped():
                self.metadata_frame.pack_forget()

            # Hide adjustment controls
            if self.adjustment_frame.winfo_ismapped():
                self.adjustment_frame.pack_forget()

            # Hide zoom windows if visible
            self._hide_zoom_windows()

        elif self.current_mode == self.MODE_ADJUST_BOUNDARIES:
            # Hide metadata in adjustment mode
            if self.metadata_frame.winfo_ismapped():
                self.metadata_frame.pack_forget()

            # Show adjustment controls if enabled
            if (
                self.adjustment_controls_visible
                and not self.adjustment_frame.winfo_ismapped()
            ):
                self.adjustment_frame.pack(
                    side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10)
                )

            # Update and show zoom windows
            self._update_static_zoom_views()

            # Ensure we have focus for keyboard input
            self.dialog.after(50, lambda: self.canvas.focus_set())

        # Update continue button text based on current mode
        continue_text = self._get_continue_button_text()
        self.continue_button.set_text(continue_text)

        # Update status message for this mode
        self._update_status_message()

        # Update visualization to match the current mode
        self._update_visualization()

    def _draw_boundary_lines_on_region(
        self,
        region,
        region_x_start,
        region_y_start,
        left_top_y,
        right_top_y,
        left_bottom_y,
        right_bottom_y,
        img_width,
    ):
        """Draw boundary lines on a zoom region."""
        region_x_end = region_x_start + region.shape[1]

        # Calculate Y positions at the edges of this region
        top_y_at_start = left_top_y + (
            (right_top_y - left_top_y) * region_x_start / img_width
        )
        top_y_at_end = left_top_y + (
            (right_top_y - left_top_y) * region_x_end / img_width
        )
        bottom_y_at_start = left_bottom_y + (
            (right_bottom_y - left_bottom_y) * region_x_start / img_width
        )
        bottom_y_at_end = left_bottom_y + (
            (right_bottom_y - left_bottom_y) * region_x_end / img_width
        )

        # Convert to region coordinates
        top_y_start_local = int(top_y_at_start - region_y_start)
        top_y_end_local = int(top_y_at_end - region_y_start)
        bottom_y_start_local = int(bottom_y_at_start - region_y_start)
        bottom_y_end_local = int(bottom_y_at_end - region_y_start)

        # Draw top boundary line if visible in region
        if (
            0 <= top_y_start_local < region.shape[0]
            or 0 <= top_y_end_local < region.shape[0]
        ):
            cv2.line(
                region,
                (0, top_y_start_local),
                (region.shape[1] - 1, top_y_end_local),
                (0, 255, 0),
                2,
            )

        # Draw bottom boundary line if visible in region
        if (
            0 <= bottom_y_start_local < region.shape[0]
            or 0 <= bottom_y_end_local < region.shape[0]
        ):
            cv2.line(
                region,
                (0, bottom_y_start_local),
                (region.shape[1] - 1, bottom_y_end_local),
                (0, 255, 0),
                2,
            )

    def _draw_compartments_on_regions(
        self,
        left_region,
        right_region,
        left_x_start,
        right_x_start,
        y_start,
        zoom_size,
        left_top_y,
        right_top_y,
        left_bottom_y,
        right_bottom_y,
        img_width,
    ):
        """Draw compartments on zoom regions."""
        # Calculate rotation angle and slopes
        if img_width > 0:
            dx = img_width
            dy = right_top_y - left_top_y
            rotation_angle = -np.arctan2(dy, dx) * 180 / np.pi
            top_slope = (right_top_y - left_top_y) / img_width
            bottom_slope = (right_bottom_y - left_bottom_y) / img_width
        else:
            rotation_angle = 0
            top_slope = 0
            bottom_slope = 0

        # Draw on left region
        self._draw_compartments_on_single_region(
            left_region,
            left_x_start,
            y_start,
            left_top_y,
            left_bottom_y,
            top_slope,
            bottom_slope,
            rotation_angle,
        )

        # Draw on right region
        self._draw_compartments_on_single_region(
            right_region,
            right_x_start,
            y_start,
            left_top_y,
            left_bottom_y,
            top_slope,
            bottom_slope,
            rotation_angle,
        )

    def _draw_compartments_on_single_region(
        self,
        region,
        region_x_start,
        region_y_start,
        left_top_y,
        left_bottom_y,
        top_slope,
        bottom_slope,
        rotation_angle,
    ):
        """Draw compartments on a single zoom region."""
        region_width = region.shape[1]
        region_height = region.shape[0]

        # Draw detected compartments
        for x1, _, x2, _ in self.detected_boundaries:
            # Check if compartment overlaps with this region
            if x2 >= region_x_start and x1 <= region_x_start + region_width:
                # Calculate center position of compartment
                center_x = (x1 + x2) / 2

                # Calculate y positions using slopes
                center_top_y = int(left_top_y + (top_slope * center_x))
                center_bottom_y = int(left_bottom_y + (bottom_slope * center_x))

                # Convert to region coordinates
                x1_local = max(0, x1 - region_x_start)
                x2_local = min(region_width, x2 - region_x_start)
                y1_local = center_top_y - region_y_start
                y2_local = center_bottom_y - region_y_start

                # Draw compartment if visible
                if (
                    x2_local > x1_local
                    and 0 <= y1_local < region_height
                    and 0 <= y2_local < region_height
                ):
                    self._draw_rotated_rectangle(
                        region,
                        x1_local,
                        y1_local,
                        x2_local,
                        y2_local,
                        rotation_angle,
                        (0, 255, 0),
                        2,
                    )

        # Draw manually placed compartments
        for comp_id, boundary in self.result_boundaries.items():
            if comp_id not in [0, 1, 2, 3, 24]:  # Skip corner and metadata markers
                if isinstance(boundary, tuple) and len(boundary) == 4:
                    x1, y1, x2, y2 = boundary

                    # Check if compartment overlaps with this region
                    if x2 >= region_x_start and x1 <= region_x_start + region_width:
                        # Convert to region coordinates
                        x1_local = max(0, x1 - region_x_start)
                        x2_local = min(region_width, x2 - region_x_start)
                        y1_local = y1 - region_y_start
                        y2_local = y2 - region_y_start

                        # Draw compartment if visible
                        if (
                            x2_local > x1_local
                            and 0 <= y1_local < region_height
                            and 0 <= y2_local < region_height
                        ):
                            self._draw_rotated_rectangle(
                                region,
                                x1_local,
                                y1_local,
                                x2_local,
                                y2_local,
                                rotation_angle,
                                (0, 255, 255),
                                2,
                            )

    def _draw_rotated_rectangle(self, image, x1, y1, x2, y2, angle, color, thickness):
        """Draw a rotated rectangle on an image."""
        # Calculate center
        rect_center_x = (x1 + x2) / 2
        rect_center_y = (y1 + y2) / 2

        # Create rectangle corners
        rect_corners = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
        )

        # Apply rotation
        rotation_matrix = cv2.getRotationMatrix2D(
            (rect_center_x, rect_center_y), angle, 1.0
        )
        ones = np.ones(shape=(len(rect_corners), 1))
        rect_corners_homog = np.hstack([rect_corners, ones])
        rotated_corners = np.dot(rotation_matrix, rect_corners_homog.T).T
        rotated_corners = rotated_corners.astype(np.int32)

        # Draw the rectangle
        cv2.polylines(image, [rotated_corners], True, color, thickness)

    def _update_static_zoom_views(self):
        """Update and show popup zoom windows for boundary adjustment mode."""
        if not hasattr(self, "left_zoom_window") or not hasattr(
            self, "right_zoom_window"
        ):
            self._create_zoom_windows()

        # Pick display_image if set, otherwise fallback to source_image
        if hasattr(self, "display_image") and self.display_image is not None:
            base = self.display_image
        else:
            base = self.source_image
        if base is None:
            return

        # Get image dimensions from the displayed (scaled/rotated) image
        h, w = base.shape[:2]
        # Define zoom regions based on corner markers or compartment boundaries
        left_x = None
        right_x = None

        # Try to use corner markers first
        if self.corner_markers:
            # Left side: average of markers 0 and 3
            left_markers = []
            if 0 in self.corner_markers:
                left_markers.append(np.mean(self.corner_markers[0][:, 0]))
            if 3 in self.corner_markers:
                left_markers.append(np.mean(self.corner_markers[3][:, 0]))

            if left_markers:
                left_x = int(np.mean(left_markers))

            # Right side: average of markers 1 and 2
            right_markers = []
            if 1 in self.corner_markers:
                right_markers.append(np.mean(self.corner_markers[1][:, 0]))
            if 2 in self.corner_markers:
                right_markers.append(np.mean(self.corner_markers[2][:, 0]))

            if right_markers:
                right_x = int(np.mean(right_markers))

        # Fallback to compartment boundaries if corner markers not available
        if left_x is None or right_x is None:
            # Get X positions from all compartment markers
            compartment_x_positions = []

            # Check detected boundaries
            if self.detected_boundaries:
                for x1, _, x2, _ in self.detected_boundaries:
                    compartment_x_positions.append(x1)
                    compartment_x_positions.append(x2)

            # Also check markers
            if self.markers:
                compartment_ids = self.config.get(
                    "compartment_marker_ids", list(range(4, 24))
                )
                for marker_id, corners in self.markers.items():
                    if marker_id in compartment_ids:
                        center_x = np.mean(corners[:, 0])
                        compartment_x_positions.append(center_x)

            if compartment_x_positions:
                if left_x is None:
                    left_x = int(min(compartment_x_positions))
                if right_x is None:
                    right_x = int(max(compartment_x_positions))

        # Final fallback to fixed positions
        if left_x is None:
            left_x = w // 4
        if right_x is None:
            right_x = (w * 3) // 4

        # Get Y position from boundary centers
        center_y = (self.top_y + self.bottom_y) // 2

        # Size of zoom region - wider for better view
        zoom_height_half = min(h // 8, 100)
        zoom_width_half = zoom_height_half * 2  # Twice as wide

        try:
            # Extract left region - extends to the right
            left_region = base[
                max(0, center_y - zoom_height_half) : min(
                    h, center_y + zoom_height_half
                ),
                max(0, left_x) : min(w, left_x + zoom_width_half * 2),
            ].copy()

            # Extract right region - extends to the left
            right_region = base[
                max(0, center_y - zoom_height_half) : min(
                    h, center_y + zoom_height_half
                ),
                max(0, right_x - zoom_width_half * 2) : min(w, right_x),
            ].copy()

            # Calculate boundary positions with offsets
            left_top_y = self.top_y + self.left_height_offset
            right_top_y = self.top_y + self.right_height_offset
            left_bottom_y = self.bottom_y + self.left_height_offset
            right_bottom_y = self.bottom_y + self.right_height_offset

            # Draw boundary lines on regions
            self._draw_boundary_lines_on_region(
                left_region,
                left_x,  # Left region starts at left_x
                center_y - zoom_height_half,
                left_top_y,
                right_top_y,
                left_bottom_y,
                right_bottom_y,
                w,
            )
            self._draw_boundary_lines_on_region(
                right_region,
                right_x - zoom_width_half * 2,  # Right region starts before right_x
                center_y - zoom_height_half,
                left_top_y,
                right_top_y,
                left_bottom_y,
                right_bottom_y,
                w,
            )

            # Draw compartments on regions
            self._draw_compartments_on_regions(
                left_region,
                right_region,
                left_x,
                right_x - zoom_width_half * 2,
                center_y - zoom_height_half,
                zoom_height_half,
                left_top_y,
                right_top_y,
                left_bottom_y,
                right_bottom_y,
                w,
            )

            # Convert to RGB
            left_rgb = cv2.cvtColor(left_region, cv2.COLOR_BGR2RGB)
            right_rgb = cv2.cvtColor(right_region, cv2.COLOR_BGR2RGB)

            # Create PIL images
            left_pil = Image.fromarray(left_rgb)
            right_pil = Image.fromarray(right_rgb)

            # Calculate scale to fit in zoom window while maintaining aspect ratio
            region_height, region_width = left_region.shape[:2]
            scale_w = self.zoom_width / region_width
            scale_h = self.zoom_height / region_height
            scale = min(scale_w, scale_h)  # Use smaller scale to fit within bounds

            # Calculate new dimensions maintaining aspect ratio
            new_width = int(region_width * scale)
            new_height = int(region_height * scale)

            # Resize with aspect ratio preserved
            left_pil = left_pil.resize((new_width, new_height), Image.LANCZOS)
            right_pil = right_pil.resize((new_width, new_height), Image.LANCZOS)

            # Convert back to numpy arrays for drawing
            left_np = np.array(left_pil)
            right_np = np.array(right_pil)

            # Draw thinner crosshairs (1 pixel instead of 2)
            cv2.line(
                left_np,
                (0, new_height // 2),
                (new_width, new_height // 2),
                (0, 255, 0),
                1,
            )
            cv2.line(
                left_np,
                (new_width // 2, 0),
                (new_width // 2, new_height),
                (0, 255, 0),
                1,
            )

            cv2.line(
                right_np,
                (0, new_height // 2),
                (new_width, new_height // 2),
                (0, 255, 0),
                1,
            )
            cv2.line(
                right_np,
                (new_width // 2, 0),
                (new_width // 2, new_height),
                (0, 255, 0),
                1,
            )

            # Show corner marker positions on zoom if available
            if self.corner_markers:
                # Left zoom - show markers 0 and 3
                for marker_id in [0, 3]:
                    if marker_id in self.corner_markers:
                        marker_x = int(np.mean(self.corner_markers[marker_id][:, 0]))
                        marker_y = int(np.mean(self.corner_markers[marker_id][:, 1]))

                        # Convert to zoom region coordinates
                        zoom_x = marker_x - (left_x - zoom_size)
                        zoom_y = marker_y - (center_y - zoom_size)

                        # Apply scale factor
                        zoom_x = int(zoom_x * scale)
                        zoom_y = int(zoom_y * scale)

                        if 0 <= zoom_x < new_width and 0 <= zoom_y < new_height:
                            # Smaller circle (3 pixels instead of 5)
                            cv2.circle(left_np, (zoom_x, zoom_y), 3, (255, 0, 0), -1)
                            # Smaller font (0.3 instead of 0.5)
                            cv2.putText(
                                left_np,
                                str(marker_id),
                                (zoom_x - 8, zoom_y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.3,
                                (255, 0, 0),
                                1,
                            )

                # Right zoom - show markers 1 and 2
                for marker_id in [1, 2]:
                    if marker_id in self.corner_markers:
                        marker_x = int(np.mean(self.corner_markers[marker_id][:, 0]))
                        marker_y = int(np.mean(self.corner_markers[marker_id][:, 1]))

                        # Convert to zoom region coordinates
                        zoom_x = marker_x - (right_x - zoom_size)
                        zoom_y = marker_y - (center_y - zoom_size)

                        # Apply scale factor
                        zoom_x = int(zoom_x * scale)
                        zoom_y = int(zoom_y * scale)

                        if 0 <= zoom_x < new_width and 0 <= zoom_y < new_height:
                            # Smaller circle (3 pixels instead of 5)
                            cv2.circle(right_np, (zoom_x, zoom_y), 3, (255, 0, 0), -1)
                            # Smaller font (0.3 instead of 0.5)
                            cv2.putText(
                                right_np,
                                str(marker_id),
                                (zoom_x - 8, zoom_y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.3,
                                (255, 0, 0),
                                1,
                            )

            # Convert back to PIL for display
            left_pil = Image.fromarray(left_np)
            right_pil = Image.fromarray(right_np)

            # Convert hex color to RGB
            hex_color = self.theme_colors["background"].lstrip("#")
            bg_r = int(hex_color[0:2], 16)
            bg_g = int(hex_color[2:4], 16)
            bg_b = int(hex_color[4:6], 16)
            bg_color = (bg_r, bg_g, bg_b)

            left_canvas = Image.new(
                "RGB", (self.zoom_width, self.zoom_height), color=bg_color
            )
            right_canvas = Image.new(
                "RGB", (self.zoom_width, self.zoom_height), color=bg_color
            )

            # Calculate position to center the resized image
            x_offset = (self.zoom_width - new_width) // 2
            y_offset = (self.zoom_height - new_height) // 2

            # Paste the resized images onto the canvases
            left_canvas.paste(left_pil, (x_offset, y_offset))
            right_canvas.paste(right_pil, (x_offset, y_offset))

            # Create PhotoImage objects
            self.left_zoom_photo = ImageTk.PhotoImage(left_canvas)
            self.right_zoom_photo = ImageTk.PhotoImage(right_canvas)

            # Update canvases
            self.left_zoom_canvas.delete("all")
            self.right_zoom_canvas.delete("all")

            self.left_zoom_canvas.create_image(
                self.zoom_width // 2, self.zoom_height // 2, image=self.left_zoom_photo
            )
            self.right_zoom_canvas.create_image(
                self.zoom_width // 2, self.zoom_height // 2, image=self.right_zoom_photo
            )

            # Position and show the zoom windows
            # Get the screen containing the dialog
            dialog_x = self.dialog.winfo_x()
            dialog_y = self.dialog.winfo_y()

            # Get current screen bounds (the screen the dialog is on)
            screen_x = self.dialog.winfo_vrootx()
            screen_y = self.dialog.winfo_vrooty()
            screen_width = self.dialog.winfo_vrootwidth()
            screen_height = self.dialog.winfo_vrootheight()

            # Position zoom windows relative to dialog position
            canvas_x = self.canvas.winfo_rootx()
            canvas_y = self.canvas.winfo_rooty()

            # Convert image coordinates to canvas coordinates
            left_canvas_x = int(left_x * self.scale_ratio + self.canvas_offset_x)
            right_canvas_x = int(right_x * self.scale_ratio + self.canvas_offset_x)
            center_canvas_y = int(center_y * self.scale_ratio + self.canvas_offset_y)

            # Calculate absolute positions
            left_zoom_x = canvas_x + left_canvas_x
            right_zoom_x = canvas_x + right_canvas_x - self.zoom_width

            # Center vertically on the extracted area
            zoom_y = canvas_y + center_canvas_y - (self.zoom_height // 2)

            # Ensure zoom windows stay on the same screen as dialog
            margin = 10
            left_zoom_x = max(
                screen_x + margin,
                min(screen_x + screen_width - self.zoom_width - margin, left_zoom_x),
            )
            right_zoom_x = max(
                screen_x + margin,
                min(screen_x + screen_width - self.zoom_width - margin, right_zoom_x),
            )
            zoom_y = max(
                screen_y + margin,
                min(screen_y + screen_height - self.zoom_height - margin, zoom_y),
            )

            # Set positions and show the windows
            self.left_zoom_window.geometry(
                f"{self.zoom_width}x{self.zoom_height}+{left_zoom_x}+{zoom_y}"
            )
            self.right_zoom_window.geometry(
                f"{self.zoom_width}x{self.zoom_height}+{right_zoom_x}+{zoom_y}"
            )

            # Show windows
            self.left_zoom_window.deiconify()
            self.right_zoom_window.deiconify()
            self.left_zoom_visible = True
            self.right_zoom_visible = True

        except Exception as e:
            self.logger.error(f"Error updating zoom views: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _create_zoom_windows(self):
        """Create popup zoom windows for left and right boundary edges."""
        # Create left zoom window
        self.left_zoom_window = tk.Toplevel(self.dialog)
        self.left_zoom_window.withdraw()  # Initially hidden
        self.left_zoom_window.overrideredirect(True)  # No window decorations
        self.left_zoom_window.attributes("-topmost", True)

        # Set up left zoom canvas - make it twice as wide
        zoom_width, zoom_height = 500, 300
        self.left_zoom_canvas = tk.Canvas(
            self.left_zoom_window,
            width=zoom_width,
            height=zoom_height,
            bg=self.theme_colors["background"],
            highlightthickness=1,
            highlightbackground=self.theme_colors["field_border"],
        )
        self.left_zoom_canvas.pack()

        # Create right zoom window
        self.right_zoom_window = tk.Toplevel(self.dialog)
        self.right_zoom_window.withdraw()  # Initially hidden
        self.right_zoom_window.overrideredirect(True)  # No window decorations
        self.right_zoom_window.attributes("-topmost", True)

        # Set up right zoom canvas
        self.right_zoom_canvas = tk.Canvas(
            self.right_zoom_window,
            width=zoom_width,
            height=zoom_height,
            bg=self.theme_colors["background"],
            highlightthickness=1,
            highlightbackground=self.theme_colors["field_border"],
        )
        self.right_zoom_canvas.pack()

        # Store dimensions
        self.zoom_width = zoom_width
        self.zoom_height = zoom_height

        # Initialize photo references
        self.left_zoom_photo = None
        self.right_zoom_photo = None

    def _switch_mode(self, new_mode):
        """
        Switch between dialog modes.

        Args:
            new_mode: New mode to switch to (MODE_METADATA, MODE_MISSING_BOUNDARIES, MODE_ADJUST_BOUNDARIES)
        """
        self.logger.debug(f"_switch_mode called: {self.current_mode} → {new_mode}")
        # Don't switch if current mode is the same
        if self.current_mode == new_mode:
            return
        # Show or hide the interpolate button based on mode
        if hasattr(self, "interpolate_button"):
            if new_mode == self.MODE_MISSING_BOUNDARIES:
                self.interpolate_button.pack(side=tk.RIGHT, padx=5)
            else:
                self.interpolate_button.pack_forget()
        # Check for validation in metadata mode before switching
        if self.current_mode == self.MODE_METADATA and not self._validate_metadata():
            # Don't switch if metadata validation fails
            return

        # Special handling when entering boundary annotation mode
        if new_mode == self.MODE_MISSING_BOUNDARIES:
            # Reset annotation index
            self.current_index = 0
            self.annotation_complete = False

            # Apply any metadata changes
            self._update_compartment_labels()

        # Store current mode
        self.current_mode = new_mode
        self.logger.debug(f"Mode switched to: {self.current_mode}")

        # Update mode indicator and buttons
        self._update_mode_indicator()

        # Force full update to recreate static visualization for new mode
        self._update_visualization(force_full_update=True)

        # Update continue button text based on new mode
        continue_text = self._get_continue_button_text()
        self.continue_button.set_text(continue_text)

    def _update_expected_depth(self, *args):
        """Update the depth to field based on the interval selection."""
        try:
            # Only update if depth_from is valid
            depth_from_str = self.depth_from.get().strip()
            if depth_from_str and depth_from_str.isdigit():
                depth_from = int(depth_from_str)

                # Get selected interval
                interval = self.interval_var.get()

                # Calculate new depth_to
                depth_to = depth_from + (20 * interval)  # 20 compartments

                # Update depth_to field
                self.depth_to.set(str(int(depth_to)))

                # Update compartment interval
                self.compartment_interval = interval

                # Update marker to compartment mapping - FIXED positions
                self.marker_to_compartment = {
                    marker_id: int(depth_from + ((marker_id - 3) * interval))
                    for marker_id in range(4, 24)
                }

                # Update boundary manager's mapping
                if hasattr(self, "boundary_manager"):
                    self.boundary_manager.marker_to_compartment = (
                        self.marker_to_compartment
                    )
                    self.boundary_manager.compartment_interval = interval

                # Also update compartment labels
                self._update_compartment_labels()

                # Force visualization update to show new labels
                self._update_visualization(force_full_update=True)

        except (ValueError, TypeError) as e:
            # Silently handle errors during auto-update
            self.logger.debug(f"Auto-update depth error: {str(e)}")

    def _update_compartment_labels(self, *args):
        """Update compartment labels based on depth information."""
        try:
            # Get depth values from interface
            depth_from_str = self.depth_from.get().strip()
            depth_to_str = self.depth_to.get().strip()
            interval = self.interval_var.get()

            # Only proceed if we have valid depth values
            if (
                depth_from_str
                and depth_to_str
                and depth_from_str.isdigit()
                and depth_to_str.isdigit()
            ):
                # Parse depth values
                depth_from = int(depth_from_str)
                depth_to = int(depth_to_str)

                # Check for valid range
                if depth_to <= depth_from:
                    return

                # Update marker to compartment mapping
                # Each marker has a FIXED position - marker 4 is always compartment 1, etc.
                new_mapping = {}
                for marker_id in range(4, 24):  # Markers 4-23 for compartments 1-20
                    compartment_number = marker_id - 3  # Marker 4 = compartment 1
                    compartment_depth = depth_from + (compartment_number * interval)
                    new_mapping[marker_id] = int(compartment_depth)

                # Update the mapping
                self.marker_to_compartment = new_mapping

                # Update boundary manager's mapping
                if hasattr(self, "boundary_manager"):
                    self.boundary_manager.marker_to_compartment = (
                        self.marker_to_compartment
                    )
                    self.boundary_manager.compartment_interval = interval

                # Force a full visualization update to show new labels
                self._update_visualization(force_full_update=True)

                # Update compartment interval
                self.compartment_interval = interval

                # Update metadata dict
                self.metadata["hole_id"] = self.hole_id.get().strip()
                self.metadata["depth_from"] = depth_from
                self.metadata["depth_to"] = depth_to
                self.metadata["compartment_interval"] = interval
        except Exception as e:
            self.logger.error(f"Error updating compartment labels: {str(e)}")

    def _validate_metadata(self):
        """Validate metadata input before switching modes."""
        try:
            hole_id = self.hole_id.get().strip()
            depth_from_str = self.depth_from.get().strip()
            depth_to_str = self.depth_to.get().strip()

            # Validate hole ID - must be 2 letters followed by 4 digits
            if not hole_id:
                DialogHelper.show_message(
                    self.dialog,
                    "Validation Error",
                    "Hole ID is required",
                    message_type="error",
                )
                return False

            # Validate hole ID format using regex - 2 letters followed by 4 digits
            if not re.match(r"^[A-Za-z]{2}\d{4}$", hole_id):
                DialogHelper.show_message(
                    self.dialog,
                    "Validation Error",
                    "Hole ID must be 2 letters followed by 4 digits (e.g., AB1234)",
                    message_type="error",
                )
                return False

            # Validate depth range if provided - must be integers
            if not depth_from_str or not depth_to_str:
                DialogHelper.show_message(
                    self.dialog,
                    "Validation Error",
                    "Depth From and Depth To are required",
                    message_type="error",
                )
                return False

            try:
                depth_from = int(depth_from_str)
                depth_to = int(depth_to_str)

                # Validate as non-negative
                if depth_from < 0 or depth_to < 0:
                    DialogHelper.show_message(
                        self.dialog,
                        "Validation Error",
                        "Depth values cannot be negative",
                        message_type="error",
                    )
                    return False

                # Validate depth_to > depth_from
                if depth_to <= depth_from:
                    DialogHelper.show_message(
                        self.dialog,
                        "Validation Error",
                        "Depth To must be greater than Depth From",
                        message_type="error",
                    )
                    return False

                # Get the compartment interval
                interval = self.interval_var.get()

                # Calculate expected depth range based on compartment count and interval
                expected_depth_range = 20 * interval  # 20 compartments is the standard
                actual_depth_range = depth_to - depth_from

                # Validate depth range matches expected range for the interval
                if actual_depth_range != expected_depth_range:
                    if not DialogHelper.confirm_dialog(
                        self.dialog,
                        "Depth Range Warning",
                        f"The expected depth range for {interval}m interval is {expected_depth_range}m, "
                        f"but you entered {actual_depth_range}m.\n\n"
                        f"With {interval}m interval, depths should be exactly {expected_depth_range}m apart.\n\n"
                        f"Do you want to continue with this non-standard depth range?",
                        yes_text="Continue",
                        no_text="Cancel",
                    ):
                        return False

                # Check depth validation if available
                if (
                    self.app
                    and hasattr(self.app, "depth_validator")
                    and self.app.depth_validator
                    and self.app.depth_validator.is_loaded
                ):
                    # Validate the depth range
                    is_valid, error_msg = self.app.depth_validator.validate_depth_range(
                        hole_id, depth_from, depth_to
                    )

                    if not is_valid:
                        # Get the valid range for more detailed error message
                        valid_range = self.app.depth_validator.get_valid_range(hole_id)

                        if valid_range:
                            min_depth, max_depth = valid_range
                            if "not found" in error_msg:
                                # Hole not in database
                                if not DialogHelper.confirm_dialog(
                                    self.dialog,
                                    "Hole ID Not Found",
                                    f"Hole ID '{hole_id}' was not found in the depth validation database.\n\n"
                                    f"This may indicate an incorrect Hole ID or a new hole not yet in the database.\n\n"
                                    f"Do you want to continue anyway?",
                                    yes_text="Continue",
                                    no_text="Cancel",
                                ):
                                    return False
                            else:
                                # Depth range exceeds limits
                                # Format the message
                                message = (
                                    f"{error_msg}\n\n"
                                    f"Valid depth range for {hole_id}: {min_depth}m to {max_depth}m\n"
                                    f"You entered: {depth_from}m to {depth_to}m\n\n"
                                    f"Do you want to continue with this depth range anyway?"
                                )

                                # Swap yes/no texts to get desired colors (yes=green, no=red)
                                user_wants_to_cancel = DialogHelper.confirm_dialog(
                                    self.dialog,
                                    DialogHelper.t("Depth Range Exceeds Limits"),
                                    message,  # Already formatted
                                    yes_text=DialogHelper.t("Cancel"),  # Green button
                                    no_text=DialogHelper.t("Continue"),  # Red button
                                )
                                # If user clicked "Cancel" (green), return False to stop
                                if user_wants_to_cancel:
                                    return False
                                # Otherwise user clicked "Continue" (red), so we proceed
                        else:
                            # No valid range found
                            if not DialogHelper.confirm_dialog(
                                self.dialog,
                                "Depth Validation Warning",
                                f"{error_msg}\n\n" f"Do you want to continue anyway?",
                                yes_text="Continue",
                                no_text="Cancel",
                            ):
                                return False

            except ValueError:
                DialogHelper.show_message(
                    self.dialog,
                    "Validation Error",
                    "Depth values must be whole numbers",
                    message_type="error",
                )
                return False

            # Store validated metadata
            self.metadata["hole_id"] = hole_id
            self.metadata["depth_from"] = depth_from
            self.metadata["depth_to"] = depth_to
            self.metadata["compartment_interval"] = interval

            return True

        except Exception as e:
            self.logger.error(f"Error validating metadata: {str(e)}")
            DialogHelper.show_message(
                self.dialog,
                "Error",
                f"An error occurred during validation: {str(e)}",
                message_type="error",
            )
            return False

    def _check_increment_validity(self):
        """
        Check if incrementing from last metadata would be valid.

        Returns:
            tuple: (is_valid, reason_message)
        """
        try:
            # Check if we have app reference
            if not self.app or not hasattr(self.app, "get_incremented_metadata"):
                return False, "Increment function not available"

            # Get incremented metadata from app
            incremented = self.app.get_incremented_metadata()

            if not incremented or not incremented.get("hole_id"):
                return False, "No previous metadata available"

            hole_id = incremented.get("hole_id", "")
            depth_from = incremented.get("depth_from")

            if depth_from is None:
                return False, "Invalid previous depth data"

            # Get the currently selected interval from the dropdown
            current_interval = self.interval_var.get()

            # Calculate what depth_to would be
            calculated_depth_to = depth_from + (20 * current_interval)

            # Perform depth validation if validator is available
            if (
                hasattr(self.app, "depth_validator")
                and self.app.depth_validator
                and self.app.depth_validator.is_loaded
            ):
                # Check if hole exists and validate depth
                valid_range = self.app.depth_validator.get_valid_range(hole_id)

                if valid_range:
                    min_depth, max_depth = valid_range

                    # Check if the new depth range would be valid
                    if depth_from < min_depth:
                        return (
                            False,
                            f"Start depth {depth_from}m is before minimum ({min_depth}m)",
                        )

                    if calculated_depth_to > max_depth:
                        return (
                            False,
                            f"End depth {calculated_depth_to}m would exceed maximum ({max_depth}m)",
                        )

                    # Also check if we're already at the maximum valid starting position
                    # (considering we need room for a full 20-compartment interval)
                    max_valid_start = max_depth - (20 * current_interval)
                    if depth_from > max_valid_start:
                        return (
                            False,
                            f"Cannot fit full interval (already at depth {depth_from}m, max {max_depth}m)",
                        )

            return True, "Can increment"

        except Exception as e:
            self.logger.error(f"Error checking increment validity: {str(e)}")
            return False, "Error checking validity"

    def _update_increment_button_state(self):
        """Update the increment button enabled/disabled state based on validity."""
        is_valid, reason = self._check_increment_validity()

        if hasattr(self, "increment_button"):
            if hasattr(self.increment_button, "set_enabled"):
                # ModernButton
                self.increment_button.set_enabled(is_valid)
                if not is_valid:
                    # Optionally update button tooltip/text
                    self.increment_button.set_text(f"Increment From Last ({reason})")
                else:
                    self.increment_button.set_text("Increment From Last")
            else:
                # ttk.Button
                state = tk.NORMAL if is_valid else tk.DISABLED
                self.increment_button.config(state=state)

                # Update button text to show reason when disabled
                if not is_valid and reason != "No previous metadata available":
                    self.increment_button.config(text=f"Increment ({reason})")
                else:
                    self.increment_button.config(text="Increment From Last")

        self.logger.debug(f"Increment button state updated: {is_valid} - {reason}")

    def _on_increment_from_last(self):
        """Handle Increment From Last button click."""
        try:
            # Check if we can increment
            is_valid, reason = self._check_increment_validity()
            if not is_valid:
                self.logger.warning(f"Cannot increment: {reason}")
                return

            # Get incremented metadata from app
            incremented = self.app.get_incremented_metadata()

            if not incremented:
                return

            # Apply the incremented values
            self.hole_id.set(incremented.get("hole_id", ""))
            self.depth_from.set(str(incremented.get("depth_from", "")))

            # Calculate depth_to based on current interval
            depth_from = incremented.get("depth_from", 0)
            interval = self.interval_var.get()
            depth_to = depth_from + (20 * interval)

            self.depth_to.set(str(depth_to))

            # Update metadata dict
            self.metadata["hole_id"] = incremented.get("hole_id", "")
            self.metadata["depth_from"] = depth_from
            self.metadata["depth_to"] = depth_to
            self.metadata["compartment_interval"] = interval

            # Update compartment labels
            self._update_compartment_labels()

            # Update status
            self.status_var.set(
                f"Incremented to {incremented.get('hole_id', '')} depths {depth_from}-{depth_to}m"
            )

        except Exception as e:
            self.logger.error(f"Error incrementing from last: {str(e)}")
            self.status_var.set("Error incrementing from last metadata")

    def _get_continue_button_text(self):
        """Get the appropriate text for the continue button based on current mode."""
        if self.current_mode == self.MODE_METADATA:
            # In metadata mode, continue moves to missing boundaries mode
            return DialogHelper.t("Continue to Boundaries")

        if self.current_mode == self.MODE_MISSING_BOUNDARIES:
            if self.missing_marker_ids and not self.annotation_complete:
                # Still need to place markers
                return DialogHelper.t("Place Markers")
            else:
                # All markers placed, ready to move to adjustment mode
                return DialogHelper.t("Continue to Adjustment")

        if self.current_mode == self.MODE_ADJUST_BOUNDARIES:
            # Final step
            return DialogHelper.t("Finish")

        # Default text
        return DialogHelper.t("Continue")

    def _get_instruction_text(self):
        """Get mode-specific instruction text."""
        if self.current_mode == self.MODE_METADATA:
            return DialogHelper.t("Enter Core Metadata")

        elif self.current_mode == self.MODE_MISSING_BOUNDARIES:
            if self.missing_marker_ids and not self.annotation_complete:
                current_id = (
                    self.missing_marker_ids[self.current_index]
                    if self.current_index < len(self.missing_marker_ids)
                    else None
                )
                if current_id == 24:
                    return DialogHelper.t("Place Metadata Marker (ID 24)")
                elif current_id in [0, 1, 2, 3]:  # Corner markers
                    # Corner markers are handled automatically, skip
                    return DialogHelper.t("Corner markers are placed automatically")
                else:
                    # Use marker_to_compartment mapping for depth display
                    compartment_number = self.marker_to_compartment.get(
                        current_id, current_id - 3
                    )
                    return DialogHelper.t(
                        f"Place Compartment at Depth {compartment_number}m"
                    )
            elif self.annotation_complete:
                return DialogHelper.t(
                    "All Markers Placed - Click Continue to Adjustment"
                )
            else:
                return DialogHelper.t(
                    "No Missing Markers - Click Continue to Adjustment"
                )

        elif self.current_mode == self.MODE_ADJUST_BOUNDARIES:
            return DialogHelper.t("Adjust Compartment Boundaries")

        # Default instruction
        return DialogHelper.t("Compartment Registration")

    def _update_status_message(self):
        """Update status message based on current mode."""
        if self.current_mode == self.MODE_METADATA:
            self.status_var.set(
                DialogHelper.t(
                    "Enter hole ID and depth information, then click Continue to Boundaries"
                )
            )

        elif self.current_mode == self.MODE_MISSING_BOUNDARIES:
            if self.missing_marker_ids and not self.annotation_complete:
                current_id = (
                    self.missing_marker_ids[self.current_index]
                    if self.current_index < len(self.missing_marker_ids)
                    else None
                )
                if current_id == 24:
                    self.status_var.set(
                        DialogHelper.t("Click to place the metadata marker")
                    )
                elif current_id in [0, 1, 2, 3]:  # Corner markers
                    # Corner markers are handled automatically
                    self.status_var.set(
                        DialogHelper.t("Corner markers are placed automatically")
                    )
                else:
                    # Get depth for compartment from marker_to_compartment mapping
                    depth = self.marker_to_compartment.get(current_id, current_id - 3)
                    self.status_var.set(
                        DialogHelper.t(
                            f"Click to place compartment at depth {depth}m (marker {current_id})"
                        )
                    )
            elif self.annotation_complete:
                # Check if there are interpolated boundaries that can be clicked
                if (
                    hasattr(self, "interpolated_boundary_to_marker")
                    and self.interpolated_boundary_to_marker
                ):
                    self.status_var.set(
                        DialogHelper.t(
                            "All markers placed. Click any orange boundary to remove it and place manually, or click 'Continue to Adjustment'."
                        )
                    )
                else:
                    self.status_var.set(
                        DialogHelper.t(
                            "All markers placed successfully. Click 'Continue to Adjustment' to proceed."
                        )
                    )
            else:
                self.status_var.set(
                    DialogHelper.t(
                        "No missing markers to place. Click 'Continue to Adjustment' to proceed."
                    )
                )

        elif self.current_mode == self.MODE_ADJUST_BOUNDARIES:
            self.status_var.set(
                DialogHelper.t(
                    "Adjust top and bottom boundaries or side heights, then click 'Finish'"
                )
            )

    def _adjust_height(self, delta: int) -> None:
        """
        Adjust the overall height by moving both top and bottom boundaries
        by the same amount, maintaining their distance.

        Args:
            delta: Change in position (positive = down, negative = up)
        """
        # Calculate the height of the image
        if self.source_image is None:
            return

        img_height = self.source_image.shape[0]

        # Debug: Log original boundaries
        self.logger.debug(
            f"Before adjustment - top_y: {self.top_y}, bottom_y: {self.bottom_y}"
        )

        # Calculate current distance between boundaries
        distance = self.bottom_y - self.top_y

        # Move both boundaries by delta while maintaining distance
        new_top_y = max(0, min(img_height - distance - 1, self.top_y + delta))
        new_bottom_y = new_top_y + distance

        # Ensure bottom doesn't go beyond image height
        if new_bottom_y >= img_height:
            new_bottom_y = img_height - 1
            new_top_y = new_bottom_y - distance

        # Update boundary positions
        self.top_y = new_top_y
        self.bottom_y = new_bottom_y

        # Debug: Log new boundaries
        self.logger.debug(
            f"After adjustment - top_y: {self.top_y}, bottom_y: {self.bottom_y}"
        )

        # IMPORTANT: Also update manually placed compartments
        self._update_manual_compartments()

        # Update visualization
        self._update_visualization()

        # Update status and static zoom views
        direction = "down" if delta > 0 else "up"
        self.status_var.set(f"Moved both boundaries {direction} by {abs(delta)} pixels")
        self._update_static_zoom_views()

        # Apply adjustments
        self._apply_adjustments()

    def _adjust_side_height(self, side, delta):
        """
        Adjust the height of a specific side of the boundary.

        Args:
            side: Which side to adjust ('left' or 'right')
            delta: Change in height (positive = down, negative = up)
        """
        # Initialize side heights if not already set
        if not hasattr(self, "left_height_offset"):
            self.left_height_offset = 0
            self.right_height_offset = 0

        # Debug: Log original offsets
        self.logger.debug(
            f"Before side adjustment - left_offset: {self.left_height_offset}, right_offset: {self.right_height_offset}"
        )

        # Adjust the appropriate side
        if side == "left":
            self.left_height_offset += delta
            self.status_var.set(f"Adjusted left side height by {delta} pixels")
        elif side == "right":
            self.right_height_offset += delta
            self.status_var.set(f"Adjusted right side height by {delta} pixels")

        # Debug: Log new offsets
        self.logger.debug(
            f"After side adjustment - left_offset: {self.left_height_offset}, right_offset: {self.right_height_offset}"
        )

        # IMPORTANT: Also update manually placed compartments
        self._update_manual_compartments()

        # Update visualization
        self._update_visualization()

        # Update static zoom views
        self._update_static_zoom_views()

        # Apply adjustments
        self._apply_adjustments()

    def _update_manual_compartments(self):
        """Update the y-coordinates of manually placed compartments based on current boundaries."""
        if not self.result_boundaries:
            return

        img_width = self.source_image.shape[1] if self.source_image is not None else 0

        # Loop through manually placed compartments
        for marker_id, boundary in list(self.result_boundaries.items()):
            # Skip metadata marker (ID 24)
            if marker_id == 24:
                continue

            # Only process rectangular compartment boundaries
            if isinstance(boundary, tuple) and len(boundary) == 4:
                x1, _, x2, _ = boundary

                # Calculate new y-coordinates using current boundary settings
                left_top_y = self.top_y + self.left_height_offset
                right_top_y = self.top_y + self.right_height_offset
                left_bottom_y = self.bottom_y + self.left_height_offset
                right_bottom_y = self.bottom_y + self.right_height_offset

                # Calculate slopes
                if img_width > 0:
                    top_slope = (right_top_y - left_top_y) / img_width
                    bottom_slope = (right_bottom_y - left_bottom_y) / img_width

                    # Calculate y values at this x-position
                    mid_x = (x1 + x2) / 2
                    new_y1 = int(left_top_y + (top_slope * mid_x))
                    new_y2 = int(left_bottom_y + (bottom_slope * mid_x))
                else:
                    new_y1 = self.top_y
                    new_y2 = self.bottom_y

                # Update the manually placed boundary
                self.result_boundaries[marker_id] = (x1, new_y1, x2, new_y2)

    def _apply_adjustments(self):
        """Apply current boundary adjustments and refresh the visualization."""
        try:
            self.logger.debug(
                f"Applying adjustments - top_y: {self.top_y}, bottom_y: {self.bottom_y}, "
                f"left_offset: {self.left_height_offset}, right_offset: {self.right_height_offset}"
            )

            # Store the original boundaries for comparison
            original_boundaries = (
                self.detected_boundaries.copy() if self.detected_boundaries else []
            )

            # Recreate boundaries using current parameters
            img_width = (
                self.source_image.shape[1] if self.source_image is not None else 0
            )
            adjusted_boundaries = []

            # Update auto-detected boundaries
            for i, (x1, _, x2, _) in enumerate(self.detected_boundaries):
                # Calculate y-coordinates based on x-position using boundary lines
                left_top_y = self.top_y + self.left_height_offset
                right_top_y = self.top_y + self.right_height_offset
                left_bottom_y = self.bottom_y + self.left_height_offset
                right_bottom_y = self.bottom_y + self.right_height_offset

                # Calculate slopes for top and bottom boundaries
                if img_width > 0:
                    top_slope = (right_top_y - left_top_y) / img_width
                    bottom_slope = (right_bottom_y - left_bottom_y) / img_width

                    # Calculate y values at the x-position of this compartment
                    mid_x = (x1 + x2) / 2
                    new_y1 = int(left_top_y + (top_slope * mid_x))
                    new_y2 = int(left_bottom_y + (bottom_slope * mid_x))
                else:
                    new_y1 = self.top_y
                    new_y2 = self.bottom_y

                # Create adjusted boundary
                adjusted_boundaries.append((x1, new_y1, x2, new_y2))

            # Update detected boundaries with adjusted ones
            self.detected_boundaries = adjusted_boundaries

            # Also adjust any manually placed compartments in result_boundaries
            for marker_id, boundary in list(self.result_boundaries.items()):
                # Skip metadata marker (ID 24) which uses a different format
                if marker_id == 24:
                    continue

                # Only process rectangular compartment boundaries
                if isinstance(boundary, tuple) and len(boundary) == 4:
                    x1, _, x2, _ = boundary

                    # Calculate new y-coordinates using the same logic
                    left_top_y = self.top_y + self.left_height_offset
                    right_top_y = self.top_y + self.right_height_offset
                    left_bottom_y = self.bottom_y + self.left_height_offset
                    right_bottom_y = self.bottom_y + self.right_height_offset

                    # Calculate slopes
                    if img_width > 0:
                        top_slope = (right_top_y - left_top_y) / img_width
                        bottom_slope = (right_bottom_y - left_bottom_y) / img_width

                        # Calculate y values at this x-position
                        mid_x = (x1 + x2) / 2
                        new_y1 = int(left_top_y + (top_slope * mid_x))
                        new_y2 = int(left_bottom_y + (bottom_slope * mid_x))
                    else:
                        new_y1 = self.top_y
                        new_y2 = self.bottom_y

                    # Update the manually placed boundary
                    self.result_boundaries[marker_id] = (x1, new_y1, x2, new_y2)

            # Update visualization
            self._update_visualization(force_full_update=True)

            # callback to main.py process_image for applying adjustments, call it
            if callable(self.on_apply_adjustments):
                current_time = time.time()
                if current_time - self._last_apply_time > self._apply_debounce_interval:
                    self._last_apply_time = current_time
                    # Create parameters dict to pass to callback
                    adjustment_params = {
                        "top_boundary": self.top_y,
                        "bottom_boundary": self.bottom_y,
                        "left_height_offset": self.left_height_offset,
                        "right_height_offset": self.right_height_offset,
                        "boundaries": self.detected_boundaries,
                    }
                    # Call the callback
                    self.on_apply_adjustments(adjustment_params)

        except Exception as e:
            self.logger.error(f"Error applying adjustments: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.status_var.set(f"Error applying adjustments: {str(e)}")

    def _on_canvas_click(self, event):
        """Handle canvas click events - simplified for compartment placement only."""
        # Ignore clicks if dialog is closing
        if (
            not hasattr(self, "dialog")
            or not self.dialog
            or not self.dialog.winfo_exists()
        ):
            return

        try:
            # Convert canvas coordinates to image coordinates
            image_x = int((event.x - self.canvas_offset_x) / self.scale_ratio)
            image_y = int((event.y - self.canvas_offset_y) / self.scale_ratio)

            # Ensure click is within image bounds
            if self.source_image is None:
                return

            img_height, img_width = self.source_image.shape[:2]
            if not (0 <= image_x < img_width and 0 <= image_y < img_height):
                return

            # Only handle clicks in missing boundaries mode
            if self.current_mode != self.MODE_MISSING_BOUNDARIES:
                return

            # DEBUG: Print current state before any operations
            self.logger.debug("=== CANVAS CLICK DEBUG START ===")
            self.logger.debug(f"Click position: image_x={image_x}, image_y={image_y}")
            self.logger.debug(
                f"Current boundaries count: {len(self.detected_boundaries)}"
            )
            self.logger.debug(
                f"Current boundary_to_marker mapping: {self.boundary_to_marker}"
            )
            self.logger.debug(f"Current missing_marker_ids: {self.missing_marker_ids}")
            self.logger.debug(
                f"Current marker_to_compartment: {self.marker_to_compartment}"
            )

            # DEBUG: Print depth mapping for all current boundaries
            self.logger.debug("Current boundary depths:")
            for idx, boundary in enumerate(self.detected_boundaries):
                marker_id = self.boundary_to_marker.get(idx, "Unknown")
                depth = self.marker_to_compartment.get(
                    marker_id, f"No depth for marker {marker_id}"
                )
                self.logger.debug(
                    f"  Boundary[{idx}]: marker_id={marker_id}, depth={depth}m, bounds={boundary}"
                )
            # Check if user clicked ANY boundary to remove it
            # First, sync the current boundaries to ensure we're checking the right data
            self._sync_from_boundary_manager()

            # Debug: Log all current boundaries before checking
            self.logger.debug("Current boundaries before click detection:")
            for idx, (x1, y1, x2, y2) in enumerate(self.detected_boundaries):
                if idx in self.boundary_to_marker:
                    marker_id = self.boundary_to_marker[idx]
                    self.logger.debug(
                        f"  Boundary[{idx}]: marker={marker_id}, x1={x1}, x2={x2}, clicked_x={image_x}, in_range={x1 <= image_x <= x2}"
                    )

            clicked_boundary = self.boundary_manager.get_boundary_at_position(
                image_x, image_y
            )

            if clicked_boundary:
                marker_id, boundary = clicked_boundary

                # Log what was actually clicked
                self.logger.info(
                    f"Click at ({image_x}, {image_y}) matched boundary for marker {marker_id}: ({boundary.x1}, {boundary.y1}, {boundary.x2}, {boundary.y2})"
                )

                # Don't allow removing corner or metadata markers
                if marker_id in [0, 1, 2, 3, 24]:
                    self.logger.debug(f"Skipping removal of special marker {marker_id}")
                    return

                # Remove the boundary
                self.logger.info(f"Removing boundary for marker {marker_id}")
                self.boundary_manager.remove_boundary(marker_id)

                # Add to missing markers for manual placement
                if marker_id not in self.missing_marker_ids:
                    self.missing_marker_ids.append(marker_id)
                    self.missing_marker_ids.sort()

                # Set index to place this marker
                self.current_index = self.missing_marker_ids.index(marker_id)
                self.annotation_complete = False

                # Update status
                depth = self.marker_to_compartment.get(marker_id, marker_id - 3)
                self.status_var.set(
                    f"Removed compartment at depth {depth}m - click to place manually"
                )

                # Update legacy format for compatibility
                self._sync_from_boundary_manager()

                # Verify the sync worked correctly
                self.logger.debug(
                    f"After sync - detected_boundaries count: {len(self.detected_boundaries)}"
                )
                self.logger.debug(
                    f"After sync - boundary_manager count: {len(self.boundary_manager.boundaries)}"
                )

                # Re-run wall detection and update visualization
                # self._rerun_wall_detection()
                self._update_visualization(force_full_update=True)
                self._update_status_message()
                return

            # If no boundary was clicked, check if we're placing a new marker
            if not self.missing_marker_ids or self.current_index >= len(
                self.missing_marker_ids
            ):
                self.logger.debug("No markers to place or index out of range")
                return

            # Get current marker to place
            current_id = self.missing_marker_ids[self.current_index]

            # DEBUG: Log placement attempt
            self.logger.debug(f"=== PLACING NEW MARKER ===")
            self.logger.debug(f"Current missing markers: {self.missing_marker_ids}")
            self.logger.debug(f"Current index: {self.current_index}")
            self.logger.debug(f"Marker to place: {current_id}")
            self.logger.debug(
                f"Expected depth: {self.marker_to_compartment.get(current_id, current_id - 3)}m"
            )

            # Skip corner and metadata markers
            if current_id in [0, 1, 2, 3, 24]:
                self.logger.debug(f"Skipping special marker {current_id}")
                self.current_index += 1
                self._update_visualization()
                self._update_status_message()
                return

            # Check if this marker ID already exists in boundaries
            # Convert to int for comparison to handle numpy.int32 vs int issues
            existing_boundary_idx = None
            current_id_int = int(current_id)

            for idx, existing_marker_id in self.boundary_to_marker.items():
                if int(existing_marker_id) == current_id_int:
                    existing_boundary_idx = idx
                    self.logger.warning(
                        f"Marker {current_id} already exists at boundary index {idx}! This should not happen."
                    )
                    # Skip to next marker
                    self.current_index += 1
                    if self.current_index >= len(self.missing_marker_ids):
                        self.annotation_complete = True
                    self._update_visualization()
                    self._update_status_message()
                    return

            # Check for overlaps with existing boundaries
            if self._would_overlap_existing(image_x):
                self.logger.debug(
                    f"Placement at x={image_x} would overlap existing boundary"
                )
                self.status_var.set(
                    "Cannot place compartment here - overlaps with existing compartment"
                )
                return

            # Calculate compartment boundaries
            half_width = self.avg_width // 2
            x1 = max(0, image_x - half_width)
            x2 = min(img_width - 1, image_x + half_width)

            # Use the vertical constraints
            y1 = self.top_y
            y2 = self.bottom_y

            self.logger.debug(f"Calculated boundary: ({x1}, {y1}, {x2}, {y2})")
            self.logger.debug(
                f"Average width: {self.avg_width}, half_width: {half_width}"
            )

            # Add boundary to manager
            success = self.boundary_manager.add_boundary(
                current_id, x1, y1, x2, y2, "manual"
            )

            if success:
                self.logger.debug(f"Added manual boundary for marker {current_id}")

                # Update status
                depth = self.marker_to_compartment.get(current_id, current_id - 3)
                self.status_var.set(f"Placed compartment at depth {depth}m")

                # Move to next marker
                self.current_index += 1
                self.temp_point = None

                if self.current_index >= len(self.missing_marker_ids):
                    self.annotation_complete = True
                    self.status_var.set(
                        "All markers placed. Click 'Continue to Adjustment' to proceed."
                    )

                # Sync to legacy format
                self._sync_from_boundary_manager()

                # # Re-run wall detection
                # self._rerun_wall_detection()

                # Update visualization
                self._update_visualization(force_full_update=True)
                self._update_status_message()
            else:
                self.logger.error(f"Failed to add boundary for marker {current_id}")

        except Exception as e:
            self.logger.error(f"Error handling canvas click: {str(e)}")
            self.logger.error(traceback.format_exc())
            # DEBUG: Log state on error
            self.logger.debug("ERROR STATE:")
            self.logger.debug(f"  detected_boundaries: {len(self.detected_boundaries)}")
            self.logger.debug(f"  boundary_to_marker: {self.boundary_to_marker}")
            self.logger.debug(f"  missing_marker_ids: {self.missing_marker_ids}")

    def _on_canvas_move(self, event):
        """Handle mouse movement on canvas for zoom lens and previews."""
        try:
            # Convert canvas coordinates to image coordinates
            image_x = int((event.x - self.canvas_offset_x) / self.scale_ratio)
            image_y = int((event.y - self.canvas_offset_y) / self.scale_ratio)

            # Check if hovering over an interpolated boundary
            if self.current_mode == self.MODE_MISSING_BOUNDARIES and hasattr(
                self, "interpolated_boundary_to_marker"
            ):
                for (
                    boundary_idx,
                    marker_id,
                ) in self.interpolated_boundary_to_marker.items():
                    if boundary_idx < len(self.detected_boundaries):
                        x1, y1, x2, y2 = self.detected_boundaries[boundary_idx]
                        if x1 <= image_x <= x2 and y1 <= image_y <= y2:
                            # Update cursor to indicate clickable
                            self.canvas.config(cursor="hand2")
                            # Update status to show this is clickable
                            depth = self.marker_to_compartment.get(
                                marker_id, marker_id - 3
                            )
                            self.status_var.set(
                                f"Click to remove interpolated compartment at depth {depth}m"
                            )
                            break
                else:
                    # Not hovering over interpolated boundary
                    self.canvas.config(cursor="")

            # Ensure the point is within the image bounds
            if self.source_image is None:
                return

            img_height, img_width = self.source_image.shape[:2]
            if not (0 <= image_x < img_width and 0 <= image_y < img_height):
                # Hide zoom lens if cursor is outside image
                if hasattr(self, "_zoom_lens") and self._zoom_lens:
                    self._zoom_lens.withdraw()
                if hasattr(self, "_zoom_lens_flipped") and self._zoom_lens_flipped:
                    self._zoom_lens_flipped.withdraw()
                self.temp_point = None
                return

            # Update temp point
            old_temp_point = self.temp_point
            self.temp_point = (image_x, image_y)

            # Only update visualization if preview would change
            if (
                self.current_mode == self.MODE_MISSING_BOUNDARIES
                and self.temp_point is not None
            ):
                # Check if we actually need to update
                if self.missing_marker_ids and self.current_index < len(
                    self.missing_marker_ids
                ):
                    current_id = self.missing_marker_ids[self.current_index]

                    # For compartment markers, check if X changed significantly
                    if (
                        old_temp_point
                        and abs(old_temp_point[0] - self.temp_point[0]) < 2
                    ):
                        # X didn't change much, skip update
                        pass
                    else:
                        self._update_visualization()

            # Show zoom lens based on mode and mouse state
            # CHANGED: Always show zoom on hover, use button state only for flipped view
            if self.current_mode == self.MODE_METADATA:
                if hasattr(self, "_zoom_lens") and self._zoom_lens:
                    # Check if right mouse button is pressed for flipped view
                    right_button_pressed = bool(
                        event.state & 0x400
                    )  # Right button mask
                    self._render_zoom(self.canvas, event, flipped=right_button_pressed)

            elif self.current_mode == self.MODE_MISSING_BOUNDARIES:
                if hasattr(self, "_zoom_lens") and self._zoom_lens:
                    if (
                        self.missing_marker_ids
                        and self.current_index < len(self.missing_marker_ids)
                        and self.missing_marker_ids[self.current_index]
                        not in [0, 1, 2, 3]
                    ):
                        # Always show zoom on hover in this mode
                        self._render_zoom(self.canvas, event)

        except Exception as e:
            self.logger.error(f"Error handling canvas movement: {str(e)}")

    def _sync_from_boundary_manager(self):
        """Sync boundary manager data to legacy format for compatibility."""
        boundaries, boundary_to_marker, boundary_types = (
            self.boundary_manager.export_to_legacy_format()
        )

        self.detected_boundaries = boundaries
        self.boundary_to_marker = boundary_to_marker
        self.boundary_types = boundary_types

    def _sync_to_boundary_manager(self):
        """Sync legacy format to boundary manager."""
        if hasattr(self, "detected_boundaries") and hasattr(self, "boundary_to_marker"):
            self.boundary_manager.import_boundaries(
                self.detected_boundaries,
                self.boundary_to_marker,
                self.boundary_types if hasattr(self, "boundary_types") else None,
            )

    def _create_synthetic_markers_for_boundaries(self):
        """
        Create synthetic ArUco marker data for interpolated and manually placed boundaries.
        This ensures wall detection works properly on all compartments.

        Returns:
            Dict of synthetic markers {marker_id: corners_array}
        """
        synthetic_markers = {}

        # Get marker size from config
        marker_size_cm = self.config.get("compartment_marker_size_cm", 2.0)

        # Get scale if available
        scale_px_per_cm = None
        if self.boundary_analysis:
            scale_px_per_cm = self.boundary_analysis.get("scale_px_per_cm")

        if not scale_px_per_cm:
            # Fallback: estimate from compartment width
            if self.detected_boundaries:
                avg_width = np.mean([b[2] - b[0] for b in self.detected_boundaries])
                # Assume compartment is ~2cm wide
                scale_px_per_cm = avg_width / 2.0

        marker_size_px = (
            int(marker_size_cm * scale_px_per_cm) if scale_px_per_cm else 40
        )

        # Process each boundary
        for idx, (x1, y1, x2, y2) in enumerate(self.detected_boundaries):
            marker_id = self.boundary_to_marker.get(idx)

            # Skip if no marker ID or if it's a real detected marker
            if marker_id is None or marker_id in self.markers:
                continue

            # Create synthetic marker centered below the compartment
            center_x = (x1 + x2) // 2
            marker_top = y2 + 10  # Small gap below compartment

            # Create marker corners (square marker)
            half_size = marker_size_px // 2
            corners = np.array(
                [
                    [center_x - half_size, marker_top],  # top-left
                    [center_x + half_size, marker_top],  # top-right
                    [center_x + half_size, marker_top + marker_size_px],  # bottom-right
                    [center_x - half_size, marker_top + marker_size_px],  # bottom-left
                ],
                dtype=np.float32,
            )

            synthetic_markers[marker_id] = corners

        self.logger.debug(
            f"Created {len(synthetic_markers)} synthetic markers for wall detection"
        )
        return synthetic_markers

    def _on_interpolate_clicked(self):
        """Toggle interpolation: first click interpolates, second click removes them."""
        # Check if we have interpolated boundaries
        interpolated_boundaries = self.boundary_manager.get_boundaries_by_type(
            "interpolated"
        )

        if interpolated_boundaries:
            # Remove all interpolated boundaries
            count = self.boundary_manager.clear_interpolated_boundaries()
            self.logger.info(f"Removed {count} interpolated boundaries")

            # Update button text
            if hasattr(self, "interpolate_button"):
                self.interpolate_button.set_text(DialogHelper.t("Interpolate"))

            # Sync to legacy format
            self._sync_from_boundary_manager()

            # Update missing markers list
            self.missing_marker_ids = self.boundary_manager.get_missing_markers()
            if self.missing_marker_ids:
                self.current_index = 0
                self.annotation_complete = False

            self.status_var.set("Removed interpolated compartments")
            self._update_visualization(force_full_update=True)
            self._update_status_message()
            return

        # No interpolated boundaries - create them
        self.logger.info("Creating interpolated boundaries for missing compartments")

        # Get current boundaries and mapping from boundary manager
        boundaries, boundary_to_marker, boundary_types = (
            self.boundary_manager.export_to_legacy_format()
        )

        if not boundaries:
            self.status_var.set("No boundaries to interpolate from")
            return

        # Get scale data
        scale_px_per_cm = None
        if self.scale_data:
            scale_px_per_cm = self.scale_data.get("scale_px_per_cm")
        elif self.boundary_analysis:
            scale_px_per_cm = self.boundary_analysis.get("scale_px_per_cm")

        if not scale_px_per_cm:
            self.logger.warning("No scale data available for interpolation")

        # Get image shape
        if self.source_image is None:
            self.logger.error("No source image available for interpolation")
            return

        image_shape = self.source_image.shape[:2]

        # Get vertical constraints
        vertical_constraints = (self.top_y, self.bottom_y)

        # Call interpolation method
        interpolation_result = self.interpolate_missing_boundaries(
            detected_boundaries=boundaries,
            vertical_constraints=vertical_constraints,
            scale_px_per_cm=scale_px_per_cm,
            config=self.config,
            image_shape=image_shape,
            boundary_to_marker=boundary_to_marker,
        )
        self.logger.debug(
            "  gap_analysis: %s", interpolation_result.get("gap_analysis")
        )

        # DEBUG: dump the raw interpolation result
        self.logger.debug(
            f"Interpolation result dict keys: {list(interpolation_result.keys())}"
        )
        self.logger.debug(
            "  interpolated_boundaries: %s",
            interpolation_result.get("interpolated_boundaries"),
        )
        self.logger.debug(
            "  interpolated_marker_ids: %s",
            interpolation_result.get("interpolated_marker_ids"),
        )

        # Process interpolation results
        interpolated = interpolation_result.get("interpolated_boundaries", [])
        interpolated_markers = interpolation_result.get("interpolated_marker_ids", [])

        if not interpolated:
            self.logger.debug(
                "No interpolated boundaries were returned → nothing to add"
            )
            self.status_var.set("No missing compartments to interpolate")
            return

        # Add interpolated boundaries to boundary manager
        added_count = 0
        for i, (x1, y1, x2, y2) in enumerate(interpolated):
            self.logger.debug(
                f"Attempting to add boundary #{i}: marker={interpolated_markers[i] if i < len(interpolated_markers) else '??'} "
                f"coords=({x1},{y1},{x2},{y2})"
            )
            if i < len(interpolated_markers):
                marker_id = interpolated_markers[i]
                if self.boundary_manager.add_boundary(
                    marker_id, x1, y1, x2, y2, "interpolated"
                ):
                    self.logger.debug(
                        f"  → boundary for marker {marker_id} added successfully"
                    )
                    added_count += 1
                else:
                    self.logger.warning(
                        f"  → boundary_manager rejected boundary for marker {marker_id}"
                    )

        self.logger.info(f"Added {added_count} interpolated boundaries")

        # Update button text
        if hasattr(self, "interpolate_button"):
            self.interpolate_button.set_text(DialogHelper.t("Remove Interpolation"))

        # Remove interpolated markers from missing list
        self.missing_marker_ids = [
            mid for mid in self.missing_marker_ids if mid not in interpolated_markers
        ]

        # If all markers are now placed, mark as complete
        if not self.missing_marker_ids:
            self.annotation_complete = True

        # Sync to legacy format
        self._sync_from_boundary_manager()

        # Re-run wall detection on newly interpolated boundaries
        self._rerun_wall_detection()
        # Update visualization
        self._update_visualization(force_full_update=True)

        # Update status
        self.status_var.set(f"Interpolated {added_count} missing compartments")
        self._update_status_message()

    def _sort_boundaries_by_depth(self):
        """
        Sort boundaries by their depth order and rebuild the boundary_to_marker mapping.
        This ensures compartments are always displayed in the correct order (1m, 2m, 3m... 20m).
        """
        if not self.detected_boundaries or not self.boundary_to_marker:
            return

        self.logger.debug("Sorting boundaries by depth order")

        # Create a list of (boundary_index, marker_id, boundary) tuples
        boundary_data = []
        for idx, boundary in enumerate(self.detected_boundaries):
            marker_id = self.boundary_to_marker.get(idx)
            if marker_id is not None:
                # Get the depth order (compartment markers 4-23 map to depths 1-20)
                if marker_id >= 4 and marker_id <= 23:
                    depth_order = (
                        marker_id - 3
                    )  # marker 4 = depth 1, marker 5 = depth 2, etc.
                else:
                    # Corner markers or metadata - put at end
                    depth_order = 1000 + marker_id

                boundary_data.append(
                    (
                        depth_order,
                        marker_id,
                        boundary,
                        (
                            self.boundary_types[idx]
                            if idx < len(self.boundary_types)
                            else "detected"
                        ),
                    )
                )

        # Sort by depth order
        boundary_data.sort(key=lambda x: x[0])

        # Rebuild the lists and mapping
        self.detected_boundaries = []
        self.boundary_types = []
        new_boundary_to_marker = {}

        for new_idx, (depth_order, marker_id, boundary, btype) in enumerate(
            boundary_data
        ):
            self.detected_boundaries.append(boundary)
            self.boundary_types.append(btype)
            new_boundary_to_marker[new_idx] = marker_id

        self.boundary_to_marker = new_boundary_to_marker

        # Debug log the new order
        self.logger.debug("Boundaries after sorting:")
        for idx, boundary in enumerate(self.detected_boundaries):
            marker_id = self.boundary_to_marker.get(idx)
            if marker_id and marker_id in self.marker_to_compartment:
                depth = self.marker_to_compartment[marker_id]
                self.logger.debug(
                    f"  Boundary[{idx}]: marker_id={marker_id}, depth={depth}m"
                )

    def _on_toggle_walls(self):
        """Toggle the wall-detection overlay on/off."""
        self.show_walls = not self.show_walls
        label = "Hide Walls" if self.show_walls else "Show Walls"
        self.toggle_walls_btn.configure(text=label)
        self._update_visualization(force_full_update=True)

    def _rerun_wall_detection(self):
        """
        Re-run wall detection ONLY on interpolated boundaries.
        Existing detected and manual boundaries remain fixed as anchors.
        """
        # 1) Guards
        if not self.boundary_analysis:
            self.logger.debug(
                "Cannot re-run wall detection - no boundary analysis data"
            )
            return
        if not getattr(self, "app", None) or not hasattr(self.app, "aruco_manager"):
            self.logger.debug(
                "Cannot re-run wall detection - missing app or aruco_manager"
            )
            return

        # 2) Pull out just the interpolated slots
        interpolated = self.boundary_manager.get_boundaries_by_type("interpolated")
        if not interpolated:
            self.logger.debug(
                "No interpolated boundaries to refine - skipping wall detection"
            )
            return

        # 3) If user has removed any, skip to avoid shifting anchors
        missing = self.boundary_manager.get_missing_markers()
        if missing:
            self.logger.info(
                f"Boundaries missing for markers {missing} - skipping wall detection"
            )
            return

        self.logger.info(
            f"Re-running wall detection on {len(interpolated)} interpolated boundaries"
        )

        # 4) Build full marker set: real + synthetic
        synthetic = self._create_synthetic_markers_for_boundaries()
        all_markers = {**self.markers, **synthetic}

        # 5) Gather fixed (detected/manual) boundaries so the analyzer won’t move them
        detected = self.boundary_manager.get_boundaries_by_type("detected")
        manual = self.boundary_manager.get_boundaries_by_type("manual")
        fixed = {mid: b.bounds for mid, b in {**detected, **manual}.items()}

        self.logger.debug(
            f"Fixed boundaries: {len(fixed)} (detected {len(detected)}, manual {len(manual)}); "
            f"Interpolated to adjust: {list(interpolated.keys())}"
        )

        # 6) Prepare metadata payload
        metadata = {
            "scale_px_per_cm": self.boundary_analysis.get("scale_px_per_cm"),
            "manual_boundaries": set(manual.keys()),
            "boundary_to_marker": self.boundary_to_marker.copy(),
            "fixed_boundaries": fixed,
            "interpolated_markers": list(interpolated.keys()),
            "boundary_adjustments": {
                "top_boundary": self.top_y,
                "bottom_boundary": self.bottom_y,
                "left_height_offset": self.left_height_offset,
                "right_height_offset": self.right_height_offset,
            },
        }

        # 7) Invoke the analyzer
        new_analysis = self.app.aruco_manager.analyze_compartment_boundaries(
            self.source_image,
            all_markers,
            compartment_count=self.config.get("compartment_count", 20),
            smart_cropping=True,
            metadata=metadata,
        )

        # 8) If we got back a new 'boundaries' list and a mapping, apply only to our interpolated ones
        if new_analysis and "boundaries" in new_analysis:
            refined = new_analysis["boundaries"]
            new_map = new_analysis.get("boundary_to_marker", {})

            updated = 0
            for mid in interpolated:
                # find the index in `refined` for this marker
                idx = next((i for i, m in new_map.items() if m == mid), None)
                if idx is None:
                    self.logger.warning(
                        f"Marker {mid} not found in new boundary_to_marker map"
                    )
                    continue

                x1, y1, x2, y2 = refined[idx]
                old = self.boundary_manager.boundaries[mid].bounds

                # overwrite the interpolated entry
                self.boundary_manager.add_boundary(mid, x1, y1, x2, y2, "interpolated")
                self.logger.debug(
                    f"Re-aligned interpolated boundary {mid}: {old} → ({x1},{y1},{x2},{y2})"
                )
                updated += 1

            self.logger.info(
                f"Updated {updated} interpolated boundaries using wall detection"
            )

            # 9) Sync and refresh
            self._sync_from_boundary_manager()
            if "wall_detection_results" in new_analysis:
                self.boundary_analysis["wall_detection_results"] = new_analysis[
                    "wall_detection_results"
                ]

        self.logger.info("Wall detection refinement complete")

    def _log_boundary_state(self, context: str = ""):
        """Log current boundary state for debugging."""
        stats = self.boundary_manager.get_stats()
        self.logger.debug(f"=== Boundary State {context} ===")
        self.logger.debug(f"Total boundaries: {stats['total_boundaries']}")
        self.logger.debug(f"By type: {stats['boundaries_by_type']}")
        self.logger.debug(f"Missing markers: {stats['missing_markers']}")
        self.logger.debug(f"Completion: {stats['completion_percentage']:.1f}%")

        # Check for consistency issues
        issues = self.boundary_manager.validate_consistency()
        if issues:
            self.logger.warning(f"Consistency issues found: {issues}")
        self.logger.debug("=== End Boundary State ===")

    def interpolate_missing_boundaries(
        self,
        detected_boundaries: List[Tuple[int, int, int, int]],
        vertical_constraints: Tuple[int, int],
        scale_px_per_cm: float,
        config: Dict[str, Any],
        image_shape: Tuple[int, int],
        boundary_to_marker: Dict[int, int] = None,
    ) -> Dict[str, Any]:
        """
        Interpolate missing compartment boundaries, honoring default & override spacings,
        and center them within each detected gap.
        """
        top_y, bottom_y = vertical_constraints

        # 1) Config → px values
        cw_cm = config.get("compartment_width_cm", 2.0)
        comp_w = int(cw_cm * scale_px_per_cm)

        default_mm = config.get("compartment_spacing_mm", 3.0)
        default_sp = int((default_mm / 10.0) * scale_px_per_cm)

        overrides = config.get("compartment_spacing_overrides", {})
        # overrides keys expected like "10-11": mm

        total = config.get("compartment_count", 20)
        min_mid, max_mid = 4, 3 + total  # valid marker IDs

        # 2) Sort existing boundaries L→R
        sorted_idx_bounds = sorted(
            enumerate(detected_boundaries), key=lambda ib: ib[1][0]  # sort by x1
        )
        existing = []
        for orig_i, (x1, y1, x2, y2) in sorted_idx_bounds:
            mid = boundary_to_marker.get(orig_i)
            if mid is not None and min_mid <= mid <= max_mid:
                existing.append((mid, x1, x2))

        # 3) Which markers are *actually* still missing?
        truly_missing = set(self.boundary_manager.get_missing_markers())

        gaps = []
        new_bounds = []
        new_ids = []

        # 4) For each interior gap, compute override or default spacing
        for i in range(len(existing) - 1):
            mid1, x1b = existing[i][0], existing[i][2]
            mid2, x2a = existing[i + 1][0], existing[i + 1][1]
            gap_span = x2a - x1b
            count = mid2 - mid1 - 1
            if count <= 0:
                continue

            # pick spacing override if present
            key = f"{mid1-3}-{mid2-3}"  # e.g. "10-11"
            sp_mm = overrides.get(key, default_mm)
            sp_px = int((sp_mm / 10.0) * scale_px_per_cm)

            gaps.append(
                {
                    "start_x": x1b,
                    "end_x": x2a,
                    "gap_size": gap_span,
                    "expected_compartments": count,
                    "after_compartment": mid1 - 3,
                    "before_compartment": mid2 - 3,
                    "spacing_px": sp_px,
                }
            )

            # total width needed = compartments + (count+1) gaps
            total_width = count * comp_w + (count + 1) * sp_px
            slack = gap_span - total_width
            # start offset so we center that group
            start = x1b + max(0, slack // 2) + sp_px

            for j in range(count):
                marker = mid1 + j + 1
                if marker not in truly_missing:
                    continue
                x1n = int(start + j * (comp_w + sp_px))
                x2n = x1n + comp_w
                # clamp
                x1n = max(0, x1n)
                x2n = min(image_shape[1], x2n)
                new_bounds.append((x1n, top_y, x2n, bottom_y))
                new_ids.append(marker)

        # 5) Edge‐cases: before first & after last
        if existing:
            first_mid, first_x1 = existing[0][0], existing[0][1]
            # left‐side
            for m in range(first_mid - 1, min_mid - 1, -1):
                if m not in truly_missing:
                    continue
                # each left step uses default_sp
                x2n = first_x1 - (first_mid - m) * (comp_w + default_sp)
                x1n = x2n - comp_w
                if x2n <= 0:
                    break
                new_bounds.insert(0, (x1n, top_y, x2n, bottom_y))
                new_ids.insert(0, m)

            # right‐side
            last_mid, last_x2 = existing[-1][0], existing[-1][2]
            for m in range(last_mid + 1, max_mid + 1):
                if m not in truly_missing:
                    continue
                x1n = last_x2 + (m - last_mid) * (comp_w + default_sp)
                x2n = x1n + comp_w
                if x1n >= image_shape[1]:
                    break
                new_bounds.append((x1n, top_y, x2n, bottom_y))
                new_ids.append(m)

        self.logger.info(
            f"Built {len(new_bounds)} interpolated boxes for markers {new_ids}"
        )
        return {
            "interpolated_boundaries": new_bounds,
            "interpolated_marker_ids": new_ids,
            "gap_analysis": gaps,
        }

    def _on_canvas_leave(self, event):
        """Handle mouse leaving the canvas."""
        try:
            # Hide zoom lens if visible
            if hasattr(self, "_zoom_lens") and self._zoom_lens:
                self._zoom_lens.withdraw()

            if hasattr(self, "_zoom_lens_flipped") and self._zoom_lens_flipped:
                self._zoom_lens_flipped.withdraw()

            # Clear temporary point
            self.temp_point = None

            # Update visualization to remove preview
            self._update_visualization()
        except Exception as e:
            self.logger.error(f"Error handling canvas leave: {str(e)}")

    def _on_canvas_right_click(self, event):
        """Handle right-click for flipped zoom lens in metadata mode."""
        if self.current_mode == self.MODE_METADATA:
            self._render_zoom(self.canvas, event, flipped=True)

    def _render_zoom(self, canvas, event, flipped=False):
        """
        Render the zoom lens for the current position.

        Args:
            canvas: The canvas widget
            event: Mouse event
            flipped: Whether to display flipped view
        """
        if self.source_image is None:
            return

        # Define zoom lens dimensions
        if flipped:
            zoom_width = getattr(self, "_zoom_width_flipped", 500)
            zoom_height = getattr(self, "_zoom_height", 350)
        else:
            zoom_width = getattr(self, "_zoom_width", 250)
            zoom_height = getattr(self, "_zoom_height", 350)
        zoom_scale = 2  # Magnification factor

        # Calculate image coordinates TODO - CHECK THIS IS RIGHT
        x = int((event.x - self.canvas_offset_x) / self.scale_ratio)
        y = int((event.y - self.canvas_offset_y) / self.scale_ratio)

        # Get image dimensions
        h, w = self.source_image.shape[:2]

        # ===================================================
        # MODIFIED: Different crop logic for missing boundaries mode
        # ===================================================
        if (
            self.current_mode == self.MODE_MISSING_BOUNDARIES
            and not self.annotation_complete
        ):
            # For missing boundaries mode, show the compartment placement area
            # Extract region around where compartment will be placed
            zoom_radius_x = (
                zoom_width // 4
            )  # Extract a quarter of the zoom window width
            zoom_radius_y = (
                zoom_height // 4
            )  # Extract a quarter of the zoom window height

            # Calculate region to extract - center on x and show full height of compartment
            center_x = x  # Where the compartment will be placed
            center_y = (self.top_y + self.bottom_y) // 2

            # Define extraction region
            left = max(0, center_x - zoom_radius_x)
            right = min(w, left + zoom_radius_x * 2)

            # Extract the vertical range of the compartment plus some margin
            vertical_height = self.bottom_y - self.top_y
            top = max(0, center_y - vertical_height // 2 - 20)
            bottom = min(h, top + vertical_height + 40)

            # Start with clean source image
            img = self.source_image.copy()
            # Extract the region first from the clean image
            region = img[top:bottom, left:right].copy()

            # Now draw preview elements ONLY on the extracted region
            if self.current_index < len(self.missing_marker_ids):
                current_id = self.missing_marker_ids[self.current_index]

                if current_id != 24:  # Only for compartments, not metadata marker
                    would_overlap = self._would_overlap_existing(x)
                    half_width = self.config.get("compartment_width_cm", 2.0) // 2

                    # Calculate positions relative to the extracted region
                    x1_global = max(0, x - half_width)
                    y1_global = self.top_y
                    x2_global = min(w - 1, x + half_width)
                    y2_global = self.bottom_y

                    # Convert to region coordinates
                    x1 = x1_global - left
                    y1 = y1_global - top
                    x2 = x2_global - left
                    y2 = y2_global - top

                    # Ensure coordinates are within the region bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(region.shape[1] - 1, x2)
                    y2 = min(region.shape[0] - 1, y2)

                    # Draw preview rectangle with overlap indication
                    preview_color = (
                        (0, 0, 255) if would_overlap else (255, 0, 255)
                    )  # Red if would overlap
                    cv2.rectangle(
                        region, (x1, y1), (x2, y2), preview_color, 2
                    )  # Thinner line for zoom

                    # Add current compartment number
                    display_number = self.marker_to_compartment.get(
                        current_id, current_id - 3
                    )

                    mid_x = (x1 + x2) // 2
                    mid_y = (y1 + y2) // 2

                    # Smaller text for zoom view
                    text_size = cv2.getTextSize(
                        f"{display_number}m", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                    )[0]
                    cv2.rectangle(
                        region,
                        (mid_x - text_size[0] // 2 - 3, mid_y - text_size[1] // 2 - 3),
                        (mid_x + text_size[0] // 2 + 3, mid_y + text_size[1] // 2 + 3),
                        (0, 0, 0),
                        -1,
                    )

                    # Add text with 'm' suffix
                    cv2.putText(
                        region,
                        f"{display_number}m",
                        (mid_x - text_size[0] // 2, mid_y + text_size[1] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        preview_color,
                        2,
                    )

        else:
            # For metadata mode, use cursor-centered crop from current visualization
            crop_w, crop_h = zoom_width // zoom_scale, zoom_height // zoom_scale

            # For flipped view, invert the Y coordinate to make movement feel natural
            if flipped:
                # Invert Y position - when mouse is at top, show bottom and vice versa
                y = h - y

            # Calculate crop area, ensuring it's within image bounds
            left = max(0, x - crop_w // 2)
            upper = max(0, y - crop_h // 2)
            right = min(w, left + crop_w)
            lower = min(h, upper + crop_h)

            # Handle edge cases where crop area might be invalid
            if left >= right or upper >= lower:
                return  # Skip if invalid crop area

            # Extract region from current visualization (with all annotations)
            region = self.display_image[upper:lower, left:right]

        try:
            # Convert to RGB
            if len(region.shape) == 2:  # Grayscale
                region_rgb = cv2.cvtColor(region, cv2.COLOR_GRAY2RGB)
            else:
                region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)

            # Convert to PIL image
            pil_region = Image.fromarray(region_rgb)

            # Apply flipping if needed
            if flipped:
                pil_region = ImageOps.flip(pil_region)

            # Resize for zoom effect
            zoomed = pil_region.resize((zoom_width, zoom_height), Image.LANCZOS)

            # Convert to PhotoImage
            tk_img = ImageTk.PhotoImage(zoomed)

            # Update the appropriate zoom lens
            if flipped:
                self._zoom_img_flipped = tk_img
                self._zoom_canvas_flipped.delete("all")
                self._zoom_canvas_flipped.create_image(0, 0, anchor=tk.NW, image=tk_img)

                # Add crosshair
                center_x, center_y = zoom_width // 2, zoom_height // 2
                self._zoom_canvas_flipped.create_line(
                    0,
                    center_y,
                    zoom_width,
                    center_y,
                    fill="red",
                    width=1,
                    tags="crosshair",
                )
                self._zoom_canvas_flipped.create_line(
                    center_x,
                    0,
                    center_x,
                    zoom_height,
                    fill="red",
                    width=1,
                    tags="crosshair",
                )
                self._zoom_canvas_flipped.create_oval(
                    center_x - 3,
                    center_y - 3,
                    center_x + 3,
                    center_y + 3,
                    fill="red",
                    outline="red",
                    tags="crosshair",
                )

                # Position flipped lens (centered on cursor in metadata mode)
                cursor_x = canvas.winfo_rootx() + event.x
                cursor_y = canvas.winfo_rooty() + event.y

                # Get the wider dimensions for flipped lens
                flipped_width = getattr(self, "_zoom_width_flipped", 500)

                # Center the zoom window on the cursor
                global_x = cursor_x - (flipped_width // 2)
                global_y = cursor_y - (zoom_height // 2)

                # Ensure the zoom lens stays on screen
                screen_width = self.dialog.winfo_screenwidth()
                screen_height = self.dialog.winfo_screenheight()

                # Add margins to keep lens fully visible
                margin = 10
                global_x = max(
                    margin, min(global_x, screen_width - flipped_width - margin)
                )
                global_y = max(
                    margin, min(global_y, screen_height - zoom_height - margin)
                )

                self._zoom_lens_flipped.geometry(
                    f"{flipped_width}x{zoom_height}+{global_x}+{global_y}"
                )
                self._zoom_lens_flipped.deiconify()
            else:
                self._zoom_img_ref = tk_img
                self._zoom_canvas.delete("all")
                self._zoom_canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)

                # Add crosshair
                center_x, center_y = zoom_width // 2, zoom_height // 2
                self._zoom_canvas.create_line(
                    0,
                    center_y,
                    zoom_width,
                    center_y,
                    fill="red",
                    width=1,
                    tags="crosshair",
                )
                self._zoom_canvas.create_line(
                    center_x,
                    0,
                    center_x,
                    zoom_height,
                    fill="red",
                    width=1,
                    tags="crosshair",
                )
                self._zoom_canvas.create_oval(
                    center_x - 3,
                    center_y - 3,
                    center_x + 3,
                    center_y + 3,
                    fill="red",
                    outline="red",
                    tags="crosshair",
                )

                # Calculate zoom lens position
                if self.current_mode == self.MODE_MISSING_BOUNDARIES:
                    # Fixed Y position above compartments (like old code)
                    # Calculate Y position based on top boundary in canvas coordinates
                    canvas_top_y = int(
                        self.top_y * self.scale_ratio + self.canvas_offset_y
                    )
                    lens_y = max(
                        10, canvas.winfo_rooty() + canvas_top_y - zoom_height - 30
                    )

                    # Center horizontally on cursor
                    screen_width = self.dialog.winfo_screenwidth()
                    lens_x = min(
                        max(10, canvas.winfo_rootx() + event.x - zoom_width // 2),
                        screen_width - zoom_width - 10,
                    )
                else:
                    # Center the zoom lens on the cursor for metadata mode
                    # This makes the red crosshair dot align with the actual cursor position
                    cursor_x = canvas.winfo_rootx() + event.x
                    cursor_y = canvas.winfo_rooty() + event.y

                    # Center the zoom window on the cursor
                    lens_x = cursor_x - (zoom_width // 2)
                    lens_y = cursor_y - (zoom_height // 2)

                    # Get current screen bounds (the screen the dialog is on)
                    screen_x = self.dialog.winfo_vrootx()
                    screen_y = self.dialog.winfo_vrooty()
                    screen_width = self.dialog.winfo_vrootwidth()
                    screen_height = self.dialog.winfo_vrootheight()

                    # Add margins to keep lens fully visible on the same screen as dialog
                    margin = 10
                    lens_x = max(
                        screen_x + margin,
                        min(lens_x, screen_x + screen_width - zoom_width - margin),
                    )
                    lens_y = max(
                        screen_y + margin,
                        min(lens_y, screen_y + screen_height - zoom_height - margin),
                    )

                self._zoom_lens.geometry(
                    f"{zoom_width}x{zoom_height}+{lens_x}+{lens_y}"
                )
                self._zoom_lens.deiconify()

        except Exception as e:
            self.logger.error(f"Error rendering zoom lens: {str(e)}")
            # Don't print full traceback for zoom rendering

    def _create_enhanced_status_display(self, parent_frame):
        """Create an enhanced status display with metadata and existing data visualization."""
        # Main container for enhanced status
        status_container = ttk.Frame(parent_frame, style="Content.TFrame")
        status_container.pack(fill=tk.X, pady=(2, 5))

        # Top row - Metadata display with EOH info
        metadata_display_frame = tk.Frame(
            status_container,
            bg=self.theme_colors.get("field_bg", "#2d2d2d"),
            relief=tk.FLAT,
            highlightbackground=self.theme_colors.get("field_border", "#3f3f3f"),
            highlightthickness=1,
        )
        metadata_display_frame.pack(fill=tk.X, padx=20, pady=(0, 5))

        # Metadata text variables
        self.metadata_display_var = tk.StringVar(value="No metadata entered")
        self.eoh_display_var = tk.StringVar(value="")

        # Metadata label
        self.metadata_display_label = tk.Label(
            metadata_display_frame,
            textvariable=self.metadata_display_var,
            font=("Arial", 14, "bold"),
            bg=self.theme_colors.get("field_bg", "#2d2d2d"),
            fg=self.theme_colors.get("text", "#e0e0e0"),
            padx=15,
            pady=8,
        )
        self.metadata_display_label.pack(side=tk.LEFT)

        # EOH label
        self.eoh_label = tk.Label(
            metadata_display_frame,
            textvariable=self.eoh_display_var,
            font=("Arial", 12),
            bg=self.theme_colors.get("field_bg", "#2d2d2d"),
            fg=self.theme_colors.get("accent_green", "#4CAF50"),
            padx=15,
            pady=8,
        )
        self.eoh_label.pack(side=tk.LEFT)

        # Existing data visualization frame
        self.existing_data_frame = tk.Frame(
            status_container,
            bg=self.theme_colors.get("background", "#1e1e1e"),
            height=40,
        )
        self.existing_data_frame.pack(fill=tk.X, padx=20, pady=(0, 5))
        self.existing_data_frame.pack_propagate(False)

        # Regular status label below
        self.status_label = tk.Label(
            status_container,
            textvariable=self.status_var,
            font=("Arial", 12),
            bg=self.theme_colors.get("field_bg", "#2d2d2d"),
            fg=self.theme_colors.get("accent_green", "#4CAF50"),
            padx=15,
            pady=5,
            relief=tk.FLAT,
            highlightbackground=self.theme_colors.get("accent_green", "#4CAF50"),
            highlightthickness=1,
        )
        self.status_label.pack(fill=tk.X, padx=20, pady=2)

        # Initial update
        self._update_metadata_display()
        self._update_existing_data_visualization()

    def _update_metadata_display(self):
        """Update the metadata display with current values and EOH info."""
        try:
            hole_id = self.hole_id.get().strip()
            depth_from_str = self.depth_from.get().strip()
            depth_to_str = self.depth_to.get().strip()

            # Update metadata text
            if hole_id and depth_from_str and depth_to_str:
                self.metadata_display_var.set(
                    f"{hole_id} - {depth_from_str}m to {depth_to_str}m"
                )

                # Check depth validation if we have app reference with depth validator
                if (
                    self.app
                    and hasattr(self.app, "depth_validator")
                    and self.app.depth_validator.is_loaded
                ):
                    valid_range = self.app.depth_validator.get_valid_range(hole_id)
                    if valid_range:
                        min_depth, max_depth = valid_range
                        self.eoh_display_var.set(f"(EOH: {max_depth}m)")
                        self.eoh_label.config(
                            fg=self.theme_colors.get("accent_green", "#4CAF50")
                        )
                    else:
                        self.eoh_display_var.set("(Hole not found in database)")
                        self.eoh_label.config(
                            fg=self.theme_colors.get("accent_yellow", "#FFEB3B")
                        )
                else:
                    self.eoh_display_var.set("(EOH validation not available)")
                    self.eoh_label.config(fg=self.theme_colors.get("text", "#e0e0e0"))
            else:
                self.metadata_display_var.set("Enter hole ID and depths")
                self.eoh_display_var.set("")

        except Exception as e:
            self.logger.error(f"Error updating metadata display: {e}")

    def _update_existing_data_visualization(self):
        """Update the visualization showing existing processed originals."""
        # Clear existing widgets
        for widget in self.existing_data_frame.winfo_children():
            widget.destroy()

        hole_id = self.hole_id.get().strip()
        if not hole_id or len(hole_id) < 6:
            # Show placeholder
            placeholder = tk.Label(
                self.existing_data_frame,
                text="Enter a valid Hole ID to see existing data",
                font=("Arial", 10),
                bg=self.theme_colors.get("background", "#1e1e1e"),
                fg=self.theme_colors.get("text", "#808080"),
            )
            placeholder.pack(expand=True)
            return

        # Check for existing files
        existing_depths = self._scan_for_existing_originals(hole_id)

        if not existing_depths:
            # No data found
            no_data_label = tk.Label(
                self.existing_data_frame,
                text=f"No existing data found for {hole_id}",
                font=("Arial", 10),
                bg=self.theme_colors.get("background", "#1e1e1e"),
                fg=self.theme_colors.get("text", "#808080"),
            )
            no_data_label.pack(expand=True)
            return

        # Create visualization
        viz_container = tk.Frame(
            self.existing_data_frame, bg=self.theme_colors.get("background", "#1e1e1e")
        )
        viz_container.pack(expand=True)

        # Add label
        tk.Label(
            viz_container,
            text="Existing originals: ",
            font=("Arial", 10),
            bg=self.theme_colors.get("background", "#1e1e1e"),
            fg=self.theme_colors.get("text", "#e0e0e0"),
        ).pack(side=tk.LEFT, padx=(0, 10))

        # Create depth boxes
        for depth_from, depth_to in sorted(existing_depths):
            box_frame = tk.Frame(
                viz_container,
                bg=self.theme_colors.get("accent_green", "#4CAF50"),
                width=50,
                height=25,
                relief=tk.RAISED,
                borderwidth=1,
            )
            box_frame.pack(side=tk.LEFT, padx=2)
            box_frame.pack_propagate(False)

            depth_label = tk.Label(
                box_frame,
                text=f"{depth_from}-{depth_to}",
                font=("Arial", 8),
                bg=self.theme_colors.get("accent_green", "#4CAF50"),
                fg="white",
            )
            depth_label.place(relx=0.5, rely=0.5, anchor="center")

    def _scan_for_existing_originals(self, hole_id):
        """Scan for existing processed original images for this hole."""
        existing_depths = []

        if not self.file_manager:
            return existing_depths

        project_code = hole_id[:2].upper() if len(hole_id) >= 2 else ""

        # Check both approved and rejected locations
        locations_to_check = []

        # Local paths
        if hasattr(self.file_manager, "dir_structure"):
            approved_dir = self.file_manager.dir_structure.get("approved_originals")
            rejected_dir = self.file_manager.dir_structure.get("rejected_originals")

            if approved_dir:
                locations_to_check.append(
                    os.path.join(approved_dir, project_code, hole_id)
                )
            if rejected_dir:
                locations_to_check.append(
                    os.path.join(rejected_dir, project_code, hole_id)
                )

        # Shared paths
        if (
            hasattr(self.file_manager, "shared_paths")
            and self.file_manager.shared_paths
        ):
            shared_approved = self.file_manager.get_shared_path(
                "approved_originals", create_if_missing=False
            )
            shared_rejected = self.file_manager.get_shared_path(
                "rejected_originals", create_if_missing=False
            )

            if shared_approved:
                locations_to_check.append(
                    os.path.join(shared_approved, project_code, hole_id)
                )
            if shared_rejected:
                locations_to_check.append(
                    os.path.join(shared_rejected, project_code, hole_id)
                )

        # Scan each location
        pattern = re.compile(
            rf"^{re.escape(hole_id)}_(\d+)-(\d+)_.*\.(jpg|jpeg|png|tiff|tif)$",
            re.IGNORECASE,
        )

        found_depths = set()
        for location in locations_to_check:
            if os.path.exists(location):
                try:
                    for filename in os.listdir(location):
                        match = pattern.match(filename)
                        if match:
                            depth_from = int(match.group(1))
                            depth_to = int(match.group(2))
                            found_depths.add((depth_from, depth_to))
                except Exception as e:
                    self.logger.debug(f"Error scanning {location}: {e}")

        return sorted(list(found_depths))

    def _initial_visualization_update(self):
        """Perform initial visualization update after canvas is properly sized."""
        try:
            # Update canvas dimensions
            self.canvas.update_idletasks()

            # Now update visualization with proper canvas dimensions
            self._update_visualization()

            # Update mode display
            self._update_mode_indicator()
        except Exception as e:
            self.logger.error(f"Error in initial visualization update: {str(e)}")

    # OPTIMIZED: Only update dynamic elements, use cached static visualization
    def _update_visualization(self, fast_mode=False, force_full_update=False):
        """
        Update the image visualization based on current mode and state.

        Args:
            fast_mode: If True, use faster updates for mouse motion (skips some elements)
            force_full_update: If True, force recreation of static elements
        """
        if self.source_image is None:
            return

        try:
            # Force cache invalidation if requested
            if force_full_update:
                self.static_viz_cache = None

            # Get or create static visualization
            viz_image = self._create_static_visualization()
            if viz_image is None:
                return

            # Draw walls overlay if enabled
            if getattr(self, "show_walls", False) and self.boundary_analysis:
                viz = self.boundary_analysis.get("wall_detection_results", {})
                # raw edges
                for x1, y1, x2, y2, color, t in viz.get("detected_edges", []):
                    cv2.line(viz_image, (x1, y1), (x2, y2), (0, 0, 255), 1)

            # Only add dynamic elements (temp_point preview) if needed
            if (
                self.current_mode == self.MODE_MISSING_BOUNDARIES
                and self.temp_point
                and not fast_mode
            ):
                if (
                    self.missing_marker_ids
                    and self.current_index < len(self.missing_marker_ids)
                    and not self.annotation_complete
                ):
                    # Add the preview overlay on top of static visualization
                    self._add_preview_overlay(viz_image)

            # Update instruction label based on current mode
            if hasattr(self, "instruction_label"):
                self.instruction_label.config(text=self._get_instruction_text())

            # Display the visualization
            self._display_visualization(viz_image)

        except Exception as e:
            self.logger.error(f"Error updating visualization: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _add_preview_overlay(self, viz_image):
        """Add the dynamic preview overlay to the visualization."""
        if (
            not self.temp_point
            or not self.missing_marker_ids
            or self.current_index >= len(self.missing_marker_ids)
        ):
            return

        img_height, img_width = viz_image.shape[:2]
        x = self.temp_point[0]
        current_id = self.missing_marker_ids[self.current_index]

        # Calculate boundary line positions
        left_top_y = self.top_y + self.left_height_offset
        right_top_y = self.top_y + self.right_height_offset
        left_bottom_y = self.bottom_y + self.left_height_offset
        right_bottom_y = self.bottom_y + self.right_height_offset

        # Calculate slopes
        if img_width > 0:
            top_slope = (right_top_y - left_top_y) / img_width
            bottom_slope = (right_bottom_y - left_bottom_y) / img_width
        else:
            top_slope = 0
            bottom_slope = 0

        # Calculate rotation angle
        if img_width > 0:
            dx = img_width
            dy = right_top_y - left_top_y
            rotation_angle = -np.arctan2(dy, dx) * 180 / np.pi
        else:
            rotation_angle = 0

        # Check if current marker is a corner marker (shouldn't happen but skip if so)
        if current_id in [0, 1, 2, 3]:
            # Skip preview for corner markers since they're handled automatically
            return

        else:  # Compartment marker
            # Calculate the compartment boundaries for preview
            half_width = self.config.get("compartment_width_cm", 2.0) // 2
            x1 = max(0, x - half_width)
            x2 = min(img_width - 1, x + half_width)
            would_overlap = self._would_overlap_existing(x)

            # Calculate y position at center_x using slope of boundary lines
            center_x = (x1 + x2) / 2

            # Calculate y-coordinates based on x-position using boundary lines
            center_top_y = int(left_top_y + (top_slope * center_x))
            center_bottom_y = int(left_bottom_y + (bottom_slope * center_x))

            # Calculate center of rectangle
            rect_center_x = center_x
            rect_center_y = (center_top_y + center_bottom_y) / 2

            # Preview color - magenta
            preview_color = (255, 0, 255)  # Magenta in BGR

            # Create rotated rectangle points for preview
            rect_corners = np.array(
                [
                    [x1, center_top_y],
                    [x2, center_top_y],
                    [x2, center_bottom_y],
                    [x1, center_bottom_y],
                ],
                dtype=np.float32,
            )

            # Apply same rotation as other compartments
            rotation_matrix = cv2.getRotationMatrix2D(
                (rect_center_x, rect_center_y), rotation_angle, 1.0
            )
            ones = np.ones(shape=(len(rect_corners), 1))
            rect_corners_homog = np.hstack([rect_corners, ones])
            rotated_corners = np.dot(rotation_matrix, rect_corners_homog.T).T

            # Convert to integer points for drawing
            rotated_corners = rotated_corners.astype(np.int32)

            # Draw the rotated rectangle preview
            cv2.polylines(viz_image, [rotated_corners], True, preview_color, 2)

            # Add current compartment depth with better visibility
            if self.current_index < len(self.missing_marker_ids):
                current_id = self.missing_marker_ids[self.current_index]
                depth_label = self.marker_to_compartment.get(current_id, current_id - 3)

                mid_x = rect_center_x
                mid_y = rect_center_y

                if would_overlap:
                    # Add warning text
                    warning_text = "OVERLAP!"
                    warning_size = cv2.getTextSize(
                        warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                    )[0]
                    cv2.rectangle(
                        viz_image,
                        (
                            int(mid_x) - warning_size[0] // 2 - 5,
                            int(mid_y) - 30 - warning_size[1],
                        ),
                        (int(mid_x) + warning_size[0] // 2 + 5, int(mid_y) - 20),
                        (0, 0, 0),
                        -1,
                    )
                    cv2.putText(
                        viz_image,
                        warning_text,
                        (int(mid_x) - warning_size[0] // 2, int(mid_y) - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )

                # Add black background for text visibility
                text_size = cv2.getTextSize(
                    f"{depth_label}m", cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3
                )[0]
                cv2.rectangle(
                    viz_image,
                    (
                        int(mid_x) - text_size[0] // 2 - 5,
                        int(mid_y) - text_size[1] // 2 - 5,
                    ),
                    (
                        int(mid_x) + text_size[0] // 2 + 5,
                        int(mid_y) + text_size[1] // 2 + 5,
                    ),
                    (0, 0, 0),
                    -1,
                )

                # Draw larger text with thicker outline
                cv2.putText(
                    viz_image,
                    f"{depth_label}m",
                    (int(mid_x) - text_size[0] // 2, int(mid_y) + text_size[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    preview_color,
                    3,
                )

    def _display_visualization(self, viz_image):
        """Display the visualization image on the canvas without recreating it."""
        try:
            # Convert to RGB for PIL
            viz_rgb = cv2.cvtColor(viz_image, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image
            pil_image = Image.fromarray(viz_rgb)

            # Get the current canvas size
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            # If canvas not yet drawn, use reasonable defaults
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 800
                canvas_height = 600

            # Calculate scaling
            img_width, img_height = pil_image.size
            scale_width = (canvas_width - 20) / img_width
            scale_height = (canvas_height - 20) / img_height
            scale = min(scale_width, scale_height)

            if scale > 2.0:
                scale = 2.0

            # Calculate new dimensions
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)

            # Resize image
            resized_image = pil_image.resize((new_width, new_height), Image.LANCZOS)

            # Create PhotoImage
            self.photo_image = ImageTk.PhotoImage(resized_image)

            # Update canvas
            self.canvas.delete("all")
            x_offset = (canvas_width - new_width) // 2
            y_offset = (canvas_height - new_height) // 2

            self.canvas.create_image(
                x_offset, y_offset, anchor=tk.NW, image=self.photo_image
            )

            # Store parameters
            self.scale_ratio = scale
            self.canvas_offset_x = x_offset
            self.canvas_offset_y = y_offset
            self.current_viz = viz_image
            self.display_image = viz_image

            # Update static zoom views if in adjustment mode
            if self.current_mode == self.MODE_ADJUST_BOUNDARIES:
                self._update_static_zoom_views()

        except Exception as e:
            self.logger.error(f"Error displaying visualization: {str(e)}")

    # NEW METHOD: Create static visualization that doesn't change during mouse movement - draws markers with their lengths, scale bar, etc.
    def _create_static_visualization(self):
        """
        Create the static visualization with all elements that don't change during interaction.
        This includes: markers, scale bar, boundaries, labels, etc.
        Returns the static image that can be cached.
        """
        if self.source_image is None:
            return None

        # Get current parameters to check if cache is valid
        current_params = {
            "top_y": self.top_y,
            "bottom_y": self.bottom_y,
            "left_height_offset": self.left_height_offset,
            "right_height_offset": self.right_height_offset,
            "markers": str(self.markers),  # Convert to string for comparison
            "detected_boundaries": str(self.detected_boundaries),
            "result_boundaries": str(self.result_boundaries),
            "mode": self.current_mode,
        }

        # Check if we can use cached version
        if (
            self.static_viz_cache is not None
            and self.static_viz_params == current_params
        ):
            return self.static_viz_cache.copy()

        # Need to recreate static visualization
        # Start with clean original image
        static_viz = self.source_image.copy()

        # Get image dimensions
        img_height, img_width = static_viz.shape[:2]

        # Extract scale data early in the method
        scale_px_per_cm = None
        scale_data = None

        # Try different sources for scale data
        if hasattr(self, "scale_data") and self.scale_data:
            scale_data = self.scale_data
            scale_px_per_cm = scale_data.get("scale_px_per_cm")
        elif hasattr(self, "boundary_analysis") and self.boundary_analysis:
            scale_px_per_cm = self.boundary_analysis.get("scale_px_per_cm")
            scale_data = {"scale_px_per_cm": scale_px_per_cm}
        # Draw markers with scale measurements
        if self.markers:
            # Get scale data if available
            scale_data = None
            scale_px_per_cm = None
            if hasattr(self, "scale_data") and self.scale_data:
                scale_data = self.scale_data
                scale_px_per_cm = scale_data.get("scale_px_per_cm", None)

            # Draw each marker with scale measurement lines
            for marker_id, corners in self.markers.items():
                # Convert corners to int
                corners_int = corners.astype(np.int32)

                # Find this marker in scale measurements if available
                marker_measurements = None
                if scale_data and "marker_measurements" in scale_data:
                    for measurement in scale_data["marker_measurements"]:
                        if measurement["marker_id"] == marker_id:
                            marker_measurements = measurement
                            break

                if marker_measurements and scale_px_per_cm:
                    # We have scale data - draw edges based on individual validity
                    edge_lengths = marker_measurements.get("edge_lengths", [])
                    diagonal_lengths = marker_measurements.get("diagonal_lengths", [])
                    scales = marker_measurements.get("scales", [])

                    # Determine which measurements are valid
                    physical_size = marker_measurements.get("physical_size_cm", 2.0)

                    # Draw each edge with its validity color
                    for i in range(4):  # 4 edges
                        if i < len(edge_lengths) and i < len(scales):
                            # Calculate expected length in pixels
                            expected_px = physical_size * scale_px_per_cm
                            actual_px = edge_lengths[i]

                            # Check if this edge is within tolerance (5 pixels)
                            tolerance_pixels = 5.0
                            is_valid = abs(actual_px - expected_px) <= tolerance_pixels

                            # Choose color based on validity
                            edge_color = (
                                (0, 255, 0) if is_valid else (0, 0, 255)
                            )  # Green if valid, red if invalid

                            # Draw the edge
                            start_corner = i
                            end_corner = (i + 1) % 4
                            cv2.line(
                                static_viz,
                                tuple(corners_int[start_corner]),
                                tuple(corners_int[end_corner]),
                                edge_color,
                                2,
                            )

                            # Show measurement on edge
                            if scale_px_per_cm:
                                edge_center_x = (
                                    corners_int[start_corner][0]
                                    + corners_int[end_corner][0]
                                ) // 2
                                edge_center_y = (
                                    corners_int[start_corner][1]
                                    + corners_int[end_corner][1]
                                ) // 2

                                # Offset text based on edge position
                                text_offset_x = 0
                                text_offset_y = -10 if i == 0 else 10 if i == 2 else 0
                                text_offset_x = -40 if i == 3 else 40 if i == 1 else 0

                                # Format text
                                px_text = f"{actual_px:.0f}px"
                                measurement_text = f"{px_text}"

                                # Draw with background
                                text_size = cv2.getTextSize(
                                    measurement_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                                )[0]
                                cv2.rectangle(
                                    static_viz,
                                    (
                                        edge_center_x
                                        + text_offset_x
                                        - text_size[0] // 2
                                        - 2,
                                        edge_center_y
                                        + text_offset_y
                                        - text_size[1]
                                        - 2,
                                    ),
                                    (
                                        edge_center_x
                                        + text_offset_x
                                        + text_size[0] // 2
                                        + 2,
                                        edge_center_y + text_offset_y + 2,
                                    ),
                                    (0, 0, 0),
                                    -1,
                                )
                                cv2.putText(
                                    static_viz,
                                    measurement_text,
                                    (
                                        edge_center_x
                                        + text_offset_x
                                        - text_size[0] // 2,
                                        edge_center_y + text_offset_y,
                                    ),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4,
                                    (255, 255, 255),
                                    1,
                                )

                    # Draw diagonals if available
                    if len(diagonal_lengths) >= 2 and len(scales) >= 6:
                        diagonals = [(0, 2), (1, 3)]

                        for i, (start, end) in enumerate(diagonals):
                            if i < len(diagonal_lengths):
                                # Calculate expected diagonal length
                                expected_diag = (
                                    physical_size * np.sqrt(2) * scale_px_per_cm
                                )
                                actual_diag = diagonal_lengths[i]

                                # Check validity
                                tolerance_pixels = 5.0
                                is_valid = (
                                    abs(actual_diag - expected_diag) / expected_diag
                                    < tolerance_pixels
                                )

                                # Choose color
                                diag_color = (0, 255, 0) if is_valid else (0, 0, 255)

                                # Draw diagonal with dashed line
                                start_pt = corners_int[start]
                                end_pt = corners_int[end]

                                # Calculate line segments for dashed effect
                                num_dashes = 10
                                for j in range(0, num_dashes, 2):
                                    t1 = j / num_dashes
                                    t2 = min((j + 1) / num_dashes, 1.0)
                                    pt1 = (
                                        int(
                                            start_pt[0] + t1 * (end_pt[0] - start_pt[0])
                                        ),
                                        int(
                                            start_pt[1] + t1 * (end_pt[1] - start_pt[1])
                                        ),
                                    )
                                    pt2 = (
                                        int(
                                            start_pt[0] + t2 * (end_pt[0] - start_pt[0])
                                        ),
                                        int(
                                            start_pt[1] + t2 * (end_pt[1] - start_pt[1])
                                        ),
                                    )
                                    cv2.line(static_viz, pt1, pt2, diag_color, 1)

                    # Draw corner points
                    for corner in corners_int:
                        cv2.circle(static_viz, tuple(corner), 3, (255, 255, 255), -1)

                else:
                    # No scale data - just draw the marker outline
                    cv2.polylines(
                        static_viz,
                        [corners_int.reshape(-1, 1, 2)],
                        True,
                        (0, 255, 255),
                        2,
                    )

                    # Draw corner points
                    for corner in corners_int:
                        cv2.circle(static_viz, tuple(corner), 3, (0, 255, 255), -1)

        # Add scale bar at bottom right
        if scale_px_per_cm:
            # Calculate scale bar dimensions
            scale_bar_length_cm = 10  # 10cm scale bar
            scale_bar_length_px = int(scale_bar_length_cm * scale_px_per_cm)
            scale_bar_height = 20
            margin = 20

            # Position at bottom right
            bar_x = img_width - margin - scale_bar_length_px
            bar_y = (
                img_height - margin - scale_bar_height - 30
            )  # Extra space for labels

            # Draw white background for scale bar area
            bg_padding = 10
            cv2.rectangle(
                static_viz,
                (bar_x - bg_padding, bar_y - 20 - bg_padding),
                (
                    bar_x + scale_bar_length_px + bg_padding,
                    bar_y + scale_bar_height + 30 + bg_padding,
                ),
                (255, 255, 255),
                -1,
            )

            # Draw checkered scale bar (1cm segments)
            for i in range(scale_bar_length_cm):
                segment_start = bar_x + int(i * scale_px_per_cm)
                segment_end = bar_x + int((i + 1) * scale_px_per_cm)

                # Alternate black and white
                color = (0, 0, 0) if i % 2 == 0 else (200, 200, 200)
                cv2.rectangle(
                    static_viz,
                    (segment_start, bar_y),
                    (segment_end, bar_y + scale_bar_height),
                    color,
                    -1,
                )

            # Draw border around scale bar
            cv2.rectangle(
                static_viz,
                (bar_x, bar_y),
                (bar_x + scale_bar_length_px, bar_y + scale_bar_height),
                (0, 0, 0),
                2,
            )

            # Add labels
            cv2.putText(
                static_viz,
                "0cm",
                (bar_x, bar_y + scale_bar_height + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

            text_size = cv2.getTextSize("10cm", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.putText(
                static_viz,
                "10cm",
                (
                    bar_x + scale_bar_length_px - text_size[0],
                    bar_y + scale_bar_height + 20,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

            # Scale info in the middle above the bar
            if scale_data:
                confidence = scale_data.get("confidence", 0.0)
                scale_text = (
                    f"{scale_px_per_cm:.1f} px/cm ({confidence:.0%} confidence)"
                )
            else:
                scale_text = f"{scale_px_per_cm:.1f} px/cm"

            text_size = cv2.getTextSize(scale_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = bar_x + (scale_bar_length_px - text_size[0]) // 2
            cv2.putText(
                static_viz,
                scale_text,
                (text_x, bar_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )

        # Draw top and bottom boundary lines with slope based on side offsets
        left_top_y = self.top_y + self.left_height_offset
        right_top_y = self.top_y + self.right_height_offset

        left_bottom_y = self.bottom_y + self.left_height_offset
        right_bottom_y = self.bottom_y + self.right_height_offset

        # Draw top boundary line
        cv2.line(
            static_viz, (0, left_top_y), (img_width, right_top_y), (0, 255, 0), 2
        )  # Green line for top boundary

        # Draw bottom boundary line
        cv2.line(
            static_viz, (0, left_bottom_y), (img_width, right_bottom_y), (0, 255, 0), 2
        )  # Green line for bottom boundary

        # Calculate rotation angle based on the slope of the boundary lines
        if img_width > 0:
            dx = img_width
            dy = right_top_y - left_top_y
            rotation_angle = -np.arctan2(dy, dx) * 180 / np.pi
        else:
            rotation_angle = 0

        # Calculate slopes for top and bottom boundary (used for compartment positioning)
        if img_width > 0:
            top_slope = (right_top_y - left_top_y) / img_width
            bottom_slope = (right_bottom_y - left_bottom_y) / img_width
        else:
            top_slope = 0
            bottom_slope = 0

        # Draw detected compartment boundaries
        for i, (x1, _, x2, _) in enumerate(self.detected_boundaries):
            # Calculate center position of compartment
            center_x = (x1 + x2) / 2

            # Calculate y position at center_x using slope of boundary lines
            center_top_y = int(left_top_y + (top_slope * center_x))
            center_bottom_y = int(left_bottom_y + (bottom_slope * center_x))

            # Calculate center of rectangle
            rect_center_x = center_x
            rect_center_y = (center_top_y + center_bottom_y) / 2

            # Create rotated rectangle points
            rotation_matrix = cv2.getRotationMatrix2D(
                (rect_center_x, rect_center_y), rotation_angle, 1.0
            )

            rect_corners = np.array(
                [
                    [x1, center_top_y],
                    [x2, center_top_y],
                    [x2, center_bottom_y],
                    [x1, center_bottom_y],
                ],
                dtype=np.float32,
            )

            ones = np.ones(shape=(len(rect_corners), 1))
            rect_corners_homog = np.hstack([rect_corners, ones])
            rotated_corners = np.dot(rotation_matrix, rect_corners_homog.T).T
            rotated_corners = rotated_corners.astype(np.int32)

            # Check if this is an interpolated boundary
            is_interpolated = False
            if (
                hasattr(self, "interpolated_boundary_to_marker")
                and i in self.interpolated_boundary_to_marker
            ):
                is_interpolated = True

            # Choose color based on boundary type
            if i < len(self.boundary_types):
                btype = self.boundary_types[i]
                if btype == "manual":
                    boundary_color = (255, 255, 0)  # Cyan for manual
                elif btype == "interpolated":
                    boundary_color = (0, 165, 255)  # Orange for interpolated
                else:
                    boundary_color = (0, 255, 0)  # Green for detected
            else:
                boundary_color = (0, 255, 0)  # Default green

            cv2.polylines(static_viz, [rotated_corners], True, boundary_color, 2)

            # Add compartment depth label
            mid_x = center_x
            mid_y = rect_center_y

            # Try to get depth from boundary_to_marker mapping first
            depth_label = None
            marker_id_for_depth = None

            # Check if we have a direct mapping for this boundary
            if hasattr(self, "boundary_to_marker") and i in self.boundary_to_marker:
                marker_id_for_depth = self.boundary_to_marker[i]
            else:
                # Fallback: Find the nearest marker below this compartment boundary
                nearest_marker_id = None
                nearest_distance = float("inf")

                for marker_id, corners in self.markers.items():
                    if marker_id in self.config.get(
                        "compartment_marker_ids", range(4, 24)
                    ):
                        marker_center_x = int(np.mean(corners[:, 0]))
                        marker_center_y = int(np.mean(corners[:, 1]))

                        if abs(marker_center_x - mid_x) < (x2 - x1) // 2:
                            vertical_dist = abs(marker_center_y - center_bottom_y)
                            if vertical_dist < nearest_distance:
                                nearest_distance = vertical_dist
                                nearest_marker_id = marker_id

                marker_id_for_depth = nearest_marker_id

            # Get the depth label from marker ID
            if marker_id_for_depth is not None and hasattr(
                self, "marker_to_compartment"
            ):
                depth_label = self.marker_to_compartment.get(
                    marker_id_for_depth, marker_id_for_depth - 3
                )

            # Draw the depth label
            if depth_label is not None:
                # Choose text color to match boundary color
                text_color = (
                    (0, 165, 255) if is_interpolated else (255, 165, 0)
                )  # Orange for interpolated, Light orange for detected

                text_size = cv2.getTextSize(
                    f"{depth_label}m", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
                )[0]

                # Black background for text
                cv2.rectangle(
                    static_viz,
                    (
                        int(mid_x) - text_size[0] // 2 - 5,
                        int(mid_y) - text_size[1] // 2 - 5,
                    ),
                    (
                        int(mid_x) + text_size[0] // 2 + 5,
                        int(mid_y) + text_size[1] // 2 + 5,
                    ),
                    (0, 0, 0),
                    -1,
                )

                # Draw text
                cv2.putText(
                    static_viz,
                    f"{depth_label}m",
                    (int(mid_x) - text_size[0] // 2, int(mid_y) + text_size[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    text_color,
                    2,
                )

        # Draw manually placed boundaries
        for comp_id, boundary in self.result_boundaries.items():
            if comp_id in [0, 1, 2, 3]:  # Corner markers
                if isinstance(boundary, np.ndarray) and boundary.shape[0] == 4:
                    corner_colors = {
                        0: (255, 0, 0),  # Blue for top-left
                        1: (0, 255, 0),  # Green for top-right
                        2: (0, 255, 255),  # Yellow for bottom-right
                        3: (255, 255, 0),  # Cyan for bottom-left
                    }
                    marker_color = corner_colors.get(comp_id, (255, 0, 255))

                    corners = boundary.astype(np.int32)
                    cv2.polylines(static_viz, [corners], True, marker_color, 3)

                    center_x = int(np.mean(corners[:, 0]))
                    center_y = int(np.mean(corners[:, 1]))
                    cv2.circle(static_viz, (center_x, center_y), 8, marker_color, -1)

                    corner_names = {0: "0 (TL)", 1: "1 (TR)", 2: "2 (BR)", 3: "3 (BL)"}
                    cv2.putText(
                        static_viz,
                        corner_names[comp_id],
                        (center_x - 20, center_y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        marker_color,
                        2,
                    )

            elif comp_id == 24:  # Metadata marker
                if isinstance(boundary, np.ndarray) and boundary.shape[0] == 4:
                    corners = boundary.astype(np.int32)
                    cv2.polylines(static_viz, [corners], True, (255, 0, 255), 2)

                    center_x = int(np.mean(corners[:, 0]))
                    center_y = int(np.mean(corners[:, 1]))
                    cv2.circle(static_viz, (center_x, center_y), 5, (255, 0, 255), -1)

                    cv2.putText(
                        static_viz,
                        "24",
                        (center_x, center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 0, 255),
                        2,
                    )

                    # Draw OCR extraction regions
                    region_x1 = max(0, center_x - 130)
                    region_x2 = min(img_width - 1, center_x + 130)

                    hole_id_region_y1 = max(0, center_y - 130)
                    hole_id_region_y2 = max(0, center_y - 20)
                    cv2.rectangle(
                        static_viz,
                        (region_x1, hole_id_region_y1),
                        (region_x2, hole_id_region_y2),
                        (255, 0, 0),
                        2,
                    )

                    cv2.putText(
                        static_viz,
                        "Hole ID",
                        (region_x1 + 10, hole_id_region_y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2,
                    )

                    depth_region_y1 = min(img_height - 1, center_y + 20)
                    depth_region_y2 = min(img_height - 1, center_y + 150)
                    cv2.rectangle(
                        static_viz,
                        (region_x1, depth_region_y1),
                        (region_x2, depth_region_y2),
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        static_viz,
                        "Depth",
                        (region_x1 + 10, depth_region_y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
            else:  # Compartment boundaries
                if isinstance(boundary, tuple) and len(boundary) == 4:
                    x1, y1, x2, y2 = boundary

                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    rotation_matrix = cv2.getRotationMatrix2D(
                        (center_x, center_y), rotation_angle, 1.0
                    )

                    rect_corners = np.array(
                        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
                    )

                    ones = np.ones(shape=(len(rect_corners), 1))
                    rect_corners_homog = np.hstack([rect_corners, ones])
                    rotated_corners = np.dot(rotation_matrix, rect_corners_homog.T).T
                    rotated_corners = rotated_corners.astype(np.int32)

                    cv2.polylines(static_viz, [rotated_corners], True, (0, 255, 255), 2)

                    depth_label = self.marker_to_compartment.get(comp_id, comp_id - 3)
                    mid_x = center_x
                    mid_y = center_y

                    text_size = cv2.getTextSize(
                        f"{depth_label}m", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
                    )[0]
                    cv2.rectangle(
                        static_viz,
                        (
                            int(mid_x) - text_size[0] // 2 - 5,
                            int(mid_y) - text_size[1] // 2 - 5,
                        ),
                        (
                            int(mid_x) + text_size[0] // 2 + 5,
                            int(mid_y) + text_size[1] // 2 + 5,
                        ),
                        (0, 0, 0),
                        -1,
                    )

                    cv2.putText(
                        static_viz,
                        f"{depth_label}m",
                        (
                            int(mid_x) - text_size[0] // 2,
                            int(mid_y) + text_size[1] // 2,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        2,
                    )

        # Cache the result
        self.static_viz_cache = static_viz.copy()
        self.static_viz_params = current_params

        return static_viz

    def _undo_last(self):
        """Undo the last annotation."""
        try:
            if self.current_mode == self.MODE_METADATA:
                # Nothing to undo in metadata mode
                self.status_var.set(DialogHelper.t("Nothing to undo in metadata mode"))
                return

            elif self.current_mode == self.MODE_MISSING_BOUNDARIES:
                # In boundary mode, undo the last placed marker

                # First check if we have manually placed any markers
                manually_placed_markers = [
                    mid
                    for mid in self.result_boundaries.keys()
                    if mid
                    not in [0, 1, 2, 3, 24]  # Exclude corner and metadata markers
                ]

                if not manually_placed_markers:
                    self.status_var.set(DialogHelper.t("Nothing to undo"))
                    return

                # Find the most recently placed marker
                if self.missing_marker_ids:
                    # We have markers still to place
                    if self.current_index > 0:
                        # Undo the last placed marker
                        self.current_index -= 1
                        current_id = self.missing_marker_ids[self.current_index]

                        # Remove it from result_boundaries if it exists
                        if current_id in self.result_boundaries:
                            del self.result_boundaries[current_id]

                        self.annotation_complete = False

                        # Get depth for display
                        depth = self.marker_to_compartment.get(
                            current_id, current_id - 3
                        )
                        self.status_var.set(
                            DialogHelper.t(
                                f"Undid compartment at depth {depth}m. Please place it again."
                            )
                        )
                    else:
                        # Check if we can undo a placed marker that's not in the current queue
                        last_placed_id = max(manually_placed_markers)
                        if last_placed_id in self.result_boundaries:
                            del self.result_boundaries[last_placed_id]

                            # Add it back to missing markers
                            self.missing_marker_ids.insert(0, last_placed_id)
                            self.current_index = 0
                            self.annotation_complete = False

                            depth = self.marker_to_compartment.get(
                                last_placed_id, last_placed_id - 3
                            )
                            self.status_var.set(
                                DialogHelper.t(
                                    f"Undid compartment at depth {depth}m. Please place it again."
                                )
                            )
                else:
                    # All markers were interpolated or placed
                    last_placed_id = max(manually_placed_markers)
                    if last_placed_id in self.result_boundaries:
                        del self.result_boundaries[last_placed_id]

                        # Add it back to missing markers
                        self.missing_marker_ids = [last_placed_id]
                        self.current_index = 0
                        self.annotation_complete = False

                        depth = self.marker_to_compartment.get(
                            last_placed_id, last_placed_id - 3
                        )
                        self.status_var.set(
                            DialogHelper.t(
                                f"Undid compartment at depth {depth}m. Please place it again."
                            )
                        )

                # Update visualization after removal
                self._update_visualization()
                self._update_status_message()

            elif self.current_mode == self.MODE_ADJUST_BOUNDARIES:
                # In adjustment mode, reset the last adjustment
                # This is a simple implementation - just resets offsets to zero
                self.left_height_offset = 0
                self.right_height_offset = 0

                # Update manual compartments
                self._update_manual_compartments()

                # Update visualization
                self._update_visualization()

                # Update static zoom views
                self._update_static_zoom_views()

                # Apply adjustments
                self._apply_adjustments()

                self.status_var.set(DialogHelper.t("Reset all side height adjustments"))

        except Exception as e:
            self.logger.error(f"Error in undo operation: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _on_continue(self):
        """Handle continue button click - advance through modes or complete workflow."""
        try:
            # Behavior depends on current mode
            if self.current_mode == self.MODE_METADATA:
                # Validate metadata before continuing
                if self._validate_metadata():
                    # After validation, check if we have missing markers to place
                    if self.missing_marker_ids:
                        # Switch to missing boundaries mode
                        self._switch_mode(self.MODE_MISSING_BOUNDARIES)
                    else:
                        # No missing markers, skip to adjustment mode
                        self._switch_mode(self.MODE_ADJUST_BOUNDARIES)

                    # Update instruction label and status
                    self._update_status_message()

            elif self.current_mode == self.MODE_MISSING_BOUNDARIES:
                # Check if all markers are placed or if we need to proceed without all markers
                if not self.missing_marker_ids or self.annotation_complete:
                    # All missing markers are placed or there were none
                    # Move to adjustment mode
                    self._switch_mode(self.MODE_ADJUST_BOUNDARIES)
                else:
                    # Ask for confirmation if not all markers are placed
                    missing_count = len(self.missing_marker_ids) - self.current_index
                    if missing_count > 0:
                        if DialogHelper.confirm_dialog(
                            self.dialog,
                            DialogHelper.t("Incomplete Annotations"),
                            DialogHelper.t(
                                f"You have {missing_count} compartments left to annotate. "
                                f"Do you want to proceed to boundary adjustment without placing all markers?"
                            ),
                            yes_text=DialogHelper.t("Proceed"),
                            no_text=DialogHelper.t("Stay Here"),
                        ):
                            # Move to adjustment mode despite missing markers
                            self._switch_mode(self.MODE_ADJUST_BOUNDARIES)
                    else:
                        # No missing markers, proceed to adjustment
                        self._switch_mode(self.MODE_ADJUST_BOUNDARIES)

                # Update instruction label and status
                self._update_status_message()

            elif self.current_mode == self.MODE_ADJUST_BOUNDARIES:
                # Final step - complete the workflow
                # Clean up resources before closing
                self._cleanup_zoom_lens()

                # Create final results
                self.final_results = {
                    "result_boundaries": self.result_boundaries,
                    "top_boundary": self.top_y,
                    "bottom_boundary": self.bottom_y,
                    "left_height_offset": self.left_height_offset,
                    "right_height_offset": self.right_height_offset,
                    "rotation_angle": self.rotation_angle,
                    "avg_width": self.avg_width,
                    "final_visualization": (
                        self.current_viz if hasattr(self, "current_viz") else None
                    ),
                    # Include metadata in results
                    "hole_id": self.hole_id.get().strip(),
                    "depth_from": int(self.depth_from.get().strip()),
                    "depth_to": int(self.depth_to.get().strip()),
                    "compartment_interval": self.interval_var.get(),
                }

                # Close the dialog
                self.dialog.destroy()

        except Exception as e:
            self.logger.error(f"Error in continue operation: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _on_reject(self):
        """Handle reject button click - signal rejection to the caller."""
        try:
            # ===================================================
            # MODIFIED CODE - Use standardized rejection handler
            # ===================================================
            # Define metadata validation callback
            def get_metadata():
                # Validate metadata first
                if not self._validate_metadata():
                    return None

                # Return validated metadata
                return {
                    "hole_id": self.hole_id.get().strip(),
                    "depth_from": int(self.depth_from.get().strip()),
                    "depth_to": int(self.depth_to.get().strip()),
                    "compartment_interval": self.interval_var.get(),
                }

            # Define cleanup callback
            def cleanup_resources():
                self._cleanup_zoom_lens()

            # Use standardized handler
            result = DialogHelper.handle_rejection(
                self.dialog,
                getattr(self, "image_path", None),  # Pass image path if available
                metadata_callback=get_metadata,
                cleanup_callback=cleanup_resources,
            )

            if result:
                # Set the rejection flag in the results
                self.final_results = result

                # Close the dialog
                self.dialog.destroy()

        except Exception as e:
            self.logger.error(f"Error in reject operation: {str(e)}")
            self.logger.error(traceback.format_exc())

            # Still try to close the dialog
            try:
                self.dialog.destroy()
            except:
                pass

    def _on_quit(self):
        """Handle quit button click - stop processing without making changes."""
        try:
            # Confirm quitting
            if DialogHelper.confirm_dialog(
                self.dialog,
                DialogHelper.t("Stop Processing"),
                DialogHelper.t(
                    "Are you sure you want to stop processing?\n\nNo modifications will be made to the current image, and processing of remaining images will be canceled."
                ),
                yes_text=DialogHelper.t("Stop Processing"),
                no_text=DialogHelper.t("Continue"),
            ):
                # Clean up resources before closing
                self._cleanup_zoom_lens()

                # Set a quit flag in the results
                self.final_results = {
                    "quit": True,
                    "message": "User stopped processing",
                }

                # Close the dialog
                self.dialog.destroy()
        except Exception as e:
            self.logger.error(f"Error in quit operation: {str(e)}")
            self.logger.error(traceback.format_exc())

            # Still try to close the dialog
            try:
                self.dialog.destroy()
            except:
                pass

    def _on_cancel(self):
        """Handle cancel button click."""
        try:
            # Confirm cancellation
            if DialogHelper.confirm_dialog(
                self.dialog,
                DialogHelper.t("Cancel Registration"),
                DialogHelper.t(
                    "Are you sure you want to cancel? All manual annotations will be lost."
                ),
                yes_text=DialogHelper.t("Yes"),
                no_text=DialogHelper.t("No"),
            ):
                # Clear result and close dialog
                self.result_boundaries = {}

                # Clean up resources before closing
                self._cleanup_zoom_lens()

                # Use destroy() to ensure the dialog closes
                self.dialog.destroy()
        except Exception as e:
            self.logger.error(f"Error in cancel operation: {str(e)}")

            # Still try to close the dialog
            try:
                self.dialog.destroy()
            except:
                pass

    def _hide_zoom_windows(self):
        """Hide the popup zoom windows when not in adjustment mode."""
        if hasattr(self, "left_zoom_window") and self.left_zoom_visible:
            self.left_zoom_window.withdraw()
            self.left_zoom_visible = False

        if hasattr(self, "right_zoom_window") and self.right_zoom_visible:
            self.right_zoom_window.withdraw()
            self.right_zoom_visible = False

    def _cleanup_zoom_lens(self):
        """Clean up zoom lens windows to prevent memory leaks."""
        try:
            if hasattr(self, "_zoom_lens") and self._zoom_lens:
                self._zoom_lens.destroy()
                self._zoom_lens = None

            if hasattr(self, "_zoom_lens_flipped") and self._zoom_lens_flipped:
                self._zoom_lens_flipped.destroy()
                self._zoom_lens_flipped = None

            # Also clean up adjustment zoom windows
            if hasattr(self, "left_zoom_window") and self.left_zoom_window:
                self.left_zoom_window.destroy()
                self.left_zoom_window = None

            if hasattr(self, "right_zoom_window") and self.right_zoom_window:
                self.right_zoom_window.destroy()
                self.right_zoom_window = None

            # Clean up photo references
            if hasattr(self, "left_zoom_photo"):
                self.left_zoom_photo = None

            if hasattr(self, "right_zoom_photo"):
                self.right_zoom_photo = None

        except Exception as e:
            self.logger.warning(f"Error cleaning up zoom lens: {str(e)}")

    def _cleanup_all_resources(self):
        """Clean up all resources to prevent memory leaks."""
        # Clean up zoom lens windows
        self._cleanup_zoom_lens()

        # Clean up any image references
        if hasattr(self, "photo_image"):
            self.photo_image = None

        if hasattr(self, "_zoom_img_ref"):
            self._zoom_img_ref = None

        if hasattr(self, "_zoom_img_flipped"):
            self._zoom_img_flipped = None

        # Clean up canvas references
        for attr_name in dir(self):
            if attr_name.startswith("_canvas"):
                setattr(self, attr_name, None)

    def _get_monitor_size(self):
        """Get the size of the monitor containing this dialog."""
        try:
            # Try to get actual monitor dimensions
            dialog_x = self.dialog.winfo_x()
            dialog_y = self.dialog.winfo_y()

            # Default to conservative size
            width, height = 1920, 1080

            # Platform-specific monitor detection
            import platform

            if platform.system() == "Windows":
                try:
                    import ctypes
                    from ctypes import wintypes

                    # Get monitor info for the point where dialog is
                    monitor = ctypes.windll.user32.MonitorFromPoint(
                        wintypes.POINT(dialog_x, dialog_y),
                        2,  # MONITOR_DEFAULTTONEAREST
                    )

                    info = wintypes.MONITORINFO()
                    info.cbSize = ctypes.sizeof(info)
                    if ctypes.windll.user32.GetMonitorInfoW(
                        monitor, ctypes.byref(info)
                    ):
                        # Get work area (excludes taskbar)
                        work_area = info.rcWork
                        width = work_area.right - work_area.left
                        height = work_area.bottom - work_area.top
                except:
                    pass

            return width, height

        except Exception as e:
            self.logger.warning(f"Could not detect monitor size: {e}")
            return 1920, 1080  # Safe fallback

    def show(self):
        """
        Show the dialog and wait for user input.
        Returns a dictionary with metadata, boundaries, and adjustment results.
        """
        self.logger.debug("CompartmentRegistrationDialog.show() method started")

        # Check thread safety
        current_thread = threading.current_thread()
        if current_thread is not threading.main_thread():
            self.logger.error(
                f"show() called from non-main thread: {current_thread.name}"
            )
            return {}

        try:
            # Ensure dialog exists
            if not hasattr(self, "dialog") or not self.dialog:
                self.logger.error("Cannot show dialog - dialog does not exist")
                return {}

            # Make dialog visible
            self.dialog.deiconify()
            self.dialog.update_idletasks()

            # Set transient relationship
            if self.parent:
                self.dialog.transient(self.parent)

            # Get monitor dimensions
            screen_width, screen_height = self._get_monitor_size()

            # Calculate dialog size based on canvas needs
            if hasattr(self, "canvas") and self.canvas.winfo_exists():
                # Get the canvas's requested size
                canvas_width = self.canvas.winfo_reqwidth()
                canvas_height = self.canvas.winfo_reqheight()

                # Add space for UI elements
                ui_padding_width = 100  # Side panels, margins
                ui_padding_height = 300  # Buttons, status, metadata

                # Calculate dialog size
                dialog_width = canvas_width + ui_padding_width
                dialog_height = canvas_height + ui_padding_height

                # Ensure it fits on screen with margins
                max_width = int(screen_width * 0.99)  # Use 95% of screen width
                max_height = int(screen_height * 0.99)  # Use 90% of screen height

                dialog_width = min(dialog_width, max_width)
                dialog_height = min(dialog_height, max_height)
            else:
                # Fallback to conservative ratios
                dialog_width = int(screen_width * 0.99)  # Increased from 0.85
                dialog_height = int(screen_height * 0.99)  # Increased from 0.80

            # Ensure minimum size
            dialog_width = max(dialog_width, 1200)
            dialog_height = max(dialog_height, 700)

            # Center the dialog
            x = (screen_width - dialog_width) // 2
            y = (screen_height - dialog_height) // 2

            # Apply geometry
            self.dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")

            # Force canvas to update to fill available space
            self.dialog.update_idletasks()

            # Now trigger visualization update with proper dimensions
            if hasattr(self, "_update_visualization"):
                self._update_visualization(force_full_update=True)

            # Ensure dialog has focus
            self.dialog.lift()
            self.dialog.focus_force()

            # Wait for dialog to close
            self.dialog.wait_window()

            # Scale transformation data to original image coordinates
            scaled_transformation_matrix = None
            scaled_cumulative_offset_y = None
            scaled_transform_center = None

            if (
                hasattr(self, "transformation_matrix")
                and self.transformation_matrix is not None
            ):
                # Get scale factor from working to original
                scale_factor = 1.0
                if hasattr(self, "scale_data") and self.scale_data:
                    scale_relationships = self.scale_data.get("scale_relationships", {})
                    scale_info = scale_relationships.get(
                        ("working", "original"), {"scale": 1.0}
                    )
                    scale_factor = scale_info["scale"]

                # Scale transformation matrix translation components
                scaled_matrix = self.transformation_matrix.copy()
                scaled_matrix[0, 2] *= scale_factor  # Scale tx
                scaled_matrix[1, 2] *= scale_factor  # Scale ty
                scaled_transformation_matrix = scaled_matrix.tolist()

                # Scale vertical offset
                if hasattr(self, "cumulative_offset_y"):
                    scaled_cumulative_offset_y = self.cumulative_offset_y * scale_factor

                # Scale transform center
                if hasattr(self, "image_aligner") and hasattr(
                    self.image_aligner, "transform_center"
                ):
                    center = self.image_aligner.transform_center
                    scaled_transform_center = [
                        center[0] * scale_factor,
                        center[1] * scale_factor,
                    ]

            # Return the final results
            if hasattr(self, "final_results"):
                return self.final_results
            else:
                # Create a default result dict
                result = {
                    "result_boundaries": self.result_boundaries,
                    "boundaries": self.detected_boundaries,
                    "top_boundary": self.top_y,
                    "bottom_boundary": self.bottom_y,
                    "left_height_offset": self.left_height_offset,
                    "right_height_offset": self.right_height_offset,
                    "rotation_angle": self.rotation_angle,
                    "avg_width": self.avg_width,
                    "final_visualization": (
                        self.current_viz if hasattr(self, "current_viz") else None
                    ),
                    # Include metadata
                    "hole_id": self.hole_id.get().strip(),
                    "depth_from": (
                        int(self.depth_from.get().strip())
                        if self.depth_from.get().strip().isdigit()
                        else None
                    ),
                    "depth_to": (
                        int(self.depth_to.get().strip())
                        if self.depth_to.get().strip().isdigit()
                        else None
                    ),
                    "compartment_interval": self.interval_var.get(),
                    "transformation_matrix": scaled_transformation_matrix if scaled_transformation_matrix is not None else [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    "cumulative_offset_y": scaled_cumulative_offset_y if scaled_cumulative_offset_y is not None else 0.0,
                    "transformation_applied": (
                        self.transformation_applied
                        if hasattr(self, "transformation_applied")
                        else False
                    ),
                    "transform_center": scaled_transform_center if scaled_transform_center is not None else [0.0, 0.0],
                }
                return result

        except Exception as e:
            self.logger.error(f"Error showing dialog: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {}


class DialogDataTracker:
    """Track what data the dialog actually uses vs recalculates"""

    def __init__(self):
        self.provided_data = {}
        self.used_data = set()
        self.recalculated_data = {}
        self.unused_data = set()

    def log_provided(self, key, value):
        self.provided_data[key] = value

    def log_used(self, key):
        self.used_data.add(key)

    def log_recalculated(self, key, original, new):
        self.recalculated_data[key] = {
            "original": original,
            "new": new,
            "match": str(original) == str(new),
        }

    def report(self):
        self.unused_data = set(self.provided_data.keys()) - self.used_data

        print("\n=== Dialog Data Usage Report ===")
        print(f"\nProvided data keys: {list(self.provided_data.keys())}")
        print(f"\nActually used: {list(self.used_data)}")
        print(f"\nUnused data: {list(self.unused_data)}")

        print("\nRecalculated data:")
        for key, info in self.recalculated_data.items():
            print(f"  {key}: Match={info['match']}")
            if not info["match"]:
                print(f"    Original: {info['original']}")
                print(f"    New: {info['new']}")
