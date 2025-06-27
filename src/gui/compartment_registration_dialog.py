# gui/compartment_registration_dialog.py

"""
Unified dialog for complete compartment registration workflow.
Combines metadata input, boundary annotation, and boundary adjustment in one interface.
"""
import time
import re
import logging
import threading
import traceback
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np

from PIL import Image, ImageTk, ImageOps


from gui.dialog_helper import DialogHelper

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
    ):
        """
        Initialize the unified compartment registration dialog.

        Args:
            parent: Parent window
            image: Image to annotate (numpy array)
            detected_boundaries: List of already detected boundaries as (x1, y1, x2, y2)
            missing_marker_ids: List of missing marker IDs that need manual annotation
            theme_colors: Optional theme colors dictionary
            gui_manager: Optional GUIManager instance for consistent styling
            boundaries_viz: Optional visualization image with detected boundaries
            original_image: Optional original high-resolution image for extraction
            output_format: Output format for saved images (default: "png")
            file_manager: FileManager instance for saving files
            metadata: Optional metadata dictionary containing hole_id, depth_from, depth_to
            vertical_constraints: Optional tuple of (min_y, max_y) for compartment placement
            marker_to_compartment: Dictionary mapping marker IDs to compartment numbers
            rotation_angle: Current rotation angle of the image (degrees)
            corner_markers: Dictionary of corner marker positions {0: (x,y), 1: (x,y)...}
            markers: Dictionary of all detected ArUco markers {id: corners}
            config: Configuration dictionary
            on_apply_adjustments: Callback function for boundary adjustments
            show_adjustment_controls: Whether to show adjustment controls by default
            image_path: Path to the original image file being processed
        """
        self.parent = parent

        # Try to get app reference through parent chain
        self.app = None
        if hasattr(parent, "master") and hasattr(parent.master, "app"):
            self.app = parent.master.app
        elif hasattr(parent, "app"):
            self.app = parent.app
        # If parent is the root window, check for app attribute
        elif hasattr(parent, "winfo_toplevel"):
            toplevel = parent.winfo_toplevel()
            if hasattr(toplevel, "app"):
                self.app = toplevel.app

        # Initialize logger early so it's available for all methods
        self.logger = logging.getLogger(__name__)

        # Store original image and visualization data
        self.source_image = image.copy() if image is not None else None
        self.source_image = (
            self.source_image.copy()
            if self.source_image is not None
            else (boundaries_viz.copy() if self.boundaries_viz is not None else None)
        )
        self.image = image.copy() if image is not None else None
        self.original_image = original_image if original_image is not None else None
        # ===================================================
        # REMOVE: Confusing multiple image references above
        # REPLACE WITH: Clear image management
        # Image management with clear naming
        # 1. source_image: The original clean image without any annotations (never modified)
        # 2. display_image: The current working image for visualization (gets updated)
        # 3. high_res_image: Optional high-resolution image for extraction purposes

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

        # Store detected boundaries
        self.detected_boundaries = (
            detected_boundaries.copy() if detected_boundaries else []
        )

        # Initialize missing_marker_ids before checking corner markers
        self.missing_marker_ids = (
            missing_marker_ids if missing_marker_ids is not None else []
        )

        # Store corner markers and markers
        self.corner_markers = corner_markers or {}
        self.markers = markers or {}
        # ===================================================

        # Store boundary analysis data
        self.boundary_analysis = boundary_analysis
        if boundary_analysis:
            # Use the average compartment width from analysis
            if "avg_compartment_width" in boundary_analysis:
                self.avg_width = boundary_analysis["avg_compartment_width"]

            # Use boundary to marker mapping if available
            if "boundary_to_marker" in boundary_analysis:
                self.boundary_to_marker = boundary_analysis["boundary_to_marker"]

        # Store scale data
        self.scale_data = scale_data

        # Store configuration and settings
        self.output_format = output_format
        self.file_manager = file_manager
        self.metadata = metadata or {}
        self.rotation_angle = rotation_angle
        self.corner_markers = corner_markers or {}
        self.markers = markers or {}
        self.config = config or {
            "compartment_marker_ids": list(range(4, 24)),
            "corner_marker_ids": [0, 1, 2, 3],
            "metadata_marker_ids": [24],
        }
        self.on_apply_adjustments = on_apply_adjustments
        self._last_apply_time = 0
        self._apply_debounce_interval = 1.0

        # Initialize mode and state tracking
        self.current_mode = self.MODE_METADATA  # Start in metadata mode
        self.current_index = 0
        self.result_boundaries = {}
        self.annotation_complete = False

        # Initialize missing_marker_ids before checking corner markers
        self.missing_marker_ids = (
            missing_marker_ids if missing_marker_ids is not None else []
        )

        # Check if we need to add corner markers to missing list
        self._check_missing_corner_markers()

        # Create metadata input variables
        self.hole_id = tk.StringVar(value=metadata.get("hole_id", ""))
        self.depth_from = tk.StringVar(value=str(metadata.get("depth_from", "")))
        self.depth_to = tk.StringVar(value=str(metadata.get("depth_to", "")))

        # Get compartment interval from metadata (default to 1 if not specified)
        self.compartment_interval = int(self.metadata.get("compartment_interval", 1))
        self.interval_var = tk.IntVar(value=self.compartment_interval)

        # Register trace on depth variables for auto-updating compartment labels
        self.depth_from.trace_add("write", self._update_compartment_labels)
        self.depth_to.trace_add("write", self._update_compartment_labels)
        self.interval_var.trace_add("write", self._update_compartment_labels)

        # Initialize scaling and canvas offset variables
        self.scale_ratio = 1.0
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0

        # Adjustment parameters
        self.adjustment_controls_visible = show_adjustment_controls
        self.left_height_offset = 0
        self.right_height_offset = 0

        # Calculate vertical constraints if not provided
        if vertical_constraints:
            self.top_y, self.bottom_y = vertical_constraints
        else:
            self._calculate_vertical_constraints()

        # Calculate average compartment width
        self._calculate_average_compartment_width()

        # Map between marker IDs and compartment numbers
        self.marker_to_compartment = marker_to_compartment or {
            4 + i: int((i + 1) * self.compartment_interval) for i in range(20)
        }

        # GUI references and theming
        self.gui_manager = gui_manager
        # If gui_manager is available, use its theme colors
        if self.gui_manager and hasattr(self.gui_manager, "theme_colors"):
            self.theme_colors = self.gui_manager.theme_colors
        else:
            # Fallback theme colors if gui_manager is unavailable
            self.theme_colors = theme_colors or {
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

        # State tracking
        self.temp_point = None

        # Cache for static visualization
        self.static_viz_cache = None
        self.static_viz_params = None  # To track when cache needs updating

        # Adjustment mode flags
        self.adjusting_top = False
        self.adjusting_bottom = False
        self.adjusting_left_side = False
        self.adjusting_right_side = False

        # Create dialog
        self.dialog = self._create_dialog()

        # Apply gui_manager ttk styles
        self.gui_manager.configure_ttk_styles(self.dialog)

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

    def _get_image_dimensions(self):
        """Get image dimensions consistently."""
        if self.source_image is not None:
            return self.source_image.shape[:2]
        return (800, 1000)  # Default dimensions if no image

    def _check_missing_corner_markers(self):
        """Check if we need to add corner markers to missing list - only if BOTH from a pair are missing."""
        # Check which corner markers we have in the detected markers
        has_marker_0 = 0 in self.markers
        has_marker_1 = 1 in self.markers
        has_marker_2 = 2 in self.markers
        has_marker_3 = 3 in self.markers

        # Log what we found
        self.logger.debug(
            f"Corner marker detection - 0: {has_marker_0}, 1: {has_marker_1}, 2: {has_marker_2}, 3: {has_marker_3}"
        )

        # Convert missing_marker_ids to a list if it isn't already
        if not isinstance(self.missing_marker_ids, list):
            self.missing_marker_ids = (
                list(self.missing_marker_ids) if self.missing_marker_ids else []
            )

        self.logger.debug(f"Initial missing_marker_ids: {self.missing_marker_ids}")

        # If we have at least one top marker, remove any missing top markers
        if has_marker_0 or has_marker_1:
            self.logger.debug("Already have at least one top corner marker")
            # Remove markers 0 and 1 from missing list if present
            if 0 in self.missing_marker_ids:
                self.missing_marker_ids.remove(0)
                self.logger.debug(
                    "Removed marker 0 from missing list - already have a top corner"
                )
            if 1 in self.missing_marker_ids:
                self.missing_marker_ids.remove(1)
                self.logger.debug(
                    "Removed marker 1 from missing list - already have a top corner"
                )
        else:
            # Both top markers are missing - we need to place one
            self.logger.info("No top corner markers detected - need to place one")
            # Make sure at least marker 0 is in the list
            if 0 not in self.missing_marker_ids:
                self.missing_marker_ids.append(0)

        # If we have at least one bottom marker, remove any missing bottom markers
        if has_marker_2 or has_marker_3:
            self.logger.debug("Already have at least one bottom corner marker")
            # Remove markers 2 and 3 from missing list if present
            if 2 in self.missing_marker_ids:
                self.missing_marker_ids.remove(2)
                self.logger.debug(
                    "Removed marker 2 from missing list - already have a bottom corner"
                )
            if 3 in self.missing_marker_ids:
                self.missing_marker_ids.remove(3)
                self.logger.debug(
                    "Removed marker 3 from missing list - already have a bottom corner"
                )
        else:
            # Both bottom markers are missing - we need to place one
            self.logger.info("No bottom corner markers detected - need to place one")
            # Make sure at least marker 2 is in the list
            if 2 not in self.missing_marker_ids:
                self.missing_marker_ids.append(2)

        # Sort the list to ensure corner markers come first
        self.missing_marker_ids.sort()
        self.logger.debug(
            f"Updated missing marker IDs after corner check: {self.missing_marker_ids}"
        )

    def _calculate_vertical_constraints(self):
        """Calculate the vertical constraints (top and bottom limits) from detected boundaries and corner markers."""
        # Collect all y-coordinates from top corner markers (0, 1)
        top_y_coords = []
        bottom_y_coords = []

        # Check all sources of corner markers
        for marker_id in [0, 1]:  # Top corners
            # Check in self.markers (detected markers) - use TOP edge
            if marker_id in self.markers:
                corners = self.markers[marker_id]
                # For top markers, use the minimum Y (top edge)
                top_edge_y = np.min(corners[:, 1])
                top_y_coords.append(top_edge_y)
                self.logger.debug(
                    f"Found top corner {marker_id} in markers, using top edge at y={top_edge_y}"
                )

            # Check in self.corner_markers - use TOP edge for detected markers
            if (
                hasattr(self, "corner_markers")
                and self.corner_markers
                and marker_id in self.corner_markers
            ):
                corners = self.corner_markers[marker_id]
                # If this is from original detection, use top edge
                if marker_id not in self.result_boundaries:  # Not manually placed
                    top_edge_y = np.min(corners[:, 1])
                    if top_edge_y not in top_y_coords:
                        top_y_coords.append(top_edge_y)
                        self.logger.debug(
                            f"Found top corner {marker_id} in corner_markers, using top edge at y={top_edge_y}"
                        )
                else:
                    # For manually placed, use center
                    center_y = np.mean(corners[:, 1])
                    if center_y not in top_y_coords:
                        top_y_coords.append(center_y)
                        self.logger.debug(
                            f"Found top corner {marker_id} in corner_markers (manual), using center at y={center_y}"
                        )

            # Check in self.result_boundaries (manually placed) - use center
            if (
                hasattr(self, "result_boundaries")
                and marker_id in self.result_boundaries
            ):
                corners = self.result_boundaries[marker_id]
                if isinstance(corners, np.ndarray):
                    center_y = np.mean(corners[:, 1])
                    if center_y not in top_y_coords:  # Avoid duplicates
                        top_y_coords.append(center_y)
                        self.logger.debug(
                            f"Found top corner {marker_id} in result_boundaries (manual) at y={center_y}"
                        )

        for marker_id in [2, 3]:  # Bottom corners
            # Check in self.markers (detected markers) - use BOTTOM edge
            if marker_id in self.markers:
                corners = self.markers[marker_id]
                # For bottom markers, use the maximum Y (bottom edge)
                bottom_edge_y = np.max(corners[:, 1])
                bottom_y_coords.append(bottom_edge_y)
                self.logger.debug(
                    f"Found bottom corner {marker_id} in markers, using bottom edge at y={bottom_edge_y}"
                )

            # Check in self.corner_markers - use BOTTOM edge for detected markers
            if (
                hasattr(self, "corner_markers")
                and self.corner_markers
                and marker_id in self.corner_markers
            ):
                corners = self.corner_markers[marker_id]
                # If this is from original detection, use bottom edge
                if marker_id not in self.result_boundaries:  # Not manually placed
                    bottom_edge_y = np.max(corners[:, 1])
                    if bottom_edge_y not in bottom_y_coords:
                        bottom_y_coords.append(bottom_edge_y)
                        self.logger.debug(
                            f"Found bottom corner {marker_id} in corner_markers, using bottom edge at y={bottom_edge_y}"
                        )
                else:
                    # For manually placed, use center
                    center_y = np.mean(corners[:, 1])
                    if center_y not in bottom_y_coords:
                        bottom_y_coords.append(center_y)
                        self.logger.debug(
                            f"Found bottom corner {marker_id} in corner_markers (manual), using center at y={center_y}"
                        )

            # Check in self.result_boundaries (manually placed) - use center
            if (
                hasattr(self, "result_boundaries")
                and marker_id in self.result_boundaries
            ):
                corners = self.result_boundaries[marker_id]
                if isinstance(corners, np.ndarray):
                    center_y = np.mean(corners[:, 1])
                    if center_y not in bottom_y_coords:  # Avoid duplicates
                        bottom_y_coords.append(center_y)
                        self.logger.debug(
                            f"Found bottom corner {marker_id} in result_boundaries (manual) at y={center_y}"
                        )

        # If we have at least one top and one bottom corner, use them
        if top_y_coords and bottom_y_coords:
            self.top_y = int(np.mean(top_y_coords))
            self.bottom_y = int(np.mean(bottom_y_coords))
            self.logger.info(
                f"Calculated constraints from corner markers: top_y={self.top_y}, bottom_y={self.bottom_y}"
            )
            return

        # Fall back to using detected boundaries TODO - FALLBACK LOGIC WILL BREAK THIS SOONER OR LATER
        if not self.detected_boundaries:
            # Default values if no boundaries are detected
            h = self.source_image.shape[0] if self.source_image is not None else 800
            self.top_y = int(h * 0.1)
            self.bottom_y = int(h * 0.9)
            self.logger.warning(
                f"Using default constraints: top_y={self.top_y}, bottom_y={self.bottom_y}"
            )
            return

        # Extract y-coordinates from detected boundaries
        top_coords = [y1 for _, y1, _, _ in self.detected_boundaries]
        bottom_coords = [y2 for _, _, _, y2 in self.detected_boundaries]

        # Calculate average top and bottom positions
        if top_coords:
            self.top_y = int(sum(top_coords) / len(top_coords))
        else:
            h = self.source_image.shape[0] if self.source_image is not None else 800
            self.top_y = int(h * 0.1)

        if bottom_coords:
            self.bottom_y = int(sum(bottom_coords) / len(bottom_coords))
        else:
            h = self.source_image.shape[0] if self.source_image is not None else 800
            self.bottom_y = int(h * 0.9)

        self.logger.info(
            f"Calculated constraints from boundaries: top_y={self.top_y}, bottom_y={self.bottom_y}"
        )

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

        # Main image canvas frame
        self.canvas_frame = ttk.Frame(main_frame, style="Content.TFrame")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Get screen dimensions for dynamic sizing
        screen_width = self.dialog.winfo_screenwidth()
        screen_height = self.dialog.winfo_screenheight()

        # Calculate dialog target dimensions (what we'll use in show())
        dialog_target_width = int(screen_width * 0.95)
        dialog_target_height = int(screen_height * 0.9)

        # Estimate space needed for other UI elements
        # Top: mode selector (~60px), padding (20px)
        # Bottom: metadata/adjustment controls (~150px), status (~40px), buttons (~60px), padding (30px)
        vertical_ui_space = 60 + 150 + 40 + 60 + 50  # ~360px
        horizontal_ui_space = 40  # Padding on sides

        # Calculate canvas dimensions
        canvas_width = dialog_target_width - horizontal_ui_space
        canvas_height = dialog_target_height - vertical_ui_space

        # Ensure minimum reasonable size
        canvas_width = max(canvas_width, 800)
        canvas_height = max(canvas_height, 400)

        # Canvas for the image display
        self.canvas = tk.Canvas(
            self.canvas_frame,
            bg=self.theme_colors["background"],
            highlightthickness=1,
            highlightbackground=self.theme_colors["field_border"],
            width=canvas_width,
            height=canvas_height,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bind mouse events
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Motion>", self._on_canvas_move)
        self.canvas.bind("<Leave>", self._on_canvas_leave)
        self.canvas.bind("<Button-3>", self._on_canvas_right_click)  # Right click
        self.canvas.bind("<B3-Motion>", self._on_canvas_move)  # Right button drag
        self.canvas.bind("<KeyPress>", self._on_key_press)

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

        # Status bar
        status_frame = ttk.Frame(main_frame, style="Content.TFrame")
        status_frame.pack(fill=tk.X, pady=(5, 0))

        self.status_var = tk.StringVar(value="")
        self.status_label = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            style="Content.TLabel",
            font=("Arial", 11),
        )
        self.status_label.pack(fill=tk.X, pady=(0, 5))

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

        # Update the status message based on the current mode
        self._update_status_message()

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
        """Create metadata input panel with hole ID and depth fields - more compact version."""
        # Main container for metadata input
        fields_frame = ttk.Frame(parent_frame, style="Content.TFrame", padding=5)
        fields_frame.pack(fill=tk.X, padx=5, pady=5)

        # Create a grid with 3 columns for more efficient space usage
        fields_frame.columnconfigure(0, weight=0)  # Label column - fixed width
        fields_frame.columnconfigure(1, weight=1)  # Value column - can expand
        fields_frame.columnconfigure(2, weight=1)  # Help/extra column - can expand

        # Custom font for entries - slightly smaller for compact layout
        custom_font = ("Arial", 12)

        # Row counter
        row = 0

        # --- Hole ID ---
        hole_id_label = ttk.Label(
            fields_frame,
            text=DialogHelper.t("Hole ID:"),
            font=("Arial", 11, "bold"),
            style="Content.TLabel",
        )
        hole_id_label.grid(row=row, column=0, sticky=tk.W, pady=2, padx=5)

        hole_id_entry = tk.Entry(
            fields_frame,
            textvariable=self.hole_id,
            font=custom_font,
            width=10,
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["text"],
            insertbackground=self.theme_colors["text"],
            highlightbackground=self.theme_colors["field_border"],
            highlightthickness=1,
            relief=tk.FLAT,
        )
        hole_id_entry.grid(row=row, column=1, sticky=tk.W, padx=(0, 5), pady=2)

        # Format help - smaller and less padding
        hole_id_help = ttk.Label(
            fields_frame,
            text=DialogHelper.t("Format: XX0000"),
            font=("Arial", 8),
            foreground="gray",
            style="Content.TLabel",
        )
        hole_id_help.grid(row=row, column=2, sticky=tk.W, pady=2)

        row += 1  # Next row

        # --- Depth Range and Interval ---
        depth_label = ttk.Label(
            fields_frame,
            text=DialogHelper.t("Depth:"),
            font=("Arial", 11, "bold"),
            style="Content.TLabel",
        )
        depth_label.grid(row=row, column=0, sticky=tk.W, pady=2, padx=5)

        # Container for depth fields with inline layout to save space
        depth_container = ttk.Frame(fields_frame, style="Content.TFrame")
        depth_container.grid(row=row, column=1, sticky=tk.W, pady=2)

        depth_from_entry = tk.Entry(
            depth_container,
            textvariable=self.depth_from,
            width=4,
            font=custom_font,
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["text"],
            insertbackground=self.theme_colors["text"],
            highlightbackground=self.theme_colors["field_border"],
            highlightthickness=1,
            relief=tk.FLAT,
        )
        depth_from_entry.pack(side=tk.LEFT)

        depth_separator = ttk.Label(
            depth_container,
            text=DialogHelper.t("-"),
            font=custom_font,
            style="Content.TLabel",
        )
        depth_separator.pack(side=tk.LEFT, padx=2)

        depth_to_entry = tk.Entry(
            depth_container,
            textvariable=self.depth_to,
            width=4,
            font=custom_font,
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["text"],
            insertbackground=self.theme_colors["text"],
            highlightbackground=self.theme_colors["field_border"],
            highlightthickness=1,
            relief=tk.FLAT,
        )
        depth_to_entry.pack(side=tk.LEFT)

        # Format help
        depth_help = ttk.Label(
            fields_frame,
            text=DialogHelper.t("Format: 00-00"),
            font=("Arial", 8),
            foreground="gray",
            style="Content.TLabel",
        )
        depth_help.grid(row=row, column=2, sticky=tk.W, pady=2)

        row += 1  # Next row

        # --- Interval ---
        interval_label = ttk.Label(
            fields_frame,
            text=DialogHelper.t("Interval:"),
            font=("Arial", 11, "bold"),
            style="Content.TLabel",
        )
        interval_label.grid(row=row, column=0, sticky=tk.W, pady=2, padx=5)

        # Create a frame for the dropdown with theme colors - more compact
        interval_combo_frame = tk.Frame(
            fields_frame,
            bg=self.theme_colors["field_bg"],
            highlightbackground=self.theme_colors["field_border"],
            highlightthickness=1,
        )
        interval_combo_frame.grid(row=row, column=1, sticky=tk.W, pady=2)

        # Create the dropdown options
        interval_options = [1, 2]

        # Create the OptionMenu widget
        interval_dropdown = tk.OptionMenu(
            interval_combo_frame,
            self.interval_var,
            *interval_options,
            command=self._update_expected_depth,
        )

        # Style the dropdown - more compact
        interval_dropdown.config(
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["text"],
            activebackground=self.theme_colors.get("hover_highlight", "#3a3a3a"),
            activeforeground=self.theme_colors["text"],
            font=("Arial", 11),
            width=3,
            highlightthickness=0,
            bd=0,
        )
        interval_dropdown["menu"].config(
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["text"],
            activebackground=self.theme_colors.get("hover_highlight", "#3a3a3a"),
            activeforeground=self.theme_colors["text"],
            font=("Arial", 11),
        )
        interval_dropdown.pack()

        # Interval help text
        interval_help = ttk.Label(
            fields_frame,
            text=DialogHelper.t("meters per compartment"),
            font=("Arial", 8),
            foreground="gray",
            style="Content.TLabel",
        )
        interval_help.grid(row=row, column=2, sticky=tk.W, pady=2)

        # ===================================================
        # NEW: Add increment button here in metadata panel
        # ===================================================
        row += 1  # Next row

        # Create increment button in the metadata panel
        if self.gui_manager:
            self.increment_button = self.gui_manager.create_modern_button(
                fields_frame,
                text=DialogHelper.t("Increment From Last"),
                color=self.theme_colors["accent_blue"],
                command=self._on_increment_from_last,
            )
            self.increment_button.grid(
                row=row, column=0, columnspan=3, pady=(10, 5), sticky=tk.W + tk.E
            )
        else:
            self.increment_button = ttk.Button(
                fields_frame,
                text=DialogHelper.t("Increment From Last"),
                command=self._on_increment_from_last,
                style="TButton",
                padding=5,
            )
            self.increment_button.grid(
                row=row, column=0, columnspan=3, pady=(10, 5), sticky=tk.W + tk.E
            )

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

            # Set up the flipped zoom canvas
            self._zoom_canvas_flipped = tk.Canvas(
                self._zoom_lens_flipped,
                width=zoom_width,
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
            self._zoom_canvas_flipped.create_line(
                0, center_y, zoom_width, center_y, fill="red", width=1, tags="crosshair"
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

            # Store dimensions for later use
            self._zoom_width = zoom_width
            self._zoom_height = zoom_height

        except Exception as e:
            self.logger.error(f"Error initializing zoom lens: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Proceed without zoom lens functionality - don't let this stop the dialog

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

        # Update continue button text based on current mode
        continue_text = self._get_continue_button_text()
        self.continue_button.set_text(continue_text)

        # Update status message for this mode
        self._update_status_message()

        # Update visualization to match the current mode
        self._update_visualization()

    def _update_static_zoom_views(self):
        """Update and show popup zoom windows for boundary adjustment mode."""
        if not hasattr(self, "left_zoom_window") or not hasattr(
            self, "right_zoom_window"
        ):
            self._create_zoom_windows()

        if self.source_image is None:
            return

        # Get image dimensions
        h, w = self.source_image.shape[:2]

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

        # Size of zoom region
        zoom_size = min(w // 8, 100)

        try:
            # Extract left region
            left_region = self.source_image[
                max(0, center_y - zoom_size) : min(h, center_y + zoom_size),
                max(0, left_x - zoom_size) : min(w, left_x + zoom_size),
            ].copy()

            # Extract right region
            right_region = self.source_image[
                max(0, center_y - zoom_size) : min(h, center_y + zoom_size),
                max(0, right_x - zoom_size) : min(w, right_x + zoom_size),
            ].copy()

            left_rgb = cv2.cvtColor(left_region, cv2.COLOR_BGR2RGB)
            right_rgb = cv2.cvtColor(right_region, cv2.COLOR_BGR2RGB)

            # Create PIL images and resize BEFORE drawing overlays
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

            # Add titles back with smaller font
            self.left_zoom_canvas.create_text(
                self.zoom_width // 2,
                15,
                text=DialogHelper.t("Left Side"),
                fill=self.theme_colors["text"],
                font=("Arial", 9, "bold"),  # Smaller font
            )

            self.right_zoom_canvas.create_text(
                self.zoom_width // 2,
                15,
                text=DialogHelper.t("Right Side"),
                fill=self.theme_colors["text"],
                font=("Arial", 9, "bold"),  # Smaller font
            )

            # Position and show the zoom windows
            # Calculate positions - above the markers being zoomed
            canvas_x = self.canvas.winfo_rootx()
            canvas_y = self.canvas.winfo_rooty()

            # Convert image coordinates to canvas coordinates
            # left_x and right_x are in image coordinates, need to convert to canvas
            left_canvas_x = int(left_x * self.scale_ratio + self.canvas_offset_x)
            right_canvas_x = int(right_x * self.scale_ratio + self.canvas_offset_x)

            # Position zoom windows centered above their respective markers
            left_zoom_x = canvas_x + left_canvas_x - (self.zoom_width // 2)
            right_zoom_x = canvas_x + right_canvas_x - (self.zoom_width // 2)

            # Position both above the canvas
            zoom_y = canvas_y - self.zoom_height - 10

            # Make sure zoom windows are visible on screen
            screen_width = self.dialog.winfo_screenwidth()
            screen_height = self.dialog.winfo_screenheight()

            left_zoom_x = max(10, min(screen_width - self.zoom_width - 10, left_zoom_x))
            right_zoom_x = max(
                10, min(screen_width - self.zoom_width - 10, right_zoom_x)
            )
            zoom_y = max(10, min(screen_height - self.zoom_height - 10, zoom_y))

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

        # Set up left zoom canvas
        zoom_width, zoom_height = 250, 300
        self.left_zoom_canvas = tk.Canvas(
            self.left_zoom_window,
            width=zoom_width,
            height=zoom_height,
            bg=self.theme_colors["background"],
            highlightthickness=1,
            highlightbackground=self.theme_colors["field_border"],
        )
        self.left_zoom_canvas.pack()

        # Add title to left zoom
        self.left_zoom_canvas.create_text(
            zoom_width // 2,
            15,
            text=DialogHelper.t("Left Side"),
            fill=self.theme_colors["text"],
            font=("Arial", 10, "bold"),
        )

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

        # Add title to right zoom
        self.right_zoom_canvas.create_text(
            zoom_width // 2,
            15,
            text=DialogHelper.t("Right Side"),
            fill=self.theme_colors["text"],
            font=("Arial", 10, "bold"),
        )

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
        # Don't switch if current mode is the same
        if self.current_mode == new_mode:
            return

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

                # Also update compartment labels
                self._update_compartment_labels()
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

                # Calculate expected depth range
                expected_range = depth_to - depth_from

                # Calculate compartment count
                compartment_count = expected_range // interval
                if compartment_count > 0:
                    # Update marker-to-compartment mapping
                    new_mapping = {}
                    for i in range(compartment_count):
                        marker_id = 4 + i
                        compartment_depth = depth_from + ((i + 1) * interval)
                        new_mapping[marker_id] = int(compartment_depth)

                    # Apply mapping and refresh visualization
                    self.marker_to_compartment = new_mapping
                    self._update_visualization()
                    self.compartment_interval = interval

                    # Update metadata dict
                    self.metadata["hole_id"] = self.hole_id.get().strip()
                    self.metadata["depth_from"] = depth_from
                    self.metadata["depth_to"] = depth_to
                    self.metadata["compartment_interval"] = interval

                    # After updating metadata, propagate to main app
                    if hasattr(self, "app") and hasattr(
                        self.app, "update_last_metadata"
                    ):
                        try:
                            self.app.update_last_metadata(
                                self.metadata.get("hole_id"),
                                self.metadata.get("depth_from"),
                                self.metadata.get("depth_to"),
                                self.metadata.get("compartment_interval", 1),
                            )
                            self.logger.debug(
                                f"Dialog updated last metadata on input change: {self.metadata}"
                            )
                        except Exception as ex:
                            self.logger.warning(
                                f"Error propagating metadata to app: {ex}"
                            )
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

    def _on_increment_from_last(self):
        """Handle Increment From Last button click."""
        try:
            # Check if we have app reference
            if not self.app or not hasattr(self.app, "get_incremented_metadata"):
                DialogHelper.show_message(
                    self.dialog,
                    "Not Available",
                    "Increment from last is not available",
                    message_type="warning",
                )
                return

            # Get incremented metadata from app
            incremented = self.app.get_incremented_metadata()

            if incremented and incremented.get("hole_id"):
                # Set values
                self.hole_id.set(incremented.get("hole_id", ""))
                self.depth_from.set(str(incremented.get("depth_from", "")))

                # Get the currently selected interval from the dropdown
                current_interval = self.interval_var.get()

                # Calculate depth_to based on the current interval selection, not the stored one
                if incremented.get("depth_from") is not None:
                    # Calculate depth_to as depth_from + (20 * current_interval)
                    depth_to = incremented.get("depth_from") + (20 * current_interval)
                    self.depth_to.set(str(depth_to))
                else:
                    self.depth_to.set(str(incremented.get("depth_to", "")))

                # Update compartment labels
                self._update_compartment_labels()

                # Update status message
                self.status_var.set(
                    f"Auto-filled with incremented metadata from last core"
                )
            else:
                # Show message if no previous metadata
                DialogHelper.show_message(
                    self.dialog,
                    "No Previous Data",
                    "No previous metadata available to increment from.",
                    message_type="info",
                )
        except Exception as e:
            self.logger.error(f"Error in increment from last: {str(e)}")
            DialogHelper.show_message(
                self.dialog,
                "Error",
                f"Failed to increment from last: {str(e)}",
                message_type="error",
            )

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
                    if current_id in [0, 1]:
                        return DialogHelper.t(
                            "Click to set the TOP boundary constraint"
                        )
                    else:
                        return DialogHelper.t(
                            "Click to set the BOTTOM boundary constraint"
                        )
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
                    if current_id in [0, 1]:
                        self.status_var.set(
                            DialogHelper.t(
                                "Click anywhere to set where the TOP edge of compartments should be"
                            )
                        )
                    else:
                        self.status_var.set(
                            DialogHelper.t(
                                "Click anywhere to set where the BOTTOM edge of compartments should be"
                            )
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

            # If there's an external callback for applying adjustments, call it
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
        """Handle canvas click events based on current mode."""
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

            # Ensure the click is within the image bounds
            if self.source_image is None:
                return

            img_height, img_width = self.source_image.shape[:2]
            if not (0 <= image_x < img_width and 0 <= image_y < img_height):
                return

            # Handle click based on current mode
            if self.current_mode == self.MODE_METADATA:
                # In metadata mode, clicks just update the zoom lens
                # This is handled by motion events
                pass

            elif self.current_mode == self.MODE_MISSING_BOUNDARIES:
                # Only handle placement in missing boundaries mode if not complete
                if self.annotation_complete or not self.missing_marker_ids:
                    return

                # Check if we're still placing markers
                if self.current_index >= len(self.missing_marker_ids):
                    return

                # Get current marker ID
                current_id = self.missing_marker_ids[self.current_index]

                # Check if click position overlaps with any existing marker
                click_point = np.array([image_x, image_y])

                # Check against all existing markers
                for existing_id, existing_corners in self.markers.items():
                    # Check if click is inside existing marker bounds
                    if (
                        cv2.pointPolygonTest(
                            existing_corners.astype(np.int32), (image_x, image_y), False
                        )
                        >= 0
                    ):
                        self.status_var.set(
                            DialogHelper.t(
                                f"Cannot place marker here - overlaps with marker {existing_id}"
                            )
                        )
                        return

                # Also check against already placed markers in this session
                for placed_id, placed_boundary in self.result_boundaries.items():
                    if placed_id == 24:  # Metadata marker
                        # Check distance from center
                        center = np.mean(placed_boundary, axis=0)
                        if (
                            np.linalg.norm(click_point - center) < 30
                        ):  # 30 pixel minimum distance
                            self.status_var.set(
                                DialogHelper.t(
                                    f"Cannot place marker here - too close to marker {placed_id}"
                                )
                            )
                            return
                    # FIXED: Check if corner marker (stored as numpy array)

                    elif placed_id in [0, 1, 2, 3]:  # Corner markers
                        if isinstance(placed_boundary, np.ndarray):
                            # Check distance from center for corner markers
                            center = np.mean(placed_boundary, axis=0)
                            if (
                                np.linalg.norm(click_point - center) < 30
                            ):  # 30 pixel minimum distance
                                self.status_var.set(
                                    DialogHelper.t(
                                        f"Cannot place marker here - too close to corner marker {placed_id}"
                                    )
                                )
                                return

                    else:  # Compartment marker
                        # Ensure it's a tuple before unpacking
                        if (
                            isinstance(placed_boundary, tuple)
                            and len(placed_boundary) == 4
                        ):
                            x1, y1, x2, y2 = placed_boundary
                            # Check if click is within compartment bounds
                            if x1 <= image_x <= x2 and y1 <= image_y <= y2:
                                self.status_var.set(
                                    DialogHelper.t(
                                        f"Cannot place marker here - overlaps with compartment {placed_id}"
                                    )
                                )
                                return
                        elif isinstance(placed_boundary, np.ndarray):
                            # Handle case where compartment boundary might be stored as array
                            self.logger.warning(
                                f"Unexpected array format for compartment {placed_id}"
                            )

                # Only check overlap for compartment markers, not metadata or corner markers
                if current_id not in [24, 0, 1, 2, 3]:
                    if self._would_overlap_existing(image_x):
                        self.status_var.set(
                            DialogHelper.t(
                                "Cannot place compartment here - overlaps with existing compartment"
                            )
                        )
                        return
                # Handle marker placement based on type
                if current_id == 24:  # Metadata marker
                    # Create a square marker centered on the click
                    marker_size = 20  # Size in pixels
                    half_size = marker_size // 2

                    # Create four corners of a square centered on the click point
                    corners = np.array(
                        [
                            [image_x - half_size, image_y - half_size],  # Top-left
                            [image_x + half_size, image_y - half_size],  # Top-right
                            [image_x + half_size, image_y + half_size],  # Bottom-right
                            [image_x - half_size, image_y + half_size],  # Bottom-left
                        ],
                        dtype=np.float32,
                    )

                    # Store the marker corners
                    self.result_boundaries[current_id] = corners

                    # Display feedback
                    self.status_var.set(
                        DialogHelper.t(
                            f"Placed metadata marker at x={image_x}, y={image_y}"
                        )
                    )

                elif current_id in [0, 1, 2, 3]:  # Corner markers
                    # Create a square marker for corner markers
                    marker_size = 30  # Slightly larger than metadata marker
                    half_size = marker_size // 2

                    # Use the actual click position for both X and Y
                    marker_x = image_x  # Use clicked X position
                    marker_y = image_y  # Use clicked Y position

                    # Create four corners of a square centered on the click point
                    corners = np.array(
                        [
                            [marker_x - half_size, marker_y - half_size],  # Top-left
                            [marker_x + half_size, marker_y - half_size],  # Top-right
                            [
                                marker_x + half_size,
                                marker_y + half_size,
                            ],  # Bottom-right
                            [marker_x - half_size, marker_y + half_size],  # Bottom-left
                        ],
                        dtype=np.float32,
                    )

                    # Store the marker corners
                    self.result_boundaries[current_id] = corners

                    # Also update corner_markers for immediate use
                    if (
                        not hasattr(self, "corner_markers")
                        or self.corner_markers is None
                    ):
                        self.corner_markers = {}
                    self.corner_markers[current_id] = corners

                    # Update markers dictionary as well
                    self.markers[current_id] = corners

                    # Display feedback
                    if current_id in [0, 1]:
                        self.status_var.set(
                            DialogHelper.t(f"Set TOP constraint at y={image_y}")
                        )
                    else:
                        self.status_var.set(
                            DialogHelper.t(f"Set BOTTOM constraint at y={image_y}")
                        )

                    # ===================================================
                    # Recalculate vertical constraints after placing corner marker
                    # ===================================================
                    self._calculate_vertical_constraints()
                    self.logger.debug(
                        f"Updated vertical constraints after placing corner marker: top_y={self.top_y}, bottom_y={self.bottom_y}"
                    )

                    # ===================================================
                    # Remove the paired corner marker from missing list
                    # ===================================================
                    if current_id in [0, 1]:  # Placed a top corner
                        # Remove the other top corner from missing list
                        other_top = 1 if current_id == 0 else 0
                        if other_top in self.missing_marker_ids:
                            self.missing_marker_ids.remove(other_top)
                            self.logger.debug(
                                f"Removed marker {other_top} from missing list - already have a top corner"
                            )
                    elif current_id in [2, 3]:  # Placed a bottom corner
                        # Remove the other bottom corner from missing list
                        other_bottom = 3 if current_id == 2 else 2
                        if other_bottom in self.missing_marker_ids:
                            self.missing_marker_ids.remove(other_bottom)
                            self.logger.debug(
                                f"Removed marker {other_bottom} from missing list - already have a bottom corner"
                            )

                else:  # Compartment marker
                    # Calculate the compartment boundaries using x-coordinate
                    half_width = self.avg_width // 2
                    x1 = max(0, image_x - half_width)

                    # Calculate y-coordinates based on x-position using the slope of boundary lines
                    img_width = self.source_image.shape[1]

                    # Calculate top and bottom boundary y values at this x position
                    left_top_y = self.top_y + self.left_height_offset
                    right_top_y = self.top_y + self.right_height_offset
                    left_bottom_y = self.bottom_y + self.left_height_offset
                    right_bottom_y = self.bottom_y + self.right_height_offset

                    # Calculate slopes
                    top_y_slope = (
                        (right_top_y - left_top_y) / img_width if img_width > 0 else 0
                    )
                    y1 = int(left_top_y + (top_y_slope * x1))

                    x2 = min(img_width - 1, image_x + half_width)
                    bottom_y_slope = (
                        (right_bottom_y - left_bottom_y) / img_width
                        if img_width > 0
                        else 0
                    )
                    y2 = int(left_bottom_y + (bottom_y_slope * x2))

                    # Store the boundary
                    self.result_boundaries[current_id] = (x1, y1, x2, y2)

                    # Get depth for display using marker_to_compartment mapping
                    depth = self.marker_to_compartment.get(current_id, current_id - 3)
                    self.status_var.set(
                        DialogHelper.t(
                            f"Placed compartment at depth {depth}m (x={image_x})"
                        )
                    )

                # Move to next marker
                self.current_index += 1
                self.temp_point = None

                # Check if we're done with all missing markers
                if self.current_index >= len(self.missing_marker_ids):
                    self.annotation_complete = True
                    self.status_var.set(
                        DialogHelper.t(
                            "All missing markers have been annotated. Click 'Continue to Adjustment' to proceed."
                        )
                    )

                # Force full update when new boundary is added
                # Update visualization and status
                self._update_visualization(force_full_update=True)
                self._update_status_message()

            elif self.current_mode == self.MODE_ADJUST_BOUNDARIES:
                # In adjustment mode, clicks do nothing - adjustments are via buttons
                pass

        except Exception as e:
            self.logger.error(f"Error handling canvas click: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _on_canvas_move(self, event):
        """Handle mouse movement on canvas for zoom lens and previews."""
        try:
            # Convert canvas coordinates to image coordinates TODO - CHECK
            image_x = int((event.x - self.canvas_offset_x) / self.scale_ratio)
            image_y = int((event.y - self.canvas_offset_y) / self.scale_ratio)

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
                and not self.annotation_complete
                and self.temp_point is not None
            ):

                # Check if we actually need to update
                if self.missing_marker_ids and self.current_index < len(
                    self.missing_marker_ids
                ):
                    current_id = self.missing_marker_ids[self.current_index]

                    # For corner markers, we only need to update if Y changed significantly
                    if current_id in [0, 1, 2, 3]:
                        if (
                            old_temp_point
                            and abs(old_temp_point[1] - self.temp_point[1]) < 5
                        ):
                            # Y didn't change much, skip update
                            pass
                        else:
                            self._update_visualization()
                    # For compartment markers, check if X changed significantly
                    elif (
                        old_temp_point
                        and abs(old_temp_point[0] - self.temp_point[0]) < 5
                    ):
                        # X didn't change much, skip update
                        pass
                    else:
                        self._update_visualization()

            # Always update zoom lens (it's lightweight)
            if self.current_mode == self.MODE_METADATA:
                if hasattr(self, "_zoom_lens") and self._zoom_lens:
                    if event.state & 0x400:  # Right button mask
                        self._render_zoom(self.canvas, event, flipped=True)
                    else:
                        self._render_zoom(self.canvas, event)
            elif self.current_mode == self.MODE_MISSING_BOUNDARIES:
                if hasattr(self, "_zoom_lens") and self._zoom_lens:
                    if (
                        self.missing_marker_ids
                        and self.current_index < len(self.missing_marker_ids)
                        and self.missing_marker_ids[self.current_index]
                        not in [0, 1, 2, 3]
                    ):
                        self._render_zoom(self.canvas, event)

        except Exception as e:
            self.logger.error(f"Error handling canvas movement: {str(e)}")

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

    def _on_key_press(self, event):
        """Handle key press events."""
        if event.char == "`":  # Backtick key
            # Toggle wall detection visualization
            self.show_wall_detection = not getattr(self, "show_wall_detection", False)
            self.status_var.set(
                f"Wall detection visualization: {'ON' if self.show_wall_detection else 'OFF'}"
            )
            self._update_visualization(force_full_update=True)

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
        zoom_width, zoom_height = getattr(self, "_zoom_width", 250), getattr(
            self, "_zoom_height", 350
        )
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
                    half_width = self.avg_width // 2

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

                # Position flipped lens (always follow cursor in metadata mode)
                global_x = canvas.winfo_rootx() + event.x - zoom_width - 10
                global_y = canvas.winfo_rooty() + event.y + 10

                self._zoom_lens_flipped.geometry(
                    f"{zoom_width}x{zoom_height}+{global_x}+{global_y}"
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
                    # Original positioning for metadata mode
                    global_x = canvas.winfo_rootx() + event.x + 10
                    global_y = canvas.winfo_rooty() + event.y + 10
                    lens_x = global_x
                    lens_y = global_y

                self._zoom_lens.geometry(
                    f"{zoom_width}x{zoom_height}+{lens_x}+{lens_y}"
                )
                self._zoom_lens.deiconify()

        except Exception as e:
            self.logger.error(f"Error rendering zoom lens: {str(e)}")
            # Don't print full traceback for zoom rendering

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

        # Check if current marker is a corner marker
        if current_id in [0, 1, 2, 3]:
            # Draw a horizontal line across the entire image width
            line_y = self.temp_point[1]

            # Choose color based on corner type
            if current_id in [0, 1]:  # Top corners
                line_color = (0, 255, 0)  # Green for top
                constraint_text = "TOP CONSTRAINT"
            else:  # Bottom corners (2, 3)
                line_color = (0, 0, 255)  # Red for bottom
                constraint_text = "BOTTOM CONSTRAINT"

            # Draw the horizontal line across entire width
            cv2.line(viz_image, (0, line_y), (img_width - 1, line_y), line_color, 2)

            # Add text label in the center
            text_size = cv2.getTextSize(
                constraint_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )[0]
            text_x = (img_width - text_size[0]) // 2

            # Add background for text
            cv2.rectangle(
                viz_image,
                (text_x - 10, line_y - text_size[1] - 5),
                (text_x + text_size[0] + 10, line_y + 5),
                (0, 0, 0),
                -1,
            )

            # Draw the text
            cv2.putText(
                viz_image,
                constraint_text,
                (text_x, line_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                line_color,
                2,
            )

        elif current_id == 24:  # Metadata marker
            # Draw a square marker preview
            marker_size = 20  # Size in pixels
            half_size = marker_size // 2

            # Calculate center point for the preview
            center_x = x
            center_y = self.temp_point[1]

            # Draw preview square
            cv2.rectangle(
                viz_image,
                (center_x - half_size, center_y - half_size),
                (center_x + half_size, center_y + half_size),
                (255, 0, 255),
                2,
            )  # Purple outline

            # Draw center point
            cv2.circle(viz_image, (center_x, center_y), 5, (255, 0, 255), -1)

            # Add text
            cv2.putText(
                viz_image,
                "24",
                (center_x, center_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 255),
                2,
            )

            # Draw the OCR extraction regions as preview
            # Horizontal range
            region_x1 = max(0, center_x - 130)
            region_x2 = min(img_width - 1, center_x + 130)

            # Hole ID region (above marker)
            hole_id_region_y1 = max(0, center_y - 130)
            hole_id_region_y2 = max(0, center_y - 20)
            cv2.rectangle(
                viz_image,
                (region_x1, hole_id_region_y1),
                (region_x2, hole_id_region_y2),
                (255, 0, 0),
                2,
            )  # Blue for hole ID

            # Add "Hole ID Region" label
            cv2.putText(
                viz_image,
                "Hole ID",
                (region_x1 + 10, hole_id_region_y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

            # Depth region (below marker)
            depth_region_y1 = min(img_height - 1, center_y + 20)
            depth_region_y2 = min(img_height - 1, center_y + 150)
            cv2.rectangle(
                viz_image,
                (region_x1, depth_region_y1),
                (region_x2, depth_region_y2),
                (0, 255, 0),
                2,
            )  # Green for depth

            # Add "Depth Region" label
            cv2.putText(
                viz_image,
                "Depth",
                (region_x1 + 10, depth_region_y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        else:  # Compartment marker
            # Calculate the compartment boundaries for preview
            half_width = self.avg_width // 2
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

        # Draw all static elements
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
                            tolerance_pixels = 2.0
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
                                tolerance_pixels = 2.0
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

        # Add scale bar at bottom left
        if scale_px_per_cm:
            # Calculate scale bar dimensions
            scale_bar_length_cm = 10  # 10cm scale bar
            scale_bar_length_px = int(scale_bar_length_cm * scale_px_per_cm)
            scale_bar_height = 20
            margin = 20

            # Position at bottom left
            bar_x = margin
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

            cv2.polylines(static_viz, [rotated_corners], True, (0, 255, 0), 2)

            # Add compartment depth label
            mid_x = center_x
            mid_y = rect_center_y

            # Find the nearest marker below this compartment boundary
            nearest_marker_id = None
            nearest_distance = float("inf")

            for marker_id, corners in self.markers.items():
                if marker_id in self.config.get("compartment_marker_ids", range(4, 24)):
                    marker_center_x = int(np.mean(corners[:, 0]))
                    marker_center_y = int(np.mean(corners[:, 1]))

                    if abs(marker_center_x - mid_x) < (x2 - x1) // 2:
                        vertical_dist = abs(marker_center_y - center_bottom_y)
                        if vertical_dist < nearest_distance:
                            nearest_distance = vertical_dist
                            nearest_marker_id = marker_id

            if nearest_marker_id is not None and hasattr(self, "marker_to_compartment"):
                depth_label = self.marker_to_compartment.get(
                    nearest_marker_id, nearest_marker_id - 3
                )

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
                    (int(mid_x) - text_size[0] // 2, int(mid_y) + text_size[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 165, 0),
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

        # ===================================================
        # INSERT: Draw wall detection visualization if enabled
        # Draw wall detection elements if enabled and available
        if getattr(self, "show_wall_detection", False) and hasattr(
            self, "boundary_analysis"
        ):
            wall_viz = self.boundary_analysis.get("wall_detection_results", {})
            if wall_viz:
                # Draw search regions
                search_regions = wall_viz.get("search_regions", [])
                for x1, y1, x2, y2, color, thickness in search_regions:
                    cv2.rectangle(static_viz, (x1, y1), (x2, y2), color, thickness)
                    # Add label
                    cv2.putText(
                        static_viz,
                        "Search",
                        (x1 + 2, y1 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                    )

                # Draw detected edges
                detected_edges = wall_viz.get("detected_edges", [])
                for x1, y1, x2, y2, color, thickness in detected_edges:
                    cv2.line(static_viz, (x1, y1), (x2, y2), color, thickness)

                # Draw final boundaries (if different from regular boundaries)
                final_boundaries = wall_viz.get("final_boundaries", [])
                if final_boundaries and self.show_wall_detection:
                    # Draw with dashed lines to distinguish
                    for x1, y1, x2, y2, color, thickness in final_boundaries:
                        # Draw dashed rectangle
                        self._draw_dashed_rectangle(
                            static_viz, (x1, y1), (x2, y2), color, thickness
                        )
        # ===================================================

        # Cache the result
        self.static_viz_cache = static_viz.copy()
        self.static_viz_params = current_params

        return static_viz

    def _draw_dashed_rectangle(self, img, pt1, pt2, color, thickness=1, dash_length=5):
        """Draw a dashed rectangle on the image."""
        x1, y1 = pt1
        x2, y2 = pt2

        # Draw dashed lines
        # Top
        for i in range(x1, x2, dash_length * 2):
            cv2.line(img, (i, y1), (min(i + dash_length, x2), y1), color, thickness)
        # Bottom
        for i in range(x1, x2, dash_length * 2):
            cv2.line(img, (i, y2), (min(i + dash_length, x2), y2), color, thickness)
        # Left
        for i in range(y1, y2, dash_length * 2):
            cv2.line(img, (x1, i), (x1, min(i + dash_length, y2)), color, thickness)
        # Right
        for i in range(y1, y2, dash_length * 2):
            cv2.line(img, (x2, i), (x2, min(i + dash_length, y2)), color, thickness)

    # ===================================================

    def _undo_last(self):
        """Undo the last annotation."""
        try:
            if self.current_mode == self.MODE_METADATA:
                # Nothing to undo in metadata mode
                self.status_var.set(DialogHelper.t("Nothing to undo in metadata mode"))
                return

            elif self.current_mode == self.MODE_MISSING_BOUNDARIES:
                # In boundary mode, undo the last placed marker
                if self.annotation_complete:
                    # If annotation is complete, undo the last compartment
                    if self.result_boundaries and self.current_index > 0:
                        self.current_index -= 1
                        current_id = self.missing_marker_ids[self.current_index]
                        if current_id in self.result_boundaries:
                            del self.result_boundaries[current_id]
                        self.annotation_complete = False

                        # Get depth for display using marker_to_compartment mapping
                        depth = self.marker_to_compartment.get(
                            current_id, current_id - 3
                        )
                        self.status_var.set(
                            DialogHelper.t(
                                f"Undid compartment at depth {depth}m. Please place it again."
                            )
                        )
                else:
                    # If we're in the middle of annotations, undo the last one
                    if self.current_index > 0:
                        self.current_index -= 1
                        current_id = self.missing_marker_ids[self.current_index]
                        if current_id in self.result_boundaries:
                            del self.result_boundaries[current_id]

                        # Get depth for display using marker_to_compartment mapping
                        depth = self.marker_to_compartment.get(
                            current_id, current_id - 3
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
            # Make sure dialog exists
            if not hasattr(self, "dialog") or not self.dialog:
                self.logger.error("Cannot show dialog - dialog does not exist")
                return {}

            # Ensure the dialog is visible
            self.dialog.deiconify()
            self.dialog.update_idletasks()  # Force layout calculation

            # For landscape orientation - maximize width, fit height to content
            screen_width = self.dialog.winfo_screenwidth()
            screen_height = self.dialog.winfo_screenheight()

            # Use nearly full screen width
            desired_width = int(screen_width * 0.95)

            # Get natural height
            natural_height = self.dialog.winfo_reqheight()

            # Constrain height if needed
            max_height = int(screen_height * 0.9)
            if natural_height > max_height:
                natural_height = max_height

            # Calculate center position manually for landscape
            if (
                self.parent
                and self.parent.winfo_exists()
                and self.parent.winfo_viewable()
            ):
                # Center on parent
                parent_x = self.parent.winfo_rootx()
                parent_y = self.parent.winfo_rooty()
                parent_width = self.parent.winfo_width()
                parent_height = self.parent.winfo_height()

                x = parent_x + (parent_width - desired_width) // 2
                y = parent_y + (parent_height - natural_height) // 2
            else:
                # Center on screen
                x = (screen_width - desired_width) // 2
                y = (screen_height - natural_height) // 2

            # Ensure dialog stays on screen
            screen_margin = 50
            taskbar_margin = 100
            x = max(screen_margin, min(x, screen_width - desired_width - screen_margin))
            y = max(
                screen_margin, min(y, screen_height - natural_height - taskbar_margin)
            )

            # Apply the landscape geometry directly
            self.dialog.geometry(f"{desired_width}x{natural_height}+{x}+{y}")

            # Set transient AFTER positioning
            if self.parent:
                self.dialog.transient(self.parent)

            # Ensure dialog is visible and on top
            self.dialog.deiconify()
            self.dialog.lift()
            self.dialog.focus_force()

            # Wait for dialog to close
            self.logger.debug("About to call wait_window")
            self.dialog.wait_window()
            self.logger.debug("wait_window completed - dialog was closed by user")

            # Return the final results
            if hasattr(self, "final_results"):
                return self.final_results
            else:
                # Create a default result dict
                result = {
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
                }
                return result

        except Exception as e:
            self.logger.error(f"Error showing dialog: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {}
