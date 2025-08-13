# gui/image_alignment_dialog.py

"""
Image Alignment Dialog - A new approach to compartment registration where the image
moves/rotates while boundaries stay fixed and horizontal.
"""

import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
import threading
from PIL import Image, ImageTk

from gui.compartment_registration_dialog import CompartmentRegistrationDialog
from gui.dialog_helper import DialogHelper


class ImageAlignmentDialog:
    """
    Enhanced compartment registration dialog that transforms the image instead of boundaries.

    This approach:
    - Keeps compartment boundaries horizontal and fixed
    - Moves/rotates the image to align with boundaries
    - Provides true WYSIWYG extraction
    - Eliminates coordinate transformation bugs
    """

    def __init__(
        self,
        parent,
        image,
        detected_boundaries,
        missing_marker_ids=None,
        theme_colors=None,
        gui_manager=None,
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
        image_path=None,
        scale_data=None,
        boundary_analysis=None,
        app=None,  # Add app parameter
    ):
        """Initialize the image alignment dialog."""
        self.logger = logging.getLogger(__name__)
        self.parent = parent

        # Get app reference - either from parameter or parent chain
        self.app = app  # Use passed app first
        if not self.app:
            if hasattr(parent, "app"):
                self.app = parent.app
                self.logger.debug(f"Got app reference from parent")
            elif hasattr(parent, "winfo_toplevel"):
                toplevel = parent.winfo_toplevel()
                if hasattr(toplevel, "app"):
                    self.app = toplevel.app
                    self.logger.debug(f"Got app reference from toplevel")
            else:
                self.logger.warning("Could not get app reference")

        # Store original parameters (NOW including the correct app reference)
        self.original_params = {
            "image": image,
            "detected_boundaries": detected_boundaries,
            "missing_marker_ids": missing_marker_ids,
            "theme_colors": theme_colors,
            "gui_manager": gui_manager,
            "original_image": original_image,
            "output_format": output_format,
            "file_manager": file_manager,
            "metadata": metadata,
            "vertical_constraints": vertical_constraints,
            "marker_to_compartment": marker_to_compartment,
            "rotation_angle": rotation_angle,
            "corner_markers": corner_markers,
            "markers": markers,
            "config": config,
            "on_apply_adjustments": on_apply_adjustments,
            "image_path": image_path,
            "scale_data": scale_data,
            "boundary_analysis": boundary_analysis,
            "app": self.app,  # NOW this will have the correct value
        }

        # Image transformation state
        self.cumulative_rotation = 0.0  # Total rotation applied
        self.cumulative_offset_y = 0.0  # Total vertical offset
        self.transform_center = None  # Will be calculated

        # Store working copies
        self.source_image = image.copy() if image is not None else None
        self.transformed_image = (
            self.source_image.copy() if self.source_image is not None else None
        )
        self.high_res_image = (
            original_image.copy() if original_image is not None else None
        )

        # Calculate the center of rotation
        self._calculate_transform_center()

        # Create the base dialog (we'll override some methods)
        self._create_wrapped_dialog()

    def _calculate_transform_center(self):
        """Calculate the center point for image transformations."""
        # Get vertical constraints
        if self.original_params["vertical_constraints"]:
            top_y, bottom_y = self.original_params["vertical_constraints"]
        else:
            # Fallback to image center
            if self.source_image is not None:
                top_y = int(self.source_image.shape[0] * 0.1)
                bottom_y = int(self.source_image.shape[0] * 0.9)
            else:
                top_y, bottom_y = 100, 500

        # Vertical center is middle of constraints
        center_y = (top_y + bottom_y) / 2

        # Get horizontal extents from boundaries
        boundaries = self.original_params["detected_boundaries"]
        if boundaries:
            min_x = min(b[0] for b in boundaries)  # Leftmost boundary
            max_x = max(b[2] for b in boundaries)  # Rightmost boundary
            center_x = (min_x + max_x) / 2
        else:
            # Fallback to image center
            if self.source_image is not None:
                center_x = self.source_image.shape[1] / 2
            else:
                center_x = 400

        self.transform_center = (center_x, center_y)
        self.logger.info(
            f"Transform center calculated at: ({center_x:.1f}, {center_y:.1f})"
        )

    def _create_wrapped_dialog(self):
        """Create the wrapped dialog with overridden adjustment methods."""
        self.logger.debug("=== Creating wrapped dialog ===")
        self.logger.debug(f"self.app: {self.app}")
        self.logger.debug(f"original_params keys: {list(self.original_params.keys())}")
        self.logger.debug(f"original_params['app']: {self.original_params.get('app')}")

        # We'll create a custom dialog that inherits behavior but overrides key methods
        self.wrapped_dialog = ImageAlignmentInternalDialog(
            parent=self.parent, alignment_handler=self, **self.original_params
        )

        self.logger.debug("=== Wrapped dialog created ===")

    def apply_image_transformation(self, rotation_delta=0, offset_y_delta=0):
        """
        Apply transformation to the image instead of moving boundaries.

        Args:
            rotation_delta: Additional rotation in degrees
            offset_y_delta: Additional vertical offset in pixels
        """
        # Update cumulative transformations
        self.cumulative_rotation += rotation_delta
        self.cumulative_offset_y += offset_y_delta

        self.logger.debug(
            f"Applying transformation - rotation: {self.cumulative_rotation:.2f}°, "
            f"offset_y: {self.cumulative_offset_y:.1f}px"
        )

        # Apply transformation to working image
        if self.source_image is not None:
            h, w = self.source_image.shape[:2]
            cx, cy = self.transform_center

            # Create rotation matrix around the calculated center
            M = cv2.getRotationMatrix2D((cx, cy), self.cumulative_rotation, 1.0)

            # Add vertical translation
            M[1, 2] += self.cumulative_offset_y

            # Transform the image
            self.transformed_image = cv2.warpAffine(
                self.source_image,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,  # Fast for preview
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )

            # Update the dialog's display
            if hasattr(self.wrapped_dialog, "update_transformed_image"):
                self.wrapped_dialog.update_transformed_image(self.transformed_image)

    def get_final_transformation_matrix(self):
        """Get the final transformation matrix for high-res image."""
        if self.high_res_image is not None:
            h, w = self.high_res_image.shape[:2]

            # Scale the transform center to high-res coordinates
            scale_factor = 1.0
            if self.source_image is not None and self.high_res_image is not None:
                scale_factor = w / self.source_image.shape[1]

            cx_highres = self.transform_center[0] * scale_factor
            cy_highres = self.transform_center[1] * scale_factor

            # Create transformation matrix
            M = cv2.getRotationMatrix2D(
                (cx_highres, cy_highres), self.cumulative_rotation, 1.0
            )

            # Scale and add vertical translation
            M[1, 2] += self.cumulative_offset_y * scale_factor

            return M, (w, h)

        return None, None

    def show(self):
        """Show the dialog and return results."""
        # Show the wrapped dialog
        result = self.wrapped_dialog.show()

        if result and not result.get("quit") and not result.get("cancelled"):
            # Apply final transformation to high-res image if needed
            if self.cumulative_rotation != 0 or self.cumulative_offset_y != 0:
                M, (w, h) = self.get_final_transformation_matrix()

                if M is not None and self.high_res_image is not None:
                    self.logger.info(
                        f"Applying final transformation to high-res image: "
                        f"rotation={self.cumulative_rotation:.2f}°, offset_y={self.cumulative_offset_y:.1f}px"
                    )

                    # Transform high-res image
                    transformed_highres = cv2.warpAffine(
                        self.high_res_image,
                        M,
                        (w, h),
                        flags=cv2.INTER_LANCZOS4,  # High quality for final
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(255, 255, 255),
                    )

                    # Update result with transformed image
                    result["transformed_image"] = transformed_highres
                    result["transformation_applied"] = True
                    result["transformation_matrix"] = M
                    result["cumulative_rotation"] = self.cumulative_rotation
                    result["cumulative_offset_y"] = self.cumulative_offset_y

                    # Boundaries stay the same - no transformation needed!
                    result["boundaries_need_transformation"] = False

        return result


class ImageAlignmentInternalDialog(CompartmentRegistrationDialog):
    """
    Internal dialog that extends CompartmentRegistrationDialog with image transformation.
    """

    def __init__(self, alignment_handler, **kwargs):
        """Initialize with reference to alignment handler."""
        self.alignment_handler = alignment_handler

        # Debug logging
        logger = logging.getLogger(__name__)
        logger.debug("=== ImageAlignmentInternalDialog INIT ===")
        logger.debug(f"alignment_handler type: {type(alignment_handler)}")
        logger.debug(f"alignment_handler has app: {hasattr(alignment_handler, 'app')}")
        if hasattr(alignment_handler, "app"):
            logger.debug(f"alignment_handler.app: {alignment_handler.app}")

        logger.debug(f"kwargs keys: {list(kwargs.keys())}")
        logger.debug(f"'app' in kwargs: {'app' in kwargs}")
        if "app" in kwargs:
            logger.debug(f"kwargs['app']: {kwargs['app']}")

        # Ensure app reference is available
        if "app" not in kwargs and hasattr(alignment_handler, "app"):
            kwargs["app"] = alignment_handler.app
            logger.debug(f"Added app to kwargs from alignment_handler")

        logger.debug(
            f"Final kwargs keys before super().__init__: {list(kwargs.keys())}"
        )
        logger.debug("=== END ImageAlignmentInternalDialog INIT ===")

        super().__init__(**kwargs)

    def _adjust_height(self, delta):
        """Override: Move image instead of boundaries."""
        # Invert delta because moving image up means boundaries appear to move down
        self.alignment_handler.apply_image_transformation(offset_y_delta=-delta)

        # Update status
        direction = "up" if delta > 0 else "down"
        self.status_var.set(f"Moved image {direction} by {abs(delta)} pixels")

    def _adjust_side_height(self, side, delta):
        """Override: Rotate image instead of creating sloped boundaries."""
        # Calculate rotation based on image width and adjustment
        if self.source_image is not None:
            img_width = self.source_image.shape[1]

            # Calculate rotation angle

            # BEFORE- Approximate: tan(angle) ≈ delta/width for small angles
            # rotation_degrees = np.degrees(np.arctan2(delta, img_width))

            # # Invert for right side
            # if side == "right":
            #     rotation_degrees = -rotation_degrees

            # after: flip the sign so “▲” on the left side → positive (ccw), “▲” on the right → negative (cw)
            rotation_degrees = -np.degrees(np.arctan2(delta, img_width))
            if side == "right":
                rotation_degrees = -rotation_degrees

            self.alignment_handler.apply_image_transformation(
                rotation_delta=rotation_degrees
            )

            self.status_var.set(
                f"Rotated image {abs(rotation_degrees):.3f}° "
                f"({'clockwise' if rotation_degrees < 0 else 'counter-clockwise'})"
            )

    def update_transformed_image(self, transformed_image):
        """Update the display with transformed image."""
        # Store the transformed image
        self.display_image = transformed_image.copy()

        # Update the static visualization cache to use transformed image
        self.static_viz_cache = None  # Force recreation

        # Update visualization
        self._update_visualization()

        # Update zoom windows if in adjustment mode
        if self.current_mode == self.MODE_ADJUST_BOUNDARIES:
            self._update_static_zoom_views()

    def _create_static_visualization(self):
        """Override to use transformed image as base _only_ in adjust mode."""

        # In Metadata or Missing-Boundaries, defer completely to the base implementation
        if self.current_mode in (
            self.MODE_METADATA,
            self.MODE_MISSING_BOUNDARIES,
        ):
            return super()._create_static_visualization()

        # === ADJUST_BOUNDARIES mode: draw strictly horizontal lines ===
        base_image = getattr(self, "display_image", None)
        if base_image is None:
            base_image = self.source_image

        static_viz = base_image.copy()
        h, w = static_viz.shape[:2]

        # horizontal top & bottom
        cv2.line(static_viz, (0, self.top_y), (w, self.top_y), (0, 255, 0), 2)
        cv2.line(static_viz, (0, self.bottom_y), (w, self.bottom_y), (0, 255, 0), 2)

        # compartments all horizontal
        for i, (x1, _, x2, _) in enumerate(self.detected_boundaries):
            cv2.rectangle(
                static_viz,
                (x1, self.top_y),
                (x2, self.bottom_y),
                (0, 255, 0),
                2,
            )
            # label
            marker_id = 4 + i
            if marker_id in self.marker_to_compartment:
                depth = self.marker_to_compartment[marker_id]
                cx, cy = (x1 + x2) // 2, (self.top_y + self.bottom_y) // 2
                text = f"{depth}m"
                sz, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                cv2.rectangle(
                    static_viz,
                    (cx - sz[0] // 2 - 5, cy - sz[1] // 2 - 5),
                    (cx + sz[0] // 2 + 5, cy + sz[1] // 2 + 5),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    static_viz,
                    text,
                    (cx - sz[0] // 2, cy + sz[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 0),
                    2,
                )

        return static_viz

    def _apply_adjustments(self):
        """Override: No boundary adjustment needed - image is already aligned!"""
        # Just trigger the callback if it exists
        if callable(self.on_apply_adjustments):
            # Pass the current boundaries without modification
            adjustment_params = {
                "top_boundary": self.top_y,
                "bottom_boundary": self.bottom_y,
                "left_height_offset": 0,  # Always 0 - no sloped boundaries
                "right_height_offset": 0,  # Always 0 - no sloped boundaries
                "boundaries": self.detected_boundaries,
                "image_transformed": True,
                "transformation_matrix": self.alignment_handler.get_final_transformation_matrix()[
                    0
                ],
            }
            self.on_apply_adjustments(adjustment_params)
