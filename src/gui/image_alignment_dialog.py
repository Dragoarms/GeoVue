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
import gc

from gui.compartment_registration_dialog import CompartmentRegistrationDialog
from gui.dialog_helper import DialogHelper
from processing.ArucoMarkersAndBlurDetectionStep.aruco_manager import ArucoManager


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
        app=None,
        initial_mode=None,
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
            "initial_mode": initial_mode,  # Pass through initial_mode
        }

        # Image transformation state
        self.cumulative_rotation = 0.0  # Total rotation applied
        self.cumulative_offset_y = 0.0  # Total vertical offset
        self.transform_center = None  # Will be calculated

        # Store working copies - avoid copying large images unnecessarily
        self.source_image = image.copy() if image is not None else None
        
        # Check for major orientation issues BEFORE creating dialog
        # This handles 90°, 180°, 270° rotations from camera orientation
        if self.source_image is not None and markers:
            self.source_image, self._major_rotation_applied = self._check_and_fix_major_orientation(
                self.source_image, markers, config
            )

            # Re-detect markers on the rotated image so boundary logic uses correct positions
            aruco = ArucoManager(config)
            markers = aruco.improve_marker_detection(self.source_image)
            self.original_params["markers"] = markers  # update forwarded params
        else:
            self._major_rotation_applied = 0.0
            
        self.transformed_image = (
            self.source_image.copy() if self.source_image is not None else None
        )

        # Store major rotation in params for reference by callers
        self.original_params["major_rotation_applied"] = self._major_rotation_applied

        # Store reference to original image but don't copy it yet (memory optimization)
        # We'll copy it only when we need to transform it
        self._original_image_ref = original_image
        self.high_res_image = None  # Will be created when needed
        self._using_reference_mode = (
            False  # Track if we're using reference mode for large images
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

    def _check_and_fix_major_orientation(
            self, 
            image: np.ndarray, 
            markers: Dict[int, np.ndarray],
            config: Dict
        ) -> Tuple[np.ndarray, float]:
            """
            Check and fix major orientation issues (90°, 180°, 270°) before fine adjustment.
            
            Args:
                image: Source image
                markers: Detected marker positions
                config: Configuration with marker IDs
                
            Returns:
                Tuple of (corrected_image, rotation_angle_applied)
            """
            if not markers:
                return image, 0.0
                
            corner_ids = config.get("corner_marker_ids", [0, 1, 2, 3])
            compartment_ids = config.get("compartment_marker_ids", list(range(4, 24)))
            
            h, w = image.shape[:2]
            
            # Get detected markers by type
            corner_markers = [(mid, markers[mid]) for mid in corner_ids if mid in markers]
            comp_markers = [
                (mid, markers[mid]) 
                for mid in compartment_ids 
                if mid in markers and mid != 24
            ]
            
            # Need minimum markers
            if len(corner_markers) < 1 or len(comp_markers) < 2:
                self.logger.debug("Insufficient markers for major orientation detection")
                return image, 0.0
                
            # Analyze marker positions
            is_landscape = w > h
            
            # Calculate average positions
            corner_centers = [np.mean(corners, axis=0) for _, corners in corner_markers]
            comp_centers = [(mid, np.mean(corners, axis=0)) for mid, corners in comp_markers]
            
            corner_avg = np.mean(corner_centers, axis=0)
            comp_avg = np.mean([c for _, c in comp_centers], axis=0)
            
            # Check marker ID ordering vs X position
            comp_centers_sorted = sorted(comp_centers, key=lambda x: x[0])  # By ID
            comp_centers_by_x = sorted(comp_centers, key=lambda x: x[1][0])  # By X position
            
            id_order_correct = True
            if len(comp_centers_sorted) >= 2:
                ids_by_id = [c[0] for c in comp_centers_sorted]
                ids_by_x = [c[0] for c in comp_centers_by_x]
                correlation = np.corrcoef(ids_by_id, ids_by_x)[0, 1]
                id_order_correct = correlation > 0
                
            # Determine rotation needed
            rotation_angle = 0.0
            rotated_image = image
            
            if not is_landscape:
                # Portrait orientation - needs 90° rotation
                if corner_avg[0] > comp_avg[0]:  # Corners to the right
                    self.logger.info("Portrait image - rotating 90° counter-clockwise")
                    rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    rotation_angle = 90.0
                else:
                    self.logger.info("Portrait image - rotating 90° clockwise")
                    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    rotation_angle = -90.0
            else:
                # Landscape - check if upside down
                if comp_avg[1] < corner_avg[1] or not id_order_correct:
                    self.logger.info("Landscape image upside down - rotating 180°")
                    rotated_image = cv2.rotate(image, cv2.ROTATE_180)
                    rotation_angle = 180.0
                    
            if rotation_angle != 0.0:
                self.logger.info(f"Applied major orientation correction: {rotation_angle}°")
                
            return rotated_image, rotation_angle

    def _get_high_res_image(self):
        """Get high-res image, creating copy only when needed (memory optimization)."""
        if self.high_res_image is None and self._original_image_ref is not None:
            try:
                # Check available memory before copying large image
                import psutil

                available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)

                # Estimate memory needed for image copy (height * width * channels * bytes_per_pixel)
                if hasattr(self._original_image_ref, "shape"):
                    h, w = self._original_image_ref.shape[:2]
                    channels = (
                        self._original_image_ref.shape[2]
                        if len(self._original_image_ref.shape) > 2
                        else 1
                    )
                    estimated_mb = (h * w * channels) / (1024 * 1024)

                    self.logger.info(
                        f"Creating high-res image copy: {w}x{h}x{channels} (~{estimated_mb:.1f} MB)"
                    )
                    self.logger.info(f"Available memory: {available_memory_mb:.1f} MB")

                    if (
                        estimated_mb > available_memory_mb * 0.5
                    ):  # Don't use more than 50% of available memory
                        self.logger.warning(
                            f"Large image ({estimated_mb:.1f} MB) may cause memory issues."
                        )
                        self.logger.warning(
                            f"Available memory: {available_memory_mb:.1f} MB"
                        )

                        # Instead of automatically resizing, use reference and warn user
                        # This preserves full quality but uses minimal memory until transformation
                        self.logger.info(
                            "Using memory-efficient reference mode to preserve image quality"
                        )
                        self.high_res_image = (
                            self._original_image_ref
                        )  # Use reference, not copy
                        self._using_reference_mode = True
                    else:
                        # Safe to copy - sufficient memory available
                        self.high_res_image = self._original_image_ref.copy()
                        self._using_reference_mode = False
                else:
                    # Fallback if shape is not available
                    self.high_res_image = self._original_image_ref.copy()

            except ImportError:
                # psutil not available, proceed without memory check
                self.logger.warning("psutil not available for memory checking")
                self.high_res_image = self._original_image_ref.copy()
            except Exception as e:
                self.logger.error(f"Error creating high-res image copy: {e}")
                self.high_res_image = self._original_image_ref.copy()

        return self.high_res_image

    def cleanup_memory(self):
        """Clean up memory by releasing image references."""
        try:
            self.logger.debug("Cleaning up image alignment dialog memory")

            # Clear image references
            self.source_image = None
            self.transformed_image = None
            self.high_res_image = None
            self._original_image_ref = None

            # Clear wrapped dialog reference
            if hasattr(self, "wrapped_dialog"):
                self.wrapped_dialog = None

            # Force garbage collection multiple times for stubborn references
            import gc

            gc.collect()
            gc.collect()  # Sometimes need multiple passes

            self.logger.debug("Image alignment dialog memory cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during memory cleanup: {e}")

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
        high_res_image = self._get_high_res_image()
        if high_res_image is not None:
            h, w = high_res_image.shape[:2]

            # Scale the transform center to high-res coordinates
            scale_factor = 1.0
            if self.source_image is not None and high_res_image is not None:
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
                high_res_image = self._get_high_res_image()

                if M is not None and high_res_image is not None:
                    self.logger.info(
                        f"Applying final transformation to high-res image: "
                        f"rotation={self.cumulative_rotation:.2f}°, offset_y={self.cumulative_offset_y:.1f}px"
                    )

                    # Check if we're in reference mode and need to handle memory carefully
                    if (
                        hasattr(self, "_using_reference_mode")
                        and self._using_reference_mode
                    ):
                        self.logger.info(
                            "Applying transformation in memory-efficient mode"
                        )
                        # Transform directly without additional copying
                        transformed_highres = cv2.warpAffine(
                            high_res_image,
                            M,
                            (w, h),
                            flags=cv2.INTER_LANCZOS4,  # High quality for final
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(255, 255, 255),
                        )
                    else:
                        # Normal transformation with copied image
                        transformed_highres = cv2.warpAffine(
                            high_res_image,
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
                    result["transform_center"] = self.transform_center  # Add this!

                    # Boundaries stay the same - no transformation needed!
                    result["boundaries_need_transformation"] = False

        # Ensure cleanup happens even if wrapped_dialog doesn't call it
        try:
            if hasattr(self, "wrapped_dialog"):
                # Clean up wrapped dialog's resources
                if hasattr(self.wrapped_dialog, "_cleanup_all_resources"):
                    self.wrapped_dialog._cleanup_all_resources()
        except Exception as e:
            self.logger.warning(f"Error cleaning wrapped dialog: {e}")

        # Clean up our own memory
        self.cleanup_memory()

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

        # Extract initial_mode before passing to parent (parent doesn't accept it)
        initial_mode = kwargs.pop("initial_mode", None)

        # Extract major_rotation_applied before passing to parent
        # (CompartmentRegistrationDialog.__init__ does not accept this kwarg)
        self.major_rotation_applied = kwargs.pop("major_rotation_applied", None)
        logger.debug(
            f"major_rotation_applied from kwargs: {self.major_rotation_applied}"
        )

        # Ensure app reference is available
        if "app" not in kwargs and hasattr(alignment_handler, "app"):
            kwargs["app"] = alignment_handler.app
            logger.debug(f"Added app to kwargs from alignment_handler")

        logger.debug(
            f"Final kwargs keys before super().__init__: {list(kwargs.keys())}"
        )

        # Call parent init ONCE
        super().__init__(**kwargs)

        # Reference the alignment handler for transformation state
        self.image_aligner = alignment_handler  # Reference to the alignment handler

        # DO NOT initialize transformation attributes here - they should be read from
        # the alignment_handler dynamically via properties to avoid stale state

        # If initial_mode was provided and differs from what parent determined, override it
        if initial_mode is not None and initial_mode != self.current_mode:
            logger.debug(
                f"Overriding parent mode {self.current_mode} with requested mode {initial_mode}"
            )
            self.current_mode = initial_mode

        logger.debug("=== END ImageAlignmentInternalDialog INIT ===")

    @property
    def transformation_applied(self):
        """Read transformation state from alignment handler."""
        if hasattr(self, "image_aligner"):
            # Consider transformation applied if any rotation or offset exists
            return (
                self.image_aligner.cumulative_rotation != 0
                or self.image_aligner.cumulative_offset_y != 0
            )
        return False

    @property
    def transformation_matrix(self):
        """Get transformation matrix from alignment handler."""
        if hasattr(self, "image_aligner"):
            M, _ = self.image_aligner.get_final_transformation_matrix()
            return M
        return None

    @property
    def cumulative_offset_y(self):
        """Get cumulative offset from alignment handler."""
        if hasattr(self, "image_aligner"):
            return self.image_aligner.cumulative_offset_y
        return 0.0

    @property
    def cumulative_rotation(self):
        """Get cumulative rotation from alignment handler."""
        if hasattr(self, "image_aligner"):
            return self.image_aligner.cumulative_rotation
        return 0.0

    @property
    def transform_center(self):
        """Get transform center from alignment handler."""
        if hasattr(self, "image_aligner"):
            return self.image_aligner.transform_center
        return [0.0, 0.0]

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

        # Transformation state is now read dynamically from alignment_handler via properties
        # Log current transformation state for debugging
        self.logger.info(
            f"Transformation state: rotation={self.cumulative_rotation:.2f}°, "
            f"offset_y={self.cumulative_offset_y:.1f}px, "
            f"matrix shape={self.transformation_matrix.shape if self.transformation_matrix is not None else 'None'}"
        )

        # Update the static visualization cache to use transformed image
        self.static_viz_cache = None  # Force recreation

        # Update visualization
        self._update_visualization()

        self.logger.info(
            f"Transformation applied: rotation={self.alignment_handler.cumulative_rotation:.2f}°, offset_y={self.alignment_handler.cumulative_offset_y:.1f}px"
        )

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
