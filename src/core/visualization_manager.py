# visualization_manager.py

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import json
import hashlib
from processing.visualization_drawer import VisualizationDrawer


class ImageTransformType(Enum):
    """Types of transformations that can be applied to images."""

    RESIZE = "resize"
    ROTATE = "rotate"
    CROP = "crop"
    COLOR_CONVERT = "color_convert"
    SKEW_CORRECT = "skew_correct"
    BOUNDARY_ADJUST = "boundary_adjust"
    MARKER_ANNOTATION = "marker_annotation"
    COMPARTMENT_OVERLAY = "compartment_overlay"
    FLIP = "flip"
    PERSPECTIVE = "perspective"
    BRIGHTNESS_CONTRAST = "brightness_contrast"


@dataclass
class ImageTransform:
    """
    Record of a single transformation applied to an image.

    Attributes:
        transform_type: Type of transformation from ImageTransformType enum
        timestamp: When the transformation was applied
        parameters: Dictionary of parameters used for the transformation
        description: Human-readable description of the transformation
    """

    transform_type: ImageTransformType
    timestamp: datetime
    parameters: Dict[str, Any]
    description: str

    def __str__(self) -> str:
        """Return a string representation of the transform."""
        return f"{self.transform_type.value}: {self.description}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert transform to dictionary for serialization."""
        return {
            "type": self.transform_type.value,
            "timestamp": self.timestamp.isoformat(),
            "parameters": self.parameters,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageTransform":
        """Create transform from dictionary."""
        return cls(
            transform_type=ImageTransformType(data["type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            parameters=data["parameters"],
            description=data["description"],
        )


@dataclass
class ImageVersion:
    """
    Represents a version of an image with its transformation history.

    Attributes:
        image: The actual image data as numpy array
        version_name: Unique name for this version
        parent_version: Name of the version this was derived from
        transforms: List of transformations applied to create this version
        scale_factor_from_original: Cumulative scale factor from original image
        metadata: Additional metadata about this version
    """

    image: np.ndarray
    version_name: str
    parent_version: Optional[str] = None
    transforms: List[ImageTransform] = field(default_factory=list)
    scale_factor_from_original: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.transforms is None:
            self.transforms = []
        if self.metadata is None:
            self.metadata = {}

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the image."""
        return self.image.shape

    @property
    def dimensions(self) -> Dict[str, int]:
        """Get image dimensions as a dictionary."""
        h, w = self.image.shape[:2]
        return {
            "width": w,
            "height": h,
            "channels": self.image.shape[2] if len(self.image.shape) > 2 else 1,
        }

    def get_memory_usage(self) -> int:
        """Calculate memory usage of this version in bytes."""
        return self.image.nbytes

    def get_hash(self) -> str:
        """Calculate hash of the image for comparison."""
        return hashlib.md5(self.image.tobytes()).hexdigest()


class VisualizationManager:
    """
    Manages all image versions and transformations throughout the processing pipeline.

    This class provides centralized management for:
    - Tracking all image versions (original, small, corrected, etc.)
    - Recording transformation history
    - Handling coordinate scaling between versions
    - Providing clear visualization outputs
    - Storing intermediate processing steps

    The manager ensures that the image processing flow is transparent and debuggable,
    with full transformation history and automatic coordinate scaling.
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the VisualizationManager.

        Args:
            logger: Optional logger instance. If not provided, creates a default logger.
            config: Optional configuration dictionary. If not provided, uses defaults.
        """
        self.logger = logger or logging.getLogger(__name__)

        # Store external config if provided
        self.external_config = config or {}

        # Main image version storage
        self.versions: Dict[str, ImageVersion] = {}

        # Quick access keys to important versions
        self.original_key = "original"
        self.working_key = "working"  # The current working version
        self.display_key = "display"  # Version for UI display

        # Transformation tracking
        self.active_transforms: List[ImageTransform] = []

        # Scale tracking between versions
        # Key: (from_version, to_version), Value: scale_factor
        self.scale_relationships: Dict[Tuple[str, str], float] = {}

        # Visualization cache for temporary visualizations
        self.viz_cache: Dict[str, np.ndarray] = {}

        # Processing metadata
        self.processing_metadata: Dict[str, Any] = {
            "image_path": None,
            "processing_start": None,
            "markers_detected": {},
            "boundaries": [],
            "compartments": [],
            "last_error": None,
        }

        # Configuration
        self.config = {
            "auto_cleanup": True,  # Automatically cleanup old versions
            "max_versions": 5,  # Maximum number of versions to keep
            "enable_caching": True,  # Enable visualization caching
        }
        # Merge external config values if provided
        if self.external_config:
            # Update internal config with any matching keys from external config
            for key in ["auto_cleanup", "max_versions", "enable_caching"]:
                if key in self.external_config:
                    self.config[key] = self.external_config[key]

        self.logger.info("VisualizationManager initialized")

    def load_image(self, image: np.ndarray, image_path: str) -> None:
        """
        Load the original image and initialize the manager.

        Args:
            image: The original image as numpy array
            image_path: Path to the image file for metadata

        Raises:
            ValueError: If image is None or invalid
        """
        if image is None:
            raise ValueError("Cannot load None image")

        if not isinstance(image, np.ndarray):
            raise ValueError(f"Image must be numpy array, got {type(image)}")

        # Clear any existing data
        self.clear()

        # Store metadata
        self.processing_metadata["image_path"] = image_path
        self.processing_metadata["processing_start"] = datetime.now()

        # Store original image
        self.versions[self.original_key] = ImageVersion(
            image=image.copy(),  # Always copy to prevent external modifications
            version_name=self.original_key,
            metadata={
                "path": image_path,
                "original_shape": image.shape,
                "original_dtype": str(image.dtype),
                "load_time": datetime.now().isoformat(),
            },
        )

        self.logger.info(
            f"Loaded original image: shape={image.shape}, dtype={image.dtype}, path={image_path}"
        )

    def create_working_copy(self, target_pixels: int = 2000000) -> np.ndarray:
        """
        Create a downsampled working copy for faster processing.

        Args:
            target_pixels: Target number of pixels for the working copy

        Returns:
            The working copy image

        Raises:
            RuntimeError: If no original image is loaded
        """
        if self.original_key not in self.versions:
            raise RuntimeError("No original image loaded")

        original = self.versions[self.original_key].image
        h, w = original.shape[:2]
        original_pixels = h * w

        self.logger.debug(
            f"Creating working copy: original size={w}x{h} ({original_pixels} pixels)"
        )

        if original_pixels > target_pixels:
            # Calculate scale to achieve target pixel count
            scale = (target_pixels / original_pixels) ** 0.5
            new_width = int(w * scale)
            new_height = int(h * scale)

            # Ensure minimum dimensions
            new_width = max(new_width, 100)
            new_height = max(new_height, 100)

            # Create resized version
            working_image = cv2.resize(
                original, (new_width, new_height), interpolation=cv2.INTER_AREA
            )

            # Create transformation record
            transform = ImageTransform(
                transform_type=ImageTransformType.RESIZE,
                timestamp=datetime.now(),
                parameters={
                    "scale": scale,
                    "original_size": (w, h),
                    "new_size": (new_width, new_height),
                    "interpolation": "INTER_AREA",
                    "target_pixels": target_pixels,
                },
                description=f"Downsampled from {w}x{h} to {new_width}x{new_height} (scale: {scale:.4f})",
            )

            # Store working version
            self.versions[self.working_key] = ImageVersion(
                image=working_image,
                version_name=self.working_key,
                parent_version=self.original_key,
                transforms=[transform],
                scale_factor_from_original=scale,
            )

            # Store scale relationship (original → working is downscale)
            self._update_scale_relationship(self.original_key, self.working_key, scale)
            # The inverse (working → original) is automatically stored
            self.logger.info(
                f"Created working copy: {new_width}x{new_height}, scale={scale:.4f}"
            )
        else:
            # Image is already small enough, use as-is
            self.versions[self.working_key] = ImageVersion(
                image=original.copy(),
                version_name=self.working_key,
                parent_version=self.original_key,
                scale_factor_from_original=1.0,
                metadata={"no_downsample_needed": True},
            )

            # Store identity scale relationships
            self._update_scale_relationship(self.working_key, self.original_key, 1.0)
            self._update_scale_relationship(self.original_key, self.working_key, 1.0)

            self.logger.info(
                "Original image used as working copy (no downsample needed)"
            )

        # Cleanup old versions if needed
        self._cleanup_old_versions()

        return self.versions[self.working_key].image

    def apply_rotation(
        self,
        version_key: str,
        angle: float,
        rotation_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply rotation to a specific image version.

        Args:
            version_key: The version to rotate
            angle: Rotation angle in degrees (positive = counter-clockwise)
            rotation_matrix: Optional pre-calculated rotation matrix. If None, will be calculated.

        Returns:
            The rotated image

        Raises:
            ValueError: If version not found
        """
        if version_key not in self.versions:
            raise ValueError(f"Version '{version_key}' not found")

        source_version = self.versions[version_key]
        h, w = source_version.image.shape[:2]
        center = (w // 2, h // 2)

        # Calculate rotation matrix if not provided
        if rotation_matrix is None:
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            self.logger.debug(
                f"Calculated rotation matrix for angle={angle}, center={center}"
            )
        else:
            # Verify the rotation matrix is appropriate for this image size
            self._verify_rotation_matrix(rotation_matrix, (w, h), angle)

        # Apply rotation
        rotated = cv2.warpAffine(
            source_version.image,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        # Create transform record
        transform = ImageTransform(
            transform_type=ImageTransformType.ROTATE,
            timestamp=datetime.now(),
            parameters={
                "angle": angle,
                "center": center,
                "image_size": (w, h),
                "rotation_matrix": (
                    rotation_matrix.tolist() if rotation_matrix is not None else None
                ),
            },
            description=f"Rotated by {angle:.2f}° around center {center}",
        )

        # Create new version with suffix
        new_key = f"{version_key}_rotated"

        # If this key already exists, add a number
        if new_key in self.versions:
            i = 2
            while f"{new_key}_{i}" in self.versions:
                i += 1
            new_key = f"{new_key}_{i}"

        self.versions[new_key] = ImageVersion(
            image=rotated,
            version_name=new_key,
            parent_version=version_key,
            transforms=source_version.transforms + [transform],
            scale_factor_from_original=source_version.scale_factor_from_original,
        )

        # Inherit scale relationships
        self._inherit_scale_relationships(new_key, version_key)

        self.logger.info(
            f"Applied rotation to {version_key}: {angle:.2f}° -> {new_key}"
        )

        # Cleanup if needed
        self._cleanup_old_versions()

        return rotated

    def apply_transform_to_original(
        self, transform_from_working: Dict[str, Any]
    ) -> np.ndarray:
        """
        Apply a transformation that was detected on the working image to the original resolution.

        Args:
            transform_from_working: Dictionary describing the transformation
                Required keys: "type" (transformation type)
                Additional keys depend on transformation type

        Returns:
            The transformed original image

        Raises:
            NotImplementedError: If transform type is not supported
            RuntimeError: If required versions are not available
        """
        if self.original_key not in self.versions:
            raise RuntimeError("No original image loaded")

        transform_type = transform_from_working.get("type")

        if transform_type == "rotation":
            angle = transform_from_working["angle"]
            # Apply rotation with recalculated matrix for original dimensions
            return self.apply_rotation(self.original_key, angle)

        else:
            raise NotImplementedError(
                f"Transform type '{transform_type}' not implemented"
            )

    def scale_coordinates(self, coords: Any, from_version: str, to_version: str) -> Any:
        """
        Scale coordinates from one image version to another.

        Handles multiple coordinate formats:
        - Tuple/List of 4 values: (x1, y1, x2, y2) bounding box
        - Numpy array of points: [[x1,y1], [x2,y2], ...]
        - List of tuples/lists: [(x1,y1,x2,y2), ...]
        - Single point: (x, y)

        Args:
            coords: Coordinates to scale
            from_version: Source version key
            to_version: Target version key

        Returns:
            Scaled coordinates in the same format as input

        Raises:
            ValueError: If coordinate format is not supported or versions don't exist
        """
        # Same version, no scaling needed
        if from_version == to_version:
            return coords

        # Validate versions exist
        if from_version not in self.versions:
            raise ValueError(f"Source version '{from_version}' not found")
        if to_version not in self.versions:
            raise ValueError(f"Target version '{to_version}' not found")

        # Get scale factor
        scale = self._get_scale_factor(from_version, to_version)

        # Handle different coordinate types
        if (
            isinstance(coords, (list, tuple))
            and len(coords) == 2
            and all(isinstance(c, (int, float)) for c in coords)
        ):
            # Single point (x, y)
            return (int(coords[0] * scale), int(coords[1] * scale))

        elif isinstance(coords, (list, tuple)) and len(coords) == 4:
            # Bounding box (x1, y1, x2, y2)
            x1, y1, x2, y2 = coords
            return (int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale))

        elif isinstance(coords, np.ndarray):
            # Numpy array of points
            scaled = coords.copy().astype(np.float32)
            scaled *= scale
            return scaled.astype(coords.dtype)

        elif (
            isinstance(coords, list)
            and coords
            and all(isinstance(c, (list, tuple)) and len(c) == 4 for c in coords)
        ):
            # List of bounding boxes
            return [self.scale_coordinates(c, from_version, to_version) for c in coords]

        elif (
            isinstance(coords, list)
            and coords
            and all(isinstance(c, (list, tuple)) and len(c) == 2 for c in coords)
        ):
            # List of points
            return [(int(x * scale), int(y * scale)) for x, y in coords]

        else:
            raise ValueError(f"Unsupported coordinate type: {type(coords)}")

    def create_visualization(
        self, base_version: str, viz_name: str, draw_function: Callable, **kwargs
    ) -> np.ndarray:
        """
        Create and optionally cache a visualization.

        Args:
            base_version: Version to use as base for visualization
            viz_name: Name for this visualization
            draw_function: Function that takes (image, **kwargs) and returns visualized image
            **kwargs: Additional arguments passed to draw_function
                Special kwargs:
                - save_as_version: If True, save as a new version (default: False)
                - cache_key: Custom cache key (default: generated from viz_name and kwargs)

        Returns:
            The visualized image

        Raises:
            ValueError: If base version not found
        """
        if base_version not in self.versions:
            raise ValueError(f"Version '{base_version}' not found")

        # Check cache if enabled
        cache_key = kwargs.pop("cache_key", self._generate_cache_key(viz_name, kwargs))
        if self.config["enable_caching"] and cache_key in self.viz_cache:
            self.logger.debug(f"Returning cached visualization: {viz_name}")
            return self.viz_cache[cache_key].copy()

        # Get base image
        base_image = self.versions[base_version].image.copy()

        # Create visualization
        try:
            viz_image = draw_function(base_image, **kwargs)
        except Exception as e:
            self.logger.error(f"Error in draw function for {viz_name}: {str(e)}")
            self.processing_metadata["last_error"] = str(e)
            raise

        # Cache it if enabled
        if self.config["enable_caching"]:
            self.viz_cache[cache_key] = viz_image.copy()

        # Optionally store as a version
        if kwargs.get("save_as_version", False):
            transform = ImageTransform(
                transform_type=ImageTransformType.MARKER_ANNOTATION,
                timestamp=datetime.now(),
                parameters={
                    "draw_function": draw_function.__name__,
                    "kwargs": str(kwargs),
                },
                description=f"Visualization: {viz_name}",
            )

            self.versions[viz_name] = ImageVersion(
                image=viz_image,
                version_name=viz_name,
                parent_version=base_version,
                transforms=self.versions[base_version].transforms + [transform],
                metadata={"visualization": True, "params": kwargs},
            )

            # Inherit scale relationships
            self._inherit_scale_relationships(viz_name, base_version)

        self.logger.debug(f"Created visualization: {viz_name} based on {base_version}")
        return viz_image

    def get_current_working_image(self) -> Optional[np.ndarray]:
        """
        Get the current working image.

        Returns:
            The working image or None if not available
        """
        if self.working_key in self.versions:
            return self.versions[self.working_key].image
        return None

    def get_original_image(self) -> Optional[np.ndarray]:
        """
        Get the original image.

        Returns:
            The original image or None if not loaded
        """
        if self.original_key in self.versions:
            return self.versions[self.original_key].image
        return None

    def get_version(self, version_key: str) -> Optional[np.ndarray]:
        """
        Get a specific version of the image.

        Args:
            version_key: Version identifier

        Returns:
            The image or None if version not found
        """
        if version_key in self.versions:
            return self.versions[version_key].image
        return None

    def update_working_image(self, new_image: np.ndarray, transform_desc: str) -> None:
        """
        Update the working image with a new version.

        Args:
            new_image: The updated image
            transform_desc: Description of what was changed
        """
        if self.working_key not in self.versions:
            raise RuntimeError("No working image to update")

        transform = ImageTransform(
            transform_type=ImageTransformType.SKEW_CORRECT,
            timestamp=datetime.now(),
            parameters={"description": transform_desc},
            description=transform_desc,
        )

        current_working = self.versions[self.working_key]
        current_working.image = new_image.copy()
        current_working.transforms.append(transform)

        # Clear any cached visualizations that depend on working image
        self._clear_dependent_cache(self.working_key)

        self.logger.info(f"Updated working image: {transform_desc}")

    def get_transformation_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of all transformations applied.

        Returns:
            Dictionary containing:
            - versions: Information about each version
            - relationships: Scale relationships between versions
            - processing_metadata: General processing metadata
            - memory_usage: Memory usage statistics
        """
        summary = {
            "versions": {},
            "relationships": {
                f"{k[0]}->{k[1]}": v for k, v in self.scale_relationships.items()
            },
            "processing_metadata": self.processing_metadata.copy(),
            "memory_usage": {"total_bytes": 0, "by_version": {}},
        }

        for key, version in self.versions.items():
            memory_usage = version.get_memory_usage()
            summary["memory_usage"]["total_bytes"] += memory_usage
            summary["memory_usage"]["by_version"][key] = memory_usage

            summary["versions"][key] = {
                "shape": version.shape,
                "dtype": str(version.image.dtype),
                "parent": version.parent_version,
                "transforms": [t.to_dict() for t in version.transforms],
                "scale_from_original": version.scale_factor_from_original,
                "metadata": version.metadata,
                "hash": version.get_hash(),
            }

        return summary

    def export_transformation_history(self, filepath: str) -> None:
        """
        Export the complete transformation history to a JSON file.

        Args:
            filepath: Path to save the JSON file
        """
        history = self.get_transformation_summary()

        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj

        history = convert_types(history)

        with open(filepath, "w") as f:
            json.dump(history, f, indent=2)

        self.logger.info(f"Exported transformation history to {filepath}")

    def clear(self) -> None:
        """Clear all stored data and reset the manager."""
        self.versions.clear()
        self.scale_relationships.clear()
        self.viz_cache.clear()
        self.active_transforms.clear()
        self.processing_metadata = {
            "image_path": None,
            "processing_start": None,
            "markers_detected": {},
            "boundaries": [],
            "compartments": [],
            "last_error": None,
        }

        self.logger.debug("VisualizationManager cleared")

    def log_current_state(self) -> None:
        """Log the current state for debugging purposes."""
        self.logger.debug("=== VisualizationManager State ===")
        self.logger.debug(f"Versions: {list(self.versions.keys())}")

        for key, version in self.versions.items():
            self.logger.debug(
                f"  {key}: shape={version.shape}, "
                f"transforms={len(version.transforms)}, "
                f"parent={version.parent_version}"
            )

        self.logger.debug(f"Scale relationships: {len(self.scale_relationships)}")
        for (from_v, to_v), scale in self.scale_relationships.items():
            self.logger.debug(f"  {from_v} -> {to_v}: {scale:.4f}")

        self.logger.debug(f"Cached visualizations: {list(self.viz_cache.keys())}")
        self.logger.debug(
            f"Total memory usage: {self._get_total_memory_usage() / 1024 / 1024:.2f} MB"
        )
        self.logger.debug("==================================")

    # ==================== ArUco Geometric Methods ====================

    def correct_image_skew(
        self, markers: Dict[int, np.ndarray], marker_config=None, version_key=None
    ) -> Dict[str, Any]:
        """
        Correct image orientation and skew based on detected markers.

        Args:
            markers: Dictionary of detected markers {id: corners}
            marker_config: Dict with 'corner_ids' and 'compartment_ids' lists
            version_key: Version to correct (defaults to working version)

        Returns:
            Dictionary containing:
                - 'image': Corrected image
                - 'rotation_matrix': Applied transformation matrix
                - 'rotation_angle': Total rotation in degrees
                - 'version_key': Key of the new corrected version
                - 'needs_redetection': Whether markers need to be re-detected
        """
        if version_key is None:
            version_key = self.working_key

        if version_key not in self.versions:
            raise ValueError(f"Version '{version_key}' not found")

        image = self.versions[version_key].image

        # Get marker ID ranges
        if marker_config is None:
            marker_config = {
                "corner_ids": [0, 1, 2, 3],
                "compartment_ids": list(range(4, 24)),
            }

        corner_ids = marker_config.get("corner_ids", [0, 1, 2, 3])
        compartment_ids = marker_config.get("compartment_ids", list(range(4, 24)))

        # Step 1: Fix major orientation issues
        self.logger.info(
            f"Checking major orientation issues with {len(markers)} detected markers"
        )
        oriented_image, orientation_matrix, orientation_angle = (
            self._fix_major_orientation(image, markers, corner_ids, compartment_ids)
        )

        # Track if orientation was corrected
        orientation_corrected = oriented_image is not image

        # If major orientation was corrected, we need re-detection
        if orientation_corrected:
            self.logger.info(
                f"Image orientation was corrected by {orientation_angle:.2f} degrees"
            )

            # Create new version for oriented image
            transform = ImageTransform(
                transform_type=ImageTransformType.ROTATE,
                timestamp=datetime.now(),
                parameters={"angle": orientation_angle, "type": "major_orientation"},
                description=f"Major orientation correction: {orientation_angle:.2f}°",
            )

            oriented_key = f"{version_key}_oriented"
            self.versions[oriented_key] = ImageVersion(
                image=oriented_image,
                version_name=oriented_key,
                parent_version=version_key,
                transforms=self.versions[version_key].transforms + [transform],
                scale_factor_from_original=self.versions[
                    version_key
                ].scale_factor_from_original,
            )

            self._inherit_scale_relationships(oriented_key, version_key)

            return {
                "image": oriented_image,
                "rotation_matrix": orientation_matrix,
                "rotation_angle": orientation_angle,
                "version_key": oriented_key,
                "needs_redetection": True,
            }

        # Step 2: Calculate fine skew angle
        skew_angle = self._calculate_skew_angle(markers, compartment_ids)

        if abs(skew_angle) > 0.01:
            # Apply skew correction
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            skew_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)

            corrected_image = cv2.warpAffine(
                image,
                skew_matrix,
                (w, h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )

            # Create new version
            transform = ImageTransform(
                transform_type=ImageTransformType.SKEW_CORRECT,
                timestamp=datetime.now(),
                parameters={"skew_angle": skew_angle, "total_angle": skew_angle},
                description=f"Fine skew correction: {skew_angle:.2f}°",
            )

            corrected_key = f"{version_key}_skew_corrected"
            self.versions[corrected_key] = ImageVersion(
                image=corrected_image,
                version_name=corrected_key,
                parent_version=version_key,
                transforms=self.versions[version_key].transforms + [transform],
                scale_factor_from_original=self.versions[
                    version_key
                ].scale_factor_from_original,
            )

            self._inherit_scale_relationships(corrected_key, version_key)

            return {
                "image": corrected_image,
                "rotation_matrix": skew_matrix,
                "rotation_angle": skew_angle,
                "version_key": corrected_key,
                "needs_redetection": abs(skew_angle) > 1.0,
            }

        # No correction needed
        return {
            "image": image,
            "rotation_matrix": np.eye(2, 3, dtype=np.float32),
            "rotation_angle": 0.0,
            "version_key": version_key,
            "needs_redetection": False,
        }

    def correct_marker_geometry(
        self,
        markers: Dict[int, np.ndarray],
        tolerance_pixels: float = 3,
        preserve_orientation: bool = False,
    ) -> Dict[int, np.ndarray]:
        """
        Correct distorted markers to perfect squares.
        This is called AFTER skew correction, so markers are already properly aligned.

        Args:
            markers: Dictionary of detected markers {id: corners}
            tolerance_pixels: Maximum edge length variance allowed
            preserve_orientation: Whether to preserve each marker's rotation

        Returns:
            Dictionary of corrected markers with same IDs
        """
        corrected_markers = {}

        for marker_id, corners in markers.items():
            # Analyze marker geometry
            edge_data = self._analyze_marker_geometry(corners, tolerance_pixels)

            if not edge_data["needs_correction"]:
                # Marker is already good
                corrected_markers[marker_id] = corners
                continue

            if not edge_data["is_correctable"]:
                # Too distorted to correct
                self.logger.warning(f"Marker {marker_id} too distorted to correct")
                corrected_markers[marker_id] = corners
                continue

            if preserve_orientation:
                # Preserve the marker's current orientation
                corrected_corners = self._create_oriented_square(
                    edge_data["center"],
                    edge_data["median_edge_length"],
                    edge_data["orientation_angle"],
                )
            else:
                # Create axis-aligned square
                corrected_corners = self._create_axis_aligned_square(
                    edge_data["center"], edge_data["median_edge_length"]
                )

            corrected_markers[marker_id] = corrected_corners

            self.logger.debug(
                f"Corrected marker {marker_id}: CV {edge_data['edge_cv']:.2%} -> 0%, "
                f"orientation: {edge_data['orientation_angle']:.1f}°"
            )

        return corrected_markers

    def estimate_scale_from_markers(
        self, markers: Dict[int, np.ndarray], marker_config: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Estimate image scale from detected markers with known physical sizes.

        Args:
            markers: Dictionary of detected markers {id: corners}
            marker_config: Configuration dictionary containing:
                - 'corner_marker_size_cm': Physical size of corner markers
                - 'compartment_marker_size_cm': Physical size of compartment markers
                - 'corner_ids': List of corner marker IDs
                - 'compartment_ids': List of compartment marker IDs
                - 'metadata_ids': List of metadata marker IDs
                - 'use_corner_markers': Whether to include corner markers

        Returns:
            Dictionary containing scale estimation results
        """
        if not markers:
            self.logger.warning("No markers provided for scale estimation")
            return {
                "scale_px_per_cm": None,
                "image_width_cm": None,
                "marker_measurements": [],
                "confidence": 0.0,
            }
        # Extract configuration values
        corner_size_cm = marker_config.get("corner_marker_size_cm", 1.0)
        compartment_size_cm = marker_config.get("compartment_marker_size_cm", 2.0)
        corner_ids = marker_config.get("corner_ids", [0, 1, 2, 3])
        compartment_ids = marker_config.get("compartment_ids", list(range(4, 24)))
        metadata_ids = marker_config.get("metadata_ids", [24])
        use_corner_markers = marker_config.get("use_corner_markers", True)

        # Collect scale measurements
        scale_measurements = []
        marker_measurements = []

        for marker_id, corners in markers.items():
            # Skip metadata markers
            if marker_id in metadata_ids:
                continue

            # Determine marker type and size
            if marker_id in corner_ids:
                if not use_corner_markers:
                    continue
                physical_size_cm = corner_size_cm
                marker_type = "corner"
            elif marker_id in compartment_ids:
                physical_size_cm = compartment_size_cm
                marker_type = "compartment"
            else:
                continue

            # Measure this marker
            measurements = self._measure_marker_for_scale(
                corners, physical_size_cm, marker_id, marker_type
            )

            if measurements["valid"]:
                scale_measurements.extend(measurements["scales"])
                marker_measurements.append(measurements)

        # Calculate robust scale estimate
        if not scale_measurements:
            self.logger.warning("No valid scale measurements obtained")
            return {
                "scale_px_per_cm": None,
                "image_width_cm": None,
                "marker_measurements": marker_measurements,
                "confidence": 0.0,
            }

        # Remove outliers using IQR method
        scale_array = np.array(scale_measurements)
        q1, q3 = np.percentile(scale_array, [25, 75])
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        filtered_scales = scale_array[
            (scale_array >= lower_bound) & (scale_array <= upper_bound)
        ]

        if len(filtered_scales) == 0:
            filtered_scales = scale_array
            self.logger.warning(
                "All scale measurements were outliers, using unfiltered data"
            )

        # Calculate final scale
        final_scale_px_per_cm = float(np.median(filtered_scales))

        # Calculate confidence
        if len(filtered_scales) > 1:
            std_dev = np.std(filtered_scales)
            mean_val = np.mean(filtered_scales)
            cv = std_dev / mean_val if mean_val > 0 else 1.0
            confidence = max(0.0, min(1.0, 1.0 - (cv - 0.02) / 0.08))
        else:
            confidence = 0.5

        # Calculate image width if we have a working version
        image_width_cm = None
        if self.working_key in self.versions:
            image_width_px = self.versions[self.working_key].shape[1]
            image_width_cm = image_width_px / final_scale_px_per_cm

        # Store in metadata
        self.processing_metadata["scale_data"] = {
            "scale_px_per_cm": final_scale_px_per_cm,
            "confidence": confidence,
        }

        self.logger.info(
            f"Scale estimation: {final_scale_px_per_cm:.2f} px/cm (confidence: {confidence:.2%})"
        )

        return {
            "scale_px_per_cm": final_scale_px_per_cm,
            "image_width_cm": image_width_cm,
            "marker_measurements": marker_measurements,
            "confidence": confidence,
            "outliers_removed": len(scale_measurements) - len(filtered_scales),
            "total_measurements": len(scale_measurements),
        }

    # ==================== ArUco Gometric Private Helper Methods ====================

    def _fix_major_orientation(
        self,
        image: np.ndarray,
        markers: Dict[int, np.ndarray],
        corner_ids: List[int],
        compartment_ids: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Fix major orientation issues (portrait/landscape, 180° rotations).

        Returns: (corrected_image, transformation_matrix, rotation_angle)
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # Get detected markers by type
        corner_markers = [(mid, markers[mid]) for mid in corner_ids if mid in markers]
        comp_markers = [
            (mid, markers[mid])
            for mid in compartment_ids
            if mid in markers and mid != 24
        ]  # Exclude metadata

        # Need minimum markers
        if len(corner_markers) < 1 or len(comp_markers) < 2:
            self.logger.warning("Insufficient markers for orientation detection")
            return image, np.eye(2, 3, dtype=np.float64), 0.0

        # Analyze marker positions
        is_landscape = w > h

        # Calculate average positions
        corner_centers = [np.mean(corners, axis=0) for _, corners in corner_markers]
        comp_centers = [
            (mid, np.mean(corners, axis=0)) for mid, corners in comp_markers
        ]

        corner_avg = np.mean(corner_centers, axis=0)
        comp_avg = np.mean([c for _, c in comp_centers], axis=0)

        # Check marker ID ordering vs X position
        comp_centers_sorted = sorted(comp_centers, key=lambda x: x[0])  # By ID
        comp_centers_by_x = sorted(comp_centers, key=lambda x: x[1][0])  # By X

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

        # Create transformation matrix
        if rotation_angle != 0:
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        else:
            rotation_matrix = np.eye(2, 3, dtype=np.float64)

        return rotated_image, rotation_matrix, rotation_angle

    def _calculate_skew_angle(
        self, markers: Dict[int, np.ndarray], compartment_ids: List[int]
    ) -> float:
        """
        Calculate fine skew angle by fitting a line through compartment markers.

        Returns: Skew angle in degrees
        """
        # Get compartment marker centers (excluding metadata)
        centers = []
        for marker_id in compartment_ids:
            if marker_id in markers and marker_id != 24:
                center = np.mean(markers[marker_id], axis=0)
                centers.append((marker_id, center))

        if len(centers) < 4:
            self.logger.warning(
                f"Only {len(centers)} compartment markers for skew estimation"
            )
            return 0.0

        # Sort by marker ID for consistent ordering
        centers.sort(key=lambda x: x[0])
        points = np.array([c[1] for c in centers])

        # Fit line through centers
        try:
            x_vals = points[:, 0]
            y_vals = points[:, 1]
            coeffs = np.polyfit(x_vals, y_vals, 1)
            slope = coeffs[0]

            # Convert to angle
            angle_rad = np.arctan(slope)
            angle_deg = np.degrees(angle_rad)

            # Normalize to [-90, 90]
            if angle_deg > 90:
                angle_deg -= 180
            elif angle_deg < -90:
                angle_deg += 180

            return angle_deg

        except Exception as e:
            self.logger.error(f"Error calculating skew angle: {e}")
            return 0.0

    def _analyze_marker_geometry(
        self, corners: np.ndarray, tolerance_pixels: float
    ) -> Dict[str, Any]:
        """
        Analyze marker geometry to determine if correction is needed.

        Returns dictionary with analysis results.
        """
        # Calculate edge lengths
        edges = []
        edge_indices = [(0, 1), (1, 2), (2, 3), (3, 0)]

        for i, j in edge_indices:
            edge_length = np.linalg.norm(corners[i] - corners[j])
            edges.append(edge_length)

        edges = np.array(edges)
        median_edge = np.median(edges)
        edge_std = np.std(edges)
        edge_mean = np.mean(edges)
        edge_cv = edge_std / edge_mean if edge_mean > 0 else 1.0

        # Calculate center and orientation
        center = np.mean(corners, axis=0)

        # Get orientation from first edge
        edge_vector = corners[1] - corners[0]
        orientation_angle = np.degrees(np.arctan2(edge_vector[1], edge_vector[0]))

        # Determine if correction needed
        needs_correction = edge_cv > 0.02  # More than 2% variation

        # Check if correctable (at least 2 valid edges)
        valid_edges = sum(1 for e in edges if abs(e - median_edge) <= tolerance_pixels)
        is_correctable = valid_edges >= 2

        return {
            "edges": edges,
            "median_edge_length": median_edge,
            "edge_cv": edge_cv,
            "center": center,
            "orientation_angle": orientation_angle,
            "needs_correction": needs_correction,
            "is_correctable": is_correctable,
            "valid_edge_count": valid_edges,
        }

    def _create_oriented_square(
        self, center: np.ndarray, size: float, angle: float
    ) -> np.ndarray:
        """
        Create a perfect square with given orientation.

        Args:
            center: Center point of the square
            size: Edge length
            angle: Orientation angle in degrees

        Returns:
            4x2 array of corner points
        """
        half_size = size / 2
        angle_rad = np.radians(angle)

        # Unit vectors for edges
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        x_vec = np.array([cos_a, sin_a])
        y_vec = np.array([-sin_a, cos_a])

        # Build corners
        corners = np.array(
            [
                center - half_size * x_vec - half_size * y_vec,  # Top-left
                center + half_size * x_vec - half_size * y_vec,  # Top-right
                center + half_size * x_vec + half_size * y_vec,  # Bottom-right
                center - half_size * x_vec + half_size * y_vec,  # Bottom-left
            ],
            dtype=np.float32,
        )

        return corners

    def _create_axis_aligned_square(
        self, center: np.ndarray, size: float
    ) -> np.ndarray:
        """
        Create an axis-aligned perfect square.

        Args:
            center: Center point of the square
            size: Edge length

        Returns:
            4x2 array of corner points
        """
        half_size = size / 2

        corners = np.array(
            [
                [center[0] - half_size, center[1] - half_size],  # Top-left
                [center[0] + half_size, center[1] - half_size],  # Top-right
                [center[0] + half_size, center[1] + half_size],  # Bottom-right
                [center[0] - half_size, center[1] + half_size],  # Bottom-left
            ],
            dtype=np.float32,
        )

        return corners

    def _measure_marker_for_scale(
        self,
        corners: np.ndarray,
        physical_size_cm: float,
        marker_id: int,
        marker_type: str,
    ) -> Dict[str, Any]:
        """
        Measure a single marker to estimate scale.

        Returns dictionary with measurement results.
        """
        measurements = {
            "marker_id": marker_id,
            "marker_type": marker_type,
            "physical_size_cm": physical_size_cm,
            "scales": [],
            "edge_lengths": [],
            "diagonal_lengths": [],
            "valid": False,
            "rejection_reason": None,
        }

        if corners.shape[0] != 4:
            measurements["rejection_reason"] = (
                f"Invalid corner count: {corners.shape[0]}"
            )
            return measurements

        # Measure edges
        edge_indices = [(0, 1), (1, 2), (2, 3), (3, 0)]
        edge_lengths = []

        for i, j in edge_indices:
            edge_length = np.linalg.norm(corners[i] - corners[j])
            edge_lengths.append(edge_length)
            scale = edge_length / physical_size_cm
            measurements["scales"].append(scale)

        measurements["edge_lengths"] = edge_lengths

        # Measure diagonals
        diagonal_indices = [(0, 2), (1, 3)]
        diagonal_physical = physical_size_cm * np.sqrt(2)
        diagonal_lengths = []

        for i, j in diagonal_indices:
            diagonal_length = np.linalg.norm(corners[i] - corners[j])
            diagonal_lengths.append(diagonal_length)
            scale = diagonal_length / diagonal_physical
            measurements["scales"].append(scale)

        measurements["diagonal_lengths"] = diagonal_lengths

        # Validate marker quality
        edge_array = np.array(edge_lengths)
        edge_cv = (
            np.std(edge_array) / np.mean(edge_array) if np.mean(edge_array) > 0 else 1.0
        )

        if edge_cv > 0.02:  # More than 2% variation
            measurements["rejection_reason"] = f"High edge variance (CV={edge_cv:.2%})"
            measurements["valid"] = False
        else:
            measurements["valid"] = True
            measurements["edge_cv"] = edge_cv

        # Check if roughly square
        if measurements["valid"] and len(diagonal_lengths) == 2:
            if min(diagonal_lengths) > 0:
                diagonal_ratio = max(diagonal_lengths) / min(diagonal_lengths)
                if diagonal_ratio > 1.02:
                    measurements["rejection_reason"] = (
                        f"Non-square (diagonal ratio={diagonal_ratio:.2f})"
                    )
                    measurements["valid"] = False

        return measurements

    # ==================== Other Private Helper Methods ====================
    def _update_scale_relationship(
        self, from_version: str, to_version: str, scale: float
    ) -> None:
        """
        Update scale relationship between two versions.

        Args:
            from_version: Source version name
            to_version: Target version name
            scale: Scale factor to go FROM source TO target
                - scale < 1.0 means downscaling (target is smaller)
                - scale > 1.0 means upscaling (target is larger)
                - scale = 1.0 means same size
        """
        # Store forward relationship
        self.logger.debug(
            f"Scale relationship: {from_version} → {to_version} = {scale:.6f} "
            + f"({'downscale' if scale < 1 else 'upscale' if scale > 1 else 'same'})"
        )
        self.scale_relationships[(from_version, to_version)] = scale

        # Automatically store inverse relationship
        inverse_scale = 1.0 / scale if scale != 0 else 1.0
        self.logger.debug(
            f"Inverse relationship: {to_version} → {from_version} = {inverse_scale:.6f}"
        )
        self.scale_relationships[(to_version, from_version)] = inverse_scale

    def _get_scale_factor(self, from_version: str, to_version: str) -> float:
        """
        Get scale factor between two versions.

        Handles transitive relationships if direct relationship not stored.
        """
        # Direct relationship exists
        if (from_version, to_version) in self.scale_relationships:
            scale = self.scale_relationships[(from_version, to_version)]
            self.logger.debug(
                f"Retrieved direct scale: {from_version} → {to_version} = {scale:.6f}"
            )
            return scale

        # Try to find path through original
        if (from_version, self.original_key) in self.scale_relationships and (
            self.original_key,
            to_version,
        ) in self.scale_relationships:
            scale1 = self.scale_relationships[(from_version, self.original_key)]
            scale2 = self.scale_relationships[(self.original_key, to_version)]
            self.logger.debug(
                f"Transitive scale ({from_version} → {self.original_key} → {to_version}): "
                f"{scale1:.6f} * {scale2:.6f} = {scale1 * scale2:.6f}"
            )
            return scale1 * scale2

        # Fallback: calculate from dimensions
        from_v = self.versions[from_version]
        to_v = self.versions[to_version]

        self.logger.debug(f"Calculating fallback scale from image shapes:")
        self.logger.debug(f"    {from_version}: shape = {from_v.shape}")
        self.logger.debug(f"    {to_version}: shape = {to_v.shape}")

        scale_x = to_v.shape[1] / from_v.shape[1]
        scale_y = to_v.shape[0] / from_v.shape[0]

        # Log warning if scales differ significantly
        if abs(scale_x - scale_y) > 0.01:
            self.logger.warning(
                f"Non-uniform scaling detected between {from_version} and {to_version}: "
                f"x={scale_x:.4f}, y={scale_y:.4f}"
            )

        scale = (scale_x + scale_y) / 2
        self.logger.debug(
            f"Calculated average scale: x={scale_x:.6f}, y={scale_y:.6f}, avg={scale:.6f}"
        )

        # Cache for both directions
        self._update_scale_relationship(from_version, to_version, scale)
        self._update_scale_relationship(to_version, from_version, 1.0 / scale)

        return scale

    def _inherit_scale_relationships(
        self, new_version: str, parent_version: str
    ) -> None:
        """Make new version inherit scale relationships from parent."""
        # New version has same scale to all other versions as parent
        for (from_v, to_v), scale in list(self.scale_relationships.items()):
            if from_v == parent_version and to_v != new_version:
                self._update_scale_relationship(new_version, to_v, scale)
            elif to_v == parent_version and from_v != new_version:
                self._update_scale_relationship(from_v, new_version, scale)

        # Identity relationship with parent
        self._update_scale_relationship(new_version, parent_version, 1.0)
        self._update_scale_relationship(parent_version, new_version, 1.0)

    def _verify_rotation_matrix(
        self, matrix: np.ndarray, image_size: Tuple[int, int], expected_angle: float
    ) -> None:
        """Verify that a rotation matrix is appropriate for the given image size."""
        h, w = image_size
        expected_center = (w // 2, h // 2)

        # Extract rotation angle from matrix
        # For a 2D rotation matrix, cos(theta) is at [0,0] and sin(theta) is at [1,0]
        cos_theta = matrix[0, 0]
        sin_theta = matrix[1, 0]
        extracted_angle = np.degrees(np.arctan2(sin_theta, cos_theta))

        # Check if angle matches
        angle_diff = abs(extracted_angle - expected_angle)
        if angle_diff > 1.0:  # 1 degree tolerance
            self.logger.warning(
                f"Rotation matrix angle ({extracted_angle:.2f}°) differs from expected ({expected_angle:.2f}°)"
            )

        # Extract center from matrix (approximate check)
        # This is complex, so just log a warning if matrix seems off
        # A proper check would solve for the center point
        tx = matrix[0, 2]
        ty = matrix[1, 2]

        # Expected translation for rotation around center
        expected_tx = (
            expected_center[0]
            - cos_theta * expected_center[0]
            + sin_theta * expected_center[1]
        )
        expected_ty = (
            expected_center[1]
            - sin_theta * expected_center[0]
            - cos_theta * expected_center[1]
        )

        if abs(tx - expected_tx) > 10 or abs(ty - expected_ty) > 10:
            self.logger.warning(
                f"Rotation matrix may not be centered correctly for image size {w}x{h}. "
                f"Expected center: {expected_center}, Matrix translation: ({tx:.1f}, {ty:.1f})"
            )

    def _cleanup_old_versions(self) -> None:
        """Remove old versions if exceeding maximum."""
        if not self.config["auto_cleanup"]:
            return

        if len(self.versions) <= self.config["max_versions"]:
            return

        # Never remove original or working versions
        protected_keys = {self.original_key, self.working_key}

        # Find versions to remove (oldest first, excluding protected)
        removable_versions = []
        for key, version in self.versions.items():
            if key not in protected_keys:
                # Get creation time from first transform or metadata
                if version.transforms:
                    timestamp = version.transforms[0].timestamp
                else:
                    timestamp = datetime.now()  # Fallback
                removable_versions.append((timestamp, key))

        # Sort by timestamp (oldest first)
        removable_versions.sort()

        # Remove oldest versions
        num_to_remove = len(self.versions) - self.config["max_versions"]
        for _, key in removable_versions[:num_to_remove]:
            self.logger.debug(f"Removing old version: {key}")
            del self.versions[key]

            # Remove associated scale relationships
            self.scale_relationships = {
                k: v
                for k, v in self.scale_relationships.items()
                if k[0] != key and k[1] != key
            }

            # Clear from cache
            self._clear_dependent_cache(key)

    def _clear_dependent_cache(self, version_key: str) -> None:
        """Clear cached visualizations that depend on a specific version."""
        # Simple approach: clear all cache when a version changes
        # More sophisticated approach would track dependencies
        self.viz_cache.clear()
        self.logger.debug(f"Cleared visualization cache due to change in {version_key}")

    def _generate_cache_key(self, viz_name: str, kwargs: Dict[str, Any]) -> str:
        """Generate a cache key for a visualization."""
        # Create a string representation of kwargs
        kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)
        # Create hash of the combination
        return f"{viz_name}_{hashlib.md5(kwargs_str.encode()).hexdigest()[:8]}"

    def _get_total_memory_usage(self) -> int:
        """Calculate total memory usage in bytes."""
        total = sum(v.get_memory_usage() for v in self.versions.values())
        total += sum(v.nbytes for v in self.viz_cache.values())
        return total

    def _apply_crop(
        self, version_key: str, crop_region: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Apply crop to a version."""
        if version_key not in self.versions:
            raise ValueError(f"Version '{version_key}' not found")

        x1, y1, x2, y2 = crop_region
        image = self.versions[version_key].image

        # Ensure crop region is within bounds
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))

        cropped = image[y1:y2, x1:x2].copy()

        # Create transform record
        transform = ImageTransform(
            transform_type=ImageTransformType.CROP,
            timestamp=datetime.now(),
            parameters={
                "crop_region": (x1, y1, x2, y2),
                "original_size": image.shape[:2],
                "new_size": cropped.shape[:2],
            },
            description=f"Cropped to region ({x1},{y1})-({x2},{y2})",
        )

        # Create new version
        new_key = f"{version_key}_cropped"
        self.versions[new_key] = ImageVersion(
            image=cropped,
            version_name=new_key,
            parent_version=version_key,
            transforms=self.versions[version_key].transforms + [transform],
            scale_factor_from_original=self.versions[
                version_key
            ].scale_factor_from_original,
        )

        return cropped

    def _apply_perspective(
        self, version_key: str, src_points: np.ndarray, dst_points: np.ndarray
    ) -> np.ndarray:
        """Apply perspective transformation."""
        if version_key not in self.versions:
            raise ValueError(f"Version '{version_key}' not found")

        image = self.versions[version_key].image
        h, w = image.shape[:2]

        # Calculate perspective matrix
        matrix = cv2.getPerspectiveTransform(
            src_points.astype(np.float32), dst_points.astype(np.float32)
        )

        # Apply transformation
        transformed = cv2.warpPerspective(image, matrix, (w, h))

        # Create transform record
        transform = ImageTransform(
            transform_type=ImageTransformType.PERSPECTIVE,
            timestamp=datetime.now(),
            parameters={
                "src_points": src_points.tolist(),
                "dst_points": dst_points.tolist(),
                "matrix": matrix.tolist(),
            },
            description="Applied perspective transformation",
        )

        # Create new version
        new_key = f"{version_key}_perspective"
        self.versions[new_key] = ImageVersion(
            image=transformed,
            version_name=new_key,
            parent_version=version_key,
            transforms=self.versions[version_key].transforms + [transform],
            scale_factor_from_original=self.versions[
                version_key
            ].scale_factor_from_original,
        )

        return transformed

    # ==================== Display Helper Methods ====================

    def get_display_image(
        self, version_key: str, max_width: int, max_height: int
    ) -> np.ndarray:
        """
        Get an image sized for display without affecting the stored version.

        Args:
            version_key: Version to display
            max_width: Maximum display width
            max_height: Maximum display height

        Returns:
            Display-ready image (resized if needed)
        """
        if version_key not in self.versions:
            raise ValueError(f"Version '{version_key}' not found")

        image = self.versions[version_key].image
        h, w = image.shape[:2]

        # Calculate scale to fit in display area
        scale_w = max_width / w
        scale_h = max_height / h
        scale = min(scale_w, scale_h)

        if scale < 1.0:
            # Need to resize
            display_w = int(w * scale)
            display_h = int(h * scale)
            return cv2.resize(
                image, (display_w, display_h), interpolation=cv2.INTER_AREA
            )

        # Return copy to prevent modifications
        return image.copy()

    def display_to_image_coords(
        self,
        display_coords: Tuple[int, int],
        version_key: str,
        display_size: Tuple[int, int],
    ) -> Tuple[int, int]:
        """
        Convert coordinates from display space to image space.

        Args:
            display_coords: (x, y) in display space
            version_key: Version being displayed
            display_size: (width, height) of display area

        Returns:
            (x, y) in image space
        """
        if version_key not in self.versions:
            raise ValueError(f"Version '{version_key}' not found")

        image = self.versions[version_key].image
        img_h, img_w = image.shape[:2]
        display_w, display_h = display_size

        # Calculate the scale factor used for display
        scale_w = display_w / img_w
        scale_h = display_h / img_h
        scale = min(scale_w, scale_h)

        # Calculate actual display dimensions
        actual_display_w = int(img_w * scale)
        actual_display_h = int(img_h * scale)

        # Calculate offsets (for centering)
        offset_x = (display_w - actual_display_w) // 2
        offset_y = (display_h - actual_display_h) // 2

        # Convert coordinates
        x_display, y_display = display_coords
        x_image = int((x_display - offset_x) / scale)
        y_image = int((y_display - offset_y) / scale)

        # Clamp to image bounds
        x_image = max(0, min(x_image, img_w - 1))
        y_image = max(0, min(y_image, img_h - 1))

        return (x_image, y_image)


class CoordinateMapper:
    """
    Handles bidirectional coordinate mapping between image and canvas space.
    Maintains consistency even when canvas is resized.
    """

    def __init__(self):
        # Image dimensions
        self.image_width = 0
        self.image_height = 0

        # Canvas dimensions
        self.canvas_width = 0
        self.canvas_height = 0

        # Current display parameters
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        # Fit mode options:
        # "fit" - Scale to fit entirely within canvas, maintain aspect ratio (may have letterboxing)
        # "fill" - Scale to fill canvas completely, maintain aspect ratio (may crop)
        # "stretch" - Stretch to fill canvas exactly (DISTORTS aspect ratio)
        # "none" - No scaling, just center
        self.fit_mode = "fit"  # Default to fit mode

    def update_image_size(self, width: int, height: int):
        """Update the original image dimensions."""
        self.image_width = width
        self.image_height = height
        self._recalculate_mapping()

    def update_canvas_size(self, width: int, height: int):
        """Update canvas dimensions and recalculate mapping."""
        self.canvas_width = width
        self.canvas_height = height
        self._recalculate_mapping()

    def _recalculate_mapping(self):
        """Recalculate scale and offsets based on current dimensions."""
        if self.image_width == 0 or self.image_height == 0:
            return

        if self.canvas_width == 0 or self.canvas_height == 0:
            return

        if self.fit_mode == "fit":
            # Scale to fit entirely within canvas while maintaining aspect ratio
            # This is what you want - fills canvas as much as possible without distortion
            scale_x = self.canvas_width / self.image_width
            scale_y = self.canvas_height / self.image_height
            self.scale = min(scale_x, scale_y)  # Use smaller scale to fit entirely

            # Calculate actual display dimensions
            display_width = self.image_width * self.scale
            display_height = self.image_height * self.scale

            # Center the image in the canvas
            self.offset_x = (self.canvas_width - display_width) / 2
            self.offset_y = (self.canvas_height - display_height) / 2

        elif self.fit_mode == "fill":
            # Scale to fill entire canvas while maintaining aspect ratio (may crop)
            scale_x = self.canvas_width / self.image_width
            scale_y = self.canvas_height / self.image_height
            self.scale = max(scale_x, scale_y)  # Use larger scale to fill entirely

            # Calculate actual display dimensions
            display_width = self.image_width * self.scale
            display_height = self.image_height * self.scale

            # Center the image (parts may be outside canvas)
            self.offset_x = (self.canvas_width - display_width) / 2
            self.offset_y = (self.canvas_height - display_height) / 2

        elif self.fit_mode == "stretch":
            # WARNING: This DISTORTS the aspect ratio
            # Stretch to fill canvas exactly
            self.scale_x = self.canvas_width / self.image_width
            self.scale_y = self.canvas_height / self.image_height
            self.scale = (self.scale_x + self.scale_y) / 2  # Average for simple access
            self.offset_x = 0
            self.offset_y = 0

        elif self.fit_mode == "none":
            # No scaling, just center at original size
            self.scale = 1.0
            self.offset_x = max(0, (self.canvas_width - self.image_width) / 2)
            self.offset_y = max(0, (self.canvas_height - self.image_height) / 2)

    def canvas_to_image(
        self, canvas_x: float, canvas_y: float
    ) -> Optional[Tuple[float, float]]:
        """
        Convert canvas coordinates to image coordinates.

        Args:
            canvas_x: X coordinate in canvas space
            canvas_y: Y coordinate in canvas space

        Returns:
            (image_x, image_y) or None if outside image bounds
        """
        # Remove offset
        x = canvas_x - self.offset_x
        y = canvas_y - self.offset_y

        # Remove scale
        if self.fit_mode == "stretch" and hasattr(self, "scale_x"):
            # Handle non-uniform scaling
            image_x = x / self.scale_x
            image_y = y / self.scale_y
        else:
            # Uniform scaling
            image_x = x / self.scale
            image_y = y / self.scale

        # Check bounds
        if 0 <= image_x < self.image_width and 0 <= image_y < self.image_height:
            return (image_x, image_y)
        else:
            return None

    def image_to_canvas(self, image_x: float, image_y: float) -> Tuple[float, float]:
        """
        Convert image coordinates to canvas coordinates.

        Args:
            image_x: X coordinate in image space
            image_y: Y coordinate in image space

        Returns:
            (canvas_x, canvas_y) in canvas space
        """
        if self.fit_mode == "stretch" and hasattr(self, "scale_x"):
            # Handle non-uniform scaling
            canvas_x = image_x * self.scale_x + self.offset_x
            canvas_y = image_y * self.scale_y + self.offset_y
        else:
            # Uniform scaling
            canvas_x = image_x * self.scale + self.offset_x
            canvas_y = image_y * self.scale + self.offset_y

        return (canvas_x, canvas_y)

    def image_rect_to_canvas(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> Tuple[float, float, float, float]:
        """Convert image rectangle coordinates to canvas coordinates."""
        c_x1, c_y1 = self.image_to_canvas(x1, y1)
        c_x2, c_y2 = self.image_to_canvas(x2, y2)
        return (c_x1, c_y1, c_x2, c_y2)

    def canvas_rect_to_image(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> Optional[Tuple[float, float, float, float]]:
        """Convert canvas rectangle coordinates to image coordinates."""
        i_p1 = self.canvas_to_image(x1, y1)
        i_p2 = self.canvas_to_image(x2, y2)

        if i_p1 is None or i_p2 is None:
            return None

        return (i_p1[0], i_p1[1], i_p2[0], i_p2[1])

    def get_visible_image_region(self) -> Tuple[int, int, int, int]:
        """Get the region of the image currently visible in the canvas."""
        # Get canvas corners in image space
        tl = self.canvas_to_image(0, 0) or (0, 0)
        br = self.canvas_to_image(self.canvas_width, self.canvas_height) or (
            self.image_width,
            self.image_height,
        )

        # Clamp to image bounds
        x1 = max(0, int(tl[0]))
        y1 = max(0, int(tl[1]))
        x2 = min(self.image_width, int(br[0]))
        y2 = min(self.image_height, int(br[1]))

        return (x1, y1, x2, y2)

    def get_display_info(self) -> Dict[str, Any]:
        """Get current display information for debugging."""
        return {
            "image_size": (self.image_width, self.image_height),
            "canvas_size": (self.canvas_width, self.canvas_height),
            "scale": self.scale,
            "offset": (self.offset_x, self.offset_y),
            "fit_mode": self.fit_mode,
            "display_size": (
                int(self.image_width * self.scale),
                int(self.image_height * self.scale),
            ),
        }
