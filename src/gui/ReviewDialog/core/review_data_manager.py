# src\gui\ReviewDialog\core\review_data_manager.py

"""

Review Data Manager - Handles compartment image data operations.

This module provides data-only operations with NO UI dependencies.
Manages image collections, indexing, and querying.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime


logger = logging.getLogger(__name__)


@dataclass
class CompartmentImage:
    """Represents a compartment image with metadata"""

    filename: str
    hole_id: str
    depth_from: float
    depth_to: float
    image_path: str
    classification: str = "Unassigned"
    moisture_status: Optional[str] = None  # "Wet" or "Dry"
    comments: str = ""
    classified_by: str = ""
    classified_date: str = ""
    active_filters: str = ""  # Added from logging_review_dialog
    csv_data: Dict = field(default_factory=dict)
    in_csv: bool = False  # Added from logging_review_dialog
    compartment_uid: Optional[str] = None

    # Peer review fields (from logging_review_dialog)
    other_reviews: Optional[List[Dict]] = None
    other_classification: Optional[str] = None
    other_reviewers: Optional[List[str]] = None

    # Review state tracking
    original_classification: Optional[str] = None
    original_comments: str = ""
    original_classified_by: Optional[str] = None  # Added
    original_classified_date: Optional[str] = None  # Added
    _has_saved_classification: bool = False

    # Image lazy loading - store as OpenCV array (BGR format) for consistency
    # CRITICAL: Must be an instance attribute, not a class attribute
    _cv2_image: Optional[object] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Initialize image attribute after dataclass init"""
        # Ensure image is None for each instance
        self._cv2_image = None

    @property
    def image(self):
        """Property to access the image array"""
        return self._cv2_image

    @image.setter
    def image(self, value):
        """Setter for image property"""
        self._cv2_image = value

    def load_image(self):
        """Load image into memory as OpenCV array (lazy loading)"""
        if self._cv2_image is None:
            try:
                import cv2

                self._cv2_image = cv2.imread(self.image_path)
                if self._cv2_image is None:
                    logger.error(f"Failed to load image {self.filename}")
            except Exception as e:
                logger.error(f"Failed to load image {self.filename}: {e}")
        return self._cv2_image

    def unload_image(self):
        """Unload image from memory to save RAM"""
        self._cv2_image = None

    def has_changed(self) -> bool:
        """Check if image has unsaved changes"""
        return (
            self.classification != self.original_classification
            or self.comments != self.original_comments
        )

    def get_display_label(self) -> str:
        """Get display label for the image"""
        label = f"{self.hole_id}\n{int(self.depth_to)}m"
        if self.moisture_status:
            label += f"\n{self.moisture_status}"
        return label


class ReviewDataManager:
    """
    Manages compartment image data with NO UI dependencies.

    Responsibilities:
    - Load images from filesystem
    - Index images for fast lookup
    - Query images by hole, depth, etc.
    - Calculate statistics
    """

    # Compartment filename pattern
    COMPARTMENT_PATTERN = re.compile(
        r"([A-Z]{2}\d{4})_CC_(\d+)(?:_(Wet|Dry))?(?:_.*)?\.(?:jpg|jpeg|png|tif|tiff|heic)$",
        re.IGNORECASE,
    )

    def __init__(self, file_manager, json_manager=None):
        """Initialize the data manager"""
        self.file_manager = file_manager
        self.json_manager = json_manager
        self.logger = logging.getLogger(__name__)

        # Data storage
        self.all_images: List[CompartmentImage] = []
        self.image_index: Dict[Tuple[str, float], CompartmentImage] = {}
        self.hole_index: Dict[str, List[CompartmentImage]] = {}

        # Metadata
        self.hole_list: List[str] = []
        self.is_loaded = False

    def rebuild_indexes(self):
        """Rebuild all indexes after images are loaded"""
        self.image_index.clear()
        self.hole_index.clear()

        for img in self.all_images:
            key = (img.hole_id, img.depth_to)
            self.image_index[key] = img

            if img.hole_id not in self.hole_index:
                self.hole_index[img.hole_id] = []
            self.hole_index[img.hole_id].append(img)

        self.hole_list = sorted(list(self.hole_index.keys()))
        self.is_loaded = True

        self.logger.info(
            f"Rebuilt indexes for {len(self.all_images)} images across {len(self.hole_list)} holes"
        )

    def load_hole_list(self) -> List[str]:
        """
        Load list of hole IDs without loading all images.
        Lightweight operation for initial display.
        """
        try:
            approved_path = self.file_manager.get_shared_path("approved_compartments")
            if not approved_path or not approved_path.exists():
                self.logger.warning("Approved compartments path not found")
                return []

            # Scan for unique hole IDs
            hole_ids = set()
            for root, dirs, files in os.walk(approved_path):
                for file in files:
                    match = self.COMPARTMENT_PATTERN.match(file)
                    if match:
                        hole_ids.add(match.group(1))

            self.hole_list = sorted(hole_ids)
            self.logger.info(f"Found {len(self.hole_list)} holes")
            return self.hole_list

        except Exception as e:
            self.logger.error(f"Error loading hole list: {e}")
            return []

    def load_images_for_hole(self, hole_id: str) -> List[CompartmentImage]:
        """
        Load images for a specific hole.

        Args:
            hole_id: Hole identifier (e.g., "SB0096")

        Returns:
            List of CompartmentImage objects for this hole
        """
        try:
            approved_path = self.file_manager.get_shared_path("approved_compartments")
            if not approved_path:
                return []

            # Get hole directory (PROJECT_CODE/HOLE_ID/)
            project_code = hole_id[:2]
            hole_dir = approved_path / project_code / hole_id

            if not hole_dir.exists():
                self.logger.warning(f"Hole directory not found: {hole_dir}")
                return []

            images = []
            for file in hole_dir.iterdir():
                if not file.is_file():
                    continue

                match = self.COMPARTMENT_PATTERN.match(file.name)
                if not match:
                    continue

                # Parse filename
                matched_hole = match.group(1)
                comp_num = int(match.group(2))
                moisture = match.group(3) if match.group(3) else None

                # Create image object
                img = CompartmentImage(
                    filename=file.name,
                    hole_id=matched_hole,
                    depth_from=0.0,  # Will be calculated
                    depth_to=0.0,
                    image_path=str(file),
                    moisture_status=moisture,
                )

                images.append(img)

            # Calculate depths from compartment numbers
            if images:
                images = self._calculate_depths(images)

            # Load reviews if json_manager available
            if self.json_manager:
                self._load_reviews_for_images(images)

            # Index images
            for img in images:
                key = (img.hole_id, img.depth_to)
                self.image_index[key] = img

            # Store in hole index
            self.hole_index[hole_id] = images

            # Add to all_images if not already present
            existing_files = {img.filename for img in self.all_images}
            for img in images:
                if img.filename not in existing_files:
                    self.all_images.append(img)

            self.logger.info(f"Loaded {len(images)} images for hole {hole_id}")
            return images

        except Exception as e:
            self.logger.error(f"Error loading images for hole {hole_id}: {e}")
            return []

    def load_all_images(self) -> List[CompartmentImage]:
        """
        Load all images from all holes.
        WARNING: Can be slow for large datasets.
        """
        if not self.hole_list:
            self.load_hole_list()

        for hole_id in self.hole_list:
            if hole_id not in self.hole_index:
                self.load_images_for_hole(hole_id)

        self.is_loaded = True
        return self.all_images

    def get_images_for_hole(self, hole_id: str) -> List[CompartmentImage]:
        """
        Get images for a specific hole (loads if not already loaded).

        Args:
            hole_id: Hole identifier

        Returns:
            List of images for this hole
        """
        if hole_id not in self.hole_index:
            return self.load_images_for_hole(hole_id)
        return self.hole_index[hole_id]

    def get_all_images(self) -> List[CompartmentImage]:
        """
        Get all loaded images.

        Returns:
            List of all CompartmentImage objects currently loaded
        """
        return self.all_images

    def get_image_by_key(
        self, hole_id: str, depth_to: float
    ) -> Optional[CompartmentImage]:
        """
        Get a specific image by hole and depth.

        Args:
            hole_id: Hole identifier
            depth_to: Ending depth

        Returns:
            CompartmentImage or None if not found
        """
        key = (hole_id, depth_to)
        return self.image_index.get(key)

    def get_statistics(self, images: List[CompartmentImage]) -> Dict[str, int]:
        """
        Calculate classification statistics for a set of images.

        Args:
            images: List of images to analyze

        Returns:
            Dictionary with classification counts
        """
        stats = {
            "total": len(images),
            "classified": 0,
            "unassigned": 0,
            "classifications": {},  # Count per classification type
        }

        for img in images:
            if img.classification and img.classification != "Unassigned":
                stats["classified"] += 1

                # Count this classification
                if img.classification not in stats["classifications"]:
                    stats["classifications"][img.classification] = 0
                stats["classifications"][img.classification] += 1
            else:
                stats["unassigned"] += 1

        return stats

    def get_unsaved_images(
        self, images: List[CompartmentImage]
    ) -> List[CompartmentImage]:
        """Get images with unsaved changes"""
        return [img for img in images if img.has_changed()]

    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================

    def _calculate_depths(
        self, images: List[CompartmentImage]
    ) -> List[CompartmentImage]:
        """
        Calculate depth intervals from compartment numbers.
        Assumes 1m intervals by default.
        """
        # Sort by compartment number (extracted from filename)
        images.sort(key=lambda x: self._extract_comp_num(x.filename))

        # Calculate depths (1m intervals)
        for i, img in enumerate(images):
            img.depth_from = float(i)
            img.depth_to = float(i + 1)

        return images

    def _extract_comp_num(self, filename: str) -> int:
        """Extract compartment number from filename"""
        match = self.COMPARTMENT_PATTERN.match(filename)
        if match:
            return int(match.group(2))
        return 0

    def _load_reviews_for_images(self, images: List[CompartmentImage]):
        """Load review data for images from JSON register"""
        if not self.json_manager:
            return

        try:
            # Get all reviews for current user
            review_df = self.json_manager.get_review_data()

            if review_df.empty:
                return

            # Create lookup dictionary
            reviews_by_key = {}
            for _, row in review_df.iterrows():
                key = (row["HoleID"], int(row["From"]), int(row["To"]))

                # Store most recent review (they're already sorted by date)
                if key not in reviews_by_key:
                    reviews_by_key[key] = row.to_dict()

            # Apply reviews to images
            for img in images:
                key = (img.hole_id, int(img.depth_from), int(img.depth_to))

                if key in reviews_by_key:
                    review = reviews_by_key[key]

                    # Load classification
                    classification = (
                        review.get("Classification")
                        or review.get("Lithology")
                        or review.get("Rock_Type")
                        or "Unassigned"
                    )
                    img.classification = str(classification)

                    # Load other fields
                    img.comments = review.get("Comments", "")
                    img.classified_by = review.get("Reviewed_By", "")
                    img.classified_date = review.get("Review_Date", "")

                    # Mark as saved
                    img.original_classification = img.classification
                    img.original_comments = img.comments
                    img._has_saved_classification = True

            self.logger.info(f"Loaded reviews for {len(images)} images")

        except Exception as e:
            self.logger.error(f"Error loading reviews: {e}")
