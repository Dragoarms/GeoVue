"""
Review Filter Engine - Pure filter logic with NO state or UI.

This module provides stateless filtering functions that can be used
across all tabs without coupling to any specific UI implementation.
"""

import logging
from typing import List, Dict, Any, Optional, Callable
import pandas as pd


logger = logging.getLogger(__name__)


class ReviewFilterEngine:
    """
    Stateless filter engine for compartment images.

    All methods are static - NO instance state.
    This allows filters to be applied consistently across different tabs.
    """

    @staticmethod
    def apply_classification_filter(images: List, mode: str) -> List:
        """
        Filter images by classification status.

        Args:
            images: List of CompartmentImage objects
            mode: Filter mode - "all", "classified", "unclassified"

        Returns:
            Filtered list of images
        """
        if mode == "all":
            return images
        elif mode == "classified":
            return [
                img
                for img in images
                if img.classification and img.classification != "Unassigned"
            ]
        elif mode == "unclassified":
            return [
                img
                for img in images
                if not img.classification or img.classification == "Unassigned"
            ]
        else:
            logger.warning(f"Unknown classification filter mode: {mode}")
            return images

    @staticmethod
    def apply_dynamic_filters(images: List, filter_configs: List[Dict]) -> List:
        """Apply dynamic filters from DynamicFilterRow configurations"""
        if not filter_configs:
            return images

        filtered = images
        logger.debug(f"Starting with {len(filtered)} images")

        for config in filter_configs:
            column = config.get("column")
            operator = config.get("operator")
            value = config.get("value")

            if not column:
                continue

            # Log the filter being applied
            logger.debug(f"Applying filter: {column} {operator} {value}")

            # Count images before this filter
            before_count = len(filtered)

            # Apply the filter - call the static method correctly
            filtered = ReviewFilterEngine._apply_single_filter(filtered, config)

            # Log how many images passed this filter
            after_count = len(filtered)
            logger.debug(
                f"Filter {column} {operator} {value}: {before_count} -> {after_count} images"
            )

            # If this filter removed all images, log which holes were removed
            if after_count < before_count:
                remaining_holes = set(img.hole_id for img in filtered)
                logger.debug(f"Remaining holes after filter: {remaining_holes}")

        logger.debug(f"Final filtered count: {len(filtered)} images")
        return filtered

    @staticmethod
    def _get_value(img, column: str) -> Any:
        """Get value from image for a given column"""
        # Check standard fields first
        if column == "HoleID":
            return img.hole_id
        elif column == "From":
            return img.depth_from
        elif column == "To":
            return img.depth_to
        elif column == "Classification":
            return img.classification
        elif column == "Moisture":
            return img.moisture_status
        elif column == "Comments":
            return img.comments
        elif column == "Reviewed_By":
            return img.classified_by
        elif column == "Review_Date":
            return img.classified_date

        # Check csv_data - handle missing csv_data gracefully
        if hasattr(img, "csv_data") and img.csv_data and column in img.csv_data:
            return img.csv_data[column]

        return None

    @staticmethod
    def _get_value(img, column: str) -> Any:
        """Get value from image for a given column"""
        # Check standard fields first
        if column == "HoleID":
            return img.hole_id
        elif column == "From":
            return img.depth_from
        elif column == "To":
            return img.depth_to
        elif column == "Classification":
            return img.classification
        elif column == "Moisture":
            return img.moisture_status
        elif column == "Comments":
            return img.comments
        elif column == "Reviewed_By":
            return img.classified_by
        elif column == "Review_Date":
            return img.classified_date

        # Check csv_data
        if column in img.csv_data:
            return img.csv_data[column]

        return None

    @staticmethod
    def _is_null(img, column: str) -> bool:
        """Check if a value is null/empty"""
        value = ReviewFilterEngine._get_value(img, column)
        if value is None:
            return True
        if pd.isna(value):
            return True
        if isinstance(value, str) and value.strip() == "":
            return True
        return False

    @staticmethod
    def _apply_numeric_operator(
        row_value: Any, operator: str, value: str, value2: Optional[str] = None
    ) -> bool:
        """Apply numeric operator"""
        try:
            row_val = float(row_value)
            filter_val = float(value) if value else 0

            if operator == "=":
                return row_val == filter_val
            elif operator == "≠":
                return row_val != filter_val
            elif operator == "<":
                return row_val < filter_val
            elif operator == "≤":
                return row_val <= filter_val
            elif operator == ">":
                return row_val > filter_val
            elif operator == "≥":
                return row_val >= filter_val
            elif operator == "between" and value2:
                return float(value) <= row_val <= float(value2)
        except (ValueError, TypeError):
            return False

        return False

    @staticmethod
    def _apply_text_operator(row_value: Any, operator: str, value: str) -> bool:
        """Apply text operator"""
        row_val = str(row_value).lower()
        filter_val = str(value).lower() if value else ""

        if operator == "equals":
            return row_val == filter_val
        elif operator == "not equals":
            return row_val != filter_val
        elif operator == "contains":
            return filter_val in row_val
        elif operator == "not contains":
            return filter_val not in row_val
        elif operator == "starts with":
            return row_val.startswith(filter_val)
        elif operator == "ends with":
            return row_val.endswith(filter_val)
        elif operator == "in":
            values = [v.strip().lower() for v in value.split(",")]
            return row_val in values
        elif operator == "not in":
            values = [v.strip().lower() for v in value.split(",")]
            return row_val not in values
        elif operator == "like":
            import re

            pattern = filter_val.replace("%", ".*")
            return bool(re.match(pattern, row_val))

        return False

    @staticmethod
    def apply_sorting(images: List, column: str, order: str = "asc") -> List:
        """
        Sort images by a column.

        Args:
            images: List of CompartmentImage objects
            column: Column name to sort by
            order: "asc" or "desc"

        Returns:
            Sorted list of images
        """
        if not column:
            return images

        def get_sort_key(img):
            """Get sort key for an image"""
            value = ReviewFilterEngine._get_value(img, column)

            # Handle nulls - sort to end
            if value is None or pd.isna(value):
                return (1, "")  # Tuple for consistent sorting

            # Try numeric sort first
            try:
                return (0, float(value))
            except (ValueError, TypeError):
                # Fall back to string sort
                return (0, str(value).lower())

        reverse = order == "desc"

        try:
            return sorted(images, key=get_sort_key, reverse=reverse)
        except Exception as e:
            logger.error(f"Error sorting by {column}: {e}")
            return images

    @staticmethod
    def _apply_single_filter(images: List, config: Dict) -> List:
        """
        Apply a single filter configuration to images.

        Args:
            images: List of CompartmentImage objects
            config: Filter configuration dict with column, operator, value, etc.

        Returns:
            Filtered list of images
        """
        column = config.get("column")
        operator = config.get("operator")
        value = config.get("value")
        value2 = config.get("value2")  # For between operator

        if not column:
            return images

        filtered = []
        for img in images:
            # Handle null operators
            if operator == "is null":
                if ReviewFilterEngine._is_null(img, column):
                    filtered.append(img)
            elif operator == "is not null":
                if not ReviewFilterEngine._is_null(img, column):
                    filtered.append(img)
            else:
                # Get the actual value
                row_value = ReviewFilterEngine._get_value(img, column)

                # Skip null values unless using null operators
                if row_value is None or pd.isna(row_value):
                    continue

                # Determine if this is a numeric or text column
                is_numeric = False
                try:
                    float(row_value)
                    is_numeric = True
                except (ValueError, TypeError):
                    pass

                # Apply the appropriate operator
                if is_numeric and operator in ["=", "≠", "<", "≤", ">", "≥", "between"]:
                    if ReviewFilterEngine._apply_numeric_operator(
                        row_value, operator, value, value2
                    ):
                        filtered.append(img)
                elif operator in [
                    "equals",
                    "not equals",
                    "contains",
                    "not contains",
                    "starts with",
                    "ends with",
                    "in",
                    "not in",
                    "like",
                ]:
                    if ReviewFilterEngine._apply_text_operator(
                        row_value, operator, value
                    ):
                        filtered.append(img)

        return filtered

    @staticmethod
    def apply_hole_filter(images: List, hole_id: str) -> List:
        """
        Filter images to a specific hole.

        Args:
            images: List of CompartmentImage objects
            hole_id: Hole identifier

        Returns:
            Filtered list
        """
        return [img for img in images if img.hole_id == hole_id]
