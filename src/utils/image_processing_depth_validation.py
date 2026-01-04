import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import math
import traceback


class DepthValidator:
    """Validates hole IDs and depth ranges against a reference CSV."""

    def __init__(self, csv_path: Optional[str] = None):
        """Initialize the depth validator."""
        self.logger = logging.getLogger(__name__)
        self.csv_path = csv_path
        self.depth_ranges = {}
        self.is_loaded = False

        self.logger.info(f"DepthValidator initialized with csv_path: {csv_path}")

        if csv_path and Path(csv_path).exists():
            self.logger.info(f"CSV file exists at {csv_path}, attempting to load")
            self.load_depth_data()
        else:
            if csv_path:
                self.logger.warning(f"CSV path provided but file not found: {csv_path}")
            else:
                self.logger.info("No CSV path provided to DepthValidator")

    def load_depth_data(self) -> bool:
        """Load and summarize depth data from CSV."""
        try:
            if not self.csv_path or not Path(self.csv_path).exists():
                self.logger.warning("Depth validation CSV not found or not configured")
                self.logger.warning(f"CSV path: {self.csv_path}")
                self.logger.warning(
                    f"Path exists: {Path(self.csv_path).exists() if self.csv_path else False}"
                )
                return False

            self.logger.info(f"Reading CSV from: {self.csv_path}")

            # Read CSV
            df = pd.read_csv(self.csv_path)
            self.logger.info(f"CSV loaded, shape: {df.shape}")
            self.logger.info(f"CSV columns: {df.columns.tolist()}")

            # Check required columns exist
            required_cols = ["HoleID", "From", "To"]
            if not all(col in df.columns for col in required_cols):
                self.logger.error(
                    f"CSV missing required columns. Found: {df.columns.tolist()}"
                )
                return False

            # Clean and process data
            df["HoleID"] = df["HoleID"].astype(str).str.strip().str.upper()
            df["From"] = pd.to_numeric(df["From"], errors="coerce")
            df["To"] = pd.to_numeric(df["To"], errors="coerce")

            # Remove rows with invalid data
            df = df.dropna(subset=["HoleID", "From", "To"])

            # Group by HoleID and get min/max depths
            grouped = (
                df.groupby("HoleID").agg({"From": "min", "To": "max"}).reset_index()
            )

            # Store as dictionary for fast lookup, rounding max depth up to nearest 20
            self.depth_ranges = {}
            for _, row in grouped.iterrows():
                min_depth = row["From"]
                max_depth = row["To"]
                # Round max depth up to nearest 20
                rounded_max_depth = math.ceil(max_depth / 20) * 20
                self.depth_ranges[row["HoleID"]] = (min_depth, rounded_max_depth)

                # Log if rounding occurred
                if rounded_max_depth != max_depth:
                    self.logger.debug(
                        f"Hole {row['HoleID']}: rounded max depth from {max_depth} to {rounded_max_depth}"
                    )

            self.is_loaded = True
            self.logger.info(
                f"Successfully loaded depth ranges for {len(self.depth_ranges)} holes"
            )
            return True

        except Exception as e:
            self.logger.error(f"Error loading depth validation CSV: {e}")
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            self.is_loaded = False
            return False

    def validate_depth_range(
        self, hole_id: str, depth_from: float, depth_to: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate if the given depth range is within the valid range for the hole.

        Args:
            hole_id: Hole identifier (case insensitive)
            depth_from: Starting depth
            depth_to: Ending depth

        Returns:
            Tuple of (is_valid, error_message)
            If is_valid is True, error_message will be None
        """
        # If no data loaded, validation passes (non-blocking)
        if not self.is_loaded:
            return True, None

        # Normalize hole_id for case-insensitive lookup
        hole_id_upper = str(hole_id).strip().upper()

        # Check if hole exists in data
        if hole_id_upper not in self.depth_ranges:
            return False, f"Hole ID '{hole_id}' not found in depth validation data"

        # Get valid range
        valid_from, valid_to = self.depth_ranges[hole_id_upper]

        # Check if depths are within range
        if depth_from < valid_from:
            return (
                False,
                f"Depth {depth_from}m is before the valid start depth of {valid_from}m for hole {hole_id}",
            )

        if depth_to > valid_to:
            return (
                False,
                f"Depth {depth_to}m is beyond the valid end depth of {valid_to}m for hole {hole_id}",
            )

        return True, None

    def get_valid_range(self, hole_id: str) -> Optional[Tuple[float, float]]:
        """
        Get the valid depth range for a hole ID.

        Args:
            hole_id: Hole identifier (case insensitive)

        Returns:
            Tuple of (min_depth, max_depth) or None if not found
        """
        if not self.is_loaded:
            return None

        hole_id_upper = str(hole_id).strip().upper()
        return self.depth_ranges.get(hole_id_upper)

    def reload(self, csv_path: Optional[str] = None) -> bool:
        """
        Reload depth data from CSV.

        Args:
            csv_path: Optional new CSV path

        Returns:
            True if successfully loaded
        """
        if csv_path:
            self.csv_path = csv_path

        self.depth_ranges = {}
        self.is_loaded = False

        if self.csv_path:
            return self.load_depth_data()

        return False
