"""
keys.py - Image key generation and filename parsing utilities.

This module provides:
- ImageKey: Immutable dataclass for unique image identification
- Filename parsing to extract hole_id, depth, moisture status
- Key normalization for consistent lookups

The ImageKey is the fundamental unit for cross-referencing data across:
- Filesystem (compartment images, original images)
- JSON registers (reviews, classifications, image properties)
- CSV datasets (geological data)

Author: George Symonds
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageKey:
    """
    Unique identifier for a compartment image.
    
    The natural key is (hole_id, depth_to) with optional moisture_status
    for distinguishing wet/dry variants.
    
    This is frozen (immutable) so it can be used as a dictionary key or in sets.
    
    Attributes:
        hole_id: Drillhole identifier (e.g., "BA0001", "KM0137")
        depth_to: End depth of the interval in meters (e.g., 45.0)
        moisture_status: Optional "Wet" or "Dry" suffix
        
    Examples:
        >>> key = ImageKey("BA0001", 45.0, "Wet")
        >>> key.to_tuple()
        ('BA0001', 45, 'Wet')
        >>> key.to_base_tuple()
        ('BA0001', 45)
    """
    hole_id: str
    depth_to: float
    moisture_status: Optional[str] = None
    
    def __post_init__(self):
        """Validate and normalize fields after initialization."""
        # We can't modify frozen dataclass directly, validation only
        if not self.hole_id:
            raise ValueError("hole_id cannot be empty")
        if self.depth_to < 0:
            raise ValueError(f"depth_to cannot be negative: {self.depth_to}")
        if self.moisture_status and self.moisture_status not in ("Wet", "Dry"):
            logger.warning(
                f"Unexpected moisture_status '{self.moisture_status}' for {self.hole_id}. "
                f"Expected 'Wet' or 'Dry'."
            )
    
    @property
    def depth_to_int(self) -> int:
        """Get depth_to as integer (for index lookups)."""
        return int(self.depth_to)
    
    @property
    def hole_id_upper(self) -> str:
        """Get hole_id in uppercase (for case-insensitive matching)."""
        return self.hole_id.upper()
    
    def to_tuple(self) -> Tuple[str, int, Optional[str]]:
        """
        Convert to tuple for use as dictionary key.
        
        Returns:
            Tuple of (hole_id_upper, depth_to_int, moisture_status)
        """
        return (self.hole_id_upper, self.depth_to_int, self.moisture_status)
    
    def to_base_tuple(self) -> Tuple[str, int]:
        """
        Convert to base tuple (without moisture) for CSV lookups.
        
        CSV data typically doesn't distinguish wet/dry, so this provides
        the key format for geological data lookups.
        
        Returns:
            Tuple of (hole_id_upper, depth_to_int)
        """
        return (self.hole_id_upper, self.depth_to_int)
    
    def matches_interval(self, hole_id: str, depth_from: float, depth_to: float) -> bool:
        """
        Check if this key matches a given interval.
        
        Args:
            hole_id: Hole ID to match
            depth_from: Start depth (not used in matching, for API consistency)
            depth_to: End depth to match
            
        Returns:
            True if hole_id and depth_to match (case-insensitive)
        """
        return (
            self.hole_id_upper == hole_id.upper() and 
            self.depth_to_int == int(depth_to)
        )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        if self.moisture_status:
            return f"{self.hole_id}_{self.depth_to_int}m_{self.moisture_status}"
        return f"{self.hole_id}_{self.depth_to_int}m"
    
    def __repr__(self) -> str:
        """Debug representation."""
        return f"ImageKey({self.hole_id!r}, {self.depth_to}, {self.moisture_status!r})"


class FilenameParser:
    """
    Parses compartment image filenames to extract metadata.
    
    Supports multiple filename patterns used in GeoVue:
    
    Pattern 1 (Compartment): BA0001_CC_045_Wet.png
        - hole_id: BA0001
        - depth_to: 45
        - moisture: Wet
        
    Pattern 2 (Original): BA0001_20-40m.jpg
        - hole_id: BA0001  
        - depth_from: 20
        - depth_to: 40
        
    Pattern 3 (Simple depth): BA0001_045.png
        - hole_id: BA0001
        - depth_to: 45
    """
    
    # Compiled regex patterns for performance
    # Pattern: BA0001_CC_045_Wet.png or BA0001_CC_045.png
    COMPARTMENT_PATTERN = re.compile(
        r'^([A-Z]{2}\d{4})_CC_(\d+)(?:_(Wet|Dry))?(?:_.*)?\.(?:png|tiff?|jpe?g)$',
        re.IGNORECASE
    )
    
    # Pattern: BA0001_20-40m.jpg (original images)
    ORIGINAL_PATTERN = re.compile(
        r'^([A-Z]{2}\d{4})_(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)m?.*\.(?:png|tiff?|jpe?g|heic)$',
        re.IGNORECASE
    )

    
    # Pattern: Simple depth - BA0001_045.png
    SIMPLE_DEPTH_PATTERN = re.compile(
        r'^([A-Z]{2}\d{4})_(\d+)\.(?:png|tiff?|jpe?g)$',
        re.IGNORECASE
    )
    
    # Configurable hole ID prefixes (can be extended)
    VALID_PREFIXES = {"BA", "NB", "SB", "KM", "OK", "BB", "BT"}
    
    def __init__(self, valid_prefixes: Optional[set] = None):
        """
        Initialize the filename parser.
        
        Args:
            valid_prefixes: Optional set of valid hole ID prefixes.
                           Defaults to common Belinga prefixes.
        """
        self.valid_prefixes = valid_prefixes or self.VALID_PREFIXES
        logger.debug(f"FilenameParser initialized with prefixes: {self.valid_prefixes}")
    
    def parse_compartment_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Parse a compartment image filename.
        
        Args:
            filename: Filename to parse (e.g., "BA0001_CC_045_Wet.png")
            
        Returns:
            Dictionary with parsed fields, or None if parsing fails:
            {
                "hole_id": str,
                "depth_to": float,
                "moisture_status": Optional[str],
                "key": ImageKey
            }
        """
        match = self.COMPARTMENT_PATTERN.match(filename)
        if not match:
            logger.debug(f"Filename '{filename}' did not match compartment pattern")
            return None
        
        hole_id = match.group(1).upper()
        depth_str = match.group(2)
        moisture = match.group(3)  # May be None
        
        # Validate prefix
        prefix = hole_id[:2]
        if prefix not in self.valid_prefixes:
            logger.debug(f"Invalid prefix '{prefix}' in filename '{filename}'")
            return None
        
        try:
            depth_to = float(depth_str)
        except ValueError:
            logger.warning(f"Could not parse depth '{depth_str}' in filename '{filename}'")
            return None
        
        # Normalize moisture status
        if moisture:
            moisture = moisture.capitalize()  # "wet" -> "Wet"
        
        result = {
            "hole_id": hole_id,
            "depth_to": depth_to,
            "depth_from": depth_to - 1.0,  # Assume 1m interval for compartments
            "moisture_status": moisture,
            "key": ImageKey(hole_id, depth_to, moisture)
        }
        
        logger.debug(f"Parsed compartment filename '{filename}': {result}")
        return result
    
    def parse_original_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Parse an original chip tray image filename.
        
        Args:
            filename: Filename to parse (e.g., "BA0001_20-40m.jpg")
            
        Returns:
            Dictionary with parsed fields, or None if parsing fails:
            {
                "hole_id": str,
                "depth_from": float,
                "depth_to": float
            }
        """
        match = self.ORIGINAL_PATTERN.match(filename)
        if not match:
            logger.debug(f"Filename '{filename}' did not match original pattern")
            return None
        
        hole_id = match.group(1).upper()
        depth_from_str = match.group(2)
        depth_to_str = match.group(3)
        
        try:
            depth_from = float(depth_from_str)
            depth_to = float(depth_to_str)
        except ValueError:
            logger.warning(f"Could not parse depths in filename '{filename}'")
            return None
        
        result = {
            "hole_id": hole_id,
            "depth_from": depth_from,
            "depth_to": depth_to
        }
        
        logger.debug(f"Parsed original filename '{filename}': {result}")
        return result
    
    def create_key_from_filename(self, filename: str) -> Optional[ImageKey]:
        """
        Create an ImageKey from a filename.
        
        Tries compartment pattern first, then falls back to simple depth.
        
        Args:
            filename: Filename to parse
            
        Returns:
            ImageKey if parsing succeeds, None otherwise
        """
        # Try compartment pattern first
        result = self.parse_compartment_filename(filename)
        if result:
            return result["key"]
        
        # Try simple depth pattern
        match = self.SIMPLE_DEPTH_PATTERN.match(filename)
        if match:
            hole_id = match.group(1).upper()
            depth_to = float(match.group(2))
            return ImageKey(hole_id, depth_to)
        
        return None
    
    def create_key(
        self, 
        hole_id: str, 
        depth_to: float, 
        moisture_status: Optional[str] = None
    ) -> ImageKey:
        """
        Create an ImageKey from explicit values.
        
        Args:
            hole_id: Drillhole identifier
            depth_to: End depth in meters
            moisture_status: Optional "Wet" or "Dry"
            
        Returns:
            New ImageKey instance
        """
        return ImageKey(
            hole_id=hole_id.upper(),
            depth_to=float(depth_to),
            moisture_status=moisture_status.capitalize() if moisture_status else None
        )


# Module-level parser instance for convenience
_default_parser = None

def get_parser(valid_prefixes: Optional[set] = None) -> FilenameParser:
    """
    Get a FilenameParser instance.
    
    Args:
        valid_prefixes: Optional custom prefixes. If None, uses cached default.
        
    Returns:
        FilenameParser instance
    """
    global _default_parser
    
    if valid_prefixes is not None:
        return FilenameParser(valid_prefixes)
    
    if _default_parser is None:
        _default_parser = FilenameParser()
    
    return _default_parser


def parse_compartment_filename(filename: str) -> Optional[Dict[str, Any]]:
    """Convenience function to parse compartment filename using default parser."""
    return get_parser().parse_compartment_filename(filename)


def create_key(
    hole_id: str, 
    depth_to: float, 
    moisture_status: Optional[str] = None
) -> ImageKey:
    """Convenience function to create an ImageKey."""
    return get_parser().create_key(hole_id, depth_to, moisture_status)
