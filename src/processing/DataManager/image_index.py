"""
image_index.py - Filesystem scanning and image path indexing.

This module provides:
- ImageIndex: Scans directories and builds key->path mappings
- Efficient O(1) lookups for compartment images
- Original image path resolution
- Metadata extraction from filenames

The ImageIndex is the primary interface for locating image files by ImageKey.

Author: George Symonds
"""

import os
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Iterator
from dataclasses import dataclass, field
from collections import defaultdict

from processing.DataManager.keys import ImageKey, FilenameParser, get_parser

logger = logging.getLogger(__name__)


@dataclass
class ImageInfo:
    """
    Complete information about an indexed image.
    
    Attributes:
        key: Unique ImageKey for this image
        path: Full path to the compartment image file
        filename: Just the filename (no path)
        hole_id: Extracted hole ID
        depth_from: Start depth of interval
        depth_to: End depth of interval
        moisture_status: "Wet", "Dry", or None
        original_path: Path to source original image (if resolved)
        file_size: Size in bytes (optional, for debugging)
    """
    key: ImageKey
    path: str
    filename: str
    hole_id: str
    depth_from: float
    depth_to: float
    moisture_status: Optional[str] = None
    original_path: Optional[str] = None
    file_size: int = 0
    
    def __hash__(self):
        return hash(self.key.to_tuple())
    
    def __eq__(self, other):
        if not isinstance(other, ImageInfo):
            return False
        return self.key == other.key


class ImageIndex:
    """
    Index of compartment and original images with O(1) key-based lookups.
    
    Scans configured directories and builds indexes:
    - compartment_index: ImageKey -> ImageInfo
    - hole_images: hole_id -> List[ImageKey] (for hole-based access)
    - original_index: (hole_id, depth_from, depth_to) -> path
    
    Usage:
        >>> index = ImageIndex()
        >>> index.add_compartment_folder("/path/to/approved")
        >>> index.add_original_folder("/path/to/originals")
        >>> index.build()
        >>> 
        >>> # O(1) lookup
        >>> info = index.get(ImageKey("BA0001", 45.0, "Wet"))
        >>> 
        >>> # Get all images for a hole
        >>> keys = index.get_keys_for_hole("BA0001")
    """
    
    # Supported image extensions
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}
    ORIGINAL_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".heic"}
    
    def __init__(self, valid_prefixes: Optional[Set[str]] = None):
        """
        Initialize the image index.
        
        Args:
            valid_prefixes: Optional set of valid hole ID prefixes.
        """
        self.parser = get_parser(valid_prefixes)
        
        # Directories to scan
        self._compartment_folders: List[Path] = []
        self._original_folders: List[Path] = []
        
        # Primary index: ImageKey.to_tuple() -> ImageInfo
        self._compartment_index: Dict[Tuple, ImageInfo] = {}
        
        # Secondary index: hole_id (upper) -> List[ImageKey]
        self._hole_images: Dict[str, List[ImageKey]] = defaultdict(list)
        
        # Original images: (hole_id, depth_from, depth_to) -> path
        self._original_index: Dict[Tuple[str, int, int], str] = {}
        
        # Metadata
        self._is_built = False
        self._build_time: float = 0
        self._total_files_scanned: int = 0
        self._total_images_indexed: int = 0
        self._unique_holes: Set[str] = set()
        
        logger.debug("ImageIndex initialized")
    
    # =========================================================================
    # Configuration
    # =========================================================================
    
    def add_compartment_folder(self, folder_path: str) -> "ImageIndex":
        """
        Add a folder to scan for compartment images.
        
        Args:
            folder_path: Path to folder containing compartment images
            
        Returns:
            self (for chaining)
        """
        path = Path(folder_path)
        if not path.exists():
            logger.warning(f"Compartment folder does not exist: {folder_path}")
        else:
            self._compartment_folders.append(path)
            logger.debug(f"Added compartment folder: {folder_path}")
        return self
    
    def add_original_folder(self, folder_path: str) -> "ImageIndex":
        """
        Add a folder to scan for original chip tray images.
        
        Args:
            folder_path: Path to folder containing original images
            
        Returns:
            self (for chaining)
        """
        path = Path(folder_path)
        if not path.exists():
            logger.warning(f"Original folder does not exist: {folder_path}")
        else:
            self._original_folders.append(path)
            logger.debug(f"Added original folder: {folder_path}")
        return self
    
    # =========================================================================
    # Building the Index
    # =========================================================================
    
    def build(self, progress_callback=None) -> "ImageIndex":
        """
        Scan all configured folders and build the indexes.
        
        Args:
            progress_callback: Optional callback(message, count) for progress updates
            
        Returns:
            self (for chaining)
        """
        start_time = time.time()
        
        logger.info("=" * 60)
        logger.info("BUILDING IMAGE INDEX")
        logger.info("=" * 60)
        logger.info(f"Compartment folders: {len(self._compartment_folders)}")
        logger.info(f"Original folders: {len(self._original_folders)}")
        
        # Clear existing indexes
        self._compartment_index.clear()
        self._hole_images.clear()
        self._original_index.clear()
        self._unique_holes.clear()
        self._total_files_scanned = 0
        self._total_images_indexed = 0
        
        # Scan compartment folders
        for folder in self._compartment_folders:
            self._scan_compartment_folder(folder, progress_callback)
        
        # Scan original folders
        for folder in self._original_folders:
            self._scan_original_folder(folder, progress_callback)
        
        # Link compartments to originals
        self._link_originals()
        
        # Build hole index
        self._build_hole_index()
        
        self._is_built = True
        self._build_time = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("IMAGE INDEX BUILD COMPLETE")
        logger.info(f"  Total files scanned: {self._total_files_scanned:,}")
        logger.info(f"  Compartments indexed: {self._total_images_indexed:,}")
        logger.info(f"  Holes with images: {len(self._unique_holes):,}")
        logger.info(f"  Original images: {len(self._original_index):,}")
        logger.info(f"  Build time: {self._build_time:.2f}s")
        logger.info("=" * 60)
        
        return self
    
    def _scan_compartment_folder(self, folder: Path, progress_callback=None):
        """
        Scan a folder for compartment images.
        
        Handles nested structure: folder/PROJECT/HOLE_ID/images
        """
        logger.info(f"Scanning compartment folder: {folder}")
        
        count = 0
        for root, dirs, files in os.walk(folder):
            for filename in files:
                self._total_files_scanned += 1
                
                # Check extension
                ext = Path(filename).suffix.lower()
                if ext not in self.IMAGE_EXTENSIONS:
                    continue
                
                # Parse filename
                parsed = self.parser.parse_compartment_filename(filename)
                if not parsed:
                    continue
                
                # Create ImageInfo
                full_path = os.path.join(root, filename)
                key = parsed["key"]
                
                info = ImageInfo(
                    key=key,
                    path=full_path,
                    filename=filename,
                    hole_id=parsed["hole_id"],
                    depth_from=parsed["depth_from"],
                    depth_to=parsed["depth_to"],
                    moisture_status=parsed["moisture_status"]
                )
                
                # Add to index (use tuple key for dict)
                key_tuple = key.to_tuple()
                
                # Handle duplicates (prefer later files, they may be updates)
                if key_tuple in self._compartment_index:
                    existing = self._compartment_index[key_tuple]
                    logger.debug(
                        f"Duplicate key {key}: replacing {existing.filename} with {filename}"
                    )
                
                self._compartment_index[key_tuple] = info
                self._unique_holes.add(key.hole_id_upper)
                self._total_images_indexed += 1
                count += 1
                
                # Progress callback
                if progress_callback and count % 5000 == 0:
                    progress_callback(f"Scanning: {count:,} images indexed", count)
        
        logger.info(f"  Indexed {count:,} compartment images from {folder.name}")
    
    def _scan_original_folder(self, folder: Path, progress_callback=None):
        """
        Scan a folder for original chip tray images.
        
        Handles nested structure: folder/PROJECT/HOLE_ID/images
        """
        logger.info(f"Scanning original folder: {folder}")
        
        count = 0
        for root, dirs, files in os.walk(folder):
            for filename in files:
                # Check extension
                ext = Path(filename).suffix.lower()
                if ext not in self.ORIGINAL_EXTENSIONS:
                    continue
                
                # Parse filename
                parsed = self.parser.parse_original_filename(filename)
                if not parsed:
                    continue
                
                full_path = os.path.join(root, filename)
                hole_id = parsed["hole_id"].upper()
                depth_from = int(parsed["depth_from"])
                depth_to = int(parsed["depth_to"])
                
                # Create key
                key = (hole_id, depth_from, depth_to)
                
                # Handle duplicates
                if key in self._original_index:
                    logger.debug(f"Duplicate original key {key}, keeping latest")
                
                self._original_index[key] = full_path
                count += 1
        
        logger.info(f"  Indexed {count:,} original images from {folder.name}")
    
    def _link_originals(self):
        """Link compartment images to their source original images."""
        linked_count = 0
        
        for key_tuple, info in self._compartment_index.items():
            # Find matching original by depth range
            # Compartment depth_to should be within an original's range
            for (orig_hole, orig_from, orig_to), orig_path in self._original_index.items():
                if info.hole_id.upper() == orig_hole:
                    if orig_from <= info.depth_to <= orig_to:
                        info.original_path = orig_path
                        linked_count += 1
                        break
        
        logger.debug(f"Linked {linked_count:,} compartments to original images")
    
    def _build_hole_index(self):
        """Build secondary index for hole-based access."""
        self._hole_images.clear()
        
        for key_tuple, info in self._compartment_index.items():
            hole_id = info.hole_id.upper()
            self._hole_images[hole_id].append(info.key)
        
        # Sort each hole's images by depth
        for hole_id in self._hole_images:
            self._hole_images[hole_id].sort(key=lambda k: (k.depth_to, k.moisture_status or ""))
        
        logger.debug(f"Built hole index for {len(self._hole_images):,} holes")
    
    # =========================================================================
    # Lookups
    # =========================================================================
    
    def get(self, key: ImageKey) -> Optional[ImageInfo]:
        """
        Get image info by key (O(1) lookup).
        
        Args:
            key: ImageKey to look up
            
        Returns:
            ImageInfo if found, None otherwise
        """
        return self._compartment_index.get(key.to_tuple())
    
    def get_by_components(
        self, 
        hole_id: str, 
        depth_to: float, 
        moisture_status: Optional[str] = None
    ) -> Optional[ImageInfo]:
        """
        Get image info by individual components.
        
        Args:
            hole_id: Hole identifier
            depth_to: End depth
            moisture_status: Optional "Wet" or "Dry"
            
        Returns:
            ImageInfo if found, None otherwise
        """
        key = ImageKey(hole_id.upper(), depth_to, moisture_status)
        return self.get(key)
    
    def get_path(self, key: ImageKey) -> Optional[str]:
        """
        Get just the file path for an image key.
        
        Args:
            key: ImageKey to look up
            
        Returns:
            File path if found, None otherwise
        """
        info = self.get(key)
        return info.path if info else None
    
    def get_original_path(self, key: ImageKey) -> Optional[str]:
        """
        Get the original (source) image path for a compartment.
        
        Args:
            key: ImageKey for the compartment
            
        Returns:
            Path to original image if found, None otherwise
        """
        info = self.get(key)
        return info.original_path if info else None
    
    def get_keys_for_hole(self, hole_id: str) -> List[ImageKey]:
        """
        Get all image keys for a specific hole.
        
        Args:
            hole_id: Hole identifier (case-insensitive)
            
        Returns:
            List of ImageKeys sorted by depth
        """
        return self._hole_images.get(hole_id.upper(), [])
    
    def get_images_for_hole(self, hole_id: str) -> List[ImageInfo]:
        """
        Get all image infos for a specific hole.
        
        Args:
            hole_id: Hole identifier (case-insensitive)
            
        Returns:
            List of ImageInfo sorted by depth
        """
        keys = self.get_keys_for_hole(hole_id)
        return [self.get(k) for k in keys if self.get(k)]
    
    def contains(self, key: ImageKey) -> bool:
        """Check if a key exists in the index."""
        return key.to_tuple() in self._compartment_index
    
    def __contains__(self, key: ImageKey) -> bool:
        """Support 'in' operator."""
        return self.contains(key)
    
    def __len__(self) -> int:
        """Return number of indexed images."""
        return len(self._compartment_index)
    
    # =========================================================================
    # Iteration
    # =========================================================================
    
    def keys(self) -> Iterator[ImageKey]:
        """Iterate over all image keys."""
        for info in self._compartment_index.values():
            yield info.key
    
    def values(self) -> Iterator[ImageInfo]:
        """Iterate over all image infos."""
        yield from self._compartment_index.values()
    
    def items(self) -> Iterator[Tuple[ImageKey, ImageInfo]]:
        """Iterate over (key, info) pairs."""
        for info in self._compartment_index.values():
            yield info.key, info
    
    def __iter__(self) -> Iterator[ImageKey]:
        """Default iteration over keys."""
        return self.keys()
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def is_built(self) -> bool:
        """Whether the index has been built."""
        return self._is_built
    
    @property
    def unique_holes(self) -> Set[str]:
        """Set of unique hole IDs in the index."""
        return self._unique_holes.copy()
    
    @property
    def hole_count(self) -> int:
        """Number of unique holes."""
        return len(self._unique_holes)
    
    @property
    def image_count(self) -> int:
        """Number of indexed images."""
        return len(self._compartment_index)
    
    @property
    def original_count(self) -> int:
        """Number of indexed original images."""
        return len(self._original_index)
    
    @property
    def build_time(self) -> float:
        """Time taken to build the index in seconds."""
        return self._build_time
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Returns:
            Dictionary with index statistics
        """
        return {
            "is_built": self._is_built,
            "compartment_folders": len(self._compartment_folders),
            "original_folders": len(self._original_folders),
            "total_files_scanned": self._total_files_scanned,
            "images_indexed": self._total_images_indexed,
            "unique_holes": len(self._unique_holes),
            "original_images": len(self._original_index),
            "build_time_seconds": self._build_time,
        }
    
    def get_depth_range(self, hole_id: str) -> Optional[Tuple[float, float]]:
        """
        Get the min/max depth range for a hole.
        
        Args:
            hole_id: Hole identifier
            
        Returns:
            Tuple of (min_depth, max_depth) or None if hole not found
        """
        keys = self.get_keys_for_hole(hole_id)
        if not keys:
            return None
        
        depths = [k.depth_to for k in keys]
        return (min(depths), max(depths))
    
    def filter_by_hole(self, hole_ids: Set[str]) -> List[ImageInfo]:
        """
        Filter images to only those in specified holes.
        
        Args:
            hole_ids: Set of hole IDs to include
            
        Returns:
            List of matching ImageInfo objects
        """
        hole_ids_upper = {h.upper() for h in hole_ids}
        result = []
        
        for info in self._compartment_index.values():
            if info.hole_id.upper() in hole_ids_upper:
                result.append(info)
        
        return sorted(result, key=lambda i: (i.hole_id, i.depth_to))
    
    def clear(self):
        """Clear all indexes."""
        self._compartment_index.clear()
        self._hole_images.clear()
        self._original_index.clear()
        self._unique_holes.clear()
        self._is_built = False
        self._build_time = 0
        self._total_files_scanned = 0
        self._total_images_indexed = 0
        logger.debug("ImageIndex cleared")
