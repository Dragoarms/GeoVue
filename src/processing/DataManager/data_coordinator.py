"""
data_coordinator.py - Unified data access API for GeoVue.

This module provides:
- DataCoordinator: Single entry point for all drillhole data operations
- Coordinates ImageIndex, RegisterStore, and GeologicalStore
- Provides CompartmentImage-compatible data retrieval
- Supports efficient filtering without massive DataFrame merges

This is the primary interface that LoggingReviewDialog and other UI components
should use instead of accessing stores directly.

Usage:
    >>> coordinator = DataCoordinator(config_manager, file_manager)
    >>> coordinator.initialize(
    ...     compartment_folders=[shared_path, local_path],
    ...     original_folder=originals_path,
    ...     csv_files=[drillhole_csv],
    ...     json_manager=json_register_manager
    ... )
    >>> 
    >>> # Get all data for a specific image
    >>> data = coordinator.get_image_data(ImageKey("BA0001", 45.0, "Wet"))
    >>> 
    >>> # Get filtered image keys
    >>> keys = coordinator.get_filtered_keys(hole_id="BA0001", classified_only=False)

Author: George Symonds
"""

import logging
import time
from typing import Dict, List, Optional, Set, Tuple, Any, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
from pathlib import Path

if TYPE_CHECKING:
    import pandas as pd

from processing.DataManager.keys import ImageKey, FilenameParser, get_parser
from processing.DataManager.schema import DataSourceSchema, DataType
from processing.DataManager.image_index import ImageIndex, ImageInfo
from processing.DataManager.geological_store import GeologicalStore
from processing.DataManager.register_store import RegisterStore, ReviewMetadata, ImageProperties
from processing.DataManager.color_map_store import ColorMapStore
from processing.DataManager.rc_metrics_store import RCMetricsStore, IntervalMetrics
from processing.DataManager.column_aliases import ColumnResolver
from processing.DataManager.survey_trace import build_trace, xyz_at_depth as trace_xyz_at_depth

logger = logging.getLogger(__name__)


@dataclass
class CompartmentData:
    """
    Complete data for a compartment image.
    
    Aggregates data from all stores into a single object that can be
    used to populate CompartmentImage instances or filter DataFrames.
    
    Attributes:
        key: The ImageKey for this compartment
        
        # From ImageIndex
        image_path: Path to the compartment image file
        original_path: Path to the source original image
        filename: Just the filename
        hole_id: Hole identifier
        depth_from: Start depth
        depth_to: End depth
        moisture_status: "Wet", "Dry", or None
        
        # From RegisterStore
        classification: Current user's classification
        tags: Set of tag IDs
        comments: User comments
        consensus_classification: Most common classification
        review_count: Number of reviewers
        agreement: Agreement level
        
        # From RegisterStore (image properties)
        wet_hex: Hex color for wet image
        dry_hex: Hex color for dry image
        combined_hex: Combined hex color (dry preferred but fill gaps with wet)
        
        # From GeologicalStore
        csv_data: Dictionary of geological data from CSVs
    """
    key: ImageKey
    
    # Image Index data
    image_path: str = ""
    original_path: Optional[str] = None
    filename: str = ""
    hole_id: str = ""
    depth_from: float = 0.0 # TODO depths can be floats or integers
    depth_to: float = 0.0 # TODO depths can be floats or integers
    moisture_status: Optional[str] = None
    
    # Register data
    classification: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    comments: str = ""
    consensus_classification: Optional[str] = None
    review_count: int = 0
    agreement: str = "none"
    
    # Image properties
    wet_hex: Optional[str] = None
    dry_hex: Optional[str] = None
    combined_hex: Optional[str] = None
    
    # Geological data
    csv_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for DataFrame row or filter evaluation.
        
        Returns all fields as a flat dictionary.
        """
        result = {
            "hole_id": self.hole_id,
            "depth_from": self.depth_from,
            "depth_to": self.depth_to,
            "moisture_status": self.moisture_status or "",
            "image_path": self.image_path,
            "original_path": self.original_path or "",
            "classification": self.classification or "",
            "comments": self.comments,
            "consensus_classification": self.consensus_classification or "",
            "review_count": self.review_count,
            "agreement": self.agreement,
            "wet_hex": self.wet_hex,
            "dry_hex": self.dry_hex,
            "combined_hex": self.combined_hex,
        }
        
        # Add CSV data
        result.update(self.csv_data)
        
        # Add tag columns as booleans
        for tag_id in self.tags:
            result[f"tag_{tag_id}"] = True
        
        return result


class DataCoordinator:
    """
    Unified data access coordinator for GeoVue.
    
    Brings together:
    - ImageIndex: Filesystem scanning and path lookups
    - RegisterStore: JSON register data (reviews, properties)
    - GeologicalStore: CSV geological data
    
    Provides a single API for:
    - Getting complete data for an image
    - Filtering images by various criteria
    - Building DataFrames for specific purposes
    
    This replaces the scattered data access in LoggingReviewDialog with
    a clean, efficient, testable interface.
    """
    
    def __init__(self, config_manager=None, file_manager=None):
        """
        Initialize the data coordinator.
        
        Args:
            config_manager: Optional ConfigManager for settings persistence
            file_manager: Optional FileManager for path resolution
        """
        self.config_manager = config_manager
        self.file_manager = file_manager
        
        # Create stores in correct order
        self._image_index = ImageIndex()
        self._geological_store = GeologicalStore(config_manager)
        # RegisterStore created without manager - set during initialize()
        self._register_store = RegisterStore(
            json_manager=None,
            geological_store=self._geological_store
        )
        
        # ColorMapStore for centralized color map access
        self._color_map_store: Optional[ColorMapStore] = None

        # RCMetricsStore for pre-computed RC logging metrics
        self._rc_metrics_store: Optional[RCMetricsStore] = None

        # State
        self._is_initialized = False
        self._initialization_time: float = 0

        # Cache for tag definitions (set externally)
        self._tag_definitions: List[Any] = []

        logger.debug("DataCoordinator initialized")
    
    # =========================================================================
    # Initialization
    # =========================================================================
    
    def initialize(
        self,
        compartment_folders: List[str],
        original_folder: Optional[str] = None,
        csv_files: Optional[List[str]] = None,
        json_manager=None,
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> "DataCoordinator":
        """
        Initialize all data stores.
        
        This is the main setup method that should be called once when
        the application starts or when data needs to be reloaded.
        
        Args:
            compartment_folders: List of folders containing compartment images
            original_folder: Optional folder containing original chip tray images
            csv_files: Optional list of CSV files to load
            json_manager: Optional JSONRegisterManager instance
            progress_callback: Optional callback(message, count) for progress
            
        Returns:
            self (for chaining)
        """
        start_time = time.time()
        
        logger.info("=" * 70)
        logger.info("INITIALIZING DATA COORDINATOR")
        logger.info("=" * 70)
        
        # Configure and build image index
        logger.info("[1/4] Configuring image index...")
        for folder in compartment_folders:
            if folder:
                self._image_index.add_compartment_folder(folder)
        
        if original_folder:
            self._image_index.add_original_folder(original_folder)
        
        if progress_callback:
            progress_callback("Scanning image folders...", 0)
        
        logger.info("[2/4] Building image index...")
        self._image_index.build(progress_callback)
        
        # Configure geological store
        logger.info("[3/4] Loading geological data sources...")
        if csv_files:
            for csv_path in csv_files:
                if csv_path and Path(csv_path).exists():
                    self._geological_store.add_source(csv_path)
        
        if progress_callback:
            progress_callback("Loading CSV data...", 0)
        
        self._geological_store.load_all()
        
        # Configure register store
        logger.info("[4/4] Building register cache...")
        if json_manager:
            self._register_store.set_manager(json_manager)
            
            if progress_callback:
                progress_callback("Building register cache...", 0)
            
            # Skip pre-caching - use lazy loading instead
            # Cache will be built on-demand when data is accessed
            logger.info("Register store configured (lazy caching enabled)")
        
        self._is_initialized = True
        self._initialization_time = time.time() - start_time
        
        # Initialize color map store
        logger.info("[5/5] Initializing color map store...")
        try:
            self._color_map_store = ColorMapStore(config_manager=self.config_manager)
        except Exception as e:
            logger.warning(f"Could not initialize ColorMapStore: {e}")
            self._color_map_store = None
        
        # Run data sanitization after all sources are loaded
        if self._geological_store.is_loaded:
            try:
                from processing.DataManager.data_sanitizer import sanitize_geological_store, log_sanitization_report
                report = sanitize_geological_store(self._geological_store)
                log_sanitization_report(report)

                if report.has_critical:
                    logger.error("Critical data issues found - some features may not work correctly")
            except Exception as e:
                logger.warning(f"Data sanitization failed: {e}")

        # Compute RC metrics for sources with mineral columns
        logger.info("[6/6] Computing RC metrics...")
        if progress_callback:
            progress_callback("Computing RC metrics...", 0)
        rc_metrics_count = self._compute_rc_metrics_for_sources(progress_callback)

        self._initialization_time = time.time() - start_time

        logger.info("=" * 70)
        logger.info("DATA COORDINATOR INITIALIZATION COMPLETE")
        logger.info(f"  Total time: {self._initialization_time:.2f}s")
        logger.info(f"  Images indexed: {self._image_index.image_count:,}")
        logger.info(f"  Holes with images: {self._image_index.hole_count:,}")
        logger.info(f"  CSV rows: {self._geological_store.total_rows:,}")
        if rc_metrics_count > 0:
            logger.info(f"  RC metrics computed: {rc_metrics_count:,} intervals")
        logger.info("=" * 70)

        return self
    
    def set_tag_definitions(self, tag_definitions: List[Any]):
        """
        Set tag definitions for tag column generation.

        Args:
            tag_definitions: List of ItemDefinition objects from ImageClassificationAndTagManager
        """
        self._tag_definitions = tag_definitions
        logger.debug(f"Set {len(tag_definitions)} tag definitions")

    def _compute_rc_metrics_for_sources(
        self,
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> int:
        """
        Auto-detect and compute RC metrics for geological sources with mineral columns.

        Looks for sources containing mineral percentage columns (min_*_pct pattern)
        and computes hardness, gangue, and zonation metrics for each interval.

        Args:
            progress_callback: Optional callback(message, count) for progress

        Returns:
            Total number of intervals processed
        """
        if not self._geological_store.is_loaded:
            logger.debug("No geological data loaded, skipping RC metrics computation")
            return 0

        total_metrics = 0
        sources_processed = 0

        # Find sources with mineral columns
        for source_name in self._geological_store.list_sources():
            source = self._geological_store.get_source(source_name)
            if source is None or source.df is None or source.df.empty:
                continue

            df = source.df

            # Check for mineral columns (min_*_pct pattern)
            mineral_cols = [c for c in df.columns if c.lower().startswith('min_') and c.lower().endswith('_pct')]

            if not mineral_cols:
                logger.debug(f"Source '{source_name}' has no mineral columns, skipping")
                continue

            logger.info(f"Found {len(mineral_cols)} mineral columns in '{source_name}', computing RC metrics...")

            # Initialize RC metrics store if needed
            if self._rc_metrics_store is None:
                self._rc_metrics_store = RCMetricsStore()
                if not self._rc_metrics_store.initialize():
                    logger.warning("Failed to initialize RC metrics store")
                    return 0

            try:
                # Compute metrics for this source
                success = self._rc_metrics_store.compute_from_dataframe(
                    df=df,
                    source_name=source_name,
                    progress_callback=progress_callback
                )

                if success:
                    sources_processed += 1
                    if self._rc_metrics_store.is_loaded:
                        total_metrics = self._rc_metrics_store.metrics_count
                    logger.info(f"RC metrics computed for '{source_name}': {total_metrics:,} intervals")
                else:
                    logger.warning(f"Failed to compute RC metrics for '{source_name}'")

            except Exception as e:
                logger.error(f"Error computing RC metrics for '{source_name}': {e}", exc_info=True)

        if sources_processed == 0:
            logger.debug("No sources with mineral columns found")
        else:
            logger.info(f"RC metrics computed from {sources_processed} source(s), {total_metrics:,} total intervals")

        return total_metrics
    
    # =========================================================================
    # Store Access (for advanced usage)
    # =========================================================================
    
    @property
    def image_index(self) -> ImageIndex:
        """Get the underlying ImageIndex."""
        return self._image_index
    
    @property
    def color_maps(self) -> Optional[ColorMapStore]:
        """Get the color map store for accessing color mappings."""
        return self._color_map_store

    @property
    def register_store(self) -> RegisterStore:
        """Get the underlying RegisterStore."""
        return self._register_store
    
    @property
    def geological_store(self) -> GeologicalStore:
        """Get the underlying GeologicalStore."""
        return self._geological_store
    
    @property
    def is_initialized(self) -> bool:
        """Whether the coordinator has been initialized."""
        return self._is_initialized

    # =========================================================================
    # RC Metrics Store
    # =========================================================================

    @property
    def rc_metrics_store(self) -> Optional[RCMetricsStore]:
        """Get the RC metrics store (may be None if not initialized)."""
        return self._rc_metrics_store

    @property
    def has_rc_metrics(self) -> bool:
        """Whether RC metrics have been computed and are available."""
        return self._rc_metrics_store is not None and self._rc_metrics_store.is_loaded

    def initialize_rc_metrics(
        self,
        mineral_codes_path: Optional[str] = None
    ) -> bool:
        """
        Initialize the RC metrics store.

        This loads the mineral codes but does not compute metrics yet.
        Call compute_rc_metrics() to calculate metrics from data.

        Args:
            mineral_codes_path: Optional path to mineral_codes.json

        Returns:
            True if successful
        """
        from pathlib import Path

        path = Path(mineral_codes_path) if mineral_codes_path else None
        self._rc_metrics_store = RCMetricsStore(mineral_codes_path=path)
        return self._rc_metrics_store.initialize()

    def compute_rc_metrics(
        self,
        source_name: str,
        hole_id_col: str = "holeid",
        from_col: str = "sampfrom",
        to_col: str = "sampto",
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> bool:
        """
        Compute RC metrics from a geological data source.

        This pre-computes hardness, gangue, and zonation metrics
        for all intervals in the specified data source.

        Args:
            source_name: Name of the geological data source (e.g., "drillhole_data")
            hole_id_col: Column name for hole ID
            from_col: Column name for from depth
            to_col: Column name for to depth
            progress_callback: Optional callback(message, percentage)

        Returns:
            True if successful
        """
        if self._rc_metrics_store is None:
            if not self.initialize_rc_metrics():
                logger.error("Failed to initialize RC metrics store")
                return False

        # Get DataFrame from geological store
        df = self._geological_store.get_dataframe(source_name)
        if df is None or df.empty:
            logger.warning(f"No data found in source '{source_name}'")
            return False

        return self._rc_metrics_store.compute_from_dataframe(
            df=df,
            source_name=source_name,
            hole_id_col=hole_id_col,
            from_col=from_col,
            to_col=to_col,
            progress_callback=progress_callback
        )

    def get_rc_metrics(self, hole_id: str, depth_to: float) -> Optional[IntervalMetrics]:
        """
        Get pre-computed RC metrics for an interval (O(1) lookup).

        Args:
            hole_id: Hole identifier
            depth_to: End depth

        Returns:
            IntervalMetrics or None if not available
        """
        if self._rc_metrics_store is None or not self._rc_metrics_store.is_loaded:
            return None
        return self._rc_metrics_store.get_metrics(hole_id, depth_to)

    def get_rc_metrics_dict(self, hole_id: str, depth_to: float) -> Dict[str, Any]:
        """
        Get pre-computed RC metrics as a dictionary.

        Args:
            hole_id: Hole identifier
            depth_to: End depth

        Returns:
            Dictionary of metrics, empty dict if not available
        """
        if self._rc_metrics_store is None or not self._rc_metrics_store.is_loaded:
            return {}
        return self._rc_metrics_store.get_metrics_dict(hole_id, depth_to)

    def get_rc_metrics_dataframe(self) -> Optional["pd.DataFrame"]:
        """
        Get all pre-computed RC metrics as a DataFrame.

        Returns:
            DataFrame with all metrics, or None if not computed
        """
        if self._rc_metrics_store is None:
            return None
        return self._rc_metrics_store.get_metrics_dataframe()

    def get_rc_metrics_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics about computed RC metrics.

        Returns:
            Dictionary with stats (empty if not computed)
        """
        if self._rc_metrics_store is None:
            return {}
        return self._rc_metrics_store.get_statistics()

    def clear_rc_metrics(self) -> None:
        """Clear all computed RC metrics."""
        if self._rc_metrics_store is not None:
            self._rc_metrics_store.clear()

    # =========================================================================
    # Single Image Data Retrieval
    # =========================================================================
    
    def get_image_data(
        self, 
        key: ImageKey, 
        include_csv: bool = True,
        csv_columns: Optional[List[str]] = None
    ) -> Optional[CompartmentData]:
        """
        Get complete data for a single image.
        
        Aggregates data from all stores into a CompartmentData object.
        
        Args:
            key: ImageKey to look up
            include_csv: Whether to include CSV geological data
            csv_columns: Optional list of specific CSV columns to include
            
        Returns:
            CompartmentData with all available data, or None if image not found
        """
        # Get image info from index
        image_info = self._image_index.get(key)
        if not image_info:
            logger.debug(f"Image not found in index: {key}")
            return None
        
        # Create CompartmentData with image info
        data = CompartmentData(
            key=key,
            image_path=image_info.path,
            original_path=image_info.original_path,
            filename=image_info.filename,
            hole_id=image_info.hole_id,
            depth_from=image_info.depth_from,
            depth_to=image_info.depth_to,
            moisture_status=image_info.moisture_status,
        )
        
        # Add register data
        review_metadata = self._register_store.get_review_metadata(key)
        data.classification = review_metadata.classification
        data.tags = review_metadata.tags
        data.comments = review_metadata.comments
        data.consensus_classification = review_metadata.consensus_classification
        data.review_count = review_metadata.review_count
        data.agreement = review_metadata.agreement
        
        # Add image properties
        properties = self._register_store.get_image_properties(key)
        data.wet_hex = properties.wet_hex
        data.dry_hex = properties.dry_hex
        data.combined_hex = properties.combined_hex
        
        # Add CSV data
        if include_csv:
            csv_row = self._geological_store.get_row(key)
            if csv_row:
                if csv_columns:
                    # Filter to requested columns
                    data.csv_data = {
                        k: v for k, v in csv_row.items() 
                        if k in csv_columns
                    }
                else:
                    data.csv_data = csv_row
        
        return data
    
    def get_image_data_dict(
        self, 
        key: ImageKey, 
        include_csv: bool = True,
        csv_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get complete data for a single image as a dictionary.
        
        Convenience method for filter evaluation and DataFrame building.
        
        Args:
            key: ImageKey to look up
            include_csv: Whether to include CSV data
            csv_columns: Optional specific columns
            
        Returns:
            Dictionary with all data, empty dict if not found
        """
        data = self.get_image_data(key, include_csv, csv_columns)
        return data.to_dict() if data else {}
    
    # =========================================================================
    # Path Lookups
    # =========================================================================
    
    def get_image_path(self, key: ImageKey) -> Optional[str]:
        """Get the file path for an image."""
        return self._image_index.get_path(key)
    
    def get_original_path(self, key: ImageKey) -> Optional[str]:
        """Get the original (source) image path for a compartment."""
        return self._image_index.get_original_path(key)
    
    # =========================================================================
    # Hole-Based Access
    # =========================================================================
    
    def get_keys_for_hole(self, hole_id: str) -> List[ImageKey]:
        """Get all image keys for a specific hole, sorted by depth."""
        return self._image_index.get_keys_for_hole(hole_id)
    
    def get_images_for_hole(self, hole_id: str) -> List[ImageInfo]:
        """Get all image infos for a specific hole."""
        return self._image_index.get_images_for_hole(hole_id)
    
    def get_data_for_hole(
        self, 
        hole_id: str, 
        include_csv: bool = True
    ) -> List[CompartmentData]:
        """
        Get complete data for all images in a hole.
        
        Args:
            hole_id: Hole identifier
            include_csv: Whether to include CSV data
            
        Returns:
            List of CompartmentData sorted by depth
        """
        keys = self.get_keys_for_hole(hole_id)
        result = []
        
        for key in keys:
            data = self.get_image_data(key, include_csv)
            if data:
                result.append(data)
        
        return result
    
    def get_unique_holes(self) -> Set[str]:
        """Get all unique hole IDs across all data sources."""
        # Combine from all stores
        holes = self._image_index.unique_holes
        holes.update(self._geological_store.get_unique_holes())
        return holes
    
    # =========================================================================
    # Filtering
    # =========================================================================
    
    def get_filtered_keys(
        self,
        hole_ids: Optional[Set[str]] = None,
        classified_only: bool = False,
        unclassified_only: bool = False,
        conflicts_only: bool = False,
        moisture_status: Optional[str] = None,
        classification: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        depth_min: Optional[float] = None,
        depth_max: Optional[float] = None,
    ) -> List[ImageKey]:
        """
        Get image keys matching filter criteria.
        
        This is the primary filtering method. Filters are applied in order
        of efficiency (fastest first).
        
        Args:
            hole_ids: Optional set of holes to include
            classified_only: Only include classified images
            unclassified_only: Only include unclassified images
            conflicts_only: Only include images with classification conflicts
            moisture_status: Filter by "Wet" or "Dry"
            classification: Filter by specific classification
            tags: Filter by having any of these tags
            depth_min: Minimum depth
            depth_max: Maximum depth
            
        Returns:
            List of matching ImageKeys
        """
        # Start with all keys
        if hole_ids:
            # Filter by holes first (most restrictive)
            candidates = []
            for hole_id in hole_ids:
                candidates.extend(self._image_index.get_keys_for_hole(hole_id))
        else:
            candidates = list(self._image_index.keys())
        
        result = []
        
        for key in candidates:
            # Apply filters in order of cheapness
            
            # Depth filter (no lookup needed)
            if depth_min is not None and key.depth_to < depth_min:
                continue
            if depth_max is not None and key.depth_to > depth_max:
                continue
            
            # Moisture filter (no lookup needed)
            if moisture_status and key.moisture_status != moisture_status:
                continue
            
            # Register-based filters (cached lookup)
            if classified_only or unclassified_only or conflicts_only or classification or tags:
                metadata = self._register_store.get_review_metadata(key)
                
                if classified_only and not metadata.classification:
                    continue
                
                if unclassified_only and metadata.classification:
                    continue
                
                if conflicts_only and metadata.agreement != "split":
                    continue
                
                if classification and metadata.classification != classification:
                    continue
                
                if tags and not (metadata.tags & tags):
                    continue
            
            result.append(key)
        
        return result
    
    def count_by_classification(self, hole_id: Optional[str] = None) -> Dict[str, int]:
        """
        Get counts of images by classification.
        
        Args:
            hole_id: Optional hole to filter by
            
        Returns:
            Dictionary of {classification: count}
        """
        if hole_id:
            keys = self.get_keys_for_hole(hole_id)
        else:
            keys = list(self._image_index.keys())
        
        counts: Dict[str, int] = {}
        
        for key in keys:
            metadata = self._register_store.get_review_metadata(key)
            cls = metadata.classification or "unclassified"
            counts[cls] = counts.get(cls, 0) + 1
        
        return counts
    
    # =========================================================================
    # DataFrame Building (for visualization)
    # =========================================================================
    
    def build_dataframe(
        self,
        keys: List[ImageKey],
        columns: Optional[List[str]] = None,
        include_tags: bool = True
    ) -> "pd.DataFrame":
        """
        Build a pandas DataFrame for a set of image keys.
        
        Used for scatterplot visualization and advanced filtering.
        
        Args:
            keys: List of ImageKeys to include
            columns: Optional list of CSV columns to include (None = all)
            include_tags: Whether to include tag columns
            
        Returns:
            DataFrame with one row per image
        """
        import pandas as pd
        
        rows = []
        
        for key in keys:
            data = self.get_image_data(key, include_csv=True, csv_columns=columns)
            if data:
                row = data.to_dict()
                
                # Add tag columns if requested
                if include_tags:
                    for tag_def in self._tag_definitions:
                        tag_col = f"tag_{tag_def.id}"
                        row[tag_col] = tag_def.id in data.tags
                
                rows.append(row)
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        
        logger.debug(f"Built DataFrame with {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def build_dataframe_for_hole(
        self,
        hole_id: str,
        columns: Optional[List[str]] = None
    ) -> "pd.DataFrame":
        """
        Build a DataFrame for all images in a hole.
        
        Args:
            hole_id: Hole identifier
            columns: Optional specific columns
            
        Returns:
            DataFrame for the hole
        """
        keys = self.get_keys_for_hole(hole_id)
        return self.build_dataframe(keys, columns)
    
    # =========================================================================
    # CSV Column Information
    # =========================================================================
    
    def get_available_columns(self) -> List[Tuple[str, DataType]]:
        """
        Get list of available columns with their data types.

        Includes:
        - CSV columns from all geological sources
        - Computed RC metrics (if available)

        Returns:
            List of (column_name, DataType) tuples
        """
        all_columns = self._geological_store.get_available_columns()

        result = []
        seen = set()

        # Add geological columns
        for source_name, columns in all_columns.items():
            for col_name, col_type in columns:
                if col_name not in seen:
                    result.append((col_name, col_type))
                    seen.add(col_name)

        # Add RC metric columns if computed
        if self.has_rc_metrics:
            rc_columns = [
                ("weighted_hardness", DataType.NUMERIC),
                ("total_gangue_pct", DataType.NUMERIC),
                ("si_gangue_pct", DataType.NUMERIC),
                ("al_gangue_pct", DataType.NUMERIC),
                ("carbonate_gangue_pct", DataType.NUMERIC),
                ("zonation_pr_pct", DataType.NUMERIC),
                ("zonation_hy_pct", DataType.NUMERIC),
                ("zonation_de_pct", DataType.NUMERIC),
                ("zonation_un_pct", DataType.NUMERIC),
                ("quartz_pct", DataType.NUMERIC),
                ("chert_pct", DataType.NUMERIC),
            ]
            for col_name, col_type in rc_columns:
                if col_name not in seen:
                    result.append((col_name, col_type))
                    seen.add(col_name)

            logger.debug(f"Added {len(rc_columns)} RC metric columns to available columns")

        return result
    
    def get_column_values(self, column: str) -> List[Any]:
        """
        Get unique values for a column (for dropdown population).
        
        Args:
            column: Column name
            
        Returns:
            List of unique values
        """
        return self._geological_store.get_column_values(column)
    
    # =========================================================================
    # Cache Management
    # =========================================================================
    
    def refresh_image(self, key: ImageKey):
        """
        Refresh cached data for a single image after edits.
        
        Call this after classification/tag changes.
        
        Args:
            key: ImageKey to refresh
        """
        self._register_store.refresh_key(key)
        logger.debug(f"Refreshed data for {key}")
    
    def refresh_hole(self, hole_id: str):
        """
        Refresh cached data for all images in a hole.
        
        Args:
            hole_id: Hole identifier
        """
        for key in self.get_keys_for_hole(hole_id):
            self._register_store.refresh_key(key)
        logger.debug(f"Refreshed data for hole {hole_id}")
    
    def invalidate_caches(self):
        """Invalidate all caches, forcing fresh lookups."""
        self._register_store.invalidate_cache()
        logger.info("All caches invalidated")
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_collar_data(self) -> "pd.DataFrame":
        """
        Get collar data from geological store with standardized column names.
        
        Looks for 'excollar', 'collar', or 'collars' data source and applies
        column name standardization so consumers get consistent column names.
        
        Column Standardization (source → standard):
            east, easting → x
            north, northing → y  
            rl, elevation → z
            holeid, hole_id → holeid
            plannedholeid → planned_holeid
            projectcode → project
        
        Returns:
            DataFrame with standardized lowercase column names, or empty DataFrame
        """
        import pandas as pd
        
        if not self._geological_store:
            logger.warning("[COLLAR] No geological_store available")
            return pd.DataFrame()
        
        available_sources = self._geological_store.list_sources()
        logger.debug(f"[COLLAR] get_collar_data searching in: {available_sources}")
        
        for source_name in ['excollar', 'collar', 'collars']:
            if source_name in available_sources:
                source = self._geological_store.get_source(source_name)
                if source and source.df is not None:
                    collar_df = source.df.copy()
                    
                    logger.debug(f"[COLLAR] Original columns: {list(collar_df.columns)}")
                    
                    # Define column mappings: {standard_name: [possible_source_names]}
                    # Source names are already lowercase from geological_store
                    column_mappings = {
                        'x': ['east', 'easting', 'x', 'collar_east', 'longitude'],
                        'y': ['north', 'northing', 'y', 'collar_north', 'latitude'],
                        'z': ['rl', 'elevation', 'z', 'collar_rl', 'z_coord', 'elev'],
                        'holeid': ['holeid', 'hole_id', 'bhid', 'drillhole', 'dhid', 'hole'],
                        'planned_holeid': ['plannedholeid', 'planned_holeid', 'planned_hole_id'],
                        'project': ['projectcode', 'project', 'project_code'],
                    }
                    
                    # Build rename map
                    rename_map = {}
                    for standard_name, source_names in column_mappings.items():
                        for src_name in source_names:
                            if src_name in collar_df.columns and src_name != standard_name:
                                rename_map[src_name] = standard_name
                                logger.debug(f"[COLLAR] Mapping '{src_name}' → '{standard_name}'")
                                break
                    
                    if rename_map:
                        collar_df = collar_df.rename(columns=rename_map)
                    
                    logger.info(f"[COLLAR] Returning {len(collar_df)} rows from '{source_name}'")
                    logger.debug(f"[COLLAR] Standardized columns: {list(collar_df.columns)}")
                    return collar_df
        
        logger.warning(f"[COLLAR] No collar source found in: {available_sources}")
        return pd.DataFrame()
    
    def get_survey_data(self) -> "pd.DataFrame":
        """
        Get survey data from geological store with standardized column names.
        
        Looks for 'exsurvey', 'survey', or 'surveys' data source and applies
        column name standardization so consumers get holeid, depth, azimuth, dip.
        
        Returns only rows with valid numeric depth, azimuth, and dip.
        Duplicate (holeid, depth) rows are dropped (first kept).
        
        Returns:
            DataFrame with standardized columns: holeid, depth, azimuth, dip;
            or empty DataFrame if no survey source found.
        """
        import pandas as pd
        
        if not self._geological_store:
            logger.warning("[SURVEY] No geological_store available")
            return pd.DataFrame()
        
        available_sources = self._geological_store.list_sources()
        logger.debug(f"[SURVEY] get_survey_data searching in: {available_sources}")
        
        for source_name in ['exsurvey', 'survey', 'surveys']:
            if source_name in available_sources:
                source = self._geological_store.get_source(source_name)
                if source and source.df is not None:
                    survey_df = source.df.copy()
                    logger.debug(f"[SURVEY] Original columns: {list(survey_df.columns)}")
                    
                    resolver = ColumnResolver(survey_df)
                    hole_col = resolver.get("hole_id")
                    depth_col = resolver.get("survey_depth")
                    azi_col = resolver.get("azimuth")
                    dip_col = resolver.get("dip")
                    
                    if not hole_col or not depth_col:
                        logger.warning(
                            f"[SURVEY] Source '{source_name}' missing hole_id or depth column, skipping"
                        )
                        continue
                    if not azi_col or not dip_col:
                        logger.warning(
                            f"[SURVEY] Source '{source_name}' missing azimuth or dip column, skipping"
                        )
                        continue
                    
                    # Build standardized DataFrame
                    out = pd.DataFrame()
                    out["holeid"] = survey_df[hole_col].astype(str).str.strip().str.upper()
                    out["depth"] = pd.to_numeric(survey_df[depth_col], errors="coerce")
                    out["azimuth"] = pd.to_numeric(survey_df[azi_col], errors="coerce")
                    out["dip"] = pd.to_numeric(survey_df[dip_col], errors="coerce")
                    
                    # Drop rows with missing required values
                    out = out.dropna(subset=["depth", "azimuth", "dip"])
                    # Drop negative depths
                    out = out[out["depth"] >= 0]
                    # Coerce azimuth to [0, 360) and dip to [-90, 0] (or allow 0–90 for inclination)
                    out["azimuth"] = out["azimuth"] % 360
                    # Dip: typically negative down; clamp to [-90, 90]
                    out.loc[out["dip"] > 90, "dip"] = 90
                    out.loc[out["dip"] < -90, "dip"] = -90
                    
                    # Deduplicate by (holeid, depth), keep first
                    out = out.drop_duplicates(subset=["holeid", "depth"], keep="first")
                    out = out.sort_values(["holeid", "depth"]).reset_index(drop=True)
                    
                    logger.info(
                        f"[SURVEY] Returning {len(out)} rows from '{source_name}'"
                    )
                    return out
        
        logger.debug(f"[SURVEY] No survey source found in: {available_sources}")
        return pd.DataFrame()
    
    def get_trace_for_hole(
        self,
        hole_id: str,
        max_depth: Optional[float] = None,
    ) -> List[Tuple[float, float, float, float]]:
        """
        Get 3D trace (depth, x, y, z) for a hole from collar + survey.

        Uses collar for (x, y, z) at depth 0 and exsurvey (or survey/surveys)
        for azimuth/dip at depth. If no survey for the hole, falls back to
        vertical hole (dip -90°) from collar. If no collar for the hole,
        returns empty list.

        Args:
            hole_id: Hole identifier (case-insensitive).
            max_depth: Optional max depth for vertical fallback (second point).

        Returns:
            List of (depth, x, y, z) sorted by depth; empty if no collar.
        """
        import pandas as pd
        
        hole_upper = str(hole_id).strip().upper()
        collar_df = self.get_collar_data()
        if collar_df.empty or "holeid" not in collar_df.columns:
            return []
        collar_match = collar_df[
            collar_df["holeid"].astype(str).str.strip().str.upper() == hole_upper
        ]
        if collar_match.empty:
            return []
        row = collar_match.iloc[0]
        try:
            x0 = float(row.get("x", float("nan")))
            y0 = float(row.get("y", float("nan")))
            z0 = float(row.get("z", float("nan")))
        except (TypeError, ValueError):
            logger.warning(f"[TRACE] Hole {hole_id} has non-numeric collar x/y/z, cannot build trace")
            return []
        if pd.isna(x0) or pd.isna(y0) or pd.isna(z0):
            logger.warning(f"[TRACE] Hole {hole_id} has missing collar x/y/z, cannot build trace")
            return []
        
        survey_df = self.get_survey_data()
        survey_rows: List[Tuple[float, float, float]] = []
        if not survey_df.empty and "holeid" in survey_df.columns:
            hole_survey = survey_df[
                survey_df["holeid"].astype(str).str.strip().str.upper() == hole_upper
            ]
            for _, r in hole_survey.iterrows():
                depth = r.get("depth")
                azi = r.get("azimuth")
                dip = r.get("dip")
                if pd.isna(depth) or pd.isna(azi) or pd.isna(dip):
                    continue
                try:
                    survey_rows.append((float(depth), float(azi), float(dip)))
                except (TypeError, ValueError):
                    continue
        
        if not survey_rows and max_depth is None:
            logger.debug(f"[TRACE] No survey for hole {hole_id}, using vertical (collar only)")
        
        trace = build_trace(
            x0, y0, z0,
            survey_rows,
            vertical_if_missing=True,
            max_depth=max_depth,
        )
        return trace
    
    def get_xyz_at_depth(
        self,
        hole_id: str,
        depth: float,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Get (x, y, z) at a given depth along a hole.

        Builds trace from collar + survey (or vertical fallback), then
        interpolates at the requested depth. No extrapolation beyond
        trace depth range; returns None if depth is outside [0, max_trace_depth]
        or if no collar for the hole.

        Args:
            hole_id: Hole identifier (case-insensitive).
            depth: Depth along hole (m).

        Returns:
            (x, y, z) or None if out of range or no collar.
        """
        trace = self.get_trace_for_hole(hole_id, max_depth=None)
        if not trace:
            return None
        return trace_xyz_at_depth(trace, depth)
    
    def add_xyz_to_interval_dataframe(
        self,
        interval_df: "pd.DataFrame",
        hole_col: Optional[str] = None,
        from_col: Optional[str] = None,
        to_col: Optional[str] = None,
        inplace: bool = False,
    ) -> "pd.DataFrame":
        """
        Add XYZ columns to an interval DataFrame at from/to depths.

        Resolves hole_id and depth_from/depth_to columns (via ColumnResolver
        if not provided), then calls get_xyz_at_depth for each row and adds
        x_from, y_from, z_from, x_to, y_to, z_to. Rows without valid XYZ get NaN.

        Args:
            interval_df: DataFrame with hole id and from/to depth columns.
            hole_col: Column name for hole ID (None = resolve via ColumnResolver).
            from_col: Column name for interval start depth (None = resolve).
            to_col: Column name for interval end depth (None = resolve).
            inplace: If True, modify interval_df in place; else return a copy.

        Returns:
            DataFrame with added columns x_from, y_from, z_from, x_to, y_to, z_to.
        """
        import pandas as pd
        
        df = interval_df if inplace else interval_df.copy()
        resolver = ColumnResolver(df)
        hole_col = hole_col or resolver.get("hole_id")
        from_col = from_col or resolver.get("depth_from")
        to_col = to_col or resolver.get("depth_to")
        if not hole_col or not from_col or not to_col:
            logger.warning(
                "[XYZ] add_xyz_to_interval_dataframe: missing hole_id, from, or to column"
            )
            return df
        
        x_from_list: List[Optional[float]] = []
        y_from_list: List[Optional[float]] = []
        z_from_list: List[Optional[float]] = []
        x_to_list: List[Optional[float]] = []
        y_to_list: List[Optional[float]] = []
        z_to_list: List[Optional[float]] = []
        
        for _, row in df.iterrows():
            hole_id = str(row.get(hole_col, "")).strip()
            try:
                d_from = float(row.get(from_col))
                d_to = float(row.get(to_col))
            except (TypeError, ValueError):
                d_from = d_to = float("nan")
            xyz_from = self.get_xyz_at_depth(hole_id, d_from) if not pd.isna(d_from) else None
            xyz_to = self.get_xyz_at_depth(hole_id, d_to) if not pd.isna(d_to) else None
            if xyz_from:
                x_from_list.append(xyz_from[0])
                y_from_list.append(xyz_from[1])
                z_from_list.append(xyz_from[2])
            else:
                x_from_list.append(None)
                y_from_list.append(None)
                z_from_list.append(None)
            if xyz_to:
                x_to_list.append(xyz_to[0])
                y_to_list.append(xyz_to[1])
                z_to_list.append(xyz_to[2])
            else:
                x_to_list.append(None)
                y_to_list.append(None)
                z_to_list.append(None)
        
        df["x_from"] = x_from_list
        df["y_from"] = y_from_list
        df["z_from"] = z_from_list
        df["x_to"] = x_to_list
        df["y_to"] = y_to_list
        df["z_to"] = z_to_list
        return df
    
    def get_hole_intervals(
        self, 
        hole_id: str, 
        source_names: Optional[List[str]] = None
    ) -> "pd.DataFrame":
        """
        Get interval data for a hole directly from geological store.
        
        This is optimized for correlation display - queries CSV data directly
        without going through image keys. Much faster than build_dataframe_for_hole().
        
        Args:
            hole_id: Hole identifier (case-insensitive)
            source_names: Optional list of specific sources to query.
                         If None, queries interval sources: exgeologyRC, exgeologyDiamond, 
                         exassay, drillhole_data, exstrat
        
        Returns:
            DataFrame with interval data (from, to, and data columns), or empty DataFrame
        """
        import pandas as pd
        
        if not self._geological_store:
            logger.warning(f"[INTERVALS] No geological_store available")
            return pd.DataFrame()
        
        # Default interval sources (prioritized)
        if source_names is None:
            source_names = [
                'exgeologyrc', 'exgeologydiamond', 'exassay', 
                'drillhole_data', 'exstrat'
            ]
        
        # Get all data for this hole from geological store
        all_data = self._geological_store.get_rows_for_hole(hole_id)
        
        if not all_data:
            logger.debug(f"[INTERVALS] No data found for hole {hole_id}")
            return pd.DataFrame()
        
        logger.debug(f"[INTERVALS] Found data in sources: {list(all_data.keys())}")
        
        # Find the best interval source (first match from prioritized list)
        result_df = None
        used_source = None
        
        for src in source_names:
            src_lower = src.lower()
            for available_src, df in all_data.items():
                if available_src.lower() == src_lower and not df.empty:
                    result_df = df.copy()
                    used_source = available_src
                    break
            if result_df is not None:
                break
        
        # Fallback: use any available interval source
        if result_df is None:
            for src_name, df in all_data.items():
                # Check if it's an interval source (has from/to columns)
                cols_lower = [c.lower() for c in df.columns]
                has_from = any(c in cols_lower for c in ['from', 'sampfrom', 'depth_from', 'geolfrom'])
                has_to = any(c in cols_lower for c in ['to', 'sampto', 'depth_to', 'geolto'])
                
                if has_from and has_to and not df.empty:
                    result_df = df.copy()
                    used_source = src_name
                    break
        
        if result_df is None:
            logger.debug(f"[INTERVALS] No interval data found for hole {hole_id}")
            return pd.DataFrame()
        
        logger.info(f"[INTERVALS] Returning {len(result_df)} rows for {hole_id} from '{used_source}'")
        return result_df
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the data coordinator."""
        stats = {
            "is_initialized": self._is_initialized,
            "initialization_time": self._initialization_time,
            "image_index": self._image_index.get_stats(),
            "geological_store": self._geological_store.get_stats(),
            "register_store": self._register_store.get_stats(),
        }

        return stats
    
    def log_stats(self):
        """Log statistics for debugging."""
        stats = self.get_stats()
        
        logger.info("=" * 50)
        logger.info("DATA COORDINATOR STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Initialized: {stats['is_initialized']}")
        logger.info(f"Init time: {stats['initialization_time']:.2f}s")
        logger.info(f"Images: {stats['image_index']['images_indexed']:,}")
        logger.info(f"Holes: {stats['image_index']['unique_holes']:,}")
        logger.info(f"CSV rows: {stats['geological_store']['total_rows']:,}")
        logger.info(f"Cached reviews: {stats['register_store']['cached_images']:,}")
        logger.info("=" * 50)

    # =========================================================================
    # Data Joining
    # =========================================================================

    def join_sources(
        self,
        primary: str,
        secondary: str,
        aggregate: bool = False,
        primary_key_cols: Optional[Dict[str, str]] = None,
        secondary_key_cols: Optional[Dict[str, str]] = None,
    ) -> Optional["JoinResult"]:
        """
        Join two data sources using interval overlap matching.

        Uses LEFT JOIN semantics - all primary rows preserved,
        secondary matched by hole_id and depth interval overlap.

        Args:
            primary: Name of primary data source (all rows kept)
            secondary: Name of secondary data source (matched by overlap)
            aggregate: If True, aggregate secondary values per primary row
            primary_key_cols: Override key column detection for primary
            secondary_key_cols: Override key column detection for secondary

        Returns:
            JoinResult with joined DataFrame and match metadata, or None if sources not found
        """
        from processing.DataManager.data_joiner import DataJoiner

        # Get source DataFrames
        primary_source = self._geological_store.get_source(primary)
        secondary_source = self._geological_store.get_source(secondary)

        if not primary_source or primary_source.df is None:
            logger.error(f"Primary source '{primary}' not found or empty")
            return None

        if not secondary_source or secondary_source.df is None:
            logger.error(f"Secondary source '{secondary}' not found or empty")
            return None

        # Auto-detect key columns if not provided
        if primary_key_cols is None:
            primary_key_cols = self._detect_interval_columns(primary_source)

        if secondary_key_cols is None:
            secondary_key_cols = self._detect_interval_columns(secondary_source)

        logger.info(f"Joining {primary} <- {secondary}")
        logger.debug(f"  Primary keys: {primary_key_cols}")
        logger.debug(f"  Secondary keys: {secondary_key_cols}")

        joiner = DataJoiner()
        result = joiner.join(
            primary_df=primary_source.df,
            secondary_df=secondary_source.df,
            primary_key_cols=primary_key_cols,
            secondary_key_cols=secondary_key_cols,
            aggregate_secondary=aggregate,
            primary_source=primary,
            secondary_source=secondary,
        )

        logger.info(result.summary())
        return result

    def _detect_interval_columns(self, source: "IndexedDataSource") -> Dict[str, str]:
        """
        Auto-detect hole_id, from, and to columns for a data source.

        Args:
            source: IndexedDataSource to inspect

        Returns:
            Dict with 'hole', 'from', 'to' keys mapped to column names
        """
        schema = source.schema
        cols = {c.lower() for c in source.df.columns}

        # Hole ID
        hole_col = schema.hole_id_column
        if not hole_col:
            for candidate in ["holeid", "hole_id", "bhid"]:
                if candidate in cols:
                    hole_col = candidate
                    break

        # From depth
        from_col = schema.from_column
        if not from_col:
            for candidate in ["geolfrom", "sampfrom", "from", "depth_from"]:
                if candidate in cols:
                    from_col = candidate
                    break

        # To depth
        to_col = schema.to_column
        if not to_col:
            for candidate in ["geolto", "sampto", "to", "depth_to"]:
                if candidate in cols:
                    to_col = candidate
                    break

        if not all([hole_col, from_col, to_col]):
            raise ValueError(
                f"Could not detect interval columns for {source.name}. "
                f"Found: hole={hole_col}, from={from_col}, to={to_col}"
            )

        return {"hole": hole_col, "from": from_col, "to": to_col}
