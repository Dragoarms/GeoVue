"""
filter_pipeline.py - Unified filtering and data population for LoggingReviewDialog.

This module consolidates all filtering logic into a clean, debuggable pipeline:

1. FilterPipeline: Orchestrates all filter operations in a single clear flow
   - Display mode filtering (hole-by-hole, all images, all intervals)
   - Dynamic filter rows (CSV columns, register columns)
   - Classification visibility (all, classified, unclassified, conflicts)
   - Scatter plot selections
   - Sorting (primary and secondary)

2. ImageDataPopulator: Ensures images have the data they need
   - Populates csv_data from GeologicalStore
   - Populates review metadata from RegisterStore
   - Handles lazy loading efficiently

3. DebugCheckpoint: Logging utility for pipeline debugging

Architecture:
    all_images → FilterPipeline.execute() → filtered_images → grid_canvas.load_images()
    
The pipeline is stateless - it takes inputs and returns outputs without side effects.
State is managed by the caller (LoggingReviewDialog).

Author: George Symonds
"""

import logging
import time
from typing import List, Dict, Set, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

# Type hints for external types (avoid circular imports)
# These will be imported from actual modules at runtime
from processing.DataManager.keys import ImageKey

logger = logging.getLogger(__name__)


# =============================================================================
# DEBUG CHECKPOINT - For pipeline debugging
# =============================================================================

class DebugCheckpoint:
    """
    Logging utility for tracking state at key pipeline stages.
    
    Usage:
        DebugCheckpoint.log("After CSV filter", images=filtered, keys=matching_keys)
        DebugCheckpoint.breakpoint("Before sorting")  # Drops into debugger
    """
    
    _enabled = True
    _verbose = False
    
    @classmethod
    def enable(cls, verbose: bool = False):
        """Enable checkpoint logging."""
        cls._enabled = True
        cls._verbose = verbose
        logger.info("DebugCheckpoint enabled (verbose=%s)", verbose)
    
    @classmethod
    def disable(cls):
        """Disable checkpoint logging."""
        cls._enabled = False
    
    @classmethod
    def log(cls, name: str, **kwargs):
        """
        Log state at a checkpoint.
        
        Args:
            name: Checkpoint name (e.g., "After CSV filter")
            **kwargs: Variables to log with their values
            
        Example:
            DebugCheckpoint.log("After hole filter", 
                               images=filtered_images, 
                               keys=matching_keys,
                               mode=display_mode)
        """
        if not cls._enabled:
            return
        
        logger.debug("=" * 60)
        logger.debug(f"CHECKPOINT: {name}")
        logger.debug("-" * 60)
        
        for key, value in kwargs.items():
            if isinstance(value, (list, set)):
                # Log collection summary
                logger.debug(f"  {key}: {type(value).__name__} with {len(value)} items")
                if cls._verbose and len(value) <= 5:
                    for i, item in enumerate(value):
                        logger.debug(f"    [{i}]: {item}")
                elif cls._verbose and len(value) > 5:
                    for i, item in enumerate(list(value)[:3]):
                        logger.debug(f"    [{i}]: {item}")
                    logger.debug(f"    ... and {len(value) - 3} more")
            elif isinstance(value, dict):
                logger.debug(f"  {key}: dict with {len(value)} keys")
                if cls._verbose and len(value) <= 5:
                    for k, v in value.items():
                        logger.debug(f"    {k}: {v}")
            elif isinstance(value, pd.DataFrame):
                logger.debug(f"  {key}: DataFrame {value.shape[0]} rows x {value.shape[1]} cols")
            elif hasattr(value, '__len__') and not isinstance(value, str):
                logger.debug(f"  {key}: {type(value).__name__} with {len(value)} items")
            else:
                logger.debug(f"  {key}: {value}")
        
        logger.debug("=" * 60)
    
    @classmethod
    def breakpoint(cls, name: str = "Debug breakpoint"):
        """
        Drop into Python debugger at this point.
        
        Args:
            name: Description of where we're stopping
            
        Usage:
            DebugCheckpoint.breakpoint("Before applying sort")
            # When hit, use pdb commands:
            # p variable - print variable
            # locals() - see all local variables
            # n - next line
            # c - continue
        """
        if cls._enabled:
            logger.info(f"BREAKPOINT: {name}")
            logger.info("Entering debugger. Use 'c' to continue, 'q' to quit.")
            breakpoint()


# =============================================================================
# FILTER CRITERIA - Input specification for filtering
# =============================================================================

class DisplayMode(Enum):
    """Display mode options for image grid."""
    HOLE_BY_HOLE = "hole_by_hole"
    ALL_IMAGES = "all_images"
    ALL_INTERVALS = "all_intervals"


class ClassificationVisibility(Enum):
    """Which images to show based on classification status."""
    ALL = "all"
    CLASSIFIED = "classified"
    UNCLASSIFIED = "unclassified"
    CONFLICTS = "disagreements"


@dataclass
class FilterCriteria:
    """
    Complete specification of what to filter.
    
    This dataclass captures ALL filter state in one place,
    making it easy to understand what filters are active.
    
    Attributes:
        display_mode: How to group/display images
        current_hole: Hole ID when in hole-by-hole mode
        unique_holes: All available holes (for hole-by-hole navigation)
        
        classification_visibility: Show all/classified/unclassified/conflicts
        show_others_reviews: Include other users' classifications
        
        dynamic_filters: List of filter configurations from UI
        scatter_selection: Set of (hole_id, depth_to) from scatter lasso
        
        sort_column: Primary sort column
        sort_order: Primary sort order ("asc" or "desc")
        secondary_sort_column: Optional secondary sort
        secondary_sort_order: Secondary sort order
        
        add_placeholders: Whether to add placeholder images for missing intervals
    """
    # Display mode
    display_mode: DisplayMode = DisplayMode.ALL_IMAGES
    current_hole: Optional[str] = None
    current_hole_index: int = 0
    unique_holes: List[str] = field(default_factory=list)
    
    # Classification visibility
    classification_visibility: ClassificationVisibility = ClassificationVisibility.ALL
    show_others_reviews: bool = False
    
    # Dynamic filters from UI
    dynamic_filters: List[Dict[str, Any]] = field(default_factory=list)
    
    # Scatter selection (lasso)
    scatter_selection: Optional[Set[Tuple[str, float]]] = None
    
    # Sorting
    sort_column: str = "depth_to"
    sort_order: str = "asc"
    secondary_sort_column: Optional[str] = None
    secondary_sort_order: str = "asc"
    
    # Interval placeholders
    add_placeholders: bool = False
    
    def get_hash(self) -> str:
        """
        Generate a hash representing current filter state.
        
        Used to detect if filters have changed and re-filtering is needed.
        """
        parts = [
            f"mode:{self.display_mode.value}",
            f"hole:{self.current_hole or 'all'}",
            f"vis:{self.classification_visibility.value}",
            f"others:{self.show_others_reviews}",
            f"sort:{self.sort_column}:{self.sort_order}",
            f"sort2:{self.secondary_sort_column}:{self.secondary_sort_order}",
        ]
        
        # Add dynamic filter configs
        for i, f in enumerate(self.dynamic_filters):
            if f.get("column") and f.get("operator"):
                parts.append(f"f{i}:{f['column']}_{f['operator']}_{f.get('value')}")
        
        # Add scatter selection state
        if self.scatter_selection:
            parts.append(f"scatter:{len(self.scatter_selection)}")
        
        return "|".join(parts)
    
    def get_descriptions(self) -> List[str]:
        """
        Get human-readable descriptions of active filters.
        
        Returns:
            List of filter descriptions for UI display
        """
        descriptions = []
        
        # Add mode filter
        if self.classification_visibility != ClassificationVisibility.ALL:
            mode_labels = {
                ClassificationVisibility.CLASSIFIED: "Classified Only",
                ClassificationVisibility.UNCLASSIFIED: "Unclassified Only",
                ClassificationVisibility.CONFLICTS: "Conflicts Only",
            }
            desc = f"Show: {mode_labels.get(self.classification_visibility, 'Unknown')}"
            if self.show_others_reviews:
                desc += " (including others)"
            descriptions.append(desc)
        
        # Add consensus indicator
        if self.show_others_reviews:
            descriptions.append("Using: Consensus classifications")
        
        # Add dynamic filters
        for f in self.dynamic_filters:
            if f.get("column") and f.get("operator") and f.get("value") is not None:
                desc = f"{f['column']} {f['operator']} {f['value']}"
                if f.get("value2"):
                    desc += f" and {f['value2']}"
                descriptions.append(desc)
        
        # Add scatter selection
        if self.scatter_selection:
            descriptions.append(f"Scatter selection: {len(self.scatter_selection)} points")
        
        return descriptions


# =============================================================================
# FILTER RESULT - Output from filtering
# =============================================================================

@dataclass
class FilterResult:
    """
    Result of filtering operation.
    
    Contains filtered images plus metadata about the filtering.
    
    Attributes:
        images: The filtered list of CompartmentImage objects
        total_before: Total images before filtering
        total_after: Total images after filtering
        csv_matches: Number of images with CSV data
        filter_descriptions: Human-readable filter descriptions
        execution_time_ms: How long filtering took
    """
    images: List[Any]  # List[CompartmentImage]
    total_before: int = 0
    total_after: int = 0
    csv_matches: int = 0
    filter_descriptions: List[str] = field(default_factory=list)
    execution_time_ms: float = 0
    
    @property
    def was_filtered(self) -> bool:
        """Whether any filtering was applied."""
        return self.total_before != self.total_after


# =============================================================================
# IMAGE DATA POPULATOR - Ensures images have required data
# =============================================================================

class ImageDataPopulator:
    """
    Populates CompartmentImage objects with data from stores.
    
    Handles:
    - CSV data from GeologicalStore
    - Review metadata from RegisterStore
    - Hex colors from hex color cache or RegisterStore
    
    This class ensures images have the data they need for:
    - Filtering (csv_data for CSV column filters)
    - Display (visualizations need csv_data)
    - Sorting (csv columns, review metadata)
    
    Usage:
        populator = ImageDataPopulator(data_coordinator)
        populator.populate_csv_data(images)
        populator.populate_review_metadata(images)
    """
    
    def __init__(self, data_coordinator=None, hex_color_cache: Optional[Dict] = None):
        """
        Initialize the populator.
        
        Args:
            data_coordinator: DataCoordinator with geological_store and register_store
            hex_color_cache: Optional pre-loaded hex color cache {(hole, from, to): hex}
        """
        self._coordinator = data_coordinator
        self._hex_cache = hex_color_cache or {}
        
        # Statistics
        self._last_populate_count = 0
        self._last_populate_time = 0.0
        
        logger.debug("ImageDataPopulator initialized (coordinator=%s, hex_cache=%d)",
                    data_coordinator is not None, len(self._hex_cache))
    
    @property
    def has_geological_store(self) -> bool:
        """Whether geological data is available."""
        if not self._coordinator:
            return False
        if not hasattr(self._coordinator, 'geological_store'):
            return False
        return self._coordinator.geological_store.is_loaded
    
    @property
    def has_register_store(self) -> bool:
        """Whether register data is available."""
        if not self._coordinator:
            return False
        if not hasattr(self._coordinator, 'register_store'):
            return False
        return self._coordinator.register_store is not None
    
    def populate_csv_data(
        self,
        images: List[Any],
        columns: Optional[List[str]] = None,
        force: bool = False
    ) -> int:
        """
        Populate csv_data attribute on images from GeologicalStore.
        
        Args:
            images: List of CompartmentImage objects
            columns: Optional list of specific columns to populate
                     If None, populates all available columns
            force: If True, repopulate even if already has data
            
        Returns:
            Number of images that received CSV data
        """
        if not self.has_geological_store:
            logger.debug("No GeologicalStore available - skipping CSV population")
            return 0
        
        start_time = time.time()
        geo_store = self._coordinator.geological_store
        populated_count = 0
        
        # DEBUG: Log available columns in store
        available_cols = geo_store.get_columns() if hasattr(geo_store, 'get_columns') else []
        logger.info(f"[POPULATE_DEBUG] Requested columns: {columns}")
        logger.info(f"[POPULATE_DEBUG] Available columns in store (first 20): {available_cols[:20]}")
        
        # DEBUG: Check column name matching
        if columns:
            columns_lower = [c.lower() for c in columns]
            matching = [c for c in available_cols if c.lower() in columns_lower]
            logger.info(f"[POPULATE_DEBUG] Matching columns: {matching}")
        
        DebugCheckpoint.log("populate_csv_data START",
                           image_count=len(images),
                           columns=columns,
                           force=force)
        
        skipped_already_has_data = 0
        no_row_data = 0
        column_not_found = 0
        first_row_logged = False
        
        for img in images:
            # Skip if already populated (unless forcing)
            if not force and img.csv_data:
                skipped_already_has_data += 1
                continue
            
            try:
                key = ImageKey(hole_id=img.hole_id, depth_to=float(img.depth_to))
                row_data = geo_store.get_row(key)
                
                if row_data:
                    # DEBUG: Log first row to see actual column names
                    if not first_row_logged:
                        logger.info(f"[POPULATE_DEBUG] First row keys: {list(row_data.keys())[:15]}")
                        if columns:
                            for col in columns:
                                found = col in row_data or col.lower() in [k.lower() for k in row_data.keys()]
                                logger.info(f"[POPULATE_DEBUG] Column '{col}' in row_data: {found}")
                        first_row_logged = True
                    
                    if columns:
                        # Strip source suffix (e.g., "fe_pct (exassay)" -> "fe_pct") for matching
                        import re
                        def strip_suffix(col: str) -> str:
                            return re.sub(r'\s*\([^)]+\)\s*$', '', col).lower()
                        
                        # Build lookup: stripped_name -> [original_display_names]
                        stripped_to_display = {}
                        for c in columns:
                            stripped = strip_suffix(c)
                            if stripped not in stripped_to_display:
                                stripped_to_display[stripped] = []
                            stripped_to_display[stripped].append(c)
                        
                        # Match row_data keys and store under display names
                        img.csv_data = {}
                        for k, v in row_data.items():
                            k_stripped = k.lower()
                            if k_stripped in stripped_to_display:
                                # Store under ALL matching display names
                                for display_name in stripped_to_display[k_stripped]:
                                    img.csv_data[display_name] = v
                                # Also store under raw name for good measure
                                img.csv_data[k] = v
                        
                        # DEBUG: Check if we got the columns we wanted
                        if not img.csv_data:
                            column_not_found += 1
                    else:
                        # All columns - store under both raw names AND display names (with source suffix)
                        img.csv_data = dict(row_data)  # Copy raw data first
                        
                        # Also store under display names for sorting/filtering compatibility
                        # We need to infer the source name from the data coordinator
                        if self._coordinator and hasattr(self._coordinator, 'geological_store'):
                            available = self._coordinator.geological_store.get_available_columns()
                            for source_name, cols in available.items():
                                col_names = [c[0] for c in cols]  # Extract just column names
                                for k, v in row_data.items():
                                    if k in col_names or k.lower() in [c.lower() for c in col_names]:
                                        # Add with source suffix as display name
                                        display_name = f"{k} ({source_name})"
                                        img.csv_data[display_name] = v
                    img.in_csv = True
                    if img.csv_data:  # Only count if we actually got data
                        populated_count += 1
                else:
                    no_row_data += 1
                    img.in_csv = False
                    if not img.csv_data:
                        img.csv_data = {}
                        
            except Exception as e:
                logger.debug(f"CSV lookup failed for {img.hole_id}_{img.depth_to}: {e}")
                img.in_csv = False
                if not img.csv_data:
                    img.csv_data = {}
        
        self._last_populate_count = populated_count
        self._last_populate_time = time.time() - start_time
        
        logger.info(f"[POPULATE_DEBUG] Stats: populated={populated_count}, skipped_has_data={skipped_already_has_data}, no_row={no_row_data}, col_not_found={column_not_found}")
        
        DebugCheckpoint.log("populate_csv_data COMPLETE",
                           populated=populated_count,
                           total=len(images),
                           time_ms=self._last_populate_time * 1000)
        
        logger.info(f"Populated CSV data: {populated_count}/{len(images)} images in {self._last_populate_time:.2f}s")
        
        return populated_count
    
    def populate_review_metadata(
        self,
        images: List[Any],
        include_others: bool = False
    ) -> int:
        """
        Populate review metadata (classification, tags, etc.) from RegisterStore.
        
        This is typically done during image scanning, but can be refreshed here.
        
        Args:
            images: List of CompartmentImage objects
            include_others: Whether to include other users' reviews
            
        Returns:
            Number of images that received review data
        """
        if not self.has_register_store:
            logger.debug("No RegisterStore available - skipping review population")
            return 0
        
        start_time = time.time()
        register_store = self._coordinator.register_store
        populated_count = 0
        
        for img in images:
            try:
                cache_key = (img.hole_id.upper(), int(img.depth_to))
                metadata = register_store._review_cache.get(cache_key)
                
                if metadata:
                    # These are typically already set during scanning,
                    # but we can update them here if needed
                    if not img.classification or str(img.classification) == "UNASSIGNED":
                        if metadata.classification:
                            img.classification = metadata.classification
                    
                    # Merge tags
                    if metadata.tags:
                        if not hasattr(img, 'tags') or not img.tags:
                            img.tags = set()
                        img.tags.update(metadata.tags)
                    
                    populated_count += 1
                    
            except Exception as e:
                logger.debug(f"Review lookup failed for {img.hole_id}_{img.depth_to}: {e}")
        
        elapsed = time.time() - start_time
        logger.info(f"Populated review metadata: {populated_count}/{len(images)} images in {elapsed:.2f}s")
        
        return populated_count
    
    def populate_hex_colors(self, images: List[Any]) -> int:
        """
        Populate hex color from cache into csv_data (for sorting/display).
        
        Args:
            images: List of CompartmentImage objects
            
        Returns:
            Number of images that received hex color data
        """
        if not self._hex_cache:
            logger.debug("No hex color cache - skipping hex population")
            return 0
        
        populated_count = 0
        
        for img in images:
            key = (img.hole_id, img.depth_from, img.depth_to)
            hex_color = self._hex_cache.get(key)
            
            if hex_color:
                if not img.csv_data:
                    img.csv_data = {}
                img.csv_data['hex_color'] = hex_color
                img.csv_data['combined_hex'] = hex_color
                populated_count += 1
        
        logger.debug(f"Populated hex colors: {populated_count}/{len(images)} images")
        return populated_count
    
    def get_csv_value(
        self,
        hole_id: str,
        depth_to: float,
        column: str
    ) -> Optional[Any]:
        """
        Get a single CSV value from GeologicalStore.
        
        Args:
            hole_id: Hole ID
            depth_to: End depth
            column: Column name
            
        Returns:
            The value or None if not found
        """
        if not self.has_geological_store:
            return None
        
        try:
            key = ImageKey(hole_id=hole_id, depth_to=float(depth_to))
            return self._coordinator.geological_store.get_value(key, column)
        except Exception as e:
            logger.debug(f"get_csv_value failed for {hole_id}_{depth_to}.{column}: {e}")
            return None


# =============================================================================
# FILTER PIPELINE - Main filtering orchestration
# =============================================================================

class FilterPipeline:
    """
    Orchestrates all filtering operations in a single, clear flow.
    
    The pipeline applies filters in this order:
    1. Display mode (hole-by-hole, all images)
    2. Dynamic filters (CSV and register columns)
    3. Scatter selection (lasso)
    4. Classification visibility (all/classified/unclassified/conflicts)
    5. Placeholder generation (for all_intervals mode)
    6. Sorting
    
    The pipeline is stateless - each execution takes inputs and returns outputs.
    State is managed by the caller (LoggingReviewDialog).
    
    Usage:
        pipeline = FilterPipeline(data_coordinator)
        
        criteria = FilterCriteria(
            display_mode=DisplayMode.HOLE_BY_HOLE,
            current_hole="BA0001",
            classification_visibility=ClassificationVisibility.UNCLASSIFIED,
            sort_column="depth_to"
        )
        
        result = pipeline.execute(all_images, criteria)
        grid_canvas.load_images(result.images)
    """
    
    def __init__(
        self,
        data_coordinator=None,
        hex_color_cache: Optional[Dict] = None,
        compute_review_metadata: Optional[Callable] = None
    ):
        """
        Initialize the filter pipeline.
        
        Args:
            data_coordinator: DataCoordinator with geological_store and register_store
            hex_color_cache: Pre-loaded hex color cache for color sorting
            compute_review_metadata: Callback to compute review metadata for an image
                                    Should accept (img) and return dict with:
                                    consensus_classification, review_count, agreement, tags
        """
        self._coordinator = data_coordinator
        self._hex_cache = hex_color_cache or {}
        self._compute_review_metadata = compute_review_metadata
        
        # Populator for ensuring images have data
        self._populator = ImageDataPopulator(data_coordinator, hex_color_cache)
        
        # Column categories for routing filters to correct store
        self._register_columns = {
            'classification', 'classified_by', 'consensus_classification', 
            'all_classifications', 'all_reviewers', 'review_count',
            'agreement', 'comments','combined_hex', 'wet_hex', 'dry_hex', 'has_wet', 'has_dry'
        }
        
        # Image attribute columns (filtered directly on CompartmentImage objects)
        self._image_attr_columns = {
            'hole_id', 'depth_from', 'depth_to', 'moisture_status', 'filename'
        }
        
        logger.info("FilterPipeline initialized (coordinator=%s)", data_coordinator is not None)
    
    def _sort_needs_csv_data(self, criteria: FilterCriteria) -> bool:
        """
        Check if the sort columns require CSV data.
        
        Returns True if primary or secondary sort column is a CSV column.
        """
        non_csv_columns = {
            "hole_id", "depth_from", "depth_to", "classification",
            "consensus_classification", "review_count", "agreement",
            "Average Hex Colour", "(none)", "", None
        }
        
        # Check primary sort
        if criteria.sort_column and criteria.sort_column not in non_csv_columns:
            if not criteria.sort_column.startswith("tag_"):
                return True
        
        # Check secondary sort
        secondary = criteria.secondary_sort_column
        if secondary and secondary not in non_csv_columns:
            if not secondary.startswith("tag_"):
                return True
        
        return False
    
    def _get_sort_columns(self, criteria: FilterCriteria) -> List[str]:
        """
        Get the CSV columns needed for sorting.
        
        Returns list of column names that need to be populated for sorting to work.
        """
        columns = []
        non_csv_columns = {
            "hole_id", "depth_from", "depth_to", "classification",
            "consensus_classification", "review_count", "agreement",
            "Average Hex Colour", "(none)", "", None
        }
        
        if criteria.sort_column and criteria.sort_column not in non_csv_columns:
            if not criteria.sort_column.startswith("tag_"):
                columns.append(criteria.sort_column)
        
        secondary = criteria.secondary_sort_column
        if secondary and secondary not in non_csv_columns:
            if not secondary.startswith("tag_"):
                columns.append(secondary)
        
        return columns
    
    def execute(
        self,
        all_images: List[Any],
        criteria: FilterCriteria
    ) -> FilterResult:
        """
        Execute the complete filter pipeline.
        
        Args:
            all_images: Complete list of all CompartmentImage objects
            criteria: FilterCriteria specifying what to filter
            
        Returns:
            FilterResult with filtered images and metadata
        """
        start_time = time.time()
        
        DebugCheckpoint.log("PIPELINE START",
                           total_images=len(all_images),
                           criteria_hash=criteria.get_hash())
        
        # Start with all images
        filtered = all_images.copy()
        total_before = len(all_images)
        
        # --- Stage 1: Display Mode Filtering ---
        filtered = self._apply_display_mode_filter(filtered, criteria)
        DebugCheckpoint.log("After display mode filter",
                           count=len(filtered),
                           mode=criteria.display_mode.value)
        
        # --- Stage 2: Dynamic Filters (CSV + Register) ---
        if criteria.dynamic_filters:
            filtered = self._apply_dynamic_filters(filtered, criteria.dynamic_filters)
            DebugCheckpoint.log("After dynamic filters",
                               count=len(filtered),
                               filter_count=len(criteria.dynamic_filters))
        
        # --- Stage 3: Scatter Selection ---
        if criteria.scatter_selection:
            filtered = self._apply_scatter_selection(filtered, criteria.scatter_selection)
            DebugCheckpoint.log("After scatter selection",
                               count=len(filtered),
                               selection_size=len(criteria.scatter_selection))
        
        # --- Stage 4: Classification Visibility ---
        filtered = self._apply_classification_visibility(
            filtered, 
            criteria.classification_visibility,
            criteria.show_others_reviews
        )
        DebugCheckpoint.log("After classification visibility",
                           count=len(filtered),
                           visibility=criteria.classification_visibility.value)
        
        # --- Stage 5: Interval Placeholders ---
        if criteria.add_placeholders and criteria.display_mode == DisplayMode.ALL_INTERVALS:
            filtered = self._add_interval_placeholders(filtered)
            DebugCheckpoint.log("After placeholder addition",
                               count=len(filtered))
        
        # --- Stage 5.5: Populate CSV data for sorting and visualization ---
        # CSV data must be loaded before sorting by CSV columns works
        # We populate ALL columns (not just sort columns) so visualizations also work
        sort_needs_csv = self._sort_needs_csv_data(criteria)
        if sort_needs_csv and self._populator.has_geological_store:
            # Pass columns=None to populate ALL columns for both sorting and viz
            populate_count = self._populator.populate_csv_data(filtered, columns=None, force=True)
            logger.debug(f"Populated CSV data for sorting/viz: {populate_count} images (all columns)")
            DebugCheckpoint.log("After CSV population for sort",
                               count=len(filtered),
                               populated=populate_count)
        
        # --- Stage 6: Sorting ---
        filtered = self._apply_sorting(filtered, criteria)
        DebugCheckpoint.log("After sorting",
                           count=len(filtered),
                           sort_column=criteria.sort_column,
                           first_3=[(img.hole_id, img.depth_to) for img in filtered[:3]])
        
        # --- Build Result ---
        execution_time = (time.time() - start_time) * 1000
        
        result = FilterResult(
            images=filtered,
            total_before=total_before,
            total_after=len(filtered),
            csv_matches=sum(1 for img in filtered if getattr(img, 'in_csv', False)),
            filter_descriptions=criteria.get_descriptions(),
            execution_time_ms=execution_time
        )
        
        logger.info(f"FilterPipeline complete: {total_before} → {len(filtered)} images in {execution_time:.1f}ms")
        
        DebugCheckpoint.log("PIPELINE COMPLETE",
                           before=total_before,
                           after=len(filtered),
                           time_ms=execution_time)
        
        return result
    
    # =========================================================================
    # Stage 1: Display Mode
    # =========================================================================
    
    def _apply_display_mode_filter(
        self,
        images: List[Any],
        criteria: FilterCriteria
    ) -> List[Any]:
        """
        Apply display mode filtering.
        
        For hole-by-hole mode, filters to just the current hole.
        """
        if criteria.display_mode != DisplayMode.HOLE_BY_HOLE:
            return images
        
        if not criteria.current_hole:
            logger.warning("Hole-by-hole mode but no current_hole set")
            return images
        
        current_hole_upper = criteria.current_hole.upper()
        filtered = [
            img for img in images
            if img.hole_id.upper() == current_hole_upper
        ]
        
        logger.debug(f"Display mode filter: {len(images)} → {len(filtered)} (hole={criteria.current_hole})")
        
        return filtered
    
    # =========================================================================
    # Stage 2: Dynamic Filters
    # =========================================================================
    
    def _apply_dynamic_filters(
        self,
        images: List[Any],
        filters: List[Dict[str, Any]]
    ) -> List[Any]:
        """
        Apply dynamic filter rows.
        
        Separates filters into CSV, register, and image attribute categories,
        queries the appropriate stores, and intersects results.
        """
        if not filters:
            return images
        
        # Build image key lookup
        image_by_key: Dict[Tuple[str, float], Any] = {}
        for img in images:
            key = (img.hole_id.upper(), float(img.depth_to))
            image_by_key[key] = img
        
        # Start with all keys as candidates
        matching_keys = set(image_by_key.keys())
        
        # Separate filters by source
        csv_filters = []
        register_filters = []
        image_attr_filters = []
        rc_metrics_filters = []
        
        for f in filters:
            if not f.get("column") or not f.get("operator"):
                continue
            
            col_display = f["column"]
            col_lower = col_display.lower().split(" (")[0].strip()
            is_rc_metrics = "(rc metrics)" in col_display.lower()
            
            if col_lower in self._image_attr_columns:
                image_attr_filters.append(f)
            elif col_lower in self._register_columns or col_lower.startswith("tag_") or col_lower.startswith("rev_"):
                register_filters.append(f)
            elif is_rc_metrics:
                rc_metrics_filters.append(f)
            else:
                csv_filters.append(f)
        
        logger.debug(
            f"Filter split: {len(csv_filters)} CSV, {len(register_filters)} register, "
            f"{len(rc_metrics_filters)} RC metrics, {len(image_attr_filters)} image attrs"
        )
        
        # Query GeologicalStore for CSV filters
        if csv_filters and self._coordinator and self._coordinator.geological_store.is_loaded:
            query_filters = [
                {
                    'column': f["column"].split(" (")[0].strip(),
                    'operator': f["operator"],
                    'value': f["value"],
                    'value2': f.get("value2"),
                    'data_type': f.get("data_type"),
                }
                for f in csv_filters
            ]
            
            logger.info(f"[DEBUG] CSV query_filters: {query_filters}")
            logger.info(f"[DEBUG] matching_keys before CSV: {len(matching_keys)}")
            
            # Debug: Check what values actually exist in the data for this column
            geo_store = self._coordinator.geological_store
            if hasattr(geo_store, '_data') and geo_store._data is not None:
                col_name = query_filters[0]['column']
                if col_name in geo_store._data.columns:
                    unique_vals = geo_store._data[col_name].dropna().unique()[:20]
                    logger.info(f"[DEBUG] Actual unique values in '{col_name}': {list(unique_vals)}")
                    
                    # Check what the 'in' filter would match
                    filter_vals = [v.strip() for v in query_filters[0]['value'].split(',')]
                    logger.info(f"[DEBUG] Filter values parsed: {filter_vals}")
                    
                    matches = geo_store._data[col_name].isin(filter_vals)
                    logger.info(f"[DEBUG] Direct isin match count: {matches.sum()}")
            
            csv_keys = self._coordinator.geological_store.query(query_filters)
            
            logger.info(f"[DEBUG] csv_keys returned: {len(csv_keys)}")
            logger.info(f"[DEBUG] Sample csv_keys (first 5): {list(csv_keys)[:5]}")
            
            matching_keys &= csv_keys
            
            logger.info(f"CSV query: {len(csv_keys)} matches, after intersection: {len(matching_keys)}")
        
        # Query RegisterStore for register filters
        if register_filters and self._coordinator and self._coordinator.register_store:
            query_filters = [
                {
                    'column': f["column"].split(" (")[0].strip(),
                    'operator': f["operator"],
                    'value': f["value"],
                    'value2': f.get("value2")
                }
                for f in register_filters
            ]
            
            register_keys = self._coordinator.register_store.query(query_filters)
            matching_keys &= register_keys
            
            logger.info(f"Register query: {len(register_keys)} matches")

        # Query RC metrics store for RC metrics filters
        if rc_metrics_filters:
            rc_keys = self._query_rc_metrics(rc_metrics_filters)
            matching_keys &= rc_keys
            logger.info(f"RC metrics query: {len(rc_keys)} matches")
        
        # Apply image attribute filters directly on images
        if image_attr_filters:
            attr_matching_keys = set()
            for key, img in image_by_key.items():
                if self._image_matches_attr_filters(img, image_attr_filters):
                    attr_matching_keys.add(key)
            matching_keys &= attr_matching_keys
            logger.info(f"Image attr filter: {len(attr_matching_keys)} matches")
        
        # Build result list preserving original order
        result = [
            img for img in images
            if (img.hole_id.upper(), float(img.depth_to)) in matching_keys
        ]
        
        logger.debug(f"Dynamic filter result: {len(images)} → {len(result)}")
        
        return result

    def _query_rc_metrics(self, filters: List[Dict[str, Any]]) -> Set[Tuple[str, float]]:
        """
        Query RC metrics DataFrame using filter criteria.

        Returns:
            Set of (hole_id_upper, depth_to) tuples matching all filters
        """
        if not filters:
            return set()

        if not self._coordinator or not getattr(self._coordinator, "has_rc_metrics", False):
            logger.info("[RC METRICS] No RC metrics available for filtering")
            return set()

        rc_df = self._coordinator.get_rc_metrics_dataframe()
        if rc_df is None or rc_df.empty:
            logger.info("[RC METRICS] RC metrics DataFrame empty")
            return set()

        df = rc_df.copy()

        # Normalize hole_id for consistent keys
        if "hole_id" in df.columns:
            df["hole_id"] = df["hole_id"].astype(str).str.strip().str.upper()

        mask = pd.Series([True] * len(df), index=df.index)

        for flt in filters:
            col_display = flt.get("column", "")
            col = col_display.split(" (")[0].strip().lower()
            op = flt.get("operator", "")
            val = flt.get("value")
            val2 = flt.get("value2")

            if not col or not op:
                continue

            # Find column (case-insensitive)
            col_match = None
            for c in df.columns:
                if c.lower() == col:
                    col_match = c
                    break

            if not col_match:
                logger.info(f"[RC METRICS] Column '{col}' not found - returning empty result")
                return set()

            col_data = df[col_match]

            try:
                if op in ('=', 'equals', '=='):
                    mask &= (col_data == val)
                elif op in ('!=', '≠', 'not equals'):
                    mask &= (col_data != val)
                elif op in ('>', 'greater than'):
                    mask &= (pd.to_numeric(col_data, errors='coerce') > float(val))
                elif op in ('>=', '≥', 'greater than or equal'):
                    mask &= (pd.to_numeric(col_data, errors='coerce') >= float(val))
                elif op in ('<', 'less than'):
                    mask &= (pd.to_numeric(col_data, errors='coerce') < float(val))
                elif op in ('<=', '≤', 'less than or equal'):
                    mask &= (pd.to_numeric(col_data, errors='coerce') <= float(val))
                elif op == 'between':
                    numeric_col = pd.to_numeric(col_data, errors='coerce')
                    mask &= (numeric_col >= float(val)) & (numeric_col <= float(val2))
                elif op == 'contains':
                    mask &= col_data.astype(str).str.contains(str(val), case=False, na=False)
                elif op == 'not contains':
                    mask &= ~col_data.astype(str).str.contains(str(val), case=False, na=False)
                elif op == 'starts with':
                    mask &= col_data.astype(str).str.startswith(str(val), na=False)
                elif op == 'ends with':
                    mask &= col_data.astype(str).str.endswith(str(val), na=False)
                elif op in ('in', 'in list'):
                    if isinstance(val, str):
                        val_list = [v.strip().lower() for v in val.split(',')]
                    else:
                        val_list = [str(v).lower() for v in val]
                    col_normalized = col_data.astype(str).str.strip().str.lower()
                    mask &= col_normalized.isin(val_list)
                elif op in ('not in', 'not in list'):
                    if isinstance(val, str):
                        val_list = [v.strip().lower() for v in val.split(',')]
                    else:
                        val_list = [str(v).lower() for v in val]
                    col_normalized = col_data.astype(str).str.strip().str.lower()
                    mask &= ~col_normalized.isin(val_list)
                elif op in ('is null', 'is empty'):
                    mask &= col_data.isna()
                elif op in ('not null', 'is not empty'):
                    mask &= col_data.notna()
            except (ValueError, TypeError):
                return set()

        filtered = df[mask]
        if "hole_id" not in filtered.columns or "depth_to" not in filtered.columns:
            return set()

        return set((str(r["hole_id"]).upper(), float(r["depth_to"])) for _, r in filtered.iterrows())
    
    def _image_matches_attr_filters(
        self,
        img: Any,
        filters: List[Dict[str, Any]]
    ) -> bool:
        """
        Check if an image matches all attribute filters.
        
        Args:
            img: CompartmentImage object
            filters: List of filter dicts for image attributes
            
        Returns:
            True if image matches all filters
        """
        for f in filters:
            col = f["column"].lower().split(" (")[0].strip()
            op = f["operator"]
            val = f.get("value", "")
            val2 = f.get("value2")
            
            # Get attribute value from image
            img_val = getattr(img, col, None)
            if img_val is None:
                return False
            
            # Apply operator
            if not self._apply_filter_operator(img_val, op, val, val2):
                return False
        
        return True
    
    def _apply_filter_operator(
        self,
        img_val: Any,
        operator: str,
        filter_val: Any,
        filter_val2: Any = None
    ) -> bool:
        """
        Apply a filter operator to compare values.
        
        Args:
            img_val: Value from the image
            operator: Comparison operator
            filter_val: Filter value to compare against
            filter_val2: Second filter value (for 'between')
            
        Returns:
            True if the comparison passes
        """
        # Handle null checks first
        if operator in ('is null', 'is empty'):
            return img_val is None or img_val == '' or (isinstance(img_val, float) and pd.isna(img_val))
        if operator in ('not null', 'is not empty', 'not empty'):
            return img_val is not None and img_val != '' and not (isinstance(img_val, float) and pd.isna(img_val))
        
        # For other operators, null values don't match
        if img_val is None or img_val == '':
            return False
        
        try:
            # String comparisons (case-insensitive)
            img_str = str(img_val).lower()
            filter_str = str(filter_val).lower() if filter_val else ""
            
            if operator in ('=', 'equals', '=='):
                return img_str == filter_str
            elif operator in ('!=', '≠', 'not equals'):
                return img_str != filter_str
            elif operator == 'contains':
                return filter_str in img_str
            elif operator == 'not contains':
                return filter_str not in img_str
            elif operator == 'starts with':
                return img_str.startswith(filter_str)
            elif operator == 'ends with':
                return img_str.endswith(filter_str)
            elif operator in ('in', 'in list'):
                val_list = [v.strip().lower() for v in str(filter_val).split(',')]
                return img_str in val_list
            elif operator in ('not in', 'not in list'):
                val_list = [v.strip().lower() for v in str(filter_val).split(',')]
                return img_str not in val_list
            
            # Numeric comparisons
            img_num = float(img_val)
            filter_num = float(filter_val)
            
            if operator in ('>', 'greater than'):
                return img_num > filter_num
            elif operator in ('>=', '≥', 'greater than or equal'):
                return img_num >= filter_num
            elif operator in ('<', 'less than'):
                return img_num < filter_num
            elif operator in ('<=', '≤', 'less than or equal'):
                return img_num <= filter_num
            elif operator == 'between' and filter_val2:
                return float(filter_val) <= img_num <= float(filter_val2)
                
        except (ValueError, TypeError):
            # If numeric conversion fails, fall back to string comparison
            return str(img_val).lower() == str(filter_val).lower()
        
        return False

    # =========================================================================
    # Stage 3: Scatter Selection
    # =========================================================================
    
    def _apply_scatter_selection(
        self,
        images: List[Any],
        selection: Set[Tuple[str, float]]
    ) -> List[Any]:
        """
        Apply scatter plot lasso selection.
        
        Only keeps images whose (hole_id, depth_to) is in the selection set.
        """
        if not selection:
            return images
        
        # Normalize selection keys
        normalized_selection = {
            (str(h).upper(), float(d))
            for h, d in selection
        }
        
        result = [
            img for img in images
            if (img.hole_id.upper(), float(img.depth_to)) in normalized_selection
        ]
        
        logger.debug(f"Scatter selection: {len(images)} → {len(result)} (selection has {len(selection)} points)")
        
        return result
    
    # =========================================================================
    # Stage 4: Classification Visibility
    # =========================================================================
    
    def _apply_classification_visibility(
        self,
        images: List[Any],
        visibility: ClassificationVisibility,
        show_others: bool
    ) -> List[Any]:
        """
        Apply classification visibility filter.
        
        Filters based on visibility mode and show_others flag:
        
        - ALL: Show everything
        - CLASSIFIED + show_others=False: Images classified by current user
        - CLASSIFIED + show_others=True: Images classified by anyone
        - UNCLASSIFIED + show_others=False: Images NOT classified by current user
        - UNCLASSIFIED + show_others=True: Images NOT classified by anyone
        - CONFLICTS: Images where reviewers disagree
        
        Args:
            images: List of CompartmentImage objects
            visibility: ClassificationVisibility enum value
            show_others: If True, consider all users' classifications; if False, only current user
            
        Returns:
            Filtered list of images
        """
        if visibility == ClassificationVisibility.ALL:
            return images
        
        result = []
        
        for img in images:
            # Get current user's classification
            my_class_str = str(img.classification) if img.classification else ""
            my_is_unassigned = my_class_str in ('', 'UNASSIGNED', 'ClassificationCategory.UNASSIGNED')
            
            # Get other users' classifications if available
            others_have_classified = False
            other_classifications = []
            
            if hasattr(img, 'other_reviews') and img.other_reviews:
                for review in img.other_reviews:
                    for field in ['classification', 'Classification', 'Lithology', 'Rock_Type']:
                        if field in review and review[field]:
                            other_class = str(review[field])
                            if other_class and other_class not in ('', 'UNASSIGNED', 'ClassificationCategory.UNASSIGNED'):
                                others_have_classified = True
                                other_classifications.append(other_class)
                            break
            
            # Determine if classified by anyone
            anyone_classified = (not my_is_unassigned) or others_have_classified
            
            if visibility == ClassificationVisibility.UNCLASSIFIED:
                if show_others:
                    # All Unclassified: No one has classified this image
                    if not anyone_classified:
                        result.append(img)
                else:
                    # My Unclassified: I haven't classified (regardless of others)
                    if my_is_unassigned:
                        result.append(img)
                    
            elif visibility == ClassificationVisibility.CLASSIFIED:
                if show_others:
                    # All Classified: Anyone has classified this image
                    if anyone_classified:
                        result.append(img)
                else:
                    # My Classified: I have classified (regardless of others)
                    if not my_is_unassigned:
                        result.append(img)
                    
            elif visibility == ClassificationVisibility.CONFLICTS:
                # Check for conflicts - need at least 2 different classifications
                all_classes = []
                if not my_is_unassigned:
                    all_classes.append(my_class_str)
                all_classes.extend(other_classifications)
                
                # Conflict = more than one unique classification value
                unique_classes = set(all_classes)
                if len(unique_classes) > 1:
                    result.append(img)
        
        logger.debug(f"Classification visibility ({visibility.value}, show_others={show_others}): {len(images)} → {len(result)}")
        
        return result
    
    # =========================================================================
    # Stage 5: Interval Placeholders
    # =========================================================================
    
    def _add_interval_placeholders(self, images: List[Any]) -> List[Any]:
        """
        Add placeholder images for missing intervals.
        
        Used in ALL_INTERVALS mode to show gaps in the data.
        """
        if not images:
            return images
        
        from collections import Counter
        
        # Group by hole
        holes_data: Dict[str, List[Any]] = {}
        for img in images:
            if img.hole_id not in holes_data:
                holes_data[img.hole_id] = []
            holes_data[img.hole_id].append(img)
        
        result = []
        
        for hole_id in sorted(holes_data.keys()):
            hole_images = sorted(holes_data[hole_id], key=lambda x: x.depth_to)
            
            if not hole_images:
                continue
            
            # Determine interval size
            intervals = []
            for i in range(1, len(hole_images)):
                interval = hole_images[i].depth_to - hole_images[i - 1].depth_to
                if interval > 0:
                    intervals.append(interval)
            
            interval_size = Counter(intervals).most_common(1)[0][0] if intervals else 1.0
            
            # Generate complete sequence
            max_depth = hole_images[-1].depth_to
            depth_map = {img.depth_to: img for img in hole_images}
            
            current_depth = hole_images[0].depth_to
            while current_depth <= max_depth + 0.01:
                if current_depth in depth_map:
                    result.append(depth_map[current_depth])
                else:
                    # Import CompartmentImage here to avoid circular import
                    from gui.logging_review_dialog import CompartmentImage
                    
                    placeholder = CompartmentImage(
                        filename=f"PLACEHOLDER_{hole_id}_{current_depth:.1f}",
                        hole_id=hole_id,
                        depth_from=current_depth - interval_size,
                        depth_to=current_depth,
                        image_path=None,
                        moisture_status=None,
                    )
                    placeholder.is_placeholder = True
                    result.append(placeholder)
                
                current_depth += interval_size
        
        logger.debug(f"Placeholder generation: {len(images)} → {len(result)}")
        
        return result
    
    # =========================================================================
    # Stage 6: Sorting
    # =========================================================================
    
    def _apply_sorting(
        self,
        images: List[Any],
        criteria: FilterCriteria
    ) -> List[Any]:
        """
        Apply sorting to images with primary and optional secondary sort.
        
        Uses Python's stable sort: sort by secondary first, then by primary.
        This correctly handles mixed asc/desc for different sort levels.
        
        Handles:
        - Basic columns (hole_id, depth_from, depth_to)
        - Classification
        - Hex color (with HSL conversion)
        - Review metadata (consensus, review_count, agreement)
        - CSV columns
        - Tags
        """
        if not images:
            return images
        
        primary_col = criteria.sort_column
        primary_reverse = criteria.sort_order == "desc"
        secondary_col = criteria.secondary_sort_column
        secondary_reverse = criteria.secondary_sort_order == "desc"
        
        # Check if secondary sort is actually set (not None or "(none)")
        has_secondary = secondary_col and secondary_col not in (None, "", "(none)")
        
        logger.info(
            f"[SORT] Sorting {len(images)} images by {primary_col} ({criteria.sort_order})"
            + (f" then by {secondary_col} ({criteria.secondary_sort_order})" if has_secondary else "")
        )
        
        try:
            result = list(images)  # Copy to avoid modifying original
            
            # Python's sort is stable, so we sort by secondary FIRST, then primary
            # This gives correct multi-level sort behavior
            
            if has_secondary:
                # Sort by secondary column first
                result = self._sort_by_column(result, secondary_col, secondary_reverse, criteria)
                logger.debug(f"[SORT] After secondary sort by {secondary_col}: first 3 = {[(img.hole_id, img.depth_to) for img in result[:3]]}")
            
            # Then sort by primary column (stable sort preserves secondary order within equal primary values)
            result = self._sort_by_column(result, primary_col, primary_reverse, criteria)
            logger.debug(f"[SORT] After primary sort by {primary_col}: first 3 = {[(img.hole_id, img.depth_to) for img in result[:3]]}")
            
            return result
                
        except Exception as e:
            logger.error(f"[SORT] Error sorting: {e}", exc_info=True)
            return images
    
    def _sort_by_column(
        self,
        images: List[Any],
        column: str,
        reverse: bool,
        criteria: FilterCriteria
    ) -> List[Any]:
        """
        Sort images by a single column.
        
        Dispatches to the appropriate sort method based on column type.
        """
        if not column or column == "(none)":
            return images
        
        if column == "Average Hex Colour":
            return self._sort_by_hex_color(images, reverse, criteria)
        
        elif column in ["hole_id", "depth_from", "depth_to"]:
            return self._sort_by_basic_column(images, column, reverse)
        
        elif column == "classification":
            return self._sort_by_classification(images, reverse, criteria.show_others_reviews)
        
        elif column in ["consensus_classification", "review_count", "agreement"]:
            return self._sort_by_review_metadata(images, column, reverse)
        
        elif column.startswith("tag_"):
            return self._sort_by_tag(images, column[4:], reverse)
        
        else:
            # CSV column
            return self._sort_by_csv_column(images, column, reverse)
    
    def _sort_by_basic_column(
        self,
        images: List[Any],
        column: str,
        reverse: bool
    ) -> List[Any]:
        """Sort by basic image attributes (pure sort, no secondary grouping)."""
        return sorted(
            images,
            key=lambda img: getattr(img, column),
            reverse=reverse
        )
    
    def _sort_by_classification(
        self,
        images: List[Any],
        reverse: bool,
        show_others: bool
    ) -> List[Any]:
        """Sort by classification (with optional consensus, pure sort)."""
        def get_key(img):
            if show_others and self._compute_review_metadata:
                metadata = self._compute_review_metadata(img)
                consensus = metadata.get("consensus_classification", "")
                return consensus or "Unassigned"
            return str(img.classification)
        
        return sorted(images, key=get_key, reverse=reverse)
    
    def _sort_by_hex_color(
        self,
        images: List[Any],
        reverse: bool,
        criteria: FilterCriteria
    ) -> List[Any]:
        """Sort by hex color using HSL conversion."""
        def hex_to_hsl(hex_color: str) -> Tuple[float, float, float]:
            """Convert hex to HSL for sorting."""
            hex_color = hex_color.lstrip("#")
            if len(hex_color) != 6:
                return (float("inf"), 0, 0)
            
            r = int(hex_color[0:2], 16) / 255.0
            g = int(hex_color[2:4], 16) / 255.0
            b = int(hex_color[4:6], 16) / 255.0
            
            max_val = max(r, g, b)
            min_val = min(r, g, b)
            diff = max_val - min_val
            
            l = (max_val + min_val) / 2.0
            
            if diff == 0:
                h = s = 0
            else:
                s = diff / (2.0 - max_val - min_val) if l > 0.5 else diff / (max_val + min_val)
                
                if max_val == r:
                    h = ((g - b) / diff + (6 if g < b else 0)) / 6.0
                elif max_val == g:
                    h = ((b - r) / diff + 2) / 6.0
                else:
                    h = ((r - g) / diff + 4) / 6.0
            
            return (h, s, l)
        
        def get_key(img):
            key = (img.hole_id, img.depth_from, img.depth_to)
            hex_color = self._hex_cache.get(key, "")
            
            if hex_color:
                h, s, l = hex_to_hsl(hex_color)
                # Could support different sort modes (hue, lightness, saturation)
                return (h, s, l)
            
            return (float("inf"), 0, 0) if not reverse else (float("-inf"), 0, 0)
        
        return sorted(images, key=get_key, reverse=reverse)
    
    def _sort_by_review_metadata(
        self,
        images: List[Any],
        column: str,
        reverse: bool
    ) -> List[Any]:
        """Sort by review metadata column (pure sort)."""
        if not self._compute_review_metadata:
            logger.warning(f"Cannot sort by {column} - no review metadata callback")
            return images
        
        def get_key(img):
            metadata = self._compute_review_metadata(img)
            return metadata.get(column, "")
        
        return sorted(images, key=get_key, reverse=reverse)
    
    def _sort_by_tag(
        self,
        images: List[Any],
        tag_id: str,
        reverse: bool
    ) -> List[Any]:
        """Sort by whether a tag is present (pure sort)."""
        def get_key(img):
            has_tag = hasattr(img, 'tags') and img.tags and tag_id in img.tags
            return has_tag
        
        return sorted(images, key=get_key, reverse=reverse)
    
    def _sort_by_csv_column(
        self,
        images: List[Any],
        column: str,
        reverse: bool
    ) -> List[Any]:
        """Sort by CSV column value (pure sort)."""
        # DEBUG: Sample csv_data state before sorting
        has_csv_data = sum(1 for img in images if img.csv_data)
        has_column = sum(1 for img in images if img.csv_data and column in img.csv_data)
        has_value = sum(1 for img in images if img.csv_data and column in img.csv_data and img.csv_data[column] is not None)
        
        logger.info(f"[SORT_DEBUG] _sort_by_csv_column: column={column}, reverse={reverse}")
        logger.info(f"[SORT_DEBUG] CSV data stats: {has_csv_data}/{len(images)} have csv_data, {has_column} have column '{column}', {has_value} have non-null values")
        
        # DEBUG: Show first 5 images BEFORE sort with their values
        logger.info(f"[SORT_DEBUG] BEFORE sort - first 5 images:")
        for i, img in enumerate(images[:5]):
            csv_keys = list(img.csv_data.keys())[:5] if img.csv_data else []
            val = img.csv_data.get(column, "MISSING") if img.csv_data else "NO_CSV_DATA"
            logger.info(f"[SORT_DEBUG]   [{i}] {img.hole_id}_{img.depth_to}: {column}={val}, csv_keys={csv_keys}")
        
        def get_key(img):
            """
            Build a comparable key for mixed-type CSV values.

            Key shape:
                (missing_flag, type_flag, value_part)

            - missing_flag: 0 = has value, 1 = missing/NaN
            - type_flag:    0 = numeric,   1 = non-numeric (string-ish)
            - value_part:
                * float for numeric values
                * lower-cased string for text values

            This guarantees Python never has to compare float vs str directly.
            """
            # Missing column or csv_data - treat as missing
            if not img.csv_data or column not in img.csv_data:
                return (1, 1, "")

            val = img.csv_data[column]

            # Explicit null-like values
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return (1, 1, "")

            # Present value
            # Numeric branch - keep numeric ordering
            if isinstance(val, (int, float, np.number)):
                try:
                    num = float(val)
                except (TypeError, ValueError):
                    # Fallback to string representation if conversion fails
                    return (0, 1, str(val).lower())
                return (0, 0, num)

            # Everything else - compare as case-insensitive string
            return (0, 1, str(val).lower())

        result = sorted(images, key=get_key, reverse=reverse)

        
        # DEBUG: Show first 5 images AFTER sort with their values
        logger.info(f"[SORT_DEBUG] AFTER sort - first 5 images:")
        for i, img in enumerate(result[:5]):
            val = img.csv_data.get(column, "MISSING") if img.csv_data else "NO_CSV_DATA"
            logger.info(f"[SORT_DEBUG]   [{i}] {img.hole_id}_{img.depth_to}: {column}={val}")
        
        # DEBUG: Show last 5 too (should have lowest/null values for desc)
        logger.info(f"[SORT_DEBUG] AFTER sort - last 5 images:")
        for i, img in enumerate(result[-5:]):
            val = img.csv_data.get(column, "MISSING") if img.csv_data else "NO_CSV_DATA"
            logger.info(f"[SORT_DEBUG]   [{len(result)-5+i}] {img.hole_id}_{img.depth_to}: {column}={val}")
        
        return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_filter_criteria_from_dialog(dialog) -> FilterCriteria:
    """
    Create FilterCriteria from LoggingReviewDialog state.
    
    This is a helper to extract all filter state from the dialog
    into a clean FilterCriteria object.
    
    Args:
        dialog: LoggingReviewDialog instance
        
    Returns:
        FilterCriteria populated from dialog state
    """
    # Get display mode
    mode_str = dialog.display_mode_var.get() if hasattr(dialog, 'display_mode_var') else "all_images"
    display_mode = {
        "hole_by_hole": DisplayMode.HOLE_BY_HOLE,
        "all_images": DisplayMode.ALL_IMAGES,
        "all_intervals": DisplayMode.ALL_INTERVALS,
    }.get(mode_str, DisplayMode.ALL_IMAGES)
    
    # Get classification visibility
    vis_str = dialog.filter_mode.get() if hasattr(dialog, 'filter_mode') else "all"
    visibility = {
        "all": ClassificationVisibility.ALL,
        "classified": ClassificationVisibility.CLASSIFIED,
        "unclassified": ClassificationVisibility.UNCLASSIFIED,
        "disagreements": ClassificationVisibility.CONFLICTS,
    }.get(vis_str, ClassificationVisibility.ALL)
    
    # Get dynamic filters
    dynamic_filters = []
    if hasattr(dialog, 'filter_rows'):
        for row in dialog.filter_rows:
            config = row.get_filter_config()
            if config.get("column") and config.get("operator"):
                dynamic_filters.append(config)
    
    # Get scatter selection
    scatter_selection = None
    if hasattr(dialog, '_scatter_selection_info') and dialog._scatter_selection_info:
        # Need to extract actual selection from scatter window if open
        if hasattr(dialog, 'grid_canvas') and dialog.grid_canvas:
            if hasattr(dialog.grid_canvas, 'zoom_preview') and dialog.grid_canvas.zoom_preview:
                # TODO: Get selection from scatter window
                pass
    
    # Build criteria
    criteria = FilterCriteria(
        display_mode=display_mode,
        current_hole=dialog.unique_holes[dialog.current_hole_index] if dialog.unique_holes and dialog.current_hole_index < len(dialog.unique_holes) else None,
        current_hole_index=dialog.current_hole_index if hasattr(dialog, 'current_hole_index') else 0,
        unique_holes=dialog.unique_holes if hasattr(dialog, 'unique_holes') else [],
        
        classification_visibility=visibility,
        show_others_reviews=dialog.show_others_var.get() if hasattr(dialog, 'show_others_var') else False,
        
        dynamic_filters=dynamic_filters,
        scatter_selection=scatter_selection,
        
        sort_column=dialog.sort_column_var.get() if hasattr(dialog, 'sort_column_var') else "depth_to",
        sort_order=dialog.sort_order_var.get() if hasattr(dialog, 'sort_order_var') else "asc",
        secondary_sort_column=dialog.sort_column_var_secondary.get() if hasattr(dialog, 'sort_column_var_secondary') else None,
        secondary_sort_order=dialog.sort_order_var_secondary.get() if hasattr(dialog, 'sort_order_var_secondary') else "asc",
        
        add_placeholders=(display_mode == DisplayMode.ALL_INTERVALS),
    )
    
    return criteria