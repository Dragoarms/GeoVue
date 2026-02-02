"""
register_store.py - JSON register data access with caching.

This module provides:
- RegisterStore: Unified interface to JSON register data
- Access to classifications, tags, comments
- Access to image properties (hex colors wet/dry/combined)
- Review metadata aggregation (consensus, agreement)

Wraps the existing JSONRegisterManager with caching and key-based lookups.

The RegisterStore does NOT own the data - it delegates to JSONRegisterManager
but provides:
1. Key-based lookups matching our ImageKey system
2. Caching of computed values (consensus, agreement)
3. Batch operations for efficiency

Author: George Symonds
"""

import logging
import time
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import pandas as pd
from processing.DataManager.keys import ImageKey

logger = logging.getLogger(__name__)


@dataclass
class ReviewMetadata:
    """
    Aggregated review metadata for an image.
    
    Computed from all users' reviews for a single image interval.
    
    Attributes:
        classification: The current user's classification (or None)
        classified_by: Username who made current user's classification
        tags: Set of tag IDs applied by current user
        comments: Current user's comments
        consensus_classification: Most common classification across all reviewers
        review_count: Number of reviewers who classified this image
        agreement: Agreement level ("unanimous", "majority", "split", "none")
        all_classifications: List of all classifications from all reviewers
        all_reviewers: List of usernames who reviewed (parallel to all_classifications)
        all_comments: List of all comments from all reviewers (parallel to all_reviewers)
        all_tags: Set of all tags from all reviewers
    """
    classification: Optional[str] = None
    classified_by: str = ""
    tags: Set[str] = field(default_factory=set)
    comments: str = ""
    consensus_classification: Optional[str] = None
    review_count: int = 0
    agreement: str = "none"
    all_classifications: List[str] = field(default_factory=list)
    all_reviewers: List[str] = field(default_factory=list)
    all_comments: List[str] = field(default_factory=list)
    all_tags: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for filter DataFrame population."""
        return {
            "classification": self.classification or "",
            "classified_by": self.classified_by or "",
            "tags": self.tags,
            "comments": self.comments,
            "consensus_classification": self.consensus_classification or "",
            "review_count": self.review_count,
            "agreement": self.agreement,
            "all_classifications": ",".join(self.all_classifications) if self.all_classifications else "",
            "all_reviewers": ",".join(self.all_reviewers) if self.all_reviewers else "",
            "all_comments": " | ".join(self.all_comments) if self.all_comments else "",
        }


@dataclass
class ImageProperties:
    """
    Image properties from the image properties register.
    
    Contains computed values like hex colors for wet/dry/combined images.
    
    Attributes:
        wet_hex: Hex color string for wet image (e.g., "#8B4513")
        dry_hex: Hex color string for dry image
        combined_hex: Averaged hex color
        has_wet: Whether wet image exists
        has_dry: Whether dry image exists
    """
    wet_hex: Optional[str] = None
    dry_hex: Optional[str] = None
    combined_hex: Optional[str] = None
    has_wet: bool = False
    has_dry: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "wet_hex": self.wet_hex,
            "dry_hex": self.dry_hex,
            "combined_hex": self.combined_hex,
            "has_wet": self.has_wet,
            "has_dry": self.has_dry,
        }


class RegisterStore:
    """
    Provides cached access to JSON register data.
    
    Wraps JSONRegisterManager to provide:
    - ImageKey-based lookups
    - Cached review metadata computation
    - Batch retrieval for efficiency
    
    Usage:
        >>> store = RegisterStore(json_register_manager)
        >>> store.build_cache()
        >>> 
        >>> # Get review data for an image
        >>> metadata = store.get_review_metadata(ImageKey("BA0001", 45.0))
        >>> print(metadata.consensus_classification)
        >>> 
        >>> # Get hex colors
        >>> props = store.get_image_properties(ImageKey("BA0001", 45.0))
        >>> print(props.wet_hex)
    """
    def __init__(self, json_manager=None, geological_store=None):
        """
        Initialize the register store.

        Args:
            json_manager: JSONRegisterManager instance for data access
            geological_store: Optional GeologicalStore for looking up depth_from
        """
        self._manager = json_manager
        self._geological_store = geological_store

        # Caches
        self._review_cache: Dict[Tuple[str, int], ReviewMetadata] = {}
        self._properties_cache: Dict[Tuple[str, int], ImageProperties] = {}

        # Cache state
        self._cache_built = False
        self._cache_build_time: float = 0
        self._cached_image_count: int = 0

        logger.debug("RegisterStore initialized")
    
    def set_manager(self, json_register_manager) -> "RegisterStore":
        """
        Set or replace the JSONRegisterManager.
        
        Args:
            json_register_manager: JSONRegisterManager instance
            
        Returns:
            self (for chaining)
        """
        self._manager = json_register_manager
        self._cache_built = False  # Invalidate cache
        logger.debug("RegisterStore manager set")
        return self
    
    @property
    def has_manager(self) -> bool:
        """Whether a JSONRegisterManager is configured."""
        return self._manager is not None
    
    @property
    def is_cache_built(self) -> bool:
        """Whether the cache has been built."""
        return self._cache_built
    
    # =========================================================================
    # Cache Building
    # =========================================================================
    
    def build_cache(self, image_keys: Optional[List[ImageKey]] = None) -> "RegisterStore":
        """
        Build or rebuild the cache.
        
        If image_keys is provided, only caches those specific keys.
        Otherwise, caches all images found in the registers.
        
        Args:
            image_keys: Optional list of specific keys to cache
            
        Returns:
            self (for chaining)
        """
        if not self._manager:
            logger.warning("Cannot build cache: no JSONRegisterManager configured")
            return self
        
        start_time = time.time()
        
        logger.info("=" * 60)
        logger.info("BUILDING REGISTER CACHE")
        logger.info("=" * 60)
        
        # Clear existing caches
        self._review_cache.clear()
        self._properties_cache.clear()
        
        if image_keys:
            # Cache specific keys
            logger.info(f"Caching {len(image_keys):,} specific image keys")
            for key in image_keys:
                self._cache_single_image(key)
        else:
            # Cache all images from compartment register
            self._cache_all_from_registers()
        
        self._cache_built = True
        self._cache_build_time = time.time() - start_time
        self._cached_image_count = len(self._review_cache)
        
        logger.info("=" * 60)
        logger.info(f"REGISTER CACHE BUILT")
        logger.info(f"  Images cached: {self._cached_image_count:,}")
        logger.info(f"  Build time: {self._cache_build_time:.2f}s")
        logger.info("=" * 60)
        
        return self
    
    def build_cache_from_review_dicts(
        self,
        user_reviews: Dict[Tuple, Dict],
        other_reviews: Dict[Tuple, List[Dict]],
        current_user: str
    ) -> "RegisterStore":
        """
        Build cache directly from pre-loaded review dicts (O(n) - no file I/O).
        
        This is the efficient path when reviews have already been loaded from JSON files.
        
        Args:
            user_reviews: {(hole_id, from, to): review_dict} for current user
            other_reviews: {(hole_id, from, to): [review_dict, ...]} for other users
            current_user: Current username for identifying user's own reviews
            
        Returns:
            self (for chaining)
        """
        start_time = time.time()
        
        logger.info("=" * 60)
        logger.info("BUILDING REGISTER CACHE FROM REVIEW DICTS")
        logger.info("=" * 60)
        logger.info(f"  user_reviews: {len(user_reviews):,} entries")
        logger.info(f"  other_reviews: {len(other_reviews):,} entries")
        
        # Clear existing caches
        self._review_cache.clear()
        self._properties_cache.clear()
        
        # Collect all unique keys from both dicts
        all_keys = set(user_reviews.keys()) | set(other_reviews.keys())
        logger.info(f"  Unique intervals: {len(all_keys):,}")
        
        count = 0
        for dict_key in all_keys:
            try:
                hole_id, depth_from, depth_to = dict_key
                
                # Convert to cache key format: (HOLE_ID_UPPER, depth_to_int)
                cache_key = (str(hole_id).upper(), int(depth_to))
                
                # Get current user's review (if exists)
                user_review = user_reviews.get(dict_key)
                
                # Get other users' reviews (if any)
                others = other_reviews.get(dict_key, [])
                
                # Extract current user's classification and tags
                classification = None
                tags = set()
                comments = ""
                classified_by = ""
                
                if user_review:
                    # Try multiple field names for classification
                    classification = (
                        user_review.get("classification") or
                        user_review.get("Classification") or
                        user_review.get("Lithology")
                    )
                    if classification in ("", "Unassigned", None):
                        classification = None
                    
                    # Get tags (may be list or absent)
                    tags_value = user_review.get("tags", [])
                    if isinstance(tags_value, list):
                        tags = set(tags_value)
                    
                    comments = user_review.get("Comments", "") or user_review.get("comments", "") or ""
                    classified_by = user_review.get("classified_by", "") or user_review.get("Reviewed_By", "") or ""
                
                # Collect ALL classifications for consensus calculation
                all_classifications = []
                all_reviewers = []
                all_comments = []
                all_tags = set(tags)  # Start with current user's tags
                
                # Add current user's classification and comments
                if classification:
                    all_classifications.append(classification)
                    all_reviewers.append(current_user)
                    all_comments.append(comments)
                
                # Add other users' classifications
                for other_review in others:
                    other_cls = (
                        other_review.get("classification") or
                        other_review.get("Classification") or
                        other_review.get("Lithology")
                    )
                    
                    # Handle legacy boolean format (e.g., 'BIFf': True)
                    if not other_cls:
                        for key, val in other_review.items():
                            if val is True and key not in (
                                'Bad Image', '_source_file', '_file_user',
                                'HoleID', 'From', 'To', 'Reviewed_By'
                            ):
                                other_cls = key
                                break
                    
                    if other_cls and other_cls not in ("", "Unassigned"):
                        all_classifications.append(other_cls)
                        reviewer = other_review.get("_file_user") or other_review.get("Reviewed_By", "unknown")
                        all_reviewers.append(reviewer)
                        other_comment = other_review.get("Comments", "") or other_review.get("comments", "") or ""
                        all_comments.append(other_comment)
                    
                    # Collect other tags
                    other_tags = other_review.get("tags", [])
                    if isinstance(other_tags, list):
                        all_tags.update(other_tags)
                
                # Compute consensus
                consensus_classification = None
                agreement = "none"
                review_count = len(all_classifications)
                
                if all_classifications:
                    counter = Counter(all_classifications)
                    most_common = counter.most_common(1)[0]
                    consensus_classification = most_common[0]
                    consensus_count = most_common[1]
                    
                    if review_count == 1:
                        agreement = "single"
                    elif consensus_count == review_count:
                        agreement = "unanimous"
                    elif consensus_count > review_count / 2:
                        agreement = "majority"
                    else:
                        agreement = "split"
                
                # Create and store ReviewMetadata
                metadata = ReviewMetadata(
                    classification=classification,
                    classified_by=classified_by,
                    tags=tags,
                    comments=comments,
                    consensus_classification=consensus_classification,
                    review_count=review_count,
                    agreement=agreement,
                    all_classifications=all_classifications,
                    all_reviewers=all_reviewers,
                    all_comments=all_comments,
                    all_tags=all_tags,
                )
                
                self._review_cache[cache_key] = metadata
                count += 1
                
                if count % 25000 == 0:
                    logger.debug(f"  Processed {count:,} intervals...")
                    
            except Exception as e:
                logger.debug(f"Error processing review key {dict_key}: {e}")
        
        self._cache_built = True
        self._cache_build_time = time.time() - start_time
        self._cached_image_count = count
        
        # Log classification distribution
        cls_counts = Counter(
            m.classification for m in self._review_cache.values() if m.classification
        )
        logger.info(f"  Classification distribution: {dict(cls_counts)}")
        
        logger.info("=" * 60)
        logger.info(f"REGISTER CACHE BUILT FROM DICTS")
        logger.info(f"  Intervals cached: {self._cached_image_count:,}")
        logger.info(f"  Build time: {self._cache_build_time:.2f}s")
        logger.info("=" * 60)
        
        return self

    def _cache_all_from_registers(self):
        """Cache all images found in the compartment register."""
        if not self._manager:
            logger.warning("No register manager available for caching")
            return
            
        try:
            # Use get_all_compartments_all_users() to get DataFrame of all compartments
            # This is the correct method from JSONRegisterManager
            df = self._manager.get_all_compartments_all_users()
            
            if df is None or df.empty:
                logger.warning("Compartment register is empty or unavailable")
                return
            
            count = 0
            # DataFrame has columns: hole_id, depth_from, depth_to, photo_status, etc.
            for _, row in df.iterrows():
                try:
                    hole_id = str(row.get('hole_id', '')).upper()
                    depth_to = float(row.get('depth_to', 0))
                    
                    if hole_id and depth_to > 0:
                        key = ImageKey(hole_id, depth_to)
                        self._cache_single_image(key)
                        count += 1
                        
                        if count % 10000 == 0:
                            logger.debug(f"  Cached {count:,} images...")
                            
                except (ValueError, TypeError) as e:
                    logger.debug(f"Skipping invalid row: {e}")
            
            logger.info(f"  Cached {count:,} images from compartment register")
            
        except Exception as e:
            logger.error(f"Error caching from registers: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def _cache_single_image(self, key: ImageKey):
        """
        Cache review metadata and properties for a single image.
        
        Args:
            key: ImageKey to cache
        """
        cache_key = key.to_base_tuple()
        
        # Cache review metadata
        metadata = self._compute_review_metadata(key)
        self._review_cache[cache_key] = metadata
        
        # Cache image properties
        properties = self._compute_image_properties(key)
        self._properties_cache[cache_key] = properties
    
    def invalidate_cache(self, key: Optional[ImageKey] = None):
        """
        Invalidate cache for a specific key or all keys.
        
        Args:
            key: Optional specific key to invalidate. If None, invalidates all.
        """
        if key is None:
            self._review_cache.clear()
            self._properties_cache.clear()
            self._cache_built = False
            logger.debug("Register cache fully invalidated")
        else:
            cache_key = key.to_base_tuple()
            self._review_cache.pop(cache_key, None)
            self._properties_cache.pop(cache_key, None)
            logger.debug(f"Register cache invalidated for {key}")
    
    def refresh_key(self, key: ImageKey):
        """
        Refresh cache for a specific key (after edits).
        
        Args:
            key: ImageKey to refresh
        """
        self._cache_single_image(key)
        logger.debug(f"Register cache refreshed for {key}")
    
    # =========================================================================
    # Review Metadata
    # =========================================================================
    
    def get_review_metadata(self, key: ImageKey, use_cache: bool = True) -> ReviewMetadata:
        """
        Get aggregated review metadata for an image.
        
        Args:
            key: ImageKey to look up
            use_cache: Whether to use cached value if available
            
        Returns:
            ReviewMetadata with classification, consensus, agreement, etc.
        """
        cache_key = key.to_base_tuple()
        
        if use_cache and cache_key in self._review_cache:
            return self._review_cache[cache_key]
        
        # Compute fresh
        metadata = self._compute_review_metadata(key)
        
        # Update cache
        self._review_cache[cache_key] = metadata
        
        return metadata
    
    def set_geological_store(self, geological_store) -> None:
        """
        Set the geological store reference for depth_from lookups.
        
        Args:
            geological_store: GeologicalStore instance
        """
        self._geological_store = geological_store
        logger.debug("Geological store reference set for interval lookups")

    def _compute_review_metadata(self, key: ImageKey) -> ReviewMetadata:
        """
        Compute review metadata by aggregating all users' reviews.
        
        Args:
            key: ImageKey to compute metadata for
            
        Returns:
            ReviewMetadata instance
        """
        if not self._manager:
            return ReviewMetadata()
        
        try:
            hole_id = key.hole_id
            depth_to = key.depth_to
            
            # Get current user's review using get_user_review(hole_id, depth_from, depth_to)
            depth_from = int(self._get_depth_from(key))
            user_review = self._manager.get_user_review(hole_id, depth_from, int(depth_to))
            
            # Get all reviews (including other users)
            all_reviews = self._manager.get_all_reviews_for_compartment(hole_id, depth_from, int(depth_to))
            
            # Extract current user's data
            classification = None
            tags = set()
            comments = ""
            
            if user_review:
                classification = user_review.get("classification")
                tags = set(user_review.get("tags", []))
                comments = user_review.get("comments", "")
            
            # Compute consensus from all reviews
            all_classifications = []
            all_tags = set()
            
            if all_reviews:
                for review in all_reviews:
                    if isinstance(review, dict):
                        cls = review.get("classification")
                        if cls:
                            all_classifications.append(cls)
                        review_tags = review.get("tags", [])
                        if review_tags:
                            all_tags.update(review_tags)
            
            # Compute consensus
            consensus_classification = None
            agreement = "none"
            review_count = len(all_classifications)
            
            if all_classifications:
                counter = Counter(all_classifications)
                most_common = counter.most_common(1)[0]
                consensus_classification = most_common[0]
                consensus_count = most_common[1]
                
                # Determine agreement level
                if review_count == 1:
                    agreement = "single"
                elif consensus_count == review_count:
                    agreement = "unanimous"
                elif consensus_count > review_count / 2:
                    agreement = "majority"
                else:
                    agreement = "split"
            
            return ReviewMetadata(
                classification=classification,
                tags=tags,
                comments=comments,
                consensus_classification=consensus_classification,
                review_count=review_count,
                agreement=agreement,
                all_classifications=all_classifications,
                all_tags=all_tags,
            )
            
        except Exception as e:
            logger.debug(f"Error computing review metadata for {key}: {e}")
            return ReviewMetadata()
    
    def get_classification(self, key: ImageKey) -> Optional[str]:
        """Get current user's classification for an image."""
        metadata = self.get_review_metadata(key)
        return metadata.classification
    
    def get_tags(self, key: ImageKey) -> Set[str]:
        """Get current user's tags for an image."""
        metadata = self.get_review_metadata(key)
        return metadata.tags
    
    def get_consensus_classification(self, key: ImageKey) -> Optional[str]:
        """Get consensus classification across all reviewers."""
        metadata = self.get_review_metadata(key)
        return metadata.consensus_classification
    
    def get_all_reviewer_data(self, key: ImageKey) -> Dict[str, Dict[str, Any]]:
        """
        Get review data broken down by reviewer.
        
        Args:
            key: ImageKey to look up
            
        Returns:
            Dictionary mapping reviewer username to their review data:
            {
                "gsymonds": {"classification": "BIFf", "comments": "...", "tags": [...]},
                "chrithompso": {"classification": "BIFhm", "comments": "...", "tags": [...]}
            }
        """
        if not self._manager:
            return {}
        
        try:
            hole_id = key.hole_id
            depth_to = key.depth_to
            depth_from = int(self._get_depth_from(key))
            
            # Get all reviews for this compartment
            all_reviews = self._manager.get_all_reviews_for_compartment(
                hole_id, depth_from, int(depth_to)
            )
            
            reviewer_data = {}
            
            if all_reviews:
                for review in all_reviews:
                    if isinstance(review, dict):
                        # Extract reviewer identifier (multiple possible fields)
                        reviewer = (
                            review.get("Reviewed_By") or
                            review.get("_file_user") or
                            review.get("reviewer") or
                            review.get("classified_by")
                        )
                        
                        if reviewer:
                            reviewer_data[reviewer] = {
                                "classification": review.get("classification", ""),
                                "comments": review.get("comments", ""),
                                "tags": review.get("tags", [])
                            }
            
            return reviewer_data
            
        except Exception as e:
            logger.debug(f"Error getting reviewer data for {key}: {e}")
            return {}

    def is_classified(self, key: ImageKey) -> bool:
        """Check if the image has been classified by current user."""
        metadata = self.get_review_metadata(key)
        return metadata.classification is not None and metadata.classification != ""
    
    def has_conflict(self, key: ImageKey) -> bool:
        """Check if there's disagreement among reviewers."""
        metadata = self.get_review_metadata(key)
        return metadata.agreement == "split"
    
    # =========================================================================
    # Image Properties
    # =========================================================================
    
    def get_image_properties(self, key: ImageKey, use_cache: bool = True) -> ImageProperties:
        """
        Get image properties (hex colors, etc.) for an image.
        
        Args:
            key: ImageKey to look up
            use_cache: Whether to use cached value if available
            
        Returns:
            ImageProperties with hex colors and flags
        """
        cache_key = key.to_base_tuple()
        
        if use_cache and cache_key in self._properties_cache:
            return self._properties_cache[cache_key]
        
        # Compute fresh
        properties = self._compute_image_properties(key)
        
        # Update cache
        self._properties_cache[cache_key] = properties
        
        return properties
    
    def _get_depth_from(self, key: ImageKey) -> float:
        """
        Get the depth_from for a key, using geological data if available.
        
        Args:
            key: ImageKey with hole_id and depth_to
            
        Returns:
            depth_from value (from CSV data or calculated as depth_to - 1.0)
        """
        # Try to get actual depth_from from geological store
        if self._geological_store:
            depth_from = self._geological_store.get_depth_from(key.hole_id, key.depth_to)
            if depth_from is not None:
                return depth_from
        
        # Fallback: assume 1m intervals
        return key.depth_to - 1.0
    
    def _compute_image_properties(self, key: ImageKey) -> ImageProperties:
        """
        Compute image properties from the image properties register.
        
        Args:
            key: ImageKey to compute properties for
            
        Returns:
            ImageProperties instance
        """
        if not self._manager:
            return ImageProperties()
        
        try:
            hole_id = key.hole_id
            depth_from = self._get_depth_from(key)  # Use actual interval from CSV
            depth_to = key.depth_to
            
            # Get hex colors from register - returns dict with 'wet_hex', 'dry_hex', 'combined_hex'
            hex_colors = self._manager.get_hex_colors_for_interval(
                hole_id, depth_from, depth_to
            )
            
            if not hex_colors:
                return ImageProperties()
            
            # Extract from dictionary (not tuple)
            wet_hex = hex_colors.get('wet_hex', '')
            dry_hex = hex_colors.get('dry_hex', '')
            combined_hex = hex_colors.get('combined_hex', '')
            
            return ImageProperties(
                wet_hex=wet_hex,
                dry_hex=dry_hex,
                combined_hex=combined_hex,
                has_wet=bool(wet_hex),  # Check for non-empty string
                has_dry=bool(dry_hex),
            )
            
        except Exception as e:
            logger.debug(f"Error computing image properties for {key}: {e}")
            return ImageProperties()
    
    def get_hex_color(self, key: ImageKey, color_type: str = "combined") -> Optional[str]:
        """
        Get a specific hex color for an image.

        Args:
            key: ImageKey to look up
            color_type: One of "wet", "dry", "combined"

        Returns:
            Hex color string or None
        """
        props = self.get_image_properties(key)

        if color_type == "wet":
            return props.wet_hex
        elif color_type == "dry":
            return props.dry_hex
        else:
            return props.combined_hex

    # =========================================================================
    # Query
    # =========================================================================

    def query(
        self,
        filters: List[Dict[str, Any]]
    ) -> Set[Tuple[str, float]]:
        """
        Query cached register data with filter criteria.
        
        Args:
            filters: List of filter dicts, each with:
                - column: str - Column name (classification, review_count, etc.)
                - operator: str - Operator
                - value: Any - Filter value
                - value2: Any - Second value for 'between'
                
        Returns:
            Set of (hole_id_upper, depth_to) tuples matching all filters
        """
        logger.info("=" * 60)
        logger.info("REGISTERSTORE.QUERY() CALLED")
        logger.info(f"  Filters: {filters}")
        logger.info(f"  Review cache size: {len(self._review_cache)}")
        logger.info(f"  Properties cache size: {len(self._properties_cache)}")
        
        if not filters:
            # Return all cached keys
            all_keys = set(self._review_cache.keys()) | set(self._properties_cache.keys())
            logger.info(f"  No filters - returning all {len(all_keys)} keys")
            return all_keys
        
        matching_keys = None
        
        for flt_idx, flt in enumerate(filters):
            col = flt.get('column', '').lower()
            op = flt.get('operator', '')
            val = flt.get('value')
            val2 = flt.get('value2')
            
            logger.info(f"  Filter {flt_idx}: column={col}, operator={op}, value={val}")
            
            passing_keys = set()
            
            # Check review cache columns
            if col in ('classification', 'classified_by', 'consensus_classification', 
                       'all_classifications', 'all_reviewers', 'review_count', 
                       'agreement', 'comments') or col.startswith('tag_') or col.startswith('rev_'):
                
                logger.info(f"    Checking {len(self._review_cache)} review cache entries...")
                
                # Sample first 5 entries for debugging
                sample_count = 0
                for cache_key, metadata in self._review_cache.items():
                    matches = self._check_metadata_match(metadata, col, op, val, val2)
                    
                    if sample_count < 5:
                        logger.info(f"      Sample {cache_key}: all_classifications={metadata.all_classifications}, matches={matches}")
                        sample_count += 1
                    
                    if matches:
                        passing_keys.add(cache_key)
                
                logger.info(f"    Review cache matches: {len(passing_keys)}")
            
            # Check properties cache columns
            elif col in ('wet_hex', 'dry_hex', 'combined_hex', 'hex_color',
                         'average hex colour', 'has_wet', 'has_dry'):
                for cache_key, props in self._properties_cache.items():
                    if self._check_properties_match(props, col, op, val, val2):
                        passing_keys.add(cache_key)

            # Intersect with previous results (AND logic)
            if matching_keys is None:
                matching_keys = passing_keys
            else:
                matching_keys &= passing_keys
        
        return matching_keys if matching_keys is not None else set()
    
    def _check_metadata_match(
        self, 
        metadata: ReviewMetadata, 
        column: str, 
        operator: str, 
        value: Any, 
        value2: Any = None
    ) -> bool:
        """Check if review metadata matches a filter criterion."""
        col_lower = column.lower()
        
        logger.debug(f"[MATCH_CHECK] column={col_lower}, operator={operator}, value={value}")
        
        # Get the value to compare
        if col_lower == 'classification':
            data_val = metadata.classification or ""
            logger.debug(f"  classification: {data_val}")
        elif col_lower == 'classified_by':
            data_val = metadata.classified_by or ""
            logger.debug(f"  classified_by: {data_val}")
        elif col_lower == 'consensus_classification':
            data_val = metadata.consensus_classification or ""
            logger.debug(f"  consensus_classification: {data_val}")
        elif col_lower == 'all_classifications':
            # Check if filter value is in the list of all classifications
            logger.debug(f"  all_classifications list: {metadata.all_classifications}")
            
            # For 'in' or 'contains' operator: check if filter_val is in the list
            if operator in ('in', 'contains'):
                classifications_lower = [c.lower() for c in metadata.all_classifications]
                result = value.lower() in classifications_lower
                logger.debug(f"  Checking if '{value}' in {metadata.all_classifications} → {result}")
                return result
            else:
                # For other operators, join and use normal logic
                data_val = ','.join(metadata.all_classifications)
                logger.debug(f"  all_classifications joined: {data_val}")
        elif col_lower == 'all_reviewers':
            # Check if value is in the list of all reviewers
            logger.debug(f"  all_reviewers list: {metadata.all_reviewers}")
            
            # For 'in' or 'contains' operator: check if filter_val is in the list
            if operator in ('in', 'contains'):
                reviewers_lower = [r.lower() for r in metadata.all_reviewers]
                result = value.lower() in reviewers_lower
                logger.debug(f"  Checking if '{value}' in {metadata.all_reviewers} → {result}")
                return result
            else:
                # For other operators, join and use normal logic
                data_val = ','.join(metadata.all_reviewers)
                logger.debug(f"  all_reviewers joined: {data_val}")
        elif col_lower == 'all_comments':
            # Check comments from all reviewers
            logger.debug(f"  all_comments list: {metadata.all_comments}")
            
            if operator in ('in', 'contains'):
                # Check if value appears in any comment
                for comment in metadata.all_comments:
                    if value.lower() in comment.lower():
                        return True
                return False
            else:
                # For other operators, join with separator
                data_val = ' | '.join(metadata.all_comments)
        elif col_lower == 'review_count':
            data_val = metadata.review_count
        elif col_lower == 'agreement':
            data_val = metadata.agreement
        elif col_lower == 'comments':
            data_val = metadata.comments or ""
        elif col_lower.startswith('tag_'):
            tag_name = column[4:]
            data_val = tag_name in metadata.tags
        elif col_lower.startswith('rev_'):
            logger.error(f"  col_lower.startswith('rev_') - RETURNING ERROR")
            return False
        else:
            return False
        
        return self._apply_operator(data_val, operator, value, value2)
    
    def _check_properties_match(
        self,
        props: ImageProperties,
        column: str,
        operator: str,
        value: Any,
        value2: Any = None
    ) -> bool:
        """Check if image properties match a filter criterion."""
        col_lower = column.lower()

        if col_lower == 'wet_hex':
            data_val = props.wet_hex or ""
        elif col_lower == 'dry_hex':
            data_val = props.dry_hex or ""
        elif col_lower in ('combined_hex', 'hex_color', 'average hex colour'):
            data_val = props.combined_hex or ""
        elif col_lower == 'has_wet':
            data_val = props.has_wet
        elif col_lower == 'has_dry':
            data_val = props.has_dry
        else:
            return False

        return self._apply_operator(data_val, operator, value, value2)

    def _apply_operator(
        self,
        data_val: Any,
        operator: str,
        filter_val: Any,
        filter_val2: Any = None
    ) -> bool:
        """Apply comparison operator."""
        logger.debug(f"[APPLY_OP] data_val={data_val!r}, operator={operator!r}, filter_val={filter_val!r}")
        
        if data_val is None:
            result = operator in ('is null', 'is empty')
            logger.debug(f"  data_val is None → {result}")
            return result
        
        try:
            if operator in ('=', 'equals', '=='):
                result = str(data_val).lower() == str(filter_val).lower()
                logger.debug(f"  equals: '{data_val}' == '{filter_val}' → {result}")
                return result
            elif operator in ('!=', '≠', 'not equals'):
                return str(data_val).lower() != str(filter_val).lower()
            elif operator in ('>', 'greater than'):
                return float(data_val) > float(filter_val)
            elif operator in ('>=', '≥', 'greater than or equal'):
                return float(data_val) >= float(filter_val)
            elif operator in ('<', 'less than'):
                return float(data_val) < float(filter_val)
            elif operator in ('<=', '≤', 'less than or equal'):
                return float(data_val) <= float(filter_val)
            elif operator == 'between':
                return float(filter_val) <= float(data_val) <= float(filter_val2)
            elif operator == 'contains':
                result = str(filter_val).lower() in str(data_val).lower()
                logger.debug(f"  contains: '{filter_val}' in '{data_val}' → {result}")
                return result
            elif operator == 'not contains':
                return str(filter_val).lower() not in str(data_val).lower()
            elif operator in ('in', 'in list'):
                val_list = [v.strip().lower() for v in str(filter_val).split(',')]
                result = str(data_val).lower() in val_list
                logger.debug(f"  in: '{data_val}' in {val_list} → {result}")
                return result
            elif operator in ('not in', 'not in list'):
                val_list = [v.strip().lower() for v in str(filter_val).split(',')]
                return str(data_val).lower() not in val_list
            elif operator in ('is null', 'is empty'):
                return data_val is None or data_val == '' or data_val == 0
            elif operator in ('not null', 'is not empty'):
                return data_val is not None and data_val != '' and data_val != 0
        except (ValueError, TypeError):
            return False
        
        return False
    
    def get_all_cached_keys(self) -> Set[Tuple[str, float]]:
        """Get all keys that have been cached."""
        return set(self._review_cache.keys()) | set(self._properties_cache.keys())
    
    def get_dataframe_for_columns(
        self,
        columns: List[str],
        keys: Optional[Set[Tuple[str, float]]] = None
    ) -> pd.DataFrame:
        """
        Build a DataFrame with register data for specified columns.
        
        Args:
            columns: Column names to include
            keys: Optional set of keys to filter by
            
        Returns:
            DataFrame with hole_id, depth_to, and requested columns
        """
        
        
        rows = []
        keys_to_use = keys if keys else self.get_all_cached_keys()
        
        for cache_key in keys_to_use:
            hole_id, depth_to = cache_key
            row = {'hole_id': hole_id, 'depth_to': depth_to}
            
            # Get review metadata
            metadata = self._review_cache.get(cache_key)
            if metadata:
                if 'classification' in columns:
                    row['classification'] = metadata.classification or ""
                if 'consensus_classification' in columns:
                    row['consensus_classification'] = metadata.consensus_classification or ""
                if 'review_count' in columns:
                    row['review_count'] = metadata.review_count
                if 'agreement' in columns:
                    row['agreement'] = metadata.agreement
                if 'comments' in columns:
                    row['comments'] = metadata.comments or ""
                if 'tags' in columns:
                    row['tags'] = ','.join(metadata.tags) if metadata.tags else ""
            
            # Get image properties
            props = self._properties_cache.get(cache_key)
            if props:
                if 'hex_color' in columns or 'combined_hex' in columns:
                    row['hex_color'] = props.combined_hex or ""
                if 'wet_hex' in columns:
                    row['wet_hex'] = props.wet_hex or ""
                if 'dry_hex' in columns:
                    row['dry_hex'] = props.dry_hex or ""

            rows.append(row)

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # =========================================================================
    # Batch Operations
    # =========================================================================
    
    def get_reviews_for_hole(self, hole_id: str) -> Dict[int, ReviewMetadata]:
        """
        Get all review metadata for a hole.
        
        Args:
            hole_id: Hole identifier
            
        Returns:
            Dictionary of {depth_to: ReviewMetadata}
        """
        result = {}
        hole_upper = hole_id.upper()
        
        for cache_key, metadata in self._review_cache.items():
            if cache_key[0] == hole_upper:
                depth_to = cache_key[1]
                result[depth_to] = metadata
        
        return result
    
    def get_unclassified_keys(self, hole_id: Optional[str] = None) -> List[ImageKey]:
        """
        Get keys for images that haven't been classified.
        
        Args:
            hole_id: Optional hole to filter by
            
        Returns:
            List of ImageKeys for unclassified images
        """
        result = []
        
        for cache_key, metadata in self._review_cache.items():
            if hole_id and cache_key[0] != hole_id.upper():
                continue
            
            if not metadata.classification:
                result.append(ImageKey(cache_key[0], float(cache_key[1])))
        
        return result
    
    def get_conflict_keys(self, hole_id: Optional[str] = None) -> List[ImageKey]:
        """
        Get keys for images with classification conflicts.
        
        Args:
            hole_id: Optional hole to filter by
            
        Returns:
            List of ImageKeys with conflicts
        """
        result = []
        
        for cache_key, metadata in self._review_cache.items():
            if hole_id and cache_key[0] != hole_id.upper():
                continue
            
            if metadata.agreement == "split":
                result.append(ImageKey(cache_key[0], float(cache_key[1])))
        
        return result
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the register store."""
        classified_count = sum(
            1 for m in self._review_cache.values() 
            if m.classification
        )
        conflict_count = sum(
            1 for m in self._review_cache.values() 
            if m.agreement == "split"
        )
        
        return {
            "has_manager": self.has_manager,
            "cache_built": self._cache_built,
            "cached_images": self._cached_image_count,
            "cache_build_time": self._cache_build_time,
            "classified_count": classified_count,
            "unclassified_count": self._cached_image_count - classified_count,
            "conflict_count": conflict_count,
        }
    
    def get_classification_counts(self) -> Dict[str, int]:
        """Get counts of each classification."""
        counts = Counter()
        
        for metadata in self._review_cache.values():
            if metadata.classification:
                counts[metadata.classification] += 1
            else:
                counts["unclassified"] += 1
        
        return dict(counts)
