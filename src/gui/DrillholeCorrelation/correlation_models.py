"""
Data models for drillhole correlation system.
Defines TieLine and DrillholeInterval dataclasses with flexible typing.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Tuple
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class TieLineType(Enum):
    """Types of visual correlation tie lines for aligning segments"""
    LITHOLOGY = "lithology"
    STRUCTURE = "structure"
    GRADE = "grade"
    ALTERATION = "alteration"
    MARKER = "marker"
    FACIES = "facies"  # Sedimentological facies correlations
    CUSTOM = "custom"


class DiscontinuityType(Enum):
    """Types of discontinuities that split drillholes into segments"""
    FAULT = "fault"
    INTRUSIVE_CONTACT = "intrusive"
    UNCONFORMITY = "unconformity"
    DETRITAL_SEQUENCE = "detrital"
    LENS = "lens"  # Laterally discontinuous units
    CORE_LOSS = "core_loss"
    WEATHERING = "weathering"
    CUSTOM = "custom"


class FeatureCategory(Enum):
    """Categories for geological features"""
    STRUCTURAL = "structural"  # Faults, intrusives, unconformities
    GEOCHEMICAL = "geochemical"  # Alteration, weathering
    LITHOLOGICAL = "lithological"  # Facies, lenses, marker beds
    CUSTOM = "custom"


class StretchMode(Enum):
    """Modes for depth stretching/compression"""
    LINEAR = "linear"  # Apply uniform stretch
    PROGRESSIVE = "progressive"  # Gradual change
    PINNED = "pinned"  # Fixed points with stretch between


@dataclass
class Feature:
    """
    Represents a geological feature that can be tracked across drillholes.
    Features can be structural (faults, intrusives), geochemical (alteration),
    lithological (facies, lenses), or custom.
    
    Similar to ItemDefinition in image classification system - flexible tagging.
    """
    feature_id: str
    category: FeatureCategory
    name: str  # User-friendly name like "Main Fault", "Chlorite Alteration", "Lens A"
    
    # Visual properties
    color: str = "#FF0000"
    line_style: str = "solid"
    line_width: int = 3
    icon: Optional[str] = None  # Optional emoji/icon
    
    # Organization
    is_active: bool = True  # Whether feature is visible/active
    is_default: bool = False  # Whether this is a built-in feature
    order: int = 0  # Display order
    sequence_number: int = 1  # Auto-naming counter (e.g., "Fault 1", "Fault 2")
    
    # References to where this feature appears
    discontinuity_ids: List[str] = field(default_factory=list)  # Discontinuities that mark this feature
    tie_line_ids: List[str] = field(default_factory=list)  # Tie lines that correlate this feature
    
    # Metadata
    confidence: float = 1.0
    notes: str = ""
    created_by: str = ""
    created_date: str = ""
    
    def __post_init__(self) -> None:
        """Post-initialization validation"""
        logger.info(f"Creating Feature: {self.feature_id} - {self.name} ({self.category.value})")
        logger.debug(f"  Discontinuities: {len(self.discontinuity_ids)}, Tie lines: {len(self.tie_line_ids)}")
        
        # Generate feature_id if not provided
        if not self.feature_id:
            import uuid
            self.feature_id = f"feature_{uuid.uuid4().hex[:8]}"
            logger.debug(f"  Generated feature_id: {self.feature_id}")
        
        # Auto-generate name if not provided
        if not self.name:
            self.name = f"{self.category.value.title()} {self.sequence_number}"
            logger.debug(f"  Generated name: {self.name}")
        
        # Set creation date if not provided
        if not self.created_date:
            self.created_date = datetime.now().isoformat()
        
        # Set default colors by category if using default red
        if self.color == "#FF0000":
            if self.category == FeatureCategory.STRUCTURAL:
                self.color = "#FF0000"  # Red
            elif self.category == FeatureCategory.GEOCHEMICAL:
                self.color = "#00FF00"  # Green
            elif self.category == FeatureCategory.LITHOLOGICAL:
                self.color = "#0000FF"  # Blue
    
    def add_discontinuity(self, discontinuity_id: str) -> None:
        """Add a discontinuity to this feature"""
        if discontinuity_id not in self.discontinuity_ids:
            self.discontinuity_ids.append(discontinuity_id)
            logger.debug(f"Added discontinuity {discontinuity_id} to feature {self.name}")
    
    def remove_discontinuity(self, discontinuity_id: str) -> None:
        """Remove a discontinuity from this feature"""
        if discontinuity_id in self.discontinuity_ids:
            self.discontinuity_ids.remove(discontinuity_id)
            logger.debug(f"Removed discontinuity {discontinuity_id} from feature {self.name}")
    
    def add_tie_line(self, tie_line_id: str) -> None:
        """Add a tie line to this feature"""
        if tie_line_id not in self.tie_line_ids:
            self.tie_line_ids.append(tie_line_id)
            logger.debug(f"Added tie line {tie_line_id} to feature {self.name}")
    
    def remove_tie_line(self, tie_line_id: str) -> None:
        """Remove a tie line from this feature"""
        if tie_line_id in self.tie_line_ids:
            self.tie_line_ids.remove(tie_line_id)
            logger.debug(f"Removed tie line {tie_line_id} from feature {self.name}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'feature_id': self.feature_id,
            'category': self.category.value,
            'name': self.name,
            'color': self.color,
            'line_style': self.line_style,
            'line_width': self.line_width,
            'icon': self.icon,
            'is_active': self.is_active,
            'is_default': self.is_default,
            'order': self.order,
            'sequence_number': self.sequence_number,
            'discontinuity_ids': self.discontinuity_ids,
            'tie_line_ids': self.tie_line_ids,
            'confidence': self.confidence,
            'notes': self.notes,
            'created_by': self.created_by,
            'created_date': self.created_date
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Feature':
        """Create from dictionary"""
        # Convert category string back to enum
        if 'category' in data and isinstance(data['category'], str):
            data['category'] = FeatureCategory(data['category'])
        return cls(**data)

@dataclass
class Discontinuity:
    """
    Represents a discontinuity within a drillhole that splits it into segments.
    Discontinuities do not have stratigraphic position - they mark boundaries.
    Can represent faults, intrusives, unconformities, lenses, etc.
    """
    discontinuity_id: str
    hole_id: str
    discontinuity_type: DiscontinuityType
    
    # Depth where discontinuity occurs in original hole coordinates
    depth_at_boundary: Union[float, int]
    
    # Segments created by this discontinuity
    segment_above_id: str
    segment_below_id: str
    
    # Feature tracking - links this discontinuity to a feature across holes
    feature_id: str = ""
    
    # Visual properties (inherited from feature if linked, otherwise custom)
    color: str = "#FF0000"  # Default red
    line_style: str = "solid"
    line_width: int = 3
    
    # Lens-specific properties
    is_lens: bool = False  # True if this is a lens (laterally discontinuous)
    lens_extent_m: float = 0.0  # Approximate lateral extent for lenses
    
    # Metadata
    confidence: float = 1.0
    notes: str = ""
    created_by: str = ""
    created_date: str = ""
    
    def __post_init__(self) -> None:
        """Post-initialization validation"""
        logger.info(f"Creating Discontinuity: {self.discontinuity_id} ({self.discontinuity_type.value}) in {self.hole_id}")
        logger.debug(f"  Boundary at: {self.depth_at_boundary}m")
        logger.debug(f"  Segments: {self.segment_above_id} | {self.segment_below_id}")
        logger.debug(f"  Feature: {self.feature_id if self.feature_id else 'None'}")
        
        self.depth_at_boundary = float(self.depth_at_boundary)
        
        # Check if this is a lens type
        if self.discontinuity_type == DiscontinuityType.LENS:
            self.is_lens = True
            logger.debug(f"  Lens extent: {self.lens_extent_m}m")
        
        # Generate discontinuity_id if not provided
        if not self.discontinuity_id:
            import uuid
            self.discontinuity_id = f"{self.discontinuity_type.value}_{uuid.uuid4().hex[:8]}"
            logger.debug(f"  Generated discontinuity_id: {self.discontinuity_id}")
        
        # Set creation date if not provided
        if not self.created_date:
            self.created_date = datetime.now().isoformat()
        
        # Set default colors by type if using default
        if self.color == "#FF0000":
            if self.discontinuity_type == DiscontinuityType.FAULT:
                self.color = "#FF0000"  # Red
            elif self.discontinuity_type == DiscontinuityType.INTRUSIVE_CONTACT:
                self.color = "#FF00FF"  # Magenta
            elif self.discontinuity_type == DiscontinuityType.UNCONFORMITY:
                self.color = "#FFA500"  # Orange
            elif self.discontinuity_type == DiscontinuityType.DETRITAL_SEQUENCE:
                self.color = "#8B4513"  # Brown
            elif self.discontinuity_type == DiscontinuityType.LENS:
                self.color = "#00CED1"  # Dark turquoise
            elif self.discontinuity_type == DiscontinuityType.CORE_LOSS:
                self.color = "#808080"  # Gray
            elif self.discontinuity_type == DiscontinuityType.WEATHERING:
                self.color = "#DEB887"  # Burlywood

@dataclass
class DrillholeSegment:
    """
    Represents a continuous segment of stratigraphy between two discontinuities.
    Segments are the basic units of correlation and have stratigraphic positions.
    Multiple tie lines can reference the same segment for reconstructing missing stratigraphy.
    """
    segment_id: str
    hole_id: str
    
    # Original depth range in the physical hole
    depth_from_original: Union[float, int]
    depth_to_original: Union[float, int]
    
    # Stratigraphic position in correlation space (Y coordinate)
    # These are calculated from tie lines and represent the segment's position
    # in "geological time" / stratigraphic column
    stratigraphic_y_top: float = 0.0
    stratigraphic_y_bottom: float = 0.0
    
    # Depth transform for this segment (corrects for down-dip drilling artifacts)
    # Can be calculated automatically from multiple tie lines or set manually
    has_transform: bool = False  # Whether this segment has a depth transform applied
    
    # Segment ordering and discontinuity references
    order_index: int = 0
    discontinuity_above_id: str = ""  # ID of Discontinuity above this segment
    discontinuity_below_id: str = ""  # ID of Discontinuity below this segment
    
    # Segment classification
    is_detrital: bool = False  # Mark detrital sequences at top of hole
    is_lens: bool = False  # Mark lens segments (laterally discontinuous)
    is_excluded: bool = False  # Mark segments to exclude from correlation
    is_reviewed: bool = False  # Has this segment been reviewed for correlation?
    
    # Tie line references - track all tie lines that touch this segment
    tie_line_ids: List[str] = field(default_factory=list)
    
    # Metadata
    notes: str = ""
    created_date: str = ""
    
    def __post_init__(self) -> None:
        """Post-initialization validation"""
        logger.debug(f"Creating DrillholeSegment: {self.segment_id} for {self.hole_id}")
        logger.debug(f"  Depth range: [{self.depth_from_original}-{self.depth_to_original}m]")
        logger.debug(f"  Stratigraphic Y: [{self.stratigraphic_y_top}-{self.stratigraphic_y_bottom}]")
        
        # Ensure depths are numeric and ordered
        self.depth_from_original = float(self.depth_from_original)
        self.depth_to_original = float(self.depth_to_original)
        
        if self.depth_from_original > self.depth_to_original:
            logger.warning(f"Segment {self.segment_id}: depth_from > depth_to, swapping")
            self.depth_from_original, self.depth_to_original = self.depth_to_original, self.depth_from_original
        
        # Generate segment_id if not provided
        if not self.segment_id:
            import uuid
            self.segment_id = f"{self.hole_id}_seg_{uuid.uuid4().hex[:6]}"
            logger.debug(f"  Generated segment_id: {self.segment_id}")
        
        # Set creation date if not provided
        if not self.created_date:
            self.created_date = datetime.now().isoformat()
    
    def contains_depth(self, depth: Union[float, int]) -> bool:
        """Check if a depth falls within this segment's original range"""
        d = float(depth)
        return self.depth_from_original <= d <= self.depth_to_original
    
    def get_segment_height(self) -> float:
        """Get the original height of this segment in meters"""
        return abs(self.depth_to_original - self.depth_from_original)
    
    def get_stratigraphic_height(self) -> float:
        """Get the stratigraphic height in correlation space"""
        return abs(self.stratigraphic_y_bottom - self.stratigraphic_y_top)
    
    def add_tie_line(self, tie_line_id: str) -> None:
        """Register a tie line that references this segment"""
        if tie_line_id not in self.tie_line_ids:
            self.tie_line_ids.append(tie_line_id)
            logger.debug(f"Added tie line {tie_line_id} to segment {self.segment_id}")
    
    def remove_tie_line(self, tie_line_id: str) -> None:
        """Remove a tie line reference from this segment"""
        if tie_line_id in self.tie_line_ids:
            self.tie_line_ids.remove(tie_line_id)
            logger.debug(f"Removed tie line {tie_line_id} from segment {self.segment_id}")
    
    def calculate_stratigraphic_position_from_ties(self, tie_lines: List['TieLine']) -> None:
        """
        Calculate stratigraphic position from multiple tie lines.
        Uses all tie lines that reference this segment to determine position.
        """
        if not tie_lines:
            logger.warning(f"No tie lines provided for segment {self.segment_id}")
            return
        
        # Filter to only tie lines that reference this segment
        segment_ties = [
            t for t in tie_lines 
            if t.source_segment_id == self.segment_id or t.target_segment_id == self.segment_id
        ]
        
        if not segment_ties:
            logger.warning(f"No tie lines reference segment {self.segment_id}")
            return
        
        logger.debug(f"Calculating stratigraphic position for {self.segment_id} from {len(segment_ties)} tie lines")
        
        # For each tie line, calculate what the segment Y range would be
        y_top_estimates = []
        y_bottom_estimates = []
        
        for tie in segment_ties:
            # Determine if this segment is source or target
            if tie.source_segment_id == self.segment_id:
                tie_depth = tie.source_depth
            else:
                tie_depth = tie.target_depth
            
            # Calculate relative position of tie within segment
            segment_height = self.get_segment_height()
            if segment_height == 0:
                relative_pos = 0.0
            else:
                relative_pos = (tie_depth - self.depth_from_original) / segment_height
            
            # Calculate segment Y range based on this tie
            y_at_top = tie.stratigraphic_y - (relative_pos * segment_height)
            y_at_bottom = tie.stratigraphic_y + ((1.0 - relative_pos) * segment_height)
            
            y_top_estimates.append(y_at_top)
            y_bottom_estimates.append(y_at_bottom)
        
        # Average all estimates (could use weighted average by confidence in future)
        self.stratigraphic_y_top = sum(y_top_estimates) / len(y_top_estimates)
        self.stratigraphic_y_bottom = sum(y_bottom_estimates) / len(y_bottom_estimates)
        
        logger.debug(f"  Calculated Y range: [{self.stratigraphic_y_top:.2f} - {self.stratigraphic_y_bottom:.2f}]")

@dataclass
class TieLine:
    """
    Visual correlation tie line for aligning marker beds/horizons between segments.
    Multiple tie lines per segment are supported for reconstructing missing stratigraphy.
    Tie lines define stratigraphic relationships and set segment positions.
    """
    source_hole: str
    source_segment_id: str  # Which segment in source hole
    source_depth: Union[float, int]  # Depth within segment
    target_hole: str
    target_segment_id: str  # Which segment in target hole
    target_depth: Union[float, int]  # Depth within segment
    line_type: TieLineType = TieLineType.LITHOLOGY
    confidence: float = 1.0  # 0.0 to 1.0
    color: str = "#FFD700"  # Default gold color
    line_width: int = 2
    line_style: str = "solid"  # solid, dashed, dotted
    notes: str = ""
    created_by: str = ""
    created_date: str = ""
    tie_id: str = ""  # Unique identifier
    
    # Stratigraphic Y position (calculated from tie line alignment)
    # This is the "master" stratigraphic position that this tie defines
    stratigraphic_y: float = 0.0
    
    # Optional feature reference (for correlating specific features like alteration zones)
    feature_id: str = ""
    
    def __post_init__(self) -> None:
        """Post-initialization validation and logging"""
        logger.debug(f"Creating TieLine: {self.source_hole}@{self.source_depth}m -> {self.target_hole}@{self.target_depth}m")
        logger.debug(f"  Type: {self.line_type.value}, Confidence: {self.confidence:.2f}")
        logger.debug(f"  Color: {self.color}, Width: {self.line_width}, Style: {self.line_style}")
        
        # Generate tie_id if not provided
        if not self.tie_id:
            import uuid
            self.tie_id = str(uuid.uuid4())[:8]
            logger.debug(f"  Generated tie_id: {self.tie_id}")
        
        # Validate confidence range
        if not 0.0 <= self.confidence <= 1.0:
            logger.warning(f"Confidence {self.confidence} out of range, clamping to [0, 1]")
            self.confidence = max(0.0, min(1.0, self.confidence))
        
        # Set creation date if not provided
        if not self.created_date:
            self.created_date = datetime.now().isoformat()


@dataclass
class DrillholeInterval:
    """Represents a single depth interval with image and data"""
    hole_id: str
    depth_from: Union[float, int]
    depth_to: Union[float, int]
    image_path: Optional[str] = None
    image_path_wet: Optional[str] = None  # Wet image path
    image_path_dry: Optional[str] = None  # Dry image path
    csv_data: Dict[str, Any] = field(default_factory=dict)
    is_placeholder: bool = False
    has_image: bool = False
    moisture_status: str = ""  # "Wet", "Dry", or ""
    interval_id: str = ""  # Unique identifier
    
    def __post_init__(self) -> None:
        """Post-initialization validation and logging"""
        logger.debug(f"Creating DrillholeInterval: {self.hole_id} [{self.depth_from}-{self.depth_to}m]")
        logger.debug(f"  Placeholder: {self.is_placeholder}, Has Image: {self.has_image}")
        
        # Ensure depths are numeric and ordered correctly
        self.depth_from = float(self.depth_from) if self.depth_from is not None else 0.0
        self.depth_to = float(self.depth_to) if self.depth_to is not None else 0.0
        
        if self.depth_from > self.depth_to:
            logger.warning(f"Depth_from ({self.depth_from}) > depth_to ({self.depth_to}), swapping")
            self.depth_from, self.depth_to = self.depth_to, self.depth_from
        
        # Generate interval_id if not provided
        if not self.interval_id:
            self.interval_id = f"{self.hole_id}_{int(self.depth_from)}_{int(self.depth_to)}"
            logger.debug(f"  Generated interval_id: {self.interval_id}")
        
        # Check for available images
        if self.image_path_wet or self.image_path_dry or self.image_path:
            self.has_image = True
            logger.debug(f"  Images: Wet={bool(self.image_path_wet)}, Dry={bool(self.image_path_dry)}, Generic={bool(self.image_path)}")
        
        # Log CSV data summary
        if self.csv_data:
            logger.debug(f"  CSV data: {len(self.csv_data)} columns")
            if len(self.csv_data) <= 5:
                logger.debug(f"    Columns: {list(self.csv_data.keys())}")
    
    def get_best_image_path(self) -> Optional[str]:
        """Get the best available image path (prefer Dry over Wet)"""
        logger.debug(f"Getting best image for {self.hole_id}@{self.depth_to}m")
        
        if self.image_path_dry:
            self.moisture_status = "Dry"
            logger.debug(f"  Using DRY image: {self.image_path_dry}")
            return self.image_path_dry
        elif self.image_path_wet:
            self.moisture_status = "Wet"
            logger.debug(f"  Using WET image: {self.image_path_wet}")
            return self.image_path_wet
        elif self.image_path:
            logger.debug(f"  Using generic image: {self.image_path}")
            return self.image_path
        
        logger.debug(f"  No image available")
        return None
    
    def get_interval_height(self) -> float:
        """Calculate interval height in meters"""
        height = abs(float(self.depth_to) - float(self.depth_from))
        logger.debug(f"Interval {self.interval_id} height: {height}m")
        return height


@dataclass
class StretchRegion:
    """Represents a depth region with a specific stretch factor"""
    start_depth: Union[float, int]
    end_depth: Union[float, int]
    scale_factor: float = 1.0  # >1 = stretch, <1 = compress
    
    def __post_init__(self) -> None:
        """Post-initialization validation"""
        self.start_depth = float(self.start_depth)
        self.end_depth = float(self.end_depth)
        
        # Ensure start < end
        if self.start_depth > self.end_depth:
            self.start_depth, self.end_depth = self.end_depth, self.start_depth
        
        # Validate scale factor
        if self.scale_factor <= 0:
            logger.error(f"Invalid scale factor {self.scale_factor}, resetting to 1.0")
            self.scale_factor = 1.0
        elif self.scale_factor > 10:
            logger.warning(f"Large scale factor {self.scale_factor} may cause display issues")
    
    def get_region_height(self) -> float:
        """Get the original height of this region"""
        return abs(self.end_depth - self.start_depth)
    
    def get_visual_height(self) -> float:
        """Get the visual height after scaling"""
        return self.get_region_height() * self.scale_factor


@dataclass  
class DepthTransform:
    """
    Represents a depth transformation for a segment using offset + piecewise regions.
    
    This enables correction for down-dip drilling artifacts by:
    - Applying a global vertical offset to the segment
    - Stretching/compressing specific depth regions with independent scale factors
    
    Visual depth is calculated by integrating scale factors across regions:
    visual_depth(d) = offset + integral_0^d scale(u) du
    """
    segment_id: str  # Changed from hole_id - transforms are per-segment
    vertical_offset_m: float = 0.0  # Global vertical shift (positive = shift down)
    stretch_regions: List[StretchRegion] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Post-initialization validation and logging"""
        logger.debug(f"Creating DepthTransform for segment {self.segment_id}")
        logger.debug(f"  Vertical offset: {self.vertical_offset_m}m")
        logger.debug(f"  Stretch regions: {len(self.stretch_regions)}")
        
        # Normalize regions (sort by start depth)
        self._normalize_regions()
    
    def _normalize_regions(self) -> None:
        """Sort regions by start depth and remove zero-length regions"""
        # Remove zero-length regions
        self.stretch_regions = [
            r for r in self.stretch_regions 
            if abs(r.end_depth - r.start_depth) > 0.001
        ]
        
        # Sort by start depth
        self.stretch_regions.sort(key=lambda r: r.start_depth)
        
        logger.debug(f"  Normalized to {len(self.stretch_regions)} regions")
    
    def add_region(self, start_depth: float, end_depth: float, scale_factor: float = 1.0) -> None:
        """Add a stretch region"""
        region = StretchRegion(start_depth, end_depth, scale_factor)
        self.stretch_regions.append(region)
        self._normalize_regions()
        logger.debug(f"Added region [{start_depth}-{end_depth}m] with scale {scale_factor}")
    
    def apply_transform(self, depth: Union[float, int]) -> float:
        """
        Apply transformation to a depth value using piecewise linear integration.
        
        Maps original depth to visual depth by:
        1. Starting with vertical offset
        2. Integrating scale factors across depth regions
        3. Using scale=1.0 outside defined regions
        """
        depth = float(depth)
        
        # Start with the vertical offset
        visual_depth = self.vertical_offset_m
        
        # Track position as we integrate through regions
        current_pos = 0.0
        
        for region in self.stretch_regions:
            # If target depth is before this region, add the remaining distance at scale=1.0
            if depth <= region.start_depth:
                visual_depth += (depth - current_pos)
                return visual_depth
            
            # Add distance before this region at scale=1.0
            if current_pos < region.start_depth:
                visual_depth += (region.start_depth - current_pos)
                current_pos = region.start_depth
            
            # If target depth is within this region, add scaled distance
            if depth <= region.end_depth:
                visual_depth += (depth - current_pos) * region.scale_factor
                return visual_depth
            else:
                # Add the full region at its scale
                visual_depth += (region.end_depth - current_pos) * region.scale_factor
                current_pos = region.end_depth
        
        # Add any remaining distance after all regions at scale=1.0
        if depth > current_pos:
            visual_depth += (depth - current_pos)
        
        return visual_depth
    
    def get_visual_range(self, min_depth: float, max_depth: float, num_samples: int = 100) -> Tuple[float, float]:
        """Calculate the visual depth range for a given original depth range"""
        if min_depth > max_depth:
            min_depth, max_depth = max_depth, min_depth
        
        visual_depths = []
        step = (max_depth - min_depth) / max(1, num_samples - 1)
        
        for i in range(num_samples):
            d = min_depth + i * step
            visual_depths.append(self.apply_transform(d))
        
        return min(visual_depths), max(visual_depths)
    
    def calculate_from_tie_lines(
        self, 
        segment: 'DrillholeSegment',
        tie_lines: List['TieLine'],
        num_regions: int = 3
    ) -> None:
        """
        Auto-calculate stretch regions from multiple tie lines on a segment.
        
        Args:
            segment: The segment being transformed
            tie_lines: List of tie lines that constrain this segment
            num_regions: Number of stretch regions to create (default 3: top, middle, bottom)
        """
        if len(tie_lines) < 2:
            logger.warning(f"Need at least 2 tie lines to calculate stretch, got {len(tie_lines)}")
            return
        
        # Filter to tie lines that reference this segment
        segment_ties = [
            t for t in tie_lines
            if t.source_segment_id == segment.segment_id or t.target_segment_id == segment.segment_id
        ]
        
        if len(segment_ties) < 2:
            logger.warning(f"Need at least 2 tie lines for segment {segment.segment_id}")
            return
        
        # Sort tie lines by depth within segment
        segment_ties.sort(key=lambda t: 
            t.source_depth if t.source_segment_id == segment.segment_id else t.target_depth
        )
        
        logger.info(f"Calculating stretch from {len(segment_ties)} tie lines for segment {segment.segment_id}")
        
        # For each pair of adjacent tie lines, calculate required stretch
        self.stretch_regions.clear()
        
        for i in range(len(segment_ties) - 1):
            tie_top = segment_ties[i]
            tie_bottom = segment_ties[i + 1]
            
            # Get depths in this segment
            if tie_top.source_segment_id == segment.segment_id:
                depth_top = tie_top.source_depth
                strat_y_top = tie_top.stratigraphic_y
            else:
                depth_top = tie_top.target_depth
                strat_y_top = tie_top.stratigraphic_y
            
            if tie_bottom.source_segment_id == segment.segment_id:
                depth_bottom = tie_bottom.source_depth
                strat_y_bottom = tie_bottom.stratigraphic_y
            else:
                depth_bottom = tie_bottom.target_depth
                strat_y_bottom = tie_bottom.stratigraphic_y
            
            # Calculate required scale to match stratigraphic distances
            original_distance = abs(depth_bottom - depth_top)
            stratigraphic_distance = abs(strat_y_bottom - strat_y_top)
            
            if original_distance > 0:
                required_scale = stratigraphic_distance / original_distance
                logger.debug(f"  Region {i}: [{depth_top:.1f}-{depth_bottom:.1f}m] requires scale {required_scale:.3f}")
                
                self.add_region(depth_top, depth_bottom, required_scale)
        
        # Set vertical offset based on first tie line
        first_tie = segment_ties[0]
        if first_tie.source_segment_id == segment.segment_id:
            first_depth = first_tie.source_depth
        else:
            first_depth = first_tie.target_depth
        
        # Offset should position the first tie line at its stratigraphic Y
        self.vertical_offset_m = first_tie.stratigraphic_y - first_depth
        logger.debug(f"  Set vertical offset: {self.vertical_offset_m}m")


@dataclass
class CorrelationSession:
    """Stores state of a correlation session"""
    session_id: str
    hole_ids: List[str] = field(default_factory=list)
    
    # Core data structures
    segments: List[DrillholeSegment] = field(default_factory=list)
    discontinuities: List[Discontinuity] = field(default_factory=list)
    features: List[Feature] = field(default_factory=list)
    tie_lines: List[TieLine] = field(default_factory=list)
    depth_transforms: List[DepthTransform] = field(default_factory=list)
    
    # Visualization settings
    viz_columns: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    created_date: str = ""
    modified_date: str = ""
    notes: str = ""
    
    def __post_init__(self) -> None:
        """Post-initialization setup and logging"""
        logger.info(f"Creating CorrelationSession: {self.session_id}")
        logger.debug(f"  Holes: {self.hole_ids}")
        logger.debug(f"  Segments: {len(self.segments)}")
        logger.debug(f"  Discontinuities: {len(self.discontinuities)}")
        logger.debug(f"  Features: {len(self.features)}")
        logger.debug(f"  Tie lines: {len(self.tie_lines)}")
        logger.debug(f"  Transforms: {len(self.depth_transforms)}")
        
        if not self.created_date:
            self.created_date = datetime.now().isoformat()
            logger.debug(f"  Created: {self.created_date}")
        
        # Set default viz columns if not provided
        if not self.viz_columns:
            self.viz_columns = [
                {"column": "Fe_pct_BEST", "color_map": "fe_grade"},
                {"column": "SiO2_pct_BEST", "color_map": "sio2_grade"},
            ]
            logger.debug(f"  Using default viz columns: {[v['column'] for v in self.viz_columns]}")
    
    # Segment queries
    def get_segments_for_hole(self, hole_id: str) -> List[DrillholeSegment]:
        """Get all segments for a specific hole, sorted by depth"""
        segments = [seg for seg in self.segments if seg.hole_id == hole_id]
        segments.sort(key=lambda s: s.depth_from_original)
        # Update order indices
        for i, seg in enumerate(segments):
            seg.order_index = i
        return segments
    
    def get_segment_by_id(self, segment_id: str) -> Optional[DrillholeSegment]:
        """Get a segment by its ID"""
        return next((s for s in self.segments if s.segment_id == segment_id), None)
    
    def get_segments_in_stratigraphic_range(self, y_min: float, y_max: float) -> List[DrillholeSegment]:
        """Get all segments that overlap a stratigraphic Y range"""
        overlapping = []
        for seg in self.segments:
            seg_min = min(seg.stratigraphic_y_top, seg.stratigraphic_y_bottom)
            seg_max = max(seg.stratigraphic_y_top, seg.stratigraphic_y_bottom)
            
            if seg_max >= y_min and seg_min <= y_max:
                overlapping.append(seg)
        
        return overlapping
    
    def has_hole_been_segmented(self, hole_id: str) -> bool:
        """Check if a hole has been reviewed and segmented"""
        segments = self.get_segments_for_hole(hole_id)
        return len(segments) > 0
    
    # Discontinuity queries
    def get_discontinuities_for_hole(self, hole_id: str) -> List[Discontinuity]:
        """Get all discontinuities for a specific hole, sorted by depth"""
        discs = [d for d in self.discontinuities if d.hole_id == hole_id]
        discs.sort(key=lambda d: d.depth_at_boundary)
        return discs
    
    def get_discontinuity_by_id(self, discontinuity_id: str) -> Optional[Discontinuity]:
        """Get a discontinuity by its ID"""
        return next((d for d in self.discontinuities if d.discontinuity_id == discontinuity_id), None)
    
    # Feature queries
    def get_feature_by_id(self, feature_id: str) -> Optional[Feature]:
        """Get a feature by its ID"""
        return next((f for f in self.features if f.feature_id == feature_id), None)
    
    def get_features_by_category(self, category: FeatureCategory) -> List[Feature]:
        """Get all features of a specific category"""
        return [f for f in self.features if f.category == category]
    
    def get_active_features(self) -> List[Feature]:
        """Get all active features, sorted by order"""
        active = [f for f in self.features if f.is_active]
        active.sort(key=lambda f: f.order)
        return active
    
    def get_next_feature_sequence(self, category: FeatureCategory) -> int:
        """Get the next sequence number for auto-naming features in a category"""
        matching_features = [f for f in self.features if f.category == category]
        if not matching_features:
            return 1
        return max(f.sequence_number for f in matching_features) + 1
    
    def create_feature(self, category: FeatureCategory, name: str = "", **kwargs) -> Feature:
        """Create a new feature with auto-generated name if needed"""
        seq_num = self.get_next_feature_sequence(category)
        if not name:
            name = f"{category.value.title()} {seq_num}"
        
        feature = Feature(
            feature_id="",  # Will be auto-generated
            category=category,
            name=name,
            sequence_number=seq_num,
            **kwargs
        )
        self.features.append(feature)
        logger.info(f"Created feature: {feature.name}")
        return feature
    
    # Tie line queries
    def get_tie_lines_for_segment(self, segment_id: str) -> List[TieLine]:
        """Get all tie lines that reference a segment"""
        return [
            t for t in self.tie_lines 
            if t.source_segment_id == segment_id or t.target_segment_id == segment_id
        ]
    
    def get_tie_lines_for_feature(self, feature_id: str) -> List[TieLine]:
        """Get all tie lines that correlate a specific feature"""
        return [t for t in self.tie_lines if t.feature_id == feature_id]
    
    # Transform queries
    def get_transform_for_segment(self, segment_id: str) -> Optional[DepthTransform]:
        """Get the depth transform associated with a segment"""
        return next((t for t in self.depth_transforms if t.segment_id == segment_id), None)
    
    def create_or_update_transform_from_ties(self, segment_id: str) -> Optional[DepthTransform]:
        """
        Create or update a depth transform for a segment based on its tie lines.
        Automatically calculates stretch regions to align all tie lines.
        """
        segment = self.get_segment_by_id(segment_id)
        if not segment:
            logger.error(f"Segment {segment_id} not found")
            return None
        
        # Get all tie lines for this segment
        tie_lines = self.get_tie_lines_for_segment(segment_id)
        
        if len(tie_lines) < 2:
            logger.warning(f"Need at least 2 tie lines to calculate transform for {segment_id}")
            return None
        
        # Find existing transform or create new one
        transform = self.get_transform_for_segment(segment_id)
        if not transform:
            transform = DepthTransform(segment_id=segment_id)
            self.depth_transforms.append(transform)
            logger.info(f"Created new DepthTransform for segment {segment_id}")
        
        # Calculate stretch regions from tie lines
        transform.calculate_from_tie_lines(segment, tie_lines)
        segment.has_transform = True
        
        logger.info(f"Updated transform for {segment_id}: offset={transform.vertical_offset_m:.1f}m, {len(transform.stretch_regions)} regions")
        return transform
    
    # Stratigraphic position updates
    def update_all_segment_positions(self) -> None:
        """Recalculate all segment stratigraphic positions from tie lines"""
        logger.info("Updating all segment stratigraphic positions from tie lines")
        for segment in self.segments:
            segment.calculate_stratigraphic_position_from_ties(self.tie_lines)
        logger.info(f"Updated {len(self.segments)} segments")