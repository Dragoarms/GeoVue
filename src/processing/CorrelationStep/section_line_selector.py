"""
Section Line Geometry Class
Handles geometric calculations for drillhole selection along a section line.
"""

import math
from dataclasses import dataclass
from typing import Tuple, List, Union
import pandas as pd


@dataclass
class SectionLine:
    """Represents a geological section line for drillhole correlation.
    
    Attributes:
        start_x: X coordinate of line start (map units)
        start_y: Y coordinate of line start (map units)
        end_x: X coordinate of line end (map units)
        end_y: Y coordinate of line end (map units)
        width: Selection corridor width perpendicular to line (map units)
    """
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    width: float
    
    @property
    def azimuth(self) -> float:
        """Calculate azimuth from start to end (degrees from North, clockwise)."""
        dx = self.end_x - self.start_x
        dy = self.end_y - self.start_y
        
        # Calculate angle from horizontal
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        
        # Convert to azimuth (0° = North, clockwise)
        azimuth = 90 - angle_deg
        if azimuth < 0:
            azimuth += 360
            
        return azimuth
    
    @property
    def length(self) -> float:
        """Calculate line length in map units."""
        dx = self.end_x - self.start_x
        dy = self.end_y - self.start_y
        return math.sqrt(dx*dx + dy*dy)
    
    def distance_along_line(self, x: Union[int, float], y: Union[int, float]) -> float:
        """Calculate distance from start along line to projection of point.
        
        Args:
            x: Point X coordinate
            y: Point Y coordinate
            
        Returns:
            Distance along line (can be negative if behind start, or > length if beyond end)
        """
        if self.length == 0:
            return 0.0
            
        # Vector from start to point
        px = x - self.start_x
        py = y - self.start_y
        
        # Unit vector along line
        dx = self.end_x - self.start_x
        dy = self.end_y - self.start_y
        length = self.length
        
        # Project point onto line
        dot_product = (px * dx + py * dy) / length
        
        return dot_product
    
    def perpendicular_distance(self, x: Union[int, float], y: Union[int, float]) -> float:
        """Calculate perpendicular distance from point to line.
        
        Args:
            x: Point X coordinate
            y: Point Y coordinate
            
        Returns:
            Absolute perpendicular distance to line
        """
        if self.length == 0:
            # Line is a point, return distance to that point
            dx = x - self.start_x
            dy = y - self.start_y
            return math.sqrt(dx*dx + dy*dy)
        
        # Vector from start to point
        px = x - self.start_x
        py = y - self.start_y
        
        # Line direction vector
        dx = self.end_x - self.start_x
        dy = self.end_y - self.start_y
        
        # Calculate perpendicular distance using cross product
        cross_product = abs(px * dy - py * dx)
        distance = cross_product / self.length
        
        return distance
    
    def is_within_corridor(self, x: Union[int, float], y: Union[int, float]) -> bool:
        """Check if point is within the selection corridor.
        
        The corridor is a bounded rectangle, not an infinite strip.
        A point is within the corridor if:
        1. Its perpendicular distance to the line is <= width/2
        2. Its projection onto the line falls between start and end (0 <= distance <= length)
        
        Args:
            x: Point X coordinate
            y: Point Y coordinate
            
        Returns:
            True if point is within the rectangular corridor
        """
        # Check perpendicular distance
        perp_dist = self.perpendicular_distance(x, y)
        if perp_dist > (self.width / 2.0):
            return False
        
        # Check if projection falls within line segment bounds
        dist_along = self.distance_along_line(x, y)
        if dist_along < 0 or dist_along > self.length:
            return False
        
        return True
    
    def get_collars_in_corridor(self, collar_df: pd.DataFrame) -> pd.DataFrame:
        """Filter collar DataFrame to only those within corridor.
        
        Args:
            collar_df: DataFrame with columns 'holeid', 'x', 'y', 'z' (lowercase)
            
        Returns:
            Filtered DataFrame with collars in corridor
        """
        if collar_df.empty:
            return collar_df
        
        # Calculate which collars are within corridor
        in_corridor = collar_df.apply(
            lambda row: self.is_within_corridor(row['x'], row['y']),
            axis=1
        )
        
        return collar_df[in_corridor].copy()
    
    def sort_collars_by_distance(self, collar_df: pd.DataFrame) -> pd.DataFrame:
        """Sort collars by distance along section line.
        
        Args:
            collar_df: DataFrame with columns 'holeid', 'x', 'y', 'z' (lowercase)
            
        Returns:
            Sorted DataFrame with added 'distance_along_section' column
        """
        if collar_df.empty:
            return collar_df
        
        # Calculate distance along section for each collar
        result = collar_df.copy()
        result['distance_along_section'] = result.apply(
            lambda row: self.distance_along_line(row['x'], row['y']),
            axis=1
        )
        
        # Sort by distance
        result = result.sort_values('distance_along_section').reset_index(drop=True)
        
        return result
    
    def get_selection_box_corners(self) -> List[Tuple[float, float]]:
        """Calculate the 4 corners of the selection box for visualization.
        
        Returns:
            List of (x, y) tuples representing box corners in order:
            [start_left, end_left, end_right, start_right]
        """
        if self.length == 0:
            # Return a small square around the point
            half_width = self.width / 2.0
            return [
                (self.start_x - half_width, self.start_y - half_width),
                (self.start_x - half_width, self.start_y + half_width),
                (self.start_x + half_width, self.start_y + half_width),
                (self.start_x + half_width, self.start_y - half_width)
            ]
        
        # Calculate perpendicular unit vector
        dx = self.end_x - self.start_x
        dy = self.end_y - self.start_y
        length = self.length
        
        # Unit vector perpendicular to line (rotate 90° CCW)
        perp_x = -dy / length
        perp_y = dx / length
        
        # Offset by half width
        offset = self.width / 2.0
        
        # Calculate 4 corners
        corners = [
            (self.start_x + perp_x * offset, self.start_y + perp_y * offset),  # Start left
            (self.end_x + perp_x * offset, self.end_y + perp_y * offset),      # End left
            (self.end_x - perp_x * offset, self.end_y - perp_y * offset),      # End right
            (self.start_x - perp_x * offset, self.start_y - perp_y * offset)   # Start right
        ]
        
        return corners
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'start_x': self.start_x,
            'start_y': self.start_y,
            'end_x': self.end_x,
            'end_y': self.end_y,
            'azimuth': self.azimuth,
            'width': self.width,
            'length': self.length
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SectionLine':
        """Create SectionLine from dictionary."""
        return cls(
            start_x=data['start_x'],
            start_y=data['start_y'],
            end_x=data['end_x'],
            end_y=data['end_y'],
            width=data['width']
        )