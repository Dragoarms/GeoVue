"""
Manages color map presets and configurations for drillhole trace visualizations.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class ColorMapType(Enum):
    """Types of color maps."""
    CATEGORICAL = "categorical"
    NUMERIC = "numeric"
    GRADIENT = "gradient"


class ColorRange:
    """Represents a numeric range with associated color."""
    
    def __init__(self, min_val: float, max_val: float, color: Tuple[int, int, int], label: str = ""):
        self.min = min_val
        self.max = max_val
        self.color = color
        self.label = label or f"{min_val:.1f}-{max_val:.1f}"
        
    def contains(self, value: float) -> bool:
        """Check if value falls within this range."""
        return self.min <= value < self.max
        
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "min": self.min,
            "max": self.max,
            "color": list(self.color),
            "label": self.label
        }
        
    @classmethod
    def from_dict(cls, data: dict) -> 'ColorRange':
        """Create from dictionary."""
        return cls(
            min_val=data["min"],
            max_val=data["max"],
            color=tuple(data["color"]),
            label=data.get("label", "")
        )


class ColorMap:
    """Represents a complete color mapping configuration."""
    
    def __init__(self, name: str, map_type: ColorMapType, description: str = ""):
        self.name = name
        self.type = map_type
        self.description = description
        self.categories: Dict[str, Tuple[int, int, int]] = {}
        self.ranges: List[ColorRange] = []
        self.default_color = (200, 200, 200)  # Default gray
        self.null_color = (255, 255, 255)     # White for null values
        
    def get_color(self, value: Any) -> Tuple[int, int, int]:
        """
        Get color for a value based on map type.
        
        Args:
            value: The value to get color for
            
        Returns:
            BGR color tuple
        """
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return self.null_color
            
        if self.type == ColorMapType.CATEGORICAL:
            return self.categories.get(str(value), self.default_color)
            
        elif self.type == ColorMapType.NUMERIC:
            try:
                num_value = float(value)
                for range_obj in self.ranges:
                    if range_obj.contains(num_value):
                        return range_obj.color
                return self.default_color
            except (ValueError, TypeError):
                return self.default_color
                
        return self.default_color
        
    def add_category(self, category: str, color: Tuple[int, int, int]):
        """Add a category color mapping."""
        self.categories[category] = color
        
    def add_range(self, color_range: ColorRange):
        """Add a numeric range."""
        self.ranges.append(color_range)
        # Sort ranges by minimum value
        self.ranges.sort(key=lambda r: r.min)
        
    def get_missing_categories(self, values: List[Any]) -> List[str]:
        """Get categories that don't have assigned colors."""
        if self.type != ColorMapType.CATEGORICAL:
            return []
            
        unique_values = set(str(v) for v in values if v is not None)
        return list(unique_values - set(self.categories.keys()))
        
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        data = {
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "default_color": list(self.default_color),
            "null_color": list(self.null_color)
        }
        
        if self.type == ColorMapType.CATEGORICAL:
            data["colors"] = {k: list(v) for k, v in self.categories.items()}
        elif self.type == ColorMapType.NUMERIC:
            data["ranges"] = [r.to_dict() for r in self.ranges]
            
        return data
        
    @classmethod
    def from_dict(cls, data: dict) -> 'ColorMap':
        """Create from dictionary."""
        color_map = cls(
            name=data["name"],
            map_type=ColorMapType(data["type"]),
            description=data.get("description", "")
        )
        
        color_map.default_color = tuple(data.get("default_color", [200, 200, 200]))
        color_map.null_color = tuple(data.get("null_color", [255, 255, 255]))
        
        if color_map.type == ColorMapType.CATEGORICAL:
            for category, color in data.get("colors", {}).items():
                color_map.add_category(category, tuple(color))
                
        elif color_map.type == ColorMapType.NUMERIC:
            for range_data in data.get("ranges", []):
                color_map.add_range(ColorRange.from_dict(range_data))
                
        return color_map


class ColorMapManager:
    """Manages loading, saving, and accessing color map presets."""
    
    def __init__(self, resources_dir: Optional[str] = None):
        """
        Initialize the color map manager.
        
        Args:
            resources_dir: Path to resources directory
        """
        self.logger = logging.getLogger(__name__)
        
        # Set resources directory
        if resources_dir:
            self.resources_dir = Path(resources_dir)
        else:
            # Default to src/resources
            self.resources_dir = Path(__file__).parent.parent / "resources"
            
        self.presets_dir = self.resources_dir / "color_presets"
        self.presets_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for loaded presets
        self.presets: Dict[str, ColorMap] = {}
        
        # Load all presets
        self.load_all_presets()
        
    def load_all_presets(self):
        """Load all preset files from the presets directory."""
        if not self.presets_dir.exists():
            self.logger.warning(f"Presets directory does not exist: {self.presets_dir}")
            return
            
        for preset_file in self.presets_dir.glob("*.json"):
            try:
                with open(preset_file, 'r') as f:
                    data = json.load(f)
                    
                color_map = ColorMap.from_dict(data)
                preset_name = preset_file.stem
                self.presets[preset_name] = color_map
                
                self.logger.info(f"Loaded preset: {preset_name}")
                
            except Exception as e:
                self.logger.error(f"Error loading preset {preset_file}: {str(e)}")
                
    def get_preset(self, name: str) -> Optional[ColorMap]:
        """Get a preset by name."""
        return self.presets.get(name)
        
    def get_preset_names(self) -> List[str]:
        """Get list of available preset names."""
        return list(self.presets.keys())
        
    def save_preset(self, name: str, color_map: ColorMap) -> bool:
        """
        Save a color map as a preset.
        
        Args:
            name: Preset name (will be used as filename)
            color_map: ColorMap to save
            
        Returns:
            True if saved successfully
        """
        try:
            preset_path = self.presets_dir / f"{name}.json"
            
            with open(preset_path, 'w') as f:
                json.dump(color_map.to_dict(), f, indent=2)
                
            # Update cache
            self.presets[name] = color_map
            
            self.logger.info(f"Saved preset: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving preset {name}: {str(e)}")
            return False
            
    def create_default_presets(self):
        """Create default presets if they don't exist."""
        # Lithology preset
        if "lithology" not in self.presets:
            lithology = ColorMap("Lithology Colors", ColorMapType.CATEGORICAL, 
                               "Standard colors for geological lithologies")
            
            lithology.add_category("Sandstone", (255, 255, 0))
            lithology.add_category("Shale", (128, 128, 128))
            lithology.add_category("Limestone", (0, 191, 255))
            lithology.add_category("Granite", (255, 192, 203))
            lithology.add_category("Basalt", (64, 64, 64))
            lithology.add_category("Quartzite", (255, 228, 196))
            lithology.add_category("Dolomite", (135, 206, 235))
            lithology.add_category("Conglomerate", (139, 69, 19))
            lithology.add_category("Breccia", (165, 42, 42))
            
            self.save_preset("lithology", lithology)
            
        # Fe grade preset
        if "fe_grade" not in self.presets:
            fe_grade = ColorMap("Fe Grade", ColorMapType.NUMERIC,
                               "Iron grade color scale")
            
            fe_grade.add_range(ColorRange(0, 30, (224, 224, 224), "<30%"))
            fe_grade.add_range(ColorRange(30, 40, (48, 48, 48), "30-40%"))
            fe_grade.add_range(ColorRange(40, 45, (255, 0, 0), "40-45%"))
            fe_grade.add_range(ColorRange(45, 50, (255, 255, 0), "45-50%"))
            fe_grade.add_range(ColorRange(50, 54, (0, 255, 0), "50-54%"))
            fe_grade.add_range(ColorRange(54, 56, (0, 255, 255), "54-56%"))
            fe_grade.add_range(ColorRange(56, 58, (0, 128, 255), "56-58%"))
            fe_grade.add_range(ColorRange(58, 60, (0, 0, 255), "58-60%"))
            fe_grade.add_range(ColorRange(60, 100, (255, 0, 255), ">60%"))
            
            self.save_preset("fe_grade", fe_grade)
            
        # SiO2 grade preset
        if "sio2_grade" not in self.presets:
            sio2_grade = ColorMap("SiO2 Grade", ColorMapType.NUMERIC,
                                 "Silica grade color scale")
            
            sio2_grade.add_range(ColorRange(0, 5, (255, 255, 255), "<5%"))
            sio2_grade.add_range(ColorRange(5, 15, (0, 255, 0), "5-15%"))
            sio2_grade.add_range(ColorRange(15, 35, (255, 0, 0), "15-35%"))
            sio2_grade.add_range(ColorRange(35, 100, (0, 0, 255), ">35%"))
            
            self.save_preset("sio2_grade", sio2_grade)