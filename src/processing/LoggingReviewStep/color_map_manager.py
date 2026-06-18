"""
Manages color map presets and configurations for drillhole trace visualizations.
Version 2.0 - Uses ConfigManager for user-specific storage with export/import support.
"""

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
        self.null_color = (240, 240, 240)     # Light gray for null values (matches grid background)
        
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
    """
    Manages loading, saving, and accessing color map presets.
    Version 2.0 - Uses ConfigManager for user-specific storage.
    """
    
    def __init__(self, config_manager):
        """
        Initialize the color map manager.
        
        Args:
            config_manager: ConfigManager instance for persistent storage
        """
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager
        
        # Cache for loaded presets
        self.presets: Dict[str, ColorMap] = {}
        
        # Load all presets from config
        self.load_all_presets()
        
        # Create defaults if none exist
        if not self.presets:
            self.logger.info("No color maps found, creating defaults")
            self.create_default_presets()
        
    def load_all_presets(self):
        """Load all presets from ConfigManager."""
        try:
            color_maps_data = self.config_manager.get("color_maps")
            
            if color_maps_data and isinstance(color_maps_data, dict):
                self.presets.clear()
                
                for name, data in color_maps_data.items():
                    try:
                        color_map = ColorMap.from_dict(data)
                        self.presets[name] = color_map
                        self.logger.debug(f"Loaded preset: {name}")
                    except Exception as e:
                        self.logger.error(f"Error loading preset {name}: {e}")
                
                self.logger.info(f"Loaded {len(self.presets)} color map presets from config")
            else:
                self.logger.info("No color maps found in config")
                
        except Exception as e:
            self.logger.error(f"Error loading color map presets: {e}")
    
    def save_all_presets(self):
        """Save all presets to ConfigManager."""
        try:
            color_maps_data = {name: cm.to_dict() for name, cm in self.presets.items()}
            self.config_manager.set("color_maps", color_maps_data)
            self.logger.info(f"Saved {len(self.presets)} color map presets to config")
            return True
        except Exception as e:
            self.logger.error(f"Error saving color map presets: {e}")
            return False
    
    def get_preset(self, name: str) -> Optional[ColorMap]:
        """Get a preset by name."""
        return self.presets.get(name)
        
    def save_preset(self, name: str, color_map: ColorMap) -> bool:
        """
        Save a color map as a preset.
        
        Args:
            name: Preset name
            color_map: ColorMap to save
            
        Returns:
            True if saved successfully
        """
        try:
            # Update name to match key
            color_map.name = name
            
            # Update cache
            self.presets[name] = color_map
            
            # Save all presets to config
            return self.save_all_presets()
            
        except Exception as e:
            self.logger.error(f"Error saving preset {name}: {e}")
            return False
    
    def delete_preset(self, name: str) -> bool:
        """
        Delete a preset.
        
        Args:
            name: Preset name to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            if name in self.presets:
                del self.presets[name]
                return self.save_all_presets()
            return False
        except Exception as e:
            self.logger.error(f"Error deleting preset {name}: {e}")
            return False
    
    def export_preset(self, name: str, file_path: str) -> bool:
        """
        Export a single preset to a JSON file.
        
        Args:
            name: Preset name to export
            file_path: Path to save the JSON file
            
        Returns:
            True if exported successfully
        """
        try:
            if name not in self.presets:
                self.logger.error(f"Preset '{name}' not found")
                return False
            
            color_map = self.presets[name]
            data = color_map.to_dict()
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Exported preset '{name}' to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting preset {name}: {e}")
            return False
    
    def export_all_presets(self, file_path: str) -> bool:
        """
        Export all presets to a single JSON file.
        
        Args:
            file_path: Path to save the JSON file
            
        Returns:
            True if exported successfully
        """
        try:
            data = {name: cm.to_dict() for name, cm in self.presets.items()}
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Exported {len(self.presets)} presets to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting all presets: {e}")
            return False
    
    def import_preset(self, file_path: str, overwrite: bool = False) -> Optional[str]:
        """
        Import a preset from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            overwrite: If True, overwrite existing preset with same name
            
        Returns:
            Name of imported preset, or None if failed
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check if it's a single preset or multiple
            if "type" in data:  # Single preset
                color_map = ColorMap.from_dict(data)
                name = color_map.name
                
                if name in self.presets and not overwrite:
                    # Generate unique name
                    base_name = name
                    counter = 1
                    while name in self.presets:
                        name = f"{base_name}_{counter}"
                        counter += 1
                    color_map.name = name
                
                self.presets[name] = color_map
                self.save_all_presets()
                
                self.logger.info(f"Imported preset: {name}")
                return name
            else:
                self.logger.error("Invalid color map file format")
                return None
                
        except Exception as e:
            self.logger.error(f"Error importing preset from {file_path}: {e}")
            return None
    
    def import_all_presets(self, file_path: str, overwrite: bool = False) -> int:
        """
        Import multiple presets from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            overwrite: If True, overwrite existing presets with same names
            
        Returns:
            Number of presets imported
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if not isinstance(data, dict):
                self.logger.error("Invalid format: expected dict of color maps")
                return 0
            
            count = 0
            for name, cm_data in data.items():
                try:
                    color_map = ColorMap.from_dict(cm_data)
                    import_name = name
                    
                    if name in self.presets and not overwrite:
                        # Generate unique name
                        base_name = name
                        counter = 1
                        while import_name in self.presets:
                            import_name = f"{base_name}_{counter}"
                            counter += 1
                        color_map.name = import_name
                    
                    self.presets[import_name] = color_map
                    count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error importing preset '{name}': {e}")
            
            if count > 0:
                self.save_all_presets()
                self.logger.info(f"Imported {count} presets")
            
            return count
            
        except Exception as e:
            self.logger.error(f"Error importing presets from {file_path}: {e}")
            return 0
            
    def create_default_presets(self):
        """Create default presets if they don't exist."""
        # Lithology preset (project-specific categories)
        if "lithology" not in self.presets:
            lithology = ColorMap("lithology", ColorMapType.CATEGORICAL, 
                               "Project-specific lithology classifications")
            
            # Amphibolite
            lithology.add_category("AMf", (0, 128, 0))
            lithology.add_category("AMp", (0, 128, 0))
            
            # Banded Iron Deposit
            lithology.add_category("BID", (0, 0, 255))
            lithology.add_category("BIDf", (0, 0, 255))
            lithology.add_category("BIDh", (0, 0, 255))
            lithology.add_category("BIDm", (0, 0, 255))
            lithology.add_category("BIDs", (0, 0, 255))
            
            # Banded Iron Formation
            lithology.add_category("BIF", (0, 128, 255))
            lithology.add_category("BIFf", (128, 255, 255))
            lithology.add_category("BIFh", (0, 128, 255))
            lithology.add_category("BIFm", (0, 128, 255))
            lithology.add_category("BIFs", (128, 0, 255))
            
            # Cavity Fill / Canga
            lithology.add_category("CF", (255, 128, 64))
            lithology.add_category("CG", (255, 128, 64))
            lithology.add_category("CGi", (255, 128, 64))
            lithology.add_category("CGs", (255, 128, 64))
            
            # Chlorite Schist
            lithology.add_category("CLQSC", (128, 255, 0))
            lithology.add_category("CLSC", (128, 255, 0))
            
            # Detritals
            lithology.add_category("DE", (255, 128, 128))
            lithology.add_category("DEc", (255, 128, 128))
            lithology.add_category("DEi", (255, 128, 128))
            lithology.add_category("DEm", (255, 128, 128))
            lithology.add_category("DEs", (255, 128, 128))
            
            # Laterite
            lithology.add_category("LAc", (255, 182, 147))
            lithology.add_category("LAt", (255, 182, 147))
            
            # Mafic
            lithology.add_category("Mafic", (0, 128, 0))
            
            # Pebble Conglomerate / Pegmatite / Phyllite
            lithology.add_category("PCS", (255, 255, 0))
            lithology.add_category("PEG", (0, 255, 0))
            lithology.add_category("PHY", (201, 130, 22))
            
            # Quartz
            lithology.add_category("QT", (192, 192, 192))
            lithology.add_category("QV", (192, 192, 192))
            
            # Schist
            lithology.add_category("SCH", (0, 128, 64))
            
            # Void
            lithology.add_category("VO", (0, 0, 0))
            
            # Fault
            lithology.add_category("ZF", (255, 0, 0))
            
            self.save_preset("lithology", lithology)
            
        # Fe grade preset
        if "fe_grade" not in self.presets:
            fe_grade = ColorMap("fe_grade", ColorMapType.NUMERIC,
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
            sio2_grade = ColorMap("sio2_grade", ColorMapType.NUMERIC,
                                 "Silica grade color scale")
            
            sio2_grade.add_range(ColorRange(0, 5, (255, 255, 255), "<5%"))
            sio2_grade.add_range(ColorRange(5, 15, (0, 255, 0), "5-15%"))
            sio2_grade.add_range(ColorRange(15, 35, (255, 0, 0), "15-35%"))
            sio2_grade.add_range(ColorRange(35, 100, (0, 0, 255), ">35%"))
            
            self.save_preset("sio2_grade", sio2_grade)
        
        # Al2O3 grade preset
        if "al2o3_grade" not in self.presets:
            al2o3_grade = ColorMap("al2o3_grade", ColorMapType.NUMERIC,
                                  "Alumina grade color scale")
            
            al2o3_grade.add_range(ColorRange(0, 2, (255, 255, 255), "<2%"))
            al2o3_grade.add_range(ColorRange(2, 4, (0, 255, 0), "2-4%"))
            al2o3_grade.add_range(ColorRange(4, 6, (255, 255, 0), "4-6%"))
            al2o3_grade.add_range(ColorRange(6, 8, (255, 128, 0), "6-8%"))
            al2o3_grade.add_range(ColorRange(8, 100, (255, 0, 0), ">8%"))
            
            self.save_preset("al2o3_grade", al2o3_grade)

    def get_preset_names(self) -> List[str]:
        """Get list of all preset names."""
        return sorted(list(self.presets.keys()))
    
    def list_color_maps(self) -> List[str]:
        """Get list of all available color map names (alias for get_preset_names)."""
        return self.get_preset_names()
    
    def list_numeric_color_maps(self) -> List[str]:
        """Get list of color maps suitable for numeric data."""
        return sorted([
            name for name, cm in self.presets.items()
            if cm.type in (ColorMapType.NUMERIC, ColorMapType.GRADIENT)
        ])
    
    def list_categorical_color_maps(self) -> List[str]:
        """Get list of color maps suitable for categorical data."""
        return sorted([
            name for name, cm in self.presets.items()
            if cm.type == ColorMapType.CATEGORICAL
        ])
    
    def get(self, name: str) -> Optional[ColorMap]:
        """Get a color map by name (alias for get_preset)."""
        return self.get_preset(name)
        