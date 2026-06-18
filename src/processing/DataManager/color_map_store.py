"""
ColorMapStore - Centralized color map management for DataCoordinator.

This module provides a facade over ColorMapManager that:
- Provides convenient access to color map presets
- Manages column-to-colormap mappings
- Offers helper methods for getting colors in different formats (BGR, RGB, hex)
- Integrates with DataCoordinator as the single source of truth

Architecture:
    ColorMapManager (in LoggingReviewStep)
        - Core CRUD for ColorMap objects
        - Preset storage via ConfigManager
        - Import/export JSON files
        - Default presets creation
    
    ColorMapStore (this module, in DataManager)
        - Wraps ColorMapManager
        - Adds column-to-colormap mappings
        - Convenience methods (get_color_hex, get_color_rgb, etc.)
        - Integration point for DataCoordinator

Usage:
    >>> store = ColorMapStore(config_manager)
    >>> 
    >>> # Get a color map
    >>> fe_map = store.get("fe_grade")
    >>> 
    >>> # Get color for a value
    >>> bgr = store.get_color("fe_grade", 55.5)
    >>> hex_color = store.get_color_hex("fe_grade", 55.5)
    >>> 
    >>> # Map columns to color maps
    >>> store.set_column_mapping("Fe_pct_BEST", "fe_grade")
    >>> bgr = store.get_color_for_column("Fe_pct_BEST", 55.5)

Author: George Symonds
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class ColorMapStore:
    """
    Centralized store for color maps, wrapping ColorMapManager.
    
    Provides:
    - Access to preset color maps (fe_grade, sio2_grade, lithology, etc.)
    - Access to user-defined color maps
    - Convenience methods for getting colors by value
    - Column-to-color-map mappings for visualization
    
    This is the interface that DataCoordinator and UI components should use
    for all color map operations.
    """
    
    def __init__(
        self,
        config_manager=None,
        presets_dir: Optional[Path] = None,
    ):
        """
        Initialize the color map store.
        
        Args:
            config_manager: ConfigManager instance for user settings
            presets_dir: Path to color preset JSON files (optional)
        """
        self._config_manager = config_manager
        self._presets_dir = presets_dir
        self._manager = None  # Lazy-loaded ColorMapManager
        self._column_mappings: Dict[str, str] = {}  # column_name -> color_map_name
        
        # Try to import and initialize ColorMapManager
        self._init_manager()
        
        # Load column mappings from config
        self._load_column_mappings()
    
    def _init_manager(self):
        """Initialize the underlying ColorMapManager."""
        try:
            # Try importing from the standard location
            try:
                from processing.LoggingReviewStep.color_map_manager import ColorMapManager
            except ImportError:
                # Try alternative locations
                try:
                    from color_map_manager import ColorMapManager
                except ImportError:
                    logger.error("Could not import ColorMapManager from any location")
                    self._manager = None
                    return
            
            self._manager = ColorMapManager(self._config_manager)
            
            # Log what was loaded
            preset_count = len(self.list_all()) if self._manager else 0
            logger.info(f"ColorMapStore initialized with {preset_count} color maps")
            
        except ImportError as e:
            logger.warning(f"Could not import ColorMapManager: {e}")
            self._manager = None
        except Exception as e:
            logger.error(f"Error initializing ColorMapManager: {e}")
            self._manager = None
    
    def _load_column_mappings(self):
        """Load column-to-colormap mappings from config."""
        if not self._config_manager:
            return
        
        try:
            # Load from viz_columns config (drill trace visualization)
            viz_columns = self._config_manager.get("viz_columns", [])
            for vc in viz_columns:
                if isinstance(vc, dict) and "column" in vc and "color_map" in vc:
                    self._column_mappings[vc["column"]] = vc["color_map"]
            
            # Also load correlation viz columns
            corr_columns = self._config_manager.get("correlation_viz_columns", [])
            for vc in corr_columns:
                if isinstance(vc, dict) and "column" in vc and "color_map" in vc:
                    self._column_mappings[vc["column"]] = vc["color_map"]
            
            # Load explicit column mappings if saved
            explicit_mappings = self._config_manager.get("column_color_mappings", {})
            if isinstance(explicit_mappings, dict):
                self._column_mappings.update(explicit_mappings)
            
            if self._column_mappings:
                logger.debug(f"Loaded {len(self._column_mappings)} column-to-colormap mappings")
                
        except Exception as e:
            logger.warning(f"Error loading column mappings: {e}")
    
    def _save_column_mappings(self):
        """Save column mappings to config."""
        if self._config_manager:
            self._config_manager.set("column_color_mappings", self._column_mappings)
    
    # =========================================================================
    # Property Access
    # =========================================================================
    
    @property
    def manager(self):
        """
        Get the underlying ColorMapManager (for advanced operations).
        
        Use this when you need direct access to ColorMapManager methods
        not exposed by ColorMapStore.
        """
        return self._manager
    
    @property
    def is_available(self) -> bool:
        """Check if the color map manager is available."""
        return self._manager is not None
    
    # =========================================================================
    # Core Access Methods
    # =========================================================================
    
    def get(self, name: str):
        """
        Get a ColorMap by name.
        
        Args:
            name: Color map name (e.g., 'fe_grade', 'lithology')
            
        Returns:
            ColorMap object or None if not found
        """
        if self._manager is None:
            return None
        
        # Try both get_preset and get methods (for compatibility)
        if hasattr(self._manager, 'get_preset'):
            return self._manager.get_preset(name)
        elif hasattr(self._manager, 'get'):
            return self._manager.get(name)
        return None
    
    def get_for_column(self, column_name: str):
        """
        Get the ColorMap configured for a specific data column.
        
        Args:
            column_name: DataFrame column name (e.g., 'Fe_pct_BEST')
            
        Returns:
            ColorMap object or None if no mapping configured
        """
        color_map_name = self._column_mappings.get(column_name)
        if color_map_name:
            return self.get(color_map_name)
        return None
    
    # =========================================================================
    # Color Retrieval Methods
    # =========================================================================
    
    def get_color(self, color_map_name: str, value: Any) -> Tuple[int, int, int]:
        """
        Get BGR color for a value using a named color map.
        
        Args:
            color_map_name: Name of the color map
            value: Value to get color for (numeric or categorical)
            
        Returns:
            BGR tuple (for OpenCV compatibility)
        """
        color_map = self.get(color_map_name)
        if color_map:
            return color_map.get_color(value)
        return (200, 200, 200)  # Default gray
    
    def get_color_for_column(self, column_name: str, value: Any) -> Tuple[int, int, int]:
        """
        Get BGR color for a value based on column's configured color map.
        
        Args:
            column_name: DataFrame column name
            value: Value to get color for
            
        Returns:
            BGR tuple
        """
        color_map = self.get_for_column(column_name)
        if color_map:
            return color_map.get_color(value)
        return (200, 200, 200)  # Default gray
    
    def get_color_rgb(self, color_map_name: str, value: Any) -> Tuple[float, float, float]:
        """
        Get RGB color (0-1 range) for matplotlib/tkinter use.
        
        Args:
            color_map_name: Name of the color map
            value: Value to get color for
            
        Returns:
            RGB tuple with values 0.0-1.0
        """
        b, g, r = self.get_color(color_map_name, value)
        return (r / 255.0, g / 255.0, b / 255.0)
    
    def get_color_hex(self, color_map_name: str, value: Any) -> str:
        """
        Get hex color string for a value.
        
        Args:
            color_map_name: Name of the color map
            value: Value to get color for
            
        Returns:
            Hex color string (e.g., '#FF0000')
        """
        b, g, r = self.get_color(color_map_name, value)
        return f"#{r:02X}{g:02X}{b:02X}"
    
    # =========================================================================
    # Listing Methods
    # =========================================================================
    
    def list_all(self) -> List[str]:
        """List all available color map names."""
        if self._manager is None:
            return []
        
        if hasattr(self._manager, 'list_color_maps'):
            return self._manager.list_color_maps()
        elif hasattr(self._manager, 'get_preset_names'):
            return self._manager.get_preset_names()
        return []
    
    def list_numeric(self) -> List[str]:
        """List color maps suitable for numeric data."""
        if self._manager is None:
            return []
        
        if hasattr(self._manager, 'list_numeric_color_maps'):
            return self._manager.list_numeric_color_maps()
        
        # Fallback: filter by type
        try:
            from processing.LoggingReviewStep.color_map_manager import ColorMapType
            return sorted([
                name for name in self.list_all()
                if self.get(name) and self.get(name).type in (ColorMapType.NUMERIC, ColorMapType.GRADIENT)
            ])
        except ImportError:
            return self.list_all()
    
    def list_categorical(self) -> List[str]:
        """List color maps suitable for categorical data."""
        if self._manager is None:
            return []
        
        if hasattr(self._manager, 'list_categorical_color_maps'):
            return self._manager.list_categorical_color_maps()
        
        # Fallback: filter by type
        try:
            from processing.LoggingReviewStep.color_map_manager import ColorMapType
            return sorted([
                name for name in self.list_all()
                if self.get(name) and self.get(name).type == ColorMapType.CATEGORICAL
            ])
        except ImportError:
            return self.list_all()
    
    # =========================================================================
    # Column Mapping Methods
    # =========================================================================
    
    def set_column_mapping(self, column_name: str, color_map_name: str):
        """
        Set the color map to use for a specific column.
        
        Args:
            column_name: DataFrame column name
            color_map_name: Name of the color map to use
        """
        self._column_mappings[column_name] = color_map_name
        self._save_column_mappings()
        logger.debug(f"Set column mapping: {column_name} -> {color_map_name}")
    
    def remove_column_mapping(self, column_name: str):
        """Remove a column mapping."""
        if column_name in self._column_mappings:
            del self._column_mappings[column_name]
            self._save_column_mappings()
    
    def get_column_mappings(self) -> Dict[str, str]:
        """Get all column-to-colormap mappings."""
        return self._column_mappings.copy()
    
    def get_column_mapping(self, column_name: str) -> Optional[str]:
        """Get the color map name for a column, or None."""
        return self._column_mappings.get(column_name)
    
    # =========================================================================
    # Conversion Methods
    # =========================================================================
    
    def create_dict_color_map(self, color_map_name: str) -> Dict[str, str]:
        """
        Convert a categorical ColorMap to a simple dict for widgets.
        
        Useful for widgets that expect {category: hex_color} format.
        
        Args:
            color_map_name: Name of the color map
            
        Returns:
            Dict mapping category names to hex colors
        """
        color_map = self.get(color_map_name)
        if color_map is None:
            return {}
        
        try:
            from processing.LoggingReviewStep.color_map_manager import ColorMapType
            
            if color_map.type == ColorMapType.CATEGORICAL:
                result = {}
                for category, bgr in color_map.categories.items():
                    b, g, r = bgr
                    result[category] = f"#{r:02X}{g:02X}{b:02X}"
                return result
        except (ImportError, AttributeError):
            pass
        
        return {}
    
    def create_numeric_dict(self, color_map_name: str) -> List[Dict[str, Any]]:
        """
        Convert a numeric ColorMap to a list of range dicts.
        
        Useful for widgets that need range information.
        
        Args:
            color_map_name: Name of the color map
            
        Returns:
            List of dicts with 'min', 'max', 'color_hex', 'label' keys
        """
        color_map = self.get(color_map_name)
        if color_map is None:
            return []
        
        try:
            from processing.LoggingReviewStep.color_map_manager import ColorMapType
            
            if color_map.type in (ColorMapType.NUMERIC, ColorMapType.GRADIENT):
                result = []
                for range_obj in color_map.ranges:
                    b, g, r = range_obj.color
                    result.append({
                        'min': range_obj.min,
                        'max': range_obj.max,
                        'color_hex': f"#{r:02X}{g:02X}{b:02X}",
                        'label': range_obj.label,
                    })
                return result
        except (ImportError, AttributeError):
            pass
        
        return []
    
    # =========================================================================
    # CRUD Operations (delegated to manager)
    # =========================================================================
    
    def save(self, name: str, color_map) -> bool:
        """
        Save a color map preset.
        
        Args:
            name: Name for the color map
            color_map: ColorMap object to save
            
        Returns:
            True if saved successfully
        """
        if self._manager is None:
            logger.warning("Cannot save color map: manager not initialized")
            return False
        
        if hasattr(self._manager, 'save_preset'):
            return self._manager.save_preset(name, color_map)
        return False
    
    def delete(self, name: str) -> bool:
        """
        Delete a color map preset.
        
        Args:
            name: Name of the color map to delete
            
        Returns:
            True if deleted successfully
        """
        if self._manager is None:
            return False
        
        if hasattr(self._manager, 'delete_preset'):
            return self._manager.delete_preset(name)
        return False
    
    def export(self, name: str, file_path: str) -> bool:
        """
        Export a single color map to a JSON file.
        
        Args:
            name: Name of the color map to export
            file_path: Path to save the JSON file
            
        Returns:
            True if exported successfully
        """
        if self._manager is None:
            return False
        
        if hasattr(self._manager, 'export_preset'):
            return self._manager.export_preset(name, file_path)
        return False
    
    def import_from_file(self, file_path: str, overwrite: bool = False) -> Optional[str]:
        """
        Import a color map from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            overwrite: If True, overwrite existing preset with same name
            
        Returns:
            Name of imported preset, or None if failed
        """
        if self._manager is None:
            return None
        
        if hasattr(self._manager, 'import_preset'):
            return self._manager.import_preset(file_path, overwrite)
        return None
    
    # =========================================================================
    # Lifecycle Methods
    # =========================================================================
    
    def reload(self):
        """Reload color maps from presets and config."""
        self._init_manager()
        self._load_column_mappings()
        logger.info("ColorMapStore reloaded")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the color map store."""
        return {
            "is_available": self.is_available,
            "total_color_maps": len(self.list_all()),
            "numeric_color_maps": len(self.list_numeric()),
            "categorical_color_maps": len(self.list_categorical()),
            "column_mappings": len(self._column_mappings),
        }


# =============================================================================
# Module-level convenience functions
# =============================================================================

def bgr_to_hex(bgr: Tuple[int, int, int]) -> str:
    """Convert BGR tuple to hex color string."""
    b, g, r = bgr
    return f"#{r:02X}{g:02X}{b:02X}"


def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color string to BGR tuple."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)


def bgr_to_rgb_float(bgr: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Convert BGR tuple to RGB float tuple (0-1 range)."""
    b, g, r = bgr
    return (r / 255.0, g / 255.0, b / 255.0)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("ColorMapStore module loaded successfully")
    
    # Basic test without config manager
    store = ColorMapStore(config_manager=None)
    
    print(f"Available: {store.is_available}")
    print(f"Color maps: {store.list_all()}")
    print(f"Stats: {store.get_stats()}")