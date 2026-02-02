"""
src/processing/DataManager/__init__.py

Drillhole data management package for GeoVue.

This package provides indexed, efficient access to:
- Compartment images (ImageIndex)
- JSON register data - reviews, classifications, image properties (RegisterStore)  
- CSV geological data (GeologicalStore)

All coordinated through a single DataCoordinator API.

Key Features:
- O(1) lookups via MultiIndex (replaces O(n) DataFrame filtering)
- Separate indexed stores (no massive merged DataFrames)
- Key-based access using ImageKey
- Lazy loading where possible

Module Structure:
    keys.py           - ImageKey dataclass, filename parsing
    schema.py         - Column and data source schema definitions
    image_index.py    - Filesystem scanning, path lookups
    geological_store.py - CSV loading with MultiIndex
    register_store.py - JSON register access with caching
    data_coordinator.py - Unified API bringing everything together
"""

# Primary imports - the main API
from processing.DataManager.keys import (
    ImageKey,
    FilenameParser,
    get_parser,
    parse_compartment_filename,
    create_key,
)

from processing.DataManager.schema import (
    DataType,
    NullHandling,
    ColumnSchema,
    DataSourceSchema,
    SchemaInferrer,
    infer_schema,
)

from processing.DataManager.image_index import (
    ImageIndex,
    ImageInfo,
)

from processing.DataManager.geological_store import (
    IndexedDataSource,
    GeologicalStore,
)

from processing.DataManager.register_store import (
    RegisterStore,
    ReviewMetadata,
    ImageProperties,
)

from processing.DataManager.data_coordinator import (
    DataCoordinator,
    CompartmentData,
)

from processing.DataManager.color_map_store import (
    ColorMapStore,
)

from processing.DataManager.rc_metrics_store import (
    RCMetricsStore,
    RCMetricsCalculator,
    MineralCodeManager,
    MineralCodeInfo,
    IntervalMetrics,
)

# Define what gets exported with "from drillhole import *"
__all__ = [
    # Keys module
    "ImageKey",
    "FilenameParser",
    "get_parser",
    "parse_compartment_filename",
    "create_key",
    
    # Schema module
    "DataType",
    "NullHandling",
    "ColumnSchema",
    "DataSourceSchema",
    "SchemaInferrer",
    "infer_schema",
    
    # Image index
    "ImageIndex",
    "ImageInfo",
    
    # Geological store
    "IndexedDataSource",
    "GeologicalStore",
    
    # Register store
    "RegisterStore",
    "ReviewMetadata",
    "ImageProperties",
    
    # Data coordinator
    "DataCoordinator",
    "CompartmentData",

    # RC Metrics store
    "RCMetricsStore",
    "RCMetricsCalculator",
    "MineralCodeManager",
    "MineralCodeInfo",
    "IntervalMetrics",
]


def create_coordinator(config_manager=None, file_manager=None) -> DataCoordinator:
    """
    Factory function to create a DataCoordinator.
    
    Convenience function for common initialization pattern.
    
    Args:
        config_manager: Optional ConfigManager for settings
        file_manager: Optional FileManager for paths
        
    Returns:
        New DataCoordinator instance
    """
    return DataCoordinator(config_manager, file_manager)
