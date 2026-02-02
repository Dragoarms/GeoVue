"""
DrillholeDataManager v2 - Refactored with indexed stores and settings dialog.

This module provides efficient data management for drillhole geological data with:
- ImageIndex: Filesystem scanning with key-based access
- RegisterStore: JSON register access for reviews, classifications, and image properties
- DataStore: Dynamic CSV loading with MultiIndex for O(1) lookups
- DataSourceSettingsDialog: UI for configuring data sources

Key Design Principles:
1. Primary key: (hole_id, depth_to) with optional moisture_status
2. Separate indexed stores - no massive merged DataFrames
3. O(1) lookups via MultiIndex or dict
4. Lazy loading where possible
5. Settings persistence via ConfigManager

Author: George Symonds
Version: 2.0.0
"""

import os
import re
import json
import logging
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================

class DataType(Enum):
    """Supported data types for columns."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    BOOLEAN = "boolean"
    DATE = "date"


class NullHandling(Enum):
    """How to handle null/missing values."""
    KEEP = "keep"           # Keep as NaN/None
    FILL_ZERO = "fill_zero"  # Replace with 0
    FILL_EMPTY = "fill_empty"  # Replace with empty string
    FILL_VALUE = "fill_value"  # Replace with custom value
    DROP = "drop"           # Drop rows with nulls


@dataclass
class ImageKey:
    """
    Unique identifier for a compartment image.
    
    The natural key is (hole_id, depth_to) with optional moisture_status
    for distinguishing wet/dry variants.
    """
    hole_id: str
    depth_to