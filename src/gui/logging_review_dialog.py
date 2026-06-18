# src\gui\logging_review_dialog.py
"""
Advanced lithology classification dialog with CSV integration and full feature set.
Implements grid-based review with undo/redo, scatter plot filtering, and automatic saving.
"""

import os
import re
import logging
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import Counter, deque, defaultdict
import json
import csv
from datetime import datetime
import threading
import pandas as pd
import cv2
from collections import Counter
# Image processing
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont

# Custom imports
from gui.progress_dialog import ProgressDialog
from gui.dialog_helper import DialogHelper
from gui.widgets.modern_button import ModernButton
from gui.widgets.dynamic_filter_row import DynamicFilterRow
from gui.widgets.collapsible_frame import CollapsibleFrame
from gui.widgets.select_on_scatterplot_widget import ScatterSelectionWidget
from gui.widgets.advanced_filter_window import AdvancedFilterWindow, FilterWindowConfig
from gui.pdf_export_dialog import PDFExportDialog
from processing.LoggingReviewStep.drillhole_data_manager import (
    DrillholeDataManager,
    DataType,
)
# New DataCoordinator imports
from processing.DataManager.keys import ImageKey
from processing.DataManager.schema import DataType as NewDataType
from processing.LoggingReviewStep.drillhole_data_visualizer import (
    DrillholeDataVisualizer,
    PlotType,
    PlotConfig,
)
from processing.LoggingReviewStep.color_map_manager import ColorMapManager

# Initialize classification manager
from gui.ReviewDialog.image_classification_and_tag_manager import (
    ImageClassificationAndTagManager,
)

from gui.ReviewDialog.filter_pipeline import (
    FilterPipeline,
    ImageDataPopulator,
    FilterCriteria,
    FilterResult,
    DisplayMode,
    ClassificationVisibility,
    DebugCheckpoint,
    create_filter_criteria_from_dialog,
)

# ============================================================================
# DATA MODELS
# ============================================================================


class ClassificationCategory(Enum):
    """Available lithology classification categories"""

    BIFF = "BIFf"
    BIFF_S = "BIFf-s"
    COMPACT = "Compact"
    BIFHM = "BIFhm"
    OTHER = "Other"
    NOT_CONFIDENT = "Not Confident"
    UNASSIGNED = ""


# FilterCriteria dataclass removed - using FilterRow widget approach instead


@dataclass
class CompartmentImage:
    """Represents a compartment image with full metadata"""

    filename: str
    hole_id: str
    depth_from: float
    depth_to: float
    image_path: str
    classification: ClassificationCategory = ClassificationCategory.UNASSIGNED
    moisture_status: Optional[str] = None  # "Wet" or "Dry"
    comments: str = ""
    classified_by: str = ""
    classified_date: str = ""
    tags: Set[str] = field(default_factory=set)  # Multiple tags (additive)
    active_filters: str = ""
    csv_data: Dict[str, Any] = field(default_factory=dict)
    in_csv: bool = False
    compartment_uid: Optional[str] = None  # Add UID field True
    _image: Optional[np.ndarray] = None
    _thumbnail: Optional[ImageTk.PhotoImage] = None

    @property
    def image(self) -> Optional[np.ndarray]:
        """Lazy load image only when accessed"""
        if self._image is None and self.image_path and os.path.exists(self.image_path):
            try:
                self._image = cv2.imread(self.image_path)
            except Exception as e:
                logging.error(f"Failed to load image {self.image_path}: {e}")
        return self._image

    def unload_image(self):
        """Free memory by unloading the image"""
        self._image = None
        self._thumbnail = None

    def get_display_label(self) -> str:
        """Get display label for the image"""
        label = f"{self.hole_id}\n{int(self.depth_to)}m"
        if self.moisture_status:
            label += f"\n{self.moisture_status}"
        return label


@dataclass
class UndoAction:
    """Represents an undoable action"""

    action_type: str  # "classify", "comment", etc.
    affected_images: List[Tuple[int, str]]  # List of (index, filename) tuples
    old_states: List[Dict[str, Any]]  # Previous state for each image
    new_states: List[Dict[str, Any]]  # New state for each image
    timestamp: datetime = field(default_factory=datetime.now)

    def get_description(self) -> str:
        """Get human-readable description of the action"""
        count = len(self.affected_images)
        if self.action_type == "classify":
            # Get the new classification from first image
            if self.new_states and "classification" in self.new_states[0]:
                classification = self.new_states[0]["classification"]
                return f"Classified {count} image(s) as {classification}"
            return f"Changed classification of {count} image(s)"
        elif self.action_type == "comment":
            if self.new_states and "comments" in self.new_states[0]:
                comment = self.new_states[0]["comments"]
                return f"Added comment to {count} image(s): {comment[:30]}..."
            return f"Changed comments on {count} image(s)"
        return f"{self.action_type} on {count} image(s)"


# ============================================================================
# CSV HANDLER - LEGACY EXPORT ONLY
# ============================================================================
# NOTE: This class is ONLY used for exporting classifications to CSV.
# Data loading is now handled by DrillholeDataManager.
# TODO: Refactor export functionality to separate module


class CSVHandler:
    """DEPRECATED: Only used for CSV export operations. Use DrillholeDataManager for data loading."""

    def __init__(self, logger):
        self.logger = logger
        self.datasets = (
            {}
        )  # {filename: {'df': DataFrame, 'mapping': {...}, 'index': dict}}
        self._lookup_cache = {}  # {(hole_id, depth): row_dict} - in-memory cache

    def load_dataset(self, file_path: str, hole_col: str, to_col: str) -> bool:
        """Load a single dataset with explicit column mapping"""
        try:
            # Determine file type and load
            file_ext = os.path.splitext(file_path)[1].lower()

            if file_ext == ".csv":
                df = pd.read_csv(file_path, low_memory=False)
            elif file_ext in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path)
            elif file_ext == ".txt":
                df = pd.read_csv(file_path, sep="\t")
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")

            # Validate columns exist
            if hole_col not in df.columns:
                raise ValueError(f"Column '{hole_col}' not found in file")
            if to_col not in df.columns:
                raise ValueError(f"Column '{to_col}' not found in file")

            # Normalize hole IDs for fast lookup
            df['_hole_normalized'] = df[hole_col].astype(str).str.strip().str.upper()
            df['_depth_int'] = pd.to_numeric(df[to_col], errors='coerce').fillna(0).astype(int)
            
            # Create multi-index for O(1) lookups
            df_indexed = df.set_index(['_hole_normalized', '_depth_int'], drop=False)
            
            # Store with mapping
            filename = os.path.basename(file_path)
            self.datasets[filename] = {
                "df": df,
                "df_indexed": df_indexed,  # Indexed version for fast lookups
                "mapping": {"hole_id": hole_col, "depth_to": to_col},
                "path": file_path,
            }

            self.logger.info(f"Loaded dataset '{filename}' with {len(df)} rows")
            self.logger.info(f"  Mapped: HoleID='{hole_col}', To='{to_col}'")
            self.logger.info(f"  Created indexed lookup for fast queries")

            return True

        except Exception as e:
            self.logger.error(f"Error loading dataset {file_path}: {e}")
            return False

    def get_data_for_image(self, hole_id: str, depth_to: float) -> Dict[str, Any]:
        """Get data from all datasets for a specific image using indexed lookup"""
        # Normalize lookup key
        hole_normalized = str(hole_id).strip().upper()

        # Check in-memory cache first
        cache_key = (hole_normalized, depth_to)
        if cache_key in self._lookup_cache:
            return self._lookup_cache[cache_key].copy()

        combined_data = {}

        for filename, dataset_info in self.datasets.items():
            lookup_index = dataset_info.get("index")
            if lookup_index is None:
                df = dataset_info["df"]
                mapping = dataset_info["mapping"]
                hole_col = mapping["hole_id"]
                to_col = mapping["depth_to"]
                index = {}
                for _, row in df.iterrows():
                    try:
                        hole_val = str(row[hole_col]).strip().upper()
                        to_val = row[to_col]
                        if pd.isna(to_val):
                            continue
                        depth_f = float(to_val)
                    except Exception:
                        continue
                    # int key
                    index[(hole_val, int(depth_f))] = row.to_dict()
                    # float key if not integer
                    if depth_f != int(depth_f):
                        index[(hole_val, depth_f)] = row.to_dict()
                dataset_info["index"] = index
                lookup_index = index

            # Try integer depth first (most common)
            try:
                depth_int = int(float(depth_to))
                lookup_key = (hole_normalized, depth_int)
                if lookup_key in lookup_index:
                    combined_data.update(lookup_index[lookup_key])
                    continue
            except (ValueError, TypeError):
                pass

            # Try float depth if different from int
            try:
                depth_float = float(depth_to)
                if depth_float != int(depth_float):
                    lookup_key = (hole_normalized, depth_float)
                    if lookup_key in lookup_index:
                        combined_data.update(lookup_index[lookup_key])
            except (ValueError, TypeError):
                pass

        # Cache the result (even if empty)
        self._lookup_cache[cache_key] = combined_data

        return combined_data.copy()

    def get_available_columns(self) -> List[str]:
        """Get all available columns across datasets"""
        columns = []

        for filename, dataset_info in self.datasets.items():
            df = dataset_info["df"]
            for col in df.columns:
                if len(self.datasets) > 1:
                    columns.append(f"{filename}:{col}")
                else:
                    columns.append(col)

        return columns

    def clear_datasets(self):
        """Clear all loaded datasets"""
        self.datasets.clear()
        self.logger.info("Cleared all datasets")

    @property
    def combined_df(self):
        """For backward compatibility - returns None since we don't combine"""
        return pd.DataFrame() if self.datasets else None

    def export_classifications(
        self,
        images: List[CompartmentImage],
        filepath: str,
        classified_only: bool = True,
    ):
        """Export classifications to CSV with one column per reviewer"""
        try:
            # First pass: collect all unique reviewers across all images
            all_reviewers = set()
            for img in images:
                if img.classified_by:
                    all_reviewers.add(img.classified_by)

                if hasattr(img, "other_reviews") and img.other_reviews:
                    for review in img.other_reviews:
                        reviewer = review.get("Reviewed_By")
                        if reviewer:
                            all_reviewers.add(reviewer)

            # Sort reviewers for consistent column order
            reviewer_list = sorted(all_reviewers)

            with open(filepath, "w", newline="", encoding="utf-8") as f:
                # Build column list
                columns = [
                    "hole_id",
                    "depth_from",
                    "depth_to",
                    "moisture_status",
                ]

                # Add classification columns for each reviewer
                for reviewer in reviewer_list:
                    columns.append(f"classification_{reviewer}")
                    columns.append(f"review_date_{reviewer}")
                    columns.append(f"comments_{reviewer}")
                    columns.append(f"tags_{reviewer}")

                columns.extend(
                    [
                        "review_count",
                        "consensus_classification",
                        "agreement",
                    ]
                )

                # Add CSV data columns if available
                if images and images[0].csv_data:
                    columns.extend(images[0].csv_data.keys())

                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()

                for img in images:
                    if classified_only and img.classification in [
                        "Unassigned",
                        ClassificationCategory.UNASSIGNED,
                    ]:
                        continue

                    # Build row with basic info
                    row = {
                        "hole_id": img.hole_id,
                        "depth_from": img.depth_from,
                        "depth_to": img.depth_to,
                        "moisture_status": img.moisture_status or "",
                    }

                    # Collect all classifications by reviewer
                    classifications_by_reviewer = {}

                    # Add current user's classification
                    if img.classified_by:
                        # Get tags for current user
                        tags_value = img.tags if hasattr(img, "tags") and img.tags else []
                        if tags_value and isinstance(tags_value, (list, set)):
                            tags_str = ", ".join(sorted(tags_value))
                        else:
                            tags_str = ""
                        
                        classifications_by_reviewer[img.classified_by] = {
                            "classification": (
                                self._get_classification_string(img.classification)
                                if hasattr(img.classification, "value")
                                else str(img.classification)
                            ),
                            "date": img.classified_date or "",
                            "comments": img.comments or "",
                            "tags": tags_str,
                        }

                    # Add other reviewers' classifications
                    if hasattr(img, "other_reviews") and img.other_reviews:
                        for review in img.other_reviews:
                            reviewer = review.get("Reviewed_By")
                            if not reviewer:
                                continue

                            # Get classification value
                            classification_value = None
                            for field_name in [
                                "classification",
                                "Classification",
                                "Lithology",
                                "Rock_Type",
                            ]:
                                if field_name in review and review[field_name]:
                                    classification_value = review[field_name]
                                    break

                            if classification_value:
                                # Get tags if available
                                tags_value = review.get("tags", [])
                                if tags_value and isinstance(tags_value, (list, set)):
                                    tags_str = ", ".join(sorted(tags_value))
                                else:
                                    tags_str = ""
                                
                                classifications_by_reviewer[reviewer] = {
                                    "classification": str(classification_value),
                                    "date": review.get("Review_Date", ""),
                                    "comments": review.get("Comments", ""),
                                    "tags": tags_str,
                                }

                    # Fill in classification columns for each reviewer
                    for reviewer in reviewer_list:
                        if reviewer in classifications_by_reviewer:
                            row[f"classification_{reviewer}"] = (
                                classifications_by_reviewer[reviewer]["classification"]
                            )
                            row[f"review_date_{reviewer}"] = (
                                classifications_by_reviewer[reviewer]["date"]
                            )
                            row[f"comments_{reviewer}"] = classifications_by_reviewer[
                                reviewer
                            ]["comments"]
                            row[f"tags_{reviewer}"] = classifications_by_reviewer[
                                reviewer
                            ].get("tags", "")
                        else:
                            row[f"classification_{reviewer}"] = ""
                            row[f"review_date_{reviewer}"] = ""
                            row[f"comments_{reviewer}"] = ""
                            row[f"tags_{reviewer}"] = ""

                    # Calculate consensus and agreement
                    all_classifications = [
                        data["classification"]
                        for data in classifications_by_reviewer.values()
                    ]
                    row["review_count"] = len(all_classifications)

                    if len(all_classifications) == 0:
                        row["consensus_classification"] = ""
                        row["agreement"] = ""
                    elif len(all_classifications) == 1:
                        # Only one reviewer - no agreement/disagreement possible
                        row["consensus_classification"] = all_classifications[0]
                        row["agreement"] = "Single"
                    else:
                        # Multiple reviewers
                        from collections import Counter

                        classification_counts = Counter(all_classifications)
                        most_common = classification_counts.most_common(1)[0]
                        row["consensus_classification"] = most_common[0]
                        # Agreement = Yes if everyone agrees, No if there's disagreement
                        row["agreement"] = (
                            "Yes"
                            if most_common[1] == len(all_classifications)
                            else "No"
                        )

                    # Add CSV data
                    row.update(img.csv_data)
                    writer.writerow(row)

            self.logger.info(
                f"Exported classifications to {filepath} with {len(reviewer_list)} reviewer columns"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to export CSV: {e}")
            return False


# ============================================================================
# COLUMN MAPPING DIALOG
# ============================================================================


class ColumnMappingDialog:
    """Dialog for mapping CSV columns to required fields"""

    def __init__(self, parent, required_field: str, available_columns: List[str]):
        self.parent = parent
        self.required_field = required_field
        self.available_columns = available_columns
        self.selected = None

    def show(self) -> Optional[str]:
        """Show dialog and return selected column"""
        dialog = tk.Toplevel(self.parent)
        dialog.title("Column Mapping Required")
        dialog.transient(self.parent)
        dialog.grab_set()

        # Center dialog
        dialog.geometry("400x200")
        x = (dialog.winfo_screenwidth() - 400) // 2
        y = (dialog.winfo_screenheight() - 200) // 2
        dialog.geometry(f"+{x}+{y}")

        # Message
        msg = ttk.Label(
            dialog,
            text=f"Please select the column that contains '{self.required_field}':",
            wraplength=350,
        )
        msg.pack(pady=10)

        # Column selection
        var = tk.StringVar()
        combo = ttk.Combobox(
            dialog, textvariable=var, values=self.available_columns, state="readonly"
        )
        combo.pack(pady=10, padx=20, fill=tk.X)

        # If there's a likely match, select it
        for col in self.available_columns:
            if self.required_field.lower() in col.lower():
                combo.set(col)
                break

        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)

        ttk.Button(
            button_frame, text="OK", command=lambda: self._on_ok(dialog, var)
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(
            side=tk.LEFT, padx=5
        )

        dialog.wait_window()
        return self.selected

    def _on_ok(self, dialog, var):
        """Handle OK button"""
        if var.get():
            self.selected = var.get()
        dialog.destroy()


# ============================================================================
# DATASET MAPPING DIALOG
# ============================================================================


class DatasetMappingDialog:
    """Dialog for mapping dataset columns to HoleID and Depth To"""

    def __init__(self, parent, gui_manager, file_path):
        self.parent = parent
        self.gui_manager = gui_manager
        self.file_path = file_path
        self.result = None
        self.columns = self._get_file_columns()

    def _get_file_columns(self):
        """Get column names from file"""
        try:
            ext = os.path.splitext(self.file_path)[1].lower()
            if ext == ".csv":
                df = pd.read_csv(self.file_path, nrows=5)
            elif ext in [".xlsx", ".xls"]:
                df = pd.read_excel(self.file_path, nrows=5)
            elif ext == ".txt":
                df = pd.read_csv(self.file_path, sep="\t", nrows=5)
            else:
                return []
            return list(df.columns)
        except:
            return []

    def show(self):
        """Show the dialog and return mapping"""
        if not self.columns:
            DialogHelper.show_message(
                self.parent,
                "Error",
                "Could not read columns from file",
                message_type="error",
            )
            return None

        # Create dialog
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("Map Dataset Columns")
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        self.dialog.geometry("500x250")

        # Center dialog
        x = (self.dialog.winfo_screenwidth() - 500) // 2
        y = (self.dialog.winfo_screenheight() - 250) // 2
        self.dialog.geometry(f"+{x}+{y}")

        # Apply theme
        self.dialog.configure(bg=self.gui_manager.theme_colors["background"])

        # Main frame
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # File info
        ttk.Label(
            main_frame,
            text=f"File: {os.path.basename(self.file_path)}",
            font=("Arial", 10, "bold"),
        ).pack(pady=(0, 20))

        ttk.Label(
            main_frame, text="Map the required columns:", font=("Arial", 10)
        ).pack(pady=(0, 10))

        # Mapping frame
        mapping_frame = ttk.Frame(main_frame)
        mapping_frame.pack(fill=tk.X, pady=10)

        # HoleID mapping
        ttk.Label(mapping_frame, text="Hole ID Column:", width=20).grid(
            row=0, column=0, sticky="w", pady=5
        )

        self.hole_var = tk.StringVar()

        # Use OptionMenu with gui_manager styling
        hole_dropdown = tk.OptionMenu(
            mapping_frame, self.hole_var, *self.columns  # Unpack columns as arguments
        )
        self.gui_manager.style_dropdown(hole_dropdown)
        hole_dropdown.config(width=25)
        hole_dropdown.grid(row=0, column=1, pady=5)

        # Auto-select if found
        for col in self.columns:
            if "hole" in col.lower() or "holeid" in col.lower():
                self.hole_var.set(col)
                break

        # Depth To mapping (explicit about To, not From)
        ttk.Label(
            mapping_frame, text="Depth To Column:", width=20, font=("Arial", 10, "bold")
        ).grid(row=1, column=0, sticky="w", pady=5)

        self.to_var = tk.StringVar()

        # Use OptionMenu with gui_manager styling
        to_dropdown = tk.OptionMenu(
            mapping_frame, self.to_var, *self.columns  # Unpack columns as arguments
        )
        self.gui_manager.style_dropdown(to_dropdown)
        to_dropdown.config(width=25)
        to_dropdown.grid(row=1, column=1, pady=5)

        # Auto-select if found
        for col in self.columns:
            if col.lower() in ["to", "sampto"] or "depth_to" in col.lower():
                self.to_var.set(col)
                break

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))

        ModernButton(
            button_frame,
            text="OK",
            color=self.gui_manager.theme_colors["accent_green"],
            command=self._on_ok,
            theme_colors=self.gui_manager.theme_colors,
        ).pack(side=tk.RIGHT, padx=(5, 0))

        ModernButton(
            button_frame,
            text="Cancel",
            color=self.gui_manager.theme_colors["secondary_bg"],
            command=self._on_cancel,
            theme_colors=self.gui_manager.theme_colors,
        ).pack(side=tk.RIGHT)

        self.dialog.wait_window()
        return self.result

    def _on_ok(self):
        """Handle OK button"""
        hole_col = self.hole_var.get()
        to_col = self.to_var.get()

        if not hole_col or not to_col:
            DialogHelper.show_message(
                self.dialog,
                "Error",
                "Please select both HoleID and Depth To columns",
                message_type="error",
            )
            return

        self.result = {"hole_col": hole_col, "to_col": to_col}
        self.dialog.destroy()

    def _on_cancel(self):
        """Handle Cancel button"""
        self.dialog.destroy()


# ============================================================================
# GRID CANVAS
# ============================================================================


class LithologyGridCanvas:
    """Advanced grid canvas with multi-select and classification"""

    @staticmethod
    def get_classification_text(classification):
        """Get text representation of classification, handling both string and enum"""
        if isinstance(classification, str):
            return classification
        elif hasattr(classification, "value"):
            return classification.value
        else:
            return str(classification)

    def __init__(self, parent, theme_colors):
        self.parent = parent
        self.theme_colors = theme_colors
        self.logger = logging.getLogger(__name__)

        # Create container
        self.container = ttk.Frame(parent)
        self.container.pack(fill=tk.BOTH, expand=True)

        # Create canvas (no horizontal scrollbar)
        self.canvas = tk.Canvas(
            self.container, bg=theme_colors["background"], highlightthickness=0
        )

        v_scrollbar = ttk.Scrollbar(
            self.container, orient="vertical", command=self.canvas.yview
        )

        self.canvas.configure(yscrollcommand=v_scrollbar.set)

        # Grid layout
        self.canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        # Grid settings - larger default size for better visibility
        self.base_cell_width = 300  # Increased from 160
        self.base_cell_height = 150  # Increased from 80
        self.scale_factor = 1.0
        self.cell_width = self.base_cell_width
        self.cell_height = self.base_cell_height
        self.padding = 0  # No padding between cells
        self.cols_per_row = 5
        self.rotation = 0

        # Tooltip support
        self.tooltip = None
        self.tooltip_id = None
        
        # Bind mouse events for tooltip
        self.canvas.bind("<Motion>", self._on_mouse_motion)
        self.canvas.bind("<Leave>", self._hide_tooltip)

        # Data visualization settings
        self.show_data_visualizations = True
        self.viz_column_width = 30  # Narrower visualization columns (legacy)
        self.viz_column_width_ratio = 0.12  # Viz columns take 12% of cell width (configurable)
        
        # Grid display settings (configurable via Viz Config dialog)
        self.show_cell_outlines = True  # Show/hide cell border outlines
        self.cell_outline_width = 2  # Default outline width (increases for classified)
        self.show_cell_labels = True  # Show/hide hole ID and depth labels
        self.show_classification_labels = True  # Show/hide classification/tag labels
        self.classification_label_position = "top-right"  # "top-right" or "top-left" (legacy)
        self.viz_column_width_ratio = 0.12  # Viz columns take 12% of cell width (configurable)
        self.viz_columns = [
            "Fe_pct_BEST",
            "SiO2_pct_BEST",
            "Al2O3_pct_BEST",
            "Logged_pct_CHHM",
        ]
        self.viz_column_configs = {}  # Will store color map configurations

        self.viz_column_labels = {
            "Fe_pct_BEST": "Fe%",
            "SiO2_pct_BEST": "SiO2%",
            "Al2O3_pct_BEST": "Al2O3%",
            "Logged_pct_CHHM": "CHHM%",
            "BIFf_2": "Type",
        }

        # Initialize color map manager
        self._init_color_maps()

        # Data column settings (for optional data display)
        self.data_column_width = 0  # Will be set when columns are configured
        self.data_columns = []  # List of data column configurations

        # Load visualization columns from config
        self.viz_columns = []
        self.viz_column_configs = {}
        self.viz_column_sizes = {}  # Per-column width/height settings
        # Will be loaded from config after initialization
        # Enable visualizations by default if columns are set
        self.show_data_visualizations = len(self.viz_columns) > 0

        # State
        self.all_images = []  # All available
        self.displayed_images = []  # Currently displayed
        self.cells = {}  # (row, col): cell_data
        self.cell_ids = {}  # Canvas item IDs
        self.image_refs = {}  # PhotoImage references

        # Lazy loading state
        self.visible_range = (0, 0)  # (start_idx, end_idx) of visible items
        self.loaded_cells = set()  # Set of indices that have been loaded
        self.pending_loads = set()  # Set of indices currently being loaded
        self.load_buffer = 100  # Number of rows to preload outside viewport

        # Selection and mode
        self.current_mode = "biff"
        self.selected_indices = set()
        self.last_selected_index = None  # Track last selection for comments
        self.persistent_selection = None  # Keep selection even after classification
        self.last_selected_indices = set()

        # Drag selection
        self.drag_start = None
        self.selection_box = None
        self.dragging = False

        # Throttle drag selection updates
        self._drag_last_ts = 0.0
        self._drag_pending = False

        # Auto-scroll while dragging
        self.auto_scroll_timer = None
        self.auto_scroll_speed = 0  # Starts at 0, accelerates
        self.auto_scroll_direction = None  # 'up' or 'down'
        self.scroll_edge_threshold = 50  # pixels from edge to trigger scroll

        # Scatter filter window (reusing zoom_preview name for compatibility)
        self.zoom_preview = None
        self.hover_cell = None

        # Color mapping for modes
        # DEPRECATED: mode_colors kept for backwards compatibility only
        # Colors now come from ClassificationManager via dialog_ref
        self.mode_colors = {
            "biff": "#4CAF50",  # Green
            "bifhm": "#f44336",  # Red
            "other": "#2196F3",  # Blue
            "not_confident": "#FF9800",  # Orange
            "clear": "#9E9E9E",  # Gray
        }

        # Reference to parent dialog (set by LoggingReviewDialog)
        self.dialog_ref = None

        # Bind events
        self._bind_events()

        self.logger.info("LithologyGridCanvas initialized")

    def _calculate_visible_range(self):
        """Calculate which images should be visible/loaded based on viewport"""
        # Get viewport bounds
        canvas_height = self.canvas.winfo_height()
        view_top = self.canvas.canvasy(0)
        view_bottom = self.canvas.canvasy(canvas_height)

        if canvas_height <= 1:  # Canvas not yet properly sized
            return (0, min(40, len(self.displayed_images)))  # Load more initially

        # Calculate row range with buffer
        row_height = self.cell_height + self.padding
        if row_height <= 0:
            return (0, 0)

        # Use load_buffer for preloading
        start_row = max(0, int(view_top // row_height) - self.load_buffer)
        end_row = int(view_bottom // row_height) + self.load_buffer + 1

        # Convert to image indices
        start_idx = max(0, start_row * self.cols_per_row)
        end_idx = min(len(self.displayed_images), (end_row + 1) * self.cols_per_row)

        return (start_idx, end_idx)

    def _on_mouse_motion(self, event):
        """Handle mouse motion for tooltips"""
        # Convert widget coordinates to canvas coordinates (required for scrolled canvas)
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # Find items under cursor using canvas coordinates
        items = self.canvas.find_overlapping(canvas_x, canvas_y, canvas_x, canvas_y)
        
        # Check if hovering over a comment indicator
        for item in items:
            tags = self.canvas.gettags(item)
            if "comment_indicator" in tags:
                # Find which cell this belongs to
                for tag in tags:
                    if tag.startswith("comment_"):
                        parts = tag.split("_")
                        if len(parts) == 3:
                            try:
                                row, col = int(parts[1]), int(parts[2])
                                if (row, col) in self.cell_ids:
                                    cell_data = self.cell_ids[(row, col)]
                                    if 'comment_data' in cell_data:
                                        self._show_tooltip(event, cell_data['comment_data'])
                                        return
                            except (ValueError, IndexError):
                                continue
        
        # No comment indicator found, hide tooltip
        self._hide_tooltip()
    
    def _show_tooltip(self, event, comment_data):
        """Show tooltip with comment text"""
        # Don't create duplicate tooltip
        if self.tooltip and self.tooltip.winfo_exists():
            # Update position if already showing
            self.tooltip.geometry(f"+{event.x_root + 15}+{event.y_root + 10}")
            return
        
        # Create tooltip window
        self.tooltip = tk.Toplevel(self.canvas)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{event.x_root + 15}+{event.y_root + 10}")
        
        # Create tooltip content
        frame = tk.Frame(
            self.tooltip,
            background="#FFFFDD",
            borderwidth=1,
            relief="solid"
        )
        frame.pack()
        
        # Main comment
        if comment_data.get('comments'):
            main_label = tk.Label(
                frame,
                text=f"Your comment:\n{comment_data['comments']}",
                background="#FFFFDD",
                foreground="#000000",
                font=("Arial", 9),
                justify="left",
                padx=8,
                pady=5
            )
            main_label.pack(anchor="w")
        
        # Other reviewers' comments
        if comment_data.get('other_comments'):
            separator = ttk.Separator(frame, orient='horizontal')
            separator.pack(fill=tk.X, padx=5, pady=3)
            
            for other in comment_data['other_comments']:
                other_label = tk.Label(
                    frame,
                    text=f"{other['reviewer']}:\n{other['text']}",
                    background="#FFFFDD",
                    foreground="#333333",
                    font=("Arial", 8, "italic"),
                    justify="left",
                    padx=8,
                    pady=3
                )
                other_label.pack(anchor="w")
    
    def _hide_tooltip(self, event=None):
        """Hide tooltip"""
        if self.tooltip:
            try:
                self.tooltip.destroy()
            except:
                pass
            self.tooltip = None

    def _load_viz_config(self):
        """Load visualization configuration from settings"""
        # Get config_manager from parent dialog
        if not self.dialog_ref or not hasattr(self.dialog_ref, 'config_manager'):
            self._use_default_viz_config()
            return
        
        saved_config = self.dialog_ref.config_manager.get("viz_columns")
        
        if saved_config and isinstance(saved_config, list):
            columns = []
            for config in saved_config:
                if isinstance(config, dict) and "column" in config and "color_map" in config:
                    col = config["column"]
                    color_map_name = config["color_map"]
                    column_size = config.get("column_size", 80)  # Default 80px
                    
                    # Get color map from dialog's manager
                    color_map = None
                    if self.dialog_ref and hasattr(self.dialog_ref, 'color_map_manager'):
                        color_map = self.dialog_ref.color_map_manager.get_preset(color_map_name)
                    if color_map:
                        columns.append(col)
                        self.viz_column_configs[col] = color_map
                        self.viz_column_sizes[col] = column_size
                        # Ensure all columns have a size (default 15%)
                        for col in self.viz_columns:
                            if col not in self.viz_column_sizes:
                                self.viz_column_sizes[col] = 15
                    else:
                        self.logger.warning(f"Color map '{color_map_name}' not found, skipping column '{col}'")
            
            if columns:
                self.viz_columns = columns
                self.show_data_visualizations = True
                self.logger.info(f"Loaded {len(columns)} viz columns from config")
            else:
                self._use_default_viz_config()
        else:
            self._use_default_viz_config()
    
    def _load_grid_display_settings(self):
        """Load grid display settings from config"""
        if not self.dialog_ref or not hasattr(self.dialog_ref, 'config_manager'):
            return
        
        config_manager = self.dialog_ref.config_manager
        
        # Load individual settings with defaults
        self.show_cell_outlines = config_manager.get("grid_show_outlines", True)
        self.cell_outline_width = config_manager.get("grid_outline_width", 2)
        self.show_cell_labels = config_manager.get("grid_show_cell_labels", True)
        self.show_classification_labels = config_manager.get("grid_show_classification_labels", True)
        self.classification_label_position = config_manager.get("grid_classification_label_position", "top-right")
        self.viz_column_height = config_manager.get("viz_column_height", 80)
        self.viz_column_font_size = config_manager.get("viz_column_font_size", 7)
        
        self.logger.debug(f"Loaded grid display settings: outlines={self.show_cell_outlines}, "
                         f"outline_width={self.cell_outline_width}, labels={self.show_cell_labels}, "
                         f"class_labels={self.show_classification_labels}, position={self.classification_label_position}")

    def _use_default_viz_config(self):
        """Use default visualization configuration"""
        self.viz_columns = [
            "Fe_pct_BEST",
            "SiO2_pct_BEST",
            "Al2O3_pct_BEST",
            "Logged_pct_CHHM",
        ]
        self.show_data_visualizations = True
        # Color maps are already initialized in __init__, no need to call again

    def _init_color_maps(self):
        """Initialize color maps for data visualizations"""
        # Color map manager will be accessed via dialog_ref when needed
        # We don't create our own instance here to avoid initialization issues
        # Default color maps will be created on-demand in the _get_*_color_map methods
        self.color_map_manager = None
        self.viz_column_configs = {}

    def _get_fe_color_map(self):
        """Get or create Fe grade color map"""
        # Try to get from dialog's color map manager first
        if self.dialog_ref and hasattr(self.dialog_ref, 'color_map_manager'):
            preset = self.dialog_ref.color_map_manager.get_preset("fe_grade")
            if preset:
                return preset

        # Create default Fe color map
        from processing.LoggingReviewStep.color_map_manager import (
            ColorMap,
            ColorMapType,
            ColorRange,
        )

        fe_map = ColorMap("Fe Grade", ColorMapType.NUMERIC)
        fe_map.add_range(ColorRange(0, 30, (224, 224, 224), "<30%"))
        fe_map.add_range(ColorRange(30, 40, (48, 48, 48), "30-40%"))
        fe_map.add_range(ColorRange(40, 45, (255, 0, 0), "40-45%"))
        fe_map.add_range(ColorRange(45, 50, (255, 255, 0), "45-50%"))
        fe_map.add_range(ColorRange(50, 54, (0, 255, 0), "50-54%"))
        fe_map.add_range(ColorRange(54, 56, (0, 255, 255), "54-56%"))
        fe_map.add_range(ColorRange(56, 58, (0, 128, 255), "56-58%"))
        fe_map.add_range(ColorRange(58, 60, (0, 0, 255), "58-60%"))
        fe_map.add_range(ColorRange(60, 100, (255, 0, 255), ">60%"))
        return fe_map

    def _get_sio2_color_map(self):
        """Get or create SiO2 grade color map"""
        # Try to get from dialog's color map manager first
        if self.dialog_ref and hasattr(self.dialog_ref, 'color_map_manager'):
            preset = self.dialog_ref.color_map_manager.get_preset("sio2_grade")
            if preset:
                return preset

        from processing.LoggingReviewStep.color_map_manager import (
            ColorMap,
            ColorMapType,
            ColorRange,
        )

        sio2_map = ColorMap("SiO2 Grade", ColorMapType.NUMERIC)
        sio2_map.add_range(ColorRange(0, 5, (255, 255, 255), "<5%"))
        sio2_map.add_range(ColorRange(5, 15, (0, 255, 0), "5-15%"))
        sio2_map.add_range(ColorRange(15, 35, (255, 0, 0), "15-35%"))
        sio2_map.add_range(ColorRange(35, 100, (0, 0, 255), ">35%"))
        return sio2_map

    def _get_al2o3_color_map(self):
        """Get or create Al2O3 grade color map"""
        # Try to get from dialog's color map manager first
        if self.dialog_ref and hasattr(self.dialog_ref, 'color_map_manager'):
            preset = self.dialog_ref.color_map_manager.get_preset("al2o3_grade")
            if preset:
                return preset

        from processing.LoggingReviewStep.color_map_manager import (
            ColorMap,
            ColorMapType,
            ColorRange,
        )

        al2o3_map = ColorMap("Al2O3 Grade", ColorMapType.NUMERIC)
        al2o3_map.add_range(ColorRange(0, 2, (255, 0, 255), "<2%"))
        al2o3_map.add_range(ColorRange(2, 5, (0, 0, 255), "2-5%"))
        al2o3_map.add_range(ColorRange(5, 10, (0, 165, 255), "5-10%"))
        al2o3_map.add_range(ColorRange(10, 15, (0, 255, 255), "10-15%"))
        al2o3_map.add_range(ColorRange(15, 100, (0, 0, 255), ">15%"))
        return al2o3_map

    def _get_chhm_color_map(self):
        """Get or create CHHM logged percentage color map"""
        # Try to get from dialog's color map manager first
        if self.dialog_ref and hasattr(self.dialog_ref, 'color_map_manager'):
            preset = self.dialog_ref.color_map_manager.get_preset("logged_pct_chhm")
            if preset:
                return preset

        from processing.LoggingReviewStep.color_map_manager import (
            ColorMap,
            ColorMapType,
            ColorRange,
        )

        chhm_map = ColorMap("CHHM Logged", ColorMapType.NUMERIC)
        chhm_map.add_range(ColorRange(0, 20, (255, 255, 255), "0-20%"))
        chhm_map.add_range(ColorRange(20, 40, (200, 200, 255), "20-40%"))
        chhm_map.add_range(ColorRange(40, 60, (150, 150, 255), "40-60%"))
        chhm_map.add_range(ColorRange(60, 80, (100, 100, 255), "60-80%"))
        chhm_map.add_range(ColorRange(80, 100, (50, 50, 255), "80-100%"))
        return chhm_map

    def _bind_events(self):
        """Bind canvas events"""
        self.canvas.bind("<Button-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_release)
        self.canvas.bind("<Motion>", self._on_mouse_motion)
        self.canvas.bind("<Leave>", self._on_mouse_leave)
        self.canvas.bind("<Button-3>", self._on_right_click)

        # Add configure event for lazy loading
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Scrolling - bind to canvas only, not all widgets
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Control-MouseWheel>", self._on_ctrl_mousewheel)

    def _on_canvas_configure(self, event):
        """Handle canvas configuration changes"""
        # Trigger lazy load update when canvas is resized
        if self.displayed_images:
            self._update_lazy_load()

    def _load_cell_content(self, row: int, col: int, idx: int, img: CompartmentImage):
        """Load actual content for a cell (image, borders, labels, etc.)"""
        # Skip if already loaded
        if idx in self.loaded_cells:
            return

        # Get cell data
        if (row, col) not in self.cells:
            return

        cell_data = self.cells[(row, col)]
        x = cell_data["x"]
        y = cell_data["y"]

        # Remove placeholder if exists
        if "placeholder" in cell_data:
            self.canvas.delete(cell_data["placeholder"])

        # IMPORTANT: Ensure CSV data is populated before creating cell with visualizations
        if self.show_data_visualizations and not img.csv_data:
            # Get reference to parent dialog to access CSV handler
            if self.dialog_ref and hasattr(self.dialog_ref, "_get_csv_data_cached"):
                csv_data, in_csv = self.dialog_ref._get_csv_data_cached(
                    img.hole_id, img.depth_to
                )
                # csv_data should already contain actual CSV columns from CSVHandler
                # Do NOT unwrap review_data - that was causing the problem
                if csv_data:
                    img.csv_data = csv_data
                    img.in_csv = in_csv

                # Debug log for first few cells
                if not hasattr(self, "_viz_debug_count"):
                    self._viz_debug_count = 0
                if self._viz_debug_count < 5:
                    self.logger.debug(
                        f"Lazy load cell {idx}: CSV data loaded with {len(csv_data) if csv_data else 0} columns"
                    )
                    if csv_data and "Fe_pct_BEST" in csv_data:
                        self.logger.debug(f"  Fe_pct_BEST: {csv_data['Fe_pct_BEST']}")
                    self._viz_debug_count += 1

        # Create the actual cell content
        self._create_cell(row, col, idx, img)

        # Apply selection state if this cell is selected
        if idx in self.selected_indices:
            self._update_cell_border(row, col, idx)

    def load_images(self, images: List[CompartmentImage], preserve_selection=False):
        """Load and display images in grid"""
        # Count placeholders vs regular images
        placeholder_count = sum(
            1 for img in images if hasattr(img, "is_placeholder") and img.is_placeholder
        )
        regular_count = len(images) - placeholder_count

        self.logger.info(
            f"Loading {len(images)} images into grid ({regular_count} regular, {placeholder_count} placeholders)"
        )

        if placeholder_count > 0:
            self.logger.debug(
                f"Placeholder images present - will display as 'Image Pending' boxes"
            )

        # Only show progress for significant number of images
        use_progress = len(images) > 10
        progress = None

        if use_progress:
            # Check if there's already a progress dialog active
            existing_progress = ProgressDialogManager._current_dialog
            if existing_progress:
                # Reuse existing dialog, just update message
                progress = existing_progress
                progress.update_progress(
                    f"Loading {len(images)} images into grid...", 50
                )
            else:
                # Create new dialog
                progress = ProgressDialogManager.show_progress(
                    self.parent,
                    "Loading Images",
                    f"Preparing to load {len(images)} images...",
                )

        def update_progress(message, percentage):
            if progress:
                progress.update_progress(message, percentage)
                # Process events to keep UI responsive
                self.parent.update_idletasks()

        # Store previous selection if preserving (use tuple of filename and hole_id + depth for uniqueness)
        prev_selection = set()
        if preserve_selection:
            for idx in self.selected_indices:
                if idx < len(self.displayed_images):
                    img = self.displayed_images[idx]
                    prev_selection.add((img.filename, img.hole_id, img.depth_to))

        # Update images
        self.displayed_images = images

        # Clear canvas and references
        update_progress("Clearing previous images...", 5)
        self.canvas.delete("all")
        self.cells.clear()
        self.cell_ids.clear()
        # Explicitly clear image references to free memory
        for ref in self.image_refs.values():
            del ref
        self.image_refs.clear()

        # Calculate optimal cell dimensions based on actual image aspect ratios
        update_progress("Calculating layout...", 10)
        if images:
            aspect_ratios = []

            for img in images[:20]:  # Sample first 20 images for performance
                if img.image is not None:
                    h, w = img.image.shape[:2]
                    # Account for rotation
                    if self.rotation in [90, 270]:
                        w, h = h, w

                    # Just use raw image aspect ratio
                    aspect_ratios.append(w / h)

        # # Calculate optimal cell dimensions based on actual image aspect ratios + viz space
        # update_progress("Calculating layout...", 10)
        # if images:
        #     aspect_ratios = []
            
        #     # Calculate viz space needed based on per-column sizes
        #     viz_column_sizes = getattr(self, 'viz_column_sizes', {})
        #     if self.show_data_visualizations and self.viz_columns:
        #         if self.rotation in [0, 180]:  # Vertical - viz on right (stacked vertically)
        #             # In vertical mode, columns stack vertically, so we need max width
        #             viz_width_needed = max([viz_column_sizes.get(col, 80) for col in self.viz_columns], default=80)
        #             viz_height_needed = 0  # Doesn't affect width calculation
        #         else:  # Horizontal - viz on bottom (arranged horizontally)
        #             # In horizontal mode, columns arrange horizontally, so we need sum of widths
        #             viz_width_needed = 0  # Doesn't affect height calculation
        #             viz_height_needed = sum([viz_column_sizes.get(col, 80) for col in self.viz_columns])
        #     else:
        #         viz_width_needed = 0
        #         viz_height_needed = 0

        #     for img in images[:20]:  # Sample first 20 images for performance
        #         if img.image is not None:
        #             h, w = img.image.shape[:2]
        #             # Account for rotation
        #             if self.rotation in [90, 270]:
        #                 w, h = h, w

        #             # Just use raw aspect ratio - viz space handled in cell layout
        #             aspect_ratios.append(w / h)

            if aspect_ratios:
                # Use median aspect ratio for consistent layout
                median_aspect = sorted(aspect_ratios)[len(aspect_ratios) // 2]

                # Calculate optimal cell dimensions
                canvas_width = self.canvas.winfo_width()
                if canvas_width > 100:
                    # Determine target columns based on aspect ratio
                    if median_aspect < 0.8:  # Portrait
                        target_cols = 5
                    elif median_aspect > 1.5:  # Landscape
                        target_cols = 3
                    else:  # Square-ish
                        target_cols = 4

                    # Calculate base image dimensions (before adding viz space)
                    base_img_width = (canvas_width - 10) // target_cols  # Small margin
                    base_img_height = int(base_img_width / median_aspect)

                    # Apply reasonable limits to image size
                    max_height = int(canvas_width * 0.6)  # Max 60% of width for height
                    if base_img_height > max_height:
                        base_img_height = max_height
                        base_img_width = int(base_img_height * median_aspect)
                    
                    # Calculate viz space to ADD to cell dimensions
                    viz_column_sizes = getattr(self, 'viz_column_sizes', {})
                    viz_add_width = 0
                    viz_add_height = 0
                    
                    self.logger.debug("="*60)
                    self.logger.debug("LOAD_IMAGES VIZ CALCULATION")
                    self.logger.debug(f"viz_column_sizes dict: {viz_column_sizes}")
                    self.logger.debug(f"viz_columns list: {self.viz_columns}")
                    self.logger.debug(f"show_data_visualizations: {self.show_data_visualizations}")
                    self.logger.debug(f"rotation: {self.rotation}")
                    
                    if self.show_data_visualizations and self.viz_columns:
                        if self.rotation in [0, 180]:  # Vertical - viz on right (stacked vertically)
                            # Size controls HEIGHT of each column
                            # Width: all columns share same width (take max as the viz area width)
                            column_sizes = [viz_column_sizes.get(col, 80) for col in self.viz_columns]
                            viz_add_width = max(column_sizes, default=80)
                            # Height: columns stack, so no addition to cell height (they fit within image height)
                            self.logger.debug(f"Vertical mode - column sizes (will be heights): {column_sizes}")
                            self.logger.debug(f"Vertical mode - viz_add_width (shared width): {viz_add_width}")
                        else:  # Horizontal - viz on bottom (arranged horizontally)
                            # Size controls WIDTH of each column
                            # Width: columns arrange side-by-side, so no addition (they fit within image width)
                            # Height: all columns share same height (take max as the viz area height)
                            column_sizes = [viz_column_sizes.get(col, 80) for col in self.viz_columns]
                            viz_add_height = max(column_sizes, default=80)
                            self.logger.debug(f"Horizontal mode - column sizes (will be widths): {column_sizes}")
                            self.logger.debug(f"Horizontal mode - viz_add_height (shared height): {viz_add_height}")
                    
                    # Cell size = image size + viz space
                    self.base_cell_width = base_img_width + viz_add_width
                    self.base_cell_height = base_img_height + viz_add_height
                    
                    self.logger.debug(f"base_img_width: {base_img_width}, base_img_height: {base_img_height}")
                    self.logger.debug(f"base_cell_width: {self.base_cell_width}, base_cell_height: {self.base_cell_height}")
                    self.logger.debug("="*60)

                # Apply scale factor
                self.cell_width = int(self.base_cell_width * self.scale_factor)
                self.cell_height = int(self.base_cell_height * self.scale_factor)

                # Recalculate columns to fit
                self.cols_per_row = max(1, (canvas_width - 10) // self.cell_width)

                self.logger.debug(
                    f"Optimized cell dimensions: {self.cell_width}x{self.cell_height} "
                    f"(aspect: {median_aspect:.2f}, cols: {self.cols_per_row})"
                )

        # Calculate grid dimensions
        canvas_width = self.canvas.winfo_width()
        if canvas_width > 100:
            # PEP-8: self.cell_width already includes viz width; do not add again
            effective_cell_width = self.cell_width
            self.cols_per_row = max(1, canvas_width // effective_cell_width)

        self.logger.debug(
            f"Grid layout: {self.cols_per_row} columns, cell size {self.cell_width}x{self.cell_height}"
        )

        # Update images
        self.displayed_images = images

        # If no images, clear everything and return early
        if not images:
            self.canvas.delete("all")
            self.cells.clear()
            self.cell_ids.clear()
            for ref in self.image_refs.values():
                del ref
            self.image_refs.clear()
            self.loaded_cells.clear()
            self.pending_loads.clear()
            self.canvas.configure(scrollregion=(0, 0, 0, 0))
            self.logger.info("No images to display - grid cleared")
            if use_progress and progress:
                progress.update_progress("No images to display", 100)
                if not (hasattr(progress, "_reused") and progress._reused):
                    progress.close()
                    ProgressDialogManager.close_current()
            return

        # Clear canvas and references (existing code continues...)

        # Clear loaded cells tracking
        self.loaded_cells.clear()
        self.pending_loads.clear()

        # Calculate initial visible range
        self.visible_range = self._calculate_visible_range()
        start_idx, end_idx = self.visible_range

        # Create placeholders for ALL images (lightweight)
        for idx in range(len(images)):
            row = idx // self.cols_per_row
            col = idx % self.cols_per_row
            # PEP-8: step exactly one cell; self.cell_width already includes viz
            col_step = self.cell_width
            x = col * col_step
            y = row * self.cell_height

            # Just create a placeholder rectangle
            placeholder_id = self.canvas.create_rectangle(
                x,
                y,
                x + self.cell_width,
                y + self.cell_height,
                fill="#2a2a2a",
                outline="#3a3a3a",
                tags=(f"placeholder_{idx}", f"cell_{idx}"),
            )

            # Store minimal cell data
            self.cells[(row, col)] = {
                "idx": idx,
                "image": images[idx],
                "x": x,
                "y": y,
                "placeholder": placeholder_id,
            }

        # NOW load only visible cells with images (OUTSIDE the placeholder loop!)
        total_to_load = min(end_idx - start_idx, len(images))
        for i, idx in enumerate(range(start_idx, min(end_idx, len(images)))):
            row = idx // self.cols_per_row
            col = idx % self.cols_per_row
            img = images[idx]

            # Ensure CSV data is populated for initial load
            if self.show_data_visualizations and not img.csv_data:
                if self.dialog_ref and hasattr(self.dialog_ref, "_get_csv_data_cached"):
                    csv_data, in_csv = self.dialog_ref._get_csv_data_cached(
                        img.hole_id, img.depth_to
                    )
                    # Only set if we got valid CSV data
                    if csv_data:
                        img.csv_data = csv_data
                    img.in_csv = in_csv

                    # Debug first few
                    if i < 5:
                        self.logger.debug(
                            f"Initial load cell {idx}: CSV data = {len(csv_data) if csv_data else 0} columns"
                        )
                        if csv_data and "Fe_pct_BEST" in csv_data:
                            self.logger.debug(
                                f"  Fe_pct_BEST: {csv_data['Fe_pct_BEST']}"
                            )

            self._load_cell_content(row, col, idx, img)
            self.loaded_cells.add(idx)

            # Update progress for initial load
            if (
                use_progress
                and total_to_load > 10
                and (i % 10 == 0 or i == total_to_load - 1)
            ):
                percentage = 15 + (75 * (i + 1) / total_to_load)
                update_progress(
                    f"Loading visible image {i + 1} of {total_to_load}...", percentage
                )

        # Restore selection
        if preserve_selection:
            self.selected_indices.clear()
            for idx, img in enumerate(images):
                if (img.filename, img.hole_id, img.depth_to) in prev_selection:
                    self.selected_indices.add(idx)
                    correct_row = idx // self.cols_per_row
                    correct_col = idx % self.cols_per_row
                    self._update_cell_border(correct_row, correct_col, idx)

        # Calculate proper scroll region
        update_progress("Finalizing layout...", 95)
        if images:
            rows = (len(images) + self.cols_per_row - 1) // self.cols_per_row
            total_width = self.cols_per_row * (self.cell_width + self.padding)
            total_height = rows * (self.cell_height + self.padding)
            self.canvas.configure(scrollregion=(0, 0, total_width, total_height))
        else:
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        self.logger.info(f"Grid loaded with {len(self.cells)} cells")
        
        # Draw classification labels AFTER all cells are created
        # This ensures consistent display including other_reviews data
        self._refresh_classifications()

        # Complete and close progress dialog
        update_progress("Complete!", 100)
        if progress:
            # Only close if we created it (not reusing)
            if not (hasattr(progress, "_reused") and progress._reused):
                progress.close()
                ProgressDialogManager.close_current()

    def _get_data_value(self, img: CompartmentImage, source: str, column: str):
        """Get data value for image from specified source and column."""
        # This would interface with the data manager to get values
        # For now, return from csv_data if available
        if column in img.csv_data:
            return img.csv_data[column]
        return None

    def _create_cell(self, row: int, col: int, idx: int, img: CompartmentImage):
        """Create a single grid cell with data columns"""
        # Calculate positions - tight packing
        # PEP-8: step one full cell; avoid double-counting viz width
        col_step = self.cell_width
        x = col * col_step
        y = row * self.cell_height

        # Calculate visualization dimensions using viz_column_width_ratio
        # Columns auto-size to fill the allocated viz area equally
        viz_column_width_ratio = getattr(self, 'viz_column_width_ratio', 0.30)
        viz_total_width = 0
        viz_total_height = 0

        if idx == 0:  # Only log for first cell to avoid spam
            self.logger.debug("="*60)
            self.logger.debug("_CREATE_CELL VIZ CALCULATION (AUTO-SIZE)")
            self.logger.debug(f"Cell {idx} (row={row}, col={col})")
            self.logger.debug(f"viz_column_width_ratio: {viz_column_width_ratio}")
            self.logger.debug(f"viz_columns count: {len(self.viz_columns) if self.viz_columns else 0}")
            self.logger.debug(f"self.cell_width: {self.cell_width}, self.cell_height: {self.cell_height}")
            self.logger.debug(f"rotation: {self.rotation}")

        if self.show_data_visualizations and self.viz_columns:
            if self.rotation in [0, 180]:  # Vertical - viz on right (columns stack vertically)
                # Viz area takes ratio of cell width, full cell height
                viz_total_width = int(self.cell_width * viz_column_width_ratio)
                viz_total_height = self.cell_height

                if idx == 0:
                    self.logger.debug(f"Vertical mode - viz takes {viz_column_width_ratio*100:.0f}% of width")
                    self.logger.debug(f"Vertical mode - viz_total_width: {viz_total_width}")
                    self.logger.debug(f"Vertical mode - viz_total_height: {viz_total_height}")
            else:  # Horizontal - viz on bottom (columns arrange horizontally)
                # Viz area takes full cell width, ratio of cell height
                viz_total_width = self.cell_width
                viz_total_height = int(self.cell_height * viz_column_width_ratio)

                if idx == 0:
                    self.logger.debug(f"Horizontal mode - viz takes {viz_column_width_ratio*100:.0f}% of height")
                    self.logger.debug(f"Horizontal mode - viz_total_width: {viz_total_width}")
                    self.logger.debug(f"Horizontal mode - viz_total_height: {viz_total_height}")

            if idx == 0:
                self.logger.debug("="*60)

        # Calculate image allocation with safeguards
        if self.rotation in [0, 180]:  # Normal orientation - viz on right
            image_alloc_width = max(50, self.cell_width - viz_total_width)  # Minimum 50px
            image_alloc_height = max(50, self.cell_height)
            # For vertical orientation, image height shouldn't exceed cell height
            # even if viz columns have heights
        else:  # Rotated 90 or 270 - viz on bottom
            image_alloc_width = max(50, self.cell_width)
            image_alloc_height = max(50, self.cell_height - viz_total_height)  # Minimum 50px

        # No background rectangle
        bg_rect = self.canvas.create_rectangle(
            x, y, x + self.cell_width, y + self.cell_height, fill="", outline=""
        )

        # Load and get actual image dimensions
        photo = self._prepare_thumbnail(img, image_alloc_width, image_alloc_height)

        # Track actual displayed image bounds
        actual_img_width = image_alloc_width
        actual_img_height = image_alloc_height
        actual_img_x = x
        actual_img_y = y

        if photo:
            self.image_refs[(row, col)] = photo
            # Get actual photo dimensions
            actual_img_width = photo.width()
            actual_img_height = photo.height()

            # Position image to fill allocated space
            img_id = self.canvas.create_image(
                x,
                y,
                image=photo,
                anchor="nw",
                tags=("cell_image", f"cell_{idx}"),
            )

            # Store actual bounds for visualization alignment
            actual_img_x = x
            actual_img_y = y
        else:
            # Check if this is a placeholder or genuinely missing image
            is_placeholder = hasattr(img, "is_placeholder") and img.is_placeholder

            if is_placeholder:
                # Create a gray placeholder rectangle
                placeholder_rect = self.canvas.create_rectangle(
                    x + 2,
                    y + 2,
                    x + image_alloc_width - 2,
                    y + image_alloc_height - 2,
                    fill="#3a3a3a",
                    outline="#555555",
                    width=1,
                    tags=("cell_placeholder", f"cell_{idx}"),
                )

                # "Image Pending" text with larger, more visible font
                img_id = self.canvas.create_text(
                    x + image_alloc_width // 2,
                    y + image_alloc_height // 2,
                    text="Image\nPending",
                    fill="#888888",
                    font=("Arial", 12, "italic"),
                    tags=("cell_image", f"cell_{idx}"),
                )

                # Add interval info at bottom
                interval_text = self.canvas.create_text(
                    x + image_alloc_width // 2,
                    y + image_alloc_height - 10,
                    text=f"{img.depth_from:.1f}-{img.depth_to:.1f}m",
                    fill="#999999",
                    font=("Arial", 9),
                    anchor="s",
                    tags=("cell_interval", f"cell_{idx}"),
                )
            else:
                # Regular "No Image" for actual missing files
                img_id = self.canvas.create_text(
                    x + image_alloc_width // 2,
                    y + image_alloc_height // 2,
                    text="No Image",
                    fill="white",
                    tags=("cell_image", f"cell_{idx}"),
                )

        # Add data visualizations aligned to actual image bounds
        # Show visualizations even for placeholders if they have CSV data
        if self.show_data_visualizations and (
            img.csv_data or not (hasattr(img, "is_placeholder") and img.is_placeholder)
        ):
            # Calculate viz position based on actual image
            if self.rotation in [0, 180]:  # Vertical - viz on right
                viz_x = actual_img_x + actual_img_width
                viz_y = actual_img_y
                viz_width = viz_total_width
                viz_height = viz_total_height  # Use calculated total height for stacked columns
                viz_orientation = "vertical"
            else:  # Horizontal - viz on bottom
                viz_x = actual_img_x
                viz_y = actual_img_y + actual_img_height
                viz_width = viz_total_width  # Use calculated total width for stacked columns
                viz_height = viz_total_height
                viz_orientation = "horizontal"

            self._add_data_visualizations(
                viz_x, viz_y, viz_width, viz_height, img, viz_orientation, idx
            )

        # Classification border - configurable outline
        border_id = None
        if getattr(self, 'show_cell_outlines', True):
            border_color = self._get_border_color(img)
            base_outline_width = getattr(self, 'cell_outline_width', 2)
            # Increase width for classified images
            border_width = (
                base_outline_width + 6 if img.classification != ClassificationCategory.UNASSIGNED else base_outline_width
            )

            # Check if saved (has classified_date means it's been saved at least once)
            is_saved = bool(
                img.classified_date
                and hasattr(self.parent.master, "last_save_state")
                and self.parent.master.last_save_state.get(img.filename)
                == img.classification
            )

            dash_pattern = None if is_saved else (5, 3)  # Dotted for unsaved

            border_id = self.canvas.create_rectangle(
                x,
                y,
                x + self.cell_width,
                y + self.cell_height,
                outline=border_color,
                width=border_width,
                dash=dash_pattern,
            )

        # Single-line label positioning based on rotation (if enabled)
        label_id = None
        label_bg = None
        if getattr(self, 'show_cell_labels', True):
            label_parts = [img.hole_id, f"{int(img.depth_to)}m"]
            if img.moisture_status:
                label_parts.append(img.moisture_status)
            label_text = " - ".join(label_parts)

            # Position label based on rotation
            if self.rotation in [90, 270]:
                # Horizontal orientation - place label on image, near top
                label_x = x + actual_img_width // 2
                label_y = y + 15
                label_anchor = "n"
                # Create smaller, more transparent background
                label_bg = self.canvas.create_rectangle(
                    label_x - 50,
                    label_y - 8,
                    label_x + 50,
                    label_y + 8,
                    fill="black",
                    outline="",
                    stipple="gray75",  # More transparent
                    tags=("cell_label_bg", f"cell_{idx}"),
                )
            else:
                # Vertical orientation - keep at bottom
                label_x = x + actual_img_width // 2  # Center in image width, not cell width
                label_y = y + self.cell_height - 3
                label_anchor = "s"

            # Don't create bottom label for placeholder images - they already have interval text
            if not (hasattr(img, "is_placeholder") and img.is_placeholder):
                label_id = self.canvas.create_text(
                    label_x,
                    label_y,
                    text=label_text,
                    fill="white",
                    font=("Arial", 8, "bold"),
                    anchor=label_anchor,
                    tags=("cell_label", f"cell_{idx}"),
                )

                # Ensure label is on top of image
                if label_id:
                    self.canvas.tag_raise(label_id)

            # Ensure label is on top of image
            if label_bg:
                self.canvas.tag_raise(label_bg)
            self.canvas.tag_raise(label_id)

        # Classification labels - SKIP drawing here, handled by _refresh_classifications()
        # Initialize variables that cell_ids expects
        class_labels_list = []
        class_label_id = None
        
        # # Build classification display (including tags)
        # classification_text_parts = []
        
        # # Add classification if assigned
        # if img.classification != ClassificationCategory.UNASSIGNED:
        #     class_text = self.get_classification_text(img.classification)
        #     classification_text_parts.append(class_text)
        
        # # Add tags
        # if hasattr(img, 'tags') and img.tags:
        #     tag_labels = []
        #     if self.dialog_ref and hasattr(self.dialog_ref, "item_manager"):
        #         for tag_id in sorted(img.tags):
        #             tag_def = self.dialog_ref.item_manager.get_tag(tag_id)
        #             if tag_def:
        #                 tag_text = f"{tag_def.icon} {tag_def.label}" if tag_def.icon else tag_def.label
        #                 tag_labels.append(tag_text)
        #     if tag_labels:
        #         classification_text_parts.extend(tag_labels)
        
        # # Create label if there's anything to display AND classification labels are enabled
        # class_label_id = None
        # if classification_text_parts and getattr(self, 'show_classification_labels', True):
        #     display_text = ' '.join(classification_text_parts)
        #     reviewer_name = img.classified_by if img.classified_by else "Unknown"
        #     full_text = f"{display_text} - by {reviewer_name}"
            
        #     # Get color from ClassificationManager
        #     if self.dialog_ref and hasattr(self.dialog_ref, "item_manager"):
        #         # Use the classification for color, not tags
        #         if img.classification != ClassificationCategory.UNASSIGNED:
        #             class_str = self.get_classification_text(img.classification)
        #             bg_color = self.dialog_ref.item_manager.get_color_for_classification(class_str)
        #         else:
        #             # Just tags, no classification - use first tag's color
        #             if hasattr(img, 'tags') and img.tags:
        #                 first_tag = next(iter(sorted(img.tags)))
        #                 bg_color = self.dialog_ref.item_manager.get_color_for_tag(first_tag)
        #             else:
        #                 bg_color = "#666666"
        #     else:
        #         bg_color = "#666666"
            
        #     text_width = len(full_text) * 5 + 8
            
        #     # Determine position based on configuration
        #     label_position = getattr(self, 'classification_label_position', 'top-right')
        #     if label_position == "top-left":
        #         # Top-left positioning
        #         bg_x1 = x + 5
        #         bg_x2 = x + 5 + text_width
        #         text_x = x + 5 + text_width // 2
        #     else:
        #         # Top-right positioning (default)
        #         bg_x1 = x + self.cell_width - text_width - 5
        #         bg_x2 = x + self.cell_width - 5
        #         text_x = x + self.cell_width - 5 - text_width // 2
            
        #     # Create background rectangle
        #     class_bg_id = self.canvas.create_rectangle(
        #         bg_x1,
        #         y + 5,
        #         bg_x2,
        #         y + 19,
        #         fill=bg_color,
        #         outline="",
        #         tags=f"class_bg_{row}_{col}",
        #     )
        #     class_labels_list.append(class_bg_id)
            
        #     # Create text label
        #     class_label_id = self.canvas.create_text(
        #         text_x,
        #         y + 12,
        #         text=full_text,
        #         fill="white",
        #         font=("Arial", 7, "bold"),
        #         anchor="center",
        #         tags=f"class_{row}_{col}",
        #     )
        #     class_labels_list.append(class_label_id)

        # # Warning overlay for missing CSV data
        warning_id = None
        if not img.in_csv:
            warning_id = self.canvas.create_text(
                x + self.cell_width // 2,
                y + 20,
                text="Missing Data!",
                fill="#ff0000",
                font=("Arial", 5, "bold"),
                anchor="center",
            )

        # Store cell data
        self.cells[(row, col)] = {"idx": idx, "image": img, "x": x, "y": y}
        self.cell_ids[(row, col)] = {
            "bg": bg_rect,
            "image": img_id,
            "border": border_id,
            "label": label_id,
            "warning": warning_id,
            "class_label": class_label_id,
            "class_labels": class_labels_list,  # Store list for consistent cleanup
        }

        # Draw data columns if configured
        if hasattr(self, "data_columns") and self.data_columns:
            data_x = x + self.cell_width

            for col_config in self.data_columns:
                # Draw column background
                col_rect = self.canvas.create_rectangle(
                    data_x,
                    y,
                    data_x + col_config["width"],
                    y + self.cell_height,
                    fill="",
                    outline=self.theme_colors["border"],
                    width=1,
                )

                # Get value for this column
                value = self._get_data_value(
                    img, col_config["source"], col_config["column"]
                )

                if value is not None:
                    # Get color from color map
                    color_map = col_config.get("color_map")
                    if color_map:
                        color = color_map.get_color(value)
                        # Fill with color
                        self.canvas.create_rectangle(
                            data_x + 1,
                            y + 1,
                            data_x + col_config["width"] - 1,
                            y + self.cell_height - 1,
                            fill=f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}",
                            outline="",
                        )

                    # Add value text
                    text_color = "white" if color and sum(color) < 384 else "black"
                    self.canvas.create_text(
                        data_x + col_config["width"] // 2,
                        y + self.cell_height // 2,
                        text=str(value)[:10],
                        fill=text_color,
                        font=("Arial", 8, "bold"),
                    )
                else:
                    # Show N/A
                    self.canvas.create_text(
                        data_x + col_config["width"] // 2,
                        y + self.cell_height // 2,
                        text="N/A",
                        fill="#888888",
                        font=("Arial", 8),
                    )

                data_x += col_config["width"]

        # Draw column headers (once per column)
        if row == 0 and hasattr(self, "data_columns"):
            header_x = x + self.cell_width
            for col_config in self.data_columns:
                # Rotated header text
                header_text = self.canvas.create_text(
                    header_x + col_config["width"] // 2,
                    y - 5,
                    text=col_config["name"],
                    fill=self.theme_colors["text"],
                    font=("Arial", 9, "bold"),
                    anchor="s",
                    angle=90 if col_config["width"] < 40 else 0,
                )
                header_x += col_config["width"]

    def _get_column_display_name(self, column_name: str) -> str:
        """
        Get user-friendly display name for a column from schema.
        
        Args:
            column_name: Column name (may include source suffix like "fe_pct_best (drillhole_data)")
            
        Returns:
            Display name from schema, or abbreviated column name as fallback
        """
        # Strip source suffix if present
        col_base = column_name.split(" (")[0].strip().lower() if " (" in column_name else column_name.lower()
        source_name = None
        if " (" in column_name and column_name.endswith(")"):
            source_name = column_name.split(" (")[1].rstrip(")")
        
        # Try to get display name from DataCoordinator -> GeologicalStore -> Schema
        if self.dialog_ref:
            data_coordinator = getattr(self.dialog_ref, 'data_coordinator', None)
            if data_coordinator and hasattr(data_coordinator, '_geological_store'):
                geo_store = data_coordinator._geological_store
                
                # Search through all sources (or specific source if provided)
                sources_to_check = [source_name] if source_name else list(geo_store._sources.keys())
                
                for src_name in sources_to_check:
                    if src_name not in geo_store._sources:
                        continue
                    indexed_source = geo_store._sources[src_name]
                    if indexed_source.schema:
                        for schema_col_name, col_schema in indexed_source.schema.columns.items():
                            if schema_col_name.lower() == col_base:
                                if col_schema.display_name:
                                    return col_schema.display_name
                                break
        
        # Check static label mappings as fallback
        static_labels = getattr(self, 'viz_column_labels', {})
        if column_name in static_labels:
            return static_labels[column_name]
        
        # Final fallback: abbreviate column name
        # "fe_pct_best" -> "Fe%"
        abbreviated = col_base.replace("_pct_", "%").replace("_best", "").replace("_", " ")
        # Title case, max 6 chars
        abbreviated = abbreviated.title()[:6]
        return abbreviated

    def _get_viz_value_with_fallback(self, img, column_name):
        """
        Get visualization value from csv_data, with fallback source support.

        Tries the primary column first, then any configured fallback sources
        in order until a valid value is found.

        Args:
            img: CompartmentImage with csv_data
            column_name: Primary column name

        Returns:
            Value from csv_data or None if no value found
        """
        if not img.csv_data:
            return None

        def get_value_for_column(col):
            """Get value from csv_data using case-insensitive lookup"""
            # Strip source suffix if present
            col_key = col.split(" (")[0].strip().lower() if " (" in col else col.lower()

            # Try direct lookup
            value = img.csv_data.get(col_key)
            if value is not None:
                return value

            # Try case-insensitive match
            for csv_key in img.csv_data.keys():
                if csv_key.lower() == col_key:
                    return img.csv_data[csv_key]

            return None

        # Try primary column
        value = get_value_for_column(column_name)
        if value is not None and not (isinstance(value, float) and value != value):  # NaN check
            return value

        # Try fallback sources if configured
        if column_name in self.viz_column_configs:
            config = self.viz_column_configs[column_name]
            if isinstance(config, dict):
                fallbacks = config.get('fallback_sources', [])
                for fallback_col in fallbacks:
                    value = get_value_for_column(fallback_col)
                    if value is not None and not (isinstance(value, float) and value != value):
                        return value

        return None

    def _add_data_visualizations(
        self, viz_x, viz_y, viz_width, viz_height, img, orientation, idx=0
    ):
        """Add data visualization bars aligned to actual image dimensions.

        Columns auto-size to fill available space equally, constrained within
        the allocated viz area.
        """
        # Debug logging
        if idx < 5:  # Log first few cells
            self.logger.debug(f"_add_data_visualizations called for cell {idx}")
            self.logger.debug(f"  viz_columns: {self.viz_columns}")
            self.logger.debug(f"  img.csv_data: {bool(img.csv_data)}")
            self.logger.debug(
                f"  show_data_visualizations: {self.show_data_visualizations}"
            )
            if img.csv_data:
                self.logger.debug(f"  csv_data keys: {list(img.csv_data.keys())[:5]}")

        if not self.viz_columns or not hasattr(img, "csv_data"):
            return

        # Get font size setting
        font_size = getattr(self, 'viz_column_font_size', 7)

        num_columns = len(self.viz_columns)
        if num_columns == 0:
            return

        # AUTO-SIZE: Distribute available space equally among columns
        # For vertical orientation: columns stack vertically, each gets equal HEIGHT
        # For horizontal orientation: columns arrange horizontally, each gets equal WIDTH
        if orientation == "vertical":
            # Each column gets equal height within the viz_height
            column_size_px = max(20, viz_height // num_columns)  # Min 20px
        else:  # horizontal
            # Each column gets equal width within the viz_width
            column_size_px = max(20, viz_width // num_columns)  # Min 20px

        # Calculate dynamic font size based on column size
        font_size = max(6, min(12, int(column_size_px / 10)))

        if idx == 0:  # Only log for first cell
            self.logger.debug("="*60)
            self.logger.debug("_ADD_DATA_VISUALIZATIONS (AUTO-SIZE)")
            self.logger.debug(f"viz_x={viz_x}, viz_y={viz_y}, viz_width={viz_width}, viz_height={viz_height}")
            self.logger.debug(f"orientation: {orientation}")
            self.logger.debug(f"num columns: {num_columns}")
            self.logger.debug(f"column_size_px (equal distribution): {column_size_px}")
            self.logger.debug(f"font_size: {font_size}")

        for i, column_name in enumerate(self.viz_columns):
            # Get value from csv_data with fallback support
            value = self._get_viz_value_with_fallback(img, column_name)

            # Get display name and font settings from config
            display_name = self._get_column_display_name(column_name)
            config_font_size = font_size  # Use auto-calculated font size
            config_bold = False
            custom_label = display_name

            # Get color and font settings from color map config
            color_hex = "#292828"  # Default gray
            if column_name in self.viz_column_configs:
                viz_config = self.viz_column_configs[column_name]

                # Handle both old format (color_map directly) and new format (dict with settings)
                if isinstance(viz_config, dict):
                    color_map = viz_config.get('color_map')
                    custom_label = viz_config.get('custom_label', display_name)
                    # Override font size only if explicitly set in config
                    if viz_config.get('font_size'):
                        config_font_size = viz_config.get('font_size', font_size)
                    config_bold = viz_config.get('bold', False)
                else:
                    # Old format - viz_config is the color_map directly
                    color_map = viz_config

                if color_map and value is not None:
                    color_bgr = color_map.get_color(value)
                    color_hex = f"#{color_bgr[2]:02x}{color_bgr[1]:02x}{color_bgr[0]:02x}"

            # Calculate text color based on brightness
            r = int(color_hex[1:3], 16)
            g = int(color_hex[3:5], 16)
            b = int(color_hex[5:7], 16)
            brightness = (r * 299 + g * 587 + b * 114) / 1000
            text_color = "white" if brightness < 128 else "black"

            # Build font tuple
            font_weight = "bold" if config_bold else "normal"
            text_font = ("Arial", config_font_size, font_weight)

            # Draw bar based on orientation - using equal distribution
            if orientation == "vertical":
                # Vertical bars - stacked vertically, each with EQUAL HEIGHT
                bar_width = viz_width  # Full width of viz area (shared by all)
                bar_height = column_size_px  # Equal height for each column
                bar_x = viz_x
                bar_y = viz_y + (i * column_size_px)  # Simple equal offset

                # Ensure bar doesn't exceed viz area
                if bar_y + bar_height > viz_y + viz_height:
                    bar_height = max(10, viz_y + viz_height - bar_y)

                if idx == 0:
                    self.logger.debug(f"  Vertical bar {i} - width={bar_width}, height={bar_height}, x={bar_x}, y={bar_y}")

                # Draw bar
                self.canvas.create_rectangle(
                    bar_x,
                    bar_y,
                    bar_x + bar_width,
                    bar_y + bar_height,
                    fill=color_hex,
                    outline="",
                    tags=("viz_bar", f"viz_{idx}"),
                )

                # Format value text
                if value is not None:
                    value_text = f"{value:.1f}" if isinstance(value, float) else str(value)
                else:
                    value_text = "-"

                # Draw text with width constraint to prevent overflow
                self.canvas.create_text(
                    bar_x + bar_width / 2,
                    bar_y + bar_height / 2,
                    text=f"{custom_label}\n{value_text}",
                    fill=text_color,
                    font=text_font,
                    anchor="center",
                    width=bar_width - 4,  # Constrain text width
                    tags=("viz_text", f"viz_{idx}"),
                )

            else:  # horizontal orientation
                # Horizontal bars - arranged horizontally, each with EQUAL WIDTH
                bar_height = viz_height  # Full height of viz area (shared by all)
                bar_width = column_size_px  # Equal width for each column
                bar_x = viz_x + (i * column_size_px)  # Simple equal offset
                bar_y = viz_y

                # Ensure bar doesn't exceed viz area
                if bar_x + bar_width > viz_x + viz_width:
                    bar_width = max(10, viz_x + viz_width - bar_x)

                if idx == 0:
                    self.logger.debug(f"  Horizontal bar {i} - width={bar_width}, height={bar_height}, x={bar_x}, y={bar_y}")

                # Draw bar
                self.canvas.create_rectangle(
                    bar_x,
                    bar_y,
                    bar_x + bar_width,
                    bar_y + bar_height,
                    fill=color_hex,
                    outline="",
                    tags=("viz_bar", f"viz_{idx}"),
                )

                # Format value text
                if value is not None:
                    value_text = f"{value:.1f}" if isinstance(value, float) else str(value)
                else:
                    value_text = "-"

                # Draw text with width constraint to prevent overflow
                self.canvas.create_text(
                    bar_x + bar_width / 2,
                    bar_y + bar_height / 2,
                    text=f"{custom_label}\n{value_text}",
                    fill=text_color,
                    font=text_font,
                    anchor="center",
                    width=bar_width - 4,  # Constrain text width
                    tags=("viz_text", f"viz_{idx}"),
                )

    def _prepare_thumbnail(
        self, img: CompartmentImage, target_width: int, target_height: int
    ):
        """Prepare thumbnail that fills the allocated space"""
        # Check if this is a placeholder image
        if hasattr(img, "is_placeholder") and img.is_placeholder:
            return None  # Return None for placeholders, handled in _create_cell

        # Access img.image property which triggers lazy loading automatically
        if img.image is None:
            return None

        # Get image and apply rotation
        image = img.image.copy()
        h, w = image.shape[:2]

        if self.rotation == 90:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            h, w = w, h
        elif self.rotation == 180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif self.rotation == 270:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            h, w = w, h

        # Calculate scale to fill the space (not fit within)
        scale_x = target_width / w
        scale_y = target_height / h

        # Use the larger scale to fill the space completely
        # This may crop some of the image but ensures no empty space
        # scale = max(scale_x, scale_y)

        # Prefer to fit within (showing full image with possible gaps):
        scale = min(scale_x, scale_y)

        new_width = int(w * scale)
        new_height = int(h * scale)

        # Resize image
        resized = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_AREA
        )

        # Crop to exact target size if needed (center crop)
        if new_width > target_width or new_height > target_height:
            x_offset = (new_width - target_width) // 2
            y_offset = (new_height - target_height) // 2
            resized = resized[
                y_offset : y_offset + target_height, x_offset : x_offset + target_width
            ]

        # Convert to PIL Image and then to PhotoImage
        pil_image = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        return ImageTk.PhotoImage(pil_image)

    def _get_classification_border_color(self, img) -> str:
        """
        Get border color based on classification - uses ClassificationManager.

        Legacy method name kept for backwards compatibility.
        Now delegates to _get_border_color().
        """
        return self._get_border_color(img)

    def _get_border_color(self, img: CompartmentImage) -> str:
        """Get border color based on classification - uses ClassificationManager"""
        # Check if showing other users' reviews (peer review mode)
        if hasattr(img, "other_classification") and img.other_classification:
            # Use lighter versions of colors for peer reviews
            other_class_str = str(img.other_classification).upper()

            # Try to get the base classification color and lighten it
            if self.dialog_ref and hasattr(self.dialog_ref, "item_manager"):
                base_color = self.dialog_ref.item_manager.get_color_for_classification(
                    str(img.other_classification)
                )
                # Return a lightened version (simple approach: use predefined light colors)
                # More sophisticated would be to programmatically lighten base_color
                if "BIFF" in other_class_str and "HM" not in other_class_str:
                    return "#90EE90"  # Light green
                elif "BIFHM" in other_class_str or "BIF HM" in other_class_str:
                    return "#FFB6C1"  # Light pink
                elif "OTHER" in other_class_str:
                    return "#ADD8E6"  # Light blue
                else:
                    return "#FFE4B5"  # Light orange
            else:
                # Fallback to hardcoded peer review colors
                if "BIFF" in other_class_str:
                    return "#90EE90"
                elif "BIFHM" in other_class_str:
                    return "#FFB6C1"
                elif "OTHER" in other_class_str:
                    return "#ADD8E6"
                else:
                    return "#FFE4B5"

        # Handle both string and enum classifications
        classification_str = str(img.classification)

        # Check for string representation of enum
        if "ClassificationCategory." in classification_str:
            classification_str = classification_str.replace(
                "ClassificationCategory.", ""
            )

        # Use ClassificationManager for colors
        if self.dialog_ref and hasattr(self.dialog_ref, "item_manager"):
            return self.dialog_ref.item_manager.get_color_for_classification(
                classification_str
            )

        # Fallback to mode_colors if manager not available (backwards compatibility)
        classification_lower = classification_str.lower()
        if classification_lower in self.mode_colors:
            return self.mode_colors[classification_lower]

        # Final fallback
        return self.theme_colors.get("border", "#666666")

    def _on_mouse_down(self, event):
        """Handle mouse down - start selection (PEP-8)"""
        self.current_event = event  # remember modifiers for additive selection
        self.drag_start = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        self.dragging = False

        # Check if clicking on a cell
        cell = self._get_cell_at(self.drag_start[0], self.drag_start[1])
        if cell:
            row, col = cell
            idx = self.cells[cell]["idx"]

            # Handle modifier keys
            if event.state & 0x0004:  # Ctrl key - toggle selection
                if idx in self.selected_indices:
                    self.selected_indices.remove(idx)
                else:
                    self.selected_indices.add(idx)
            elif event.state & 0x0001:  # Shift key - range selection
                if self.selected_indices:
                    last_idx = max(self.selected_indices)
                    start = min(last_idx, idx)
                    end = max(last_idx, idx)
                    for i in range(start, end + 1):
                        self.selected_indices.add(i)
                else:
                    self.selected_indices.add(idx)
            else:
                # Single click - toggle selection (not replace)
                if idx in self.selected_indices:
                    self.selected_indices.remove(idx)
                else:
                    self.selected_indices.add(idx)

            # Always track last selection for comments
            if self.selected_indices:
                self.last_selected_indices = self.selected_indices.copy()

            # Only update border if cell is loaded
            if (row, col) in self.cell_ids:
                self._update_cell_border(row, col, idx)

    def _on_mouse_drag(self, event):
        """Handle drag selection (PEP-8)"""
        if not self.drag_start:
            return

        self.current_event = event  # track modifiers during drag
        self.dragging = True
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # Draw selection box
        if self.selection_box:
            self.canvas.delete(self.selection_box)

        # Get color from current mode (classification or tag) - use actual color!
        box_color = self.theme_colors["accent_blue"]  # Default to accent color
        if self.dialog_ref and hasattr(self.dialog_ref, "item_manager"):
            # Check if current mode is a tag
            tag_def = self.dialog_ref.item_manager.get_tag(self.current_mode)
            if tag_def and tag_def.is_active:
                box_color = tag_def.color
            else:
                # Try as classification
                class_def = self.dialog_ref.item_manager.get_classification(self.current_mode)
                if class_def and class_def.is_active:
                    box_color = class_def.color
        
        self.selection_box = self.canvas.create_rectangle(
            self.drag_start[0],
            self.drag_start[1],
            canvas_x,
            canvas_y,
            outline=box_color,
            width=3,  # Make it more visible
            dash=(3, 3),
        )
        # Make sure the box is above tiles/borders
        try:
            self.canvas.tag_raise(self.selection_box)
        except Exception:
            pass
        # Flush draw so it appears immediately
        self.canvas.update_idletasks()

        # Throttle selection updates to reduce lag on huge grids
        now = time.time()
        if (now - self._drag_last_ts) >= (1.0 / 60.0):
            self._drag_last_ts = now
            self._update_drag_selection(
                self.drag_start[0], self.drag_start[1], canvas_x, canvas_y
            )
        else:
            if not self._drag_pending:
                self._drag_pending = True

                def _deferred():
                    self._drag_pending = False
                    # Check if drag is still active (may have been released)
                    if self.drag_start is None:
                        return
                    # Use latest pointer location
                    cx = self.canvas.canvasx(
                        self.canvas.winfo_pointerx() - self.canvas.winfo_rootx()
                    )
                    cy = self.canvas.canvasy(
                        self.canvas.winfo_pointery() - self.canvas.winfo_rooty()
                    )
                    self._update_drag_selection(
                        self.drag_start[0], self.drag_start[1], cx, cy
                    )

                self.canvas.after(
                    16, _deferred
                )  # PEP-8: schedule via a Tk widget we own

        # Check if near edge and start auto-scrolling
        self._check_edge_scroll(event.y)

    def _on_mouse_release(self, event):
        """Handle mouse release - apply classification"""
        # Stop auto-scrolling
        self._stop_auto_scroll()
        
        if self.selection_box:
            self.canvas.delete(self.selection_box)
            self.selection_box = None

        if self.dragging and self.selected_indices:
            # Store selection before applying classification
            self.persistent_selection = list(self.selected_indices)

            # Apply current mode to selected images
            self._apply_classification_to_selected()

            # Don't clear selection immediately - keep for reference
            # Selection will be cleared only when user clicks elsewhere

        self.drag_start = None
        self.dragging = False

    def _check_edge_scroll(self, mouse_y):
        """Check if mouse is near edge and trigger auto-scrolling."""
        canvas_height = self.canvas.winfo_height()

        # Check if near top edge
        if mouse_y < self.scroll_edge_threshold:
            if self.auto_scroll_direction != "up":
                self.auto_scroll_direction = "up"
                self.auto_scroll_speed = 0  # Reset speed when changing direction
            self._start_auto_scroll()
        # Check if near bottom edge
        elif mouse_y > (canvas_height - self.scroll_edge_threshold):
            if self.auto_scroll_direction != "down":
                self.auto_scroll_direction = "down"
                self.auto_scroll_speed = 0  # Reset speed when changing direction
            self._start_auto_scroll()
        else:
            # Not near edge, stop scrolling
            self._stop_auto_scroll()

    def _start_auto_scroll(self):
        """Start or continue auto-scrolling."""
        if self.auto_scroll_timer is None:
            self._perform_auto_scroll()

    def _stop_auto_scroll(self):
        """Stop auto-scrolling."""
        if self.auto_scroll_timer:
            self.canvas.after_cancel(self.auto_scroll_timer)
            self.auto_scroll_timer = None
        self.auto_scroll_speed = 0
        self.auto_scroll_direction = None

    def _perform_auto_scroll(self):
        """Perform one step of auto-scrolling with acceleration."""
        if self.auto_scroll_direction is None:
            return

        # Accelerate: start slow, increase to max
        max_speed = 5  # Maximum scroll units per step
        acceleration = 0.05  # How fast to accelerate

        self.auto_scroll_speed = min(self.auto_scroll_speed + acceleration, max_speed)

        # Calculate scroll amount (minimum 1 unit)
        scroll_amount = max(1, int(self.auto_scroll_speed))

        # Scroll in the appropriate direction
        if self.auto_scroll_direction == "up":
            self.canvas.yview_scroll(-scroll_amount, "units")
        elif self.auto_scroll_direction == "down":
            self.canvas.yview_scroll(scroll_amount, "units")

        # Schedule next scroll with faster interval as speed increases
        # Start at 50ms, decrease to 10ms as speed increases
        interval = max(10, int(50 - (self.auto_scroll_speed * 2)))
        self.auto_scroll_timer = self.canvas.after(interval, self._perform_auto_scroll)

    def _on_mouse_motion(self, event):
        """Handle mouse motion"""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        cell = self._get_cell_at(canvas_x, canvas_y)

        if cell != self.hover_cell:
            self.hover_cell = cell
            # Scatter filter doesn't update per hover - it's for bulk filtering

    def _on_mouse_leave(self, event):
        """Handle mouse leaving canvas"""
        # Don't hide zoom window - let user close it manually
        self.hover_cell = None

    def _on_right_click(self, event):
        """Handle right click"""
        # Right-click reserved for future context menu
        pass

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        if not (event.state & 0x0004):
            old_start, old_end = self.visible_range

            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

            self._update_lazy_load()

            # Auto-skip: when scrolling forward (down), images that leave the
            # viewport above are marked not_<tag> if auto-skip is enabled and
            # the current mode is a tag.
            scrolling_down = event.delta < 0
            if scrolling_down:
                new_start, new_end = self.visible_range
                self._auto_skip_scrolled_past(old_start, new_start)

    def _on_ctrl_mousewheel(self, event):
        """Handle Ctrl+MouseWheel for zoom"""
        # Calculate zoom factor
        if event.delta > 0:
            factor = 1.1
        else:
            factor = 0.9

        # Update scale - no upper limit
        new_scale = self.scale_factor * factor
        new_scale = max(0.3, min(5.0, new_scale))  # Allow up to 5x zoom

        if new_scale != self.scale_factor:
            self.scale_factor = new_scale
            self.cell_width = int(self.base_cell_width * self.scale_factor)
            self.cell_height = int(self.base_cell_height * self.scale_factor)

            self.logger.debug(
                f"Zoom: scale={self.scale_factor:.2f}, cell={self.cell_width}x{self.cell_height}"
            )

            # Reload with new scale
            self.load_images(self.displayed_images, preserve_selection=True)

    def _update_lazy_load(self):
        """Update lazy loading based on current viewport"""
        new_range = self._calculate_visible_range()
        start_idx, end_idx = new_range

        # Load newly visible cells - prioritize immediate loading
        cells_to_load = []
        for idx in range(start_idx, min(end_idx, len(self.displayed_images))):
            if idx not in self.loaded_cells and idx not in self.pending_loads:
                self.pending_loads.add(idx)
                row = idx // self.cols_per_row
                col = idx % self.cols_per_row
                cells_to_load.append((row, col, idx, self.displayed_images[idx]))

        # Load cells in batches for better performance
        batch_size = 10
        for i in range(0, len(cells_to_load), batch_size):
            batch = cells_to_load[i : i + batch_size]
            # Load first batch immediately, others with minimal delay
            if i == 0:
                for row, col, idx, img in batch:
                    self._load_cell_deferred(row, col, idx, img)
            else:
                # Small delay for subsequent batches to prevent UI freezing
                self.canvas.after(
                    i * 10,
                    lambda b=batch: [
                        self._load_cell_deferred(r, c, i, img) for r, c, i, img in b
                    ],
                )

        # Increase unload buffer to prevent thrashing
        for idx in list(self.loaded_cells):
            if idx < start_idx - 100 or idx > end_idx + 100:  # Increased buffer
                self._unload_cell(idx)

        self.visible_range = new_range

    def _load_cell_deferred(self, row, col, idx, img):
        """Deferred cell loading"""
        if idx not in self.loaded_cells:
            self._load_cell_content(row, col, idx, img)
            self.loaded_cells.add(idx)
        self.pending_loads.discard(idx)

    def _unload_cell(self, idx):
        """Unload a cell to save memory but keep placeholder visible"""
        row = idx // self.cols_per_row
        col = idx % self.cols_per_row

        if (row, col) in self.cell_ids:
            # Delete all canvas items for this cell
            for item_id in self.cell_ids[(row, col)].values():
                if item_id:
                    self.canvas.delete(item_id)
            del self.cell_ids[(row, col)]

        # Clear image reference
        if (row, col) in self.image_refs:
            del self.image_refs[(row, col)]

        # ALWAYS recreate placeholder to prevent empty cells
        if (row, col) in self.cells:
            cell_data = self.cells[(row, col)]
            x, y = cell_data["x"], cell_data["y"]

            # Create more visible placeholder
            placeholder_id = self.canvas.create_rectangle(
                x,
                y,
                x + self.cell_width,
                y + self.cell_height,
                fill="#2a2a2a",
                outline="#3a3a3a",
                width=1,
                tags=(f"placeholder_{idx}", f"cell_{idx}"),
            )

            # Add loading text
            text_id = self.canvas.create_text(
                x + self.cell_width // 2,
                y + self.cell_height // 2,
                text="Loading...",
                fill="#666666",
                font=("Arial", 10),
                tags=(f"loading_{idx}", f"cell_{idx}"),
            )

            cell_data["placeholder"] = placeholder_id
            cell_data["loading_text"] = text_id

        self.loaded_cells.discard(idx)

    def _get_cell_at(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """Get cell at canvas position"""
        for (row, col), cell_data in self.cells.items():
            cell_x = cell_data["x"]
            cell_y = cell_data["y"]

            if (
                cell_x <= x <= cell_x + self.cell_width
                and cell_y <= y <= cell_y + self.cell_height
            ):
                return (row, col)

        return None

    def _update_drag_selection(self, x0, y0, x1, y1):
        """Update selection based on drag box (PEP-8)"""
        min_x, max_x = min(x0, x1), max(x0, x1)
        min_y, max_y = min(y0, y1), max(y0, y1)

        # Only clear if not holding Ctrl (for additive selection)
        if not (
            hasattr(self, "current_event")
            and self.current_event
            and self.current_event.state & 0x0004
        ):
            self.selected_indices.clear()

        # Compute indices covered by the rect across the entire grid (off-screen included)
        col_span = self.cell_width + self.padding
        row_span = self.cell_height + self.padding

        # Guard against invalid spans (PEP-8)
        if col_span <= 0 or row_span <= 0 or self.cols_per_row <= 0:
            return

        start_col = max(0, int(min_x // col_span))
        end_col = max(0, int(max_x // col_span))
        start_row = max(0, int(min_y // row_span))
        end_row = max(0, int(max_y // row_span))

        total = len(self.displayed_images)
        max_row = (total + self.cols_per_row - 1) // self.cols_per_row
        end_row = min(end_row, max_row)

        new_selected = set()
        for row in range(start_row, end_row + 1):
            base = row * self.cols_per_row
            if base >= total:
                break
            for col in range(start_col, end_col + 1):
                idx = base + col
                if idx >= total or col < 0 or col >= self.cols_per_row:
                    continue
                # Skip placeholders
                img = self.displayed_images[idx]
                if hasattr(img, "is_placeholder") and img.is_placeholder:
                    continue
                new_selected.add(idx)

        prev_selected = self.selected_indices.copy()
        # Apply selection (Ctrl = additive)
        if (
            hasattr(self, "current_event")
            and self.current_event
            and (self.current_event.state & 0x0004)
        ):
            self.selected_indices.update(new_selected)
        else:
            self.selected_indices = new_selected

        # Store as last selected for comments
        if self.selected_indices:
            self.last_selected_indices = self.selected_indices.copy()

        # Update borders only for affected loaded cells
        affected = prev_selected.union(self.selected_indices)
        for idx in affected:
            row = idx // self.cols_per_row
            col = idx % self.cols_per_row
            if (row, col) in self.cell_ids:
                self._update_cell_border(row, col, idx)

    def _update_cell_border(self, row: int, col: int, idx: int):
        """Update cell border based on selection"""
        if (row, col) not in self.cell_ids:
            return

        border_id = self.cell_ids[(row, col)]["border"]
        img = self.displayed_images[idx]

        # Get base color from classification
        color = self._get_border_color(img)

        # Determine border width based on classification status
        if img.classification != ClassificationCategory.UNASSIGNED:
            # Previously classified - make VERY visible
            base_width = 8  # Much thicker for classified images
        else:
            base_width = 2  # Thin for unclassified

        if idx in self.selected_indices:
            # Selected - add extra thickness
            width = base_width + 2
            # No dash for solid appearance
            self.canvas.itemconfig(border_id, outline=color, width=width, dash=())
        elif idx == self.last_selected_index:
            # Last selected (for comments) - slightly thicker
            width = base_width + 1
            self.canvas.itemconfig(border_id, outline=color, width=width, dash=())
        else:
            # Normal - use base width
            width = base_width
            self.canvas.itemconfig(border_id, outline=color, width=width, dash=())

        # Update statistics whenever borders change
        if hasattr(self, "dialog_ref"):
            self.dialog_ref._update_statistics()

    def refresh_all_cells(self):
        """Refresh all visible cells (e.g., after classification colors change)"""
        if not hasattr(self, "cells") or not self.cells:
            return

        for (row, col), cell_data in self.cells.items():
            idx = cell_data["idx"]
            if idx < len(self.displayed_images):
                # Update the border color
                self._update_cell_border(row, col, idx)

        self.logger.debug("Refreshed all cell borders with new colors")

    def _apply_classification_to_selected(self):
        """Apply current classification/tag mode to selected images with toggle behavior"""
        if not self.selected_indices:
            return
        
        # Block classification if classification labels are hidden (to prevent mistakes)
        if not getattr(self, 'show_classification_labels', True):
            self.logger.warning("Classification blocked: classification labels are hidden")
            if self.dialog_ref:
                from gui.dialog_helper import DialogHelper
                DialogHelper.show_message(
                    self.dialog_ref.dialog,
                    "Classification Disabled",
                    "Classification is disabled while classification labels are hidden.\n\n"
                    "Enable 'Show Classification Labels' in Viz Config to allow classifications.",
                    message_type="warning"
                )
            return
        
        # Check if current mode is a tag or classification
        is_tag = False
        if self.dialog_ref and hasattr(self.dialog_ref, "item_manager"):
            is_tag = self.dialog_ref.item_manager.get_tag(self.current_mode) is not None
        
        if is_tag:
            # Handle tag toggle (independent of classification)
            self._apply_tag_to_selected(self.current_mode)
            return

        self.logger.info(
            f"Applying {self.current_mode} to {len(self.selected_indices)} images"
        )

        # Capture state for undo BEFORE making changes
        undo_action = self._capture_undo_state(self.selected_indices, "classify")

        # Store for comments
        self.last_selected_indices = self.selected_indices.copy()

        # Get classification label from ClassificationManager
        dialog = self.dialog_ref
        class_def = None
        if dialog and hasattr(dialog, "item_manager"):
            class_def = dialog.item_manager.get_classification(self.current_mode)

        if class_def:
            target_classification = class_def.label
        else:
            # Fallback for unknown mode
            target_classification = "Unassigned"

        # Apply with toggle behavior and capture new states
        changed = False
        new_states = []
        updated_images = []  # Track which images were updated

        for idx in self.selected_indices:
            if idx < len(self.displayed_images):
                img = self.displayed_images[idx]

                # Toggle behavior: if already has this classification, clear it
                if (
                    self.current_mode != "clear"
                    and img.classification == target_classification
                ):
                    img.classification = ClassificationCategory.UNASSIGNED
                else:
                    img.classification = target_classification

                # Update metadata - save username in lowercase for consistency
                img.classified_by = os.getenv("USERNAME", "Unknown").lower()
                img.classified_date = datetime.now().isoformat()
                # Capture new state
                new_state = {
                    "classification": img.classification,
                    "comments": img.comments,
                    "classified_by": img.classified_by,
                    "classified_date": img.classified_date,
                    "tags": list(img.tags) if hasattr(img, 'tags') and img.tags else []
                }
                new_states.append(new_state)
                updated_images.append(img)
                changed = True

        # Clear selection and refresh ONLY the changed cells
        if changed:
            undo_action.new_states = new_states

            # Push to undo stack via dialog
            if hasattr(self, "dialog_ref") and self.dialog_ref:
                self.dialog_ref._push_undo_action(undo_action)

                # IMPORTANT: Also update the all_images list with changes
                for updated_img in updated_images:
                    # Find and update corresponding image in all_images
                    for all_img in self.dialog_ref.all_images:
                        if (
                            all_img.hole_id == updated_img.hole_id
                            and all_img.depth_to == updated_img.depth_to
                            and all_img.filename == updated_img.filename
                        ):
                            all_img.classification = updated_img.classification
                            all_img.classified_by = updated_img.classified_by
                            all_img.classified_date = updated_img.classified_date
                            all_img.comments = updated_img.comments
                            # Also copy tags
                            if hasattr(updated_img, 'tags'):
                                all_img.tags = updated_img.tags.copy()
                            break

            # Clear selection and refresh only loaded cells
            self.selected_indices.clear()
            self._refresh_classifications()

            # Update borders for any loaded cells that were selected
            for idx in list(self.last_selected_indices):
                row = idx // self.cols_per_row
                col = idx % self.cols_per_row
                if (row, col) in self.cell_ids:
                    self._update_cell_border(row, col, idx)
            self.mark_changed()  # Mark unsaved changes
            if (
                hasattr(self, "dialog_ref")
                and self.dialog_ref
                and hasattr(self.dialog_ref, "_bump_classification_epoch")
            ):
                self.dialog_ref._bump_classification_epoch()

            # Check for hole completion and auto-advance
            if hasattr(self, "dialog_ref") and self.dialog_ref:
                self.dialog_ref._auto_advance_if_complete()

    def _apply_tag_to_selected(self, tag_id: str):
        """Toggle tag on selected images (independent of classification, like QAQC BAD flag)"""
        if not self.selected_indices:
            return
        
        self.logger.info(f"Toggling tag '{tag_id}' on {len(self.selected_indices)} images")
        
        # Get tag definition
        tag_def = None
        if self.dialog_ref and hasattr(self.dialog_ref, "item_manager"):
            tag_def = self.dialog_ref.item_manager.get_tag(tag_id)
        
        if not tag_def:
            self.logger.warning(f"Tag '{tag_id}' not found")
            return
        
        # Capture state for undo
        undo_action = self._capture_undo_state(self.selected_indices, "tag")
        
        # Store for potential comments
        self.last_selected_indices = self.selected_indices.copy()
        
        changed = False
        new_states = []
        updated_images = []
        
        for idx in self.selected_indices:
            if idx < len(self.displayed_images):
                img = self.displayed_images[idx]
                
                # Initialize tags set if needed
                if not hasattr(img, 'tags') or img.tags is None:
                    img.tags = set()
                
                # Toggle tag (like QAQC toggles bad_image flag)
                if tag_id in img.tags:
                    img.tags.remove(tag_id)
                    self.logger.debug(f"Removed tag '{tag_id}' from image {img.filename}")
                else:
                    img.tags.add(tag_id)
                    self.logger.debug(f"Added tag '{tag_id}' to image {img.filename}")
                
                # Update metadata
                img.classified_by = os.getenv("USERNAME", "Unknown").lower()
                img.classified_date = datetime.now().isoformat()
                
                # Capture new state
                new_state = {
                    "classification": img.classification,
                    "comments": img.comments,
                    "classified_by": img.classified_by,
                    "classified_date": img.classified_date,
                    "tags": list(img.tags) if img.tags else []
                }
                new_states.append(new_state)
                updated_images.append(img)
                changed = True
        
        if changed:
            undo_action.new_states = new_states
            
            # Push to undo stack
            if self.dialog_ref:
                self.dialog_ref._push_undo_action(undo_action)
                
                # Update all_images list (like classifications do)
                for updated_img in updated_images:
                    for all_img in self.dialog_ref.all_images:
                        if (all_img.hole_id == updated_img.hole_id and
                            all_img.depth_to == updated_img.depth_to and
                            all_img.filename == updated_img.filename):
                            all_img.tags = updated_img.tags.copy()
                            all_img.classified_by = updated_img.classified_by
                            all_img.classified_date = updated_img.classified_date
                            break
            
            # Clear selection and refresh
            self.selected_indices.clear()
            self._refresh_classifications()
            
            if self.dialog_ref and hasattr(self.dialog_ref, 'mark_changed'):
                self.dialog_ref.mark_changed()
        
        self.logger.info(f"Tag toggle complete - {len(updated_images)} images modified")

    def _auto_skip_scrolled_past(self, old_start: int, new_start: int):
        """Mark images that scrolled out of view as not_<tag> when auto-skip is on.
        
        Only fires when:
        - The dialog has auto_skip_tag_var enabled
        - The current mode is a tag (not a classification)
        - Images actually scrolled upward out of the viewport
        """
        if new_start <= old_start:
            return
        if not self.dialog_ref:
            return
        if not getattr(self.dialog_ref, "auto_skip_tag_var", None):
            return
        if not self.dialog_ref.auto_skip_tag_var.get():
            return

        # Only auto-skip when the active mode is a tag
        tag_def = None
        if hasattr(self.dialog_ref, "item_manager"):
            tag_def = self.dialog_ref.item_manager.get_tag(self.current_mode)
        if not tag_def:
            return

        neg_tag = f"not_{self.current_mode}"
        count = 0

        for idx in range(old_start, min(new_start, len(self.displayed_images))):
            img = self.displayed_images[idx]
            if not hasattr(img, "tags") or img.tags is None:
                img.tags = set()

            # Skip if already positively tagged or already skipped
            if self.current_mode in img.tags or neg_tag in img.tags:
                continue

            img.tags.add(neg_tag)

            # Propagate to all_images
            for all_img in self.dialog_ref.all_images:
                if (all_img.hole_id == img.hole_id
                        and all_img.depth_to == img.depth_to
                        and all_img.filename == img.filename):
                    if not hasattr(all_img, "tags") or all_img.tags is None:
                        all_img.tags = set()
                    all_img.tags.add(neg_tag)
                    break
            count += 1

        if count > 0:
            self.dialog_ref.has_unsaved_changes = True
            self.logger.debug(
                f"Auto-skip: marked {count} images as not_{self.current_mode}"
            )

    def _capture_undo_state(self, indices: Set[int], action_type: str) -> UndoAction:
        """Capture current state for undo functionality"""
        affected_images = []
        old_states = []

        for idx in indices:
            if idx < len(self.displayed_images):
                img = self.displayed_images[idx]
                affected_images.append((idx, img.filename))

                # Capture current state
                old_state = {
                    "classification": img.classification,
                    "comments": img.comments,
                    "classified_by": img.classified_by,
                    "classified_date": img.classified_date,
                    "tags": list(img.tags) if hasattr(img, 'tags') and img.tags else []
                }
                old_states.append(old_state)

        return UndoAction(
            action_type=action_type,
            affected_images=affected_images,
            old_states=old_states,
            new_states=[],  # Will be filled after action
        )

    def mark_changed(self):
        """Mark that changes have been made"""
        if hasattr(self.parent.master, "has_unsaved_changes"):
            self.parent.master.has_unsaved_changes = True

    def _refresh_classifications(self):
        """Refresh only the classification displays without reloading images"""
        for (row, col), cell_data in self.cells.items():
            idx = cell_data["idx"]

            # Skip cells that haven't been loaded yet (lazy loading)
            if (row, col) not in self.cell_ids:
                continue

            img = self.displayed_images[idx]

            # Update border color and width
            if "border" not in self.cell_ids[(row, col)]:
                continue
            border_id = self.cell_ids[(row, col)]["border"]

            # Check if cell outlines should be shown
            if not getattr(self, 'show_cell_outlines', True):
                # Hide border by setting width to 0
                self.canvas.itemconfig(border_id, outline="", width=0)
            else:
                # Get classification color
                border_color = self._get_border_color(img)

                # Get base outline width from settings
                base_outline_width = getattr(self, 'cell_outline_width', 2)

                # Much thicker borders for classified images
                if img.classification != ClassificationCategory.UNASSIGNED:
                    # Previously classified - add extra width
                    if idx in self.selected_indices:
                        border_width = base_outline_width + 8  # Extra thick when selected
                    else:
                        border_width = base_outline_width + 6  # Thick for classified
                else:
                    # Unclassified
                    if idx in self.selected_indices:
                        border_width = base_outline_width + 2  # Slightly thick when selected
                    else:
                        border_width = base_outline_width  # Use configured width
                    border_color = (
                        "#CCCCCC"
                        if border_color == self.theme_colors.get("border", "#CCCCCC")
                        else border_color
                    )

                self.canvas.itemconfig(border_id, outline=border_color, width=border_width)

            # Update classification labels in top-right with background
            # Clean up old single-label approach
            if (
                "class_label" in self.cell_ids[(row, col)]
                and self.cell_ids[(row, col)]["class_label"]
            ):
                self.canvas.delete(self.cell_ids[(row, col)]["class_label"])
                self.cell_ids[(row, col)]["class_label"] = None

            if (
                "class_bg" in self.cell_ids[(row, col)]
                and self.cell_ids[(row, col)]["class_bg"]
            ):
                self.canvas.delete(self.cell_ids[(row, col)]["class_bg"])
                self.cell_ids[(row, col)]["class_bg"] = None

            # Clean up new multi-label approach
            if "class_labels" in self.cell_ids[(row, col)]:
                for label_id in self.cell_ids[(row, col)]["class_labels"]:
                    self.canvas.delete(label_id)
                self.cell_ids[(row, col)]["class_labels"] = []
            
            # ALWAYS ensure class_labels exists - initialize if needed
            if "class_labels" not in self.cell_ids[(row, col)]:
                self.cell_ids[(row, col)]["class_labels"] = []

            # Build list of all classifications to display (current user + others)
            classifications_to_display = []

            if img.classification != ClassificationCategory.UNASSIGNED:
                classification_text = self.get_classification_text(img.classification)
                
                # Append tags to classification text (like QAQC BAD flag)
                if hasattr(img, 'tags') and img.tags:
                    tag_labels = []
                    if self.dialog_ref and hasattr(self.dialog_ref, "item_manager"):
                        for tag_id in sorted(img.tags):
                            tag_def = self.dialog_ref.item_manager.get_tag(tag_id)
                            if tag_def:
                                # Use icon + label or just label
                                tag_text = f"{tag_def.icon} {tag_def.label}" if tag_def.icon else tag_def.label
                                tag_labels.append(tag_text)
                    
                    if tag_labels:
                        classification_text = f"{classification_text} {' '.join(tag_labels)}"
                
                reviewer_name = img.classified_by if img.classified_by else "Unknown"
                classifications_to_display.append(
                    {
                        "text": f"{classification_text} - by {reviewer_name}",
                        "classification": img.classification,
                        "has_comments": bool(img.comments),
                    }
                )

            # Add other reviewers' classifications
            if hasattr(img, "other_reviews") and img.other_reviews:
                for review in img.other_reviews:
                    classification_value = None
                    for field_name in [
                        "classification",
                        "Classification",
                        "Lithology",
                        "Rock_Type",
                    ]:
                        if field_name in review and review[field_name]:
                            classification_value = review[field_name]
                            break

                    if classification_value:
                        reviewer_name = review.get("Reviewed_By", "Unknown")
                        has_comments = bool(review.get("Comments", ""))
                        classifications_to_display.append(
                            {
                                "text": f"{classification_value} - by {reviewer_name}",
                                "classification": classification_value,
                                "has_comments": has_comments,
                            }
                        )


            # Also show tags for unclassified images (like QAQC shows BAD on unclassified)
            if img.classification == ClassificationCategory.UNASSIGNED:
                if hasattr(img, 'tags') and img.tags:
                    tag_labels = []
                    if self.dialog_ref and hasattr(self.dialog_ref, "item_manager"):
                        for tag_id in sorted(img.tags):
                            tag_def = self.dialog_ref.item_manager.get_tag(tag_id)
                            if tag_def:
                                tag_text = f"{tag_def.icon} {tag_def.label}" if tag_def.icon else tag_def.label
                                tag_labels.append(tag_text)
                    
                    if tag_labels:
                        tags_text = ' '.join(tag_labels)
                        reviewer_name = img.classified_by if img.classified_by else "Unknown"
                        classifications_to_display.append(
                            {
                                "text": f"{tags_text} - by {reviewer_name}",
                                "classification": None,  # No classification, just tags
                                "has_comments": bool(img.comments),
                            }
                        )

            # Add comment indicator in top-left corner if any comments exist
            has_any_comments = False
            if img.comments:
                has_any_comments = True
            if hasattr(img, "other_reviews") and img.other_reviews:
                for review in img.other_reviews:
                    if review.get("Comments", ""):
                        has_any_comments = True
                        break

            if has_any_comments:
                comment_x = cell_data["x"] + 5
                comment_y = cell_data["y"] + 5

                # Create comment indicator background
                comment_bg = self.canvas.create_rectangle(
                    comment_x,
                    comment_y,
                    comment_x + 20,
                    comment_y + 16,
                    fill="#FFA500",  # Orange background
                    outline="",
                    tags=(f"comment_{row}_{col}", "comment_indicator"),
                )
                self.cell_ids[(row, col)]["class_labels"].append(comment_bg)

                # Create comment icon
                comment_icon = self.canvas.create_text(
                    comment_x + 10,
                    comment_y + 8,
                    text="💬",
                    fill="white",
                    font=("Arial", 10),
                    anchor="center",
                    tags=(f"comment_{row}_{col}", "comment_indicator"),
                )
                self.cell_ids[(row, col)]["class_labels"].append(comment_icon)
                
                # Store comment data for tooltip
                comment_data = {
                    'bg_id': comment_bg,
                    'icon_id': comment_icon,
                    'row': row,
                    'col': col,
                    'idx': idx,
                    'comments': img.comments if img.comments else "",
                    'other_comments': []
                }
                
                # Collect other reviewers' comments
                if hasattr(img, "other_reviews") and img.other_reviews:
                    for review in img.other_reviews:
                        comment_text = review.get("Comments", "")
                        if comment_text:
                            reviewer = review.get("Reviewed By", "Unknown")
                            comment_data['other_comments'].append({
                                'reviewer': reviewer,
                                'text': comment_text
                            })
                
                # Store in cell data for tooltip lookup
                if 'comment_data' not in self.cell_ids[(row, col)]:
                    self.cell_ids[(row, col)]['comment_data'] = comment_data

# Display all classifications stacked vertically in TOP LEFT
            # Only show if classification labels are enabled
            if classifications_to_display and getattr(self, 'show_classification_labels', True):
                x_start = cell_data["x"] + 5  # Left edge with padding
                y_start = cell_data["y"] + 30  # Below comment indicator
                line_height = 16  # Height for each classification line

                # Store all created label IDs for cleanup
                if "class_labels" not in self.cell_ids[(row, col)]:
                    self.cell_ids[(row, col)]["class_labels"] = []

                # Check if showing others' reviews and add consensus label
                show_others = (
                    self.dialog_ref.show_others_var.get() 
                    if self.dialog_ref and hasattr(self.dialog_ref, "show_others_var") 
                    else False
                )
                
                if show_others and len(classifications_to_display) > 0:
                    # Calculate consensus
                    all_class_strs = []
                    for class_info in classifications_to_display:
                        if class_info["classification"]:
                            class_str = (
                                class_info["classification"].value
                                if hasattr(class_info["classification"], "value")
                                else str(class_info["classification"])
                            )
                            all_class_strs.append(class_str)
                    
                    if all_class_strs:
                        from collections import Counter
                        consensus = Counter(all_class_strs).most_common(1)[0][0]
                        is_conflict = len(set(all_class_strs)) > 1
                        
                        # Add consensus label at top
                        consensus_text = f"{'Conflict' if is_conflict else consensus} - Consensus"
                        consensus_color = "#FF6B6B" if is_conflict else "#4CAF50"
                        
                        text_width = len(consensus_text) * 5 + 8
                        
                        # Create consensus background
                        consensus_bg = self.canvas.create_rectangle(
                            x_start,
                            y_start,
                            x_start + text_width,
                            y_start + 14,
                            fill=consensus_color,
                            outline="",
                            tags=f"class_{row}_{col}",
                        )
                        self.cell_ids[(row, col)]["class_labels"].append(consensus_bg)
                        
                        # Create consensus text
                        consensus_label = self.canvas.create_text(
                            x_start + text_width // 2,
                            y_start + 7,
                            text=consensus_text,
                            fill="white",
                            font=("Arial", 7, "bold"),
                            anchor="center",
                            tags=f"class_{row}_{col}",
                        )
                        self.cell_ids[(row, col)]["class_labels"].append(consensus_label)
                        
                        # Move y_start down for individual labels
                        y_start += line_height

                for i, class_info in enumerate(classifications_to_display):
                    y = y_start + (i * line_height)

                    # Determine color using ClassificationManager
                    if self.dialog_ref and hasattr(self.dialog_ref, "item_manager"):
                        # Convert classification to string for lookup
                        class_str = (
                            class_info["classification"].value
                            if hasattr(class_info["classification"], "value")
                            else str(class_info["classification"])
                        )
                        bg_color = (
                            self.dialog_ref.item_manager.get_color_for_classification(
                                class_str
                            )
                        )
                    else:
                        # Fallback - try to get color from classification enum
                        from gui.ReviewDialog.image_classification_and_tag_manager import ImageClassificationAndTagManager
                        
                        class_colors = {
                            ClassificationCategory.BIFF: "#4CAF50",
                            ClassificationCategory.BIFHM: "#f44336",
                            ClassificationCategory.OTHER: "#2196F3",
                            ClassificationCategory.NOT_CONFIDENT: "#FF9800",
                            "BIFf": "#4CAF50",
                            "BIFf-s": "#9ACD32",
                            "Compact": "#FF9800",
                            "BIF HM": "#f44336",
                            "BIFhm": "#f44336",
                            "Other": "#2196F3",
                            "Not Confident": "#FF9800",
                        }
                        bg_color = class_colors.get(
                            class_info["classification"], self.theme_colors["accent_blue"]
                        )

                    # Don't add comment indicator to text - it's shown separately in top-left
                    display_text = class_info["text"]

                    text_width = len(display_text) * 5 + 8

                    # Create background rectangle (LEFT aligned)
                    class_bg = self.canvas.create_rectangle(
                        x_start,
                        y,
                        x_start + text_width,
                        y + 14,
                        fill=bg_color,
                        outline="",
                        tags=f"class_{row}_{col}",
                    )
                    self.cell_ids[(row, col)]["class_labels"].append(class_bg)

                    # Create text label (LEFT aligned)
                    class_label = self.canvas.create_text(
                        x_start + text_width // 2,
                        y + 7,
                        text=display_text,
                        fill="white",
                        font=("Arial", 7, "bold"),
                        anchor="center",
                        tags=f"class_{row}_{col}",
                    )
                    self.cell_ids[(row, col)]["class_labels"].append(class_label)

    def set_mode(self, mode: str):
        """Set classification mode"""
        self.current_mode = mode
        self.logger.debug(f"Classification mode set to: {mode}")

        # Update cursor
        if mode == "clear":
            self.canvas.configure(cursor="X_cursor")
        else:
            self.canvas.configure(cursor="hand2")

    def add_comment_to_last_selected(self, comment: str):
        """Add comment to last selected images - appends to existing comments"""
        if not self.last_selected_indices:
            return

        self.logger.info(f"Adding comment to {len(self.last_selected_indices)} images")

        # Capture state for undo
        undo_action = self._capture_undo_state(self.last_selected_indices, "comment")
        new_states = []

        for idx in self.last_selected_indices:
            if idx < len(self.displayed_images):
                img = self.displayed_images[idx]

                # Append comment with timestamp if there's existing text
                if img.comments and comment:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                    img.comments = f"{img.comments}\n[{timestamp}] {comment}"
                else:
                    img.comments = comment

                # Capture new state
                new_state = {
                    "classification": img.classification,
                    "comments": img.comments,
                    "classified_by": img.classified_by,
                    "classified_date": img.classified_date,
                    "tags": list(img.tags) if hasattr(img, 'tags') and img.tags else []
                }
                new_states.append(new_state)

        undo_action.new_states = new_states

        # Push to undo stack
        if hasattr(self, "dialog_ref") and self.dialog_ref:
            self.dialog_ref._push_undo_action(undo_action)

    def rotate_images(self, angle: int = 90):
        """Rotate all images"""
        self.rotation = (self.rotation + angle) % 360
        self.logger.info(f"Rotating images to {self.rotation} degrees")

        self.load_images(self.displayed_images, preserve_selection=True)

    def toggle_data_visualizations(self):
        """Toggle display of data visualizations"""
        self.show_data_visualizations = not self.show_data_visualizations
        self.load_images(self.displayed_images, preserve_selection=True)

    def set_visualization_columns(self, columns, labels=None):
        """Set which columns to visualize with optional custom labels"""
        self.viz_columns = columns  # No limit on visualization columns
        if labels:
            self.viz_column_labels = labels
        self.load_images(self.displayed_images, preserve_selection=True)

    def get_statistics(self) -> Dict[str, int]:
        """Get classification statistics - dynamic based on ItemManager"""
        stats = {"total": len(self.displayed_images)}

        # Get all classification IDs from manager
        dialog = self.dialog_ref
        if dialog and hasattr(dialog, "item_manager"):
            for class_def in dialog.item_manager.get_all_classifications():
                stats[class_def.id] = 0

        # Always include unassigned
        stats["unassigned"] = 0

        for img in self.displayed_images:
            # Handle both string and enum classifications
            class_str = self.get_classification_text(img.classification)

            # Normalize for comparison
            class_str_normalized = class_str.strip().upper() if class_str else ""

            if class_str_normalized in ["", "UNASSIGNED"]:
                stats["unassigned"] += 1
            else:
                # Try to match to a known classification
                matched = False
                if dialog and hasattr(dialog, "item_manager"):
                    for class_def in dialog.item_manager.get_all_classifications():
                        # Match by label or ID
                        if (
                            class_str_normalized == class_def.label.upper()
                            or class_str_normalized == class_def.id.upper()
                        ):
                            stats[class_def.id] += 1
                            matched = True
                            break

                if not matched:
                    # Unknown classification - add to unassigned for now
                    stats["unassigned"] += 1

        return stats
    
    def cleanup(self):
        """Clean up canvas resources including tooltip bindings"""
        # Unbind tooltip events
        try:
            self.canvas.unbind("<Motion>")
            self.canvas.unbind("<Leave>")
            self.logger.debug("Unbound tooltip events from canvas")
        except Exception as e:
            self.logger.warning(f"Error unbinding tooltip events: {e}")
        
        # Hide any active tooltip
        self._hide_tooltip()
        
        self.logger.info("LithologyGridCanvas cleanup complete")


# ============================================================================
# PROGRESS DIALOG MANAGER
# ============================================================================


class ProgressDialogManager:
    """Manages progress dialogs to prevent overlapping"""

    _instance = None
    _current_dialog = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def show_progress(cls, parent, title, message):
        """Show a progress dialog, closing any existing one first"""
        instance = cls.get_instance()

        # Close existing dialog if present
        if instance._current_dialog:
            try:
                instance._current_dialog.close()
            except:
                pass

        # Create new dialog
        from gui.progress_dialog import ProgressDialog

        instance._current_dialog = ProgressDialog(parent, title, message)
        return instance._current_dialog

    @classmethod
    def close_current(cls):
        """Close the current progress dialog if any"""
        instance = cls.get_instance()
        if instance._current_dialog:
            try:
                instance._current_dialog.close()
            except:
                pass
            instance._current_dialog = None


# ============================================================================
# MAIN DIALOG
# ============================================================================


class LoggingReviewDialog:
    """Advanced lithology classification dialog with full feature set"""

    AUTOSAVE_INTERVAL = 30000  # 30 seconds

    def __init__(
        self, parent, file_manager, gui_manager, config_manager, json_manager=None
    ):
        self.parent = parent
        self.file_manager = file_manager
        self.gui_manager = gui_manager
        self.config_manager = config_manager  # Store the ConfigManager instance
        self.config = (
            config_manager.as_dict()
        )  # Also keep the dict for backwards compatibility
        self.json_manager = json_manager

        # Initialize unified manager for both classifications and tags
        self.item_manager = ImageClassificationAndTagManager(
            config_manager=self.config_manager
        )

        # Initialize logger
        self.logger = logging.getLogger(__name__)
        # State tracking for scatter selections

        # FilterPipeline will be initialized after data_coordinator is set (see below)
        self.filter_pipeline = None

        self.pre_scatter_displayed_images = None  # Store state before scatter selection
        self._scatter_selection_info = None  # For active filters display

        # Compartment pattern (configurable)
        pattern_str = self.config.get(
            "compartment_pattern",
            r"([A-Z]{2}\d{4})_CC_(\d+)(?:_(Wet|Dry))?(?:_.*)?\.(?:png|tiff|jpg)$",
        )
        self.compartment_pattern = re.compile(pattern_str, re.IGNORECASE)


        # Use app-level DataCoordinator if available (already loaded with data)
        self.data_coordinator = None
        self.drillhole_data_manager = None  # Keep for backward compatibility, but deprecated
        
        if hasattr(parent, 'app') and hasattr(parent.app, 'data_coordinator'):
            self.data_coordinator = parent.app.data_coordinator
            self.logger.info("Using app-level DataCoordinator (already loaded with data)")
        elif hasattr(parent, 'app') and hasattr(parent.app, 'drillhole_data_manager'):
            # OLD: Backward compatibility for old DrillholeDataManager
            self.drillhole_data_manager = parent.app.drillhole_data_manager
            self.logger.warning("Using deprecated DrillholeDataManager - should migrate to DataCoordinator")
            
            # Log status to confirm data is available
            if hasattr(self.drillhole_data_manager, 'log_status'):
                self.drillhole_data_manager.log_status()
        else:
            # Fallback: create new instance (shouldn't happen in normal operation)
            self.logger.error("Neither DataCoordinator nor DrillholeDataManager found!")
        
        self.data_visualizer = DrillholeDataVisualizer()
        self.color_map_manager = ColorMapManager(self.config_manager)
        
        # Initialize filter pipeline (must be after data_coordinator and hex_color_cache are set)
        self.filter_pipeline = FilterPipeline(
            data_coordinator=self.data_coordinator,
            hex_color_cache=getattr(self, 'hex_color_cache', {}),
            compute_review_metadata=self._compute_review_metadata
        )
        self.undo_stack = deque(maxlen=50)
        self.redo_stack = deque(maxlen=50)

        # State
        self.all_images = []
        self.displayed_images = []
        self.filter_rows = []
        self.exclude_classified = False
        self.hide_classified = False  # Add this line
        self._classification_epoch = 0  # PEP-8: bump whenever images are (re)classified

        # Hole-by-hole mode
        self.hole_by_hole_mode = False  # Default to all images mode
        self.current_hole_index = 0
        self.unique_holes = []
        self.hole_nav_frame = None
        self.hole_label = None
        self.auto_advance_enabled = True  # Auto-advance when hole complete

        # Performance optimization: image index and CSV cache
        self.image_index = {}  # {(hole_id, depth_to): CompartmentImage}
        self.csv_data_cache = {}  # {(hole_id, depth_to): csv_data_dict}
        self.csv_match_cache = {}  # {(hole_id, depth_to): bool}
        self.hex_color_cache = {}  # {(hole_id, depth_from, depth_to): combined_hex} - loaded once at startup
        self.last_filter_hash = None  # Track if filters changed

        # Add change tracking
        self.has_unsaved_changes = False
        self.last_save_state = {}  # Track state at last save
        self.auto_skip_tag_var = None  # Initialized when tag UI is built

        # UI references
        self.dialog = None
        self.grid_canvas = None
        self.stats_label = None
        self.status_label = None
        self.comment_entry = None

        # Track which holes we've logged for performance
        self._logged_holes = set()

        self.logger.info("LoggingReviewDialog initialized")

    def _has_csv_data_available(self) -> bool:
        """Check if CSV data is available from any data manager."""
        if self.data_coordinator and self.data_coordinator.is_initialized:
            return self.data_coordinator.geological_store.is_loaded
        elif self.drillhole_data_manager:
            return bool(self.drillhole_data_manager.data_sources)
        return False
    
    def _get_csv_value(self, hole_id: str, depth_to: float, column: str) -> Any:
        """Get a single CSV value using the best available data manager."""
        if self.data_coordinator and self.data_coordinator.geological_store.is_loaded:
            try:
                key = ImageKey(hole_id=hole_id, depth_to=float(depth_to))
                self.logger.debug(f"Looking up CSV value for {key} column '{column}' via DataCoordinator")
                return self.data_coordinator.geological_store.get_value(key, column)
            except Exception as e:
                self.logger.debug(f"DataCoordinator lookup failed: {e}")
                return None
        elif self.drillhole_data_manager and self.drillhole_data_manager.data_sources:
            try:
                interval_data = self.drillhole_data_manager.get_data_for_interval(hole_id, depth_to)
                return interval_data.get(column.lower())
            except Exception as e:
                self.logger.debug(f"DrillholeDataManager lookup failed: {e}")
                return None
        return None

    def _populate_csv_column_for_filter(self, column_name: str) -> List[Any]:
        """
        Populate CSV values for a specific column on-demand.
        Used when a filter is applied to a CSV column.
        
        Returns list of values aligned with self.all_images order.
        """
        if not self.data_coordinator or not self.data_coordinator.geological_store.is_loaded:
            return [np.nan] * len(self.all_images)
        
        # Strip source suffix if present
        col_base = column_name.split(" (")[0].strip().lower()
        
        self.logger.debug(f"Fetching CSV values for column: {column_name}")
        start_time = time.time()
        
        values = []
        hits = 0
        
        for img in self.all_images:
            try:
                key = ImageKey(hole_id=img.hole_id, depth_to=float(img.depth_to))
                val = self.data_coordinator.geological_store.get_value(key, col_base)
                if val is not None:
                    values.append(val)
                    hits += 1
                else:
                    values.append(np.nan)
            except Exception:
                values.append(np.nan)
        
        elapsed = time.time() - start_time
        self.logger.debug(f"Fetched {hits}/{len(self.all_images)} values for '{column_name}' in {elapsed:.2f}s")
        
        return values

    def prepare_scatter_data(self, value_columns: List[str]) -> pd.DataFrame:
        """
        Prepare DataFrame for scatter plot from all_images with CSV data.
        
        Args:
            value_columns: List of CSV column names to include (e.g., ['Fe_pct', 'SiO2_pct'])
            
        Returns:
            DataFrame with hole_id, depth_to, moisture_status, and requested value columns
        """
        data_rows = []
        
        for img in self.all_images:
            # Only include images that have CSV data
            if not img.csv_data:
                continue
                
            row = {
                'hole_id': img.hole_id,
                'depth_to': img.depth_to,
                'depth_from': img.depth_from,
                'moisture_status': img.moisture_status or 'Unknown',
                'filename': img.filename,
                'image_path': img.image_path,
                'classification': self._get_classification_string(img.classification) if img.classification else '',
            }
            
            # Add requested value columns from CSV data
            for col in value_columns:
                row[col] = img.csv_data.get(col, np.nan)
            
            data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        
        self.logger.info(f"Prepared scatter data: {len(df)} rows with columns {df.columns.tolist()}")
        return df

    def apply_scatter_selection(self, selected_df: pd.DataFrame):
        """
        Filter image grid to show only images matching scatter selection.
        Handles multiple images per HoleID+depth (Wet/Dry) automatically.
        
        Args:
            selected_df: DataFrame with hole_id and depth_to columns
        """
        if selected_df.empty:
            self.logger.info("Scatter selection is empty - showing all images")
            self._apply_filters()  # Revert to normal filtering
            return
        
        # Create set of (hole_id, depth_to) tuples for fast lookup
        selected_keys = set(
            zip(selected_df['hole_id'], selected_df['depth_to'])
        )
        
        self.logger.info(f"Applying scatter selection: {len(selected_keys)} unique depth intervals")
        
        # Filter displayed_images to matching keys
        filtered_images = [
            img for img in self.all_images
            if (img.hole_id, img.depth_to) in selected_keys
        ]
        
        # Update displayed images
        self.displayed_images = filtered_images
        
        # Reload grid
        if self.grid_canvas:
            self.grid_canvas.load_images(self.displayed_images)
        
        # Update statistics
        self._update_statistics()
        
        wet_count = sum(1 for img in filtered_images if img.moisture_status == 'Wet')
        dry_count = sum(1 for img in filtered_images if img.moisture_status == 'Dry')
        
        self._update_status(
            f"Scatter selection: {len(filtered_images)} images "
            f"({wet_count} Wet, {dry_count} Dry) from {len(selected_keys)} depth intervals"
        )

    def show(self):
        """Show the dialog with non-blocking startup (PEP-8)."""
        self.logger.info("Opening logging review dialog")

        # 1) Create the dialog FIRST so the user sees it immediately
        self._create_dialog()

        # 2) Kick off image scanning on a background thread, keep UI responsive
        progress = ProgressDialogManager.show_progress(
            self.dialog, "Loading", "Scanning compartment images."
        )

        # 2.5) Validate data manager has data before scanning
        if self.data_coordinator:
            if not self.data_coordinator.is_initialized:
                self.logger.warning("DataCoordinator not initialized!")
            elif not self.data_coordinator.geological_store.is_loaded:
                self.logger.warning("GeologicalStore has no CSV data loaded (filters will use base columns only)")
            else:
                source_count = len(self.data_coordinator.geological_store.get_data_sources())
                row_count = self.data_coordinator.geological_store.total_rows
                self.logger.info(f"DataCoordinator ready: {row_count:,} rows from {source_count} CSV sources")
                # NOTE: RegisterStore cache will be built AFTER _scan_images() loads the actual image keys
        elif self.drillhole_data_manager:
            # DEPRECATED: Fallback to old DrillholeDataManager
            if not hasattr(self.drillhole_data_manager, 'data_sources') or not self.drillhole_data_manager.data_sources:
                self.logger.warning("DrillholeDataManager has no data sources loaded!")
            else:
                self.logger.info(f"[DEPRECATED] Using DrillholeDataManager: {len(self.drillhole_data_manager.data_sources)} sources")
        else:
            self.logger.error("No data manager available - cannot load geological data!")
            DialogHelper.show_message(
                self.dialog,
                "No Data Loaded",
                "No geological data has been loaded into the Data Manager.\n\n"
                "Please load data sources before opening the Logging Review Dialog.",
                message_type="warning"
            )
            self.dialog.destroy()
            return
        
        # 3) Scan images after a delay (allows dialog to render first)
        def _on_scan_complete():
            # Called on Tk main thread via after_idle
            try:
                if not self.all_images:
                    DialogHelper.show_message(
                        self.parent,
                        "No Images",
                        "No compartment images found in approved folder.",
                        message_type="info",
                    )
                    return

                self.logger.info(f"Found {len(self.all_images)} total images")

                # Load hex color cache for fast color-based sorting
                self._load_hex_color_cache()

                # No need to load CSV - DrillholeDataManager already has it loaded!
                # Just update filter columns with available data
                self._update_filter_columns()

                # Update UI
                self._update_statistics()
                self._update_status("Ready. Apply filters to load images.")
            finally:
                try:
                    ProgressDialogManager.close_current()
                except Exception:
                    pass

        def _scan_task():
            try:
                self._scan_images()
            except Exception as e:
                self.logger.error(f"Error scanning images: {e}", exc_info=True)
            finally:
                # Switch back to main thread to update UI
                self.dialog.after_idle(_on_scan_complete)

        threading.Thread(target=_scan_task, daemon=True).start()

        # 3) Keep processing events while the dialog is open
        self.dialog.wait_window()

    def _debug_json_files(self):
        """Debug which JSON files are being used"""
        if not self.json_manager:
            self.logger.warning("No JSON manager available")
            return

        self.logger.info("=== JSON FILE DEBUG ===")

        # Check for compartment reviews file
        if hasattr(self.json_manager, "compartment_reviews_file"):
            review_file = self.json_manager.compartment_reviews_file
            self.logger.info(f"Compartment reviews file path: {review_file}")

            if review_file and review_file.exists():
                self.logger.info(f"File exists: {review_file.name}")
                self.logger.info(f"File size: {review_file.stat().st_size} bytes")

                # Check if it matches current user
                current_user = os.getenv("USERNAME", "Unknown").lower()
                if current_user in str(review_file).lower():
                    self.logger.info(
                        f"✓ File appears to be for current user: {current_user}"
                    )
                else:
                    self.logger.warning(
                        f"⚠ File may not be for current user: {current_user}"
                    )
            else:
                self.logger.warning("Compartment reviews file does not exist")

        # List all JSON files in the register directory
        if hasattr(self.json_manager, "register_dir"):
            register_dir = self.json_manager.register_dir
            if register_dir and register_dir.exists():
                json_files = list(register_dir.glob("compartment_reviews*.json"))
                self.logger.info(f"Found {len(json_files)} compartment review files:")
                for f in json_files:
                    self.logger.info(f"  - {f.name} ({f.stat().st_size} bytes)")

    def _scan_images(self):
        """Scan for all compartment images"""
        self.all_images = []
        hole_depths = {}  # Track depths per hole for interval calculation

        try:
            approved_path = self.file_manager.get_shared_path("approved_compartments")
            if not approved_path or not approved_path.exists():
                self.logger.warning("Approved compartments path not found")
                return

            self.logger.info(f"Scanning: {approved_path}")

            # First pass: collect all depths per hole
            temp_images = []
            file_count = 0
            log_interval = 5000  # Log every 5000 files

            for root, dirs, files in os.walk(approved_path):
                for file in files:
                    file_count += 1
                    if file_count % log_interval == 0:
                        self.logger.debug(
                            f"Scanning progress: {file_count} files processed..."
                        )

                    match = self.compartment_pattern.match(file)
                    if match:
                        hole_id = match.group(1)
                        # Handle zero-padded depth values (e.g., "001" -> 1)
                        depth_str = match.group(2)
                        depth_to = float(depth_str.lstrip("0") or "0")
                        moisture = match.group(3)  # Wet/Dry if present

                        # Store depth for this hole
                        if hole_id not in hole_depths:
                            hole_depths[hole_id] = []
                        hole_depths[hole_id].append(depth_to)

                        temp_images.append(
                            {
                                "filename": file,
                                "hole_id": hole_id,
                                "depth_to": depth_to,
                                "moisture": moisture,
                                "path": os.path.join(root, file),
                            }
                        )

            # Calculate intervals for each hole
            hole_intervals = {}
            for hole_id, depths in hole_depths.items():
                if len(depths) > 1:
                    # Sort depths and calculate most common interval
                    sorted_depths = sorted(depths)
                    intervals = []
                    for i in range(1, len(sorted_depths)):
                        interval = sorted_depths[i] - sorted_depths[i - 1]
                        if interval > 0:  # Only positive intervals
                            intervals.append(interval)

                    # Use most common interval, or 1 if no valid intervals
                    if intervals:
                        from collections import Counter

                        interval_counts = Counter(intervals)
                        hole_intervals[hole_id] = interval_counts.most_common(1)[0][0]
                    else:
                        hole_intervals[hole_id] = 1
                else:
                    # Single depth, default to 1m interval
                    hole_intervals[hole_id] = 1

            # Debug JSON files before loading
            self._debug_json_files()

            # Batch load reviews - separate current user from others
            current_user = os.getenv("USERNAME", "Unknown")
            user_reviews = {}  # Current user's reviews
            other_reviews = {}  # Other users' reviews

            if self.json_manager:
                try:
                    self.logger.info(f"=== REVIEW LOADING DEBUG ===")
                    self.logger.info(f"Current user (from environment): {current_user}")

                    # Load ALL review files (current user + other users)
                    all_review_files = self.json_manager.get_all_user_files("review")
                    self.logger.info(
                        f"Found {len(all_review_files)} review files to load"
                    )

                    # Process each review file
                    for review_file in all_review_files:
                        try:
                            # Extract username from filename (e.g., "compartment_reviews_gsymonds_AAD7J01CY3.json")
                            filename = review_file.name
                            if "_" in filename:
                                # Split and extract username part
                                parts = filename.replace(".json", "").split("_")
                                if (
                                    len(parts) >= 3
                                ):  # compartment_reviews_username_deviceid
                                    file_username = parts[
                                        2
                                    ].lower()  # The username part
                                else:
                                    file_username = "unknown"
                            else:
                                file_username = "unknown"

                            # Load the review data
                            with open(review_file, "r", encoding="utf-8") as f:
                                review_data = json.load(f)

                            if isinstance(review_data, list) and review_data:
                                self.logger.info(
                                    f"Loading {len(review_data)} reviews from {filename} (user: {file_username})"
                                )

                                # Process each review
                                for review in review_data:
                                    key = (
                                        review.get("HoleID"),
                                        review.get("From"),
                                        review.get("To"),
                                    )

                                    # Check if this is current user's review
                                    if file_username == current_user.lower():
                                        user_reviews[key] = review
                                    else:
                                        # Other user's review
                                        if key not in other_reviews:
                                            other_reviews[key] = []
                                        review["_source_file"] = (
                                            filename  # Track which file it came from
                                        )
                                        review["_file_user"] = (
                                            file_username  # Track the user
                                        )
                                        other_reviews[key].append(review)

                        except Exception as e:
                            self.logger.error(
                                f"Error loading review file {review_file.name}: {e}"
                            )

                    # Show summary
                    if user_reviews:
                        # Get classification distribution for current user
                        classifications = {}
                        for review in user_reviews.values():
                            # Check lowercase first (matches save format), then uppercase for backwards compatibility
                            classification = (
                                review.get("classification")
                                or review.get("Classification")
                                or review.get("Lithology")
                                or "Unassigned"
                            )
                            classifications[classification] = (
                                classifications.get(classification, 0) + 1
                            )
                        self.logger.info(
                            f"Current user classifications: {classifications}"
                        )

                    else:
                        self.logger.warning(
                            "No reviews found for current user in any review file"
                        )
                    self.logger.info(f"=== REVIEW LOADING SUMMARY ===")
                    self.logger.info(
                        f"Loaded {len(user_reviews)} reviews by {current_user}, "
                        f"{len(other_reviews)} compartments reviewed by others"
                    )
                    
                    # DEBUG: Log structure of review dicts for RegisterStore integration
                    self.logger.info("=== REVIEW DICT STRUCTURE DEBUG ===")
                    if user_reviews:
                        sample_key = next(iter(user_reviews.keys()))
                        sample_review = user_reviews[sample_key]
                        self.logger.info(f"user_reviews sample key: {sample_key}")
                        self.logger.info(f"user_reviews sample key types: ({type(sample_key[0]).__name__}, {type(sample_key[1]).__name__}, {type(sample_key[2]).__name__})")
                        self.logger.info(f"user_reviews sample value keys: {list(sample_review.keys())}")
                        self.logger.info(f"user_reviews sample value: {sample_review}")
                    
                    if other_reviews:
                        sample_key = next(iter(other_reviews.keys()))
                        sample_list = other_reviews[sample_key]
                        self.logger.info(f"other_reviews sample key: {sample_key}")
                        self.logger.info(f"other_reviews sample value (list len={len(sample_list)})")
                        if sample_list:
                            self.logger.info(f"other_reviews sample value[0] keys: {list(sample_list[0].keys())}")
                            self.logger.info(f"other_reviews sample value[0]: {sample_list[0]}")
                    self.logger.info("=== END REVIEW DICT STRUCTURE DEBUG ===")

                except Exception as e:
                    self.logger.error(f"Error loading reviews: {e}", exc_info=True)
                    user_reviews = {}
                    other_reviews = {}

            # Store other reviews for later use
            self.other_user_reviews = other_reviews
            
            # Build RegisterStore cache from the loaded dicts (O(n), no file I/O)
            if self.data_coordinator and self.data_coordinator.register_store:
                self.data_coordinator.register_store.build_cache_from_review_dicts(
                    user_reviews,
                    other_reviews,
                    current_user
                )

            # Second pass: create images with calculated intervals
            for img_data in temp_images:
                interval = hole_intervals.get(img_data["hole_id"], 1)
                depth_from = img_data["depth_to"] - interval

                # Skip UID extraction during initial scan for performance
                # UID will be extracted lazily when image is actually displayed
                compartment_uid = None

                img = CompartmentImage(
                    filename=img_data["filename"],
                    hole_id=img_data["hole_id"],
                    depth_from=depth_from,
                    depth_to=img_data["depth_to"],
                    image_path=img_data["path"],
                    moisture_status=img_data["moisture"],
                    compartment_uid=compartment_uid,  # Include UID
                )

                # CSV data will be loaded lazily when needed (during filtering/display)
                # No need to pre-load for 80,000+ images during startup
                img.csv_data = {}
                img.in_csv = False

                # Load review from user's data first
                review_key = (img_data["hole_id"], depth_from, img_data["depth_to"])
                if review_key in user_reviews:
                    latest_review = user_reviews[review_key]

                    # Load classification - check multiple possible field names (including lowercase)
                    classification_value = None
                    for field_name in [
                        "classification",
                        "Classification",
                        "Lithology",
                        "Rock_Type",
                    ]:
                        if field_name in latest_review:
                            classification_value = latest_review.get(field_name)
                            if classification_value:
                                break

                    # Convert classification string to enum
                    if classification_value and classification_value != "":
                        img.classification = self._string_to_enum(str(classification_value))
                    else:
                        img.classification = ClassificationCategory.UNASSIGNED

                    # Load other review metadata
                    img.comments = latest_review.get("Comments", "")
                    img.classified_by = latest_review.get("Reviewed_By", "")
                    img.classified_date = latest_review.get("Review_Date", "")
                    
                    # Load tags from register
                    tags_value = latest_review.get("tags", [])
                    if tags_value:
                        img.tags = set(tags_value) if isinstance(tags_value, list) else set()
                    else:
                        img.tags = set()

                    # DO NOT initialize csv_data here - it will be loaded from actual CSV
                    # Review data is stored in separate attributes above
                    # Only store review_data in csv_data if it already has CSV columns
                    # This prevents blocking CSV data from being loaded later

                    # Mark as saved in last_save_state
                    if not hasattr(self, "last_save_state"):
                        self.last_save_state = {}
                    self.last_save_state[img.filename] = img.classification
                    img._has_saved_classification = True

                    # Track original loaded state to avoid re-saving unchanged items
                    img.original_classification = img.classification
                    img.original_comments = img.comments
                    img.original_classified_by = img.classified_by
                    img.original_classified_date = img.classified_date
                    img.original_tags = img.tags.copy() if hasattr(img, "tags") and img.tags else set()

                self.all_images.append(img)
                # Build index for fast lookups
                self.image_index[(img.hole_id, img.depth_to)] = img

            self.logger.info(
                f"Scanned {len(self.all_images)} compartment images "
                f"({len(temp_images)} processed from {file_count} total files)"
            )
            self.logger.info(
                f"Found {len(hole_intervals)} unique holes with intervals: {dict(list(hole_intervals.items())[:5])}..."
            )
            self.logger.info(f"Built image index with {len(self.image_index)} entries")


        except Exception as e:
            self.logger.error(f"Error scanning images: {e}")

    def _string_to_enum(self, classification_str):
        """Map string classification to enum for compatibility"""
        if not classification_str or classification_str == "Unassigned":
            return ClassificationCategory.UNASSIGNED

        # Try exact match with enum values
        for cat in ClassificationCategory:
            if cat.value == classification_str:
                return cat

        # No match, return UNASSIGNED
        return ClassificationCategory.UNASSIGNED

    def _get_classification_string(self, classification) -> str:
        """Safely get classification as string, handles both enum and string"""
        if classification is None or classification == "":
            return "Unassigned"
        if hasattr(classification, "value"):
            return classification.value
        return str(classification)

    def _get_csv_data_cached(
        self, hole_id: str, depth_to: float
    ) -> Tuple[Dict[str, Any], bool]:
        """Get CSV data using DataCoordinator (preferred) or DrillholeDataManager (deprecated)"""
        csv_data = {}
        in_csv = False
        
        # Try DataCoordinator first (new architecture)
        if self.data_coordinator and self.data_coordinator.is_initialized:
            try:
                key = ImageKey(hole_id=hole_id, depth_to=float(depth_to))
                row_data = self.data_coordinator.geological_store.get_row(key)
                if row_data:
                    csv_data = row_data
                    in_csv = True
                    self.logger.debug(f"CSV hit via DataCoordinator: {hole_id}_{depth_to}")
            except Exception as e:
                self.logger.debug(f"DataCoordinator lookup failed for {hole_id}_{depth_to}: {e}")
        
        # Fallback to DrillholeDataManager (deprecated)
        elif self.drillhole_data_manager and self.drillhole_data_manager.data_sources:
            try:
                interval_data = self.drillhole_data_manager.get_data_for_interval(hole_id, depth_to)
                if interval_data:
                    csv_data = interval_data
                    in_csv = True
            except Exception as e:
                self.logger.debug(f"DrillholeDataManager lookup failed for {hole_id}_{depth_to}: {e}")
        
        # No data managers available
        if not self.data_coordinator and not self.drillhole_data_manager:
            in_csv = True  # No data loaded, assume all OK to avoid false negatives

        return csv_data, in_csv

    def _get_consensus_classification(self, img: CompartmentImage) -> str:
        """Get consensus classification for an image when showing other users' reviews
        
        Returns the most common classification across all reviewers, or empty string if none.
        """
        # Collect all classifications
        all_classifications = []
        
        # Add current user's classification
        if img.classification and img.classification != ClassificationCategory.UNASSIGNED:
            classification_str = self._get_classification_string(img.classification)
            all_classifications.append(classification_str)
        
        # Add other reviewers' classifications
        if hasattr(img, "other_reviews") and img.other_reviews:
            for review in img.other_reviews:
                classification_value = None
                for field_name in ["classification", "Classification", "Lithology", "Rock_Type"]:
                    if field_name in review and review[field_name]:
                        classification_value = review[field_name]
                        break
                if classification_value:
                    all_classifications.append(str(classification_value))
        
        # Return consensus
        if not all_classifications:
            return ""  # No classifications from anyone
        elif len(all_classifications) == 1:
            return all_classifications[0]
        else:
            # Multiple classifications - return most common
            from collections import Counter
            most_common = Counter(all_classifications).most_common(1)[0][0]
            return most_common

    def _compute_review_metadata(self, img: CompartmentImage) -> dict:
        """Compute review metadata (consensus, count, agreement, tags) for an image
        
        Returns dict with keys:
        - consensus_classification: str
        - review_count: int
        - agreement: str ("Yes", "No", "Single", "")
        - tags: set of tag IDs across all reviewers
        """
        # Collect all classifications
        all_classifications = []
        all_tags = set()
        
        # Add current user's classification
        if img.classification and img.classification != ClassificationCategory.UNASSIGNED:
            classification_str = self._get_classification_string(img.classification)
            all_classifications.append(classification_str)
        
        # Add current user's tags
        if hasattr(img, 'tags') and img.tags:
            all_tags.update(img.tags)
        
        # Add other reviewers' classifications and tags
        if hasattr(img, "other_reviews") and img.other_reviews:
            for review in img.other_reviews:
                # Get classification
                classification_value = None
                for field_name in ["classification", "Classification", "Lithology", "Rock_Type"]:
                    if field_name in review and review[field_name]:
                        classification_value = review[field_name]
                        break
                if classification_value:
                    all_classifications.append(str(classification_value))
                
                # Get tags
                tags_value = review.get("tags", [])
                if tags_value:
                    if isinstance(tags_value, str):
                        # Parse string representation of set/list
                        try:
                            import ast
                            tags_value = ast.literal_eval(tags_value)
                        except:
                            # If it fails, split by comma
                            tags_value = [t.strip() for t in tags_value.split(",") if t.strip()]
                    if isinstance(tags_value, (list, set)):
                        all_tags.update(tags_value)
        
        # Calculate consensus
        review_count = len(all_classifications)
        
        if review_count == 0:
            consensus = ""
            agreement = ""
        elif review_count == 1:
            consensus = all_classifications[0]
            agreement = "Single"
        else:
            # Multiple reviewers
            from collections import Counter
            classification_counts = Counter(all_classifications)
            most_common = classification_counts.most_common(1)[0]
            consensus = most_common[0]
            # Agreement = Yes if everyone agrees, No if there's disagreement
            agreement = "Yes" if most_common[1] == review_count else "No"
        
        return {
            "consensus_classification": consensus,
            "review_count": review_count,
            "agreement": agreement,
            "tags": all_tags
        }

    def _load_hex_color_cache(self):
        """Load all hex colors from image properties register into memory cache for fast sorting"""
        if not self.json_manager:
            return
        
        try:
            self.logger.info("Loading hex color cache from image properties register...")
            all_props = self.json_manager.get_all_image_properties()
            
            # Build lookup dictionary: (hole_id, depth_from, depth_to) -> {combined_hex, wet_hex, dry_hex}
            for prop in all_props:
                hole_id = prop.get("HoleID")
                depth_from = prop.get("Depth_From")
                depth_to = prop.get("Depth_To")
                
                if hole_id and depth_from is not None and depth_to is not None:
                    key = (hole_id, float(depth_from), float(depth_to))
                    combined_hex = prop.get("Combined_Hex", "")
                    wet_hex = prop.get("Wet_Hex", "")
                    dry_hex = prop.get("Dry_Hex", "")
                    
                    # Derive combined if not present
                    if not combined_hex:
                        combined_hex = dry_hex or wet_hex
                    
                    self.hex_color_cache[key] = {
                        "combined_hex": combined_hex,
                        "wet_hex": wet_hex,
                        "dry_hex": dry_hex
                    }
            
            self.logger.info(f"Loaded {len(self.hex_color_cache)} hex colors into cache")
        except Exception as e:
            self.logger.error(f"Error loading hex color cache: {e}")

    def _clear_csv_cache(self):
        """Clear CSV cache when data changes"""
        # self.csv_data_cache.clear()
        # self.csv_match_cache.clear()
        self.logger.debug("DIDN'T CLEAR CSV data cache")

    def _auto_load_drillhole_csv(self):
        """No longer needed - DrillholeDataManager loads data at app startup"""
        # Data is already loaded in self.drillhole_data_manager.data_sources
        # Just update filter columns to reflect available data
        if self.drillhole_data_manager and self.drillhole_data_manager.data_sources:
            self._update_filter_columns()
            self.logger.info("Using data already loaded by DrillholeDataManager")

    def _populate_drillhole_data_for_images(self):
        """Populate DrillholeDataManager data for all images"""
        if not self.drillhole_data_manager or not hasattr(
            self.drillhole_data_manager, "get_available_columns"
        ):
            return

        dm_columns = self.drillhole_data_manager.get_available_columns()
        if not dm_columns:
            return

        # Cache hole data to avoid repeated queries
        hole_data_cache = {}

        for img in self.all_images:
            if img.hole_id not in hole_data_cache:
                hole_data_cache[img.hole_id] = (
                    self.drillhole_data_manager.get_data_for_hole(img.hole_id)
                )

            hole_data = hole_data_cache[img.hole_id]
            if not hole_data.empty:
                # Find matching row for this depth
                matching_rows = hole_data[hole_data["to"] == img.depth_to]
                if not matching_rows.empty:
                    # Add DrillholeDataManager data to image's csv_data
                    for source, cols in dm_columns.items():
                        for col_name, col_type in cols:
                            if col_name in matching_rows.columns:
                                # Store with source identifier
                                display_name = f"{col_name} ({source})"
                                img.csv_data[display_name] = matching_rows.iloc[0][
                                    col_name
                                ]

    def _detect_all_reviewers(self) -> Dict[str, List[str]]:
        """
        Detect all reviewers from JSONRegisterManager (authoritative source).
        
        Returns:
            Dict mapping reviewer name to list of column IDs:
            {
                "gsymonds": ["rev_gsymonds_classification", "rev_gsymonds_comments", ...],
                "jsmith": ["rev_jsmith_classification", ...]
            }
        """
        reviewers = set()
        
        # Query JSONRegisterManager for ALL reviewers from loaded JSON files
        if hasattr(self, 'json_register_manager') and self.json_register_manager:
            try:
                # Get all reviews from all users
                all_reviews_df = self.json_register_manager.get_all_compartments_all_users()
                
                if all_reviews_df is not None and not all_reviews_df.empty:
                    # Check for reviewer columns (different possible names)
                    reviewer_cols = ['Reviewed_By', 'reviewer', 'classified_by', '_file_user']
                    for col in reviewer_cols:
                        if col in all_reviews_df.columns:
                            reviewer_names = all_reviews_df[col].dropna().unique()
                            reviewers.update(reviewer_names)
                    
                    self.logger.debug(f"Found {len(reviewers)} reviewers from JSONRegisterManager")
                else:
                    self.logger.warning("No reviews found in JSONRegisterManager")
                    
            except Exception as e:
                self.logger.warning(f"Could not get reviewers from JSONRegisterManager: {e}")
        
        # Fallback: scan images (less reliable, but better than nothing)
        if not reviewers:
            self.logger.debug("Falling back to scanning images for reviewers")
            for img in self.all_images[:1000]:  # Sample only
                if getattr(img, "classified_by", None):
                    reviewers.add(img.classified_by)
        
        # Build per-reviewer column names
        reviewer_columns = {}
        for reviewer in sorted(reviewers):
            safe = reviewer.lower().replace(" ", "_").replace(".", "_")
            reviewer_columns[reviewer] = [
                f"rev_{safe}_classification",
                f"rev_{safe}_comments",
                f"rev_{safe}_tags",
            ]
        
        self.logger.info(f"Detected {len(reviewers)} reviewers: {sorted(reviewers)}")
        return reviewer_columns

    def _update_filter_columns(self):
        """Update filter columns after loading CSV data"""
        self.logger.debug("=== _update_filter_columns ===")
        
        # Key columns that should NOT be duplicated across sources
        # These are the standard join keys that appear in every CSV
        KEY_COLUMNS = {
            'holeid', 'hole_id', 'bhid', 'drillhole_id',  # Hole ID variants
            'from', 'depth_from', 'sampfrom', 'geolfrom', 'from_m',  # From depth variants
            'to', 'depth_to', 'sampto', 'geolto', 'to_m',  # To depth variants
        }
        
        # Base columns from register/image data (not CSV)
        base_columns = [
            "hole_id",
            "depth_from",
            "depth_to",
            "classification",  # Current user's classification
            "classified_by",  # Current user who classified
            "consensus_classification",  # Computed: most common across all reviewers
            "all_classifications",  # Computed: comma-separated list of all classifications
            "all_reviewers",  # Computed: comma-separated list of all reviewer usernames
            "review_count",  # Computed: number of reviewers
            "agreement",  # Computed: unanimous/majority/split/none
            "moisture_status",
            "comments",
            "hex_color",  # From image properties register - for color-based filtering
            "combined_hex",  # Combined hex color (alias for hex_color)
            "wet_hex",  # Wet variant hex color
            "dry_hex",  # Dry variant hex color
            # Validation columns from QAQC register
            "validation_status",  # "flagged", "pending", "resolved", or ""
            "validation_severity",  # "warning", "review", "critical", or ""
            "validation_source",  # Analysis that flagged this (e.g., "random_forest", "mahalanobis")
            "validation_message",  # Brief description of the validation issue
        ]
        
        # Add tag columns (one per tag definition)
        if hasattr(self, 'item_manager') and self.item_manager:
            for tag_def in self.item_manager.get_all_tags():
                tag_column_name = f"tag_{tag_def.id}"
                base_columns.append(tag_column_name)
                self.logger.debug(f"Added tag column: {tag_column_name}")
        
        # Detect all reviewers and add per-reviewer columns
        self._reviewer_columns = self._detect_all_reviewers()
        for reviewer, cols in self._reviewer_columns.items():
            for col in cols:
                base_columns.append(col)
                self.logger.debug(f"Added reviewer column: {col}")
        
        combined_columns = base_columns.copy()
        self.logger.debug(f"Base columns: {len(base_columns)}")

        # Track which key columns we've already added (to avoid duplicates)
        added_key_columns = set()
        for col in base_columns:
            col_lower = col.lower().replace('_', '').replace('-', '')
            if col_lower in KEY_COLUMNS or col.lower() in KEY_COLUMNS:
                added_key_columns.add(col_lower)
        
        # Get CSV columns from DataCoordinator (preferred) or DrillholeDataManager (deprecated)
        csv_columns_added = 0
        csv_columns_skipped = 0
        
        if self.data_coordinator and self.data_coordinator.geological_store.is_loaded:
            # Use DataCoordinator's GeologicalStore
            available = self.data_coordinator.geological_store.get_available_columns()
            source_count = len(available)
            self.logger.debug(f"Found {source_count} data sources in GeologicalStore")
            
            for source_name, cols in available.items():
                self.logger.debug(f"Processing source '{source_name}': {len(cols)} columns")
                
                for col_name, col_type in cols:
                    # Normalize column name for key column detection
                    col_lower = col_name.lower().replace('_', '').replace('-', '')
                    
                    # Check if this is a key column
                    is_key_column = col_lower in KEY_COLUMNS or col_name.lower() in KEY_COLUMNS
                    
                    if is_key_column:
                        # Skip if we already have this key column
                        if col_lower in added_key_columns:
                            csv_columns_skipped += 1
                            continue
                        # Add without source suffix (key columns are universal)
                        display_name = col_name
                        added_key_columns.add(col_lower)
                    else:
                        # Non-key column: add with source identifier if multiple sources
                        if source_count > 1:
                            display_name = f"{col_name} ({source_name})"
                        else:
                            display_name = col_name
                    
                    if display_name not in combined_columns:
                        combined_columns.append(display_name)
                        csv_columns_added += 1
            
            self.logger.info(f"Added {csv_columns_added} CSV columns from {source_count} sources (skipped {csv_columns_skipped} duplicate key columns)")
        else:
            self.logger.warning("GeologicalStore not loaded - CSV columns unavailable")

        # Add RC metrics columns if available
        rc_metrics_added = 0
        if self.data_coordinator and self.data_coordinator.has_rc_metrics:
            try:
                rc_df = self.data_coordinator.get_rc_metrics_dataframe()
                if rc_df is not None and not rc_df.empty:
                    # Add RC metrics columns (skip hole_id and depth_to which are key columns)
                    for col in rc_df.columns:
                        col_lower = col.lower()
                        if col_lower in ('hole_id', 'depth_to'):
                            continue
                        display_name = f"{col} (RC Metrics)"
                        if display_name not in combined_columns:
                            combined_columns.append(display_name)
                            rc_metrics_added += 1
                    self.logger.info(f"Added {rc_metrics_added} RC metrics columns")
            except Exception as e:
                self.logger.warning(f"Could not add RC metrics columns: {e}")

        self.logger.info(f"Total filter columns: {len(combined_columns)}")

        # Build columns_info for filter rows
        columns_info = {}
        for col in combined_columns:
            columns_info[col] = (
                self.all_images[0].csv_data.get(col) if self.all_images else None
            )

        # Build register_data DataFrame from ALL images (not just displayed)
        # so we have data for the filter dropdowns
        images_to_use = self.all_images if self.all_images else self.displayed_images
        self.logger.debug(f"Building register_data from {len(images_to_use) if images_to_use else 0} images")
        
        if images_to_use:
            # Build base columns from image metadata and JSON register
            data_dict = {
                "hole_id": [img.hole_id for img in images_to_use],
                "depth_from": [img.depth_from for img in images_to_use],
                "depth_to": [img.depth_to for img in images_to_use],
                "classification": [
                    str(img.classification).replace("ClassificationCategory.", "") 
                    if img.classification else "" 
                    for img in images_to_use
                ],
                "classified_by": [img.classified_by or "" for img in images_to_use],
                "moisture_status": [img.moisture_status or "" for img in images_to_use],
                "comments": [img.comments or "" for img in images_to_use],
            }
            
            # Add hex_color - use hex color cache (faster than csv_data)
            # FAST: Pre-loaded cache from image properties register
            data_dict["hex_color"] = []
            data_dict["combined_hex"] = []
            data_dict["wet_hex"] = []
            data_dict["dry_hex"] = []
            
            for img in images_to_use:
                key = (img.hole_id, img.depth_from, img.depth_to)
                hex_data = self.hex_color_cache.get(key, {})
                
                # Extract hex strings from cache dict (cache stores dicts with combined/wet/dry keys)
                if isinstance(hex_data, dict):
                    combined = hex_data.get("combined_hex", "")
                    wet_hex = hex_data.get("wet_hex", "")
                    dry_hex = hex_data.get("dry_hex", "")
                else:
                    # Fallback if cache contains old string format
                    combined = hex_data if hex_data else ""
                    wet_hex = combined
                    dry_hex = combined
                
                data_dict["hex_color"].append(combined)
                data_dict["combined_hex"].append(combined)
                data_dict["wet_hex"].append(wet_hex)
                data_dict["dry_hex"].append(dry_hex)
            
            # Add consensus/review data - compute actual values for filtering
            self.logger.debug("Computing review metadata for all images...")
            consensus_list = []
            review_count_list = []
            agreement_list = []
            all_reviewers_list = []
            all_classifications_list = []
            
            for img in images_to_use:
                review_metadata = self._compute_review_metadata(img)
                consensus_list.append(review_metadata["consensus_classification"])
                review_count_list.append(review_metadata["review_count"])
                agreement_list.append(review_metadata["agreement"])
                
                # Build list of all reviewers for this image
                reviewers = []
                if img.classified_by:
                    reviewers.append(img.classified_by)
                if hasattr(img, "other_reviews") and img.other_reviews:
                    for review in img.other_reviews:
                        reviewer = review.get("Reviewed_By", "")
                        if reviewer and reviewer not in reviewers:
                            reviewers.append(reviewer)
                all_reviewers_list.append(",".join(reviewers) if reviewers else "")
                
                # Build list of all classifications for this image
                classifications = []
                if img.classification and img.classification != ClassificationCategory.UNASSIGNED:
                    classifications.append(self._get_classification_string(img.classification))
                if hasattr(img, "other_reviews") and img.other_reviews:
                    for review in img.other_reviews:
                        for field in ["classification", "Classification", "Lithology", "Rock_Type"]:
                            if field in review and review[field]:
                                class_val = str(review[field])
                                if class_val not in classifications:
                                    classifications.append(class_val)
                                break
                all_classifications_list.append(",".join(classifications) if classifications else "")
            
            data_dict["consensus_classification"] = consensus_list
            data_dict["review_count"] = review_count_list
            data_dict["agreement"] = agreement_list
            data_dict["all_reviewers"] = all_reviewers_list
            data_dict["all_classifications"] = all_classifications_list
            
            self.logger.debug(f"Computed review metadata for {len(images_to_use)} images")
            
            # Add tag columns - use image.tags directly (fast, already loaded)
            if hasattr(self, 'item_manager') and self.item_manager:
                for tag_def in self.item_manager.get_all_tags():
                    tag_column_name = f"tag_{tag_def.id}"
                    data_dict[tag_column_name] = [
                        "Yes" if (img.tags and tag_def.id in img.tags) else "No"
                        for img in images_to_use
                    ]

            # Add per-reviewer columns - METADATA ONLY (like CSV columns)
            # RegisterStore is indexed - fetch on-demand during filtering
            if hasattr(self, '_reviewer_columns') and self._reviewer_columns:
                self.logger.debug(f"Adding per-reviewer columns for {len(self._reviewer_columns)} reviewers")
                
                # Initialize columns with None placeholders (will be fetched on-demand)
                for reviewer, cols in self._reviewer_columns.items():
                    for col in cols:
                        data_dict[col] = [None] * len(images_to_use)
                
                self.logger.debug(f"Added per-reviewer column placeholders (on-demand fetch enabled)")

            
            # Add CSV columns - METADATA ONLY: Store column info, fetch on-demand during filtering
            # GeologicalStore is indexed for O(1) lookups - no need to pre-populate 127k rows
            if self.data_coordinator and self.data_coordinator.geological_store.is_loaded:
                self.logger.info("Adding CSV column metadata (values fetched on-demand)...")
                
                available = self.data_coordinator.geological_store.get_available_columns()
                source_count = len(available)
                
                # Store CSV column metadata for on-demand querying
                csv_column_metadata = {}
                
                for source_name, cols in available.items():
                    for col_name, col_type in cols:
                        # Skip key columns already added
                        col_lower = col_name.lower().replace('_', '').replace('-', '')
                        if col_lower in added_key_columns or col_name.lower() in added_key_columns:
                            continue
                        
                        if source_count > 1:
                            display_name = f"{col_name} ({source_name})"
                        else:
                            display_name = col_name
                        
                        if display_name not in data_dict:
                            # Store metadata for on-demand querying
                            csv_column_metadata[display_name] = {
                                'source': source_name,
                                'column': col_name,
                                'type': col_type
                            }
                            
                            # Add placeholder column (None = will be fetched on-demand)
                            data_dict[display_name] = [None] * len(images_to_use)
                
                # Store metadata for filter system
                self.csv_column_metadata = csv_column_metadata
                
                self.logger.info(f"Added {len(csv_column_metadata)} CSV columns (metadata only, on-demand fetch enabled)")
                
            else:
                # DataCoordinator not available - no CSV columns
                self.csv_column_metadata = {}
                self.logger.warning("GeologicalStore not loaded - CSV columns unavailable")

            # Add RC metrics columns if available
            if self.data_coordinator and self.data_coordinator.has_rc_metrics:
                try:
                    rc_df = self.data_coordinator.get_rc_metrics_dataframe()
                    if rc_df is not None and not rc_df.empty:
                        rc_metrics_added = 0

                        # Vectorized merge instead of O(n*m) lookups
                        # Create DataFrame with image keys for efficient merge
                        image_keys = pd.DataFrame({
                            'hole_id': [img.hole_id.upper() for img in images_to_use],
                            'depth_to': [img.depth_to for img in images_to_use]
                        })

                        # Normalize RC metrics hole_id to uppercase for matching
                        rc_df_normalized = rc_df.copy()
                        rc_df_normalized['hole_id'] = rc_df_normalized['hole_id'].str.upper()

                        # Drop duplicates to ensure 1:1 merge (keep first occurrence)
                        rc_df_normalized = rc_df_normalized.drop_duplicates(
                            subset=['hole_id', 'depth_to'], keep='first'
                        )

                        # Single merge aligns all RC metrics with images
                        merged = image_keys.merge(
                            rc_df_normalized,
                            on=['hole_id', 'depth_to'],
                            how='left'
                        )

                        # Add each RC metrics column to data_dict
                        for col in rc_df.columns:
                            if col.lower() in ('hole_id', 'depth_to'):
                                continue
                            display_name = f"{col} (RC Metrics)"
                            if display_name not in data_dict:
                                data_dict[display_name] = merged[col].tolist()
                                rc_metrics_added += 1

                        self.logger.info(f"Added {rc_metrics_added} RC metrics columns with data")
                except Exception as e:
                    self.logger.warning(f"Could not populate RC metrics columns: {e}")

            register_data = pd.DataFrame(data_dict)
        else:
            register_data = pd.DataFrame(columns=combined_columns)

        # Store register_data for lazy loading
        self.filter_register_data = register_data

        # Update any existing filter rows
        for row in self.filter_rows:
            row.update_columns(columns_info, register_data)

        # Build sort columns list (used for both dropdowns)
        sort_columns = [
            "Average Hex Colour",
            "hole_id",
            "depth_from",
            "depth_to",
            "classification",
            "consensus_classification",
            "review_count",
            "agreement",
            "hex_color",
            "combined_hex",
            "wet_hex",
            "dry_hex",
            # Validation columns
            "validation_status",
            "validation_severity",
            "validation_source",
        ]
        
        # Add per-reviewer columns to sort options
        if hasattr(self, '_reviewer_columns') and self._reviewer_columns:
            for reviewer, cols in self._reviewer_columns.items():
                for col in cols:
                    if col not in sort_columns:
                        sort_columns.append(col)
        
        # Add CSV columns to sort options
        for col in combined_columns:
            if col not in sort_columns:
                sort_columns.append(col)
        
        # Update primary sort dropdown with new columns
        if hasattr(self, 'sort_dropdown'):
            # Check if it's a searchable dropdown or regular OptionMenu
            if hasattr(self.sort_dropdown, 'update_items'):
                # ThemedSearchableOptionMenu
                self.sort_dropdown.update_items(sort_columns)
            else:
                # Regular OptionMenu
                menu = self.sort_dropdown["menu"]
                menu.delete(0, "end")
                for col in sort_columns:
                    menu.add_command(
                        label=col,
                        command=lambda c=col: self.sort_column_var.set(c)
                    )
        
        # Update secondary sort dropdown with new columns
        if hasattr(self, 'sort_dropdown_secondary'):
            sort_columns_with_none = ["(none)"] + sort_columns
            # Check if it's a searchable dropdown or regular OptionMenu
            if hasattr(self.sort_dropdown_secondary, 'update_items'):
                # ThemedSearchableOptionMenu
                self.sort_dropdown_secondary.update_items(sort_columns_with_none)
            else:
                # Regular OptionMenu
                menu2 = self.sort_dropdown_secondary["menu"]
                menu2.delete(0, "end")
                for col in sort_columns_with_none:
                    menu2.add_command(
                        label=col,
                        command=lambda c=col: self.sort_column_var_secondary.set(c)
                    )

    def get_csv_value_for_image(self, img, column_display_name):
        """
        Fetch CSV value on-demand from GeologicalStore.
        
        Args:
            img: ImageInfo object
            column_display_name: Display name of column (e.g. "Fe% (exassay)")
        
        Returns:
            Value from GeologicalStore, or None if not found
        """
        if not hasattr(self, 'csv_column_metadata') or column_display_name not in self.csv_column_metadata:
            return None
        
        metadata = self.csv_column_metadata[column_display_name]
        source = metadata['source']
        column = metadata['column']
        
        from processing.DataManager.keys import ImageKey
        key = ImageKey(img.hole_id, img.depth_to)
        
        # Fetch just this one value from the indexed GeologicalStore
        return self.data_coordinator.geological_store.get_value(key, column, source_name=source)
    
    def get_unique_values_for_column(self, column_display_name, max_sample=1000):
        """
        Get unique values for a column (for filter dropdowns).
        
        Samples images for CSV columns, uses register_data for register columns.
        
        Args:
            column_display_name: Display name of column
            max_sample: Maximum images to sample for unique values
        
        Returns:
            List of unique values
        """
        # Check if it's a CSV column (needs sampling from GeologicalStore)
        if hasattr(self, 'csv_column_metadata') and column_display_name in self.csv_column_metadata:
            metadata = self.csv_column_metadata[column_display_name]
            source = metadata['source']
            column = metadata['column']
            
            # Sample images to get representative unique values
            from processing.DataManager.keys import ImageKey
            sample_images = self.all_images[:max_sample] if len(self.all_images) > max_sample else self.all_images
            
            unique_values = set()
            for img in sample_images:
                key = ImageKey(img.hole_id, img.depth_to)
                value = self.data_coordinator.geological_store.get_value(key, column, source_name=source)
                if value is not None and pd.notna(value):
                    unique_values.add(value)
            
            return sorted(list(unique_values))
        
        # Register column - use register_data
        if hasattr(self, 'filter_register_data') and column_display_name in self.filter_register_data.columns:
            return self.filter_register_data[column_display_name].dropna().unique().tolist()

        return []

    def _create_dialog(self):
        """Create the main dialog window"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("Lithology Classification")
        # self.dialog.transient(self.parent)

        # Start maximized
        self.dialog.state("zoomed")  # Windows
        # For cross-platform compatibility, also set a fallback geometry
        width = int(self.dialog.winfo_screenwidth() * 0.95)
        height = int(self.dialog.winfo_screenheight() * 0.90)
        x = (self.dialog.winfo_screenwidth() - width) // 2
        y = (self.dialog.winfo_screenheight() - height) // 2
        self.dialog.geometry(f"{width}x{height}+{x}+{y}")

        # Apply theme
        if self.gui_manager:
            self.gui_manager.configure_ttk_styles(self.dialog)
            self.dialog.configure(bg=self.gui_manager.theme_colors["background"])

        # Create UI
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self._create_ui_content(main_frame)

        # Bind keyboard shortcuts
        self._bind_shortcuts()

        # Handle close
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_close)

        self.logger.info("Dialog window created")

        # Initialize hole-by-hole mode if we have images
        if self.all_images:
            # Get unique holes from all images
            holes = sorted(set(img.hole_id for img in self.all_images))
            self.unique_holes = holes

            if self.unique_holes and self.hole_by_hole_mode:
                self.current_hole_index = 0
                self._update_hole_navigation()
                # Note: Don't apply filters here as it will be done in show() method

    def _create_ui_content(self, parent):
        """Create main UI layout"""
        # Configure grid
        parent.grid_rowconfigure(1, weight=1)
        parent.grid_columnconfigure(1, weight=1)  # Image area gets most space

        # Top toolbar
        self._create_toolbar(parent)

        # Left panel - thin filters
        left_frame = ttk.Frame(parent, width=200)  # Narrower width
        left_frame.grid(row=1, column=0, sticky="ns", padx=(0, 5))
        left_frame.grid_propagate(False)  # Maintain fixed width
        self._create_left_panel(left_frame)

        # Right panel - image grid
        right_frame = ttk.Frame(parent)
        right_frame.grid(row=1, column=1, sticky="nsew")
        self._create_right_panel(right_frame)
        
        # Load visualization configuration after grid canvas is created
        self.grid_canvas._load_viz_config()

        # Bottom status
        self._create_status_bar(parent)

    def _create_toolbar(self, parent):
        """Create toolbar with classification modes and actions"""
        toolbar = ttk.Frame(parent)
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 5))

        # Classification modes - now dynamic
        mode_frame = ttk.LabelFrame(toolbar, text="Classification Mode", padding=5)
        mode_frame.pack(side=tk.LEFT, padx=5)

        # Store reference to mode_frame for rebuilding
        self.mode_frame = mode_frame

        # Build classification buttons dynamically
        self._build_classification_buttons()

        # Add settings button for managing classifications
        settings_btn = ModernButton(
            mode_frame,
            text="⚙ Settings",
            command=self._open_classification_settings,
            color="#6c757d",  # Gray color for settings
            theme_colors=self.gui_manager.theme_colors,
        )
        settings_btn.pack(side=tk.LEFT, padx=(10, 2))

        # Tags section moved to left panel - removed from toolbar

        # Initialize sort vars (controls will be in filter panel)
        self.sort_column_var = tk.StringVar(value="depth_to")
        self.sort_order_var = tk.StringVar(value="asc")
        self.color_sort_mode = tk.StringVar(value="hue")

        # Display controls frame
        display_frame = ttk.LabelFrame(toolbar, text="Display", padding=5)
        display_frame.pack(side=tk.LEFT, padx=10)

        # Rotation buttons
        ModernButton(
            display_frame,
            text="↻ Rotate",
            command=lambda: self.grid_canvas.rotate_images(90),
            color=self.gui_manager.theme_colors["secondary_bg"],
            theme_colors=self.gui_manager.theme_colors,
        ).pack(side=tk.LEFT, padx=2)

        # Data visualization toggle
        self.show_viz_button = ModernButton(
            display_frame,
            text="📊 Data Viz",
            command=self._toggle_visualizations,
            color=self.gui_manager.theme_colors["accent_blue"],
            theme_colors=self.gui_manager.theme_colors,
        )
        self.show_viz_button.pack(side=tk.LEFT, padx=2)

        # Configure visualizations button
        self.config_viz_button = ModernButton(
            display_frame,
            text="⚙ Viz Config",
            command=self._configure_visualizations,
            color=self.gui_manager.theme_colors["secondary_bg"],
            theme_colors=self.gui_manager.theme_colors,
        )
        self.config_viz_button.pack(side=tk.LEFT, padx=2)

        # Scale controls
        ModernButton(
            display_frame,
            text="Bigger",
            command=lambda: self._scale_images(1.2),
            color=self.gui_manager.theme_colors["secondary_bg"],
            theme_colors=self.gui_manager.theme_colors,
        ).pack(side=tk.LEFT, padx=2)

        ModernButton(
            display_frame,
            text="Smaller",
            command=lambda: self._scale_images(0.8),
            color=self.gui_manager.theme_colors["secondary_bg"],
            theme_colors=self.gui_manager.theme_colors,
        ).pack(side=tk.LEFT, padx=2)

        # Export/Save
        action_frame = ttk.Frame(toolbar)
        action_frame.pack(side=tk.RIGHT, padx=5)

        ModernButton(
            action_frame,
            text="💾 Save",
            command=self._manual_save,
            color=self.gui_manager.theme_colors["accent_green"],
            theme_colors=self.gui_manager.theme_colors,
        ).pack(side=tk.LEFT, padx=2)

        ModernButton(
            action_frame,
            text="Export CSV",
            command=self._export_csv,
            color=self.gui_manager.theme_colors["accent_blue"],
            theme_colors=self.gui_manager.theme_colors,
        ).pack(side=tk.LEFT, padx=2)

        ModernButton(
            action_frame,
            text="📄 Export PDF",
            command=self._show_pdf_export,
            color="#dc3545",  # Red color for PDF
            theme_colors=self.gui_manager.theme_colors,
        ).pack(side=tk.LEFT, padx=2)

        ModernButton(
            action_frame,
            text="Import CSV Data",
            command=self._import_csv_data,
            color=self.gui_manager.theme_colors["accent_green"],
            theme_colors=self.gui_manager.theme_colors,
        ).pack(side=tk.LEFT, padx=2)

    def _build_classification_buttons(self):
        """Build classification buttons from ClassificationManager"""
        # Clear existing buttons
        if hasattr(self, "mode_buttons"):
            for btn in self.mode_buttons.values():
                btn.destroy()

        self.mode_buttons = {}

        # Get active classifications from manager
        active_classifications = self.item_manager.get_active_classifications()

        # Create button for each active classification
        for class_def in active_classifications:
            # Create button with classification's color and label
            btn = ModernButton(
                self.mode_frame,
                text=class_def.label,
                command=lambda cid=class_def.id: self._set_mode(cid),
                color=class_def.color,
                theme_colors=self.gui_manager.theme_colors,
            )
            btn.pack(side=tk.LEFT, padx=2)
            self.mode_buttons[class_def.id] = btn

            # Display keybinding in button text if it exists
            if class_def.keybinding:
                btn.config(text=f"{class_def.label} [{class_def.keybinding}]")

    # Tooltip system removed - keybindings now shown in button labels

    def _build_tag_buttons(self):
        """Build tag buttons from ItemManager"""
        for widget in self.tag_frame.winfo_children():
            widget.destroy()

        self.tag_buttons = {}

        active_tags = self.item_manager.get_active_tags()

        if not active_tags:
            no_tags_label = ttk.Label(
                self.tag_frame, text="No tags configured", font=("Arial", 9, "italic")
            )
            no_tags_label.pack(pady=5)
            print(f"DEBUG: [LoggingReview] No active tags found")
            return

        print(f"DEBUG: [LoggingReview] Creating {len(active_tags)} tag buttons")

        # Row of tag buttons
        btn_row = ttk.Frame(self.tag_frame)
        btn_row.pack(fill=tk.X)

        for tag in active_tags:
            button_text = f"{tag.icon} {tag.label}" if tag.icon else tag.label
            if tag.keybinding:
                button_text += f" [{tag.keybinding}]"

            btn = ModernButton(
                btn_row,
                text=button_text,
                color=tag.color,
                command=lambda t=tag.id: self._set_mode(t),
                theme_colors=self.gui_manager.theme_colors,
            )
            btn.pack(side=tk.LEFT, padx=2)
            self.tag_buttons[tag.id] = btn

        # Auto-skip checkbox: when enabled, scrolling past images in tag mode
        # marks them as "not_<tag>" so they won't reappear as unreviewed
        if not hasattr(self, "auto_skip_tag_var"):
            self.auto_skip_tag_var = tk.BooleanVar(value=False)

        skip_row = ttk.Frame(self.tag_frame)
        skip_row.pack(fill=tk.X, pady=(4, 0))

        auto_skip_cb = ttk.Checkbutton(
            skip_row,
            text="Auto-skip scrolled (mark unseen as 'not' active tag)",
            variable=self.auto_skip_tag_var,
        )
        auto_skip_cb.pack(side=tk.LEFT)

        hint = ttk.Label(
            skip_row,
            text="  Shift+key = skip selected",
            font=("Arial", 8, "italic"),
        )
        hint.pack(side=tk.LEFT)

    def _open_classification_settings(self):
        """Open unified classification and tag settings dialog"""
        from gui.ReviewDialog.classification_settings_dialog import (
            ClassificationSettingsDialog,
        )

        dialog = ClassificationSettingsDialog(
            self.dialog,
            self.item_manager,
            self.gui_manager,
        )

        result = dialog.show()

        # Always rebuild UI after closing settings dialog (even if no explicit changes)
        # This ensures active/inactive toggles are reflected immediately
        self._rebuild_classification_ui()

        if result:
            self.logger.info("Classification settings modified - UI rebuilt")

    def _rebuild_classification_ui(self):
        """Rebuild classification UI after settings change"""
        # Rebuild toolbar buttons
        self._build_classification_buttons()

        # Rebuild tag buttons
        self._build_tag_buttons()
        
        # Rebuild tag filter checkboxes
        if hasattr(self, 'tag_checks_frame'):
            self._rebuild_tag_filter_checks()

        # Rebind keyboard shortcuts (includes both classifications and tags)
        self._bind_shortcuts()

        # Refresh the grid to show new colors
        if self.grid_canvas:
            self.grid_canvas.refresh_all_cells()

        self.logger.info("Classification UI rebuilt after settings change")

    def _toggle_scatter_filter(self):
        """Toggle scatter plot filter window - loads data from stores, NOT images"""
        # Check if grid canvas is initialized
        if not self.grid_canvas:
            self.logger.warning("Cannot toggle scatter filter - grid canvas not initialized")
            messagebox.showwarning(
                "Not Ready",
                "The grid canvas is not initialized yet. Please wait for the dialog to finish loading."
            )
            return
        
        if not self.grid_canvas.zoom_preview:  # Reusing zoom_preview variable name for compatibility
            # Build DataFrame from DataCoordinator stores - NOT from images!
            scatter_df = self._build_scatter_dataframe()
            
            if scatter_df is None or scatter_df.empty:
                self.logger.warning("No data available for scatter plot")
                messagebox.showwarning(
                    "No Data",
                    "No geological data available for the scatter plot.\n"
                    "Please ensure CSV data files are loaded."
                )
                return
            
            # Configure the window
            config = FilterWindowConfig(
                window_title="Geological Data Scatter Plot Filter",
                window_width=1400,
                window_height=900,
                default_x_axis="Fe_pct_BEST",
                default_y_axis="SiO2_pct_BEST",
                default_color_by="hex_color",  # Color by hex color from register
                default_point_size=15,
                default_point_alpha=0.6,
                lasso_color="#00ff00",
                lasso_alpha=0.3,
                selection_outline_color="#ffff00",
                selection_outline_width=2,
            )
            
            # Create scatter filter window with pre-built DataFrame
            # No images or data_manager needed - data already loaded!
            self.grid_canvas.zoom_preview = AdvancedFilterWindow(
                parent=self.dialog,
                gui_manager=self.gui_manager,
                data=scatter_df,  # Pass DataFrame directly - NOT images!
                images=None,      # Don't pass images
                data_manager=None,  # Don't need - data already in DataFrame
                color_map_manager=self.color_map_manager,
                item_manager=self.item_manager,
                config=config,
                on_selection=self._handle_scatter_selection,
                on_filter_change=self._handle_filter_change,
            )
            self.grid_canvas.zoom_preview.show()
            self.logger.info(f"Opened scatter filter with {len(scatter_df)} data points")
        else:
            if self.grid_canvas.zoom_preview.is_open:
                self.grid_canvas.zoom_preview.hide()
            else:
                self.grid_canvas.zoom_preview.show()
    
    def _build_scatter_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Build DataFrame for scatter plot from DataCoordinator stores.
        
        Combines:
        - GeologicalStore: CSV columns (Fe%, SiO2%, etc.)
        - RegisterStore: hex_color, classification
        
        Returns:
            DataFrame with hole_id, depth_to, and data columns
        """
        if not self.data_coordinator:
            self.logger.warning("DataCoordinator not available for scatter plot")
            return None
        
        self.logger.info("Building scatter DataFrame from DataCoordinator stores...")
        start_time = time.time()
        
        scatter_df = None
        
        # Get CSV data from GeologicalStore
        if self.data_coordinator.geological_store.is_loaded:
            geo_store = self.data_coordinator.geological_store
            
            # Get all interval data sources (not collar/survey)
            sources = geo_store.get_data_sources()
            csv_dfs = []
            
            for source_name, source in sources.items():
                if source.is_loaded and source.df is not None:
                    # Check if this is an interval dataset (has hole_id and depth_to columns)
                    df = source.df
                    hole_col = None
                    depth_col = None
                    
                    for col in df.columns:
                        if col.lower() in ('holeid', 'hole_id'):
                            hole_col = col
                        if col.lower() in ('to', 'sampto', 'depth_to', 'geolto'):
                            depth_col = col
                    
                    if hole_col and depth_col:
                        # Create standardized copy with hole_id and depth_to
                        source_df = df.copy()
                        source_df['hole_id'] = source_df[hole_col].astype(str).str.upper()
                        source_df['depth_to'] = pd.to_numeric(source_df[depth_col], errors='coerce')
                        source_df['_source'] = source_name
                        csv_dfs.append(source_df)
                        self.logger.debug(f"  Added {len(source_df)} rows from {source_name}")
            
            if csv_dfs:
                # Merge all CSV sources on (hole_id, depth_to)
                scatter_df = csv_dfs[0][['hole_id', 'depth_to', '_source']].copy()
                
                # Add numeric columns from each source
                for source_df in csv_dfs:
                    # Get numeric columns
                    numeric_cols = source_df.select_dtypes(include=[np.number]).columns.tolist()
                    for col in numeric_cols:
                        if col not in scatter_df.columns and col.lower() not in ('hole_id', 'depth_to'):
                            # Merge this column
                            merge_df = source_df[['hole_id', 'depth_to', col]].drop_duplicates(['hole_id', 'depth_to'])
                            scatter_df = scatter_df.merge(
                                merge_df,
                                on=['hole_id', 'depth_to'],
                                how='left'
                            )
                
                self.logger.info(f"  CSV data: {len(scatter_df)} rows, {len(scatter_df.columns)} columns")
        
        # Add hex colors - try register store first, fallback to hex_color_cache
        if scatter_df is not None and not scatter_df.empty:
            register_store = self.data_coordinator.register_store if self.data_coordinator else None
            # Only use register store for hex colors if _properties_cache exists AND has data
            use_register_for_hex = (register_store and 
                                    hasattr(register_store, '_properties_cache') and 
                                    register_store._properties_cache)
            
            if use_register_for_hex:
                self.logger.debug(f"  Adding hex colors from register store ({len(register_store._properties_cache)} entries)...")
                
                hex_colors = []
                wet_hex_colors = []
                dry_hex_colors = []
                combined_hex_colors = []
                
                for _, row in scatter_df.iterrows():
                    key = (row['hole_id'], int(row['depth_to']))
                    props = register_store._properties_cache.get(key)
                    if props:
                        hex_colors.append(props.combined_hex or "")
                        wet_hex_colors.append(props.wet_hex or "")
                        dry_hex_colors.append(props.dry_hex or "")
                        combined_hex_colors.append(props.combined_hex or "")
                    else:
                        hex_colors.append("")
                        wet_hex_colors.append("")
                        dry_hex_colors.append("")
                        combined_hex_colors.append("")
                
                scatter_df['hex_color'] = hex_colors
                scatter_df['wet_hex'] = wet_hex_colors
                scatter_df['dry_hex'] = dry_hex_colors
                scatter_df['combined_hex'] = combined_hex_colors
                
                hex_count = sum(1 for h in hex_colors if h)
                self.logger.info(f"  Added hex color columns from register ({hex_count}/{len(scatter_df)} with values)")
                
            elif hasattr(self, 'hex_color_cache') and self.hex_color_cache:
                # Fallback to dialog's hex_color_cache (loaded at startup)
                self.logger.debug(f"  Adding hex colors from hex_color_cache ({len(self.hex_color_cache)} entries)...")
                
                # Build lookup by (hole_id, depth_to) - cache key is (hole_id, depth_from, depth_to)
                hex_by_hole_depth_to = {}
                for (hole_id, depth_from, depth_to), hex_val in self.hex_color_cache.items():
                    key = (str(hole_id).upper(), int(depth_to))
                    
                    # Extract hex string from dict (cache stores dicts with combined/wet/dry keys)
                    if isinstance(hex_val, dict):
                        hex_str = hex_val.get("combined_hex", "")
                    else:
                        hex_str = hex_val if hex_val else ""
                    
                    if hex_str:
                        hex_by_hole_depth_to[key] = hex_str
                
                self.logger.debug(f"  Built hex lookup with {len(hex_by_hole_depth_to)} unique (hole_id, depth_to) entries")
                
                hex_colors = []
                for _, row in scatter_df.iterrows():
                    key = (str(row['hole_id']).upper(), int(row['depth_to']))
                    hex_colors.append(hex_by_hole_depth_to.get(key, ""))
                
                scatter_df['hex_color'] = hex_colors
                scatter_df['combined_hex'] = hex_colors  # Same as hex_color
                scatter_df['wet_hex'] = hex_colors  # Use combined for all (no separate wet/dry in cache)
                scatter_df['dry_hex'] = hex_colors
                
                hex_count = sum(1 for h in hex_colors if h)
                self.logger.info(f"  Added hex color columns from cache ({hex_count}/{len(scatter_df)} with values)")
            else:
                self.logger.warning("  No hex color source available")
        
        # Add classification and tags from register store or json_manager
        if scatter_df is not None and not scatter_df.empty:
            register_store = self.data_coordinator.register_store if self.data_coordinator else None
            use_register = register_store and getattr(register_store, '_cache_built', False)
            
            if use_register:
                self.logger.debug("  Adding classifications from register cache...")
                
                classifications = []
                consensus_classifications = []
                review_counts = []
                agreements = []
                
                for _, row in scatter_df.iterrows():
                    key = (row['hole_id'], int(row['depth_to']))
                    metadata = register_store._review_cache.get(key)
                    if metadata:
                        classifications.append(metadata.classification or "")
                        consensus_classifications.append(getattr(metadata, 'consensus_classification', "") or "")
                        review_counts.append(getattr(metadata, 'review_count', 0) or 0)
                        agreements.append(getattr(metadata, 'agreement', "") or "")
                    else:
                        classifications.append("")
                        consensus_classifications.append("")
                        review_counts.append(0)
                        agreements.append("")
                
                scatter_df['classification'] = classifications
                scatter_df['consensus_classification'] = consensus_classifications
                scatter_df['review_count'] = review_counts
                scatter_df['agreement'] = agreements
                
                class_count = sum(1 for c in classifications if c)
                self.logger.debug(f"  Added register columns: classification ({class_count}), consensus, review_count, agreement")
                
            elif hasattr(self, 'json_manager') and self.json_manager:
                # Fallback to json_manager
                self.logger.debug("  Adding classifications from json_manager (register cache not built)...")
                
                review_lookup = {}
                try:
                    all_reviews = self.json_manager.get_all_image_reviews()
                    for review in all_reviews:
                        hole_id = review.get("HoleID", "")
                        depth_to = review.get("Depth_To")
                        if hole_id and depth_to is not None:
                            key = (str(hole_id).upper(), int(float(depth_to)))
                            review_lookup[key] = review
                    self.logger.debug(f"  Built review lookup with {len(review_lookup)} entries")
                except Exception as e:
                    self.logger.warning(f"  Error building review lookup: {e}")
                
                classifications = []
                consensus_classifications = []
                review_counts = []
                agreements = []
                
                for _, row in scatter_df.iterrows():
                    key = (str(row['hole_id']).upper(), int(row['depth_to']))
                    review = review_lookup.get(key, {})
                    classifications.append(review.get('Classification', "") or "")
                    consensus_classifications.append(review.get('Consensus_Classification', "") or "")
                    review_counts.append(review.get('Review_Count', 0) or 0)
                    agreements.append(review.get('Agreement', "") or "")
                
                scatter_df['classification'] = classifications
                scatter_df['consensus_classification'] = consensus_classifications
                scatter_df['review_count'] = review_counts
                scatter_df['agreement'] = agreements
                
                class_count = sum(1 for c in classifications if c)
                self.logger.debug(f"  Added columns from json_manager: classification ({class_count})")
            else:
                self.logger.warning("  No classification source available")
                
                # Add tag columns (tag_<tag_id> as boolean)
                active_tags = self.item_manager.get_active_tags()
                if active_tags:
                    self.logger.debug(f"  Adding {len(active_tags)} tag columns...")
                    for tag_def in active_tags:
                        tag_col = f"tag_{tag_def.id}"
                        tag_values = []
                        for _, row in scatter_df.iterrows():
                            # Key must be (str, int) to match cache!
                            key = (row['hole_id'], int(row['depth_to']))
                            metadata = register_store._review_cache.get(key)
                            if metadata and tag_def.id in metadata.tags:
                                tag_values.append(True)
                            else:
                                tag_values.append(False)
                        scatter_df[tag_col] = tag_values
                        tag_count = sum(tag_values)
                        self.logger.debug(f"    Tag '{tag_def.label}': {tag_count} rows tagged")
        
        elapsed = time.time() - start_time
        if scatter_df is not None:
            self.logger.info(f"Built scatter DataFrame: {len(scatter_df)} rows in {elapsed:.2f}s")
        
        return scatter_df

    def _handle_scatter_selection(self, selected_df: pd.DataFrame):
        """
        Handle lasso selection from scatter plot.
        
        The selected_df contains rows from GeologicalStore with hole_id and depth_to.
        We use these keys to filter displayed images.
        """
        if selected_df is None or selected_df.empty:
            self.logger.info("Empty scatter selection")
            # Clear scatter selection info
            self._scatter_selection_info = None
            self._update_active_filters_with_selection(0, 0)
            return
        
        self.logger.info(f"Processing scatter selection: {len(selected_df)} data points selected")
        
        # Extract (hole_id, depth_to) keys from selection
        selected_keys = set()
        
        # Find hole_id column (case-insensitive)
        hole_col = None
        depth_col = None
        for col in selected_df.columns:
            if col.lower() in ('hole_id', 'holeid'):
                hole_col = col
            if col.lower() in ('depth_to', 'to', 'sampto'):
                depth_col = col
        
        if not hole_col or not depth_col:
            self.logger.warning(f"Cannot find hole_id/depth_to columns in selection. Columns: {list(selected_df.columns)}")
            return
        
        for _, row in selected_df.iterrows():
            try:
                hole_id = str(row[hole_col]).upper()
                depth_to = float(row[depth_col])
                selected_keys.add((hole_id, depth_to))
            except (ValueError, TypeError):
                continue
        
        self.logger.info(f"Selection contains {len(selected_keys)} unique depth intervals")
        
        if not selected_keys:
            return
        
        # Filter to images matching selected keys
        matching_images = []
        for img in self.all_images:
            key = (img.hole_id.upper(), float(img.depth_to))
            if key in selected_keys:
                matching_images.append(img)
        
        self.logger.info(f"Found {len(matching_images)} images matching selection")
        
        # Store scatter selection info for active filters display
        self._scatter_selection_info = f"Scatter selection: {len(selected_df)} points → {len(matching_images)} images"
        
        # Update active filters display with selection info
        self._update_active_filters_with_selection(len(selected_df), len(matching_images))
        
        if matching_images:
            # Update displayed images
            self.displayed_images = matching_images
            self._reload_grid()
            
            # Update status
            if hasattr(self, 'status_label') and self.status_label:
                self.status_label.config(
                    text=f"Showing {len(matching_images)} images from scatter selection"
                )
        else:
            self.logger.warning("No images found matching scatter selection")
            DialogHelper.show_message(
                self.dialog,
                "No Matches",
                f"No images found for the {len(selected_keys)} selected intervals.\n"
                "The selected data points may not have corresponding images.",
                message_type="info"
            )
    
    def _update_active_filters_with_selection(self, points_selected: int, images_found: int):
        """Update active filters display including scatter selection info"""
        # Get dynamic filter descriptions
        filter_descriptions = self._get_dynamic_filter_descriptions()
        
        # Add scatter selection info if active
        if points_selected > 0:
            self._scatter_selection_info = f"Scatter selection: {points_selected} points → {images_found} images"
            filter_descriptions.append(self._scatter_selection_info)
        else:
            self._scatter_selection_info = None
        
        self.update_active_filters_display(filter_descriptions)

    def _handle_filter_change(self, filtered_df: pd.DataFrame):
        """
        Handle filter changes from advanced filter window (scatter plot).
        Updates the image grid to show only images matching the filtered data.
        Clears any active scatter selection since filter context has changed.
        """
        # Clear scatter selection state when filters change
        self.pre_scatter_displayed_images = None
        self.logger.info("Filter change detected - clearing scatter selection state")
        
        if filtered_df.empty:
            self.logger.warning("Filter resulted in empty dataset - showing all images")
            self.displayed_images = self.all_images.copy()
            if self.grid_canvas:
                self.grid_canvas.load_images(self.displayed_images)
            self._update_statistics()
            self._update_status("⚠ No images match current filters")
            return
        
        logger.info(f"Advanced filter applied: {len(filtered_df)} data rows match")
        
        # Create set of (hole_id, depth_to) tuples from filtered data
        filtered_keys = set()
        for _, row in filtered_df.iterrows():
            hole_id = row.get("hole_id", "")
            depth_to = row.get("depth_to", row.get("to", None))
            if hole_id and depth_to is not None:
                filtered_keys.add((hole_id, depth_to))
        
        logger.info(f"Filter keys: {len(filtered_keys)} unique intervals from {len(set(k[0] for k in filtered_keys))} holes")
        
        # Match images to filtered data
        matched_images = []
        for img in self.all_images:
            key = (img.hole_id, img.depth_to)
            if key in filtered_keys:
                matched_images.append(img)
        
        if not matched_images:
            logger.warning("No images found matching filter criteria")
            self._update_status("⚠ No images match filter criteria")
            return
        
        logger.info(f"✓ Filter matched {len(matched_images)} images")
        
        # Update displayed images
        self.displayed_images = matched_images
        
        # Reload grid
        if self.grid_canvas:
            logger.info(f"Loading {len(matched_images)} filtered images into grid...")
            self.grid_canvas.load_images(matched_images)
            logger.info("Grid reload complete")
        
        # Update UI
        self._update_statistics()
        
        # Status bar update
        wet_count = sum(1 for img in matched_images if img.moisture_status == 'Wet')
        dry_count = sum(1 for img in matched_images if img.moisture_status == 'Dry')
        
        # Get filter descriptions from AdvancedFilterWindow
        filter_descriptions = []
        if hasattr(self, 'advanced_filter_window') and self.advanced_filter_window:
            if hasattr(self.advanced_filter_window, 'get_active_filter_descriptions'):
                filter_descriptions = self.advanced_filter_window.get_active_filter_descriptions() or []
        
        # Update active filters display with scatter plot filters
        self.update_active_filters_display(filter_descriptions)
        
        self._update_status(
            f"Scatter plot filter applied: {len(matched_images)} images "
            f"({wet_count} Wet, {dry_count} Dry) from {len(filtered_keys)} depth intervals"
        )
        
        logger.info(f"Filter application complete: {len(matched_images)} images displayed")

    def _hex_to_hsl(self, hex_color):
        """Convert hex color to HSL for sorting"""
        # Remove # if present
        hex_color = hex_color.lstrip("#")

        # Convert to RGB (0-1 range)
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0

        # Calculate HSL
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        diff = max_val - min_val

        # Lightness
        l = (max_val + min_val) / 2.0

        if diff == 0:
            h = s = 0  # achromatic
        else:
            # Saturation
            s = (
                diff / (2.0 - max_val - min_val)
                if l > 0.5
                else diff / (max_val + min_val)
            )

            # Hue
            if max_val == r:
                h = ((g - b) / diff + (6 if g < b else 0)) / 6.0
            elif max_val == g:
                h = ((b - r) / diff + 2) / 6.0
            else:
                h = ((r - g) / diff + 4) / 6.0

        return (h, s, l)  # Return as tuple for sorting

    def _apply_sorting(self):
        """Trigger filter refresh - sorting now happens at end of filter pipeline"""
        if not self.all_images:
            return
        
        sort_column = self.sort_column_var.get()
        sort_order = self.sort_order_var.get()
        
        self.logger.info(f"[SORT] User requested sort: {sort_column} ({sort_order})")
        
        # Force re-filter which will apply sort at the end
        self.last_filter_hash = None
        self._apply_filters(force=True)
        
        self._update_status(f"Sorted by {sort_column} ({sort_order})")

    def update_active_filters_display(self, filter_descriptions: List[str]):
        """Update the active filters summary display
        
        Args:
            filter_descriptions: List of filter description strings (e.g., "Fe_pct_BEST > 68")
        """
        # Clear existing display
        for widget in self.active_filters_container.winfo_children():
            widget.destroy()
        
        # Check if Show mode filter is active (not "all")
        current_mode = self.filter_mode.get() if hasattr(self, 'filter_mode') else "all"
        mode_filter_active = current_mode != "all"
        
        # Check if showing other users' reviews
        show_others = self.show_others_var.get() if hasattr(self, 'show_others_var') else False
        
        # Build complete filter list including show mode
        all_filters = []
        
        # Add show mode filter if active (with unified labels)
        if mode_filter_active:
            mode_labels = {
                "unclassified": "Show: Unclassified" + (" (all users)" if show_others else " (mine only)"),
                "classified": "Show: Classified" + (" (all users)" if show_others else " (mine only)"),
                "disagreements": "Show: Conflicts Only"
            }
            if current_mode in mode_labels:
                all_filters.append(mode_labels[current_mode])
        
        # Filter out duplicates from incoming descriptions
        # (some callers may include mode info we already added)
        skip_patterns = ["Show:", "Using:", "Consensus"]
        
        # Add dataset filters (excluding any that look like mode/consensus descriptions)
        for desc in filter_descriptions:
            if not any(pattern in desc for pattern in skip_patterns):
                all_filters.append(desc)
        
        if not all_filters:
            # Show "no filters" message
            self.no_filters_label = ttk.Label(
                self.active_filters_container,
                text="No active filters\n(Click 'Filter Dataset' to add filters)",
                style="Content.TLabel",
                justify=tk.CENTER,
            )
            self.no_filters_label.pack(pady=20)
            if hasattr(self, 'clear_dataset_filters_btn'):
                self.clear_dataset_filters_btn.config(state=tk.DISABLED)
        else:
            # Show each filter
            for i, desc in enumerate(all_filters):
                filter_label = ttk.Label(
                    self.active_filters_container,
                    text=f"• {desc}",
                    style="Content.TLabel",
                    wraplength=180,
                )
                filter_label.pack(anchor="w", padx=5, pady=2)
                
                if i < len(all_filters) - 1:
                    # Add AND between filters
                    and_label = ttk.Label(
                        self.active_filters_container,
                        text="AND",
                        style="Content.TLabel",
                        font=("Arial", 8, "italic"),
                    )
                    and_label.pack(anchor="w", padx=20, pady=1)
            
            if hasattr(self, 'clear_dataset_filters_btn'):
                self.clear_dataset_filters_btn.config(state=tk.NORMAL)
        
        self.logger.info(f"Updated active filters display: {len(all_filters)} filters (mode={current_mode})")
    
    def _clear_all_filters(self):
        """Clear ALL filters including dataset filters, scatter selections, and show mode"""
        # Clear dataset filter rows
        self._clear_dataset_filters()
        
        # Clear scatter selection state
        self.pre_scatter_displayed_images = None
        self._scatter_selection_info = None
        
        # Clear scatter plot selection if window exists
        if hasattr(self, 'grid_canvas') and self.grid_canvas:
            if hasattr(self.grid_canvas, 'zoom_preview') and self.grid_canvas.zoom_preview:
                self.grid_canvas.zoom_preview._clear_selection()
        
        # Reset show mode to "all"
        if hasattr(self, 'filter_mode'):
            self.filter_mode.set("all")
        
        # Reset displayed images to all images
        self.displayed_images = self.all_images.copy()
        
        # Update active filters display
        self.update_active_filters_display([])
        
        # Reload grid
        if self.grid_canvas:
            self.grid_canvas.load_images(self.displayed_images)
        
        # Update UI
        self._update_statistics()
        self._update_status("All filters cleared - showing full dataset")
        
        self.logger.info("Cleared all filters (dataset + scatter + show mode)")
    
    def _add_dataset_filter_row(self):
        """Add a new dynamic filter row to the filter panel"""
        if not self._has_csv_data_available():
            messagebox.showwarning(
                "No Data",
                "Cannot add filters - no CSV data available.\n"
                "Please load a dataset first."
            )
            return
        
        try:
            # Get available columns from geological_store
            if self.data_coordinator and self.data_coordinator.geological_store.is_loaded:
                # get_available_columns returns Dict[source_name, List[Tuple[column_name, DataType]]]
                # Flatten it to get all unique column names
                available_cols = self.data_coordinator.geological_store.get_available_columns()
                columns = []
                for source_name, col_list in available_cols.items():
                    for col_name, col_type in col_list:
                        # Add column with source prefix if multiple sources
                        if len(available_cols) > 1:
                            columns.append(f"{col_name} ({source_name})")
                        else:
                            columns.append(col_name)
                
                # Remove duplicates while preserving order
                columns = list(dict.fromkeys(columns))
            else:
                self.logger.error("Cannot add filter - no data available")
                return
            
            if not columns:
                messagebox.showwarning("No Columns", "No columns available for filtering")
                return
            
            # Add register/review columns FIRST (before CSV columns)
            register_columns = [
                "hole_id",
                "depth_from", 
                "depth_to",
                "classification",           # Current user's classification
                "classified_by",            # Current user who classified
                "consensus_classification", # Most common across all reviewers
                "all_classifications",      # Comma-separated: "BIFhm,BIFs,BIFhm"
                "all_reviewers",            # Comma-separated: "gsymonds,jsmith"
                "review_count",             # Number of reviewers
                "agreement",                # unanimous/majority/split/none
                "moisture_status",
                "comments",
                "combined_hex",
                "wet_hex", 
                "dry_hex",
            ]
            
            # Add tag columns
            if hasattr(self, 'item_manager') and self.item_manager:
                for tag_def in self.item_manager.get_all_tags():
                    register_columns.append(f"tag_{tag_def.id}")
            
            # Merge register columns with CSV columns (register first, then CSV)
            columns = register_columns + [c for c in columns if c not in register_columns]

            # Add RC metrics columns if available
            if self.data_coordinator and self.data_coordinator.has_rc_metrics:
                try:
                    rc_df = self.data_coordinator.get_rc_metrics_dataframe()
                    if rc_df is not None and not rc_df.empty:
                        for col in rc_df.columns:
                            if col.lower() in ('hole_id', 'depth_to'):
                                continue
                            display_name = f"{col} (RC Metrics)"
                            if display_name not in columns:
                                columns.append(display_name)
                except Exception as e:
                    self.logger.warning(f"Could not add RC metrics columns: {e}")

            # Build columns_info dict with data types
            columns_info = {}
            
            # Get a sample DataFrame by building from first source
            data_sources = self.data_coordinator.geological_store.get_data_sources()
            data_sample = None
            
            if data_sources:
                # Get first loaded source
                for source_name, source in data_sources.items():
                    if source.is_loaded and source.df is not None:
                        data_sample = source.df
                        break
            
            if data_sample is not None and not data_sample.empty:
                for col in columns:
                    # Strip source suffix if present
                    col_base = col.split(" (")[0].strip()
                    if col_base.lower() in data_sample.columns:
                        columns_info[col] = str(data_sample[col_base.lower()].dtype)
                    else:
                        columns_info[col] = 'object'  # Default
            
            # Create filter row
            filter_index = len(self.filter_rows)

            # Use filter_register_data if available (has RC metrics), otherwise build sample
            register_data_to_use = None
            if hasattr(self, 'filter_register_data') and self.filter_register_data is not None and not self.filter_register_data.empty:
                register_data_to_use = self.filter_register_data
            elif data_sample is not None and not data_sample.empty:
                register_data_to_use = data_sample
            else:
                # Build a minimal sample from available sources
                sample_rows = []
                for source_name, source in data_sources.items():
                    if source.is_loaded and source.df is not None:
                        sample_rows.append(source.df.head(100))
                if sample_rows:
                    register_data_to_use = pd.concat(sample_rows, ignore_index=True)

            filter_row = DynamicFilterRow(
                parent=self.filter_container,
                gui_manager=self.gui_manager,
                columns_info=columns_info,
                register_data=register_data_to_use if register_data_to_use is not None else pd.DataFrame(),
                on_remove_callback=lambda idx=filter_index: self._remove_dataset_filter_row(idx),
                index=filter_index,
                get_column_schema_callback=self._get_unique_values_for_column,
                get_unique_values_callback=self._get_unique_values_for_column
            )
            self.filter_rows.append(filter_row)
            
            # Enable clear button
            if hasattr(self, 'clear_dataset_filters_btn'):
                self.clear_dataset_filters_btn.config(state=tk.NORMAL)
            
            self.logger.info(f"Added filter row, total: {len(self.filter_rows)}")
            
        except Exception as e:
            self.logger.error(f"Error adding filter row: {e}", exc_info=True)
            messagebox.showerror("Error", f"Could not add filter row: {e}")
    
    def _get_unique_values_for_column(self, column: str) -> List[str]:
        """
        Get unique values for a column from the appropriate data source.
        
        Used by DynamicFilterRow to populate autocomplete dropdowns.
        """
        # Strip source suffix if present
        col_base = column.split(" (")[0].strip()
        col_lower = col_base.lower()
        
        # Per-reviewer columns (rev_username_classification, etc.)
        # Extract reviewer name from column (e.g., "rev_gsymonds_classification" -> "gsymonds")
        if col_lower.startswith('rev_'):
            # Parse: rev_{reviewer}_{field}
            parts = column.split('_')
            if len(parts) >= 3:
                reviewer = parts[1]  # e.g., "gsymonds"
                field = '_'.join(parts[2:])  # e.g., "classification"
                
                # Sample from JSONRegisterManager
                values = set()
                if hasattr(self, 'json_register_manager') and self.json_register_manager:
                    try:
                        df = self.json_register_manager.get_all_compartments_all_users()
                        if df is not None and not df.empty:
                            # Filter for this reviewer
                            reviewer_mask = df.get('Reviewed_By', df.get('reviewer', df.get('classified_by'))) == reviewer
                            reviewer_df = df[reviewer_mask]
                            
                            if not reviewer_df.empty:
                                if field == 'classification':
                                    col_vals = reviewer_df.get('classification', reviewer_df.get('Classification', pd.Series()))
                                elif field == 'comments':
                                    col_vals = reviewer_df.get('comments', reviewer_df.get('Comments', pd.Series()))
                                elif field == 'tags':
                                    return ['Yes', 'No']  # Tags are boolean-ish
                                else:
                                    col_vals = pd.Series()
                                
                                values = set(col_vals.dropna().unique())
                    except Exception as e:
                        self.logger.debug(f"Could not get values for {column}: {e}")
                
                return sorted([str(v) for v in values if str(v).strip()])[:200]
            return []
        
        # Register columns - get from all_images
        register_columns = {
            'hole_id', 'depth_from', 'depth_to', 'classification', 'classified_by',
            'consensus_classification', 'moisture_status', 'comments', 'agreement',
            'combined_hex', 'wet_hex', 'dry_hex'
        }
        
        # Special handling for classification-related columns - include all reviewers' values
        if col_lower in {'classification', 'all_classifications', 'consensus_classification'}:
            values = set()
            for img in self.all_images[:2000]:  # Sample for performance
                # Current user's classification
                if img.classification:
                    val = self._get_classification_string(img.classification)
                    if val and val.strip():
                        values.add(val)
                
                # Other reviewers' classifications
                if hasattr(img, 'other_reviews') and img.other_reviews:
                    for review in img.other_reviews:
                        if isinstance(review, dict):
                            classification = (
                                review.get("classification") or
                                review.get("Classification") or
                                review.get("Lithology")
                            )
                            if classification and str(classification).strip():
                                values.add(str(classification))
            return sorted(list(values))[:200]
        
        # Special handling for classified_by/all_reviewers - include all reviewer names
        if col_lower in {'classified_by', 'all_reviewers'}:
            values = set()
            for img in self.all_images[:2000]:
                # Current user
                if img.classified_by and str(img.classified_by).strip():
                    values.add(str(img.classified_by))
                
                # Other reviewers
                if hasattr(img, 'other_reviews') and img.other_reviews:
                    for review in img.other_reviews:
                        if isinstance(review, dict):
                            reviewer = (
                                review.get("Reviewed_By") or
                                review.get("_file_user") or
                                review.get("reviewer") or
                                review.get("classified_by")
                            )
                            if reviewer and str(reviewer).strip():
                                values.add(str(reviewer))
            return sorted(list(values))[:200]
        
        if col_lower in register_columns:
            values = set()
            for img in self.all_images[:2000]:  # Sample for performance
                val = getattr(img, col_lower, None)
                if val is not None and str(val).strip():
                    values.add(str(val))
            return sorted(list(values))[:200]
        
        # Tag columns
        if col_lower.startswith('tag_'):
            return ['True', 'False']
        
        # CSV columns (geology data store) - use new helper method that samples efficiently
        # This calls get_unique_values_for_column() which we added earlier
        unique_vals = self.get_unique_values_for_column(column, max_sample=2000)
        if unique_vals:
            # Convert to strings and return (already sorted by helper)
            return [str(v) for v in unique_vals if str(v).strip()][:200]
        
        return []

    def _remove_dataset_filter_row(self, index):
        """Remove a filter row by index"""
        if 0 <= index < len(self.filter_rows):
            filter_row = self.filter_rows[index]
            filter_row.destroy()
            self.filter_rows.pop(index)
            
            # Re-index remaining rows
            for i, row in enumerate(self.filter_rows):
                row.index = i
            
            # Disable clear button if no filters remain
            if len(self.filter_rows) == 0 and hasattr(self, 'clear_dataset_filters_btn'):
                self.clear_dataset_filters_btn.config(state=tk.DISABLED)
            
            self.logger.info(f"Removed filter row {index}, remaining: {len(self.filter_rows)}")
    
    def _clear_dataset_filters(self):
        """Clear all dataset filter rows"""
        for filter_row in self.filter_rows:
            filter_row.destroy()
        self.filter_rows.clear()
        
        # Disable clear button
        if hasattr(self, 'clear_dataset_filters_btn'):
            self.clear_dataset_filters_btn.config(state=tk.DISABLED)
        
        # Reset to unfiltered data
        self.displayed_images = self.all_images.copy()
        
        # Clear active filters display
        self.update_active_filters_display([])
        
        # Update grid
        if self.grid_canvas:
            self.grid_canvas.load_images(self.displayed_images)
        
        self._update_statistics()
        self._update_status("All dataset filters cleared - showing full dataset")
        
        self.logger.info("Cleared all dataset filters")
    
    def _apply_dataset_filters(self):
        """Apply the current filter configuration to the dataset using store queries"""
        if not self.filter_rows:
            self._update_status("No filters to apply")
            return
        
        # Get filter logic
        logic = self.filter_logic_var.get() if hasattr(self, 'filter_logic_var') else "AND"
        
        # Build filter descriptions for display
        filter_descriptions = []
        csv_filters = []
        register_filters = []
        
        # Register columns that come from register store
        register_columns = {'classification', 'classified_date', 'moisture_status', 
                          'wet_hex', 'dry_hex', 'combined_hex', 'comments'}
        
        for row in self.filter_rows:
            config = row.get_filter_config()
            if config["column"] and config["operator"] and config["value"]:
                # Build description
                desc = f"{config['column']} {config['operator']} {config['value']}"
                if config.get("value2"):
                    desc += f" and {config['value2']}"
                filter_descriptions.append(desc)
                
                # Split into CSV vs register filters
                col_lower = config["column"].lower().split(" (")[0].strip()
                
                if col_lower in register_columns or col_lower.startswith("tag_"):
                    register_filters.append(config)
                else:
                    csv_filters.append(config)
        
        if not filter_descriptions:
            self._update_status("No valid filters to apply")
            return
        
        self.logger.info(f"Applying {len(filter_descriptions)} filters with {logic} logic")
        self.logger.debug(f"Filter split: {len(csv_filters)} CSV, {len(register_filters)} register")
        
        try:
            # Build image key lookup
            image_key_to_img = {}
            for img in self.all_images:
                key = (img.hole_id, img.depth_to)
                image_key_to_img[key] = img
            
            # Start with all image keys as candidates
            matching_keys = set(image_key_to_img.keys())
            
            # Query GeologicalStore for CSV filters
            if csv_filters and self.data_coordinator and self.data_coordinator.geological_store.is_loaded:
                query_filters = [
                    {
                        'column': cfg["column"].split(" (")[0].strip(),
                        'operator': cfg["operator"],
                        'value': cfg["value"],
                        'value2': cfg.get("value2")
                    }
                    for cfg in csv_filters
                ]
                
                csv_keys = self.data_coordinator.geological_store.query(query_filters)
                matching_keys &= csv_keys
                
                self.logger.info(f"CSV query: {len(csv_keys)} matches")
            
            # Query RegisterStore for register filters
            if register_filters and self.data_coordinator and self.data_coordinator.register_store:
                query_filters = [
                    {
                        'column': cfg["column"].split(" (")[0].strip(),
                        'operator': cfg["operator"],
                        'value': cfg["value"],
                        'value2': cfg.get("value2")
                    }
                    for cfg in register_filters
                ]
                
                register_keys = self.data_coordinator.register_store.query(query_filters)
                matching_keys &= register_keys
                
                self.logger.info(f"Register query: {len(register_keys)} matches")
            
            # Build result list from matching keys
            # Preserve the order from all_images (which may be user-sorted)
            matched_images = []
            for img in self.all_images:
                key = (img.hole_id.upper(), float(img.depth_to))
                if key in matching_keys:
                    matched_images.append(img)
            
            self.logger.info(f"Filter result: {len(matched_images)}/{len(image_key_to_img)} images match")
            
            # Update displayed images
            self.displayed_images = matched_images
            
            # Sort at end of filter pipeline
            self._sort_displayed_images(trigger_source="_apply_dataset_filters")
            
            # Update active filters display
            self.update_active_filters_display(filter_descriptions)
            
            # Reload grid
            if self.grid_canvas:
                self.grid_canvas.load_images(self.displayed_images)
            
            # Update UI
            self._update_statistics()
            self._update_status(
                f"Dataset filters applied: {len(matched_images)} images match "
                f"({len(filter_descriptions)} filter{'s' if len(filter_descriptions) > 1 else ''})"
            )
            
        except Exception as e:
            self.logger.error(f"Error applying dataset filters: {e}", exc_info=True)
            messagebox.showerror("Filter Error", f"Could not apply filters: {e}")
            self._update_status("Error applying filters")

    def _load_column_data_for_filter(self, column_name):
        """Load actual data for a column when selected in a filter row (for lazy-loaded CSV columns)
        
        Returns:
            Updated register_data DataFrame, or None if no update needed
        """
        # Check if this is a CSV column with lazy-loaded data
        if not self.all_images:
            return None
        
        # Check if column exists but has NaN values (lazy loaded)
        if hasattr(self, 'filter_register_data') and self.filter_register_data is not None:
            if column_name in self.filter_register_data.columns:
                # Check if column is all NaN (lazy loaded)
                if self.filter_register_data[column_name].isna().all():
                    self.logger.debug(f"Loading lazy data for filter column: {column_name}")
                    
                    # Extract base column name and source (if present)
                    if " (" in column_name and column_name.endswith(")"):
                        base_col = column_name.split(" (")[0].strip()
                        source = column_name.split(" (")[1].rstrip(")")
                    else:
                        base_col = column_name
                        source = None
                    
                    # Load data for all images
                    column_values = []
                    for img in self.all_images:
                        value = None
                        
                        # Try to get from data_coordinator
                        if self.data_coordinator and self.data_coordinator.geological_store.is_loaded:
                            try:
                                key = (img.hole_id.upper(), int(img.depth_to))
                                value = self.data_coordinator.geological_store.get_value(key, base_col, source)
                            except Exception as e:
                                self.logger.debug(f"Error getting value for {key}, {base_col}: {e}")
                        
                        # Fallback to csv_data cache if available
                        if value is None and img.csv_data:
                            value = img.csv_data.get(base_col)
                        
                        column_values.append(value if value is not None else np.nan)
                    
                    # Update the register_data DataFrame
                    self.filter_register_data[column_name] = column_values
                    self.logger.debug(f"Loaded {len([v for v in column_values if pd.notna(v)])} non-null values for {column_name}")
                    
                    # Update all filter rows with the new data
                    for row in self.filter_rows:
                        row.register_data = self.filter_register_data
                        row._cache_unique_values(column_name)
                    
                    return self.filter_register_data
        
        return None

    def _get_column_schema_for_filter(self, column_name, source_name=None):
        """Get schema information for a column from data_coordinator
        
        Args:
            column_name: Column name (without source suffix)
            source_name: Optional source name
            
        Returns:
            Schema object or None
        """
        if not self.data_coordinator or not self.data_coordinator.geological_store.is_loaded:
            return None
        
        try:
            # Search through all sources if source not specified
            if source_name:
                sources_to_check = [source_name]
            else:
                sources_to_check = list(self.data_coordinator.geological_store._sources.keys())
            
            for src in sources_to_check:
                indexed_source = self.data_coordinator.geological_store._sources.get(src)
                if indexed_source and indexed_source.schema:
                    # Look for column in schema (case-insensitive)
                    for schema_col_name, col_schema in indexed_source.schema.columns.items():
                        if schema_col_name.lower() == column_name.lower():
                            return col_schema
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Error getting schema for {column_name}: {e}")
            return None

    def _reload_grid(self):
        """Reload the grid canvas with current displayed_images"""
        # Apply moisture preference before loading
        self._apply_moisture_preference()

        if self.grid_canvas:
            self.grid_canvas.load_images(self.displayed_images)

        self._update_statistics()

        self.logger.debug(f"Grid reloaded with {len(self.displayed_images)} images")

    def apply_interval_filter(self, active_intervals: List[Tuple[str, float, float]], filter_descriptions: List[str]):
        """Apply interval-based filtering from scatter window
        
        Args:
            active_intervals: List of (hole_id, depth_from, depth_to) tuples that passed filters
            filter_descriptions: List of human-readable filter descriptions for display
        """
        self.logger.info(f"Applying interval filter: {len(active_intervals)} active intervals")
        
        # Update active filters display
        self.update_active_filters_display(filter_descriptions)
        
        if not active_intervals:
            # No filters - show all images
            self.displayed_images = self.all_images.copy()
        else:
            # Convert intervals to set for fast lookup
            interval_set = set(active_intervals)
            
            # Filter images to only those matching active intervals
            self.displayed_images = [
                img for img in self.all_images
                if (img.hole_id, img.depth_from, img.depth_to) in interval_set
            ]
            
            self.logger.info(f"  Filtered to {len(self.displayed_images)} images from {len(self.all_images)} total")
        
        # Now apply Show Classified/Unclassified/All/Conflicts filters to this filtered set
        self._apply_classification_visibility_filter()
        
        # Apply moisture preference filtering
        self._apply_moisture_preference()

        # Update grid
        self._update_grid()
        self._update_stats()

    def _apply_moisture_preference(self):
        """
        Filter displayed images based on wet/dry moisture preference setting.

        Preferences:
        - "both": Show all images (current behavior)
        - "wet_preferred": Show wet if available, else show dry
        - "dry_preferred": Show dry if available, else show wet
        """
        pref = self.config_manager.get("moisture_preference", "both")

        if pref == "both":
            # No filtering needed
            return

        self.logger.debug(f"Applying moisture preference: {pref}")

        # Group images by interval (hole_id, depth_from, depth_to)
        intervals = defaultdict(list)
        for img in self.displayed_images:
            key = (img.hole_id, img.depth_from, img.depth_to)
            intervals[key].append(img)

        result = []
        for key, interval_images in intervals.items():
            wet_images = [i for i in interval_images if i.moisture_status == "Wet"]
            dry_images = [i for i in interval_images if i.moisture_status == "Dry"]
            other_images = [i for i in interval_images if i.moisture_status not in ("Wet", "Dry")]

            if pref == "wet_preferred":
                # Prefer wet, fallback to dry or other
                if wet_images:
                    result.extend(wet_images)
                elif dry_images:
                    result.extend(dry_images)
                else:
                    result.extend(other_images)
            elif pref == "dry_preferred":
                # Prefer dry, fallback to wet or other
                if dry_images:
                    result.extend(dry_images)
                elif wet_images:
                    result.extend(wet_images)
                else:
                    result.extend(other_images)

        before_count = len(self.displayed_images)
        self.displayed_images = result
        self.logger.debug(f"Moisture preference applied: {before_count} -> {len(result)} images")

    # def _apply_classification_visibility_filter(self, images: List = None) -> List:
    #     """
    #     Apply Show Classified/Unclassified/All/Conflicts filter.
        
    #     Args:
    #         images: List to filter (uses self.displayed_images if None)
            
    #     Returns:
    #         Filtered list of images
    #     """
    #     if images is None:
    #         images = self.displayed_images
        
    #     # Get current filter mode
    #     mode = self.filter_mode.get() if hasattr(self, 'filter_mode') else "all"
        
    #     if mode == "unclassified":
    #         # Show only unclassified images
    #         filtered = [
    #             img for img in images
    #             if not img.classification or 
    #                str(img.classification) in ('', 'UNASSIGNED', 'ClassificationCategory.UNASSIGNED')
    #         ]
    #         self.logger.debug(f"Unclassified filter: {len(filtered)}/{len(images)}")
    #     elif mode == "classified":
    #         # Show only classified images
    #         filtered = [
    #             img for img in images
    #             if img.classification and 
    #                str(img.classification) not in ('', 'UNASSIGNED', 'ClassificationCategory.UNASSIGNED')
    #         ]
    #         self.logger.debug(f"Classified filter: {len(filtered)}/{len(images)}")
    #     elif mode == "disagreements":
    #         # Show only images with review conflicts
    #         filtered = [
    #             img for img in images
    #             if hasattr(img, 'other_reviews') and img.other_reviews and
    #                any(r.get('classification') != str(img.classification) 
    #                    for r in img.other_reviews.values() if r.get('classification'))
    #         ]
    #         self.logger.debug(f"Conflicts filter: {len(filtered)}/{len(images)}")
    #     else:
    #         # "all" - no additional filtering
    #         filtered = images
        
    #     return filtered

    # def _sort_displayed_images(self, trigger_source: str = "unknown"):
    #     """
    #     Sort displayed_images based on current sort settings.
    #     Called at end of filter pipeline - sorts smallest dataset possible.
    #     """
    #     if not self.displayed_images:
    #         self.logger.debug(f"[SORT] No images to sort (trigger: {trigger_source})")
    #         return
        
    #     sort_column = self.sort_column_var.get() if hasattr(self, 'sort_column_var') else "depth_to"
    #     sort_order = self.sort_order_var.get() if hasattr(self, 'sort_order_var') else "asc"
    #     reverse = (sort_order == "desc")
        
    #     self.logger.info(f"[SORT] Sorting {len(self.displayed_images)} images by {sort_column} ({sort_order}) | trigger: {trigger_source}")
        
    #     # Log first 3 BEFORE
    #     self.logger.info(f"[SORT] BEFORE - first 3: {[(img.hole_id, img.depth_to) for img in self.displayed_images[:3]]}")
        
    #     try:
    #         if sort_column == "Average Hex Colour":
    #             def get_hex_sort_value(img):
    #                 key = (img.hole_id, img.depth_from, img.depth_to)
    #                 combined_hex = self.hex_color_cache.get(key, "")
    #                 if combined_hex:
    #                     h, s, l = self._hex_to_hsl(combined_hex)
    #                     color_mode = self.color_sort_mode.get() if hasattr(self, "color_sort_mode") else "hue"
    #                     if color_mode == "lightness":
    #                         return (l, h, s)
    #                     elif color_mode == "saturation":
    #                         return (s, h, l)
    #                     return (h, s, l)
    #                 return (float("inf"), 0, 0) if not reverse else (float("-inf"), 0, 0)
    #             self.displayed_images.sort(key=get_hex_sort_value, reverse=reverse)
                
    #         elif sort_column in ["hole_id", "depth_from", "depth_to"]:
    #             self.displayed_images.sort(key=lambda img: (getattr(img, sort_column), img.hole_id, img.depth_to), reverse=reverse)
                
    #         elif sort_column == "classification":
    #             show_others = self.show_others_var.get() if hasattr(self, "show_others_var") else False
    #             def get_class_key(img):
    #                 if show_others:
    #                     consensus = self._get_consensus_classification(img)
    #                     return (consensus if consensus else "Unassigned", img.hole_id, img.depth_to)
    #                 return (str(img.classification), img.hole_id, img.depth_to)
    #             self.displayed_images.sort(key=get_class_key, reverse=reverse)
                
    #         elif sort_column in ["consensus_classification", "review_count", "agreement"]:
    #             def get_review_key(img):
    #                 metadata = self._compute_review_metadata(img)
    #                 return (metadata.get(sort_column, ""), img.hole_id, img.depth_to)
    #             self.displayed_images.sort(key=get_review_key, reverse=reverse)
                
    #         elif sort_column.startswith("tag_"):
    #             tag_id = sort_column[4:]
    #             def get_tag_key(img):
    #                 metadata = self._compute_review_metadata(img)
    #                 return (tag_id in metadata.get("tags", set()), img.hole_id, img.depth_to)
    #             self.displayed_images.sort(key=get_tag_key, reverse=reverse)
                
    #         else:
    #             # CSV column
    #             def get_csv_sort_value(img):
    #                 if img.csv_data and sort_column in img.csv_data:
    #                     val = img.csv_data[sort_column]
    #                     if val is None or (isinstance(val, float) and pd.isna(val)):
    #                         return (float("inf") if not reverse else float("-inf"), img.hole_id, img.depth_to)
    #                     return (val, img.hole_id, img.depth_to)
    #                 return (float("inf") if not reverse else float("-inf"), img.hole_id, img.depth_to)
    #             self.displayed_images.sort(key=get_csv_sort_value, reverse=reverse)
            
    #         # Log first 3 AFTER
    #         self.logger.info(f"[SORT] AFTER - first 3: {[(img.hole_id, img.depth_to) for img in self.displayed_images[:3]]}")
                
    #     except Exception as e:
    #         self.logger.error(f"[SORT] Error sorting: {e}", exc_info=True)

    def _set_filter_mode(self, mode: str):
        """
        Set the filter mode and apply all filters.
        
        Modes:
            "all" - Show all images
            "disagreements" - Show images where reviewers disagree (always uses consensus)
            "classified_all" - Classified by anyone (uses consensus)
            "classified_mine" - Classified by current user only
            "classified" - Legacy: same as classified_mine
            "unclassified_all" - Not classified by anyone (uses consensus)
            "unclassified_mine" - Not classified by current user
            "unclassified" - Legacy: same as unclassified_mine
        """
        # Map new modes to internal mode + show_others setting
        mode_mapping = {
            "all": ("all", True),  # Show all images with all users' reviews visible
            "disagreements": ("disagreements", True),
            "classified_all": ("classified", True),
            "classified_mine": ("classified", False),
            "classified": ("classified", False),  # Legacy
            "unclassified_all": ("unclassified", True),
            "unclassified_mine": ("unclassified", False),
            "unclassified": ("unclassified", False),  # Legacy
        }
        
        internal_mode, use_others = mode_mapping.get(mode, ("all", False))
        
        self.logger.info(f"[FILTER_MODE] Setting mode={mode} → internal={internal_mode}, use_others={use_others}")
        
        self.filter_mode.set(internal_mode)
        
        # Always update show_others and populate/clear other_reviews data
        current_show_others = self.show_others_var.get() if hasattr(self, 'show_others_var') else False
        if current_show_others != use_others:
            self.show_others_var.set(use_others)
        
        # Always call to ensure img.other_reviews is properly populated/cleared
        self._toggle_others_reviews()
        
        # Update exclude_var for backward compatibility
        self.exclude_var.set(internal_mode == "unclassified")

        # Build filter descriptions from dynamic filter rows
        filter_descriptions = self._get_dynamic_filter_descriptions()
        
        # Add scatter selection info if active
        if hasattr(self, '_scatter_selection_info') and self._scatter_selection_info:
            filter_descriptions.append(self._scatter_selection_info)
        
        self.update_active_filters_display(filter_descriptions)

        # Apply all filters: dynamic rows + classification mode
        self._apply_filters(force=True)

        # Update scatter plot filtered data if window is open
        self._update_scatter_filtered_data()

    def _filter_by_validation_flags(self, flagged_keys: Set[Tuple[str, float]]):
        """
        Filter the review dialog to show only flagged items from QAQC validation.

        Args:
            flagged_keys: Set of (hole_id, depth_to) tuples that have validation flags
        """
        if not flagged_keys:
            messagebox.showinfo("No Flags", "No pending validation flags to filter by.")
            return

        self.logger.info(f"Filtering by {len(flagged_keys)} validation flags")

        # Filter displayed images to only those in flagged_keys
        filtered_images = []
        for img in self.all_images:
            key = (img.hole_id, img.depth_to)
            if key in flagged_keys:
                filtered_images.append(img)

        if not filtered_images:
            messagebox.showinfo("No Matches", "No images match the validation flags.")
            return

        # Update displayed images
        self.displayed_images = filtered_images

        # Update filter display
        filter_descriptions = [f"Validation Flags: {len(flagged_keys)} flagged items"]
        self.update_active_filters_display(filter_descriptions)

        # Update the grid
        self._update_grid()
        self._update_stats()

        self.logger.info(f"Filtered to {len(filtered_images)} flagged images")

    def _get_dynamic_filter_descriptions(self) -> List[str]:
        """Get descriptions of active dynamic filter rows"""
        descriptions = []
        for row in self.filter_rows:
            config = row.get_filter_config()
            if config["column"] and config["operator"] and config["value"]:
                desc = f"{config['column']} {config['operator']} {config['value']}"
                if config.get("value2"):
                    desc += f" and {config['value2']}"
                descriptions.append(desc)
        return descriptions
    
    def _update_scatter_filtered_data(self):
        """Update the scatter plot's filtered_data based on current filter rows"""
        if not hasattr(self, 'grid_canvas') or not self.grid_canvas:
            return
        if not hasattr(self.grid_canvas, 'zoom_preview') or not self.grid_canvas.zoom_preview:
            return
        
        scatter_window = self.grid_canvas.zoom_preview
        
        # Build filtered DataFrame based on dynamic filter rows
        if scatter_window.current_data is not None and not scatter_window.current_data.empty:
            filtered_df = scatter_window.current_data.copy()
            
            # Apply each filter row
            for row in self.filter_rows:
                config = row.get_filter_config()
                if not config["column"] or not config["operator"]:
                    continue
                
                col = config["column"]
                op = config["operator"]
                val = config["value"]
                val2 = config.get("value2")
                
                if col not in filtered_df.columns:
                    continue
                
                try:
                    if op == "=":
                        filtered_df = filtered_df[filtered_df[col] == val]
                    elif op == "!=":
                        filtered_df = filtered_df[filtered_df[col] != val]
                    elif op == ">":
                        filtered_df = filtered_df[filtered_df[col] > float(val)]
                    elif op == ">=":
                        filtered_df = filtered_df[filtered_df[col] >= float(val)]
                    elif op == "<":
                        filtered_df = filtered_df[filtered_df[col] < float(val)]
                    elif op == "<=":
                        filtered_df = filtered_df[filtered_df[col] <= float(val)]
                    elif op == "between" and val2:
                        filtered_df = filtered_df[(filtered_df[col] >= float(val)) & (filtered_df[col] <= float(val2))]
                    elif op == "contains":
                        filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(str(val), case=False, na=False)]
                    elif op == "starts with":
                        filtered_df = filtered_df[filtered_df[col].astype(str).str.startswith(str(val), na=False)]
                    elif op == "is null":
                        filtered_df = filtered_df[filtered_df[col].isna()]
                    elif op == "is not null":
                        filtered_df = filtered_df[filtered_df[col].notna()]
                except (ValueError, TypeError) as e:
                    self.logger.debug(f"Filter error for {col} {op} {val}: {e}")
                    continue
            
            # Update the scatter window's filtered data
            scatter_window.filtered_data = filtered_df
            self.logger.debug(f"Updated scatter filtered_data: {len(filtered_df)} rows (from {len(scatter_window.current_data)})")
            
            # Refresh the plot if in filtered mode
            if scatter_window.plot_mode == "filtered":
                scatter_window._update_scatter()

    def _create_left_panel(self, parent):
        """Create narrow left panel with filters and stats"""
        # Tags section - moved from toolbar (placed first, under classification mode conceptually)
        tag_frame = ttk.LabelFrame(parent, text="Tags", padding=5)
        tag_frame.pack(fill=tk.X, padx=5, pady=5)

        # Store reference to tag_frame for rebuilding
        self.tag_frame = tag_frame

        # Build tag buttons dynamically
        self._build_tag_buttons()

        # Comment entry
        comment_frame = ttk.LabelFrame(parent, text="Add Comment", padding=5)
        comment_frame.pack(fill=tk.X, padx=5, pady=5)

        # Create themed entry frame
        entry_frame = tk.Frame(
            comment_frame,
            bg=self.gui_manager.theme_colors["field_bg"],
            highlightbackground=self.gui_manager.theme_colors["field_border"],
            highlightthickness=1,
        )
        entry_frame.pack(fill=tk.BOTH, expand=True)

        self.comment_entry = tk.Entry(
            entry_frame,
            font=("Arial", 10),
            bg=self.gui_manager.theme_colors["field_bg"],
            fg=self.gui_manager.theme_colors["text"],
            bd=0,
            insertbackground=self.gui_manager.theme_colors["text"],
        )
        self.comment_entry.pack(padx=5, pady=3, fill=tk.X)
        self.comment_entry.bind("<Return>", lambda e: self._apply_comment())

        ModernButton(
            comment_frame,
            text="Apply to Selected",
            command=self._apply_comment,
            color=self.gui_manager.theme_colors["accent_blue"],
            theme_colors=self.gui_manager.theme_colors,
        ).pack(fill=tk.X, pady=2)

        # Statistics - collapsible
        self.stats_collapsible = CollapsibleFrame(
            parent,
            text="Statistics",
            expanded=False,  # Start collapsed
            bg=self.gui_manager.theme_colors["background"],
            fg=self.gui_manager.theme_colors["text"],
            title_bg=self.gui_manager.theme_colors["secondary_bg"],
            title_fg=self.gui_manager.theme_colors["text"],
            content_bg=self.gui_manager.theme_colors["background"],
            border_color=self.gui_manager.theme_colors.get("border", "#3f3f3f"),
            arrow_color=self.gui_manager.theme_colors.get("accent_blue", "#3a7ca5")
        )
        self.stats_collapsible.pack(fill=tk.X, padx=5, pady=5)

        self.stats_label = ttk.Label(
            self.stats_collapsible.content_frame, text="No images loaded", font=("Arial", 9)
        )
        self.stats_label.pack(fill=tk.X)

        # Display Mode - collapsible frame showing current mode
        
        
        self.hole_nav_frame = CollapsibleFrame(
            parent, 
            text="Mode: All Images",  # Will be updated dynamically
            expanded=False,  # Start collapsed
            bg=self.gui_manager.theme_colors["background"],
            fg=self.gui_manager.theme_colors["text"],
            title_bg=self.gui_manager.theme_colors["secondary_bg"],
            title_fg=self.gui_manager.theme_colors["text"],
            content_bg=self.gui_manager.theme_colors["background"],
            border_color=self.gui_manager.theme_colors.get("border", "#3f3f3f"),
            arrow_color=self.gui_manager.theme_colors.get("accent_blue", "#3a7ca5")
        )
        self.hole_nav_frame.pack(fill=tk.X, padx=5, pady=2)

        self.display_mode_var = tk.StringVar(
            value="all_images"
        )  # Default to all images

        # Create radio button frame with custom styling
        modes_frame = ttk.Frame(self.hole_nav_frame.content_frame)
        modes_frame.pack(fill=tk.X, pady=(0, 5))

        # Define display modes (removed drillhole_correlation)
        display_modes = [
            ("hole_by_hole", "Hole-By-Hole"),
            ("all_images", "All Images"),
            ("all_intervals", "All Intervals"),
        ]

        for mode_value, mode_text in display_modes:
            frame = ttk.Frame(modes_frame)
            frame.pack(fill=tk.X, pady=2)

            # Create custom styled radio button
            radio = tk.Radiobutton(
                frame,
                text=mode_text,
                variable=self.display_mode_var,
                value=mode_value,
                command=self._on_display_mode_changed,
                bg=self.gui_manager.theme_colors["background"],
                fg=self.gui_manager.theme_colors["text"],
                activebackground=self.gui_manager.theme_colors.get(
                    "hover", self.gui_manager.theme_colors["secondary_bg"]
                ),
                activeforeground=self.gui_manager.theme_colors["text"],
                selectcolor=self.gui_manager.theme_colors["accent_green"],
                font=("Arial", 9),
                cursor="hand2",
                pady=2,
            )
            radio.pack(fill=tk.X)

        # Show other users' reviews checkbox moved to filters frame

        # Navigation buttons container - only show in hole-by-hole mode
        self.nav_buttons_frame = ttk.Frame(self.hole_nav_frame.content_frame)
        # Don't pack yet - will be shown/hidden based on mode
        
        # Navigation buttons
        self.prev_hole_btn = ModernButton(
            self.nav_buttons_frame,
            text="â—€ Previous",
            command=self._previous_hole,
            color=self.gui_manager.theme_colors["accent_blue"],
            theme_colors=self.gui_manager.theme_colors,
        )
        self.prev_hole_btn.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)

        self.next_hole_btn = ModernButton(
            self.nav_buttons_frame,
            text="Next â–¶",
            command=self._next_hole,
            color=self.gui_manager.theme_colors["accent_blue"],
            theme_colors=self.gui_manager.theme_colors,
        )
        self.next_hole_btn.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)

        # Auto-advance checkbox - only visible in hole-by-hole mode
        self.auto_advance_var = tk.BooleanVar(value=True)
        self.auto_advance_check = ttk.Checkbutton(
            self.nav_buttons_frame,
            text="Auto-advance on completion",
            variable=self.auto_advance_var,
            command=self._toggle_auto_advance,
        )
        # Don't pack yet - will be shown with nav buttons in hole-by-hole mode

        # Filters
        filter_frame = ttk.LabelFrame(parent, text="Filters", padding=5)
        filter_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Track filter mode
        self.filter_mode = tk.StringVar(value="all")
        self.exclude_var = tk.BooleanVar(value=False)  # Keep for backward compatibility

        # Filter Dataset button at top - most prominent
        self.filter_dataset_btn = self.gui_manager.create_modern_button(
            filter_frame,
            text="📊 Filter Dataset (Scatter Plot)",
            command=self._toggle_scatter_filter,
            color=self.gui_manager.theme_colors["accent_green"],  # Green for prominence
        )
        self.filter_dataset_btn.pack(fill=tk.X, pady=(0, 10))

        # === Dynamic Filter Rows Section ===
        dynamic_filter_section = ttk.LabelFrame(filter_frame, text="Dataset Filters", padding=5)
        dynamic_filter_section.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Filter logic selector (AND/OR)
        logic_frame = ttk.Frame(dynamic_filter_section)
        logic_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(
            logic_frame,
            text="Combine filters with:",
            style="Content.TLabel"
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        self.filter_logic_var = tk.StringVar(value="AND")
        logic_dropdown = tk.OptionMenu(
            logic_frame,
            self.filter_logic_var,
            "AND",
            "OR"
        )
        self.gui_manager.style_dropdown(logic_dropdown)
        logic_dropdown.config(width=6)
        logic_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Separator(dynamic_filter_section, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        # Scrollable container for filter rows
        filter_scroll_frame = ttk.Frame(dynamic_filter_section)
        filter_scroll_frame.pack(fill=tk.BOTH, expand=True)
        
        self.filter_canvas = tk.Canvas(
            filter_scroll_frame,
            bg=self.gui_manager.theme_colors["background"],
            highlightthickness=0,
            height=200  # Fixed height for filter rows area
        )
        filter_scrollbar = ttk.Scrollbar(filter_scroll_frame, command=self.filter_canvas.yview)
        self.filter_canvas.configure(yscrollcommand=filter_scrollbar.set)
        
        self.filter_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        filter_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.filter_container = ttk.Frame(self.filter_canvas)
        self.filter_canvas.create_window(0, 0, anchor="nw", window=self.filter_container)
        
        def on_filter_configure(event):
            self.filter_canvas.configure(scrollregion=self.filter_canvas.bbox("all"))
        
        self.filter_container.bind("<Configure>", on_filter_configure)
        
        ttk.Separator(dynamic_filter_section, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        # Filter action buttons - compact single row
        button_frame = ttk.Frame(dynamic_filter_section)
        button_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Single row for Add/Clear/Apply
        action_row = ttk.Frame(button_frame)
        action_row.pack(fill=tk.X, pady=1)
        
        add_filter_btn = ModernButton(
            action_row,
            text="+ Add",
            command=self._add_dataset_filter_row,
            color=self.gui_manager.theme_colors["accent_green"],
            theme_colors=self.gui_manager.theme_colors,
            width=6,
        )
        add_filter_btn.pack(side=tk.LEFT, padx=(0, 2), expand=True, fill=tk.X)
        
        self.clear_dataset_filters_btn = ModernButton(
            action_row,
            text="Clear Filters",
            command=self._clear_dataset_filters,
            color=self.gui_manager.theme_colors.get("accent_orange", "#ff9800"),
            theme_colors=self.gui_manager.theme_colors,
            width=10,
        )
        self.clear_dataset_filters_btn.pack(side=tk.LEFT, padx=(2, 0), expand=True, fill=tk.X)
        self.clear_dataset_filters_btn.config(state=tk.DISABLED)
        
        # Note: Apply button removed - classification filter buttons (All/Classified/Unclassified/Conflicts)
        # now act as the apply mechanism and will apply both dynamic filter rows AND classification filters
        
        # Quick show mode buttons - 3x2 grid with clear labeling
        quick_frame = ttk.LabelFrame(button_frame, text="Quick Show", padding=3)
        quick_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Row 0: Show All | Conflicts
        quick_row0 = ttk.Frame(quick_frame)
        quick_row0.pack(fill=tk.X, pady=1)
        
        self.show_all_btn = ModernButton(
            quick_row0,
            text="Show All",
            command=lambda: self._set_filter_mode("all"),
            color=self.gui_manager.theme_colors["accent_blue"],
            theme_colors=self.gui_manager.theme_colors,
            width=10,
        )
        self.show_all_btn.pack(side=tk.LEFT, padx=(0, 2), expand=True, fill=tk.X)
        
        self.show_conflicts_btn = ModernButton(
            quick_row0,
            text="Conflicts",
            command=lambda: self._set_filter_mode("disagreements"),
            color=self.gui_manager.theme_colors["accent_red"],
            theme_colors=self.gui_manager.theme_colors,
            width=10,
        )
        self.show_conflicts_btn.pack(side=tk.LEFT, padx=(2, 0), expand=True, fill=tk.X)
        
        # Row 1: All Classified | My Classified
        quick_row1 = ttk.Frame(quick_frame)
        quick_row1.pack(fill=tk.X, pady=1)
        
        self.show_all_classified_btn = ModernButton(
            quick_row1,
            text="All Classified",
            command=lambda: self._set_filter_mode("classified_all"),
            color=self.gui_manager.theme_colors["accent_green"],
            theme_colors=self.gui_manager.theme_colors,
            width=10,
        )
        self.show_all_classified_btn.pack(side=tk.LEFT, padx=(0, 2), expand=True, fill=tk.X)
        
        self.show_my_classified_btn = ModernButton(
            quick_row1,
            text="My Classified",
            command=lambda: self._set_filter_mode("classified_mine"),
            color=self.gui_manager.theme_colors["accent_green"],
            theme_colors=self.gui_manager.theme_colors,
            width=10,
        )
        self.show_my_classified_btn.pack(side=tk.LEFT, padx=(2, 0), expand=True, fill=tk.X)
        
        # Row 2: All Unclassified | My Unclassified
        quick_row2 = ttk.Frame(quick_frame)
        quick_row2.pack(fill=tk.X, pady=1)
        
        self.show_all_unclassified_btn = ModernButton(
            quick_row2,
            text="All Unclassified",
            command=lambda: self._set_filter_mode("unclassified_all"),
            color=self.gui_manager.theme_colors.get("accent_orange", "#ff9800"),
            theme_colors=self.gui_manager.theme_colors,
            width=10,
        )
        self.show_all_unclassified_btn.pack(side=tk.LEFT, padx=(0, 2), expand=True, fill=tk.X)
        
        self.show_my_unclassified_btn = ModernButton(
            quick_row2,
            text="My Unclassified",
            command=lambda: self._set_filter_mode("unclassified_mine"),
            color=self.gui_manager.theme_colors.get("accent_orange", "#ff9800"),
            theme_colors=self.gui_manager.theme_colors,
            width=10,
        )
        self.show_my_unclassified_btn.pack(side=tk.LEFT, padx=(2, 0), expand=True, fill=tk.X)

        # Add sorting controls below filter buttons - more compact
        sort_frame = ttk.LabelFrame(filter_frame, text="Sort By", padding=3)
        sort_frame.pack(fill=tk.X, pady=5)

        # Primary sort container
        sort_container = ttk.Frame(sort_frame)
        sort_container.pack(fill=tk.X, pady=2)
        
        # Sort column dropdown - includes register metadata columns
        sort_columns = [
            "Average Hex Colour", 
            "hole_id", 
            "depth_from", 
            "depth_to", 
            "classification",
            "consensus_classification",
            "review_count",
            "agreement",
            "hex_color",
            "combined_hex",
            "wet_hex",
            "dry_hex",
        ]
        
        # Add geological data columns from DataCoordinator
        if self.data_coordinator and self.data_coordinator.geological_store.is_loaded:
            available_cols = self.data_coordinator.geological_store.get_available_columns()
            for source_name, cols in available_cols.items():
                for col_name, col_type in cols:
                    if col_name.lower() not in [c.lower() for c in sort_columns]:
                        sort_columns.append(col_name)
        
        # Add tag columns to sort options
        if hasattr(self, 'item_manager') and self.item_manager:
            for tag_def in self.item_manager.get_all_tags():
                sort_columns.append(f"tag_{tag_def.id}")
        
        # Use searchable dropdown for sort column
        try:
            self.sort_dropdown = self.gui_manager.create_searchable_optionmenu(
                parent=sort_container,
                items=sort_columns,
                variable=self.sort_column_var,
                width=15,
                placeholder="Select column...",
                on_change=None,  # Don't auto-sort
                dropdown_mode="overlay",
            )
            self.sort_dropdown.pack(side=tk.LEFT, padx=(0, 3))
        except Exception as e:
            # Fallback to standard dropdown
            print(f"Failed to create searchable sort dropdown: {e}")
            self.sort_dropdown = tk.OptionMenu(
                sort_container,
                self.sort_column_var,
                *sort_columns,
            )
            self.gui_manager.style_dropdown(self.sort_dropdown)
            self.sort_dropdown.config(width=13)
            self.sort_dropdown.pack(side=tk.LEFT, padx=(0, 3))

        # Sort order dropdown - no auto-trigger
        sort_order_dropdown = tk.OptionMenu(
            sort_container,
            self.sort_order_var,
            "asc",
            "desc",
        )
        self.gui_manager.style_dropdown(sort_order_dropdown)
        sort_order_dropdown.config(width=6)
        sort_order_dropdown.pack(side=tk.LEFT)
        
        # Update menu labels to be more descriptive
        menu = sort_order_dropdown["menu"]
        menu.delete(0, "end")
        menu.add_command(label="→ Asc", command=lambda: self.sort_order_var.set("asc"))
        menu.add_command(label="↓ Desc", command=lambda: self.sort_order_var.set("desc"))

        # Secondary sort (then by)
        secondary_container = ttk.Frame(sort_frame)
        secondary_container.pack(fill=tk.X, pady=(5, 2))

        ttk.Label(
            secondary_container,
            text="Then by:",
            style="Content.TLabel",
            font=("Arial", 8)
        ).pack(side=tk.LEFT, padx=(0, 3))

        # Initialize secondary sort variables
        self.sort_column_var_secondary = tk.StringVar(value="")  # Empty means no secondary sort
        self.sort_order_var_secondary = tk.StringVar(value="asc")

        # Secondary sort column dropdown - use searchable
        sort_columns_with_none = ["(none)"] + sort_columns
        try:
            self.sort_dropdown_secondary = self.gui_manager.create_searchable_optionmenu(
                parent=secondary_container,
                items=sort_columns_with_none,
                variable=self.sort_column_var_secondary,
                width=13,
                placeholder="(none)",
                on_change=None,  # Don't auto-sort
                dropdown_mode="overlay",
            )
            self.sort_dropdown_secondary.pack(side=tk.LEFT, padx=(0, 3))
        except Exception as e:
            # Fallback to standard dropdown
            print(f"Failed to create searchable secondary sort dropdown: {e}")
            self.sort_dropdown_secondary = tk.OptionMenu(
                secondary_container,
                self.sort_column_var_secondary,
                *sort_columns_with_none,
            )
            self.gui_manager.style_dropdown(self.sort_dropdown_secondary)
            self.sort_dropdown_secondary.config(width=11)
            self.sort_dropdown_secondary.pack(side=tk.LEFT, padx=(0, 3))

        # Secondary sort order dropdown - no auto-trigger
        sort_order_dropdown_secondary = tk.OptionMenu(
            secondary_container,
            self.sort_order_var_secondary,
            "asc",
            "desc",
        )
        self.gui_manager.style_dropdown(sort_order_dropdown_secondary)
        sort_order_dropdown_secondary.config(width=5)
        sort_order_dropdown_secondary.pack(side=tk.LEFT)

        # Update secondary menu labels
        menu2 = sort_order_dropdown_secondary["menu"]
        menu2.delete(0, "end")
        menu2.add_command(label="→", command=lambda: self.sort_order_var_secondary.set("asc"))
        menu2.add_command(label="↓", command=lambda: self.sort_order_var_secondary.set("desc"))

        # Add Sort button to apply sorting
        ModernButton(
            secondary_container,
            text="Sort",
            command=self._apply_sorting,
            color=self.gui_manager.theme_colors["accent_blue"],
            theme_colors=self.gui_manager.theme_colors
        ).pack(side=tk.LEFT, padx=(10, 0))

        # Active Filters Summary (read-only display of filters from scatter window)
        summary_frame = ttk.LabelFrame(filter_frame, text="Active Filters", padding=5)
        summary_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Scrollable area for filter list
        filter_scroll = ttk.Frame(summary_frame)
        filter_scroll.pack(fill=tk.BOTH, expand=True)
        
        filter_canvas = tk.Canvas(
            filter_scroll,
            bg=self.gui_manager.theme_colors["background"],
            highlightthickness=0,
            height=150,
        )
        filter_scrollbar = ttk.Scrollbar(filter_scroll, command=filter_canvas.yview)
        filter_canvas.configure(yscrollcommand=filter_scrollbar.set)
        
        filter_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        filter_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.active_filters_container = ttk.Frame(filter_canvas)
        filter_canvas.create_window(0, 0, anchor="nw", window=self.active_filters_container)
        
        def on_filters_configure(event):
            filter_canvas.configure(scrollregion=filter_canvas.bbox("all"))
        
        self.active_filters_container.bind("<Configure>", on_filters_configure)
        
        # Initially show "No active filters"
        self.no_filters_label = ttk.Label(
            self.active_filters_container,
            text="No active filters\n(Click 'Filter Dataset' to add filters)",
            style="Content.TLabel",
            justify=tk.CENTER,
        )
        self.no_filters_label.pack(pady=20)
        
        # Internal variable for showing other users' reviews (controlled by Quick Show buttons)
        self.show_others_var = tk.BooleanVar(value=False)

    def _get_column_display_name(self, column_name: str) -> str:
        """Get user-friendly display name for a column"""
        if self.item_manager:
            return self.item_manager.get_column_display_name(column_name)
        return column_name

    def _toggle_others_reviews(self):
        """Toggle display of other users' reviews - populates img.other_reviews for ALL images"""
        show_others = self.show_others_var.get()

        if show_others and hasattr(self, "other_user_reviews"):
            # Apply other users' reviews to ALL images (not just displayed)
            populated_count = 0
            for img in self.all_images:
                key = (img.hole_id, img.depth_from, img.depth_to)
                if key in self.other_user_reviews:
                    # Store other reviews in image for display
                    img.other_reviews = self.other_user_reviews[key]
                    populated_count += 1

                    # Get consensus classification if multiple reviews
                    classifications = []
                    for review in img.other_reviews:
                        # Check multiple possible field names (including lowercase)
                        classification_value = None
                        for field_name in [
                            "classification",
                            "Classification",
                            "Lithology",
                            "Rock_Type",
                        ]:
                            if field_name in review and review[field_name]:
                                classification_value = review[field_name]
                                break
                        if classification_value:
                            classifications.append(classification_value)

                    # Store most common classification from others
                    if classifications:
                        most_common = Counter(classifications).most_common(1)[0][0]
                        img.other_classification = most_common
                        img.other_reviewers = [
                            r.get("Reviewed_By", "Unknown") for r in img.other_reviews
                        ]
                else:
                    # No other reviews for this image
                    img.other_reviews = None
                    img.other_classification = None
                    img.other_reviewers = None
            
            self.logger.info(f"Populated other_reviews for {populated_count} images")
        else:
            # Clear other reviews from ALL images
            for img in self.all_images:
                img.other_reviews = None
                img.other_classification = None
                img.other_reviewers = None
            
            self.logger.info("Cleared other_reviews from all images")

        # Note: Caller is responsible for calling _apply_filters() after this
        # to avoid duplicate filter calls

    def _create_right_panel(self, parent):
        """Create right panel with image grid"""
        self.grid_canvas = LithologyGridCanvas(parent, self.gui_manager.theme_colors)
        self.grid_canvas.dialog_ref = self  # Add reference to dialog

    def _create_status_bar(self, parent):
        """Create bottom status bar"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(5, 0))

        self.status_label = ttk.Label(status_frame, text="Ready", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Undo/Redo indicators
        ttk.Label(status_frame, text="Ctrl+Z: Undo | Ctrl+Y: Redo").pack(
            side=tk.RIGHT, padx=5
        )

    def _bump_classification_epoch(self):
        """Increment classification epoch to invalidate filter cache (PEP-8)"""
        if not hasattr(self, "_classification_epoch"):
            self._classification_epoch = 0
        self._classification_epoch += 1

    def _apply_filters(self, force: bool = False):
        """Apply filters using the FilterPipeline."""
        # Build criteria from current UI state
        criteria = create_filter_criteria_from_dialog(self)
        
        # DEBUG: Log filter row configs
        self.logger.info(f"[DEBUG] Dynamic filters: {criteria.dynamic_filters}")
        
        # Check if filters changed
        current_hash = criteria.get_hash()
        if not force and current_hash == self.last_filter_hash and self.displayed_images:
            self.logger.debug("Filters unchanged, skipping re-filter")
            return
        
        self.last_filter_hash = current_hash
        
        # Execute pipeline
        result = self.filter_pipeline.execute(self.all_images, criteria)
        
        # Update displayed images
        self.displayed_images = result.images
        
        # Unload images no longer displayed
        displayed_set = set(id(img) for img in self.displayed_images)
        for img in self.all_images:
            if id(img) not in displayed_set:
                img.unload_image()
        
        # Update UI
        self.update_active_filters_display(result.filter_descriptions)
        self.grid_canvas.load_images(self.displayed_images)
        self._update_statistics()
        
        # Status message
        status = f"Displaying {result.total_after} images"
        if result.csv_matches > 0:
            status += f" ({result.csv_matches} with CSV data)"
        self._update_status(status)
        
        self.logger.info(f"Filter applied: {result.total_before} → {result.total_after} in {result.execution_time_ms:.0f}ms")

    def _set_mode(self, mode: str):
        """Set classification mode"""
        self.grid_canvas.set_mode(mode)
        self._update_status(f"Mode: {mode.upper()}")

    def _apply_negative_tag_to_selected(self, tag_id: str):
        """Mark selected images as explicitly NOT matching the given tag.
        
        Adds 'not_<tag_id>' to the image's tags, which serves as a
        persistent 'reviewed and skipped' marker so the image won't
        appear as unreviewed for this tag in future sessions.
        """
        if not self.grid_canvas or not self.grid_canvas.selected_indices:
            self._update_status("Select images first, then Shift+key to skip-tag")
            return

        neg_tag = f"not_{tag_id}"
        tag_def = self.item_manager.get_tag(tag_id)
        label = tag_def.label if tag_def else tag_id

        count = 0
        for idx in self.grid_canvas.selected_indices:
            if idx < len(self.grid_canvas.displayed_images):
                img = self.grid_canvas.displayed_images[idx]
                if not hasattr(img, "tags") or img.tags is None:
                    img.tags = set()
                if neg_tag not in img.tags:
                    img.tags.add(neg_tag)
                    # Also propagate to all_images
                    for all_img in self.all_images:
                        if (all_img.hole_id == img.hole_id
                                and all_img.depth_to == img.depth_to
                                and all_img.filename == img.filename):
                            if not hasattr(all_img, "tags") or all_img.tags is None:
                                all_img.tags = set()
                            all_img.tags.add(neg_tag)
                            break
                    count += 1

        if count > 0:
            self.has_unsaved_changes = True
            self.grid_canvas.selected_indices.clear()
            self.grid_canvas._refresh_classifications()
            self._update_status(f"Marked {count} images as NOT '{label}'")

    def _apply_comment(self):
        """Apply comment to last selected images"""
        comment = self.comment_entry.get().strip()
        if comment:
            self.grid_canvas.add_comment_to_last_selected(comment)
            self.comment_entry.delete(0, tk.END)
            self._update_status(f"Comment added to selected images")

    def _push_undo_action(self, action: UndoAction):
        """Push an action to the undo stack"""
        self.undo_stack.append(action)
        # Clear redo stack when new action is performed
        self.redo_stack.clear()
        self.logger.debug(f"Pushed undo action: {action.get_description()}")

        # Update status to show undo is available
        self._update_status(f"{action.get_description()} (Ctrl+Z to undo)")

    def _apply_state_to_images(self, action: UndoAction, use_old_state: bool):
        """Apply saved state to images (for undo/redo)"""
        states_to_apply = action.old_states if use_old_state else action.new_states

        for (idx, filename), state in zip(action.affected_images, states_to_apply):
            # Find the image - it might have moved in the list
            img = None
            current_idx = idx

            # First try the original index
            if (
                idx < len(self.displayed_images)
                and self.displayed_images[idx].filename == filename
            ):
                img = self.displayed_images[idx]
            else:
                # Search for it by filename
                for i, display_img in enumerate(self.displayed_images):
                    if display_img.filename == filename:
                        img = display_img
                        current_idx = i
                        break

            if img:
                # Restore state
                img.classification = state["classification"]
                img.comments = state["comments"]
                img.classified_by = state["classified_by"]
                img.classified_date = state["classified_date"]
                # Restore tags if present in state
                if "tags" in state:
                    img.tags = set(state["tags"]) if state["tags"] else set()

                # IMPORTANT: Also update in all_images
                for all_img in self.all_images:
                    if (
                        all_img.hole_id == img.hole_id
                        and all_img.depth_to == img.depth_to
                        and all_img.filename == img.filename
                    ):
                        all_img.classification = img.classification
                        all_img.comments = img.comments
                        # Also restore tags in all_images
                        if "tags" in state:
                            all_img.tags = set(state["tags"]) if state["tags"] else set()
                        all_img.classified_by = img.classified_by
                        all_img.classified_date = img.classified_date
                        break

        # Refresh the display with error handling
        try:
            if self.grid_canvas:
                self.grid_canvas._refresh_classifications()
        except KeyError as e:
            self.logger.warning(
                f"Could not refresh all classifications - some cells may not be loaded: {e}"
            )

        # Refresh the grid display
        if self.grid_canvas:
            self.grid_canvas._refresh_classifications()
            self.grid_canvas.mark_changed()

        self._update_statistics()

        # Mark that we have unsaved changes (will be saved by autosave)
        self.has_unsaved_changes = True

    def _toggle_hide_classified(self):
        """Toggle hiding/showing of classified images"""
        self.hide_classified = not self.hide_classified

        self.logger.info(f"Toggled hide_classified to: {self.hide_classified}")

        if self.hide_classified:
            self.hide_btn.configure(text="Show All")
        else:
            self.hide_btn.configure(text="Hide Classified")

        # Always reapply filters to rebuild displayed_images properly
        self._apply_filters()

    def _on_display_mode_changed(self):
        """Handle display mode change"""
        mode = self.display_mode_var.get()
        self.logger.info(f"Display mode changed to: {mode}")

        # Update hole_by_hole_mode based on selection
        self.hole_by_hole_mode = mode == "hole_by_hole"

        # Update collapsible frame header text based on mode
        mode_text_map = {
            "hole_by_hole": "Mode: Hole-By-Hole",
            "all_images": "Mode: All Images",
            "all_intervals": "Mode: All Intervals",
        }
        
        if mode == "hole_by_hole" and self.unique_holes and self.current_hole_index < len(self.unique_holes):
            current_hole = self.unique_holes[self.current_hole_index]
            header_text = f"Mode: Hole-By-Hole - {current_hole} ({self.current_hole_index + 1}/{len(self.unique_holes)})"
        else:
            header_text = mode_text_map.get(mode, "Mode: All Images")
        
        self.hole_nav_frame.set_text(header_text)

        # Show/hide navigation controls based on mode
        if mode == "hole_by_hole":
            # Show navigation buttons and auto-advance
            self.nav_buttons_frame.pack(fill=tk.X, pady=(10, 5))
            self.auto_advance_check.pack(fill=tk.X, pady=2)
            
            # Enable navigation
            self.prev_hole_btn.set_state("normal")
            self.next_hole_btn.set_state("normal")
            
            # Update to current hole if available
            if self.unique_holes and self.current_hole_index < len(self.unique_holes):
                self._update_hole_navigation()
        else:
            # Hide navigation buttons and auto-advance for other modes
            self.nav_buttons_frame.pack_forget()
            self.auto_advance_check.pack_forget()

        # Reapply filters to update display
        self._apply_filters()

    def _toggle_hole_mode(self):
        """Toggle hole-by-hole mode"""
        # Toggle between hole_by_hole and all_images
        current = self.display_mode_var.get()
        if current == "hole_by_hole":
            self.display_mode_var.set("all_images")
        else:
            self.display_mode_var.set("hole_by_hole")
        self._on_display_mode_changed()

        if self.hole_by_hole_mode:
            # Enable hole mode
            self.hole_mode_button.configure(
                color=self.gui_manager.theme_colors["accent_green"]
            )

            # Get unique holes from all images
            holes = sorted(set(img.hole_id for img in self.all_images))
            self.unique_holes = holes

            if self.unique_holes:
                self.current_hole_index = 0
                self._update_hole_navigation()
                self._apply_filters()  # Reapply filters for current hole
            else:
                self.logger.warning("No holes found")
                self._toggle_hole_mode()  # Turn it back off
        else:
            # Disable hole mode
            self.hole_mode_button.configure(
                color=self.gui_manager.theme_colors["secondary_bg"]
            )
            self.prev_hole_btn.configure(state="disabled")
            self.next_hole_btn.configure(state="disabled")
            self.hole_label.configure(text="All Holes")
            self._apply_filters()  # Reapply filters for all holes

    def _previous_hole(self):
        """Navigate to previous hole"""
        if self.current_hole_index > 0:
            self.current_hole_index -= 1
            self._update_hole_navigation()
            self._apply_filters()

    def _next_hole(self):
        """Navigate to next hole"""
        if self.current_hole_index < len(self.unique_holes) - 1:
            self.current_hole_index += 1
            self._update_hole_navigation()
            self._apply_filters()

    def _update_hole_navigation(self):
        """Update hole navigation UI"""
        if not self.hole_by_hole_mode or not self.unique_holes:
            return

        current_hole = self.unique_holes[self.current_hole_index]

        # Update collapsible frame header
        header_text = f"Mode: Hole-By-Hole - {current_hole} ({self.current_hole_index + 1}/{len(self.unique_holes)})"
        self.hole_nav_frame.set_text(header_text)

        # Update button states
        self.prev_hole_btn.configure(
            state="normal" if self.current_hole_index > 0 else "disabled"
        )
        self.next_hole_btn.configure(
            state=(
                "normal"
                if self.current_hole_index < len(self.unique_holes) - 1
                else "disabled"
            )
        )

        self.logger.info(f"Navigated to hole {current_hole}")

    def _toggle_auto_advance(self):
        """Toggle auto-advance setting"""
        self.auto_advance_enabled = self.auto_advance_var.get()
        self.logger.info(
            f"Auto-advance {'enabled' if self.auto_advance_enabled else 'disabled'}"
        )

    def _check_hole_completion(self):
        """Check if all filtered images in current hole are classified"""
        if not self.hole_by_hole_mode or not self.displayed_images:
            return False

        # Check if all displayed images are classified
        unclassified = sum(
            1
            for img in self.displayed_images
            if img.classification == ClassificationCategory.UNASSIGNED
        )

        self.logger.debug(
            f"Hole completion check: {unclassified} unclassified of {len(self.displayed_images)}"
        )

        return unclassified == 0

    def _auto_advance_if_complete(self):
        """Auto-advance to next hole if current hole is complete"""
        if not self.hole_by_hole_mode or not self.auto_advance_enabled:
            return

        if self._check_hole_completion():
            current_hole = self.unique_holes[self.current_hole_index]

            # Check if there's a next hole
            if self.current_hole_index < len(self.unique_holes) - 1:
                self.logger.info(
                    f"Hole {current_hole} complete, auto-advancing to next hole"
                )

                # Show brief notification
                self._update_status(
                    f"✓ Hole {current_hole} complete! Moving to next hole..."
                )

                # Schedule the advance after a brief delay so user sees the message
                self.dialog.after(1000, self._next_hole)
            else:
                # This was the last hole
                self.logger.info(
                    f"Hole {current_hole} complete - this was the last hole"
                )
                self._update_status(f"✓ All holes complete!")

                # Optionally show completion dialog
                self.dialog.after(500, lambda: self._show_completion_dialog())

    def _show_completion_dialog(self):
        """Show dialog when all holes are complete"""
        DialogHelper.show_message(
            self.dialog,
            "Classification Complete",
            f"All {len(self.unique_holes)} holes have been classified!\n\n"
            "You can:\n"
            "• Review your classifications\n"
            "• Export the results\n"
            "• Turn off hole-by-hole mode to see all images",
            message_type="info",
        )

    def _rotate_images(self):
        """Rotate all images 90 degrees"""
        self.grid_canvas.rotate_images(90)

    def _scale_images(self, factor: float):
        """Scale image display size"""
        new_scale = self.grid_canvas.scale_factor * factor
        new_scale = max(0.3, min(5.0, new_scale))  # Match the canvas limit

        # Check minimum size constraint
        if new_scale < self.grid_canvas.scale_factor:
            # Zooming out - check if it would make cells too small
            test_width = int(self.grid_canvas.base_cell_width * new_scale)
            if test_width < 50:  # Minimum 50px width
                return  # Don't scale further

        if new_scale != self.grid_canvas.scale_factor:
            self.grid_canvas.scale_factor = new_scale
            # Don't manually set cell dimensions - let load_images recalculate
            # with the new scale factor, including proper aspect ratios and viz space
            self.grid_canvas.load_images(self.displayed_images, preserve_selection=True)

            self.logger.debug(f"Scaled to {new_scale:.2f}x")

    def _import_csv_data(self):
        """Import multiple CSV data files using DrillholeDataManager"""
        file_paths = filedialog.askopenfilenames(
            parent=self.dialog,
            title="Select CSV Data Files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            multiple=True,
        )

        if not file_paths:
            return

        # Import ProgressDialog
        from gui.progress_dialog import ProgressDialog

        # Create progress dialog for loading only
        progress = ProgressDialog(self.dialog, "Importing Data", "Loading CSV files...")

        # Store results for later display
        results = {}
        success_count = 0

        def import_task():
            """Task to run in background thread"""
            nonlocal results, success_count

            try:
                # Load CSV files
                progress.update_progress("Loading CSV files...", 30)
                results = self.drillhole_data_manager.load_csv_files(list(file_paths))

                # Count successes and show interval info
                for filename, status in results.items():
                    if "Successfully loaded" in status:
                        success_count += 1
                        # Log the actual intervals from the loaded data
                        if filename in self.drillhole_data_manager.data_info:
                            info = self.drillhole_data_manager.data_info[filename]
                            interval_scale = info["interval_scale"].value
                            row_count = info["row_count"]
                            hole_count = len(info["hole_ids"])
                            self.logger.info(
                                f"Loaded {filename}: {row_count} rows, {hole_count} holes, {interval_scale}m intervals"
                            )

                if success_count == 0:
                    progress.update_progress("No files loaded successfully", 100)
                    return

                # Update UI without harmonization
                progress.update_progress("Updating filters...", 80)

                def update_ui():
                    try:
                        # DON'T Clear CSV cache BEFORE populating new data
                        # self._clear_csv_cache()
                        self.last_filter_hash = None

                        # Populate DrillholeDataManager data for all images
                        self._populate_drillhole_data_for_images()

                        # Update filter columns with the new data
                        self._update_filter_columns()

                        # Trigger re-filter to apply new data
                        self._apply_filters()

                        self.logger.info(
                            f"Updated filters with {success_count} new data sources"
                        )
                    except Exception as e:
                        self.logger.error(f"Error updating UI: {e}")

                # Use after_idle to ensure it runs on main thread
                self.dialog.after_idle(update_ui)

                progress.update_progress("Import complete!", 100)

            except Exception as e:
                self.logger.error(f"Error during import: {e}")
                progress.update_progress(f"Error: {str(e)}", 100)

        # Run the import task with progress
        progress.run_with_progress(import_task)

        # Show results after progress dialog closes
        if results:
            message_lines = ["Data Import Results:\n"]
            for filename, status in results.items():
                if "Successfully loaded" in status:
                    message_lines.append(f"✓ {filename}: {status}")
                else:
                    message_lines.append(f"✗ {filename}: {status}")

            DialogHelper.show_message(
                self.dialog,
                "Import Complete",
                "\n".join(message_lines),
                message_type="info" if success_count > 0 else "warning",
            )

    def _export_csv(self):
        """Export classifications to CSV with column selection dialog"""
        if not self.all_images:
            DialogHelper.show_message(
                self.dialog,
                "No Data",
                "No images available to export.",
                message_type="warning",
            )
            return

        # Load other users' reviews for all images (for consensus data)
        if hasattr(self, "other_user_reviews"):
            for img in self.all_images:
                key = (img.hole_id, img.depth_from, img.depth_to)
                if key in self.other_user_reviews:
                    img.other_reviews = self.other_user_reviews[key]

        try:
            from gui.csv_export_dialog import CSVExportDialog
            
            export_dialog = CSVExportDialog(
                parent=self.dialog,
                all_images=self.all_images,
                gui_manager=self.gui_manager,
                data_coordinator=self.data_coordinator,
                drillhole_data_manager=self.drillhole_data_manager,
                item_manager=self.item_manager,
                config_manager=self.config_manager,
                hex_color_cache=getattr(self, 'hex_color_cache', {}),
                other_user_reviews=getattr(self, 'other_user_reviews', {}),
            )
            
            result = export_dialog.show()
            
            if result:
                self._update_status(f"Exported to {Path(result).name}")
                
        except ImportError as e:
            self.logger.warning(f"CSVExportDialog not available, using legacy export: {e}")
            # Fall back to legacy export
            # self._export_csv_legacy()
            self.logger.error(f"Error showing CSV export dialog: {e}", exc_info=True)
        except Exception as e:
            self.logger.error(f"Error showing CSV export dialog: {e}", exc_info=True)
            DialogHelper.show_message(
                self.dialog,
                "Export Error",
                f"Failed to open CSV export dialog:\n{str(e)}",
                message_type="error",
            )



    def _show_pdf_export(self):
        """Show PDF export dialog"""
        try:
            self.logger.info("="*60)
            self.logger.info("PDF EXPORT DEBUG - Starting _show_pdf_export")
            self.logger.info("="*60)
            
            # Debug: Check grid_canvas state
            self.logger.info(f"[DEBUG] hasattr grid_canvas: {hasattr(self, 'grid_canvas')}")
            if hasattr(self, 'grid_canvas'):
                self.logger.info(f"[DEBUG] grid_canvas type: {type(self.grid_canvas)}")
                self.logger.info(f"[DEBUG] hasattr viz_columns: {hasattr(self.grid_canvas, 'viz_columns')}")
                if hasattr(self.grid_canvas, 'viz_columns'):
                    self.logger.info(f"[DEBUG] viz_columns: {self.grid_canvas.viz_columns}")
                    self.logger.info(f"[DEBUG] viz_columns length: {len(self.grid_canvas.viz_columns) if self.grid_canvas.viz_columns else 0}")
                self.logger.info(f"[DEBUG] hasattr viz_column_configs: {hasattr(self.grid_canvas, 'viz_column_configs')}")
                if hasattr(self.grid_canvas, 'viz_column_configs'):
                    self.logger.info(f"[DEBUG] viz_column_configs keys: {list(self.grid_canvas.viz_column_configs.keys()) if self.grid_canvas.viz_column_configs else 'empty'}")
                self.logger.info(f"[DEBUG] hasattr viz_column_sizes: {hasattr(self.grid_canvas, 'viz_column_sizes')}")
                if hasattr(self.grid_canvas, 'viz_column_sizes'):
                    self.logger.info(f"[DEBUG] viz_column_sizes: {self.grid_canvas.viz_column_sizes}")
            
            # Get visualizer - create if needed and configure with current viz columns
            visualizer = None
            if hasattr(self.grid_canvas, 'drillhole_visualizer'):
                visualizer = self.grid_canvas.drillhole_visualizer
                self.logger.info(f"[DEBUG] Got visualizer from grid_canvas.drillhole_visualizer")
            elif hasattr(self, 'drillhole_data_visualizer'):
                visualizer = self.drillhole_data_visualizer
                self.logger.info(f"[DEBUG] Got visualizer from self.drillhole_data_visualizer")
            else:
                # Create visualizer from data manager if available
                if self.data_coordinator or self.drillhole_data_manager:
                    from processing.LoggingReviewStep.drillhole_data_visualizer import (
                        DrillholeDataVisualizer,
                        VisualizationMode,
                    )
                    visualizer = DrillholeDataVisualizer(mode=VisualizationMode.STITCHABLE)
                    self.logger.info("[DEBUG] Created NEW visualizer for PDF export")
            
            self.logger.info(f"[DEBUG] visualizer is None: {visualizer is None}")
            if visualizer:
                self.logger.info(f"[DEBUG] visualizer type: {type(visualizer)}")
                self.logger.info(f"[DEBUG] visualizer.plot_configs before config: {len(visualizer.plot_configs)}")
            
            # Configure visualizer with grid canvas viz columns if available
            viz_columns_available = hasattr(self.grid_canvas, 'viz_columns') and self.grid_canvas.viz_columns
            self.logger.info(f"[DEBUG] viz_columns_available: {viz_columns_available}")
            
            if visualizer and viz_columns_available:
                # Clear any existing configs
                visualizer.plot_configs = []
                self.logger.info(f"[DEBUG] Cleared plot_configs, now configuring {len(self.grid_canvas.viz_columns)} columns")
                
                # Get a sample of csv_data keys to help with column name matching
                sample_csv_keys = []
                if self.displayed_images:
                    sample_img = self.displayed_images[0]
                    if hasattr(sample_img, 'csv_data') and sample_img.csv_data:
                        sample_csv_keys = list(sample_img.csv_data.keys())
                        self.logger.info(f"[DEBUG] Sample csv_data keys: {sample_csv_keys[:20]}")
                
                for col_name in self.grid_canvas.viz_columns:
                    self.logger.info(f"[DEBUG] Processing column: {col_name}")
                    
                    # Skip spacer columns
                    if col_name == "--- Spacer ---":
                        self.logger.info(f"[DEBUG]   Skipping spacer column")
                        continue
                    
                    # Get color map from grid canvas config
                    color_map_obj = None
                    if hasattr(self.grid_canvas, 'viz_column_configs') and col_name in self.grid_canvas.viz_column_configs:
                        config = self.grid_canvas.viz_column_configs[col_name]
                        self.logger.info(f"[DEBUG]   config type: {type(config)}")
                        # Config may be a dict with 'color_map' key or the color map directly
                        if isinstance(config, dict):
                            color_map_obj = config.get('color_map')
                            self.logger.info(f"[DEBUG]   extracted color_map from dict: {type(color_map_obj)}")
                        else:
                            color_map_obj = config
                            self.logger.info(f"[DEBUG]   using config directly as color_map: {type(color_map_obj)}")
                    else:
                        self.logger.info(f"[DEBUG]   No config found for column {col_name}")
                    
                    # Get column size if available
                    col_width = 40  # Default width for PDF
                    if hasattr(self.grid_canvas, 'viz_column_sizes') and col_name in self.grid_canvas.viz_column_sizes:
                        col_width = self.grid_canvas.viz_column_sizes.get(col_name, 40)
                    self.logger.info(f"[DEBUG]   col_width: {col_width}")
                    
                    # CRITICAL: Map the display column name to the actual csv_data key
                    # Strip source suffix if present (e.g., "fe_pct (exassay)" -> "fe_pct")
                    csv_col_key = col_name.split(" (")[0].strip().lower() if " (" in col_name else col_name.lower()
                    self.logger.info(f"[DEBUG]   Initial csv_col_key (stripped): {csv_col_key}")
                    
                    # Try to find a matching key in csv_data
                    matched_key = None
                    if sample_csv_keys:
                        # First try exact lowercase match
                        for csv_key in sample_csv_keys:
                            if csv_key.lower() == csv_col_key:
                                matched_key = csv_key
                                break
                        
                        # If no match, try partial matching (e.g., "fe_pct" might match "fe_pct_best")
                        if not matched_key:
                            for csv_key in sample_csv_keys:
                                csv_key_lower = csv_key.lower()
                                # Check if the stripped key is a prefix or contains the key
                                if csv_key_lower.startswith(csv_col_key) or csv_col_key in csv_key_lower:
                                    matched_key = csv_key
                                    self.logger.info(f"[DEBUG]   Partial match found: {csv_col_key} -> {csv_key}")
                                    break
                    
                    # Use matched key or fall back to stripped key
                    final_col_key = matched_key if matched_key else csv_col_key
                    self.logger.info(f"[DEBUG]   Final column key for PlotConfig: {final_col_key}")
                    
                    # Create PlotConfig for this column
                    plot_config = PlotConfig(
                        plot_type=PlotType.SOLID_COLUMN,
                        width=col_width,
                    )
                    plot_config.columns = [final_col_key]  # Use the MATCHED key, not the display name
                    plot_config.title = col_name  # Keep display name for title
                    
                    # Pass color map object via custom_params (as _create_solid_column expects)
                    if color_map_obj is not None:
                        plot_config.custom_params["color_map_obj"] = color_map_obj
                        self.logger.info(f"[DEBUG]   Added color_map_obj to custom_params")
                    else:
                        self.logger.info(f"[DEBUG]   WARNING: No color_map_obj for {col_name}")
                    
                    visualizer.add_plot_column(plot_config)
                    self.logger.info(f"[DEBUG]   Added PlotConfig: columns={plot_config.columns}, width={plot_config.width}")
                
                self.logger.info(f"[DEBUG] Configured PDF visualizer with {len(visualizer.plot_configs)} columns")
            else:
                self.logger.warning(f"[DEBUG] NOT configuring visualizer - visualizer={visualizer is not None}, viz_columns_available={viz_columns_available}")
            
            # Debug: Check sample image csv_data
            if self.displayed_images:
                sample_img = self.displayed_images[0]
                self.logger.info(f"[DEBUG] Sample image: {sample_img.filename}")
                self.logger.info(f"[DEBUG] Sample image csv_data exists: {hasattr(sample_img, 'csv_data')}")
                if hasattr(sample_img, 'csv_data'):
                    self.logger.info(f"[DEBUG] Sample image csv_data is None: {sample_img.csv_data is None}")
                    self.logger.info(f"[DEBUG] Sample image csv_data type: {type(sample_img.csv_data)}")
                    if sample_img.csv_data:
                        self.logger.info(f"[DEBUG] Sample image csv_data keys: {list(sample_img.csv_data.keys())[:10]}")  # First 10 keys
                        # Check if viz columns are in csv_data
                        if hasattr(self.grid_canvas, 'viz_columns'):
                            for col in self.grid_canvas.viz_columns[:3]:  # Check first 3
                                self.logger.info(f"[DEBUG] Column '{col}' in csv_data: {col in sample_img.csv_data}")
                                if col in sample_img.csv_data:
                                    self.logger.info(f"[DEBUG]   Value: {sample_img.csv_data[col]}")
            
            # Create and show export dialog
            # Pass DataCoordinator if available, fall back to DrillholeDataManager
            data_manager_for_export = self.data_coordinator if self.data_coordinator else self.drillhole_data_manager
            self.logger.info(f"[DEBUG] data_manager_for_export type: {type(data_manager_for_export)}")
            
            export_dialog = PDFExportDialog(
                parent=self.dialog,
                all_images=self.all_images,
                displayed_images=self.displayed_images,
                gui_manager=self.gui_manager,
                drillhole_data_manager=data_manager_for_export,
                drillhole_visualizer=visualizer,
                item_manager=self.item_manager,
                data_coordinator=self.data_coordinator,
            )
            self.logger.info("[DEBUG] PDFExportDialog created, calling show()")
            export_dialog.show()
            
        except Exception as e:
            self.logger.error(f"Error showing PDF export: {e}", exc_info=True)
            DialogHelper.show_message(
                self.dialog,
                "Export Error",
                f"Failed to open PDF export dialog:\n{str(e)}",
            )


    def _manual_save(self):
        """Save classified and/or tagged images to JSON register"""
        if not self.all_images:
            DialogHelper.show_message(
                self.dialog, "No Data", "No images to save.", message_type="info"
            )
            return

        try:
            reviews_to_save = []
            for img in self.all_images:
                has_classification = not (
                    img.classification == "Unassigned"
                    or img.classification == ClassificationCategory.UNASSIGNED
                    or img.classification == "ClassificationCategory.UNASSIGNED"
                    or img.classification == ""
                    or not img.classification
                )
                has_tags = hasattr(img, "tags") and img.tags and len(img.tags) > 0

                if not has_classification and not has_tags:
                    continue

                # Skip unchanged images to avoid incrementing review count
                original_tags = set(img.original_tags) if hasattr(img, "original_tags") else set()
                current_tags = img.tags if hasattr(img, "tags") and img.tags else set()
                
                if (
                    hasattr(img, "original_classification")
                    and img.classification == img.original_classification
                    and hasattr(img, "original_comments")
                    and img.comments == img.original_comments
                    and original_tags == current_tags
                ):
                    continue

                tags_list = list(img.tags) if hasattr(img, "tags") and img.tags else []
                
                classification_str = ""
                if has_classification:
                    classification_str = (
                        str(self._get_classification_string(img.classification))
                        if hasattr(img.classification, "value")
                        else str(img.classification)
                    )

                reviews_to_save.append(
                    {
                        "hole_id": img.hole_id,
                        "depth_from": img.depth_from,
                        "depth_to": img.depth_to,
                        "comments": img.comments,
                        "classification": classification_str,
                        "tags": tags_list,
                    }
                )

            # Batch save to JSON register
            if self.json_manager and reviews_to_save:
                stats = self.json_manager.batch_update_compartment_reviews(
                    reviews_to_save
                )
                reviews_saved = stats["updated"] + stats["created"]
            else:
                reviews_saved = 0

            # Update save state for classified AND tagged images
            self.last_save_state = {}
            for img in self.all_images:
                has_class = img.classification != ClassificationCategory.UNASSIGNED
                has_tags = hasattr(img, "tags") and img.tags and len(img.tags) > 0
                if has_class or has_tags:
                    tags_frozen = frozenset(img.tags) if has_tags else frozenset()
                    self.last_save_state[img.filename] = (img.classification, tags_frozen)

            # Update original state so next save detects only new changes
            for img in self.all_images:
                if hasattr(img, "tags") and img.tags:
                    img.original_tags = set(img.tags)
                img.original_classification = img.classification
                img.original_comments = img.comments

            self.has_unsaved_changes = False

            # Refresh display
            if self.grid_canvas:
                self.grid_canvas._refresh_classifications()
            self._update_statistics()

            tag_count = sum(1 for img in self.all_images if hasattr(img, "tags") and img.tags)
            self._update_status(f"Saved {reviews_saved} reviews ({tag_count} tagged) to register")

        except Exception as e:
            self.logger.error(f"Save failed: {e}")
            DialogHelper.show_message(
                self.dialog,
                "Save Failed",
                f"Failed to save: {str(e)}",
                message_type="error",
            )

    def _schedule_autosave(self):
        """Schedule next autosave"""
        self.autosave_job = self.dialog.after(self.AUTOSAVE_INTERVAL, self._autosave)

    def _autosave(self):
        """Perform autosave only if there are changes"""
        if self.all_images and self.has_unsaved_changes:
            try:
                reviews_to_save = []
                for img in self.all_images:
                    has_classification = (
                        img.classification != "Unassigned"
                        and img.classification != ClassificationCategory.UNASSIGNED
                        and img.classification != "ClassificationCategory.UNASSIGNED"
                        and img.classification != ""
                        and img.classification
                    )
                    has_tags = hasattr(img, "tags") and img.tags and len(img.tags) > 0

                    if not has_classification and not has_tags:
                        continue

                    tags_list = list(img.tags) if has_tags else []

                    classification_str = ""
                    if has_classification:
                        classification_str = (
                            str(img.classification.value)
                            if hasattr(img.classification, "value")
                            else str(img.classification)
                        )

                    reviews_to_save.append(
                        {
                            "hole_id": img.hole_id,
                            "depth_from": img.depth_from,
                            "depth_to": img.depth_to,
                            "comments": img.comments,
                            "classification": classification_str,
                            "moisture_status": img.moisture_status,
                            "classified_by": img.classified_by,
                            "classified_date": img.classified_date,
                            "active_filters": img.active_filters,
                            "compartment_uid": img.compartment_uid,
                            "tags": tags_list,
                        }
                    )

                if self.json_manager and reviews_to_save:
                    stats = self.json_manager.batch_update_compartment_reviews(
                        reviews_to_save
                    )
                    reviews_saved = stats["updated"] + stats["created"]
                    self.logger.debug(f"Autosaved {reviews_saved} reviews to register")
                else:
                    reviews_saved = 0

                if self.autosave_path and self.displayed_images:
                    csv_handler = CSVHandler(self.logger)
                    csv_handler.export_classifications(
                        self.all_images, str(self.autosave_path), classified_only=True
                    )

                self.logger.debug(f"Autosaved {reviews_saved} reviews to register")
                self.has_unsaved_changes = False

                # Update last save state for classified AND tagged images
                self.last_save_state = {}
                for img in self.all_images:
                    has_class = (
                        img.classification != "Unassigned"
                        and img.classification != ClassificationCategory.UNASSIGNED
                        and img.classification
                    )
                    has_tags = hasattr(img, "tags") and img.tags and len(img.tags) > 0
                    if has_class or has_tags:
                        tags_frozen = frozenset(img.tags) if has_tags else frozenset()
                        self.last_save_state[img.filename] = (img.classification, tags_frozen)

                # Update original state so next cycle detects only new changes
                for img in self.all_images:
                    if hasattr(img, "tags") and img.tags:
                        img.original_tags = set(img.tags)
                    img.original_classification = img.classification
                    img.original_comments = img.comments

            except Exception as e:
                self.logger.error(f"Autosave failed: {e}")

        self._schedule_autosave()

    def _bind_shortcuts(self):
        """Bind keyboard shortcuts - unified for both classifications and tags.
        
        Shift+<tag key> applies the negative tag (not_<tag_id>) to selected images,
        marking them as explicitly reviewed-and-skipped for that tag.
        """
        if hasattr(self, "_item_bindings"):
            for binding in self._item_bindings:
                try:
                    self.dialog.unbind(binding)
                except:
                    pass

        self._item_bindings = []

        all_items_with_keys = []
        
        for class_def in self.item_manager.get_active_classifications():
            if class_def.keybinding:
                all_items_with_keys.append((class_def.keybinding, class_def.id))
        
        for tag_def in self.item_manager.get_active_tags():
            if tag_def.keybinding:
                all_items_with_keys.append((tag_def.keybinding, tag_def.id))

        for key, item_id in all_items_with_keys:
            binding = f"<Key-{key}>"
            self.dialog.bind(binding, lambda e, iid=item_id: self._set_mode(iid))
            self._item_bindings.append(binding)

        # Shift+key for tag negative (skip/exclude) � only for tags
        for tag_def in self.item_manager.get_active_tags():
            if tag_def.keybinding:
                shift_binding = f"<Shift-Key-{tag_def.keybinding.upper()}>"
                self.dialog.bind(
                    shift_binding,
                    lambda e, tid=tag_def.id: self._apply_negative_tag_to_selected(tid),
                )
                self._item_bindings.append(shift_binding)
                self.logger.debug(f"Bound Shift+{tag_def.keybinding} to not_{tag_def.id}")

        # Other shortcuts (unchanged)
        self.dialog.bind("<Control-z>", lambda e: self._undo_last_action())
        self.dialog.bind("<Control-y>", lambda e: self._redo_last_action())
        self.dialog.bind("<Control-s>", lambda e: self._manual_save())

    def _undo(self):
        """Undo last action"""
        if not self.undo_stack:
            self._update_status("Nothing to undo")
            return

        action = self.undo_stack.pop()

        # Apply the old state
        self._apply_state_to_images(action, use_old_state=True)

        # Move to redo stack
        self.redo_stack.append(action)

        self.logger.info(f"Undid: {action.get_description()}")
        self._update_status(f"Undid: {action.get_description()} (Ctrl+Y to redo)")

    def _redo(self):
        """Redo last undone action"""
        if not self.redo_stack:
            self._update_status("Nothing to redo")
            return

        action = self.redo_stack.pop()

        # Apply the new state
        self._apply_state_to_images(action, use_old_state=False)

        # Move back to undo stack
        self.undo_stack.append(action)

        self.logger.info(f"Redid: {action.get_description()}")
        self._update_status(f"Redid: {action.get_description()} (Ctrl+Z to undo)")

    def _update_statistics(self):
        """Update statistics display with saved vs pending - dynamic"""
        if not self.displayed_images:
            self.stats_label.config(text="No images loaded")
            return

        stats = self.grid_canvas.get_statistics() if self.grid_canvas else {}

        # Calculate saved vs pending (handles both old str and new tuple format)
        saved_count = 0
        pending_count = 0
        for img in self.displayed_images:
            has_class = img.classification != "Unassigned" and img.classification
            has_tags = hasattr(img, "tags") and img.tags and len(img.tags) > 0
            if has_class or has_tags:
                saved_state = self.last_save_state.get(img.filename)
                if saved_state is None:
                    pending_count += 1
                elif isinstance(saved_state, tuple):
                    saved_class, saved_tags = saved_state
                    current_tags = frozenset(img.tags) if has_tags else frozenset()
                    if saved_class == img.classification and saved_tags == current_tags:
                        saved_count += 1
                    else:
                        pending_count += 1
                else:
                    # Legacy format: just classification string
                    if saved_state == img.classification:
                        saved_count += 1
                    else:
                        pending_count += 1

        # Build lines dynamically based on active classifications
        lines = [f"Total: {stats.get('total', 0)}"]

        # Add line for each active classification
        for class_def in self.item_manager.get_active_classifications():
            count = stats.get(class_def.id, 0)
            # Truncate label if too long
            label = (
                class_def.label[:15] if len(class_def.label) > 15 else class_def.label
            )
            lines.append(f"{label}: {count}")

        # Always show unassigned
        lines.append(f"Unassigned: {stats.get('unassigned', 0)}")

        # Tag counts
        for tag_def in self.item_manager.get_active_tags():
            tag_count = sum(
                1 for img in self.displayed_images
                if hasattr(img, "tags") and img.tags and tag_def.id in img.tags
            )
            if tag_count > 0:
                label = tag_def.label[:15] if len(tag_def.label) > 15 else tag_def.label
                icon = f"{tag_def.icon} " if tag_def.icon else ""
                lines.append(f"{icon}{label}: {tag_count}")

        # Add save status
        lines.extend(
            [
                "",
                f"✓ Saved: {saved_count}",
                f"⚠ Pending: {pending_count}",
            ]
        )

        # Add hole progress if in hole-by-hole mode
        if self.hole_by_hole_mode and self.unique_holes:
            current_hole = self.unique_holes[self.current_hole_index]
            hole_progress = (
                f"Hole {self.current_hole_index + 1}/{len(self.unique_holes)}"
            )

            # Calculate completion percentage for current hole
            if stats.get("total", 0) > 0:
                classified = stats.get("total", 0) - stats.get("unassigned", 0)
                percent = (classified / stats.get("total", 0)) * 100
                hole_progress += f"\n{percent:.0f}% complete"

            lines.extend(["", hole_progress])
        else:
            lines.append(f"\nTotal Available: {len(self.all_images)}")

        self.stats_label.config(text="\n".join(lines))

    def _update_status(self, message: str):
        """Update status bar"""
        self.status_label.config(text=message)

    def _toggle_visualizations(self):
        """Toggle data visualizations display"""
        if self.grid_canvas:
            self.grid_canvas.toggle_data_visualizations()
            # Update button color to indicate state
            if self.grid_canvas.show_data_visualizations:
                self.show_viz_button.configure(
                    color=self.gui_manager.theme_colors["accent_green"]
                )
            else:
                self.show_viz_button.configure(
                    color=self.gui_manager.theme_colors["accent_blue"]
                )

    def _configure_visualizations(self):
        """Open dialog to configure which data columns to visualize"""
        from gui.ReviewDialog.viz_column_settings_dialog import VizColumnSettingsDialog
        
        # Open settings dialog
        # Pass DataCoordinator if available, fall back to DrillholeDataManager
        data_manager_for_viz = self.data_coordinator if self.data_coordinator else self.drillhole_data_manager
        
        settings_dialog = VizColumnSettingsDialog(
            parent=self.dialog,
            gui_manager=self.gui_manager,
            data_coordinator=self.data_coordinator,  # NEW - single source
            config_manager=self.config_manager,
            image_index=self.image_index  # Pass for loading sample images
        )
        
        result = settings_dialog.show()
        
        if result:
            # Apply new configuration
            columns = [config["column"] for config in result]
            
            # Update color map configs - store full config dict for each column
            self.grid_canvas.viz_column_configs.clear()
            for config in result:
                col = config["column"]
                color_map_name = config["color_map"]
                color_map = self.color_map_manager.get_preset(color_map_name)
                
                # Store complete config including color map, label, font settings
                self.grid_canvas.viz_column_configs[col] = {
                    'color_map': color_map,
                    'custom_label': config.get('custom_label', col),
                    'font_size': config.get('font_size', 8),
                    'bold': config.get('bold', False)
                }
                
                if not color_map:
                    self.logger.warning(f"Color map '{color_map_name}' not found for column '{col}'")
            
            # Reload display settings from config_manager
            self.grid_canvas.show_cell_outlines = self.config_manager.get("grid_show_outlines", True)
            self.grid_canvas.cell_outline_width = self.config_manager.get("grid_outline_width", 2)
            self.grid_canvas.show_cell_labels = self.config_manager.get("grid_show_cell_labels", True)
            self.grid_canvas.show_classification_labels = self.config_manager.get("grid_show_classification_labels", True)
            self.grid_canvas.classification_label_position = self.config_manager.get("grid_classification_label_position", "top-right")
            self.grid_canvas.viz_column_width_ratio = self.config_manager.get("viz_column_width_ratio", 0.30)

            self.logger.info(f"Reloaded display settings: outlines={self.grid_canvas.show_cell_outlines}, "
                           f"class_labels={self.grid_canvas.show_classification_labels}")

            # Set visualization columns and refresh
            self.grid_canvas.set_visualization_columns(columns)

            self._update_status(f"Updated visualizations: {len(columns)} columns")

    def _count_pending_changes(self):
        """Count unsaved classifications and tags for close verification."""
        pending_classifications = 0
        pending_tags = 0
        for img in self.all_images:
            has_class = not (
                img.classification == "Unassigned"
                or img.classification == ClassificationCategory.UNASSIGNED
                or img.classification == "ClassificationCategory.UNASSIGNED"
                or img.classification == ""
                or not img.classification
            )
            has_tags = hasattr(img, "tags") and img.tags and len(img.tags) > 0

            original_tags = set(img.original_tags) if hasattr(img, "original_tags") else set()
            current_tags = img.tags if has_tags else set()

            class_changed = (
                has_class
                and hasattr(img, "original_classification")
                and img.classification != img.original_classification
            )
            tags_changed = original_tags != current_tags

            if class_changed:
                pending_classifications += 1
            if tags_changed:
                pending_tags += 1

        return pending_classifications, pending_tags

    def _on_close(self):
        """Handle dialog close with explicit pending-change verification."""
        pending_class, pending_tags = self._count_pending_changes()
        has_pending = self.has_unsaved_changes or pending_class > 0 or pending_tags > 0

        if has_pending:
            parts = []
            if pending_class > 0:
                parts.append(f"{pending_class} classification(s)")
            if pending_tags > 0:
                parts.append(f"{pending_tags} tag change(s)")
            if not parts:
                parts.append("unsaved changes")
            detail = " and ".join(parts)

            result = DialogHelper.show_message(
                self.dialog,
                "Unsaved Changes",
                f"You have {detail} that have not been saved to the register.\n\n"
                "Would you like to save them before closing?",
                message_type="question",
            )
            if result == "yes":
                self._manual_save()
            elif result == "cancel":
                return  # Don't close

        # Clean up grid canvas resources (tooltips, bindings)
        if hasattr(self, 'grid') and self.grid:
            try:
                self.grid.cleanup()
            except Exception as e:
                self.logger.warning(f"Error during grid cleanup: {e}")

        # Unload all images from memory
        for img in self.all_images:
            img.unload_image()

        self.logger.info("Dialog closed, memory cleaned up")
        self.dialog.destroy()