"""
DrillholeColumnWidget - Single column display for drillhole intervals.
Displays images and data for one drillhole with continuous depth axis.
"""

import tkinter as tk
from tkinter import ttk
import logging
import math
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass
import os
import numpy as np
import cv2
from PIL import Image, ImageTk

# Import data models
from gui.DrillholeCorrelation.correlation_models import (
    DrillholeInterval,
    DrillholeSegment,
    DepthTransform,
    StretchRegion,
    StretchMode,
    Discontinuity,
    DiscontinuityType
)
from gui.DrillholeCorrelation.trace_processing import (
    TraceScaleResult,
    scale_trace_values,
)

logger = logging.getLogger(__name__)


class DrillholeColumnWidget(tk.Frame):
    """
    Widget displaying a single drillhole as a continuous column.
    
    Features:
    - Continuous depth axis with placeholder cells for gaps
    - Wet/Dry image selection (prefer Dry)
    - 0.1m depth resolution for data plotting
    - Optional data visualization columns
    - Stretch/compress transformation support
    - Lazy loading for performance
    """

    MIN_GRAPH_COLUMN_CONTENT_WIDTH = 70
    
    def __init__(
        self,
        parent: tk.Widget,
        hole_id: str,
        gui_manager: Any,  # GUIManager instance
        data_manager: Any,  # DrillholeDataManager instance  
        file_manager: Any,  # FileManager instance
        config_manager: Any,  # ConfigManager instance
        color_map_manager: Optional[Any] = None,  # ColorMapManager instance
        data_visualizer: Optional[Any] = None,  # DrillholeDataVisualizer instance
        width: int = 200,
        show_depth_ruler: bool = True,
        show_data_columns: bool = True,
        viz_columns: Optional[List[Dict[str, Any]]] = None,
        moisture_preference: str = "Dry",
        **kwargs
    ):
        """
        Initialize DrillholeColumnWidget.
        
        Args:
            parent: Parent widget
            hole_id: Drillhole identifier
            gui_manager: REQUIRED - GUIManager for theming
            data_manager: REQUIRED - DrillholeDataManager for data access
            file_manager: REQUIRED - FileManager for path resolution
            config_manager: REQUIRED - ConfigManager for settings
            color_map_manager: Optional - ColorMapManager for data colors
            data_visualizer: Optional - DrillholeDataVisualizer for plots
            width: Column width in pixels
            show_depth_ruler: Show depth scale on left
            show_data_columns: Show data visualization columns
            viz_columns: List of visualization column configs
        """
        logger.info(f"Initializing DrillholeColumnWidget for hole: {hole_id}")
        logger.debug(f"  Width: {width}px, Depth ruler: {show_depth_ruler}, Data cols: {show_data_columns}")
        
        # Validate required managers
        if not all([gui_manager, data_manager, file_manager, config_manager]):
            logger.error("Missing required managers!")
            logger.error(f"  gui_manager: {gui_manager is not None}")
            logger.error(f"  data_manager: {data_manager is not None}")
            logger.error(f"  file_manager: {file_manager is not None}")
            logger.error(f"  config_manager: {config_manager is not None}")
            raise ValueError("All managers (gui, data, file, config) are required!")
        
        # Initialize frame with theme colors
        super().__init__(parent, bg=gui_manager.theme_colors["background"], **kwargs)
        
        # Store managers
        self.gui_manager = gui_manager
        self.data_manager = data_manager
        self.file_manager = file_manager
        self.config_manager = config_manager
        self.color_map_manager = color_map_manager
        self.data_visualizer = data_visualizer
        
        # Theme shortcuts
        self.theme_colors = gui_manager.theme_colors
        self.fonts = gui_manager.fonts
        
        # Core properties
        self.hole_id = hole_id
        self.width = width
        self.show_depth_ruler = show_depth_ruler
        self.show_data_columns = show_data_columns
        
        # Visualization columns
        self.viz_columns = viz_columns or config_manager.get("correlation_viz_columns", [])
        logger.debug(f"  Viz columns: {[v.get('column') for v in self.viz_columns]}")
        
        # Moisture preference for image display (Wet/Dry)
        self.moisture_preference = moisture_preference
        
        # Cell settings for pure data visualization (NO IMAGES)
        # Cell height = pixels per meter for continuous data display
        self.cell_height = config_manager.get("correlation_data_viz_cell_height", 10)
        
        # Cell width = 0 because no images are displayed in this widget
        # Only depth ruler + data columns
        self.cell_width = 0
        
        # Header height for consistent alignment across all columns
        self.header_height = 45  # Increased to accommodate axis labels
        
        # Depth ruler width - wide enough for 3-digit depths (100-999m)
        self.ruler_width = 40
        
        # Alternating row colors for readability
        self.row_color_even = self.theme_colors.get("field_bg", "#2d2d2d")
        self.row_color_odd = self._darken_color(self.row_color_even, 0.1)
        
        logger.debug(f"  Cell dimensions: {self.cell_width}x{self.cell_height}px (data viz only, no images)")
        
        # Zoom window visible range (for indicators)
        self.zoom_window_depth_range: Optional[Tuple[float, float]] = None
        
        # Data storage
        self.intervals: List[DrillholeInterval] = []
        self.depth_transforms: List[DepthTransform] = []
        self.image_cache: Dict[str, ImageTk.PhotoImage] = {}
        self.canvas_items: Dict[str, int] = {}  # canvas item IDs
        
        # Correlation data (will be populated by parent dialog)
        self.segments = []  # List of DrillholeSegment
        self.discontinuities = []  # List of Discontinuity
        
        # Callbacks for parent dialog
        self.on_discontinuity_add_requested = None  # Callback(hole_id, depth, type)
        self.on_discontinuity_remove_requested = None  # Callback(hole_id, depth)
        self.on_warp_bar_requested = None  # Callback(segment)
        self.on_televiewer_open_requested = None  # Callback(hole_id, depth)
        self.on_layout_changed = None  # Callback after width/height changes
        
        # Scrolling and viewport
        self.visible_range: Tuple[float, float] = (0, 100)  # Visible depth range
        self.loaded_intervals: set = set()  # Indices of loaded intervals
        
        # Data visualization settings (from config)
        self.data_column_min_width = self._get_config_int("correlation_data_column_min_width", 50)
        self.data_column_max_width = self._get_config_int("correlation_data_column_max_width", 260)
        self.data_column_width = self._normalise_data_column_width(
            self._get_config_int("correlation_data_column_width", 70)
        )
        self.show_data_viz = show_data_columns and bool(viz_columns)
        
        # Image display mode: 'none', 'color', 'thumbnail'
        self.image_mode = config_manager.get("correlation_image_mode", "thumbnail")
        self.thumbnail_width = config_manager.get("correlation_thumbnail_width", 160)
        self.thumbnail_min_width = config_manager.get("correlation_thumbnail_min_width", 40)
        self.thumbnail_max_width = config_manager.get("correlation_thumbnail_max_width", 420)
        self.thumbnail_lazy_margin_px = max(0, self._get_config_int("correlation_lazy_thumbnail_margin_px", 250))
        self.thumbnail_lazy_batch_size = max(1, self._get_config_int("correlation_lazy_thumbnail_batch_size", 16))
        self.color_bar_width = config_manager.get("correlation_color_bar_width", 20)
        
        # Load average colors for color-bar mode
        self.average_colors = {}  # {depth_to: hex_color}
        self._load_average_colors()
        
        # Column resize tracking
        self.resizing_column: Optional[int] = None  # Index of column being resized
        self.resize_start_x: Optional[int] = None
        self._viz_source_candidates_cache: Dict[str, List[str]] = {}
        self.thumbnail_refs: Dict[str, ImageTk.PhotoImage] = {}
        self.thumbnail_loaded_indices: Set[int] = set()
        self.thumbnail_missing_indices: Set[int] = set()
        self.thumbnail_canvas_items: Dict[str, int] = {}
        self.thumbnail_load_after_id: Optional[str] = None
        self._thumbnail_x_start: Optional[int] = None
        self._last_thumbnail_visible_range: Tuple[float, float] = (0.0, 1000.0)
        
        # Build UI
        self._create_ui()
        
        # Load data
        self.load_hole_data()

    def _get_config_int(self, key: str, default: int) -> int:
        """Read an integer config value, tolerating legacy string values."""
        try:
            return int(self.config_manager.get(key, default))
        except (TypeError, ValueError):
            return default

    def _minimum_data_column_width_for_content(self) -> int:
        """Return the minimum graph width needed to keep headers legible."""
        width = self.MIN_GRAPH_COLUMN_CONTENT_WIDTH
        for viz_config in self.viz_columns or []:
            column_name = viz_config.get("column", "")
            display_name = (
                viz_config.get("display_name")
                or viz_config.get("custom_label")
                or column_name
            )
            if display_name == column_name:
                display_name = self._create_column_display_name(column_name)
            width = max(width, min(140, len(str(display_name)) * 6 + 16))
        return width

    def _normalise_data_column_width(self, width: Any) -> int:
        """Clamp graph column width to configured and content-aware bounds."""
        try:
            numeric_width = int(width)
        except (TypeError, ValueError):
            numeric_width = 70

        min_width = max(
            int(self.data_column_min_width),
            self._minimum_data_column_width_for_content(),
        )
        max_width = max(int(self.data_column_max_width), min_width)
        return max(min_width, min(max_width, numeric_width))

    def _cancel_thumbnail_load(self) -> None:
        """Cancel any pending lazy thumbnail batch."""
        after_id = getattr(self, "thumbnail_load_after_id", None)
        if after_id:
            try:
                self.after_cancel(after_id)
            except Exception:
                pass
            self.thumbnail_load_after_id = None

    def _reset_thumbnail_state(self) -> None:
        """Clear lazy thumbnail bookkeeping before a redraw."""
        self._cancel_thumbnail_load()
        if hasattr(self, "thumbnail_refs"):
            self.thumbnail_refs.clear()
        else:
            self.thumbnail_refs = {}
        self.thumbnail_loaded_indices = set()
        self.thumbnail_missing_indices = set()
        self.thumbnail_canvas_items = {}
        self._thumbnail_x_start = None
        self._last_thumbnail_visible_range = (0.0, 1000.0)

    def _apply_total_width(self, total_width: Optional[int] = None) -> None:
        """Apply current total width to every fixed-width child container."""
        width = int(total_width if total_width is not None else self.get_total_width())
        self.configure(width=width)
        if hasattr(self, "main_frame"):
            self.main_frame.configure(width=width)
        if hasattr(self, "canvas_frame"):
            self.canvas_frame.configure(width=width)
        if hasattr(self, "canvas"):
            self.canvas.configure(width=width)

    @staticmethod
    def _cover_resize_dimensions(
        source_width: int,
        source_height: int,
        target_width: int,
        target_height: int,
    ) -> Tuple[int, int]:
        """Return uniform-scale dimensions that cover the target rectangle."""
        if source_width <= 0 or source_height <= 0:
            return max(1, target_width), max(1, target_height)

        scale = max(target_width / source_width, target_height / source_height)
        return (
            max(1, int(math.ceil(source_width * scale))),
            max(1, int(math.ceil(source_height * scale))),
        )
    
    def _truncate_text(self, text: str, max_chars: int) -> str:
        """Truncate text with ellipsis if too long."""
        if not text:
            return ""
        if len(text) <= max_chars:
            return text
        if max_chars <= 1:
            return "…"
        return text[:max_chars - 1] + "…"
    
    def _parse_column_with_source(self, column_spec: str) -> Tuple[str, Optional[str]]:
        """
        Parse a column specification that may include source name.
        
        Format: "column_name (source_name)" or just "column_name"
        
        Args:
            column_spec: Column specification like "fe_pct (exassay)" or "fe_pct"
            
        Returns:
            Tuple of (column_name, source_name or None)
        """
        import re
        # Match pattern: column_name (source_name)
        match = re.match(r'^(.+?)\s*\(([^)]+)\)$', column_spec)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return column_spec, None

    def _normalise_viz_source_name(self, source_name: Optional[str]) -> Optional[str]:
        """Treat legacy/generic source labels as an unqualified source lookup."""
        if not source_name:
            return None
        source_text = str(source_name).strip()
        if source_text.lower() in {"data", "all", "all sources", "any", "any source"}:
            return None
        return source_text

    def _column_lookup_candidates(self, column_name: str) -> List[str]:
        """Return compatible column names for legacy saved viz configs."""
        if not column_name:
            return []

        candidates = [column_name.strip()]
        lower_name = candidates[0].lower()
        if lower_name.endswith("_pct") and not lower_name.endswith("_pct_best"):
            candidates.append(f"{candidates[0]}_best")
        elif lower_name.endswith("_ppm") and not lower_name.endswith("_ppm_best"):
            candidates.append(f"{candidates[0]}_best")

        # Preserve order while dropping case-insensitive duplicates.
        result: List[str] = []
        seen = set()
        for candidate in candidates:
            key = candidate.lower()
            if candidate and key not in seen:
                result.append(candidate)
                seen.add(key)
        return result

    def _viz_source_candidates(self, source_name: Optional[str]) -> List[str]:
        """Return source names to try for a viz lookup, preserving priority."""
        normalised_source = self._normalise_viz_source_name(source_name)
        if normalised_source:
            return [normalised_source]

        cache = getattr(self, "_viz_source_candidates_cache", None)
        if cache is None:
            cache = {}
            self._viz_source_candidates_cache = cache
        cache_key = "__generic__"
        if cache_key in cache:
            return list(cache[cache_key])

        candidates: List[str] = []
        if hasattr(self.data_manager, "get_correlation_source_names_for_holes"):
            try:
                candidates.extend(
                    self.data_manager.get_correlation_source_names_for_holes([self.hole_id])
                    or []
                )
            except Exception as e:
                logger.debug(f"Could not get correlation source priority: {e}")

        store = getattr(self.data_manager, "geological_store", None)
        if store and hasattr(store, "list_sources"):
            try:
                candidates.extend(store.list_sources() or [])
            except Exception as e:
                logger.debug(f"Could not list geological sources: {e}")

        result: List[str] = []
        seen = set()
        for candidate in candidates:
            key = str(candidate).strip().lower()
            if key and key not in seen:
                result.append(str(candidate))
                seen.add(key)
        cache[cache_key] = result
        return result

    def _get_value_from_interval_csv(
        self,
        interval: 'DrillholeInterval',
        column_spec: str,
        column_candidates: List[str],
        source_name: Optional[str] = None,
    ) -> Optional[float]:
        """Fast local lookup against the interval row already loaded for drawing."""
        requested_lower = column_spec.lower().strip()
        candidate_lowers = {candidate.lower().strip() for candidate in column_candidates}
        for col, val in interval.csv_data.items():
            col_lower = str(col).lower().strip()
            base_col, key_source = self._parse_column_with_source(str(col))
            base_lower = base_col.lower().strip()
            source_matches = source_name is None or (
                key_source is not None and key_source.lower().strip() == source_name.lower().strip()
            )

            if (
                col_lower == requested_lower
                or col_lower in candidate_lowers
                or (base_lower in candidate_lowers and source_matches)
            ):
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass

        return None
    
    def _get_value_for_interval(
        self, 
        interval: 'DrillholeInterval', 
        column_spec: str,
        source_name: Optional[str] = None,
    ) -> Optional[float]:
        """
        Get a numeric value for an interval, handling source-specific columns.
        
        Args:
            interval: The interval to get data for
            column_spec: Column specification like "fe_pct (exassay)" or "fe_pct"
            source_name: Optional source selected separately in visualization config
            
        Returns:
            Float value or None if not found/invalid
        """
        column_name, parsed_source_name = self._parse_column_with_source(column_spec)
        source_name = self._normalise_viz_source_name(parsed_source_name or source_name)
        column_candidates = self._column_lookup_candidates(column_name)

        # Generic "Data" configs should first use the interval row already
        # selected for this hole. This avoids repeated source-family/collar
        # lookups and lets RC assay rows draw directly from *_BEST columns.
        if not source_name:
            value = self._get_value_from_interval_csv(
                interval,
                column_spec,
                column_candidates,
                source_name=None,
            )
            if value is not None:
                return value

        if source_name and hasattr(self.data_manager, "get_interval_source_value"):
            for candidate in column_candidates:
                try:
                    value = self.data_manager.get_interval_source_value(
                        self.hole_id,
                        interval.depth_from,
                        interval.depth_to,
                        candidate,
                        source_name,
                    )
                    if value is not None:
                        return float(value)
                except Exception as e:
                    logger.debug(f"Interval source lookup failed for {candidate}: {e}")

        if not source_name and hasattr(self.data_manager, "get_interval_source_value"):
            for source_candidate in self._viz_source_candidates(None):
                for candidate in column_candidates:
                    try:
                        value = self.data_manager.get_interval_source_value(
                            self.hole_id,
                            interval.depth_from,
                            interval.depth_to,
                            candidate,
                            source_candidate,
                        )
                        if value is not None:
                            return float(value)
                    except Exception as e:
                        logger.debug(
                            f"Interval lookup failed for {candidate} ({source_candidate}): {e}"
                        )

        # Try the geological store first. If a source was supplied, this is a
        # targeted lookup; otherwise the store searches loaded sources in order.
        if hasattr(self.data_manager, 'geological_store'):
            for candidate in column_candidates:
                try:
                    from processing.DataManager.keys import ImageKey
                    key = ImageKey(
                        hole_id=self.hole_id,
                        depth_to=interval.depth_to
                    )
                    value = self.data_manager.geological_store.get_value(key, candidate, source_name)
                    if value is not None:
                        try:
                            return float(value)
                        except (ValueError, TypeError):
                            pass
                except Exception as e:
                    logger.debug(f"Source lookup failed for {candidate}: {e}")
        
        value = self._get_value_from_interval_csv(
            interval,
            column_spec,
            column_candidates,
            source_name=source_name,
        )
        if value is not None:
            return value
        
        return None

    def _create_column_display_name(self, column_name: str) -> str:
        """
        Create a user-friendly display name from a raw column name.
        
        Examples:
            fe_pct_best -> Fe %
            sio2_pct_best -> SiO2 %
            al2o3_pct -> Al2O3 %
        """
        if not column_name:
            return ""
        
        alias = column_name
        
        # Common replacements
        replacements = {
            '_pct_': ' % ',
            '_pct': ' %',
            'pct_': '% ',
            '_best': '',
            '_BEST': '',
            'logged_': '',
            'Logged_': '',
        }
        
        for old, new in replacements.items():
            alias = alias.replace(old, new)
        
        # Replace underscores with spaces
        alias = alias.replace('_', ' ')
        
        # Title case
        alias = alias.strip().title()
        
        # Fix common chemical formulas
        chemical_fixes = {
            'Sio2': 'SiO2',
            'Al2o3': 'Al2O3',
            'Fe2o3': 'Fe2O3',
            'Cao': 'CaO',
            'Mgo': 'MgO',
            'Tio2': 'TiO2',
            'Loi': 'LOI',
            'Mno': 'MnO',
            'K2o': 'K2O',
            'Na2o': 'Na2O',
            'P2o5': 'P2O5',
        }
        
        for old, new in chemical_fixes.items():
            alias = alias.replace(old, new)
        
        return alias
    
    def _darken_color(self, hex_color: str, factor: float) -> str:
        """Darken a hex color by a factor (0.0-1.0)."""
        try:
            # Remove # prefix
            hex_color = hex_color.lstrip('#')
            
            # Parse RGB
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            
            # Darken
            r = max(0, int(r * (1 - factor)))
            g = max(0, int(g * (1 - factor)))
            b = max(0, int(b * (1 - factor)))
            
            return f"#{r:02x}{g:02x}{b:02x}"
        except Exception:
            return hex_color

    def _blend_color(self, foreground: str, background: str, foreground_weight: float = 0.45) -> str:
        """Blend two Tk colors to create a softer canvas fill."""
        try:
            foreground_weight = max(0.0, min(1.0, float(foreground_weight)))
            bg_weight = 1.0 - foreground_weight
            fr, fg, fb = (value // 256 for value in self.winfo_rgb(foreground))
            br, bg, bb = (value // 256 for value in self.winfo_rgb(background))
            r = int(fr * foreground_weight + br * bg_weight)
            g = int(fg * foreground_weight + bg * bg_weight)
            b = int(fb * foreground_weight + bb * bg_weight)
            return f"#{r:02x}{g:02x}{b:02x}"
        except Exception:
            return foreground
    
    def _create_ui(self) -> None:
        """Create the widget UI components."""
        logger.debug(f"Creating UI for {self.hole_id}")
        
        # Calculate total width for proper sizing
        total_width = self.get_total_width()
        
        # Set widget width explicitly
        self.configure(width=total_width)
        
        # Main container with explicit width
        self.main_frame = tk.Frame(
            self, 
            bg=self.theme_colors["secondary_bg"],
            width=total_width,
            highlightbackground=self.theme_colors["border"],
            highlightcolor=self.theme_colors["border"],
            highlightthickness=1,
            bd=0,
        )
        self.main_frame.pack(fill="both", expand=True)
        self.main_frame.pack_propagate(False)  # Maintain width
        
        # Header with hole ID
        self._create_header()
        
        # Content area with canvas
        self._create_canvas_area()
        
        # Footer with controls (optional)
        self._create_footer()
    
    def _create_header(self) -> None:
        """Create header with hole ID and controls."""
        header_frame = tk.Frame(
            self.main_frame, 
            bg=self.theme_colors["secondary_bg"],
            height=30
        )
        header_frame.pack(fill="x", padx=1, pady=1)
        header_frame.pack_propagate(False)
        
        # Store header reference for parent access
        self.header_frame = header_frame
        
        # Hole ID label
        hole_label = tk.Label(
            header_frame,
            text=self.hole_id,
            font=self.fonts["heading"],
            bg=self.theme_colors["secondary_bg"],
            fg=self.theme_colors["text"]
        )
        hole_label.pack(side="left", padx=5)
        
        # Lock button (small clickable label)
        self.lock_btn = tk.Label(
            header_frame,
            text="🔓",
            font=("Segoe UI", 10),
            bg=self.theme_colors["secondary_bg"],
            fg=self.theme_colors["text"],
            cursor="hand2"
        )
        self.lock_btn.pack(side="right", padx=2)
        self.lock_btn.bind("<Button-1>", lambda e: self._toggle_lock())
        
        logger.debug(f"Header created for {self.hole_id}")
    
    def _toggle_lock(self) -> None:
        """Toggle lock state and notify parent dialog."""
        # Call parent's toggle method if it exists
        parent_widget = self.master
        while parent_widget and not isinstance(parent_widget, tk.Canvas):
            parent_widget = parent_widget.master
        
        # Find the CorrelationDialog
        correlation_dialog = None
        check_widget = parent_widget
        while check_widget:
            if hasattr(check_widget, 'toggle_column_lock'):
                correlation_dialog = check_widget
                break
            check_widget = check_widget.master if hasattr(check_widget, 'master') else None
        
        if correlation_dialog:
            correlation_dialog.toggle_column_lock(self.hole_id)
        else:
            logger.warning("Could not find parent CorrelationDialog to toggle lock")
    
    def update_lock_visual(self, is_locked: bool) -> None:
        """Update lock button appearance based on state."""
        if hasattr(self, 'lock_btn'):
            self.lock_btn.config(
                text="🔒" if is_locked else "🔓",
                fg=self.theme_colors["accent_red"] if is_locked else self.theme_colors["text"]
            )
            
        # Update border color
        if hasattr(self, 'main_frame'):
            border_color = self.theme_colors["accent_red"] if is_locked else self.theme_colors["border"]
            border_width = 2 if is_locked else 1
            self.main_frame.config(
                highlightbackground=border_color,
                highlightcolor=border_color,
                highlightthickness=border_width,
            )
    
    def _create_canvas_area(self) -> None:
        """Create fixed-size canvas for intervals and data columns (no internal scrolling)."""
        # Calculate total width FIRST using get_total_width()
        total_width = self.get_total_width()
        
        # Canvas frame with explicit width
        canvas_frame = tk.Frame(
            self.main_frame,
            bg=self.theme_colors["field_bg"],
            highlightbackground=self.theme_colors["border"],
            highlightthickness=1,
            width=total_width
        )
        self.canvas_frame = canvas_frame
        canvas_frame.pack(fill="both", expand=True, padx=0, pady=0)
        canvas_frame.pack_propagate(False)  # Maintain explicit width
        
        # Create canvas without scrollbar - full display
        self.canvas = tk.Canvas(
            canvas_frame,
            bg=self.theme_colors["field_bg"],
            highlightthickness=0,
            width=total_width
        )
        
        # Pack canvas (no scrollbar)
        self.canvas.pack(fill="both", expand=True)
        
        # Bind configure event
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Bind column resize events
        self._bind_column_resize()
        
        # Bind right-click for context menu
        self.canvas.bind("<Button-3>", self._on_right_click)
        
        # Bind motion events to update cursor position in parent
        self.canvas.bind("<Motion>", self._on_canvas_motion, add="+")
        
        # Context menu state
        self.context_menu = None
        self.context_click_depth = None
        
        logger.debug(f"Canvas created: {total_width}px wide (no internal scrolling)")
    
    def _create_footer(self) -> None:
        """Create footer with status info."""
        footer_frame = tk.Frame(
            self.main_frame,
            bg=self.theme_colors["secondary_bg"],
            height=25
        )
        footer_frame.pack(fill="x", padx=1, pady=1)
        footer_frame.pack_propagate(False)
        
        # Status label
        self.status_label = tk.Label(
            footer_frame,
            text="Ready",
            font=self.fonts["small"],
            bg=self.theme_colors["secondary_bg"],
            fg=self.theme_colors["text"]
        )
        self.status_label.pack(side="left", padx=5)
        
        # Interval count
        self.count_label = tk.Label(
            footer_frame,
            text="0 intervals",
            font=self.fonts["small"],
            bg=self.theme_colors["secondary_bg"],
            fg=self.theme_colors["text"]
        )
        self.count_label.pack(side="right", padx=5)
    
    def _load_average_colors(self) -> None:
        """Load average colors from compartment_image_properties.json for color-bar mode."""
        if self.image_mode != "color":
            return
        
        logger.debug(f"Loading average colors for {self.hole_id}")
        
        try:
            # Get path to compartment_image_properties.json
            register_data_folder = self.config_manager.get("shared_register_data_folder", "")
            if not register_data_folder:
                logger.warning("No register data folder configured")
                return
            
            properties_file = os.path.join(register_data_folder, "compartment_image_properties.json")
            
            if not os.path.exists(properties_file):
                logger.warning(f"Properties file not found: {properties_file}")
                return
            
            import json
            with open(properties_file, 'r') as f:
                all_properties = json.load(f)
            
            # Get properties for this hole
            if self.hole_id in all_properties:
                hole_props = all_properties[self.hole_id]
                for depth_str, props in hole_props.items():
                    if "avg_color" in props:
                        depth = float(depth_str)
                        self.average_colors[depth] = props["avg_color"]
                
                logger.info(f"Loaded {len(self.average_colors)} average colors for {self.hole_id}")
            else:
                logger.debug(f"No properties found for {self.hole_id}")
                
        except Exception as e:
            logger.error(f"Error loading average colors: {e}", exc_info=True)
    
    def load_hole_data(self) -> None:
        """Load drillhole data from data manager."""
        logger.info(f"=" * 80)
        logger.info(f"LOADING DATA FOR HOLE: {self.hole_id}")
        logger.info(f"=" * 80)
        
        try:
            # Get interval data directly from geological store (fast path)
            # Prefer get_hole_intervals() for correlation display
            if hasattr(self.data_manager, 'get_hole_intervals'):
                hole_data = self.data_manager.get_hole_intervals(self.hole_id)
            elif hasattr(self.data_manager, 'build_dataframe_for_hole'):
                hole_data = self.data_manager.build_dataframe_for_hole(self.hole_id)
            else:
                # Legacy fallback
                hole_data = self.data_manager.get_data_for_hole(self.hole_id)
                if isinstance(hole_data, list):
                    logger.warning(f"get_data_for_hole returned list - cannot process")
                    hole_data = None
            
            if hole_data is None or (hasattr(hole_data, 'empty') and hole_data.empty):
                logger.error(f"❌ NO DATA FOUND for {self.hole_id}")
                self._create_placeholder_intervals(0, 100)  # Default 100m
                # CRITICAL: Still need to call _update_canvas to set dimensions
                self._update_canvas()
                return
            
            logger.info(f"✔ Loaded {len(hole_data)} rows from data manager")
            logger.info(f"  Available columns: {list(hole_data.columns)}")
            logger.info(f"  Data shape: {hole_data.shape}")
            
            # Find min/max depths
            depth_cols = self._find_depth_columns(hole_data)
            if not depth_cols:
                logger.error(f"No depth columns found for {self.hole_id}")
                self._create_placeholder_intervals(0, 100)  # Default 100m
                # CRITICAL: Still need to call _update_canvas to set dimensions
                self._update_canvas()
                return
            
            min_depth = hole_data[depth_cols["from"]].min()
            max_depth = hole_data[depth_cols["to"]].max()
            logger.info(f"Depth range for {self.hole_id}: {min_depth}m to {max_depth}m")
            
            # Create intervals
            self._create_intervals(hole_data, min_depth, max_depth, depth_cols)
            
            # Update canvas
            self._update_canvas()
            
            # Update status
            self.count_label.config(text=f"{len(self.intervals)} intervals")
            self.status_label.config(text=f"{min_depth:.1f}-{max_depth:.1f}m")
            
        except Exception as e:
            logger.error(f"Error loading hole data: {e}", exc_info=True)
            self.status_label.config(text="Error loading data")
    
    def _find_depth_columns(self, data: Any) -> Dict[str, str]:
        """Find depth column names in data."""
        logger.debug("Finding depth columns in data")
        
        # Common depth column patterns
        from_patterns = ["from", "depth_from", "from_depth", "from_m", "depthfrom"]
        to_patterns = ["to", "depth_to", "to_depth", "to_m", "depthto"]
        
        columns = data.columns.tolist()
        columns_lower = [c.lower() for c in columns]
        
        depth_cols = {}
        
        # Find FROM column
        for pattern in from_patterns:
            for i, col_lower in enumerate(columns_lower):
                if pattern in col_lower:
                    depth_cols["from"] = columns[i]
                    logger.debug(f"  Found FROM column: {columns[i]}")
                    break
            if "from" in depth_cols:
                break
        
        # Find TO column
        for pattern in to_patterns:
            for i, col_lower in enumerate(columns_lower):
                if pattern in col_lower:
                    depth_cols["to"] = columns[i]
                    logger.debug(f"  Found TO column: {columns[i]}")
                    break
            if "to" in depth_cols:
                break
        
        if len(depth_cols) != 2:
            logger.warning(f"Could not find depth columns. Found: {depth_cols}")
        
        return depth_cols
    
    def _create_intervals(
        self, 
        data: Any, 
        min_depth: Union[float, int], 
        max_depth: Union[float, int],
        depth_cols: Dict[str, str]
    ) -> None:
        """Create DrillholeInterval objects from data."""
        logger.info(f"")
        logger.info(f"CREATING INTERVALS FOR {self.hole_id}")
        logger.info(f"  Depth range: {min_depth}m to {max_depth}m")
        logger.info(f"  Depth columns: FROM={depth_cols.get('from')}, TO={depth_cols.get('to')}")
        logger.info(f"  Total intervals to create: {int(max_depth) - int(min_depth) + 1}")
        
        self.intervals = []
        
        # Build image lookup from data_manager's image index (O(1) lookups)
        # This replaces file-by-file searching with pre-indexed access
        image_lookup = self._build_image_lookup()
        
        # Round to integer meter boundaries
        start_depth = int(min_depth)
        end_depth = int(max_depth) + 1
        verbose_interval_logging = bool(
            self.config_manager.get("correlation_verbose_interval_logging", False)
        )
        
        # Create 1m intervals
        for depth in range(start_depth, end_depth):
            depth_from = float(depth)
            depth_to = float(depth + 1)
            
            # Find data for this interval
            interval_data = data[
                (data[depth_cols["from"]] <= depth_from) & 
                (data[depth_cols["to"]] >= depth_to)
            ]
            
            # Extract CSV data if available
            csv_data = {}
            if not interval_data.empty:
                # Take first matching row and normalize keys to lowercase for consistent lookup
                raw_data = interval_data.iloc[0].to_dict()
                # Store both original and lowercase keys for flexibility
                csv_data = {k.lower(): v for k, v in raw_data.items()}
                # Also add original casing
                csv_data.update(raw_data)
                if verbose_interval_logging:
                    logger.debug(f"  [OK] Depth {depth_from}-{depth_to}m: Found data with {len(raw_data)} columns")
            elif verbose_interval_logging:
                logger.debug(f"  [WARN] Depth {depth_from}-{depth_to}m: NO DATA FOUND")
            
            # Get images from pre-built lookup (depth_to is the key)
            image_paths = image_lookup.get(depth_to, {"wet": None, "dry": None, "generic": None})
            
            # Determine best paths (prefer dry, fall back to wet, then generic)
            wet_path = image_paths.get("wet")
            dry_path = image_paths.get("dry")
            generic_path = image_paths.get("generic")
            
            # If no moisture-specific images, use generic for both
            if not wet_path and not dry_path and generic_path:
                dry_path = generic_path  # Prefer to show as "dry"
            
            # Create interval
            interval = DrillholeInterval(
                hole_id=self.hole_id,
                depth_from=depth_from,
                depth_to=depth_to,
                image_path_wet=wet_path,
                image_path_dry=dry_path,
                csv_data=csv_data,
                is_placeholder=not (wet_path or dry_path)
            )
            
            self.intervals.append(interval)
            
            if verbose_interval_logging and (wet_path or dry_path):
                logger.debug(f"  [IMG] Interval {depth_from}-{depth_to}m: wet={bool(wet_path)}, dry={bool(dry_path)}")
            elif verbose_interval_logging:
                logger.debug(f"  [WARN] Interval {depth_from}-{depth_to}m: PLACEHOLDER (no images)")
        
        intervals_with_images = sum(1 for i in self.intervals if not i.is_placeholder)
        intervals_with_data = sum(1 for i in self.intervals if i.csv_data)
        logger.info(f"")
        logger.info(f"INTERVAL CREATION COMPLETE:")
        logger.info(f"  Total intervals: {len(self.intervals)}")
        logger.info(f"  With images: {intervals_with_images}")
        logger.info(f"  With CSV data: {intervals_with_data}")
        logger.info(f"=" * 80)
    
    def _build_image_lookup(self) -> Dict[float, Dict[str, Optional[str]]]:
        """
        Build a depth-keyed lookup of image paths using the data_manager's image index.
        
        This provides O(1) access to images by depth instead of file-by-file searching.
        Handles wet/dry variants and multiple images at same depth.
        
        Returns:
            Dict mapping depth_to -> {"wet": path, "dry": path, "generic": path}
        """
        lookup: Dict[float, Dict[str, Optional[str]]] = {}
        
        # Check if data_manager has image index access
        if not self.data_manager:
            logger.warning("No data_manager available for image lookup")
            return lookup
        
        try:
            # Get all images for this hole from the image index
            # This is a single call that returns all indexed images
            if hasattr(self.data_manager, 'get_images_for_hole'):
                images = self.data_manager.get_images_for_hole(self.hole_id)
            elif hasattr(self.data_manager, 'image_index'):
                images = self.data_manager.image_index.get_images_for_hole(self.hole_id)
            else:
                logger.warning("data_manager doesn't provide image access methods")
                return lookup
            
            if not images:
                logger.debug(f"No indexed images found for {self.hole_id}")
                return lookup
            
            logger.info(f"Building image lookup from {len(images)} indexed images for {self.hole_id}")
            
            # Build lookup by depth
            for img_info in images:
                depth_to = float(img_info.depth_to)
                moisture = img_info.moisture_status  # "Wet", "Dry", or None
                
                # Initialize entry for this depth if needed
                if depth_to not in lookup:
                    lookup[depth_to] = {"wet": None, "dry": None, "generic": None}
                
                # Assign path based on moisture status
                if moisture == "Wet":
                    # Only overwrite if not already set (prefer first found)
                    if lookup[depth_to]["wet"] is None:
                        lookup[depth_to]["wet"] = img_info.path
                elif moisture == "Dry":
                    if lookup[depth_to]["dry"] is None:
                        lookup[depth_to]["dry"] = img_info.path
                else:
                    # No moisture status = generic
                    if lookup[depth_to]["generic"] is None:
                        lookup[depth_to]["generic"] = img_info.path
            
            # Log summary
            depths_with_wet = sum(1 for v in lookup.values() if v["wet"])
            depths_with_dry = sum(1 for v in lookup.values() if v["dry"])
            depths_with_generic = sum(1 for v in lookup.values() if v["generic"])
            logger.info(f"  Image lookup built: {len(lookup)} depths, wet={depths_with_wet}, dry={depths_with_dry}, generic={depths_with_generic}")
            
            return lookup
            
        except Exception as e:
            logger.error(f"Error building image lookup: {e}", exc_info=True)
            return lookup
    
    def _create_placeholder_intervals(
        self, 
        min_depth: Union[float, int], 
        max_depth: Union[float, int]
    ) -> None:
        """Create placeholder intervals when no data available."""
        logger.info(f"Creating placeholder intervals for {self.hole_id}: {min_depth}-{max_depth}m")
        
        self.intervals = []
        
        for depth in range(int(min_depth), int(max_depth)):
            interval = DrillholeInterval(
                hole_id=self.hole_id,
                depth_from=float(depth),
                depth_to=float(depth + 1),
                is_placeholder=True
            )
            self.intervals.append(interval)
        
        logger.debug(f"Created {len(self.intervals)} placeholder intervals")
    
    def _update_canvas(self) -> None:
        """Update canvas with intervals."""
        logger.debug(f"Updating canvas for {self.hole_id}")
        
        # Clear existing items
        self._reset_thumbnail_state()
        self.canvas.delete("all")
        self.canvas_items.clear()
        self.image_cache.clear()
        
        if not self.intervals:
            logger.warning("No intervals to display")
            return
        
        # Calculate total height (header + transformed interval display)
        total_height = self.get_total_height()
        logger.debug(f"Total canvas height: {total_height}px (header={self.header_height}, {len(self.intervals)} intervals)")
        
        # Draw depth ruler if enabled
        if self.show_depth_ruler:
            self._draw_depth_ruler()
        
        # Draw image column based on mode
        if self.image_mode != "none":
            self._draw_image_column()
        
        # Draw data visualization columns
        if self.show_data_viz:
            self._draw_data_columns()
        
        # Set canvas height to display all content
        self.canvas.configure(height=total_height)
        
        # CRITICAL: Set the outer widget frame height so canvas window displays correctly
        # Without this, the widget collapses when placed in parent canvas via create_window()
        self.configure(height=total_height)
        self.main_frame.configure(height=total_height)
        
        # Load all intervals (no lazy loading)
        self._load_all_intervals()
        
        # Draw segments and discontinuities on top
        self._draw_segments_and_discontinuities()
        
        # Draw zoom window indicators if range is set
        self._draw_zoom_indicators()
    
    def get_total_height(self) -> int:
        """Get total height of the column in pixels."""
        if not self.intervals:
            return self.header_height
        return self.header_height + self._content_height_px()

    def _active_depth_transform(self) -> Optional[DepthTransform]:
        """Return the latest depth transform for this column, if any."""
        if not self.depth_transforms:
            return None
        return self.depth_transforms[-1]

    def _depth_min_max(self) -> Tuple[float, float]:
        """Return the displayed depth range."""
        if not self.intervals:
            return 0.0, 0.0
        return (
            float(min(interval.depth_from for interval in self.intervals)),
            float(max(interval.depth_to for interval in self.intervals)),
        )

    def _display_units_for_depth(self, depth: Union[float, int]) -> float:
        """Map a measured depth to transformed display units from column top."""
        depth = float(depth)
        depth_min, _depth_max = self._depth_min_max()
        transform = self._active_depth_transform()
        if transform is None:
            return max(0.0, depth - depth_min)

        top = transform.apply_transform(depth_min)
        return max(0.0, transform.apply_transform(depth) - top)

    def _depth_for_display_units(self, display_units: float) -> Optional[float]:
        """Invert transformed display units back to measured depth."""
        if not self.intervals:
            return None

        display_units = max(0.0, float(display_units))
        depth_min, depth_max = self._depth_min_max()
        transform = self._active_depth_transform()
        if transform is None:
            return max(depth_min, min(depth_max, depth_min + display_units))

        target = transform.apply_transform(depth_min) + display_units
        low = depth_min
        high = depth_max
        for _ in range(40):
            mid = (low + high) / 2.0
            if transform.apply_transform(mid) < target:
                low = mid
            else:
                high = mid
        return max(depth_min, min(depth_max, (low + high) / 2.0))

    def _content_height_px(self) -> int:
        """Return transformed content height below the header."""
        if not self.intervals:
            return 0
        _depth_min, depth_max = self._depth_min_max()
        units = self._display_units_for_depth(depth_max)
        return max(1, int(math.ceil(units * self.cell_height)))

    def _interval_content_bounds(self, interval: DrillholeInterval) -> Tuple[float, float]:
        """Return transformed Y bounds below the header for an interval."""
        y_top = self._display_units_for_depth(interval.depth_from) * self.cell_height
        y_bottom = self._display_units_for_depth(interval.depth_to) * self.cell_height
        if y_bottom < y_top:
            y_top, y_bottom = y_bottom, y_top
        if y_bottom - y_top < 1:
            y_bottom = y_top + 1
        return y_top, y_bottom

    def _interval_content_center(self, interval: DrillholeInterval) -> float:
        """Return transformed interval center below the header."""
        y_top, y_bottom = self._interval_content_bounds(interval)
        return (y_top + y_bottom) / 2.0
    
    def set_moisture_preference(self, preference: str) -> None:
        """
        Set the moisture preference for image display.
        
        Args:
            preference: "Wet" or "Dry"
        """
        if preference not in ("Wet", "Dry"):
            logger.warning(f"Invalid moisture preference: {preference}, using 'Dry'")
            preference = "Dry"
        
        if self.moisture_preference != preference:
            self.moisture_preference = preference
            logger.info(f"Moisture preference set to {preference} for {self.hole_id}")
            
            # Redraw to update images
            self._update_canvas()
    
    def render_to_image(self, scale_factor: int = 1) -> Optional[Image.Image]:
        """
        Render the column widget to a PIL Image for export.
        
        Args:
            scale_factor: Multiplier for resolution (1=normal, 4=high-res)
            
        Returns:
            PIL Image of the rendered column, or None on error
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Calculate dimensions
            width = self.get_total_width() * scale_factor
            height = self.get_total_height() * scale_factor
            
            if width <= 0 or height <= 0:
                logger.warning(f"Invalid dimensions for export: {width}x{height}")
                return None
            
            # Create image with background color
            bg_color = self.theme_colors.get("field_bg", "#2d2d2d")
            # Convert hex to RGB tuple
            bg_rgb = tuple(int(bg_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            
            img = Image.new("RGB", (width, height), color=bg_rgb)
            draw = ImageDraw.Draw(img)
            
            # Scale factors for drawing
            scaled_header_height = self.header_height * scale_factor
            scaled_cell_height = self.cell_height * scale_factor
            scaled_ruler_width = 30 * scale_factor if self.show_depth_ruler else 0
            scaled_thumbnail_width = self.thumbnail_width * scale_factor if self.image_mode == "thumbnail" else 0
            scaled_color_bar_width = self.color_bar_width * scale_factor if self.image_mode == "color" else 0
            scaled_data_column_width = self.data_column_width * scale_factor
            
            # Colors
            text_color = self.theme_colors.get("text", "#e0e0e0")
            border_color = self.theme_colors.get("border", "#3f3f3f")
            header_bg = self.theme_colors.get("secondary_bg", "#252526")
            
            # Convert hex colors to RGB
            text_rgb = tuple(int(text_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            border_rgb = tuple(int(border_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            header_rgb = tuple(int(header_bg.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            
            # Try to load a font, fall back to default
            try:
                font_size = 10 * scale_factor
                font = ImageFont.truetype("arial.ttf", font_size)
                small_font = ImageFont.truetype("arial.ttf", 8 * scale_factor)
            except Exception:
                font = ImageFont.load_default()
                small_font = font
            
            x_offset = 0
            
            # Draw depth ruler
            if self.show_depth_ruler:
                # Ruler header
                draw.rectangle(
                    [0, 0, scaled_ruler_width, scaled_header_height],
                    fill=header_rgb,
                    outline=border_rgb
                )
                draw.text(
                    (scaled_ruler_width // 2, scaled_header_height // 2),
                    "m",
                    fill=text_rgb,
                    font=small_font,
                    anchor="mm"
                )
                
                # Ruler body
                draw.rectangle(
                    [0, scaled_header_height, scaled_ruler_width, height],
                    fill=header_rgb,
                    outline=border_rgb
                )
                
                # Depth labels
                for i, interval in enumerate(self.intervals):
                    if int(interval.depth_to) % 5 == 0:
                        y_pos = scaled_header_height + (i + 1) * scaled_cell_height
                        draw.line(
                            [(scaled_ruler_width - 10 * scale_factor, y_pos), (scaled_ruler_width, y_pos)],
                            fill=text_rgb,
                            width=2 * scale_factor
                        )
                        draw.text(
                            (scaled_ruler_width - 15 * scale_factor, y_pos),
                            str(int(interval.depth_to)),
                            fill=text_rgb,
                            font=small_font,
                            anchor="rm"
                        )
                
                x_offset = scaled_ruler_width
            
            # Draw image column (thumbnails)
            if self.image_mode == "thumbnail" and scaled_thumbnail_width > 0:
                # Header
                draw.rectangle(
                    [x_offset, 0, x_offset + scaled_thumbnail_width, scaled_header_height],
                    fill=header_rgb,
                    outline=border_rgb
                )
                draw.text(
                    (x_offset + scaled_thumbnail_width // 2, scaled_header_height // 2),
                    "Image",
                    fill=text_rgb,
                    font=small_font,
                    anchor="mm"
                )
                
                # Draw thumbnails
                for i, interval in enumerate(self.intervals):
                    y_pos = scaled_header_height + i * scaled_cell_height
                    
                    # Get image based on moisture preference
                    image_path = self._get_preferred_image_path(interval)
                    
                    if image_path and os.path.exists(image_path):
                        try:
                            thumb = cv2.imread(image_path)
                            if thumb is not None:
                                thumb = cv2.rotate(thumb, cv2.ROTATE_90_CLOCKWISE)
                                thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                                thumb_pil = Image.fromarray(thumb)
                                
                                # Scale to cover the full cell, cropping overflow.
                                orig_w, orig_h = thumb_pil.size
                                new_w, new_h = self._cover_resize_dimensions(
                                    orig_w,
                                    orig_h,
                                    scaled_thumbnail_width,
                                    scaled_cell_height,
                                )
                                
                                thumb_pil = thumb_pil.resize(
                                    (new_w, new_h),
                                    Image.Resampling.LANCZOS
                                )

                                crop_x = max(0, (new_w - scaled_thumbnail_width) // 2)
                                crop_y = max(0, (new_h - scaled_cell_height) // 2)
                                thumb_pil = thumb_pil.crop((
                                    crop_x,
                                    crop_y,
                                    crop_x + scaled_thumbnail_width,
                                    crop_y + scaled_cell_height,
                                ))
                                
                                img.paste(thumb_pil, (x_offset, y_pos))
                        except Exception as e:
                            logger.debug(f"Error loading thumbnail for export: {e}")
                    else:
                        # Draw placeholder
                        row_color = self.row_color_even if i % 2 == 0 else self.row_color_odd
                        row_rgb = tuple(int(row_color.lstrip('#')[j:j+2], 16) for j in (0, 2, 4))
                        draw.rectangle(
                            [x_offset, y_pos, x_offset + scaled_thumbnail_width, y_pos + scaled_cell_height],
                            fill=row_rgb,
                            outline=border_rgb
                        )
                
                x_offset += scaled_thumbnail_width
            
            # Draw data columns
            if self.show_data_viz and self.viz_columns:
                for col_idx, viz_config in enumerate(self.viz_columns):
                    column_name = viz_config.get("column", "")
                    col_x = x_offset + col_idx * scaled_data_column_width
                    
                    # Header
                    draw.rectangle(
                        [col_x, 0, col_x + scaled_data_column_width, scaled_header_height],
                        fill=header_rgb,
                        outline=border_rgb
                    )
                    # Get display name for header
                    display_name = viz_config.get("display_name") or viz_config.get("custom_label") or column_name
                    if display_name == column_name:
                        display_name = self._create_column_display_name(column_name)
                    
                    # Truncate for export
                    max_export_chars = max(3, scaled_data_column_width // (8 * scale_factor))
                    header_text = self._truncate_text(display_name, max_export_chars)
                    
                    draw.text(
                        (col_x + scaled_data_column_width // 2, scaled_header_height // 2),
                        header_text,
                        fill=text_rgb,
                        font=small_font,
                        anchor="mm"
                    )
                    
                    # Draw alternating row backgrounds and data
                    for i, interval in enumerate(self.intervals):
                        y_pos = scaled_header_height + i * scaled_cell_height
                        
                        # Row background
                        row_color = self.row_color_even if i % 2 == 0 else self.row_color_odd
                        row_rgb = tuple(int(row_color.lstrip('#')[j:j+2], 16) for j in (0, 2, 4))
                        draw.rectangle(
                            [col_x, y_pos, col_x + scaled_data_column_width, y_pos + scaled_cell_height],
                            fill=row_rgb
                        )
                        
                        # Get data value and draw bar
                        value = self._get_interval_value(
                            interval,
                            column_name,
                            viz_config.get("source"),
                        )
                        if value is not None:
                            # Simple bar visualization
                            bar_color = self.theme_colors.get("accent_blue", "#3a7ca5")
                            bar_rgb = tuple(int(bar_color.lstrip('#')[j:j+2], 16) for j in (0, 2, 4))
                            
                            # Normalize value (assume 0-100 range for simplicity)
                            normalized = min(1.0, max(0.0, value / 100.0))
                            bar_width = int(normalized * (scaled_data_column_width - 10 * scale_factor))
                            
                            if bar_width > 0:
                                draw.rectangle(
                                    [col_x + 5 * scale_factor, y_pos + 2 * scale_factor,
                                     col_x + 5 * scale_factor + bar_width, y_pos + scaled_cell_height - 2 * scale_factor],
                                    fill=bar_rgb
                                )
            
            logger.info(f"Rendered {self.hole_id} to image: {width}x{height}px (scale={scale_factor})")
            return img
            
        except Exception as e:
            logger.error(f"Error rendering column to image: {e}", exc_info=True)
            return None
    
    def _get_interval_value(
        self,
        interval: DrillholeInterval,
        column_name: str,
        source_name: Optional[str] = None,
    ) -> Optional[float]:
        """Get numeric value from interval CSV data."""
        return self._get_value_for_interval(interval, column_name, source_name)
    
    def _get_preferred_image_path(self, interval: DrillholeInterval) -> Optional[str]:
        """
        Get image path based on current moisture preference.
        
        Falls back to available image if preferred isn't available.
        """
        if self.moisture_preference == "Dry":
            # Prefer dry, fall back to wet
            if interval.image_path_dry:
                return interval.image_path_dry
            elif interval.image_path_wet:
                return interval.image_path_wet
        else:
            # Prefer wet, fall back to dry
            if interval.image_path_wet:
                return interval.image_path_wet
            elif interval.image_path_dry:
                return interval.image_path_dry
        
        # Fall back to generic
        return interval.image_path if hasattr(interval, 'image_path') else None
    
    def get_total_width(self) -> int:
        """Get total width of the column in pixels."""
        width = 0
        
        # Add depth ruler width
        if self.show_depth_ruler:
            width += self.ruler_width  # Use instance variable (40px)
        
        # Add image column width based on mode
        if self.image_mode == "thumbnail":
            width += self.thumbnail_width
        elif self.image_mode == "color":
            width += self.color_bar_width
        # 'none' mode adds 0 width
        
        # Add data column widths
        if self.show_data_viz and self.viz_columns:
            width += len(self.viz_columns) * self.data_column_width
        
        return width

    def _get_data_columns_x_start(self) -> int:
        """Get the x position where data visualization columns begin."""
        x_start = 0
        if self.show_depth_ruler:
            x_start += self.ruler_width
        if self.image_mode == "thumbnail":
            x_start += self.thumbnail_width
        elif self.image_mode == "color":
            x_start += self.color_bar_width
        return x_start



    def _get_column_data_range(
        self, 
        column_name: str, 
        viz_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Get the min/max data range for a column.
        
        Args:
            column_name: Name of the data column
            viz_config: Optional visualization config with auto_scale, min_value, max_value
            
        Returns:
            Tuple of (min_value, max_value) or (None, None) if no data
        """
        source_name = viz_config.get("source") if viz_config else None
        values = self.get_viz_raw_values(column_name, source_name)
        if not values:
            return None, None

        scaled = self._scale_viz_values(values, viz_config)
        return scaled.plot_min, scaled.plot_max

    def _get_column_range_label(
        self,
        column_name: str,
        viz_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Return a compact range/scale label for a viz column header."""
        source_name = viz_config.get("source") if viz_config else None
        values = self.get_viz_raw_values(column_name, source_name)
        if not values:
            return None

        scaled = self._scale_viz_values(values, viz_config)
        if scaled.mode in {"per_hole_minmax", "per_hole_percentile", "zscore", "global"}:
            return scaled.label

        min_val = scaled.plot_min
        max_val = scaled.plot_max
        if abs(max_val) >= 100 or abs(min_val) >= 100:
            return f"{min_val:.0f}-{max_val:.0f}"
        if abs(max_val) >= 10 or abs(min_val) >= 10:
            return f"{min_val:.1f}-{max_val:.1f}"
        return f"{min_val:.2f}-{max_val:.2f}"

    def _scale_viz_values(
        self,
        values: List[float],
        viz_config: Optional[Dict[str, Any]] = None,
    ) -> TraceScaleResult:
        """Apply the configured trace scale mode to raw values."""
        viz_config = viz_config or {}
        return scale_trace_values(
            values,
            viz_config.get("scale_mode", "raw"),
            auto_scale=viz_config.get("auto_scale", True),
            min_value=viz_config.get("min_value"),
            max_value=viz_config.get("max_value"),
            global_min=viz_config.get("global_min_value"),
            global_max=viz_config.get("global_max_value"),
        )

    @staticmethod
    def _value_to_plot_x(value: float, scaled: TraceScaleResult, x_pos: int, width: int) -> int:
        """Convert a scaled value to a clamped x coordinate inside a viz column."""
        value_range = scaled.plot_max - scaled.plot_min
        if value_range == 0:
            normalized = 0.5
        else:
            normalized = (value - scaled.plot_min) / value_range
        normalized = max(0.0, min(1.0, normalized))
        return x_pos + int(normalized * (width - 10)) + 5

    def get_viz_raw_values(
        self,
        column_name: str,
        source_name: Optional[str] = None,
    ) -> List[float]:
        """Return raw finite values for a configured viz column."""
        point_values = self._get_point_values_for_viz_range(column_name, source_name)
        if point_values:
            return [value for _depth, value in point_values]

        values: List[float] = []
        for interval in self.intervals:
            value = self._get_value_for_interval(interval, column_name, source_name)
            if value is not None and not np.isnan(value):
                values.append(float(value))
        return values

    def _get_point_values_for_viz_range(
        self,
        column_name: str,
        source_name: Optional[str],
    ) -> List[Tuple[float, float]]:
        """Return raw point samples for the current hole/depth range."""
        if not self.intervals:
            return []
        if not hasattr(self.data_manager, "get_point_values_for_interval"):
            return []

        try:
            min_depth = min(float(interval.depth_from) for interval in self.intervals)
            max_depth = max(float(interval.depth_to) for interval in self.intervals)
            for source_candidate in self._viz_source_candidates(source_name):
                for column_candidate in self._column_lookup_candidates(column_name):
                    point_values = self.data_manager.get_point_values_for_interval(
                        self.hole_id,
                        min_depth,
                        max_depth,
                        column_candidate,
                        source_candidate,
                    )
                    if point_values:
                        return point_values
        except Exception as e:
            logger.debug(f"Point series lookup failed for {column_name} ({source_name}): {e}")
        return []

    def update_viz_columns(
        self,
        viz_columns: List[Dict[str, Any]],
        show_data_viz: Optional[bool] = None,
    ) -> None:
        """
        Update visualization column configuration and redraw the widget.

        Args:
            viz_columns: List of visualization column configs.
            show_data_viz: Optional override for whether graph columns are visible.
        """
        logger.debug(f"Updating viz columns for {self.hole_id}")
        self.viz_columns = viz_columns
        if show_data_viz is None:
            self.show_data_viz = self.show_data_columns and bool(viz_columns)
        else:
            self.show_data_viz = bool(show_data_viz) and bool(viz_columns)
        
        # Re-read display settings from config (they may have changed)
        self.cell_height = self.config_manager.get("correlation_data_viz_cell_height", 10)
        self.thumbnail_width = self.config_manager.get("correlation_thumbnail_width", 160)
        self.data_column_min_width = self._get_config_int("correlation_data_column_min_width", 50)
        self.data_column_max_width = self._get_config_int("correlation_data_column_max_width", 260)
        self.data_column_width = self._normalise_data_column_width(
            self._get_config_int("correlation_data_column_width", 70)
        )
        
        logger.debug(f"Updated display settings: cell_height={self.cell_height}, thumbnail_width={self.thumbnail_width}, data_col_width={self.data_column_width}")
        
        # Clear thumbnail cache since size may have changed
        if hasattr(self, 'thumbnail_refs'):
            self.thumbnail_refs.clear()
        
        # Recalculate total width using get_total_width()
        total_width = self.get_total_width()
        
        logger.debug(f"Updated total width to {total_width}px for {len(viz_columns)} columns")
        
        self._apply_total_width(total_width)
        
        # Update canvas scroll region for new cell height
        canvas_height = self.get_total_height()
        self.canvas.configure(scrollregion=(0, 0, total_width, canvas_height))
        
        # Redraw
        self._update_canvas()

    def set_data_viz_visible(self, visible: bool) -> None:
        """Show or hide data visualization columns without changing saved config."""
        self.show_data_viz = bool(visible) and bool(self.viz_columns)
        self.update_viz_columns(self.viz_columns, show_data_viz=self.show_data_viz)

    def _draw_interval_placeholder(self, interval: DrillholeInterval, y_pos: int) -> None:
        """Draw placeholder for interval (only if showing images)."""
        if not self.show_images:
            return  # Don't draw image placeholders if not showing images
        
        x_offset = 30 if self.show_depth_ruler else 0
        
        # Draw border
        self.canvas.create_rectangle(
            x_offset, y_pos,
            x_offset + self.cell_width, y_pos + self.cell_height,
            outline=self.theme_colors["border"],
            fill=self.theme_colors["field_bg"],
            tags=f"interval_{interval.interval_id}"
        )
        # Draw rectangle
        rect_id = self.canvas.create_rectangle(
            x_offset, y_pos,
            x_offset + self.cell_width, y_pos + self.cell_height,
            fill=self.theme_colors["field_bg"] if not interval.is_placeholder else "#2a2a2a",
            outline=self.theme_colors["border"],
            width=1,
            tags=(f"interval_{interval.interval_id}", "placeholder")
        )
        
        # Add depth text
        text_id = self.canvas.create_text(
            x_offset + self.cell_width // 2,
            y_pos + self.cell_height // 2,
            text=f"{interval.depth_from:.0f}-{interval.depth_to:.0f}m",
            fill=self.theme_colors["text"] if not interval.is_placeholder else "#666666",
            font=self.fonts["small"],
            tags=(f"interval_{interval.interval_id}", "depth_text")
        )
        
        self.canvas_items[interval.interval_id] = {"rect": rect_id, "text": text_id}
    
    def _draw_depth_ruler(self) -> None:
        """Draw depth scale on left side."""
        if not self.intervals:
            return
        
        logger.debug("Drawing depth ruler")
        
        # Draw ruler header (to align with data column headers)
        self.canvas.create_rectangle(
            0, 0, self.ruler_width, self.header_height,
            fill=self.theme_colors["secondary_bg"],
            outline=self.theme_colors["border"],
            width=1,
            tags="ruler_header"
        )
        
        self.canvas.create_text(
            self.ruler_width // 2, self.header_height // 2,
            text="m",
            fill=self.theme_colors["text"],
            font=self.fonts["small"],
            tags="ruler_header"
        )
        
        # Draw ruler background (below header)
        total_height = self._content_height_px()
        self.canvas.create_rectangle(
            0, self.header_height, self.ruler_width, self.header_height + total_height,
            fill=self.theme_colors["secondary_bg"],
            outline=self.theme_colors["border"],
            width=1,
            tags="ruler"
        )
        
        # Draw depth markers every 5m using transformed positions.
        depth_min, depth_max = self._depth_min_max()
        first_tick = int(math.ceil(depth_min / 5.0) * 5)
        last_tick = int(math.floor(depth_max / 5.0) * 5)
        for depth in range(first_tick, last_tick + 1, 5):
            y_base = self._depth_to_canvas_y(float(depth))
            if y_base is None:
                continue
            y_pos = self.header_height + y_base
            self.canvas.create_line(
                self.ruler_width - 10, y_pos, self.ruler_width, y_pos,
                fill=self.theme_colors["text"],
                width=2,
                tags="ruler"
            )
            self.canvas.create_text(
                self.ruler_width - 12, y_pos,
                text=f"{depth}",
                fill=self.theme_colors["text"],
                font=self.fonts["small"],
                anchor="e",
                tags="ruler"
            )
    
    def _draw_data_columns(self) -> None:
        """Draw data visualization columns."""
        if not self.show_data_viz or not self.viz_columns:
            logger.debug(f"[WARN] Not drawing data columns: show_data_viz={self.show_data_viz}, viz_columns={len(self.viz_columns) if self.viz_columns else 0}")
            return
        
        if not self.data_visualizer:
            logger.warning("[ERROR] No data_visualizer available for drawing data columns")
            return
        
        logger.debug(f"Drawing {len(self.viz_columns)} data columns for {self.hole_id}")
        logger.debug(f"  Intervals available: {len(self.intervals)}")
        for idx, col in enumerate(self.viz_columns):
            logger.debug(f"  Column {idx}: {col}")
        
        # Starting X position after ruler and image column.
        x_start = self._get_data_columns_x_start()
        
        # Calculate total width for alternating backgrounds
        total_data_width = len(self.viz_columns) * self.data_column_width
        
        # Draw alternating row backgrounds FIRST (behind data)
        for i, interval in enumerate(self.intervals):
            y_base_top, y_base_bottom = self._interval_content_bounds(interval)
            y_top = self.header_height + y_base_top
            y_bottom = self.header_height + y_base_bottom
            
            # Alternate colors
            row_color = self.row_color_even if i % 2 == 0 else self.row_color_odd
            
            self.canvas.create_rectangle(
                x_start, y_top,
                x_start + total_data_width, y_bottom,
                fill=row_color,
                outline="",  # No outline for subtle effect
                tags="row_background"
            )
        
        # Draw each data column
        for col_idx, viz_config in enumerate(self.viz_columns):
            column_name = viz_config.get("column")
            color_map_name = viz_config.get("color_map", "")
            viz_type = viz_config.get("type", "line")  # 'line' or 'bar'
            
            # Get display name (alias) for header - fall back to column name if not set
            display_name = viz_config.get("display_name") or viz_config.get("custom_label") or column_name
            # If no alias stored, try to create one
            if display_name == column_name:
                display_name = self._create_column_display_name(column_name)
            
            x_pos = x_start + (col_idx * self.data_column_width)
            
            # Draw column header
            self.canvas.create_rectangle(
                x_pos, 0,
                x_pos + self.data_column_width, self.header_height,
                fill=self.theme_colors["secondary_bg"],
                outline=self.theme_colors["border"],
                width=1,
                tags="data_column_header"
            )
            
            # Calculate max characters based on column width (approx 6px per char)
            max_chars = max(3, (self.data_column_width - 4) // 6)
            truncated_name = self._truncate_text(display_name, max_chars)
            
            # Draw column name in top portion of header
            self.canvas.create_text(
                x_pos + self.data_column_width // 2, 12,
                text=truncated_name,
                fill=self.theme_colors["text"],
                font=self.fonts["tiny"] if "tiny" in self.fonts else self.fonts["small"],
                width=0,  # Disable wrapping (0 = no wrap)
                tags="data_column_header"
            )
            
            # Calculate data range / scale label for the header.
            range_text = self._get_column_range_label(column_name, viz_config)
            if range_text:
                # Draw axis range in bottom portion of header
                self.canvas.create_text(
                    x_pos + self.data_column_width // 2, self.header_height - 10,
                    text=range_text,
                    fill=self.theme_colors["subtext"],
                    font=("Segoe UI", 7),
                    width=0,
                    tags="data_column_header"
                )
            
            # Draw column border (vertical line between columns)
            if col_idx > 0:
                self.canvas.create_line(
                    x_pos, self.header_height,
                    x_pos, self.header_height + self._content_height_px(),
                    fill=self.theme_colors["border"],
                    width=1,
                    tags="column_divider"
                )
            
            # Draw data for each interval
            if viz_type == "line":
                self._draw_line_graph(column_name, color_map_name, x_pos, self.data_column_width, viz_config)
            elif viz_type == "bar":
                self._draw_bar_chart(column_name, color_map_name, x_pos, self.data_column_width, viz_config)
    
    def _bind_column_resize(self) -> None:
        """Bind mouse events for resizing data columns."""
        if not self.show_data_viz:
            return
        
        self.canvas.bind("<ButtonPress-1>", self._on_column_resize_start)
        self.canvas.bind("<B1-Motion>", self._on_column_resize_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_column_resize_end)
        self.canvas.bind("<Motion>", self._on_column_hover)

    def is_on_data_column_resize_handle(self, x: int, y: int) -> bool:
        """Return True when a canvas coordinate is on a graph-column divider."""
        if not self.show_data_viz or not self.viz_columns:
            return False
        if y < self.header_height:
            return False

        x_start = self._get_data_columns_x_start()
        for col_idx in range(len(self.viz_columns)):
            col_x = x_start + (col_idx * self.data_column_width)
            divider_x = col_x + self.data_column_width
            if abs(x - divider_x) < 5:
                return True
        return False
    
    def _on_column_hover(self, event: tk.Event) -> None:
        """Change cursor when hovering over column dividers."""
        if not self.show_data_viz:
            return
        
        x = event.x
        y = event.y
        
        if self.is_on_data_column_resize_handle(x, y):
            self.canvas.config(cursor="sb_h_double_arrow")
            return
        
        self.canvas.config(cursor="")
    
    def _on_column_resize_start(self, event: tk.Event) -> None:
        """Start resizing a data column."""
        if not self.show_data_viz:
            return
        
        x = event.x
        y = event.y
        
        if y < self.header_height:
            return
        
        # Check which data-column divider was clicked after the ruler and image area.
        x_start = self._get_data_columns_x_start()
        
        for col_idx in range(len(self.viz_columns)):
            col_x = x_start + (col_idx * self.data_column_width)
            divider_x = col_x + self.data_column_width
            
            if abs(x - divider_x) < 5:
                self.resizing_column = col_idx
                self.resize_start_x = x
                self.canvas.config(cursor="sb_h_double_arrow")
                logger.debug(f"Started resizing column {col_idx}")
                return "break"
        return None
    
    def _on_column_resize_drag(self, event: tk.Event) -> None:
        """Handle column resize dragging."""
        if self.resizing_column is None:
            return
        
        delta_x = event.x - self.resize_start_x
        new_width = self.data_column_width + delta_x
        
        # Clamp to min/max
        new_width = self._normalise_data_column_width(new_width)
        
        if new_width != self.data_column_width:
            self.data_column_width = int(new_width)
            self.resize_start_x = event.x
            total_width = self.get_total_width()
            total_height = self.get_total_height()
            self._apply_total_width(total_width)
            self.canvas.configure(
                scrollregion=(0, 0, total_width, total_height),
            )
            
            # Redraw
            self._update_canvas()
            if callable(self.on_layout_changed):
                self.on_layout_changed()
            
            logger.debug(f"Resized column to {self.data_column_width}px")
        return "break"
    
    def _on_column_resize_end(self, event: tk.Event) -> None:
        """End column resizing."""
        if self.resizing_column is not None:
            logger.info(f"Finished resizing column {self.resizing_column} to {self.data_column_width}px")
            self.resizing_column = None
            self.resize_start_x = None
            self.canvas.config(cursor="")
            
            # Update config
            self.config_manager.set("correlation_data_column_width", self.data_column_width)
            return "break"
        return None

    def _draw_line_graph(
        self, 
        column_name: str, 
        color_map_name: str, 
        x_pos: int, 
        width: int,
        viz_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Draw line graph for a data column.
        
        Args:
            column_name: Name of data column to visualize
            color_map_name: Name of color map to apply
            x_pos: X position to start drawing
            width: Width of the column
        """
        logger.debug(f"Drawing line graph for {self.hole_id}: {column_name}")
        logger.debug(f"  X position: {x_pos}, Width: {width}")
        logger.debug(f"  Color map: {color_map_name}")
        
        if not self.intervals:
            logger.warning(f"  ❌ No intervals available")
            return
        
        source_name = viz_config.get("source") if viz_config else None
        point_values = self._get_point_values_for_viz_range(column_name, source_name)
        if point_values:
            self._draw_point_line_graph(
                point_values,
                column_name,
                color_map_name,
                x_pos,
                width,
                viz_config,
            )
            return

        # Collect data points
        data_points = []
        missing_count = 0
        invalid_count = 0
        
        for i, interval in enumerate(self.intervals):
            value = self._get_value_for_interval(interval, column_name, source_name)
            
            if value is None:
                missing_count += 1
                continue
            
            if np.isnan(value):
                invalid_count += 1
                continue
            
            # Calculate Y position (center of interval, offset by header)
            y_pos = self.header_height + self._interval_content_center(interval)
            data_points.append((value, y_pos))
        
        logger.debug(f"  Valid points: {len(data_points)}")
        logger.debug(f"  Missing column: {missing_count}")
        logger.debug(f"  Invalid/NaN values: {invalid_count}")
        
        if not data_points:
            logger.warning(f"  ❌ NO VALID DATA POINTS for column {column_name}")
            return
        
        logger.debug(f"  Drawing {len(data_points)} data points")
        
        values = [v for v, _ in data_points]
        scaled = self._scale_viz_values(values, viz_config)
        
        if not color_map_name or not self.color_map_manager:
            coords = []
            for (_raw_value, y_pos), plot_value in zip(data_points, scaled.values):
                coords.extend([self._value_to_plot_x(plot_value, scaled, x_pos, width), y_pos])
            if len(coords) >= 4:
                self.canvas.create_line(
                    *coords,
                    fill=self.theme_colors["accent_blue"],
                    width=2,
                    tags=f"data_line_{column_name}",
                )
            return

        # Draw line segments
        for i in range(len(data_points) - 1):
            _raw_value1, y1 = data_points[i]
            _raw_value2, y2 = data_points[i + 1]
            value1 = scaled.values[i]
            value2 = scaled.values[i + 1]
            x1 = self._value_to_plot_x(value1, scaled, x_pos, width)
            x2 = self._value_to_plot_x(value2, scaled, x_pos, width)
            
            # Get color from color map
            color = self._get_color_for_value(
                value1,
                column_name,
                color_map_name,
                scaled.plot_min,
                scaled.plot_max,
            )
            
            self.canvas.create_line(
                x1, y1, x2, y2,
                fill=color,
                width=2,
                tags=f"data_line_{column_name}"
            )

    def _draw_point_line_graph(
        self,
        point_values: List[Tuple[float, float]],
        column_name: str,
        color_map_name: str,
        x_pos: int,
        width: int,
        viz_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Draw raw point samples as a continuous line at their true depths."""
        values = [value for _, value in point_values]
        if len(values) < 2:
            return

        scaled = self._scale_viz_values(values, viz_config)
        line_points: List[Tuple[int, float, float]] = []
        for (depth, _raw_value), value in zip(point_values, scaled.values):
            y_base = self._depth_to_canvas_y(depth)
            if y_base is None:
                continue
            x = self._value_to_plot_x(value, scaled, x_pos, width)
            y = self.header_height + y_base
            line_points.append((x, y, value))

        if not color_map_name or not self.color_map_manager:
            coords = []
            for x, y, _value in line_points:
                coords.extend([x, y])
            if len(coords) >= 4:
                self.canvas.create_line(
                    *coords,
                    fill=self.theme_colors["accent_blue"],
                    width=1,
                    tags=f"data_point_line_{column_name}",
                )
            return

        for i in range(len(line_points) - 1):
            x1, y1, value1 = line_points[i]
            x2, y2, _value2 = line_points[i + 1]
            color = self._get_color_for_value(
                value1,
                column_name,
                color_map_name,
                scaled.plot_min,
                scaled.plot_max,
            )
            self.canvas.create_line(
                x1,
                y1,
                x2,
                y2,
                fill=color,
                width=1,
                tags=f"data_point_line_{column_name}",
            )
    
    def _draw_bar_chart(
        self, 
        column_name: str, 
        color_map_name: str, 
        x_pos: int, 
        width: int,
        viz_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Draw bar chart for a data column.
        
        Args:
            column_name: Name of data column to visualize
            color_map_name: Name of color map to apply
            x_pos: X position to start drawing
            width: Width of the column
            viz_config: Optional visualization configuration with auto_scale, min_value, max_value
        """
        logger.debug(f"Drawing bar chart for {self.hole_id}: {column_name}")
        logger.debug(f"  X position: {x_pos}, Width: {width}")
        logger.debug(f"  Color map: {color_map_name}")
        
        if not self.intervals:
            logger.warning(f"  ❌ No intervals available")
            return
        
        source_name = viz_config.get("source") if viz_config else None
        point_values = self._get_point_values_for_viz_range(column_name, source_name)
        if point_values:
            self._draw_point_bar_chart(
                point_values,
                column_name,
                color_map_name,
                x_pos,
                width,
                viz_config,
            )
            return

        # Collect all values once for both scaling and drawing.
        interval_values = []
        values = []
        missing_count = 0
        invalid_count = 0
        
        for i, interval in enumerate(self.intervals):
            value = self._get_value_for_interval(interval, column_name, source_name)
            
            if value is None:
                missing_count += 1
                continue
            
            if np.isnan(value):
                invalid_count += 1
                continue
            
            interval_values.append((i, value))
            values.append(value)
        
        logger.debug(f"  Valid values: {len(values)}")
        logger.debug(f"  Missing column: {missing_count}")
        logger.debug(f"  Invalid/NaN values: {invalid_count}")
        
        if not values:
            logger.warning(f"  ❌ NO VALID DATA for column {column_name}")
            return
        
        logger.debug(f"  Drawing {len(values)} bars")
        logger.debug(f"  Value range: {min(values):.2f} to {max(values):.2f}")
        
        scaled = self._scale_viz_values(values, viz_config)
        
        # Draw bars for each interval
        for (i, _raw_value), plot_value in zip(interval_values, scaled.values):
            try:
                # Calculate bar dimensions (account for header offset)
                y_base_top, y_base_bottom = self._interval_content_bounds(self.intervals[i])
                y_top = self.header_height + y_base_top
                y_bottom = self.header_height + y_base_bottom
                
                bar_width = self._value_to_plot_x(plot_value, scaled, x_pos, width) - (x_pos + 5)
                
                # Get color from color map
                color = self._get_color_for_value(
                    plot_value,
                    column_name,
                    color_map_name,
                    scaled.plot_min,
                    scaled.plot_max,
                )
                row_bg = self.row_color_even if i % 2 == 0 else self.row_color_odd
                fill_color = self._blend_color(color, row_bg, 0.42)
                
                # Draw bar
                self.canvas.create_rectangle(
                    x_pos + 5, y_top + 2,
                    x_pos + 5 + bar_width, y_bottom - 2,
                    fill=fill_color,
                    outline=self._blend_color(color, row_bg, 0.55),
                    stipple="gray75",
                    width=1,
                    tags=f"data_bar_{column_name}"
                )
                
            except (ValueError, TypeError):
                pass

    def _draw_point_bar_chart(
        self,
        point_values: List[Tuple[float, float]],
        column_name: str,
        color_map_name: str,
        x_pos: int,
        width: int,
        viz_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Draw raw point samples as sub-bars at their true downhole positions."""
        values = [value for _, value in point_values]
        if not values:
            return

        scaled = self._scale_viz_values(values, viz_config)
        bar_height = max(1.0, min(3.0, self.cell_height / 3.0))
        half_height = bar_height / 2.0

        logger.debug(f"  Drawing {len(point_values)} point sub-bars for {column_name}")

        for (depth, _raw_value), plot_value in zip(point_values, scaled.values):
            try:
                y_base = self._depth_to_canvas_y(depth)
                if y_base is None:
                    continue

                y_center = self.header_height + y_base
                bar_width = self._value_to_plot_x(plot_value, scaled, x_pos, width) - (x_pos + 5)
                color = self._get_color_for_value(
                    plot_value,
                    column_name,
                    color_map_name,
                    scaled.plot_min,
                    scaled.plot_max,
                )
                row_bg = self.row_color_even
                for index, interval in enumerate(self.intervals):
                    if interval.depth_from <= depth <= interval.depth_to:
                        row_bg = self.row_color_even if index % 2 == 0 else self.row_color_odd
                        break
                fill_color = self._blend_color(color, row_bg, 0.45)

                self.canvas.create_rectangle(
                    x_pos + 5,
                    y_center - half_height,
                    x_pos + 5 + bar_width,
                    y_center + half_height,
                    fill=fill_color,
                    outline="",
                    tags=f"data_point_bar_{column_name}",
                )
            except (ValueError, TypeError):
                continue
    
    def _draw_image_column(self) -> None:
        """Draw image column (thumbnails or color bars based on mode)."""
        if self.image_mode == "none" or not self.intervals:
            return
        
        logger.debug(f"Drawing image column in '{self.image_mode}' mode")
        
        # Starting X position after ruler.
        x_start = self.ruler_width if self.show_depth_ruler else 0
        
        if self.image_mode == "color":
            self._draw_color_bars(x_start)
        elif self.image_mode == "thumbnail":
            self._draw_thumbnails(x_start)
    
    def _draw_color_bars(self, x_start: int) -> None:
        """Draw color bars showing average image colors."""
        logger.debug(f"Drawing {len(self.intervals)} color bars")
        
        # Draw header for image column
        self.canvas.create_rectangle(
            x_start, 0,
            x_start + self.color_bar_width, self.header_height,
            fill=self.theme_colors["secondary_bg"],
            outline=self.theme_colors["border"],
            width=1,
            tags="image_column_header"
        )
        
        self.canvas.create_text(
            x_start + self.color_bar_width // 2, self.header_height // 2,
            text="Color",
            fill=self.theme_colors["text"],
            font=self.fonts["small"],
            tags="image_column_header"
        )
        
        for i, interval in enumerate(self.intervals):
            y_base_top, y_base_bottom = self._interval_content_bounds(interval)
            y_pos = self.header_height + y_base_top
            interval_height = max(1, y_base_bottom - y_base_top)
            
            # Get average color for this depth
            avg_color = self.average_colors.get(interval.depth_to, "#3a3a3a")
            
            # Alternating row background
            row_color = self.row_color_even if i % 2 == 0 else self.row_color_odd
            
            # Draw background first
            self.canvas.create_rectangle(
                x_start, y_pos,
                x_start + self.color_bar_width, y_pos + interval_height,
                fill=row_color,
                outline="",
                tags=f"color_bar_bg_{interval.interval_id}"
            )
            
            # Draw color rectangle (slightly inset for visual clarity)
            self.canvas.create_rectangle(
                x_start + 2, y_pos + 1,
                x_start + self.color_bar_width - 2, y_pos + interval_height - 1,
                fill=avg_color,
                outline=self.theme_colors["border"],
                width=1,
                tags=f"color_bar_{interval.interval_id}"
            )
        
        logger.debug(f"Drew {len(self.intervals)} color bars")
    
    def _draw_thumbnails(self, x_start: int) -> None:
        """Draw lightweight thumbnail placeholders and lazy-load visible images."""
        logger.debug(f"Drawing {len(self.intervals)} thumbnail placeholders")
        self._reset_thumbnail_state()
        self._thumbnail_x_start = x_start
        
        # Draw header for image column
        self.canvas.create_rectangle(
            x_start, 0,
            x_start + self.thumbnail_width, self.header_height,
            fill=self.theme_colors["secondary_bg"],
            outline=self.theme_colors["border"],
            width=1,
            tags="image_column_header"
        )
        
        self.canvas.create_text(
            x_start + self.thumbnail_width // 2, self.header_height // 2,
            text="Image",
            fill=self.theme_colors["text"],
            font=self.fonts["small"],
            tags="image_column_header"
        )
        
        for i, interval in enumerate(self.intervals):
            y_base_top, y_base_bottom = self._interval_content_bounds(interval)
            y_pos = self.header_height + y_base_top
            interval_height = max(1, y_base_bottom - y_base_top)
            
            # Alternating row background
            row_color = self.row_color_even if i % 2 == 0 else self.row_color_odd
            
            # Draw background first
            self.canvas.create_rectangle(
                x_start, y_pos,
                x_start + self.thumbnail_width, y_pos + interval_height,
                fill=row_color,
                outline="",
                tags=f"thumbnail_bg_{interval.interval_id}"
            )

            avg_color = self.average_colors.get(interval.depth_to, "#2a2a2a")
            placeholder_id = self.canvas.create_rectangle(
                x_start + 2, y_pos + 1,
                x_start + self.thumbnail_width - 2, y_pos + interval_height - 1,
                fill=avg_color,
                outline=self.theme_colors["border"],
                width=1,
                tags=f"thumbnail_{interval.interval_id}"
            )
            self.thumbnail_canvas_items[interval.interval_id] = placeholder_id

        logger.debug(
            f"Thumbnail placeholders ready for {self.hole_id}; "
            f"lazy loading visible images in {self.thumbnail_lazy_batch_size}-image batches"
        )
        self.load_visible_thumbnails(0, max(1000, self.canvas.winfo_height()))

    def _thumbnail_indices_for_visible_range(
        self,
        visible_top: Optional[float],
        visible_bottom: Optional[float],
        margin_px: Optional[int] = None,
    ) -> List[int]:
        """Return interval indices overlapping the visible thumbnail band."""
        if not self.intervals or self.cell_height <= 0:
            return []

        margin = self.thumbnail_lazy_margin_px if margin_px is None else max(0, int(margin_px))

        if visible_top is None or visible_bottom is None:
            visible_top = 0.0
            visible_bottom = float(max(1000, getattr(self.canvas, "winfo_height", lambda: 1000)()))

        top = min(float(visible_top), float(visible_bottom)) - margin
        bottom = max(float(visible_top), float(visible_bottom)) + margin

        if bottom < self.header_height:
            return []

        visible_start = top - self.header_height
        visible_end = bottom - self.header_height
        indices = []
        for index, interval in enumerate(self.intervals):
            y_top, y_bottom = self._interval_content_bounds(interval)
            if y_bottom >= visible_start and y_top <= visible_end:
                indices.append(index)
        return indices

    def load_visible_thumbnails(
        self,
        visible_top: Optional[float] = None,
        visible_bottom: Optional[float] = None,
    ) -> None:
        """Schedule lazy loading for thumbnail images near the current viewport."""
        if self.image_mode != "thumbnail" or not self.intervals:
            return

        if self._thumbnail_x_start is None:
            self._thumbnail_x_start = self.ruler_width if self.show_depth_ruler else 0

        if visible_top is None or visible_bottom is None:
            visible_top = 0.0
            visible_bottom = float(max(1000, self.canvas.winfo_height()))

        self._last_thumbnail_visible_range = (float(visible_top), float(visible_bottom))
        self._cancel_thumbnail_load()

        try:
            self.thumbnail_load_after_id = self.after_idle(self._load_visible_thumbnail_batch)
        except Exception:
            self._load_visible_thumbnail_batch()

    def _load_visible_thumbnail_batch(self) -> None:
        """Load one small batch of thumbnails for the last requested viewport."""
        self.thumbnail_load_after_id = None

        if self.image_mode != "thumbnail" or self._thumbnail_x_start is None:
            return

        indices = self._thumbnail_indices_for_visible_range(*self._last_thumbnail_visible_range)
        pending = [
            index for index in indices
            if index not in self.thumbnail_loaded_indices
            and index not in self.thumbnail_missing_indices
        ]

        if not pending:
            return

        batch = pending[:self.thumbnail_lazy_batch_size]
        loaded_count = 0
        missing_count = 0

        for index in batch:
            if self._load_thumbnail_index(index):
                loaded_count += 1
            else:
                missing_count += 1

        logger.debug(
            f"Lazy thumbnails for {self.hole_id}: loaded={loaded_count}, "
            f"missing={missing_count}, remaining_visible={max(0, len(pending) - len(batch))}"
        )

        if len(pending) > len(batch):
            try:
                self.thumbnail_load_after_id = self.after(1, self._load_visible_thumbnail_batch)
            except Exception:
                pass

    def _load_thumbnail_index(self, index: int) -> bool:
        """Load a single thumbnail into the image column."""
        if index < 0 or index >= len(self.intervals):
            return False

        interval = self.intervals[index]
        image_path = self._get_preferred_image_path(interval)
        if not image_path or not os.path.exists(image_path):
            self.thumbnail_missing_indices.add(index)
            return False

        try:
            img = cv2.imread(image_path)
            if img is None:
                self.thumbnail_missing_indices.add(index)
                return False

            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            h, w = img.shape[:2]

            y_base_top, y_base_bottom = self._interval_content_bounds(interval)
            interval_height = max(1, int(round(y_base_bottom - y_base_top)))

            new_w, new_h = self._cover_resize_dimensions(
                w,
                h,
                self.thumbnail_width,
                interval_height,
            )
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            crop_x = max(0, (new_w - self.thumbnail_width) // 2)
            crop_y = max(0, (new_h - interval_height) // 2)
            resized = resized[
                crop_y:crop_y + interval_height,
                crop_x:crop_x + self.thumbnail_width
            ]

            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(pil_img)

            y_pos = self.header_height + y_base_top
            image_id = self.canvas.create_image(
                self._thumbnail_x_start,
                y_pos,
                image=photo,
                anchor="nw",
                tags=f"thumbnail_image_{interval.interval_id}"
            )

            self.thumbnail_refs[interval.interval_id] = photo
            self.thumbnail_canvas_items[interval.interval_id] = image_id
            self.thumbnail_loaded_indices.add(index)
            return True

        except Exception as e:
            logger.debug(f"Error loading thumbnail for {interval.depth_to}m: {e}")
            self.thumbnail_missing_indices.add(index)
            return False

    def _get_color_for_value(
        self, 
        value: float, 
        column_name: str, 
        color_map_name: str, 
        min_val: float, 
        max_val: float
    ) -> str:
        """
        Get color for a value using color map.
        
        Args:
            value: Data value
            column_name: Column name
            color_map_name: Color map preset name
            min_val: Minimum value in dataset
            max_val: Maximum value in dataset
            
        Returns:
            Hex color string
        """
        if not self.color_map_manager or not color_map_name:
            # Default color
            return self.theme_colors["accent_blue"]
        
        try:
            # Get color map preset
            color_map = self.color_map_manager.get_preset(color_map_name)
            
            if not color_map:
                return self.theme_colors["accent_blue"]
            
            # Normalize value to 0-1 range
            normalized = (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5
            
            # Get color from map (returns BGR tuple)
            bgr_color = color_map.get_color(value)
            
            # Convert BGR to hex
            hex_color = f"#{bgr_color[2]:02x}{bgr_color[1]:02x}{bgr_color[0]:02x}"
            
            return hex_color
            
        except Exception as e:
            logger.debug(f"Error getting color for value: {e}")
            return self.theme_colors["accent_blue"]

    def _load_all_intervals(self) -> None:
        """No longer used - images are drawn directly in _draw_image_column."""
        # This method is kept for backward compatibility but does nothing
        # Images are now drawn in _draw_thumbnails() or _draw_color_bars()
        pass
    
    def _load_interval_image(self, index: int) -> None:
        """No longer used - images are drawn directly in _draw_image_column."""
        # This method is kept for backward compatibility but does nothing
        pass
        
        if index >= len(self.intervals):
            return
        
        interval = self.intervals[index]
        image_path = interval.get_best_image_path()
        
        if not image_path or not os.path.exists(image_path):
            return
        
        logger.debug(f"Loading image for interval {index}: {os.path.basename(image_path)}")
        
        try:
            # Load and resize image
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"Failed to load image: {image_path}")
                return
            
            # Resize to fit cell
            x_offset = 30 if self.show_depth_ruler else 0
            target_width = self.cell_width
            target_height = self.cell_height
            
            h, w = img.shape[:2]
            scale = min(target_width/w, target_height/h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Convert to PhotoImage
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(pil_img)
            
            # Cache the photo
            cache_key = f"{interval.interval_id}_{index}"
            self.image_cache[cache_key] = photo
            
            # Update canvas
            y_base_top, y_base_bottom = self._interval_content_bounds(interval)
            y_pos = y_base_top
            target_height = max(1, int(round(y_base_bottom - y_base_top)))
            
            # Delete placeholder text
            if interval.interval_id in self.canvas_items:
                if "text" in self.canvas_items[interval.interval_id]:
                    self.canvas.delete(self.canvas_items[interval.interval_id]["text"])
            
            # Add image
            img_id = self.canvas.create_image(
                x_offset + target_width // 2,
                y_pos + target_height // 2,
                image=photo,
                tags=(f"interval_{interval.interval_id}", "image")
            )
            
            if interval.interval_id not in self.canvas_items:
                self.canvas_items[interval.interval_id] = {}
            self.canvas_items[interval.interval_id]["image"] = img_id
            
            # Add moisture indicator
            if interval.moisture_status:
                moisture_id = self.canvas.create_text(
                    x_offset + target_width - 5,
                    y_pos + 5,
                    text=interval.moisture_status[0],  # W or D
                    fill="cyan" if interval.moisture_status == "Wet" else "orange",
                    font=("Arial", 8, "bold"),
                    anchor="ne",
                    tags=(f"interval_{interval.interval_id}", "moisture")
                )
                self.canvas_items[interval.interval_id]["moisture"] = moisture_id
            
        except Exception as e:
            logger.error(f"Error loading image for interval {index}: {e}")
    
    def _unload_interval_image(self, index: int) -> None:
        """Unload image to save memory."""
        if index >= len(self.intervals):
            return
        
        interval = self.intervals[index]
        cache_key = f"{interval.interval_id}_{index}"
        
        # Remove from cache
        if cache_key in self.image_cache:
            del self.image_cache[cache_key]
        
        # Remove from canvas and restore placeholder
        if interval.interval_id in self.canvas_items:
            items = self.canvas_items[interval.interval_id]
            if "image" in items:
                self.canvas.delete(items["image"])
                del items["image"]
            if "moisture" in items:
                self.canvas.delete(items["moisture"])
                del items["moisture"]
            
            # Restore depth text
            y_base_top, y_base_bottom = self._interval_content_bounds(interval)
            y_pos = y_base_top
            interval_height = max(1, y_base_bottom - y_base_top)
            x_offset = 30 if self.show_depth_ruler else 0
            
            text_id = self.canvas.create_text(
                x_offset + self.cell_width // 2,
                y_pos + interval_height / 2,
                text=f"{interval.depth_from:.0f}-{interval.depth_to:.0f}m",
                fill="#666666",
                font=self.fonts["small"],
                tags=(f"interval_{interval.interval_id}", "depth_text")
            )
            items["text"] = text_id
    
    def _on_canvas_configure(self, event: tk.Event) -> None:
        """Handle canvas resize."""
        # Update canvas size to match content
        if self.intervals:
            total_height = self.get_total_height()
            self.canvas.configure(height=total_height)
    
    def apply_depth_transform(self, transform: DepthTransform) -> None:
        """Apply a depth transformation to the display."""
        logger.info(
            f"Applying depth transform to {self.hole_id}: "
            f"{len(transform.stretch_regions)} region(s)"
        )
        
        # Keep one active manual/display transform per segment id.
        self.depth_transforms = [
            existing
            for existing in self.depth_transforms
            if existing.segment_id != transform.segment_id
        ]
        self.depth_transforms.append(transform)
        
        # Recalculate positions and redraw
        self._update_canvas()
    
    def get_depth_at_y(self, y_pos: int) -> Optional[float]:
        """Get depth value at a Y pixel position."""
        return self._canvas_y_to_depth(float(y_pos))
    
    def get_y_for_depth(self, depth: Union[float, int]) -> Optional[int]:
        """Get Y pixel position for a depth value."""
        depth = float(depth)
        
        y_pos = self._depth_to_canvas_y(depth)
        return int(y_pos) if y_pos is not None else None

    def _draw_segments_and_discontinuities(self):
        """Draw segment boundaries and discontinuity markers on canvas"""
        if not hasattr(self, 'segments') or not self.segments:
            return
        
        canvas_width = self.canvas.winfo_width()
        if canvas_width <= 0:
            canvas_width = self.width
        
        # Delete existing segment/discontinuity items
        self.canvas.delete("segment_boundary")
        self.canvas.delete("discontinuity_marker")
        self.canvas.delete("segment_label")
        
        logger.debug(f"Drawing {len(self.segments)} segments and {len(self.discontinuities)} discontinuities")
        
        # Draw segments
        for segment in self.segments:
            y_top = self._depth_to_canvas_y(segment.depth_from_original)
            y_bottom = self._depth_to_canvas_y(segment.depth_to_original)
            
            if y_top is None or y_bottom is None:
                continue
            
            # Draw segment boundary rectangle (subtle)
            self.canvas.create_rectangle(
                0, y_top,
                canvas_width, y_bottom,
                outline="#1976d2",
                width=2,
                dash=(5, 3),
                tags="segment_boundary"
            )
            
            # Draw segment label
            mid_y = (y_top + y_bottom) / 2
            label_text = f"Seg {segment.order_index + 1}"
            if segment.is_detrital:
                label_text += " (Det)"
            elif segment.is_lens:
                label_text += " (Lens)"
            
            self.canvas.create_text(
                canvas_width - 5, mid_y,
                text=label_text,
                anchor="e",
                font=("Segoe UI", 8),
                fill="#1976d2",
                tags="segment_label"
            )
        
        # Draw discontinuities
        for disc in self.discontinuities:
            y = self._depth_to_canvas_y(disc.depth_at_boundary)
            
            if y is None:
                continue
            
            # Draw discontinuity line
            line_width = disc.line_width
            if disc.is_lens:
                # Dashed line for lenses
                self.canvas.create_line(
                    -10, y,
                    canvas_width + 10, y,
                    fill=disc.color,
                    width=line_width,
                    dash=(8, 4),
                    tags="discontinuity_marker"
                )
            else:
                # Solid line for faults/intrusives
                self.canvas.create_line(
                    -10, y,
                    canvas_width + 10, y,
                    fill=disc.color,
                    width=line_width,
                    tags="discontinuity_marker"
                )
            
            # Draw discontinuity label
            label = disc.discontinuity_type.value[:4].upper()
    
    def set_zoom_window_range(self, depth_from: Optional[float], depth_to: Optional[float]) -> None:
        """
        Set the visible range in the zoom window for indicator display.
        
        Args:
            depth_from: Start depth of visible range (None to clear)
            depth_to: End depth of visible range (None to clear)
        """
        if depth_from is None or depth_to is None:
            self.zoom_window_depth_range = None
        else:
            self.zoom_window_depth_range = (depth_from, depth_to)
        
        # Redraw indicators
        self._draw_zoom_indicators()
    
    def _draw_zoom_indicators(self) -> None:
        """Draw subtle indicators showing which range is visible in zoom window."""
        # Clear existing indicators
        self.canvas.delete("zoom_indicator")
        
        if not self.zoom_window_depth_range or not self.intervals:
            return
        
        depth_from, depth_to = self.zoom_window_depth_range
        
        # Get Y positions for the range
        y_from = self._depth_to_canvas_y(depth_from)
        y_to = self._depth_to_canvas_y(depth_to)
        
        if y_from is None or y_to is None:
            return
        
        # Add header offset
        y_from += self.header_height
        y_to += self.header_height
        
        # Get canvas width
        canvas_width = self.canvas.winfo_width() or self.get_total_width()
        
        # Draw subtle highlight bar on the right edge
        indicator_width = 4
        indicator_color = self.theme_colors.get("accent_yellow", "#e5c07b")
        
        # Right edge indicator bar
        self.canvas.create_rectangle(
            canvas_width - indicator_width, y_from,
            canvas_width, y_to,
            fill=indicator_color,
            outline="",
            stipple="gray50",  # Semi-transparent
            tags="zoom_indicator"
        )
        
        # Top bracket
        self.canvas.create_line(
            canvas_width - 8, y_from,
            canvas_width, y_from,
            fill=indicator_color,
            width=2,
            tags="zoom_indicator"
        )
        
        # Bottom bracket
        self.canvas.create_line(
            canvas_width - 8, y_to,
            canvas_width, y_to,
            fill=indicator_color,
            width=2,
            tags="zoom_indicator"
        )
    
    def _depth_to_canvas_y(self, depth: float) -> Optional[float]:
        """Convert depth in meters to canvas Y coordinate"""
        if not self.intervals:
            return None

        depth_min, depth_max = self._depth_min_max()
        depth = float(depth)
        if depth < depth_min:
            return 0
        if depth > depth_max:
            return self._content_height_px()
        return self._display_units_for_depth(depth) * self.cell_height
    
    def update_segments_and_discontinuities(self, segments: List[DrillholeSegment], discontinuities: List[Discontinuity]):
        """Update segments and discontinuities from parent dialog"""
        self.segments = segments
        self.discontinuities = discontinuities
        
        # Redraw to show changes
        self._draw_segments_and_discontinuities()
        
        logger.info(f"Updated {self.hole_id}: {len(segments)} segments, {len(discontinuities)} discontinuities")

    def _on_right_click(self, event):
        """Handle right-click to show context menu"""
        # Get depth at click position
        canvas_y = self.canvas.canvasy(event.y)
        depth = self._canvas_y_to_depth(canvas_y)
        
        if depth is not None:
            self._show_context_menu(event.x_root, event.y_root, depth)
    
    def _on_canvas_motion(self, event):
        """Handle mouse motion to update cursor position in parent dialog"""
        # Get canvas Y coordinate
        canvas_y = self.canvas.canvasy(event.y)
        
        # Find the parent CorrelationDialog
        parent_widget = self.master
        while parent_widget:
            if hasattr(parent_widget, 'update_cursor_position'):
                parent_widget.update_cursor_position(canvas_y, self.hole_id)
                break
            parent_widget = parent_widget.master if hasattr(parent_widget, 'master') else None
    
    def _canvas_y_to_depth(self, canvas_y: float) -> Optional[float]:
        """Convert canvas Y coordinate to depth in meters"""
        if not self.intervals or self.cell_height <= 0:
            return None

        content_y = float(canvas_y)
        if content_y >= self.header_height:
            content_y -= self.header_height
        content_y = max(0.0, min(content_y, float(self._content_height_px())))
        return self._depth_for_display_units(content_y / self.cell_height)
    
    def _show_context_menu(self, x: int, y: int, depth: float):
        """Show context menu for adding discontinuities"""
        from gui.DrillholeCorrelation.correlation_models import DiscontinuityType
        
        if self.context_menu:
            self.context_menu.destroy()
        
        self.context_menu = tk.Menu(self, tearoff=0)
        self.context_click_depth = depth

        self.context_menu.add_command(
            label="Open Televiewer Here...",
            command=lambda d=depth: self._request_open_televiewer(d),
        )
        self.context_menu.add_separator()
        
        # Add Discontinuity submenu
        disc_menu = tk.Menu(self.context_menu, tearoff=0)
        
        discontinuity_types = [
            ("Fault", DiscontinuityType.FAULT),
            ("Intrusive Contact", DiscontinuityType.INTRUSIVE_CONTACT),
            ("Unconformity", DiscontinuityType.UNCONFORMITY),
            ("Detrital Sequence", DiscontinuityType.DETRITAL_SEQUENCE),
            ("Lens", DiscontinuityType.LENS),
            ("Core Loss", DiscontinuityType.CORE_LOSS),
            ("Weathering", DiscontinuityType.WEATHERING),
        ]
        
        for label, dtype in discontinuity_types:
            disc_menu.add_command(
                label=label,
                command=lambda d=depth, dt=dtype: self._request_add_discontinuity(d, dt)
            )
        
        self.context_menu.add_cascade(label="Add Discontinuity", menu=disc_menu)
        self.context_menu.add_separator()
        
        # Check if there's a discontinuity at this depth
        can_remove = self._has_discontinuity_near(depth)
        self.context_menu.add_command(
            label="Remove Discontinuity",
            command=lambda: self._request_remove_discontinuity(depth),
            state=tk.NORMAL if can_remove else tk.DISABLED
        )
        
        self.context_menu.add_separator()
        
        # Stretch/compress option
        segment = self._find_segment_at_depth(depth)
        self.context_menu.add_command(
            label="Edit Stretch Regions...",
            command=lambda: self._request_warp_bar(segment),
            state=tk.NORMAL if segment else tk.DISABLED
        )
        
        # Show menu
        self.context_menu.post(x, y)
    
    def _request_add_discontinuity(self, depth: float, discontinuity_type):
        """Request parent dialog to add discontinuity"""
        logger.info(f"Requesting to add {discontinuity_type.value} at {depth:.1f}m in {self.hole_id}")
        
        if self.on_discontinuity_add_requested:
            self.on_discontinuity_add_requested(self.hole_id, depth, discontinuity_type)
        else:
            logger.warning("No handler for on_discontinuity_add_requested")
    
    def _request_remove_discontinuity(self, depth: float):
        """Request parent dialog to remove discontinuity"""
        logger.info(f"Requesting to remove discontinuity at {depth:.1f}m in {self.hole_id}")
        
        if self.on_discontinuity_remove_requested:
            self.on_discontinuity_remove_requested(self.hole_id, depth)

    def _request_open_televiewer(self, depth: float):
        """Request parent dialog to open the televiewer viewer at this depth."""
        logger.info(f"Requesting televiewer open at {depth:.1f}m in {self.hole_id}")
        if self.on_televiewer_open_requested:
            self.on_televiewer_open_requested(self.hole_id, depth)
        else:
            logger.warning("No handler for on_televiewer_open_requested")
    
    def _request_warp_bar(self, segment):
        """Request parent dialog to show warp bar"""
        if segment and self.on_warp_bar_requested:
            self.on_warp_bar_requested(segment)
    
    def _has_discontinuity_near(self, depth: float, tolerance: float = 1.0) -> bool:
        """Check if there's a discontinuity near this depth"""
        for disc in self.discontinuities:
            if abs(disc.depth_at_boundary - depth) <= tolerance:
                return True
        return False
    
    def _find_segment_at_depth(self, depth: float):
        """Find segment containing this depth"""
        for segment in self.segments:
            if segment.contains_depth(depth):
                return segment
        return None
