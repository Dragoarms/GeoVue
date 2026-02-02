"""
CorrelationDialog - Main dialog for drillhole correlation.
Contains multiple DrillholeColumnWidgets with synchronized scrolling and tie lines.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging
from typing import Dict, List, Optional, Any, Tuple
import os
import sys

# Add parent directory to path for imports when running standalone
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import data models
from gui.DrillholeCorrelation.correlation_models import (
    CorrelationSession,
    TieLine,
    DepthTransform,
    TieLineType
)

# Import custom widgets
from gui.widgets.drillhole_column_widget import DrillholeColumnWidget
from gui.DrillholeCorrelation.correlation_models import (
    CorrelationSession,
    DrillholeSegment,
    Discontinuity,
    DiscontinuityType,
    Feature,
    FeatureCategory
)
from gui.widgets.modern_button import ModernButton
from gui.widgets.tie_line_canvas import TieLineCanvas
from gui.DrillholeCorrelation.synchronized_zoom_window import SynchronizedZoomWindow

# Import dialogs
from gui.DrillholeCorrelation.drillhole_selection_dialog import DrillholeSelectionDialog
from gui.DrillholeCorrelation.viz_column_settings_dialog import VizColumnSettingsDialog

# Import helper
from gui.dialog_helper import DialogHelper

logger = logging.getLogger(__name__)


class CorrelationDialog(tk.Toplevel):
    """
    Main dialog for drillhole correlation visualization and analysis.
    
    Features:
    - Multiple drillhole columns with synchronized scrolling
    - Interactive tie line creation between holes
    - Depth stretching/compression for dip correction
    - Data visualization columns
    - Save/load correlation sessions
    """
    
    def __init__(
        self,
        parent: Optional[tk.Widget],
        gui_manager: Any,
        data_manager: Any,
        file_manager: Any,
        config_manager: Any,
        color_map_manager: Optional[Any] = None,
        data_visualizer: Optional[Any] = None,
        translator: Optional[Any] = None,
        dialog_helper: Optional[Any] = None,
        initial_holes: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        test_mode: bool = False,
        modal: bool = False,
        **kwargs
    ):
        """
        Initialize CorrelationDialog.
        
        Args:
            parent: Parent window (can be None for standalone)
            gui_manager: REQUIRED - GUIManager for theming
            data_manager: REQUIRED - DrillholeDataManager for data
            file_manager: REQUIRED - FileManager for paths
            config_manager: REQUIRED - ConfigManager for settings
            color_map_manager: Optional - ColorMapManager for colors
            translator: Optional - Translator for i18n
            dialog_helper: Optional - DialogHelper for dialogs
            initial_holes: List of hole IDs to load initially
            session_id: Session ID for loading saved session
            test_mode: Whether running in test mode
        """
        logger.info("Initializing CorrelationDialog")
        logger.debug(f"  Test mode: {test_mode}")
        logger.debug(f"  Initial holes: {initial_holes}")
        logger.debug(f"  Session ID: {session_id}")
        
        # Validate managers
        if not all([gui_manager, data_manager, file_manager, config_manager]):
            logger.error("Missing required managers!")
            raise ValueError("All managers (gui, data, file, config) are required!")
        
        # Initialize dialog
        if test_mode and parent is None:
            # Create root for test mode
            self.test_root = tk.Tk()
            self.test_root.withdraw()
            parent = self.test_root
        else:
            self.test_root = None
        
        super().__init__(parent, **kwargs)
        
        # Store managers
        self.gui_manager = gui_manager
        self.data_manager = data_manager
        self.file_manager = file_manager
        self.config_manager = config_manager
        self.color_map_manager = color_map_manager
        self.translator = translator or self._create_dummy_translator()
        self.dialog_helper = dialog_helper or DialogHelper(self, translator=self.translator, gui_manager=gui_manager)
        
        # Create data visualizer if not provided
        if data_visualizer is None:
            try:
                from processing.LoggingReviewStep.drillhole_data_visualizer import DrillholeDataVisualizer
                self.data_visualizer = DrillholeDataVisualizer()
                logger.info("Created DrillholeDataVisualizer")
            except Exception as e:
                logger.warning(f"Could not create DrillholeDataVisualizer: {e}")
                self.data_visualizer = None
        else:
            self.data_visualizer = data_visualizer
        
        # Theme shortcuts
        self.theme_colors = gui_manager.theme_colors
        self.fonts = gui_manager.fonts
        
        # Dialog properties
        self.test_mode = test_mode
        self.parent = parent
        
        # Configure dialog
        self.title(self.translator.translate("Drillhole Correlation"))
        dialog_width = config_manager.get("correlation_dialog_width", 1200)
        dialog_height = config_manager.get("correlation_dialog_height", 800)
        self.geometry(f"{dialog_width}x{dialog_height}")
        self.configure(bg=self.theme_colors["background"])
        
        # Session data
        self.session = CorrelationSession(
            session_id=session_id or self._generate_session_id(),
            hole_ids=initial_holes or [],
            viz_columns=config_manager.get("viz_columns", [])
        )
        
        # Column widgets
        self.column_widgets: Dict[str, DrillholeColumnWidget] = {}
        self.column_frames: Dict[str, tk.Frame] = {}
        
        # Tie lines
        self.tie_lines: List[TieLine] = []
        self.tie_line_canvas: Optional[TieLineCanvas] = None
        
        # Interaction state
        self.creating_tie_line = False
        self.tie_line_start: Optional[Tuple[str, float]] = None  # (hole_id, depth)
        self.selected_region: Optional[Tuple[str, float, float]] = None  # (hole_id, from_depth, to_depth)
        
        # Column positioning state
        self.column_locked: Dict[str, bool] = {}  # Track lock state per column
        self.dragging_column: Optional[str] = None  # Currently dragging column
        self.drag_mode: Optional[str] = None  # 'horizontal' or 'vertical'
        
        # Synchronized zoom window (optional floating window)
        self.zoom_window = None
        self.zoom_window_visible = False
        
        # Moisture preference for image display (Wet/Dry toggle)
        self.moisture_preference = "Dry"  # Default to Dry images

        # Build UI
        self._create_ui()
        
        # Load initial holes
        if initial_holes:
            for hole_id in initial_holes:
                self.add_hole(hole_id)
        
        # Load session if provided
        if session_id:
            self.load_session(session_id)
        
        # Configure window behavior
        if not test_mode:
            if modal:
                # Only make modal if explicitly requested
                self.transient(parent)
                self.grab_set()
                # Center dialog
                self._center_dialog()
            else:
                # Non-modal: maximize window
                self.state('zoomed')  # Windows
                # For cross-platform:
                try:
                    self.attributes('-zoomed', True)  # Linux
                except:
                    pass
        else:
            self._center_dialog()
        
        # Bind close event
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        
        logger.info("CorrelationDialog initialized successfully")
        
        # Auto-open drillhole selection dialog if no initial holes
        if not initial_holes and not test_mode:
            self.after(100, self._show_hole_selection)  # Delay slightly to let dialog render
    
    def _create_dummy_translator(self) -> Any:
        """Create dummy translator for standalone mode."""
        class DummyTranslator:
            def translate(self, text: str) -> str:
                return text
        return DummyTranslator()
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _create_ui(self) -> None:
        """Create the dialog UI."""
        logger.debug("Creating CorrelationDialog UI")
        
        # Main container
        main_frame = tk.Frame(self, bg=self.theme_colors["background"])
        main_frame.pack(fill="both", expand=True)
        
        # Toolbar
        self._create_toolbar(main_frame)
        
        # Content area
        self._create_content_area(main_frame)
        
        # Status bar
        self._create_status_bar(main_frame)
        
        logger.debug("UI creation complete")
    
    def _create_toolbar(self, parent: tk.Widget) -> None:
        """Create toolbar with controls."""
        toolbar = tk.Frame(
            parent,
            bg=self.theme_colors["secondary_bg"],
            height=40
        )
        toolbar.pack(fill="x", padx=2, pady=2)
        toolbar.pack_propagate(False)
        
        # Add/Remove holes
        ModernButton(
            toolbar,
            text="➕ " + self.translator.translate("Add Holes"),
            command=self._show_hole_selection,
            color=self.theme_colors["accent_green"],
            theme_colors=self.theme_colors
        ).pack(side="left", padx=2, pady=5)
        
        ModernButton(
            toolbar,
            text="➖ " + self.translator.translate("Remove Hole"),
            command=self._remove_selected_hole,
            color=self.theme_colors["accent_red"],
            theme_colors=self.theme_colors
        ).pack(side="left", padx=2, pady=5)
        
        # Separator
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=10)
        
        # Tie line controls
        self.tie_line_btn = ModernButton(
            toolbar,
            text="🔗 " + self.translator.translate("Create Tie Line"),
            command=self._toggle_tie_line_mode,
            color=self.theme_colors["accent_blue"],
            theme_colors=self.theme_colors
        )
        self.tie_line_btn.pack(side="left", padx=2, pady=5)
        
        
        ModernButton(
            toolbar,
            text="✂️ " + self.translator.translate("Clear Tie Lines"),
            command=self._clear_tie_lines,
            color=self.theme_colors["accent_yellow"],
            theme_colors=self.theme_colors
        ).pack(side="left", padx=2, pady=5)
        
        # Separator
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=10)
        
        # Visualization settings
        ModernButton(
            toolbar,
            text="📊 " + self.translator.translate("Configure Data Viz"),
            command=self._show_viz_settings,
            color=self.theme_colors["accent_blue"],
            theme_colors=self.theme_colors
        ).pack(side="left", padx=2, pady=5)
        
        # Toggle zoom window
        self.zoom_window_btn = ModernButton(
            toolbar,
            text="🔍 " + self.translator.translate("Zoom Window"),
            command=self._toggle_zoom_window,
            color=self.theme_colors["accent_blue"],
            theme_colors=self.theme_colors
        )
        self.zoom_window_btn.pack(side="left", padx=2, pady=5)
        
        # Stretch/Compress
        ModernButton(
            toolbar,
            text="↕️ " + self.translator.translate("Stretch/Compress"),
            command=self._show_stretch_dialog,
            color=self.theme_colors["accent_blue"],
            theme_colors=self.theme_colors
        ).pack(side="left", padx=2, pady=5)
        
        # Separator
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=10)
        
        # Wet/Dry Toggle
        self.moisture_toggle_btn = ModernButton(
            toolbar,
            text="💧 Dry",  # Will update text based on state
            command=self._toggle_moisture_preference,
            color=self.theme_colors["accent_blue"],
            theme_colors=self.theme_colors
        )
        self.moisture_toggle_btn.pack(side="left", padx=2, pady=5)
        self._update_moisture_button_text()
        
        # Export Menu Button
        self.export_btn = ModernButton(
            toolbar,
            text="📷 " + self.translator.translate("Export"),
            command=self._show_export_menu,
            color=self.theme_colors["accent_green"],
            theme_colors=self.theme_colors
        )
        self.export_btn.pack(side="left", padx=2, pady=5)
        
        # Right side controls
        ModernButton(
            toolbar,
            text="💾 " + self.translator.translate("Save Session"),
            command=self._save_session,
            color=self.theme_colors["accent_green"],
            theme_colors=self.theme_colors
        ).pack(side="right", padx=2, pady=5)
        
        ModernButton(
            toolbar,
            text="📁 " + self.translator.translate("Load Session"),
            command=self._load_session,
            color=self.theme_colors["accent_blue"],
            theme_colors=self.theme_colors
        ).pack(side="right", padx=2, pady=5)
    
    def _create_content_area(self, parent: tk.Widget) -> None:
        """Create main content area with independently positioned columns."""
        # Main canvas container (no paned window needed)
        canvas_container = tk.Frame(
            parent,
            bg=self.theme_colors["field_bg"],
            highlightbackground=self.theme_colors["border"],
            highlightthickness=1
        )
        canvas_container.pack(fill="both", expand=True, padx=2, pady=2)
        
        # Create main scrollable canvas for stratigraphic correlation
        self.canvas = tk.Canvas(
            canvas_container,
            bg=self.theme_colors["field_bg"],
            highlightthickness=0
        )
        
        # Scrollbars - both horizontal and vertical
        h_scrollbar = ttk.Scrollbar(
            canvas_container,
            orient="horizontal",
            command=self.canvas.xview
        )
        self.v_scrollbar = ttk.Scrollbar(
            canvas_container,
            orient="vertical",
            command=self.canvas.yview  # Direct scroll, no sync
        )
        
        # Configure canvas
        self.canvas.configure(
            xscrollcommand=h_scrollbar.set,
            yscrollcommand=self.v_scrollbar.set
        )
        
        # Pack scrollbars and canvas
        h_scrollbar.pack(side="bottom", fill="x")
        self.v_scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # (Canvas container already packed above - no paned window)
        
        # Initialize zoom and positioning
        self.zoom_level = 1.0
        self.column_positions = {}  # {hole_id: (x, y)}
        self.column_windows = {}  # {hole_id: canvas_window_id}
        self.next_x_position = 10  # Starting X position for columns
        self.column_width = self.config_manager.get("correlation_column_width", 200)
        self.column_spacing = 5  # Gap between columns (compact layout)
        
        # Set large initial scroll region for stratigraphic space
        # This will be updated as columns are added
        self.canvas.configure(scrollregion=(0, 0, 2000, 20000))
        
        # Bind canvas events
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", self._on_mousewheel)  # Linux scroll up
        self.canvas.bind("<Button-5>", self._on_mousewheel)  # Linux scroll down
        self.canvas.bind("<Control-MouseWheel>", self._on_zoom)
        self.canvas.bind("<Control-Button-4>", self._on_zoom)  # Linux zoom in
        self.canvas.bind("<Control-Button-5>", self._on_zoom)  # Linux zoom out
        self.canvas.bind("<Motion>", self._on_mouse_motion)  # Track cursor for zoom window
        
        # Create tie line overlay canvas
        self._create_tie_line_overlay()
    
    def _toggle_zoom_window(self) -> None:
        """Toggle visibility of the zoom window."""
        if self.zoom_window_visible:
            self._hide_zoom_window()
        else:
            self._show_zoom_window()
    
    def _show_zoom_window(self) -> None:
        """Show the synchronized zoom window."""
        logger.info("Showing zoom window")
        
        if self.zoom_window is None:
            # Create zoom window
            self.zoom_window = SynchronizedZoomWindow(
                parent=self,
                gui_manager=self.gui_manager,
                config_manager=self.config_manager,
                color_map_manager=self.color_map_manager
            )
            
            # Add all existing drillholes
            for hole_id, column_widget in self.column_widgets.items():
                self.zoom_window.add_drillhole(hole_id, column_widget)
        
        # Show window
        self.zoom_window.deiconify()
        self.zoom_window_visible = True
        self.zoom_window_btn.config(text="🔍 " + self.translator.translate("Hide Zoom"))
        
        # Update with current cursor position
        self._update_zoom_window_position()
        
        logger.debug("Zoom window shown")
    
    def _hide_zoom_window(self) -> None:
        """Hide the zoom window."""
        logger.info("Hiding zoom window")
        
        if self.zoom_window:
            self.zoom_window.withdraw()
        
        self.zoom_window_visible = False
        self.zoom_window_btn.config(text="🔍 " + self.translator.translate("Zoom Window"))
        
        # Clear zoom indicators on all column widgets
        for hole_id, widget in self.column_widgets.items():
            widget.set_zoom_window_range(None, None)
        
        logger.debug("Zoom window hidden")
    
    def _update_zoom_window_position(self) -> None:
        """Update zoom window with current cursor positions."""
        if not self.zoom_window_visible or not self.zoom_window:
            return
        
        # Check if zoom window still exists (may have been destroyed)
        try:
            if not self.zoom_window.winfo_exists():
                self.zoom_window = None
                self.zoom_window_visible = False
                return
        except tk.TclError:
            self.zoom_window = None
            self.zoom_window_visible = False
            return
        
        # Calculate depth for each hole at current mouse position
        depths_by_hole = {}
        
        for hole_id, column_widget in self.column_widgets.items():
            if hole_id not in self.column_positions:
                continue
            
            # Get last known mouse position
            if not hasattr(self, '_last_canvas_y'):
                continue
            
            col_x, col_y = self.column_positions[hole_id]
            relative_y = self._last_canvas_y - col_y
            
            # Convert to depth
            depth = column_widget.get_depth_at_y(int(relative_y))
            if depth is not None:
                depths_by_hole[hole_id] = depth
        
        if depths_by_hole:
            self.zoom_window.update_position(depths_by_hole)
    
    def update_cursor_position(self, canvas_y: float, hole_id: str) -> None:
        """
        Update cursor position from a child drillhole column widget.
        
        Called by DrillholeColumnWidget._on_canvas_motion when mouse moves over column.
        This ensures zoom window updates even when hovering over column widgets.
        
        Args:
            canvas_y: Y coordinate on the column's canvas
            hole_id: ID of the hole being hovered
        """
        # Store last position 
        self._last_canvas_y = canvas_y
        self._last_hover_hole = hole_id
        
        # Update zoom window if visible
        if self.zoom_window_visible and self.zoom_window:
            # Check if zoom window still exists
            try:
                if not self.zoom_window.winfo_exists():
                    self.zoom_window = None
                    self.zoom_window_visible = False
                    return
            except tk.TclError:
                self.zoom_window = None
                self.zoom_window_visible = False
                return
            
            # Get the column widget
            if hole_id in self.column_widgets:
                column_widget = self.column_widgets[hole_id]
                
                # Calculate depth at this Y position
                depth = column_widget._canvas_y_to_depth(canvas_y - column_widget.header_height)
                
                if depth is not None:
                    # Update zoom window for all holes at this equivalent depth
                    depths_by_hole = {}
                    for hid, widget in self.column_widgets.items():
                        depths_by_hole[hid] = depth
                    
                    self.zoom_window.update_position(depths_by_hole)
                    
                    # Update zoom indicators on all column widgets
                    for hid, widget in self.column_widgets.items():
                        # Calculate visible range based on zoom window settings
                        intervals_before = self.config_manager.get("zoom_window_intervals_before", 3)
                        intervals_after = self.config_manager.get("zoom_window_intervals_after", 3)
                        range_start = depth - intervals_before
                        range_end = depth + intervals_after + 1
                        widget.set_zoom_window_range(range_start, range_end)
    
    def _on_mouse_motion(self, event: tk.Event) -> None:
        """Handle mouse motion for zoom window cursor tracking."""
        # Convert event coordinates to canvas coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Store last position for zoom window updates
        self._last_canvas_y = canvas_y
        
        # Update zoom window if visible
        if self.zoom_window_visible:
            self._update_zoom_window_position()

    def _create_tie_line_overlay(self) -> None:
        """Create transparent overlay for tie lines."""
        # This will be implemented in tie_line_canvas.py
        logger.debug("Tie line overlay placeholder created")
    
    def _on_mousewheel(self, event: tk.Event) -> None:
        """Handle mouse wheel scrolling (vertical scroll)."""
        if event.delta:
            # Windows/MacOS
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        elif event.num == 4:
            # Linux scroll up
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            # Linux scroll down
            self.canvas.yview_scroll(1, "units")
        
        # Update zoom window after scroll
        if self.zoom_window_visible:
            self._update_zoom_window_position()
    
    def _on_zoom(self, event: tk.Event) -> None:
        """Handle Ctrl+MouseWheel zoom."""
        # Determine zoom direction
        if event.delta:
            # Windows/MacOS
            zoom_delta = 1 if event.delta > 0 else -1
        elif event.num == 4:
            # Linux zoom in
            zoom_delta = 1
        elif event.num == 5:
            # Linux zoom out
            zoom_delta = -1
        else:
            return
        
        # Calculate new zoom level
        zoom_factor = 1.1 if zoom_delta > 0 else 0.9
        new_zoom = self.zoom_level * zoom_factor
        
        # Clamp zoom level
        min_zoom = 0.1
        max_zoom = 5.0
        new_zoom = max(min_zoom, min(max_zoom, new_zoom))
        
        if new_zoom != self.zoom_level:
            logger.debug(f"Zooming from {self.zoom_level:.2f} to {new_zoom:.2f}")
            self._apply_zoom(new_zoom)
    
    def _apply_zoom(self, new_zoom: float) -> None:
        """Apply zoom transformation to all columns (scale positions AND sizes)."""
        old_zoom = self.zoom_level
        self.zoom_level = new_zoom
        zoom_ratio = new_zoom / old_zoom
        
        # Scale all column positions
        for hole_id, (x, y) in self.column_positions.items():
            new_x = x * zoom_ratio
            new_y = y * zoom_ratio
            self.column_positions[hole_id] = (new_x, new_y)
            
            # Update canvas window position
            if hole_id in self.column_windows:
                self.canvas.coords(self.column_windows[hole_id], new_x, new_y)
            
            # Scale the widget itself
            if hole_id in self.column_widgets:
                widget = self.column_widgets[hole_id]
                
                # Scale cell height
                new_cell_height = int(widget.cell_height * zoom_ratio)
                widget.cell_height = max(10, new_cell_height)  # Minimum 10px
                
                # Redraw with new scale
                widget._update_canvas()
        
        # Snap columns to prevent overlap
        self._snap_columns_to_grid()
        
        # Update scroll region
        self._update_scroll_region()
        
        # Update status
        self.status_label.config(text=f"Zoom: {self.zoom_level:.1f}x")
    
    def _create_status_bar(self, parent: tk.Widget) -> None:
        """Create status bar at bottom."""
        status_bar = tk.Frame(
            parent,
            bg=self.theme_colors["secondary_bg"],
            height=25
        )
        status_bar.pack(fill="x", padx=2, pady=2)
        status_bar.pack_propagate(False)
        
        # Status label
        self.status_label = tk.Label(
            status_bar,
            text=self.translator.translate("Ready"),
            font=self.fonts["small"],
            bg=self.theme_colors["secondary_bg"],
            fg=self.theme_colors["text"],
            anchor="w"
        )
        self.status_label.pack(side="left", padx=5)
        
        # Info labels
        self.holes_label = tk.Label(
            status_bar,
            text="0 " + self.translator.translate("holes"),
            font=self.fonts["small"],
            bg=self.theme_colors["secondary_bg"],
            fg=self.theme_colors["text"]
        )
        self.holes_label.pack(side="right", padx=5)
        
        self.ties_label = tk.Label(
            status_bar,
            text="0 " + self.translator.translate("tie lines"),
            font=self.fonts["small"],
            bg=self.theme_colors["secondary_bg"],
            fg=self.theme_colors["text"]
        )
        self.ties_label.pack(side="right", padx=5)
    
    def add_hole(self, hole_id: str, y_offset: float = 0) -> None:
        """
        Add a drillhole column to the display.
        
        Args:
            hole_id: Drillhole identifier
            y_offset: Vertical offset in stratigraphic space (pixels)
        """
        logger.info(f"Adding hole: {hole_id} at Y offset: {y_offset}")
        
        if hole_id in self.column_widgets:
            logger.warning(f"Hole {hole_id} already added")
            return
        
        try:
            # Create column widget (no parent frame needed, direct canvas child)
            column_widget = DrillholeColumnWidget(
                parent=self.canvas,
                hole_id=hole_id,
                gui_manager=self.gui_manager,
                data_manager=self.data_manager,
                file_manager=self.file_manager,
                config_manager=self.config_manager,
                color_map_manager=self.color_map_manager,
                data_visualizer=self.data_visualizer,
                width=self.column_width,
                show_depth_ruler=True,
                show_data_columns=True,
                viz_columns=self.session.viz_columns
            )
            
            # Position column on canvas
            x_pos = self.next_x_position
            y_pos = y_offset
            
            # Create canvas window for the widget
            # Get dimensions explicitly - widget must have dimensions set
            widget_width = column_widget.get_total_width()
            widget_height = column_widget.get_total_height()
            logger.debug(f"Creating canvas window for {hole_id}: {widget_width}x{widget_height}px")
            
            window_id = self.canvas.create_window(
                x_pos, y_pos,
                anchor="nw",
                window=column_widget,
                width=widget_width,
                height=widget_height,
                tags=(f"column_{hole_id}", "drillhole_column")
            )
            
            # Store references
            self.column_widgets[hole_id] = column_widget
            self.column_windows[hole_id] = window_id
            self.column_positions[hole_id] = (x_pos, y_pos)
            self.column_locked[hole_id] = False  # Start unlocked
            
            # Connect correlation callbacks
            column_widget.on_discontinuity_add_requested = self._handle_discontinuity_add
            column_widget.on_discontinuity_remove_requested = self._handle_discontinuity_remove
            column_widget.on_warp_bar_requested = self._handle_warp_bar_request
            
            # Update next X position for next column using actual widget width
            actual_width = column_widget.get_total_width()
            self.next_x_position += actual_width + self.column_spacing
            logger.debug(f"Column {hole_id} actual width: {actual_width}px, next X: {self.next_x_position}")
            
            # Update session
            if hole_id not in self.session.hole_ids:
                self.session.hole_ids.append(hole_id)
            
            # Enable column dragging for vertical positioning
            self._bind_column_drag(hole_id, column_widget)
            
            # Force geometry update so canvas window knows widget size
            column_widget.update_idletasks()
            
            # Update scroll region to encompass all columns
            self._update_scroll_region()
            
            # Update status
            self._update_status()
            
            # Add to zoom window if visible
            if self.zoom_window_visible and self.zoom_window:
                self.zoom_window.add_drillhole(hole_id, column_widget)
            
            logger.info(f"Successfully added hole: {hole_id} at ({x_pos}, {y_pos})")
            
        except Exception as e:
            logger.error(f"Error adding hole {hole_id}: {e}", exc_info=True)
            self.dialog_helper.show_error(
                self.translator.translate("Error"),
                self.translator.translate("Failed to add hole") + f": {hole_id}\n{str(e)}"
            )
    
    def _bind_column_drag(self, hole_id: str, widget: DrillholeColumnWidget) -> None:
        """
        Bind drag events for column repositioning.
        - Header drag: reorder columns horizontally (X position)
        - Body drag: adjust stratigraphic position vertically (Y position, when unlocked)
        """
        drag_data = {"x": 0, "y": 0, "dragging": False, "mode": None}
        
        # Get header frame from widget
        if not hasattr(widget, 'main_frame'):
            logger.warning(f"Widget for {hole_id} missing main_frame attribute")
            return
        
        # Find header frame (first child of main_frame)
        header_frame = None
        for child in widget.main_frame.winfo_children():
            if isinstance(child, tk.Frame):
                header_frame = child
                break
        
        if not header_frame:
            logger.warning(f"Could not find header frame for {hole_id}")
            return
        
        # HEADER DRAG - Horizontal reordering (X position only)
        def on_header_press(event):
            drag_data["x"] = event.x_root
            drag_data["dragging"] = True
            drag_data["mode"] = "horizontal"
            self.dragging_column = hole_id
            self.drag_mode = "horizontal"
            header_frame.config(cursor="sb_h_double_arrow")
            logger.debug(f"Started horizontal drag for {hole_id}")
        
        def on_header_drag(event):
            if not drag_data["dragging"] or drag_data["mode"] != "horizontal":
                return
            
            # Calculate X delta
            delta_x = event.x_root - drag_data["x"]
            drag_data["x"] = event.x_root
            
            # Update column X position only
            if hole_id in self.column_positions:
                x, y = self.column_positions[hole_id]
                new_x = max(0, x + delta_x)  # Don't go negative
                self.column_positions[hole_id] = (new_x, y)
                
                # Move canvas window
                if hole_id in self.column_windows:
                    self.canvas.coords(self.column_windows[hole_id], new_x, y)
                
                logger.debug(f"Moved {hole_id} to X={new_x}")
        
        def on_header_release(event):
            if drag_data["mode"] == "horizontal":
                drag_data["dragging"] = False
                drag_data["mode"] = None
                self.dragging_column = None
                self.drag_mode = None
                header_frame.config(cursor="")
                
                # Auto-snap to grid positions for cleaner layout
                self._snap_columns_to_grid()
                
                logger.info(f"Final X position for {hole_id}: {self.column_positions[hole_id][0]}")
        
        # BODY DRAG - Vertical stratigraphic positioning (Y position only, when unlocked)
        def on_body_press(event):
            # Check if column is locked
            if self.column_locked.get(hole_id, False):
                logger.debug(f"Column {hole_id} is locked, vertical drag disabled")
                return
            
            drag_data["y"] = event.y_root
            drag_data["dragging"] = True
            drag_data["mode"] = "vertical"
            self.dragging_column = hole_id
            self.drag_mode = "vertical"
            widget.config(cursor="sb_v_double_arrow")
            logger.debug(f"Started vertical drag for {hole_id}")
        
        def on_body_drag(event):
            if not drag_data["dragging"] or drag_data["mode"] != "vertical":
                return
            
            # Check lock state
            if self.column_locked.get(hole_id, False):
                return
            
            # Calculate Y delta
            delta_y = event.y_root - drag_data["y"]
            drag_data["y"] = event.y_root
            
            # Update column Y position only
            if hole_id in self.column_positions:
                x, y = self.column_positions[hole_id]
                new_y = max(0, y + delta_y)  # Don't go negative
                self.column_positions[hole_id] = (x, new_y)
                
                # Move canvas window
                if hole_id in self.column_windows:
                    self.canvas.coords(self.column_windows[hole_id], x, new_y)
                
                logger.debug(f"Moved {hole_id} to Y={new_y} (stratigraphic position)")
        
        def on_body_release(event):
            if drag_data["mode"] == "vertical":
                drag_data["dragging"] = False
                drag_data["mode"] = None
                self.dragging_column = None
                self.drag_mode = None
                widget.config(cursor="")
                logger.info(f"Final stratigraphic Y position for {hole_id}: {self.column_positions[hole_id][1]}")
        
        # Bind header for horizontal reordering
        header_frame.bind("<ButtonPress-1>", on_header_press)
        header_frame.bind("<B1-Motion>", on_header_drag)
        header_frame.bind("<ButtonRelease-1>", on_header_release)
        
        # Bind canvas area for vertical stratigraphic positioning
        if hasattr(widget, 'canvas'):
            widget.canvas.bind("<ButtonPress-1>", on_body_press)
            widget.canvas.bind("<B1-Motion>", on_body_drag)
            widget.canvas.bind("<ButtonRelease-1>", on_body_release)
    
    def _snap_columns_to_grid(self) -> None:
        """Auto-snap columns to grid positions with zoom-adjusted spacing."""
        if not self.column_positions:
            return
        
        # Sort columns by current X position
        sorted_columns = sorted(
            self.column_positions.items(),
            key=lambda item: item[1][0]  # Sort by X position
        )
        
        # Calculate zoom-adjusted spacing accounting for actual widget widths
        zoomed_spacing = int(self.column_spacing * self.zoom_level)
        
        # Reassign X positions with consistent spacing
        x_position = int(50 * self.zoom_level)  # Starting X scaled by zoom
        
        for hole_id, (old_x, y) in sorted_columns:
            self.column_positions[hole_id] = (x_position, y)
            
            # Update canvas window
            if hole_id in self.column_windows:
                self.canvas.coords(self.column_windows[hole_id], x_position, y)
            
            # Get actual widget width (including data columns)
            if hole_id in self.column_widgets:
                widget = self.column_widgets[hole_id]
                actual_width = widget.get_total_width() if hasattr(widget, 'get_total_width') else self.column_width
                zoomed_widget_width = int(actual_width * self.zoom_level)
            else:
                zoomed_widget_width = int(self.column_width * self.zoom_level)
            
            x_position += zoomed_widget_width + zoomed_spacing
            logger.debug(f"Snapped {hole_id} to X={x_position - zoomed_widget_width - zoomed_spacing}, width={zoomed_widget_width}")
        
        # Update scroll region
        self._update_scroll_region()
    
    def _toggle_selected_column_lock(self) -> None:
        """Toggle lock for the currently selected/hovered column."""
        # For now, toggle all columns or require selection
        # TODO: Implement column selection mechanism
        if not self.column_widgets:
            return
        
        # If only one column, toggle it
        if len(self.column_widgets) == 1:
            hole_id = list(self.column_widgets.keys())[0]
            self.toggle_column_lock(hole_id)
            return
        
        # Otherwise, show dialog to select which column to lock/unlock
        from tkinter import simpledialog
        hole_id = simpledialog.askstring(
            "Select Column",
            f"Enter hole ID to lock/unlock:\n{', '.join(self.column_widgets.keys())}",
            parent=self
        )
        
        if hole_id and hole_id in self.column_widgets:
            self.toggle_column_lock(hole_id)
        elif hole_id:
            messagebox.showwarning("Invalid Hole ID", f"Hole ID '{hole_id}' not found")

    def toggle_column_lock(self, hole_id: str) -> None:
        """Toggle lock state for a column (prevents vertical dragging)."""
        if hole_id not in self.column_widgets:
            return
        
        # Toggle lock state
        current_state = self.column_locked.get(hole_id, False)
        self.column_locked[hole_id] = not current_state
        
        # Update visual feedback
        self._update_column_lock_visual(hole_id)
        
        logger.info(f"Column {hole_id} {'locked' if self.column_locked[hole_id] else 'unlocked'}")
    
    def _update_column_lock_visual(self, hole_id: str) -> None:
        """Update visual indicator for column lock state."""
        if hole_id not in self.column_widgets:
            return
        
        widget = self.column_widgets[hole_id]
        is_locked = self.column_locked.get(hole_id, False)
        
        # Update widget's lock visual
        if hasattr(widget, 'update_lock_visual'):
            widget.update_lock_visual(is_locked)
    
    def _update_scroll_region(self) -> None:
        """Update canvas scroll region to encompass all columns (zoom-aware)."""
        if not self.column_positions:
            return
        
        # Find bounds of all columns
        max_x = 0
        max_y = 0
        
        zoomed_width = int(self.column_width * self.zoom_level)
        
        for hole_id, (x, y) in self.column_positions.items():
            if hole_id in self.column_widgets:
                widget = self.column_widgets[hole_id]
                # Get actual column height
                column_height = widget.get_total_height() if hasattr(widget, 'get_total_height') else 1000
                
                max_x = max(max_x, x + zoomed_width + 100)  # Add padding
                max_y = max(max_y, y + column_height + 100)  # Add padding
        
        # Set scroll region with some extra space
        max_x = max(max_x, int(2000 * self.zoom_level))  # Minimum width scaled
        max_y = max(max_y, int(20000 * self.zoom_level))  # Minimum height scaled
        
        self.canvas.configure(scrollregion=(0, 0, max_x, max_y))
        logger.debug(f"Updated scroll region to: (0, 0, {max_x}, {max_y}) at zoom {self.zoom_level:.2f}x")
    
    def remove_hole(self, hole_id: str) -> None:
        """Remove a drillhole column."""
        logger.info(f"Removing hole: {hole_id}")
        
        if hole_id not in self.column_widgets:
            logger.warning(f"Hole {hole_id} not found")
            return
        
        # Remove widget
        self.column_widgets[hole_id].destroy()
        del self.column_widgets[hole_id]
        
        # Remove canvas window
        if hole_id in self.column_windows:
            self.canvas.delete(self.column_windows[hole_id])
            del self.column_windows[hole_id]
        
        # Remove position tracking
        if hole_id in self.column_positions:
            del self.column_positions[hole_id]
        
        # Update session
        if hole_id in self.session.hole_ids:
            self.session.hole_ids.remove(hole_id)
        
        # Remove associated tie lines
        self.tie_lines = [
            tie for tie in self.tie_lines
            if tie.source_hole != hole_id and tie.target_hole != hole_id
        ]
        
        # Update scroll region
        self._update_scroll_region()
        
        # Update status
        self._update_status()
        
        # Remove from zoom window if visible
        if self.zoom_window_visible and self.zoom_window:
            self.zoom_window.remove_drillhole(hole_id)
        
        logger.info(f"Removed hole: {hole_id}")
    
    def _reposition_columns_after_removal(self) -> None:
        """Reposition remaining columns to close gaps after removal."""
        if not self.column_widgets:
            self.next_x_position = 10
            return
        
        # Sort remaining columns by current X position
        sorted_holes = sorted(
            self.column_widgets.keys(),
            key=lambda h: self.column_positions.get(h, (0, 0))[0]
        )
        
        # Reposition sequentially
        x_pos = 10
        for hole_id in sorted_holes:
            widget = self.column_widgets[hole_id]
            _, y_pos = self.column_positions[hole_id]
            
            # Update position
            self.column_positions[hole_id] = (x_pos, y_pos)
            
            # Move canvas window
            if hole_id in self.column_windows:
                self.canvas.coords(self.column_windows[hole_id], x_pos, y_pos)
            
            # Next position
            x_pos += widget.get_total_width() + self.column_spacing
        
        self.next_x_position = x_pos
        self._update_scroll_region()
        
        logger.info(f"Repositioned {len(sorted_holes)} columns after removal")
    
    def _sync_vertical_scroll(self, *args) -> None:
        """Synchronize vertical scrolling across all columns."""
        # Apply scroll to all column canvases
        for column_widget in self.column_widgets.values():
            if hasattr(column_widget, 'canvas'):
                column_widget.canvas.yview(*args)
    
    def _on_column_scroll(self, hole_id: str, first: str, last: str) -> None:
        """Handle scroll from individual column."""
        # Update main scrollbar position
        self.v_scrollbar.set(first, last)
        
        # Sync other columns
        for other_id, column_widget in self.column_widgets.items():
            if other_id != hole_id and hasattr(column_widget, 'canvas'):
                column_widget.canvas.yview("moveto", first)
    
    def _on_columns_configure(self, event: tk.Event) -> None:
        """Handle columns frame resize."""
        # Update canvas scroll region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def _on_canvas_configure(self, event: tk.Event) -> None:
        """Handle canvas resize."""
        # Update scroll region when canvas is resized
        self._update_scroll_region()

    def _show_hole_selection(self) -> None:
        """Show drillhole selection dialog."""
        logger.info("Opening hole selection dialog")
        
        try:
            # Get collar data
            collar_data = self.data_manager.get_collar_data()
            
            if collar_data.empty:
                self.dialog_helper.show_warning(
                    self.translator.translate("No Data"),
                    self.translator.translate("No collar data available")
                )
                return
            
            # Open selection dialog
            dialog = DrillholeSelectionDialog(
                parent=self,
                collar_data=collar_data,
                gui_manager=self.gui_manager,
                config_manager=self.config_manager,
                translator=self.translator,
                dialog_helper=self.dialog_helper,
                initial_selection={"hole_ids": self.session.hole_ids}
            )
            
            # Wait for dialog
            self.wait_window(dialog)
            
            # Process result
            if dialog.result:
                selected_holes = dialog.result.get("hole_ids", [])
                logger.info(f"Selected {len(selected_holes)} holes")
                
                # Add new holes
                for hole_id in selected_holes:
                    if hole_id not in self.column_widgets:
                        self.add_hole(hole_id)
                
                # Remove deselected holes
                for hole_id in list(self.column_widgets.keys()):
                    if hole_id not in selected_holes:
                        self.remove_hole(hole_id)
        
        except Exception as e:
            logger.error(f"Error in hole selection: {e}", exc_info=True)
            self.dialog_helper.show_error(
                self.translator.translate("Error"),
                str(e)
            )
    
    def _remove_selected_hole(self) -> None:
        """Show dialog to select which hole to remove."""
        logger.info("Showing hole removal selection dialog")
        
        if not self.column_widgets:
            self.dialog_helper.show_info(
                self.translator.translate("No Holes"),
                self.translator.translate("No holes to remove.")
            )
            return
        
        # Create selection dialog
        dialog = tk.Toplevel(self)
        dialog.title(self.translator.translate("Remove Hole"))
        dialog.configure(bg=self.theme_colors["background"])
        dialog.transient(self)
        dialog.grab_set()
        
        # Size and center
        dialog.geometry("300x400")
        dialog.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - 300) // 2
        y = self.winfo_y() + (self.winfo_height() - 400) // 2
        dialog.geometry(f"+{x}+{y}")
        
        # Instructions
        tk.Label(
            dialog,
            text=self.translator.translate("Select hole(s) to remove:"),
            font=self.fonts["normal"],
            bg=self.theme_colors["background"],
            fg=self.theme_colors["text"]
        ).pack(pady=(15, 10), padx=15)
        
        # Listbox with scrollbar
        list_frame = tk.Frame(dialog, bg=self.theme_colors["background"])
        list_frame.pack(fill="both", expand=True, padx=15, pady=5)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        
        listbox = tk.Listbox(
            list_frame,
            selectmode=tk.MULTIPLE,
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["text"],
            font=self.fonts["normal"],
            yscrollcommand=scrollbar.set,
            height=12
        )
        listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=listbox.yview)
        
        # Populate with hole IDs (sorted by X position)
        sorted_holes = sorted(
            self.column_widgets.keys(),
            key=lambda h: self.column_positions.get(h, (0, 0))[0]
        )
        for hole_id in sorted_holes:
            listbox.insert(tk.END, hole_id)
        
        # Buttons
        btn_frame = tk.Frame(dialog, bg=self.theme_colors["background"])
        btn_frame.pack(fill="x", padx=15, pady=15)
        
        def do_remove():
            selected_indices = listbox.curselection()
            if not selected_indices:
                return
            
            # Get selected hole IDs
            holes_to_remove = [listbox.get(i) for i in selected_indices]
            
            # Confirm if multiple
            if len(holes_to_remove) > 1:
                if not self.dialog_helper.show_question(
                    self.translator.translate("Confirm Removal"),
                    self.translator.translate(f"Remove {len(holes_to_remove)} holes?")
                ):
                    return
            
            dialog.destroy()
            
            # Remove holes (in reverse order to avoid index issues)
            for hole_id in holes_to_remove:
                self.remove_hole(hole_id)
            
            # Reposition remaining columns
            self._reposition_columns_after_removal()
        
        ModernButton(
            btn_frame,
            text=self.translator.translate("Remove Selected"),
            command=do_remove,
            color=self.theme_colors["accent_red"],
            theme_colors=self.theme_colors
        ).pack(side="left", padx=(0, 10))
        
        ModernButton(
            btn_frame,
            text=self.translator.translate("Cancel"),
            command=dialog.destroy,
            color=self.theme_colors["accent_blue"],
            theme_colors=self.theme_colors
        ).pack(side="left")
    
    def _toggle_tie_line_mode(self) -> None:
        """Toggle tie line creation mode."""
        self.creating_tie_line = not self.creating_tie_line
        
        if self.creating_tie_line:
            logger.info("Entering tie line creation mode")
            self.tie_line_btn.config(relief="sunken")
            self.status_label.config(text=self.translator.translate("Click start and end points for tie line"))
        else:
            logger.info("Exiting tie line creation mode")
            self.tie_line_btn.config(relief="raised")
            self.status_label.config(text=self.translator.translate("Ready"))
            self.tie_line_start = None
    
    def _clear_tie_lines(self) -> None:
        """Clear all tie lines."""
        logger.info("Clearing all tie lines")
        
        result = self.dialog_helper.show_question(
            self.translator.translate("Clear Tie Lines"),
            self.translator.translate("Remove all tie lines?")
        )
        
        if result:
            self.tie_lines.clear()
            self.session.tie_lines.clear()
            self._update_status()
            # TODO: Refresh tie line canvas
    
    def _show_viz_settings(self) -> None:
        """Show visualization settings dialog."""
        logger.info("Opening viz settings dialog")
        
        from gui.DrillholeCorrelation.correlation_viz_settings_dialog import CorrelationVizSettingsDialog
        
        def on_settings_saved(viz_columns: List[Dict[str, Any]]):
            """Callback when settings are saved."""
            logger.info(f"Updating viz columns for all drillholes: {len(viz_columns)} columns")
            
            # Update all column widgets
            for hole_id, widget in self.column_widgets.items():
                if hasattr(widget, 'update_viz_columns'):
                    widget.update_viz_columns(viz_columns)
            
            # Re-snap columns to account for width changes
            self._snap_columns_to_grid()
            self._update_scroll_region()
        
        # Get columns grouped by source from data manager
        columns_by_source = {}
        try:
            if hasattr(self.data_manager, 'get_available_columns'):
                columns_data = self.data_manager.get_available_columns()
                
                # Handle dict format: {source_name: [(col_name, type), ...]}
                if isinstance(columns_data, dict):
                    for source_name, cols in columns_data.items():
                        col_names = []
                        for col_info in cols:
                            col_name = col_info[0] if isinstance(col_info, tuple) else col_info
                            col_names.append(col_name)
                        columns_by_source[source_name] = sorted(col_names)
                
                # Handle list format: [(col_name, type), ...]
                elif isinstance(columns_data, list):
                    col_names = []
                    for col_info in columns_data:
                        col_name = col_info[0] if isinstance(col_info, tuple) else col_info
                        col_names.append(col_name)
                    columns_by_source["Data"] = sorted(col_names)
            
            logger.debug(f"Found {len(columns_by_source)} data sources for viz settings")
            for source, cols in columns_by_source.items():
                logger.debug(f"  {source}: {len(cols)} columns")
                
        except Exception as e:
            logger.warning(f"Could not get columns by source: {e}")
        
        dialog = CorrelationVizSettingsDialog(
            parent=self,
            gui_manager=self.gui_manager,
            config_manager=self.config_manager,
            color_map_manager=self.color_map_manager,
            translator=self.translator,
            callback=on_settings_saved,
            columns_by_source=columns_by_source
        )
    
    def _show_stretch_dialog(self) -> None:
        """Show stretch/compress dialog."""
        logger.info("Opening stretch/compress dialog")
        
        # TODO: Implement stretch dialog
        self.dialog_helper.show_info(
            self.translator.translate("Stretch/Compress"),
            self.translator.translate("Stretch/compress feature coming soon")
        )
    
    def _save_session(self) -> None:
        """Save current session."""
        logger.info("Saving session")
        
        # TODO: Implement session saving
        self.dialog_helper.show_info(
            self.translator.translate("Save"),
            self.translator.translate("Session saving coming soon")
        )
    
    def _load_session(self) -> None:
        """Load saved session."""
        logger.info("Loading session")
        
        # TODO: Implement session loading
        self.dialog_helper.show_info(
            self.translator.translate("Load"),
            self.translator.translate("Session loading coming soon")
        )
    
    def load_session(self, session_id: str) -> None:
        """Load a specific session by ID."""
        logger.info(f"Loading session: {session_id}")
        # TODO: Implement session loading logic
    
    def _update_status(self) -> None:
        """Update status bar information."""
        hole_count = len(self.column_widgets)
        tie_count = len(self.tie_lines)
        
        self.holes_label.config(text=f"{hole_count} " + self.translator.translate("holes"))
        self.ties_label.config(text=f"{tie_count} " + self.translator.translate("tie lines"))
    
    def _center_dialog(self) -> None:
        """Center dialog on screen or parent."""
        self.update_idletasks()
        
        if self.parent and self.parent.winfo_viewable():
            # Center on parent
            x = self.parent.winfo_x() + (self.parent.winfo_width() - self.winfo_width()) // 2
            y = self.parent.winfo_y() + (self.parent.winfo_height() - self.winfo_height()) // 2
        else:
            # Center on screen
            x = (self.winfo_screenwidth() - self.winfo_width()) // 2
            y = (self.winfo_screenheight() - self.winfo_height()) // 2
        
        self.geometry(f"+{x}+{y}")
    
    def _on_close(self) -> None:
        """Handle dialog close."""
        logger.info("Closing CorrelationDialog")
        
        # Check for unsaved changes
        if self.tie_lines or len(self.column_widgets) > 0:
            result = DialogHelper.confirm_dialog(
                self,
                self.translator.translate("Unsaved Changes"),
                self.translator.translate("Close without saving?"),
                yes_text=self.translator.translate("Yes"),
                no_text=self.translator.translate("No")
            )
            if not result:
                return
        
        # Clean up
        for widget in self.column_widgets.values():
            widget.destroy()
        
        self.destroy()
        
        # Quit if in test mode
        if self.test_root:
            self.test_root.quit()

    def _handle_discontinuity_add(self, hole_id: str, depth: float, discontinuity_type: DiscontinuityType):
        """Handle request to add discontinuity from column widget"""
        logger.info(f"Adding {discontinuity_type.value} to {hole_id} at {depth:.1f}m")
        
        try:
            # Check if hole already has segments
            segments = self.session.get_segments_for_hole(hole_id)
            
            if not segments:
                # First discontinuity - create initial segments
                self._create_initial_segments(hole_id, depth, discontinuity_type)
            else:
                # Find segment at this depth
                segment = None
                for seg in segments:
                    if seg.contains_depth(depth):
                        segment = seg
                        break
                
                if segment:
                    self._split_segment(segment, depth, discontinuity_type)
                else:
                    messagebox.showwarning(
                        "Invalid Location",
                        "Cannot add discontinuity at an existing discontinuity boundary"
                    )
                    return
            
            # Update column to show new segments
            self._update_column_segments(hole_id)
            
            logger.info(f"Successfully added discontinuity")
            
        except Exception as e:
            logger.error(f"Error adding discontinuity: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to add discontinuity: {str(e)}")
    
    def _create_initial_segments(self, hole_id: str, depth: float, discontinuity_type: DiscontinuityType):
        """Create initial segments when first discontinuity is added to a hole"""
        # Determine feature category
        if discontinuity_type.value in ['fault', 'intrusive', 'unconformity']:
            category = FeatureCategory.STRUCTURAL
        else:
            category = FeatureCategory.LITHOLOGICAL
        
        # Create or get feature
        feature = self.session.create_feature(category)
        
        # Get depth range from column intervals
        column = self.column_widgets.get(hole_id)
        if column and column.intervals:
            depth_min = min(iv.depth_from for iv in column.intervals)
            depth_max = max(iv.depth_to for iv in column.intervals)
        else:
            depth_min = 0.0
            depth_max = 300.0
        
        # Create top segment
        seg_top = DrillholeSegment(
            segment_id="",
            hole_id=hole_id,
            depth_from_original=depth_min,
            depth_to_original=depth,
            order_index=0
        )
        
        # Create bottom segment
        seg_bottom = DrillholeSegment(
            segment_id="",
            hole_id=hole_id,
            depth_from_original=depth,
            depth_to_original=depth_max,
            order_index=1
        )
        
        # Create discontinuity
        disc = Discontinuity(
            discontinuity_id="",
            hole_id=hole_id,
            discontinuity_type=discontinuity_type,
            depth_at_boundary=depth,
            segment_above_id=seg_top.segment_id,
            segment_below_id=seg_bottom.segment_id,
            feature_id=feature.feature_id
        )
        
        # Link segments to discontinuity
        seg_top.discontinuity_below_id = disc.discontinuity_id
        seg_bottom.discontinuity_above_id = disc.discontinuity_id
        
        # Add to session
        self.session.segments.extend([seg_top, seg_bottom])
        self.session.discontinuities.append(disc)
        feature.add_discontinuity(disc.discontinuity_id)
        
        logger.info(f"Created initial segments for {hole_id}: {feature.name}")
    
    def _split_segment(self, segment: DrillholeSegment, depth: float, discontinuity_type: DiscontinuityType):
        """Split an existing segment at the specified depth"""
        # Determine feature category
        if discontinuity_type.value in ['fault', 'intrusive', 'unconformity']:
            category = FeatureCategory.STRUCTURAL
        else:
            category = FeatureCategory.LITHOLOGICAL
        
        # Create feature
        feature = self.session.create_feature(category)
        
        # Create new segments
        seg_top = DrillholeSegment(
            segment_id="",
            hole_id=segment.hole_id,
            depth_from_original=segment.depth_from_original,
            depth_to_original=depth,
            order_index=segment.order_index,
            discontinuity_above_id=segment.discontinuity_above_id
        )
        
        seg_bottom = DrillholeSegment(
            segment_id="",
            hole_id=segment.hole_id,
            depth_from_original=depth,
            depth_to_original=segment.depth_to_original,
            order_index=segment.order_index + 1,
            discontinuity_below_id=segment.discontinuity_below_id
        )
        
        # Create discontinuity
        disc = Discontinuity(
            discontinuity_id="",
            hole_id=segment.hole_id,
            discontinuity_type=discontinuity_type,
            depth_at_boundary=depth,
            segment_above_id=seg_top.segment_id,
            segment_below_id=seg_bottom.segment_id,
            feature_id=feature.feature_id
        )
        
        # Link segments
        seg_top.discontinuity_below_id = disc.discontinuity_id
        seg_bottom.discontinuity_above_id = disc.discontinuity_id
        
        # Remove old segment
        self.session.segments.remove(segment)
        
        # Add new segments and discontinuity
        self.session.segments.extend([seg_top, seg_bottom])
        self.session.discontinuities.append(disc)
        feature.add_discontinuity(disc.discontinuity_id)
        
        logger.info(f"Split segment {segment.segment_id}: {feature.name}")
    
    def _update_column_segments(self, hole_id: str):
        """Update column widget with current segments and discontinuities"""
        column = self.column_widgets.get(hole_id)
        if not column:
            return
        
        # Update segments and discontinuities
        column.segments = self.session.get_segments_for_hole(hole_id)
        column.discontinuities = self.session.get_discontinuities_for_hole(hole_id)
        
        # Trigger redraw if method exists
        if hasattr(column, 'redraw'):
            column.redraw()
        elif hasattr(column, '_redraw_canvas'):
            column._redraw_canvas()
    
    def _handle_discontinuity_remove(self, hole_id: str, depth: float):
        """Handle request to remove discontinuity"""
        logger.info(f"Remove discontinuity at {depth:.1f}m in {hole_id}")
        # TODO: Implement merge segments logic
        messagebox.showinfo("Not Implemented", "Discontinuity removal not yet implemented")
    
    def _handle_warp_bar_request(self, segment):
        """Handle request to show warp bar for segment"""
        logger.info(f"Warp bar requested for segment {segment.segment_id}")
        # TODO: Implement warp bar dialog
        messagebox.showinfo("Not Implemented", "Warp bar not yet implemented")
    
    # =========================================================================
    # Moisture Preference Toggle
    # =========================================================================
    
    def _toggle_moisture_preference(self) -> None:
        """Toggle between Wet and Dry image preference."""
        if self.moisture_preference == "Dry":
            self.moisture_preference = "Wet"
        else:
            self.moisture_preference = "Dry"
        
        logger.info(f"Moisture preference changed to: {self.moisture_preference}")
        
        # Update button text
        self._update_moisture_button_text()
        
        # Update all column widgets
        for hole_id, widget in self.column_widgets.items():
            widget.set_moisture_preference(self.moisture_preference)
        
        # Update zoom window if visible
        if self.zoom_window_visible and self.zoom_window:
            self.zoom_window.set_moisture_preference(self.moisture_preference)
        
        # Update status
        self.status_label.config(text=f"Showing {self.moisture_preference} images")
    
    def _update_moisture_button_text(self) -> None:
        """Update the moisture toggle button text based on current state."""
        if hasattr(self, 'moisture_toggle_btn'):
            if self.moisture_preference == "Dry":
                self.moisture_toggle_btn.config(text="🔆 Dry")
            else:
                self.moisture_toggle_btn.config(text="💧 Wet")
    
    # =========================================================================
    # Export Functionality
    # =========================================================================
    
    def _show_export_menu(self) -> None:
        """Show export options menu."""
        menu = tk.Menu(self, tearoff=0)
        
        menu.add_command(
            label="Export Individual Columns...",
            command=self._export_individual_columns
        )
        menu.add_command(
            label="Export Combined View...",
            command=self._export_combined_view
        )
        menu.add_separator()
        menu.add_command(
            label="Export High-Res Individual...",
            command=lambda: self._export_individual_columns(high_res=True)
        )
        menu.add_command(
            label="Export High-Res Combined...",
            command=lambda: self._export_combined_view(high_res=True)
        )
        
        # Position menu below the export button
        try:
            x = self.export_btn.winfo_rootx()
            y = self.export_btn.winfo_rooty() + self.export_btn.winfo_height()
            menu.post(x, y)
        except Exception:
            menu.post(self.winfo_pointerx(), self.winfo_pointery())
    
    def _export_individual_columns(self, high_res: bool = False) -> None:
        """Export each drillhole column as individual image files."""
        from tkinter import filedialog
        
        if not self.column_widgets:
            messagebox.showwarning(
                self.translator.translate("No Data"),
                self.translator.translate("No drillhole columns to export.")
            )
            return
        
        # Ask for output directory
        output_dir = filedialog.askdirectory(
            title=self.translator.translate("Select Export Directory"),
            parent=self
        )
        
        if not output_dir:
            return
        
        logger.info(f"Exporting individual columns to: {output_dir} (high_res={high_res})")
        
        scale_factor = 4 if high_res else 1
        suffix = "_highres" if high_res else ""
        
        exported = 0
        for hole_id, widget in self.column_widgets.items():
            try:
                # Render widget to image
                img = widget.render_to_image(scale_factor=scale_factor)
                if img:
                    filename = f"{hole_id}_column{suffix}.png"
                    filepath = os.path.join(output_dir, filename)
                    img.save(filepath, "PNG")
                    logger.info(f"  Exported: {filename}")
                    exported += 1
            except Exception as e:
                logger.error(f"Error exporting {hole_id}: {e}", exc_info=True)
        
        messagebox.showinfo(
            self.translator.translate("Export Complete"),
            self.translator.translate(f"Exported {exported} column images to:\n{output_dir}")
        )
    
    def _export_combined_view(self, high_res: bool = False) -> None:
        """Export all drillhole columns as a single combined image."""
        from tkinter import filedialog
        from PIL import Image
        
        if not self.column_widgets:
            messagebox.showwarning(
                self.translator.translate("No Data"),
                self.translator.translate("No drillhole columns to export.")
            )
            return
        
        # Ask for output file
        filetypes = [
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg"),
            ("All files", "*.*")
        ]
        
        suffix = "_highres" if high_res else ""
        default_name = f"correlation_view{suffix}.png"
        
        filepath = filedialog.asksaveasfilename(
            title=self.translator.translate("Save Combined View"),
            defaultextension=".png",
            filetypes=filetypes,
            initialfile=default_name,
            parent=self
        )
        
        if not filepath:
            return
        
        logger.info(f"Exporting combined view to: {filepath} (high_res={high_res})")
        
        scale_factor = 4 if high_res else 1
        spacing = 10 * scale_factor  # Gap between columns
        
        # Render all columns
        column_images = []
        max_height = 0
        total_width = 0
        
        # Sort by X position to maintain visual order
        sorted_holes = sorted(
            self.column_widgets.keys(),
            key=lambda h: self.column_positions.get(h, (0, 0))[0]
        )
        
        for hole_id in sorted_holes:
            widget = self.column_widgets[hole_id]
            try:
                img = widget.render_to_image(scale_factor=scale_factor)
                if img:
                    column_images.append((hole_id, img))
                    max_height = max(max_height, img.height)
                    total_width += img.width + spacing
            except Exception as e:
                logger.error(f"Error rendering {hole_id}: {e}", exc_info=True)
        
        if not column_images:
            messagebox.showerror(
                self.translator.translate("Export Failed"),
                self.translator.translate("Could not render any columns.")
            )
            return
        
        # Remove last spacing
        total_width -= spacing
        
        # Create combined image
        combined = Image.new("RGB", (total_width, max_height), color=(30, 30, 30))
        
        x_offset = 0
        for hole_id, img in column_images:
            # Center vertically if heights differ
            y_offset = (max_height - img.height) // 2
            combined.paste(img, (x_offset, y_offset))
            x_offset += img.width + spacing
        
        # Save
        combined.save(filepath)
        logger.info(f"Combined view exported: {filepath} ({combined.width}x{combined.height})")
        
        messagebox.showinfo(
            self.translator.translate("Export Complete"),
            self.translator.translate(f"Exported combined view to:\n{filepath}\n\nSize: {combined.width}x{combined.height}px")
        )

# ============================================================================
# TEST HARNESS
# ============================================================================

def create_test_managers():
    """Create mock managers for testing."""
    import pandas as pd
    
    logger.info("Creating test managers")
    
    # Mock GUI Manager
    class TestGUIManager:
        def __init__(self):
            self.theme_colors = {
                "background": "#1e1e1e",
                "secondary_bg": "#252526",
                "text": "#e0e0e0",
                "field_bg": "#2d2d2d",
                "field_border": "#3f3f3f",
                "border": "#3f3f3f",
                "accent_blue": "#3a7ca5",
                "accent_green": "#4a8259",
                "accent_red": "#9e4a4a",
                "accent_yellow": "#e5c07b",
            }
            self.fonts = {
                "heading": ("Arial", 12, "bold"),
                "normal": ("Arial", 10),
                "small": ("Arial", 9),
                "label": ("Arial", 11, "bold"),
            }
        
        def style_dropdown(self, widget, width=20):
            widget.config(width=width)
    
    # Mock Data Manager
    class TestDataManager:
        def __init__(self):
            # Create sample data
            self.holes = ["BA1234", "BA1235", "BA1236", "BA1237"]
            self.data = {}
            
            for hole in self.holes:
                # Create sample intervals
                intervals = []
                for depth in range(0, 100, 1):
                    intervals.append({
                        "HoleID": hole,
                        "From": float(depth),
                        "To": float(depth + 1),
                        "Fe_pct_BEST": 45 + (depth % 20),
                        "SiO2_pct_BEST": 25 - (depth % 15),
                        "Al2O3_pct_BEST": 5 + (depth % 10) / 2,
                    })
                self.data[hole] = pd.DataFrame(intervals)
        
        def get_hole_data(self, hole_id):
            return self.data.get(hole_id, pd.DataFrame())
        
        def get_collar_data(self):
            collars = []
            for i, hole in enumerate(self.holes):
                collars.append({
                    "HOLEID": hole,
                    "X": 1000 + i * 50,
                    "Y": 2000 + i * 30,
                    "Z": 500 - i * 10,
                    "PROJECT": "TEST",
                    "PLANNED_HOLEID": hole + "_P"
                })
            return pd.DataFrame(collars)
    
    # Mock File Manager
    class TestFileManager:
        def __init__(self):
            self.base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Mock Config Manager
    class TestConfigManager:
        def __init__(self):
            self.config = {
                "correlation_dialog_width": 1200,
                "correlation_dialog_height": 800,
                "correlation_column_width": 200,
                "correlation_cell_height": 100,
                "correlation_default_section_width": 50.0,
                "shared_folder_approved_compartments_folder": "",
                "viz_columns": [
                    {"column": "Fe_pct_BEST", "color_map": "fe_grade"},
                    {"column": "SiO2_pct_BEST", "color_map": "sio2_grade"},
                ]
            }
        
        def get(self, key, default=None):
            return self.config.get(key, default)
        
        def set(self, key, value):
            self.config[key] = value
    
    # Mock Color Map Manager
    class TestColorMapManager:
        def get_color_for_value(self, value, color_map):
            # Simple color mapping
            if "fe" in color_map.lower():
                if value > 60:
                    return "#ff0000"
                elif value > 50:
                    return "#ff8800"
                else:
                    return "#00ff00"
            return "#888888"
    
    return (
        TestGUIManager(),
        TestDataManager(),
        TestFileManager(),
        TestConfigManager(),
        TestColorMapManager()
    )


def run_test():
    """Run test harness for CorrelationDialog."""
    import logging
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting CorrelationDialog test harness")
    
    # Create test managers
    gui_mgr, data_mgr, file_mgr, config_mgr, color_mgr = create_test_managers()
    
    # Create dialog
    dialog = CorrelationDialog(
        parent=None,
        gui_manager=gui_mgr,
        data_manager=data_mgr,
        file_manager=file_mgr,
        config_manager=config_mgr,
        color_map_manager=color_mgr,
        initial_holes=["BA1234", "BA1235"],
        test_mode=True
    )
    
    # Run
    dialog.mainloop()
    
    logger.info("Test harness completed")


if __name__ == "__main__":
    run_test()
