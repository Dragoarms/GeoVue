"""
Column Settings Dialog - View and customize data source column configurations.

Provides a dialog for:
- Viewing all loaded data sources and their columns
- Editing column display names (aliases)
- Setting data types (numeric, categorical, text, date)
- Configuring decimal places for numeric display
- Setting null/NaN handling behavior
- Viewing column statistics (unique values, null count, etc.)

Similar UI pattern to classification and tag manager dialogs.

Author: George Symonds
"""

import tkinter as tk
from tkinter import ttk
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

from gui.dialog_helper import DialogHelper
from gui.widgets.modern_button import ModernButton

# Import ColorMapEditorDialog - lazy import in method to avoid circular imports
# from gui.color_map_editor_dialog import ColorMapEditorDialog

logger = logging.getLogger(__name__)


@dataclass
class ColumnDisplayConfig:
    """Configuration for a single column's display in the dialog."""
    source_name: str
    display_name: str
    data_type: str
    decimals: int
    null_handling: str
    is_key: bool
    is_visible: bool
    unique_count: int
    null_count: int
    total_count: int
    color_map: Optional[str] = None
    
    # UI state variables (set during row creation)
    alias_var: Optional[tk.StringVar] = None
    type_var: Optional[tk.StringVar] = None
    decimals_var: Optional[tk.IntVar] = None
    nulls_var: Optional[tk.StringVar] = None
    visible_var: Optional[tk.BooleanVar] = None
    colormap_menu: Optional[tk.OptionMenu] = None  # Reference for refreshing


class ColumnSettingsDialog:
    """
    Modal dialog for viewing and editing data source column configurations.

    Features:
    - Collapsible sections per data source file
    - Sortable/filterable column list
    - Inline editing of column properties
    - Statistics display (unique values, null counts)
    - Bulk operations (reset to defaults, apply to all)
    """

    # Data type options
    DATA_TYPES = ["numeric", "categorical", "text", "date", "boolean", "hex_color"]

    # Null handling options (must match NullHandling enum values in schema.py)
    NULL_HANDLING = ["keep", "drop", "fill_zero", "fill_empty", "fill_value"]

    # Column widths for grid layout alignment
    COLUMN_WIDTHS = {
        "visible": 50,      # Checkbox
        "name": 150,        # Column name
        "alias": 150,       # Alias entry
        "type": 100,        # Data type dropdown
        "decimals": 60,     # Decimals spinbox
        "nulls": 100,       # Null handling dropdown
        "colormap": 120,    # Color map dropdown + button
        "unique": 60,       # Unique count
        "nullcount": 100,   # Null count
        "key": 40,          # Key indicator
        "issues": 200,      # Data issues/warnings
    }
    
    # Register data column definitions (fixed structure)
    REGISTER_COLUMNS = {
        "review_metadata": {
            "display_name": "Review Metadata",
            "columns": [
                ("classification", "Classification", "categorical"),
                ("consensus_classification", "Consensus Classification", "categorical"),
                ("review_count", "Review Count", "numeric"),
                ("reviewed_by", "Reviewed By", "text"),
                ("agreement", "Agreement Level", "categorical"),
                ("tags", "Tags", "text"),
                ("comments", "Comments", "text"),
            ]
        },
        "image_properties": {
            "display_name": "Image Properties",
            "columns": [
                ("wet_hex", "Wet Hex Color", "hex_color"),
                ("dry_hex", "Dry Hex Color", "hex_color"),
                ("combined_hex", "Combined Hex Color", "hex_color"),
                ("has_wet", "Has Wet Image", "boolean"),
                ("has_dry", "Has Dry Image", "boolean"),
                ("scale_px_per_cm", "Scale (px/cm)", "numeric"),
            ]
        },
        "compartment_corners": {
            "display_name": "Compartment Corners",
            "columns": [
                ("source_image_uid", "Source Image UID", "text"),
                ("scale_px_per_cm", "Scale (px/cm)", "numeric"),
                ("corner_top_left", "Corner Top Left", "text"),
                ("corner_top_right", "Corner Top Right", "text"),
                ("corner_bottom_left", "Corner Bottom Left", "text"),
                ("corner_bottom_right", "Corner Bottom Right", "text"),
            ]
        },
        "compartment_register": {
            "display_name": "Compartment Register",
            "columns": [
                ("photo_status", "Photo Status", "categorical"),
                ("uid", "Compartment UID", "text"),
                ("processed_date", "Processed Date", "date"),
                ("file_hash", "File Hash", "text"),
            ]
        },
        "image_metadata": {
            "display_name": "Image Metadata",
            "columns": [
                ("hole_id", "Hole ID", "categorical"),
                ("depth_from", "Depth From", "numeric"),
                ("depth_to", "Depth To", "numeric"),
                ("image_path", "Image Path", "text"),
                ("original_image", "Original Image", "text"),
            ]
        },
        "rc_metrics": {
            "display_name": "RC Metrics (Computed)",
            "columns": [
                ("weighted_hardness", "Weighted Hardness", "numeric"),
                ("total_gangue_pct", "Total Gangue %", "numeric"),
                ("si_gangue_pct", "Si Gangue %", "numeric"),
                ("al_gangue_pct", "Al Gangue %", "numeric"),
                ("carbonate_gangue_pct", "Carbonate Gangue %", "numeric"),
                ("zonation_pr_pct", "Zonation PR %", "numeric"),
                ("zonation_hy_pct", "Zonation HY %", "numeric"),
                ("zonation_de_pct", "Zonation DE %", "numeric"),
                ("zonation_un_pct", "Zonation UN %", "numeric"),
                ("quartz_pct", "Quartz %", "numeric"),
                ("chert_pct", "Chert %", "numeric"),
            ]
        }
    }
    
    def __init__(
        self,
        parent,
        gui_manager,
        data_coordinator=None,
        config_manager=None,
        on_save_callback: Optional[Callable] = None,
    ):
        """
        Initialize the column settings dialog.
        
        Args:
            parent: Parent window
            gui_manager: GUIManager for theming
            data_coordinator: DataCoordinator with loaded data sources
            config_manager: ConfigManager for saving settings
            on_save_callback: Optional callback when settings are saved
        """
        self.parent = parent
        self.gui_manager = gui_manager
        self.theme = gui_manager.theme_colors
        self.data_coordinator = data_coordinator
        self.config_manager = config_manager
        self.on_save_callback = on_save_callback
        
        # Dialog state
        self.dialog = None
        self.result = None
        
        # Data state
        self.source_configs: Dict[str, List[ColumnDisplayConfig]] = {}
        self.source_frames: Dict[str, tk.Frame] = {}
        self.source_expanded: Dict[str, tk.BooleanVar] = {}
        
        # Register column configs (separate from geological CSV sources)
        self.register_configs: Dict[str, List[ColumnDisplayConfig]] = {}

        # Filter state
        self.filter_var = None
        self.show_keys_var = None
        self.show_hidden_var = None

        # Issue tracking - maps column names to issue messages
        self.column_issues: Dict[str, List[str]] = {}

        logger.info("ColumnSettingsDialog initialized")
        
        self._create_dialog()
    
    def _create_dialog(self):
        """Create the main dialog window."""
        # Create dialog directly (not using DialogHelper) to get full window controls
        # DialogHelper.create_dialog calls transient() which removes minimize/maximize buttons
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("Column Settings")
        self.dialog.configure(bg=self.theme["background"])

        # Don't call transient() - it removes minimize/maximize buttons on Windows
        # Don't call grab_set() - it prevents combobox dropdowns from working

        # Configure ttk styles for dark theme comboboxes
        self._configure_ttk_styles()

        # Build UI (creates scrollable_frame)
        self._build_ui()

        # Load data into UI (must be after _build_ui creates scrollable_frame)
        self._load_column_data()

        # Set minimum size and maximize the window
        self.dialog.minsize(800, 600)

        # Maximize the window on open
        try:
            self.dialog.state('zoomed')  # Windows
        except tk.TclError:
            try:
                self.dialog.attributes('-zoomed', True)  # Linux
            except tk.TclError:
                # Fallback: set to screen size
                screen_width = self.dialog.winfo_screenwidth()
                screen_height = self.dialog.winfo_screenheight()
                self.dialog.geometry(f"{screen_width}x{screen_height}+0+0")

        # Handle window close event
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_ttk_styles(self):
        """Configure ttk widget styles for dark theme."""
        # Note: We use tk.OptionMenu instead of ttk.Combobox for dropdowns
        # because ttk.Combobox has issues with dropdown popups in certain contexts
        # Only configure styles for ttk widgets we actually use (Scrollbar)
        style = ttk.Style(self.dialog)

        # Configure Scrollbar for dark theme
        style.configure(
            "Vertical.TScrollbar",
            background=self.theme["secondary_bg"],
            troughcolor=self.theme["background"],
            arrowcolor=self.theme["text"],
        )
    
    def _build_ui(self):
        """Build the dialog UI layout."""
        # Main container
        main_frame = tk.Frame(self.dialog, bg=self.theme["background"])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # TOP: Title and filter bar
        self._build_header(main_frame)

        # WARNINGS: Sanitization issues panel (collapsible)
        self._build_warnings_panel(main_frame)

        # MIDDLE: Scrollable content area with data sources
        self._build_content_area(main_frame)

        # BOTTOM: Action buttons
        self._build_bottom_buttons(main_frame)
    
    def _build_header(self, parent):
        """Build the header section with title and filters."""
        header_frame = tk.Frame(parent, bg=self.theme["background"])
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Title
        title_label = tk.Label(
            header_frame,
            text="Data Source Column Configuration",
            font=("Arial", 14, "bold"),
            bg=self.theme["background"],
            fg=self.theme["text"],
        )
        title_label.pack(side=tk.LEFT)
        
        # Filter controls (right side)
        filter_frame = tk.Frame(header_frame, bg=self.theme["background"])
        filter_frame.pack(side=tk.RIGHT)
        
        # Search filter
        tk.Label(
            filter_frame,
            text="Filter:",
            bg=self.theme["background"],
            fg=self.theme["text"],
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        self.filter_var = tk.StringVar()
        self.filter_var.trace_add("write", self._on_filter_change)
        
        filter_entry = tk.Entry(
            filter_frame,
            textvariable=self.filter_var,
            bg=self.theme["field_bg"],
            fg=self.theme["text"],
            width=20,
        )
        filter_entry.pack(side=tk.LEFT, padx=(0, 15))
        
        # Show key columns checkbox
        self.show_keys_var = tk.BooleanVar(value=False)
        keys_cb = tk.Checkbutton(
            filter_frame,
            text="Show Key Columns",
            variable=self.show_keys_var,
            bg=self.theme["background"],
            fg=self.theme["text"],
            selectcolor=self.theme.get("accent_green", "#47b881"),
            activebackground=self.theme["background"],
            activeforeground=self.theme["text"],
            command=self._on_filter_change,
        )
        keys_cb.pack(side=tk.LEFT, padx=(0, 10))

        # Show hidden columns checkbox
        self.show_hidden_var = tk.BooleanVar(value=True)
        hidden_cb = tk.Checkbutton(
            filter_frame,
            text="Show Hidden",
            variable=self.show_hidden_var,
            bg=self.theme["background"],
            fg=self.theme["text"],
            selectcolor=self.theme.get("accent_green", "#47b881"),
            activebackground=self.theme["background"],
            activeforeground=self.theme["text"],
            command=self._on_filter_change,
        )
        hidden_cb.pack(side=tk.LEFT)

    def _build_warnings_panel(self, parent):
        """Collect data sanitization issues and build per-column issue tracking."""
        # Clear existing issues
        self.column_issues.clear()

        # Get sanitization report from coordinator
        if self.data_coordinator and hasattr(self.data_coordinator, '_geological_store'):
            try:
                from processing.DataManager.data_sanitizer import sanitize_geological_store
                report = sanitize_geological_store(self.data_coordinator._geological_store)

                # Map issues to column names
                for issue in report.issues:
                    if issue.column:
                        col_lower = issue.column.lower()
                        if col_lower not in self.column_issues:
                            self.column_issues[col_lower] = []
                        self.column_issues[col_lower].append(issue.message)

                # Show summary if there are issues
                if self.column_issues:
                    total_issues = sum(len(msgs) for msgs in self.column_issues.values())
                    summary_frame = tk.Frame(parent, bg=self.theme.get("accent_orange", "#d97a4a"))
                    summary_frame.pack(fill=tk.X, pady=(0, 5))

                    tk.Label(
                        summary_frame,
                        text=f"⚠ {total_issues} data warnings - affected columns highlighted in orange",
                        font=("Arial", 10, "bold"),
                        bg=self.theme.get("accent_orange", "#d97a4a"),
                        fg="white",
                        padx=10,
                        pady=5,
                    ).pack(side=tk.LEFT)

            except Exception as e:
                logger.debug(f"Could not get sanitization report: {e}")

    def _refresh_issues(self):
        """Re-run sanitization and update row highlighting."""
        # This would re-detect issues - for now just log
        logger.debug("Issue refresh requested")
        # Future: re-run sanitization and update row backgrounds

    def _build_content_area(self, parent):
        """Build the scrollable content area."""
        # Container with scrollbar
        container = tk.Frame(parent, bg=self.theme["background"])
        container.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for scrolling
        self.canvas = tk.Canvas(
            container,
            bg=self.theme["background"],
            highlightthickness=0,
        )
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # Scrollable frame inside canvas
        self.scrollable_frame = tk.Frame(self.canvas, bg=self.theme["background"])
        self.canvas_window = self.canvas.create_window(
            (0, 0),
            window=self.scrollable_frame,
            anchor="nw",
        )
        
        # Bind scroll events
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfig(self.canvas_window, width=e.width)
        )
        
        # Enable mousewheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
    
    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling."""
        try:
            if self.canvas.winfo_exists():
                self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except tk.TclError:
            # Canvas was destroyed, unbind to prevent further errors
            try:
                self.canvas.unbind_all("<MouseWheel>")
            except:
                pass
    
    def _on_filter_change(self, *args):
        """Handle changes to the filter criteria."""
        filter_text = self.filter_var.get().lower().strip()
        show_keys = self.show_keys_var.get()
        show_hidden = self.show_hidden_var.get()
        
        # Iterate through all source frames and their column rows
        for source_name, content_frame in self.source_frames.items():
            for child in content_frame.winfo_children():
                # Skip header row (it doesn't have column_config)
                if not hasattr(child, 'column_config'):
                    continue
                
                config = child.column_config
                
                # Determine if row should be visible
                visible = True
                
                # Filter by text (match column name or display name)
                if filter_text:
                    name_match = filter_text in config.source_name.lower()
                    display_match = filter_text in (config.display_name or "").lower()
                    if not (name_match or display_match):
                        visible = False
                
                # Filter key columns
                if config.is_key and not show_keys:
                    visible = False
                
                # Filter hidden columns
                if not config.is_visible and not show_hidden:
                    visible = False
                
                # Show/hide the row
                if visible:
                    child.pack(fill=tk.X, padx=5, pady=1)
                else:
                    child.pack_forget()
    
    def _build_bottom_buttons(self, parent):
        """Build the bottom action buttons."""
        button_frame = tk.Frame(parent, bg=self.theme["background"])
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Left side - bulk actions
        left_buttons = tk.Frame(button_frame, bg=self.theme["background"])
        left_buttons.pack(side=tk.LEFT)
        
        ModernButton(
            left_buttons,
            text="Expand All",
            color=self.theme["secondary_bg"],
            command=self._expand_all,
            theme_colors=self.theme,
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ModernButton(
            left_buttons,
            text="Collapse All",
            color=self.theme["secondary_bg"],
            command=self._collapse_all,
            theme_colors=self.theme,
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ModernButton(
            left_buttons,
            text="Reset to Defaults",
            color=self.theme["accent_red"],
            command=self._reset_to_defaults,
            theme_colors=self.theme,
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        # Right side - save/cancel
        right_buttons = tk.Frame(button_frame, bg=self.theme["background"])
        right_buttons.pack(side=tk.RIGHT)
        
        ModernButton(
            right_buttons,
            text="Save",
            color=self.theme["accent_green"],
            command=self._on_save,
            theme_colors=self.theme,
        ).pack(side=tk.LEFT, padx=5)
        
        ModernButton(
            right_buttons,
            text="Cancel",
            color=self.theme["secondary_bg"],
            command=self._on_close,
            theme_colors=self.theme,
        ).pack(side=tk.LEFT)
    
    def _load_column_data(self):
        """Load column data from data coordinator."""
        if not self.data_coordinator:
            logger.warning("No data coordinator available")
            self._show_no_data_message()
            return
        
        # Get geological store
        geo_store = self.data_coordinator._geological_store
        if not geo_store:
            logger.warning("No geological store available")
            self._show_no_data_message()
            return
        
        # Get all data sources
        sources = geo_store.get_data_sources()
        if not sources:
            logger.warning("No data sources loaded")
            self._show_no_data_message()
            return
        
        logger.info(f"Loading column data from {len(sources)} sources")
        
        for source_name, indexed_source in sources.items():
            schema = indexed_source.schema
            df = indexed_source.df
            
            # Use ACTUAL row count from loaded data
            actual_row_count = indexed_source.row_count
            
            # Update schema with actual row count
            schema.row_count = actual_row_count
            
            columns = []
            for col_name, col_schema in schema.columns.items():
                # Get statistics from DataFrame
                if df is not None and col_name.lower() in [c.lower() for c in df.columns]:
                    # Find actual column name (case-insensitive)
                    actual_col = next((c for c in df.columns if c.lower() == col_name.lower()), col_name)
                    col_data = df[actual_col]
                    unique_count = int(col_data.nunique())
                    null_count = int(col_data.isna().sum())
                    total_count = len(col_data)
                else:
                    unique_count = 0
                    null_count = 0
                    total_count = actual_row_count
                
                config = ColumnDisplayConfig(
                    source_name=col_schema.source_name,
                    display_name=col_schema.display_name or col_schema.source_name,
                    data_type=col_schema.data_type.value,
                    decimals=getattr(col_schema, 'decimals', 2),
                    null_handling=col_schema.null_handling.value,
                    is_key=col_schema.is_key_column,
                    is_visible=col_schema.is_visible,
                    unique_count=unique_count,
                    null_count=null_count,
                    total_count=total_count,
                    color_map=col_schema.color_map,
                )
                columns.append(config)
            
            # Sort columns: keys first, then by order/name
            columns.sort(key=lambda c: (not c.is_key, c.source_name.lower()))
            
            self.source_configs[source_name] = columns
            
            # Create section for this source
            self._create_source_section(source_name, schema, columns)
        
        logger.info(f"Loaded {sum(len(c) for c in self.source_configs.values())} columns from {len(self.source_configs)} sources")
        
        # Also load register columns
        self._load_register_columns()

    def _show_no_data_message(self):
        """Show message when no data is available."""
        msg_frame = tk.Frame(self.scrollable_frame, bg=self.theme["background"])
        msg_frame.pack(fill=tk.X, padx=20, pady=50)
        
        tk.Label(
            msg_frame,
            text="No data sources loaded.\n\nPlease configure data source folders in Settings.",
            font=("Arial", 12),
            bg=self.theme["background"],
            fg=self.theme["text"],
            justify=tk.CENTER,
        ).pack()
    
    def _load_register_columns(self):
        """Load register data columns (reviews, image properties, compartment data)."""
        logger.info("Loading register column definitions...")
        
        # Load saved register settings from config
        saved_settings = {}
        if self.config_manager:
            saved_settings = self.config_manager.get("register_column_settings", {})
        
        for section_key, section_def in self.REGISTER_COLUMNS.items():
            columns = []
            saved_section = saved_settings.get(section_key, {})
            
            for col_name, col_display, col_type in section_def["columns"]:
                # Load saved settings or use defaults
                saved_col = saved_section.get(col_name, {})
                
                config = ColumnDisplayConfig(
                    source_name=col_name,
                    display_name=saved_col.get("display_name", col_display),
                    data_type=saved_col.get("data_type", col_type),
                    decimals=saved_col.get("decimals", 2 if col_type == "numeric" else 0),
                    null_handling=saved_col.get("null_handling", "keep"),
                    is_key=col_name in ("hole_id", "depth_to", "depth_from", "uid", "source_image_uid"),
                    is_visible=saved_col.get("is_visible", True),
                    unique_count=0,  # Not applicable for register columns
                    null_count=0,
                    total_count=0,
                    color_map=saved_col.get("color_map", None),
                )
                columns.append(config)
            
            self.register_configs[section_key] = columns
            
            # Create the section in UI
            self._create_register_section(
                section_key,
                section_def["display_name"],
                columns
            )
        
        logger.info(f"Loaded {sum(len(c) for c in self.register_configs.values())} register columns from {len(self.register_configs)} sections")
    
    def _create_register_section(self, section_key: str, display_name: str, columns: List[ColumnDisplayConfig]):
        """Create a collapsible section for register data (simplified, no file schema)."""
        # Section container with distinct styling
        section_frame = tk.Frame(
            self.scrollable_frame,
            bg=self.theme["secondary_bg"],
            highlightbackground=self.theme.get("accent_blue", "#3a7ca5"),
            highlightthickness=2,
        )
        section_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Header (clickable to expand/collapse)
        source_name = f"register_{section_key}"
        self.source_expanded[source_name] = tk.BooleanVar(value=True)
        
        header_frame = tk.Frame(section_frame, bg=self.theme["secondary_bg"])
        header_frame.pack(fill=tk.X)
        
        # Expand/collapse indicator
        expand_label = tk.Label(
            header_frame,
            text="▼" if self.source_expanded[source_name].get() else "▶",
            font=("Arial", 10),
            bg=self.theme["secondary_bg"],
            fg=self.theme["text"],
            cursor="hand2",
        )
        expand_label.pack(side=tk.LEFT, padx=(10, 5), pady=8)
        
        # Section name with badge
        info_text = f"📋 {display_name} (Register Data) - {len(columns)} columns"
        
        header_label = tk.Label(
            header_frame,
            text=info_text,
            font=("Arial", 11, "bold"),
            bg=self.theme["secondary_bg"],
            fg=self.theme.get("accent_blue", "#3a7ca5"),
            cursor="hand2",
        )
        header_label.pack(side=tk.LEFT, pady=8)
        
        # Info label
        info_label = tk.Label(
            header_frame,
            text="(JSON Register Data)",
            font=("Arial", 8),
            bg=self.theme["secondary_bg"],
            fg=self.theme.get("subtext", "#888888"),
        )
        info_label.pack(side=tk.RIGHT, padx=10, pady=8)
        
        # Bind click to toggle
        def toggle_section(event=None):
            expanded = not self.source_expanded[source_name].get()
            self.source_expanded[source_name].set(expanded)
            expand_label.config(text="▼" if expanded else "▶")
            if expanded:
                self.source_frames[source_name].pack(fill=tk.X)
            else:
                self.source_frames[source_name].pack_forget()
        
        expand_label.bind("<Button-1>", toggle_section)
        header_label.bind("<Button-1>", toggle_section)
        
        # Content frame (collapsible)
        content_frame = tk.Frame(section_frame, bg=self.theme["background"])
        content_frame.pack(fill=tk.X)
        self.source_frames[source_name] = content_frame
        
        # Column headers
        self._create_column_headers(content_frame)
        
        # Column rows
        for col_config in columns:
            self._create_column_row(content_frame, source_name, col_config)
    
    def _create_source_section(self, source_name: str, schema, columns: List[ColumnDisplayConfig]):
        """Create a collapsible section for a data source."""
        # Section container
        section_frame = tk.Frame(
            self.scrollable_frame,
            bg=self.theme["secondary_bg"],
            highlightbackground=self.theme["border"],
            highlightthickness=1,
        )
        section_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Header (clickable to expand/collapse)
        self.source_expanded[source_name] = tk.BooleanVar(value=True)
        
        header_frame = tk.Frame(section_frame, bg=self.theme["secondary_bg"])
        header_frame.pack(fill=tk.X)
        
        # Expand/collapse indicator
        expand_label = tk.Label(
            header_frame,
            text="▼" if self.source_expanded[source_name].get() else "▶",
            font=("Arial", 10),
            bg=self.theme["secondary_bg"],
            fg=self.theme["text"],
            cursor="hand2",
        )
        expand_label.pack(side=tk.LEFT, padx=(10, 5), pady=8)
        
        # Source name and info
        info_text = f"{source_name} ({schema.dataset_type.value.upper()}) - {len(columns)} columns, {schema.row_count:,} rows"
        
        header_label = tk.Label(
            header_frame,
            text=info_text,
            font=("Arial", 11, "bold"),
            bg=self.theme["secondary_bg"],
            fg=self.theme["text"],
            cursor="hand2",
        )
        header_label.pack(side=tk.LEFT, pady=8)

        # Select/Deselect All buttons for this source
        select_buttons_frame = tk.Frame(header_frame, bg=self.theme["secondary_bg"])
        select_buttons_frame.pack(side=tk.LEFT, padx=(20, 0))

        def select_all_in_source():
            """Select all visible checkboxes for columns in this source."""
            if source_name in self.source_configs:
                for col_config in self.source_configs[source_name]:
                    if col_config.visible_var:
                        col_config.visible_var.set(True)

        def deselect_all_in_source():
            """Deselect all visible checkboxes for columns in this source."""
            if source_name in self.source_configs:
                for col_config in self.source_configs[source_name]:
                    if col_config.visible_var:
                        col_config.visible_var.set(False)

        select_all_btn = tk.Button(
            select_buttons_frame,
            text="Select All",
            font=("Arial", 8),
            bg=self.theme.get("accent_blue", "#4a90d9"),
            fg="white",
            relief="flat",
            padx=6,
            pady=2,
            cursor="hand2",
            command=select_all_in_source,
        )
        select_all_btn.pack(side=tk.LEFT, padx=(0, 5))

        deselect_all_btn = tk.Button(
            select_buttons_frame,
            text="Deselect All",
            font=("Arial", 8),
            bg=self.theme.get("accent_orange", "#d97a4a"),
            fg="white",
            relief="flat",
            padx=6,
            pady=2,
            cursor="hand2",
            command=deselect_all_in_source,
        )
        deselect_all_btn.pack(side=tk.LEFT)

        # Get file modified date
        try:
            import os
            from datetime import datetime
            mtime = os.path.getmtime(schema.file_path)
            modified_date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            date_info = f" | Updated: {modified_date}"
        except:
            date_info = ""
        
        # Source name and info
        info_text = f"{source_name} ({schema.dataset_type.value.upper()}) - {len(columns)} columns, {schema.row_count:,} rows{date_info}"
        
        # File path (smaller, right side)
        path_label = tk.Label(
            header_frame,
            text=schema.file_path,
            font=("Arial", 8),
            bg=self.theme["secondary_bg"],
            fg=self.theme.get("subtext", "#888888"),
        )
        path_label.pack(side=tk.RIGHT, padx=10, pady=8)
        
        # Bind click to toggle
        def toggle_section(event=None):
            expanded = not self.source_expanded[source_name].get()
            self.source_expanded[source_name].set(expanded)
            expand_label.config(text="▼" if expanded else "▶")
            if expanded:
                self.source_frames[source_name].pack(fill=tk.X)
            else:
                self.source_frames[source_name].pack_forget()
        
        expand_label.bind("<Button-1>", toggle_section)
        header_label.bind("<Button-1>", toggle_section)
        
        # Content frame (collapsible)
        content_frame = tk.Frame(section_frame, bg=self.theme["background"])
        content_frame.pack(fill=tk.X)
        self.source_frames[source_name] = content_frame
        
        # Column headers
        self._create_column_headers(content_frame)
        
        # Column rows
        for col_config in columns:
            self._create_column_row(content_frame, source_name, col_config)
    
    def _create_column_headers(self, parent):
        """Create the header row for columns table using grid layout."""
        header_frame = tk.Frame(parent, bg=self.theme["field_bg"])
        header_frame.pack(fill=tk.X, padx=5, pady=(5, 0))

        headers = [
            ("Visible", self.COLUMN_WIDTHS["visible"]),
            ("Column Name", self.COLUMN_WIDTHS["name"]),
            ("Alias", self.COLUMN_WIDTHS["alias"]),
            ("Type", self.COLUMN_WIDTHS["type"]),
            ("Dec", self.COLUMN_WIDTHS["decimals"]),
            ("Null Handling", self.COLUMN_WIDTHS["nulls"]),
            ("Color Map", self.COLUMN_WIDTHS["colormap"]),
            ("Uniq", self.COLUMN_WIDTHS["unique"]),
            ("Nulls", self.COLUMN_WIDTHS["nullcount"]),
            ("Key", self.COLUMN_WIDTHS["key"]),
            ("Issues", self.COLUMN_WIDTHS["issues"]),
        ]

        for col_idx, (text, width) in enumerate(headers):
            label = tk.Label(
                header_frame,
                text=text,
                font=("Arial", 9, "bold"),
                bg=self.theme["field_bg"],
                fg=self.theme["text"],
                anchor="w",
            )
            label.grid(row=0, column=col_idx, sticky="w", padx=2, pady=3)
            header_frame.columnconfigure(col_idx, minsize=width)
    
    def _create_column_row(self, parent, source_name: str, config: ColumnDisplayConfig):
        """Create a row for a single column using grid layout."""
        # Row frame with alternating background
        row_idx = len([w for w in parent.winfo_children() if isinstance(w, tk.Frame)]) - 1
        row_bg = self.theme["background"] if row_idx % 2 == 0 else self.theme["secondary_bg"]

        # Check if this column has issues - highlight orange
        col_lower = config.source_name.lower()
        has_issues = col_lower in self.column_issues
        issue_messages = self.column_issues.get(col_lower, [])

        if has_issues:
            row_bg = self.theme.get("accent_orange", "#d97a4a")
        elif config.is_key:
            row_bg = self.theme.get("highlight_bg", "#3a5a7a")

        row_frame = tk.Frame(parent, bg=row_bg)

        # Store issue status for tooltip
        row_frame.has_issues = has_issues
        row_frame.issue_messages = issue_messages
        row_frame.pack(fill=tk.X, padx=5, pady=1)

        # Store tag for filtering
        row_frame.column_config = config

        col = 0

        # Visible checkbox
        config.visible_var = tk.BooleanVar(value=config.is_visible)
        visible_cb = tk.Checkbutton(
            row_frame,
            variable=config.visible_var,
            bg=row_bg,
            fg=self.theme["text"],
            selectcolor=self.theme.get("accent_green", "#47b881"),
            activebackground=row_bg,
        )
        visible_cb.grid(row=0, column=col, sticky="w", padx=2)
        row_frame.columnconfigure(col, minsize=self.COLUMN_WIDTHS["visible"])
        col += 1

        # Column name (read-only label)
        name_label = tk.Label(
            row_frame,
            text=config.source_name,
            font=("Arial", 9),
            bg=row_bg,
            fg=self.theme["text"],
            anchor="w",
        )
        name_label.grid(row=0, column=col, sticky="w", padx=2)
        row_frame.columnconfigure(col, minsize=self.COLUMN_WIDTHS["name"])
        col += 1

        # Alias entry
        config.alias_var = tk.StringVar(value=config.display_name)
        alias_entry = tk.Entry(
            row_frame,
            textvariable=config.alias_var,
            font=("Arial", 9),
            bg=self.theme["field_bg"],
            fg=self.theme["text"],
            width=18,
        )
        alias_entry.grid(row=0, column=col, sticky="w", padx=2)
        row_frame.columnconfigure(col, minsize=self.COLUMN_WIDTHS["alias"])
        col += 1

        # Data type dropdown (using tk.OptionMenu - works reliably unlike ttk.Combobox)
        config.type_var = tk.StringVar(value=config.data_type)
        type_menu = tk.OptionMenu(row_frame, config.type_var, *self.DATA_TYPES)
        type_menu.config(
            bg=self.theme["field_bg"],
            fg=self.theme["text"],
            activebackground=self.theme.get("accent_blue", "#4a90d9"),
            activeforeground="white",
            highlightthickness=0,
            width=8,
            font=("Arial", 9),
        )
        type_menu["menu"].config(
            bg=self.theme["field_bg"],
            fg=self.theme["text"],
            activebackground=self.theme.get("accent_blue", "#4a90d9"),
            activeforeground="white",
        )
        type_menu.grid(row=0, column=col, sticky="w", padx=2)
        row_frame.columnconfigure(col, minsize=self.COLUMN_WIDTHS["type"])
        col += 1

        # Decimals spinbox
        config.decimals_var = tk.IntVar(value=config.decimals)
        decimals_spin = tk.Spinbox(
            row_frame,
            from_=0,
            to=6,
            textvariable=config.decimals_var,
            width=4,
            font=("Arial", 9),
            bg=self.theme["field_bg"],
            fg=self.theme["text"],
        )
        decimals_spin.grid(row=0, column=col, sticky="w", padx=2)
        row_frame.columnconfigure(col, minsize=self.COLUMN_WIDTHS["decimals"])
        col += 1

        # Enable/disable decimals based on data type
        def update_decimals_state(*args):
            if config.type_var.get() == "numeric":
                decimals_spin.config(state="normal")
            else:
                decimals_spin.config(state="disabled")

        config.type_var.trace_add("write", update_decimals_state)
        update_decimals_state()

        # Null handling dropdown (using tk.OptionMenu - works reliably)
        config.nulls_var = tk.StringVar(value=config.null_handling)
        nulls_menu = tk.OptionMenu(row_frame, config.nulls_var, *self.NULL_HANDLING)
        nulls_menu.config(
            bg=self.theme["field_bg"],
            fg=self.theme["text"],
            activebackground=self.theme.get("accent_blue", "#4a90d9"),
            activeforeground="white",
            highlightthickness=0,
            width=8,
            font=("Arial", 9),
        )
        nulls_menu["menu"].config(
            bg=self.theme["field_bg"],
            fg=self.theme["text"],
            activebackground=self.theme.get("accent_blue", "#4a90d9"),
            activeforeground="white",
        )
        nulls_menu.grid(row=0, column=col, sticky="w", padx=2)
        row_frame.columnconfigure(col, minsize=self.COLUMN_WIDTHS["nulls"])
        col += 1

        # Color map dropdown + edit button frame
        colormap_frame = tk.Frame(row_frame, bg=row_bg)
        colormap_frame.grid(row=0, column=col, sticky="w", padx=2)
        row_frame.columnconfigure(col, minsize=self.COLUMN_WIDTHS["colormap"])
        col += 1

        available_maps = ["(None)"]
        if self.data_coordinator and self.data_coordinator.color_maps:
            try:
                if config.data_type == "numeric":
                    available_maps.extend(self.data_coordinator.color_maps.list_numeric())
                elif config.data_type == "categorical":
                    available_maps.extend(self.data_coordinator.color_maps.list_categorical())
                else:
                    available_maps.extend(self.data_coordinator.color_maps.list_all())
            except Exception:
                pass

        config.colormap_var = tk.StringVar(value=config.color_map or "(None)")
        colormap_menu = tk.OptionMenu(colormap_frame, config.colormap_var, *available_maps)
        colormap_menu.config(
            bg=self.theme["field_bg"],
            fg=self.theme["text"],
            activebackground=self.theme.get("accent_blue", "#4a90d9"),
            activeforeground="white",
            highlightthickness=0,
            width=8,
            font=("Arial", 9),
        )
        colormap_menu["menu"].config(
            bg=self.theme["field_bg"],
            fg=self.theme["text"],
            activebackground=self.theme.get("accent_blue", "#4a90d9"),
            activeforeground="white",
        )
        colormap_menu.pack(side=tk.LEFT)
        config.colormap_menu = colormap_menu

        edit_btn = tk.Button(
            colormap_frame,
            text="...",
            font=("Arial", 8),
            width=2,
            bg=self.theme["field_bg"],
            fg=self.theme["text"],
            activebackground=self.theme.get("accent_blue", "#4a90d9"),
            activeforeground="white",
            command=lambda c=config, s=source_name: self._open_color_map_editor(c, s),
        )
        edit_btn.pack(side=tk.LEFT, padx=2)

        # Unique count
        unique_label = tk.Label(
            row_frame,
            text=f"{config.unique_count:,}" if config.unique_count > 0 else "-",
            font=("Arial", 9),
            bg=row_bg,
            fg=self.theme["text"],
            anchor="e",
        )
        unique_label.grid(row=0, column=col, sticky="e", padx=2)
        row_frame.columnconfigure(col, minsize=self.COLUMN_WIDTHS["unique"])
        col += 1

        # Null count with percentage
        null_pct = (config.null_count / config.total_count * 100) if config.total_count > 0 else 0
        null_text = f"{config.null_count:,} ({null_pct:.0f}%)" if config.total_count > 0 else "-"
        null_label = tk.Label(
            row_frame,
            text=null_text,
            font=("Arial", 9),
            bg=row_bg,
            fg=self.theme["accent_red"] if null_pct > 50 else self.theme["text"],
            anchor="e",
        )
        null_label.grid(row=0, column=col, sticky="e", padx=2)
        row_frame.columnconfigure(col, minsize=self.COLUMN_WIDTHS["nullcount"])
        col += 1

        # Key indicator
        key_text = "K" if config.is_key else ""
        key_label = tk.Label(
            row_frame,
            text=key_text,
            font=("Arial", 10, "bold"),
            bg=row_bg,
            fg=self.theme["accent_blue"] if config.is_key else self.theme.get("subtext", "#888"),
        )
        key_label.grid(row=0, column=col, sticky="w", padx=2)
        row_frame.columnconfigure(col, minsize=self.COLUMN_WIDTHS["key"])

        # Toggle key status on click
        def toggle_key_status(event):
            config.is_key = not config.is_key
            key_label.config(
                text="K" if config.is_key else "",
                fg=self.theme["accent_blue"] if config.is_key else self.theme.get("subtext", "#888")
            )
            # Update row background
            new_bg = self.theme.get("highlight_bg", "#3a5a7a") if config.is_key else row_bg
            row_frame.config(bg=new_bg)
            for child in row_frame.winfo_children():
                if isinstance(child, (tk.Label, tk.Checkbutton, tk.Frame)):
                    try:
                        child.config(bg=new_bg)
                    except tk.TclError:
                        pass

        key_label.bind("<Button-1>", toggle_key_status)
        key_label.config(cursor="hand2")
        col += 1

        # Issues column - show data warnings/errors for this column
        issues_text = ""
        if has_issues and issue_messages:
            # Join multiple issues with semicolon, truncate if too long
            issues_text = "; ".join(issue_messages)
            if len(issues_text) > 50:
                issues_text = issues_text[:47] + "..."

        issues_label = tk.Label(
            row_frame,
            text=issues_text,
            font=("Arial", 8),
            bg=row_bg,
            fg="white" if has_issues else self.theme.get("subtext", "#888"),
            anchor="w",
        )
        issues_label.grid(row=0, column=col, sticky="w", padx=2)
        row_frame.columnconfigure(col, minsize=self.COLUMN_WIDTHS["issues"])

        # Add tooltip for full issue text if truncated
        if has_issues and issue_messages:
            full_issues_text = "\n".join(issue_messages)
            def show_tooltip(event, text=full_issues_text):
                # Simple tooltip - create temporary label
                tooltip = tk.Toplevel(row_frame)
                tooltip.wm_overrideredirect(True)
                tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
                label = tk.Label(
                    tooltip,
                    text=text,
                    bg="#333333",
                    fg="white",
                    font=("Arial", 9),
                    padx=5,
                    pady=3,
                    justify=tk.LEFT,
                )
                label.pack()
                # Store reference to destroy later
                issues_label._tooltip = tooltip

            def hide_tooltip(event):
                if hasattr(issues_label, '_tooltip'):
                    issues_label._tooltip.destroy()
                    del issues_label._tooltip

            issues_label.bind("<Enter>", show_tooltip)
            issues_label.bind("<Leave>", hide_tooltip)


    def _open_color_map_editor(self, config: ColumnDisplayConfig, source_name: str):
        """
        Open the color map editor dialog for a column.
        
        Args:
            config: ColumnDisplayConfig for the column
            source_name: Name of the data source containing this column
        """
        try:
            # Lazy import to avoid circular imports
            from gui.color_map_editor_dialog import ColorMapEditorDialog
        except ImportError:
            logger.error("Could not import ColorMapEditorDialog")
            DialogHelper.show_message(
                self.dialog,
                "Error",
                "Color Map Editor is not available.",
                message_type="error"
            )
            return
        
        # Get color map manager
        color_map_manager = None
        if self.data_coordinator and self.data_coordinator.color_maps:
            color_map_manager = self.data_coordinator.color_maps.manager
        
        if color_map_manager is None:
            DialogHelper.show_message(
                self.dialog,
                "Error",
                "Color Map Manager is not available.\nPlease check if color maps loaded correctly.",
                message_type="error"
            )
            return
        
        # Get current color map if selected
        current_map_name = config.colormap_var.get() if config.colormap_var else "(None)"
        initial_color_map = None
        if current_map_name and current_map_name != "(None)":
            initial_color_map = color_map_manager.get_preset(current_map_name)
        
        # Get data values for the histogram preview
        data_values = None
        if self.data_coordinator:
            try:
                data_values = self.data_coordinator.get_column_values(config.source_name)
                logger.debug(f"Fetched {len(data_values) if data_values else 0} values for column '{config.source_name}'")
            except Exception as e:
                logger.debug(f"Could not fetch data values for '{config.source_name}': {e}")
        
        # Callback to refresh dropdown after save
        def on_color_map_saved():
            self._refresh_colormap_dropdown(config)
        
        # Open the editor dialog
        logger.info(f"Opening color map editor for column '{config.source_name}'")
        
        editor = ColorMapEditorDialog(
            parent=self.dialog,
            gui_manager=self.gui_manager,
            color_map_manager=color_map_manager,
            data_column=config.source_name,
            data_values=data_values,
            initial_color_map=initial_color_map,
            on_save_callback=on_color_map_saved,
            data_coordinator=self.data_coordinator,
        )
        
        result = editor.show()
        
        # If a new color map was created, select it
        if result and hasattr(result, 'name'):
            config.colormap_var.set(result.name)
            logger.info(f"Selected new color map: {result.name}")
    
    def _refresh_colormap_dropdown(self, config: ColumnDisplayConfig):
        """
        Refresh the color map dropdown options after a new map is created.

        Args:
            config: ColumnDisplayConfig with the dropdown to refresh
        """
        if not config.colormap_menu:
            return

        # Get current selection
        current_value = config.colormap_var.get() if config.colormap_var else "(None)"

        # Build new list of available maps
        available_maps = ["(None)"]
        if self.data_coordinator and self.data_coordinator.color_maps:
            try:
                if config.data_type == "numeric":
                    available_maps.extend(self.data_coordinator.color_maps.list_numeric())
                elif config.data_type == "categorical":
                    available_maps.extend(self.data_coordinator.color_maps.list_categorical())
                else:
                    available_maps.extend(self.data_coordinator.color_maps.list_all())
            except Exception as e:
                logger.debug(f"Could not refresh color maps: {e}")

        # Update OptionMenu - clear and repopulate
        menu = config.colormap_menu["menu"]
        menu.delete(0, "end")
        for map_name in available_maps:
            menu.add_command(
                label=map_name,
                command=lambda v=map_name: config.colormap_var.set(v)
            )

        # Restore selection if still valid, otherwise keep current
        if current_value in available_maps:
            config.colormap_var.set(current_value)

        logger.debug(f"Refreshed colormap dropdown with {len(available_maps)} options")
    
    def _expand_all(self):
        """Expand all source sections."""
        for source_name in self.source_expanded:
            self.source_expanded[source_name].set(True)
            self.source_frames[source_name].pack(fill=tk.X)
    
    def _collapse_all(self):
        """Collapse all source sections."""
        for source_name in self.source_expanded:
            self.source_expanded[source_name].set(False)
            self.source_frames[source_name].pack_forget()
    
    def _reset_to_defaults(self):
        """Reset all columns to their inferred defaults."""
        result = DialogHelper.confirm_dialog(
            self.dialog,
            "Reset to Defaults",
            "This will reset all column settings to their auto-detected defaults.\n\nContinue?",
            yes_text="Reset",
            no_text="Cancel",
        )
        
        if result:
            # Reload from coordinator
            for widget in self.scrollable_frame.winfo_children():
                widget.destroy()
            
            self.source_configs.clear()
            self.source_frames.clear()
            self.source_expanded.clear()
            
            self._load_column_data()
            
            DialogHelper.show_message(
                self.dialog,
                "Reset Complete",
                "Column settings have been reset to defaults.",
                message_type="info",
            )
    
    def _on_save(self):
        """Save column settings."""
        logger.info("Saving column settings...")

        # Track visibility changes for debugging
        visibility_changes = []

        # Update schemas in geological store
        if self.data_coordinator and self.data_coordinator._geological_store:
            geo_store = self.data_coordinator._geological_store

            for source_name, columns in self.source_configs.items():
                if source_name not in geo_store._sources:
                    continue

                indexed_source = geo_store._sources[source_name]
                schema = indexed_source.schema

                for config in columns:
                    col_name = config.source_name
                    if col_name not in schema.columns:
                        continue

                    col_schema = schema.columns[col_name]

                    # Get new visibility value from checkbox
                    new_visible = config.visible_var.get() if config.visible_var else config.is_visible
                    old_visible = col_schema.is_visible

                    # Track changes for debugging
                    if new_visible != old_visible:
                        visibility_changes.append(f"{source_name}.{col_name}: {old_visible} -> {new_visible}")

                    # Update from UI
                    col_schema.display_name = config.alias_var.get() if config.alias_var else config.display_name
                    col_schema.is_visible = new_visible
                    col_schema.is_key_column = config.is_key
                    
                    # Data type
                    if config.type_var:
                        from processing.DataManager.schema import DataType
                        col_schema.data_type = DataType.from_string(config.type_var.get())
                    
                    # Null handling
                    if config.nulls_var:
                        from processing.DataManager.schema import NullHandling
                        col_schema.null_handling = NullHandling(config.nulls_var.get())
                    
                    # Decimals
                    if config.decimals_var:
                        col_schema.decimals = config.decimals_var.get()
                    
                    # Color map
                    if config.colormap_var:
                        colormap_value = config.colormap_var.get()
                        col_schema.color_map = None if colormap_value == "(None)" else colormap_value
                    
                    # Only save categories for categorical with <100 unique values
                    if col_schema.data_type == DataType.CATEGORICAL and config.unique_count >= 100:
                        col_schema.categories = None
        
        # Serialize schemas to config
        if self.config_manager and self.data_coordinator and self.data_coordinator._geological_store:
            geo_store = self.data_coordinator._geological_store
            schemas_dict = {}
            
            for source_name, indexed_source in geo_store.get_data_sources().items():
                schema = indexed_source.schema
                schemas_dict[source_name] = schema.to_dict()
            
            self.config_manager.set("column_schemas", schemas_dict)
            logger.info(f"Saved column settings for {len(schemas_dict)} sources to config")

            # Log visibility changes for debugging
            if visibility_changes:
                logger.info(f"Visibility changes saved: {len(visibility_changes)} columns")
                for change in visibility_changes[:10]:  # Log first 10
                    logger.debug(f"  {change}")
            else:
                logger.debug("No visibility changes to save")
        
        # Save register column settings
        if self.config_manager and self.register_configs:
            register_settings = {}
            for section_key, columns in self.register_configs.items():
                section_settings = {}
                for config in columns:
                    section_settings[config.source_name] = {
                        "display_name": config.alias_var.get() if config.alias_var else config.display_name,
                        "data_type": config.type_var.get() if config.type_var else config.data_type,
                        "decimals": config.decimals_var.get() if config.decimals_var else config.decimals,
                        "null_handling": config.nulls_var.get() if config.nulls_var else config.null_handling,
                        "is_visible": config.visible_var.get() if config.visible_var else config.is_visible,
                        "color_map": config.colormap_var.get() if config.colormap_var and config.colormap_var.get() != "(None)" else None,
                    }
                register_settings[section_key] = section_settings
            
            self.config_manager.set("register_column_settings", register_settings)
            logger.info(f"Saved register column settings for {len(register_settings)} sections to config")
        
        # Call callback
        if self.on_save_callback:
            self.on_save_callback()

        self.result = True

        # Close the dialog immediately after save (no confirmation dialog)
        self._on_close()

    def _on_close(self):
        """Close the dialog."""
        try:
            # Unbind mousewheel to prevent errors after dialog closes
            if hasattr(self, 'canvas') and self.canvas.winfo_exists():
                self.canvas.unbind_all("<MouseWheel>")
        except tk.TclError:
            pass  # Canvas already destroyed

        try:
            if hasattr(self, 'dialog') and self.dialog.winfo_exists():
                self.dialog.destroy()
        except tk.TclError:
            pass  # Dialog already destroyed
    
    def show(self):
        """Show the dialog and wait for it to close."""
        # Bring dialog to front and focus
        self.dialog.lift()
        self.dialog.focus_force()

        # Wait for the dialog to close
        self.dialog.wait_window()
        return self.result


# =============================================================================
# Test Function
# =============================================================================

def test_column_settings_dialog():
    """Test function to run the dialog standalone."""
    import sys
    
    print("Starting ColumnSettingsDialog test...")
    
    root = tk.Tk()
    root.withdraw()
    
    # Create dummy gui_manager
    class DummyGUIManager:
        theme_colors = {
            "background": "#2b2b2b",
            "secondary_bg": "#3c3c3c",
            "field_bg": "#4a4a4a",
            "field_border": "#5a5a5a",
            "text": "#ffffff",
            "border": "#5a5a5a",
            "accent_blue": "#4a6fae",
            "accent_green": "#5a9f5a",
            "accent_red": "#c74440",
            "checkbox_bg": "#4a4a4a",
            "highlight_bg": "#3a5a7a",
        }
        
        def style_dropdown(self, dropdown, width=None):
            pass
    
    gui_manager = DummyGUIManager()
    
    # Create dialog (no data coordinator - will show "no data" message)
    dialog = ColumnSettingsDialog(
        parent=root,
        gui_manager=gui_manager,
        data_coordinator=None,
        config_manager=None,
    )
    
    result = dialog.show()
    
    print(f"Dialog result: {result}")
    print("Test complete")
    
    root.destroy()


if __name__ == "__main__":
    test_column_settings_dialog()
