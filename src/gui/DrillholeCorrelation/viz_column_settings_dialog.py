"""
Visualization Column Settings Dialog for Correlation.
Allows configuration of data columns displayed alongside drillhole images.
With searchable dropdowns and columns grouped by data source.
"""

import tkinter as tk
from tkinter import ttk
import logging
from typing import List, Dict, Any, Optional, Tuple

from gui.dialog_helper import DialogHelper
from gui.widgets.modern_button import ModernButton

logger = logging.getLogger(__name__)


class VizColumnSettingsDialog(tk.Toplevel):
    """
    Dialog for configuring visualization columns in correlation view.
    
    Features:
    - Searchable column dropdowns grouped by data source
    - Color map selection with preview
    - Visualization type selection (line/bar)
    - Add/remove/reorder columns
    """
    
    def __init__(
        self,
        parent: tk.Widget,
        gui_manager: Any,
        config_manager: Any,
        data_manager: Any,
        color_map_manager: Optional[Any] = None,
        translator: Optional[Any] = None,
        dialog_helper: Optional[Any] = None,
        current_columns: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize visualization settings dialog.
        
        Args:
            parent: Parent widget
            gui_manager: GUIManager for theming
            config_manager: ConfigManager for settings
            data_manager: DataCoordinator for available columns
            color_map_manager: ColorMapManager for color presets
            translator: Translator for i18n
            dialog_helper: DialogHelper for dialogs
            current_columns: Current visualization column configuration
        """
        super().__init__(parent)
        
        logger.info("Initializing VizColumnSettingsDialog")
        
        # Store managers
        self.gui_manager = gui_manager
        self.config_manager = config_manager
        self.data_manager = data_manager
        self.color_map_manager = color_map_manager
        self.translator = translator or self._create_dummy_translator()
        self.dialog_helper = dialog_helper or DialogHelper(self, translator=self.translator, gui_manager=gui_manager)
        
        # Theme shortcuts
        self.theme_colors = gui_manager.theme_colors
        self.fonts = gui_manager.fonts
        
        # Dialog properties
        self.title(self.translator.translate("Visualization Settings"))
        self.geometry("900x650")
        self.configure(bg=self.theme_colors["background"])
        
        # Data
        self.current_columns = current_columns or config_manager.get("correlation_viz_columns", [])
        self.column_rows: List[VizColumnRow] = []
        
        # Get columns grouped by source
        self.columns_by_source = self._get_columns_by_source()
        self.available_color_maps = self._get_available_color_maps()
        
        # Flatten for searchable dropdown
        self.all_columns_flat = self._flatten_columns()
        
        # Result
        self.result: Optional[List[Dict[str, Any]]] = None
        
        logger.debug(f"Available sources: {list(self.columns_by_source.keys())}")
        logger.debug(f"Total columns: {len(self.all_columns_flat)}")
        logger.debug(f"Available color maps: {len(self.available_color_maps)}")
        logger.debug(f"Current columns: {self.current_columns}")
        
        # Build UI
        self._create_ui()
        
        # Load current configuration
        self._load_current_config()
        
        # Make modal (but don't grab_set - it prevents combobox dropdowns from working)
        self.transient(parent)
        # self.grab_set()  # Disabled: prevents ttk.Combobox dropdowns from appearing
        
        # Center dialog
        self._center_dialog()
    
    def _create_dummy_translator(self) -> Any:
        """Create dummy translator if none provided."""
        class DummyTranslator:
            def translate(self, text: str) -> str:
                return text
        return DummyTranslator()
    
    def _get_columns_by_source(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Get columns grouped by data source.
        
        Returns:
            Dict of {source_name: [(column_name, alias), ...]}
        """
        columns_by_source = {}
        
        try:
            # Try to get from data_manager's geological_store
            if hasattr(self.data_manager, 'geological_store'):
                geo_store = self.data_manager.geological_store
                if hasattr(geo_store, 'get_available_columns'):
                    raw_columns = geo_store.get_available_columns()
                    
                    for source_name, cols in raw_columns.items():
                        # cols is [(column_name, data_type), ...]
                        # Filter to numeric columns and create aliases
                        source_cols = []
                        for col_name, data_type in cols:
                            # Skip internal columns
                            if col_name.startswith('_'):
                                continue
                            # Create alias (cleaner display name)
                            alias = self._create_column_alias(col_name)
                            source_cols.append((col_name, alias))
                        
                        if source_cols:
                            # Use cleaner source name
                            display_name = self._clean_source_name(source_name)
                            columns_by_source[display_name] = sorted(source_cols, key=lambda x: x[1])
                    
                    logger.debug(f"Got columns from {len(columns_by_source)} sources")

            # Add RC metrics if available
            if hasattr(self.data_manager, 'has_rc_metrics') and self.data_manager.has_rc_metrics:
                try:
                    rc_df = self.data_manager.get_rc_metrics_dataframe()
                    if rc_df is not None and not rc_df.empty:
                        rc_cols = []
                        for col in rc_df.columns:
                            if col.lower() in ('hole_id', 'depth_to'):
                                continue
                            alias = self._create_column_alias(col)
                            rc_cols.append((col, alias))
                        if rc_cols:
                            columns_by_source["RC Metrics"] = sorted(rc_cols, key=lambda x: x[1])
                            logger.debug(f"Added {len(rc_cols)} RC metrics columns")
                except Exception as e:
                    logger.warning(f"Could not add RC metrics: {e}")

            if columns_by_source:
                return columns_by_source

            # Fallback to direct data_manager method
            if hasattr(self.data_manager, 'get_available_columns'):
                columns = self.data_manager.get_available_columns()
                if isinstance(columns, dict):
                    for source_name, cols in columns.items():
                        source_cols = [(c, self._create_column_alias(c)) for c in cols if not c.startswith('_')]
                        if source_cols:
                            display_name = self._clean_source_name(source_name)
                            columns_by_source[display_name] = sorted(source_cols, key=lambda x: x[1])
                elif isinstance(columns, list):
                    columns_by_source["Data"] = [(c, self._create_column_alias(c)) for c in columns if not c.startswith('_')]
                return columns_by_source
            
        except Exception as e:
            logger.error(f"Error getting columns: {e}")
        
        # Fallback to common columns
        columns_by_source["Assay"] = [
            ("fe_pct_best", "Fe %"),
            ("sio2_pct_best", "SiO2 %"),
            ("al2o3_pct_best", "Al2O3 %"),
            ("p_pct_best", "P %"),
            ("mn_pct_best", "Mn %"),
            ("loi_pct_best", "LOI %"),
        ]
        logger.debug("Using fallback column list")
        
        return columns_by_source
    
    def _clean_source_name(self, name: str) -> str:
        """Clean up source name for display."""
        # Remove file extensions
        name = name.replace('.csv', '').replace('.CSV', '')
        # Remove common prefixes
        for prefix in ['ex', 'EX', 'Ex']:
            if name.lower().startswith(prefix.lower()) and len(name) > len(prefix):
                name = name[len(prefix):]
        # Capitalize first letter
        return name.capitalize() if name else name
    
    def _create_column_alias(self, column_name: str) -> str:
        """Create a cleaner alias for a column name."""
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
        }
        for old, new in chemical_fixes.items():
            alias = alias.replace(old, new)
        
        return alias
    
    def _flatten_columns(self) -> List[Tuple[str, str, str]]:
        """
        Flatten columns into a single list for searchable dropdown.
        
        Returns:
            List of (column_name, alias, source_name) tuples
        """
        flat = []
        for source_name, cols in self.columns_by_source.items():
            for col_name, alias in cols:
                flat.append((col_name, alias, source_name))
        return flat
    
    def _get_available_color_maps(self) -> List[str]:
        """Get list of available color map presets."""
        presets = []
        
        try:
            if self.color_map_manager:
                if hasattr(self.color_map_manager, 'get_preset_names'):
                    presets = self.color_map_manager.get_preset_names()
                elif hasattr(self.color_map_manager, 'list_presets'):
                    presets = self.color_map_manager.list_presets()
                logger.debug(f"Got {len(presets)} color maps from manager")
        except Exception as e:
            logger.error(f"Error getting color maps: {e}")
        
        if not presets:
            # Fallback presets
            presets = [
                "fe_grade", "sio2_grade", "al2o3_grade",
                "viridis", "plasma", "coolwarm"
            ]
            logger.debug("Using fallback color map list")
        
        return sorted(presets)
    
    def _create_ui(self) -> None:
        """Create the dialog UI."""
        logger.debug("Creating UI")
        
        # Main container
        main_frame = tk.Frame(self, bg=self.theme_colors["background"])
        main_frame.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Title
        title_label = tk.Label(
            main_frame,
            text=self.translator.translate("Configure Data Visualization Columns"),
            font=self.fonts["heading"],
            bg=self.theme_colors["background"],
            fg=self.theme_colors["text"]
        )
        title_label.pack(pady=(0, 5))
        
        # Instructions
        instructions = tk.Label(
            main_frame,
            text=self.translator.translate("Add data columns to display alongside drillhole images. Use the search box to find columns."),
            font=self.fonts["normal"],
            bg=self.theme_colors["background"],
            fg=self.theme_colors["text"],
            wraplength=800
        )
        instructions.pack(pady=(0, 15))
        
        # Column list frame with scrollbar
        list_frame = tk.Frame(
            main_frame,
            bg=self.theme_colors["secondary_bg"],
            highlightbackground=self.theme_colors["border"],
            highlightthickness=1
        )
        list_frame.pack(fill="both", expand=True, pady=10)
        
        # Header row
        header_frame = tk.Frame(list_frame, bg=self.theme_colors["secondary_bg"])
        header_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        tk.Label(
            header_frame,
            text=self.translator.translate("Column"),
            font=self.fonts["bold"] if hasattr(self.fonts, "bold") else self.fonts["normal"],
            bg=self.theme_colors["secondary_bg"],
            fg=self.theme_colors["text"],
            width=35,
            anchor="w"
        ).pack(side="left", padx=(0, 20))
        
        tk.Label(
            header_frame,
            text=self.translator.translate("Color Map"),
            font=self.fonts["bold"] if hasattr(self.fonts, "bold") else self.fonts["normal"],
            bg=self.theme_colors["secondary_bg"],
            fg=self.theme_colors["text"],
            width=20,
            anchor="w"
        ).pack(side="left", padx=(0, 20))
        
        tk.Label(
            header_frame,
            text=self.translator.translate("Type"),
            font=self.fonts["bold"] if hasattr(self.fonts, "bold") else self.fonts["normal"],
            bg=self.theme_colors["secondary_bg"],
            fg=self.theme_colors["text"],
            width=10,
            anchor="w"
        ).pack(side="left")
        
        # Separator
        ttk.Separator(list_frame, orient="horizontal").pack(fill="x", padx=10)
        
        # Scrollable area
        scroll_frame = tk.Frame(list_frame, bg=self.theme_colors["secondary_bg"])
        scroll_frame.pack(fill="both", expand=True)
        
        self.canvas = tk.Canvas(
            scroll_frame,
            bg=self.theme_colors["secondary_bg"],
            highlightthickness=0
        )
        scrollbar = ttk.Scrollbar(scroll_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # Frame inside canvas for column rows
        self.columns_container = tk.Frame(
            self.canvas,
            bg=self.theme_colors["secondary_bg"]
        )
        self.canvas_window = self.canvas.create_window(
            0, 0,
            anchor="nw",
            window=self.columns_container
        )
        
        # Pack canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind canvas events
        self.columns_container.bind("<Configure>", self._on_container_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Mousewheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Add column button
        add_frame = tk.Frame(main_frame, bg=self.theme_colors["background"])
        add_frame.pack(fill="x", pady=10)
        
        ModernButton(
            add_frame,
            text="+ " + self.translator.translate("Add Column"),
            command=self._add_column_row,
            color=self.theme_colors["accent_green"],
            theme_colors=self.theme_colors
        ).pack(side="left")
        
        # Button frame
        button_frame = tk.Frame(main_frame, bg=self.theme_colors["background"])
        button_frame.pack(fill="x", pady=(10, 0))
        
        ModernButton(
            button_frame,
            text=self.translator.translate("Cancel"),
            command=self._on_cancel,
            color=self.theme_colors["secondary_bg"],
            theme_colors=self.theme_colors
        ).pack(side="right", padx=5)
        
        ModernButton(
            button_frame,
            text=self.translator.translate("Apply"),
            command=self._on_ok,
            color=self.theme_colors["accent_green"],
            theme_colors=self.theme_colors
        ).pack(side="right", padx=5)
        
        # Apply to all checkbox
        self.apply_to_all_var = tk.BooleanVar(value=True)
        apply_check = tk.Checkbutton(
            button_frame,
            text=self.translator.translate("Save as default"),
            variable=self.apply_to_all_var,
            bg=self.theme_colors["background"],
            fg=self.theme_colors["text"],
            selectcolor=self.theme_colors["field_bg"],
            activebackground=self.theme_colors["background"],
            font=self.fonts["normal"]
        )
        apply_check.pack(side="left", padx=5)
    
    def _on_canvas_configure(self, event: tk.Event) -> None:
        """Handle canvas resize."""
        # Update width of container to match canvas
        self.canvas.itemconfig(self.canvas_window, width=event.width)
    
    def _on_mousewheel(self, event: tk.Event) -> None:
        """Handle mousewheel scrolling."""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def _load_current_config(self) -> None:
        """Load current column configuration."""
        logger.debug(f"Loading {len(self.current_columns)} current columns")
        
        for col_config in self.current_columns:
            self._add_column_row(
                column=col_config.get("column", ""),
                color_map=col_config.get("color_map", ""),
                viz_type=col_config.get("type", "bar")
            )
    
    def _add_column_row(self, column: str = "", color_map: str = "", viz_type: str = "bar") -> None:
        """Add a new column configuration row."""
        logger.debug(f"Adding column row: column={column}, color_map={color_map}, type={viz_type}")
        
        row = VizColumnRow(
            parent=self.columns_container,
            gui_manager=self.gui_manager,
            columns_by_source=self.columns_by_source,
            all_columns_flat=self.all_columns_flat,
            available_color_maps=self.available_color_maps,
            column=column,
            color_map=color_map,
            viz_type=viz_type,
            on_remove=lambda r: self._remove_row(r),
            translator=self.translator
        )
        
        self.column_rows.append(row)
        
        # Update canvas scroll region
        self.canvas.after(10, self._update_scroll_region)
    
    def _remove_row(self, row: 'VizColumnRow') -> None:
        """Remove a column row."""
        logger.debug(f"Removing column row")
        
        if row in self.column_rows:
            self.column_rows.remove(row)
            row.destroy()
            self._update_scroll_region()
    
    def _on_container_configure(self, event: tk.Event) -> None:
        """Handle container resize."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def _update_scroll_region(self) -> None:
        """Update canvas scroll region."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def _on_ok(self) -> None:
        """Handle OK button."""
        logger.info("Saving visualization settings")
        
        # Collect configuration
        self.result = []
        
        for row in self.column_rows:
            config = row.get_config()
            if config["column"] and config["color_map"]:
                self.result.append(config)
                logger.debug(f"  Added: {config}")
        
        logger.info(f"Saved {len(self.result)} column configurations")
        
        # Save to config if apply to all
        if self.apply_to_all_var.get():
            self.config_manager.set("correlation_viz_columns", self.result)
            logger.debug("Saved to config manager")
        
        self.destroy()
    
    def _on_cancel(self) -> None:
        """Handle Cancel button."""
        logger.info("Canceling visualization settings")
        self.result = None
        self.destroy()
    
    def _center_dialog(self) -> None:
        """Center dialog on parent."""
        self.update_idletasks()
        
        parent = self.master
        if parent:
            x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
            y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
            self.geometry(f"+{x}+{y}")


class VizColumnRow(tk.Frame):
    """
    Single row for column configuration with searchable grouped dropdown.
    """
    
    def __init__(
        self,
        parent: tk.Widget,
        gui_manager: Any,
        columns_by_source: Dict[str, List[Tuple[str, str]]],
        all_columns_flat: List[Tuple[str, str, str]],
        available_color_maps: List[str],
        column: str = "",
        color_map: str = "",
        viz_type: str = "bar",
        on_remove: Optional[Any] = None,
        translator: Optional[Any] = None
    ):
        """Initialize column row."""
        super().__init__(parent, bg=gui_manager.theme_colors["secondary_bg"])
        
        self.gui_manager = gui_manager
        self.columns_by_source = columns_by_source
        self.all_columns_flat = all_columns_flat
        self.on_remove = on_remove
        self.translator = translator or self._dummy_translator()
        
        # Theme shortcuts
        self.theme_colors = gui_manager.theme_colors
        self.fonts = gui_manager.fonts
        
        # Variables
        self.column_var = tk.StringVar(value=column)
        self.color_map_var = tk.StringVar(value=color_map)
        self.viz_type_var = tk.StringVar(value=viz_type)
        
        # Build column name -> alias lookup
        self.column_aliases = {col: alias for col, alias, _ in all_columns_flat}
        
        # Build UI
        self._build_ui(available_color_maps)
        
        # Pack self
        self.pack(fill="x", padx=10, pady=3)
    
    def _dummy_translator(self) -> Any:
        """Dummy translator."""
        class DT:
            def translate(self, text: str) -> str:
                return text
        return DT()
    
    def _build_ui(self, color_maps: List[str]) -> None:
        """Build row UI with searchable dropdown."""
        # Row frame with border
        row_frame = tk.Frame(
            self,
            bg=self.theme_colors["field_bg"],
            highlightbackground=self.theme_colors["border"],
            highlightthickness=1
        )
        row_frame.pack(fill="x", expand=True)
        
        # Content frame
        content = tk.Frame(row_frame, bg=self.theme_colors["field_bg"])
        content.pack(fill="x", padx=10, pady=8)
        
        # Column dropdown (searchable grouped)
        column_frame = tk.Frame(content, bg=self.theme_colors["field_bg"])
        column_frame.pack(side="left", padx=(0, 15))
        
        # Try to use gui_manager's searchable dropdown
        try:
            self.column_dropdown = self.gui_manager.create_searchable_optionmenu(
                parent=column_frame,
                items=self._build_grouped_items(),
                variable=self.column_var,
                width=30,
                placeholder="Search columns...",
                on_change=None
            )
            self.column_dropdown.pack()
        except Exception as e:
            logger.debug(f"Using fallback dropdown: {e}")
            # Fallback to standard combobox
            self.column_dropdown = ttk.Combobox(
                column_frame,
                textvariable=self.column_var,
                values=[col for col, _, _ in self.all_columns_flat],
                width=30,
                state="readonly"
            )
            self.column_dropdown.pack()
        
        # Color map dropdown
        color_frame = tk.Frame(content, bg=self.theme_colors["field_bg"])
        color_frame.pack(side="left", padx=(0, 15))
        
        color_menu = ttk.Combobox(
            color_frame,
            textvariable=self.color_map_var,
            values=color_maps,
            width=18,
            state="readonly"
        )
        color_menu.pack()
        
        # Viz type dropdown
        type_frame = tk.Frame(content, bg=self.theme_colors["field_bg"])
        type_frame.pack(side="left", padx=(0, 15))
        
        type_menu = ttk.Combobox(
            type_frame,
            textvariable=self.viz_type_var,
            values=["bar", "line"],
            width=8,
            state="readonly"
        )
        type_menu.pack()
        
        # Remove button
        ModernButton(
            content,
            text="✕",
            command=lambda: self.on_remove(self) if self.on_remove else None,
            color=self.theme_colors["accent_red"],
            theme_colors=self.theme_colors,
            width=30,
            height=25
        ).pack(side="right")
    
    def _build_grouped_items(self) -> List[str]:
        """
        Build flat list with source group headers for searchable dropdown.
        
        Returns list like:
        ["-- Assay --", "Fe %", "SiO2 %", "-- Geology --", "Lithology", ...]
        """
        items = []
        for source_name, cols in sorted(self.columns_by_source.items()):
            # Add group header
            items.append(f"── {source_name} ──")
            # Add columns (display alias)
            for col_name, alias in cols:
                items.append(alias)
        return items
    
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration for this row."""
        # Map alias back to column name
        display_value = self.column_var.get()
        column_name = display_value
        display_name = display_value
        
        # Check if it's an alias
        for col, alias, _ in self.all_columns_flat:
            if alias == display_value:
                column_name = col
                display_name = alias
                break
        
        # Skip group headers
        if display_value.startswith("──"):
            column_name = ""
            display_name = ""
        
        return {
            "column": column_name.lower() if column_name else "",  # Normalize to lowercase
            "display_name": display_name,  # Keep alias for display
            "color_map": self.color_map_var.get(),
            "type": self.viz_type_var.get()
        }
    
    def destroy(self) -> None:
        """Clean up row."""
        logger.debug("Destroying VizColumnRow")
        super().destroy()