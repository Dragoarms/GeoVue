"""
CorrelationVizSettingsDialog - Configure data visualization columns for correlation.
Based on viz_column_settings_dialog.py but with additional options for line/bar charts
and axis limits.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple

from gui.widgets.modern_button import ModernButton
from gui.widgets.themed_searchable_optionmenu import ThemedSearchableOptionMenu
from gui.DrillholeCorrelation.trace_processing import (
    SCALE_MODE_LABELS,
    canonical_scale_mode,
    scale_mode_display_name,
)

logger = logging.getLogger(__name__)


class CorrelationVizSettingsDialog(tk.Toplevel):
    """
    Dialog for configuring correlation data visualization columns.
    
    Features:
    - Add/remove/reorder visualization columns
    - Choose data source and column with searchable dropdowns
    - Choose visualization type (line/bar)
    - Select color maps
    - Configure axis limits (auto or manual)
    """
    
    def __init__(
        self,
        parent: tk.Widget,
        gui_manager: Any,
        config_manager: Any,
        color_map_manager: Any,
        translator: Optional[Any] = None,
        callback: Optional[Callable] = None,
        columns_by_source: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize dialog.
        
        Args:
            parent: Parent widget
            gui_manager: GUIManager for theming
            config_manager: ConfigManager for settings
            color_map_manager: ColorMapManager for color maps
            translator: Translator for i18n
            callback: Callback function when settings are saved
        """
        super().__init__(parent)
        
        self.gui_manager = gui_manager
        self.config_manager = config_manager
        self.color_map_manager = color_map_manager
        self.translator = translator or self._dummy_translator()
        self.callback = callback
        
        self.theme_colors = gui_manager.theme_colors
        self.fonts = gui_manager.fonts
        
        # Store columns by source for dropdowns
        self.columns_by_source = columns_by_source or {}
        
        # Determine default source (prefer "exassay" variants)
        self.default_source = self._get_default_source()
        logger.debug(f"Available sources: {list(self.columns_by_source.keys())}")
        logger.debug(f"Default source: {self.default_source}")
        
        # Load current settings
        self.viz_columns = config_manager.get("correlation_viz_columns", []).copy()
        self.selected_row = None
        self.column_rows: List[CorrelationVizRow] = []
        
        # Configure dialog
        self.title(self.translator.translate("Configure Data Visualizations"))
        self.geometry("1100x600")
        self.configure(bg=self.theme_colors["background"])
        self.transient(parent)
        # Don't use grab_set() - it prevents ttk.Combobox/OptionMenu dropdowns from working
        # self.grab_set()
        
        # Build UI
        self._create_ui()
        
        # Center dialog
        self._center_dialog()
        
        logger.info("CorrelationVizSettingsDialog opened")
    
    def _dummy_translator(self):
        """Create dummy translator."""
        class DummyTranslator:
            def translate(self, text: str) -> str:
                return text
        return DummyTranslator()
    
    def _get_default_source(self) -> str:
        """Get default data source (prefer exassay variants)."""
        if not self.columns_by_source:
            return ""
        
        sources = list(self.columns_by_source.keys())
        
        # Prefer sources containing "assay" (case-insensitive)
        for source in sources:
            if "assay" in source.lower():
                return source
        
        # Otherwise return first source
        return sources[0] if sources else ""

    @staticmethod
    def _default_viz_type_for_source(source_name: str) -> str:
        """Choose a sensible initial viz type for a source."""
        if "geophysics" in (source_name or "").lower():
            return "line"
        return "bar"

    @staticmethod
    def _default_scale_mode_for_source(source_name: str) -> str:
        """Choose a sensible initial scale mode for a source."""
        if "geophysics" in (source_name or "").lower():
            return "per_hole_percentile"
        return "raw"
    
    def _center_dialog(self) -> None:
        """Center dialog on parent."""
        self.update_idletasks()
        x = self.master.winfo_x() + (self.master.winfo_width() - self.winfo_width()) // 2
        y = self.master.winfo_y() + (self.master.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")
    
    def _create_ui(self) -> None:
        """Create dialog UI."""
        # Header
        header = tk.Frame(self, bg=self.theme_colors["secondary_bg"], height=60)
        header.pack(fill="x", padx=2, pady=2)
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text=self.translator.translate("Manage Visualization Columns"),
            font=self.fonts["heading"],
            bg=self.theme_colors["secondary_bg"],
            fg=self.theme_colors["text"]
        ).pack(pady=5)
        
        tk.Label(
            header,
            text=self.translator.translate("Configure which data columns to display as line/bar graphs next to drillhole sections"),
            font=self.fonts["small"],
            bg=self.theme_colors["secondary_bg"],
            fg=self.theme_colors["text"]
        ).pack(pady=2)
        
        tk.Label(
            header,
            text=self.translator.translate("Click a row to select it for reordering. Use buttons below to add/remove columns or reset to defaults."),
            font=self.fonts["small"],
            bg=self.theme_colors["secondary_bg"],
            fg=self.theme_colors["subtext"]
        ).pack(pady=2)
        
        # Scrollable list area
        list_frame = tk.Frame(self, bg=self.theme_colors["field_bg"])
        list_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Canvas with scrollbar
        canvas = tk.Canvas(
            list_frame,
            bg=self.theme_colors["field_bg"],
            highlightthickness=0
        )
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        
        self.rows_container = tk.Frame(canvas, bg=self.theme_colors["field_bg"])
        canvas.create_window(0, 0, anchor="nw", window=self.rows_container)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        self.rows_container.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        # Populate rows
        self._populate_rows()
        
        # Control buttons
        controls = tk.Frame(self, bg=self.theme_colors["background"])
        controls.pack(fill="x", padx=10, pady=10)
        
        # Left side buttons
        left_buttons = tk.Frame(controls, bg=self.theme_colors["background"])
        left_buttons.pack(side="left")
        
        ModernButton(
            left_buttons,
            text="↑ " + self.translator.translate("Move Up"),
            command=self._move_up,
            color=self.theme_colors["accent_blue"],
            theme_colors=self.theme_colors
        ).pack(side="left", padx=2)
        
        ModernButton(
            left_buttons,
            text="↓ " + self.translator.translate("Move Down"),
            command=self._move_down,
            color=self.theme_colors["accent_blue"],
            theme_colors=self.theme_colors
        ).pack(side="left", padx=2)
        
        ModernButton(
            left_buttons,
            text="+ " + self.translator.translate("Add Column"),
            command=self._add_column,
            color=self.theme_colors["accent_green"],
            theme_colors=self.theme_colors
        ).pack(side="left", padx=2)
        
        ModernButton(
            left_buttons,
            text="🎨 " + self.translator.translate("Manage Color Maps"),
            command=self._manage_color_maps,
            color=self.theme_colors["accent_blue"],
            theme_colors=self.theme_colors
        ).pack(side="left", padx=2)
        
        # Right side buttons
        right_buttons = tk.Frame(controls, bg=self.theme_colors["background"])
        right_buttons.pack(side="right")
        
        ModernButton(
            right_buttons,
            text=self.translator.translate("RESET TO DEFAULT"),
            command=self._reset_to_default,
            color=self.theme_colors["accent_yellow"],
            theme_colors=self.theme_colors
        ).pack(side="left", padx=2)
        
        # Display Settings section
        display_frame = tk.LabelFrame(
            self,
            text=self.translator.translate("Display Settings"),
            font=self.fonts["small"],
            bg=self.theme_colors["background"],
            fg=self.theme_colors["text"],
            padx=10,
            pady=5
        )
        display_frame.pack(fill="x", padx=10, pady=(5, 10))
        
        # Settings row
        settings_row = tk.Frame(display_frame, bg=self.theme_colors["background"])
        settings_row.pack(fill="x", pady=5)
        
        # Image Width setting
        tk.Label(
            settings_row,
            text=self.translator.translate("Image Width (px):"),
            font=self.fonts["small"],
            bg=self.theme_colors["background"],
            fg=self.theme_colors["text"]
        ).pack(side="left", padx=(0, 5))
        
        self.thumbnail_width_var = tk.IntVar(
            value=self.config_manager.get("correlation_thumbnail_width", 160)
        )
        thumbnail_spinbox = tk.Spinbox(
            settings_row,
            from_=60,
            to=420,
            increment=10,
            width=6,
            textvariable=self.thumbnail_width_var,
            font=self.fonts["small"],
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["text"],
            buttonbackground=self.theme_colors["secondary_bg"]
        )
        thumbnail_spinbox.pack(side="left", padx=(0, 20))
        
        # Cell Height setting
        tk.Label(
            settings_row,
            text=self.translator.translate("Cell Height (px):"),
            font=self.fonts["small"],
            bg=self.theme_colors["background"],
            fg=self.theme_colors["text"]
        ).pack(side="left", padx=(0, 5))
        
        self.cell_height_var = tk.IntVar(
            value=self.config_manager.get("correlation_data_viz_cell_height", 10)
        )
        cell_spinbox = tk.Spinbox(
            settings_row,
            from_=8,
            to=50,
            increment=2,
            width=6,
            textvariable=self.cell_height_var,
            font=self.fonts["small"],
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["text"],
            buttonbackground=self.theme_colors["secondary_bg"]
        )
        cell_spinbox.pack(side="left", padx=(0, 20))
        
        # Data Column Width setting
        tk.Label(
            settings_row,
            text=self.translator.translate("Data Column Width (px):"),
            font=self.fonts["small"],
            bg=self.theme_colors["background"],
            fg=self.theme_colors["text"]
        ).pack(side="left", padx=(0, 5))
        
        self.data_column_width_var = tk.IntVar(
            value=self.config_manager.get("correlation_data_column_width", 50)
        )
        data_col_spinbox = tk.Spinbox(
            settings_row,
            from_=20,
            to=260,
            increment=10,
            width=6,
            textvariable=self.data_column_width_var,
            font=self.fonts["small"],
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["text"],
            buttonbackground=self.theme_colors["secondary_bg"]
        )
        data_col_spinbox.pack(side="left")
        
        # Bottom buttons
        bottom_buttons = tk.Frame(self, bg=self.theme_colors["background"])
        bottom_buttons.pack(fill="x", padx=10, pady=10)
        
        ModernButton(
            bottom_buttons,
            text=self.translator.translate("Cancel"),
            command=self._on_cancel,
            color=self.theme_colors["text"],
            theme_colors=self.theme_colors
        ).pack(side="right", padx=5)
        
        ModernButton(
            bottom_buttons,
            text=self.translator.translate("Done"),
            command=self._on_done,
            color=self.theme_colors["accent_green"],
            theme_colors=self.theme_colors
        ).pack(side="right", padx=5)
    
    def _save_current_row_values(self) -> None:
        """Save current row values back to viz_columns before repopulating."""
        for i, row in enumerate(self.column_rows):
            if i < len(self.viz_columns):
                self.viz_columns[i] = row.get_config()
    
    def _populate_rows(self) -> None:
        """Populate visualization column rows."""
        # Clear existing rows
        for row in self.column_rows:
            row.destroy()
        self.column_rows.clear()
        
        # Get available columns from data manager (would need to pass this in)
        # For now, allow any text input
        
        # Create row for each viz column
        for idx, viz_config in enumerate(self.viz_columns):
            row = CorrelationVizRow(
                parent=self.rows_container,
                gui_manager=self.gui_manager,
                color_map_manager=self.color_map_manager,
                viz_config=viz_config,
                index=idx,
                on_select=lambda i: self._select_row(i),
                on_delete=lambda i: self._delete_row(i),
                columns_by_source=self.columns_by_source,
                default_source=self.default_source
            )
            row.pack(fill="x", padx=5, pady=2)
            self.column_rows.append(row)
    
    def _select_row(self, index: int) -> None:
        """Select a row."""
        self.selected_row = index
        
        # Update visual selection
        for i, row in enumerate(self.column_rows):
            row.set_selected(i == index)
    
    def _delete_row(self, index: int) -> None:
        """Delete a row."""
        # Save current row values before modifying
        self._save_current_row_values()
        
        if 0 <= index < len(self.viz_columns):
            del self.viz_columns[index]
            self._populate_rows()
    
    def _move_up(self) -> None:
        """Move selected row up."""
        if self.selected_row is None or self.selected_row == 0:
            return
        
        # Save current row values before modifying
        self._save_current_row_values()
        
        # Swap with previous
        idx = self.selected_row
        self.viz_columns[idx], self.viz_columns[idx - 1] = self.viz_columns[idx - 1], self.viz_columns[idx]
        self.selected_row = idx - 1
        
        self._populate_rows()
        self._select_row(self.selected_row)
    
    def _move_down(self) -> None:
        """Move selected row down."""
        if self.selected_row is None or self.selected_row >= len(self.viz_columns) - 1:
            return
        
        # Save current row values before modifying
        self._save_current_row_values()
        
        # Swap with next
        idx = self.selected_row
        self.viz_columns[idx], self.viz_columns[idx + 1] = self.viz_columns[idx + 1], self.viz_columns[idx]
        self.selected_row = idx + 1
        
        self._populate_rows()
        self._select_row(self.selected_row)
    
    def _add_column(self) -> None:
        """Add a new column."""
        # Save current row values before modifying
        self._save_current_row_values()
        
        new_config = {
            "source": self.default_source,
            "column": "",
            "type": self._default_viz_type_for_source(self.default_source),
            "scale_mode": self._default_scale_mode_for_source(self.default_source),
            "color_map": "",
            "min_value": None,
            "max_value": None,
            "auto_scale": True
        }
        self.viz_columns.append(new_config)
        self._populate_rows()
    
    def _manage_color_maps(self) -> None:
        """Open color map manager (would need to implement)."""
        messagebox.showinfo(
            "Color Maps",
            "Color map management coming soon!"
        )
    
    def _reset_to_default(self) -> None:
        """Reset to default settings."""
        if messagebox.askyesno(
            "Confirm Reset",
            "Reset visualization columns to defaults?"
        ):
            self.viz_columns = [
                {
                    "source": self.default_source,
                    "column": "Fe_pct_BEST",
                    "color_map": "fe_grade",
                    "type": "bar",
                    "scale_mode": "raw",
                    "min_value": None,
                    "max_value": None,
                    "auto_scale": True
                },
                {
                    "source": self.default_source,
                    "column": "SiO2_pct_BEST",
                    "color_map": "sio2_grade",
                    "type": "bar",
                    "scale_mode": "raw",
                    "min_value": None,
                    "max_value": None,
                    "auto_scale": True
                },
                {
                    "source": self.default_source,
                    "column": "Al2O3_pct_BEST",
                    "color_map": "al2o3_grade",
                    "type": "bar",
                    "scale_mode": "raw",
                    "min_value": None,
                    "max_value": None,
                    "auto_scale": True
                }
            ]
            self._populate_rows()
    
    def _on_cancel(self) -> None:
        """Cancel and close."""
        self.destroy()
    
    def _on_done(self) -> None:
        """Save settings and close."""
        # Collect settings from rows
        updated_columns = []
        for row in self.column_rows:
            config = row.get_config()
            if config.get("column"):  # Only include if column name is set
                updated_columns.append(config)
        
        self._save_config_updates(
            {
                "correlation_viz_columns": updated_columns,
                "correlation_thumbnail_width": self.thumbnail_width_var.get(),
                "correlation_data_viz_cell_height": self.cell_height_var.get(),
                "correlation_data_column_width": self.data_column_width_var.get(),
            }
        )

        logger.info(f"Saved {len(updated_columns)} visualization columns")

        callback = self.callback
        parent = self.master
        self.destroy()

        # Redrawing all correlation columns can take noticeable time. Let the
        # settings dialog close first, then start the refresh from the parent
        # event loop so the window does not look frozen on Done.
        if callback:
            try:
                parent.after_idle(lambda: callback(updated_columns))
            except Exception:
                callback(updated_columns)

    def _save_config_updates(self, updates: Dict[str, Any]) -> None:
        """Persist several config values with one disk write when supported."""
        editable_keys = set(getattr(self.config_manager, "USER_SETTINGS_KEYS", []))
        can_batch = (
            editable_keys
            and all(key in editable_keys for key in updates)
            and hasattr(self.config_manager, "user_settings")
            and hasattr(self.config_manager, "config")
            and hasattr(self.config_manager, "_save_user_settings")
        )

        if can_batch:
            self.config_manager.user_settings.update(updates)
            self.config_manager.config.update(updates)
            self.config_manager._save_user_settings()
            return

        for key, value in updates.items():
            self.config_manager.set(key, value)

class CorrelationVizRow(tk.Frame):
    """Row for configuring a single visualization column with source selection."""
    
    def __init__(
        self,
        parent: tk.Widget,
        gui_manager: Any,
        color_map_manager: Any,
        viz_config: Dict[str, Any],
        index: int,
        on_select: Callable[[int], None],
        on_delete: Callable[[int], None],
        columns_by_source: Optional[Dict[str, List[str]]] = None,
        default_source: str = ""
    ):
        """Initialize row."""
        self.gui_manager = gui_manager
        self.color_map_manager = color_map_manager
        self.theme_colors = gui_manager.theme_colors
        self.fonts = gui_manager.fonts
        self.viz_config = viz_config
        self.index = index
        self.on_select = on_select
        self.on_delete = on_delete
        self.is_selected = False
        self.columns_by_source = columns_by_source or {}
        self.default_source = default_source
        
        super().__init__(
            parent,
            bg=self.theme_colors["field_bg"],
            highlightbackground=self.theme_colors["border"],
            highlightthickness=1
        )
        
        self._create_widgets()
    
    def _create_widgets(self) -> None:
        """Create row widgets with source and column dropdowns."""
        # Get available sources
        sources = list(self.columns_by_source.keys())
        stored_column, stored_source = self._split_column_source(
            self.viz_config.get("column", "")
        )
        
        # Determine initial source from config or default
        initial_source = self.viz_config.get("source") or stored_source or self.default_source
        if initial_source not in sources and sources:
            initial_source = sources[0]
        
        # Column name label
        tk.Label(
            self,
            text="Column:",
            font=self.fonts["small"],
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["text"]
        ).pack(side="left", padx=5)
        
        # Source dropdown (if multiple sources available)
        self.source_var = tk.StringVar(value=initial_source)
        
        if len(sources) > 1:
            # Create frame for source dropdown with border
            source_combo_frame = tk.Frame(
                self,
                bg=self.theme_colors["field_bg"],
                highlightbackground=self.theme_colors["field_border"],
                highlightthickness=1,
                bd=0,
            )
            source_combo_frame.pack(side="left", padx=5)
            
            source_dropdown = tk.OptionMenu(
                source_combo_frame, 
                self.source_var, 
                *sources,
                command=self._on_source_change
            )
            self.gui_manager.style_dropdown(source_dropdown, width=12)
            source_dropdown.pack()
        
        # Separator
        tk.Label(
            self,
            text="→",
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["border"]
        ).pack(side="left", padx=2)
        
        # Column dropdown (searchable)
        self.column_var = tk.StringVar(value=stored_column)
        
        # Get columns for current source
        current_columns = self.columns_by_source.get(initial_source, [])
        stored_column = self._canonical_column_for_source(stored_column, current_columns)
        
        # Create searchable dropdown for columns
        self.column_dropdown = self.gui_manager.create_searchable_optionmenu(
            parent=self,
            items=current_columns,
            variable=self.column_var,
            width=18,
            placeholder="Search columns..."
        )
        self.column_dropdown.pack(side="left", padx=5)
        
        # Separator
        tk.Label(
            self,
            text="—",
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["border"]
        ).pack(side="left", padx=5)
        
        # Type dropdown
        tk.Label(
            self,
            text="Type:",
            font=self.fonts["small"],
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["text"]
        ).pack(side="left", padx=5)
        
        # Create frame for type dropdown with border
        type_combo_frame = tk.Frame(
            self,
            bg=self.theme_colors["field_bg"],
            highlightbackground=self.theme_colors["field_border"],
            highlightthickness=1,
            bd=0,
        )
        type_combo_frame.pack(side="left", padx=5)
        
        initial_type = self.viz_config.get("type")
        if not initial_type:
            initial_type = CorrelationVizSettingsDialog._default_viz_type_for_source(initial_source)
        self.type_var = tk.StringVar(value=initial_type)
        type_options = ["bar", "line"]
        
        self.type_dropdown = tk.OptionMenu(
            type_combo_frame, self.type_var, *type_options
        )
        self.gui_manager.style_dropdown(self.type_dropdown, width=6)
        self.type_dropdown.pack()

        # Separator
        tk.Label(
            self,
            text="--",
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["border"]
        ).pack(side="left", padx=5)

        # Scale dropdown
        tk.Label(
            self,
            text="Scale:",
            font=self.fonts["small"],
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["text"]
        ).pack(side="left", padx=5)

        scale_combo_frame = tk.Frame(
            self,
            bg=self.theme_colors["field_bg"],
            highlightbackground=self.theme_colors["field_border"],
            highlightthickness=1,
            bd=0,
        )
        scale_combo_frame.pack(side="left", padx=5)

        initial_scale_mode = self.viz_config.get("scale_mode")
        if not initial_scale_mode:
            initial_scale_mode = CorrelationVizSettingsDialog._default_scale_mode_for_source(initial_source)
        self.scale_mode_var = tk.StringVar(value=scale_mode_display_name(initial_scale_mode))
        scale_options = [label for _key, label in SCALE_MODE_LABELS]
        self.scale_mode_dropdown = tk.OptionMenu(
            scale_combo_frame, self.scale_mode_var, *scale_options
        )
        self.gui_manager.style_dropdown(self.scale_mode_dropdown, width=12)
        self.scale_mode_dropdown.pack()
        
        # Separator
        tk.Label(
            self,
            text="—",
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["border"]
        ).pack(side="left", padx=5)
        
        # Color Map
        tk.Label(
            self,
            text="Color Map:",
            font=self.fonts["small"],
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["text"]
        ).pack(side="left", padx=5)
        
        # Create frame for color map dropdown with border
        color_map_combo_frame = tk.Frame(
            self,
            bg=self.theme_colors["field_bg"],
            highlightbackground=self.theme_colors["field_border"],
            highlightthickness=1,
            bd=0,
        )
        color_map_combo_frame.pack(side="left", padx=5)
        
        # Get available color maps
        color_maps = list(self.color_map_manager.presets.keys()) if self.color_map_manager else []
        if not color_maps:
            color_maps = ["default"]
        
        self.color_map_var = tk.StringVar(value=self.viz_config.get("color_map", color_maps[0] if color_maps else ""))
        
        self.color_map_dropdown = tk.OptionMenu(
            color_map_combo_frame, self.color_map_var, *color_maps
        )
        self.gui_manager.style_dropdown(self.color_map_dropdown, width=12)
        self.color_map_dropdown.pack()
        
        # Separator
        tk.Label(
            self,
            text="—",
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["border"]
        ).pack(side="left", padx=5)
        
        # Auto-scale checkbox
        self.auto_scale_var = tk.BooleanVar(value=self.viz_config.get("auto_scale", True))
        self.auto_scale_check = tk.Checkbutton(
            self,
            text="Auto",
            variable=self.auto_scale_var,
            command=self._toggle_auto_scale,
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["text"],
            selectcolor=self.theme_colors["secondary_bg"],
            activebackground=self.theme_colors["field_bg"],
            activeforeground=self.theme_colors["text"]
        )
        self.auto_scale_check.pack(side="left", padx=2)
        
        # Min value
        tk.Label(
            self,
            text="Min:",
            font=self.fonts["small"],
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["text"]
        ).pack(side="left", padx=2)
        
        self.min_entry = tk.Entry(
            self,
            font=self.fonts["small"],
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["text"],
            insertbackground=self.theme_colors["text"],
            relief=tk.FLAT,
            bd=1,
            highlightbackground=self.theme_colors["field_border"],
            highlightthickness=1,
            width=6
        )
        self.min_entry.insert(0, str(self.viz_config.get("min_value", "")))
        self.min_entry.pack(side="left", padx=2)
        
        # Max value
        tk.Label(
            self,
            text="Max:",
            font=self.fonts["small"],
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["text"]
        ).pack(side="left", padx=2)
        
        self.max_entry = tk.Entry(
            self,
            font=self.fonts["small"],
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["text"],
            insertbackground=self.theme_colors["text"],
            relief=tk.FLAT,
            bd=1,
            highlightbackground=self.theme_colors["field_border"],
            highlightthickness=1,
            width=6
        )
        self.max_entry.insert(0, str(self.viz_config.get("max_value", "")))
        self.max_entry.pack(side="left", padx=2)
        
        # Update entry states based on auto-scale
        self._toggle_auto_scale()
        
        # Delete button
        delete_btn = ModernButton(
            self,
            text="✖",
            command=lambda: self.on_delete(self.index),
            color=self.theme_colors["accent_red"],
            theme_colors=self.theme_colors,
        )
        delete_btn.pack(side="right", padx=5)
        
        # Bind click to select
        self.bind("<Button-1>", lambda e: self.on_select(self.index))
        for widget in self.winfo_children():
            if not isinstance(widget, tk.Entry) and not isinstance(widget, ttk.Combobox):
                widget.bind("<Button-1>", lambda e: self.on_select(self.index))
    
    def _on_source_change(self, *args) -> None:
        """Handle source selection change - update column dropdown."""
        new_source = self.source_var.get()
        new_columns = self.columns_by_source.get(new_source, [])
        
        logger.debug(f"Source changed to {new_source}, {len(new_columns)} columns available")
        
        # Update column dropdown items
        if hasattr(self.column_dropdown, 'set_items'):
            self.column_dropdown.set_items(new_columns)
        elif hasattr(self.column_dropdown, 'update_items'):
            self.column_dropdown.update_items(new_columns)
        
        # Clear current selection if not in new source
        current_col = self.column_var.get()
        canonical_col = self._canonical_column_for_source(current_col, new_columns)
        if canonical_col != current_col:
            self.column_var.set(canonical_col)
            current_col = canonical_col

        if current_col and current_col not in new_columns:
            self.column_var.set("")
            if hasattr(self.column_dropdown, 'clear'):
                self.column_dropdown.clear()

        if "geophysics" in (new_source or "").lower():
            self.type_var.set("line")
            self.scale_mode_var.set(scale_mode_display_name("per_hole_percentile"))

    @staticmethod
    def _split_column_source(column_spec: str) -> Tuple[str, Optional[str]]:
        """Split legacy 'column (source)' configs into separate fields."""
        if not column_spec:
            return "", None

        text = column_spec.strip()
        if text.endswith(")") and "(" in text:
            column_name, source_name = text.rsplit("(", 1)
            return column_name.strip(), source_name[:-1].strip() or None

        return text, None

    @staticmethod
    def _canonical_column_for_source(column_name: str, columns: List[str]) -> str:
        """Promote legacy bare assay names to the actual source column casing."""
        if not column_name or not columns:
            return column_name

        by_lower = {str(col).lower(): str(col) for col in columns}
        lower_name = column_name.lower()
        if lower_name in by_lower:
            return by_lower[lower_name]

        candidates = []
        if lower_name.endswith("_pct") and not lower_name.endswith("_pct_best"):
            candidates.append(f"{lower_name}_best")
        if lower_name.endswith("_ppm") and not lower_name.endswith("_ppm_best"):
            candidates.append(f"{lower_name}_best")

        for candidate in candidates:
            if candidate in by_lower:
                return by_lower[candidate]
        return column_name
    
    def _toggle_auto_scale(self) -> None:
        """Toggle min/max entry states based on auto-scale."""
        if self.auto_scale_var.get():
            self.min_entry.config(state="disabled")
            self.max_entry.config(state="disabled")
        else:
            self.min_entry.config(state="normal")
            self.max_entry.config(state="normal")
    
    def set_selected(self, selected: bool) -> None:
        """Set selection state."""
        self.is_selected = selected
        if selected:
            self.config(highlightbackground=self.theme_colors["accent_blue"], highlightthickness=2)
        else:
            self.config(highlightbackground=self.theme_colors["border"], highlightthickness=1)
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        config = {
            "source": self.source_var.get(),
            "column": self.column_var.get().strip(),
            "type": self.type_var.get(),
            "scale_mode": canonical_scale_mode(self.scale_mode_var.get()),
            "color_map": self.color_map_var.get(),
            "auto_scale": self.auto_scale_var.get()
        }
        
        # Parse min/max values
        if not config["auto_scale"]:
            try:
                min_text = self.min_entry.get().strip()
                config["min_value"] = float(min_text) if min_text else None
            except ValueError:
                config["min_value"] = None
            
            try:
                max_text = self.max_entry.get().strip()
                config["max_value"] = float(max_text) if max_text else None
            except ValueError:
                config["max_value"] = None
        else:
            config["min_value"] = None
            config["max_value"] = None
        
        return config
