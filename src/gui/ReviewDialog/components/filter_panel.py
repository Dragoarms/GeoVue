# src\gui\ReviewDialog\components\filter_panel.py

"""
Filter Panel Component - Collapsible filter UI.

Provides a collapsible panel with dynamic filter rows and classification mode filtering.
Uses CollapsibleFrame and DynamicFilterRow widgets.
"""

import tkinter as tk
from tkinter import ttk
from typing import List, Dict, Callable, Optional
import logging


logger = logging.getLogger(__name__)


class FilterPanel:
    """
    Manages filter UI with collapsible panel and dynamic filter rows.

    Features:
    - Collapsible panel using CollapsibleFrame
    - Dynamic filter rows using DynamicFilterRow
    - Classification mode filtering (all/classified/unclassified)
    - Auto-collapse with active filter summary
    - Add/remove filter rows
    """

    def __init__(
        self,
        parent,
        gui_manager,
        data_manager,
        on_filters_changed_callback: Optional[Callable] = None,
        on_apply_filters: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Initialize filter panel.

        Args:
            parent: Parent widget
            gui_manager: GUIManager for theming
            data_manager: ReviewDataManager for column info
            on_filters_changed_callback: Called when filters change
        """
        self.parent = parent
        self.gui_manager = gui_manager
        self.data_manager = data_manager
        self.on_filters_changed_callback = on_filters_changed_callback
        # PEP-8: allow legacy/new kw name; prefer explicit mapping
        if self.on_filters_changed_callback is None and on_apply_filters is not None:
            self.on_filters_changed_callback = on_apply_filters
        # PEP-8: keep unused kwargs for future options; harmless stash
        self._extra_kwargs = kwargs
        self.logger = logging.getLogger(__name__)

        # Theme colors
        self.theme_colors = gui_manager.theme_colors

        # State
        self.filter_rows: List = []  # DynamicFilterRow instances
        self.classification_mode = tk.StringVar(value="all")
        self.auto_hide_var = tk.BooleanVar(value=False)

        # Widgets
        self.collapsible_frame = None
        self.filter_container = None
        self.filter_canvas = None

    def create_panel(self) -> tk.Frame:
        """
        Create the filter panel UI.

        Returns:
            Frame containing the filter panel
        """
        # Import widgets
        from gui.widgets.collapsible_frame import CollapsibleFrame
        from gui.widgets.modern_button import ModernButton

        # Create collapsible frame
        self.collapsible_frame = CollapsibleFrame(
            self.parent,
            text="Filters (0 active)",
            expanded=True,
            bg=self.theme_colors["background"],
            fg=self.theme_colors["text"],
            title_bg=self.theme_colors["secondary_bg"],
            title_fg=self.theme_colors["text"],
            content_bg=self.theme_colors["background"],
            border_color=self.theme_colors["field_border"],
            arrow_color=self.theme_colors["accent_blue"],
        )
        self.collapsible_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        # Content frame (inside collapsible)
        content = self.collapsible_frame.content_frame

        # === CLASSIFICATION MODE SECTION ===
        mode_frame = tk.Frame(content, bg=self.theme_colors["background"])
        mode_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(mode_frame, text="Show:", font=("Arial", 9, "bold")).pack(
            side=tk.LEFT, padx=(0, 5)
        )

        # Mode buttons
        self.mode_buttons = {}
        for mode, label in [
            ("all", "All"),
            ("unclassified", "Unclassified"),
            ("classified", "Classified"),
        ]:
            btn = ModernButton(
                mode_frame,
                text=label,
                command=lambda m=mode: self._set_classification_mode(m),
                color=(
                    self.theme_colors["accent_blue"]
                    if mode == "all"
                    else self.theme_colors["secondary_bg"]
                ),
                theme_colors=self.theme_colors,
            )
            btn.pack(side=tk.LEFT, padx=2)
            self.mode_buttons[mode] = btn

        # === FILTER CONTROLS ===
        controls_frame = tk.Frame(content, bg=self.theme_colors["background"])
        controls_frame.pack(fill=tk.X, pady=(0, 5))

        ModernButton(
            controls_frame,
            text="+ Add Filter",
            command=self._add_filter,
            color=self.theme_colors["accent_green"],
            theme_colors=self.theme_colors,
        ).pack(side=tk.LEFT, padx=2)

        ModernButton(
            controls_frame,
            text="Clear All",
            command=self._clear_filters,
            color=self.theme_colors["accent_blue"],
            theme_colors=self.theme_colors,
        ).pack(side=tk.LEFT, padx=2)

        ModernButton(
            controls_frame,
            text="Apply",
            command=self._apply_filters,
            color=self.theme_colors["accent_green"],
            theme_colors=self.theme_colors,
        ).pack(side=tk.LEFT, padx=2)

        # Auto-hide checkbox
        self.auto_hide_check = tk.Checkbutton(
            content,
            text="Auto-collapse when applied",
            variable=self.auto_hide_var,
            bg=self.theme_colors["background"],
            fg=self.theme_colors["text"],
            activebackground=self.theme_colors["background"],
            activeforeground=self.theme_colors["text"],
            selectcolor=self.theme_colors["accent_green"],
            font=("Arial", 9),
            cursor="hand2",
            bd=0,
            highlightthickness=0,
        )
        self.auto_hide_check.pack(fill=tk.X, pady=(5, 10))

        # === SCROLLABLE FILTER CONTAINER ===
        scroll_frame = tk.Frame(content, bg=self.theme_colors["background"])
        scroll_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas for scrolling
        self.filter_canvas = tk.Canvas(
            scroll_frame,
            bg=self.theme_colors["background"],
            highlightthickness=0,
            height=200,
        )
        scrollbar = ttk.Scrollbar(scroll_frame, command=self.filter_canvas.yview)
        self.filter_canvas.configure(yscrollcommand=scrollbar.set)

        self.filter_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Container frame inside canvas
        self.filter_container = tk.Frame(
            self.filter_canvas, bg=self.theme_colors["background"]
        )
        self.filter_canvas.create_window(
            0, 0, anchor="nw", window=self.filter_container
        )

        # Update scroll region when container changes
        self.filter_container.bind(
            "<Configure>",
            lambda e: self.filter_canvas.configure(
                scrollregion=self.filter_canvas.bbox("all")
            ),
        )

        return self.collapsible_frame

    def _set_classification_mode(self, mode: str):
        """Set classification filter mode"""
        self.classification_mode.set(mode)

        # Update button colors
        for btn_mode, btn in self.mode_buttons.items():
            if btn_mode == mode:
                btn.update_color(self.theme_colors["accent_blue"])
            else:
                btn.update_color(self.theme_colors["secondary_bg"])

        # Trigger callback
        if self.on_filters_changed_callback:
            self.on_filters_changed_callback()

    def _add_filter(self, default_column: str = None):
        """Add a new filter row"""
        from gui.widgets.dynamic_filter_row import DynamicFilterRow

        # Get columns info from data manager
        columns_info = self._get_columns_info()

        # Create register data for the filter row
        register_data = self._get_register_data()

        # Create filter row
        filter_row = DynamicFilterRow(
            parent=self.filter_container,
            gui_manager=self.gui_manager,
            columns_info=columns_info,
            register_data=register_data,
            on_remove_callback=self._remove_filter,
            index=len(self.filter_rows),
            on_column_selected_callback=self._load_column_data_for_filter,
            get_column_schema_callback=self._get_column_schema_for_filter,
        )

        self.filter_rows.append(filter_row)

        # Set default column if provided
        if default_column and default_column in columns_info:
            filter_row.column_var.set(default_column)
            filter_row._on_column_change()

        # Update scroll region
        self.filter_container.update_idletasks()
        self.filter_canvas.configure(scrollregion=self.filter_canvas.bbox("all"))

        self.logger.debug(f"Added filter row, total: {len(self.filter_rows)}")

    def _remove_filter(self, index: int):
        """Remove a filter row"""
        if index < len(self.filter_rows):
            filter_row = self.filter_rows[index]
            filter_row.destroy()
            self.filter_rows.pop(index)

            # Re-index remaining filters
            for i, row in enumerate(self.filter_rows):
                row.index = i

            self.logger.debug(
                f"Removed filter {index}, {len(self.filter_rows)} remaining"
            )

            # Update heading
            self._update_heading()

    def _clear_filters(self):
        """Clear all filter rows"""
        for filter_row in self.filter_rows:
            filter_row.destroy()

        self.filter_rows.clear()
        self.logger.debug("Cleared all filters")

        # Update heading
        self._update_heading()

        # Trigger callback
        if self.on_filters_changed_callback:
            self.on_filters_changed_callback()

    def _apply_filters(self):
        """Apply filters and notify caller"""
        # Update heading with active filters
        self._update_heading()

        # Trigger callback
        if self.on_filters_changed_callback:
            self.on_filters_changed_callback()

        # Auto-collapse if enabled
        if self.auto_hide_var.get() and self.collapsible_frame.expanded:
            self.collapsible_frame.toggle()

    def _update_heading(self):
        """Update collapsible frame heading with filter summary"""
        # Get active filter configs
        active_filters = [
            f for f in self.get_active_filters() if f.get("value")  # Has a value set
        ]

        count = len(active_filters)

        if count == 0:
            heading = "Filters (0 active)"
        else:
            # Build summary string (show first 2)
            summaries = []
            for f in active_filters[:2]:
                column = f.get("column", "")
                operator = f.get("operator", "")
                value = f.get("value", "")

                # Shorten operator symbols for display
                op_display = operator
                if operator == "equals":
                    op_display = "="
                elif operator == "not equals":
                    op_display = "≠"
                elif operator == "contains":
                    op_display = "∋"

                summary = f"{column}{op_display}{value}"
                summaries.append(summary)

            if count > 2:
                summaries.append("...")

            heading = f"Filters ({count} active): " + ", ".join(summaries)

        # Update collapsible frame text
        if self.collapsible_frame:
            self.collapsible_frame.set_text(heading)

    def get_active_filters(self) -> List[Dict]:
        """
        Get all active filter configurations.

        Returns:
            List of filter config dicts from DynamicFilterRow
        """
        filters = []
        for filter_row in self.filter_rows:
            config = filter_row.get_filter_config()
            # Only include if has a value or is null operator
            if config.get("value") or config.get("operator") in ["is null", "not null"]:
                filters.append(config)

        return filters

    def get_classification_mode(self) -> str:
        """Get current classification filter mode"""
        return self.classification_mode.get()

    def add_permanent_filter(self, column: str, value: str):
        """
        Add a permanent filter (e.g., HoleID filter in hole-by-hole mode).

        Args:
            column: Column name
            value: Filter value
        """
        self._add_filter(default_column=column)

        # Set the value
        if self.filter_rows:
            filter_row = self.filter_rows[-1]
            filter_row.value_var.set(value)

    def update_columns(self, new_columns_info: Dict, new_register_data):
        """
        Update available columns for all filter rows.

        Args:
            new_columns_info: New columns info dict
            new_register_data: New register DataFrame
        """
        for filter_row in self.filter_rows:
            filter_row.update_columns(new_columns_info, new_register_data)

    # ========================================================================
    # PRIVATE HELPERS
    # ========================================================================

    def _get_columns_info(self) -> Dict:
        """Get columns info from data manager"""
        # Build columns from standard fields + CSV data
        columns_info = {
            "HoleID": {"type": "text"},
            "From": {"type": "numeric"},
            "To": {"type": "numeric"},
            "Classification": {"type": "text"},
            "Moisture": {"type": "text"},
            "Comments": {"type": "text"},
            "Reviewed_By": {"type": "text"},
            "Review_Date": {"type": "text"},
        }

        # Add CSV columns from first image if available
        if self.data_manager.all_images:
            first_img = self.data_manager.all_images[0]
            for key in first_img.csv_data.keys():
                if key not in columns_info:
                    columns_info[key] = {"type": "auto"}  # Auto-detect type

        return columns_info

    def _get_register_data(self):
        """Get register data as DataFrame for DynamicFilterRow"""
        import pandas as pd

        # Build DataFrame from images
        rows = []
        for img in self.data_manager.all_images:
            row = {
                "HoleID": img.hole_id,
                "From": img.depth_from,
                "To": img.depth_to,
                "Classification": img.classification,
                "Moisture": img.moisture_status,
                "Comments": img.comments,
                "Reviewed_By": img.classified_by,
                "Review_Date": img.classified_date,
            }

            # Add CSV data
            row.update(img.csv_data)

            rows.append(row)

        if rows:
            return pd.DataFrame(rows)
        else:
            # Return empty DataFrame with columns
            return pd.DataFrame(columns=list(self._get_columns_info().keys()))
