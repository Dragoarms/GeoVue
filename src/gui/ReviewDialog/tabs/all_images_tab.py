# src/gui/ReviewDialog/tabs/all_images_tab.py

"""
All images review tab.
Shows all images at once with sorting and filtering capabilities.
"""

import tkinter as tk
from tkinter import ttk
from typing import List, Dict, Any, Optional, Callable
import logging

from gui.ReviewDialog.tabs.base_review_tab import BaseReviewTab
from gui.ReviewDialog.components.filter_panel import FilterPanel
from gui.ReviewDialog.components.classification_toolbar import ClassificationToolbar
from gui.ReviewDialog.components.review_grid_canvas import ReviewGridCanvas
from gui.ReviewDialog.components.statistics_display import StatisticsDisplay
from gui.ReviewDialog.components.display_controls_panel import DisplayControlsPanel


class AllImagesTab(BaseReviewTab):
    """
    All images review mode.

    Shows all images with sorting and advanced filtering.
    No hole navigation - displays everything at once.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize all images tab.

        Args:
            *args, **kwargs: Passed to BaseReviewTab
        """
        super().__init__(*args, **kwargs)

        # Sorting state
        self.sort_column = "hole_id"
        self.sort_direction = "asc"

        # UI references
        self.sort_dropdown = None
        self.sort_direction_var = None
        self.summary_label = None

        # Translation
        self.t = (
            self.gui_manager.t
            if self.gui_manager and hasattr(self.gui_manager, "t")
            else lambda x: x
        )

    def create_tab_content(self) -> tk.Frame:
        """Create all images specific layout"""
        # Configure grid
        self.container.grid_rowconfigure(2, weight=1)
        self.container.grid_columnconfigure(1, weight=1)

        # Top: Sorting controls
        sort_frame = self._create_sorting_controls(self.container)
        sort_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        # Left: Filter panel (narrow)
        filter_container = ttk.Frame(self.container, width=250)
        filter_container.grid(row=1, column=0, rowspan=2, sticky="ns", padx=(5, 2))
        filter_container.grid_propagate(False)

        # Create filter panel
        self.filter_panel = FilterPanel(
            parent=filter_container,
            gui_manager=self.gui_manager,
            data_manager=self.data_manager,
            on_apply_filters=self.apply_filters,
        )
        filter_frame = self.filter_panel.create_panel()
        filter_frame.pack(fill=tk.BOTH, expand=True)

        # Center-right: Grid and toolbar
        right_frame = ttk.Frame(self.container)
        right_frame.grid(row=1, column=1, rowspan=2, sticky="nsew", padx=(2, 5))
        right_frame.grid_rowconfigure(1, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        # Classification toolbar
        toolbar_frame = ttk.Frame(right_frame)
        toolbar_frame.grid(row=0, column=0, sticky="ew", pady=(0, 2))

        self.classification_toolbar = ClassificationToolbar(
            parent=toolbar_frame,
            on_classify=self._on_classify,
            theme_colors=self.theme_colors,
            gui_manager=self.gui_manager,
        )
        toolbar_widget = self.classification_toolbar.create_toolbar()
        toolbar_widget.pack(side=tk.LEFT, fill=tk.Y)

        # Display controls panel
        self.display_controls = DisplayControlsPanel(
            parent=toolbar_frame,
            theme_colors=self.theme_colors,
            on_rotate=self._rotate_images,
            on_scale=self._scale_images,
            on_toggle_viz=self._toggle_data_viz,
            on_configure_viz=(
                self._configure_viz if hasattr(self, "_configure_viz") else None
            ),
            gui_manager=self.gui_manager,
        )
        controls_panel = self.display_controls.create_panel()
        controls_panel.pack(side=tk.RIGHT, padx=5)

        # Grid canvas
        grid_frame = ttk.Frame(right_frame)
        grid_frame.grid(row=1, column=0, sticky="nsew")

        self.grid_canvas = ReviewGridCanvas(
            parent=grid_frame,
            theme_colors=self.theme_colors,
            on_selection_change=self._on_selection_change,
            on_cell_click=lambda idx: self._on_cell_click(idx),
        )
        # Canvas packs itself in __init__

        # Wire up drag-to-classify callback
        self.grid_canvas.set_drag_classify_callback(self._on_drag_classify)

        # Bottom: Statistics
        stats_frame = ttk.Frame(self.container)
        stats_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        self.statistics_display = StatisticsDisplay(
            parent=stats_frame,
            theme_colors=self.theme_colors,
            gui_manager=self.gui_manager,
        )
        stats_widget = self.statistics_display.create_display()
        stats_widget.pack(fill=tk.X)

        return self.container

    def _create_sorting_controls(self, parent) -> tk.Frame:
        """Create sorting controls"""
        from gui.widgets.modern_button import ModernButton

        frame = ttk.Frame(parent)

        # Background
        if "secondary_bg" in self.theme_colors:
            frame.configure(style="Sort.TFrame")

        # Label
        tk.Label(
            frame,
            text=self.t("Sort by:"),
            bg=self.theme_colors.get("secondary_bg", "#2e2e2e"),
            fg=self.theme_colors.get("text", "#ffffff"),
            font=("Arial", 10),
        ).pack(side=tk.LEFT, padx=(5, 2))

        # Sort column dropdown
        self.sort_column_var = tk.StringVar(value=self.sort_column)

        sort_options = [
            "hole_id",
            "depth_to",
            "depth_from",
            "classification",
            "moisture_status",
        ]

        self.sort_dropdown = ttk.OptionMenu(
            frame,
            self.sort_column_var,
            self.sort_column,
            *sort_options,
            command=self._on_sort_changed,
        )
        self.sort_dropdown.pack(side=tk.LEFT, padx=5)

        # Direction radio buttons
        self.sort_direction_var = tk.StringVar(value=self.sort_direction)

        asc_radio = ttk.Radiobutton(
            frame,
            text=self.t("Ascending ↑"),
            variable=self.sort_direction_var,
            value="asc",
            command=self._on_sort_changed,
        )
        asc_radio.pack(side=tk.LEFT, padx=5)

        desc_radio = ttk.Radiobutton(
            frame,
            text=self.t("Descending ↓"),
            variable=self.sort_direction_var,
            value="desc",
            command=self._on_sort_changed,
        )
        desc_radio.pack(side=tk.LEFT, padx=5)

        # Apply sort button - using ModernButton
        apply_btn = ModernButton(
            frame,
            text=self.t("Apply Sort"),
            command=self._apply_sort,
            color=self.theme_colors.get("accent_blue", "#2196F3"),
            theme_colors=self.theme_colors,
        )
        apply_btn.pack(side=tk.LEFT, padx=10)

        # Summary label
        self.summary_label = tk.Label(
            frame,
            text="",
            bg=self.theme_colors.get("secondary_bg", "#2e2e2e"),
            fg=self.theme_colors.get("subtext", "#aaaaaa"),
            font=("Arial", 9),
        )
        self.summary_label.pack(side=tk.RIGHT, padx=10)

        return frame

    def load_images_to_display(self):
        """Load all images with filtering and sorting"""
        print(f"DEBUG: AllImagesTab.load_images_to_display() called")

        # Gate: Don't load if data not ready
        if not self._check_data_ready():
            return

        # Get all images
        all_images = self.data_manager.get_all_images()

        # Apply classification filter
        classification_mode = (
            self.filter_panel.get_classification_mode() if self.filter_panel else "all"
        )
        filtered = self.filter_engine.apply_classification_filter(
            all_images, classification_mode
        )

        # Apply dynamic filters
        filter_configs = (
            self.filter_panel.get_active_filters() if self.filter_panel else []
        )
        filtered = self.filter_engine.apply_dynamic_filters(filtered, filter_configs)

        # Apply sorting
        sorted_images = self._sort_images(filtered)

        self.displayed_images = sorted_images

        # Update grid
        if self.grid_canvas:
            self.grid_canvas.load_images(sorted_images)

        # Update summary
        self._update_summary()

        self.logger.debug(
            f"Loaded {len(sorted_images)} images (sorted by {self.sort_column})"
        )

    def _sort_images(self, images: List) -> List:
        """
        Sort images based on current sort settings.

        Args:
            images: List of images to sort

        Returns:
            Sorted list of images
        """
        if not images:
            return images

        # Get sort key function
        def get_sort_key(img):
            if self.sort_column == "hole_id":
                return img.hole_id
            elif self.sort_column == "depth_to":
                return (img.hole_id, img.depth_to)
            elif self.sort_column == "depth_from":
                return (img.hole_id, img.depth_from)
            elif self.sort_column == "classification":
                return img.classification or "ZZZZZ"  # Put unassigned at end
            elif self.sort_column == "moisture_status":
                return img.moisture_status or "ZZZ"
            else:
                # Try CSV data
                return img.csv_data.get(self.sort_column, "")

        # Sort
        reverse = self.sort_direction == "desc"
        return sorted(images, key=get_sort_key, reverse=reverse)

    def _on_sort_changed(self, *args):
        """Handle sort dropdown or radio change"""
        # Update state from UI
        self.sort_column = self.sort_column_var.get()
        self.sort_direction = self.sort_direction_var.get()

    def _apply_sort(self):
        """Apply current sort settings"""
        self.load_images_to_display()
        self._update_statistics()

    def _update_summary(self):
        """Update summary label"""
        if not self.summary_label:
            return

        total = len(self.displayed_images)

        if total == 0:
            self.summary_label.configure(text=self.t("No images to display"))
        else:
            # Count unique holes
            unique_holes = len(set(img.hole_id for img in self.displayed_images))
            summary_text = self.t("Showing {total} images from {holes} hole(s)").format(
                total=f"{total:,}", holes=unique_holes
            )
            self.summary_label.configure(text=summary_text)

    def update_sort_options(self, csv_columns: List[str]):
        """
        Update sort dropdown with CSV columns.

        Args:
            csv_columns: List of CSV column names to add to sort options
        """
        if not self.sort_dropdown:
            return

        # Combine base options with CSV columns
        base_options = [
            "hole_id",
            "depth_to",
            "depth_from",
            "classification",
            "moisture_status",
        ]

        all_options = base_options + csv_columns

        # Update dropdown menu
        menu = self.sort_dropdown["menu"]
        menu.delete(0, "end")

        for option in all_options:
            menu.add_command(
                label=option, command=lambda val=option: self.sort_column_var.set(val)
            )

        self.logger.debug(f"Updated sort options: {len(all_options)} total")
