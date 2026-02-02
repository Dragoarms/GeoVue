# src\gui\ReviewDialog\tabs\hole_by_hole_tab.py

"""
Hole-by-hole review tab.
Shows images for one hole at a time with navigation controls.
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


class HoleByHoleTab(BaseReviewTab):
    """
    Hole-by-hole review mode.

    Shows one hole at a time with prev/next navigation.
    Auto-advances when hole is complete (optional).
    """

    def __init__(self, *args, auto_advance: bool = True, **kwargs):
        """
        Initialize hole-by-hole tab.

        Args:
            auto_advance: Auto-advance to next hole when complete
            *args, **kwargs: Passed to BaseReviewTab
        """
        super().__init__(*args, **kwargs)

        self.current_hole_index = 0
        self.hole_ids: List[str] = []
        self.auto_advance_enabled = auto_advance

        # UI references
        self.hole_label = None
        self.prev_btn = None
        self.next_btn = None
        self.auto_advance_var = None

        # Translation
        self.t = (
            self.gui_manager.t
            if self.gui_manager and hasattr(self.gui_manager, "t")
            else lambda x: x
        )

    def create_tab_content(self) -> tk.Frame:
        """Create hole-by-hole specific layout"""
        # Configure grid
        self.container.grid_rowconfigure(2, weight=1)
        self.container.grid_columnconfigure(1, weight=1)

        # Top: Hole navigation
        nav_frame = self._create_hole_navigation(self.container)
        nav_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        # Left: Filter panel (narrow)
        filter_container = ttk.Frame(self.container, width=250)
        filter_container.grid(row=1, column=0, rowspan=2, sticky="ns", padx=(5, 2))
        filter_container.grid_propagate(False)

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

        # Unified toolbar - Classification + All Controls
        toolbar_frame = ttk.Frame(right_frame)
        toolbar_frame.grid(row=0, column=0, sticky="ew", pady=(0, 2))

        # Left: Classification toolbar
        self.classification_toolbar = ClassificationToolbar(
            parent=toolbar_frame,
            on_classify=self._on_classify_with_hook,
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

        # Wire up drag-to-classify callback with hook
        self.grid_canvas.set_drag_classify_callback(self._on_drag_classify_with_hook)

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

    def _create_hole_navigation(self, parent) -> tk.Frame:
        """Create hole navigation controls"""
        from gui.widgets.modern_button import ModernButton

        frame = ttk.Frame(parent)

        # Background
        if "secondary_bg" in self.theme_colors:
            frame.configure(style="Nav.TFrame")

        # Previous button - using ModernButton
        self.prev_btn = ModernButton(
            frame,
            text=self.t("◄◄ Prev Hole"),
            command=self._previous_hole,
            color=self.theme_colors.get("secondary_bg", "#3e3e3e"),
            theme_colors=self.theme_colors,
        )
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        self.prev_btn.set_state("disabled")  # Start disabled

        # Hole label (shows current hole)
        self.hole_label = tk.Label(
            frame,
            text=self.t("No holes loaded"),
            bg=self.theme_colors.get("secondary_bg", "#2e2e2e"),
            fg=self.theme_colors.get("text", "#ffffff"),
            font=("Arial", 11, "bold"),
            padx=20,
        )
        self.hole_label.pack(side=tk.LEFT, padx=10)

        # Next button - using ModernButton
        self.next_btn = ModernButton(
            frame,
            text=self.t("Next Hole ►►"),
            command=self._next_hole,
            color=self.theme_colors.get("secondary_bg", "#3e3e3e"),
            theme_colors=self.theme_colors,
        )
        self.next_btn.pack(side=tk.LEFT, padx=5)
        self.next_btn.set_state("disabled")  # Start disabled

        # Auto-advance checkbox - using custom checkbox
        self.auto_advance_var = tk.BooleanVar(value=self.auto_advance_enabled)
        auto_check = self.gui_manager.create_custom_checkbox(
            frame,
            text=self.t("Auto-advance when complete"),
            variable=self.auto_advance_var,
            command=self._toggle_auto_advance,
        )
        auto_check.pack(side=tk.LEFT, padx=20)

        return frame

    def load_images_to_display(self):
        """Load images for current hole only"""
        print(f"DEBUG: HoleByHoleTab.load_images_to_display() called")

        # Gate: Don't load if data not ready
        if not self._check_data_ready():
            return

        if not self.hole_ids:
            # Initialize hole list
            all_images = self.data_manager.get_all_images()
            self.hole_ids = sorted(set(img.hole_id for img in all_images))

            if self.hole_ids:
                self._update_hole_navigation()

        if not self.hole_ids:
            self.displayed_images = []
            if self.grid_canvas:
                self.grid_canvas.load_images([])
            return

        # Get current hole
        current_hole = self.hole_ids[self.current_hole_index]

        # Get images for this hole
        hole_images = self.data_manager.get_images_for_hole(current_hole)

        # Apply classification filter
        classification_mode = (
            self.filter_panel.get_classification_mode() if self.filter_panel else "all"
        )
        filtered = self.filter_engine.apply_classification_filter(
            hole_images, classification_mode
        )

        # Apply dynamic filters
        filter_configs = (
            self.filter_panel.get_active_filters() if self.filter_panel else []
        )
        filtered = self.filter_engine.apply_dynamic_filters(filtered, filter_configs)

        self.displayed_images = filtered

        # Update grid
        if self.grid_canvas:
            self.grid_canvas.load_images(filtered)

        self.logger.debug(f"Loaded {len(filtered)} images for hole {current_hole}")

    def _update_hole_navigation(self):
        """Update hole navigation UI state"""
        if not self.hole_ids:
            return

        current_hole = self.hole_ids[self.current_hole_index]
        total_holes = len(self.hole_ids)

        # Update label - using translation
        label_text = self.t("Hole {hole} ({current}/{total})").format(
            hole=current_hole, current=self.current_hole_index + 1, total=total_holes
        )
        self.hole_label.configure(text=label_text)

        # Update button states using ModernButton's set_state method
        self.prev_btn.set_state("normal" if self.current_hole_index > 0 else "disabled")
        self.next_btn.set_state(
            "normal" if self.current_hole_index < total_holes - 1 else "disabled"
        )

    def _previous_hole(self):
        """Navigate to previous hole"""
        if self.current_hole_index > 0:
            self.current_hole_index -= 1
            self._update_hole_navigation()
            self.load_images_to_display()
            self._update_statistics()

    def _next_hole(self):
        """Navigate to next hole"""
        if self.current_hole_index < len(self.hole_ids) - 1:
            self.current_hole_index += 1
            self._update_hole_navigation()
            self.load_images_to_display()
            self._update_statistics()

    def _toggle_auto_advance(self):
        """Toggle auto-advance setting"""
        self.auto_advance_enabled = self.auto_advance_var.get()
        self.logger.info(f"Auto-advance: {self.auto_advance_enabled}")

    def _check_hole_completion(self):
        """Check if current hole is complete and auto-advance"""
        if not self.displayed_images:
            return

        # Check if all displayed images are classified
        unclassified = sum(
            1 for img in self.displayed_images if img.classification == "Unassigned"
        )

        if unclassified == 0 and self.current_hole_index < len(self.hole_ids) - 1:
            # Hole complete, advance after short delay
            current_hole = self.hole_ids[self.current_hole_index]
            self.logger.info(f"Hole {current_hole} complete, auto-advancing")

            # Show notification (if statistics display has status method)
            if hasattr(self.statistics_display, "set_status"):
                self.statistics_display.set_status(
                    f"✓ Hole {current_hole} complete! Moving to next..."
                )

            # Advance after 1 second
            self.container.after(1000, self._next_hole)

    # ========================================================================
    # WRAPPERS TO ADD AUTO-ADVANCE HOOK
    # ========================================================================

    def _on_classify_with_hook(
        self, category: str, moisture: str = None, comment: str = None
    ):
        """Wrapper to add auto-advance check after classification"""
        self._on_classify(category, moisture, comment)
        if self.auto_advance_enabled:
            self._check_hole_completion()

    def _on_drag_classify_with_hook(self, selected_indices: List[int]):
        """Wrapper to add auto-advance check after drag classification"""
        self._on_drag_classify(selected_indices)
        if self.auto_advance_enabled:
            self._check_hole_completion()
