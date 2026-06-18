# src/gui/ReviewDialog/tabs/peer_review_tab.py

"""
Peer review tab.
Shows ONLY images that have reviews (from any user).
Provides filtering for conflicts, consensus, and review comparison.
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


class PeerReviewTab(BaseReviewTab):
    """
    Peer review mode.

    Shows ONLY images that have been reviewed (by anyone).
    Never shows unreviewed images.
    Highlights conflicts and consensus between reviewers.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize peer review tab.

        Args:
            *args, **kwargs: Passed to BaseReviewTab
        """
        super().__init__(*args, **kwargs)

        # Peer review state
        self.peer_mode = (
            "all_reviewed"  # all_reviewed, conflicts, consensus, my_vs_others
        )

        # UI references
        self.mode_buttons = {}
        self.peer_info_label = None

        # Translation
        self.t = (
            self.gui_manager.t
            if self.gui_manager and hasattr(self.gui_manager, "t")
            else lambda x: x
        )

    def create_tab_content(self) -> tk.Frame:
        """Create peer review specific layout"""
        # Configure grid
        self.container.grid_rowconfigure(2, weight=1)
        self.container.grid_columnconfigure(1, weight=1)

        # Top: Peer review controls
        peer_frame = self._create_peer_panel(self.container)
        peer_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

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

        # Grid canvas with peer review overlay support
        grid_frame = ttk.Frame(right_frame)
        grid_frame.grid(row=1, column=0, sticky="nsew")

        self.grid_canvas = ReviewGridCanvas(
            parent=grid_frame,
            theme_colors=self.theme_colors,
            on_selection_change=self._on_selection_change,
            on_cell_click=lambda idx: self._on_cell_click(idx),
            show_peer_info=True,  # Enable peer review overlay
        )

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

    def _create_peer_panel(self, parent) -> tk.Frame:
        """Create peer review mode selection panel"""
        from gui.widgets.modern_button import ModernButton

        frame = ttk.Frame(parent)

        # Label
        tk.Label(
            frame,
            text=self.t("Review Filter:"),
            bg=self.theme_colors.get("secondary_bg", "#2e2e2e"),
            fg=self.theme_colors.get("text", "#ffffff"),
            font=("Arial", 10, "bold"),
        ).pack(side=tk.LEFT, padx=(5, 10))

        # Mode buttons - ONLY show reviewed images
        modes = [
            ("all_reviewed", self.t("All Reviewed")),
            ("conflicts", self.t("Conflicts")),
            ("consensus", self.t("Consensus")),
            ("my_vs_others", self.t("My vs Others")),
            ("multi_reviewed", self.t("2+ Reviewers")),
        ]

        for mode_key, mode_label in modes:
            btn = ModernButton(
                frame,
                text=mode_label,
                command=lambda m=mode_key: self._set_peer_mode(m),
                color=(
                    self.theme_colors.get("accent_blue", "#2196F3")
                    if mode_key == "all_reviewed"
                    else self.theme_colors.get("secondary_bg", "#3e3e3e")
                ),
                theme_colors=self.theme_colors,
            )
            btn.pack(side=tk.LEFT, padx=2)
            self.mode_buttons[mode_key] = btn

        # Peer info label
        self.peer_info_label = tk.Label(
            frame,
            text="",
            bg=self.theme_colors.get("secondary_bg", "#2e2e2e"),
            fg=self.theme_colors.get("subtext", "#aaaaaa"),
            font=("Arial", 9),
        )
        self.peer_info_label.pack(side=tk.RIGHT, padx=10)

        return frame

    def load_images_to_display(self):
        """Load ONLY images that have reviews"""
        print(f"DEBUG: PeerReviewTab.load_images_to_display() called")

        # Gate: Don't load if data not ready
        if not self._check_data_ready():
            return

        # Get all images
        all_images = self.data_manager.get_all_images()

        # Load peer review data
        if self.peer_review_manager:
            self.peer_review_manager.load_all_user_reviews(all_images)

        # CRITICAL: Filter to ONLY reviewed images
        # An image is "reviewed" if it has ANY review from ANY user
        reviewed_images = []
        for img in all_images:
            # Check if current user reviewed it
            has_my_review = img.classification and img.classification != "Unassigned"

            # Check if others reviewed it
            has_peer_reviews = False
            if self.peer_review_manager:
                peer_reviews = self.peer_review_manager.get_peer_reviews(img)
                has_peer_reviews = len(peer_reviews) > 0

            # Include ONLY if someone reviewed it
            if has_my_review or has_peer_reviews:
                reviewed_images.append(img)

        # Now apply peer mode filtering on reviewed images only
        if self.peer_mode == "conflicts":
            # Only images where reviews disagree
            filtered = [
                img
                for img in reviewed_images
                if self.peer_review_manager
                and self.peer_review_manager.has_review_conflicts(img)
            ]
        elif self.peer_mode == "consensus":
            # Only images where all reviews agree
            filtered = [img for img in reviewed_images if self._has_consensus(img)]
        elif self.peer_mode == "my_vs_others":
            # Only images I reviewed that others also reviewed
            filtered = [
                img
                for img in reviewed_images
                if (
                    img.classification
                    and img.classification != "Unassigned"
                    and self.peer_review_manager
                    and len(self.peer_review_manager.get_peer_reviews(img)) > 0
                )
            ]
        elif self.peer_mode == "multi_reviewed":
            # Only images with 2+ total reviewers
            filtered = [
                img for img in reviewed_images if self._get_reviewer_count(img) >= 2
            ]
        else:  # all_reviewed
            filtered = reviewed_images

        # Apply classification filter (on the already filtered reviewed images)
        classification_mode = (
            self.filter_panel.get_classification_mode() if self.filter_panel else "all"
        )
        filtered = self.filter_engine.apply_classification_filter(
            filtered, classification_mode
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

        # Update peer info
        self._update_peer_info()

        self.logger.debug(
            f"Loaded {len(filtered)} reviewed images (mode: {self.peer_mode})"
        )

    def _set_peer_mode(self, mode: str):
        """Set peer review display mode"""
        self.peer_mode = mode

        # Update button colors
        for btn_mode, btn in self.mode_buttons.items():
            if btn_mode == mode:
                btn.update_color(self.theme_colors.get("accent_blue", "#2196F3"))
            else:
                btn.update_color(self.theme_colors.get("secondary_bg", "#3e3e3e"))

        # Reload display
        self.load_images_to_display()
        self._update_statistics()

    def _update_peer_info(self):
        """Update peer review information label"""
        if not self.peer_info_label:
            return

        total = len(self.displayed_images)

        # Count conflicts and consensus
        conflicts = 0
        consensus = 0
        multi_reviewed = 0

        for img in self.displayed_images:
            reviewer_count = self._get_reviewer_count(img)
            if reviewer_count >= 2:
                multi_reviewed += 1
                if (
                    self.peer_review_manager
                    and self.peer_review_manager.has_review_conflicts(img)
                ):
                    conflicts += 1
                elif self._has_consensus(img):
                    consensus += 1

        # Get total reviewers if available
        total_reviewers = 0
        if self.peer_review_manager:
            total_reviewers = self.peer_review_manager.total_reviewers

        info_text = self.t(
            "{total} reviewed | {multi} multi-reviewed | {conflicts} conflicts | {consensus} consensus"
        ).format(
            total=total, multi=multi_reviewed, conflicts=conflicts, consensus=consensus
        )

        self.peer_info_label.configure(text=info_text)

    def _has_consensus(self, img) -> bool:
        """Check if all reviews agree for an image"""
        if not self.peer_review_manager:
            return False

        all_reviews = self.peer_review_manager.get_all_reviews_for_image(img)
        if len(all_reviews) < 2:
            return False

        # Get all classifications
        classifications = set()
        for review in all_reviews:
            if review.get("Classification"):
                classifications.add(review["Classification"])

        # Consensus if all have same classification
        return len(classifications) == 1

    def _get_reviewer_count(self, img) -> int:
        """Get total number of reviewers for an image"""
        count = 0

        # Count current user if reviewed
        if img.classification and img.classification != "Unassigned":
            count += 1

        # Count peer reviewers
        if self.peer_review_manager:
            peer_reviews = self.peer_review_manager.get_peer_reviews(img)
            # Get unique reviewers from peer reviews
            reviewers = set()
            for review in peer_reviews:
                reviewer = review.get("Reviewed_By")
                if reviewer:
                    reviewers.add(reviewer)
            count += len(reviewers)

        return count
