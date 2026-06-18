# src/gui/ReviewDialog/tabs/base_review_tab.py

"""
Base class for Review Dialog tabs.
Provides common functionality and interface that all tabs must implement.
"""

import tkinter as tk
from tkinter import ttk
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
import logging
from gui.dialog_helper import DialogHelper
from gui.widgets.modern_button import ModernButton


class BaseReviewTab(ABC):
    """
    Base class for review tabs.

    Provides common structure and callbacks that concrete tabs can use.
    All tabs share the same Phase 1 managers and Phase 2 components.
    """

    def __init__(
        self,
        parent: tk.Widget,
        data_manager,
        filter_engine,
        state_manager,
        peer_review_manager,
        gui_manager,
        theme_colors: Dict[str, str],
        on_classification_changed: Optional[Callable] = None,
        on_selection_changed: Optional[Callable] = None,
    ):
        """
        Initialize base tab.

        Args:
            parent: Parent widget
            data_manager: ReviewDataManager instance
            filter_engine: ReviewFilterEngine instance
            state_manager: ReviewStateManager instance
            peer_review_manager: PeerReviewManager instance
            theme_colors: Theme color dictionary
            on_classification_changed: Callback when classification changes
            on_selection_changed: Callback when selection changes
        """
        self.parent = parent
        self.data_manager = data_manager
        self.filter_engine = filter_engine
        self.state_manager = state_manager
        self.peer_review_manager = peer_review_manager
        self.gui_manager = gui_manager
        self.theme_colors = theme_colors
        self._on_classification_changed = on_classification_changed
        self._on_selection_changed = on_selection_changed

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Container frame
        self.container = ttk.Frame(parent)

        # UI components (Phase 2 components will be created by subclasses)
        self.filter_panel = None
        self.grid_canvas = None
        self.classification_toolbar = None
        self.statistics_display = None
        self.display_controls = None

        # Current displayed images
        self.displayed_images = []

    @abstractmethod
    def create_tab_content(self) -> tk.Frame:
        """
        Create the tab's UI layout.

        Must be implemented by subclasses to define their specific layout.

        Returns:
            Container frame with tab content
        """
        pass

    @abstractmethod
    def load_images_to_display(self):
        """
        Load images based on tab-specific logic.

        Must be implemented by subclasses to define how they filter/load images.
        """
        pass

    def _check_data_ready(self) -> bool:
        """Check if data manager is ready before loading grid"""
        if not self.data_manager or not self.data_manager.is_loaded:
            print(f"DEBUG: {self.__class__.__name__} skipping load - data not ready")
            return False
        return True

    def get_container(self) -> tk.Frame:
        """Get the tab's container frame"""
        return self.container

    def apply_filters(self):
        """
        Apply filters and reload display.

        This is the common entry point called by filter panel.
        """
        self.load_images_to_display()
        self._update_statistics()

    def classify_selection(
        self, category: str, moisture: str = None, comment: str = None
    ):
        """
        Classify selected images.

        Args:
            category: Classification category
            moisture: Moisture status (optional)
            comment: Classification comment (optional)
        """
        print(f"DEBUG: classify_selection() called - category={category}")

        if not self.grid_canvas:
            return

        selected_indices = self.grid_canvas.get_selected_indices()
        print(f"DEBUG: classify_selection() - {len(selected_indices)} images selected")
        if not selected_indices:
            return

        # Convert indices to image objects
        selected = [
            self.displayed_images[idx]
            for idx in selected_indices
            if idx < len(self.displayed_images)
        ]
        if not selected:
            return

        # Create undo action BEFORE making changes
        if self.state_manager:
            undo_action = self.state_manager.create_action(
                "classify", self.displayed_images, selected_indices
            )

        # Update images and capture new states
        import os
        from datetime import datetime

        new_states = []
        for img in selected:
            img.classification = category
            if moisture:
                img.moisture_status = moisture
            if comment:
                img.comments = comment

            # Update metadata
            img.classified_by = os.getenv("USERNAME", "Unknown").lower()
            img.classified_date = datetime.now().isoformat()

            # Capture new state
            new_states.append(
                {
                    "classification": img.classification,
                    "comments": img.comments,
                    "classified_by": img.classified_by,
                    "classified_date": img.classified_date,
                }
            )

        # Push complete action to undo stack
        if self.state_manager:
            undo_action.new_states = new_states
            self.state_manager.push_action(undo_action)

        # Notify changes
        if self._on_classification_changed:
            self._on_classification_changed(selected)

        # Refresh display
        self.grid_canvas.refresh_images(selected)
        self._update_statistics()

    def _update_statistics(self):
        """Update statistics display"""
        if not self.statistics_display:
            return

        stats = self._calculate_statistics()
        self.statistics_display.update_statistics(stats)

    def _calculate_statistics(self) -> Dict[str, Any]:
        """
        Calculate statistics for displayed images.

        Returns:
            Statistics dictionary
        """
        total = len(self.displayed_images)
        classified = sum(
            1 for img in self.displayed_images if img.classification != "Unassigned"
        )
        unassigned = total - classified

        # Count by category
        categories = {}
        for img in self.displayed_images:
            cat = img.classification
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "total": total,
            "classified": classified,
            "unassigned": unassigned,
            "categories": categories,
        }

    def _on_selection_change(self, selected_images: List):
        """
        Handle selection change from grid canvas.

        Args:
            selected_images: List of selected images
        """
        if self._on_selection_changed:
            self._on_selection_changed(selected_images)

    # ========================================================================
    # DISPLAY CONTROL METHODS (used by DisplayControlsPanel)
    # ========================================================================

    def _rotate_images(self, angle: int):
        """Rotate all displayed images by angle degrees"""
        if self.grid_canvas:
            self.grid_canvas.rotate_images(angle)

    def _scale_images(self, factor: float):
        """Scale image display size by factor"""
        if self.grid_canvas:
            new_scale = self.grid_canvas.scale_factor * factor
            self.grid_canvas.set_scale(new_scale)

    def _toggle_data_viz(self):
        """Toggle data visualization column display"""
        if self.grid_canvas:
            self.grid_canvas.toggle_data_visualizations()
            self.logger.debug(
                f"Data visualizations: {'ON' if self.grid_canvas.show_data_visualizations else 'OFF'}"
            )

    def _configure_viz(self):
        """Open configuration dialog for visualization columns"""
        if not self.grid_canvas:
            return

        # Get parent dialog
        parent_dialog = self.parent
        while parent_dialog and not isinstance(parent_dialog, tk.Toplevel):
            parent_dialog = parent_dialog.master

        if not parent_dialog:
            parent_dialog = self.parent

        # Create configuration dialog
        dialog = DialogHelper.create_dialog(
            parent_dialog, title="Configure Data Visualizations", modal=True
        )

        ttk.Label(
            dialog, text="Select data columns to visualize:", font=("Arial", 10)
        ).pack(pady=10)

        # Get available columns from displayed images' CSV data
        available_columns = set()
        for img in self.displayed_images[:100]:  # Sample first 100 images
            if hasattr(img, "csv_data") and img.csv_data:
                for col in img.csv_data.keys():
                    if any(
                        x in col.lower()
                        for x in [
                            "pct",
                            "ppm",
                            "grade",
                            "biff",
                            "chhm",
                            "fe",
                            "sio2",
                            "al2o3",
                        ]
                    ):
                        available_columns.add(col)

        # Add default columns
        default_cols = [
            "Fe_pct_BEST",
            "SiO2_pct_BEST",
            "Al2O3_pct_BEST",
            "Logged_pct_CHHM",
            "BIFf_2",
        ]
        available_columns.update(default_cols)

        # Convert to sorted list
        available_columns = sorted(list(available_columns))

        # If no columns found, use defaults only
        if not available_columns:
            available_columns = default_cols

        # Create scrollable frame for checkboxes
        canvas_frame = ttk.Frame(dialog)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        canvas = tk.Canvas(canvas_frame, height=300)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas_window = canvas.create_window(
            (0, 0), window=scrollable_frame, anchor="nw"
        )

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Checkboxes for each column
        selected_vars = {}
        for col in available_columns[:30]:  # Limit to 30 columns for UI
            var = tk.BooleanVar(value=col in self.grid_canvas.viz_columns)
            selected_vars[col] = var
            ttk.Checkbutton(scrollable_frame, text=col, variable=var).pack(
                anchor="w", padx=20, pady=2
            )

        # Update scroll region
        scrollable_frame.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))

        def apply_selection():
            selected = [col for col, var in selected_vars.items() if var.get()]
            self.grid_canvas.set_visualization_columns(selected)
            dialog.destroy()
            # Update status if available
            if hasattr(self, "_update_status"):
                self._update_status(f"Visualizing {len(selected)} data columns")
            self.logger.info(
                f"Set {len(selected)} visualization columns: {selected[:4]}..."
            )

        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)

        ModernButton(
            button_frame,
            text="Apply",
            command=apply_selection,
            color=self.theme_colors.get("accent_green", "#4CAF50"),
            theme_colors=self.theme_colors,
        ).pack(side=tk.LEFT, padx=5)

        ModernButton(
            button_frame,
            text="Cancel",
            command=dialog.destroy,
            color=self.theme_colors.get("secondary_bg", "#2d2d2d"),
            theme_colors=self.theme_colors,
        ).pack(side=tk.LEFT)

    # ========================================================================
    # CLASSIFICATION METHODS (consistent across all tabs)
    # ========================================================================

    def _on_classify(self, category: str, moisture: str = None, comment: str = None):
        """Handle classification from toolbar - consistent across all tabs"""
        self.classify_selection(category, moisture, comment)

    def _on_drag_classify(self, selected_indices: List[int]):
        """
        Handle drag-to-classify - TOGGLES classification like QAQC.

        Args:
            selected_indices: List of image indices to classify
        """
        if not selected_indices or not self.classification_toolbar:
            return

        # Get active classification category from toolbar
        active_category = self.classification_toolbar.get_active_category()

        # Get images from indices
        images_to_process = [
            self.displayed_images[idx]
            for idx in selected_indices
            if idx < len(self.displayed_images)
        ]

        if not images_to_process:
            return

        # Check if ANY selected image already has this classification
        # If so, we REMOVE it from all. Otherwise, we ADD it to all.
        has_classification = any(
            img.classification == active_category for img in images_to_process
        )

        if has_classification:
            # TOGGLE OFF - Remove classification from all that have it
            for img in images_to_process:
                if img.classification == active_category:
                    img.classification = "Unassigned"
                    img.classified_by = ""
                    img.classified_date = ""
            action = "Removed"
        else:
            # TOGGLE ON - Apply classification
            self.classify_selection(active_category)
            action = "Applied"

        # Refresh display
        if self.grid_canvas:
            self.grid_canvas.refresh_images(images_to_process)
            self._update_statistics()

        self.logger.debug(
            f"{action} {active_category} to/from {len(images_to_process)} images"
        )

    def _on_cell_click(self, idx: int):
        """Handle cell click from grid canvas - base implementation does nothing"""
        # Subclasses can override if they need specific behavior
        pass
