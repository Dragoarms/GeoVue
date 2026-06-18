# src\gui\ReviewDialog\components\statistics_display.py
"""
Statistics Display Component - Shows classification statistics.

Displays classification counts, percentages, and progress indicators.
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, List
import logging


logger = logging.getLogger(__name__)


class StatisticsDisplay:
    """
    Manages statistics display panel.

    Features:
    - Total/classified/unassigned counts
    - Classification breakdown
    - Progress bar
    - Percentage calculations
    - Color-coded display
    """

    def __init__(self, parent, theme_colors: Dict[str, str], gui_manager=None):
        """
        Initialize statistics display.

        Args:
            parent: Parent widget
            theme_colors: Theme color dictionary
            gui_manager: GUIManager for theming (optional)
        """
        self.parent = parent
        self.gui_manager = gui_manager
        self.theme_colors = theme_colors
        self.logger = logging.getLogger(__name__)

        # Widgets
        self.stats_frame = None
        self.stats_labels = {}
        self.progress_bar = None

    def create_display(self) -> tk.Frame:
        """
        Create the statistics display panel.

        Returns:
            Frame containing the statistics display
        """
        # Main stats frame
        self.stats_frame = tk.Frame(self.parent, bg=self.theme_colors["background"])

        # === SUMMARY SECTION ===
        summary_frame = tk.Frame(self.stats_frame, bg=self.theme_colors["background"])
        summary_frame.pack(side=tk.LEFT, padx=10)

        # Total images
        self._create_stat_label(summary_frame, "Total Images:", "total", row=0)

        # Classified
        self._create_stat_label(
            summary_frame,
            "Classified:",
            "classified",
            row=1,
            fg=self.theme_colors["accent_green"],
        )

        # Unassigned
        self._create_stat_label(
            summary_frame,
            "Unassigned:",
            "unassigned",
            row=2,
            fg=self.theme_colors["accent_red"],
        )

        # === PROGRESS BAR ===
        progress_frame = tk.Frame(self.stats_frame, bg=self.theme_colors["background"])
        progress_frame.pack(side=tk.LEFT, padx=20, fill=tk.X, expand=True)

        tk.Label(
            progress_frame,
            text="Progress:",
            bg=self.theme_colors["background"],
            fg=self.theme_colors["text"],
            font=("Arial", 9, "bold"),
        ).pack(anchor=tk.W)

        # Create custom progress bar
        self.progress_bar = self._create_progress_bar(progress_frame)
        self.progress_bar.pack(fill=tk.X, pady=5)

        # Percentage label
        self.stats_labels["percentage"] = tk.Label(
            progress_frame,
            text="0%",
            bg=self.theme_colors["background"],
            fg=self.theme_colors["text"],
            font=("Arial", 9),
        )
        self.stats_labels["percentage"].pack(anchor=tk.W)

        # === BREAKDOWN SECTION ===
        breakdown_frame = tk.Frame(self.stats_frame, bg=self.theme_colors["background"])
        breakdown_frame.pack(side=tk.LEFT, padx=10)

        tk.Label(
            breakdown_frame,
            text="Classification Breakdown:",
            bg=self.theme_colors["background"],
            fg=self.theme_colors["text"],
            font=("Arial", 9, "bold"),
        ).pack(anchor=tk.W)

        # Breakdown container (will be populated dynamically)
        self.breakdown_container = tk.Frame(
            breakdown_frame, bg=self.theme_colors["background"]
        )
        self.breakdown_container.pack(fill=tk.BOTH, expand=True)

        return self.stats_frame

    def update_statistics(self, stats: Dict):
        """
        Update statistics display.

        Args:
            stats: Statistics dict with keys:
                - total: Total image count
                - classified: Classified count
                - unassigned: Unassigned count
                - classifications: Dict of {category: count}
        """
        total = stats.get("total", 0)
        classified = stats.get("classified", 0)
        unassigned = stats.get("unassigned", 0)

        # Update summary labels
        if "total" in self.stats_labels:
            self.stats_labels["total"].config(text=str(total))

        if "classified" in self.stats_labels:
            self.stats_labels["classified"].config(text=str(classified))

        if "unassigned" in self.stats_labels:
            self.stats_labels["unassigned"].config(text=str(unassigned))

        # Update progress bar
        if total > 0:
            percentage = (classified / total) * 100
            self._update_progress_bar(percentage)

            if "percentage" in self.stats_labels:
                self.stats_labels["percentage"].config(
                    text=f"{percentage:.1f}% Complete"
                )
        else:
            self._update_progress_bar(0)
            if "percentage" in self.stats_labels:
                self.stats_labels["percentage"].config(text="0% Complete")

        # Update breakdown
        classifications = stats.get("classifications", {})
        self._update_breakdown(classifications)

    def _create_stat_label(
        self, parent, label_text: str, key: str, row: int, fg: str = None
    ):
        """Create a label/value pair for statistics"""
        frame = tk.Frame(parent, bg=self.theme_colors["background"])
        frame.grid(row=row, column=0, sticky="w", pady=2)

        # Label
        tk.Label(
            frame,
            text=label_text,
            bg=self.theme_colors["background"],
            fg=self.theme_colors["text"],
            font=("Arial", 9, "bold"),
            width=12,
            anchor="w",
        ).pack(side=tk.LEFT)

        # Value
        value_label = tk.Label(
            frame,
            text="0",
            bg=self.theme_colors["background"],
            fg=fg or self.theme_colors["text"],
            font=("Arial", 9),
            width=6,
            anchor="e",
        )
        value_label.pack(side=tk.LEFT)

        self.stats_labels[key] = value_label

    def _create_progress_bar(self, parent) -> tk.Canvas:
        """Create custom progress bar"""
        canvas = tk.Canvas(
            parent,
            width=200,
            height=20,
            bg=self.theme_colors["field_bg"],
            highlightthickness=1,
            highlightbackground=self.theme_colors["field_border"],
        )

        # Store canvas for updates
        self.progress_canvas = canvas
        self.progress_rect = None

        return canvas

    def _update_progress_bar(self, percentage: float):
        """Update progress bar fill"""
        if not hasattr(self, "progress_canvas"):
            return

        # Clear previous rectangle
        if self.progress_rect:
            self.progress_canvas.delete(self.progress_rect)

        # Calculate fill width
        canvas_width = self.progress_canvas.winfo_width() or 200
        fill_width = int((percentage / 100) * canvas_width)

        # Determine color based on percentage
        if percentage < 33:
            color = self.theme_colors["accent_red"]
        elif percentage < 66:
            color = self.theme_colors["accent_yellow"]
        else:
            color = self.theme_colors["accent_green"]

        # Draw filled rectangle
        if fill_width > 0:
            self.progress_rect = self.progress_canvas.create_rectangle(
                0, 0, fill_width, 20, fill=color, outline=""
            )

    def _update_breakdown(self, classifications: Dict[str, int]):
        """Update classification breakdown display"""
        # Clear existing breakdown
        for widget in self.breakdown_container.winfo_children():
            widget.destroy()

        if not classifications:
            tk.Label(
                self.breakdown_container,
                text="No classifications yet",
                bg=self.theme_colors["background"],
                fg=self.theme_colors["text"],
                font=("Arial", 9, "italic"),
            ).pack(anchor=tk.W)
            return

        # Sort by count (descending)
        sorted_items = sorted(classifications.items(), key=lambda x: x[1], reverse=True)

        # Display each classification
        for category, count in sorted_items:
            frame = tk.Frame(
                self.breakdown_container, bg=self.theme_colors["background"]
            )
            frame.pack(fill=tk.X, pady=1)

            # Color indicator
            color = self._get_category_color(category)
            indicator = tk.Label(
                frame,
                text="■",
                bg=self.theme_colors["background"],
                fg=color,
                font=("Arial", 12),
            )
            indicator.pack(side=tk.LEFT, padx=(0, 5))

            # Category name
            tk.Label(
                frame,
                text=category,
                bg=self.theme_colors["background"],
                fg=self.theme_colors["text"],
                font=("Arial", 9),
                width=15,
                anchor="w",
            ).pack(side=tk.LEFT)

            # Count
            tk.Label(
                frame,
                text=str(count),
                bg=self.theme_colors["background"],
                fg=self.theme_colors["text"],
                font=("Arial", 9, "bold"),
                width=4,
                anchor="e",
            ).pack(side=tk.LEFT)

    def _get_category_color(self, category: str) -> str:
        """Get color for a classification category"""
        category_lower = category.lower()

        if "biff" in category_lower:
            return self.theme_colors["accent_blue"]
        elif "bifhm" in category_lower:
            return self.theme_colors["accent_green"]
        elif "other" in category_lower:
            return self.theme_colors["text"]
        elif "confident" in category_lower:
            return self.theme_colors["accent_yellow"]
        elif "unassigned" in category_lower:
            return self.theme_colors["accent_red"]
        else:
            return self.theme_colors["accent_blue"]

    def get_summary_text(self, stats: Dict) -> str:
        """
        Get summary text for status bar.

        Args:
            stats: Statistics dict

        Returns:
            Formatted summary string
        """
        total = stats.get("total", 0)
        classified = stats.get("classified", 0)
        unassigned = stats.get("unassigned", 0)

        if total == 0:
            return "No images loaded"

        percentage = (classified / total) * 100 if total > 0 else 0

        return f"Total: {total} | Classified: {classified} | Unassigned: {unassigned} | {percentage:.1f}% Complete"
