# src\gui\ReviewDialog\components\classification_toolbar.py
"""
Classification Toolbar Component - Classification buttons and shortcuts.

Provides classification buttons with keyboard shortcuts and quick actions.
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, Callable, List, Optional
import logging


logger = logging.getLogger(__name__)


class ClassificationToolbar:
    """
    Manages classification toolbar with buttons and keyboard shortcuts.

    Features:
    - Configurable classification categories
    - Keyboard shortcuts (1-9 for categories, C for comment)
    - Undo/Redo buttons and shortcuts
    - Save button
    - Visual feedback on hover
    """

    # Default classification categories
    DEFAULT_CATEGORIES = ["BIFf", "BIFhm", "Other", "Not Confident", "Unassigned"]

    def __init__(
        self,
        parent,
        on_classify: Callable,
        theme_colors: Dict[str, str],
        gui_manager=None,
        categories: List[str] = None,
    ):
        """
        Initialize classification toolbar.

        Args:
            parent: Parent widget
            on_classify: Callback function for classification: func(classification: str)
            theme_colors: Theme color dictionary
            gui_manager: GUIManager for theming (optional)
            categories: List of classification categories (optional)
        """
        self.parent = parent
        self.gui_manager = gui_manager
        self.theme_colors = theme_colors
        self.on_classify_callback = on_classify
        self.categories = categories or self.DEFAULT_CATEGORIES
        self.logger = logging.getLogger(__name__)

        # Callbacks dict for backward compatibility
        self.callbacks = {"on_classify": on_classify}

        # Track active classification mode for drag-to-classify
        self.active_category = self.categories[0] if self.categories else "Unassigned"

        # Widgets
        self.toolbar_frame = None
        self.category_buttons = {}
        self.undo_button = None
        self.redo_button = None

    def create_toolbar(self) -> tk.Frame:
        """
        Create the classification toolbar.

        Returns:
            Frame containing the toolbar
        """
        from gui.widgets.modern_button import ModernButton

        # Main toolbar frame
        self.toolbar_frame = tk.Frame(self.parent, bg=self.theme_colors["background"])

        # === CLASSIFICATION SECTION ===
        classify_frame = ttk.LabelFrame(
            self.toolbar_frame, text="Classify Selected", padding=5
        )
        classify_frame.pack(side=tk.LEFT, padx=5)

        # Create button for each category
        for i, category in enumerate(self.categories, start=1):
            # Determine color based on category
            if "BIF" in category or "Confident" in category:
                color = self.theme_colors["accent_blue"]
            elif category == "Other":
                color = self.theme_colors["secondary_bg"]
            elif category == "Unassigned":
                color = self.theme_colors["accent_red"]
            else:
                color = self.theme_colors["accent_green"]

            btn = ModernButton(
                classify_frame,
                text=f"{i}. {category}" if i <= 9 else category,
                command=lambda c=category: self._on_classify(c),
                color=color,
                theme_colors=self.theme_colors,
            )
            btn.pack(side=tk.LEFT, padx=2)

            self.category_buttons[category] = btn

        # === ACTIONS SECTION ===
        actions_frame = ttk.LabelFrame(self.toolbar_frame, text="Actions", padding=5)
        actions_frame.pack(side=tk.LEFT, padx=5)

        # Comment button
        ModernButton(
            actions_frame,
            text="💬 Comment",
            command=self._on_comment,
            color=self.theme_colors["accent_blue"],
            theme_colors=self.theme_colors,
        ).pack(side=tk.LEFT, padx=2)

        # Clear classification button
        ModernButton(
            actions_frame,
            text="✖ Clear",
            command=lambda: self._on_classify("Unassigned"),
            color=self.theme_colors["accent_red"],
            theme_colors=self.theme_colors,
        ).pack(side=tk.LEFT, padx=2)

        # === UNDO/REDO SECTION ===
        undo_frame = ttk.LabelFrame(self.toolbar_frame, text="Undo/Redo", padding=5)
        undo_frame.pack(side=tk.LEFT, padx=5)

        self.undo_button = ModernButton(
            undo_frame,
            text="↶ Undo",
            command=self._on_undo,
            color=self.theme_colors["secondary_bg"],
            theme_colors=self.theme_colors,
        )
        self.undo_button.pack(side=tk.LEFT, padx=2)

        self.redo_button = ModernButton(
            undo_frame,
            text="↷ Redo",
            command=self._on_redo,
            color=self.theme_colors["secondary_bg"],
            theme_colors=self.theme_colors,
        )
        self.redo_button.pack(side=tk.LEFT, padx=2)

        # === SAVE SECTION ===
        save_frame = ttk.LabelFrame(self.toolbar_frame, text="Save", padding=5)
        save_frame.pack(side=tk.LEFT, padx=5)

        ModernButton(
            save_frame,
            text="💾 Save All",
            command=self._on_save,
            color=self.theme_colors["accent_green"],
            theme_colors=self.theme_colors,
        ).pack(side=tk.LEFT, padx=2)

        return self.toolbar_frame

    def bind_keyboard_shortcuts(self, root_widget):
        """
        Bind keyboard shortcuts to root widget.

        Args:
            root_widget: Widget to bind shortcuts to (usually dialog window)
        """
        # Classification shortcuts (1-9)
        for i, category in enumerate(self.categories[:9], start=1):
            root_widget.bind(str(i), lambda e, c=category: self._on_classify(c))

        # Comment (C key)
        root_widget.bind("c", lambda e: self._on_comment())
        root_widget.bind("C", lambda e: self._on_comment())

        # Undo/Redo
        root_widget.bind("<Control-z>", lambda e: self._on_undo())
        root_widget.bind("<Control-Z>", lambda e: self._on_undo())
        root_widget.bind("<Control-y>", lambda e: self._on_redo())
        root_widget.bind("<Control-Y>", lambda e: self._on_redo())

        # Save
        root_widget.bind("<Control-s>", lambda e: self._on_save())
        root_widget.bind("<Control-S>", lambda e: self._on_save())

        # Clear (Delete key)
        root_widget.bind("<Delete>", lambda e: self._on_classify("Unassigned"))

        self.logger.info("Keyboard shortcuts bound")

    def update_undo_redo_state(self, can_undo: bool, can_redo: bool):
        """
        Update undo/redo button states.

        Args:
            can_undo: Whether undo is available
            can_redo: Whether redo is available
        """
        if self.undo_button:
            if can_undo:
                self.undo_button.update_color(self.theme_colors["accent_blue"])
                self.undo_button.config(state=tk.NORMAL)
            else:
                self.undo_button.update_color(self.theme_colors["secondary_bg"])
                # Don't disable, just dim - let it still show tooltip

        if self.redo_button:
            if can_redo:
                self.redo_button.update_color(self.theme_colors["accent_blue"])
                self.redo_button.config(state=tk.NORMAL)
            else:
                self.redo_button.update_color(self.theme_colors["secondary_bg"])

    def set_categories(self, categories: List[str]):
        """
        Update classification categories.

        Args:
            categories: New list of categories
        """
        self.categories = categories

        # Recreate toolbar if it exists
        if self.toolbar_frame:
            # Store parent
            parent = self.toolbar_frame.master

            # Destroy old toolbar
            self.toolbar_frame.destroy()

            # Recreate
            self.parent = parent
            self.create_toolbar()

    # ========================================================================
    # CALLBACK WRAPPERS
    # ========================================================================

    def _on_classify(self, classification: str):
        """Handle classification button click"""
        # Track active category for drag-to-classify
        self.active_category = classification

        if "on_classify" in self.callbacks:
            self.callbacks["on_classify"](classification)
        else:
            self.logger.warning("No on_classify callback registered")

    def get_active_category(self) -> str:
        """Get the currently active classification category"""
        return self.active_category

    def _on_comment(self):
        """Handle comment button click"""
        if "on_comment" in self.callbacks:
            self.callbacks["on_comment"]()
        else:
            self.logger.warning("No on_comment callback registered")

    def _on_undo(self):
        """Handle undo button click"""
        if "on_undo" in self.callbacks:
            self.callbacks["on_undo"]()
        else:
            self.logger.warning("No on_undo callback registered")

    def _on_redo(self):
        """Handle redo button click"""
        if "on_redo" in self.callbacks:
            self.callbacks["on_redo"]()
        else:
            self.logger.warning("No on_redo callback registered")

    def _on_save(self):
        """Handle save button click"""
        if "on_save" in self.callbacks:
            self.callbacks["on_save"]()
        else:
            self.logger.warning("No on_save callback registered")
