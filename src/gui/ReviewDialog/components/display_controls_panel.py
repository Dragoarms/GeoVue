# src/gui/ReviewDialog/components/display_controls_panel.py

"""
Display Controls Panel - Unified controls for image display.

Provides consistent rotate/scale/data viz controls across all tabs.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DisplayControlsPanel:
    """
    Manages display control buttons (rotate/scale/data viz).

    Ensures consistent behavior across all tabs.
    """

    def __init__(
        self,
        parent: tk.Widget,
        theme_colors: Dict[str, str],
        on_rotate: Callable[[int], None],
        on_scale: Callable[[float], None],
        on_toggle_viz: Callable[[], None],
        on_configure_viz: Optional[Callable[[], None]] = None,
        gui_manager=None,
    ):
        """
        Initialize display controls.

        Args:
            parent: Parent widget
            theme_colors: Theme color dictionary
            on_rotate: Callback for rotation (angle: int)
            on_scale: Callback for scaling (factor: float)
            on_toggle_viz: Callback for toggling data viz
            on_configure_viz: Optional callback for configuring viz columns
            gui_manager: GUI manager for theming
        """
        self.parent = parent
        self.theme_colors = theme_colors
        self.gui_manager = gui_manager
        self.on_rotate = on_rotate
        self.on_scale = on_scale
        self.on_toggle_viz = on_toggle_viz
        self.on_configure_viz = on_configure_viz

        self.viz_enabled = False
        self.viz_toggle_btn = None
        self.viz_config_btn = None

    def create_panel(self) -> tk.Frame:
        """Create the display controls panel"""
        from gui.widgets.modern_button import ModernButton

        # Main container
        container = ttk.Frame(self.parent)

        # Rotate control (single button, always 90° clockwise like QAQC)
        ttk.Label(container, text="Rotate:").pack(side=tk.LEFT, padx=(0, 1))
        ModernButton(
            container,
            text="↻",
            command=lambda: self.on_rotate(90),
            color=self.theme_colors.get("secondary_bg", "#3e3e3e"),
            theme_colors=self.theme_colors,
            width=3,
        ).pack(side=tk.LEFT, padx=1)

        # Scale controls
        ttk.Label(container, text="Scale:").pack(side=tk.LEFT, padx=(5, 1))
        ModernButton(
            container,
            text="+",
            command=lambda: self.on_scale(1.2),
            color=self.theme_colors.get("secondary_bg", "#3e3e3e"),
            theme_colors=self.theme_colors,
            width=2,
        ).pack(side=tk.LEFT, padx=1)

        ModernButton(
            container,
            text="−",
            command=lambda: self.on_scale(0.8),
            color=self.theme_colors.get("secondary_bg", "#3e3e3e"),
            theme_colors=self.theme_colors,
            width=2,
        ).pack(side=tk.LEFT, padx=1)

        # Data viz toggle
        self.viz_toggle_btn = ModernButton(
            container,
            text="📊",
            command=self._toggle_viz,
            color=self.theme_colors.get("accent_blue", "#2196F3"),
            theme_colors=self.theme_colors,
            width=3,
        )
        self.viz_toggle_btn.pack(side=tk.LEFT, padx=(5, 1))

        # Configure viz button
        if self.on_configure_viz:
            self.viz_config_btn = ModernButton(
                container,
                text="⚙",
                command=self.on_configure_viz,
                color=self.theme_colors.get("secondary_bg", "#3e3e3e"),
                theme_colors=self.theme_colors,
                width=3,
            )
            self.viz_config_btn.pack(side=tk.LEFT, padx=1)

        return container

    def _toggle_viz(self):
        """Toggle data visualization and update button"""
        self.viz_enabled = not self.viz_enabled

        # Update button appearance
        if self.viz_enabled:
            self.viz_toggle_btn.update_color(
                self.theme_colors.get("accent_green", "#4CAF50")
            )
        else:
            self.viz_toggle_btn.update_color(
                self.theme_colors.get("accent_blue", "#2196F3")
            )

        # Call the callback
        self.on_toggle_viz()

        logger.debug(f"Data viz toggled: {self.viz_enabled}")

    def set_viz_state(self, enabled: bool):
        """Set visualization state programmatically"""
        if self.viz_enabled != enabled:
            self._toggle_viz()
