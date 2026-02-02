"""
TieLineCanvas - Overlay canvas for drawing correlation tie lines between drillholes.
This widget creates a transparent overlay for interactive tie line creation and display.
"""

import tkinter as tk
import logging
from typing import Dict, List, Optional, Tuple, Any
import math

from gui.DrillholeCorrelation.correlation_models import TieLine, TieLineType

logger = logging.getLogger(__name__)


class TieLineCanvas(tk.Canvas):
    """
    Transparent overlay canvas for drawing tie lines between drillholes.
    
    Features:
    - Interactive tie line creation with click-and-drag
    - Different line styles and colors by type
    - Confidence visualization through line thickness/opacity
    - Editable tie lines with context menu
    """
    
    def __init__(
        self,
        parent: tk.Widget,
        gui_manager: Any,
        theme_colors: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Initialize TieLineCanvas.
        
        Args:
            parent: Parent widget
            gui_manager: GUIManager for theming
            theme_colors: Optional theme colors override
        """
        logger.info("Initializing TieLineCanvas")
        
        # Get theme colors
        self.theme_colors = theme_colors or gui_manager.theme_colors
        
        # Initialize canvas with transparency
        super().__init__(
            parent,
            bg="",  # Transparent background
            highlightthickness=0,
            **kwargs
        )
        
        # Tie line storage
        self.tie_lines: List[TieLine] = []
        self.line_items: Dict[str, int] = {}  # tie_id -> canvas item id
        
        # Interaction state
        self.creating_line = False
        self.start_point: Optional[Tuple[float, float]] = None
        self.temp_line_id: Optional[int] = None
        self.selected_tie: Optional[str] = None
        
        # Line style defaults
        self.line_styles = {
            TieLineType.LITHOLOGY: {
                "color": "#FFD700",  # Gold
                "width": 2,
                "dash": None
            },
            TieLineType.STRUCTURE: {
                "color": "#00CED1",  # Dark turquoise
                "width": 3,
                "dash": (5, 3)
            },
            TieLineType.GRADE: {
                "color": "#FF6347",  # Tomato
                "width": 2,
                "dash": None
            },
            TieLineType.ALTERATION: {
                "color": "#9370DB",  # Medium purple
                "width": 2,
                "dash": (3, 2)
            },
            TieLineType.MARKER: {
                "color": "#32CD32",  # Lime green
                "width": 3,
                "dash": None
            },
            TieLineType.CUSTOM: {
                "color": "#C0C0C0",  # Silver
                "width": 2,
                "dash": (2, 2)
            }
        }
        
        # Bind events
        self._bind_events()
        
        logger.debug("TieLineCanvas initialized")
    
    def _bind_events(self) -> None:
        """Bind canvas events."""
        self.bind("<Button-1>", self._on_click)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Button-3>", self._on_right_click)  # Context menu
        self.bind("<Motion>", self._on_hover)
        
        logger.debug("TieLineCanvas events bound")
    
    def add_tie_line(self, tie_line: TieLine) -> None:
        """
        Add a tie line to the canvas.
        
        Args:
            tie_line: TieLine object to add
        """
        logger.info(f"Adding tie line: {tie_line.tie_id}")
        logger.debug(f"  From: {tie_line.source_hole}@{tie_line.source_depth}m")
        logger.debug(f"  To: {tie_line.target_hole}@{tie_line.target_depth}m")
        logger.debug(f"  Type: {tie_line.line_type.value}")
        
        # Store tie line
        self.tie_lines.append(tie_line)
        
        # Draw on canvas
        self._draw_tie_line(tie_line)
    
    def _draw_tie_line(self, tie_line: TieLine) -> None:
        """Draw a tie line on the canvas."""
        # Get positions from hole columns
        # TODO: Need to interface with column widgets to get pixel positions
        
        # For now, create placeholder line
        style = self.line_styles.get(tie_line.line_type, self.line_styles[TieLineType.CUSTOM])
        
        # Apply confidence to width
        width = int(style["width"] * tie_line.confidence)
        if width < 1:
            width = 1
        
        # Create line (placeholder coordinates for now)
        x1, y1 = 100, 100  # TODO: Get from source column
        x2, y2 = 300, 150  # TODO: Get from target column
        
        line_id = self.create_line(
            x1, y1, x2, y2,
            fill=tie_line.color or style["color"],
            width=tie_line.line_width or width,
            dash=style["dash"],
            tags=(f"tie_{tie_line.tie_id}", "tie_line"),
            capstyle="round",
            smooth=True
        )
        
        # Store reference
        self.line_items[tie_line.tie_id] = line_id
        
        logger.debug(f"Drew tie line {tie_line.tie_id} as canvas item {line_id}")
    
    def remove_tie_line(self, tie_id: str) -> None:
        """
        Remove a tie line by ID.
        
        Args:
            tie_id: ID of tie line to remove
        """
        logger.info(f"Removing tie line: {tie_id}")
        
        # Remove from list
        self.tie_lines = [t for t in self.tie_lines if t.tie_id != tie_id]
        
        # Remove from canvas
        if tie_id in self.line_items:
            self.delete(self.line_items[tie_id])
            del self.line_items[tie_id]
            logger.debug(f"Removed canvas item for tie {tie_id}")
    
    def clear_all(self) -> None:
        """Clear all tie lines."""
        logger.info("Clearing all tie lines")
        
        # Clear canvas
        for item_id in self.line_items.values():
            self.delete(item_id)
        
        # Clear storage
        self.tie_lines.clear()
        self.line_items.clear()
        
        logger.debug("All tie lines cleared")
    
    def set_creation_mode(self, enabled: bool) -> None:
        """
        Enable or disable tie line creation mode.
        
        Args:
            enabled: Whether to enable creation mode
        """
        self.creating_line = enabled
        
        if enabled:
            logger.info("Entered tie line creation mode")
            self.config(cursor="crosshair")
        else:
            logger.info("Exited tie line creation mode")
            self.config(cursor="")
            self._cancel_temp_line()
    
    def _on_click(self, event: tk.Event) -> None:
        """Handle mouse click."""
        if not self.creating_line:
            # Check if clicking on existing tie line for selection
            item = self.find_closest(event.x, event.y)[0]
            tags = self.gettags(item)
            
            for tag in tags:
                if tag.startswith("tie_"):
                    tie_id = tag[4:]  # Remove "tie_" prefix
                    self._select_tie_line(tie_id)
                    return
        else:
            # Start creating new tie line
            logger.debug(f"Starting tie line at ({event.x}, {event.y})")
            self.start_point = (event.x, event.y)
            
            # Create temporary line
            self.temp_line_id = self.create_line(
                event.x, event.y, event.x, event.y,
                fill="#FFFF00",
                width=2,
                dash=(5, 5),
                tags="temp_line"
            )
    
    def _on_drag(self, event: tk.Event) -> None:
        """Handle mouse drag."""
        if self.creating_line and self.temp_line_id and self.start_point:
            # Update temporary line
            self.coords(
                self.temp_line_id,
                self.start_point[0], self.start_point[1],
                event.x, event.y
            )
    
    def _on_release(self, event: tk.Event) -> None:
        """Handle mouse release."""
        if self.creating_line and self.temp_line_id and self.start_point:
            # Finalize tie line creation
            logger.debug(f"Ending tie line at ({event.x}, {event.y})")
            
            # TODO: Convert pixel coordinates to hole/depth coordinates
            # TODO: Create actual TieLine object
            # TODO: Show tie line properties dialog
            
            # For now, just remove temp line
            self._cancel_temp_line()
    
    def _on_right_click(self, event: tk.Event) -> None:
        """Handle right click for context menu."""
        # Find closest tie line
        item = self.find_closest(event.x, event.y)[0]
        tags = self.gettags(item)
        
        tie_id = None
        for tag in tags:
            if tag.startswith("tie_"):
                tie_id = tag[4:]
                break
        
        if tie_id:
            logger.debug(f"Right click on tie line: {tie_id}")
            self._show_context_menu(event.x_root, event.y_root, tie_id)
    
    def _on_hover(self, event: tk.Event) -> None:
        """Handle mouse hover."""
        # Find item under cursor
        item = self.find_closest(event.x, event.y)[0]
        tags = self.gettags(item)
        
        is_tie_line = any(tag.startswith("tie_") for tag in tags)
        
        if is_tie_line and not self.creating_line:
            self.config(cursor="hand2")
        elif self.creating_line:
            self.config(cursor="crosshair")
        else:
            self.config(cursor="")
    
    def _cancel_temp_line(self) -> None:
        """Cancel temporary line creation."""
        if self.temp_line_id:
            self.delete(self.temp_line_id)
            self.temp_line_id = None
        self.start_point = None
    
    def _select_tie_line(self, tie_id: str) -> None:
        """Select a tie line."""
        logger.debug(f"Selecting tie line: {tie_id}")
        
        # Clear previous selection
        if self.selected_tie and self.selected_tie in self.line_items:
            self.itemconfig(self.line_items[self.selected_tie], width=2)
        
        # Highlight selected
        if tie_id in self.line_items:
            self.selected_tie = tie_id
            self.itemconfig(self.line_items[tie_id], width=4)
    
    def _show_context_menu(self, x: int, y: int, tie_id: str) -> None:
        """Show context menu for tie line."""
        logger.debug(f"Showing context menu for tie {tie_id}")
        
        # Create menu
        menu = tk.Menu(self, tearoff=0)
        menu.configure(
            bg=self.theme_colors["menu_bg"],
            fg=self.theme_colors["menu_fg"],
            activebackground=self.theme_colors["menu_active_bg"],
            activeforeground=self.theme_colors["menu_active_fg"]
        )
        
        # Add menu items
        menu.add_command(label="Edit Properties", command=lambda: self._edit_tie_line(tie_id))
        menu.add_command(label="Change Color", command=lambda: self._change_tie_color(tie_id))
        menu.add_separator()
        menu.add_command(label="Delete", command=lambda: self.remove_tie_line(tie_id))
        
        # Show menu
        menu.post(x, y)
    
    def _edit_tie_line(self, tie_id: str) -> None:
        """Edit tie line properties."""
        logger.info(f"Editing tie line: {tie_id}")
        # TODO: Show properties dialog
    
    def _change_tie_color(self, tie_id: str) -> None:
        """Change tie line color."""
        logger.info(f"Changing color for tie: {tie_id}")
        # TODO: Show color picker
    
    def update_positions(self) -> None:
        """Update all tie line positions (e.g., after scroll or resize)."""
        logger.debug("Updating all tie line positions")
        
        for tie_line in self.tie_lines:
            if tie_line.tie_id in self.line_items:
                # TODO: Recalculate positions from column widgets
                pass
    
    def get_tie_lines_for_hole(self, hole_id: str) -> List[TieLine]:
        """
        Get all tie lines connected to a specific hole.
        
        Args:
            hole_id: Hole identifier
            
        Returns:
            List of TieLine objects
        """
        return [
            tie for tie in self.tie_lines
            if tie.source_hole == hole_id or tie.target_hole == hole_id
        ]
    
    def get_tie_lines_at_depth(
        self, 
        hole_id: str, 
        depth: float, 
        tolerance: float = 1.0
    ) -> List[TieLine]:
        """
        Get tie lines near a specific depth in a hole.
        
        Args:
            hole_id: Hole identifier
            depth: Depth value
            tolerance: Depth tolerance for matching
            
        Returns:
            List of TieLine objects
        """
        result = []
        
        for tie in self.tie_lines:
            if tie.source_hole == hole_id:
                if abs(tie.source_depth - depth) <= tolerance:
                    result.append(tie)
            elif tie.target_hole == hole_id:
                if abs(tie.target_depth - depth) <= tolerance:
                    result.append(tie)
        
        return result
