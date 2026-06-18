"""
Three-state toggle widget for logging review.
States: None (untouched), True (checked), False (unchecked)
"""

import tkinter as tk
from typing import Optional, Callable


class ThreeStateToggle(tk.Frame):
    """
    A toggle button with three states:
    - None: Not yet interacted (gray)
    - True: Explicitly checked (green)
    - False: Explicitly unchecked (blue)
    """
    
    def __init__(self, parent, text: str, theme_colors: dict, 
                 initial_value: Optional[bool] = None,
                 on_change: Optional[Callable] = None):
        """
        Initialize the three-state toggle.
        
        Args:
            parent: Parent widget
            text: Label text
            theme_colors: Theme color dictionary
            initial_value: Initial state (None, True, or False)
            on_change: Callback when value changes
        """
        super().__init__(parent)
        
        self.text = text
        self.theme_colors = theme_colors
        self.on_change = on_change
        self._value = initial_value
        self._has_interacted = False
        
        # Create the button
        self.button = tk.Button(
            self,
            text=text,
            font=('Arial', 9, 'bold'),
            relief=tk.RAISED,
            bd=2,
            padx=5,
            pady=3,
            width=12,
            command=self._toggle
        )
        self.button.pack(fill=tk.BOTH, expand=True)
        
        # Set initial appearance
        self._update_appearance()
    
    def _toggle(self):
        """Handle toggle button click."""
        self._has_interacted = True
        
        # Cycle through states: None -> True -> False -> True -> False...
        if self._value is None:
            self._value = True
        elif self._value is True:
            self._value = False
        else:  # False
            self._value = True
            
        self._update_appearance()
        
        # Call callback if provided
        if self.on_change:
            self.on_change()
    
    def _update_appearance(self):
        """Update button appearance based on state."""
        if not self._has_interacted and self._value is None:
            # Untouched state - gray
            self.button.config(
                bg=self.theme_colors["secondary_bg"],
                fg=self.theme_colors["text"],
                activebackground=self.theme_colors["secondary_bg"]
            )
        elif self._value is True:
            # Checked state - green
            self.button.config(
                bg=self.theme_colors["accent_green"],
                fg='white',
                activebackground=self.theme_colors["accent_green"]
            )
        else:  # False or None with interaction
            # Unchecked state - blue
            self.button.config(
                bg=self.theme_colors["accent_blue"],
                fg='white',
                activebackground=self.theme_colors["accent_blue"]
            )
    
    def get(self) -> Optional[bool]:
        """Get the current value."""
        return self._value
    
    def set(self, value: Optional[bool], mark_as_interacted: bool = False):
        """Set the value programmatically."""
        self._value = value
        if mark_as_interacted:
            self._has_interacted = True
        self._update_appearance()
    
    def has_interacted(self) -> bool:
        """Check if user has interacted with this toggle."""
        return self._has_interacted
    
    def reset_interaction(self):
        """Reset interaction state."""
        self._has_interacted = False
        self._update_appearance()