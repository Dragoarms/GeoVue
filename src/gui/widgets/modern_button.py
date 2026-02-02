import tkinter as tk
from gui.dialog_helper import DialogHelper
import logging


class ModernButton:
    """
    A modern styled button with hover effects.
    """

    def __init__(
        self,
        parent,
        text,
        color,
        command,
        icon=None,
        theme_colors=None,
        fonts=None,
        enabled=True,
        width=None,
        height=None,
    ):
        """
        Initialize a modern styled button. Uses dialog manager to translate display text.

        Args:
            parent: Parent widget
            text: Button text
            color: Background color
            command: Callback function
            icon: Optional text icon
            theme_colors: Theme colors dictionary
            fonts: Fonts dictionary from GUI manager
            enabled: Whether button is enabled
            width: Optional width (not currently used, for API compatibility)
            height: Optional height (not currently used, for API compatibility)
        """
        self.logger = logging.getLogger(__name__)
        self.parent = parent
        self.text = text
        self.base_color = color
        self.command = command
        self.icon = icon
        self.theme_colors = theme_colors or {}
        self.fonts = fonts or {}
        self.enabled = enabled
        self.width = width
        self.height = height

        # translate all button text
        text = DialogHelper.t(text)

        # Create the button frame with subtle border for depth
        border_color = self._lighten_color(color, 0.1)
        self.frame = tk.Frame(
            parent,
            background=color,
            highlightbackground=border_color,
            highlightthickness=1,
            bd=0,
            cursor="hand2",
        )

        # Prefix with icon if provided
        display_text = f"{icon}  {text}" if icon else text

        # Use the button font from GUI manager, with fallback
        button_font = self.fonts.get("button", ("Segoe UI Semibold", 10))

        # Create the label for button text with automatic contrast
        text_color = self._get_contrast_color(color)
        self.label = tk.Label(
            self.frame,
            text=display_text,
            background=color,
            foreground=text_color,
            font=button_font,
            padx=8,
            pady=4,
            cursor="hand2",
            width=self.width if self.width is not None else 0,
            height=self.height if self.height is not None else 0,
        )
        self.label.pack(fill=tk.BOTH, expand=True)

        # Apply initial enabled state
        if not self.enabled:
            self.set_state("disabled")

        # Bind events
        self.frame.bind("<Enter>", self._on_enter)
        self.label.bind("<Enter>", self._on_enter)
        self.frame.bind("<Leave>", self._on_leave)
        self.label.bind("<Leave>", self._on_leave)
        self.frame.bind("<Button-1>", self._on_click)
        self.label.bind("<Button-1>", self._on_click)

    def focus_set(self):
        """
        Implement focus_set to make ModernButton compatible with standard Tkinter widgets.
        This delegates focus to the underlying canvas or frame.
        """
        if hasattr(self, "canvas") and self.canvas:
            self.canvas.focus_set()
        elif hasattr(self, "frame") and self.frame:
            self.frame.focus_set()
        # If neither exists, silently do nothing
        return self

    def focus_get(self):
        """
        Get the widget that currently has focus.

        Returns:
            Widget that has focus or None
        """
        if hasattr(self, "frame") and self.frame:
            return self.frame.focus_get()
        return None

    def set_text(self, text):
        """
        Update the button text.

        Args:
            text (str): New text for the button

        Returns:
            ModernButton: Self for method chaining
        """
        try:
            self.text = text

            # Update displayed text with icon if present
            display_text = (
                f"{self.icon} {text}" if hasattr(self, "icon") and self.icon else text
            )

            # Update the label text if it exists
            if hasattr(self, "label") and self.label and hasattr(self.label, "config"):
                self.label.config(text=display_text)

            return self
        except Exception as e:
            # Just log and continue - don't crash the app for a button text update
            print(f"Error updating button text: {str(e)}")
            return self

    def update_button(self, text=None, color=None):
        """Update both text and color of the button."""
        if text is not None:
            self.set_text(text)
        if color is not None:
            self.configure_color(color)
        return self

    def configure_color(self, new_color):
        """Change the button's color dynamically."""
        self.base_color = new_color
        text_color = self._get_contrast_color(new_color)

        # Update the frame and label backgrounds and text color
        if hasattr(self, "frame") and self.frame:
            self.frame.config(background=new_color)
        if hasattr(self, "label") and self.label:
            self.label.config(background=new_color, foreground=text_color)
    
    def _update_colors(self):
        """Update colors after base_color change. Alias for configure_color."""
        self.configure_color(self.base_color)

    def update_color(self, new_color):
        """Alias for configure_color for backward compatibility."""
        self.configure_color(new_color)

    def _on_enter(self, event):
        """Handle mouse enter event."""
        if not self.enabled:
            return

        # Use a lighter color for hover effect
        hover_color = self._lighten_color(self.base_color, 0.15)
        text_color = self._get_contrast_color(hover_color)
        self.frame.config(background=hover_color)
        self.label.config(background=hover_color, foreground=text_color)

    def _on_leave(self, event):
        """Handle mouse leave event."""
        if not self.enabled:
            return

        # Restore original color with proper contrast
        text_color = self._get_contrast_color(self.base_color)
        self.frame.config(background=self.base_color)
        self.label.config(background=self.base_color, foreground=text_color)

    def _on_click(self, event):
        """Handle mouse click event."""
        self.logger.debug(f"Button '{self.text}' clicked, enabled={self.enabled}")
        
        if not self.enabled:
            self.logger.debug(f"Button '{self.text}' is disabled, ignoring click")
            return

        # Use a darker color for click effect
        click_color = self._darken_color(self.base_color, 0.15)
        text_color = self._get_contrast_color(click_color)
        self.frame.config(background=click_color)
        self.label.config(background=click_color, foreground=text_color)

        # Execute command
        if self.command:
            self.logger.debug(f"Executing command for button '{self.text}'")
            try:
                self.command()
                self.logger.debug(f"Command executed successfully for button '{self.text}'")
            except Exception as e:
                self.logger.error(f"Error executing command for button '{self.text}': {e}", exc_info=True)
        else:
            self.logger.warning(f"No command defined for button '{self.text}'")

        # Reset color after a short delay
        self.frame.after(100, self._reset_color)

    def bind(self, sequence, func, add=None):
        """
        Bind an event to the button.

        Args:
            sequence: Event sequence (e.g., '<Return>', '<Escape>')
            func: Function to call when event occurs
            add: Optional, whether to add to existing bindings

        Returns:
            Binding ID that can be used with unbind
        """
        # Bind to both frame and label to ensure event is caught
        frame_id = self.frame.bind(sequence, func, add)
        self.label.bind(sequence, func, add)
        return frame_id

    def unbind(self, sequence, funcid=None):
        """
        Unbind an event from the button.

        Args:
            sequence: Event sequence to unbind
            funcid: Optional function ID returned by bind
        """
        self.frame.unbind(sequence, funcid)
        self.label.unbind(sequence, funcid)
        return self

    def bind_all(self, sequence, func, add=None):
        """
        Bind an event to all instances of this button class.

        Args:
            sequence: Event sequence
            func: Function to call
            add: Whether to add to existing bindings
        """
        return self.frame.bind_all(sequence, func, add)

    def unbind_all(self, sequence):
        """
        Unbind an event from all instances.

        Args:
            sequence: Event sequence to unbind
        """
        self.frame.unbind_all(sequence)
        return self

    def event_generate(self, sequence, **kwargs):
        """
        Generate an event on the button.

        Args:
            sequence: Event sequence to generate
            **kwargs: Additional event parameters
        """
        self.frame.event_generate(sequence, **kwargs)
        return self

    def _reset_color(self, event=None):
        """Reset button color to original."""
        try:
            if self.frame and self.frame.winfo_exists():
                text_color = self._get_contrast_color(self.base_color)
                self.frame.config(background=self.base_color)

            if self.label and self.label.winfo_exists():
                self.label.config(background=self.base_color, foreground=text_color)
        except (tk.TclError, RuntimeError) as e:
            # Widget already destroyed, ignore error
            pass

    def _lighten_color(self, color_hex, factor=0.1):
        """Lighten a color by the given factor."""
        # Convert hex to RGB
        r, g, b = self._hex_to_rgb(color_hex)

        # Lighten
        r = min(255, int(r + (255 - r) * factor))
        g = min(255, int(g + (255 - g) * factor))
        b = min(255, int(b + (255 - b) * factor))

        # Convert back to hex
        return f"#{r:02x}{g:02x}{b:02x}"

    def _darken_color(self, color_hex, factor=0.1):
        """Darken a color by the given factor."""
        # Convert hex to RGB
        r, g, b = self._hex_to_rgb(color_hex)

        # Darken
        r = max(0, int(r * (1 - factor)))
        g = max(0, int(g * (1 - factor)))
        b = max(0, int(b * (1 - factor)))

        # Convert back to hex
        return f"#{r:02x}{g:02x}{b:02x}"

    def _hex_to_rgb(self, color_hex):
        """Convert hex color to RGB tuple."""
        # Handle different hex formats
        color_hex = color_hex.lstrip("#")
        if len(color_hex) == 3:
            color_hex = "".join([c * 2 for c in color_hex])

        # Convert to RGB
        return tuple(int(color_hex[i : i + 2], 16) for i in (0, 2, 4))

    def _get_contrast_color(self, bg_color):
        """
        Calculate appropriate text color (black or white) for maximum contrast.
        Uses WCAG relative luminance formula.

        Args:
            bg_color: Background color in hex format

        Returns:
            str: Either "white" or "black" for best contrast
        """
        try:
            # Get RGB values
            r, g, b = self._hex_to_rgb(bg_color)

            # Calculate relative luminance (WCAG formula)
            # Convert to 0-1 range
            r = r / 255.0
            g = g / 255.0
            b = b / 255.0

            # Apply gamma correction
            r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
            g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
            b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4

            # Calculate luminance
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

            # Use white text for dark backgrounds (luminance < 0.5)
            # Use black text for light backgrounds (luminance >= 0.5)
            return "white" if luminance < 0.5 else "black"
        except:
            # Fallback to white if any error occurs
            return "white"

    def pack(self, **kwargs):
        """Pack the button frame."""
        self.frame.pack(**kwargs)
        return self

    def grid(self, **kwargs):
        """Grid the button frame."""
        self.frame.grid(**kwargs)
        return self

    def place(self, **kwargs):
        """Place the button frame."""
        self.frame.place(**kwargs)
        return self

    def pack_forget(self):
        """Remove the button from pack geometry management."""
        self.frame.pack_forget()
        return self

    def grid_forget(self):
        """Remove the button from grid geometry management."""
        self.frame.grid_forget()
        return self

    def place_forget(self):
        """Remove the button from place geometry management."""
        self.frame.place_forget()
        return self

    def destroy(self):
        """Destroy the button widget."""
        if hasattr(self, "frame") and self.frame:
            self.frame.destroy()
        return self

    def winfo_exists(self):
        """Check if the button widget still exists."""
        if hasattr(self, "frame") and self.frame:
            try:
                return self.frame.winfo_exists()
            except:
                return False
        return False

    def set_state(self, state):
        """
        Set the button state.

        Args:
            state: 'normal' or 'disabled'
        """
        if state == "normal":
            self.enabled = True
            self.frame.config(cursor="hand2")
            self.label.config(cursor="hand2")
            self.frame.config(background=self.base_color)
            self.label.config(background=self.base_color)
        else:
            self.enabled = False
            self.frame.config(cursor="")
            self.label.config(cursor="")
            # Use a grayed-out color for disabled state
            disabled_color = self._mix_colors(self.base_color, "#888888", 0.7)
            self.frame.config(background=disabled_color)
            self.label.config(background=disabled_color, foreground="#cccccc")

        return self

    def _mix_colors(self, color1, color2, weight=0.5):
        """Mix two colors based on weight."""
        r1, g1, b1 = self._hex_to_rgb(color1)
        r2, g2, b2 = self._hex_to_rgb(color2)

        r = int(r1 * weight + r2 * (1 - weight))
        g = int(g1 * weight + g2 * (1 - weight))
        b = int(b1 * weight + b2 * (1 - weight))

        return f"#{r:02x}{g:02x}{b:02x}"

    def configure(self, **kwargs):
        """Configure button properties."""
        if "text" in kwargs:
            self.set_text(kwargs.pop("text"))

        if "state" in kwargs:
            self.set_state(kwargs.pop("state"))

        if "command" in kwargs:
            self.command = kwargs.pop("command")

        if "color" in kwargs:
            # Use the existing configure_color method
            self.configure_color(kwargs.pop("color"))

        # Handle font separately - apply it to the label, not the frame
        if "font" in kwargs:
            font = kwargs.pop("font")
            if hasattr(self, "label") and self.label:
                self.label.configure(font=font)

        # Apply remaining configurations to the frame
        if kwargs:
            self.frame.configure(**kwargs)

        return self

    # Alias for configure
    config = configure
