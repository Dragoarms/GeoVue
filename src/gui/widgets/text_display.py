# gui/widgets/text_display.py

import tkinter as tk

def create_text_display(parent, theme_colors, font, height=10, wrap=tk.WORD, readonly=True):
    
    """
    Creates a themed multi-line text display widget.

    The widget is styled using theme colors and optionally made read-only, making it ideal
    for displaying logs, status messages, or static multi-line content.

    Args:
        parent: Parent Tkinter widget.
        theme_colors: Dictionary of theme color values.
        font: Font tuple used in the text display.
        height: Number of visible text lines.
        wrap: Text wrapping mode (e.g., tk.WORD).
        readonly: Whether the widget is read-only.

    Returns:
        A Tkinter Text widget with applied styling.
    """


    text_widget = tk.Text(
        parent,
        height=height,
        wrap=wrap,
        font=font,
        bg=theme_colors["field_bg"],
        fg=theme_colors["text"],
        insertbackground=theme_colors["text"],
        relief=tk.FLAT,
        bd=1,
        highlightbackground=theme_colors["field_border"],
        highlightthickness=1
    )
    if readonly:
        text_widget.config(state=tk.DISABLED)
    return text_widget
