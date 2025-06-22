import tkinter as tk
from tkinter import ttk
from .entry_with_validation import create_entry_with_validation
from .themed_combobox import create_themed_combobox  # Add this import

def create_field_with_label(
    parent, label_text, variable, theme_colors, fonts,
    translator=None, field_type="entry", readonly=False,
    width=20, validate_func=None, values=None, placeholder=None
):
    """
    Creates a labeled input field (entry or combobox) with consistent horizontal layout and theming.

    The label appears on the left, and the field (entry or combobox) on the right. Supports both editable
    and read-only entry fields, and optionally styled comboboxes with provided values.

    Args:
        parent: Parent Tkinter widget.
        label_text: Text for the label shown to the left of the input.
        variable: Tkinter variable (StringVar or similar) bound to the field.
        theme_colors: Dictionary of theme color values.
        fonts: Dictionary of font styles.
        translator: Optional translation function for label text.
        field_type: One of "entry" or "combobox".
        readonly: Whether the field should be read-only.
        width: Width of the label area.
        validate_func: Optional function triggered on input validation.
        values: List of values for combobox options (used only if field_type="combobox").
        placeholder: Optional placeholder text for entry fields.

    Returns:
        A tuple of (Frame, field widget) for layout integration and direct access.
    """
    frame = ttk.Frame(parent, style='Content.TFrame')
    frame.pack(fill=tk.X, pady=5)

    label = ttk.Label(
        frame,
        text=translator(label_text) if translator else label_text,
        width=width,
        anchor='w',
        style='Content.TLabel'
    )
    label.pack(side=tk.LEFT)

    if field_type == "entry":
        if readonly:
            field = tk.Entry(
                frame,
                textvariable=variable,
                state='readonly',
                font=fonts["normal"],
                bg=theme_colors["field_bg"],
                fg=theme_colors["text"],
                readonlybackground=theme_colors["field_bg"],
                relief=tk.FLAT,
                bd=1,
                highlightbackground=theme_colors["field_border"],
                highlightthickness=1
            )
        else:
            field = create_entry_with_validation(
                frame, variable, theme_colors, fonts["normal"], validate_func=validate_func, placeholder=placeholder
            )
        field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

    elif field_type == "combobox":
        field = create_themed_combobox(
            frame,
            variable,
            values or [],
            theme_colors,
            fonts,
            width=15,
            readonly=readonly,
            validate_func=validate_func
        )
        field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

    # Store original text and translator for updates
    frame._original_text = label_text
    frame._translator = translator
    frame._label = label
    
    # Add method to update translation
    def update_translation(new_translator=None):
        if new_translator:
            frame._translator = new_translator
        if frame._translator and frame._original_text:
            frame._label.config(text=frame._translator(frame._original_text))
    
    frame.update_translation = update_translation
    
    return frame, field