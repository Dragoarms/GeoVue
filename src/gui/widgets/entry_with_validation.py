# gui/widgets/entry_with_validation.py

import tkinter as tk


def create_entry_with_validation(
    parent,
    textvariable,
    theme_colors,
    font,
    validate_func=None,
    width=None,
    placeholder=None,
):
    entry = tk.Entry(
        parent,
        textvariable=textvariable,
        font=font,
        bg=theme_colors["field_bg"],
        fg=theme_colors["text"],
        insertbackground=theme_colors["text"],
        relief=tk.FLAT,
        bd=1,
        highlightbackground=theme_colors["field_border"],
        highlightthickness=1,
        width=width,
    )

    # Track whether placeholder is active
    entry.placeholder_active = False

    def apply_placeholder():
        if not textvariable.get() and placeholder:
            entry.placeholder_active = True
            textvariable.set(placeholder)
            entry.config(fg="#999999", font=(font[0], font[1], "italic"))

    def clear_placeholder(event=None):
        if entry.placeholder_active and entry.get() == placeholder:
            entry.placeholder_active = False
            textvariable.set("")
            entry.config(fg=theme_colors["text"], font=font)

    def restore_placeholder(event=None):
        if not entry.get() and placeholder:
            apply_placeholder()

    # Only apply placeholder if textvariable is empty
    if placeholder:
        current_value = textvariable.get()
        if current_value and current_value != placeholder:
            # There's already a real value, don't use placeholder
            entry.config(fg=theme_colors["text"], font=font)
        else:
            # No value, apply placeholder
            apply_placeholder()

        entry.bind("<FocusIn>", clear_placeholder)
        entry.bind("<FocusOut>", restore_placeholder)

    if validate_func:
        textvariable.trace_add("write", validate_func)

    return entry
