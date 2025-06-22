# gui/widgets/entry_with_validation.py

import tkinter as tk

def create_entry_with_validation(parent, textvariable, theme_colors, font, validate_func=None, width=None, placeholder=None):
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
        width=width
    )

    def apply_placeholder():
        if not textvariable.get():
            entry.insert(0, placeholder)
            entry.config(fg="#999999", font=(font[0], font[1], "italic"))

    def clear_placeholder(event=None):
        if entry.get() == placeholder:
            entry.delete(0, tk.END)
            entry.config(fg=theme_colors["text"], font=font)

    def restore_placeholder(event=None):
        if not entry.get():
            apply_placeholder()

    if placeholder:
        apply_placeholder()
        entry.bind("<FocusIn>", clear_placeholder)
        entry.bind("<FocusOut>", restore_placeholder)

    if validate_func:
        textvariable.trace_add("write", validate_func)

    return entry
