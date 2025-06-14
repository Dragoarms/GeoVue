import tkinter as tk
from tkinter import ttk

def create_custom_checkbox(parent, text, variable, theme_colors, command=None, translator=None):
    """
    Create a styled checkbox consistent with the application's theme.

    Args:
        parent: The parent Tkinter widget.
        text: The label text.
        variable: A tk.BooleanVar controlling the checkbox state.
        theme_colors: The current theme color dictionary.
        command: Optional function to run when toggled.
        translator: Optional translation function (e.g. t())

    Returns:
        Frame: The complete checkbox frame.
    """
    frame = ttk.Frame(parent, style='Content.TFrame')

    checkbox_size = 18
    checkbox_frame = tk.Frame(
        frame,
        width=checkbox_size,
        height=checkbox_size,
        bg=theme_colors["checkbox_bg"],
        highlightbackground=theme_colors["field_border"],
        highlightthickness=1,
    )
    checkbox_frame.pack(side=tk.LEFT, padx=(0, 5))
    checkbox_frame.pack_propagate(False)

    checkmark = tk.Label(
        checkbox_frame,
        text="✓",
        bg=theme_colors["checkbox_bg"],
        fg=theme_colors["checkbox_fg"],
        font=("Arial", 12, "bold")
    )
    

    def update_state(*_):
        if variable.get():
            checkmark.pack(fill=tk.BOTH, expand=True)
        else:
            checkmark.pack_forget()

    def toggle(_=None):
        variable.set(not variable.get())
        if command:
            command()

    variable.trace_add("write", update_state)
    update_state()

    checkbox_frame.bind("<Button-1>", toggle)
    checkmark.bind("<Button-1>", toggle)

    translated_text = translator(text) if translator else text
    label = ttk.Label(frame, text=translated_text, style='Content.TLabel')
    label.pack(side=tk.LEFT)
    label.bind("<Button-1>", toggle)

    checkmark.config(cursor="hand2")
    label.config(cursor="hand2")
    checkbox_frame.config(cursor="hand2")
    
    # Store original text and translator for updates
    frame._original_text = text
    frame._translator = translator
    frame._label = label
    
    # Add method to update translation
    def update_translation(new_translator=None):
        if new_translator:
            frame._translator = new_translator
        if frame._translator and frame._original_text:
            frame._label.config(text=frame._translator(frame._original_text))
    
    frame.update_translation = update_translation

    return frame