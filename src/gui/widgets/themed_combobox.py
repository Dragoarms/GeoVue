# gui/widgets/themed_combobox.py

import tkinter as tk
from tkinter import ttk
import logging


def create_themed_combobox(
    parent,
    variable,
    values,
    theme_colors,
    fonts,
    width=10,
    readonly=True,
    validate_func=None,
):
    """Create a styled ttk.Combobox with custom appearance using theme_colors."""
    try:
        # ===================================================
        # CRITICAL: Set dropdown colors BEFORE anything else
        # ===================================================
        try:
            root = parent.winfo_toplevel()

            # Debug: Print current theme colors
            print(
                f"DEBUG: Setting dropdown colors - bg: {theme_colors['field_bg']}, fg: {theme_colors['text']}"
            )

            # Apply to all TCombobox dropdowns - use multiple patterns for compatibility
            root.option_add("*TCombobox*Listbox.background", theme_colors["field_bg"])
            root.option_add("*TCombobox*Listbox.foreground", theme_colors["text"])
            root.option_add(
                "*TCombobox*Listbox.selectBackground",
                theme_colors.get("accent_blue", "#3a7ca5"),
            )
            root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
            root.option_add("*TCombobox*Listbox.font", fonts["normal"])

            # Additional patterns that Windows might use
            root.option_add("*Listbox.background", theme_colors["field_bg"])
            root.option_add("*Listbox.foreground", theme_colors["text"])
            root.option_add(
                "*Listbox.selectBackground", theme_colors.get("accent_blue", "#3a7ca5")
            )
            root.option_add("*Listbox.selectForeground", "#ffffff")

            # Force immediate update
            root.update_idletasks()

            print(f"DEBUG: Dropdown colors set successfully")
        except Exception as e:
            print(f"ERROR: Failed to set dropdown colors: {e}")
            logging.error(f"Could not set dropdown colors: {e}")

        # Use a unique style name to avoid conflicts
        import random

        style_name = f"ThemedCustom{random.randint(1000,9999)}.TCombobox"

        # Configure the style
        style = ttk.Style()

        # Debug: Check existing styles
        print(f"DEBUG: Creating style {style_name}")

        style.configure(
            style_name,
            fieldbackground=theme_colors["field_bg"],
            background=theme_colors["field_bg"],
            foreground=theme_colors["text"],
            bordercolor=theme_colors["field_border"],
            arrowcolor=theme_colors["text"],
            selectbackground=theme_colors.get("accent_blue", "#3a7ca5"),
            selectforeground="#ffffff",
            insertcolor=theme_colors["text"],
        )

        # Debug: Print style configuration
        print(
            f"DEBUG: Style configured with fieldbackground={theme_colors['field_bg']}"
        )

        # Add style mapping for different states
        style.map(
            style_name,
            fieldbackground=[
                ("readonly", theme_colors["field_bg"]),
                ("disabled", theme_colors["secondary_bg"]),
                ("active", theme_colors["field_bg"]),
            ],
            foreground=[
                ("readonly", theme_colors["text"]),
                ("disabled", theme_colors["field_border"]),
                ("active", theme_colors["text"]),
            ],
            background=[
                ("readonly", theme_colors["field_bg"]),
                ("disabled", theme_colors["secondary_bg"]),
                ("active", theme_colors["field_bg"]),
            ],
            arrowcolor=[
                ("disabled", theme_colors["field_border"]),
                ("pressed", theme_colors.get("accent_blue", "#3a7ca5")),
                ("active", theme_colors.get("accent_blue", "#3a7ca5")),
            ],
        )

        # Create and configure the combobox with custom style
        combo = ttk.Combobox(
            parent,
            textvariable=variable,
            values=values,
            width=width,
            font=fonts["normal"],
            style=style_name,  # Use the custom style
            state="readonly" if readonly else "normal",
        )

        # Store theme colors and style name for later updates
        combo._theme_colors = theme_colors
        combo._style_name = style_name

        # Debug: Verify combobox creation
        print(f"DEBUG: Combobox created with style {style_name}")

        def update_theme_method(new_theme_colors=None):
            if new_theme_colors:
                combo._theme_colors = new_theme_colors
            try:
                style = ttk.Style()
                # Update the custom style
                style.configure(
                    combo._style_name,
                    fieldbackground=combo._theme_colors["field_bg"],
                    background=combo._theme_colors["field_bg"],
                    foreground=combo._theme_colors["text"],
                    bordercolor=combo._theme_colors["field_border"],
                    arrowcolor=combo._theme_colors["text"],
                    selectbackground=combo._theme_colors.get("accent_blue", "#3a7ca5"),
                    selectforeground="#ffffff",
                )

                style.map(
                    combo._style_name,
                    fieldbackground=[
                        ("readonly", combo._theme_colors["field_bg"]),
                        ("disabled", combo._theme_colors["secondary_bg"]),
                        ("active", combo._theme_colors["field_bg"]),
                    ],
                    foreground=[
                        ("readonly", combo._theme_colors["text"]),
                        ("disabled", combo._theme_colors["field_border"]),
                        ("active", combo._theme_colors["text"]),
                    ],
                    background=[
                        ("readonly", combo._theme_colors["field_bg"]),
                        ("disabled", combo._theme_colors["secondary_bg"]),
                        ("active", combo._theme_colors["field_bg"]),
                    ],
                    arrowcolor=[
                        ("disabled", combo._theme_colors["field_border"]),
                        ("pressed", combo._theme_colors["accent_blue"]),
                        ("active", combo._theme_colors["accent_blue"]),
                    ],
                )

                # Update dropdown colors
                try:
                    root = combo.winfo_toplevel()
                    root.option_add(
                        "*TCombobox*Listbox.background", combo._theme_colors["field_bg"]
                    )
                    root.option_add(
                        "*TCombobox*Listbox.foreground", combo._theme_colors["text"]
                    )
                except:
                    pass

                return True
            except Exception as e:
                logging.error(f"Failed to update combobox theme: {e}")
                return False

        # Attach the update method to the combobox
        combo.update_theme = update_theme_method

        if validate_func:
            variable.trace_add("write", validate_func)

        return combo

    except Exception as e:
        logging.error(f"Error creating themed combobox: {e}")
        # Fallback to basic combobox
        return ttk.Combobox(
            parent,
            textvariable=variable,
            values=values,
            width=width,
            state="readonly" if readonly else "normal",
        )
