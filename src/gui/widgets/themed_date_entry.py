# gui/widgets/themed_date_entry.py

import logging
import tkinter as tk

from gui.widgets.entry_with_validation import create_entry_with_validation

logger = logging.getLogger(__name__)


def create_themed_date_entry(
    parent,
    textvariable,
    theme_colors,
    font,
    width=None,
    placeholder=None,
    date_pattern="yyyy-mm-dd",
    **kwargs,
):
    validate_func = kwargs.pop("validate_func", None)

    try:
        from tkcalendar import DateEntry
    except ModuleNotFoundError:
        return create_entry_with_validation(
            parent,
            textvariable,
            theme_colors,
            font,
            validate_func=validate_func,
            width=width,
            placeholder=placeholder,
        )

    options = {
        "textvariable": textvariable,
        "date_pattern": date_pattern,
        "font": font,
        "background": theme_colors["field_bg"],
        "foreground": theme_colors["text"],
        "bordercolor": theme_colors["field_border"],
        "headersbackground": theme_colors["field_bg"],
        "headersforeground": theme_colors["text"],
        "selectbackground": theme_colors.get("accent_blue", theme_colors["field_border"]),
        "selectforeground": theme_colors["text"],
    }
    if width is not None:
        options["width"] = width
    options.update(kwargs)

    date_entry = DateEntry(parent, **options)

    try:
        date_entry.configure(
            background=theme_colors["field_bg"],
            foreground=theme_colors["text"],
            bordercolor=theme_colors["field_border"],
            insertbackground=theme_colors["text"],
        )
    except (AttributeError, tk.TclError) as exc:
        logger.debug("DateEntry configure skipped", exc_info=exc)

    date_entry.placeholder_active = False
    if placeholder and not textvariable.get():
        date_entry.placeholder_active = True
        textvariable.set(placeholder)

    if validate_func:
        textvariable.trace_add("write", validate_func)

    return date_entry
