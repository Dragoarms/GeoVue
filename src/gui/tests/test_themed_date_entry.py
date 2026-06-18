import builtins
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import gui.widgets.themed_date_entry as themed_date_entry


class TestThemedDateEntryFallback(unittest.TestCase):
    def test_falls_back_when_tkcalendar_missing(self):
        parent = object()
        textvariable = object()
        theme_colors = {
            "field_bg": "#111111",
            "text": "#eeeeee",
            "field_border": "#333333",
        }
        font = ("Arial", 10)
        placeholder = "YYYY-MM-DD"
        sentinel_entry = object()

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "tkcalendar":
                raise ModuleNotFoundError("No module named 'tkcalendar'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with patch(
                "gui.widgets.themed_date_entry.create_entry_with_validation",
                return_value=sentinel_entry,
            ) as create_entry_mock:
                result = themed_date_entry.create_themed_date_entry(
                    parent,
                    textvariable,
                    theme_colors,
                    font,
                    placeholder=placeholder,
                )

        self.assertIs(result, sentinel_entry)
        create_entry_mock.assert_called_once_with(
            parent,
            textvariable,
            theme_colors,
            font,
            validate_func=None,
            width=None,
            placeholder=placeholder,
        )


class FakeTextVariable:
    def __init__(self, value=""):
        self._value = value
        self.trace_calls = []

    def get(self):
        return self._value

    def set(self, value):
        self._value = value

    def trace_add(self, mode, callback):
        self.trace_calls.append((mode, callback))


class TestThemedDateEntryTkcalendar(unittest.TestCase):
    def test_uses_dateentry_options_and_validation(self):
        parent = object()
        textvariable = FakeTextVariable("")
        theme_colors = {
            "field_bg": "#111111",
            "text": "#eeeeee",
            "field_border": "#333333",
            "accent_blue": "#00aaff",
        }
        font = ("Arial", 10)
        placeholder = "YYYY-MM-DD"
        validate_func = MagicMock()
        date_entry_instance = MagicMock()
        date_entry_class = MagicMock(return_value=date_entry_instance)
        fake_module = types.SimpleNamespace(DateEntry=date_entry_class)

        with patch.dict(sys.modules, {"tkcalendar": fake_module}):
            result = themed_date_entry.create_themed_date_entry(
                parent,
                textvariable,
                theme_colors,
                font,
                width=18,
                placeholder=placeholder,
                validate_func=validate_func,
            )

        self.assertIs(result, date_entry_instance)
        date_entry_class.assert_called_once()
        called_parent = date_entry_class.call_args.args[0]
        called_options = date_entry_class.call_args.kwargs
        self.assertIs(called_parent, parent)
        self.assertIs(called_options["textvariable"], textvariable)
        self.assertEqual(called_options["date_pattern"], "yyyy-mm-dd")
        self.assertEqual(called_options["font"], font)
        self.assertEqual(called_options["background"], theme_colors["field_bg"])
        self.assertEqual(called_options["foreground"], theme_colors["text"])
        self.assertEqual(called_options["bordercolor"], theme_colors["field_border"])
        self.assertEqual(
            called_options["headersbackground"], theme_colors["field_bg"]
        )
        self.assertEqual(called_options["headersforeground"], theme_colors["text"])
        self.assertEqual(
            called_options["selectbackground"], theme_colors["accent_blue"]
        )
        self.assertEqual(called_options["selectforeground"], theme_colors["text"])
        self.assertEqual(called_options["width"], 18)

        self.assertEqual(textvariable.get(), placeholder)
        self.assertTrue(date_entry_instance.placeholder_active)
        self.assertEqual(textvariable.trace_calls, [("write", validate_func)])
