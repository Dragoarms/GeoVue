import sys
import os
import unittest
import tkinter as tk
from tkinter import ttk

from gui.dialog_helper import DialogHelper
from gui.gui_manager import GUIManager

gui_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.dirname(gui_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, gui_dir)


try:
    import logging_review_report_dialog as report_dialog
except Exception:
    report_dialog = None


class TestLoggingReviewReportDialog(unittest.TestCase):
    def setUp(self):
        self.root = None
        try:
            self.root = tk.Tk()
            self.root.withdraw()
        except tk.TclError as exc:
            self.skipTest(f"Tk not available in headless environment: {exc}")

    def tearDown(self):
        DialogHelper.set_gui_manager(None)
        if self.root:
            self.root.destroy()

    def test_module_exists(self):
        self.assertIsNotNone(report_dialog)

    def test_default_pages_defined(self):
        self.assertTrue(hasattr(report_dialog, "DEFAULT_PAGES"))
        self.assertIn("cover", report_dialog.DEFAULT_PAGES)
        self.assertIn("summary_stats", report_dialog.DEFAULT_PAGES)
        self.assertIn("outliers", report_dialog.DEFAULT_PAGES)

    def test_dialog_builds_with_theme(self):
        gui_manager = GUIManager()
        DialogHelper.set_gui_manager(gui_manager)
        data_stub = type("DataStub", (), {"is_initialized": False})()

        dialog = report_dialog.LoggingReviewReportDialog(
            self.root, gui_manager, data_stub
        )
        try:
            self.assertEqual(
                dialog.logger_listbox.cget("background"),
                gui_manager.theme_colors["field_bg"],
            )
        finally:
            dialog.dialog.destroy()

    def test_dialog_builds_with_missing_normal_font(self):
        class StubGUIManager:
            def __init__(self):
                self.theme_colors = {
                    "background": "#f5f5f5",
                    "field_bg": "#ffffff",
                    "text": "#000000",
                    "field_border": "#cccccc",
                    "accent_blue": "#1e90ff",
                    "accent_green": "#3cb371",
                    "accent_red": "#cd5c5c",
                }
                self.fonts = {"small": ("Arial", 8)}

            def configure_ttk_styles(self, root=None):
                return None

            def apply_theme(self, widget):
                return None

            def create_custom_checkbox(self, parent, text, variable):
                return tk.Checkbutton(parent, text=text, variable=variable)

            def create_entry_with_validation(self, parent, variable):
                return ttk.Entry(parent, textvariable=variable)

            def create_modern_button(self, parent, text, color, command):
                return ttk.Button(parent, text=text, command=command)

        gui_manager = StubGUIManager()
        DialogHelper.set_gui_manager(gui_manager)
        data_stub = type("DataStub", (), {"is_initialized": False})()

        dialog = report_dialog.LoggingReviewReportDialog(
            self.root, gui_manager, data_stub
        )
        try:
            self.assertIsNotNone(dialog)
        finally:
            dialog.dialog.destroy()

