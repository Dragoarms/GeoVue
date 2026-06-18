import os
import sys
import unittest
from unittest.mock import Mock, patch

# Ensure src is on the path for gui imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, src_dir)

class TestVizColumnSettingsDialogReset(unittest.TestCase):
    def _import_dialog_modules(self):
        from gui.ReviewDialog.viz_column_settings_dialog import VizColumnSettingsDialog
        from gui.dialog_helper import DialogHelper
        return VizColumnSettingsDialog, DialogHelper

    def _make_dialog_instance(self):
        VizColumnSettingsDialog, _ = self._import_dialog_modules()
        dialog = VizColumnSettingsDialog.__new__(VizColumnSettingsDialog)
        dialog.dialog = object()
        dialog.column_rows = [Mock(), Mock()]
        dialog._load_default_config = Mock()
        dialog.logger = Mock()
        return dialog

    def test_reset_to_default_confirmed_resets_state(self):
        _, DialogHelper = self._import_dialog_modules()
        dialog = self._make_dialog_instance()
        rows = list(dialog.column_rows)

        with patch.object(DialogHelper, "confirm_dialog", return_value=True) as confirm_mock:
            dialog._reset_to_default()

        confirm_mock.assert_called_once_with(
            dialog.dialog,
            "Reset Configuration",
            "Reset all visualization columns to defaults?\n\nThis will remove your current configuration.",
            yes_text="Reset",
            no_text="Cancel",
        )
        for row in rows:
            row.destroy.assert_called_once()
        self.assertEqual(dialog.column_rows, [])
        dialog._load_default_config.assert_called_once()
        dialog.logger.info.assert_called_once_with("Reset to default configuration")

    def test_reset_to_default_cancelled_keeps_state(self):
        _, DialogHelper = self._import_dialog_modules()
        dialog = self._make_dialog_instance()
        rows = list(dialog.column_rows)

        with patch.object(DialogHelper, "confirm_dialog", return_value=False) as confirm_mock:
            dialog._reset_to_default()

        confirm_mock.assert_called_once()
        for row in rows:
            row.destroy.assert_not_called()
        self.assertEqual(dialog.column_rows, rows)
        dialog._load_default_config.assert_not_called()
        dialog.logger.info.assert_not_called()
