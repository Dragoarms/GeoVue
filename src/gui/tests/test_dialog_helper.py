import unittest
from unittest.mock import Mock, patch

from gui.dialog_helper import DialogHelper


class TestDialogHelperShowQuestion(unittest.TestCase):
    def test_show_question_uses_confirm_dialog(self):
        root = object()
        helper = DialogHelper(root=root)

        with patch.object(DialogHelper, "confirm_dialog", return_value=True) as confirm_mock:
            result = helper.show_question("Confirm", "Proceed?")

        confirm_mock.assert_called_once_with(
            root, "Confirm", "Proceed?", yes_text="Yes", no_text="No"
        )
        self.assertTrue(result)

    def test_show_question_requires_root(self):
        helper = DialogHelper(root=None)

        with self.assertRaises(ValueError):
            helper.show_question("Confirm", "Proceed?")
