# gui\__init__.py

from gui.dialog_helper import DialogHelper
from gui.duplicate_handler import DuplicateHandler
from gui.gui_manager import GUIManager
from gui.qaqc_manager import QAQCManager
from gui.first_run_dialog import FirstRunDialog
from gui.compartment_registration_dialog import CompartmentRegistrationDialog
from gui.main_gui import MainGUI
from gui.widgets.collapsible_frame import CollapsibleFrame
from gui.widgets.custom_checkbox import create_custom_checkbox
from gui.widgets.entry_with_validation import create_entry_with_validation
from gui.widgets.field_with_label import create_field_with_label
from gui.widgets.modern_button import ModernButton
from gui.widgets.text_display import create_text_display
from gui.widgets.themed_combobox import create_themed_combobox
from gui.drillhole_trace_designer import DrillholeTraceDesigner
from gui.logging_review_dialog import LoggingReviewDialog
from gui.progress_dialog import ProgressDialog
from gui.widgets.modern_notebook import ModernNotebook


# Optional: Define __all__ to control what gets imported with from gui import *
__all__ = [
    'DialogHelper',
    'DuplicateHandler',
    'GUIManager',
    'QAQCManager',
    'CompartmentRegistrationDialog',
    'MainGUI',
    'CollapsibleFrame',
    'create_custom_checkbox',
    'create_entry_with_validation',
    'create_field_with_label',
    'ModernButton',
    'create_text_display',
    'create_themed_combobox',
    'DrillholeTraceDesigner',
    'LoggingReviewDialog',
    'ProgressDialog',
    'ModernNotebook',
    'FirstRunDialog'
]