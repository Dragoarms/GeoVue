"""
src\gui\ReviewDialog\__init__.py

Review Dialog Module - Modular lithology classification system.

Architecture:
- core/: Business logic and data management (NO UI)
- components/: Reusable UI components
- tabs/: Tab-specific implementations
"""

from .core import (
    ReviewDataManager, # OBSOLETE
    CompartmentImage, # OBSOLETE
    ReviewFilterEngine, # OBSOLETE
    ReviewStateManager, # OBSOLETE
    UndoAction, # OBSOLETE
    PeerReviewManager, # OBSOLETE
    ReviewExportManager, # OBSOLETE
)
from .image_classification_and_tag_manager import (
    ImageClassificationAndTagManager,
    ItemDefinition,
)
from .classification_settings_dialog import ClassificationSettingsDialog

__all__ = [
    "ReviewDataManager",
    "CompartmentImage",
    "ReviewFilterEngine",
    "ReviewStateManager",
    "UndoAction",
    "PeerReviewManager",
    "ReviewExportManager",
    "ImageClassificationAndTagManager",
    "ItemDefinition",
    "ClassificationSettingsDialog",
]
