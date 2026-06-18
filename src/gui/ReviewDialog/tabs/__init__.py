"""
Phase 3 Tabs and Phase 2 Components for Review Dialog.

This package contains both:
- Phase 2 UI Components (reusable widgets)
- Phase 3 Tab Classes (tab containers that use Phase 2 components)

Phase 3 Tabs (Container classes):
- base_review_tab.py - Abstract base class for all tabs
- hole_by_hole_tab.py - Hole-by-hole review with navigation
- all_images_tab.py - All images view with sorting
- peer_review_tab.py - Peer review workflow

Phase 2 Components (UI Widgets - already exist):
- filter_panel.py - Filter configuration panel
- classification_toolbar.py - Classification buttons
- review_grid_canvas.py - Image grid display
- statistics_display.py - Stats summary widget
- lazy_image_grid.py - Lazy-loading grid helper
"""

__version__ = "3.0.0"

# Make tab classes easily importable
from .base_review_tab import BaseReviewTab
from .hole_by_hole_tab import HoleByHoleTab
from .all_images_tab import AllImagesTab
from .peer_review_tab import PeerReviewTab

__all__ = [
    'BaseReviewTab',
    'HoleByHoleTab',
    'AllImagesTab',
    'PeerReviewTab'
]
