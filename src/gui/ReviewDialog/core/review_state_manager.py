"""
Review State Manager - Manages undo/redo and selection state.

This module provides state management for user actions without UI coupling.
"""

import logging
from typing import List, Dict, Set, Optional, Any
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime


logger = logging.getLogger(__name__)


@dataclass
class UndoAction:
    """Represents an action that can be undone"""

    action_type: str  # "classify", "comment", "bulk_edit"
    affected_images: List[tuple]  # List of (index, filename) tuples
    old_states: List[Dict[str, Any]]  # Previous state for each image
    new_states: List[Dict[str, Any]]  # New state for each image
    timestamp: datetime = field(default_factory=datetime.now)

    def get_description(self) -> str:
        """Get human-readable description of the action"""
        count = len(self.affected_images)

        if self.action_type == "classify":
            if self.new_states and "classification" in self.new_states[0]:
                classification = self.new_states[0]["classification"]
                return f"Classified {count} image(s) as {classification}"
            return f"Changed classification of {count} image(s)"

        elif self.action_type == "comment":
            if self.new_states and "comments" in self.new_states[0]:
                comment = self.new_states[0]["comments"]
                preview = comment[:30] + "..." if len(comment) > 30 else comment
                return f"Added comment to {count} image(s): {preview}"
            return f"Changed comments on {count} image(s)"

        return f"{self.action_type} on {count} image(s)"


class ReviewStateManager:
    """
    Manages review state including undo/redo and selection.

    This class is UI-agnostic - it tracks state changes without
    knowing about the UI implementation.
    """

    MAX_UNDO_STACK = 50

    def __init__(self):
        """Initialize state manager"""
        self.logger = logging.getLogger(__name__)

        # Undo/Redo stacks
        self.undo_stack: deque = deque(maxlen=self.MAX_UNDO_STACK)
        self.redo_stack: deque = deque(maxlen=self.MAX_UNDO_STACK)

        # Selection state
        self.selected_indices: Set[int] = set()
        self.last_selected_index: Optional[int] = None

        # Save state tracking
        self.last_save_state: Dict[str, str] = {}  # filename -> classification

    # ========================================================================
    # UNDO/REDO OPERATIONS
    # ========================================================================

    def push_action(self, action: UndoAction):
        """
        Push an action onto the undo stack.

        Args:
            action: UndoAction to push
        """
        self.undo_stack.append(action)

        # Clear redo stack when new action is pushed
        self.redo_stack.clear()

        self.logger.debug(f"Pushed action: {action.get_description()}")

    def can_undo(self) -> bool:
        """Check if undo is available"""
        return len(self.undo_stack) > 0

    def can_redo(self) -> bool:
        """Check if redo is available"""
        return len(self.redo_stack) > 0

    def undo(self) -> Optional[UndoAction]:
        """
        Pop the last action from undo stack.

        Returns:
            UndoAction to restore, or None if stack is empty
        """
        if not self.can_undo():
            return None

        action = self.undo_stack.pop()
        self.redo_stack.append(action)

        self.logger.debug(f"Undoing: {action.get_description()}")
        return action

    def redo(self) -> Optional[UndoAction]:
        """
        Pop the last action from redo stack.

        Returns:
            UndoAction to reapply, or None if stack is empty
        """
        if not self.can_redo():
            return None

        action = self.redo_stack.pop()
        self.undo_stack.append(action)

        self.logger.debug(f"Redoing: {action.get_description()}")
        return action

    def clear_undo_history(self):
        """Clear both undo and redo stacks"""
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.logger.debug("Cleared undo/redo history")

    # ========================================================================
    # SELECTION MANAGEMENT
    # ========================================================================

    def set_selection(self, indices: Set[int]):
        """
        Set the current selection.

        Args:
            indices: Set of selected image indices
        """
        self.selected_indices = set(indices)

        if indices:
            self.last_selected_index = max(indices)

    def add_to_selection(self, index: int):
        """Add an index to selection"""
        self.selected_indices.add(index)
        self.last_selected_index = index

    def remove_from_selection(self, index: int):
        """Remove an index from selection"""
        self.selected_indices.discard(index)

        if index == self.last_selected_index:
            self.last_selected_index = (
                max(self.selected_indices) if self.selected_indices else None
            )

    def toggle_selection(self, index: int):
        """Toggle selection of an index"""
        if index in self.selected_indices:
            self.remove_from_selection(index)
        else:
            self.add_to_selection(index)

    def select_range(self, start_index: int, end_index: int):
        """
        Select a range of indices.

        Args:
            start_index: Starting index (inclusive)
            end_index: Ending index (inclusive)
        """
        if start_index > end_index:
            start_index, end_index = end_index, start_index

        for i in range(start_index, end_index + 1):
            self.selected_indices.add(i)

        self.last_selected_index = end_index

    def clear_selection(self):
        """Clear all selections"""
        self.selected_indices.clear()
        self.last_selected_index = None

    def select_all(self, max_index: int):
        """Select all indices up to max_index"""
        self.selected_indices = set(range(max_index))
        self.last_selected_index = max_index - 1 if max_index > 0 else None

    def get_selection(self) -> Set[int]:
        """Get current selection"""
        return self.selected_indices.copy()

    def has_selection(self) -> bool:
        """Check if any items are selected"""
        return len(self.selected_indices) > 0

    # ========================================================================
    # SAVE STATE TRACKING
    # ========================================================================

    def mark_as_saved(self, images: List):
        """
        Mark current state as saved.

        Args:
            images: List of CompartmentImage objects
        """
        self.last_save_state = {img.filename: img.classification for img in images}

        self.logger.info(f"Marked {len(images)} images as saved")

    def has_unsaved_changes(self, images: List) -> bool:
        """
        Check if there are unsaved changes.

        Args:
            images: List of CompartmentImage objects

        Returns:
            True if any images have changed since last save
        """
        for img in images:
            # Check if filename exists in last save state
            if img.filename not in self.last_save_state:
                # New image not in last save
                if img.classification and img.classification != "Unassigned":
                    return True
            else:
                # Check if classification changed
                if img.classification != self.last_save_state[img.filename]:
                    return True

        return False

    def get_unsaved_count(self, images: List) -> int:
        """
        Get count of unsaved images.

        Args:
            images: List of CompartmentImage objects

        Returns:
            Number of images with unsaved changes
        """
        count = 0

        for img in images:
            if img.filename not in self.last_save_state:
                if img.classification and img.classification != "Unassigned":
                    count += 1
            else:
                if img.classification != self.last_save_state[img.filename]:
                    count += 1

        return count

    # ========================================================================
    # ACTION HELPERS
    # ========================================================================

    def create_action(
        self, action_type: str, images: List, indices: List[int]
    ) -> UndoAction:
        """
        Create an UndoAction by capturing current state.

        Args:
            action_type: Type of action
            images: Full list of images
            indices: Indices of affected images

        Returns:
            UndoAction with current state captured
        """
        affected_images = []
        old_states = []

        for idx in indices:
            if idx < len(images):
                img = images[idx]
                affected_images.append((idx, img.filename))

                old_states.append(
                    {
                        "classification": img.classification,
                        "comments": img.comments,
                        "classified_by": img.classified_by,
                        "classified_date": img.classified_date,
                    }
                )

        return UndoAction(
            action_type=action_type,
            affected_images=affected_images,
            old_states=old_states,
            new_states=[],  # Will be filled after action is applied
        )

    def record_action(
        self,
        action_type: str,
        images: List,
        indices: List[int],
        new_states: List[Dict[str, Any]],
    ):
        """
        Record an action with both old and new states.

        Args:
            action_type: Type of action
            images: Full list of images
            indices: Indices of affected images
            new_states: New states after action
        """
        action = self.create_action(action_type, images, indices)
        action.new_states = new_states
        self.push_action(action)
