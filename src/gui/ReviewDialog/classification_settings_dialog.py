# src/gui/ReviewDialog/classification_settings_dialog.py
"""
Classification Settings Dialog - Expandable row-based interface.
Each classification/tag is a self-contained expandable row.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging

from gui.widgets.modern_button import ModernButton
from gui.widgets.classification_item_row import ClassificationItemRow
from gui.ReviewDialog.image_classification_and_tag_manager import (
    ImageClassificationAndTagManager,
    ItemDefinition,
)


class ClassificationSettingsDialog:
    """Dialog for managing classification and tag definitions"""

    def __init__(self, parent, item_manager, gui_manager):
        """
        Initialize settings dialog with expandable rows

        Args:
            parent: Parent window
            item_manager: ImageClassificationAndTagManager instance
            gui_manager: GUIManager for theming
        """
        self.parent = parent
        self.item_manager = item_manager
        self.gui_manager = gui_manager
        self.logger = logging.getLogger(__name__)

        self.dialog = None
        self.changes_made = False

        # Row tracking
        self.classification_rows = {}  # {item_id: ClassificationItemRow}
        self.tag_rows = {}  # {item_id: ClassificationItemRow}
        self.selected_row_id = None  # Currently selected for move operations
        self.selected_section = None  # "classification" or "tag"

        print(f"DEBUG: [ClassificationSettings] Dialog initializing")
        print(
            f"DEBUG: [ClassificationSettings] Item manager has {len(self.item_manager.items)} total items"
        )

        classifications = self.item_manager.get_all_classifications()
        tags = self.item_manager.get_all_tags()
        print(
            f"DEBUG: [ClassificationSettings] - {len(classifications)} classifications"
        )
        print(f"DEBUG: [ClassificationSettings] - {len(tags)} tags")

    def show(self) -> bool:
        """
        Show the dialog

        Returns:
            True if changes were made, False otherwise
        """
        print(f"DEBUG: [ClassificationSettings] Opening dialog")
        self._create_dialog()
        self.dialog.wait_window()
        print(
            f"DEBUG: [ClassificationSettings] Dialog closed, changes_made={self.changes_made}"
        )
        return self.changes_made

    def _create_dialog(self):
        """Create the dialog window"""
        print(f"DEBUG: [ClassificationSettings] Creating dialog window")

        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("Classification Settings")
        self.dialog.transient(self.parent)
        # Don't use grab_set() - it prevents dropdown widgets from working properly
        # self.dialog.grab_set()

        # Size and position
        width = 800
        height = 700
        x = (self.dialog.winfo_screenwidth() - width) // 2
        y = (self.dialog.winfo_screenheight() - height) // 2
        self.dialog.geometry(f"{width}x{height}+{x}+{y}")

        # Apply theme
        theme = self.gui_manager.theme_colors
        self.dialog.configure(bg=theme["background"])
        self.gui_manager.configure_ttk_styles(self.dialog)

        print(f"DEBUG: [ClassificationSettings] Theme applied")

        # Main container
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Manage Classifications & Tags",
            font=("Arial", 14, "bold"),
        )
        title_label.pack(pady=(0, 5))

        # Subtitle
        subtitle_label = ttk.Label(
            main_frame,
            text="Classifications are mutually exclusive, tags are additive.",
            font=("Arial", 9, "italic"),
        )
        subtitle_label.pack(pady=(0, 10))

        # Content area - scrollable
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Create scrollable canvas
        self.canvas = tk.Canvas(
            content_frame, bg=theme["background"], highlightthickness=0
        )
        scrollbar = ttk.Scrollbar(
            content_frame, orient="vertical", command=self.canvas.yview
        )
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Enable mousewheel scrolling - bind to canvas, not all widgets
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        # Also bind to scrollable frame for when mouse is over content
        self.scrollable_frame.bind("<MouseWheel>", self._on_mousewheel)

        # Create sections
        self._create_classifications_section()
        self._create_tags_section()

        # Bottom buttons
        self._create_button_panel(main_frame)

        # Bind close
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_close)

        print(f"DEBUG: [ClassificationSettings] Dialog creation complete")

    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _create_classifications_section(self):
        """Create the classifications section with rows"""
        print(f"DEBUG: [ClassificationSettings] Creating classifications section")

        # Section frame
        section_frame = ttk.LabelFrame(
            self.scrollable_frame, text="Classifications", padding=10
        )
        section_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Help text
        help_label = ttk.Label(
            section_frame,
            text="Click a row to select it for reordering. Click [▼] to expand and edit.",
            font=("Arial", 9, "italic"),
        )
        help_label.pack(pady=(0, 10))

        # Container for rows
        self.classifications_container = ttk.Frame(section_frame)
        self.classifications_container.pack(fill=tk.BOTH, expand=True)

        # Load classification rows
        classifications = self.item_manager.get_all_classifications()
        print(
            f"DEBUG: [ClassificationSettings] Loading {len(classifications)} classification rows"
        )

        for item in classifications:
            self._add_classification_row(item)

        # Action buttons
        button_frame = ttk.Frame(section_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))

        self.class_move_up_btn = ModernButton(
            button_frame,
            text="↑ Move Up",
            color="#6c757d",
            command=lambda: self._move_item("classification", "up"),
            theme_colors=self.gui_manager.theme_colors,
        )
        self.class_move_up_btn.pack(side=tk.LEFT, padx=2)

        self.class_move_down_btn = ModernButton(
            button_frame,
            text="↓ Move Down",
            color="#6c757d",
            command=lambda: self._move_item("classification", "down"),
            theme_colors=self.gui_manager.theme_colors,
        )
        self.class_move_down_btn.pack(side=tk.LEFT, padx=2)

        ModernButton(
            button_frame,
            text="➕ Add Classification",
            color="#28a745",
            command=self._add_new_classification,
            theme_colors=self.gui_manager.theme_colors,
        ).pack(side=tk.LEFT, padx=2)

        print(f"DEBUG: [ClassificationSettings] Classifications section complete")

    def _create_tags_section(self):
        """Create the tags section with rows"""
        print(f"DEBUG: [ClassificationSettings] Creating tags section")

        # Section frame
        section_frame = ttk.LabelFrame(self.scrollable_frame, text="Tags", padding=10)
        section_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Help text
        help_label = ttk.Label(
            section_frame,
            text="Click a row to select it for reordering. Click [▼] to expand and edit.",
            font=("Arial", 9, "italic"),
        )
        help_label.pack(pady=(0, 10))

        # Container for rows
        self.tags_container = ttk.Frame(section_frame)
        self.tags_container.pack(fill=tk.BOTH, expand=True)

        # Load tag rows
        tags = self.item_manager.get_all_tags()
        print(f"DEBUG: [ClassificationSettings] Loading {len(tags)} tag rows")

        for item in tags:
            self._add_tag_row(item)

        # Action buttons
        button_frame = ttk.Frame(section_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))

        self.tag_move_up_btn = ModernButton(
            button_frame,
            text="↑ Move Up",
            color="#6c757d",
            command=lambda: self._move_item("tag", "up"),
            theme_colors=self.gui_manager.theme_colors,
        )
        self.tag_move_up_btn.pack(side=tk.LEFT, padx=2)

        self.tag_move_down_btn = ModernButton(
            button_frame,
            text="↓ Move Down",
            color="#6c757d",
            command=lambda: self._move_item("tag", "down"),
            theme_colors=self.gui_manager.theme_colors,
        )
        self.tag_move_down_btn.pack(side=tk.LEFT, padx=2)

        ModernButton(
            button_frame,
            text="➕ Add Tag",
            color="#28a745",
            command=self._add_new_tag,
            theme_colors=self.gui_manager.theme_colors,
        ).pack(side=tk.LEFT, padx=2)

        print(f"DEBUG: [ClassificationSettings] Tags section complete")

    def _add_classification_row(self, item):
        """Add a classification row"""
        print(
            f"DEBUG: [ClassificationSettings] Adding classification row for '{item.id}'"
        )

        callbacks = {
            "on_update": self._on_row_update,
            "on_delete": self._on_row_delete,
            "on_toggle_active": self._on_row_toggle_active,
            "on_select": lambda item_id: self._on_row_select(item_id, "classification"),
        }

        row = ClassificationItemRow(
            self.classifications_container, item, self.gui_manager, callbacks
        )

        self.classification_rows[item.id] = row

    def _add_tag_row(self, item):
        """Add a tag row"""
        print(f"DEBUG: [ClassificationSettings] Adding tag row for '{item.id}'")

        callbacks = {
            "on_update": self._on_row_update,
            "on_delete": self._on_row_delete,
            "on_toggle_active": self._on_row_toggle_active,
            "on_select": lambda item_id: self._on_row_select(item_id, "tag"),
        }

        row = ClassificationItemRow(
            self.tags_container, item, self.gui_manager, callbacks
        )

        self.tag_rows[item.id] = row

    def _on_row_select(self, item_id, section):
        """Handle row selection"""
        print(
            f"DEBUG: [ClassificationSettings] Row selected: {item_id} in {section} section"
        )

        # Deselect previous
        if self.selected_row_id:
            print(
                f"DEBUG: [ClassificationSettings] Deselecting previous: {self.selected_row_id}"
            )
            old_row = self._get_row(self.selected_row_id, self.selected_section)
            if old_row:
                old_row.set_selected(False)

        # Select new
        self.selected_row_id = item_id
        self.selected_section = section

        new_row = self._get_row(item_id, section)
        if new_row:
            new_row.set_selected(True)

        print(
            f"DEBUG: [ClassificationSettings] Selection updated: row={item_id}, section={section}"
        )

    def _get_row(self, item_id, section):
        """Get row by ID and section"""
        if section == "classification":
            return self.classification_rows.get(item_id)
        elif section == "tag":
            return self.tag_rows.get(item_id)
        return None

    def _on_row_update(self, item_id, changes):
        """Handle row update"""
        print(f"DEBUG: [ClassificationSettings] Update requested for '{item_id}'")
        print(f"DEBUG: [ClassificationSettings] Changes: {changes}")

        try:
            # Check if this is a newly created item that needs proper ID
            needs_new_id = False
            if "label" in changes:
                # Check if ID looks like a temporary one
                if (item_id.startswith("new_tag") or item_id.startswith("new_classification")) and item_id.replace("_", "").replace("new", "").replace("tag", "").replace("classification", "").isdigit() or item_id in ["new_tag", "new_classification"]:
                    # Only regenerate ID if label is not the default
                    if changes["label"] not in ["New Tag", "New Classification"]:
                        needs_new_id = True
            
            if needs_new_id:
                # Generate proper ID from label
                new_id = changes["label"].lower().replace(" ", "_").replace("-", "_")
                # Remove any special characters
                new_id = "".join(c for c in new_id if c.isalnum() or c == "_")
                
                print(f"DEBUG: [ClassificationSettings] Regenerating ID: '{item_id}' → '{new_id}'")
                
                # Check if new ID already exists
                if new_id != item_id and self.item_manager.get_item(new_id):
                    messagebox.showerror(
                        "Duplicate ID",
                        f"An item with ID '{new_id}' already exists. Please choose a different label."
                    )
                    return
                
                # Get the old item
                old_item = self.item_manager.get_item(item_id)
                if old_item:
                    # Delete old item
                    self.item_manager.delete_item(item_id)
                    
                    # Create new item with proper ID
                    self.item_manager.add_item(
                        item_id=new_id,
                        label=changes.get("label", old_item.label),
                        color=changes.get("color", old_item.color),
                        item_type=old_item.item_type,
                        keybinding=changes.get("keybinding", old_item.keybinding),
                        icon=changes.get("icon", old_item.icon),
                    )
                    
                    # Rebuild the section to reflect new ID
                    section = "classification" if old_item.item_type == "classification" else "tag"
                    self._rebuild_section(section)
                    
                    self.changes_made = True
                    print(f"DEBUG: [ClassificationSettings] Item renamed from '{item_id}' to '{new_id}'")
                    messagebox.showinfo("Success", f"Created '{changes['label']}' with ID '{new_id}'")
                    return
            
            # Normal update
            self.item_manager.update_item(item_id, **changes)
            print(f"DEBUG: [ClassificationSettings] Item updated successfully")

            # Get updated item
            updated_item = self.item_manager.get_item(item_id)

            # Update row display
            row = self._get_row(item_id, updated_item.item_type)
            if row:
                row.update_from_item(updated_item)
                print(f"DEBUG: [ClassificationSettings] Row display updated")

            self.changes_made = True
            messagebox.showinfo("Success", f"Updated '{updated_item.label}'")

        except Exception as e:
            print(f"DEBUG: [ClassificationSettings] Update failed: {e}")
            self.logger.error(f"Error updating item: {e}")
            messagebox.showerror("Error", f"Failed to update: {str(e)}")

    def _on_row_delete(self, item_id):
        """Handle row deletion"""
        print(f"DEBUG: [ClassificationSettings] Delete requested for '{item_id}'")

        item = self.item_manager.get_item(item_id)
        if not item:
            print(f"DEBUG: [ClassificationSettings] Item not found: {item_id}")
            return

        # Confirm
        result = messagebox.askyesno(
            "Confirm Delete",
            f"Delete '{item.label}'?\n\nThis cannot be undone.",
            icon="warning",
        )

        if not result:
            print(f"DEBUG: [ClassificationSettings] Delete cancelled by user")
            return

        try:
            # Delete via manager
            self.item_manager.delete_item(item_id)
            print(f"DEBUG: [ClassificationSettings] Item deleted from manager")

            # Remove row
            row = self._get_row(item_id, item.item_type)
            if row:
                row.destroy()

                if item.item_type == "classification":
                    del self.classification_rows[item_id]
                elif item.item_type == "tag":
                    del self.tag_rows[item_id]

                print(f"DEBUG: [ClassificationSettings] Row removed from UI")

            # Clear selection if this was selected
            if self.selected_row_id == item_id:
                self.selected_row_id = None
                self.selected_section = None

            self.changes_made = True
            messagebox.showinfo("Success", f"Deleted '{item.label}'")

        except Exception as e:
            print(f"DEBUG: [ClassificationSettings] Delete failed: {e}")
            self.logger.error(f"Error deleting item: {e}")
            messagebox.showerror("Error", f"Failed to delete: {str(e)}")

    def _on_row_toggle_active(self, item_id, is_active):
        """Handle active state toggle"""
        print(
            f"DEBUG: [ClassificationSettings] Active toggle for '{item_id}': {is_active}"
        )

        try:
            self.item_manager.update_item(item_id, is_active=is_active)
            print(f"DEBUG: [ClassificationSettings] Active state updated in manager")
            self.changes_made = True

        except Exception as e:
            print(f"DEBUG: [ClassificationSettings] Toggle active failed: {e}")
            self.logger.error(f"Error toggling active: {e}")

    def _move_item(self, section, direction):
        """Move selected item up or down"""
        if not self.selected_row_id or self.selected_section != section:
            print(
                f"DEBUG: [ClassificationSettings] Move {direction} - no item selected in {section} section"
            )
            return

        print(
            f"DEBUG: [ClassificationSettings] Move {direction} for '{self.selected_row_id}' in {section}"
        )

        # Get current order
        if section == "classification":
            items = self.item_manager.get_all_classifications()
        else:
            items = self.item_manager.get_all_tags()

        print(f"DEBUG: [ClassificationSettings] Current order: {[i.id for i in items]}")

        # Find current index
        try:
            current_idx = next(
                i for i, item in enumerate(items) if item.id == self.selected_row_id
            )
        except StopIteration:
            print(f"DEBUG: [ClassificationSettings] Item not found in list")
            return

        # Calculate new index
        if direction == "up":
            new_idx = max(0, current_idx - 1)
        else:  # down
            new_idx = min(len(items) - 1, current_idx + 1)

        if new_idx == current_idx:
            print(
                f"DEBUG: [ClassificationSettings] Already at boundary, cannot move {direction}"
            )
            return

        # Swap
        items[current_idx], items[new_idx] = items[new_idx], items[current_idx]
        new_order = [item.id for item in items]

        print(f"DEBUG: [ClassificationSettings] New order: {new_order}")

        # Update manager
        self.item_manager.reorder_items(new_order, section)

        # Rebuild UI
        self._rebuild_section(section)

        self.changes_made = True

    def _rebuild_section(self, section):
        """Rebuild a section's rows in the new order"""
        print(f"DEBUG: [ClassificationSettings] Rebuilding {section} section")

        if section == "classification":
            # Destroy all rows
            for row in self.classification_rows.values():
                row.destroy()
            self.classification_rows.clear()

            # Recreate in new order
            items = self.item_manager.get_all_classifications()
            for item in items:
                self._add_classification_row(item)

        elif section == "tag":
            # Destroy all rows
            for row in self.tag_rows.values():
                row.destroy()
            self.tag_rows.clear()

            # Recreate in new order
            items = self.item_manager.get_all_tags()
            for item in items:
                self._add_tag_row(item)

        # Reselect the moved item
        if self.selected_row_id:
            row = self._get_row(self.selected_row_id, section)
            if row:
                row.set_selected(True)

        print(f"DEBUG: [ClassificationSettings] Section rebuilt")

    def _add_new_classification(self):
        """Add a new classification"""
        print(f"DEBUG: [ClassificationSettings] Add new classification clicked")

        # Generate unique ID
        base_id = "new_classification"
        item_id = base_id
        counter = 1
        while self.item_manager.get_item(item_id):
            item_id = f"{base_id}_{counter}"
            counter += 1

        print(f"DEBUG: [ClassificationSettings] Generated ID: {item_id}")

        try:
            # Add to manager
            new_item = self.item_manager.add_item(
                item_id=item_id,
                label="New Classification",
                color="#4CAF50",
                item_type="classification",
                keybinding=None,
            )

            print(
                f"DEBUG: [ClassificationSettings] New item added to manager: {new_item.id}"
            )

            # Add row
            self._add_classification_row(new_item)

            # Select and expand the new row
            self._on_row_select(new_item.id, "classification")
            row = self.classification_rows.get(new_item.id)
            if row:
                row.toggle_expand()

            self.changes_made = True
            print(
                f"DEBUG: [ClassificationSettings] New classification row added and expanded"
            )

        except Exception as e:
            print(f"DEBUG: [ClassificationSettings] Add new classification failed: {e}")
            self.logger.error(f"Error adding classification: {e}")
            messagebox.showerror("Error", f"Failed to add: {str(e)}")

    def _add_new_tag(self):
        """Add a new tag"""
        print(f"DEBUG: [ClassificationSettings] Add new tag clicked")

        # Generate unique ID
        base_id = "new_tag"
        item_id = base_id
        counter = 1
        while self.item_manager.get_item(item_id):
            item_id = f"{base_id}_{counter}"
            counter += 1

        print(f"DEBUG: [ClassificationSettings] Generated ID: {item_id}")

        try:
            # Add to manager
            new_item = self.item_manager.add_item(
                item_id=item_id,
                label="New Tag",
                color="#FFD700",
                item_type="tag",
                keybinding=None,
                icon="🏷️",
            )

            print(
                f"DEBUG: [ClassificationSettings] New item added to manager: {new_item.id}"
            )

            # Add row
            self._add_tag_row(new_item)

            # Select and expand the new row
            self._on_row_select(new_item.id, "tag")
            row = self.tag_rows.get(new_item.id)
            if row:
                row.toggle_expand()

            self.changes_made = True
            print(f"DEBUG: [ClassificationSettings] New tag row added and expanded")

        except Exception as e:
            print(f"DEBUG: [ClassificationSettings] Add new tag failed: {e}")
            self.logger.error(f"Error adding tag: {e}")
            messagebox.showerror("Error", f"Failed to add: {str(e)}")

    def _create_button_panel(self, parent):
        """Create the bottom button panel"""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(10, 0))

        ModernButton(
            button_frame,
            text="Cancel",
            color="#6c757d",
            command=self._on_cancel,
            theme_colors=self.gui_manager.theme_colors,
        ).pack(side=tk.RIGHT, padx=2)

        ModernButton(
            button_frame,
            text="Done",
            color="#28a745",
            command=self._on_done,
            theme_colors=self.gui_manager.theme_colors,
        ).pack(side=tk.RIGHT, padx=2)

    def _cleanup(self):
        """Clean up resources before closing"""
        # Unbind mousewheel to prevent errors after destroy
        try:
            if hasattr(self, 'canvas'):
                self.canvas.unbind("<MouseWheel>")
            if hasattr(self, 'scrollable_frame'):
                self.scrollable_frame.unbind("<MouseWheel>")
        except Exception as e:
            # Silently ignore cleanup errors
            pass

    def _on_cancel(self):
        """Handle cancel"""
        print(f"DEBUG: [ClassificationSettings] Cancel clicked")
        self.changes_made = False
        self._cleanup()
        self.dialog.destroy()

    def _on_done(self):
        """Handle done"""
        print(f"DEBUG: [ClassificationSettings] Done clicked")
        self._cleanup()
        self.dialog.destroy()

    def _on_close(self):
        """Handle window close"""
        print(f"DEBUG: [ClassificationSettings] Window close requested")
        self._on_done()
