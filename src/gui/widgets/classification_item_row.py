# src\gui\widgets\classification_item_row.py


"""
Expandable row widget for managing classification/tag items.
Each row is self-contained with inline editing capabilities.
"""

import tkinter as tk
from tkinter import ttk, colorchooser
from gui.widgets.modern_button import ModernButton


class ClassificationItemRow:
    """
    A self-contained, expandable row for a classification or tag item.
    Header always visible, details expand/collapse on demand.
    """

    def __init__(self, parent, item, gui_manager, callbacks):
        """
        Initialize expandable row

        Args:
            parent: Parent frame
            item: ItemDefinition instance
            gui_manager: GUIManager for theming
            callbacks: Dict with {
                'on_update': func(item_id, changes),
                'on_delete': func(item_id),
                'on_toggle_active': func(item_id, is_active),
                'on_select': func(item_id)
            }
        """
        self.parent = parent
        self.item = item
        self.gui_manager = gui_manager
        self.theme = gui_manager.theme_colors
        self.callbacks = callbacks
        self.is_expanded = False
        self.is_selected = False

        # Form variables
        self.label_var = tk.StringVar(value=item.label)
        self.color_var = tk.StringVar(value=item.color)
        self.key_var = tk.StringVar(value=item.keybinding or "")
        self.icon_var = tk.StringVar(value=item.icon or "")
        self.active_var = tk.BooleanVar(value=item.is_active)

        print(
            f"DEBUG: [Row-{item.id}] Creating row: type={item.item_type}, label={item.label}"
        )
        print(
            f"DEBUG: [Row-{item.id}] - Active: {item.is_active}, Default: {item.is_default}"
        )

        self._create_widgets()

    def _create_widgets(self):
        """Create the row structure"""
        # Main container frame
        self.main_frame = ttk.Frame(self.parent)
        self.main_frame.pack(fill=tk.X, pady=2, padx=5)

        # Create header (always visible)
        self._create_header()

        # Create details section (hidden by default)
        self._create_details()

    def _create_header(self):
        """Create the always-visible header bar"""
        print(f"DEBUG: [Row-{self.item.id}] Creating header bar")

        # Header frame with border for selection highlighting
        self.header_frame = tk.Frame(
            self.main_frame,
            bg=self.theme["secondary_bg"],
            highlightthickness=2,
            highlightbackground=self.theme["border"],
            cursor="hand2",
        )
        self.header_frame.pack(fill=tk.X)

        # Left side: Active checkbox
        self.active_check = tk.Checkbutton(
            self.header_frame,
            variable=self.active_var,
            command=self._on_toggle_active,
            bg=self.theme["secondary_bg"],
            activebackground=self.theme["secondary_bg"],
            selectcolor=self.theme.get("checkbox_bg", "#4a4a4a"),
            cursor="hand2",
        )
        self.active_check.pack(side=tk.LEFT, padx=(5, 3))

        # Color square indicator
        self.color_canvas = tk.Canvas(
            self.header_frame,
            width=25,
            height=20,
            bg=self.item.color,
            highlightthickness=1,
            highlightbackground=self.theme["border"],
            cursor="hand2",
        )
        self.color_canvas.pack(side=tk.LEFT, padx=(0, 8))

        # Icon (if present)
        if self.item.icon:
            icon_label = tk.Label(
                self.header_frame,
                text=self.item.icon,
                bg=self.theme["secondary_bg"],
                fg=self.theme["text"],
                font=("Segoe UI Emoji", 12),
                cursor="hand2",
            )
            icon_label.pack(side=tk.LEFT, padx=(0, 3))
            self._bind_header_click(icon_label)

        # Label text
        label_text = self.item.label
        font_weight = "bold" if self.item.is_active else "normal"
        self.label_widget = tk.Label(
            self.header_frame,
            text=label_text,
            bg=self.theme["secondary_bg"],
            fg=self.theme["text"],
            font=("Arial", 10, font_weight),
            anchor="w",
            width=18,
            cursor="hand2",
        )
        self.label_widget.pack(side=tk.LEFT, padx=(0, 10))

        # Keybinding badge
        key_text = f"[{self.item.keybinding}]" if self.item.keybinding else "[-]"
        self.key_label = tk.Label(
            self.header_frame,
            text=key_text,
            bg=self.theme["accent_blue"],
            fg="white",
            font=("Arial", 9, "bold"),
            padx=4,
            pady=2,
            cursor="hand2",
        )
        self.key_label.pack(side=tk.LEFT, padx=(0, 8))

        # Type badge
        type_text = self.item.item_type.upper()[:4]  # "CLAS" or "TAG"
        type_color = (
            self.theme.get("accent_green", "#28a745")
            if self.item.item_type == "classification"
            else self.theme.get("accent_blue", "#007bff")
        )
        self.type_badge = tk.Label(
            self.header_frame,
            text=type_text,
            bg=type_color,
            fg="white",
            font=("Arial", 8, "bold"),
            padx=5,
            pady=2,
            cursor="hand2",
        )
        self.type_badge.pack(side=tk.LEFT, padx=(0, 8))

        # Default badge (if applicable)
        if self.item.is_default:
            self.default_badge = tk.Label(
                self.header_frame,
                text="DEFAULT",
                bg=self.theme.get("secondary_bg", "#3a3a3a"),
                fg=self.theme.get("subtext", "#aaaaaa"),
                font=("Arial", 7, "bold"),
                padx=4,
                pady=2,
                relief=tk.RIDGE,
                borderwidth=1,
                cursor="hand2",
            )
            self.default_badge.pack(side=tk.LEFT, padx=(0, 8))
            self._bind_header_click(self.default_badge)

        # Spacer to push expand button to right
        spacer = tk.Frame(self.header_frame, bg=self.theme["secondary_bg"])
        spacer.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Expand/collapse button
        expand_text = "▲" if self.is_expanded else "▼"
        self.expand_button = tk.Label(
            self.header_frame,
            text=expand_text,
            bg=self.theme["secondary_bg"],
            fg=self.theme["text"],
            font=("Arial", 12, "bold"),
            padx=10,
            pady=5,
            cursor="hand2",
        )
        self.expand_button.pack(side=tk.RIGHT, padx=5)
        self.expand_button.bind("<Button-1>", lambda e: self.toggle_expand())

        # Bind header clicks for selection (but NOT the expand button or checkbox)
        self._bind_header_click(self.header_frame)
        self._bind_header_click(self.color_canvas)
        self._bind_header_click(self.label_widget)
        self._bind_header_click(self.key_label)
        self._bind_header_click(self.type_badge)

        print(f"DEBUG: [Row-{self.item.id}] Header created successfully")

    def _bind_header_click(self, widget):
        """Bind click event to widget for row selection"""
        widget.bind("<Button-1>", lambda e: self._on_header_click())

    def _on_header_click(self):
        """Handle header click - selects the row"""
        print(f"DEBUG: [Row-{self.item.id}] Header clicked - selecting row")
        if self.callbacks.get("on_select"):
            self.callbacks["on_select"](self.item.id)

    def _create_details(self):
        """Create the expandable details section"""
        print(f"DEBUG: [Row-{self.item.id}] Creating details section")

        # Details frame (hidden by default)
        self.details_frame = tk.Frame(
            self.main_frame, bg=self.theme["background"], relief=tk.SOLID, borderwidth=1
        )
        # Don't pack yet - will be shown on expand

        # Inner padding frame
        inner_frame = tk.Frame(self.details_frame, bg=self.theme["background"])
        inner_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Use gui_manager for consistent field styling
        # Label field
        label_frame, self.label_entry = self.gui_manager.create_field_with_label(
            inner_frame,
            "Label:",
            self.label_var,
            field_type="entry",
            readonly=self.item.is_default,  # Can't change label of defaults
            width=15,
        )

        # ID display (always readonly, auto-generated)
        id_text = (
            f"{self.item.id} (auto-generated)"
            if not self.item.is_default
            else f"{self.item.id} (built-in)"
        )
        id_var = tk.StringVar(value=id_text)
        id_frame, id_entry = self.gui_manager.create_field_with_label(
            inner_frame, "ID:", id_var, field_type="entry", readonly=True, width=15
        )

        # Color picker row
        color_frame = ttk.Frame(inner_frame)
        color_frame.pack(fill=tk.X, pady=5)

        color_label = ttk.Label(color_frame, text="Color:", width=15, anchor="w")
        color_label.pack(side=tk.LEFT)

        color_entry_frame = ttk.Frame(color_frame)
        color_entry_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.color_entry = ttk.Entry(
            color_entry_frame, textvariable=self.color_var, width=10
        )
        self.color_entry.pack(side=tk.LEFT, padx=(0, 5))

        # Color preview
        self.color_preview = tk.Canvas(
            color_entry_frame,
            width=40,
            height=25,
            bg=self.item.color,
            highlightthickness=1,
            highlightbackground=self.theme["border"],
        )
        self.color_preview.pack(side=tk.LEFT, padx=(0, 5))

        # Pick color button
        pick_button = ModernButton(
            color_entry_frame,
            text="Pick Color",
            color="#6c757d",
            command=self._pick_color,
            theme_colors=self.theme,
        )
        pick_button.pack(side=tk.LEFT)

        # Update preview when color changes
        self.color_var.trace_add("write", self._update_color_preview)

        # Keybinding field
        key_frame, self.key_entry = self.gui_manager.create_field_with_label(
            inner_frame, "Keybinding:", self.key_var, field_type="entry", width=15
        )
        # Add hint text
        hint_label = ttk.Label(
            key_frame, text="(e.g., 1, F1, q)", font=("Arial", 8, "italic")
        )
        hint_label.pack(side=tk.LEFT, padx=(5, 0))

        # Icon field (only for tags)
        if self.item.item_type == "tag":
            icon_frame, self.icon_entry = self.gui_manager.create_field_with_label(
                inner_frame, "Icon:", self.icon_var, field_type="entry", width=15
            )
            icon_hint = ttk.Label(
                icon_frame, text="(emoji character)", font=("Arial", 8, "italic")
            )
            icon_hint.pack(side=tk.LEFT, padx=(5, 0))

        # Button row
        button_frame = ttk.Frame(inner_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))

        # Update button
        self.update_button = ModernButton(
            button_frame,
            text="💾 Update",
            color="#007bff",
            command=self._on_update,
            theme_colors=self.theme,
        )
        self.update_button.pack(side=tk.RIGHT, padx=2)

        # Delete button (only for custom items)
        if not self.item.is_default:
            self.delete_button = ModernButton(
                button_frame,
                text="🗑️ Delete",
                color="#dc3545",
                command=self._on_delete,
                theme_colors=self.theme,
            )
            self.delete_button.pack(side=tk.RIGHT, padx=2)

        print(f"DEBUG: [Row-{self.item.id}] Details section created")

    def toggle_expand(self):
        """Toggle expansion of details section"""
        self.is_expanded = not self.is_expanded

        print(f"DEBUG: [Row-{self.item.id}] Toggling expansion: {self.is_expanded}")

        if self.is_expanded:
            self.details_frame.pack(fill=tk.X, pady=(0, 2))
            self.expand_button.config(text="▲")
            print(f"DEBUG: [Row-{self.item.id}] Row expanded - showing details")
        else:
            self.details_frame.pack_forget()
            self.expand_button.config(text="▼")
            print(f"DEBUG: [Row-{self.item.id}] Row collapsed - hiding details")

    def set_selected(self, selected: bool):
        """Update visual selection state"""
        self.is_selected = selected

        print(f"DEBUG: [Row-{self.item.id}] Selection state changed: {selected}")

        if selected:
            # Highlight border
            self.header_frame.config(
                highlightbackground=self.theme.get("accent_blue", "#007bff"),
                highlightthickness=3,
            )
            # Slightly tint background
            highlight_bg = self.theme.get("hover_highlight", "#404040")
            self.header_frame.config(bg=highlight_bg)
            # Update all header widgets background
            for widget in [self.label_widget, self.key_label, self.type_badge]:
                if hasattr(widget, "config"):
                    widget.config(bg=highlight_bg)
            if hasattr(self, "default_badge"):
                self.default_badge.config(bg=highlight_bg)
        else:
            # Reset to normal
            self.header_frame.config(
                highlightbackground=self.theme["border"], highlightthickness=2
            )
            normal_bg = self.theme["secondary_bg"]
            self.header_frame.config(bg=normal_bg)
            for widget in [self.label_widget, self.key_label, self.type_badge]:
                if hasattr(widget, "config"):
                    widget.config(bg=normal_bg)
            if hasattr(self, "default_badge"):
                self.default_badge.config(bg=normal_bg)

    def _on_toggle_active(self):
        """Handle active checkbox toggle"""
        is_active = self.active_var.get()
        print(
            f"DEBUG: [Row-{self.item.id}] Active toggled: {self.item.is_active} → {is_active}"
        )

        # Update label font immediately
        font_weight = "bold" if is_active else "normal"
        self.label_widget.config(font=("Arial", 10, font_weight))

        # Callback to parent
        if self.callbacks.get("on_toggle_active"):
            self.callbacks["on_toggle_active"](self.item.id, is_active)

    def _update_color_preview(self, *args):
        """Update color preview when color var changes"""
        try:
            color = self.color_var.get()
            # Validate hex color
            if color.startswith("#") and len(color) in [4, 7]:
                self.color_preview.config(bg=color)
                self.color_canvas.config(bg=color)
        except:
            pass

    def _pick_color(self):
        """Open color picker dialog"""
        print(f"DEBUG: [Row-{self.item.id}] Opening color picker")

        color = colorchooser.askcolor(
            color=self.color_var.get(), title=f"Pick color for {self.item.label}"
        )

        if color[1]:  # User selected a color
            self.color_var.set(color[1])
            print(f"DEBUG: [Row-{self.item.id}] Color picked: {color[1]}")

    def _on_update(self):
        """Handle update button click"""
        print(f"DEBUG: [Row-{self.item.id}] Update button clicked")

        # Gather changes
        changes = {}

        new_label = self.label_var.get().strip()
        if new_label != self.item.label:
            changes["label"] = new_label
            print(
                f"DEBUG: [Row-{self.item.id}] Label changed: '{self.item.label}' → '{new_label}'"
            )

        new_color = self.color_var.get().strip()
        if new_color != self.item.color:
            changes["color"] = new_color
            print(
                f"DEBUG: [Row-{self.item.id}] Color changed: '{self.item.color}' → '{new_color}'"
            )

        new_key = self.key_var.get().strip()
        if new_key != (self.item.keybinding or ""):
            changes["keybinding"] = new_key if new_key else None
            print(
                f"DEBUG: [Row-{self.item.id}] Keybinding changed: '{self.item.keybinding}' → '{new_key}'"
            )

        if self.item.item_type == "tag":
            new_icon = self.icon_var.get().strip()
            if new_icon != (self.item.icon or ""):
                changes["icon"] = new_icon if new_icon else None
                print(
                    f"DEBUG: [Row-{self.item.id}] Icon changed: '{self.item.icon}' → '{new_icon}'"
                )

        if changes:
            print(
                f"DEBUG: [Row-{self.item.id}] Calling update callback with changes: {changes}"
            )
            if self.callbacks.get("on_update"):
                self.callbacks["on_update"](self.item.id, changes)
        else:
            print(f"DEBUG: [Row-{self.item.id}] No changes detected")

    def _on_delete(self):
        """Handle delete button click"""
        print(f"DEBUG: [Row-{self.item.id}] Delete button clicked")

        if self.callbacks.get("on_delete"):
            self.callbacks["on_delete"](self.item.id)

    def update_from_item(self, item):
        """Refresh display from updated item data"""
        print(f"DEBUG: [Row-{self.item.id}] Updating display from item data")

        self.item = item

        # Update header
        self.active_var.set(item.is_active)
        self.color_canvas.config(bg=item.color)

        label_text = item.label
        font_weight = "bold" if item.is_active else "normal"
        self.label_widget.config(text=label_text, font=("Arial", 10, font_weight))

        key_text = f"[{item.keybinding}]" if item.keybinding else "[-]"
        self.key_label.config(text=key_text)

        # Update form variables
        self.label_var.set(item.label)
        self.color_var.set(item.color)
        self.key_var.set(item.keybinding or "")
        if self.item.item_type == "tag":
            self.icon_var.set(item.icon or "")

        print(f"DEBUG: [Row-{self.item.id}] Display updated")

    def destroy(self):
        """Destroy the row widget"""
        print(f"DEBUG: [Row-{self.item.id}] Destroying row")
        if self.main_frame:
            self.main_frame.destroy()
