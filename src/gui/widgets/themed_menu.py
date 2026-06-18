# gui/widgets/themed_menu.py

import tkinter as tk
from tkinter import ttk
import logging
from typing import Dict, List, Optional, Callable, Any

logger = logging.getLogger(__name__)


class ThemedMenu:
    """
    A custom menu widget that mimics tk.Menu but with full theming support.
    Appears as a popup window with styled menu items.
    """
    
    def __init__(self, parent, tearoff=0, **kwargs):
        """
        Initialize the themed menu.
        
        Args:
            parent: Parent widget (usually a menubar or another menu)
            tearoff: Ignored (for compatibility with tk.Menu)
            **kwargs: Additional arguments (ignored for compatibility)
        """
        self.parent = parent
        self.items = []  # List of menu items
        self.popup_window = None
        self.current_index = -1
        self.theme_colors = {}
        self.fonts = {}
        self.gui_manager = None
        
        # Track cascade menus
        self.cascade_menus = {}  # index -> ThemedMenu instance
        self.cascade_timers = {}  # index -> after timer ID
        
        # Get theme from parent if available
        self._inherit_theme()
    
    def _inherit_theme(self):
        """Inherit theme settings from parent or GUI manager."""
        # Try to get GUI manager from parent chain
        widget = self.parent
        while widget:
            if hasattr(widget, '_gui_manager'):
                self.gui_manager = widget._gui_manager
                break
            try:
                widget = widget.master
            except:
                break
        
        # Get theme colors and fonts
        if self.gui_manager:
            self.theme_colors = self.gui_manager.theme_colors.copy()
            self.fonts = self.gui_manager.fonts.copy()
        else:
            # Default theme if no GUI manager found
            self.theme_colors = {
                "menu_bg": "#252526",
                "menu_fg": "#e0e0e0",
                "menu_active_bg": "#3a7ca5",
                "menu_active_fg": "#ffffff",
                "field_border": "#3f3f3f",
                "background": "#1e1e1e",
                "text": "#e0e0e0",
                "separator": "#3f3f3f"
            }
            self.fonts = {
                "normal": ("Arial", 10),
                "small": ("Arial", 9)
            }
    
    def add_command(self, label="", command=None, accelerator=None, state="normal", **kwargs):
        """
        Add a command item to the menu.
        
        Args:
            label: Menu item text
            command: Callback function
            accelerator: Keyboard shortcut text (displayed only)
            state: 'normal' or 'disabled'
            **kwargs: Additional options (for compatibility)
        """
        from gui.dialog_helper import DialogHelper
        
        item = {
            "type": "command",
            "label": DialogHelper.t(label) if hasattr(DialogHelper, 't') else label,
            "command": command,
            "accelerator": accelerator,
            "state": state,
            "index": len(self.items)
        }
        self.items.append(item)
        return len(self.items) - 1
    
    def add_separator(self):
        """Add a separator line to the menu."""
        item = {
            "type": "separator",
            "index": len(self.items)
        }
        self.items.append(item)
        return len(self.items) - 1
    
    def add_cascade(self, label="", menu=None, state="normal", **kwargs):
        """
        Add a cascade (submenu) item.
        
        Args:
            label: Menu item text
            menu: ThemedMenu instance for the submenu
            state: 'normal' or 'disabled'
            **kwargs: Additional options (for compatibility)
        """
        from gui.dialog_helper import DialogHelper
        
        index = len(self.items)
        item = {
            "type": "cascade",
            "label": DialogHelper.t(label) if hasattr(DialogHelper, 't') else label,
            "menu": menu,
            "state": state,
            "index": index
        }
        self.items.append(item)
        
        # Store cascade menu reference
        if menu:
            self.cascade_menus[index] = menu
            # Pass theme to cascade menu
            menu.theme_colors = self.theme_colors.copy()
            menu.fonts = self.fonts.copy()
            menu.gui_manager = self.gui_manager
        
        return index
    
    def add_checkbutton(self, label="", variable=None, command=None, onvalue=1, offvalue=0, state="normal", **kwargs):
        """
        Add a checkbutton item to the menu.
        
        Args:
            label: Menu item text
            variable: tk.Variable to bind to
            command: Callback function
            onvalue: Value when checked
            offvalue: Value when unchecked
            state: 'normal' or 'disabled'
            **kwargs: Additional options (for compatibility)
        """
        from gui.dialog_helper import DialogHelper
        
        item = {
            "type": "checkbutton",
            "label": DialogHelper.t(label) if hasattr(DialogHelper, 't') else label,
            "variable": variable,
            "command": command,
            "onvalue": onvalue,
            "offvalue": offvalue,
            "state": state,
            "index": len(self.items)
        }
        self.items.append(item)
        return len(self.items) - 1
    
    def add_radiobutton(self, label="", variable=None, value=None, command=None, state="normal", **kwargs):
        """
        Add a radiobutton item to the menu.
        
        Args:
            label: Menu item text
            variable: tk.Variable to bind to
            value: Value when selected
            command: Callback function
            state: 'normal' or 'disabled'
            **kwargs: Additional options (for compatibility)
        """
        from gui.dialog_helper import DialogHelper
        
        item = {
            "type": "radiobutton",
            "label": DialogHelper.t(label) if hasattr(DialogHelper, 't') else label,
            "variable": variable,
            "value": value,
            "command": command,
            "state": state,
            "index": len(self.items)
        }
        self.items.append(item)
        return len(self.items) - 1
    
    def delete(self, index1, index2=None):
        """Delete menu items from index1 to index2 (inclusive)."""
        if index2 is None:
            index2 = index1
        
        # Convert string indices
        if isinstance(index1, str):
            if index1.lower() == "all" or index1 == "0":
                index1 = 0
                index2 = len(self.items) - 1
            elif index1.lower() == "end":
                index1 = len(self.items) - 1
                index2 = len(self.items) - 1
        
        # Delete items
        for i in range(index2, index1 - 1, -1):
            if 0 <= i < len(self.items):
                # Clean up cascade menu references
                if i in self.cascade_menus:
                    del self.cascade_menus[i]
                self.items.pop(i)
        
        # Reindex remaining items and cascade menus
        new_cascade_menus = {}
        for i, item in enumerate(self.items):
            item["index"] = i
            # Update cascade menu references
            old_index = None
            for old_idx, menu in self.cascade_menus.items():
                if menu == item.get("menu"):
                    old_index = old_idx
                    break
            if old_index is not None:
                new_cascade_menus[i] = self.cascade_menus[old_index]
        
        self.cascade_menus = new_cascade_menus
    
    def entryconfig(self, index, **kwargs):
        """Configure a menu item."""
        if 0 <= index < len(self.items):
            item = self.items[index]
            
            # Update item properties
            if "label" in kwargs:
                from gui.dialog_helper import DialogHelper
                item["label"] = DialogHelper.t(kwargs["label"]) if hasattr(DialogHelper, 't') else kwargs["label"]
            if "command" in kwargs:
                item["command"] = kwargs["command"]
            if "state" in kwargs:
                item["state"] = kwargs["state"]
            if "variable" in kwargs:
                item["variable"] = kwargs["variable"]
            if "value" in kwargs:
                item["value"] = kwargs["value"]
            if "onvalue" in kwargs:
                item["onvalue"] = kwargs["onvalue"]
            if "offvalue" in kwargs:
                item["offvalue"] = kwargs["offvalue"]
            if "accelerator" in kwargs:
                item["accelerator"] = kwargs["accelerator"]
            if "menu" in kwargs and item["type"] == "cascade":
                item["menu"] = kwargs["menu"]
                self.cascade_menus[index] = kwargs["menu"]
    
    # Alias for compatibility
    entryconfigure = entryconfig
    
    def post(self, x, y):
        """
        Display the menu at the specified coordinates.
        
        Args:
            x: X coordinate (screen coordinates)
            y: Y coordinate (screen coordinates)
        """
        # Close any existing popup
        self.unpost()
        
        # Create popup window
        self.popup_window = tk.Toplevel()
        self.popup_window.overrideredirect(True)
        self.popup_window.attributes("-topmost", True)
        
        # Apply theme
        self.popup_window.configure(bg=self.theme_colors.get("menu_bg", "#252526"))
        
        # Create menu frame with border
        menu_frame = tk.Frame(
            self.popup_window,
            bg=self.theme_colors.get("menu_bg", "#252526"),
            highlightbackground=self.theme_colors.get("field_border", "#3f3f3f"),
            highlightthickness=1,
            bd=0
        )
        menu_frame.pack(fill="both", expand=True)
        
        # Create menu items
        self.item_frames = []
        for i, item in enumerate(self.items):
            if item["type"] == "separator":
                # Create separator
                sep = tk.Frame(
                    menu_frame,
                    height=1,
                    bg=self.theme_colors.get("separator", "#3f3f3f")
                )
                sep.pack(fill="x", padx=5, pady=2)
                self.item_frames.append(sep)
            else:
                # Create menu item frame
                item_frame = self._create_menu_item(menu_frame, item, i)
                item_frame.pack(fill="x")
                self.item_frames.append(item_frame)
        
        # Update window size
        self.popup_window.update_idletasks()
        
        # Position the menu
        # Ensure menu stays on screen
        menu_width = self.popup_window.winfo_reqwidth()
        menu_height = self.popup_window.winfo_reqheight()
        screen_width = self.popup_window.winfo_screenwidth()
        screen_height = self.popup_window.winfo_screenheight()
        
        # Adjust position if menu would go off screen
        if x + menu_width > screen_width:
            x = screen_width - menu_width - 5
        if y + menu_height > screen_height:
            y = screen_height - menu_height - 5
        
        self.popup_window.geometry(f"+{x}+{y}")
        
        # Bind events
        self.popup_window.bind("<Leave>", self._on_leave_menu)
        self.popup_window.bind("<Button-1>", self._on_click_outside)
        self.popup_window.bind("<Escape>", lambda e: self.unpost())
        
        # Bind keyboard navigation
        self.popup_window.bind("<Up>", self._on_key_up)
        self.popup_window.bind("<Down>", self._on_key_down)
        self.popup_window.bind("<Return>", self._on_key_enter)
        self.popup_window.bind("<Right>", self._on_key_right)
        self.popup_window.bind("<Left>", self._on_key_left)
        
        # Focus the popup for keyboard events
        self.popup_window.focus_set()
        
        # Reset current selection
        self.current_index = -1
    
    def _create_menu_item(self, parent, item, index):
        """Create a single menu item widget."""
        # Item frame
        item_frame = tk.Frame(
            parent,
            bg=self.theme_colors.get("menu_bg", "#252526"),
            height=30
        )
        
        # Padding frame for consistent sizing
        padding_frame = tk.Frame(
            item_frame,
            bg=self.theme_colors.get("menu_bg", "#252526")
        )
        padding_frame.pack(fill="both", expand=True, padx=2, pady=1)
        
        if item["type"] == "command":
            # Command item with label and optional accelerator
            self._create_command_item(padding_frame, item, index)
        elif item["type"] == "cascade":
            # Cascade item with label and arrow
            self._create_cascade_item(padding_frame, item, index)
        elif item["type"] == "checkbutton":
            # Checkbutton item
            self._create_checkbutton_item(padding_frame, item, index)
        elif item["type"] == "radiobutton":
            # Radiobutton item
            self._create_radiobutton_item(padding_frame, item, index)
        
        # Store item reference
        item_frame._item = item
        item_frame._index = index
        padding_frame._item = item
        padding_frame._index = index
        
        return item_frame
    
    def _create_command_item(self, parent, item, index):
        """Create a command menu item."""
        # Main container
        container = tk.Frame(parent, bg=parent["bg"])
        container.pack(fill="both", expand=True, padx=15, pady=5)
        
        # Label
        label = tk.Label(
            container,
            text=item["label"],
            bg=parent["bg"],
            fg=self.theme_colors.get("menu_fg", "#e0e0e0") if item["state"] == "normal" else self.theme_colors.get("field_border", "#666"),
            font=self.fonts.get("normal", ("Arial", 10)),
            anchor="w"
        )
        label.pack(side="left", fill="x", expand=True)
        
        # Accelerator (if any)
        if item.get("accelerator"):
            accel_label = tk.Label(
                container,
                text=item["accelerator"],
                bg=parent["bg"],
                fg=self.theme_colors.get("field_border", "#666"),
                font=self.fonts.get("small", ("Arial", 9)),
                anchor="e"
            )
            accel_label.pack(side="right", padx=(20, 0))
        
        # Bind events if enabled
        if item["state"] == "normal":
            self._bind_item_events(parent, container, label, index)
            if item.get("accelerator"):
                self._bind_item_events(parent, container, accel_label, index)
    
    def _create_cascade_item(self, parent, item, index):
        """Create a cascade menu item."""
        # Main container
        container = tk.Frame(parent, bg=parent["bg"])
        container.pack(fill="both", expand=True, padx=15, pady=5)
        
        # Label
        label = tk.Label(
            container,
            text=item["label"],
            bg=parent["bg"],
            fg=self.theme_colors.get("menu_fg", "#e0e0e0") if item["state"] == "normal" else self.theme_colors.get("field_border", "#666"),
            font=self.fonts.get("normal", ("Arial", 10)),
            anchor="w"
        )
        label.pack(side="left", fill="x", expand=True)
        
        # Arrow indicator
        arrow_label = tk.Label(
            container,
            text="▶",
            bg=parent["bg"],
            fg=self.theme_colors.get("menu_fg", "#e0e0e0") if item["state"] == "normal" else self.theme_colors.get("field_border", "#666"),
            font=self.fonts.get("small", ("Arial", 8))
        )
        arrow_label.pack(side="right", padx=(20, 0))
        
        # Bind events if enabled
        if item["state"] == "normal":
            self._bind_item_events(parent, container, label, index)
            self._bind_item_events(parent, container, arrow_label, index)
    
    def _create_checkbutton_item(self, parent, item, index):
        """Create a checkbutton menu item."""
        # Main container
        container = tk.Frame(parent, bg=parent["bg"])
        container.pack(fill="both", expand=True, padx=15, pady=5)
        
        # Check mark (if checked)
        check_text = "✓" if item.get("variable") and item["variable"].get() == item.get("onvalue", 1) else "  "
        check_label = tk.Label(
            container,
            text=check_text,
            bg=parent["bg"],
            fg=self.theme_colors.get("menu_fg", "#e0e0e0"),
            font=self.fonts.get("normal", ("Arial", 10)),
            width=2
        )
        check_label.pack(side="left")
        
        # Label
        label = tk.Label(
            container,
            text=item["label"],
            bg=parent["bg"],
            fg=self.theme_colors.get("menu_fg", "#e0e0e0") if item["state"] == "normal" else self.theme_colors.get("field_border", "#666"),
            font=self.fonts.get("normal", ("Arial", 10)),
            anchor="w"
        )
        label.pack(side="left", fill="x", expand=True)
        
        # Store check label reference for updates
        parent._check_label = check_label
        
        # Bind events if enabled
        if item["state"] == "normal":
            self._bind_item_events(parent, container, label, index)
            self._bind_item_events(parent, container, check_label, index)
    
    def _create_radiobutton_item(self, parent, item, index):
        """Create a radiobutton menu item."""
        # Main container
        container = tk.Frame(parent, bg=parent["bg"])
        container.pack(fill="both", expand=True, padx=15, pady=5)
        
        # Radio mark (if selected)
        radio_text = "•" if item.get("variable") and item["variable"].get() == item.get("value") else "  "
        radio_label = tk.Label(
            container,
            text=radio_text,
            bg=parent["bg"],
            fg=self.theme_colors.get("menu_fg", "#e0e0e0"),
            font=self.fonts.get("normal", ("Arial", 10)),
            width=2
        )
        radio_label.pack(side="left")
        
        # Label
        label = tk.Label(
            container,
            text=item["label"],
            bg=parent["bg"],
            fg=self.theme_colors.get("menu_fg", "#e0e0e0") if item["state"] == "normal" else self.theme_colors.get("field_border", "#666"),
            font=self.fonts.get("normal", ("Arial", 10)),
            anchor="w"
        )
        label.pack(side="left", fill="x", expand=True)
        
        # Store radio label reference for updates
        parent._radio_label = radio_label
        
        # Bind events if enabled
        if item["state"] == "normal":
            self._bind_item_events(parent, container, label, index)
            self._bind_item_events(parent, container, radio_label, index)
    
    def _bind_item_events(self, parent, container, widget, index):
        """Bind mouse events to a menu item widget."""
        def on_enter(e):
            self._highlight_item(index)
        
        def on_leave(e):
            # Don't unhighlight if we're leaving to a cascade menu
            if hasattr(e, 'x_root') and hasattr(e, 'y_root'):
                widget_at_cursor = e.widget.winfo_containing(e.x_root, e.y_root)
                if widget_at_cursor and self._is_cascade_menu_widget(widget_at_cursor):
                    return
            # Only unhighlight if we're at this index
            if self.current_index == index:
                self._unhighlight_item(index)
        
        def on_click(e):
            self._execute_item(index)
        
        # Bind to all widgets
        for w in [parent, container, widget]:
            w.bind("<Enter>", on_enter)
            w.bind("<Leave>", on_leave)
            w.bind("<Button-1>", on_click)
            w.config(cursor="hand2")
    
    def _is_cascade_menu_widget(self, widget):
        """Check if a widget belongs to a cascade menu."""
        # Walk up the widget hierarchy to find a Toplevel
        w = widget
        while w:
            if isinstance(w, tk.Toplevel):
                # Check if this toplevel belongs to any of our cascade menus
                for cascade_menu in self.cascade_menus.values():
                    if hasattr(cascade_menu, 'popup_window') and cascade_menu.popup_window == w:
                        return True
                return False
            try:
                w = w.master
            except:
                break
        return False
    
    def _highlight_item(self, index):
        """Highlight a menu item."""
        # Cancel any pending cascade timers
        for timer_id in self.cascade_timers.values():
            self.popup_window.after_cancel(timer_id)
        self.cascade_timers.clear()
        
        # Unhighlight previous item
        if self.current_index >= 0 and self.current_index < len(self.item_frames):
            self._unhighlight_item(self.current_index)
        
        # Highlight new item
        self.current_index = index
        if 0 <= index < len(self.item_frames):
            item_frame = self.item_frames[index]
            item = self.items[index]
            
            if item["type"] != "separator" and item["state"] == "normal":
                # Apply highlight colors
                self._apply_highlight_to_frame(item_frame, True)
                
                # If it's a cascade, set timer to show submenu
                if item["type"] == "cascade" and item.get("menu"):
                    timer_id = self.popup_window.after(300, lambda: self._show_cascade(index))
                    self.cascade_timers[index] = timer_id
    
    def _unhighlight_item(self, index):
        """Remove highlight from a menu item."""
        if 0 <= index < len(self.item_frames):
            item_frame = self.item_frames[index]
            item = self.items[index]
            
            if item["type"] != "separator":
                # Remove highlight colors
                self._apply_highlight_to_frame(item_frame, False)
                
                # Hide cascade menu if not hovering over it
                if item["type"] == "cascade" and index in self.cascade_menus:
                    # Set a timer to hide cascade (allows time to move to cascade)
                    self.popup_window.after(100, lambda: self._check_hide_cascade(index))
    
    def _apply_highlight_to_frame(self, frame, highlight):
        """Apply or remove highlight colors from a frame and its children."""
        bg_color = self.theme_colors.get("menu_active_bg", "#3a7ca5") if highlight else self.theme_colors.get("menu_bg", "#252526")
        fg_color = self.theme_colors.get("menu_active_fg", "#ffffff") if highlight else self.theme_colors.get("menu_fg", "#e0e0e0")
        
        # Apply to frame and all children recursively
        def apply_colors(widget):
            try:
                if isinstance(widget, (tk.Frame, tk.Label)):
                    widget.config(bg=bg_color)
                    if isinstance(widget, tk.Label):
                        # Don't change color of disabled items
                        if hasattr(widget.master, '_item') and widget.master._item.get("state") != "normal":
                            widget.config(fg=self.theme_colors.get("field_border", "#666"))
                        else:
                            widget.config(fg=fg_color)
                
                # Recurse to children
                for child in widget.winfo_children():
                    apply_colors(child)
            except:
                pass
        
        apply_colors(frame)
    
    def _show_cascade(self, index):
        """Show a cascade submenu."""
        if index not in self.cascade_menus:
            return
        
        cascade_menu = self.cascade_menus[index]
        if not cascade_menu:
            return
        
        # Get position of the cascade item
        item_frame = self.item_frames[index]
        x = item_frame.winfo_rootx() + item_frame.winfo_width() - 5
        y = item_frame.winfo_rooty()
        
        # Show the cascade menu
        cascade_menu.post(x, y)
        
        # Set up bidirectional references
        cascade_menu._parent_menu = self
        cascade_menu._parent_index = index
    
    def _check_hide_cascade(self, index):
        """Check if we should hide a cascade menu."""
        if index not in self.cascade_menus:
            return
        
        cascade_menu = self.cascade_menus[index]
        if not cascade_menu or not hasattr(cascade_menu, 'popup_window') or not cascade_menu.popup_window:
            return
        
        # Check if mouse is over the cascade menu or this menu item
        x, y = self.popup_window.winfo_pointerxy()
        widget_at_cursor = self.popup_window.winfo_containing(x, y)
        
        # Don't hide if cursor is over the cascade menu or the cascade item
        if widget_at_cursor:
            if self._is_cascade_menu_widget(widget_at_cursor):
                return
            # Check if cursor is over the cascade item
            try:
                if self.current_index == index:
                    return
            except:
                pass
        
        # Hide the cascade
        cascade_menu.unpost()
    
    def _execute_item(self, index):
        """Execute a menu item's action."""
        if 0 <= index < len(self.items):
            item = self.items[index]
            
            if item["state"] != "normal":
                return
            
            if item["type"] == "command":
                # Close menu first
                self.unpost()
                # Execute command
                if item.get("command"):
                    try:
                        item["command"]()
                    except Exception as e:
                        logger.error(f"Error executing menu command: {e}")
            
            elif item["type"] == "checkbutton":
                # Toggle variable
                if item.get("variable"):
                    current = item["variable"].get()
                    new_value = item.get("offvalue", 0) if current == item.get("onvalue", 1) else item.get("onvalue", 1)
                    item["variable"].set(new_value)
                    
                    # Update display
                    if hasattr(self.item_frames[index], '_check_label'):
                        self.item_frames[index]._check_label.config(
                            text="✓" if new_value == item.get("onvalue", 1) else "  "
                        )
                
                # Execute command if any
                if item.get("command"):
                    try:
                        item["command"]()
                    except Exception as e:
                        logger.error(f"Error executing checkbutton command: {e}")
            
            elif item["type"] == "radiobutton":
                # Set variable
                if item.get("variable") and item.get("value") is not None:
                    item["variable"].set(item["value"])
                    
                    # Update all radio items with same variable
                    for i, other_item in enumerate(self.items):
                        if other_item["type"] == "radiobutton" and other_item.get("variable") == item["variable"]:
                            if hasattr(self.item_frames[i], '_radio_label'):
                                self.item_frames[i]._radio_label.config(
                                    text="•" if i == index else "  "
                                )
                
                # Close menu
                self.unpost()
                
                # Execute command if any
                if item.get("command"):
                    try:
                        item["command"]()
                    except Exception as e:
                        logger.error(f"Error executing radiobutton command: {e}")
            
            elif item["type"] == "cascade":
                # Cascade is handled by hover
                pass
    
    def _on_leave_menu(self, event):
        """Handle mouse leaving the menu."""
        # Check if we're entering a cascade menu
        x, y = event.x_root, event.y_root
        widget_at_cursor = event.widget.winfo_containing(x, y)
        
        if widget_at_cursor and self._is_cascade_menu_widget(widget_at_cursor):
            return
        
        # Close menu if mouse left the window area
        if hasattr(event, 'x') and hasattr(event, 'y'):
            if event.x < 0 or event.y < 0 or \
               event.x > self.popup_window.winfo_width() or \
               event.y > self.popup_window.winfo_height():
                # Give a small delay to allow entering cascade menus
                self.popup_window.after(500, self._check_close_menu)
    
    def _check_close_menu(self):
        """Check if we should close the menu."""
        if not self.popup_window or not self.popup_window.winfo_exists():
            return
        
        # Check if mouse is over this menu or any cascade
        x, y = self.popup_window.winfo_pointerxy()
        widget_at_cursor = self.popup_window.winfo_containing(x, y)
        
        if widget_at_cursor:
            # Check if over this menu
            if self._widget_in_window(widget_at_cursor, self.popup_window):
                return
            # Check if over any cascade menu
            if self._is_cascade_menu_widget(widget_at_cursor):
                return
        
        # Close if not over any menu
        self.unpost()
    
    def _widget_in_window(self, widget, window):
        """Check if a widget is inside a specific window."""
        w = widget
        while w:
            if w == window:
                return True
            try:
                w = w.master
            except:
                break
        return False
    
    def _on_click_outside(self, event):
        """Handle clicks outside menu items."""
        # Check if click was on a menu item
        widget = event.widget
        while widget:
            if hasattr(widget, '_item'):
                return  # Click was on a menu item
            try:
                widget = widget.master
                if widget == self.popup_window:
                    break
            except:
                break
        
        # Click was outside menu items - close menu
        self.unpost()
    
    def _on_key_up(self, event):
        """Handle up arrow key."""
        if not self.items:
            return
        
        # Find previous non-separator item
        new_index = self.current_index - 1
        while new_index >= 0 and self.items[new_index]["type"] == "separator":
            new_index -= 1
        
        if new_index < 0:
            # Wrap to bottom
            new_index = len(self.items) - 1
            while new_index >= 0 and self.items[new_index]["type"] == "separator":
                new_index -= 1
        
        if new_index >= 0:
            self._highlight_item(new_index)
    
    def _on_key_down(self, event):
        """Handle down arrow key."""
        if not self.items:
            return
        
        # Find next non-separator item
        new_index = self.current_index + 1
        while new_index < len(self.items) and self.items[new_index]["type"] == "separator":
            new_index += 1
        
        if new_index >= len(self.items):
            # Wrap to top
            new_index = 0
            while new_index < len(self.items) and self.items[new_index]["type"] == "separator":
                new_index += 1
        
        if new_index < len(self.items):
            self._highlight_item(new_index)
    
    def _on_key_enter(self, event):
        """Handle enter key."""
        if self.current_index >= 0:
            self._execute_item(self.current_index)
    
    def _on_key_right(self, event):
        """Handle right arrow key."""
        if self.current_index >= 0 and self.current_index < len(self.items):
            item = self.items[self.current_index]
            if item["type"] == "cascade" and item["state"] == "normal":
                self._show_cascade(self.current_index)
    
    def _on_key_left(self, event):
        """Handle left arrow key."""
        # If this is a cascade menu, close it and return to parent
        if hasattr(self, '_parent_menu') and hasattr(self, '_parent_index'):
            parent_menu = self._parent_menu
            parent_index = self._parent_index
            self.unpost()
            # Re-highlight the parent item
            if parent_menu.popup_window and parent_menu.popup_window.winfo_exists():
                parent_menu._highlight_item(parent_index)
                parent_menu.popup_window.focus_set()
    
    def unpost(self):
        """Hide the menu."""
        # Cancel any pending cascade timers
        if hasattr(self, 'cascade_timers'):
            for timer_id in self.cascade_timers.values():
                try:
                    self.popup_window.after_cancel(timer_id)
                except:
                    pass
            self.cascade_timers.clear()
        
        # Close any open cascade menus
        for cascade_menu in self.cascade_menus.values():
            if cascade_menu and hasattr(cascade_menu, 'unpost'):
                cascade_menu.unpost()
        
        # Destroy popup window
        if self.popup_window:
            try:
                self.popup_window.destroy()
            except:
                pass
            self.popup_window = None
        
        self.current_index = -1
        self.item_frames = []
    
    def update_theme(self, theme_colors=None, fonts=None, gui_manager=None):
        """
        Update the menu's theme.
        
        Args:
            theme_colors: New theme colors dictionary
            fonts: New fonts dictionary
            gui_manager: GUI manager instance
        """
        if gui_manager:
            self.gui_manager = gui_manager
            self.theme_colors = gui_manager.theme_colors.copy()
            self.fonts = gui_manager.fonts.copy()
        else:
            if theme_colors:
                self.theme_colors.update(theme_colors)
            if fonts:
                self.fonts.update(fonts)
        
        # Update cascade menus
        for cascade_menu in self.cascade_menus.values():
            if cascade_menu:
                cascade_menu.update_theme(self.theme_colors, self.fonts, self.gui_manager)


class ThemedMenuBar(tk.Frame):
    """
    A themed menu bar that uses ThemedMenu for dropdowns.
    Key Methods:
        Menu Creation:
            add_command() - Add a clickable item
            add_separator() - Add a divider line
            add_cascade() - Add a submenu
            add_checkbutton() - Add a toggle item
            add_radiobutton() - Add a radio option

        Menu Control:
            post(x, y) - Show menu at coordinates
            unpost() - Hide the menu
            delete(index1, index2) - Remove items
            entryconfig(index, **options) - Modify items

        Theme Updates:
            update_theme() - Update colors when theme changes
    """
    
    def __init__(self, parent, gui_manager=None, **kwargs):
        """
        Initialize the themed menu bar.
        
        Args:
            parent: Parent widget (usually root window)
            gui_manager: GUIManager instance for theming
            **kwargs: Additional frame options
        """
        # Get theme colors from gui_manager or use defaults
        if gui_manager:
            theme_colors = gui_manager.theme_colors
            fonts = gui_manager.fonts
        else:
            theme_colors = {
                "menu_bg": "#252526",
                "menu_fg": "#e0e0e0",
                "menu_active_bg": "#3a7ca5",
                "menu_active_fg": "#ffffff",
                "background": "#1e1e1e"
            }
            fonts = {"normal": ("Arial", 10)}
        
        super().__init__(parent, bg=theme_colors.get("menu_bg", "#252526"), **kwargs)
        
        self.gui_manager = gui_manager
        self.theme_colors = theme_colors
        self.fonts = fonts
        self.menus = {}  # Menu name -> ThemedMenu instance
        self.menu_buttons = {}  # Menu name -> Button widget
        
        # Store reference to gui_manager for child menus
        self._gui_manager = gui_manager
        
        # Configure the frame
        self.pack(fill="x", side="top")
    
    def add_cascade(self, label, menu):
        """
        Add a menu to the menu bar.
        
        Args:
            label: Menu label
            menu: ThemedMenu instance
        """
        from gui.dialog_helper import DialogHelper
        
        # Store menu
        self.menus[label] = menu
        
        # Pass theme to menu
        menu.update_theme(self.theme_colors, self.fonts, self.gui_manager)
        
        # Create menu button
        button = tk.Label(
            self,
            text=DialogHelper.t(label) if hasattr(DialogHelper, 't') else label,
            bg=self.theme_colors.get("menu_bg", "#252526"),
            fg=self.theme_colors.get("menu_fg", "#e0e0e0"),
            font=self.fonts.get("normal", ("Arial", 10)),
            padx=15,
            pady=5,
            cursor="hand2"
        )
        button.pack(side="left")
        
        # Store button reference
        self.menu_buttons[label] = button
        
        # Bind events
        button.bind("<Enter>", lambda e: self._on_menu_hover(label))
        button.bind("<Leave>", lambda e: self._on_menu_leave(label))
        button.bind("<Button-1>", lambda e: self._on_menu_click(label))
        
        return menu
    
    def _on_menu_hover(self, label):
        """Handle mouse hover over menu button."""
        button = self.menu_buttons[label]
        button.config(
            bg=self.theme_colors.get("menu_active_bg", "#3a7ca5"),
            fg=self.theme_colors.get("menu_active_fg", "#ffffff")
        )
    
    def _on_menu_leave(self, label):
        """Handle mouse leave from menu button."""
        button = self.menu_buttons[label]
        button.config(
            bg=self.theme_colors.get("menu_bg", "#252526"),
            fg=self.theme_colors.get("menu_fg", "#e0e0e0")
        )
    
    def _on_menu_click(self, label):
        """Handle click on menu button."""
        menu = self.menus[label]
        button = self.menu_buttons[label]
        
        # Get button position
        x = button.winfo_rootx()
        y = button.winfo_rooty() + button.winfo_height()
        
        # Show menu
        menu.post(x, y)
    
    def update_theme(self, theme_colors=None, fonts=None, gui_manager=None):
        """Update the menu bar theme."""
        if gui_manager:
            self.gui_manager = gui_manager
            self.theme_colors = gui_manager.theme_colors
            self.fonts = gui_manager.fonts
        else:
            if theme_colors:
                self.theme_colors.update(theme_colors)
            if fonts:
                self.fonts.update(fonts)
        
        # Update frame background
        self.config(bg=self.theme_colors.get("menu_bg", "#252526"))
        
        # Update menu buttons
        for label, button in self.menu_buttons.items():
            button.config(
                bg=self.theme_colors.get("menu_bg", "#252526"),
                fg=self.theme_colors.get("menu_fg", "#e0e0e0"),
                font=self.fonts.get("normal", ("Arial", 10))
            )
        
        # Update menus
        for menu in self.menus.values():
            menu.update_theme(self.theme_colors, self.fonts, self.gui_manager)