# gui/widgets/themed_searchable_optionmenu.py

import tkinter as tk
from tkinter import ttk
from typing import List, Callable, Optional, Union
import weakref


class ThemedSearchableOptionMenu(tk.Frame):
    """
    A themed, searchable dropdown widget for data values.

    - Uses GUIManager for theme colors and fonts
    - Items are treated as data (no translation)
    - Filters the item list as the user types
    - Shows a themed dropdown Toplevel for suggestions
    """

    def __init__(
        self,
        parent,
        gui_manager,
        items: List[str],
        variable: Optional[tk.StringVar] = None,
        width: int = 20,
        placeholder: Optional[str] = None,
        on_change: Optional[Callable[[str], None]] = None,
        dropdown_mode: str = "toplevel",
        max_dropdown_height: int = 8,
        manage_parent_keys: Optional[List[str]] = None,
        *args,
        **kwargs,
    ):
        """
        Args:
            parent: parent widget
            gui_manager: GUIManager instance for theming
            items: list of data values to show
            variable: optional StringVar to bind to the selected value
            width: width of the entry field
            placeholder: optional placeholder text
            on_change: optional callback called with the selected value
            dropdown_mode: "toplevel" (default) or "overlay" (in-window frame)
            max_dropdown_height: maximum number of visible items in dropdown
            manage_parent_keys: optional list of parent window key bindings to save/restore
                               (e.g., ['<Up>', '<Down>', '<Return>'] to prevent conflicts)
        """
        super().__init__(parent, *args, **kwargs)

        # Store references (use weakref for gui_manager to avoid circular refs)
        self.gui_manager_ref = weakref.ref(gui_manager)
        self.theme_colors = gui_manager.theme_colors
        self.fonts = gui_manager.fonts
        self.on_change = on_change
        self.max_dropdown_height = max_dropdown_height

        # Store items directly (no translation needed)
        self._items = items[:] if items else []

        # Variable representing the selected value
        self.variable = variable or tk.StringVar()

        # Internal variable for entry text
        self._entry_var = tk.StringVar()

        # Placeholder text
        self.placeholder = placeholder
        self._showing_placeholder = False
        self._focus_cleared = False  # Track if we've cleared on focus

        # Dropdown state
        self._dropdown_window = None
        self._listbox = None
        self._highlighted_index = -1
        self._just_opened = False  # Flag to prevent immediate close after opening
        self._pending_hide_check = None  # Track pending after() callback
        self._selection_in_progress = False  # Flag to prevent close during selection
        self._parent_toplevel = None  # Track parent toplevel for stacking/positioning
        self._overlay_frame = None  # In-window overlay dropdown frame
        self._dropdown_mode = dropdown_mode

        # Trace IDs for cleanup
        self._external_trace_id = None

        # Track if we're updating programmatically to avoid loops
        self._updating = False
        
        # Store parent window bindings for restoration (only if specified)
        self._saved_bindings = {}
        self._manage_parent_keys = manage_parent_keys if manage_parent_keys else []

        # Initial theming
        self.configure(bg=self.theme_colors["background"])

        # Main layout
        self._build_widgets(width)

        # Setup traces
        self._setup_traces()

        # Initial value
        if self.variable.get():
            self._set_display_value(self.variable.get())
        elif self.placeholder:
            self._set_placeholder()

        # Register for theme updates if possible
        self._register_with_gui_manager()

        # Cleanup on destroy
        self.bind("<Destroy>", self._on_destroy)

    def _get_gui_manager(self):
        """Get gui_manager from weakref."""
        return self.gui_manager_ref() if self.gui_manager_ref else None

    def _register_with_gui_manager(self):
        """Register this widget for theme updates."""
        gui_manager = self._get_gui_manager()
        if gui_manager and hasattr(gui_manager, "register_custom_widget"):
            gui_manager.register_custom_widget(self)
        elif gui_manager and hasattr(gui_manager, "themed_widgets"):
            if not isinstance(gui_manager.themed_widgets, dict):
                gui_manager.themed_widgets = {}
            gui_manager.themed_widgets[f"searchable_optionmenu_{id(self)}"] = self

    def _build_widgets(self, width: int):
        """Build the main widget UI."""
        # Border frame around the entry and button
        self._outer_frame = tk.Frame(
            self,
            bg=self.theme_colors["field_bg"],
            highlightbackground=self.theme_colors["field_border"],
            highlightthickness=1,
            bd=0,
        )
        self._outer_frame.pack(fill=tk.X, expand=True)

        # Entry
        self.entry = tk.Entry(
            self._outer_frame,
            textvariable=self._entry_var,
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["text"],
            insertbackground=self.theme_colors["text"],
            font=self.fonts["normal"],
            bd=0,
            width=width,
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0), pady=2)

        # Dropdown button
        self.button = tk.Label(
            self._outer_frame,
            text="▼",
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["text"],
            font=self.fonts["small"],
            width=2,
            cursor="hand2",
        )
        self.button.pack(side=tk.RIGHT, padx=3, pady=0)

        # Bind events
        self.entry.bind("<KeyRelease>", self._on_entry_key)
        self.entry.bind("<Button-1>", self._on_entry_click)
        self.entry.bind("<FocusIn>", self._on_entry_focus_in)
        self.entry.bind("<FocusOut>", self._on_entry_focus_out)
        self.entry.bind("<Up>", self._on_arrow_up)
        self.entry.bind("<Down>", self._on_arrow_down)
        self.entry.bind("<Return>", self._on_return_key)
        self.entry.bind("<Escape>", self._on_escape_key)

        self.button.bind("<Button-1>", self._on_button_click)

    def _setup_traces(self):
        """Setup variable traces."""
        # Keep entry_var in sync with variable
        if self.variable:
            self._external_trace_id = self.variable.trace_add(
                "write", self._on_external_variable_change
            )

    def _set_placeholder(self):
        """Show placeholder text."""
        if not self._updating:
            self._showing_placeholder = True
            self.entry.config(fg=self.theme_colors.get("subtext", "#888888"))
            self._updating = True
            self._entry_var.set(self.placeholder or "")
            self._updating = False

    def _clear_placeholder(self):
        """Clear placeholder text."""
        if self._showing_placeholder:
            self._showing_placeholder = False
            self.entry.config(fg=self.theme_colors["text"])
            self._updating = True
            self._entry_var.set("")
            self._updating = False

    def _set_display_value(self, value: str):
        """Set the display value."""
        self._updating = True
        self._entry_var.set(value)
        self._updating = False
        self.entry.config(fg=self.theme_colors["text"])
        self._showing_placeholder = False

    def _on_entry_click(self, event):
        """Handle entry click."""
        if self._showing_placeholder:
            self._clear_placeholder()
        self._show_dropdown()

    def _on_button_click(self, event):
        """Handle button click."""
        if self._dropdown_window and self._dropdown_window.winfo_viewable():
            self._hide_dropdown()
        else:
            if self._showing_placeholder:
                self._clear_placeholder()
            self._show_dropdown()

    def _on_entry_focus_in(self, event):
        """Handle entry focus in."""
        if self._showing_placeholder:
            self._clear_placeholder()
        
        # Clear entry on first focus if it has a value (makes selection easier)
        if not self._focus_cleared and self._entry_var.get() and not self._showing_placeholder:
            current_val = self.variable.get()
            if current_val and current_val in self._items:
                self._updating = True
                self._entry_var.set("")
                self._updating = False
                self._focus_cleared = True
                self._show_dropdown()

    def _on_entry_focus_out(self, event):
        """Handle entry focus out."""
        # Cancel any existing pending check
        if self._pending_hide_check:
            try:
                self.after_cancel(self._pending_hide_check)
            except (tk.TclError, ValueError):
                pass
        # Use after with delay to check focus after it's fully transferred
        # A small delay prevents premature closing when clicking on dropdown
        self._pending_hide_check = self.after(150, self._check_focus_and_hide)

    def _check_focus_and_hide(self):
        """Check if focus left the widget completely."""
        # If dropdown was just opened, don't close it yet
        if self._just_opened:
            return

        # If a selection is in progress, don't close
        if self._selection_in_progress:
            return

        try:
            focus_widget = self.focus_get()

            # Check if dropdown exists and is being interacted with
            if self._dropdown_window and self._dropdown_window.winfo_exists():
                # Check if mouse is over the dropdown window
                try:
                    mouse_x = self._dropdown_window.winfo_pointerx()
                    mouse_y = self._dropdown_window.winfo_pointery()
                    
                    win_x = self._dropdown_window.winfo_rootx()
                    win_y = self._dropdown_window.winfo_rooty()
                    win_w = self._dropdown_window.winfo_width()
                    win_h = self._dropdown_window.winfo_height()
                    
                    # If mouse is within dropdown bounds, don't close
                    if (win_x <= mouse_x <= win_x + win_w and 
                        win_y <= mouse_y <= win_y + win_h):
                        return
                except tk.TclError:
                    pass
            elif self._overlay_frame and self._overlay_frame.winfo_exists():
                # Check if mouse is over the overlay dropdown frame
                try:
                    mouse_x = self._overlay_frame.winfo_pointerx()
                    mouse_y = self._overlay_frame.winfo_pointery()
                    widget_at_pointer = self._overlay_frame.winfo_containing(
                        mouse_x, mouse_y
                    )
                    if widget_at_pointer and str(widget_at_pointer).startswith(
                        str(self._overlay_frame)
                    ):
                        return
                except tk.TclError:
                    pass
            
            focus_widget = self.focus_get()
            
            # In nested dialog scenarios, focus_get() may briefly return None
            # or a widget in the parent dialog. Be lenient and don't close
            # unless focus has clearly moved outside our parent toplevel.
            if not focus_widget:
                # Focus is None - don't close, let focus settle
                return
                
            # Check if focus is still within our dropdown
            if self._dropdown_window and self._dropdown_window.winfo_exists():
                if focus_widget == self._listbox:
                    return  # Keep dropdown open
                # Check if focus is on any child of dropdown window
                try:
                    focus_toplevel = focus_widget.winfo_toplevel()
                    if focus_toplevel == self._dropdown_window:
                        return  # Keep dropdown open
                except tk.TclError:
                    pass
            elif self._overlay_frame and self._overlay_frame.winfo_exists():
                try:
                    if focus_widget and str(focus_widget).startswith(
                        str(self._overlay_frame)
                    ):
                        return
                except tk.TclError:
                    pass
                    
            # Check if focus is our entry
            if focus_widget == self.entry:
                return
            
            # Check if focus is still in our widget hierarchy
            try:
                if str(focus_widget).startswith(str(self)):
                    return
            except tk.TclError:
                pass
            
            # Check if focus is still within our parent toplevel for toplevel dropdowns
            if self._dropdown_mode == "toplevel":
                try:
                    my_toplevel = self.winfo_toplevel()
                    focus_toplevel = focus_widget.winfo_toplevel()
                    if focus_toplevel == my_toplevel:
                        return
                except tk.TclError:
                    pass
                
            self._hide_dropdown()
            self._validate_and_clear_if_invalid()
        except tk.TclError:
            pass

    def _on_entry_key(self, event):
        """Handle key release in entry."""
        if event.keysym in ("Up", "Down", "Return", "Escape"):
            return  # Handled by specific bindings
            
        if self._showing_placeholder:
            self._clear_placeholder()
            
        if not self._updating:
            self._show_dropdown()
            self._update_dropdown_filter()

    def _on_arrow_up(self, event):
        """Navigate dropdown with up arrow."""
        if self._listbox and self._listbox.size() > 0:
            if self._highlighted_index > 0:
                self._highlighted_index -= 1
            else:
                self._highlighted_index = self._listbox.size() - 1
            self._update_highlight()
        return "break"

    def _on_arrow_down(self, event):
        """Navigate dropdown with down arrow."""
        if not self._dropdown_window or not self._dropdown_window.winfo_exists():
            self._show_dropdown()
        elif self._listbox and self._listbox.size() > 0:
            if self._highlighted_index < self._listbox.size() - 1:
                self._highlighted_index += 1
            else:
                self._highlighted_index = 0
            self._update_highlight()
        return "break"

    def _on_return_key(self, event):
        """Select highlighted item with Return."""
        if self._listbox and self._highlighted_index >= 0:
            self._select_index(self._highlighted_index)
        else:
            # No item highlighted - check if typed text matches an item exactly
            typed = self._entry_var.get().strip()
            if typed and typed in self._items:
                self._set_selection(typed)
            else:
                # Invalid entry - clear it
                self._hide_dropdown()
                self.clear()
        return "break"

    def _on_escape_key(self, event):
        """Hide dropdown with Escape."""
        self._hide_dropdown()
        return "break"

    def _update_highlight(self):
        """Update listbox highlighting."""
        if not self._listbox:
            return
            
        self._listbox.selection_clear(0, tk.END)
        if 0 <= self._highlighted_index < self._listbox.size():
            self._listbox.selection_set(self._highlighted_index)
            self._listbox.see(self._highlighted_index)

    def _on_external_variable_change(self, *args):
        """Handle external variable change."""
        if not self._updating:
            value = self.variable.get()
            if value:
                self._set_display_value(value)
            elif self.placeholder:
                self._set_placeholder()

    def _show_dropdown(self):
        """Show the dropdown window."""
        # Cancel any pending hide check to prevent race conditions
        if self._pending_hide_check:
            try:
                self.after_cancel(self._pending_hide_check)
                self._pending_hide_check = None
            except (tk.TclError, ValueError):
                pass

        # Create/show dropdown based on mode
        if self._dropdown_mode == "overlay":
            if not self._overlay_frame or not self._overlay_frame.winfo_exists():
                self._create_overlay_dropdown()
            if self._overlay_frame and not self._overlay_frame.winfo_ismapped():
                self._position_overlay_dropdown()
                self._overlay_frame.lift()
                self._just_opened = True
                self.after(300, self._clear_just_opened_flag)
        else:
            # Toplevel dropdown
            if not self._dropdown_window or not self._dropdown_window.winfo_exists():
                self._create_dropdown_window()

            if not self._dropdown_window.winfo_viewable():
                self._position_dropdown()
                self._dropdown_window.deiconify()

                # Set flag to prevent immediate close, clear after a longer delay
                # Use 300ms to ensure FocusOut checks (scheduled at 150ms) complete first
                self._just_opened = True
                self.after(300, self._clear_just_opened_flag)

        self._update_dropdown_filter()

    def _clear_just_opened_flag(self):
        """Clear the just-opened flag after a short delay."""
        self._just_opened = False

    def _clear_selection_flag(self):
        """Clear the selection-in-progress flag."""
        self._selection_in_progress = False

    def _on_dropdown_focus_out(self, event):
        """Handle dropdown window FocusOut with proper cancellation."""
        # Cancel any existing pending check to prevent duplicate closures
        if self._pending_hide_check:
            try:
                self.after_cancel(self._pending_hide_check)
            except (tk.TclError, ValueError):
                pass
        # Schedule a new check with delay
        self._pending_hide_check = self.after(150, self._check_focus_and_hide)

    def _create_overlay_dropdown(self):
        """Create an in-window overlay dropdown frame."""
        self._parent_toplevel = self.winfo_toplevel()

        self._overlay_frame = tk.Frame(
            self._parent_toplevel,
            bg=self.theme_colors["field_border"],
            highlightbackground=self.theme_colors["field_border"],
            highlightthickness=1,
        )

        scrollbar = tk.Scrollbar(self._overlay_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._listbox = tk.Listbox(
            self._overlay_frame,
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["text"],
            selectbackground=self.theme_colors.get("accent_blue", "#0078d4"),
            selectforeground="#ffffff",
            font=self.fonts["small"],
            height=min(self.max_dropdown_height, 8),
            yscrollcommand=scrollbar.set,
            bd=0,
            highlightthickness=0,
            exportselection=False,
        )
        self._listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self._listbox.yview)

        # Bind events
        self._listbox.bind("<<ListboxSelect>>", self._on_listbox_select)
        self._listbox.bind("<Button-1>", self._on_listbox_single_click)
        self._listbox.bind("<Double-Button-1>", self._on_listbox_double_click)
        self._listbox.bind(
            "<Return>",
            lambda e: self._select_index(self._listbox.curselection()[0])
            if self._listbox.curselection()
            else None,
        )

    def _position_overlay_dropdown(self):
        """Position overlay dropdown within the parent toplevel."""
        if not self._overlay_frame or not self._parent_toplevel:
            return

        self.update_idletasks()

        # Convert root coords to parent toplevel coords
        parent_x = self._parent_toplevel.winfo_rootx()
        parent_y = self._parent_toplevel.winfo_rooty()

        x = self.winfo_rootx() - parent_x
        y = self.winfo_rooty() - parent_y + self.winfo_height()
        width = self.winfo_width()

        item_count = min(
            self._listbox.size() if self._listbox else 8, self.max_dropdown_height
        )
        height = max(100, item_count * 22 + 4)

        # Keep within parent toplevel bounds
        parent_h = self._parent_toplevel.winfo_height()
        if y + height > parent_h - 10:
            y = self.winfo_rooty() - parent_y - height

        self._overlay_frame.place(x=x, y=y, width=width, height=height)

    def _create_dropdown_window(self):
        """Create the dropdown Toplevel window."""
        # Find the toplevel parent window to set proper stacking
        parent_toplevel = self.winfo_toplevel()
        self._parent_toplevel = parent_toplevel
        
        self._dropdown_window = tk.Toplevel(parent_toplevel)
        self._dropdown_window.withdraw()
        self._dropdown_window.overrideredirect(True)
        
        # Make transient to parent to ensure proper stacking
        if parent_toplevel != self._dropdown_window:
            self._dropdown_window.transient(parent_toplevel)
        
        # Frame for border
        frame = tk.Frame(
            self._dropdown_window,
            bg=self.theme_colors["field_border"],
            highlightbackground=self.theme_colors["field_border"],
            highlightthickness=1,
        )
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Listbox
        self._listbox = tk.Listbox(
            frame,
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["text"],
            selectbackground=self.theme_colors.get("accent_blue", "#0078d4"),
            selectforeground="#ffffff",
            font=self.fonts["small"],
            height=min(self.max_dropdown_height, 8),
            yscrollcommand=scrollbar.set,
            bd=0,
            highlightthickness=0,
            exportselection=False,
        )
        self._listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self._listbox.yview)
        
        # Bind events
        self._listbox.bind("<<ListboxSelect>>", self._on_listbox_select)
        self._listbox.bind("<Button-1>", self._on_listbox_single_click)
        self._listbox.bind("<Double-Button-1>", self._on_listbox_double_click)
        self._listbox.bind("<Return>", lambda e: self._select_index(self._listbox.curselection()[0]) if self._listbox.curselection() else None)
        
        # Close dropdown when clicking outside (use delay to prevent race with click handlers)
        # Use the same cancellation logic as entry FocusOut to prevent multiple checks
        self._dropdown_window.bind("<FocusOut>", self._on_dropdown_focus_out)

    def _position_dropdown(self):
        """Position the dropdown below the entry."""
        if not self._dropdown_window:
            return
            
        self.update_idletasks()
        
        # Get entry position
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height()
        width = self.winfo_width()
        
        # Calculate height based on items
        item_count = min(self._listbox.size() if self._listbox else 8, self.max_dropdown_height)
        height = max(100, item_count * 22 + 4)
        
        # Check if dropdown would go off screen
        screen_height = self.winfo_screenheight()
        if y + height > screen_height - 50:
            # Show above instead
            y = self.winfo_rooty() - height
            
        self._dropdown_window.geometry(f"{width}x{height}+{x}+{y}")
        
        # Lift the dropdown window to ensure it appears on top
        self._dropdown_window.lift()
        self._dropdown_window.attributes('-topmost', True)
        
        # Remove topmost after a brief moment to allow normal interaction
        self._dropdown_window.after(
            100,
            lambda: self._dropdown_window.attributes("-topmost", False)
            if self._dropdown_window.winfo_exists()
            else None,
        )

    def _hide_dropdown(self):
        """Hide the dropdown window."""
        # Clear flags and cancel pending checks
        self._just_opened = False
        if self._pending_hide_check:
            try:
                self.after_cancel(self._pending_hide_check)
                self._pending_hide_check = None
            except (tk.TclError, ValueError):
                pass

        if self._overlay_frame and self._overlay_frame.winfo_exists():
            self._overlay_frame.place_forget()
        if self._dropdown_window and self._dropdown_window.winfo_exists():
            self._dropdown_window.withdraw()
        self._highlighted_index = -1

    def _update_dropdown_filter(self):
        """Update dropdown with filtered items."""
        if not self._listbox:
            return
            
        typed = self._entry_var.get().lower() if not self._showing_placeholder else ""
        self._listbox.delete(0, tk.END)
        
        # Filter items based on what's typed
        filtered_items = []
        for item in self._items:
            if not typed or typed in item.lower():
                filtered_items.append(item)
                
        # Show all if nothing matches
        if not filtered_items:
            filtered_items = self._items[:]
            
        for item in filtered_items:
            self._listbox.insert(tk.END, item)
            
        # Update dropdown height/position
        if self._dropdown_mode == "overlay":
            self._position_overlay_dropdown()
        else:
            self._position_dropdown()
        
        # Reset highlight
        self._highlighted_index = -1

    def _on_listbox_select(self, event):
        """Handle listbox selection."""
        if not self._listbox.curselection():
            return
        self._highlighted_index = self._listbox.curselection()[0]

    def _on_listbox_double_click(self, event):
        """Handle double click on listbox item."""
        self._selection_in_progress = True
        try:
            if self._listbox.curselection():
                self._select_index(self._listbox.curselection()[0])
        finally:
            self.after(50, self._clear_selection_flag)
    
    def _on_listbox_single_click(self, event):
        """Handle single click on listbox item."""
        # Set flag to prevent FocusOut from closing dropdown during selection
        self._selection_in_progress = True
        try:
            # Get the item under cursor
            index = self._listbox.nearest(event.y)
            if 0 <= index < self._listbox.size():
                self._select_index(index)
        finally:
            # Clear flag after selection (with small delay to ensure hide check sees it)
            self.after(50, self._clear_selection_flag)

    def _select_index(self, index):
        """Select item at given index."""
        if not self._listbox or index < 0:
            return
            
        selected_value = self._listbox.get(index)
        self._set_selection(selected_value)

    def _set_selection(self, value: str):
        """Set the selected value."""
        self._updating = True
        self.variable.set(value)
        self._entry_var.set(value)
        self._updating = False
        
        self.entry.config(fg=self.theme_colors["text"])
        self._showing_placeholder = False
        self._focus_cleared = False  # Reset for next focus
        
        self._hide_dropdown()
        
        # Clear focus from entry after selection
        self.entry.selection_clear()
        self.focus_set()  # Move focus to frame (away from entry)
        
        # Fire callback
        if self.on_change:
            try:
                self.on_change(value)
            except Exception as e:
                gui_manager = self._get_gui_manager()
                if gui_manager:
                    print(f"Error in on_change callback: {e}")

    def set_items(self, items: List[str]):
        """Replace the list of available options."""
        self._items = items[:] if items else []
        if self._dropdown_window and self._dropdown_window.winfo_viewable():
            self._update_dropdown_filter()

    def update_items(self, items: List[str]):
        """Compatibility alias for code expecting update_items()."""
        self.set_items(items)

    def get(self) -> str:
        """Return the current value."""
        return self.variable.get()

    def set(self, value: str):
        """Set the current value programmatically."""
        if value not in self._items and value:
            self._items.append(value)
            
        self.variable.set(value)

    def clear(self):
        """Clear the current selection."""
        self._updating = True
        self.variable.set("")
        self._entry_var.set("")
        self._updating = False
        
        if self.placeholder:
            self._set_placeholder()

    def update_theme(self, theme_colors=None, fonts=None, gui_manager=None):
        """Update colors/fonts when theme changes."""
        if gui_manager:
            self.gui_manager_ref = weakref.ref(gui_manager)
            self.theme_colors = gui_manager.theme_colors
            self.fonts = gui_manager.fonts
        else:
            if theme_colors:
                self.theme_colors = theme_colors
            if fonts:
                self.fonts = fonts
                
        # Apply new colors
        self.configure(bg=self.theme_colors["background"])
        
        self._outer_frame.config(
            bg=self.theme_colors["field_bg"],
            highlightbackground=self.theme_colors["field_border"],
        )
        
        fg_color = (
            self.theme_colors["text"]
            if not self._showing_placeholder
            else self.theme_colors.get("subtext", "#888888")
        )
        
        self.entry.config(
            bg=self.theme_colors["field_bg"],
            fg=fg_color,
            insertbackground=self.theme_colors["text"],
            font=self.fonts["normal"],
        )
        
        self.button.config(
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["text"],
            font=self.fonts["small"],
        )
        
        # Update dropdown if it exists
        if self._dropdown_window and self._dropdown_window.winfo_exists():
            frame = self._dropdown_window.winfo_children()[0]
            frame.config(
                bg=self.theme_colors["field_border"],
                highlightbackground=self.theme_colors["field_border"],
            )
            
            if self._listbox:
                self._listbox.config(
                    bg=self.theme_colors["field_bg"],
                    fg=self.theme_colors["text"],
                    selectbackground=self.theme_colors.get("accent_blue", "#0078d4"),
                    font=self.fonts["small"],
                )
    
    def _validate_and_clear_if_invalid(self):
        """Validate entry and clear if invalid."""
        if not self._showing_placeholder:
            typed = self._entry_var.get().strip()
            # If something is typed but not in items list, clear it
            if typed and typed not in self._items:
                self.clear()
            elif not typed and self.placeholder:
                self._set_placeholder()
    
    def _save_and_unbind_parent_keys(self):
        """Save and temporarily unbind parent window key bindings that conflict with our entry."""
        # Only manage keys if specified
        if not self._manage_parent_keys:
            return
            
        try:
            # Find the toplevel window
            parent_window = self.winfo_toplevel()
            if not parent_window:
                return
            
            # Save existing bindings for specified keys
            for key in self._manage_parent_keys:
                # Get the current binding
                current_binding = parent_window.bind(key)
                if current_binding:
                    self._saved_bindings[key] = current_binding
                    # Temporarily unbind it
                    parent_window.unbind(key)
        except Exception as e:
            # Fail silently - binding management is not critical
            pass
    
    def _restore_parent_keys(self):
        """Restore parent window key bindings."""
        # Only restore if we saved any
        if not self._saved_bindings:
            return
            
        try:
            # Find the toplevel window
            parent_window = self.winfo_toplevel()
            if not parent_window:
                return
            
            # Restore saved bindings
            for key, binding in self._saved_bindings.items():
                parent_window.bind(key, binding)
            
            # Clear saved bindings
            self._saved_bindings.clear()
        except Exception as e:
            # Fail silently
            pass

    def _on_destroy(self, event):
        """Cleanup when widget is destroyed."""
        if event.widget != self:
            return
            
        # Remove variable traces
        if self._external_trace_id and self.variable:
            try:
                self.variable.trace_remove("write", self._external_trace_id)
            except:
                pass
                
        # Destroy dropdowns
        if self._overlay_frame:
            try:
                self._overlay_frame.destroy()
            except:
                pass
        if self._dropdown_window:
            try:
                self._dropdown_window.destroy()
            except:
                pass
                
        # Unregister from gui_manager
        gui_manager = self._get_gui_manager()
        if gui_manager:
            if hasattr(gui_manager, "unregister_custom_widget"):
                gui_manager.unregister_custom_widget(self)
            elif hasattr(gui_manager, "themed_widgets"):
                widget_key = f"searchable_optionmenu_{id(self)}"
                gui_manager.themed_widgets.pop(widget_key, None)
        
        # Clear references to prevent memory leaks
        self._listbox = None
        self._dropdown_window = None
        self._overlay_frame = None
        self.variable = None
        self._entry_var = None
        self.on_change = None
        self._items = []