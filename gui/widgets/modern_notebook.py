# gui/widgets/modern_notebook.py

import tkinter as tk
from typing import Dict, List, Optional, Callable


class ModernNotebook:
    """
    A modern styled notebook widget with full control over appearance.
    Replaces ttk.Notebook to avoid theming limitations.
    """
    
    def __init__(self, parent, theme_colors: dict, fonts: dict = None):
        """
        Initialize the modern notebook.
        
        Args:
            parent: Parent widget
            theme_colors: Theme color dictionary
            fonts: Font dictionary (optional)
        """
        self.parent = parent
        self.theme_colors = theme_colors
        self.fonts = fonts or {"normal": ("Arial", 10), "bold": ("Arial", 10, "bold")}
        
        # Track tabs and pages
        self.tabs: List[Dict] = []  # List of tab info dicts
        self.pages: Dict[int, tk.Frame] = {}  # Tab index -> page frame
        self.current_tab = 0
        self.tab_buttons: Dict[int, tk.Frame] = {}  # Tab index -> button frame
        
        # Create main container
        self.container = tk.Frame(parent, bg=theme_colors["background"])
        
        # Create tab bar container
        self.tab_bar = tk.Frame(
            self.container,
            bg=theme_colors["background"],
            height=40
        )
        self.tab_bar.pack(fill=tk.X, side=tk.TOP)
        self.tab_bar.pack_propagate(False)
        
        # Create page container
        self.page_container = tk.Frame(
            self.container,
            bg=theme_colors["background"],
            highlightbackground=theme_colors["border"],
            highlightthickness=1
        )
        self.page_container.pack(fill=tk.BOTH, expand=True, pady=(0, 0))
        
        # Bind events
        self._bind_var = tk.StringVar()  # For tab change events
    
    def add(self, page_widget: tk.Frame, text: str, **kwargs):
        """
        Add a new tab and page.
        
        Args:
            page_widget: The frame/widget to display as the page content
            text: Tab text
            **kwargs: Additional options (e.g., state='disabled')
        """
        tab_index = len(self.tabs)
        
        # Store tab info
        tab_info = {
            "text": text,
            "state": kwargs.get("state", "normal"),
            "index": tab_index
        }
        self.tabs.append(tab_info)
        
        # Create tab button
        tab_button = self._create_tab_button(tab_index, text, tab_info["state"])
        self.tab_buttons[tab_index] = tab_button
        
        # Store page widget
        self.pages[tab_index] = page_widget
        
        page_widget.pack_forget()
        page_widget.grid_forget()
        page_widget.place_forget()
        
        # Show first tab by default
        if tab_index == 0:
            self._switch_to_tab(0)
        
        return tab_index
    
    def _create_tab_button(self, index: int, text: str, state: str) -> tk.Frame:
        """Create a styled tab button."""
        # Tab button container
        tab_frame = tk.Frame(
            self.tab_bar,
            bg=self.theme_colors["secondary_bg"],
            cursor="hand2" if state == "normal" else ""
        )
        tab_frame.pack(side=tk.LEFT, padx=(2, 0), pady=(5, 0))
        
        # Tab label
        tab_label = tk.Label(
            tab_frame,
            text=text,
            bg=self.theme_colors["secondary_bg"],
            fg=self.theme_colors["text"] if state == "normal" else self.theme_colors["field_border"],
            font=self.fonts["normal"],
            padx=20,
            pady=10,
            cursor="hand2" if state == "normal" else ""
        )
        tab_label.pack(fill=tk.BOTH, expand=True)
        
        # Store references
        tab_frame._label = tab_label
        tab_frame._index = index
        tab_frame._state = state
        
        # Bind click events if enabled
        if state == "normal":
            tab_frame.bind("<Button-1>", lambda e: self._on_tab_click(index))
            tab_label.bind("<Button-1>", lambda e: self._on_tab_click(index))
            
            # Hover effects
            def on_enter(e):
                if index != self.current_tab:
                    tab_frame.config(bg=self.theme_colors["hover_highlight"])
                    tab_label.config(bg=self.theme_colors["hover_highlight"])
            
            def on_leave(e):
                if index != self.current_tab:
                    tab_frame.config(bg=self.theme_colors["secondary_bg"])
                    tab_label.config(bg=self.theme_colors["secondary_bg"])
            
            tab_frame.bind("<Enter>", on_enter)
            tab_label.bind("<Enter>", on_enter)
            tab_frame.bind("<Leave>", on_leave)
            tab_label.bind("<Leave>", on_leave)
        
        return tab_frame
    
    def _on_tab_click(self, index: int):
        """Handle tab click."""
        if self.tabs[index]["state"] == "normal":
            self._switch_to_tab(index)
    
    def _switch_to_tab(self, index: int):
        """Switch to a specific tab."""
        # Hide current page
        if self.current_tab in self.pages:
            self.pages[self.current_tab].pack_forget()
        
        # Update tab appearances
        for i, tab_frame in self.tab_buttons.items():
            if i == index:
                # Selected tab
                tab_frame.config(bg=self.theme_colors["accent_blue"])
                tab_frame._label.config(
                    bg=self.theme_colors["accent_blue"],
                    fg="#ffffff"
                )
            else:
                # Unselected tabs
                bg_color = (self.theme_colors["secondary_bg"] 
                           if tab_frame._state == "normal" 
                           else self.theme_colors["field_bg"])
                fg_color = (self.theme_colors["text"] 
                           if tab_frame._state == "normal" 
                           else self.theme_colors["field_border"])
                
                tab_frame.config(bg=bg_color)
                tab_frame._label.config(bg=bg_color, fg=fg_color)
        
        # Show new page
        self.current_tab = index
        if index in self.pages:
            self.pages[index].pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Trigger change event
        self._bind_var.set(str(index))
        self.container.event_generate("<<NotebookTabChanged>>")
        
    def select(self, tab_id=None):
        """
        Select a tab by index or tab_id, or return current tab if no argument.
        
        Args:
            tab_id: Tab to select, or None to return current tab
            
        Returns:
            Current tab index if tab_id is None
        """
        # Return current tab if no argument
        if tab_id is None:
            return self.current_tab
            
        # Select the specified tab
        if isinstance(tab_id, int):
            self._switch_to_tab(tab_id)
        else:
            # Try to find by text if string passed
            for i, tab in enumerate(self.tabs):
                if tab["text"] == str(tab_id):
                    self._switch_to_tab(i)
                    break
    
    def tab(self, tab_id, option=None, **kwargs):
        """Configure or query tab options."""
        if isinstance(tab_id, str):
            # Find tab by text
            tab_index = None
            for i, tab in enumerate(self.tabs):
                if tab["text"] == tab_id:
                    tab_index = i
                    break
        else:
            tab_index = int(tab_id)
        
        if tab_index is None or tab_index >= len(self.tabs):
            return None
        
        # Query mode
        if option is None and not kwargs:
            return self.tabs[tab_index].copy()
        
        # Get specific option
        if option and not kwargs:
            return self.tabs[tab_index].get(option)
        
        # Set options
        if "state" in kwargs:
            new_state = kwargs["state"]
            self.tabs[tab_index]["state"] = new_state
            
            # Update button appearance and behavior
            tab_frame = self.tab_buttons[tab_index]
            tab_frame._state = new_state
            
            if new_state == "disabled":
                # Disable the tab
                tab_frame.config(
                    bg=self.theme_colors["field_bg"],
                    cursor=""
                )
                tab_frame._label.config(
                    fg=self.theme_colors["field_border"],
                    cursor=""
                )
                # Unbind events
                tab_frame.unbind("<Button-1>")
                tab_frame._label.unbind("<Button-1>")
                tab_frame.unbind("<Enter>")
                tab_frame._label.unbind("<Enter>")
                tab_frame.unbind("<Leave>")
                tab_frame._label.unbind("<Leave>")
            else:
                # Enable the tab
                bg_color = (self.theme_colors["accent_blue"] 
                           if tab_index == self.current_tab 
                           else self.theme_colors["secondary_bg"])
                fg_color = ("#ffffff" 
                           if tab_index == self.current_tab 
                           else self.theme_colors["text"])
                
                tab_frame.config(bg=bg_color, cursor="hand2")
                tab_frame._label.config(fg=fg_color, cursor="hand2")
                
                # Re-bind events
                tab_frame.bind("<Button-1>", lambda e: self._on_tab_click(tab_index))
                tab_frame._label.bind("<Button-1>", lambda e: self._on_tab_click(tab_index))
                
                # Re-bind hover events
                def on_enter(e):
                    if tab_index != self.current_tab:
                        tab_frame.config(bg=self.theme_colors["hover_highlight"])
                        tab_frame._label.config(bg=self.theme_colors["hover_highlight"])
                
                def on_leave(e):
                    if tab_index != self.current_tab:
                        tab_frame.config(bg=self.theme_colors["secondary_bg"])
                        tab_frame._label.config(bg=self.theme_colors["secondary_bg"])
                
                tab_frame.bind("<Enter>", on_enter)
                tab_frame._label.bind("<Enter>", on_enter)
                tab_frame.bind("<Leave>", on_leave)
                tab_frame._label.bind("<Leave>", on_leave)
        
        if "text" in kwargs:
            self.tabs[tab_index]["text"] = kwargs["text"]
            self.tab_buttons[tab_index]._label.config(text=kwargs["text"])
    
    def index(self, tab_id):
        """Get the index of a tab."""
        if tab_id == "current":
            return self.current_tab
        elif tab_id == "end":
            return len(self.tabs)
        elif isinstance(tab_id, int):
            return tab_id
        else:
            # Try to find by text
            for i, tab in enumerate(self.tabs):
                if tab["text"] == str(tab_id):
                    return i
        return None
    
    def tabs(self):
        """Return list of tab indices."""
        return list(range(len(self.tabs)))
    
    def bind(self, event, callback):
        """Bind events to the notebook."""
        self.container.bind(event, callback)
    
    def pack(self, **kwargs):
        """Pack the notebook container."""
        self.container.pack(**kwargs)
        return self
    
    def grid(self, **kwargs):
        """Grid the notebook container."""
        self.container.grid(**kwargs)
        return self
    
    def place(self, **kwargs):
        """Place the notebook container."""
        self.container.place(**kwargs)
        return self
    
    def update_theme(self, theme_colors: dict):
        """Update the notebook theme."""
        self.theme_colors = theme_colors
        
        # Update container backgrounds
        self.container.config(bg=theme_colors["background"])
        self.tab_bar.config(bg=theme_colors["background"])
        self.page_container.config(
            bg=theme_colors["background"],
            highlightbackground=theme_colors["border"]
        )
        
        # Update all tab buttons
        for i, tab_frame in self.tab_buttons.items():
            if i == self.current_tab:
                # Selected tab
                tab_frame.config(bg=theme_colors["accent_blue"])
                tab_frame._label.config(
                    bg=theme_colors["accent_blue"],
                    fg="#ffffff"
                )
            else:
                # Unselected tabs
                state = self.tabs[i]["state"]
                bg_color = (theme_colors["secondary_bg"] 
                           if state == "normal" 
                           else theme_colors["field_bg"])
                fg_color = (theme_colors["text"] 
                           if state == "normal" 
                           else theme_colors["field_border"])
                
                tab_frame.config(bg=bg_color)
                tab_frame._label.config(bg=bg_color, fg=fg_color)
    
    def winfo_children(self):
        """Return list of page widgets for compatibility."""
        return list(self.pages.values())
    
    def configure(self, **kwargs):
        """Configure notebook options."""
        if "style" in kwargs:
            # Ignore style for compatibility
            pass
        return self
    
    # Alias
    config = configure