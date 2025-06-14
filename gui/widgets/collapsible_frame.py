import tkinter as tk
from tkinter import ttk
from gui.dialog_helper import DialogHelper



class CollapsibleFrame(ttk.Frame):
    """A frame that can be expanded or collapsed with a toggle button."""
    
    def __init__(self, parent, text="", expanded=False, bg="#252526", fg="#e0e0e0", 
                 title_bg="#252526", title_fg="#e0e0e0", content_bg="#1e1e1e", 
                 border_color="#3f3f3f", arrow_color="#3a7ca5", **kwargs):
        """Initialize the collapsible frame with theme support."""
        ttk.Frame.__init__(self, parent, **kwargs)
        
        # Save theme colors
        self.bg = bg
        self.fg = fg
        self.title_bg = title_bg
        self.title_fg = title_fg
        self.content_bg = content_bg
        self.border_color = border_color
        self.arrow_color = arrow_color
        
        # Create the header with a border
        self.header_frame = tk.Frame(
            self, 
            bg=self.title_bg,
            highlightbackground=self.border_color,
            highlightthickness=1
        )
        self.header_frame.pack(fill=tk.X)
        
        # Create toggle button with arrow indicator
        self.toggle_button = tk.Label(
            self.header_frame, 
            text="▼ " if expanded else "▶ ",
            cursor="hand2",
            bg=self.title_bg,
            fg=self.arrow_color,
            font=("Arial", 11, "bold"),
            padx=5,
            pady=8
        )
        self.toggle_button.pack(side=tk.LEFT)
        self.toggle_button.bind("<Button-1>", self.toggle)
        
        # Create header label
        self.header_label = tk.Label(
            self.header_frame, 
            text=DialogHelper.t(text),
            cursor="hand2",
            font=("Arial", 11, "bold"),
            bg=self.title_bg,
            fg=self.title_fg,
            padx=5,
            pady=8
        )
        self.header_label.pack(side=tk.LEFT, fill=tk.X, expand=False, anchor="w")
        self.header_label.bind("<Button-1>", self.toggle)
        
        # Add separator line when collapsed
        self.separator = ttk.Separator(self, orient="horizontal")
        if not expanded:
            self.separator.pack(fill=tk.X)
        
        # Content frame
        self.content_frame = tk.Frame(
            self, 
            bg=self.content_bg,
            padx=15,
            pady=15
        )
        
        # Set initial state
        self.expanded = expanded
        if expanded:
            self.content_frame.pack(fill=tk.BOTH, expand=True)


    def set_text(self, text):
        """
        Update the frame title text.
        
        Args:
            text: New title text
        """
        if hasattr(self, 'header_label'):
            self.header_label.config(text=text)

    def toggle(self, event=None):
        """Toggle the expanded/collapsed state."""
        if self.expanded:
            self.content_frame.pack_forget()
            self.toggle_button.configure(text="▶ ")
            self.separator.pack(fill=tk.X)
            self.expanded = False
        else:
            self.separator.pack_forget()
            self.content_frame.pack(fill=tk.BOTH, expand=True)
            self.toggle_button.configure(text="▼ ")
            self.expanded = True

    def update_theme(self, **kwargs):
        """
        Update the theme colors of the collapsible frame.
        
        Args:
            **kwargs: Theme color values including:
                bg: Background color
                fg: Foreground text color
                title_bg: Title bar background color
                title_fg: Title text color
                content_bg: Content area background color
                border_color: Border color
                arrow_color: Arrow indicator color
        """
        try:
            # Update stored theme colors
            self.bg = kwargs.get("bg", self.bg)
            self.fg = kwargs.get("fg", self.fg)
            self.title_bg = kwargs.get("title_bg", self.title_bg)
            self.title_fg = kwargs.get("title_fg", self.title_fg)
            self.content_bg = kwargs.get("content_bg", self.content_bg)
            self.border_color = kwargs.get("border_color", self.border_color)
            self.arrow_color = kwargs.get("arrow_color", self.arrow_color)
            
            # Update header frame - use tkinter config for tk.Frame
            if isinstance(self.header_frame, tk.Frame):
                self.header_frame.config(
                    bg=self.title_bg,
                    highlightbackground=self.border_color
                )
            
            # Update toggle button - use tkinter config for tk.Label
            if isinstance(self.toggle_button, tk.Label):
                self.toggle_button.config(
                    bg=self.title_bg,
                    fg=self.arrow_color
                )
            
            # Update header label - use tkinter config for tk.Label
            if isinstance(self.header_label, tk.Label):
                self.header_label.config(
                    bg=self.title_bg,
                    fg=self.title_fg
                )
            
            # Update content frame - use tkinter config for tk.Frame
            if isinstance(self.content_frame, tk.Frame):
                self.content_frame.config(
                    bg=self.content_bg
                )
            
            # Log successful update
            import logging
            logging.debug(f"Successfully updated CollapsibleFrame theme")
            
        except Exception as e:
            import logging
            logging.error(f"Failed to update CollapsibleFrame theme: {e}")
            import traceback
            logging.error(traceback.format_exc())
