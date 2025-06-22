
import json
import os
import sys
import logging
import tkinter as tk
from tkinter import ttk
import traceback


# gui_manager.py
from gui.widgets.modern_button import ModernButton
from gui.widgets.collapsible_frame import CollapsibleFrame
from gui.widgets.custom_checkbox import create_custom_checkbox
from gui.widgets.themed_combobox import create_themed_combobox
from gui.widgets.entry_with_validation import create_entry_with_validation
from gui.widgets.field_with_label import create_field_with_label
from gui.widgets.text_display import create_text_display
from gui.widgets.modern_notebook import ModernNotebook
from gui.dialog_helper import DialogHelper


# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GUIManager:
    """
    Manages GUI appearance, themes, and widget styling across the application.
    Provides centralized control over colors, fonts, and styling for consistency.
    """
    
    def __init__(self, file_manager=None, config_manager=None):
        """
        Initialize the GUI manager with default settings and themes.
        
        Args:
            file_manager: Optional FileManager instance for loading/saving preferences
            config_manager: Optional ConfigManager for storing settings
        """
        self.file_manager = file_manager
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)


        
        # Add a dictionary to track themed widgets
        self.themed_widgets = {}
        # Define theme colors for dark and light themes
        self.theme_modes = {
            "dark": {
                "background": "#1e1e1e",          # Dark background
                "secondary_bg": "#252526",        # Slightly lighter background for contrast
                "text": "#e0e0e0",                # Light text
                "accent_blue": "#3a7ca5",         # Muted blue accent
                "accent_green": "#4a8259",        # Muted green accent
                "accent_red": "#9e4a4a",          # Muted red for quit button
                "field_bg": "#2d2d2d",            # Form field background
                "field_border": "#3f3f3f",        # Form field border
                "hover_highlight": "#3a3a3a",     # Highlight color for hover effects
                "accent_error": "#5c2d2d",        # Light red for error/invalid state
                "accent_valid": "#2d5c2d",        # Light green for valid state
                "checkbox_bg": "#353535",         # Checkbox background
                "checkbox_fg": "#4a8259",         # Checkbox foreground when checked
                "progress_bg": "#252526",         # Progress bar background
                "progress_fg": "#4a8259",         # Progress bar foreground
                "menu_bg": "#252526",             # Menu background
                "menu_fg": "#e0e0e0",             # Menu text (was #2d2d2d, too dark for contrast)
                "menu_active_bg": "#3a7ca5",      # Menu active background
                "menu_active_fg": "#e0e0e0",      # Menu active text (was #ffffff)
                "border": "#3f3f3f",              # Border color
                "separator": "#3f3f3f"            # Separator color
            },            
            "light": {
                "background": "#f5f5f5",          # Light background
                "secondary_bg": "#e8e8e8",        # Slightly darker background for contrast
                "text": "#333333",                # Dark text
                "accent_blue": "#4a90c0",         # Blue accent
                "accent_green": "#5aa06c",        # Green accent
                "accent_red": "#c05a5a",          # Red for quit button
                "field_bg": "#ffffff",            # Form field background
                "field_border": "#cccccc",        # Form field border
                "hover_highlight": "#dddddd",     # Highlight color for hover effects
                "accent_error": "#ffebeb",        # Light red for error/invalid state
                "accent_valid": "#ebffeb",        # Light green for valid state
                "checkbox_bg": "#ffffff",         # Checkbox background
                "checkbox_fg": "#5aa06c",         # Checkbox foreground when checked
                "progress_bg": "#e8e8e8",         # Progress bar background
                "progress_fg": "#5aa06c",         # Progress bar foreground
                "menu_bg": "#f0f0f0",             # Menu background
                "menu_fg": "#333333",             # Menu text
                "menu_active_bg": "#4a90c0",      # Menu active background
                "menu_active_fg": "#ffffff",      # Menu active text
                "border": "#cccccc",              # Border color
                "separator": "#dddddd"            # Separator color
            }
        }
        
        # Set the initial theme based on saved preference
        self.current_theme = self._load_theme_preference() or "dark"
        self.theme_colors = self.theme_modes[self.current_theme]
        



        # Font configurations
        self.fonts = {
            "title": ("Arial", 16, "bold"),
            "subtitle": ("Arial", 14, "bold"),
            "heading": ("Arial", 12, "bold"),
            "normal": ("Arial", 10),
            "small": ("Arial", 9),
            "code": ("Consolas", 10),
            "button": ("Arial", 11),
        }
    
    def t(self, text):
        """
        Translate text if translator is available.
        
        Args:
            text: Text to translate
            
        Returns:
            Translated text or original if no translator
        """
        return DialogHelper.t(text)
    
    def _load_theme_preference(self):
        """
        Load theme preference from config.
        
        Returns:
            Theme name or None if not found
        """
        try:
            if not self.file_manager:
                return None
                
            config_path = os.path.join(self.file_manager.base_dir, "Scripts", "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if 'theme' in config:
                        theme = config['theme']
                        self.logger.info(f"Loaded theme preference: {theme}")
                        return theme
        except Exception as e:
            self.logger.error(f"Error loading theme preference: {str(e)}")
        return None
    
    def save_theme_preference(self):
        """Save current theme preference to config."""
        try:
            if not self.file_manager:
                return
                
            config_dir = os.path.join(self.file_manager.base_dir, "Program Resources")
            os.makedirs(config_dir, exist_ok=True)
            config_path = os.path.join(config_dir, "config.json")
            
            # Load existing config if it exists
            config = {}
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
            
            # Update theme setting
            config['theme'] = self.current_theme
            
            # Save updated config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            self.logger.info(f"Saved theme preference: {self.current_theme}")
        except Exception as e:
            self.logger.error(f"Error saving theme preference: {str(e)}")
    
    def toggle_theme(self):
        """
        Toggle between light and dark themes.
        
        Returns:
            New theme name
        """
        self.current_theme = "dark" if self.current_theme == "light" else "light"
        self.theme_colors = self.theme_modes[self.current_theme]
        self.save_theme_preference()
        return self.current_theme
    
    def get_theme_label(self):
        """Get label for theme toggle menu item."""
        return self.t("Switch to Light Theme") if self.current_theme == "dark" else self.t("Switch to Dark Theme")
    
    def configure_ttk_styles(self, root=None):
        """
        Configure ttk styles based on current theme.
        
        Args:
            root: Optional root window for style configuration
        """
        print(f"DEBUG: configure_ttk_styles called, root: {root}")
        print(f"DEBUG: root type: {type(root).__name__}, id: {id(root)}")
        
        if root is not None:
            try:
                print(f"DEBUG: root exists: {root.winfo_exists()}, class: {root.winfo_class()}")
                print(f"DEBUG: root geometry: {root.winfo_geometry()}")
            except Exception as e:
                print(f"DEBUG: Error checking root widget: {str(e)}")
        
        style = ttk.Style(root) if root else ttk.Style()
        print(f"DEBUG: Created ttk.Style with root: {root is not None}")
              
        # Set theme colors
        colors = self.theme_colors
        
        # Configure frame styles with complete styling
        style.configure('TFrame', background=colors["background"])
        style.configure('Main.TFrame', background=colors["background"])
        style.configure('Header.TFrame', background=colors["secondary_bg"])
        style.configure('Content.TFrame', background=colors["background"])
        style.configure('Footer.TFrame', background=colors["background"])
        style.configure('Section.TFrame', background=colors["secondary_bg"], 
                    borderwidth=1, relief="raised", bordercolor=colors["border"])

        # Configure LabelFrame styles
        style.configure('TLabelframe', 
                    background=colors["background"],
                    foreground=colors["text"],
                    bordercolor=colors["border"])
        style.configure('TLabelframe.Label', 
                    background=colors["background"],
                    foreground=colors["text"],
                    font=self.fonts["heading"])

        # Configure label styles with complete styling
        style.configure('TLabel', 
                    background=colors["background"], 
                    foreground=colors["text"], 
                    font=self.fonts["normal"])
        style.configure('Title.TLabel', 
                    # CHECK
                    background=colors["background"], 
                    foreground=colors["text"], 
                    font=self.fonts["title"])
        style.configure('SectionTitle.TLabel', 
                    background=colors["secondary_bg"], 
                    foreground=colors["text"], 
                    font=self.fonts["subtitle"])
        style.configure('Content.TLabel', 
                    background=colors["background"], 
                    foreground=colors["text"])
        style.configure('Value.TLabel', 
                    background=colors["background"], 
                    foreground=colors["accent_blue"], 
                    font=self.fonts["normal"])
        style.configure('Instructions.TLabel', 
                background=colors["background"], 
                foreground=colors["text"], 
                font=self.fonts["title"],
                padding=15)
        
        # Configure combobox styles
        style.configure('TCombobox',
            fieldbackground=colors["field_bg"],
            background=colors["field_bg"],
            foreground=colors["text"],
            bordercolor=colors["field_border"],
            lightcolor=colors["field_bg"],
            darkcolor=colors["field_bg"],
            arrowcolor=colors["text"],
            insertcolor=colors["text"]
        )

        style.map('TCombobox',
            fieldbackground=[
                ('readonly', colors["field_bg"]),
                ('focus', colors["field_bg"]),
                ('disabled', colors["secondary_bg"])
            ],
            foreground=[
                ('readonly', colors["text"]),
                ('focus', colors["text"]),
                ('disabled', colors["field_border"])
            ],
            background=[
                ('readonly', colors["field_bg"]),
                ('focus', colors["field_bg"]),
                ('disabled', colors["secondary_bg"])
            ],
            arrowcolor=[
                ('active', colors["accent_blue"]), 
                ('pressed', colors["accent_blue"]),
                ('!active', colors["text"])
            ]
        )
        
        # Also configure the custom style
        style.configure('ThemedCustom.TCombobox',
            fieldbackground=colors["field_bg"],
            background=colors["field_bg"],
            foreground=colors["text"],
            bordercolor=colors["field_border"],
            lightcolor=colors["field_bg"],
            darkcolor=colors["field_bg"],
            arrowcolor=colors["text"]
        )
        
        style.map('ThemedCustom.TCombobox',
            fieldbackground=[('readonly', colors["field_bg"]), ('disabled', colors["secondary_bg"])],
            foreground=[('readonly', colors["text"]), ('disabled', colors["field_border"])],
            background=[('readonly', colors["field_bg"]), ('disabled', colors["secondary_bg"])],
            arrowcolor=[
                ('disabled', colors["field_border"]),
                ('pressed', colors["accent_blue"]),
                ('active', colors["accent_blue"])
            ]
        )

        # Configure button styles with complete styling
        style.configure('TButton', 
                background=colors["secondary_bg"], 
                foreground=colors["text"])
        style.map('TButton',
                background=[('active', colors["hover_highlight"])],
                foreground=[('active', colors["text"])])
        
        style.configure('Custom.TCombobox',
            fieldbackground=colors["field_bg"],
            background=colors["field_bg"],
            foreground=colors["text"],
            bordercolor=colors["field_border"],
            lightcolor=colors["field_bg"],
            darkcolor=colors["field_bg"],
            arrowcolor=colors["text"],
            insertcolor=colors["text"]
        )
        
        style.map('Custom.TCombobox',
            fieldbackground=[('readonly', colors["field_bg"]), ('disabled', colors["secondary_bg"])],
            foreground=[('readonly', colors["text"]), ('disabled', colors["field_border"])],
            background=[('readonly', colors["field_bg"]), ('disabled', colors["secondary_bg"])],
            arrowcolor=[
                ('disabled', colors["field_border"]),
                ('pressed', colors["accent_blue"]),
                ('active', colors["accent_blue"])
            ]
        )
        
        # Configure checkbox styles with complete styling
        style.configure('TCheckbutton', 
                    background=colors["background"], 
                    foreground=colors["text"])
        style.map('TCheckbutton', 
                background=[('active', colors["background"])],
                foreground=[('active', colors["text"])])
        
        # Configure entry styles with complete styling
        style.configure('TEntry', 
                    fieldbackground=colors["field_bg"],
                    foreground=colors["text"],
                    insertcolor=colors["text"],
                    bordercolor=colors["field_border"])
        
        # Configure progress bar with complete styling
        style.configure('Horizontal.TProgressbar', 
                    background=colors["accent_green"],
                    troughcolor=colors["secondary_bg"],
                    bordercolor=colors["border"],
                    lightcolor=colors["accent_green"],
                    darkcolor=colors["accent_green"])
        style.configure('Themed.Horizontal.TProgressbar', 
                    background=colors["accent_green"],
                    troughcolor=colors["secondary_bg"],
                    bordercolor=colors["border"],
                    lightcolor=colors["accent_green"],
                    darkcolor=colors["accent_green"])
        
        # Configure scrollbar with complete styling
        style.configure('TScrollbar', 
                    background=colors["secondary_bg"],
                    troughcolor=colors["background"],
                    bordercolor=colors["border"],
                    arrowcolor=colors["text"])
        style.map('TScrollbar',
                background=[('active', colors["hover_highlight"])],
                arrowcolor=[('active', colors["accent_blue"])])
        
        # Configure scale/slider with complete styling
        style.configure('Horizontal.TScale',
                    background=colors["background"],
                    troughcolor=colors["secondary_bg"],
                    slidercolor=colors["accent_blue"])
        
        # Configure separator with theme colors
        style.configure('TSeparator', 
                    background=colors["separator"])
        
        # Configure notebook and tab styles
        style.configure('TNotebook',
            background=colors["background"],
            borderwidth=0
        )
        
        style.configure('TNotebook.Tab',
            background=colors["secondary_bg"],
            foreground=colors["text"],
            padding=[12, 8],
            font=self.fonts["normal"]
        )
        
        style.map('TNotebook.Tab',
            background=[
                ('selected', colors["accent_blue"]),
                ('active', colors["hover_highlight"]),
                ('!selected', colors["secondary_bg"])
            ],
            foreground=[
                ('selected', '#ffffff'),
                ('active', colors["text"]),
                ('!selected', colors["text"])
            ]
        )
        
        # Also create a custom themed notebook style
        style.configure('Themed.TNotebook',
            background=colors["background"],
            borderwidth=0,
            tabposition='n'
        )
        
        style.configure('Themed.TNotebook.Tab',
            background=colors["secondary_bg"],
            foreground=colors["text"],
            padding=[20, 10],
            font=self.fonts["normal"],
            borderwidth=0
        )
        
        # IMPORTANT: Put disabled state FIRST in the map
        style.map('Themed.TNotebook.Tab',
            background=[
                ('disabled', colors["field_bg"]),  # Disabled tabs use field background
                ('selected', colors["accent_blue"]),
                ('active', colors["hover_highlight"]),
                ('!selected', colors["secondary_bg"])
            ],
            foreground=[
                ('disabled', colors["field_border"]),  # Disabled text is muted
                ('selected', '#ffffff'),
                ('active', colors["text"]),
                ('!selected', colors["text"])
            ],
            expand=[('selected', [1, 1, 1, 0])]
        )
        
        # Configure the tab area background
        style.configure('Themed.TNotebook.Tab',
            background=colors["secondary_bg"],
            tabmargins=[2, 5, 2, 0]
        )
        
        # Force theme reloading - important for complete theme switching
        style.theme_use(style.theme_use())
    
    def create_modern_button(self, parent, text, color, command, icon=None, grid_pos=None):
        """
        Create a modern styled button with hover effects.
        
        Args:
            parent: Parent widget
            text: Button text
            color: Base button color
            command: Button command
            icon: Optional text icon
            grid_pos: Optional grid position tuple (row, col)
        
        Returns:
            ModernButton instance
        """
        button = ModernButton(
            parent,
            text,
            color,
            command,
            icon=icon,
            theme_colors=self.theme_colors
        )
        
        # Apply grid positioning if specified
        if grid_pos:
            button.grid(row=grid_pos[0], column=grid_pos[1], padx=5, pady=5)
            
        return button
    
    def create_collapsible_frame(self, parent, title, expanded=False):
        """
        Create a themed collapsible frame.
        
        Args:
            parent: Parent widget
            title: Frame title
            expanded: Whether frame is initially expanded
        
        Returns:
            CollapsibleFrame instance
        """
        frame = CollapsibleFrame(
            parent,
            text=title,
            expanded=expanded,
            bg=self.theme_colors["secondary_bg"],
            fg=self.theme_colors["text"],
            title_bg=self.theme_colors["secondary_bg"],
            title_fg=self.theme_colors["text"],
            content_bg=self.theme_colors["background"],
            border_color=self.theme_colors["border"],
            arrow_color=self.theme_colors["accent_blue"]
        )
        frame.pack(fill=tk.X, pady=(0, 10))
        return frame
    
    def create_custom_checkbox(self, parent, text, variable, command=None):
        """
        Wrapper for custom checkbox creation using theme colors and translation.

        Args:
            parent: Parent widget.
            text: Checkbox label.
            variable: BooleanVar controlling state.
            command: Optional function to run when toggled.

        Returns:
            Frame containing the checkbox.
        """
        return create_custom_checkbox(
            parent=parent,
            text=text,
            variable=variable,
            theme_colors=self.theme_colors,
            command=command,
            translator=self.t
        )

    def create_combobox(self, parent, variable, values, width=10, readonly=True, validate_func=None):
        return create_themed_combobox(
            parent,
            variable=variable,
            values=values,
            theme_colors=self.theme_colors,
            fonts=self.fonts,
            width=width,
            readonly=readonly,
            validate_func=validate_func
        )

    def create_entry_with_validation(self, parent, textvariable, validate_func=None, width=None, placeholder=None):
        return create_entry_with_validation(
            parent=parent,
            textvariable=textvariable,
            theme_colors=self.theme_colors,
            font=self.fonts["normal"],
            validate_func=validate_func,
            width=width,
            placeholder=placeholder
        )

    def create_field_with_label(
        self, parent, label_text, variable,
        field_type="entry", readonly=False, width=20,
        validate_func=None, values=None, placeholder=None
    ):
        return create_field_with_label(
            parent=parent,
            label_text=label_text,
            variable=variable,
            theme_colors=self.theme_colors,
            fonts=self.fonts,
            translator=self.t,
            field_type=field_type,
            readonly=readonly,
            width=width,
            validate_func=validate_func,
            values=values,
            placeholder=placeholder
        )

    def create_text_display(self, parent, height=10, wrap=tk.WORD, readonly=True):
        return create_text_display(
            parent=parent,
            theme_colors=self.theme_colors,
            font=self.fonts["code"],
            height=height,
            wrap=wrap,
            readonly=readonly
        )

    def update_custom_widget_theme(self, root_widget):
        """
        Apply updated theme styles to all custom widgets.

        Args:
            root_widget: The top-level widget (typically a Frame or the root window)
        """
        try:
            # Apply new ttk styles
            self.configure_ttk_styles(root_widget)
            
            # Update ComboBox style explicitly
            self._update_combobox_style(self.theme_colors)

            # Create a dictionary to track widget names
            widget_dict = {}
            
            # Find all widgets and organize by type for special handling
            def collect_widgets(widget, path="root"):
                widget_type = type(widget).__name__
                widget_class = widget.__class__.__name__ if hasattr(widget, '__class__') else widget_type
                
                # Track this widget
                widget_dict[path] = widget
                
                # Recurse into children
                for i, child in enumerate(widget.winfo_children()):
                    child_path = f"{path}.{i}"
                    collect_widgets(child, child_path)
            
            # Collect all widgets
            collect_widgets(root_widget)
            
            # Update any explicitly tracked widgets first (these take priority)
            if hasattr(self, 'themed_widgets'):
                for name, widget in self.themed_widgets.items():
                    try:
                        # Check if widget still exists
                        if not hasattr(widget, 'winfo_exists') or widget.winfo_exists():
                            if hasattr(widget, 'update_theme') and callable(getattr(widget, 'update_theme')):
                                widget.update_theme(**self.theme_colors)
                            self.logger.debug(f"Updated tracked widget: {name}")
                    except Exception as e:
                        self.logger.error(f"Error updating tracked widget {name}: {e}")
            
            # First pass: Update all widgets using standard methods
            for path, widget in widget_dict.items():
                try:
                    widget_type = type(widget).__name__
                    is_custom = hasattr(widget, 'update_theme')

                    self.logger.debug(f"\n🔧 Updating widget at path: {path}")
                    self.logger.debug(f"   ├─ Widget type: {widget_type}")
                    self.logger.debug(f"   └─ Is custom: {is_custom}")

                    if isinstance(widget, CollapsibleFrame):
                        if hasattr(widget, 'update_theme'):
                            self.logger.debug("   🔁 Using CollapsibleFrame.update_theme()")
                            widget.update_theme(**self.theme_colors)
                        else:
                            self.logger.debug("   🧪 Falling back to _update_collapsible_frame()")
                            self._update_collapsible_frame(widget, path, self.theme_colors)

                    elif isinstance(widget, tk.Button):
                        theme_kwargs = {
                            "bg": self.theme_colors["secondary_bg"],
                            "fg": self.theme_colors["text"],
                            "activebackground": self.theme_colors["hover_highlight"],
                            "activeforeground": self.theme_colors["text"],
                            "relief": tk.FLAT,
                            "highlightbackground": self.theme_colors["field_border"]
                        }
                        widget.config(**theme_kwargs)

                    elif isinstance(widget, ttk.Combobox):
                        continue  # Skip here - handled in second pass
                    
                    elif hasattr(widget, '__class__') and widget.__class__.__name__ == 'ModernNotebook':
                        # Update ModernNotebook theme
                        if hasattr(widget, 'update_theme'):
                            widget.update_theme(self.theme_colors)
                            self.logger.debug(f"Updated ModernNotebook theme at {path}")


                    elif isinstance(widget, tk.OptionMenu):
                        self.style_dropdown(widget)

                    elif isinstance(widget, (tk.Text, tk.Entry)):
                        theme_kwargs = {
                            "bg": self.theme_colors["field_bg"],
                            "fg": self.theme_colors["text"],
                            "insertbackground": self.theme_colors["text"]
                        }
                        widget.config(**theme_kwargs)

                    elif isinstance(widget, (tk.Label, tk.Canvas, tk.Frame)):
                        theme_kwargs = {
                            "bg": self.theme_colors["background"],
                            "fg": self.theme_colors["text"] if isinstance(widget, tk.Label) else None
                        }
                        widget.config(**{k:v for k,v in theme_kwargs.items() if v is not None})

                    elif is_custom:
                        widget.update_theme(self.theme_colors)

                except Exception as e:
                    self.logger.error(f"❌ Error updating widget at {path} ({widget_type}): {e}")
                    self.logger.error("↪️ Full traceback:\n" + traceback.format_exc())

            # Second pass: Force refresh ComboBox widgets
            for path, widget in widget_dict.items():
                if not isinstance(widget, ttk.Combobox):
                    continue

                try:
                    custom_style = "Custom.TCombobox"
                    widget.configure(style=custom_style)

                    if 'readonly' in widget.state():
                        widget.state(['!readonly'])
                        widget.state(['readonly'])

                    self.logger.debug(f"✅ Refreshed Combobox at {path}")

                    # Force refresh
                    if 'readonly' in widget.state():
                        widget.state(['!readonly'])
                        widget.state(['readonly'])

                except Exception as e:
                    self.logger.error(f"❌ Could not refresh ComboBox at {path}: {e}")
                    self.logger.error("↪️ Full traceback:\n" + traceback.format_exc())

        except Exception as e:
            self.logger.error(f"❌ Error in update_custom_widget_theme: {e}")
            self.logger.error("↪️ Full traceback:\n" + traceback.format_exc())

    def _update_combobox_style(self, theme_colors):
        """Update the custom Combobox style used across the application."""
        style = ttk.Style()

        custom_style = "Custom.TCombobox"

        if not style.layout(custom_style):
            self.logger.debug(f"🧪 Creating custom style: {custom_style}")
            style.configure(custom_style,
                foreground=theme_colors["text"],
                background=theme_colors["field_bg"],
                fieldbackground=theme_colors["field_bg"],
                arrowcolor=theme_colors["text"]
            )
            style.map(custom_style,
                fieldbackground=[('readonly', theme_colors["field_bg"])],
                selectbackground=[('readonly', theme_colors.get("selection_bg", theme_colors["accent_blue"]))],
                selectforeground=[('readonly', theme_colors.get("selection_fg", "#ffffff"))],
                background=[('readonly', theme_colors["field_bg"])],
                foreground=[('readonly', theme_colors["text"])],
                arrowcolor=[('active', theme_colors["accent_blue"]), ('!active', theme_colors["text"])]
            )
        else:
            self.logger.debug(f"✅ Reusing existing style: {custom_style}")


    def _update_collapsible_frame(self, frame, name, theme_colors):
        """Manually update a CollapsibleFrame if it doesn't have update_theme."""
        logging.debug(f"Manual update for CollapsibleFrame {name}")
        
        try:
            # Update stored theme colors
            if hasattr(frame, 'bg'):
                frame.bg = theme_colors["background"]
            if hasattr(frame, 'fg'):
                frame.fg = theme_colors["text"]
            if hasattr(frame, 'title_bg'):
                frame.title_bg = theme_colors["secondary_bg"]
            if hasattr(frame, 'title_fg'):
                frame.title_fg = theme_colors["text"]
            if hasattr(frame, 'content_bg'):
                frame.content_bg = theme_colors["background"]
            if hasattr(frame, 'border_color'):
                frame.border_color = theme_colors["border"]
            if hasattr(frame, 'arrow_color'):
                frame.arrow_color = theme_colors["accent_blue"]
            
            # Update header frame
            if hasattr(frame, 'header_frame') and isinstance(frame.header_frame, tk.Frame):
                frame.header_frame.config(
                    bg=theme_colors["secondary_bg"],
                    highlightbackground=theme_colors["border"]
                )
            
            # Update toggle button
            if hasattr(frame, 'toggle_button') and isinstance(frame.toggle_button, tk.Label):
                frame.toggle_button.config(
                    bg=theme_colors["secondary_bg"],
                    fg=theme_colors["accent_blue"]
                )
            
            # Update header label
            if hasattr(frame, 'header_label') and isinstance(frame.header_label, tk.Label):
                frame.header_label.config(
                    bg=theme_colors["secondary_bg"],
                    fg=theme_colors["text"]
                )
            
            # Update content frame
            if hasattr(frame, 'content_frame') and isinstance(frame.content_frame, tk.Frame):
                frame.content_frame.config(
                    bg=theme_colors["background"]
                )
            
            logging.debug(f"Manual update completed for CollapsibleFrame {name}")
            
        except Exception as e:
            logging.error(f"Manual update failed for CollapsibleFrame {name}: {e}")

    def style_dropdown(self, dropdown, width=10):
        """
        Apply theme styling to a dropdown menu.
        
        Args:
            dropdown: The dropdown (OptionMenu) widget to style
            width: Optional width for the dropdown
        """
        dropdown.config(
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["text"],
            activebackground=self.theme_colors["hover_highlight"],
            activeforeground=self.theme_colors["text"],
            font=self.fonts["normal"],
            width=width,
            highlightthickness=0,
            bd=0
        )
        dropdown["menu"].config(
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["text"],
            activebackground=self.theme_colors["hover_highlight"],
            activeforeground=self.theme_colors["text"],
            font=self.fonts["normal"],
            bd=0
        )
    

    def configure_standard_tags(self, text_widget):
        """
        Configure standard tags for a text widget.
        
        Args:
            text_widget: Text widget to configure
        """
        text_widget.tag_configure("error", foreground="#ff6b6b")
        text_widget.tag_configure("warning", foreground="#feca57")
        text_widget.tag_configure("success", foreground="#1dd1a1")
        text_widget.tag_configure("info", foreground="#54a0ff")
    
    def update_widget_theme(self, widget):
        """
        Recursively update theme styling for a widget and its children.
        
        Args:
            widget: Root widget to update
        """
        self._apply_theme_to_widget(widget)
        
        # Force ttk style update
        self.configure_ttk_styles()
    
    def apply_theme(self, widget):
        """
        Apply theme styling to a widget and its children.
        Alias for update_widget_theme for backward compatibility.
        
        Args:
            widget: Widget to apply theme to
        """
        self.update_widget_theme(widget)
    

    def _apply_theme_to_widget(self, widget):
        """
        Recursively apply background/foreground styling to non-ttk widgets
        based on the current theme.
        
        Args:
            widget: Widget to apply styling to
        """
        if not widget.winfo_exists():
            return
            
        colors = self.theme_colors  # Automatically matches current theme

        try:
            # Input fields (tk.Entry, tk.Text)
            if isinstance(widget, (tk.Entry, tk.Text)):
                widget.configure(
                    background=colors["field_bg"],
                    foreground=colors["text"],
                    insertbackground=colors["text"],
                    disabledbackground=colors["field_bg"],
                    disabledforeground=colors["field_border"]
                )

            # Standard buttons
            elif isinstance(widget, tk.Button):
                widget.configure(
                    background=colors["secondary_bg"],
                    foreground=colors["text"],
                    activebackground=colors["hover_highlight"],
                    activeforeground=colors["text"]
                )

            # Labels
            elif isinstance(widget, tk.Label):
                widget.configure(
                    background=colors["background"],
                    foreground=colors["text"]
                )

            # Frames
            elif isinstance(widget, tk.Frame):
                widget.configure(background=colors["background"])

            # ModernButton
            elif isinstance(widget, ModernButton):
                widget.theme_colors = colors.copy()
                # Refresh button appearance
                state = 'normal' if widget.enabled else 'disabled'
                widget.set_state(state)
                
            # Canvas
            elif isinstance(widget, tk.Canvas):
                widget.config(bg=colors["background"])
                
            # Menu handling
            if hasattr(widget, 'menuname'):
                try:
                    menu = widget.nametowidget(widget.menuname)
                    menu.config(
                        bg=colors["field_bg"],
                        fg=colors["text"],
                        activebackground=colors["hover_highlight"],
                        activeforeground=colors["text"]
                    )
                except:
                    pass

            # Recurse into child widgets if it's a container
            if isinstance(widget, (tk.Frame, ttk.Frame)):
                for child in widget.winfo_children():
                    self._apply_theme_to_widget(child)

        except Exception as e:
            # Skip widgets that don't support these options
            self.logger.debug(f"Could not apply theme to widget {widget}: {e}")
    
    def setup_menubar(self, root, menus):
        """
        Set up a themed menubar with provided menu definitions.
        
        Args:
            root: Root window
            menus: Dictionary of menu definitions
            
        Returns:
            The menubar
        """
        colors = self.theme_colors
        
        menubar = tk.Menu(
            root, 
            bg=colors["menu_bg"], 
            fg=colors["menu_fg"]
        )
        root.config(menu=menubar)
        
        # Create each menu from the definitions
        for menu_name, menu_items in menus.items():
            menu = tk.Menu(
                menubar, 
                tearoff=0, 
                bg=colors["menu_bg"], 
                fg=colors["menu_fg"],
                activebackground=colors["menu_active_bg"],
                activeforeground=colors["menu_active_fg"]
            )
            
            # Add items to this menu
            for item in menu_items:
                if item['type'] == 'command':
                    menu.add_command(
                        label=self.t(item['label']),
                        command=item['command']
                    )
                elif item['type'] == 'separator':
                    menu.add_separator()
                elif item['type'] == 'cascade':
                    submenu = tk.Menu(
                        menu, 
                        tearoff=0, 
                        bg=colors["menu_bg"], 
                        fg=colors["menu_fg"],
                        activebackground=colors["menu_active_bg"],
                        activeforeground=colors["menu_active_fg"]
                    )
                    
                    for subitem in item['items']:
                        if subitem['type'] == 'command':
                            submenu.add_command(
                                label=self.t(subitem['label']),
                                command=subitem['command']
                            )
                        elif subitem['type'] == 'separator':
                            submenu.add_separator()
                        elif subitem['type'] == 'radiobutton':
                            submenu.add_radiobutton(
                                label=self.t(subitem['label']),
                                value=subitem['value'],
                                variable=subitem['variable'],
                                command=subitem['command']
                            )
                    
                    menu.add_cascade(
                        label=self.t(item['label']),
                        menu=submenu
                    )
                elif item['type'] == 'radiobutton':
                    menu.add_radiobutton(
                        label=self.t(item['label']),
                        value=item['value'],
                        variable=item['variable'],
                        command=item['command']
                    )
            
            menubar.add_cascade(label=self.t(menu_name), menu=menu)
        
        return menubar
    

    # GUI SETTINGS ABOVE
    #=======================================================================================================================
    # GUI FRAME CREATION BELOW

    
    def create_main_window(self, root, title, width=1200, height=900, center=True):
        """
        Set up a main application window with theme styling.
        
        Args:
            root: Tkinter root window
            title: Window title
            width: Desired window width (default 1200)
            height: Desired window height (default 900)
            center: Whether to center the window on screen
            
        Returns:
            Dictionary of main window components
        """
        # Set window title
        root.title(self.t(title))
        
        # Configure window size
        window_width = min(width, root.winfo_screenwidth() - 100)
        window_height = min(height, root.winfo_screenheight() - 100)
        
        # Center window if requested
        if center:
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            x_position = (screen_width - window_width) // 2
            y_position = (screen_height - window_height) // 2
            root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
        else:
            root.geometry(f"{window_width}x{window_height}")
        
        # Configure the theme for root window
        root.configure(bg=self.theme_colors["background"])
        
        # Configure ttk styles for the current theme
        self.configure_ttk_styles(root)
        
        # Create the main container with three distinct sections
        # 1. Header frame (fixed at top)
        # 2. Content frame (scrollable, main content)
        # 3. Footer frame (fixed at bottom, action buttons)
        
        # Main container fills the window
        main_container = ttk.Frame(root, style='Main.TFrame')
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # 1. Header frame - title at top
        header_frame = ttk.Frame(main_container, style='Header.TFrame', height=40)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        
        # Title label in header
        title_label = ttk.Label(
            header_frame, 
            text=self.t(title),
            style='Title.TLabel'
        )
        title_label.pack(pady=8)
        
        # 2. Content frame - scrollable main area
        content_outer_frame = ttk.Frame(main_container, style='Content.TFrame')
        content_outer_frame.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
        
        # Create canvas with scrollbar for scrollable content
        canvas = tk.Canvas(
            content_outer_frame, 
            bg=self.theme_colors["background"],
            highlightthickness=0  # Remove border
        )
        scrollbar = ttk.Scrollbar(content_outer_frame, orient="vertical", command=canvas.yview)
        
        # Configure canvas
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Frame inside canvas for content
        content_frame = ttk.Frame(canvas, style='Content.TFrame', padding=15)
        canvas_window = canvas.create_window((0, 0), window=content_frame, anchor="nw", tags="content_frame")
        
        # Configure canvas scrolling
        def configure_canvas(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfigure("content_frame", width=event.width)
        
        content_frame.bind("<Configure>", configure_canvas)
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure("content_frame", width=e.width))
        
        # Enable mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # 3. Footer frame - fixed at bottom
        footer_frame = ttk.Frame(main_container, style='Footer.TFrame', height=100)
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=10)
        
        # Return a dictionary of all the components
        return {
            'main_container': main_container,
            'header_frame': header_frame,
            'title_label': title_label,
            'content_outer_frame': content_outer_frame,
            'canvas': canvas,
            'scrollbar': scrollbar,
            'content_frame': content_frame,
            'footer_frame': footer_frame,
            'root': root
        }

    def create_section_frame(self, parent, title=None, padding=10):
        """
        Create a section frame with optional title.
        
        Args:
            parent: Parent widget
            title: Optional section title
            padding: Padding inside the frame
            
        Returns:
            Frame widget
        """
        # Create a section frame
        section_frame = ttk.Frame(parent, style='Section.TFrame', padding=padding)
        section_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Add title if provided
        if title:
            title_label = ttk.Label(
                section_frame,
                text=self.t(title),
                style='SectionTitle.TLabel'
            )
            title_label.pack(anchor='w', pady=(0, 5))
        
        return section_frame

    def create_status_section(self, parent, height=10):
        """
        Create a status text section with progress bar.
        
        Args:
            parent: Parent widget
            height: Height of the status text area
            
        Returns:
            Dictionary with status components
        """
        # Progress section
        progress_frame = self.create_section_frame(parent, title="Progress")
        
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(
            progress_frame,
            variable=progress_var,
            orient=tk.HORIZONTAL,
            mode='determinate',
            style='Themed.Horizontal.TProgressbar'
        )
        progress_bar.pack(fill=tk.X, pady=5)
        
        # Status section
        status_frame = self.create_section_frame(parent, title="Detailed Status")
        
        # Status text with scrollbar
        status_container = ttk.Frame(status_frame, style='Content.TFrame')
        status_container.pack(fill=tk.BOTH, expand=True)
        
        status_text = self.create_text_display(
            status_container, 
            height=height,
            wrap=tk.WORD,
            readonly=True
        )
        status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar for results
        scrollbar = ttk.Scrollbar(status_container, command=status_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        status_text.config(yscrollcommand=scrollbar.set)
        
        # Configure tags for status text
        self.configure_standard_tags(status_text)
        
        return {
            'progress_frame': progress_frame,
            'progress_var': progress_var,
            'progress_bar': progress_bar,
            'status_frame': status_frame,
            'status_text': status_text
        }

    def create_button_row(self, parent, button_configs, side="bottom", anchor="se", padx=10, pady=10):
        """
        Create a row of ModernButtons.
        
        Args:
            parent: Parent widget
            button_configs: List of button configurations
            side: Pack side parameter
            anchor: Pack anchor parameter
            padx: Pack padx parameter
            pady: Pack pady parameter
            
        Returns:
            Frame containing the buttons
        """
        button_row = ttk.Frame(parent, style='Footer.TFrame')
        button_row.pack(side=side, anchor=anchor, padx=padx, pady=pady)
        
        # Add spacing between buttons
        button_padding = 5
        
        # Create each button
        buttons = {}
        for config in button_configs:
            button = self.create_modern_button(
                button_row,
                config['text'],
                config.get('color', self.theme_colors["accent_blue"]),
                config['command'],
                icon=config.get('icon'),
            )
            button.pack(side="left", padx=button_padding)
            
            # Store reference with the provided name
            if 'name' in config:
                buttons[config['name']] = button
        
        return button_row, buttons
