"""
Dialog for configuring color maps for drillhole trace visualizations.
"""

import tkinter as tk
from tkinter import ttk, colorchooser
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging

from gui.dialog_helper import DialogHelper
from processing.LoggingReviewStep.color_map_manager import ColorMap, ColorMapType, ColorRange, ColorMapManager

logger = logging.getLogger(__name__)


class ColorMapConfigDialog:
    """
    Dialog for configuring color maps.
    
    Allows users to:
    - Select from presets
    - Customize colors for categories
    - Define numeric ranges and colors
    - Save custom presets
    """
    
    def __init__(self, parent, gui_manager, color_map_manager: ColorMapManager,
                 data_values: List[Any] = None, current_map: ColorMap = None):
        """
        Initialize the color map configuration dialog.
        
        Args:
            parent: Parent window
            gui_manager: GUIManager for theming
            color_map_manager: ColorMapManager instance
            data_values: Sample data values to show categories/ranges
            current_map: Current color map to edit
        """
        self.parent = parent
        self.gui_manager = gui_manager
        self.color_map_manager = color_map_manager
        self.data_values = data_values or []
        self.current_map = current_map
        self.result = None
        
        # Create the dialog
        self._create_dialog()
        
    def _create_dialog(self):
        """Create the main dialog."""
        self.dialog = DialogHelper.create_dialog(
            self.parent,
            title="Configure Color Map",
            modal=True,
            size_ratio=0.6,
            min_width=800,
            min_height=600
        )
        
        # Apply theme
        theme_colors = self.gui_manager.theme_colors
        self.dialog.configure(bg=theme_colors["background"])
        
        # Main container
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top section - preset selection
        self._create_preset_section(main_frame)
        
        # Middle section - color configuration
        self._create_config_section(main_frame)
        
        # Bottom section - buttons
        self._create_button_section(main_frame)
        
        # Load current map if provided
        if self.current_map:
            self._load_color_map(self.current_map)
            
    def _create_preset_section(self, parent):
        """Create the preset selection section."""
        preset_frame = ttk.LabelFrame(parent, text=DialogHelper.t("Presets"), padding="10")
        preset_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Preset dropdown
        ttk.Label(preset_frame, text=DialogHelper.t("Select Preset:")).pack(side=tk.LEFT, padx=(0, 10))
        
        preset_names = ["Custom"] + self.color_map_manager.get_preset_names()
        self.preset_var = tk.StringVar(value="Custom")
        
        # Custom styled dropdown frame
        preset_combo_frame = tk.Frame(
            preset_frame,
            bg=self.gui_manager.theme_colors["field_bg"],
            highlightbackground=self.gui_manager.theme_colors["field_border"],
            highlightthickness=1,
            bd=0
        )
        preset_combo_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        # Preset dropdown
        self.preset_dropdown = tk.OptionMenu(
            preset_combo_frame,
            self.preset_var,
            *preset_names
        )
        self.gui_manager.style_dropdown(self.preset_dropdown, width=30)
        self.preset_dropdown.pack()
        
        # Store reference to update later
        self.preset_combo_frame = preset_combo_frame
        
        # Load preset button
        load_button = self.gui_manager.create_modern_button(
            preset_frame,
            text=DialogHelper.t("Load"),
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._load_preset
        )
        load_button.pack(side=tk.LEFT)
        
        # Type selection
        ttk.Label(preset_frame, text=DialogHelper.t("Type:")).pack(side=tk.LEFT, padx=(20, 10))
        
        self.type_var = tk.StringVar(value=ColorMapType.CATEGORICAL.value)
        
        # Custom styled dropdown frame for type
        type_combo_frame = tk.Frame(
            preset_frame,
            bg=self.gui_manager.theme_colors["field_bg"],
            highlightbackground=self.gui_manager.theme_colors["field_border"],
            highlightthickness=1,
            bd=0
        )
        type_combo_frame.pack(side=tk.LEFT)
        
        # Type dropdown
        type_dropdown = tk.OptionMenu(
            type_combo_frame,
            self.type_var,
            *[t.value for t in ColorMapType]
        )
        self.gui_manager.style_dropdown(type_dropdown, width=15)
        type_dropdown.pack()
        
        # Bind type change
        self.type_var.trace_add("write", self._on_type_change)
        
    def _create_config_section(self, parent):
        """Create the color configuration section."""
        # Container with scrollbar
        config_container = ttk.Frame(parent)
        config_container.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Canvas for scrolling
        canvas = tk.Canvas(
            config_container,
            bg=self.gui_manager.theme_colors["field_bg"],
            highlightthickness=0
        )
        scrollbar = ttk.Scrollbar(config_container, orient="vertical", command=canvas.yview)
        
        self.config_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=self.config_frame, anchor="nw")
        
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure canvas scrolling
        def configure_scroll(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
            
        self.config_frame.bind("<Configure>", configure_scroll)
        
        # Initialize with categorical config
        self._create_categorical_config()
        
    def _create_categorical_config(self):
        """Create configuration UI for categorical color map."""
        # Clear existing content
        for widget in self.config_frame.winfo_children():
            widget.destroy()
            
        # Get unique categories from data
        if self.data_values:
            categories = sorted(set(str(v) for v in self.data_values 
                                  if v is not None and not (isinstance(v, float) and np.isnan(v))))
        else:
            categories = []
            
        # Add category color selectors
        ttk.Label(
            self.config_frame,
            text=DialogHelper.t("Category Colors:"),
            font=self.gui_manager.fonts["heading"]
        ).pack(anchor=tk.W, pady=(0, 10))
        
        self.category_widgets = {}
        
        for category in categories:
            self._add_category_selector(category)
            
        # Add button for new categories
        add_button = self.gui_manager.create_modern_button(
            self.config_frame,
            text=DialogHelper.t("Add Category"),
            color=self.gui_manager.theme_colors["accent_green"],
            command=self._add_new_category
        )
        add_button.pack(pady=10)
        
        # Default color selector
        self._add_default_color_selector()
        
    def _create_numeric_config(self):
        """Create configuration UI for numeric color map."""
        # Clear existing content
        for widget in self.config_frame.winfo_children():
            widget.destroy()
            
        # Range configuration
        ttk.Label(
            self.config_frame,
            text=DialogHelper.t("Numeric Ranges:"),
            font=self.gui_manager.fonts["heading"]
        ).pack(anchor=tk.W, pady=(0, 10))
        
        self.range_widgets = []
        
        # Add button for new range
        add_button = self.gui_manager.create_modern_button(
            self.config_frame,
            text=DialogHelper.t("Add Range"),
            color=self.gui_manager.theme_colors["accent_green"],
            command=self._add_range_selector
        )
        add_button.pack(pady=10)
        
        # Default color selector
        self._add_default_color_selector()
        
    def _add_category_selector(self, category: str, color: Tuple[int, int, int] = None):
        """Add a category color selector."""
        frame = ttk.Frame(self.config_frame)
        frame.pack(fill=tk.X, pady=2)
        
        # Category label
        ttk.Label(frame, text=category, width=20).pack(side=tk.LEFT, padx=(0, 10))
        
        # Color display
        color = color or (200, 200, 200)
        color_display = tk.Canvas(frame, width=30, height=20, 
                                 highlightthickness=1,
                                 highlightbackground=self.gui_manager.theme_colors["border"])
        color_display.pack(side=tk.LEFT, padx=(0, 10))
        
        # Convert BGR to RGB for display
        rgb_color = f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"
        color_rect = color_display.create_rectangle(2, 2, 28, 18, fill=rgb_color, outline="")
        
        # Color button
        def choose_color():
            # Convert current BGR to RGB for color chooser
            init_color = (color[2], color[1], color[0])
            rgb = colorchooser.askcolor(initialcolor=init_color, parent=self.dialog)
            if rgb[0]:
                # Convert back to BGR
                new_color = (int(rgb[0][2]), int(rgb[0][1]), int(rgb[0][0]))
                self.category_widgets[category]['color'] = new_color
                # Update display
                rgb_hex = f"#{int(rgb[0][0]):02x}{int(rgb[0][1]):02x}{int(rgb[0][2]):02x}"
                color_display.itemconfig(color_rect, fill=rgb_hex)
                
        color_button = self.gui_manager.create_modern_button(
            frame,
            text=DialogHelper.t("Choose Color"),
            color=self.gui_manager.theme_colors["accent_blue"],
            command=choose_color
        )
        color_button.pack(side=tk.LEFT)
        
        # Note: Eyedropper tool would require significant platform-specific code
        # or external libraries. The built-in colorchooser provides good functionality
        # for color selection without the added complexity.
        
        # Store widget references
        self.category_widgets[category] = {
            'frame': frame,
            'color': color,
            'display': color_display,
            'rect': color_rect
        }
        
    def _add_range_selector(self, min_val: float = 0, max_val: float = 100, 
                           color: Tuple[int, int, int] = None, label: str = ""):
        """Add a numeric range selector."""
        frame = ttk.Frame(self.config_frame)
        frame.pack(fill=tk.X, pady=2)
        
        # Min value
        ttk.Label(frame, text=DialogHelper.t("Min:")).pack(side=tk.LEFT, padx=(0, 5))
        min_var = tk.DoubleVar(value=min_val)
        min_entry = self.gui_manager.create_entry_with_validation(
            frame, min_var, width=10
        )
        min_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        # Max value
        ttk.Label(frame, text=DialogHelper.t("Max:")).pack(side=tk.LEFT, padx=(0, 5))
        max_var = tk.DoubleVar(value=max_val)
        max_entry = self.gui_manager.create_entry_with_validation(
            frame, max_var, width=10
        )
        max_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        # Label
        ttk.Label(frame, text=DialogHelper.t("Label:")).pack(side=tk.LEFT, padx=(0, 5))
        label_var = tk.StringVar(value=label)
        label_entry = ttk.Entry(frame, textvariable=label_var, width=15)
        label_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        # Color display
        color = color or (200, 200, 200)
        color_display = tk.Canvas(frame, width=30, height=20,
                                 highlightthickness=1,
                                 highlightbackground=self.gui_manager.theme_colors["border"])
        color_display.pack(side=tk.LEFT, padx=(0, 10))
        
        # Convert BGR to RGB for display
        rgb_color = f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"
        color_rect = color_display.create_rectangle(2, 2, 28, 18, fill=rgb_color, outline="")
        
        # Color button
        def choose_color():
            init_color = (color[2], color[1], color[0])
            rgb = colorchooser.askcolor(initialcolor=init_color, parent=self.dialog)
            if rgb[0]:
                new_color = (int(rgb[0][2]), int(rgb[0][1]), int(rgb[0][0]))
                widget_info['color'] = new_color
                rgb_hex = f"#{int(rgb[0][0]):02x}{int(rgb[0][1]):02x}{int(rgb[0][2]):02x}"
                color_display.itemconfig(color_rect, fill=rgb_hex)
                
        color_button = self.gui_manager.create_modern_button(
            frame,
            text=DialogHelper.t("Color"),
            color=self.gui_manager.theme_colors["accent_blue"],
            command=choose_color
        )
        color_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Remove button
        def remove_range():
            frame.destroy()
            self.range_widgets.remove(widget_info)
            
        remove_button = self.gui_manager.create_modern_button(
            frame,
            text="✖",
            color=self.gui_manager.theme_colors["accent_red"],
            command=remove_range
        )
        remove_button.pack(side=tk.LEFT)
        
        # Store widget info
        widget_info = {
            'frame': frame,
            'min_var': min_var,
            'max_var': max_var,
            'label_var': label_var,
            'color': color,
            'display': color_display,
            'rect': color_rect
        }
        self.range_widgets.append(widget_info)
        
    def _add_default_color_selector(self):
        """Add default and null color selectors."""
        # Separator
        ttk.Separator(self.config_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=20)
        
        # Default color
        default_frame = ttk.Frame(self.config_frame)
        default_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(
            default_frame,
            text=DialogHelper.t("Default Color (unmatched values):"),
            width=30
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        self.default_color = (200, 200, 200)
        # ===================================================
        # Store as instance attribute, not local variable
        # ===================================================
        self.default_display = tk.Canvas(default_frame, width=30, height=20,
                                   highlightthickness=1,
                                   highlightbackground=self.gui_manager.theme_colors["border"])
        self.default_display.pack(side=tk.LEFT, padx=(0, 10))
        
        rgb_color = f"#{self.default_color[2]:02x}{self.default_color[1]:02x}{self.default_color[0]:02x}"
        self.default_rect = self.default_display.create_rectangle(2, 2, 28, 18, fill=rgb_color, outline="")



        def choose_default():
            init_color = (self.default_color[2], self.default_color[1], self.default_color[0])
            rgb = colorchooser.askcolor(initialcolor=init_color, parent=self.dialog)
            if rgb[0]:
                self.default_color = (int(rgb[0][2]), int(rgb[0][1]), int(rgb[0][0]))
                rgb_hex = f"#{int(rgb[0][0]):02x}{int(rgb[0][1]):02x}{int(rgb[0][2]):02x}"
                self.default_display.itemconfig(self.default_rect, fill=rgb_hex)
                
        default_button = self.gui_manager.create_modern_button(
            default_frame,
            text=DialogHelper.t("Choose Color"),
            color=self.gui_manager.theme_colors["accent_blue"],
            command=choose_default
        )
        default_button.pack(side=tk.LEFT)
        
        # Null color
        null_frame = ttk.Frame(self.config_frame)
        null_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(
            null_frame,
            text=DialogHelper.t("Null/Missing Value Color:"),
            width=30
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        self.null_color = (255, 255, 255)
        self.null_display = tk.Canvas(null_frame, width=30, height=20,
                                    highlightthickness=1,
                                    highlightbackground=self.gui_manager.theme_colors["border"])
        self.null_display.pack(side=tk.LEFT, padx=(0, 10))
        
        rgb_color = f"#{self.null_color[2]:02x}{self.null_color[1]:02x}{self.null_color[0]:02x}"
        self.null_rect = self.null_display.create_rectangle(2, 2, 28, 18, fill=rgb_color, outline="")
        
        def choose_null():
            init_color = (self.null_color[2], self.null_color[1], self.null_color[0])
            rgb = colorchooser.askcolor(initialcolor=init_color, parent=self.dialog)
            if rgb[0]:
                self.null_color = (int(rgb[0][2]), int(rgb[0][1]), int(rgb[0][0]))
                rgb_hex = f"#{int(rgb[0][0]):02x}{int(rgb[0][1]):02x}{int(rgb[0][2]):02x}"
                self.null_display.itemconfig(self.null_rect, fill=rgb_hex)
                
        null_button = self.gui_manager.create_modern_button(
            null_frame,
            text=DialogHelper.t("Choose Color"),
            color=self.gui_manager.theme_colors["accent_blue"],
            command=choose_null
        )
        null_button.pack(side=tk.LEFT)
        
    def _add_new_category(self):
        """Add a new category."""
        # Simple dialog to get category name
        dialog = tk.Toplevel(self.dialog)
        dialog.title(DialogHelper.t("New Category"))
        dialog.transient(self.dialog)
        dialog.grab_set()
        
        # Apply theme
        dialog.configure(bg=self.gui_manager.theme_colors["background"])
        
        # Content frame
        content_frame = ttk.Frame(dialog, padding="20")
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(content_frame, text=DialogHelper.t("Category Name:")).pack(anchor=tk.W)
        
        name_var = tk.StringVar()
        entry = ttk.Entry(content_frame, textvariable=name_var, width=30)
        entry.pack(fill=tk.X, pady=(5, 15))
        entry.focus()
        
        def on_ok():
            category = name_var.get().strip()
            if category and category not in self.category_widgets:
                self._add_category_selector(category)
            dialog.destroy()
            
        def on_cancel():
            dialog.destroy()
            
        button_frame = ttk.Frame(content_frame)
        button_frame.pack()
        
        ok_button = self.gui_manager.create_modern_button(
            button_frame,
            text=DialogHelper.t("OK"),
            color=self.gui_manager.theme_colors["accent_green"],
            command=on_ok
        )
        ok_button.pack(side=tk.LEFT, padx=(0, 10))
        
        cancel_button = self.gui_manager.create_modern_button(
            button_frame,
            text=DialogHelper.t("Cancel"),
            color=self.gui_manager.theme_colors["accent_red"],
            command=on_cancel
        )
        cancel_button.pack(side=tk.LEFT)
        
        dialog.bind("<Return>", lambda e: on_ok())
        dialog.bind("<Escape>", lambda e: on_cancel())
        
        # Center dialog on parent
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() - dialog.winfo_width()) // 2
        y = (dialog.winfo_screenheight() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
        
    def _create_button_section(self, parent):
        """Create the button section."""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Save as preset
        save_preset_button = self.gui_manager.create_modern_button(
            button_frame,
            text=DialogHelper.t("Save as Preset"),
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._save_as_preset
        )
        save_preset_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Right-side buttons
        right_frame = ttk.Frame(button_frame)
        right_frame.pack(side=tk.RIGHT)
        
        # OK button
        ok_button = self.gui_manager.create_modern_button(
            right_frame,
            text=DialogHelper.t("OK"),
            color=self.gui_manager.theme_colors["accent_green"],
            command=self._on_ok
        )
        ok_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Cancel button
        cancel_button = self.gui_manager.create_modern_button(
            right_frame,
            text=DialogHelper.t("Cancel"),
            color=self.gui_manager.theme_colors["accent_red"],
            command=self.dialog.destroy
        )
        cancel_button.pack(side=tk.LEFT)
        
    def _on_type_change(self, *args):
        """Handle color map type change."""
        map_type = ColorMapType(self.type_var.get())
        
        if map_type == ColorMapType.CATEGORICAL:
            self._create_categorical_config()
        elif map_type == ColorMapType.NUMERIC:
            self._create_numeric_config()
            
    def _load_preset(self):
        """Load selected preset."""
        preset_name = self.preset_var.get()
        if preset_name == "Custom":
            return
            
        preset = self.color_map_manager.get_preset(preset_name)
        if preset:
            # ===================================================
            # First, trigger the type change to update UI
            # ===================================================
            self.type_var.set(preset.type.value)
            
            # Force UI update for the correct type
            if preset.type == ColorMapType.CATEGORICAL:
                self._create_categorical_config()
            elif preset.type == ColorMapType.NUMERIC:
                self._create_numeric_config()
                
            # Now load the color map data
            self._load_color_map(preset)

    def _load_color_map(self, color_map: ColorMap):
        """Load a color map into the UI."""
        # Set type
        self.type_var.set(color_map.type.value)
        
        # Set default colors
        self.default_color = color_map.default_color
        self.null_color = color_map.null_color
        
        # ===================================================
        # Update the default/null color displays
        # ===================================================
        if hasattr(self, 'default_rect'):
            rgb_hex = f"#{self.default_color[2]:02x}{self.default_color[1]:02x}{self.default_color[0]:02x}"
            self.default_display.itemconfig(self.default_rect, fill=rgb_hex)
            
        if hasattr(self, 'null_rect'):
            rgb_hex = f"#{self.null_color[2]:02x}{self.null_color[1]:02x}{self.null_color[0]:02x}"
            self.null_display.itemconfig(self.null_rect, fill=rgb_hex)
        
        if color_map.type == ColorMapType.CATEGORICAL:
            # Clear existing categories first
            for widget_info in list(self.category_widgets.values()):
                widget_info['frame'].destroy()
            self.category_widgets.clear()
            
            # Load categories (ColorMapManager already puts them in categories attribute)
            for category, color in color_map.categories.items():
                self._add_category_selector(category, color)
                    
        elif color_map.type == ColorMapType.NUMERIC:
            # Clear existing ranges first
            for widget_info in self.range_widgets[:]:
                widget_info['frame'].destroy()
            self.range_widgets.clear()
            
            # Load ranges
            for range_obj in color_map.ranges:
                self._add_range_selector(
                    range_obj.min,
                    range_obj.max,
                    range_obj.color,
                    range_obj.label
                )
                
    def _save_as_preset(self):
        """Save current configuration as a preset."""
        # Get preset name
        dialog = tk.Toplevel(self.dialog)
        dialog.title(DialogHelper.t("Save Preset"))
        dialog.transient(self.dialog)
        dialog.grab_set()
        
        # Apply theme
        dialog.configure(bg=self.gui_manager.theme_colors["background"])
        
        # Content frame
        content_frame = ttk.Frame(dialog, padding="20")
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(content_frame, text=DialogHelper.t("Preset Name:")).pack(anchor=tk.W)
        
        name_var = tk.StringVar()
        entry = ttk.Entry(content_frame, textvariable=name_var, width=30)
        entry.pack(fill=tk.X, pady=(5, 15))
        
        ttk.Label(content_frame, text=DialogHelper.t("Description:")).pack(anchor=tk.W)
        
        # Text widget with theme colors and border
        desc_text = tk.Text(
            content_frame, 
            width=40, 
            height=3,
            bg=self.gui_manager.theme_colors["field_bg"],
            fg=self.gui_manager.theme_colors["text"],
            insertbackground=self.gui_manager.theme_colors["text"],
            highlightbackground=self.gui_manager.theme_colors["field_border"],
            highlightcolor=self.gui_manager.theme_colors["accent_blue"],
            highlightthickness=1,
            bd=0
        )
        desc_text.pack(fill=tk.X, pady=(5, 15))
        
        def on_ok():
            name = name_var.get().strip()
            if name:
                color_map = self._build_color_map()
                color_map.name = name
                color_map.description = desc_text.get("1.0", tk.END).strip()
                
                if self.color_map_manager.save_preset(name, color_map):
                    # Update preset dropdown
                    preset_names = ["Custom"] + self.color_map_manager.get_preset_names()
                    self.preset_var.set(name)
                    
                    # Update dropdown with new preset list
                    self.preset_dropdown.destroy()
                    self.preset_dropdown = tk.OptionMenu(
                        self.preset_combo_frame,
                        self.preset_var,
                        *preset_names
                    )
                    self.gui_manager.style_dropdown(self.preset_dropdown, width=30)
                    self.preset_dropdown.pack()
                    
                    DialogHelper.show_message(
                        dialog,
                        DialogHelper.t("Success"),
                        DialogHelper.t("Preset saved successfully"),
                        message_type="info"
                    )
                    
            dialog.destroy()
            
        button_frame = ttk.Frame(content_frame)
        button_frame.pack()
        
        ok_button = self.gui_manager.create_modern_button(
            button_frame,
            text=DialogHelper.t("OK"),
            color=self.gui_manager.theme_colors["accent_green"],
            command=on_ok
        )
        ok_button.pack(side=tk.LEFT, padx=(0, 10))
        
        cancel_button = self.gui_manager.create_modern_button(
            button_frame,
            text=DialogHelper.t("Cancel"),
            color=self.gui_manager.theme_colors["accent_red"],
            command=dialog.destroy
        )
        cancel_button.pack(side=tk.LEFT)
        
        # Center dialog on parent
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() - dialog.winfo_width()) // 2
        y = (dialog.winfo_screenheight() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
                  
    def _build_color_map(self) -> ColorMap:
        """Build color map from current UI state."""
        map_type = ColorMapType(self.type_var.get())
        color_map = ColorMap("Custom", map_type)
        
        color_map.default_color = self.default_color
        color_map.null_color = self.null_color
        
        if map_type == ColorMapType.CATEGORICAL:
            # Get categories
            for category, widget_info in self.category_widgets.items():
                color_map.add_category(category, widget_info['color'])
                
        elif map_type == ColorMapType.NUMERIC:
            # Get ranges
            for widget_info in self.range_widgets:
                try:
                    color_range = ColorRange(
                        widget_info['min_var'].get(),
                        widget_info['max_var'].get(),
                        widget_info['color'],
                        widget_info['label_var'].get()
                    )
                    color_map.add_range(color_range)
                except:
                    pass  # Skip invalid ranges
                    
        return color_map
        
    def _on_ok(self):
        """Handle OK button."""
        self.result = self._build_color_map()
        self.dialog.destroy()
        
    def show(self) -> Optional[ColorMap]:
        """
        Show the dialog and return the configured color map.
        
        Returns:
            ColorMap or None if cancelled
        """
        self.dialog.wait_window()
        return self.result