# # gui/widgets/viz_column_item_row.py

# """
# Expandable row widget for managing visualization column configurations.
# Each row is self-contained with inline editing capabilities.
# """

# import tkinter as tk
# from tkinter import ttk
# import logging
# from typing import Dict, Any, Callable, Optional

# from gui.widgets.modern_button import ModernButton


# class VizColumnItemRow:
#     """
#     A self-contained, expandable row for a visualization column configuration.
#     Header always visible, details expand/collapse on demand.
#     """

#     def __init__(
#         self,
#         parent: tk.Widget,
#         config: Dict[str, Any],
#         gui_manager,
#         available_columns: list,
#         available_presets: list,
#         callbacks: Dict[str, Callable],
#         csv_handler=None,
#         color_map_manager=None,
#         drillhole_data_manager=None
#     ):
#         """
#         Initialize expandable row

#         Args:
#             parent: Parent frame
#             config: Configuration dict with keys:
#                 - column: str - column name
#                 - color_map: str - color map name
#                 - column_size: int - width in pixels
#                 - custom_label: str - display label
#                 - font_size: int - value font size
#                 - bold: bool - value font bold
#                 - decimal_places: int - decimal places for numeric values
#                 - header_bold: bool - header font bold
#                 - header_underline: bool - header underline
#                 - header_size: int - header font size
#                 - pack_remaining_space: bool - fill remaining width
#             gui_manager: GUIManager for theming
#             available_columns: List of available column names
#             available_presets: List of available color map presets
#             callbacks: Dict with {
#                 'on_update': func(row_id, changes),
#                 'on_remove': func(row_id),
#                 'on_select': func(row_id),
#                 'on_edit_color_map': func(color_map_name, column_name),
#                 'on_manage_color_maps': func(),
#                 'on_preview_update': func()
#             }
#             csv_handler: CSV data handler (optional)
#             color_map_manager: Color map manager
#             drillhole_data_manager: Drillhole data manager
#         """
#         self.parent = parent
#         self.config = config
#         self.gui_manager = gui_manager
#         self.theme = gui_manager.theme_colors
#         self.fonts = gui_manager.fonts
#         self.available_columns = available_columns
#         self.available_presets = available_presets
#         self.callbacks = callbacks
#         self.csv_handler = csv_handler
#         self.color_map_manager = color_map_manager
#         self.drillhole_data_manager = drillhole_data_manager
        
#         self.is_expanded = False
#         self.is_selected = False
        
#         # Generate unique ID
#         self.row_id = id(self)
        
#         self.logger = logging.getLogger(__name__)
        
#         # Form variables
#         self.column_var = tk.StringVar(value=config.get("column", ""))
#         self.label_var = tk.StringVar(value=config.get("custom_label", ""))
#         self.data_type_var = tk.StringVar(value=config.get("data_type", "auto"))
#         self.column_size_var = tk.IntVar(value=config.get("column_size", 80))
#         self.color_map_var = tk.StringVar(value=config.get("color_map", ""))
#         self.font_size_var = tk.IntVar(value=config.get("font_size", 8))
#         self.bold_var = tk.BooleanVar(value=config.get("bold", False))
#         self.decimal_var = tk.IntVar(value=config.get("decimal_places", 2))
#         self.header_size_var = tk.IntVar(value=config.get("header_size", 8))
#         self.header_bold_var = tk.BooleanVar(value=config.get("header_bold", True))
#         self.header_underline_var = tk.BooleanVar(value=config.get("header_underline", False))
#         self.pack_remaining_var = tk.BooleanVar(value=config.get("pack_remaining_space", False))
        
#         self._create_widgets()

#     def _create_widgets(self):
#         """Create the row structure"""
#         # Main container frame
#         self.main_frame = tk.Frame(
#             self.parent,
#             bg=self.theme["secondary_bg"]
#         )
#         self.main_frame.pack(fill=tk.X, pady=2, padx=5)

#         # Create header (always visible)
#         self._create_header()

#         # Create details section (hidden by default)
#         self._create_details()

#     def _create_header(self):
#         """Create the always-visible header bar"""
#         # Header frame with border for selection highlighting
#         self.header_frame = tk.Frame(
#             self.main_frame,
#             bg=self.theme["secondary_bg"],
#             highlightthickness=2,
#             highlightbackground=self.theme["border"],
#             cursor="hand2",
#         )
#         self.header_frame.pack(fill=tk.X)

#         # Color bar preview (small colored rectangle)
#         self.color_canvas = tk.Canvas(
#             self.header_frame,
#             width=40,
#             height=20,
#             bg="#888888",  # Default gray
#             highlightthickness=1,
#             highlightbackground=self.theme["border"],
#             cursor="hand2",
#         )
#         self.color_canvas.pack(side=tk.LEFT, padx=(5, 8))
        
#         # Draw a sample gradient/color on the canvas
#         self._update_color_preview()

#         # Label text
#         label_text = self._get_display_label()
#         self.label_widget = tk.Label(
#             self.header_frame,
#             text=label_text,
#             bg=self.theme["secondary_bg"],
#             fg=self.theme["text"],
#             font=("Arial", 10, "normal"),
#             anchor="w",
#             width=20,
#             cursor="hand2",
#         )
#         self.label_widget.pack(side=tk.LEFT, padx=(0, 10))

#         # Size badge
#         size_text = f"{self.config.get('column_size', 80)}px"
#         if self.config.get("pack_remaining_space", False):
#             size_text = "FLEX"
        
#         self.size_badge = tk.Label(
#             self.header_frame,
#             text=size_text,
#             bg=self.theme.get("accent_blue", "#007bff"),
#             fg="white",
#             font=("Arial", 9, "bold"),
#             padx=4,
#             pady=2,
#             cursor="hand2",
#         )
#         self.size_badge.pack(side=tk.LEFT, padx=(0, 8))

#         # Type badge
#         type_text = self._get_type_badge_text()
#         type_color = self._get_type_badge_color()
        
#         self.type_badge = tk.Label(
#             self.header_frame,
#             text=type_text,
#             bg=type_color,
#             fg="white",
#             font=("Arial", 8, "bold"),
#             padx=5,
#             pady=2,
#             cursor="hand2",
#         )
#         self.type_badge.pack(side=tk.LEFT, padx=(0, 8))

#         # Spacer to push expand button to right
#         spacer = tk.Frame(self.header_frame, bg=self.theme["secondary_bg"])
#         spacer.pack(side=tk.LEFT, fill=tk.X, expand=True)

#         # Expand/collapse button
#         expand_text = "▲" if self.is_expanded else "▼"
#         self.expand_button = tk.Label(
#             self.header_frame,
#             text=expand_text,
#             bg=self.theme["secondary_bg"],
#             fg=self.theme["text"],
#             font=("Arial", 12, "bold"),
#             padx=10,
#             pady=5,
#             cursor="hand2",
#         )
#         self.expand_button.pack(side=tk.RIGHT, padx=5)
#         self.expand_button.bind("<Button-1>", lambda e: self.toggle_expand())

#         # Bind header clicks for selection
#         self._bind_header_click(self.header_frame)
#         self._bind_header_click(self.color_canvas)
#         self._bind_header_click(self.label_widget)
#         self._bind_header_click(self.size_badge)
#         self._bind_header_click(self.type_badge)

#     def _bind_header_click(self, widget):
#         """Bind click event to widget for row selection"""
#         widget.bind("<Button-1>", lambda e: self._on_select())

#     def _on_select(self):
#         """Handle row selection"""
#         if self.callbacks.get("on_select"):
#             self.callbacks["on_select"](self.row_id)

#     def _create_details(self):
#         """Create the expandable details section with all editing controls"""
#         # Details frame (hidden by default)
#         self.details_frame = tk.Frame(
#             self.main_frame,
#             bg=self.theme["background"],
#             highlightthickness=1,
#             highlightbackground=self.theme["border"],
#         )

#         # Inner padding frame
#         inner_frame = tk.Frame(self.details_frame, bg=self.theme["background"])
#         inner_frame.pack(fill=tk.X, padx=10, pady=10)

#         # Configure grid for proper layout
#         inner_frame.columnconfigure(1, weight=1)  # Column dropdown
#         inner_frame.columnconfigure(3, weight=1)  # Label entry
#         inner_frame.columnconfigure(5, weight=1)  # Type dropdown
#         inner_frame.columnconfigure(8, weight=1)  # Color map dropdown

#         # ROW 0: Column, Label, Type, Size
#         row = 0
        
#         # Column field
#         ttk.Label(inner_frame, text="Column:").grid(
#             row=row, column=0, sticky="w", padx=(0, 5)
#         )
        
#         special_columns = ["--- Spacer ---"]
#         all_columns = special_columns + self.available_columns
        
#         column_frame = tk.Frame(
#             inner_frame,
#             bg=self.theme["field_bg"],
#             highlightbackground=self.theme["field_border"],
#             highlightthickness=1,
#             bd=0
#         )
#         column_frame.grid(row=row, column=1, sticky="ew", padx=(0, 10))
        
#         self.column_dropdown = tk.OptionMenu(
#             column_frame, self.column_var, *all_columns,
#             command=self._on_column_change
#         )
#         self.gui_manager.style_dropdown(self.column_dropdown, width=20)
#         self.column_dropdown.pack()
        
#         # Label field
#         ttk.Label(inner_frame, text="Label:").grid(
#             row=row, column=2, sticky="w", padx=(0, 5)
#         )
        
#         self.label_entry = ttk.Entry(
#             inner_frame,
#             textvariable=self.label_var,
#             width=15
#         )
#         self.label_entry.grid(row=row, column=3, sticky="ew", padx=(0, 10))
        
#         # Type field
#         ttk.Label(inner_frame, text="Type:").grid(
#             row=row, column=4, sticky="w", padx=(0, 5)
#         )
        
#         type_frame = tk.Frame(
#             inner_frame,
#             bg=self.theme["field_bg"],
#             highlightbackground=self.theme["field_border"],
#             highlightthickness=1,
#             bd=0
#         )
#         type_frame.grid(row=row, column=5, sticky="ew", padx=(0, 10))
        
#         self.type_dropdown = tk.OptionMenu(
#             type_frame, self.data_type_var, "auto", "numeric", "categorical"
#         )
#         self.gui_manager.style_dropdown(self.type_dropdown, width=10)
#         self.type_dropdown.pack()
        
#         # Size field (individual dimension: height in portrait, width in landscape)
#         ttk.Label(inner_frame, text="Size:").grid(
#             row=row, column=6, sticky="w", padx=(0, 5)
#         )
        
#         size_spinbox = tk.Spinbox(
#             inner_frame,
#             from_=40,
#             to=200,
#             width=5,
#             textvariable=self.column_size_var,
#             bg=self.theme["field_bg"],
#             fg=self.theme["text"],
#             buttonbackground=self.theme["field_bg"],
#             font=self.fonts["normal"],
#             command=self._on_size_change
#         )
#         size_spinbox.grid(row=row, column=7, sticky="w")
#         self.column_size_var.trace_add("write", lambda *args: self._on_size_change())
        
#         ttk.Label(inner_frame, text="px").grid(
#             row=row, column=8, sticky="w", padx=(2, 5)
#         )
        
#         ttk.Label(
#             inner_frame,
#             text="(height in portrait, width in landscape)",
#             font=("Arial", 7, "italic"),
#             foreground="#888888"
#         ).grid(row=row, column=9, sticky="w", padx=(0, 0))

#         # ROW 1: Color Map and Pack Remaining Space
#         row += 1
        
#         ttk.Label(inner_frame, text="Color Map:").grid(
#             row=row, column=0, sticky="w", padx=(0, 5), pady=(10, 0)
#         )
        
#         preset_options = ["Create New..."] + self.available_presets
        
#         color_map_frame = tk.Frame(
#             inner_frame,
#             bg=self.theme["field_bg"],
#             highlightbackground=self.theme["field_border"],
#             highlightthickness=1,
#             bd=0
#         )
#         color_map_frame.grid(row=row, column=1, columnspan=3, sticky="ew", padx=(0, 10), pady=(10, 0))
        
#         self.color_map_dropdown = tk.OptionMenu(
#             color_map_frame, self.color_map_var, *preset_options,
#             command=self._on_color_map_change
#         )
#         self.gui_manager.style_dropdown(self.color_map_dropdown, width=20)
#         self.color_map_dropdown.pack()
        
#         # Edit button
#         ModernButton(
#             inner_frame,
#             text="✏️ Edit",
#             command=self._edit_current_color_map,
#             color=self.theme["accent_blue"],
#             theme_colors=self.theme,
#             width=8
#         ).grid(row=row, column=4, padx=2, pady=(10, 0))
        
#         # Manage button
#         ModernButton(
#             inner_frame,
#             text="🎨 Manage",
#             command=self._manage_color_maps,
#             color=self.theme["accent_blue"],
#             theme_colors=self.theme,
#             width=10
#         ).grid(row=row, column=5, columnspan=2, padx=2, pady=(10, 0))
        
#         # Pack remaining space checkbox
#         pack_check = tk.Checkbutton(
#             inner_frame,
#             text="📦 Pack Remaining Space",
#             variable=self.pack_remaining_var,
#             bg=self.theme["background"],
#             fg=self.theme["text"],
#             selectcolor=self.theme["field_bg"],
#             activebackground=self.theme["background"],
#             activeforeground=self.theme["text"],
#             font=self.fonts["normal"],
#             command=self._on_pack_remaining_change
#         )
#         pack_check.grid(row=row, column=7, columnspan=2, sticky="w", pady=(10, 0), padx=(10, 0))

#         # ROW 2: Value formatting
#         row += 1
        
#         ttk.Label(inner_frame, text="Value Font:").grid(
#             row=row, column=0, sticky="w", padx=(0, 5), pady=(10, 0)
#         )
        
#         font_spinbox = tk.Spinbox(
#             inner_frame,
#             from_=6,
#             to=20,
#             width=3,
#             textvariable=self.font_size_var,
#             bg=self.theme["field_bg"],
#             fg=self.theme["text"],
#             buttonbackground=self.theme["field_bg"],
#             font=self.fonts["normal"]
#         )
#         font_spinbox.grid(row=row, column=1, sticky="w", pady=(10, 0))
        
#         ttk.Label(inner_frame, text="pt").grid(
#             row=row, column=2, sticky="w", padx=(2, 10), pady=(10, 0)
#         )
        
#         # Bold checkbox
#         bold_check = tk.Checkbutton(
#             inner_frame,
#             text="Bold",
#             variable=self.bold_var,
#             bg=self.theme["background"],
#             fg=self.theme["text"],
#             selectcolor=self.theme["field_bg"],
#             activebackground=self.theme["background"],
#             activeforeground=self.theme["text"],
#             font=self.fonts["normal"]
#         )
#         bold_check.grid(row=row, column=3, sticky="w", pady=(10, 0))
        
#         # Decimal places
#         ttk.Label(inner_frame, text="Decimals:").grid(
#             row=row, column=4, sticky="w", padx=(0, 5), pady=(10, 0)
#         )
        
#         decimal_spinbox = tk.Spinbox(
#             inner_frame,
#             from_=0,
#             to=6,
#             width=3,
#             textvariable=self.decimal_var,
#             bg=self.theme["field_bg"],
#             fg=self.theme["text"],
#             buttonbackground=self.theme["field_bg"],
#             font=self.fonts["normal"]
#         )
#         decimal_spinbox.grid(row=row, column=5, sticky="w", pady=(10, 0))

#         # ROW 3: Header formatting
#         row += 1
        
#         ttk.Label(inner_frame, text="Header Font:").grid(
#             row=row, column=0, sticky="w", padx=(0, 5), pady=(10, 0)
#         )
        
#         header_font_spinbox = tk.Spinbox(
#             inner_frame,
#             from_=6,
#             to=20,
#             width=3,
#             textvariable=self.header_size_var,
#             bg=self.theme["field_bg"],
#             fg=self.theme["text"],
#             buttonbackground=self.theme["field_bg"],
#             font=self.fonts["normal"]
#         )
#         header_font_spinbox.grid(row=row, column=1, sticky="w", pady=(10, 0))
        
#         ttk.Label(inner_frame, text="pt").grid(
#             row=row, column=2, sticky="w", padx=(2, 10), pady=(10, 0)
#         )
        
#         # Header bold checkbox
#         header_bold_check = tk.Checkbutton(
#             inner_frame,
#             text="Bold",
#             variable=self.header_bold_var,
#             bg=self.theme["background"],
#             fg=self.theme["text"],
#             selectcolor=self.theme["field_bg"],
#             activebackground=self.theme["background"],
#             activeforeground=self.theme["text"],
#             font=self.fonts["normal"]
#         )
#         header_bold_check.grid(row=row, column=3, sticky="w", pady=(10, 0))
        
#         # Header underline checkbox
#         header_underline_check = tk.Checkbutton(
#             inner_frame,
#             text="Underline",
#             variable=self.header_underline_var,
#             bg=self.theme["background"],
#             fg=self.theme["text"],
#             selectcolor=self.theme["field_bg"],
#             activebackground=self.theme["background"],
#             activeforeground=self.theme["text"],
#             font=self.fonts["normal"]
#         )
#         header_underline_check.grid(row=row, column=4, columnspan=2, sticky="w", pady=(10, 0))

#         # ROW 4: Buttons
#         row += 1
        
#         button_frame = tk.Frame(inner_frame, bg=self.theme["background"])
#         button_frame.grid(row=row, column=0, columnspan=9, sticky="ew", pady=(15, 0))
        
#         # Remove button
#         ModernButton(
#             button_frame,
#             text="🗑️ Remove Column",
#             command=self._on_remove,
#             color=self.theme["accent_red"],
#             theme_colors=self.theme
#         ).pack(side=tk.RIGHT, padx=2)
        
#         # Update button (triggers preview refresh)
#         ModernButton(
#             button_frame,
#             text="💾 Update",
#             command=self._on_update,
#             color=self.theme["accent_green"],
#             theme_colors=self.theme
#         ).pack(side=tk.RIGHT, padx=2)

#     def toggle_expand(self):
#         """Toggle expansion of details section"""
#         self.is_expanded = not self.is_expanded

#         if self.is_expanded:
#             self.details_frame.pack(fill=tk.X, pady=(0, 2))
#             self.expand_button.config(text="▲")
#         else:
#             self.details_frame.pack_forget()
#             self.expand_button.config(text="▼")

#     def set_selected(self, selected: bool):
#         """Update visual selection state"""
#         self.is_selected = selected

#         if selected:
#             # Highlight border
#             self.header_frame.config(
#                 highlightbackground=self.theme.get("accent_blue", "#007bff"),
#                 highlightthickness=3,
#             )
#             # Slightly tint background
#             highlight_bg = self.theme.get("hover_highlight", "#404040")
#             self.header_frame.config(bg=highlight_bg)
#             # Update all header widgets background
#             for widget in [self.label_widget, self.size_badge, self.type_badge]:
#                 if hasattr(widget, "config"):
#                     widget.config(bg=highlight_bg)
#         else:
#             # Reset to normal
#             self.header_frame.config(
#                 highlightbackground=self.theme["border"], 
#                 highlightthickness=2
#             )
#             normal_bg = self.theme["secondary_bg"]
#             self.header_frame.config(bg=normal_bg)
#             for widget in [self.label_widget, self.size_badge, self.type_badge]:
#                 if hasattr(widget, "config"):
#                     widget.config(bg=normal_bg)

#     def _get_display_label(self) -> str:
#         """Get the display label for the header"""
#         custom_label = self.config.get("custom_label", "")
#         column_name = self.config.get("column", "")
        
#         if custom_label:
#             return custom_label
#         elif column_name == "--- Spacer ---":
#             return "--- Spacer ---"
#         else:
#             return column_name

#     def _get_type_badge_text(self) -> str:
#         """Get badge text for column type"""
#         column_name = self.config.get("column", "")
#         if column_name == "--- Spacer ---":
#             return "SPACER"
        
#         data_type = self.config.get("data_type", "auto")
#         if data_type == "numeric":
#             return "NUM"
#         elif data_type == "categorical":
#             return "CAT"
#         else:
#             return "AUTO"

#     def _get_type_badge_color(self) -> str:
#         """Get badge color for column type"""
#         column_name = self.config.get("column", "")
#         if column_name == "--- Spacer ---":
#             return "#6c757d"  # Gray
        
#         data_type = self.config.get("data_type", "auto")
#         if data_type == "numeric":
#             return self.theme.get("accent_green", "#28a745")
#         elif data_type == "categorical":
#             return self.theme.get("accent_blue", "#007bff")
#         else:
#             return "#6c757d"  # Gray

#     def _update_color_preview(self):
#         """Update the color bar preview in the header"""
#         color_map_name = self.config.get("color_map", "")
        
#         # Clear canvas
#         self.color_canvas.delete("all")
        
#         if not color_map_name or not self.color_map_manager:
#             # Draw gray placeholder
#             self.color_canvas.create_rectangle(
#                 0, 0, 40, 20,
#                 fill="#888888",
#                 outline=""
#             )
#             return
        
#         try:
#             # Get color map via ColorMapStore (wraps ColorMapManager)
#             color_map = None
#             if hasattr(self.color_map_manager, 'get'):
#                 # ColorMapStore.get() method
#                 color_map = self.color_map_manager.get(color_map_name)
#             elif hasattr(self.color_map_manager, 'get_preset'):
#                 # ColorMapManager.get_preset() method
#                 color_map = self.color_map_manager.get_preset(color_map_name)
            
#             if not color_map:
#                 self.color_canvas.create_rectangle(
#                     0, 0, 40, 20,
#                     fill="#888888",
#                     outline=""
#                 )
#                 return
            
#             # Get ranges from ColorMap object
#             ranges = color_map.ranges if hasattr(color_map, 'ranges') else []
#             if not ranges:
#                 self.color_canvas.create_rectangle(
#                     0, 0, 40, 20,
#                     fill="#888888",
#                     outline=""
#                 )
#                 return
            
#             # Draw color bars for each range
#             num_ranges = len(ranges)
#             bar_width = 40 / num_ranges if num_ranges > 0 else 40
            
#             for i, range_def in enumerate(ranges):
#                 color = range_def.get("color", "#888888")
#                 x1 = int(i * bar_width)
#                 x2 = int((i + 1) * bar_width)
                
#                 self.color_canvas.create_rectangle(
#                     x1, 0, x2, 20,
#                     fill=color,
#                     outline=""
#                 )
#         except Exception as e:
#             self.logger.error(f"Error updating color preview: {e}")
#             self.color_canvas.create_rectangle(
#                 0, 0, 40, 20,
#                 fill="#888888",
#                 outline=""
#             )

#     def _on_column_change(self, *args):
#         """Handle column selection change"""
#         # Auto-populate label if empty
#         if not self.label_var.get():
#             self.label_var.set(self.column_var.get())
        
#         # Update header
#         self._update_header_display()

#     def _on_color_map_change(self, *args):
#         """Handle color map selection change"""
#         self._update_color_preview()

#     def _on_size_change(self, *args):
#         """Handle size change"""
#         self._update_header_display()

#     def _on_pack_remaining_change(self):
#         """Handle pack remaining space toggle"""
#         self._update_header_display()
        
#         # Notify parent to update preview
#         if self.callbacks.get("on_preview_update"):
#             self.callbacks["on_preview_update"]()

#     def _update_header_display(self):
#         """Update header display elements"""
#         # Update label
#         self.label_widget.config(text=self._get_display_label())
        
#         # Update size badge
#         if self.pack_remaining_var.get():
#             self.size_badge.config(text="FLEX")
#         else:
#             self.size_badge.config(text=f"{self.column_size_var.get()}px")
        
#         # Update type badge
#         self.type_badge.config(
#             text=self._get_type_badge_text(),
#             bg=self._get_type_badge_color()
#         )

#     def _edit_current_color_map(self):
#         """Edit the currently selected color map"""
#         color_map_name = self.color_map_var.get()
#         column_name = self.column_var.get()
        
#         if not color_map_name or color_map_name == "Create New...":
#             # Create new color map
#             if self.callbacks.get("on_edit_color_map"):
#                 self.callbacks["on_edit_color_map"](None, column_name)
#         else:
#             # Edit existing
#             if self.callbacks.get("on_edit_color_map"):
#                 self.callbacks["on_edit_color_map"](color_map_name, column_name)

#     def _manage_color_maps(self):
#         """Open color map management dialog"""
#         if self.callbacks.get("on_manage_color_maps"):
#             self.callbacks["on_manage_color_maps"]()

#     def _on_update(self):
#         """Handle update button click"""
#         # Gather changes
#         changes = self.get_config()
        
#         # Update config
#         self.config = changes
        
#         # Update displays
#         self._update_header_display()
#         self._update_color_preview()
        
#         # Notify parent
#         if self.callbacks.get("on_update"):
#             self.callbacks["on_update"](self.row_id, changes)

#     def _on_remove(self):
#         """Handle remove button click"""
#         if self.callbacks.get("on_remove"):
#             self.callbacks["on_remove"](self.row_id)

#     def get_config(self) -> Dict[str, Any]:
#         """Return current configuration dict"""
#         return {
#             "column": self.column_var.get(),
#             "color_map": self.color_map_var.get(),
#             "column_size": self.column_size_var.get(),
#             "custom_label": self.label_var.get(),
#             "data_type": self.data_type_var.get(),
#             "font_size": self.font_size_var.get(),
#             "bold": self.bold_var.get(),
#             "decimal_places": self.decimal_var.get(),
#             "header_size": self.header_size_var.get(),
#             "header_bold": self.header_bold_var.get(),
#             "header_underline": self.header_underline_var.get(),
#             "pack_remaining_space": self.pack_remaining_var.get()
#         }

#     def update_from_config(self, config: Dict[str, Any]):
#         """Update row from configuration dict"""
#         self.config = config
        
#         # Update form variables
#         self.column_var.set(config.get("column", ""))
#         self.label_var.set(config.get("custom_label", ""))
#         self.data_type_var.set(config.get("data_type", "auto"))
#         self.column_size_var.set(config.get("column_size", 80))
#         self.color_map_var.set(config.get("color_map", ""))
#         self.font_size_var.set(config.get("font_size", 8))
#         self.bold_var.set(config.get("bold", False))
#         self.decimal_var.set(config.get("decimal_places", 2))
#         self.header_size_var.set(config.get("header_size", 8))
#         self.header_bold_var.set(config.get("header_bold", True))
#         self.header_underline_var.set(config.get("header_underline", False))
#         self.pack_remaining_var.set(config.get("pack_remaining_space", False))
        
#         # Update displays
#         self._update_header_display()
#         self._update_color_preview()

#     def update_color_map_list(self, presets: list):
#         """Update the available color map presets"""
#         self.available_presets = presets
        
#         # Rebuild dropdown menu
#         menu = self.color_map_dropdown["menu"]
#         menu.delete(0, "end")
        
#         options = ["Create New..."] + presets
#         for option in options:
#             menu.add_command(
#                 label=option,
#                 command=lambda value=option: self.color_map_var.set(value)
#             )

#     def destroy(self):
#         """Destroy the row widget"""
#         if self.main_frame:
#             self.main_frame.destroy()


# gui/widgets/viz_column_item_row.py

"""
Simplified expandable row widget for managing visualization column configurations.
Each row has: Column selector, Color Map selector, Custom Label.
Font sizes are controlled globally in the parent dialog.
"""

import tkinter as tk
from tkinter import ttk
import logging
from typing import Dict, Any, Callable, Optional

from gui.widgets.modern_button import ModernButton


class VizColumnItemRow:
    """
    A simplified row for visualization column configuration.
    Contains: Column dropdown, Color Map dropdown, Custom Label entry.
    """

    def __init__(
        self,
        parent: tk.Widget,
        config: Dict[str, Any],
        gui_manager,
        available_columns: list,
        available_presets: list,
        callbacks: Dict[str, Callable],
        csv_handler=None,
        color_map_manager=None,
        drillhole_data_manager=None
    ):
        """
        Initialize row widget.

        Args:
            parent: Parent frame
            config: Configuration dict with keys:
                - column: str - column name
                - color_map: str - color map name
                - custom_label: str - display label
            gui_manager: GUIManager for theming
            available_columns: List of available column names
            available_presets: List of available color map presets
            callbacks: Dict with {
                'on_update': func(row_id, changes),
                'on_remove': func(row_id),
                'on_select': func(row_id),
                'on_edit_color_map': func(color_map_name, column_name),
                'on_manage_color_maps': func(),
                'on_preview_update': func()
            }
            csv_handler: Not used (legacy)
            color_map_manager: Color map manager
            drillhole_data_manager: Not used (legacy)
        """
        self.parent = parent
        self.config = config
        self.gui_manager = gui_manager
        self.theme = gui_manager.theme_colors
        self.fonts = gui_manager.fonts
        self.available_columns = available_columns
        self.available_presets = available_presets
        self.callbacks = callbacks
        self.color_map_manager = color_map_manager
        
        self.is_selected = False
        self.row_id = id(self)
        self.logger = logging.getLogger(__name__)
        
        # Form variables - simplified
        self.column_var = tk.StringVar(value=config.get("column", ""))
        self.label_var = tk.StringVar(value=config.get("custom_label", config.get("column", "")))
        self.color_map_var = tk.StringVar(value=config.get("color_map", ""))

        # Fallback sources for data coverage gaps
        self.fallback_sources = config.get("fallback_sources", [])
        self.fallback_chips = []  # Widgets for chip display

        self._create_widgets()

    def _create_widgets(self):
        """Create the simplified row structure - all inline, no expansion"""
        # Main container frame with selection border
        self.main_frame = tk.Frame(
            self.parent,
            bg=self.theme["secondary_bg"],
            highlightthickness=2,
            highlightbackground=self.theme["border"]
        )
        self.main_frame.pack(fill=tk.X, pady=2, padx=5)
        
        # Inner content frame
        content_frame = tk.Frame(self.main_frame, bg=self.theme["secondary_bg"])
        content_frame.pack(fill=tk.X, padx=8, pady=6)
        
        # Configure grid columns
        content_frame.columnconfigure(1, weight=1)  # Column dropdown expands
        content_frame.columnconfigure(3, weight=1)  # Color map dropdown expands
        content_frame.columnconfigure(5, weight=1)  # Label entry expands
        
        # Row 0: All controls inline
        row = 0
        col = 0
        
        # === Color preview canvas ===
        self.color_canvas = tk.Canvas(
            content_frame,
            width=30,
            height=20,
            bg="#888888",
            highlightthickness=1,
            highlightbackground=self.theme["border"],
            cursor="hand2"
        )
        self.color_canvas.grid(row=row, column=col, padx=(0, 8), sticky="w")
        self.color_canvas.bind("<Button-1>", lambda e: self._on_select())
        col += 1
        
        # === Column selector ===
        ttk.Label(content_frame, text="Column:").grid(
            row=row, column=col, sticky="w", padx=(0, 4)
        )
        col += 1
        
        special_columns = ["--- Spacer ---"]
        all_columns = special_columns + self.available_columns

        # Use searchable dropdown for column selection
        self.column_dropdown = self.gui_manager.create_searchable_optionmenu(
            parent=content_frame,
            items=all_columns,
            variable=self.column_var,
            width=22,
            placeholder="Search columns...",
            on_change=self._on_column_change,
            dropdown_mode="overlay",
        )
        self.column_dropdown.grid(row=row, column=col, sticky="ew", padx=(0, 12))
        col += 1
        
        # === Color Map selector ===
        ttk.Label(content_frame, text="Color Map:").grid(
            row=row, column=col, sticky="w", padx=(0, 4)
        )
        col += 1
        
        preset_options = self.available_presets if self.available_presets else ["(none)"]

        # Use searchable dropdown for color map selection
        self.color_map_dropdown = self.gui_manager.create_searchable_optionmenu(
            parent=content_frame,
            items=preset_options,
            variable=self.color_map_var,
            width=16,
            placeholder="Search maps...",
            on_change=self._on_color_map_change,
            dropdown_mode="overlay",
        )
        self.color_map_dropdown.grid(row=row, column=col, sticky="ew", padx=(0, 12))
        col += 1
        
        # === Custom Label ===
        ttk.Label(content_frame, text="Label:").grid(
            row=row, column=col, sticky="w", padx=(0, 4)
        )
        col += 1

        # Use tk.Entry with explicit theme colors for proper dark mode support
        self.label_entry = tk.Entry(
            content_frame,
            textvariable=self.label_var,
            width=12,
            bg=self.theme.get("field_bg", "#2d2d2d"),
            fg=self.theme.get("text", "#ffffff"),
            insertbackground=self.theme.get("text", "#ffffff"),
            highlightbackground=self.theme.get("field_border", "#555555"),
            highlightthickness=1,
            relief="flat"
        )
        self.label_entry.grid(row=row, column=col, sticky="ew", padx=(0, 12))
        self.label_entry.bind("<FocusOut>", lambda e: self._notify_update())
        self.label_entry.bind("<Return>", lambda e: self._notify_update())
        col += 1

        # === Edit Color Map button (small width for emoji) ===
        ModernButton(
            content_frame,
            text="✏️",
            command=self._edit_current_color_map,
            color=self.theme["accent_blue"],
            theme_colors=self.theme,
            width=2  # Small width for emoji button (in characters)
        ).grid(row=row, column=col, padx=(0, 4))
        col += 1

        # === Remove button (small width for emoji) ===
        ModernButton(
            content_frame,
            text="🗑️",
            command=self._on_remove,
            color=self.theme["accent_red"],
            theme_colors=self.theme,
            width=2  # Small width for emoji button (in characters)
        ).grid(row=row, column=col, padx=(0, 0))

        # === Row 1: Fallback sources ===
        row = 1

        ttk.Label(content_frame, text="Fallbacks:").grid(
            row=row, column=0, sticky="w", padx=(0, 4), pady=(4, 0)
        )

        # Frame to hold fallback chips
        self.fallback_frame = tk.Frame(content_frame, bg=self.theme["secondary_bg"])
        self.fallback_frame.grid(row=row, column=1, columnspan=5, sticky="w", padx=(0, 8), pady=(4, 0))

        # Render any existing fallback chips
        self._render_fallback_chips()

        # Add Fallback button
        ModernButton(
            content_frame,
            text="+ Add",
            command=self._add_fallback,
            color=self.theme.get("accent_green", "#4a8259"),
            theme_colors=self.theme,
            width=5
        ).grid(row=row, column=7, padx=(4, 0), pady=(4, 0), sticky="w")

        # Update color preview
        self._update_color_preview()
        
        # Bind selection on main frame click
        self.main_frame.bind("<Button-1>", lambda e: self._on_select())

    def _update_color_preview(self):
        """Update the color preview canvas with gradient from color map"""
        self.color_canvas.delete("all")
        
        color_map_name = self.color_map_var.get()
        if not color_map_name or not self.color_map_manager:
            # Default gray gradient
            self.color_canvas.create_rectangle(0, 0, 30, 20, fill="#888888", outline="")
            return
        
        try:
            # Try to get color map
            color_map = None
            if hasattr(self.color_map_manager, 'get_preset'):
                color_map = self.color_map_manager.get_preset(color_map_name)
            elif hasattr(self.color_map_manager, 'get'):
                color_map = self.color_map_manager.get(color_map_name)
            
            if color_map and hasattr(color_map, 'get_color_for_value'):
                # Draw gradient
                width = 30
                for x in range(width):
                    # Map x to 0-100 range
                    value = (x / width) * 100
                    bgr = color_map.get_color_for_value(value)
                    if bgr:
                        # Convert BGR to hex
                        hex_color = f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"
                        self.color_canvas.create_line(x, 0, x, 20, fill=hex_color)
            else:
                self.color_canvas.create_rectangle(0, 0, 30, 20, fill="#888888", outline="")
        except Exception as e:
            self.logger.debug(f"Could not render color preview: {e}")
            self.color_canvas.create_rectangle(0, 0, 30, 20, fill="#888888", outline="")

    def _on_column_change(self, *args):
        """Handle column selection change"""
        new_column = self.column_var.get()

        # Auto-update label if it matches old column or is empty
        current_label = self.label_var.get()
        if not current_label or current_label == self.config.get("column", ""):
            # Strip source suffix from label (e.g., "Fe_pct (assay)" -> "Fe_pct")
            label = new_column
            if new_column != "--- Spacer ---":
                # Remove " (source_name)" suffix if present
                if " (" in new_column and new_column.endswith(")"):
                    label = new_column.rsplit(" (", 1)[0]
            else:
                label = ""
            self.label_var.set(label)

        self.config["column"] = new_column
        self._notify_update()

    def _render_fallback_chips(self):
        """Render fallback source chips in the fallback frame"""
        # Clear existing chips
        for widget in self.fallback_chips:
            widget.destroy()
        self.fallback_chips.clear()

        # Create chip for each fallback source
        for idx, source in enumerate(self.fallback_sources):
            chip = self._create_chip(source, idx)
            self.fallback_chips.append(chip)

    def _create_chip(self, text, index):
        """Create a removable chip widget for a fallback source"""
        chip_frame = tk.Frame(
            self.fallback_frame,
            bg=self.theme.get("accent_blue", "#3a7ca5"),
            padx=4,
            pady=2
        )
        chip_frame.pack(side=tk.LEFT, padx=(0, 4))

        # Strip source suffix for display (e.g., "Fe_pct (assay)" -> "Fe_pct")
        display_text = text
        if " (" in text and text.endswith(")"):
            display_text = text.rsplit(" (", 1)[0]

        label = tk.Label(
            chip_frame,
            text=display_text,
            bg=self.theme.get("accent_blue", "#3a7ca5"),
            fg="white",
            font=("Arial", 8)
        )
        label.pack(side=tk.LEFT)

        # Remove button
        remove_btn = tk.Label(
            chip_frame,
            text=" ×",
            bg=self.theme.get("accent_blue", "#3a7ca5"),
            fg="white",
            font=("Arial", 9, "bold"),
            cursor="hand2"
        )
        remove_btn.pack(side=tk.LEFT)
        remove_btn.bind("<Button-1>", lambda e, i=index: self._remove_fallback(i))

        return chip_frame

    def _add_fallback(self):
        """Open dropdown to add a fallback source"""
        # Create a popup window with searchable dropdown
        popup = tk.Toplevel(self.parent)
        popup.title("Add Fallback Source")
        popup.transient(self.parent)
        # NOTE: Don't use grab_set() - it prevents searchable dropdown from working
        popup.focus_set()  # Just focus, don't grab

        # Position near the button
        popup.geometry(f"+{self.parent.winfo_rootx() + 50}+{self.parent.winfo_rooty() + 50}")

        frame = tk.Frame(popup, bg=self.theme["background"], padx=10, pady=10)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Select fallback column:").pack(anchor="w", pady=(0, 5))

        # Filter out already selected columns and the primary column
        available = [c for c in self.available_columns
                     if c not in self.fallback_sources and c != self.column_var.get()]

        if not available:
            ttk.Label(frame, text="No available columns to add").pack(pady=10)
            ModernButton(
                frame,
                text="Close",
                command=popup.destroy,
                color=self.theme["secondary_bg"],
                theme_colors=self.theme
            ).pack(pady=5)
            return

        selected_var = tk.StringVar()
        dropdown = self.gui_manager.create_searchable_optionmenu(
            parent=frame,
            items=available,
            variable=selected_var,
            width=30,
            placeholder="Search columns...",
            dropdown_mode="overlay",
        )
        dropdown.pack(fill=tk.X, pady=5)

        def on_add():
            col = selected_var.get()
            if col and col not in self.fallback_sources:
                self.fallback_sources.append(col)
                self._render_fallback_chips()
                self._notify_update()
            popup.destroy()

        btn_frame = tk.Frame(frame, bg=self.theme["background"])
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        ModernButton(
            btn_frame,
            text="Cancel",
            command=popup.destroy,
            color=self.theme["secondary_bg"],
            theme_colors=self.theme
        ).pack(side=tk.LEFT, padx=(0, 5))

        ModernButton(
            btn_frame,
            text="Add",
            command=on_add,
            color=self.theme.get("accent_green", "#4a8259"),
            theme_colors=self.theme
        ).pack(side=tk.LEFT)

    def _remove_fallback(self, index):
        """Remove a fallback source by index"""
        if 0 <= index < len(self.fallback_sources):
            self.fallback_sources.pop(index)
            self._render_fallback_chips()
            self._notify_update()

    def _on_color_map_change(self, *args):
        """Handle color map selection change"""
        self.config["color_map"] = self.color_map_var.get()
        self._update_color_preview()
        self._notify_update()

    def _edit_current_color_map(self):
        """Open color map editor for current selection"""
        color_map_name = self.color_map_var.get()
        column_name = self.column_var.get()
        
        if self.callbacks.get("on_edit_color_map"):
            self.callbacks["on_edit_color_map"](color_map_name, column_name)

    def _on_remove(self):
        """Handle remove button click"""
        if self.callbacks.get("on_remove"):
            self.callbacks["on_remove"](self.row_id)

    def _on_select(self):
        """Handle row selection"""
        if self.callbacks.get("on_select"):
            self.callbacks["on_select"](self.row_id)

    def _notify_update(self):
        """Notify parent of changes"""
        if self.callbacks.get("on_update"):
            self.callbacks["on_update"](self.row_id, self.get_config())
        if self.callbacks.get("on_preview_update"):
            self.callbacks["on_preview_update"]()

    def set_selected(self, selected: bool):
        """Update visual selection state"""
        self.is_selected = selected
        
        if selected:
            self.main_frame.config(
                highlightbackground=self.theme.get("accent_blue", "#007bff"),
                highlightthickness=3
            )
        else:
            self.main_frame.config(
                highlightbackground=self.theme["border"],
                highlightthickness=2
            )

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration including fallback sources"""
        return {
            "column": self.column_var.get(),
            "color_map": self.color_map_var.get(),
            "custom_label": self.label_var.get() or self.column_var.get(),
            "fallback_sources": self.fallback_sources.copy()  # Include fallback sources
        }

    def update_color_map_list(self, presets: list):
        """Update available color map presets"""
        self.available_presets = presets
        
        # Rebuild dropdown menu
        menu = self.color_map_dropdown["menu"]
        menu.delete(0, "end")
        
        for preset in presets:
            menu.add_command(
                label=preset,
                command=lambda p=preset: self._set_color_map(p)
            )

    def _set_color_map(self, preset_name: str):
        """Set color map from dropdown selection"""
        self.color_map_var.set(preset_name)
        self._on_color_map_change()

    def destroy(self):
        """Clean up the widget"""
        self.main_frame.destroy()

    # Legacy property for compatibility
    @property
    def frame(self):
        """Return main frame for legacy compatibility"""
        return self.main_frame