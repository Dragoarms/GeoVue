# """
# Dialog for configuring visualization columns displayed in the grid canvas.
# Version 3.0 - Uses widget-based expandable rows with draggable preview boundaries.
# """

# import tkinter as tk
# from tkinter import ttk
# import logging
# from typing import List, Dict, Any, Optional

# from gui.dialog_helper import DialogHelper
# from gui.widgets.modern_button import ModernButton
# from gui.widgets.field_with_label import create_field_with_label
# from gui.widgets.viz_column_item_row import VizColumnItemRow


# class VizColumnSettingsDialog:
#     """
#     Dialog for configuring visualization columns.
#     Similar to ClassificationSettingsDialog but for data visualizations.
#     """
    
#     def __init__(self, parent, gui_manager, data_coordinator, 
#                  config_manager, image_index=None):
#         self.parent = parent
#         self.gui_manager = gui_manager
#         self.theme_colors = gui_manager.theme_colors
#         self.data_coordinator = data_coordinator  # NEW ARCHITECTURE
#         self.config_manager = config_manager
#         self.image_index = image_index
        
#         # Access stores via coordinator
#         self.geological_store = data_coordinator.geological_store if data_coordinator else None
#         self.color_map_store = data_coordinator.color_maps if data_coordinator else None
#         self.image_index = image_index
#         self.logger = logging.getLogger(__name__)
        
#         # Get ColorMapManager - prefer from store, fallback to creating new
#         self.color_map_manager = None
#         if self.color_map_store and hasattr(self.color_map_store, 'manager'):
#             self.color_map_manager = self.color_map_store.manager
        
#         if self.color_map_manager is None:
#             self.logger.error(f"Could not import ColorMapManager:VizColumnSettingsDialog init")
        
#         # Use fonts from gui_manager for consistent theming
#         self.fonts = gui_manager.fonts
        
#         self.dialog = None
#         self.result = None
#         self.column_rows: List[VizColumnItemRow] = []
#         self.selected_row = None  # Track selected row for move operations
#         self.color_map_editor_window = None  # Track the non-modal editor window

#         # Draggable preview state
#         self.drag_active = False
#         self.drag_type = None  # 'column_boundary' or 'viz_height'
#         self.drag_start_x = 0
#         self.drag_start_y = 0
#         self.drag_column_index = None
#         self.drag_hover_cursor = False
        
#         # Get available columns and presets
#         self.available_columns = self._get_available_columns()
#         self.available_presets = self.color_map_manager.get_preset_names()
        
#         self.logger.debug(f"Initialized with {len(self.available_columns)} columns and {len(self.available_presets)} presets")
        
#         if not self.available_presets:
#             # Create default presets if none exist
#             self.logger.info("No presets found, creating defaults")
#             self.color_map_manager.create_default_presets()
#             self.available_presets = self.color_map_manager.get_preset_names()
    
#     def _get_font_family(self):
#         """Get font family from gui_manager fonts"""
#         # Extract family from the normal font tuple
#         if self.fonts and "normal" in self.fonts:
#             return self.fonts["normal"][0]  # First element is family name
#         return "Arial"  # Fallback

#     def _get_available_columns(self):
#         """Get all available columns from GeologicalStore via DataCoordinator"""
#         columns = []
        
#         # Get columns from GeologicalStore (new architecture)
#         if self.geological_store and hasattr(self.geological_store, 'get_available_columns'):
#             try:
#                 available = self.geological_store.get_available_columns()
#                 source_count = len(available)
                
#                 for source_name, cols in available.items():
#                     for col_name, col_type in cols:
#                         if source_count > 1:
#                             display_name = f"{col_name} ({source_name})"
#                         else:
#                             display_name = col_name
#                         if display_name not in columns:
#                             columns.append(display_name)
                
#                 self.logger.debug(f"Got {len(columns)} columns from GeologicalStore")
#             except Exception as e:
#                 self.logger.error(f"Error getting GeologicalStore columns: {e}")
        
#         # Add defaults if nothing available
#         if not columns:
#             self.logger.warning("No columns found, using defaults")
#             columns = [
#                 "Fe_pct_BEST",
#                 "SiO2_pct_BEST",
#                 "Al2O3_pct_BEST",
#                 "Logged_pct_CHHM",
#                 "BIFf_2"
#             ]
        
#         return columns

#     def show(self) -> Optional[List[Dict[str, str]]]:
#         """Show dialog and return result"""
#         self._create_dialog()
#         self.dialog.wait_window()
#         return self.result
    
#     def _create_dialog(self):
#         """Create the dialog window"""
#         self.dialog = DialogHelper.create_dialog(
#             self.parent,
#             title="Configure Data Visualizations",
#             modal=False
#         )
        
#         self.dialog.configure(bg=self.theme_colors["background"])
#         self.dialog.geometry("950x600")
        
#         # Main container
#         main_frame = ttk.Frame(self.dialog, padding=10)
#         main_frame.pack(fill=tk.BOTH, expand=True)
        
#         # Title and subtitle
#         title_label = ttk.Label(
#             main_frame,
#             text="Manage Visualization Columns",
#             font=("Arial", 14, "bold")
#         )
#         title_label.pack(pady=(0, 5))
        
#         subtitle_label = ttk.Label(
#             main_frame,
#             text="Configure which data columns to display as colored bars next to images in the grid",
#             font=("Arial", 10)
#         )
#         subtitle_label.pack(pady=(0, 10))
        
#         # Instruction text
#         instruction_frame = tk.Frame(main_frame, bg=self.theme_colors["secondary_bg"],
#                                     highlightbackground=self.theme_colors["border"],
#                                     highlightthickness=1)
#         instruction_frame.pack(fill=tk.X, pady=(0, 10))
        
#         instruction_text = "Click a row to select it for reordering. Use buttons below to add/remove columns or reset to defaults."
#         ttk.Label(instruction_frame, text=instruction_text, 
#                  font=("Arial", 9)).pack(padx=10, pady=5)
        
#         # ========== CREATE SIDE-BY-SIDE LAYOUT WITH RESIZABLE PANED WINDOW ==========
#         # Use PanedWindow for resizable panels
#         content_container = tk.PanedWindow(
#             main_frame,
#             orient=tk.HORIZONTAL,
#             sashwidth=8,
#             sashrelief=tk.RAISED,
#             bg=self.theme_colors["border"],
#             borderwidth=0
#         )
#         content_container.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
#         # ========== LEFT PANEL: Column List ==========
#         left_panel = tk.Frame(content_container, bg=self.theme_colors["background"])
#         content_container.add(left_panel, minsize=600, stretch="always")
        
#         # Scrollable area for column rows
#         scroll_frame = tk.Frame(left_panel, bg=self.theme_colors["background"],
#                                highlightbackground=self.theme_colors["border"],
#                                highlightthickness=1)
#         scroll_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
#         # Canvas and scrollbar
#         canvas = tk.Canvas(scroll_frame, bg=self.theme_colors["background"],
#                           highlightthickness=0)
#         scrollbar = tk.Scrollbar(scroll_frame, orient="vertical", command=canvas.yview)
        
#         self.scrollable_frame = tk.Frame(canvas, bg=self.theme_colors["background"])
#         self.scrollable_frame.bind(
#             "<Configure>",
#             lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
#         )
        
#         canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
#         canvas.configure(yscrollcommand=scrollbar.set)
        
#         canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
#         scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
#         # Bind mousewheel - store the binding ID for cleanup
#         def _on_mousewheel(event):
#             # Check if canvas still exists before scrolling
#             if canvas.winfo_exists():
#                 canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
#         # Store references for cleanup
#         self._canvas = canvas
#         self._mousewheel_func = _on_mousewheel
#         # Use bind instead of bind_all to limit scope to this canvas
#         canvas.bind("<MouseWheel>", _on_mousewheel)
#         canvas.bind("<Button-4>", _on_mousewheel)  # Linux scroll up
#         canvas.bind("<Button-5>", _on_mousewheel)  # Linux scroll down
        
#         # Control buttons frame (below scrollable area in left panel)
#         button_frame = tk.Frame(left_panel, bg=self.theme_colors["background"])
#         button_frame.pack(fill=tk.X, pady=5)
        
#         # Left side - add/move buttons
#         ModernButton(
#             button_frame,
#             text="↑ Move Up",
#             command=self._move_up,
#             color=self.theme_colors["secondary_bg"],
#             theme_colors=self.theme_colors
#         ).pack(side=tk.LEFT, padx=2)
        
#         ModernButton(
#             button_frame,
#             text="↓ Move Down",
#             command=self._move_down,
#             color=self.theme_colors["secondary_bg"],
#             theme_colors=self.theme_colors
#         ).pack(side=tk.LEFT, padx=2)
        
#         ModernButton(
#             button_frame,
#             text="➕ Add Column",
#             command=self._add_column,
#             color=self.theme_colors["accent_green"],
#             theme_colors=self.theme_colors
#         ).pack(side=tk.LEFT, padx=2)
        
#         # Center - manage color maps button (IMPORTANT!)
#         ModernButton(
#             button_frame,
#             text="🎨 Manage Color Maps",
#             command=self._open_color_map_manager,
#             color=self.theme_colors["accent_blue"],
#             theme_colors=self.theme_colors
#         ).pack(side=tk.LEFT, padx=10)
        
#         # Right side - reset button
#         ModernButton(
#             button_frame,
#             text="RESET TO DEFAULT",
#             command=self._reset_to_default,
#             color=self.theme_colors["accent_blue"],
#             theme_colors=self.theme_colors
#         ).pack(side=tk.RIGHT, padx=2)
        
#         # ========== RIGHT PANEL: Preview ==========
#         right_panel = tk.Frame(content_container, bg=self.theme_colors["background"])
#         content_container.add(right_panel, minsize=250, stretch="never")
        
#         # ========== PREVIEW VISUALIZATION SECTION ==========
#         self._create_preview_section(right_panel)
        
#         # ========== GRID DISPLAY SETTINGS SECTION ==========
#         separator = ttk.Separator(main_frame, orient='horizontal')
#         separator.pack(fill=tk.X, pady=10)
        
#         grid_settings_label = ttk.Label(
#             main_frame,
#             text="Grid Display Settings",
#             font=("Arial", 12, "bold")
#         )
#         grid_settings_label.pack(anchor="w", pady=(0, 5))
        
#         # Grid settings frame
#         grid_settings_frame = tk.Frame(main_frame, bg=self.theme_colors["secondary_bg"],
#                                        highlightbackground=self.theme_colors["border"],
#                                        highlightthickness=1)
#         grid_settings_frame.pack(fill=tk.X, pady=(0, 10))
        
#         # Row 1: Viz width and outline settings
#         row1_frame = tk.Frame(grid_settings_frame, bg=self.theme_colors["secondary_bg"])
#         row1_frame.pack(fill=tk.X, padx=10, pady=5)
        
#         # Viz column width ratio
#         ttk.Label(row1_frame, text="Viz Column Width:").pack(side=tk.LEFT, padx=(0, 5))
#         self.viz_width_var = tk.IntVar(value=int(self.config_manager.get("viz_column_width_ratio", 0.30) * 100))
#         viz_width_spinbox = tk.Spinbox(
#             row1_frame, from_=10, to=50, width=5,
#             textvariable=self.viz_width_var,
#             bg=self.theme_colors["field_bg"],
#             fg=self.theme_colors["text"],
#             command=self._update_preview,
#             validate="key",
#             validatecommand=(self.dialog.register(lambda x: x.isdigit() or x == ""), '%P')
#         )
#         viz_width_spinbox.pack(side=tk.LEFT)
#         self.viz_width_var.trace_add("write", lambda *args: self._update_preview())
#         ttk.Label(row1_frame, text="% of cell").pack(side=tk.LEFT, padx=(2, 20))
        
#         # Viz column height (for landscape orientation)
#         ttk.Label(row1_frame, text="Column Height:").pack(side=tk.LEFT, padx=(0, 5))
#         self.viz_height_var = tk.IntVar(value=self.config_manager.get("viz_column_height", 80))
#         viz_height_spinbox = tk.Spinbox(
#             row1_frame, from_=40, to=200, width=5,
#             textvariable=self.viz_height_var,
#             bg=self.theme_colors["field_bg"],
#             fg=self.theme_colors["text"],
#             command=self._on_height_change,
#             validate="key",
#             validatecommand=(self.dialog.register(lambda x: x.isdigit() or x == ""), '%P')
#         )

#         viz_height_spinbox.pack(side=tk.LEFT)
#         self.viz_height_var.trace_add("write", lambda *args: self._on_height_change())
#         ttk.Label(row1_frame, text="px").pack(side=tk.LEFT, padx=(2, 20))
        
#         # Font size
#         ttk.Label(row1_frame, text="Font Size:").pack(side=tk.LEFT, padx=(0, 5))
#         self.viz_font_size_var = tk.IntVar(value=self.config_manager.get("viz_column_font_size", 7))
#         font_size_spinbox = tk.Spinbox(
#             row1_frame, from_=6, to=12, width=4,
#             textvariable=self.viz_font_size_var,
#             bg=self.theme_colors["field_bg"],
#             fg=self.theme_colors["text"],
#             command=self._update_preview
#         )
#         font_size_spinbox.pack(side=tk.LEFT)
#         self.viz_font_size_var.trace_add("write", lambda *args: self._update_preview())
#         ttk.Label(row1_frame, text="pt").pack(side=tk.LEFT, padx=(2, 0))
        
#         # Show outlines checkbox
#         self.show_outlines_var = tk.BooleanVar(value=self.config_manager.get("grid_show_outlines", True))
#         outline_cb = tk.Checkbutton(
#             row1_frame, text="Show Cell Outlines",
#             variable=self.show_outlines_var,
#             bg=self.theme_colors["secondary_bg"],
#             fg=self.theme_colors["text"],
#             selectcolor=self.theme_colors["checkbox_bg"],
#             activebackground=self.theme_colors["secondary_bg"]
#         )
#         outline_cb.pack(side=tk.LEFT, padx=(0, 10))
        
#         # Outline width
#         ttk.Label(row1_frame, text="Outline Width:").pack(side=tk.LEFT, padx=(0, 5))
#         self.outline_width_var = tk.IntVar(value=self.config_manager.get("grid_outline_width", 2))
#         outline_spinbox = tk.Spinbox(
#             row1_frame, from_=1, to=10, width=4,
#             textvariable=self.outline_width_var,
#             bg=self.theme_colors["field_bg"],
#             fg=self.theme_colors["text"]
#         )
#         outline_spinbox.pack(side=tk.LEFT)
#         ttk.Label(row1_frame, text="px").pack(side=tk.LEFT, padx=(2, 0))
        
#         # Row 2: Label settings
#         row2_frame = tk.Frame(grid_settings_frame, bg=self.theme_colors["secondary_bg"])
#         row2_frame.pack(fill=tk.X, padx=10, pady=5)
        
#         # Show cell labels (hole ID/depth)
#         self.show_cell_labels_var = tk.BooleanVar(value=self.config_manager.get("grid_show_cell_labels", True))
#         cell_labels_cb = tk.Checkbutton(
#             row2_frame, text="Show Hole ID / Depth Labels",
#             variable=self.show_cell_labels_var,
#             bg=self.theme_colors["secondary_bg"],
#             fg=self.theme_colors["text"],
#             selectcolor=self.theme_colors["checkbox_bg"],
#             activebackground=self.theme_colors["secondary_bg"]
#         )
#         cell_labels_cb.pack(side=tk.LEFT, padx=(0, 20))
        
#         # Show classification labels
#         self.show_class_labels_var = tk.BooleanVar(value=self.config_manager.get("grid_show_classification_labels", True))
#         class_labels_cb = tk.Checkbutton(
#             row2_frame, text="Show Classification Labels",
#             variable=self.show_class_labels_var,
#             bg=self.theme_colors["secondary_bg"],
#             fg=self.theme_colors["text"],
#             selectcolor=self.theme_colors["checkbox_bg"],
#             activebackground=self.theme_colors["secondary_bg"],
#             command=self._on_class_labels_toggle
#         )
#         class_labels_cb.pack(side=tk.LEFT, padx=(0, 20))
        
#         # Classification label position
#         ttk.Label(row2_frame, text="Label Position:").pack(side=tk.LEFT, padx=(0, 5))
#         self.label_position_var = tk.StringVar(value=self.config_manager.get("grid_classification_label_position", "top-right"))
#         position_frame = tk.Frame(row2_frame, bg=self.theme_colors["field_bg"],
#                                   highlightbackground=self.theme_colors["field_border"],
#                                   highlightthickness=1)
#         position_frame.pack(side=tk.LEFT)
#         position_dropdown = tk.OptionMenu(position_frame, self.label_position_var, "top-right", "top-left")
#         self.gui_manager.style_dropdown(position_dropdown, width=10)
#         position_dropdown.pack()
        
#         # Warning label for hidden classification labels
#         self.class_warning_label = ttk.Label(
#             grid_settings_frame,
#             text="⚠ Classification is disabled when labels are hidden",
#             font=("Arial", 9, "italic"),
#             foreground="#ff9900"
#         )
#         if not self.show_class_labels_var.get():
#             self.class_warning_label.pack(anchor="w", padx=10, pady=(0, 5))
        
#         # ========== BOTTOM BUTTONS ==========
#         bottom_frame = tk.Frame(main_frame, bg=self.theme_colors["background"])
#         bottom_frame.pack(side=tk.BOTTOM, pady=10)
        
#         ModernButton(
#             bottom_frame,
#             text="Cancel",
#             command=self._on_cancel,
#             color=self.theme_colors["secondary_bg"],
#             theme_colors=self.theme_colors
#         ).pack(side=tk.LEFT, padx=5)
        
#         ModernButton(
#             bottom_frame,
#             text="Done",
#             command=self._on_done,
#             color=self.theme_colors["accent_green"],
#             theme_colors=self.theme_colors
#         ).pack(side=tk.LEFT, padx=5)
        
#         # Load current config from settings
#         self._load_current_config()
    
#     def _create_preview_section(self, parent_panel):
#         """Create the preview visualization section with validation"""
#         # No separator needed in side-by-side layout
        
#         preview_label = ttk.Label(
#             parent_panel,
#             text="Preview Visualization",
#             font=("Arial", 12, "bold")
#         )
#         preview_label.pack(anchor="w", pady=(0, 5))
        
#         preview_frame = tk.Frame(parent_panel, bg=self.theme_colors["secondary_bg"],
#                                  highlightbackground=self.theme_colors["border"],
#                                  highlightthickness=1)
#         preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 0))
        
#         # Controls row
#         controls_frame = tk.Frame(preview_frame, bg=self.theme_colors["secondary_bg"])
#         controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
#         # Orientation toggle
#         ttk.Label(controls_frame, text="Orientation:").pack(side=tk.LEFT, padx=(0, 5))
#         self.preview_orientation_var = tk.StringVar(value="portrait")
        
#         orientation_frame = tk.Frame(controls_frame, bg=self.theme_colors["field_bg"],
#                                       highlightbackground=self.theme_colors["field_border"],
#                                       highlightthickness=1)
#         orientation_frame.pack(side=tk.LEFT, padx=(0, 20))
        
#         orientation_dropdown = tk.OptionMenu(
#             orientation_frame, self.preview_orientation_var, 
#             "portrait", "landscape",
#             command=lambda _: self._update_preview()
#         )
#         self.gui_manager.style_dropdown(orientation_dropdown, width=10)
#         orientation_dropdown.pack()
        
#         # Cell dimensions (read-only - determined by loaded images)
#         ttk.Label(controls_frame, text="Cell Size:").pack(side=tk.LEFT, padx=(0, 5))
#         self.preview_width_var = tk.IntVar(value=self.config_manager.get("grid_cell_width", 300))
#         self.preview_height_var = tk.IntVar(value=self.config_manager.get("grid_cell_height", 150))
        
#         self.cell_size_label = ttk.Label(
#             controls_frame,
#             text=f"{self.preview_width_var.get()}×{self.preview_height_var.get()}px",
#             font=("Arial", 9, "bold"),
#             foreground=self.theme_colors["text"]
#         )
#         self.cell_size_label.pack(side=tk.LEFT, padx=(0, 20))
        
#         # Auto-scale fonts checkbox
#         self.auto_scale_fonts_var = tk.BooleanVar(value=False)
#         auto_scale_cb = tk.Checkbutton(
#             controls_frame, text="Auto-scale fonts",
#             variable=self.auto_scale_fonts_var,
#             bg=self.theme_colors["secondary_bg"],
#             fg=self.theme_colors["text"],
#             selectcolor=self.theme_colors["checkbox_bg"],
#             activebackground=self.theme_colors["secondary_bg"],
#             command=self._on_auto_scale_toggle
#         )
#         auto_scale_cb.pack(side=tk.LEFT, padx=(20, 10))
        
#         # Refresh button
#         ModernButton(
#             controls_frame,
#             text="🔄 Refresh",
#             command=self._update_preview,
#             color=self.theme_colors["accent_blue"],
#             theme_colors=self.theme_colors
#         ).pack(side=tk.LEFT, padx=(10, 0))
        
#         # Auto-adjust button
#         ModernButton(
#             controls_frame,
#             text="⚙️ Auto-adjust Widths",
#             command=self._auto_adjust_column_widths,
#             color=self.theme_colors["accent_green"],
#             theme_colors=self.theme_colors
#         ).pack(side=tk.LEFT, padx=(5, 0))

#         # Grid scale buttons
#         scale_frame = tk.Frame(controls_frame, bg=self.theme_colors["secondary_bg"])
#         scale_frame.pack(side=tk.LEFT, padx=(10, 0))
        
#         ttk.Label(scale_frame, text="Grid Scale:").pack(side=tk.LEFT, padx=(0, 5))
        
#         smaller_btn = ModernButton(
#             scale_frame,
#             text="−",
#             command=self._decrease_grid_scale,
#             color=self.theme_colors.get("accent_blue", "#007bff"),
#             theme_colors=self.theme_colors,
#             width=30
#         )
#         smaller_btn.pack(side=tk.LEFT, padx=2)
        
#         bigger_btn = ModernButton(
#             scale_frame,
#             text="+",
#             command=self._increase_grid_scale,
#             color=self.theme_colors.get("accent_blue", "#007bff"),
#             theme_colors=self.theme_colors,
#             width=30
#         )
#         bigger_btn.pack(side=tk.LEFT, padx=2)
        
#         # Warning labels container
#         warnings_frame = tk.Frame(preview_frame, bg=self.theme_colors["secondary_bg"])
#         warnings_frame.pack(fill=tk.X, padx=10)
        
#         self.width_warning_label = ttk.Label(
#             warnings_frame,
#             text="",
#             font=("Arial", 9, "italic"),
#             foreground="#ff9900"
#         )
        
#         self.dimension_info_label = ttk.Label(
#             warnings_frame,
#             text="",
#             font=("Arial", 9),
#             foreground=self.theme_colors["text"]
#         )
#         self.dimension_info_label.pack(anchor="w", pady=(2, 5))
        
#         # Canvas for preview
#         canvas_container = tk.Frame(preview_frame, bg=self.theme_colors["background"])
#         canvas_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
#         self.preview_canvas = tk.Canvas(
#             canvas_container,
#             bg="#FFFFFF",
#             highlightthickness=1,
#             highlightbackground=self.theme_colors["border"]
#         )
#         self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        
#         # Bind drag events for interactive preview
#         self.preview_canvas.bind("<Motion>", self._on_preview_motion)
#         self.preview_canvas.bind("<Button-1>", self._on_preview_click)
#         self.preview_canvas.bind("<B1-Motion>", self._on_preview_drag)
#         self.preview_canvas.bind("<ButtonRelease-1>", self._on_preview_release)
#         self.preview_canvas.bind("<Leave>", self._on_preview_leave)
        
#         # Store draggable boundaries (updated during draw)
#         self.drag_boundaries = []
        
#         # Initial preview
#         self._update_preview()
    
#     def _on_auto_scale_toggle(self):
#         """Handle auto-scale fonts toggle"""
#         if self.auto_scale_fonts_var.get():
#             self._auto_scale_fonts()

#     def _update_preview(self):
#         """Update the preview visualization with validation"""
#         if not hasattr(self, 'preview_canvas'):
#             return
        
#         # Clear canvas
#         self.preview_canvas.delete("all")
        
#         # Reset drag boundaries
#         self.drag_boundaries = []
        
#         if not self.column_rows:
#             self.preview_canvas.create_text(
#                 150, 75,
#                 text="No columns configured",
#                 fill=self.theme_colors["text"],
#                 font=self.fonts["normal"]
#             )
#             return
        
#         # Get dimensions
#         dims = self._calculate_actual_dimensions()
        
#         # Update dimension info
#         viz_count = len(self.column_rows)
#         per_col_width = dims['viz_total_width'] // viz_count if viz_count > 0 else 0
        
#         info_text = (f"📐 Viz: {dims['viz_total_width']}px total "
#                     f"({per_col_width}px per column × {viz_count} columns) | "
#                     f"Image: {dims['image_width']}×{dims['image_height']}px")
        
#         self.dimension_info_label.config(text=info_text)
        
#         # Validate and show warnings
#         self._validate_column_widths()
        
#         orientation = self.preview_orientation_var.get()
#         preview_width = self.preview_width_var.get()
#         preview_height = self.preview_height_var.get()
        
#         # Draw sample visualization
#         if orientation == "portrait":
#             self._draw_portrait_preview(preview_width, preview_height, dims)
#         else:
#             self._draw_landscape_preview(preview_width, preview_height, dims)

#     def _load_sample_image(self, target_width, target_height):
#         """Try to load a sample image from the image index"""
#         if not self.image_index:
#             return None
        
#         try:
#             all_images = self.image_index.get_all_images()
#             if not all_images or len(all_images) == 0:
#                 return None
            
#             # Get first image
#             first_img_path = all_images[0]
            
#             from PIL import Image, ImageTk
#             pil_img = Image.open(first_img_path)
            
#             # Resize to fit target dimensions
#             pil_img = pil_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
#             # Convert to PhotoImage
#             tk_image = ImageTk.PhotoImage(pil_img)
            
#             return tk_image
#         except Exception as e:
#             self.logger.debug(f"Could not load sample image: {e}")
#             return None

#     def _draw_portrait_preview(self, width, height, dims):
#         """Draw portrait orientation preview - VERTICAL columns on RIGHT of TALL cell"""
#         # SWAP dimensions for portrait - make it TALL
#         actual_width = height  # Portrait is TALL, so width = original height
#         actual_height = width  # Height = original width (swapped)
        
#         viz_total_width = dims['viz_total_width']
#         flex_width = dims['flex_width']
        
#         # For portrait, viz columns take vertical space, so calculate proportionally
#         viz_width_portrait = int(viz_total_width * (actual_width / width))
#         image_width = actual_width - viz_width_portrait
        
#         # Get canvas dimensions
#         self.preview_canvas.update_idletasks()
#         canvas_width = self.preview_canvas.winfo_width()
#         canvas_height = self.preview_canvas.winfo_height()
        
#         # Calculate scale to fit cell in canvas with minimal padding for larger preview
#         padding = 20  # Smaller padding = larger mockup
#         scale_x = (canvas_width - padding) / actual_width if actual_width > 0 else 1.0
#         scale_y = (canvas_height - padding) / actual_height if actual_height > 0 else 1.0
#         scale = min(scale_x, scale_y, 1.0)  # Never scale UP, only down
        
#         # Scaled dimensions
#         scaled_width = int(actual_width * scale)
#         scaled_height = int(actual_height * scale)
#         scaled_viz_width = int(viz_width_portrait * scale)
#         scaled_image_width = int(image_width * scale)
        
#         # Center the mockup
#         offset_x = (canvas_width - scaled_width) // 2
#         offset_y = (canvas_height - scaled_height) // 2
        
#         # Draw cell border if outlines enabled
#         if self.show_outlines_var.get():
#             outline_width = self.outline_width_var.get()
#             self.preview_canvas.create_rectangle(
#                 offset_x, offset_y,
#                 offset_x + scaled_width, offset_y + scaled_height,
#                 outline="#000000",
#                 width=outline_width
#             )
        
#         # Draw image area on the LEFT - try to load actual image, fallback to checkerboard
#         img_x = offset_x
        
#         # Try to load sample image
#         sample_image = self._load_sample_image(scaled_image_width, scaled_height)
        
#         if sample_image:
#             # Draw actual image
#             self.preview_canvas.create_image(
#                 img_x, offset_y,
#                 image=sample_image,
#                 anchor="nw"
#             )
#             # Store reference to prevent garbage collection
#             self.preview_canvas._sample_image_portrait = sample_image
#         else:
#             # Fallback to checkerboard pattern
#             checker_size = max(10, int(20 * scale))
            
#             for row_idx in range(0, scaled_height, checker_size):
#                 for col_idx in range(0, scaled_image_width, checker_size):
#                     color = "#F0F0F0" if (row_idx // checker_size + col_idx // checker_size) % 2 == 0 else "#FFFFFF"
#                     self.preview_canvas.create_rectangle(
#                         img_x + col_idx, offset_y + row_idx,
#                         img_x + col_idx + checker_size, offset_y + row_idx + checker_size,
#                         fill=color,
#                         outline=""
#                     )
            
#             # Add dimension labels (only if no image)
#             self.preview_canvas.create_text(
#                 img_x + scaled_image_width // 2,
#                 offset_y + scaled_height // 2 - int(10 * scale),
#                 text="Image Area",
#                 fill="#999999",
#                 font=(self._get_font_family(), max(6, int(10 * scale)))
#             )
            
#             self.preview_canvas.create_text(
#                 img_x + scaled_image_width // 2,
#                 offset_y + scaled_height // 2 + int(10 * scale),
#                 text=f"{image_width}×{actual_height}px",
#                 fill="#666666",
#                 font=(self._get_font_family(), max(6, int(8 * scale)))
#             )
        
#         # Draw image area border
#         self.preview_canvas.create_rectangle(
#             img_x, offset_y,
#             img_x + scaled_image_width, offset_y + scaled_height,
#             outline=self.theme_colors["border"],
#             width=1
#         )
        
#         # Draw VERTICAL viz columns on the RIGHT
#         x = offset_x + scaled_image_width
        
#         # In portrait, all columns have same WIDTH (use viz_width_portrait)
#         # Individual column HEIGHT is controlled by column_size setting
        
#         y = offset_y
#         for idx, row in enumerate(self.column_rows):
#             config = row.get_config()
            
#             # In PORTRAIT: column_size = individual HEIGHT
#             if config.get("pack_remaining_space", False):
#                 actual_col_height = 50  # Default for flex in portrait
#                 # Calculate based on remaining space
#                 total_fixed_height = sum(
#                     r.column_size_var.get() for r in self.column_rows 
#                     if not r.pack_remaining_var.get()
#                 )
#                 flex_count = sum(1 for r in self.column_rows if r.pack_remaining_var.get())
#                 remaining = actual_height - total_fixed_height
#                 actual_col_height = remaining // flex_count if flex_count > 0 else 50
#             else:
#                 actual_col_height = config.get("column_size", 80)  # This is HEIGHT in portrait
            
#             # All columns use same width in portrait
#             scaled_viz_col_width = scaled_viz_width
#             scaled_col_height = int(actual_col_height * scale)
            
#             label = config.get("custom_label", config.get("column", ""))
            
#             # Draw column rectangle with light gray background - STACKED VERTICALLY
#             self.preview_canvas.create_rectangle(
#                 x, y,
#                 x + scaled_viz_col_width, y + scaled_col_height,
#                 fill="#E8E8E8",
#                 outline=self.theme_colors["border"] if self.show_outlines_var.get() else ""
#             )

#             # Store draggable boundary (bottom edge of this column for vertical stacking)
#             if idx < len(self.column_rows) - 1:  # Not the last column
#                 row_config = row.get_config()
#                 if not row_config.get("pack_remaining_space", False):
#                     self.drag_boundaries.append({
#                         'type': 'column_boundary',
#                         'column_index': idx,
#                         'y': y + scaled_col_height,  # Horizontal line for vertical stacking
#                         'x1': x,
#                         'x2': x + scaled_viz_col_width
#                     })
                    
#                     # Draw subtle drag indicator line (horizontal for vertical stack)
#                     self.preview_canvas.create_line(
#                         x, y + scaled_col_height,
#                         x + scaled_viz_col_width, y + scaled_col_height,
#                         fill="#0066CC",
#                         width=2,
#                         dash=(2, 2),
#                         tags="drag_indicator"
#                     )
            
#             # Get sample color
#             sample_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"]
#             bar_color = sample_colors[idx % len(sample_colors)]
            
#             # Draw colored bar INSIDE the column area
#             bar_padding = max(2, int(5 * scale))
#             bar_width = scaled_viz_col_width - (bar_padding * 2)
#             bar_height = max(10, int(20 * scale))
            
#             bar_y = y + int(scaled_col_height * 0.4)  # Position in middle
            
#             self.preview_canvas.create_rectangle(
#                 x + bar_padding, bar_y,
#                 x + bar_padding + bar_width, bar_y + bar_height,
#                 fill=bar_color,
#                 outline=""
#             )
            
#             # Draw header INSIDE column area
#             header_font_size = max(5, int(config.get("header_size", 8) * scale * 0.7))
#             header_bold = "bold" if config.get("header_bold", False) else "normal"
            
#             font_family = self._get_font_family()
#             font_spec = (font_family, header_font_size, header_bold)
            
#             # Truncate label if too long
#             max_label_len = max(3, int(10 * scale))
#             display_label = label[:max_label_len] + "..." if len(label) > max_label_len else label
            
#             self.preview_canvas.create_text(
#                 x + scaled_viz_col_width // 2,
#                 y + int(scaled_col_height * 0.15),
#                 text=display_label,
#                 fill="#000000",
#                 font=font_spec,
#                 anchor="n"
#             )
            
#             # Draw sample value INSIDE column area
#             value_font_size = max(5, int(config.get("font_size", 8) * scale * 0.7))
#             value_bold = "bold" if config.get("bold", False) else "normal"
#             value_font = (font_family, value_font_size, value_bold)
#             decimal_places = config.get("decimal_places", 2)
            
#             sample_val = 45.1 + idx * 5
#             formatted_val = f"{sample_val:.{decimal_places}f}"
            
#             self.preview_canvas.create_text(
#                 x + scaled_viz_col_width // 2,
#                 y + int(scaled_col_height * 0.75),
#                 text=formatted_val,
#                 fill="#000000",
#                 font=value_font
#             )
            
#             y += scaled_col_height
        
        
#         # Add global width boundary (vertical line at RIGHT edge of ALL columns)
#         self.drag_boundaries.append({
#             'type': 'global_viz_width',
#             'x': x,  # Right edge of all columns
#             'y1': offset_y,
#             'y2': offset_y + scaled_height
#         })
        
#         # Draw global width drag indicator
#         self.preview_canvas.create_line(
#             x, offset_y,
#             x, offset_y + scaled_height,
#             fill="#FF6600",  # Orange for global boundaries
#             width=3,
#             dash=(4, 4),
#             tags="drag_indicator_global"
#         )
        
#         # Add scale indicator at bottom
#         scale_text = f"Scale: {scale:.2f}x" if scale < 1.0 else "Scale: 1:1 (actual size)"
#         self.preview_canvas.create_text(
#             canvas_width // 2,
#             canvas_height - 10,
#             text=scale_text,
#             fill="#666666",
#             font=(self._get_font_family(), 7)
#         )
    
#     def _draw_landscape_preview(self, width, height, dims):
#         """Draw landscape orientation preview - HORIZONTAL columns BOTTOM of WIDE cell"""
#         viz_height = dims['viz_height']
#         flex_width = dims['flex_width']
        
#         # Get canvas dimensions
#         self.preview_canvas.update_idletasks()
#         canvas_width = self.preview_canvas.winfo_width()
#         canvas_height = self.preview_canvas.winfo_height()
        
#         # Calculate scale to fit cell in canvas
#         padding = 40
#         scale_x = (canvas_width - padding) / width if width > 0 else 1.0
#         scale_y = (canvas_height - padding) / height if height > 0 else 1.0
#         scale = min(scale_x, scale_y, 1.0)
        
#         # Scaled dimensions
#         scaled_width = int(width * scale)
#         scaled_height = int(height * scale)
#         scaled_viz_height = int(viz_height * scale)
        
#         # Center the mockup
#         offset_x = (canvas_width - scaled_width) // 2
#         offset_y = (canvas_height - scaled_height) // 2
        
#         # Draw cell border if outlines enabled
#         if self.show_outlines_var.get():
#             outline_width = self.outline_width_var.get()
#             self.preview_canvas.create_rectangle(
#                 offset_x, offset_y,
#                 offset_x + scaled_width, offset_y + scaled_height,
#                 outline="#000000",
#                 width=outline_width
#             )
        
#         # Calculate image area (TOP)
#         img_height = scaled_height - scaled_viz_height
        
#         # Draw image area ON TOP - try to load actual image, fallback to checkerboard
#         img_y = offset_y
        
#         # Try to load sample image
#         sample_image = self._load_sample_image(scaled_width, img_height)
        
#         if sample_image:
#             # Draw actual image
#             self.preview_canvas.create_image(
#                 offset_x, img_y,
#                 image=sample_image,
#                 anchor="nw"
#             )
#             # Store reference to prevent garbage collection
#             self.preview_canvas._sample_image_landscape = sample_image
#         else:
#             # Fallback to checkerboard
#             checker_size = max(10, int(20 * scale))
            
#             for row_idx in range(0, img_height, checker_size):
#                 for col_idx in range(0, scaled_width, checker_size):
#                     color = "#F0F0F0" if (row_idx // checker_size + col_idx // checker_size) % 2 == 0 else "#FFFFFF"
#                     self.preview_canvas.create_rectangle(
#                         offset_x + col_idx, img_y + row_idx,
#                         offset_x + col_idx + checker_size, img_y + row_idx + checker_size,
#                         fill=color,
#                         outline=""
#                     )
            
#             # Image area label (only if no image)
#             actual_img_height = height - viz_height
#             self.preview_canvas.create_text(
#                 offset_x + scaled_width // 2,
#                 img_y + img_height // 2,
#                 text=f"Image Area\n{width}×{actual_img_height}px",
#                 fill="#999999",
#                 font=(self._get_font_family(), max(6, int(9 * scale))),
#                 justify="center"
#             )
        
#         # Image area border
#         self.preview_canvas.create_rectangle(
#             offset_x, img_y,
#             offset_x + scaled_width, img_y + img_height,
#             outline=self.theme_colors["border"],
#             width=1
#         )
        
#         # Store draggable boundary for viz height (horizontal line between image and viz)
#         self.drag_boundaries.append({
#             'type': 'viz_height',
#             'y': offset_y + img_height,
#             'x1': offset_x,
#             'x2': offset_x + scaled_width
#         })
        
#         # Draw drag indicator line
#         self.preview_canvas.create_line(
#             offset_x, offset_y + img_height,
#             offset_x + scaled_width, offset_y + img_height,
#             fill="#0066CC",
#             width=2,
#             dash=(2, 2),
#             tags="drag_indicator"
#         )
        
#         # Draw HORIZONTAL viz columns on BOTTOM (side-by-side)
#         viz_y = offset_y + img_height
#         x = offset_x
        
#         sample_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"]
        
#         for idx, row in enumerate(self.column_rows):
#             config = row.get_config()
#             label = config.get("custom_label", config.get("column", ""))
            
#             # Determine this column's width
#             if config.get("pack_remaining_space", False):
#                 actual_viz_col_width = flex_width
#             else:
#                 actual_viz_col_width = config.get("column_size", 80)
            
#             scaled_viz_col_width = int(actual_viz_col_width * scale)
            
#             # Draw column rectangle - SIDE BY SIDE
#             self.preview_canvas.create_rectangle(
#                 x, viz_y,
#                 x + scaled_viz_col_width, viz_y + scaled_viz_height,
#                 fill="#E8E8E8",
#                 outline=self.theme_colors["border"] if self.show_outlines_var.get() else ""
#             )
            
#             # Store draggable boundary (right edge of this column)
#             if idx < len(self.column_rows) - 1:  # Not the last column
#                 row_config = row.get_config()
#                 if not row_config.get("pack_remaining_space", False):
#                     self.drag_boundaries.append({
#                         'type': 'column_boundary',
#                         'column_index': idx,
#                         'x': x + scaled_viz_col_width,
#                         'y1': viz_y,
#                         'y2': viz_y + scaled_viz_height
#                     })
                    
#                     # Draw subtle drag indicator line (vertical)
#                     self.preview_canvas.create_line(
#                         x + scaled_viz_col_width, viz_y,
#                         x + scaled_viz_col_width, viz_y + scaled_viz_height,
#                         fill="#0066CC",
#                         width=2,
#                         dash=(2, 2),
#                         tags="drag_indicator"
#                     )
            
#             # Get sample color
#             bar_color = sample_colors[idx % len(sample_colors)]
            
#             # Draw colored bar INSIDE the column area
#             bar_padding = max(2, int(5 * scale))
#             bar_width = scaled_viz_col_width - (bar_padding * 2)
#             bar_height = max(10, int(20 * scale))
            
#             bar_y_pos = viz_y + int(scaled_viz_height * 0.4)
            
#             self.preview_canvas.create_rectangle(
#                 x + bar_padding, bar_y_pos,
#                 x + bar_padding + bar_width, bar_y_pos + bar_height,
#                 fill=bar_color,
#                 outline=""
#             )
            
#             # Draw header INSIDE column area
#             header_font_size = max(5, int(config.get("header_size", 8) * scale * 0.7))
#             header_bold = "bold" if config.get("header_bold", False) else "normal"
            
#             font_family = self._get_font_family()
#             font_spec = (font_family, header_font_size, header_bold)
            
#             # Truncate label if too long
#             max_label_len = max(3, int(10 * scale))
#             display_label = label[:max_label_len] + "..." if len(label) > max_label_len else label
            
#             self.preview_canvas.create_text(
#                 x + scaled_viz_col_width // 2,
#                 viz_y + int(scaled_viz_height * 0.15),
#                 text=display_label,
#                 fill="#000000",
#                 font=font_spec,
#                 anchor="n"
#             )
            
#             # Draw sample value INSIDE column area
#             value_font_size = max(5, int(config.get("font_size", 8) * scale * 0.7))
#             value_bold = "bold" if config.get("bold", False) else "normal"
#             value_font = (font_family, value_font_size, value_bold)
#             decimal_places = config.get("decimal_places", 2)
            
#             sample_val = 45.1 + idx * 5
#             formatted_val = f"{sample_val:.{decimal_places}f}"
            
#             self.preview_canvas.create_text(
#                 x + scaled_viz_col_width // 2,
#                 viz_y + int(scaled_viz_height * 0.75),
#                 text=formatted_val,
#                 fill="#000000",
#                 font=value_font
#             )
            
#             x += scaled_viz_col_width
        
#         # Scale indicator
#         scale_text = f"Scale: {scale:.2f}x" if scale < 1.0 else "Scale: 1:1 (actual size)"
#         self.preview_canvas.create_text(
#             canvas_width // 2,
#             canvas_height - 10,
#             text=scale_text,
#             fill="#666666",
#             font=(font_family, 7)
#         )

#     def _on_class_labels_toggle(self):
#         """Handle toggle of classification labels checkbox"""
#         if self.show_class_labels_var.get():
#             self.class_warning_label.pack_forget()
#         else:
#             self.class_warning_label.pack(anchor="w", padx=10, pady=(0, 5))

#     def _load_current_config(self):
#         """Load current configuration from ConfigManager"""
#         saved_config = self.config_manager.get("viz_columns")
        
#         self.logger.debug(f"Loading viz config: {saved_config}")
        
#         if saved_config and isinstance(saved_config, list):
#             # Load saved configuration
#             self.logger.info(f"Loading {len(saved_config)} saved viz columns")
#             for config in saved_config:
#                 if isinstance(config, dict) and "column" in config and "color_map" in config:
#                     column_size = config.get("column_size", 80)
#                     custom_label = config.get("custom_label", "")
#                     font_size = config.get("font_size", 8)
#                     bold = config.get("bold", False)
#                     decimal_places = config.get("decimal_places", 2)
#                     header_bold = config.get("header_bold", True)
#                     header_underline = config.get("header_underline", False)
#                     header_size = config.get("header_size", 8)
#                     self._add_column_row(
#                         config["column"], config["color_map"], column_size, custom_label, 
#                         font_size, bold, decimal_places, header_bold, header_underline, header_size
#                     )
#         else:
#             # Load defaults
#             self.logger.info("No saved config found, loading defaults")
#             self._load_default_config()
    
#     def _load_default_config(self):
#         """Load default configuration"""
#         defaults = [
#             ("Fe_pct_BEST", "fe_grade"),
#             ("SiO2_pct_BEST", "sio2_grade"),
#             ("Al2O3_pct_BEST", "al2o3_grade"),
#             ("Logged_pct_CHHM", "fe_grade")
#         ]
        
#         self.logger.debug(f"Available columns: {self.available_columns}")
#         self.logger.debug(f"Available presets: {self.available_presets}")
        
#         for column, color_map in defaults:
#             # Only add if column exists in available columns
#             if column in self.available_columns:
#                 self.logger.debug(f"Adding default column: {column}")
#                 self._add_column_row(column, color_map)
#             else:
#                 # Try to find similar column
#                 for avail_col in self.available_columns:
#                     if column.lower() in avail_col.lower():
#                         self.logger.debug(f"Adding similar column: {avail_col} (for {column})")
#                         self._add_column_row(avail_col, color_map)
#                         break
    
#     def _add_column(self):
#         """Add new column row"""
#         # Get first available column not already used
#         used_columns = [row.column_var.get() for row in self.column_rows]
#         available = [col for col in self.available_columns if col not in used_columns]
        
#         if available:
#             # Try to pick an intelligent default color map based on column name
#             default_map = "fe_grade" if self.available_presets and "fe_grade" in self.available_presets else (self.available_presets[0] if self.available_presets else "")
#             col_lower = available[0].lower()
#             if "fe" in col_lower and "fe_grade" in self.available_presets:
#                 default_map = "fe_grade"
#             elif ("sio2" in col_lower or "silica" in col_lower) and "sio2_grade" in self.available_presets:
#                 default_map = "sio2_grade"
#             elif ("al2o3" in col_lower or "alumina" in col_lower) and "al2o3_grade" in self.available_presets:
#                 default_map = "al2o3_grade"
#             elif ("lith" in col_lower or "type" in col_lower) and "lithology" in self.available_presets:
#                 default_map = "lithology"
            
#             self._add_column_row(available[0], default_map)
#         else:
#             DialogHelper.show_message(
#                 self.dialog,
#                 "No Columns Available",
#                 "All available columns are already added.",
#                 message_type="info"
#             )
    
#     def _on_row_select(self, row: VizColumnItemRow):
#         """Handle row selection"""
#         # Deselect all rows first
#         for r in self.column_rows:
#             r.set_selected(False)
        
#         # Select the clicked row
#         row.set_selected(True)
#         self.selected_row = row
        
#         self.logger.debug(f"Selected row: {row.column_var.get()}")

#     def _on_width_change(self):
#         """Handle viz width change with validation"""
#         self._validate_column_widths()
#         self._update_preview()
    
#     def _on_height_change(self):
#         """Handle viz height change with auto font scaling"""
#         if hasattr(self, 'auto_scale_fonts_var') and self.auto_scale_fonts_var.get():
#             self._auto_scale_fonts()
#         self._update_preview()
    
#     def _validate_column_widths(self):
#         """Validate that viz columns don't overflow the cell"""
#         viz_ratio = self.viz_width_var.get() / 100.0
        
#         if viz_ratio > 0.5:
#             # Show warning if viz takes more than 50% of cell
#             if hasattr(self, 'width_warning_label'):
#                 self.width_warning_label.config(
#                     text=f"⚠️ Viz columns use {int(viz_ratio * 100)}% of cell width - may cause cramped images"
#                 )
#                 if not self.width_warning_label.winfo_ismapped():
#                     self.width_warning_label.pack(anchor="w", padx=10, pady=(0, 5))
#         else:
#             if hasattr(self, 'width_warning_label') and self.width_warning_label.winfo_ismapped():
#                 self.width_warning_label.pack_forget()
    
#     def _auto_scale_fonts(self):
#         """Auto-scale fonts based on cell height"""
#         # Get current cell height from preview
#         cell_height = self.preview_height_var.get()
        
#         # Calculate optimal font size (roughly 1pt per 15px of height)
#         optimal_font_size = max(6, min(12, cell_height // 15))
        
#         self.viz_font_size_var.set(optimal_font_size)
#         self.logger.debug(f"Auto-scaled font to {optimal_font_size}pt for {cell_height}px cell height")
    
#     def _calculate_actual_dimensions(self):
#         """Calculate actual pixel dimensions for viz columns with pack_remaining_space support"""
#         # Get base cell dimensions (from config or defaults)
#         base_cell_width = self.config_manager.get("grid_cell_width", 300)
#         base_cell_height = self.config_manager.get("grid_cell_height", 150)
        
#         # Calculate viz column dimensions (handle empty spinbox)
#         try:
#             viz_ratio = self.viz_width_var.get() / 100.0
#         except:
#             viz_ratio = 0.30  # Default 30%
#             self.viz_width_var.set(30)
        
#         viz_total_width = int(base_cell_width * viz_ratio)
        
#         try:
#             viz_height = self.viz_height_var.get()
#         except:
#             viz_height = 80  # Default
#             self.viz_height_var.set(80)

#         # Prevent viz from exceeding cell boundaries
#         viz_total_width = min(viz_total_width, int(base_cell_width * 0.5))  # Max 50% of cell
#         viz_height = min(viz_height, base_cell_height - 20)  # Leave 20px for image
        
#         # Calculate individual column widths accounting for pack_remaining_space
#         fixed_width_total = 0
#         flex_count = 0
        
#         for row in self.column_rows:
#             config = row.get_config()
#             if config.get("pack_remaining_space", False):
#                 flex_count += 1
#             else:
#                 fixed_width_total += config.get("column_size", 80)
        
#         # Calculate flex column width
#         remaining_width = max(0, viz_total_width - fixed_width_total)
#         flex_width = remaining_width // flex_count if flex_count > 0 else 0
        
#         # Image area dimensions
#         image_width = base_cell_width - viz_total_width
#         image_height = base_cell_height
        
#         return {
#             'cell_width': base_cell_width,
#             'cell_height': base_cell_height,
#             'viz_total_width': viz_total_width,
#             'viz_height': viz_height,
#             'image_width': image_width,
#             'image_height': image_height,
#             'viz_ratio': viz_ratio,
#             'fixed_width_total': fixed_width_total,
#             'flex_count': flex_count,
#             'flex_width': flex_width
#         }

#     def _get_column_width_per_viz(self):
#         """Calculate width per visualization column"""
#         dims = self._calculate_actual_dimensions()
#         num_columns = len(self.column_rows)
        
#         if num_columns == 0:
#             return 0
        
#         return dims['viz_total_width'] // num_columns
    
#     def _auto_adjust_column_widths(self):
#         """Auto-adjust individual column widths to fit properly"""
#         if not self.column_rows:
#             return
        
#         per_column_width = self._get_column_width_per_viz()
        
#         # Update all column rows
#         for row in self.column_rows:
#             row.column_size_var.set(per_column_width)
        
#         self.logger.info(f"Auto-adjusted column widths to {per_column_width}px each")

#     def _add_column_row(self, column="", color_map="", column_size=80, custom_label="", 
#                        font_size=8, bold=False, decimal_places=2, header_bold=True,
#                        header_underline=False, header_size=8, pack_remaining_space=False):
#         """Add a new column configuration row using widget"""
        
#         # Build config dict
#         config = {
#             "column": column,
#             "color_map": color_map,
#             "column_size": column_size,
#             "custom_label": custom_label if custom_label else column,
#             "data_type": "auto",
#             "font_size": font_size,
#             "bold": bold,
#             "decimal_places": decimal_places,
#             "header_bold": header_bold,
#             "header_underline": header_underline,
#             "header_size": header_size,
#             "pack_remaining_space": pack_remaining_space
#         }
        
#         # Create callbacks dict
#         callbacks = {
#             'on_update': self._on_row_update,
#             'on_remove': self._on_row_remove,
#             'on_select': self._on_row_select,
#             'on_edit_color_map': self._on_edit_color_map,
#             'on_manage_color_maps': self._open_color_map_manager,
#             'on_preview_update': self._update_preview
#         }
        
#         # Create widget
#         row_widget = VizColumnItemRow(
#             parent=self.scrollable_frame,
#             config=config,
#             gui_manager=self.gui_manager,
#             available_columns=self.available_columns,
#             available_presets=self.available_presets,
#             callbacks=callbacks,
#             csv_handler=None,
#             color_map_manager=self.color_map_manager,
#             drillhole_data_manager=None
#         )
        
#         self.column_rows.append(row_widget)
#         self.logger.debug(f"Added column row: {column} ({color_map})")

#     def _select_row(self, row: VizColumnItemRow):
#         """Select a row for moving"""
#         # Deselect all first
#         for r in self.column_rows:
#             r.frame.configure(highlightthickness=1)
        
#         # Select this row
#         row.frame.configure(highlightthickness=3)
#         self.selected_row = row
    
#     def _remove_column_row(self, row: VizColumnItemRow):
#         """Remove a column row"""
#         if len(self.column_rows) > 1:  # Keep at least one
#             self.column_rows.remove(row)
#             row.frame.destroy()
#         else:
#             DialogHelper.show_message(
#                 self.dialog,
#                 "Cannot Remove",
#                 "At least one visualization column must remain.",
#                 message_type="warning"
#             )
    
#     def _open_color_map_manager(self):
#         """
#         Open the modal color map editor dialog.
#         Uses modal dialog to avoid dialog stacking issues.
#         """
#         from ..color_map_editor_dialog import ColorMapEditorDialog
        
#         self.logger.info("Opening color map editor dialog")
        
#         # Get the first selected column to use as context
#         data_column = None
#         if self.column_rows:
#             data_column = self.column_rows[0].column_var.get()
        
#         # Get sample data for preview
#         data_values = []
        
#         self.logger.info(f"Getting sample data for column: {data_column}")
        
#         # Use GeologicalStore via DataCoordinator (new architecture)
#         if data_column and self.geological_store:
#             try:
#                 # Strip any source suffix like " (source_name)"
#                 column_base = data_column.split(" (")[0].strip()
                
#                 # Try to get column values from geological store
#                 if hasattr(self.geological_store, 'get_column_values'):
#                     data_values = self.geological_store.get_column_values(column_base)
#                 elif hasattr(self.geological_store, 'get_unique_values'):
#                     data_values = list(self.geological_store.get_unique_values(column_base))
                
#                 if data_values:
#                     # Limit to 1000 samples for performance
#                     data_values = data_values[:1000] if len(data_values) > 1000 else data_values
#                     self.logger.info(f"✓ Found {len(data_values)} values for '{column_base}' from GeologicalStore")
#                 else:
#                     self.logger.warning(f"Column '{column_base}' not found in GeologicalStore")
                    
#             except Exception as e:
#                 self.logger.error(f"Error getting sample data from GeologicalStore: {e}", exc_info=True)
        
#         if not data_values:
#             self.logger.warning(f"No sample data available for column: {data_column}")
        
#         self.logger.info(f"Creating ColorMapEditorDialog with {len(data_values)} data values")
        
#         # Create editor dialog - ColorMapEditorDialog doesn't use drillhole_data_manager
#         dialog = ColorMapEditorDialog(
#             parent=self.dialog,
#             gui_manager=self.gui_manager,
#             color_map_manager=self.color_map_manager,
#             data_column=data_column,
#             data_values=data_values,
#             initial_color_map=None
#         )
        
#         # Show and wait - dialog.show() handles the modal behavior internally
#         result = dialog.show()
        
#         self.logger.info(f"ColorMapEditorDialog closed, result: {result is not None}")
        
#         if result:
#             # Refresh all row dropdowns with updated presets
#             self.available_presets = self.color_map_manager.get_preset_names()
#             for row in self.column_rows:
#                 row.update_color_map_list(self.available_presets)
            
#             self.logger.info("Color map editor closed with changes, refreshed dropdowns")

#     def _on_row_update(self, row_id, changes):
#         """Handle row update callback"""
#         self.logger.debug(f"Row {row_id} updated: {changes}")
#         self._update_preview()
    
#     def _on_row_remove(self, row_id):
#         """Handle row removal callback"""
#         # Find and remove the row
#         row_to_remove = None
#         for row in self.column_rows:
#             if row.row_id == row_id:
#                 row_to_remove = row
#                 break
        
#         if row_to_remove:
#             # Destroy widget
#             row_to_remove.destroy()
#             # Remove from list
#             self.column_rows.remove(row_to_remove)
#             # Clear selection if this was selected
#             if self.selected_row == row_to_remove:
#                 self.selected_row = None
#             # Update preview
#             self._update_preview()
#             self.logger.info(f"Removed row {row_id}")
    
#     def _on_row_select(self, row_id):
#         """Handle row selection callback"""
#         # Deselect all
#         for row in self.column_rows:
#             row.set_selected(False)
        
#         # Select the clicked row
#         for row in self.column_rows:
#             if row.row_id == row_id:
#                 row.set_selected(True)
#                 self.selected_row = row
#                 self.logger.debug(f"Selected row {row_id}")
#                 break
    
#     def _on_edit_color_map(self, row: VizColumnItemRow):
#         """
#         Open color map editor for the selected column.
#         Callback from VizColumnItemRow.
#         """
#         column_name = row.column_var.get()
#         color_map_name = row.color_map_var.get()
        
#         self.logger.info(f"Edit color map: {color_map_name} for column {column_name}")
        
#         # Get data column for color map
#         data_column = column_name
#         if not data_column or data_column == "--- Spacer ---":
#             DialogHelper.show_message(
#                 self.dialog,
#                 "No Column Selected",
#                 "Please select a data column first.",
#                 message_type="warning"
#             )
#             return
        
#         # Extract column name (remove source suffix)
#         column_base = data_column
#         if isinstance(data_column, str):
#             # Handle formats like "fe_pct (exassay)" or "('p_pct', 'DataType.NUMERIC')"
#             if " (" in data_column:
#                 column_base = data_column.split(" (")[0].strip()
#             elif data_column.startswith("("):
#                 # Tuple string format
#                 try:
#                     import ast
#                     parsed = ast.literal_eval(data_column)
#                     if isinstance(parsed, tuple):
#                         column_base = parsed[0]
#                 except:
#                     pass
        
#         # Get sample data from GeologicalStore via DataCoordinator
#         data_values = []
        
#         if self.geological_store:
#             try:
#                 # Get all unique values from the geological store
#                 source_df = self.geological_store.get_source_dataframe(column_base)
#                 if source_df is not None and column_base in source_df.columns:
#                     values = source_df[column_base].dropna().unique()
#                     data_values = list(values[:1000])  # Limit to 1000 samples
#                     self.logger.info(f"Got {len(data_values)} values for '{column_base}' from GeologicalStore")
#             except Exception as e:
#                 self.logger.debug(f"Could not get data from GeologicalStore: {e}")
        
#         # Import here to avoid circular imports
#         from gui.color_map_editor_dialog import ColorMapEditorDialog
        
#         # Get the actual ColorMapManager (not the store wrapper)
#         color_map_manager = None
#         if self.color_map_store and hasattr(self.color_map_store, 'manager'):
#             color_map_manager = self.color_map_store.manager
        
#         if not color_map_manager:
#             DialogHelper.show_message(
#                 self.dialog,
#                 "Color Map Manager Unavailable",
#                 "Cannot edit color maps - manager not initialized.",
#                 message_type="error"
#             )
#             return
        
#         # Create editor dialog
#         dialog = ColorMapEditorDialog(
#             parent=self.dialog,
#             gui_manager=self.gui_manager,
#             color_map_manager=color_map_manager,
#             data_column=data_column,
#             data_values=data_values,
#             initial_color_map=None
#         )
        
#         result = dialog.show()
        
#         if result:
#             # Refresh all row dropdowns
#             if color_map_manager and hasattr(color_map_manager, 'get_preset_names'):
#                 self.available_presets = color_map_manager.get_preset_names()
#                 for r in self.column_rows:
#                     r.update_color_map_list(self.available_presets)
#                 self.logger.info("Color map editor closed with changes")

#     def _on_color_maps_updated(self):
#         """
#         Callback when color maps are updated in the manager window.
#         Refresh the dropdowns in all rows.
#         """
#         self.logger.info("Color maps updated, refreshing dropdowns")
        
#         # Get updated preset list
#         self.available_presets = self.color_map_manager.get_preset_names()
        
#         # Update all row dropdowns
#         for row in self.column_rows:
#             row.update_color_map_list(self.available_presets)
        
#         self.logger.debug(f"Updated {len(self.column_rows)} rows with {len(self.available_presets)} presets")
    
#     def _move_up(self):
#         """Move selected row up"""
#         if not hasattr(self, 'selected_row') or self.selected_row not in self.column_rows:
#             DialogHelper.show_message(
#                 self.dialog,
#                 "No Selection",
#                 "Please click a row to select it first.",
#                 message_type="info"
#             )
#             return
        
#         idx = self.column_rows.index(self.selected_row)
#         if idx > 0:
#             # Swap in list
#             self.column_rows[idx], self.column_rows[idx-1] = self.column_rows[idx-1], self.column_rows[idx]
#             # Rebuild UI
#             self._rebuild_rows()
    
#     def _move_down(self):
#         """Move selected row down"""
#         if not hasattr(self, 'selected_row') or self.selected_row not in self.column_rows:
#             DialogHelper.show_message(
#                 self.dialog,
#                 "No Selection",
#                 "Please click a row to select it first.",
#                 message_type="info"
#             )
#             return
        
#         idx = self.column_rows.index(self.selected_row)
#         if idx < len(self.column_rows) - 1:
#             # Swap in list
#             self.column_rows[idx], self.column_rows[idx+1] = self.column_rows[idx+1], self.column_rows[idx]
#             # Rebuild UI
#             self._rebuild_rows()
    
#     def _on_preview_motion(self, event):
#         """Handle mouse motion over preview - show resize cursor"""
#         if self.drag_active:
#             return  # Already dragging
        
#         # Check if mouse is over a draggable boundary
#         over_boundary = False
#         for boundary in self.drag_boundaries:
#             if self._is_near_boundary(event.x, event.y, boundary):
#                 over_boundary = True
#                 break
        
#         # Update cursor
#         if over_boundary:
#             if not self.drag_hover_cursor:
#                 if self.preview_orientation_var.get() == "portrait":
#                     self.preview_canvas.config(cursor="sb_h_double_arrow")
#                 else:
#                     self.preview_canvas.config(cursor="sb_v_double_arrow")
#                 self.drag_hover_cursor = True
#         else:
#             if self.drag_hover_cursor:
#                 self.preview_canvas.config(cursor="")
#                 self.drag_hover_cursor = False
    
#     def _on_preview_click(self, event):
#         """Handle mouse click on preview - start drag"""
#         # Check if clicked on a boundary
#         for boundary in self.drag_boundaries:
#             if self._is_near_boundary(event.x, event.y, boundary):
#                 self.drag_active = True
#                 self.drag_type = boundary['type']
#                 self.drag_start_x = event.x
#                 self.drag_start_y = event.y
#                 self.drag_column_index = boundary.get('column_index')
#                 self.logger.debug(f"Started drag: type={self.drag_type}, column={self.drag_column_index}")
#                 break
    
#     def _on_preview_drag(self, event):
#         """Handle dragging on preview - update dimensions"""
#         if not self.drag_active:
#             return
        
#         orientation = self.preview_orientation_var.get()
        
#         if self.drag_type == 'column_boundary':
#             # Portrait: horizontal boundary = adjust individual column HEIGHT
#             # Landscape: vertical boundary = adjust individual column WIDTH
            
#             if orientation == 'portrait':
#                 # Adjust individual column HEIGHT (drag up/down)
#                 delta_y = event.y - self.drag_start_y
                
#                 # Scale delta back to actual size
#                 canvas_height = self.preview_canvas.winfo_height()
#                 actual_height = self.preview_width_var.get()  # Swapped for portrait
                
#                 if canvas_height > 0:
#                     scale = min((canvas_height - 40) / actual_height if actual_height > 0 else 1.0, 1.0)
#                     actual_delta = int(delta_y / scale) if scale > 0 else delta_y
#                 else:
#                     actual_delta = delta_y
                
#                 # Update column size (height in portrait)
#                 if self.drag_column_index is not None and 0 <= self.drag_column_index < len(self.column_rows):
#                     row = self.column_rows[self.drag_column_index]
                    
#                     if row.pack_remaining_var.get():
#                         self.logger.debug("Cannot resize flexible column")
#                         return
                    
#                     current_size = row.column_size_var.get()
#                     new_size = max(30, min(200, current_size + actual_delta))
#                     row.column_size_var.set(new_size)
#                     row._update_header_display()
                    
#                     self.drag_start_y = event.y
#                     self._update_preview()
            
#             else:  # landscape
#                 # Adjust individual column WIDTH (drag left/right)
#                 delta_x = event.x - self.drag_start_x
                
#                 canvas_width = self.preview_canvas.winfo_width()
#                 actual_width = self.preview_width_var.get()
                
#                 if canvas_width > 0:
#                     scale = min((canvas_width - 40) / actual_width if actual_width > 0 else 1.0, 1.0)
#                     actual_delta = int(delta_x / scale) if scale > 0 else delta_x
#                 else:
#                     actual_delta = delta_x
                
#                 # Update column size (width in landscape)
#                 if self.drag_column_index is not None and 0 <= self.drag_column_index < len(self.column_rows):
#                     row = self.column_rows[self.drag_column_index]
                    
#                     if row.pack_remaining_var.get():
#                         self.logger.debug("Cannot resize flexible column")
#                         return
                    
#                     current_size = row.column_size_var.get()
#                     new_size = max(30, min(200, current_size + actual_delta))
#                     row.column_size_var.set(new_size)
#                     row._update_header_display()
                    
#                     self.drag_start_x = event.x
#                     self._update_preview()
        
#         elif self.drag_type == 'global_viz_width':
#             # Portrait: adjust GLOBAL width (all columns) by dragging left/right
#             delta_x = event.x - self.drag_start_x
            
#             canvas_width = self.preview_canvas.winfo_width()
#             actual_width = self.preview_height_var.get()  # Swapped for portrait
            
#             if canvas_width > 0:
#                 scale = min((canvas_width - 40) / actual_width if actual_width > 0 else 1.0, 1.0)
#                 # Convert delta to percentage change
#                 pixel_delta = int(delta_x / scale) if scale > 0 else delta_x
#                 cell_width = self.preview_height_var.get()  # Swapped
#                 percent_delta = int((pixel_delta / cell_width) * 100) if cell_width > 0 else 0
#             else:
#                 percent_delta = 0
            
#             # Update global viz width percentage
#             current_width = self.viz_width_var.get()
#             new_width = max(10, min(50, current_width + percent_delta))
#             self.viz_width_var.set(new_width)
            
#             self.drag_start_x = event.x
#             self._update_preview()
        
#         elif self.drag_type == 'global_viz_height':
#             # Landscape: adjust GLOBAL height (all columns) by dragging up/down
#             delta_y = event.y - self.drag_start_y
            
#             canvas_height = self.preview_canvas.winfo_height()
#             actual_height = self.preview_height_var.get()
            
#             if canvas_height > 0:
#                 scale = min((canvas_height - 40) / actual_height if actual_height > 0 else 1.0, 1.0)
#                 actual_delta = int(delta_y / scale) if scale > 0 else delta_y
#             else:
#                 actual_delta = delta_y
            
#             # Update global viz height
#             current_height = self.viz_height_var.get()
#             new_height = max(20, min(200, current_height + actual_delta))
#             self.viz_height_var.set(new_height)
            
#             self.drag_start_y = event.y
#             self._update_preview()
        
#         elif self.drag_type == 'viz_height':
#             # Legacy - can remove if not used
#             pass

#     def _on_preview_release(self, event):
#         """Handle mouse release - end drag"""
#         if self.drag_active:
#             self.logger.debug(f"Ended drag: type={self.drag_type}")
#             self.drag_active = False
#             self.drag_type = None
#             self.drag_column_index = None
    
#     def _on_preview_leave(self, event):
#         """Handle mouse leaving preview - reset cursor"""
#         self.drag_hover_cursor = False
#         self.preview_canvas.config(cursor="")
    
#     def _decrease_grid_scale(self):
#         """Decrease grid cell dimensions by 10%"""
#         current_width = self.preview_width_var.get()
#         current_height = self.preview_height_var.get()
        
#         new_width = max(200, int(current_width * 0.9))
#         new_height = max(100, int(current_height * 0.9))
        
#         self.preview_width_var.set(new_width)
#         self.preview_height_var.set(new_height)
        
#         # Update label
#         self.cell_size_label.config(text=f"{new_width}×{new_height}px")
        
#         self._update_preview()
#         self.logger.info(f"Decreased grid scale to {new_width}×{new_height}px")
    
#     def _increase_grid_scale(self):
#         """Increase grid cell dimensions by 10%"""
#         current_width = self.preview_width_var.get()
#         current_height = self.preview_height_var.get()
        
#         new_width = min(800, int(current_width * 1.1))
#         new_height = min(600, int(current_height * 1.1))
        
#         self.preview_width_var.set(new_width)
#         self.preview_height_var.set(new_height)
        
#         # Update label
#         self.cell_size_label.config(text=f"{new_width}×{new_height}px")
        
#         self._update_preview()
#         self.logger.info(f"Increased grid scale to {new_width}×{new_height}px")

#     def _is_near_boundary(self, x, y, boundary):
#         """Check if point (x,y) is near a boundary"""
#         tolerance = 5  # pixels
        
#         if boundary['type'] == 'column_boundary':
#             # Can be vertical or horizontal depending on orientation
#             if 'x' in boundary:
#                 # Vertical line (landscape individual widths)
#                 bx = boundary['x']
#                 return abs(x - bx) <= tolerance and boundary['y1'] <= y <= boundary['y2']
#             elif 'y' in boundary:
#                 # Horizontal line (portrait individual heights)
#                 by = boundary['y']
#                 return abs(y - by) <= tolerance and boundary['x1'] <= x <= boundary['x2']
        
#         elif boundary['type'] == 'global_viz_width':
#             # Vertical line for portrait global width
#             bx = boundary['x']
#             return abs(x - bx) <= tolerance and boundary['y1'] <= y <= boundary['y2']
        
#         elif boundary['type'] == 'global_viz_height':
#             # Horizontal line for landscape global height
#             by = boundary['y']
#             return abs(y - by) <= tolerance and boundary['x1'] <= x <= boundary['x2']
        
#         elif boundary['type'] == 'viz_height':
#             # Horizontal line (legacy)
#             by = boundary['y']
#             return abs(y - by) <= tolerance and boundary['x1'] <= x <= boundary['x2']
        
#         return False
    
#     def _rebuild_rows(self):
#         """Rebuild row display after reordering"""
#         # Unpack all containers
#         for row in self.column_rows:
#             row.container.pack_forget()
        
#         # Repack in order
#         for row in self.column_rows:
#             row.container.pack(fill=tk.X, pady=2)
    
#     def _reset_to_default(self):
#         """Reset to default configuration"""
#         # Use proper confirmation dialog
#         confirmed = DialogHelper.confirm_dialog(
#             self.dialog,
#             "Reset to Default",
#             "This will reset all visualization columns to defaults. Continue?",
#             yes_text="Reset",
#             no_text="Cancel"
#         )
        
#         if confirmed:
#             self.logger.info("Resetting to default configuration")
#             # Clear existing
#             for row in self.column_rows:
#                 row.frame.destroy()
#             self.column_rows.clear()
            
#             # Load defaults
#             self._load_default_config()
            
#             self.logger.info(f"Reset complete: {len(self.column_rows)} default columns loaded")
    
#     def _on_done(self):
#         """Handle done button"""
#         # Collect configuration
#         self.result = [row.get_config() for row in self.column_rows]
        
#         # Save viz column configuration
#         self.config_manager.set("viz_columns", self.result)
#         self.logger.info(f"Saved viz column configuration: {self.result}")
        
#         # Save grid display settings
#         self.config_manager.set("viz_column_width_ratio", self.viz_width_var.get() / 100.0)
#         self.config_manager.set("viz_column_height", self.viz_height_var.get())
#         self.config_manager.set("viz_column_font_size", self.viz_font_size_var.get())
#         self.config_manager.set("grid_show_outlines", self.show_outlines_var.get())
#         self.config_manager.set("grid_outline_width", self.outline_width_var.get())
#         self.config_manager.set("grid_show_cell_labels", self.show_cell_labels_var.get())
#         self.config_manager.set("grid_show_classification_labels", self.show_class_labels_var.get())
#         self.config_manager.set("grid_classification_label_position", self.label_position_var.get())
#         self.logger.info(f"Saved grid display settings: width_ratio={self.viz_width_var.get()}%, "
#                         f"height={self.viz_height_var.get()}px, font_size={self.viz_font_size_var.get()}pt, "
#                         f"outlines={self.show_outlines_var.get()}, outline_width={self.outline_width_var.get()}, "
#                         f"cell_labels={self.show_cell_labels_var.get()}, class_labels={self.show_class_labels_var.get()}, "
#                         f"position={self.label_position_var.get()}")
        
#         # Unbind mousewheel to prevent errors after dialog closes
#         if hasattr(self, '_canvas') and self._canvas.winfo_exists():
#             try:
#                 self._canvas.unbind("<MouseWheel>")
#                 self._canvas.unbind("<Button-4>")
#                 self._canvas.unbind("<Button-5>")
#                 self.logger.debug("Unbound mousewheel events")
#             except Exception as e:
#                 self.logger.warning(f"Error unbinding mousewheel: {e}")
        
#         self.dialog.destroy()

#     def _on_cancel(self):
#         """Handle cancel button"""
#         # Unbind mousewheel to prevent errors after dialog closes
#         if hasattr(self, '_canvas') and self._canvas.winfo_exists():
#             try:
#                 self._canvas.unbind("<MouseWheel>")
#                 self._canvas.unbind("<Button-4>")
#                 self._canvas.unbind("<Button-5>")
#                 self.logger.debug("Unbound mousewheel events")
#             except Exception as e:
#                 self.logger.warning(f"Error unbinding mousewheel: {e}")
        
#         self.dialog.destroy()

"""
Dialog for configuring visualization columns displayed in the grid canvas.
Simplified version with global font controls and equal column sizing.
"""

import tkinter as tk
from tkinter import ttk
import logging
from typing import List, Dict, Any, Optional

from gui.dialog_helper import DialogHelper
from gui.widgets.modern_button import ModernButton
from gui.widgets.viz_column_item_row import VizColumnItemRow


class VizColumnSettingsDialog:
    """
    Dialog for configuring visualization columns.
    Simplified: columns fill available space equally, font sizes are global.
    """
    
    def __init__(self, parent, gui_manager, data_coordinator, 
                 config_manager, image_index=None):
        self.parent = parent
        self.gui_manager = gui_manager
        self.theme_colors = gui_manager.theme_colors
        self.data_coordinator = data_coordinator
        self.config_manager = config_manager
        self.image_index = image_index
        
        # Access stores via coordinator
        self.geological_store = data_coordinator.geological_store if data_coordinator else None
        self.color_map_store = data_coordinator.color_maps if data_coordinator else None
        self.logger = logging.getLogger(__name__)
        
        # Get ColorMapManager - prefer from store, fallback to creating new
        self.color_map_manager = None
        if self.color_map_store and hasattr(self.color_map_store, 'manager'):
            self.color_map_manager = self.color_map_store.manager
        
        if self.color_map_manager is None:
            try:
                from processing.LoggingReviewStep.color_map_manager import ColorMapManager
                self.color_map_manager = ColorMapManager(config_manager)
                self.logger.debug("Created fallback ColorMapManager")
            except ImportError as e:
                self.logger.error(f"Could not import ColorMapManager: {e}")
        
        # Use fonts from gui_manager
        self.fonts = gui_manager.fonts
        
        self.dialog = None
        self.result = None
        self.column_rows: List[VizColumnItemRow] = []
        self.selected_row = None
        
        # Get available columns and presets
        self.available_columns = self._get_available_columns()
        self.available_presets = []
        if self.color_map_manager:
            self.available_presets = self.color_map_manager.get_preset_names()
        
        self.logger.debug(f"Initialized with {len(self.available_columns)} columns and {len(self.available_presets)} presets")
        
        if not self.available_presets and self.color_map_manager:
            self.logger.info("No presets found, creating defaults")
            self.color_map_manager.create_default_presets()
            self.available_presets = self.color_map_manager.get_preset_names()

    def _get_available_columns(self):
        """Get all available columns from GeologicalStore with schema metadata.

        Returns clean column names without source suffix unless there are
        duplicate column names across different sources.
        
        Also builds self.column_metadata dict with display_name and color_map
        from the DataManager schema for each column.
        """
        columns = []
        # Track column name -> source mapping for internal use
        self.column_source_map = {}
        # NEW: Track schema metadata (display_name, color_map) for each column
        self.column_metadata = {}

        if self.geological_store and hasattr(self.geological_store, 'get_available_columns'):
            try:
                available = self.geological_store.get_available_columns()
                # Get schemas for looking up display_name and color_map
                schemas = self.geological_store.get_schemas() if hasattr(self.geological_store, 'get_schemas') else {}

                # First pass: collect all column names and detect duplicates
                column_sources = {}  # col_name -> list of sources
                for source_name, cols in available.items():
                    for col_name, col_type in cols:
                        if col_name not in column_sources:
                            column_sources[col_name] = []
                        column_sources[col_name].append(source_name)

                # Second pass: build column list, only add source suffix for duplicates
                for source_name, cols in available.items():
                    schema = schemas.get(source_name)
                    for col_name, col_type in cols:
                        if len(column_sources.get(col_name, [])) > 1:
                            # Column exists in multiple sources - add suffix
                            dropdown_name = f"{col_name} ({source_name})"
                        else:
                            # Column is unique - use clean name
                            dropdown_name = col_name

                        if dropdown_name not in columns:
                            columns.append(dropdown_name)
                            self.column_source_map[dropdown_name] = source_name
                            
                            # Get schema metadata for this column
                            col_schema = schema.get_column(col_name) if schema else None
                            schema_display_name = col_schema.display_name if col_schema else col_name
                            schema_color_map = col_schema.color_map if col_schema else None
                            
                            # Store metadata for later lookup (keyed by dropdown name)
                            self.column_metadata[dropdown_name] = {
                                'display_name': schema_display_name,
                                'color_map': schema_color_map,
                                'source': source_name,
                                'source_name': col_name  # Original column name without suffix
                            }

                self.logger.debug(f"Got {len(columns)} columns from GeologicalStore with schema metadata")
            except Exception as e:
                self.logger.error(f"Error getting GeologicalStore columns: {e}")

        # Add RC metrics columns if available
        if self.data_coordinator and self.data_coordinator.has_rc_metrics:
            try:
                rc_df = self.data_coordinator.get_rc_metrics_dataframe()
                if rc_df is not None and not rc_df.empty:
                    rc_count = 0
                    for col in rc_df.columns:
                        if col.lower() in ('hole_id', 'depth_to'):
                            continue
                        dropdown_name = f"{col} (RC Metrics)"
                        if dropdown_name not in columns:
                            columns.append(dropdown_name)
                            self.column_source_map[dropdown_name] = "RC Metrics"
                            # Add metadata for RC metrics columns (no schema, use col name as display)
                            self.column_metadata[dropdown_name] = {
                                'display_name': col,
                                'color_map': None,
                                'source': "RC Metrics",
                                'source_name': col
                            }
                            rc_count += 1
                    self.logger.debug(f"Added {rc_count} RC metrics columns")
            except Exception as e:
                self.logger.warning(f"Could not add RC metrics columns: {e}")

        if not columns:
            self.logger.warning("No columns found, using defaults")
            columns = [
                "Fe_pct_BEST",
                "SiO2_pct_BEST",
                "Al2O3_pct_BEST",
                "Logged_pct_CHHM",
                "BIFf_2"
            ]
            # Add fallback metadata for default columns
            for col in columns:
                self.column_metadata[col] = {
                    'display_name': col,
                    'color_map': None,
                    'source': 'default',
                    'source_name': col
                }

        return columns

    def show(self) -> Optional[List[Dict[str, str]]]:
        """Show dialog and return result.
        
        Dialog is non-modal to allow dropdown menus to work properly.
        """
        self._create_dialog()
        # Focus the dialog without grab_set() to allow dropdowns to work
        self.dialog.focus_set()
        self.dialog.wait_window()
        return self.result
    
    def _create_dialog(self):
        """Create the dialog window"""
        self.dialog = DialogHelper.create_dialog(
            self.parent,
            title="Configure Data Visualizations",
            modal=False  # Non-modal allows OptionMenu/Combobox dropdowns to work
        )
        
        self.dialog.configure(bg=self.theme_colors["background"])
        self.dialog.geometry("800x550")
        self.dialog.minsize(700, 450)
        
        # Main container
        main_frame = ttk.Frame(self.dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="Configure Visualization Columns",
            font=("Arial", 14, "bold")
        )
        title_label.pack(pady=(0, 5))
        
        subtitle_label = ttk.Label(
            main_frame,
            text="Add data columns to display as colored bars next to images. Columns fill available space equally.",
            font=("Arial", 10)
        )
        subtitle_label.pack(pady=(0, 10))
        
        # ========== COLUMN LIST SECTION ==========
        columns_label = ttk.Label(
            main_frame,
            text="Visualization Columns",
            font=("Arial", 11, "bold")
        )
        columns_label.pack(anchor="w", pady=(0, 5))
        
        # Scrollable area for column rows
        scroll_frame = tk.Frame(main_frame, bg=self.theme_colors["background"],
                               highlightbackground=self.theme_colors["border"],
                               highlightthickness=1)
        scroll_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Canvas and scrollbar
        canvas = tk.Canvas(scroll_frame, bg=self.theme_colors["background"],
                          highlightthickness=0, height=200)
        scrollbar = tk.Scrollbar(scroll_frame, orient="vertical", command=canvas.yview)
        
        self.scrollable_frame = tk.Frame(canvas, bg=self.theme_colors["background"])
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Store canvas reference for cleanup
        self._canvas = canvas
        
        # Bind mousewheel
        def _on_mousewheel(event):
            if canvas.winfo_exists():
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind("<MouseWheel>", _on_mousewheel)
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        
        # Control buttons
        button_frame = tk.Frame(main_frame, bg=self.theme_colors["background"])
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        ModernButton(
            button_frame,
            text="↑ Move Up",
            command=self._move_up,
            color=self.theme_colors["secondary_bg"],
            theme_colors=self.theme_colors
        ).pack(side=tk.LEFT, padx=2)
        
        ModernButton(
            button_frame,
            text="↓ Move Down",
            command=self._move_down,
            color=self.theme_colors["secondary_bg"],
            theme_colors=self.theme_colors
        ).pack(side=tk.LEFT, padx=2)
        
        ModernButton(
            button_frame,
            text="➕ Add Column",
            command=self._add_column,
            color=self.theme_colors["accent_green"],
            theme_colors=self.theme_colors
        ).pack(side=tk.LEFT, padx=2)
        
        ModernButton(
            button_frame,
            text="🎨 Manage Color Maps",
            command=self._open_color_map_manager,
            color=self.theme_colors["accent_blue"],
            theme_colors=self.theme_colors
        ).pack(side=tk.LEFT, padx=10)
        
        ModernButton(
            button_frame,
            text="Reset to Default",
            command=self._reset_to_default,
            color=self.theme_colors["accent_blue"],
            theme_colors=self.theme_colors
        ).pack(side=tk.RIGHT, padx=2)
        
        # ========== GLOBAL SETTINGS SECTION ==========
        separator = ttk.Separator(main_frame, orient='horizontal')
        separator.pack(fill=tk.X, pady=5)
        
        settings_label = ttk.Label(
            main_frame,
            text="Display Settings",
            font=("Arial", 11, "bold")
        )
        settings_label.pack(anchor="w", pady=(5, 5))
        
        settings_frame = tk.Frame(main_frame, bg=self.theme_colors["secondary_bg"],
                                 highlightbackground=self.theme_colors["border"],
                                 highlightthickness=1)
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Row 1: Viz width ratio, font sizes
        row1_frame = tk.Frame(settings_frame, bg=self.theme_colors["secondary_bg"])
        row1_frame.pack(fill=tk.X, padx=10, pady=8)
        
        # Viz column width ratio
        ttk.Label(row1_frame, text="Viz Column Width:").pack(side=tk.LEFT, padx=(0, 5))
        self.viz_width_var = tk.IntVar(value=int(self.config_manager.get("viz_column_width_ratio", 0.30) * 100))
        viz_width_spinbox = tk.Spinbox(
            row1_frame, from_=10, to=50, width=5,
            textvariable=self.viz_width_var,
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["text"]
        )
        viz_width_spinbox.pack(side=tk.LEFT)
        ttk.Label(row1_frame, text="%").pack(side=tk.LEFT, padx=(2, 20))
        
        # Header font size
        ttk.Label(row1_frame, text="Header Font:").pack(side=tk.LEFT, padx=(0, 5))
        self.header_font_size_var = tk.IntVar(value=self.config_manager.get("viz_header_font_size", 7))
        header_font_spinbox = tk.Spinbox(
            row1_frame, from_=5, to=14, width=4,
            textvariable=self.header_font_size_var,
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["text"]
        )
        header_font_spinbox.pack(side=tk.LEFT)
        ttk.Label(row1_frame, text="pt").pack(side=tk.LEFT, padx=(2, 20))
        
        # Value font size
        ttk.Label(row1_frame, text="Value Font:").pack(side=tk.LEFT, padx=(0, 5))
        self.value_font_size_var = tk.IntVar(value=self.config_manager.get("viz_value_font_size", 8))
        value_font_spinbox = tk.Spinbox(
            row1_frame, from_=5, to=14, width=4,
            textvariable=self.value_font_size_var,
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["text"]
        )
        value_font_spinbox.pack(side=tk.LEFT)
        ttk.Label(row1_frame, text="pt").pack(side=tk.LEFT, padx=(2, 0))

        # NOTE: Decimal places removed - should come from Data Manager settings

        # Row 2: Cell outline settings
        row2_frame = tk.Frame(settings_frame, bg=self.theme_colors["secondary_bg"])
        row2_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        
        # Show outlines checkbox
        self.show_outlines_var = tk.BooleanVar(value=self.config_manager.get("grid_show_outlines", True))
        outline_cb = tk.Checkbutton(
            row2_frame, text="Show Cell Outlines",
            variable=self.show_outlines_var,
            bg=self.theme_colors["secondary_bg"],
            fg=self.theme_colors["text"],
            selectcolor=self.theme_colors.get("checkbox_bg", self.theme_colors["field_bg"]),
            activebackground=self.theme_colors["secondary_bg"]
        )
        outline_cb.pack(side=tk.LEFT, padx=(0, 10))
        
        # Outline width
        ttk.Label(row2_frame, text="Width:").pack(side=tk.LEFT, padx=(0, 5))
        self.outline_width_var = tk.IntVar(value=self.config_manager.get("grid_outline_width", 2))
        outline_spinbox = tk.Spinbox(
            row2_frame, from_=1, to=6, width=3,
            textvariable=self.outline_width_var,
            bg=self.theme_colors["field_bg"],
            fg=self.theme_colors["text"]
        )
        outline_spinbox.pack(side=tk.LEFT)
        ttk.Label(row2_frame, text="px").pack(side=tk.LEFT, padx=(2, 20))
        
        # Show cell labels
        self.show_cell_labels_var = tk.BooleanVar(value=self.config_manager.get("grid_show_cell_labels", True))
        cell_labels_cb = tk.Checkbutton(
            row2_frame, text="Show Hole ID / Depth",
            variable=self.show_cell_labels_var,
            bg=self.theme_colors["secondary_bg"],
            fg=self.theme_colors["text"],
            selectcolor=self.theme_colors.get("checkbox_bg", self.theme_colors["field_bg"]),
            activebackground=self.theme_colors["secondary_bg"]
        )
        cell_labels_cb.pack(side=tk.LEFT, padx=(0, 10))
        
        # Show classification labels
        self.show_class_labels_var = tk.BooleanVar(value=self.config_manager.get("grid_show_classification_labels", True))
        class_labels_cb = tk.Checkbutton(
            row2_frame, text="Show Classification Labels",
            variable=self.show_class_labels_var,
            bg=self.theme_colors["secondary_bg"],
            fg=self.theme_colors["text"],
            selectcolor=self.theme_colors.get("checkbox_bg", self.theme_colors["field_bg"]),
            activebackground=self.theme_colors["secondary_bg"]
        )
        class_labels_cb.pack(side=tk.LEFT, padx=(0, 10))
        
        # Label position
        ttk.Label(row2_frame, text="Position:").pack(side=tk.LEFT, padx=(0, 5))
        self.label_position_var = tk.StringVar(value=self.config_manager.get("grid_classification_label_position", "top-right"))
        position_frame = tk.Frame(row2_frame, bg=self.theme_colors["field_bg"],
                                  highlightbackground=self.theme_colors["field_border"],
                                  highlightthickness=1)
        position_frame.pack(side=tk.LEFT)
        position_dropdown = tk.OptionMenu(position_frame, self.label_position_var, "top-right", "top-left")
        self.gui_manager.style_dropdown(position_dropdown, width=9)
        position_dropdown.pack()

        # Row 3: Wet/Dry image preference
        row3_frame = tk.Frame(settings_frame, bg=self.theme_colors["secondary_bg"])
        row3_frame.pack(fill=tk.X, padx=10, pady=(0, 8))

        ttk.Label(row3_frame, text="Image Preference:").pack(side=tk.LEFT, padx=(0, 10))

        self.moisture_pref_var = tk.StringVar(
            value=self.config_manager.get("moisture_preference", "both")
        )

        # Radio buttons for moisture preference
        moisture_options = [
            ("Show Both", "both"),
            ("Prefer Wet", "wet_preferred"),
            ("Prefer Dry", "dry_preferred"),
        ]

        for text, value in moisture_options:
            rb = tk.Radiobutton(
                row3_frame,
                text=text,
                variable=self.moisture_pref_var,
                value=value,
                bg=self.theme_colors["secondary_bg"],
                fg=self.theme_colors["text"],
                selectcolor=self.theme_colors.get("checkbox_bg", self.theme_colors["field_bg"]),
                activebackground=self.theme_colors["secondary_bg"],
                activeforeground=self.theme_colors["text"]
            )
            rb.pack(side=tk.LEFT, padx=(0, 15))

        # ========== BOTTOM BUTTONS ==========
        bottom_frame = tk.Frame(main_frame, bg=self.theme_colors["background"])
        bottom_frame.pack(side=tk.BOTTOM, pady=10)
        
        ModernButton(
            bottom_frame,
            text="Cancel",
            command=self._on_cancel,
            color=self.theme_colors["secondary_bg"],
            theme_colors=self.theme_colors
        ).pack(side=tk.LEFT, padx=5)
        
        ModernButton(
            bottom_frame,
            text="Done",
            command=self._on_done,
            color=self.theme_colors["accent_green"],
            theme_colors=self.theme_colors
        ).pack(side=tk.LEFT, padx=5)
        
        # Load current config
        self._load_current_config()

    def _load_current_config(self):
        """Load current configuration from ConfigManager"""
        saved_config = self.config_manager.get("viz_columns")
        
        self.logger.debug(f"Loading viz config: {saved_config}")
        
        if saved_config and isinstance(saved_config, list):
            self.logger.info(f"Loading {len(saved_config)} saved viz columns")
            for config in saved_config:
                if isinstance(config, dict) and "column" in config:
                    self._add_column_row(
                        config.get("column", ""),
                        config.get("color_map", ""),
                        config.get("custom_label", "")
                    )
        else:
            self._load_default_config()

    def _load_default_config(self):
        """Load default visualization columns from schema.
        
        Finds preferred columns (Fe, SiO2, Al2O3, logged) and uses their
        schema-defined color_map and display_name values as defaults.
        """
        # Preferred column patterns to look for (in priority order)
        preferred_patterns = ["fe_pct", "sio2_pct", "al2o3_pct", "logged_pct", "p_pct", "loi_pct"]
        added_columns = set()
        max_defaults = 4  # Limit default columns
        
        # First pass: find columns matching preferred patterns
        for pattern in preferred_patterns:
            if len(added_columns) >= max_defaults:
                break
            
            for col in self.available_columns:
                if col in added_columns:
                    continue
                    
                col_lower = col.lower()
                if pattern in col_lower:
                    # _add_column_row will use schema defaults for color_map and label
                    self._add_column_row(col)
                    added_columns.add(col)
                    break  # Only add first match for each pattern
        
        # If no preferred columns found, add first available column
        if not added_columns and self.available_columns:
            self._add_column_row(self.available_columns[0])

    def _add_column_row(self, column="", color_map="", custom_label=""):
        """Add a new column configuration row.
        
        Args:
            column: Column name (as shown in dropdown)
            color_map: Color map preset name. If empty, uses schema default.
            custom_label: Display label. If empty, uses schema display_name.
        """
        # Look up schema metadata for defaults if not provided
        metadata = getattr(self, 'column_metadata', {}).get(column, {})
        
        # Use schema color_map as default if not provided
        if not color_map and column:
            color_map = metadata.get('color_map', '') or ''
        
        # Use schema display_name as default label if not provided
        if not custom_label and column:
            custom_label = metadata.get('display_name', column) or column
        
        config = {
            "column": column,
            "color_map": color_map,
            "custom_label": custom_label if custom_label else column
        }
        
        callbacks = {
            'on_update': self._on_row_update,
            'on_remove': self._on_row_remove,
            'on_select': self._on_row_select,
            'on_edit_color_map': self._on_edit_color_map,
            'on_manage_color_maps': self._open_color_map_manager,
            'on_preview_update': lambda: None  # No preview in simplified version
        }
        
        row_widget = VizColumnItemRow(
            parent=self.scrollable_frame,
            config=config,
            gui_manager=self.gui_manager,
            available_columns=self.available_columns,
            available_presets=self.available_presets,
            callbacks=callbacks,
            color_map_manager=self.color_map_manager
        )
        
        self.column_rows.append(row_widget)
        self.logger.debug(f"Added column row: {column} ({color_map})")

    def _on_row_update(self, row_id, changes):
        """Handle row update callback"""
        self.logger.debug(f"Row {row_id} updated: {changes}")

    def _on_row_remove(self, row_id):
        """Handle row removal"""
        if len(self.column_rows) <= 1:
            DialogHelper.show_message(
                self.dialog,
                "Cannot Remove",
                "At least one visualization column must remain.",
                message_type="warning"
            )
            return
        
        # Find and remove the row
        for row in self.column_rows:
            if row.row_id == row_id:
                self.column_rows.remove(row)
                row.destroy()
                self.logger.debug(f"Removed row {row_id}")
                break

    def _on_row_select(self, row_id):
        """Handle row selection"""
        for row in self.column_rows:
            row.set_selected(row.row_id == row_id)
            if row.row_id == row_id:
                self.selected_row = row

    def _on_edit_color_map(self, color_map_name, column_name):
        """Open color map editor for specific color map"""
        self._open_color_map_manager()

    def _add_column(self):
        """Add new column row with schema-based defaults.
        
        Uses color_map and display_name from the DataManager schema
        instead of hardcoded guessing.
        """
        used_columns = [row.column_var.get() for row in self.column_rows]
        available = [col for col in self.available_columns if col not in used_columns]
        
        if available:
            col = available[0]
            # _add_column_row will automatically look up schema defaults
            # for color_map and display_name (custom_label)
            self._add_column_row(col)
        else:
            DialogHelper.show_message(
                self.dialog,
                "No Columns Available",
                "All available columns are already added.",
                message_type="info"
            )

    def _move_up(self):
        """Move selected row up"""
        if not self.selected_row:
            return
        
        idx = self.column_rows.index(self.selected_row)
        if idx > 0:
            # Swap in list
            self.column_rows[idx], self.column_rows[idx-1] = self.column_rows[idx-1], self.column_rows[idx]
            # Repack all
            self._repack_rows()

    def _move_down(self):
        """Move selected row down"""
        if not self.selected_row:
            return
        
        idx = self.column_rows.index(self.selected_row)
        if idx < len(self.column_rows) - 1:
            # Swap in list
            self.column_rows[idx], self.column_rows[idx+1] = self.column_rows[idx+1], self.column_rows[idx]
            # Repack all
            self._repack_rows()

    def _repack_rows(self):
        """Repack all rows in current order"""
        for row in self.column_rows:
            row.main_frame.pack_forget()
        for row in self.column_rows:
            row.main_frame.pack(fill=tk.X, pady=2, padx=5)

    def _reset_to_default(self):
        """Reset to default configuration"""
        confirmed = DialogHelper.confirm_dialog(
            self.dialog,
            "Reset Configuration",
            "Reset all visualization columns to defaults?\n\nThis will remove your current configuration.",
            yes_text="Reset",
            no_text="Cancel"
        )
        
        if confirmed:
            # Clear existing
            for row in self.column_rows:
                row.destroy()
            self.column_rows.clear()
            
            # Load defaults
            self._load_default_config()
            
            self.logger.info("Reset to default configuration")

    def _open_color_map_manager(self):
        """Open color map editor dialog"""
        from gui.color_map_editor_dialog import ColorMapEditorDialog
        
        self.logger.info("Opening color map editor dialog")
        
        # Get sample data for preview
        data_values = []
        data_column = None
        
        if self.selected_row:
            data_column = self.selected_row.column_var.get()
        elif self.column_rows:
            data_column = self.column_rows[0].column_var.get()
        
        if data_column and self.geological_store:
            try:
                column_base = data_column.split(" (")[0].strip()
                
                if hasattr(self.geological_store, 'get_column_values'):
                    data_values = self.geological_store.get_column_values(column_base)
                elif hasattr(self.geological_store, 'get_unique_values'):
                    data_values = list(self.geological_store.get_unique_values(column_base))
                
                if data_values:
                    data_values = data_values[:1000] if len(data_values) > 1000 else data_values
                    
            except Exception as e:
                self.logger.error(f"Error getting sample data: {e}")
        
        dialog = ColorMapEditorDialog(
            parent=self.dialog,
            gui_manager=self.gui_manager,
            color_map_manager=self.color_map_manager,
            data_column=data_column,
            data_values=data_values,
            initial_color_map=None
        )
        
        result = dialog.show()
        
        if result:
            # Refresh presets
            self.available_presets = self.color_map_manager.get_preset_names()
            for row in self.column_rows:
                row.update_color_map_list(self.available_presets)
            
            self.logger.info("Color map editor closed, refreshed presets")

    def _on_done(self):
        """Handle done button"""
        # Collect column configurations
        self.result = [row.get_config() for row in self.column_rows]
        
        # Save viz column configuration
        self.config_manager.set("viz_columns", self.result)
        self.logger.info(f"Saved viz column configuration: {self.result}")
        
        # Save global display settings
        self.config_manager.set("viz_column_width_ratio", self.viz_width_var.get() / 100.0)
        self.config_manager.set("viz_header_font_size", self.header_font_size_var.get())
        self.config_manager.set("viz_value_font_size", self.value_font_size_var.get())
        # NOTE: Decimal places not saved here - comes from Data Manager settings
        self.config_manager.set("grid_show_outlines", self.show_outlines_var.get())
        self.config_manager.set("grid_outline_width", self.outline_width_var.get())
        self.config_manager.set("grid_show_cell_labels", self.show_cell_labels_var.get())
        self.config_manager.set("grid_show_classification_labels", self.show_class_labels_var.get())
        self.config_manager.set("grid_classification_label_position", self.label_position_var.get())
        self.config_manager.set("moisture_preference", self.moisture_pref_var.get())

        self.logger.info(f"Saved display settings: width_ratio={self.viz_width_var.get()}%, "
                        f"header_font={self.header_font_size_var.get()}pt, "
                        f"value_font={self.value_font_size_var.get()}pt")
        
        # Cleanup and close
        self._cleanup()
        self.dialog.destroy()

    def _on_cancel(self):
        """Handle cancel button"""
        self._cleanup()
        self.dialog.destroy()

    def _cleanup(self):
        """Clean up resources before closing"""
        try:
            if hasattr(self, '_canvas') and self._canvas.winfo_exists():
                self._canvas.unbind("<MouseWheel>")
                self._canvas.unbind("<Enter>")
                self._canvas.unbind("<Leave>")
        except Exception as e:
            self.logger.debug(f"Cleanup warning: {e}")