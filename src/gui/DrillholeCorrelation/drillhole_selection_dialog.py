"""
Drillhole Selection Dialog
Main dialog for selecting drillholes and defining correlation section lines.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional, Dict, List
import pandas as pd
import logging
from gui.widgets.minimap_canvas import MinimapCanvas
from gui.widgets.drillhole_order_list import DrillholeOrderList
from gui.widgets.dynamic_filter_row import DynamicFilterRow
from gui.widgets.modern_button import ModernButton
from processing.CorrelationStep.section_line_selector import SectionLine


class DrillholeSelectionDialog(tk.Toplevel):
    """Dialog for selecting drillholes along a section line.
    
    Features:
    - Interactive minimap with collar locations
    - Draw section line to select drillholes
    - Filter drillholes by HoleID
    - Reorder selected drillholes
    - Configure section width
    """
    
    def __init__(self, parent, collar_data: pd.DataFrame,
                 gui_manager, config_manager, translator, dialog_helper,
                 initial_selection: Optional[Dict] = None):
        """Initialize drillhole selection dialog.
        
        Args:
            parent: Parent window
            collar_data: DataFrame with columns 'holeid', 'x', 'y', 'z' (lowercase)
            gui_manager: GUIManager for theming
            config_manager: ConfigManager for settings
            translator: Translator for i18n
            dialog_helper: DialogHelper for dialogs
            initial_selection: Optional dict with 'hole_ids', 'section_line' for editing
        """
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)        
        # Store managers
        self.gui_manager = gui_manager
        self.config_manager = config_manager
        self.translator = translator
        self.dialog_helper = dialog_helper
        
        # Get dialog size from config
        dialog_width = self.config_manager.get('correlation_dialog_width', 1200)
        dialog_height = self.config_manager.get('correlation_dialog_height', 800)

        self.title(self.translator.translate("Drillhole Selection"))
        self.geometry(f"{dialog_width}x{dialog_height}")
        
        # Set dialog background to match theme
        self.configure(bg=gui_manager.theme_colors['background'])
        
        # Data
        self.collar_data = collar_data.copy()
        self.filtered_collar_data = collar_data.copy()
        self.initial_selection = initial_selection
        
        # Result
        self.result: Optional[Dict] = None
        
        # State
        self.section_width = self.config_manager.get('correlation_default_section_width', 50.0)
        
        # Create UI
        self._create_ui()
        
        # Load initial selection if provided
        if initial_selection:
            self._load_initial_selection()
        
        # Make modal (only if parent is visible)
        if parent.winfo_viewable():
            self.transient(parent)
            self.grab_set()
            
            # Center on parent
            self.update_idletasks()
            x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
            y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
            self.geometry(f"+{x}+{y}")
        else:
            # Parent is hidden - center on screen instead
            self.update_idletasks()
            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()
            x = (screen_width - self.winfo_width()) // 2
            y = (screen_height - self.winfo_height()) // 2
            self.geometry(f"+{x}+{y}")
    
    def _create_ui(self):
        """Create dialog UI."""
        # Main container
        main_frame = tk.Frame(self, bg=self.gui_manager.theme_colors['background'])
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(
            main_frame,
            text=self.translator.translate("Select Drillholes for Correlation"),
            font=self.gui_manager.fonts['heading'],
            bg=self.gui_manager.theme_colors['background'],
            fg=self.gui_manager.theme_colors['text']
        )
        title_label.pack(pady=(0, 10))
        
        # Filter frame
        self._create_filter_frame(main_frame)
        
        # Main content area (pack before button frame so buttons stay at bottom)
        content_frame = tk.Frame(main_frame, bg=self.gui_manager.theme_colors['background'])
        content_frame.pack(side='top', fill='both', expand=True, pady=10)
        
        # Left panel - Minimap
        left_frame = tk.Frame(content_frame, bg=self.gui_manager.theme_colors['field_bg'], relief='flat', borderwidth=0)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        self._create_minimap_panel(left_frame)
        
        # Right panel - Selection list
        right_frame = tk.Frame(content_frame, bg=self.gui_manager.theme_colors['field_bg'], relief='flat', borderwidth=0)
        right_frame.pack(side='right', fill='both', padx=(5, 0))
        right_frame.config(width=300)
        
        self._create_selection_panel(right_frame)
        
        # Bottom button frame
        self._create_button_frame(main_frame)  

    def _create_filter_frame(self, parent):
        """Create filter controls frame using DynamicFilterRow.
        
        Args:
            parent: Parent widget
        """
        filter_outer_frame = tk.Frame(parent, bg=self.gui_manager.theme_colors['background'])
        filter_outer_frame.pack(fill='x', pady=(0, 5))
        
        # Label
        label = tk.Label(
            filter_outer_frame,
            text=self.translator.translate("Filter Drillholes") + ":",
            font=self.gui_manager.fonts['label'],
            bg=self.gui_manager.theme_colors['background'],
            fg=self.gui_manager.theme_colors['text']
        )
        label.pack(side='left', padx=(0, 10), pady=8)
        
        # Container for filter row (don't expand, fixed width)
        filter_container = tk.Frame(filter_outer_frame, bg=self.gui_manager.theme_colors['background'])
        filter_container.pack(side='left', pady=8, padx=5)
        
        # Create columns info - only filterable columns (lowercase per geological_store standard)
        filterable_columns = ['holeid', 'planned_holeid', 'project']
        
        # Create subset of collar_data with only filterable columns
        self.filterable_collar_data = self.collar_data[filterable_columns].copy()
        
        columns_info = {
            'holeid': {'type': 'text'},
            'planned_holeid': {'type': 'text'},
            'project': {'type': 'text'}
        }
        
        # Create DynamicFilterRow
        self.filter_row = DynamicFilterRow(
            parent=filter_container,
            gui_manager=self.gui_manager,
            columns_info=columns_info,
            register_data=self.filterable_collar_data,
            on_remove_callback=lambda idx: None,  # Single filter row, no removal needed
            index=0
        )
        
        # Set default to filter by holeid with "not null" operator (shows all by default)
        self.filter_row.column_var.set('holeid')
        self.filter_row.operator_var.set('not null')
        
        # Apply and Clear buttons
        self.apply_filter_btn = ModernButton(
            filter_outer_frame,
            text=self.translator.translate("Apply Filter"),
            command=self._apply_filter,
            color=self.gui_manager.theme_colors['accent_blue'],
            theme_colors=self.gui_manager.theme_colors
        )
        self.apply_filter_btn.pack(side='left', padx=5)
        
        self.clear_filter_btn = ModernButton(
            filter_outer_frame,
            text=self.translator.translate("Clear Filter"),
            command=self._clear_filter,
            color=self.gui_manager.theme_colors['secondary_bg'],
            theme_colors=self.gui_manager.theme_colors
        )
        self.clear_filter_btn.pack(side='left', padx=5)
        
        # Info label
        self.filter_info_label = tk.Label(
            filter_outer_frame,
            text=self.translator.translate("Showing %d drillholes") % len(self.collar_data),
            font=self.gui_manager.fonts['small'],
            bg=self.gui_manager.theme_colors['background'],
            fg=self.gui_manager.theme_colors['text']
        )
        self.filter_info_label.pack(side='left', padx=10)
    
    def _create_minimap_panel(self, parent):
        """Create minimap panel.
        
        Args:
            parent: Parent widget
        """
        # Header
        header = tk.Frame(parent, bg=self.gui_manager.theme_colors['accent_blue'])
        header.pack(fill='x')
        
        header_label = tk.Label(
            header,
            text=self.translator.translate("Collar Map"),
            font=self.gui_manager.fonts['label'],
            bg=self.gui_manager.theme_colors['accent_blue'],
            fg=self.gui_manager.theme_colors['text'],
            anchor='w',
            padx=10,
            pady=8
        )
        header_label.pack(fill='x')
        
        # Minimap canvas
        canvas_frame = tk.Frame(parent, bg=self.gui_manager.theme_colors['field_bg'])
        canvas_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.minimap = MinimapCanvas(canvas_frame, self.config_manager, self.gui_manager, width=600, height=450)
        self.minimap.pack(fill='both', expand=True)
        self.minimap.on_selection_changed = self._on_minimap_selection_changed
        
        # Set default section width
        self.minimap.set_default_section_width(self.section_width)
        
        # Load collar data
        print("[DEBUG] Loading collar data into minimap...")
        self.minimap.set_collar_data(self.collar_data)
        print("[DEBUG] Collar data loaded")
        
        # Force immediate zoom to fit and render
        print(f"[DEBUG] Minimap has {len(self.minimap.collar_data) if hasattr(self.minimap, 'collar_data') else 0} collars")
        print(f"[DEBUG] Map bounds: X({self.minimap.map_min_x:.1f} to {self.minimap.map_max_x:.1f}), Y({self.minimap.map_min_y:.1f} to {self.minimap.map_max_y:.1f})")
        
        self.update_idletasks()  # Ensure canvas is sized
        self.minimap.zoom_to_fit()
        self.minimap.render()
        
        print(f"[DEBUG] After zoom_to_fit: zoom_level={self.minimap.zoom_level}")
        print(f"[DEBUG] Canvas size: {self.minimap.winfo_width()}x{self.minimap.winfo_height()}")
        # Controls
        control_frame = tk.Frame(parent, bg=self.gui_manager.theme_colors['field_bg'])
        control_frame.pack(fill='x', padx=5, pady=5)
        
        # Zoom buttons
        zoom_label = tk.Label(
            control_frame,
            text=self.translator.translate("Zoom") + ":",
            font=self.gui_manager.fonts['small'],
            bg=self.gui_manager.theme_colors['field_bg']
        )
        zoom_label.pack(side='left', padx=5)
        
        ModernButton(
            control_frame,
            text="-",
            command=lambda: self._zoom(-0.2),
            color=self.gui_manager.theme_colors['accent_blue'],
            theme_colors=self.gui_manager.theme_colors
        ).pack(side='left', padx=2)
        
        ModernButton(
            control_frame,
            text=self.translator.translate("Fit"),
            command=self.minimap.zoom_to_fit,
            color=self.gui_manager.theme_colors['accent_blue'],
            theme_colors=self.gui_manager.theme_colors
        ).pack(side='left', padx=2)
        
        ModernButton(
            control_frame,
            text="+",
            command=lambda: self._zoom(0.2),
            color=self.gui_manager.theme_colors['accent_blue'],
            theme_colors=self.gui_manager.theme_colors
        ).pack(side='left', padx=2)
        
        # Section line controls - just Clear button (drawing is always active)
        self.clear_line_btn = ModernButton(
            control_frame,
            text=self.translator.translate("Clear Line"),
            command=self._clear_section_line,
            color=self.gui_manager.theme_colors['accent_red'],
            theme_colors=self.gui_manager.theme_colors
        )
        self.clear_line_btn.pack(side='left', padx=2)
        self.clear_line_btn.set_state('disabled')  # Disable initially
    
    def _create_selection_panel(self, parent):
        """Create selection list panel.
        
        Args:
            parent: Parent widget
        """
        # Order list
        self.order_list = DrillholeOrderList(parent, self.gui_manager, self.translator)
        self.order_list.pack(fill='both', expand=True)
        self.order_list.on_order_changed = self._on_order_changed
        self.order_list.on_remove_hole = self._on_remove_hole
        
        # Section width control
        width_frame = tk.Frame(parent, bg=self.gui_manager.theme_colors['field_bg'])
        width_frame.pack(fill='x', padx=10, pady=10)
        
        width_label = tk.Label(
            width_frame,
            text=self.translator.translate("Section Width") + ":",
            font=self.gui_manager.fonts['label'],
            bg=self.gui_manager.theme_colors['field_bg'],
            fg=self.gui_manager.theme_colors['text']
        )
        width_label.pack(side='left')
        
        self.width_var = tk.StringVar(value=str(self.section_width))
        
        width_entry = tk.Entry(
            width_frame,
            textvariable=self.width_var,
            font=self.gui_manager.fonts['normal'],
            width=10
        )
        width_entry.pack(side='left', padx=5)
        
        width_unit_label = tk.Label(
            width_frame,
            text=self.translator.translate("meters"),
            font=self.gui_manager.fonts['small'],
            bg=self.gui_manager.theme_colors['field_bg'],
            fg=self.gui_manager.theme_colors['text']
        )
        width_unit_label.pack(side='left')
        
        ModernButton(
            width_frame,
            text=self.translator.translate("Apply"),
            command=self._apply_section_width,
            color=self.gui_manager.theme_colors['accent_blue'],
            theme_colors=self.gui_manager.theme_colors
        ).pack(side='left', padx=10)
    
    def _create_button_frame(self, parent):
        """Create bottom button frame.
        
        Args:
            parent: Parent widget
        """
        button_frame = tk.Frame(parent, bg=self.gui_manager.theme_colors['background'])
        button_frame.pack(side='bottom', fill='x', pady=(10, 0))
        
        # OK button
        ModernButton(
            button_frame,
            text=self.translator.translate("OK"),
            command=self._on_ok,
            color=self.gui_manager.theme_colors['accent_green'],
            theme_colors=self.gui_manager.theme_colors
        ).pack(side='right', padx=5)
        
        # Cancel button
        ModernButton(
            button_frame,
            text=self.translator.translate("Cancel"),
            command=self._on_cancel,
            color=self.gui_manager.theme_colors['secondary_bg'],
            theme_colors=self.gui_manager.theme_colors
        ).pack(side='right', padx=5)
        
        # Apply button
        ModernButton(
            button_frame,
            text=self.translator.translate("Apply"),
            command=self._on_apply,
            color=self.gui_manager.theme_colors['accent_blue'],
            theme_colors=self.gui_manager.theme_colors
        ).pack(side='right', padx=5)
    
    def _apply_filter(self):
        """Apply the filter from DynamicFilterRow."""
        try:
            # Get filter configuration
            filter_config = self.filter_row.get_filter_config()
            
            # Special handling for "not null" operator - show all
            if filter_config['operator'] == 'not null':
                self.filtered_collar_data = self.collar_data.copy()
                print(f"[DEBUG] Filter 'not null' - showing all {len(self.filtered_collar_data)} collars")
            elif not filter_config['value'] or not filter_config['value'].strip():
                # No value provided - show all
                self.filtered_collar_data = self.collar_data.copy()
                print(f"[DEBUG] No filter value - showing all {len(self.filtered_collar_data)} collars")
            else:
                # Apply filter to each row
                column = filter_config['column']
                operator = filter_config['operator']
                value = filter_config['value'].strip()
                
                print(f"[DEBUG] Applying filter: column={column}, operator={operator}, value={value}")
                
                # Build mask based on operator
                col_data = self.collar_data[column].astype(str).str.upper()
                value_upper = value.upper()
                
                if operator == 'contains':
                    mask = col_data.str.contains(value_upper, na=False)
                elif operator == 'equals':
                    mask = col_data == value_upper
                elif operator == 'starts with':
                    mask = col_data.str.startswith(value_upper, na=False)
                elif operator == 'ends with':
                    mask = col_data.str.endswith(value_upper, na=False)
                elif operator == 'in':
                    # Handle comma-separated list
                    values_list = [v.strip().upper() for v in value.split(',')]
                    mask = col_data.isin(values_list)
                elif operator == 'not in':
                    values_list = [v.strip().upper() for v in value.split(',')]
                    mask = ~col_data.isin(values_list)
                else:
                    # Default to contains
                    mask = col_data.str.contains(value_upper, na=False)
                
                self.filtered_collar_data = self.collar_data[mask].copy()
                print(f"[DEBUG] Filter applied - {len(self.filtered_collar_data)} collars match")
            
            # Update minimap with filtered data
            self.minimap.set_collar_data(self.filtered_collar_data)
            
            # Force zoom to fit and render
            self.minimap.zoom_to_fit()
            self.minimap.render()
            
            # Update info label
            count = len(self.filtered_collar_data)
            self.filter_info_label.config(
                text=self.translator.translate("Showing %d drillholes") % count
            )
            
            print(f"[DEBUG] Filter applied: {count} drillholes shown (from {len(self.collar_data)} total)")
            
        except Exception as e:
            print(f"[ERROR] Error applying filter: {e}")
            import traceback
            traceback.print_exc()
            self.dialog_helper.show_error(
                self.translator.translate("Filter Error"),
                self.translator.translate(f"Error applying filter: {str(e)}")
            )
    
    def _clear_filter(self):
        """Clear the filter and show all drillholes."""
        print("[DEBUG] Clear Filter button clicked!")
        
        # Reset filter row to defaults
        self.filter_row.value_var.set("")
        self.filter_row.value2_var.set("")
        self.filter_row.column_var.set('holeid')
        self.filter_row.operator_var.set('not null')
        
        # Show all data
        self.filtered_collar_data = self.collar_data.copy()
        self.minimap.set_collar_data(self.filtered_collar_data)
        
        # Force zoom to fit and render
        self.minimap.zoom_to_fit()
        self.minimap.render()
        
        # Update info label
        count = len(self.filtered_collar_data)
        self.filter_info_label.config(
            text=self.translator.translate("Showing %d drillholes") % count
        )
        
        print(f"[DEBUG] Filter cleared - now showing {count} drillholes")

    def _zoom(self, delta: float):
        """Zoom in or out, maintaining the center point.
        
        Args:
            delta: Zoom delta (positive = zoom in, negative = zoom out)
        """
        self.logger.debug(f"_zoom called with delta={delta}")
        
        # Calculate new zoom
        old_zoom = self.minimap.zoom_level
        factor = 1 + delta
        new_zoom = old_zoom * factor
        new_zoom = max(
            self.minimap.min_zoom,
            min(self.minimap.max_zoom, new_zoom)
        )
        
        # Get canvas center point
        canvas_w = self.minimap.winfo_width() or 500
        canvas_h = self.minimap.winfo_height() or 400
        center_x = canvas_w / 2
        center_y = canvas_h / 2
        
        # Get map coordinate at center (before zoom)
        map_x, map_y = self.minimap.canvas_to_map(center_x, center_y)
        
        # Update zoom
        self.minimap.zoom_level = new_zoom
        
        # Adjust pan so that map coordinate stays at center
        # Based on: canvas_x = map_x * zoom + pan_x
        # and: canvas_y = -map_y * zoom + pan_y (Y inverted)
        self.minimap.pan_offset_x = center_x - map_x * new_zoom
        self.minimap.pan_offset_y = center_y + map_y * new_zoom  # Plus for inverted Y
        
        self.logger.debug(f"Zoom changed from {old_zoom:.6f} to {new_zoom:.6f}")
        self.minimap.render()
    
    def _clear_section_line(self):
        """Clear section line."""
        self.minimap.clear_section_line()
        self.order_list.clear_all()
        self.clear_line_btn.set_state('disabled')
    
    def _apply_section_width(self):
        """Apply section width change."""
        try:
            width = float(self.width_var.get())
            if width <= 0:
                raise ValueError("Width must be positive")
            
            self.section_width = width
            
            # Update existing section line if it exists
            self.minimap.set_section_line_width(width)
            
            # Also set as default for new lines
            self.minimap.set_default_section_width(width)
            
        except ValueError as e:
            self.dialog_helper.show_error(
                self.translator.translate("Invalid Width"),
                self.translator.translate("Please enter a valid positive number")
            )
    
    def _on_minimap_selection_changed(self, selected_ids: List[str]):
        """Handle selection change from minimap.
        
        Args:
            selected_ids: List of selected hole IDs
        """
        # Get distances from section line
        if self.minimap.section_line and not self.filtered_collar_data.empty:
            # Get collars that match selected_ids
            selected_collars = self.filtered_collar_data[
                self.filtered_collar_data['holeid'].isin(selected_ids)
            ]
            
            # Sort by distance - this adds DISTANCE_ALONG_SECTION column
            sorted_collars = self.minimap.section_line.sort_collars_by_distance(selected_collars)
            
            # Check if sort actually worked (returns dataframe with DISTANCE_ALONG_SECTION column)
            if not sorted_collars.empty and 'distance_along_section' in sorted_collars.columns:
                hole_ids = sorted_collars['holeid'].tolist()
                distances = sorted_collars['distance_along_section'].tolist()
                self.order_list.set_drillholes(hole_ids, distances)
            else:
                # Fallback: just use the selected_ids without distances
                self.order_list.set_drillholes(selected_ids)
        else:
            self.order_list.set_drillholes(selected_ids)
        
        # Enable clear line button if section line exists
        if self.minimap.section_line is not None:
            self.clear_line_btn.set_state('normal')
        else:
            self.clear_line_btn.set_state('disabled')
        
        # Force minimap refresh to show highlighted collars
        self.minimap.render()
        
    def _on_order_changed(self):
        """Handle order change in list."""
        # Could update minimap visualization if needed
        pass
    
    def _on_remove_hole(self, hole_id: str):
        """Handle hole removal from list.
        
        Args:
            hole_id: Removed hole ID
        """
        # Update minimap selection
        current_selection = self.order_list.get_ordered_hole_ids()
        self.minimap.selected_collar_ids = current_selection
        self.minimap.render()
        
        if not current_selection:
            self.clear_line_btn.set_state('disabled')
    
    def _load_initial_selection(self):
        """Load initial selection if provided."""
        if not self.initial_selection:
            return
        
        # Load section line
        if 'section_line' in self.initial_selection:
            section_data = self.initial_selection['section_line']
            self.minimap.section_line = SectionLine.from_dict(section_data)
            self.section_width = section_data['width']
            self.width_var.set(str(self.section_width))
        
        # Load hole data (with ordering and distances)
        if 'hole_data' in self.initial_selection:
            hole_data = self.initial_selection['hole_data']
            hole_ids = [item['hole_id'] for item in hole_data]
            distances = [item['distance'] for item in hole_data]
            self.order_list.set_drillholes(hole_ids, distances)
            
            self.minimap.selected_collar_ids = hole_ids
            self.minimap.render()
            self.clear_line_btn.set_state('normal')
    
    def _on_apply(self):
        """Apply button handler."""
        self._build_result()
        # Don't close dialog on Apply
    
    def _on_ok(self):
        """OK button handler."""
        self._build_result()
        self.destroy()
    
    def _on_cancel(self):
        """Cancel button handler."""
        self.result = None
        self.destroy()
    
    def _build_result(self):
        """Build result dictionary with ordered hole IDs."""
        # Get hole IDs in their current order (respects user reordering)
        hole_ids = self.order_list.get_ordered_hole_ids()
        
        # Store full context for next time dialog is opened
        self.result = {
            'hole_ids': hole_ids,  # This is what calling code uses - ordered list
            # Internal data for reopening dialog with same selection
            '_internal': {
                'hole_data': self.order_list.get_hole_data(),
                'section_line': self.minimap.section_line.to_dict() if self.minimap.section_line else None,
                'section_width': self.section_width
            }
        }
    
    def get_selection_result(self) -> Optional[Dict]:
        """Get selection result.
        
        Returns:
            Dict with 'hole_ids' (ordered list) and '_internal' (for reopening dialog)
            or None if cancelled
        """
        return self.result


def main():
    """Test the dialog standalone."""
    print("Starting drillhole selection dialog test...")
    
    # Create mock collar data
    import numpy as np
    
    np.random.seed(42)
    
    print("Generating test collar data...")
    # Generate collar locations around coordinates 352300, 104100
    center_x = 352300
    center_y = 104100
    
    # Create 6x5 grid (30 holes) with 50m spacing
    x_coords = []
    y_coords = []
    hole_ids = []
    
    for row in range(5):
        for col in range(6):
            # Center the grid around target coordinates
            x = center_x + (col - 2.5) * 50  # Offset by 2.5 to center 6 columns
            y = center_y + (row - 2) * 50    # Offset by 2 to center 5 rows
            
            hole_ids.append(f"OKAB{row*6 + col + 1:03d}")
            x_coords.append(x)
            y_coords.append(y)
    
    # Add some random variation to elevation (realistic terrain)
    z_coords = [100 + np.random.uniform(-10, 30) for _ in range(30)]
    
    collar_data = pd.DataFrame({
        'HOLEID': hole_ids,
        'X': x_coords,
        'Y': y_coords,
        'Z': z_coords
    })
    
    print(f"  Created {len(collar_data)} collars around ({center_x}, {center_y})")
    print(f"  X range: {collar_data['X'].min():.1f} to {collar_data['X'].max():.1f}")
    print(f"  Y range: {collar_data['Y'].min():.1f} to {collar_data['Y'].max():.1f}")
    
    # Create root window
    print("Creating root window...")
    root = tk.Tk()
    root.withdraw()  # Hide root window
    
    # Create mock managers for standalone testing
    print("Creating mock managers...")
    
    class MockGUIManager:
        theme_colors = {
            'background': '#ECF0F1',
            'field_bg': 'white',
            'accent_blue': '#3498DB',
            'accent_green': '#27AE60',
            'accent_red': '#E74C3C',
            'accent_yellow': '#E67E22',
            'text': '#2C3E50',
            'secondary_bg': '#95A5A6'
        }
        
        fonts = {
            'heading': ('Arial', 14, 'bold'),
            'label': ('Arial', 10, 'bold'),
            'normal': ('Arial', 10),
            'small': ('Arial', 9),
            'button': ('Arial', 10, 'bold'),
            'title': ('Arial', 14, 'bold')
        }
    
    class MockConfigManager:
        def get(self, key, default):
            return default
    
    class MockTranslator:
        def translate(self, text):
            return text
    
    class MockDialogHelper:
        def show_error(self, title, message):
            import tkinter.messagebox
            tkinter.messagebox.showerror(title, message)
        
        def show_info(self, title, message):
            import tkinter.messagebox
            tkinter.messagebox.showinfo(title, message)
    
    gui_manager = MockGUIManager()
    config_manager = MockConfigManager()
    translator = MockTranslator()
    dialog_helper = MockDialogHelper()
    
    # Create dialog
    print("Creating dialog...")
    try:
        dialog = DrillholeSelectionDialog(
            root, collar_data,
            gui_manager, config_manager, translator, dialog_helper
        )
        print("Dialog created successfully!")
        print("Waiting for user interaction...")
        
        # Wait for dialog
        root.wait_window(dialog)
        
        print("Dialog closed.")
    except Exception as e:
        print(f"ERROR creating dialog: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Get result
    result = dialog.get_selection_result()
    
    if result:
        print("\n" + "="*60)
        print("SELECTION RESULT")
        print("="*60)
        print(f"\nSelected {len(result['hole_ids'])} drillholes:")
        for i, (hole_id, dist) in enumerate(zip(result['hole_ids'], result['distances']), 1):
            dist_str = f"{dist:.1f}m" if dist is not None else "N/A"
            print(f"  {i}. {hole_id} - Distance: {dist_str}")
        
        if result['section_line']:
            sl = result['section_line']
            print(f"\nSection Line:")
            print(f"  Start: ({sl['start_x']:.1f}, {sl['start_y']:.1f})")
            print(f"  End: ({sl['end_x']:.1f}, {sl['end_y']:.1f})")
            print(f"  Azimuth: {sl['azimuth']:.1f}°")
            print(f"  Length: {sl['length']:.1f}m")
            print(f"  Width: {sl['width']:.1f}m")
        
        print("\n" + "="*60)
    else:
        print("\nSelection cancelled")
    
    root.destroy()


if __name__ == '__main__':
    main()