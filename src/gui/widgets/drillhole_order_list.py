"""
Drillhole Order List Widget
Displays selected drillholes in order with drag-to-reorder functionality.
"""

import tkinter as tk
from tkinter import ttk
from typing import List, Optional, Callable, Dict


class DrillholeOrderList(tk.Frame):
    """List widget for displaying and reordering selected drillholes.
    
    Features:
    - Display drillholes with distances
    - Drag-to-reorder
    - Remove individual items
    - Sort by distance
    - Clear all
    """
    
    def __init__(self, parent, gui_manager, translator, **kwargs):
        """Initialize drillhole order list.
        
        Args:
            parent: Parent widget
            gui_manager: GUIManager for theming
            translator: Translator for i18n
            **kwargs: Additional frame options
        """
        super().__init__(parent, **kwargs)
        
        # Store managers
        self.gui_manager = gui_manager
        self.translator = translator
        
        # Data
        self.hole_data: List[Dict[str, any]] = []  # [{'hole_id': str, 'distance': float}, ...]
        
        # Callbacks
        self.on_order_changed: Optional[Callable[[], None]] = None
        self.on_remove_hole: Optional[Callable[[str], None]] = None
        
        # Drag state
        self.dragged_index: Optional[int] = None
        self.drag_start_y: Optional[int] = None
        
        # Build UI
        self._create_ui()
    
    def _create_ui(self):
        """Create the list UI."""
        # Header frame
        header_frame = tk.Frame(self, bg=self.gui_manager.theme_colors['accent_blue'])
        header_frame.pack(fill='x', padx=2, pady=2)
        
        self.header_label = tk.Label(
            header_frame,
            text=self.translator.translate("Selected Drillholes") + " (0)",
            font=self.gui_manager.fonts['label'],
            bg=self.gui_manager.theme_colors['accent_blue'],
            fg=self.gui_manager.theme_colors['text'],
            anchor='w',
            padx=5,
            pady=5
        )
        self.header_label.pack(fill='x')
        
        # Scrollable list container
        container = tk.Frame(self, bg=self.gui_manager.theme_colors['field_bg'])
        container.pack(fill='both', expand=True, padx=2, pady=2)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(container)
        scrollbar.pack(side='right', fill='y')
        
        # Canvas for scrolling
        self.canvas = tk.Canvas(
            container,
            bg=self.gui_manager.theme_colors['field_bg'],
            highlightthickness=0,
            yscrollcommand=scrollbar.set
        )
        self.canvas.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.canvas.yview)
        
        # Frame inside canvas
        self.list_frame = tk.Frame(self.canvas, bg=self.gui_manager.theme_colors['field_bg'])
        self.canvas_window = self.canvas.create_window(
            (0, 0),
            window=self.list_frame,
            anchor='nw'
        )
        
        # Configure scrolling
        self.list_frame.bind('<Configure>', self._on_frame_configure)
        self.canvas.bind('<Configure>', self._on_canvas_configure)
        
        # Button frame
        button_frame = tk.Frame(self, bg=self.gui_manager.theme_colors['background'])
        button_frame.pack(fill='x', padx=2, pady=2)
        
        self.sort_button = tk.Button(
            button_frame,
            text=self.translator.translate("Sort by Distance"),
            command=self.sort_by_distance,
            bg=self.gui_manager.theme_colors['accent_blue'],
            fg=self.gui_manager.theme_colors['text'],
            relief='flat',
            padx=10,
            pady=5,
            font=self.gui_manager.fonts['normal']
        )
        self.sort_button.pack(side='left', padx=2)
        
        self.clear_button = tk.Button(
            button_frame,
            text=self.translator.translate("Clear All"),
            command=self.clear_all,
            bg=self.gui_manager.theme_colors['accent_red'],
            fg=self.gui_manager.theme_colors['text'],
            relief='flat',
            padx=10,
            pady=5,
            font=self.gui_manager.fonts['normal']
        )
        self.clear_button.pack(side='left', padx=2)
    
    def _on_frame_configure(self, event=None):
        """Update scrollregion when frame size changes."""
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))
    
    def _on_canvas_configure(self, event):
        """Update frame width when canvas is resized."""
        self.canvas.itemconfig(self.canvas_window, width=event.width)
    
    def set_drillholes(self, hole_ids: List[str], distances: Optional[List[float]] = None):
        """Set the list of drillholes.
        
        Args:
            hole_ids: List of drillhole IDs
            distances: Optional list of distances along section (same order as hole_ids)
        """
        self.hole_data.clear()
        
        if distances is None:
            distances = [None] * len(hole_ids)
        
        for hole_id, distance in zip(hole_ids, distances):
            self.hole_data.append({
                'hole_id': hole_id,
                'distance': distance
            })
        
        self._rebuild_list()
    
    def add_drillhole(self, hole_id: str, distance: Optional[float] = None):
        """Add a drillhole to the list.
        
        Args:
            hole_id: Drillhole ID
            distance: Distance along section
        """
        # Check if already exists
        if any(item['hole_id'] == hole_id for item in self.hole_data):
            return
        
        self.hole_data.append({
            'hole_id': hole_id,
            'distance': distance
        })
        
        self._rebuild_list()
    
    def remove_drillhole(self, hole_id: str):
        """Remove a drillhole from the list.
        
        Args:
            hole_id: Drillhole ID to remove
        """
        self.hole_data = [item for item in self.hole_data if item['hole_id'] != hole_id]
        self._rebuild_list()
        
        if self.on_remove_hole:
            self.on_remove_hole(hole_id)
    
    def clear_all(self):
        """Remove all drillholes from the list."""
        if self.hole_data:
            self.hole_data.clear()
            self._rebuild_list()
            
            if self.on_order_changed:
                self.on_order_changed()
    
    def sort_by_distance(self):
        """Sort drillholes by distance along section."""
        # Separate items with and without distances
        with_distance = [item for item in self.hole_data if item['distance'] is not None]
        without_distance = [item for item in self.hole_data if item['distance'] is None]
        
        # Sort items with distances
        with_distance.sort(key=lambda x: x['distance'])
        
        # Combine
        self.hole_data = with_distance + without_distance
        
        self._rebuild_list()
        
        if self.on_order_changed:
            self.on_order_changed()
    
    def get_ordered_hole_ids(self) -> List[str]:
        """Get current drillhole order.
        
        Returns:
            List of hole IDs in current order
        """
        return [item['hole_id'] for item in self.hole_data]
    
    def get_hole_data(self) -> List[Dict[str, any]]:
        """Get complete hole data with distances.
        
        Returns:
            List of dicts with 'hole_id' and 'distance' keys
        """
        return self.hole_data.copy()
    
    def _rebuild_list(self):
        """Rebuild the list display."""
        # Clear existing items
        for widget in self.list_frame.winfo_children():
            widget.destroy()
        
        # Update header
        count = len(self.hole_data)
        self.header_label.config(
            text=self.translator.translate("Selected Drillholes") + f" ({count})"
        )
        
        # Create list items
        for idx, item in enumerate(self.hole_data):
            self._create_list_item(idx, item)
        
        # Update scroll region
        self.list_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))
    
    def _create_list_item(self, index: int, item: Dict[str, any]):
        """Create a single list item widget.
        
        Args:
            index: Item index
            item: Item data dict
        """
        # Item frame
        item_frame = tk.Frame(
            self.list_frame,
            bg=self.gui_manager.theme_colors['field_bg'],
            relief='flat',
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=self.gui_manager.theme_colors.get('field_border', '#3f3f3f')
        )
        item_frame.pack(fill='x', padx=5, pady=2)
        
        # Number label
        number_label = tk.Label(
            item_frame,
            text=f"{index + 1}.",
            font=self.gui_manager.fonts['label'],
            bg=self.gui_manager.theme_colors['field_bg'],
            fg=self.gui_manager.theme_colors['text'],
            width=3,
            anchor='e'
        )
        number_label.pack(side='left', padx=(5, 2))
        
        # Hole ID label
        hole_id = item['hole_id']
        distance = item['distance']
        
        if distance is not None:
            label_text = f"{hole_id} ({distance:.1f}m)"
        else:
            label_text = hole_id
        
        hole_label = tk.Label(
            item_frame,
            text=label_text,
            font=self.gui_manager.fonts['normal'],
            bg=self.gui_manager.theme_colors['field_bg'],
            fg=self.gui_manager.theme_colors['text'],
            anchor='w'
        )
        hole_label.pack(side='left', fill='x', expand=True, padx=5)
        
        # Remove button
        remove_btn = tk.Button(
            item_frame,
            text="×",
            font=self.gui_manager.fonts['title'],
            bg=self.gui_manager.theme_colors['accent_red'],
            fg=self.gui_manager.theme_colors['text'],
            relief='flat',
            width=2,
            command=lambda: self.remove_drillhole(hole_id)
        )
        remove_btn.pack(side='right', padx=2, pady=2)
        
        # Bind drag events
        item_frame.bind('<Button-1>', lambda e, i=index: self._on_drag_start(e, i))
        item_frame.bind('<B1-Motion>', lambda e, i=index: self._on_drag_motion(e, i))
        item_frame.bind('<ButtonRelease-1>', lambda e, i=index: self._on_drag_end(e, i))
        
        # Also bind to labels for better UX
        for widget in [number_label, hole_label]:
            widget.bind('<Button-1>', lambda e, i=index: self._on_drag_start(e, i))
            widget.bind('<B1-Motion>', lambda e, i=index: self._on_drag_motion(e, i))
            widget.bind('<ButtonRelease-1>', lambda e, i=index: self._on_drag_end(e, i))
        
        # Change cursor on hover
        for widget in [item_frame, number_label, hole_label]:
            widget.bind('<Enter>', lambda e: e.widget.config(cursor='hand2'))
            widget.bind('<Leave>', lambda e: e.widget.config(cursor=''))
    
    def _on_drag_start(self, event, index: int):
        """Start dragging an item.
        
        Args:
            event: Mouse event
            index: Item index being dragged
        """
        self.dragged_index = index
        self.drag_start_y = event.y_root
        
        # Gray out the dragged item
        item_frames = self.list_frame.winfo_children()
        if 0 <= index < len(item_frames):
            item_frames[index].config(
                bg=self.gui_manager.theme_colors.get('secondary_bg', '#888888'),
                relief='flat'
            )
    
    def _on_drag_motion(self, event, index: int):
        """Handle drag motion - show drop indicator line.
        
        Args:
            event: Mouse event
            index: Item index being dragged
        """
        if self.dragged_index is None:
            return
        
        # Don't process if we haven't moved enough
        if abs(event.y_root - self.drag_start_y) < 5:
            return
        
        # Calculate which position the mouse is over
        y_pos = event.y_root - self.list_frame.winfo_rooty()
        item_frames = self.list_frame.winfo_children()
        
        if not item_frames:
            return
        
        # Find target position (gap between items where we'd insert)
        target_gap = 0  # 0 means before first item
        cumulative_height = 0
        
        for i, frame in enumerate(item_frames):
            frame_height = frame.winfo_height()
            if y_pos < cumulative_height + frame_height / 2:
                target_gap = i
                break
            cumulative_height += frame_height
            target_gap = i + 1
        
        # Reset all frames to normal
        bg_color = self.gui_manager.theme_colors['field_bg']
        for i, frame in enumerate(item_frames):
            if i == self.dragged_index:
                # Keep dragged item grayed out
                frame.config(
                    bg=self.gui_manager.theme_colors.get('secondary_bg', '#888888'),
                    relief='flat',
                    highlightthickness=0
                )
            else:
                # Normal
                frame.config(
                    bg=bg_color,
                    relief='flat',
                    highlightthickness=1,
                    highlightbackground=self.gui_manager.theme_colors.get('field_border', '#3f3f3f')
                )
        
        # Show thick line indicator at drop position
        if target_gap < len(item_frames):
            # Show line above this item
            item_frames[target_gap].config(
                highlightthickness=3,
                highlightbackground=self.gui_manager.theme_colors.get('accent_yellow', '#FFD700')
            )
        elif target_gap > 0 and target_gap - 1 < len(item_frames):
            # Show line below last item
            item_frames[target_gap - 1].config(
                highlightthickness=3,
                highlightbackground=self.gui_manager.theme_colors.get('accent_yellow', '#FFD700')
            )
    
    def _on_drag_end(self, event, index: int):
        """Finish dragging an item - perform the reorder.
        
        Args:
            event: Mouse event
            index: Item index being dragged
        """
        if self.dragged_index is None:
            return
        
        # Calculate final drop position
        y_pos = event.y_root - self.list_frame.winfo_rooty()
        item_frames = self.list_frame.winfo_children()
        
        if not item_frames:
            self.dragged_index = None
            return
        
        # Find target gap
        target_gap = 0
        cumulative_height = 0
        
        for i, frame in enumerate(item_frames):
            frame_height = frame.winfo_height()
            if y_pos < cumulative_height + frame_height / 2:
                target_gap = i
                break
            cumulative_height += frame_height
            target_gap = i + 1
        
        # Convert gap to target index
        # If dragging down, target_gap is where we insert after removal
        # If dragging up, target_gap is direct
        target_index = target_gap
        if target_gap > self.dragged_index:
            target_index = target_gap - 1
        
        # Reorder data
        if target_index != self.dragged_index and 0 <= target_index < len(self.hole_data):
            item = self.hole_data.pop(self.dragged_index)
            self.hole_data.insert(target_index, item)
            
            # Rebuild list
            self._rebuild_list()
            
            # Callback
            if self.on_order_changed:
                self.on_order_changed()
        else:
            # Just reset visual
            self._rebuild_list()
        
        # Reset drag state
        self.dragged_index = None
        self.drag_start_y = None