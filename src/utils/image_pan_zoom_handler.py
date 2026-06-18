"""
Image Pan and Zoom Handler for tkinter applications.
Provides mouse wheel zoom and click-drag pan functionality.
"""

import tkinter as tk
from PIL import Image, ImageTk
import logging
from typing import Optional, Tuple


class ImagePanZoomHandler:
    """
    Handles pan and zoom functionality for images displayed in tkinter widgets.
    
    Features:
    - Mouse wheel zoom (centered on cursor position)
    - Click and drag to pan
    - Maintains image quality during zoom
    - Configurable zoom limits and speed
    """
    
    def __init__(self, canvas: tk.Canvas, theme_colors: dict = None):
        """
        Initialize the pan/zoom handler.
        
        Args:
            canvas: The tkinter Canvas widget to handle
            theme_colors: Optional theme colors dictionary
        """
        self.canvas = canvas
        self.theme_colors = theme_colors or {}
        self.logger = logging.getLogger(__name__)
        
        # Image references
        self.original_image: Optional[Image.Image] = None
        self.photo_image: Optional[ImageTk.PhotoImage] = None
        self.canvas_image_id: Optional[int] = None
        
        # Zoom and pan state
        self.zoom_level = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        self.zoom_speed = 1.2  # Zoom factor per wheel click
        
        # Pan state
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.is_panning = False
        
        # Canvas viewport
        self.viewport_x = 0
        self.viewport_y = 0
        
        # Bind events
        self._bind_events()
        
    def _bind_events(self):
        """Bind mouse events for pan and zoom."""
        # Zoom events
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)  # Windows/macOS
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)    # Linux scroll up
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)    # Linux scroll down
        
        # Pan events
        self.canvas.bind("<ButtonPress-1>", self._on_pan_start)
        self.canvas.bind("<B1-Motion>", self._on_pan_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_pan_end)
        
        # Canvas resize
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        
    def set_image(self, image: Image.Image):
        """
        Set the image to display.
        
        Args:
            image: PIL Image object
        """
        self.original_image = image
        self.zoom_level = 1.0
        self.viewport_x = 0
        self.viewport_y = 0
        
        # Calculate minimum zoom to fit entire image
        if self.original_image:
            orig_width, orig_height = self.original_image.size
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                scale_x = canvas_width / orig_width
                scale_y = canvas_height / orig_height
                self.min_zoom = min(scale_x, scale_y, 1.0)  # Don't go below fit-to-window or 100%
            else:
                self.min_zoom = 0.1
                
        self._update_display()
        
    def _calculate_display_size(self) -> Tuple[int, int]:
        """Calculate the size for displaying the image based on zoom level."""
        if not self.original_image:
            return 0, 0
            
        orig_width, orig_height = self.original_image.size
        
        # Get canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not yet rendered, use reasonable defaults
            canvas_width = 800
            canvas_height = 600
        
        if self.zoom_level == 1.0:
            # Fit to canvas while maintaining aspect ratio
            scale_x = canvas_width / orig_width
            scale_y = canvas_height / orig_height
            scale = min(scale_x, scale_y)
            
            display_width = int(orig_width * scale)
            display_height = int(orig_height * scale)
        else:
            # Apply zoom
            base_scale_x = canvas_width / orig_width
            base_scale_y = canvas_height / orig_height
            base_scale = min(base_scale_x, base_scale_y)
            
            display_width = int(orig_width * base_scale * self.zoom_level)
            display_height = int(orig_height * base_scale * self.zoom_level)
            
        return display_width, display_height
        
    def _update_display(self):
        """Update the displayed image."""
        if not self.original_image:
            return
            
        try:
            # Calculate display size
            display_width, display_height = self._calculate_display_size()
            
            if display_width <= 0 or display_height <= 0:
                return
            
            # Resize image
            resized = self.original_image.resize(
                (display_width, display_height),
                Image.Resampling.LANCZOS
            )
            
            # Convert to PhotoImage
            self.photo_image = ImageTk.PhotoImage(resized)
            
            # Update or create canvas image
            if self.canvas_image_id:
                self.canvas.itemconfig(self.canvas_image_id, image=self.photo_image)
            else:
                self.canvas_image_id = self.canvas.create_image(
                    self.viewport_x,
                    self.viewport_y,
                    anchor=tk.NW,
                    image=self.photo_image
                )
            
            # Update canvas scroll region
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
            # Center image if it's smaller than canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            if display_width < canvas_width or display_height < canvas_height:
                # Center the image
                new_x = max(0, (canvas_width - display_width) // 2)
                new_y = max(0, (canvas_height - display_height) // 2)
                if self.canvas_image_id:
                    self.canvas.coords(self.canvas_image_id, new_x, new_y)
                    self.viewport_x = new_x
                    self.viewport_y = new_y
            
        except Exception as e:
            self.logger.error(f"Error updating display: {str(e)}")
            
    def _on_mouse_wheel(self, event):
        """Handle mouse wheel zoom."""
        if not self.original_image:
            return
            
        # Get cursor position relative to canvas
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Determine zoom direction
        if event.num == 4 or event.delta > 0:
            # Zoom in
            new_zoom = self.zoom_level * self.zoom_speed
        else:
            # Zoom out
            new_zoom = self.zoom_level / self.zoom_speed
            
        # Clamp zoom level
        new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))
        
        if new_zoom != self.zoom_level:
            # Calculate zoom center point in image coordinates
            old_width, old_height = self._calculate_display_size()
            
            # Update zoom level
            self.zoom_level = new_zoom
            
            # Get new dimensions
            new_width, new_height = self._calculate_display_size()
            
            # Calculate new viewport position to keep cursor point fixed
            zoom_factor = new_zoom / (self.zoom_level / self.zoom_speed if event.delta > 0 else self.zoom_level * self.zoom_speed)
            
            # Adjust viewport to keep zoom centered on cursor
            self.viewport_x = canvas_x - (canvas_x - self.viewport_x) * (new_width / old_width)
            self.viewport_y = canvas_y - (canvas_y - self.viewport_y) * (new_height / old_height)
            
            # Update display
            self._update_display()
            
            # Move the image to the new viewport position
            if self.canvas_image_id:
                self.canvas.coords(self.canvas_image_id, self.viewport_x, self.viewport_y)
                
    def _on_pan_start(self, event):
        """Start panning operation."""
        self.canvas.configure(cursor="fleur")  # Grabbing hand cursor
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.is_panning = True
        
    def _on_pan_motion(self, event):
        """Handle panning motion."""
        if not self.is_panning or not self.canvas_image_id:
            return
            
        # Calculate movement
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        
        # Update viewport position
        self.viewport_x += dx
        self.viewport_y += dy
        
        # Move the image
        self.canvas.coords(self.canvas_image_id, self.viewport_x, self.viewport_y)
        
        # Update pan start position
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        
    def _on_pan_end(self, event):
        """End panning operation."""
        self.canvas.configure(cursor="hand2")  # Back to pointer cursor
        self.is_panning = False
        
    def _on_canvas_resize(self, event):
        """Handle canvas resize."""
        if self.original_image and self.zoom_level == 1.0:
            # Recenter image when canvas is resized at default zoom
            self._center_image()
            self._update_display()
            
    def _center_image(self):
        """Center the image in the canvas."""
        if not self.canvas_image_id:
            return
            
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        display_width, display_height = self._calculate_display_size()
        
        # Calculate centered position
        self.viewport_x = (canvas_width - display_width) // 2
        self.viewport_y = (canvas_height - display_height) // 2
        
        # Update image position
        self.canvas.coords(self.canvas_image_id, self.viewport_x, self.viewport_y)
        
    def reset_view(self):
        """Reset zoom and center the image."""
        self.zoom_level = 1.0
        self.viewport_x = 0
        self.viewport_y = 0
        self._center_image()
        self._update_display()
        
    def get_image_coords_from_canvas(self, canvas_x: int, canvas_y: int) -> Optional[Tuple[int, int]]:
        """
        Convert canvas coordinates to image coordinates.
        
        Args:
            canvas_x: X coordinate on canvas
            canvas_y: Y coordinate on canvas
            
        Returns:
            Tuple of (x, y) in original image coordinates, or None if outside image
        """
        if not self.original_image or not self.canvas_image_id:
            return None
            
        # Get image position and size
        display_width, display_height = self._calculate_display_size()
        
        # Check if point is within image bounds
        rel_x = canvas_x - self.viewport_x
        rel_y = canvas_y - self.viewport_y
        
        if rel_x < 0 or rel_y < 0 or rel_x >= display_width or rel_y >= display_height:
            return None
            
        # Convert to original image coordinates
        orig_width, orig_height = self.original_image.size
        image_x = int(rel_x * orig_width / display_width)
        image_y = int(rel_y * orig_height / display_height)
        
        return image_x, image_y