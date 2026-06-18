"""
Original Image Viewer Dialog - Fully Optimized

Shows original chip tray images with:
- Top: 20 compartment slots centered on current image depth range
- Middle: Current original image (pan/zoom when corners off, fit when corners on)
- Bottom: Filmstrip grouped by depth interval with navigation

Performance optimizations:
- Single directory scan with UID caching
- Downscaled working copies for smooth interaction
- Lazy loading of compartment images
"""

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Set
from PIL import Image, ImageTk, ImageDraw, ImageFont
import os
import shutil
import re
from collections import defaultdict
import gc
import time

from gui.dialog_helper import DialogHelper
from gui.widgets.modern_button import ModernButton
from gui.widgets.themed_searchable_optionmenu import ThemedSearchableOptionMenu

logger = logging.getLogger(__name__)


class ImageCache:
    """Manages full-resolution image caching with memory management."""
    
    def __init__(self, max_cache_size=10):
        self.max_cache_size = max_cache_size
        self.cache = {}  # path -> PIL Image
        self.access_order = []  # LRU tracking
        
    def get(self, path: str) -> Optional[Image.Image]:
        """Get image from cache or load it."""
        path = str(path)
        
        if path in self.cache:
            self.access_order.remove(path)
            self.access_order.append(path)
            return self.cache[path]
        
        try:
            img = Image.open(path)
            img.load()
            
            while len(self.cache) >= self.max_cache_size and self.access_order:
                oldest = self.access_order.pop(0)
                if oldest in self.cache:
                    del self.cache[oldest]
            
            self.cache[path] = img
            self.access_order.append(path)
            return img
        except Exception as e:
            logger.error(f"Failed to load image {path}: {e}")
            return None
    
    def clear(self):
        """Clear all cached images."""
        self.cache.clear()
        self.access_order.clear()
        gc.collect()


class PanZoomImageViewer:
    """Image viewer with pan and zoom capabilities."""
    
    def __init__(self, canvas: tk.Canvas):
        self.canvas = canvas
        self.original_image = None
        self.working_image = None  # Full resolution
        self.preview_image = None  # Downscaled for fast interaction
        self.display_image = None
        self.photo = None
        
        # Pan/zoom state
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.min_scale = 0.1
        self.max_scale = 5.0
        
        # Drag state
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.is_dragging = False
        self.is_zooming = False
        
        # Pan/zoom enabled flag
        self.pan_zoom_enabled = True
        
        # Performance: delayed full-res rendering
        self._pending_render = None
        self._use_preview = False

        # Corner hover tracking
        self._corner_hover_callback = None
        self._current_corners_data = None
        
        # Bind events
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_drag_end)
        self.canvas.bind("<MouseWheel>", self._on_zoom)
        
    def load_image(self, image: Image.Image) -> bool:
        """Load a PIL image for display with performance optimization."""
        try:
            self.original_image = image
            
            # Downscale for reasonable display size (max 5000px on longest side)
            max_dimension = 5000
            if image.width > max_dimension or image.height > max_dimension:
                scale = min(max_dimension / image.width, max_dimension / image.height)
                logger.info(f"Downscaling image from {image.size} to {int(image.width * scale)}x{int(image.height * scale)} for display")
                self.working_image = image.resize(
                    (int(image.width * scale), int(image.height * scale)), 
                    Image.LANCZOS
                )
                self.downscale_factor = scale
            else:
                self.working_image = image
                self.downscale_factor = 1.0
            
            # Create preview image - reduced color palette for fast interaction
            preview = self.working_image.convert('RGB')
            preview = preview.quantize(colors=64).convert('RGB')
            self.preview_image = preview
            
            logger.debug(f"Created preview image: {self.working_image.size} with reduced color palette")
            
            self.scale = 1.0
            self.offset_x = 0
            self.offset_y = 0
            self._use_preview = False
            self.update_display()
            return True
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return False
    
    def set_pan_zoom_enabled(self, enabled: bool):
        """Enable or disable pan/zoom interaction."""
        self.pan_zoom_enabled = enabled
        
    def fit_to_canvas(self):
        """Fit image to canvas size."""
        if not self.working_image:
            return
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width < 10 or canvas_height < 10:
            return
        
        width_scale = canvas_width / self.working_image.width
        height_scale = canvas_height / self.working_image.height
        self.scale = min(width_scale, height_scale) * 0.95
        
        self.offset_x = 0
        self.offset_y = 0
        self.update_display()
        
    def update_display(self, force_full_res=False):
        """Update the canvas display."""
        # Throttle updates to prevent lag (max 30fps)
        if hasattr(self, '_last_update'):
            if time.time() - self._last_update < 0.03:  # 30fps max
                return
        self._last_update = time.time()
        
        if not self.working_image:
            return
        
        # Choose source image: preview during interaction (no overlay), full-res when idle (with overlay)
        if self._use_preview and not force_full_res and self.preview_image:
            # Use preview without overlay for smooth interaction
            source_image = self.preview_image
            # Adjust scale for preview
            preview_scale_factor = self.preview_image.width / self.working_image.width
            effective_scale = self.scale * preview_scale_factor
            logger.debug("Using preview image (corners hidden during interaction)")
        else:
            # Use the overlay image if available, otherwise working image
            source_image = getattr(self, 'overlay_image', self.working_image)
            effective_scale = self.scale
        
        # Calculate display size
        display_width = int(source_image.width * effective_scale)
        display_height = int(source_image.height * effective_scale)
        
        # Resize for display (use fast resampling during interaction)
        if effective_scale != 1.0:
            # Use NEAREST for preview (fast), LANCZOS for full-res (quality)
            resample_method = Image.NEAREST if self._use_preview else Image.LANCZOS
            self.display_image = source_image.resize(
                (display_width, display_height),
                resample_method
            )
        else:
            self.display_image = source_image
        
        self.photo = ImageTk.PhotoImage(self.display_image)
        
        # Clear and draw
        self.canvas.delete("all")
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        x = canvas_width // 2 + self.offset_x
        y = canvas_height // 2 + self.offset_y
        
        self.canvas.create_image(x, y, image=self.photo, anchor="center")
        
    def draw_corner_overlays(self, corners_data: List[Dict], depth_to_compartment_map: Dict, store_for_redraw=True, original_image_size: tuple = None, uid: str = None):
        """Draw corner outlines on the image with depth labels - uses cached overlay if available."""
        if not self.working_image:
            logger.warning("Cannot draw corners: no working image loaded")
            return
        
        if not corners_data:
            logger.warning("Cannot draw corners: no corner data provided")
            return
        
        # Store for redrawing after pan/zoom (only on initial draw)
        if store_for_redraw:
            self._pending_corners_data = (corners_data, depth_to_compartment_map, original_image_size, uid)
        
        # Check if we have a cached overlay for this UID
        if uid and hasattr(self, 'preloaded_overlay_images') and uid in self.preloaded_overlay_images:
            logger.debug(f"✅ Using cached overlay image for UID {uid}")
            self.overlay_image = self.preloaded_overlay_images[uid]
            self.update_display(force_full_res=True)
            logger.info(f"   Called update_display(force_full_res=True) with cached overlay")
            return
        
        # Scale corner coordinates if image was downscaled
        if original_image_size and original_image_size != self.working_image.size:
            scale_x = self.working_image.width / original_image_size[0]
            scale_y = self.working_image.height / original_image_size[1]
            logger.info(f"Scaling corner coordinates: {scale_x:.3f}x, {scale_y:.3f}y")
            
            # Scale all corner coordinates
            scaled_corners_data = []
            for corner_data in corners_data:
                scaled_corner = corner_data.copy()
                if 'corners' in scaled_corner:
                    scaled_corner['corners'] = [
                        (int(x * scale_x), int(y * scale_y)) 
                        for x, y in corner_data['corners']
                    ]
                scaled_corners_data.append(scaled_corner)
            corners_data = scaled_corners_data
        
        logger.info(f"Drawing corner overlays for {len(corners_data)} compartments")
        
        # Create overlay on working image
        overlay = self.working_image.copy()
        draw = ImageDraw.Draw(overlay, 'RGBA')
        
        # Calculate scaling based on image dimensions
        # Use image height as reference (typical chip tray image is ~4000px tall)
        img_height = overlay.size[1]
        
        # If original_image_size provided, use it to determine proper scaling
        if original_image_size:
            # Calculate relative to original size for proper proportions
            reference_height = original_image_size[1]
            scale_factor = reference_height / 4000.0
        else:
            # Fall back to using current image size
            scale_factor = img_height / 4000.0
        
        # Scale line width and font size - proportional to original image
        # But apply downscale factor if image was downscaled
        downscale_ratio = img_height / (original_image_size[1] if original_image_size else img_height)
        
        line_width = max(2, int(20 * scale_factor * downscale_ratio))  # Scales properly with downscaled images
        font_size = max(30, int(80 * scale_factor * downscale_ratio))  # Scales properly with downscaled images
        padding = max(8, int(25 * scale_factor * downscale_ratio))  # Scales properly with downscaled images
        
        logger.info(f"Overlay scaling: image={overlay.size}, scale_factor={scale_factor:.2f}, line_width={line_width}px, font_size={font_size}pt")
        
        # Try to load a scaled font
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("arialbd.ttf", font_size)  # Try bold
            except:
                font = ImageFont.load_default()
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta']
        
        drawn_count = 0
        for idx, corner_data in enumerate(corners_data):
            if 'corners' not in corner_data:
                logger.warning(f"Corner data {idx}: missing 'corners' key")
                continue
            
            corners = corner_data['corners']
            depth = corner_data.get('depth', 0)
            
            if len(corners) != 4:
                logger.warning(f"Corner data {idx} (depth {depth}m): invalid corners count {len(corners)}")
                continue
            
            logger.debug(f"Drawing corner {idx} at depth {depth}m: {corners}")
            
            # Draw bright green outline only (no fill) - scaled thickness
            draw.polygon(corners, outline='#00FF00', width=line_width)
            
            # Draw depth label BELOW the compartment (at bottom-left corner)
            # Corners are typically [bottom-left, bottom-right, top-right, top-left]
            label_x = corners[0][0]
            label_y = corners[0][1] + 5  # 5px below bottom edge
            
            label = f"{depth}m"
            
            # Draw background box with scaled padding
            bbox = draw.textbbox((label_x, label_y), label, font=font, anchor="lt")
            bg_box = [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding]
            draw.rectangle(bg_box, fill=(0, 200, 0, 230))  # Green semi-transparent background (matches border)
            
            # Draw bold white text
            # Note: PIL doesn't support bold attribute directly, font is already loaded
            draw.text((label_x, label_y), label, fill='white', font=font, anchor="lt")
            
            drawn_count += 1
            # Calculate label position for logging
            label_pos_x = corners[0][0]
            label_pos_y = corners[0][1]
            logger.debug(f"  Corner {idx}: depth {depth}m, label at ({label_pos_x:.0f}, {label_pos_y:.0f})")
        
        logger.info(f"✅ Successfully drew {drawn_count}/{len(corners_data)} corner overlays")
        logger.info(f"   Overlay image size: {overlay.size}")
        logger.info(f"   Overlay will be displayed on canvas")
        
        # Store overlay and update display
        self.overlay_image = overlay
        logger.info(f"   Stored overlay_image attribute")
        
        # Cache the drawn overlay for this UID for instant reuse
        if uid and hasattr(self, 'preloaded_overlay_images'):
            self.preloaded_overlay_images[uid] = overlay.copy()
            logger.debug(f"   Cached overlay image for UID {uid}")
        
        # Store corners data for hover detection (store the scaled coordinates)
        self._current_corners_data = corners_data
        
        self.update_display(force_full_res=True)
        logger.info(f"   Called update_display(force_full_res=True)")
        
    def clear_overlays(self):
        """Remove corner overlays."""
        if hasattr(self, 'overlay_image'):
            delattr(self, 'overlay_image')
            logger.debug("Cleared corner overlays")
        # Clear pending corners data
        if hasattr(self, '_pending_corners_data'):
            self._pending_corners_data = None
        self.update_display(force_full_res=True)
        
    def _hex_to_rgba(self, color_name: str, alpha: int = 255) -> Tuple[int, int, int, int]:
        """Convert color name to RGBA tuple."""
        color_map = {
            'red': (255, 0, 0),
            'blue': (0, 0, 255),
            'green': (0, 255, 0),
            'orange': (255, 165, 0),
            'purple': (128, 0, 128),
            'yellow': (255, 255, 0),
            'cyan': (0, 255, 255),
            'magenta': (255, 0, 255),
        }
        rgb = color_map.get(color_name, (255, 255, 255))
        return rgb + (alpha,)
        
    def _on_drag_start(self, event):
        """Start dragging."""
        if not self.pan_zoom_enabled:
            return
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.is_dragging = True
        self._use_preview = True  # Switch to preview for fast interaction
        self._cancel_pending_render()
        
    def _on_drag(self, event):
        """Handle dragging."""
        if not self.pan_zoom_enabled or not self.is_dragging:
            return
        
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        
        self.offset_x += dx
        self.offset_y += dy
        
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        
        self.update_display()
        
    def _on_drag_end(self, event):
        """End dragging."""
        self.is_dragging = False
        self._schedule_full_res_render()
        
    def _on_zoom(self, event):
        """Handle zoom towards mouse position."""
        if not self.pan_zoom_enabled:
            return
        
        self._use_preview = True  # Switch to preview for fast zooming
        self._cancel_pending_render()
        
        # Get mouse position in canvas coordinates
        mouse_x = event.x
        mouse_y = event.y
        
        # Calculate mouse position in image coordinates (before zoom)
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Current image position on canvas
        img_x_before = -self.offset_x
        img_y_before = -self.offset_y
        
        # Mouse position relative to image (before zoom)
        img_mouse_x = (mouse_x - canvas_width/2) / self.scale + img_x_before
        img_mouse_y = (mouse_y - canvas_height/2) / self.scale + img_y_before
        
        # Zoom in/out
        old_scale = self.scale
        if event.delta > 0:
            self.scale *= 1.1
        else:
            self.scale /= 1.1
        
        self.scale = max(self.min_scale, min(self.scale, self.max_scale))
        
        # Adjust offset to keep mouse point stationary
        # New image position that keeps mouse point in same place
        new_img_x = img_mouse_x - (mouse_x - canvas_width/2) / self.scale
        new_img_y = img_mouse_y - (mouse_y - canvas_height/2) / self.scale
        
        self.offset_x = -new_img_x
        self.offset_y = -new_img_y
        
        self.update_display()
        
        # Schedule full-res render after zooming stops
        self._schedule_full_res_render()
    
    def _on_mouse_motion(self, event):
        """Handle mouse motion for corner hover detection."""
        if not self._corner_hover_callback or not self._current_corners_data:
            return
        
        # Get mouse position relative to displayed image
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Convert to image coordinates (accounting for scale and offset)
        if not self.working_image:
            return
        
        # Get image position on canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        img_width = int(self.working_image.width * self.scale)
        img_height = int(self.working_image.height * self.scale)
        
        img_x = (canvas_width - img_width) / 2 + self.offset_x
        img_y = (canvas_height - img_height) / 2 + self.offset_y
        
        # Check if mouse is over the image
        if canvas_x < img_x or canvas_x > img_x + img_width:
            self._corner_hover_callback(None, False)
            return
        if canvas_y < img_y or canvas_y > img_y + img_height:
            self._corner_hover_callback(None, False)
            return
        
        # Convert to image pixel coordinates
        rel_x = (canvas_x - img_x) / self.scale
        rel_y = (canvas_y - img_y) / self.scale
        
        # Check which corner polygon contains this point
        for corner_data in self._current_corners_data:
            if 'corners' not in corner_data:
                continue
            
            corners = corner_data['corners']
            depth = corner_data.get('depth')
            
            # Check if point is inside polygon using ray casting
            if self._point_in_polygon(rel_x, rel_y, corners):
                self._corner_hover_callback(depth, True)
                return
        
        # Not over any corner
        self._corner_hover_callback(None, False)
    
    def _point_in_polygon(self, x, y, polygon):
        """Check if point (x,y) is inside polygon using ray casting algorithm."""
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def set_corner_hover_callback(self, callback, corners_data):
        """Set callback for corner hover events."""
        self._corner_hover_callback = callback
        self._current_corners_data = corners_data

    def _schedule_full_res_render(self):
        """Schedule full resolution render after interaction stops."""
        self._cancel_pending_render()
        # Render full-res after 200ms of no interaction
        self._pending_render = self.canvas.after(200, self._render_full_resolution)
    
    def _cancel_pending_render(self):
        """Cancel pending full resolution render."""
        if self._pending_render:
            self.canvas.after_cancel(self._pending_render)
            self._pending_render = None
    
    def _render_full_resolution(self):
        """Render full resolution image and corners (called after interaction stops)."""
        self._use_preview = False
        self._pending_render = None
        self.update_display(force_full_res=True)
        
        # Redraw corners if they should be showing
        if hasattr(self, '_pending_corners_data') and self._pending_corners_data:
            logger.debug("Redrawing corners after motion stopped")
            if len(self._pending_corners_data) == 4:
                corners_data, depth_map, original_image_size, uid = self._pending_corners_data
                self.draw_corner_overlays(corners_data, depth_map, store_for_redraw=False, original_image_size=original_image_size, uid=uid)
            else:
                # Fallback for old format
                corners_data, depth_map, original_image_size = self._pending_corners_data
                self.draw_corner_overlays(corners_data, depth_map, store_for_redraw=False, original_image_size=original_image_size)
        else:
            logger.debug("Rendered full resolution image")


class CompartmentSlotStrip:
    """Strip showing 20 compartment slots with hover support and multi-source handling."""
    
    def __init__(self, parent, theme_colors, on_hover_callback=None, on_click_callback=None):
        self.theme_colors = theme_colors
        self.on_hover_callback = on_hover_callback
        self.on_click_callback = on_click_callback
        self.parent = parent
        
        # Compartment image aspect ratio (414x932) = 2.25:1 (height:width)
        self.compartment_aspect_ratio = 932 / 414  # ~2.25
        
        # Create main frame with dynamic height
        self.frame = tk.Frame(parent, bg=theme_colors["secondary_bg"])
        self.frame.pack_propagate(False)
        
        # Inner frame for slots (no canvas/scrollbar needed)
        self.inner_frame = tk.Frame(self.frame, bg=theme_colors["secondary_bg"])
        self.inner_frame.pack(fill=tk.BOTH, expand=True)
        
        # Compartment slots
        self.slots = []
        self.current_uid = None
        self.active_slots = set()
        
        # Thumbnail cache to avoid reloading
        self._thumbnail_cache = {}  # path -> PhotoImage
        
        # Calculate slot dimensions from window width
        self.slot_width = 60  # Will be recalculated
        self.slot_height = 135
        self._calculate_and_create_slots()
    
    def _calculate_and_create_slots(self):
        """Calculate slot dimensions and create 20 slots."""
        # Wait for window to be mapped
        self.parent.update_idletasks()
        
        # Get actual window width (use winfo_width after update_idletasks)
        window_width = self.parent.winfo_width()
        if window_width < 100:  # Not ready yet, use reasonable default
            window_width = 1400
        
        # Calculate slot width: fit 20 slots with tight padding
        padding_per_slot = 2  # 1px on each side - TIGHTER
        total_padding = 20 * padding_per_slot
        available_width = window_width - total_padding - 20  # Less margin
        self.slot_width = max(80, available_width // 20)  # Minimum 80px (BIGGER)
        
        # Calculate height from aspect ratio (2.25:1)
        self.slot_height = int(self.slot_width * self.compartment_aspect_ratio)
        
        # Update frame height to fit slots + padding
        self.frame.configure(height=self.slot_height + 20)
        
        logger.debug(f"Compartment slots: {self.slot_width}x{self.slot_height} (window: {window_width}px)")
        
        # Configure grid to center slots
        # Column 0: left spacer (weight=1, expands to fill left side)
        # Columns 1-20: slots (no weight, fixed size)
        # Column 21: right spacer (weight=1, expands to fill right side)
        
        self.inner_frame.columnconfigure(0, weight=1)  # Left spacer
        self.inner_frame.columnconfigure(21, weight=1)  # Right spacer
        
        # Create slots in columns 1-20 (centered between spacers)
        for i in range(20):
            # Canvas for drawing borders
            slot_canvas = tk.Canvas(
                self.inner_frame,
                width=self.slot_width,
                height=self.slot_height,
                bg=self.theme_colors["field_bg"],
                highlightthickness=0,
                bd=0
            )
            slot_canvas.grid(row=0, column=i + 1, padx=1, pady=2)  # Columns 1-20
            
            # Image label - fills entire canvas
            img_label = tk.Label(
                slot_canvas,
                bg=self.theme_colors["field_bg"],
                relief=tk.FLAT,
                borderwidth=0
            )
            img_label.place(x=0, y=0, width=self.slot_width, height=self.slot_height)
            
            # Depth label - overlay at top with colored background
            depth_label = tk.Label(
                slot_canvas,
                text="",
                bg=self.theme_colors["accent_blue"],
                fg="white",
                font=("Arial", 9, "bold"),
                padx=3,
                pady=1
            )
            depth_label.place(x=2, y=2)
            
            # Store references
            slot_info = {
                'canvas': slot_canvas,
                'depth_label': depth_label,
                'img_label': img_label,
                'image': None,
                'depth': None,
                'comp_data': None,
                'is_active': True,
                'border_id': None  # For red border rectangle
            }
            
            # Bind events
            for widget in [slot_canvas, img_label, depth_label]:
                widget.bind("<Enter>", lambda e, idx=i: self._on_hover(idx, True))
                widget.bind("<Leave>", lambda e, idx=i: self._on_hover(idx, False))
                widget.bind("<Button-1>", lambda e, idx=i: self._on_click(idx))
            
            self.slots.append(slot_info)
    
    def pack(self, **kwargs):
        """Pack the frame widget."""
        self.frame.pack(**kwargs)
    
    def _calculate_compartment_aspect_ratio(self, compartment_images: Dict) -> float:
        """
        Calculate average aspect ratio from actual compartment images.
        
        Args:
            compartment_images: Dict of depth -> comp_info
            
        Returns:
            float: height/width ratio, defaults to 1.5 if no images found
        """
        aspect_ratios = []
        
        # Sample up to 10 compartment images to determine aspect ratio
        sample_count = 0
        for depth, comp_info in compartment_images.items():
            if sample_count >= 10:
                break
                
            # Handle case where multiple compartments at same depth
            comp_data = comp_info[0] if isinstance(comp_info, list) else comp_info
            
            if comp_data.get('path') and Path(comp_data['path']).exists():
                try:
                    img = Image.open(comp_data['path'])
                    aspect_ratio = img.height / img.width
                    aspect_ratios.append(aspect_ratio)
                    sample_count += 1
                except Exception as e:
                    logger.debug(f"Could not read aspect ratio from {comp_data['path']}: {e}")
                    continue
        
        if aspect_ratios:
            avg_aspect = sum(aspect_ratios) / len(aspect_ratios)
            logger.info(f"Calculated compartment aspect ratio: {avg_aspect:.2f} from {len(aspect_ratios)} samples")
            return avg_aspect
        
        # Default to portrait 2:3 ratio if no images found
        logger.info("Using default aspect ratio 1.5 (no compartment images found)")
        return 1.5
            
    def update_slots(self, compartment_images: Dict, center_depth: int, 
                     depth_range: int = 20, current_uid: str = None):
        """
        Update compartment slots with multi-source awareness.
        
        Args:
            compartment_images: Dict of depth -> comp_info with 'uid', 'path', 'depth', 'name'
            center_depth: Center depth to display
            depth_range: Range of depths to show
            current_uid: UID of current original image
        """
        self.current_uid = current_uid
        self.active_slots.clear()
        
        # Calculate actual aspect ratio from compartment images (first time only)
        if not hasattr(self, '_aspect_ratio_calculated') and compartment_images:
            actual_ratio = self._calculate_compartment_aspect_ratio(compartment_images)
            if abs(actual_ratio - self.compartment_aspect_ratio) > 0.1:  # Significant difference
                # Reconfigure slot dimensions
                self.compartment_aspect_ratio = actual_ratio
                screen_width = self.frame.winfo_screenwidth()
                available_width = int(screen_width * 0.9)
                slot_width = min(80, (available_width - 60) // 20)
                new_slot_height = int(slot_width * actual_ratio)
                
                # Update all slot canvases with new dimensions
                for slot in self.slots:
                    slot['canvas'].config(width=slot_width, height=new_slot_height)
                
                logger.info(f"Updated slot dimensions to {slot_width}x{new_slot_height} (ratio: {actual_ratio:.2f})")
            
            self._aspect_ratio_calculated = True
        
        # Calculate depth range - for a 60-80m tray, we want compartments 61-80
        # So if center_depth is 70, we want 61-80 (from start+1 to end)
        start_depth = center_depth - depth_range // 2
        end_depth = start_depth + depth_range
        
        logger.debug(f"Compartment slot range: {start_depth+1}-{end_depth}m (center: {center_depth}m, range: {depth_range} slots)")
        logger.debug(f"  Will check depths: {list(range(start_depth + 1, end_depth + 1))}")
        
        # Clear slot data AND displayed images (must clear img_label to prevent showing old images)
        for slot in self.slots:
            slot['depth_label'].config(text="")
            slot['img_label'].config(image="")  # CRITICAL: Clear displayed image
            slot['image'] = None
            slot['depth'] = None
            slot['comp_data'] = None
            slot['is_active'] = False
            # Clear any borders
            if slot['border_id']:
                slot['canvas'].delete(slot['border_id'])
                slot['border_id'] = None
        
        # Populate slots (use FROM depth: start_depth+1 to end_depth)
        populated_count = 0
        active_count = 0
        
        # DEBUG: Log what depths are actually available
        available_depths = sorted(compartment_images.keys())
        expected_depths = list(range(start_depth + 1, end_depth + 1))
        logger.info(f"📊 Depth Analysis:")
        logger.info(f"   Expected depths: {expected_depths}")
        logger.info(f"   Available depths in range: {[d for d in available_depths if start_depth < d <= end_depth]}")
        logger.info(f"   Missing depths: {[d for d in expected_depths if d not in compartment_images]}")
        
        for idx, depth in enumerate(range(start_depth + 1, end_depth + 1)):
            logger.debug(f"Slot {idx}: checking depth {depth}m (slot exists: {idx < len(self.slots)})")
            if idx >= len(self.slots):
                break
                
            slot = self.slots[idx]
            slot['depth'] = depth
            slot['depth_label'].config(text=f"{depth}m")
            
            # Check if we have compartment at this depth
            if depth in compartment_images:
                comp_info = compartment_images[depth]
                
                # Always treat as active - no greying
                is_from_current = True
                
                if comp_info.get('path') and Path(comp_info['path']).exists():
                    try:
                        # Check thumbnail cache first
                        cache_key = f"{comp_info['path']}_{self.slot_width}x{self.slot_height}"
                        
                        if cache_key in self._thumbnail_cache:
                            photo = self._thumbnail_cache[cache_key]
                        else:
                            # Load and resize image to fill slot
                            img = Image.open(comp_info['path'])
                            
                            # Resize to fill the slot while maintaining aspect ratio
                            img_resized = img.resize((self.slot_width, self.slot_height), Image.LANCZOS)
                            
                            # No greying - always show full color
                            photo = ImageTk.PhotoImage(img_resized)
                            self._thumbnail_cache[cache_key] = photo
                        
                        self.active_slots.add(idx)
                        slot['img_label'].config(image=photo)
                        slot['image'] = photo
                        slot['comp_data'] = comp_info
                        slot['is_active'] = is_from_current
                        
                        # Ensure image label and depth label are on top
                        slot['img_label'].lift()
                        slot['depth_label'].lift()
                        
                        populated_count += 1
                        if is_from_current:
                            active_count += 1
                        
                        # (Source indicator removed - not needed in new design)
                            
                    except Exception as e:
                        logger.error(f"Error loading compartment thumbnail for depth {depth}m: {e}")
            else:
                # Empty compartment slot - nothing to configure
                pass
        
        logger.info(f"Compartment slots: {populated_count}/20 populated ({active_count} from current image, {populated_count - active_count} from other sources)")
                
    def _on_hover(self, index: int, is_hovering: bool):
        """Handle hover events with red border."""
        if index < len(self.slots):
            slot = self.slots[index]
            if slot['depth'] is not None and self.on_hover_callback:
                self.on_hover_callback(slot['depth'], is_hovering, slot.get('is_active', True))
                
            # Visual feedback with red border
            canvas = slot['canvas']
            if is_hovering and slot['comp_data']:
                # Draw thick red border
                if slot['border_id']:
                    canvas.delete(slot['border_id'])
                slot['border_id'] = canvas.create_rectangle(
                    0, 0, self.slot_width, self.slot_height,
                    outline='red',
                    width=4
                )
            else:
                # Remove border
                if slot['border_id']:
                    canvas.delete(slot['border_id'])
                    slot['border_id'] = None
                    
    def _on_click(self, index: int):
        """Handle click events."""
        if index < len(self.slots):
            slot = self.slots[index]
            if slot['comp_data'] and self.on_click_callback:
                self.on_click_callback(slot['depth'], slot['comp_data'])

class DepthGroupedFilmstrip:
    """Filmstrip grouped by depth interval (handles multiple images per interval)."""
    
    def __init__(self, parent, theme_colors, on_select_callback):
        self.theme_colors = theme_colors
        self.on_select_callback = on_select_callback
        
        self.container = ttk.Frame(parent, style="Content.TFrame")
        
        # Create canvas for horizontal scrolling
        self.canvas = tk.Canvas(
            self.container,
            bg=theme_colors["secondary_bg"],
            height=120,  # Compact height for thumbnails + labels
            highlightthickness=0
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add horizontal scrollbar
        scrollbar = ttk.Scrollbar(
            self.container,
            orient=tk.HORIZONTAL,
            command=self.canvas.xview
        )
        scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.configure(xscrollcommand=scrollbar.set)
        
        # Thumbnails frame inside canvas
        self.thumbs_frame = tk.Frame(
            self.canvas,
            bg=theme_colors["secondary_bg"]
        )
        self.canvas_window = self.canvas.create_window((0, 0), window=self.thumbs_frame, anchor='nw')
        
        # Bind frame resize to update scroll region
        self.thumbs_frame.bind('<Configure>', self._on_frame_configure)
        
        self.thumbnails = []
        self.depth_groups = []  # List of depth intervals
        self.current_group_index = 0

        # Bind mouse wheel for horizontal scrolling
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", self._on_mousewheel)  # Linux scroll up
        self.canvas.bind("<Button-5>", self._on_mousewheel)  # Linux scroll down
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel for horizontal scrolling."""
        # Windows and MacOS
        if event.num == 4 or event.delta > 0:
            self.canvas.xview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:
            self.canvas.xview_scroll(1, "units")

    def _on_frame_configure(self, event=None):
        """Update scroll region when frame size changes."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _scroll_to_current(self, current_index: int):
        """Auto-scroll to keep current thumbnail visible and centered."""
        if not hasattr(self, '_filmstrip_widgets') or not self._filmstrip_widgets:
            return
        
        # Find which widget group contains the current index
        target_widget = None
        for widget_info in self._filmstrip_widgets:
            group_indices = [idx for idx, _ in widget_info['images_in_group']]
            if current_index in group_indices:
                target_widget = widget_info['btn']
                break
        
        if not target_widget:
            return
        
        # Wait for widget to be visible
        self.canvas.update_idletasks()
        
        try:
            # Get widget position relative to thumbs_frame
            widget_x = target_widget.winfo_x()
            widget_width = target_widget.winfo_width()
            widget_center = widget_x + widget_width / 2
            
            # Get canvas dimensions
            canvas_width = self.canvas.winfo_width()
            
            # Calculate scroll position to center the widget
            # Canvas shows from scroll_pos to scroll_pos + canvas_width
            # We want widget_center to be at canvas_width / 2
            target_scroll_pos = widget_center - canvas_width / 2
            
            # Get total scrollable width
            total_width = self.thumbs_frame.winfo_width()
            
            # Clamp to valid range [0, total_width - canvas_width]
            max_scroll = max(0, total_width - canvas_width)
            target_scroll_pos = max(0, min(target_scroll_pos, max_scroll))
            
            # Convert to canvas scroll units (0.0 to 1.0)
            if total_width > 0:
                scroll_fraction = target_scroll_pos / total_width
                self.canvas.xview_moveto(scroll_fraction)
                
        except Exception as e:
            logger.debug(f"Error auto-scrolling filmstrip: {e}")
        
    def pack(self, **kwargs):
        """Pack the container."""
        self.container.pack(**kwargs)
        
    def update_filmstrip(self, original_images: List[Dict], current_index: int = 0):
        """
        Update filmstrip, grouping images by depth interval.
        
        Args:
            original_images: List of original image info dicts
            current_index: Index of currently displayed image
        """
        # Check if we need to rebuild (data changed) or just update highlight
        rebuild_needed = (not hasattr(self, '_last_original_images') or 
                         len(original_images) != len(self._last_original_images) or
                         any(img1['path'] != img2['path'] for img1, img2 in zip(original_images, self._last_original_images)))
        
        if rebuild_needed:
            # Full rebuild needed
            self._build_filmstrip_widgets(original_images, current_index)
            self._last_original_images = original_images.copy()
        
        # Update highlight for current image
        self._update_filmstrip_highlight(current_index)
    
    def _build_filmstrip_widgets(self, original_images: List[Dict], current_index: int = 0):
        """Build filmstrip widgets from scratch (called once per hole)."""
        # Clear existing
        for widget in self.thumbs_frame.winfo_children():
            widget.destroy()
        
        self.thumbnails = []
        self.depth_groups = []
        self._filmstrip_widgets = []  # Track widgets for highlight updates
        
        # Group images by depth interval
        depth_map = defaultdict(list)
        for idx, img_info in enumerate(original_images):
            depth_key = (img_info['depth_from'], img_info['depth_to'])
            depth_map[depth_key].append((idx, img_info))
        
        # Sort by depth
        sorted_depths = sorted(depth_map.keys())
        
        # Create thumbnails
        for depth_key in sorted_depths:
            images_in_group = depth_map[depth_key]
            
            # Find which image in this group is current
            group_has_current = any(idx == current_index for idx, _ in images_in_group)
            
            # Use first image in group for thumbnail
            first_idx, first_img = images_in_group[0]
            
            thumb_frame = tk.Frame(
                self.thumbs_frame,
                bg=self.theme_colors["secondary_bg"]
            )
            thumb_frame.pack(side=tk.LEFT, padx=5)
            
            try:
                # Check for pre-loaded thumbnail first
                depth_key = (first_img['depth_from'], first_img['depth_to'])
                if hasattr(self, 'parent') and hasattr(self.parent, 'preloaded_filmstrip') and depth_key in self.parent.preloaded_filmstrip:
                    photo = self.parent.preloaded_filmstrip[depth_key]
                    logger.debug(f"Using pre-loaded filmstrip thumbnail for depth {depth_key}")
                else:
                    # Generate thumbnail on the fly (fallback)
                    img = Image.open(first_img['path'])
                    
                    # Create thumbnail - smaller and clearer
                    target_height = 60
                    scale = target_height / img.height
                    new_width = int(img.width * scale)
                    img_resized = img.resize((new_width, target_height), Image.LANCZOS)
                    img_resized = img_resized.convert('RGB')
                    
                    logger.debug(f"Filmstrip thumbnail: {img.size} -> {img_resized.size}")
                    
                    photo = ImageTk.PhotoImage(img_resized)
                
                # Highlight if current - MUCH more prominent
                if group_has_current:
                    border_color = "#00FF00"  # Bright green
                    border_width = 8  # Thick border
                else:
                    border_color = "#555555"  # Dark gray
                    border_width = 1  # Thin border
                
                # Container for navigation buttons if multiple images
                if len(images_in_group) > 1:
                    nav_frame = tk.Frame(thumb_frame, bg=self.theme_colors["secondary_bg"])
                    nav_frame.pack()
                    
                    # Previous button
                    prev_btn = tk.Label(
                        nav_frame,
                        text="◀",
                        bg=self.theme_colors["accent_blue"],
                        fg="white",
                        font=("Arial", 10, "bold"),
                        cursor="hand2",
                        padx=3
                    )
                    prev_btn.pack(side=tk.LEFT)
                    
                    # Count label
                    count_label = tk.Label(
                        nav_frame,
                        text=f"{len(images_in_group)}",
                        bg=self.theme_colors["accent_blue"],
                        fg="white",
                        font=("Arial", 8, "bold"),
                        padx=4
                    )
                    count_label.pack(side=tk.LEFT)
                    
                    # Next button
                    next_btn = tk.Label(
                        nav_frame,
                        text="▶",
                        bg=self.theme_colors["accent_blue"],
                        fg="white",
                        font=("Arial", 10, "bold"),
                        cursor="hand2",
                        padx=3
                    )
                    next_btn.pack(side=tk.LEFT)
                    
                    # Bind navigation
                    def go_prev(e, group=images_in_group):
                        # Find current in group, go to previous
                        current = getattr(self.on_select_callback.__self__, 'current_index', 0) if hasattr(self.on_select_callback, '__self__') else 0
                        indices = [idx for idx, _ in group]
                        if current in indices:
                            pos = indices.index(current)
                            self.on_select_callback(indices[(pos - 1) % len(indices)])
                        else:
                            self.on_select_callback(indices[0])
                    
                    def go_next(e, group=images_in_group):
                        # Find current in group, go to next
                        current = getattr(self.on_select_callback.__self__, 'current_index', 0) if hasattr(self.on_select_callback, '__self__') else 0
                        indices = [idx for idx, _ in group]
                        if current in indices:
                            pos = indices.index(current)
                            self.on_select_callback(indices[(pos + 1) % len(indices)])
                        else:
                            self.on_select_callback(indices[0])
                    
                    prev_btn.bind("<Button-1>", go_prev)
                    next_btn.bind("<Button-1>", go_next)
                
                btn = tk.Label(
                    thumb_frame,
                    image=photo,
                    bg=border_color,
                    borderwidth=border_width,
                    relief="solid",
                    cursor="hand2"
                )
                btn.image = photo
                btn.pack()
                
                # Extract HoleID and depth range from filename
                # E.g., "KM0227_0-20_Original.JPG" -> "KM0227 - 0-20m" and "Original"
                filename = first_img['name']
                hole_id = first_img.get('hole_id', '')
                depth_from = depth_key[0]
                depth_to = depth_key[1]
                
                # Extract suffix (anything after depth range)
                # Pattern: HoleID_DepthFrom-DepthTo_SUFFIX.ext
                import re
                suffix_match = re.search(rf'{hole_id}_{depth_from}-{depth_to}_(.+?)\.', filename)
                suffix = suffix_match.group(1) if suffix_match else ''
                suffix = suffix.replace('_', ' ').strip()  # Clean up underscores
                
                # Line 1: HoleID - Depth range
                line1_text = f"{hole_id} - {depth_from}-{depth_to}m"
                line1_label = tk.Label(
                    thumb_frame,
                    text=line1_text,
                    bg=self.theme_colors["accent_blue"] if group_has_current else self.theme_colors["secondary_bg"],
                    fg="white" if group_has_current else self.theme_colors["text"],
                    font=("Arial", 9, "bold")
                )
                line1_label.pack()
                
                # Line 2: Suffix (if exists)
                if suffix:
                    line2_label = tk.Label(
                        thumb_frame,
                        text=suffix,
                        bg=self.theme_colors["secondary_bg"],
                        fg=self.theme_colors["text"],
                        font=("Arial", 7)
                    )
                    line2_label.pack()
                
                # Bind click - if multiple images, cycle through them
                btn.bind("<Button-1>", lambda e, idx=first_idx: self.on_select_callback(idx))
                
                self.thumbnails.append(btn)
                self.depth_groups.append(images_in_group)
                
                # Store widget references for highlight updates
                self._filmstrip_widgets.append({
                    'btn': btn,
                    'line1_label': line1_label,
                    'images_in_group': images_in_group
                })
                
            except Exception as e:
                logger.error(f"Error creating filmstrip thumbnail: {e}")

    def _update_filmstrip_highlight(self, current_index: int):
        """Update only the highlight borders without rebuilding widgets."""
        if not hasattr(self, '_filmstrip_widgets'):
            return
        
        for widget_info in self._filmstrip_widgets:
            group_indices = [idx for idx, _ in widget_info['images_in_group']]
            group_has_current = current_index in group_indices
            
            # Update border
            btn = widget_info['btn']
            line1_label = widget_info['line1_label']
            
            if group_has_current:
                btn.config(bg="#00FF00", borderwidth=8)
                line1_label.config(bg=self.theme_colors["accent_blue"], fg="white")
            else:
                btn.config(bg="#555555", borderwidth=1)
                line1_label.config(bg=self.theme_colors["secondary_bg"], fg=self.theme_colors["text"])
        
        # Auto-scroll to keep current thumbnail visible/centered
        self._scroll_to_current(current_index)

class OptimizedOriginalImageViewer:
    """Optimized Original Image Viewer with full feature set."""
    
    def __init__(self, parent, file_manager, gui_manager=None, json_register_manager=None):
        self.parent = parent
        self.file_manager = file_manager
        self.gui_manager = gui_manager
        self.json_register_manager = json_register_manager
        
        # Data cache - ENHANCED
        self.hole_id = ""
        self.original_images = []
        self.compartment_images = {}  # depth -> comp_info (or list of comp_info)
        self.compartment_by_uid = {}  # uid -> {depth -> comp_info}
        self.current_index = 0
        self.uid_map = {}  # uid -> original_info
        # Removed: moisture_state - not needed for viewer
        self.depth_interval_images = {}  # (depth_from, depth_to) -> [image_indices]
        
        # Hover state for corner highlighting
        self.highlighted_depth = None
        
        # Compartment hover popup
        self._hover_popup = None
        self._hover_popup_label = None
        self._current_hover_image = None
        
        # Add image cache for performance
        self.image_cache = ImageCache(max_cache_size=30)
        
        # PRE-LOADING CACHES - for instant switching
        self.preloaded_images = {}  # index -> {'full': PIL, 'working': PIL, 'preview': PIL, 'photo': ImageTk}
        self.preloaded_corners = {}  # uid -> List[Dict] of corner data
        self.preloaded_filmstrip = {}  # (depth_from, depth_to) -> ImageTk.PhotoImage
        self.preloaded_overlay_images = {}  # uid -> PIL.Image (final drawn overlay)
        
        # Create window
        self.window = tk.Toplevel(parent)
        self.window.title("Original Image Viewer")
        
        # Apply theming
        if self.gui_manager:
            self.gui_manager.apply_theme(self.window)
            self.theme_colors = self.gui_manager.theme_colors
            self.fonts = self.gui_manager.fonts
            self.window.configure(bg=self.theme_colors["background"])
        else:
            self.theme_colors = {
                "background": "#2b2b2b",
                "secondary_bg": "#3c3c3c",
                "text": "#ffffff",
                "field_bg": "#1e1e1e",
                "accent_green": "#5aa06c",
                "accent_blue": "#4a90c0",
                "accent_red": "#ff6b6b",
            }
            self.fonts = {
                "normal": ("Arial", 10),
                "heading": ("Arial", 12, "bold"),
            }
            self.window.configure(bg=self.theme_colors["background"])
        
        # Start maximized
        try:
            self.window.state('zoomed')
        except:
            try:
                self.window.attributes('-zoomed', True)
            except:
                pass
        
        # Build UI
        self._build_ui()
        
        # Bind keyboard shortcuts
        self.window.bind("<Left>", lambda e: self.navigate_previous())
        self.window.bind("<Right>", lambda e: self.navigate_next())
        self.window.bind("<Delete>", lambda e: self.reprocess_current())
        self.window.bind("<Return>", lambda e: self.load_hole() if self.hole_id_var.get() else None)
        self.window.bind("<Up>", lambda e: self.navigate_same_interval_previous())
        self.window.bind("<Down>", lambda e: self.navigate_same_interval_next())
        
        # Bind cleanup on close
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)
        
        logger.info("Optimized Original Image Viewer initialized")
    
    def _on_close(self):
        """Handle window close with cleanup."""
        self._cleanup_resources()
        self.window.destroy()
        
    def navigate_same_interval_previous(self):
        """Navigate to previous image at same depth interval."""
        if not self.original_images or self.current_index >= len(self.original_images):
            return
            
        current_img = self.original_images[self.current_index]
        depth_from = current_img['depth_from']
        depth_to = current_img['depth_to']
        
        # Find all images at same depth interval
        same_interval_indices = []
        for idx, img_info in enumerate(self.original_images):
            if img_info['depth_from'] == depth_from and img_info['depth_to'] == depth_to:
                same_interval_indices.append(idx)
        
        if len(same_interval_indices) <= 1:
            # No other images at this interval
            return
            
        # Find current position in the interval group
        try:
            current_pos = same_interval_indices.index(self.current_index)
            # Move to previous in group (wrap around)
            new_pos = (current_pos - 1) % len(same_interval_indices)
            new_index = same_interval_indices[new_pos]
            
            self.display_image(new_index)
            self._update_interval_status(new_pos + 1, len(same_interval_indices))
            
        except ValueError:
            logger.error(f"Current index not found in interval group")
            
    def navigate_same_interval_next(self):
        """Navigate to next image at same depth interval."""
        if not self.original_images or self.current_index >= len(self.original_images):
            return
            
        current_img = self.original_images[self.current_index]
        depth_from = current_img['depth_from']
        depth_to = current_img['depth_to']
        
        # Find all images at same depth interval
        same_interval_indices = []
        for idx, img_info in enumerate(self.original_images):
            if img_info['depth_from'] == depth_from and img_info['depth_to'] == depth_to:
                same_interval_indices.append(idx)
        
        if len(same_interval_indices) <= 1:
            # No other images at this interval
            return
            
        # Find current position in the interval group
        try:
            current_pos = same_interval_indices.index(self.current_index)
            # Move to next in group (wrap around)
            new_pos = (current_pos + 1) % len(same_interval_indices)
            new_index = same_interval_indices[new_pos]
            
            self.display_image(new_index)
            self._update_interval_status(new_pos + 1, len(same_interval_indices))
            
        except ValueError:
            logger.error(f"Current index not found in interval group")
            
    def _update_interval_status(self, current_num: int, total_num: int):
        """Update status label to show which image in the interval we're viewing."""
        if total_num > 1:
            current_img = self.original_images[self.current_index]
            depth_from = current_img['depth_from']
            depth_to = current_img['depth_to']
            
            # Add image name or identifier if available
            img_name = current_img.get('name', '').split('_')[0] if current_img.get('name') else ''
            
            self.status_label.config(
                text=f"{self.hole_id} | {depth_from}-{depth_to}m | Image {current_num}/{total_num}"
            )

    def _build_ui(self):
        """Build the complete UI."""
        # Top control bar
        control_frame = ttk.Frame(self.window, style="Content.TFrame")
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(
            control_frame,
            text="HoleID:",
            style="Content.TLabel"
        ).pack(side=tk.LEFT, padx=5)
        
        # HoleID searchable dropdown
        self.hole_id_var = tk.StringVar()
        
        # Get hole list from approved originals
        hole_list = self._get_available_holes()
        
        if self.gui_manager:
            # Use new searchable option menu with parent key management
            hole_widget = ThemedSearchableOptionMenu(
                control_frame,
                gui_manager=self.gui_manager,
                items=hole_list,
                variable=self.hole_id_var,
                width=20,
                placeholder="Type hole ID...",
                on_change=self._on_hole_selected,
                manage_parent_keys=['<Up>', '<Down>', '<Return>', '<Left>', '<Right>']
            )
        else:
            # Fallback to regular combobox if no gui_manager
            hole_widget = ttk.Combobox(
                control_frame,
                textvariable=self.hole_id_var,
                values=hole_list,
                width=20
            )
        hole_widget.pack(side=tk.LEFT, padx=5)
        
        # Bind focus loss and Enter key for immediate loading
        if hasattr(hole_widget, 'entry'):
            # For ThemedSearchableOptionMenu, bind to the entry widget
            hole_widget.entry.bind("<FocusOut>", lambda e: self._on_hole_filter())
            hole_widget.entry.bind("<Return>", lambda e: self._on_hole_filter())
        else:
            # Fallback for regular combobox
            hole_widget.bind("<FocusOut>", lambda e: self._on_hole_filter())
            hole_widget.bind("<Return>", lambda e: self._on_hole_filter())
        
        # Store reference for updates
        self.hole_widget = hole_widget
        
        # Load button
        ModernButton(
            control_frame,
            text="Load Hole",
            color=self.theme_colors["accent_blue"],
            command=self.load_hole,
            theme_colors=self.theme_colors,
            fonts=self.fonts
        ).pack(side=tk.LEFT, padx=5)
        
        # Toggle for corners overlay
        self.show_corners_var = tk.BooleanVar(value=False)
        
        if self.gui_manager:
            self.gui_manager.create_custom_checkbox(
                control_frame,
                text="Show Corners",
                variable=self.show_corners_var,
                command=self.toggle_corners
            ).pack(side=tk.LEFT, padx=20)
        else:
            tk.Checkbutton(
                control_frame,
                text="Show Corners",
                variable=self.show_corners_var,
                command=self.toggle_corners,
                bg=self.theme_colors["background"],
                fg=self.theme_colors["text"]
            ).pack(side=tk.LEFT, padx=20)
        
        # Reprocess button
        ModernButton(
            control_frame,
            text="⟳ Reprocess",
            color=self.theme_colors["accent_red"],
            command=self.reprocess_current,
            theme_colors=self.theme_colors,
            fonts=self.fonts
        ).pack(side=tk.LEFT, padx=5)
        
        self.status_label = ttk.Label(
            control_frame,
            text="Enter a HoleID to begin",
            style="Content.TLabel"
        )
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # Main content (3 panels)
        main_frame = ttk.Frame(self.window, style="Content.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # TOP: Compartment slots (20 slots)
        self.compartment_strip = CompartmentSlotStrip(
            main_frame,
            self.theme_colors,
            on_hover_callback=self.on_compartment_hover
        )
        self.compartment_strip.pack(fill=tk.X, pady=(0, 10))
        
        # MIDDLE: Original image with pan/zoom
        canvas_frame = tk.Frame(main_frame, bg=self.theme_colors["secondary_bg"])
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.canvas = tk.Canvas(
            canvas_frame,
            bg=self.theme_colors["secondary_bg"],
            highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.image_viewer = PanZoomImageViewer(self.canvas)
        
        # BOTTOM: Filmstrip + Navigation
        bottom_frame = ttk.Frame(main_frame, style="Content.TFrame")
        bottom_frame.pack(fill=tk.X)
        
        # Navigation row
        nav_row = ttk.Frame(bottom_frame, style="Content.TFrame")
        nav_row.pack(fill=tk.X, pady=5)
        
        # Left arrow
        ModernButton(
            nav_row,
            text="◀",
            color=self.theme_colors["accent_blue"],
            command=self.navigate_previous,
            theme_colors=self.theme_colors,
            fonts={"normal": ("Arial", 14)},
            width=3,
            height=1
        ).pack(side=tk.LEFT, padx=5)
        
        # Close button (centered)
        ModernButton(
            nav_row,
            text="Close",
            color=self.theme_colors["field_bg"],
            command=self.window.destroy,
            theme_colors=self.theme_colors,
            fonts=self.fonts,
            width=8,
            height=1
        ).pack(side=tk.LEFT, padx=20, expand=True)
        
        # Right arrow
        ModernButton(
            nav_row,
            text="▶",
            color=self.theme_colors["accent_blue"],
            command=self.navigate_next,
            theme_colors=self.theme_colors,
            fonts={"normal": ("Arial", 14)},
            width=3,
            height=1
        ).pack(side=tk.RIGHT, padx=5)
        
        # Filmstrip
        self.filmstrip = DepthGroupedFilmstrip(
            bottom_frame,
            self.theme_colors,
            self.on_filmstrip_select
        )
        self.filmstrip.pack(fill=tk.X)
        
    def _on_hole_filter(self):
        """Handle HoleID filter change on focus loss or Enter."""
        new_hole = self.hole_id_var.get().strip().upper()
        if new_hole and new_hole != self.hole_id:
            logger.debug(f"HoleID filter changed: {self.hole_id} -> {new_hole}")
            self.load_hole()

    def _get_available_holes(self) -> List[str]:
        """Get list of available hole IDs from approved originals folder."""
        holes = set()
        
        try:
            approved_path = self.file_manager.get_shared_path("approved_originals", create_if_missing=False)
            if not approved_path or not approved_path.exists():
                return []
            
            # Scan for project code folders
            for project_folder in approved_path.iterdir():
                if project_folder.is_dir():
                    # Scan for hole folders
                    for hole_folder in project_folder.iterdir():
                        if hole_folder.is_dir():
                            holes.add(hole_folder.name)
            
            return sorted(list(holes))
            
        except Exception as e:
            logger.error(f"Error getting available holes: {e}")
            return []
    
    def load_hole(self):
        """Load all data for the specified hole with async pre-loading."""
        hole_id = self.hole_id_var.get().strip().upper()
        
        if not hole_id:
            DialogHelper.show_message(
                self.window,
                "Input Required",
                "Please enter a HoleID",
                message_type="warning"
            )
            return
        
        logger.info(f"Loading hole: {hole_id}")
        
        # Only cleanup if we're loading a DIFFERENT hole
        if self.hole_id and self.hole_id != hole_id:
            self._cleanup_resources()
        
        # Clear old data
        self.hole_id = hole_id
        self.original_images = []
        self.compartment_images = {}
        self.uid_map = {}
        self.current_index = 0
        
        # Clear pre-loading caches
        self.preloaded_images.clear()
        self.preloaded_corners.clear()
        self.preloaded_filmstrip.clear()
        
        # Import progress dialog
        from gui.progress_dialog import ProgressDialog
        
        # Create progress dialog
        progress = ProgressDialog(
            self.window,
            title="Loading Hole Data",
            message=f"Scanning images for {hole_id}...",
            modal=True
        )
        
        # Run loading in background thread
        def load_task():
            try:
                # Step 1: Scan for original images (fast)
                progress.update_progress(f"Scanning images for {hole_id}...", 5)
                self._load_originals(hole_id)
                
                if not self.original_images:
                    progress.close()
                    self.window.after(0, lambda: DialogHelper.show_message(
                        self.window,
                        "No Data",
                        f"No images found for {hole_id}",
                        message_type="info"
                    ))
                    self.window.after(0, lambda: self.status_label.config(text="No images found"))
                    return
                
                # Step 2: Load compartment metadata (fast)
                progress.update_progress(f"Loading compartment metadata...", 10)
                self._load_compartments(hole_id)
                
                # Step 3: Pre-fetch ALL corner data in one bulk query (fast)
                progress.update_progress(f"Fetching corner data...", 15)
                self._preload_all_corners()
                
                # Step 4: Pre-load ALL original images (slow - main bottleneck)
                total_images = len(self.original_images)
                for idx, img_info in enumerate(self.original_images):
                    percent = 15 + (idx / total_images) * 60  # 15% -> 75%
                    progress.update_progress(f"Loading {img_info['name']}...", percent)
                    self._preload_single_image(idx, img_info)
                
                # Step 5: Pre-generate filmstrip thumbnails (medium speed)
                progress.update_progress(f"Generating filmstrip thumbnails...", 80)
                self._preload_filmstrip_thumbnails()
                
                # Step 6: Display first image
                progress.update_progress(f"Preparing display...", 95)
                
                # Schedule display on main thread
                self.window.after(0, lambda: self.display_image(0))
                self.window.after(0, lambda: self.status_label.config(
                    text=f"{len(self.original_images)} originals, {len(self.compartment_images)} compartments"
                ))
                
                progress.update_progress(f"Complete!", 100)
                
            except Exception as e:
                logger.error(f"Error loading hole: {e}")
                import traceback
                traceback.print_exc()
                progress.close()
                self.window.after(0, lambda: DialogHelper.show_message(
                    self.window,
                    "Load Error",
                    f"Failed to load:\n{str(e)}",
                    message_type="error"
                ))
        
        # Start background thread
        import threading
        thread = threading.Thread(target=load_task, daemon=True)
        thread.start()
            
    def _load_originals(self, hole_id: str):
        """Load original images for hole (single scan with UID caching)."""
        approved_path = self.file_manager.get_shared_path("approved_originals", create_if_missing=False)
        if not approved_path:
            return
        
        project_code = hole_id[:2].upper() if len(hole_id) >= 2 else ""
        
        search_paths = [
            approved_path / project_code / hole_id,
            approved_path / hole_id,
        ]
        
        for path in search_paths:
            if not path.exists():
                continue
            
            logger.info(f"Scanning: {path}")
            
            for img_file in path.iterdir():
                if img_file.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}:
                    continue
                
                if not img_file.stem.upper().startswith(hole_id.upper()):
                    continue
                
                # Extract UID once
                try:
                    uid = self.file_manager.extract_uid_from_any_image(str(img_file))
                    if not uid:
                        logger.debug(f"  ⚠️ No UID found in: {img_file.name}")
                except Exception as e:
                    logger.warning(f"  ❌ UID extraction failed for {img_file.name}: {e}")
                    uid = None
                logger.info(f"📷 Original Image: {img_file.name}")
                logger.info(f"   UID extracted: {uid if uid else 'NONE/FAILED'}")
                
                # Extract depth from filename
                match = re.search(r'_(\d+)-(\d+)', img_file.stem)
                depth_from = int(match.group(1)) if match else 0
                depth_to = int(match.group(2)) if match else 0
                
                img_info = {
                    'path': img_file,
                    'uid': uid,
                    'depth_from': depth_from,
                    'depth_to': depth_to,
                    'name': img_file.name
                }
                
                self.original_images.append(img_info)
                
                if uid:
                    self.uid_map[uid] = img_info
            
            break  # Found the folder
        
        # Sort by depth
        self.original_images.sort(key=lambda x: x['depth_from'])
        logger.info(f"Loaded {len(self.original_images)} originals")
    
    def _preload_all_corners(self):
        """Pre-fetch ALL corner data for the hole in one bulk query."""
        if not self.json_register_manager or not self.hole_id:
            return
        
        try:
            logger.info(f"Pre-fetching all corner data for hole: {self.hole_id}")
            corners_df = self.json_register_manager.get_compartment_corners_data(hole_id=self.hole_id)
            
            if corners_df.empty or 'Source_Image_UID' not in corners_df.columns:
                logger.warning(f"No corner data available for hole {self.hole_id}")
                return
            
            # Organize by UID for fast lookup
            for uid in corners_df['Source_Image_UID'].unique():
                matching = corners_df[corners_df['Source_Image_UID'] == uid]
                corners_data = []
                
                # Find the image info for this UID to get depth_from
                img_info = self.uid_map.get(uid)
                if not img_info:
                    continue
                
                depth_from = img_info.get('depth_from', 0)
                
                for _, row in matching.iterrows():
                    corners = [
                        [int(row['Top_Left_X']), int(row['Top_Left_Y'])],
                        [int(row['Top_Right_X']), int(row['Top_Right_Y'])],
                        [int(row['Bottom_Right_X']), int(row['Bottom_Right_Y'])],
                        [int(row['Bottom_Left_X']), int(row['Bottom_Left_Y'])]
                    ]
                    
                    compartment_num = row.get('Compartment_Number', 0)
                    actual_depth = depth_from + compartment_num
                    
                    corners_data.append({
                        'corners': corners,
                        'depth': actual_depth
                    })
                
                self.preloaded_corners[uid] = corners_data
            
            logger.info(f"Pre-loaded corner data for {len(self.preloaded_corners)} unique UIDs")
            
        except Exception as e:
            logger.error(f"Error pre-loading corners: {e}")
    
    def _preload_single_image(self, index: int, img_info: Dict):
        """Pre-load a single original image with all its variants."""
        try:
            img_path = img_info['path']
            
            # Load full resolution image
            full_img = Image.open(img_path)
            full_img.load()  # Force load into memory
            
            # Create downscaled working copy (same logic as PanZoomImageViewer.load_image)
            max_dimension = 5000
            if full_img.width > max_dimension or full_img.height > max_dimension:
                scale = min(max_dimension / full_img.width, max_dimension / full_img.height)
                working_img = full_img.resize(
                    (int(full_img.width * scale), int(full_img.height * scale)), 
                    Image.LANCZOS
                )
                downscale_factor = scale
            else:
                working_img = full_img
                downscale_factor = 1.0
            
            # Create preview image (reduced color palette for fast interaction)
            preview_img = working_img.convert('RGB')
            preview_img = preview_img.quantize(colors=64).convert('RGB')
            
            # Store in cache
            self.preloaded_images[index] = {
                'full': full_img,
                'working': working_img,
                'preview': preview_img,
                'downscale_factor': downscale_factor
            }
            
            logger.debug(f"Pre-loaded image {index}: {img_info['name']} (full={full_img.size}, working={working_img.size})")
            
        except Exception as e:
            logger.error(f"Error pre-loading image {index} ({img_info.get('name')}): {e}")
    
    def _preload_filmstrip_thumbnails(self):
        """Pre-generate all filmstrip thumbnails."""
        try:
            # Group images by depth interval (same logic as FilmstripNavigator)
            depth_map = defaultdict(list)
            for idx, img_info in enumerate(self.original_images):
                depth_key = (img_info['depth_from'], img_info['depth_to'])
                depth_map[depth_key].append((idx, img_info))
            
            # Generate thumbnail for each depth interval
            for depth_key, images_in_group in depth_map.items():
                # Use first image in group for thumbnail
                first_idx, first_img = images_in_group[0]
                
                # Check if we have pre-loaded this image
                if first_idx in self.preloaded_images:
                    # Use working copy for thumbnail generation (already downscaled)
                    img = self.preloaded_images[first_idx]['working']
                else:
                    # Fallback: load from disk (shouldn't happen if pre-loading worked)
                    img = Image.open(first_img['path'])
                
                # Create thumbnail
                target_height = 60
                scale = target_height / img.height
                new_width = int(img.width * scale)
                img_resized = img.resize((new_width, target_height), Image.LANCZOS)
                img_resized = img_resized.convert('RGB')
                
                photo = ImageTk.PhotoImage(img_resized)
                self.preloaded_filmstrip[depth_key] = photo
                
            logger.info(f"Pre-generated {len(self.preloaded_filmstrip)} filmstrip thumbnails")
            
        except Exception as e:
            logger.error(f"Error pre-loading filmstrip thumbnails: {e}")
        
    def _load_compartments(self, hole_id: str):
        """Load compartment images with UID tracking."""
        # Try approved compartments first, then compartments (extracted)
        compartments_path = self.file_manager.get_shared_path("approved_compartments", create_if_missing=False)
        if not compartments_path:
            compartments_path = self.file_manager.get_shared_path("compartments", create_if_missing=False)
        
        if not compartments_path:
            logger.warning("No approved_compartments or compartments path configured")
            return
        
        logger.info(f"=" * 80)
        logger.info(f"LOADING COMPARTMENTS FOR: {hole_id}")
        logger.info(f"Base compartments path: {compartments_path}")
        
        # Clear existing data
        self.compartment_images = {}
        self.compartment_by_uid = {}
        
        project_code = hole_id[:2].upper() if len(hole_id) >= 2 else ""
        
        search_paths = [
            compartments_path / project_code / hole_id,
            compartments_path / hole_id,
        ]
        
        logger.info(f"Searching in paths:")
        for path in search_paths:
            logger.info(f"  - {path} (exists: {path.exists()})")
        
        for path in search_paths:
            if not path.exists():
                continue
            
            logger.info(f"Scanning directory: {path}")
            
            # Single scan for all compartments
            files_scanned = 0
            files_matched = 0
            
            for img_file in path.iterdir():
                files_scanned += 1
                
                if not img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    logger.debug(f"  Skipped (not image): {img_file.name}")
                    continue
                
                # Parse depth from filename (e.g., "OK0069_CC_123_Wet.jpg" or "OK0069_CC_123.jpg")
                # Pattern: [HoleID]_CC_[Depth]_[Wet/Dry/optional].[ext]
                match = re.search(r'_CC_(\d+)(?:_(Wet|Dry))?', img_file.stem)
                if not match:
                    logger.debug(f"  Skipped (no depth match): {img_file.name}")
                    continue
                
                depth = int(match.group(1))
                moisture = match.group(2)  # 'Wet', 'Dry', or None
                
                # Extract UID once
                try:
                    uid = self.file_manager.extract_uid_from_any_image(str(img_file))
                    if not uid:
                        # Only log first few to avoid spam
                        if files_matched <= 3:
                            logger.warning(f"  ⚠️ No UID in: {img_file.name}")
                except Exception as e:
                    logger.warning(f"  ❌ UID extraction failed for {img_file.name}: {e}")
                    uid = None
                
                files_matched += 1
                uid_display = f"{uid[:8]}...{uid[-8:]}" if uid and len(uid) > 16 else (uid if uid else 'NONE')
                logger.debug(f"  Matched: {img_file.name} -> depth={depth}, uid={uid_display}")
                
                comp_info = {
                    'path': img_file,
                    'uid': uid,
                    'depth': depth,
                    'name': img_file.name
                }
                
                # Store by depth (handling multiple compartments at same depth)
                if depth not in self.compartment_images:
                    self.compartment_images[depth] = comp_info
                    if depth in [20, 40, 60, 80, 100, 120, 140, 160, 180]:  # Log boundary depths
                        logger.info(f"   ✅ Stored NEW compartment at depth {depth}: {img_file.name}")
                elif isinstance(self.compartment_images[depth], list):
                    self.compartment_images[depth].append(comp_info)
                    if depth in [20, 40, 60, 80, 100, 120, 140, 160, 180]:
                        logger.info(f"   📚 Added to LIST at depth {depth} (now {len(self.compartment_images[depth])} items)")
                else:
                    # Convert to list if we have multiple at same depth
                    existing = self.compartment_images[depth]
                    self.compartment_images[depth] = [existing, comp_info]
                    if depth in [20, 40, 60, 80, 100, 120, 140, 160, 180]:
                        logger.info(f"   🔄 Converted to LIST at depth {depth}")
                
                # Store by UID for quick lookup
                if uid:
                    if uid not in self.compartment_by_uid:
                        self.compartment_by_uid[uid] = {}
                    self.compartment_by_uid[uid][depth] = comp_info
            
            logger.info(f"Scan complete: {files_scanned} files scanned, {files_matched} matched pattern")
            break  # Found the folder
        
        logger.info(f"COMPARTMENT LOADING SUMMARY:")
        logger.info(f"  Total depths with compartments: {len(self.compartment_images)}")
        logger.info(f"  Unique source UIDs: {len(self.compartment_by_uid)}")
        
        # DEBUG: Show all loaded depths
        all_depths = sorted(self.compartment_images.keys())
        logger.info(f"  All loaded depths: {all_depths[:30]}{'...' if len(all_depths) > 30 else ''}")
        
        # Check for gaps at interval boundaries
        gaps_at_boundaries = [d for d in [20, 40, 60, 80, 100, 120, 140, 160, 180] if d not in self.compartment_images and d <= max(all_depths, default=0)]
        if gaps_at_boundaries:
            logger.warning(f"  ⚠️ Missing compartments at interval boundaries: {gaps_at_boundaries}")
        
        if self.compartment_by_uid:
            logger.info(f"  Source UIDs found:")
            for uid in list(self.compartment_by_uid.keys())[:5]:  # Show first 5
                depth_count = len(self.compartment_by_uid[uid])
                logger.info(f"    - {uid[:16]}... ({depth_count} compartments)")
            if len(self.compartment_by_uid) > 5:
                logger.info(f"    ... and {len(self.compartment_by_uid) - 5} more")
        
        logger.info(f"=" * 80)

    def display_image(self, index: int, force_reload: bool = False):
        """Display original at index."""
        if not self.original_images or index >= len(self.original_images):
            return
        
        self.current_index = index
        img_info = self.original_images[index]
        
        logger.info(f"=" * 80)
        logger.info(f"DISPLAYING IMAGE: {img_info['name']}")
        logger.info(f"  UID: {img_info.get('uid', 'NONE')}")
        logger.info(f"  Depth range: {img_info['depth_from']}-{img_info['depth_to']}m")
        logger.info(f"  Center depth: {(img_info['depth_from'] + img_info['depth_to']) // 2}m")
        
        # Load image from cache
        image = self.image_cache.get(str(img_info['path']))
        if image and self.image_viewer.load_image(image):
            # Update display mode based on corners toggle
            if self.show_corners_var.get():
                self._show_with_corners(img_info)
            else:
                self._show_without_corners(img_info)
            
            # Update filmstrip
            self.filmstrip.update_filmstrip(self.original_images, index)
            
            # Update status with interval info
            depth_from = img_info['depth_from']
            depth_to = img_info['depth_to']
            
            # Count images at this interval
            same_interval_count = sum(1 for img in self.original_images 
                                     if img['depth_from'] == depth_from 
                                     and img['depth_to'] == depth_to)
            
            if same_interval_count > 1:
                # Find position in interval
                same_interval_indices = [i for i, img in enumerate(self.original_images)
                                        if img['depth_from'] == depth_from 
                                        and img['depth_to'] == depth_to]
                current_pos = same_interval_indices.index(index) + 1
                
                self.status_label.config(
                    text=f"{self.hole_id} | {depth_from}-{depth_to}m | Image {current_pos}/{same_interval_count}"
                )
            else:
                self.status_label.config(
                    text=f"{self.hole_id} | {depth_from}-{depth_to}m"
                )

    def _show_with_corners(self, img_info: Dict):
        """Show image fitted to canvas with corner overlays."""
        logger.debug(f"Showing image with corners: {img_info.get('name')}")
        
        # Enable pan/zoom even with corners (user can zoom in to inspect)
        self.image_viewer.set_pan_zoom_enabled(True)
        
        # Fit to canvas initially
        self.image_viewer.fit_to_canvas()
        
        # Get corners from register
        logger.info(f"=" * 80)
        logger.info(f"GETTING CORNERS FOR IMAGE:")
        logger.info(f"  Image: {img_info.get('name')}")
        logger.info(f"  UID: {img_info.get('uid', 'NONE')}")
        
        corners_data = self._get_corners_for_image(img_info)
        
        logger.info(f"Corner data result: {len(corners_data) if corners_data else 0} datasets found")
        
        if corners_data:
            logger.info(f"✅ Found {len(corners_data)} corner datasets for UID {img_info.get('uid')}")
            for idx, corner in enumerate(corners_data):
                logger.info(f"  Corner {idx}: depth={corner.get('depth')}, corners={len(corner.get('corners', []))} points")
            
            # Don't scale here - let draw_corner_overlays handle all scaling
            # Just pass the original image size so it can calculate the correct scale factor
            
            # Need to map compartment depths
            depth_map = {c['depth']: self.compartment_images.get(c['depth'], {}) for c in corners_data}
            
            # Calculate original image size (before downscaling)
            if hasattr(self.image_viewer, 'downscale_factor') and self.image_viewer.downscale_factor < 1.0:
                scale = self.image_viewer.downscale_factor
                original_size = (int(self.image_viewer.working_image.width / scale), 
                               int(self.image_viewer.working_image.height / scale))
                logger.info(f"Original image size: {original_size}, working size: {self.image_viewer.working_image.size}, scale: {scale:.3f}")
            else:
                original_size = None
            
            self.image_viewer.draw_corner_overlays(corners_data, depth_map, original_image_size=original_size, uid=img_info.get('uid'))
            
            # Set up hover callback for corners on main image
            self.image_viewer.set_corner_hover_callback(
                lambda depth, is_hovering: self.on_compartment_hover(depth, is_hovering, True) if depth else None,
                corners_data
            )
        else:
            logger.warning(f"No corner data found for UID {img_info.get('uid')}")
        
        # Update compartment strip - show ALL compartments, marking active/inactive
        center_depth = (img_info['depth_from'] + img_info['depth_to']) // 2
        depth_range = img_info['depth_to'] - img_info['depth_from']  # Calculate actual range
        
        # Get compartments from current image
        current_uid = img_info.get('uid')
        active_compartments = self._get_compartments_for_display(
            center_depth, 
            current_uid, 
            depth_range,
            depth_from=img_info['depth_from'],
            depth_to=img_info['depth_to']
        )
        
        self.compartment_strip.update_slots(
            active_compartments,
            center_depth,
            depth_range=depth_range,  # Use actual range, not hardcoded 20
            current_uid=current_uid
        )
        
    def _show_without_corners(self, img_info: Dict):
        """Show image with pan/zoom enabled, no overlays."""
        logger.debug(f"Showing image without corners: {img_info.get('name')}")
        
        # Enable pan/zoom
        self.image_viewer.set_pan_zoom_enabled(True)
        
        # Clear overlays
        self.image_viewer.clear_overlays()
        
        # Fit to canvas initially
        self.image_viewer.fit_to_canvas()
        
        # Update compartment strip
        center_depth = (img_info['depth_from'] + img_info['depth_to']) // 2
        depth_range = img_info['depth_to'] - img_info['depth_from']  # Calculate actual range
        current_uid = img_info.get('uid')
        active_compartments = self._get_compartments_for_display(
            center_depth, 
            current_uid, 
            depth_range,
            depth_from=img_info['depth_from'],
            depth_to=img_info['depth_to']
        )
        
        self.compartment_strip.update_slots(
            active_compartments,
            center_depth,
            depth_range=depth_range,  # Use actual range, not hardcoded 20
            current_uid=current_uid
        )

    def _get_corners_for_image(self, img_info: Dict) -> List[Dict]:
        """Get corner data for image - uses pre-loaded cache if available."""
        uid = img_info.get('uid')
        
        if not uid:
            logger.warning(f"  ❌ No UID in img_info")
            return []
        
        # Check pre-loaded cache first
        if uid in self.preloaded_corners:
            logger.debug(f"  ✅ Using pre-loaded corner data for UID {uid} ({len(self.preloaded_corners[uid])} corners)")
            return self.preloaded_corners[uid]
        
        # Fallback: query register (shouldn't happen if pre-loading worked)
        logger.warning(f"  Corner data not pre-loaded for UID {uid}, querying register...")
        
        if not self.json_register_manager:
            logger.warning(f"  ❌ No json_register_manager available")
            return []
        
        try:
            logger.info(f"  Querying corners data for hole: {self.hole_id}")
            corners_df = self.json_register_manager.get_compartment_corners_data(hole_id=self.hole_id)
            
            if corners_df.empty or 'Source_Image_UID' not in corners_df.columns:
                logger.warning(f"  ❌ No corner data in register for hole {self.hole_id}")
                return []
            
            # Filter for this image
            matching = corners_df[corners_df['Source_Image_UID'] == uid]
            logger.info(f"  Found {len(matching)} matching records for UID {uid}")
            
            corners_data = []
            depth_from = img_info.get('depth_from', 0)
            
            for _, row in matching.iterrows():
                corners = [
                    [int(row['Top_Left_X']), int(row['Top_Left_Y'])],
                    [int(row['Top_Right_X']), int(row['Top_Right_Y'])],
                    [int(row['Bottom_Right_X']), int(row['Bottom_Right_Y'])],
                    [int(row['Bottom_Left_X']), int(row['Bottom_Left_Y'])]
                ]
                
                compartment_num = row.get('Compartment_Number', 0)
                actual_depth = depth_from + compartment_num
                
                corners_data.append({
                    'corners': corners,
                    'depth': actual_depth
                })
            
            return corners_data
            
        except Exception as e:
            logger.error(f"Error getting corners: {e}")
            return []
            
    def _get_compartments_for_display(self, center_depth: int, current_uid: str, depth_range: int = 20, depth_from: int = 0, depth_to: int = 20) -> Dict:
        """
        Get compartments for display, mixing all available compartments but 
        marking which are from current image.
        
        Args:
            center_depth: Center depth of the tray
            current_uid: UID of current original image
            depth_range: Total depth range of the tray (e.g., 10, 20, 40)
            depth_from: Starting depth of tray (e.g., 0, 20, 40)
            depth_to: Ending depth of tray (e.g., 20, 40, 60)
        
        Returns dict with all compartments, regardless of source image.
        """
        display_compartments = {}
        
        # Detect compartment interval by looking at available compartments in this range
        available_depths = sorted([d for d in self.compartment_images.keys() 
                                  if depth_from < d <= depth_to])
        
        if not available_depths:
            return display_compartments
        
        # Detect interval from spacing between consecutive compartments
        if len(available_depths) >= 2:
            interval = available_depths[1] - available_depths[0]
        else:
            interval = 1  # Fallback to 1m
        
        logger.debug(f"Detected compartment interval: {interval}m for range {depth_from}-{depth_to}m")
        logger.debug(f"Available compartments in range: {available_depths[:10]}{'...' if len(available_depths) > 10 else ''}")
        
        # Generate expected depth sequence based on interval
        # For 0-20m at 1m interval: [1, 2, 3, ..., 20]
        # For 0-40m at 2m interval: [2, 4, 6, ..., 40]
        expected_depths = list(range(depth_from + interval, depth_to + 1, interval))
        
        logger.debug(f"Expected depths: {expected_depths[:10]}{'...' if len(expected_depths) > 10 else ''}")
        
        # Get compartments at expected depths
        for depth in expected_depths:
            if depth in self.compartment_images:
                comp_data = self.compartment_images[depth]
                
                # Handle case where multiple compartments exist at this depth
                if isinstance(comp_data, list):
                    # Prefer compartment from current image if available
                    found_current = False
                    for comp in comp_data:
                        if comp.get('uid') == current_uid:
                            display_compartments[depth] = comp
                            found_current = True
                            break
                    
                    # If no compartment from current image, use first available
                    if not found_current:
                        display_compartments[depth] = comp_data[0]
                else:
                    # Single compartment at this depth
                    display_compartments[depth] = comp_data
                    
        return display_compartments

    def toggle_corners(self):
        """Toggle corner overlay display."""
        if self.original_images and self.current_index < len(self.original_images):
            self.display_image(self.current_index)
            
    def navigate_previous(self):
        """Navigate to previous image."""
        if self.current_index > 0:
            self.display_image(self.current_index - 1)
            
    def navigate_next(self):
        """Navigate to next image."""
        if self.current_index < len(self.original_images) - 1:
            self.display_image(self.current_index + 1)
            
    def on_filmstrip_select(self, index: int):
        """Callback when image selected from filmstrip."""
        self.display_image(index)
        
    def on_compartment_hover(self, depth: int, is_hovering: bool, is_active: bool):
        """Handle hover over compartment - show popup with full compartment image."""
        if is_hovering and depth in self.compartment_images:
            comp_info = self.compartment_images[depth]
            
            # Get compartment info
            if isinstance(comp_info, list):
                comp_info = comp_info[0]  # Use first if multiple
            
            comp_path = comp_info.get('path')
            if not comp_path or not Path(comp_path).exists():
                return
            
            # Create popup if it doesn't exist
            if not self._hover_popup:
                self._hover_popup = tk.Toplevel(self.window)
                self._hover_popup.withdraw()
                self._hover_popup.overrideredirect(True)
                self._hover_popup.attributes("-topmost", True)
                
                # Frame with border
                frame = tk.Frame(
                    self._hover_popup,
                    bg=self.theme_colors["accent_blue"],
                    relief=tk.SOLID,
                    borderwidth=2
                )
                frame.pack()
                
                # Label for image
                self._hover_popup_label = tk.Label(
                    frame,
                    bg=self.theme_colors["secondary_bg"]
                )
                self._hover_popup_label.pack(padx=2, pady=2)
            
            try:
                # Load compartment image at larger size
                img = Image.open(comp_path)
                
                # Resize to reasonable popup size (max 300px wide, maintain aspect)
                max_width = 300
                aspect = img.height / img.width
                display_width = min(max_width, img.width)
                display_height = int(display_width * aspect)
                
                img_resized = img.resize((display_width, display_height), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img_resized)
                
                self._hover_popup_label.config(image=photo)
                self._current_hover_image = photo  # Keep reference
                
                # Position popup near mouse cursor
                x = self.window.winfo_pointerx() + 20
                y = self.window.winfo_pointery() + 20
                
                # Adjust if too close to screen edge
                screen_width = self.window.winfo_screenwidth()
                screen_height = self.window.winfo_screenheight()
                
                if x + display_width + 50 > screen_width:
                    x = self.window.winfo_pointerx() - display_width - 20
                if y + display_height + 50 > screen_height:
                    y = self.window.winfo_pointery() - display_height - 20
                
                self._hover_popup.geometry(f"+{x}+{y}")
                self._hover_popup.deiconify()
                
            except Exception as e:
                logger.error(f"Error showing hover popup: {e}")
        else:
            # Hide popup
            if self._hover_popup:
                self._hover_popup.withdraw()
                self._current_hover_image = None

    def on_compartment_click(self, depth: int, comp_data: Dict):
        """Handle click on compartment - switch to source image if different."""
        if not comp_data or not comp_data.get('uid'):
            return
            
        target_uid = comp_data['uid']
        
        # Find original image with this UID
        for idx, img_info in enumerate(self.original_images):
            if img_info.get('uid') == target_uid:
                if idx != self.current_index:
                    logger.info(f"Switching to image {idx} for compartment at depth {depth}")
                    self.display_image(idx)
                break

    def reprocess_current(self):
        """Reprocess current image - move to 'Images to Process' for reprocessing."""
        if not self.original_images or self.current_index >= len(self.original_images):
            DialogHelper.show_message(
                self.window,
                "No Image",
                "No image selected",
                message_type="warning"
            )
            return
        
        img_info = self.original_images[self.current_index]
        
        # Confirm using themed dialog
        confirm = DialogHelper.confirm_dialog(
            self.window,
            "Confirm Reprocess",
            f"Move '{img_info['name']}' to 'Images to Process'?\n\n"
            f"The image will be reprocessed and any duplicate compartments\n"
            f"will be handled by the processing pipeline.\n\n"
            f"Continue?",
            yes_text="Yes, Reprocess",
            no_text="Cancel"
        )
        
        if not confirm:
            return
        
        try:
            logger.info(f"=" * 80)
            logger.info(f"REPROCESSING: {img_info['name']}")
            logger.info(f"  UID: {img_info.get('uid', 'NONE')}")
            
            # Move original to 'Images to Process'
            images_to_process = self.file_manager.get_shared_path("images_to_process", create_if_missing=True)
            dest = images_to_process / img_info['name']
            
            # Handle filename conflicts
            if dest.exists():
                base, ext = os.path.splitext(img_info['name'])
                counter = 1
                while dest.exists():
                    dest = images_to_process / f"{base}_{counter}{ext}"
                    counter += 1
            
            logger.info(f"Moving original from: {img_info['path']}")
            logger.info(f"                  to: {dest}")
            shutil.move(str(img_info['path']), str(dest))
            logger.info(f"✅ Original moved successfully")
            logger.info(f"=" * 80)
            
            DialogHelper.show_message(
                self.window,
                "Reprocess Initiated",
                f"'{img_info['name']}' moved to 'Images to Process'\n\n"
                f"The processing pipeline will:\n"
                f"• Detect and handle duplicate compartments\n"
                f"• Update register entries\n"
                f"• Extract fresh compartments",
                message_type="info"
            )
            
            # Remove from viewer's list and show next
            self.original_images.pop(self.current_index)
            
            # Clear from in-memory caches
            if img_info.get('uid'):
                uid = img_info['uid']
                # Remove compartments from cache (visual only - files remain)
                for depth in list(self.compartment_images.keys()):
                    comp_info = self.compartment_images[depth]
                    if isinstance(comp_info, list):
                        # Handle list of compartments
                        self.compartment_images[depth] = [c for c in comp_info if c.get('uid') != uid]
                        if not self.compartment_images[depth]:
                            del self.compartment_images[depth]
                    elif comp_info.get('uid') == uid:
                        del self.compartment_images[depth]
                
                # Remove from UID cache
                if uid in self.compartment_by_uid:
                    del self.compartment_by_uid[uid]
            
            # Show next image or clear display
            if self.original_images:
                if self.current_index >= len(self.original_images):
                    self.current_index = len(self.original_images) - 1
                self.display_image(self.current_index)
            else:
                self.status_label.config(text="No images remaining")
                # Clear the display
                if hasattr(self, 'image_viewer'):
                    self.image_viewer.clear_overlays()
                
        except Exception as e:
            logger.error(f"Error during reprocess: {e}")
            import traceback
            traceback.print_exc()
            DialogHelper.show_message(
                self.window,
                "Reprocess Error",
                f"Failed to move image:\n{str(e)}",
                message_type="error"
            )

    def _cleanup_resources(self):
        """Clean up resources and free memory. Before loading new hole."""
        logger.info("Cleaning up resources")
        # Clear pre-loaded caches
        logger.info("Clearing pre-loaded caches...")
        self.preloaded_images.clear()
        self.preloaded_corners.clear()
        self.preloaded_filmstrip.clear()
        if hasattr(self, 'preloaded_overlay_images'):
            self.preloaded_overlay_images.clear()
        # Clean up hover popup
        if self._hover_popup:
            try:
                self._hover_popup.destroy()
            except:
                pass
            self._hover_popup = None
            self._current_hover_image = None
        
        # Clear image cache
        if hasattr(self, 'image_cache'):
            self.image_cache.clear()
        
        # Clear photo references
        if hasattr(self, 'image_viewer') and self.image_viewer:
            self.image_viewer.photo = None
            if hasattr(self.image_viewer, 'overlay_image'):
                delattr(self.image_viewer, 'overlay_image')
        
        # Clear compartment images
        if hasattr(self, 'compartment_strip'):
            for slot in self.compartment_strip.slots:
                slot['image'] = None
        
        # Clear filmstrip thumbnails
        if hasattr(self, 'filmstrip'):
            self.filmstrip.thumbnails = []
        
        # Force garbage collection
        gc.collect()

    def _on_hole_selected(self, hole_id: str):
            """Callback when hole is selected from searchable dropdown."""
            if hole_id:
                # Trigger filter/load when item selected from dropdown
                self._on_hole_filter()

    def _show_corner_overlays_with_highlight(self, img_info: Dict, highlight_depth: int):
        """Show corner overlays with specific depth highlighted."""
        # Get corners from register
        corners_data = self._get_corners_for_image(img_info)
        
        if corners_data and self.image_viewer.working_image:
            # Create overlay on working image
            overlay = self.image_viewer.working_image.copy()
            draw = ImageDraw.Draw(overlay, 'RGBA')
            
            # Try to load a font
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta']
            
            for idx, corner_data in enumerate(corners_data):
                if 'corners' not in corner_data:
                    continue
                
                corners = corner_data['corners']
                depth = corner_data.get('depth', 0)
                
                if len(corners) != 4:
                    continue
                
                # Highlight if this is the hovered depth
                is_highlighted = (depth == highlight_depth)
                
                if is_highlighted:
                    color = 'yellow'
                    width = 6
                    alpha = 100
                else:
                    color = colors[idx % len(colors)]
                    width = 3
                    alpha = 30
                
                # Draw polygon outline
                draw.polygon(corners, outline=color, width=width)
                
                # Draw semi-transparent fill
                rgba_color = self.image_viewer._hex_to_rgba(color, alpha=alpha)
                draw.polygon(corners, fill=rgba_color)
                
                # Draw depth label at center
                center_x = sum(c[0] for c in corners) / 4
                center_y = sum(c[1] for c in corners) / 4
                
                label = f"{depth}m"
                
                # Draw label background
                bbox = draw.textbbox((center_x, center_y), label, font=font, anchor="mm")
                bg_color = (255, 255, 0, 255) if is_highlighted else (0, 0, 0, 180)
                draw.rectangle(bbox, fill=bg_color)
                
                # Draw label text
                text_color = 'black' if is_highlighted else color
                draw.text((center_x, center_y), label, fill=text_color, font=font, anchor="mm")
            
            self.image_viewer.overlay_image = overlay
            self.image_viewer._use_preview = False  # Force full resolution for highlight
            self.image_viewer.update_display(force_full_res=True)

    def refresh_hole_list(self):
        """Refresh the available holes in the dropdown."""
        if hasattr(self, 'hole_widget') and isinstance(self.hole_widget, ThemedSearchableOptionMenu):
            hole_list = self._get_available_holes()
            self.hole_widget.set_items(hole_list)

def open_original_image_viewer(parent, file_manager, gui_manager=None, json_register_manager=None):
    """Open the optimized original image viewer."""
    return OptimizedOriginalImageViewer(parent, file_manager, gui_manager, json_register_manager)