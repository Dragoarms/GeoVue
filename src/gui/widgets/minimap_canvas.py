"""
Minimap Canvas Widget
Interactive canvas for displaying collar locations and drawing section lines.
"""

import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
from typing import Optional, Callable, List, Tuple, Union
from processing.CorrelationStep.section_line_selector import SectionLine
from PIL import Image, ImageTk
import warnings


class MinimapCanvas(tk.Canvas):
    """Interactive minimap for drillhole collar selection.
    
    Features:
    - Display collar locations as circles
    - Zoom and pan controls
    - Click-drag to draw section line
    - Highlight selected collars
    - Calculate collars within selection corridor
    """
    
    def __init__(self, parent, config_manager, gui_manager, width=500, height=400, **kwargs):
        """Initialize minimap canvas.
        
        Args:
            parent: Parent widget
            config_manager: ConfigManager for settings
            gui_manager: GUIManager for theme colors
            width: Canvas width in pixels
            height: Canvas height in pixels
            **kwargs: Additional canvas options
        """
        self.config_manager = config_manager
        self.gui_manager = gui_manager
        
        # Get background color from theme
        bg_color = self.gui_manager.theme_colors.get('field_bg', '#F5F5F5')
        
        super().__init__(parent, width=width, height=height, bg=bg_color, 
                        highlightthickness=1, highlightbackground='gray', **kwargs)
        
        # Data
        self.collar_data: Optional[pd.DataFrame] = None
        self.section_line: Optional[SectionLine] = None
        self.selected_collar_ids: List[str] = []
        
        # Visual settings from config
        self.collar_color = self.config_manager.get('correlation_collar_color', '#3498DB')
        self.collar_size = self.config_manager.get('correlation_collar_size', 8)
        self.selected_collar_color = self.config_manager.get('correlation_selected_color', '#E74C3C')
        self.selected_collar_size = self.config_manager.get('correlation_selected_size', 12)
        self.section_line_color = self.config_manager.get('correlation_line_color', '#FF6B6B')
        self.section_line_width = self.config_manager.get('correlation_line_width', 3)
        self.selection_box_color = self.config_manager.get('correlation_box_color', '#FF6B6B')
        self.selection_box_alpha = 50  # 0-255
        
        # Coordinate transformation
        self.zoom_level = self.config_manager.get('correlation_zoom_default', 1.0)
        self.min_zoom = self.config_manager.get('correlation_zoom_min', 0.001)
        self.max_zoom = self.config_manager.get('correlation_zoom_max', 5.0)
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        
        # Map bounds (set when data loaded)
        self.map_min_x = 0.0
        self.map_max_x = 1000.0
        self.map_min_y = 0.0
        self.map_max_y = 1000.0
        
        # Drawing state
        self.draw_start_x: Optional[float] = None
        self.draw_start_y: Optional[float] = None
        self.temp_line_id: Optional[int] = None
        self.temp_box_id: Optional[int] = None
        
        # Panning state
        self.panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        
        # Callbacks
        self.on_selection_changed: Optional[Callable[[List[str]], None]] = None
        
        # Canvas item tracking
        self.collar_items = {}  # hole_id -> canvas_id
        self.collar_labels = {}  # hole_id -> label_id
        self.section_box_id: Optional[int] = None
        self.section_line_id: Optional[int] = None
        
        # Background image state
        self.background_image: Optional[Image.Image] = None
        self.background_photo: Optional[ImageTk.PhotoImage] = None
        self.background_item_id: Optional[int] = None
        self.background_bounds: Optional[Tuple[float, float, float, float]] = None  # (min_x, min_y, max_x, max_y)
        self.background_alpha = 0.5  # Transparency 0.0-1.0
        self.background_z_range: Optional[Tuple[float, float]] = None  # Z range for coloring
        
        # Cache for background rendering (to avoid redrawing on every render)
        self._bg_cache_zoom: Optional[float] = None
        self._bg_cache_pan_x: Optional[float] = None
        self._bg_cache_pan_y: Optional[float] = None
        self._bg_cache_size: Optional[Tuple[int, int]] = None
        self._bg_cache_crop_bounds: Optional[Tuple[float, float, float, float]] = None
        self._bg_needs_redraw = False
        
        # Default section width
        self.default_section_width = 100.0
        
        # Bind events
        self._bind_events()
    
    def set_default_section_width(self, width: float):
        """Set the default width for new section lines.
        
        Args:
            width: Width in map units
        """
        self.default_section_width = float(width)
    
    def _bind_events(self):
        """Bind mouse and keyboard events."""
        # Zoom
        self.bind('<MouseWheel>', self._on_mousewheel)
        self.bind('<Button-4>', self._on_mousewheel)  # Linux scroll up
        self.bind('<Button-5>', self._on_mousewheel)  # Linux scroll down
        
        # Left-click = draw section line (always active)
        self.bind('<ButtonPress-1>', self._on_draw_start)
        self.bind('<B1-Motion>', self._on_draw_motion)
        self.bind('<ButtonRelease-1>', self._on_draw_end)
        
        # Right-click = pan
        self.bind('<ButtonPress-3>', self._start_pan)
        self.bind('<B3-Motion>', self._do_pan)
        self.bind('<ButtonRelease-3>', self._end_pan)
    
    def set_background_image(self, image_path: str, world_file_path: Optional[str] = None,
                            bounds: Optional[Tuple[float, float, float, float]] = None,
                            alpha: float = 0.5) -> bool:
        """Load and validate a background image/raster for the map.
        
        Supported formats:
        - GeoTIFF (.tif, .tiff) with embedded georeferencing
        - PNG (.png) with world file (.pgw)
        - JPEG (.jpg, .jpeg) with world file (.jgw)
        - BMP (.bmp) with world file (.bpw)
        
        For GeoTIFF files containing elevation data, validation checks ensure
        the data represents actual topography and not just a color ramp.
        
        Args:
            image_path: Path to image/GeoTIFF file
            world_file_path: Optional path to world file (for non-GeoTIFF images)
            bounds: Optional (min_x, min_y, max_x, max_y) if georeferencing not available
            alpha: Transparency level (0.0 = transparent, 1.0 = opaque)
            
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            import os
            
            if not os.path.exists(image_path):
                print(f"Background image not found: {image_path}")
                return False
            
            # Determine file type
            ext = os.path.splitext(image_path)[1].lower()
            
            if ext in ['.tif', '.tiff']:
                # Try to load as GeoTIFF
                success = self._load_geotiff(image_path, alpha)
            else:
                # Load as regular image with world file
                success = self._load_image_with_worldfile(image_path, world_file_path, bounds, alpha)
            
            if success:
                print(f"Background image loaded successfully: {image_path}")
                self._bg_needs_redraw = True  # Force redraw on next render
                self.render()
            
            return success
            
        except Exception as e:
            print(f"Error loading background image: {e}")
            return False
    
    def _load_geotiff(self, path: str, alpha: float) -> bool:
        """Load and validate a GeoTIFF file.
        
        Args:
            path: Path to GeoTIFF file
            alpha: Transparency level
            
        Returns:
            True if successfully loaded and validated
        """
        try:
            # Try rasterio first (best for GeoTIFF)
            try:
                import rasterio
                from rasterio.warp import calculate_default_transform, reproject, Resampling
                
                with rasterio.open(path) as src:
                    # Get bounds
                    bounds = src.bounds
                    self.background_bounds = (bounds.left, bounds.bottom, bounds.right, bounds.top)
                    
                    # Read data
                    data = src.read(1)  # Read first band
                    print(f"[DEBUG] GeoTIFF data shape: {data.shape}, dtype: {data.dtype}")
                    
                    # Validate elevation data
                    if not self._validate_elevation_data(data):
                        print("Warning: GeoTIFF does not appear to contain valid elevation data")
                        print("Data may be a color ramp or invalid - skipping background")
                        return False
                    
                    # IMPORTANT: Get Z range from collar data NOW and store it
                    z_range = None
                    print(f"[DEBUG] Checking for collar data...")
                    print(f"[DEBUG]   self.collar_data is None: {self.collar_data is None}")
                    if self.collar_data is not None:
                        print(f"[DEBUG]   collar_data.empty: {self.collar_data.empty}")
                        print(f"[DEBUG]   collar_data shape: {self.collar_data.shape}")
                        print(f"[DEBUG]   collar_data columns: {list(self.collar_data.columns)}")
                    
                    if self.collar_data is not None and not self.collar_data.empty and 'z' in self.collar_data.columns:
                        z_min = self.collar_data['z'].min()
                        z_max = self.collar_data['z'].max()
                        # Add some padding for better visualization
                        z_padding = (z_max - z_min) * 0.2
                        z_range = (z_min - z_padding, z_max + z_padding)
                        print(f"[DEBUG] ✓ Collar Z range: {z_min:.1f} to {z_max:.1f}")
                        print(f"[DEBUG] ✓ Using padded range for topography: {z_range[0]:.1f} to {z_range[1]:.1f}")
                    else:
                        print(f"[DEBUG] ✗ No collar data available for Z range - will use topography range")
                    
                    # Store Z range for future use
                    self.background_z_range = z_range
                    
                    # Convert elevation to RGB image for display
                    print(f"[DEBUG] Converting elevation to RGB image...")
                    img = self._elevation_to_image(data, z_range=z_range)
                    print(f"[DEBUG] Image created: {img.size}, mode: {img.mode}")
                    
                    # Downsample if image is enormous (save memory)
                    MAX_SOURCE_SIZE = 8000  # pixels
                    width, height = img.size
                    if width > MAX_SOURCE_SIZE or height > MAX_SOURCE_SIZE:
                        scale = min(MAX_SOURCE_SIZE / width, MAX_SOURCE_SIZE / height)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        print(f"Downsampling source image: {width}x{height} -> {new_width}x{new_height}")
                        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    self.background_image = img
                    self.background_alpha = alpha
                    return True
                    
            except ImportError:
                # Fallback to PIL if rasterio not available
                print("rasterio not available, attempting to load with PIL...")
                # PIL can read TIFF but won't have georeferencing
                img = Image.open(path)
                
                # Try to read world file
                world_file = self._find_world_file(path)
                if world_file:
                    bounds_data = self._read_world_file(world_file)
                    if bounds_data:
                        # Calculate bounds from world file parameters
                        width, height = img.size
                        pixel_x, rot1, rot2, pixel_y, upper_left_x, upper_left_y = bounds_data
                        
                        min_x = upper_left_x
                        max_x = upper_left_x + width * pixel_x
                        max_y = upper_left_y
                        min_y = upper_left_y + height * pixel_y
                        
                        self.background_bounds = (min_x, min_y, max_x, max_y)
                        self.background_image = img
                        self.background_alpha = alpha
                        return True
                
                print("Warning: No georeferencing found for TIFF file")
                return False
                
        except Exception as e:
            print(f"Error loading GeoTIFF: {e}")
            return False
    
    def _validate_elevation_data(self, data: np.ndarray) -> bool:
        """Validate that raster data represents elevation, not a color ramp.
        
        Args:
            data: Numpy array of raster values
            
        Returns:
            True if data appears to be valid elevation data
        """
        # Remove NaN/NoData values
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) == 0:
            return False
        
        # Check 1: Data should have reasonable variance
        # A simple 0-255 ramp would have very high variance relative to range
        data_std = np.std(valid_data)
        data_range = np.ptp(valid_data)  # peak-to-peak (max - min)
        
        if data_range < 1e-6:  # Essentially flat
            print("Validation failed: Data has no variation")
            return False
        
        # Check 2: Not a simple linear ramp
        # For a perfect 0-255 ramp, sorting would give us a nearly perfect linear sequence
        # Real elevation data has irregular patterns
        sample_size = min(1000, len(valid_data))
        sample = np.random.choice(valid_data.flatten(), sample_size, replace=False)
        sample_sorted = np.sort(sample)
        
        # Check if sorted values form a nearly linear sequence (ramp test)
        # Real terrain should have more irregular distribution
        if len(sample_sorted) > 10:
            # Linear fit
            x = np.arange(len(sample_sorted))
            coeffs = np.polyfit(x, sample_sorted, 1)
            fitted = np.polyval(coeffs, x)
            r_squared = 1 - (np.sum((sample_sorted - fitted)**2) / np.sum((sample_sorted - np.mean(sample_sorted))**2))
            
            if r_squared > 0.999:  # Almost perfect linear fit = likely a ramp
                print(f"Validation failed: Data appears to be a linear ramp (R²={r_squared:.6f})")
                return False
        
        # Check 3: Value range - elevation data should be reasonable
        # Most elevation data is in meters, could be -500 to 9000 range typically
        # But also accept normalized data (0-1) or feet (large numbers)
        min_val = np.min(valid_data)
        max_val = np.max(valid_data)
        
        # If data is in 0-255 range with integer values, it's likely an image not elevation
        if max_val <= 255 and min_val >= 0 and np.allclose(valid_data, valid_data.astype(int)):
            unique_vals = len(np.unique(valid_data))
            if unique_vals <= 256:  # Likely 8-bit image data
                print(f"Validation failed: Data appears to be 8-bit image data ({unique_vals} unique values)")
                return False
        
        print(f"Elevation data validated: range=[{min_val:.2f}, {max_val:.2f}], std={data_std:.2f}")
        return True
    
    def _elevation_to_image(self, data: np.ndarray, z_range: Optional[Tuple[float, float]] = None) -> Image.Image:
        """Convert elevation data to RGB image with hillshade effect.
        
        Args:
            data: Elevation data array
            z_range: Optional (min_z, max_z) to use for color scaling (e.g., from drillholes)
            
        Returns:
            PIL Image
        """
        # Identify NoData values (common values: -99999, -9999, very large negatives)
        # Also exclude NaN and Inf
        valid_mask = (
            ~np.isnan(data) & 
            ~np.isinf(data) & 
            (data > -10000)  # Exclude common NoData values
        )
        
        if not valid_mask.any():
            # All data is invalid
            print("[DEBUG] All elevation data is invalid/NoData")
            return Image.new('RGB', data.shape[::-1], color=(200, 200, 200))
        
        # Get min/max from valid data only
        valid_data = data[valid_mask]
        
        # Use provided z_range if available (e.g., from drillhole collar elevations)
        if z_range is not None:
            data_min, data_max = z_range
            print(f"[DEBUG] Using drillhole elevation range: {data_min:.1f} to {data_max:.1f}")
        else:
            data_min = np.min(valid_data)
            data_max = np.max(valid_data)
            print(f"[DEBUG] Using topography elevation range: {data_min:.1f} to {data_max:.1f}")
        
        if data_max - data_min < 1e-6:
            # Flat data
            normalized = np.zeros_like(data, dtype=float)
        else:
            # Normalize only valid data
            normalized = np.zeros_like(data, dtype=float)
            normalized[valid_mask] = (data[valid_mask] - data_min) / (data_max - data_min)
            # Set invalid data to a neutral value (will be colored as NoData)
            normalized[~valid_mask] = 0.5
        
        # Apply hillshade effect for better visualization
        # Simple gradient-based shading
        try:
            gy, gx = np.gradient(normalized)
            slope = np.sqrt(gx**2 + gy**2)
            
            # Normalize slope (only for valid areas)
            slope_max = np.max(slope[valid_mask]) if valid_mask.any() else 1.0
            slope_norm = slope / (slope_max + 1e-6)
            
            # Combine elevation and slope for shading
            # Higher elevation = lighter, higher slope = darker
            # Make hillshade more dramatic (increase slope influence)
            shaded = normalized * 0.5 + (1 - slope_norm) * 0.5
            
            # Increase contrast
            shaded = np.power(shaded, 0.8)  # Gamma correction for more contrast
            shaded = np.clip(shaded, 0, 1)
            
        except Exception as e:
            print(f"[DEBUG] Hillshade failed: {e}, using simple coloring")
            # Fallback to simple elevation coloring
            shaded = normalized
        
        # Convert to 8-bit and apply colormap (grayscale with brown tones)
        img_array = (shaded * 255).astype(np.uint8)
        
        # Create RGB image with terrain-like colors
        # Low = dark brown/green, high = light tan/white
        img = Image.fromarray(img_array, mode='L')
        
        # Apply a terrain-like colormap
        # Low values (0) = dark green/brown, high values (255) = white/tan
        img = img.convert('RGB')
        
        # Simple terrain coloring
        pixels = np.array(img)
        # Low elevations: darker green-brown (50, 40, 30)
        # Mid elevations: tan (180, 150, 100)
        # High elevations: white-ish (240, 235, 220)
        
        r = pixels[:,:,0]
        g = pixels[:,:,1]
        b = pixels[:,:,2]
        
        # Interpolate colors based on elevation
        norm = r / 255.0
        
        r_out = (50 + norm * 190).astype(np.uint8)
        g_out = (40 + norm * 195).astype(np.uint8)
        b_out = (30 + norm * 190).astype(np.uint8)
        
        colored = np.stack([r_out, g_out, b_out], axis=2)
        
        # Set NoData pixels to transparent/gray
        colored[~valid_mask] = [220, 220, 220]  # Light gray for NoData
        
        img = Image.fromarray(colored, mode='RGB')
        
        return img
    
    def _load_image_with_worldfile(self, image_path: str, world_file_path: Optional[str],
                                   bounds: Optional[Tuple[float, float, float, float]],
                                   alpha: float) -> bool:
        """Load a regular image with world file georeferencing.
        
        Args:
            image_path: Path to image
            world_file_path: Path to world file
            bounds: Manual bounds if no world file
            alpha: Transparency
            
        Returns:
            True if successful
        """
        try:
            img = Image.open(image_path)
            
            # Downsample if image is enormous (save memory)
            MAX_SOURCE_SIZE = 8000  # pixels
            width, height = img.size
            if width > MAX_SOURCE_SIZE or height > MAX_SOURCE_SIZE:
                scale = min(MAX_SOURCE_SIZE / width, MAX_SOURCE_SIZE / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                print(f"Downsampling source image: {width}x{height} -> {new_width}x{new_height}")
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Try to find/read world file
            if world_file_path is None:
                world_file_path = self._find_world_file(image_path)
            
            if world_file_path:
                bounds_data = self._read_world_file(world_file_path)
                if bounds_data:
                    width, height = img.size
                    pixel_x, rot1, rot2, pixel_y, upper_left_x, upper_left_y = bounds_data
                    
                    min_x = upper_left_x
                    max_x = upper_left_x + width * pixel_x
                    max_y = upper_left_y
                    min_y = upper_left_y + height * pixel_y
                    
                    self.background_bounds = (min_x, min_y, max_x, max_y)
                    self.background_image = img
                    self.background_alpha = alpha
                    return True
            
            # Use manual bounds if provided
            if bounds:
                self.background_bounds = bounds
                self.background_image = img
                self.background_alpha = alpha
                return True
            
            print("No georeferencing available for image")
            return False
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def _find_world_file(self, image_path: str) -> Optional[str]:
        """Find associated world file for an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Path to world file or None
        """
        import os
        
        base = os.path.splitext(image_path)[0]
        ext = os.path.splitext(image_path)[1].lower()
        
        # World file extensions
        world_exts = {
            '.tif': ['.tfw', '.tifw', '.tiffw'],
            '.tiff': ['.tfw', '.tifw', '.tiffw'],
            '.png': ['.pgw', '.pngw'],
            '.jpg': ['.jgw', '.jpgw'],
            '.jpeg': ['.jgw', '.jpegw'],
            '.bmp': ['.bpw', '.bmpw']
        }
        
        if ext in world_exts:
            for world_ext in world_exts[ext]:
                world_path = base + world_ext
                if os.path.exists(world_path):
                    return world_path
        
        return None
    
    def _read_world_file(self, world_file_path: str) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Read world file parameters.
        
        World file format (6 lines):
        1. pixel size in x direction
        2. rotation about y-axis
        3. rotation about x-axis
        4. pixel size in y direction (negative)
        5. x-coordinate of upper left pixel center
        6. y-coordinate of upper left pixel center
        
        Args:
            world_file_path: Path to world file
            
        Returns:
            Tuple of (pixel_x, rot1, rot2, pixel_y, ul_x, ul_y) or None
        """
        try:
            with open(world_file_path, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 6:
                    pixel_x = float(lines[0].strip())
                    rot1 = float(lines[1].strip())
                    rot2 = float(lines[2].strip())
                    pixel_y = float(lines[3].strip())
                    ul_x = float(lines[4].strip())
                    ul_y = float(lines[5].strip())
                    
                    return (pixel_x, rot1, rot2, pixel_y, ul_x, ul_y)
        except Exception as e:
            print(f"Error reading world file: {e}")
        
        return None
    
    def clear_background_image(self):
        """Remove the background image."""
        self.background_image = None
        self.background_photo = None
        self.background_item_id = None
        self.background_bounds = None
        self.background_z_range = None
        
        # Clear cache
        self._bg_cache_zoom = None
        self._bg_cache_pan_x = None
        self._bg_cache_pan_y = None
        self._bg_cache_size = None
        self._bg_cache_crop_bounds = None
        self._bg_needs_redraw = False
        
        self.render()
    
    def reload_topography_with_collar_range(self):
        """Reload topography image using current collar data Z range.
        
        Call this after collar data is loaded to get proper coloring.
        """
        if self.background_image is None:
            print("[DEBUG] No topography loaded to reload")
            return
        
        # This assumes we still have access to the original data
        # For now, just force a redraw which will use cached image
        # TODO: Store raw elevation data to enable true recoloring
        print("[DEBUG] Topography reload requested but raw data not stored")
        print("[DEBUG] Recommendation: Reload topography file after loading collar data")

    def set_collar_data(self, collar_df: pd.DataFrame):
        """Load collar location data.
        
        Args:
            collar_df: DataFrame with columns 'holeid', 'x', 'y', 'z' (lowercase)
        """
        self.collar_data = collar_df.copy()
        
        if not collar_df.empty:
            # Calculate map bounds with padding
            self.map_min_x = float(collar_df['x'].min())
            self.map_max_x = float(collar_df['x'].max())
            self.map_min_y = float(collar_df['y'].min())
            self.map_max_y = float(collar_df['y'].max())
            
            # Add 10% padding
            x_range = self.map_max_x - self.map_min_x
            y_range = self.map_max_y - self.map_min_y
            padding_x = max(x_range * 0.1, 10)  # At least 10 units
            padding_y = max(y_range * 0.1, 10)
            
            self.map_min_x -= padding_x
            self.map_max_x += padding_x
            self.map_min_y -= padding_y
            self.map_max_y += padding_y
        
        # Auto-fit to data after a brief delay to ensure canvas is sized
        # Don't render yet - zoom_to_fit will call render
        self.after(100, self.zoom_to_fit)
    
    def zoom_to_fit(self):
        """Reset zoom and pan to show all collars."""
        if self.collar_data is None or self.collar_data.empty:
            self.zoom_level = 1.0
            self.pan_offset_x = 0.0
            self.pan_offset_y = 0.0
            return
        
        # Calculate zoom to fit
        map_width = self.map_max_x - self.map_min_x
        map_height = self.map_max_y - self.map_min_y
        
        canvas_width = self.winfo_width() or 500
        canvas_height = self.winfo_height() or 400
        
        # Scale to fit (with margin)
        margin = 0.9  # Use 90% of canvas
        zoom_x = (canvas_width * margin) / map_width if map_width > 0 else 1.0
        zoom_y = (canvas_height * margin) / map_height if map_height > 0 else 1.0
        
        self.zoom_level = min(zoom_x, zoom_y)
        self.zoom_level = max(self.min_zoom, min(self.max_zoom, self.zoom_level))
        
        # Center the view
        self.pan_offset_x = canvas_width / 2 - (self.map_min_x + map_width / 2) * self.zoom_level
        # Y-axis is inverted (north = up), so we negate the map center Y
        self.pan_offset_y = canvas_height / 2 + (self.map_min_y + map_height / 2) * self.zoom_level
        
        # Render with new zoom
        self.render()
    
    def map_to_canvas(self, map_x: Union[int, float], map_y: Union[int, float]) -> Tuple[float, float]:
        """Convert map coordinates to canvas coordinates.
        
        Args:
            map_x: X coordinate in map units (easting)
            map_y: Y coordinate in map units (northing)
            
        Returns:
            (canvas_x, canvas_y) tuple
            
        Note:
            Y-axis is inverted - canvas Y increases downward, but map Y (north) increases upward
        """
        canvas_x = float(map_x) * self.zoom_level + self.pan_offset_x
        canvas_y = -float(map_y) * self.zoom_level + self.pan_offset_y  # Invert Y for north-up orientation
        return canvas_x, canvas_y
    
    def canvas_to_map(self, canvas_x: Union[int, float], canvas_y: Union[int, float]) -> Tuple[float, float]:
        """Convert canvas coordinates to map coordinates.
        
        Args:
            canvas_x: X coordinate in canvas pixels
            canvas_y: Y coordinate in canvas pixels
            
        Returns:
            (map_x, map_y) tuple
            
        Note:
            Y-axis is inverted - canvas Y increases downward, but map Y (north) increases upward
        """
        map_x = (float(canvas_x) - self.pan_offset_x) / self.zoom_level
        map_y = -(float(canvas_y) - self.pan_offset_y) / self.zoom_level  # Invert Y for north-up orientation
        return map_x, map_y
    
    def render(self):
        """Render all elements on the canvas."""
        import traceback
        import io
        
        # Get caller info
        stack = traceback.extract_stack()
        if len(stack) >= 2:
            caller = stack[-2]
            caller_info = f"{caller.filename.split('/')[-1]}:{caller.lineno} in {caller.name}"
        else:
            caller_info = "unknown"
        
        print(f"\n[RENDER] ===== START RENDER ===== zoom={self.zoom_level:.6f} called from {caller_info}")
        
        # Only delete temporary drawing items and selection items (these change)
        if self.section_box_id:
            self.delete(self.section_box_id)
            self.section_box_id = None
        if self.section_line_id:
            self.delete(self.section_line_id)
            self.section_line_id = None
        if self.temp_line_id:
            self.delete(self.temp_line_id)
            self.temp_line_id = None
        if self.temp_box_id:
            self.delete(self.temp_box_id)
            self.temp_box_id = None
        
        # Draw background image first (bottom layer) - only if it needs updating
        if self.background_image is not None and self.background_bounds is not None:
            # Check if background needs to be redrawn
            if self._should_redraw_background():
                # Delete old background before drawing new one
                if self.background_item_id:
                    self.delete(self.background_item_id)
                    self.background_item_id = None
                self._draw_background_image()
                self._bg_needs_redraw = False
            elif self.background_photo is not None:
                # Background hasn't changed, but we need to ensure it's drawn
                if self.background_item_id is None or not self.find_withtag(self.background_item_id):
                    # Just redraw the cached image without regenerating
                    self._draw_cached_background()
        
        if self.collar_data is None or self.collar_data.empty:
            # Show "No data" message
            canvas_width = self.winfo_width() or 500
            canvas_height = self.winfo_height() or 400
            self.create_text(canvas_width/2, canvas_height/2, 
                           text="No collar data loaded", 
                           fill='gray', font=('Arial', 12))
            return
        
        # Draw selection box (if exists)
        if self.section_line is not None:
            self._draw_selection_box()
        
        # Draw collars
        collars_drawn = 0
        collars_onscreen = 0
        canvas_w = self.winfo_width() or 500
        canvas_h = self.winfo_height() or 400
        
        for idx, row in self.collar_data.iterrows():
            hole_id = row['holeid']
            is_selected = hole_id in self.selected_collar_ids
            canvas_x, canvas_y = self.map_to_canvas(row['x'], row['y'])
            
            # Count how many are actually on screen
            if -50 <= canvas_x <= canvas_w + 50 and -50 <= canvas_y <= canvas_h + 50:
                collars_onscreen += 1
            
            self._draw_collar(row['x'], row['y'], hole_id, is_selected)
            collars_drawn += 1
        
        print(f"[RENDER] Drew {collars_drawn} collars, {collars_onscreen} visible on screen (canvas {canvas_w}x{canvas_h})")
        print(f"[RENDER] collar_items dict now has {len(self.collar_items)} entries")
        print(f"[RENDER] collar_labels dict now has {len(self.collar_labels)} entries")
        
        # Draw section line (if exists)
        if self.section_line is not None:
            self._draw_section_line()
        
        # Draw status info overlay
        self._draw_status_info()
    
    def _draw_collar(self, map_x: Union[int, float], map_y: Union[int, float], hole_id: str, is_selected: bool):
        """Draw a single collar marker.
        
        Args:
            map_x: X coordinate in map units
            map_y: Y coordinate in map units
            hole_id: Drillhole ID
            is_selected: Whether collar is selected
        """
        canvas_x, canvas_y = self.map_to_canvas(map_x, map_y)
        
        size = self.selected_collar_size if is_selected else self.collar_size
        color = self.selected_collar_color if is_selected else self.collar_color
        
        # Calculate circle bounds
        x1 = canvas_x - size/2
        y1 = canvas_y - size/2
        x2 = canvas_x + size/2
        y2 = canvas_y + size/2
        
        # Update existing item if it exists, otherwise create new
        if hole_id in self.collar_items:
            item_id = self.collar_items[hole_id]
            # Update position and appearance
            self.coords(item_id, x1, y1, x2, y2)
            self.itemconfig(item_id, fill=color, outline='black', width=1)
        else:
            # Create new collar
            item_id = self.create_oval(x1, y1, x2, y2, fill=color, outline='black', width=1)
            self.collar_items[hole_id] = item_id
        
        # Handle labels - only show if zoomed in enough
        if self.zoom_level > 0.5:
            text_color = self.gui_manager.theme_colors.get('text', '#000000')
            
            if hole_id in self.collar_labels:
                # Update existing label position
                label_id = self.collar_labels[hole_id]
                self.coords(label_id, canvas_x, canvas_y + size/2 + 10)
                self.itemconfig(label_id, fill=text_color)
            else:
                # Create new label
                label_id = self.create_text(canvas_x, canvas_y + size/2 + 10, 
                                           text=hole_id, font=('Arial', 8), anchor='n',
                                           fill=text_color)
                self.collar_labels[hole_id] = label_id
        else:
            # Delete label if zoomed out
            if hole_id in self.collar_labels:
                self.delete(self.collar_labels[hole_id])
                del self.collar_labels[hole_id]
    
    def _draw_selection_box(self):
        """Draw the section line selection box."""
        if self.section_line is None:
            return
        
        corners = self.section_line.get_selection_box_corners()
        canvas_corners = [self.map_to_canvas(x, y) for x, y in corners]
        
        # Flatten coordinates for polygon
        coords = []
        for x, y in canvas_corners:
            coords.extend([x, y])
        
        # Draw semi-transparent box
        color_rgb = self._hex_to_rgb(self.selection_box_color)
        fill_color = f'#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}'
        
        self.section_box_id = self.create_polygon(
            coords,
            fill=fill_color,
            outline=self.selection_box_color,
            width=2,
            stipple='gray50'  # Simple transparency effect
        )
    
    def _draw_section_line(self):
        """Draw the section line centerline."""
        if self.section_line is None:
            return
        
        start_canvas = self.map_to_canvas(self.section_line.start_x, self.section_line.start_y)
        end_canvas = self.map_to_canvas(self.section_line.end_x, self.section_line.end_y)
        
        self.section_line_id = self.create_line(
            start_canvas[0], start_canvas[1],
            end_canvas[0], end_canvas[1],
            fill=self.section_line_color,
            width=self.section_line_width,
            arrow=tk.LAST
        )
    
    def _should_redraw_background(self) -> bool:
        """Check if background image needs to be redrawn.
        
        Returns:
            True if background should be regenerated
        """
        if self.background_photo is None:
            print("[DEBUG] Redraw: No cached photo")
            return True
            
        if self._bg_needs_redraw:
            print("[DEBUG] Redraw: Flag set")
            return True
        
        # Check if zoom or pan changed significantly
        canvas_size = (self.winfo_width(), self.winfo_height())
        
        # Redraw if zoom changed by more than 5%
        if self._bg_cache_zoom is None:
            print("[DEBUG] Redraw: No cached zoom")
            return True
            
        zoom_change = abs(self.zoom_level - self._bg_cache_zoom) / self._bg_cache_zoom
        if zoom_change > 0.05:
            print(f"[DEBUG] Redraw: Zoom changed {zoom_change*100:.1f}%")
            return True
        
        # Redraw if pan changed by more than 20 pixels
        if self._bg_cache_pan_x is None:
            print("[DEBUG] Redraw: No cached pan")
            return True
            
        pan_x_change = abs(self.pan_offset_x - self._bg_cache_pan_x)
        pan_y_change = abs(self.pan_offset_y - self._bg_cache_pan_y)
        
        if pan_x_change > 20:
            print(f"[DEBUG] Redraw: Pan X changed {pan_x_change:.1f}px")
            return True
        if pan_y_change > 20:
            print(f"[DEBUG] Redraw: Pan Y changed {pan_y_change:.1f}px")
            return True
        
        # Redraw if canvas size changed
        if self._bg_cache_size is None or self._bg_cache_size != canvas_size:
            print(f"[DEBUG] Redraw: Canvas size changed from {self._bg_cache_size} to {canvas_size}")
            return True
        
        print("[DEBUG] Using CACHED background (no redraw needed)")
        return False
    
    def _draw_background_image(self):
        """Draw the background image/raster on the canvas (regenerates the image).
        
        Only processes the portion of the image visible in the current view.
        """
        if self.background_image is None or self.background_bounds is None:
            print("[DEBUG] _draw_background_image: No background image or bounds")
            return
        
        print("\n[DEBUG] ===== DRAWING BACKGROUND IMAGE =====")
        
        try:
            # Get visible area in map coordinates
            canvas_w = self.winfo_width() or 500
            canvas_h = self.winfo_height() or 400
            print(f"[DEBUG] Canvas size: {canvas_w}x{canvas_h}")
            
            # Calculate visible map bounds
            top_left_x, top_left_y = self.canvas_to_map(0, 0)
            bottom_right_x, bottom_right_y = self.canvas_to_map(canvas_w, canvas_h)
            
            # Ensure min < max (handle any coordinate system)
            visible_min_x = min(top_left_x, bottom_right_x)
            visible_max_x = max(top_left_x, bottom_right_x)
            visible_min_y = min(top_left_y, bottom_right_y)
            visible_max_y = max(top_left_y, bottom_right_y)
            
            print(f"[DEBUG] Visible map bounds: X({visible_min_x:.1f} to {visible_max_x:.1f}), Y({visible_min_y:.1f} to {visible_max_y:.1f})")
            
            # Get image bounds
            img_min_x, img_min_y, img_max_x, img_max_y = self.background_bounds
            print(f"[DEBUG] Image bounds: X({img_min_x:.1f} to {img_max_x:.1f}), Y({img_min_y:.1f} to {img_max_y:.1f})")
            
            # Calculate intersection (visible portion of image)
            crop_min_x = max(visible_min_x, img_min_x)
            crop_max_x = min(visible_max_x, img_max_x)
            crop_min_y = max(visible_min_y, img_min_y)
            crop_max_y = min(visible_max_y, img_max_y)
            print(f"[DEBUG] Crop bounds: X({crop_min_x:.1f} to {crop_max_x:.1f}), Y({crop_min_y:.1f} to {crop_max_y:.1f})")
            
            # Check if image is visible at all
            if crop_min_x >= crop_max_x or crop_min_y >= crop_max_y:
                print("[DEBUG] Image completely outside visible area - SKIPPING")
                return
            
            # Calculate crop region in image pixel coordinates
            img_w, img_h = self.background_image.size
            img_map_w = img_max_x - img_min_x
            img_map_h = img_max_y - img_min_y
            print(f"[DEBUG] Source image size: {img_w}x{img_h} pixels")
            print(f"[DEBUG] Image map size: {img_map_w:.1f}x{img_map_h:.1f} map units")
            
            # Convert map crop bounds to image pixel bounds
            crop_left = int((crop_min_x - img_min_x) / img_map_w * img_w)
            crop_right = int((crop_max_x - img_min_x) / img_map_w * img_w)
            crop_top = int((img_max_y - crop_max_y) / img_map_h * img_h)  # Y is inverted
            crop_bottom = int((img_max_y - crop_min_y) / img_map_h * img_h)
            
            # Clamp to image bounds
            crop_left = max(0, min(crop_left, img_w))
            crop_right = max(0, min(crop_right, img_w))
            crop_top = max(0, min(crop_top, img_h))
            crop_bottom = max(0, min(crop_bottom, img_h))
            
            if crop_right <= crop_left or crop_bottom <= crop_top:
                print("[DEBUG] Invalid crop dimensions - SKIPPING")
                return
            
            print(f"[DEBUG] Cropping image pixels: left={crop_left}, top={crop_top}, right={crop_right}, bottom={crop_bottom}")
            
            # Crop to visible region only
            cropped = self.background_image.crop((crop_left, crop_top, crop_right, crop_bottom))
            print(f"[DEBUG] Cropped image size: {cropped.size}")
            
            # Calculate target size on canvas
            crop_canvas_tl = self.map_to_canvas(crop_min_x, crop_max_y)
            crop_canvas_br = self.map_to_canvas(crop_max_x, crop_min_y)
            
            target_width = abs(crop_canvas_br[0] - crop_canvas_tl[0])
            target_height = abs(crop_canvas_br[1] - crop_canvas_tl[1])
            print(f"[DEBUG] Target canvas size: {target_width:.1f}x{target_height:.1f}")
            
            if target_width < 1 or target_height < 1:
                print("[DEBUG] Target size too small - SKIPPING")
                return
            
            # Safety limits - skip if too large
            MAX_SIZE = 2000
            if target_width > MAX_SIZE or target_height > MAX_SIZE:
                print(f"[DEBUG] Background region too large ({target_width:.0f}x{target_height:.0f}), SKIPPING")
                print(f"[DEBUG] Try zooming in to display topography")
                return
            
            # Resize cropped region to target canvas size
            target_width = int(target_width)
            target_height = int(target_height)
            print(f"[DEBUG] Resizing to: {target_width}x{target_height}")
            
            resized = cropped.resize(
                (target_width, target_height),
                Image.Resampling.LANCZOS
            )
            print(f"[DEBUG] Resize complete")
            
            # Apply transparency
            if self.background_alpha < 1.0:
                if resized.mode != 'RGBA':
                    resized = resized.convert('RGBA')
                
                alpha = resized.split()[3]
                alpha = alpha.point(lambda p: int(p * self.background_alpha))
                resized.putalpha(alpha)
            
            # Convert to PhotoImage
            print(f"[DEBUG] Converting to PhotoImage...")
            self.background_photo = ImageTk.PhotoImage(resized)
            print(f"[DEBUG] PhotoImage created successfully")
            
            # Cache current state
            self._bg_cache_zoom = self.zoom_level
            self._bg_cache_pan_x = self.pan_offset_x
            self._bg_cache_pan_y = self.pan_offset_y
            self._bg_cache_size = (self.winfo_width(), self.winfo_height())
            self._bg_cache_crop_bounds = (crop_min_x, crop_min_y, crop_max_x, crop_max_y)
            
            # Draw on canvas
            print(f"[DEBUG] Drawing cached background...")
            self._draw_cached_background()
            print(f"[DEBUG] ===== BACKGROUND DRAWING COMPLETE =====\n")
            
        except MemoryError as e:
            print(f"[ERROR] MemoryError rendering background - image too large for current zoom")
            print(f"[ERROR] Try zooming in to display topography")
            import traceback
            traceback.print_exc()
        except Exception as e:
            print(f"[ERROR] Error drawing background image: {e}")
            import traceback
            traceback.print_exc()

    def _draw_cached_background(self):
        """Draw the already-generated background photo without regenerating."""
        if self.background_photo is None or self._bg_cache_crop_bounds is None:
            print("[DEBUG] _draw_cached_background: No photo or crop bounds")
            return
        
        try:
            # Use cached crop bounds (the actual region that was rendered)
            crop_min_x, crop_min_y, crop_max_x, crop_max_y = self._bg_cache_crop_bounds
            
            # Convert crop bounds to canvas coordinates
            top_left = self.map_to_canvas(crop_min_x, crop_max_y)
            bottom_right = self.map_to_canvas(crop_max_x, crop_min_y)
            
            # Draw on canvas
            center_x = (top_left[0] + bottom_right[0]) / 2
            center_y = (top_left[1] + bottom_right[1]) / 2
            
            print(f"[DEBUG] Placing background image at canvas position ({center_x:.1f}, {center_y:.1f})")
            
            self.background_item_id = self.create_image(
                center_x, center_y,
                image=self.background_photo,
                anchor='center'
            )
            
            print(f"[DEBUG] Background item ID: {self.background_item_id}")
            
            # Send to back
            self.tag_lower(self.background_item_id)
            print(f"[DEBUG] Background sent to back layer")
            
        except Exception as e:
            print(f"[ERROR] Error drawing cached background: {e}")
            import traceback
            traceback.print_exc()
    
    def _draw_status_info(self):
        """Draw status information overlay (coordinates at center)."""
        if not hasattr(self, '_status_text_ids'):
            self._status_text_ids = []
        
        # Clear old status text
        for text_id in self._status_text_ids:
            self.delete(text_id)
        self._status_text_ids.clear()
        
        # Get center point coordinates
        canvas_w = self.winfo_width() or 500
        canvas_h = self.winfo_height() or 400
        center_map_x, center_map_y = self.canvas_to_map(canvas_w / 2, canvas_h / 2)
        
        # Draw coordinate text only (top-left corner)
        text_color = self.gui_manager.theme_colors.get('text', '#000000')
        text_id = self.create_text(
            10, 10,
            text=f"E: {center_map_x:.1f}  N: {center_map_y:.1f}",
            font=('Arial', 9),
            anchor='nw',
            fill=text_color
        )
        self._status_text_ids.append(text_id)

    def _on_draw_start(self, event):
        """Start drawing section line."""
        # Convert to map coordinates
        self.draw_start_x, self.draw_start_y = self.canvas_to_map(event.x, event.y)
        
        # Set crosshair cursor
        self.config(cursor='crosshair')
        
        # Clear previous preview if exists
        if self.temp_line_id:
            self.delete(self.temp_line_id)
            self.temp_line_id = None
        if self.temp_box_id:
            self.delete(self.temp_box_id)
            self.temp_box_id = None
    
    def _on_draw_motion(self, event):
        """Update section line preview while dragging."""
        if self.draw_start_x is None:
            return
        
        # Delete previous preview elements
        if self.temp_line_id:
            self.delete(self.temp_line_id)
        if self.temp_box_id:
            self.delete(self.temp_box_id)
        
        # Draw preview line (dashed)
        start_canvas = self.map_to_canvas(self.draw_start_x, self.draw_start_y)
        self.temp_line_id = self.create_line(
            start_canvas[0], start_canvas[1],
            event.x, event.y,
            fill=self.section_line_color,
            width=2,
            dash=(5, 5)
        )
        
        # Draw preview selection box
        current_x, current_y = self.canvas_to_map(event.x, event.y)
        
        # Get current width from existing section line or default
        preview_width = self.default_section_width
        if self.section_line is not None:
            preview_width = self.section_line.width
        
        # Create temporary section line for preview box
        temp_section = SectionLine(
            start_x=self.draw_start_x,
            start_y=self.draw_start_y,
            end_x=current_x,
            end_y=current_y,
            width=preview_width  # Use current width setting
        )
        
        # Draw preview box
        corners = temp_section.get_selection_box_corners()
        canvas_corners = [self.map_to_canvas(x, y) for x, y in corners]
        coords = []
        for x, y in canvas_corners:
            coords.extend([x, y])
        
        color_rgb = self._hex_to_rgb(self.selection_box_color)
        fill_color = f'#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}'
        
        self.temp_box_id = self.create_polygon(
            coords,
            fill=fill_color,
            outline=self.selection_box_color,
            width=2,
            stipple='gray50',
            dash=(5, 5)
        )
    
    def _on_draw_end(self, event):
        """Finish drawing section line."""
        if self.draw_start_x is None:
            return
        
        # Delete preview elements
        if self.temp_line_id:
            self.delete(self.temp_line_id)
            self.temp_line_id = None
        if self.temp_box_id:
            self.delete(self.temp_box_id)
            self.temp_box_id = None
        
        # Reset cursor
        self.config(cursor='')
        
        # Convert end point to map coordinates
        end_x, end_y = self.canvas_to_map(event.x, event.y)
        
        # Get current width from existing section line, or use default
        current_width = self.default_section_width
        if self.section_line is not None:
            current_width = self.section_line.width
        
        # Create section line with current width setting
        self.section_line = SectionLine(
            start_x=self.draw_start_x,
            start_y=self.draw_start_y,
            end_x=end_x,
            end_y=end_y,
            width=current_width
        )
        
        # Reset drawing state
        self.draw_start_x = None
        self.draw_start_y = None
        
        # Check if Shift key is held - if so, add to selection instead of replacing
        # Shift key bit mask: 0x0001
        append_selection = bool(event.state & 0x0001)
        
        print(f"[DRAW_END] Shift held: {append_selection}, event.state: {event.state}")
        
        # Update selection
        self.update_selection_from_line(append=append_selection)
        
        # Clear the section line visualization after selection is made
        # Keep the section_line data but remove visual elements
        if self.section_box_id:
            self.delete(self.section_box_id)
            self.section_box_id = None
        if self.section_line_id:
            self.delete(self.section_line_id)
            self.section_line_id = None
        
        # Render
        self.render()
    
    def set_section_line_width(self, width: Union[int, float]):
        """Update section line width and recalculate selection.
        
        Args:
            width: New width in map units
        """
        if self.section_line is not None:
            self.section_line.width = float(width)
            self.update_selection_from_line()
            self.render()
    
    def update_selection_from_line(self, append=False):
        """Update selected collars based on current section line.
        
        Args:
            append: If True, add to existing selection. If False, replace selection.
        """
        if self.section_line is None or self.collar_data is None:
            if not append:  # Only clear if not appending
                self.selected_collar_ids = []
            return
        
        # Get collars in corridor
        selected_collars = self.section_line.get_collars_in_corridor(self.collar_data)
        
        # Sort by distance
        selected_collars = self.section_line.sort_collars_by_distance(selected_collars)
        
        # Get new hole IDs
        new_hole_ids = selected_collars['holeid'].tolist()
        
        print(f"[UPDATE_SELECTION] Append mode: {append}")
        print(f"[UPDATE_SELECTION] Existing selection: {len(self.selected_collar_ids)} holes")
        print(f"[UPDATE_SELECTION] New holes from line: {len(new_hole_ids)} holes")
        
        # Update selection - append or replace
        if append:
            # Toggle mode: add if not present, remove if already selected
            existing_set = set(self.selected_collar_ids)
            added = 0
            removed = 0
            
            for hole_id in new_hole_ids:
                if hole_id in existing_set:
                    # Already selected - remove it (toggle off)
                    self.selected_collar_ids.remove(hole_id)
                    existing_set.remove(hole_id)
                    removed += 1
                else:
                    # Not selected - add it (toggle on)
                    self.selected_collar_ids.append(hole_id)
                    existing_set.add(hole_id)
                    added += 1
            
            print(f"[UPDATE_SELECTION] Toggle mode: added {added}, removed {removed}")
            print(f"[UPDATE_SELECTION] After toggle: {len(self.selected_collar_ids)} holes total")
        else:
            # Replace selection
            self.selected_collar_ids = new_hole_ids
            print(f"[UPDATE_SELECTION] Replaced with {len(self.selected_collar_ids)} holes")
        
        # Callback
        if self.on_selection_changed:
            self.on_selection_changed(self.selected_collar_ids)
    
    def clear_section_line(self):
        """Remove section line and clear selection."""
        self.section_line = None
        self.selected_collar_ids = []
        
        if self.section_box_id:
            self.delete(self.section_box_id)
            self.section_box_id = None
        if self.section_line_id:
            self.delete(self.section_line_id)
            self.section_line_id = None
        
        self.render()
        
        if self.on_selection_changed:
            self.on_selection_changed(self.selected_collar_ids)
    
    def _on_mousewheel(self, event):
        """Handle zoom with mouse wheel."""
        # Determine zoom direction
        if event.num == 4 or event.delta > 0:  # Scroll up = zoom in
            zoom_factor = 1.1
            direction = "IN"
        else:  # Scroll down = zoom out
            zoom_factor = 0.9
            direction = "OUT"
        
        old_zoom = self.zoom_level
        
        # Calculate new zoom
        new_zoom = self.zoom_level * zoom_factor
        new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))
        
        print(f"\n[MOUSEWHEEL] Zoom {direction}: {old_zoom:.6f} -> {new_zoom:.6f} (factor={zoom_factor})")
        
        # Zoom towards mouse cursor
        mouse_x = event.x
        mouse_y = event.y
        
        # Map coordinate at mouse position (before zoom)
        map_x, map_y = self.canvas_to_map(mouse_x, mouse_y)
        
        # Update zoom
        self.zoom_level = new_zoom
        
        # Adjust pan so that map coordinate stays under mouse
        # Note: Y-axis is inverted in map_to_canvas (canvas_y = -map_y * zoom + pan_y)
        # So: pan_y = canvas_y + map_y * zoom
        self.pan_offset_x = mouse_x - map_x * self.zoom_level
        self.pan_offset_y = mouse_y + map_y * self.zoom_level  # Changed from minus to plus
        
        print(f"[MOUSEWHEEL] New pan: ({self.pan_offset_x:.1f}, {self.pan_offset_y:.1f})")
        
        # Re-render
        self.render()
    
    def _start_pan(self, event):
        """Start panning."""
        self.panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.config(cursor='fleur')
    
    def _do_pan(self, event):
        """Pan the view."""
        if not self.panning:
            return
        
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        
        self.pan_offset_x += dx
        self.pan_offset_y += dy
        
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        
        self.render()
    
    def _end_pan(self, event):
        """End panning."""
        self.panning = False
        self.config(cursor='')
    
    @staticmethod
    def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def get_section_line(self) -> Optional[SectionLine]:
        """Get current section line.
        
        Returns:
            SectionLine object or None
        """
        return self.section_line
    
    def get_selected_collar_ids(self) -> List[str]:
        """Get list of selected collar IDs.
        
        Returns:
            List of hole IDs
        """
        return self.selected_collar_ids.copy()