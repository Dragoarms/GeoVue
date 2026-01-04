#!/usr/bin/env python3
"""
GeoVue Validation Tool

A GUI application for validating compartment extraction by overlaying 
compartment boundaries on original images and allowing visual verification
of the processing pipeline.
"""

import sys
import os
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageTk, ImageDraw
import pandas as pd
import numpy as np
import cv2

# Add src to path for imports
current_dir = Path(__file__).parent
src_path = current_dir / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

# Import config manager after path setup
try:
    from core.config_manager import ConfigManager
except ImportError:
    ConfigManager = None
    logger.warning("ConfigManager not available - using manual path selection")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageViewer:
    """Image viewer with zoom and pan capabilities."""
    
    def __init__(self, canvas: tk.Canvas):
        self.canvas = canvas
        self.image = None
        self.photo_image = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.drag_data = {"x": 0, "y": 0}
        
        # Bind events
        self.canvas.bind("<ButtonPress-1>", self.start_drag)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<Button-4>", self.zoom)  # Linux
        self.canvas.bind("<Button-5>", self.zoom)  # Linux
        
    def load_image(self, image_path: str) -> bool:
        """Load and display an image."""
        try:
            self.image = Image.open(image_path)
            self.scale = 1.0
            self.offset_x = 0
            self.offset_y = 0
            self.update_display()
            return True
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return False
    
    def draw_compartment_overlays(self, corners_data: List[Dict], show_labels: bool = True, image_record: Dict = None):
        """Draw compartment boundary overlays on the image with rotation correction and extracted images."""
        if not self.image:
            logger.warning("No image loaded for drawing overlays")
            return
        
        logger.info(f"Drawing overlays for {len(corners_data)} compartments")
        
        # Check for rotation data in the first record
        rotation_angle = 0.0
        if image_record and 'Total_Rotation_Angle' in image_record:
            rotation_angle = float(image_record['Total_Rotation_Angle'])
            logger.info(f"Image rotation angle: {rotation_angle:.2f}°")
        
        # Create composite image with space for extracted compartments above
        original_height = self.image.height
        original_width = self.image.width
        compartment_height = 200  # Height for extracted compartment display
        total_height = original_height + compartment_height + 20  # Extra padding
        
        # Create larger canvas
        composite_image = Image.new('RGB', (original_width, total_height), color='white')
        
        # Paste original image at bottom
        composite_image.paste(self.image, (0, compartment_height + 20))
        
        # Create draw object for the composite image
        draw = ImageDraw.Draw(composite_image)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta']
        drawn_count = 0
        
        # First pass: collect extracted images and their positions
        extracted_images = []
        
        for i, compartment in enumerate(corners_data):
            logger.debug(f"Processing compartment {i+1}: {compartment}")
            
            if 'corners' not in compartment:
                logger.warning(f"Compartment {i+1} missing 'corners' key")
                continue
                
            corners = compartment['corners']
            if len(corners) != 4:
                logger.warning(f"Compartment {i+1} has {len(corners)} corners, expected 4")
                continue
                
            # Apply rotation correction if needed
            corrected_corners = corners
            if rotation_angle != 0:
                corrected_corners = self.apply_rotation_to_corners(corners, rotation_angle, original_width, original_height)
            
            # Get color for this compartment
            color = colors[i % len(colors)]
            compartment_number = compartment.get('compartment_number', i+1)
            
            logger.info(f"Drawing compartment {compartment_number} with rotation correction: {rotation_angle:.2f}° in {color}")
            
            # Adjust corners for composite image (shift down by compartment_height + 20)
            adjusted_corners = [[corner[0], corner[1] + compartment_height + 20] for corner in corrected_corners]
            polygon_points = [tuple(corner) for corner in adjusted_corners]
            
            # Draw outline
            for j in range(len(polygon_points)):
                start = polygon_points[j]
                end = polygon_points[(j + 1) % len(polygon_points)]
                draw.line([start, end], fill=color, width=3)
            
            # Draw corner dots
            for corner in adjusted_corners:
                x, y = corner
                draw.ellipse([x-5, y-5, x+5, y+5], fill=color, outline='white', width=1)
            
            # Draw label on original image area
            if show_labels:
                center_x = sum(corner[0] for corner in adjusted_corners) / 4
                center_y = sum(corner[1] for corner in adjusted_corners) / 4
                label = str(compartment_number)
                draw.text((center_x, center_y), label, fill=color, anchor='mm')
                
            # Collect extracted compartment image info
            extracted_info = self.get_extracted_compartment_info(compartment, corrected_corners, color, i)
            if extracted_info:
                extracted_images.append(extracted_info)
                
            drawn_count += 1
        
        # Second pass: paste extracted images onto composite
        for img_info in extracted_images:
            try:
                composite_image.paste(img_info['image'], (img_info['x'], img_info['y']))
                
                # Draw border around extracted image
                draw.rectangle([img_info['x']-2, img_info['y']-2, 
                              img_info['x'] + img_info['image'].width + 2, 
                              img_info['y'] + img_info['image'].height + 2], 
                             outline=img_info['color'], width=2)
                
                # Label the extracted image
                draw.text((img_info['x'], img_info['y'] + img_info['image'].height + 5), 
                         f"Comp {img_info['compartment_number']}", fill=img_info['color'])
                
                logger.info(f"Displayed extracted compartment {img_info['compartment_number']} at ({img_info['x']}, {img_info['y']})")
                
            except Exception as e:
                logger.error(f"Error pasting extracted compartment image: {e}")
            
        logger.info(f"Successfully drew {drawn_count} compartment overlays with rotation correction and {len(extracted_images)} extracted images")
        
        # Update the displayed image
        self.image_with_overlay = composite_image
        self.update_display()
    
    def apply_rotation_to_corners(self, corners: List[List[float]], angle: float, img_width: int, img_height: int) -> List[List[float]]:
        """Apply rotation correction to corner coordinates."""
        if angle == 0:
            return corners
            
        # Convert to numpy array
        corners_array = np.array(corners, dtype=np.float32)
        
        # Create rotation matrix around image center
        center = (img_width / 2, img_height / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)  # Negative to correct rotation
        
        # Add homogeneous coordinate
        ones = np.ones((corners_array.shape[0], 1), dtype=np.float32)
        corners_homog = np.hstack([corners_array, ones])
        
        # Apply rotation
        rotated_corners = np.dot(rotation_matrix, corners_homog.T).T
        
        return rotated_corners.tolist()
    
    def get_extracted_compartment_info(self, compartment: Dict, corners: List[List[float]], color: str, index: int) -> Optional[Dict]:
        """Get extracted compartment image info for display."""
        if not hasattr(self, 'compartment_images_folder') or not self.compartment_images_folder:
            return None
            
        # Calculate position based on compartment X coordinate
        min_x = min(corner[0] for corner in corners)
        max_x = max(corner[0] for corner in corners)
        center_x = (min_x + max_x) / 2
        
        # Look for extracted compartment image
        compartment_number = compartment.get('compartment_number', index + 1)
        
        # Try to find the extracted image file
        # Common patterns for compartment image naming
        search_patterns = [
            f"*compartment_{compartment_number}*",
            f"*comp_{compartment_number}*", 
            f"*{compartment_number}*"
        ]
        
        extracted_image_path = None
        for pattern in search_patterns:
            for img_path in self.compartment_images_folder.rglob(pattern):
                if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}:
                    extracted_image_path = img_path
                    break
            if extracted_image_path:
                break
        
        if extracted_image_path:
            try:
                # Load and resize extracted image
                extracted_img = Image.open(extracted_image_path)
                
                # Resize to fit in display area
                max_width = 150
                max_height = 180
                extracted_img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                
                # Position aligned with compartment center
                display_x = int(center_x - extracted_img.width / 2)
                display_y = 10  # Top margin
                
                # Ensure it doesn't go off edges
                display_x = max(0, min(display_x, self.image.width - extracted_img.width))
                
                return {
                    'image': extracted_img,
                    'x': display_x,
                    'y': display_y,
                    'color': color,
                    'compartment_number': compartment_number
                }
                
            except Exception as e:
                logger.error(f"Error loading extracted compartment image {extracted_image_path}: {e}")
                return None
        else:
            logger.debug(f"No extracted image found for compartment {compartment_number}")
            return None
    
    def update_display(self):
        """Update the canvas display with current image, scale, and offset."""
        # Use overlay image if it exists, otherwise use the base image
        display_source = getattr(self, 'image_with_overlay', self.image)
        
        if not display_source:
            return
            
        # Calculate display size
        width = int(display_source.width * self.scale)
        height = int(display_source.height * self.scale)
        
        # Resize image
        if self.scale != 1.0:
            display_image = display_source.resize((width, height), Image.Resampling.LANCZOS)
        else:
            display_image = display_source
        
        # Convert to PhotoImage
        self.photo_image = ImageTk.PhotoImage(display_image)
        
        # Clear canvas and draw image
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.photo_image)
        
        # Update scroll region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def start_drag(self, event):
        """Start dragging the image."""
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y
    
    def drag(self, event):
        """Drag the image."""
        delta_x = event.x - self.drag_data["x"]
        delta_y = event.y - self.drag_data["y"]
        self.offset_x += delta_x
        self.offset_y += delta_y
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y
        self.update_display()
    
    def zoom(self, event):
        """Zoom the image."""
        if event.delta > 0 or event.num == 4:
            self.scale *= 1.1
        else:
            self.scale /= 1.1
        
        # Limit zoom range
        self.scale = max(0.1, min(self.scale, 5.0))
        self.update_display()
    
    def fit_to_canvas(self):
        """Fit the image to the canvas size."""
        if not self.image:
            return
            
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
            
        scale_x = canvas_width / self.image.width
        scale_y = canvas_height / self.image.height
        self.scale = min(scale_x, scale_y) * 0.9  # 90% to add some padding
        
        # Center the image
        self.offset_x = (canvas_width - self.image.width * self.scale) / 2
        self.offset_y = (canvas_height - self.image.height * self.scale) / 2
        
        self.update_display()


class ValidationTool:
    """Main validation tool GUI application."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("GeoVue Validation Tool")
        self.root.geometry("1400x900")
        
        # Initialize config manager
        if ConfigManager:
            # Determine default config path (same logic as main app)
            if getattr(sys, "frozen", False):
                # Running as compiled executable
                base_path = sys._MEIPASS
            else:
                # Running as script - config is in src folder
                base_path = src_path
            
            default_config_path = os.path.join(base_path, "config.json")
            if os.path.exists(default_config_path):
                self.config_manager = ConfigManager(default_config_path)
            else:
                logger.warning(f"Config file not found at {default_config_path}")
                self.config_manager = None
        else:
            self.config_manager = None
        
        # Data storage - initialize with config paths if available
        self.original_images_folder = None
        self.compartment_images_folder = None
        self.register_data_folder = None
        self.current_image_data = None
        self.image_list = []
        self.current_index = 0
        
        self.setup_ui()
        self.setup_styles()
        
        # Pre-populate paths from config if available
        self.load_config_paths()
    
    def load_config_paths(self):
        """Load paths from config manager and pre-populate the UI."""
        if not self.config_manager:
            logger.info("Config manager not available - using manual path selection")
            return
        
        try:
            # Get paths from config manager
            storage_type = self.config_manager.get('storage_type', 'local')
            logger.info(f"Storage type: {storage_type}")
            
            # Try to get paths from explicit config settings first (for shared/both storage)
            shared_processed_originals = self.config_manager.get('shared_folder_processed_originals')
            shared_approved_folder = self.config_manager.get('shared_folder_approved_folder') 
            shared_extracted_compartments = self.config_manager.get('shared_folder_extracted_compartments_folder')
            shared_register_data = self.config_manager.get('shared_folder_register_data_folder')
            
            # Set original images folder - prefer approved folder, fall back to processed originals
            if shared_approved_folder and Path(shared_approved_folder).exists():
                self.original_images_folder = Path(shared_approved_folder)
                logger.info(f"Set original images folder from config (approved): {self.original_images_folder}")
            elif shared_processed_originals and Path(shared_processed_originals).exists():
                self.original_images_folder = Path(shared_processed_originals)
                logger.info(f"Set original images folder from config (processed): {self.original_images_folder}")
            
            # Set compartments folder 
            if shared_extracted_compartments and Path(shared_extracted_compartments).exists():
                self.compartment_images_folder = Path(shared_extracted_compartments)
                logger.info(f"Set compartment images folder from config: {self.compartment_images_folder}")
            
            # Set register data folder
            if shared_register_data and Path(shared_register_data).exists():
                self.register_data_folder = Path(shared_register_data)
                logger.info(f"Set register data folder from config: {self.register_data_folder}")
            
            # If shared paths didn't work, try local paths
            if storage_type in ['local', 'both'] and not all([self.original_images_folder, self.compartment_images_folder, self.register_data_folder]):
                local_base_path = self.config_manager.get('local_folder_path')
                if local_base_path and Path(local_base_path).exists():
                    base_path = Path(local_base_path)
                    logger.info(f"Trying local paths from: {base_path}")
                    
                    # Try various folder names for processed originals
                    if not self.original_images_folder:
                        processed_names = ['ProcessedOriginals', 'Processed Originals', 'OriginalImages']
                        for folder_name in processed_names:
                            processed_originals = base_path / folder_name
                            if processed_originals.exists():
                                self.original_images_folder = processed_originals
                                logger.info(f"Set original images folder from local config: {processed_originals}")
                                break
                    
                    # Try various folder names for compartments  
                    if not self.compartment_images_folder:
                        compartment_names = ['ChipCompartments', 'Chip Compartments', 'Compartments']
                        for folder_name in compartment_names:
                            compartments = base_path / folder_name
                            if compartments.exists():
                                self.compartment_images_folder = compartments
                                logger.info(f"Set compartment images folder from local config: {compartments}")
                                break
                        
                    # Try various folder names for register data
                    if not self.register_data_folder:
                        register_names = ['RegisterData', 'Register Data', 'Chip Tray Register', 'Data']
                        for folder_name in register_names:
                            register_data = base_path / folder_name
                            if register_data.exists() and any(register_data.glob("*.json")):
                                self.register_data_folder = register_data
                                logger.info(f"Set register data folder from local config: {register_data}")
                                break
                        
        except Exception as e:
            logger.error(f"Error loading config paths: {e}")
        
        # Update UI with loaded paths
        self.update_folder_displays()
    
    def update_folder_displays(self):
        """Update the folder display variables with current paths."""
        if hasattr(self, 'orig_folder_var') and self.original_images_folder:
            self.orig_folder_var.set(str(self.original_images_folder))
        
        if hasattr(self, 'comp_folder_var') and self.compartment_images_folder:
            self.comp_folder_var.set(str(self.compartment_images_folder))
            
        if hasattr(self, 'reg_folder_var') and self.register_data_folder:
            self.reg_folder_var.set(str(self.register_data_folder))
        
        # Update status with what was auto-configured (only if status_var exists)
        if hasattr(self, 'status_var'):
            configured_count = sum([
                1 for folder in [self.original_images_folder, self.compartment_images_folder, self.register_data_folder] 
                if folder is not None
            ])
            
            if configured_count > 0:
                status_msg = f"Auto-configured {configured_count}/3 folders from GeoVue config"
                if configured_count == 3:
                    status_msg += " - Ready to Auto-Load!"
                else:
                    status_msg += " - Use Browse buttons for missing folders"
                self.status_var.set(status_msg)
            else:
                self.status_var.set("No folders auto-configured - Please select folders manually")
        
    def setup_styles(self):
        """Setup custom styles for the application."""
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Section.TLabel', font=('Arial', 10, 'bold'))
        
    def setup_ui(self):
        """Setup the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="GeoVue Processing Validation Tool", style='Title.TLabel')
        title_label.pack(pady=(0, 10))
        
        # Top frame for folder selection
        folder_frame = ttk.LabelFrame(main_frame, text="Folder Selection", padding=10)
        folder_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Folder selection widgets
        self.setup_folder_selection(folder_frame)
        
        # Middle frame for image navigation and controls
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.setup_navigation_controls(nav_frame)
        
        # Bottom frame for image display
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        self.setup_image_display(display_frame)
        
    def setup_folder_selection(self, parent):
        """Setup folder selection controls."""
        # Original images folder
        orig_frame = ttk.Frame(parent)
        orig_frame.pack(fill=tk.X, pady=2)
        ttk.Label(orig_frame, text="Original Images:").pack(side=tk.LEFT, padx=(0, 10))
        self.orig_folder_var = tk.StringVar()
        ttk.Entry(orig_frame, textvariable=self.orig_folder_var, state='readonly').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        ttk.Button(orig_frame, text="Browse", command=self.select_original_folder).pack(side=tk.RIGHT)
        
        # Compartment images folder
        comp_frame = ttk.Frame(parent)
        comp_frame.pack(fill=tk.X, pady=2)
        ttk.Label(comp_frame, text="Compartments:").pack(side=tk.LEFT, padx=(0, 10))
        self.comp_folder_var = tk.StringVar()
        ttk.Entry(comp_frame, textvariable=self.comp_folder_var, state='readonly').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        ttk.Button(comp_frame, text="Browse", command=self.select_compartment_folder).pack(side=tk.RIGHT)
        
        # Register data folder
        reg_frame = ttk.Frame(parent)
        reg_frame.pack(fill=tk.X, pady=2)
        ttk.Label(reg_frame, text="Register Data:").pack(side=tk.LEFT, padx=(0, 10))
        self.reg_folder_var = tk.StringVar()
        ttk.Entry(reg_frame, textvariable=self.reg_folder_var, state='readonly').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        ttk.Button(reg_frame, text="Browse", command=self.select_register_folder).pack(side=tk.RIGHT)
        
        # Auto-load and Load Data buttons
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="Auto-Load from Config", command=self.auto_load_data).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Load Validation Data", command=self.load_data).pack(side=tk.LEFT)
        
    def setup_navigation_controls(self, parent):
        """Setup image navigation controls."""
        # Image selection
        select_frame = ttk.Frame(parent)
        select_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Label(select_frame, text="Image:").pack(side=tk.LEFT, padx=(0, 5))
        self.image_combo = ttk.Combobox(select_frame, state='readonly', width=30)
        self.image_combo.pack(side=tk.LEFT, padx=(0, 5))
        self.image_combo.bind('<<ComboboxSelected>>', self.on_image_selected)
        
        # Navigation buttons
        nav_buttons_frame = ttk.Frame(parent)
        nav_buttons_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Button(nav_buttons_frame, text="Previous", command=self.previous_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_buttons_frame, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=2)
        
        # View controls
        view_frame = ttk.Frame(parent)
        view_frame.pack(side=tk.LEFT)
        
        ttk.Button(view_frame, text="Fit to Window", command=self.fit_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(view_frame, text="Actual Size", command=self.actual_size).pack(side=tk.LEFT, padx=2)
        
        # Overlay controls
        overlay_frame = ttk.Frame(parent)
        overlay_frame.pack(side=tk.RIGHT)
        
        self.show_overlays_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(overlay_frame, text="Show Overlays", variable=self.show_overlays_var, 
                       command=self.toggle_overlays).pack(side=tk.LEFT, padx=2)
        
        self.show_labels_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(overlay_frame, text="Show Labels", variable=self.show_labels_var,
                       command=self.toggle_overlays).pack(side=tk.LEFT, padx=2)
    
    def setup_image_display(self, parent):
        """Setup image display area."""
        # Create main display frame
        display_container = ttk.Frame(parent)
        display_container.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for image
        image_frame = ttk.LabelFrame(display_container, text="Original Image with Compartment Overlays", padding=5)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Canvas with scrollbars
        canvas_frame = ttk.Frame(image_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg='white')
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack scrollbars and canvas
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Initialize image viewer
        self.image_viewer = ImageViewer(self.canvas)
        
        # Right panel for information
        info_frame = ttk.LabelFrame(display_container, text="Image Information", padding=5)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        info_frame.configure(width=300)
        info_frame.pack_propagate(False)
        
        # Information display
        self.info_text = tk.Text(info_frame, width=35, height=20, wrap=tk.WORD)
        info_scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scrollbar.set)
        
        info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Please select folders and load data")
        status_bar = ttk.Label(parent, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
    
    def auto_load_data(self):
        """Auto-load data using paths from config manager."""
        try:
            # Re-load paths from config in case they've changed
            self.load_config_paths()
            
            # Check if all required paths are set
            if not all([self.original_images_folder, self.register_data_folder]):
                missing = []
                if not self.original_images_folder:
                    missing.append("Original Images folder")
                if not self.register_data_folder:
                    missing.append("Register Data folder")
                
                messagebox.showwarning("Missing Paths", 
                    f"Could not auto-load: Missing {', '.join(missing)}.\n"
                    f"Please check your GeoVue configuration or use Browse buttons.")
                return
            
            # Load the data
            self.load_data()
            
        except Exception as e:
            logger.error(f"Error in auto-load: {e}")
            messagebox.showerror("Auto-Load Error", f"Failed to auto-load data: {str(e)}")
    
    def select_original_folder(self):
        """Select the original images folder."""
        folder = filedialog.askdirectory(title="Select Original Images Folder")
        if folder:
            self.original_images_folder = Path(folder)
            self.orig_folder_var.set(str(folder))
    
    def select_compartment_folder(self):
        """Select the compartment images folder."""
        folder = filedialog.askdirectory(title="Select Compartment Images Folder")
        if folder:
            self.compartment_images_folder = Path(folder)
            self.comp_folder_var.set(str(folder))
    
    def select_register_folder(self):
        """Select the register data folder."""
        folder = filedialog.askdirectory(title="Select Register Data Folder")
        if folder:
            self.register_data_folder = Path(folder)
            self.reg_folder_var.set(str(folder))
    
    def load_data(self):
        """Load and process the data from selected folders."""
        if not all([self.original_images_folder, self.compartment_images_folder, self.register_data_folder]):
            messagebox.showerror("Error", "Please select all three folders before loading data.")
            return
        
        try:
            self.status_var.set("Loading data...")
            self.root.update()
            
            # Load register data
            register_data = self.load_register_data()
            if not register_data:
                messagebox.showerror("Error", "No valid register data found.")
                return
            
            # Find matching original images
            self.image_list = self.find_matching_images(register_data)
            if not self.image_list:
                messagebox.showerror("Error", "No matching images found between register and original images folder.")
                return
            
            # Update UI
            image_names = [img['display_name'] for img in self.image_list]
            self.image_combo['values'] = image_names
            if image_names:
                self.image_combo.current(0)
                self.current_index = 0
                self.load_current_image()
            
            self.status_var.set(f"Loaded {len(self.image_list)} images for validation")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.status_var.set("Error loading data")
    
    def load_register_data(self) -> Dict:
        """Load register data from JSON files."""
        register_data = {}
        
        # Look for register files in the data folder
        for file_path in self.register_data_folder.glob("*.json"):
            if "corners" in file_path.name.lower():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for record in data:
                                if all(key in record for key in ['HoleID', 'Original_Filename']):
                                    key = f"{record['HoleID']}_{record['Original_Filename']}"
                                    # Store multiple records per image key
                                    if key not in register_data:
                                        register_data[key] = []
                                    register_data[key].append(record)
                        logger.info(f"Loaded {len(data)} records from {file_path.name}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
        
        total_records = sum(len(records) for records in register_data.values())
        logger.info(f"Total register records loaded: {total_records} across {len(register_data)} images")
        return register_data
    
    def find_matching_images(self, register_data: Dict) -> List[Dict]:
        """Find original images that have corresponding register data."""
        matching_images = []
        
        # Get all image files from original images folder
        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        
        for image_path in self.original_images_folder.rglob("*"):
            if image_path.suffix.lower() in image_extensions:
                # Find ALL records that match this image
                matching_records = []
                
                # Check each group of records in register_data
                for key, records in register_data.items():
                    if records:  # records is now a list
                        original_filename = records[0].get('Original_Filename', '')
                        
                        # Check if filenames match (with or without extension)
                        if (image_path.name == original_filename or 
                            image_path.stem == Path(original_filename).stem):
                            matching_records = records  # Use all records for this image
                            break
                
                if matching_records:
                    # Extract corners data from ALL matching records
                    all_corners_data = []
                    for record in matching_records:
                        corners = self.extract_corners_from_record(record)
                        if corners:
                            # Add compartment number to each corner set
                            for corner_set in corners:
                                corner_set['compartment_number'] = record.get('Compartment_Number', 0)
                            all_corners_data.extend(corners)
                    
                    # Sort by compartment number if available
                    all_corners_data.sort(key=lambda x: x.get('compartment_number', 0))
                    
                    # Use the first record for general info, but keep all corners
                    matching_images.append({
                        'image_path': image_path,
                        'register_record': matching_records[0],  # Keep first record for general info
                        'all_records': matching_records,  # Keep all records
                        'corners_data': all_corners_data,
                        'display_name': f"{matching_records[0].get('HoleID', 'Unknown')} - {image_path.name}"
                    })
                    
                    logger.info(f"Found {len(all_corners_data)} compartments for {image_path.name}")
        
        logger.info(f"Found {len(matching_images)} matching images")
        return matching_images
    
    def extract_corners_from_record(self, record: Dict) -> List[Dict]:
        """Extract corner coordinates from a register record."""
        corners_data = []
        
        # Check for named corner fields (Top_Left_X, Top_Right_X, etc.)
        corner_names = {
            'Top_Left': 0,      # TL
            'Top_Right': 1,     # TR
            'Bottom_Right': 2,  # BR
            'Bottom_Left': 3    # BL
        }
        
        corners = [None, None, None, None]
        
        for corner_name, index in corner_names.items():
            x_key = f"{corner_name}_X"
            y_key = f"{corner_name}_Y"
            
            if x_key in record and y_key in record:
                try:
                    x = float(record[x_key])
                    y = float(record[y_key])
                    corners[index] = [x, y]
                except (ValueError, TypeError):
                    logger.warning(f"Invalid corner values for {corner_name}: X={record.get(x_key)}, Y={record.get(y_key)}")
        
        # Check if we have all 4 corners
        if all(corner is not None for corner in corners):
            corners_data.append({'corners': corners})
            logger.info(f"Extracted corners: {corners}")
        else:
            # Also check for individual corner fields (Corner_1_X, Corner_1_Y, etc.) as fallback
            corner_fields = {}
            for key, value in record.items():
                if key.startswith('Corner_') and ('_X' in key or '_Y' in key):
                    corner_fields[key] = value
            
            if corner_fields:
                # Group by corner number
                corners_by_num = {}
                for key, value in corner_fields.items():
                    parts = key.split('_')
                    if len(parts) >= 3:
                        corner_num = parts[1]
                        coord = parts[2]  # X or Y
                        
                        if corner_num not in corners_by_num:
                            corners_by_num[corner_num] = {}
                        corners_by_num[corner_num][coord] = value
                
                # Convert to corners format
                if corners_by_num:
                    corners = []
                    for i in ['1', '2', '3', '4']:  # TL, TR, BR, BL
                        if i in corners_by_num and 'X' in corners_by_num[i] and 'Y' in corners_by_num[i]:
                            corners.append([corners_by_num[i]['X'], corners_by_num[i]['Y']])
                    
                    if len(corners) == 4:
                        corners_data.append({'corners': corners})
        
        # Check for JSON-encoded corners data
        for key, value in record.items():
            if 'corner' in key.lower() and isinstance(value, str):
                try:
                    parsed_data = json.loads(value)
                    if isinstance(parsed_data, dict) and 'corners' in parsed_data:
                        corners_data.append(parsed_data)
                except:
                    pass
        
        return corners_data
    
    def load_current_image(self):
        """Load the currently selected image and its data."""
        if not self.image_list or self.current_index >= len(self.image_list):
            return
        
        self.current_image_data = self.image_list[self.current_index]
        image_path = self.current_image_data['image_path']
        
        # Load image
        if self.image_viewer.load_image(str(image_path)):
            # Update overlays
            self.update_overlays()
            
            # Update information panel
            self.update_info_panel()
            
            self.status_var.set(f"Image {self.current_index + 1} of {len(self.image_list)}: {image_path.name}")
        else:
            messagebox.showerror("Error", f"Failed to load image: {image_path}")
    
    def update_overlays(self):
        """Update the compartment overlays on the current image."""
        if not self.current_image_data or not self.show_overlays_var.get():
            logger.info("Not updating overlays - no data or overlays disabled")
            return
        
        corners_data = self.current_image_data.get('corners_data', [])
        logger.info(f"Updating overlays with {len(corners_data)} compartments")
        
        # Log all compartment data for debugging
        for i, comp_data in enumerate(corners_data):
            logger.info(f"Compartment {i+1} data: {comp_data}")
            if 'corners' in comp_data:
                corners = comp_data['corners']
                logger.info(f"  Corners count: {len(corners)}")
                logger.info(f"  Corners: {corners}")
            else:
                logger.warning(f"  Missing 'corners' key in compartment {i+1}")
        
        if corners_data:
            # Get the first record for rotation data
            image_record = self.current_image_data.get('register_record', {})
            self.image_viewer.draw_compartment_overlays(
                corners_data, 
                show_labels=self.show_labels_var.get(),
                image_record=image_record
            )
        else:
            logger.warning("No corners data found for overlay")
    
    def update_info_panel(self):
        """Update the information panel with current image data."""
        if not self.current_image_data:
            return
        
        self.info_text.delete(1.0, tk.END)
        
        record = self.current_image_data['register_record']
        all_records = self.current_image_data.get('all_records', [record])
        image_path = self.current_image_data['image_path']
        corners_data = self.current_image_data['corners_data']
        
        info_text = "IMAGE INFORMATION\n"
        info_text += "=" * 30 + "\n\n"
        
        info_text += f"File: {image_path.name}\n"
        info_text += f"Path: {image_path}\n\n"
        
        info_text += "REGISTER DATA\n"
        info_text += "-" * 20 + "\n"
        info_text += f"Total Compartments in Register: {len(all_records)}\n"
        info_text += f"HoleID: {record.get('HoleID', 'N/A')}\n"
        info_text += f"Depth From: {record.get('Depth_From', 'N/A')}\n"
        info_text += f"Depth To: {record.get('Depth_To', 'N/A')}\n"
        info_text += f"Original Filename: {record.get('Original_Filename', 'N/A')}\n"
        info_text += f"Photo Status: {record.get('Photo_Status', 'N/A')}\n"
        
        info_text += f"\nCOMPARTMENTS FOUND\n"
        info_text += "-" * 20 + "\n"
        info_text += f"Count: {len(corners_data)}\n\n"
        
        for i, compartment in enumerate(corners_data):
            info_text += f"Compartment {i+1}:\n"
            if 'corners' in compartment:
                corners = compartment['corners']
                for j, corner in enumerate(corners):
                    corner_names = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left']
                    info_text += f"  {corner_names[j]}: ({corner[0]:.1f}, {corner[1]:.1f})\n"
            info_text += "\n"
        
        # Find related compartment images
        related_images = self.find_related_compartment_images()
        if related_images:
            info_text += f"RELATED COMPARTMENT IMAGES\n"
            info_text += "-" * 30 + "\n"
            for img in related_images[:10]:  # Show first 10
                info_text += f"  {img.name}\n"
            if len(related_images) > 10:
                info_text += f"  ... and {len(related_images) - 10} more\n"
        
        self.info_text.insert(1.0, info_text)
    
    def find_related_compartment_images(self) -> List[Path]:
        """Find compartment images related to the current original image."""
        if not self.current_image_data or not self.compartment_images_folder:
            return []
        
        record = self.current_image_data['register_record']
        hole_id = record.get('HoleID', '')
        original_filename = record.get('Original_Filename', '')
        
        # Search for compartment images
        related_images = []
        
        # Common patterns for compartment image naming
        search_patterns = [
            f"*{hole_id}*",
            f"*{Path(original_filename).stem}*"
        ]
        
        for pattern in search_patterns:
            for img_path in self.compartment_images_folder.rglob(pattern):
                if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}:
                    if img_path not in related_images:
                        related_images.append(img_path)
        
        return sorted(related_images)
    
    def on_image_selected(self, event):
        """Handle image selection from dropdown."""
        selection = self.image_combo.current()
        if selection >= 0:
            self.current_index = selection
            self.load_current_image()
    
    def previous_image(self):
        """Navigate to previous image."""
        if self.image_list and self.current_index > 0:
            self.current_index -= 1
            self.image_combo.current(self.current_index)
            self.load_current_image()
    
    def next_image(self):
        """Navigate to next image."""
        if self.image_list and self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.image_combo.current(self.current_index)
            self.load_current_image()
    
    def fit_image(self):
        """Fit image to canvas."""
        self.image_viewer.fit_to_canvas()
    
    def actual_size(self):
        """Show image at actual size."""
        self.image_viewer.scale = 1.0
        self.image_viewer.offset_x = 0
        self.image_viewer.offset_y = 0
        self.image_viewer.update_display()
    
    def toggle_overlays(self):
        """Toggle overlay display."""
        if self.current_image_data:
            if self.show_overlays_var.get():
                self.update_overlays()
            else:
                # Show image without overlays - remove the overlay attribute
                if hasattr(self.image_viewer, 'image_with_overlay'):
                    del self.image_viewer.image_with_overlay
                self.image_viewer.update_display()
    
    def run(self):
        """Run the application."""
        self.root.mainloop()


def main():
    """Main entry point."""
    app = ValidationTool()
    app.run()


if __name__ == "__main__":
    main()