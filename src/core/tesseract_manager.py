# core/tesseract_manager.py

"""
    Manages Tesseract OCR detection, installation guidance, and OCR functionality.
    Designed to gracefully handle cases where Tesseract is not installed.
"""
    

import os
import platform
import re
import math
import importlib.util
import webbrowser
import traceback
import numpy as np
import cv2
from typing import Dict, Optional, Any
import logging
logger = logging.getLogger(__name__)
import tkinter as tk
from tkinter import ttk
import queue
from gui.dialog_helper import DialogHelper



class TesseractManager:

    def __init__(self):
        """Initialize the Tesseract manager with default paths and settings."""
        self.is_available = False
        self.version = None
        self.pytesseract = None
        self.install_instructions = {
            'Windows': 'https://github.com/UB-Mannheim/tesseract/wiki',
            }
        
        # Store last successful metadata for auto-filling
        self.last_successful_metadata = {
            'hole_id': None,
            'depth_from': None,
            'depth_to': None,
            'compartment_interval': 1
        }
        
        # Try to detect Tesseract
        self.detect_tesseract()

    def get_incremented_metadata(self):
        """
        Generate metadata based on the last successfully processed tray.
        
        Returns:
            dict: New metadata with incremented depths
        """
        if not self.last_successful_metadata.get('hole_id'):
            # No previous metadata available
            return {
                'hole_id': None,
                'depth_from': None,
                'depth_to': None,
                'compartment_interval': 1
            }
        
        # Use last hole ID
        hole_id = self.last_successful_metadata.get('hole_id')
        
        # Last depth_to becomes new depth_from
        depth_from = self.last_successful_metadata.get('depth_to')
        
        # Get interval (default to 1 if not available)
        interval = self.last_successful_metadata.get('compartment_interval', 1)
        
        # Calculate new depth_to (add 20 × interval)
        depth_to = depth_from + (20 * interval) if depth_from is not None else None
        
        return {
            'hole_id': hole_id,
            'depth_from': depth_from,
            'depth_to': depth_to,
            'compartment_interval': interval
        }
    
    def detect_tesseract(self) -> bool:
        """
        Detect if Tesseract OCR is installed and available in the system.
        
        Returns:
            bool: True if Tesseract is available, False otherwise
        """
        logger.info("Tesseract OCR detection disabled for reduced executable size")
        self.is_available = False
        return False
        
        # First check if pytesseract can be imported
        try:
            # Try to import pytesseract
            if importlib.util.find_spec("pytesseract") is None:
                logger.warning("pytesseract package not found")
                return False
            
            # Import pytesseract for OCR functionality
            import pytesseract
            from pytesseract import Output
            self.pytesseract = pytesseract
            
            # Try various methods to find Tesseract executable
            if platform.system() == 'Windows':
                # Common installation locations on Windows
                possible_paths = [
                    r'C:\Program Files\Tesseract-OCR',
                    r'C:\Program Files (x86)\Tesseract-OCR',
                    os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'Tesseract-OCR'),
                    os.path.join(os.environ.get('APPDATA', ''), 'Tesseract-OCR')
                ]
                
                # Find the first valid path
                tesseract_path = None
                for path in possible_paths:
                    exe_path = os.path.join(path, 'tesseract.exe')
                    if os.path.exists(exe_path):
                        tesseract_path = path
                        break
                
                if tesseract_path:
                    # Add to system PATH if it exists
                    os.environ['PATH'] = os.environ['PATH'] + os.pathsep + tesseract_path
                    pytesseract.pytesseract.tesseract_cmd = os.path.join(tesseract_path, 'tesseract.exe')
                    logger.info(f"Found Tesseract at: {tesseract_path}")
            
            # Test if Tesseract works by getting version
            try:
                self.version = pytesseract.get_tesseract_version()
                logger.info(f"Tesseract OCR version {self.version} detected")
                self.is_available = True
                return True
            except Exception as e:
                logger.warning(f"Tesseract is installed but not working correctly: {str(e)}")
                return False
                
        except ImportError:
            logger.warning("Failed to import pytesseract module")
            return False
        except Exception as e:
            logger.warning(f"Error detecting Tesseract: {str(e)}")
            return False
    
    def get_installation_instructions(self) -> str:
        """
        Get installation instructions for the current platform.
        
        Returns:
            str: Installation instructions for Tesseract OCR
        """
        system = platform.system()
        base_instructions = (
            "To use OCR features, you need to:\n\n"
            "1. Install Tesseract OCR for your platform\n"
            "2. Install the pytesseract Python package: pip install pytesseract\n\n"
        )
        
        if system in self.install_instructions:
            platform_instructions = f"For {system} systems: {self.install_instructions[system]}"
        else:
            platform_instructions = "Visit https://github.com/tesseract-ocr/tesseract for installation instructions."
        
        return base_instructions + platform_instructions
    
    def show_installation_dialog(self, parent: Optional[tk.Tk] = None) -> None:
        """
        Show a dialog with installation instructions for Tesseract OCR.
        
        Args:
            parent: Optional parent Tkinter window
        """
        instructions = self.get_installation_instructions()
        system = platform.system()
        
        # If no parent window, just log the instructions
        if parent is None:
            logger.info(DialogHelper.t(f"Tesseract OCR not available. {instructions}"))
            return
        
        # Create a custom dialog
        dialog = tk.Toplevel(parent)
        dialog.title(DialogHelper.t("Tesseract OCR Required"))
        dialog.grab_set()  # Make dialog modal
        
        # Add icon and header
        header_frame = ttk.Frame(dialog, padding="10")
        header_frame.pack(fill=tk.X)
        
        # Use system-specific icons
        if system == 'Windows':
            try:
                dialog.iconbitmap(default='')  # Default Windows icon
            except:
                pass
                
        # Header label
        header_label = ttk.Label(
            header_frame, 
            text=DialogHelper.t("Tesseract OCR Required for Text Recognition"),
            font=("Arial", 12, "bold")
        )
        header_label.pack(pady=10)
        
        # Instructions text
        text_frame = ttk.Frame(dialog, padding="10")
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        instructions_text = tk.Text(text_frame, wrap=tk.WORD, height=10)
        instructions_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        instructions_text.insert(tk.END, instructions)
        instructions_text.config(state=tk.DISABLED)
        
        # Scrollbar for text
        scrollbar = ttk.Scrollbar(instructions_text, command=instructions_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        instructions_text.config(yscrollcommand=scrollbar.set)
        
        # Action buttons
        button_frame = ttk.Frame(dialog, padding="10")
        button_frame.pack(fill=tk.X)
        
        # Platform-specific install button
        if system in self.install_instructions:
            install_url = self.install_instructions[system].split(' ')[0]
            install_button = ttk.Button(
                button_frame, 
                text=DialogHelper.t(f"Download for {system}"), 
                command=lambda: webbrowser.open(install_url)
            )
            install_button.pack(side=tk.LEFT, padx=5)
        
        close_button = ttk.Button(
            button_frame, 
            text=DialogHelper.t("Continue without OCR"), 
            command=dialog.destroy
        )
        close_button.pack(side=tk.RIGHT, padx=5)
    
    # Function to extract patterns from OCR
    def extract_metadata_with_composite(self, image, markers, original_filename=None, progress_queue=None):
        """
        Extract metadata using a composite approach that combines marker-based and ROI-based methods.
        Return early with minimal metadata structure if OCR is disabled.
        
        Args:
            image: Input image
            markers: Detected ArUco markers
            original_filename: Original image filename
            progress_queue: Optional progress queue
            
        Returns:
            Dictionary with extracted metadata
        """
        # Initialize result structure with default values
        metadata = {
            'hole_id': None,
            'depth_from': None,
            'depth_to': None,
            'confidence': 0.0,
            'metadata_region': None,
            'metadata_region_viz': None
        }
        
        # Return early with minimal structure if OCR is disabled but still allow 
        # for manual entry and validation
        if not self.config.get('enable_ocr', True):
            self.logger.info("OCR is disabled, returning minimal metadata structure")
            # Extract filename metadata if available to pre-populate the dialog
            if original_filename:
                filename_metadata = self.file_manager.extract_metadata_from_filename(original_filename)
                if filename_metadata:
                    metadata.update(filename_metadata)
                    metadata['confidence'] = 100.0  # High confidence for filename metadata
                    metadata['from_filename'] = True
            return metadata
        
        try:
            # Get image dimensions
            h, w = image.shape[:2]
            
            # Check if we have necessary markers
            if not markers:
                logger.warning("No markers found for metadata region extraction")
                return {
                    'hole_id': None,
                    'depth_from': None,
                    'depth_to': None,
                    'confidence': 0.0,
                    'metadata_region': None
                }
            
            # Define metadata region using markers
            metadata_region = None  # Initialize the variable
            hole_id_region = None
            depth_region = None
            
            # Look for marker ID 24 which is between hole ID and depth labels
            if 24 in markers:
                # Use marker 24 as a reference point
                marker24_corners = markers[24]
                marker24_center_x = int(np.mean(marker24_corners[:, 0]))
                marker24_center_y = int(np.mean(marker24_corners[:, 1]))
                
                # The hole ID label is likely above marker 24
                hole_id_region_y1 = max(0, marker24_center_y - 130)  # 130px above marker center
                hole_id_region_y2 = max(0, marker24_center_y - 20)   # 20px above marker center
                
                # The depth label is likely below marker 24
                depth_region_y1 = min(h, marker24_center_y + 20)    # 20px below marker center
                depth_region_y2 = min(h, marker24_center_y + 150)   # 150px below marker center
                
                # Horizontal range for both regions (centered on marker 24)
                # Use smaller width (100px instead of 130px)
                region_x1 = max(0, marker24_center_x - 100)  
                region_x2 = min(w, marker24_center_x + 100)
                
                # Extract the hole ID and depth regions
                hole_id_region = image[hole_id_region_y1:hole_id_region_y2, region_x1:region_x2].copy()
                depth_region = image[depth_region_y1:depth_region_y2, region_x1:region_x2].copy()
                
                # Create visualization
                viz_image = image.copy()
                # Draw hole ID region in blue
                cv2.rectangle(viz_image, 
                            (region_x1, hole_id_region_y1), 
                            (region_x2, hole_id_region_y2), 
                            (255, 0, 0), 2)
                # Draw marker 24 in purple
                cv2.circle(viz_image, (marker24_center_x, marker24_center_y), 10, (255, 0, 255), -1)
                # Draw depth region in green
                cv2.rectangle(viz_image, 
                            (region_x1, depth_region_y1), 
                            (region_x2, depth_region_y2), 
                            (0, 255, 0), 2)
                            
                # Create a metadata region by combining both regions
                metadata_region = image[hole_id_region_y1:depth_region_y2, region_x1:region_x2].copy()
                metadata_region_viz = viz_image
                
            else:
                # Simplified case for marker 24 not found - since this is handled elsewhere
                logger.warning("Marker 24 not found, skipping metadata extraction")
                return {
                    'hole_id': None,
                    'depth_from': None,
                    'depth_to': None,
                    'confidence': 0.0,
                    'metadata_region': None
                }
            
            # ALWAYS save visualization to memory cache regardless of debug settings
            # This is crucial for duplicate detection dialog
            if hasattr(self, 'extractor') and hasattr(self.extractor, 'visualization_cache'):
                if 'current_processing' not in self.extractor.visualization_cache:
                    self.extractor.visualization_cache['current_processing'] = {}

                self.extractor.visualization_cache['current_processing'].update({
                    'metadata_region': metadata_region,
                    'metadata_region_viz': metadata_region_viz,
                    'hole_id_region': hole_id_region,
                    'depth_region': depth_region
                })

            
            # Process hole ID and depth regions separately
            hole_id_results = []
            depth_results = []
            
            if hole_id_region is not None and depth_region is not None:
                # Create preprocessing methods specifically tailored for label text
                preprocessing_methods = [
                    ("original", lambda img: img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
                    ("threshold1", lambda img: cv2.threshold(
                        img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 
                        127, 255, cv2.THRESH_BINARY)[1]),
                    ("threshold2", lambda img: cv2.threshold(
                        img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 
                        100, 255, cv2.THRESH_BINARY)[1]),
                    ("threshold3", lambda img: cv2.threshold(
                        img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 
                        150, 255, cv2.THRESH_BINARY)[1]),
                    ("adaptive", lambda img: cv2.adaptiveThreshold(
                        img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 
                        255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
                    ("otsu", lambda img: cv2.threshold(
                        img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 
                        0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
                    ("canny", lambda img: cv2.Canny(
                        img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 
                        50, 150))
                ]
                
                # Create composite images for hole ID and depth
                hole_id_composite_images = []
                depth_composite_images = []
                
                # Process hole ID region
                hole_id_confidence = 0
                partial_hole_id_prefix = None
                
                # Process all methods
                for method_name, method_func in preprocessing_methods:
                    try:
                        # Process image
                        processed_img = method_func(hole_id_region)
                        
                        # Add progress reporting
                        if progress_queue:
                            progress_queue.put((f"OCR processing method: {method_name}", None))
                        
                        # Run OCR specifically for hole ID pattern - try all PSM modes
                        for psm in [7, 8, 6]:  # Include PSM 6 like in older code
                            config = f"--psm {psm} --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                            text = self.pytesseract.image_to_string(processed_img, config=config).strip()
                            
                            # Look for complete hole ID pattern
                            match = re.search(r'([A-Z]{2}\d{4})', text.replace(" ", ""))
                            hole_id = match.group(1) if match else None
                            
                            # If we found a complete hole ID
                            if hole_id:
                                hole_id_results.append({
                                    'hole_id': hole_id,
                                    'method': method_name,
                                    'psm': psm,
                                    'text': text,
                                    'processed_image': processed_img,
                                    'confidence': 100.0  # Full match gets high confidence
                                })
                                
                                # Add image to composite with text overlay
                                labeled_img = processed_img.copy()
                                if len(labeled_img.shape) == 2:
                                    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_GRAY2BGR)
                                
                                # Add text overlay
                                cv2.putText(
                                    labeled_img,
                                    f"{method_name} PSM{psm}: {hole_id}",
                                    (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 0, 255),
                                    2
                                )
                                
                                hole_id_composite_images.append(labeled_img)
                                hole_id_confidence = 100.0
                                break
                            
                            # If complete hole ID not found, look for just the project code (first two letters)
                            if not hole_id:
                                prefix_match = re.search(r'([A-Z]{2})', text.replace(" ", ""))
                                if prefix_match and hasattr(self, 'config'):
                                    potential_prefix = prefix_match.group(1).upper()
                                    valid_prefixes = self.config.get('valid_hole_prefixes', [])
                                    
                                    # Check if this prefix is valid
                                    if valid_prefixes and potential_prefix in valid_prefixes:
                                        logger.info(f"Found valid project code: {potential_prefix}")
                                        if partial_hole_id_prefix is None or hole_id_confidence < 50.0:
                                            partial_hole_id_prefix = potential_prefix
                                            hole_id_confidence = 50.0  # Partial match gets medium confidence
                                            
                                            # Record this as a partial result - just store the prefix
                                            hole_id_results.append({
                                                'hole_id': potential_prefix,  # Store only the prefix, no placeholders
                                                'method': method_name,
                                                'psm': psm,
                                                'text': text,
                                                'processed_image': processed_img,
                                                'confidence': 50.0,
                                                'partial': True
                                            })
                                            
                                            # Add image to composite
                                            labeled_img = processed_img.copy()
                                            if len(labeled_img.shape) == 2:
                                                labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_GRAY2BGR)
                                            
                                            cv2.putText(
                                                labeled_img,
                                                f"{method_name} PSM{psm}: {potential_prefix} (Partial)",
                                                (10, 30),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.7,
                                                (0, 255, 255),  # Yellow for partial match
                                                2
                                            )
                                            
                                            hole_id_composite_images.append(labeled_img)
                                            
                    except Exception as e:
                        logger.error(f"Error processing hole ID with {method_name}: {str(e)}")
                
                # Process depth region - try all methods
                for method_name, method_func in preprocessing_methods:
                    try:
                        # Process image
                        processed_img = method_func(depth_region)
                        
                        # Add progress reporting
                        if progress_queue:
                            progress_queue.put((f"OCR processing depth method: {method_name}", None))
                        
                        # Run OCR for depth pattern
                        for psm in [7, 6, 3]:  # Include all useful PSM modes
                            config = f"--psm {psm} --oem 3 -c tessedit_char_whitelist=0123456789-."
                            text = self.pytesseract.image_to_string(processed_img, config=config).strip()
                            
                            # Try to extract depth pattern - more flexible regex
                            match = re.search(r'(\d+)[\s\-–—]*(?:to)?[\s\-–—]*(\d+)', text)
                            if match:
                                depth_from = int(match.group(1))
                                depth_to = int(match.group(2))
                                
                                # Validate depth values - ensure they're non-negative
                                if depth_from >= 0 and depth_to >= 0 and depth_to > depth_from:
                                    depth_results.append({
                                        'depth_from': depth_from,
                                        'depth_to': depth_to,
                                        'method': method_name,
                                        'psm': psm,
                                        'text': text,
                                        'processed_image': processed_img,
                                        'confidence': 100.0
                                    })
                                    
                                    # Add image to composite with text overlay
                                    labeled_img = processed_img.copy()
                                    if len(labeled_img.shape) == 2:
                                        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_GRAY2BGR)
                                    
                                    # Add text overlay
                                    cv2.putText(
                                        labeled_img,
                                        f"{method_name} PSM{psm}: {depth_from}-{depth_to}",
                                        (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7,
                                        (0, 0, 255),
                                        2
                                    )
                                    
                                    depth_composite_images.append(labeled_img)
                                    break
                        
                    except Exception as e:
                        logger.error(f"Error processing depth with {method_name}: {str(e)}")
                
                # Create and save composite images
                if hole_id_composite_images and hasattr(self, 'config') and self.config.get('save_debug_images', False):
                    file_manager = getattr(self, 'file_manager', None)
                    if file_manager is not None:
                        # Create a grid layout for hole ID images
                        grid_size = math.ceil(math.sqrt(len(hole_id_composite_images)))
                        grid_h = grid_size
                        grid_w = grid_size
                        
                        # Find max dimensions
                        max_h = max(img.shape[0] for img in hole_id_composite_images)
                        max_w = max(img.shape[1] for img in hole_id_composite_images)
                        
                        # Create grid
                        grid = np.ones((max_h * grid_h, max_w * grid_w, 3), dtype=np.uint8) * 255
                        
                        # Place images
                        for i, img in enumerate(hole_id_composite_images):
                            if i >= grid_h * grid_w:
                                break
                                
                            row = i // grid_w
                            col = i % grid_w
                            
                            y = row * max_h
                            x = col * max_w
                            
                            h, w = img.shape[:2]
                            grid[y:y+h, x:x+w] = img
                        
                        # Save composite
                        file_manager.save_temp_debug_image(
                            grid,
                            original_filename,
                            "hole_id_composite"
                        )
                
                # Same for depth composite
                if depth_composite_images and hasattr(self, 'config') and self.config.get('save_debug_images', False):
                    file_manager = getattr(self, 'file_manager', None)
                    if file_manager is not None:
                        # Create a grid layout for depth images
                        grid_size = math.ceil(math.sqrt(len(depth_composite_images)))
                        grid_h = grid_size
                        grid_w = grid_size
                        
                        # Find max dimensions
                        max_h = max(img.shape[0] for img in depth_composite_images)
                        max_w = max(img.shape[1] for img in depth_composite_images)
                        
                        # Create grid
                        grid = np.ones((max_h * grid_h, max_w * grid_w, 3), dtype=np.uint8) * 255
                        
                        # Place images
                        for i, img in enumerate(depth_composite_images):
                            if i >= grid_h * grid_w:
                                break
                                
                            row = i // grid_w
                            col = i % grid_w
                            
                            y = row * max_h
                            x = col * max_w
                            
                            h, w = img.shape[:2]
                            grid[y:y+h, x:x+w] = img
                        
                        # Save composite
                        file_manager.save_temp_debug_image(
                            grid,
                            original_filename,
                            "depth_composite"
                        )
            
            # Vote on results
            # For hole ID
            hole_id_votes = {}
            for result in hole_id_results:
                hole_id = result.get('hole_id')
                if hole_id:
                    hole_id_votes[hole_id] = hole_id_votes.get(hole_id, 0) + 1
            
            # For depths
            depth_from_votes = {}
            depth_to_votes = {}
            for result in depth_results:
                depth_from = result.get('depth_from')
                depth_to = result.get('depth_to')
                
                # Only consider non-negative values
                if depth_from is not None and depth_from >= 0:
                    # Round to nearest integer for voting
                    depth_from_int = int(round(depth_from))
                    depth_from_votes[depth_from_int] = depth_from_votes.get(depth_from_int, 0) + 1
                
                if depth_to is not None and depth_to >= 0:
                    # Round to nearest integer for voting
                    depth_to_int = int(round(depth_to))
                    depth_to_votes[depth_to_int] = depth_to_votes.get(depth_to_int, 0) + 1
            
            # Get the most voted values
            final_hole_id = None
            if hole_id_votes:
                final_hole_id = max(hole_id_votes.items(), key=lambda x: x[1])[0]
            
            final_depth_from = None
            if depth_from_votes:
                final_depth_from = max(depth_from_votes.items(), key=lambda x: x[1])[0]
            
            final_depth_to = None
            if depth_to_votes:
                final_depth_to = max(depth_to_votes.items(), key=lambda x: x[1])[0]
            
            # Get the compartment interval from config (default to 1 if not available)
            compartment_interval = 1
            if hasattr(self, 'config'):
                compartment_interval = self.config.get('compartment_interval', 1)
            
            # Calculate expected depth range (20 compartments × interval)
            expected_range = 20 * compartment_interval
            
            # Additional validation for depths
            if final_depth_from is not None:
                # Ensure non-negative
                if final_depth_from < 0:
                    final_depth_from = 0
                    logger.warning("Negative depth_from detected, setting to 0")
                
                # Round to nearest multiple of the compartment interval
                final_depth_from = round(final_depth_from / compartment_interval) * compartment_interval
                logger.info(f"Adjusted depth_from to align with {compartment_interval}m interval: {final_depth_from}")
            
            if final_depth_to is not None:
                # Ensure non-negative
                if final_depth_to < 0:
                    final_depth_to = expected_range  # Default to one full tray depth
                    logger.warning(f"Negative depth_to detected, setting to {final_depth_to}")
                
                # Round to nearest multiple of the compartment interval
                final_depth_to = round(final_depth_to / compartment_interval) * compartment_interval
                logger.info(f"Adjusted depth_to to align with {compartment_interval}m interval: {final_depth_to}")
            
            # Make sure depth_to > depth_from and they maintain the proper interval difference
            if final_depth_from is not None and final_depth_to is not None:
                if final_depth_to <= final_depth_from:
                    # Set depth_to to be depth_from + expected_range
                    final_depth_to = final_depth_from + expected_range
                    logger.info(f"Corrected depth range to: {final_depth_from}-{final_depth_to}")
                
                # Ensure the difference matches the expected range (20 × interval)
                diff = final_depth_to - final_depth_from
                if diff != expected_range:
                    # Adjust depth_to to maintain proper interval
                    final_depth_to = final_depth_from + expected_range
                    logger.info(f"Adjusted depth range to maintain expected {expected_range}m interval: {final_depth_from}-{final_depth_to}")
            
            # Calculate confidence based on vote consistency
            confidence = 0.0
            
            # Hole ID confidence
            if final_hole_id and hole_id_votes:
                # Percentage of agreement
                agreement = min(100.0, hole_id_votes[final_hole_id] / len(hole_id_results) * 100 if hole_id_results else 0)
                confidence += min(agreement, 50.0)  # Max 50 points from hole ID

            # Depth confidence
            if final_depth_from is not None and final_depth_to is not None and depth_from_votes and depth_to_votes:
                # Percentage of agreement for each value
                # Make sure the keys exist in the dictionaries
                from_agreement = min(100.0, depth_from_votes.get(final_depth_from, 0) / len(depth_results) * 100 if depth_results else 0)
                to_agreement = min(100.0, depth_to_votes.get(final_depth_to, 0) / len(depth_results) * 100 if depth_results else 0)
                
                # Average the agreements
                avg_agreement = (from_agreement + to_agreement) / 2
                confidence += min(avg_agreement, 50.0)  # Max 50 points from depths

            # Ensure final confidence never exceeds 100%
            confidence = min(confidence, 100.0)
            
            # Flag indicating if we have a partial hole ID (just project code)
            is_partial = any(r.get('partial', False) for r in hole_id_results if r.get('hole_id') == final_hole_id)
            
            # Final result
            result = {
                'hole_id': final_hole_id,
                'depth_from': final_depth_from,
                'depth_to': final_depth_to,
                'confidence': confidence,
                'metadata_region': metadata_region,
                'metadata_region_viz': metadata_region_viz,
                'ocr_text': f"Composite OCR: ID={final_hole_id}, Depth={final_depth_from}-{final_depth_to}",
                'partial_hole_id': is_partial,
                'compartment_interval': compartment_interval  # Include the interval in result
            }
            
            if (
                hasattr(self, 'extractor') and 
                hasattr(self.extractor, 'visualization_cache') and
                'current_processing' in self.extractor.visualization_cache
            ):
                boundaries_viz = self.extractor.visualization_cache['current_processing'].get('compartment_boundaries_viz')
                if boundaries_viz is not None:
                    result['compartment_boundaries_viz'] = boundaries_viz

            # Log final results
            logger.info("------------------------------------------------")
            logger.info("OCR results:")
            if final_hole_id:
                logger.info(f"Hole ID: {final_hole_id} (votes: {hole_id_votes.get(final_hole_id, 0)}/{len(hole_id_results) or 1})")
                if is_partial:
                    logger.info("Partial match - only project code detected")
            else:
                logger.info("Hole ID: None")
                
            if final_depth_from is not None and final_depth_to is not None:
                logger.info(f"Depth range: {final_depth_from}-{final_depth_to} "
                        f"(votes: {depth_from_votes.get(final_depth_from, 0)}/{len(depth_results) or 1}, "
                        f"{depth_to_votes.get(final_depth_to, 0)}/{len(depth_results) or 1})")
            else:
                logger.info("Depth range: None")
                
            logger.info(f"Confidence: {confidence:.1f}%")
            logger.info("------------------------------------------------")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in composite OCR extraction: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'hole_id': None,
                'depth_from': None,
                'depth_to': None,
                'confidence': 0.0,
                'metadata_region': None
            }