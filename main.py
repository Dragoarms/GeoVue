# src\main.py

"""
This is a work-in-progress tool for extracting individual compartment images from chip tray panoramas. It uses ArUco markers to detect compartment boundaries and can optionally pull metadata via Tesseract OCR if labels are visible in the photo.

Current Features:
Simple folder-based interface — no need to set things up manually

Uses ArUco markers for alignment and compartment detection

Optional OCR to pull metadata straight from the tray

Keeps image quality as high as possible during processing

Shows visual debug output to help troubleshoot detection issues

Automatically names and organizes outputs

Supports common image formats like JPG and PNG

Basic error handling and logging

QAQC step to review extracted compartments

Can skip or process specific images based on filters

Auto-generates an Excel register of processed trays

Supports multiple languages (UI - not OCR)

Checks for script updates automatically

Some advanced config options for users who want more control

Status:
Still under development — some things are a bit rough around the edges and may change depending on what works best in the field...
Happy to hear suggestions or bug reports if you're trying it.

"""

# ===========================================
# main.py - GeoVue
# ===========================================

# Import Standard Libraries
import sys
import logging
import os
import json
import tkinter as tk
from tkinter import messagebox
import platform
import argparse
import queue
from typing import Tuple
import traceback
import cv2
import numpy as np
import threading
import traceback
import re
import shutil
import pandas as pd
from datetime import datetime
import time

# Version detection
try:
    if sys.version_info >= (3, 11):
        import tomllib  # built-in in Python 3.11+
    else:
        import tomli as tomllib
except ImportError:
    tomllib = None
    print("⚠️ tomli not installed — version fallback to unknown.")

def get_version_from_pyproject():
    try:
        pyproject_path = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)
        return pyproject_data["project"]["version"]
    except Exception as e:
        print(f"Error reading pyproject.toml: {e}")
        return "unknown"

__version__ = get_version_from_pyproject()

# Logging Configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - Line %(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M'
)

# Create a logger instance and explicitly set its level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Explicitly set this logger to DEBUG
logger.info(f"Starting GeoVue v{__version__}")


# ===================================================
# Module-specific logging levels
# ===================================================
# Turn off debug logs for noisy modules
logging.getLogger('processing.aruco_manager').setLevel(logging.INFO)  # Only INFO and above
logging.getLogger('processing.blur_detector').setLevel(logging.WARNING)  # Only WARNING and above
logging.getLogger('core.tesseract_manager').setLevel(logging.INFO)

# Keep debug for modules you're actively debugging
logging.getLogger('gui.duplicate_handler').setLevel(logging.DEBUG)
logging.getLogger('gui.qaqc_manager').setLevel(logging.DEBUG)
logging.getLogger('gui.compartment_registration_dialog').setLevel(logging.INFO)

# ===========================================
# Module Imports
# ===========================================

# Import package modules
from core import (
    FileManager, 
    TranslationManager, 
    RepoUpdater, 
    TesseractManager, 
    ConfigManager  
)

from gui import (
    DialogHelper,
    GUIManager,
    DuplicateHandler,
    CompartmentRegistrationDialog, 
    MainGUI,
    QAQCManager
)

from gui.widgets import *

from processing import (
    BlurDetector,
    DrillholeTraceGenerator,
    ArucoManager
)

from utils import (
    OneDrivePathManager
)

# ===========================================
# Main Application Class
# ===========================================

class ChipTrayApp:
    """Main application class that orchestrates the chip tray processing pipeline."""
    
    def __init__(self):
        """Initialize the application with all necessary components."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing GeoVue")
        
        # ===================================================
        # STEP 1: INITIALIZE CONFIGURATION
        # ===================================================
        # Determine default config path
        if getattr(sys, 'frozen', False):
            # Running as compiled executable
            base_path = sys._MEIPASS
        else:
            # Running as script
            base_path = os.path.dirname(os.path.abspath(__file__))
        
        default_config_path = os.path.join(base_path, "config.json")
        
        # Initialize config manager
        self.config_manager = ConfigManager(default_config_path)
        # Get config as dictionary for easy access
        self.config = self.config_manager.as_dict()

        # ===================================================
        # STEP 2: INITIALIZE NON-GUI CORE COMPONENTS
        # ===================================================
        # Create a single progress queue for the entire app
        self.progress_queue = queue.Queue()
        
        # Initialize visualization cache
        self.visualization_cache = {}
        
        # Initialize last successful metadata
        self.last_successful_metadata = {
            'hole_id': None,
            'depth_from': None,
            'depth_to': None,
            'compartment_interval': 1
        }
        
        # Initialize empty metadata for OCR
        self.metadata = {}
        
        # Initialize FileManager with config_manager
        output_dir = self.config_manager.get('local_folder_path')
        self.file_manager = FileManager(base_dir=output_dir, config_manager=self.config_manager)

        # Initialize translation system early
        script_dir = os.path.dirname(os.path.abspath(__file__))
        translations_path = os.path.join(script_dir, "resources", "translations.csv")
        self.translator = TranslationManager(
            file_manager=self.file_manager,
            config=self.config_manager.as_dict(),
            csv_path=translations_path
        )
        self.t = self.translator.translate  # Shorthand for translation function
        
        # Set translator in DialogHelper early to avoid NoneType errors
        DialogHelper.set_translator(self.translator)

        
        # ===================================================
        # STEP 3: CREATE ROOT WINDOW
        # ===================================================
        # Create the root window (needed for all GUI components)
        self.root = tk.Tk()
        # Don't withdraw yet - we might need it for first run dialog
        self.root.title("GeoVue")
        
        # ===================================================
        # STEP 4: HANDLE FIRST RUN SETUP
        # ===================================================
        # Check if first run
        if self.config_manager.is_first_run() or not self._check_configuration():
            # Initialize minimal GUI manager for dialog
            self.gui_manager = GUIManager(self.file_manager, self.config_manager)
            DialogHelper.set_gui_manager(self.gui_manager)
            
            # Show first run dialog
            from gui.first_run_dialog import FirstRunDialog
            dialog = FirstRunDialog(self.root, self.gui_manager)
            result = dialog.show()
            
            if not result:
                # User cancelled
                self.logger.info("User cancelled first run setup")
                self.root.destroy()
                sys.exit(0)
            
            # Save configuration from setup
            self.config_manager.set('storage_type', result['storage_type'])
            self.config_manager.set('local_folder_path', result['local_folder_path'])
            if result.get('shared_folder_path'):
                self.config_manager.set('shared_folder_path', result['shared_folder_path'])
            
            # Save all folder paths
            for key, value in result['folder_paths'].items():
                if value:  # Only save non-None values
                    self.config_manager.set(key, value)
            
            # Mark as initialized
            self.config_manager.mark_initialized()
            
            # Update config reference
            self.config = self.config_manager.as_dict()

            # Get output directory from config
            output_dir = self.config_manager.get('local_folder_path') or self.config_manager.get('output_directory')

            # Re-initialize FileManager with new path
            if output_dir and output_dir != self.file_manager.base_dir:
                self.file_manager = FileManager(base_dir=output_dir)
        
        # ===================================================
        # NOW HIDE ROOT WINDOW UNTIL FULLY LOADED
        # ===================================================
        self.root.withdraw()  # Hide until everything is initialized
        
        # ===================================================
        # STEP 5: INITIALIZE GUI COMPONENTS
        # ===================================================
        # Initialize GUI manager (if not already created during first run)
        if not hasattr(self, 'gui_manager'):
            self.gui_manager = GUIManager(self.file_manager, self.config_manager)
            DialogHelper.set_gui_manager(self.gui_manager)
        
        # Initialize OneDrive manager in silent mode to avoid dialogs during startup
        self.onedrive_manager = OneDrivePathManager(self.root, silent=True)
        
        # Set application icon
        self._set_application_icon()
        
        # ===================================================
        # STEP 6: INITIALIZE PROCESSING COMPONENTS
        # ===================================================
        self._initialize_processing_components()
        
        # ===================================================
        # STEP 7: INITIALIZE UI COMPONENTS
        # ===================================================
        self._initialize_ui_components()
        
        # ===================================================
        # STEP 8: FINALIZE INITIALIZATION
        # ===================================================
        # Load saved language preference
        self._load_language_preference()
        
        # Complete initialization
        self.processing_complete = False
        
        self.logger.info("GeoVue initialization complete")




    def _set_application_icon(self):
        """Set the application icon for all windows."""
        try:
            # Try using the resources package helper function first
            try:
                from resources import get_logo_path
                logo_path = str(get_logo_path())
                self.logger.debug(f"Got logo path from resources: {logo_path}")
                
                if os.path.exists(logo_path):
                    self.logger.info(f"Found logo using resources package: {logo_path}")
                else:
                    # Path from resources package doesn't exist, will try alternatives
                    logo_path = None
                    self.logger.warning(f"Logo not found at path from resources: {logo_path}")
            except (ImportError, ModuleNotFoundError) as e:
                self.logger.warning(f"Couldn't import from resources: {e}")
                logo_path = None
            
            # If we couldn't get a valid path from the resources package, use fallback methods
            if not logo_path or not os.path.exists(logo_path):
                # Get logo path from config
                config_logo_path = self.config.get('logo_path')
                logo_found = False
                
                # Possible base directories to check
                base_dirs = [
                    os.path.dirname(os.path.abspath(__file__)),  # Script directory
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources"),  # resources subfolder
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # Parent of script directory
                    self.file_manager.base_dir,  # FileManager base directory
                    os.path.join(self.file_manager.base_dir, "Program Resources")  # Program Resources folder
                ]
                
                # Possible relative paths to check
                relative_paths = [
                    config_logo_path,  # Path from config
                    "logo.png",  # Default filename
                    "resources/logo.png",  # In resources subfolder
                    "resources\\logo.png",  # Windows path style
                ]
                
                # Log all paths we're checking
                self.logger.debug(f"Checking for logo in base dirs: {base_dirs}")
                self.logger.debug(f"Checking relative paths: {relative_paths}")
                
                # Try each combination of base dir and relative path
                for base_dir in base_dirs:
                    if not base_dir:
                        continue
                        
                    for rel_path in relative_paths:
                        if not rel_path:
                            continue
                            
                        # Construct full path
                        full_path = os.path.join(base_dir, rel_path)
                        self.logger.debug(f"Checking for logo at: {full_path}")
                        
                        if os.path.exists(full_path):
                            logo_path = full_path
                            logo_found = True
                            self.logger.info(f"Found logo at: {logo_path}")
                            break
                    
                    if logo_found:
                        break
            
            if not logo_path or not os.path.exists(logo_path):
                self.logger.warning("Logo file not found in any of the checked locations")
                return False
            
            # Use PIL to load the image
            from PIL import Image, ImageTk
            
            # Load the image and convert to PhotoImage
            logo_img = Image.open(logo_path)
            icon_image = ImageTk.PhotoImage(logo_img)
            
            # Set the icon for all windows (including future ones)
            self.root.iconphoto(True, icon_image)
            
            # Keep a reference to prevent garbage collection
            self._icon_image = icon_image
            
            # For Windows, also set the taskbar icon
            if platform.system() == 'Windows':
                try:
                    import tempfile
                    
                    # If the logo is not an ICO file, convert it
                    if not logo_path.lower().endswith('.ico'):
                        # Create a temporary ICO file
                        ico_path = os.path.join(tempfile.gettempdir(), "app_icon.ico")
                        
                        # Save a multi-resolution ICO file correctly
                        ico_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (256, 256)]
                        logo_img.save(ico_path, format='ICO', sizes=ico_sizes)
                        
                        # Set the icon for the window
                        self.root.iconbitmap(ico_path)
                        self.logger.info(f"Set Windows taskbar icon using converted ICO: {ico_path}")
                    else:
                        # Use ICO file directly
                        self.root.iconbitmap(logo_path)
                        self.logger.info(f"Set Windows taskbar icon using existing ICO: {logo_path}")
                except Exception as ico_err:
                    self.logger.warning(f"Could not set Windows taskbar icon: {ico_err}")
                    self.logger.warning(traceback.format_exc())
            
            self.logger.info("Successfully set application icon")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting application icon: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    
    def _initialize_core_components(self):
        """Initialize core system components."""

        # Get config path from the same directory as the main script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")
        translations_path = os.path.join(script_dir, "resources", "translations.csv")
        
        # Check if config exists, if not look in base_dir
        if not os.path.exists(config_path):
            # Fallback to default location in case it moved
            config_path = os.path.join(self.file_manager.base_dir, "Program Resources", "config.json")
            
        self.logger.info(f"Using config file at: {config_path}")
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.as_dict()
        
        # Initialize GUI manager for theming
        self.gui_manager = GUIManager(self.file_manager, self.config_manager)
        
        # Initialize translation system
        self.translator = TranslationManager(file_manager=self.file_manager,config=self.config,csv_path=translations_path)
        self.t = self.translator.translate  # Shorthand for translation function

        # For convenience, expose translation to DialogHelper
        DialogHelper.set_translator(self.translator)
        
        # Set GUI manager in DialogHelper for themed dialogs
        DialogHelper.set_gui_manager(self.gui_manager)
        
        # Initialize update checker
        self.update_checker = RepoUpdater()
        
        # Initialize visualization cache for sharing images between components
        self.visualization_cache = {}
        self.onedrive_manager = OneDrivePathManager(self.root)
    
    def _initialize_processing_components(self):
        """Initialize image processing components."""
        # OCR component
        self.tesseract_manager = TesseractManager()
        self.tesseract_manager.config = self.config
        self.tesseract_manager.file_manager = self.file_manager
        self.tesseract_manager.extractor = self  # Give it access to visualization_cache
        
        # Blur detection component
        self.blur_detector = BlurDetector(
            threshold=self.config.get('blur_threshold', 150.0),
            roi_ratio=self.config.get('blur_roi_ratio', 0.8)
        )
        
        # ArUco marker processing component - pass the app reference
        self.aruco_manager = ArucoManager(
            self.config, 
            self.progress_queue,
            app=self  # Pass app reference instead of pipeline
        )
        
        # Initialize drill trace generator
        self.trace_generator = DrillholeTraceGenerator(
            config=self.config,
            progress_queue=self.progress_queue,
            root=self.root,  # Pass the existing root
            file_manager=self.file_manager
        )

    def _initialize_ui_components(self):
        """Initialize user interface components."""
        # GUI management - already initialized in __init__
        # Just verify it exists
        if not hasattr(self, 'gui_manager'):
            self.gui_manager = GUIManager(self.file_manager, self.config_manager)
        
        # Initialize duplicate checker for GUI dialogs
        # Use the processed_originals directory from dir_structure
        processed_dir = self.file_manager.dir_structure.get("processed_originals", self.file_manager.base_dir)
        self.duplicate_handler = DuplicateHandler(processed_dir, onedrive_manager=self.onedrive_manager)
        self.duplicate_handler.parent = self  # Give it access to visualization_cache
        self.duplicate_handler.logger = logging.getLogger(__name__)
        self.duplicate_handler.root = self.root  # Set root for dialog creation
        
        # Initialize QAQC manager
        self.qaqc_manager = QAQCManager(self.root, self.file_manager, self)
        self.qaqc_manager.config = self.config

        # Main application GUI - pass the existing root window
        self.main_gui = MainGUI(self)
    
    def _perform_startup_version_check(self):
            """
            Perform version check during startup.
            Shows a non-blocking notification if update is available.
            """
            try:
                # ===================================================
                # SIMPLIFIED: No need to create translator - already initialized
                # ===================================================
                # Get version info
                version_info = self.update_checker.compare_versions()
                
                self.logger.info(f"Version check - Local: {version_info['local_version']}, "
                                f"GitHub: {version_info['github_version']}, "
                                f"Update available: {version_info['update_available']}")
                
                # Only show dialog if update is available
                if version_info['update_available'] and not version_info.get('error'):
                    # Create dialog using DialogHelper
                    dialog = DialogHelper.create_dialog(
                        self.root,
                        self.t("Update Available"),
                        modal=False,  # Non-blocking
                        topmost=True,
                        size_ratio=0.3,
                        min_width=400,
                        min_height=180
                    )
                    
                    # Create content frame
                    content_frame = tk.Frame(dialog, bg=self.gui_manager.theme_colors["background"])
                    content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
                    
                    # Construct the message
                    message = (
                        self.t('A new version is available!') + "\n\n" +
                        self.t('Current version') + f": {version_info['local_version']}\n" +
                        self.t('Latest version') + f": {version_info['github_version']}\n\n" +
                        self.t('Contact your administrator for the update.')
                    )
                    
                    # Message label
                    label = tk.Label(
                        content_frame,
                        text=message,
                        justify=tk.CENTER,
                        bg=self.gui_manager.theme_colors["background"],
                        fg=self.gui_manager.theme_colors["text"],
                        font=self.gui_manager.fonts["normal"]
                    )
                    label.pack(pady=(0, 20))
                    
                    # OK button
                    ok_btn = self.gui_manager.create_modern_button(
                        content_frame,
                        text=self.t("OK"),
                        color=self.gui_manager.theme_colors["accent_blue"],
                        command=dialog.destroy
                    )
                    ok_btn.pack()
                    
                    # Auto-close after 10 seconds
                    dialog.after(10000, lambda: dialog.destroy() if dialog.winfo_exists() else None)
                    
            except Exception as e:
                # Don't let version check errors prevent app startup
                self.logger.warning(f"Version check failed during startup: {str(e)}")

    def _load_language_preference(self):
        """Load language preference from config file."""
        try:
            config_path = os.path.join(self.file_manager.base_dir, "Program Resources", "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if 'language' in config:
                        self.translator.set_language(config['language'])
                        self.logger.info(f"Loaded language preference: {config['language']}")
        except Exception as e:
            self.logger.error(f"Error loading language preference: {str(e)}")
    
    def _save_language_preference(self, language_code):
        """Save language preference to config file."""
        try:
            config_dir = os.path.join(self.file_manager.base_dir, "Program Resources")
            os.makedirs(config_dir, exist_ok=True)
            config_path = os.path.join(config_dir, "config.json")
            
            # Load existing config if it exists
            config = {}
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
            
            # Update language setting
            config['language'] = language_code
            
            # Save updated config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            self.logger.info(f"Saved language preference: {language_code}")
        except Exception as e:
            self.logger.error(f"Error saving language preference: {str(e)}")

    def run_dialog_on_main_thread(self, dialog_class, args=None, kwargs=None):
        args = args or tuple()
        kwargs = kwargs or {}
        
        self.logger.info(f"Creating dialog {dialog_class.__name__} on main thread")
        
        try:
            dialog = dialog_class(*args, **kwargs)
            return dialog.show()
        except Exception as e:
            self.logger.error(f"Dialog error: {type(e).__name__}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}

    def clear_visualization_cache(self):
        self.visualization_cache["current_processing"] = {}
        self.visualization_cache["annotation_status"] = {"annotated_markers": []}
        self.aruco_manager.viz_steps.clear()
        self.logger.debug("Cleared visualization cache and viz_steps for new image")

    def process_image(self, image_path: str) -> bool:
        """
        Process a single chip tray image directly on the main thread.
        
        This method performs the complete image processing pipeline:
        1. Load and preprocess the image
        2. Detect ArUco markers
        3. Correct image orientation/skew
        4. Extract compartment boundaries
        5. ALWAYS show compartment registration dialog
        6. Extract individual compartments
        7. Add to QAQC queue
        8. Handle duplicates
        9. Move processed files
        
        Args:
            image_path: Path to the image file to be processed
        
        Returns:
            bool: True if processing succeeded, False otherwise
        """
        # ===================================================
        # STEP 0: INITIALIZATION AND SETUP
        # ===================================================
        # Initialize logging for this method
        self.logger = logging.getLogger(__name__)

        print(f"DEBUG: Starting to process image: {image_path}")
        print(f"DEBUG: Clearing Caches")

        # Initialize message collection for this image
        processing_messages = []
        def add_progress_message(message, progress=None):
            """Helper to collect messages during processing"""
            processing_messages.append((message, progress))
            print(f"DEBUG: Progress: {message}")
            self.logger.info(message)

        # Clear any cached visualization data from previous image
        self.clear_visualization_cache()

        # Initialize or get progress queue for UI updates
        if not hasattr(self, 'progress_queue'):
            self.progress_queue = queue.Queue()
            print("DEBUG: Created new progress_queue")
        else:
            print("DEBUG: Using existing progress_queue")

        # Ensure duplicate_handler is initialized for checking processed images
        if not hasattr(self, 'duplicate_handler') or self.duplicate_handler is None:
            print("DEBUG: Initializing duplicate_handler")
            self.duplicate_handler = DuplicateHandler(
                self.file_manager.dir_structure["processed_originals"],
                onedrive_manager=self.onedrive_manager
            )
            self.duplicate_handler.parent = self
            self.duplicate_handler.root = self.root
        else:
            print("DEBUG: Using existing duplicate_handler")

        try:
            # ===================================================
            # STEP 1: LOAD AND VALIDATE IMAGE
            # ===================================================
            # Try to read the image using PIL first (more format support)
            try:
                print("DEBUG: Attempting to read image with PIL")
                from PIL import Image as PILImage
                
                # Open with PIL
                pil_img = PILImage.open(image_path)
                print(f"DEBUG: PIL image mode: {pil_img.mode}, size: {pil_img.size}")
                
                # Convert to numpy array for OpenCV processing
                original_image = np.array(pil_img)
                print(f"DEBUG: Converted to numpy array, shape: {original_image.shape}")
                
                # Convert RGB to BGR for OpenCV compatibility if needed
                if len(original_image.shape) == 3 and original_image.shape[2] == 3:
                    print("DEBUG: Converting RGB to BGR for OpenCV compatibility")
                    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
                    
                self.logger.info(f"Successfully read image with PIL: {image_path}")
                
            except Exception as e:
                self.logger.warning(f"Failed to read image with PIL, trying OpenCV: {str(e)}")
                print(f"DEBUG: PIL reading failed with error: {str(e)}")
                
                # Fallback to OpenCV if PIL fails
                print("DEBUG: Attempting to read image with OpenCV")
                original_image = cv2.imread(image_path)
                if original_image is None:
                    error_msg = f"Failed to read image with both PIL and OpenCV: {image_path}"
                    print(f"DEBUG: {error_msg}")
                    self.logger.error(error_msg)
                    add_progress_message(error_msg, None)
                    return False
                else:
                    print(f"DEBUG: OpenCV successfully read image, shape: {original_image.shape}")
            
            # Update progress with image info
            base_name = os.path.basename(image_path)
            status_msg = f"Processing Image: {base_name}"
            self.logger.info(status_msg)
            add_progress_message(status_msg, None)
            
            # ===================================================
            # STEP 2: CREATE DOWNSAMPLED IMAGE FOR FASTER PROCESSING
            # ===================================================
            # For processing speed create a temporary downscaled image
            h, w = original_image.shape[:2]
            original_pixels = h * w
            target_pixels = 2000000  # Target ~2 million pixels (e.g., 1414x1414)
            print(f"DEBUG: Original image dimensions: {w}x{h}, total pixels: {original_pixels}")
            
            if original_pixels > target_pixels:
                scale = (target_pixels / original_pixels) ** 0.5
                new_width = int(w * scale)
                new_height = int(h * scale)
                print(f"DEBUG: Resizing image with scale factor: {scale:.4f}")
                
                # Create downsampled image  
                small_image = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                print(f"DEBUG: Resized image to {new_width}x{new_height}")
                
                add_progress_message(f"Resized image from {w}x{h} to {new_width}x{new_height} for processing", None)
            else:
                # Image is already small enough
                small_image = original_image.copy()
                print("DEBUG: Image already small enough, no resizing needed")
                self.logger.info(f"Image already small ({w}x{h}), using as is for processing")
                add_progress_message(f"Image already small ({w}x{h}), using as is for processing", None)
            
            # Store the small image as an instance variable so it's accessible to the metadata dialog
            self.small_image = small_image.copy()
            
            # ===================================================
            # STEP 3: CHECK IF FILE WAS PREVIOUSLY PROCESSED
            # ===================================================
            # Check if this file has been previously processed (to extract metadata from filename)
            previously_processed = self.file_manager.check_original_file_processed(image_path)
            print(f"DEBUG: Previously processed status: {previously_processed}")

            # Use existing metadata (e.g., from filename) or initialize empty
            metadata = getattr(self, 'metadata', {})
            print(f"DEBUG: Initial metadata: {metadata}")

            # ===================================================
            # STEP 4: DETECT ARUCO MARKERS
            # ===================================================
            # IMPORTANT: Always detect markers regardless of metadata source
            add_progress_message("Detecting ArUco markers...", None)

            
            print("DEBUG: Starting ArUco marker detection")
            # Use ArucoManager to detect and improve marker detection
            markers = self.aruco_manager.improve_marker_detection(small_image)
            print(f"DEBUG: Detected {len(markers)} ArUco markers: {sorted(markers.keys())}")
            
            # Report marker detection status
            expected_markers = set(self.config['corner_marker_ids'] + 
                                self.config['compartment_marker_ids'])
            
            # ===================================================
            # MODIFIED CODE: Only add metadata markers if OCR is enabled
            # ===================================================
            # Only expect metadata markers if OCR is enabled and available
            if self.config.get('enable_ocr', True) and self.tesseract_manager.is_available:
                expected_markers.update(self.config['metadata_marker_ids'])
                print("DEBUG: OCR is enabled - including metadata markers in expected set")
            else:
                print("DEBUG: OCR is disabled - excluding metadata markers from expected set")
            
            detected_markers = set(markers.keys())
            missing_markers = expected_markers - detected_markers
            missing_marker_ids = list(missing_markers)  # Convert to list for later use
            
            print(f"DEBUG: Expected markers: {sorted(expected_markers)}")
            print(f"DEBUG: Missing markers: {sorted(missing_markers)}")

            # Check if we should remove metadata marker from missing list
            if not self.config.get('enable_ocr', True) or not self.tesseract_manager.is_available:
                if 24 in missing_marker_ids:
                    missing_marker_ids.remove(24)
                    print("DEBUG: Removed marker 24 from missing list (OCR disabled or unavailable)")
                    self.logger.info("Removed metadata marker 24 from missing list (OCR disabled or unavailable)")
            
             
            status_msg = f"Detected {len(detected_markers)}/{len(expected_markers)} ArUco markers"
            self.logger.info(status_msg)
            if self.progress_queue:
                add_progress_message(status_msg, None)

                
                if missing_markers:
                    add_progress_message(f"Missing markers: {sorted(missing_markers)}", None)
    

            # ===================================================
            # STEP 5: CORRECT IMAGE ORIENTATION AND SKEW
            # ===================================================
            # Attempt to correct skew using ArucoManager BEFORE OCR processing
            if self.progress_queue:
                add_progress_message("Correcting image orientation...", None)


            # Initialize rotation variables
            rotation_matrix = None
            rotation_angle = 0.0

            try:
                print("DEBUG: Attempting to correct image skew on small image")
                # Use ArucoManager for image correction
                corrected_small_image, rotation_matrix, rotation_angle = self.aruco_manager.correct_image_skew(
                    small_image, markers, return_transform=True
                )
                
                print(f"DEBUG: Skew correction applied, rotation angle: {rotation_angle:.2f}°")
                
                if corrected_small_image is not small_image:  # If correction was applied
                    print("DEBUG: Image was actually modified during skew correction")
                    # Re-detect markers on corrected image
                    markers = self.aruco_manager.improve_marker_detection(corrected_small_image)
                    print(f"DEBUG: Re-detected {len(markers)} markers after correction")
                    small_image = corrected_small_image  # Update small image for further processing
                    
                    self.logger.info(f"Image orientation corrected, angle: {rotation_angle:.2f}°, re-detected {len(markers)} markers")
                    add_progress_message(f"Corrected image orientation (angle: {rotation_angle:.2f}°), re-detected {len(markers)} markers", None)
    
                else:
                    print("DEBUG: No skew correction needed (or applied)")
            except Exception as e:
                self.logger.warning(f"Skew correction failed: {str(e)}")
                self.logger.error(traceback.format_exc())
                print(f"DEBUG: Skew correction failed with error: {str(e)}")
                # Continue with uncorrected image

            # Recalculate missing markers since skew correction may have found more
            detected_markers = set(markers.keys())
            missing_markers = expected_markers - detected_markers
            missing_marker_ids = list(missing_markers)  # Update the list
            
            print(f"DEBUG: After skew correction - detected markers: {sorted(detected_markers)}")
            print(f"DEBUG: After skew correction - missing markers: {sorted(missing_marker_ids)}")
            
            if missing_marker_ids:
                self.logger.info(f"After correction, still missing markers: {sorted(missing_marker_ids)}")
                if self.progress_queue:
                    add_progress_message(f"Still missing markers after correction: {sorted(missing_marker_ids)}", None)
    

            # ===================================================
            # STEP 6: ANALYZE COMPARTMENT BOUNDARIES (WITHOUT UI)
            # ===================================================
            # Extract initial compartment boundaries without dialog
            if self.progress_queue:
                add_progress_message("Analyzing compartment boundaries...", None)


            try:
                # For debugging
                self.logger.debug(f"Analyzing {len(markers)} markers for compartment boundaries: {sorted(markers.keys())}")
                print(f"DEBUG: Starting compartment boundary analysis with {len(markers)} markers")
                
                # Get initial boundary analysis without UI
                boundary_analysis = self.aruco_manager.analyze_compartment_boundaries(
                    small_image, markers, 
                    compartment_count=self.config.get('compartment_count', 20),
                    smart_cropping=True,
                    metadata=metadata
                )
                
                # Extract analysis results
                boundaries = boundary_analysis['boundaries']
                boundaries_viz = boundary_analysis['visualization']
                boundary_missing_markers = boundary_analysis['missing_marker_ids']  # Use different name
                vertical_constraints = boundary_analysis['vertical_constraints']
                marker_to_compartment = boundary_analysis['marker_to_compartment']

                # Merge missing markers - keep original missing markers and add any new ones
                if boundary_missing_markers:
                    # Combine both lists and remove duplicates
                    all_missing = list(set(missing_marker_ids + boundary_missing_markers))
                    missing_marker_ids = sorted(all_missing)
                    print(f"DEBUG: Combined missing markers: {missing_marker_ids}")
                
                # Store top/bottom boundaries for later use
                if vertical_constraints:
                    self.top_y, self.bottom_y = vertical_constraints
                    
                # # Store visualization in cache for dialog use
                # if boundaries_viz is not None:
                #     print(f"DEBUG: Adding boundaries_viz to visualization_cache, shape: {boundaries_viz.shape}")
                #     self.visualization_cache.setdefault('current_processing', {})['compartment_boundaries_viz'] = boundaries_viz
                # else:
                #     print("DEBUG: boundaries_viz is None, cannot add to visualization_cache")

                print(f"DEBUG: Extracted {len(boundaries)} compartment boundaries")
                if boundaries:
                    print(f"DEBUG: First boundary coordinates: {boundaries[0]}")
                    print(f"DEBUG: Last boundary coordinates: {boundaries[-1]}")

                # Simply check if boundaries were found
                if not boundaries:
                    error_msg = "Failed to extract compartment boundaries"
                    print(f"DEBUG: {error_msg}")
                    self.logger.error(error_msg)
                    add_progress_message(error_msg, None)
    
                    return False
                
            except Exception as e:
                error_msg = f"Error in compartment boundary extraction: {str(e)}"
                print(f"DEBUG: {error_msg}")
                print(f"DEBUG: {traceback.format_exc()}")
                self.logger.error(error_msg)
                self.logger.error(traceback.format_exc())
                if self.progress_queue:
                    add_progress_message(error_msg, None)
                return False

            # ===================================================
            # STEP 7: EXTRACT METADATA (FROM OCR OR FILENAME)
            # ===================================================
            # Extract metadata region for OCR and visualization using Tesseract_manager
            if self.progress_queue:
                add_progress_message("Extracting metadata from detected labels...", None)


            print("DEBUG: Starting metadata region extraction with TesseractManager")

            # Make sure TesseractManager has access to necessary references
            print("DEBUG: Setting up TesseractManager references")
            self.tesseract_manager.file_manager = self.file_manager
            self.tesseract_manager.extractor = self

            # Check if we already have metadata from the filename
            if previously_processed:
                # Use the metadata from the filename and skip OCR extraction
                metadata = previously_processed
                print(f"DEBUG: Using metadata from filename: {metadata}")
                self.logger.info(f"Using metadata from filename: {metadata}")
                if self.progress_queue:
                    add_progress_message(f"Using metadata from filename: Hole ID={metadata.get('hole_id')}, Depth={metadata.get('depth_from')}-{metadata.get('depth_to')}m", None)
                
                # Create minimal OCR metadata for dialog display
                print("DEBUG: Creating metadata structure for dialog display")
                ocr_metadata = {
                    'hole_id': metadata.get('hole_id'),
                    'depth_from': metadata.get('depth_from'),
                    'depth_to': metadata.get('depth_to'),
                    'confidence': metadata.get('confidence', 100.0),
                    'from_filename': True,
                    'metadata_region': None
                }
            else:
                # No existing metadata, either perform OCR or create minimal structure
                print("DEBUG: No existing metadata, checking OCR status")
                if self.progress_queue:
                    add_progress_message("Preparing metadata input...", None)
    
                
                # Check if OCR is enabled and available
                if self.config['enable_ocr'] and self.tesseract_manager.is_available:
                    print("DEBUG: OCR is enabled and Tesseract is available")
                    add_progress_message("Extracting metadata with OCR...", None)
    
                    
                    # First try the composite method
                    final_viz = self.aruco_manager.viz_steps.get("final_boundaries", small_image)

                    ocr_metadata = self.tesseract_manager.extract_metadata_with_composite(
                        final_viz, markers, original_filename=image_path, progress_queue=self.progress_queue
                    )
                    
                    # Check if we got good results from the composite method
                    composite_confidence = ocr_metadata.get('confidence', 0)
                    composite_has_data = (ocr_metadata.get('hole_id') is not None and 
                                        ocr_metadata.get('depth_from') is not None and 
                                        ocr_metadata.get('depth_to') is not None)
                    
                    print(f"DEBUG: OCR results - confidence: {composite_confidence:.1f}%, has complete data: {composite_has_data}")
                    print(f"DEBUG: OCR extracted hole_id: {ocr_metadata.get('hole_id')}")
                    print(f"DEBUG: OCR extracted depth_from: {ocr_metadata.get('depth_from')}")
                    print(f"DEBUG: OCR extracted depth_to: {ocr_metadata.get('depth_to')}")
                    
                    # If the composite method didn't yield good results, just log it
                    if not composite_has_data or composite_confidence < self.config['ocr_confidence_threshold']:
                        print(f"DEBUG: OCR confidence too low ({composite_confidence:.1f}%), will prompt user for metadata")
                        self.logger.info(f"OCR confidence too low ({composite_confidence:.1f}%), will prompt user for metadata")

                    # Log OCR results
                    ocr_log_msg = f"OCR Results: Confidence={ocr_metadata.get('confidence', 0):.1f}%"
                    if ocr_metadata.get('hole_id'):
                        ocr_log_msg += f", Hole ID={ocr_metadata['hole_id']}"
                    if ocr_metadata.get('depth_from') is not None and ocr_metadata.get('depth_to') is not None:
                        ocr_log_msg += f", Depth={ocr_metadata['depth_from']}-{ocr_metadata['depth_to']}"
                    
                    print(f"DEBUG: {ocr_log_msg}")
                    self.logger.info(ocr_log_msg)
                    add_progress_message(ocr_log_msg, None)
                else:
                    # OCR is disabled, create a minimal metadata structure
                    print("DEBUG: OCR is disabled, creating minimal metadata structure")
                    ocr_metadata = {
                        'hole_id': None,
                        'depth_from': None,
                        'depth_to': None,
                        'confidence': 0.0,
                        'metadata_region': None
                    }
                    
                    # Try to extract metadata from filename if available
                    if image_path:
                        filename_metadata = self.file_manager.extract_metadata_from_filename(image_path)
                        if filename_metadata:
                            ocr_metadata.update(filename_metadata)
                            ocr_metadata['confidence'] = 100.0  # High confidence for filename metadata
                            ocr_metadata['from_filename'] = True
                            
                            print(f"DEBUG: Extracted metadata from filename: {filename_metadata}")
                    
                    if 24 in missing_marker_ids:
                        missing_marker_ids.remove(24)
                        print("DEBUG: Removed marker 24 from missing list since OCR is disabled")
                        self.logger.info("Removed metadata marker 24 from missing list (OCR disabled)")

            # ===================================================
            # STEP 8: ALWAYS SHOW COMPARTMENT REGISTRATION DIALOG
            # ===================================================
            # ALWAYS show the compartment registration dialog
            if self.progress_queue:
                add_progress_message("Opening compartment registration dialog...", None)


            print("DEBUG: Preparing to show compartment registration dialog")

            # # Get visualization for dialog - use boundaries_viz if available
            compartment_viz = small_image.copy()
            
            # # Store necessary data in visualization cache for dialog
            # if hasattr(self, 'visualization_cache'):
            #     self.visualization_cache.setdefault('current_processing', {})['compartment_boundaries_viz'] = compartment_viz
            #     self.visualization_cache.setdefault('current_processing', {})['original_image'] = small_image.copy()
            #     self.visualization_cache.setdefault('current_processing', {})['all_markers'] = markers

            # # Copy visualization from aruco_manager.viz_steps to visualization_cache if needed
            # if hasattr(self, 'aruco_manager') and hasattr(self.aruco_manager, 'viz_steps') and 'final_boundaries' in self.aruco_manager.viz_steps:
            #     print("DEBUG: Found 'final_boundaries' in aruco_manager.viz_steps, copying to visualization_cache")
            #     self.visualization_cache.setdefault('current_processing', {})['compartment_boundaries_viz'] = self.aruco_manager.viz_steps['final_boundaries']
            #     print(f"DEBUG: Copied final_boundaries visualization with shape {self.aruco_manager.viz_steps['final_boundaries'].shape}")

            try:
                # Create unified registration dialog
                print("DEBUG: Creating combined registration dialog")

                # Define the callback function for re-processing with adjustments
                def process_image_callback(params):
                    # Re-analyze boundaries with adjustment parameters
                    temp_metadata = dict(ocr_metadata) if ocr_metadata else {}
                    temp_metadata['boundary_adjustments'] = params
                    
                    new_analysis = self.aruco_manager.analyze_compartment_boundaries(
                        small_image, markers, 
                        compartment_count=self.config.get('compartment_count', 20),
                        smart_cropping=True,
                        metadata=temp_metadata
                    )
                    
                    return {
                        'boundaries': new_analysis['boundaries'],
                        'visualization': new_analysis['visualization']
                    }
                
                # Debug missing markers before dialog
                print(f"DEBUG: Missing marker IDs before dialog: {missing_marker_ids}")
                print(f"DEBUG: Type of missing_marker_ids: {type(missing_marker_ids)}")
                
                # Call handle_markers_and_boundaries with appropriate parameters
                result = self.handle_markers_and_boundaries(
                    compartment_viz,
                    boundaries,
                    missing_marker_ids=missing_marker_ids,
                    metadata=ocr_metadata,
                    vertical_constraints=(self.top_y, self.bottom_y) if hasattr(self, 'top_y') and hasattr(self, 'bottom_y') else None,
                    marker_to_compartment=marker_to_compartment,
                    rotation_angle=rotation_angle,
                    corner_markers=self.corner_markers if hasattr(self, 'corner_markers') else None,
                    markers=markers,
                    initial_mode=0,  # Start in MODE_METADATA
                    on_apply_adjustments=process_image_callback,
                    image_path=image_path
                )

                print("DEBUG: Registration dialog completed")

                # Process the result based on complete metadata and boundaries
                if result:
                    if result.get('quit', False):
                        # User chose to quit processing
                        print("DEBUG: User quit processing")
                        self.logger.info("User stopped processing")
                        add_progress_message("Processing stopped by user", None)
                        
                        
                        # Set a flag to stop processing remaining images
                        self.processing_complete = True
                        return False  # Return early to stop processing
                    
                    print(f"DEBUG: Result dict: {result}")
                    if result.get('rejected', False):
                        # Handle rejected case using centralized handler
                        print("DEBUG: Image rejected by user")
                        
                        exit_handled = self.handle_image_exit(
                            image_path=image_path,
                            result=result,
                            metadata=metadata,  # May be partial at this point
                            compartments=None,  # No compartments extracted yet
                            processing_messages=processing_messages,
                            add_progress_message=add_progress_message
                        )
                        
                        return exit_handled  # True = handled, continue to next image
                    
                    else:
                        # Use all the data from the registration dialog
                        metadata = {
                            'hole_id': result.get('hole_id'),
                            'depth_from': result.get('depth_from'),
                            'depth_to': result.get('depth_to'),
                            'compartment_interval': result.get('compartment_interval', 1)
                        }
                        
                        # Update boundaries based on result
                        if 'top_boundary' in result and 'bottom_boundary' in result:
                            self.top_y = result['top_boundary']
                            self.bottom_y = result['bottom_boundary']
                            
                        # Store result boundaries for later use
                        self.result_boundaries = result.get('result_boundaries', {})
                        
                        # # Update visualization cache if available
                        # if 'final_visualization' in result and result['final_visualization'] is not None:
                        #     self.visualization_cache.setdefault('current_processing', {})['compartment_boundaries_viz'] = result['final_visualization']
                        #     print("DEBUG: Updated visualization cache with final visualization")
                            
                        # IMPORTANT: Re-apply boundary adjustments to get final boundaries
                        if any(key in result for key in ['top_boundary', 'bottom_boundary', 'left_height_offset', 'right_height_offset']):
                            print("DEBUG: Re-applying boundary adjustments to get final boundaries")
                            
                            # Create adjustment parameters from dialog result
                            adjustment_params = {
                                'top_boundary': result.get('top_boundary', self.top_y),
                                'bottom_boundary': result.get('bottom_boundary', self.bottom_y),
                                'left_height_offset': result.get('left_height_offset', 0),
                                'right_height_offset': result.get('right_height_offset', 0)
                            }
                            
                            # Update metadata with boundary adjustments
                            metadata['boundary_adjustments'] = adjustment_params
                            
                            # Re-analyze boundaries with adjustments
                            print(f"DEBUG: Re-analyzing boundaries with adjustments: {adjustment_params}")
                            adjusted_analysis = self.aruco_manager.analyze_compartment_boundaries(
                                small_image, markers,
                                compartment_count=self.config.get('compartment_count', 20),
                                smart_cropping=True,
                                metadata=metadata
                            )
                            
                            # Update boundaries with adjusted values
                            boundaries = adjusted_analysis['boundaries']
                            print(f"DEBUG: Updated boundaries after adjustments: {len(boundaries)} compartments")
                            
                            # Merge manually placed compartments if any exist
                            # Check if we have manually placed compartments to merge
                            if 'result_boundaries' in result and result['result_boundaries']:
                                print(f"DEBUG: Found {len(result['result_boundaries'])} manually placed compartments to merge")
                                
                                # Convert manually placed compartments to boundary format
                                for marker_id, boundary in result['result_boundaries'].items():
                                    if marker_id == 24:  # Skip metadata marker
                                        continue
                                    
                                    # Check if this compartment is already in boundaries
                                    compartment_exists = False
                                    for existing_boundary in boundaries:
                                        # Check if x-coordinates overlap significantly
                                        x1_existing, _, x2_existing, _ = existing_boundary
                                        if isinstance(boundary, tuple) and len(boundary) == 4:
                                            x1_new, _, x2_new, _ = boundary
                                            # Check for overlap
                                            if (x1_new < x2_existing and x2_new > x1_existing):
                                                compartment_exists = True
                                                break
                                    
                                    if not compartment_exists and isinstance(boundary, tuple) and len(boundary) == 4:
                                        boundaries.append(boundary)
                                        print(f"DEBUG: Added manually placed compartment for marker {marker_id}")
                                
                                # Sort boundaries by x-coordinate
                                boundaries.sort(key=lambda b: b[0])
                                print(f"DEBUG: Total boundaries after merging: {len(boundaries)}")
                            
                            # # Update visualization if available
                            # if adjusted_analysis.get('visualization') is not None:
                            #     self.visualization_cache.setdefault('current_processing', {})['compartment_boundaries_viz'] = adjusted_analysis['visualization']

                        print(f"DEBUG: User completed registration with metadata: {metadata}")
                        self.logger.info(f"User completed registration with metadata: {metadata}")

                else:
                    # User canceled the dialog
                    print("DEBUG: Dialog canceled by user")
                    self.logger.warning("Image processing canceled by user")
                    if self.progress_queue:
                        add_progress_message("Processing canceled by user", None)
        
                    return False  # Return early to skip processing
            except Exception as e:
                self.logger.error(f"Metadata dialog error: {str(e)}")
                self.logger.error(traceback.format_exc())
                print(f"DEBUG: Metadata dialog error: {str(e)}")
                print(f"DEBUG: {traceback.format_exc()}")
                if self.progress_queue:
                    add_progress_message(f"Metadata dialog error: {str(e)}", None)
                return False



            # ===================================================
            # STEP 10: APPLY SKEW CORRECTION TO HIGH-RES IMAGE
            # ===================================================
            # Apply skew correction to the original high-resolution image if needed
            if rotation_matrix is not None and rotation_angle != 0.0:
                print(f"DEBUG: Applying orientation correction to high-res image (angle: {rotation_angle:.2f}°)")
                if self.progress_queue:
                    add_progress_message(f"Applying orientation correction to high-resolution image (angle: {rotation_angle:.2f}°)...", None)
                
                # Get dimensions for the output image
                h, w = original_image.shape[:2]
                
                # IMPORTANT: Recalculate rotation matrix for the high-res image dimensions
                # The rotation matrix from the small image has translation components specific to the small image size
                high_res_center = (w // 2, h // 2)
                high_res_rotation_matrix = cv2.getRotationMatrix2D(high_res_center, rotation_angle, 1.0)
                
                # Apply the rotation with the correctly sized matrix
                corrected_original_image = cv2.warpAffine(
                    original_image,
                    high_res_rotation_matrix,
                    (w, h),  # Explicitly use width, height order
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255)
                )
                # Use the corrected image for further processing
                original_image = corrected_original_image
                print(f"DEBUG: High-resolution image orientation corrected with {rotation_angle:.2f}° rotation")
                self.logger.info(f"Applied orientation correction to high-resolution image (angle: {rotation_angle:.2f}°)")
            else:
                print("DEBUG: No orientation correction needed for high-resolution image")


            # ===================================================
            # STEP 11: SCALE BOUNDARIES TO ORIGINAL IMAGE SIZE
            # ===================================================
            
            # Report number of compartments found
            status_msg = f"Found {len(boundaries)}/{self.config['compartment_count']} compartments"
            print(f"DEBUG: {status_msg}")
            self.logger.info(status_msg)
            if self.progress_queue:
                add_progress_message(status_msg, None)

            
            # Scale up the coordinates from small image to original image if necessary
            if small_image.shape != original_image.shape:
                print("DEBUG: Need to scale boundaries to original image size")
                if self.progress_queue:
                    add_progress_message("Scaling boundaries to original image size...", None)
                    
                # Calculate scale factors
                scale_x = original_image.shape[1] / small_image.shape[1]
                scale_y = original_image.shape[0] / small_image.shape[0]
                print(f"DEBUG: Scale factors: x={scale_x:.4f}, y={scale_y:.4f}")

                # First merge any manually placed compartments before scaling
                if hasattr(self, 'result_boundaries') and self.result_boundaries:
                    print(f"DEBUG: Found {len(self.result_boundaries)} result boundaries to merge before scaling")
                    for marker_id, boundary in self.result_boundaries.items():
                        if marker_id == 24:  # Metadata marker - not a compartment boundary
                            # Metadata markers are for OCR, not compartments
                            # Will be scaled separately below
                            print(f"DEBUG: Found metadata marker {marker_id} - will scale separately")
                            continue
                        if marker_id in [0, 1, 2, 3]:  # Corner markers - not compartment boundaries
                            # Corner markers define constraints, not compartments
                            # Will be scaled separately below
                            print(f"DEBUG: Found corner marker {marker_id} - will scale separately")
                            continue
                        if isinstance(boundary, tuple) and len(boundary) == 4:
                            # Check if this boundary is already in our list
                            boundary_exists = False
                            for existing in boundaries:
                                if (abs(existing[0] - boundary[0]) < 5 and 
                                    abs(existing[1] - boundary[1]) < 5):
                                    boundary_exists = True
                                    break
                            
                            if not boundary_exists:
                                boundaries.append(boundary)
                                print(f"DEBUG: Added compartment boundary for marker {marker_id} before scaling")
                    
                    # Sort boundaries by x-coordinate
                    boundaries.sort(key=lambda b: b[0])
                
                # Scale the boundary adjustments if they exist
                scaled_left_offset = 0
                scaled_right_offset = 0
                scaled_top_y = int(self.top_y * scale_y) if hasattr(self, 'top_y') else None
                scaled_bottom_y = int(self.bottom_y * scale_y) if hasattr(self, 'bottom_y') else None
                
                if metadata and 'boundary_adjustments' in metadata:
                    adjustments = metadata['boundary_adjustments']
                    print(f"DEBUG: Scaling boundary adjustments from small to original image")
                    
                    # Scale the adjustments
                    scaled_top_y = int(adjustments['top_boundary'] * scale_y)
                    scaled_bottom_y = int(adjustments['bottom_boundary'] * scale_y)
                    scaled_left_offset = int(adjustments['left_height_offset'] * scale_y)
                    scaled_right_offset = int(adjustments['right_height_offset'] * scale_y)
                    
                    print(f"DEBUG: Original adjustments: {adjustments}")
                    print(f"DEBUG: Scaled adjustments: top={scaled_top_y}, bottom={scaled_bottom_y}, "
                          f"left_offset={scaled_left_offset}, right_offset={scaled_right_offset}")
                
                # Scale up the compartment boundaries and apply adjustments mathematically
                compartment_boundaries = []
                img_width = original_image.shape[1]
                
                for x1, y1, x2, y2 in boundaries:
                    # Scale the x coordinates
                    scaled_x1 = int(x1 * scale_x)
                    scaled_x2 = int(x2 * scale_x)
                    
                    # If we have boundary adjustments, calculate adjusted y values
                    if scaled_top_y is not None and scaled_bottom_y is not None:
                        # Calculate the center x position of this compartment
                        center_x = (scaled_x1 + scaled_x2) / 2
                        
                        # Calculate y positions based on the sloped boundaries
                        left_top_y = scaled_top_y + scaled_left_offset
                        right_top_y = scaled_top_y + scaled_right_offset
                        left_bottom_y = scaled_bottom_y + scaled_left_offset
                        right_bottom_y = scaled_bottom_y + scaled_right_offset
                        
                        # Calculate slopes
                        if img_width > 0:
                            top_slope = (right_top_y - left_top_y) / img_width
                            bottom_slope = (right_bottom_y - left_bottom_y) / img_width
                            
                            # Calculate y values at this x position
                            scaled_y1 = int(left_top_y + (top_slope * center_x))
                            scaled_y2 = int(left_bottom_y + (bottom_slope * center_x))
                        else:
                            scaled_y1 = scaled_top_y
                            scaled_y2 = scaled_bottom_y
                    else:
                        # No adjustments, just scale the y coordinates
                        scaled_y1 = int(y1 * scale_y)
                        scaled_y2 = int(y2 * scale_y)
                    
                    compartment_boundaries.append((scaled_x1, scaled_y1, scaled_x2, scaled_y2))
                
                # Scale result_boundaries (manually placed markers)
                if hasattr(self, 'result_boundaries') and self.result_boundaries:
                    scaled_result_boundaries = {}
                    for marker_id, boundary in self.result_boundaries.items():
                        if marker_id == 24 or marker_id in [0, 1, 2, 3]:  # Special markers
                            if isinstance(boundary, np.ndarray):
                                scaled_corners = boundary.copy()
                                scaled_corners[:, 0] *= scale_x
                                scaled_corners[:, 1] *= scale_y
                                scaled_result_boundaries[marker_id] = scaled_corners.astype(np.float32)
                                print(f"DEBUG: Scaled special marker {marker_id}")
                        elif isinstance(boundary, tuple) and len(boundary) == 4:
                            # This is a manually placed compartment - apply same scaling logic
                            x1, y1, x2, y2 = boundary
                            scaled_x1 = int(x1 * scale_x)
                            scaled_x2 = int(x2 * scale_x)
                            
                            if scaled_top_y is not None and scaled_bottom_y is not None:
                                center_x = (scaled_x1 + scaled_x2) / 2
                                left_top_y = scaled_top_y + scaled_left_offset
                                right_top_y = scaled_top_y + scaled_right_offset
                                left_bottom_y = scaled_bottom_y + scaled_left_offset
                                right_bottom_y = scaled_bottom_y + scaled_right_offset
                                
                                if img_width > 0:
                                    top_slope = (right_top_y - left_top_y) / img_width
                                    bottom_slope = (right_bottom_y - left_bottom_y) / img_width
                                    scaled_y1 = int(left_top_y + (top_slope * center_x))
                                    scaled_y2 = int(left_bottom_y + (bottom_slope * center_x))
                                else:
                                    scaled_y1 = scaled_top_y
                                    scaled_y2 = scaled_bottom_y
                            else:
                                scaled_y1 = int(y1 * scale_y)
                                scaled_y2 = int(y2 * scale_y)
                                
                            scaled_result_boundaries[marker_id] = (scaled_x1, scaled_y1, scaled_x2, scaled_y2)
                            print(f"DEBUG: Scaled compartment boundary for marker {marker_id}")
                    
                    self.result_boundaries = scaled_result_boundaries
                
                # Update vertical constraints
                if hasattr(self, 'top_y') and hasattr(self, 'bottom_y'):
                    self.top_y = scaled_top_y if scaled_top_y is not None else int(self.top_y * scale_y)
                    self.bottom_y = scaled_bottom_y if scaled_bottom_y is not None else int(self.bottom_y * scale_y)
                    print(f"DEBUG: Updated vertical constraints: top_y={self.top_y}, bottom_y={self.bottom_y}")
                    
                print(f"DEBUG: Scaled {len(compartment_boundaries)} boundaries to original image size")
                if compartment_boundaries:
                    print(f"DEBUG: First scaled boundary: {compartment_boundaries[0]}")
                    print(f"DEBUG: Last scaled boundary: {compartment_boundaries[-1]}")
                    
                self.logger.info(f"Scaled {len(compartment_boundaries)} compartment boundaries to original image")
            else:
                # No scaling needed
                print("DEBUG: No scaling needed - small image and original have same dimensions")
                compartment_boundaries = boundaries

            # ===================================================
            # STEP 12: EXTRACT INDIVIDUAL COMPARTMENTS
            # ===================================================
            # Extract compartments from the ORIGINAL high-resolution image
            if self.progress_queue:
                add_progress_message("Extracting high-resolution compartments...", None)
                
            # Use ArucoManager to extract compartments
            print("DEBUG: Extracting compartments from high-resolution image")
            compartments, _ = self.aruco_manager.extract_compartments(
                original_image, compartment_boundaries)

            print(f"DEBUG: Extracted {len(compartments)} high-resolution compartments")
            
            # ===================================================
            # STEP 13: CHECK FOR DUPLICATES
            # ===================================================
            # Check for duplicates BEFORE adding to QAQC queue or saving
            if (metadata.get('hole_id') and 
                metadata.get('depth_from') is not None and 
                metadata.get('depth_to') is not None):
                
                # Loop for handling metadata modification and re-checking duplicates
                continue_processing = True
                while continue_processing:
                    print("DEBUG: Checking for duplicates with metadata: hole_id={}, depth={}-{}m".format(
                        metadata.get('hole_id'), metadata.get('depth_from'), metadata.get('depth_to')))
                    
                    # Check for duplicates - this method should run on main thread
                    print("DEBUG: Calling duplicate_handler.check_duplicate")
                    duplicate_result = self.duplicate_handler.check_duplicate(
                        metadata['hole_id'], 
                        metadata['depth_from'], 
                        metadata['depth_to'], 
                        small_image,  # Use the downsampled image
                        image_path,
                        extracted_compartments=compartments  # Pass extracted compartments
                    )
                    
                    print(f"DEBUG: Duplicate check result: {duplicate_result}")
                    
                    # Process the result based on its type
                    if isinstance(duplicate_result, dict):
                        if duplicate_result.get('quit', False):
                            # User chose to quit processing
                            print("DEBUG: User chose to quit processing")
                            self.logger.info("User stopped processing via duplicate dialog")
                            # Set processing complete flag
                            self.processing_complete = True
                            # Don't move any files - just stop processing
                            return False  # Stop processing                            
                        # If the result includes 'selective_replacement' flag, handle it specially
                        if duplicate_result.get('selective_replacement', False):
                            # We'll extract compartments to Temp_Review but won't trigger QAQC yet
                            print("DEBUG: Selective replacement selected")
                            self.logger.info(f"Selective replacement selected for {metadata['hole_id']} {metadata['depth_from']}-{metadata['depth_to']}m")
                            # Continue with processing but flag for later QAQC
                            continue_processing = False  # Exit the loop
                            # TODO - if gaps are extracted and used then this file shouldn't go to the skipped originals folder it's effectively a 'selective replace'
                        elif duplicate_result.get('action') == 'keep_with_gaps':
                            # User chose to keep original but fill gaps
                            print("DEBUG: User chose to keep original and fill gaps")
                            
                            gaps_result = {
                                'skipped': True,  # Original is skipped
                                'action': 'keep_with_gaps',
                                'missing_depths': duplicate_result.get('missing_depths', []),
                                'hole_id': metadata.get('hole_id'),
                                'depth_from': metadata.get('depth_from'),
                                'depth_to': metadata.get('depth_to'),
                                'compartment_interval': metadata.get('compartment_interval', 1)
                            }
                            
                            exit_handled = self.handle_image_exit(
                                image_path=image_path,
                                result=gaps_result,
                                metadata=metadata,
                                compartments=compartments,
                                processing_messages=processing_messages,
                                add_progress_message=add_progress_message
                            )
                            
                            return exit_handled
                        
                        else:
                            # User chose to modify metadata - update metadata and check again in next loop iteration
                            print(f"DEBUG: User modified metadata from {metadata} to {duplicate_result}")
                            self.logger.info(f"User modified metadata from {metadata} to {duplicate_result}")
                            metadata = duplicate_result
                            # Continue the loop to re-check duplicates with new metadata
                    elif duplicate_result == True:
                        # No duplicates found, continue with processing
                        print("DEBUG: No duplicates found, continuing with processing")
                        continue_processing = False  # Exit the loop
                    elif duplicate_result == False:
                        # User chose to skip processing
                        print("DEBUG: User chose to skip processing")
                        
                        skip_result = {
                            'skipped': True,
                            'hole_id': metadata.get('hole_id'),
                            'depth_from': metadata.get('depth_from'), 
                            'depth_to': metadata.get('depth_to'),
                            'compartment_interval': metadata.get('compartment_interval', 1)
                        }
                        
                        exit_handled = self.handle_image_exit(
                            image_path=image_path,
                            result=skip_result,
                            metadata=metadata,
                            compartments=compartments,  # We have compartments at this point
                            processing_messages=processing_messages,
                            add_progress_message=add_progress_message
                        )
                        
                        return exit_handled
                    

            # Ensure GUI is responsive and progress is updated after duplicate dialog
            if self.progress_queue:
                add_progress_message("Continuing to next image...", None)
            
            # Force GUI update to process any pending events
            if hasattr(self, 'main_gui'):
                self.main_gui.root.update_idletasks()


            # ===================================================
            # STEP 14: ADD TO QAQC QUEUE
            # ===================================================            
            # Only proceed with QAQC and saving if we passed the duplicate check
            # Create QAQC manager if it doesn't exist
            if not hasattr(self, 'qaqc_manager') or self.qaqc_manager is None:
                print("DEBUG: Creating QAQC manager")
                from gui.qaqc_manager import QAQCManager
                self.qaqc_manager = QAQCManager(self.root, self.file_manager, self)
                self.qaqc_manager.config = self.config

            # Verify we have valid metadata before adding to QAQC queue
            if (metadata.get('hole_id') and 
                metadata.get('depth_from') is not None and 
                metadata.get('depth_to') is not None):
                
                # Validate hole ID format
                hole_id = metadata['hole_id']
                if not re.match(r'^[A-Za-z]{2}\d{4}$', hole_id):
                    self.logger.warning(f"Invalid hole ID format: {hole_id}, skipping QAQC")
                    return False
                
                # Validate depth values are integers
                depth_from = metadata['depth_from']
                depth_to = metadata['depth_to']
                
                if not isinstance(depth_from, int) and depth_from != int(depth_from):
                    self.logger.warning(f"Invalid depth_from value: {depth_from}, skipping QAQC")
                    return False
                    
                if not isinstance(depth_to, int) and depth_to != int(depth_to):
                    self.logger.warning(f"Invalid depth_to value: {depth_to}, skipping QAQC")
                    return False
                
                # Validate depth_to > depth_from
                if depth_to <= depth_from:
                    self.logger.warning(f"Invalid depth range: {depth_from}-{depth_to}, skipping QAQC")
                    return False
                
                # Check if this is a selective replacement based on duplicate_result
                is_selective_replacement = (isinstance(duplicate_result, dict) and 
                                        duplicate_result.get('selective_replacement', False))
                # Store this flag for later use in upload
                self._current_selective_replacement = is_selective_replacement

                # Add the tray to the QAQC manager's queue
                print(f"DEBUG: Adding tray to QAQC review queue with {len(compartments)} compartments")
                print(f"DEBUG: Selective replacement: {is_selective_replacement}")
                self.qaqc_manager.add_tray_for_review(
                    metadata['hole_id'],
                    metadata['depth_from'],
                    metadata['depth_to'],
                    image_path,
                    compartments,
                    is_selective_replacement=is_selective_replacement
                )
                self.logger.info(f"Added {metadata['hole_id']} {metadata['depth_from']}-{metadata['depth_to']}m to review queue")
            
            # ===================================================
            # STEP 15: BLUR DETECTION (OPTIONAL)
            # ===================================================
            # Detect blur in the compartments
            if self.config['enable_blur_detection']:
                print("DEBUG: Blur detection enabled, analyzing image sharpness")
                if self.progress_queue:
                    add_progress_message("Analyzing image sharpness...", None)
                    
                blur_results = self.detect_blur_in_compartments(compartments, os.path.basename(image_path))
                print(f"DEBUG: Blur detection complete, found {sum(1 for r in blur_results if r.get('is_blurry', False))} blurry compartments")
                
                # Flag blurry images if enabled
                if self.config['flag_blurry_images']:
                    print("DEBUG: Adding blur indicators to compartments")
                    compartments = self.add_blur_indicators(compartments, blur_results)

            # ===================================================
            # STEP 17: REGISTER WITH DUPLICATE HANDLER
            # ===================================================
            # Register with duplicate handler when processing is complete and successful
            if (metadata.get('hole_id') and 
                metadata.get('depth_from') is not None and 
                metadata.get('depth_to') is not None):
                # Register the processed entry with the duplicate handler
                print("DEBUG: Registering processed entry with duplicate handler")
                output_files = []
                expected_count = self.config['compartment_count']
                total_depth = int(metadata['depth_to']) - int(metadata['depth_from'])
                depth_increment = self.config['compartment_interval']
                
                # Create the expected filenames
                for i in range(expected_count):
                    comp_depth_to = int(metadata['depth_from']) + ((i + 1) * depth_increment)
                    filename = f"{metadata['hole_id']}_CC_{int(comp_depth_to)}.{self.config['output_format']}"
                    output_files.append(filename)
                
                print(f"DEBUG: Registering with duplicate handler: {len(output_files)} expected output files")
                self.duplicate_handler.register_processed_entry(
                    metadata['hole_id'], 
                    metadata['depth_from'], 
                    metadata['depth_to'], 
                    output_files
                )

                self.update_last_metadata(
                    metadata['hole_id'],
                    metadata['depth_from'],
                    metadata['depth_to'],
                    metadata.get('compartment_interval', 1)
                )

                # ===================================================
                # STEP 18: HANDLE FINAL SAVE AND REGISTER UPDATE
                # ===================================================
                # Build result for successful processing
                final_result = {
                    'hole_id': metadata['hole_id'],
                    'depth_from': metadata['depth_from'],
                    'depth_to': metadata['depth_to'],
                    'compartment_interval': metadata.get('compartment_interval', 1),
                    'selective_replacement': getattr(self, '_current_selective_replacement', False),
                    'rejected': False
                }
                
                exit_handled = self.handle_image_exit(
                    image_path=image_path,
                    result=final_result,
                    metadata=metadata,
                    compartments=compartments,
                    processing_messages=processing_messages,
                    add_progress_message=add_progress_message
                )
                
                if not exit_handled:
                    self.logger.error("Failed to save processed image")
                    return False

            # ===================================================
            # STEP 19: FINAL SUMMARY AND CLEANUP
            # ===================================================
            # Final summary
            success_msg = f"Successfully processed {base_name}: added {len(compartments)} compartments to review queue"
            if metadata.get('hole_id'):
                success_msg += f" for hole {metadata['hole_id']}"
                
            print(f"DEBUG: {success_msg}")
            self.logger.info(success_msg)
            if self.progress_queue:
                add_progress_message(success_msg, None)
                self.main_gui.check_progress(schedule_next=False)


            # FLUSH LOGIC: Send all collected messages to main GUI
            if self.progress_queue and processing_messages:
                # Option 1: Send all messages at once
                for msg, progress in processing_messages:
                    self.progress_queue.put((msg, progress))
                
                # Force GUI update
                if hasattr(self, 'main_gui') and hasattr(self.main_gui, 'root'):
                    self.main_gui.root.update_idletasks()
                    self.main_gui.check_progress(schedule_next=False)

            
            # Clear metadata for next image
            print("DEBUG: Clearing metadata for next image")
            self.metadata = {}
            print("DEBUG: Resetting annotation state in visualization_cache")
            if hasattr(self, 'visualization_cache'):
                if 'annotation_status' in self.visualization_cache:
                    self.visualization_cache['annotation_status'] = {'annotated_markers': []}
                # Also clear current processing data
                if 'current_processing' in self.visualization_cache:
                    self.visualization_cache['current_processing'] = {}
            
            print("DEBUG: process_image completed successfully")
            # Only set processing_complete if user explicitly chose to quit
            if not hasattr(self, 'processing_complete'):
                self.processing_complete = False
            print(f"DEBUG: processing_complete flag: {self.processing_complete}")
            
            return True  # Processing succeeded
                    
        except Exception as e:
            # ===================================================
            # ERROR HANDLING
            # ===================================================
            # Handle any unexpected errors during processing
            error_msg = f"Error processing {image_path}: {str(e)}"
            print(f"DEBUG: CRITICAL ERROR: {error_msg}")
            print(f"DEBUG: {traceback.format_exc()}")
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
            # Update progress queue if available
            add_progress_message(error_msg, None)

              
            # FLUSH LOGIC: Send all messages including the error
            if self.progress_queue and processing_messages:
                for msg, progress in processing_messages:
                    self.progress_queue.put((msg, progress))
            
            # Reset metadata for next image
            self.metadata = {}
            
            # Reset annotation state in visualization_cache
            print("DEBUG: Resetting annotation state in visualization_cache after exception")
            if hasattr(self, 'visualization_cache'):
                if 'annotation_status' in self.visualization_cache:
                    self.visualization_cache['annotation_status'] = {'annotated_markers': []}
                # Also clear current processing data
                if 'current_processing' in self.visualization_cache:
                    self.visualization_cache['current_processing'] = {}
            
            print("DEBUG: process_image failed")
            return False
        
    def process_folder(self, folder_path: str) -> Tuple[int, int]:
        """
        Process all images in a folder, one at a time on the main thread.
        
        Args:
            folder_path: Path to folder containing images
            
        Returns:
            Tuple of (number of images processed, number of images failed)
        """
        print(f"DEBUG: Starting process_folder with path: {folder_path}")
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        print(f"DEBUG: Logger initialized: {self.logger}")
        
        # Ensure progress_queue exists
        if not hasattr(self, 'progress_queue'):
            print("DEBUG: Creating new progress_queue")
            self.progress_queue = queue.Queue()
        else:
            print("DEBUG: Using existing progress_queue")
        
        # Get all image files from the folder
        print("DEBUG: Scanning for image files")
        image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                    if os.path.isfile(os.path.join(folder_path, f)) and 
                    f.lower().endswith(image_extensions)]
        
        print(f"DEBUG: Found {len(image_files)} image files with extensions {image_extensions}")
        
        if not image_files:
            warning_msg = f"No image files found in {folder_path}"
            print(f"DEBUG: {warning_msg}")
            self.logger.warning(warning_msg)
            if self.progress_queue:
                self.progress_queue.put((warning_msg, None))
            return 0, 0
        
        # Log found images
        info_msg = f"Found {len(image_files)} image files"
        print(f"DEBUG: {info_msg}")
        self.logger.info(info_msg)
        if self.progress_queue:
            self.progress_queue.put((info_msg, None))
        
        # Process each image one by one
        successful = 0
        failed = 0
        
        print(f"DEBUG: Starting to process {len(image_files)} images")
        for i, image_path in enumerate(image_files):
            try:
                # Update progress
                progress = ((i + 1) / len(image_files)) * 100
                progress_msg = f"Processing Image: {i+1}/{len(image_files)}: {os.path.basename(image_path)}"
                print(f"DEBUG: {progress_msg} (progress: {progress:.1f}%)")
                self.logger.info(progress_msg)
                
                print(f"DEBUG: Updating progress queue with {progress:.1f}%")
                self.progress_queue.put((progress_msg, progress))
                
                # Process the image on the main thread
                print(f"DEBUG: Calling process_image for {os.path.basename(image_path)}")
                result = self.process_image(image_path)
                print(f"DEBUG: process_image returned {result}")
                
                if result:
                    successful += 1
                    print(f"DEBUG: Successful count incremented to {successful}")
                else:
                    failed += 1
                    print(f"DEBUG: Failed count incremented to {failed}")
                
                print(f"DEBUG: Finished processing image {i+1}/{len(image_files)}, continuing to next...")
                # Update GUI to ensure it's responsive
                if hasattr(self, 'main_gui') and hasattr(self.main_gui, 'root'):
                    self.main_gui.root.update_idletasks()
                    
            except Exception as e:
                error_msg = f"Error processing {image_path}: {str(e)}"
                print(f"DEBUG: Exception caught: {error_msg}")
                print(f"DEBUG: Exception type: {type(e).__name__}")
                self.logger.error(error_msg)
                self.logger.error(traceback.format_exc())
                self.progress_queue.put((error_msg, None))
                failed += 1
                print(f"DEBUG: Failed count incremented to {failed}")
        
        # Log summary
        summary_msg = f"Processing complete: {successful} successful, {failed} failed"
        print(f"DEBUG: {summary_msg}")
        self.logger.info(summary_msg)
        if self.progress_queue:
            self.progress_queue.put((summary_msg, 100))  # Set progress to 100%
        
        print(f"DEBUG: Returning from process_folder: ({successful}, {failed})")
        return successful, failed

    def handle_image_exit(self, 
                        image_path: str,
                        result: dict,
                        metadata: dict = None,
                        compartments: list = None,
                        processing_messages: list = None,
                        add_progress_message=None) -> bool:
        """
        Centralized handler for all image processing exit points.
        Handles file saving, register updates, and cleanup consistently.
        
        Args:
            image_path: Path to the original image
            result: Result dictionary containing exit status and metadata
            metadata: Current metadata (may be partial depending on exit point)
            compartments: Extracted compartments (if any)
            processing_messages: List of progress messages to flush
            add_progress_message: Function to add progress messages
            
        Returns:
            bool: True if handled successfully, False otherwise
        """
        try:
            # Determine exit type
            is_rejected = result.get('rejected', False)
            is_quit = result.get('quit', False)
            is_skipped = result.get('skipped', False)
            is_selective = result.get('selective_replacement', False)
            
            # Use result metadata if more complete than current metadata
            if metadata is None:
                metadata = {}
            
            # Merge result data into metadata (result takes precedence)
            final_metadata = {
                'hole_id': result.get('hole_id', metadata.get('hole_id')),
                'depth_from': result.get('depth_from', metadata.get('depth_from')),
                'depth_to': result.get('depth_to', metadata.get('depth_to')),
                'compartment_interval': result.get('compartment_interval', metadata.get('compartment_interval', 1))
            }
            
            # Handle quit case - don't save anything
            if is_quit:
                print("DEBUG: User quit processing")
                self.logger.info("User stopped processing")
                if add_progress_message:
                    add_progress_message("Processing stopped by user", None)
                self.processing_complete = True
                return False
            
            # Validate we have minimum required metadata
            if not all([final_metadata.get('hole_id'), 
                    final_metadata.get('depth_from') is not None,
                    final_metadata.get('depth_to') is not None]):
                self.logger.warning("Cannot save file - missing required metadata")
                if add_progress_message:
                    add_progress_message("Cannot save file - missing metadata", None)
                return False
            
            # Update last metadata for next image
            self.update_last_metadata(
                final_metadata['hole_id'],
                final_metadata['depth_from'],
                final_metadata['depth_to'],
                final_metadata['compartment_interval']
            )
            
            # Get OneDrive manager if available
            onedrive_mgr = None
            if hasattr(self, 'onedrive_manager') and self.onedrive_manager is not None:
                onedrive_mgr = self.onedrive_manager
                print(f"DEBUG: OneDrive manager available for file save")
            
            # Save the original file
            print(f"DEBUG: Saving file with status - rejected:{is_rejected}, skipped:{is_skipped}, selective:{is_selective}")
            local_path, upload_success = self.file_manager.save_original_file(
                image_path,
                final_metadata['hole_id'],
                final_metadata['depth_from'],
                final_metadata['depth_to'],
                onedrive_manager=onedrive_mgr,
                is_processed=True,  # All handled files are "processed"
                is_rejected=is_rejected,
                is_selective=is_selective
            )
            
            # Update register ONLY for successfully uploaded images
            if upload_success and hasattr(self, 'qaqc_manager') and hasattr(self.qaqc_manager, 'register_manager'):
                original_filename = os.path.basename(image_path)
                
                # Build comments based on exit type
                comments = []
                if is_rejected:
                    rejection_reason = result.get('rejection_reason', 'No reason provided')
                    comments.append(f"Rejected by user during processing. Reason: {rejection_reason}")
                elif is_skipped:
                    comments.append("Skipped by user during duplicate check")
                elif is_selective:
                    comments.append("Selective compartment replacement")
                
                # Add any gap filling info if available
                if result.get('action') == 'keep_with_gaps':
                    missing_depths = result.get('missing_depths', [])
                    if missing_depths:
                        comments.append(f"Filled gaps at depths: {missing_depths}")
                
                comments_str = ". ".join(comments) if comments else None
                
                self.qaqc_manager.register_manager.update_original_image(
                    final_metadata['hole_id'],
                    final_metadata['depth_from'],
                    final_metadata['depth_to'],
                    original_filename,
                    is_approved=not is_rejected,  # False for rejected, True for others
                    upload_success=True,  # Always True here since we're in the upload_success block
                    uploaded_by=os.getenv("USERNAME", "Unknown"),
                    comments=comments_str
                )
                
                status_type = "rejected" if is_rejected else ("skipped" if is_skipped else "processed")
                self.logger.info(f"Updated register for successfully uploaded {status_type} image")
            elif not upload_success:
                self.logger.info(f"Image not added to register (upload failed or OneDrive not available)")
            
            # Add appropriate progress message
            if add_progress_message:
                if local_path:
                    # Check the actual filename to see status
                    actual_filename = os.path.basename(local_path)
                    
                    if is_rejected:
                        if '_UPLOADED' in actual_filename:
                            add_progress_message("Rejected image uploaded to OneDrive and saved locally", None)
                        elif '_UPLOAD_FAILED' in actual_filename:
                            add_progress_message("Rejected image saved locally (OneDrive upload failed)", None)
                        else:
                            add_progress_message("Rejected image saved locally", None)
                    elif is_skipped:
                        msg = "File skipped - saved to Failed and Skipped folder"
                        if upload_success:
                            msg += " and uploaded to OneDrive"
                        add_progress_message(msg, None)
                    else:
                        msg = "Image saved"
                        if upload_success:
                            msg += " and uploaded to OneDrive"
                        elif onedrive_mgr:
                            msg += " (OneDrive upload failed)"
                        add_progress_message(msg, None)
                else:
                    add_progress_message("Failed to save file", None)
            
            # Extract compartments for gaps if needed
            if result.get('action') == 'keep_with_gaps' and compartments:
                missing_depths = result.get('missing_depths', [])
                if missing_depths:
                    print(f"DEBUG: Extracting {len(missing_depths)} missing compartments")
                    start_depth = int(final_metadata['depth_from'])
                    
                    for i, comp in enumerate(compartments):
                        if comp is not None:
                            comp_depth = start_depth + i + 1
                            if comp_depth in missing_depths:
                                try:
                                    saved_path = self.file_manager.save_compartment(
                                        comp,
                                        final_metadata['hole_id'],
                                        comp_depth,
                                        has_data=False,
                                        output_format=self.config['output_format']
                                    )
                                    if saved_path:
                                        self.logger.info(f"Extracted missing compartment at depth {comp_depth}m")
                                except Exception as e:
                                    self.logger.error(f"Error saving missing compartment: {str(e)}")
            
            return True  # Successfully handled
            
        except Exception as e:
            self.logger.error(f"Error in handle_image_exit: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            # IMPORTANT: Emergency save should NOT delete the original file
            # We need to ensure the original image remains untouched if we can't save it properly
            
            try:
                print("DEBUG: Error occurred - checking if original file still exists")
                
                if os.path.exists(image_path):
                    self.logger.info(f"Original file still exists at: {image_path}")
                    if add_progress_message:
                        add_progress_message("Error saving file - original file preserved", None)
                else:
                    self.logger.error(f"CRITICAL: Original file no longer exists: {image_path}")
                    if add_progress_message:
                        add_progress_message("CRITICAL ERROR: Original file may be lost!", None)
                        
            except Exception as check_error:
                self.logger.error(f"Error checking original file: {str(check_error)}")
            
            return False        

    def detect_blur_in_compartments(self, compartments, base_filename=None):
        """
        Detect blur in extracted compartment images using BlurDetector.
        
        Args:
            compartments: List of compartment images
            base_filename: Original filename for generating visualization filenames
            
        Returns:
            List of dictionaries with blur analysis results
        """
        # Delegate to BlurDetector
        if not self.config['enable_blur_detection']:
            # Return empty results if blur detection is disabled
            return [{'index': i, 'is_blurry': False, 'variance': 0.0} for i in range(len(compartments))]
        
        # Configure blur detector with current settings
        self.blur_detector.threshold = self.config['blur_threshold']
        self.blur_detector.roi_ratio = self.config['blur_roi_ratio']
        
        # Generate visualizations if enabled in config
        generate_viz = self.config['save_blur_visualizations']
        return self.blur_detector.batch_analyze_images(compartments, generate_viz)
    
    def add_blur_indicators(self, compartment_images, blur_results):
        """
        Add visual indicators to blurry compartment images.
        
        Args:
            compartment_images: List of compartment images
            blur_results: List of blur analysis results
            
        Returns:
            List of compartment images with blur indicators added
        """
        # Skip if not enabled
        if not self.config['flag_blurry_images']:
            return compartment_images
            
        # Delegate to BlurDetector or process here for simple indicators
        # This is kept in the app class because it's more UI focused
        import cv2
        import numpy as np
        
        result_images = []
        
        for i, image in enumerate(compartment_images):
            # Find the blur result for this image
            result = next((r for r in blur_results if r['index'] == i), None)
            
            if result and result.get('is_blurry', False):
                # Create a copy for modification
                marked_image = image.copy()
                
                # Get image dimensions
                h, w = marked_image.shape[:2]
                
                # Add a red border
                border_thickness = max(3, min(h, w) // 50)  # Scale with image size
                cv2.rectangle(
                    marked_image, 
                    (0, 0), 
                    (w - 1, h - 1), 
                    (0, 0, 255),  # Red in BGR
                    border_thickness
                )
                
                # Add "BLURRY" text
                font_scale = max(0.5, min(h, w) / 500)  # Scale with image size
                text_size, _ = cv2.getTextSize(
                    "BLURRY", 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, 
                    2
                )
                
                # Position text in top-right corner
                text_x = w - text_size[0] - 10
                text_y = text_size[1] + 10
                
                # Add background rectangle for better visibility
                cv2.rectangle(
                    marked_image,
                    (text_x - 5, text_y - text_size[1] - 5),
                    (text_x + text_size[0] + 5, text_y + 5),
                    (0, 0, 0),
                    -1
                )
                
                # Add text
                cv2.putText(
                    marked_image,
                    "BLURRY",
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 255),  # Red in BGR
                    2
                )
                
                # Add variance value in smaller text
                variance_text = f"Var: {result.get('variance', 0):.1f}"
                cv2.putText(
                    marked_image,
                    variance_text,
                    (text_x, text_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.7,
                    (0, 200, 255),  # Orange in BGR
                    1
                )
                
                result_images.append(marked_image)
            else:
                # Keep original image
                result_images.append(image)
        
        return result_images
    
    def handle_missing_compartments(self, image, compartment_boundaries, missing_marker_ids, metadata=None, 
                                vertical_constraints=None, marker_to_compartment=None, is_metadata_marker=False, markers=None):
        """
        Display the manual annotation dialog for missing compartments on the main thread.
        
        Args:
            image: The input image
            compartment_boundaries: List of already detected boundaries
            missing_marker_ids: List of marker IDs that need manual annotation
            metadata: Optional metadata dictionary 
            vertical_constraints: Optional tuple of (min_y, max_y) for compartment placement
            marker_to_compartment: Dictionary mapping marker IDs to compartment numbers
            is_metadata_marker: Whether metadata marker selection is needed
            markers: Dictionary of all detected ArUco markers {id: corners}
            
        Returns:
            Dictionary of dialog results including manually annotated boundaries and 
            metadata marker (if selected)
        """
        print("DEBUG: Running wrapper 'handle_missing_compartments' for 'handle_markers_and_boundaries')")
        return self.handle_markers_and_boundaries(
            image, 
            compartment_boundaries, 
            missing_marker_ids=missing_marker_ids,
            metadata=metadata, 
            vertical_constraints=vertical_constraints,
            marker_to_compartment=marker_to_compartment,
            markers=markers,  # Pass markers to unified handler
            initial_mode=1  # MODE_MISSING_BOUNDARIES
        )

    def handle_boundary_adjustments(self, image, detected_boundaries, missing_marker_ids=None, 
                            metadata=None, vertical_constraints=None, 
                            rotation_angle=0.0, corner_markers=None, marker_to_compartment=None, markers=None):
        """
        Display the boundary adjustment dialog for all images.
        
        Args:
            image: The input image
            detected_boundaries: List of already detected boundaries
            missing_marker_ids: Optional list of marker IDs that need manual annotation
            metadata: Optional metadata dictionary 
            vertical_constraints: Optional tuple of (min_y, max_y) for compartment placement
            rotation_angle: Current rotation angle of the image (degrees)
            corner_markers: Dictionary of corner marker positions {0: (x,y), 1: (x,y)...}
            marker_to_compartment: Dictionary mapping marker IDs to compartment numbers
            markers: Dictionary of all detected ArUco markers {id: corners}
            
        Returns:
            Dictionary containing adjusted boundaries and parameters
        """
        print("DEBUG: Running wrapper 'handle_boundary_adjustments' for 'handle_markers_and_boundaries')")
        return self.handle_markers_and_boundaries(
            image, 
            detected_boundaries, 
            missing_marker_ids=missing_marker_ids,
            metadata=metadata, 
            vertical_constraints=vertical_constraints,
            rotation_angle=rotation_angle,
            corner_markers=corner_markers,
            marker_to_compartment=marker_to_compartment,
            markers=markers,  # Pass markers to unified handler
            initial_mode=2,  # MODE_ADJUST_BOUNDARIES
            show_adjustment_controls=True  # Make adjustment controls visible by default
        )

    def handle_markers_and_boundaries(self, image, detected_boundaries, missing_marker_ids=None, 
                            metadata=None, vertical_constraints=None, 
                            rotation_angle=0.0, corner_markers=None, marker_to_compartment=None, 
                            initial_mode=2, markers=None, show_adjustment_controls=True,
                            on_apply_adjustments=None, image_path=None):
        """
        Unified handler for boundary adjustments and marker placement.

        Args:
            image: The input image
            detected_boundaries: List of already detected boundaries
            missing_marker_ids: Optional list of marker IDs that need manual annotation
            metadata: Optional metadata dictionary 
            vertical_constraints: Optional tuple of (min_y, max_y) for compartment placement
            rotation_angle: Current rotation angle of the image (degrees)
            corner_markers: Dictionary of corner marker positions {0: (x,y), 1: (x,y)...}
            marker_to_compartment: Dictionary mapping marker IDs to compartment numbers
            initial_mode: Initial dialog mode (0=metadata, 1=missing boundaries, 2=adjustment)
            markers: Dictionary of all detected ArUco markers {id: corners}
            show_adjustment_controls: Whether to show adjustment controls by default
                
        Returns:
            Dictionary containing dialog results (boundaries, markers, visualization)
        """
        print(f"DEBUG: HANDLE_MARKERS START - initial_mode: {initial_mode}")
        print(f"DEBUG: HANDLE_MARKERS START - missing_marker_ids: {missing_marker_ids}")

        # Map mode number to descriptive name for logging
        mode_names = {
            0: "metadata registration", 
            1: "marker placement",
            2: "boundary adjustment"
        }
        operation_name = mode_names.get(initial_mode, "unknown operation")
        
        self.logger.info(f"Handling {operation_name} dialog on main thread")
        
        print(f"DEBUG: Starting {operation_name} function")
        print(f"DEBUG: Current thread: {threading.current_thread().name}")
        print(f"DEBUG: Is main thread: {threading.current_thread() is threading.main_thread()}")
        print(f"DEBUG: Input parameters - image shape: {image.shape if hasattr(image, 'shape') else 'unknown'}")
        print(f"DEBUG: detected_boundaries count: {len(detected_boundaries)}")
        print(f"DEBUG: missing_marker_ids: {missing_marker_ids}")
        print(f"DEBUG: vertical_constraints: {vertical_constraints}")
        print(f"DEBUG: rotation_angle: {rotation_angle}")
        
        # Get the original image with no annotations if available
        original_unannotated_image = None
        if hasattr(self, 'visualization_cache'):
            print("DEBUG: Checking for original image in visualization_cache")
            original_unannotated_image = self.visualization_cache.get('current_processing', {}).get('original_image')
            if original_unannotated_image is not None:
                print("DEBUG: Found original unannotated image in visualization_cache")
                # Use the original image instead of any annotated version
                image = original_unannotated_image.copy()
        
        # If we have a small_image attribute, use that as a fallback
        if original_unannotated_image is None and hasattr(self, 'small_image'):
            print("DEBUG: Using small_image as original image")
            image = self.small_image.copy()
        
        # Get visualization (will be passed as boundaries_viz but lower priority)
        boundaries_viz = None
        if hasattr(self, 'visualization_cache'):
            boundaries_viz = self.visualization_cache.get('current_processing', {}).get('compartment_boundaries_viz')
            print(f"DEBUG: boundaries_viz from cache: {'Found' if boundaries_viz is not None else 'Not found'}")

        # Check if we're focused on marker placement (vs. just boundary adjustments)
        is_placing_markers = missing_marker_ids and len(missing_marker_ids) > 0
        
        # Check if we're specifically annotating a metadata marker (ID 24)
        is_metadata_marker = is_placing_markers and 24 in missing_marker_ids
        print(f"DEBUG: Is metadata marker annotation: {is_metadata_marker}")
        
        # CRITICAL FIX: Check if we've already annotated these missing markers
        if is_placing_markers and hasattr(self, 'visualization_cache') and 'annotation_status' in self.visualization_cache:
            already_annotated = self.visualization_cache['annotation_status'].get('annotated_markers', [])
            if already_annotated:
                print(f"DEBUG: Some markers were already annotated: {already_annotated}")
                
                # Check if all requested markers were already annotated
                if set(missing_marker_ids).issubset(set(already_annotated)):
                    print(f"DEBUG: All requested markers {missing_marker_ids} were already annotated")
                    
                    if initial_mode == 1:  # MODE_MISSING_BOUNDARIES
                        print("DEBUG: In missing boundaries mode - skipping dialog for already annotated markers")
                        return {}  # Return empty dict to signal "already done"
                    
                    print("DEBUG: In adjustment mode - still showing dialog despite markers being annotated")
                
                # Filter out already annotated markers from the list to process
                if isinstance(missing_marker_ids, list):
                    missing_marker_ids = [m for m in missing_marker_ids if m not in already_annotated]
                elif isinstance(missing_marker_ids, set):
                    missing_marker_ids = {m for m in missing_marker_ids if m not in already_annotated}
                
                print(f"DEBUG: Filtered missing_marker_ids after removing already annotated: {missing_marker_ids}")
        
        # Create the callback function for re-extracting boundaries
        def apply_adjustments_callback(adjustment_params):
            try:
                print(f"DEBUG: Received adjustment params: {adjustment_params}")
                
                # Create a temporary metadata dict with adjustments
                temp_metadata = {}
                if metadata:
                    temp_metadata.update(metadata)
                
                # Add boundary adjustments
                temp_metadata['boundary_adjustments'] = adjustment_params
                
                # Re-extract boundaries
                if hasattr(self, 'aruco_manager'):
                    print("DEBUG: Re-extracting compartment boundaries with adjustments")
                    
                    # Set a flag to avoid showing manual annotation dialog during re-extraction
                    temp_metadata['skip_annotation_dialog'] = True
                    
                    # Re-extract compartment boundaries
                    new_boundaries, new_viz = self.aruco_manager.extract_compartment_boundaries(
                        image, 
                        markers, 
                        compartment_count=self.config.get('compartment_count', 20),
                        smart_cropping=True,
                        parent_window=self.root,
                        metadata=temp_metadata
                    )
                    
                    print(f"DEBUG: Re-extracted {len(new_boundaries)} boundaries")
                    
                    # Return both boundaries and visualization
                    return {
                        'boundaries': new_boundaries,
                        'visualization': new_viz
                    }
                else:
                    print("DEBUG: aruco_manager not available")
                    return None
            except Exception as e:
                print(f"DEBUG: Error in apply_adjustments_callback: {str(e)}")
                import traceback
                print(f"DEBUG: {traceback.format_exc()}")
                return None
        
        try:
            # Thread check - hard fail if wrong
            if threading.current_thread() is not threading.main_thread():
                print("DEBUG: Not on main thread - raising exception!")
                raise RuntimeError(f"❌ {operation_name} called from non-main thread!")

            # Get visualization if available
            boundaries_viz = None
            if hasattr(self, 'visualization_cache'):
                print("DEBUG: visualization_cache exists")
                boundaries_viz = self.visualization_cache.get('current_processing', {}).get('compartment_boundaries_viz')
                print(f"DEBUG: boundaries_viz from cache: {'Found' if boundaries_viz is not None else 'Not found'}")
            
            # Get markers from aruco_manager if available
            if markers is None:
                markers = {}
                if hasattr(self, 'aruco_manager'):
                    markers = getattr(self.aruco_manager, 'cached_markers', {})
                    print(f"DEBUG: Got {len(markers)} markers from aruco_manager")

            # Check visualization cache for more complete marker data
            if hasattr(self, 'visualization_cache') and 'current_processing' in self.visualization_cache:
                cached_markers = self.visualization_cache['current_processing'].get('all_markers')
                if cached_markers and len(cached_markers) > len(markers):
                    markers = cached_markers
                    print(f"DEBUG: Using {len(markers)} markers from visualization_cache instead")
            
            # Ensure we have a valid marker_to_compartment mapping
            if marker_to_compartment is None:
                compartment_interval = metadata.get('compartment_interval', 
                                                self.config.get('compartment_interval', 1.0))
                marker_to_compartment = {
                    4+i: int((i+1) * compartment_interval) for i in range(20)
                }
                print(f"DEBUG: Created marker_to_compartment mapping: {marker_to_compartment}")

            # Create and show dialog
            print(f"DEBUG: About to create CompartmentRegistrationDialog with initial_mode={initial_mode}")
            print(f"DEBUG: self.root is None: {self.root is None}")
            print(f"DEBUG: image is None: {image is None}")
            print(f"DEBUG: detected_boundaries length: {len(detected_boundaries)}")
            
            # Set app reference on root for dialog access
            self.root.app = self
            
            dialog = CompartmentRegistrationDialog(
                self.root,
                image,
                detected_boundaries,
                missing_marker_ids,
                theme_colors=self.gui_manager.theme_colors,
                gui_manager=self.gui_manager,
                boundaries_viz=boundaries_viz,  # This will only be used as fallback
                original_image=getattr(self, 'original_image', None),
                output_format=self.config.get('output_format', 'png'),
                file_manager=self.file_manager,
                metadata=metadata,
                vertical_constraints=vertical_constraints,
                marker_to_compartment=marker_to_compartment,
                is_metadata_marker=is_metadata_marker,
                rotation_angle=rotation_angle,
                corner_markers=corner_markers,
                markers=markers,  # Pass all detected markers
                config=self.config,  # Pass the config
                on_apply_adjustments=on_apply_adjustments,  # Pass the callback function
                image_path=image_path
            )
            
            # Set the initial mode after creation
            dialog.current_mode = initial_mode
            dialog._update_mode_indicator()  # Update UI to reflect the mode
            
            print(f"DEBUG: CompartmentRegistrationDialog object created in mode {initial_mode}")

            # Show dialog and get results
            print("DEBUG: About to show dialog")
            result = dialog.show()
            print(f"DEBUG: Dialog result received: {result is not None}")
            print(f"DEBUG: HANDLE_MARKERS - dialog.show() returned")
            print(f"DEBUG: HANDLE_MARKERS - result type: {type(result)}")
            print(f"DEBUG: HANDLE_MARKERS - result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
            
            
            # Store markers that were actually annotated
            if result and is_placing_markers:
                print("DEBUG: Result received, updating visualization cache")
                if hasattr(self, 'visualization_cache'):
                    # Get existing annotated markers
                    existing = self.visualization_cache.get('annotation_status', {}).get('annotated_markers', [])
                    
                    # Determine which markers were actually annotated
                    actually_annotated = []
                    
                    # If result contains result_boundaries, use those to determine which markers were annotated
                    if 'result_boundaries' in result and isinstance(result['result_boundaries'], dict):
                        actually_annotated = list(result['result_boundaries'].keys())
                        print(f"DEBUG: Markers actually annotated according to result_boundaries: {actually_annotated}")
                    else:
                        # Fall back to missing_marker_ids as a best guess
                        actually_annotated = missing_marker_ids
                        print(f"DEBUG: No result_boundaries found, using missing_marker_ids: {actually_annotated}")
                    
                    # Combine with existing annotated markers
                    all_annotated = list(set(existing + list(actually_annotated)))
                    
                    # # Record which markers were annotated
                    # self.visualization_cache.setdefault('annotation_status', {})['annotated_markers'] = all_annotated
                    # print(f"DEBUG: Updated annotated markers in cache: {all_annotated}")
                    
                    # # Update visualization if available
                    # if 'final_visualization' in result:
                    #     print("DEBUG: Found final_visualization in result")
                    #     if hasattr(self, 'visualization_cache'):
                    #         print("DEBUG: Updating visualization_cache")
                    #         self.visualization_cache.setdefault('current_processing', {})['compartment_boundaries_viz'] = result['final_visualization']

            # Check for boundary adjustments in the result
            if result and isinstance(result, dict):
                # Check for boundary adjustment data
                boundary_fields = ['top_boundary', 'bottom_boundary', 'left_height_offset', 'right_height_offset']
                has_boundary_adjustments = any(field in result for field in boundary_fields)
                
                if has_boundary_adjustments:
                    print("DEBUG: Found boundary adjustments in dialog result")
                    # Create boundary adjustments dictionary
                    boundary_adjustments = {
                        'top_boundary': result.get('top_boundary'),
                        'bottom_boundary': result.get('bottom_boundary'),
                        'left_height_offset': result.get('left_height_offset', 0),
                        'right_height_offset': result.get('right_height_offset', 0)
                    }
                    
                    # Store the adjustments in metadata if provided
                    if metadata is not None:
                        metadata['boundary_adjustments'] = boundary_adjustments
                        print(f"DEBUG: Stored boundary adjustments in metadata: {boundary_adjustments}")
                    
                    # # Also update the visualization cache with any new visualization
                    # if 'final_visualization' in result:
                    #     self.visualization_cache.setdefault('current_processing', {})['compartment_boundaries_viz'] = result['final_visualization']
                    #     print("DEBUG: Updated visualization in cache with adjusted version")

            print(f"DEBUG: HANDLE_MARKERS END - returning result? {result is not None}")
            return result or {}

        except Exception as e:
            print(f"❌ Exception in {operation_name}: {e}")
            print(f"DEBUG: Exception type: {type(e).__name__}")
            traceback.print_exc()
            if hasattr(self, 'logger'):
                self.logger.error(f"Error in {operation_name}: {e}")
                self.logger.error(traceback.format_exc())
            return {}  # Return empty dict on error

    def save_compartments(self, compartments, base_filename, metadata=None):
        """
        Save extracted compartment images to disk using FileManager.
        
        Args:
            compartments: List of compartment images
            base_filename: Base filename to use for compartment images
            metadata: Optional metadata for naming compartments
            
        Returns:
            Number of successfully saved compartments
        """
        if not metadata or not all(key in metadata for key in ['hole_id', 'depth_from', 'depth_to']):
            self.logger.error("Missing required metadata for saving compartments")
            return 0
            
        # Extract metadata values
        hole_id = metadata['hole_id']
        depth_from = int(metadata['depth_from'])
        depth_to = int(metadata['depth_to'])
        
        # Calculate depth increment per compartment using the configured interval
        compartment_interval = self.config['compartment_interval']
        
        # Check for blur if enabled
        if self.config['enable_blur_detection']:
            blur_results = self.detect_blur_in_compartments(compartments, base_filename)
            
            # Add blur indicators if enabled
            if self.config['flag_blurry_images']:
                compartments = self.add_blur_indicators(compartments, blur_results)
                
            # Save blur visualizations if enabled
            if self.config['save_blur_visualizations']:
                for result in blur_results:
                    if 'visualization' in result:
                        i = result['index']
                        
                        # Calculate depth for this compartment
                        comp_depth_from = depth_from + (i * compartment_interval)
                        comp_depth_to = comp_depth_from + compartment_interval
                        compartment_depth = int(comp_depth_to)
                        
                        # Save blur analysis visualization
                        self.file_manager.save_blur_analysis(
                            result['visualization'],
                            hole_id,
                            compartment_depth
                        )
        
        # Count saved compartments
        saved_count = 0
        
        # Process each compartment
        for i, compartment in enumerate(compartments):
            try:
                # Calculate depth for this compartment
                comp_depth_from = depth_from + (i * compartment_interval)
                comp_depth_to = comp_depth_from + compartment_interval
                compartment_depth = int(comp_depth_to)
                
                # Save the compartment image
                self.file_manager.save_compartment(
                    compartment,
                    hole_id,
                    compartment_depth,
                    False,  # has_data flag
                    self.config['output_format']
                )
                
                saved_count += 1
                
            except Exception as e:
                self.logger.error(f"Error saving compartment {i+1}: {str(e)}")
        
        return saved_count
    
    def create_visualization_image(self, original_image, processing_images):
        """
        Create a visualization showing original image alongside processing steps.
        
        Args:
            original_image: The original input image
            processing_images: List of (step_name, image) tuples
            
        Returns:
            Visualization collage as numpy array
        """
        import cv2
        import numpy as np
        
        # Helper function to ensure 3-channel image
        def ensure_3channel(img):
            if len(img.shape) == 2:  # Grayscale
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif len(img.shape) == 3:
                if img.shape[2] == 1:  # Single channel
                    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 3:  # Already color
                    return img
                elif img.shape[2] == 4:  # RGBA
                    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            
            # Fallback - create a blank color image
            return np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

        # Determine the number of images in the collage
        num_images = 1 + len(processing_images)
        
        # Determine layout
        if num_images <= 3:
            rows, cols = 1, num_images
        else:
            rows = 2
            cols = (num_images + 1) // 2  # Ceiling division
        
        # Resize images to a standard height
        target_height = 600
        resized_images = []
        
        # Resize original image - ensure 3 channels
        original_image = ensure_3channel(original_image)
        h, w = original_image.shape[:2]
        scale = target_height / h
        resized_original = cv2.resize(original_image, (int(w * scale), target_height))
        resized_images.append(("Original Image", resized_original))
        
        # Resize processing images
        for name, img in processing_images:
            # Ensure 3 channels before resizing
            img = ensure_3channel(img)
            h, w = img.shape[:2]
            scale = target_height / h
            resized = cv2.resize(img, (int(w * scale), target_height))
            resized_images.append((name, resized))
        
        # Calculate canvas size
        max_width = max(img[1].shape[1] for img in resized_images)
        canvas_width = max_width * cols + 20 * (cols + 1)  # Add padding
        canvas_height = target_height * rows + 60 * (rows + 1)  # Add space for titles
        
        # Create canvas
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
        
        # Place images on canvas
        for i, (name, img) in enumerate(resized_images):
            row = i // cols
            col = i % cols
            
            # Calculate position
            x = 20 + col * (max_width + 20)
            y = 60 + row * (target_height + 60)
            
            # Place image
            h, w = img.shape[:2]
            canvas[y:y+h, x:x+w] = img
            
            # Add title
            cv2.putText(canvas, name, (x, y - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        
        # Add main title
        cv2.putText(canvas, "Chip Tray Extraction Process", 
                (canvas_width // 2 - 200, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        
        return canvas
        
    def debug_threads(label="Thread check"):
        """Print active threads, pending tkinter after() calls, and current stack trace."""
        
        current = threading.current_thread()
        print(f"\n===== {label} =====")
        print(f"Current thread: {current.name} (main={current is threading.main_thread()})")
        
        # List all running threads
        print(f"Active threads ({threading.active_count()}):")
        for t in threading.enumerate():
            print(f"  - {t.name} {'[DAEMON]' if t.daemon else ''}")
        
        # Check tkinter after() calls if we can access root
        try:
            from tkinter import _default_root as root
            if root:
                after_ids = root.tk.call('after', 'info')
                if after_ids:
                    print(f"Pending after() calls ({len(after_ids)}):")
                    for after_id in after_ids:
                        script = root.tk.call('after', 'info', after_id)
                        print(f"  - ID {after_id}: {script[:50]}{'...' if len(script) > 50 else ''}")
        except Exception as e:
            print(f"Could not check tkinter after() calls: {e}")
        
        # Print current thread's stack trace
        print("Current stack trace:")
        for line in traceback.format_stack()[:-1]:  # Exclude this function call
            print(f"  {line.strip()}")
        print("=" * (len(label) + 14))

    def update_last_metadata(self, hole_id, depth_from, depth_to, compartment_interval=1):
        """
        Update the last successful metadata for use with increment functionality.
        
        Args:
            hole_id: The hole ID
            depth_from: Starting depth
            depth_to: Ending depth
            compartment_interval: Interval between compartments (1 or 2 meters)
        """
        self.last_successful_metadata = {
            'hole_id': hole_id,
            'depth_from': depth_from,
            'depth_to': depth_to,
            'compartment_interval': compartment_interval
        }
        self.logger.info(f"Updated last metadata: {hole_id} {depth_from}-{depth_to}m (interval: {compartment_interval}m)")
    
    def get_incremented_metadata(self):
        """
        Generate metadata based on the last successfully processed tray.
        
        Returns:
            dict: New metadata with incremented depths
        """
        if not self.last_successful_metadata.get('hole_id'):
            # No previous metadata available
            return None
        
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

    def _check_configuration(self) -> bool:
        """
        Check if the application is properly configured.
        
        Returns:
            True if configured, False if first run needed
        """
        # Check for essential configuration
        local_path = self.config_manager.get('local_folder_path')
        
        # Check if local path exists
        if not local_path or not os.path.exists(local_path):
            self.logger.info("No valid local folder path configured - first run needed")
            return False
            
        # Check if FileManager can be initialized with the path
        try:
            test_fm = FileManager(local_path)
            # Check if key directories exist
            if not os.path.exists(test_fm.base_dir):
                self.logger.info("Base directory doesn't exist - first run needed")
                return False
        except Exception as e:
            self.logger.error(f"Error checking configuration: {e}")
            return False
            
        return True
    
    def _apply_first_run_config(self, config: dict):
        """
        Apply configuration from first run dialog.
        
        Args:
            config: Configuration dictionary from first run dialog
        """
        # Save all configuration values
        if config.get('local_folder_path'):
            self.config_manager.set('local_folder_path', config['local_folder_path'])
            self.config_manager.set('FileManager_output_directory', config['local_folder_path'])
            
        if config.get('shared_folder_path'):
            self.config_manager.set('shared_folder_path', config['shared_folder_path'])
            
        if config.get('storage_type'):
            self.config_manager.set('storage_type', config['storage_type'])
            
        # Save specific folder paths if provided
        folder_paths = config.get('folder_paths', {})
        for key, path in folder_paths.items():
            self.config_manager.set(key, path)
            
        # Re-initialize FileManager with new path
        # TODO - check if we need to add an way to re-initialize FileManager if the configuration is changed in the mainGUI
        self.file_manager = FileManager(config['local_folder_path'], config_manager=self.config_manager)
        
        self.logger.info("First run configuration applied successfully")



    def run(self):
        """Run the application."""
        # Make sure root window is showing
        self.root.deiconify()
        
        # Set folder path in GUI if it was provided via command line
        if hasattr(self, 'folder_path') and self.folder_path:
            self.main_gui.folder_var.set(self.folder_path)
            # Trigger validation if needed
            if hasattr(self.main_gui, '_update_folder_color'):
                self.main_gui._update_folder_color()
        
        # Start the main event loop
        self.root.mainloop()

# ===========================================
# Application Entry Point
# ===========================================


def main():
    """Main entry point for the application."""
    try:
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description="GeoVue Chip Tray Processor")
        parser.add_argument('--version', action='version', version=f'GeoVue v{__version__}')
        parser.add_argument('--folder', type=str, help='Process all images in the specified folder')
        args = parser.parse_args()
        
        # Create the application
        app = ChipTrayApp()
        
        # If folder argument was provided, pre-populate the folder field in the GUI
        # but don't process it automatically
        if args.folder and os.path.isdir(args.folder):
            logger.info(f"Pre-populating folder field with: {args.folder}")
            app.folder_path = args.folder  # Store for later use in the GUI
        
        # Run the application - will always show the GUI
        app.run()
        
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}")
        logger.critical(traceback.format_exc())
        
        # Show error dialog if possible
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "Error", 
                f"An unexpected error occurred:\n\n{str(e)}\n\nPlease check the log file for details."
            )
            root.destroy()
        except:
            pass
        
        # Exit with error code
        sys.exit(1)

if __name__ == "__main__":
    main()