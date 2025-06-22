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
from typing import List, Dict
import re
import shutil
import pandas as pd
from datetime import datetime
import time
from pillow_heif import register_heif_opener # TODO - Add in HEIC support and conversion to JPG
from gui.first_run_dialog import FirstRunDialog
from resources import get_logo_path
from processing.blur_detector import BlurDetector


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
logger.setLevel(logging.DEBUG) 
logger.info(f"Starting GeoVue v{__version__}")


# ===================================================
# Module-specific logging levels DEBUG, INFO, WARNING, ERROR 
# ===================================================
# Turn off or down debug logs for noisy modules
logging.getLogger('processing.aruco_manager').setLevel(logging.WARNING)
logging.getLogger('processing.blur_detector').setLevel(logging.WARNING)  
logging.getLogger('core.tesseract_manager').setLevel(logging.WARNING)
logging.getLogger('gui.duplicate_handler').setLevel(logging.WARNING)
logging.getLogger('gui.qaqc_manager').setLevel(logging.WARNING)
logging.getLogger('gui.main_gui').setLevel(logging.WARNING)
logging.getLogger('gui.compartment_registration_dialog').setLevel(logging.WARNING)
logging.getLogger('gui.first_run_dialog').setLevel(logging.DEBUG)
logging.getLogger('gui.dialog_helper').setLevel(logging.DEBUG)

# Suppress third-party debug logs (set to WARNING or ERROR as needed)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("PIL.Image").setLevel(logging.ERROR)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING) 


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
    JSONRegisterManager
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

        # Initialize update checker early to catch first runs
        self.update_checker = RepoUpdater()

        
        # ===================================================
        # STEP 3: CREATE ROOT WINDOW
        # ===================================================
        # Create the root window (needed for all GUI components)
        self.root = tk.Tk()
        self.root.title("GeoVue")
        
        # Make root window small and centered
        self.root.geometry("100x100")
        
        # Center the small root window
        self.root.update_idletasks()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - 100) // 2
        y = (screen_height - 100) // 2
        self.root.geometry(f"100x100+{x}+{y}")
        
        
        # ===================================================
        # STEP 3.5: CHECK FOR UPDATES BEFORE FIRST RUN
        # ===================================================
        # Initialize minimal GUI manager for dialogs
        self.gui_manager = GUIManager(self.file_manager, self.config_manager)
        DialogHelper.set_gui_manager(self.gui_manager)

        # Set background to match theme
        self.root.configure(bg=self.gui_manager.theme_colors["background"])
        
        # Check for updates immediately (blocking)
        self._perform_startup_version_check()
        
        # ===================================================
        # STEP 4: HANDLE FIRST RUN SETUP
        # ===================================================
        # Check if first run
        if self.config_manager.is_first_run() or not self._check_configuration():
            # Ensure we're on the main thread before creating dialog
            self.logger.debug(f"Current thread: {threading.current_thread().name}")
            if threading.current_thread() is not threading.main_thread():
                self.logger.error("CRITICAL: Attempting to create FirstRunDialog from non-main thread!")
                self.logger.error(f"Current thread: {threading.current_thread().name}")
                raise RuntimeError("FirstRunDialog must be created on the main thread")
            
            # Initialize minimal GUI manager for dialog
            self.gui_manager = GUIManager(self.file_manager, self.config_manager)
            
            # Show first run dialog
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
            output_dir = self.config_manager.get('local_folder_path')

            # Re-initialize FileManager with new path
            if output_dir and output_dir != self.file_manager.base_dir:
                self.file_manager = FileManager(base_dir=output_dir, config_manager=self.config_manager)
            else:
                # ADD THIS: If we didn't recreate FileManager, still update shared paths
                self.file_manager.initialize_shared_paths()

        # ===================================================
        # STEP 5: INITIALIZE GUI COMPONENTS
        # ===================================================
        # Initialize GUI manager (if not already created during first run)
        if not hasattr(self, 'gui_manager'):
            self.gui_manager = GUIManager(self.file_manager, self.config_manager)
            DialogHelper.set_gui_manager(self.gui_manager)
        

        # Initialize Register Manager using FileManager
        register_base_path = self.file_manager.get_shared_path('register', create_if_missing=True)
        if not register_base_path:
            # Try to prompt user for the register path
            self.logger.info("No Chip Tray Register path configured, prompting user...")
            
            register_base_path = self.file_manager.prompt_for_shared_path(
                'register',
                'Select Chip Tray Register Folder',
                'The Chip Tray Register folder is not configured. Would you like to select it now?',
                is_file=False
            )
            
            if register_base_path:
                # User selected a path, try to initialize register manager
                self.logger.info(f"User selected register path: {register_base_path}")
                self.register_manager = JSONRegisterManager(str(register_base_path), self.logger)
            else:
                # User cancelled - show error and exit
                self.logger.error("User cancelled register path selection")
                DialogHelper.show_message(
                    self.root,
                    self.t("Configuration Error"),
                    self.t("Chip Tray Register path must be configured. Please restart and configure the shared folder path."),
                    message_type="error"
                )
                self.root.destroy()
                sys.exit(1)
        else:
            # Path exists, initialize normally
            self.register_manager = JSONRegisterManager(str(register_base_path), self.logger)
        
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
        # Load saved language preference from ConfigManager
        language = self.config_manager.get("language")
        if language:
            self.translator.set_language(language)
            self.logger.info(f"Loaded language preference from config: {language}")
        
        # Complete initialization
        self.processing_complete = False
        
        self.logger.info("GeoVue initialization complete")


 
    def _perform_startup_version_check(self):
        """
        Perform version check during startup.
        Shows a modal notification if update is available during first run,
        non-modal otherwise.
        """
        try:
            # Get version info
            version_info = self.update_checker.compare_versions()
            
            self.logger.info(f"Version check - Local: {version_info['local_version']}, "
                            f"GitHub: {version_info['github_version']}, "
                            f"Update available: {version_info['update_available']}")
            
            # Only show dialog if update is available
            if version_info['update_available'] and not version_info.get('error'):
                # Check if we're in first run mode
                is_first_run = self.config_manager.is_first_run() or not self._check_configuration()
                
                # Create dialog using DialogHelper - only valid parameters
                dialog = DialogHelper.create_dialog(
                    self.root,
                    self.t("Update Available"),
                    modal=is_first_run,  # Modal during first run, non-modal otherwise
                    topmost=True
                )
                
                # Prevent resizing
                dialog.resizable(False, False)
                
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
                    color=self.gui_manager.theme_colors["accent_green"],
                    command=dialog.destroy
                )
                ok_btn.pack()

                # Update dialog to calculate correct size
                dialog.update_idletasks()
                
                # Now center the dialog with size constraints
                DialogHelper.center_dialog(
                    dialog,
                    self.root,
                    max_width=450,
                    max_height=200
                )

                # FORCE dialog to absolute top during first run
                if is_first_run:
                    dialog.lift()
                    dialog.attributes('-topmost', True)
                    dialog.focus_force()
                    # Update to ensure it's visible
                    dialog.update()
                    # Lift again to be sure
                    dialog.lift()
                
                # If not first run, auto-close after 10 seconds
                if not is_first_run:
                    dialog.after(10000, lambda: dialog.destroy() if dialog.winfo_exists() else None)
                
                # During first run, wait for the dialog to be closed
                if is_first_run:
                    dialog.wait_window()
            
        except Exception as e:
            # Don't let version check errors prevent app startup
            self.logger.warning(f"Version check failed during startup: {str(e)}")
            # Don't show error dialog during startup - just log it

    def _set_application_icon(self):
        """Set the application icon for all windows."""
        try:
            # Try using the resources package helper function first
            try:
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
        
        # Initialize visualization cache for sharing images between components
        self.visualization_cache = {}
    
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
        self.duplicate_handler = DuplicateHandler(file_manager=self.file_manager)
        self.duplicate_handler.parent = self  # Give it access to visualization_cache
        self.duplicate_handler.logger = logging.getLogger(__name__)
        self.duplicate_handler.root = self.root  # Set root for dialog creation
        
        # Initialize QAQC manager
        self.qaqc_manager = QAQCManager(
            file_manager=self.file_manager,
            translator_func=self.t,
            config_manager=self.config_manager,
            app=self,
            logger=self.logger,
            register_manager=self.register_manager
        )

        # Main application GUI - pass the existing root window
        self.main_gui = MainGUI(self)


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
        5. Show the compartment registration dialog.
        6. On completion of the compartment registration dialog:
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

        self.logger.debug(f" Starting to process image: {image_path}")
        self.logger.debug(f" Clearing Caches")

        # Clear any cached visualization data from previous image
        self.clear_visualization_cache()

        # Initialize or get progress queue for UI updates
        if not hasattr(self, 'progress_queue'):
            self.progress_queue = queue.Queue()
            self.logger.debug(" Created new progress_queue")
        else:
            self.logger.debug(" Using existing progress_queue")

        # Initialize message collection for this image
        processing_messages = []
        
        def add_progress_message(message, progress=None, message_type="info"):
            """Helper to collect and route status messages with optional type formatting."""
            processing_messages.append((message, progress))
            self.logger.info(message)

            # GUI Status Update
            if hasattr(self, 'main_gui') and hasattr(self.main_gui, 'update_status'):
                try:
                    self.main_gui.update_status(message, status_type=message_type)
                except Exception as e:
                    self.logger.debug(f"Could not update GUI status: {e}")

            # Optional: update progress bar or external queue
            if hasattr(self, 'progress_queue') and self.progress_queue:
                self.progress_queue.put((message, progress))

        try:
            # ===================================================
            # STEP 1: LOAD AND VALIDATE IMAGE
            # ===================================================
            # Try to read the image using PIL first (more format support)
            try:
                self.logger.debug(" Attempting to read image with PIL")
                from PIL import Image as PILImage
                
                # Open with PIL
                pil_img = PILImage.open(image_path)
                self.logger.debug(f" PIL image mode: {pil_img.mode}, size: {pil_img.size}")
                
                # Convert to numpy array for OpenCV processing
                original_image = np.array(pil_img)
                self.logger.debug(f" Converted to numpy array, shape: {original_image.shape}")
                
                # Convert RGB to BGR for OpenCV compatibility if needed
                if len(original_image.shape) == 3 and original_image.shape[2] == 3:
                    self.logger.debug(" Converting RGB to BGR for OpenCV compatibility")
                    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
                    
                self.logger.info(f"Successfully read image with PIL: {image_path}")
                
            except Exception as e:
                self.logger.warning(f"Failed to read image with PIL, trying OpenCV: {str(e)}")
                self.logger.debug(f" PIL reading failed with error: {str(e)}")
                
                # Fallback to OpenCV if PIL fails
                self.logger.debug(" Attempting to read image with OpenCV")
                original_image = cv2.imread(image_path)
                if original_image is None:
                    error_msg = f"Failed to read image with both PIL and OpenCV: {image_path}"
                    self.logger.debug(f" {error_msg}")
                    self.logger.error(error_msg)
                    add_progress_message(error_msg, None, message_type="error")
                    return False
                else:
                    self.logger.debug(f" OpenCV successfully read image, shape: {original_image.shape}")
            
            # Update progress with image info
            base_name = os.path.basename(image_path)
            status_msg = f"Processing Image: {base_name}"
            self.logger.info(status_msg)
            add_progress_message(status_msg, None, message_type="info")
            
            # ===================================================
            # STEP 2: CREATE DOWNSAMPLED IMAGE FOR FASTER PROCESSING
            # ===================================================
            # For processing speed create a temporary downscaled image
            h, w = original_image.shape[:2]
            original_pixels = h * w
            target_pixels = 2000000  # Target ~2 million pixels (e.g., 1414x1414)
            self.logger.debug(f" Original image dimensions: {w}x{h}, total pixels: {original_pixels}")
            
            if original_pixels > target_pixels:
                scale = (target_pixels / original_pixels) ** 0.5
                new_width = int(w * scale)
                new_height = int(h * scale)
                self.logger.debug(f" Resizing image with scale factor: {scale:.4f}")
                
                # Create downsampled image  
                small_image = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                self.logger.debug(f" Resized image to {new_width}x{new_height}")
                
                add_progress_message(f"Resized image from {w}x{h} to {new_width}x{new_height} for processing", None)
            else:
                # Image is already small enough
                small_image = original_image.copy()
                self.logger.info(f"Image already small ({w}x{h}), using as is for processing")
                add_progress_message(f"Image already small ({w}x{h}), using as is for processing", None)
            
            # Store the small image as an instance variable so it's accessible to the metadata dialog
            self.small_image = small_image.copy()
            
            # ===================================================
            # STEP 3: CHECK IF FILE WAS PREVIOUSLY PROCESSED
            # ===================================================
            # Check if this file has been previously processed (to extract metadata from filename)
            previously_processed = self.file_manager.check_original_file_processed(image_path)
            self.logger.debug(f" Previously processed status: {previously_processed}")

            # Use existing metadata (e.g., from filename) or initialize empty
            metadata = getattr(self, 'metadata', {})
            self.logger.debug(f" Initial metadata: {metadata}")

            # ===================================================
            # STEP 4: DETECT ARUCO MARKERS
            # ===================================================
            # IMPORTANT: Always detect markers regardless of metadata source
            add_progress_message("Detecting ArUco markers...", None)

            self.logger.debug(" Starting ArUco marker detection")
            # Use ArucoManager to detect and improve marker detection
            markers = self.aruco_manager.improve_marker_detection(small_image)
            self.logger.debug(f" Detected {len(markers)} ArUco markers: {sorted(markers.keys())}")
            
            # Report marker detection status
            expected_markers = set(self.config['corner_marker_ids'] + 
                                self.config['compartment_marker_ids'])
            
            # Only expect metadata markers if OCR is enabled and available
            if self.config.get('enable_ocr', True) and self.tesseract_manager.is_available:
                expected_markers.update(self.config['metadata_marker_ids'])
                self.logger.debug(" OCR is enabled - including metadata markers in expected set")
            else:
                self.logger.debug(" OCR is disabled - excluding metadata markers from expected set")
            
            detected_markers = set(markers.keys())
            missing_markers = expected_markers - detected_markers
            missing_marker_ids = list(missing_markers)  # Convert to list for later use
            
            self.logger.debug(f" Expected markers: {sorted(expected_markers)}")
            self.logger.debug(f" Missing markers: {sorted(missing_markers)}")

            # Check if we should remove metadata marker from missing list
            if not self.config.get('enable_ocr', True) or not self.tesseract_manager.is_available:
                if 24 in missing_marker_ids:
                    missing_marker_ids.remove(24)
                    self.logger.debug(" Removed marker 24 from missing list (OCR disabled or unavailable)")
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
                self.logger.debug(" Attempting to correct image skew on small image")
                # Use ArucoManager for image correction
                corrected_small_image, rotation_matrix, rotation_angle = self.aruco_manager.correct_image_skew(
                    small_image, markers, return_transform=True
                )
                
                self.logger.debug(f" Skew correction applied, rotation angle: {rotation_angle:.2f}°")
                
                if corrected_small_image is not small_image:  # If correction was applied
                    self.logger.debug(" Image was actually modified during skew correction")
                    # Re-detect markers on corrected image
                    markers = self.aruco_manager.improve_marker_detection(corrected_small_image)
                    self.logger.debug(f" Re-detected {len(markers)} markers after correction")
                    small_image = corrected_small_image  # Update small image for further processing
                    
                    self.logger.info(f"Image orientation corrected, angle: {rotation_angle:.2f}°, re-detected {len(markers)} markers")
                    add_progress_message(f"Corrected image orientation (angle: {rotation_angle:.2f}°), re-detected {len(markers)} markers", None)
    
                else:
                    self.logger.debug(" No skew correction needed (or applied)")
            except Exception as e:
                self.logger.warning(f"Skew correction failed: {str(e)}")
                self.logger.error(traceback.format_exc())
                self.logger.debug(f" Skew correction failed with error: {str(e)}")
                # Continue with uncorrected image

            # Recalculate missing markers since skew correction may have found more
            detected_markers = set(markers.keys())
            missing_markers = expected_markers - detected_markers
            missing_marker_ids = list(missing_markers)  # Update the list
            
            self.logger.debug(f" After skew correction - detected markers: {sorted(detected_markers)}")
            self.logger.debug(f" After skew correction - missing markers: {sorted(missing_marker_ids)}")
            
            if missing_marker_ids:
                self.logger.info(f"After correction, still missing markers: {sorted(missing_marker_ids)}")
                if self.progress_queue:
                    add_progress_message(f"Still missing markers after correction: {sorted(missing_marker_ids)}", None)
    

            # ===================================================
            # STEP 5.5: ESTIMATE IMAGE SCALE FROM MARKERS AND CORRECT SKEWED MARKERS TO SQUARES
            # ===================================================
            
            # Correct skewed markers before scale estimation - Reconstructs markers to squares using two valid edges - sometimes the detection is slightly off and you get deformed squares.
            if markers and len(markers) > 0:
                add_progress_message("Correcting skewed markers...", None)
                try:
                    corrected_markers = self.aruco_manager.correct_skewed_markers(
                        markers,
                        tolerance_pixels=5.0,  # 5 pixel tolerance
                        max_correction_angle=30.0  # Maximum 30 degree correction
                    )
                    
                    # Count how many markers were corrected
                    corrected_count = sum(1 for mid in markers 
                                        if not np.array_equal(markers[mid], corrected_markers[mid]))
                    
                    if corrected_count > 0:
                        self.logger.info(f"Corrected {corrected_count} skewed markers")
                        add_progress_message(f"Corrected {corrected_count} skewed markers", None)
                        markers = corrected_markers
                        
                        # Update cached markers
                        self.aruco_manager.cached_markers = markers.copy()
                        
                        # Update visualization cache
                        if hasattr(self, 'visualization_cache'):
                            self.visualization_cache.setdefault('current_processing', {})['all_markers'] = markers
                            
                except Exception as e:
                    self.logger.error(f"Error correcting skewed markers: {str(e)}")

            scale_data = {}
            self.current_scale_data = None  # Initialize

            if markers and len(markers) >= self.config.get('scale_min_markers_required', 4):
                add_progress_message("Estimating image scale from markers...", None)

                try:
                    # Estimate scale using detected markers
                    scale_data = self.aruco_manager.estimate_image_scale(
                        markers, 
                        use_corner_markers=self.config.get('use_corner_markers_for_scale', False)
                    )

                    if scale_data.get('scale_px_per_cm'):
                        # Store as instance variable for later use
                        self.current_scale_data = scale_data

                        scale_msg = f"Image scale: {scale_data['scale_px_per_cm']:.1f} px/cm"
                        if scale_data.get('image_width_cm'):
                            scale_msg += f" (image width: {scale_data['image_width_cm']:.1f} cm)"
                        scale_msg += f" - Confidence: {scale_data.get('confidence', 0):.0%}"

                        self.logger.info(scale_msg)
                        add_progress_message(scale_msg, None)

                        # Store in visualization cache for dialog access
                        if hasattr(self, 'visualization_cache'):
                            self.visualization_cache.setdefault('current_processing', {})['scale_data'] = scale_data
                    else:
                        self.logger.warning("Could not estimate image scale from markers")

                except Exception as e:
                    self.logger.error(f"Error estimating image scale: {str(e)}")


            # ===================================================
            # STEP 6: ANALYZE COMPARTMENT BOUNDARIES (WITHOUT UI)
            # ===================================================
            # Extract initial compartment boundaries without dialog
            if self.progress_queue:
                add_progress_message("Analyzing compartment boundaries...", None)


            try:
                # For debugging
                self.logger.debug(f"Analyzing {len(markers)} markers for compartment boundaries: {sorted(markers.keys())}")
                self.logger.debug(f" Starting compartment boundary analysis with {len(markers)} markers")
                
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
                    self.logger.debug(f" Combined missing markers: {missing_marker_ids}")
                
                # Store top/bottom boundaries for later use
                if vertical_constraints:
                    self.top_y, self.bottom_y = vertical_constraints
                    
                self.logger.debug(f" Extracted {len(boundaries)} compartment boundaries")
                if boundaries:
                    self.logger.debug(f" First boundary coordinates: {boundaries[0]}")
                    self.logger.debug(f" Last boundary coordinates: {boundaries[-1]}")

                # Simply check if boundaries were found
                if not boundaries:
                    error_msg = "Failed to extract compartment boundaries"
                    self.logger.debug(f" {error_msg}")
                    self.logger.error(error_msg)
                    add_progress_message(error_msg, None)
    
                    return False
                
            except Exception as e:
                error_msg = f"Error in compartment boundary extraction: {str(e)}"
                self.logger.debug(f" {error_msg}")
                self.logger.debug(f" {traceback.format_exc()}")
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


            self.logger.debug(" Starting metadata region extraction with TesseractManager")

            # Make sure TesseractManager has access to necessary references
            self.logger.debug(" Setting up TesseractManager references")
            self.tesseract_manager.file_manager = self.file_manager
            self.tesseract_manager.extractor = self

            # Check if we already have metadata from the filename
            if previously_processed:
                # Use the metadata from the filename and skip OCR extraction
                metadata = previously_processed
                self.logger.debug(f" Using metadata from filename: {metadata}")
                self.logger.info(f"Using metadata from filename: {metadata}")
                if self.progress_queue:
                    add_progress_message(f"Using metadata from filename: Hole ID={metadata.get('hole_id')}, Depth={metadata.get('depth_from')}-{metadata.get('depth_to')}m", None)
                
                # Create minimal OCR metadata for dialog display
                self.logger.debug(" Creating metadata structure for dialog display")
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
                self.logger.debug(" No existing metadata, checking OCR status")
                if self.progress_queue:
                    add_progress_message("Preparing metadata input...", None)
    
                
                # Check if OCR is enabled and available
                if self.config['enable_ocr'] and self.tesseract_manager.is_available:
                    self.logger.debug(" OCR is enabled and Tesseract is available")
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
                    
                    self.logger.debug(f" OCR results - confidence: {composite_confidence:.1f}%, has complete data: {composite_has_data}")
                    self.logger.debug(f" OCR extracted hole_id: {ocr_metadata.get('hole_id')}")
                    self.logger.debug(f" OCR extracted depth_from: {ocr_metadata.get('depth_from')}")
                    self.logger.debug(f" OCR extracted depth_to: {ocr_metadata.get('depth_to')}")
                    
                    # If the composite method didn't yield good results, just log it
                    if not composite_has_data or composite_confidence < self.config['ocr_confidence_threshold']:
                        self.logger.debug(f" OCR confidence too low ({composite_confidence:.1f}%), will prompt user for metadata")
                        self.logger.info(f"OCR confidence too low ({composite_confidence:.1f}%), will prompt user for metadata")

                    # Log OCR results
                    ocr_log_msg = f"OCR Results: Confidence={ocr_metadata.get('confidence', 0):.1f}%"
                    if ocr_metadata.get('hole_id'):
                        ocr_log_msg += f", Hole ID={ocr_metadata['hole_id']}"
                    if ocr_metadata.get('depth_from') is not None and ocr_metadata.get('depth_to') is not None:
                        ocr_log_msg += f", Depth={ocr_metadata['depth_from']}-{ocr_metadata['depth_to']}"
                    
                    self.logger.debug(f" {ocr_log_msg}")
                    self.logger.info(ocr_log_msg)
                    add_progress_message(ocr_log_msg, None)
                else:
                    # OCR is disabled, create a minimal metadata structure
                    self.logger.debug(" OCR is disabled, creating minimal metadata structure")
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
                            
                            self.logger.debug(f" Extracted metadata from filename: {filename_metadata}")
                    
                    if 24 in missing_marker_ids:
                        missing_marker_ids.remove(24)
                        self.logger.debug(" Removed marker 24 from missing list since OCR is disabled")
                        self.logger.info("Removed metadata marker 24 from missing list (OCR disabled)")

            # ===================================================
            # STEP 8: ALWAYS SHOW COMPARTMENT REGISTRATION DIALOG
            # ===================================================
            # ALWAYS show the compartment registration dialog
            if self.progress_queue:
                add_progress_message("Opening compartment registration dialog...", None)


            self.logger.debug(" Preparing to show compartment registration dialog")

            # # Get visualization for dialog - use boundaries_viz if available
            compartment_viz = small_image.copy()
            
            try:
                # Create unified registration dialog
                self.logger.debug(" Creating combined registration dialog")

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
                self.logger.debug(f" Missing marker IDs before dialog: {missing_marker_ids}")
                self.logger.debug(f" Type of missing_marker_ids: {type(missing_marker_ids)}")
                
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
                    image_path=image_path,
                    scale_data=self.current_scale_data if hasattr(self, 'current_scale_data') else None
                )

                self.logger.debug(" Registration dialog completed")

                # Process the result based on complete metadata and boundaries
                if result:
                    if result.get('quit', False):
                        # User chose to quit processing
                        self.logger.debug(" User quit processing")
                        self.logger.info("User stopped processing")
                        add_progress_message("Processing stopped by user", None)
                        
                        
                        # Set a flag to stop processing remaining images
                        self.processing_complete = True
                        return False  # Return early to stop processing
                    
                    self.logger.debug(f" Result dict: {result}")
                    if result.get('rejected', False):
                        # Handle rejected case using centralized handler
                        self.logger.debug(" Image rejected by user")
                        
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
                        
                        # IMPORTANT: Re-apply boundary adjustments to get final boundaries
                        if any(key in result for key in ['top_boundary', 'bottom_boundary', 'left_height_offset', 'right_height_offset']):
                            self.logger.debug(" Re-applying boundary adjustments to get final boundaries")
                            
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
                            self.logger.debug(f" Re-analyzing boundaries with adjustments: {adjustment_params}")
                            adjusted_analysis = self.aruco_manager.analyze_compartment_boundaries(
                                small_image, markers,
                                compartment_count=self.config.get('compartment_count', 20),
                                smart_cropping=True,
                                metadata=metadata
                            )
                            
                            # Update boundaries with adjusted values
                            boundaries = adjusted_analysis['boundaries']
                            self.logger.debug(f" Updated boundaries after adjustments: {len(boundaries)} compartments")
                            
                            # Merge manually placed compartments if any exist
                            # Check if we have manually placed compartments to merge
                            if 'result_boundaries' in result and result['result_boundaries']:
                                self.logger.debug(f" Found {len(result['result_boundaries'])} manually placed compartments to merge")
                                
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
                                        self.logger.debug(f" Added manually placed compartment for marker {marker_id}")
                                
                                # Sort boundaries by x-coordinate
                                boundaries.sort(key=lambda b: b[0])
                                self.logger.debug(f" Total boundaries after merging: {len(boundaries)}")
                         
                        self.logger.debug(f" User completed registration with metadata: {metadata}")
                        self.logger.info(f"User completed registration with metadata: {metadata}")

                else:
                    # User canceled the dialog
                    self.logger.debug(" Dialog canceled by user")
                    self.logger.warning("Image processing canceled by user")
                    if self.progress_queue:
                        add_progress_message("Processing canceled by user", None)
        
                    return False  # Return early to skip processing
            except Exception as e:
                self.logger.error(f"Metadata dialog error: {str(e)}")
                self.logger.error(traceback.format_exc())
                self.logger.debug(f" Metadata dialog error: {str(e)}")
                self.logger.debug(f" {traceback.format_exc()}")
                if self.progress_queue:
                    add_progress_message(f"Metadata dialog error: {str(e)}", None)
                return False



            # ===================================================
            # STEP 10: APPLY SKEW CORRECTION TO HIGH-RES IMAGE
            # ===================================================
            # Apply skew correction to the original high-resolution image if needed
            if rotation_matrix is not None and rotation_angle != 0.0:
                self.logger.debug(f" Applying orientation correction to high-res image (angle: {rotation_angle:.2f}°)")
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
                self.logger.debug(f" High-resolution image orientation corrected with {rotation_angle:.2f}° rotation")
                self.logger.info(f"Applied orientation correction to high-resolution image (angle: {rotation_angle:.2f}°)")
            else:
                self.logger.debug(" No orientation correction needed for high-resolution image")


            # ===================================================
            # STEP 11: SCALE BOUNDARIES TO ORIGINAL IMAGE SIZE
            # ===================================================
            
            # Report number of compartments found
            status_msg = f"Found {len(boundaries)}/{self.config['compartment_count']} compartments"
            self.logger.debug(f" {status_msg}")
            self.logger.info(status_msg)
            if self.progress_queue:
                add_progress_message(status_msg, None)

            
            # Scale up the coordinates from small image to original image if necessary
            if small_image.shape != original_image.shape:
                self.logger.debug(" Need to scale boundaries to original image size")
                if self.progress_queue:
                    add_progress_message("Scaling boundaries to original image size...", None)
                    
                # Calculate scale factors
                scale_x = original_image.shape[1] / small_image.shape[1]
                scale_y = original_image.shape[0] / small_image.shape[0]
                self.logger.debug(f" Scale factors: x={scale_x:.4f}, y={scale_y:.4f}")

                # Store the scale factor so we can adjust px/cm measurements later
                self.image_scale_factor = scale_x  # Assuming x and y scale are similar
                
                # Also adjust the scale data if it exists
                if hasattr(self, 'current_scale_data') and self.current_scale_data:
                    # Store both the original and adjusted scales
                    self.current_scale_data['scale_px_per_cm_small'] = self.current_scale_data['scale_px_per_cm']
                    self.current_scale_data['scale_px_per_cm_original'] = self.current_scale_data['scale_px_per_cm'] * scale_x
                    self.current_scale_data['scale_factor'] = scale_x
                    self.logger.debug(f" Stored scale factor {scale_x:.4f} in current_scale_data")
                    self.logger.debug(f" Small image scale: {self.current_scale_data['scale_px_per_cm_small']:.2f} px/cm")
                    self.logger.debug(f" Original image scale: {self.current_scale_data['scale_px_per_cm_original']:.2f} px/cm")


                # First merge any manually placed compartments before scaling
                if hasattr(self, 'result_boundaries') and self.result_boundaries:
                    self.logger.debug(f" Found {len(self.result_boundaries)} result boundaries to merge before scaling")
                    for marker_id, boundary in self.result_boundaries.items():
                        if marker_id == 24:  # Metadata marker - not a compartment boundary
                            # Metadata markers are for OCR, not compartments
                            # Will be scaled separately below
                            self.logger.debug(f" Found metadata marker {marker_id} - will scale separately")
                            continue
                        if marker_id in [0, 1, 2, 3]:  # Corner markers - not compartment boundaries
                            # Corner markers define constraints, not compartments
                            # Will be scaled separately below
                            self.logger.debug(f" Found corner marker {marker_id} - will scale separately")
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
                                self.logger.debug(f" Added compartment boundary for marker {marker_id} before scaling")
                    
                    # Sort boundaries by x-coordinate
                    boundaries.sort(key=lambda b: b[0])
                
                # Scale the boundary adjustments if they exist
                scaled_left_offset = 0
                scaled_right_offset = 0
                scaled_top_y = int(self.top_y * scale_y) if hasattr(self, 'top_y') else None
                scaled_bottom_y = int(self.bottom_y * scale_y) if hasattr(self, 'bottom_y') else None
                
                if metadata and 'boundary_adjustments' in metadata:
                    adjustments = metadata['boundary_adjustments']
                    self.logger.debug(f" Scaling boundary adjustments from small to original image")
                    
                    # Scale the adjustments
                    scaled_top_y = int(adjustments['top_boundary'] * scale_y)
                    scaled_bottom_y = int(adjustments['bottom_boundary'] * scale_y)
                    scaled_left_offset = int(adjustments['left_height_offset'] * scale_y)
                    scaled_right_offset = int(adjustments['right_height_offset'] * scale_y)
                    
                    self.logger.debug(f" Original adjustments: {adjustments}")
                    self.logger.debug(f" Scaled adjustments: top={scaled_top_y}, bottom={scaled_bottom_y}, "
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
                                self.logger.debug(f" Scaled special marker {marker_id}")
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
                            self.logger.debug(f" Scaled compartment boundary for marker {marker_id}")
                    
                    self.result_boundaries = scaled_result_boundaries
                
                # Update vertical constraints
                if hasattr(self, 'top_y') and hasattr(self, 'bottom_y'):
                    self.top_y = scaled_top_y if scaled_top_y is not None else int(self.top_y * scale_y)
                    self.bottom_y = scaled_bottom_y if scaled_bottom_y is not None else int(self.bottom_y * scale_y)
                    self.logger.debug(f" Updated vertical constraints: top_y={self.top_y}, bottom_y={self.bottom_y}")
                    
                self.logger.debug(f" Scaled {len(compartment_boundaries)} boundaries to original image size")
                if compartment_boundaries:
                    self.logger.debug(f" First scaled boundary: {compartment_boundaries[0]}")
                    self.logger.debug(f" Last scaled boundary: {compartment_boundaries[-1]}")
                    
                self.logger.info(f"Scaled {len(compartment_boundaries)} compartment boundaries to original image")
            else:
                # No scaling needed
                self.logger.debug(" No scaling needed - small image and original have same dimensions")
                compartment_boundaries = boundaries
            
            # ===================================================
            # STEP 12: CONVERT BOUNDARIES TO CORNERS AND EXTRACT COMPARTMENTS AS NP ARRAYS FOR DUPLICATE CHECK DIALOG
            # ===================================================
            self.logger.debug(" Converting boundaries to corners format")
            corners_list = []

            depth_from = metadata.get('depth_from')
            if depth_from is None:
                raise ValueError("Missing required metadata: 'depth_from' is not set before compartment boundary conversion.")

            compartment_interval = metadata.get('compartment_interval', 1)


            for i, (x1, y1, x2, y2) in enumerate(compartment_boundaries):
                # Convert boundary to 4 corners (top_left, top_right, bottom_right, bottom_left)
                corners = {
                    'depth_to': depth_from + ((i + 1) * compartment_interval),
                    'corners': [
                        [x1, y1],  # top_left
                        [x2, y1],  # top_right
                        [x2, y2],  # bottom_right
                        [x1, y2]   # bottom_left
                    ]
                }
                corners_list.append(corners)
            self.logger.debug(f" Created corners for {len(corners_list)} compartments")
            
            self.logger.debug(" Extracting compartments for duplicate check display")
            compartments = self.extract_compartments_from_boundaries(original_image, compartment_boundaries)
            self.logger.debug(f" Extracted {len(compartments)} compartments from boundaries")
            # ===================================================
            # STEP 13: CHECK FOR DUPLICATES AND HANDLE ALL EXITS
            # ===================================================
            # Check for duplicates BEFORE saving
            if (metadata.get('hole_id') and 
                metadata.get('depth_from') is not None and 
                metadata.get('depth_to') is not None):
                
                # Loop for handling metadata modification and re-checking duplicates
                continue_processing = True
                while continue_processing:
                    self.logger.debug(" Checking for duplicates with metadata: hole_id={}, depth={}-{}m".format(
                        metadata.get('hole_id'), metadata.get('depth_from'), metadata.get('depth_to')))
                    
                    # Check for duplicates - this method should run on main thread
                    self.logger.debug(" Calling duplicate_handler.check_duplicate")
                    duplicate_result = self.duplicate_handler.check_duplicate(
                        metadata['hole_id'], 
                        metadata['depth_from'], 
                        metadata['depth_to'], 
                        small_image,  # Use the downsampled image
                        image_path,
                        extracted_compartments=compartments  # Pass extracted compartments
                    )
                    
                    self.logger.debug(f" Duplicate check result: {duplicate_result}")
                    
                    # Process the result based on action
                    if isinstance(duplicate_result, dict) and 'action' in duplicate_result:
                        action = duplicate_result.get('action')
                        
                        if action == 'quit':
                            # User chose to quit processing
                            self.logger.debug(" User chose to quit processing")
                            self.logger.info("User stopped processing via duplicate dialog")
                            self.processing_complete = True
                            return False  # Stop processing
                        
                        elif action == 'modify_metadata':
                            # User modified metadata - update and re-check
                            self.logger.debug(f" User modified metadata from {metadata} to {duplicate_result}")
                            self.logger.info(f"User modified metadata from {metadata} to {duplicate_result}")
                            
                            # Update metadata with new values
                            metadata['hole_id'] = duplicate_result.get('hole_id', metadata['hole_id'])
                            metadata['depth_from'] = duplicate_result.get('depth_from', metadata['depth_from'])
                            metadata['depth_to'] = duplicate_result.get('depth_to', metadata['depth_to'])
                            
                            # Continue the loop to re-check duplicates with new metadata
                            # (stays in while loop)
                        
                        else:
                            # All other actions exit the loop and go to handle_image_exit
                            continue_processing = False
                            
                            # Build result for handle_image_exit
                            result = {
                                'action': action,
                                'hole_id': metadata.get('hole_id'),
                                'depth_from': metadata.get('depth_from'),
                                'depth_to': metadata.get('depth_to'),
                                'compartment_interval': metadata.get('compartment_interval', 1),
                                'boundaries': compartment_boundaries,
                                'corners_list': corners_list
                            }
                            
                            # Add action-specific data
                            if action == 'skip':
                                result['skipped'] = True
                            
                            elif action == 'continue':
                                # No duplicates found - normal processing
                                pass
                            
                            elif action == 'replace_all':
                                # Delete existing compartment files BEFORE saving new ones
                                files_to_delete = duplicate_result.get('files_to_delete', [])
                                if files_to_delete:
                                    add_progress_message(f"Deleting {len(files_to_delete)} existing compartment files...", None)
                                    deleted_count = 0
                                    
                                    for file_path in files_to_delete:
                                        try:
                                            if os.path.exists(file_path):
                                                os.remove(file_path)
                                                deleted_count += 1
                                                self.logger.info(f"Deleted existing compartment: {file_path}")
                                        except Exception as e:
                                            self.logger.error(f"Failed to delete {file_path}: {str(e)}")
                                    
                                    add_progress_message(f"Deleted {deleted_count} existing compartment files", None)
                            
                            elif action == 'selective_replacement':
                                result['selective_replacement'] = True
                            
                            elif action == 'keep_with_gaps':
                                result['skipped'] = True  # Don't overwrite existing compartments
                                result['action'] = 'keep_with_gaps'
                                result['missing_depths'] = duplicate_result.get('missing_depths', [])
                            
                            elif action == 'reject':
                                # Include all rejection details
                                result.update(duplicate_result)
                            
                            # Call handle_image_exit for all non-modify actions
                            saved_paths = self.handle_image_exit(
                                image_path=image_path,
                                result=result,
                                metadata=metadata,
                                compartments=compartments,
                                processing_messages=processing_messages,
                                add_progress_message=add_progress_message
                            )
                            
                            # ===================================================
                            # STEP 14: BLUR DETECTION (OPTIONAL)
                            # ===================================================
                            # Detect blur in saved compartments if enabled
                            if self.config['enable_blur_detection'] and saved_paths:
                                self.logger.debug(f" Blur detection enabled, analyzing {len(saved_paths)} saved compartments")
                                if self.progress_queue:
                                    add_progress_message("Analyzing image sharpness...", None)
                                
                                blur_results = self.detect_blur_in_saved_compartments(saved_paths, os.path.basename(image_path))
                                self.logger.debug(f" Blur detection complete, found {sum(1 for r in blur_results if r.get('is_blurry', False))} blurry compartments")
                            
                            # ===================================================
                            # STEP 15: CLEANUP AND PREPARE FOR NEXT IMAGE
                            # ===================================================
                            # Final summary message
                            if saved_paths:
                                success_msg = f"Successfully processed {os.path.basename(image_path)}"
                                if metadata.get('hole_id'):
                                    success_msg += f" for {metadata['hole_id']} {metadata['depth_from']}-{metadata['depth_to']}m"
                                if action == 'selective_replacement':
                                    success_msg += " (selective replacement)"
                                elif action == 'keep_with_gaps':
                                    missing_count = len(result.get('missing_depths', []))
                                    success_msg += f" (filled {missing_count} gaps)"
                            else:
                                success_msg = f"Processed {os.path.basename(image_path)}"
                                if action == 'skip':
                                    success_msg += " (skipped - keeping existing)"
                                elif action == 'reject':
                                    success_msg += " (rejected)"
                            
                            self.logger.debug(f" {success_msg}")
                            self.logger.info(success_msg)
                            if self.progress_queue:
                                add_progress_message(success_msg, None)
                            
                            # Clear metadata for next image
                            self.logger.debug(" Clearing metadata for next image")
                            self.metadata = {}
                            
                            # Reset annotation state in visualization_cache
                            self.logger.debug(" Resetting annotation state in visualization_cache")
                            if hasattr(self, 'visualization_cache'):
                                if 'annotation_status' in self.visualization_cache:
                                    self.visualization_cache['annotation_status'] = {'annotated_markers': []}
                                # Also clear current processing data
                                if 'current_processing' in self.visualization_cache:
                                    self.visualization_cache['current_processing'] = {}
                            
                            self.logger.debug(" process_image completed successfully")
                            return saved_paths is not None
                            
                    else:
                        # Legacy response format or error
                        self.logger.warning(f"Unexpected duplicate check result format: {duplicate_result}")
                        continue_processing = False
                        return False

            # If we get here without metadata, we can't process
            else:
                self.logger.error("Cannot process image - missing required metadata")
                if add_progress_message:
                    add_progress_message("Cannot process image - missing metadata", None)
                return False
                    
        except Exception as e:
            # ===================================================
            # ERROR HANDLING
            # ===================================================
            error_msg = f"Error processing {image_path}: {str(e)}"
            self.logger.debug(f" CRITICAL ERROR: {error_msg}")
            self.logger.debug(f" {traceback.format_exc()}")
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
            # Post error message directly to main GUI
            if self.progress_queue:
                self.progress_queue.put((error_msg, None))
                if hasattr(self, 'main_gui') and hasattr(self.main_gui, 'check_progress'):
                    try:
                        self.main_gui.check_progress(schedule_next=False)
                    except Exception as gui_e:
                        self.logger.debug(f"Could not flush error to GUI: {gui_e}")
            
            # Reset metadata for next image
            self.metadata = {}
            
            # Reset annotation state in visualization_cache
            self.logger.debug(" Resetting annotation state in visualization_cache after exception")
            if hasattr(self, 'visualization_cache'):
                if 'annotation_status' in self.visualization_cache:
                    self.visualization_cache['annotation_status'] = {'annotated_markers': []}
                if 'current_processing' in self.visualization_cache:
                    self.visualization_cache['current_processing'] = {}

            self.logger.debug(" process_image failed")
            return False

        
    def process_folder(self, folder_path: str) -> Tuple[int, int]:
        """
        Process all images in a folder, one at a time on the main thread.
        
        Args:
            folder_path: Path to folder containing images
            
        Returns:
            Tuple of (number of images processed, number of images failed)
        """
        self.logger.debug(f" Starting process_folder with path: {folder_path}")
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f" Logger initialized: {self.logger}")
        
        # Ensure progress_queue exists
        if not hasattr(self, 'progress_queue'):
            self.logger.debug(" Creating new progress_queue")
            self.progress_queue = queue.Queue()
        else:
            self.logger.debug(" Using existing progress_queue")
        
        # Get all image files from the folder
        self.logger.debug(" Scanning for image files")
        image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                    if os.path.isfile(os.path.join(folder_path, f)) and 
                    f.lower().endswith(image_extensions)]
        
        self.logger.debug(f" Found {len(image_files)} image files with extensions {image_extensions}")
        
        if not image_files:
            warning_msg = f"No image files found in {folder_path}"
            self.logger.debug(f" {warning_msg}")
            self.logger.warning(warning_msg)
            if self.progress_queue:
                self.progress_queue.put((warning_msg, None))
            return 0, 0
        
        # Log found images
        info_msg = f"Found {len(image_files)} image files"
        self.logger.debug(f" {info_msg}")
        self.logger.info(info_msg)
        if self.progress_queue:
            self.progress_queue.put((info_msg, None))
        
        # Process each image one by one
        successful = 0
        failed = 0
        
        self.logger.debug(f" Starting to process {len(image_files)} images")
        for i, image_path in enumerate(image_files):
            try:
                # Update progress
                progress = ((i + 1) / len(image_files)) * 100
                progress_msg = f"Processing Image: {i+1}/{len(image_files)}: {os.path.basename(image_path)}"
                self.logger.debug(f" {progress_msg} (progress: {progress:.1f}%)")
                self.logger.info(progress_msg)
                
                self.logger.debug(f" Updating progress queue with {progress:.1f}%")
                self.progress_queue.put((progress_msg, progress))
                
                # Process the image on the main thread
                self.logger.debug(f" Calling process_image for {os.path.basename(image_path)}")
                result = self.process_image(image_path)
                self.logger.debug(f" process_image returned {result}")
                
                if result:
                    successful += 1
                    self.logger.debug(f" Successful count incremented to {successful}")
                else:
                    failed += 1
                    self.logger.debug(f" Failed count incremented to {failed}")
                
                self.logger.debug(f" Finished processing image {i+1}/{len(image_files)}, continuing to next...")
                # Update GUI to ensure it's responsive
                if hasattr(self, 'main_gui') and hasattr(self.main_gui, 'root'):
                    self.main_gui.root.update_idletasks()
                    
            except Exception as e:
                error_msg = f"Error processing {image_path}: {str(e)}"
                self.logger.debug(f" Exception caught: {error_msg}")
                self.logger.debug(f" Exception type: {type(e).__name__}")
                self.logger.error(error_msg)
                self.logger.error(traceback.format_exc())
                self.progress_queue.put((error_msg, None))
                failed += 1
                self.logger.debug(f" Failed count incremented to {failed}")
        
        # Log summary
        summary_msg = f"Processing complete: {successful} successful, {failed} failed"
        self.logger.debug(f" {summary_msg}")
        self.logger.info(summary_msg)
        if self.progress_queue:
            self.progress_queue.put((summary_msg, 100))  # Set progress to 100%
        
        self.logger.debug(f" Returning from process_folder: ({successful}, {failed})")
        return successful, failed

    def handle_image_exit(self, 
                        image_path: str,
                        result: dict,
                        metadata: dict = None,
                        compartments: list = None,
                        processing_messages: list = None,
                        add_progress_message=None) -> List[str]:
        """
        Centralized handler for all image processing exit points.
        Handles compartment extraction, file saving, register updates, and cleanup.
        
        Args:
            image_path: Path to the original image
            result: Result dictionary containing exit status, metadata, boundaries, corners
            metadata: Current metadata (may be partial depending on exit point)
            compartments: Extracted compartment numpy arrays
            processing_messages: List of progress messages to flush
            add_progress_message: Function to add progress messages
            
        Returns:
            List[str]: List of saved compartment paths (empty list if none saved)
        """
        saved_compartment_paths = []
        saved_compartment_indices = []  # Track which indices were successfully saved
        
        try:
            # Determine exit type and action
            is_rejected = result.get('rejected', False)
            is_quit = result.get('quit', False)
            is_skipped = result.get('skipped', False)
            is_selective = result.get('selective_replacement', False)
            action = result.get('action', '')
            
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
                self.logger.debug(" User quit processing")
                self.logger.info("User stopped processing")
                if add_progress_message:
                    add_progress_message("Processing stopped by user", None)
                self.processing_complete = True
                return []
            
            # Validate we have minimum required metadata
            if not all([final_metadata.get('hole_id'), 
                    final_metadata.get('depth_from') is not None,
                    final_metadata.get('depth_to') is not None]):
                self.logger.warning("Cannot save file - missing required metadata")
                if add_progress_message:
                    add_progress_message("Cannot save file - missing metadata", None)
                return []
            
            # Update last metadata for next image
            self.update_last_metadata(
                final_metadata['hole_id'],
                final_metadata['depth_from'],
                final_metadata['depth_to'],
                final_metadata['compartment_interval']
            )
            
            # ===================================================
            # SAVE COMPARTMENTS (unless skipped)
            # ===================================================
            if not is_skipped and compartments and not is_rejected:
                # Determine suffix based on action
                if is_selective:
                    suffix = "new"  # For side-by-side comparison
                else:
                    suffix = "temp"  # Normal processing
                
                self.logger.debug(f" Saving {len(compartments)} compartments with suffix '{suffix}'")
                if add_progress_message:
                    add_progress_message(f"Saving {len(compartments)} compartments...", None)
                
                # Save each compartment
                start_depth = int(final_metadata['depth_from'])
                compartment_interval = int(final_metadata['compartment_interval'])
                
                for i, comp in enumerate(compartments):
                    if comp is not None:
                        comp_depth = start_depth + ((i + 1) * compartment_interval)
                        try:
                            saved_path = self.file_manager.save_temp_compartment(
                                comp,
                                final_metadata['hole_id'],
                                comp_depth,
                                suffix=suffix
                            )
                            if saved_path:
                                saved_compartment_paths.append(saved_path)
                                saved_compartment_indices.append(i)  # Track which index was saved
                        except Exception as e:
                            self.logger.error(f"Error saving compartment at depth {comp_depth}: {str(e)}")
            
            # Extract compartments for gaps if needed (keep_with_gaps action)
            elif action == 'keep_with_gaps' and compartments:
                missing_depths = result.get('missing_depths', [])
                if missing_depths:
                    self.logger.debug(f" Extracting {len(missing_depths)} missing compartments")
                    if add_progress_message:
                        add_progress_message(f"Saving {len(missing_depths)} missing compartments...", None)
                        
                    start_depth = int(final_metadata['depth_from'])
                    compartment_interval = int(final_metadata['compartment_interval'])
                    
                    for i, comp in enumerate(compartments):
                        if comp is not None:
                            comp_depth = start_depth + ((i + 1) * compartment_interval)
                            if comp_depth in missing_depths:
                                try:
                                    saved_path = self.file_manager.save_temp_compartment(
                                        comp,
                                        final_metadata['hole_id'],
                                        comp_depth,
                                        suffix="temp"
                                    )
                                    if saved_path:
                                        saved_compartment_paths.append(saved_path)
                                        saved_compartment_indices.append(i)  # Track which index was saved
                                        self.logger.info(f"Saved missing compartment at depth {comp_depth}m")
                                except Exception as e:
                                    self.logger.error(f"Error saving missing compartment: {str(e)}")
            # ===================================================
            # PREPARE COMPARTMENT DATA FOR REGISTERS
            # ===================================================
            compartment_data_for_original = {}  # For original image register
            compartment_updates = []  # For compartment register
            
            # Process corners data and prepare updates
            if hasattr(self, 'register_manager'):
                self.logger.debug(f" Preparing data for register update")
                self.logger.debug(f" Has register_manager: {hasattr(self, 'register_manager')}")
                
                # ===================================================
                # INSERT: Debug statements for variable availability
                # ===================================================
                self.logger.debug(f" Checking available data:")
                print(f"  - result type: {type(result)}")
                print(f"  - result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
                print(f"  - 'corners_list' in result: {'corners_list' in result if isinstance(result, dict) else False}")
                print(f"  - final_metadata: {final_metadata}")
                print(f"  - saved_compartment_indices: {saved_compartment_indices}")
                print(f"  - hasattr(self, 'current_scale_data'): {hasattr(self, 'current_scale_data')}")
                
                if hasattr(self, 'current_scale_data'):
                    print(f"  - self.current_scale_data: {self.current_scale_data}")
                else:
                    print(f"  - self.current_scale_data: NOT SET")
                
                start_depth = int(final_metadata['depth_from'])
                compartment_interval = int(final_metadata['compartment_interval'])
                
                # Process ALL corners data for original image register
                # This includes all compartments, not just saved ones
                if 'corners_list' in result and result['corners_list']:
                    # ===================================================
                    # INSERT: Debug corners_list content
                    # ===================================================
                    self.logger.debug(f" corners_list found with {len(result['corners_list'])} entries")
                    
                    for i, corner_data in enumerate(result['corners_list']):
                        # ===================================================
                        # INSERT: Debug individual corner data
                        # ===================================================
                        self.logger.debug(f" Processing corner {i+1}:")
                        print(f"  - corner_data type: {type(corner_data)}")
                        print(f"  - corner_data keys: {corner_data.keys() if isinstance(corner_data, dict) else 'Not a dict'}")
                        
                        # Extract corners in compact format for original image register
                        if 'corners' in corner_data:
                            corners = corner_data['corners']
                            # ===================================================
                            # INSERT: Debug corners format
                            # ===================================================
                            print(f"  - corners type: {type(corners)}")
                            print(f"  - corners length: {len(corners) if isinstance(corners, list) else 'Not a list'}")
                            if isinstance(corners, list) and len(corners) > 0:
                                print(f"  - first corner: {corners[0]}")
                            
                            # Corners should be a list of 4 points: [TL, TR, BR, BL]
                            if isinstance(corners, list) and len(corners) == 4:
                                # Ensure each corner is a list (not tuple)
                                compartment_data_for_original[str(i + 1)] = [
                                    list(corners[0]),  # top_left
                                    list(corners[1]),  # top_right
                                    list(corners[2]),  # bottom_right
                                    list(corners[3])   # bottom_left
                                ]
                            else:
                                self.logger.warning(f"Invalid corners format for compartment {i+1}")
                                # ===================================================
                                # INSERT: More detail on invalid format
                                # ===================================================
                                print(f"WARNING: Invalid corners format - expected list of 4 points, got: {corners}")
                else:
                    # ===================================================
                    # INSERT: Debug when no corners_list
                    # ===================================================
                    self.logger.debug(f" No corners_list found in result or it's empty")
                    
                # Prepare compartment register updates ONLY for successfully saved compartments
                # Use saved_compartment_indices to ensure proper alignment
                # ===================================================
                # INSERT: Debug saved compartment processing
                # ===================================================
                self.logger.debug(f" Processing {len(saved_compartment_indices)} saved compartments")
                
                for saved_idx, compartment_idx in enumerate(saved_compartment_indices):
                    comp_depth_from = start_depth + (compartment_idx * compartment_interval)
                    comp_depth_to = start_depth + ((compartment_idx + 1) * compartment_interval)
                    
                    # ===================================================
                    # INSERT: Debug compartment metadata
                    # ===================================================
                    self.logger.debug(f" Compartment {compartment_idx+1} (saved_idx={saved_idx}):")
                    print(f"  - depth range: {comp_depth_from} - {comp_depth_to}")
                    
                    update = {
                        'hole_id': final_metadata['hole_id'],
                        'depth_from': comp_depth_from,
                        'depth_to': comp_depth_to,
                        'photo_status': 'For Review',  # Changed from 'Extracted' to 'For Review'
                        'processed_by': os.getenv("USERNAME", "System")  # Changed from 'approved_by'
                    }
                    # Calculate individual compartment width if we have scale data and corners
                    if (hasattr(self, 'current_scale_data') and self.current_scale_data and 
                        'corners_list' in result and compartment_idx < len(result['corners_list'])):
                        
                        scale_px_per_cm = self.current_scale_data.get('scale_px_per_cm')
                        # ===================================================
                        # INSERT: Debug scale calculation
                        # ===================================================
                        print(f"  - scale_px_per_cm (from small image): {scale_px_per_cm}")
                        
                        if scale_px_per_cm and scale_px_per_cm > 0:
                            # Get the corners for this specific compartment
                            corner_data = result['corners_list'][compartment_idx]
                            if 'corners' in corner_data:
                                corners = corner_data['corners']
                                if isinstance(corners, list) and len(corners) >= 4:
                                    # Calculate width from corners (top-left to top-right)
                                    # corners format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                                    x1 = corners[0][0]  # top_left x
                                    x2 = corners[1][0]  # top_right x
                                    width_px = abs(x2 - x1)
                                    
                                    # ===================================================
                                    # REMOVE: Old check
                                    # REPLACE WITH: Use stored scale data
                                    # ===================================================
                                    # Check if we have scale adjustment info
                                    if 'scale_px_per_cm_original' in self.current_scale_data:
                                        # Use the pre-calculated original image scale
                                        adjusted_scale_px_per_cm = self.current_scale_data['scale_px_per_cm_original']
                                        print(f"  - Using pre-calculated original scale: {adjusted_scale_px_per_cm:.2f}")
                                    elif 'scale_factor' in self.current_scale_data:
                                        # Use the stored scale factor
                                        adjusted_scale_px_per_cm = scale_px_per_cm * self.current_scale_data['scale_factor']
                                        print(f"  - Adjusting with stored scale factor: {self.current_scale_data['scale_factor']:.4f}")
                                        print(f"  - adjusted scale_px_per_cm: {adjusted_scale_px_per_cm:.2f}")
                                    elif hasattr(self, 'image_scale_factor'):
                                        # Fall back to instance variable if available
                                        adjusted_scale_px_per_cm = scale_px_per_cm * self.image_scale_factor
                                        print(f"  - Adjusting with instance scale factor: {self.image_scale_factor:.4f}")
                                        print(f"  - adjusted scale_px_per_cm: {adjusted_scale_px_per_cm:.2f}")
                                    else:
                                        # No adjustment available - scale might be wrong
                                        adjusted_scale_px_per_cm = scale_px_per_cm
                                        print(f"  - WARNING: No scale adjustment available - width calculation may be incorrect!")
                                    
                                    compartment_width_cm = round(width_px / adjusted_scale_px_per_cm, 2)
                                    update['image_width_cm'] = compartment_width_cm
                                    # ===================================================
                                    # INSERT: Debug width calculation
                                    # ===================================================
                                    print(f"  - compartment width: {width_px}px = {compartment_width_cm}cm")
                    else:
                        # ===================================================
                        # INSERT: Debug why scale not calculated
                        # ===================================================
                        print(f"  - Width calculation skipped:")
                        print(f"    - has current_scale_data: {hasattr(self, 'current_scale_data')}")
                        print(f"    - current_scale_data not None: {self.current_scale_data is not None if hasattr(self, 'current_scale_data') else 'N/A'}")
                        print(f"    - corners_list in result: {'corners_list' in result}")
                        print(f"    - compartment_idx < len(corners_list): {compartment_idx < len(result['corners_list']) if 'corners_list' in result else 'N/A'}")
                    
                    compartment_updates.append(update)
            
            # ===================================================
            # SAVE ORIGINAL FILE
            # ===================================================
            self.logger.debug(f" Saving original file with status - rejected:{is_rejected}, skipped:{is_skipped}, selective:{is_selective}")
            local_path, upload_success = self.file_manager.save_original_file(
                image_path,
                final_metadata['hole_id'],
                final_metadata['depth_from'],
                final_metadata['depth_to'],
                is_processed=True,  # All handled files are "processed"
                is_rejected=is_rejected,
                is_selective=is_selective,
                is_skipped=is_skipped
            )
            
            # ===================================================
            # UPDATE REGISTERS
            # ===================================================
            if hasattr(self, 'register_manager'):
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
                if action == 'keep_with_gaps':
                    missing_depths = result.get('missing_depths', [])
                    if missing_depths:
                        comments.append(f"Filled gaps at depths: {missing_depths}")
                
                comments_str = ". ".join(comments) if comments else None
                
                # Extract scale data for original image register
                scale_px_per_cm = None
                scale_confidence = None
                if hasattr(self, 'current_scale_data') and self.current_scale_data:
                    scale_px_per_cm = round(self.current_scale_data['scale_px_per_cm'], 2)
                    scale_confidence = round(self.current_scale_data.get('confidence', 0.0), 3)
                
                # Update original image register with scale and ALL compartment data
                self.register_manager.update_original_image(
                    final_metadata['hole_id'],
                    final_metadata['depth_from'],
                    final_metadata['depth_to'],
                    original_filename,
                    is_approved=not is_rejected,
                    upload_success=upload_success,
                    uploaded_by=os.getenv("USERNAME", "Unknown"),
                    comments=comments_str,
                    scale_px_per_cm=scale_px_per_cm,
                    scale_confidence=scale_confidence,
                    compartment_data=compartment_data_for_original if compartment_data_for_original else None
                )
                
                status_type = "rejected" if is_rejected else ("skipped" if is_skipped else "processed")
                self.logger.info(f"Updated original image register for {status_type} image")
                
                # Batch update compartment register
                if compartment_updates:
                    try:
                        updated = self.register_manager.batch_update_compartments(compartment_updates)
                        self.logger.info(f"Updated {updated} compartment records")
                        if add_progress_message:
                            add_progress_message(f"Updated {updated} compartment records", None)
                    except Exception as e:
                        self.logger.error(f"Error updating compartment records: {str(e)}")
            
            # Add appropriate progress message
            if add_progress_message:
                if local_path:
                    # Check the actual filename to see status
                    actual_filename = os.path.basename(local_path)
                    
                    if is_rejected:
                        if '_UPLOADED' in actual_filename:
                            add_progress_message("Rejected image uploaded to shared folder and saved locally", None)
                        elif '_UPLOAD_FAILED' in actual_filename:
                            add_progress_message("Rejected image saved locally (shared folder upload failed)", None)
                        else:
                            add_progress_message("Rejected image saved locally", None)
                    elif is_skipped:
                        msg = "File skipped - saved to appropriate folder"
                        if upload_success:
                            msg += " and uploaded to shared folder"
                        add_progress_message(msg, None)
                    else:
                        msg = f"Saved original image and {len(saved_compartment_paths)} compartments"
                        if upload_success:
                            msg += " (uploaded to shared folder)"
                        elif not self.file_manager.shared_paths:
                            msg += " (shared folder not configured)"
                        else:
                            msg += " (shared folder upload failed)"
                        add_progress_message(msg, None)
                else:
                    add_progress_message("Failed to save file", None)
            
            return saved_compartment_paths  # Return paths for blur detection
            
        except Exception as e:
            self.logger.error(f"Error in handle_image_exit: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            # Check if original file still exists
            try:
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
            
            return []  # Return empty list on error

    def detect_blur_in_saved_compartments(self, compartment_paths: List[str], source_filename: str) -> List[Dict]:
        """
        Detect blur in saved compartment files.
        
        Args:
            compartment_paths: List of paths to saved compartment images
            source_filename: Original source filename for logging
            
        Returns:
            List of blur detection results
        """
        blur_results = []
        
        if not hasattr(self, 'blur_detector'):
            # Initialize blur detector if not already created
            blur_threshold = self.config.get('blur_threshold', 100.0)
            roi_ratio = self.config.get('blur_roi_ratio', 0.8)
            self.blur_detector = BlurDetector(threshold=blur_threshold, roi_ratio=roi_ratio)
        
        # Load and analyze each saved compartment
        for i, path in enumerate(compartment_paths):
            try:
                if os.path.exists(path):
                    # Load the saved image
                    comp_img = cv2.imread(path)
                    if comp_img is not None:
                        # Analyze for blur
                        is_blurry, variance = self.blur_detector.is_blurry(comp_img)
                        
                        # Extract depth from filename (e.g., "XX1234_CC_001_temp.png" -> 1)
                        depth_match = re.search(r'_CC_(\d+)', os.path.basename(path))
                        depth = int(depth_match.group(1)) if depth_match else i + 1
                        
                        result = {
                            'index': i,
                            'depth': depth,
                            'is_blurry': is_blurry,
                            'variance': variance,
                            'path': path
                        }
                        blur_results.append(result)
                        
                        if is_blurry:
                            self.logger.warning(f"Blurry compartment detected at depth {depth}m (variance: {variance:.2f})")
                    else:
                        self.logger.error(f"Failed to load compartment image: {path}")
                else:
                    self.logger.error(f"Compartment file not found: {path}")
                    
            except Exception as e:
                self.logger.error(f"Error analyzing blur for compartment {i}: {str(e)}")
        
        # Log summary
        blurry_count = sum(1 for r in blur_results if r.get('is_blurry', False))
        if blurry_count > 0:
            self.logger.info(f"Blur detection complete for {source_filename}: {blurry_count}/{len(compartment_paths)} compartments are blurry")
        
        return blur_results

    def handle_missing_compartments(self, image, compartment_boundaries, missing_marker_ids, metadata=None, 
                                vertical_constraints=None, marker_to_compartment=None, is_metadata_marker=False, markers=None, scale_data=None):
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
        self.logger.debug(" Running wrapper 'handle_missing_compartments' for 'handle_markers_and_boundaries')")
        return self.handle_markers_and_boundaries(
            image, 
            compartment_boundaries, 
            missing_marker_ids=missing_marker_ids,
            metadata=metadata, 
            vertical_constraints=vertical_constraints,
            marker_to_compartment=marker_to_compartment,
            markers=markers,  # Pass markers to unified handler
            scale_data = scale_data,
            initial_mode=1  # MODE_MISSING_BOUNDARIES
        )

    def handle_boundary_adjustments(self, image, detected_boundaries, missing_marker_ids=None, 
                            metadata=None, vertical_constraints=None, 
                            rotation_angle=0.0, corner_markers=None, marker_to_compartment=None, markers=None, scale_data=None):
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
        self.logger.debug(" Running wrapper 'handle_boundary_adjustments' for 'handle_markers_and_boundaries')")
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
            scale_data=scale_data, 
            initial_mode=2,  # MODE_ADJUST_BOUNDARIES
            show_adjustment_controls=True  # Make adjustment controls visible by default
        )

    def extract_compartments_from_boundaries(self, image: np.ndarray, boundaries: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        """
        Extract compartment regions from image using boundaries. This stores the compartments as np arrays for display in the duplicate handler dialog
        
        Args:
            image: Source image
            boundaries: List of (x1, y1, x2, y2) tuples
            
        Returns:
            List of numpy arrays (compartment images)
        """
        compartments = []
        for x1, y1, x2, y2 in boundaries:
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            # Extract compartment
            compartment = image[y1:y2, x1:x2]
            compartments.append(compartment)
        
        return compartments


    def handle_markers_and_boundaries(self, image, detected_boundaries, missing_marker_ids=None, 
                            metadata=None, vertical_constraints=None, 
                            rotation_angle=0.0, corner_markers=None, marker_to_compartment=None, 
                            initial_mode=2, markers=None, show_adjustment_controls=True,
                            on_apply_adjustments=None, image_path=None, scale_data=None):
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
        self.logger.debug(f" HANDLE_MARKERS START - initial_mode: {initial_mode}")
        self.logger.debug(f" HANDLE_MARKERS START - missing_marker_ids: {missing_marker_ids}")

        # Map mode number to descriptive name for logging
        mode_names = {
            0: "metadata registration", 
            1: "marker placement",
            2: "boundary adjustment"
        }
        operation_name = mode_names.get(initial_mode, "unknown operation")
        
        self.logger.info(f"Handling {operation_name} dialog on main thread")
        
        self.logger.debug(f" Starting {operation_name} function")
        self.logger.debug(f" Current thread: {threading.current_thread().name}")
        self.logger.debug(f" Is main thread: {threading.current_thread() is threading.main_thread()}")
        self.logger.debug(f" Input parameters - image shape: {image.shape if hasattr(image, 'shape') else 'unknown'}")
        self.logger.debug(f" detected_boundaries count: {len(detected_boundaries)}")
        self.logger.debug(f" missing_marker_ids: {missing_marker_ids}")
        self.logger.debug(f" vertical_constraints: {vertical_constraints}")
        self.logger.debug(f" rotation_angle: {rotation_angle}")
        
        # Get the original image with no annotations if available
        original_unannotated_image = None
        if hasattr(self, 'visualization_cache'):
            self.logger.debug(" Checking for original image in visualization_cache")
            original_unannotated_image = self.visualization_cache.get('current_processing', {}).get('original_image')
            if original_unannotated_image is not None:
                self.logger.debug(" Found original unannotated image in visualization_cache")
                # Use the original image instead of any annotated version
                image = original_unannotated_image.copy()
        
        # If we have a small_image attribute, use that as a fallback
        if original_unannotated_image is None and hasattr(self, 'small_image'):
            self.logger.debug(" Using small_image as original image")
            image = self.small_image.copy()
        
        # Get visualization (will be passed as boundaries_viz but lower priority)
        boundaries_viz = None
        if hasattr(self, 'visualization_cache'):
            boundaries_viz = self.visualization_cache.get('current_processing', {}).get('compartment_boundaries_viz')
            self.logger.debug(f" boundaries_viz from cache: {'Found' if boundaries_viz is not None else 'Not found'}")

        # Check if we're focused on marker placement (vs. just boundary adjustments)
        is_placing_markers = missing_marker_ids and len(missing_marker_ids) > 0
        
        # Check if we're specifically annotating a metadata marker (ID 24)
        is_metadata_marker = is_placing_markers and 24 in missing_marker_ids
        self.logger.debug(f" Is metadata marker annotation: {is_metadata_marker}")
        
        # CRITICAL FIX: Check if we've already annotated these missing markers
        if is_placing_markers and hasattr(self, 'visualization_cache') and 'annotation_status' in self.visualization_cache:
            already_annotated = self.visualization_cache['annotation_status'].get('annotated_markers', [])
            if already_annotated:
                self.logger.debug(f" Some markers were already annotated: {already_annotated}")
                
                # Check if all requested markers were already annotated
                if set(missing_marker_ids).issubset(set(already_annotated)):
                    self.logger.debug(f" All requested markers {missing_marker_ids} were already annotated")
                    
                    if initial_mode == 1:  # MODE_MISSING_BOUNDARIES
                        self.logger.debug(" In missing boundaries mode - skipping dialog for already annotated markers")
                        return {}  # Return empty dict to signal "already done"
                    
                    self.logger.debug(" In adjustment mode - still showing dialog despite markers being annotated")
                
                # Filter out already annotated markers from the list to process
                if isinstance(missing_marker_ids, list):
                    missing_marker_ids = [m for m in missing_marker_ids if m not in already_annotated]
                elif isinstance(missing_marker_ids, set):
                    missing_marker_ids = {m for m in missing_marker_ids if m not in already_annotated}
                
                self.logger.debug(f" Filtered missing_marker_ids after removing already annotated: {missing_marker_ids}")
        
        # Create the callback function for re-extracting boundaries
        def apply_adjustments_callback(adjustment_params):
            try:
                self.logger.debug(f" Received adjustment params: {adjustment_params}")
                
                # Create a temporary metadata dict with adjustments
                temp_metadata = {}
                if metadata:
                    temp_metadata.update(metadata)
                
                # Add boundary adjustments
                temp_metadata['boundary_adjustments'] = adjustment_params
                
                # Re-extract boundaries
                if hasattr(self, 'aruco_manager'):
                    self.logger.debug(" Re-extracting compartment boundaries with adjustments")
                    
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
                    
                    self.logger.debug(f" Re-extracted {len(new_boundaries)} boundaries")
                    
                    # Return both boundaries and visualization
                    return {
                        'boundaries': new_boundaries,
                        'visualization': new_viz
                    }
                else:
                    self.logger.debug(" aruco_manager not available")
                    return None
            except Exception as e:
                self.logger.debug(f" Error in apply_adjustments_callback: {str(e)}")
                import traceback
                self.logger.debug(f" {traceback.format_exc()}")
                return None
        
        try:
            # Thread check - hard fail if wrong
            if threading.current_thread() is not threading.main_thread():
                self.logger.debug(" Not on main thread - raising exception!")
                raise RuntimeError(f"❌ {operation_name} called from non-main thread!")

            # Get visualization if available
            boundaries_viz = None
            if hasattr(self, 'visualization_cache'):
                self.logger.debug(" visualization_cache exists")
                boundaries_viz = self.visualization_cache.get('current_processing', {}).get('compartment_boundaries_viz')
                self.logger.debug(f" boundaries_viz from cache: {'Found' if boundaries_viz is not None else 'Not found'}")
            
            # Get markers from aruco_manager if available
            if markers is None:
                markers = {}
                if hasattr(self, 'aruco_manager'):
                    markers = getattr(self.aruco_manager, 'cached_markers', {})
                    self.logger.debug(f" Got {len(markers)} markers from aruco_manager")

            # Check visualization cache for more complete marker data
            if hasattr(self, 'visualization_cache') and 'current_processing' in self.visualization_cache:
                cached_markers = self.visualization_cache['current_processing'].get('all_markers')
                if cached_markers and len(cached_markers) > len(markers):
                    markers = cached_markers
                    self.logger.debug(f" Using {len(markers)} markers from visualization_cache instead")
            
            # Ensure we have a valid marker_to_compartment mapping
            if marker_to_compartment is None:
                compartment_interval = metadata.get('compartment_interval', 
                                                self.config.get('compartment_interval', 1.0))
                marker_to_compartment = {
                    4+i: int((i+1) * compartment_interval) for i in range(20)
                }
                self.logger.debug(f" Created marker_to_compartment mapping: {marker_to_compartment}")

            # Create and show dialog
            self.logger.debug(f" About to create CompartmentRegistrationDialog with initial_mode={initial_mode}")
            self.logger.debug(f" self.root is None: {self.root is None}")
            self.logger.debug(f" image is None: {image is None}")
            self.logger.debug(f" detected_boundaries length: {len(detected_boundaries)}")
            
            # Set app reference on root for dialog access
            self.root.app = self
            
            # Get scale data from visualization cache if available
            if scale_data is None and hasattr(self, 'visualization_cache') and 'current_processing' in self.visualization_cache:
                scale_data = self.visualization_cache['current_processing'].get('scale_data')
                self.logger.debug(f" Got scale_data from visualization_cache: {scale_data is not None}")

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
                image_path=image_path,
                scale_data = scale_data
            )
            
            # Set the initial mode after creation
            dialog.current_mode = initial_mode
            dialog._update_mode_indicator()  # Update UI to reflect the mode
            
            self.logger.debug(f" CompartmentRegistrationDialog object created in mode {initial_mode}")

            # Show dialog and get results
            self.logger.debug(" About to show dialog")
            result = dialog.show()
            self.logger.debug(f" Dialog result received: {result is not None}")
            self.logger.debug(f" HANDLE_MARKERS - dialog.show() returned")
            self.logger.debug(f" HANDLE_MARKERS - result type: {type(result)}")
            self.logger.debug(f" HANDLE_MARKERS - result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
            
            
            # Store markers that were actually annotated
            if result and is_placing_markers:
                self.logger.debug(" Result received, updating visualization cache")
                if hasattr(self, 'visualization_cache'):
                    # Get existing annotated markers
                    existing = self.visualization_cache.get('annotation_status', {}).get('annotated_markers', [])
                    
                    # Determine which markers were actually annotated
                    actually_annotated = []
                    
                    # If result contains result_boundaries, use those to determine which markers were annotated
                    if 'result_boundaries' in result and isinstance(result['result_boundaries'], dict):
                        actually_annotated = list(result['result_boundaries'].keys())
                        self.logger.debug(f" Markers actually annotated according to result_boundaries: {actually_annotated}")
                    else:
                        # Fall back to missing_marker_ids as a best guess
                        actually_annotated = missing_marker_ids
                        self.logger.debug(f" No result_boundaries found, using missing_marker_ids: {actually_annotated}")
                    
                    # Combine with existing annotated markers
                    all_annotated = list(set(existing + list(actually_annotated)))
                    
                    # # Record which markers were annotated
                    # self.visualization_cache.setdefault('annotation_status', {})['annotated_markers'] = all_annotated
                    # print(f"DEBUG: Updated annotated markers in cache: {all_annotated}")
                    
                    # # Update visualization if available
                    # if 'final_visualization' in result:
                    #     self.logger.debug(" Found final_visualization in result")
                    #     if hasattr(self, 'visualization_cache'):
                    #         self.logger.debug(" Updating visualization_cache")
                    #         self.visualization_cache.setdefault('current_processing', {})['compartment_boundaries_viz'] = result['final_visualization']

            # Check for boundary adjustments in the result
            if result and isinstance(result, dict):
                # Check for boundary adjustment data
                boundary_fields = ['top_boundary', 'bottom_boundary', 'left_height_offset', 'right_height_offset']
                has_boundary_adjustments = any(field in result for field in boundary_fields)
                
                if has_boundary_adjustments:
                    self.logger.debug(" Found boundary adjustments in dialog result")
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
                        self.logger.debug(f" Stored boundary adjustments in metadata: {boundary_adjustments}")
                    
                    # # Also update the visualization cache with any new visualization
                    # if 'final_visualization' in result:
                    #     self.visualization_cache.setdefault('current_processing', {})['compartment_boundaries_viz'] = result['final_visualization']
                    #     self.logger.debug(" Updated visualization in cache with adjusted version")

            self.logger.debug(f" HANDLE_MARKERS END - returning result? {result is not None}")
            return result or {}

        except Exception as e:
            print(f"❌ Exception in {operation_name}: {e}")
            self.logger.debug(f" Exception type: {type(e).__name__}")
            traceback.print_exc()
            if hasattr(self, 'logger'):
                self.logger.error(f"Error in {operation_name}: {e}")
                self.logger.error(traceback.format_exc())
            return {}  # Return empty dict on error

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