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
# logging.getLogger('core.tesseract_manager').setLevel(logging.WARNING) TODO - fully remove all tesseract manager support and uses
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
    ConfigManager,
    VisualizationManager,
    VisualizationType
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

from processing.pipeline import (
    load_image_to_np_array,
    apply_transform
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
        # # OCR component TODO - FULLY REMOVE TESSERACT MANAGER REFERENCES AND OCR SUPPORT
        # self.tesseract_manager = TesseractManager()
        # self.tesseract_manager.config = self.config
        # self.tesseract_manager.file_manager = self.file_manager
        # self.tesseract_manager.extractor = self  # Give it access to visualization_cache
        
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
        Process a single chip tray image using full resolution throughout.
        
        This method performs the complete image processing pipeline:
        1. Load image at full resolution
        2. Detect ArUco markers
        3. Correct image orientation/skew
        4. Extract compartment boundaries
        5. Show registration dialog with downsampled visualization
        6. Apply adjustments back to full resolution
        7. Save results
        
        Args:
            image_path: Path to the image file to be processed
            
        Returns:
            bool: True if processing succeeded, False otherwise
        """
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"Starting to process image: {image_path}")
        
        # Initialize visualization manager for this image
        if not hasattr(self, 'viz_manager'):
            self.viz_manager = VisualizationManager(self.file_manager)
        self.viz_manager.clear()
        
        # Initialize progress handling
        processing_messages = []
        
        def add_progress_message(message, progress=None, message_type="info"):
            processing_messages.append((message, progress))
            self.logger.info(message)
            if hasattr(self, 'main_gui') and hasattr(self.main_gui, 'update_status'):
                try:
                    self.main_gui.update_status(message, status_type=message_type)
                except Exception:
                    pass
            if hasattr(self, 'progress_queue') and self.progress_queue:
                self.progress_queue.put((message, progress))
        
        try:
            
            # Step 1: Load full resolution image
            add_progress_message(f"Loading image: {os.path.basename(image_path)}", None)

            original_image = load_image_to_np_array(image_path)
            if original_image is None:
                add_progress_message("Failed to load image", None, "error")
                return False

            # Store original image and its path for metadata preservation
            self.viz_manager.store_image_in_viz_cache(VisualizationType.ORIGINAL_IMAGE, original_image)
            self.viz_manager.store_original_image_path(image_path)

            # Check if previously processed
            previously_processed = self.file_manager.check_original_file_processed(image_path)
            
            # Step 2: Detect markers at full resolution
            add_progress_message("Detecting ArUco markers...", None)
            
            markers = self.aruco_manager.improve_marker_detection(original_image)
            
            # Calculate expected markers
            expected = set(self.config['corner_marker_ids'] + self.config['compartment_marker_ids'])
            missing = list(expected - set(markers.keys()))
            
            add_progress_message(f"Detected {len(markers)}/{len(expected)} ArUco markers", None)
            if missing:
                add_progress_message(f"Missing markers: {sorted(missing)}", None)
            
            # Step 3: Correct image orientation at full resolution
            add_progress_message("Correcting image orientation...", None)
            
            corrected_image, rotation_matrix, rotation_angle = self.aruco_manager.correct_image_skew(
                original_image, markers, return_transform=True
            )
            
            if corrected_image is not original_image:
                # Re-detect markers on corrected image
                markers = self.aruco_manager.improve_marker_detection(corrected_image)
                self.viz_manager.store_image_in_viz_cache(VisualizationType.CORRECTED_IMAGE, corrected_image)
                add_progress_message(f"Corrected orientation (angle: {rotation_angle:.2f}°)", None)
            else:
                corrected_image = original_image
                rotation_angle = 0.0
            
            # Step 4: Correct skewed markers
            if markers:
                add_progress_message("Correcting skewed markers...", None)
                corrected_markers = self.aruco_manager.correct_skewed_markers(markers)
                
                corrected_count = sum(1 for mid in markers 
                                    if not np.array_equal(markers[mid], corrected_markers[mid]))
                
                if corrected_count > 0:
                    markers = corrected_markers
                    add_progress_message(f"Corrected {corrected_count} skewed markers", None)
            
            # Step 5: Estimate scale from markers
            self._estimate_scale_from_markers(markers, add_progress_message)
            
            # Step 6: Extract compartment boundaries at full resolution
            add_progress_message("Analyzing compartment boundaries...", None)
            
            analysis = self.aruco_manager.analyze_compartment_boundaries(
                corrected_image, markers,
                compartment_count=self.config.get('compartment_count', 20),
                smart_cropping=True
            )
            
            boundaries = analysis['boundaries']
            vertical_constraints = analysis['vertical_constraints']
            marker_to_compartment = analysis['marker_to_compartment']
            
            if not boundaries:
                add_progress_message("Failed to extract compartment boundaries", None, "error")
                return False
            
            # Create and store boundary visualization
            boundaries_viz = self.viz_manager.create_boundary_visualization(
                corrected_image, boundaries, vertical_constraints
            )
            self.viz_manager.store_image_in_viz_cache(VisualizationType.BOUNDARIES_INITIAL, boundaries_viz)
            
            # Store full-resolution data
            self.viz_manager.store_metadata('markers', markers)
            self.viz_manager.store_metadata('boundaries', boundaries)
            self.viz_manager.store_metadata('vertical_constraints', vertical_constraints)
            self.viz_manager.store_metadata('rotation_angle', rotation_angle)
            
            # Step 7: Show registration dialog with DOWNSAMPLED visualization
            add_progress_message("Opening compartment registration dialog...", None)
            
            # Get display image and scale boundaries for dialog
            display_image = self.viz_manager.get_image_from_viz_cache(VisualizationType.CORRECTED_IMAGE, for_display=True)
            display_boundaries = self.viz_manager.scale_boundaries_to_display(boundaries)
            
            # Scale markers for display
            display_markers = {}
            for marker_id, corners in markers.items():
                display_markers[marker_id] = corners * self.viz_manager.display_scale_factor
            
            # Scale vertical constraints
            if vertical_constraints:
                display_constraints = self.viz_manager.scale_coordinates_to_display(vertical_constraints)
            else:
                display_constraints = None
            
            # Create metadata dict from previous processing or start fresh
            metadata = self._prepare_metadata(previously_processed, image_path)
            
            # Show dialog with downsampled data
            result = self._show_registration_dialog(
                display_image,
                display_boundaries,
                display_markers,
                missing,
                metadata,
                display_constraints,
                marker_to_compartment,
                rotation_angle,
                image_path
            )
            
            if not result:
                add_progress_message("Processing canceled by user", None)
                return False
            
            # Handle dialog result
            if result.get('quit', False):
                self.processing_complete = True
                return False
                
            if result.get('rejected', False):
                return self._handle_rejected_image(
                    image_path, result, metadata, processing_messages, add_progress_message
                )
            
            # Step 8: Scale adjustments back to full resolution
            final_boundaries = self._apply_dialog_adjustments(
                result, corrected_image, markers, boundaries
            )
            
            # Step 9: Extract and save compartments at full resolution
            compartments = self.extract_compartments_from_boundaries(corrected_image, final_boundaries)
            
            # Step 10: Handle duplicates and save
            final_metadata = {
                'hole_id': result.get('hole_id'),
                'depth_from': result.get('depth_from'),
                'depth_to': result.get('depth_to'),
                'compartment_interval': result.get('compartment_interval', 1)
            }
            
            return self._handle_duplicates_and_save(
                image_path, final_metadata, final_boundaries, 
                corrected_image, compartments, processing_messages, add_progress_message
            )
            
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            if hasattr(self, 'progress_queue'):
                self.progress_queue.put((f"Error: {str(e)}", None))
                
            return False

    def _show_registration_dialog(self, display_image, display_boundaries, display_markers,
                                missing_marker_ids, metadata, display_constraints,
                                marker_to_compartment, rotation_angle, image_path):
        """Show the registration dialog with downsampled visualization."""
        
        # Define callback for re-processing with adjustments
        def process_image_callback(params):
            # Scale adjustment parameters back to full resolution
            full_params = {
                'top_boundary': int(params['top_boundary'] / self.viz_manager.display_scale_factor),
                'bottom_boundary': int(params['bottom_boundary'] / self.viz_manager.display_scale_factor),
                'left_height_offset': int(params['left_height_offset'] / self.viz_manager.display_scale_factor),
                'right_height_offset': int(params['right_height_offset'] / self.viz_manager.display_scale_factor)
            }
            
            # Get full resolution data
            full_image = self.viz_manager.get_image_from_viz_cache(VisualizationType.CORRECTED_IMAGE)
            full_markers = self.viz_manager.get_metadata('markers')
            
            # Re-analyze at full resolution
            temp_metadata = {'boundary_adjustments': full_params}
            new_analysis = self.aruco_manager.analyze_compartment_boundaries(
                full_image, full_markers,
                compartment_count=self.config.get('compartment_count', 20),
                smart_cropping=True,
                metadata=temp_metadata
            )
            
            # Scale results back to display resolution
            display_boundaries = self.viz_manager.scale_boundaries_to_display(new_analysis['boundaries'])
            
            # Create display visualization
            display_viz = self.viz_manager.create_boundary_visualization(
                display_image, display_boundaries
            )
            
            return {
                'boundaries': display_boundaries,
                'visualization': display_viz
            }
        
        # Create the dialog with display-resolution data
        dialog = CompartmentRegistrationDialog(
            self.root,
            display_image,
            display_boundaries,
            missing_marker_ids,
            theme_colors=self.gui_manager.theme_colors,
            gui_manager=self.gui_manager,
            metadata=metadata,
            vertical_constraints=display_constraints,
            marker_to_compartment=marker_to_compartment,
            rotation_angle=rotation_angle,
            markers=display_markers,
            config=self.config,
            on_apply_adjustments=process_image_callback,
            image_path=image_path,
            scale_data=self.viz_manager.get_metadata('scale_data')
        )
        
        dialog.current_mode = 0  # Start in metadata mode
        dialog._update_mode_indicator()
        
        return dialog.show()

    def _apply_dialog_adjustments(self, result, full_image, full_markers, original_boundaries):
        """Apply dialog adjustments to full resolution boundaries."""
        
        if not any(key in result for key in ['top_boundary', 'bottom_boundary', 
                                            'left_height_offset', 'right_height_offset']):
            # No adjustments made
            return original_boundaries
        
        # Scale adjustments from display to full resolution
        adjustment_params = {
            'top_boundary': int(result.get('top_boundary', 0) / self.viz_manager.display_scale_factor),
            'bottom_boundary': int(result.get('bottom_boundary', 0) / self.viz_manager.display_scale_factor),
            'left_height_offset': int(result.get('left_height_offset', 0) / self.viz_manager.display_scale_factor),
            'right_height_offset': int(result.get('right_height_offset', 0) / self.viz_manager.display_scale_factor)
        }
        
        # Re-analyze boundaries at full resolution with adjustments
        metadata = {'boundary_adjustments': adjustment_params}
        
        adjusted_analysis = self.aruco_manager.analyze_compartment_boundaries(
            full_image, full_markers,
            compartment_count=self.config.get('compartment_count', 20),
            smart_cropping=True,
            metadata=metadata
        )
        
        return adjusted_analysis['boundaries']

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
        image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.heic', '.heif')
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
        
        # Get the original image with no annotations if available TODO - CHECK THIS - THIS MIGHT BE BREAKING THINGS! IF THE SYSTEM FALLS BACK TO AN UNEXPECTED FILE!
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
            self.logger.error(" Using small_image as original image")
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
                boundaries_viz=boundaries_viz,  # This will only be used as fallback TODO - REMOVE ALL FALLBACKS - ALL FALLBACKS SHOULD ERROR
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
