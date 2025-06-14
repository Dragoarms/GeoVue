# utils/onedrive_path_manager.py

import os
import logging
from typing import Optional
from tkinter import filedialog
from core.translator import TranslationManager

# Local imports
from gui.dialog_helper import DialogHelper


class OneDrivePathManager:
    """
    Manages finding OneDrive/shared paths for the project based on user configuration.
    """
    
    def __init__(self, root=None, file_manager=None, config_manager=None, silent=False):
        """
        Initialize the OneDrive path manager.
        
        Args:
            root: Optional tkinter root for dialogs
            file_manager: Optional FileManager instance for folder structure
            config_manager: Optional ConfigManager to get user settings
            silent: If True, don't show any dialogs (for initialization)
        """
        self.logger = logging.getLogger(__name__)
        self.root = root
        self.file_manager = file_manager
        self.config_manager = config_manager
        self.silent = silent
        
        # Get folder names from FileManager if available
        if file_manager:
            # Use FileManager's folder names (just the folder name, not full path)
            self.FOLDER_NAMES = {
                'register': 'Chip Tray Register',
                'images': 'Images to Process',
                'processed': 'Processed Original Images',
                'compartments': 'Extracted Compartment Images',
                'traces': 'Drillhole Traces',
                'debugging': 'Debugging'
            }
            self.SUBFOLDER_NAMES = {
                'approved': 'Approved Originals',
                'rejected': 'Rejected Originals',
                'blur': 'Blur Analysis',
                'debug': 'Debug Images',
                'register_data': 'Register Data (Do not edit)'
            }
            self.EXCEL_REGISTER_NAME = "Chip_Tray_Register.xlsx"
        else:
            # Fallback to correct structure (matching FileManager)
            self.FOLDER_NAMES = {
                'register': 'Chip Tray Register',
                'images': 'Images to Process',
                'processed': 'Processed Original Images',
                'compartments': 'Extracted Compartment Images',
                'traces': 'Drillhole Traces',
                'debugging': 'Debugging'
            }
            self.SUBFOLDER_NAMES = {
                'approved': 'Approved Originals',
                'rejected': 'Rejected Originals',
                'blur': 'Blur Analysis',
                'debug': 'Debug Images',
                'register_data': 'Register Data (Do not edit)'
            }
            self.EXCEL_REGISTER_NAME = "Chip_Tray_Register.xlsx"
        
        # Get base paths from config if available
        self._shared_base_path = None
        self._chip_tray_folder_path = None
        
        if config_manager:
            # Get the shared folder path from config (set during first run)
            shared_path = config_manager.get('shared_folder_path')
            if shared_path and os.path.exists(shared_path):
                self._shared_base_path = shared_path
                # If shared path is directly the chip tray folder, use it
                if os.path.basename(shared_path) == "GeoVue Chip Tray Photos":
                    self._chip_tray_folder_path = shared_path
                else:
                    # Check if GeoVue folder exists under shared path
                    geovue_path = os.path.join(shared_path, "GeoVue Chip Tray Photos")
                    if os.path.exists(geovue_path):
                        self._chip_tray_folder_path = geovue_path
                
                if self._chip_tray_folder_path:
                    self.logger.info(f"Using configured shared folder path: {self._chip_tray_folder_path}")
            else:
                self.logger.info("No shared folder path configured")
        
        # Cache found paths to avoid repetitive searching
        self._approved_folder_path = None
        self._register_path = None
        self._processed_originals_path = None
        self._drill_traces_path = None
        self._rejected_folder_path = None
        self._register_data_path = None


    def _get_path(self, path_type: str, config_key: str, 
                    dialog_title: str, dialog_message: str,
                    is_file: bool = False) -> Optional[str]:
            """
            Generic method to get a path - checks cache first, then config, then prompts user.
            
            Args:
                path_type: Type of path for cache attribute (e.g., '_approved_folder_path')
                config_key: Key in settings.json (e.g., 'onedrive_approved_folder')
                dialog_title: Title for browse dialog if path not found
                dialog_message: Message for confirmation dialog if path not found
                is_file: True if looking for a file, False for directory
                
            Returns:
                Path if found, None otherwise
            """
            # ===================================================
            # Check cached path first - no logging for cache hits
            # ===================================================
            cached_path = getattr(self, path_type, None)
            if cached_path is not None and os.path.exists(cached_path):
                return cached_path
            
            # Try to get from config
            if self.config_manager:
                config_path = self.config_manager.get(config_key)
                if config_path and os.path.exists(config_path):
                    setattr(self, path_type, config_path)
                    self.logger.info(f"Loaded {config_key} from config: {config_path}")
                    return config_path
            
            # If not in config and not silent, prompt user
            if self.root is not None and not self.silent:
                if DialogHelper.confirm_dialog(
                    self.root,
                    DialogHelper.t("Path Not Found"),
                    dialog_message
                ):
                    if is_file:
                        selected_path = filedialog.askopenfilename(
                            title=dialog_title,
                            filetypes=[("Excel files", "*.xlsx")] if config_key == 'onedrive_register_path' else []
                        )
                    else:
                        selected_path = filedialog.askdirectory(title=dialog_title)
                        
                    if selected_path:
                        setattr(self, path_type, selected_path)
                        self.logger.info(f"User selected {config_key}: {selected_path}")
                        # Save to config
                        if self.config_manager:
                            self.config_manager.set(config_key, selected_path)
                        return selected_path
            
            self.logger.warning(f"Path not found for {config_key}")
            return None
    

    def get_chip_tray_folder_path(self) -> Optional[str]:
        """Get the path to the Chip Tray Photos folder."""
        return self._get_path(
            '_chip_tray_folder_path',
            'shared_folder_path',
            DialogHelper.t('Select Chip Tray Photos Folder'),
            DialogHelper.t('The configured chip tray folder path does not exist. Would you like to browse for the correct location?')
        )

    def get_register_data_path(self) -> Optional[str]:
        """Get the path to the Register Data folder."""
        return self._get_path(
            '_register_data_path',
            'onedrive_register_data_folder',
            DialogHelper.t('Select Register Data Folder'),
            DialogHelper.t('The Register Data folder path does not exist. Would you like to browse for the correct location?')
        )

    def get_approved_folder_path(self) -> Optional[str]:
        """Get the path to the approved images folder."""
        return self._get_path(
            '_approved_folder_path',
            'onedrive_approved_folder',
            DialogHelper.t('Select Approved Images Folder'),
            DialogHelper.t('The Approved Images folder path does not exist. Would you like to browse for the correct location?')
        )

    def get_register_path(self) -> Optional[str]:
        """Get the path to the Excel register file."""
        return self._get_path(
            '_register_path',
            'onedrive_register_path',
            DialogHelper.t('Select Excel Register File'),
            DialogHelper.t('The Excel register path does not exist. Would you like to browse for the correct location?'),
            is_file=True
        )

    def get_processed_originals_path(self) -> Optional[str]:
        """Get the path to the Processed Originals folder."""
        return self._get_path(
            '_processed_originals_path',
            'onedrive_processed_originals',
            DialogHelper.t('Select Processed Originals Folder'),
            DialogHelper.t('The Processed Originals path does not exist. Would you like to browse for the correct location?')
        )

    def get_drill_traces_path(self) -> Optional[str]:
        """Get the path to the Drill Traces folder."""
        return self._get_path(
            '_drill_traces_path',
            'onedrive_drill_traces',
            DialogHelper.t('Select Drill Traces Folder'),
            DialogHelper.t('The Drill Traces path does not exist. Would you like to browse for the correct location?')
        )

    def get_rejected_folder_path(self) -> Optional[str]:
        """Get the path to the Rejected folder."""
        return self._get_path(
            '_rejected_folder_path',
            'onedrive_rejected_folder',
            DialogHelper.t('Select Rejected Folder'),
            DialogHelper.t('The Rejected folder path does not exist. Would you like to browse for the correct location?')
        )