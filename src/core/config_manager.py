# core/config_manager.py

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigManager:
    """
    Manages application configuration with dual-config approach:
    - Default config (read-only, bundled with app)
    - User settings (writable, stored in AppData / GeoVue)
    """
    
    # Define which settings are user-editable
    USER_SETTINGS_KEYS = [
        "language", "theme", "program_initialized", "storage_type",
        "local_folder_path",
        "local_folder_images_to_process", "local_folder_processed_originals",
        "local_folder_approved_originals", "local_folder_rejected_originals",
        "local_folder_chip_compartments", "local_folder_approved_compartments",
        "local_folder_temp_review", "local_folder_drill_traces",
        "local_folder_debugging", "local_folder_blur_analysis",
        "local_folder_debug_images", 
         
        "shared_folder_path",
        "shared_folder_register_excel_path", "shared_folder_approved_folder",
        "shared_folder_processed_originals",
        "shared_folder_rejected_folder", "shared_folder_drill_traces",
        "shared_folder_register_path", "shared_folder_register_data_folder",
        "shared_folder_extracted_compartments_folder", "shared_folder_approved_compartments_folder", 
        "shared_folder_review_compartments_folder",

        "output_format", "jpeg_quality", "enable_blur_detection", 
        "blur_threshold", "blur_roi_ratio", "flag_blurry_images",
        "blurry_threshold_percentage", "save_blur_visualizations",
        "compartment_count", "compartment_interval", "valid_hole_prefixes",
        "enable_prefix_validation", "review_toggles",

        "compartment_marker_size_cm",
        "corner_marker_size_cm", 
        "metadata_marker_size_cm",
        "scale_estimation_enabled",
        "scale_outlier_threshold_iqr",
        "scale_max_cv_threshold",
        "scale_min_markers_required",
        "use_corner_markers_for_scale",
]
    
    def __init__(self, default_config_path: str):
        """
        Initialize configuration manager with default and user configs.
        
        Args:
            default_config_path: Path to default read-only config
        """
        self.logger = logging.getLogger(__name__)
        self.default_config_path = default_config_path
        
        # User settings always in AppData
        appdata = os.getenv('APPDATA')
        self.user_settings_dir = Path(appdata) / 'GeoVue'
        self.user_settings_path = self.user_settings_dir / 'settings.json'
        
        # Create settings directory if needed
        self.user_settings_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configurations
        self.default_config = self._load_default_config()
        self.user_settings = self._load_or_create_user_settings()
        
        # Merged config (user settings override defaults)
        self.config = self._merge_configs()
        
    def _load_default_config(self) -> Dict[str, Any]:
        """Load the default read-only configuration."""
        try:
            if os.path.exists(self.default_config_path):
                with open(self.default_config_path, 'r') as f:
                    return json.load(f)
            else:
                self.logger.warning(f"Default config not found at {self.default_config_path}")
                return {}
        except Exception as e:
            self.logger.error(f"Error loading default config: {e}")
            return {}
    
    def _load_or_create_user_settings(self) -> Dict[str, Any]:
        """Load user settings or create default user settings file."""
        if os.path.exists(self.user_settings_path):
            try:
                with open(self.user_settings_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading user settings: {e}")
                return self._create_default_user_settings()
        else:
            return self._create_default_user_settings()
    
    def _create_default_user_settings(self) -> Dict[str, Any]:
        """Create default user settings file with editable options."""
        default_settings = {
            "language": "en",
            "theme": "dark",
            "program_initialized": False,
            "local_folder_path": "",
            "output_format": "png",
            "jpeg_quality": 100,
            "enable_blur_detection": True,
            "blur_threshold": 207.24,
            "blur_roi_ratio": 0.8,
            "flag_blurry_images": False,
            "blurry_threshold_percentage": 10.0,
            "save_blur_visualizations": True,
            "compartment_count": 20,
            "compartment_interval": 1,
            "valid_hole_prefixes": ["BA", "NB", "SB", "KM"],
            "enable_prefix_validation": True,
            "shared_folder_path": "",
            "storage_type": "",  # "cloud" or "local" or "both"
            "shared_folder_approved_compartments_folder": "",
            "shared_folder_extracted_compartments_folder": "",
            "shared_folder_review_compartments_folder": "",
            "shared_folder_approved_folder": "",
            "shared_folder_processed_originals": "",
            "shared_folder_rejected_folder": "",
            "shared_folder_drill_traces": "",
            "shared_folder_register_path": "",
            "shared_folder_register_data_folder": "",
            "review_toggles": [
                "Bad Image", "BIFf", "Compact", 
                "Porous", "+ QZ", "+ CHH/M"
            ]
        }
        
        # Save the default settings
        self._save_user_settings(default_settings)
        return default_settings
    
    def _merge_configs(self) -> Dict[str, Any]:
        """Merge default config with user settings (user settings take precedence)."""
        merged = self.default_config.copy()
        
        # Only override with user settings that are meant to be user-editable
        for key in self.USER_SETTINGS_KEYS:
            if key in self.user_settings:
                merged[key] = self.user_settings[key]
                
        return merged
    
    def _save_user_settings(self, settings: Optional[Dict[str, Any]] = None) -> None:
        """Save user settings to file."""
        try:
            if settings is None:
                settings = self.user_settings

            # Ensure directory exists before writing
            self.user_settings_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.user_settings_path, 'w') as f:
                json.dump(settings, f, indent=4)

            self.logger.debug(f"✅ Saved user settings to {self.user_settings_path}")
        except Exception as e:
            self.logger.error(f"❌ Error saving user settings: {e}")
            raise

   
    def _load_language_preference(self):
        """Load language preference using ConfigManager."""
        try:
            language = self.config_manager.get('language', 'en')
            self.translator.set_language(language)
            self.logger.info(f"Loaded language preference: {language}")
        except Exception as e:
            self.logger.error(f"Error loading language preference: {str(e)}")
    def _save_language_preference(self, language_code):
        """Save language preference using ConfigManager."""
        try:
            self.config_manager.set('language', language_code)
            self.logger.info(f"Saved language preference: {language_code}")
        except Exception as e:
            self.logger.error(f"Error saving language preference: {str(e)}")




    def get(self, key: str, default=None):
        """Get a configuration value."""
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value in user settings."""
        if key not in self.USER_SETTINGS_KEYS:
            self.logger.warning(f"Attempted to modify non-user setting: {key}")
            return
            
        # Update both user settings and merged config
        self.user_settings[key] = value
        self.config[key] = value
        self._save_user_settings()

    def as_dict(self) -> Dict[str, Any]:
        """Get the merged configuration as a dictionary."""
        return self.config.copy()

    def is_first_run(self) -> bool:
        """Check if this is the first run (no local folder path set)."""
        return 'local_folder_path' not in self.user_settings

    def mark_initialized(self):
        """Mark the application as initialized."""
        self.set('program_initialized', True)

        # DEBUG: Confirm it's saved correctly
        try:
            if self.user_settings_path.exists():
                with open(self.user_settings_path, 'r') as f:
                    updated_settings = json.load(f)
                program_flag = updated_settings.get("program_initialized")
                self.logger.info(f"✅ program_initialized flag after save: {program_flag}")
            else:
                self.logger.warning(f"⚠️ settings.json not found at {self.user_settings_path}")
        except Exception as e:
            self.logger.error(f"❌ Failed to verify updated settings.json: {e}")
