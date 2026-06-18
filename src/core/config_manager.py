# core/config_manager.py

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from enum import Enum


class EnumSafeJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles Enum objects by converting them to their values.
    
    This is a safety net for cases where enum objects (like DataType, NullHandling)
    accidentally end up in the settings dict without being converted via to_dict().
    """
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


class ConfigManager:
    """
    Manages application configuration with dual-config approach:
    - Default config (read-only, bundled with app)
    - User settings (writable, stored in AppData / GeoVue)
    """

    # Define which settings are user-editable
    USER_SETTINGS_KEYS = [
        "language",
        "theme",
        "classifications",
        "tags",
        "viz_columns",  # Visualization column configuration - logging review dialog
        "viz_column_width_ratio",  # Width ratio for viz columns (0.0-1.0)
        "viz_column_height",  # Default column height for landscape mode
        "viz_column_font_size",  # Font size for viz column text
        "viz_header_font_size",  # Font size for viz column headers
        "viz_value_font_size",  # Font size for viz column values
        "viz_decimal_places",  # Decimal places for viz column numeric values
        "grid_show_outlines",  # Show/hide cell outlines
        "grid_outline_width",  # Cell outline width in pixels
        "grid_show_cell_labels",  # Show/hide hole ID and depth labels
        "grid_show_classification_labels",  # Show/hide classification labels
        "grid_classification_label_position",  # "top-right" or "top-left"
        "color_maps",  # Color map presets for data visualizations
        "program_initialized",
        "storage_type",
        "local_folder_path",
        "local_folder_images_to_process",
        "local_folder_processed_originals",
        "local_folder_approved_originals",
        "local_folder_rejected_originals",
        "local_folder_chip_compartments",
        "local_folder_approved_compartments",
        "local_folder_temp_review",
        "local_folder_drill_traces",
        "local_folder_debugging",
        "local_folder_blur_analysis",
        "local_folder_debug_images",
        "shared_folder_path",
        "shared_folder_register_excel_path",
        "shared_folder_datasets",  # Folder containing drillhole CSVs (replaces single CSV path)
        "shared_folder_approved_folder",
        "shared_folder_processed_originals",
        "shared_folder_rejected_folder",
        "shared_folder_drill_traces",
        "shared_folder_register_path",
        "shared_folder_register_data_folder",
        "shared_folder_extracted_compartments_folder",
        "shared_folder_approved_compartments_folder",
        "shared_folder_review_compartments_folder",
        "shared_folder_cross_sections",  # Folder containing section PDFs (Section Tool integration)
        "output_format",
        "jpeg_quality",
        "enable_blur_detection",
        "blur_threshold",
        "blur_roi_ratio",
        "flag_blurry_images",
        "blurry_threshold_percentage",
        "save_blur_visualizations",
        "compartment_count",
        "compartment_interval",
        "valid_hole_prefixes",
        "enable_prefix_validation",
        "review_toggles",
        "compartment_marker_size_cm",
        "corner_marker_size_cm",
        "metadata_marker_size_cm",
        "compartment_spacing_mm",
        "compartment_spacing_overrides",
        "scale_estimation_enabled",
        "scale_outlier_threshold_iqr",
        "scale_max_cv_threshold",
        "scale_min_markers_required",
        "use_corner_markers_for_scale",
        "zoom_width",
        "zoom_height",
        "zoom_scale",
        "canvas_update_delay_ms",
        "min_marker_size",
        "adjustment_step_px",
        "debounce_interval_s",
        "marker_preview_size",
        "static_zoom_width",
        "static_zoom_height",
        "static_zoom_region_size",
        "zoom_region_size",
        "auto_loop_mode",
        "auto_loop_min_markers",
        "auto_loop_require_filename_metadata",
        # Drillhole Correlation settings
        "correlation_dialog_width",
        "correlation_dialog_height",
        "correlation_default_section_width",
        "correlation_minimap_bg_color",
        "correlation_collar_color",
        "correlation_assay_collar_color",
        "correlation_no_assay_collar_color",
        "correlation_collar_size",
        "correlation_selected_color",
        "correlation_selected_size",
        "correlation_line_color",
        "correlation_line_width",
        "correlation_box_color",
        "correlation_zoom_default",
        "correlation_zoom_min",
        "correlation_zoom_max",
        "correlation_column_width",
        "correlation_column_spacing",
        "correlation_cell_height",
        "correlation_data_viz_cell_height",
        # Data visualization settings
        "correlation_data_column_width",
        "correlation_data_column_min_width",
        "correlation_data_column_max_width",
        "correlation_thumbnail_width",
        "correlation_thumbnail_min_width",
        "correlation_thumbnail_max_width",
        "correlation_lazy_thumbnail_margin_px",
        "correlation_lazy_thumbnail_batch_size",
        "correlation_verbose_interval_logging",
        "correlation_debug_minimap",
        "correlation_color_bar_width",
        "correlation_image_mode",
        "correlation_show_images",
        "correlation_viz_columns",
        # DataCoordinator settings
        "geological_data_sources",
        # Column settings dialog
        "column_schemas",
        "register_column_settings",
        # QAQC settings
        "qaqc_rules",  # User-defined QAQC validation rules
        # Snowflake connection
        "snowflake_user",  # User's Snowflake login email
        "snowflake_custom_tables",  # User-added Snowflake table definitions [{database, schema, table, geovue_name, enabled}]
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
        appdata = os.getenv("APPDATA")
        if not appdata:
            # Fallback for non-Windows or missing APPDATA
            appdata = os.path.expanduser("~/.config")

        self.user_settings_dir = Path(appdata) / "GeoVue"
        self.user_settings_path = self.user_settings_dir / "settings.json"

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
                with open(self.default_config_path, "r") as f:
                    return json.load(f)
            else:
                self.logger.warning(
                    f"Default config not found at {self.default_config_path}"
                )
                return {}
        except Exception as e:
            self.logger.error(f"Error loading default config: {e}")
            return {}

    def _load_or_create_user_settings(self) -> Dict[str, Any]:
        """Load user settings or create default user settings file."""
        if os.path.exists(self.user_settings_path):
            try:
                with open(self.user_settings_path, "r") as f:
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
            "auto_loop_mode": False,
            "auto_loop_min_markers": 20,
            "auto_loop_require_filename_metadata": True,
            "compartment_count": 20,
            "compartment_interval": 1,
            "compartment_spacing_mm": 3.0,
            "compartment_spacing_mm_overrides": {"10-11": 5.0},
            "zoom_width": 200,
            "zoom_height": 300,
            "zoom_scale": 2,
            "canvas_update_delay_ms": 100,
            "min_marker_size": 20,
            "adjustment_step_px": 1,
            "debounce_interval_s": 0.1,
            "marker_preview_size": 40,
            "static_zoom_width": 200,
            "static_zoom_height": 400,
            "static_zoom_region_size": 100,
            "zoom_region_size": 100,
            "valid_hole_prefixes": ["BA", "NB", "SB", "KM", "OK", "BB", "BT"],
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
            "shared_folder_cross_sections": "",
            "compartment_pattern": r"([A-Z]{2}\d{4})_CC_(\d+)(?:_(Wet|Dry))?(?:_.*)?\.(?:png|tiff|jpg)$",
            "review_toggles": [
                "Bad Image",
                "BIFf",
                "Compact",
                "Porous",
                "+ QZ",
                "+ CHH/M",
            ],
            "viz_columns": [
                {"column": "Fe_pct_BEST", "color_map": "fe_grade"},
                {"column": "SiO2_pct_BEST", "color_map": "sio2_grade"},
                {"column": "Al2O3_pct_BEST", "color_map": "al2o3_grade"},
                {"column": "Logged_pct_CHHM", "color_map": "fe_grade"}
            ],
            # Correlation dialog settings
            "correlation_column_width": 200,
            "correlation_column_spacing": 20,
            "correlation_cell_height": 100,
            "correlation_data_viz_cell_height": 10,  # 10px per meter for pure data viz
            "correlation_zoom_default": 1.0,
            "correlation_zoom_min": 0.1,
            "correlation_zoom_max": 5.0,
            "correlation_collar_color": "#3498DB",
            "correlation_assay_collar_color": "#2ECC71",
            "correlation_no_assay_collar_color": "#F1C40F",
            # Correlation dialog Data visualization settings
            "correlation_data_column_width": 70,
            "correlation_data_column_min_width": 50,
            "correlation_data_column_max_width": 260,
            "correlation_thumbnail_width": 160,
            "correlation_thumbnail_min_width": 40,
            "correlation_thumbnail_max_width": 420,
            "correlation_lazy_thumbnail_margin_px": 250,
            "correlation_lazy_thumbnail_batch_size": 16,
            "correlation_verbose_interval_logging": False,
            "correlation_debug_minimap": False,
            "correlation_color_bar_width": 20,
            "correlation_image_mode": "thumbnail",
            "correlation_show_images": False,  # Show images in drillhole columns
            "correlation_viz_columns": [
                {
                    "column": "Fe_pct_BEST",
                    "color_map": "fe_grade",
                    "type": "bar",
                    "scale_mode": "raw",
                    "min_value": 0.0,
                    "max_value": 70.0,
                    "auto_scale": True
                },
                {
                    "column": "SiO2_pct_BEST",
                    "color_map": "sio2_grade",
                    "type": "bar",
                    "scale_mode": "raw",
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "auto_scale": True
                },
                {
                    "column": "Al2O3_pct_BEST",
                    "color_map": "al2o3_grade",
                    "type": "bar",
                    "scale_mode": "raw",
                    "min_value": 0.0,
                    "max_value": 30.0,
                    "auto_scale": True
                }
            ],
            # Color map presets (persisted)
            "color_maps": {},
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

            with open(self.user_settings_path, "w") as f:
                # Use EnumSafeJSONEncoder to handle any enum objects that weren't
                # converted via to_dict() - this is a safety net
                json.dump(settings, f, indent=4, cls=EnumSafeJSONEncoder)

            self.logger.debug(f"Saved user settings to {self.user_settings_path}")
        except Exception as e:
            self.logger.error(f"Error saving user settings: {e}")
            raise

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
        return "local_folder_path" not in self.user_settings

    def mark_initialized(self):
        """Mark the application as initialized."""
        self.set("program_initialized", True)

        # DEBUG: Confirm it's saved correctly
        try:
            if self.user_settings_path.exists():
                with open(self.user_settings_path, "r") as f:
                    updated_settings = json.load(f)
                program_flag = updated_settings.get("program_initialized")
                self.logger.info(
                    f"✅ program_initialized flag after save: {program_flag}"
                )
            else:
                self.logger.warning(
                    f"⚠️ settings.json not found at {self.user_settings_path}"
                )
        except Exception as e:
            self.logger.error(f"❌ Failed to verify updated settings.json: {e}")
