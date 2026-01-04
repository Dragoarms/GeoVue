# src\resources\__init__.py

import importlib.resources as pkg_resources
import pathlib
from pathlib import Path
import sys
import logging

logger = logging.getLogger(__name__)


def get_logo_path():
    return pkg_resources.files(__package__) / "logo.ico"


def get_translations_path():
    return pkg_resources.files(__package__) / "translations.csv"


def get_excel_template_path():
    """Get the path to the Excel register template file."""
    return (
        pkg_resources.files(__package__)
        / "Register Template File"
        / "Chip_Tray_Register.xltx"
    )


def get_color_preset_path(preset_name: str) -> Path:
    """
    Get the path to a color preset JSON file.

    Args:
        preset_name: Name of the preset file (e.g., 'fe_grade.json', 'lithology.json')

    Returns:
        Path to the color preset file

    Raises:
        FileNotFoundError: If the preset file doesn't exist
    """
    # First try using importlib.resources (preferred method)
    try:
        preset_path = pkg_resources.files(__package__) / "color_presets" / preset_name
        if preset_path.exists():
            return preset_path
    except Exception:
        pass

    # Fallback to file system paths for PyInstaller compatibility
    possible_paths = [
        # Development path
        Path(__file__).parent / "color_presets" / preset_name,
        # Packaged path (same directory structure)
        Path(sys.executable).parent / "resources" / "color_presets" / preset_name,
        # Alternative packaged path
        Path(sys.executable).parent
        / "_internal"
        / "resources"
        / "color_presets"
        / preset_name,
    ]

    for path in possible_paths:
        if path.exists():
            logger.debug(f"Found color preset at: {path}")
            return path

    # If not found, raise error
    raise FileNotFoundError(f"Color preset file not found: {preset_name}")


def list_color_presets() -> list:
    """
    List all available color preset files.

    Returns:
        List of color preset filenames
    """
    presets = []

    # First try using importlib.resources
    try:
        color_presets_dir = pkg_resources.files(__package__) / "color_presets"
        if color_presets_dir.is_dir():
            presets = [
                f.name for f in color_presets_dir.iterdir() if f.name.endswith(".json")
            ]
            if presets:
                return presets
    except Exception:
        pass

    # Fallback to file system paths
    preset_dirs = [
        Path(__file__).parent / "color_presets",
        Path(sys.executable).parent / "resources" / "color_presets",
        Path(sys.executable).parent / "_internal" / "resources" / "color_presets",
    ]

    for preset_dir in preset_dirs:
        if preset_dir.exists() and preset_dir.is_dir():
            return [f.name for f in preset_dir.glob("*.json")]

    return []
