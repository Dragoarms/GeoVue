"""Platform configuration definitions for GeoVue Capture hardware variants."""

from .base import PlatformConfig
from .leadscrew_dm556 import LeadscrewDM556Config
from .pilbara_tmc2225 import PilbaraTMC2225Config

PLATFORMS = {
    "leadscrew": LeadscrewDM556Config,
    "pilbara": PilbaraTMC2225Config,
}


def detect_platform():
    """Auto-detect which platform we're running on based on hardware hints."""
    import os

    # Check for touchscreen (Pilbara has one)
    touchscreen_paths = [
        "/dev/input/touchscreen",
        "/sys/class/input/event0/device/name",
    ]
    for path in touchscreen_paths:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    if "touch" in f.read().lower():
                        return "pilbara"
            except (IOError, PermissionError):
                pass

    # Check for platform marker file
    marker = os.path.expanduser("~/.geovue_platform")
    if os.path.exists(marker):
        with open(marker) as f:
            name = f.read().strip().lower()
            if name in PLATFORMS:
                return name

    # Default to leadscrew
    return "leadscrew"


def get_platform_config(name=None):
    """Get platform configuration by name, or auto-detect."""
    if name is None:
        name = detect_platform()
    if name not in PLATFORMS:
        raise ValueError(f"Unknown platform '{name}'. Available: {list(PLATFORMS.keys())}")
    return PLATFORMS[name]()
