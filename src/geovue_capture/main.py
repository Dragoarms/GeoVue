#!/usr/bin/env python3
"""
GeoVue Capture 2.8.0 - Entry point.

Usage:
    sudo python3 -m geovue_capture [--platform leadscrew|pilbara] [--lang en|fr]
"""

import argparse
import logging
import sys

from . import __version__, __app_name__
from .platforms import get_platform_config, PLATFORMS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description=f"{__app_name__} v{__version__}",
    )
    parser.add_argument(
        "--platform",
        choices=list(PLATFORMS.keys()),
        default=None,
        help="Hardware platform (auto-detected if omitted)",
    )
    parser.add_argument(
        "--lang",
        choices=["en", "fr"],
        default="en",
        help="UI language (default: en)",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Launch in fullscreen/kiosk mode",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{__app_name__} {__version__}",
    )

    args = parser.parse_args()

    # Set language
    from .gui.i18n import set_language
    set_language(args.lang)

    # Load platform config
    config = get_platform_config(args.platform)
    logger.info(f"{__app_name__} v{__version__}")
    logger.info(f"Platform: {config.platform_name} - {config.platform_description}")

    # Ensure storage directories exist
    config.ensure_directories()

    # Initialize hardware
    from .hardware.gpio_manager import create_gpio_backend
    gpio = create_gpio_backend()

    from .hardware.limit_switches import LimitSwitchMonitor
    limits = LimitSwitchMonitor(
        gpio=gpio,
        home_pin=config.limit_switch_pins.home,
        far_pin=config.limit_switch_pins.far,
        ground_pin=config.limit_switch_pins.ground,
    )
    limits.start()

    from .hardware.stepper import StepperController
    stepper = StepperController(config=config, gpio=gpio, limit_monitor=limits)

    # Launch GUI
    try:
        from .gui.capture_app import GeoVueCaptureApp
        app = GeoVueCaptureApp(
            config=config,
            stepper=stepper,
            limits=limits,
            fullscreen=args.fullscreen,
        )
        app.run()
    except ImportError:
        logger.error(
            "GUI module not yet built. Run Claude Code on the Pi to complete "
            "gui/capture_app.py (see CLAUDE.md for instructions)."
        )
        sys.exit(1)
    finally:
        stepper.shutdown()
        limits.stop()
        gpio.cleanup()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
