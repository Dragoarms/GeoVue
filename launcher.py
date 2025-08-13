# launcher.py
"""
Production-ready launcher with proper separation of concerns and error handling.
"""

import sys
import os
import time
import threading
import logging
from typing import Optional, Any, TYPE_CHECKING
import gc

# Configure logging for production diagnostics
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Type checking guard for PyInstaller
if TYPE_CHECKING:

    class _ModifiedSys:
        _MEIPASS: str
        frozen: bool

else:
    _ModifiedSys = sys


def setup_paths() -> str:
    """
    Setup Python path for frozen and development environments.

    Returns:
        str: Base path for the application

    Raises:
        RuntimeError: If path setup fails
    """
    try:
        if getattr(sys, "frozen", False):
            # Running as compiled executable
            modified_sys = sys  # type: Any
            base_path = modified_sys._MEIPASS
            logger.info(f"Running frozen app from: {base_path}")
            return base_path
        else:
            # Running in development
            base_path = os.path.dirname(os.path.abspath(__file__))
            src_path = os.path.join(base_path, "src")

            if not os.path.exists(src_path):
                raise RuntimeError(f"Source directory not found: {src_path}")

            if src_path not in sys.path:
                sys.path.insert(0, src_path)
                logger.info(f"Added to Python path: {src_path}")

            return base_path

    except Exception as e:
        logger.error(f"Path setup failed: {e}")
        raise RuntimeError(f"Failed to setup paths: {str(e)}")


def inject_launcher_config(cleanup_required: bool = True, splash_instance=None) -> bool:
    """
    Inject launcher configuration for main app communication.
    """
    try:
        import types

        # Create configuration module
        launcher_config = types.ModuleType("launcher_config")
        setattr(launcher_config, "cleanup_required", cleanup_required)
        setattr(launcher_config, "launcher_was_used", True)
        setattr(launcher_config, "splash_was_shown", cleanup_required)
        setattr(launcher_config, "injection_timestamp", time.time())
        setattr(
            launcher_config, "splash_instance", splash_instance
        )  # Assign dynamically to avoid type checker error

        # Inject into sys.modules
        sys.modules["launcher_config"] = launcher_config

        logger.info("Launcher configuration injected successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to inject launcher config: {e}")
        return False


class SplashManager:
    """
    Thread-safe splash screen manager with proper cleanup.
    """

    def __init__(self):
        self.splash: Optional[Any] = None
        self._lock = threading.Lock()
        self._closed = False
        self._close_timer: Optional[threading.Timer] = None
        self._splash_shown = False

    def create_splash(self) -> bool:
        """
        Create and show splash screen with error recovery.

        Returns:
            bool: True if splash was created successfully
        """
        with self._lock:
            if self._closed:
                return False

        try:
            from gui.splash_screen import SplashScreen

            # Create splash screen (it will use its own shared root internally)
            self.splash = SplashScreen()
            self.splash.show()
            self._splash_shown = True
            logger.info("Splash screen created successfully")

            # Access the actual Toplevel window inside the wrapper
            splash_window = self.splash.splash  # This is the tk.Toplevel

            # bump it above every window just long enough to grab the front
            splash_window.attributes("-topmost", True)
            splash_window.lift()  # now it's really on top
            # then immediately clear the topmost flag
            splash_window.after_idle(
                lambda: splash_window.attributes("-topmost", False)
            )

            return True
        except Exception as e:
            logger.warning(f"Could not create splash screen: {e}")
            self.splash = None
            return False

    def was_shown(self) -> bool:
        """Check if splash was successfully shown."""
        return self._splash_shown

    def update_status(self, message: str) -> None:
        """Thread-safe status update."""
        with self._lock:
            if self.splash is not None and not self._closed:
                try:
                    self.splash.update_status(message)
                except Exception as e:
                    logger.debug(f"Could not update splash status: {e}")

    def _close_splash_safe(self) -> None:
        """Thread-safe splash closure that runs on main thread."""
        try:
            if self.splash is not None and not self._closed:
                self.splash.close()
                self._closed = True
                self.splash = None
                logger.info("Splash screen closed safely")
        except Exception:
            pass

    def close_immediately(self) -> None:
        """Force immediate splash closure (thread-safe)."""
        with self._lock:
            # Cancel any pending timer
            if self._close_timer is not None:
                self._close_timer.cancel()
                self._close_timer = None

            # Use thread-safe close
            self._close_splash_safe()

    def _close_splash(self) -> None:
        """Internal splash cleanup with resource management."""
        with self._lock:
            if self.splash is not None and not self._closed:
                try:
                    self.splash.close()
                    logger.info("Splash screen closed")
                except Exception as e:
                    logger.error(f"Error closing splash: {e}")
                finally:
                    self._closed = True
                    self.splash = None

                    # Force garbage collection
                    gc.collect()


def show_error_dialog(error_message: str) -> None:
    """Display error dialog with console fallback."""
    logger.error(f"Showing error dialog: {error_message}")

    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("GeoVue Error", error_message)
        root.destroy()
    except Exception as e:
        logger.error(f"Could not show error dialog: {e}")
        print(f"ERROR: {error_message}")


def main() -> int:
    """
    Main entry point with production-grade error handling.

    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    logger.info("Launcher starting...")
    splash_manager = SplashManager()

    try:
        # Step 1: Setup paths
        try:
            base_path = setup_paths()
        except RuntimeError as e:
            show_error_dialog(str(e))
            return 1

        # Step 2: Create splash screen
        has_splash = splash_manager.create_splash()

        if has_splash:
            splash_manager.update_status("Loading modules...")

        # Step 3: Inject configuration based on splash status
        # Pass the splash instance if it exists
        config_injected = inject_launcher_config(
            cleanup_required=has_splash,
            splash_instance=splash_manager.splash if has_splash else None,
        )
        if not config_injected:
            logger.warning("Failed to inject launcher config, continuing anyway")

        # Step 4: Import main module
        try:
            import main as main_module
        except ImportError as e:
            error_msg = f"Failed to import main module: {str(e)}"
            logger.error(error_msg)

            if has_splash:
                splash_manager.close_immediately()

            show_error_dialog(error_msg)
            return 1

        if has_splash:
            splash_manager.update_status("Starting GeoVue...")

        # Step 5: Verify main entry point
        if not hasattr(main_module, "main"):
            error_msg = "Main module missing 'main' function"
            logger.error(error_msg)

            if has_splash:
                splash_manager.close_immediately()

            show_error_dialog(error_msg)
            return 1

        # Step 7: Run the main application
        logger.info("Starting main application...")

        try:
            main_module.main()
            logger.info("Application exited normally")
            return 0

        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
            return 130  # Standard Unix exit code for SIGINT

        except SystemExit as e:
            exit_code = e.code if isinstance(e.code, int) else 0
            logger.info(f"Application exited with code: {exit_code}")
            return exit_code

        except Exception as e:
            error_msg = f"Application error: {str(e)}"
            logger.error(error_msg, exc_info=True)

            splash_manager.close_immediately()
            show_error_dialog(f"Failed to start GeoVue:\n\n{str(e)}")
            return 1

    except Exception as e:
        error_msg = f"Launcher error: {str(e)}"
        logger.critical(error_msg, exc_info=True)

        splash_manager.close_immediately()
        show_error_dialog(error_msg)
        return 1

    finally:
        splash_manager.close_immediately()
        logger.info("Launcher cleanup complete")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
