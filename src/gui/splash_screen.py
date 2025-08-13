# src/gui/splash_screen.py
import tkinter as tk
from tkinter import ttk
import os
import sys
from PIL import Image, ImageTk
from typing import Optional


class SplashScreen:
    """Lightweight splash screen that shows immediately on startup."""

    _shared_root: Optional[tk.Tk] = None  # Class variable to track shared root

    @classmethod
    def get_shared_root(cls) -> tk.Tk:
        """Get or create a shared root window for the application."""
        if cls._shared_root is None or not cls._shared_root.winfo_exists():
            cls._shared_root = tk.Tk()
            cls._shared_root.withdraw()  # Hide the actual root
        return cls._shared_root

    def __init__(self, parent_root: Optional[tk.Tk] = None):
        """
        Initialize splash screen.

        Args:
            parent_root: Optional parent Tk instance to use. If None, creates/uses shared root.
        """
        # Use provided root or get shared root
        self.parent_root = parent_root or self.get_shared_root()

        # Create splash as Toplevel instead of root
        self.splash = tk.Toplevel(self.parent_root)
        self.splash.overrideredirect(True)  # Remove window decorations
        self.splash.configure(bg="#1e1e1e")  # Dark background

        # Keep reference to logo to prevent garbage collection
        self.logo_photo: Optional[ImageTk.PhotoImage] = None

        # Center the splash screen
        width = 300
        height = 450
        screen_width = self.splash.winfo_screenwidth()
        screen_height = self.splash.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.splash.geometry(f"{width}x{height}+{x}+{y}")

        # Add a border
        self.splash.configure(highlightbackground="#3a7ca5", highlightthickness=2)

        # Create content
        self._create_content()

        # Ensure splash is on top
        self.splash.lift()

    def _create_content(self):
        """Create splash screen content."""
        # Main container
        container = tk.Frame(self.splash, bg="#1e1e1e")
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Try to load logo
        try:
            # FIX: Correct path resolution
            if getattr(sys, "frozen", False):
                # Running as compiled executable
                base_path = sys._MEIPASS  # type: ignore
                logo_path = os.path.join(base_path, "resources", "full_logo.png")
            else:
                # Running in development - splash_screen.py is in src/gui/
                current_dir = os.path.dirname(os.path.abspath(__file__))  # gui folder
                src_dir = os.path.dirname(current_dir)  # src folder
                logo_path = os.path.join(src_dir, "resources", "full_logo.png")

            print(f"[SplashScreen] Looking for logo at: {logo_path}")
            print(f"[SplashScreen] Logo exists: {os.path.exists(logo_path)}")

            if os.path.exists(logo_path):
                # Only import PIL if logo exists
                logo_image = Image.open(logo_path)
                logo_image = logo_image.resize((200, 200), Image.Resampling.LANCZOS)

                # Create PhotoImage and keep reference
                self.logo_photo = ImageTk.PhotoImage(logo_image, master=self.splash)

                logo_label = tk.Label(container, image=self.logo_photo, bg="#1e1e1e")
                logo_label.pack(pady=(0, 20))
            else:
                print("[SplashScreen] Logo not found, showing text only")
        except Exception as e:
            # If logo fails, just show text
            print(f"[SplashScreen] Error loading logo: {e}")

        # App name
        title_label = tk.Label(
            container,
            text="GeoVue",
            font=("Arial", 24, "bold"),
            bg="#1e1e1e",
            fg="#e0e0e0",
        )
        title_label.pack()

        # Loading message
        self.status_label = tk.Label(
            container,
            text="Initializing...",
            font=("Arial", 10),
            bg="#1e1e1e",
            fg="#a0a0a0",
        )
        self.status_label.pack(pady=(20, 10))

        # Version label - read from pyproject.toml if possible
        version = "v1.0.0"  # Default
        try:
            # Try to read version from pyproject.toml
            if getattr(sys, "frozen", False):
                # For frozen app, version should be hardcoded
                pass
            else:
                # In development, try to read from pyproject.toml
                project_root = os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
                pyproject_path = os.path.join(project_root, "pyproject.toml")
                if os.path.exists(pyproject_path):
                    with open(pyproject_path, "r") as f:
                        for line in f:
                            if line.startswith("version = "):
                                version = f"v{line.split('=')[1].strip().strip('\"')}"
                                break
        except:
            pass

        version_label = tk.Label(
            container,
            text=f"{version}\nGeorge Symonds 2025",
            font=("Arial", 8),
            bg="#1e1e1e",
            fg="#606060",
        )
        version_label.pack(side=tk.BOTTOM)

    def update_status(self, message):
        """Update status message."""
        if hasattr(self, "status_label") and self.splash.winfo_exists():
            self.status_label.config(text=message)
            self.splash.update()

    def show(self):
        """Show the splash screen."""
        self.splash.update()
        self.splash.deiconify()
        # Ensure it stays on top
        self.splash.lift()

    def close(self):
        """Close the splash screen without touching the shared root."""
        try:
            # drop the photo ref first (optional)
            self.logo_photo = None
            # this will raise TclError if it's already gone
            self.splash.destroy()
        except (AttributeError, tk.TclError):
            # either self.splash wasn't set, or the Tcl window is already dead
            pass
