# gui/splash_screen.py
import tkinter as tk
from tkinter import ttk
import os
import sys

class SplashScreen:
    """Lightweight splash screen that shows immediately on startup."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)  # Remove window decorations
        self.root.configure(bg='#1e1e1e')  # Dark background
        
        # Center the splash screen
        width = 400
        height = 250
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
        # Add a border
        self.root.configure(highlightbackground='#3a7ca5', highlightthickness=2)
        
        # Create content
        self._create_content()
        
    def _create_content(self):
        """Create splash screen content."""
        # Main container
        container = tk.Frame(self.root, bg='#1e1e1e')
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Try to load logo
        try:
            # FIX: Correct path resolution
            if getattr(sys, 'frozen', False):
                # Running as compiled executable
                base_path = sys._MEIPASS
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
                from PIL import Image, ImageTk
                logo_image = Image.open(logo_path)
                logo_image = logo_image.resize((200, 200), Image.Resampling.LANCZOS)
                self.logo_photo = ImageTk.PhotoImage(logo_image)
                
                logo_label = tk.Label(container, image=self.logo_photo, bg='#1e1e1e')
                logo_label.pack(pady=(0, 20))
            else:
                print(f"[SplashScreen] Logo not found, showing text only")
        except Exception as e:
            # If logo fails, just show text
            print(f"[SplashScreen] Error loading logo: {e}")
        
        # App name
        title_label = tk.Label(
            container,
            text="GeoVue",
            font=("Arial", 24, "bold"),
            bg='#1e1e1e',
            fg='#e0e0e0'
        )
        title_label.pack()
        
        # Loading message
        self.status_label = tk.Label(
            container,
            text="Initializing...",
            font=("Arial", 10),
            bg='#1e1e1e',
            fg='#a0a0a0'
        )
        self.status_label.pack(pady=(20, 10))
        
        # Progress bar
        self.progress = ttk.Progressbar(
            container,
            mode='indeterminate',
            length=300
        )
        self.progress.pack(pady=(0, 10))
        self.progress.start(10)  # Start animation
        
        # Version label - read from pyproject.toml if possible
        version = "v2.0.3"  # Default
        try:
            # Try to read version from pyproject.toml
            if getattr(sys, 'frozen', False):
                # For frozen app, version should be hardcoded
                pass
            else:
                # In development, try to read from pyproject.toml
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                pyproject_path = os.path.join(project_root, "pyproject.toml")
                if os.path.exists(pyproject_path):
                    with open(pyproject_path, 'r') as f:
                        for line in f:
                            if line.startswith('version = '):
                                version = f"v{line.split('=')[1].strip().strip('\"')}"
                                break
        except:
            pass
            
        version_label = tk.Label(
            container,
            text=f"{version}\nGeorge Symonds 2025",
            font=("Arial", 8),
            bg='#1e1e1e',
            fg='#606060'
        )
        version_label.pack(side=tk.BOTTOM)
    
    def update_status(self, message):
        """Update status message."""
        if hasattr(self, 'status_label'):
            self.status_label.config(text=message)
            self.root.update()
    
    def show(self):
        """Show the splash screen."""
        self.root.update()
        self.root.deiconify()
        
    def close(self):
        """Close the splash screen."""
        try:
            self.progress.stop()
            self.root.destroy()
        except:
            pass