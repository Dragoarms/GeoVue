# launcher.py
"""
Launcher script that shows splash screen immediately.
This should be the entry point for PyInstaller.
"""

import sys
import os

def setup_paths():
    """Setup the Python path correctly for both frozen and development environments."""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        base_path = sys._MEIPASS
        # For frozen apps, the src content is extracted to _MEIPASS root
        return base_path
    else:
        # Running in development
        base_path = os.path.dirname(os.path.abspath(__file__))
        # Add src to path
        src_path = os.path.join(base_path, 'src')
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        return base_path

if __name__ == "__main__":
    # Setup paths first
    base_path = setup_paths()
    
    # Import and show splash screen
    try:
        from gui.splash_screen import SplashScreen
        splash = SplashScreen()
        splash.show()
        has_splash = True
    except Exception as e:
        print(f"Warning: Could not create splash screen: {e}")
        splash = None
        has_splash = False
    
    # Import main module
    try:
        if has_splash:
            splash.update_status("Loading modules...")
        
        # Import main after showing splash
        import main
        
        if has_splash:
            splash.update_status("Starting GeoVue...")
            # Store splash reference for main to use during init
            main._splash = splash
        
        # Run main application
        main.main()
        
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
        
        # Close splash if it exists
        if has_splash and splash:
            try:
                splash.close()
            except:
                pass
        
        # Show error dialog
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Error", f"Failed to start GeoVue:\n\n{str(e)}")
            root.destroy()
        except:
            pass
        
        sys.exit(1)