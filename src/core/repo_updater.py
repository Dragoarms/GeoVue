# core/repo_updater.py

"""
Simple version checker for the application.
Checks GitHub for newer versions and prompts user to download manually.
"""

import os
import sys
import ssl
import re
import logging
import urllib.request
import tomllib
import PIL.Image
import PIL.ImageTk
import tkinter as tk
import json
import base64


# Configure logging
logger = logging.getLogger(__name__)


class RepoUpdater:
    """
    Simple version checker that compares local version with GitHub version.
    Prompts user to contact administrator for updates.
    """
    
    def __init__(self, 
                config_manager=None,
                github_repo="https://github.com/Dragoarms/GeoVue/",
                branch="main",
                token=None,
                dialog_helper=None):
        """
        Initialize the repository updater.
        
        Args:
            config_manager: Configuration manager instance
            github_repo: URL to the GitHub repository
            branch: Branch to check version from
            token: GitHub personal access token for private repositories
            dialog_helper: DialogHelper instance for user interactions
        """
        self.github_repo = github_repo.rstrip(".git")  # Remove .git if present
        self.branch = branch
        self.token = token
        self.logger = logging.getLogger(__name__)
        self.dialog_helper = dialog_helper
        self.config_manager = config_manager
        
        # Extract owner and repo name from the URL
        match = re.search(r'github\.com/([^/]+)/([^/.]+)', github_repo)
        if match:
            self.owner = match.group(1)
            self.repo = match.group(2)
            self.logger.info(f"Extracted GitHub info - Owner: George Symonds, Repo: {self.repo}")
        else:
            self.owner = None
            self.repo = None
            self.logger.error(f"Failed to extract owner/repo from URL: {github_repo}")
    
    def get_local_version(self) -> str:
        """
        Get the local version from pyproject.toml.
        
        Returns:
            str: Local version as a string
        """
        try:
            # First try to get version from main module if already loaded
            import __main__
            if hasattr(__main__, '__version__'):
                self.logger.info("local version:" + __main__.__version__)
                return __main__.__version__
        except Exception:
            pass
        
        # Otherwise read from pyproject.toml
        try:
            # Try multiple possible locations for pyproject.toml
            possible_paths = [
                os.path.join(os.path.dirname(__file__), "..", "pyproject.toml"),
                os.path.join(os.path.dirname(__file__), "..", "..", "pyproject.toml"),
                os.path.join(os.path.dirname(sys.executable), "pyproject.toml"),
                "pyproject.toml"
            ]
            
            for pyproject_path in possible_paths:
                if os.path.exists(pyproject_path):
                    if tomllib:
                        with open(pyproject_path, "rb") as f:
                            pyproject_data = tomllib.load(f)
                        return pyproject_data["project"]["version"]
                    else:
                        # Fallback for when tomllib is not available
                        with open(pyproject_path, "r") as f:
                            content = f.read()
                            match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
                            if match:
                                return match.group(1)
        except Exception as e:
            self.logger.warning(f"Error reading version from pyproject.toml: {e}")
        
        return "1.0.0"  # Default fallback version
    

    def get_github_version(self) -> str:
        try:
            api_url = f"https://api.github.com/repos/{self.owner}/{self.repo}/contents/pyproject.toml"
            
            request = urllib.request.Request(api_url)
            request.add_header("Accept", "application/vnd.github.v3+json")
            request.add_header("User-Agent", "GeoVue-Version-Checker")
            
            context = ssl._create_unverified_context()
            
            with urllib.request.urlopen(request, context=context) as response:
                data = json.loads(response.read().decode('utf-8'))

                content = base64.b64decode(data['content']).decode('utf-8')
                
                # Now search for version in the decoded content
                match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
                if match:
                    version = match.group(1)
                    self.logger.info(f"Found GitHub version: {version}")
                    return version
                
        except Exception as e:
            self.logger.error(f"Error: {e}")
            return "Unknown"



    def compare_versions(self) -> dict:
        """
        Compare local and GitHub versions.

        Returns:
            Dictionary with comparison results
        """
        local_version = self.get_local_version()
        github_version = self.get_github_version()

        result = {
            'local_version': local_version,
            'github_version': github_version,
            'update_available': False,
            'error': None
        }

        if github_version == "Unknown":
            result['error'] = "Could not connect to GitHub to check for updates"
            return result
            
        if local_version == "Unknown":
            result['error'] = "Could not determine local version"
            return result

        try:
            # Convert to tuples of integers for comparison
            local_parts = tuple(map(int, local_version.split('.')))
            github_parts = tuple(map(int, github_version.split('.')))

            # Pad with zeros if versions have different number of parts
            max_length = max(len(local_parts), len(github_parts))
            local_parts = local_parts + (0,) * (max_length - len(local_parts))
            github_parts = github_parts + (0,) * (max_length - len(github_parts))

            result['update_available'] = github_parts > local_parts

            return result
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Error comparing versions: {str(e)}")
            return result

    def is_version_too_old(self, local_version: str, github_version: str, max_major_behind: int = 0, max_minor_behind: int = 1) -> bool:
        """
        Check if local version is too far behind the GitHub version.
        
        Args:
            local_version: Local version string (e.g., "1.8.0")
            github_version: GitHub version string (e.g., "2.0.0")
            max_major_behind: Maximum major versions allowed to be behind (default: 1)
            max_minor_behind: Maximum minor versions allowed to be behind if major is same (default: 3)
        
        Returns:
            bool: True if version is too old and should be blocked
        """
        try:
            # Parse versions
            local_parts = list(map(int, local_version.split('.')))
            github_parts = list(map(int, github_version.split('.')))
            
            # Ensure both have 3 parts (major.minor.patch)
            while len(local_parts) < 3:
                local_parts.append(0)
            while len(github_parts) < 3:
                github_parts.append(0)
            
            local_major, local_minor, local_patch = local_parts[:3]
            github_major, github_minor, github_patch = github_parts[:3]
            
            # Check major version difference
            major_diff = github_major - local_major
            
            if major_diff > max_major_behind:
                # Too many major versions behind
                self.logger.warning(f"Version too old: {major_diff} major versions behind (max allowed: {max_major_behind})")
                return True
            
            if major_diff == 0:
                # Same major version, check minor difference
                minor_diff = github_minor - local_minor
                
                if minor_diff > max_minor_behind:
                    # Too many minor versions behind
                    self.logger.warning(f"Version too old: {minor_diff} minor versions behind (max allowed: {max_minor_behind})")
                    return True
            
            # Version is acceptable
            return False
            
        except Exception as e:
            self.logger.error(f"Error comparing versions: {e}")
            # On error, don't block the user
            return False

    def check_and_update(self, parent_window=None, block_if_too_old: bool = True) -> dict:
        """
        Check for updates and notify user if available.
        Optionally close app after the notification dialog if version is too old.
        
        Args:
            parent_window: Optional parent window for dialogs
            block_if_too_old: Whether to close the app if version is too old
            
        Returns:
            Dictionary with update result
        """
        # Check if update is available
        version_check = self.compare_versions()
        
        if version_check['error']:
            self.logger.warning(f"Error checking for updates: {version_check['error']}")
            
            # Show error dialog if dialog helper is available
            if self.dialog_helper and parent_window:
                self.dialog_helper.show_message(
                    parent_window,
                    self.dialog_helper.t("Error"),
                    self.dialog_helper.t("Could not check for updates") + f": {version_check['error']}",
                    message_type="error"
                )
            
            return {'success': False, 'message': version_check['error'], 'blocked': False}
        
        # Check if version is too old
        if block_if_too_old and version_check['github_version'] != "Unknown":
            is_too_old = self.is_version_too_old(
                version_check['local_version'],
                version_check['github_version']
            )
            
            if is_too_old:
                self.logger.error(f"Version {version_check['local_version']} is too old compared to {version_check['github_version']}")
                
                if self.dialog_helper and parent_window is None:
                    self.show_version_blocked_dialog(parent_window, version_check)
                    
                return {
                    'success': False,
                    'message': "Version too old - execution blocked",
                    'blocked': True,
                    'update_available': True
                }
        
        if not version_check['update_available']:
            self.logger.info("No updates available.")
            
            # Show info dialog if dialog helper is available
            if self.dialog_helper and parent_window:
                self.dialog_helper.show_message(
                    parent_window,
                    self.dialog_helper.t("No Updates"),
                    self.dialog_helper.t("You are running the latest version.") + f"\n\nVersion: {version_check['local_version']}",
                    message_type="info"
                )
            
            return {'success': True, 'message': "No updates available", 'updated': False, 'blocked': False}
        
        # Update is available but not mandatory - show update dialog
        if self.dialog_helper: # and parent_window:
            self._show_update_dialog(parent_window, version_check)
            
            return {
                'success': True, 
                'message': "Update notification shown", 
                'updated': False,
                'update_available': True,
                'blocked': False
            }
        
        return {'success': True, 'message': "Update check completed", 'updated': False, 'blocked': False}

    def _add_logo_header(self, parent_frame, theme_colors):
        """Add logo and GeoVue title header to dialogs."""
        # Create container for logo and title
        header_container = tk.Frame(
            parent_frame,
            bg=theme_colors["background"] if theme_colors else "white"
        )
        header_container.pack(pady=(0, 20))
        
        try:
            import PIL.Image
            import PIL.ImageTk
            
            # Get logo path - use full_logo.png like in main_gui
            logo_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                "resources", 
                "full_logo.png"
            )
            
            if os.path.exists(logo_path):
                # Load and resize logo
                logo_image = PIL.Image.open(logo_path)
                
                # Resize to 60 pixels height (smaller than main GUI for dialogs)
                logo_height = 60
                aspect_ratio = logo_image.width / logo_image.height
                logo_width = int(logo_height * aspect_ratio)
                logo_image = logo_image.resize(
                    (logo_width, logo_height), 
                    PIL.Image.Resampling.LANCZOS
                )
                
                # Convert to PhotoImage
                logo_photo = PIL.ImageTk.PhotoImage(logo_image)
                
                # Create label for logo
                logo_label = tk.Label(
                    header_container,
                    image=logo_photo,
                    bg=theme_colors["background"] if theme_colors else "white"
                )
                logo_label.image = logo_photo  # Keep a reference
                logo_label.pack(side=tk.LEFT, padx=(0, 15))
                
                # Add GeoVue title next to logo
                title_label = tk.Label(
                    header_container,
                    text="GeoVue",
                    font=("Arial", 20, "bold"),
                    bg=theme_colors["background"] if theme_colors else "white",
                    fg=theme_colors["text"] if theme_colors else "black"
                )
                title_label.pack(side=tk.LEFT)
                
                self.logger.debug("Logo and title added to dialog")
            else:
                # Fallback to just title if logo not found
                self.logger.debug(f"Logo not found at: {logo_path}")
                title_label = tk.Label(
                    header_container,
                    text="GeoVue",
                    font=("Arial", 20, "bold"),
                    bg=theme_colors["background"] if theme_colors else "white",
                    fg=theme_colors["text"] if theme_colors else "black"
                )
                title_label.pack()
                
        except Exception as e:
            self.logger.debug(f"Could not load logo for dialog: {e}")
            # Fallback to just title
            title_label = tk.Label(
                header_container,
                text="GeoVue",
                font=("Arial", 20, "bold"),
                bg=theme_colors["background"] if theme_colors else "white",
                fg=theme_colors["text"] if theme_colors else "black"
            )
            title_label.pack()


    def show_version_dialog(self, parent_window, version_check, *, 
                        is_blocking=False):
        """Show version dialog - either for updates or blocking."""
        import tkinter as tk
        import webbrowser
        import urllib.parse
        import os
        
        # Store dialog result for blocking dialogs
        if is_blocking:
            self._dialog_result = None
        
        # Dialog title
        title = "Version Too Old" if is_blocking else "Update Available"
        
        dialog = self.dialog_helper.create_dialog(
            parent_window,
            self.dialog_helper.t(title),
            modal=True,
            topmost=True
        )
        
        if is_blocking:
            dialog.protocol("WM_DELETE_WINDOW", lambda: None)
        
        theme_colors = None
        if hasattr(self.dialog_helper, 'gui_manager') and self.dialog_helper.gui_manager:
            theme_colors = self.dialog_helper.gui_manager.theme_colors
        
        # Adjust padding based on dialog type
        padding = 40 if is_blocking else 30
        main_frame = tk.Frame(
            dialog, 
            bg=theme_colors["background"] if theme_colors else "white", 
            padx=padding, 
            pady=30 if is_blocking else 20
        )
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        self._add_logo_header(main_frame, theme_colors)
        
        # Warning icon only for blocking
        if is_blocking:
            warning_label = tk.Label(
                main_frame,
                text="⛔",
                font=("Arial", 36),
                bg=theme_colors["background"] if theme_colors else "white",
                fg=theme_colors["accent_red"] if theme_colors else "red"
            )
            warning_label.pack(pady=(0, 15))
        
        # Version message - different for each type
        if is_blocking:
            version_text = (
                self.dialog_helper.t("This version of GeoVue is too old and cannot be used.") + "\n\n" +
                self.dialog_helper.t("Your version") + f": {version_check['local_version']}\n" +
                self.dialog_helper.t("Latest version") + f": {version_check['github_version']}\n\n" +
                self.dialog_helper.t("You must update to continue using GeoVue.")
            )
        else:
            version_text = (
                self.dialog_helper.t("A new version is available!") + "\n\n" +
                self.dialog_helper.t("Current version") + f": {version_check['local_version']}\n" +
                self.dialog_helper.t("Latest version") + f": {version_check['github_version']}"
            )
        
        version_label = tk.Label(
            main_frame,
            text=version_text,
            justify=tk.CENTER,
            bg=theme_colors["background"] if theme_colors else "white",
            fg=theme_colors["text"] if theme_colors else "black",
            font=("Arial", 11)
        )
        version_label.pack(pady=(0, 20))
        
        # Contact info - same for both
        contact_label = tk.Label(
            main_frame,
            text=self.dialog_helper.t("For the update please contact:"),
            justify=tk.CENTER,
            bg=theme_colors["background"] if theme_colors else "white",
            fg=theme_colors["text"] if theme_colors else "black",
            font=("Arial", 10)
        )
        contact_label.pack(pady=(0, 5))
        
        name_label = tk.Label(
            main_frame,
            text="George Symonds",
            justify=tk.CENTER,
            bg=theme_colors["background"] if theme_colors else "white",
            fg=theme_colors["text"] if theme_colors else "black",
            font=("Arial", 11, "bold")
        )
        name_label.pack(pady=(0, 5))
        
        # Clickable email
        email = "George.Symonds@Fortescue.com"
        email_label = tk.Label(
            main_frame,
            text=email,
            justify=tk.CENTER,
            bg=theme_colors["background"] if theme_colors else "white",
            fg=theme_colors["accent_blue"] if theme_colors else "#0066cc",
            font=("Arial", 10, "underline"),
            cursor="hand2"
        )
        email_label.pack(pady=(0, 20))
        
        # Email content - same for both based on your requirement
        def open_email(event):
            subject = "GeoVue Update Request"
            body = (
                f"Hi,\n\n"
                f"I would like to request the update for GeoVue.\n\n"
                f"Current version: {version_check['local_version']}\n"
                f"Latest version: {version_check['github_version']}\n\n"
            )
            
            # URL encode the subject and body
            mailto_url = f"mailto:{email}?subject={urllib.parse.quote(subject)}&body={urllib.parse.quote(body)}"
            webbrowser.open(mailto_url)
        
        email_label.bind("<Button-1>", open_email)
        
        # Add hover effect
        def on_enter(e):
            email_label.config(fg=theme_colors["accent_green"] if theme_colors else "#00aa00")
        
        def on_leave(e):
            email_label.config(fg=theme_colors["accent_blue"] if theme_colors else "#0066cc")
        
        email_label.bind("<Enter>", on_enter)
        email_label.bind("<Leave>", on_leave)
        
        # Button with different behavior
        if is_blocking:
            button_text = "Exit GeoVue"
            button_color = theme_colors["accent_red"] if theme_colors else "red"
            
            def button_command():
                self._dialog_result = "exit"
                dialog.destroy()
                if parent_window:
                    parent_window.quit()
                    parent_window.destroy()
                os._exit(1)
        else:
            button_text = "OK"
            button_color = theme_colors["accent_blue"] if theme_colors else "blue"
            button_command = dialog.destroy
        
        if hasattr(self.dialog_helper, 'gui_manager') and self.dialog_helper.gui_manager:
            button = self.dialog_helper.gui_manager.create_modern_button(
                main_frame,
                text=self.dialog_helper.t(button_text),
                color=button_color,
                command=button_command
            )
        else:
            button = tk.Button(
                main_frame,
                text=self.dialog_helper.t(button_text),
                command=button_command,
                bg=button_color if is_blocking else None,
                fg="white" if is_blocking else None
            )
        button.pack()
        
        # Center dialog with appropriate size
        dialog.update_idletasks()
        self.dialog_helper.center_dialog(
            dialog,
            parent_window,
            max_width=550 if is_blocking else 500,
            max_height=500 if is_blocking else 450
        )
        
        # Show dialog
        dialog.transient(None)
        dialog.deiconify()
        dialog.lift()
        dialog.focus_force()
        dialog.grab_set()
        
        if is_blocking:
            dialog.update()  # Force display
            self.logger.info(f"calling wait window for show_version_blocked_dialog")
        
        dialog.wait_window()
        
        if is_blocking:
            self.logger.debug("After wait")
            if self._dialog_result == "exit":
                os._exit(1)

    # Then create the wrapper methods for compatibility
    def show_version_blocked_dialog(self, parent_window, version_check):
        """Show a blocking dialog when version is too old and exit the application."""
        self.show_version_dialog(parent_window, version_check, is_blocking=True)

    def _show_update_dialog(self, parent_window, version_check):
        """Show a custom update dialog with logo and clickable email."""
        self.show_version_dialog(parent_window, version_check, is_blocking=False)


    # def _show_update_dialog(self, parent_window, version_check):
    #     """Show a custom update dialog with logo and clickable email."""
    #     import tkinter as tk
    #     import webbrowser
    #     import urllib.parse
        
    #     # Create dialog
    #     dialog = self.dialog_helper.create_dialog(
    #         parent_window,
    #         self.dialog_helper.t("Update Available"),
    #         modal=True,
    #         topmost=True
    #     )
        
    #     # Get theme colors
    #     theme_colors = None
    #     if hasattr(self.dialog_helper, 'gui_manager') and self.dialog_helper.gui_manager:
    #         theme_colors = self.dialog_helper.gui_manager.theme_colors
        
    #     # Main container
    #     main_frame = tk.Frame(
    #         dialog, 
    #         bg=theme_colors["background"] if theme_colors else "white",
    #         padx=30,
    #         pady=20
    #     )
    #     main_frame.pack(fill=tk.BOTH, expand=True)
        
    #     # Add logo and title header
    #     self._add_logo_header(main_frame, theme_colors)
        
    #     # Version info - centered
    #     version_text = (
    #         self.dialog_helper.t("A new version is available!") + "\n\n" +
    #         self.dialog_helper.t("Current version") + f": {version_check['local_version']}\n" +
    #         self.dialog_helper.t("Latest version") + f": {version_check['github_version']}"
    #     )
        
    #     version_label = tk.Label(
    #         main_frame,
    #         text=version_text,
    #         justify=tk.CENTER,
    #         bg=theme_colors["background"] if theme_colors else "white",
    #         fg=theme_colors["text"] if theme_colors else "black",
    #         font=("Arial", 11)
    #     )
    #     version_label.pack(pady=(0, 20))
        
    #     # Contact info
    #     contact_label = tk.Label(
    #         main_frame,
    #         text=self.dialog_helper.t("For the update please contact:"),
    #         justify=tk.CENTER,
    #         bg=theme_colors["background"] if theme_colors else "white",
    #         fg=theme_colors["text"] if theme_colors else "black",
    #         font=("Arial", 10)
    #     )
    #     contact_label.pack(pady=(0, 5))
        
    #     # Name
    #     name_label = tk.Label(
    #         main_frame,
    #         text="George Symonds",
    #         justify=tk.CENTER,
    #         bg=theme_colors["background"] if theme_colors else "white",
    #         fg=theme_colors["text"] if theme_colors else "black",
    #         font=("Arial", 11, "bold")
    #     )
    #     name_label.pack(pady=(0, 5))
        
    #     # Clickable email
    #     email = "George.Symonds@Fortescue.com"
    #     email_label = tk.Label(
    #         main_frame,
    #         text=email,
    #         justify=tk.CENTER,
    #         bg=theme_colors["background"] if theme_colors else "white",
    #         fg=theme_colors["accent_blue"] if theme_colors else "#0066cc",
    #         font=("Arial", 10, "underline"),
    #         cursor="hand2"
    #     )
    #     email_label.pack(pady=(0, 20))
        
    #     # Make email clickable with pre-filled subject and body
    #     def open_email(event):
    #         subject = "GeoVue Update Request"
    #         body = (
    #             f"Hi, \n\n"
    #             f"I would like to request the update for GeoVue.\n\n"
    #             f"Current version: {version_check['local_version']}\n"
    #             f"Latest version: {version_check['github_version']}\n\n"
    #         )
            
    #         # URL encode the subject and body
    #         mailto_url = f"mailto:{email}?subject={urllib.parse.quote(subject)}&body={urllib.parse.quote(body)}"
    #         webbrowser.open(mailto_url)
        
    #     email_label.bind("<Button-1>", open_email)
        
    #     # Add hover effect for email
    #     def on_enter(e):
    #         email_label.config(fg=theme_colors["accent_green"] if theme_colors else "#00aa00")
        
    #     def on_leave(e):
    #         email_label.config(fg=theme_colors["accent_blue"] if theme_colors else "#0066cc")
        
    #     email_label.bind("<Enter>", on_enter)
    #     email_label.bind("<Leave>", on_leave)
        
    #     # OK button
    #     if hasattr(self.dialog_helper, 'gui_manager') and self.dialog_helper.gui_manager:
    #         ok_button = self.dialog_helper.gui_manager.create_modern_button(
    #             main_frame,
    #             text=self.dialog_helper.t("OK"),
    #             color=theme_colors["accent_blue"],
    #             command=dialog.destroy
    #         )
    #     else:
    #         ok_button = tk.Button(
    #             main_frame,
    #             text=self.dialog_helper.t("OK"),
    #             command=dialog.destroy
    #         )
    #     ok_button.pack()
        
    #     # Center dialog
    #     dialog.update_idletasks()
    #     self.dialog_helper.center_dialog(
    #         dialog,
    #         parent_window,
    #         max_width=500,
    #         max_height=450
    #     )
    #     # Force focus and make sure it's on top
    #     dialog.transient(None)
    #     dialog.deiconify()
    #     dialog.lift()
    #     dialog.focus_force()
    #     dialog.grab_set()

            
    # def show_version_blocked_dialog(self, parent_window, version_check):
    #     """Show a blocking dialog when version is too old and exit the application."""
    #     import tkinter as tk
    #     import sys
    #     import webbrowser
    #     import urllib.parse
        
    #     # Store whether dialog was closed
    #     self._dialog_result = None
        
    #     # If no parent window, create a temporary root
    #     # temp_root = None
    #     # if parent_window is None:
    #     #     temp_root = tk.Tk()
    #     #     temp_root.withdraw()  # Hide the temp root
    #     #     parent_window = temp_root

    #     # Create dialog
    #     dialog = self.dialog_helper.create_dialog(
    #         parent_window,
    #         self.dialog_helper.t("Version Too Old"),
    #         modal=True,
    #         topmost=True
    #     )
        
        
    #     # Make it unclosable except through our button
    #     dialog.protocol("WM_DELETE_WINDOW", lambda: None)
        
    #     # Get theme colors
    #     theme_colors = None
    #     if hasattr(self.dialog_helper, 'gui_manager') and self.dialog_helper.gui_manager:
    #         theme_colors = self.dialog_helper.gui_manager.theme_colors
        
    #     # Main container
    #     main_frame = tk.Frame(
    #         dialog, 
    #         bg=theme_colors["background"] if theme_colors else "white",
    #         padx=40,
    #         pady=30
    #     )
    #     main_frame.pack(fill=tk.BOTH, expand=True)
        
    #     # Add logo and title header
    #     self._add_logo_header(main_frame, theme_colors)
        
    #     # Warning icon below header
    #     warning_label = tk.Label(
    #         main_frame,
    #         text="⛔",
    #         font=("Arial", 36),
    #         bg=theme_colors["background"] if theme_colors else "white",
    #         fg=theme_colors["accent_red"] if theme_colors else "red"
    #     )
    #     warning_label.pack(pady=(0, 15))
        
    #     # Error message
    #     error_text = (
    #         self.dialog_helper.t("This version of GeoVue is too old and cannot be used.") + "\n\n" +
    #         self.dialog_helper.t("Your version") + f": {version_check['local_version']}\n" +
    #         self.dialog_helper.t("Latest version") + f": {version_check['github_version']}\n\n" +
    #         self.dialog_helper.t("You must update to continue using GeoVue.")
    #     )
        
    #     error_label = tk.Label(
    #         main_frame,
    #         text=error_text,
    #         justify=tk.CENTER,
    #         bg=theme_colors["background"] if theme_colors else "white",
    #         fg=theme_colors["text"] if theme_colors else "black",
    #         font=("Arial", 11)
    #     )
    #     error_label.pack(pady=(0, 20))
        
    #     # Contact info
    #     contact_label = tk.Label(
    #         main_frame,
    #         text=self.dialog_helper.t("Please contact your administrator:"),
    #         justify=tk.CENTER,
    #         bg=theme_colors["background"] if theme_colors else "white",
    #         fg=theme_colors["text"] if theme_colors else "black",
    #         font=("Arial", 10)
    #     )
    #     contact_label.pack(pady=(0, 5))
        
    #     name_label = tk.Label(
    #         main_frame,
    #         text="George Symonds",
    #         justify=tk.CENTER,
    #         bg=theme_colors["background"] if theme_colors else "white",
    #         fg=theme_colors["text"] if theme_colors else "black",
    #         font=("Arial", 11, "bold")
    #     )
    #     name_label.pack(pady=(0, 5))
        
    #     # Clickable email
    #     email = "George.Symonds@Fortescue.com"
    #     email_label = tk.Label(
    #         main_frame,
    #         text=email,
    #         justify=tk.CENTER,
    #         bg=theme_colors["background"] if theme_colors else "white",
    #         fg=theme_colors["accent_blue"] if theme_colors else "#0066cc",
    #         font=("Arial", 10, "underline"),
    #         cursor="hand2"
    #     )
    #     email_label.pack(pady=(0, 20))
        
    #     # Make email clickable with pre-filled subject and body
    #     def open_email(event):
    #         subject = "URGENT: GeoVue Update Required"
    #         body = (
    #             f"Hi,\n\n"
    #             f"I am unable to use GeoVue as my version is too old.\n\n"
    #             f"Current version: {version_check['local_version']}\n"
    #             f"Required version: {version_check['github_version']}\n\n"
    #             f"Please provide the updated version"
    #         )
            
    #         # URL encode the subject and body
    #         mailto_url = f"mailto:{email}?subject={urllib.parse.quote(subject)}&body={urllib.parse.quote(body)}"
    #         webbrowser.open(mailto_url)
        
    #     email_label.bind("<Button-1>", open_email)
        
    #     # Add hover effect
    #     def on_enter(e):
    #         email_label.config(fg=theme_colors["accent_green"] if theme_colors else "#00aa00")
        
    #     def on_leave(e):
    #         email_label.config(fg=theme_colors["accent_blue"] if theme_colors else "#0066cc")
        
    #     email_label.bind("<Enter>", on_enter)
    #     email_label.bind("<Leave>", on_leave)
        
    #     # Exit button with modified behavior
    #     def exit_application():
    #         self._dialog_result = "exit"
    #         dialog.destroy()
            
    #         # Clean up temp root if we created one
    #         # if temp_root:
    #         #     try:
    #         #         temp_root.destroy()
    #         #     except:
    #         #         pass
            
    #         # Destroy parent if exists and not temp
    #         if parent_window: # and not temp_root:
    #             # try:
    #             parent_window.quit()
    #             parent_window.destroy()
    #             # except:
    #             #     pass
            
    #         # Force exit
    #         os._exit(1)
        
    #     if hasattr(self.dialog_helper, 'gui_manager') and self.dialog_helper.gui_manager:
    #         exit_button = self.dialog_helper.gui_manager.create_modern_button(
    #             main_frame,
    #             text=self.dialog_helper.t("Exit GeoVue"),
    #             color=theme_colors["accent_red"],
    #             command=exit_application
    #         )
    #     else:
    #         exit_button = tk.Button(
    #             main_frame,
    #             text=self.dialog_helper.t("Exit GeoVue"),
    #             command=exit_application,
    #             bg="red",
    #             fg="white"
    #         )
    #     exit_button.pack()
        
    #     # Center dialog
    #     dialog.update_idletasks()
    #     self.dialog_helper.center_dialog(
    #         dialog,
    #         parent_window,
    #         max_width=550,
    #         max_height=500
    #     )
        

    #     # IMPORTANT: Make sure dialog is visible before waiting
    #     dialog.transient(None)
    #     dialog.deiconify()
    #     dialog.lift()
    #     dialog.focus_force()
    #     dialog.grab_set()
    #     dialog.update()  # Force the dialog to be displayed
        
    #     self.logger.info(f"calling wait window for show_version_blocked_dialog")
        
    #     print(f"Dialog exists before wait: {dialog.winfo_exists()}")
    #     dialog.wait_window()
    #     self.logger.debug("After wait")

    #     # Now exit after dialog is closed
    #     if self._dialog_result == "exit":
    #         os._exit(1)