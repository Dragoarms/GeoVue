# core/repo_updater.py

"""
Simple version checker for the application.
Checks GitHub for newer versions and prompts user to download manually.
"""

import os
import sys
import ssl
import json
import re
import logging
import urllib.request
import tomllib


# Configure logging
logger = logging.getLogger(__name__)


class RepoUpdater:
    """
    Simple version checker that compares local version with GitHub version.
    Prompts user to contact administrator for updates.
    """
    
    def __init__(self, 
                config_manager=None,
                github_repo="https://github.com/Dragoarms/Geological-Chip-Tray-Compartment-Extractor",
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
        else:
            self.owner = None
            self.repo = None
    
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
        """
        Get the latest version from GitHub pyproject.toml.
        
        Returns:
            str: Latest version as a string, or "Unknown" if not found
        """
        try:
            if not self.owner or not self.repo:
                return "Unknown"
                
            # First try to get version from GitHub API releases
            try:
                api_url = f"https://api.github.com/repos/{self.owner}/{self.repo}/releases/latest"
                
                request = urllib.request.Request(api_url)
                request.add_header("Accept", "application/vnd.github.v3+json")
                
                if self.token:
                    request.add_header("Authorization", f"token {self.token}")
                
                context = ssl._create_unverified_context()
                
                with urllib.request.urlopen(request, context=context) as response:
                    release_data = json.loads(response.read().decode('utf-8'))
                    tag_name = release_data.get('tag_name', '')
                    # Remove 'v' prefix if present
                    if tag_name.startswith('v'):
                        return tag_name[1:]
                    elif tag_name:
                        return tag_name
            except:
                # If releases API fails, continue to check pyproject.toml
                pass
                
            # Get version from pyproject.toml on GitHub
            raw_url = f"https://raw.githubusercontent.com/{self.owner}/{self.repo}/{self.branch}/pyproject.toml"
            
            context = ssl._create_unverified_context()
            
            # Create a request object
            request = urllib.request.Request(raw_url)
            
            # Add authorization header if token is provided
            if self.token:
                request.add_header("Authorization", f"token {self.token}")
            
            # Make the request
            response = urllib.request.urlopen(request, context=context)
            content = response.read().decode('utf-8')
            
            # Parse version from pyproject.toml
            match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                return match.group(1)
            
            return "Unknown"
        except Exception as e:
            self.logger.error(f"Error getting GitHub version: {str(e)}")
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

        if local_version == "Unknown" or github_version == "Unknown":
            result['error'] = "Could not determine versions"
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
    
    def check_and_update(self, parent_window=None) -> dict:
        """
        Check for updates and notify user if available.
        
        Args:
            parent_window: Optional parent window for dialogs
            
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
                    self.dialog_helper.t("update_check_failed"),
                    self.dialog_helper.t("could_not_check_updates") + f": {version_check['error']}",
                    message_type="error"
                )
            
            return {'success': False, 'message': version_check['error']}
        
        if not version_check['update_available']:
            self.logger.info("No updates available.")
            
            # Show info dialog if dialog helper is available
            if self.dialog_helper and parent_window:
                self.dialog_helper.show_message(
                    parent_window,
                    self.dialog_helper.t("no_updates_available"),
                    self.dialog_helper.t("latest_version") + f": {version_check['local_version']}",
                    message_type="info"
                )
            
            return {'success': True, 'message': "No updates available", 'updated': False}
        
        # Update is available - notify user to contact administrator
        if self.dialog_helper and parent_window:
            message = (
                self.dialog_helper.t("new_version_available") + f": {version_check['github_version']}\n" +
                self.dialog_helper.t("current_version") + f": {version_check['local_version']}\n\n" +
                self.dialog_helper.t("contact_for_update") + "\n" +
                "George Symonds [george.symonds@fortescue.com]"
            )
            
            # Show message
            self.dialog_helper.show_message(
                parent_window,
                self.dialog_helper.t("update_available"),
                message,
                message_type="info"
            )
            
            return {
                'success': True, 
                'message': "Update notification shown", 
                'updated': False,
                'update_available': True
            }
        
        return {'success': True, 'message': "Update check completed", 'updated': False}