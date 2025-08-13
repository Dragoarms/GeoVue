# utils/json_register_manager.py

"""
JSON Register Manager for data storage with Excel Power Query integration.

This module handles all data operations using JSON files as the primary storage,
with Excel files using Power Query for read-only data display. This approach:
- Eliminates Excel file locking issues
- Allows for faster, more reliable data operations
- Enables users to add custom columns in Excel without affecting the source data
- Provides automatic Power Query setup for new Excel files
- Creates user-specific JSON files to prevent multi-user conflicts

The JSON structure is designed to be Power Query friendly, with separate files
for compartments and original images data. Each user gets their own set of JSON
files with a unique suffix based on username and computer name.

Author: George Symonds
Created: 2025
"""

import os
import sys
import json
import time
import shutil
import logging
import traceback
import socket
import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, Any, Callable
from contextlib import contextmanager
import threading
import pandas as pd
from openpyxl import workbook, load_workbook
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.utils import get_column_letter


class JSONRegisterManager:
    """
    Manages register data using JSON files with Excel Power Query integration.

    This class provides thread-safe access to drilling register data stored in JSON,
    with automatic creation of Excel files configured with Power Query connections.
    Each user gets their own set of JSON files to prevent conflicts.
    """

    # File names (base names without user suffix)
    COMPARTMENT_JSON_BASE = "compartment_register"
    ORIGINAL_JSON_BASE = "original_images_register"
    REVIEW_JSON_BASE = "compartment_reviews"
    COMPARTMENT_CORNERS_JSON_BASE = "compartment_corners"

    EXCEL_FILE = "Chip_Tray_Register.xlsx"
    DATA_SUBFOLDER = "Register Data (Do not edit)"

    # Sheet names
    COMPARTMENT_SHEET = "Compartment Register"
    MANUAL_SHEET = "Manual Entries"
    ORIGINAL_SHEET = "Original Images Register"
    REVIEW_SHEET = "Compartment Reviews"
    COMPARTMENT_CORNERS_SHEET = "Compartment Corners"

    @staticmethod
    def get_user_suffix() -> str:
        """
        Get a unique suffix for this user/computer combination.
        Uses username and computer name to create a readable identifier.

        Returns:
            String suffix like "_JohnDoe_DESKTOP123"
        """
        try:
            username = os.getenv("USERNAME", "Unknown").replace(" ", "")
            computername = socket.gethostname().replace(" ", "").replace("-", "")

            # Clean up any problematic characters
            username = "".join(c for c in username if c.isalnum() or c in "_-")
            computername = "".join(c for c in computername if c.isalnum() or c in "_-")

            # Limit length to keep filenames reasonable
            username = username[:20]
            computername = computername[:20]

            return f"_{username}_{computername}"
        except Exception:
            # Fallback to a hash-based identifier if we can't get clean names
            try:
                unique_id = f"{os.getenv('USERNAME', 'user')}_{socket.gethostname()}"
                short_hash = hashlib.md5(unique_id.encode()).hexdigest()[:8]
                return f"_User{short_hash}"
            except Exception:
                return "_DefaultUser"

    @staticmethod
    def check_existing_files_static(
        base_path: str, check_all_users: bool = False
    ) -> Dict[str, bool]:
        """
        Static method to check which files exist without creating an instance.

        Args:
            base_path: Directory path to check
            check_all_users: If True, checks if ANY user files exist (not just current user)

        Returns:
            Dictionary with file existence status
        """
        base = Path(base_path)
        data_path = base / JSONRegisterManager.DATA_SUBFOLDER

        result = {
            "excel": (base / JSONRegisterManager.EXCEL_FILE).exists(),
            "compartment_json": False,
            "original_json": False,
            "review_json": False,
            "compartment_corners_json": False,
            "data_folder": (
                data_path.exists() and any(data_path.iterdir())
                if data_path.exists()
                else False
            ),
        }

        if data_path.exists():
            if check_all_users:
                # Check if ANY user files exist
                result["compartment_json"] = any(
                    f.name.startswith(JSONRegisterManager.COMPARTMENT_JSON_BASE)
                    and f.suffix == ".json"
                    for f in data_path.iterdir()
                )
                result["original_json"] = any(
                    f.name.startswith(JSONRegisterManager.ORIGINAL_JSON_BASE)
                    and f.suffix == ".json"
                    for f in data_path.iterdir()
                )
                result["review_json"] = any(
                    f.name.startswith(JSONRegisterManager.REVIEW_JSON_BASE)
                    and f.suffix == ".json"
                    for f in data_path.iterdir()
                )
                result["compartment_corners_json"] = any(
                    f.name.startswith(JSONRegisterManager.COMPARTMENT_CORNERS_JSON_BASE)
                    and f.suffix == ".json"
                    for f in data_path.iterdir()
                )
            else:
                # Check only current user's files
                suffix = JSONRegisterManager.get_user_suffix()
                result["compartment_json"] = (
                    data_path
                    / f"{JSONRegisterManager.COMPARTMENT_JSON_BASE}{suffix}.json"
                ).exists()
                result["original_json"] = (
                    data_path / f"{JSONRegisterManager.ORIGINAL_JSON_BASE}{suffix}.json"
                ).exists()
                result["review_json"] = (
                    data_path / f"{JSONRegisterManager.REVIEW_JSON_BASE}{suffix}.json"
                ).exists()
                result["compartment_corners_json"] = (
                    data_path
                    / f"{JSONRegisterManager.COMPARTMENT_CORNERS_JSON_BASE}{suffix}.json"
                ).exists()

        return result

    @staticmethod
    def has_existing_data_static(base_path: str) -> bool:
        """Static method to check if any register data exists without creating an instance."""
        existing = JSONRegisterManager.check_existing_files_static(
            base_path, check_all_users=True
        )
        return any(
            [
                existing["compartment_json"],
                existing["original_json"],
                existing["review_json"],
                existing["compartment_corners_json"],
            ]
        )

    @staticmethod
    def get_all_user_files_static(base_path: str, file_type: str) -> List[Path]:
        """
        Get all user files of a specific type.

        Args:
            base_path: Base directory path
            file_type: One of 'compartment', 'original', 'review', 'corners'

        Returns:
            List of Path objects for all matching files
        """
        data_path = Path(base_path) / JSONRegisterManager.DATA_SUBFOLDER
        if not data_path.exists():
            return []

        base_names = {
            "compartment": JSONRegisterManager.COMPARTMENT_JSON_BASE,
            "original": JSONRegisterManager.ORIGINAL_JSON_BASE,
            "review": JSONRegisterManager.REVIEW_JSON_BASE,
            "corners": JSONRegisterManager.COMPARTMENT_CORNERS_JSON_BASE,
        }

        if file_type not in base_names:
            return []

        base_name = base_names[file_type]
        return sorted(
            [
                f
                for f in data_path.iterdir()
                if f.name.startswith(base_name) and f.suffix == ".json"
            ]
        )

    @staticmethod
    def get_data_summary_static(base_path: str) -> Dict[str, Any]:
        """
        Static method to get summary of existing data without creating an instance.
        Now aggregates data across all user files.

        Args:
            base_path: Directory path to check

        Returns:
            Dictionary with data counts and info
        """
        base = Path(base_path)
        data_path = base / JSONRegisterManager.DATA_SUBFOLDER

        summary = {
            "compartment_count": 0,
            "original_count": 0,
            "review_count": 0,
            "corner_count": 0,
            "has_excel": (base / JSONRegisterManager.EXCEL_FILE).exists(),
            "has_json_data": False,
            "user_files": {
                "compartment": [],
                "original": [],
                "review": [],
                "corners": [],
            },
        }

        try:
            # Get all user files and aggregate counts
            for file_type, key in [
                ("compartment", "compartment_count"),
                ("original", "original_count"),
                ("review", "review_count"),
                ("corners", "corner_count"),
            ]:
                files = JSONRegisterManager.get_all_user_files_static(
                    base_path, file_type
                )
                for file_path in files:
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                summary[key] += len(data)
                                summary["has_json_data"] = True
                                # Extract user info from filename
                                user_info = file_path.stem.replace(
                                    JSONRegisterManager.COMPARTMENT_JSON_BASE, ""
                                )
                                user_info = user_info.replace(
                                    JSONRegisterManager.ORIGINAL_JSON_BASE, ""
                                )
                                user_info = user_info.replace(
                                    JSONRegisterManager.REVIEW_JSON_BASE, ""
                                )
                                user_info = user_info.replace(
                                    JSONRegisterManager.COMPARTMENT_CORNERS_JSON_BASE,
                                    "",
                                )
                                if user_info.startswith("_"):
                                    user_info = user_info[1:]
                                summary["user_files"][file_type].append(
                                    {
                                        "user": user_info,
                                        "file": file_path.name,
                                        "count": len(data),
                                    }
                                )
                    except Exception:
                        pass

        except Exception:
            pass  # Silent fail for static check

        return summary

    def __init__(self, base_path: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the JSON Register Manager.

        Args:
            base_path: Directory path for register files
            logger: Optional logger instance
        """
        self.base_path = Path(base_path)
        self.logger = logger or logging.getLogger(__name__)

        # Get user suffix for this instance
        self.user_suffix = self.get_user_suffix()
        self.logger.info(
            f"JSONRegisterManager initialized with user suffix: {self.user_suffix}"
        )

        # Excel file goes in the base path
        self.excel_path = self.base_path / self.EXCEL_FILE

        # Data files go in subfolder
        self.data_path = self.base_path / self.DATA_SUBFOLDER

        # File paths for JSON data - now with user suffix
        self.compartment_json_path = (
            self.data_path / f"{self.COMPARTMENT_JSON_BASE}{self.user_suffix}.json"
        )
        self.original_json_path = (
            self.data_path / f"{self.ORIGINAL_JSON_BASE}{self.user_suffix}.json"
        )
        self.review_json_path = (
            self.data_path / f"{self.REVIEW_JSON_BASE}{self.user_suffix}.json"
        )
        self.compartment_corners_json_path = (
            self.data_path
            / f"{self.COMPARTMENT_CORNERS_JSON_BASE}{self.user_suffix}.json"
        )

        # Lock files in data folder - also with user suffix
        self.compartment_lock = self.compartment_json_path.with_suffix(".json.lock")
        self.original_lock = self.original_json_path.with_suffix(".json.lock")
        self.review_lock = self.review_json_path.with_suffix(".json.lock")
        self.corners_lock = self.compartment_corners_json_path.with_suffix(".json.lock")

        # Use RLock instead of Lock to prevent deadlocks from nested calls
        self._thread_lock = threading.RLock()

        # Track lock acquisition order to prevent deadlocks
        self._lock_order = ["compartment", "original", "review", "corners"]

        # Lock monitoring for debugging
        self._lock_stats = {
            "acquisitions": 0,
            "failures": 0,
            "contentions": 0,
            "max_wait_time": 0.0,
        }

        # Initialize files if needed
        self._initialize_files()

    def get_all_user_files(self, file_type: str) -> List[Path]:
        """
        Get all user files of a specific type for this instance.

        Args:
            file_type: One of 'compartment', 'original', 'review', 'corners'

        Returns:
            List of Path objects for all matching files
        """
        return self.get_all_user_files_static(str(self.base_path), file_type)

    def check_existing_files(self) -> Dict[str, bool]:
        """
        Check which files already exist for the current user.

        Returns:
            Dictionary with file existence status
        """
        return {
            "excel": self.excel_path.exists(),
            "compartment_json": self.compartment_json_path.exists(),
            "original_json": self.original_json_path.exists(),
            "review_json": self.review_json_path.exists(),
            "compartment_corners_json": self.compartment_corners_json_path.exists(),
            "data_folder": self.data_path.exists() and any(self.data_path.iterdir()),
        }

    def has_existing_data(self) -> bool:
        """Check if any register data already exists (for any user)."""
        return self.has_existing_data_static(str(self.base_path))

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of existing data (aggregated across all users).

        Returns:
            Dictionary with data counts and info
        """
        summary = self.get_data_summary_static(str(self.base_path))
        summary["current_user"] = (
            self.user_suffix[1:]
            if self.user_suffix.startswith("_")
            else self.user_suffix
        )
        summary["lock_stats"] = self._lock_stats.copy()
        return summary

    def _initialize_files(self) -> None:
        """Initialize JSON files and Excel with Power Query if they don't exist."""
        # Create directory
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Ensured base path exists: {self.base_path}")

        self.data_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Ensured data path exists: {self.data_path}")

        # Initialize JSON files only if they don't exist
        if not self.compartment_json_path.exists():
            self._write_json_file(self.compartment_json_path, self.compartment_lock, [])
            self.logger.info(f"Created compartment JSON: {self.compartment_json_path}")
        else:
            self.logger.info(
                f"Compartment JSON already exists: {self.compartment_json_path}"
            )

        if not self.original_json_path.exists():
            self._write_json_file(self.original_json_path, self.original_lock, [])
            self.logger.info(f"Created original images JSON: {self.original_json_path}")
        else:
            self.logger.info(
                f"Original images JSON already exists: {self.original_json_path}"
            )

        if not self.review_json_path.exists():
            self._write_json_file(self.review_json_path, self.review_lock, [])
            self.logger.info(f"Created review JSON: {self.review_json_path}")
        else:
            self.logger.info(f"Review JSON already exists: {self.review_json_path}")

        # Initialize compartment corners JSON file only if it doesn't exist
        if not self.compartment_corners_json_path.exists():
            self._write_json_file(
                self.compartment_corners_json_path, self.corners_lock, []
            )
            self.logger.info(
                f"Created compartment corners JSON: {self.compartment_corners_json_path}"
            )
        else:
            self.logger.info(
                f"Compartment corners JSON already exists: {self.compartment_corners_json_path}"
            )

        # Use template instead of creating from scratch
        # Create Excel from template only if it doesn't exist
        if not self.excel_path.exists():
            self.logger.info(
                f"Excel file does not exist, creating from template: {self.excel_path}"
            )
            self._create_excel_from_template()
        else:
            self.logger.info(f"Excel file already exists: {self.excel_path}")

    def _create_excel_from_template(self) -> None:
        """Properly create Excel file from a .xltx template."""
        try:
            # Try to get template path from resources package first
            try:
                from resources import get_excel_template_path

                template_path = str(get_excel_template_path())
                self.logger.info(f"Got template path from resources: {template_path}")
            except (ImportError, FileNotFoundError):
                # Fallback to manual path construction
                if getattr(sys, "frozen", False):
                    base_path = sys._MEIPASS
                else:
                    base_path = os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))
                    )

                template_path = os.path.join(
                    base_path,
                    "resources",
                    "Register Template File",
                    "Chip_Tray_Register.xltx",
                )
                self.logger.info(f"Using fallback template path: {template_path}")

            if not os.path.exists(template_path):
                raise FileNotFoundError(f"Template file not found at: {template_path}")

            self.logger.info(f"Opening template file: {template_path}")

            # Use win32com to properly handle Excel template
            try:
                import win32com.client as win32

                # Create Excel instance
                excel = win32.gencache.EnsureDispatch("Excel.Application")
                excel.Visible = False
                excel.DisplayAlerts = False

                try:
                    # Open the template file
                    workbook = excel.Workbooks.Open(template_path)

                    # Set the author property
                    workbook.BuiltinDocumentProperties("Author").Value = (
                        "George Symonds"
                    )

                    # Save as regular Excel file (.xlsx)
                    workbook.SaveAs(
                        str(self.excel_path),
                        FileFormat=51,  # xlOpenXMLWorkbook = 51 (.xlsx format)
                    )

                    # Close the workbook
                    workbook.Close(SaveChanges=False)

                    self.logger.info(
                        f"Excel file created successfully from template using win32com"
                    )

                finally:
                    # Ensure Excel is closed
                    excel.Quit()

            except ImportError:
                self.logger.error(
                    "win32com.client not available. Please install pywin32."
                )
                raise
            except Exception as com_error:
                self.logger.error(
                    f"COM error creating Excel from template: {com_error}"
                )
                raise

            if self.excel_path.exists():
                self.logger.info(
                    f"Excel file created successfully at: {self.excel_path}"
                )
            else:
                raise FileNotFoundError(
                    f"Excel file was not created at: {self.excel_path}"
                )

        except Exception as e:
            self.logger.error(f"Error creating Excel from template: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def _try_acquire_lock(self, lock_path: Path) -> bool:
        """Try to acquire a lock once."""
        try:
            if lock_path.exists():
                # Check if lock is stale (older than 300 seconds for sync operations)
                lock_age = time.time() - lock_path.stat().st_mtime
                stale_timeout = 300  # 5 minutes for long operations
                if lock_age > stale_timeout:
                    self.logger.warning(
                        f"Removing stale lock (age: {lock_age:.1f}s): {lock_path}"
                    )
                    lock_path.unlink()
                else:
                    # Log why we can't acquire
                    self.logger.debug(
                        f"Lock exists and is fresh (age: {lock_age:.1f}s): {lock_path}"
                    )
                    return False

            # Create lock with user info
            lock_data = {
                "pid": os.getpid(),
                "user": self.user_suffix,
                "timestamp": time.time(),
            }
            lock_path.write_text(json.dumps(lock_data))
            return True

        except Exception:
            return False

    def _acquire_file_lock_with_backoff(
        self, lock_path: Path, timeout: int = 30
    ) -> bool:
        """Acquire file lock with exponential backoff."""
        start_time = time.time()
        backoff = 0.1  # Start with 100ms
        wait_time = 0.0

        while time.time() - start_time < timeout:
            if self._try_acquire_lock(lock_path):
                # Update lock stats
                self._lock_stats["acquisitions"] += 1
                if wait_time > 0:
                    self._lock_stats["contentions"] += 1
                    self._lock_stats["max_wait_time"] = max(
                        self._lock_stats["max_wait_time"], wait_time
                    )
                return True

            time.sleep(min(backoff, 2.0))  # Cap at 2 seconds
            wait_time = time.time() - start_time
            backoff *= 1.5  # Exponential backoff

        self._lock_stats["failures"] += 1
        return False

    def _acquire_file_lock(self, lock_path: Path, timeout: int = 30) -> bool:
        """Acquire file lock with timeout and backoff."""
        return self._acquire_file_lock_with_backoff(lock_path, timeout)

    def _release_file_lock(self, lock_path: Path) -> None:
        """Release file lock."""
        try:
            if lock_path.exists():
                lock_path.unlink()
        except Exception as e:
            self.logger.warning(f"Error releasing lock {lock_path}: {e}")

    def _refresh_file_lock(self, lock_path: Path) -> bool:
        """Refresh a lock by updating its timestamp."""
        try:
            if lock_path.exists():
                # Update the lock file with new timestamp
                lock_data = {
                    "pid": os.getpid(),
                    "user": self.user_suffix,
                    "timestamp": time.time(),
                    "refreshed": True,
                }
                lock_path.write_text(json.dumps(lock_data))
                self.logger.debug(f"Refreshed lock: {lock_path}")
                return True
            else:
                self.logger.warning(f"Lock to refresh doesn't exist: {lock_path}")
                return False
        except Exception as e:
            self.logger.error(f"Error refreshing lock {lock_path}: {e}")
            return False

    def _acquire_file_locks_ordered(
        self, required_locks: List[str], timeout: int = 30
    ) -> Dict[str, bool]:
        """
        Acquire multiple file locks in a consistent order to prevent deadlocks.

        Args:
            required_locks: List of lock names needed ('compartment', 'original', 'review', 'corners')
            timeout: Timeout in seconds

        Returns:
            Dictionary mapping lock names to success status
        """
        lock_map = {
            "compartment": self.compartment_lock,
            "original": self.original_lock,
            "review": self.review_lock,
            "corners": self.corners_lock,
        }

        acquired = {}

        # Always acquire locks in the same order
        for lock_name in self._lock_order:
            if lock_name in required_locks:
                lock_path = lock_map[lock_name]
                if self._acquire_file_lock(lock_path, timeout):
                    acquired[lock_name] = True
                else:
                    # Failed to acquire - release all previously acquired locks
                    for prev_lock in acquired:
                        self._release_file_lock(lock_map[prev_lock])
                    return {name: False for name in required_locks}

        return acquired

    def _release_file_locks_ordered(self, locks_to_release: List[str]) -> None:
        """Release multiple file locks in reverse order of acquisition."""
        lock_map = {
            "compartment": self.compartment_lock,
            "original": self.original_lock,
            "review": self.review_lock,
            "corners": self.corners_lock,
        }

        # Release in reverse order
        for lock_name in reversed(self._lock_order):
            if lock_name in locks_to_release:
                self._release_file_lock(lock_map[lock_name])

    @contextmanager
    def file_locks(self, *lock_names):
        """Context manager for acquiring multiple file locks safely."""
        acquired = self._acquire_file_locks_ordered(list(lock_names))
        try:
            if all(acquired.values()):
                yield
            else:
                raise RuntimeError(f"Failed to acquire necessary locks: {lock_names}")
        finally:
            self._release_file_locks_ordered(list(lock_names))

    def _read_json_file(self, file_path: Path, lock_path: Path) -> List[Dict]:
        """Read JSON file with locking and error handling."""
        if not self._acquire_file_lock(lock_path):
            raise RuntimeError(f"Could not acquire lock for {file_path}")

        try:
            if not file_path.exists():
                # File doesn't exist - return empty list (normal for first run)
                self.logger.info(f"JSON file does not exist yet: {file_path}")
                return []

            # Check if file is empty
            if file_path.stat().st_size == 0:
                self.logger.warning(f"JSON file is empty: {file_path}")
                return []

            # Try to read and parse JSON
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Validate that we got a list
                if not isinstance(data, list):
                    self.logger.error(f"JSON file does not contain a list: {file_path}")
                    # Try to recover if it's a dict by wrapping in list
                    if isinstance(data, dict):
                        self.logger.warning("Converting single dict to list")
                        return [data]
                    else:
                        self.logger.error("Invalid data format, returning empty list")
                        return []

                # Validate that list contains dictionaries
                if data and not all(isinstance(item, dict) for item in data):
                    self.logger.error(
                        f"JSON file contains non-dictionary items: {file_path}"
                    )
                    # Filter out non-dict items
                    valid_data = [item for item in data if isinstance(item, dict)]
                    self.logger.warning(
                        f"Filtered out {len(data) - len(valid_data)} invalid items"
                    )
                    return valid_data

                return data

            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decode error in {file_path}: {e}")

                # Try to read backup if available
                backup_path = file_path.with_suffix(".json.backup")
                if backup_path.exists():
                    self.logger.info("Attempting to read from backup file")
                    try:
                        with open(backup_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            self.logger.info("Successfully recovered data from backup")
                            # Restore the backup over the corrupted file
                            shutil.copy2(backup_path, file_path)
                            return data
                    except Exception as backup_error:
                        self.logger.error(f"Backup file also corrupted: {backup_error}")

                # If we can't recover, return empty list
                self.logger.error("Unable to recover data, returning empty list")
                return []

            except UnicodeDecodeError as e:
                self.logger.error(f"Unicode decode error in {file_path}: {e}")
                # Try different encodings
                for encoding in ["utf-8-sig", "latin-1", "cp1252"]:
                    try:
                        with open(file_path, "r", encoding=encoding) as f:
                            data = json.load(f)
                        self.logger.info(
                            f"Successfully read file with {encoding} encoding"
                        )
                        return data if isinstance(data, list) else []
                    except:
                        continue
                self.logger.error("Failed to read file with any encoding")
                return []

            except Exception as e:
                self.logger.error(f"Unexpected error reading {file_path}: {e}")
                self.logger.error(traceback.format_exc())
                return []

        finally:
            self._release_file_lock(lock_path)

    def _write_json_file(
        self, file_path: Path, lock_path: Path, data: List[Dict]
    ) -> None:
        """Write JSON file with locking, backup, and validation."""
        if not self._acquire_file_lock(lock_path):
            raise RuntimeError(f"Could not acquire lock for {file_path}")

        try:
            # Validate input data
            if not isinstance(data, list):
                raise ValueError(f"Data must be a list, got {type(data)}")

            # Create backup if file exists and is valid
            if file_path.exists() and file_path.stat().st_size > 0:
                backup_path = file_path.with_suffix(".json.backup")
                try:
                    # Verify the existing file is valid JSON before backing up
                    with open(file_path, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                    # Only backup if existing file is valid
                    shutil.copy2(file_path, backup_path)
                    self.logger.debug(f"Created backup: {backup_path}")
                except Exception as e:
                    self.logger.warning(
                        f"Existing file appears corrupted, skipping backup: {e}"
                    )

            # If data is empty, add an example record with all columns
            if not data:
                if "compartment_register" in str(file_path):
                    # Example compartment record with all columns
                    data = [
                        {
                            "HoleID": "INITIALISING",
                            "From": 0,
                            "To": 1,
                            "Photo_Status": "Example",
                            "Processed_Date": datetime.now().isoformat(),
                            "Processed_By": "System",
                            "Comments": "This is an example record - please delete after adding real data",
                            "Average_Hex_Color": "#000000",
                            "Image_Width_Cm": 2.0,
                        }
                    ]
                elif "original_images_register" in str(file_path):
                    # Example original image record WITHOUT nested compartments
                    data = [
                        {
                            "HoleID": "INITIALISING",
                            "Depth_From": 0,
                            "Depth_To": 20,
                            "Original_Filename": "EXAMPLE_0-20_Original.jpg",
                            "File_Count": 1,
                            "All_Filenames": "EXAMPLE_0-20_Original.jpg",
                            "Approved_Upload_Date": datetime.now().isoformat(),
                            "Approved_Upload_Status": "Example",
                            "Rejected_Upload_Date": None,
                            "Rejected_Upload_Status": None,
                            "Uploaded_By": "System",
                            "Comments": "This is an example record - please delete after adding real data",
                            "Scale_PxPerCm": 50.0,
                            "Scale_Confidence": 0.95,
                        }
                    ]
                elif "compartment_reviews" in str(file_path):
                    # Example review record with all columns including toggles
                    data = [
                        {
                            "HoleID": "INITIALISING",
                            "From": 0,
                            "To": 1,
                            "Reviewed_By": "System",
                            "Review_Number": 1,
                            "Review_Date": datetime.now().isoformat(),
                            "Initial_Review_Date": datetime.now().isoformat(),
                            "Comments": "This is an example record - please delete after adding real data",
                            "Bad Image": False,
                            "BIFf": False,
                            "+ QZ": False,
                            "+ CHH/M": False,
                        }
                    ]

                elif "compartment_corners" in str(file_path):
                    # Example compartment corner record
                    data = [
                        {
                            "HoleID": "INITIALISING",
                            "Depth_From": 0,
                            "Depth_To": 20,
                            "Original_Filename": "EXAMPLE_0-20_Original.jpg",
                            "Compartment_Number": 1,
                            "Top_Left_X": 0,
                            "Top_Left_Y": 0,
                            "Top_Right_X": 100,
                            "Top_Right_Y": 0,
                            "Bottom_Right_X": 100,
                            "Bottom_Right_Y": 100,
                            "Bottom_Left_X": 0,
                            "Bottom_Left_Y": 100,
                            "Scale_PxPerCm": 50.0,
                            "Scale_Confidence": 0.95,
                        }
                    ]

            # Create parent directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to temporary file first
            temp_path = file_path.with_suffix(".json.tmp")
            try:
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(
                        data,
                        f,
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                        separators=(",", ": "),
                    )

                # Verify the temporary file is valid
                with open(temp_path, "r", encoding="utf-8") as f:
                    verified_data = json.load(f)

                # If verification passes, move temp to final location
                temp_path.replace(file_path)
                self.logger.debug(
                    f"Successfully wrote {len(data)} records to {file_path}"
                )

            except Exception as e:
                # Clean up temp file on error
                if temp_path.exists():
                    temp_path.unlink()
                raise e

        except Exception as e:
            self.logger.error(f"Error writing to {file_path}: {e}")
            raise

        finally:
            self._release_file_lock(lock_path)

    def update_compartments_by_source_uid(
        self, source_uid: str, new_status: str = "Reprocessed"
    ) -> int:
        """
        Update all compartments from a specific source image UID.
        Used during reprocessing to mark old compartments.

        Args:
            source_uid: Source image UID
            new_status: Status to set for existing compartments

        Returns:
            Number of compartments updated
        """
        with self._thread_lock:
            try:
                data = self._read_json_file(
                    self.compartment_json_path, self.compartment_lock
                )

                updated_count = 0
                for record in data:
                    if record.get("Source_Image_UID") == source_uid:
                        record["Photo_Status"] = new_status
                        record["Reprocessed_Date"] = datetime.now().isoformat()
                        updated_count += 1

                if updated_count > 0:
                    self._write_json_file(
                        self.compartment_json_path, self.compartment_lock, data
                    )
                    self.logger.info(
                        f"Updated {updated_count} compartments for reprocessing of UID {source_uid}"
                    )

                return updated_count

            except Exception as e:
                self.logger.error(f"Error updating compartments by source UID: {e}")
                return 0

    def update_original_image(
        self,
        hole_id: str,
        depth_from: int,
        depth_to: int,
        original_filename: str,
        is_approved: bool,
        upload_success: bool,
        uploaded_by: Optional[str] = None,
        comments: Optional[str] = None,
        total_rotation_angle: Optional[float] = None,
        scale_px_per_cm: Optional[float] = None,
        scale_confidence: Optional[float] = None,
        compartment_data: Optional[Dict[str, List[List[int]]]] = None,
        transformation_matrix: Optional[List[List[float]]] = None,
        cumulative_offset_y: Optional[float] = None,
        transformation_applied: Optional[bool] = None,
        transform_center: Optional[List[float]] = None,
        uid: Optional[str] = None,
        final_filename: Optional[str] = None,
        is_rejected: Optional[bool] = None,
        is_skipped: Optional[bool] = None,
        is_selective: Optional[bool] = None,
    ) -> bool:
        """
        Update or create an original image entry with flattened compartment storage.

        Args:
            hole_id: Hole identifier
            depth_from: Starting depth
            depth_to: Ending depth
            original_filename: Name of the original file
            is_approved: Whether the image is approved
            upload_success: Whether upload was successful
            uploaded_by: User who uploaded (optional)
            comments: Comments (optional)
            scale_px_per_cm: Scale in pixels per cm for this image
            scale_confidence: Confidence of scale detection
            compartment_data: Dict with compartment numbers as keys, values are corner coordinates
                            in compact format [[TL], [TR], [BR], [BL]]
                            e.g. {"1": [[0,0], [100,0], [100,100], [0,100]], "2": [...]}
        """
        with self._thread_lock:
            try:
                # Use file locks for both original and corners
                with self.file_locks("original", "corners"):
                    # Read current data
                    original_data = []
                    if self.original_json_path.exists():
                        with open(self.original_json_path, "r", encoding="utf-8") as f:
                            original_data = json.load(f)

                    corners_data = []
                    if self.compartment_corners_json_path.exists():
                        with open(
                            self.compartment_corners_json_path, "r", encoding="utf-8"
                        ) as f:
                            corners_data = json.load(f)

                    # Ensure depths are integers
                    depth_from = int(depth_from)
                    depth_to = int(depth_to)

                    timestamp = datetime.now().isoformat()
                    username = uploaded_by or os.getenv("USERNAME", "Unknown")
                    status = "Uploaded" if upload_success else "Failed"

                    # Generate UID if not provided
                    if uid is None:
                        uid = str(uuid.uuid4())
                        self.logger.info(f"Generated new UID: {uid}")

                    # Generate the final filename if not provided
                    if final_filename is None:
                        # Build expected filename based on status
                        if is_rejected:
                            suffix = "_Rejected"
                        elif is_skipped:
                            suffix = "_Skipped"
                        elif is_selective:
                            suffix = "_Selected_Compartments"
                        else:
                            suffix = "_Original"

                        # Get extension from original filename
                        _, ext = os.path.splitext(original_filename)
                        final_filename = (
                            f"{hole_id}_{int(depth_from)}-{int(depth_to)}{suffix}{ext}"
                        )
                    # Find existing record - try UID first, then fall back to original matching
                    record_found = False
                    reprocessing = False
                    for i, record in enumerate(original_data):
                        # First try to match by UID if record has one
                        if "UID" in record and record["UID"] == uid:
                            record_found = True
                            reprocessing = True
                            self.logger.info(
                                f"Found existing record by UID {uid} - updating for reprocessing"
                            )
                            original_data[i].update(
                                {
                                    "Uploaded_By": username,
                                    "Comments": comments,
                                    "UID": uid,  # Ensure UID is stored
                                    "Final_Filename": final_filename,  # Store final filename
                                    "Reprocessed_Date": datetime.now().isoformat(),
                                    "Reprocessed_Count": record.get(
                                        "Reprocessed_Count", 0
                                    )
                                    + 1,
                                }
                            )

                            if is_approved:
                                original_data[i]["Approved_Upload_Date"] = timestamp
                                original_data[i]["Approved_Upload_Status"] = status
                            else:
                                original_data[i]["Rejected_Upload_Date"] = timestamp
                                original_data[i]["Rejected_Upload_Status"] = status

                            # Update scale data
                            if scale_px_per_cm is not None:
                                original_data[i]["Scale_PxPerCm"] = scale_px_per_cm
                            if scale_confidence is not None:
                                original_data[i]["Scale_Confidence"] = scale_confidence

                            # Add rotation angle if provided
                            if total_rotation_angle is not None:
                                original_data[i][
                                    "Total_Rotation_Angle"
                                ] = total_rotation_angle

                            # Add transformation data if provided
                            if transformation_matrix is not None:
                                original_data[i][
                                    "Transformation_Matrix"
                                ] = transformation_matrix
                                # Extract Y offset from transformation matrix if not provided separately
                                if (
                                    cumulative_offset_y is None
                                    and len(transformation_matrix) > 1
                                    and len(transformation_matrix[1]) > 2
                                ):
                                    cumulative_offset_y = transformation_matrix[1][2]
                            if cumulative_offset_y is not None:
                                original_data[i][
                                    "Cumulative_Offset_Y"
                                ] = cumulative_offset_y
                            if transformation_applied is not None:
                                original_data[i][
                                    "Transformation_Applied"
                                ] = transformation_applied
                            if transform_center is not None:
                                original_data[i]["Transform_Center"] = transform_center

                            break
                        # Fall back to original matching for backward compatibility
                        elif (
                            "UID" not in record  # Only for records without UID
                            and record["HoleID"] == hole_id
                            and record["Depth_From"] == depth_from
                            and record["Depth_To"] == depth_to
                            and record["Original_Filename"] == original_filename
                        ):
                            # Update existing record
                            record_found = True
                            original_data[i].update(
                                {
                                    "Uploaded_By": username,
                                    "Comments": comments,
                                    "UID": uid,  # Add UID to existing record
                                    "Final_Filename": final_filename,  # Store final filename
                                }
                            )

                            if is_approved:
                                original_data[i]["Approved_Upload_Date"] = timestamp
                                original_data[i]["Approved_Upload_Status"] = status
                            else:
                                original_data[i]["Rejected_Upload_Date"] = timestamp
                                original_data[i]["Rejected_Upload_Status"] = status

                            # Update scale data
                            if scale_px_per_cm is not None:
                                original_data[i]["Scale_PxPerCm"] = scale_px_per_cm
                            if scale_confidence is not None:
                                original_data[i]["Scale_Confidence"] = scale_confidence

                            # Add rotation angle if provided
                            if total_rotation_angle is not None:
                                original_data[i][
                                    "Total_Rotation_Angle"
                                ] = total_rotation_angle

                            # Add transformation data if provided
                            if transformation_matrix is not None:
                                original_data[i][
                                    "Transformation_Matrix"
                                ] = transformation_matrix
                            if cumulative_offset_y is not None:
                                original_data[i][
                                    "Cumulative_Offset_Y"
                                ] = cumulative_offset_y
                            if transformation_applied is not None:
                                original_data[i][
                                    "Transformation_Applied"
                                ] = transformation_applied
                            if transform_center is not None:
                                original_data[i]["Transform_Center"] = transform_center

                            break

                    if not record_found:
                        # Create new record
                        new_record = {
                            "HoleID": hole_id,
                            "Depth_From": depth_from,
                            "Depth_To": depth_to,
                            "Original_Filename": original_filename,
                            "Final_Filename": final_filename,  # ADD THIS
                            "UID": uid,  # ADD THIS
                            "File_Count": 1,
                            "All_Filenames": original_filename,
                            "Approved_Upload_Date": timestamp if is_approved else None,
                            "Approved_Upload_Status": status if is_approved else None,
                            "Rejected_Upload_Date": (
                                timestamp if not is_approved else None
                            ),
                            "Rejected_Upload_Status": (
                                status if not is_approved else None
                            ),
                            "Uploaded_By": username,
                            "Comments": comments,
                            "Reprocessed_Date": None,
                            "Reprocessed_Count": 0,
                        }

                        # Add scale data if provided
                        if scale_px_per_cm is not None:
                            new_record["Scale_PxPerCm"] = scale_px_per_cm
                        if scale_confidence is not None:
                            new_record["Scale_Confidence"] = scale_confidence
                        # Add rotation angle if provided
                        if total_rotation_angle is not None:
                            new_record["Total_Rotation_Angle"] = total_rotation_angle

                        # Add transformation data - always include defaults
                        if transformation_matrix is not None:
                            new_record["Transformation_Matrix"] = transformation_matrix
                        else:
                            new_record["Transformation_Matrix"] = [
                                [1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                            ]

                        if cumulative_offset_y is not None:
                            new_record["Cumulative_Offset_Y"] = cumulative_offset_y
                        elif (
                            transformation_matrix is not None
                            and len(transformation_matrix) > 1
                        ):
                            # Extract from transformation matrix if available
                            new_record["Cumulative_Offset_Y"] = transformation_matrix[
                                1
                            ][2]
                        else:
                            new_record["Cumulative_Offset_Y"] = 0.0

                        if transformation_applied is not None:
                            new_record["Transformation_Applied"] = (
                                transformation_applied
                            )
                        else:
                            new_record["Transformation_Applied"] = False

                        if transform_center is not None:
                            new_record["Transform_Center"] = transform_center
                        else:
                            new_record["Transform_Center"] = [0.0, 0.0]

                        original_data.append(new_record)

                    # Now handle compartment data separately
                    if compartment_data is not None:
                        # Remove existing corners for this image
                        corners_data = [
                            c
                            for c in corners_data
                            if not (
                                c["HoleID"] == hole_id
                                and c["Depth_From"] == depth_from
                                and c["Depth_To"] == depth_to
                                and c["Original_Filename"] == original_filename
                            )
                        ]

                        # Add new corners
                        for comp_num, corners in compartment_data.items():
                            if isinstance(corners, list) and len(corners) == 4:
                                corner_record = {
                                    "HoleID": hole_id,
                                    "Depth_From": depth_from,
                                    "Depth_To": depth_to,
                                    "Original_Filename": original_filename,
                                    "Compartment_Number": int(comp_num),
                                    "Top_Left_X": corners[0][0],
                                    "Top_Left_Y": corners[0][1],
                                    "Top_Right_X": corners[1][0],
                                    "Top_Right_Y": corners[1][1],
                                    "Bottom_Right_X": corners[2][0],
                                    "Bottom_Right_Y": corners[2][1],
                                    "Bottom_Left_X": corners[3][0],
                                    "Bottom_Left_Y": corners[3][1],
                                }

                                # Copy scale data if available
                                if scale_px_per_cm is not None:
                                    corner_record["Scale_PxPerCm"] = scale_px_per_cm
                                if scale_confidence is not None:
                                    corner_record["Scale_Confidence"] = scale_confidence

                                # Add UID if provided
                                if uid is not None:
                                    corner_record["Source_Image_UID"] = uid

                                corners_data.append(corner_record)

                    # Save both files
                    with open(self.original_json_path, "w", encoding="utf-8") as f:
                        json.dump(
                            original_data, f, indent=2, ensure_ascii=False, default=str
                        )

                    with open(
                        self.compartment_corners_json_path, "w", encoding="utf-8"
                    ) as f:
                        json.dump(
                            corners_data, f, indent=2, ensure_ascii=False, default=str
                        )

                    self.logger.info(f"Updated original image: {original_filename}")
                    return True

            except Exception as e:
                self.logger.error(f"Error updating original image: {e}")
                self.logger.error(traceback.format_exc())
                return False

    def update_compartment(
        self,
        hole_id: str,
        depth_from: int,
        depth_to: int,
        photo_status: str,
        processed_by: Optional[str] = None,
        comments: Optional[str] = None,
        image_width_cm: Optional[float] = None,
        source_image_uid: Optional[str] = None,
    ) -> bool:
        """
        Update or create a compartment entry.

        Args:
            hole_id: Hole identifier
            depth_from: Starting depth
            depth_to: Ending depth
            photo_status: Status of the photo (e.g., "For Review", "Approved", "Rejected")
            processed_by: Person who processed (optional)
            comments: Comments about the compartment (optional)
            image_width_cm: Individual compartment width in centimeters (optional)
        """
        with self._thread_lock:
            try:
                # Read current data
                data = self._read_json_file(
                    self.compartment_json_path, self.compartment_lock
                )

                # Convert to records for easier manipulation
                records = {(r["HoleID"], r["From"], r["To"]): r for r in data}

                # Ensure depths are integers
                depth_from = int(depth_from)
                depth_to = int(depth_to)

                # Update or create record
                key = (hole_id, depth_from, depth_to)
                timestamp = datetime.now().isoformat()
                username = processed_by or os.getenv("USERNAME", "Unknown")

                if key in records:
                    # Update existing
                    records[key].update(
                        {
                            "Photo_Status": photo_status,
                            "Processed_Date": timestamp,
                            "Processed_By": username,
                        }
                    )
                    if comments:
                        records[key]["Comments"] = comments

                    # Add image width if provided
                    if image_width_cm is not None:
                        records[key]["Image_Width_Cm"] = image_width_cm
                    if source_image_uid is not None:
                        records[key]["Source_Image_UID"] = source_image_uid

                else:
                    # Create new
                    new_record = {
                        "HoleID": hole_id,
                        "From": depth_from,
                        "To": depth_to,
                        "Photo_Status": photo_status,
                        "Processed_Date": timestamp,
                        "Processed_By": username,
                        "Comments": comments,
                    }

                    # Add image width to new record if provided
                    if image_width_cm is not None:
                        new_record["Image_Width_Cm"] = image_width_cm
                    if source_image_uid is not None:
                        new_record["Source_Image_UID"] = source_image_uid

                    records[key] = new_record

                # Convert back to list and save
                data = list(records.values())
                self._write_json_file(
                    self.compartment_json_path, self.compartment_lock, data
                )

                self.logger.info(
                    f"Updated compartment: {hole_id} {depth_from}-{depth_to}"
                )
                return True

            except Exception as e:
                self.logger.error(f"Error updating compartment: {e}")
                return False

    def get_original_image_by_uid(self, uid: str) -> Optional[Dict[str, Any]]:
        """
        Get original image record by UID.

        Args:
            uid: Unique identifier of the record

        Returns:
            Record dictionary or None if not found
        """
        with self._thread_lock:
            try:
                data = self._read_json_file(self.original_json_path, self.original_lock)

                for record in data:
                    if record.get("UID") == uid:
                        return record.copy()

                return None

            except Exception as e:
                self.logger.error(f"Error getting record by UID: {e}")
                return None

    def get_compartment_corners_from_image(
        self,
        hole_id: str,
        depth_from: int,
        depth_to: int,
        original_filename: str,
        compartment_num: str,
    ) -> Optional[Dict[str, Tuple[int, int]]]:
        """
        Get corners for a specific compartment from a specific original image.

        Args:
            hole_id: Hole identifier
            depth_from: Starting depth
            depth_to: Ending depth
            original_filename: Name of the original image file
            compartment_num: Compartment number (e.g., "1", "2", "3")

        Returns:
            Dictionary with 'top_left', 'top_right', 'bottom_right', 'bottom_left' as (x, y) tuples
        """
        with self._thread_lock:
            try:
                data = self._read_json_file(
                    self.compartment_corners_json_path, self.corners_lock
                )

                depth_from = int(depth_from)
                depth_to = int(depth_to)
                comp_num_int = int(compartment_num)

                for record in data:
                    if (
                        record["HoleID"] == hole_id
                        and record["Depth_From"] == depth_from
                        and record["Depth_To"] == depth_to
                        and record["Original_Filename"] == original_filename
                        and record["Compartment_Number"] == comp_num_int
                    ):

                        return {
                            "top_left": (record["Top_Left_X"], record["Top_Left_Y"]),
                            "top_right": (record["Top_Right_X"], record["Top_Right_Y"]),
                            "bottom_right": (
                                record["Bottom_Right_X"],
                                record["Bottom_Right_Y"],
                            ),
                            "bottom_left": (
                                record["Bottom_Left_X"],
                                record["Bottom_Left_Y"],
                            ),
                        }

                return None

            except Exception as e:
                self.logger.error(f"Error getting compartment corners: {e}")
                return None

    def batch_remove_compartments(self, removals: List[Dict]) -> int:
        """
        Batch remove compartments from flattened corner records.

        Args:
            removals: List of dictionaries with keys:
                - hole_id: Hole identifier
                - depth_from: Starting depth
                - depth_to: Ending depth
                - original_filename: Name of the original image file
                - compartment_num: Compartment number to remove

        Returns:
            Number of successful removals
        """
        successful_removals = 0

        with self._thread_lock:
            try:
                # Read data once
                data = self._read_json_file(
                    self.compartment_corners_json_path, self.corners_lock
                )

                # Create a set of records to remove for efficiency
                removal_keys = set()
                for removal in removals:
                    key = (
                        removal["hole_id"],
                        int(removal["depth_from"]),
                        int(removal["depth_to"]),
                        removal["original_filename"],
                        int(removal["compartment_num"]),
                    )
                    removal_keys.add(key)

                # Filter out records that match removal criteria
                filtered_data = []
                for record in data:
                    record_key = (
                        record["HoleID"],
                        record["Depth_From"],
                        record["Depth_To"],
                        record["Original_Filename"],
                        record["Compartment_Number"],
                    )

                    if record_key not in removal_keys:
                        filtered_data.append(record)
                    else:
                        successful_removals += 1
                        self.logger.info(
                            f"Removed compartment {record['Compartment_Number']} from {record['Original_Filename']}"
                        )

                # Save if we made changes
                if successful_removals > 0:
                    self._write_json_file(
                        self.compartment_corners_json_path,
                        self.corners_lock,
                        filtered_data,
                    )
                    self.logger.info(
                        f"Batch removed {successful_removals} compartments"
                    )

            except Exception as e:
                self.logger.error(f"Error in batch_remove_compartments: {e}")

        return successful_removals

    def _merge_compartment_data(self, stats, file_manager, current_suffix):
        """Merge compartment data from all users."""
        all_compartments = {}
        current_compartments = {}

        # First, load current user's data
        current_data = self._read_json_file(
            self.compartment_json_path, self.compartment_lock
        )
        for record in current_data:
            key = (record["HoleID"], record["From"], record["To"])
            current_compartments[key] = record

        # Process each user file
        for file_path in self.get_all_user_files("compartment"):
            # Skip current user's file
            if current_suffix in str(file_path):
                continue

            # Check if file is locked
            lock_path = file_path.with_suffix(".json.lock")
            if lock_path.exists():
                self.logger.warning(f"Skipping locked file: {file_path}")
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if not isinstance(data, list):
                    continue

                # Extract user info from filename
                user_info = file_path.stem.replace(self.COMPARTMENT_JSON_BASE, "")
                if user_info.startswith("_"):
                    user_info = user_info[1:]

                # Process each record
                for record in data:
                    if not isinstance(record, dict):
                        continue

                    key = (record["HoleID"], record["From"], record["To"])

                    # Check if we already have this compartment
                    if key in current_compartments:
                        stats["compartments_conflicts"] += 1

                        # Resolve conflict by checking which file exists
                        if file_manager:
                            existing_status = current_compartments[key].get(
                                "Photo_Status", ""
                            )
                            new_status = record.get("Photo_Status", "")

                            # Check if new record's file exists
                            new_file_exists = self._check_compartment_file_exists(
                                record["HoleID"], record["To"], new_status, file_manager
                            )

                            # Check if current record's file exists
                            current_file_exists = self._check_compartment_file_exists(
                                current_compartments[key]["HoleID"],
                                current_compartments[key]["To"],
                                existing_status,
                                file_manager,
                            )

                            # Prefer record with existing file
                            if new_file_exists and not current_file_exists:
                                record["Merged_From_User"] = user_info
                                all_compartments[key] = record
                            elif not new_file_exists and not current_file_exists:
                                # Both missing, keep most recent
                                if record.get(
                                    "Processed_Date", ""
                                ) > current_compartments[key].get("Processed_Date", ""):
                                    record["Merged_From_User"] = user_info
                                    all_compartments[key] = record
                    else:
                        # New compartment
                        record["Merged_From_User"] = user_info
                        all_compartments[key] = record
                        stats["compartments_merged"] += 1

                # Clean up source file by resetting to empty
                self._cleanup_user_file(file_path, lock_path)

            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                stats["errors"].append(f"Error in {file_path.name}: {str(e)}")

        # Merge with current data
        for key, record in all_compartments.items():
            current_compartments[key] = record

        # Save merged data
        merged_data = list(current_compartments.values())
        self._write_json_file(
            self.compartment_json_path, self.compartment_lock, merged_data
        )

    def _merge_original_image_data(self, stats, file_manager, current_suffix):
        """Merge original image data from all users."""
        all_originals = {}
        current_originals = {}

        # Load current user's data
        current_data = self._read_json_file(self.original_json_path, self.original_lock)
        for record in current_data:
            # Use filename as part of key since multiple files can exist per depth range
            key = (
                record["HoleID"],
                record["Depth_From"],
                record["Depth_To"],
                record["Original_Filename"],
            )
            current_originals[key] = record

        # Process each user file
        for file_path in self.get_all_user_files("original"):
            if current_suffix in str(file_path):
                continue

            lock_path = file_path.with_suffix(".json.lock")
            if lock_path.exists():
                self.logger.warning(f"Skipping locked file: {file_path}")
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if not isinstance(data, list):
                    continue

                user_info = file_path.stem.replace(self.ORIGINAL_JSON_BASE, "")
                if user_info.startswith("_"):
                    user_info = user_info[1:]

                for record in data:
                    if not isinstance(record, dict):
                        continue

                    key = (
                        record["HoleID"],
                        record["Depth_From"],
                        record["Depth_To"],
                        record["Original_Filename"],
                    )

                    if key in current_originals:
                        stats["originals_conflicts"] += 1

                        # Check file existence to resolve conflict
                        if file_manager:
                            new_file_exists = self._check_original_file_exists(
                                record["HoleID"],
                                record["Depth_From"],
                                record["Depth_To"],
                                record["Original_Filename"],
                                record.get("Approved_Upload_Status") == "Uploaded",
                                file_manager,
                            )

                            current_file_exists = self._check_original_file_exists(
                                current_originals[key]["HoleID"],
                                current_originals[key]["Depth_From"],
                                current_originals[key]["Depth_To"],
                                current_originals[key]["Original_Filename"],
                                current_originals[key].get("Approved_Upload_Status")
                                == "Uploaded",
                                file_manager,
                            )

                            if new_file_exists and not current_file_exists:
                                record["Merged_From_User"] = user_info
                                all_originals[key] = record
                    else:
                        record["Merged_From_User"] = user_info
                        all_originals[key] = record
                        stats["originals_merged"] += 1

                self._cleanup_user_file(file_path, lock_path)

            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                stats["errors"].append(f"Error in {file_path.name}: {str(e)}")

        # Merge and save
        for key, record in all_originals.items():
            current_originals[key] = record

        merged_data = list(current_originals.values())
        self._write_json_file(self.original_json_path, self.original_lock, merged_data)

    def _merge_review_data(self, stats, current_suffix):
        """Merge review data from all users (keep all reviews from all users)."""
        # For reviews, we want to keep ALL reviews from all users
        all_reviews = []

        # Load current user's reviews
        current_data = self._read_json_file(self.review_json_path, self.review_lock)
        all_reviews.extend(current_data)

        # Process each user file
        for file_path in self.get_all_user_files("review"):
            if current_suffix in str(file_path):
                continue

            lock_path = file_path.with_suffix(".json.lock")
            if lock_path.exists():
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    # Keep all reviews - they're user-specific
                    all_reviews.extend(data)
                    stats["reviews_merged"] += len(data)

                self._cleanup_user_file(file_path, lock_path)

            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")

        # Save merged reviews
        self._write_json_file(self.review_json_path, self.review_lock, all_reviews)

    def _merge_corner_data(self, stats, current_suffix):
        """Merge compartment corner data from all users."""
        all_corners = {}

        # Load current data
        current_data = self._read_json_file(
            self.compartment_corners_json_path, self.corners_lock
        )
        for record in current_data:
            key = (
                record["HoleID"],
                record["Depth_From"],
                record["Depth_To"],
                record["Original_Filename"],
                record["Compartment_Number"],
            )
            all_corners[key] = record

        # Process user files
        for file_path in self.get_all_user_files("corners"):
            if current_suffix in str(file_path):
                continue

            lock_path = file_path.with_suffix(".json.lock")
            if lock_path.exists():
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    for record in data:
                        key = (
                            record["HoleID"],
                            record["Depth_From"],
                            record["Depth_To"],
                            record["Original_Filename"],
                            record["Compartment_Number"],
                        )

                        if key not in all_corners:
                            all_corners[key] = record
                            stats["corners_merged"] += 1

                self._cleanup_user_file(file_path, lock_path)

            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")

        # Save merged data
        merged_data = list(all_corners.values())
        self._write_json_file(
            self.compartment_corners_json_path, self.corners_lock, merged_data
        )

    def _check_compartment_file_exists(
        self, hole_id, depth_to, photo_status, file_manager
    ):
        """Check if a compartment image file exists."""
        try:
            # Determine which folder to check based on status
            if photo_status == "For Review":
                base_path = file_manager.get_shared_path(
                    "review_compartments", create_if_missing=False
                )
            else:
                base_path = file_manager.get_shared_path(
                    "approved_compartments", create_if_missing=False
                )

            if not base_path:
                return False

            project_code = hole_id[:2].upper()
            hole_path = base_path / project_code / hole_id

            if not hole_path.exists():
                return False

            # Check for any matching compartment file
            patterns = [
                f"{hole_id}_CC_{depth_to:03d}.png",
                f"{hole_id}_CC_{depth_to:03d}_Wet.png",
                f"{hole_id}_CC_{depth_to:03d}_Dry.png",
            ]

            for pattern in patterns:
                if any(pattern.lower() in f.lower() for f in os.listdir(hole_path)):
                    return True

            return False
        except:
            return False

    def _check_original_file_exists(
        self, hole_id, depth_from, depth_to, filename, is_approved, file_manager
    ):
        """Check if an original image file exists."""
        try:
            if is_approved:
                base_path = file_manager.get_shared_path(
                    "processed_originals", create_if_missing=False
                )
            else:
                base_path = file_manager.get_shared_path(
                    "rejected_originals", create_if_missing=False
                )

            if not base_path:
                return False

            project_code = hole_id[:2].upper()
            hole_path = base_path / project_code / hole_id

            if not hole_path.exists():
                return False

            # Check if the specific file exists
            return filename in os.listdir(hole_path)
        except:
            return False

    def _cleanup_user_file(self, file_path, lock_path):
        """Clean up a user file by resetting to empty structure."""
        # Always reset to empty structure with example entry
        try:
            # Use the _write_json_file method which adds example entries for empty lists
            self._write_json_file(file_path, lock_path, [])
            self.logger.info(f"Reset file to example structure: {file_path}")
        except Exception as e:
            self.logger.error(f"Error resetting {file_path}: {e}")

    def update_compartment_review(
        self,
        hole_id: str,
        depth_from: int,
        depth_to: int,
        reviewed_by: Optional[str] = None,
        comments: Optional[str] = None,
        review_number: Optional[int] = None,
        **kwargs,
    ) -> bool:
        """
        Update or create a user-specific compartment review entry.
        Each user has ONE row per compartment, with review_number tracking edits.

        This method preserves all existing fields in the JSON even if they're not
        in the current config, ensuring no data loss when toggles are modified.

        Args:
            hole_id: Hole identifier
            depth_from: Starting depth
            depth_to: Ending depth
            reviewed_by: Person who reviewed (defaults to current user)
            comments: Review comments
            review_number: Explicit review number (if None, auto-increments)
            **kwargs: Additional fields from config (toggle fields, etc.)
        """
        with self._thread_lock:
            try:
                # Read current data
                data = self._read_json_file(self.review_json_path, self.review_lock)

                # Ensure depths are integers
                depth_from = int(depth_from)
                depth_to = int(depth_to)

                # Get username
                username = reviewed_by or os.getenv("USERNAME", "Unknown")

                # Create timestamp
                timestamp = datetime.now().isoformat()

                # Find existing review by this user for this compartment
                existing_index = None
                existing_review = None
                for idx, review in enumerate(data):
                    if (
                        review["HoleID"] == hole_id
                        and review["From"] == depth_from
                        and review["To"] == depth_to
                        and review["Reviewed_By"] == username
                    ):
                        existing_index = idx
                        existing_review = (
                            review.copy()
                        )  # Keep a copy to preserve all fields
                        break

                if existing_index is not None:
                    # Update existing review - preserve all existing fields
                    if review_number is not None:
                        use_review_number = review_number
                    else:
                        use_review_number = existing_review.get("Review_Number", 0) + 1

                    # Start with existing data to preserve all fields
                    updated_review = existing_review.copy()

                    # Update core fields
                    updated_review.update(
                        {
                            "Review_Number": use_review_number,
                            "Review_Date": timestamp,
                            "Comments": comments,
                        }
                    )

                    # Update fields from kwargs (new toggle values)
                    updated_review.update(kwargs)

                    # Replace the record
                    data[existing_index] = updated_review

                    self.logger.info(
                        f"Updated existing review (edit #{use_review_number}) by {username} for: {hole_id} {depth_from}-{depth_to}"
                    )
                else:
                    # Create new review record
                    use_review_number = (
                        review_number if review_number is not None else 1
                    )

                    new_record = {
                        "HoleID": hole_id,
                        "From": depth_from,
                        "To": depth_to,
                        "Reviewed_By": username,
                        "Review_Number": use_review_number,
                        "Review_Date": timestamp,
                        "Initial_Review_Date": timestamp,
                        "Comments": comments,
                    }

                    # Add all kwargs (toggle fields from config)
                    new_record.update(kwargs)

                    # Add the new record
                    data.append(new_record)

                    self.logger.info(
                        f"Created new review #{use_review_number} by {username} for: {hole_id} {depth_from}-{depth_to}"
                    )

                # Save updated data
                self._write_json_file(self.review_json_path, self.review_lock, data)
                return True

            except Exception as e:
                self.logger.error(f"Error updating compartment review: {e}")
                self.logger.error(traceback.format_exc())
                return False

    def batch_update_compartments(self, updates: List[Dict]) -> int:
        """
        Batch update multiple compartment entries.

        NOTE: This should NOT create individual entries in the original image register.
        It only updates the compartment register.
        """
        successful_updates = 0

        with self._thread_lock:
            # Only need compartment lock for this operation
            try:
                with self.file_locks("compartment"):
                    # Read current compartment data
                    comp_data = []
                    if self.compartment_json_path.exists():
                        # Handle empty or invalid JSON files
                        try:
                            # Check if file is empty
                            if self.compartment_json_path.stat().st_size == 0:
                                self.logger.warning(
                                    "Compartment JSON file is empty, starting with empty list"
                                )
                                comp_data = []
                            else:
                                with open(
                                    self.compartment_json_path, "r", encoding="utf-8"
                                ) as f:
                                    comp_data = json.load(f)
                        except json.JSONDecodeError as e:
                            self.logger.error(
                                f"JSON decode error in compartment file: {e}"
                            )
                            self.logger.warning("Starting with empty compartment list")
                            comp_data = []
                        except Exception as e:
                            self.logger.error(f"Error reading compartment file: {e}")
                            comp_data = []

                    comp_records = {
                        (r["HoleID"], r["From"], r["To"]): r for r in comp_data
                    }

                    # Process all compartment updates
                    for update in updates:
                        try:
                            hole_id = update["hole_id"]
                            depth_from = int(update["depth_from"])
                            depth_to = int(update["depth_to"])
                            photo_status = update["photo_status"]
                            source_uid = update.get("source_image_uid")

                            key = (hole_id, depth_from, depth_to)
                            timestamp = datetime.now().isoformat()
                            username = update.get(
                                "processed_by", os.getenv("USERNAME", "Unknown")
                            )

                            # Update compartment register
                            if key in comp_records:
                                # Update existing
                                comp_records[key].update(
                                    {
                                        "Photo_Status": photo_status,
                                        "Processed_Date": timestamp,
                                        "Processed_By": username,
                                    }
                                )
                                if update.get("comments"):
                                    comp_records[key]["Comments"] = update["comments"]
                                if "image_width_cm" in update:
                                    comp_records[key]["Image_Width_Cm"] = update[
                                        "image_width_cm"
                                    ]
                                if source_uid:
                                    comp_records[key]["Source_Image_UID"] = source_uid
                            else:
                                # Create new
                                new_record = {
                                    "HoleID": hole_id,
                                    "From": depth_from,
                                    "To": depth_to,
                                    "Photo_Status": photo_status,
                                    "Processed_Date": timestamp,
                                    "Processed_By": username,
                                    "Comments": update.get("comments"),
                                }
                                if "image_width_cm" in update:
                                    new_record["Image_Width_Cm"] = update[
                                        "image_width_cm"
                                    ]

                                comp_records[key] = new_record

                            successful_updates += 1

                        except Exception as e:
                            self.logger.error(
                                f"Error in batch update for {update}: {e}"
                            )

                    # Write all updates at once
                    if successful_updates > 0:
                        # Save compartment data with compact formatting
                        with open(
                            self.compartment_json_path, "w", encoding="utf-8"
                        ) as f:
                            json.dump(
                                list(comp_records.values()),
                                f,
                                indent=2,
                                ensure_ascii=False,
                                default=str,
                                separators=(",", ": "),
                            )

                        self.logger.info(
                            f"Batch updated {successful_updates} compartments"
                        )

            except Exception as e:
                self.logger.error(f"Error in batch_update_compartments: {e}")

        return successful_updates

    def batch_update_compartment_colors(self, color_updates: List[Dict]) -> int:
        """
        Batch update compartment hex colors.

        Args:
            color_updates: List of dictionaries with keys:
                - hole_id: Hole ID
                - depth_from: Starting depth
                - depth_to: Ending depth
                - average_hex_color: Hex color string

        Returns:
            Number of successful updates
        """
        if not color_updates:
            return 0

        successful_updates = 0

        with self._thread_lock:
            try:
                # Read current data
                data = self._read_json_file(
                    self.compartment_json_path, self.compartment_lock
                )

                # Convert to records for easier manipulation
                records = {(r["HoleID"], r["From"], r["To"]): r for r in data}

                # Apply updates
                for update in color_updates:
                    try:
                        hole_id = update["hole_id"]
                        depth_from = int(update["depth_from"])
                        depth_to = int(update["depth_to"])
                        hex_color = update["average_hex_color"]

                        key = (hole_id, depth_from, depth_to)

                        if key in records:
                            # Add or update the Average_Hex_Color field
                            records[key]["Average_Hex_Color"] = hex_color
                            successful_updates += 1
                        else:
                            self.logger.warning(
                                f"No matching compartment found for {hole_id} {depth_from}-{depth_to}m"
                            )

                    except Exception as e:
                        self.logger.error(f"Error updating color for {update}: {e}")

                # Save if we made updates
                if successful_updates > 0:
                    data = list(records.values())
                    self._write_json_file(
                        self.compartment_json_path, self.compartment_lock, data
                    )
                    self.logger.info(
                        f"Successfully updated {successful_updates} hex colors"
                    )

            except Exception as e:
                self.logger.error(f"Error batch updating hex colors: {e}")

        return successful_updates

    def get_user_review(
        self,
        hole_id: str,
        depth_from: int,
        depth_to: int,
        username: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Get the review by a specific user for a compartment.
        Returns ALL fields stored in the JSON, not just current config fields.

        Args:
            hole_id: Hole identifier
            depth_from: Starting depth
            depth_to: Ending depth
            username: User to get review for (defaults to current user)

        Returns:
            Review dictionary with all stored fields or None if no review exists
        """
        with self._thread_lock:
            try:
                data = self._read_json_file(self.review_json_path, self.review_lock)

                # Default to current user
                username = username or os.getenv("USERNAME", "Unknown")

                # Ensure depths are integers
                depth_from = int(depth_from)
                depth_to = int(depth_to)

                # Find review for this compartment by this user
                for review in data:
                    if (
                        review["HoleID"] == hole_id
                        and review["From"] == depth_from
                        and review["To"] == depth_to
                        and review["Reviewed_By"] == username
                    ):
                        return (
                            review.copy()
                        )  # Return a copy to prevent accidental modification

                return None

            except Exception as e:
                self.logger.error(f"Error getting user review: {e}")
                return None

    def get_all_reviews_for_compartment(
        self, hole_id: str, depth_from: int, depth_to: int
    ) -> List[Dict]:
        """
        Get all reviews for a specific compartment (one per user).
        Returns ALL fields for each review, preserving legacy data.

        Returns:
            List of review dictionaries with all stored fields
        """
        with self._thread_lock:
            try:
                data = self._read_json_file(self.review_json_path, self.review_lock)

                # Ensure depths are integers
                depth_from = int(depth_from)
                depth_to = int(depth_to)

                # Filter reviews for this compartment
                reviews = [
                    r.copy()
                    for r in data
                    if r["HoleID"] == hole_id
                    and r["From"] == depth_from
                    and r["To"] == depth_to
                ]

                # Sort by most recent review date
                reviews.sort(key=lambda x: x.get("Review_Date", ""), reverse=True)
                return reviews

            except Exception as e:
                self.logger.error(f"Error getting all reviews: {e}")
                return []

    def get_compartment_data(self, hole_id: Optional[str] = None) -> pd.DataFrame:
        """Get compartment data as DataFrame (current user's data only)."""
        with self._thread_lock:
            try:
                data = self._read_json_file(
                    self.compartment_json_path, self.compartment_lock
                )
                df = pd.DataFrame(data)

                # Ensure Average_Hex_Color column exists for backwards compatibility
                if not df.empty and "Average_Hex_Color" not in df.columns:
                    df["Average_Hex_Color"] = None

                if hole_id and not df.empty:
                    df = df[df["HoleID"] == hole_id]

                return df

            except Exception as e:
                self.logger.error(f"Error getting compartment data: {e}")
                return pd.DataFrame()

    def get_original_image_data(self, hole_id: Optional[str] = None) -> pd.DataFrame:
        """Get original image data as DataFrame (current user's data only)."""
        with self._thread_lock:
            try:
                data = self._read_json_file(self.original_json_path, self.original_lock)
                df = pd.DataFrame(data)

                if hole_id and not df.empty:
                    df = df[df["HoleID"] == hole_id]

                return df

            except Exception as e:
                self.logger.error(f"Error getting original image data: {e}")
                return pd.DataFrame()

    def get_compartment_corners_data(
        self, hole_id: Optional[str] = None
    ) -> pd.DataFrame:
        """Get compartment corners data as DataFrame (current user's data only)."""
        with self._thread_lock:
            try:
                data = self._read_json_file(
                    self.compartment_corners_json_path, self.corners_lock
                )
                df = pd.DataFrame(data)

                if hole_id and not df.empty:
                    df = df[df["HoleID"] == hole_id]

                return df

            except Exception as e:
                self.logger.error(f"Error getting compartment corners data: {e}")
                return pd.DataFrame()

    def get_review_data(self, hole_id: Optional[str] = None) -> pd.DataFrame:
        """
        Get review data as DataFrame (current user's data only).
        Returns ALL fields stored in the JSON, not just current config fields.
        """
        with self._thread_lock:
            try:
                data = self._read_json_file(self.review_json_path, self.review_lock)
                df = pd.DataFrame(data)

                if hole_id and not df.empty:
                    df = df[df["HoleID"] == hole_id]

                return df

            except Exception as e:
                self.logger.error(f"Error getting review data: {e}")
                return pd.DataFrame()

    def get_review_field_summary(self) -> Dict[str, int]:
        """
        Get a summary of all fields that appear in review data (current user only).
        Useful for understanding what legacy fields exist.

        Returns:
            Dictionary mapping field names to count of records containing that field
        """
        with self._thread_lock:
            try:
                data = self._read_json_file(self.review_json_path, self.review_lock)

                field_counts = {}
                for record in data:
                    for field in record:
                        field_counts[field] = field_counts.get(field, 0) + 1

                return field_counts

            except Exception as e:
                self.logger.error(f"Error getting review field summary: {e}")
                return {}

    def get_lock_statistics(self) -> Dict[str, Any]:
        """Get current lock statistics for debugging."""
        return self._lock_stats.copy()

    def merge_all_user_registers(self, file_manager=None, cleanup_backups=True):
        """
        Merge all user-specific register files into the MAIN register files.
        After merging, other user files are reset to empty structure.

        Args:
            file_manager: Optional FileManager instance to check file existence
            cleanup_backups: Whether to remove backup files after successful merge

        Returns:
            Dictionary with merge statistics
        """
        stats = {
            "compartments_merged": 0,
            "compartments_conflicts": 0,
            "originals_merged": 0,
            "originals_conflicts": 0,
            "reviews_merged": 0,
            "corners_merged": 0,
            "backups_cleaned": 0,
            "errors": [],
        }

        # Get main register paths (without user suffix)
        main_paths = self._get_main_register_paths()
        main_locks = self._get_main_lock_paths()

        try:
            # Acquire all locks for main files
            with self.file_locks("compartment", "original", "review", "corners"):
                # Merge each type of data
                self._merge_compartment_data_to_main(
                    stats,
                    file_manager,
                    main_paths["compartment"],
                    main_locks["compartment"],
                )
                self._merge_original_image_data_to_main(
                    stats, file_manager, main_paths["original"], main_locks["original"]
                )
                self._merge_review_data_to_main(
                    stats, main_paths["review"], main_locks["review"]
                )
                self._merge_corner_data_to_main(
                    stats, main_paths["corners"], main_locks["corners"]
                )

                # Cleanup backup files if requested
                if cleanup_backups:
                    stats["backups_cleaned"] = self._cleanup_backup_files(main_paths)

        except Exception as e:
            self.logger.error(f"Error during merge: {str(e)}")
            stats["errors"].append(str(e))

        return stats

    def _merge_compartment_data_to_main(
        self, stats, file_manager, main_path, main_lock
    ):
        """Merge compartment data from all users to main file."""
        all_compartments = {}

        # First, load data from main file if it exists
        if main_path.exists():
            try:
                with open(main_path, "r", encoding="utf-8") as f:
                    main_data = json.load(f)
                    if isinstance(main_data, list):
                        for record in main_data:
                            if isinstance(record, dict) and "HoleID" in record:
                                key = (record["HoleID"], record["From"], record["To"])
                                all_compartments[key] = record
            except Exception as e:
                self.logger.error(f"Error reading main compartment file: {e}")

        # Process each user file
        for file_path in self.get_all_user_files("compartment"):
            # Skip main file if it exists
            if file_path == main_path:
                continue

            # Check if file is locked
            lock_path = file_path.with_suffix(".json.lock")
            if lock_path.exists():
                self.logger.warning(f"Skipping locked file: {file_path}")
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if not isinstance(data, list):
                    continue

                # Extract user info from filename
                user_info = file_path.stem.replace(self.COMPARTMENT_JSON_BASE, "")
                if user_info.startswith("_"):
                    user_info = user_info[1:]

                # Process each record
                for record in data:
                    if not isinstance(record, dict):
                        continue

                    key = (record["HoleID"], record["From"], record["To"])

                    # Check if we already have this compartment
                    if key in all_compartments:
                        stats["compartments_conflicts"] += 1

                        # Resolve conflict by checking which file exists
                        if file_manager:
                            existing_status = all_compartments[key].get(
                                "Photo_Status", ""
                            )
                            new_status = record.get("Photo_Status", "")

                            # Check if new record's file exists
                            new_file_exists = self._check_compartment_file_exists(
                                record["HoleID"], record["To"], new_status, file_manager
                            )

                            # Check if current record's file exists
                            current_file_exists = self._check_compartment_file_exists(
                                all_compartments[key]["HoleID"],
                                all_compartments[key]["To"],
                                existing_status,
                                file_manager,
                            )

                            # Prefer record with existing file
                            if new_file_exists and not current_file_exists:
                                record["Merged_From_User"] = user_info
                                all_compartments[key] = record
                            elif not new_file_exists and not current_file_exists:
                                # Both missing, keep most recent
                                if record.get("Processed_Date", "") > all_compartments[
                                    key
                                ].get("Processed_Date", ""):
                                    record["Merged_From_User"] = user_info
                                    all_compartments[key] = record
                    else:
                        # New compartment
                        record["Merged_From_User"] = user_info
                        all_compartments[key] = record
                        stats["compartments_merged"] += 1

                # Clean up source file by resetting to empty
                self._cleanup_user_file(file_path, lock_path)

            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                stats["errors"].append(f"Error in {file_path.name}: {str(e)}")

        # Save merged data to MAIN file
        merged_data = list(all_compartments.values())
        self._write_json_file(main_path, main_lock, merged_data)
        self.logger.info(
            f"Wrote {len(merged_data)} compartment records to main register"
        )

    def _merge_original_image_data_to_main(
        self, stats, file_manager, main_path, main_lock
    ):
        """Merge original image data from all users to main file."""
        all_originals = {}

        # Load main file data first
        if main_path.exists():
            try:
                with open(main_path, "r", encoding="utf-8") as f:
                    main_data = json.load(f)
                    if isinstance(main_data, list):
                        for record in main_data:
                            if isinstance(record, dict) and "HoleID" in record:
                                key = (
                                    record["HoleID"],
                                    record["Depth_From"],
                                    record["Depth_To"],
                                    record["Original_Filename"],
                                )
                                all_originals[key] = record
            except Exception as e:
                self.logger.error(f"Error reading main original file: {e}")

        # Process user files
        for file_path in self.get_all_user_files("original"):
            if file_path == main_path:
                continue

            lock_path = file_path.with_suffix(".json.lock")
            if lock_path.exists():
                self.logger.warning(f"Skipping locked file: {file_path}")
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if not isinstance(data, list):
                    continue

                user_info = file_path.stem.replace(self.ORIGINAL_JSON_BASE, "")
                if user_info.startswith("_"):
                    user_info = user_info[1:]

                for record in data:
                    if not isinstance(record, dict):
                        continue

                    key = (
                        record["HoleID"],
                        record["Depth_From"],
                        record["Depth_To"],
                        record["Original_Filename"],
                    )

                    if key in all_originals:
                        stats["originals_conflicts"] += 1

                        # Check file existence to resolve conflict
                        if file_manager:
                            new_file_exists = self._check_original_file_exists(
                                record["HoleID"],
                                record["Depth_From"],
                                record["Depth_To"],
                                record["Original_Filename"],
                                record.get("Approved_Upload_Status") == "Uploaded",
                                file_manager,
                            )

                            current_file_exists = self._check_original_file_exists(
                                all_originals[key]["HoleID"],
                                all_originals[key]["Depth_From"],
                                all_originals[key]["Depth_To"],
                                all_originals[key]["Original_Filename"],
                                all_originals[key].get("Approved_Upload_Status")
                                == "Uploaded",
                                file_manager,
                            )

                            if new_file_exists and not current_file_exists:
                                record["Merged_From_User"] = user_info
                                all_originals[key] = record
                    else:
                        record["Merged_From_User"] = user_info
                        all_originals[key] = record
                        stats["originals_merged"] += 1

                self._cleanup_user_file(file_path, lock_path)

            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                stats["errors"].append(f"Error in {file_path.name}: {str(e)}")

        # Save to main file
        merged_data = list(all_originals.values())
        self._write_json_file(main_path, main_lock, merged_data)
        self.logger.info(
            f"Wrote {len(merged_data)} original image records to main register"
        )

    def _merge_review_data_to_main(self, stats, main_path, main_lock):
        """Merge review data from all users to main file (keep all reviews)."""
        all_reviews = []

        # Load main file data
        if main_path.exists():
            try:
                with open(main_path, "r", encoding="utf-8") as f:
                    main_data = json.load(f)
                    if isinstance(main_data, list):
                        all_reviews.extend(main_data)
            except Exception as e:
                self.logger.error(f"Error reading main review file: {e}")

        # Process user files
        for file_path in self.get_all_user_files("review"):
            if file_path == main_path:
                continue

            lock_path = file_path.with_suffix(".json.lock")
            if lock_path.exists():
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    # Keep all reviews - they're user-specific
                    all_reviews.extend(data)
                    stats["reviews_merged"] += len(data)

                self._cleanup_user_file(file_path, lock_path)

            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")

        # Save to main file
        self._write_json_file(main_path, main_lock, all_reviews)
        self.logger.info(f"Wrote {len(all_reviews)} review records to main register")

    def _merge_corner_data_to_main(self, stats, main_path, main_lock):
        """Merge compartment corner data from all users to main file."""
        all_corners = {}

        # Load main file data
        if main_path.exists():
            try:
                with open(main_path, "r", encoding="utf-8") as f:
                    main_data = json.load(f)
                    if isinstance(main_data, list):
                        for record in main_data:
                            if isinstance(record, dict):
                                key = (
                                    record["HoleID"],
                                    record["Depth_From"],
                                    record["Depth_To"],
                                    record["Original_Filename"],
                                    record["Compartment_Number"],
                                )
                                all_corners[key] = record
            except Exception as e:
                self.logger.error(f"Error reading main corners file: {e}")

        # Process user files
        for file_path in self.get_all_user_files("corners"):
            if file_path == main_path:
                continue

            lock_path = file_path.with_suffix(".json.lock")
            if lock_path.exists():
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    for record in data:
                        key = (
                            record["HoleID"],
                            record["Depth_From"],
                            record["Depth_To"],
                            record["Original_Filename"],
                            record["Compartment_Number"],
                        )

                        if key not in all_corners:
                            all_corners[key] = record
                            stats["corners_merged"] += 1

                self._cleanup_user_file(file_path, lock_path)

            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")

        # Save to main file
        merged_data = list(all_corners.values())
        self._write_json_file(main_path, main_lock, merged_data)
        self.logger.info(f"Wrote {len(merged_data)} corner records to main register")

    def _cleanup_backup_files(self, main_paths: Dict[str, Path]) -> int:
        """Clean up backup files after successful merge."""
        cleaned = 0

        for file_type, main_path in main_paths.items():
            backup_path = main_path.with_suffix(".json.backup")
            if backup_path.exists():
                try:
                    # Verify main file is valid before removing backup
                    with open(main_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            # Main file is valid, safe to remove backup
                            backup_path.unlink()
                            cleaned += 1
                            self.logger.info(f"Removed backup file: {backup_path.name}")
                except Exception as e:
                    self.logger.warning(
                        f"Keeping backup file {backup_path.name} due to error: {e}"
                    )

        return cleaned

    def _get_main_register_paths(self) -> Dict[str, Path]:
        """Get paths to the main register files (without user suffix)."""
        return {
            "compartment": self.data_path / f"{self.COMPARTMENT_JSON_BASE}.json",
            "original": self.data_path / f"{self.ORIGINAL_JSON_BASE}.json",
            "review": self.data_path / f"{self.REVIEW_JSON_BASE}.json",
            "corners": self.data_path / f"{self.COMPARTMENT_CORNERS_JSON_BASE}.json",
        }

    def _get_main_lock_paths(self) -> Dict[str, Path]:
        """Get paths to the main register lock files."""
        main_paths = self._get_main_register_paths()
        return {key: path.with_suffix(".json.lock") for key, path in main_paths.items()}

    # Additional helper methods for aggregating data across users
    def get_all_compartment_data(self, hole_id: Optional[str] = None) -> pd.DataFrame:
        """
        Get compartment data from ALL users as a DataFrame.
        Includes a 'User_Source' column to identify which user created each record.
        """
        all_data = []

        for file_path in self.get_all_user_files("compartment"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list) and data:
                        # Extract user from filename
                        user_suffix = file_path.stem.replace(
                            self.COMPARTMENT_JSON_BASE, ""
                        )
                        if user_suffix.startswith("_"):
                            user_suffix = user_suffix[1:]

                        # Add user source to each record
                        for record in data:
                            record["User_Source"] = user_suffix

                        all_data.extend(data)
            except Exception as e:
                self.logger.error(f"Error reading {file_path}: {e}")

        df = pd.DataFrame(all_data)

        if hole_id and not df.empty:
            df = df[df["HoleID"] == hole_id]

        return df

    def get_all_original_image_data(
        self, hole_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get original image data from ALL users as a DataFrame.
        Includes a 'User_Source' column to identify which user created each record.
        """
        all_data = []

        for file_path in self.get_all_user_files("original"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list) and data:
                        # Extract user from filename
                        user_suffix = file_path.stem.replace(
                            self.ORIGINAL_JSON_BASE, ""
                        )
                        if user_suffix.startswith("_"):
                            user_suffix = user_suffix[1:]

                        # Add user source to each record
                        for record in data:
                            record["User_Source"] = user_suffix

                        all_data.extend(data)
            except Exception as e:
                self.logger.error(f"Error reading {file_path}: {e}")

        df = pd.DataFrame(all_data)

        if hole_id and not df.empty:
            df = df[df["HoleID"] == hole_id]

        return df

    # Methods to add to JSONRegisterManager class in json_register_manager.py

    def merge_all_user_registers_efficient(
        self,
        file_manager: Optional[Any] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        confirm_callback: Optional[Callable[[str, str], bool]] = None,
        checkpoint_logger: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Efficiently merge all user registers with batch operations and user confirmation.

        Args:
            file_manager: Optional FileManager for file existence checks
            progress_callback: Function(message, percentage) for progress updates
            confirm_callback: Function(title, message) that returns True/False for confirmations
            checkpoint_logger: Optional CheckpointLogger instance for detailed logging

        Returns:
            Dictionary with merge statistics
        """
        stats = {
            "compartments_merged": 0,
            "compartments_conflicts": 0,
            "originals_merged": 0,
            "originals_conflicts": 0,
            "reviews_merged": 0,
            "corners_merged": 0,
            "errors": [],
            "users_processed": set(),
        }

        try:
            # Collect all user files first
            all_user_files = {
                "compartment": self.get_all_user_files("compartment"),
                "original": self.get_all_user_files("original"),
                "review": self.get_all_user_files("review"),
                "corners": self.get_all_user_files("corners"),
            }

            # Count total users (excluding current user)
            all_users = set()
            current_user_identifier = (
                self.user_suffix[1:]
                if self.user_suffix.startswith("_")
                else self.user_suffix
            )

            for file_type, files in all_user_files.items():
                for file_path in files:
                    user_suffix = self._extract_user_from_filename(
                        file_path.stem, file_type
                    )
                    if user_suffix and user_suffix != current_user_identifier:
                        all_users.add(user_suffix)

            if not all_users:
                if progress_callback:
                    progress_callback("No other user data to merge", 100)
                self.logger.info("No other user data found to merge")
                return stats

            # Get main register paths
            main_paths = self._get_main_register_paths()

            # Process each data type with confirmation
            data_types = [
                ("compartment", "Compartment Register", self._batch_merge_compartments),
                ("original", "Original Images Register", self._batch_merge_originals),
                ("review", "Review Register", self._batch_merge_reviews),
                ("corners", "Compartment Corners", self._batch_merge_corners),
            ]

            for idx, (data_type, display_name, merge_func) in enumerate(data_types):
                # Count records to merge
                files_to_merge = [
                    f
                    for f in all_user_files[data_type]
                    if self._extract_user_from_filename(f.stem, data_type) in all_users
                ]

                if not files_to_merge:
                    continue

                # Get record counts
                total_records = 0
                user_counts = {}

                for file_path in files_to_merge:
                    user = self._extract_user_from_filename(file_path.stem, data_type)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                # Filter out example/initialization records
                                count = len(
                                    [
                                        r
                                        for r in data
                                        if isinstance(r, dict)
                                        and r.get("HoleID") != "INITIALISING"
                                    ]
                                )
                                if count > 0:
                                    user_counts[user] = count
                                    total_records += count
                    except Exception as e:
                        self.logger.warning(
                            f"Error reading {file_path} for counting: {e}"
                        )

                if total_records == 0:
                    continue

                # Ask for confirmation if callback provided
                if confirm_callback:
                    message = f"Merge {display_name}?\n\n"
                    message += f"Users with data: {len(user_counts)}\n"
                    message += f"Total records to process: {total_records}\n\n"

                    # Show first 5 users
                    for user, count in sorted(user_counts.items())[:5]:
                        message += f"  • {user}: {count} records\n"
                    if len(user_counts) > 5:
                        message += f"  • ... and {len(user_counts)-5} more users\n"

                    if not confirm_callback(f"Merge {display_name}", message):
                        self.logger.info(f"User skipped merging {display_name}")
                        continue

                # Perform merge with progress
                base_progress = idx * 25
                if progress_callback:
                    progress_callback(f"Merging {display_name}...", base_progress)

                # Execute the merge
                merge_func(
                    files_to_merge,
                    main_paths[data_type],
                    stats,
                    file_manager,
                    progress_callback,
                    base_progress,
                    checkpoint_logger,
                )

            # Final cleanup - just truncate user files without writing examples
            if progress_callback:
                progress_callback("Cleaning up user files...", 95)

            self._cleanup_all_user_files(all_user_files, stats)

            if progress_callback:
                progress_callback("Merge complete", 100)

            return stats

        except Exception as e:
            self.logger.error(f"Error in merge_all_user_registers_efficient: {e}")
            stats["errors"].append(str(e))
            return stats

    def _batch_merge_compartments(
        self,
        files_to_merge: List[Path],
        main_path: Path,
        stats: Dict[str, Any],
        file_manager: Optional[Any],
        progress_callback: Optional[Callable],
        base_progress: float,
        checkpoint_logger: Optional[Any],
    ) -> None:
        """
        Batch merge compartment data with minimal file I/O.

        Args:
            files_to_merge: List of user files to merge
            main_path: Path to main register file
            stats: Statistics dictionary to update
            file_manager: Optional FileManager for conflict resolution
            progress_callback: Optional progress callback
            base_progress: Base progress percentage
            checkpoint_logger: Optional checkpoint logger
        """
        # Get lock for main file
        main_lock = main_path.with_suffix(".json.lock")

        # Load main file data once
        all_compartments = {}
        if main_path.exists():
            try:
                if self._acquire_file_lock(main_lock):
                    try:
                        with open(main_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            for record in data:
                                if (
                                    isinstance(record, dict)
                                    and record.get("HoleID") != "INITIALISING"
                                ):
                                    key = (
                                        record["HoleID"],
                                        record["From"],
                                        record["To"],
                                    )
                                    all_compartments[key] = record
                    finally:
                        self._release_file_lock(main_lock)
                else:
                    self.logger.warning(
                        "Could not acquire lock for main compartment file"
                    )
            except Exception as e:
                self.logger.error(f"Error loading main compartment file: {e}")

        # Process all user files
        total_files = len(files_to_merge)

        for file_idx, file_path in enumerate(files_to_merge):
            if progress_callback and file_idx % 10 == 0:  # Update every 10 files
                progress = base_progress + (file_idx / total_files * 20)
                progress_callback(
                    f"Processing compartment file {file_idx+1}/{total_files}...",
                    progress,
                )

            try:
                # Skip if file is locked
                lock_path = file_path.with_suffix(".json.lock")
                if lock_path.exists():
                    self.logger.warning(f"Skipping locked file: {file_path}")
                    continue

                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if not isinstance(data, list):
                    continue

                user_info = self._extract_user_from_filename(
                    file_path.stem, "compartment"
                )

                # Process records in memory
                for record in data:
                    if (
                        not isinstance(record, dict)
                        or record.get("HoleID") == "INITIALISING"
                    ):
                        continue

                    key = (record["HoleID"], record["From"], record["To"])

                    if checkpoint_logger:
                        checkpoint_logger.log_action(
                            "Compartments processed", f"{key[0]} {key[1]}-{key[2]}m"
                        )

                    if key in all_compartments:
                        stats["compartments_conflicts"] += 1

                        # Resolve conflict
                        if file_manager:
                            existing_status = all_compartments[key].get(
                                "Photo_Status", ""
                            )
                            new_status = record.get("Photo_Status", "")

                            # Check if files exist
                            new_file_exists = self._check_compartment_file_exists(
                                record["HoleID"], record["To"], new_status, file_manager
                            )
                            current_file_exists = self._check_compartment_file_exists(
                                all_compartments[key]["HoleID"],
                                all_compartments[key]["To"],
                                existing_status,
                                file_manager,
                            )

                            # Prefer record with existing file
                            if new_file_exists and not current_file_exists:
                                record["Merged_From_User"] = user_info
                                all_compartments[key] = record
                            elif not new_file_exists and not current_file_exists:
                                # Both missing, keep most recent
                                if record.get("Processed_Date", "") > all_compartments[
                                    key
                                ].get("Processed_Date", ""):
                                    record["Merged_From_User"] = user_info
                                    all_compartments[key] = record
                        else:
                            # No file manager, keep most recent
                            if record.get("Processed_Date", "") > all_compartments[
                                key
                            ].get("Processed_Date", ""):
                                record["Merged_From_User"] = user_info
                                all_compartments[key] = record
                    else:
                        # New compartment
                        record["Merged_From_User"] = user_info
                        all_compartments[key] = record
                        stats["compartments_merged"] += 1

            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                stats["errors"].append(f"Error in {file_path.name}: {str(e)}")

        # Write merged data once
        merged_data = list(all_compartments.values())
        try:
            # Use the write method with proper locking
            self._write_json_file(
                main_path, main_path.with_suffix(".json.lock"), merged_data
            )
            self.logger.info(
                f"Wrote {len(merged_data)} compartment records to main register"
            )
        except Exception as e:
            self.logger.error(f"Error writing main compartment file: {e}")
            stats["errors"].append(f"Failed to write main compartment file: {str(e)}")

    def _batch_merge_originals(
        self,
        files_to_merge: List[Path],
        main_path: Path,
        stats: Dict[str, Any],
        file_manager: Optional[Any],
        progress_callback: Optional[Callable],
        base_progress: float,
        checkpoint_logger: Optional[Any],
    ) -> None:
        """Batch merge original image data with minimal file I/O."""
        main_lock = main_path.with_suffix(".json.lock")

        # Load main file data once
        all_originals = {}
        if main_path.exists():
            try:
                if self._acquire_file_lock(main_lock):
                    try:
                        with open(main_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            for record in data:
                                if (
                                    isinstance(record, dict)
                                    and record.get("HoleID") != "INITIALISING"
                                ):
                                    key = (
                                        record["HoleID"],
                                        record["Depth_From"],
                                        record["Depth_To"],
                                        record["Original_Filename"],
                                    )
                                    all_originals[key] = record
                    finally:
                        self._release_file_lock(main_lock)
            except Exception as e:
                self.logger.error(f"Error loading main originals file: {e}")

        # Process all user files
        total_files = len(files_to_merge)

        for file_idx, file_path in enumerate(files_to_merge):
            if progress_callback and file_idx % 10 == 0:
                progress = base_progress + (file_idx / total_files * 20)
                progress_callback(
                    f"Processing original file {file_idx+1}/{total_files}...", progress
                )

            try:
                lock_path = file_path.with_suffix(".json.lock")
                if lock_path.exists():
                    continue

                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if not isinstance(data, list):
                    continue

                user_info = self._extract_user_from_filename(file_path.stem, "original")

                for record in data:
                    if (
                        not isinstance(record, dict)
                        or record.get("HoleID") == "INITIALISING"
                    ):
                        continue

                    key = (
                        record["HoleID"],
                        record["Depth_From"],
                        record["Depth_To"],
                        record["Original_Filename"],
                    )

                    if checkpoint_logger:
                        checkpoint_logger.log_action(
                            "Originals processed", f"{key[0]} {key[1]}-{key[2]}m"
                        )

                    if key in all_originals:
                        stats["originals_conflicts"] += 1
                        # Similar conflict resolution as compartments
                    else:
                        record["Merged_From_User"] = user_info
                        all_originals[key] = record
                        stats["originals_merged"] += 1

            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                stats["errors"].append(f"Error in {file_path.name}: {str(e)}")

        # Write merged data
        merged_data = list(all_originals.values())
        try:
            self._write_json_file(
                main_path, main_path.with_suffix(".json.lock"), merged_data
            )
            self.logger.info(
                f"Wrote {len(merged_data)} original records to main register"
            )
        except Exception as e:
            self.logger.error(f"Error writing main originals file: {e}")
            stats["errors"].append(f"Failed to write main originals file: {str(e)}")

    def _batch_merge_reviews(
        self,
        files_to_merge: List[Path],
        main_path: Path,
        stats: Dict[str, Any],
        file_manager: Optional[Any],
        progress_callback: Optional[Callable],
        base_progress: float,
        checkpoint_logger: Optional[Any],
    ) -> None:
        """Batch merge review data (keep all reviews from all users)."""
        main_lock = main_path.with_suffix(".json.lock")

        # Load existing reviews
        all_reviews = []
        if main_path.exists():
            try:
                if self._acquire_file_lock(main_lock):
                    try:
                        with open(main_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                all_reviews.extend(data)
                    finally:
                        self._release_file_lock(main_lock)
            except Exception as e:
                self.logger.error(f"Error loading main reviews file: {e}")

        # Process all user files
        total_files = len(files_to_merge)

        for file_idx, file_path in enumerate(files_to_merge):
            if progress_callback and file_idx % 10 == 0:
                progress = base_progress + (file_idx / total_files * 20)
                progress_callback(
                    f"Processing review file {file_idx+1}/{total_files}...", progress
                )

            try:
                lock_path = file_path.with_suffix(".json.lock")
                if lock_path.exists():
                    continue

                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    # Keep all reviews - they're user-specific
                    review_count = 0
                    for record in data:
                        if (
                            isinstance(record, dict)
                            and record.get("HoleID") != "INITIALISING"
                        ):
                            all_reviews.append(record)
                            review_count += 1

                            if checkpoint_logger:
                                checkpoint_logger.log_action(
                                    "Reviews processed",
                                    f"{record['HoleID']} {record['From']}-{record['To']}m by {record.get('Reviewed_By', 'Unknown')}",
                                )

                    stats["reviews_merged"] += review_count

            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                stats["errors"].append(f"Error in {file_path.name}: {str(e)}")

        # Write merged reviews
        try:
            self._write_json_file(
                main_path, main_path.with_suffix(".json.lock"), all_reviews
            )
            self.logger.info(
                f"Wrote {len(all_reviews)} review records to main register"
            )
        except Exception as e:
            self.logger.error(f"Error writing main reviews file: {e}")
            stats["errors"].append(f"Failed to write main reviews file: {str(e)}")

    def _batch_merge_corners(
        self,
        files_to_merge: List[Path],
        main_path: Path,
        stats: Dict[str, Any],
        file_manager: Optional[Any],
        progress_callback: Optional[Callable],
        base_progress: float,
        checkpoint_logger: Optional[Any],
    ) -> None:
        """Batch merge compartment corner data."""
        main_lock = main_path.with_suffix(".json.lock")

        # Load existing corners
        all_corners = {}
        if main_path.exists():
            try:
                if self._acquire_file_lock(main_lock):
                    try:
                        with open(main_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            for record in data:
                                if isinstance(record, dict):
                                    key = (
                                        record["HoleID"],
                                        record["Depth_From"],
                                        record["Depth_To"],
                                        record["Original_Filename"],
                                        record["Compartment_Number"],
                                    )
                                    all_corners[key] = record
                    finally:
                        self._release_file_lock(main_lock)
            except Exception as e:
                self.logger.error(f"Error loading main corners file: {e}")

        # Process all user files
        total_files = len(files_to_merge)

        for file_idx, file_path in enumerate(files_to_merge):
            if progress_callback and file_idx % 10 == 0:
                progress = base_progress + (file_idx / total_files * 20)
                progress_callback(
                    f"Processing corners file {file_idx+1}/{total_files}...", progress
                )

            try:
                lock_path = file_path.with_suffix(".json.lock")
                if lock_path.exists():
                    continue

                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    for record in data:
                        if isinstance(record, dict):
                            key = (
                                record["HoleID"],
                                record["Depth_From"],
                                record["Depth_To"],
                                record["Original_Filename"],
                                record["Compartment_Number"],
                            )

                            if checkpoint_logger:
                                checkpoint_logger.log_action(
                                    "Corners processed", f"{key[0]} comp {key[4]}"
                                )

                            if key not in all_corners:
                                all_corners[key] = record
                                stats["corners_merged"] += 1

            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                stats["errors"].append(f"Error in {file_path.name}: {str(e)}")

        # Write merged corners
        merged_data = list(all_corners.values())
        try:
            self._write_json_file(
                main_path, main_path.with_suffix(".json.lock"), merged_data
            )
            self.logger.info(
                f"Wrote {len(merged_data)} corner records to main register"
            )
        except Exception as e:
            self.logger.error(f"Error writing main corners file: {e}")
            stats["errors"].append(f"Failed to write main corners file: {str(e)}")

    def _cleanup_all_user_files(
        self, all_user_files: Dict[str, List[Path]], stats: Dict[str, Any]
    ) -> None:
        """
        Clean up all user files efficiently without example data.
        Only cleans up files from OTHER users, not the current user.

        Args:
            all_user_files: Dictionary of file type to list of file paths
            stats: Statistics dictionary for tracking operations
        """
        current_user_identifier = (
            self.user_suffix[1:]
            if self.user_suffix.startswith("_")
            else self.user_suffix
        )

        for file_type, files in all_user_files.items():
            for file_path in files:
                # Skip current user's files
                user_suffix = self._extract_user_from_filename(
                    file_path.stem, file_type
                )
                if user_suffix == current_user_identifier:
                    continue

                # Skip main files (those without user suffix)
                if not any(suffix in str(file_path) for suffix in ["_", "-"]):
                    continue

                try:
                    lock_path = file_path.with_suffix(".json.lock")

                    # Just write empty array without examples
                    if self._acquire_file_lock(lock_path):
                        try:
                            with open(file_path, "w", encoding="utf-8") as f:
                                json.dump([], f)
                            self.logger.debug(f"Cleaned up user file: {file_path}")
                        finally:
                            self._release_file_lock(lock_path)
                    else:
                        self.logger.warning(
                            f"Could not acquire lock to clean up: {file_path}"
                        )

                except Exception as e:
                    self.logger.error(f"Error cleaning {file_path}: {e}")
                    stats["errors"].append(
                        f"Failed to clean {file_path.name}: {str(e)}"
                    )

    def _extract_user_from_filename(self, stem: str, file_type: str) -> str:
        """
        Extract user identifier from filename stem.

        Args:
            stem: File stem (filename without extension)
            file_type: Type of file ('compartment', 'original', 'review', 'corners')

        Returns:
            User identifier string
        """
        base_names = {
            "compartment": self.COMPARTMENT_JSON_BASE,
            "original": self.ORIGINAL_JSON_BASE,
            "review": self.REVIEW_JSON_BASE,
            "corners": self.COMPARTMENT_CORNERS_JSON_BASE,
        }

        base_name = base_names.get(file_type, "")
        if not base_name:
            return ""

        # Remove base name from stem
        user_part = stem.replace(base_name, "")

        # Remove leading underscore if present
        if user_part.startswith("_"):
            return user_part[1:]

        return user_part

    # ===== Transactions =====#

    def begin_transaction(self) -> Dict[str, Any]:
        """
        Begin a transaction by creating snapshots of current data.
        Returns a transaction context.
        """
        transaction = {
            "id": datetime.now().isoformat(),
            "snapshots": {},
            "locks_held": [],
            "start_time": time.time(),
            "last_refresh": time.time(),
        }

        try:
            # Create snapshots of all data files
            files_to_snapshot = [
                ("compartment", self.compartment_json_path, self.compartment_lock),
                ("original", self.original_json_path, self.original_lock),
                ("review", self.review_json_path, self.review_lock),
                ("corners", self.compartment_corners_json_path, self.corners_lock),
            ]

            for name, path, lock in files_to_snapshot:
                if path.exists():
                    # Acquire lock
                    if self._acquire_file_lock(lock):
                        transaction["locks_held"].append((name, lock))

                        # Read current data
                        with open(path, "r", encoding="utf-8") as f:
                            transaction["snapshots"][name] = json.load(f)
                    else:
                        # Failed to acquire lock - rollback
                        self.rollback_transaction(transaction)
                        raise RuntimeError(f"Could not acquire lock for {name}")

            self.logger.info(f"Transaction {transaction['id']} started")
            return transaction

        except Exception as e:
            self.logger.error(f"Failed to begin transaction: {e}")
            self.rollback_transaction(transaction)
            raise

    def commit_transaction(self, transaction: Dict[str, Any]) -> None:
        """Commit a transaction by releasing locks."""
        try:
            # Release all locks
            for name, lock in transaction["locks_held"]:
                self._release_file_lock(lock)

            self.logger.info(f"Transaction {transaction['id']} committed")

        except Exception as e:
            self.logger.error(f"Error committing transaction: {e}")
            raise

    def rollback_transaction(self, transaction: Dict[str, Any]) -> None:
        """Rollback a transaction by restoring snapshots."""
        try:
            # Restore snapshots
            file_mappings = {
                "compartment": self.compartment_json_path,
                "original": self.original_json_path,
                "review": self.review_json_path,
                "corners": self.compartment_corners_json_path,
            }

            for name, snapshot_data in transaction["snapshots"].items():
                if name in file_mappings:
                    path = file_mappings[name]
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(
                            snapshot_data, f, indent=2, ensure_ascii=False, default=str
                        )
                    self.logger.info(f"Rolled back {name} data")

            # Release all locks
            for name, lock in transaction["locks_held"]:
                self._release_file_lock(lock)

            self.logger.info(f"Transaction {transaction['id']} rolled back")

        except Exception as e:
            self.logger.error(f"Error during rollback: {e}")
            # Still try to release locks
            for name, lock in transaction.get("locks_held", []):
                try:
                    self._release_file_lock(lock)
                except:
                    pass
