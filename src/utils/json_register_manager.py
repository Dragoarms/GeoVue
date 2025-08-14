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
from dataclasses import dataclass
from enum import Enum
from openpyxl import workbook, load_workbook
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.utils import get_column_letter
import structlog

try:
    import win32com.client
except ImportError:
    win32com = None


# ==================== CUSTOM EXCEPTIONS ====================

class JSONRegisterError(Exception):
    """Base exception for all JSON register operations."""
    pass


class FileOperationError(JSONRegisterError):
    """Raised when file operations fail."""
    def __init__(self, message: str, file_path: Optional[str] = None, operation: Optional[str] = None):
        super().__init__(message)
        self.file_path = file_path
        self.operation = operation


class DataValidationError(JSONRegisterError):
    """Raised when data validation fails."""
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        super().__init__(message)
        self.field = field
        self.value = value


class RecordNotFoundError(JSONRegisterError):
    """Raised when a requested record cannot be found."""
    def __init__(self, message: str, record_key: Optional[str] = None):
        super().__init__(message)
        self.record_key = record_key


class LockAcquisitionError(JSONRegisterError):
    """Raised when file locks cannot be acquired."""
    def __init__(self, message: str, lock_path: Optional[str] = None, timeout: Optional[int] = None):
        super().__init__(message)
        self.lock_path = lock_path
        self.timeout = timeout


class DataCorruptionError(JSONRegisterError):
    """Raised when data corruption is detected."""
    def __init__(self, message: str, file_path: Optional[str] = None, backup_available: bool = False):
        super().__init__(message)
        self.file_path = file_path
        self.backup_available = backup_available


class ExternalDependencyError(JSONRegisterError):
    """Raised when external dependencies fail."""
    def __init__(self, message: str, dependency: Optional[str] = None, retry_after: Optional[int] = None):
        super().__init__(message)
        self.dependency = dependency
        self.retry_after = retry_after


# ==================== UTILITY FUNCTIONS ====================

def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple = (FileOperationError, LockAcquisitionError)
) -> Callable:
    """Decorator to retry operations with exponential backoff."""
    def wrapper(*args, **kwargs):
        delay = base_delay
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                last_exception = e
                if attempt == max_retries:
                    break
                
                # Log retry attempt
                if hasattr(args[0], 'logger'):
                    args[0].logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                
                time.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
        
        raise last_exception
    return wrapper


def setup_structured_logger(name: str, base_path: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """Setup structured logging with JSON output."""
    if base_path:
        log_file = Path(base_path) / "logs" / f"{name}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure file handler with JSON formatting
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Setup Python logging
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=logging.INFO,
            handlers=[file_handler]
        )
    
    return structlog.get_logger(name)


class PhotoStatus(Enum):
    """Valid photo status values."""
    FOR_REVIEW = "For Review"
    APPROVED = "Approved"
    REJECTED = "Rejected"
    IN_PROGRESS = "In Progress"


@dataclass
class CompartmentProcessingMetadata:
    """Data model for compartment processing metadata fields."""
    photo_status: str
    processed_date: str
    processed_by: str
    comments: Optional[str] = None
    image_width_cm: Optional[float] = None

    @classmethod
    def create_default(cls, photo_status: str = PhotoStatus.FOR_REVIEW.value, 
                      processed_by: Optional[str] = None) -> 'CompartmentProcessingMetadata':
        """Create default processing metadata with current timestamp."""
        valid_statuses = {status.value for status in PhotoStatus}
        if photo_status not in valid_statuses:
            raise ValueError(f"Invalid photo_status: {photo_status}. Must be one of {valid_statuses}")
        
        timestamp = datetime.now().isoformat()
        username = processed_by or os.getenv("USERNAME", "Unknown")
        return cls(
            photo_status=photo_status,
            processed_date=timestamp,
            processed_by=username
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "Photo_Status": self.photo_status,
            "Processed_Date": self.processed_date,
            "Processed_By": self.processed_by,
            "Comments": self.comments,
            "Image_Width_Cm": self.image_width_cm
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompartmentProcessingMetadata':
        """Create instance from dictionary data."""
        defaults = cls.create_default(
            photo_status=data.get("Photo_Status", PhotoStatus.FOR_REVIEW.value),
            processed_by=data.get("Processed_By")
        )
        
        return cls(
            photo_status=data.get("Photo_Status", defaults.photo_status),
            processed_date=data.get("Processed_Date", defaults.processed_date),
            processed_by=data.get("Processed_By", defaults.processed_by),
            comments=data.get("Comments"),
            image_width_cm=data.get("Image_Width_Cm")
        )

    def validate(self) -> None:
        """Validate processing metadata fields."""
        if not self.photo_status:
            raise DataValidationError("photo_status cannot be empty", field="photo_status", value=self.photo_status)
        if not self.processed_by:
            raise DataValidationError("processed_by cannot be empty", field="processed_by", value=self.processed_by)
        
        valid_statuses = {status.value for status in PhotoStatus}
        if self.photo_status not in valid_statuses:
            raise DataValidationError(
                f"Invalid photo_status: {self.photo_status}. Must be one of {valid_statuses}",
                field="photo_status",
                value=self.photo_status
            )
        
        # Validate processed_date format
        try:
            datetime.fromisoformat(self.processed_date.replace('Z', '+00:00'))
        except (ValueError, AttributeError) as e:
            raise DataValidationError(
                f"Invalid processed_date format: {self.processed_date}. Expected ISO format.",
                field="processed_date",
                value=self.processed_date
            ) from e
        
        # Validate image_width_cm if provided
        if self.image_width_cm is not None:
            if not isinstance(self.image_width_cm, (int, float)) or self.image_width_cm <= 0:
                raise DataValidationError(
                    f"image_width_cm must be a positive number, got: {self.image_width_cm}",
                    field="image_width_cm",
                    value=self.image_width_cm
                )


@dataclass
class CornerRecordConfig:
    """Configuration for creating corner records."""
    hole_id: str
    depth_from: int
    depth_to: int
    original_filename: str
    compartment_num: int
    corners: List[List[int]]
    processing_metadata: Optional[CompartmentProcessingMetadata] = None
    scale_px_per_cm: Optional[float] = None
    scale_confidence: Optional[float] = None
    source_image_uid: Optional[str] = None

    def validate(self) -> None:
        """Validate corner record configuration."""
        if not self.hole_id or not isinstance(self.hole_id, str):
            raise DataValidationError("hole_id must be a non-empty string", field="hole_id", value=self.hole_id)
        
        if not isinstance(self.depth_from, int) or self.depth_from < 0:
            raise DataValidationError("depth_from must be a non-negative integer", field="depth_from", value=self.depth_from)
        
        if not isinstance(self.depth_to, int) or self.depth_to < 0:
            raise DataValidationError("depth_to must be a non-negative integer", field="depth_to", value=self.depth_to)
        
        if self.depth_to <= self.depth_from:
            raise DataValidationError(
                f"depth_to ({self.depth_to}) must be greater than depth_from ({self.depth_from})",
                field="depth_to",
                value=self.depth_to
            )
        
        if not self.original_filename or not isinstance(self.original_filename, str):
            raise DataValidationError("original_filename must be a non-empty string", field="original_filename", value=self.original_filename)
        
        if not isinstance(self.compartment_num, int) or self.compartment_num < 1:
            raise DataValidationError("compartment_num must be a positive integer", field="compartment_num", value=self.compartment_num)
        
        # Validate corners format
        if not isinstance(self.corners, list) or len(self.corners) != 4:
            raise DataValidationError("corners must be a list of 4 coordinate pairs", field="corners", value=self.corners)
        
        for i, corner in enumerate(self.corners):
            if not isinstance(corner, list) or len(corner) != 2:
                raise DataValidationError(f"corner {i} must be a list of 2 coordinates", field=f"corners[{i}]", value=corner)
            if not all(isinstance(coord, int) for coord in corner):
                raise DataValidationError(f"corner {i} coordinates must be integers", field=f"corners[{i}]", value=corner)
        
        # Validate optional fields
        if self.scale_px_per_cm is not None:
            if not isinstance(self.scale_px_per_cm, (int, float)) or self.scale_px_per_cm <= 0:
                raise DataValidationError("scale_px_per_cm must be a positive number", field="scale_px_per_cm", value=self.scale_px_per_cm)
        
        if self.scale_confidence is not None:
            if not isinstance(self.scale_confidence, (int, float)) or not (0 <= self.scale_confidence <= 1):
                raise DataValidationError("scale_confidence must be a number between 0 and 1", field="scale_confidence", value=self.scale_confidence)
        
        if self.processing_metadata is not None:
            try:
                self.processing_metadata.validate()
            except DataValidationError as e:
                raise DataValidationError(f"processing_metadata validation failed: {e}", field="processing_metadata", value=self.processing_metadata) from e


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
        # Use structured logger if none provided
        if logger is None:
            try:
                self.logger = setup_structured_logger("json_register_manager", str(base_path))
            except Exception:
                # Fallback to regular logger if structured logging fails
                self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

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

    @retry_with_backoff
    def _read_json_file(self, file_path: Path, lock_path: Path) -> List[Dict]:
        """Read JSON file with locking and error handling."""
        if lock_path and not self._acquire_file_lock(lock_path):
            raise LockAcquisitionError(
                f"Could not acquire lock for {file_path}",
                lock_path=str(lock_path),
                timeout=30
            )

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

    @retry_with_backoff
    def _write_json_file(
        self, file_path: Path, lock_path: Path, data: List[Dict]
    ) -> None:
        """Write JSON file with locking, backup, and validation."""
        if lock_path and not self._acquire_file_lock(lock_path):
            raise LockAcquisitionError(
                f"Could not acquire lock for {file_path}",
                lock_path=str(lock_path),
                timeout=30
            )

        try:
            # Validate input data
            if not isinstance(data, list):
                raise DataValidationError(
                    f"Data must be a list, got {type(data).__name__}",
                    field="data",
                    value=type(data)
                )
            
            # Validate list contents if not empty
            if data:
                for i, item in enumerate(data):
                    if not isinstance(item, dict):
                        raise DataValidationError(
                            f"List item {i} must be a dict, got {type(item).__name__}",
                            field=f"data[{i}]",
                            value=type(item)
                        )

            # Convert Path if needed
            if not isinstance(file_path, Path):
                file_path = Path(file_path)

            # Create parent directory if needed
            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError) as e:
                raise FileOperationError(
                    f"Cannot create directory: {str(e)}",
                    file_path=str(file_path.parent),
                    operation="mkdir"
                ) from e

            # Create backup if file exists and is valid
            backup_created = False
            if file_path.exists():
                try:
                    file_stats = file_path.stat()
                    if file_stats.st_size > 0:
                        backup_path = file_path.with_suffix(".json.backup")
                        
                        # Verify existing file is valid JSON before backing up
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                existing_data = json.load(f)
                            
                            # Create backup
                            shutil.copy2(file_path, backup_path)
                            backup_created = True
                            
                            self.logger.debug(
                                "Created backup before write",
                                file_path=str(file_path),
                                backup_path=str(backup_path),
                                original_size=file_stats.st_size
                            )
                            
                        except json.JSONDecodeError:
                            self.logger.warning(
                                "Existing file appears corrupted, proceeding without backup",
                                file_path=str(file_path)
                            )
                        except Exception as backup_error:
                            self.logger.warning(
                                "Could not create backup file",
                                file_path=str(file_path),
                                backup_error=str(backup_error)
                            )
                            
                except (OSError, PermissionError) as e:
                    self.logger.warning(
                        "Cannot access existing file for backup",
                        file_path=str(file_path),
                        access_error=str(e)
                    )

            # Write to temporary file with atomic operation
            temp_path = file_path.with_suffix(".json.tmp")
            
            try:
                # Write data to temporary file
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(
                        data,
                        f,
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                        separators=(",", ": "),
                    )

                # Verify the temporary file is valid JSON
                try:
                    with open(temp_path, "r", encoding="utf-8") as f:
                        verified_data = json.load(f)
                    
                    # Validate structure matches expected
                    if not isinstance(verified_data, list):
                        raise DataCorruptionError(
                            "Temporary file validation failed: not a list",
                            file_path=str(temp_path)
                        )
                    
                    if len(verified_data) != len(data):
                        raise DataCorruptionError(
                            f"Temporary file validation failed: length mismatch {len(verified_data)} != {len(data)}",
                            file_path=str(temp_path)
                        )
                        
                except json.JSONDecodeError as e:
                    raise DataCorruptionError(
                        f"Temporary file validation failed: {str(e)}",
                        file_path=str(temp_path)
                    ) from e

                # Atomic move from temp to final location
                try:
                    temp_path.replace(file_path)
                except (OSError, PermissionError) as e:
                    raise FileOperationError(
                        f"Cannot move temporary file to final location: {str(e)}",
                        file_path=str(file_path),
                        operation="atomic_move"
                    ) from e
                
                self.logger.debug(
                    f"Successfully wrote JSON file: {file_path} (records: {len(data)}, backup: {backup_created})"
                )

            except Exception as temp_error:
                # Clean up temporary file on any error
                try:
                    if temp_path.exists():
                        temp_path.unlink()
                except Exception as cleanup_error:
                    self.logger.warning(
                        f"Could not clean up temporary file {temp_path}: {cleanup_error}"
                    )
                
                # Re-raise the original error
                raise temp_error

        except (DataValidationError, DataCorruptionError, FileOperationError, LockAcquisitionError):
            # Re-raise specific exceptions
            raise
        except (OSError, IOError, PermissionError) as e:
            raise FileOperationError(
                f"File system error during write: {str(e)}",
                file_path=str(file_path),
                operation="write"
            ) from e
        except Exception as e:
            self.logger.error(
                f"Unexpected error writing JSON file {file_path}: {type(e).__name__}: {str(e)}"
            )
            raise FileOperationError(
                f"Unexpected error writing JSON file: {str(e)}",
                file_path=str(file_path),
                operation="write"
            ) from e

        finally:
            if lock_path:
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

                        # Add new corners using modular function
                        for comp_num, corners in compartment_data.items():
                            if isinstance(corners, list) and len(corners) == 4:
                                # Create default processing metadata for new corners
                                processing_metadata = CompartmentProcessingMetadata.create_default("For Review")
                                
                                corner_record = self._create_corner_record_with_processing(
                                    hole_id=hole_id,
                                    depth_from=depth_from,
                                    depth_to=depth_to,
                                    original_filename=original_filename,
                                    compartment_num=int(comp_num),
                                    corners=corners,
                                    processing_metadata=processing_metadata,
                                    scale_px_per_cm=scale_px_per_cm,
                                    scale_confidence=scale_confidence,
                                    source_image_uid=uid
                                )

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

    # === Modular Corner Processing Functions ===
    
    def _ensure_corner_processing_fields(self, corner_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure corner record has all required processing metadata fields with defaults.
        Maintains backward compatibility with existing corner records.
        
        Args:
            corner_record: Existing corner record dictionary
            
        Returns:
            Corner record with all processing fields ensured
        """
        defaults = CompartmentProcessingMetadata.create_default("For Review").to_dict()
        
        # Add missing processing fields with defaults
        for field, default_value in defaults.items():
            if field not in corner_record:
                corner_record[field] = default_value
                
        return corner_record
    
    def _create_corner_record_with_processing(
        self,
        hole_id: str,
        depth_from: int,
        depth_to: int,
        original_filename: str,
        compartment_num: int,
        corners: List[List[int]],
        processing_metadata: Optional[CompartmentProcessingMetadata] = None,
        scale_px_per_cm: Optional[float] = None,
        scale_confidence: Optional[float] = None,
        source_image_uid: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a complete corner record with coordinate and processing data.
        
        Args:
            hole_id: Hole identifier
            depth_from: Starting depth
            depth_to: Ending depth
            original_filename: Original image filename
            compartment_num: Compartment number
            corners: Corner coordinates [[TL], [TR], [BR], [BL]]
            processing_metadata: Processing metadata or None for defaults
            scale_px_per_cm: Scale in pixels per cm
            scale_confidence: Scale detection confidence
            source_image_uid: Source image UID
            
        Returns:
            Complete corner record dictionary
        """
        if processing_metadata is None:
            processing_metadata = CompartmentProcessingMetadata.create_default("For Review")
            
        # Core coordinate data
        corner_record = {
            "HoleID": hole_id,
            "Depth_From": depth_from,
            "Depth_To": depth_to,
            "Original_Filename": original_filename,
            "Compartment_Number": compartment_num,
            "Top_Left_X": corners[0][0],
            "Top_Left_Y": corners[0][1],
            "Top_Right_X": corners[1][0],
            "Top_Right_Y": corners[1][1],
            "Bottom_Right_X": corners[2][0],
            "Bottom_Right_Y": corners[2][1],
            "Bottom_Left_X": corners[3][0],
            "Bottom_Left_Y": corners[3][1],
        }
        
        # Add processing metadata
        corner_record.update(processing_metadata.to_dict())
        
        # Add optional scale data
        if scale_px_per_cm is not None:
            corner_record["Scale_PxPerCm"] = scale_px_per_cm
        if scale_confidence is not None:
            corner_record["Scale_Confidence"] = scale_confidence
        if source_image_uid is not None:
            corner_record["Source_Image_UID"] = source_image_uid
            
        return corner_record
    
    def update_corner_processing_metadata(
        self,
        hole_id: str,
        depth_from: int,
        depth_to: int,
        original_filename: str,
        compartment_num: int,
        processing_metadata: CompartmentProcessingMetadata
    ) -> bool:
        """
        Update processing metadata for a specific corner record.
        Preserves coordinate and scale data while updating processing fields.
        
        Args:
            hole_id: Hole identifier
            depth_from: Starting depth
            depth_to: Ending depth
            original_filename: Original image filename
            compartment_num: Compartment number
            processing_metadata: New processing metadata
            
        Returns:
            True if update successful
            
        Raises:
            DataValidationError: If input parameters are invalid
            RecordNotFoundError: If the specified record cannot be found
            FileOperationError: If file operations fail
        """
        operation_id = str(uuid.uuid4())[:8]
        record_key = f"{hole_id}_{depth_from}-{depth_to}_comp{compartment_num}"
        
        # Structured logging with context
        log_context = {
            "operation": "update_corner_processing_metadata",
            "operation_id": operation_id,
            "hole_id": hole_id,
            "depth_from": depth_from,
            "depth_to": depth_to,
            "original_filename": original_filename,
            "compartment_num": compartment_num,
            "record_key": record_key
        }
        
        self.logger.info("Starting corner processing metadata update", **log_context)
        
        # Input validation
        try:
            if not hole_id or not isinstance(hole_id, str):
                raise DataValidationError("hole_id must be a non-empty string", field="hole_id", value=hole_id)
            
            depth_from = int(depth_from)
            depth_to = int(depth_to)
            compartment_num = int(compartment_num)
            
            if depth_from < 0 or depth_to < 0:
                raise DataValidationError("Depth values must be non-negative", field="depth", value=(depth_from, depth_to))
            
            if not original_filename or not isinstance(original_filename, str):
                raise DataValidationError("original_filename must be a non-empty string", field="original_filename", value=original_filename)
                
            # Validate processing metadata
            processing_metadata.validate()
            
        except (ValueError, TypeError) as e:
            raise DataValidationError(f"Invalid input parameters: {str(e)}", field="input_validation") from e
        
        with self._thread_lock:
            try:
                # Read current data with error handling
                data = self._read_json_file(
                    self.compartment_corners_json_path, self.corners_lock
                )
                
                if not isinstance(data, list):
                    raise DataCorruptionError(
                        f"Expected list data structure, got {type(data)}",
                        file_path=str(self.compartment_corners_json_path)
                    )
                
                # Create backup before modification
                original_data = [record.copy() for record in data]
                
                # Find and update the record
                target_record = None
                record_index = None
                
                for i, record in enumerate(data):
                    if (
                        record.get("HoleID") == hole_id
                        and record.get("Depth_From") == depth_from
                        and record.get("Depth_To") == depth_to
                        and record.get("Original_Filename") == original_filename
                        and record.get("Compartment_Number") == compartment_num
                    ):
                        target_record = record
                        record_index = i
                        break
                
                if target_record is None:
                    raise RecordNotFoundError(
                        f"Corner record not found: {record_key}",
                        record_key=record_key
                    )
                
                # Preserve existing coordinate data before update
                preserved_fields = {}
                coordinate_fields = [
                    "Top_Left_X", "Top_Left_Y", "Top_Right_X", "Top_Right_Y",
                    "Bottom_Right_X", "Bottom_Right_Y", "Bottom_Left_X", "Bottom_Left_Y"
                ]
                for field in coordinate_fields:
                    if field in target_record:
                        preserved_fields[field] = target_record[field]
                
                # Update processing metadata
                data[record_index].update(processing_metadata.to_dict())
                
                # Restore preserved coordinate fields
                data[record_index].update(preserved_fields)
                
                # Write updated data with rollback capability
                try:
                    self._write_json_file(
                        self.compartment_corners_json_path, self.corners_lock, data
                    )
                    
                    self.logger.info(
                        "Successfully updated corner processing metadata",
                        record_updated=True,
                        **log_context
                    )
                    return True
                    
                except (OSError, IOError, PermissionError) as e:
                    # Rollback data on write failure
                    try:
                        self._write_json_file(
                            self.compartment_corners_json_path, self.corners_lock, original_data
                        )
                        self.logger.warning("Rolled back data after write failure", **log_context)
                    except Exception as rollback_error:
                        self.logger.error(
                            "Critical: Failed to rollback data after write failure",
                            rollback_error=str(rollback_error),
                            **log_context
                        )
                    
                    raise FileOperationError(
                        f"Failed to write updated corner data: {str(e)}",
                        file_path=str(self.compartment_corners_json_path),
                        operation="write"
                    ) from e
                    
            except (DataCorruptionError, RecordNotFoundError, FileOperationError):
                # Re-raise specific exceptions with context
                raise
            except json.JSONDecodeError as e:
                raise DataCorruptionError(
                    f"JSON data corruption detected: {str(e)}",
                    file_path=str(self.compartment_corners_json_path)
                ) from e
            except Exception as e:
                self.logger.error(
                    "Unexpected error during corner processing metadata update",
                    error=str(e),
                    error_type=type(e).__name__,
                    **log_context
                )
                raise FileOperationError(
                    f"Unexpected error updating corner processing metadata: {str(e)}",
                    operation="update_corner_processing_metadata"
                ) from e
    
    def batch_update_corner_processing_metadata(
        self,
        updates: List[Dict[str, Any]]
    ) -> int:
        """
        Batch update processing metadata for multiple corner records.
        
        Args:
            updates: List of update dictionaries with keys:
                - hole_id, depth_from, depth_to, original_filename, compartment_num
                - processing_metadata: CompartmentProcessingMetadata instance
                
        Returns:
            Number of successfully updated records
            
        Raises:
            DataValidationError: If update data is invalid
            FileOperationError: If file operations fail
        """
        operation_id = str(uuid.uuid4())[:8]
        
        log_context = {
            "operation": "batch_update_corner_processing_metadata", 
            "operation_id": operation_id,
            "batch_size": len(updates) if updates else 0
        }
        
        self.logger.info("Starting batch corner processing metadata update", **log_context)
        
        # Input validation
        if not isinstance(updates, list):
            raise DataValidationError("updates must be a list", field="updates", value=type(updates))
        
        if not updates:
            self.logger.warning("Empty updates list provided", **log_context)
            return 0
        
        successful_updates = 0
        failed_updates = []
        
        # Validate all updates before processing
        validated_updates = []
        for i, update in enumerate(updates):
            try:
                # Validate update structure
                required_keys = ["hole_id", "depth_from", "depth_to", "original_filename", "compartment_num", "processing_metadata"]
                missing_keys = [key for key in required_keys if key not in update]
                if missing_keys:
                    raise DataValidationError(f"Missing required keys: {missing_keys}", field=f"updates[{i}]")
                
                # Validate and normalize data types
                normalized_update = {
                    "hole_id": str(update["hole_id"]),
                    "depth_from": int(update["depth_from"]),
                    "depth_to": int(update["depth_to"]), 
                    "original_filename": str(update["original_filename"]),
                    "compartment_num": int(update["compartment_num"]),
                    "processing_metadata": update["processing_metadata"]
                }
                
                # Validate processing metadata
                if not isinstance(normalized_update["processing_metadata"], CompartmentProcessingMetadata):
                    raise DataValidationError(
                        f"processing_metadata must be CompartmentProcessingMetadata instance",
                        field=f"updates[{i}].processing_metadata"
                    )
                
                normalized_update["processing_metadata"].validate()
                validated_updates.append(normalized_update)
                
            except (ValueError, TypeError, DataValidationError) as e:
                failed_updates.append({
                    "index": i,
                    "update": update,
                    "error": str(e)
                })
                self.logger.warning(
                    f"Validation failed for update {i}",
                    update_index=i,
                    validation_error=str(e),
                    **log_context
                )
        
        if not validated_updates:
            raise DataValidationError(
                f"No valid updates found. {len(failed_updates)} validation failures.",
                field="updates"
            )
        
        with self._thread_lock:
            transaction_data = None
            try:
                # Read current data with error handling
                data = self._read_json_file(
                    self.compartment_corners_json_path, self.corners_lock
                )
                
                if not isinstance(data, list):
                    raise DataCorruptionError(
                        f"Expected list data structure, got {type(data)}",
                        file_path=str(self.compartment_corners_json_path)
                    )
                
                # Create backup for rollback
                transaction_data = [record.copy() for record in data]
                
                # Create lookup for efficient updates
                record_lookup = {}
                for i, record in enumerate(data):
                    try:
                        key = (
                            record.get("HoleID"),
                            record.get("Depth_From"),
                            record.get("Depth_To"),
                            record.get("Original_Filename"),
                            record.get("Compartment_Number")
                        )
                        record_lookup[key] = i
                    except KeyError as e:
                        self.logger.warning(
                            f"Skipping malformed record {i}: missing key {e}",
                            record_index=i,
                            **log_context
                        )
                
                # Apply validated updates with individual error handling
                for update in validated_updates:
                    try:
                        key = (
                            update["hole_id"],
                            update["depth_from"],
                            update["depth_to"],
                            update["original_filename"],
                            update["compartment_num"]
                        )
                        
                        if key in record_lookup:
                            record_index = record_lookup[key]
                            
                            # Preserve coordinate data before update
                            coordinate_fields = [
                                "Top_Left_X", "Top_Left_Y", "Top_Right_X", "Top_Right_Y",
                                "Bottom_Right_X", "Bottom_Right_Y", "Bottom_Left_X", "Bottom_Left_Y"
                            ]
                            preserved_fields = {
                                field: data[record_index].get(field)
                                for field in coordinate_fields
                                if field in data[record_index]
                            }
                            
                            # Apply update
                            data[record_index].update(update["processing_metadata"].to_dict())
                            
                            # Restore preserved fields
                            data[record_index].update(preserved_fields)
                            
                            successful_updates += 1
                        else:
                            failed_updates.append({
                                "update": update,
                                "error": f"Record not found: {key}"
                            })
                            
                    except Exception as e:
                        failed_updates.append({
                            "update": update,
                            "error": f"Update failed: {str(e)}"
                        })
                        self.logger.warning(
                            f"Individual update failed",
                            update_key=key,
                            update_error=str(e),
                            **log_context
                        )
                
                # Write updates if any were successful
                if successful_updates > 0:
                    try:
                        self._write_json_file(
                            self.compartment_corners_json_path, self.corners_lock, data
                        )
                        
                        self.logger.info(
                            "Batch update completed successfully",
                            successful_updates=successful_updates,
                            failed_updates=len(failed_updates),
                            **log_context
                        )
                        
                    except (OSError, IOError, PermissionError) as e:
                        # Rollback on write failure
                        if transaction_data:
                            try:
                                self._write_json_file(
                                    self.compartment_corners_json_path, self.corners_lock, transaction_data
                                )
                                self.logger.warning("Rolled back batch update after write failure", **log_context)
                            except Exception as rollback_error:
                                self.logger.error(
                                    "Critical: Failed to rollback batch update",
                                    rollback_error=str(rollback_error),
                                    **log_context
                                )
                        
                        raise FileOperationError(
                            f"Failed to write batch corner updates: {str(e)}",
                            file_path=str(self.compartment_corners_json_path),
                            operation="batch_write"
                        ) from e
                
                # Log any failed updates
                if failed_updates:
                    self.logger.warning(
                        f"Batch update had {len(failed_updates)} failures",
                        failed_updates=failed_updates,
                        **log_context
                    )
                
                return successful_updates
                
            except (DataCorruptionError, FileOperationError):
                # Re-raise specific exceptions
                raise
            except json.JSONDecodeError as e:
                raise DataCorruptionError(
                    f"JSON data corruption during batch update: {str(e)}",
                    file_path=str(self.compartment_corners_json_path)
                ) from e
            except Exception as e:
                self.logger.error(
                    "Unexpected error during batch corner processing update",
                    error=str(e),
                    error_type=type(e).__name__,
                    successful_updates=successful_updates,
                    **log_context
                )
                raise FileOperationError(
                    f"Unexpected error in batch corner processing update: {str(e)}",
                    operation="batch_update_corner_processing_metadata"
                ) from e
    
    def get_corner_processing_metadata(
        self,
        hole_id: str,
        depth_from: int,
        depth_to: int,
        original_filename: str,
        compartment_num: int
    ) -> Optional[CompartmentProcessingMetadata]:
        """
        Get processing metadata for a specific corner record.
        
        Args:
            hole_id: Hole identifier
            depth_from: Starting depth
            depth_to: Ending depth
            original_filename: Original image filename
            compartment_num: Compartment number
            
        Returns:
            CompartmentProcessingMetadata instance or None if not found
        """
        with self._thread_lock:
            try:
                data = self._read_json_file(
                    self.compartment_corners_json_path, self.corners_lock
                )
                
                depth_from = int(depth_from)
                depth_to = int(depth_to)
                compartment_num = int(compartment_num)
                
                for record in data:
                    if (
                        record["HoleID"] == hole_id
                        and record["Depth_From"] == depth_from
                        and record["Depth_To"] == depth_to
                        and record["Original_Filename"] == original_filename
                        and record["Compartment_Number"] == compartment_num
                    ):
                        return CompartmentProcessingMetadata.from_dict(record)
                
                return None
                
            except Exception as e:
                self.logger.error(f"Error getting corner processing metadata: {e}")
                return None

    def validate_corner_data_consistency(self) -> Dict[str, Any]:
        """
        Validate consistency between corner coordinates and processing metadata.
        
        Returns:
            Dictionary with validation results and statistics
        """
        validation_results = {
            "total_records": 0,
            "records_with_coordinates": 0,
            "records_with_processing_metadata": 0,
            "records_missing_processing_fields": 0,
            "consistency_issues": [],
            "validation_passed": True
        }
        
        with self._thread_lock:
            try:
                data = self._read_json_file(
                    self.compartment_corners_json_path, self.corners_lock
                )
                
                required_coordinate_fields = [
                    "Top_Left_X", "Top_Left_Y", "Top_Right_X", "Top_Right_Y",
                    "Bottom_Right_X", "Bottom_Right_Y", "Bottom_Left_X", "Bottom_Left_Y"
                ]
                
                required_processing_fields = [
                    "Photo_Status", "Processed_Date", "Processed_By", "Comments", "Image_Width_Cm"
                ]
                
                validation_results["total_records"] = len(data)
                
                for i, record in enumerate(data):
                    record_id = f"{record.get('HoleID', 'Unknown')}_{record.get('Depth_From', 'Unknown')}-{record.get('Depth_To', 'Unknown')}_comp{record.get('Compartment_Number', 'Unknown')}"
                    
                    # Check coordinate fields
                    has_coordinates = all(field in record for field in required_coordinate_fields)
                    if has_coordinates:
                        validation_results["records_with_coordinates"] += 1
                    else:
                        missing_coord_fields = [field for field in required_coordinate_fields if field not in record]
                        validation_results["consistency_issues"].append({
                            "record_id": record_id,
                            "issue": "missing_coordinate_fields",
                            "missing_fields": missing_coord_fields
                        })
                        validation_results["validation_passed"] = False
                    
                    # Check processing metadata fields
                    has_processing = all(field in record for field in required_processing_fields)
                    if has_processing:
                        validation_results["records_with_processing_metadata"] += 1
                    else:
                        missing_proc_fields = [field for field in required_processing_fields if field not in record]
                        validation_results["records_missing_processing_fields"] += 1
                        validation_results["consistency_issues"].append({
                            "record_id": record_id,
                            "issue": "missing_processing_fields",
                            "missing_fields": missing_proc_fields
                        })
                
                # Log validation summary
                self.logger.info(
                    f"Corner data validation: {validation_results['total_records']} total records, "
                    f"{validation_results['records_with_coordinates']} with coordinates, "
                    f"{validation_results['records_with_processing_metadata']} with processing metadata, "
                    f"{validation_results['records_missing_processing_fields']} missing processing fields"
                )
                
                if not validation_results["validation_passed"]:
                    self.logger.warning(f"Corner data validation found {len(validation_results['consistency_issues'])} issues")
                
                return validation_results
                
            except Exception as e:
                self.logger.error(f"Error during corner data validation: {e}")
                validation_results["validation_passed"] = False
                validation_results["consistency_issues"].append({
                    "record_id": "VALIDATION_ERROR",
                    "issue": "validation_exception",
                    "error": str(e)
                })
                return validation_results

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

        # Load current data with backward compatibility
        current_data = self._read_json_file(
            self.compartment_corners_json_path, self.corners_lock
        )
        for record in current_data:
            # Ensure backward compatibility by adding missing processing fields
            record = self._ensure_corner_processing_fields(record)
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
                        # Ensure backward compatibility for merged records
                        record = self._ensure_corner_processing_fields(record)
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
                                # Always ensure UID is present when available
                                if source_uid:
                                    comp_records[key]["Source_Image_UID"] = source_uid
                            else:
                                # Create new record
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
                                # Always include UID when available
                                if source_uid:
                                    new_record["Source_Image_UID"] = source_uid

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

    def cleanup_placeholder_records(self) -> Dict[str, int]:
        """
        Remove any placeholder INITIALISING records from all register files.
        
        Returns:
            Dict with counts of records removed per file type
            
        Raises:
            FileOperationError: If critical file operations fail
        """
        operation_id = str(uuid.uuid4())[:8]
        
        cleanup_stats = {
            "compartments": 0,
            "originals": 0, 
            "reviews": 0,
            "corners": 0,
            "errors": []
        }
        
        log_context = {
            "operation": "cleanup_placeholder_records",
            "operation_id": operation_id
        }
        
        self.logger.info("Starting placeholder records cleanup", **log_context)
        
        register_configs = [
            {
                "name": "compartments",
                "path": self.compartment_json_path,
                "lock_type": "compartment"
            },
            {
                "name": "originals", 
                "path": self.original_images_json_path,
                "lock_type": "original"
            },
            {
                "name": "reviews",
                "path": self.review_json_path,
                "lock_type": "review"
            },
            {
                "name": "corners",
                "path": self.compartment_corners_json_path,
                "lock_type": "corners"
            }
        ]
        
        with self._thread_lock:
            for config in register_configs:
                register_name = config["name"]
                file_path = config["path"]
                lock_type = config["lock_type"]
                
                register_context = {
                    **log_context,
                    "register": register_name,
                    "file_path": str(file_path)
                }
                
                try:
                    self.logger.debug(f"Cleaning {register_name} register", **register_context)
                    
                    # Validate file exists
                    if not file_path.exists():
                        self.logger.info(f"Skipping {register_name}: file does not exist", **register_context)
                        continue
                    
                    with self.file_locks(lock_type):
                        # Read current data with validation
                        data = self._read_json_file(file_path, None)
                        
                        if not isinstance(data, list):
                            raise DataCorruptionError(
                                f"Expected list structure in {register_name} register, got {type(data)}",
                                file_path=str(file_path)
                            )
                        
                        original_count = len(data)
                        
                        # Filter out placeholder records with validation
                        cleaned_data = []
                        invalid_records = 0
                        
                        for i, record in enumerate(data):
                            if not isinstance(record, dict):
                                invalid_records += 1
                                self.logger.warning(
                                    f"Skipping non-dict record at index {i}",
                                    record_index=i,
                                    record_type=type(record).__name__,
                                    **register_context
                                )
                                continue
                            
                            hole_id = record.get("HoleID")
                            if hole_id != "INITIALISING":
                                cleaned_data.append(record)
                        
                        records_removed = original_count - len(cleaned_data)
                        cleanup_stats[register_name] = records_removed
                        
                        # Write cleaned data only if changes were made
                        if records_removed > 0:
                            # Create backup before cleanup
                            backup_data = data.copy()
                            
                            try:
                                lock_path = file_path.with_suffix(".json.lock")
                                self._write_json_file(file_path, lock_path, cleaned_data)
                                
                                self.logger.info(
                                    f"Successfully cleaned {register_name} register",
                                    records_removed=records_removed,
                                    original_count=original_count,
                                    final_count=len(cleaned_data),
                                    invalid_records=invalid_records,
                                    **register_context
                                )
                                
                            except (OSError, IOError, PermissionError) as e:
                                # Attempt rollback
                                try:
                                    self._write_json_file(file_path, lock_path, backup_data)
                                    self.logger.warning(
                                        f"Rolled back {register_name} after write failure",
                                        **register_context
                                    )
                                except Exception as rollback_error:
                                    self.logger.error(
                                        f"Critical: Failed to rollback {register_name} after cleanup failure",
                                        rollback_error=str(rollback_error),
                                        **register_context
                                    )
                                
                                error_msg = f"Failed to write cleaned {register_name} data"
                                cleanup_stats["errors"].append({
                                    "register": register_name,
                                    "error": error_msg,
                                    "details": str(e)
                                })
                                
                                raise FileOperationError(
                                    f"{error_msg}: {str(e)}",
                                    file_path=str(file_path),
                                    operation="cleanup_write"
                                ) from e
                        else:
                            self.logger.debug(
                                f"No placeholder records found in {register_name}",
                                original_count=original_count,
                                invalid_records=invalid_records,
                                **register_context
                            )
                
                except (DataCorruptionError, FileOperationError):
                    # Re-raise specific exceptions for critical errors
                    raise
                except json.JSONDecodeError as e:
                    error_msg = f"JSON corruption detected in {register_name} register"
                    cleanup_stats["errors"].append({
                        "register": register_name,
                        "error": error_msg,
                        "details": str(e)
                    })
                    self.logger.error(
                        error_msg,
                        json_error=str(e),
                        **register_context
                    )
                    # Continue with other registers for JSON errors
                except LockAcquisitionError as e:
                    error_msg = f"Could not acquire lock for {register_name} register"
                    cleanup_stats["errors"].append({
                        "register": register_name,
                        "error": error_msg,
                        "details": str(e)
                    })
                    self.logger.warning(
                        error_msg,
                        lock_error=str(e),
                        **register_context
                    )
                    # Continue with other registers for lock errors
                except Exception as e:
                    error_msg = f"Unexpected error cleaning {register_name} register"
                    cleanup_stats["errors"].append({
                        "register": register_name,
                        "error": error_msg,
                        "details": str(e)
                    })
                    self.logger.error(
                        error_msg,
                        error=str(e),
                        error_type=type(e).__name__,
                        **register_context
                    )
                    # Continue with other registers for unexpected errors
            
            # Summary logging
            total_removed = sum(cleanup_stats[key] for key in ["compartments", "originals", "reviews", "corners"])
            error_count = len(cleanup_stats["errors"])
            
            if error_count == 0:
                self.logger.info(
                    "Placeholder cleanup completed successfully",
                    total_records_removed=total_removed,
                    **log_context
                )
            else:
                self.logger.warning(
                    "Placeholder cleanup completed with errors",
                    total_records_removed=total_removed,
                    error_count=error_count,
                    errors=cleanup_stats["errors"],
                    **log_context
                )
        
        return cleanup_stats

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
        """Get compartment corners data as DataFrame (current user's data only) with processing metadata."""
        with self._thread_lock:
            try:
                data = self._read_json_file(
                    self.compartment_corners_json_path, self.corners_lock
                )
                
                # Ensure backward compatibility for all records
                processed_data = []
                for record in data:
                    processed_record = self._ensure_corner_processing_fields(record)
                    processed_data.append(processed_record)
                
                df = pd.DataFrame(processed_data)

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

        # Load main file data with backward compatibility
        if main_path.exists():
            try:
                with open(main_path, "r", encoding="utf-8") as f:
                    main_data = json.load(f)
                    if isinstance(main_data, list):
                        for record in main_data:
                            if isinstance(record, dict):
                                # Ensure backward compatibility for main file records
                                record = self._ensure_corner_processing_fields(record)
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
                        # Ensure backward compatibility for user file records
                        record = self._ensure_corner_processing_fields(record)
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
                                if isinstance(record, dict):
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
                    if not isinstance(record, dict):
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
                                if isinstance(record, dict):
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
                    if not isinstance(record, dict):
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
                        if isinstance(record, dict):
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
