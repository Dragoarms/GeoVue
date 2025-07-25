# utils/json_register_manager.py

"""
JSON Register Manager for data storage with Excel Power Query integration.

This module handles all data operations using JSON files as the primary storage,
with Excel files using Power Query for read-only data display. This approach:
- Eliminates Excel file locking issues
- Allows for faster, more reliable data operations  
- Enables users to add custom columns in Excel without affecting the source data
- Provides automatic Power Query setup for new Excel files

The JSON structure is designed to be Power Query friendly, with separate files
for compartments and original images data. Compartment corners are now stored
in a flattened structure for better scalability with 100,000+ images.

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
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, Any
from contextlib import contextmanager
import threading
import pandas as pd
# import numpy as np
from openpyxl import workbook, load_workbook
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.utils import get_column_letter


class JSONRegisterManager:
   """
   Manages register data using JSON files with Excel Power Query integration.
   
   This class provides thread-safe access to drilling register data stored in JSON,
   with automatic creation of Excel files configured with Power Query connections.
   """
   
   # File names
   COMPARTMENT_JSON = "compartment_register.json"
   ORIGINAL_JSON = "original_images_register.json"
   REVIEW_JSON = "compartment_reviews.json"
   COMPARTMENT_CORNERS_JSON = "compartment_corners.json"  # NEW: Flattened compartment data
   EXCEL_FILE = "Chip_Tray_Register.xlsx"
   DATA_SUBFOLDER = "Register Data (Do not edit)"
   
   # Sheet names
   COMPARTMENT_SHEET = "Compartment Register"
   MANUAL_SHEET = "Manual Entries"
   ORIGINAL_SHEET = "Original Images Register"
   REVIEW_SHEET = "Compartment Reviews"
   COMPARTMENT_CORNERS_SHEET = "Compartment Corners"  # NEW
   
   

   @staticmethod
   def check_existing_files_static(base_path: str) -> Dict[str, bool]:
       """
       Static method to check which files exist without creating an instance.
       
       Args:
           base_path: Directory path to check
           
       Returns:
           Dictionary with file existence status
       """
       base = Path(base_path)
       data_path = base / JSONRegisterManager.DATA_SUBFOLDER
       
       return {
           'excel': (base / JSONRegisterManager.EXCEL_FILE).exists(),
           'compartment_json': (data_path / JSONRegisterManager.COMPARTMENT_JSON).exists(),
           'original_json': (data_path / JSONRegisterManager.ORIGINAL_JSON).exists(),
           'review_json': (data_path / JSONRegisterManager.REVIEW_JSON).exists(),
           'compartment_corners_json': (data_path / JSONRegisterManager.COMPARTMENT_CORNERS_JSON).exists(),
           'data_folder': data_path.exists() and any(data_path.iterdir()) if data_path.exists() else False
       }
   
   @staticmethod
   def has_existing_data_static(base_path: str) -> bool:
       """Static method to check if any register data exists without creating an instance."""
       existing = JSONRegisterManager.check_existing_files_static(base_path)
       return any(existing.values())
   
   @staticmethod
   def get_data_summary_static(base_path: str) -> Dict[str, Any]:
       """
       Static method to get summary of existing data without creating an instance.
       
       Args:
           base_path: Directory path to check
           
       Returns:
           Dictionary with data counts and info
       """
       base = Path(base_path)
       data_path = base / JSONRegisterManager.DATA_SUBFOLDER
       
       summary = {
           'compartment_count': 0,
           'original_count': 0,
           'review_count': 0,
           'corner_count': 0,
           'has_excel': (base / JSONRegisterManager.EXCEL_FILE).exists(),
           'has_json_data': False
       }
       
       try:
           comp_path = data_path / JSONRegisterManager.COMPARTMENT_JSON
           if comp_path.exists():
               with open(comp_path, 'r', encoding='utf-8') as f:
                   data = json.load(f)
                   summary['compartment_count'] = len(data)
                   summary['has_json_data'] = True
                   
           orig_path = data_path / JSONRegisterManager.ORIGINAL_JSON
           if orig_path.exists():
               with open(orig_path, 'r', encoding='utf-8') as f:
                   data = json.load(f)
                   summary['original_count'] = len(data)
                   summary['has_json_data'] = True
                   
           review_path = data_path / JSONRegisterManager.REVIEW_JSON
           if review_path.exists():
               with open(review_path, 'r', encoding='utf-8') as f:
                   data = json.load(f)
                   summary['review_count'] = len(data)
                   summary['has_json_data'] = True
                   
           corner_path = data_path / JSONRegisterManager.COMPARTMENT_CORNERS_JSON
           if corner_path.exists():
               with open(corner_path, 'r', encoding='utf-8') as f:
                   data = json.load(f)
                   summary['corner_count'] = len(data)
                   summary['has_json_data'] = True
                   
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
       
       # Excel file goes in the base path
       self.excel_path = self.base_path / self.EXCEL_FILE
       
       # Data files go in subfolder
       self.data_path = self.base_path / self.DATA_SUBFOLDER
       
       # File paths for JSON data
       self.compartment_json_path = self.data_path / self.COMPARTMENT_JSON
       self.original_json_path = self.data_path / self.ORIGINAL_JSON
       self.review_json_path = self.data_path / self.REVIEW_JSON
       self.compartment_corners_json_path = self.data_path / self.COMPARTMENT_CORNERS_JSON
       
       # Lock files in data folder
       self.compartment_lock = self.compartment_json_path.with_suffix('.json.lock')
       self.original_lock = self.original_json_path.with_suffix('.json.lock')
       self.review_lock = self.review_json_path.with_suffix('.json.lock')
       self.corners_lock = self.compartment_corners_json_path.with_suffix('.json.lock')

       # Use RLock instead of Lock to prevent deadlocks from nested calls
       self._thread_lock = threading.RLock()
       
       # Track lock acquisition order to prevent deadlocks
       self._lock_order = ['compartment', 'original', 'review', 'corners']
       
       # Lock monitoring for debugging
       self._lock_stats = {
           'acquisitions': 0,
           'failures': 0,
           'contentions': 0,
           'max_wait_time': 0.0
       }
       
       # Initialize files if needed
       self._initialize_files()
       
       # Migrate existing nested data if needed
       self._migrate_nested_compartments()
   
   def check_existing_files(self) -> Dict[str, bool]:
       """
       Check which files already exist.
       
       Returns:
           Dictionary with file existence status
       """
       return {
           'excel': self.excel_path.exists(),
           'compartment_json': self.compartment_json_path.exists(),
           'original_json': self.original_json_path.exists(),
           'review_json': self.review_json_path.exists(),
           'compartment_corners_json': self.compartment_corners_json_path.exists(),
           'data_folder': self.data_path.exists() and any(self.data_path.iterdir())
       }
   
   def has_existing_data(self) -> bool:
       """Check if any register data already exists."""
       existing = self.check_existing_files()
       return any(existing.values())
   
   def get_data_summary(self) -> Dict[str, Any]:
       """
       Get summary of existing data.
       
       Returns:
           Dictionary with data counts and info
       """
       summary = {
           'compartment_count': 0,
           'original_count': 0,
           'review_count': 0,
           'corner_count': 0,
           'has_excel': self.excel_path.exists(),
           'has_json_data': False,
           'lock_stats': self._lock_stats.copy()
       }
       
       try:
           if self.compartment_json_path.exists():
               data = self._read_json_file(self.compartment_json_path, self.compartment_lock)
               summary['compartment_count'] = len(data)
               summary['has_json_data'] = True
               
           if self.original_json_path.exists():
               data = self._read_json_file(self.original_json_path, self.original_lock)
               summary['original_count'] = len(data)
               summary['has_json_data'] = True
               
           if self.review_json_path.exists():
               data = self._read_json_file(self.review_json_path, self.review_lock)
               summary['review_count'] = len(data)
               summary['has_json_data'] = True
               
           if self.compartment_corners_json_path.exists():
               data = self._read_json_file(self.compartment_corners_json_path, self.corners_lock)
               summary['corner_count'] = len(data)
               summary['has_json_data'] = True
               
       except Exception as e:
           self.logger.error(f"Error getting data summary: {e}")
           
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
           self.logger.info(f"Compartment JSON already exists: {self.compartment_json_path}")
       
       if not self.original_json_path.exists():
           self._write_json_file(self.original_json_path, self.original_lock, [])
           self.logger.info(f"Created original images JSON: {self.original_json_path}")
       else:
           self.logger.info(f"Original images JSON already exists: {self.original_json_path}")
       
       if not self.review_json_path.exists():
           self._write_json_file(self.review_json_path, self.review_lock, [])
           self.logger.info(f"Created review JSON: {self.review_json_path}")
       else:
           self.logger.info(f"Review JSON already exists: {self.review_json_path}")
       
       # Initialize compartment corners JSON file only if it doesn't exist
       if not self.compartment_corners_json_path.exists():
           self._write_json_file(self.compartment_corners_json_path, self.corners_lock, [])
           self.logger.info(f"Created compartment corners JSON: {self.compartment_corners_json_path}")
       else:
           self.logger.info(f"Compartment corners JSON already exists: {self.compartment_corners_json_path}")
       

       # Use template instead of creating from scratch
       # Create Excel from template only if it doesn't exist
       if not self.excel_path.exists():
           self.logger.info(f"Excel file does not exist, creating from template: {self.excel_path}")
           self._create_excel_from_template()
       else:
           self.logger.info(f"Excel file already exists: {self.excel_path}")
   
   def _migrate_nested_compartments(self) -> None:
       """
       One-time migration of nested compartment data to flattened structure.
       This checks if there's nested data in the original_images_register.json
       and migrates it to the compartment_corners.json file.
       """
       try:
           # Check if migration is needed
           if not self.original_json_path.exists():
               return
               
           original_data = self._read_json_file(self.original_json_path, self.original_lock)
           corners_data = self._read_json_file(self.compartment_corners_json_path, self.corners_lock)
           
           # Check if we already have corner data
           if corners_data:
               self.logger.info("Compartment corners already exist, skipping migration")
               return
           
           # Check if any original records have nested Compartments
           needs_migration = any('Compartments' in record and record['Compartments'] 
                                for record in original_data)
           
           if not needs_migration:
               self.logger.info("No nested compartment data found, skipping migration")
               return
           
           self.logger.info("Starting migration of nested compartment data...")
           
           # Migrate data
           migrated_corners = []
           updated_originals = []
           
           for record in original_data:
               # Copy record without Compartments field
               updated_record = {k: v for k, v in record.items() if k != 'Compartments'}
               updated_originals.append(updated_record)
               
               # Extract compartments if they exist
               if 'Compartments' in record and record['Compartments']:
                   for comp_num, corners in record['Compartments'].items():
                       if isinstance(corners, list) and len(corners) == 4:
                           corner_record = {
                               'HoleID': record['HoleID'],
                               'Depth_From': record['Depth_From'],
                               'Depth_To': record['Depth_To'],
                               'Original_Filename': record['Original_Filename'],
                               'Compartment_Number': int(comp_num),
                               'Top_Left_X': corners[0][0],
                               'Top_Left_Y': corners[0][1],
                               'Top_Right_X': corners[1][0],
                               'Top_Right_Y': corners[1][1],
                               'Bottom_Right_X': corners[2][0],
                               'Bottom_Right_Y': corners[2][1],
                               'Bottom_Left_X': corners[3][0],
                               'Bottom_Left_Y': corners[3][1]
                           }
                           
                           # Add scale data if available
                           if 'Scale_PxPerCm' in record:
                               corner_record['Scale_PxPerCm'] = record['Scale_PxPerCm']
                           if 'Scale_Confidence' in record:
                               corner_record['Scale_Confidence'] = record['Scale_Confidence']
                               
                           migrated_corners.append(corner_record)
           
           # Save migrated data
           if migrated_corners:
               self._write_json_file(self.compartment_corners_json_path, self.corners_lock, migrated_corners)
               self.logger.info(f"Migrated {len(migrated_corners)} compartment corners")
               
               # Update original records (remove Compartments field)
               self._write_json_file(self.original_json_path, self.original_lock, updated_originals)
               self.logger.info("Removed nested Compartments from original image records")
           
       except Exception as e:
           self.logger.error(f"Error during compartment migration: {e}")
           self.logger.error(traceback.format_exc())
   
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
                   if getattr(sys, 'frozen', False):
                       base_path = sys._MEIPASS
                   else:
                       base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                   
                   template_path = os.path.join(base_path, "resources", "Register Template File", "Chip_Tray_Register.xltx")
                   self.logger.info(f"Using fallback template path: {template_path}")

               if not os.path.exists(template_path):
                   raise FileNotFoundError(f"Template file not found at: {template_path}")

               self.logger.info(f"Opening template file: {template_path}")

               # Use win32com to properly handle Excel template
               try:
                   import win32com.client as win32
                   
                   # Create Excel instance
                   excel = win32.gencache.EnsureDispatch('Excel.Application')
                   excel.Visible = False
                   excel.DisplayAlerts = False
                   
                   try:
                       # Open the template file
                       workbook = excel.Workbooks.Open(template_path)
                       
                       # Set the author property
                       workbook.BuiltinDocumentProperties("Author").Value = "George Symonds"
                       
                       # Save as regular Excel file (.xlsx)
                       workbook.SaveAs(
                           str(self.excel_path),
                           FileFormat=51  # xlOpenXMLWorkbook = 51 (.xlsx format)
                       )
                       
                       # Close the workbook
                       workbook.Close(SaveChanges=False)
                       
                       self.logger.info(f"Excel file created successfully from template using win32com")
                       
                   finally:
                       # Ensure Excel is closed
                       excel.Quit()
                       
               except ImportError:
                   self.logger.error("win32com.client not available. Please install pywin32.")
                   raise
               except Exception as com_error:
                   self.logger.error(f"COM error creating Excel from template: {com_error}")
                   raise

               if self.excel_path.exists():
                   self.logger.info(f"Excel file created successfully at: {self.excel_path}")
               else:
                   raise FileNotFoundError(f"Excel file was not created at: {self.excel_path}")

           except Exception as e:
               self.logger.error(f"Error creating Excel from template: {e}")
               self.logger.error(traceback.format_exc())
               raise
   
   def _try_acquire_lock(self, lock_path: Path) -> bool:
       """Try to acquire a lock once."""
       try:
           if lock_path.exists():
               # Check if lock is stale (older than 60 seconds)
               lock_age = time.time() - lock_path.stat().st_mtime
               if lock_age > 60:
                   self.logger.warning(f"Removing stale lock: {lock_path}")
                   lock_path.unlink()
               else:
                   return False
           
           # Create lock
           lock_path.write_text(str(os.getpid()))
           return True
           
       except Exception:
           return False
   
   def _acquire_file_lock_with_backoff(self, lock_path: Path, timeout: int = 30) -> bool:
       """Acquire file lock with exponential backoff."""
       start_time = time.time()
       backoff = 0.1  # Start with 100ms
       wait_time = 0.0
       
       while time.time() - start_time < timeout:
           if self._try_acquire_lock(lock_path):
               # Update lock stats
               self._lock_stats['acquisitions'] += 1
               if wait_time > 0:
                   self._lock_stats['contentions'] += 1
                   self._lock_stats['max_wait_time'] = max(self._lock_stats['max_wait_time'], wait_time)
               return True
           
           time.sleep(min(backoff, 2.0))  # Cap at 2 seconds
           wait_time = time.time() - start_time
           backoff *= 1.5  # Exponential backoff
       
       self._lock_stats['failures'] += 1
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
   
   def _acquire_file_locks_ordered(self, required_locks: List[str], timeout: int = 30) -> Dict[str, bool]:
       """
       Acquire multiple file locks in a consistent order to prevent deadlocks.
       
       Args:
           required_locks: List of lock names needed ('compartment', 'original', 'review', 'corners')
           timeout: Timeout in seconds
           
       Returns:
           Dictionary mapping lock names to success status
       """
       lock_map = {
           'compartment': self.compartment_lock,
           'original': self.original_lock,
           'review': self.review_lock,
           'corners': self.corners_lock
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
           'compartment': self.compartment_lock,
           'original': self.original_lock,
           'review': self.review_lock,
           'corners': self.corners_lock
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
                with open(file_path, 'r', encoding='utf-8') as f:
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
                    self.logger.error(f"JSON file contains non-dictionary items: {file_path}")
                    # Filter out non-dict items
                    valid_data = [item for item in data if isinstance(item, dict)]
                    self.logger.warning(f"Filtered out {len(data) - len(valid_data)} invalid items")
                    return valid_data
                    
                return data
                
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decode error in {file_path}: {e}")
                
                # Try to read backup if available
                backup_path = file_path.with_suffix('.json.backup')
                if backup_path.exists():
                    self.logger.info("Attempting to read from backup file")
                    try:
                        with open(backup_path, 'r', encoding='utf-8') as f:
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
                for encoding in ['utf-8-sig', 'latin-1', 'cp1252']:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            data = json.load(f)
                        self.logger.info(f"Successfully read file with {encoding} encoding")
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
   
   def _write_json_file(self, file_path: Path, lock_path: Path, data: List[Dict]) -> None:
        """Write JSON file with locking, backup, and validation."""
        if not self._acquire_file_lock(lock_path):
            raise RuntimeError(f"Could not acquire lock for {file_path}")
        
        try:
            # Validate input data
            if not isinstance(data, list):
                raise ValueError(f"Data must be a list, got {type(data)}")
                
            # Create backup if file exists and is valid
            if file_path.exists() and file_path.stat().st_size > 0:
                backup_path = file_path.with_suffix('.json.backup')
                try:
                    # Verify the existing file is valid JSON before backing up
                    with open(file_path, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                    # Only backup if existing file is valid
                    shutil.copy2(file_path, backup_path)
                    self.logger.debug(f"Created backup: {backup_path}")
                except Exception as e:
                    self.logger.warning(f"Existing file appears corrupted, skipping backup: {e}")
            
            # If data is empty, add an example record with all columns
            if not data:
                if "compartment_register" in str(file_path):
                    # Example compartment record with all columns
                    data = [{
                        "HoleID": "INITIALISING",
                        "From": 0,
                        "To": 1,
                        "Photo_Status": "Example",
                        "Processed_Date": datetime.now().isoformat(),
                        "Processed_By": "System",
                        "Comments": "This is an example record - please delete after adding real data",
                        "Average_Hex_Color": "#000000",
                        "Image_Width_Cm": 2.0
                    }]
                elif "original_images_register" in str(file_path):
                    # Example original image record WITHOUT nested compartments
                    data = [{
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
                        "Scale_Confidence": 0.95
                    }]
                elif "compartment_reviews" in str(file_path):
                    # Example review record with all columns including toggles
                    data = [{
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
                        "+ CHH/M": False
                    }]
                    
                elif "compartment_corners" in str(file_path):
                    # Example compartment corner record
                    data = [{
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
                        "Scale_Confidence": 0.95
                    }]
             
            # Create parent directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to temporary file first
            temp_path = file_path.with_suffix('.json.tmp')
            try:
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str, separators=(',', ': '))
                
                # Verify the temporary file is valid
                with open(temp_path, 'r', encoding='utf-8') as f:
                    verified_data = json.load(f)
                    
                # If verification passes, move temp to final location
                temp_path.replace(file_path)
                self.logger.debug(f"Successfully wrote {len(data)} records to {file_path}")
                
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


   def update_compartment(self, hole_id: str, depth_from: int, depth_to: int,
                            photo_status: str, processed_by: Optional[str] = None,
                            comments: Optional[str] = None, 
                            image_width_cm: Optional[float] = None) -> bool:
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
                data = self._read_json_file(self.compartment_json_path, self.compartment_lock)
                
                # Convert to records for easier manipulation
                records = {(r['HoleID'], r['From'], r['To']): r for r in data}
                
                # Ensure depths are integers
                depth_from = int(depth_from)
                depth_to = int(depth_to)

                # Update or create record
                key = (hole_id, depth_from, depth_to)
                timestamp = datetime.now().isoformat()
                username = processed_by or os.getenv("USERNAME", "Unknown")
                
                if key in records:
                    # Update existing
                    records[key].update({
                        'Photo_Status': photo_status,
                        'Processed_Date': timestamp,
                        'Processed_By': username
                    })
                    if comments:
                        records[key]['Comments'] = comments
                    
                    # Add image width if provided
                    if image_width_cm is not None:
                        records[key]['Image_Width_Cm'] = image_width_cm

                else:
                    # Create new
                    new_record = {
                        'HoleID': hole_id,
                        'From': depth_from,
                        'To': depth_to,
                        'Photo_Status': photo_status,
                        'Processed_Date': timestamp,
                        'Processed_By': username,
                        'Comments': comments
                    }
                    
                    # Add image width to new record if provided
                    if image_width_cm is not None:
                        new_record['Image_Width_Cm'] = image_width_cm
                    
                    records[key] = new_record
                
                # Convert back to list and save
                data = list(records.values())
                self._write_json_file(self.compartment_json_path, self.compartment_lock, data)
                
                self.logger.info(f"Updated compartment: {hole_id} {depth_from}-{depth_to}")
                return True
                
            except Exception as e:
                self.logger.error(f"Error updating compartment: {e}")
                return False


   def update_original_image(self, hole_id: str, depth_from: int, depth_to: int,
                           original_filename: str, is_approved: bool,
                           upload_success: bool, uploaded_by: Optional[str] = None,
                           comments: Optional[str] = None,
                           scale_px_per_cm: Optional[float] = None,
                           scale_confidence: Optional[float] = None,
                           compartment_data: Optional[Dict[str, List[List[int]]]] = None) -> bool:
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
               with self.file_locks('original', 'corners'):
                   # Read current data
                   original_data = []
                   if self.original_json_path.exists():
                       with open(self.original_json_path, 'r', encoding='utf-8') as f:
                           original_data = json.load(f)
                   
                   corners_data = []
                   if self.compartment_corners_json_path.exists():
                       with open(self.compartment_corners_json_path, 'r', encoding='utf-8') as f:
                           corners_data = json.load(f)
                   
                   # Ensure depths are integers
                   depth_from = int(depth_from)
                   depth_to = int(depth_to)
                   
                   timestamp = datetime.now().isoformat()
                   username = uploaded_by or os.getenv("USERNAME", "Unknown")
                   status = "Uploaded" if upload_success else "Failed"
                   
                   # Find existing record for this specific file
                   record_found = False
                   for i, record in enumerate(original_data):
                       if (record['HoleID'] == hole_id and 
                           record['Depth_From'] == depth_from and 
                           record['Depth_To'] == depth_to and
                           record['Original_Filename'] == original_filename):
                           # Update existing record
                           record_found = True
                           original_data[i].update({
                               'Uploaded_By': username,
                               'Comments': comments
                           })
                           
                           if is_approved:
                               original_data[i]['Approved_Upload_Date'] = timestamp
                               original_data[i]['Approved_Upload_Status'] = status
                           else:
                               original_data[i]['Rejected_Upload_Date'] = timestamp
                               original_data[i]['Rejected_Upload_Status'] = status
                           
                           # Update scale data
                           if scale_px_per_cm is not None:
                               original_data[i]['Scale_PxPerCm'] = scale_px_per_cm
                           if scale_confidence is not None:
                               original_data[i]['Scale_Confidence'] = scale_confidence
                           
                           break
                   
                   if not record_found:
                       # Create new record
                       new_record = {
                           'HoleID': hole_id,
                           'Depth_From': depth_from,
                           'Depth_To': depth_to,
                           'Original_Filename': original_filename,
                           'File_Count': 1,
                           'All_Filenames': original_filename,
                           'Approved_Upload_Date': timestamp if is_approved else None,
                           'Approved_Upload_Status': status if is_approved else None,
                           'Rejected_Upload_Date': timestamp if not is_approved else None,
                           'Rejected_Upload_Status': status if not is_approved else None,
                           'Uploaded_By': username,
                           'Comments': comments
                       }
                       
                       # Add scale data if provided
                       if scale_px_per_cm is not None:
                           new_record['Scale_PxPerCm'] = scale_px_per_cm
                       if scale_confidence is not None:
                           new_record['Scale_Confidence'] = scale_confidence
                       
                       original_data.append(new_record)
                   
                   # Now handle compartment data separately
                   if compartment_data is not None:
                       # Remove existing corners for this image
                       corners_data = [c for c in corners_data 
                                     if not (c['HoleID'] == hole_id and 
                                           c['Depth_From'] == depth_from and 
                                           c['Depth_To'] == depth_to and
                                           c['Original_Filename'] == original_filename)]
                       
                       # Add new corners
                       for comp_num, corners in compartment_data.items():
                           if isinstance(corners, list) and len(corners) == 4:
                               corner_record = {
                                   'HoleID': hole_id,
                                   'Depth_From': depth_from,
                                   'Depth_To': depth_to,
                                   'Original_Filename': original_filename,
                                   'Compartment_Number': int(comp_num),
                                   'Top_Left_X': corners[0][0],
                                   'Top_Left_Y': corners[0][1],
                                   'Top_Right_X': corners[1][0],
                                   'Top_Right_Y': corners[1][1],
                                   'Bottom_Right_X': corners[2][0],
                                   'Bottom_Right_Y': corners[2][1],
                                   'Bottom_Left_X': corners[3][0],
                                   'Bottom_Left_Y': corners[3][1]
                               }
                               
                               # Copy scale data if available
                               if scale_px_per_cm is not None:
                                   corner_record['Scale_PxPerCm'] = scale_px_per_cm
                               if scale_confidence is not None:
                                   corner_record['Scale_Confidence'] = scale_confidence
                               
                               corners_data.append(corner_record)
                   
                   # Save both files
                   with open(self.original_json_path, 'w', encoding='utf-8') as f:
                       json.dump(original_data, f, indent=2, ensure_ascii=False, default=str)
                   
                   with open(self.compartment_corners_json_path, 'w', encoding='utf-8') as f:
                       json.dump(corners_data, f, indent=2, ensure_ascii=False, default=str)
                   
                   self.logger.info(f"Updated original image: {original_filename}")
                   return True
               
           except Exception as e:
               self.logger.error(f"Error updating original image: {e}")
               self.logger.error(traceback.format_exc())
               return False

   def get_compartment_corners_from_image(self, hole_id: str, depth_from: int, depth_to: int, 
                                        original_filename: str, compartment_num: str) -> Optional[Dict[str, Tuple[int, int]]]:
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
               data = self._read_json_file(self.compartment_corners_json_path, self.corners_lock)
               
               depth_from = int(depth_from)
               depth_to = int(depth_to)
               comp_num_int = int(compartment_num)
               
               for record in data:
                   if (record['HoleID'] == hole_id and 
                       record['Depth_From'] == depth_from and 
                       record['Depth_To'] == depth_to and
                       record['Original_Filename'] == original_filename and
                       record['Compartment_Number'] == comp_num_int):
                       
                       return {
                           'top_left': (record['Top_Left_X'], record['Top_Left_Y']),
                           'top_right': (record['Top_Right_X'], record['Top_Right_Y']),
                           'bottom_right': (record['Bottom_Right_X'], record['Bottom_Right_Y']),
                           'bottom_left': (record['Bottom_Left_X'], record['Bottom_Left_Y'])
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
               data = self._read_json_file(self.compartment_corners_json_path, self.corners_lock)
               
               # Create a set of records to remove for efficiency
               removal_keys = set()
               for removal in removals:
                   key = (
                       removal['hole_id'],
                       int(removal['depth_from']),
                       int(removal['depth_to']),
                       removal['original_filename'],
                       int(removal['compartment_num'])
                   )
                   removal_keys.add(key)
               
               # Filter out records that match removal criteria
               filtered_data = []
               for record in data:
                   record_key = (
                       record['HoleID'],
                       record['Depth_From'],
                       record['Depth_To'],
                       record['Original_Filename'],
                       record['Compartment_Number']
                   )
                   
                   if record_key not in removal_keys:
                       filtered_data.append(record)
                   else:
                       successful_removals += 1
                       self.logger.info(f"Removed compartment {record['Compartment_Number']} from {record['Original_Filename']}")
               
               # Save if we made changes
               if successful_removals > 0:
                   self._write_json_file(self.compartment_corners_json_path, self.corners_lock, filtered_data)
                   self.logger.info(f"Batch removed {successful_removals} compartments")
                   
           except Exception as e:
               self.logger.error(f"Error in batch_remove_compartments: {e}")
               
       return successful_removals

   def get_all_compartments_for_tray(self, hole_id: str, depth_from: int, depth_to: int) -> Dict[str, Dict]:
       """
       Get all compartments from all images for a specific tray.
       
       Returns:
           Dictionary mapping filenames to their compartment data
           e.g., {
               "SB0020_120-140_Original.jpg": {
                   "scale_px_per_cm": 50.0,
                   "scale_confidence": 0.95,
                   "compartments": {
                       "1": [[0,0], [100,0], [100,100], [0,100]],
                       "2": [[100,0], [200,0], [200,100], [100,100]]
                   }
               }
           }
       """
       with self._thread_lock:
           try:
               # Read original image data for scale info
               original_data = self._read_json_file(self.original_json_path, self.original_lock)
               corners_data = self._read_json_file(self.compartment_corners_json_path, self.corners_lock)
               
               depth_from = int(depth_from)
               depth_to = int(depth_to)
               
               result = {}
               
               # First, get all filenames and scale data from original records
               for record in original_data:
                   if (record['HoleID'] == hole_id and 
                       record['Depth_From'] == depth_from and 
                       record['Depth_To'] == depth_to):
                       
                       filename = record['Original_Filename']
                       result[filename] = {
                           'scale_px_per_cm': record.get('Scale_PxPerCm'),
                           'scale_confidence': record.get('Scale_Confidence'),
                           'compartments': {}
                       }
               
               # Then, populate compartments from corners data
               for corner in corners_data:
                   if (corner['HoleID'] == hole_id and 
                       corner['Depth_From'] == depth_from and 
                       corner['Depth_To'] == depth_to):
                       
                       filename = corner['Original_Filename']
                       if filename in result:
                           comp_num = str(corner['Compartment_Number'])
                           # Convert back to nested list format for compatibility
                           result[filename]['compartments'][comp_num] = [
                               [corner['Top_Left_X'], corner['Top_Left_Y']],
                               [corner['Top_Right_X'], corner['Top_Right_Y']],
                               [corner['Bottom_Right_X'], corner['Bottom_Right_Y']],
                               [corner['Bottom_Left_X'], corner['Bottom_Left_Y']]
                           ]
               
               return result
               
           except Exception as e:
               self.logger.error(f"Error getting all compartments for tray: {e}")
               return {}

   def get_compartment_scale(self, hole_id: str, depth_from: int, depth_to: int) -> Optional[Dict[str, float]]:
       """
       Get scale data from the first image found for a compartment.
       """
       all_compartments = self.get_all_compartments_for_tray(hole_id, depth_from, depth_to)
       
       # Return scale from first image found
       for filename, data in all_compartments.items():
           scale_data = {}
           if data.get('scale_px_per_cm') is not None:
               scale_data['scale_px_per_cm'] = data['scale_px_per_cm']
           if data.get('scale_confidence') is not None:
               scale_data['scale_confidence'] = data['scale_confidence']
           
           # Get image width from compartment register for backwards compatibility
           with self._thread_lock:
               comp_data = self._read_json_file(self.compartment_json_path, self.compartment_lock)
               for record in comp_data:
                   if (record['HoleID'] == hole_id and 
                       record['From'] == int(depth_from) and 
                       record['To'] == int(depth_to)):
                       if 'Image_Width_Cm' in record:
                           scale_data['image_width_cm'] = record['Image_Width_Cm']
                       break
           
           if scale_data:
               return scale_data
       
       return None

   def update_compartment_review(self, hole_id: str, depth_from: int, depth_to: int,
                               reviewed_by: Optional[str] = None, comments: Optional[str] = None,
                               review_number: Optional[int] = None,
                               **kwargs) -> bool:
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
                   if (review['HoleID'] == hole_id and 
                       review['From'] == depth_from and 
                       review['To'] == depth_to and
                       review['Reviewed_By'] == username):
                       existing_index = idx
                       existing_review = review.copy()  # Keep a copy to preserve all fields
                       break
               
               if existing_index is not None:
                   # Update existing review - preserve all existing fields
                   if review_number is not None:
                       use_review_number = review_number
                   else:
                       use_review_number = existing_review.get('Review_Number', 0) + 1
                   
                   # Start with existing data to preserve all fields
                   updated_review = existing_review.copy()
                   
                   # Update core fields
                   updated_review.update({
                       'Review_Number': use_review_number,
                       'Review_Date': timestamp,
                       'Comments': comments,
                   })
                   
                   # Update fields from kwargs (new toggle values)
                   updated_review.update(kwargs)
                   
                   # Replace the record
                   data[existing_index] = updated_review
                   
                   self.logger.info(f"Updated existing review (edit #{use_review_number}) by {username} for: {hole_id} {depth_from}-{depth_to}")
               else:
                   # Create new review record
                   use_review_number = review_number if review_number is not None else 1
                   
                   new_record = {
                       'HoleID': hole_id,
                       'From': depth_from,
                       'To': depth_to,
                       'Reviewed_By': username,
                       'Review_Number': use_review_number,
                       'Review_Date': timestamp,
                       'Initial_Review_Date': timestamp,
                       'Comments': comments,
                   }
                   
                   # Add all kwargs (toggle fields from config)
                   new_record.update(kwargs)
                   
                   # Add the new record
                   data.append(new_record)
                   
                   self.logger.info(f"Created new review #{use_review_number} by {username} for: {hole_id} {depth_from}-{depth_to}")
               
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
                with self.file_locks('compartment'):
                    # Read current compartment data
                    comp_data = []
                    if self.compartment_json_path.exists():
                        # Handle empty or invalid JSON files
                        try:
                            # Check if file is empty
                            if self.compartment_json_path.stat().st_size == 0:
                                self.logger.warning("Compartment JSON file is empty, starting with empty list")
                                comp_data = []
                            else:
                                with open(self.compartment_json_path, 'r', encoding='utf-8') as f:
                                    comp_data = json.load(f)
                        except json.JSONDecodeError as e:
                            self.logger.error(f"JSON decode error in compartment file: {e}")
                            self.logger.warning("Starting with empty compartment list")
                            comp_data = []
                        except Exception as e:
                            self.logger.error(f"Error reading compartment file: {e}")
                            comp_data = []
                        
                    comp_records = {(r['HoleID'], r['From'], r['To']): r for r in comp_data}
                    
                    # Process all compartment updates
                    for update in updates:
                        try:
                            hole_id = update['hole_id']
                            depth_from = int(update['depth_from'])
                            depth_to = int(update['depth_to'])
                            photo_status = update['photo_status']
                            
                            key = (hole_id, depth_from, depth_to)
                            timestamp = datetime.now().isoformat()
                            username = update.get('processed_by', os.getenv("USERNAME", "Unknown"))
                            
                            # Update compartment register
                            if key in comp_records:
                                # Update existing
                                comp_records[key].update({
                                    'Photo_Status': photo_status,
                                    'Processed_Date': timestamp,
                                    'Processed_By': username
                                })
                                if update.get('comments'):
                                    comp_records[key]['Comments'] = update['comments']
                                if 'image_width_cm' in update:
                                    comp_records[key]['Image_Width_Cm'] = update['image_width_cm']
                            else:
                                # Create new
                                new_record = {
                                    'HoleID': hole_id,
                                    'From': depth_from,
                                    'To': depth_to,
                                    'Photo_Status': photo_status,
                                    'Processed_Date': timestamp,
                                    'Processed_By': username,
                                    'Comments': update.get('comments')
                                }
                                if 'image_width_cm' in update:
                                    new_record['Image_Width_Cm'] = update['image_width_cm']
                                
                                comp_records[key] = new_record
                            
                            successful_updates += 1
                            
                        except Exception as e:
                            self.logger.error(f"Error in batch update for {update}: {e}")
                    
                    # Write all updates at once
                    if successful_updates > 0:
                        # Save compartment data with compact formatting
                        with open(self.compartment_json_path, 'w', encoding='utf-8') as f:
                            json.dump(list(comp_records.values()), f, indent=2, 
                                    ensure_ascii=False, default=str, separators=(',', ': '))
                        
                        self.logger.info(f"Batch updated {successful_updates} compartments")
                        
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
               data = self._read_json_file(self.compartment_json_path, self.compartment_lock)
               
               # Convert to records for easier manipulation
               records = {(r['HoleID'], r['From'], r['To']): r for r in data}
               
               # Apply updates
               for update in color_updates:
                   try:
                       hole_id = update['hole_id']
                       depth_from = int(update['depth_from'])
                       depth_to = int(update['depth_to'])
                       hex_color = update['average_hex_color']
                       
                       key = (hole_id, depth_from, depth_to)
                       
                       if key in records:
                           # Add or update the Average_Hex_Color field
                           records[key]['Average_Hex_Color'] = hex_color
                           successful_updates += 1
                       else:
                           self.logger.warning(f"No matching compartment found for {hole_id} {depth_from}-{depth_to}m")
                   
                   except Exception as e:
                       self.logger.error(f"Error updating color for {update}: {e}")
               
               # Save if we made updates
               if successful_updates > 0:
                   data = list(records.values())
                   self._write_json_file(self.compartment_json_path, self.compartment_lock, data)
                   self.logger.info(f"Successfully updated {successful_updates} hex colors")
               
           except Exception as e:
               self.logger.error(f"Error batch updating hex colors: {e}")
               
       return successful_updates

   def get_user_review(self, hole_id: str, depth_from: int, depth_to: int, 
                      username: Optional[str] = None) -> Optional[Dict]:
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
                   if (review['HoleID'] == hole_id and 
                       review['From'] == depth_from and 
                       review['To'] == depth_to and
                       review['Reviewed_By'] == username):
                       return review.copy()  # Return a copy to prevent accidental modification
               
               return None
               
           except Exception as e:
               self.logger.error(f"Error getting user review: {e}")
               return None
   
   def get_all_reviews_for_compartment(self, hole_id: str, depth_from: int, depth_to: int) -> List[Dict]:
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
               reviews = [r.copy() for r in data 
                         if r['HoleID'] == hole_id 
                         and r['From'] == depth_from 
                         and r['To'] == depth_to]
               
               # Sort by most recent review date
               reviews.sort(key=lambda x: x.get('Review_Date', ''), reverse=True)
               return reviews
               
           except Exception as e:
               self.logger.error(f"Error getting all reviews: {e}")
               return []

   def get_compartment_data(self, hole_id: Optional[str] = None) -> pd.DataFrame:
       """Get compartment data as DataFrame."""
       with self._thread_lock:
           try:
               data = self._read_json_file(self.compartment_json_path, self.compartment_lock)
               df = pd.DataFrame(data)

               # Ensure Average_Hex_Color column exists for backwards compatibility
               if not df.empty and 'Average_Hex_Color' not in df.columns:
                   df['Average_Hex_Color'] = None
               
               if hole_id and not df.empty:
                   df = df[df['HoleID'] == hole_id]
               
               return df
               
           except Exception as e:
               self.logger.error(f"Error getting compartment data: {e}")
               return pd.DataFrame()
   
   def get_original_image_data(self, hole_id: Optional[str] = None) -> pd.DataFrame:
       """Get original image data as DataFrame."""
       with self._thread_lock:
           try:
               data = self._read_json_file(self.original_json_path, self.original_lock)
               df = pd.DataFrame(data)
               
               if hole_id and not df.empty:
                   df = df[df['HoleID'] == hole_id]
               
               return df
               
           except Exception as e:
               self.logger.error(f"Error getting original image data: {e}")
               return pd.DataFrame()
   
   def get_compartment_corners_data(self, hole_id: Optional[str] = None) -> pd.DataFrame:
       """Get compartment corners data as DataFrame."""
       with self._thread_lock:
           try:
               data = self._read_json_file(self.compartment_corners_json_path, self.corners_lock)
               df = pd.DataFrame(data)
               
               if hole_id and not df.empty:
                   df = df[df['HoleID'] == hole_id]
               
               return df
               
           except Exception as e:
               self.logger.error(f"Error getting compartment corners data: {e}")
               return pd.DataFrame()
   
   def get_review_data(self, hole_id: Optional[str] = None) -> pd.DataFrame:
       """
       Get review data as DataFrame.
       Returns ALL fields stored in the JSON, not just current config fields.
       """
       with self._thread_lock:
           try:
               data = self._read_json_file(self.review_json_path, self.review_lock)
               df = pd.DataFrame(data)
               
               if hole_id and not df.empty:
                   df = df[df['HoleID'] == hole_id]
               
               return df
               
           except Exception as e:
               self.logger.error(f"Error getting review data: {e}")
               return pd.DataFrame()

   def get_review_field_summary(self) -> Dict[str, int]:
       """
       Get a summary of all fields that appear in review data.
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