#!/usr/bin/env python3
"""
Comprehensive test suite for JsonRegisterManager class.
Tests all JSON register methods, file locking, and data persistence.
"""

import sys
import os
import unittest
import tempfile
import shutil
import json
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.json_register_manager import JSONRegisterManager, get_user_suffix


class TestJsonRegisterManagerCore(unittest.TestCase):
    """Test core JsonRegisterManager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = Mock()
        self.register_manager = JSONRegisterManager(self.temp_dir, self.logger)
        
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test JsonRegisterManager initialization."""
        self.assertEqual(self.register_manager.base_path, self.temp_dir)
        self.assertIsNotNone(self.register_manager.logger)
        
        # Check file paths are set correctly
        expected_files = [
            "compartment_register.json",
            "original_images_register.json", 
            "compartment_reviews.json"
        ]
        
        for filename in expected_files:
            self.assertIn(filename, str(self.register_manager.files[filename.split('.')[0].replace('_', '')]))
    
    def test_static_methods(self):
        """Test static utility methods."""
        # Test user suffix generation
        suffix = get_user_suffix()
        self.assertIsInstance(suffix, str)
        self.assertTrue(len(suffix) > 0)
        
        # Test file existence checking
        test_file = Path(self.temp_dir) / "test.json"
        test_file.write_text('{"test": "data"}')
        
        exists = JSONRegisterManager.check_existing_files_static(
            self.temp_dir, ["test"]
        )
        self.assertTrue(exists["test"])
        
        # Test data summary
        summary = JSONRegisterManager.get_data_summary_static(self.temp_dir)
        self.assertIsInstance(summary, dict)
        self.assertIn("total_files", summary)
    
    def test_file_creation_and_initialization(self):
        """Test that register files are created and initialized properly."""
        # Force initialization
        self.register_manager._initialize_files()
        
        # Check files exist
        for file_key, file_path in self.register_manager.files.items():
            self.assertTrue(file_path.exists(), f"File {file_key} not created")
            
            # Check valid JSON structure
            with open(file_path, 'r') as f:
                data = json.load(f)
                self.assertIsInstance(data, dict)
    
    def test_file_locking_mechanism(self):
        """Test file locking prevents concurrent access."""
        lock_name = "compartment"
        
        # Test acquiring lock
        with self.register_manager.file_locks(lock_name):
            lock_file = self.register_manager.lock_files[lock_name]
            self.assertTrue(lock_file.exists())
            
            # Test that second lock attempt fails quickly
            start_time = time.time()
            success = self.register_manager._try_acquire_lock(lock_file)
            end_time = time.time()
            
            self.assertFalse(success)
            self.assertLess(end_time - start_time, 1.0)  # Should fail quickly
        
        # Lock should be released after context
        self.assertFalse(lock_file.exists())
    
    def test_concurrent_lock_acquisition(self):
        """Test file locking works correctly with multiple threads."""
        results = []
        errors = []
        
        def acquire_lock_thread(thread_id):
            try:
                with self.register_manager.file_locks("compartment"):
                    results.append(f"Thread {thread_id} acquired lock")
                    time.sleep(0.1)  # Hold lock briefly
                    results.append(f"Thread {thread_id} released lock")
            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=acquire_lock_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Check results
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertEqual(len(results), 6)  # 3 acquire + 3 release
        
        # Verify proper ordering (no overlapping critical sections)
        acquire_count = 0
        for result in results:
            if "acquired" in result:
                acquire_count += 1
            elif "released" in result:
                acquire_count -= 1
            
            # Should never have more than 1 thread in critical section
            self.assertLessEqual(acquire_count, 1)
    
    def test_data_loading_and_saving(self):
        """Test loading and saving register data."""
        test_data = {
            "AB1234": {
                "100": {
                    "HoleID": "AB1234",
                    "From": 95,
                    "To": 100,
                    "Photo_Status": "OK_Wet"
                }
            }
        }
        
        # Save test data
        with self.register_manager.file_locks("compartment"):
            with open(self.register_manager.files["compartment"], 'w') as f:
                json.dump(test_data, f, indent=2)
        
        # Load data back
        loaded_data = self.register_manager.load_register_data("compartment")
        
        self.assertEqual(loaded_data, test_data)
        self.assertEqual(loaded_data["AB1234"]["100"]["Photo_Status"], "OK_Wet")
    
    def test_register_backup_and_recovery(self):
        """Test register backup and recovery mechanisms."""
        # Create test data
        test_data = {"test": "data", "timestamp": time.time()}
        
        file_path = self.register_manager.files["compartment"]
        
        # Save original data
        with open(file_path, 'w') as f:
            json.dump(test_data, f)
        
        # Create backup
        backup_path = self.register_manager._create_backup(file_path)
        self.assertTrue(backup_path.exists())
        
        # Verify backup content
        with open(backup_path, 'r') as f:
            backup_data = json.load(f)
        
        self.assertEqual(backup_data, test_data)
        
        # Test restoration
        corrupted_data = {"corrupted": True}
        with open(file_path, 'w') as f:
            json.dump(corrupted_data, f)
        
        # Restore from backup
        success = self.register_manager._restore_from_backup(file_path, backup_path)
        self.assertTrue(success)
        
        # Verify restoration
        with open(file_path, 'r') as f:
            restored_data = json.load(f)
        
        self.assertEqual(restored_data, test_data)


class TestJsonRegisterManagerOperations(unittest.TestCase):
    """Test register operation methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = Mock()
        self.register_manager = JSONRegisterManager(self.temp_dir, self.logger)
        
        # Initialize with test data
        self._setup_test_data()
        
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _setup_test_data(self):
        """Set up test register data."""
        compartment_data = {
            "AB1234": {
                "100": {
                    "HoleID": "AB1234",
                    "From": 95,
                    "To": 100,
                    "Photo_Status": "Found",
                    "Source_Image_UID": "test-uid-1"
                },
                "105": {
                    "HoleID": "AB1234", 
                    "From": 100,
                    "To": 105,
                    "Photo_Status": "Empty",
                    "Source_Image_UID": "test-uid-1"
                }
            },
            "CD5678": {
                "200": {
                    "HoleID": "CD5678",
                    "From": 195,
                    "To": 200,
                    "Photo_Status": "OK_Dry",
                    "Source_Image_UID": "test-uid-2"
                }
            }
        }
        
        original_data = {
            "test-uid-1": {
                "UID": "test-uid-1",
                "HoleID": "AB1234",
                "From": 95,
                "To": 105,
                "Status": "Processed"
            },
            "test-uid-2": {
                "UID": "test-uid-2", 
                "HoleID": "CD5678",
                "From": 195,
                "To": 200,
                "Status": "Processed"
            }
        }
        
        # Save test data
        with open(self.register_manager.files["compartment"], 'w') as f:
            json.dump(compartment_data, f, indent=2)
            
        with open(self.register_manager.files["originalimages"], 'w') as f:
            json.dump(original_data, f, indent=2)
    
    def test_update_compartments_by_source_uid(self):
        """Test updating compartment records by source UID."""
        test_uid = "test-uid-1"
        new_status = "Reprocessed"
        
        updated_count = self.register_manager.update_compartments_by_source_uid(
            test_uid, new_status
        )
        
        # Should update 2 compartments for AB1234
        self.assertEqual(updated_count, 2)
        
        # Verify updates
        data = self.register_manager.load_register_data("compartment")
        self.assertEqual(data["AB1234"]["100"]["Photo_Status"], new_status)
        self.assertEqual(data["AB1234"]["105"]["Photo_Status"], new_status)
        
        # CD5678 should remain unchanged
        self.assertEqual(data["CD5678"]["200"]["Photo_Status"], "OK_Dry")
    
    def test_get_original_image_by_uid(self):
        """Test retrieving original image data by UID."""
        test_uid = "test-uid-1"
        
        original_data = self.register_manager.get_original_image_by_uid(test_uid)
        
        self.assertIsNotNone(original_data)
        self.assertEqual(original_data["UID"], test_uid)
        self.assertEqual(original_data["HoleID"], "AB1234")
        self.assertEqual(original_data["Status"], "Processed")
        
        # Test non-existent UID
        missing_data = self.register_manager.get_original_image_by_uid("nonexistent")
        self.assertIsNone(missing_data)
    
    def test_batch_update_compartments(self):
        """Test batch updating multiple compartment records."""
        updates = [
            {
                "hole_id": "AB1234",
                "depth": 100,
                "photo_status": "OK_Wet",
                "review_date": "2024-01-15"
            },
            {
                "hole_id": "CD5678", 
                "depth": 200,
                "photo_status": "Damaged",
                "review_date": "2024-01-15"
            }
        ]
        
        updated_count = self.register_manager.batch_update_compartments(updates)
        
        self.assertEqual(updated_count, 2)
        
        # Verify updates
        data = self.register_manager.load_register_data("compartment")
        self.assertEqual(data["AB1234"]["100"]["Photo_Status"], "OK_Wet")
        self.assertEqual(data["AB1234"]["100"]["Review_Date"], "2024-01-15")
        self.assertEqual(data["CD5678"]["200"]["Photo_Status"], "Damaged")
    
    def test_compartment_data_consistency(self):
        """Test data consistency checks and validation."""
        # Test data structure validation
        data = self.register_manager.load_register_data("compartment")
        
        for hole_id, hole_data in data.items():
            self.assertIsInstance(hole_data, dict)
            
            for depth_key, compartment in hole_data.items():
                # Check required fields
                required_fields = ["HoleID", "From", "To", "Photo_Status"]
                for field in required_fields:
                    self.assertIn(field, compartment)
                
                # Check data types
                self.assertIsInstance(compartment["From"], (int, float))
                self.assertIsInstance(compartment["To"], (int, float))
                self.assertIsInstance(compartment["Photo_Status"], str)
                
                # Check logical constraints
                self.assertLess(compartment["From"], compartment["To"])
                self.assertEqual(compartment["HoleID"], hole_id)
    
    def test_register_synchronization(self):
        """Test synchronization between different register files."""
        # Update original image status
        original_updates = {
            "test-uid-1": {
                "Status": "Reprocessed",
                "Reprocess_Date": "2024-01-15"
            }
        }
        
        # This should trigger compartment updates
        with self.register_manager.file_locks("originalimages"):
            data = self.register_manager.load_register_data("originalimages")
            data["test-uid-1"].update(original_updates["test-uid-1"])
            
            with open(self.register_manager.files["originalimages"], 'w') as f:
                json.dump(data, f, indent=2)
        
        # Verify compartments can be queried consistently
        compartment_data = self.register_manager.load_register_data("compartment")
        ab1234_compartments = [
            comp for comp in compartment_data["AB1234"].values() 
            if comp.get("Source_Image_UID") == "test-uid-1"
        ]
        
        self.assertEqual(len(ab1234_compartments), 2)
    
    def test_error_handling_corrupted_files(self):
        """Test error handling for corrupted register files."""
        # Corrupt a file
        corrupted_file = self.register_manager.files["compartment"]
        with open(corrupted_file, 'w') as f:
            f.write("invalid json content {")
        
        # Should handle gracefully
        data = self.register_manager.load_register_data("compartment")
        
        # Should return empty dict on corruption
        self.assertEqual(data, {})
        
        # Logger should be called with error
        self.logger.error.assert_called()
    
    def test_performance_with_large_datasets(self):
        """Test performance with larger datasets."""
        # Create large test dataset
        large_data = {}
        
        for i in range(100):  # 100 holes
            hole_id = f"TEST{i:04d}"
            large_data[hole_id] = {}
            
            for j in range(50):  # 50 compartments per hole
                depth = j * 5
                large_data[hole_id][str(depth)] = {
                    "HoleID": hole_id,
                    "From": depth - 2.5,
                    "To": depth + 2.5,
                    "Photo_Status": "Found",
                    "Source_Image_UID": f"uid-{i}-{j}"
                }
        
        # Time the save operation
        start_time = time.time()
        
        with open(self.register_manager.files["compartment"], 'w') as f:
            json.dump(large_data, f)
        
        save_time = time.time() - start_time
        
        # Time the load operation
        start_time = time.time()
        loaded_data = self.register_manager.load_register_data("compartment")
        load_time = time.time() - start_time
        
        # Performance assertions (should be reasonable)
        self.assertLess(save_time, 5.0)  # Should save within 5 seconds
        self.assertLess(load_time, 2.0)  # Should load within 2 seconds
        
        # Verify data integrity
        self.assertEqual(len(loaded_data), 100)
        self.assertEqual(len(loaded_data["TEST0000"]), 50)


if __name__ == '__main__':
    unittest.main()