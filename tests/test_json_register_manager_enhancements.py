#!/usr/bin/env python3
"""
Comprehensive test suite for JSON register manager enhancements.

This test suite validates:
1. PhotoStatus enum functionality
2. CompartmentProcessingMetadata creation, validation, serialization
3. CornerRecordConfig functionality
4. Corner processing metadata integration
5. Backward compatibility with existing data
6. Cleanup of placeholder records
7. Thread safety of new operations
8. Data validation and error handling
9. Integration with existing compartment/corner workflows
10. UID consistency improvements
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
from datetime import datetime
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.json_register_manager import (
    JSONRegisterManager, 
    PhotoStatus, 
    CompartmentProcessingMetadata,
    CornerRecordConfig,
    DataValidationError
)


class TestPhotoStatusEnum(unittest.TestCase):
    """Test PhotoStatus enum functionality."""
    
    def test_photo_status_values(self):
        """Test that PhotoStatus enum has expected values."""
        self.assertEqual(PhotoStatus.FOR_REVIEW.value, "For Review")
        self.assertEqual(PhotoStatus.APPROVED.value, "Approved")
        self.assertEqual(PhotoStatus.REJECTED.value, "Rejected")
        self.assertEqual(PhotoStatus.IN_PROGRESS.value, "In Progress")
    
    def test_photo_status_enum_members(self):
        """Test that PhotoStatus enum has all expected members."""
        expected_members = {"FOR_REVIEW", "APPROVED", "REJECTED", "IN_PROGRESS"}
        actual_members = {member.name for member in PhotoStatus}
        self.assertEqual(actual_members, expected_members)
    
    def test_photo_status_string_conversion(self):
        """Test string representation of PhotoStatus values."""
        self.assertEqual(str(PhotoStatus.FOR_REVIEW.value), "For Review")
        self.assertEqual(str(PhotoStatus.APPROVED.value), "Approved")
        
    def test_photo_status_iteration(self):
        """Test iteration over PhotoStatus enum."""
        values = [status.value for status in PhotoStatus]
        expected_values = ["For Review", "Approved", "Rejected", "In Progress"]
        self.assertEqual(set(values), set(expected_values))


class TestCompartmentProcessingMetadata(unittest.TestCase):
    """Test CompartmentProcessingMetadata dataclass functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_user = "test_user"
        self.test_timestamp = datetime.now().isoformat()
    
    def test_create_default_valid_status(self):
        """Test creation of default metadata with valid status."""
        metadata = CompartmentProcessingMetadata.create_default(
            PhotoStatus.APPROVED.value, 
            self.test_user
        )
        
        self.assertEqual(metadata.photo_status, PhotoStatus.APPROVED.value)
        self.assertEqual(metadata.processed_by, self.test_user)
        self.assertIsNotNone(metadata.processed_date)
        self.assertIsNone(metadata.comments)
        self.assertIsNone(metadata.image_width_cm)
    
    def test_create_default_invalid_status(self):
        """Test creation fails with invalid status."""
        with self.assertRaises(ValueError) as context:
            CompartmentProcessingMetadata.create_default("Invalid Status")
        
        self.assertIn("Invalid photo_status", str(context.exception))
        self.assertIn("Invalid Status", str(context.exception))
    
    def test_create_default_no_user(self):
        """Test creation with no user specified uses environment variable."""
        with patch.dict(os.environ, {'USERNAME': 'env_user'}):
            metadata = CompartmentProcessingMetadata.create_default()
            self.assertEqual(metadata.processed_by, 'env_user')
    
    def test_create_default_no_username_env(self):
        """Test creation with no USERNAME environment variable."""
        with patch.dict(os.environ, {}, clear=True):
            if 'USERNAME' in os.environ:
                del os.environ['USERNAME']
            metadata = CompartmentProcessingMetadata.create_default()
            self.assertEqual(metadata.processed_by, 'Unknown')
    
    def test_to_dict_serialization(self):
        """Test serialization to dictionary."""
        metadata = CompartmentProcessingMetadata(
            photo_status=PhotoStatus.APPROVED.value,
            processed_date=self.test_timestamp,
            processed_by=self.test_user,
            comments="Test comment",
            image_width_cm=25.4
        )
        
        result = metadata.to_dict()
        expected = {
            "Photo_Status": PhotoStatus.APPROVED.value,
            "Processed_Date": self.test_timestamp,
            "Processed_By": self.test_user,
            "Comments": "Test comment",
            "Image_Width_Cm": 25.4
        }
        
        self.assertEqual(result, expected)
    
    def test_to_dict_none_values(self):
        """Test serialization with None values."""
        metadata = CompartmentProcessingMetadata(
            photo_status=PhotoStatus.FOR_REVIEW.value,
            processed_date=self.test_timestamp,
            processed_by=self.test_user
        )
        
        result = metadata.to_dict()
        self.assertIsNone(result["Comments"])
        self.assertIsNone(result["Image_Width_Cm"])
    
    def test_from_dict_complete_data(self):
        """Test deserialization from complete dictionary."""
        data = {
            "Photo_Status": PhotoStatus.APPROVED.value,
            "Processed_Date": self.test_timestamp,
            "Processed_By": self.test_user,
            "Comments": "Test comment",
            "Image_Width_Cm": 25.4
        }
        
        metadata = CompartmentProcessingMetadata.from_dict(data)
        
        self.assertEqual(metadata.photo_status, PhotoStatus.APPROVED.value)
        self.assertEqual(metadata.processed_date, self.test_timestamp)
        self.assertEqual(metadata.processed_by, self.test_user)
        self.assertEqual(metadata.comments, "Test comment")
        self.assertEqual(metadata.image_width_cm, 25.4)
    
    def test_from_dict_minimal_data(self):
        """Test deserialization from minimal dictionary with defaults."""
        data = {"Photo_Status": PhotoStatus.REJECTED.value}
        
        metadata = CompartmentProcessingMetadata.from_dict(data)
        
        self.assertEqual(metadata.photo_status, PhotoStatus.REJECTED.value)
        self.assertIsNotNone(metadata.processed_date)
        self.assertIsNotNone(metadata.processed_by)
        self.assertIsNone(metadata.comments)
        self.assertIsNone(metadata.image_width_cm)
    
    def test_from_dict_empty_data(self):
        """Test deserialization from empty dictionary uses defaults."""
        data = {}
        
        metadata = CompartmentProcessingMetadata.from_dict(data)
        
        self.assertEqual(metadata.photo_status, PhotoStatus.FOR_REVIEW.value)
        self.assertIsNotNone(metadata.processed_date)
        self.assertIsNotNone(metadata.processed_by)
    
    def test_validate_success(self):
        """Test validation passes for valid metadata."""
        metadata = CompartmentProcessingMetadata(
            photo_status=PhotoStatus.APPROVED.value,
            processed_date=self.test_timestamp,
            processed_by=self.test_user
        )
        
        # Should not raise any exception
        metadata.validate()
    
    def test_validate_empty_status(self):
        """Test validation fails for empty status."""
        metadata = CompartmentProcessingMetadata(
            photo_status="",
            processed_date=self.test_timestamp,
            processed_by=self.test_user
        )
        
        with self.assertRaises(DataValidationError) as context:
            metadata.validate()
        
        self.assertIn("photo_status cannot be empty", str(context.exception))
    
    def test_validate_empty_processed_by(self):
        """Test validation fails for empty processed_by."""
        metadata = CompartmentProcessingMetadata(
            photo_status=PhotoStatus.APPROVED.value,
            processed_date=self.test_timestamp,
            processed_by=""
        )
        
        with self.assertRaises(DataValidationError) as context:
            metadata.validate()
        
        self.assertIn("processed_by cannot be empty", str(context.exception))
    
    def test_validate_invalid_status(self):
        """Test validation fails for invalid status."""
        metadata = CompartmentProcessingMetadata(
            photo_status="Invalid Status",
            processed_date=self.test_timestamp,
            processed_by=self.test_user
        )
        
        with self.assertRaises(DataValidationError) as context:
            metadata.validate()
        
        self.assertIn("Invalid photo_status", str(context.exception))
    
    def test_roundtrip_serialization(self):
        """Test that serialization and deserialization are symmetric."""
        original = CompartmentProcessingMetadata.create_default(
            PhotoStatus.IN_PROGRESS.value,
            self.test_user
        )
        original.comments = "Test roundtrip"
        original.image_width_cm = 30.0
        
        # Serialize and deserialize
        serialized = original.to_dict()
        restored = CompartmentProcessingMetadata.from_dict(serialized)
        
        self.assertEqual(original.photo_status, restored.photo_status)
        self.assertEqual(original.processed_date, restored.processed_date)
        self.assertEqual(original.processed_by, restored.processed_by)
        self.assertEqual(original.comments, restored.comments)
        self.assertEqual(original.image_width_cm, restored.image_width_cm)


class TestCornerRecordConfig(unittest.TestCase):
    """Test CornerRecordConfig dataclass functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.sample_corners = [[100, 200], [300, 200], [300, 400], [100, 400]]
        self.sample_config = CornerRecordConfig(
            hole_id="TEST001",
            depth_from=10,
            depth_to=20,
            original_filename="test_image.jpg",
            compartment_num=1,
            corners=self.sample_corners
        )
    
    def test_corner_record_config_creation(self):
        """Test basic creation of CornerRecordConfig."""
        self.assertEqual(self.sample_config.hole_id, "TEST001")
        self.assertEqual(self.sample_config.depth_from, 10)
        self.assertEqual(self.sample_config.depth_to, 20)
        self.assertEqual(self.sample_config.original_filename, "test_image.jpg")
        self.assertEqual(self.sample_config.compartment_num, 1)
        self.assertEqual(self.sample_config.corners, self.sample_corners)
        self.assertIsNone(self.sample_config.processing_metadata)
        self.assertIsNone(self.sample_config.scale_px_per_cm)
        self.assertIsNone(self.sample_config.scale_confidence)
        self.assertIsNone(self.sample_config.source_image_uid)
    
    def test_corner_record_config_with_metadata(self):
        """Test creation with processing metadata."""
        metadata = CompartmentProcessingMetadata.create_default()
        config = CornerRecordConfig(
            hole_id="TEST001",
            depth_from=10,
            depth_to=20,
            original_filename="test_image.jpg",
            compartment_num=1,
            corners=self.sample_corners,
            processing_metadata=metadata
        )
        
        self.assertEqual(config.processing_metadata, metadata)
    
    def test_corner_record_config_with_scale_info(self):
        """Test creation with scale information."""
        config = CornerRecordConfig(
            hole_id="TEST001",
            depth_from=10,
            depth_to=20,
            original_filename="test_image.jpg",
            compartment_num=1,
            corners=self.sample_corners,
            scale_px_per_cm=50.0,
            scale_confidence=0.95
        )
        
        self.assertEqual(config.scale_px_per_cm, 50.0)
        self.assertEqual(config.scale_confidence, 0.95)
    
    def test_corner_record_config_with_source_uid(self):
        """Test creation with source image UID."""
        test_uid = str(uuid.uuid4())
        config = CornerRecordConfig(
            hole_id="TEST001",
            depth_from=10,
            depth_to=20,
            original_filename="test_image.jpg",
            compartment_num=1,
            corners=self.sample_corners,
            source_image_uid=test_uid
        )
        
        self.assertEqual(config.source_image_uid, test_uid)


class TestJSONRegisterManagerEnhancements(unittest.TestCase):
    """Test JSONRegisterManager enhancement functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = Mock()
        self.manager = JSONRegisterManager(self.temp_dir, self.logger)
        
        # Create test data directory
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @contextmanager
    def _create_test_data_with_placeholders(self):
        """Create test data files with INITIALISING placeholder records."""
        compartment_data = [
            {"HoleID": "INITIALISING", "Depth_From": 0, "Depth_To": 0},
            {"HoleID": "TEST001", "Depth_From": 10, "Depth_To": 20, "Photo_Status": "For Review"}
        ]
        
        original_data = [
            {"HoleID": "INITIALISING", "Original_Filename": "placeholder.jpg"},
            {"HoleID": "TEST001", "Original_Filename": "test.jpg", "UID": str(uuid.uuid4())}
        ]
        
        review_data = [
            {"HoleID": "INITIALISING", "Comments": "Placeholder"},
            {"HoleID": "TEST001", "Comments": "Test comment"}
        ]
        
        corner_data = [
            {"HoleID": "INITIALISING", "Corner_1_X": 0, "Corner_1_Y": 0},
            {"HoleID": "TEST001", "Corner_1_X": 100, "Corner_1_Y": 200, "Photo_Status": "For Review"}
        ]
        
        # Write test data
        with open(self.manager.compartment_json_path, 'w') as f:
            json.dump(compartment_data, f)
        with open(self.manager.original_json_path, 'w') as f:
            json.dump(original_data, f)
        with open(self.manager.review_json_path, 'w') as f:
            json.dump(review_data, f)
        with open(self.manager.compartment_corners_json_path, 'w') as f:
            json.dump(corner_data, f)
            
        yield
    
    def test_cleanup_placeholder_records(self):
        """Test cleanup of INITIALISING placeholder records."""
        with self._create_test_data_with_placeholders():
            # Perform cleanup
            cleanup_stats = self.manager.cleanup_placeholder_records()
            
            # Verify cleanup stats
            self.assertEqual(cleanup_stats["compartments"], 1)
            self.assertEqual(cleanup_stats["originals"], 1)
            self.assertEqual(cleanup_stats["reviews"], 1)
            self.assertEqual(cleanup_stats["corners"], 1)
            
            # Verify data files no longer contain INITIALISING records
            with open(self.manager.compartment_json_path, 'r') as f:
                compartment_data = json.load(f)
            self.assertTrue(all(record.get("HoleID") != "INITIALISING" for record in compartment_data))
            
            with open(self.manager.original_json_path, 'r') as f:
                original_data = json.load(f)
            self.assertTrue(all(record.get("HoleID") != "INITIALISING" for record in original_data))
            
            with open(self.manager.review_json_path, 'r') as f:
                review_data = json.load(f)
            self.assertTrue(all(record.get("HoleID") != "INITIALISING" for record in review_data))
            
            with open(self.manager.compartment_corners_json_path, 'r') as f:
                corner_data = json.load(f)
            self.assertTrue(all(record.get("HoleID") != "INITIALISING" for record in corner_data))
    
    def test_cleanup_placeholder_records_no_placeholders(self):
        """Test cleanup when no placeholder records exist."""
        # Initialize files without placeholders
        self.manager.initialize_register_files()
        
        cleanup_stats = self.manager.cleanup_placeholder_records()
        
        # Should report no records cleaned
        self.assertEqual(cleanup_stats["compartments"], 0)
        self.assertEqual(cleanup_stats["originals"], 0)
        self.assertEqual(cleanup_stats["reviews"], 0)
        self.assertEqual(cleanup_stats["corners"], 0)
    
    def test_corner_processing_metadata_integration(self):
        """Test integration of corner processing metadata."""
        self.manager.initialize_register_files()
        
        # Create corner record with processing metadata
        metadata = CompartmentProcessingMetadata.create_default(
            PhotoStatus.APPROVED.value,
            "test_user"
        )
        metadata.comments = "Test integration"
        
        result = self.manager.add_compartment_corners_record(
            hole_id="TEST001",
            depth_from=10,
            depth_to=20,
            original_filename="test.jpg",
            compartment_num=1,
            corners=[[100, 200], [300, 200], [300, 400], [100, 400]],
            processing_metadata=metadata
        )
        
        self.assertTrue(result)
        
        # Verify metadata was stored correctly
        stored_metadata = self.manager.get_corner_processing_metadata(
            "TEST001", 10, 20, "test.jpg", 1
        )
        
        self.assertIsNotNone(stored_metadata)
        self.assertEqual(stored_metadata.photo_status, PhotoStatus.APPROVED.value)
        self.assertEqual(stored_metadata.processed_by, "test_user")
        self.assertEqual(stored_metadata.comments, "Test integration")
    
    def test_update_corner_processing_metadata(self):
        """Test updating corner processing metadata."""
        self.manager.initialize_register_files()
        
        # Add initial corner record
        self.manager.add_compartment_corners_record(
            hole_id="TEST001",
            depth_from=10,
            depth_to=20,
            original_filename="test.jpg",
            compartment_num=1,
            corners=[[100, 200], [300, 200], [300, 400], [100, 400]]
        )
        
        # Update metadata
        new_metadata = CompartmentProcessingMetadata.create_default(
            PhotoStatus.REJECTED.value,
            "reviewer"
        )
        new_metadata.comments = "Needs rework"
        
        result = self.manager.update_corner_processing_metadata(
            hole_id="TEST001",
            depth_from=10,
            depth_to=20,
            original_filename="test.jpg",
            compartment_num=1,
            processing_metadata=new_metadata
        )
        
        self.assertTrue(result)
        
        # Verify update
        stored_metadata = self.manager.get_corner_processing_metadata(
            "TEST001", 10, 20, "test.jpg", 1
        )
        
        self.assertEqual(stored_metadata.photo_status, PhotoStatus.REJECTED.value)
        self.assertEqual(stored_metadata.processed_by, "reviewer")
        self.assertEqual(stored_metadata.comments, "Needs rework")
    
    def test_batch_update_corner_processing_metadata(self):
        """Test batch updating of corner processing metadata."""
        self.manager.initialize_register_files()
        
        # Add multiple corner records
        for i in range(3):
            self.manager.add_compartment_corners_record(
                hole_id="TEST001",
                depth_from=i*10,
                depth_to=(i+1)*10,
                original_filename="test.jpg",
                compartment_num=i+1,
                corners=[[100+i*10, 200], [300+i*10, 200], [300+i*10, 400], [100+i*10, 400]]
            )
        
        # Batch update
        updates = [
            {
                "hole_id": "TEST001",
                "depth_from": 0,
                "depth_to": 10,
                "original_filename": "test.jpg",
                "compartment_num": 1,
                "processing_metadata": CompartmentProcessingMetadata.create_default(
                    PhotoStatus.APPROVED.value, "batch_user"
                )
            },
            {
                "hole_id": "TEST001",
                "depth_from": 10,
                "depth_to": 20,
                "original_filename": "test.jpg",
                "compartment_num": 2,
                "processing_metadata": CompartmentProcessingMetadata.create_default(
                    PhotoStatus.REJECTED.value, "batch_user"
                )
            }
        ]
        
        result = self.manager.batch_update_corner_processing_metadata(updates)
        self.assertEqual(result, 2)
        
        # Verify updates
        metadata1 = self.manager.get_corner_processing_metadata("TEST001", 0, 10, "test.jpg", 1)
        metadata2 = self.manager.get_corner_processing_metadata("TEST001", 10, 20, "test.jpg", 2)
        
        self.assertEqual(metadata1.photo_status, PhotoStatus.APPROVED.value)
        self.assertEqual(metadata2.photo_status, PhotoStatus.REJECTED.value)
    
    def test_validate_corner_data_consistency(self):
        """Test corner data consistency validation."""
        self.manager.initialize_register_files()
        
        # Add valid corner record
        self.manager.add_compartment_corners_record(
            hole_id="TEST001",
            depth_from=10,
            depth_to=20,
            original_filename="test.jpg",
            compartment_num=1,
            corners=[[100, 200], [300, 200], [300, 400], [100, 400]]
        )
        
        # Manually add invalid record (missing coordinates)
        with open(self.manager.compartment_corners_json_path, 'r') as f:
            data = json.load(f)
        
        data.append({
            "HoleID": "TEST002",
            "Depth_From": 20,
            "Depth_To": 30,
            "Original_Filename": "test2.jpg",
            "Compartment": 1,
            "Photo_Status": "For Review",
            # Missing corner coordinates
        })
        
        with open(self.manager.compartment_corners_json_path, 'w') as f:
            json.dump(data, f)
        
        # Validate consistency
        results = self.manager.validate_corner_data_consistency()
        
        self.assertFalse(results["validation_passed"])
        self.assertEqual(results["records_missing_coordinate_fields"], 1)
        self.assertGreater(len(results["consistency_issues"]), 0)
    
    def test_uid_consistency_in_operations(self):
        """Test UID consistency across operations."""
        self.manager.initialize_register_files()
        
        test_uid = str(uuid.uuid4())
        
        # Add original image with UID
        self.manager.add_original_image_record(
            hole_id="TEST001",
            depth_from=10,
            depth_to=20,
            original_filename="test.jpg",
            uid=test_uid
        )
        
        # Add corner record with same UID
        self.manager.add_compartment_corners_record(
            hole_id="TEST001",
            depth_from=10,
            depth_to=20,
            original_filename="test.jpg",
            compartment_num=1,
            corners=[[100, 200], [300, 200], [300, 400], [100, 400]],
            source_image_uid=test_uid
        )
        
        # Verify UID consistency
        original_record = self.manager.get_original_image_by_uid(test_uid)
        self.assertIsNotNone(original_record)
        self.assertEqual(original_record["UID"], test_uid)
        
        corners_data = self.manager.get_compartment_corners_data()
        corner_record = corners_data[corners_data["HoleID"] == "TEST001"].iloc[0]
        self.assertEqual(corner_record["Source_Image_UID"], test_uid)
    
    def test_backward_compatibility_existing_data(self):
        """Test backward compatibility with existing data structures."""
        # Create old-style data without new fields
        old_compartment_data = [
            {
                "HoleID": "TEST001",
                "Depth_From": 10,
                "Depth_To": 20,
                "Original_Filename": "test.jpg",
                "Compartment": 1,
                # Missing processing metadata fields
            }
        ]
        
        old_corner_data = [
            {
                "HoleID": "TEST001",
                "Depth_From": 10,
                "Depth_To": 20,
                "Original_Filename": "test.jpg",
                "Compartment": 1,
                "Corner_1_X": 100,
                "Corner_1_Y": 200,
                "Corner_2_X": 300,
                "Corner_2_Y": 200,
                "Corner_3_X": 300,
                "Corner_3_Y": 400,
                "Corner_4_X": 100,
                "Corner_4_Y": 400,
                # Missing processing metadata fields
            }
        ]
        
        # Write old-style data
        with open(self.manager.compartment_json_path, 'w') as f:
            json.dump(old_compartment_data, f)
        with open(self.manager.compartment_corners_json_path, 'w') as f:
            json.dump(old_corner_data, f)
        
        # Test that manager can read and handle old data
        compartment_df = self.manager.get_compartment_data()
        self.assertEqual(len(compartment_df), 1)
        
        corners_df = self.manager.get_compartment_corners_data()
        self.assertEqual(len(corners_df), 1)
        
        # Test that new metadata can be added to old records
        result = self.manager.update_corner_processing_metadata(
            hole_id="TEST001",
            depth_from=10,
            depth_to=20,
            original_filename="test.jpg",
            compartment_num=1,
            processing_metadata=CompartmentProcessingMetadata.create_default()
        )
        
        self.assertTrue(result)


class TestThreadSafetyEnhancements(unittest.TestCase):
    """Test thread safety of new operations."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = Mock()
        self.manager = JSONRegisterManager(self.temp_dir, self.logger)
        self.manager.initialize_register_files()
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_concurrent_corner_metadata_updates(self):
        """Test concurrent corner metadata updates."""
        # Add initial corner record
        self.manager.add_compartment_corners_record(
            hole_id="TEST001",
            depth_from=10,
            depth_to=20,
            original_filename="test.jpg",
            compartment_num=1,
            corners=[[100, 200], [300, 200], [300, 400], [100, 400]]
        )
        
        def update_metadata(status, user):
            """Update metadata with specific status and user."""
            metadata = CompartmentProcessingMetadata.create_default(status, user)
            return self.manager.update_corner_processing_metadata(
                hole_id="TEST001",
                depth_from=10,
                depth_to=20,
                original_filename="test.jpg",
                compartment_num=1,
                processing_metadata=metadata
            )
        
        # Run concurrent updates
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(update_metadata, PhotoStatus.APPROVED.value, "user1"),
                executor.submit(update_metadata, PhotoStatus.REJECTED.value, "user2"),
                executor.submit(update_metadata, PhotoStatus.IN_PROGRESS.value, "user3")
            ]
            
            results = [future.result() for future in futures]
        
        # All updates should succeed
        self.assertTrue(all(results))
        
        # Final state should be consistent
        final_metadata = self.manager.get_corner_processing_metadata(
            "TEST001", 10, 20, "test.jpg", 1
        )
        self.assertIsNotNone(final_metadata)
        self.assertIn(final_metadata.photo_status, [s.value for s in PhotoStatus])
    
    def test_concurrent_placeholder_cleanup(self):
        """Test concurrent placeholder cleanup operations."""
        # Create multiple manager instances for the same directory
        managers = [
            JSONRegisterManager(self.temp_dir, Mock()) for _ in range(3)
        ]
        
        # Add placeholder data to each manager's view
        for manager in managers:
            with open(manager.compartment_json_path, 'w') as f:
                json.dump([
                    {"HoleID": "INITIALISING", "Depth_From": 0},
                    {"HoleID": "TEST001", "Depth_From": 10}
                ], f)
        
        def cleanup_placeholders(manager):
            """Clean up placeholders."""
            return manager.cleanup_placeholder_records()
        
        # Run concurrent cleanups
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(cleanup_placeholders, manager) 
                for manager in managers
            ]
            
            results = [future.result() for future in futures]
        
        # At least one should succeed, others might get locks
        successful_cleanups = [r for r in results if r["compartments"] > 0]
        self.assertGreaterEqual(len(successful_cleanups), 1)
        
        # Final result should be clean
        final_manager = JSONRegisterManager(self.temp_dir, Mock())
        final_stats = final_manager.cleanup_placeholder_records()
        self.assertEqual(final_stats["compartments"], 0)  # No more placeholders
    
    def test_concurrent_validation_operations(self):
        """Test concurrent validation operations."""
        # Add test data
        for i in range(5):
            self.manager.add_compartment_corners_record(
                hole_id=f"TEST{i:03d}",
                depth_from=i*10,
                depth_to=(i+1)*10,
                original_filename=f"test{i}.jpg",
                compartment_num=1,
                corners=[[100, 200], [300, 200], [300, 400], [100, 400]]
            )
        
        def validate_data():
            """Validate corner data consistency."""
            return self.manager.validate_corner_data_consistency()
        
        # Run concurrent validations
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(validate_data) for _ in range(3)]
            results = [future.result() for future in futures]
        
        # All validations should succeed and return consistent results
        self.assertTrue(all(r["validation_passed"] for r in results))
        self.assertTrue(all(r["total_records"] == results[0]["total_records"] for r in results))


class TestDataValidationAndErrorHandling(unittest.TestCase):
    """Test data validation and error handling for enhancements."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = Mock()
        self.manager = JSONRegisterManager(self.temp_dir, self.logger)
        self.manager.initialize_register_files()
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_invalid_corner_processing_metadata(self):
        """Test handling of invalid corner processing metadata."""
        # Test with invalid status
        invalid_metadata = CompartmentProcessingMetadata(
            photo_status="Invalid Status",
            processed_date=datetime.now().isoformat(),
            processed_by="test_user"
        )
        
        with self.assertRaises(ValueError):
            invalid_metadata.validate()
    
    def test_corner_record_with_invalid_corners(self):
        """Test corner record creation with invalid corner data."""
        # Test with insufficient corner points
        invalid_corners = [[100, 200], [300, 200]]  # Only 2 points
        
        # This should not raise an error in CornerRecordConfig creation
        # but should be handled appropriately by the manager
        config = CornerRecordConfig(
            hole_id="TEST001",
            depth_from=10,
            depth_to=20,
            original_filename="test.jpg",
            compartment_num=1,
            corners=invalid_corners
        )
        
        self.assertEqual(len(config.corners), 2)
        
        # The manager should handle this gracefully when adding the record
        result = self.manager.add_compartment_corners_record(
            hole_id="TEST001",
            depth_from=10,
            depth_to=20,
            original_filename="test.jpg",
            compartment_num=1,
            corners=invalid_corners
        )
        
        # Should still succeed - validation happens at different level
        self.assertTrue(result)
    
    def test_missing_required_fields_handling(self):
        """Test handling of missing required fields."""
        # Try to create metadata with None values
        with self.assertRaises(TypeError):
            CompartmentProcessingMetadata(
                photo_status=None,
                processed_date=datetime.now().isoformat(),
                processed_by="test_user"
            )
    
    def test_file_corruption_resilience(self):
        """Test resilience to file corruption."""
        # Corrupt the corner data file
        with open(self.manager.compartment_corners_json_path, 'w') as f:
            f.write("invalid json content")
        
        # Operations should handle corruption gracefully
        try:
            result = self.manager.get_compartment_corners_data()
            # Should return empty DataFrame or handle gracefully
            self.assertIsNotNone(result)
        except Exception as e:
            # Should log error but not crash
            self.logger.error.assert_called()
    
    def test_lock_timeout_handling(self):
        """Test handling of lock timeouts."""
        # Create a lock file that won't be released
        lock_path = self.manager.corners_lock
        lock_path.write_text(json.dumps({
            "pid": 99999,  # Non-existent PID
            "timestamp": time.time(),
            "user": "test_user"
        }))
        
        # Operations should either wait or handle timeout appropriately
        start_time = time.time()
        
        try:
            # This should either succeed after timeout or fail gracefully
            result = self.manager.add_compartment_corners_record(
                hole_id="TEST001",
                depth_from=10,
                depth_to=20,
                original_filename="test.jpg",
                compartment_num=1,
                corners=[[100, 200], [300, 200], [300, 400], [100, 400]]
            )
            
            # If it succeeds, it should have taken some time to acquire lock
            # or the lock should have been determined stale
            self.assertIsNotNone(result)
            
        except RuntimeError as e:
            # Should fail with appropriate error message
            self.assertIn("lock", str(e).lower())
        
        elapsed_time = time.time() - start_time
        self.assertLess(elapsed_time, 60)  # Should not hang indefinitely


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)