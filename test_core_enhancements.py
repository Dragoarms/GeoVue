#!/usr/bin/env python3
"""
Core unit tests for JSON register manager enhancements.
Tests only the core functionality without Excel/COM dependencies.
"""

import unittest
import os
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.json_register_manager import (
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


class TestCompartmentProcessingMetadata(unittest.TestCase):
    """Test CompartmentProcessingMetadata dataclass functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_user = "test_user"
        self.test_timestamp = datetime.now().isoformat()
    
    def test_create_default_valid_status(self):
        """Test creation with valid status."""
        metadata = CompartmentProcessingMetadata.create_default(
            photo_status=PhotoStatus.APPROVED.value,
            processed_by=self.test_user
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
    
    def test_to_dict_serialization(self):
        """Test serialization to dictionary."""
        metadata = CompartmentProcessingMetadata(
            photo_status=PhotoStatus.FOR_REVIEW.value,
            processed_date=self.test_timestamp,
            processed_by=self.test_user,
            comments="Test comment",
            image_width_cm=2.5
        )
        
        data_dict = metadata.to_dict()
        
        self.assertEqual(data_dict["Photo_Status"], PhotoStatus.FOR_REVIEW.value)
        self.assertEqual(data_dict["Processed_Date"], self.test_timestamp)
        self.assertEqual(data_dict["Processed_By"], self.test_user)
        self.assertEqual(data_dict["Comments"], "Test comment")
        self.assertEqual(data_dict["Image_Width_Cm"], 2.5)
    
    def test_from_dict_complete_data(self):
        """Test creating from dictionary with complete data."""
        data = {
            "Photo_Status": PhotoStatus.REJECTED.value,
            "Processed_Date": self.test_timestamp,
            "Processed_By": self.test_user,
            "Comments": "Test rejection",
            "Image_Width_Cm": 1.8
        }
        
        metadata = CompartmentProcessingMetadata.from_dict(data)
        
        self.assertEqual(metadata.photo_status, PhotoStatus.REJECTED.value)
        self.assertEqual(metadata.processed_date, self.test_timestamp)
        self.assertEqual(metadata.processed_by, self.test_user)
        self.assertEqual(metadata.comments, "Test rejection")
        self.assertEqual(metadata.image_width_cm, 1.8)
    
    def test_from_dict_minimal_data(self):
        """Test creating from dictionary with minimal data."""
        data = {}
        
        metadata = CompartmentProcessingMetadata.from_dict(data)
        
        self.assertEqual(metadata.photo_status, PhotoStatus.FOR_REVIEW.value)
        self.assertIsNotNone(metadata.processed_date)
        self.assertIsNotNone(metadata.processed_by)
        self.assertIsNone(metadata.comments)
        self.assertIsNone(metadata.image_width_cm)
    
    def test_validate_success(self):
        """Test validation succeeds with valid data."""
        metadata = CompartmentProcessingMetadata.create_default()
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
            photo_status=PhotoStatus.APPROVED.value,
            processed_by="test_user"
        )
        original.comments = "Test comment"
        original.image_width_cm = 2.0
        
        # Serialize and deserialize
        data_dict = original.to_dict()
        restored = CompartmentProcessingMetadata.from_dict(data_dict)
        
        # Compare all fields
        self.assertEqual(original.photo_status, restored.photo_status)
        self.assertEqual(original.processed_date, restored.processed_date)
        self.assertEqual(original.processed_by, restored.processed_by)
        self.assertEqual(original.comments, restored.comments)
        self.assertEqual(original.image_width_cm, restored.image_width_cm)


class TestCornerRecordConfig(unittest.TestCase):
    """Test CornerRecordConfig dataclass functionality."""
    
    def test_corner_record_creation(self):
        """Test creating a corner record configuration."""
        corners = [[100, 100], [200, 100], [200, 200], [100, 200]]
        metadata = CompartmentProcessingMetadata.create_default()
        
        config = CornerRecordConfig(
            hole_id="TEST001",
            depth_from=100,
            depth_to=101,
            original_filename="test.jpg",
            compartment_num=1,
            corners=corners,
            processing_metadata=metadata,
            scale_px_per_cm=50.0,
            scale_confidence=0.95,
            source_image_uid="test-uid-123"
        )
        
        self.assertEqual(config.hole_id, "TEST001")
        self.assertEqual(config.depth_from, 100)
        self.assertEqual(config.depth_to, 101)
        self.assertEqual(config.original_filename, "test.jpg")
        self.assertEqual(config.compartment_num, 1)
        self.assertEqual(config.corners, corners)
        self.assertEqual(config.processing_metadata, metadata)
        self.assertEqual(config.scale_px_per_cm, 50.0)
        self.assertEqual(config.scale_confidence, 0.95)
        self.assertEqual(config.source_image_uid, "test-uid-123")
    
    def test_corner_record_optional_fields(self):
        """Test corner record with optional fields as None."""
        corners = [[0, 0], [100, 0], [100, 100], [0, 100]]
        
        config = CornerRecordConfig(
            hole_id="TEST002",
            depth_from=200,
            depth_to=201,
            original_filename="test2.jpg",
            compartment_num=2,
            corners=corners
        )
        
        self.assertIsNone(config.processing_metadata)
        self.assertIsNone(config.scale_px_per_cm)
        self.assertIsNone(config.scale_confidence)
        self.assertIsNone(config.source_image_uid)


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPhotoStatusEnum))
    suite.addTests(loader.loadTestsFromTestCase(TestCompartmentProcessingMetadata))
    suite.addTests(loader.loadTestsFromTestCase(TestCornerRecordConfig))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.wasSuccessful():
        print("\n[SUCCESS] All core enhancement tests passed!")
    else:
        print("\n[FAILED] Some tests failed. Please review the output above.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)