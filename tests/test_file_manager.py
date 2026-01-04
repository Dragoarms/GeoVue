#!/usr/bin/env python3
"""
Comprehensive test suite for FileManager class.
Tests all UID methods, PNG embedding, and core file operations.
"""

import sys
import os
import unittest
import tempfile
import shutil
import uuid
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import piexif
from PIL import Image as PILImage

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.file_manager import FileManager


class TestFileManagerUID(unittest.TestCase):
    """Test suite for FileManager UID functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = Mock()
        self.config_manager.get.return_value = None  # Return None for shared_folder_path
        self.file_manager = FileManager(self.temp_dir, self.config_manager)
        
        # Create test images
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.test_image[:] = [255, 0, 0]  # Red image
        
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_embed_uid_in_jpeg(self):
        """Test UID embedding in JPEG images."""
        jpeg_path = os.path.join(self.temp_dir, "test.jpg")
        cv2.imwrite(jpeg_path, self.test_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        test_uid = str(uuid.uuid4())
        embedded_uid = self.file_manager.embed_uid_in_image(jpeg_path, test_uid)
        
        self.assertEqual(embedded_uid, test_uid)
        
        # Verify extraction
        extracted_uid = self.file_manager.extract_uid_from_image(jpeg_path)
        self.assertEqual(extracted_uid, test_uid)
    
    def test_embed_uid_in_png_array(self):
        """Test UID embedding in PNG from numpy array."""
        test_uid = str(uuid.uuid4())
        
        png_bytes = self.file_manager.embed_uid_in_png(self.test_image, test_uid)
        self.assertIsNotNone(png_bytes)
        self.assertIsInstance(png_bytes, bytes)
        
        # Save to temp file and verify
        png_path = os.path.join(self.temp_dir, "test_from_array.png")
        with open(png_path, "wb") as f:
            f.write(png_bytes)
        
        extracted_uid = self.file_manager.extract_uid_from_png(png_path)
        self.assertEqual(extracted_uid, test_uid)
    
    def test_save_png_with_uid(self):
        """Test saving PNG with embedded UID."""
        png_path = os.path.join(self.temp_dir, "test_save.png")
        test_uid = str(uuid.uuid4())
        
        success = self.file_manager.save_png_with_uid(self.test_image, png_path, test_uid)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(png_path))
        
        # Verify UID extraction
        extracted_uid = self.file_manager.extract_uid_from_png(png_path)
        self.assertEqual(extracted_uid, test_uid)
    
    def test_embed_uid_in_any_image_png(self):
        """Test generic UID embedding for PNG files."""
        png_path = os.path.join(self.temp_dir, "test_any.png")
        cv2.imwrite(png_path, self.test_image)
        
        test_uid = str(uuid.uuid4())
        embedded_uid = self.file_manager.embed_uid_in_any_image(png_path, test_uid)
        
        self.assertEqual(embedded_uid, test_uid)
        
        # Verify extraction
        extracted_uid = self.file_manager.extract_uid_from_any_image(png_path)
        self.assertEqual(extracted_uid, test_uid)
    
    def test_embed_uid_in_any_image_tiff(self):
        """Test generic UID embedding for TIFF files."""
        tiff_path = os.path.join(self.temp_dir, "test_any.tiff")
        cv2.imwrite(tiff_path, self.test_image)
        
        test_uid = str(uuid.uuid4())
        embedded_uid = self.file_manager.embed_uid_in_any_image(tiff_path, test_uid)
        
        self.assertEqual(embedded_uid, test_uid)
        
        # Verify extraction
        extracted_uid = self.file_manager.extract_uid_from_any_image(tiff_path)
        self.assertEqual(extracted_uid, test_uid)
    
    def test_copy_with_metadata_preserves_uid(self):
        """Test that copying files preserves embedded UIDs."""
        # Test with JPEG
        jpeg_path = os.path.join(self.temp_dir, "source.jpg")
        jpeg_copy_path = os.path.join(self.temp_dir, "copy.jpg")
        cv2.imwrite(jpeg_path, self.test_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        test_uid = str(uuid.uuid4())
        self.file_manager.embed_uid_in_image(jpeg_path, test_uid)
        
        # Copy with metadata
        success = self.file_manager.copy_with_metadata(jpeg_path, jpeg_copy_path)
        self.assertTrue(success)
        
        # Verify UID preserved
        extracted_uid = self.file_manager.extract_uid_from_image(jpeg_copy_path)
        self.assertEqual(extracted_uid, test_uid)
        
        # Test with PNG
        png_path = os.path.join(self.temp_dir, "source.png")
        png_copy_path = os.path.join(self.temp_dir, "copy.png")
        
        self.file_manager.save_png_with_uid(self.test_image, png_path, test_uid)
        success = self.file_manager.copy_with_metadata(png_path, png_copy_path)
        self.assertTrue(success)
        
        extracted_uid = self.file_manager.extract_uid_from_png(png_copy_path)
        self.assertEqual(extracted_uid, test_uid)
    
    def test_uid_generation_when_none_provided(self):
        """Test automatic UID generation when none provided."""
        jpeg_path = os.path.join(self.temp_dir, "auto_uid.jpg")
        cv2.imwrite(jpeg_path, self.test_image)
        
        embedded_uid = self.file_manager.embed_uid_in_any_image(jpeg_path)
        self.assertIsNotNone(embedded_uid)
        
        # Should be valid UUID
        uuid.UUID(embedded_uid)  # Will raise ValueError if invalid
        
        # Should be extractable
        extracted_uid = self.file_manager.extract_uid_from_any_image(jpeg_path)
        self.assertEqual(extracted_uid, embedded_uid)
    
    def test_png_uid_extraction_fallback(self):
        """Test PNG UID extraction from Description field as fallback."""
        png_path = os.path.join(self.temp_dir, "description_uid.png")
        test_uid = str(uuid.uuid4())
        
        # Create PNG with UID in Description only
        img = PILImage.fromarray(cv2.cvtColor(self.test_image, cv2.COLOR_BGR2RGB))
        from PIL.PngImagePlugin import PngInfo
        png_info = PngInfo()
        png_info.add_text("Description", f"Source Image UID: {test_uid}")
        img.save(png_path, pnginfo=png_info)
        
        extracted_uid = self.file_manager.extract_uid_from_png(png_path)
        self.assertEqual(extracted_uid, test_uid)
    
    def test_save_compartment_with_uid(self):
        """Test saving compartment images with UID preservation."""
        test_uid = str(uuid.uuid4())
        hole_id = "AB1234"
        compartment_num = 5
        
        # Test PNG format
        saved_path = self.file_manager.save_compartment(
            self.test_image, hole_id, compartment_num, 
            output_format="png", source_uid=test_uid
        )
        
        self.assertIsNotNone(saved_path)
        self.assertTrue(os.path.exists(saved_path))
        
        # Verify UID embedded
        extracted_uid = self.file_manager.extract_uid_from_png(saved_path)
        self.assertEqual(extracted_uid, test_uid)
        
        # Test JPEG format
        saved_path_jpg = self.file_manager.save_compartment(
            self.test_image, hole_id, compartment_num + 1, 
            output_format="jpg", source_uid=test_uid
        )
        
        extracted_uid_jpg = self.file_manager.extract_uid_from_image(saved_path_jpg)
        self.assertEqual(extracted_uid_jpg, test_uid)
    
    def test_error_handling_invalid_image_paths(self):
        """Test error handling for invalid image paths."""
        invalid_path = "/nonexistent/path/test.jpg"
        
        # Should not raise exception
        extracted_uid = self.file_manager.extract_uid_from_any_image(invalid_path)
        self.assertIsNone(extracted_uid)
        
        # Embedding should handle gracefully
        test_uid = str(uuid.uuid4())
        result_uid = self.file_manager.embed_uid_in_any_image(invalid_path, test_uid)
        self.assertEqual(result_uid, test_uid)  # Should return provided UID
    
    def test_unsupported_format_handling(self):
        """Test handling of unsupported image formats."""
        test_path = os.path.join(self.temp_dir, "test.bmp")
        cv2.imwrite(test_path, self.test_image)
        
        test_uid = str(uuid.uuid4())
        
        # Should handle gracefully
        embedded_uid = self.file_manager.embed_uid_in_any_image(test_path, test_uid)
        self.assertEqual(embedded_uid, test_uid)
        
        # Extraction should return None for unsupported format
        extracted_uid = self.file_manager.extract_uid_from_any_image(test_path)
        self.assertIsNone(extracted_uid)


class TestFileManagerCore(unittest.TestCase):
    """Test core FileManager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = Mock()
        self.config_manager.get.return_value = None  # Return None for shared_folder_path
        self.file_manager = FileManager(self.temp_dir, self.config_manager)
        
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_directory_structure_creation(self):
        """Test that directory structure is created correctly."""
        self.file_manager.create_base_directories()
        
        # Check main directories exist
        expected_dirs = [
            "Images to Process",
            "Processed Original Images",
            "Extracted Compartment Images",
            "Debugging"
        ]
        
        for dir_name in expected_dirs:
            dir_path = os.path.join(self.temp_dir, dir_name)
            self.assertTrue(os.path.exists(dir_path), f"Directory {dir_name} not created")
    
    def test_get_hole_dir_creates_project_structure(self):
        """Test that hole directories create proper project structure."""
        hole_id = "AB1234"
        
        hole_dir = self.file_manager.get_hole_dir("approved_originals", hole_id)
        
        # Should create ProjectCode/HoleID structure
        expected_path = os.path.join(
            self.temp_dir, 
            "Processed Original Images",
            "Approved Originals", 
            "AB", 
            "AB1234"
        )
        
        self.assertEqual(hole_dir, expected_path)
        self.assertTrue(os.path.exists(hole_dir))
    
    def test_extract_metadata_from_filename(self):
        """Test metadata extraction from standard filenames."""
        test_cases = [
            ("AB1234_40-60_Original.jpg", "AB1234", 40.0, 60.0),
            ("CD5678_100-105_Original_Skipped.png", "CD5678", 100.0, 105.0),
            ("BB9999_0-5.tiff", "BB9999", 0.0, 5.0),
        ]
        
        for filename, expected_hole, expected_from, expected_to in test_cases:
            metadata = self.file_manager.extract_metadata_from_filename(filename)
            
            self.assertIsNotNone(metadata, f"Failed to extract from {filename}")
            self.assertEqual(metadata["hole_id"], expected_hole)
            self.assertEqual(metadata["depth_from"], expected_from)
            self.assertEqual(metadata["depth_to"], expected_to)
            self.assertTrue(metadata["from_filename"])
    
    def test_safe_file_rename(self):
        """Test safe file renaming with error handling."""
        # Create test file
        test_file = os.path.join(self.temp_dir, "test_original.jpg")
        with open(test_file, "wb") as f:
            f.write(b"test data")
        
        new_file = os.path.join(self.temp_dir, "test_renamed.jpg")
        
        success = self.file_manager.rename_file_safely(test_file, new_file)
        
        self.assertTrue(success)
        self.assertFalse(os.path.exists(test_file))
        self.assertTrue(os.path.exists(new_file))
    
    def test_heif_conversion(self):
        """Test HEIF to JPEG conversion."""
        # Skip if pillow-heif not properly installed
        try:
            from pillow_heif import register_heif_opener
        except ImportError:
            self.skipTest("pillow-heif not available")
        
        # Create mock HEIF file path
        heif_path = os.path.join(self.temp_dir, "test.heic")
        
        # This would need actual HEIF data to test properly
        # For now, test the error handling
        result = self.file_manager.convert_heif_to_jpeg(heif_path)
        self.assertIsNone(result)  # Should fail gracefully with non-existent file


if __name__ == '__main__':
    unittest.main()