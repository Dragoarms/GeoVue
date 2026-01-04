#!/usr/bin/env python3
"""
Focused test for PNG UID embedding fix.
Tests the specific PngInfo import issue and PNG metadata functionality.
"""

import sys
import os
import unittest
import tempfile
import shutil
import uuid
import cv2
import numpy as np

# Add src to path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestPngUidFix(unittest.TestCase):
    """Test the PNG UID embedding fix specifically."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Import FileManager directly to avoid dependency issues
        try:
            from core.file_manager import FileManager
            from unittest.mock import Mock
            
            # Create a proper config_manager mock
            config_manager = Mock()
            config_manager.get.return_value = None  # Return None for shared_folder_path
            self.file_manager = FileManager(self.temp_dir, config_manager)
            
            # Create test image
            self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            self.test_image[:] = [255, 0, 0]  # Red test image
            
        except ImportError as e:
            self.skipTest(f"Cannot import FileManager: {e}")
            
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_pnginfo_import_available(self):
        """Test that PngInfo can be imported (fix verification)."""
        try:
            from PIL.PngImagePlugin import PngInfo
            png_info = PngInfo()
            png_info.add_text("test", "value")
            self.assertIsNotNone(png_info)
        except ImportError:
            self.fail("PngInfo import failed - fix not properly applied")
    
    def test_png_uid_embedding_works(self):
        """Test that PNG UID embedding now works with the fix."""
        test_uid = str(uuid.uuid4())
        
        # Test the embed_uid_in_png method directly
        png_bytes = self.file_manager.embed_uid_in_png(self.test_image, test_uid)
        
        # Should not return None (which was the failure mode)
        self.assertIsNotNone(png_bytes, "PNG UID embedding returned None - fix failed")
        self.assertIsInstance(png_bytes, bytes)
        self.assertGreater(len(png_bytes), 0)
        
        # Save to file and verify extraction
        png_path = os.path.join(self.temp_dir, "test_uid.png")
        with open(png_path, "wb") as f:
            f.write(png_bytes)
        
        # Extract UID back
        extracted_uid = self.file_manager.extract_uid_from_png(png_path)
        self.assertEqual(extracted_uid, test_uid, "UID extraction failed")
    
    def test_save_png_with_uid_method(self):
        """Test the save_png_with_uid method works correctly."""
        png_path = os.path.join(self.temp_dir, "test_save_uid.png")
        test_uid = str(uuid.uuid4())
        
        # This method was failing before the fix
        success = self.file_manager.save_png_with_uid(self.test_image, png_path, test_uid)
        
        self.assertTrue(success, "save_png_with_uid returned False")
        self.assertTrue(os.path.exists(png_path), "PNG file was not created")
        
        # Verify UID embedded
        extracted_uid = self.file_manager.extract_uid_from_png(png_path)
        self.assertEqual(extracted_uid, test_uid, "UID not properly embedded/extracted")
    
    def test_png_metadata_structure(self):
        """Test PNG metadata structure is correct."""
        test_uid = str(uuid.uuid4())
        png_path = os.path.join(self.temp_dir, "test_metadata.png")
        
        # Save with UID
        self.file_manager.save_png_with_uid(self.test_image, png_path, test_uid)
        
        # Check metadata directly with PIL
        from PIL import Image as PILImage
        img = PILImage.open(png_path)
        
        # Should have both SourceUID and Description
        self.assertIn("SourceUID", img.info, "SourceUID metadata missing")
        self.assertIn("Description", img.info, "Description metadata missing")
        
        # Values should be correct
        self.assertEqual(img.info["SourceUID"], test_uid)
        self.assertEqual(img.info["Description"], f"Source Image UID: {test_uid}")
    
    def test_png_uid_fallback_extraction(self):
        """Test UID extraction fallback from Description field."""
        test_uid = str(uuid.uuid4())
        png_path = os.path.join(self.temp_dir, "test_fallback.png")
        
        # Create PNG with only Description (simulating edge case)
        from PIL import Image as PILImage
        from PIL.PngImagePlugin import PngInfo
        
        rgb_image = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb_image)
        
        png_info = PngInfo()
        png_info.add_text("Description", f"Source Image UID: {test_uid}")
        # Deliberately not adding SourceUID
        
        pil_image.save(png_path, pnginfo=png_info)
        
        # Should still extract UID from Description
        extracted_uid = self.file_manager.extract_uid_from_png(png_path)
        self.assertEqual(extracted_uid, test_uid, "Fallback extraction failed")
    
    def test_error_handling_graceful(self):
        """Test error handling doesn't crash the application."""
        # Test with invalid image array
        invalid_image = np.array([])
        test_uid = str(uuid.uuid4())
        
        # Should handle gracefully
        try:
            png_bytes = self.file_manager.embed_uid_in_png(invalid_image, test_uid)
            # Should return None or handle gracefully
            self.assertIsNone(png_bytes)
        except Exception as e:
            # Should not raise unhandled exceptions
            self.fail(f"Unhandled exception in error case: {e}")


class TestPngUidIntegration(unittest.TestCase):
    """Test PNG UID integration with compartment saving."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        try:
            from core.file_manager import FileManager
            from unittest.mock import Mock
            
            # Create a proper config_manager mock
            config_manager = Mock()
            config_manager.get.return_value = None  # Return None for shared_folder_path
            self.file_manager = FileManager(self.temp_dir, config_manager)
            self.file_manager.create_base_directories()
            
            self.test_image = np.zeros((150, 200, 3), dtype=np.uint8)
            self.test_image[:] = [0, 255, 0]  # Green test image
            
        except ImportError as e:
            self.skipTest(f"Cannot import FileManager: {e}")
            
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_compartment_save_preserves_uid(self):
        """Test that saving compartments preserves UID correctly."""
        test_uid = str(uuid.uuid4())
        hole_id = "TEST123"
        compartment_num = 5
        
        # Save compartment with PNG format (uses the fixed methods)
        saved_path = self.file_manager.save_compartment(
            self.test_image, hole_id, compartment_num,
            output_format="png", source_uid=test_uid
        )
        
        self.assertIsNotNone(saved_path, "Compartment save failed")
        self.assertTrue(os.path.exists(saved_path), "Compartment file not created")
        
        # Verify UID preserved
        extracted_uid = self.file_manager.extract_uid_from_png(saved_path)
        self.assertEqual(extracted_uid, test_uid, "UID not preserved in compartment")
        
        # Verify filename format
        expected_filename = f"{hole_id}_CC_{compartment_num:03d}.png"
        self.assertTrue(saved_path.endswith(expected_filename))
    
    def test_temp_compartment_uid_handling(self):
        """Test temporary compartment UID handling."""
        test_uid = str(uuid.uuid4())
        hole_id = "TEMP456"
        compartment_depth = 25
        
        # Save temp compartment
        temp_path = self.file_manager.save_temp_compartment(
            self.test_image, hole_id, compartment_depth,
            suffix="temp", source_uid=test_uid
        )
        
        self.assertIsNotNone(temp_path, "Temp compartment save failed")
        
        # Verify UID embedded
        extracted_uid = self.file_manager.extract_uid_from_png(temp_path)
        self.assertEqual(extracted_uid, test_uid, "UID not preserved in temp compartment")
        
        # Verify temp filename format
        expected_filename = f"{hole_id}_CC_{compartment_depth:03d}_temp.png"
        self.assertTrue(temp_path.endswith(expected_filename))


if __name__ == '__main__':
    # Run just these focused tests
    unittest.main(verbosity=2)