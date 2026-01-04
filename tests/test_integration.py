#!/usr/bin/env python3
"""
Integration tests for GeoVue components.
Tests end-to-end workflows and component interactions.
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
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.file_manager import FileManager
from utils.json_register_manager import JSONRegisterManager


class TestImageProcessingWorkflow(unittest.TestCase):
    """Test complete image processing workflow with UID tracking."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = Mock()
        self.config_manager.get.return_value = None  # Return None for shared_folder_path
        
        # Set up file manager
        self.file_manager = FileManager(self.temp_dir, self.config_manager)
        self.file_manager.create_base_directories()
        
        # Set up register manager
        register_dir = os.path.join(self.temp_dir, "register")
        os.makedirs(register_dir, exist_ok=True)
        self.register_manager = JSONRegisterManager(register_dir)
        
        # Create test image
        self.test_image = np.zeros((200, 300, 3), dtype=np.uint8)
        self.test_image[:] = [0, 255, 0]  # Green test image
        
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_complete_original_image_workflow(self):
        """Test complete workflow: original image -> compartments -> register."""
        # Step 1: Process original image
        hole_id = "AB1234"
        depth_from = 95
        depth_to = 105
        
        # Create original image file
        original_path = os.path.join(self.temp_dir, f"{hole_id}_{depth_from}-{depth_to}_Original.jpg")
        cv2.imwrite(original_path, self.test_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Generate and embed UID
        image_uid = str(uuid.uuid4())
        embedded_uid = self.file_manager.embed_uid_in_any_image(original_path, image_uid)
        self.assertEqual(embedded_uid, image_uid)
        
        # Step 2: Save original file through file manager
        saved_path, upload_success = self.file_manager.save_original_file(
            original_path, hole_id, depth_from, depth_to,
            is_processed=True, is_rejected=False, image_uid=image_uid
        )
        
        self.assertIsNotNone(saved_path)
        self.assertTrue(os.path.exists(saved_path))
        
        # Verify UID preserved in saved file
        extracted_uid = self.file_manager.extract_uid_from_any_image(saved_path)
        self.assertEqual(extracted_uid, image_uid)
        
        # Step 3: Register original image
        original_data = {
            embedded_uid: {
                "UID": embedded_uid,
                "HoleID": hole_id,
                "From": depth_from,
                "To": depth_to,
                "Status": "Processed",
                "Local_Path": saved_path
            }
        }
        
        with self.register_manager.file_locks("originalimages"):
            with open(self.register_manager.files["originalimages"], 'w') as f:
                import json
                json.dump(original_data, f, indent=2)
        
        # Step 4: Create compartments from original
        compartments = []
        for i in range(3):  # Create 3 compartments
            compartment_image = self.test_image.copy()
            # Add some variation to each compartment
            compartment_image[i*20:(i+1)*20, :] = [255, 0, 0]  # Add red stripe
            
            compartment_path = self.file_manager.save_compartment(
                compartment_image, hole_id, i+1, 
                output_format="png", source_uid=image_uid
            )
            
            self.assertIsNotNone(compartment_path)
            compartments.append(compartment_path)
            
            # Verify UID embedded in compartment
            comp_uid = self.file_manager.extract_uid_from_png(compartment_path)
            self.assertEqual(comp_uid, image_uid)
        
        # Step 5: Register compartments
        compartment_data = {}
        for i, comp_path in enumerate(compartments):
            depth = depth_from + (i * 5)  # 5m intervals
            
            if hole_id not in compartment_data:
                compartment_data[hole_id] = {}
            
            compartment_data[hole_id][str(depth)] = {
                "HoleID": hole_id,
                "From": depth - 2.5,
                "To": depth + 2.5,
                "Photo_Status": "Found",
                "Source_Image_UID": image_uid,
                "Local_Path": comp_path
            }
        
        with self.register_manager.file_locks("compartment"):
            with open(self.register_manager.files["compartment"], 'w') as f:
                import json
                json.dump(compartment_data, f, indent=2)
        
        # Step 6: Verify complete workflow
        # Check original image can be retrieved by UID
        original_record = self.register_manager.get_original_image_by_uid(image_uid)
        self.assertIsNotNone(original_record)
        self.assertEqual(original_record["HoleID"], hole_id)
        
        # Check compartments can be updated by source UID
        updated_count = self.register_manager.update_compartments_by_source_uid(
            image_uid, "Reprocessed"
        )
        self.assertEqual(updated_count, 3)
        
        # Verify all compartments updated
        comp_data = self.register_manager.load_register_data("compartment")
        for depth_key in comp_data[hole_id]:
            self.assertEqual(comp_data[hole_id][depth_key]["Photo_Status"], "Reprocessed")
    
    def test_qaqc_review_workflow(self):
        """Test QA/QC review workflow with register updates."""
        # Create compartment for review
        hole_id = "CD5678"
        compartment_num = 10
        source_uid = str(uuid.uuid4())
        
        # Create compartment image
        compartment_image = self.test_image.copy()
        temp_path = self.file_manager.save_temp_compartment(
            compartment_image, hole_id, compartment_num, 
            suffix="temp", source_uid=source_uid
        )
        
        self.assertIsNotNone(temp_path)
        self.assertTrue(os.path.exists(temp_path))
        
        # Verify UID in temp compartment
        temp_uid = self.file_manager.extract_uid_from_png(temp_path)
        self.assertEqual(temp_uid, source_uid)
        
        # Simulate QA/QC review process
        review_result = self.file_manager.save_reviewed_compartment(
            compartment_image, hole_id, compartment_num,
            status="Wet", output_format="png", source_uid=source_uid
        )
        
        self.assertIsNotNone(review_result["local_path"])
        self.assertTrue(os.path.exists(review_result["local_path"]))
        
        # Check UID preserved through review
        reviewed_uid = self.file_manager.extract_uid_from_png(review_result["local_path"])
        self.assertEqual(reviewed_uid, source_uid)
        
        # Register the review
        compartment_data = {
            hole_id: {
                str(compartment_num): {
                    "HoleID": hole_id,
                    "From": compartment_num - 2.5,
                    "To": compartment_num + 2.5,
                    "Photo_Status": "OK_Wet",
                    "Source_Image_UID": source_uid,
                    "Review_Path": review_result["local_path"]
                }
            }
        }
        
        with self.register_manager.file_locks("compartment"):
            with open(self.register_manager.files["compartment"], 'w') as f:
                import json
                json.dump(compartment_data, f, indent=2)
        
        # Clean up temp files
        self.file_manager.cleanup_temp_compartments(hole_id, [temp_path])
    
    def test_reprocessing_workflow(self):
        """Test reprocessing workflow with UID tracking."""
        hole_id = "EF9012"
        original_uid = str(uuid.uuid4())
        
        # Create initial processing record
        initial_data = {
            original_uid: {
                "UID": original_uid,
                "HoleID": hole_id,
                "Status": "Processed"
            }
        }
        
        # Create compartment data
        compartment_data = {
            hole_id: {
                "100": {
                    "HoleID": hole_id,
                    "Photo_Status": "Found",
                    "Source_Image_UID": original_uid
                },
                "105": {
                    "HoleID": hole_id,
                    "Photo_Status": "OK_Dry", 
                    "Source_Image_UID": original_uid
                }
            }
        }
        
        # Save initial state
        with self.register_manager.file_locks("originalimages", "compartment"):
            with open(self.register_manager.files["originalimages"], 'w') as f:
                import json
                json.dump(initial_data, f, indent=2)
            
            with open(self.register_manager.files["compartment"], 'w') as f:
                json.dump(compartment_data, f, indent=2)
        
        # Trigger reprocessing
        updated_count = self.register_manager.update_compartments_by_source_uid(
            original_uid, "Reprocessed"
        )
        
        self.assertEqual(updated_count, 2)
        
        # Verify reprocessing status
        comp_data = self.register_manager.load_register_data("compartment")
        self.assertEqual(comp_data[hole_id]["100"]["Photo_Status"], "Reprocessed")
        self.assertEqual(comp_data[hole_id]["105"]["Photo_Status"], "Reprocessed")
        
        # Original record should still exist
        original_record = self.register_manager.get_original_image_by_uid(original_uid)
        self.assertIsNotNone(original_record)
        self.assertEqual(original_record["Status"], "Processed")
    
    def test_error_recovery_workflow(self):
        """Test error recovery and data consistency."""
        hole_id = "GH3456"
        source_uid = str(uuid.uuid4())
        
        # Simulate partial failure - compartment saved but register update failed
        compartment_image = self.test_image.copy()
        saved_path = self.file_manager.save_compartment(
            compartment_image, hole_id, 1,
            output_format="png", source_uid=source_uid
        )
        
        self.assertIsNotNone(saved_path)
        
        # Verify UID is intact for recovery
        recovered_uid = self.file_manager.extract_uid_from_png(saved_path)
        self.assertEqual(recovered_uid, source_uid)
        
        # Simulate recovery by registering the orphaned compartment
        recovery_data = {
            hole_id: {
                "5": {  # Depth extracted from filename or metadata
                    "HoleID": hole_id,
                    "From": 2.5,
                    "To": 7.5,
                    "Photo_Status": "Recovered",
                    "Source_Image_UID": source_uid,
                    "Recovery_Path": saved_path
                }
            }
        }
        
        with self.register_manager.file_locks("compartment"):
            with open(self.register_manager.files["compartment"], 'w') as f:
                import json
                json.dump(recovery_data, f, indent=2)
        
        # Verify recovery successful
        comp_data = self.register_manager.load_register_data("compartment")
        self.assertEqual(comp_data[hole_id]["5"]["Photo_Status"], "Recovered")
        self.assertEqual(comp_data[hole_id]["5"]["Source_Image_UID"], source_uid)


class TestConfigurationIntegration(unittest.TestCase):
    """Test configuration and settings integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = Mock()
        self.config_manager.get.return_value = None  # Return None for shared_folder_path
        
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_shared_folder_configuration(self):
        """Test shared folder path configuration."""
        shared_dir = os.path.join(self.temp_dir, "shared")
        os.makedirs(shared_dir, exist_ok=True)
        
        # Configure shared paths
        self.config_manager.get.return_value = shared_dir
        
        file_manager = FileManager(self.temp_dir, self.config_manager)
        file_manager.initialize_shared_paths()
        
        # Verify shared paths configured
        self.assertIsNotNone(file_manager.shared_paths)
        self.assertIn("register", file_manager.shared_paths)
        
        # Test path creation
        register_path = file_manager.get_shared_path("register")
        self.assertIsNotNone(register_path)
        self.assertTrue(register_path.exists())
    
    def test_output_format_configuration(self):
        """Test different output format configurations."""
        file_manager = FileManager(self.temp_dir, self.config_manager)
        file_manager.create_base_directories()
        
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_uid = str(uuid.uuid4())
        hole_id = "TEST001"
        
        # Test PNG format
        png_path = file_manager.save_compartment(
            test_image, hole_id, 1, output_format="png", source_uid=test_uid
        )
        self.assertTrue(png_path.endswith('.png'))
        extracted_uid = file_manager.extract_uid_from_png(png_path)
        self.assertEqual(extracted_uid, test_uid)
        
        # Test JPEG format
        jpg_path = file_manager.save_compartment(
            test_image, hole_id, 2, output_format="jpg", source_uid=test_uid
        )
        self.assertTrue(jpg_path.endswith('.jpg'))
        extracted_uid = file_manager.extract_uid_from_image(jpg_path)
        self.assertEqual(extracted_uid, test_uid)
        
        # Test TIFF format
        tiff_path = file_manager.save_compartment_with_data(
            test_image, hole_id, 3, output_format="tiff", source_uid=test_uid
        )
        self.assertTrue(tiff_path.endswith('.tiff'))
        extracted_uid = file_manager.extract_uid_from_any_image(tiff_path)
        self.assertEqual(extracted_uid, test_uid)


if __name__ == '__main__':
    unittest.main()