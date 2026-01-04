import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import Mock, MagicMock, patch, call
import tkinter as tk
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os
import shutil

class TestQAQCManager(unittest.TestCase):
    """Test suite for QAQCManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock dependencies
        self.root = tk.Tk()
        self.root.withdraw()  # Hide test window
        
        self.file_manager = Mock()
        self.translator_func = Mock(side_effect=lambda x: x)  # Return input as-is
        self.config_manager = Mock()
        self.app = Mock()
        self.app.root = self.root
        self.app.config = {
            'compartment_interval': 5,
            'output_format': 'png'
        }
        self.app.gui_manager = Mock()
        self.app.gui_manager.theme_colors = {
            'field_bg': '#ffffff',
            'text': '#000000'
        }
        self.app.gui_manager.create_modern_button = Mock(return_value=Mock())
        
        self.logger = Mock()
        self.register_manager = Mock()
        
        from gui.qaqc_manager import QAQCManager
        # Create test instance
        self.qaqc = QAQCManager(
            self.file_manager,
            self.translator_func,
            self.config_manager,
            self.app,
            self.logger,
            self.register_manager
        )
        
        # Create temp directory for tests
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after tests."""
        # Clean up temp directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        # Destroy tkinter window
        self.root.destroy()
    
    def test_initialization(self):
        """Test QAQCManager initialization."""
        self.assertIsNotNone(self.qaqc)
        self.assertEqual(self.qaqc.STATUS_OK, "OK")
        self.assertEqual(self.qaqc.STATUS_DAMAGED, "Damaged")
        self.assertIn("OK_Wet", self.qaqc.VALID_QAQC_STATUSES)
        self.assertIsNotNone(self.qaqc.COMPARTMENT_FILE_PATTERN)
    
    def test_review_item_creation(self):
        """Test ReviewItem dataclass functionality."""
        from gui.qaqc_manager import ReviewItem
        
        item = ReviewItem(
            filename="AB1234_CC_100.png",
            hole_id="AB1234",
            depth_from=95,
            depth_to=100,
            compartment_depth=100,
            image_path="/path/to/image.png"
        )
        
        self.assertEqual(item.filename, "AB1234_CC_100.png")
        self.assertEqual(item.hole_id, "AB1234")
        self.assertEqual(item.moisture, "unknown")
        self.assertEqual(item.quality, "unreviewed")
        self.assertFalse(item.is_reviewed)
    
    def test_find_all_compartment_files(self):
        """Test finding compartment files in directory."""
        # Create test files
        test_files = [
            "AB1234_CC_100.png",
            "AB1234_CC_105_Wet.png",
            "CD5678_CC_200_Dry.tiff",
            "not_a_compartment.txt",
            "AB1234_tray_001.jpg"  # Should not match
        ]
        
        for filename in test_files:
            Path(self.temp_dir, filename).touch()
        
        # Test finding files
        files = self.qaqc._find_all_compartment_files(self.temp_dir)
        
        # Should find only compartment files
        self.assertEqual(len(files), 3)
        filenames = [f[1] for f in files]
        self.assertIn("AB1234_CC_100.png", filenames)
        self.assertIn("AB1234_CC_105_Wet.png", filenames)
        self.assertIn("CD5678_CC_200_Dry.tiff", filenames)
    
    def test_group_files_by_interval(self):
        """Test grouping files by hole and depth interval."""
        test_files = [
            (f"{self.temp_dir}/AB1234_CC_100.png", "AB1234_CC_100.png"),
            (f"{self.temp_dir}/AB1234_CC_100_Wet.png", "AB1234_CC_100_Wet.png"),
            (f"{self.temp_dir}/AB1234_CC_105.png", "AB1234_CC_105.png"),
            (f"{self.temp_dir}/CD5678_CC_200.png", "CD5678_CC_200.png"),
        ]
        
        grouped = self.qaqc._group_files_by_interval(test_files)
        
        # Check grouping
        self.assertEqual(len(grouped), 3)  # 3 unique hole/depth combinations
        self.assertIn(("AB1234", 100), grouped)
        self.assertIn(("AB1234", 105), grouped)
        self.assertIn(("CD5678", 200), grouped)
        
        # Check that duplicates are grouped together
        self.assertEqual(len(grouped[("AB1234", 100)]), 2)
    
    def test_get_register_status(self):
        """Test retrieving register status for compartment."""
        # Mock register data
        mock_df = pd.DataFrame({
            'HoleID': ['AB1234', 'AB1234', 'AB1234'],
            'From': [95, 100, 105],
            'To': [100, 105, 110],
            'Photo_Status': ['OK_Wet', 'Found', 'Empty']
        })
        
        self.qaqc.compartment_register = {'AB1234': mock_df}
        
        # Test getting status
        status = self.qaqc._get_register_status('AB1234', 100)
        self.assertEqual(status, 'OK_Wet')
        
        status = self.qaqc._get_register_status('AB1234', 105)
        self.assertEqual(status, 'Found')
        
        # Test non-existent
        status = self.qaqc._get_register_status('AB1234', 115)
        self.assertIsNone(status)
    
    def test_update_review_item_from_status(self):
        """Test updating ReviewItem based on GUI status."""
        from gui.qaqc_manager import ReviewItem
        
        item = ReviewItem(
            filename="test.png",
            hole_id="AB1234",
            depth_from=95,
            depth_to=100,
            compartment_depth=100,
            image_path="/path/to/test.png"
        )
        
        # Test Wet status
        self.qaqc._update_review_item_from_status(item, "Wet")
        self.assertEqual(item.moisture, "Wet")
        self.assertEqual(item.quality, "OK")
        self.assertTrue(item.is_reviewed)
        
        # Test Empty status
        item2 = ReviewItem(
            filename="test2.png",
            hole_id="AB1234",
            depth_from=100,
            depth_to=105,
            compartment_depth=105,
            image_path="/path/to/test2.png"
        )
        
        self.qaqc._update_review_item_from_status(item2, "Empty")
        self.assertEqual(item2.quality, "Empty")
        self.assertTrue(item2.is_reviewed)
    
    def test_batch_update_register(self):
        """Test batch updating register entries."""
        from gui.qaqc_manager import ReviewItem
        
        items = [
            ReviewItem(
                filename="test1.png",
                hole_id="AB1234",
                depth_from=95,
                depth_to=100,
                compartment_depth=100,
                image_path="/path/to/test1.png",
                quality="OK",
                moisture="Wet",
                is_reviewed=True
            ),
            ReviewItem(
                filename="test2.png",
                hole_id="AB1234",
                depth_from=100,
                depth_to=105,
                compartment_depth=105,
                image_path="/path/to/test2.png",
                quality="Empty",
                is_reviewed=True
            )
        ]
        
        # Mock successful update
        self.register_manager.batch_update_compartments.return_value = 2
        
        # Execute
        self.qaqc._batch_update_register(items)
        
        # Verify
        self.register_manager.batch_update_compartments.assert_called_once()
        updates = self.register_manager.batch_update_compartments.call_args[0][0]
        
        self.assertEqual(len(updates), 2)
        self.assertEqual(updates[0]['photo_status'], 'OK_Wet')
        self.assertEqual(updates[1]['photo_status'], 'Empty')
        self.assertEqual(self.qaqc.stats['register_updated'], 2)
    
    @patch('cv2.imread')
    def test_display_image_scaling(self, mock_imread):
        """Test image display with proper scaling."""
        # Create test image
        test_image = np.zeros((1000, 800, 3), dtype=np.uint8)
        mock_imread.return_value = test_image
        
        # Create mock frame data
        frame_data = {
            'canvas': Mock(),
            'image_label': Mock(),
            'depth_label': Mock(),
            'canvas_window': 1,
            'status_window': 2,
            'status_label': Mock()
        }
        
        # Mock canvas dimensions
        frame_data['canvas'].winfo_width.return_value = 300
        frame_data['canvas'].winfo_height.return_value = 400
        frame_data['canvas'].coords = Mock()
        frame_data['canvas'].itemconfig = Mock()
        
        # Test display
        with patch('PIL.Image.fromarray') as mock_fromarray:
            with patch('PIL.ImageTk.PhotoImage') as mock_photoimage:
                self.qaqc._display_image_in_frame(
                    frame_data,
                    image_path="/test/image.png",
                    depth_text="100m"
                )
        
        # Verify image was scaled down (original 1000x800 to fit in 280x380)
        mock_fromarray.assert_called_once()
        call_args = mock_fromarray.call_args[0][0]
        self.assertLess(call_args.shape[0], 380)  # Height should be less than max
        self.assertLess(call_args.shape[1], 280)  # Width should be less than max
    
    def test_create_review_window(self):
        """Test review window creation."""
        # Set up test data
        self.qaqc.current_tray = {
            'hole_id': 'AB1234',
            'depth_from': 100,
            'depth_to': 200,
            'compartments': []
        }
        
        # Create window
        self.qaqc._create_review_window()
        
        # Verify window exists
        self.assertIsNotNone(self.qaqc.review_window)
        self.assertTrue(self.qaqc.review_window.winfo_exists())
        
        # Verify 3 frames created
        self.assertEqual(len(self.qaqc.frames), 3)
        self.assertIn('wet', self.qaqc.frames)
        self.assertIn('current', self.qaqc.frames)
        self.assertIn('dry', self.qaqc.frames)
        
        # Clean up
        self.qaqc.review_window.destroy()
    
    def test_keyboard_shortcuts(self):
        """Test keyboard shortcut handling."""
        # Mock frame setup
        self.qaqc.frames = {
            'current': {'moisture_var': Mock()}
        }
        
        # Mock _set_frame_status
        self.qaqc._set_frame_status = Mock()
        
        # Test shortcuts
        self.qaqc._keyboard_shortcut('current', 'ok')
        self.qaqc._set_frame_status.assert_called_with('current', 'OK')
        
        self.qaqc._keyboard_shortcut('current', 'damaged')
        self.qaqc._set_frame_status.assert_called_with('current', 'Damaged')
    
    def test_navigation_validation(self):
        """Test navigation requires status to be set."""
        from gui.qaqc_manager import ReviewItem
        
        # Set up test data
        self.qaqc.current_hole_items = [
            ReviewItem(
                filename="test1.png",
                hole_id="AB1234",
                depth_from=95,
                depth_to=100,
                compartment_depth=100,
                image_path="/test1.png"
            ),
            ReviewItem(
                filename="test2.png",
                hole_id="AB1234",
                depth_from=100,
                depth_to=105,
                compartment_depth=105,
                image_path="/test2.png"
            )
        ]
        
        self.qaqc.current_compartment_index = 0
        self.qaqc.current_tray = {'compartment_statuses': {}}
        self.qaqc.review_window = Mock()
        
        # Mock dialog helper
        with patch('gui.dialog_helper.DialogHelper.show_message') as mock_dialog:
            # Try to navigate without status
            self.qaqc._on_next()
            
            # Should show warning
            mock_dialog.assert_called_once()
            self.assertEqual(self.qaqc.current_compartment_index, 0)  # Should not advance
    
    def test_batch_save_compartments(self):
        """Test batch saving of reviewed compartments."""
        from gui.qaqc_manager import ReviewItem
        
        # Create test items
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        items = [
            ReviewItem(
                filename="test1.png",
                hole_id="AB1234",
                depth_from=95,
                depth_to=100,
                compartment_depth=100,
                image_path="/test1.png",
                quality="OK",
                moisture="Wet",
                _image=test_image
            )
        ]
        
        # Mock file manager save
        self.file_manager.save_reviewed_compartment.return_value = {
            'local_path': '/local/test.png',
            'upload_success': True
        }
        
        # Execute
        self.qaqc._batch_save_compartments(items)
        
        # Verify
        self.file_manager.save_reviewed_compartment.assert_called_once()
        call_args = self.file_manager.save_reviewed_compartment.call_args[1]
        self.assertEqual(call_args['hole_id'], 'AB1234')
        self.assertEqual(call_args['compartment_depth'], 100)
        self.assertEqual(call_args['status'], 'OK_Wet')
        
        # Check stats
        self.assertEqual(self.qaqc.stats['saved_locally'], 1)
        self.assertEqual(self.qaqc.stats['uploaded'], 1)