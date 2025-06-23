import unittest
import numpy as np
import os
import threading
import tempfile
import types
import importlib
import sys
import cv2

from core.visualization_manager import VisualizationManager
from processing.aruco_manager import ArucoManager


class TestVisualizationPipeline(unittest.TestCase):
    def test_scaling_and_projection(self):
        h, w = 2000, 13000
        img = np.zeros((h, w, 3), dtype=np.uint8)
        vm = VisualizationManager()
        vm.load_image(img, "dummy.jpg")
        working = vm.create_working_copy(target_pixels=2000000)
        expected_scale = (2000000 / (h * w)) ** 0.5
        expected_size = (int(h * expected_scale), int(w * expected_scale))
        self.assertEqual(working.shape[:2], expected_size)

        vm.apply_rotation(vm.working_key, 10)
        rect = np.array([[50, 50], [300, 50], [300, 250], [50, 250]], dtype=np.float32)
        theta = np.radians(15)
        rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta),  np.cos(theta)]], dtype=np.float32)
        rotated_rect = rect.dot(rot_mat.T)

        projected = vm.scale_coordinates(rotated_rect, vm.working_key, vm.original_key)
        manual_scaled = rotated_rect * (1 / expected_scale)
        np.testing.assert_allclose(projected, manual_scaled, atol=1)


class TestArucoManager(unittest.TestCase):
    def test_analyze_compartment_boundaries_empty(self):
        manager = ArucoManager(config={})
        result = manager.analyze_compartment_boundaries(None, {})
        self.assertEqual(result["boundaries"], [])
        self.assertIn("vertical_constraints", result)


class TestCompartmentRegistrationDialog(unittest.TestCase):
    def test_import_in_thread_raises(self):
        module_name = "gui.compartment_registration_dialog"
        sys.modules.pop(module_name, None)
        exc = []

        def target():
            try:
                importlib.import_module(module_name)
            except Exception as e:
                exc.append(e)

        t = threading.Thread(target=target)
        t.start()
        t.join()
        self.assertTrue(exc and isinstance(exc[0], RuntimeError))


class DummyArucoManager:
    def improve_marker_detection(self, img):
        return {}

    def analyze_compartment_boundaries(self, image, markers, **kwargs):
        h = image.shape[0]
        return {
            'boundaries': [],
            'missing_marker_ids': [],
            'vertical_constraints': (0, h),
            'marker_to_compartment': {},
            'corner_markers': {},
            'boundary_to_marker': {},
            'marker_data': [],
            'detected_compartment_markers': {},
            'expected_marker_ids': [],
            'scale_px_per_cm': None,
            'compartment_interval': 1
        }


class DummyFileManager:
    def check_original_file_processed(self, path):
        return False


class DummyDialog:
    def __init__(self, *args, **kwargs):
        pass

    def show(self):
        return None


class DummyGeoVue:
    def __init__(self):
        self.logger = importlib.import_module('logging').getLogger('test')
        self.viz_manager = VisualizationManager()
        self.file_manager = DummyFileManager()
        self.aruco_manager = DummyArucoManager()
        self.tesseract_manager = types.SimpleNamespace(is_available=False)
        self.config = {
            'corner_marker_ids': [0, 1, 2, 3],
            'compartment_marker_ids': list(range(4, 24)),
            'metadata_marker_ids': [24],
            'enable_ocr': False,
            'compartment_count': 20
        }
        self.progress_queue = importlib.import_module('queue').Queue()
        self.main_gui = types.SimpleNamespace(update_status=lambda *a, **k: None)

    def clear_visualization_cache(self):
        self.viz_manager.clear()

    # Bind the original process_image method

    process_image = importlib.import_module('src.main').GeoVue.process_image


class TestProcessImage(unittest.TestCase):
    def test_large_image_downscale(self):
        h, w = 2000, 13000
        img = np.zeros((h, w, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, img)
            path = tmp.name
        try:
            app = DummyGeoVue()
            # Patch dialog and skew correction
            vm = app.viz_manager
            def dummy_correct(markers):
                return {
                    'image': vm.get_current_working_image(),
                    'rotation_matrix': None,
                    'rotation_angle': 0.0,
                    'version_key': vm.working_key,
                    'needs_redetection': False
                }
            vm.correct_image_skew = dummy_correct
            module = importlib.import_module('gui.compartment_registration_dialog')
            module.CompartmentRegistrationDialog = DummyDialog
            result = app.process_image(path)
            self.assertFalse(result)
            working_img = app.viz_manager.get_current_working_image()
            self.assertIsNotNone(working_img)
            scale = app.viz_manager._get_scale_factor('working', 'original')
            self.assertLess(scale, 1.0)
        finally:
            os.remove(path)


if __name__ == '__main__':
    unittest.main()
