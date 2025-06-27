import tkinter as tk
import json
import cv2
from PIL import Image, ImageTk

from core.file_manager import FileManager
from gui.gui_manager import GUIManager
from gui.dialog_helper import DialogHelper
from gui.compartment_registration_dialog import CompartmentRegistrationDialog
from processing.aruco_manager import ArucoManager
from tests.synthetic_data import generate_test_tray_image


def run():
    with open('src/config.json', 'r') as f:
        config = json.load(f)
    image, _ = generate_test_tray_image(config)

    aruco = ArucoManager(config)
    markers = aruco.detect_markers(image)
    analysis = aruco.analyze_compartment_boundaries(image, markers, smart_cropping=False)

    root = tk.Tk()
    fm = FileManager()
    gm = GUIManager(fm)
    DialogHelper.set_gui_manager(gm)

    dlg = CompartmentRegistrationDialog(
        parent=root,
        image=image,
        detected_boundaries=analysis.get('boundaries', []),
        markers=markers,
        boundary_analysis=analysis,
        gui_manager=gm,
        file_manager=fm,
        metadata={'hole_id': 'TEST1', 'depth_from': 0, 'depth_to': 10},
        vertical_constraints=analysis.get('vertical_constraints'),
        marker_to_compartment=analysis.get('marker_to_compartment'),
    )

    root.mainloop()


if __name__ == '__main__':
    run()
