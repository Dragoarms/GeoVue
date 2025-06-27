import tkinter as tk
from PIL import Image, ImageTk
import cv2
import json

from core.visualization_manager import VisualizationManager
from processing.aruco_manager import ArucoManager
from tests.synthetic_data import generate_test_tray_image


def _draw_analysis(image, markers, boundaries):
    viz = image.copy()
    for mid, corners in markers.items():
        pts = corners.astype(int).reshape((-1, 1, 2))
        cv2.polylines(viz, [pts], True, (0, 0, 255), 2)
    for x1, y1, x2, y2 in boundaries:
        cv2.line(viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return viz


def run():
    with open('src/config.json', 'r') as f:
        config = json.load(f)
    image, _ = generate_test_tray_image(config)

    aruco = ArucoManager(config)
    markers = aruco.detect_markers(image)
    analysis = aruco.analyze_compartment_boundaries(image, markers, smart_cropping=False)

    viz = _draw_analysis(image, markers, analysis.get('boundaries', []))

    root = tk.Tk()
    root.title('ArucoManager Test')
    photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB)))
    label = tk.Label(root, image=photo)
    label.pack()
    root.mainloop()


if __name__ == '__main__':
    run()
