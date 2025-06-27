import tkinter as tk
from PIL import Image, ImageTk
import cv2

from core.visualization_manager import VisualizationManager
from tests.synthetic_data import generate_test_tray_image
import json


def _draw_markers(image, markers):
    """Utility draw function for VisualizationManager."""
    viz = image.copy()
    for mid, corners in markers.items():
        pts = corners.astype(int).reshape((-1, 1, 2))
        cv2.polylines(viz, [pts], True, (0, 0, 255), 2)
        center = tuple(corners.mean(axis=0).astype(int))
        cv2.putText(viz, str(mid), center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return viz


def run():
    """Open a window showing VisualizationManager output."""
    with open('src/config.json', 'r') as f:
        config = json.load(f)
    image, markers = generate_test_tray_image(config)

    manager = VisualizationManager()
    manager.load_image(image, 'synthetic.png')
    manager.create_working_copy()

    viz = manager.create_visualization('working', 'markers', _draw_markers, markers=markers)

    root = tk.Tk()
    root.title('VisualizationManager Test')
    photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB)))
    label = tk.Label(root, image=photo)
    label.pack()
    root.mainloop()


if __name__ == '__main__':
    run()
