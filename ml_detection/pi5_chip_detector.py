#!/usr/bin/env python3
"""
Fast chip tray cell detector using YOLO ONNX on Raspberry Pi.
Optimized for higher FPS with threading.

Usage:
    python pi5_chip_detector.py --model models/yolo_detector.onnx
"""

import argparse
import time
import threading
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2


class FastDetector:
    def __init__(self, model_path: str, input_size: int = 640):
        print(f"Loading model: {model_path}")
        self.model = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.model.get_inputs()[0].name
        self.input_size = input_size
        print("Model loaded!")

    def preprocess(self, image: np.ndarray) -> tuple:
        h, w = image.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h = (self.input_size - new_h) // 2
        pad_w = (self.input_size - new_w) // 2
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        blob = padded.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]

        return blob, scale, pad_w, pad_h

    def detect(self, image: np.ndarray, conf_thresh: float = 0.5) -> list:
        blob, scale, pad_w, pad_h = self.preprocess(image)
        output = self.model.run(None, {self.input_name: blob})[0]

        predictions = output[0].T
        boxes = []
        scores = []

        for pred in predictions:
            conf = pred[4]
            if conf < conf_thresh:
                continue

            x_center, y_center, w, h = pred[:4]
            x_center = (x_center - pad_w) / scale
            y_center = (y_center - pad_h) / scale
            w = w / scale
            h = h / scale

            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)

            boxes.append([x1, y1, x2, y2])
            scores.append(float(conf))

        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(
                [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes],
                scores, conf_thresh, 0.5
            )
            if len(indices) > 0:
                indices = indices.flatten()
                return [(boxes[i], scores[i]) for i in indices]

        return []


def main():
    parser = argparse.ArgumentParser(description="Chip Tray Detector")
    parser.add_argument("--model", required=True, help="Path to YOLO ONNX model")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--resolution", default="1280x720", help="Display resolution")
    parser.add_argument("--infer-size", type=int, default=640, help="Inference size (320/640)")
    args = parser.parse_args()

    w, h = map(int, args.resolution.split("x"))

    # Load detector
    detector = FastDetector(args.model, args.infer_size)

    # Setup camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (w, h), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)

    # Autofocus
    try:
        picam2.set_controls({"AfMode": 2, "AfTrigger": 0})
        print("Autofocus enabled")
    except:
        pass

    print(f"Resolution: {w}x{h}, Inference: {args.infer_size}x{args.infer_size}")
    print("Press 'q' to quit, 's' to save")

    cv2.namedWindow("Chip Tray Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Chip Tray Detector", 1280, 720)

    # Shared state for threading
    frame_lock = threading.Lock()
    latest_frame = None
    latest_detections = []
    running = True

    def capture_thread():
        nonlocal latest_frame, running
        while running:
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            with frame_lock:
                latest_frame = frame_bgr

    def detect_thread():
        nonlocal latest_detections, running
        while running:
            with frame_lock:
                frame = latest_frame
            if frame is not None:
                latest_detections = detector.detect(frame, args.conf)
            time.sleep(0.01)

    # Start threads
    cap_thread = threading.Thread(target=capture_thread, daemon=True)
    det_thread = threading.Thread(target=detect_thread, daemon=True)
    cap_thread.start()
    det_thread.start()

    fps_times = deque(maxlen=30)
    save_count = 0

    try:
        while running:
            t0 = time.time()

            with frame_lock:
                frame = latest_frame
            detections = latest_detections

            if frame is None:
                time.sleep(0.01)
                continue

            display = frame.copy()

            # Draw boxes
            for (x1, y1, x2, y2), conf in detections:
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display, f"{conf:.0%}", (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # FPS
            fps_times.append(time.time() - t0)
            fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0

            cv2.putText(display, f"Cells: {len(detections)} | FPS: {fps:.1f}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.imshow("Chip Tray Detector", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                running = False
            elif key == ord('s'):
                save_count += 1
                cv2.imwrite(f"detection_{save_count:04d}.jpg", display)
                print(f"\nSaved: detection_{save_count:04d}.jpg")

    except KeyboardInterrupt:
        running = False
    finally:
        running = False
        cv2.destroyAllWindows()
        picam2.stop()
        print("\nDone")


if __name__ == "__main__":
    main()
