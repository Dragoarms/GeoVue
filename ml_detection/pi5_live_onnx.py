#!/usr/bin/env python3
"""
Live chip tray detection on Raspberry Pi using ONNX Runtime.
Supports both GUI mode (on Pi display) and headless mode (over SSH).

Usage:
    # GUI mode (requires display):
    python pi5_live_onnx.py --detector models/yolo_detector.onnx --classifier models/classifier.onnx

    # Headless mode (saves frames to disk):
    python pi5_live_onnx.py --detector models/yolo_detector.onnx --classifier models/classifier.onnx --headless
"""

import argparse
import time
import os

import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2

# Class labels for the 4-class classifier
CLASS_LABELS = ["Dry", "Wet", "Empty", "Bad"]
CLASS_COLORS = {
    "Dry": (0, 255, 0),      # Green
    "Wet": (255, 0, 0),      # Blue
    "Empty": (128, 128, 128), # Gray
    "Bad": (0, 0, 255),      # Red
}


def preprocess_yolo(image: np.ndarray, input_size: int = 640) -> tuple:
    """Preprocess image for YOLO inference."""
    h, w = image.shape[:2]
    scale = min(input_size / h, input_size / w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(image, (new_w, new_h))

    pad_h = (input_size - new_h) // 2
    pad_w = (input_size - new_w) // 2
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    blob = padded.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)
    blob = np.expand_dims(blob, 0)

    return blob, scale, pad_w, pad_h


def postprocess_yolo(output: np.ndarray, scale: float, pad_w: int, pad_h: int,
                     conf_thresh: float = 0.5, iou_thresh: float = 0.5) -> list:
    """Postprocess YOLO output to get bounding boxes."""
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
            scores, conf_thresh, iou_thresh
        )
        if len(indices) > 0:
            indices = indices.flatten()
            return [(boxes[i], scores[i]) for i in indices]

    return []


def preprocess_classifier(image: np.ndarray, input_size: int = 224) -> np.ndarray:
    """Preprocess image for classifier inference."""
    resized = cv2.resize(image, (input_size, input_size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    blob = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    blob = (blob - mean) / std

    blob = blob.transpose(2, 0, 1)
    blob = np.expand_dims(blob, 0)
    return blob


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


class ChipTrayDetector:
    """Combined YOLO detector + classifier for chip tray analysis."""

    def __init__(self, detector_path: str, classifier_path: str):
        print(f"Loading YOLO detector: {detector_path}")
        self.detector = ort.InferenceSession(
            detector_path,
            providers=['CPUExecutionProvider']
        )
        self.detector_input = self.detector.get_inputs()[0].name

        print(f"Loading classifier: {classifier_path}")
        self.classifier = ort.InferenceSession(
            classifier_path,
            providers=['CPUExecutionProvider']
        )
        self.classifier_input = self.classifier.get_inputs()[0].name

        print("Models loaded successfully!")

    def detect_and_classify(self, frame: np.ndarray, conf_thresh: float = 0.5) -> list:
        """Detect cells and classify each one."""
        results = []

        # Run YOLO detection
        blob, scale, pad_w, pad_h = preprocess_yolo(frame)
        yolo_output = self.detector.run(None, {self.detector_input: blob})[0]
        detections = postprocess_yolo(yolo_output, scale, pad_w, pad_h, conf_thresh)

        h, w = frame.shape[:2]

        # Classify each detected cell
        for (x1, y1, x2, y2), det_conf in detections:
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            cell_img = frame[y1:y2, x1:x2]
            blob = preprocess_classifier(cell_img)
            cls_output = self.classifier.run(None, {self.classifier_input: blob})[0]
            probs = softmax(cls_output[0])
            class_idx = np.argmax(probs)
            class_conf = probs[class_idx]
            class_label = CLASS_LABELS[class_idx]

            results.append({
                'box': (x1, y1, x2, y2),
                'det_conf': det_conf,
                'class': class_label,
                'class_conf': float(class_conf),
            })

        return results

    def draw_results(self, frame: np.ndarray, results: list) -> np.ndarray:
        """Draw detection and classification results on frame."""
        annotated = frame.copy()
        counts = {label: 0 for label in CLASS_LABELS}

        for r in results:
            x1, y1, x2, y2 = r['box']
            label = r['class']
            conf = r['class_conf']
            color = CLASS_COLORS[label]
            counts[label] += 1

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            text = f"{label} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, text, (x1 + 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                       lineType=cv2.LINE_AA)

        # Draw summary
        y_offset = 40
        for label in CLASS_LABELS:
            text = f"{label}: {counts[label]}"
            color = CLASS_COLORS[label]
            cv2.putText(annotated, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2,
                       lineType=cv2.LINE_AA)
            y_offset += 35

        return annotated


def run_headless(detector: ChipTrayDetector, resolution: tuple, conf: float, output_dir: str):
    """Run detection in headless mode (no display), save frames to files."""
    print("Starting headless mode...")
    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    picam2 = Picamera2()

    # Configure for still capture at requested resolution
    config = picam2.create_still_configuration(
        main={"size": resolution, "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # Let camera warm up

    # Enable autofocus
    try:
        picam2.set_controls({"AfMode": 2, "AfTrigger": 0})
        time.sleep(1)
        print("Autofocus enabled")
    except Exception as e:
        print(f"Autofocus not available: {e}")

    print("\nCapturing frames... Press Ctrl+C to stop")
    frame_count = 0

    try:
        while True:
            # Capture frame
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Run detection
            start_time = time.time()
            results = detector.detect_and_classify(frame_bgr, conf)
            inference_time = time.time() - start_time

            # Draw results
            annotated = detector.draw_results(frame_bgr, results)

            # Count by class
            counts = {}
            for r in results:
                counts[r['class']] = counts.get(r['class'], 0) + 1

            # Save frame
            frame_count += 1
            filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(filename, annotated)

            # Print status
            print(f"\rFrame {frame_count}: {counts} | Inference: {inference_time*1000:.0f}ms | Saved: {filename}", end="", flush=True)

            # Wait a bit between captures
            time.sleep(0.5)

    except KeyboardInterrupt:
        print(f"\n\nStopped. Captured {frame_count} frames to {output_dir}")
    finally:
        picam2.stop()


def run_gui(detector: ChipTrayDetector, resolution: tuple, conf: float):
    """Run detection with GUI preview using OpenCV window."""
    print("Starting GUI mode with OpenCV display...")

    picam2 = Picamera2()

    # Configure camera - use requested resolution
    config = picam2.create_preview_configuration(
        main={"size": resolution, "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)

    # Enable autofocus
    try:
        picam2.set_controls({"AfMode": 2, "AfTrigger": 0})
        print("Autofocus enabled")
    except Exception as e:
        print(f"Autofocus not available: {e}")

    print(f"\nResolution: {resolution}")
    print("Running detection... Press 'q' to quit, 's' to save frame")

    cv2.namedWindow("Chip Tray Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Chip Tray Detection", 1280, 720)

    frame_count = 0
    fps_times = []

    try:
        while True:
            start_time = time.time()

            # Capture frame
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Run detection
            results = detector.detect_and_classify(frame_bgr, conf)

            # Draw results
            annotated = detector.draw_results(frame_bgr, results)

            # Calculate FPS
            fps_times.append(time.time() - start_time)
            if len(fps_times) > 30:
                fps_times.pop(0)
            fps = 1.0 / (sum(fps_times) / len(fps_times))

            # Draw FPS
            cv2.putText(annotated, f"FPS: {fps:.1f}", (20, annotated.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # Show frame
            cv2.imshow("Chip Tray Detection", annotated)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                frame_count += 1
                filename = f"capture_{frame_count:04d}.jpg"
                cv2.imwrite(filename, annotated)
                print(f"\nSaved: {filename}")

            # Print status
            if results:
                counts = {}
                for r in results:
                    counts[r['class']] = counts.get(r['class'], 0) + 1
                print(f"\rDetected: {counts} | FPS: {fps:.1f}", end="", flush=True)

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        cv2.destroyAllWindows()
        picam2.stop()


def main():
    parser = argparse.ArgumentParser(description="Chip Tray Detection (ONNX)")
    parser.add_argument("--detector", required=True, help="Path to YOLO ONNX model")
    parser.add_argument("--classifier", required=True, help="Path to classifier ONNX model")
    parser.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--resolution", default="1920x1080", help="Resolution WxH")
    parser.add_argument("--headless", action="store_true", help="Run without display (save frames to files)")
    parser.add_argument("--output", default="./output", help="Output directory for headless mode")

    args = parser.parse_args()

    # Parse resolution
    w, h = map(int, args.resolution.split("x"))

    # Load models
    detector = ChipTrayDetector(args.detector, args.classifier)

    if args.headless:
        run_headless(detector, (w, h), args.conf, args.output)
    else:
        run_gui(detector, (w, h), args.conf)


if __name__ == "__main__":
    main()
