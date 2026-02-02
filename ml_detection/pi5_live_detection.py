#!/usr/bin/env python3
"""
Raspberry Pi 5 Live Chip Tray Detection + Classification

This script combines:
1. YOLO object detection - Find cell locations in frame
2. 4-class classifier - Classify each cell as Dry/Wet/Empty/Bad

Requirements:
- Raspberry Pi 5 with camera module
- YOLO model for cell detection
- Cell classifier model (optional, for status classification)
"""

import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

# Try to import picamera2 (only available on Pi)
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    print("Note: picamera2 not available. Using OpenCV camera.")

from ultralytics import YOLO

# Import classifier components
try:
    from cell_classifier import CellLabels, CellClassifierInference
    CLASSIFIER_AVAILABLE = True
except ImportError:
    CLASSIFIER_AVAILABLE = False
    print("Note: Cell classifier not available. Detection only mode.")


class ChipTrayDetector:
    """
    Real-time chip tray cell detector with optional classification.

    Pipeline:
    1. Capture frame from camera
    2. Run YOLO to detect cell bounding boxes
    3. For each detected cell, crop and classify (if classifier loaded)
    4. Draw annotated frame with boxes and labels
    """

    # Class colors (BGR for OpenCV)
    CLASS_COLORS = {
        "Dry": (0, 165, 255),     # Orange
        "Wet": (255, 0, 0),       # Blue
        "Empty": (128, 128, 128), # Gray
        "Bad": (0, 0, 255),       # Red
        "cell": (0, 255, 0),      # Green (detection only)
    }

    def __init__(
        self,
        detector_path: str,
        classifier_path: Optional[str] = None,
        detection_confidence: float = 0.5,
        classification_confidence: float = 0.5,
        img_size: int = 640,
        use_picamera: bool = True,
        camera_resolution: tuple = (1280, 720),
    ):
        self.detection_confidence = detection_confidence
        self.classification_confidence = classification_confidence
        self.img_size = img_size
        self.use_picamera = use_picamera and PICAMERA_AVAILABLE
        self.camera_resolution = camera_resolution

        # Load YOLO detector
        print(f"Loading detector from: {detector_path}")
        self.detector = YOLO(detector_path)
        print("Detector loaded!")

        # Load classifier (optional)
        self.classifier = None
        if classifier_path and CLASSIFIER_AVAILABLE:
            print(f"Loading classifier from: {classifier_path}")
            self.classifier = CellClassifierInference(
                model_path=classifier_path,
                device="cpu",  # Use CPU on Pi for stability
            )
            print("Classifier loaded!")
        elif classifier_path:
            print("Warning: Classifier requested but not available")

        # FPS tracking
        self.fps_history = []
        self.fps_window = 30

        # Camera
        self.camera = None

        # Stats
        self.frame_count = 0
        self.detection_count = 0

    def setup_camera(self):
        """Initialize camera."""
        if self.use_picamera:
            print("Initializing Picamera2...")
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                main={"size": self.camera_resolution, "format": "RGB888"}
            )
            self.camera.configure(config)
            self.camera.start()
            print(f"Picamera2 started at {self.camera_resolution}")
        else:
            print("Initializing OpenCV camera...")
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_resolution[1])
            if not self.camera.isOpened():
                raise RuntimeError("Failed to open camera")
            print(f"OpenCV camera opened at {self.camera_resolution}")

    def get_frame(self) -> Optional[np.ndarray]:
        """Capture a frame from the camera."""
        if self.use_picamera:
            frame = self.camera.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = self.camera.read()
            if not ret:
                return None
        return frame

    def detect_cells(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Run YOLO detection on frame.
        Returns list of (x1, y1, x2, y2, confidence) tuples.
        """
        results = self.detector.predict(
            frame,
            imgsz=self.img_size,
            conf=self.detection_confidence,
            verbose=False,
        )

        detections = []
        boxes = results[0].boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = box.conf[0].cpu().numpy()
            detections.append((x1, y1, x2, y2, float(confidence)))

        return detections

    def classify_cells(
        self,
        frame: np.ndarray,
        detections: List[Tuple[int, int, int, int, float]],
    ) -> List[Tuple[str, float]]:
        """
        Classify each detected cell.
        Returns list of (label, confidence) tuples.
        """
        if not self.classifier or not detections:
            return [("cell", 1.0) for _ in detections]

        # Crop cells
        crops = []
        for x1, y1, x2, y2, _ in detections:
            # Ensure valid crop coordinates
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                crops.append(crop)
            else:
                crops.append(np.zeros((10, 10, 3), dtype=np.uint8))

        # Batch classify
        classifications = self.classifier.classify_batch(crops)

        return classifications

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Tuple[int, int, int, int, float]],
        classifications: List[Tuple[str, float]],
        fps: float = 0,
    ) -> np.ndarray:
        """Draw bounding boxes with classification labels."""
        annotated = frame.copy()

        # Count by class
        class_counts = {"Dry": 0, "Wet": 0, "Empty": 0, "Bad": 0, "cell": 0}

        for i, (x1, y1, x2, y2, det_conf) in enumerate(detections):
            label, class_conf = classifications[i]
            class_counts[label] = class_counts.get(label, 0) + 1

            # Get color for this class
            color = self.CLASS_COLORS.get(label, (0, 255, 0))

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            if self.classifier:
                text = f"{label} {class_conf:.0%}"
            else:
                text = f"Cell {det_conf:.0%}"

            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                annotated,
                (x1, y1 - text_h - 8),
                (x1 + text_w + 8, y1),
                color,
                -1,
            )

            # Draw text
            cv2.putText(
                annotated,
                text,
                (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        # Draw info panel
        self._draw_info_panel(annotated, fps, class_counts)

        return annotated

    def _draw_info_panel(
        self,
        frame: np.ndarray,
        fps: float,
        class_counts: dict,
    ):
        """Draw info panel with FPS and class counts."""
        h, w = frame.shape[:2]

        # Background for info
        cv2.rectangle(frame, (10, 10), (200, 130), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (200, 130), (255, 255, 255), 1)

        y_offset = 30
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        total = sum(class_counts.values())
        y_offset += 25
        cv2.putText(frame, f"Cells: {total}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Per-class counts (if classifier active)
        if self.classifier:
            for label in ["Dry", "Wet", "Empty", "Bad"]:
                count = class_counts.get(label, 0)
                if count > 0:
                    y_offset += 20
                    color = self.CLASS_COLORS[label]
                    cv2.putText(frame, f"{label}: {count}", (20, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def calculate_fps(self, elapsed_time: float) -> float:
        """Calculate rolling average FPS."""
        if elapsed_time > 0:
            current_fps = 1.0 / elapsed_time
            self.fps_history.append(current_fps)
            if len(self.fps_history) > self.fps_window:
                self.fps_history.pop(0)
            return sum(self.fps_history) / len(self.fps_history)
        return 0

    def run(
        self,
        display: bool = True,
        save_video: bool = False,
        output_path: str = None,
    ):
        """Main detection + classification loop."""
        print("\n" + "=" * 50)
        print("Starting Live Detection")
        print("=" * 50)
        print("Controls:")
        print("  q - Quit")
        print("  s - Save screenshot")
        print("  c - Toggle classifier (if loaded)")
        print("=" * 50)

        self.setup_camera()

        video_writer = None
        if save_video:
            output_path = output_path or "detection_output.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                output_path, fourcc, 30.0, self.camera_resolution
            )
            print(f"Recording to: {output_path}")

        screenshot_count = 0
        classifier_enabled = self.classifier is not None

        try:
            while True:
                start_time = time.time()

                # Capture frame
                frame = self.get_frame()
                if frame is None:
                    print("Failed to capture frame")
                    continue

                self.frame_count += 1

                # Detect cells
                detections = self.detect_cells(frame)
                self.detection_count += len(detections)

                # Classify cells (if enabled)
                if classifier_enabled and self.classifier:
                    classifications = self.classify_cells(frame, detections)
                else:
                    classifications = [("cell", 1.0) for _ in detections]

                # Calculate FPS
                elapsed = time.time() - start_time
                fps = self.calculate_fps(elapsed)

                # Draw annotations
                annotated = self.draw_detections(frame, detections, classifications, fps)

                # Save video frame
                if video_writer:
                    video_writer.write(annotated)

                # Display
                if display:
                    cv2.imshow("Chip Tray Detection", annotated)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Quitting...")
                        break
                    elif key == ord('s'):
                        screenshot_count += 1
                        filename = f"screenshot_{screenshot_count}.jpg"
                        cv2.imwrite(filename, annotated)
                        print(f"Saved: {filename}")
                    elif key == ord('c'):
                        if self.classifier:
                            classifier_enabled = not classifier_enabled
                            status = "ON" if classifier_enabled else "OFF"
                            print(f"Classifier: {status}")

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            # Cleanup
            if video_writer:
                video_writer.release()
            if self.use_picamera:
                self.camera.stop()
            else:
                self.camera.release()
            cv2.destroyAllWindows()

            # Stats
            print("\n--- Session Stats ---")
            print(f"Frames processed: {self.frame_count}")
            print(f"Total detections: {self.detection_count}")
            if self.frame_count > 0:
                print(f"Avg detections/frame: {self.detection_count / self.frame_count:.1f}")

    def process_single_image(
        self,
        image_path: str,
        output_path: str = None,
    ):
        """Process a single image file."""
        print(f"Processing: {image_path}")

        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Detect
        detections = self.detect_cells(frame)
        print(f"Detected {len(detections)} cells")

        # Classify
        if self.classifier:
            classifications = self.classify_cells(frame, detections)
        else:
            classifications = [("cell", 1.0) for _ in detections]

        # Draw
        annotated = self.draw_detections(frame, detections, classifications)

        if output_path:
            cv2.imwrite(output_path, annotated)
            print(f"Saved: {output_path}")
        else:
            cv2.imshow("Detection Result", annotated)
            print("Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return detections, classifications


def main():
    parser = argparse.ArgumentParser(description="Pi 5 Chip Tray Detection + Classification")

    # Model paths
    parser.add_argument(
        "--detector",
        type=str,
        required=True,
        help="Path to YOLO detector model"
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default=None,
        help="Path to cell classifier model (optional)"
    )

    # Mode
    parser.add_argument(
        "--mode",
        choices=["live", "image"],
        default="live",
        help="Detection mode"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Image path for single image mode"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for processed image/video"
    )

    # Detection settings
    parser.add_argument(
        "--detection-conf",
        type=float,
        default=0.5,
        help="Detection confidence threshold (0-1)"
    )
    parser.add_argument(
        "--classification-conf",
        type=float,
        default=0.5,
        help="Classification confidence threshold (0-1)"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="YOLO inference image size"
    )

    # Camera settings
    parser.add_argument(
        "--resolution",
        type=str,
        default="1280x720",
        help="Camera resolution (WxH)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable display (headless mode)"
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save video output"
    )
    parser.add_argument(
        "--use-opencv-camera",
        action="store_true",
        help="Use OpenCV camera instead of Picamera2"
    )

    args = parser.parse_args()

    # Parse resolution
    res_parts = args.resolution.split('x')
    resolution = (int(res_parts[0]), int(res_parts[1]))

    # Create detector
    detector = ChipTrayDetector(
        detector_path=args.detector,
        classifier_path=args.classifier,
        detection_confidence=args.detection_conf,
        classification_confidence=args.classification_conf,
        img_size=args.img_size,
        use_picamera=not args.use_opencv_camera,
        camera_resolution=resolution,
    )

    if args.mode == "live":
        detector.run(
            display=not args.no_display,
            save_video=args.save_video,
            output_path=args.output,
        )
    elif args.mode == "image":
        if not args.image:
            raise ValueError("--image required for image mode")
        detector.process_single_image(args.image, args.output)


if __name__ == "__main__":
    main()
