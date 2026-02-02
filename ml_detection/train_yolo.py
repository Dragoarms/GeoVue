"""
Train YOLOv8 Model for Chip Tray Cell Detection

This script trains a YOLOv8 model on the prepared dataset.
Supports multiple model sizes for different deployment targets.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import torch


def train_model(
    dataset_yaml: str,
    model_size: str = "n",  # n, s, m, l, x
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "auto",
    project: str = "runs/detect",
    name: str = "chip_tray_detector",
    resume: bool = False,
    pretrained: bool = True,
):
    """
    Train YOLOv8 model

    Model sizes:
    - n (nano): Fastest, smallest, best for Pi 5
    - s (small): Good balance
    - m (medium): More accurate
    - l (large): High accuracy
    - x (extra-large): Maximum accuracy
    """
    # Select device
    if device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"

    print(f"Training on device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    model_name = f"yolo11{model_size}.pt" if pretrained else f"yolo11{model_size}.yaml"
    print(f"\nLoading model: {model_name}")
    model = YOLO(model_name)

    # Train
    print(f"\nStarting training...")
    print(f"  Dataset: {dataset_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}")

    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        resume=resume,
        # Augmentation settings
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        # Training settings
        patience=20,  # Early stopping patience
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
    )

    print("\nTraining complete!")
    print(f"Best model saved to: {project}/{name}/weights/best.pt")

    return results


def export_for_pi(
    model_path: str,
    export_format: str = "ncnn",  # ncnn is optimized for ARM
    img_size: int = 640,
):
    """
    Export trained model for Raspberry Pi deployment

    Recommended formats for Pi 5:
    - ncnn: Best performance on ARM (recommended)
    - onnx: Good compatibility
    - tflite: TensorFlow Lite (alternative)
    """
    print(f"\nExporting model for Raspberry Pi...")
    print(f"  Source: {model_path}")
    print(f"  Format: {export_format}")

    model = YOLO(model_path)

    # Export
    exported_path = model.export(
        format=export_format,
        imgsz=img_size,
        half=False,  # FP32 for better compatibility
        simplify=True,
    )

    print(f"Exported model: {exported_path}")
    return exported_path


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for Chip Tray Detection")
    parser.add_argument(
        "--mode",
        choices=["train", "export", "both"],
        default="train",
        help="Operation mode"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(Path(__file__).parent / "yolo_dataset" / "dataset.yaml"),
        help="Path to dataset.yaml"
    )
    parser.add_argument(
        "--model-size",
        choices=["n", "s", "m", "l", "x"],
        default="n",
        help="Model size (n=nano for Pi, s=small, m=medium)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Image size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto, cpu, 0, 1, etc.)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="runs/detect/chip_tray_detector/weights/best.pt",
        help="Path to trained model (for export)"
    )
    parser.add_argument(
        "--export-format",
        choices=["ncnn", "onnx", "tflite", "openvino"],
        default="ncnn",
        help="Export format for Pi deployment"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint"
    )

    args = parser.parse_args()

    if args.mode in ["train", "both"]:
        train_model(
            dataset_yaml=args.dataset,
            model_size=args.model_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            device=args.device,
            resume=args.resume,
        )

    if args.mode in ["export", "both"]:
        export_for_pi(
            model_path=args.model_path,
            export_format=args.export_format,
            img_size=args.img_size,
        )


if __name__ == "__main__":
    main()
