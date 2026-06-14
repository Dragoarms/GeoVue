"""Export models to ONNX format for Raspberry Pi deployment."""

import torch
from pathlib import Path

# Export YOLO detector
print("Exporting YOLO detector to ONNX...")
from ultralytics import YOLO

yolo_path = Path("runs/detect/runs/detect/chip_tray_detector/weights/best.pt")
if yolo_path.exists():
    model = YOLO(str(yolo_path))
    model.export(format="onnx", imgsz=640, simplify=True)
    print(f"YOLO exported to: {yolo_path.with_suffix('.onnx')}")
else:
    print(f"YOLO model not found at {yolo_path}")

# Export classifier
print("\nExporting classifier to ONNX...")
from cell_classifier import CellClassifierModel

classifier_path = Path("classifier_output/checkpoints/best_model.pt")
if classifier_path.exists():
    # Load checkpoint
    checkpoint = torch.load(classifier_path, map_location="cpu")

    # Create model
    model = CellClassifierModel(
        model_name="mobilenet_v3_small",
        num_classes=4,
        pretrained=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Export to ONNX
    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_path = classifier_path.parent / "classifier.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    print(f"Classifier exported to: {onnx_path}")
else:
    print(f"Classifier not found at {classifier_path}")

print("\nDone! Copy the .onnx files to your Raspberry Pi.")
