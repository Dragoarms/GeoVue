# GeoVue Chip Tray Detection + Classification

Real-time detection and classification of chip tray cells on Raspberry Pi 5.

## Overview

This system combines two ML models:

1. **YOLO Detector** - Finds cell bounding boxes in camera frames
2. **4-Class Classifier** - Classifies each cell as:
   - **Dry** (orange) - Normal dry chip samples
   - **Wet** (blue) - Wet chip samples
   - **Empty** (gray) - Empty compartments
   - **Bad** (red) - Poor quality images

## Quick Start

### Step 1: Prepare YOLO Dataset

```bash
cd ml_detection
pip install -r requirements.txt

# Convert corner coordinates to YOLO format
python prepare_yolo_dataset.py
```

### Step 2: Train YOLO Detector

```bash
# Train nano model (best for Pi 5)
python train_yolo.py --model-size n --epochs 100
```

### Step 3: Train Cell Classifier

Before training, you need labeled images:
- `*_Dry.png` - Dry samples (existing)
- `*_Wet.png` - Wet samples (existing)
- `*_Empty.png` - Empty cells (manually label some)
- `*_Bad.png` - Bad images (auto-detected from register OR manually label)

```bash
# Train 4-class classifier
python train_classifier.py --epochs 50 --model mobilenet_v3_small
```

### Step 4: Export for Pi 5

```bash
# Export YOLO to NCNN format
python train_yolo.py --mode export --export-format ncnn
```

### Step 5: Deploy to Raspberry Pi 5

Copy to Pi:
```bash
scp -r runs/detect/chip_tray_detector/weights/best_ncnn_model pi@raspberrypi:~/
scp classifier_output/checkpoints/best_model.pt pi@raspberrypi:~/
scp pi5_live_detection.py cell_classifier.py pi@raspberrypi:~/
```

Run on Pi:
```bash
# Detection only
python pi5_live_detection.py --detector best_ncnn_model

# Detection + Classification
python pi5_live_detection.py \
    --detector best_ncnn_model \
    --classifier best_model.pt
```

## Labeling Empty Cells

To add Empty cell training data:

1. Copy some compartment images that show empty cells
2. Rename them with the pattern: `{HoleID}_CC_{Depth}_Empty.png`
3. Place in the approved compartment images folder
4. Re-run classifier training

Example:
```
OK0001_CC_015_Empty.png
KM0042_CC_003_Empty.png
```

## Command Reference

### prepare_yolo_dataset.py

Converts corner coordinates to YOLO format. Creates `yolo_dataset/` folder.

### train_yolo.py

```
--mode          train|export|both
--model-size    n|s|m|l|x (n=nano, best for Pi 5)
--epochs        Training epochs (default: 100)
--export-format ncnn|onnx|tflite (ncnn best for ARM)
```

### train_classifier.py

```
--model         mobilenet_v3_small|mobilenet_v3_large|efficientnet_b0|resnet18
--epochs        Training epochs (default: 50)
--batch-size    Batch size (default: 32)
--images-path   Path to approved compartment images
--register-path Path to register data (for Bad status detection)
```

### pi5_live_detection.py

```
--detector      Path to YOLO model (required)
--classifier    Path to classifier model (optional)
--mode          live|image
--detection-conf    Detection threshold (default: 0.5)
--resolution    Camera resolution WxH (default: 1280x720)
```

Keyboard controls:
- `q` - Quit
- `s` - Save screenshot
- `c` - Toggle classifier on/off

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING (Windows PC)                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  compartment_corners.json ─────► prepare_yolo_dataset.py    │
│         │                              │                     │
│         │                              ▼                     │
│         │                       yolo_dataset/                │
│         │                              │                     │
│         │                              ▼                     │
│         │                       train_yolo.py                │
│         │                              │                     │
│         │                              ▼                     │
│         │                    best_ncnn_model/                │
│         │                                                    │
│  Approved Compartment Images ──► train_classifier.py        │
│  + Photo_Status=*_Bad               │                        │
│                                      ▼                       │
│                              best_model.pt                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE (Pi 5)                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Camera Frame ──► YOLO Detector ──► Cell Bounding Boxes     │
│                         │                   │                │
│                         │                   ▼                │
│                         │          Cell Classifier           │
│                         │                   │                │
│                         ▼                   ▼                │
│                    Annotated Frame (Dry/Wet/Empty/Bad)       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Expected Performance on Pi 5

| Mode | Model | FPS | Notes |
|------|-------|-----|-------|
| Detection only | YOLOv11n | 20-30 | Fastest |
| Detection + Classification | YOLO + MobileNetV3 | 10-15 | Full pipeline |

## Class Colors

| Class | Color | Description |
|-------|-------|-------------|
| Dry | Orange | Normal dry chip samples |
| Wet | Blue | Wet chip samples |
| Empty | Gray | No chips in compartment |
| Bad | Red | Poor image quality |

## Troubleshooting

### "No samples found" during classifier training
- Check that images follow naming pattern: `{HoleID}_CC_{Depth}_{Label}.png`
- Verify approved compartment images path is correct

### Low detection accuracy
- Train for more epochs
- Check training data quality
- Lower confidence threshold: `--detection-conf 0.3`

### Slow FPS on Pi 5
- Use YOLO nano model
- Reduce resolution: `--resolution 640x480`
- Disable classifier: don't pass `--classifier`

### Camera not working on Pi
- Test: `libcamera-hello`
- Enable camera: `sudo raspi-config`
- Use OpenCV fallback: `--use-opencv-camera`

## Files

```
ml_detection/
├── prepare_yolo_dataset.py  # YOLO data prep
├── train_yolo.py            # YOLO training
├── cell_classifier.py       # 4-class classifier module
├── train_classifier.py      # Classifier training
├── pi5_live_detection.py    # Pi 5 deployment
├── requirements.txt         # Dependencies
├── README.md               # This file
├── yolo_dataset/           # Generated YOLO dataset
├── classifier_output/      # Classifier training outputs
└── runs/                   # YOLO training outputs
```
