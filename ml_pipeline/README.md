# GeoVue ML Pipeline - Wet/Dry Classifier

An independent machine learning pipeline for training compartment wet/dry classifiers using GeoVue's approved compartment images.

## Features

- **Transfer Learning**: Uses pretrained ImageNet models (ResNet, EfficientNet, MobileNet)
- **Data Augmentation**: Automatic augmentation for training robustness
- **Live Visualization**: Real-time training progress plots
- **Early Stopping**: Prevents overfitting
- **Class Balancing**: Automatic class weight calculation for imbalanced data
- **Stratified Splitting**: Splits by hole ID to prevent data leakage

## Installation

```bash
# Navigate to the ml_pipeline directory
cd ml_pipeline

# Install dependencies
pip install -r requirements.txt

# For GPU support, install PyTorch with CUDA:
# https://pytorch.org/get-started/locally/
```

## Quick Start

### 1. Interactive Mode (Recommended for first time)

```bash
python run_training.py --interactive
```

This will guide you through:
- Setting up your data paths
- Choosing model architecture
- Configuring training parameters

### 2. Command Line Mode

```bash
# Basic training with default settings
python run_training.py --shared-folder "C:/GeoVue/Shared"

# Custom model and parameters
python run_training.py \
    --shared-folder "C:/GeoVue/Shared" \
    --model resnet34 \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.0001

# Check data without training
python run_training.py --shared-folder "C:/GeoVue/Shared" --data-check-only
```

### 3. Evaluate a Trained Model

```bash
python -m ml_pipeline.evaluate ml_output/checkpoints/best_model.pt \
    --shared-folder "C:/GeoVue/Shared" \
    --output-dir ml_output/evaluation
```

## Data Requirements

The pipeline looks for approved compartment images with wet/dry labels in the filename:

```
Shared Folder/
└── Extracted Compartment Images/
    └── Approved Compartment Images/
        └── [PROJECT]/
            └── [HOLE_ID]/
                ├── BA0001_CC_001_Wet.png   ← Labeled as "Wet"
                ├── BA0001_CC_002_Dry.png   ← Labeled as "Dry"
                └── BA0001_CC_003.png       ← Unlabeled (skipped)
```

**Filename Pattern**: `{HoleID}_CC_{Depth}_{Wet|Dry}.png`

## Model Architectures

| Model | Parameters | Speed | Accuracy |
|-------|------------|-------|----------|
| `resnet18` | 11M | Fast | Good |
| `resnet34` | 21M | Medium | Better |
| `resnet50` | 23M | Medium | Best |
| `efficientnet_b0` | 5M | Fast | Very Good |
| `mobilenet_v3_small` | 2M | Fastest | Baseline |

## Output Files

After training, you'll find:

```
ml_output/
├── config.json                    # Training configuration
├── dataset_info.json              # Dataset statistics
├── training_progress.png          # Training visualization
├── checkpoints/
│   ├── best_model.pt              # Best validation accuracy
│   ├── final_model.pt             # Final epoch model
│   └── checkpoint_epoch_*.pt      # Periodic checkpoints
├── logs/
│   └── training_history.json      # Epoch-by-epoch metrics
└── evaluation/
    ├── evaluation_results.json    # Test metrics
    └── confusion_matrix.png       # Confusion matrix plot
```

## Training Progress Visualization

During training, you'll see live plots showing:
- Training and validation loss
- Training and validation accuracy
- Learning rate schedule
- Overfitting detection (train vs val accuracy)

## Using a Trained Model

### Python API

```python
from ml_pipeline.evaluate import predict_single, predict_batch
from pathlib import Path

# Single prediction
label, confidence = predict_single(
    model_path=Path("ml_output/checkpoints/best_model.pt"),
    image_path=Path("path/to/compartment.png"),
)
print(f"Prediction: {label} ({confidence:.1%} confidence)")

# Batch prediction
image_paths = [Path("img1.png"), Path("img2.png"), Path("img3.png")]
results = predict_batch(
    model_path=Path("ml_output/checkpoints/best_model.pt"),
    image_paths=image_paths,
)
for path, (label, conf) in zip(image_paths, results):
    print(f"{path.name}: {label} ({conf:.1%})")
```

## Tips

1. **Data Quality**: Ensure your approved compartments are correctly labeled (Wet/Dry in filename)

2. **Minimum Data**: You need at least 10 labeled samples per class, but 100+ is recommended

3. **GPU Training**: Use a CUDA-capable GPU for faster training. Install PyTorch with CUDA support.

4. **Early Stopping**: If validation loss stops improving for 10 epochs, training stops automatically

5. **Model Selection**: Start with `resnet18` for quick experiments, use `resnet50` for best accuracy

## Troubleshooting

### "No labeled samples found"
- Check that your compartment images have `_Wet.png` or `_Dry.png` suffixes
- Verify the folder structure matches the expected pattern

### "CUDA out of memory"
- Reduce batch size: `--batch-size 16`
- Use a smaller model: `--model mobilenet_v3_small`

### "Import error: torch not found"
- Install PyTorch: `pip install torch torchvision`

## Integration with GeoVue

The trained model can be integrated back into GeoVue for automatic wet/dry prediction during compartment extraction. See the main CHANGELOG for ML pipeline integration notes.
