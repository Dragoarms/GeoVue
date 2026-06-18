# Lithology Image Baseline Training Notes

## Recommended dataset

Use the CLR-corrected 8-class weathering-aware image set:

`ml_detection/lithology_classifier_images/weathering8_clr_120each`

This set was copied from the approved compartment image source. Source images
were not moved, deleted, renamed, or edited.

## Why this set

- Label definition uses assay oxide CLR features instead of raw closed
  geochemical percentages.
- Gamma is robust-normalised within hole before interval summarisation.
- Depth/weathering features are included in the label-definition pass.
- The copied training set has 8 classes, 120 images per class, and class-level
  hole counts stored in:
  `ml_detection/lithology_classifier_images/weathering8_clr_120each/lithology_image_copy_summary.json`

## Data check

```powershell
python ml_pipeline\run_training.py --approved-compartments ml_detection\lithology_classifier_images\weathering8_clr_120each --auto-class-folders --model mobilenet_v3_small --epochs 1 --batch-size 16 --lr 0.0005 --output-dir ml_output\lithology_weathering8_clr --no-pretrained --num-workers 0 --data-check-only
```

## Smoke training command

This command has been run successfully for one epoch:

```powershell
python ml_pipeline\run_training.py --approved-compartments ml_detection\lithology_classifier_images\weathering8_clr_120each --auto-class-folders --model mobilenet_v3_small --epochs 1 --batch-size 16 --lr 0.0005 --output-dir ml_output\lithology_weathering8_clr_smoke --no-pretrained --num-workers 0 --no-viz
```

## Full baseline training command

Use pretrained weights if available locally or if network access is allowed:

```powershell
python ml_pipeline\run_training.py --approved-compartments ml_detection\lithology_classifier_images\weathering8_clr_120each --auto-class-folders --model mobilenet_v3_small --epochs 50 --batch-size 16 --lr 0.0005 --output-dir ml_output\lithology_weathering8_clr --num-workers 0 --no-viz
```

If pretrained weights cannot be loaded, add `--no-pretrained`.

## Current safeguards

- Class folder names are auto-detected.
- Checkpoints save the real class names.
- Splitting is globally hole-aware: each hole is assigned to only one of
  train/validation/test.
- Dataset info includes per-class hole counts.
