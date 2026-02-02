"""
Batch prediction script for pending compartment images.
Runs the wet/dry classifier on unlabeled images and outputs results.
"""

import csv
import sys
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_pipeline.evaluate import predict_batch


def find_pending_images(pending_folder: Path) -> list:
    """Find all PNG images in the pending folder."""
    images = list(pending_folder.rglob("*.png"))
    print(f"Found {len(images)} images to classify")
    return images


def run_batch_predictions(
    model_path: Path,
    pending_folder: Path,
    output_csv: Path,
) -> None:
    """Run predictions on all pending images and save to CSV."""

    print("=" * 60)
    print("GeoVue ML Pipeline - Batch Wet/Dry Classification")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Pending folder: {pending_folder}")
    print()

    # Find images
    image_paths = find_pending_images(pending_folder)

    if not image_paths:
        print("No images found!")
        return

    # Run predictions
    print("\nRunning predictions...")
    results = predict_batch(model_path, image_paths)

    # Collect statistics
    dry_count = sum(1 for r in results if r[0] == "Dry")
    wet_count = sum(1 for r in results if r[0] == "Wet")
    error_count = sum(1 for r in results if r[0] == "Error")

    high_confidence = sum(1 for r in results if r[1] >= 0.9)
    medium_confidence = sum(1 for r in results if 0.7 <= r[1] < 0.9)
    low_confidence = sum(1 for r in results if r[1] < 0.7 and r[0] != "Error")

    # Write results to CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_path",
            "hole_id",
            "depth_code",
            "predicted_label",
            "confidence",
            "confidence_level"
        ])

        for image_path, (label, confidence) in zip(image_paths, results):
            # Extract hole_id and depth from filename
            filename = image_path.stem  # e.g., OK0012_CC_001_temp
            parts = filename.split("_")
            hole_id = parts[0] if len(parts) > 0 else ""
            depth_code = parts[2] if len(parts) > 2 else ""

            # Confidence level
            if confidence >= 0.9:
                conf_level = "high"
            elif confidence >= 0.7:
                conf_level = "medium"
            else:
                conf_level = "low"

            writer.writerow([
                str(image_path),
                hole_id,
                depth_code,
                label,
                f"{confidence:.4f}",
                conf_level
            ])

    # Print summary
    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"\nTotal images classified: {len(results)}")
    print(f"\nPredictions:")
    print(f"  Dry: {dry_count} ({100*dry_count/len(results):.1f}%)")
    print(f"  Wet: {wet_count} ({100*wet_count/len(results):.1f}%)")
    if error_count > 0:
        print(f"  Errors: {error_count}")

    print(f"\nConfidence Distribution:")
    print(f"  High (>=90%):   {high_confidence} images")
    print(f"  Medium (70-90%): {medium_confidence} images")
    print(f"  Low (<70%):      {low_confidence} images")

    print(f"\nResults saved to: {output_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch classify pending compartment images")
    parser.add_argument(
        "--model",
        type=str,
        default="ml_output/checkpoints/best_model.pt",
        help="Path to trained model",
    )
    parser.add_argument(
        "--pending-folder",
        type=str,
        required=True,
        help="Path to folder with pending images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ml_output/batch_predictions.csv",
        help="Output CSV path",
    )

    args = parser.parse_args()

    run_batch_predictions(
        Path(args.model),
        Path(args.pending_folder),
        Path(args.output),
    )
