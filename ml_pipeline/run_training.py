#!/usr/bin/env python
"""
GeoVue ML Pipeline - Quick Start Training Script

This script provides an easy way to train the compartment classifier
on your approved compartment images.

Usage:
    python run_training.py --shared-folder "C:/path/to/shared/folder"

Or configure paths interactively:
    python run_training.py --interactive
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_pipeline.config import create_config, MLPipelineConfig
from ml_pipeline.data_loader import CompartmentDataset
from ml_pipeline.train import train


def interactive_setup() -> MLPipelineConfig:
    """Interactive configuration setup."""
    print("=" * 60)
    print("GeoVue ML Pipeline - Interactive Setup")
    print("=" * 60)
    print()

    # Get shared folder path
    print("Please provide the path to your GeoVue shared folder.")
    print("This folder should contain:")
    print("  - Extracted Compartment Images/Approved Compartment Images/")
    print("  - Chip Tray Register/Register Data (Do not edit)/")
    print()

    shared_folder = input("Shared folder path: ").strip().strip('"')

    if not shared_folder:
        print("Error: Shared folder path is required.")
        sys.exit(1)

    shared_path = Path(shared_folder)
    if not shared_path.exists():
        print(f"Warning: Path does not exist: {shared_path}")
        confirm = input("Continue anyway? (y/n): ").strip().lower()
        if confirm != "y":
            sys.exit(1)

    # Check for approved compartments
    approved_path = shared_path / "Extracted Compartment Images" / "Approved Compartment Images"
    if not approved_path.exists():
        print(f"Warning: Approved compartments folder not found: {approved_path}")
        alt_path = input("Enter alternative approved compartments path (or Enter to continue): ").strip().strip('"')
        if alt_path:
            approved_path = Path(alt_path)

    # Training parameters
    print("\n--- Training Parameters ---")
    print("Press Enter to use defaults.")

    model_name = input("Model architecture [resnet18]: ").strip() or "resnet18"
    epochs = input("Number of epochs [50]: ").strip()
    epochs = int(epochs) if epochs else 50
    batch_size = input("Batch size [32]: ").strip()
    batch_size = int(batch_size) if batch_size else 32
    lr = input("Learning rate [0.001]: ").strip()
    lr = float(lr) if lr else 0.001

    output_dir = input("Output directory [ml_output]: ").strip() or "ml_output"

    # Create config
    config = create_config(
        shared_folder=str(shared_path),
        approved_compartments=str(approved_path) if approved_path.exists() else None,
        model_name=model_name,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        output_dir=Path(output_dir),
        checkpoint_dir=Path(output_dir) / "checkpoints",
        logs_dir=Path(output_dir) / "logs",
    )

    return config


def quick_data_check(config: MLPipelineConfig) -> bool:
    """Quick check if training data is available."""
    print("\n--- Data Check ---")

    if not config.approved_compartments_path:
        print("Error: No approved compartments path configured.")
        return False

    if not config.approved_compartments_path.exists():
        print(f"Error: Path does not exist: {config.approved_compartments_path}")
        return False

    # Quick scan
    dataset = CompartmentDataset(config)
    try:
        num_samples = dataset.scan_approved_compartments()
        missing_classes = dataset.missing_classes()
        if missing_classes:
            print(
                "\nMissing required class samples: "
                f"{', '.join(missing_classes)}"
            )
            print("You need images named like: HoleID_CC_001_Empty.png")
            return False

        print("\nPer-class hole counts:")
        for label in dataset.get_class_names():
            holes = {s.hole_id for s in dataset.samples if s.label == label}
            print(f"  - {label}: {len(holes)} holes, {dataset.stats[label]} samples")

        if num_samples < 10:
            print(f"\nWarning: Only {num_samples} labeled samples found.")
            print("You need images in class-labelled folders or named with a class suffix.")
            confirm = input("Continue with limited data? (y/n): ").strip().lower()
            return confirm == "y"
        return True
    except Exception as e:
        print(f"Error scanning data: {e}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="GeoVue ML Pipeline - Compartment Classifier Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_training.py --shared-folder "C:/GeoVue/Shared"
  python run_training.py --interactive
  python run_training.py --shared-folder "C:/GeoVue/Shared" --model resnet34 --epochs 100
        """,
    )

    parser.add_argument(
        "--shared-folder",
        type=str,
        help="Path to GeoVue shared folder",
    )
    parser.add_argument(
        "--approved-compartments",
        type=str,
        help="Direct path to approved compartments folder",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive setup mode",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50", "efficientnet_b0", "mobilenet_v3_small"],
        help="Model architecture (default: resnet18)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ml_output",
        help="Output directory (default: ml_output)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable live visualization",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        help="Comma-separated class names. Use when training from arbitrary class folders.",
    )
    parser.add_argument(
        "--auto-class-folders",
        action="store_true",
        help="Use immediate subfolder names under --approved-compartments as class names.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not load ImageNet pretrained weights.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes.",
    )
    parser.add_argument(
        "--data-check-only",
        action="store_true",
        help="Only check data, don't train",
    )

    args = parser.parse_args()

    # Interactive or command-line setup
    if args.interactive:
        config = interactive_setup()
    elif args.shared_folder or args.approved_compartments:
        class_names = None
        if args.class_names:
            class_names = [name.strip() for name in args.class_names.split(",") if name.strip()]

        config = create_config(
            shared_folder=args.shared_folder,
            approved_compartments=args.approved_compartments,
            model_name=args.model,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            class_names=class_names,
            auto_class_names_from_folders=args.auto_class_folders,
            pretrained=not args.no_pretrained,
            num_workers=args.num_workers,
            output_dir=Path(args.output_dir),
            checkpoint_dir=Path(args.output_dir) / "checkpoints",
            logs_dir=Path(args.output_dir) / "logs",
        )
    else:
        # Try auto-detection from GeoVue config
        print("Attempting to load paths from GeoVue config...")
        config = create_config(
            model_name=args.model,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            pretrained=not args.no_pretrained,
            num_workers=args.num_workers,
            output_dir=Path(args.output_dir),
            checkpoint_dir=Path(args.output_dir) / "checkpoints",
            logs_dir=Path(args.output_dir) / "logs",
        )

        if not config.approved_compartments_path:
            print("\nCould not auto-detect paths. Please use:")
            print("  --shared-folder to specify your GeoVue shared folder")
            print("  --interactive for interactive setup")
            sys.exit(1)

    # Data check
    if not quick_data_check(config):
        sys.exit(1)

    if args.data_check_only:
        print("\nData check complete. Use --help for training options.")
        sys.exit(0)

    # Train
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)

    try:
        model, history = train(config, visualize=not args.no_viz)
        print("\nTraining completed successfully!")
        print(f"Best model saved to: {config.checkpoint_dir / 'best_model.pt'}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
