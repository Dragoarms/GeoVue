"""
Train 4-Class Cell Classifier

Classes: Dry, Wet, Empty, Bad

Usage:
    python train_classifier.py --epochs 50 --model mobilenet_v3_small
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from cell_classifier import (
    ClassifierConfig,
    CellDataset,
    CellLabels,
    CellClassifierModel,
    create_data_loaders,
)


def train_epoch(
    model: nn.Module,
    train_loader,
    criterion,
    optimizer,
    device,
) -> tuple:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}, "
                  f"Loss: {loss.item():.4f}, "
                  f"Acc: {100.*correct/total:.2f}%")

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader,
    criterion,
    device,
) -> tuple:
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # Per-class tracking
    class_correct = {i: 0 for i in range(4)}
    class_total = {i: 0 for i in range(4)}

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1

    avg_loss = running_loss / len(val_loader)
    accuracy = 100. * correct / total

    # Print per-class accuracy
    print("  Per-class accuracy:")
    for idx, label in CellLabels.IDX_TO_LABEL.items():
        if class_total[idx] > 0:
            class_acc = 100. * class_correct[idx] / class_total[idx]
            print(f"    {label}: {class_acc:.2f}% ({class_correct[idx]}/{class_total[idx]})")

    return avg_loss, accuracy


def train(config: ClassifierConfig):
    """Main training loop."""
    print("=" * 60)
    print("4-Class Cell Classifier Training")
    print("=" * 60)

    # Setup device
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print("\n--- Loading Data ---")
    dataset = CellDataset(config)
    dataset.load_bad_status_from_registers()
    dataset.scan_images()

    if len(dataset.samples) == 0:
        print("ERROR: No samples found! Check your data paths.")
        return

    train_samples, val_samples, test_samples = dataset.split_data()

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        config, train_samples, val_samples, test_samples
    )

    # Create model
    print("\n--- Creating Model ---")
    model = CellClassifierModel(
        model_name=config.model_name,
        num_classes=config.num_classes,
        pretrained=config.pretrained,
        dropout_rate=config.dropout_rate,
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {config.model_name}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function with class weights
    class_weights = torch.tensor(dataset.get_class_weights(), dtype=torch.float32).to(device)
    print(f"\nClass weights: {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    # Early stopping
    best_val_acc = 0.0
    patience_counter = 0

    # Output directory
    output_dir = config.output_path or Path(__file__).parent / "classifier_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    print(f"\nOutput directory: {output_dir}")
    print("\n--- Starting Training ---")

    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}")
        print("-" * 40)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

        # Update scheduler
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']

        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            # Save best model
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "model_name": config.model_name,
                "num_classes": config.num_classes,
            }
            torch.save(checkpoint, checkpoints_dir / "best_model.pt")
            print(f"  -> New best model saved! (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

        # Periodic checkpoint
        if epoch % 10 == 0:
            torch.save(checkpoint, checkpoints_dir / f"epoch_{epoch}.pt")

    # Save final model
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
        "val_loss": val_loss,
        "model_name": config.model_name,
        "num_classes": config.num_classes,
    }
    torch.save(checkpoint, checkpoints_dir / "final_model.pt")

    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save config
    config_dict = {
        "model_name": config.model_name,
        "num_classes": config.num_classes,
        "image_size": config.image_size,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "epochs": config.epochs,
        "best_val_acc": best_val_acc,
        "timestamp": datetime.now().isoformat(),
        "class_labels": CellLabels.IDX_TO_LABEL,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {checkpoints_dir / 'best_model.pt'}")
    print("=" * 60)

    # Final evaluation on test set
    print("\n--- Test Set Evaluation ---")
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Train 4-Class Cell Classifier")
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenet_v3_small",
        choices=["mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0", "resnet18"],
        help="Model architecture"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--image-size", type=int, default=224, help="Image size")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda)")

    # Data paths
    parser.add_argument(
        "--images-path",
        type=str,
        default=r"C:\Users\georg\Pictures\Shared folder EX\Extracted Compartment Images\Approved Compartment Images",
        help="Path to approved compartment images"
    )
    parser.add_argument(
        "--register-path",
        type=str,
        default=r"C:\Users\georg\Pictures\Shared folder EX\Chip Tray Register\Register Data (Do not edit)",
        help="Path to register data (for Bad status)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--empty-images-path",
        type=str,
        default=r"C:\GeoVue Chip Tray Photos\empty",
        help="Path to empty cell images"
    )
    parser.add_argument(
        "--additional-paths",
        type=str,
        nargs="*",
        default=[],
        help="Additional image folders to scan"
    )

    args = parser.parse_args()

    # Build list of additional image paths
    additional_paths = []
    if args.empty_images_path:
        additional_paths.append(Path(args.empty_images_path))
    for p in args.additional_paths:
        additional_paths.append(Path(p))

    config = ClassifierConfig(
        approved_images_path=Path(args.images_path),
        additional_image_paths=additional_paths if additional_paths else None,
        register_data_path=Path(args.register_path),
        output_path=Path(args.output_path) if args.output_path else None,
        model_name=args.model,
        image_size=args.image_size,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        device=args.device,
    )

    train(config)


if __name__ == "__main__":
    main()
