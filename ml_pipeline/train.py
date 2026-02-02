"""
Training script for ML Pipeline.

Trains wet/dry classifier with progress visualization and early stopping.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .config import MLPipelineConfig, save_config
from .data_loader import CompartmentDataset, create_data_loaders
from .model import WetDryClassifier, create_model, save_checkpoint
from .visualize import TrainingVisualizer


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


class TrainingHistory:
    """Track training metrics over epochs."""

    def __init__(self):
        self.train_loss: List[float] = []
        self.train_acc: List[float] = []
        self.val_loss: List[float] = []
        self.val_acc: List[float] = []
        self.learning_rates: List[float] = []
        self.epoch_times: List[float] = []

    def update(
        self,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float,
        epoch_time: float,
    ) -> None:
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)

    def to_dict(self) -> Dict:
        return {
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
            "learning_rates": self.learning_rates,
            "epoch_times": self.epoch_times,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


if TORCH_AVAILABLE:

    def train_epoch(
        model: WetDryClassifier,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
    ) -> Tuple[float, float]:
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

        avg_loss = running_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate(
        model: WetDryClassifier,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> Tuple[float, float]:
        """Validate the model."""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

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

        avg_loss = running_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def train(
        config: MLPipelineConfig,
        visualize: bool = True,
        save_every: int = 5,
    ) -> Tuple[WetDryClassifier, TrainingHistory]:
        """
        Train the wet/dry classifier.

        Args:
            config: Pipeline configuration
            visualize: Whether to show live training plots
            save_every: Save checkpoint every N epochs

        Returns:
            Trained model and training history
        """
        print("=" * 60)
        print("GeoVue ML Pipeline - Wet/Dry Classifier Training")
        print("=" * 60)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Create output directories
        config.output_dir.mkdir(parents=True, exist_ok=True)
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config.logs_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        save_config(config, config.output_dir / "config.json")

        # Load data
        print("Loading data...")
        dataset = CompartmentDataset(config)
        num_samples = dataset.scan_approved_compartments()

        if num_samples < 10:
            raise ValueError(f"Not enough labeled samples: {num_samples}")

        # Export dataset info
        dataset.export_dataset_info(config.output_dir / "dataset_info.json")

        # Split data
        train_samples, val_samples, test_samples = dataset.split_data(
            validation_split=config.validation_split,
            test_split=config.test_split,
        )

        # Create data loaders
        print("\nCreating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            config, train_samples, val_samples, test_samples
        )

        # Create model
        print("\nCreating model...")
        model, device = create_model(config)

        # Loss and optimizer
        class_weights = torch.tensor(dataset.get_class_weights(), dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # Training state
        history = TrainingHistory()
        early_stopping = EarlyStopping(patience=config.early_stopping_patience)
        best_val_acc = 0.0

        # Visualizer
        visualizer = None
        if visualize:
            visualizer = TrainingVisualizer(config.output_dir)
            visualizer.setup()

        print("\n" + "=" * 60)
        print("Training...")
        print("=" * 60)

        for epoch in range(1, config.num_epochs + 1):
            start_time = time.time()

            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device
            )

            # Validate
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            # Update scheduler
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            # Record epoch time
            epoch_time = time.time() - start_time

            # Update history
            history.update(train_loss, train_acc, val_loss, val_acc, current_lr, epoch_time)

            # Print progress
            print(
                f"Epoch {epoch:3d}/{config.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
            )

            # Update visualizer
            if visualizer:
                visualizer.update(history)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(
                    model, optimizer, epoch, val_loss, val_acc,
                    config.checkpoint_dir / "best_model.pt"
                )

            # Periodic checkpoint
            if epoch % save_every == 0:
                save_checkpoint(
                    model, optimizer, epoch, val_loss, val_acc,
                    config.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
                )

            # Early stopping
            if early_stopping(val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

        # Save final model and history
        save_checkpoint(
            model, optimizer, epoch, val_loss, val_acc,
            config.checkpoint_dir / "final_model.pt"
        )
        history.save(config.logs_dir / "training_history.json")

        # Final visualization
        if visualizer:
            visualizer.save_final_plots()
            visualizer.close()

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Model saved to: {config.checkpoint_dir / 'best_model.pt'}")
        print(f"Training history saved to: {config.logs_dir / 'training_history.json'}")

        return model, history


def main():
    """Main entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train wet/dry classifier")
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
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50", "efficientnet_b0", "mobilenet_v3_small"],
        help="Model architecture",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ml_output",
        help="Output directory",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable live visualization",
    )

    args = parser.parse_args()

    # Create config
    from .config import create_config

    config = create_config(
        shared_folder=args.shared_folder,
        approved_compartments=args.approved_compartments,
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=Path(args.output_dir),
        checkpoint_dir=Path(args.output_dir) / "checkpoints",
        logs_dir=Path(args.output_dir) / "logs",
    )

    # Train
    train(config, visualize=not args.no_viz)


if __name__ == "__main__":
    main()
