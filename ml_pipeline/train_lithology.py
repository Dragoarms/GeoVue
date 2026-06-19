"""Train a multi-class lithology classifier from a GeoVue manifest."""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .config import MLPipelineConfig, save_config
from .lithology_manifest import load_manifest_csv, summarise_rows

try:
    from .visualize import TrainingVisualizer
except Exception:
    TrainingVisualizer = None

if TORCH_AVAILABLE:
    from .data_loader import CompartmentDataset, CompartmentSample, create_data_loaders
    from .model import create_model, save_checkpoint
else:
    from .data_loader import CompartmentDataset, CompartmentSample


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
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []
        self.learning_rates = []
        self.epoch_times = []

    def update(self, train_loss, train_acc, val_loss, val_acc, lr, epoch_time) -> None:
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)

    def to_dict(self) -> dict:
        return {
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
            "learning_rates": self.learning_rates,
            "epoch_times": self.epoch_times,
        }

    def save(self, path: Path) -> None:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def _resolve_class_names(manifest_path: Path, requested: Iterable[str] | None = None) -> list[str]:
    if requested:
        return [str(name).strip() for name in requested if str(name).strip()]
    labels = sorted({row.label for row in load_manifest_csv(manifest_path) if row.label})
    if not labels:
        raise ValueError(f"No labels found in manifest: {manifest_path}")
    return labels


def load_manifest_dataset(
    manifest_path: Path,
    class_names: list[str],
) -> CompartmentDataset:
    """Load manifest rows into the existing CompartmentDataset split machinery."""
    rows = load_manifest_csv(manifest_path)
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}

    config = MLPipelineConfig(class_names=class_names, num_classes=len(class_names))
    dataset = CompartmentDataset(config)
    dataset.samples = []
    dataset.stats = dataset._empty_stats()

    missing_files = 0
    skipped_labels = Counter()
    for row in rows:
        if row.label not in label_to_idx:
            skipped_labels[row.label] += 1
            continue
        path = Path(row.image_path)
        if not path.exists():
            missing_files += 1
            continue
        dataset.samples.append(
            CompartmentSample(
                path=path,
                label=row.label,
                hole_id=row.hole_id or path.stem,
                depth=row.depth_to if row.depth_to is not None else 0.0,
                label_idx=label_to_idx[row.label],
            )
        )
        dataset.stats[row.label] += 1

    if missing_files:
        print(f"Skipped {missing_files} manifest row(s) with missing image files.")
    if skipped_labels:
        print(f"Skipped labels outside selected classes: {dict(skipped_labels)}")

    return dataset


if TORCH_AVAILABLE:

    def _train_epoch(model, train_loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
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
        return running_loss / max(1, len(train_loader)), 100.0 * correct / max(1, total)


    def _validate(model, val_loader, criterion, device):
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
        return running_loss / max(1, len(val_loader)), 100.0 * correct / max(1, total)


    def train_from_manifest(
        manifest_path: Path,
        config: MLPipelineConfig,
        *,
        visualize: bool = False,
        save_every: int = 5,
    ):
        """Train a classifier using image paths and labels from a manifest CSV."""
        print("=" * 60)
        print("GeoVue ML Pipeline - Lithology Classifier Training")
        print("=" * 60)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Manifest: {manifest_path}")

        config.output_dir.mkdir(parents=True, exist_ok=True)
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config.logs_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, config.output_dir / "config.json")
        shutil.copy2(manifest_path, config.output_dir / "training_manifest.csv")

        dataset = load_manifest_dataset(manifest_path, config.class_names or [])
        if len(dataset.samples) < 10:
            raise ValueError(f"Not enough labelled samples: {len(dataset.samples)}")

        missing_classes = dataset.missing_classes()
        if missing_classes:
            raise ValueError(
                "Missing training samples for class(es): "
                f"{', '.join(missing_classes)}"
            )

        print("\nDataset summary:")
        print(summarise_rows(load_manifest_csv(manifest_path)))
        dataset.export_dataset_info(config.output_dir / "dataset_info.json")

        train_samples, val_samples, test_samples = dataset.split_data(
            validation_split=config.validation_split,
            test_split=config.test_split,
        )
        print("\nCreating data loaders...")
        train_loader, val_loader, _ = create_data_loaders(
            config, train_samples, val_samples, test_samples
        )

        print("\nCreating model...")
        model, device = create_model(config)
        class_weights = torch.tensor(dataset.get_class_weights(), dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        history = TrainingHistory()
        early_stopping = EarlyStopping(patience=config.early_stopping_patience)
        best_val_acc = 0.0
        visualizer = None
        if visualize:
            if TrainingVisualizer is None:
                raise RuntimeError("Training visualization dependencies are not available.")
            visualizer = TrainingVisualizer(config.output_dir)
            visualizer.setup()

        print("\n" + "=" * 60)
        print("Training...")
        print("=" * 60)

        final_epoch = 0
        final_val_loss = 0.0
        final_val_acc = 0.0
        for epoch in range(1, config.num_epochs + 1):
            start = datetime.now()
            train_loss, train_acc = _train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = _validate(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            lr = optimizer.param_groups[0]["lr"]
            epoch_time = (datetime.now() - start).total_seconds()
            history.update(train_loss, train_acc, val_loss, val_acc, lr, epoch_time)

            print(
                f"Epoch {epoch:3d}/{config.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                f"LR: {lr:.2e} | Time: {epoch_time:.1f}s",
                flush=True,
            )
            if visualizer:
                visualizer.update(history)

            final_epoch = epoch
            final_val_loss = val_loss
            final_val_acc = val_acc
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    val_loss,
                    val_acc,
                    config.checkpoint_dir / "best_model.pt",
                    dataset.get_class_names(),
                )
            if epoch % save_every == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    val_loss,
                    val_acc,
                    config.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt",
                    dataset.get_class_names(),
                )
            if early_stopping(val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

        save_checkpoint(
            model,
            optimizer,
            final_epoch,
            final_val_loss,
            final_val_acc,
            config.checkpoint_dir / "final_model.pt",
            dataset.get_class_names(),
        )
        history.save(config.logs_dir / "training_history.json")
        if visualizer:
            visualizer.save_final_plots()
            visualizer.close()

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Model saved to: {config.checkpoint_dir / 'best_model.pt'}")
        return model, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("ml_output/lithology_classifier"))
    parser.add_argument("--classes", default=None, help="Comma-separated class labels in output order.")
    parser.add_argument(
        "--model",
        default="mobilenet_v3_small",
        choices=["resnet18", "resnet34", "resnet50", "efficientnet_b0", "mobilenet_v3_small"],
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--data-check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    if not TORCH_AVAILABLE:
        raise SystemExit("PyTorch is not available. Install torch and torchvision first.")

    args = parse_args()
    class_names = _resolve_class_names(
        args.manifest,
        [part.strip() for part in args.classes.split(",")] if args.classes else None,
    )
    config = MLPipelineConfig(
        model_name=args.model,
        class_names=class_names,
        num_classes=len(class_names),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=args.image_size,
        num_workers=args.num_workers,
        validation_split=args.validation_split,
        test_split=args.test_split,
        pretrained=not args.no_pretrained,
        output_dir=args.output_dir,
        checkpoint_dir=args.output_dir / "checkpoints",
        logs_dir=args.output_dir / "logs",
    )

    dataset = load_manifest_dataset(args.manifest, class_names)
    print(f"Manifest contains {len(dataset.samples)} usable samples.")
    print(f"Class names: {', '.join(class_names)}")
    for label in class_names:
        holes = {sample.hole_id for sample in dataset.samples if sample.label == label}
        print(f"  - {label}: {dataset.stats[label]} samples across {len(holes)} holes")

    if args.data_check_only:
        return

    train_from_manifest(args.manifest, config, visualize=not args.no_viz)


if __name__ == "__main__":
    main()
