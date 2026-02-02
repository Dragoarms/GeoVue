"""
Visualization utilities for ML Pipeline.

Provides live training progress plots and final analysis charts.
"""

import json
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np

try:
    import matplotlib
    matplotlib.use("TkAgg")  # Use interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available for visualization")


class TrainingVisualizer:
    """Live training progress visualization."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.fig: Optional[Figure] = None
        self.axes: Dict[str, Axes] = {}
        self.is_setup = False

    def setup(self) -> None:
        """Set up the visualization figure."""
        if not MATPLOTLIB_AVAILABLE:
            return

        plt.ion()  # Interactive mode

        self.fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        self.fig.suptitle("GeoVue ML Pipeline - Training Progress", fontsize=14)

        self.axes = {
            "loss": axes[0, 0],
            "accuracy": axes[0, 1],
            "lr": axes[1, 0],
            "comparison": axes[1, 1],
        }

        # Configure axes
        self.axes["loss"].set_xlabel("Epoch")
        self.axes["loss"].set_ylabel("Loss")
        self.axes["loss"].set_title("Training & Validation Loss")
        self.axes["loss"].grid(True, alpha=0.3)

        self.axes["accuracy"].set_xlabel("Epoch")
        self.axes["accuracy"].set_ylabel("Accuracy (%)")
        self.axes["accuracy"].set_title("Training & Validation Accuracy")
        self.axes["accuracy"].grid(True, alpha=0.3)

        self.axes["lr"].set_xlabel("Epoch")
        self.axes["lr"].set_ylabel("Learning Rate")
        self.axes["lr"].set_title("Learning Rate Schedule")
        self.axes["lr"].grid(True, alpha=0.3)
        self.axes["lr"].set_yscale("log")

        self.axes["comparison"].set_xlabel("Training Accuracy (%)")
        self.axes["comparison"].set_ylabel("Validation Accuracy (%)")
        self.axes["comparison"].set_title("Train vs Validation (Overfitting Check)")
        self.axes["comparison"].grid(True, alpha=0.3)

        plt.tight_layout()
        self.fig.canvas.draw()
        plt.show(block=False)

        self.is_setup = True

    def update(self, history: "TrainingHistory") -> None:
        """Update plots with new training data."""
        if not MATPLOTLIB_AVAILABLE or not self.is_setup:
            return

        epochs = list(range(1, len(history.train_loss) + 1))

        # Clear and redraw
        for ax in self.axes.values():
            ax.clear()

        # Loss plot
        self.axes["loss"].plot(epochs, history.train_loss, "b-", label="Train", linewidth=2)
        self.axes["loss"].plot(epochs, history.val_loss, "r-", label="Validation", linewidth=2)
        self.axes["loss"].set_xlabel("Epoch")
        self.axes["loss"].set_ylabel("Loss")
        self.axes["loss"].set_title("Training & Validation Loss")
        self.axes["loss"].legend()
        self.axes["loss"].grid(True, alpha=0.3)

        # Accuracy plot
        self.axes["accuracy"].plot(epochs, history.train_acc, "b-", label="Train", linewidth=2)
        self.axes["accuracy"].plot(epochs, history.val_acc, "r-", label="Validation", linewidth=2)
        self.axes["accuracy"].set_xlabel("Epoch")
        self.axes["accuracy"].set_ylabel("Accuracy (%)")
        self.axes["accuracy"].set_title("Training & Validation Accuracy")
        self.axes["accuracy"].legend()
        self.axes["accuracy"].grid(True, alpha=0.3)
        self.axes["accuracy"].set_ylim(0, 100)

        # Learning rate plot
        self.axes["lr"].plot(epochs, history.learning_rates, "g-", linewidth=2)
        self.axes["lr"].set_xlabel("Epoch")
        self.axes["lr"].set_ylabel("Learning Rate")
        self.axes["lr"].set_title("Learning Rate Schedule")
        self.axes["lr"].grid(True, alpha=0.3)
        self.axes["lr"].set_yscale("log")

        # Overfitting check
        self.axes["comparison"].plot(
            history.train_acc, history.val_acc, "ko-", markersize=4, alpha=0.7
        )
        # Draw diagonal line (perfect = no overfitting)
        max_acc = max(max(history.train_acc), max(history.val_acc))
        self.axes["comparison"].plot([0, max_acc], [0, max_acc], "r--", alpha=0.5, label="y=x")
        self.axes["comparison"].set_xlabel("Training Accuracy (%)")
        self.axes["comparison"].set_ylabel("Validation Accuracy (%)")
        self.axes["comparison"].set_title("Train vs Validation (Overfitting Check)")
        self.axes["comparison"].grid(True, alpha=0.3)
        self.axes["comparison"].set_aspect("equal")

        # Add current stats text
        if len(epochs) > 0:
            current_epoch = epochs[-1]
            stats_text = (
                f"Epoch: {current_epoch}\n"
                f"Train Acc: {history.train_acc[-1]:.2f}%\n"
                f"Val Acc: {history.val_acc[-1]:.2f}%\n"
                f"Best Val: {max(history.val_acc):.2f}%"
            )
            self.axes["comparison"].text(
                0.02, 0.98, stats_text,
                transform=self.axes["comparison"].transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save_final_plots(self) -> None:
        """Save final visualization plots."""
        if not MATPLOTLIB_AVAILABLE or not self.is_setup:
            return

        # Save the current figure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(
            self.output_dir / "training_progress.png",
            dpi=150,
            bbox_inches="tight",
        )
        print(f"Training plots saved to: {self.output_dir / 'training_progress.png'}")

    def close(self) -> None:
        """Close the visualization window."""
        if MATPLOTLIB_AVAILABLE and self.fig is not None:
            plt.ioff()
            plt.close(self.fig)


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    output_path: Optional[Path] = None,
    title: str = "Confusion Matrix",
) -> None:
    """Plot confusion matrix."""
    if not MATPLOTLIB_AVAILABLE:
        return

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Raw counts
    im1 = ax1.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax1.figure.colorbar(im1, ax=ax1)
    ax1.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=f"{title} (Counts)",
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    # Normalized
    im2 = ax2.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax2.figure.colorbar(im2, ax=ax2)
    ax2.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=f"{title} (Normalized)",
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Add text annotations
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            ax2.text(
                j, i, format(cm_normalized[i, j], ".2%"),
                ha="center", va="center",
                color="white" if cm_normalized[i, j] > 0.5 else "black",
            )

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to: {output_path}")

    plt.show()


def plot_training_history_from_file(history_path: Path, output_path: Optional[Path] = None) -> None:
    """Load and plot training history from JSON file."""
    if not MATPLOTLIB_AVAILABLE:
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Training History Analysis", fontsize=14)

    # Loss
    axes[0, 0].plot(epochs, history["train_loss"], "b-", label="Train", linewidth=2)
    axes[0, 0].plot(epochs, history["val_loss"], "r-", label="Validation", linewidth=2)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training & Validation Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(epochs, history["train_acc"], "b-", label="Train", linewidth=2)
    axes[0, 1].plot(epochs, history["val_acc"], "r-", label="Validation", linewidth=2)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].set_title("Training & Validation Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Learning rate
    axes[1, 0].plot(epochs, history["learning_rates"], "g-", linewidth=2)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Learning Rate")
    axes[1, 0].set_title("Learning Rate Schedule")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale("log")

    # Epoch time
    axes[1, 1].bar(epochs, history["epoch_times"], color="steelblue", alpha=0.7)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Time (seconds)")
    axes[1, 1].set_title("Epoch Training Time")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"History plot saved to: {output_path}")

    plt.show()


def plot_sample_predictions(
    images: List[np.ndarray],
    true_labels: List[str],
    pred_labels: List[str],
    confidences: List[float],
    output_path: Optional[Path] = None,
    num_samples: int = 16,
) -> None:
    """Plot sample predictions with images."""
    if not MATPLOTLIB_AVAILABLE:
        return

    num_samples = min(num_samples, len(images))
    cols = 4
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten()

    for i in range(num_samples):
        ax = axes[i]
        ax.imshow(images[i])
        ax.axis("off")

        # Color based on correctness
        correct = true_labels[i] == pred_labels[i]
        color = "green" if correct else "red"

        ax.set_title(
            f"True: {true_labels[i]}\nPred: {pred_labels[i]} ({confidences[i]:.1%})",
            fontsize=9,
            color=color,
        )

    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Sample Predictions", fontsize=14)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Sample predictions saved to: {output_path}")

    plt.show()
