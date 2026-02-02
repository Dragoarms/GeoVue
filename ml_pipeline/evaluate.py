"""
Evaluation script for ML Pipeline.

Evaluates trained model on test set with detailed metrics.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report,
        confusion_matrix,
        roc_auc_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install for detailed metrics.")

from .config import MLPipelineConfig, load_config
from .data_loader import CompartmentDataset, create_data_loaders, CompartmentSample
from .model import load_checkpoint
from .visualize import plot_confusion_matrix, plot_sample_predictions


if TORCH_AVAILABLE:

    def evaluate_model(
        model_path: Path,
        config: MLPipelineConfig,
        output_dir: Optional[Path] = None,
    ) -> Dict:
        """
        Evaluate trained model on test set.

        Args:
            model_path: Path to trained model checkpoint
            config: Pipeline configuration
            output_dir: Directory to save evaluation results

        Returns:
            Dictionary of evaluation metrics
        """
        if output_dir is None:
            output_dir = config.output_dir / "evaluation"

        output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("GeoVue ML Pipeline - Model Evaluation")
        print("=" * 60)

        # Load model
        print(f"\nLoading model from: {model_path}")
        model, checkpoint = load_checkpoint(model_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        print(f"Model: {checkpoint.get('model_name', 'unknown')}")
        print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"Checkpoint accuracy: {checkpoint.get('accuracy', 0):.2f}%")

        # Load test data
        print("\nLoading test data...")
        dataset = CompartmentDataset(config)
        dataset.scan_approved_compartments()
        train_samples, val_samples, test_samples = dataset.split_data(
            validation_split=config.validation_split,
            test_split=config.test_split,
        )

        _, _, test_loader = create_data_loaders(
            config, train_samples, val_samples, test_samples
        )

        # Evaluate
        print("\nRunning evaluation on test set...")
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)

                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of "Wet"

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Calculate metrics
        metrics = {}

        metrics["accuracy"] = accuracy_score(all_labels, all_preds) * 100
        metrics["precision_macro"] = precision_score(all_labels, all_preds, average="macro") * 100
        metrics["recall_macro"] = recall_score(all_labels, all_preds, average="macro") * 100
        metrics["f1_macro"] = f1_score(all_labels, all_preds, average="macro") * 100

        # Per-class metrics
        metrics["precision_dry"] = precision_score(all_labels, all_preds, pos_label=0) * 100
        metrics["recall_dry"] = recall_score(all_labels, all_preds, pos_label=0) * 100
        metrics["f1_dry"] = f1_score(all_labels, all_preds, pos_label=0) * 100

        metrics["precision_wet"] = precision_score(all_labels, all_preds, pos_label=1) * 100
        metrics["recall_wet"] = recall_score(all_labels, all_preds, pos_label=1) * 100
        metrics["f1_wet"] = f1_score(all_labels, all_preds, pos_label=1) * 100

        # ROC AUC
        try:
            metrics["roc_auc"] = roc_auc_score(all_labels, all_probs) * 100
        except ValueError:
            metrics["roc_auc"] = 0.0

        # Print results
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.2f}%")
        print(f"  Precision: {metrics['precision_macro']:.2f}%")
        print(f"  Recall:    {metrics['recall_macro']:.2f}%")
        print(f"  F1 Score:  {metrics['f1_macro']:.2f}%")
        print(f"  ROC AUC:   {metrics['roc_auc']:.2f}%")

        print(f"\nPer-Class Metrics:")
        print(f"  Dry  - Precision: {metrics['precision_dry']:.2f}%, Recall: {metrics['recall_dry']:.2f}%, F1: {metrics['f1_dry']:.2f}%")
        print(f"  Wet  - Precision: {metrics['precision_wet']:.2f}%, Recall: {metrics['recall_wet']:.2f}%, F1: {metrics['f1_wet']:.2f}%")

        # Classification report
        print("\nDetailed Classification Report:")
        print(classification_report(
            all_labels, all_preds,
            target_names=["Dry", "Wet"],
            digits=4,
        ))

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        print("\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"              Dry    Wet")
        print(f"  Actual Dry  {cm[0, 0]:5d}  {cm[0, 1]:5d}")
        print(f"  Actual Wet  {cm[1, 0]:5d}  {cm[1, 1]:5d}")

        # Save results
        results = {
            "model_path": str(model_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_accuracy": checkpoint.get("accuracy"),
            "test_samples": len(test_samples),
            "metrics": metrics,
            "confusion_matrix": cm.tolist(),
        }

        results_path = output_dir / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

        # Plot confusion matrix
        plot_confusion_matrix(
            all_labels.tolist(),
            all_preds.tolist(),
            ["Dry", "Wet"],
            output_path=output_dir / "confusion_matrix.png",
            title="Test Set Confusion Matrix",
        )

        return metrics

    def predict_single(
        model_path: Path,
        image_path: Path,
        config: Optional[MLPipelineConfig] = None,
    ) -> Tuple[str, float]:
        """
        Predict wet/dry for a single image.

        Args:
            model_path: Path to trained model
            image_path: Path to compartment image

        Returns:
            Tuple of (label, confidence)
        """
        from PIL import Image
        from torchvision import transforms

        # Load model
        model, _ = load_checkpoint(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        # Load and preprocess image
        image_size = config.image_size if config else 224
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            confidence, predicted = probs.max(1)

        label = "Wet" if predicted.item() == 1 else "Dry"
        return label, confidence.item()

    def predict_batch(
        model_path: Path,
        image_paths: List[Path],
        config: Optional[MLPipelineConfig] = None,
    ) -> List[Tuple[str, float]]:
        """
        Predict wet/dry for multiple images.

        Args:
            model_path: Path to trained model
            image_paths: List of paths to compartment images

        Returns:
            List of (label, confidence) tuples
        """
        from PIL import Image
        from torchvision import transforms

        # Load model
        model, _ = load_checkpoint(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        # Preprocess
        image_size = config.image_size if config else 224
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        results = []

        for image_path in image_paths:
            try:
                image = Image.open(image_path).convert("RGB")
                input_tensor = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    probs = F.softmax(output, dim=1)
                    confidence, predicted = probs.max(1)

                label = "Wet" if predicted.item() == 1 else "Dry"
                results.append((label, confidence.item()))
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append(("Error", 0.0))

        return results


def main():
    """Main entry point for evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate wet/dry classifier")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config JSON file",
    )
    parser.add_argument(
        "--shared-folder",
        type=str,
        help="Path to GeoVue shared folder",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ml_output/evaluation",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Load config
    if args.config:
        config = load_config(Path(args.config))
    else:
        from .config import create_config
        config = create_config(shared_folder=args.shared_folder)

    # Evaluate
    evaluate_model(
        Path(args.model_path),
        config,
        Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
