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
        precision_recall_fscore_support,
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

    def _generated_class_names(num_classes: int) -> List[str]:
        """Return legacy fallback names for checkpoints without class metadata."""
        if num_classes == 2:
            return ["Dry", "Wet"]
        if num_classes == 3:
            return ["Dry", "Wet", "Empty"]
        return [f"Class_{idx}" for idx in range(num_classes)]

    def _is_generic_class_names(class_names: List[str]) -> bool:
        """Return True when names are placeholder Class_0, Class_1, ... labels."""
        return all(name == f"Class_{idx}" for idx, name in enumerate(class_names))

    def _resolve_class_names(
        checkpoint: Dict,
        config: Optional[MLPipelineConfig] = None,
    ) -> Tuple[List[str], str]:
        """
        Resolve the authoritative class order.

        Checkpoint metadata wins when it contains real names. If an older
        checkpoint only has generated Class_N names, use config names when
        they have the same length.
        """
        config_names = list(config.class_names or []) if config and config.class_names else []
        checkpoint_names = list(checkpoint.get("class_names") or [])
        num_classes = int(
            checkpoint.get("num_classes")
            or len(checkpoint_names)
            or len(config_names)
            or len(CompartmentDataset.DEFAULT_CLASS_NAMES)
        )

        if not checkpoint_names:
            checkpoint_names = _generated_class_names(num_classes)

        if len(checkpoint_names) != num_classes:
            raise ValueError(
                "Checkpoint class metadata is inconsistent: "
                f"num_classes={num_classes}, class_names={checkpoint_names}"
            )

        if config_names:
            if len(config_names) != num_classes:
                raise ValueError(
                    "Config class metadata is inconsistent with checkpoint: "
                    f"checkpoint num_classes={num_classes}, config class_names={config_names}"
                )
            if checkpoint_names == config_names:
                return checkpoint_names, "checkpoint/config"
            if _is_generic_class_names(checkpoint_names):
                return config_names, "config (checkpoint has generic class names)"
            raise ValueError(
                "Class order mismatch between checkpoint and config.\n"
                f"  checkpoint: {checkpoint_names}\n"
                f"  config:     {config_names}\n"
                "Refusing to evaluate because the confusion matrix would be invalid."
            )

        return checkpoint_names, "checkpoint"

    def _checkpoint_class_names(checkpoint: Dict) -> List[str]:
        """Return class names from checkpoint metadata with a sane fallback."""
        return _resolve_class_names(checkpoint)[0]

    def _resolve_device(config: MLPipelineConfig) -> torch.device:
        """Resolve evaluation device with a helpful CUDA error."""
        if config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        device = torch.device(config.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "Config requested CUDA, but torch.cuda.is_available() is false. "
                "This usually means the installed PyTorch build is CPU-only."
            )
        return device

    def _infer_config_path(model_path: Path) -> Optional[Path]:
        """Infer the saved config path from a checkpoint path."""
        candidates = [
            model_path.parent.parent / "config.json",
            model_path.parent / "config.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _label_for_index(class_names: List[str], label_idx: int) -> str:
        if 0 <= label_idx < len(class_names):
            return class_names[label_idx]
        return f"Class_{label_idx}"

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

        class_names, class_names_source = _resolve_class_names(checkpoint, config)
        config.class_names = class_names
        config.num_classes = len(class_names)

        device = _resolve_device(config)
        model = model.to(device)
        model.eval()

        print(f"Model: {checkpoint.get('model_name', 'unknown')}")
        print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"Checkpoint accuracy: {checkpoint.get('accuracy', 0):.2f}%")
        print(f"Class order source: {class_names_source}")
        print("Class order:")
        for idx, class_name in enumerate(class_names):
            print(f"  {idx}: {class_name}")
        print(f"Using device: {device}")

        # Load test data
        print("\nLoading test data...")
        dataset = CompartmentDataset(config)
        dataset.scan_approved_compartments()
        dataset_class_names = dataset.get_class_names()
        if dataset_class_names != class_names:
            raise ValueError(
                "Dataset class order does not match checkpoint/config class order.\n"
                f"  dataset:    {dataset_class_names}\n"
                f"  evaluation: {class_names}\n"
                "Refusing to evaluate because the confusion matrix would be invalid."
            )

        train_samples, val_samples, test_samples = dataset.split_data(
            validation_split=config.validation_split,
            test_split=config.test_split,
        )
        if not test_samples:
            raise ValueError("No test samples were selected; cannot evaluate the model.")

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
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        max_observed_idx = max(
            int(all_labels.max()) if all_labels.size else -1,
            int(all_preds.max()) if all_preds.size else -1,
        )
        if max_observed_idx >= len(class_names):
            raise ValueError(
                f"Observed label index {max_observed_idx}, but class order only has "
                f"{len(class_names)} classes: {class_names}"
            )
        if all_probs.ndim == 2 and all_probs.shape[1] != len(class_names):
            raise ValueError(
                "Model output width does not match class order: "
                f"output={all_probs.shape[1]}, classes={len(class_names)}"
            )
        label_indices = list(range(len(class_names)))

        # Calculate metrics
        metrics = {}

        metrics["accuracy"] = accuracy_score(all_labels, all_preds) * 100
        metrics["precision_macro"] = precision_score(
            all_labels, all_preds, labels=label_indices, average="macro", zero_division=0
        ) * 100
        metrics["recall_macro"] = recall_score(
            all_labels, all_preds, labels=label_indices, average="macro", zero_division=0
        ) * 100
        metrics["f1_macro"] = f1_score(
            all_labels, all_preds, labels=label_indices, average="macro", zero_division=0
        ) * 100

        # Per-class metrics
        per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
            all_labels,
            all_preds,
            labels=label_indices,
            zero_division=0,
        )
        metrics["per_class"] = {}
        for idx, label in enumerate(class_names):
            key = label.lower()
            metrics["per_class"][label] = {
                "precision": per_precision[idx] * 100,
                "recall": per_recall[idx] * 100,
                "f1": per_f1[idx] * 100,
                "support": int(per_support[idx]),
            }
            metrics[f"precision_{key}"] = per_precision[idx] * 100
            metrics[f"recall_{key}"] = per_recall[idx] * 100
            metrics[f"f1_{key}"] = per_f1[idx] * 100

        # ROC AUC
        try:
            if all_probs.ndim == 2 and all_probs.shape[1] == 2:
                metrics["roc_auc"] = roc_auc_score(all_labels, all_probs[:, 1]) * 100
            else:
                metrics["roc_auc"] = roc_auc_score(
                    all_labels,
                    all_probs,
                    labels=list(range(all_probs.shape[1])),
                    multi_class="ovr",
                    average="macro",
                ) * 100
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
        for label in class_names:
            class_metrics = metrics["per_class"][label]
            print(
                f"  {label:<6} - Precision: {class_metrics['precision']:.2f}%, "
                f"Recall: {class_metrics['recall']:.2f}%, "
                f"F1: {class_metrics['f1']:.2f}%, "
                f"Support: {class_metrics['support']}"
            )

        # Classification report
        print("\nDetailed Classification Report:")
        print(classification_report(
            all_labels, all_preds,
            labels=label_indices,
            target_names=class_names,
            digits=4,
            zero_division=0,
        ))

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds, labels=label_indices)
        print("\nConfusion Matrix:")
        name_width = max(8, max(len(name) for name in class_names) + 2)
        cell_width = max(7, name_width)
        print(" " * name_width + "Predicted")
        print("Actual".ljust(name_width) + "".join(name.rjust(cell_width) for name in class_names))
        for idx, label in enumerate(class_names):
            row = "".join(f"{cm[idx, j]:{cell_width}d}" for j in range(len(class_names)))
            print(label.ljust(name_width) + row)

        # Save results
        results = {
            "model_path": str(model_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_accuracy": checkpoint.get("accuracy"),
            "test_samples": len(test_samples),
            "class_names": class_names,
            "class_order_source": class_names_source,
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
            class_names,
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
        Predict the compartment class for a single image.

        Args:
            model_path: Path to trained model
            image_path: Path to compartment image

        Returns:
            Tuple of (label, confidence)
        """
        from PIL import Image
        from torchvision import transforms

        # Load model
        model, checkpoint = load_checkpoint(model_path)
        class_names = _checkpoint_class_names(checkpoint)
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

        label = _label_for_index(class_names, predicted.item())
        return label, confidence.item()

    def predict_batch(
        model_path: Path,
        image_paths: List[Path],
        config: Optional[MLPipelineConfig] = None,
    ) -> List[Tuple[str, float]]:
        """
        Predict compartment classes for multiple images.

        Args:
            model_path: Path to trained model
            image_paths: List of paths to compartment images

        Returns:
            List of (label, confidence) tuples
        """
        from PIL import Image
        from torchvision import transforms

        # Load model
        model, checkpoint = load_checkpoint(model_path)
        class_names = _checkpoint_class_names(checkpoint)
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

                label = _label_for_index(class_names, predicted.item())
                results.append((label, confidence.item()))
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append(("Error", 0.0))

        return results


def main():
    """Main entry point for evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate compartment classifier")
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
        "--approved-compartments",
        type=str,
        help="Path to labeled classifier image folders",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        help="Comma-separated class names in training index order",
    )
    parser.add_argument(
        "--auto-class-folders",
        action="store_true",
        help="Use immediate subfolder names as class names",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Evaluation device: auto, cpu, or cuda",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )

    args = parser.parse_args()
    model_path = Path(args.model_path)

    # Load config
    if args.config:
        config_path = Path(args.config)
        print(f"Using config: {config_path}")
        config = load_config(config_path)
    else:
        from .config import create_config

        config_path = _infer_config_path(model_path)
        if config_path:
            print(f"Using config: {config_path}")
            config = load_config(config_path)
        else:
            config = create_config(shared_folder=args.shared_folder)

    if args.approved_compartments:
        config.approved_compartments_path = Path(args.approved_compartments)
    if args.class_names:
        config.class_names = [name.strip() for name in args.class_names.split(",") if name.strip()]
        config.num_classes = len(config.class_names)
    if args.auto_class_folders:
        config.auto_class_names_from_folders = True
    if args.device:
        config.device = args.device

    output_dir = Path(args.output_dir) if args.output_dir else None

    # Evaluate
    evaluate_model(
        model_path,
        config,
        output_dir,
    )


if __name__ == "__main__":
    main()
