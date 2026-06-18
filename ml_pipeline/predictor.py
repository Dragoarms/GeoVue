"""
ML Predictor Service for QAQC Integration.

Provides a singleton predictor that can be used throughout the application
to classify compartment images.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging

# Check for PyTorch availability
try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class WetDryPredictor:
    """
    Singleton predictor for compartment classification.

    Loads the model once and provides fast predictions for compartment images.
    """

    _instance = None
    _model = None
    _device = None
    _transform = None
    _model_path = None
    _is_loaded = False
    _class_names = None

    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.90
    MEDIUM_CONFIDENCE_THRESHOLD = 0.70

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._model = None
            self._device = None
            self._transform = None
            self._is_loaded = False
            self._class_names = ["Dry", "Wet"]

    @property
    def is_available(self) -> bool:
        """Check if ML prediction is available."""
        return TORCH_AVAILABLE and self._is_loaded

    def load_model(self, model_path: Optional[Path] = None) -> bool:
        """
        Load the compartment classifier model.

        Args:
            model_path: Path to model checkpoint. If None, tries default locations.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available - ML predictions disabled")
            return False

        # Try to find model path
        if model_path is None:
            # Try common locations
            search_paths = [
                Path("ml_output/checkpoints/best_model.pt"),
                Path(__file__).parent.parent / "ml_output" / "checkpoints" / "best_model.pt",
            ]

            for path in search_paths:
                if path.exists():
                    model_path = path
                    break

        if model_path is None or not Path(model_path).exists():
            logger.warning(f"Model not found at {model_path}")
            return False

        try:
            from .model import load_checkpoint

            logger.info(f"Loading compartment classifier from: {model_path}")

            self._model, checkpoint = load_checkpoint(model_path)
            self._class_names = list(checkpoint.get("class_names") or ["Dry", "Wet"])
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = self._model.to(self._device)
            self._model.eval()

            # Setup transform
            image_size = 224  # Default
            self._transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

            self._model_path = model_path
            self._is_loaded = True

            logger.info(f"Model loaded successfully (device: {self._device})")
            logger.info(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
            logger.info(f"  Checkpoint accuracy: {checkpoint.get('accuracy', 0):.2f}%")
            logger.info(f"  Classes: {', '.join(self._class_names)}")

            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._is_loaded = False
            return False

    def _label_for_index(self, label_idx: int) -> str:
        """Map a model output index to a label."""
        if self._class_names and 0 <= label_idx < len(self._class_names):
            return self._class_names[label_idx]
        return f"Class_{label_idx}"

    def predict_single(self, image_path: str) -> Tuple[str, float]:
        """
        Predict the compartment class for a single image.

        Args:
            image_path: Path to compartment image

        Returns:
            Tuple of (label, confidence)
        """
        if not self.is_available:
            return ("unknown", 0.0)

        try:
            image = Image.open(image_path).convert("RGB")
            input_tensor = self._transform(image).unsqueeze(0).to(self._device)

            with torch.no_grad():
                output = self._model(input_tensor)
                probs = F.softmax(output, dim=1)
                confidence, predicted = probs.max(1)

            label = self._label_for_index(predicted.item())
            return (label, confidence.item())

        except Exception as e:
            logger.error(f"Prediction error for {image_path}: {e}")
            return ("unknown", 0.0)

    def predict_batch(
        self,
        image_paths: List[str],
        batch_size: int = 32,
    ) -> List[Tuple[str, float]]:
        """
        Predict compartment classes for multiple images efficiently using batched inference.

        Stacks images into tensors and runs a single forward pass per batch,
        instead of one model call per image.

        Args:
            image_paths: List of paths to compartment images
            batch_size: Number of images per forward pass

        Returns:
            List of (label, confidence) tuples, in the same order as image_paths
        """
        if not self.is_available:
            return [("unknown", 0.0) for _ in image_paths]

        # Pre-load and transform all images, tracking which indices failed
        tensors = []
        index_map = []  # maps tensor position -> original index
        results: List[Tuple[str, float]] = [("unknown", 0.0)] * len(image_paths)

        for i, image_path in enumerate(image_paths):
            try:
                image = Image.open(image_path).convert("RGB")
                tensor = self._transform(image)
                tensors.append(tensor)
                index_map.append(i)
            except Exception as e:
                logger.error(f"Failed to load image {image_path}: {e}")
                results[i] = ("unknown", 0.0)

        if not tensors:
            return results

        # Run batched inference
        with torch.no_grad():
            for batch_start in range(0, len(tensors), batch_size):
                batch_tensors = tensors[batch_start : batch_start + batch_size]
                batch_indices = index_map[batch_start : batch_start + batch_size]

                batch = torch.stack(batch_tensors).to(self._device)
                outputs = self._model(batch)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidences, predicted = probs.max(1)

                for j, orig_idx in enumerate(batch_indices):
                    label = self._label_for_index(predicted[j].item())
                    results[orig_idx] = (label, confidences[j].item())

        return results

    def predict_review_items(self, items: List) -> Dict[str, Tuple[str, float]]:
        """
        Predict compartment classes for a list of ReviewItems.

        Only predicts for items that are unclassified (moisture == "unknown").

        Args:
            items: List of ReviewItem objects

        Returns:
            Dictionary mapping image_path to (label, confidence)
        """
        if not self.is_available:
            return {}

        # Filter to only unclassified items
        unclassified = [
            item for item in items
            if getattr(item, 'moisture', 'unknown') == 'unknown'
            and hasattr(item, 'image_path')
            and item.image_path
            and os.path.exists(item.image_path)
        ]

        if not unclassified:
            return {}

        logger.info(f"Running ML predictions on {len(unclassified)} unclassified items...")

        predictions = {}
        for item in unclassified:
            label, confidence = self.predict_single(item.image_path)
            predictions[item.image_path] = (label, confidence)

        # Log summary
        label_counts = {}
        for label, _ in predictions.values():
            label_counts[label] = label_counts.get(label, 0) + 1
        high_conf = sum(1 for _, (l, c) in predictions.items() if c >= self.HIGH_CONFIDENCE_THRESHOLD)
        label_summary = ", ".join(
            f"{count} {label}" for label, count in sorted(label_counts.items())
        )

        logger.info(f"ML Predictions: {label_summary} ({high_conf} high confidence)")

        return predictions

    def get_confidence_level(self, confidence: float) -> str:
        """Get confidence level string based on threshold."""
        if confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
            return "high"
        elif confidence >= self.MEDIUM_CONFIDENCE_THRESHOLD:
            return "medium"
        else:
            return "low"

    def unload_model(self):
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._is_loaded = False
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded")


# Global singleton instance
_predictor = None


def get_predictor() -> WetDryPredictor:
    """Get the global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = WetDryPredictor()
    return _predictor


def ensure_model_loaded(model_path: Optional[Path] = None) -> bool:
    """
    Ensure the model is loaded, loading it if necessary.

    Returns:
        True if model is available, False otherwise.
    """
    predictor = get_predictor()
    if predictor.is_available:
        return True
    return predictor.load_model(model_path)
