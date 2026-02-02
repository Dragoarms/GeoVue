"""
Model definitions for ML Pipeline.

Uses transfer learning with pretrained CNNs for wet/dry classification.
"""

from typing import Optional, Tuple
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from torchvision import models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch torchvision")

from .config import MLPipelineConfig


if TORCH_AVAILABLE:

    class WetDryClassifier(nn.Module):
        """
        Wet/Dry classifier using transfer learning.

        Supports multiple backbone architectures with pretrained ImageNet weights.
        """

        def __init__(
            self,
            model_name: str = "resnet18",
            pretrained: bool = True,
            freeze_backbone: bool = False,
            dropout_rate: float = 0.5,
            num_classes: int = 2,
        ):
            super().__init__()

            self.model_name = model_name
            self.num_classes = num_classes

            # Load backbone
            if model_name == "resnet18":
                weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                backbone = models.resnet18(weights=weights)
                num_features = backbone.fc.in_features
                backbone.fc = nn.Identity()

            elif model_name == "resnet34":
                weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
                backbone = models.resnet34(weights=weights)
                num_features = backbone.fc.in_features
                backbone.fc = nn.Identity()

            elif model_name == "resnet50":
                weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
                backbone = models.resnet50(weights=weights)
                num_features = backbone.fc.in_features
                backbone.fc = nn.Identity()

            elif model_name == "efficientnet_b0":
                weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
                backbone = models.efficientnet_b0(weights=weights)
                num_features = backbone.classifier[1].in_features
                backbone.classifier = nn.Identity()

            elif model_name == "mobilenet_v3_small":
                weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
                backbone = models.mobilenet_v3_small(weights=weights)
                num_features = backbone.classifier[0].in_features
                backbone.classifier = nn.Identity()

            else:
                raise ValueError(f"Unknown model: {model_name}")

            self.backbone = backbone
            self.num_features = num_features

            # Freeze backbone if requested
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False

            # Classification head
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate / 2),
                nn.Linear(256, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.backbone(x)
            return self.classifier(features)

        def get_features(self, x: torch.Tensor) -> torch.Tensor:
            """Extract features without classification."""
            return self.backbone(x)

        def unfreeze_backbone(self, layers: int = -1) -> None:
            """
            Unfreeze backbone layers for fine-tuning.

            Args:
                layers: Number of layers to unfreeze from end (-1 for all)
            """
            params = list(self.backbone.parameters())
            if layers == -1:
                for param in params:
                    param.requires_grad = True
            else:
                for param in params[-layers:]:
                    param.requires_grad = True

    def create_model(config: MLPipelineConfig) -> Tuple[WetDryClassifier, torch.device]:
        """Create model and determine device."""

        # Determine device
        if config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(config.device)

        print(f"Using device: {device}")

        # Create model
        model = WetDryClassifier(
            model_name=config.model_name,
            pretrained=config.pretrained,
            freeze_backbone=config.freeze_backbone,
            dropout_rate=config.dropout_rate,
        )

        model = model.to(device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Model: {config.model_name}")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")

        return model, device

    def save_checkpoint(
        model: WetDryClassifier,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        accuracy: float,
        path: Path,
    ) -> None:
        """Save model checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "accuracy": accuracy,
            "model_name": model.model_name,
            "num_classes": model.num_classes,
        }

        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(
        path: Path,
        config: Optional[MLPipelineConfig] = None,
    ) -> Tuple[WetDryClassifier, dict]:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")

        model_name = checkpoint.get("model_name", "resnet18")
        num_classes = checkpoint.get("num_classes", 2)

        model = WetDryClassifier(
            model_name=model_name,
            pretrained=False,
            num_classes=num_classes,
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        return model, checkpoint
