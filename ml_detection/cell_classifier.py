"""
4-Class Cell Classifier for GeoVue Chip Trays

Classes:
- Dry: Normal dry chip samples
- Wet: Wet chip samples
- Empty: Empty compartments (no chips)
- Bad: Poor quality images (marked with _Bad in Photo_Status)

This classifier is designed to work with the YOLO cell detector for
real-time classification on Raspberry Pi 5.
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import random
from collections import Counter

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ClassifierConfig:
    """Configuration for the 4-class cell classifier."""
    # Paths
    approved_images_path: Path = None
    additional_image_paths: List[Path] = None  # Extra folders (e.g., empty images)
    register_data_path: Path = None
    output_path: Path = None

    # Model settings
    model_name: str = "mobilenet_v3_small"  # Optimized for Pi 5
    image_size: int = 224
    num_classes: int = 4
    dropout_rate: float = 0.5
    pretrained: bool = True

    # Training settings
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 50
    early_stopping_patience: int = 10

    # Data splits
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1

    # Device
    device: str = "auto"


# ============================================================================
# Label Definitions
# ============================================================================

class CellLabels:
    """Cell classification labels."""
    DRY = "Dry"
    WET = "Wet"
    EMPTY = "Empty"
    BAD = "Bad"

    LABEL_TO_IDX = {DRY: 0, WET: 1, EMPTY: 2, BAD: 3}
    IDX_TO_LABEL = {0: DRY, 1: WET, 2: EMPTY, 3: BAD}

    # Colors for visualization (BGR for OpenCV)
    COLORS = {
        DRY: (0, 165, 255),    # Orange
        WET: (255, 0, 0),      # Blue
        EMPTY: (128, 128, 128), # Gray
        BAD: (0, 0, 255),      # Red
    }

    @classmethod
    def get_color(cls, label: str) -> Tuple[int, int, int]:
        return cls.COLORS.get(label, (255, 255, 255))


# ============================================================================
# Data Loading
# ============================================================================

@dataclass
class CellSample:
    """A single cell sample for training/inference."""
    path: Path
    label: str
    label_idx: int
    hole_id: str
    depth_code: str
    source: str  # "filename" or "register"


class CellDataset:
    """Dataset manager for 4-class cell classification."""

    def __init__(self, config: ClassifierConfig):
        self.config = config
        self.samples: List[CellSample] = []
        self.stats: Dict[str, int] = {
            CellLabels.DRY: 0,
            CellLabels.WET: 0,
            CellLabels.EMPTY: 0,
            CellLabels.BAD: 0,
            "unlabeled": 0,
        }
        self.bad_images: set = set()

    def load_bad_status_from_registers(self) -> int:
        """
        Load images marked as Bad from compartment registers.
        Returns count of bad images found.
        """
        if not self.config.register_data_path:
            return 0

        register_dir = self.config.register_data_path
        if not register_dir.exists():
            print(f"Warning: Register directory not found: {register_dir}")
            return 0

        bad_count = 0

        # Scan compartment_register_*.json files for Photo_Status containing "_Bad"
        for json_file in register_dir.glob("compartment_register_*.json"):
            try:
                with open(json_file, 'r') as f:
                    records = json.load(f)

                for record in records:
                    status = record.get("Photo_Status", "")
                    if "_Bad" in status:
                        hole_id = record.get("HoleID", "")
                        depth_from = record.get("From", 0)
                        depth_to = record.get("To", 0)

                        # Build potential image identifiers
                        # Format: HoleID_CC_DepthCode
                        depth_code = f"{depth_from:03d}"
                        image_key = f"{hole_id}_CC_{depth_code}"
                        self.bad_images.add(image_key.upper())
                        bad_count += 1

            except Exception as e:
                print(f"Warning: Error reading {json_file}: {e}")

        print(f"Found {bad_count} images marked as Bad in registers")
        return bad_count

    def scan_images(self) -> int:
        """
        Scan approved compartment images and assign labels.

        Labeling logic:
        1. If filename ends with _Empty.png -> Empty
        2. If filename ends with _Bad.png -> Bad
        3. If image key in bad_images set (from register) -> Bad
        4. If filename ends with _Wet.png -> Wet
        5. If filename ends with _Dry.png -> Dry
        6. Otherwise -> unlabeled (skipped)
        """
        self.samples = []
        self.stats = {k: 0 for k in self.stats}

        # Collect all image directories to scan
        scan_dirs = []
        if self.config.approved_images_path and self.config.approved_images_path.exists():
            scan_dirs.append(self.config.approved_images_path)
        if self.config.additional_image_paths:
            for path in self.config.additional_image_paths:
                if path and path.exists():
                    scan_dirs.append(path)

        if not scan_dirs:
            raise ValueError("No valid image paths configured")

        # Pattern: {HoleID}_CC_{DepthCode}[_temp]_{Label}.png
        # Handles both standard and _temp_ variants
        pattern = re.compile(
            r"^([A-Za-z0-9]+)_CC_(\d{3})(?:_temp)?_(Wet|Dry|Empty|Bad)\.png$",
            re.IGNORECASE
        )

        # Scan all directories
        for scan_dir in scan_dirs:
            print(f"Scanning: {scan_dir}")
            for img_file in scan_dir.rglob("*.png"):
                match = pattern.match(img_file.name)

                if match:
                    hole_id = match.group(1).upper()
                    depth_code = match.group(2)
                    filename_label = match.group(3).capitalize()

                    # Check if this image is marked Bad in register
                    image_key = f"{hole_id}_CC_{depth_code}"

                    # Determine final label
                    if filename_label == "Bad":
                        label = CellLabels.BAD
                        source = "filename"
                    elif filename_label == "Empty":
                        label = CellLabels.EMPTY
                        source = "filename"
                    elif image_key in self.bad_images:
                        label = CellLabels.BAD
                        source = "register"
                    elif filename_label == "Wet":
                        label = CellLabels.WET
                        source = "filename"
                    elif filename_label == "Dry":
                        label = CellLabels.DRY
                        source = "filename"
                    else:
                        self.stats["unlabeled"] += 1
                        continue

                sample = CellSample(
                    path=img_file,
                    label=label,
                    label_idx=CellLabels.LABEL_TO_IDX[label],
                    hole_id=hole_id,
                    depth_code=depth_code,
                    source=source,
                )
                self.samples.append(sample)
                self.stats[label] += 1
            else:
                self.stats["unlabeled"] += 1

        print(f"\nFound {len(self.samples)} labeled samples:")
        for label in [CellLabels.DRY, CellLabels.WET, CellLabels.EMPTY, CellLabels.BAD]:
            print(f"  - {label}: {self.stats[label]}")
        print(f"  - Unlabeled (skipped): {self.stats['unlabeled']}")

        return len(self.samples)

    def get_class_weights(self) -> List[float]:
        """Calculate class weights for imbalanced data."""
        counts = [
            max(self.stats[CellLabels.DRY], 1),
            max(self.stats[CellLabels.WET], 1),
            max(self.stats[CellLabels.EMPTY], 1),
            max(self.stats[CellLabels.BAD], 1),
        ]
        total = sum(counts)
        n_classes = len(counts)

        # Inverse frequency weighting
        weights = [total / (n_classes * c) for c in counts]
        return weights

    def split_data(
        self,
        seed: int = 42,
    ) -> Tuple[List[CellSample], List[CellSample], List[CellSample]]:
        """
        Split data into train/val/test sets.
        Stratifies by label and hole_id to prevent data leakage.
        """
        random.seed(seed)
        np.random.seed(seed)

        # Group by hole_id per label
        holes_by_label: Dict[str, Dict[str, List[CellSample]]] = {
            label: {} for label in CellLabels.LABEL_TO_IDX.keys()
        }

        for sample in self.samples:
            if sample.hole_id not in holes_by_label[sample.label]:
                holes_by_label[sample.label][sample.hole_id] = []
            holes_by_label[sample.label][sample.hole_id].append(sample)

        train_samples = []
        val_samples = []
        test_samples = []

        for label in CellLabels.LABEL_TO_IDX.keys():
            holes = list(holes_by_label[label].keys())
            random.shuffle(holes)

            n_holes = len(holes)
            if n_holes == 0:
                continue

            n_test = max(1, int(n_holes * self.config.test_ratio))
            n_val = max(1, int(n_holes * self.config.val_ratio))

            test_holes = holes[:n_test]
            val_holes = holes[n_test:n_test + n_val]
            train_holes = holes[n_test + n_val:]

            for hole in test_holes:
                test_samples.extend(holes_by_label[label][hole])
            for hole in val_holes:
                val_samples.extend(holes_by_label[label][hole])
            for hole in train_holes:
                train_samples.extend(holes_by_label[label][hole])

        # Shuffle each split
        random.shuffle(train_samples)
        random.shuffle(val_samples)
        random.shuffle(test_samples)

        print(f"\nData split:")
        print(f"  - Train: {len(train_samples)} samples")
        print(f"  - Validation: {len(val_samples)} samples")
        print(f"  - Test: {len(test_samples)} samples")

        return train_samples, val_samples, test_samples


# ============================================================================
# PyTorch Components
# ============================================================================

if TORCH_AVAILABLE:

    class CellClassifierDataset(Dataset):
        """PyTorch Dataset for cell classification."""

        def __init__(
            self,
            samples: List[CellSample],
            transform=None,
            image_size: int = 224,
        ):
            self.samples = samples
            self.transform = transform
            self.image_size = image_size

            if self.transform is None:
                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                ])

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
            sample = self.samples[idx]
            image = Image.open(sample.path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            return image, sample.label_idx


    class CellClassifierModel(nn.Module):
        """
        4-Class Cell Classifier using transfer learning.

        Optimized for Raspberry Pi 5 deployment with MobileNetV3.
        """

        def __init__(
            self,
            model_name: str = "mobilenet_v3_small",
            num_classes: int = 4,
            pretrained: bool = True,
            dropout_rate: float = 0.5,
        ):
            super().__init__()

            self.model_name = model_name
            self.num_classes = num_classes

            # Load backbone
            if model_name == "mobilenet_v3_small":
                weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
                backbone = models.mobilenet_v3_small(weights=weights)
                num_features = backbone.classifier[0].in_features
                backbone.classifier = nn.Identity()

            elif model_name == "mobilenet_v3_large":
                weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
                backbone = models.mobilenet_v3_large(weights=weights)
                num_features = backbone.classifier[0].in_features
                backbone.classifier = nn.Identity()

            elif model_name == "efficientnet_b0":
                weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
                backbone = models.efficientnet_b0(weights=weights)
                num_features = backbone.classifier[1].in_features
                backbone.classifier = nn.Identity()

            elif model_name == "resnet18":
                weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                backbone = models.resnet18(weights=weights)
                num_features = backbone.fc.in_features
                backbone.fc = nn.Identity()

            else:
                raise ValueError(f"Unknown model: {model_name}")

            self.backbone = backbone
            self.num_features = num_features

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

        def predict_with_confidence(
            self,
            x: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Predict class and confidence.
            Returns (predicted_class, confidence)
            """
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            confidence, predicted = torch.max(probs, dim=1)
            return predicted, confidence


    def create_data_loaders(
        config: ClassifierConfig,
        train_samples: List[CellSample],
        val_samples: List[CellSample],
        test_samples: List[CellSample],
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create PyTorch data loaders with augmentation."""

        # Training transforms with augmentation
        train_transform = transforms.Compose([
            transforms.Resize((config.image_size + 32, config.image_size + 32)),
            transforms.RandomCrop(config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # Validation/test transforms
        eval_transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        train_dataset = CellClassifierDataset(train_samples, train_transform, config.image_size)
        val_dataset = CellClassifierDataset(val_samples, eval_transform, config.image_size)
        test_dataset = CellClassifierDataset(test_samples, eval_transform, config.image_size)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        return train_loader, val_loader, test_loader


# ============================================================================
# Inference Helper
# ============================================================================

class CellClassifierInference:
    """
    Inference wrapper for cell classifier.
    Designed for integration with YOLO detector on Pi 5.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        image_size: int = 224,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")

        # Determine device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)

        model_name = checkpoint.get("model_name", "mobilenet_v3_small")
        num_classes = checkpoint.get("num_classes", 4)

        self.model = CellClassifierModel(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Transform for inference
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        print(f"Classifier loaded: {model_name}, {num_classes} classes, device={self.device}")

    def classify_crop(
        self,
        image: np.ndarray,  # BGR image from OpenCV
    ) -> Tuple[str, float]:
        """
        Classify a cropped cell image.

        Args:
            image: BGR numpy array (OpenCV format)

        Returns:
            (label, confidence) tuple
        """
        # Convert BGR to RGB PIL Image
        rgb = image[:, :, ::-1]
        pil_image = Image.fromarray(rgb)

        # Transform
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            predicted, confidence = self.model.predict_with_confidence(tensor)

        label = CellLabels.IDX_TO_LABEL[predicted.item()]
        conf = confidence.item()

        return label, conf

    def classify_batch(
        self,
        images: List[np.ndarray],
    ) -> List[Tuple[str, float]]:
        """
        Classify multiple cropped cell images in a batch.

        Args:
            images: List of BGR numpy arrays

        Returns:
            List of (label, confidence) tuples
        """
        if not images:
            return []

        # Convert and transform all images
        tensors = []
        for img in images:
            rgb = img[:, :, ::-1]
            pil_image = Image.fromarray(rgb)
            tensor = self.transform(pil_image)
            tensors.append(tensor)

        batch = torch.stack(tensors).to(self.device)

        # Predict
        with torch.no_grad():
            predicted, confidence = self.model.predict_with_confidence(batch)

        results = []
        for i in range(len(images)):
            label = CellLabels.IDX_TO_LABEL[predicted[i].item()]
            conf = confidence[i].item()
            results.append((label, conf))

        return results
