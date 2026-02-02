"""
Data loader for ML Pipeline.

Scans approved compartment images for Wet/Dry labels and creates training datasets.
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch torchvision")

from .config import MLPipelineConfig


@dataclass
class CompartmentSample:
    """A single compartment sample for training."""
    path: Path
    label: str  # "Wet" or "Dry"
    hole_id: str
    depth: int
    label_idx: int  # 0 for Dry, 1 for Wet


class CompartmentDataset:
    """Dataset of compartment images with wet/dry labels."""

    # Label mapping
    LABEL_TO_IDX = {"Dry": 0, "Wet": 1}
    IDX_TO_LABEL = {0: "Dry", 1: "Wet"}

    def __init__(self, config: MLPipelineConfig):
        self.config = config
        self.samples: List[CompartmentSample] = []
        self.stats: Dict[str, int] = {"Wet": 0, "Dry": 0, "unlabeled": 0}

    def scan_approved_compartments(self) -> int:
        """
        Scan approved compartments folder for images with Wet/Dry labels.

        Returns:
            Number of labeled samples found
        """
        if not self.config.approved_compartments_path:
            raise ValueError("approved_compartments_path not configured")

        root = self.config.approved_compartments_path
        if not root.exists():
            raise FileNotFoundError(f"Approved compartments folder not found: {root}")

        # Pattern: {HoleID}_CC_{Depth}_{Wet|Dry}.png
        pattern = re.compile(r"^([A-Za-z0-9]+)_CC_(\d{3})_(Wet|Dry)\.png$", re.IGNORECASE)

        self.samples = []
        self.stats = {"Wet": 0, "Dry": 0, "unlabeled": 0}

        # Walk through project/hole structure
        for project_dir in root.iterdir():
            if not project_dir.is_dir():
                continue

            for hole_dir in project_dir.iterdir():
                if not hole_dir.is_dir():
                    continue

                for file in hole_dir.iterdir():
                    if not file.is_file():
                        continue

                    match = pattern.match(file.name)
                    if match:
                        hole_id = match.group(1)
                        depth = int(match.group(2))
                        label = match.group(3).capitalize()  # Normalize to Wet/Dry

                        sample = CompartmentSample(
                            path=file,
                            label=label,
                            hole_id=hole_id,
                            depth=depth,
                            label_idx=self.LABEL_TO_IDX[label],
                        )
                        self.samples.append(sample)
                        self.stats[label] += 1
                    elif file.suffix.lower() == ".png":
                        self.stats["unlabeled"] += 1

        print(f"Found {len(self.samples)} labeled samples:")
        print(f"  - Wet: {self.stats['Wet']}")
        print(f"  - Dry: {self.stats['Dry']}")
        print(f"  - Unlabeled (skipped): {self.stats['unlabeled']}")

        return len(self.samples)

    def get_class_weights(self) -> Tuple[float, float]:
        """Calculate class weights for imbalanced data."""
        total = self.stats["Wet"] + self.stats["Dry"]
        if total == 0:
            return (1.0, 1.0)

        # Inverse frequency weighting
        wet_weight = total / (2 * max(self.stats["Wet"], 1))
        dry_weight = total / (2 * max(self.stats["Dry"], 1))

        return (dry_weight, wet_weight)

    def split_data(
        self,
        validation_split: float = 0.2,
        test_split: float = 0.1,
        seed: int = 42,
    ) -> Tuple[List[CompartmentSample], List[CompartmentSample], List[CompartmentSample]]:
        """
        Split data into train/validation/test sets.

        Stratifies by label and hole_id to prevent data leakage.
        """
        np.random.seed(seed)

        # Group by hole_id to prevent leakage
        holes_by_label: Dict[str, Dict[str, List[CompartmentSample]]] = {
            "Wet": {},
            "Dry": {},
        }

        for sample in self.samples:
            if sample.hole_id not in holes_by_label[sample.label]:
                holes_by_label[sample.label][sample.hole_id] = []
            holes_by_label[sample.label][sample.hole_id].append(sample)

        train_samples: List[CompartmentSample] = []
        val_samples: List[CompartmentSample] = []
        test_samples: List[CompartmentSample] = []

        for label in ["Wet", "Dry"]:
            holes = list(holes_by_label[label].keys())
            np.random.shuffle(holes)

            n_holes = len(holes)
            n_test = max(1, int(n_holes * test_split))
            n_val = max(1, int(n_holes * validation_split))

            test_holes = holes[:n_test]
            val_holes = holes[n_test : n_test + n_val]
            train_holes = holes[n_test + n_val :]

            for hole in test_holes:
                test_samples.extend(holes_by_label[label][hole])
            for hole in val_holes:
                val_samples.extend(holes_by_label[label][hole])
            for hole in train_holes:
                train_samples.extend(holes_by_label[label][hole])

        print(f"\nData split:")
        print(f"  - Train: {len(train_samples)} samples")
        print(f"  - Validation: {len(val_samples)} samples")
        print(f"  - Test: {len(test_samples)} samples")

        return train_samples, val_samples, test_samples

    def export_dataset_info(self, path: Path) -> None:
        """Export dataset information to JSON for analysis."""
        info = {
            "total_samples": len(self.samples),
            "stats": self.stats,
            "class_weights": self.get_class_weights(),
            "samples": [
                {
                    "path": str(s.path),
                    "label": s.label,
                    "hole_id": s.hole_id,
                    "depth": s.depth,
                }
                for s in self.samples
            ],
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(info, f, indent=2)

        print(f"Dataset info exported to: {path}")


if TORCH_AVAILABLE:

    class WetDryDataset(Dataset):
        """PyTorch Dataset for wet/dry classification."""

        def __init__(
            self,
            samples: List[CompartmentSample],
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

            # Load image
            image = Image.open(sample.path).convert("RGB")

            # Apply transforms
            if self.transform:
                image = self.transform(image)

            return image, sample.label_idx

    def create_data_loaders(
        config: MLPipelineConfig,
        train_samples: List[CompartmentSample],
        val_samples: List[CompartmentSample],
        test_samples: List[CompartmentSample],
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

        # Validation/test transforms (no augmentation)
        eval_transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        train_dataset = WetDryDataset(train_samples, train_transform, config.image_size)
        val_dataset = WetDryDataset(val_samples, eval_transform, config.image_size)
        test_dataset = WetDryDataset(test_samples, eval_transform, config.image_size)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader, test_loader
