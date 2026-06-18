"""
Data loader for ML Pipeline.

Scans approved compartment images for Dry/Wet/Empty labels and creates training datasets.
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter, defaultdict

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
    label: str  # "Dry", "Wet", or "Empty"
    hole_id: str
    depth: float
    label_idx: int


class CompartmentDataset:
    """Dataset of compartment images with classifier labels."""

    # Label mapping
    DEFAULT_CLASS_NAMES = ("Dry", "Wet", "Empty")
    CLASS_NAMES = DEFAULT_CLASS_NAMES
    LABEL_TO_IDX = {label: idx for idx, label in enumerate(CLASS_NAMES)}
    IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}

    def __init__(self, config: MLPipelineConfig):
        self.config = config
        self.CLASS_NAMES = tuple(self._resolve_class_names())
        self.LABEL_TO_IDX = {label: idx for idx, label in enumerate(self.CLASS_NAMES)}
        self.IDX_TO_LABEL = {idx: label for label, idx in self.LABEL_TO_IDX.items()}
        self.samples: List[CompartmentSample] = []
        self.stats: Dict[str, int] = self._empty_stats()

    def _empty_stats(self) -> Dict[str, int]:
        return {**{label: 0 for label in self.CLASS_NAMES}, "unlabeled": 0}

    @classmethod
    def class_names(cls) -> List[str]:
        return list(cls.CLASS_NAMES)

    def get_class_names(self) -> List[str]:
        return list(self.CLASS_NAMES)

    def _resolve_class_names(self) -> List[str]:
        if self.config.class_names:
            return list(self.config.class_names)

        if self.config.auto_class_names_from_folders:
            root = self.config.approved_compartments_path
            if not root or not root.exists():
                raise ValueError(
                    "auto_class_names_from_folders requires an existing "
                    "approved_compartments_path"
                )
            class_names = sorted(
                path.name for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")
            )
            if not class_names:
                raise ValueError(f"No class folders found in {root}")
            return class_names

        return list(self.DEFAULT_CLASS_NAMES)

    def _label_from_path(self, root: Path, path: Path) -> Optional[str]:
        """Return the closest class folder label between root and path, if any."""
        label_lookup = {label.lower(): label for label in self.CLASS_NAMES}
        try:
            rel_parts = path.relative_to(root).parts[:-1]
        except ValueError:
            rel_parts = path.parts[:-1]

        for part in reversed(rel_parts):
            label = label_lookup.get(part.lower())
            if label:
                return label
        return None

    def _filename_label(self, path: Path) -> Optional[str]:
        label_lookup = {label.lower(): label for label in self.CLASS_NAMES}
        suffix = path.stem.rsplit("_", 1)[-1]
        return label_lookup.get(suffix.lower())

    @staticmethod
    def _parse_depth(value: str) -> float:
        return float(value.replace("p", ".").replace("P", "."))

    def scan_approved_compartments(self) -> int:
        """
        Scan approved compartments folder for images with Dry/Wet/Empty labels.

        Returns:
            Number of labeled samples found
        """
        if not self.config.approved_compartments_path:
            raise ValueError("approved_compartments_path not configured")

        root = self.config.approved_compartments_path
        if not root.exists():
            raise FileNotFoundError(f"Approved compartments folder not found: {root}")

        # Pattern: {HoleID}_CC_{Depth}...
        # Folder labels such as classifier_images/Empty/... are treated as truth;
        # filename labels are a fallback for the legacy approved-compartment tree.
        pattern = re.compile(
            r"^([A-Za-z0-9]+)_CC_(\d+(?:[pP]\d+|\.\d+)?)(?:_|\.|$)",
            re.IGNORECASE,
        )

        self.samples = []
        self.stats = self._empty_stats()

        image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
        for file in root.rglob("*"):
            if not file.is_file():
                continue
            if file.suffix.lower() not in image_extensions:
                continue

            match = pattern.match(file.name)
            folder_label = self._label_from_path(root, file)

            if not match:
                self.stats["unlabeled"] += 1
                continue

            filename_label = self._filename_label(file)
            label = folder_label or filename_label
            if label not in self.LABEL_TO_IDX:
                self.stats["unlabeled"] += 1
                continue

            hole_id = match.group(1)
            depth = self._parse_depth(match.group(2))

            sample = CompartmentSample(
                path=file,
                label=label,
                hole_id=hole_id,
                depth=depth,
                label_idx=self.LABEL_TO_IDX[label],
            )
            self.samples.append(sample)
            self.stats[label] += 1

        print(f"Found {len(self.samples)} labeled samples:")
        for label in self.CLASS_NAMES:
            print(f"  - {label}: {self.stats[label]}")
        print(f"  - Unlabeled (skipped): {self.stats['unlabeled']}")

        return len(self.samples)

    def missing_classes(self) -> List[str]:
        """Return labels that do not currently have any samples."""
        return [label for label in self.CLASS_NAMES if self.stats[label] == 0]

    def get_class_weights(self) -> Tuple[float, ...]:
        """Calculate class weights for imbalanced data."""
        counts = [self.stats[label] for label in self.CLASS_NAMES]
        total = sum(counts)
        if total == 0:
            return tuple(1.0 for _ in self.CLASS_NAMES)

        # Inverse frequency weighting
        n_classes = len(self.CLASS_NAMES)
        return tuple(total / (n_classes * max(count, 1)) for count in counts)

    def _split_data_by_label_holes(
        self,
        validation_split: float,
        test_split: float,
        seed: int,
    ) -> Tuple[List[CompartmentSample], List[CompartmentSample], List[CompartmentSample]]:
        """Legacy split: holes are grouped inside each label only."""
        np.random.seed(seed)

        holes_by_label: Dict[str, Dict[str, List[CompartmentSample]]] = {
            label: {} for label in self.CLASS_NAMES
        }

        for sample in self.samples:
            if sample.hole_id not in holes_by_label[sample.label]:
                holes_by_label[sample.label][sample.hole_id] = []
            holes_by_label[sample.label][sample.hole_id].append(sample)

        train_samples: List[CompartmentSample] = []
        val_samples: List[CompartmentSample] = []
        test_samples: List[CompartmentSample] = []

        for label in self.CLASS_NAMES:
            holes = list(holes_by_label[label].keys())
            np.random.shuffle(holes)

            if not holes:
                continue

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

        return train_samples, val_samples, test_samples

    def _split_data_by_global_holes(
        self,
        validation_split: float,
        test_split: float,
        seed: int,
    ) -> Tuple[List[CompartmentSample], List[CompartmentSample], List[CompartmentSample]]:
        """Assign each hole to exactly one split to avoid leakage."""
        rng = np.random.default_rng(seed)
        holes: Dict[str, List[CompartmentSample]] = defaultdict(list)
        for sample in self.samples:
            holes[sample.hole_id].append(sample)

        hole_items = list(holes.items())
        rng.shuffle(hole_items)
        hole_items.sort(key=lambda item: len(item[1]), reverse=True)

        split_targets = {
            "train": 1.0 - validation_split - test_split,
            "val": validation_split,
            "test": test_split,
        }
        split_samples: Dict[str, List[CompartmentSample]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        split_label_counts = {
            split: Counter({label: 0 for label in self.CLASS_NAMES})
            for split in split_samples
        }
        total_label_counts = Counter(sample.label for sample in self.samples)
        total_samples = max(1, len(self.samples))

        for _, samples in hole_items:
            hole_counts = Counter(sample.label for sample in samples)
            hole_size = len(samples)

            best_split = "train"
            best_score = float("inf")
            for candidate_split in split_targets:
                score = 0.0
                for split, ratio in split_targets.items():
                    target_total = max(1.0, total_samples * ratio)
                    new_total = len(split_samples[split])
                    if split == candidate_split:
                        new_total += hole_size
                    score += ((new_total - target_total) / target_total) ** 2

                    for label in self.CLASS_NAMES:
                        label_target = max(1.0, total_label_counts[label] * ratio)
                        new_label_total = split_label_counts[split][label]
                        if split == candidate_split:
                            new_label_total += hole_counts[label]
                        score += 2.0 * ((new_label_total - label_target) / label_target) ** 2

                if score < best_score:
                    best_score = score
                    best_split = candidate_split

            split_samples[best_split].extend(samples)
            split_label_counts[best_split].update(hole_counts)

        return split_samples["train"], split_samples["val"], split_samples["test"]

    def _print_split_summary(
        self,
        train_samples: List[CompartmentSample],
        val_samples: List[CompartmentSample],
        test_samples: List[CompartmentSample],
    ) -> None:
        def summarize(samples: List[CompartmentSample]) -> Tuple[int, int, Dict[str, int]]:
            label_counts = Counter(sample.label for sample in samples)
            return (
                len(samples),
                len({sample.hole_id for sample in samples}),
                {label: label_counts.get(label, 0) for label in self.CLASS_NAMES},
            )

        print(f"\nData split:")
        for name, samples in [("Train", train_samples), ("Validation", val_samples), ("Test", test_samples)]:
            count, holes, label_counts = summarize(samples)
            print(f"  - {name}: {count} samples across {holes} holes")
            print(f"    labels: {label_counts}")

        split_holes = {
            "train": {sample.hole_id for sample in train_samples},
            "val": {sample.hole_id for sample in val_samples},
            "test": {sample.hole_id for sample in test_samples},
        }
        leakage = (
            split_holes["train"] & split_holes["val"]
            or split_holes["train"] & split_holes["test"]
            or split_holes["val"] & split_holes["test"]
        )
        if leakage:
            print("  ! Warning: split contains overlapping holes")
        else:
            print("  - Hole leakage check: passed")

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
        if getattr(self.config, "split_by_hole", True):
            train_samples, val_samples, test_samples = self._split_data_by_global_holes(
                validation_split, test_split, seed
            )
        else:
            train_samples, val_samples, test_samples = self._split_data_by_label_holes(
                validation_split, test_split, seed
            )

        self._print_split_summary(train_samples, val_samples, test_samples)

        return train_samples, val_samples, test_samples

    def export_dataset_info(self, path: Path) -> None:
        """Export dataset information to JSON for analysis."""
        info = {
            "total_samples": len(self.samples),
            "class_names": list(self.CLASS_NAMES),
            "stats": self.stats,
            "class_weights": self.get_class_weights(),
            "per_class_hole_counts": {
                label: len({s.hole_id for s in self.samples if s.label == label})
                for label in self.CLASS_NAMES
            },
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
        """PyTorch Dataset for compartment classification."""

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
