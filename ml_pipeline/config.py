"""
Configuration management for ML Pipeline.

Reads GeoVue config to locate data paths, or accepts manual path configuration.
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import tomli


@dataclass
class MLPipelineConfig:
    """Configuration for the ML pipeline."""

    # Data paths
    shared_folder_path: Optional[Path] = None
    approved_compartments_path: Optional[Path] = None
    register_data_path: Optional[Path] = None
    datasets_path: Optional[Path] = None

    # Training parameters
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 50
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    test_split: float = 0.1
    split_by_hole: bool = True

    # Model parameters
    model_name: str = "resnet18"  # resnet18, resnet34, efficientnet_b0
    num_classes: int = 3
    class_names: Optional[List[str]] = None
    auto_class_names_from_folders: bool = False
    pretrained: bool = True
    freeze_backbone: bool = False
    dropout_rate: float = 0.5

    # Image parameters
    image_size: int = 224
    num_workers: int = 4

    # Output paths
    output_dir: Path = field(default_factory=lambda: Path("ml_output"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("ml_output/checkpoints"))
    logs_dir: Path = field(default_factory=lambda: Path("ml_output/logs"))

    # Device
    device: str = "auto"  # auto, cuda, cpu

    def __post_init__(self):
        """Ensure paths are Path objects."""
        if self.shared_folder_path:
            self.shared_folder_path = Path(self.shared_folder_path)
        if self.approved_compartments_path:
            self.approved_compartments_path = Path(self.approved_compartments_path)
        if self.register_data_path:
            self.register_data_path = Path(self.register_data_path)
        if self.datasets_path:
            self.datasets_path = Path(self.datasets_path)

        self.output_dir = Path(self.output_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.logs_dir = Path(self.logs_dir)
        if self.class_names:
            self.class_names = [str(name) for name in self.class_names]
            self.num_classes = len(self.class_names)


def find_geovue_config() -> Optional[Path]:
    """Find GeoVue's config file."""
    possible_paths = [
        Path.home() / "AppData" / "Local" / "GeoVue" / "config.toml",
        Path.home() / ".geovue" / "config.toml",
        Path(__file__).parent.parent / "config.toml",
    ]

    for path in possible_paths:
        if path.exists():
            return path

    return None


def load_geovue_paths() -> Dict[str, Optional[Path]]:
    """Load data paths from GeoVue's config."""
    paths = {
        "shared_folder_path": None,
        "approved_compartments_path": None,
        "register_data_path": None,
        "datasets_path": None,
    }

    config_path = find_geovue_config()
    if not config_path:
        return paths

    try:
        with open(config_path, "rb") as f:
            config = tomli.load(f)

        # Get shared folder path
        shared_path = config.get("shared_folder_path")
        if shared_path:
            paths["shared_folder_path"] = Path(shared_path)

            # Build derived paths
            base = Path(shared_path)

            # Look for approved compartments
            compartments_path = base / "Extracted Compartment Images" / "Approved Compartment Images"
            if compartments_path.exists():
                paths["approved_compartments_path"] = compartments_path

            # Look for register data
            register_path = base / "Chip Tray Register" / "Register Data (Do not edit)"
            if register_path.exists():
                paths["register_data_path"] = register_path

        # Get datasets path
        datasets_path = config.get("shared_folder_datasets")
        if datasets_path:
            paths["datasets_path"] = Path(datasets_path)

    except Exception as e:
        print(f"Warning: Could not load GeoVue config: {e}")

    return paths


def create_config(
    shared_folder: Optional[str] = None,
    approved_compartments: Optional[str] = None,
    register_data: Optional[str] = None,
    **kwargs
) -> MLPipelineConfig:
    """
    Create ML pipeline configuration.

    Args:
        shared_folder: Path to GeoVue shared folder (auto-derives other paths)
        approved_compartments: Direct path to approved compartments folder
        register_data: Direct path to register data folder
        **kwargs: Additional config parameters

    Returns:
        MLPipelineConfig instance
    """
    # Start with GeoVue config if available
    geovue_paths = load_geovue_paths()

    # Override with explicit paths
    if shared_folder:
        base = Path(shared_folder)
        geovue_paths["shared_folder_path"] = base
        geovue_paths["approved_compartments_path"] = (
            base / "Extracted Compartment Images" / "Approved Compartment Images"
        )
        geovue_paths["register_data_path"] = (
            base / "Chip Tray Register" / "Register Data (Do not edit)"
        )

    if approved_compartments:
        geovue_paths["approved_compartments_path"] = Path(approved_compartments)

    if register_data:
        geovue_paths["register_data_path"] = Path(register_data)

    # Create config
    config = MLPipelineConfig(
        shared_folder_path=geovue_paths.get("shared_folder_path"),
        approved_compartments_path=geovue_paths.get("approved_compartments_path"),
        register_data_path=geovue_paths.get("register_data_path"),
        datasets_path=geovue_paths.get("datasets_path"),
        **kwargs
    )

    return config


def save_config(config: MLPipelineConfig, path: Path) -> None:
    """Save configuration to JSON file."""
    data = {
        "shared_folder_path": str(config.shared_folder_path) if config.shared_folder_path else None,
        "approved_compartments_path": str(config.approved_compartments_path) if config.approved_compartments_path else None,
        "register_data_path": str(config.register_data_path) if config.register_data_path else None,
        "datasets_path": str(config.datasets_path) if config.datasets_path else None,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "num_epochs": config.num_epochs,
        "early_stopping_patience": config.early_stopping_patience,
        "validation_split": config.validation_split,
        "test_split": config.test_split,
        "split_by_hole": config.split_by_hole,
        "model_name": config.model_name,
        "num_classes": config.num_classes,
        "class_names": config.class_names,
        "auto_class_names_from_folders": config.auto_class_names_from_folders,
        "pretrained": config.pretrained,
        "freeze_backbone": config.freeze_backbone,
        "dropout_rate": config.dropout_rate,
        "image_size": config.image_size,
        "num_workers": config.num_workers,
        "output_dir": str(config.output_dir),
        "checkpoint_dir": str(config.checkpoint_dir),
        "logs_dir": str(config.logs_dir),
        "device": config.device,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_config(path: Path) -> MLPipelineConfig:
    """Load configuration from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)

    return MLPipelineConfig(
        shared_folder_path=data.get("shared_folder_path"),
        approved_compartments_path=data.get("approved_compartments_path"),
        register_data_path=data.get("register_data_path"),
        datasets_path=data.get("datasets_path"),
        batch_size=data.get("batch_size", 32),
        learning_rate=data.get("learning_rate", 0.001),
        num_epochs=data.get("num_epochs", 50),
        early_stopping_patience=data.get("early_stopping_patience", 10),
        validation_split=data.get("validation_split", 0.2),
        test_split=data.get("test_split", 0.1),
        split_by_hole=data.get("split_by_hole", True),
        model_name=data.get("model_name", "resnet18"),
        num_classes=data.get("num_classes", 3),
        class_names=data.get("class_names"),
        auto_class_names_from_folders=data.get("auto_class_names_from_folders", False),
        pretrained=data.get("pretrained", True),
        freeze_backbone=data.get("freeze_backbone", False),
        dropout_rate=data.get("dropout_rate", 0.5),
        image_size=data.get("image_size", 224),
        num_workers=data.get("num_workers", 4),
        output_dir=Path(data.get("output_dir", "ml_output")),
        checkpoint_dir=Path(data.get("checkpoint_dir", "ml_output/checkpoints")),
        logs_dir=Path(data.get("logs_dir", "ml_output/logs")),
        device=data.get("device", "auto"),
    )
