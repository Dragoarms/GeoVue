"""
processing/embedding_trainer.py

Utilities for generating neural network embeddings from images and
tabular data. Provides a simple dataset class, embedding model and
helper functions for producing and visualising embeddings.
"""

from typing import Sequence, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image


class ImageTabularDataset(Dataset):
    """Dataset combining image files with accompanying tabular values."""

    def __init__(self, image_paths: Sequence[str], tabular_values: Sequence[Sequence[float]], transform: Optional[transforms.Compose] = None) -> None:
        self.image_paths = list(image_paths)
        self.tabular_values = [list(v) for v in tabular_values]
        self.transform = transform or transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        tabular = torch.tensor(self.tabular_values[idx], dtype=torch.float32)
        return image, tabular


class EmbeddingModel(nn.Module):
    """Simple model that combines ResNet features with tabular features."""

    def __init__(self, tabular_dim: int) -> None:
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(base.children())[:-1])
        self.tabular_net = nn.Sequential(
            nn.Linear(tabular_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
        )

    def forward(self, img: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        x1 = self.cnn(img).squeeze()
        x2 = self.tabular_net(tabular)
        return torch.cat((x1, x2), dim=1)


def generate_embeddings(image_paths: Sequence[str], tabular_values: Sequence[Sequence[float]], batch_size: int = 16, device: Optional[torch.device] = None) -> np.ndarray:
    """Generate embeddings for provided images and tabular values."""

    device = device or torch.device("cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageTabularDataset(image_paths, tabular_values, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = EmbeddingModel(len(tabular_values[0])).to(device).eval()

    embeddings = []
    with torch.no_grad():
        for imgs, tabs in loader:
            imgs = imgs.to(device)
            tabs = tabs.to(device)
            out = model(imgs, tabs)
            embeddings.append(out.cpu().numpy())

    return np.vstack(embeddings)


def plot_embeddings(embeddings: np.ndarray, save_path: str, gui_manager=None) -> None:
    """Plot embeddings using PCA and save the figure."""

    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    plt.style.use('dark_background')
    if gui_manager:
        colors = gui_manager.theme_colors
        plt.rcParams['text.color'] = colors['text']
        plt.rcParams['axes.labelcolor'] = colors['text']
        plt.rcParams['axes.edgecolor'] = colors['text']
        plt.rcParams['axes.facecolor'] = colors['background']
        plt.rcParams['figure.facecolor'] = colors['background']
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], s=20, alpha=0.7)
    plt.title('Embedding Projection (PCA)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
