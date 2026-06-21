from __future__ import annotations

import numpy as np
import pandas as pd

from ml_pipeline.embedding_projection import load_embedding_projection, merge_projection_columns


def _embedding_row(path: str, vector: np.ndarray, idx: int) -> dict:
    arr = np.asarray(vector, dtype=np.float32)
    return {
        "path": path,
        "hole_id": "BA0001",
        "depth": float(idx),
        "label": "test",
        "embedding": arr.tobytes(),
        "embedding_dim": int(arr.shape[0]),
        "embedding_dtype": "float32",
    }


class FakeEmbeddingStore:
    rows = [
        _embedding_row("C:/chips/BA0001_CC_001.png", np.array([1.0, 0.0, 0.0]), 1),
        _embedding_row("C:/chips/BA0001_CC_002.png", np.array([0.8, 0.2, 0.0]), 2),
        _embedding_row("C:/chips/BA0001_CC_003.png", np.array([0.0, 1.0, 0.0]), 3),
        _embedding_row("C:/chips/BA0001_CC_004.png", np.array([0.0, 0.8, 0.2]), 4),
    ]

    def __init__(self, _db_path):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def get_metadata(self, key, default=None):
        return "test_model" if key == "embedding_model" else default

    def iter_rows(self, *, model_name=None, batch_size=4096):
        assert model_name == "test_model"
        yield self.rows


def test_load_embedding_projection_builds_pca_coordinates(monkeypatch):
    monkeypatch.setattr("ml_pipeline.embedding_projection.SQLiteEmbeddingStore", FakeEmbeddingStore)

    result = load_embedding_projection("unused.sqlite", method="pca", use_cache=False)

    assert result.method == "pca"
    assert result.loaded_from_cache is False
    assert {"image_path", "path_key", "hole_id", "depth_to", "umap_x", "umap_y"}.issubset(result.dataframe.columns)
    assert len(result.dataframe) == 4
    assert result.dataframe[["umap_x", "umap_y"]].notna().all().all()


def test_merge_projection_columns_averages_duplicate_interval_images():
    data = pd.DataFrame(
        {
            "hole_id": ["BA0001", "BA0001", "BA0002"],
            "depth_to": [1.0, 2.0, 1.0],
            "Fe_pct_BEST": [60.0, 61.0, 40.0],
        }
    )
    projection = pd.DataFrame(
        {
            "hole_id": ["BA0001", "BA0001", "BA0001"],
            "depth_to": [1.0, 1.0, 2.0],
            "umap_x": [2.0, 4.0, 10.0],
            "umap_y": [6.0, 8.0, 20.0],
            "projection_method": ["pca", "pca", "pca"],
        }
    )

    merged = merge_projection_columns(data, projection)

    assert len(merged) == len(data)
    first = merged.loc[merged["depth_to"] == 1.0].iloc[0]
    assert first["umap_x"] == 3.0
    assert first["umap_y"] == 7.0
    assert merged.loc[merged["depth_to"] == 2.0, "umap_x"].iloc[0] == 10.0
    assert merged.loc[merged["hole_id"] == "BA0002", "umap_x"].isna().all()
