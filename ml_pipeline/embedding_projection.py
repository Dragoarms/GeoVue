"""2D projection helpers for GeoVue image embeddings.

The advanced filter window uses these helpers to plot visual-embedding space
without rebuilding neural embeddings. UMAP is preferred when `umap-learn` is
installed; otherwise a deterministic PCA projection keeps the feature usable.
Projection results are cached beside the SQLite embedding DB so reopening the
window is fast and does not create tracked git artifacts under source folders.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ml_pipeline.embedding_store import SQLiteEmbeddingStore, decode_vector, path_key


@dataclass(frozen=True)
class ProjectionResult:
    """A 2D embedding projection aligned by normalized image path."""

    dataframe: pd.DataFrame
    method: str
    cache_path: Path
    loaded_from_cache: bool = False


def _pca_projection(matrix: np.ndarray) -> np.ndarray:
    """Return a deterministic 2D PCA projection using NumPy SVD."""
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)
    centered = matrix.astype(np.float32, copy=True)
    centered -= centered.mean(axis=0, keepdims=True)
    if centered.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    coords = centered @ components
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0], dtype=np.float32)])
    return coords[:, :2].astype(np.float32)


def _umap_projection(matrix: np.ndarray, *, random_state: int = 42) -> Optional[np.ndarray]:
    """Return a UMAP projection when umap-learn is available, else None."""
    try:
        import umap  # type: ignore
    except Exception:
        return None

    if matrix.shape[0] < 3:
        return _pca_projection(matrix)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(30, max(2, matrix.shape[0] - 1)),
        min_dist=0.05,
        metric="cosine",
        random_state=random_state,
        verbose=False,
    )
    return reducer.fit_transform(matrix).astype(np.float32)


def projection_cache_path(db_path: str | Path, model_name: str, method: str = "auto") -> Path:
    """Return the cache path used for a DB/model projection."""
    db = Path(db_path)
    safe_model = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(model_name or "model"))
    return db.with_name(f"{db.stem}_{safe_model}_{method}_projection.parquet")


def _read_cache(cache_path: Path) -> Optional[pd.DataFrame]:
    if not cache_path.exists():
        return None
    try:
        return pd.read_parquet(cache_path)
    except Exception:
        return None


def _write_cache(cache_path: Path, df: pd.DataFrame) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
    except Exception:
        # Cache writes are an optimization only. The caller still receives data.
        return


def load_embedding_projection(
    db_path: str | Path,
    *,
    model_name: Optional[str] = None,
    method: str = "auto",
    image_paths: Optional[Iterable[str | Path]] = None,
    use_cache: bool = True,
) -> ProjectionResult:
    """Load or build a 2D projection for embeddings in an SQLite store.

    Args:
        db_path: SQLite embedding DB from `build_similarity_index`.
        model_name: Optional model name. Defaults to DB metadata.
        method: `auto`, `umap`, or `pca`. `auto` tries UMAP then PCA.
        image_paths: Optional paths to keep in the returned DataFrame.
        use_cache: When true, reuse/write a Parquet projection cache.
    """
    db_path = Path(db_path)
    method = (method or "auto").lower()
    if method not in {"auto", "umap", "pca"}:
        raise ValueError(f"Unsupported projection method: {method}")

    with SQLiteEmbeddingStore(db_path) as store:
        resolved_model = model_name or store.get_metadata("embedding_model", "")
        if not resolved_model:
            raise RuntimeError("Embedding DB has no embedding_model metadata")

        cache_path = projection_cache_path(db_path, resolved_model, method)
        cached = _read_cache(cache_path) if use_cache else None
        allowed_keys = {path_key(path) for path in image_paths} if image_paths is not None else None
        if cached is not None and not cached.empty:
            if allowed_keys is not None and "path_key" in cached.columns:
                cached = cached[cached["path_key"].isin(allowed_keys)].copy()
            return ProjectionResult(cached, str(cached.get("projection_method", pd.Series([method])).iloc[0]), cache_path, True)

        rows = []
        vectors = []
        for batch in store.iter_rows(model_name=resolved_model, batch_size=4096):
            for row in batch:
                key = path_key(row["path"])
                if allowed_keys is not None and key not in allowed_keys:
                    continue
                rows.append(
                    {
                        "image_path": row["path"],
                        "path_key": key,
                        "hole_id": row["hole_id"] or "",
                        "depth_to": row["depth"],
                        "embedding_label": row["label"] or "",
                    }
                )
                vectors.append(decode_vector(row["embedding"], row["embedding_dim"], row["embedding_dtype"]))

    if not vectors:
        empty = pd.DataFrame(columns=["image_path", "path_key", "hole_id", "depth_to", "umap_x", "umap_y", "projection_method"])
        return ProjectionResult(empty, method, projection_cache_path(db_path, model_name or "model", method), False)

    matrix = np.vstack(vectors).astype(np.float32)
    coords = None
    used_method = method
    if method in {"auto", "umap"}:
        coords = _umap_projection(matrix)
        if coords is not None:
            used_method = "umap"
        elif method == "umap":
            used_method = "pca"
    if coords is None:
        coords = _pca_projection(matrix)
        used_method = "pca"

    out = pd.DataFrame(rows)
    out["umap_x"] = coords[:, 0]
    out["umap_y"] = coords[:, 1]
    out["projection_method"] = used_method

    cache_path = projection_cache_path(db_path, resolved_model, method)
    if use_cache and image_paths is None:
        _write_cache(cache_path, out)
    return ProjectionResult(out, used_method, cache_path, False)


def merge_projection_columns(data: pd.DataFrame, projection: pd.DataFrame) -> pd.DataFrame:
    """Merge projection columns into an AdvancedFilterWindow DataFrame."""
    if data is None or data.empty or projection is None or projection.empty:
        return data
    out = data.copy()
    if "image_path" in out.columns:
        out["path_key"] = out["image_path"].map(path_key)
        return out.merge(
            projection[["path_key", "umap_x", "umap_y", "projection_method"]],
            on="path_key",
            how="left",
        )
    if {"hole_id", "depth_to"}.issubset(out.columns):
        proj = projection[["hole_id", "depth_to", "umap_x", "umap_y", "projection_method"]].copy()
        proj["depth_to"] = pd.to_numeric(proj["depth_to"], errors="coerce")
        out["depth_to"] = pd.to_numeric(out["depth_to"], errors="coerce")
        # The projection DB is image-path based, while the scatter table is
        # interval based. Average duplicate wet/dry/provenance points to one
        # interval coordinate so the merge does not multiply rows.
        proj = (
            proj.dropna(subset=["hole_id", "depth_to", "umap_x", "umap_y"])
            .groupby(["hole_id", "depth_to"], as_index=False)
            .agg({"umap_x": "mean", "umap_y": "mean", "projection_method": "first"})
        )
        return out.merge(proj, on=["hole_id", "depth_to"], how="left")
    return out
