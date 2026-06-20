from __future__ import annotations

import numpy as np
from pathlib import Path

from ml_pipeline.embedding_store import (
    ImageEmbeddingRecord,
    SQLiteEmbeddingStore,
    normalize_vector,
)


def test_sqlite_embedding_store_searches_by_cosine_similarity():
    with SQLiteEmbeddingStore(":memory:") as store:
        store.upsert_embedding(
            ImageEmbeddingRecord(
                path="C:/fake/a.png",
                vector=np.array([1.0, 0.0, 0.0]),
                model_name="test_model",
                hole_id="BA0001",
                depth=1.0,
                label="BIFf",
                tags=("jasperlitic",),
            )
        )
        store.upsert_embedding(
            ImageEmbeddingRecord(
                path="C:/fake/b.png",
                vector=np.array([0.9, 0.1, 0.0]),
                model_name="test_model",
                hole_id="BA0001",
                depth=2.0,
                label="BIFf-s",
            )
        )
        store.upsert_embedding(
            ImageEmbeddingRecord(
                path="C:/fake/c.png",
                vector=np.array([-1.0, 0.0, 0.0]),
                model_name="test_model",
                hole_id="BA0002",
                depth=1.0,
                label="BIFhm",
            )
        )

        hits = store.search(np.array([1.0, 0.0, 0.0]), model_name="test_model", top_k=2)

    assert [hit.path for hit in hits] == ["C:/fake/a.png", "C:/fake/b.png"]
    assert hits[0].score > hits[1].score
    assert hits[0].tags == ("jasperlitic",)


def test_sqlite_embedding_store_filters_tags_and_excluded_paths():
    with SQLiteEmbeddingStore(":memory:") as store:
        store.upsert_embedding(
            ImageEmbeddingRecord(
                path="C:/fake/a.png",
                vector=np.array([1.0, 0.0]),
                model_name="test_model",
                tags=("keep",),
            )
        )
        store.upsert_embedding(
            ImageEmbeddingRecord(
                path="C:/fake/b.png",
                vector=np.array([0.99, 0.01]),
                model_name="test_model",
                tags=("skip",),
            )
        )

        hits = store.search(
            np.array([1.0, 0.0]),
            model_name="test_model",
            required_tags=("keep",),
            exclude_paths=("C:/fake/a.png",),
        )

    assert hits == []


def test_sqlite_embedding_store_reports_missing_and_reuses_existing_fingerprint():
    with SQLiteEmbeddingStore(":memory:") as store:
        paths = ["C:/fake/a.png", "C:/fake/b.png"]
        assert store.stale_or_missing_paths(paths, model_name="test_model") == [
            Path("C:/fake/a.png"),
            Path("C:/fake/b.png"),
        ]

        store.upsert_embedding(
            ImageEmbeddingRecord(
                path="C:/fake/a.png",
                vector=np.array([1.0, 0.0]),
                model_name="test_model",
                image_mtime_ns=None,
                image_size_bytes=None,
            )
        )

        assert store.stale_or_missing_paths(paths, model_name="test_model") == [
            Path("C:/fake/b.png")
        ]


def test_normalize_vector_handles_zero_vector():
    out = normalize_vector(np.array([0.0, 0.0]))
    assert np.array_equal(out, np.array([0.0, 0.0], dtype=np.float32))
