"""SQLite-backed image embedding store for local similarity search."""

from __future__ import annotations

import heapq
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


SUPPORTED_DTYPES = {"float16": np.float16, "float32": np.float32}


@dataclass(frozen=True)
class ImageEmbeddingRecord:
    """A normalized image embedding plus searchable metadata."""

    path: str
    vector: np.ndarray
    model_name: str
    hole_id: str = ""
    depth: float | None = None
    label: str = ""
    tags: tuple[str, ...] = ()
    image_mtime_ns: int | None = None
    image_size_bytes: int | None = None


@dataclass(frozen=True)
class SimilarityResult:
    """A nearest-neighbour search hit."""

    path: str
    score: float
    hole_id: str
    depth: float | None
    label: str
    tags: tuple[str, ...]


def path_key(path: str | Path) -> str:
    """Normalize paths for case-insensitive Windows lookup."""
    return str(path).replace("\\", "/").lower()


def image_fingerprint(path: str | Path) -> tuple[int | None, int | None]:
    """Return mtime/size fingerprint for stale-cache detection."""
    try:
        stat = Path(path).stat()
    except OSError:
        return None, None
    return stat.st_mtime_ns, stat.st_size


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Return a float32 unit vector suitable for cosine similarity."""
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        return arr
    return arr / norm


def encode_vector(vector: np.ndarray, dtype: str = "float16") -> bytes:
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported embedding dtype: {dtype}")
    return normalize_vector(vector).astype(SUPPORTED_DTYPES[dtype]).tobytes()


def decode_vector(blob: bytes, dim: int, dtype: str) -> np.ndarray:
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported embedding dtype: {dtype}")
    return np.frombuffer(blob, dtype=SUPPORTED_DTYPES[dtype], count=dim).astype(np.float32)


class SQLiteEmbeddingStore:
    """Small local SQLite database for normalized image embeddings."""

    def __init__(self, db_path: str | Path):
        self.db_path = db_path
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "SQLiteEmbeddingStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;

            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL,
                path_key TEXT NOT NULL,
                hole_id TEXT,
                depth REAL,
                label TEXT,
                tags_json TEXT NOT NULL DEFAULT '[]',
                model_name TEXT NOT NULL,
                embedding_dim INTEGER NOT NULL,
                embedding_dtype TEXT NOT NULL,
                embedding BLOB NOT NULL,
                image_mtime_ns INTEGER,
                image_size_bytes INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_embeddings_model
                ON embeddings(model_name);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_embeddings_path_model
                ON embeddings(path_key, model_name);
            CREATE INDEX IF NOT EXISTS idx_embeddings_hole_depth
                ON embeddings(hole_id, depth);
            CREATE INDEX IF NOT EXISTS idx_embeddings_label
                ON embeddings(label);

            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def set_metadata(self, key: str, value: Any) -> None:
        self.conn.execute(
            """
            INSERT INTO metadata(key, value)
            VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (key, json.dumps(value)),
        )
        self.conn.commit()

    def get_metadata(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        return json.loads(row["value"])

    def count(self, model_name: str | None = None) -> int:
        if model_name:
            row = self.conn.execute(
                "SELECT COUNT(*) AS n FROM embeddings WHERE model_name = ?",
                (model_name,),
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) AS n FROM embeddings").fetchone()
        return int(row["n"])

    def upsert_embedding(
        self,
        record: ImageEmbeddingRecord,
        *,
        dtype: str = "float16",
        commit: bool = True,
    ) -> None:
        vec = normalize_vector(record.vector)
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """
            INSERT INTO embeddings(
                path, path_key, hole_id, depth, label, tags_json, model_name,
                embedding_dim, embedding_dtype, embedding, image_mtime_ns,
                image_size_bytes, created_at, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path_key, model_name) DO UPDATE SET
                path=excluded.path,
                hole_id=excluded.hole_id,
                depth=excluded.depth,
                label=excluded.label,
                tags_json=excluded.tags_json,
                model_name=excluded.model_name,
                embedding_dim=excluded.embedding_dim,
                embedding_dtype=excluded.embedding_dtype,
                embedding=excluded.embedding,
                image_mtime_ns=excluded.image_mtime_ns,
                image_size_bytes=excluded.image_size_bytes,
                updated_at=excluded.updated_at
            """,
            (
                record.path,
                path_key(record.path),
                record.hole_id,
                record.depth,
                record.label,
                json.dumps(list(record.tags)),
                record.model_name,
                int(vec.shape[0]),
                dtype,
                encode_vector(vec, dtype),
                record.image_mtime_ns,
                record.image_size_bytes,
                now,
                now,
            ),
        )
        if commit:
            self.conn.commit()

    def commit(self) -> None:
        self.conn.commit()

    def get_embedding(self, path: str | Path, model_name: str | None = None) -> np.ndarray | None:
        if model_name:
            row = self.conn.execute(
                """
                SELECT embedding, embedding_dim, embedding_dtype
                FROM embeddings
                WHERE path_key = ? AND model_name = ?
                """,
                (path_key(path), model_name),
            ).fetchone()
        else:
            row = self.conn.execute(
                """
                SELECT embedding, embedding_dim, embedding_dtype
                FROM embeddings
                WHERE path_key = ?
                """,
                (path_key(path),),
            ).fetchone()
        if row is None:
            return None
        return decode_vector(row["embedding"], row["embedding_dim"], row["embedding_dtype"])

    def stale_or_missing_paths(
        self,
        paths: Sequence[str | Path],
        *,
        model_name: str,
        chunk_size: int = 800,
    ) -> list[Path]:
        """Return paths not indexed for this model, or whose file fingerprint changed."""
        out: list[Path] = []
        path_list = [Path(p) for p in paths]
        for start in range(0, len(path_list), chunk_size):
            chunk = path_list[start : start + chunk_size]
            keys = [path_key(path) for path in chunk]
            placeholders = ",".join("?" for _ in keys)
            rows = self.conn.execute(
                f"""
                SELECT path_key, image_mtime_ns, image_size_bytes
                FROM embeddings
                WHERE model_name = ? AND path_key IN ({placeholders})
                """,
                [model_name, *keys],
            ).fetchall()
            known = {row["path_key"]: row for row in rows}
            for path, key in zip(chunk, keys):
                mtime_ns, size_bytes = image_fingerprint(path)
                row = known.get(key)
                if row is None:
                    out.append(path)
                elif row["image_mtime_ns"] != mtime_ns or row["image_size_bytes"] != size_bytes:
                    out.append(path)
        return out

    def iter_rows(
        self,
        *,
        model_name: str | None = None,
        batch_size: int = 4096,
    ) -> Iterator[list[sqlite3.Row]]:
        sql = (
            "SELECT path, hole_id, depth, label, tags_json, embedding, "
            "embedding_dim, embedding_dtype FROM embeddings"
        )
        params: list[Any] = []
        if model_name:
            sql += " WHERE model_name = ?"
            params.append(model_name)
        sql += " ORDER BY id"
        cursor = self.conn.execute(sql, params)
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            yield rows

    def search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int = 50,
        model_name: str | None = None,
        required_tags: Iterable[str] | None = None,
        excluded_tags: Iterable[str] | None = None,
        exclude_paths: Iterable[str | Path] | None = None,
        batch_size: int = 4096,
    ) -> list[SimilarityResult]:
        """Return top cosine-similar embeddings using chunked SQLite reads."""
        query = normalize_vector(query_vector)
        required = set(required_tags or ())
        excluded = set(excluded_tags or ())
        excluded_path_keys = {path_key(path) for path in (exclude_paths or ())}

        heap: list[tuple[float, int, sqlite3.Row]] = []
        seen = 0
        for rows in self.iter_rows(model_name=model_name, batch_size=batch_size):
            kept_rows = []
            vectors = []
            for row in rows:
                if path_key(row["path"]) in excluded_path_keys:
                    continue
                tags = tuple(json.loads(row["tags_json"] or "[]"))
                tag_set = set(tags)
                if required and not required.issubset(tag_set):
                    continue
                if excluded and excluded.intersection(tag_set):
                    continue
                kept_rows.append(row)
                vectors.append(decode_vector(row["embedding"], row["embedding_dim"], row["embedding_dtype"]))
            if not vectors:
                continue

            matrix = np.vstack(vectors)
            scores = matrix @ query
            for row, score in zip(kept_rows, scores):
                item = (float(score), seen, row)
                seen += 1
                if len(heap) < top_k:
                    heapq.heappush(heap, item)
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, item)

        results = []
        for score, _, row in sorted(heap, key=lambda item: item[0], reverse=True):
            results.append(
                SimilarityResult(
                    path=row["path"],
                    score=score,
                    hole_id=row["hole_id"] or "",
                    depth=row["depth"],
                    label=row["label"] or "",
                    tags=tuple(json.loads(row["tags_json"] or "[]")),
                )
            )
        return results


def records_from_vectors(
    paths: Sequence[str | Path],
    vectors: np.ndarray,
    *,
    model_name: str,
    metadata_by_path: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[ImageEmbeddingRecord]:
    """Pair batch vectors with path metadata for store insertion."""
    metadata_by_path = metadata_by_path or {}
    records = []
    for path, vector in zip(paths, vectors):
        key = path_key(path)
        metadata = metadata_by_path.get(key, {})
        mtime_ns, size_bytes = image_fingerprint(path)
        records.append(
            ImageEmbeddingRecord(
                path=str(path),
                vector=vector,
                model_name=model_name,
                hole_id=str(metadata.get("hole_id", "")),
                depth=metadata.get("depth"),
                label=str(metadata.get("label", "")),
                tags=tuple(metadata.get("tags", ())),
                image_mtime_ns=mtime_ns,
                image_size_bytes=size_bytes,
            )
        )
    return records
