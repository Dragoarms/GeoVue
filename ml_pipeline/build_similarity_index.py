"""Build and query a local SQLite image-similarity embedding index."""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .embedding_store import (
    ImageEmbeddingRecord,
    SQLiteEmbeddingStore,
    image_fingerprint,
    path_key,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
IMAGE_KEY_PATTERN = re.compile(
    r"(?P<hole>[A-Za-z]{1,4}\d{3,5})_CC_(?P<depth>\d+(?:[pP.]\d+)?)",
    re.IGNORECASE,
)
MODEL_NAME = "mobilenet_v3_small_imagenet_128"
EMBED_EDGE = 128


def scan_images(image_root: Path) -> list[Path]:
    """Return all supported image paths below the root."""
    return [
        path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def parse_image_metadata(path: str | Path) -> dict:
    """Extract common GeoVue metadata from compartment filenames."""
    path = Path(path)
    match = IMAGE_KEY_PATTERN.search(path.name)
    if not match:
        return {"hole_id": "", "depth": None}
    depth_text = match.group("depth").replace("p", ".").replace("P", ".")
    try:
        depth = float(depth_text)
    except ValueError:
        depth = None
    return {"hole_id": match.group("hole").upper(), "depth": depth}


def load_manifest_metadata(manifest: Path | None) -> dict[str, dict]:
    """Optional label/tag metadata keyed by normalized image path."""
    if manifest is None or not manifest.exists():
        return {}
    out = {}
    with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            image_path = row.get("image_path") or row.get("path") or ""
            if not image_path:
                continue
            tags = tuple(tag for tag in (row.get("tags") or "").split("|") if tag)
            out[path_key(image_path)] = {
                "hole_id": row.get("hole_id") or "",
                "depth": _to_float(row.get("depth_to") or row.get("depth")),
                "label": row.get("label") or row.get("classification") or "",
                "tags": tags,
            }
    return out


def _to_float(value) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


class MobileNetEmbedder:
    """MobileNetV3-small feature extractor matching the visual builder."""

    def __init__(self, *, pretrained: bool = True, edge: int = EMBED_EDGE, device: str = "auto"):
        try:
            import torch
            from torchvision import models, transforms
        except Exception as exc:
            raise RuntimeError("torch/torchvision are required to build embeddings") from exc

        self.torch = torch
        self.transforms = transforms
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.model = models.mobilenet_v3_small(weights=weights)
        self.model.classifier = torch.nn.Identity()
        self.model.eval().to(self.device)
        self.edge = edge
        self.model_name = MODEL_NAME if pretrained else "mobilenet_v3_small_scratch_128"
        self.transform = transforms.Compose(
            [
                transforms.Resize((edge, edge)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def embed_paths(self, paths: Sequence[Path], batch_size: int = 64) -> tuple[list[Path], np.ndarray]:
        from PIL import Image

        vector_paths = []
        vectors = []
        with self.torch.no_grad():
            for start in range(0, len(paths), batch_size):
                batch_paths = paths[start : start + batch_size]
                images = []
                kept_paths = []
                for path in batch_paths:
                    try:
                        images.append(self.transform(Image.open(path).convert("RGB")))
                        kept_paths.append(path)
                    except Exception:
                        continue
                if not images:
                    continue
                batch = self.torch.stack(images).to(self.device)
                out = self.model(batch).float().cpu().numpy().astype(np.float32)
                vector_paths.extend(kept_paths)
                vectors.extend(out)
        return vector_paths, np.asarray(vectors, dtype=np.float32)

    def embed_one(self, path: Path) -> np.ndarray:
        _, vectors = self.embed_paths([path], batch_size=1)
        if vectors.size == 0:
            raise ValueError(f"Could not read query image: {path}")
        return vectors[0]

def build_index(args: argparse.Namespace) -> None:
    image_root = args.image_root
    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")

    metadata = load_manifest_metadata(args.manifest)
    paths = scan_images(image_root)
    if args.limit:
        paths = paths[: args.limit]
    print(f"Scanned {len(paths):,} images under {image_root}")

    embedder = MobileNetEmbedder(pretrained=not args.no_pretrained, device=args.device)
    with SQLiteEmbeddingStore(args.db) as store:
        store.set_metadata("embedding_model", embedder.model_name)
        store.set_metadata("embedding_edge", EMBED_EDGE)
        stale = store.stale_or_missing_paths(paths, model_name=embedder.model_name)
        print(f"Need embeddings for {len(stale):,} missing/stale images.")
        start_time = time.time()
        done = 0
        for start in range(0, len(stale), args.batch_size):
            batch_paths = stale[start : start + args.batch_size]
            kept_paths, vectors = embedder.embed_paths(batch_paths, batch_size=args.batch_size)
            if len(vectors) != len(batch_paths):
                print("Warning: one or more images in this batch could not be decoded.")

            for path, vector in zip(kept_paths, vectors):
                key = path_key(path)
                row_metadata = dict(parse_image_metadata(path))
                row_metadata.update(metadata.get(key, {}))
                mtime_ns, size_bytes = image_fingerprint(path)
                store.upsert_embedding(
                    ImageEmbeddingRecord(
                        path=str(path),
                        vector=vector,
                        model_name=embedder.model_name,
                        hole_id=row_metadata.get("hole_id") or "",
                        depth=row_metadata.get("depth"),
                        label=row_metadata.get("label") or "",
                        tags=tuple(row_metadata.get("tags") or ()),
                        image_mtime_ns=mtime_ns,
                        image_size_bytes=size_bytes,
                    ),
                    dtype=args.dtype,
                    commit=False,
                )
            store.commit()
            done += len(batch_paths)
            rate = done / max(1e-6, time.time() - start_time)
            print(f"Embedded {done:,}/{len(stale):,} ({rate:.1f} img/s)", flush=True)

        print(f"Database ready: {args.db}")
        print(f"Stored embeddings: {store.count(embedder.model_name):,}")


def search_index(args: argparse.Namespace) -> None:
    embedder = MobileNetEmbedder(pretrained=not args.no_pretrained, device=args.device)
    query_vector = None
    with SQLiteEmbeddingStore(args.db) as store:
        if args.use_stored_query:
            query_vector = store.get_embedding(args.query_image, model_name=embedder.model_name)
        if query_vector is None:
            query_vector = embedder.embed_one(args.query_image)
        results = store.search(
            query_vector,
            top_k=args.top_k,
            model_name=embedder.model_name,
            exclude_paths=[args.query_image] if args.exclude_query else [],
        )
    for rank, result in enumerate(results, start=1):
        depth = "" if result.depth is None else f"{result.depth:g}"
        print(
            f"{rank:>3}. {result.score:.4f}  {result.hole_id} {depth}  "
            f"{result.label}  {result.path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build or update the SQLite embedding index.")
    build.add_argument("--image-root", type=Path, required=True)
    build.add_argument("--db", type=Path, default=Path("ml_output/image_similarity/geovue_embeddings.sqlite"))
    build.add_argument("--manifest", type=Path, help="Optional training manifest for labels/tags.")
    build.add_argument("--batch-size", type=int, default=64)
    build.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    build.add_argument("--limit", type=int, default=0)
    build.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    build.add_argument("--no-pretrained", action="store_true")
    build.set_defaults(func=build_index)

    search = sub.add_parser("search", help="Search nearest images for one query image.")
    search.add_argument("--db", type=Path, default=Path("ml_output/image_similarity/geovue_embeddings.sqlite"))
    search.add_argument("--query-image", type=Path, required=True)
    search.add_argument("--top-k", type=int, default=50)
    search.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    search.add_argument("--no-pretrained", action="store_true")
    search.add_argument("--use-stored-query", action="store_true")
    search.add_argument("--include-query", dest="exclude_query", action="store_false")
    search.set_defaults(func=search_index, exclude_query=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
