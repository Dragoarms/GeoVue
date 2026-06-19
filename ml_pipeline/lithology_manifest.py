"""Build auditable lithology training manifests from GeoVue review data.

The manifest is the bridge between the lithology review dialog and the ML
pipeline.  It keeps the approved image store read-only while recording exactly
which image paths, classes, and tags were used for a training run.
"""

from __future__ import annotations

import ast
import csv
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
UNASSIGNED_VALUES = {
    "",
    "unassigned",
    "none",
    "null",
    "classificationcategory.unassigned",
    "not confident",
    "not_confident",
}


@dataclass(frozen=True)
class LithologyManifestRow:
    """One image sample selected for lithology classifier training."""

    image_path: str
    label: str
    hole_id: str
    depth_from: float | None
    depth_to: float | None
    moisture_status: str
    tags: tuple[str, ...]
    source: str

    def to_csv_row(self) -> dict[str, str]:
        return {
            "image_path": self.image_path,
            "label": self.label,
            "hole_id": self.hole_id,
            "depth_from": "" if self.depth_from is None else f"{self.depth_from:g}",
            "depth_to": "" if self.depth_to is None else f"{self.depth_to:g}",
            "moisture_status": self.moisture_status,
            "tags": "|".join(self.tags),
            "source": self.source,
        }


def label_key(value: Any) -> str:
    """Return a comparison key that tolerates case and punctuation variants."""
    text = str(value or "").strip()
    text = text.replace("ClassificationCategory.", "")
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def normalise_label(value: Any, allowed_labels: Sequence[str]) -> str | None:
    """Map a raw review label onto one of the configured class labels."""
    raw = str(value or "").strip().replace("ClassificationCategory.", "")
    if raw.lower() in UNASSIGNED_VALUES:
        return None

    lookup = {label_key(label): label for label in allowed_labels}
    return lookup.get(label_key(raw))


def _parse_tag_text(value: str) -> list[str]:
    text = value.strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, (list, tuple, set)):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [part.strip() for part in re.split(r"[,|;]", text) if part.strip()]


def normalise_tags(value: Any) -> tuple[str, ...]:
    """Return a stable tuple of tag IDs from list/set/string review data."""
    if value is None:
        return ()
    if isinstance(value, str):
        tags = _parse_tag_text(value)
    elif isinstance(value, Mapping):
        tags = [str(k).strip() for k, v in value.items() if v and str(k).strip()]
    elif isinstance(value, Iterable):
        tags = [str(item).strip() for item in value if str(item).strip()]
    else:
        tags = [str(value).strip()]
    return tuple(sorted({tag for tag in tags if tag}))


def _classification_from_image(img: Any) -> Any:
    classification = getattr(img, "classification", "")
    if hasattr(classification, "value"):
        return classification.value
    return classification


def _tags_from_metadata_or_image(
    img: Any,
    metadata_fn: Optional[Callable[[Any], Mapping[str, Any]]],
) -> tuple[str, ...]:
    if metadata_fn is not None:
        try:
            metadata = metadata_fn(img)
        except Exception:
            metadata = {}
        tags = metadata.get("tags") if isinstance(metadata, Mapping) else None
        if tags:
            return normalise_tags(tags)
    return normalise_tags(getattr(img, "tags", ()))


def _image_is_readable_sample(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS and path.exists()


def build_manifest_rows_from_images(
    images: Sequence[Any],
    class_labels: Sequence[str],
    *,
    consensus_fn: Optional[Callable[[Any], str]] = None,
    metadata_fn: Optional[Callable[[Any], Mapping[str, Any]]] = None,
    use_consensus: bool = True,
    include_missing_images: bool = False,
    max_per_class: int | None = None,
    required_tags: Iterable[str] | None = None,
    excluded_tags: Iterable[str] | None = None,
    seed: int = 42,
) -> list[LithologyManifestRow]:
    """Create training rows from loaded review images.

    Args:
        images: GeoVue ``CompartmentImage``-like objects.
        class_labels: Display labels to train as mutually-exclusive classes.
        consensus_fn: Optional helper for consensus classification.
        metadata_fn: Optional helper that returns aggregated review tags.
        use_consensus: Prefer consensus labels when available.
        include_missing_images: Keep rows whose image path does not exist.
        max_per_class: Optional deterministic cap per class for balanced trials.
        required_tags: If supplied, samples must contain all of these tags.
        excluded_tags: If supplied, samples with any of these tags are skipped.
        seed: Shuffle seed used only when ``max_per_class`` caps rows.
    """
    labels = [str(label).strip() for label in class_labels if str(label).strip()]
    if not labels:
        return []

    required = set(normalise_tags(required_tags or ()))
    excluded = set(normalise_tags(excluded_tags or ()))

    rows_by_path: dict[str, LithologyManifestRow] = {}
    for img in images:
        raw_label = None
        source = "classification"
        if use_consensus and consensus_fn is not None:
            try:
                raw_label = consensus_fn(img)
            except Exception:
                raw_label = None
            if raw_label:
                source = "consensus"

        if not raw_label:
            raw_label = _classification_from_image(img)
            source = "classification"

        label = normalise_label(raw_label, labels)
        if label is None:
            continue

        image_path = str(getattr(img, "image_path", "") or "").strip()
        if not image_path:
            continue
        path = Path(image_path)
        if not include_missing_images and not _image_is_readable_sample(path):
            continue

        tags = _tags_from_metadata_or_image(img, metadata_fn)
        tag_set = set(tags)
        if required and not required.issubset(tag_set):
            continue
        if excluded and excluded.intersection(tag_set):
            continue

        # Resolve only after existence checks so tests can use lightweight temp files.
        key = str(path.resolve()).lower() if path.exists() else str(path).lower()
        if key in rows_by_path:
            continue

        rows_by_path[key] = LithologyManifestRow(
            image_path=str(path),
            label=label,
            hole_id=str(getattr(img, "hole_id", "") or "").strip(),
            depth_from=_to_float_or_none(getattr(img, "depth_from", None)),
            depth_to=_to_float_or_none(getattr(img, "depth_to", None)),
            moisture_status=str(getattr(img, "moisture_status", "") or "").strip(),
            tags=tags,
            source=source,
        )

    rows = list(rows_by_path.values())
    if max_per_class is not None and max_per_class > 0:
        rows = _cap_rows_per_class(rows, int(max_per_class), seed)
    return rows


def _to_float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _cap_rows_per_class(
    rows: Sequence[LithologyManifestRow],
    max_per_class: int,
    seed: int,
) -> list[LithologyManifestRow]:
    grouped: dict[str, list[LithologyManifestRow]] = defaultdict(list)
    for row in rows:
        grouped[row.label].append(row)

    rng = random.Random(seed)
    capped: list[LithologyManifestRow] = []
    for label in sorted(grouped):
        group = list(grouped[label])
        rng.shuffle(group)
        capped.extend(group[:max_per_class])

    return sorted(capped, key=lambda row: (row.hole_id, row.depth_to or -1, row.image_path))


def write_manifest_csv(rows: Sequence[LithologyManifestRow], path: Path) -> Path:
    """Write rows to CSV and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "label",
        "hole_id",
        "depth_from",
        "depth_to",
        "moisture_status",
        "tags",
        "source",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())
    return path


def load_manifest_csv(path: Path) -> list[LithologyManifestRow]:
    """Load a lithology manifest produced by ``write_manifest_csv``."""
    rows: list[LithologyManifestRow] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                LithologyManifestRow(
                    image_path=raw.get("image_path", ""),
                    label=raw.get("label", ""),
                    hole_id=raw.get("hole_id", ""),
                    depth_from=_to_float_or_none(raw.get("depth_from")),
                    depth_to=_to_float_or_none(raw.get("depth_to")),
                    moisture_status=raw.get("moisture_status", ""),
                    tags=normalise_tags(raw.get("tags", "")),
                    source=raw.get("source", ""),
                )
            )
    return rows


def summarise_rows(rows: Sequence[LithologyManifestRow]) -> dict[str, Any]:
    """Return counts useful for UI previews and training audit JSON."""
    class_counts = Counter(row.label for row in rows)
    tag_counts = Counter(tag for row in rows for tag in row.tags)
    holes_by_label: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.hole_id:
            holes_by_label[row.label].add(row.hole_id)

    return {
        "total_samples": len(rows),
        "class_counts": dict(sorted(class_counts.items())),
        "hole_counts": {
            label: len(holes)
            for label, holes in sorted(holes_by_label.items())
        },
        "tag_counts": dict(tag_counts.most_common()),
    }


def write_summary_json(rows: Sequence[LithologyManifestRow], path: Path) -> Path:
    """Write a compact dataset audit summary next to the manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summarise_rows(rows), indent=2), encoding="utf-8")
    return path
