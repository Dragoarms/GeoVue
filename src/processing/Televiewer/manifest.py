"""Manifest helpers for processed televiewer datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1
PROCESSING_VERSION = "geovue-televiewer-prototype-1"


@dataclass
class TeleviewerManifest:
    """Serializable index for a processed televiewer hole."""

    project_code: str
    hole_id: str
    schema_version: int = SCHEMA_VERSION
    processing_version: str = PROCESSING_VERSION
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    raw_files: List[Dict[str, Any]] = field(default_factory=list)
    coverage: Dict[str, Any] = field(default_factory=dict)
    viewer: Dict[str, Any] = field(default_factory=dict)
    telemetry: Dict[str, Any] = field(default_factory=dict)
    qc: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def placeholder_manifest(project_code: str, hole_id: str) -> TeleviewerManifest:
    """Create a manifest that records the planned dataset contract before decoding."""
    return TeleviewerManifest(
        project_code=project_code,
        hole_id=hole_id,
        viewer={
            "hole_id": hole_id,
            "data_url": "viewer_data.json",
            "chip_tray_manifest_url": "chip_tray_manifest.json",
            "raw_dir": "raw_by_record",
            "resampled_dir": "slices_1m",
        },
        notes=[
            "Placeholder manifest: TFD decoding has not populated viewer_data.json yet."
        ],
    )


def load_manifest(path: Path | str) -> Dict[str, Any]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_manifest(path: Path | str, manifest: TeleviewerManifest | Dict[str, Any]) -> Path:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.to_dict() if isinstance(manifest, TeleviewerManifest) else manifest
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest_path


def discover_manifests(root: Path | str | None) -> List[Path]:
    """Find processed televiewer manifests under Televiewer Datasets."""
    if root is None:
        return []
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(root_path.glob(f"*/*/processed/{MANIFEST_NAME}"))


def manifest_status(manifest_path: Path | str) -> str:
    """Return a compact status string for launcher dialogs."""
    path = Path(manifest_path)
    if not path.exists():
        return "missing manifest"
    try:
        manifest = load_manifest(path)
    except Exception:
        return "invalid manifest"
    viewer = manifest.get("viewer", {}) if isinstance(manifest, dict) else {}
    data_url = viewer.get("data_url", "viewer_data.json")
    data_path = path.parent / data_url
    if data_path.exists():
        return "viewer data ready"
    return "manifest only"
