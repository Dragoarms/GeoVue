"""Shared-folder path helpers for processed televiewer datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, List, Optional

TFD_HOLE_RE = re.compile(r"([A-Z]{2,4}\d{3,6})", re.IGNORECASE)
PROJECT_CODE_RE = re.compile(r"^([A-Z]+)")
DEFAULT_TELEVIEWER_DATASETS_PATH = Path(
    r"C:\Users\gsymonds\Fortescue Metals Group\Gabon - Belinga - Exploration Drilling\03 - Reverse Circulation\Chip Tray Photos\Televiewer Datasets"
)


@dataclass(frozen=True)
class TeleviewerDatasetPaths:
    """Canonical folder layout for one project/hole televiewer dataset."""

    root: Path
    project_code: str
    hole_id: str
    project_dir: Path
    hole_dir: Path
    raw_dir: Path
    processed_dir: Path
    qc_dir: Path
    annotations_dir: Path
    raw_by_record_dir: Path
    slices_1m_dir: Path
    thumbnails_dir: Path
    full_strip_dir: Path

    @property
    def manifest_path(self) -> Path:
        return self.processed_dir / "manifest.json"

    @property
    def viewer_data_path(self) -> Path:
        return self.processed_dir / "viewer_data.json"

    @property
    def chip_tray_manifest_path(self) -> Path:
        return self.processed_dir / "chip_tray_manifest.json"

    def ensure(self) -> None:
        """Create the folders that are safe to create before decoding starts."""
        for path in (
            self.project_dir,
            self.hole_dir,
            self.raw_dir,
            self.processed_dir,
            self.qc_dir,
            self.annotations_dir,
            self.raw_by_record_dir,
            self.slices_1m_dir,
            self.thumbnails_dir,
            self.full_strip_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class TeleviewerImportPlan:
    """A planned TFD import target before any bytes are copied or decoded."""

    source_path: Path
    project_code: str
    hole_id: str
    paths: TeleviewerDatasetPaths


def infer_hole_id_from_tfd(path: Path | str) -> str:
    """Infer a hole id from a TFD filename without requiring vendor metadata."""
    stem = Path(path).stem.upper().strip()
    match = TFD_HOLE_RE.search(stem)
    if match:
        return match.group(1)
    return re.sub(r"[^A-Z0-9]+", "_", stem).strip("_") or "UNKNOWN_HOLE"


def infer_project_code_from_hole(hole_id: str) -> str:
    """Use the leading letters of a GeoVue-style hole id as the project code."""
    match = PROJECT_CODE_RE.match(hole_id.upper().strip())
    return match.group(1) if match else "UNKNOWN"


def get_televiewer_root(file_manager, create_if_missing: bool = True) -> Optional[Path]:
    """Return GeoVue's configured shared Televiewer Datasets root."""
    candidates: List[Path] = []
    if file_manager is not None:
        try:
            root = file_manager.get_shared_path(
                "televiewer_datasets", create_if_missing=create_if_missing
            )
        except Exception:
            root = None
        if root:
            candidates.append(Path(root))

        config_manager = getattr(file_manager, "config_manager", None)
        if config_manager is not None:
            configured = config_manager.get("shared_folder_televiewer_datasets")
            if configured:
                candidates.append(Path(configured))

            shared_root = config_manager.get("shared_folder_path")
            if shared_root:
                candidates.append(Path(shared_root) / "Televiewer Datasets")

            approved = config_manager.get("shared_folder_approved_compartments_folder")
            if approved:
                approved_path = Path(approved)
                try:
                    candidates.append(approved_path.parents[1] / "Televiewer Datasets")
                except IndexError:
                    pass

    candidates.append(DEFAULT_TELEVIEWER_DATASETS_PATH)

    for candidate in _unique_paths(candidates):
        if _ensure_candidate(candidate, create_if_missing):
            return candidate
    return None


def build_dataset_paths(
    file_manager,
    project_code: str,
    hole_id: str,
    create_if_missing: bool = True,
) -> TeleviewerDatasetPaths:
    """Build canonical paths for one processed televiewer dataset."""
    root = get_televiewer_root(file_manager, create_if_missing=create_if_missing)
    if root is None:
        raise FileNotFoundError("GeoVue Televiewer Datasets folder is not configured")

    safe_project = re.sub(r"[^A-Z0-9_-]+", "_", project_code.upper()).strip("_")
    safe_hole = re.sub(r"[^A-Z0-9_-]+", "_", hole_id.upper()).strip("_")
    project_dir = root / (safe_project or "UNKNOWN")
    hole_dir = project_dir / (safe_hole or "UNKNOWN_HOLE")
    processed_dir = hole_dir / "processed"
    return TeleviewerDatasetPaths(
        root=root,
        project_code=safe_project or "UNKNOWN",
        hole_id=safe_hole or "UNKNOWN_HOLE",
        project_dir=project_dir,
        hole_dir=hole_dir,
        raw_dir=hole_dir / "raw",
        processed_dir=processed_dir,
        qc_dir=hole_dir / "qc",
        annotations_dir=hole_dir / "annotations",
        raw_by_record_dir=processed_dir / "raw_by_record",
        slices_1m_dir=processed_dir / "slices_1m",
        thumbnails_dir=processed_dir / "thumbnails",
        full_strip_dir=processed_dir / "full_strip",
    )


def plan_tfd_imports(file_manager, tfd_paths: Iterable[str | Path]) -> List[TeleviewerImportPlan]:
    """Plan canonical import locations for selected TFD files."""
    plans: List[TeleviewerImportPlan] = []
    for source in tfd_paths:
        source_path = Path(source)
        hole_id = infer_hole_id_from_tfd(source_path)
        project_code = infer_project_code_from_hole(hole_id)
        paths = build_dataset_paths(file_manager, project_code, hole_id)
        plans.append(
            TeleviewerImportPlan(
                source_path=source_path,
                project_code=project_code,
                hole_id=hole_id,
                paths=paths,
            )
        )
    return plans


def find_tfd_files(paths: Iterable[str | Path]) -> List[Path]:
    """Return unique `.tfd` files from selected files and folders."""
    found: List[Path] = []
    seen = set()
    for source in paths:
        path = Path(source)
        candidates: Iterable[Path]
        if path.is_dir():
            candidates = path.rglob("*.tfd")
        elif path.is_file() and path.suffix.lower() == ".tfd":
            candidates = (path,)
        else:
            candidates = ()

        for candidate in candidates:
            resolved = candidate.resolve()
            key = str(resolved).lower()
            if key not in seen:
                found.append(resolved)
                seen.add(key)
    return sorted(found, key=lambda item: str(item).lower())


def _ensure_candidate(path: Path, create_if_missing: bool) -> bool:
    if path.exists():
        return path.is_dir()
    if not create_if_missing:
        return False
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path.is_dir()
    except OSError:
        return False


def _unique_paths(paths: Iterable[Path]) -> List[Path]:
    unique: List[Path] = []
    seen = set()
    for path in paths:
        key = str(path).lower()
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique
