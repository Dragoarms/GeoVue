"""Enrich decoded televiewer viewer data with GeoVue context."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pandas as pd
except Exception:  # pragma: no cover - GeoVue runtime normally has pandas
    pd = None

try:
    from processing.DataManager.keys import FilenameParser
except Exception:  # pragma: no cover - keep import-safe for isolated decoder tests
    FilenameParser = None

from .paths import TeleviewerDatasetPaths

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

POINT_SOURCE_SPECS = {
    "geophysicsdetails": {
        "target": "geophysics",
        "depth_aliases": ["depth", "measurement_depth", "point_depth", "md", "measured_depth"],
    },
}

INTERVAL_SOURCE_TARGETS = {
    "summary_logging_assays": "assays",
    "sample_best_assays": "assays",
    "exassay": "assays",
    "normative_mineralogy": "mineralogy",
}

CHIP_PATTERNS = [
    re.compile(r"(?P<hole>[A-Z]{2,4}\d{3,6})[^\d]+(?P<from>\d+(?:\.\d+)?)\s*[-_]\s*(?P<to>\d+(?:\.\d+)?)\s*m?", re.IGNORECASE),
    re.compile(r"(?P<hole>[A-Z]{2,4}\d{3,6})_CC_(?P<to>\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"(?P<hole>[A-Z]{2,4}\d{3,6})[^\d]+(?P<to>\d{1,4})(?:m)?", re.IGNORECASE),
]


@dataclass
class TeleviewerEnrichmentResult:
    """Summary of enrichment written for one Televiewer dataset."""

    viewer_data_path: Path
    chip_manifest_path: Optional[Path] = None
    collar_rows: int = 0
    survey_rows: int = 0
    trace_rows: int = 0
    geophysics_rows: int = 0
    assay_rows: int = 0
    mineralogy_rows: int = 0
    chip_rows: int = 0
    warnings: List[str] = field(default_factory=list)


class TeleviewerDataEnricher:
    """Join decoded OTV assets with existing GeoVue datasets where available."""

    def __init__(self, file_manager=None, data_manager=None, logger=None):
        self.file_manager = file_manager
        self.data_manager = data_manager
        self.logger = logger

    def enrich_dataset(
        self,
        paths: TeleviewerDatasetPaths,
        hole_id: str,
        max_depth: Optional[float] = None,
    ) -> TeleviewerEnrichmentResult:
        """Enrich a decoded Televiewer dataset in place."""
        viewer_data_path = paths.viewer_data_path
        if not viewer_data_path.exists():
            raise FileNotFoundError(f"Viewer data not found: {viewer_data_path}")

        payload = _load_json(viewer_data_path)
        result = TeleviewerEnrichmentResult(viewer_data_path=viewer_data_path)
        hole_upper = str(hole_id).strip().upper()
        max_depth = max_depth or _payload_max_depth(payload)

        if self.data_manager is not None:
            self._merge_collar_survey_trace(payload, hole_upper, max_depth, result)
            self._merge_source_layers(payload, hole_upper, result)
        else:
            result.warnings.append("No GeoVue data manager supplied; leaving fallback trace/data layers.")

        chip_rows = self.build_chip_tray_manifest(hole_upper, paths.chip_tray_manifest_path)
        if chip_rows:
            result.chip_manifest_path = paths.chip_tray_manifest_path
            result.chip_rows = len(chip_rows)
        else:
            result.warnings.append("No approved chip tray images found for this hole.")

        payload.setdefault("stats", {})["enrichment"] = {
            "collar_rows": result.collar_rows,
            "survey_rows": result.survey_rows,
            "trace_rows": result.trace_rows,
            "geophysics_rows": result.geophysics_rows,
            "assay_rows": result.assay_rows,
            "mineralogy_rows": result.mineralogy_rows,
            "chip_rows": result.chip_rows,
            "warnings": result.warnings,
        }
        _write_json(viewer_data_path, payload)
        return result

    def _merge_collar_survey_trace(
        self,
        payload: Dict[str, Any],
        hole_id: str,
        max_depth: Optional[float],
        result: TeleviewerEnrichmentResult,
    ) -> None:
        if not self.data_manager:
            return
        try:
            collar_df = self.data_manager.get_collar_data()
            if _is_dataframe(collar_df) and not collar_df.empty:
                collar_rows = _filter_hole_rows(collar_df, hole_id)
                if not collar_rows.empty:
                    payload["collar"] = _records(collar_rows)
                    result.collar_rows = len(collar_rows)
        except Exception as exc:
            result.warnings.append(f"Could not load collar data: {exc}")

        try:
            survey_df = self.data_manager.get_survey_data()
            if _is_dataframe(survey_df) and not survey_df.empty:
                survey_rows = _filter_hole_rows(survey_df, hole_id)
                if not survey_rows.empty:
                    payload["survey"] = _records(survey_rows)
                    result.survey_rows = len(survey_rows)
        except Exception as exc:
            result.warnings.append(f"Could not load survey data: {exc}")

        try:
            if hasattr(self.data_manager, "get_trace_for_hole"):
                trace = self.data_manager.get_trace_for_hole(hole_id, max_depth=max_depth)
                if trace:
                    payload["trace"] = [
                        {"depth_m": float(depth), "x": float(x), "y": float(y), "z": float(z)}
                        for depth, x, y, z in trace
                    ]
                    result.trace_rows = len(trace)
        except Exception as exc:
            result.warnings.append(f"Could not build trace from collar/survey: {exc}")

    def _merge_source_layers(
        self,
        payload: Dict[str, Any],
        hole_id: str,
        result: TeleviewerEnrichmentResult,
    ) -> None:
        if not self.data_manager or not hasattr(self.data_manager, "_geological_store"):
            return
        store = getattr(self.data_manager, "_geological_store", None)
        if not store or not hasattr(store, "list_sources"):
            return

        for source_name in store.list_sources():
            source_lower = str(source_name).lower()
            try:
                df = self._source_rows(source_name, hole_id)
            except Exception as exc:
                result.warnings.append(f"Could not read source {source_name}: {exc}")
                continue
            if not _is_dataframe(df) or df.empty:
                continue

            if source_lower in POINT_SOURCE_SPECS:
                rows = self._normalise_point_rows(df, source_name)
                payload[POINT_SOURCE_SPECS[source_lower]["target"]] = rows
                result.geophysics_rows += len(rows)
            else:
                target = _target_for_source(source_lower)
                if not target:
                    continue
                rows = self._normalise_interval_rows(df, target)
                if target == "assays":
                    existing = payload.get("assays") or []
                    payload["assays"] = _prefer_rows(existing, rows)
                    result.assay_rows = len(payload["assays"])
                elif target == "mineralogy":
                    payload["mineralogy"] = rows
                    result.mineralogy_rows = len(rows)

    def _source_rows(self, source_name: str, hole_id: str):
        if hasattr(self.data_manager, "_get_source_rows_for_hole"):
            return self.data_manager._get_source_rows_for_hole(source_name, hole_id)
        store = getattr(self.data_manager, "_geological_store", None)
        if store and hasattr(store, "get_rows_for_hole"):
            all_rows = store.get_rows_for_hole(hole_id)
            if isinstance(all_rows, dict):
                for name, df in all_rows.items():
                    if str(name).lower() == str(source_name).lower():
                        return df
        return _empty_df()

    def _normalise_point_rows(self, df, source_name: str) -> List[Dict[str, Any]]:
        depth_col = self._resolve_column(source_name, df, POINT_SOURCE_SPECS[source_name.lower()]["depth_aliases"])
        if not depth_col:
            return []
        rows = []
        for row in _records(df):
            out = dict(row)
            out["DEPTH"] = _number(row.get(depth_col))
            rows.append(out)
        return sorted(rows, key=lambda item: _number(item.get("DEPTH")) or 0.0)

    def _normalise_interval_rows(self, df, target: str) -> List[Dict[str, Any]]:
        from_col = _resolve_column_from_df(df, ["sampfrom", "geolfrom", "depth_from", "from", "interval_from", "top"])
        to_col = _resolve_column_from_df(df, ["sampto", "geolto", "depth_to", "to", "interval_to", "bottom"])
        rows = []
        for row in _records(df):
            out = dict(row)
            if from_col:
                out["SAMPFROM"] = _number(row.get(from_col))
            if to_col:
                out["SAMPTO"] = _number(row.get(to_col))
            rows.append(out)
        return rows

    def _resolve_column(self, source_name: str, df, aliases: Sequence[str]) -> Optional[str]:
        if self.data_manager and hasattr(self.data_manager, "_resolve_source_column_name"):
            found = self.data_manager._resolve_source_column_name(source_name, list(aliases))
            if found:
                return found
        return _resolve_column_from_df(df, aliases)

    def build_chip_tray_manifest(self, hole_id: str, output_path: Path) -> List[Dict[str, Any]]:
        roots = self._chip_roots()
        rows: List[Dict[str, Any]] = []
        for root in roots:
            if not root or not root.exists():
                continue
            for image_path in root.rglob("*"):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                parsed = parse_chip_image_path(image_path, hole_id)
                if parsed:
                    rows.append(parsed)

        rows = sorted(_dedupe_chip_rows(rows), key=lambda row: (row["from_m"], row["to_m"], row["url"]))
        if rows:
            _write_json(output_path, {"hole_id": hole_id, "images": rows})
        return rows

    def _chip_roots(self) -> List[Path]:
        roots: List[Path] = []
        if self.file_manager is not None:
            for key in ("approved_compartments", "chip_compartments"):
                try:
                    path = self.file_manager.get_shared_path(key, create_if_missing=False)
                except Exception:
                    path = None
                if path:
                    roots.append(Path(path))
            for key in ("approved_compartments", "chip_compartments"):
                path = getattr(self.file_manager, "dir_structure", {}).get(key)
                if path:
                    roots.append(Path(path))
        # Preserve user's known project structure as a fallback when configured paths are absent.
        known = Path(
            r"C:\Users\gsymonds\Fortescue Metals Group\Gabon - Belinga - Exploration Drilling\03 - Reverse Circulation\Chip Tray Photos\Extracted Compartment Images\Approved Compartment Images"
        )
        roots.append(known)
        return _unique_existing_or_candidate_paths(roots)


def parse_chip_image_path(image_path: Path, hole_id: str) -> Optional[Dict[str, Any]]:
    hole_upper = str(hole_id).strip().upper()
    name = image_path.name
    if FilenameParser is not None:
        try:
            parsed = FilenameParser().parse_compartment_filename(name)
        except Exception:
            parsed = None
        if parsed and str(parsed.get("hole_id", "")).upper() == hole_upper:
            to_m = float(parsed["depth_to"])
            return _chip_row(image_path, to_m - 1.0, to_m)

    for pattern in CHIP_PATTERNS:
        match = pattern.search(name)
        if not match:
            continue
        if match.group("hole").upper() != hole_upper:
            continue
        if "from" in match.groupdict() and match.groupdict().get("from") is not None:
            from_m = float(match.group("from"))
            to_m = float(match.group("to"))
        else:
            to_m = float(match.group("to"))
            from_m = to_m - 1.0
        return _chip_row(image_path, from_m, to_m)
    return None


def _chip_row(image_path: Path, from_m: float, to_m: float) -> Dict[str, Any]:
    return {
        "from_m": float(from_m),
        "to_m": float(to_m),
        "url": image_path.resolve().as_uri(),
        "label": f"{from_m:g}-{to_m:g} m",
        "path": str(image_path),
    }


def _dedupe_chip_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for row in rows:
        key = (round(float(row["from_m"]), 3), round(float(row["to_m"]), 3), row["url"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _target_for_source(source_lower: str) -> Optional[str]:
    for token, target in INTERVAL_SOURCE_TARGETS.items():
        if token in source_lower:
            return target
    return None


def _prefer_rows(existing: Sequence[Dict[str, Any]], candidate: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return list(candidate) if candidate else list(existing)


def _filter_hole_rows(df, hole_id: str):
    hole_col = _resolve_column_from_df(df, ["holeid", "hole_id", "bhid", "drillhole_id", "dhid", "hole"])
    if not hole_col:
        return _empty_df()
    return df[df[hole_col].astype(str).str.strip().str.upper() == hole_id].copy()


def _records(df) -> List[Dict[str, Any]]:
    if not _is_dataframe(df) or df.empty:
        return []
    clean = df.copy()
    for col in clean.columns:
        if hasattr(clean[col], "where"):
            clean[col] = clean[col].where(clean[col].notna(), None)
    return clean.to_dict(orient="records")


def _resolve_column_from_df(df, aliases: Sequence[str]) -> Optional[str]:
    if not _is_dataframe(df):
        return None
    lookup = {str(col).lower().strip(): col for col in df.columns}
    for alias in aliases:
        found = lookup.get(str(alias).lower().strip())
        if found is not None:
            return found
    return None


def _is_dataframe(value) -> bool:
    return pd is not None and isinstance(value, pd.DataFrame)


def _empty_df():
    if pd is None:
        return None
    return pd.DataFrame()


def _number(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if pd is not None and pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _payload_max_depth(payload: Dict[str, Any]) -> Optional[float]:
    stats = payload.get("stats") or {}
    for key in ("maxDepthMeter", "max_depth_meter", "max_row_depth_m"):
        value = _number(stats.get(key))
        if value is not None:
            return value
    return None


def _load_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    return path


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _unique_existing_or_candidate_paths(paths: Iterable[Path]) -> List[Path]:
    unique = []
    seen = set()
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique
