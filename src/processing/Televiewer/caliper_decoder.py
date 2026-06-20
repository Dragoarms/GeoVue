"""Vendor-free extraction for mechanical caliper TFD files."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
import re
import statistics
import struct
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .paths import TeleviewerDatasetPaths

DEFAULT_TOOL_OD_MM = 39.0
DEFAULT_CALIPER_POINTS = [
    (122.0, 10.2),
    (231.95, 15.2),
    (356.0, 20.3),
    (456.85, 25.4),
    (589.0, 30.5),
]
CALIPER_WORD_OFFSET = 204


@dataclass
class CaliperDiameterRow:
    """One depth-indexed mechanical caliper sample."""

    depth_m: float
    timestamp: str
    raw_caliper_word: int
    caliper_arm_mm: float
    borehole_diameter_mm: float
    tool_od_mm: float
    source_record_start: int

    def to_csv_row(self) -> Dict[str, Any]:
        return {
            "depth_m": self.depth_m,
            "timestamp": self.timestamp,
            "raw_caliper_word": self.raw_caliper_word,
            "caliper_arm_mm": self.caliper_arm_mm,
            "borehole_diameter_mm": self.borehole_diameter_mm,
            "tool_od_mm": self.tool_od_mm,
            "source_record_start": self.source_record_start,
        }

    def to_viewer_row(self) -> Dict[str, Any]:
        return {
            "DEPTH": self.depth_m,
            "CALIPER_MM": self.borehole_diameter_mm,
            "CALIPER_ARM_MM": self.caliper_arm_mm,
            "SOURCE": "TFD_CALIPER",
        }


class CaliperDecoder:
    """Extract borehole diameter from a mechanical caliper TFD."""

    def __init__(self, tool_od_mm: float = DEFAULT_TOOL_OD_MM):
        self.tool_od_mm = float(tool_od_mm)

    def decode_to_dataset(
        self,
        source_path: Path,
        paths: TeleviewerDatasetPaths,
    ) -> List[CaliperDiameterRow]:
        rows = self.read_diameter_rows(source_path)
        if not rows:
            raise ValueError(f"No depth-indexed caliper rows found in {source_path}")
        self.write_csv(paths.processed_dir / "caliper_diameter.csv", rows)
        self.update_viewer_data(paths, rows)
        self.update_manifest(paths, rows)
        return rows

    def read_diameter_rows(self, source_path: Path) -> List[CaliperDiameterRow]:
        blob = Path(source_path).read_bytes()
        cal_points = _extract_caliper_points(blob) or DEFAULT_CALIPER_POINTS
        rows: List[CaliperDiameterRow] = []
        for record_start, record_length, _previous, _next, record in _iter_linked_records(blob):
            if record_length != 474 or len(record) < CALIPER_WORD_OFFSET + 2:
                continue
            depth = _read_float64(record, 26)
            timestamp = _read_timestamp(record)
            if not timestamp or not math.isfinite(depth) or depth <= 1.0:
                continue
            raw_value = struct.unpack_from("<H", record, CALIPER_WORD_OFFSET)[0]
            arm_mm = _multilin(raw_value, cal_points)
            diameter_mm = self.tool_od_mm + 2.0 * arm_mm
            rows.append(
                CaliperDiameterRow(
                    depth_m=float(depth),
                    timestamp=timestamp,
                    raw_caliper_word=int(raw_value),
                    caliper_arm_mm=float(arm_mm),
                    borehole_diameter_mm=float(diameter_mm),
                    tool_od_mm=self.tool_od_mm,
                    source_record_start=record_start,
                )
            )
        return sorted(rows, key=lambda row: row.depth_m)

    @staticmethod
    def write_csv(path: Path, rows: Sequence[CaliperDiameterRow]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = list(rows[0].to_csv_row().keys())
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row.to_csv_row())

    @staticmethod
    def update_viewer_data(
        paths: TeleviewerDatasetPaths,
        rows: Sequence[CaliperDiameterRow],
    ) -> None:
        viewer_path = paths.viewer_data_path
        if not viewer_path.exists():
            return
        try:
            payload = json.loads(viewer_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        existing = [
            row
            for row in payload.get("geophysics", [])
            if row.get("SOURCE") != "TFD_CALIPER"
        ]
        existing.extend(row.to_viewer_row() for row in rows)
        payload["geophysics"] = existing
        stats = payload.setdefault("stats", {})
        stats["caliperRows"] = len(rows)
        stats["caliperDiameterMmMin"] = min(row.borehole_diameter_mm for row in rows)
        stats["caliperDiameterMmMedian"] = statistics.median(
            row.borehole_diameter_mm for row in rows
        )
        stats["caliperDiameterMmMax"] = max(row.borehole_diameter_mm for row in rows)
        with viewer_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    @staticmethod
    def update_manifest(
        paths: TeleviewerDatasetPaths,
        rows: Sequence[CaliperDiameterRow],
    ) -> None:
        manifest_path = paths.manifest_path
        if not manifest_path.exists():
            return
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        qc = payload.setdefault("qc", {})
        qc["caliper_diameter_csv"] = "caliper_diameter.csv"
        telemetry = payload.setdefault("telemetry", {})
        telemetry["caliper_rows"] = len(rows)
        telemetry["caliper_diameter_mm_min"] = min(row.borehole_diameter_mm for row in rows)
        telemetry["caliper_diameter_mm_median"] = statistics.median(
            row.borehole_diameter_mm for row in rows
        )
        telemetry["caliper_diameter_mm_max"] = max(row.borehole_diameter_mm for row in rows)
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")


def _extract_caliper_points(blob: bytes) -> Optional[List[Tuple[float, float]]]:
    points: List[Tuple[float, float]] = []
    text = "".join(chr(byte) if byte in (9, 10, 13) or 32 <= byte < 127 else " " for byte in blob[:250_000])
    for raw, mm in re.findall(r"CalPoint\d+\s*=\s*([0-9.]+)\s*;\s*([0-9.]+)", text):
        points.append((float(raw), float(mm)))
    return sorted(points) if len(points) >= 2 else None


def _multilin(raw_value: float, points: Sequence[Tuple[float, float]]) -> float:
    if raw_value <= points[0][0]:
        x0, y0 = points[0]
        x1, y1 = points[1]
    elif raw_value >= points[-1][0]:
        x0, y0 = points[-2]
        x1, y1 = points[-1]
    else:
        x0, y0 = points[0]
        x1, y1 = points[1]
        for (left_x, left_y), (right_x, right_y) in zip(points, points[1:]):
            if left_x <= raw_value <= right_x:
                x0, y0 = left_x, left_y
                x1, y1 = right_x, right_y
                break
    return y0 + (raw_value - x0) * (y1 - y0) / (x1 - x0)


def _iter_linked_records(blob: bytes) -> Iterable[Tuple[int, int, int, int, bytes]]:
    seen = set()
    offset = 0
    while 0 <= offset < len(blob) and offset not in seen:
        seen.add(offset)
        if offset + 12 > len(blob):
            break
        record_length, previous_pointer, next_pointer = struct.unpack_from("<III", blob, offset)
        if record_length <= 0 or offset + record_length > len(blob):
            break
        record = blob[offset:offset + record_length]
        yield offset, record_length, previous_pointer, next_pointer, record
        if next_pointer == 0 or next_pointer == offset or next_pointer > len(blob):
            break
        offset = next_pointer


def _read_float64(record: bytes, offset: int) -> float:
    if offset + 8 > len(record):
        return float("nan")
    return struct.unpack_from("<d", record, offset)[0]


def _read_timestamp(record: bytes) -> Optional[str]:
    if 58 > len(record):
        return None
    year, month, _weekday, day, hour, minute, second, millisecond = struct.unpack_from("<8H", record, 42)
    if not (
        1900 <= year <= 2200
        and 1 <= month <= 12
        and 1 <= day <= 31
        and 0 <= hour < 24
        and 0 <= minute < 60
        and 0 <= second < 60
    ):
        return None
    return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}.{millisecond:03d}"
