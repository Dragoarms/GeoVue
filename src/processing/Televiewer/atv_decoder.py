"""Vendor-free extraction for ABI40 acoustic televiewer TFD files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import csv
import json
import math
from pathlib import Path
import statistics
import struct
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .manifest import TeleviewerManifest, write_manifest
from .paths import TeleviewerImportPlan
from .tfd_decoder import DEFAULT_PIXELS_PER_METER, NO_DATA_RGB, TfdDecodeResult

ATV_PROCESSING_VERSION = "geovue-televiewer-atv-prototype-1"
ATV_WIDTH_PX = 180
ATV_AMPLITUDE_OFFSET = 176
ATV_TRAVELTIME_OFFSET = 536
ATV_AUXILIARY_OFFSET = 896


@dataclass
class AtvImageRecord:
    """One depth-indexed ATV scanline."""

    index: int
    linked_index: int
    record_start: int
    record_length: int
    timestamp: Optional[str]
    depth_m: float
    amplitude_row_index: Optional[int] = None
    traveltime_row_index: Optional[int] = None
    auxiliary_row_index: Optional[int] = None

    def to_csv_row(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "linked_index": self.linked_index,
            "record_start": self.record_start,
            "record_length": self.record_length,
            "timestamp": self.timestamp or "",
            "depth_m": self.depth_m,
            "amplitude_row_index": self.amplitude_row_index,
            "traveltime_row_index": self.traveltime_row_index,
            "auxiliary_row_index": self.auxiliary_row_index,
        }

    def to_viewer_row(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "depth_m": self.depth_m,
            "timestamp": self.timestamp,
            "valid": True,
        }


class AtvDecoder:
    """Decode ABI40 ATV numeric image records into GeoVue viewer slices."""

    def __init__(self, pixels_per_meter: int = DEFAULT_PIXELS_PER_METER):
        self.pixels_per_meter = int(pixels_per_meter)

    @staticmethod
    def looks_like_atv(tfd_path: Path) -> bool:
        try:
            sample = Path(tfd_path).read_bytes()[:250_000]
        except OSError:
            return False
        return b"ToolName = ATV" in sample and b"ABI40" in sample

    def decode_to_dataset(self, plan: TeleviewerImportPlan) -> TfdDecodeResult:
        plan.paths.ensure()
        records, amplitude, depths = self.read_atv_records(plan.source_path)
        if not records:
            raise ValueError(f"No valid ABI40 ATV image records found in {plan.source_path}")

        row_pitch_m = self._estimate_row_pitch(depths)
        first_meter = math.floor(float(depths[0]))
        max_depth_meter = math.ceil(float(depths[-1]))
        records_csv = plan.paths.processed_dir / "records.csv"
        segment_manifest_csv = plan.paths.processed_dir / "meter_segments_manifest.csv"
        summary_json = plan.paths.processed_dir / "extraction_summary.json"
        stretched_zone_report_csv = plan.paths.qc_dir / "stretched_zone_report.csv"
        viewer_data_json = plan.paths.viewer_data_path

        self._write_records_csv(records_csv, records)
        segment_rows = self._write_meter_slices(
            plan,
            amplitude,
            depths,
            first_meter,
            max_depth_meter,
        )
        self._write_segment_manifest(segment_manifest_csv, segment_rows)
        stretch_rows = self._write_stretched_zone_report(
            stretched_zone_report_csv,
            depths,
            row_pitch_m,
        )
        summary = self._write_summary(
            summary_json,
            plan,
            records,
            depths,
            first_meter,
            max_depth_meter,
            row_pitch_m,
            segment_rows,
            stretch_rows,
        )
        self._write_viewer_data(
            viewer_data_json,
            plan,
            records,
            first_meter,
            max_depth_meter,
            row_pitch_m,
            summary,
        )
        manifest_path = self._write_manifest(
            plan,
            records,
            first_meter,
            max_depth_meter,
            row_pitch_m,
            summary,
        )
        return TfdDecodeResult(
            plan=plan,
            records=records,  # type: ignore[arg-type]
            first_meter=first_meter,
            max_depth_meter=max_depth_meter,
            row_pitch_m=row_pitch_m,
            pixels_per_meter=self.pixels_per_meter,
            records_csv=records_csv,
            segment_manifest_csv=segment_manifest_csv,
            summary_json=summary_json,
            stretched_zone_report_csv=stretched_zone_report_csv,
            viewer_data_json=viewer_data_json,
            manifest_json=manifest_path,
        )

    def read_atv_records(self, tfd_path: Path) -> Tuple[List[AtvImageRecord], np.ndarray, np.ndarray]:
        blob = Path(tfd_path).read_bytes()
        rows: List[Tuple[AtvImageRecord, bytes]] = []
        for linked_index, (record_start, record_length, _previous, _next, record) in enumerate(_iter_linked_records(blob)):
            depth = _read_float64(record, 26)
            timestamp = _read_timestamp(record)
            if not timestamp or not math.isfinite(depth) or not (1.0 < depth < 500.0):
                continue
            rows.append(
                (
                    AtvImageRecord(
                        index=len(rows),
                        linked_index=linked_index,
                        record_start=record_start,
                        record_length=record_length,
                        timestamp=timestamp,
                        depth_m=float(depth),
                    ),
                    record,
                )
            )
        rows.sort(key=lambda item: item[0].depth_m)
        amplitude, amp_indices = _extract_channel(rows, ATV_AMPLITUDE_OFFSET)
        traveltime, tt_indices = _extract_channel(rows, ATV_TRAVELTIME_OFFSET)
        auxiliary, aux_indices = _extract_channel(rows, ATV_AUXILIARY_OFFSET)
        for row_index, (record, _raw) in enumerate(rows):
            record.amplitude_row_index = amp_indices.get(row_index)
            record.traveltime_row_index = tt_indices.get(row_index)
            record.auxiliary_row_index = aux_indices.get(row_index)
        depths = np.asarray([rows[row_index][0].depth_m for row_index in sorted(amp_indices)], dtype=float)
        amplitude = amplitude[np.argsort(depths)]
        depths = np.sort(depths)
        self._write_channel_previews(rows, amplitude, traveltime, auxiliary)
        return [record for record, _raw in rows], amplitude, depths

    def _write_channel_previews(
        self,
        rows: Sequence[Tuple[AtvImageRecord, bytes]],
        amplitude: np.ndarray,
        traveltime: np.ndarray,
        auxiliary: np.ndarray,
    ) -> None:
        # Preview writing is handled during dataset decode when paths are available.
        return None

    def _write_meter_slices(
        self,
        plan: TeleviewerImportPlan,
        amplitude: np.ndarray,
        depths: np.ndarray,
        first_meter: int,
        max_depth_meter: int,
    ) -> List[Dict[str, Any]]:
        normalized, _lo, _hi = _robust_uint8(amplitude)
        segment_rows: List[Dict[str, Any]] = []
        no_data_row = np.array(NO_DATA_RGB, dtype=np.uint8)
        for meter_start in range(first_meter, max_depth_meter):
            meter_end = meter_start + 1
            target_depths = meter_start + (np.arange(self.pixels_per_meter) + 0.5) / self.pixels_per_meter
            rows = np.empty((self.pixels_per_meter, normalized.shape[1], 3), dtype=np.uint8)
            rows[:, :, :] = no_data_row
            indices = np.searchsorted(depths, target_depths)
            for target_index, insert_index in enumerate(indices):
                candidates = []
                if insert_index < len(depths):
                    candidates.append(insert_index)
                if insert_index > 0:
                    candidates.append(insert_index - 1)
                if not candidates:
                    continue
                nearest = min(candidates, key=lambda idx: abs(depths[idx] - target_depths[target_index]))
                if abs(depths[nearest] - target_depths[target_index]) <= 0.006:
                    rows[target_index] = np.repeat(normalized[nearest][:, None], 3, axis=1)

            resampled_path = plan.paths.slices_1m_dir / f"{plan.hole_id}_{meter_start:03d}_{meter_end:03d}m_resampled.jpg"
            Image.fromarray(rows).save(resampled_path, quality=95)
            covered = target_depths[(target_depths >= depths[0]) & (target_depths <= depths[-1])]
            segment_rows.append(
                {
                    "meter_start_m": meter_start,
                    "meter_end_m": meter_end,
                    "file": str(resampled_path),
                    "height_px": self.pixels_per_meter,
                    "mode": "atv_amplitude_depth_resampled",
                    "covered_depth_start_m": float(covered[0]) if len(covered) else "",
                    "covered_depth_end_m": float(covered[-1]) if len(covered) else "",
                    "covered_rows_px": int(len(covered)),
                }
            )
        return segment_rows

    @staticmethod
    def _write_records_csv(path: Path, records: Sequence[AtvImageRecord]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].to_csv_row().keys()))
            writer.writeheader()
            for record in records:
                writer.writerow(record.to_csv_row())

    @staticmethod
    def _write_segment_manifest(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_stretched_zone_report(
        path: Path,
        depths: np.ndarray,
        row_pitch_m: float,
    ) -> List[Dict[str, Any]]:
        deltas = np.diff(depths)
        threshold = row_pitch_m * 0.5
        zones: List[Dict[str, Any]] = []
        current_start: Optional[int] = None
        for index, delta in enumerate(deltas, start=1):
            if delta < threshold:
                if current_start is None:
                    current_start = index
            elif current_start is not None:
                zones.append(_zone_row(current_start, index - 1, depths, threshold))
                current_start = None
        if current_start is not None:
            zones.append(_zone_row(current_start, len(depths) - 1, depths, threshold))

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = ["start_index", "end_index", "start_depth_m", "end_depth_m", "record_count", "threshold_m"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(zones)
        return zones

    def _write_summary(
        self,
        path: Path,
        plan: TeleviewerImportPlan,
        records: Sequence[AtvImageRecord],
        depths: np.ndarray,
        first_meter: int,
        max_depth_meter: int,
        row_pitch_m: float,
        segment_rows: Sequence[Dict[str, Any]],
        stretch_rows: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        summary = {
            "source": str(plan.source_path),
            "file_size_bytes": plan.source_path.stat().st_size if plan.source_path.exists() else None,
            "image_type": "ATV ABI40 amplitude candidate",
            "record_count": len(records),
            "texture_width_px": ATV_WIDTH_PX,
            "depth_interval_m_per_record": row_pitch_m,
            "row_pitch_m": row_pitch_m,
            "pixels_per_meter_depth_resampled": self.pixels_per_meter,
            "first_depth_m": float(depths[0]),
            "last_depth_m": float(depths[-1]),
            "min_row_depth_m": float(depths[0]),
            "max_row_depth_m": float(depths[-1]),
            "first_meter": first_meter,
            "max_depth_meter": max_depth_meter,
            "first_timestamp": records[0].timestamp,
            "last_timestamp": records[-1].timestamp,
            "stall_zone_count": len(stretch_rows),
            "depth_resampled_segment_count": sum(
                1 for row in segment_rows if row["mode"] == "atv_amplitude_depth_resampled"
            ),
            "amplitude_offset": ATV_AMPLITUDE_OFFSET,
            "amplitude_byte_plane": "even",
            "traveltime_candidate_offset": ATV_TRAVELTIME_OFFSET,
            "auxiliary_candidate_offset": ATV_AUXILIARY_OFFSET,
            "processing_version": ATV_PROCESSING_VERSION,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return summary

    @staticmethod
    def _write_viewer_data(
        path: Path,
        plan: TeleviewerImportPlan,
        records: Sequence[AtvImageRecord],
        first_meter: int,
        max_depth_meter: int,
        row_pitch_m: float,
        summary: Dict[str, Any],
    ) -> None:
        trace = [
            {"depth_m": float(depth), "x": 0.0, "y": 0.0, "z": -float(depth)}
            for depth in range(first_meter, max_depth_meter + 1)
        ]
        payload = {
            "holeId": plan.hole_id,
            "collar": [
                {
                    "HOLEID": plan.hole_id,
                    "PROJECTCODE": plan.project_code,
                    "DEPTH": float(max_depth_meter),
                    "BEST_X": 0.0,
                    "BEST_Y": 0.0,
                    "BEST_Z": 0.0,
                }
            ],
            "survey": [],
            "trace": trace,
            "geophysics": [],
            "mineralogy": [],
            "assays": [],
            "tfdAps544": [record.to_viewer_row() for record in records],
            "comparison": {},
            "stats": {
                **summary,
                "firstMeter": first_meter,
                "maxDepthMeter": max_depth_meter,
                "rowPitchM": row_pitch_m,
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    @staticmethod
    def _write_manifest(
        plan: TeleviewerImportPlan,
        records: Sequence[AtvImageRecord],
        first_meter: int,
        max_depth_meter: int,
        row_pitch_m: float,
        summary: Dict[str, Any],
    ) -> Path:
        manifest = TeleviewerManifest(
            project_code=plan.project_code,
            hole_id=plan.hole_id,
            processing_version=ATV_PROCESSING_VERSION,
            raw_files=[
                {
                    "name": plan.source_path.name,
                    "source_path": str(plan.source_path),
                    "size_bytes": plan.source_path.stat().st_size if plan.source_path.exists() else None,
                    "source_kind": "atv",
                }
            ],
            coverage={
                "first_meter": first_meter,
                "max_depth_meter": max_depth_meter,
                "first_depth_m": summary["first_depth_m"],
                "max_row_depth_m": summary["max_row_depth_m"],
                "row_pitch_m": row_pitch_m,
                "pixels_per_meter": summary["pixels_per_meter_depth_resampled"],
            },
            viewer={
                "hole_id": plan.hole_id,
                "data_url": "viewer_data.json",
                "chip_tray_manifest_url": "chip_tray_manifest.json",
                "raw_dir": "slices_1m",
                "resampled_dir": "slices_1m",
                "first_meter": first_meter,
                "max_depth_meter": max_depth_meter,
            },
            telemetry={
                "record_count": len(records),
                "fields": [
                    "ATV Amplitude candidate",
                    "ATV TravelTime candidate",
                    "ATV auxiliary candidate",
                ],
            },
            qc={
                "stretched_zone_report": "../qc/stretched_zone_report.csv",
                "extraction_summary": "extraction_summary.json",
                "records_csv": "records.csv",
                "meter_segments_manifest": "meter_segments_manifest.csv",
            },
            notes=[
                "ATV ABI40 extraction is a vendor-free prototype. "
                "Primary texture uses even byte plane at record offset 176, width 180."
            ],
        )
        return write_manifest(plan.paths.manifest_path, manifest)

    @staticmethod
    def _estimate_row_pitch(depths: np.ndarray) -> float:
        deltas = [delta for delta in np.diff(depths) if delta > 0]
        return float(statistics.median(deltas)) if deltas else 0.004


def _extract_channel(
    rows: Sequence[Tuple[AtvImageRecord, bytes]],
    start: int,
    width: int = ATV_WIDTH_PX,
) -> Tuple[np.ndarray, Dict[int, int]]:
    channel_rows: List[np.ndarray] = []
    index_map: Dict[int, int] = {}
    for row_index, (_record, raw) in enumerate(rows):
        end = start + width * 2
        if len(raw) < end:
            continue
        raw_bytes = np.frombuffer(raw[start:end], dtype=np.uint8)
        channel_rows.append(raw_bytes[::2].astype(np.float32))
        index_map[row_index] = len(channel_rows) - 1
    if not channel_rows:
        return np.empty((0, width), dtype=np.float32), {}
    return np.vstack(channel_rows), index_map


def _robust_uint8(array: np.ndarray) -> Tuple[np.ndarray, float, float]:
    lo, hi = np.nanpercentile(array, [1, 99])
    if hi <= lo:
        hi = lo + 1
    return np.clip((array - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8), float(lo), float(hi)


def _iter_linked_records(blob: bytes) -> List[Tuple[int, int, int, int, bytes]]:
    records: List[Tuple[int, int, int, int, bytes]] = []
    seen = set()
    offset = 0
    while 0 <= offset < len(blob) and offset not in seen:
        seen.add(offset)
        if offset + 12 > len(blob):
            break
        record_length, previous_pointer, next_pointer = struct.unpack_from("<III", blob, offset)
        if record_length <= 0 or offset + record_length > len(blob):
            break
        records.append(
            (
                offset,
                record_length,
                previous_pointer,
                next_pointer,
                blob[offset:offset + record_length],
            )
        )
        if next_pointer == 0 or next_pointer == offset or next_pointer > len(blob):
            break
        offset = next_pointer
    return records


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
    try:
        datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None
    return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}.{millisecond:03d}"


def _zone_row(start_index: int, end_index: int, depths: np.ndarray, threshold: float) -> Dict[str, Any]:
    return {
        "start_index": start_index,
        "end_index": end_index,
        "start_depth_m": float(depths[start_index]),
        "end_depth_m": float(depths[end_index]),
        "record_count": end_index - start_index + 1,
        "threshold_m": threshold,
    }
