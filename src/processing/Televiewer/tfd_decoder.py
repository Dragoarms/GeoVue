"""Vendor-free extraction for optical televiewer TFD files.

The BA0007 OTV files use a linked-record binary structure. Image records contain
small JPEG strips plus downhole metadata. This decoder keeps the parsing rules
explicit and conservative: records without a valid 900x8 JPEG strip are retained
only for traversal and are not exported as image strips.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
import csv
import json
import math
from pathlib import Path
import statistics
import struct
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

from .manifest import TeleviewerManifest, PROCESSING_VERSION, write_manifest
from .paths import TeleviewerImportPlan

NO_DATA_RGB = (225, 225, 225)
DEFAULT_PIXELS_PER_METER = 500
EXPECTED_STRIP_SIZE = (900, 8)


@dataclass
class TfdImageRecord:
    """One decoded JPEG strip record from a TFD file."""

    index: int
    linked_index: int
    record_start: int
    record_length: int
    previous_pointer: int
    next_pointer: int
    timestamp: Optional[str]
    depth_m: float
    depth_duplicate_m: float
    jpeg_offset: int
    jpeg_length: int
    width_px: int
    height_px: int
    channel_payload_bytes: int
    roll_deg: Optional[float] = None
    mroll_deg: Optional[float] = None
    tilt_deg: Optional[float] = None
    magfield_uT: Optional[float] = None
    azimuth_deg: Optional[float] = None
    gravity_g: Optional[float] = None
    taps_c: Optional[float] = None
    depth_delta_from_prev_m: Optional[float] = None
    time_delta_from_prev_s: Optional[float] = None
    speed_from_prev_m_per_min: Optional[float] = None
    jpeg_bytes: bytes = field(default=b"", repr=False)

    def row_start_depth(self) -> float:
        return self.depth_m

    def row_end_depth(self, row_pitch_m: float) -> float:
        return self.depth_m + self.height_px * row_pitch_m

    def to_csv_row(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "linked_index": self.linked_index,
            "record_start": self.record_start,
            "record_length": self.record_length,
            "previous_pointer": self.previous_pointer,
            "next_pointer": self.next_pointer,
            "timestamp": self.timestamp or "",
            "depth_m": self.depth_m,
            "depth_duplicate_m": self.depth_duplicate_m,
            "jpeg_offset": self.jpeg_offset,
            "jpeg_length": self.jpeg_length,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "channel_payload_bytes": self.channel_payload_bytes,
            "depth_delta_from_prev_m": self.depth_delta_from_prev_m,
            "time_delta_from_prev_s": self.time_delta_from_prev_s,
            "speed_from_prev_m_per_min": self.speed_from_prev_m_per_min,
            "roll_deg": self.roll_deg,
            "mroll_deg": self.mroll_deg,
            "tilt_deg": self.tilt_deg,
            "magfield_uT": self.magfield_uT,
            "azimuth_deg": self.azimuth_deg,
            "gravity_g": self.gravity_g,
            "taps_c": self.taps_c,
        }

    def to_viewer_row(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "depth_m": self.depth_m,
            "timestamp": self.timestamp,
            "roll_deg": self.roll_deg,
            "mroll_deg": self.mroll_deg,
            "tilt_deg": self.tilt_deg,
            "magfield_uT": self.magfield_uT,
            "azimuth_deg": self.azimuth_deg,
            "gravity_g": self.gravity_g,
            "taps_c": self.taps_c,
            "valid": True,
        }


@dataclass
class TfdDecodeResult:
    """Output paths and summary for a decoded TFD dataset."""

    plan: TeleviewerImportPlan
    records: List[TfdImageRecord]
    first_meter: int
    max_depth_meter: int
    row_pitch_m: float
    pixels_per_meter: int
    records_csv: Path
    segment_manifest_csv: Path
    summary_json: Path
    stretched_zone_report_csv: Path
    viewer_data_json: Path
    manifest_json: Path


class TfdDecoder:
    """Decode TFD image records into GeoVue's televiewer dataset layout."""

    def __init__(
        self,
        pixels_per_meter: int = DEFAULT_PIXELS_PER_METER,
        expected_strip_size: Tuple[int, int] = EXPECTED_STRIP_SIZE,
    ):
        self.pixels_per_meter = int(pixels_per_meter)
        self.expected_strip_size = expected_strip_size

    def decode_to_dataset(self, plan: TeleviewerImportPlan) -> TfdDecodeResult:
        """Decode one TFD file into its planned GeoVue dataset folder."""
        plan.paths.ensure()
        records = self.read_image_records(plan.source_path)
        if not records:
            raise ValueError(f"No valid OTV JPEG strip records found in {plan.source_path}")

        row_pitch_m = self._estimate_row_pitch(records)
        first_row_depth = min(record.depth_m for record in records)
        max_row_depth = max(record.row_end_depth(row_pitch_m) for record in records)
        first_meter = math.floor(first_row_depth)
        max_depth_meter = math.ceil(max_row_depth)

        records_csv = plan.paths.processed_dir / "records.csv"
        segment_manifest_csv = plan.paths.processed_dir / "meter_segments_manifest.csv"
        summary_json = plan.paths.processed_dir / "extraction_summary.json"
        stretched_zone_report_csv = plan.paths.qc_dir / "stretched_zone_report.csv"
        viewer_data_json = plan.paths.viewer_data_path

        self._write_records_csv(records_csv, records)
        segment_rows = self._write_meter_slices(
            plan,
            records,
            first_meter,
            max_depth_meter,
            row_pitch_m,
        )
        self._write_segment_manifest(segment_manifest_csv, segment_rows)
        stretch_rows = self._write_stretched_zone_report(
            stretched_zone_report_csv,
            records,
            row_pitch_m,
        )
        summary = self._write_summary(
            summary_json,
            plan,
            records,
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
            records=records,
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

    def read_image_records(self, tfd_path: Path) -> List[TfdImageRecord]:
        blob = Path(tfd_path).read_bytes()
        linked_records = self._iter_linked_records(blob)
        image_records: List[TfdImageRecord] = []
        previous_image: Optional[TfdImageRecord] = None

        for linked_index, (record_start, record_length, previous_pointer, next_pointer) in enumerate(linked_records):
            record = blob[record_start:record_start + record_length]
            strip = self._find_jpeg_strip(record)
            if strip is None:
                continue
            jpeg_start, jpeg_bytes, width_px, height_px = strip

            depth = self._read_float64(record, 26)
            depth_duplicate = self._read_float64(record, 34)
            if not self._is_plausible_depth(depth):
                continue

            record_obj = TfdImageRecord(
                index=len(image_records),
                linked_index=linked_index,
                record_start=record_start,
                record_length=record_length,
                previous_pointer=previous_pointer,
                next_pointer=next_pointer,
                timestamp=self._read_timestamp(record),
                depth_m=depth,
                depth_duplicate_m=depth_duplicate if self._is_plausible_depth(depth_duplicate) else depth,
                jpeg_offset=record_start + jpeg_start,
                jpeg_length=len(jpeg_bytes),
                width_px=width_px,
                height_px=height_px,
                channel_payload_bytes=max(0, jpeg_start - 64),
                jpeg_bytes=jpeg_bytes,
                **self._read_telemetry(record),
            )
            if previous_image is not None:
                delta = record_obj.depth_m - previous_image.depth_m
                record_obj.depth_delta_from_prev_m = delta
                time_delta = self._time_delta_seconds(previous_image.timestamp, record_obj.timestamp)
                record_obj.time_delta_from_prev_s = time_delta
                if time_delta and time_delta > 0:
                    record_obj.speed_from_prev_m_per_min = delta / time_delta * 60.0
            image_records.append(record_obj)
            previous_image = record_obj

        return image_records

    def _find_jpeg_strip(self, record: bytes) -> Optional[Tuple[int, bytes, int, int]]:
        """Find the decodable JPEG strip inside a record.

        Some TFD channel payloads contain false JPEG SOI bytes before the real
        strip. Try every SOI marker and keep the first decodable strip matching
        the expected televiewer dimensions.
        """
        search_from = 0
        while True:
            jpeg_start = record.find(b"\xff\xd8", search_from)
            if jpeg_start < 0:
                return None
            jpeg_end = record.find(b"\xff\xd9", jpeg_start)
            if jpeg_end < 0:
                return None
            jpeg_bytes = record[jpeg_start:jpeg_end + 2]
            search_from = jpeg_start + 1
            try:
                with Image.open(BytesIO(jpeg_bytes)) as image:
                    image.load()
                    width_px, height_px = image.size
            except (UnidentifiedImageError, OSError):
                continue
            if self.expected_strip_size and (width_px, height_px) != self.expected_strip_size:
                continue
            return jpeg_start, jpeg_bytes, width_px, height_px
    def _iter_linked_records(self, blob: bytes) -> List[Tuple[int, int, int, int]]:
        records: List[Tuple[int, int, int, int]] = []
        seen = set()
        offset = 0
        while 0 <= offset < len(blob) and offset not in seen:
            seen.add(offset)
            if offset + 12 > len(blob):
                break
            record_length, previous_pointer, next_pointer = struct.unpack_from("<III", blob, offset)
            if record_length <= 0 or offset + record_length > len(blob):
                break
            records.append((offset, record_length, previous_pointer, next_pointer))
            if next_pointer == 0 or next_pointer == offset or next_pointer > len(blob):
                break
            offset = next_pointer
        return records

    @staticmethod
    def _read_float64(record: bytes, offset: int) -> float:
        if offset + 8 > len(record):
            return float("nan")
        return struct.unpack_from("<d", record, offset)[0]

    @staticmethod
    def _is_plausible_depth(value: float) -> bool:
        return math.isfinite(value) and -10_000.0 < value < 10_000.0

    @staticmethod
    def _read_timestamp(record: bytes) -> Optional[str]:
        if 58 > len(record):
            return None
        year, month, _weekday, day, hour, minute, second, millisecond = struct.unpack_from("<8H", record, 42)
        try:
            if not (1900 <= year <= 2200):
                return None
            return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}.{millisecond:03d}"
        except ValueError:
            return None

    @staticmethod
    def _read_telemetry(record: bytes) -> Dict[str, Optional[float]]:
        empty = {
            "roll_deg": None,
            "mroll_deg": None,
            "tilt_deg": None,
            "magfield_uT": None,
            "azimuth_deg": None,
            "gravity_g": None,
            "taps_c": None,
        }
        if 205 > len(record):
            return empty
        roll, mroll, tilt, magfield, azimuth, gravity, taps = struct.unpack_from("<7h", record, 191)
        values = {
            "roll_deg": roll / 10.0,
            "mroll_deg": mroll / 10.0,
            "tilt_deg": tilt / 10.0,
            "magfield_uT": magfield / 100.0,
            "azimuth_deg": azimuth / 10.0,
            "gravity_g": gravity / 10000.0,
            "taps_c": taps / 100.0,
        }
        if not (0 <= values["roll_deg"] <= 360 and 0 <= values["azimuth_deg"] <= 360):
            return empty
        return values

    @staticmethod
    def _time_delta_seconds(previous: Optional[str], current: Optional[str]) -> Optional[float]:
        if not previous or not current:
            return None
        try:
            return (datetime.fromisoformat(current) - datetime.fromisoformat(previous)).total_seconds()
        except ValueError:
            return None

    @staticmethod
    def _estimate_row_pitch(records: Sequence[TfdImageRecord]) -> float:
        deltas = [
            record.depth_delta_from_prev_m
            for record in records[1:]
            if record.depth_delta_from_prev_m is not None and record.depth_delta_from_prev_m > 0
        ]
        if not deltas:
            return 0.002
        median_record_delta = statistics.median(deltas)
        strip_height = statistics.median([record.height_px for record in records]) or 8
        return median_record_delta / strip_height

    @staticmethod
    def _decode_strip(record: TfdImageRecord) -> np.ndarray:
        with Image.open(BytesIO(record.jpeg_bytes)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)

    def _write_meter_slices(
        self,
        plan: TeleviewerImportPlan,
        records: Sequence[TfdImageRecord],
        first_meter: int,
        max_depth_meter: int,
        row_pitch_m: float,
    ) -> List[Dict[str, Any]]:
        strip_arrays = [self._decode_strip(record) for record in records]
        width_px = records[0].width_px
        no_data_row = np.array(NO_DATA_RGB, dtype=np.uint8)
        segment_rows: List[Dict[str, Any]] = []

        all_depths: List[float] = []
        all_pixels: List[np.ndarray] = []
        for record, strip in zip(records, strip_arrays):
            for row_index in range(record.height_px):
                all_depths.append(record.depth_m + row_index * row_pitch_m)
                all_pixels.append(strip[row_index])
        depth_array = np.asarray(all_depths, dtype=float)
        pixel_array = np.asarray(all_pixels, dtype=np.uint8)
        order = np.argsort(depth_array)
        depth_array = depth_array[order]
        pixel_array = pixel_array[order]

        for meter_start in range(first_meter, max_depth_meter):
            meter_end = meter_start + 1
            overlapping = [
                (record, strip)
                for record, strip in zip(records, strip_arrays)
                if record.depth_m < meter_end and record.row_end_depth(row_pitch_m) > meter_start
            ]
            if overlapping:
                raw_image = np.vstack([strip for _record, strip in overlapping])
                raw_path = plan.paths.raw_by_record_dir / self._raw_slice_name(plan.hole_id, meter_start, meter_end)
                Image.fromarray(raw_image).save(raw_path, quality=95)
                segment_rows.append(
                    {
                        "meter_start_m": meter_start,
                        "meter_end_m": meter_end,
                        "file": str(raw_path),
                        "height_px": int(raw_image.shape[0]),
                        "mode": "raw_by_record",
                        "covered_depth_start_m": "",
                        "covered_depth_end_m": "",
                        "covered_rows_px": int(raw_image.shape[0]),
                    }
                )

            target_depths = meter_start + (np.arange(self.pixels_per_meter) + 0.5) / self.pixels_per_meter
            rows = np.empty((self.pixels_per_meter, width_px, 3), dtype=np.uint8)
            rows[:, :, :] = no_data_row
            indices = np.searchsorted(depth_array, target_depths)
            for target_index, insert_index in enumerate(indices):
                candidates = []
                if insert_index < len(depth_array):
                    candidates.append(insert_index)
                if insert_index > 0:
                    candidates.append(insert_index - 1)
                if not candidates:
                    continue
                nearest = min(candidates, key=lambda idx: abs(depth_array[idx] - target_depths[target_index]))
                if abs(depth_array[nearest] - target_depths[target_index]) <= max(row_pitch_m * 1.5, 0.003):
                    rows[target_index] = pixel_array[nearest]

            resampled_path = plan.paths.slices_1m_dir / self._resampled_slice_name(plan.hole_id, meter_start, meter_end)
            Image.fromarray(rows).save(resampled_path, quality=95)
            covered = target_depths[
                (target_depths >= depth_array[0]) & (target_depths <= depth_array[-1])
            ]
            segment_rows.append(
                {
                    "meter_start_m": meter_start,
                    "meter_end_m": meter_end,
                    "file": str(resampled_path),
                    "height_px": self.pixels_per_meter,
                    "mode": "depth_resampled",
                    "covered_depth_start_m": float(covered[0]) if len(covered) else "",
                    "covered_depth_end_m": float(covered[-1]) if len(covered) else "",
                    "covered_rows_px": int(len(covered)),
                }
            )
        return segment_rows

    @staticmethod
    def _raw_slice_name(hole_id: str, start: int, end: int) -> str:
        return f"{hole_id}_{start:03d}_{end:03d}m_raw.jpg"

    @staticmethod
    def _resampled_slice_name(hole_id: str, start: int, end: int) -> str:
        return f"{hole_id}_{start:03d}_{end:03d}m_resampled.jpg"

    @staticmethod
    def _write_records_csv(path: Path, records: Sequence[TfdImageRecord]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(records[0].to_csv_row().keys())
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(record.to_csv_row())

    @staticmethod
    def _write_segment_manifest(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_stretched_zone_report(
        path: Path,
        records: Sequence[TfdImageRecord],
        row_pitch_m: float,
    ) -> List[Dict[str, Any]]:
        deltas = [record.depth_delta_from_prev_m for record in records if record.depth_delta_from_prev_m is not None]
        positive = [delta for delta in deltas if delta > 0]
        threshold = (statistics.median(positive) * 0.5) if positive else row_pitch_m * 4
        zones: List[Dict[str, Any]] = []
        current: List[TfdImageRecord] = []
        for record in records[1:]:
            delta = record.depth_delta_from_prev_m
            if delta is not None and delta < threshold:
                current.append(record)
            elif current:
                zones.append(_zone_row(current, threshold))
                current = []
        if current:
            zones.append(_zone_row(current, threshold))

        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["start_index", "end_index", "start_depth_m", "end_depth_m", "record_count", "threshold_m"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(zones)
        return zones

    def _write_summary(
        self,
        path: Path,
        plan: TeleviewerImportPlan,
        records: Sequence[TfdImageRecord],
        first_meter: int,
        max_depth_meter: int,
        row_pitch_m: float,
        segment_rows: Sequence[Dict[str, Any]],
        stretch_rows: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        deltas = [record.depth_delta_from_prev_m for record in records if record.depth_delta_from_prev_m is not None]
        time_deltas = [record.time_delta_from_prev_s for record in records if record.time_delta_from_prev_s is not None]
        speeds = [record.speed_from_prev_m_per_min for record in records if record.speed_from_prev_m_per_min is not None]
        summary = {
            "source": str(plan.source_path),
            "file_size_bytes": plan.source_path.stat().st_size if plan.source_path.exists() else None,
            "record_count": len(records),
            "jpeg_strip_size_px": [records[0].width_px, records[0].height_px],
            "depth_interval_m_per_record": _safe_median([d for d in deltas if d > 0]),
            "row_pitch_m": row_pitch_m,
            "pixels_per_meter_depth_resampled": self.pixels_per_meter,
            "first_depth_m": min(record.depth_m for record in records),
            "last_depth_m": max(record.depth_m for record in records),
            "min_row_depth_m": min(record.depth_m for record in records),
            "max_row_depth_m": max(record.row_end_depth(row_pitch_m) for record in records),
            "first_meter": first_meter,
            "max_depth_meter": max_depth_meter,
            "first_timestamp": records[0].timestamp,
            "last_timestamp": records[-1].timestamp,
            "median_depth_delta_m": _safe_median(deltas),
            "mean_depth_delta_m": _safe_mean(deltas),
            "min_depth_delta_m": min(deltas) if deltas else None,
            "max_depth_delta_m": max(deltas) if deltas else None,
            "depth_delta_lt_half_median_count": sum(
                1 for delta in deltas if delta is not None and delta < ((_safe_median([d for d in deltas if d > 0]) or row_pitch_m * records[0].height_px) * 0.5)
            ),
            "stall_zone_count": len(stretch_rows),
            "median_time_delta_s": _safe_median(time_deltas),
            "mean_speed_m_per_min": _safe_mean(speeds),
            "median_speed_m_per_min": _safe_median(speeds),
            "raw_segment_count": sum(1 for row in segment_rows if row["mode"] == "raw_by_record"),
            "depth_resampled_segment_count": sum(1 for row in segment_rows if row["mode"] == "depth_resampled"),
            "depth_resampled_no_data_color_rgb": list(NO_DATA_RGB),
            "processing_version": PROCESSING_VERSION,
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
        records: Sequence[TfdImageRecord],
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
        records: Sequence[TfdImageRecord],
        first_meter: int,
        max_depth_meter: int,
        row_pitch_m: float,
        summary: Dict[str, Any],
    ) -> Path:
        manifest = TeleviewerManifest(
            project_code=plan.project_code,
            hole_id=plan.hole_id,
            raw_files=[
                {
                    "name": plan.source_path.name,
                    "source_path": str(plan.source_path),
                    "size_bytes": plan.source_path.stat().st_size if plan.source_path.exists() else None,
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
                "raw_dir": "raw_by_record",
                "resampled_dir": "slices_1m",
                "first_meter": first_meter,
                "max_depth_meter": max_depth_meter,
            },
            telemetry={
                "record_count": len(records),
                "fields": [
                    "roll_deg",
                    "mroll_deg",
                    "tilt_deg",
                    "magfield_uT",
                    "azimuth_deg",
                    "gravity_g",
                    "taps_c",
                ],
            },
            qc={
                "stretched_zone_report": "../qc/stretched_zone_report.csv",
                "extraction_summary": "extraction_summary.json",
                "records_csv": "records.csv",
                "meter_segments_manifest": "meter_segments_manifest.csv",
            },
        )
        return write_manifest(plan.paths.manifest_path, manifest)


def _zone_row(records: Sequence[TfdImageRecord], threshold: float) -> Dict[str, Any]:
    return {
        "start_index": records[0].index,
        "end_index": records[-1].index,
        "start_depth_m": records[0].depth_m,
        "end_depth_m": records[-1].depth_m,
        "record_count": len(records),
        "threshold_m": threshold,
    }


def _safe_median(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    return statistics.median(clean) if clean else None


def _safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    return sum(clean) / len(clean) if clean else None
