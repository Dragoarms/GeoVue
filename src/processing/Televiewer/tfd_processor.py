"""Televiewer TFD processing into GeoVue's shared dataset layout."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Iterable, List

from .atv_decoder import AtvDecoder
from .caliper_decoder import CaliperDecoder
from .enrichment import TeleviewerDataEnricher, TeleviewerEnrichmentResult
from .manifest import placeholder_manifest, write_manifest
from .paths import TeleviewerImportPlan, plan_tfd_imports
from .tfd_decoder import DEFAULT_PIXELS_PER_METER, TfdDecodeResult, TfdDecoder


@dataclass
class TeleviewerFileProcessResult:
    """Result for one source TFD in a folder/file batch."""

    plan: TeleviewerImportPlan
    status: str
    message: str
    decode_result: TfdDecodeResult | None = None


@dataclass
class TeleviewerBatchProcessResult:
    """Summary of a Televiewer folder/file batch run."""

    file_results: List[TeleviewerFileProcessResult]

    @property
    def decoded_count(self) -> int:
        return sum(1 for result in self.file_results if result.status == "decoded")

    @property
    def registered_count(self) -> int:
        return sum(1 for result in self.file_results if result.status == "registered")

    @property
    def failed_count(self) -> int:
        return sum(1 for result in self.file_results if result.status == "failed")

    @property
    def total_count(self) -> int:
        return len(self.file_results)


class TeleviewerProcessor:
    """Prepare and decode televiewer TFD datasets into GeoVue storage."""

    def __init__(self, file_manager, logger=None, data_manager=None):
        self.file_manager = file_manager
        self.logger = logger
        self.data_manager = data_manager
        self.last_batch_result: TeleviewerBatchProcessResult | None = None

    def plan_imports(self, tfd_paths: Iterable[str | Path]) -> List[TeleviewerImportPlan]:
        return plan_tfd_imports(self.file_manager, tfd_paths)

    def prepare_imports(
        self,
        tfd_paths: Iterable[str | Path],
        copy_raw: bool = False,
        decode: bool = False,
        enrich: bool = True,
        pixels_per_meter: int = DEFAULT_PIXELS_PER_METER,
    ) -> List[TeleviewerImportPlan]:
        """Create dataset folders and optionally decode selected TFD files.

        The default remains non-destructive and does not copy large TFDs. When
        ``decode`` is true, processed slices, records, QC outputs, manifest, and
        viewer data are written into ``Televiewer Datasets/<PROJECT>/<HOLE>``.
        """
        if decode:
            self.last_batch_result = self.process_tfd_paths(
                tfd_paths,
                copy_raw=copy_raw,
                enrich=enrich,
                pixels_per_meter=pixels_per_meter,
            )
            return [result.plan for result in self.last_batch_result.file_results]

        plans = self.plan_imports(tfd_paths)
        for plan in plans:
            plan.paths.ensure()
            self._write_source_metadata(plan, raw_file_copied=copy_raw)
            if copy_raw:
                destination = plan.paths.raw_dir / plan.source_path.name
                if not destination.exists() or destination.stat().st_size != plan.source_path.stat().st_size:
                    shutil.copy2(plan.source_path, destination)
            if decode:
                self.decode_plan(plan, pixels_per_meter=pixels_per_meter, enrich=enrich)
            elif not plan.paths.viewer_data_path.exists():
                manifest = placeholder_manifest(plan.project_code, plan.hole_id)
                manifest.raw_files = [
                    {
                        "name": plan.source_path.name,
                        "source_path": str(plan.source_path),
                        "size_bytes": plan.source_path.stat().st_size if plan.source_path.exists() else None,
                    }
                ]
                write_manifest(plan.paths.manifest_path, manifest)
        return plans

    def process_tfd_paths(
        self,
        tfd_paths: Iterable[str | Path],
        copy_raw: bool = False,
        enrich: bool = True,
        pixels_per_meter: int = DEFAULT_PIXELS_PER_METER,
    ) -> TeleviewerBatchProcessResult:
        """Register and decode a batch of TFD files.

        Folder-level imports commonly contain OTV/ATV image files mixed with
        calibration, caliper, e-log, and magsus logs. Unsupported non-image TFDs
        are preserved as raw source metadata and do not abort the batch.
        """
        file_results: List[TeleviewerFileProcessResult] = []
        plans = self.plan_imports(tfd_paths)
        caliper_plans: List[TeleviewerImportPlan] = []
        for plan in plans:
            try:
                plan.paths.ensure()
                self._write_source_metadata(plan, raw_file_copied=copy_raw)
                if copy_raw:
                    destination = plan.paths.raw_dir / plan.source_path.name
                    if (
                        not destination.exists()
                        or destination.stat().st_size != plan.source_path.stat().st_size
                    ):
                        shutil.copy2(plan.source_path, destination)

                source_kind = self._source_kind(plan.source_path)
                if source_kind == "caliper":
                    rows = CaliperDecoder().decode_to_dataset(plan.source_path, plan.paths)
                    caliper_plans.append(plan)
                    self._sync_manifest_raw_files(plan)
                    file_results.append(
                        TeleviewerFileProcessResult(
                            plan=plan,
                            status="registered",
                            message=f"Extracted {len(rows)} caliper diameter rows.",
                        )
                    )
                    continue

                decode_result = self.decode_plan(
                    plan,
                    pixels_per_meter=pixels_per_meter,
                    enrich=enrich,
                )
                self._sync_manifest_raw_files(plan)
                file_results.append(
                    TeleviewerFileProcessResult(
                        plan=plan,
                        status="decoded",
                        message=f"Decoded {len(decode_result.records)} image strip records.",
                        decode_result=decode_result,
                    )
                )
            except ValueError as exc:
                message = str(exc)
                if "No valid OTV JPEG strip records found" in message:
                    if self._source_kind(plan.source_path) == "atv" or AtvDecoder.looks_like_atv(plan.source_path):
                        try:
                            decode_result = AtvDecoder(
                                pixels_per_meter=pixels_per_meter
                            ).decode_to_dataset(plan)
                            if enrich:
                                self.enrich_plan(plan, max_depth=decode_result.max_depth_meter)
                            self._sync_manifest_raw_files(plan)
                            file_results.append(
                                TeleviewerFileProcessResult(
                                    plan=plan,
                                    status="decoded",
                                    message=(
                                        f"Decoded {len(decode_result.records)} ATV image rows "
                                        "from ABI40 amplitude data."
                                    ),
                                    decode_result=decode_result,
                                )
                            )
                            continue
                        except Exception as atv_exc:
                            message = f"{message}; ATV fallback failed: {atv_exc}"
                    if not plan.paths.viewer_data_path.exists() and not plan.paths.manifest_path.exists():
                        manifest = placeholder_manifest(plan.project_code, plan.hole_id)
                        manifest.raw_files = self._read_source_raw_files(plan)
                        write_manifest(plan.paths.manifest_path, manifest)
                    self._sync_manifest_raw_files(plan)
                    file_results.append(
                        TeleviewerFileProcessResult(
                            plan=plan,
                            status="registered",
                            message="Registered source TFD; no compatible OTV JPEG image strips found.",
                        )
                    )
                    if self.logger:
                        self.logger.info("Registered non-image or unsupported TFD: %s", plan.source_path)
                    continue
                file_results.append(
                    TeleviewerFileProcessResult(
                        plan=plan,
                        status="failed",
                        message=message,
                    )
                )
                if self.logger:
                    self.logger.error("Televiewer TFD processing failed for %s: %s", plan.source_path, exc)
            except Exception as exc:
                file_results.append(
                    TeleviewerFileProcessResult(
                        plan=plan,
                        status="failed",
                        message=str(exc),
                    )
                )
                if self.logger:
                    self.logger.error("Televiewer TFD processing failed for %s: %s", plan.source_path, exc)

        for plan in caliper_plans:
            try:
                rows = CaliperDecoder().read_diameter_rows(plan.source_path)
                CaliperDecoder.write_csv(plan.paths.processed_dir / "caliper_diameter.csv", rows)
                CaliperDecoder.update_viewer_data(plan.paths, rows)
                CaliperDecoder.update_manifest(plan.paths, rows)
            except Exception as exc:
                if self.logger:
                    self.logger.warning("Could not re-apply caliper data for %s: %s", plan.source_path, exc)

        result = TeleviewerBatchProcessResult(file_results=file_results)
        self.last_batch_result = result
        return result

    def decode_imports(
        self,
        tfd_paths: Iterable[str | Path],
        copy_raw: bool = False,
        enrich: bool = True,
        pixels_per_meter: int = DEFAULT_PIXELS_PER_METER,
    ) -> List[TfdDecodeResult]:
        """Decode selected TFD files and return output summaries."""
        plans = self.plan_imports(tfd_paths)
        results: List[TfdDecodeResult] = []
        for plan in plans:
            plan.paths.ensure()
            self._write_source_metadata(plan, raw_file_copied=copy_raw)
            if copy_raw:
                destination = plan.paths.raw_dir / plan.source_path.name
                if not destination.exists() or destination.stat().st_size != plan.source_path.stat().st_size:
                    shutil.copy2(plan.source_path, destination)
            results.append(self.decode_plan(plan, pixels_per_meter=pixels_per_meter, enrich=enrich))
        return results

    def decode_plan(
        self,
        plan: TeleviewerImportPlan,
        pixels_per_meter: int = DEFAULT_PIXELS_PER_METER,
        enrich: bool = True,
    ) -> TfdDecodeResult:
        """Decode one already-planned TFD import."""
        if self.logger:
            self.logger.info(f"Decoding televiewer TFD: {plan.source_path}")
        decoder = TfdDecoder(pixels_per_meter=pixels_per_meter)
        result = decoder.decode_to_dataset(plan)
        if enrich:
            self.enrich_plan(plan, max_depth=result.max_depth_meter)
        if self.logger:
            self.logger.info(
                "Decoded televiewer TFD %s: %s records, %s-%s m",
                plan.source_path,
                len(result.records),
                result.first_meter,
                result.max_depth_meter,
            )
        return result

    def enrich_plan(
        self,
        plan: TeleviewerImportPlan,
        max_depth: float | None = None,
    ) -> TeleviewerEnrichmentResult:
        """Enrich an already-decoded dataset with GeoVue data and chip images."""
        enricher = TeleviewerDataEnricher(
            file_manager=self.file_manager,
            data_manager=self.data_manager,
            logger=self.logger,
        )
        return enricher.enrich_dataset(plan.paths, plan.hole_id, max_depth=max_depth)

    def _write_source_metadata(
        self,
        plan: TeleviewerImportPlan,
        raw_file_copied: bool = False,
    ) -> None:
        metadata_path = plan.paths.raw_dir / "source_metadata.json"
        existing: dict = {}
        if metadata_path.exists():
            try:
                existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}

        raw_files = existing.get("raw_files")
        if not isinstance(raw_files, list):
            raw_files = []
        entry = {
            "name": plan.source_path.name,
            "source_path": str(plan.source_path),
            "size_bytes": plan.source_path.stat().st_size if plan.source_path.exists() else None,
            "registered_utc": datetime.now(timezone.utc).isoformat(),
            "raw_file_copied": raw_file_copied,
            "source_kind": self._source_kind(plan.source_path),
        }
        deduped = [
            item
            for item in raw_files
            if str(item.get("source_path", "")).lower() != str(plan.source_path).lower()
        ]
        deduped.append(entry)
        payload = {
            "project_code": plan.project_code,
            "hole_id": plan.hole_id,
            "source_path": entry["source_path"],
            "source_name": entry["name"],
            "source_size_bytes": entry["size_bytes"],
            "last_registered_utc": entry["registered_utc"],
            "raw_file_copied": raw_file_copied,
            "raw_files": deduped,
        }
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _read_source_raw_files(self, plan: TeleviewerImportPlan) -> List[dict]:
        metadata_path = plan.paths.raw_dir / "source_metadata.json"
        if not metadata_path.exists():
            return []
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        raw_files = payload.get("raw_files")
        return raw_files if isinstance(raw_files, list) else []

    def _sync_manifest_raw_files(self, plan: TeleviewerImportPlan) -> None:
        if not plan.paths.manifest_path.exists():
            return
        raw_files = self._read_source_raw_files(plan)
        if not raw_files:
            return
        try:
            payload = json.loads(plan.paths.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        payload["raw_files"] = raw_files
        with plan.paths.manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    @staticmethod
    def _source_kind(source_path: Path) -> str:
        name = source_path.name.lower()
        for token, kind in (
            ("otv", "otv"),
            ("atv", "atv"),
            ("caliper", "caliper"),
            ("elog", "elog"),
            ("magsus", "magsus"),
        ):
            if token in name:
                return kind
        return "unknown"
