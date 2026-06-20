"""Launcher dialog for processed televiewer datasets."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tkinter as tk
from tkinter import ttk
from urllib.parse import unquote, urlparse
import webbrowser

from gui.dialog_helper import DialogHelper
from processing.Televiewer.manifest import discover_manifests, load_manifest, manifest_status
from processing.Televiewer.paths import get_televiewer_root

from .local_server import TeleviewerWebServer


class TeleviewerViewerDialog:
    """Open the preserved WebGL televiewer viewer from GeoVue."""

    _server: TeleviewerWebServer | None = None

    def __init__(
        self,
        parent,
        gui_manager,
        file_manager,
        config_manager=None,
        register_manager=None,
        initial_hole_id: str | None = None,
        initial_start_depth: float | None = None,
        initial_range_m: float | None = None,
        auto_open: bool = False,
    ):
        self.parent = parent
        self.gui_manager = gui_manager
        self.file_manager = file_manager
        self.config_manager = config_manager
        self.register_manager = register_manager
        self.initial_hole_id = str(initial_hole_id).strip().upper() if initial_hole_id else ""
        self.auto_open = auto_open
        self.repo_root = Path(__file__).resolve().parents[3]
        self.web_root = Path(__file__).resolve().parent / "web"
        self.rows = []
        self.start_var = tk.StringVar(value=_format_depth(initial_start_depth, default="52"))
        self.range_var = tk.StringVar(value=_format_depth(initial_range_m, default="4"))
        self.geometry_var = tk.StringVar(value="flat")
        self.chip_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready")
        self._create_dialog()
        self._refresh()
        if self.initial_hole_id:
            found = self._select_manifest_for_hole(self.initial_hole_id)
            if auto_open and found:
                self.dialog.after(50, self._open_selected)
            elif auto_open:
                self.status_var.set(f"No processed televiewer dataset found for {self.initial_hole_id}.")

    @classmethod
    def _get_server(cls, web_root: Path) -> TeleviewerWebServer:
        if cls._server is None:
            cls._server = TeleviewerWebServer(web_root)
        return cls._server

    def _create_dialog(self) -> None:
        self.dialog = DialogHelper.create_dialog(
            self.parent,
            DialogHelper.t("Televiewer Viewer"),
            modal=False,
            topmost=False,
        )
        self.dialog.configure(bg=self.gui_manager.theme_colors["background"])
        self.dialog.geometry("900x560")

        frame = ttk.Frame(self.dialog, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(
            frame,
            text=DialogHelper.t("Televiewer Datasets"),
            font=self.gui_manager.fonts.get("heading"),
        )
        title.pack(anchor="w", pady=(0, 8))

        columns = ("project", "hole", "status", "manifest")
        self.tree = ttk.Treeview(frame, columns=columns, show="headings", height=12)
        for column, label, width in (
            ("project", "Project", 110),
            ("hole", "Hole", 130),
            ("status", "Status", 150),
            ("manifest", "Manifest", 480),
        ):
            self.tree.heading(column, text=label)
            self.tree.column(column, width=width, anchor="w")
        self.tree.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(frame)
        controls.pack(fill=tk.X, pady=(10, 4))

        ttk.Label(controls, text=DialogHelper.t("Start m")).pack(side=tk.LEFT)
        ttk.Entry(controls, textvariable=self.start_var, width=8).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(controls, text=DialogHelper.t("Range m")).pack(side=tk.LEFT)
        ttk.Entry(controls, textvariable=self.range_var, width=8).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(controls, text=DialogHelper.t("Geometry")).pack(side=tk.LEFT)
        ttk.Combobox(
            controls,
            textvariable=self.geometry_var,
            values=("flat", "wrapped"),
            state="readonly",
            width=10,
        ).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Checkbutton(
            controls,
            text=DialogHelper.t("Chip tray"),
            variable=self.chip_var,
        ).pack(side=tk.LEFT, padx=(4, 10))

        actions = ttk.Frame(frame)
        actions.pack(fill=tk.X, pady=(8, 0))
        self.gui_manager.create_modern_button(
            actions,
            text=DialogHelper.t("Refresh"),
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._refresh,
        ).pack(side=tk.LEFT, padx=(0, 6))
        self.gui_manager.create_modern_button(
            actions,
            text=DialogHelper.t("Open Selected"),
            color=self.gui_manager.theme_colors["accent_green"],
            command=self._open_selected,
        ).pack(side=tk.LEFT, padx=6)
        self.gui_manager.create_modern_button(
            actions,
            text=DialogHelper.t("Open BA0007 Prototype"),
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._open_ba0007_prototype,
        ).pack(side=tk.LEFT, padx=6)
        self.gui_manager.create_modern_button(
            actions,
            text=DialogHelper.t("Close"),
            color=self.gui_manager.theme_colors["accent_red"],
            command=self.dialog.destroy,
        ).pack(side=tk.RIGHT)

        ttk.Label(frame, textvariable=self.status_var).pack(anchor="w", pady=(8, 0))

    def _refresh(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.rows.clear()

        root = get_televiewer_root(self.file_manager, create_if_missing=False)
        manifests = discover_manifests(root)
        for manifest_path in manifests:
            try:
                manifest = load_manifest(manifest_path)
            except Exception:
                manifest = {}
            project = manifest.get("project_code", manifest_path.parents[2].name)
            hole = manifest.get("hole_id", manifest_path.parents[1].name)
            row = {
                "kind": "manifest",
                "project": project,
                "hole": hole,
                "manifest_path": manifest_path,
            }
            self.rows.append(row)
            self.tree.insert(
                "",
                "end",
                iid=str(len(self.rows) - 1),
                values=(project, hole, manifest_status(manifest_path), str(manifest_path)),
            )

        scratch_ready = self._scratch_viewer_data_path().exists()
        suffix = " BA0007 prototype available." if scratch_ready else " BA0007 prototype data not found."
        self.status_var.set(f"Found {len(manifests)} processed televiewer manifest(s).{suffix}")

    def _select_manifest_for_hole(self, hole_id: str) -> bool:
        target = str(hole_id).strip().upper()
        if not target:
            return False
        for index, row in enumerate(self.rows):
            if str(row.get("hole", "")).strip().upper() == target:
                iid = str(index)
                self.tree.selection_set(iid)
                self.tree.focus(iid)
                self.tree.see(iid)
                return True
        return False

    def _open_selected(self) -> None:
        selection = self.tree.selection()
        if not selection:
            DialogHelper.show_message(
                self.dialog,
                DialogHelper.t("Televiewer Viewer"),
                DialogHelper.t("Select a processed televiewer manifest first."),
                message_type="warning",
            )
            return
        row = self.rows[int(selection[0])]
        self._open_manifest(row["manifest_path"])

    def _open_manifest(self, manifest_path: Path) -> None:
        manifest = load_manifest(manifest_path)
        viewer = manifest.get("viewer", {})
        hole_dir = manifest_path.parents[1]
        processed_dir = manifest_path.parent
        data_ref = viewer.get("data_url") or "viewer_data.json"
        data_candidate = Path(data_ref)
        data_path = data_candidate if data_candidate.is_absolute() else processed_dir / data_candidate
        if not data_path.exists():
            DialogHelper.show_message(
                self.dialog,
                DialogHelper.t("Televiewer Viewer"),
                DialogHelper.t(
                    "This Televiewer dataset has a manifest but no viewer_data.json. "
                    "Process it with 'Decode image slices now' before opening the viewer."
                ),
                message_type="warning",
            )
            self.status_var.set(f"Manifest only: {manifest_path}")
            return

        server = self._get_server(self.web_root)
        hole_id = viewer.get("hole_id") or manifest.get("hole_id", "UNKNOWN")
        project_code = manifest.get("project_code", "")
        dataset_mount = self._dataset_mount_name(hole_id, hole_dir)
        server.add_mount(dataset_mount, hole_dir)

        def processed_url(value: str, fallback: str) -> str:
            rel = value or fallback
            candidate = Path(rel)
            if candidate.is_absolute():
                return rel.replace("\\", "/")
            try:
                rel_to_hole = (processed_dir / candidate).resolve().relative_to(hole_dir.resolve())
            except ValueError:
                rel_to_hole = Path("processed") / candidate
            return server.mounted_path(dataset_mount, rel_to_hole)

        chip_manifest_url = self._served_chip_manifest_url(server, processed_dir, viewer, hole_id)
        if not chip_manifest_url:
            chip_manifest_url = processed_url(viewer.get("chip_tray_manifest_url"), "chip_tray_manifest.json")

        params = self._base_viewer_params()
        annotation_urls = self._served_annotation_urls(server, hole_id, project_code)
        coverage = manifest.get("coverage", {})
        params.update(
            {
                "holeId": hole_id,
                "dataUrl": processed_url(viewer.get("data_url"), "viewer_data.json"),
                "chipTrayManifestUrl": chip_manifest_url,
                "rawDir": processed_url(viewer.get("raw_dir"), "raw_by_record"),
                "resampledDir": processed_url(viewer.get("resampled_dir"), "slices_1m"),
                "firstMeter": viewer.get("first_meter") or coverage.get("first_meter") or 1,
                "maxDepthMeter": viewer.get("max_depth_meter") or coverage.get("max_depth_meter") or 102,
                **annotation_urls,
            }
        )
        self._open_url(server.viewer_url(params))

    def _served_annotation_urls(self, server: TeleviewerWebServer, hole_id: str, project_code: str = "") -> dict:
        """Expose Televiewer annotations through JSONRegisterManager-backed routes."""
        route_name = f"televiewer_annotations/{_safe_slug(hole_id)}.json"
        if self.register_manager is None:
            url = server.add_json(route_name, self._empty_annotation_payload(hole_id))
            return {"annotationsUrl": url}

        def getter():
            return self._read_annotation_payload(hole_id)

        def writer(payload):
            return self._write_annotation_payload(hole_id, project_code, payload)

        url = server.add_json_route(route_name, getter, writer)
        return {"annotationsUrl": url, "annotationsSaveUrl": url}

    def _read_annotation_payload(self, hole_id: str) -> dict:
        payload = self._empty_annotation_payload(hole_id)
        if self.register_manager is None:
            return payload
        key = str(hole_id).strip().upper()
        data = self.register_manager.read_televiewer_annotations(key)
        entry = data.get(key, {}) if isinstance(data, dict) else {}
        stored_payload = entry.get("payload") if isinstance(entry, dict) else None
        if isinstance(stored_payload, dict):
            payload.update(stored_payload)
        if isinstance(entry, dict) and entry:
            payload["register"] = {
                "updated_at": entry.get("updated_at", ""),
                "updated_by": entry.get("updated_by", ""),
                "project_code": entry.get("project_code", ""),
                "schema": entry.get("schema", ""),
            }
        return payload

    def _write_annotation_payload(self, hole_id: str, project_code: str, payload: dict) -> dict:
        if self.register_manager is None:
            raise RuntimeError("No JSON register manager is available for Televiewer annotations")
        if not isinstance(payload, dict):
            raise ValueError("Televiewer annotation payload must be a JSON object")
        clean_payload = dict(payload)
        clean_payload["hole_id"] = str(hole_id).strip().upper()
        clean_payload.pop("register", None)
        ok = self.register_manager.update_televiewer_annotations(
            hole_id,
            clean_payload,
            project_code=project_code,
        )
        if not ok:
            raise RuntimeError("Televiewer annotation register update failed")
        return {"saved": True, "hole_id": clean_payload["hole_id"]}

    @staticmethod
    def _empty_annotation_payload(hole_id: str) -> dict:
        return {
            "schema": "televiewer_annotations.v1",
            "hole_id": str(hole_id).strip().upper(),
            "observations": [],
            "planePoints": [],
            "planes": [],
        }

    def _served_chip_manifest_url(
        self,
        server: TeleviewerWebServer,
        processed_dir: Path,
        viewer: dict,
        hole_id: str,
    ) -> str | None:
        chip_manifest_ref = viewer.get("chip_tray_manifest_url") or "chip_tray_manifest.json"
        chip_manifest_path = Path(chip_manifest_ref)
        if not chip_manifest_path.is_absolute():
            chip_manifest_path = processed_dir / chip_manifest_path
        if not chip_manifest_path.exists():
            return None

        try:
            with chip_manifest_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return None

        rows = payload if isinstance(payload, list) else payload.get("images") or payload.get("rows") or []
        existing_paths = []
        for row in rows:
            image_path = self._chip_row_path(row)
            if image_path and image_path.exists():
                existing_paths.append(image_path.resolve())
        if not existing_paths:
            return None

        try:
            common_root = Path(os.path.commonpath([str(path.parent) for path in existing_paths])).resolve()
        except ValueError:
            return None

        mount_hash = hashlib.sha1(str(common_root).encode("utf-8")).hexdigest()[:8]
        mount_name = f"chip_{_safe_slug(hole_id)}_{mount_hash}"
        server.add_mount(mount_name, common_root)

        rewritten_rows = []
        for row in rows:
            rewritten = dict(row)
            image_path = self._chip_row_path(row)
            if image_path and image_path.exists():
                try:
                    rel = image_path.resolve().relative_to(common_root)
                    rewritten["url"] = server.mounted_path(mount_name, rel)
                    rewritten["path"] = str(image_path)
                except ValueError:
                    pass
            rewritten_rows.append(rewritten)

        if isinstance(payload, list):
            served_payload = rewritten_rows
        else:
            served_payload = dict(payload)
            if "images" in served_payload or "rows" not in served_payload:
                served_payload["images"] = rewritten_rows
            else:
                served_payload["rows"] = rewritten_rows
        virtual_name = f"chip_manifest_{_safe_slug(hole_id)}_{mount_hash}.json"
        return server.add_json(virtual_name, served_payload)

    @staticmethod
    def _dataset_mount_name(hole_id: str, hole_dir: Path) -> str:
        digest = hashlib.sha1(str(Path(hole_dir).resolve()).encode("utf-8")).hexdigest()[:8]
        return f"dataset_{_safe_slug(hole_id)}_{digest}"

    @staticmethod
    def _chip_row_path(row: dict) -> Path | None:
        path_value = row.get("path") or row.get("file") or row.get("src")
        url_value = row.get("url")
        if not path_value and isinstance(url_value, str) and url_value.lower().startswith("file:"):
            parsed = urlparse(url_value)
            path_value = unquote(parsed.path)
            if re.match(r"^/[A-Za-z]:/", path_value):
                path_value = path_value[1:]
        if not path_value:
            return None
        try:
            return Path(path_value)
        except (TypeError, ValueError):
            return None

    def _open_ba0007_prototype(self) -> None:
        scratch_root = self.repo_root / "tfd_extract_preview"
        if not self._scratch_viewer_data_path().exists():
            DialogHelper.show_message(
                self.dialog,
                DialogHelper.t("Televiewer Viewer"),
                DialogHelper.t("BA0007 prototype data was not found in the ignored scratch folder."),
                message_type="warning",
            )
            return
        server = self._get_server(self.web_root)
        server.add_mount("scratch", scratch_root)
        params = self._base_viewer_params()
        params.update(
            {
                "holeId": "BA0007",
                "dataUrl": server.mounted_path("scratch", "otv_cylinder_viewer/BA0007_viewer_data.json"),
                "chipTrayManifestUrl": server.mounted_path("scratch", "otv_cylinder_viewer/chip_tray_manifest.json"),
                "rawDir": server.mounted_path("scratch", "BA0007_meter_segments_raw_by_record"),
                "resampledDir": server.mounted_path(
                    "scratch", "BA0007_meter_segments_depth_resampled_500px_per_m"
                ),
                "firstMeter": 1,
                "maxDepthMeter": 102,
            }
        )
        self._open_url(server.viewer_url(params))

    def _base_viewer_params(self) -> dict:
        return {
            "start": self.start_var.get().strip() or "1",
            "range": self.range_var.get().strip() or "4",
            "geometry": self.geometry_var.get(),
            "chip": "1" if self.chip_var.get() else "0",
            "v": "geovue-integration",
        }

    def _open_url(self, url: str) -> None:
        webbrowser.open(url)
        self.status_var.set(f"Opened {url}")

    def _scratch_viewer_data_path(self) -> Path:
        return self.repo_root / "tfd_extract_preview" / "otv_cylinder_viewer" / "BA0007_viewer_data.json"


def open_televiewer_for_hole(
    parent,
    gui_manager,
    file_manager,
    config_manager=None,
    register_manager=None,
    hole_id: str | None = None,
    start_depth: float | None = None,
    range_m: float | None = 4.0,
):
    return TeleviewerViewerDialog(
        parent,
        gui_manager,
        file_manager,
        config_manager=config_manager,
        register_manager=register_manager,
        initial_hole_id=hole_id,
        initial_start_depth=start_depth,
        initial_range_m=range_m,
        auto_open=bool(hole_id),
    )


def _format_depth(value: float | None, default: str) -> str:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if abs(number - round(number)) < 0.001:
        return str(int(round(number)))
    return f"{number:.2f}".rstrip("0").rstrip(".")


def _safe_slug(value: object) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_").lower()
    return slug or "televiewer"
