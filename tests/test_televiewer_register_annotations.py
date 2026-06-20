import json
import logging
from pathlib import Path
import shutil
import uuid

import pytest
from utils.json_register_manager import JSONRegisterManager
from gui.Televiewer.televiewer_viewer_dialog import TeleviewerViewerDialog


@pytest.fixture
def temp_register_workspace():
    workspace = Path.cwd() / f"televiewer_register_test_{uuid.uuid4().hex}"
    workspace.mkdir()
    try:
        yield workspace
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _annotation_manager(workspace):
    manager = JSONRegisterManager.__new__(JSONRegisterManager)
    manager.logger = logging.getLogger(__name__)
    manager.data_path = workspace / "Register Data (Do not edit)"
    manager.data_path.mkdir(parents=True, exist_ok=True)
    manager.televiewer_annotations_path = (
        manager.data_path / JSONRegisterManager.TELEVIEWER_ANNOTATIONS_JSON
    )
    manager.televiewer_annotations_lock = manager.televiewer_annotations_path.with_suffix(
        ".json.lock"
    )
    manager._acquire_file_lock = lambda _lock_path: True
    manager._release_file_lock = lambda _lock_path: None
    return manager


def test_update_televiewer_annotations_round_trips_one_hole(
    temp_register_workspace, monkeypatch
):
    monkeypatch.setenv("USERNAME", "TeleviewerTester")
    manager = _annotation_manager(temp_register_workspace)

    payload = {
        "hole_id": "ba0007",
        "planes": [{"id": "plane-1", "category": "Bedding", "dip": 42.0}],
        "observations": [{"depth_m": 52.3, "comment": "test point"}],
    }

    assert manager.update_televiewer_annotations(
        "ba0007",
        payload,
        project_code="BA",
        user="codex-test",
    )

    stored = manager.read_televiewer_annotations("BA0007")
    assert set(stored) == {"BA0007"}
    entry = stored["BA0007"]
    assert entry["hole_id"] == "BA0007"
    assert entry["project_code"] == "BA"
    assert entry["updated_by"] == "codex-test"
    assert entry["schema"] == "televiewer_annotations.v1"
    assert entry["payload"]["planes"][0]["category"] == "Bedding"
    assert entry["payload"]["observations"][0]["depth_m"] == 52.3

    annotations_path = (
        manager.data_path / JSONRegisterManager.TELEVIEWER_ANNOTATIONS_JSON
    )
    assert annotations_path.exists()
    with annotations_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    assert "BA0007" in raw


def test_update_televiewer_annotations_preserves_existing_project_code(
    temp_register_workspace,
):
    manager = _annotation_manager(temp_register_workspace)

    assert manager.update_televiewer_annotations(
        "BA0007",
        {"planes": []},
        project_code="BA",
    )
    assert manager.update_televiewer_annotations(
        "BA0007",
        {"planes": [{"id": "plane-2", "category": "Fracture"}]},
    )

    stored = manager.read_televiewer_annotations("BA0007")["BA0007"]
    assert stored["project_code"] == "BA"
    assert stored["payload"]["planes"][0]["category"] == "Fracture"


def test_update_televiewer_annotations_rejects_missing_hole_or_bad_payload(
    temp_register_workspace,
):
    manager = _annotation_manager(temp_register_workspace)

    assert manager.update_televiewer_annotations("", {"planes": []}) is False
    assert manager.update_televiewer_annotations("BA0007", ["not", "a", "dict"]) is False
    assert manager.read_televiewer_annotations() == {}


class _FakeServer:
    def __init__(self):
        self.routes = {}

    def add_json_route(self, name, getter, writer=None):
        self.routes[name] = (getter, writer)
        return f"/_json/{name}"

    def add_json(self, name, payload):
        self.routes[name] = (lambda: payload, None)
        return f"/_virtual/{name}"


class _ValueVar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class _FakeViewerServer(_FakeServer):
    def __init__(self):
        super().__init__()
        self.mounts = {}
        self.viewer_params = []

    def add_mount(self, name, path):
        self.mounts[name] = Path(path).resolve()

    def mounted_path(self, mount, relative_path=""):
        relative = str(relative_path).replace("\\", "/").strip("/")
        return f"/{mount}/{relative}" if relative else f"/{mount}"

    def viewer_url(self, params):
        self.viewer_params.append(dict(params))
        return f"http://127.0.0.1/viewer/{len(self.viewer_params)}"


def test_viewer_dialog_annotation_route_uses_register_manager(temp_register_workspace):
    register_manager = _annotation_manager(temp_register_workspace)
    dialog = TeleviewerViewerDialog.__new__(TeleviewerViewerDialog)
    dialog.register_manager = register_manager

    server = _FakeServer()
    urls = dialog._served_annotation_urls(server, "ba0007", project_code="BA")

    assert urls == {
        "annotationsUrl": "/_json/televiewer_annotations/ba0007.json",
        "annotationsSaveUrl": "/_json/televiewer_annotations/ba0007.json",
    }
    getter, writer = server.routes["televiewer_annotations/ba0007.json"]
    assert getter()["planes"] == []

    result = writer(
        {
            "planes": [{"id": "plane-1", "category": "Bedding"}],
            "observations": [{"depth_m": 55.2, "kind": "point"}],
        }
    )

    assert result == {"saved": True, "hole_id": "BA0007"}
    loaded = getter()
    assert loaded["planes"][0]["category"] == "Bedding"
    assert loaded["observations"][0]["depth_m"] == 55.2
    assert loaded["register"]["project_code"] == "BA"



def _write_manifest(root, project_code, hole_id):
    processed_dir = root / project_code / hole_id / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "viewer_data.json").write_text(
        json.dumps({"hole_id": hole_id, "segments": []}),
        encoding="utf-8",
    )
    manifest_path = processed_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "project_code": project_code,
                "hole_id": hole_id,
                "viewer": {
                    "hole_id": hole_id,
                    "data_url": "viewer_data.json",
                    "chip_tray_manifest_url": "chip_tray_manifest.json",
                    "raw_dir": "raw_by_record",
                    "resampled_dir": "slices_1m",
                },
                "coverage": {"first_meter": 1, "max_depth_meter": 100},
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_viewer_dialog_uses_stable_unique_mounts_for_multiple_open_holes(
    temp_register_workspace,
    monkeypatch,
):
    server = _FakeViewerServer()
    monkeypatch.setattr(
        TeleviewerViewerDialog,
        "_get_server",
        classmethod(lambda cls, web_root: server),
    )

    dialog = TeleviewerViewerDialog.__new__(TeleviewerViewerDialog)
    dialog.web_root = temp_register_workspace / "web"
    dialog.register_manager = None
    dialog.start_var = _ValueVar("10")
    dialog.range_var = _ValueVar("4")
    dialog.geometry_var = _ValueVar("flat")
    dialog.chip_var = _ValueVar(True)
    opened_urls = []
    dialog._open_url = opened_urls.append

    first_manifest = _write_manifest(temp_register_workspace, "BA", "BA0007")
    second_manifest = _write_manifest(temp_register_workspace, "BA", "BA0008")

    dialog._open_manifest(first_manifest)
    dialog._open_manifest(second_manifest)

    first_data_url = server.viewer_params[0]["dataUrl"]
    second_data_url = server.viewer_params[1]["dataUrl"]
    first_mount = first_data_url.split("/")[1]
    second_mount = second_data_url.split("/")[1]

    assert opened_urls == ["http://127.0.0.1/viewer/1", "http://127.0.0.1/viewer/2"]
    assert first_mount.startswith("dataset_ba0007_")
    assert second_mount.startswith("dataset_ba0008_")
    assert first_mount != second_mount
    assert server.mounts[first_mount] == first_manifest.parents[1].resolve()
    assert server.mounts[second_mount] == second_manifest.parents[1].resolve()
