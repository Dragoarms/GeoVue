"""Small mounted HTTP server for the Televiewer WebGL viewer."""

from __future__ import annotations

from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Dict, Mapping
from urllib.parse import quote, urlencode, unquote, urlparse


JsonGetter = Callable[[], Any]
JsonWriter = Callable[[Any], Any]


class _MountedRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - BaseHTTPRequestHandler name
        return

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        request_path = unquote(parsed.path)
        if request_path in ("", "/"):
            self.send_response(302)
            self.send_header("Location", "/web/index.html")
            self.end_headers()
            return

        parts = [part for part in request_path.strip("/").split("/") if part]
        if not parts:
            self.send_error(404)
            return

        mount_name = parts[0]
        if mount_name == "_virtual":
            self._serve_virtual_json("/".join(parts[1:]))
            return
        if mount_name == "_json":
            self._serve_json_route("/".join(parts[1:]))
            return

        root = self.server.mounts.get(mount_name)  # type: ignore[attr-defined]
        if root is None:
            self.send_error(404, f"Unknown televiewer mount: {mount_name}")
            return

        try:
            root_path = Path(root).resolve()
            target = (root_path.joinpath(*parts[1:])).resolve()
            if root_path != target and root_path not in target.parents:
                self.send_error(403)
                return
            if target.is_dir():
                target = target / "index.html"
            if not target.exists() or not target.is_file():
                self.send_error(404)
                return
            content = target.read_bytes()
        except OSError as exc:
            self.send_error(500, str(exc))
            return

        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
        self._handle_json_route_write()

    def do_PUT(self):  # noqa: N802 - BaseHTTPRequestHandler API
        self._handle_json_route_write()

    def _handle_json_route_write(self) -> None:
        parsed = urlparse(self.path)
        parts = [part for part in unquote(parsed.path).strip("/").split("/") if part]
        if len(parts) < 2 or parts[0] != "_json":
            self.send_error(403, "Televiewer writes require a registered JSON route")
            return

        name = "/".join(parts[1:])
        route = self.server.json_routes.get(name)  # type: ignore[attr-defined]
        if route is None:
            self.send_error(404, f"Unknown JSON route: {name}")
            return
        _getter, writer = route
        if writer is None:
            self.send_error(405, f"JSON route is read-only: {name}")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 5 * 1024 * 1024:
                self.send_error(413, "Invalid Televiewer JSON payload size")
                return
            raw = self.rfile.read(content_length)
            payload = json.loads(raw.decode("utf-8"))
            result = writer(payload)
        except json.JSONDecodeError as exc:
            self.send_error(400, f"Invalid JSON: {exc}")
            return
        except Exception as exc:
            self.send_error(500, str(exc))
            return

        self._send_json({"ok": True, "result": result})

    def _send_json(self, payload: Any) -> None:
        content = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _serve_virtual_json(self, name: str) -> None:
        payload = self.server.virtual_payloads.get(name)  # type: ignore[attr-defined]
        if payload is None:
            self.send_error(404, f"Unknown virtual televiewer payload: {name}")
            return
        self._send_json(payload)

    def _serve_json_route(self, name: str) -> None:
        route = self.server.json_routes.get(name)  # type: ignore[attr-defined]
        if route is None:
            self.send_error(404, f"Unknown JSON route: {name}")
            return
        getter, _writer = route
        try:
            payload = getter()
        except Exception as exc:
            self.send_error(500, str(exc))
            return
        self._send_json(payload)


class TeleviewerWebServer:
    """Serve the static viewer and selected dataset folders on localhost."""

    def __init__(self, web_root: Path):
        self.web_root = Path(web_root).resolve()
        self.mounts: Dict[str, Path] = {"web": self.web_root}
        self.virtual_payloads: Dict[str, Any] = {}
        self.json_routes: Dict[str, tuple[JsonGetter, JsonWriter | None]] = {}
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: Thread | None = None
        self.port: int | None = None

    def add_mount(self, name: str, path: Path) -> None:
        self.mounts[name] = Path(path).resolve()
        if self._httpd is not None:
            self._httpd.mounts = dict(self.mounts)  # type: ignore[attr-defined]

    def add_json(self, name: str, payload: Any) -> str:
        safe_name = name.replace("\\", "/").strip("/")
        self.virtual_payloads[safe_name] = payload
        if self._httpd is not None:
            self._httpd.virtual_payloads = dict(self.virtual_payloads)  # type: ignore[attr-defined]
        return self.mounted_path("_virtual", safe_name)

    def add_json_route(
        self,
        name: str,
        getter: JsonGetter,
        writer: JsonWriter | None = None,
    ) -> str:
        safe_name = name.replace("\\", "/").strip("/")
        self.json_routes[safe_name] = (getter, writer)
        if self._httpd is not None:
            self._httpd.json_routes = dict(self.json_routes)  # type: ignore[attr-defined]
        return self.mounted_path("_json", safe_name)

    def start(self) -> None:
        if self._httpd is not None:
            return
        handler = partial(_MountedRequestHandler)
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._httpd.mounts = dict(self.mounts)  # type: ignore[attr-defined]
        self._httpd.virtual_payloads = dict(self.virtual_payloads)  # type: ignore[attr-defined]
        self._httpd.json_routes = dict(self.json_routes)  # type: ignore[attr-defined]
        self.port = int(self._httpd.server_address[1])
        self._thread = Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        self._httpd = None
        self._thread = None
        self.port = None

    def mounted_path(self, mount: str, relative_path: str | Path = "") -> str:
        relative = str(relative_path).replace("\\", "/").strip("/")
        parts = [quote(mount)]
        if relative:
            parts.extend(quote(part) for part in relative.split("/") if part)
        return "/" + "/".join(parts)

    def viewer_url(self, params: Mapping[str, object] | None = None) -> str:
        self.start()
        assert self.port is not None
        query = urlencode(params or {})
        base = f"http://127.0.0.1:{self.port}/web/index.html"
        return f"{base}?{query}" if query else base
