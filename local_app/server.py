"""Nocturne Video local server: HTTP API + dashboard hosting.

Stdlib only. Binds 127.0.0.1:8788. Routes:
  GET  /                      dashboard
  GET  /api/bootstrap         app/endpoint/s3 status
  POST /api/generate          submit a generation (JSON, base64 media inline)
  POST /api/healthcheck       submit worker healthcheck (stages volume)
  GET  /api/jobs              recent jobs
  POST /api/jobs/<id>/cancel  cancel a queued/running job
  GET  /api/generations?q=&task=&tag=&sort=&limit=&offset=   gallery + search
  POST /api/generations/<id>/tags    replace an item's tags
  POST /api/generations/<id>/favorite  toggle the star
  GET  /api/generations/<id>/reuse   original submission spec for refilling
  POST /api/generations/<id>/sync   re-download from S3
  POST /api/sync-missing      sync every generation missing its local file
  GET  /api/s3/status         bucket connectivity
  GET  /api/templates         prompt sections per task
  GET  /video/<gen_id>        stream the local MP4 (Range-capable)
"""

from __future__ import annotations

import json
import mimetypes
import re
import subprocess
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import config, database, runpod_client, service

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
mime = mimetypes.MimeTypes()
mime.add_type("video/mp4", ".mp4")


class NocturneServer:
    def __init__(self):
        self.db = database.Database(config.DB_PATH)
        self.svc = service.Service(self.db)

    # -- handlers -------------------------------------------------------------

    def handle(self, method: str, path: str, query: dict, body: bytes) -> tuple[int, dict | bytes, str]:
        if path == "/api/bootstrap" and method == "GET":
            return 200, self.svc.bootstrap(), "json"
        if path == "/api/templates" and method == "GET":
            return 200, {"tasks": service.TASKS, "sections": service.PROMPT_SECTIONS}, "json"
        if path == "/api/generate" and method == "POST":
            payload = json.loads(body.decode("utf-8"))
            return 200, self.svc.submit(payload), "json"
        if path == "/api/healthcheck" and method == "POST":
            return 200, self.svc.healthcheck(), "json"
        if path == "/api/jobs" and method == "GET":
            return 200, {"jobs": self.db.list_jobs(int(query.get("limit", ["50"])[0]))}, "json"
        if path == "/api/s3/status" and method == "GET":
            return 200, self.svc.s3_status(), "json"
        if path == "/api/sync-missing" and method == "POST":
            return 200, self.svc.sync_missing(), "json"
        if path == "/api/generations" and method == "GET":
            rows = self.db.search_generations(
                q=query.get("q", [""])[0],
                task=query.get("task", [""])[0],
                tag=query.get("tag", [""])[0],
                sort=query.get("sort", ["newest"])[0],
                limit=min(int(query.get("limit", ["60"])[0]), 200),
                offset=int(query.get("offset", ["0"])[0]),
            )
            for row in rows:
                row["meta_json"] = json.loads(row.get("meta_json") or "{}")
                row["tags"] = json.loads(row.get("tags") or "[]")
            return 200, {"generations": rows, "stats": self.db.stats(),
                         "tags": self.db.tag_counts()}, "json"
        match = re.match(r"^/api/generations/([^/]+)/tags$", path)
        if match and method == "POST":
            payload = json.loads(body.decode("utf-8") or "{}")
            return 200, self.svc.set_tags(match.group(1), payload.get("tags") or []), "json"
        match = re.match(r"^/api/generations/([^/]+)/favorite$", path)
        if match and method == "POST":
            payload = json.loads(body.decode("utf-8") or "{}")
            return 200, self.svc.set_favorite(match.group(1), bool(payload.get("favorited"))), "json"
        match = re.match(r"^/api/generations/([^/]+)/reuse$", path)
        if match and method == "GET":
            return 200, self.svc.reuse(match.group(1)), "json"
        match = re.match(r"^/api/jobs/([^/]+)/cancel$", path)
        if match and method == "POST":
            return 200, self.svc.cancel(match.group(1)), "json"
        match = re.match(r"^/api/generations/([^/]+)/sync$", path)
        if match and method == "POST":
            return 200, self.svc.sync_generation(match.group(1)), "json"
        match = re.match(r"^/video/([^/]+)$", path)
        if match and method == "GET":
            gen = self.db.get_generation(match.group(1))
            if not gen or not gen.get("video_path"):
                return 404, {"error": "video not available locally; sync first"}, "json"
            return "FILE", Path(gen["video_path"]), "file"
        if path == "/" and method == "GET":
            return "FILE", WEB_DIR / "index.html", "file"
        if method == "GET":
            asset = (WEB_DIR / path.lstrip("/")).resolve()
            if asset.is_file() and str(asset).startswith(str(WEB_DIR)):
                return "FILE", asset, "file"
        return 404, {"error": f"no route for {method} {path}"}, "json"


def make_handler(server_state: NocturneServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # noqa: N802 — quiet
            pass

        def _send_json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path, range_header: str | None) -> None:
            size = path.stat().st_size
            ctype = mime.guess_type(str(path))[0] or "application/octet-stream"
            start, end = 0, size - 1
            if range_header and (m := re.match(r"bytes=(\d*)-(\d*)$", range_header.strip())):
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = int(m.group(2))
            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            end = min(end, size - 1)
            self.send_response(206 if (start > 0 or end < size - 1) else 200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            if start > 0 or end < size - 1:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as fh:
                fh.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    block = fh.read(min(1024 * 256, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)

        def _dispatch(self, method: str) -> None:
            parsed = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(parsed.query)
            body = b""
            if method == "POST":
                length = int(self.headers.get("Content-Length") or 0)
                if length > config.MAX_BODY_BYTES:
                    self._send_json(413, {"error": "payload too large"})
                    return
                body = self.rfile.read(length) if length else b""
            try:
                code, payload, kind = server_state.handle(method, parsed.path, query, body)
            except service.ServiceError as exc:
                self._send_json(400, {"error": str(exc)})
                return
            except runpod_client.RunpodClientError as exc:
                self._send_json(502, {"error": str(exc)})
                return
            except json.JSONDecodeError as exc:
                self._send_json(400, {"error": f"invalid JSON: {exc}"})
                return
            except Exception as exc:  # noqa: BLE001
                self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            if kind == "file":
                file_path, range_header = payload, self.headers.get("Range")
                try:
                    self._send_file(file_path, range_header)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            self._send_json(code, payload)

        def do_GET(self):  # noqa: N802
            self._dispatch("GET")

        def do_POST(self):  # noqa: N802
            self._dispatch("POST")

    return Handler


def main(open_browser: bool = False) -> None:
    state = NocturneServer()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        httpd = ThreadingHTTPServer((config.HOST, config.PORT), make_handler(state))
    except OSError as exc:
        if getattr(exc, "errno", None) == 48:
            print(f"Port {config.PORT} is already in use: another Nocturne Video "
                  f"instance is running at http://{config.HOST}:{config.PORT} "
                  f"(stop it first, or set NOCTURNE_PORT to use a different port).",
                  flush=True)
            raise SystemExit(1)
        raise
    url = f"http://{config.HOST}:{config.PORT}"
    print(f"Nocturne Video dashboard: {url} "
          f"(endpoint: {config.endpoint_id() or 'not configured'})", flush=True)
    if open_browser:
        subprocess.Popen(["open", url])
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        state.svc.shutdown()


if __name__ == "__main__":
    main()
