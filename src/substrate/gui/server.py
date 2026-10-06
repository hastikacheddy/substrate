"""The GUI's HTTP layer: a small JSON API over `App` and the static page. Standard library only.

It binds to the loopback address only, and refuses requests whose Host or Origin is not this server (so a web page open in another tab
cannot drive it, nor can a DNS-rebinding trick), requires JSON for every POST, and serves only the files in `static/` by exact name.
It runs the experiments the project's own `experiments/` folder lists and nothing else: a path is resolved and refused unless it stays
inside that folder.
"""
from __future__ import annotations

import json
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..errors import SubstrateError
from . import serialize as ser
from .app import App, GuiError

STATIC = Path(__file__).resolve().parent / "static"
MAX_BODY = 1_000_000
CSP = "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; script-src 'self'; base-uri 'none'; form-action 'none'"


class Handler(BaseHTTPRequestHandler):
    server_version = "substrate-gui"
    protocol_version = "HTTP/1.1"

    # -- plumbing -------------------------------------------------------------------------------------------------------------
    @property
    def app(self) -> App:
        return self.server.app

    def log_message(self, fmt, *args):                    # quiet: the page polls
        pass

    def _send(self, status: int, body: bytes, content_type: str, extra: dict | None = None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", CSP)
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, status: int = 200, extra: dict | None = None):
        self._send(status, json.dumps(ser.clean(payload), allow_nan=False).encode(), "application/json; charset=utf-8", extra)

    def _error(self, status: int, message: str, extra: dict | None = None):
        self._json({"error": message}, status, extra)

    def _allowed(self) -> bool:
        """The request must be addressed to this server, from this server's own page (or from a non-browser client)."""
        port = self.server.server_address[1]
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host", "") not in hosts:
            self._error(403, "unexpected Host header")
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin not in {f"http://{h}" for h in hosts}:
            self._error(403, "cross-origin requests are refused")
            return False
        return True

    # -- GET --------------------------------------------------------------------------------------------------------------------------
    def do_GET(self):
        if not self._allowed():
            return
        url = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path.startswith("/api/"):
                return self._json(self._api_get(url.path, query))
            return self._static(url.path)
        except GuiError as error:
            self._error(error.status, str(error))
        except SubstrateError as error:
            self._error(400, str(error))
        except Exception as error:
            self._error(500, f"{type(error).__name__}: {error}")

    def _api_get(self, path: str, q: dict):
        if path == "/api/status":
            return self.app.status()
        if path == "/api/experiments":
            return {"experiments": self.app.list_experiments()}
        if path == "/api/experiment":
            return self.app.experiment(q.get("path", ""))
        if path == "/api/job":
            try:
                since = int(q.get("since", 0))
            except ValueError:
                raise GuiError("since must be an integer") from None
            return self.app.jobs.get(q.get("id", "")).to_dict(since)
        if path == "/api/references":
            return {"references": self.app.references_listing()}
        if path == "/api/transfer":
            names = [n for n in q.get("refs", "").split(",") if n]
            names = self.app._check_names(names)
            cached = self.app.cached_transfer(names)
            return {"cached": cached is not None, "result": cached}
        raise GuiError(f"no such endpoint {path}", 404)

    def _static(self, path: str):
        name = "index.html" if path in ("/", "") else path.removeprefix("/static/") if path.startswith("/static/") else None
        target = STATIC / name if name and "/" not in name and "\\" not in name else None
        if target is None or not target.is_file():
            return self._error(404, "not found")
        kind = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self._send(200, target.read_bytes(), f"{kind}; charset=utf-8" if kind.startswith("text/") or kind.endswith("javascript") else kind)

    # -- POST -------------------------------------------------------------------------------------------------------------------------
    def do_POST(self):
        # Read the body before judging the request, so a refusal never leaves an unread body on a kept-alive connection.
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            self.close_connection = True
            return self._error(400, "bad Content-Length", {"Connection": "close"})
        if not 0 <= length <= MAX_BODY:
            self.close_connection = True                       # too large to read: answer and hang up
            return self._error(413, "request too large", {"Connection": "close"})
        raw = self.rfile.read(length) if length else b""
        if not self._allowed():
            return
        if not self.headers.get("Content-Type", "").startswith("application/json"):
            return self._error(415, "POST bodies must be application/json")
        try:
            body = json.loads(raw or b"{}")
            if not isinstance(body, dict):
                raise ValueError("the body must be a JSON object")
        except ValueError as error:
            return self._error(400, f"bad JSON: {error}")
        try:
            return self._json(self._api_post(urlparse(self.path).path, body))
        except GuiError as error:
            self._error(error.status, str(error))
        except SubstrateError as error:
            self._error(400, str(error))
        except Exception as error:
            self._error(500, f"{type(error).__name__}: {error}")

    def _api_post(self, path: str, b: dict):
        app = self.app
        if path == "/api/run":
            return app.start_run(b.get("path", ""), b.get("overrides"), b.get("samples"), b.get("seed"))
        if path == "/api/transfer/run":
            names, recompute = app._check_names(b.get("refs")), bool(b.get("recompute"))
            return app.start_job("transfer study", lambda log: app.run_transfer(names, recompute, log))
        if path == "/api/transfer/rates":
            names, barrier = app._check_names(b.get("refs")), b.get("barrier", 0.4)
            return app.start_job("downstream rates", lambda log: app.run_rates(names, barrier, log))
        if path == "/api/transfer/learning":
            names = app._check_names(b.get("refs"))
            target, source = b.get("target"), b.get("source") or None
            sigmas, counts = b.get("sigmas", [0.3]), b.get("counts", [0, 1, 2, 3, 6])
            return app.start_job("learning curve", lambda log: app.run_learning(names, target, source, sigmas, counts, log))
        raise GuiError(f"no such endpoint {path}", 404)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, address, app: App):
        super().__init__(address, Handler)
        self.app = app


def make_server(app: App | None = None, port: int = 8765) -> Server:
    """A server bound to 127.0.0.1 (port 0 picks a free one)."""
    return Server(("127.0.0.1", port), app or App())


def serve(port: int = 8765, root=None, open_browser: bool = True) -> int:
    app = App(root)
    try:
        server = make_server(app, port)
    except OSError as error:
        print(f"error: cannot listen on 127.0.0.1:{port} ({error}); try --port", flush=True)
        return 1
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"substrate GUI on {url}  (project: {app.root})  Ctrl+C to stop", flush=True)
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
    finally:
        server.server_close()
    return 0
