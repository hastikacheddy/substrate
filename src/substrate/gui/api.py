"""The GUI's endpoints, apart from any transport: what each URL does with an `App`, and what a failure answers.

`server.py` puts these behind an HTTP server on this machine; the static demo (deploy/huggingface/static_space) puts the same functions behind a web worker in the visitor's own
browser, where there is no server and no thread. Nothing here reads a socket, a header or a thread: a request is (method, path, query, JSON body) and the answer is (status, JSON body).
"""
from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse

from ..errors import SubstrateError
from . import serialize as ser
from .app import App, GuiError


def get(app: App, path: str, q: dict):
    if path == "/api/status":
        return app.status()
    if path == "/api/experiments":
        return {"experiments": app.list_experiments()}
    if path == "/api/experiment":
        return app.experiment(q.get("path", ""))
    if path == "/api/job":
        try:
            since = int(q.get("since", 0))
        except ValueError:
            raise GuiError("since must be an integer") from None
        return app.jobs.get(q.get("id", "")).to_dict(since)
    if path == "/api/references":
        return {"references": app.references_listing()}
    if path == "/api/transfer":
        names = [n for n in q.get("refs", "").split(",") if n]
        names = app._check_names(names)
        cached = app.cached_transfer(names)
        return {"cached": cached is not None, "result": cached}
    raise GuiError(f"no such endpoint {path}", 404)


def post(app: App, path: str, b: dict):
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


def answer(app: App, method: str, target: str, body_text: str | None = None) -> tuple[int, str]:
    """The status and the JSON text the server would answer `method target` with. `target` is a path with its query string; `body_text` is a POST's body."""
    try:
        url = urlparse(target)
        if not url.path.startswith("/api/"):
            raise GuiError("not found", 404)
        if method == "GET":
            payload = get(app, url.path, {k: v[0] for k, v in parse_qs(url.query).items()})
        elif method == "POST":
            try:
                body = json.loads(body_text or "{}")
                if not isinstance(body, dict):
                    raise ValueError("the body must be a JSON object")
            except ValueError as error:
                raise GuiError(f"bad JSON: {error}") from None
            payload = post(app, url.path, body)
        else:
            raise GuiError(f"{method} is not supported", 405)
        return 200, json.dumps(ser.clean(payload), allow_nan=False)
    except GuiError as error:
        return error.status, json.dumps({"error": str(error)})
    except SubstrateError as error:
        return 400, json.dumps({"error": str(error)})
    except Exception as error:                                          # a bug or an environment failure: reported, never swallowed
        return 500, json.dumps({"error": f"{type(error).__name__}: {error}"})
