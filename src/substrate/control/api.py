"""The HTTP API of the control plane.

The API validates, authorises and records; it never computes. A request to run something becomes a run in the registry whose tasks workers pick up. What it
takes care over:

* **Authentication** by bearer token (an API key); the principal, with its tenant and role, comes from the registry and nothing else in the request is believed.
* **Authorisation** per route (the action it needs) and per tenant (every lookup is filtered by the caller's tenant, so another tenant's identifiers answer 404).
* **Bounded input**: request bodies are capped, bodies are parsed into models that reject unknown fields, and counts and sizes are checked by the same limits as everywhere.
* **Errors that say what is wrong with the request and nothing about the server**: no stack traces, no paths, no SQL.
* **Safe responses**: `nosniff`, no caching, a content-security policy that allows nothing, artifacts served with their stored type and as attachments unless they are JSON.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .. import __version__
from ..errors import (AmbiguousPathError, DatasetUnavailableError, NoPathError, QCError, ResourceLimitError, SubstrateError, ValidationError)
from .artifacts import ArtifactError, Artifacts, encode_nonfinite
from .fabric.client import FetchError
from .orchestrator import Orchestrator
from .registry import Registry
from .security import Action, AuthorizationError, Principal, QuotaExceededError

MAX_BODY_BYTES = 1_000_000
log = logging.getLogger("substrate.control.api")


class SafeJSONResponse(JSONResponse):
    """JSON that survives NaN and infinity (written as tagged values, as in stored artifacts) instead of failing to render."""

    def render(self, content: Any) -> bytes:
        import json
        return json.dumps(encode_nonfinite(content), ensure_ascii=False, allow_nan=False, separators=(",", ":"), default=str).encode("utf-8")


class BodyLimit:
    """Refuses a request whose body is larger than `limit` bytes, whether it says so in Content-Length or only turns out to be when it is read."""

    def __init__(self, app: ASGIApp, limit: int = MAX_BODY_BYTES):
        self.app, self.limit = app, limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > self.limit):
            return await self._refuse(scope, receive, send)
        seen, refused, started = 0, False, False

        async def limited() -> Message:
            """Passes the body through until it is too long; then the application is told the client went away, so it stops reading, and nothing
            it answers after that is sent (the framework may turn the interrupted read into an error response of its own, which would hide the 413)."""
            nonlocal seen, refused
            message = await receive()
            if message["type"] == "http.request" and not refused:
                seen += len(message.get("body", b""))
                if seen > self.limit:
                    refused = True
            return {"type": "http.disconnect"} if refused else message

        async def guarded(message: Message) -> None:
            nonlocal started
            if refused:
                return
            started = started or message["type"] == "http.response.start"
            await send(message)

        try:
            await self.app(scope, limited, guarded)
        except Exception:
            if not refused:
                raise
        if refused and not started:
            await self._refuse(scope, receive, send)

    async def _refuse(self, scope: Scope, receive: Receive, send: Send) -> None:
        await SafeJSONResponse({"detail": f"the request body is larger than {self.limit:,} bytes"}, status_code=413)(scope, receive, send)


class SecurityHeaders:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                present = {k.lower() for k, _ in headers}
                for name, value in ((b"x-content-type-options", b"nosniff"), (b"cache-control", b"no-store"), (b"referrer-policy", b"no-referrer"),
                                    (b"content-security-policy", b"default-src 'none'; frame-ancestors 'none'"), (b"x-frame-options", b"DENY")):
                    if name not in present:
                        headers.append((name, value))
            await send(message)

        await self.app(scope, receive, with_headers)


# -- request bodies ---------------------------------------------------------------------------------------------------------------------------
class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetPin(Strict):
    id: str = Field(min_length=1, max_length=80)
    version: int = Field(ge=1)


class ExperimentIn(Strict):
    name: str = Field(min_length=1, max_length=200)
    spec: dict
    datasets: list[DatasetPin] = Field(default_factory=list, max_length=100)


class RunIn(Strict):
    experiment_id: str = Field(pattern=r"^EXP-\d{6,}$")
    seed: int | None = None
    n_samples: int | None = Field(default=None, ge=0)


class IngestIn(Strict):
    adapter: str = Field(min_length=1, max_length=80)
    query: dict = Field(default_factory=dict, max_length=10)


class TransferIn(Strict):
    name: str = Field(default="transfer study", min_length=1, max_length=200)
    references: list[str] = Field(min_length=2, max_length=1000)
    window_ev: float | None = Field(default=None, gt=0, le=50)
    sigma_ev: float = Field(default=0.05, gt=0, le=10)


def create_app(registry: Registry, artifacts: Artifacts, orchestrator: Orchestrator, *, metrics=None, fabric=None, max_body: int = MAX_BODY_BYTES) -> FastAPI:
    app = FastAPI(title="substrate control plane", version=__version__, docs_url=None, redoc_url=None, default_response_class=SafeJSONResponse)
    app.add_middleware(SecurityHeaders)
    app.add_middleware(BodyLimit, limit=max_body)
    observer = orchestrator.observer

    # -- errors ------------------------------------------------------------------------------------------------------------------------------
    @app.exception_handler(SubstrateError)
    async def substrate_error(request: Request, error: SubstrateError):
        if isinstance(error, AuthorizationError):
            observer.auth_failed("forbidden")
            return SafeJSONResponse({"detail": str(error)}, status_code=403)
        if isinstance(error, QuotaExceededError):
            return SafeJSONResponse({"detail": str(error), "kind": "quota"}, status_code=429)
        if isinstance(error, ResourceLimitError):
            return SafeJSONResponse({"detail": str(error), "kind": "limit"}, status_code=413)
        if isinstance(error, FetchError):
            return SafeJSONResponse({"detail": str(error), "kind": "upstream"}, status_code=502)
        if isinstance(error, (DatasetUnavailableError, ArtifactError)):
            return SafeJSONResponse({"detail": str(error)}, status_code=404)
        if isinstance(error, (ValidationError, NoPathError, AmbiguousPathError)):
            return SafeJSONResponse({"detail": str(error)}, status_code=422)
        if isinstance(error, QCError):
            return SafeJSONResponse({"detail": "the quantum-chemistry program is not available"}, status_code=503)
        return SafeJSONResponse({"detail": str(error)}, status_code=400)

    @app.exception_handler(LookupError)
    async def not_found(request: Request, error: LookupError):
        return SafeJSONResponse({"detail": str(error.args[0]) if error.args else "not found"}, status_code=404)

    @app.exception_handler(Exception)
    async def unexpected(request: Request, error: Exception):
        log.exception("unhandled error in %s %s", request.method, request.url.path)
        return SafeJSONResponse({"detail": "internal error"}, status_code=500)

    # -- authentication ------------------------------------------------------------------------------------------------------------------------
    def current(authorization: str | None = Header(default=None)) -> Principal:
        scheme, _, token = (authorization or "").partition(" ")
        principal = registry.authenticate(token.strip()) if scheme.lower() == "bearer" and token.strip() else None
        if principal is None:
            observer.auth_failed("missing" if not authorization else "invalid")
            raise HTTPException(status_code=401, detail="a valid API key is required", headers={"WWW-Authenticate": "Bearer"})
        return principal

    def allowed(action: Action):
        def check(principal: Principal = Depends(current)) -> Principal:
            principal.require(action)
            return principal
        return check

    def found(value, what: str):
        if value is None:
            raise LookupError(what)
        return value

    # -- health and metrics -----------------------------------------------------------------------------------------------------------------------
    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "version": __version__}

    @app.get("/readyz")
    def readyz():
        import sqlalchemy as sa
        with registry.engine.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
        return {"status": "ready"}

    @app.get("/metrics")
    def metrics_endpoint(principal: Principal = Depends(allowed(Action.METRICS_READ))):
        if metrics is None:
            raise LookupError("metrics are not enabled")
        return Response(metrics.render(), media_type="text/plain; version=0.0.4")

    # -- models ---------------------------------------------------------------------------------------------------------------------------------
    @app.get("/models")
    def models(principal: Principal = Depends(allowed(Action.EXPERIMENT_READ))):
        reg = orchestrator.pipeline.registry
        return {"engines": [{"name": e.name, "version": e.version, "scale": e.scale.label, "kinds": list(e.kinds), "approximations": list(e.approximations)}
                            for e in reg.engines], "translators": [{"name": t.name, "version": t.version, "source": t.source.label, "target": t.target.label,
                                                                    "target_kind": t.target_kind, "approximations": list(t.approximations)} for t in reg.translators]}

    # -- experiments ----------------------------------------------------------------------------------------------------------------------------
    @app.post("/experiments", status_code=201)
    def create_experiment(body: ExperimentIn, principal: Principal = Depends(allowed(Action.EXPERIMENT_CREATE))):
        orchestrator.check_spec(principal, body.spec, [p.model_dump() for p in body.datasets])
        return registry.add_experiment(principal, body.name, body.spec, [p.model_dump() for p in body.datasets])

    @app.get("/experiments")
    def list_experiments(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0), principal: Principal = Depends(allowed(Action.EXPERIMENT_READ))):
        return {"experiments": registry.list_experiments(principal, limit, offset)}

    @app.get("/experiments/{experiment_id}")
    def get_experiment(experiment_id: str, principal: Principal = Depends(allowed(Action.EXPERIMENT_READ))):
        return found(registry.get_experiment(principal, experiment_id), f"no experiment {experiment_id}")

    # -- runs -----------------------------------------------------------------------------------------------------------------------------------
    @app.post("/runs", status_code=202)
    def create_run(body: RunIn, principal: Principal = Depends(allowed(Action.RUN_CREATE))):
        return orchestrator.submit(principal, body.experiment_id, seed=body.seed, n_samples=body.n_samples)

    @app.get("/runs")
    def list_runs(experiment_id: str | None = Query(None, pattern=r"^EXP-\d{6,}$"), limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
                  principal: Principal = Depends(allowed(Action.RUN_READ))):
        return {"runs": registry.list_runs(principal, experiment_id, limit, offset)}

    @app.get("/runs/{run_id}")
    def get_run(run_id: str, principal: Principal = Depends(allowed(Action.RUN_READ))):
        return found(orchestrator.describe(principal, run_id), f"no run {run_id}")

    @app.post("/runs/{run_id}/cancel")
    def cancel_run(run_id: str, principal: Principal = Depends(allowed(Action.RUN_CANCEL))):
        return {"cancelled_tasks": orchestrator.cancel(principal, run_id)}

    @app.get("/runs/{run_id}/artifacts")
    def run_artifacts(run_id: str, principal: Principal = Depends(allowed(Action.ARTIFACT_READ))):
        found(registry.get_run(principal, run_id), f"no run {run_id}")
        return {"artifacts": [{k: a[k] for k in ("sha256", "name", "media_type", "size")} for a in registry.run_artifacts(principal, run_id)]}

    @app.get("/runs/{run_id}/lineage")
    def run_lineage(run_id: str, principal: Principal = Depends(allowed(Action.RUN_READ))):
        found(registry.get_run(principal, run_id), f"no run {run_id}")
        chain = f"lineage:{run_id}"
        intact, bad = registry.verify_chain(chain, principal.tenant_id)
        return {"run": run_id, "intact": intact, "first_bad_event": bad,
                "events": [{k: e[k] for k in ("seq", "ts", "kind", "payload", "hash")} for e in registry.events(chain, principal.tenant_id)]}

    # -- reproducing ------------------------------------------------------------------------------------------------------------------------------
    @app.post("/reproduce/{run_id}", status_code=202)
    def reproduce(run_id: str, principal: Principal = Depends(allowed(Action.RUN_REPRODUCE))):
        return orchestrator.reproduce(principal, run_id)

    @app.get("/reproduce/{run_id}")
    def reproductions(run_id: str, principal: Principal = Depends(allowed(Action.RUN_READ))):
        original = found(registry.get_run(principal, run_id), f"no run {run_id}")
        rows = []
        for run in registry.reproductions_of(principal, run_id):
            status = (run["status"] or {}).get("reproducibility")
            rows.append({"run": run["id"], "state": run["state"], "reproducibility": status, "result_digest": run["result_digest"]})
        return {"original": run_id, "result_digest": original["result_digest"], "reproductions": rows}

    # -- artifacts --------------------------------------------------------------------------------------------------------------------------------
    @app.get("/artifacts/{sha256}")
    def artifact(sha256: str, principal: Principal = Depends(allowed(Action.ARTIFACT_READ))):
        data, meta = artifacts.read(principal, sha256)
        headers = {"ETag": f'"{sha256}"', "Content-Disposition": f'attachment; filename="{sha256[:12]}"'}
        if meta["media_type"] == "application/json":
            headers.pop("Content-Disposition")
        return Response(data, media_type=meta["media_type"], headers=headers)

    # -- transfers --------------------------------------------------------------------------------------------------------------------------------
    @app.post("/transfers", status_code=202)
    def create_transfer(body: TransferIn, principal: Principal = Depends(allowed(Action.TRANSFER_RUN))):
        return orchestrator.submit_transfer(principal, body.name, body.references, window_ev=body.window_ev, sigma_ev=body.sigma_ev)

    @app.get("/transfers/{run_id}")
    def get_transfer(run_id: str, principal: Principal = Depends(allowed(Action.RUN_READ))):
        description = found(orchestrator.describe(principal, run_id), f"no run {run_id}")
        return {**description, "matrix": orchestrator.transfer_result(principal, run_id)}

    # -- datasets ---------------------------------------------------------------------------------------------------------------------------------
    @app.get("/datasets")
    def list_datasets(principal: Principal = Depends(allowed(Action.DATASET_READ))):
        return {"datasets": registry.list_datasets()}

    @app.get("/datasets/adapters")
    def dataset_adapters(principal: Principal = Depends(allowed(Action.DATASET_READ))):
        return {"adapters": fabric.describe() if fabric is not None else []}

    @app.post("/datasets/ingest", status_code=201)
    def ingest_dataset(body: IngestIn, principal: Principal = Depends(allowed(Action.DATASET_INGEST))):
        if fabric is None:
            raise LookupError("dataset ingestion is not enabled")
        return fabric.ingest(principal, body.adapter, body.query)

    @app.get("/datasets/{dataset_id}")
    def get_dataset(dataset_id: str, principal: Principal = Depends(allowed(Action.DATASET_READ))):
        return found(registry.get_dataset(dataset_id), f"no dataset {dataset_id}")

    @app.get("/datasets/{dataset_id}/{version}")
    def get_dataset_version(dataset_id: str, version: int, principal: Principal = Depends(allowed(Action.DATASET_READ))):
        return found(registry.get_dataset(dataset_id, version), f"no dataset {dataset_id} version {version}")

    # -- audit ----------------------------------------------------------------------------------------------------------------------------------------
    @app.get("/audit")
    def audit(limit: int = Query(100, ge=1, le=1000), principal: Principal = Depends(allowed(Action.AUDIT_READ))):
        chain = f"audit:{principal.tenant_id}"
        intact, bad = registry.verify_chain(chain, principal.tenant_id)
        events = registry.events(chain, principal.tenant_id)[-limit:]
        return {"intact": intact, "first_bad_event": bad, "events": [{k: e[k] for k in ("seq", "ts", "kind", "payload", "hash")} for e in events]}

    return app
