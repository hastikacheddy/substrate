"""Seeing what the platform is doing: Prometheus metrics, OpenTelemetry traces and structured logs, all driven by the orchestrator's `Observer` hooks.

* **Metrics.** Counters and histograms are labelled only with things that have a handful of values (task kind, queue, outcome, status axis and state): never with a
  run, task or user identifier, which would grow without bound. Queue depth, the number of active runs and what finished in the last day are read from the
  database at scrape time, so they are correct whichever process was last to touch them. The counters and the task-time histogram live in the process that
  recorded the event: in a deployment where workers are separate processes from the API, the API's /metrics shows the submissions and refusals it saw,
  and the `*_recent` gauges are what show the workers' results.
* **Traces.** Every task becomes a span. The trace id is derived from the run id, so the tasks of one run, executed by different workers in different processes,
  appear as one trace without anything having to pass a trace context between them.
* **Logs.** One JSON object per line with the run, task and trace ids. API tokens and anything that looks like a credential are redacted before a record is written.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Iterable

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from prometheus_client.core import GaugeMetricFamily
from opentelemetry import trace
from opentelemetry.trace import NonRecordingSpan, SpanContext, Status, StatusCode, TraceFlags

from .. import __version__
from .orchestrator import Observer
from .registry import Registry
from .status import AXES, RunStatus
from .taskqueue import ClaimedTask, TaskQueue

TASK_BUCKETS = (0.01, 0.05, 0.1, 0.5, 1, 2, 5, 15, 60, 300, 1800, 7200)
RECENT_WINDOW_S = 86_400.0          # how far back the database-derived "recent" gauges look
RECENT_ROWS = 20_000                # and the most rows one scrape will read for them


class MetricsObserver(Observer):
    """Counters and histograms for runs, tasks and blocked requests, on a registry of their own (so tests and processes do not share global state)."""

    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.runs_submitted = Counter("substrate_runs_submitted_total", "Runs accepted.", registry=r)
        self.runs_finished = Counter("substrate_runs_finished_total", "Runs that ended, by final state and by the worst of their six status axes.", ["state", "overall"], registry=r)
        self.run_axis = Counter("substrate_run_axis_total", "Finished runs by status axis and the state that axis ended in.", ["axis", "state"], registry=r)
        self.tasks_started = Counter("substrate_tasks_started_total", "Tasks taken by a worker.", ["kind", "queue"], registry=r)
        self.tasks_finished = Counter("substrate_tasks_finished_total", "Task attempts by how they ended; `axis` is the status axis of a failure.", ["kind", "outcome", "axis"], registry=r)
        self.task_seconds = Histogram("substrate_task_seconds", "Wall time of a task attempt.", ["kind"], buckets=TASK_BUCKETS, registry=r)
        self.blocked_total = Counter("substrate_blocked_total", "Requests or tasks stopped by a limit, a quota, a policy or an authorisation check.", ["reason"], registry=r)
        self.auth_failures = Counter("substrate_auth_failures_total", "Requests that could not be authenticated or were not permitted.", ["reason"], registry=r)

    def run_submitted(self, run: dict) -> None:
        self.runs_submitted.inc()

    def task_started(self, task: ClaimedTask) -> None:
        self.tasks_started.labels(task.kind, task.queue).inc()

    def task_finished(self, task: ClaimedTask, seconds: float, outcome: str, axis: str | None) -> None:
        self.tasks_finished.labels(task.kind, outcome, axis or "none").inc()
        self.task_seconds.labels(task.kind).observe(seconds)

    def run_finished(self, run: dict, status: RunStatus) -> None:
        self.runs_finished.labels(run["state"], status.overall().value).inc()
        for axis, value in status.axes.items():
            self.run_axis.labels(axis, value.state.value).inc()

    def blocked_request(self, reason: str) -> None:
        self.blocked_total.labels(reason).inc()

    def blocked(self, kind: str) -> None:                       # the Observer hook
        self.blocked_request(kind)

    def auth_failed(self, reason: str) -> None:
        self.auth_failures.labels(reason).inc()

    def render(self) -> bytes:
        return generate_latest(self.registry)


class LiveGauges:
    """Queue depth per queue and active runs per state, read from the database whenever the metrics are scraped."""

    def __init__(self, registry: Registry, queue: TaskQueue | None = None):
        self.registry, self.queue = registry, queue or TaskQueue(registry)

    def collect(self) -> Iterable[GaugeMetricFamily]:
        depth = GaugeMetricFamily("substrate_queue_depth", "Tasks ready to be taken, by queue.", labels=["queue"])
        for queue, n in sorted(self.queue.depth().items()):
            depth.add_metric([queue], n)
        yield depth
        active = GaugeMetricFamily("substrate_runs_active", "Runs that have not ended, by state.", labels=["state"])
        for state, n in sorted(self._active_runs().items()):
            active.add_metric([state], n)
        yield active
        info = GaugeMetricFamily("substrate_build_info", "The version of the software serving these metrics.", labels=["version"])
        info.add_metric([__version__], 1)
        yield info
        yield from self._recent()

    def _active_runs(self) -> dict[str, int]:
        import sqlalchemy as sa
        from . import db
        with self.registry.engine.connect() as conn:
            rows = conn.execute(sa.select(db.runs.c.state, sa.func.count()).where(db.runs.c.state.in_(["queued", "running"])).group_by(db.runs.c.state)).all()
        return {state: n for state, n in rows}

    def _recent(self) -> Iterable[GaugeMetricFamily]:
        """What the workers did, read from the database: a worker is another process, so its own counters never reach the process that serves /metrics.
        These are gauges over a sliding window (not counters: a count over the last day can fall), bounded in what they read (RECENT_ROWS) and in their labels."""
        import sqlalchemy as sa
        from . import db
        cutoff = self.registry.clock() - RECENT_WINDOW_S
        tasks = GaugeMetricFamily("substrate_tasks_recent", "Tasks by state and kind: those not yet finished, and those that finished in the last 24 hours.", labels=["state", "kind"])
        failures = GaugeMetricFamily("substrate_task_failures_recent", "Failed tasks of the last 24 hours by the status axis the failure belongs to.", labels=["axis"])
        seconds = GaugeMetricFamily("substrate_task_seconds_recent", "Wall time of the tasks that succeeded in the last 24 hours, by kind and statistic (mean, max).", labels=["kind", "stat"])
        runs = GaugeMetricFamily("substrate_runs_recent", "Runs that ended in the last 24 hours, by final state.", labels=["state"])
        axes = GaugeMetricFamily("substrate_run_axis_recent", "Runs that ended in the last 24 hours, by status axis and the state that axis ended in.", labels=["axis", "state"])
        with self.registry.engine.connect() as conn:
            open_tasks = conn.execute(sa.select(db.tasks.c.state, db.tasks.c.kind, sa.func.count()).where(db.tasks.c.state.in_(["pending", "ready", "running"]))
                                      .group_by(db.tasks.c.state, db.tasks.c.kind)).all()
            closed = conn.execute(sa.select(db.tasks.c.state, db.tasks.c.kind, db.tasks.c.error, db.tasks.c.seconds)
                                  .where(db.tasks.c.finished_at >= cutoff).order_by(db.tasks.c.finished_at.desc()).limit(RECENT_ROWS)).all()
            ended = conn.execute(sa.select(db.runs.c.state, db.runs.c.status).where(db.runs.c.finished_at >= cutoff).order_by(db.runs.c.finished_at.desc()).limit(RECENT_ROWS)).all()
        by_state: dict[tuple[str, str], int] = {(state, kind): n for state, kind, n in open_tasks}
        failed_axes: dict[str, int] = {}
        durations: dict[str, list[float]] = {}
        for state, kind, error, took in closed:
            by_state[(state, kind)] = by_state.get((state, kind), 0) + 1
            if state == "failed":
                axis = (error or {}).get("axis") if isinstance(error, dict) else None
                failed_axes[axis if axis in AXES else "execution"] = failed_axes.get(axis if axis in AXES else "execution", 0) + 1
            if state == "succeeded" and took is not None:
                durations.setdefault(kind, []).append(float(took))
        for (state, kind), n in sorted(by_state.items()):
            tasks.add_metric([state, kind], n)
        for axis, n in sorted(failed_axes.items()):
            failures.add_metric([axis], n)
        for kind, values in sorted(durations.items()):
            seconds.add_metric([kind, "mean"], sum(values) / len(values))
            seconds.add_metric([kind, "max"], max(values))
        run_states: dict[str, int] = {}
        axis_states: dict[tuple[str, str], int] = {}
        for state, status in ended:
            run_states[state] = run_states.get(state, 0) + 1
            for axis, value in (status or {}).items() if isinstance(status, dict) else ():
                if axis in AXES and isinstance(value, dict) and isinstance(value.get("state"), str):
                    axis_states[(axis, value["state"])] = axis_states.get((axis, value["state"]), 0) + 1
        for state, n in sorted(run_states.items()):
            runs.add_metric([state], n)
        for (axis, state), n in sorted(axis_states.items()):
            axes.add_metric([axis, state], n)
        yield from (tasks, failures, seconds, runs, axes)

    def describe(self):
        return []


def with_live_gauges(metrics: MetricsObserver, registry: Registry, queue: TaskQueue | None = None) -> MetricsObserver:
    metrics.registry.register(LiveGauges(registry, queue))
    return metrics


# -- traces ---------------------------------------------------------------------------------------------------------------------------------
def trace_id_for(run_id: str) -> int:
    """A 128-bit trace id derived from the run id, so every worker computes the same one."""
    return int(hashlib.sha256(run_id.encode()).hexdigest()[:32], 16) or 1


class TracingObserver(Observer):
    """A span per task attempt, all of a run's spans in one trace. `tracer` defaults to the global one; pass your own for tests."""

    provider = None             # set by configure_tracing: the private provider whose exporter needs flushing at exit

    def __init__(self, tracer=None):
        self.tracer = tracer or trace.get_tracer("substrate")
        self._open: dict[str, object] = {}

    def flush(self) -> None:
        if self.provider is not None:
            self.provider.force_flush()

    def shutdown(self) -> None:
        if self.provider is not None:
            self.provider.shutdown()

    def task_started(self, task: ClaimedTask) -> None:
        parent = NonRecordingSpan(SpanContext(trace_id=trace_id_for(task.run_id), span_id=int(hashlib.sha256(task.run_id.encode()).hexdigest()[-16:], 16) or 1,
                                              is_remote=True, trace_flags=TraceFlags(TraceFlags.SAMPLED)))
        span = self.tracer.start_span(f"task {task.kind}", context=trace.set_span_in_context(parent),
                                      attributes={"substrate.run": task.run_id, "substrate.task": task.id, "substrate.task.name": task.name, "substrate.task.kind": task.kind,
                                                  "substrate.queue": task.queue, "substrate.attempt": task.attempts, "substrate.tenant": task.tenant_id})
        self._open[task.id] = span

    def task_finished(self, task: ClaimedTask, seconds: float, outcome: str, axis: str | None) -> None:
        span = self._open.pop(task.id, None)
        if span is None:
            return
        span.set_attribute("substrate.outcome", outcome)
        if axis:
            span.set_attribute("substrate.failure_axis", axis)
        if outcome not in ("succeeded",):
            span.set_status(Status(StatusCode.ERROR, outcome))
        span.end()


def configure_tracing(environ=None, exporter=None) -> "TracingObserver | None":
    """A TracingObserver that sends its spans to the OTLP/HTTP collector named by OTEL_EXPORTER_OTLP_ENDPOINT, or None when that variable is not set (tracing is off
    by default). The endpoint must be an http(s) URL without credentials in it; `exporter` is for tests. The provider is private to the observer, not installed as the
    process-wide default, so nothing else in the process starts emitting spans because of this call."""
    import os
    from urllib.parse import urlsplit
    environ = os.environ if environ is None else environ
    endpoint = (environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or "").strip()
    if not endpoint:
        return None
    parts = urlsplit(endpoint)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password or parts.fragment:
        raise ValueError("OTEL_EXPORTER_OTLP_ENDPOINT must be an http(s) URL with a host and no credentials or fragment")
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    if exporter is None:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError:
            raise ImportError("OTEL_EXPORTER_OTLP_ENDPOINT is set but the OTLP exporter is not installed (pip install 'substrate[otlp]')") from None
        exporter = OTLPSpanExporter(endpoint=endpoint.rstrip("/") + "/v1/traces")
    provider = TracerProvider(resource=Resource.create({"service.name": environ.get("OTEL_SERVICE_NAME", "substrate-worker")}))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    observer = TracingObserver(provider.get_tracer("substrate"))
    observer.provider = provider
    return observer


# -- logs -------------------------------------------------------------------------------------------------------------------------------------
TOKEN = re.compile(r"\bsbs_[A-Za-z0-9]+_[A-Za-z0-9_\-]+")
AUTH_HEADER = re.compile(r"""(?i)\bauthorization\b\s*[=:]\s*(?:(?:bearer|basic|token|digest)\s+)?[^\s'"]+""")
BEARER = re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=\-]{8,}")
SECRETISH = re.compile(r"""(?i)\b(password|passwd|secret|token|api[_-]?key|authorization|bearer)\b\s*[=:]\s*[^\s'"]+""")
SECRET_KEY = re.compile(r"(?i)(password|passwd|secret|token|api[_-]?key|authorization|credential|cookie)")


def redact(text: str) -> str:
    """`text` with API tokens and `name=value` credentials blanked out."""
    text = TOKEN.sub(lambda m: m.group(0).split("_")[0] + "_" + m.group(0).split("_")[1] + "_***", text)
    text = AUTH_HEADER.sub("Authorization=***", text)                                   # a whole header value, whatever its length
    text = BEARER.sub(lambda m: m.group(1) + " ***", text)
    return SECRETISH.sub(lambda m: re.split(r"[=:]", m.group(0), maxsplit=1)[0] + "=***", text)


class JsonFormatter(logging.Formatter):
    """One JSON object per record. Extra fields given to the logger become top-level keys unless their name looks like a credential."""

    RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        entry = {"ts": round(record.created, 3), "level": record.levelname, "logger": record.name, "msg": redact(record.getMessage())}
        for key, value in record.__dict__.items():
            if key in self.RESERVED or key.startswith("_"):
                continue
            entry[key] = "***" if SECRET_KEY.search(key) else (redact(value) if isinstance(value, str) else value)
        if record.exc_info:
            entry["exception"] = redact(self.formatException(record.exc_info))
        return json.dumps(entry, default=str, sort_keys=True)


class LoggingObserver(Observer):
    def __init__(self, logger: logging.Logger | None = None):
        self.log = logger or logging.getLogger("substrate.control")

    def run_submitted(self, run: dict) -> None:
        self.log.info("run submitted", extra={"run": run["id"], "experiment": run["experiment_id"], "trace_id": format(trace_id_for(run["id"]), "032x")})

    def task_started(self, task: ClaimedTask) -> None:
        self.log.info("task started", extra={"run": task.run_id, "task": task.id, "kind": task.kind, "queue": task.queue, "attempt": task.attempts,
                                             "trace_id": format(trace_id_for(task.run_id), "032x")})

    def task_finished(self, task: ClaimedTask, seconds: float, outcome: str, axis: str | None) -> None:
        level = logging.INFO if outcome == "succeeded" else logging.WARNING
        self.log.log(level, "task finished", extra={"run": task.run_id, "task": task.id, "kind": task.kind, "outcome": outcome, "axis": axis or "none",
                                                    "seconds": round(seconds, 4), "trace_id": format(trace_id_for(task.run_id), "032x")})

    def run_finished(self, run: dict, status: RunStatus) -> None:
        self.log.info("run finished", extra={"run": run["id"], "state": run["state"], "overall": status.overall().value,
                                             "axes": {a: s.state.value for a, s in status.axes.items()}, "trace_id": format(trace_id_for(run["id"]), "032x")})

    def blocked(self, kind: str) -> None:
        self.log.warning("blocked", extra={"reason": kind})

    def auth_failed(self, reason: str) -> None:
        self.log.warning("authentication failed", extra={"reason": reason})


class CompositeObserver(Observer):
    """Passes every event to each of several observers."""

    def __init__(self, *observers: Observer):
        self.observers = list(observers)

    def _each(self, name: str, *args) -> None:
        for observer in self.observers:
            getattr(observer, name)(*args)

    def run_submitted(self, run): self._each("run_submitted", run)
    def task_started(self, task): self._each("task_started", task)
    def task_finished(self, task, seconds, outcome, axis): self._each("task_finished", task, seconds, outcome, axis)
    def run_finished(self, run, status): self._each("run_finished", run, status)
    def blocked(self, kind): self._each("blocked", kind)
    def auth_failed(self, reason): self._each("auth_failed", reason)

