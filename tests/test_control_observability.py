"""Metrics, traces and logs: bounded labels, live gauges, one trace per run across workers, and no credential in any log line."""
import io
import json
import logging

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("prometheus_client")
pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

from substrate.control.artifacts import Artifacts, LocalArtifactStore  # noqa: E402
from substrate.control.observability import (CompositeObserver, JsonFormatter, LiveGauges, LoggingObserver, MetricsObserver, TracingObserver,  # noqa: E402
                                             redact, trace_id_for, with_live_gauges)
from substrate.control.orchestrator import Observer, Orchestrator  # noqa: E402
from substrate.control.registry import Registry  # noqa: E402
from substrate.control.security import Role  # noqa: E402
from substrate.control.worker import Worker  # noqa: E402
from substrate.experiment import read_spec  # noqa: E402

from conftest import EXPERIMENTS  # noqa: E402


@pytest.fixture
def world(tmp_path):
    registry = Registry(f"sqlite:///{tmp_path / 'r.sqlite3'}")
    tenant, admin, _ = registry.bootstrap("lab")
    registry.set_quota(admin, "concurrent_runs", 50)
    who = registry.create_principal(tenant, "ada", Role.RESEARCHER)
    return {"registry": registry, "tenant": tenant, "who": who, "admin": admin, "artifacts": Artifacts(registry, LocalArtifactStore(tmp_path / "s")), "tmp": tmp_path}


def build(world, observer):
    orchestrator = Orchestrator(world["registry"], world["artifacts"], observer=observer)
    exp = world["registry"].add_experiment(world["who"], "x", read_spec(EXPERIMENTS / "proton_transfer.yaml"))["id"]
    return orchestrator, exp


def sample(metrics, name, **labels):
    return metrics.registry.get_sample_value(name, labels)


# -- metrics ------------------------------------------------------------------------------------------------------------------------------
def test_a_run_is_counted_task_by_task_and_axis_by_axis(world):
    metrics = MetricsObserver()
    orchestrator, exp = build(world, metrics)
    orchestrator.submit(world["who"], exp, n_samples=0)
    Worker(orchestrator).run_until_idle()
    assert sample(metrics, "substrate_runs_submitted_total") == 1
    assert sample(metrics, "substrate_tasks_started_total", kind="solve", queue="cpu") == 2 and sample(metrics, "substrate_tasks_started_total", kind="validate", queue="cpu") == 1
    assert sample(metrics, "substrate_tasks_finished_total", kind="solve", outcome="succeeded", axis="none") == 2
    assert sample(metrics, "substrate_task_seconds_count", kind="solve") == 2 and sample(metrics, "substrate_task_seconds_sum", kind="solve") > 0
    assert sample(metrics, "substrate_runs_finished_total", state="succeeded", overall="PASS") == 1
    assert sample(metrics, "substrate_run_axis_total", axis="physics", state="PASS") == 1 and sample(metrics, "substrate_run_axis_total", axis="uncertainty", state="SKIPPED") == 1


def test_failures_blocks_and_auth_failures_are_counted_by_reason(world, monkeypatch):
    metrics = MetricsObserver()
    orchestrator, exp = build(world, metrics)
    monkeypatch.setenv("SUBSTRATE_MAX_GRID_POINTS", "10")
    orchestrator.submit(world["who"], exp, n_samples=0)
    metrics.auth_failed("bad-token")
    metrics.auth_failed("bad-token")
    Worker(orchestrator).run_until_idle()
    assert sample(metrics, "substrate_blocked_total", reason="ResourceLimitError") == 1
    assert sample(metrics, "substrate_tasks_finished_total", kind="solve", outcome="failed", axis="security") == 1
    assert sample(metrics, "substrate_auth_failures_total", reason="bad-token") == 2
    assert sample(metrics, "substrate_runs_finished_total", state="failed", overall="FAIL") == 1


def test_the_gauges_are_read_from_the_database_when_scraped(world):
    metrics = with_live_gauges(MetricsObserver(), world["registry"])
    orchestrator, exp = build(world, metrics)
    orchestrator.submit(world["who"], exp, n_samples=0)
    assert sample(metrics, "substrate_queue_depth", queue="cpu") == 1 and sample(metrics, "substrate_runs_active", state="queued") == 1
    worker = Worker(orchestrator)
    worker.run_once()
    assert sample(metrics, "substrate_runs_active", state="running") == 1 and sample(metrics, "substrate_runs_active", state="queued") is None
    worker.run_until_idle()
    assert sample(metrics, "substrate_queue_depth", queue="cpu") is None and sample(metrics, "substrate_runs_active", state="running") is None
    assert sample(metrics, "substrate_build_info", version="0.1.0") == 1


def test_what_a_separate_worker_did_is_visible_from_the_database_not_from_its_counters(world):
    api_metrics = with_live_gauges(MetricsObserver(), world["registry"])               # the process that serves /metrics
    api, exp = build(world, api_metrics)
    api.submit(world["who"], exp, n_samples=0)
    worker_side, _ = build(world, LoggingObserver())                                    # another process: its own observer, no shared counters
    Worker(worker_side).run_until_idle()
    assert sample(api_metrics, "substrate_tasks_finished_total", kind="solve", outcome="succeeded", axis="none") is None       # the counter never left the worker
    assert sample(api_metrics, "substrate_tasks_recent", state="succeeded", kind="solve") == 2 and sample(api_metrics, "substrate_tasks_recent", state="succeeded", kind="validate") == 1
    assert sample(api_metrics, "substrate_runs_recent", state="succeeded") == 1
    assert sample(api_metrics, "substrate_run_axis_recent", axis="physics", state="PASS") == 1 and sample(api_metrics, "substrate_run_axis_recent", axis="uncertainty", state="SKIPPED") == 1
    assert sample(api_metrics, "substrate_task_seconds_recent", kind="solve", stat="mean") > 0
    assert sample(api_metrics, "substrate_task_seconds_recent", kind="solve", stat="max") >= sample(api_metrics, "substrate_task_seconds_recent", kind="solve", stat="mean")
    assert sample(api_metrics, "substrate_task_failures_recent", axis="execution") is None


def test_a_failure_is_counted_on_the_axis_it_belongs_to(world, monkeypatch):
    metrics = with_live_gauges(MetricsObserver(), world["registry"])
    orchestrator, exp = build(world, None)
    monkeypatch.setenv("SUBSTRATE_MAX_GRID_POINTS", "10")
    orchestrator.submit(world["who"], exp, n_samples=0)
    Worker(orchestrator).run_until_idle()
    assert sample(metrics, "substrate_task_failures_recent", axis="security") == 1
    assert sample(metrics, "substrate_tasks_recent", state="failed", kind="solve") == 1
    assert sample(metrics, "substrate_runs_recent", state="failed") == 1 and sample(metrics, "substrate_run_axis_recent", axis="security", state="FAIL") == 1


def test_the_recent_gauges_forget_what_is_older_than_a_day_and_read_a_bounded_number_of_rows(world, monkeypatch):
    import substrate.control.observability as obs
    metrics = with_live_gauges(MetricsObserver(), world["registry"])
    orchestrator, exp = build(world, None)
    orchestrator.submit(world["who"], exp, n_samples=0)
    Worker(orchestrator).run_until_idle()
    monkeypatch.setattr(obs, "RECENT_ROWS", 1)
    assert sum(1 for s in metrics.registry.collect() for sample_ in s.samples if sample_.name == "substrate_tasks_recent" and sample_.value) == 1      # one row, not all three
    monkeypatch.setattr(obs, "RECENT_ROWS", 20_000)
    real = world["registry"].clock
    monkeypatch.setattr(world["registry"], "clock", lambda: real() + 2 * 86_400)
    names = {sample_.name for s in metrics.registry.collect() for sample_ in s.samples}
    assert "substrate_tasks_recent" not in names and "substrate_runs_recent" not in names and "substrate_run_axis_recent" not in names


def test_the_text_format_is_what_prometheus_reads_and_no_label_names_a_run_or_a_person(world):
    metrics = with_live_gauges(MetricsObserver(), world["registry"])
    orchestrator, exp = build(world, metrics)
    orchestrator.submit(world["who"], exp, n_samples=0)
    Worker(orchestrator).run_until_idle()
    text = metrics.render().decode()
    assert "# TYPE substrate_tasks_finished_total counter" in text and "# TYPE substrate_task_seconds histogram" in text and "# TYPE substrate_queue_depth gauge" in text
    assert "RUN-" not in text and "EXP-" not in text and "P-0" not in text and "T-0" not in text      # nothing unbounded or personal in a label


def test_two_metrics_objects_do_not_share_state():
    a, b = MetricsObserver(), MetricsObserver()
    a.auth_failed("x")
    assert sample(a, "substrate_auth_failures_total", reason="x") == 1 and sample(b, "substrate_auth_failures_total", reason="x") is None


# -- traces -------------------------------------------------------------------------------------------------------------------------------
def tracing():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return TracingObserver(provider.get_tracer("test")), exporter


def test_every_task_is_a_span_and_all_of_a_runs_spans_share_one_trace_even_across_workers(world):
    observer, exporter = tracing()
    orchestrator, exp = build(world, observer)
    run = orchestrator.submit(world["who"], exp, n_samples=0)
    other = orchestrator.submit(world["who"], exp, n_samples=0)
    # two workers with their own observers (as separate processes would have), taking turns
    first, second = Worker(orchestrator, worker_id="a"), Worker(orchestrator, worker_id="b")
    while first.run_once() | second.run_once():
        pass
    spans = exporter.get_finished_spans()
    by_run = {}
    for span in spans:
        by_run.setdefault(span.attributes["substrate.run"], []).append(span)
    assert len(spans) == 14 and set(by_run) == {run["id"], other["id"]}
    for run_id, run_spans in by_run.items():
        assert {format(s.context.trace_id, "032x") for s in run_spans} == {format(trace_id_for(run_id), "032x")}
        assert sorted(s.attributes["substrate.task.name"] for s in run_spans)[:2] == ["00-solve:quantum.double_well_1d", "01-translate:quantum_to_reaction.sequential_tunneling"]
        assert all(s.attributes["substrate.outcome"] == "succeeded" and s.status.status_code.name == "UNSET" for s in run_spans)       # success is the absence of an error
    assert trace_id_for(run["id"]) != trace_id_for(other["id"]) and trace_id_for("RUN-000001") == trace_id_for("RUN-000001")


def test_a_failed_task_is_an_error_span_with_its_axis(world, monkeypatch):
    observer, exporter = tracing()
    orchestrator, exp = build(world, observer)
    monkeypatch.setenv("SUBSTRATE_MAX_GRID_POINTS", "10")
    orchestrator.submit(world["who"], exp, n_samples=0)
    Worker(orchestrator).run_until_idle()
    failed = [s for s in exporter.get_finished_spans() if s.attributes["substrate.outcome"] == "failed"]
    assert len(failed) == 1 and failed[0].attributes["substrate.failure_axis"] == "security" and failed[0].status.status_code.name == "ERROR"


def test_a_span_that_was_never_started_is_ignored_on_finish():
    observer, exporter = tracing()
    from substrate.control.taskqueue import ClaimedTask
    task = ClaimedTask("RUN-1/00", "RUN-1", "T-1", "n", "solve", "cpu", {}, [], 1, 3, 10.0, {})
    observer.task_finished(task, 0.1, "succeeded", None)
    assert exporter.get_finished_spans() == ()


def test_tracing_is_off_unless_an_endpoint_is_set_and_the_endpoint_is_checked(world):
    from substrate.control.observability import configure_tracing
    assert configure_tracing({}) is None and configure_tracing({"OTEL_EXPORTER_OTLP_ENDPOINT": "  "}) is None
    for bad in ("ftp://collector:4318", "collector:4318", "http://user:pw@collector:4318", "http://collector:4318/#x", "file:///etc/passwd", "http://"):
        with pytest.raises(ValueError):
            configure_tracing({"OTEL_EXPORTER_OTLP_ENDPOINT": bad})


def test_configured_tracing_sends_a_runs_spans_to_the_exporter_under_one_trace(world):
    from substrate.control.observability import configure_tracing
    exporter = InMemorySpanExporter()
    tracing = configure_tracing({"OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4318", "OTEL_SERVICE_NAME": "w1"}, exporter=exporter)
    assert isinstance(tracing, TracingObserver)
    orchestrator, exp = build(world, tracing)
    run = orchestrator.submit(world["who"], exp, n_samples=0)
    Worker(orchestrator).run_until_idle()
    from opentelemetry import trace as otel_trace
    assert otel_trace.get_tracer_provider() is not tracing.provider          # the process-wide provider is untouched: only this observer's spans are exported
    tracing.flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 7 and {s.context.trace_id for s in spans} == {trace_id_for(run["id"])}
    assert all(s.resource.attributes["service.name"] == "w1" for s in spans)
    tracing.shutdown()


def test_the_real_exporter_is_built_from_the_endpoint_without_touching_the_network(world):
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    from substrate.control.observability import configure_tracing
    tracing = configure_tracing({"OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:9/"})
    assert isinstance(tracing, TracingObserver) and tracing.provider is not None
    tracing.shutdown()


# -- logs ---------------------------------------------------------------------------------------------------------------------------------
def logger_into(stream):
    log = logging.getLogger(f"test.{id(stream)}")
    log.setLevel(logging.DEBUG)
    log.propagate = False
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    log.addHandler(handler)
    return log


def test_log_lines_are_json_with_the_ids_needed_to_follow_a_run(world):
    stream = io.StringIO()
    orchestrator, exp = build(world, LoggingObserver(logger_into(stream)))
    run = orchestrator.submit(world["who"], exp, n_samples=0)
    Worker(orchestrator).run_until_idle()
    lines = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [l["msg"] for l in lines][:3] == ["run submitted", "task started", "task finished"] and lines[-1]["msg"] == "run finished"
    assert all(l.get("run") == run["id"] and l["trace_id"] == format(trace_id_for(run["id"]), "032x") for l in lines)
    assert lines[-1]["axes"]["execution"] == "PASS" and lines[-1]["overall"] == "PASS" and lines[2]["outcome"] == "succeeded"


@pytest.mark.parametrize("text, expect", [
    ("using key sbs_K00001_AbCd-EfGh_ijKL123 now", "using key sbs_K00001_*** now"),
    ("password=hunter2 and more", "password=*** and more"),
    ("Authorization: Bearer abcdef123456", "Authorization=***"),
    ("authorization=Basic dXNlcjpwdw==", "Authorization=***"),
    ("curl -H 'Authorization: Bearer abc.def-ghi'", "curl -H 'Authorization=***'"),
    ("api_key: 12345", "api_key=***"),
    ("token = abc123 end", "token =*** end"),
    ("bearer is just a word", "bearer is just a word"),
    ("sent Bearer abcdefghijklmnop upstream", "sent Bearer *** upstream"),
    ("nothing sensitive here: 12345", "nothing sensitive here: 12345"),
])
def test_tokens_and_credentials_are_redacted_from_text(text, expect):
    assert redact(text) == expect


def test_redaction_leaves_no_trace_of_the_secret_itself():
    for secret in ("hunter2", "AbCd-EfGh_ijKL123", "abcdef123456", "dXNlcjpwdw==", "abc.def-ghi"):
        for template in ("password={}", "using sbs_K00001_{}", "Authorization: Bearer {}", "authorization=Basic {}", "x token: {} y"):
            assert secret not in redact(template.format(secret)), (template, secret)


def test_no_log_record_can_carry_a_credential_in_its_message_its_fields_or_its_traceback():
    stream = io.StringIO()
    log = logger_into(stream)
    token = "sbs_K00007_SuperSecretValue-123"
    try:
        raise RuntimeError(f"failed with {token}")
    except RuntimeError:
        log.error("login with %s failed", token, extra={"api_key": "abc", "password": "pw", "note": f"retry {token}", "count": 3}, exc_info=True)
    out = stream.getvalue()
    assert "SuperSecretValue" not in out and "pw" not in json.loads(out)["password"] and json.loads(out)["password"] == "***" and json.loads(out)["api_key"] == "***"
    assert json.loads(out)["count"] == 3 and "sbs_K00007_***" in out


# -- the pieces together ----------------------------------------------------------------------------------------------------------------------
def test_a_composite_observer_gives_every_observer_every_event(world):
    metrics, (tracer, exporter) = MetricsObserver(), tracing()
    stream = io.StringIO()
    orchestrator, exp = build(world, CompositeObserver(metrics, tracer, LoggingObserver(logger_into(stream))))
    orchestrator.submit(world["who"], exp, n_samples=0)
    Worker(orchestrator).run_until_idle()
    orchestrator.observer.auth_failed("nope")
    orchestrator.observer.blocked("Quota")
    assert sample(metrics, "substrate_runs_finished_total", state="succeeded", overall="PASS") == 1 and len(exporter.get_finished_spans()) == 7
    assert "run finished" in stream.getvalue() and "authentication failed" in stream.getvalue() and sample(metrics, "substrate_blocked_total", reason="Quota") == 1
    assert sample(metrics, "substrate_auth_failures_total", reason="nope") == 1


def test_the_base_observer_ignores_everything():
    observer = Observer()
    for call in (lambda: observer.run_submitted({}), lambda: observer.task_started(None), lambda: observer.task_finished(None, 0, "x", None),
                 lambda: observer.run_finished({}, None), lambda: observer.blocked("x"), lambda: observer.auth_failed("x")):
        assert call() is None
