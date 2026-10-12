"""Cases of the metrics, traces and log redaction that the first tests did not pin, found by mutation testing."""
import json
import logging

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("prometheus_client")
pytest.importorskip("opentelemetry.sdk")

import sqlalchemy as sa  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

from substrate.control import db  # noqa: E402
from substrate.control.observability import JsonFormatter, MetricsObserver, TracingObserver, configure_tracing, redact, trace_id_for, with_live_gauges  # noqa: E402
from substrate.control.worker import Worker  # noqa: E402

from test_control_observability import build, sample, world  # noqa: E402,F401


# -- redaction ------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("word", ["password", "passwd", "secret", "token", "api_key", "api-key", "apikey", "authorization", "bearer", "PASSWORD", "Api_Key"])
def test_every_name_that_looks_like_a_credential_has_its_value_blanked(word):
    assert redact(f"connecting with {word}=hunter2 now").lower() == f"connecting with {word}=*** now".lower()       # (an Authorization header is rewritten with that capital)
    assert redact(f"{word}: hunter2").lower() == f"{word}=***".lower()
    assert "hunter2" not in redact(f'{{"{word}": "x"}} {word}=hunter2')


@pytest.mark.parametrize("name", ["password", "passwd", "secret", "token", "api_key", "api-key", "apikey", "authorization", "credential", "cookie", "Set-Cookie", "DB_PASSWORD",
                                  "x_credentials", "SESSION_TOKEN"])
def test_a_log_field_whose_name_looks_like_a_credential_is_never_written(name):
    record = logging.LogRecord("substrate.control", logging.INFO, __file__, 1, "msg", (), None)
    setattr(record, name, "hunter2")
    setattr(record, "harmless", "visible")
    written = json.loads(JsonFormatter().format(record))
    assert written[name] == "***" and written["harmless"] == "visible" and "hunter2" not in json.dumps(written)


def test_the_machinery_fields_of_a_record_are_not_copied_into_the_log_line():
    record = logging.LogRecord("substrate.control", logging.WARNING, __file__, 1, "hello %s", ("there",), None)
    written = json.loads(JsonFormatter().format(record))
    assert set(written) == {"ts", "level", "logger", "msg"} and written["msg"] == "hello there" and written["level"] == "WARNING"


# -- the database-derived gauges ---------------------------------------------------------------------------------------------------------
def test_running_tasks_are_counted_and_skipped_ones_are_not_failures(world, monkeypatch):
    metrics = with_live_gauges(MetricsObserver(), world["registry"])
    orchestrator, exp = build(world, None)
    orchestrator.submit(world["who"], exp, n_samples=0)
    claimed = orchestrator.queue.claim("a-worker", ["cpu"])                                   # taken, not finished
    assert sample(metrics, "substrate_tasks_recent", state="running", kind=claimed.kind) == 1
    orchestrator.queue.fail(claimed.id, "a-worker", {"type": "X", "axis": "physics"}, retry=False)          # fails for good: everything after it is skipped
    assert sample(metrics, "substrate_task_failures_recent", axis="physics") == 1
    assert sample(metrics, "substrate_task_failures_recent", axis="execution") is None                      # the skipped tasks are not failures
    assert sample(metrics, "substrate_tasks_recent", state="skipped", kind="solve") == 2 and sample(metrics, "substrate_tasks_recent", state="running", kind=claimed.kind) is None


def test_a_stored_axis_name_this_version_does_not_know_never_becomes_a_label(world):
    metrics = with_live_gauges(MetricsObserver(), world["registry"])
    orchestrator, exp = build(world, None)
    orchestrator.submit(world["who"], exp, n_samples=0)
    Worker(orchestrator).run_until_idle()
    with world["registry"].engine.begin() as conn:
        conn.execute(sa.update(db.tasks).where(db.tasks.c.name == "check").values(state="failed", error={"axis": "gravity", "message": "x"}, seconds=999.0))
        conn.execute(sa.update(db.runs).values(status={"physics": {"state": "PASS", "reasons": []}, "gravity": {"state": "FAIL", "reasons": []}, "security": "not a dict"}))
    names = {(s.name, tuple(sorted(s.labels.items()))) for family in metrics.registry.collect() for s in family.samples}
    assert ("substrate_task_failures_recent", (("axis", "execution"),)) in names and not any("gravity" in str(labels) for _, labels in names)
    assert sample(metrics, "substrate_run_axis_recent", axis="physics", state="PASS") == 1 and sample(metrics, "substrate_run_axis_recent", axis="security", state="FAIL") is None
    assert sample(metrics, "substrate_task_seconds_recent", kind="check", stat="max") is None       # a failed task's time is not the time of successful ones


def test_one_scrape_reads_a_bounded_number_of_runs_as_well_as_tasks(world, monkeypatch):
    import substrate.control.observability as obs
    metrics = with_live_gauges(MetricsObserver(), world["registry"])
    orchestrator, exp = build(world, None)
    for _ in range(3):
        orchestrator.submit(world["who"], exp, n_samples=0)
    Worker(orchestrator).run_until_idle()
    assert sample(metrics, "substrate_runs_recent", state="succeeded") == 3
    monkeypatch.setattr(obs, "RECENT_ROWS", 2)
    assert sample(metrics, "substrate_runs_recent", state="succeeded") == 2


# -- traces ---------------------------------------------------------------------------------------------------------------------------
def test_a_trace_id_is_128_bits_derived_only_from_the_run_id():
    ids = [trace_id_for(f"RUN-{n:06d}") for n in range(1, 40)]
    assert all(0 < i < 2**128 for i in ids) and max(i.bit_length() for i in ids) > 120 and len(set(ids)) == len(ids)
    assert trace_id_for("RUN-000001") == trace_id_for("RUN-000001") != trace_id_for("RUN-000002")


def test_a_span_says_which_run_task_queue_attempt_and_tenant_it_belongs_to(world):
    exporter = InMemorySpanExporter()
    tracing = configure_tracing({"OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4318"}, exporter=exporter)
    orchestrator, exp = build(world, tracing)
    run = orchestrator.submit(world["who"], exp, n_samples=0)
    Worker(orchestrator).run_until_idle()
    tracing.flush()
    spans = exporter.get_finished_spans()
    assert spans and all(s.attributes["substrate.tenant"] == world["tenant"] and s.attributes["substrate.run"] == run["id"] and s.attributes["substrate.attempt"] == 1 for s in spans)
    assert {s.attributes["substrate.queue"] for s in spans} == {"cpu"} and all(s.name.startswith("task ") for s in spans)
    tracing.shutdown()
