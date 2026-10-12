"""The platform command line: set up, submit, work, check, reproduce, manage keys; the key comes from the environment and the server is loopback-only unless told otherwise."""
import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from substrate.cli import main  # noqa: E402
from substrate.control.cli import create_server_app, _orchestrator, _open  # noqa: E402

from conftest import EXPERIMENTS  # noqa: E402


@pytest.fixture
def site(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SUBSTRATE_DB", f"sqlite:///{tmp_path / 'p.sqlite3'}")
    monkeypatch.setenv("SUBSTRATE_STORE", str(tmp_path / "artifacts"))
    monkeypatch.delenv("SUBSTRATE_API_KEY", raising=False)
    assert main(["platform", "init", "--tenant", "lab", "--admin", "root"]) == 0
    token = capsys.readouterr().out.strip().splitlines()[-1]
    monkeypatch.setenv("SUBSTRATE_API_KEY", token)
    return {"token": token, "tmp": tmp_path, "capsys": capsys, "monkeypatch": monkeypatch}


def run_cli(site, *argv):
    code = main(["platform", *argv])
    captured = site["capsys"].readouterr()
    return code, captured.out, captured.err


def test_init_creates_a_tenant_and_prints_the_administrators_key_once(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SUBSTRATE_DB", f"sqlite:///{tmp_path / 'p.sqlite3'}")
    monkeypatch.setenv("SUBSTRATE_STORE", str(tmp_path / "a"))
    assert main(["platform", "init"]) == 0
    out = capsys.readouterr().out
    assert "tenant T-0001 (default)" in out and "administrator P-0001 (admin)" in out and "shown once" in out
    token = out.strip().splitlines()[-1]
    assert token.startswith("sbs_K00001_")
    registry, _ = _open(type("A", (), {"db": None, "store": None})())
    assert registry.authenticate(token).name == "admin"
    secret = token.split("_", 2)[2]
    import sqlalchemy as sa
    from substrate.control import db
    with registry.engine.connect() as conn:                                                  # whatever the database: no value in any table holds the secret, only its hash
        for table in db.metadata.sorted_tables:
            assert not any(secret in repr(tuple(row)) for row in conn.execute(sa.select(table))), table.name
    if (tmp_path / "p.sqlite3").exists():
        with open(tmp_path / "p.sqlite3", "rb") as handle:
            assert secret.encode() not in handle.read()                                      # nor the file itself


def test_every_command_that_acts_as_someone_needs_a_valid_key(site):
    site["monkeypatch"].delenv("SUBSTRATE_API_KEY")
    for argv in (["status", "RUN-000001"], ["submit", str(EXPERIMENTS / "proton_transfer.yaml")], ["cancel", "RUN-000001"], ["reproduce", "RUN-000001"], ["apikey", "create", "x", "--role", "VIEWER"]):
        code, out, err = run_cli(site, *argv)
        assert code == 1 and "valid API key is required" in err and out == ""
    site["monkeypatch"].setenv("SUBSTRATE_API_KEY", "sbs_K00001_wrong")
    assert run_cli(site, "status", "RUN-000001")[0] == 1


def test_submit_wait_runs_the_experiment_and_prints_the_six_axis_status(site):
    code, out, err = run_cli(site, "submit", str(EXPERIMENTS / "proton_transfer.yaml"), "--samples", "0", "--wait")
    assert code == 0, err
    lines = out.strip().splitlines()
    assert lines[0] == "EXP-000001 -> RUN-000001" and lines[1].startswith("RUN-000001: succeeded") and "7 succeeded" in lines[1]
    assert [l.split(":")[0] for l in lines[2:]] == ["Execution", "Physics", "Validation", "Security", "Reproducibility", "Uncertainty"]
    code, again, _ = run_cli(site, "status", "RUN-000001")
    assert code == 0 and again.strip().splitlines()[0].startswith("RUN-000001: succeeded")


def test_submit_without_wait_leaves_the_run_for_a_worker_which_then_finishes_it(site):
    code, out, _ = run_cli(site, "submit", str(EXPERIMENTS / "proton_transfer.yaml"), "--samples", "0", "--name", "my experiment")
    assert code == 0 and out.strip() == "EXP-000001 -> RUN-000001"
    assert run_cli(site, "status", "RUN-000001")[1].startswith("RUN-000001: queued")
    code, out, _ = run_cli(site, "worker", "--queues", "cpu", "--until-idle")
    assert code == 0 and "ran 7 task(s)" in out
    assert run_cli(site, "status", "RUN-000001")[1].startswith("RUN-000001: succeeded")


def test_a_workers_log_goes_to_stderr_as_json_without_secrets(site):
    import json
    run_cli(site, "submit", str(EXPERIMENTS / "proton_transfer.yaml"), "--samples", "0")
    code, out, err = run_cli(site, "worker", "--until-idle")
    records = [json.loads(line) for line in err.splitlines() if line.startswith("{")]
    assert code == 0 and {r["msg"] for r in records} >= {"task started", "task finished", "run finished"} and site["token"] not in err and site["token"] not in out


def test_a_worker_refuses_a_bad_tracing_endpoint_and_traces_when_given_a_good_one(site):
    site["monkeypatch"].setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://user:pw@collector:4318")
    code, out, err = run_cli(site, "worker", "--until-idle")
    assert code == 1 and err.startswith("error: OTEL_EXPORTER_OTLP_ENDPOINT") and "pw" not in err
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    site["monkeypatch"].setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:9")          # nothing listens there; with nothing to send, nothing is attempted
    code, out, err = run_cli(site, "worker", "--until-idle")
    assert code == 0 and "ran 0 task(s)" in out


def test_a_bad_experiment_is_refused_before_it_is_stored(site, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("experiment:\n  nonsense: true\n", encoding="utf-8")
    code, out, err = run_cli(site, "submit", str(bad))
    assert code == 1 and err.startswith("error:")
    assert run_cli(site, "status", "RUN-000001")[0] == 1                                       # nothing was created


def test_reproduce_runs_again_and_reports_the_comparison(site):
    run_cli(site, "submit", str(EXPERIMENTS / "proton_transfer.yaml"), "--samples", "2", "--wait")
    code, out, err = run_cli(site, "reproduce", "RUN-000001", "--wait")
    assert code == 0, err
    assert out.splitlines()[0] == "RUN-000001 -> RUN-000002 (a reproduction)" and "matches RUN-000001" in out


def test_cancel_stops_a_queued_run(site):
    run_cli(site, "submit", str(EXPERIMENTS / "proton_transfer.yaml"), "--samples", "0")
    code, out, _ = run_cli(site, "cancel", "RUN-000001")
    assert code == 0 and "cancelled 7 task(s)" in out
    assert run_cli(site, "status", "RUN-000001")[1].startswith("RUN-000001: cancelled")
    assert run_cli(site, "cancel", "RUN-999999")[0] == 1


def test_a_run_a_model_refuses_exits_with_status_2_so_scripts_can_tell(site, tmp_path):
    import yaml
    from substrate.experiment import read_spec
    spec = read_spec(EXPERIMENTS / "proton_transfer_molecular.yaml")
    (spec["experiment"] if "experiment" in spec else spec)["system"]["parameters"]["oo_equilibrium"]["value"] = 2.3          # the molecular model finds no double well
    refusing = tmp_path / "refuses.yaml"
    refusing.write_text(yaml.safe_dump(spec), encoding="utf-8")
    code, out, err = run_cli(site, "submit", str(refusing), "--samples", "0", "--wait")
    assert code == 2, (out, err)
    assert "RUN-000001: failed" in out and out.split("Physics:")[1].splitlines()[0].strip().startswith("FAIL")
    assert out.split("Execution:")[1].splitlines()[0].strip().startswith("PASS")                    # the machinery worked; the model said no


def test_only_administrators_manage_keys_and_a_new_key_works_and_can_be_revoked(site):
    code, out, _ = run_cli(site, "apikey", "create", "vera", "--role", "VIEWER")
    assert code == 0 and "VIEWER" in out
    viewer_token = out.strip().splitlines()[-1]
    site["monkeypatch"].setenv("SUBSTRATE_API_KEY", viewer_token)
    assert run_cli(site, "status", "RUN-000001")[0] == 1                                       # authenticates (the error is only that there is no such run)
    code, _, err = run_cli(site, "apikey", "create", "eve", "--role", "ADMIN")
    assert code == 1 and "may not admin" in err
    site["monkeypatch"].setenv("SUBSTRATE_API_KEY", site["token"])
    key_id = viewer_token.split("_")[1]
    assert run_cli(site, "apikey", "revoke", key_id)[1].strip() == "revoked" and run_cli(site, "apikey", "revoke", key_id)[1].strip() == f"no active key {key_id}"
    site["monkeypatch"].setenv("SUBSTRATE_API_KEY", viewer_token)
    assert "valid API key is required" in run_cli(site, "status", "RUN-000001")[2]


def test_serve_listens_only_on_loopback_unless_asked_to(site):
    for host in ("0.0.0.0", "192.168.1.5", "::", "example.org"):
        code, out, err = run_cli(site, "serve", "--host", host)
        assert code == 1 and "loopback-only" in err and "--allow-external" in err


def test_the_served_application_has_metrics_and_data_adapters_wired_in(site):
    args = type("A", (), {"db": None, "store": None, "isolation": None})()
    registry, artifacts = _open(args)
    app = create_server_app(registry, artifacts, _orchestrator(registry, artifacts))
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {site['token']}"}
    assert client.get("/metrics", headers=headers).status_code == 200 and client.get("/datasets/adapters", headers=headers).json()["adapters"]
    assert client.post("/datasets/ingest", json={"adapter": "nist.ion_energetics"}, headers=headers).status_code == 201
    bare = TestClient(create_server_app(registry, artifacts, _orchestrator(registry, artifacts), metrics=False))
    assert bare.get("/metrics", headers=headers).status_code == 404


def test_the_isolation_setting_chooses_the_executor(site, monkeypatch):
    from substrate.control.sandbox import ProcessExecutor
    from substrate.control.steps import InlineExecutor
    args = type("A", (), {"db": None, "store": None})()
    registry, artifacts = _open(args)
    assert isinstance(_orchestrator(registry, artifacts, "inline").executor, InlineExecutor)
    assert isinstance(_orchestrator(registry, artifacts, "process").executor, ProcessExecutor)
    monkeypatch.setenv("SUBSTRATE_WORKER_ISOLATION", "process")
    assert isinstance(_orchestrator(registry, artifacts).executor, ProcessExecutor)


def test_without_the_extras_the_command_explains_what_to_install_instead_of_crashing(monkeypatch, capsys, tmp_path):
    import builtins
    import sys
    real_import = builtins.__import__

    def refuse(name, *a, **k):
        if name == "sqlalchemy" or name.startswith("sqlalchemy."):
            raise ImportError("No module named 'sqlalchemy'", name="sqlalchemy")
        return real_import(name, *a, **k)

    for module in [m for m in list(sys.modules) if m.startswith(("sqlalchemy", "substrate.control"))]:
        monkeypatch.delitem(sys.modules, module)
    monkeypatch.setattr(builtins, "__import__", refuse)
    monkeypatch.setenv("SUBSTRATE_DB", f"sqlite:///{tmp_path / 'x.sqlite3'}")
    assert main(["platform", "init"]) == 1
    err = capsys.readouterr().err
    assert "pip install 'substrate[platform]'" in err and "sqlalchemy is missing" in err


# -- where things are kept ------------------------------------------------------------------------------------------------------------------------
def test_the_database_url_can_come_from_a_secret_file_so_it_is_not_in_the_environment(tmp_path, monkeypatch):
    from substrate.control.cli import database_url
    secret = tmp_path / "db_url"
    secret.write_text("postgresql+psycopg2://app:pw@db/substrate\n", encoding="utf-8")
    monkeypatch.delenv("SUBSTRATE_DB", raising=False)
    monkeypatch.setenv("SUBSTRATE_DB_FILE", str(secret))
    assert database_url(None) == "postgresql+psycopg2://app:pw@db/substrate"
    monkeypatch.setenv("SUBSTRATE_DB", "sqlite:///x.sqlite3")
    assert database_url(None) == "sqlite:///x.sqlite3" and database_url("sqlite:///y.sqlite3") == "sqlite:///y.sqlite3"          # an explicit choice wins
    monkeypatch.delenv("SUBSTRATE_DB")
    monkeypatch.delenv("SUBSTRATE_DB_FILE")
    assert database_url(None) == "sqlite:///substrate-platform.sqlite3"


def test_an_unreadable_or_empty_secret_file_is_a_clear_error_that_does_not_echo_its_contents(tmp_path, monkeypatch):
    from substrate import SubstrateError
    from substrate.control.cli import database_url
    monkeypatch.delenv("SUBSTRATE_DB", raising=False)
    monkeypatch.setenv("SUBSTRATE_DB_FILE", str(tmp_path / "missing"))
    with pytest.raises(SubstrateError, match="cannot read the database URL"):
        database_url(None)
    empty = tmp_path / "empty"
    empty.write_text("  \n", encoding="utf-8")
    monkeypatch.setenv("SUBSTRATE_DB_FILE", str(empty))
    with pytest.raises(SubstrateError, match="is empty"):
        database_url(None)


def test_the_artifact_store_is_a_directory_or_an_s3_location(tmp_path, monkeypatch):
    from substrate import SubstrateError
    from substrate.control.artifacts import LocalArtifactStore, S3ArtifactStore
    from substrate.control.cli import artifact_store
    assert isinstance(artifact_store(str(tmp_path / "a")), LocalArtifactStore)
    built = []
    monkeypatch.setattr(S3ArtifactStore, "from_boto3", classmethod(lambda cls, bucket, prefix="artifacts", **options: built.append((bucket, prefix, options)) or "store"))
    monkeypatch.delenv("SUBSTRATE_S3_ENDPOINT", raising=False)
    assert artifact_store("s3://results/lab-a") == "store" and built[-1] == ("results", "lab-a", {})
    monkeypatch.setenv("SUBSTRATE_S3_ENDPOINT", "http://minio:9000")
    artifact_store("s3://results")
    assert built[-1] == ("results", "artifacts", {"endpoint_url": "http://minio:9000"})
    with pytest.raises(SubstrateError, match="s3://bucket/prefix"):
        artifact_store("s3://")


# -- the health check, and stopping a worker ------------------------------------------------------------------------------------------------
def test_check_succeeds_on_a_database_that_has_its_tables_and_writes_nothing(site):
    code, out, err = run_cli(site, "check")
    assert (code, out.strip(), err) == (0, "ready", "")


def test_check_fails_on_an_empty_database_without_creating_its_tables(tmp_path, monkeypatch, capsys):
    import sqlalchemy as sa
    path = tmp_path / "empty.sqlite3"
    monkeypatch.setenv("SUBSTRATE_DB", f"sqlite:///{path}")
    monkeypatch.setenv("SUBSTRATE_STORE", str(tmp_path))
    assert main(["platform", "check"]) == 1
    assert "not ready: OperationalError" in capsys.readouterr().err
    with sa.create_engine(f"sqlite:///{path}").connect() as conn:
        assert sa.inspect(conn).get_table_names() == []                                    # looking did not create the schema


def test_check_fails_on_a_missing_artifact_directory(site):
    site["monkeypatch"].setenv("SUBSTRATE_STORE", str(site["tmp"] / "nowhere"))
    code, out, err = run_cli(site, "check")
    assert code == 1 and out == "" and "artifact directory" in err


def test_check_names_only_the_type_of_a_database_failure_so_the_password_stays_out_of_the_output(site):
    pytest.importorskip("psycopg2")
    site["monkeypatch"].setenv("SUBSTRATE_DB", "postgresql+psycopg2://substrate:hunter2-not-a-real-pw@127.0.0.1:1/substrate")
    code, out, err = run_cli(site, "check")
    assert code == 1 and "not ready: OperationalError" in err and "hunter2" not in out + err and "127.0.0.1" not in out + err


@pytest.mark.skipif(__import__("os").name == "nt", reason="POSIX signals: a Windows process cannot be sent SIGTERM")
def test_a_worker_stops_on_sigterm_instead_of_waiting_to_be_killed(site):
    import os
    import signal
    import subprocess
    import sys
    import time
    env = {**os.environ, "SUBSTRATE_DB": f"sqlite:///{site['tmp'] / 'p.sqlite3'}", "SUBSTRATE_STORE": str(site["tmp"] / "artifacts")}
    worker = subprocess.Popen([sys.executable, "-m", "substrate", "platform", "worker", "--queues", "cpu"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        time.sleep(4)                                                                     # imported, connected, idle and polling
        assert worker.poll() is None
        worker.send_signal(signal.SIGTERM)
        assert worker.wait(timeout=10) == 0
    finally:
        if worker.poll() is None:
            worker.kill()
        worker.communicate()
