"""The HTTP API: who gets in, what each role may do, what another tenant cannot see, how bad input is refused, and the whole life of a run through it."""
import copy

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("prometheus_client")

from fastapi.testclient import TestClient  # noqa: E402

from substrate import Pipeline, default_registry  # noqa: E402
from substrate.control import db  # noqa: E402
from substrate.control.api import SafeJSONResponse, create_app  # noqa: E402
from substrate.control.artifacts import Artifacts, LocalArtifactStore  # noqa: E402
from substrate.control.fabric.adapters import NistIonEnergetics, PubChemCompound  # noqa: E402
from substrate.control.fabric.client import SafeHTTPClient  # noqa: E402
from substrate.control.fabric.service import Fabric  # noqa: E402
from substrate.control.observability import MetricsObserver, with_live_gauges  # noqa: E402
from substrate.control.orchestrator import Orchestrator  # noqa: E402
from substrate.control.registry import Registry  # noqa: E402
from substrate.control.security import Role  # noqa: E402
from substrate.control.worker import Worker  # noqa: E402
from substrate.experiment import parse_experiment, read_spec  # noqa: E402

from conftest import EXPERIMENTS  # noqa: E402


def spec_of(name):
    return read_spec(EXPERIMENTS / f"{name}.yaml")


REMOTE = {"url": "http://127.0.0.1:9"}                      # nothing listens here: the tests that fetch use the bundled set, and the ones that must fail rely on it


@pytest.fixture
def world(tmp_path):
    registry = Registry(f"sqlite:///{tmp_path / 'r.sqlite3'}")
    tenant, admin, admin_token = registry.bootstrap("lab")
    registry.set_quota(admin, "concurrent_runs", 50)
    metrics = with_live_gauges(MetricsObserver(), registry)
    orchestrator = Orchestrator(registry, Artifacts(registry, LocalArtifactStore(tmp_path / "store")), observer=metrics)
    fabric = Fabric(registry, orchestrator.artifacts, SafeHTTPClient({"127.0.0.1"}, allow_loopback=True, sleep=lambda s: None),
                    {"nist.ion_energetics": NistIonEnergetics(), "pubchem.compound": PubChemCompound(REMOTE["url"])})
    app = create_app(registry, orchestrator.artifacts, orchestrator, metrics=metrics, fabric=fabric)
    tokens = {"admin": admin_token}
    for name, role in (("ada", Role.RESEARCHER), ("vera", Role.VIEWER), ("olga", Role.OPERATOR), ("svc", Role.SERVICE)):
        tokens[name] = registry.issue_api_key(registry.create_principal(tenant, name, role).id)
    other_tenant, boris, tokens["boris"] = registry.bootstrap("lab-b", "boris")
    return {"registry": registry, "orch": orchestrator, "app": app, "client": TestClient(app, raise_server_exceptions=False), "tokens": tokens, "metrics": metrics,
            "tenant": tenant, "admin": admin, "tmp": tmp_path}


def h(world, who):
    return {"Authorization": f"Bearer {world['tokens'][who]}"}


def create_experiment(world, who="ada", name="proton_transfer", spec=None, expect=201):
    response = world["client"].post("/experiments", json={"name": name, "spec": spec or spec_of(name)}, headers=h(world, who))
    assert response.status_code == expect, response.text
    return response.json()


def run_and_wait(world, experiment_id, who="ada", n_samples=0):
    response = world["client"].post("/runs", json={"experiment_id": experiment_id, "n_samples": n_samples}, headers=h(world, who))
    assert response.status_code == 202, response.text
    Worker(world["orch"]).run_until_idle()
    return response.json()["id"]


# -- open endpoints and the surface -------------------------------------------------------------------------------------------------------
def test_health_and_readiness_need_no_key_and_the_docs_pages_are_off(world):
    c = world["client"]
    assert c.get("/healthz").json() == {"status": "ok", "version": "0.1.0"} and c.get("/readyz").json() == {"status": "ready"}
    assert c.get("/docs").status_code == 404 and c.get("/redoc").status_code == 404                 # these load their code from a CDN: off by default


def test_every_response_carries_the_security_headers_whatever_its_status(world):
    c = world["client"]
    for response in (c.get("/healthz"), c.get("/runs"), c.get("/nowhere"), c.post("/runs", content=b"{not json", headers=h(world, "ada"))):
        assert response.headers["x-content-type-options"] == "nosniff" and response.headers["cache-control"] == "no-store"
        assert "default-src 'none'" in response.headers["content-security-policy"] and response.headers["x-frame-options"] == "DENY"
        assert response.headers["referrer-policy"] == "no-referrer"


# -- authentication -----------------------------------------------------------------------------------------------------------------------
def test_no_key_or_a_bad_one_is_a_401_with_a_challenge_and_is_counted(world):
    c, m = world["client"], world["metrics"]
    missing = c.get("/runs")
    assert missing.status_code == 401 and missing.headers["www-authenticate"] == "Bearer"
    for bad in ("Bearer nonsense", "Bearer sbs_K00001_wrong", "Basic abc", "Bearer ", world["tokens"]["ada"], "bearer"):
        assert c.get("/runs", headers={"Authorization": bad}).status_code == 401, bad
    assert m.registry.get_sample_value("substrate_auth_failures_total", {"reason": "missing"}) == 1
    assert m.registry.get_sample_value("substrate_auth_failures_total", {"reason": "invalid"}) == 6


def test_a_key_in_the_url_or_a_cookie_is_not_a_credential(world):
    c = world["client"]
    assert c.get("/runs", params={"token": world["tokens"]["ada"], "api_key": world["tokens"]["ada"]}).status_code == 401
    assert c.get("/runs", cookies={"Authorization": f"Bearer {world['tokens']['ada']}"}).status_code == 401


def test_a_revoked_key_stops_working_at_once(world):
    from substrate.control.security import split_token
    c = world["client"]
    assert c.get("/runs", headers=h(world, "ada")).status_code == 200
    world["registry"].revoke_api_key(split_token(world["tokens"]["ada"])[0])
    assert c.get("/runs", headers=h(world, "ada")).status_code == 401


# -- roles ---------------------------------------------------------------------------------------------------------------------------------
def test_each_role_gets_exactly_the_routes_it_should(world):
    c = world["client"]
    exp = create_experiment(world)["id"]
    run_id = run_and_wait(world, exp)
    expected = {  # (method, path): who may call it
        ("GET", "/experiments"): {"admin", "ada", "vera"}, ("GET", f"/experiments/{exp}"): {"admin", "ada", "vera"},
        ("GET", "/runs"): {"admin", "ada", "vera", "olga"}, ("GET", f"/runs/{run_id}"): {"admin", "ada", "vera", "olga"},
        ("GET", f"/runs/{run_id}/artifacts"): {"admin", "ada", "vera"}, ("GET", f"/runs/{run_id}/lineage"): {"admin", "ada", "vera", "olga"},
        ("GET", "/models"): {"admin", "ada", "vera"}, ("GET", "/datasets"): {"admin", "ada", "vera", "olga"},
        ("GET", "/metrics"): {"admin", "olga"}, ("GET", "/audit"): {"admin", "olga"},
    }
    for (method, path), allowed in expected.items():
        for who in ("admin", "ada", "vera", "olga", "svc"):
            status = c.request(method, path, headers=h(world, who)).status_code
            assert (status == 200) == (who in allowed), (method, path, who, status)
            assert status in (200, 403)


def test_writing_routes_are_closed_to_the_roles_that_may_not_write(world):
    c = world["client"]
    exp = create_experiment(world)["id"]
    for who in ("vera", "olga", "svc"):
        assert c.post("/experiments", json={"name": "x", "spec": spec_of("proton_transfer")}, headers=h(world, who)).status_code == 403
        assert c.post("/runs", json={"experiment_id": exp}, headers=h(world, who)).status_code == 403
        assert c.post("/transfers", json={"references": [exp, exp]}, headers=h(world, who)).status_code == 403
    run_id = run_and_wait(world, exp)
    assert c.post(f"/runs/{run_id}/cancel", headers=h(world, "vera")).status_code == 403
    assert c.post(f"/reproduce/{run_id}", headers=h(world, "olga")).status_code == 403


# -- tenants ---------------------------------------------------------------------------------------------------------------------------------
def test_another_tenant_sees_nothing_and_cannot_tell_what_exists(world):
    c = world["client"]
    exp = create_experiment(world)["id"]
    run_id = run_and_wait(world, exp)
    sha = c.get(f"/runs/{run_id}/artifacts", headers=h(world, "ada")).json()["artifacts"][0]["sha256"]
    boris = h(world, "boris")
    for path in (f"/experiments/{exp}", f"/runs/{run_id}", f"/runs/{run_id}/artifacts", f"/runs/{run_id}/lineage", f"/artifacts/{sha}", f"/reproduce/{run_id}"):
        assert c.get(path, headers=boris).status_code == 404, path
        assert c.get(path.replace(exp, "EXP-999999").replace(run_id, "RUN-999999").replace(sha, "0" * 64), headers=boris).status_code == 404       # same answer as for absent
    assert c.get("/experiments", headers=boris).json() == {"experiments": []} and c.get("/runs", headers=boris).json() == {"runs": []}
    assert c.post("/runs", json={"experiment_id": exp}, headers=boris).status_code == 404
    assert c.post(f"/runs/{run_id}/cancel", headers=boris).status_code == 404
    assert c.post(f"/reproduce/{run_id}", headers=boris).status_code == 404
    assert c.get("/audit", headers=boris).json()["events"] == []


# -- bounded input -----------------------------------------------------------------------------------------------------------------------------
def test_a_body_over_the_limit_is_refused_whether_or_not_it_says_how_long_it_is(tmp_path):
    registry = Registry(f"sqlite:///{tmp_path / 'r.sqlite3'}")
    _, admin, token = registry.bootstrap("lab")
    orchestrator = Orchestrator(registry, Artifacts(registry, LocalArtifactStore(tmp_path / "s")))
    client = TestClient(create_app(registry, orchestrator.artifacts, orchestrator, max_body=2000), raise_server_exceptions=False)
    headers = {"Authorization": f"Bearer {token}"}
    big = {"name": "x", "spec": {"filler": "y" * 5000}}
    assert client.post("/experiments", json=big, headers=headers).status_code == 413
    assert client.post("/experiments", content=iter([b'{"name": "x", "spec": {"a": "' + b"z" * 1500, b"z" * 1500 + b'"}}']), headers=headers).status_code == 413       # chunked
    assert client.post("/experiments", content=b"x", headers={**headers, "Content-Length": "abc"}).status_code == 413            # a length that is not a number is not trusted
    assert client.post("/experiments", json={"name": "x", "spec": {"nonsense": True}}, headers=headers).status_code == 422       # under the limit it reaches the handler


@pytest.mark.parametrize("body", [
    {"name": "x", "spec": {}, "role": "ADMIN"},                                             # a field that does not exist
    {"name": "", "spec": {}}, {"name": "x"}, {"spec": {}}, {"name": "x", "spec": "text"}, {"name": "x" * 201, "spec": {}},
    {"name": "x", "spec": {}, "datasets": [{"id": "d", "version": 0}]}, {"name": "x", "spec": {}, "datasets": [{"id": "d", "version": 1, "extra": 1}]},
])
def test_malformed_experiment_bodies_are_422(world, body):
    assert world["client"].post("/experiments", json=body, headers=h(world, "ada")).status_code == 422


@pytest.mark.parametrize("body", [{"experiment_id": "x"}, {"experiment_id": "EXP-1"}, {"experiment_id": "EXP-000001", "seed": 1.5}, {"experiment_id": "EXP-000001", "n_samples": -1},
                                  {"experiment_id": "EXP-000001", "n_samples": "ten"}, {"experiment_id": "EXP-000001", "extra": True}, {}])
def test_malformed_run_bodies_are_422(world, body):
    assert world["client"].post("/runs", json=body, headers=h(world, "ada")).status_code == 422


@pytest.mark.parametrize("query", ["limit=0", "limit=100000", "limit=-3", "offset=-1", "limit=abc", "experiment_id=not-an-id"])
def test_pagination_and_filters_are_bounded(world, query):
    assert world["client"].get(f"/runs?{query}", headers=h(world, "ada")).status_code == 422


def test_a_spec_that_could_never_run_is_refused_when_stored_and_nothing_is_stored(world):
    bad_number = copy.deepcopy(spec_of("proton_transfer"))
    (bad_number["experiment"] if "experiment" in bad_number else bad_number)["system"]["parameters"]["mass"]["sigma"] = -1
    unroutable = copy.deepcopy(spec_of("proton_transfer"))
    (unroutable["experiment"] if "experiment" in unroutable else unroutable)["propagation"] = ["electronic_structure"]
    for spec in (bad_number, unroutable, {"nonsense": True}):
        response = world["client"].post("/experiments", json={"name": "bad", "spec": spec}, headers=h(world, "ada"))
        assert response.status_code == 422 and "detail" in response.json()
    assert world["client"].get("/experiments", headers=h(world, "ada")).json() == {"experiments": []}


# -- errors that do not leak -----------------------------------------------------------------------------------------------------------------
def test_limit_quota_and_missing_things_have_their_own_status_codes(world):
    c = world["client"]
    exp = create_experiment(world)["id"]
    assert c.post("/runs", json={"experiment_id": exp, "n_samples": 10**6}, headers=h(world, "ada")).status_code == 413                  # a resource limit
    assert c.post("/runs", json={"experiment_id": "EXP-999999"}, headers=h(world, "ada")).status_code == 404
    world["registry"].set_quota(world["admin"], "runs_per_day", 1)
    assert c.post("/runs", json={"experiment_id": exp}, headers=h(world, "ada")).status_code == 202
    blocked = c.post("/runs", json={"experiment_id": exp}, headers=h(world, "ada"))
    assert blocked.status_code == 429 and blocked.json()["kind"] == "quota"
    assert c.get("/runs/RUN-999999", headers=h(world, "ada")).status_code == 404 and c.get("/artifacts/" + "f" * 64, headers=h(world, "ada")).status_code == 404
    assert c.get("/artifacts/not-a-hash", headers=h(world, "ada")).status_code == 404 and c.get("/datasets/none", headers=h(world, "ada")).status_code == 404


def test_an_unexpected_failure_is_a_500_that_says_nothing_about_the_server(world, monkeypatch):
    def explode(*a, **k):
        raise RuntimeError("secret detail at C:\\\\Users\\\\private\\\\db.sqlite3 with password=hunter2")

    monkeypatch.setattr(world["orch"], "describe", explode)
    response = world["client"].get("/runs/RUN-000001", headers=h(world, "ada"))
    assert response.status_code == 500 and response.json() == {"detail": "internal error"}
    assert "hunter2" not in response.text and "private" not in response.text and "Traceback" not in response.text


def test_a_missing_quantum_chemistry_program_is_a_503_without_details(world, monkeypatch):
    from substrate import QCError

    def no_program(*a, **k):
        raise QCError("PySCF worker failed (exit 1): /home/user/.substrate-qc/bin/python: not found")

    monkeypatch.setattr(world["orch"], "submit", no_program)
    exp = create_experiment(world)["id"]
    response = world["client"].post("/runs", json={"experiment_id": exp}, headers=h(world, "ada"))
    assert response.status_code == 503 and "substrate-qc" not in response.text


def test_non_finite_numbers_are_written_as_tagged_values_not_a_crash():
    body = SafeJSONResponse({"x": float("nan"), "y": [float("inf"), 1.0]}).body.decode()
    assert body == '{"x":{"__float__":"nan"},"y":[{"__float__":"inf"},1.0]}'


# -- the life of a run -------------------------------------------------------------------------------------------------------------------------
def test_a_run_from_creation_to_its_artifacts_and_lineage(world):
    c = world["client"]
    experiment = create_experiment(world)
    assert experiment["id"] == "EXP-000001" and experiment["spec_sha256"] and c.get(f"/experiments/{experiment['id']}", headers=h(world, "vera")).json()["name"] == "proton_transfer"
    run_id = run_and_wait(world, experiment["id"])
    run = c.get(f"/runs/{run_id}", headers=h(world, "vera")).json()
    assert run["state"] == "succeeded" and run["progress"] == {"succeeded": 7} and run["status"]["execution"]["state"] == "PASS" and len(run["tasks"]) == 7
    assert [r["id"] for r in c.get("/runs", params={"experiment_id": experiment["id"]}, headers=h(world, "ada")).json()["runs"]] == [run_id]

    artifacts = c.get(f"/runs/{run_id}/artifacts", headers=h(world, "ada")).json()["artifacts"]
    names = {a["name"]: a for a in artifacts}
    assert {"report.json", "report.md", "final_observables.json", "status.json"} <= set(names)
    report = c.get(f"/artifacts/{names['report.json']['sha256']}", headers=h(world, "vera"))
    assert report.status_code == 200 and report.headers["content-type"].startswith("application/json") and report.json()["run"] == run_id
    assert report.headers["etag"] == f'"{names["report.json"]["sha256"]}"' and "content-disposition" not in report.headers
    markdown = c.get(f"/artifacts/{names['report.md']['sha256']}", headers=h(world, "vera"))
    assert markdown.headers["content-type"].startswith("text/markdown") and markdown.headers["content-disposition"].startswith("attachment") and b"## Route" in markdown.content

    lineage = c.get(f"/runs/{run_id}/lineage", headers=h(world, "ada")).json()
    assert lineage["intact"] is True and [e["payload"]["index"] for e in lineage["events"]] == [0, 1, 2]


def test_cancelling_over_the_api(world):
    c = world["client"]
    exp = create_experiment(world)["id"]
    run_id = c.post("/runs", json={"experiment_id": exp}, headers=h(world, "ada")).json()["id"]
    cancelled = c.post(f"/runs/{run_id}/cancel", headers=h(world, "olga"))
    assert cancelled.status_code == 200 and cancelled.json() == {"cancelled_tasks": 8}                    # this example asks for a 100-draw ensemble, so its run has 8 tasks
    assert c.get(f"/runs/{run_id}", headers=h(world, "ada")).json()["state"] == "cancelled"


def test_reproducing_over_the_api_and_reading_the_verdict(world):
    c = world["client"]
    exp = create_experiment(world)["id"]
    run_id = run_and_wait(world, exp, n_samples=3)
    again = c.post(f"/reproduce/{run_id}", headers=h(world, "ada"))
    assert again.status_code == 202 and again.json()["reproduces"] == run_id
    Worker(world["orch"]).run_until_idle()
    report = c.get(f"/reproduce/{run_id}", headers=h(world, "vera")).json()
    assert report["original"] == run_id and len(report["reproductions"]) == 1
    only = report["reproductions"][0]
    assert only["state"] == "succeeded" and only["reproducibility"]["state"] == "PASS" and len(only["result_digest"]) == 64 == len(report["result_digest"])      # digests are not compared: see test_control_orchestrator
    cancelled = c.post("/runs", json={"experiment_id": exp}, headers=h(world, "ada")).json()["id"]
    c.post(f"/runs/{cancelled}/cancel", headers=h(world, "olga"))
    assert c.post(f"/reproduce/{cancelled}", headers=h(world, "ada")).status_code == 422                       # only a run that succeeded can be reproduced


# -- reading the platform ------------------------------------------------------------------------------------------------------------------------
def test_the_models_endpoint_lists_every_engine_and_translator_with_its_approximations(world):
    models = world["client"].get("/models", headers=h(world, "vera")).json()
    engine_names = {e["name"] for e in models["engines"]}
    assert {"quantum.double_well_1d"} <= {k for e in models["engines"] for k in e["kinds"]} and len(engine_names) >= 6
    assert all(e["approximations"] is not None and e["scale"] for e in models["engines"]) and len(models["translators"]) >= 5
    assert all(t["approximations"] for t in models["translators"])


def test_datasets_are_listed_and_read_by_version(world):
    world["registry"].put_dataset("nist.pa", source="NIST", license="public domain", schema={"x": "float"}, records=[{"x": 1.0}], provenance={})
    world["registry"].put_dataset("nist.pa", source="NIST", license="public domain", schema={"x": "float"}, records=[{"x": 2.0}], provenance={})
    c = world["client"]
    assert [d["version"] for d in c.get("/datasets", headers=h(world, "vera")).json()["datasets"]] == [1, 2]
    assert c.get("/datasets/nist.pa", headers=h(world, "vera")).json()["version"] == 2 and c.get("/datasets/nist.pa/1", headers=h(world, "vera")).json()["records"] == [{"x": 1.0}]
    assert c.get("/datasets/nist.pa/9", headers=h(world, "vera")).status_code == 404


def test_datasets_are_ingested_through_the_api_by_those_who_may_and_the_adapters_are_listed(world):
    c = world["client"]
    adapters = c.get("/datasets/adapters", headers=h(world, "vera")).json()["adapters"]
    assert {a["id"] for a in adapters} == {"nist.ion_energetics", "pubchem.compound"} and c.get("/datasets/adapters").status_code == 401
    created = c.post("/datasets/ingest", json={"adapter": "nist.ion_energetics"}, headers=h(world, "ada"))
    assert created.status_code == 201 and created.json()["dataset_id"] == "nist.ion_energetics" and created.json()["version"] == 1 and len(created.json()["records"]) == 10
    assert c.get("/datasets/nist.ion_energetics/1", headers=h(world, "vera")).json()["checksum"] == created.json()["checksum"]
    for who in ("vera", "olga", "svc"):
        assert c.post("/datasets/ingest", json={"adapter": "nist.ion_energetics"}, headers=h(world, who)).status_code == 403
    assert c.post("/datasets/ingest", json={"adapter": "nope"}, headers=h(world, "ada")).status_code == 422
    assert c.post("/datasets/ingest", json={"adapter": "nist.ion_energetics", "query": {"x": 1}}, headers=h(world, "ada")).status_code == 422
    assert c.post("/datasets/ingest", json={"adapter": "pubchem.compound", "query": {"name": "../../etc/passwd"}}, headers=h(world, "ada")).status_code == 422
    assert c.post("/datasets/ingest", json={"adapter": "x", "extra": 1}, headers=h(world, "ada")).status_code == 422
    unreachable = c.post("/datasets/ingest", json={"adapter": "pubchem.compound", "query": {"name": "water"}}, headers=h(world, "ada"))
    assert unreachable.status_code == 502 and unreachable.json()["kind"] == "upstream" and "127.0.0.1:9" in unreachable.json()["detail"]


def test_without_a_fabric_ingestion_is_a_404(world):
    app = create_app(world["registry"], world["orch"].artifacts, world["orch"])
    response = TestClient(app).post("/datasets/ingest", json={"adapter": "nist.ion_energetics"}, headers=h(world, "ada"))
    assert response.status_code == 404 and TestClient(app).get("/datasets/adapters", headers=h(world, "ada")).json() == {"adapters": []}


def test_the_audit_trail_shows_what_happened_and_notices_tampering(world):
    import sqlalchemy as sa
    c = world["client"]
    exp = create_experiment(world)["id"]
    run_and_wait(world, exp)
    c.post("/runs", json={"experiment_id": exp, "n_samples": 10**6}, headers=h(world, "ada"))
    audit = c.get("/audit", headers=h(world, "olga")).json()
    kinds = [e["kind"] for e in audit["events"]]
    assert audit["intact"] is True and kinds[:2] == ["quota.set", "experiment.create"] and "run.create" in kinds and "run.finish" in kinds
    seq = audit["events"][1]["seq"]
    with world["registry"].engine.begin() as conn:
        conn.execute(sa.update(db.events).where(db.events.c.seq == seq).values(payload={"edited": True}))
    after = c.get("/audit", headers=h(world, "olga")).json()
    assert after["intact"] is False and after["first_bad_event"] == seq


def test_the_metrics_endpoint_is_for_operators_and_shows_the_work_done(world):
    c = world["client"]
    run_and_wait(world, create_experiment(world)["id"])
    assert c.get("/metrics").status_code == 401 and c.get("/metrics", headers=h(world, "vera")).status_code == 403
    text = c.get("/metrics", headers=h(world, "olga")).text
    assert "substrate_runs_submitted_total 1.0" in text and 'substrate_runs_finished_total{overall="PASS",state="succeeded"} 1.0' in text
    assert "substrate_auth_failures_total" in text and 'substrate_auth_failures_total{reason="missing"} 1.0' in text and "substrate_build_info" in text


def test_without_metrics_configured_the_endpoint_is_a_404(world):
    app = create_app(world["registry"], world["orch"].artifacts, world["orch"])
    assert TestClient(app).get("/metrics", headers=h(world, "olga")).status_code == 404


# -- transfer studies ----------------------------------------------------------------------------------------------------------------------------
def test_a_transfer_study_runs_the_real_cross_prediction_and_can_be_reproduced(world):
    c = world["client"]
    base = spec_of("zundel_calibrated")
    deeper = copy.deepcopy(base)
    (deeper["experiment"] if "experiment" in deeper else deeper)["system"]["parameters"]["morse_depth"]["value"] = 5.0
    first = create_experiment(world, name="reference a", spec=base)["id"]
    second = create_experiment(world, name="reference c", spec=deeper)["id"]
    response = c.post("/transfers", json={"name": "a and c", "references": [first, second]}, headers=h(world, "ada"))
    assert response.status_code == 202, response.text
    run_id = response.json()["id"]
    Worker(world["orch"]).run_until_idle()
    study = c.get(f"/transfers/{run_id}", headers=h(world, "vera")).json()
    assert study["state"] == "succeeded" and study["status"]["execution"]["state"] == "PASS" and study["status"]["validation"]["state"] == "PASS"
    matrix = study["matrix"]
    assert matrix["names"] == [first, second] and set(matrix["pairs"]) == {f"{a} -> {b}" for a in (first, second) for b in (first, second)}
    cross = matrix["pairs"][f"{first} -> {second}"]
    assert cross["rmse_window_ev"] > 0.01 and matrix["pairs"][f"{first} -> {first}"]["rmse_window_ev"] < 1e-6 and cross["window_ev"] == 1.5
    assert "->" in matrix["text"] or first in matrix["text"]

    # it is an experiment like any other: stored, listed, and reproducible
    assert any(e["name"] == "a and c" for e in c.get("/experiments", headers=h(world, "ada")).json()["experiments"])
    again = c.post(f"/reproduce/{run_id}", headers=h(world, "ada"))
    assert again.status_code == 202
    Worker(world["orch"]).run_until_idle()
    verdict = c.get(f"/reproduce/{run_id}", headers=h(world, "ada")).json()["reproductions"][0]
    assert verdict["state"] == "succeeded" and verdict["reproducibility"]["state"] == "PASS"


@pytest.mark.parametrize("body", [{"references": ["EXP-000001"]}, {"references": []}, {"references": ["EXP-000001", "EXP-000001"]}, {"references": ["a", "b"], "window_ev": -1},
                                  {"references": ["a", "b"], "window_ev": 0}, {"references": ["a", "b"], "sigma_ev": 0}, {"references": ["a", "b"], "oops": 1}])
def test_a_transfer_study_needs_two_distinct_references_and_sane_numbers(world, body):
    assert world["client"].post("/transfers", json=body, headers=h(world, "ada")).status_code in (422,)


def test_references_must_exist_in_your_own_tenant_and_must_not_be_studies_themselves(world):
    c = world["client"]
    mine = create_experiment(world, name="mine", spec=spec_of("zundel_calibrated"))["id"]
    theirs = c.post("/experiments", json={"name": "theirs", "spec": spec_of("zundel_calibrated")}, headers=h(world, "boris")).json()["id"]
    assert c.post("/transfers", json={"references": [mine, theirs]}, headers=h(world, "ada")).status_code == 422          # boris's experiment is not visible to ada
    assert c.post("/transfers", json={"references": [mine, "EXP-999999"]}, headers=h(world, "ada")).status_code == 422
    other = create_experiment(world, name="other", spec=spec_of("zundel_calibrated"))["id"]
    study = c.post("/transfers", json={"references": [mine, other]}, headers=h(world, "ada")).json()
    assert c.get(f"/transfers/{study['id']}", headers=h(world, "boris")).status_code == 404
    spec_id = world["registry"].get_run(world["admin"], study["id"])["experiment_id"]
    assert c.post("/transfers", json={"references": [mine, spec_id]}, headers=h(world, "ada")).status_code == 422          # a study cannot be a reference
