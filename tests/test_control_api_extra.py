"""Three more cases of the API found by mutation testing: the metrics media type, a readiness check that really asks the database, and an audit trail that belongs to the caller's own tenant."""
import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")

from test_control_api import create_experiment, h, spec_of, world  # noqa: E402,F401


def test_metrics_are_served_in_the_prometheus_text_format(world):
    response = world["client"].get("/metrics", headers=h(world, "olga"))
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/plain; version=0.0.4")
    assert "# TYPE substrate_queue_depth gauge" in response.text


def test_readiness_means_the_database_answers(world, monkeypatch):
    assert world["client"].get("/readyz").status_code == 200

    def down(*args, **kwargs):
        raise ConnectionError("the database is not there")

    monkeypatch.setattr(world["registry"].engine, "connect", down)
    response = world["client"].get("/readyz")
    assert response.status_code >= 500 and "database" not in response.text                       # not ready, and the reason is not told to the caller
    assert world["client"].get("/healthz").status_code == 200                                      # being alive does not depend on it


def test_every_tenant_reads_its_own_audit_trail_and_not_another(world):
    c = world["client"]
    create_experiment(world, "ada", "proton_transfer")
    mine = c.get("/audit", headers=h(world, "olga")).json()
    assert mine["intact"] is True and any(e["kind"] == "experiment.create" for e in mine["events"])
    # the other tenant (boris, administrator of lab-b) has made nothing yet, and sees none of lab's events
    theirs = c.get("/audit", headers=h(world, "boris")).json()
    assert theirs["intact"] is True and not any(e["kind"] == "experiment.create" for e in theirs["events"])
    response = c.post("/experiments", json={"name": "b", "spec": spec_of("proton_transfer")}, headers=h(world, "boris"))
    assert response.status_code == 201
    after = c.get("/audit", headers=h(world, "boris")).json()
    assert [e["kind"] for e in after["events"]].count("experiment.create") == 1 and all(e["payload"].get("by") != world["admin"].id for e in after["events"])
    assert [e["kind"] for e in c.get("/audit", headers=h(world, "olga")).json()["events"]].count("experiment.create") == 1       # and lab's trail did not gain it
