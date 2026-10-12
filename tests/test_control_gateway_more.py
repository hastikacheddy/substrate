"""Cases of the quantum gateway that the first tests did not pin, found by mutation testing."""
import pytest

pytest.importorskip("sqlalchemy")

from test_control_gateway import make_world  # noqa: E402


def test_a_quantum_block_that_names_no_provider_means_the_simulator_and_is_allowed(tmp_path):
    world = make_world(tmp_path)
    assert world["gateway"].authorize(world["tenant"], world["who"].id, "RUN-000001", "RUN-000001/00", {"type": "quantum"}) is True
    events = [e for e in world["registry"].events(f"audit:{world['tenant']}", world["tenant"]) if e["kind"] == "qpu.authorize"]
    assert len(events) == 1 and events[0]["payload"]["provider"] == "simulator" and events[0]["payload"]["hardware"] is False


def test_in_a_whole_run_the_submitter_and_only_the_submitter_is_charged_for_each_quantum_step(tmp_path):
    from substrate.control.security import Role
    from substrate.control.worker import Worker
    from substrate.qpu import QuantumPolicy
    from test_control_gateway import QUANTUM, spec_with

    w = make_world(tmp_path, QuantumPolicy(providers=("simulator",), hardware_providers=("simulator",)))      # the simulator, declared a device, so that every step is metered
    other = w["registry"].create_principal(w["tenant"], "bob", Role.RESEARCHER)
    w["registry"].set_quota(w["admin"], "qpu_jobs_per_day", 50)
    experiment = w["registry"].add_experiment(w["who"], "x", spec_with(QUANTUM))
    run = w["orch"].submit(w["who"], experiment["id"], n_samples=0)
    Worker(w["orch"], queues=("cpu", "quantum")).run_until_idle()
    assert w["orch"].describe(w["who"], run["id"])["state"] == "succeeded"
    steps = [e for e in w["registry"].events(f"audit:{w['tenant']}", w["tenant"]) if e["kind"] == "qpu.authorize"]
    assert len(steps) == 5 and all(e["payload"]["hardware"] is True for e in steps)
    assert w["registry"].used_today(w["who"], "qpu_jobs_per_day") == 5                                   # the person who submitted paid for every step
    assert w["registry"].used_today(other, "qpu_jobs_per_day") == 0 and w["registry"].used_today(w["admin"], "qpu_jobs_per_day") == 0
