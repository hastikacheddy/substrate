"""The quantum gateway and the quantum backend inside a real run: policy and quota decide whether a task may start, and the answer matches the classical one."""
import copy
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")

from substrate import Pipeline, ResourceLimitError, ValidationError, default_registry  # noqa: E402
from substrate.control.artifacts import Artifacts, LocalArtifactStore  # noqa: E402
from substrate.control.gateway import QPUGateway  # noqa: E402
from substrate.control.orchestrator import REPRODUCTION_RTOL, Orchestrator  # noqa: E402
from substrate.control.registry import Registry  # noqa: E402
from substrate.control.repro import compare_summaries, observables_summary  # noqa: E402
from substrate.control.security import QuotaExceededError, Role  # noqa: E402
from substrate.control.worker import Worker  # noqa: E402
from substrate.experiment import parse_experiment, read_spec  # noqa: E402
from substrate.qpu import QuantumPolicy, QuantumPolicyError  # noqa: E402

from conftest import EXPERIMENTS  # noqa: E402

QUANTUM = {"type": "quantum", "provider": "simulator"}


def spec_with(backend=None, name="proton_transfer_electronic"):
    spec = copy.deepcopy(read_spec(EXPERIMENTS / f"{name}.yaml"))
    body = spec["experiment"] if "experiment" in spec else spec
    if backend is not None:
        body["backend"] = backend
    body.setdefault("ensemble", {})["n_samples"] = 0
    return spec


def make_world(tmp_path, policy=None):
    registry = Registry(f"sqlite:///{tmp_path / 'r.sqlite3'}")
    tenant, admin, _ = registry.bootstrap("lab")
    registry.set_quota(admin, "concurrent_runs", 20)
    gateway = QPUGateway(registry, policy)
    orchestrator = Orchestrator(registry, Artifacts(registry, LocalArtifactStore(tmp_path / "s")), gateway=gateway)
    return {"registry": registry, "tenant": tenant, "admin": admin, "gateway": gateway, "orch": orchestrator, "who": registry.create_principal(tenant, "ada", Role.RESEARCHER)}


@pytest.fixture
def world(tmp_path):
    return make_world(tmp_path)


def run(world, spec, queues=("cpu", "chemistry", "quantum")):
    experiment = world["registry"].add_experiment(world["who"], "x", spec)
    submitted = world["orch"].submit(world["who"], experiment["id"], n_samples=0)
    Worker(world["orch"], queues=queues).run_until_idle()
    return submitted["id"], world["orch"].describe(world["who"], submitted["id"])


def events(world, kind):
    return [e for e in world["registry"].events(f"audit:{world['tenant']}", world["tenant"]) if e["kind"] == kind]


# -- the gateway on its own -------------------------------------------------------------------------------------------------------------------
def test_a_classical_task_does_not_touch_the_gateway(world):
    assert world["gateway"].authorize(world["tenant"], world["who"].id, "RUN-1", "RUN-1/00", {"type": "classical", "provider": "scipy"}) is False
    assert events(world, "qpu.authorize") == []


def test_the_simulator_is_authorised_and_the_authorisation_is_recorded(world):
    assert world["gateway"].authorize(world["tenant"], world["who"].id, "RUN-1", "RUN-1/00", QUANTUM) is True
    event = events(world, "qpu.authorize")[0]["payload"]
    assert event == {"run": "RUN-1", "task": "RUN-1/00", "provider": "simulator", "hardware": False}
    assert world["registry"].used_today(world["who"], "qpu_jobs_per_day") == 0                      # a simulator is not metered


def test_a_provider_outside_the_policy_is_blocked_and_audited(tmp_path):
    w = make_world(tmp_path, QuantumPolicy(providers=("simulator",)))
    with pytest.raises(QuantumPolicyError, match="'ibm_brisbane' is not allowed"):
        w["gateway"].authorize(w["tenant"], w["who"].id, "RUN-1", "RUN-1/00", {"type": "quantum", "provider": "ibm_brisbane"})
    blocked = events(w, "qpu.blocked")[0]["payload"]
    assert blocked["result"] == "BLOCKED" and blocked["provider"] == "ibm_brisbane" and blocked["run"] == "RUN-1" and "not allowed" in blocked["reason"]


def test_a_device_costs_one_unit_of_the_submitters_allowance_per_task_and_the_allowance_starts_at_zero(tmp_path):
    w = make_world(tmp_path, QuantumPolicy(providers=("simulator",), hardware_providers=("simulator",)))     # the simulator, declared a device, to exercise the metering
    with pytest.raises(QuotaExceededError, match="qpu_jobs_per_day"):
        w["gateway"].authorize(w["tenant"], w["who"].id, "RUN-1", "RUN-1/00", QUANTUM)
    assert events(w, "quota.blocked")[-1]["payload"]["metric"] == "qpu_jobs_per_day" and events(w, "qpu.authorize") == []
    w["registry"].set_quota(w["admin"], "qpu_jobs_per_day", 2)
    for n in range(2):
        assert w["gateway"].authorize(w["tenant"], w["who"].id, "RUN-1", f"RUN-1/0{n}", QUANTUM) is True
    with pytest.raises(QuotaExceededError):
        w["gateway"].authorize(w["tenant"], w["who"].id, "RUN-1", "RUN-1/02", QUANTUM)
    assert w["registry"].used_today(w["who"], "qpu_jobs_per_day") == 2 and [e["payload"]["hardware"] for e in events(w, "qpu.authorize")] == [True, True]


def test_a_device_job_for_a_submitter_who_no_longer_exists_cannot_be_charged(tmp_path):
    w = make_world(tmp_path, QuantumPolicy(providers=("simulator",), hardware_providers=("simulator",)))
    with pytest.raises(QuantumPolicyError, match="no longer exists"):
        w["gateway"].authorize(w["tenant"], "P-9999", "RUN-1", "RUN-1/00", QUANTUM)


# -- inside a run ---------------------------------------------------------------------------------------------------------------------------
def test_the_same_experiment_on_the_quantum_backend_gives_the_classical_answers(world):
    _, classical = run(world, spec_with())
    quantum_id, quantum = run(world, spec_with(QUANTUM))
    assert classical["state"] == quantum["state"] == "succeeded"
    classical_id = classical["id"]
    a, b = (world["orch"].final_observables(world["tenant"], r) for r in (classical_id, quantum_id))
    comparison = compare_summaries(a, b, rtol=REPRODUCTION_RTOL)
    assert comparison.matches and comparison.max_relative_difference < 1e-9, comparison.to_dict()
    assert len(events(world, "qpu.authorize")) == 5 and all(e["payload"]["hardware"] is False for e in events(world, "qpu.authorize"))   # every step of the quantum run


def test_the_provenance_and_the_report_say_which_solver_ran(world):
    run_id, _ = run(world, spec_with(QUANTUM))
    report_sha = [a for a in world["registry"].run_artifacts(world["admin"], run_id) if a["name"] == "report.json"][0]["sha256"]
    report = world["orch"].artifacts.read_json(world["tenant"], report_sha)
    backends = {s["name"]: s["backend"] for s in report["steps"]}
    assert report["backend"] == {"type": "quantum", "provider": "simulator"} and backends["electronic.evb_two_state"] == "quantum.simulator"
    assert all(b is None for name, b in backends.items() if "." in name and name.startswith(("electronic_to", "quantum_to")))        # translators have no solver


def test_the_steps_of_a_quantum_experiment_wait_for_a_worker_that_serves_the_quantum_queue(world):
    experiment = world["registry"].add_experiment(world["who"], "x", spec_with(QUANTUM))
    submitted = world["orch"].submit(world["who"], experiment["id"], n_samples=0)
    Worker(world["orch"], queues=("cpu", "chemistry")).run_until_idle()
    description = world["orch"].describe(world["who"], submitted["id"])
    assert description["state"] == "running" and {t["queue"] for t in description["tasks"] if t["kind"] in ("solve", "translate")} == {"quantum"}
    assert [t["state"] for t in description["tasks"]][:2] == ["succeeded", "succeeded"]                  # validate and ingest ran; the first step is waiting
    Worker(world["orch"], queues=("quantum",)).run_until_idle()
    Worker(world["orch"], queues=("cpu",)).run_until_idle()
    assert world["orch"].describe(world["who"], submitted["id"])["state"] == "succeeded"


def test_a_job_larger_than_the_policy_allows_fails_the_security_axis_without_a_retry(tmp_path):
    w = make_world(tmp_path, QuantumPolicy(max_circuits=1000))
    run_id, description = run(w, spec_with(QUANTUM))
    failed = [t for t in description["tasks"] if t["state"] == "failed"]
    assert len(failed) == 1 and failed[0]["attempts"] == 1 and failed[0]["error"]["type"] == "QuantumPolicyError" and failed[0]["error"]["axis"] == "security"
    assert "circuits exceeds the policy limit of 1,000" in failed[0]["error"]["message"]
    assert description["status"]["security"]["state"] == "FAIL" and description["status"]["execution"]["state"] == "PASS"
    assert events(w, "policy.blocked")[0]["payload"]["error"] == "QuantumPolicyError"


def test_a_provider_the_policy_does_not_allow_stops_the_first_step_at_the_gateway(tmp_path):
    w = make_world(tmp_path, QuantumPolicy(providers=("elsewhere",)))
    _, description = run(w, spec_with(QUANTUM))
    failed = [t for t in description["tasks"] if t["state"] == "failed"][0]
    assert failed["name"].startswith("00-") and failed["error"]["type"] == "QuantumPolicyError" and description["status"]["security"]["state"] == "FAIL"
    assert events(w, "qpu.blocked") and not events(w, "qpu.authorize")


def test_a_quantum_run_in_isolated_processes_obeys_the_same_policy(tmp_path):
    from substrate.control.sandbox import ProcessExecutor
    policy = QuantumPolicy(max_circuits=1000)
    w = make_world(tmp_path, policy)
    w["orch"].executor = ProcessExecutor(memory_bytes=None, quantum_policy=policy)
    _, description = run(w, spec_with(QUANTUM))
    failed = [t for t in description["tasks"] if t["state"] == "failed"]
    assert len(failed) == 1 and failed[0]["error"]["type"] == "QuantumPolicyError" and failed[0]["attempts"] == 1       # rebuilt as its own class, so it is not retried


def test_an_isolated_quantum_run_gives_the_same_answer(tmp_path):
    from substrate.control.sandbox import ProcessExecutor
    w = make_world(tmp_path)
    w["orch"].executor = ProcessExecutor(memory_bytes=None, quantum_policy=w["gateway"].policy)
    run_id, description = run(w, spec_with(QUANTUM, "proton_transfer_electronic"))
    assert description["state"] == "succeeded"
    exp = parse_experiment(spec_with())
    direct = Pipeline(default_registry()).run(exp.system, exp.propagation).final
    assert compare_summaries(observables_summary(direct.observables), w["orch"].final_observables(w["tenant"], run_id), rtol=REPRODUCTION_RTOL).matches


# -- the backend block of an experiment --------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("block", [{"type": "quantum", "provider": "ibm_brisbane"}, {"type": "hpc"}, {"type": "quantum", "provider": "simulator", "token": "x"}, "quantum", 5,
                                   {"type": "classical", "provider": "cuda"}])
def test_an_experiment_naming_a_backend_that_is_not_provided_is_refused_when_parsed(block):
    with pytest.raises(ValidationError):
        parse_experiment(spec_with(block))


def test_the_default_and_the_explicit_blocks_normalise_to_the_same_thing():
    assert parse_experiment(spec_with()).backend == {"type": "classical", "provider": "scipy"}
    assert parse_experiment(spec_with({"type": "quantum"})).backend == {"type": "quantum", "provider": "simulator"}
    assert parse_experiment(spec_with({"type": "quantum", "provider": "simulator", "device": "aer"})).backend == {"type": "quantum", "provider": "simulator", "device": "aer"}


def test_a_run_of_a_quantum_experiment_is_reproducible_like_any_other(world):
    run_id, _ = run(world, spec_with(QUANTUM))
    again = world["orch"].reproduce(world["who"], run_id)
    Worker(world["orch"], queues=("cpu", "quantum")).run_until_idle()
    description = world["orch"].describe(world["who"], again["id"])
    assert description["status"]["reproducibility"]["state"] == "PASS"
