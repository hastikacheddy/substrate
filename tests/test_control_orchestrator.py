"""The control plane end to end: submit an experiment, let workers run its tasks, and check what comes out and how each kind of failure is recorded."""
import copy
import threading
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")

from substrate import NoPathError, Pipeline, ResourceLimitError, ValidationError, default_registry  # noqa: E402
from substrate.control import tasks as tasks_module  # noqa: E402
from substrate.control.artifacts import Artifacts, LocalArtifactStore  # noqa: E402
from substrate.control.orchestrator import REPRODUCTION_RTOL, Observer, Orchestrator  # noqa: E402
from substrate.control.registry import Registry  # noqa: E402
from substrate.control.repro import compare_summaries, observables_summary, result_digest  # noqa: E402
from substrate.control.security import AuthorizationError, QuotaExceededError, Role  # noqa: E402
from substrate.control.status import State  # noqa: E402
from substrate.control.steps import InlineExecutor  # noqa: E402
from substrate.control.tasks import HANDLERS  # noqa: E402
from substrate.control.worker import Worker  # noqa: E402
from substrate.experiment import parse_experiment, read_spec  # noqa: E402

EXPERIMENTS = Path(__file__).resolve().parent.parent / "experiments"


def spec_of(name):
    return read_spec(EXPERIMENTS / f"{name}.yaml")


def system_params(spec):
    return (spec["experiment"] if "experiment" in spec else spec)["system"]["parameters"]


class Clock:
    def __init__(self):
        self.t = 1_800_000_000.0

    def __call__(self):
        return self.t


@pytest.fixture
def world(tmp_path):
    clock = Clock()
    registry = Registry(f"sqlite:///{tmp_path / 'r.sqlite3'}", clock=clock)
    tenant, admin, _ = registry.bootstrap("lab")
    registry.set_quota(admin, "concurrent_runs", 50)
    artifacts = Artifacts(registry, LocalArtifactStore(tmp_path / "store"))
    orchestrator = Orchestrator(registry, artifacts)
    return {"clock": clock, "registry": registry, "tenant": tenant, "admin": admin, "artifacts": artifacts, "orch": orchestrator,
            "researcher": registry.create_principal(tenant, "ada", Role.RESEARCHER), "viewer": registry.create_principal(tenant, "vera", Role.VIEWER),
            "operator": registry.create_principal(tenant, "olga", Role.OPERATOR), "tmp": tmp_path}


def experiment(world, name="proton_transfer_pathway", spec=None, datasets=None, who=None):
    return world["registry"].add_experiment(who or world["researcher"], name, spec or spec_of(name), datasets)["id"]


def work(world, queues=("cpu", "chemistry"), **kw):
    return Worker(world["orch"], queues=queues, **kw).run_until_idle()


def run_to_end(world, name="proton_transfer_pathway", n_samples=0, spec=None, who=None, **kw):
    who = who or world["researcher"]
    run = world["orch"].submit(who, experiment(world, name, spec, who=who), n_samples=n_samples, **kw)
    work(world)
    return run["id"], world["orch"].describe(who, run["id"])


def states(description):
    return {a: s["state"] for a, s in description["status"].items()}


# -- the happy path ---------------------------------------------------------------------------------------------------------------------
def test_a_run_through_every_scale_succeeds_and_gives_exactly_what_the_pipeline_gives_directly(world):
    run_id, description = run_to_end(world)
    assert description["state"] == "succeeded" and description["progress"] == {"succeeded": 13}
    status = description["status"]
    assert (status["execution"]["state"], status["validation"]["state"], status["security"]["state"], status["reproducibility"]["state"]) == ("PASS",) * 4
    assert status["uncertainty"]["state"] == "SKIPPED" and status["physics"]["state"] in ("PASS", "WARN")

    spec = spec_of("proton_transfer_pathway")
    exp = parse_experiment(spec)
    direct = Pipeline(default_registry()).run(exp.system, exp.propagation).final
    stored = world["orch"].final_observables(world["tenant"], run_id)
    assert compare_summaries(observables_summary(direct.observables), stored, rtol=REPRODUCTION_RTOL).matches
    assert len(description["result_digest"]) == 64


def test_a_run_whose_steps_each_run_in_their_own_process_gives_the_same_answer(world):
    from substrate.control.sandbox import ProcessExecutor
    world["orch"].executor = ProcessExecutor(memory_bytes=None)
    run_id, description = run_to_end(world, "proton_transfer")
    assert description["state"] == "succeeded"
    exp = parse_experiment(spec_of("proton_transfer"))
    direct = Pipeline(default_registry()).run(exp.system, exp.propagation).final
    assert compare_summaries(observables_summary(direct.observables), world["orch"].final_observables(world["tenant"], run_id), rtol=REPRODUCTION_RTOL).matches


def test_the_task_graph_is_validate_ingest_one_task_per_step_check_report(world):
    exp_id = experiment(world, "proton_transfer")
    run = world["orch"].submit(world["researcher"], exp_id, n_samples=0)
    rows = world["registry"].run_tasks(world["researcher"], run["id"])
    assert [(t["name"], t["kind"]) for t in rows] == [("validate", "validate"), ("ingest", "ingest"), ("00-solve:quantum.double_well_1d", "solve"),
                                                      ("01-translate:quantum_to_reaction.sequential_tunneling", "translate"), ("02-solve:reaction.mass_action_ode", "solve"),
                                                      ("check", "check"), ("report", "report")]
    assert [t["depends_on"] for t in rows] == [[], [f"{run['id']}/00"], [f"{run['id']}/01"], [f"{run['id']}/02"], [f"{run['id']}/03"], [f"{run['id']}/04"],
                                               [f"{run['id']}/05"]]
    assert {t["queue"] for t in rows} == {"cpu"} and rows[2]["payload"] == {"index": 0, "label": "solve:quantum.double_well_1d"}


def test_an_ensemble_adds_an_uncertainty_task_and_matches_the_direct_run(world):
    exp_id = experiment(world, "proton_transfer")
    run = world["orch"].submit(world["researcher"], exp_id, n_samples=6, seed=3)
    assert "uncertainty" in [t["name"] for t in world["registry"].run_tasks(world["researcher"], run["id"])]
    work(world)
    description = world["orch"].describe(world["researcher"], run["id"])
    assert description["state"] == "succeeded" and description["status"]["uncertainty"] == {"state": "PASS", "reasons": ["6 draws"]}
    spec = spec_of("proton_transfer")
    exp = parse_experiment(spec)
    direct = Pipeline(default_registry()).run(exp.system, exp.propagation, n_samples=6, seed=3)
    rows = world["registry"].run_tasks(world["researcher"], run["id"])
    uncertainty = world["artifacts"].read_json(world["tenant"], [t for t in rows if t["name"] == "uncertainty"][0]["output"])
    assert uncertainty["ensemble"]["n_ok"] == direct.ensemble["n_ok"] == 6 and uncertainty["ensemble"]["varied"] == direct.ensemble["varied"]
    assert compare_summaries(observables_summary(direct.final.observables), world["orch"].final_observables(world["tenant"], run["id"]), rtol=REPRODUCTION_RTOL).matches


def test_each_step_leaves_a_link_in_the_run_lineage_and_consecutive_steps_agree(world):
    run_id, _ = run_to_end(world, "proton_transfer")
    events = world["registry"].events(f"lineage:{run_id}", world["tenant"])
    assert [e["payload"]["index"] for e in events] == [0, 1, 2] and all(e["kind"] == "step" for e in events)
    for before, after in zip(events, events[1:]):
        assert before["payload"]["output_fingerprint"] == after["payload"]["input_fingerprint"]
    assert world["registry"].verify_chain(f"lineage:{run_id}", world["tenant"]) == (True, None)
    assert all(len(e["payload"]["artifact"]) == 64 and e["payload"]["worker"].startswith("worker-") for e in events)


def test_the_run_leaves_a_report_its_observables_and_a_status_document(world):
    run_id, _ = run_to_end(world)
    names = [a["name"] for a in world["registry"].run_artifacts(world["admin"], run_id)]
    assert {"final_observables.json", "report.json", "report.md", "status.json"} <= set(names) and "task-validate.json" in names
    report_sha = [a for a in world["registry"].run_artifacts(world["admin"], run_id) if a["name"] == "report.json"][0]["sha256"]
    report = world["artifacts"].read_json(world["tenant"], report_sha)
    assert report["run"] == run_id and len(report["steps"]) == 9 and report["assumptions"] and report["environment_hash"]
    hops = report["discarded_information"]
    assert len(hops) == 4 and all(h["information"] and h["scale_in"] != h["scale_out"] for h in hops)                    # what each of the four translations dropped
    assert [s["validity"] for s in report["steps"] if s["step"] == "translate"] and all(s["discarded"] for s in report["steps"] if s["step"] == "translate")
    assert all(s["discarded"] == [] for s in report["steps"] if s["step"] == "solve")
    markdown = world["artifacts"].read(world["viewer"], [a for a in world["registry"].run_artifacts(world["admin"], run_id) if a["name"] == "report.md"][0]["sha256"])[0]
    assert b"## Route" in markdown and b"## Assumptions made along the way" in markdown and b"## Information discarded between scales" in markdown
    status_sha = [a for a in world["registry"].run_artifacts(world["admin"], run_id) if a["name"] == "status.json"][0]["sha256"]
    document = world["artifacts"].read_json(world["tenant"], status_sha)
    assert document["state"] == "succeeded" and document["overall"] in ("PASS", "WARN")


def test_closing_a_run_is_first_wins_and_audited_once(world):
    run_id, _ = run_to_end(world, "proton_transfer")
    assert world["orch"].finalize_if_done(world["tenant"], run_id) is None
    finishes = [e for e in world["registry"].events(f"audit:{world['tenant']}", world["tenant"]) if e["kind"] == "run.finish"]
    assert len(finishes) == 1 and finishes[0]["payload"]["run"] == run_id


# -- reproducing -----------------------------------------------------------------------------------------------------------------------
def test_a_reproduction_runs_the_same_experiment_and_matches(world):
    run_id, original = run_to_end(world, "proton_transfer", n_samples=4)
    again = world["orch"].reproduce(world["researcher"], run_id)
    assert again["reproduces"] == run_id and (again["seed"], again["n_samples"]) == (original["seed"], 4) and again["experiment_id"] == original["experiment_id"]
    work(world)
    description = world["orch"].describe(world["researcher"], again["id"])
    assert description["state"] == "succeeded" and len(description["result_digest"]) == 64
    reproducibility = description["status"]["reproducibility"]
    assert reproducibility["state"] == "PASS" and f"matches {run_id}" in reproducibility["reasons"][0]
    import re
    assert float(re.search(r"largest relative difference ([0-9.e+-]+)", reproducibility["reasons"][0]).group(1)) <= REPRODUCTION_RTOL
    # The digests are deliberately not compared. Two honest runs of the same experiment on the same machine can differ in the last bits of the ODE solver's output
    # (multithreaded BLAS rounds differently from run to run, and the adaptive integrator amplifies that to about 1e-9: docs/findings.md), which changes a ten-digit
    # digest; this test failed about one run in thirty for that reason until the comparison, which is the arbiter, was made the assertion.


class PerturbingExecutor(InlineExecutor):
    """Makes the last step's result differ by a relative `amount`, as a changed or non-deterministic solver would."""

    def __init__(self, amount=1e-3):
        super().__init__()
        self.amount = amount

    def apply(self, pipeline, request):
        data = super().apply(pipeline, request)
        if request.index == 2:
            for q in data["observables"].values():
                if isinstance(q["value"], float):
                    q["value"] *= 1.0 + self.amount
        return data


def test_solver_level_noise_is_not_a_failed_reproduction_but_a_real_difference_is(world):
    run_id, _ = run_to_end(world, "proton_transfer")
    world["orch"].executor = PerturbingExecutor(1e-9)                                           # the size of the ODE solver's own tolerance
    noisy = world["orch"].reproduce(world["researcher"], run_id)
    work(world)
    assert world["orch"].describe(world["researcher"], noisy["id"])["status"]["reproducibility"]["state"] == "PASS"
    world["orch"].executor = PerturbingExecutor(1e-4)
    different = world["orch"].reproduce(world["researcher"], run_id)
    work(world)
    assert world["orch"].describe(world["researcher"], different["id"])["status"]["reproducibility"]["state"] == "FAIL"


def test_a_reproduction_that_does_not_match_fails_the_reproducibility_axis_and_names_what_differs(world):
    run_id, _ = run_to_end(world, "proton_transfer")
    world["orch"].executor = PerturbingExecutor()
    again = world["orch"].reproduce(world["researcher"], run_id)
    work(world)
    description = world["orch"].describe(world["researcher"], again["id"])
    reproducibility = description["status"]["reproducibility"]
    assert description["state"] == "succeeded" and reproducibility["state"] == "FAIL" and "results differ from" in reproducibility["reasons"][0]
    assert description["result_digest"] != world["orch"].describe(world["researcher"], run_id)["result_digest"]


def test_only_a_successful_run_of_your_own_tenant_can_be_reproduced(world):
    ok, _ = run_to_end(world, "proton_transfer")
    broken_spec = spec_of("proton_transfer_molecular")
    system_params(broken_spec)["oo_equilibrium"]["value"] = 2.3
    bad, description = run_to_end(world, spec=broken_spec, name="refuses")
    assert description["state"] == "failed"
    with pytest.raises(ValidationError, match="only a run that succeeded"):
        world["orch"].reproduce(world["researcher"], bad)
    other, boris, _ = world["registry"].bootstrap("lab-b", "boris")
    with pytest.raises(LookupError):
        world["orch"].reproduce(boris, ok)
    with pytest.raises(AuthorizationError):
        world["orch"].reproduce(world["viewer"], ok)


# -- the ways a run can go wrong, and where each one is recorded -------------------------------------------------------------------------
def test_a_model_that_refuses_is_a_physics_failure_that_is_not_retried_and_stops_everything_after_it(world):
    spec = spec_of("proton_transfer_molecular")
    system_params(spec)["oo_equilibrium"]["value"] = 2.3
    run_id, description = run_to_end(world, spec=spec, name="refuses")
    assert description["state"] == "failed"
    failed = [t for t in description["tasks"] if t["state"] == "failed"]
    assert len(failed) == 1 and failed[0]["attempts"] == 1 and "fewer than two wells" in failed[0]["error"]["message"] and failed[0]["error"]["axis"] == "physics"
    assert {t["state"] for t in description["tasks"] if t["name"] in ("check", "report")} == {"skipped"}
    status = states(description)
    assert status["physics"] == "FAIL" and status["execution"] == "PASS" and status["validation"] == "PASS" and status["security"] == "PASS"
    assert description["status"]["execution"]["reasons"][0].startswith("stopped early")


def test_a_crashing_step_is_retried_with_backoff_and_then_succeeds(world):
    calls = {"n": 0}

    def flaky_check(context):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient failure")
        return HANDLERS["check"](context)

    exp_id = experiment(world, "proton_transfer")
    run = world["orch"].submit(world["researcher"], exp_id, n_samples=0)
    worker = Worker(world["orch"], handlers={**HANDLERS, "check": flaky_check})
    for _ in range(40):
        if not worker.run_once():
            world["clock"].t += 5                                                              # wait out the backoff
        if world["orch"].describe(world["researcher"], run["id"])["state"] == "succeeded":
            break
    description = world["orch"].describe(world["researcher"], run["id"])
    check = [t for t in description["tasks"] if t["name"] == "check"][0]
    assert description["state"] == "succeeded" and check["attempts"] == 2 and states(description)["execution"] == "PASS"


def test_a_step_that_always_crashes_fails_after_its_attempts_and_is_an_execution_failure(world):
    def broken(context):
        raise RuntimeError("always broken")

    exp_id = experiment(world, "proton_transfer")
    run = world["orch"].submit(world["researcher"], exp_id, n_samples=0)
    worker = Worker(world["orch"], handlers={**HANDLERS, "check": broken})
    for _ in range(60):
        if not worker.run_once():
            world["clock"].t += 10
        if world["orch"].describe(world["researcher"], run["id"])["state"] == "failed":
            break
    description = world["orch"].describe(world["researcher"], run["id"])
    check = [t for t in description["tasks"] if t["name"] == "check"][0]
    assert description["state"] == "failed" and check["attempts"] == 3 and check["error"]["type"] == "RuntimeError"
    assert description["status"]["execution"]["state"] == "FAIL" and "always broken" in description["status"]["execution"]["reasons"][0]
    assert description["status"]["physics"]["state"] == "UNKNOWN" or description["status"]["physics"]["state"] in ("PASS", "WARN")


def test_a_request_over_a_limit_is_a_security_failure_with_an_audit_event_and_no_retry(world, monkeypatch):
    monkeypatch.setenv("SUBSTRATE_MAX_GRID_POINTS", "1000")
    spec = spec_of("proton_transfer_electronic")
    system_params(spec)["n_scan"] = {"value": 5000, "unit": "1"}
    run_id, description = run_to_end(world, spec=spec, name="too-big")
    failed = [t for t in description["tasks"] if t["state"] == "failed"][0]
    assert failed["attempts"] == 1 and failed["error"]["type"] == "ResourceLimitError" and failed["error"]["axis"] == "security"
    assert states(description)["security"] == "FAIL" and states(description)["execution"] == "PASS"
    blocked = [e for e in world["registry"].events(f"audit:{world['tenant']}", world["tenant"]) if e["kind"] == "policy.blocked"]
    assert len(blocked) == 1 and blocked[0]["payload"]["result"] == "BLOCKED" and blocked[0]["payload"]["run"] == run_id


def test_a_user_who_has_used_their_compute_allowance_has_their_tasks_refused(world):
    world["registry"].set_quota(world["admin"], "compute_seconds_per_day", 0)
    run_id, description = run_to_end(world, "proton_transfer")
    failed = [t for t in description["tasks"] if t["state"] == "failed"]
    assert failed[0]["name"] == "validate" and failed[0]["error"]["type"] == "QuotaExceededError" and states(description)["security"] == "FAIL"


def test_compute_time_is_charged_to_the_person_who_submitted(world):
    run_to_end(world, "proton_transfer")
    assert world["registry"].used_by(world["tenant"], world["researcher"].id, "compute_seconds_per_day") > 0
    assert world["registry"].used_by(world["tenant"], world["viewer"].id, "compute_seconds_per_day") == 0


def test_datasets_pinned_by_an_experiment_must_exist_unchanged(world):
    registry = world["registry"]
    stored = registry.put_dataset("nist.pa", source="NIST WebBook", license="public domain", schema={"x": "float"}, records=[{"x": 1.0}], provenance={"by": "test"})
    exp_id = experiment(world, "proton_transfer", datasets=[{"id": "nist.pa", "version": stored["version"]}])
    run = world["orch"].submit(world["researcher"], exp_id, n_samples=0)
    work(world)
    assert world["orch"].describe(world["researcher"], run["id"])["state"] == "succeeded"
    report = world["artifacts"].read_json(world["tenant"], [a for a in registry.run_artifacts(world["admin"], run["id"]) if a["name"] == "report.json"][0]["sha256"])
    assert report["datasets"][0]["id"] == "nist.pa" and report["datasets"][0]["checksum"] == stored["checksum"]

    missing = world["orch"].submit(world["researcher"], experiment(world, "proton_transfer", datasets=[{"id": "nist.pa", "version": 7}]), n_samples=0)
    work(world)
    description = world["orch"].describe(world["researcher"], missing["id"])
    assert description["state"] == "failed" and states(description)["reproducibility"] == "FAIL" and "version 7" in description["status"]["reproducibility"]["reasons"][0]
    assert [t for t in description["tasks"] if t["name"] == "ingest"][0]["attempts"] == 1

    import sqlalchemy as sa
    from substrate.control import db
    with registry.engine.begin() as conn:                                                      # someone edits the stored records behind the registry's back
        conn.execute(sa.update(db.datasets).where(db.datasets.c.dataset_id == "nist.pa").values(records=[{"x": 2.0}]))
    altered = world["orch"].submit(world["researcher"], exp_id, n_samples=0)
    work(world)
    description = world["orch"].describe(world["researcher"], altered["id"])
    assert states(description)["reproducibility"] == "FAIL" and "no longer matches" in description["status"]["reproducibility"]["reasons"][0]


def test_a_new_version_of_a_dataset_is_a_new_version_and_the_old_one_stays(world):
    registry = world["registry"]
    first = registry.put_dataset("d", source="s", license="l", schema={}, records=[1, 2], provenance={})
    same = registry.put_dataset("d", source="s", license="l", schema={}, records=[1, 2], provenance={})
    second = registry.put_dataset("d", source="s", license="l", schema={}, records=[1, 2, 3], provenance={})
    assert (first["version"], same["version"], second["version"]) == (1, 1, 2)
    assert registry.get_dataset("d", 1)["records"] == [1, 2] and registry.get_dataset("d")["version"] == 2 and registry.get_dataset("d", 9) is None
    assert [d["version"] for d in registry.list_datasets()] == [1, 2]


# -- refused before anything is created ---------------------------------------------------------------------------------------------------
def test_requests_that_cannot_run_are_refused_at_submission_and_leave_no_run(world):
    exp_id = experiment(world, "proton_transfer")
    with pytest.raises(ResourceLimitError, match="n_samples"):
        world["orch"].submit(world["researcher"], exp_id, n_samples=10**6)
    with pytest.raises(ValidationError, match="n_samples"):
        world["orch"].submit(world["researcher"], exp_id, n_samples=-1)
    with pytest.raises(ValidationError, match="seed"):
        world["orch"].submit(world["researcher"], exp_id, seed=2.5)
    nothing = copy.deepcopy(spec_of("proton_transfer"))
    for parameter in system_params(nothing).values():
        parameter.pop("sigma", None)
    with pytest.raises(ValidationError, match="no parameter has a sigma"):
        world["orch"].submit(world["researcher"], experiment(world, "no-sigma", nothing), n_samples=5)
    unroutable = copy.deepcopy(spec_of("proton_transfer"))
    (unroutable["experiment"] if "experiment" in unroutable else unroutable)["propagation"] = ["electronic_structure"]
    with pytest.raises((NoPathError, ValidationError)):
        world["orch"].submit(world["researcher"], experiment(world, "unroutable", unroutable), n_samples=0)
    with pytest.raises(ValidationError, match="datasets must be"):
        world["orch"].submit(world["researcher"], experiment(world, "bad-pins", spec_of("proton_transfer"), datasets=[{"id": "x"}]), n_samples=0)
    assert world["registry"].list_runs(world["researcher"]) == []


def test_roles_and_tenants_decide_who_can_submit_cancel_and_see(world):
    exp_id = experiment(world, "proton_transfer")
    with pytest.raises(AuthorizationError):
        world["orch"].submit(world["viewer"], exp_id)
    with pytest.raises(AuthorizationError):
        world["orch"].submit(world["operator"], exp_id)
    run = world["orch"].submit(world["researcher"], exp_id, n_samples=0)
    with pytest.raises(AuthorizationError):
        world["orch"].cancel(world["viewer"], run["id"])
    _, boris, _ = world["registry"].bootstrap("lab-b", "boris")
    with pytest.raises(LookupError):
        world["orch"].submit(boris, exp_id)
    assert world["orch"].describe(boris, run["id"]) is None
    with pytest.raises(LookupError):
        world["orch"].cancel(boris, run["id"])
    assert world["orch"].describe(world["viewer"], run["id"])["state"] == "queued"


def test_daily_allowances_apply_at_submission_and_a_refused_submission_costs_nothing(world):
    registry, who = world["registry"], world["researcher"]
    exp_id = experiment(world, "proton_transfer")
    registry.set_quota(world["admin"], "runs_per_day", 2)
    world["orch"].submit(who, exp_id, n_samples=0)
    world["orch"].submit(who, exp_id, n_samples=0)
    with pytest.raises(QuotaExceededError, match="runs_per_day"):
        world["orch"].submit(who, exp_id, n_samples=0)
    registry.set_quota(world["admin"], "runs_per_day", 100)
    registry.set_quota(world["admin"], "tasks_per_day", 14 + 6)
    with pytest.raises(QuotaExceededError, match="tasks_per_day"):
        world["orch"].submit(who, exp_id, n_samples=0)                                         # 7 tasks per run: the third run would need 21
    assert registry.used_today(who, "runs_per_day") == 2                                       # the blocked submissions took nothing


def test_a_submission_blocked_by_the_concurrency_limit_is_refunded(world):
    registry, who = world["registry"], world["researcher"]
    exp_id = experiment(world, "proton_transfer")
    registry.set_quota(world["admin"], "concurrent_runs", 1)
    world["orch"].submit(who, exp_id, n_samples=0)
    with pytest.raises(QuotaExceededError, match="already active"):
        world["orch"].submit(who, exp_id, n_samples=0)
    assert registry.used_today(who, "runs_per_day") == 1 and registry.used_today(who, "tasks_per_day") == 7


# -- cancelling -------------------------------------------------------------------------------------------------------------------------
def test_cancelling_stops_the_rest_and_the_run_ends_cancelled(world):
    exp_id = experiment(world, "proton_transfer")
    run = world["orch"].submit(world["researcher"], exp_id, n_samples=0)
    worker = Worker(world["orch"])
    worker.run_once()
    worker.run_once()
    assert world["orch"].cancel(world["operator"], run["id"]) == 5
    description = world["orch"].describe(world["researcher"], run["id"])
    assert description["state"] == "cancelled" and states(description)["execution"] == "FAIL" and worker.run_once() is False
    assert [t["state"] for t in description["tasks"]][:2] == ["succeeded", "succeeded"] and {t["state"] for t in description["tasks"][2:]} == {"cancelled"}
    cancels = [e for e in world["registry"].events(f"audit:{world['tenant']}", world["tenant"]) if e["kind"] == "run.cancel"]
    assert cancels[0]["payload"] == {"by": world["operator"].id, "run": run["id"], "tasks": 5}


# -- workers --------------------------------------------------------------------------------------------------------------------------
def test_a_worker_only_runs_the_queues_it_serves(world, monkeypatch):
    monkeypatch.setattr(tasks_module, "queue_for", lambda step, backend=None: "chemistry" if step.kind == "translate" else "cpu")
    exp_id = experiment(world, "proton_transfer")
    run = world["orch"].submit(world["researcher"], exp_id, n_samples=0)
    Worker(world["orch"], queues=("cpu",)).run_until_idle()
    description = world["orch"].describe(world["researcher"], run["id"])
    assert description["state"] == "running" and [t["state"] for t in description["tasks"]].count("ready") == 1
    assert [t for t in description["tasks"] if t["state"] == "ready"][0]["queue"] == "chemistry"
    Worker(world["orch"], queues=("chemistry",)).run_until_idle()
    Worker(world["orch"], queues=("cpu",)).run_until_idle()
    assert world["orch"].describe(world["researcher"], run["id"])["state"] == "succeeded"


def test_a_worker_that_died_mid_task_costs_a_retry_not_the_run(world):
    exp_id = experiment(world, "proton_transfer")
    run = world["orch"].submit(world["researcher"], exp_id, n_samples=0)
    dead = world["orch"].queue.claim("dead-worker", ["cpu"], lease_s=30)                       # takes `validate` and is never heard from again
    world["clock"].t += 31
    survivor = Worker(world["orch"])
    for _ in range(40):
        if not survivor.run_once():
            world["clock"].t += 5
        if world["orch"].describe(world["researcher"], run["id"])["state"] == "succeeded":
            break
    description = world["orch"].describe(world["researcher"], run["id"])
    assert description["state"] == "succeeded" and [t for t in description["tasks"] if t["id"] == dead.id][0]["attempts"] == 2


def test_several_workers_at_once_finish_several_runs_each_task_exactly_once(world):
    ids = [world["orch"].submit(world["researcher"], experiment(world, "proton_transfer", spec_of("proton_transfer")), n_samples=0)["id"] for _ in range(3)]
    errors = []

    def serve(n):
        try:
            Worker(world["orch"], worker_id=f"w{n}").run_until_idle()
        except Exception as error:                                                             # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=serve, args=(n,)) for n in range(3)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    for run_id in ids:
        description = world["orch"].describe(world["researcher"], run_id)
        problems = [(t["name"], t["state"], t["attempts"], t["error"]) for t in description["tasks"] if t["state"] != "succeeded" or t["attempts"] != 1]
        assert description["state"] == "succeeded" and not problems, problems
    assert len([e for e in world["registry"].events(f"audit:{world['tenant']}", world["tenant"]) if e["kind"] == "run.finish"]) == 3


def test_an_observer_sees_the_life_of_a_run(world):
    seen = {"submitted": 0, "started": 0, "finished": [], "run": []}

    class Recorder(Observer):
        def run_submitted(self, run):
            seen["submitted"] += 1

        def task_started(self, task):
            seen["started"] += 1

        def task_finished(self, task, seconds, outcome, axis):
            seen["finished"].append((outcome, axis))

        def run_finished(self, run, status):
            seen["run"].append((run["state"], status.overall()))

    world["orch"].observer = Recorder()
    run_to_end(world, "proton_transfer")
    assert seen["submitted"] == 1 and seen["started"] == 7 and seen["finished"] == [("succeeded", None)] * 7
    assert seen["run"] == [("succeeded", State.PASS)] or seen["run"][0][0] == "succeeded"
