"""Four more cases of the orchestrator found by mutation testing: model warnings reaching the physics axis, a refused submission being counted, a reproduction keeping the original's seed,
and a reproduction being judged at the platform's tolerance rather than at the comparison's tighter default."""
import pytest

pytest.importorskip("sqlalchemy")

from substrate.control.orchestrator import REPRODUCTION_RTOL, Observer  # noqa: E402
from substrate.control.security import QuotaExceededError  # noqa: E402

from test_control_orchestrator import PerturbingExecutor, run_to_end, work, world, experiment  # noqa: E402,F401


def test_the_warnings_a_model_gave_reach_the_physics_axis_of_the_run(world):
    _, description = run_to_end(world, "proton_transfer_molecular")
    physics = description["status"]["physics"]
    assert description["state"] == "succeeded" and physics["state"] == "WARN"
    assert any("[molecular_to_reaction.tst]" in reason and "tunnelling is significant" in reason for reason in physics["reasons"])
    _, clean = run_to_end(world, "proton_transfer_electronic")
    assert clean["status"]["physics"] == {"state": "PASS", "reasons": []}


class Counting(Observer):
    def __init__(self):
        self.blocked_kinds = []

    def blocked(self, kind):
        self.blocked_kinds.append(kind)


def test_a_submission_that_is_refused_is_counted_and_charges_nothing(world):
    counting = Counting()
    world["orch"].observer = counting
    exp = experiment(world, "proton_transfer_electronic")
    world["registry"].set_quota(world["admin"], "runs_per_day", 1)
    world["orch"].submit(world["researcher"], exp, n_samples=0)
    with pytest.raises(QuotaExceededError):
        world["orch"].submit(world["researcher"], exp, n_samples=0)
    assert counting.blocked_kinds == ["submit"]
    assert world["registry"].used_today(world["researcher"], "runs_per_day") == 1 and world["registry"].used_today(world["researcher"], "tasks_per_day") > 0


def test_a_reproduction_uses_the_seed_the_run_had_even_when_that_was_not_the_experiments(world):
    run_id, original = run_to_end(world, "proton_transfer", n_samples=3, seed=123)             # the experiment file says 7; this run was started with another
    assert original["seed"] == 123
    again = world["orch"].reproduce(world["researcher"], run_id)
    assert again["seed"] == 123 and again["n_samples"] == 3 and again["reproduces"] == run_id


def test_a_difference_between_the_default_tolerance_and_the_platforms_is_not_a_failed_reproduction(world):
    run_id, _ = run_to_end(world, "proton_transfer")
    world["orch"].executor = PerturbingExecutor(1e-7)                                          # (the perturbation is applied to the third step, which this experiment has)                                          # above compare_summaries' default of 1e-8, below the platform's 1e-6
    assert 1e-8 < 1e-7 < REPRODUCTION_RTOL
    noisy = world["orch"].reproduce(world["researcher"], run_id)
    work(world)
    status = world["orch"].describe(world["researcher"], noisy["id"])["status"]["reproducibility"]
    assert status["state"] == "PASS" and "matches" in status["reasons"][0]


# -- environment drift, mixed endings, a lost race to close a run, and the status document of a reproduction ------------------------------------------------
import json  # noqa: E402

import sqlalchemy as sa  # noqa: E402

from substrate.control import db  # noqa: E402


def test_results_that_match_from_a_different_software_environment_are_a_warning_not_a_pass(world):
    run_id, _ = run_to_end(world, "proton_transfer_electronic")
    with world["registry"].engine.begin() as conn:
        conn.execute(sa.update(db.runs).where(db.runs.c.id == run_id).values(environment_hash="a-different-environment"))
    again = world["orch"].reproduce(world["researcher"], run_id)
    work(world)
    status = world["orch"].describe(world["researcher"], again["id"])["status"]["reproducibility"]
    assert status["state"] == "WARN" and "software environment differs" in status["reasons"][0]


def test_a_run_with_both_a_failed_and_a_cancelled_task_ends_failed(world):
    run = world["orch"].submit(world["researcher"], experiment(world, "proton_transfer_electronic"), n_samples=0)["id"]
    names = [t["name"] for t in world["registry"].run_tasks(world["researcher"], run)]
    with world["registry"].engine.begin() as conn:
        for index, state in enumerate(["succeeded", "failed"] + ["cancelled"] * (len(names) - 2)):
            conn.execute(sa.update(db.tasks).where(db.tasks.c.run_id == run, db.tasks.c.name == names[index]).values(state=state, error={"axis": "execution", "message": "boom"} if state == "failed" else None))
    world["orch"].finalize_if_done(world["tenant"], run)
    assert world["registry"].get_run(world["researcher"], run)["state"] == "failed"


def test_a_worker_that_loses_the_race_to_close_a_run_writes_and_announces_nothing(world, monkeypatch):
    run = world["orch"].submit(world["researcher"], experiment(world, "proton_transfer_electronic"), n_samples=0)["id"]
    finished = []

    class Watching(Observer):
        def run_finished(self, run, status):
            finished.append(run["id"])

    world["orch"].observer = Watching()
    monkeypatch.setattr(world["orch"].queue, "run_is_finished", lambda tenant, run_id: True)
    monkeypatch.setattr(world["registry"], "finish_run", lambda *a, **k: False)                  # another worker closed it first
    before = len(world["registry"].events(f"audit:{world['tenant']}", world["tenant"]))
    assert world["orch"].finalize_if_done(world["tenant"], run) is None
    assert finished == [] and len(world["registry"].events(f"audit:{world['tenant']}", world["tenant"])) == before
    assert [a["name"] for a in world["registry"].run_artifacts(world["admin"], run)] == []        # no status document either


def test_the_status_document_of_a_reproduction_says_what_it_was_compared_with(world):
    run_id, _ = run_to_end(world, "proton_transfer_electronic")
    again = world["orch"].reproduce(world["researcher"], run_id)
    work(world)
    artifact = [a for a in world["registry"].run_artifacts(world["admin"], again["id"]) if a["name"] == "status.json"][0]
    document = json.loads(world["artifacts"].read_for(world["tenant"], artifact["sha256"]))
    assert document["run"] == again["id"] and document["state"] == "succeeded"
    assert document["reproduction"]["of"] == run_id and document["reproduction"]["matches"] is True and document["reproduction"]["max_relative_difference"] < 1e-6
    original = [a for a in world["registry"].run_artifacts(world["admin"], run_id) if a["name"] == "status.json"][0]
    assert "reproduction" not in json.loads(world["artifacts"].read_for(world["tenant"], original["sha256"]))
