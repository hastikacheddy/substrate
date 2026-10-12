"""The six-axis run status: which axis a failure lands on, and what each axis says for each kind of outcome."""
import pytest

from substrate import QCError, ResourceLimitError, ValidationError
from substrate.errors import AmbiguousPathError, DatasetUnavailableError, NoPathError
from substrate.control.security import AuthorizationError, QuotaExceededError
from substrate.control.status import AXES, AxisStatus, RunStatus, State, assess, classify_failure, is_deterministic_refusal


def task(name, kind, state="succeeded", axis=None, message="boom"):
    return {"name": name, "kind": kind, "state": state, "error": {"axis": axis, "message": message} if axis else None}


def run(**changes):
    return {"environment_hash": "h", "seed": 1, "n_samples": 0, "reproduces": None, **changes}


def good_tasks():
    return [task("validate", "validate"), task("ingest", "ingest"), task("00-solve", "solve"), task("01-translate", "translate"), task("check", "check")]


def states(status):
    return {axis: status.axes[axis].state for axis in AXES}


# -- classification ------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind, error, axis", [
    ("validate", ValidationError("x"), "validation"),
    ("validate", NoPathError("x"), "validation"),
    ("validate", AmbiguousPathError("x"), "validation"),
    ("solve", ValidationError("barrier below zero-point energy"), "physics"),
    ("translate", ValidationError("x"), "physics"),
    ("check", ValidationError("x"), "physics"),
    ("solve", RuntimeError("segfault"), "execution"),
    ("solve", QCError("no program"), "execution"),
    ("validate", QCError("no program"), "execution"),
    ("solve", ResourceLimitError("too big"), "security"),
    ("validate", ResourceLimitError("too big"), "security"),
    ("solve", QuotaExceededError("used up"), "security"),
    ("solve", AuthorizationError("no"), "security"),
    ("ingest", DatasetUnavailableError("no such dataset"), "reproducibility"),
    ("ingest", LookupError("a bug, not a missing dataset"), "execution"),
    ("ingest", ValidationError("checksum"), "reproducibility"),
    ("ingest", RuntimeError("disk"), "execution"),
])
def test_where_a_failure_happened_and_what_it_was_decide_its_axis(kind, error, axis):
    assert classify_failure(kind, error) == axis


@pytest.mark.parametrize("error, refusal", [(ValidationError("x"), True), (ResourceLimitError("x"), True), (QuotaExceededError("x"), True),
                                            (AuthorizationError("x"), True), (DatasetUnavailableError("x"), True), (RuntimeError("x"), False), (QCError("x"), False), (TimeoutError("x"), False)])
def test_only_deterministic_refusals_are_not_worth_retrying(error, refusal):
    assert is_deterministic_refusal(error) is refusal


# -- the status object ---------------------------------------------------------------------------------------------------------------
def test_a_new_status_knows_nothing_yet():
    status = RunStatus()
    assert set(status.axes) == set(AXES) and all(a.state is State.UNKNOWN for a in status.axes.values()) and status.overall() is State.UNKNOWN


def test_the_overall_state_is_the_worst_axis_and_skipped_axes_do_not_count():
    status = RunStatus()
    for axis in AXES:
        status.set(axis, State.PASS)
    assert status.overall() is State.PASS
    status.set("uncertainty", State.SKIPPED)
    assert status.overall() is State.PASS
    status.set("physics", State.WARN, "a")
    assert status.overall() is State.WARN
    status.set("security", State.UNKNOWN)
    assert status.overall() is State.UNKNOWN                                   # not knowing is worse than a warning
    status.set("execution", State.FAIL, "crash")
    assert status.overall() is State.FAIL
    only_skipped = RunStatus({a: AxisStatus(State.SKIPPED) for a in AXES})
    assert only_skipped.overall() is State.PASS


def test_worsen_never_improves_and_collects_reasons():
    status = RunStatus()
    status.set("physics", State.WARN, "first")
    status.worsen("physics", State.PASS, "second")
    assert status.axes["physics"].state is State.WARN and status.axes["physics"].reasons == ["first", "second"]
    status.worsen("physics", State.FAIL, "third")
    assert status.axes["physics"].state is State.FAIL and status.axes["physics"].reasons == ["first", "second", "third"]


def test_an_unknown_axis_is_an_error():
    with pytest.raises(KeyError, match="unknown status axis"):
        RunStatus().set("vibes", State.PASS)


def test_the_status_round_trips_through_json_and_prints_one_line_per_axis():
    import json
    status = assess(run(), good_tasks())
    again = RunStatus.from_dict(json.loads(json.dumps(status.to_dict())))
    assert again.to_dict() == status.to_dict()
    lines = status.summary().splitlines()
    assert len(lines) == 6 and lines[0].startswith("Execution:") and "PASS" in lines[0] and "SKIPPED" in lines[5]


# -- assessing runs ---------------------------------------------------------------------------------------------------------------------
def test_a_clean_run_passes_everything_it_applies_to():
    status = assess(run(), good_tasks())
    assert states(status) == {"execution": State.PASS, "physics": State.PASS, "validation": State.PASS, "security": State.PASS,
                              "reproducibility": State.PASS, "uncertainty": State.SKIPPED}
    assert status.overall() is State.PASS


def test_a_model_that_refuses_is_a_physics_failure_and_not_an_execution_failure():
    tasks = [task("validate", "validate"), task("ingest", "ingest"), task("00-solve", "solve"),
             task("01-translate", "translate", "failed", "physics", "barrier is below the zero-point energy"),
             task("02-solve", "solve", "skipped"), task("check", "check", "skipped")]
    status = assess(run(), tasks)
    assert status.axes["physics"].state is State.FAIL and "barrier is below the zero-point energy" in status.axes["physics"].reasons[0]
    assert status.axes["execution"].state is State.PASS and "2 task(s) not run" in status.axes["execution"].reasons[0]
    assert status.axes["validation"].state is State.PASS and status.overall() is State.FAIL


def test_a_crashed_solver_is_an_execution_failure_and_leaves_physics_unjudged():
    tasks = [task("validate", "validate"), task("ingest", "ingest"), task("00-solve", "solve", "failed", "execution", "worker lost"), task("check", "check", "skipped")]
    status = assess(run(), tasks)
    assert status.axes["execution"].state is State.FAIL and status.axes["execution"].reasons == ["00-solve: worker lost"]
    assert status.axes["physics"].state is State.UNKNOWN and status.axes["physics"].reasons == ["the models did not all run"]


def test_a_bad_request_is_a_validation_failure():
    status = assess(run(), [task("validate", "validate", "failed", "validation", "n_x must be a whole number"), task("ingest", "ingest", "skipped"),
                            task("00-solve", "solve", "skipped")])
    assert status.axes["validation"].state is State.FAIL and status.axes["validation"].reasons == ["validate: n_x must be a whole number"]
    assert status.axes["physics"].state is State.UNKNOWN and status.axes["execution"].state is State.PASS


def test_a_policy_breach_is_a_security_failure():
    status = assess(run(), [task("validate", "validate"), task("00-solve", "solve", "failed", "security", "n_x: 5,000 exceeds the limit of 1,000")])
    assert status.axes["security"].state is State.FAIL and "exceeds the limit" in status.axes["security"].reasons[0]
    assert status.axes["execution"].state is State.PASS


def test_a_cancelled_run_is_an_execution_failure():
    tasks = good_tasks()
    tasks[3]["state"], tasks[4]["state"] = "cancelled", "cancelled"
    status = assess(run(), tasks)
    assert status.axes["execution"].state is State.FAIL and status.axes["execution"].reasons == ["the run was cancelled"]


def test_a_run_that_has_not_finished_is_unknown_not_passing():
    tasks = good_tasks()
    tasks[3]["state"], tasks[4]["state"] = "running", "pending"
    status = assess(run(), tasks)
    assert status.axes["execution"].state is State.UNKNOWN and status.overall() in (State.UNKNOWN, State.FAIL)
    assert status.axes["physics"].state is State.UNKNOWN


def test_warnings_from_the_models_and_non_finite_observables_make_physics_a_warning():
    status = assess(run(), good_tasks(), {"warnings": [f"[m] warning {i}" for i in range(8)], "non_finite": ["rate", "barrier"]})
    physics = status.axes["physics"]
    assert physics.state is State.WARN and physics.reasons[:5] == [f"[m] warning {i}" for i in range(5)] and physics.reasons[5] == "non-finite values in: rate, barrier"
    assert status.overall() is State.WARN


def test_uncertainty_is_skipped_without_an_ensemble_and_judged_with_one():
    assert assess(run(n_samples=0), good_tasks()).axes["uncertainty"].state is State.SKIPPED
    ok = assess(run(n_samples=100), good_tasks(), {"ensemble": {"n_requested": 100, "n_ok": 100, "n_failed": 0}})
    assert ok.axes["uncertainty"].state is State.PASS and ok.axes["uncertainty"].reasons == ["100 draws"]
    some = assess(run(n_samples=100), good_tasks(), {"ensemble": {"n_requested": 100, "n_ok": 91, "n_failed": 9}})
    assert some.axes["uncertainty"].state is State.WARN and "9 of 100 draws" in some.axes["uncertainty"].reasons[0]
    assert assess(run(n_samples=100), good_tasks()).axes["uncertainty"].state is State.UNKNOWN
    broken = good_tasks() + [task("uncertainty", "uncertainty", "failed", "execution", "no root parameter has a sigma")]
    assert assess(run(n_samples=100), broken).axes["uncertainty"].state is State.FAIL


def test_reproducibility_needs_a_recorded_environment_and_seed():
    assert assess(run(environment_hash=""), good_tasks()).axes["reproducibility"].state is State.FAIL
    assert assess(run(seed=None), good_tasks()).axes["reproducibility"].state is State.FAIL
    assert assess(run(seed=0), good_tasks()).axes["reproducibility"].state is State.PASS            # a zero seed is a seed


def test_a_missing_or_altered_dataset_is_a_reproducibility_failure():
    tasks = [task("validate", "validate"), task("ingest", "ingest", "failed", "reproducibility", "dataset pubchem.compound version 3 not found"),
             task("00-solve", "solve", "skipped")]
    status = assess(run(), tasks)
    assert status.axes["reproducibility"].state is State.FAIL and "version 3" in status.axes["reproducibility"].reasons[0]


def test_a_reproduction_is_judged_by_its_numbers_and_its_environment():
    reproduction = run(reproduces="RUN-000001")
    match = {"matches": True, "max_relative_difference": 2e-12, "differing": []}
    same = assess(reproduction, good_tasks(), {"comparison": match, "original_environment_hash": "h"})
    assert same.axes["reproducibility"].state is State.PASS and "matches RUN-000001" in same.axes["reproducibility"].reasons[0]
    other_env = assess(reproduction, good_tasks(), {"comparison": match, "original_environment_hash": "different"})
    assert other_env.axes["reproducibility"].state is State.WARN and "environment differs" in other_env.axes["reproducibility"].reasons[0]
    differs = assess(reproduction, good_tasks(), {"comparison": {"matches": False, "max_relative_difference": 0.02, "differing": ["barrier", "rate"]},
                                                  "original_environment_hash": "h"})
    assert differs.axes["reproducibility"].state is State.FAIL and "barrier, rate" in differs.axes["reproducibility"].reasons[0]
    nothing = assess(reproduction, good_tasks(), {"original_environment_hash": "h"})
    assert nothing.axes["reproducibility"].state is State.UNKNOWN
