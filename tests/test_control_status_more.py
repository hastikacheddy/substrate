"""Cases of the six-axis assessment that the first tests did not pin, found by mutation testing: each one below made a one-line change to `status.py` that the suite did not notice."""
import pytest

from substrate.control.status import AXES, AxisStatus, RunStatus, State, assess

from test_control_status import good_tasks, run, states, task


def test_a_failed_task_with_no_recorded_axis_is_an_execution_failure():
    """A row written by something other than the worker (or by an older version) may have no axis; the default is the axis of a crash, not of a model's refusal."""
    broken = [task("validate", "validate"), {"name": "00-solve", "kind": "solve", "state": "failed", "error": {"message": "lost"}},
              {"name": "01-translate", "kind": "translate", "state": "failed", "error": None}]
    status = assess(run(), broken)
    assert status.axes["execution"].state is State.FAIL and status.axes["execution"].reasons == ["00-solve: lost", "01-translate: failed"]
    assert status.axes["physics"].state is State.UNKNOWN


def test_a_run_whose_only_unfinished_task_is_running_has_not_finished():
    tasks = good_tasks()
    tasks[4]["state"] = "running"
    status = assess(run(), tasks)
    assert status.axes["execution"].state is State.UNKNOWN and status.axes["execution"].reasons == ["the run has not finished"]


def test_a_request_is_validated_only_if_the_input_checks_succeeded():
    for state in ("pending", "ready", "running", "skipped", "cancelled"):
        status = assess(run(), [task("validate", "validate", state)] + good_tasks()[1:])
        assert status.axes["validation"].state is State.UNKNOWN and status.axes["validation"].reasons == ["the request was not validated"], state
    without_the_check = assess(run(), good_tasks()[1:])                                    # other tasks succeeded, but nothing validated the request
    assert without_the_check.axes["validation"].state is State.UNKNOWN


def test_physics_cannot_pass_without_a_model_having_run():
    status = assess(run(), [task("validate", "validate"), task("ingest", "ingest"), task("report", "report")])
    assert status.axes["physics"].state is State.UNKNOWN and status.axes["physics"].reasons == ["the models did not all run"]
    partly = assess(run(), [task("validate", "validate"), task("00-solve", "solve"), task("01-translate", "translate", "skipped")])
    assert partly.axes["physics"].state is State.UNKNOWN


def test_either_kind_of_physics_finding_alone_is_a_warning():
    only_warnings = assess(run(), good_tasks(), {"warnings": ["[m] the tunnelling parameter is not applied"]})
    assert only_warnings.axes["physics"].state is State.WARN and only_warnings.axes["physics"].reasons == ["[m] the tunnelling parameter is not applied"]
    only_non_finite = assess(run(), good_tasks(), {"non_finite": ["rate"]})
    assert only_non_finite.axes["physics"].state is State.WARN and only_non_finite.axes["physics"].reasons == ["non-finite values in: rate"]
    neither = assess(run(), good_tasks(), {"warnings": [], "non_finite": []})
    assert neither.axes["physics"].state is State.PASS


def test_one_dropped_draw_is_reported_and_the_draws_shown_are_the_good_ones():
    one = assess(run(n_samples=100), good_tasks(), {"ensemble": {"n_requested": 100, "n_ok": 99, "n_failed": 1}})
    assert one.axes["uncertainty"].state is State.WARN and one.axes["uncertainty"].reasons == ["1 of 100 draws were invalid and dropped"]
    clean = assess(run(n_samples=100), good_tasks(), {"ensemble": {"n_requested": 120, "n_ok": 98, "n_failed": 0}})
    assert clean.axes["uncertainty"].reasons == ["98 draws"]


def test_a_transfer_study_is_judged_like_any_other_run():
    study = [task("validate", "validate_transfer"), task("reference:ref_a", "reference"), task("reference:ref_b", "reference"), task("transfer", "transfer"),
             task("report", "transfer_report")]
    status = assess(run(), study)
    assert states(status) == {"execution": State.PASS, "physics": State.PASS, "validation": State.PASS, "security": State.PASS,
                              "reproducibility": State.PASS, "uncertainty": State.SKIPPED}
    refused = assess(run(), [task("validate", "validate_transfer", "failed", "validation", "a study needs two surfaces")] + study[1:])
    assert refused.axes["validation"].state is State.FAIL
    failed_reference = assess(run(), [study[0], task("reference:ref_a", "reference", "failed", "physics", "no fit"), study[2], study[3]])
    assert failed_reference.axes["physics"].state is State.FAIL
    unfinished = assess(run(), study[:2] + [task("transfer", "transfer", "pending")])
    assert unfinished.axes["physics"].state is State.UNKNOWN


def test_a_stored_status_with_an_axis_this_version_does_not_know_keeps_only_the_axes_it_does():
    stored = RunStatus().to_dict() | {"vibes": {"state": "PASS", "reasons": []}}
    again = RunStatus.from_dict(stored)
    assert set(again.axes) == set(AXES) and "vibes" not in again.to_dict()
    assert isinstance(again.axes["physics"], AxisStatus)
