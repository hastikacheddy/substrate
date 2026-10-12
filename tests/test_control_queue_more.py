"""Cases of the task queue that the first tests did not pin, found by mutation testing. The queue keeps several invariants (a task has a lease owner exactly while it is running; a task is
ready only when everything it depends on has succeeded; a task that fails for good skips its dependents in the same transaction), and several of its conditions are redundant given those:
the last test checks the invariants themselves, so that the redundancy is a defence in depth and not a hole."""
import pytest

pytest.importorskip("sqlalchemy")

from substrate.control import db  # noqa: E402
from substrate.control.taskqueue import LeaseLost  # noqa: E402

import sqlalchemy as sa  # noqa: E402

from test_control_queue import clock, diamond, drain_one, make_run, states, world  # noqa: E402,F401


def test_a_claim_holds_the_task_for_exactly_the_lease_it_asked_for(world, clock):
    from substrate.control.registry import TaskSpec
    make_run(world, [TaskSpec("only", "validate")])
    task = world["queue"].claim("w1", ["cpu"], lease_s=100.0)
    row = world["queue"].task(world["tenant"], task.id)
    assert row["lease_expires"] == clock.t + 100.0 and row["lease_owner"] == "w1" and row["attempts"] == 1 and row["started_at"] == clock.t
    clock.t += 99
    assert world["queue"].reap_expired() == 0                                           # still held
    clock.t += 2
    assert world["queue"].reap_expired() == 1                                           # now lost: handed back for another attempt
    assert states(world, row["run_id"]) == {"only": "ready"}


def test_the_invariants_that_make_some_conditions_redundant_hold_after_every_kind_of_ending(world, clock):
    """A task has a lease owner exactly while it is running; a failed or skipped task has dependents that are not waiting on it."""
    run = make_run(world, diamond(max_attempts=1))
    queue, registry = world["queue"], world["registry"]

    def check():
        for row in registry.run_tasks(world["who"], run):
            assert (row["lease_owner"] is not None) == (row["state"] == "running"), row["name"]
            assert (row["lease_expires"] is not None) == (row["state"] == "running"), row["name"]
            if row["state"] in ("failed", "skipped", "cancelled", "succeeded"):
                assert row["finished_at"] is not None
    a = queue.claim("w1", ["cpu"])
    check()
    queue.complete(a.id, "w1", "out", 0.1)
    check()
    b, c = queue.claim("w1", ["cpu"]), queue.claim("w2", ["cpu"])
    check()
    assert queue.fail(b.id, "w1", {"type": "X"}, retry=False) == "failed"
    check()
    now = states(world, run)
    assert now["d"] == "skipped" and now["c"] == "running"                               # the dependent of the failure is skipped at once; the other branch is untouched
    with pytest.raises(LeaseLost):
        queue.complete(b.id, "w1", None, 0.1)                                           # the failed task cannot be completed afterwards
    with pytest.raises(LeaseLost):
        queue.fail(b.id, "w1", {"type": "X"}, retry=False)
    queue.complete(c.id, "w2", "out", 0.1)
    check()
    assert queue.run_is_finished(world["tenant"], run)


def test_a_task_that_is_not_running_cannot_be_completed_or_failed_even_by_its_old_owner(world):
    from substrate.control.registry import TaskSpec
    run = make_run(world, [TaskSpec("t", "validate"), TaskSpec("u", "solve", depends_on=["t"])])
    task = world["queue"].claim("w1", ["cpu"])
    world["queue"].cancel_run(world["tenant"], run)
    for action in (lambda: world["queue"].complete(task.id, "w1", None, 0.1), lambda: world["queue"].fail(task.id, "w1", {"type": "X"}, retry=True)):
        with pytest.raises(LeaseLost):
            action()
    assert states(world, run) == {"t": "cancelled", "u": "cancelled"}
    # and a hand-made inconsistent row (a ready task that still names an owner) is not treated as held
    with world["registry"].engine.begin() as conn:
        conn.execute(sa.update(db.tasks).where(db.tasks.c.id == task.id).values(state="ready", lease_owner="w1", lease_expires=None))
    with pytest.raises(LeaseLost):
        world["queue"].complete(task.id, "w1", None, 0.1)
    with pytest.raises(LeaseLost):
        world["queue"].fail(task.id, "w1", {"type": "X"}, retry=False)


def test_failing_for_good_skips_pending_and_ready_dependents_but_not_the_finished_or_the_running(world):
    from substrate.control.registry import TaskSpec
    run = make_run(world, [TaskSpec("a", "validate"), TaskSpec("done", "solve", depends_on=["a"]), TaskSpec("late", "solve", depends_on=["a", "done"]),
                           TaskSpec("x", "solve")])
    queue = world["queue"]
    a = queue.claim("w1", ["cpu"])
    x = queue.claim("w2", ["cpu"])
    assert {a.name, x.name} == {"a", "x"}
    queue.complete(a.id, "w1", None, 0.1)                                               # wakes "done"
    done = queue.claim("w1", ["cpu"])
    assert done.name == "done"
    # force the awkward states by hand: "late" is ready (as if woken) and "done" has finished; then x fails for good with "late" depending on x as well
    with world["registry"].engine.begin() as conn:
        conn.execute(sa.update(db.tasks).where(db.tasks.c.run_id == run, db.tasks.c.name == "late").values(depends_on=[a.id, done.id, x.id], state="ready"))
        conn.execute(sa.update(db.tasks).where(db.tasks.c.id == done.id).values(depends_on=[a.id, x.id]))            # and "done", already running, also names x
    assert queue.fail(x.id, "w2", {"type": "X"}, retry=False) == "failed"
    now = states(world, run)
    assert now["late"] == "skipped" and now["done"] == "running"                         # a ready dependent is skipped; a running one is left to finish
    queue.complete(done.id, "w1", None, 0.1)
    assert states(world, run)["done"] == "succeeded"                                     # and a finished task is not turned into a skipped one by a later failure


def test_a_task_waits_for_every_dependency_to_have_succeeded_not_merely_ended(world):
    from substrate.control.registry import TaskSpec
    run = make_run(world, [TaskSpec("good", "validate"), TaskSpec("bad", "validate"), TaskSpec("both", "solve", depends_on=["good", "bad"]), TaskSpec("other", "validate")])
    queue = world["queue"]
    with world["registry"].engine.begin() as conn:                                      # "bad" has ended in failure, by hand, so that nothing has skipped its dependent
        conn.execute(sa.update(db.tasks).where(db.tasks.c.run_id == run, db.tasks.c.name == "bad").values(state="failed"))
    for _ in range(2):
        claimed = queue.claim("w1", ["cpu"])
        queue.complete(claimed.id, "w1", None, 0.1)                                     # each completion looks for tasks that can now start
    assert states(world, run) == {"good": "succeeded", "bad": "failed", "both": "pending", "other": "succeeded"}


def test_a_run_is_finished_only_when_it_has_tasks_and_none_is_waiting_ready_or_running(world):
    from substrate.control.registry import TaskSpec
    run = make_run(world, [TaskSpec("t", "validate")])
    queue = world["queue"]
    assert queue.run_is_finished(world["tenant"], run) is False                         # ready
    task = queue.claim("w1", ["cpu"])
    assert queue.run_is_finished(world["tenant"], run) is False                         # running: nothing is waiting, but it has not ended
    queue.complete(task.id, "w1", None, 0.1)
    assert queue.run_is_finished(world["tenant"], run) is True
    assert queue.run_is_finished(world["tenant"], "RUN-999999") is False                # a run with no tasks (or none of this tenant's) is not finished
    assert queue.run_is_finished("T-9999", run) is False
