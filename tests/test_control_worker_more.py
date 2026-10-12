"""Cases of the worker that the first tests did not pin, found by mutation testing: how long a running task is held, what happens to a task nobody knows how to run, and what a refusal by a
limit or a quota leaves behind (an audit event and a count), but only when the worker still held the task."""
import pytest

pytest.importorskip("sqlalchemy")

from substrate import ResourceLimitError  # noqa: E402
from substrate.control.orchestrator import Observer  # noqa: E402
from substrate.control.worker import LEASE_MARGIN_S, Worker  # noqa: E402

from test_control_orchestrator import experiment, states, work, world  # noqa: E402,F401


class Recorder(Observer):
    def __init__(self):
        self.blocked_kinds, self.finished = [], []

    def blocked(self, kind):
        self.blocked_kinds.append(kind)

    def task_finished(self, task, seconds, outcome, axis):
        self.finished.append((task.kind, outcome, axis))


def submit(world, name="proton_transfer_electronic"):
    return world["orch"].submit(world["researcher"], experiment(world, name), n_samples=0)["id"]


def test_a_running_task_is_held_for_its_own_timeout_plus_a_margin(world):
    run = submit(world)
    seen = []

    def watch(context):
        row = world["orch"].queue.task(context.task.tenant_id, context.task.id)
        seen.append((row["lease_expires"] - world["clock"].t, context.task.timeout_s))
        return {"ok": True}, None

    Worker(world["orch"], handlers={"validate": watch}).run_once()
    ((lease, timeout),) = seen
    assert timeout > 0 and lease == pytest.approx(timeout + LEASE_MARGIN_S) and LEASE_MARGIN_S == 60.0


def test_a_task_kind_nobody_can_run_fails_with_a_clear_error_and_is_not_swallowed(world):
    run = submit(world)
    Worker(world["orch"], handlers={}).run_once()
    task = [t for t in world["registry"].run_tasks(world["researcher"], run) if t["name"] == "validate"][0]
    assert task["state"] == "ready" and task["attempts"] == 1                                    # a crash is worth another try, so it waits for its backoff
    assert task["error"]["type"] == "LookupError" and task["error"]["message"] == "no handler for task kind 'validate'" and task["error"]["axis"] == "execution"


def refuse_by_a_limit(context):
    raise ResourceLimitError("n_x: 5,000 exceeds the limit of 1,000")


def test_a_refusal_by_a_limit_is_audited_and_counted_once_and_is_not_retried(world):
    run = submit(world)
    recorder = Recorder()
    Worker(world["orch"], handlers={"validate": refuse_by_a_limit}, observer=recorder).run_once()
    assert recorder.blocked_kinds == ["ResourceLimitError"] and recorder.finished == [("validate", "failed", "security")]
    blocked = [e for e in world["registry"].events(f"audit:{world['tenant']}", world["tenant"]) if e["kind"] == "policy.blocked"]
    assert len(blocked) == 1 and blocked[0]["payload"]["error"] == "ResourceLimitError" and blocked[0]["payload"]["result"] == "BLOCKED" and blocked[0]["payload"]["run"] == run
    assert states_of(world, run)["validate"] == "failed"


def states_of(world, run):
    return {t["name"]: t["state"] for t in world["registry"].run_tasks(world["researcher"], run)}


def test_a_refusal_that_arrives_after_the_task_was_taken_away_leaves_no_record_of_a_block(world):
    run = submit(world)
    recorder = Recorder()

    def cancelled_then_refused(context):
        world["orch"].cancel(world["researcher"], run)                                           # the run is cancelled while the step is still running
        raise ResourceLimitError("too big")

    Worker(world["orch"], handlers={"validate": cancelled_then_refused}, observer=recorder).run_once()
    assert recorder.blocked_kinds == [] and recorder.finished == [("validate", "lost", "security")]
    assert not [e for e in world["registry"].events(f"audit:{world['tenant']}", world["tenant"]) if e["kind"] == "policy.blocked"]


# -- retries, long messages, the task limit and an idle worker ----------------------------------------------------------------------------------
def crash(context):
    raise RuntimeError("segfault in the solver " + "x" * 5000)


def test_a_crash_is_reported_as_a_retry_until_the_attempts_run_out_and_its_message_is_cut_to_2000_characters(world):
    run = submit(world)
    recorder = Recorder()
    worker = Worker(world["orch"], handlers={"validate": crash}, observer=recorder)
    for _ in range(3):
        world["clock"].t += 3600                                                                  # past any backoff
        worker.run_once()
    assert recorder.finished == [("validate", "retry", "execution"), ("validate", "retry", "execution"), ("validate", "failed", "execution")]
    task = [t for t in world["registry"].run_tasks(world["researcher"], run) if t["name"] == "validate"][0]
    assert task["state"] == "failed" and task["attempts"] == 3 and len(task["error"]["message"]) == 2000 and task["error"]["message"].startswith("segfault in the solver")


def test_run_until_idle_stops_at_exactly_the_number_of_tasks_it_was_given(world):
    submit(world)
    assert Worker(world["orch"]).run_until_idle(max_tasks=2) == 2
    assert Worker(world["orch"]).run_until_idle(max_tasks=0) == 0
    assert Worker(world["orch"]).run_until_idle() > 0                                             # and without a limit it finishes what is left


class Stop:
    """An event that says how often it was asked and how long it was told to wait."""
    def __init__(self, rounds):
        self.rounds, self.asked, self.waits = rounds, 0, []

    def is_set(self):
        self.asked += 1
        return self.asked > self.rounds

    def wait(self, seconds):
        self.waits.append(seconds)


def test_a_worker_with_nothing_to_do_waits_for_the_poll_interval_instead_of_spinning(world):
    worker = Worker(world["orch"])
    stop = Stop(rounds=3)
    worker.serve(stop, poll_s=0.25)
    assert stop.waits == [0.25, 0.25, 0.25]
    submit(world)
    busy = Stop(rounds=2)
    worker.serve(busy, poll_s=0.25)
    assert busy.waits == []                                                                       # while there is work it does not wait between tasks
