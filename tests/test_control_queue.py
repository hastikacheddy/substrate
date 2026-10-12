"""The task queue as a state machine: who may take what, when dependents wake up, how failures retry or cascade, and that a task is taken once."""
import threading

import pytest

pytest.importorskip("sqlalchemy")

from substrate.control.registry import Registry, TaskSpec  # noqa: E402
from substrate.control.security import Role  # noqa: E402
from substrate.control.taskqueue import LeaseLost, TaskQueue  # noqa: E402

SPEC = {"system": {"name": "w", "scale": "quantum", "kind": "quantum.double_well_1d", "parameters": {}}}


class Clock:
    def __init__(self):
        self.t = 1_800_000_000.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def world(tmp_path, clock):
    registry = Registry(f"sqlite:///{tmp_path / 'q.sqlite3'}", clock=clock)
    tenant, admin, _ = registry.bootstrap("lab")
    registry.set_quota(admin, "concurrent_runs", 100)
    researcher = registry.create_principal(tenant, "ada", Role.RESEARCHER)
    experiment = registry.add_experiment(researcher, "x", SPEC)
    return {"registry": registry, "tenant": tenant, "who": researcher, "exp": experiment["id"], "queue": TaskQueue(registry), "admin": admin}


def make_run(world, tasks):
    return world["registry"].create_run(world["who"], world["exp"], seed=0, n_samples=0, environment={}, environment_hash="h", tasks=tasks)["id"]


def states(world, run):
    return {t["name"]: t["state"] for t in world["registry"].run_tasks(world["who"], run)}


def diamond(**kw):
    return [TaskSpec("a", "validate", **kw), TaskSpec("b", "solve", depends_on=["a"], **kw), TaskSpec("c", "solve", depends_on=["a"], **kw),
            TaskSpec("d", "check", depends_on=["b", "c"], **kw)]


def drain_one(world, worker="w1", queues=("cpu",)):
    claimed = world["queue"].claim(worker, queues)
    assert claimed is not None
    world["queue"].complete(claimed.id, worker, None, 0.1)
    return claimed


# -- taking work ----------------------------------------------------------------------------------------------------------------------
def test_only_ready_tasks_are_taken_oldest_first_and_each_only_once(world):
    run = make_run(world, diamond())
    first = world["queue"].claim("w1", ["cpu"])
    assert first.name == "a" and first.attempts == 1 and first.run_id == run
    assert world["queue"].claim("w2", ["cpu"]) is None                                    # b, c, d are pending until a succeeds
    assert states(world, run)["a"] == "running"


def test_a_worker_only_takes_tasks_from_the_queues_it_serves(world):
    run = make_run(world, [TaskSpec("chem", "solve", queue="chemistry"), TaskSpec("plain", "solve", queue="cpu"), TaskSpec("q", "solve", queue="quantum")])
    assert world["queue"].claim("w", ["gpu"]) is None
    assert world["queue"].claim("w", ["cpu"]).name == "plain"
    assert world["queue"].claim("w", ["cpu"]) is None
    assert {world["queue"].claim("w", ["chemistry", "quantum"]).name, world["queue"].claim("w", ["chemistry", "quantum"]).name} == {"chem", "q"}
    assert world["queue"].depth() == {}


def test_the_first_claim_marks_the_run_started_once(world, clock):
    run = make_run(world, diamond())
    assert world["registry"].get_run(world["who"], run)["state"] == "queued"
    world["queue"].claim("w", ["cpu"])
    started = world["registry"].get_run(world["who"], run)
    assert started["state"] == "running" and started["started_at"] == clock.t


def test_the_claim_carries_what_a_worker_needs(world):
    make_run(world, [TaskSpec("t", "solve", payload={"index": 3}, max_attempts=5, timeout_s=12.5, resources={"cpu": 2})])
    claimed = world["queue"].claim("w", ["cpu"])
    assert (claimed.kind, claimed.payload, claimed.max_attempts, claimed.timeout_s, claimed.resources, claimed.depends_on) == ("solve", {"index": 3}, 5, 12.5, {"cpu": 2}, [])


def test_many_workers_racing_for_many_tasks_take_each_task_exactly_once(world):
    make_run(world, [TaskSpec(f"t{i}", "solve") for i in range(40)])
    taken, lock = [], threading.Lock()

    def worker(n):
        while True:
            claimed = world["queue"].claim(f"w{n}", ["cpu"])
            if claimed is None:
                return
            with lock:
                taken.append(claimed.id)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(taken) == 40 and len(set(taken)) == 40


# -- completing and waking dependents ------------------------------------------------------------------------------------------------
def test_dependents_wake_only_when_every_dependency_has_succeeded(world):
    run = make_run(world, diamond())
    drain_one(world)                                                                         # a
    assert states(world, run) == {"a": "succeeded", "b": "ready", "c": "ready", "d": "pending"}
    drain_one(world)                                                                         # b
    assert states(world, run)["d"] == "pending"                                              # still waiting for c
    drain_one(world)                                                                         # c
    assert states(world, run) == {"a": "succeeded", "b": "succeeded", "c": "succeeded", "d": "ready"}
    drain_one(world)
    assert world["queue"].run_is_finished(world["tenant"], run)


def test_completing_returns_the_tasks_it_woke_and_records_output_and_time(world):
    run = make_run(world, diamond())
    claimed = world["queue"].claim("w", ["cpu"])
    woken = world["queue"].complete(claimed.id, "w", "ab" * 32, 1.5)
    assert sorted(woken) == [f"{run}/01", f"{run}/02"]
    row = world["queue"].task(world["tenant"], claimed.id)
    assert (row["state"], row["output"], row["seconds"], row["lease_owner"]) == ("succeeded", "ab" * 32, 1.5, None)


def test_only_the_holder_can_complete_and_only_once(world):
    make_run(world, diamond())
    claimed = world["queue"].claim("w1", ["cpu"])
    with pytest.raises(LeaseLost):
        world["queue"].complete(claimed.id, "w2", None, 0.0)
    world["queue"].complete(claimed.id, "w1", None, 0.0)
    with pytest.raises(LeaseLost):
        world["queue"].complete(claimed.id, "w1", None, 0.0)


# -- failing: retry, give up, cascade --------------------------------------------------------------------------------------------------
def test_a_retryable_failure_waits_out_an_exponential_backoff_and_then_gives_up(world, clock):
    run = make_run(world, [TaskSpec("t", "solve", max_attempts=4)])
    waits = []
    for attempt in range(1, 4):
        claimed = world["queue"].claim("w", ["cpu"])
        assert claimed.attempts == attempt
        assert world["queue"].fail(claimed.id, "w", {"type": "Boom"}, retry=True) == "ready"
        row = world["queue"].task(world["tenant"], claimed.id)
        waits.append(row["not_before"] - clock.t)
        assert world["queue"].claim("w", ["cpu"]) is None                                    # not before the backoff has passed
        clock.t += waits[-1]
    assert waits == [2.0, 4.0, 8.0]
    last = world["queue"].claim("w", ["cpu"])
    assert last.attempts == 4
    assert world["queue"].fail(last.id, "w", {"type": "Boom"}, retry=True) == "failed"        # no attempts left
    assert states(world, run)["t"] == "failed"


def test_the_backoff_is_capped(world, clock):
    queue = TaskQueue(world["registry"], backoff_base_s=100.0, backoff_cap_s=250.0)
    make_run(world, [TaskSpec("t", "solve", max_attempts=10)])
    delays = []
    for _ in range(4):
        claimed = queue.claim("w", ["cpu"])
        queue.fail(claimed.id, "w", {}, retry=True)
        delays.append(queue.task(world["tenant"], claimed.id)["not_before"] - clock.t)
        clock.t += delays[-1]
    assert delays == [100.0, 200.0, 250.0, 250.0]


def test_a_failure_that_will_not_be_retried_fails_the_task_and_skips_everything_downstream(world):
    run = make_run(world, [TaskSpec("a", "validate"), TaskSpec("b", "solve", depends_on=["a"]), TaskSpec("c", "solve", depends_on=["b"]),
                           TaskSpec("d", "solve", depends_on=["a"]), TaskSpec("e", "check", depends_on=["c", "d"])])
    drain_one(world)                                                                         # a
    claimed = world["queue"].claim("w", ["cpu"])                                             # b
    assert claimed.name == "b"
    assert world["queue"].fail(claimed.id, "w", {"type": "ValidationError", "axis": "physics"}, retry=False) == "failed"
    assert states(world, run) == {"a": "succeeded", "b": "failed", "c": "skipped", "d": "ready", "e": "skipped"}     # d does not depend on b
    skipped = world["registry"].run_tasks(world["who"], run)
    assert [t["error"] for t in skipped if t["name"] == "c"] == [{"reason": f"dependency {claimed.id} failed"}]
    drain_one(world)                                                                         # d still runs
    assert world["queue"].run_is_finished(world["tenant"], run)


def test_only_the_holder_can_fail_a_task(world):
    make_run(world, diamond())
    claimed = world["queue"].claim("w1", ["cpu"])
    with pytest.raises(LeaseLost):
        world["queue"].fail(claimed.id, "w2", {}, retry=False)


# -- leases ----------------------------------------------------------------------------------------------------------------------------
def test_a_task_whose_worker_vanished_is_given_to_someone_else_and_the_late_result_is_refused(world, clock):
    run = make_run(world, [TaskSpec("t", "solve", max_attempts=3)])
    claimed = world["queue"].claim("dead", ["cpu"], lease_s=60)
    assert world["queue"].reap_expired() == 0                                                # the lease is still good
    clock.t += 61
    assert world["queue"].reap_expired() == 1
    row = world["queue"].task(world["tenant"], claimed.id)
    assert row["state"] == "ready" and row["error"]["type"] == "LeaseExpired" and row["attempts"] == 1
    clock.t += 3                                                                             # past the first backoff
    again = world["queue"].claim("alive", ["cpu"])
    assert again.id == claimed.id and again.attempts == 2
    with pytest.raises(LeaseLost):
        world["queue"].complete(claimed.id, "dead", None, 0.0)                               # the dead worker comes back too late
    world["queue"].complete(again.id, "alive", None, 0.0)
    assert states(world, run) == {"t": "succeeded"}


def test_repeated_lease_loss_ends_in_failure_not_a_loop(world, clock):
    run = make_run(world, [TaskSpec("t", "solve", max_attempts=2), TaskSpec("u", "check", depends_on=["t"])])
    for _ in range(2):
        world["queue"].claim("w", ["cpu"], lease_s=10)
        clock.t += 11
        world["queue"].reap_expired()
        clock.t += 10
    assert states(world, run) == {"t": "failed", "u": "skipped"}


def test_renewing_a_lease_keeps_the_task(world, clock):
    make_run(world, [TaskSpec("t", "solve")])
    claimed = world["queue"].claim("w", ["cpu"], lease_s=60)
    clock.t += 50
    assert world["queue"].renew(claimed.id, "w", 60) is True
    clock.t += 50
    assert world["queue"].reap_expired() == 0
    assert world["queue"].renew(claimed.id, "someone-else", 60) is False


# -- cancelling and looking ------------------------------------------------------------------------------------------------------------
def test_cancelling_a_run_stops_waiting_tasks_and_refuses_the_running_ones_result(world):
    run = make_run(world, diamond())
    claimed = world["queue"].claim("w", ["cpu"])
    assert world["queue"].cancel_run(world["tenant"], run) == 4
    assert set(states(world, run).values()) == {"cancelled"}
    with pytest.raises(LeaseLost):
        world["queue"].complete(claimed.id, "w", None, 0.0)
    assert world["queue"].claim("w", ["cpu"]) is None and world["queue"].run_is_finished(world["tenant"], run)
    assert world["queue"].cancel_run(world["tenant"], run) == 0                              # nothing left to cancel


def test_one_tenant_cannot_cancel_or_look_at_anothers_run(world):
    run = make_run(world, diamond())
    other, _, _ = world["registry"].bootstrap("lab-b", "boris")
    assert world["queue"].cancel_run(other, run) == 0
    assert world["queue"].counts(other, run) == {} and world["queue"].task(other, f"{run}/00") is None
    assert states(world, run)["a"] == "ready"


def test_counts_depth_and_finished(world):
    run = make_run(world, diamond())
    assert world["queue"].counts(world["tenant"], run) == {"ready": 1, "pending": 3}
    assert world["queue"].depth() == {"cpu": 1} and not world["queue"].run_is_finished(world["tenant"], run)
    assert not world["queue"].run_is_finished(world["tenant"], "RUN-999999")                 # a run with no tasks is not "finished"
