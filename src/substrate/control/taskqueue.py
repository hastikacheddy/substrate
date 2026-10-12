"""The task queue: a state machine over the `tasks` table that any number of workers, in any number of processes, can share.

A task is `pending` until everything it depends on has succeeded, then `ready`; a worker takes a ready task whose queue it serves and holds it under a
lease (`running`); finishing moves it to `succeeded` or, after a failure that is worth another try, back to `ready` after a backoff, or to `failed`.
When a task fails for good, everything downstream of it is marked `skipped` straight away, so a run can never wait on work that will not happen.

Taking a task is a single conditional UPDATE (only if it is still `ready`), so two workers cannot take the same task, on SQLite or PostgreSQL. A worker
that dies mid-task simply stops renewing its lease; `reap_expired` hands the task to someone else (counting the lost attempt), and the dead worker's late
`complete` is refused because the lease is no longer its own.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import sqlalchemy as sa

from ..errors import SubstrateError
from . import db
from .registry import Registry

ACTIVE = ("pending", "ready", "running")
TERMINAL = ("succeeded", "failed", "skipped", "cancelled")


class LeaseLost(SubstrateError):
    """A worker tried to finish a task it no longer holds (its lease expired and the task went elsewhere, or the run was cancelled)."""


@dataclass(frozen=True)
class ClaimedTask:
    id: str
    run_id: str
    tenant_id: str
    name: str
    kind: str
    queue: str
    payload: dict
    depends_on: list
    attempts: int
    max_attempts: int
    timeout_s: float
    resources: dict


class TaskQueue:
    def __init__(self, registry: Registry, *, backoff_base_s: float = 2.0, backoff_cap_s: float = 300.0):
        self.registry = registry
        self.engine = registry.engine
        self.clock = registry.clock
        self.backoff_base_s, self.backoff_cap_s = backoff_base_s, backoff_cap_s

    # -- taking work -----------------------------------------------------------------------------------------------------------
    def claim(self, worker_id: str, queues: Sequence[str], *, lease_s: float = 300.0) -> ClaimedTask | None:
        """The oldest ready task on one of `queues`, now held by `worker_id` for `lease_s` seconds, or None."""
        now = self.clock()
        with self.engine.connect() as conn:
            candidates = conn.execute(
                sa.select(db.tasks.c.id).where(db.tasks.c.state == "ready", db.tasks.c.queue.in_(list(queues)), db.tasks.c.not_before <= now)
                .order_by(db.tasks.c.id).limit(20)).scalars().all()
        for task_id in candidates:
            with self.engine.begin() as conn:
                taken = conn.execute(
                    sa.update(db.tasks).where(db.tasks.c.id == task_id, db.tasks.c.state == "ready")
                    .values(state="running", lease_owner=worker_id, lease_expires=now + lease_s, started_at=now, attempts=db.tasks.c.attempts + 1))
                if taken.rowcount != 1:
                    continue                                                          # someone else got it first: try the next
                row = conn.execute(sa.select(db.tasks).where(db.tasks.c.id == task_id)).one()
            self.registry.mark_run_started(row.tenant_id, row.run_id)
            return ClaimedTask(row.id, row.run_id, row.tenant_id, row.name, row.kind, row.queue, row.payload, row.depends_on, row.attempts,
                               row.max_attempts, row.timeout_s, row.resources)
        return None

    def renew(self, task_id: str, worker_id: str, lease_s: float = 300.0) -> bool:
        with self.engine.begin() as conn:
            done = conn.execute(sa.update(db.tasks).where(db.tasks.c.id == task_id, db.tasks.c.state == "running", db.tasks.c.lease_owner == worker_id)
                                .values(lease_expires=self.clock() + lease_s))
        return done.rowcount == 1

    # -- finishing --------------------------------------------------------------------------------------------------------------
    def complete(self, task_id: str, worker_id: str, output: str | None, seconds: float) -> list[str]:
        """Mark the task succeeded and make ready every pending task of the run whose dependencies have now all succeeded. Returns their ids."""
        now = self.clock()
        with self.engine.begin() as conn:
            done = conn.execute(
                sa.update(db.tasks).where(db.tasks.c.id == task_id, db.tasks.c.state == "running", db.tasks.c.lease_owner == worker_id)
                .values(state="succeeded", output=output, seconds=seconds, finished_at=now, lease_owner=None, lease_expires=None, error=None))
            if done.rowcount != 1:
                raise LeaseLost(f"{worker_id} no longer holds {task_id}")
            run_id = conn.execute(sa.select(db.tasks.c.run_id).where(db.tasks.c.id == task_id)).scalar_one()
            return self._promote(conn, run_id)

    def _promote(self, conn, run_id: str) -> list[str]:
        rows = conn.execute(sa.select(db.tasks.c.id, db.tasks.c.state, db.tasks.c.depends_on).where(db.tasks.c.run_id == run_id)).all()
        state = {r.id: r.state for r in rows}
        promoted = []
        for row in rows:
            if row.state == "pending" and all(state.get(d) == "succeeded" for d in row.depends_on):
                conn.execute(sa.update(db.tasks).where(db.tasks.c.id == row.id, db.tasks.c.state == "pending").values(state="ready"))
                promoted.append(row.id)
        return promoted

    def fail(self, task_id: str, worker_id: str, error: dict, *, retry: bool) -> str:
        """Record a failure. With `retry` and attempts left the task goes back to `ready` after an exponential backoff; otherwise it is `failed` and
        everything that depends on it, directly or not, is `skipped`. Returns the task's new state."""
        now = self.clock()
        with self.engine.begin() as conn:
            row = conn.execute(sa.select(db.tasks.c.attempts, db.tasks.c.max_attempts, db.tasks.c.run_id, db.tasks.c.lease_owner, db.tasks.c.state)
                               .where(db.tasks.c.id == task_id)).one()
            if row.state != "running" or row.lease_owner != worker_id:
                raise LeaseLost(f"{worker_id} no longer holds {task_id}")
            if retry and row.attempts < row.max_attempts:
                delay = min(self.backoff_cap_s, self.backoff_base_s * 2 ** (row.attempts - 1))
                conn.execute(sa.update(db.tasks).where(db.tasks.c.id == task_id).values(
                    state="ready", not_before=now + delay, lease_owner=None, lease_expires=None, error={**error, "retrying_in_s": delay}))
                return "ready"
            conn.execute(sa.update(db.tasks).where(db.tasks.c.id == task_id).values(
                state="failed", error=error, finished_at=now, lease_owner=None, lease_expires=None))
            self._skip_dependents(conn, row.run_id, task_id)
            return "failed"

    def _skip_dependents(self, conn, run_id: str, failed_id: str) -> None:
        rows = conn.execute(sa.select(db.tasks.c.id, db.tasks.c.state, db.tasks.c.depends_on).where(db.tasks.c.run_id == run_id)).all()
        doomed, changed = {failed_id}, True
        while changed:                                                                # transitive closure over "depends on something doomed"
            changed = False
            for row in rows:
                if row.id not in doomed and any(d in doomed for d in row.depends_on):
                    doomed.add(row.id)
                    changed = True
        for row in rows:
            if row.id in doomed and row.id != failed_id and row.state in ("pending", "ready"):
                conn.execute(sa.update(db.tasks).where(db.tasks.c.id == row.id).values(
                    state="skipped", finished_at=self.clock(), error={"reason": f"dependency {failed_id} failed"}))

    def reap_expired(self) -> int:
        """Take back tasks whose worker stopped renewing its lease: another attempt if any are left, else failed."""
        now = self.clock()
        with self.engine.connect() as conn:
            lost = conn.execute(sa.select(db.tasks.c.id, db.tasks.c.lease_owner).where(db.tasks.c.state == "running", db.tasks.c.lease_expires < now)).all()
        count = 0
        for task_id, owner in lost:
            try:
                self.fail(task_id, owner, {"type": "LeaseExpired", "axis": "execution", "message": "the worker stopped renewing its lease"}, retry=True)
                count += 1
            except LeaseLost:
                pass                                                                   # it finished (or was reaped) between the query and now
        return count

    def cancel_run(self, tenant_id: str, run_id: str) -> int:
        """Cancel every task of the run that has not finished. A task already running keeps running, but its worker's `complete` is refused."""
        now = self.clock()
        with self.engine.begin() as conn:
            changed = conn.execute(sa.update(db.tasks).where(db.tasks.c.run_id == run_id, db.tasks.c.tenant_id == tenant_id, db.tasks.c.state.in_(list(ACTIVE)))
                                   .values(state="cancelled", finished_at=now, lease_owner=None, lease_expires=None, error={"reason": "run cancelled"}))
        return changed.rowcount

    # -- looking --------------------------------------------------------------------------------------------------------------------
    def counts(self, tenant_id: str, run_id: str) -> dict[str, int]:
        with self.engine.connect() as conn:
            rows = conn.execute(sa.select(db.tasks.c.state, sa.func.count()).where(db.tasks.c.run_id == run_id, db.tasks.c.tenant_id == tenant_id)
                                .group_by(db.tasks.c.state)).all()
        return {state: n for state, n in rows}

    def depth(self) -> dict[str, int]:
        """Ready tasks per queue, across every tenant: what an operator watches."""
        with self.engine.connect() as conn:
            rows = conn.execute(sa.select(db.tasks.c.queue, sa.func.count()).where(db.tasks.c.state == "ready").group_by(db.tasks.c.queue)).all()
        return {queue: n for queue, n in rows}

    def run_is_finished(self, tenant_id: str, run_id: str) -> bool:
        counts = self.counts(tenant_id, run_id)
        return bool(counts) and not any(counts.get(s, 0) for s in ACTIVE)

    def task(self, tenant_id: str, task_id: str) -> dict | None:
        with self.engine.connect() as conn:
            row = conn.execute(sa.select(db.tasks).where(db.tasks.c.id == task_id, db.tasks.c.tenant_id == tenant_id)).first()
        return None if row is None else dict(row._mapping)
