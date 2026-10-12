"""A worker: takes a task from the queues it serves, runs its handler, stores the output, and reports what happened.

A worker acts for a tenant, never as a person, and sees only the task it holds. Failures are classified (`status.classify_failure`) and only the ones a retry
could change are retried: a model that refused its input, a breach of a limit or a quota, a missing dataset will all happen again, so they fail the task at once.
"""
from __future__ import annotations

import threading
import time
import uuid

from .notify import Notifier
from .orchestrator import Orchestrator, Observer
from .security import QuotaExceededError
from .status import classify_failure, is_deterministic_refusal
from .taskqueue import ClaimedTask, LeaseLost
from ..experiment import parse_experiment
from .tasks import HANDLERS, STEP_KINDS, TaskContext

LEASE_MARGIN_S = 60.0


class Worker:
    def __init__(self, orchestrator: Orchestrator, *, queues=("cpu",), worker_id: str | None = None, handlers=None, observer: Observer | None = None,
                 notifier: Notifier | None = None):
        self.orchestrator = orchestrator
        self.registry, self.artifacts, self.queue = orchestrator.registry, orchestrator.artifacts, orchestrator.queue
        self.queues = tuple(queues)
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.handlers = HANDLERS if handlers is None else handlers
        self.observer = observer or orchestrator.observer
        self.notifier = notifier or orchestrator.notifier

    def run_once(self) -> bool:
        """Take at most one task and finish it, one way or another. Returns whether there was one."""
        self.queue.reap_expired()
        task = self.queue.claim(self.worker_id, self.queues)
        if task is None:
            return False
        self.queue.renew(task.id, self.worker_id, task.timeout_s + LEASE_MARGIN_S)
        self.observer.task_started(task)
        try:
            self._execute(task)
        finally:
            self.orchestrator.finalize_if_done(task.tenant_id, task.run_id)
            self.notifier.publish()                                                   # what depended on this task can be claimed now, by whichever worker is idle
        return True

    def run_until_idle(self, max_tasks: int = 10_000) -> int:
        done = 0
        while done < max_tasks and self.run_once():
            done += 1
        return done

    def serve(self, stop: threading.Event, poll_s: float = 0.5, idle_s: float = 10.0) -> None:
        """Run tasks until `stop` is set. With no work, wait `poll_s` and ask again; with a notifier that is connected, wait for its signal, asking anyway every `idle_s`
        (a retry that is due after a back-off, for one, sends no signal)."""
        self.notifier.start()
        while not stop.is_set():
            if not self.run_once():
                self.notifier.wait(stop, poll_s, idle_s)

    # -- one task ---------------------------------------------------------------------------------------------------------------------
    def _execute(self, task: ClaimedTask) -> None:
        tenant, run_id = task.tenant_id, task.run_id
        started = time.perf_counter()
        try:
            run = self.registry.run_in_tenant(tenant, run_id)
            creator = run["created_by"]
            limit, used = self.registry.quota(tenant, "compute_seconds_per_day"), self.registry.used_by(tenant, creator, "compute_seconds_per_day")
            if used >= limit:
                raise QuotaExceededError(f"compute_seconds_per_day is used up ({used:.0f} of {limit:.0f} s)")
            handler = self.handlers.get(task.kind)
            if handler is None:
                raise LookupError(f"no handler for task kind '{task.kind}'")
            context = TaskContext(task, self.registry, self.artifacts, self.orchestrator.pipeline, self.orchestrator.executor)
            if task.kind in STEP_KINDS:                                           # every step that wants a quantum backend goes through the gateway first
                self.orchestrator.gateway.authorize(tenant, creator, run_id, task.id, parse_experiment(context.experiment["spec"]).backend)
            output, lineage = handler(context)
            seconds = time.perf_counter() - started
            reference = self.artifacts.write_json(tenant, output, name=f"task-{task.name}.json", run_id=run_id)
            self.queue.complete(task.id, self.worker_id, reference.sha256, seconds)
        except LeaseLost:
            self.observer.task_finished(task, time.perf_counter() - started, "lost", None)
            return
        except Exception as error:                                               # noqa: BLE001 (every failure is classified and recorded, none is swallowed)
            axis = classify_failure(task.kind, error)
            info = {"type": type(error).__name__, "message": str(error)[:2000], "axis": axis}
            try:                                                                  # the outcome is recorded first: nothing after this can leave the task `running`
                outcome = self.queue.fail(task.id, self.worker_id, info, retry=not is_deterministic_refusal(error))
            except LeaseLost:
                outcome = "lost"
            if axis == "security" and outcome != "lost":
                self.registry.audit(tenant, "policy.blocked", task=task.id, run=run_id, error=type(error).__name__, message=info["message"][:300], result="BLOCKED")
                self.observer.blocked(type(error).__name__)
            self.observer.task_finished(task, time.perf_counter() - started, "retry" if outcome == "ready" else outcome, axis)
            return
        try:
            self.registry.record_usage(tenant, creator, "compute_seconds_per_day", seconds)
            if lineage:
                self.registry.append_event(f"lineage:{run_id}", tenant, "step", {**lineage, "artifact": reference.sha256, "worker": self.worker_id})
        finally:
            self.observer.task_finished(task, seconds, "succeeded", None)
