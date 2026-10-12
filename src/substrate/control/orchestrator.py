"""Submitting runs, cancelling them, reproducing them, and closing them out with a six-axis status.

The orchestrator is the only place that knows how a run begins (authorisation, quotas, the task graph, the environment record) and how it ends
(the status, the result digest, the comparison with the run it reproduces). Workers only execute tasks.
"""
from __future__ import annotations

from ..errors import ValidationError
from ..experiment import parse_experiment
from ..limits import whole_number
from ..pipeline import Pipeline
from .artifacts import Artifacts
from .gateway import QPUGateway
from .notify import Notifier
from .registry import Registry
from .repro import compare_summaries, environment_fingerprint, environment_hash
from .security import Action, Principal
from .status import RunStatus, assess
from .steps import InlineExecutor
from .tasks import build_tasks, build_transfer_tasks, is_transfer_spec, parse_transfer_spec
from .taskqueue import TaskQueue

TERMINAL_RUN_STATES = ("succeeded", "failed", "cancelled")

#: How closely a reproduction must agree with the run it reproduces. The solvers are adaptive: a change of one unit in the last place of an input
#: rate constant (2.2e-16, as can come from a different order of summation) moves the integrated output by up to ~1e-9 relative, the tolerance the
#: ODE solver is run at. So bitwise agreement is not to be expected and 1e-8 would be flaky; 1e-6 is a thousand times the solver tolerance and still
#: far below any difference that matters (a changed parameter moves results by 1e-3 or more).
REPRODUCTION_RTOL = 1e-6


class Observer:
    """Hooks for metrics and traces. Every method does nothing; `observability.PrometheusObserver` and friends override the ones they care about."""

    def run_submitted(self, run: dict) -> None: ...
    def task_started(self, task) -> None: ...
    def task_finished(self, task, seconds: float, outcome: str, axis: str | None) -> None: ...
    def run_finished(self, run: dict, status: RunStatus) -> None: ...
    def blocked(self, kind: str) -> None: ...
    def auth_failed(self, reason: str) -> None: ...


class Orchestrator:
    def __init__(self, registry: Registry, artifacts: Artifacts, *, pipeline: Pipeline | None = None, executor=None, observer: Observer | None = None,
                 gateway: QPUGateway | None = None, notifier: Notifier | None = None):
        from ..defaults import default_registry
        self.registry, self.artifacts = registry, artifacts
        self.pipeline = pipeline or Pipeline(default_registry())
        self.gateway = gateway or QPUGateway(registry)
        self.executor = executor or InlineExecutor(self.gateway.policy)
        self.observer = observer or Observer()
        self.notifier = notifier or Notifier()                                               # tells idle workers that there is something to claim; the queue stays the database
        self.queue = TaskQueue(registry)

    # -- beginning ----------------------------------------------------------------------------------------------------------------
    def _plan(self, principal: Principal, spec: dict, datasets: list[dict], n_samples: int | None):
        """(tasks, ensemble size) for a stored spec of either kind. Raises for anything that cannot run."""
        if is_transfer_spec(spec):
            if n_samples:
                raise ValidationError("a transfer study has no ensemble")
            return build_transfer_tasks(spec, lambda ref: (self.registry.experiment_in_tenant(principal.tenant_id, ref) or {}).get("spec"), self.pipeline), 0
        parsed = parse_experiment(spec)
        samples = parsed.n_samples if n_samples is None else whole_number("run", "n_samples", n_samples, minimum=0, maximum="max_ensemble")
        return build_tasks(spec, self.pipeline, n_samples=samples, datasets=datasets), samples

    def check_spec(self, principal: Principal, spec: dict, datasets: list[dict] | None = None) -> int:
        """Refuse a spec that could never run, before it is stored. Returns how many tasks a run of it would have."""
        return len(self._plan(principal, spec, datasets or [], None)[0])

    def submit_transfer(self, principal: Principal, name: str, references: list[str], *, window_ev: float | None = None, sigma_ev: float = 0.05) -> dict:
        """Store a transfer study as an experiment (so it is hashed, listed and reproducible like any other) and start a run of it."""
        principal.require(Action.TRANSFER_RUN)
        spec = {"transfer": {"references": list(references), "window_ev": window_ev, "sigma_ev": sigma_ev}}
        parse_transfer_spec(spec)
        self.check_spec(principal, spec)
        experiment = self.registry.add_experiment(principal, name, spec)
        return self.submit(principal, experiment["id"])

    def transfer_result(self, principal: Principal, run_id: str) -> dict | None:
        principal.require(Action.RUN_READ)
        if self.registry.get_run(principal, run_id) is None:
            return None
        return self._output(principal.tenant_id, self.registry.tasks_in_tenant(principal.tenant_id, run_id), "transfer")

    def submit(self, principal: Principal, experiment_id: str, *, seed: int | None = None, n_samples: int | None = None, reproduces: str | None = None) -> dict:
        principal.require(Action.RUN_REPRODUCE if reproduces else Action.RUN_CREATE)
        experiment = self.registry.experiment_in_tenant(principal.tenant_id, experiment_id)
        if experiment is None:
            raise LookupError(f"no experiment {experiment_id}")
        tasks, samples = self._plan(principal, experiment["spec"], experiment["datasets"], n_samples)                    # refuses before anything is spent
        seed = (0 if is_transfer_spec(experiment["spec"]) else parse_experiment(experiment["spec"]).seed) if seed is None else whole_number("run", "seed", seed)
        charges = {"runs_per_day": 1.0, "tasks_per_day": float(len(tasks))}
        done = []
        try:
            for metric, amount in charges.items():
                self.registry.consume(principal, metric, amount)
                done.append((metric, amount))
            fingerprint = environment_fingerprint()
            run = self.registry.create_run(principal, experiment_id, seed=seed, n_samples=samples, environment=fingerprint,
                                           environment_hash=environment_hash(fingerprint), tasks=tasks, reproduces=reproduces)
        except BaseException:
            for metric, amount in done:                                                      # a request that was refused did not use its allowance
                self.registry.record_usage(principal.tenant_id, principal.id, metric, -amount)
            self.observer.blocked("submit")
            raise
        self.observer.run_submitted(run)
        self.notifier.publish()
        return run

    def reproduce(self, principal: Principal, run_id: str) -> dict:
        """A new run of the same experiment with the same seed and ensemble size; when it finishes it is compared with this one."""
        principal.require(Action.RUN_REPRODUCE)
        original = self.registry.run_in_tenant(principal.tenant_id, run_id)
        if original is None:
            raise LookupError(f"no run {run_id}")
        if original["state"] != "succeeded":
            raise ValidationError(f"{run_id} is {original['state']}: only a run that succeeded can be reproduced")
        return self.submit(principal, original["experiment_id"], seed=original["seed"], n_samples=original["n_samples"], reproduces=run_id)

    def cancel(self, principal: Principal, run_id: str) -> int:
        principal.require(Action.RUN_CANCEL)
        run = self.registry.run_in_tenant(principal.tenant_id, run_id)
        if run is None:
            raise LookupError(f"no run {run_id}")
        cancelled = self.queue.cancel_run(principal.tenant_id, run_id)
        self.registry.audit(principal.tenant_id, "run.cancel", by=principal.id, run=run_id, tasks=cancelled)
        self.finalize_if_done(principal.tenant_id, run_id)
        return cancelled

    # -- ending --------------------------------------------------------------------------------------------------------------------
    def _output(self, tenant_id: str, rows: list[dict], name: str) -> dict | None:
        for row in rows:
            if row["name"] == name and row["state"] == "succeeded" and row["output"]:
                return self.artifacts.read_json(tenant_id, row["output"])
        return None

    def final_observables(self, tenant_id: str, run_id: str) -> dict | None:
        """The stored summary of a run's final observables, or None if it did not get as far as writing one."""
        report = self._output(tenant_id, self.registry.tasks_in_tenant(tenant_id, run_id), "report")
        return None if report is None else self.artifacts.read_json(tenant_id, report["observables"])

    def finalize_if_done(self, tenant_id: str, run_id: str) -> RunStatus | None:
        """If every task of the run has ended, record how it ended. Safe to call from any worker, any number of times: the first one wins."""
        run = self.registry.run_in_tenant(tenant_id, run_id)
        if run is None or run["state"] in TERMINAL_RUN_STATES or not self.queue.run_is_finished(tenant_id, run_id):
            return None
        rows = self.registry.tasks_in_tenant(tenant_id, run_id)
        check, uncertainty, report = (self._output(tenant_id, rows, n) for n in ("check", "uncertainty", "report"))
        details = {"warnings": (check or {}).get("warnings", []), "non_finite": (check or {}).get("non_finite", []),
                   "ensemble": (uncertainty or {}).get("ensemble")}
        comparison = None
        if run["reproduces"]:
            original = self.registry.run_in_tenant(tenant_id, run["reproduces"])
            theirs = self.final_observables(tenant_id, run["reproduces"]) if original else None
            ours = self.artifacts.read_json(tenant_id, report["observables"]) if report else None
            if theirs is not None and ours is not None:
                comparison = compare_summaries(theirs, ours, rtol=REPRODUCTION_RTOL)
                details["comparison"] = comparison.to_dict()
            details["original_environment_hash"] = original["environment_hash"] if original else None
        status = assess(run, rows, details)
        states = {r["state"] for r in rows}
        state = "failed" if "failed" in states else "cancelled" if "cancelled" in states else "succeeded"
        if not self.registry.finish_run(tenant_id, run_id, state, status.to_dict(), report["result_digest"] if report else None):
            return None                                                                          # another worker closed it first
        document = {"run": run_id, "state": state, "overall": status.overall().value, "status": status.to_dict()}
        if comparison is not None:
            document["reproduction"] = {"of": run["reproduces"], **comparison.to_dict()}
        self.artifacts.write_json(tenant_id, document, name="status.json", run_id=run_id)
        self.registry.audit(tenant_id, "run.finish", run=run_id, state=state, overall=status.overall().value)
        self.observer.run_finished({**run, "state": state}, status)
        return status

    # -- looking --------------------------------------------------------------------------------------------------------------------
    def describe(self, principal: Principal, run_id: str) -> dict | None:
        """A run as the API shows it: its row, progress counts and the status so far."""
        principal.require(Action.RUN_READ)
        run = self.registry.get_run(principal, run_id)
        if run is None:
            return None
        tasks = self.registry.run_tasks(principal, run_id)
        progress = self.queue.counts(principal.tenant_id, run_id)
        return {**{k: run[k] for k in ("id", "experiment_id", "state", "seed", "n_samples", "reproduces", "created_at", "started_at", "finished_at",
                                       "environment_hash", "result_digest")},
                "status": run["status"], "progress": progress,
                "tasks": [{k: t[k] for k in ("id", "name", "kind", "queue", "state", "attempts", "seconds")} | {"error": t["error"]} for t in tasks]}

