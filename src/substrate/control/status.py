"""A run's status on several axes, not one word.

"FAILED" hides what kind of failure it was, and the kinds need different responses: a solver that crashed is retried, a model that refused
because its assumptions do not hold is a scientific result, a request that broke a policy is a security event, a dataset that has vanished
is a reproducibility problem, and a prediction with a wide spread is a statement about confidence. So a run carries six independent verdicts:

    Execution        did the computation run to the end?
    Physics          did the models accept their inputs (validity regimes, invariants), and what did they warn about?
    Validation       was the request itself well formed (schema, units, ranges)?
    Security         did anything breach a policy (a limit, a quota, an authorisation)?
    Reproducibility  are the environment, seed and data versions recorded, resolved and (on a rerun) matching?
    Uncertainty      was spread estimated, and did it complete?

The axis a failure belongs to is decided by *where* it happened (see `classify_failure`), not by parsing a message.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..errors import DatasetUnavailableError, QCError, ResourceLimitError, SubstrateError, ValidationError


class State(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"          # the axis does not apply to this run (no ensemble was asked for)
    UNKNOWN = "UNKNOWN"          # the axis applies but the run ended before it could be judged


AXES = ("execution", "physics", "validation", "security", "reproducibility", "uncertainty")
_SEVERITY = {State.FAIL: 3, State.UNKNOWN: 2, State.WARN: 1, State.PASS: 0, State.SKIPPED: 0}


@dataclass
class AxisStatus:
    state: State = State.UNKNOWN
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"state": self.state.value, "reasons": list(self.reasons)}

    @classmethod
    def from_dict(cls, d: dict) -> "AxisStatus":
        return cls(State(d["state"]), list(d.get("reasons", [])))


@dataclass
class RunStatus:
    axes: dict[str, AxisStatus] = field(default_factory=lambda: {a: AxisStatus() for a in AXES})

    def set(self, axis: str, state: State, *reasons: str) -> None:
        if axis not in AXES:
            raise KeyError(f"unknown status axis '{axis}' (have: {', '.join(AXES)})")
        self.axes[axis] = AxisStatus(state, [r for r in reasons if r])

    def worsen(self, axis: str, state: State, *reasons: str) -> None:
        """Move an axis to `state` only if that is worse than where it is; reasons accumulate either way."""
        current = self.axes[axis]
        keep = state if _SEVERITY[state] > _SEVERITY[current.state] else current.state
        self.axes[axis] = AxisStatus(keep, current.reasons + [r for r in reasons if r])

    def overall(self) -> State:
        """The worst axis: FAIL, then UNKNOWN, then WARN, else PASS. Axes that do not apply are ignored."""
        worst = max(self.axes.values(), key=lambda a: _SEVERITY[a.state])
        return State.PASS if worst.state is State.SKIPPED else worst.state

    def to_dict(self) -> dict:
        return {a: s.to_dict() for a, s in self.axes.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "RunStatus":
        status = cls()
        for axis, value in d.items():
            if axis in AXES:
                status.axes[axis] = AxisStatus.from_dict(value)
        return status

    def summary(self) -> str:
        width = max(len(a) for a in AXES) + 1
        lines = [f"{a.capitalize() + ':':<{width + 1}} {self.axes[a].state.value}" + (f"  ({'; '.join(self.axes[a].reasons)})" if self.axes[a].reasons else "")
                 for a in AXES]
        return "\n".join(lines)


#: the kinds of task, and the axis a deterministic refusal in each belongs to
INPUT_TASKS = {"validate", "validate_transfer"}
DATA_TASKS = {"ingest"}
SCIENCE_TASKS = {"solve", "translate", "uncertainty", "check"}


def classify_failure(task_kind: str, error: BaseException) -> str:
    """The status axis a failure belongs to, from the kind of task that raised it and the type of the exception.

    A policy refusal is a security matter wherever it happens. A ValidationError is a bad request if it came from the input checks and a
    physical refusal (a model whose assumptions do not hold) if it came from the science; a missing or altered dataset is reproducibility; anything
    else (a crashed program, a timeout, a bug) is execution.
    """
    if isinstance(error, ResourceLimitError) or type(error).__name__ in ("QuotaExceededError", "AuthorizationError", "QuantumPolicyError"):
        return "security"
    if task_kind in DATA_TASKS and isinstance(error, (ValidationError, DatasetUnavailableError)):
        return "reproducibility"
    if task_kind in INPUT_TASKS and isinstance(error, SubstrateError) and not isinstance(error, QCError):
        return "validation"                                  # includes "no route between these scales" and "several equally short routes"
    if isinstance(error, ValidationError):
        return "physics"
    return "execution"


def is_deterministic_refusal(error: BaseException) -> bool:
    """A refusal that a retry cannot change: the same input will be refused again."""
    return isinstance(error, (ValidationError, ResourceLimitError, DatasetUnavailableError)) or type(error).__name__ in ("QuotaExceededError", "AuthorizationError")


SCIENCE_KINDS = ("solve", "translate", "uncertainty", "check", "reference", "transfer")


def assess(run: dict, tasks: list[dict], details: dict | None = None) -> RunStatus:
    """The six-axis status of a finished (or abandoned) run, from its row, its task rows and what the tasks found.

    `details` may hold: `warnings` (provenance warnings from the models), `non_finite` (observables containing NaN or infinity), `ensemble`
    (the sampling summary: `n_requested`, `n_ok`, `n_failed`), and `comparison` (a `repro.Comparison.to_dict()` against the run this one
    reproduces) with `original_environment_hash`. Failed tasks carry `error["axis"]`, set by the worker when the failure happened.
    """
    details = details or {}
    status = RunStatus()
    by_state = {s: [t for t in tasks if t["state"] == s] for s in ("succeeded", "failed", "skipped", "cancelled", "pending", "ready", "running")}
    failures = {}
    for task in by_state["failed"]:
        axis = (task.get("error") or {}).get("axis", "execution")
        failures.setdefault(axis, []).append(f"{task['name']}: {(task.get('error') or {}).get('message', 'failed')}")

    # execution: did the machinery run, regardless of what the models made of the input?
    if failures.get("execution"):
        status.set("execution", State.FAIL, *failures["execution"])
    elif by_state["cancelled"]:
        status.set("execution", State.FAIL, "the run was cancelled")
    elif by_state["pending"] or by_state["ready"] or by_state["running"]:
        status.set("execution", State.UNKNOWN, "the run has not finished")
    elif by_state["skipped"]:
        status.set("execution", State.PASS, f"stopped early: {len(by_state['skipped'])} task(s) not run because a task before them was refused")
    else:
        status.set("execution", State.PASS)

    # validation: was the request well formed?
    validated = any(t["kind"] in INPUT_TASKS and t["state"] == "succeeded" for t in tasks)
    if failures.get("validation"):
        status.set("validation", State.FAIL, *failures["validation"])
    elif validated:
        status.set("validation", State.PASS)
    else:
        status.set("validation", State.UNKNOWN, "the request was not validated")

    # physics: did the models accept their inputs, and what did they warn about?
    science = [t for t in tasks if t["kind"] in SCIENCE_KINDS]
    warnings = list(details.get("warnings") or [])
    non_finite = list(details.get("non_finite") or [])
    if failures.get("physics"):
        status.set("physics", State.FAIL, *failures["physics"])
    elif science and all(t["state"] == "succeeded" for t in science):
        if warnings or non_finite:
            status.set("physics", State.WARN, *warnings[:5], *([f"non-finite values in: {', '.join(non_finite)}"] if non_finite else []))
        else:
            status.set("physics", State.PASS)
    else:
        status.set("physics", State.UNKNOWN, "the models did not all run")

    # security: did anything breach a policy?
    if failures.get("security"):
        status.set("security", State.FAIL, *failures["security"])
    else:
        status.set("security", State.PASS)

    # reproducibility: is everything needed to run it again recorded, resolved and (on a rerun) matching?
    recorded = bool(run.get("environment_hash")) and run.get("seed") is not None
    comparison = details.get("comparison")
    if failures.get("reproducibility"):
        status.set("reproducibility", State.FAIL, *failures["reproducibility"])
    elif not recorded:
        status.set("reproducibility", State.FAIL, "the environment or seed was not recorded")
    elif run.get("reproduces"):
        if comparison is None:
            status.set("reproducibility", State.UNKNOWN, f"{run['reproduces']} has no results to compare with")
        elif not comparison["matches"]:
            status.set("reproducibility", State.FAIL, f"results differ from {run['reproduces']} (largest relative difference "
                       f"{comparison['max_relative_difference']:.3g} in {', '.join(comparison['differing'][:5]) or 'the set of observables'})")
        elif details.get("original_environment_hash") not in (None, run.get("environment_hash")):
            status.set("reproducibility", State.WARN, f"results match {run['reproduces']} but the software environment differs")
        else:
            status.set("reproducibility", State.PASS, f"matches {run['reproduces']} (largest relative difference {comparison['max_relative_difference']:.3g})")
    else:
        status.set("reproducibility", State.PASS)

    # uncertainty: was spread estimated, and did it complete?
    ensemble = details.get("ensemble")
    uncertainty_failed = [t for t in tasks if t["kind"] == "uncertainty" and t["state"] == "failed"]
    if not run.get("n_samples"):
        status.set("uncertainty", State.SKIPPED, "no ensemble was requested")
    elif failures.get("uncertainty") or uncertainty_failed:
        status.set("uncertainty", State.FAIL, *(failures.get("uncertainty") or [f"{t['name']}: {(t.get('error') or {}).get('message', 'failed')}" for t in uncertainty_failed]))
    elif ensemble is None:
        status.set("uncertainty", State.UNKNOWN, "the ensemble did not run")
    elif ensemble.get("n_failed", 0) > 0:
        status.set("uncertainty", State.WARN, f"{ensemble['n_failed']} of {ensemble['n_requested']} draws were invalid and dropped")
    else:
        status.set("uncertainty", State.PASS, f"{ensemble['n_ok']} draws")
    return status
