"""Applying one planned step to one system, as plain data in and plain data out: the unit of work both executors (in-process and sandboxed) perform."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..ir import ScientificSystem
from ..pipeline import Pipeline
from ..qpu import QuantumPolicy, backend_from_config

CLASSICAL = {"type": "classical", "provider": "scipy"}


@dataclass(frozen=True)
class StepRequest:
    root: dict                        # the root system, as JSON data
    propagation: list[str]
    index: int
    system: dict                      # the system this step is applied to
    timeout_s: float
    backend: dict = field(default_factory=lambda: dict(CLASSICAL))

    def to_dict(self) -> dict:
        return asdict(self)


def apply_step_data(pipeline: Pipeline, request: StepRequest, policy: QuantumPolicy | None = None) -> ScientificSystem:
    if request.backend != CLASSICAL:
        pipeline = Pipeline(pipeline.registry, backend_from_config(request.backend, policy))      # the same engines on the solver the experiment asked for
    root = ScientificSystem.from_dict(request.root)
    steps = pipeline.plan(root, request.propagation)
    return pipeline.apply_step(steps[request.index], ScientificSystem.from_dict(request.system))


class InlineExecutor:
    """Runs the step in the worker's own process. Fast, but a runaway step can only be waited for; see `sandbox.ProcessExecutor` for the isolated kind."""

    quantum_policy: QuantumPolicy | None = None

    def __init__(self, quantum_policy: QuantumPolicy | None = None):
        self.quantum_policy = quantum_policy

    def apply(self, pipeline: Pipeline, request: StepRequest) -> dict:
        return apply_step_data(pipeline, request, self.quantum_policy).to_dict()
