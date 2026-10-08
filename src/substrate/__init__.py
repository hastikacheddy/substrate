"""substrate: propagate scientific models across scales through a common intermediate representation."""
__version__ = "0.1.0"          # keep equal to pyproject.toml

from .backends import ClassicalBackend, SolverBackend, get_backend, register_backend
from .base import Engine, Registry, Translator, ValidationIssue
from .defaults import default_registry
from .errors import (
    AmbiguousPathError, NoPathError, QCError, QCUnavailableError, ResourceLimitError, SubstrateError, ValidationError,
)
from .experiment import Experiment, load_experiment
from .ir import ProvenanceRecord, Quantity, Scale, ScientificSystem
from .pipeline import Pipeline, RunResult

__all__ = [
    "ClassicalBackend", "SolverBackend", "get_backend", "register_backend",
    "Engine", "Registry", "Translator", "ValidationIssue", "default_registry",
    "AmbiguousPathError", "NoPathError", "QCError", "QCUnavailableError", "ResourceLimitError", "SubstrateError", "ValidationError",
    "Experiment", "load_experiment",
    "ProvenanceRecord", "Quantity", "Scale", "ScientificSystem",
    "Pipeline", "RunResult",
]
