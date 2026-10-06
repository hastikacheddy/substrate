"""Contracts: Engine (solves a system at one scale), Translator (maps between scales), Registry."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass

from .backends import SolverBackend
from .errors import AmbiguousPathError, NoPathError, SubstrateError, ValidationError
from .ir import Scale, ScientificSystem


@dataclass(frozen=True)
class ValidationIssue:
    severity: str            # "error" blocks the pipeline; "warning" is recorded in provenance
    message: str


class Engine(ABC):
    """Solves systems at one scale and returns them with observables (and final state) filled in."""

    name: str
    version: str = "0.1.0"
    scale: Scale
    kinds: tuple[str, ...]
    approximations: tuple[str, ...] = ()

    @abstractmethod
    def solve(self, system: ScientificSystem, backend: SolverBackend) -> ScientificSystem: ...

    def check_kind(self, system: ScientificSystem) -> None:
        """Guard for engines called directly rather than through the Registry."""
        if system.scale != self.scale or system.kind not in self.kinds:
            raise ValidationError(
                f"{self.name} solves {self.scale.label} systems of kind {', '.join(self.kinds)}; "
                f"got {system.scale.label}/{system.kind}"
            )


class Translator(ABC):
    """A documented, validated map from a *solved* system at `source` to an unsolved one at `target`.

    `approximations` is the scientific contract of the arrow: every assumption the translation
    makes. `validate` is where those assumptions are checked against the actual numbers.
    """

    name: str
    version: str = "0.1.0"
    source: Scale
    target: Scale
    target_kind: str                  # the model kind of the system this translator produces
    source_kinds: tuple[str, ...] = ()   # model kinds it accepts; empty means any kind at `source`
    approximations: tuple[str, ...] = ()

    def accepts(self, kind: str | None) -> bool:
        return not self.source_kinds or kind is None or kind in self.source_kinds

    def validate(self, system: ScientificSystem) -> list[ValidationIssue]:
        return []

    @abstractmethod
    def translate(self, system: ScientificSystem) -> ScientificSystem: ...


class Registry:
    def __init__(self):
        self._engines: dict[tuple[Scale, str], Engine] = {}
        self._translators: list[Translator] = []

    def register_engine(self, engine: Engine) -> None:
        for kind in engine.kinds:
            self._engines[(engine.scale, kind)] = engine

    def register_translator(self, translator: Translator) -> None:
        self._translators.append(translator)

    def engine_for(self, scale: Scale, kind: str) -> Engine:
        """The engine that solves systems of this scale and model kind."""
        try:
            return self._engines[(scale, kind)]
        except KeyError:
            have = ", ".join(f"{s.label}/{k}" for s, k in self._engines) or "none"
            raise SubstrateError(
                f"no engine registered for scale '{scale.label}' kind '{kind}' (have: {have})"
            ) from None

    def find_path(self, source: Scale, target: Scale, source_kind: str | None = None) -> list[Translator]:
        """The fewest-hop chain of translators from `source` (a system of `source_kind`) to `target`.

        Translators only apply to the model kinds they accept. If several *distinct* chains tie for
        shortest, that is a scientific choice (different physics, e.g. classical TST vs quantum
        tunnelling), so it raises AmbiguousPathError rather than picking one silently.
        """
        if source == target:
            return []
        for hops in range(1, len(Scale)):
            routes = self._routes(source, source_kind, target, hops, frozenset({source}))
            if len(routes) == 1:
                return routes[0]
            if routes:
                options = "; ".join(_describe(source, r) for r in routes)
                raise AmbiguousPathError(
                    f"{len(routes)} distinct {hops}-hop routes from '{source.label}' to '{target.label}': {options}. "
                    f"Name the intermediate scales in `propagation` to choose one."
                )
        raise NoPathError(
            f"no translator chain from '{source.label}' to '{target.label}' "
            f"(from '{source.label}' you can reach: {self._reachable(source, source_kind)})"
        )

    def _routes(self, scale, kind, target, hops, visited) -> list[list[Translator]]:
        found = []
        for t in self._translators:
            if t.source != scale or t.target in visited or not t.accepts(kind):
                continue
            if t.target == target:
                if hops == 1:
                    found.append([t])
            elif hops > 1:
                found += [[t] + rest for rest in self._routes(t.target, t.target_kind, target, hops - 1, visited | {t.target})]
        return found

    def _reachable(self, source: Scale, source_kind: str | None) -> str:
        seen, queue = {source}, deque([(source, source_kind)])
        while queue:
            scale, kind = queue.popleft()
            for t in self._translators:
                if t.source == scale and t.accepts(kind) and t.target not in seen:
                    seen.add(t.target)
                    queue.append((t.target, t.target_kind))
        return ", ".join(sorted(s.label for s in seen if s != source)) or "nothing"


def _describe(source: Scale, route: list[Translator]) -> str:
    scales = " -> ".join([source.label] + [t.target.label for t in route])
    return f"{scales} [{', '.join(t.name for t in route)}]"
