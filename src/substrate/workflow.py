"""A minimal DAG executor: named tasks, explicit dependencies, topological order."""
from __future__ import annotations

from dataclasses import dataclass
from graphlib import TopologicalSorter
from typing import Any, Callable


@dataclass(frozen=True)
class Task:
    name: str
    fn: Callable[..., Any]
    deps: tuple[str, ...] = ()


class Workflow:
    def __init__(self):
        self._tasks: dict[str, Task] = {}

    def add(self, name: str, fn: Callable[..., Any], deps: tuple[str, ...] = ()) -> None:
        if name in self._tasks:
            raise ValueError(f"duplicate task '{name}'")
        missing = [d for d in deps if d not in self._tasks]
        if missing:
            raise ValueError(f"task '{name}' depends on unknown task(s): {', '.join(missing)}")
        self._tasks[name] = Task(name, fn, tuple(deps))

    def run(self) -> dict[str, Any]:
        """Run every task once, dependencies first; each task receives its dependencies' results positionally."""
        order = TopologicalSorter({t.name: set(t.deps) for t in self._tasks.values()}).static_order()
        results: dict[str, Any] = {}
        for name in order:
            task = self._tasks[name]
            results[name] = task.fn(*(results[d] for d in task.deps))
        return results
