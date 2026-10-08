"""Experiment specs: a starting system, the scales to propagate through, and an optional ensemble."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .errors import ValidationError
from .ir import Quantity, Scale, ScientificSystem
from .limits import finite, whole_number
from .safeload import read_spec_file


@dataclass
class Experiment:
    id: str
    phenomenon: str
    system: ScientificSystem
    propagation: list[Scale]
    n_samples: int = 0
    seed: int = 0
    notes: dict = field(default_factory=dict)


def read_spec(path: str | Path) -> dict:
    """The experiment file as a plain dict (YAML or JSON). The file is not trusted: see `safeload` for what is refused."""
    return read_spec_file(path)


def load_experiment(path: str | Path) -> Experiment:
    return parse_experiment(read_spec(path))


def _parameter(owner: str, name: str, p) -> Quantity:
    """One declared parameter: a number (or array) with a unit, finite, with a finite non-negative sigma if it has one."""
    if not isinstance(p, dict) or "value" not in p or "unit" not in p:
        raise ValidationError(f"experiment '{owner}': parameter '{name}' needs a value and a unit")
    q = Quantity(p["value"], p["unit"], p.get("sigma"), "input")
    finite(owner, f"parameter '{name}'", q.value)
    if q.sigma is not None:
        finite(owner, f"the sigma of parameter '{name}'", q.sigma)
        if np.any(np.asarray(q.sigma, dtype=float) < 0):
            raise ValidationError(f"experiment '{owner}': the sigma of parameter '{name}' must not be negative")
    return q


def parse_experiment(raw: dict) -> Experiment:
    if not isinstance(raw, dict):
        raise ValidationError(f"an experiment must be a mapping (got {type(raw).__name__})")
    spec = raw.get("experiment", raw)
    try:
        s = spec["system"]
        system = ScientificSystem(
            name=s.get("name", spec.get("phenomenon", "system")),
            scale=Scale.parse(s["scale"]),
            kind=s["kind"],
            structure=s.get("structure", {}),             # non-numeric configuration, e.g. a molecule for a QC engine
            parameters={name: _parameter(spec.get("id", s.get("name", "system")), name, p) for name, p in s.get("parameters", {}).items()},
        )
        ensemble = spec.get("ensemble", {})
        notes = {}
        if "calibration" in spec:                         # hints for fitting the model engines to this system's surface (see transfer.fit_settings)
            from .calibration import BONDS                  # imported here: the calibration module pulls in the fitting stack
            hints = spec["calibration"]
            window = hints.get("window_ev") if isinstance(hints, dict) else None
            bad_window = window is not None and (isinstance(window, bool) or not isinstance(window, (int, float)) or not window > 0)
            bad_bonds = isinstance(hints, dict) and "bonds" in hints and hints["bonds"] not in BONDS
            if not isinstance(hints, dict) or set(hints) - {"window_ev", "bonds"} or bad_window or bad_bonds:
                raise ValidationError(f"experiment '{spec.get('id', system.name)}': `calibration` takes only a positive `window_ev` "
                                      f"and a `bonds` of {' or '.join(BONDS)} (got {hints})")
            notes["calibration"] = {k: (v if k == "bonds" else float(v)) for k, v in hints.items()}
        return Experiment(
            id=spec.get("id", system.name),
            phenomenon=spec.get("phenomenon", ""),
            system=system,
            propagation=[Scale.parse(x) for x in spec.get("propagation", [system.scale.label])],
            n_samples=whole_number(system.name, "ensemble n_samples", ensemble.get("n_samples", 0), minimum=0, maximum="max_ensemble"),
            seed=whole_number(system.name, "ensemble seed", ensemble.get("seed", 0)),
            notes=notes,
        )
    except KeyError as e:
        raise ValidationError(f"experiment spec is missing {e}") from None
