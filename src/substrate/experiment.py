"""Experiment specs: a starting system, the scales to propagate through, and an optional ensemble."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ValidationError
from .ir import Quantity, Scale, ScientificSystem


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
    """The experiment file as a plain dict (YAML or JSON)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml
        return yaml.safe_load(text)
    return json.loads(text)


def load_experiment(path: str | Path) -> Experiment:
    return parse_experiment(read_spec(path))


def parse_experiment(raw: dict) -> Experiment:
    spec = raw.get("experiment", raw)
    try:
        s = spec["system"]
        system = ScientificSystem(
            name=s.get("name", spec.get("phenomenon", "system")),
            scale=Scale.parse(s["scale"]),
            kind=s["kind"],
            structure=s.get("structure", {}),             # non-numeric configuration, e.g. a molecule for a QC engine
            parameters={
                name: Quantity(p["value"], p["unit"], p.get("sigma"), "input")
                for name, p in s.get("parameters", {}).items()
            },
        )
        ensemble = spec.get("ensemble", {})
        notes = {}
        if "calibration" in spec:                         # hints for fitting the model engines to this system's surface (see transfer.fit_settings)
            hints = spec["calibration"]
            window = hints.get("window_ev") if isinstance(hints, dict) else None
            bad_window = window is not None and (isinstance(window, bool) or not isinstance(window, (int, float)) or not window > 0)
            if not isinstance(hints, dict) or set(hints) - {"window_ev"} or bad_window:
                raise ValidationError(f"experiment '{spec.get('id', system.name)}': `calibration` takes only a positive `window_ev` (got {hints})")
            notes["calibration"] = {k: float(v) for k, v in hints.items()}
        return Experiment(
            id=spec.get("id", system.name),
            phenomenon=spec.get("phenomenon", ""),
            system=system,
            propagation=[Scale.parse(x) for x in spec.get("propagation", [system.scale.label])],
            n_samples=int(ensemble.get("n_samples", 0)),
            seed=int(ensemble.get("seed", 0)),
            notes=notes,
        )
    except KeyError as e:
        raise ValidationError(f"experiment spec is missing {e}") from None
