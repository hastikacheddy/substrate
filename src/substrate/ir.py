"""Scientific Intermediate Representation (SIR).

Every model, at every scale, is a ScientificSystem. Engines solve systems, translators
map a solved system at one scale to an unsolved system at another. Nothing else crosses
a scale boundary.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Any

import numpy as np

from .errors import ValidationError

_MISSING = object()

#: Parameters named `context.<something>` belong to the experiment rather than to one scale (a substrate
#: concentration, a binding rate). The pipeline carries them across every translator unchanged, so a scale far
#: up the chain can use them without every scale in between having to know they exist.
CONTEXT_PREFIX = "context."


class Scale(IntEnum):
    QUANTUM = 0
    ELECTRONIC_STRUCTURE = 1
    MOLECULAR = 2
    REACTION = 3
    BIOPHYSICAL = 4
    BIOLOGICAL = 5

    @classmethod
    def parse(cls, value: "str | int | Scale") -> "Scale":
        if isinstance(value, str):
            try:
                return cls[value.strip().upper()]
            except KeyError:
                names = ", ".join(s.name.lower() for s in cls)
                raise ValidationError(f"unknown scale '{value}' (known: {names})") from None
        return cls(value)

    @property
    def label(self) -> str:
        return self.name.lower()


@dataclass(eq=False)
class Quantity:
    """A number or array with a unit, an optional 1-sigma uncertainty, and where it came from.

    `band` is the (16th, 84th) percentile of an ensemble, set only for scalars. It exists because a
    standard deviation misleads for exponentially sensitive quantities such as rate constants, where the
    spread can exceed the value itself.
    """

    value: Any
    unit: str
    sigma: Any = None
    source: str = ""
    band: tuple[float, float] | None = None

    def __post_init__(self):
        self.value = _as_numeric(self.value)
        if self.sigma is not None:
            self.sigma = _as_numeric(self.sigma)

    def to_dict(self) -> dict:
        d = {"value": _jsonable(self.value), "unit": self.unit}
        if self.sigma is not None:
            d["sigma"] = _jsonable(self.sigma)
        if self.source:
            d["source"] = self.source
        if self.band is not None:
            d["band"] = list(self.band)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Quantity":
        band = d.get("band")
        return cls(d["value"], d["unit"], d.get("sigma"), d.get("source", ""), tuple(band) if band else None)


@dataclass
class ProvenanceRecord:
    """One link in a system's lineage: what produced it, from what, under which approximations."""

    step: str                      # "solve" | "translate" | "uncertainty"
    name: str
    version: str
    scale_in: str
    scale_out: str
    input_fingerprint: str
    output_fingerprint: str
    backend: str | None = None
    approximations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class ScientificSystem:
    name: str
    scale: Scale
    kind: str                                   # which model family; engines claim kinds
    parameters: dict[str, Quantity] = field(default_factory=dict)
    state: dict[str, Quantity] = field(default_factory=dict)
    observables: dict[str, Quantity] = field(default_factory=dict)
    structure: dict[str, Any] = field(default_factory=dict)   # non-numeric model definition (species, reactions, ...)
    dynamics: str = ""                          # human-readable statement of the governing equation
    provenance: list[ProvenanceRecord] = field(default_factory=list)

    # -- access ---------------------------------------------------------------
    def param(self, name: str, unit: str | None = None, default: Any = _MISSING):
        return self._get(self.parameters, "parameter", name, unit, default)

    def obs(self, name: str, unit: str | None = None, default: Any = _MISSING):
        return self._get(self.observables, "observable", name, unit, default)

    def _get(self, table: dict, what: str, name: str, unit: str | None, default: Any):
        q = table.get(name)
        if q is None:
            if default is _MISSING:
                raise ValidationError(f"{self.name}: missing {what} '{name}'")
            return default
        if unit is not None and q.unit != unit:
            raise ValidationError(f"{self.name}: {what} '{name}' is in '{q.unit}', expected '{unit}'")
        return q.value

    def uncertainty(self) -> dict[str, Any]:
        """All attached 1-sigma uncertainties, keyed 'parameters.<name>' / 'observables.<name>'."""
        out = {}
        for table_name in ("parameters", "state", "observables"):
            for name, q in getattr(self, table_name).items():
                if q.sigma is not None:
                    out[f"{table_name}.{name}"] = q.sigma
        return out

    # -- identity ---------------------------------------------------------------
    def fingerprint(self) -> str:
        """Content hash (provenance excluded). Numbers are rounded to ~12 significant digits first so
        the hash is stable against last-bit float noise; it identifies lineage, not bitwise equality."""
        payload = {
            "name": self.name,
            "scale": self.scale.label,
            "kind": self.kind,
            "dynamics": self.dynamics,
            "structure": _canon(self.structure),
            **{
                table: {k: _canon_quantity(q) for k, q in getattr(self, table).items()}
                for table in ("parameters", "state", "observables")
            },
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def evolve(self, **changes) -> "ScientificSystem":
        """Copy with fresh containers so the original is never mutated by later steps."""
        copies = dict(
            parameters=dict(self.parameters),
            state=dict(self.state),
            observables=dict(self.observables),
            structure=json.loads(json.dumps(self.structure)),
            provenance=list(self.provenance),
        )
        return replace(self, **{**copies, **changes})

    # -- serialisation ------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "scale": self.scale.label,
            "kind": self.kind,
            "dynamics": self.dynamics,
            "parameters": {k: q.to_dict() for k, q in self.parameters.items()},
            "state": {k: q.to_dict() for k, q in self.state.items()},
            "observables": {k: q.to_dict() for k, q in self.observables.items()},
            "structure": self.structure,
            "provenance": [p.to_dict() for p in self.provenance],
        }

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)

    @classmethod
    def from_dict(cls, d: dict) -> "ScientificSystem":
        return cls(
            name=d["name"],
            scale=Scale.parse(d["scale"]),
            kind=d["kind"],
            dynamics=d.get("dynamics", ""),
            parameters={k: Quantity.from_dict(v) for k, v in d.get("parameters", {}).items()},
            state={k: Quantity.from_dict(v) for k, v in d.get("state", {}).items()},
            observables={k: Quantity.from_dict(v) for k, v in d.get("observables", {}).items()},
            structure=d.get("structure", {}),
            provenance=[ProvenanceRecord(**p) for p in d.get("provenance", [])],
        )

    @classmethod
    def from_json(cls, text: str) -> "ScientificSystem":
        return cls.from_dict(json.loads(text))


def _as_numeric(v):
    if isinstance(v, np.ndarray):
        return v.item() if v.ndim == 0 else v
    if isinstance(v, (list, tuple)):
        return np.asarray(v, dtype=float)
    return float(v)


def _jsonable(v):
    return v.tolist() if isinstance(v, np.ndarray) else v


def _canon_quantity(q: Quantity) -> dict:
    return {"value": _canon(q.value), "unit": q.unit, "sigma": _canon(q.sigma), "source": q.source,
            "band": _canon(q.band)}


def _round_mantissa(a: np.ndarray) -> np.ndarray:
    """Round to 40 mantissa bits (~12 significant digits), vectorised."""
    m, e = np.frexp(np.ascontiguousarray(a, dtype=np.float64))
    return np.ldexp(np.round(m * 2.0**40), e - 40)


def _canon(obj):
    if isinstance(obj, np.ndarray):
        return {"shape": list(obj.shape), "sha": hashlib.sha256(_round_mantissa(obj).tobytes()).hexdigest()}
    if isinstance(obj, float):
        return float(f"{obj:.12g}")
    if isinstance(obj, dict):
        return {k: _canon(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canon(v) for v in obj]
    return obj
