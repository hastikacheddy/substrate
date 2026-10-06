"""Turning solved systems into JSON for the GUI, and deciding what is worth plotting at each scale.

The server decides *what* to show (which arrays make a potential curve, which are a time course, which numbers are control
coefficients); the page only knows how to draw three generic plot specs:

    {"type": "line",    "title", "xlabel", "ylabel", "x": [...], "series": [{"name", "y": [...], "dashed"}], "hlines": [{"y", "label"}]}
    {"type": "heatmap", "title", "xlabel", "ylabel", "zlabel", "x": [...], "y": [...], "z": [[...]]}      # z[ix][iy]
    {"type": "bars",    "title", "ylabel", "labels": [...], "values": [...]}

Nothing here computes science: every number shown was produced by an engine or translator.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..ir import ProvenanceRecord, Quantity, ScientificSystem

DIGITS = 6                                  # decimals kept in plotted arrays: far below any physical resolution here, and keeps payloads small


# -- plain JSON ----------------------------------------------------------------------------------------------------------------------
def clean(obj: Any) -> Any:
    """Recursively convert numpy types to Python ones and non-finite floats to None (JSON has no NaN)."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    return obj


def rounded(values, digits: int = DIGITS):
    """A numpy array as nested lists, rounded, with non-finite entries as None."""
    return clean(np.round(np.asarray(values, dtype=float), digits))


# -- quantities ----------------------------------------------------------------------------------------------------------------------
def quantity(q: Quantity) -> dict:
    """A scalar keeps its value, sigma and 68% band; an array is summarised (the plots carry the data)."""
    v = q.value
    if isinstance(v, np.ndarray) and v.ndim > 0:
        finite = v[np.isfinite(v)] if np.issubdtype(v.dtype, np.number) else np.array([])
        out = {"unit": q.unit, "array": {"shape": list(v.shape), "min": float(finite.min()) if finite.size else None,
                                         "max": float(finite.max()) if finite.size else None}}
        if v.size <= 4:
            out["array"]["values"] = clean(v.ravel().tolist())
        return clean(out)
    out = {"unit": q.unit, "value": clean(v if not isinstance(v, np.ndarray) else v.item())}
    if q.sigma is not None and np.ndim(q.sigma) == 0:
        out["sigma"] = clean(q.sigma)
    if q.band is not None:
        out["band"] = clean(list(q.band))
    if q.source:
        out["source"] = q.source
    return out


# -- provenance ----------------------------------------------------------------------------------------------------------------------
def record(rec: ProvenanceRecord) -> dict:
    return {"component": rec.name, "step": rec.step, "version": rec.version, "seconds": rec.seconds, "backend": rec.backend,
            "approximations": list(rec.approximations), "warnings": list(rec.warnings),
            "input": rec.input_fingerprint, "output": rec.output_fingerprint}


def _last_record(system: ScientificSystem) -> ProvenanceRecord | None:
    for rec in reversed(system.provenance):
        if rec.step != "uncertainty":
            return rec
    return None


def ensemble_warnings(system: ScientificSystem) -> list[str]:
    return [w for rec in system.provenance if rec.step == "uncertainty" for w in rec.warnings]


# -- the stages of a run ---------------------------------------------------------------------------------------------------------------
def stages(steps, trace) -> list[dict]:
    """One stage per scale: the translation into it (if any), its solve, and the solved system with its plots."""
    out, pending = [], None
    for step, system in zip(steps, trace):
        rec = _last_record(system)
        info = record(rec) if rec else {"component": step.component.name, "step": step.kind}
        if step.kind == "translate":
            pending = info
            continue
        out.append({"index": len(out), "scale": step.scale_out.label, "translate": pending, "solve": info,
                    "system": describe(system)})
        pending = None
    return out


def describe(system: ScientificSystem) -> dict:
    scalars = {n: quantity(q) for n, q in system.observables.items()}
    return {
        "name": system.name, "scale": system.scale.label, "kind": system.kind, "fingerprint": system.fingerprint(),
        "dynamics": system.dynamics,
        "parameters": {n: quantity(q) for n, q in system.parameters.items()},
        "observables": scalars,
        "plots": plots_for(system),
        "uncertainty_notes": ensemble_warnings(system),
    }


# -- plots ------------------------------------------------------------------------------------------------------------------------------------
def plots_for(system: ScientificSystem) -> list[dict]:
    if not system.observables:
        return []
    kind, obs = system.kind, system.observables
    plots: list[dict] = []
    if kind.startswith("electronic."):
        plots += _electronic(system)
    elif kind.startswith("quantum."):
        plots += _quantum(system)
    elif kind.startswith("molecular."):
        plots += _molecular(system)
    elif kind.startswith("reaction."):
        plots += _time_course(system, "Species over time", "population")
    elif kind.startswith("biophysical."):
        plots += _enzyme(system)
    elif kind.startswith("biological."):
        plots += _time_course(system, "Pathway over time", "concentration")
        plots += _control(system, "flux_control_", "Control of the pathway flux")
    return [p for p in plots if p]


def _arr(system, name, table="observables"):
    q = getattr(system, table).get(name)
    return None if q is None or not isinstance(q.value, np.ndarray) else q.value


def _electronic(system) -> list[dict]:
    plots = []
    x, r, e = _arr(system, "surface_x"), _arr(system, "surface_r"), _arr(system, "surface_energy")
    if x is not None and r is not None and e is not None and e.shape == (len(x), len(r)):
        plots.append({"type": "heatmap", "title": "Ground-state surface E(x, R)", "xlabel": "proton position x (Å)",
                      "ylabel": "heavy-atom distance R (Å)", "zlabel": "energy above the minimum (eV)",
                      "x": rounded(x), "y": rounded(r), "z": rounded(e - np.nanmin(e), 4), "clip": 3.0})
    coordinate, energy = _arr(system, "scan_coordinate"), _arr(system, "scan_energy")
    if coordinate is not None and energy is not None:
        series = [{"name": "ground state", "y": rounded(energy, 5)}]
        for name, label in (("diabatic_a", "reactant diabat"), ("diabatic_b", "product diabat"), ("excited_state_energy", "excited state")):
            values = _arr(system, name)
            if values is not None and len(values) == len(coordinate):
                series.append({"name": label, "y": rounded(values, 5), "dashed": True})
        top = float(np.nanmin(energy)) + 3.0
        plots.append({"type": "line", "title": "Potential along the proton coordinate", "xlabel": "proton position (Å)",
                      "ylabel": "energy (eV)", "x": rounded(coordinate), "series": series, "ymax": top})
    return plots


def _quantum(system) -> list[dict]:
    x, v = _arr(system, "pes_x", "parameters"), _arr(system, "pes_energy", "parameters")
    if (x is None or v is None) and system.kind == "quantum.double_well_1d":
        from ..engines.quantum import double_well                       # the analytic well: the same function the engine solves on
        p = system.parameters
        a = float(p["half_separation"].value)
        x = np.linspace(-2.0 * a, 2.0 * a, 241)
        v = double_well(x, float(p["barrier_height"].value), a, float(p["reaction_energy"].value))
    if x is None or v is None:
        return []
    hlines = []
    levels = _arr(system, "energy_levels")
    if levels is not None:
        top = float(np.nanmax(v[: max(len(v) // 2, 1)]))                       # label only the levels below the walls
        hlines = [{"y": float(en), "label": f"E{i}"} for i, en in enumerate(levels) if float(en) < top + 0.5][:8]
    return [{"type": "line", "title": "Potential and proton energy levels", "xlabel": "proton position (Å)", "ylabel": "energy (eV)",
             "x": rounded(x), "series": [{"name": "potential", "y": rounded(v, 5)}], "hlines": hlines,
             "ymax": float(np.nanmin(v)) + 1.6}]


def _molecular(system) -> list[dict]:
    path_x, path_e, path_r = _arr(system, "mep_x"), _arr(system, "mep_energy"), _arr(system, "mep_r")
    plots = []
    if path_x is not None and path_e is not None:
        plots.append({"type": "line", "title": "Relaxed energy profile (heavy atoms follow the proton)", "xlabel": "proton position (Å)",
                      "ylabel": "energy (eV)", "x": rounded(path_x), "series": [{"name": "relaxed profile", "y": rounded(path_e, 5)}]})
    if path_x is not None and path_r is not None:
        plots.append({"type": "line", "title": "Heavy-atom distance along the path", "xlabel": "proton position (Å)", "ylabel": "R (Å)",
                      "x": rounded(path_x), "series": [{"name": "R", "y": rounded(path_r, 5)}]})
    return plots


def _time_course(system, title: str, unit_word: str) -> list[dict]:
    t, c = _arr(system, "time"), _arr(system, "concentrations")
    if t is None or c is None or c.ndim != 2 or c.shape[0] != len(t):
        return []
    species = system.structure.get("species") or [f"species {i}" for i in range(c.shape[1])]
    names = [s if isinstance(s, str) else str(s.get("id", i)) if isinstance(s, dict) else str(s) for i, s in enumerate(species)]
    unit = system.observables["concentrations"].unit
    return [{"type": "line", "title": title, "xlabel": f"time ({system.observables['time'].unit})", "ylabel": f"{unit_word} ({unit})",
             "x": rounded(t, 9), "series": [{"name": names[i] if i < len(names) else f"species {i}", "y": rounded(c[:, i], 9)}
                                            for i in range(c.shape[1])]}]


def _control(system, prefix: str, title: str) -> list[dict]:
    items = [(n[len(prefix):], q.value) for n, q in system.observables.items()
             if n.startswith(prefix) and not isinstance(q.value, np.ndarray)]
    if not items:
        return []
    return [{"type": "bars", "title": title + " (coefficients sum to 1)", "ylabel": "control coefficient",
             "labels": [n for n, _ in items], "values": clean([float(v) for _, v in items])}]


def _enzyme(system) -> list[dict]:
    plots = []
    pops = [(n[len("population_"):], q.value) for n, q in system.observables.items() if n.startswith("population_")]
    if pops:
        plots.append({"type": "bars", "title": "Enzyme state populations at steady state", "ylabel": "fraction of enzyme",
                      "labels": [n for n, _ in pops], "values": clean([float(v) for _, v in pops])})
    plots += _control(system, "control_", "Control of the turnover rate")
    return plots
