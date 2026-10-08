"""What the GUI does, without any HTTP: list and run experiments, run long computations as background jobs, and assemble the
transfer study's results. `server.py` only routes requests to these methods, so everything here can be tested directly.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
import traceback
import uuid
from pathlib import Path

import numpy as np

from .. import __version__ as VERSION
from ..calibration import FitSettings
from ..defaults import default_registry
from ..errors import SubstrateError, ValidationError
from ..experiment import load_experiment, read_spec
from ..ir import Quantity
from ..pipeline import Pipeline
from ..transfer import (
    Reference, _column_comparison, baselines, fit_settings, learning_curve, make_reference, predict_surface, rate_comparison, transfer_matrix,
)
from . import serialize as ser

TRANSFER_REFERENCE_DISTANCE = 2.7        # where every transfer fit defines its coupling: inside every reference's distance range
TRANSFER_VERSION = 4                     # bump to invalidate saved transfer results when the analysis changes
MAX_JOBS = 60
MAX_REFERENCES = 24                       # a study compares every reference with every other, so its cost grows with the square


class GuiError(Exception):
    """A request the GUI refuses: carries the HTTP status to answer with."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


# =====================================================================================================================
# background jobs
# =====================================================================================================================
class Job:
    def __init__(self, label: str):
        self.id = uuid.uuid4().hex[:12]
        self.label = label
        self.status = "queued"                  # queued | running | done | error
        self.messages: list[dict] = []
        self.result = None
        self.error: dict | None = None
        self.started = time.time()
        self.finished: float | None = None
        self._lock = threading.Lock()

    def log(self, text: str) -> None:
        with self._lock:
            self.messages.append({"t": round(time.time() - self.started, 1), "text": text})

    def to_dict(self, since: int = 0) -> dict:
        with self._lock:
            return {"id": self.id, "label": self.label, "status": self.status, "messages": self.messages[since:], "n_messages": len(self.messages),
                    "elapsed": round((self.finished or time.time()) - self.started, 1), "error": self.error,
                    "result": self.result if self.status == "done" else None}


class JobManager:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, label: str, fn, *, wait: bool = False) -> Job:
        job = Job(label)
        with self._lock:
            self.jobs[job.id] = job
            for old in list(self.jobs)[:-MAX_JOBS]:
                self.jobs.pop(old, None)

        def target():
            job.status = "running"
            try:
                job.result = ser.clean(fn(job.log))
                job.status = "done"
            except GuiError as error:
                job.error, job.status = {"type": "GuiError", "message": str(error)}, "error"
            except Exception as error:          # an environment failure (a missing program, a bug): reported, never swallowed
                job.error = {"type": type(error).__name__, "message": str(error)[:600], "trace": traceback.format_exc()[-1500:]}
                job.status = "error"
            finally:
                job.finished = time.time()

        thread = threading.Thread(target=target, daemon=True, name=f"job-{job.id}")
        thread.start()
        if wait:
            thread.join()
        return job

    def get(self, job_id: str) -> Job:
        try:
            return self.jobs[job_id]
        except KeyError:
            raise GuiError(f"no job '{job_id}'", 404) from None


# =====================================================================================================================
# the application
# =====================================================================================================================
def default_root() -> Path:
    """The project folder holding `experiments/`: the package's own checkout, else the current directory."""
    here = Path(__file__).resolve().parents[3]
    return here if (here / "experiments").is_dir() else Path.cwd()


class App:
    def __init__(self, root: str | Path | None = None, cache_dir: str | Path | None = None):
        self.root = Path(root).resolve() if root else default_root()
        self.experiments_dir = self.root / "experiments"
        base = Path(cache_dir) if cache_dir else Path(os.environ.get("SUBSTRATE_CACHE_DIR") or Path.home() / ".cache" / "substrate")
        self.cache_dir = base / "gui"
        self.registry = default_registry()
        self.pipeline = Pipeline(self.registry)
        self.jobs = JobManager()
        self._pyscf: bool | None = None
        self._references: dict[str, Reference] = {}
        self._reference_lock = threading.Lock()

    # -- status ---------------------------------------------------------------------------------------------------------------
    def pyscf_available(self) -> bool:
        if self._pyscf is None:
            from ..qc import PySCFProgram
            try:
                self._pyscf = bool(PySCFProgram().available())
            except Exception:
                self._pyscf = False
        return self._pyscf

    def status(self) -> dict:
        return {"version": VERSION, "root": str(self.root), "experiments": len(self.list_experiments()), "pyscf": self.pyscf_available()}

    # -- experiments ----------------------------------------------------------------------------------------------------------
    def _resolve(self, rel: str) -> Path:
        if not isinstance(rel, str) or not rel:
            raise GuiError("an experiment path is required")
        path = (self.experiments_dir / rel).resolve()
        try:
            path.relative_to(self.experiments_dir.resolve())
        except ValueError:
            raise GuiError("that path is outside the experiments folder", 403) from None
        if path.suffix.lower() not in (".yaml", ".yml", ".json"):
            raise GuiError("an experiment file is a .yaml or .json file")
        if not path.is_file():
            raise GuiError(f"no experiment '{rel}'", 404)
        return path

    def list_experiments(self) -> list[dict]:
        out = []
        for path in sorted(self.experiments_dir.rglob("*")):
            if path.suffix.lower() not in (".yaml", ".yml", ".json") or not path.is_file():
                continue
            rel = path.relative_to(self.experiments_dir).as_posix()
            try:
                spec = read_spec(path)
                spec = spec.get("experiment", spec)
                system = spec["system"]
                kind = system["kind"]
                out.append({"path": rel, "group": path.parent.relative_to(self.experiments_dir).as_posix() if path.parent != self.experiments_dir else "",
                            "id": spec.get("id", path.stem), "phenomenon": spec.get("phenomenon", ""), "name": system.get("name", path.stem),
                            "scale": system["scale"], "kind": kind, "propagation": list(spec.get("propagation", [system["scale"]])),
                            "needs_qc": kind.startswith("electronic.qc"), "summary": _header(path).split("\n")[0] if _header(path) else ""})
            except Exception as error:         # a malformed file is listed as such, not hidden
                out.append({"path": rel, "group": "", "id": path.stem, "invalid": f"{type(error).__name__}: {error}"})
        return out

    def experiment(self, rel: str) -> dict:
        path = self._resolve(rel)
        try:
            exp = load_experiment(path)
        except SubstrateError as error:
            raise GuiError(str(error)) from None
        parameters = {n: ser.quantity(q) for n, q in exp.system.parameters.items() if not isinstance(q.value, np.ndarray)}
        return {"path": rel, "id": exp.id, "phenomenon": exp.phenomenon, "name": exp.system.name, "scale": exp.system.scale.label,
                "kind": exp.system.kind, "propagation": [s.label for s in exp.propagation], "samples": exp.n_samples, "seed": exp.seed,
                "header": _header(path), "parameters": parameters, "needs_qc": exp.system.kind.startswith("electronic.qc"),
                "structure": ser.clean(_brief_structure(exp.system.structure))}

    # -- running --------------------------------------------------------------------------------------------------------------
    def run(self, rel: str, overrides: dict | None = None, samples: int | None = None, seed: int | None = None, log=None) -> dict:
        """Run an experiment (optionally with edited parameters and an ensemble) and return everything the page shows. A refusal by
        the physics (a validation error) is a normal result with `ok: false`, the stages that did complete and the one that refused."""
        path = self._resolve(rel)
        try:
            exp = load_experiment(path)
            self._apply_overrides(exp.system, overrides or {})
        except SubstrateError as error:
            raise GuiError(str(error)) from None
        n = exp.n_samples if samples is None else int(samples)
        if not 0 <= n <= 2000:
            raise GuiError("the ensemble size must be between 0 and 2000")
        seed = exp.seed if seed is None else int(seed)
        if log:
            log(f"running {exp.id} through {' → '.join(s.label for s in exp.propagation)}" + (f" with {n} ensemble draws" if n else ""))
        started = time.perf_counter()
        meta = {"path": rel, "overrides": ser.clean(overrides or {}), "samples": n, "seed": seed, "input_fingerprint": exp.system.fingerprint()}
        try:
            plan = self.pipeline.plan(exp.system, exp.propagation)
        except SubstrateError as error:
            return self._refusal(exp, [], error, started, [], meta)
        plan_info = [{"scale": s.scale_out.label, "kind": s.kind, "component": s.component.name} for s in plan]
        try:
            result = self.pipeline.run(exp.system, exp.propagation, n_samples=n, seed=seed, experiment_id=exp.id)
        except SubstrateError as error:
            return self._refusal(exp, plan, error, started, plan_info, meta)
        return {"ok": True, "experiment": exp.id, "seconds": round(time.perf_counter() - started, 3), **meta, "plan": plan_info,
                "stages": ser.stages(result.steps, result.trace), "ensemble": ser.clean(result.ensemble) if result.ensemble else None,
                "warnings": result.warnings()}

    def _refusal(self, exp, plan, error, started, plan_info, meta) -> dict:
        """Re-run the chain step by step to find which stage refused, keeping every stage that completed."""
        completed, trace, failed, system = [], [], None, exp.system
        for index, step in enumerate(plan):
            try:
                system = self.pipeline._apply(step, system)
            except SubstrateError as step_error:
                failed = {"index": index, "scale": step.scale_out.label, "component": step.component.name, "kind": step.kind,
                          "message": str(step_error)}
                break
            completed.append(step)
            trace.append(system)
        return {"ok": False, "experiment": exp.id, "seconds": round(time.perf_counter() - started, 3), **meta,
                "error": {"type": type(error).__name__, "message": str(error)}, "failed": failed, "plan": plan_info,
                "stages": ser.stages(completed, trace), "warnings": [], "ensemble": None}

    @staticmethod
    def _apply_overrides(system, overrides: dict) -> None:
        for name, change in overrides.items():
            if name not in system.parameters:
                raise ValidationError(f"no parameter '{name}' to change (have: {', '.join(system.parameters)})")
            old = system.parameters[name]
            if isinstance(old.value, np.ndarray):
                raise ValidationError(f"'{name}' is an array and cannot be edited here")
            change = change if isinstance(change, dict) else {"value": change}
            value = change.get("value", old.value)
            sigma = change["sigma"] if "sigma" in change else old.sigma
            for label, number in (("value", value), ("sigma", sigma)):
                if number is not None and (isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number)):
                    raise ValidationError(f"the {label} of '{name}' must be a finite number")
            if sigma is not None and sigma < 0:
                raise ValidationError(f"the sigma of '{name}' must not be negative")
            system.parameters[name] = Quantity(float(value), old.unit, None if sigma is None else float(sigma), "input")

    def start_run(self, rel: str, overrides=None, samples=None, seed=None) -> dict:
        self._resolve(rel)                                              # refuse a bad path at once, not inside the job
        job = self.jobs.submit(f"run {rel}", lambda log: self.run(rel, overrides, samples, seed, log))
        return job.to_dict()

    # -- the transfer study ---------------------------------------------------------------------------------------------------
    def references_listing(self) -> list[dict]:
        out = []
        for path in sorted((self.experiments_dir / "references").glob("*.yaml")):
            try:
                spec = read_spec(path)
                spec = spec.get("experiment", spec)
                s = spec["system"]
                molecule, method, p = s.get("structure", {}).get("molecule", {}), s.get("structure", {}).get("method", {}), s.get("parameters", {})
                out.append({"name": path.stem, "molecule": molecule.get("template", "custom"),
                            "method": f"{method.get('theory', '?')}/{method.get('basis', '?')}",
                            "range": [p["distance_min"]["value"], p["distance_max"]["value"]],
                            "grid": [int(p.get("n_x", {}).get("value", 0)), int(p.get("n_r", {}).get("value", 0))]})
            except Exception as error:
                out.append({"name": path.stem, "invalid": f"{type(error).__name__}: {error}"})
        return out

    def _reference_path(self, name: str) -> Path:
        if not isinstance(name, str) or not name.replace("_", "").isalnum():
            raise GuiError(f"'{name}' is not a reference name")
        return self._resolve(f"references/{name}.yaml")

    def transfer_key(self, names: list[str]) -> str:
        digest = hashlib.sha256(json.dumps({"v": TRANSFER_VERSION, "names": sorted(names), "r": TRANSFER_REFERENCE_DISTANCE,
                                            "files": [self._reference_path(n).read_bytes().hex() for n in sorted(names)]}).encode()).hexdigest()
        return digest[:16]

    def cached_transfer(self, names: list[str]) -> dict | None:
        path = self.cache_dir / f"transfer-{self.transfer_key(names)}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def reference(self, name: str, log=None) -> Reference:
        """A calibrated reference, computed once per server: its energies come from the QC cache, its calibration is a few seconds."""
        with self._reference_lock:
            if name in self._references:
                return self._references[name]
            path = self._reference_path(name)
            experiment = load_experiment(path)
            if experiment.system.kind.startswith("electronic.qc") and not self.pyscf_available():
                raise GuiError("these references are real quantum-chemistry surfaces and PySCF is not available here "
                               "(on Windows run scripts/setup_qc_env.sh in WSL); energies already in the on-disk cache are used when it is", 503)
            started = time.perf_counter()
            if log:
                log(f"{name}: reading the scan (energies come from the on-disk cache where they exist)")
            solved = self.pipeline.run(experiment.system, experiment.propagation).final
            if log:
                run, cached = solved.obs("calculations_run", default=None), solved.obs("calculations_cached", default=None)
                log(f"{name}: " + (f"{run:.0f} computed, {cached:.0f} cached" if run is not None else "a model surface")
                    + f" ({time.perf_counter() - started:.0f} s); fitting the model")
            ref = make_reference(name, experiment.system, solved, fit_settings(experiment, FitSettings(reference_distance=TRANSFER_REFERENCE_DISTANCE)))
            self._references[name] = ref
            return ref

    def run_transfer(self, names: list[str], recompute: bool = False, log=None) -> dict:
        names = self._check_names(names)
        if not recompute:
            cached = self.cached_transfer(names)
            if cached is not None:
                if log:
                    log("loaded the saved result (use Recompute to redo it)")
                return cached
        started = time.perf_counter()
        refs = {n: self.reference(n, log) for n in names}
        if log:
            log("comparing every calibration with every reference")
        payload = transfer_payload(refs)
        payload["seconds"] = round(time.perf_counter() - started, 1)
        payload["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = ser.clean(payload)
        (self.cache_dir / f"transfer-{self.transfer_key(names)}.json").write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def _check_names(self, names) -> list[str]:
        if not isinstance(names, list) or len(names) < 2 or len(names) > MAX_REFERENCES:
            raise GuiError(f"choose between 2 and {MAX_REFERENCES} references")
        if len(set(names)) != len(names):
            raise GuiError("a reference was chosen twice")
        for n in names:
            self._reference_path(n)
        return list(names)

    def run_rates(self, names: list[str], barrier: float, log=None) -> dict:
        names = self._check_names(names)
        if not isinstance(barrier, (int, float)) or not 0.02 <= barrier <= 1.5:
            raise GuiError("the barrier height must be between 0.02 and 1.5 eV")
        refs = {n: self.reference(n, log) for n in names}
        if log:
            log(f"running the real chain at the distance where each reference's barrier is {barrier:g} eV (new energies are computed if they are not cached)")
        rows = rate_comparison(refs, refs, self.pipeline, barriers=(float(barrier),), progress=log)
        cells, real = {}, {}
        for row in rows:
            if row.source == "-":
                real[row.target] = {"note": row.note}
                continue
            cells[f"{row.source}|{row.target}"] = {"ratio": row.ratio, "k_model": row.k_model, "kie_model": row.kie_model, "note": row.note}
            real[row.target] = {"distance": row.distance, "k": row.k_real, "kie": row.kie_real, "note": real.get(row.target, {}).get("note", "")}
        return ser.clean({"names": names, "barrier": barrier, "cells": cells, "real": real})

    def run_learning(self, names: list[str], target: str, source: str | None, sigmas: list[float], counts: list[int], log=None) -> dict:
        names = self._check_names(names)
        if target not in names or (source is not None and source not in names):
            raise GuiError("the target and source must be among the chosen references")
        if not sigmas or any(not isinstance(s, (int, float)) or not 0.01 <= s <= 100 for s in sigmas):
            raise GuiError("prior widths must be between 0.01 and 100 (relative to each parameter)")
        if not counts or any(not isinstance(k, int) or not 0 <= k <= 6 for k in counts):
            raise GuiError("the numbers of training distances must be between 0 and 6")
        t = self.reference(target, log)
        settings = FitSettings(reference_distance=TRANSFER_REFERENCE_DISTANCE, window_ev=t.window_ev)       # the target's own fit window
        s = self.reference(source, log) if source else None
        if log:
            log("fitting from scratch and with the prior")
        scratch = learning_curve(t, counts=[k for k in counts if k > 0], settings=settings)
        out = {"target": target, "source": source, "counts": sorted(counts),
               "scratch": [_curve_point(p) for p in scratch], "priors": {}}
        if s is not None:
            for sigma in sigmas:
                curve = learning_curve(t, counts=sorted(counts), source=s.calibration, relative_sigma=float(sigma), settings=settings)
                out["priors"][f"{sigma:g}"] = [_curve_point(p) for p in curve]
        out["floor"] = learning_curve(t, counts=(6,), settings=settings)[0].rmse_ev
        return ser.clean(out)

    def start_job(self, label: str, fn) -> dict:
        return self.jobs.submit(label, fn).to_dict()


# =====================================================================================================================
# helpers
# =====================================================================================================================
def _header(path: Path) -> str:
    """The leading comment block of an experiment file, as plain text."""
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            lines.append(line.lstrip("#").strip())
        elif line.strip():
            break
    return "\n".join(lines).strip()


def _brief_structure(structure: dict) -> dict:
    """Non-numeric configuration, trimmed so a long atom list does not swamp the page."""
    out = {}
    for key, value in structure.items():
        if key == "molecule" and isinstance(value, dict):
            atoms = value.get("atoms")
            out[key] = {**{k: v for k, v in value.items() if k != "atoms"}, **({"atoms": len(atoms)} if atoms else {})}
        elif key == "parameter_covariance":
            out[key] = {"names": value.get("names")}
        else:
            out[key] = value
    return out


def _curve_point(point) -> dict:
    return {"k": point.n_distances, "rmse": point.rmse_ev, "barrier_error": point.barrier_error_mean_abs, "wells_wrong": point.wells_wrong,
            "note": point.note}


def transfer_payload(refs: dict[str, Reference], window_ev: float | None = None) -> dict:
    """Everything the transfer page draws, for every reference and every pair. Plain data; no engine objects. Each target is scored over its
    own fit window unless `window_ev` overrides it."""
    names = list(refs)
    matrix = transfer_matrix(refs, window_ev)
    grid = lambda f: [[f(matrix[(s, t)]) for t in names] for s in names]            # [source][target]
    items = []
    for name, ref in refs.items():
        cal = ref.calibration
        columns = _column_comparison(ref.x, ref.e, ref.e, ref.r)
        items.append({
            "name": name, "molecule": ref.molecule, "method": ref.method, "symmetric": ref.symmetric,
            "x": ser.rounded(ref.x), "r": ser.rounded(ref.r), "e": ser.rounded(ref.e, 4),
            "barrier": [c.barrier_reference for c in columns], "wells": [c.wells_reference for c in columns],
            "window": ref.window_ev if window_ev is None else window_ev, "baselines": baselines(ref, window_ev),
            "calibration": {"bonds": cal.bonds, "parameters": cal.parameters, "sigma": cal.sigma, "pinned": cal.pinned, "poorly_determined": cal.poorly_determined,
                            "shift": cal.shift_ev, "reduced_chi2": cal.reduced_chi2, "n_points": cal.n_points, "n_free": cal.n_free,
                            "starts_agreeing": cal.starts_agreeing, "starts": cal.settings["starts"], "rmse": cal.rmse_ev,
                            "max_error": cal.max_error_ev, "reference_distance": cal.fixed["reference_distance"]},
        })
    pairs = {}
    for (s, t), prediction in matrix.items():
        target = refs[t]
        model = predict_surface(refs[s].calibration, target.x, target.r) + prediction.shift_ev
        pairs[f"{s}|{t}"] = {"model": ser.rounded(model, 4), "barrier": [c.barrier_model for c in prediction.columns],
                             "shift": prediction.shift_ev, "rmse": prediction.rmse_ev, "max_error": prediction.max_error_ev,
                             "barrier_error": prediction.barrier_error_mean_abs, "wells_wrong": prediction.wells_wrong}
    return {"names": names, "reference_distance": TRANSFER_REFERENCE_DISTANCE, "refs": items,
            "matrix": {"rmse": grid(lambda p: p.rmse_window), "barrier": grid(lambda p: p.barrier_error_mean_abs),
                       "wells": grid(lambda p: p.wells_wrong), "missed": grid(lambda p: p.barrier_missed)},
            "pairs": pairs}
