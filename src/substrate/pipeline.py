"""Compile an experiment into a workflow of (solve | translate) steps and execute it.

The scale chain is not hard-coded: the Registry finds translator paths between the scales an
experiment names. Every hop is validated, and every product carries provenance back to the root.
"""
from __future__ import annotations

import functools
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from .backends import ClassicalBackend, SolverBackend
from .base import Engine, Registry, Translator
from .limits import whole_number
from .errors import SubstrateError, ValidationError
from .ir import CONTEXT_PREFIX, ProvenanceRecord, Scale, ScientificSystem
from .workflow import Workflow


@dataclass
class Step:
    kind: str                                  # "solve" | "translate"
    component: Engine | Translator
    scale_out: Scale

    @property
    def label(self) -> str:
        return f"{self.kind}:{self.component.name}"


@dataclass
class RunResult:
    experiment_id: str
    root: ScientificSystem
    trace: list[ScientificSystem]              # every system produced, in execution order
    ensemble: dict | None = None
    steps: list[Step] = field(default_factory=list)

    @property
    def final(self) -> ScientificSystem:
        return self.trace[-1]

    def warnings(self) -> list[str]:
        out = []
        for system in self.trace:
            for rec in system.provenance:
                out += [f"[{rec.name}] {w}" for w in rec.warnings if f"[{rec.name}] {w}" not in out]
        return out

    def save(self, directory: str | Path) -> Path:
        """Persist the input, every intermediate system, and a manifest of fingerprints."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "00_input.json").write_text(self.root.to_json(indent=1))
        for i, system in enumerate(self.trace, start=1):
            kind = next(r.step for r in reversed(system.provenance) if r.step != "uncertainty")
            (directory / f"{i:02d}_{system.scale.label}_{kind}.json").write_text(system.to_json(indent=1))
        manifest = {
            "experiment_id": self.experiment_id,
            "input": self.root.fingerprint(),
            "trace": [s.fingerprint() for s in self.trace],
            "ensemble": self.ensemble,
        }
        (directory / "run.json").write_text(json.dumps(manifest, indent=1))
        return directory


class Pipeline:
    def __init__(self, registry: Registry, backend: SolverBackend | None = None):
        self.registry = registry
        self.backend = backend or ClassicalBackend()

    # -- planning ---------------------------------------------------------------
    def plan(self, root: ScientificSystem, propagation: Sequence[Scale | str]) -> list[Step]:
        scales = [Scale.parse(s) for s in propagation]
        if not scales or scales[0] != root.scale:
            scales.insert(0, root.scale)
        steps = [Step("solve", self.registry.engine_for(root.scale, root.kind), root.scale)]
        kind = root.kind                       # translators apply to model kinds, so track it along the chain
        for here, there in zip(scales, scales[1:]):
            for translator in self.registry.find_path(here, there, kind):
                steps.append(Step("translate", translator, translator.target))
                engine = self.registry.engine_for(translator.target, translator.target_kind)
                steps.append(Step("solve", engine, translator.target))
                kind = translator.target_kind
        return steps

    # -- execution ----------------------------------------------------------------
    def run(
        self,
        root: ScientificSystem,
        propagation: Sequence[Scale | str],
        *,
        n_samples: int = 0,
        seed: int = 0,
        experiment_id: str = "",
    ) -> RunResult:
        n_samples = whole_number("pipeline", "n_samples", n_samples, minimum=0, maximum="max_ensemble")      # refused before anything runs
        steps = self.plan(root, propagation)
        trace = self._execute(root, steps)
        result = RunResult(experiment_id or root.name, root, trace, steps=steps)
        if n_samples > 0:
            result.trace, result.ensemble = self._with_ensemble(root, steps, trace, n_samples, seed)
        return result

    def _execute(self, root: ScientificSystem, steps: list[Step]) -> list[ScientificSystem]:
        wf = Workflow()
        previous = None
        for i, step in enumerate(steps):
            name = f"{i}:{step.label}"
            if previous is None:
                wf.add(name, functools.partial(self._apply, step, root))
            else:
                wf.add(name, functools.partial(self._apply, step), (previous,))
            previous = name
        return list(wf.run().values())

    def _apply(self, step: Step, system: ScientificSystem) -> ScientificSystem:
        started = time.perf_counter()
        warnings: list[str] = []
        backend = None
        if step.kind == "solve":
            backend = self.backend.name
            out = step.component.solve(system, self.backend)
        else:
            issues = step.component.validate(system)
            errors = [i.message for i in issues if i.severity == "error"]
            if errors:
                raise ValidationError(f"{step.component.name}: " + "; ".join(errors), issues)
            warnings = [i.message for i in issues if i.severity == "warning"]
            out = step.component.translate(system)
            for name, q in system.parameters.items():       # experiment-level context rides along unchanged
                if name.startswith(CONTEXT_PREFIX) and name not in out.parameters:
                    out.parameters[name] = q
        out = out.evolve(provenance=list(system.provenance))
        out.provenance.append(ProvenanceRecord(
            step=step.kind,
            name=step.component.name,
            version=step.component.version,
            scale_in=system.scale.label,
            scale_out=out.scale.label,
            input_fingerprint=system.fingerprint(),
            output_fingerprint=out.fingerprint(),
            backend=backend,
            approximations=list(step.component.approximations),
            warnings=warnings,
            seconds=time.perf_counter() - started,
        ))
        return out

    # -- uncertainty ----------------------------------------------------------------
    def _draw(self, root: ScientificSystem, rng: np.random.Generator) -> ScientificSystem:
        """One draw of the root's uncertain parameters. Parameters with a `sigma` are independent Gaussians, except those named
        in `structure["parameter_covariance"]`, which are drawn jointly from a multivariate Gaussian with that covariance
        matrix (so correlations, e.g. from a calibration, are respected instead of treated as independent)."""
        draw = root.evolve()
        joint = root.structure.get("parameter_covariance")
        joint_names = list(joint["names"]) if joint else []
        for name, q in root.parameters.items():
            if name not in joint_names and q.sigma is not None and np.ndim(q.sigma) == 0 and q.sigma > 0:
                draw.parameters[name] = replace(q, value=float(rng.normal(q.value, q.sigma)), sigma=None)
        if joint_names:
            mean = np.array([root.parameters[n].value for n in joint_names], dtype=float)
            cov = np.asarray(joint["matrix"], dtype=float)
            floor = np.array([joint.get("minimum", {}).get(n, -np.inf) for n in joint_names], dtype=float)
            for _ in range(10_000):                  # a truncated normal: redraw the vector until it respects the physical floor
                values = rng.multivariate_normal(mean, cov, method="eigh")
                if np.all(values >= floor):
                    break
            else:
                raise SubstrateError(
                    "parameter_covariance: could not draw a vector above the stated minimums (is the mean outside them?)")
            for name, value in zip(joint_names, values):
                draw.parameters[name] = replace(root.parameters[name], value=float(value), sigma=None)
        return draw

    def _uncertain_names(self, root: ScientificSystem) -> list[str]:
        joint = root.structure.get("parameter_covariance")
        if joint:
            missing = [n for n in joint["names"] if n not in root.parameters or np.ndim(root.parameters[n].value) != 0]
            matrix = np.asarray(joint["matrix"], dtype=float)
            if missing or matrix.shape != (len(joint["names"]),) * 2 or not np.allclose(matrix, matrix.T, atol=1e-12):
                raise SubstrateError(
                    f"parameter_covariance must be a symmetric matrix over scalar parameters of the system (bad: {missing or 'shape/symmetry'})")
            if np.linalg.eigvalsh(matrix).min() < -1e-9 * max(1.0, np.abs(matrix).max()):
                raise SubstrateError("parameter_covariance is not positive semi-definite")
        names = {n for n, q in root.parameters.items() if q.sigma is not None and np.ndim(q.sigma) == 0 and q.sigma > 0}
        return sorted(names | set(joint["names"] if joint else []))

    def _with_ensemble(self, root, steps, nominal, n_samples, seed):
        """Re-run the whole chain with root parameters drawn from their uncertainty (independent Gaussians, plus an optional
        joint covariance block); attach the spread of every derived parameter/observable as its sigma. Samples that fail
        validation are counted and dropped."""
        rng = np.random.default_rng(seed)
        varied = self._uncertain_names(root)
        if not varied:
            raise SubstrateError("n_samples > 0 but no root parameter has a sigma")
        samples, failed = [], 0
        for _ in range(n_samples):
            draw = self._draw(root, rng)
            try:
                samples.append(self._execute(draw, steps))
            except ValidationError:
                failed += 1
        info = {"n_requested": n_samples, "n_ok": len(samples), "n_failed": failed, "seed": seed,
                "varied": varied}
        if len(samples) < 2:
            raise SubstrateError(f"ensemble produced {len(samples)} valid sample(s) of {n_samples}; cannot estimate spread")

        annotated = []
        for i, system in enumerate(nominal):
            tables = {"parameters": dict(system.parameters), "observables": dict(system.observables)}
            for table_name, table in tables.items():
                for name, q in table.items():
                    if q.sigma is not None:
                        continue
                    stack = [getattr(s[i], table_name).get(name) for s in samples]
                    if any(x is None or np.shape(x.value) != np.shape(q.value) for x in stack):
                        continue
                    spread = np.std([x.value for x in stack], axis=0, ddof=1)
                    # identical draws still give ~1e-16 relative noise from the mean; that is not a spread
                    if np.any(spread > 1e-10 * np.abs(q.value)):
                        band = None
                        if np.ndim(q.value) == 0:
                            lo, hi = np.percentile([x.value for x in stack], [16, 84])
                            band = (float(lo), float(hi))
                        table[name] = replace(q, sigma=spread, band=band)
            out = system.evolve(**tables)
            out.provenance.append(ProvenanceRecord(
                step="uncertainty",
                name="ensemble.monte_carlo",
                version="0.1.0",
                scale_in=system.scale.label,
                scale_out=system.scale.label,
                input_fingerprint=system.fingerprint(),
                output_fingerprint=out.fingerprint(),
                approximations=["root parameter uncertainties treated as independent Gaussians; "
                                "whole chain re-run per sample; sigma is the sample standard deviation"],
                warnings=[f"{failed} of {n_samples} samples failed validation and were dropped"] if failed else [],
            ))
            annotated.append(out)
        return annotated, info
