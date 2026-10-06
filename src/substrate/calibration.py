"""Calibration: fit the cheap model engines to a reference surface, typically real quantum chemistry.

The model engines (`electronic.evb_two_state_2d`, its 1D slice, and the analytic double well) run in milliseconds but their default
parameters are illustrative. Quantum chemistry is right but costs seconds per geometry. Calibration fits the model's parameters to
a reference surface so the model can stand in for the expensive calculation, for example inside an uncertainty ensemble.

What it deliberately does NOT do is hide where the model is wrong. A calibration reports, besides the fitted parameters:

  * how well the fit reproduces the reference *where it matters* (the low-energy region the nuclei sample), and per heavy-atom
    distance the number of wells, the barrier and the well positions, for the reference and for the model;
  * which parameters sit on a bound (the data cannot pin them) and how well each is determined;
  * leave-one-distance-out validation: how well the fit predicts a distance it never saw.

Uncertainty. The covariance is a Laplace approximation scaled by the misfit, so it measures how far the parameters can move before
the fit visibly degrades. It is NOT the model's error against the reference: the model has a fixed functional form, and the part of
the reference it cannot represent shows up as a systematic bias that no parameter uncertainty captures. Compare the calibrated
model's predictions with the reference's directly (see `examples/calibrate_against_real_chemistry.py`) before trusting either.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .engines.electronic import KIND as EVB1D_KIND, KIND_2D as EVB2D_KIND, morse
from .engines.quantum import KIND as DOUBLE_WELL_KIND, double_well
from .errors import SubstrateError, ValidationError
from .ir import Quantity, Scale, ScientificSystem
from .pes import locate_wells, prominent_minima

#: observables copied from the reference system onto the calibrated model so it runs the same experiment
_CONTEXT_PARAMETERS = ("particle_mass", "heavy_atom_mass", "temperature")


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    unit: str
    lower: float
    upper: float
    start: float          # the model engine's illustrative default, used as one starting point


#: Generous on purpose. A bound that is too tight silently becomes the answer: an earlier version capped the coupling at 3 eV and the
#: fit sat on that cap; with the cap raised the data put it at an interior optimum of 4.1 eV.
EVB2D_PARAMETERS = (
    ParameterSpec("morse_depth", "eV", 0.5, 15.0, 4.6),
    ParameterSpec("morse_alpha", "1/angstrom", 0.3, 6.0, 2.2),
    ParameterSpec("morse_r_eq", "angstrom", 0.7, 1.4, 0.96),
    ParameterSpec("coupling", "eV", 0.0, 20.0, 0.6),
    ParameterSpec("coupling_decay", "1/angstrom", 0.0, 10.0, 3.0),
    ParameterSpec("oo_depth", "eV", 0.001, 10.0, 0.4),
    ParameterSpec("oo_alpha", "1/angstrom", 0.2, 8.0, 2.5),
    ParameterSpec("oo_equilibrium", "angstrom", 1.8, 4.0, 2.7),
)
OFFSET_SPEC = ParameterSpec("diabatic_offset", "eV", -5.0, 5.0, 0.0)
_BOUND_TOLERANCE = 1e-3          # a parameter within this fraction of its range of a bound counts as sitting on it
_SYMMETRY_TOLERANCE = 1e-6       # eV: the reference is treated as symmetric (offset fixed at 0) below this


@dataclass(frozen=True)
class FitSettings:
    """How the fit weighs the reference. These are modelling decisions that change the answer; they are recorded in the result.

    window_ev   only reference points at most this far above the surface minimum are fitted: the nuclear problem samples the
                surface up to the barrier top plus a few zero-point energies, and a far wall (tens of eV) would otherwise dominate
    sigma_ev    the tolerance, flat across the window. 10 meV because rates depend on energies exponentially (kT = 26 meV).
                A relative tolerance (10% of the energy) was tried and let the barrier region drift by 0.07 eV, a 2.5x rate error.
    """

    window_ev: float = 1.5
    sigma_ev: float = 0.01
    starts: int = 12
    seed: int = 0
    reference_distance: float | None = None      # where the coupling is defined; default the middle of the fitted distance range
    bounds: tuple[tuple[str, float, float], ...] = ()   # ((parameter, lower, upper), ...) overriding the generous defaults
    max_points: int = 1500      # a dense reference is thinned (evenly) to about this many fitted points: nine parameters do not
                                # need 15,000, and the cost of every fit scales with the number used
    prior: tuple[tuple[str, float, float], ...] = ()    # ((parameter, mean, sigma), ...): a Gaussian prior, e.g. the values fitted
                                # to another molecule or method. It enters as one extra residual (value - mean) / sigma per entry, so a
                                # small sigma holds the parameter near the mean and a large one lets the data decide. The energy zero
                                # is never given a prior.


def evb2d_energy(x, r, p: dict[str, float], r_ref: float):
    """The 2D valence-bond ground-state energy in closed form (the same surface the engine computes by diagonalisation)."""
    va = morse(r / 2.0 + x, p["morse_depth"], p["morse_alpha"], p["morse_r_eq"])
    vb = morse(r / 2.0 - x, p["morse_depth"], p["morse_alpha"], p["morse_r_eq"]) + p.get("diabatic_offset", 0.0)
    delta = p["coupling"] * np.exp(-p["coupling_decay"] * (r - r_ref))
    return 0.5 * (va + vb) - np.sqrt((0.5 * (va - vb)) ** 2 + delta**2) + morse(r, p["oo_depth"], p["oo_alpha"], p["oo_equilibrium"])


@dataclass
class ColumnComparison:
    """One heavy-atom distance: the reference's and the model's double-well structure."""

    distance: float
    wells_reference: int
    wells_model: int
    barrier_reference: float | None
    barrier_model: float | None
    well_x_reference: float | None
    well_x_model: float | None


@dataclass
class Calibration:
    target_name: str
    parameters: dict[str, float]                    # fitted values, model-engine names
    fixed: dict[str, float]                         # held fixed (reference distance, offset when the target is symmetric)
    pinned: dict[str, str]                          # parameters sitting on a bound -> "lower" / "upper"
    shift_ev: float                                 # constant between the model's energy zero and the reference's
    covariance_names: list[str]                     # parameters that are free and not pinned, in covariance order
    covariance: list[list[float]]
    sigma: dict[str, float]
    correlation: list[list[float]]
    reduced_chi2: float
    n_points: int
    n_free: int
    condition_ratio: float                          # smallest / largest singular value of the weighted Jacobian
    starts_agreeing: int                            # multi-start runs reaching (within 1%) the best cost
    rmse_ev: dict[str, float]                       # by window, e.g. "0.5" -> rmse over reference points up to 0.5 eV
    max_error_ev: dict[str, float]
    columns: list[ColumnComparison]
    settings: dict
    grid: dict                                      # x_extent, distance_min, distance_max of the reference
    context: dict = field(default_factory=dict)     # {name: (value, unit, sigma)} copied from the reference system
    cross_validation: dict | None = None

    # -- diagnostics ------------------------------------------------------------------------------------------------------
    @property
    def poorly_determined(self) -> list[str]:
        """Free parameters whose fit uncertainty exceeds half their value."""
        return [n for n in self.covariance_names if self.sigma[n] > 0.5 * abs(self.parameters[n])]

    def summary(self) -> str:
        lines = [f"Calibration of the 2D valence-bond model against '{self.target_name}'",
                 f"  {self.n_points} reference points in the fitted window, {self.n_free} free parameters (incl. an energy zero), "
                 f"reduced chi2 {self.reduced_chi2:.2f}, {self.starts_agreeing}/{self.settings['starts']} starts reached the best fit", ""]
        lines.append(f"  {'parameter':<18}{'value':>10}  {'+/- (fit)':>10}   note")
        for name, value in self.parameters.items():
            note = f"ON ITS {self.pinned[name].upper()} BOUND: not determined by the data" if name in self.pinned else (
                "poorly determined" if name in self.poorly_determined else "")
            sigma = f"{self.sigma[name]:.3g}" if name in self.sigma else "-"
            lines.append(f"  {name:<18}{value:>10.4g}  {sigma:>10}   {note}")
        lines += ["", "  fit quality against the reference (eV):"]
        for window, rmse in self.rmse_ev.items():
            lines.append(f"    points up to {window:>4} eV: rmse {rmse:.4f}, worst {self.max_error_ev[window]:.4f}")
        lines += ["", "  per heavy-atom distance (double-well structure; barrier in eV):",
                  f"    {'R (A)':>7}{'wells ref/model':>17}{'barrier ref':>13}{'model':>8}{'error':>8}"]
        for c in self.columns:
            if c.barrier_reference is not None and c.barrier_model is not None:
                tail = f"{c.barrier_reference:>13.3f}{c.barrier_model:>8.3f}{c.barrier_model - c.barrier_reference:>+8.3f}"
            else:
                tail = f"{'-':>13}{'-':>8}{'-':>8}"
            lines.append(f"    {c.distance:>7.2f}{f'{c.wells_reference}/{c.wells_model}':>17}{tail}")
        if self.cross_validation:
            cv = self.cross_validation
            lines += ["", f"  leave-one-distance-out (interior distances): rmse {cv['rmse_mean']:.4f} eV on average, worst {cv['rmse_max']:.4f};"
                          f" barrier error {cv['barrier_error_mean_abs']:.3f} eV on average, worst {cv['barrier_error_max_abs']:.3f}"]
        lines += ["", "  The fit uncertainty is NOT the model's error against the reference: compare predictions directly."]
        return "\n".join(lines)

    # -- building calibrated systems -------------------------------------------------------------------------------------------
    def system(self, kind: str = EVB2D_KIND, *, scan_distance: float | None = None, distance: float | None = None,
               name: str | None = None, with_uncertainty: bool = True, context: dict | None = None) -> ScientificSystem:
        """A model system carrying the fitted parameters.

        kind = electronic.evb_two_state_2d   the calibrated 2D model; with_uncertainty attaches the fit's marginal sigmas and the
                                             joint covariance, so an ensemble draws correlated parameters
        kind = electronic.evb_two_state      the 1D scan at one heavy-atom `distance` (the coupling evaluated there); no uncertainty
        `context` overrides the particle mass, heavy-atom mass, temperature and any context.* parameters copied from the reference.
        """
        ctx = {**self.context, **(context or {})}
        ctx_params = {n: Quantity(v[0], v[1], v[2] if len(v) > 2 else None, "input") for n, v in ctx.items()}
        p = {**self.parameters, **self.fixed}
        r_ref = self.fixed["reference_distance"]
        if kind == EVB2D_KIND:
            params = {n: Quantity(p[n], spec.unit, self.sigma.get(n) if with_uncertainty else None, "calibration")
                      for n, spec in ((s.name, s) for s in (*EVB2D_PARAMETERS, OFFSET_SPEC)) if n in p}
            params["reference_distance"] = Quantity(r_ref, "angstrom", source="calibration")
            params["x_extent"] = Quantity(self.grid["x_extent"], "angstrom", source="calibration")
            params["distance_min"] = Quantity(self.grid["distance_min"], "angstrom", source="calibration")
            params["distance_max"] = Quantity(self.grid["distance_max"], "angstrom", source="calibration")
            params["n_x"], params["n_r"] = Quantity(41, "1"), Quantity(21, "1")
            if scan_distance is not None:
                params["scan_distance"] = Quantity(scan_distance, "angstrom")
            params.update(ctx_params)
            structure = {}
            if with_uncertainty and self.covariance_names:
                structure["parameter_covariance"] = {"names": list(self.covariance_names), "matrix": self.covariance,
                                                     "minimum": self._minimums(EVB2D_PARAMETERS)}
            return ScientificSystem(name or f"{self.target_name} (calibrated model)", Scale.ELECTRONIC_STRUCTURE, EVB2D_KIND,
                                    parameters=params, structure=structure)
        if kind == EVB1D_KIND:
            if distance is None:
                raise ValidationError("a 1D model needs the heavy-atom `distance` to take the slice at")
            coupling = p["coupling"] * float(np.exp(-p["coupling_decay"] * (distance - r_ref)))
            params = {
                "donor_acceptor_distance": Quantity(distance, "angstrom", source="calibration"),
                "morse_depth": Quantity(p["morse_depth"], "eV", source="calibration"),
                "morse_alpha": Quantity(p["morse_alpha"], "1/angstrom", source="calibration"),
                "morse_r_eq": Quantity(p["morse_r_eq"], "angstrom", source="calibration"),
                "coupling": Quantity(coupling, "eV", source="calibration"),
                "diabatic_offset": Quantity(p.get("diabatic_offset", 0.0), "eV", source="calibration"),
                "scan_min_bond_length": Quantity(distance / 2.0 - self.grid["x_extent"], "angstrom", source="calibration"),
                "n_scan": Quantity(241, "1"),
            }
            params.update(ctx_params)
            return ScientificSystem(name or f"{self.target_name} (calibrated model, R={distance:g})", Scale.ELECTRONIC_STRUCTURE,
                                    EVB1D_KIND, parameters=params)
        raise ValidationError(f"no calibrated model of kind '{kind}' (have: {EVB2D_KIND}, {EVB1D_KIND})")

    def _minimums(self, specs) -> dict[str, float]:
        """The physical floor of each uncertain parameter (a depth or a distance cannot be negative), so ensembles redraw rather
        than discard runs for a draw the model correctly rejects."""
        lower = {s.name: s.lower for s in specs}
        return {n: float(lower[n]) for n in self.covariance_names if n in lower}

    # -- persistence --------------------------------------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["columns"] = [dict(c.__dict__) for c in self.columns]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        d = dict(d)
        d["columns"] = [ColumnComparison(**c) for c in d["columns"]]
        d["context"] = {k: tuple(v) for k, v in d.get("context", {}).items()}
        return cls(**d)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        return cls.from_dict(json.loads(Path(path).read_text()))


# =====================================================================================================================
# fitting
# =====================================================================================================================
def _reference(target: ScientificSystem):
    for name in ("surface_x", "surface_r", "surface_energy"):
        if name not in target.observables:
            raise ValidationError(f"{target.name}: the calibration target has not been solved to a 2D surface ({name} missing)")
    x = np.asarray(target.obs("surface_x", "angstrom"), dtype=float)
    r = np.asarray(target.obs("surface_r", "angstrom"), dtype=float)
    e = np.asarray(target.obs("surface_energy", "eV"), dtype=float)
    if e.shape != (len(x), len(r)) or not np.all(np.isfinite(e)):
        raise ValidationError(f"{target.name}: surface_energy must be a finite (len(x), len(r)) table")
    return x, r, e - e.min()


def _is_symmetric(x: np.ndarray, e: np.ndarray) -> bool:
    return bool(np.allclose(x, -x[::-1], atol=1e-9) and np.abs(e - e[::-1]).max() < _SYMMETRY_TOLERANCE)


def _fit(x, r, e, settings: FitSettings, symmetric: bool, r_ref: float, columns=None, starts=None, initial=None):
    """Weighted least squares over the points of `e` within the window (optionally only some distance columns).
    `initial` (a full solution vector) replaces the first starting point, for warm-started refits.
    Returns (names, scipy result, mask, costs, (lower, upper))."""
    specs = list(EVB2D_PARAMETERS) + ([] if symmetric else [OFFSET_SPEC])
    names = [s.name for s in specs]
    override = {name: (lo, hi) for name, lo, hi in settings.bounds}
    unknown = set(override) - set(names)
    if unknown:
        raise ValidationError(f"bounds given for unknown parameter(s): {sorted(unknown)} (free parameters: {names})")
    lower = np.array([override.get(s.name, (s.lower, s.upper))[0] for s in specs] + [-200.0])
    upper = np.array([override.get(s.name, (s.lower, s.upper))[1] for s in specs] + [200.0])
    start = np.array([float(np.clip(s.start, lo, hi)) for s, lo, hi in zip(specs, lower[:-1], upper[:-1])] + [0.0])
    prior_index, prior_mean, prior_sigma = [], [], []
    for name, mean, sigma in settings.prior:
        if name not in names:
            raise ValidationError(f"prior given for '{name}', which is not a free parameter here (free parameters: {names})")
        if not sigma > 0:
            raise ValidationError(f"prior sigma for '{name}' must be positive, got {sigma}")
        prior_index.append(names.index(name))
        prior_mean.append(float(mean))
        prior_sigma.append(float(sigma))
        start[prior_index[-1]] = float(np.clip(mean, lower[prior_index[-1]], upper[prior_index[-1]]))     # begin where the prior is
    prior_index, prior_mean, prior_sigma = np.array(prior_index, dtype=int), np.array(prior_mean), np.array(prior_sigma)
    keep_x = np.ones(len(x), dtype=bool) if not symmetric else x >= -1e-12          # a symmetric surface counts once, not twice
    X, R = np.meshgrid(x, r, indexing="ij")
    mask = (e <= settings.window_ev) & keep_x[:, None]
    if columns is not None:
        column_mask = np.zeros(len(r), dtype=bool)
        column_mask[list(columns)] = True
        mask &= column_mask[None, :]
    if mask.sum() > settings.max_points:
        stride = int(np.ceil(np.sqrt(mask.sum() / settings.max_points)))
        keep_rows = np.union1d(np.arange(0, len(x), stride), [len(x) // 2])           # always keep the centre ...
        keep_cols = np.union1d(np.arange(0, len(r), stride), [len(r) - 1])            # ... and the far end of the distance range
        thin = np.zeros_like(mask)
        thin[np.ix_(keep_rows, keep_cols)] = True
        mask &= thin
    if mask.sum() + len(prior_index) <= len(specs) + 1:       # a prior is information too
        raise ValidationError("too few reference points in the fitting window to determine the parameters")

    def model(theta):
        p = dict(zip(names, theta[:-1]))
        return evb2d_energy(X[mask], R[mask], p, r_ref) + theta[-1]

    def residual(theta):
        data = (model(theta) - e[mask]) / settings.sigma_ev
        return np.concatenate([data, (theta[prior_index] - prior_mean) / prior_sigma]) if len(prior_index) else data

    rng = np.random.default_rng(settings.seed)
    best, costs = None, []
    for k in range(starts or settings.starts):
        theta0 = start.copy()
        if k:
            theta0[:-1] = np.clip(start[:-1] * rng.uniform(0.6, 1.5, len(specs)), lower[:-1] + 1e-9, upper[:-1] - 1e-9)
        if k == 0 and initial is not None:
            theta0 = np.clip(initial, lower + 1e-9, upper - 1e-9)
        else:
            theta0[-1] = float(np.median(e[mask] - model(np.append(theta0[:-1], 0.0))))  # a sensible energy zero for this start
        try:
            solution = least_squares(residual, theta0, bounds=(lower, upper), x_scale="jac", max_nfev=2000)
        except (ValueError, FloatingPointError):
            continue
        costs.append(solution.cost)
        if best is None or solution.cost < best.cost:
            best = solution
    if best is None:
        raise SubstrateError("the calibration fit failed from every starting point")
    return names, best, mask, costs, (lower, upper)


def _column_comparison(x, e_ref, e_model, r) -> list[ColumnComparison]:
    out = []
    for j, rj in enumerate(r):
        wells_ref, wells_model = locate_wells(e_ref[:, j]), locate_wells(e_model[:, j])
        out.append(ColumnComparison(
            float(rj), len(prominent_minima(e_ref[:, j])), len(prominent_minima(e_model[:, j])),
            float(e_ref[wells_ref[1], j] - e_ref[wells_ref[0], j]) if wells_ref else None,
            float(e_model[wells_model[1], j] - e_model[wells_model[0], j]) if wells_model else None,
            float(abs(x[wells_ref[0]])) if wells_ref else None, float(abs(x[wells_model[0]])) if wells_model else None,
        ))
    return out


def calibrate_evb_2d(target: ScientificSystem, settings: FitSettings | None = None, *,
                     cross_validate: bool = True) -> Calibration:
    """Fit the 2D valence-bond model to a solved system exposing `surface_x`, `surface_r`, `surface_energy`
    (for example the output of the quantum-chemistry engine)."""
    settings = settings or FitSettings()
    x, r, e = _reference(target)
    symmetric = _is_symmetric(x, e)
    r_ref = settings.reference_distance if settings.reference_distance is not None else float(0.5 * (r.min() + r.max()))
    names, best, mask, costs, (lower, upper) = _fit(x, r, e, settings, symmetric, r_ref)

    theta = best.x
    parameters = dict(zip(names, (float(v) for v in theta[:-1])))
    span = upper - lower
    pinned = {n: ("lower" if abs(v - lo) < _BOUND_TOLERANCE * s else "upper")
              for n, v, lo, hi, s in zip(names, theta[:-1], lower[:-1], upper[:-1], span[:-1])
              if abs(v - lo) < _BOUND_TOLERANCE * s or abs(v - hi) < _BOUND_TOLERANCE * s}
    free = [i for i, n in enumerate(names) if n not in pinned] + [len(names)]            # parameters not pinned, plus the energy zero
    jacobian = best.jac[:, free]
    n_free = len(free)
    dof = int(mask.sum()) - n_free
    if dof < 1:
        raise ValidationError("fewer reference points than free parameters: cannot estimate the fit uncertainty")
    reduced_chi2 = float(np.sum(best.fun[: int(mask.sum())] ** 2) / dof)               # the data's misfit only, not the prior's penalty
    covariance_full = max(reduced_chi2, 1.0) * np.linalg.pinv(jacobian.T @ jacobian, rcond=1e-12)
    cov_names = [names[i] for i in free[:-1]]
    covariance = covariance_full[:-1, :-1]                                              # marginalise over the energy zero
    sigma = {n: float(np.sqrt(max(covariance[i, i], 0.0))) for i, n in enumerate(cov_names)}
    sd = np.sqrt(np.clip(np.diag(covariance), 1e-300, None))
    correlation = covariance / np.outer(sd, sd)
    singular = np.linalg.svd(jacobian, compute_uv=False)

    X, R = np.meshgrid(x, r, indexing="ij")
    p_full = {**parameters, "diabatic_offset": parameters.get("diabatic_offset", 0.0)}
    model = evb2d_energy(X, R, p_full, r_ref) + theta[-1]
    err = model - e
    rmse, worst = {}, {}
    for window in (0.5, 1.0, settings.window_ev):
        m = e <= window
        rmse[f"{window:g}"], worst[f"{window:g}"] = float(np.sqrt((err[m] ** 2).mean())), float(np.abs(err[m]).max())

    fixed = {"reference_distance": float(r_ref)}
    if symmetric:
        fixed["diabatic_offset"] = 0.0
    context = {}
    for name in _CONTEXT_PARAMETERS:
        if name in target.parameters:
            q = target.parameters[name]
            context[name] = (float(q.value), q.unit, None if q.sigma is None else float(q.sigma))
    for name, q in target.parameters.items():
        if name.startswith("context."):
            context[name] = (float(q.value), q.unit, None if q.sigma is None else float(q.sigma))

    calibration = Calibration(
        target_name=target.name, parameters=parameters, fixed=fixed, pinned=pinned, shift_ev=float(theta[-1]),
        covariance_names=cov_names, covariance=covariance.tolist(), sigma=sigma, correlation=correlation.tolist(),
        reduced_chi2=reduced_chi2, n_points=int(mask.sum()), n_free=n_free,
        condition_ratio=float(singular.min() / singular.max()),
        starts_agreeing=int(sum(c <= best.cost * 1.01 + 1e-12 for c in costs)),
        rmse_ev=rmse, max_error_ev=worst, columns=_column_comparison(x, e, model - 0.0, r),
        settings={"window_ev": settings.window_ev, "sigma_ev": settings.sigma_ev, "starts": settings.starts, "seed": settings.seed,
                  "symmetric_reference": symmetric, "prior": [list(entry) for entry in settings.prior]},
        grid={"x_extent": float(np.abs(x).max()), "distance_min": float(r.min()), "distance_max": float(r.max())},
        context=context,
    )
    if cross_validate:
        calibration.cross_validation = leave_one_distance_out(target, settings, warm_start=theta)
    return calibration


def leave_one_distance_out(target: ScientificSystem, settings: FitSettings | None = None, starts: int = 1,
                           max_folds: int = 12, warm_start=None) -> dict:
    """Refit without one heavy-atom distance at a time and predict that distance: an honest out-of-sample error.
    The first and last distances are extrapolations, not interpolations, and are excluded. At most `max_folds` interior distances
    are held out (evenly spaced), so the cost does not grow with the grid. Each refit starts from the full fit's solution
    (`warm_start`, fitted here if not given): removing one distance only nudges the answer."""
    settings = settings or FitSettings()
    x, r, e = _reference(target)
    symmetric = _is_symmetric(x, e)
    r_ref = settings.reference_distance if settings.reference_distance is not None else float(0.5 * (r.min() + r.max()))
    if warm_start is None:
        warm_start = _fit(x, r, e, settings, symmetric, r_ref)[1].x
    X, R = np.meshgrid(x, r, indexing="ij")
    rows = []
    interior = np.arange(1, len(r) - 1)
    if len(interior) > max_folds:                                                    # a dense grid needs no more folds than a coarse one
        interior = np.unique(np.round(np.linspace(interior[0], interior[-1], max_folds)).astype(int))
    for held in interior:
        names, best, _, _, _ = _fit(x, r, e, settings, symmetric, r_ref, columns=[j for j in range(len(r)) if j != held],
                                    starts=starts, initial=warm_start)
        p = {**dict(zip(names, best.x[:-1])), "diabatic_offset": dict(zip(names, best.x[:-1])).get("diabatic_offset", 0.0)}
        predicted = evb2d_energy(x, r[held], p, r_ref) + best.x[-1]
        column = e[:, held]
        inside = column <= settings.window_ev
        wells_ref, wells_model = locate_wells(column), locate_wells(predicted)
        barrier_error = (float((predicted[wells_model[1]] - predicted[wells_model[0]]) - (column[wells_ref[1]] - column[wells_ref[0]]))
                         if wells_ref and wells_model else None)
        rows.append({"distance": float(r[held]), "rmse": float(np.sqrt(((predicted - column)[inside] ** 2).mean())),
                     "barrier_error": barrier_error})
    barrier_errors = [abs(row["barrier_error"]) for row in rows if row["barrier_error"] is not None]
    return {
        "rows": rows, "rmse_mean": float(np.mean([row["rmse"] for row in rows])), "rmse_max": float(np.max([row["rmse"] for row in rows])),
        "barrier_error_mean_abs": float(np.mean(barrier_errors)) if barrier_errors else float("nan"),
        "barrier_error_max_abs": float(np.max(barrier_errors)) if barrier_errors else float("nan"),
    }


# =====================================================================================================================
# the analytic double well
# =====================================================================================================================
@dataclass
class DoubleWellFit:
    """The quartic double well V = V0 (1 - (x/a)^2)^2 + (dE/2)(x/a) fitted to a 1D slice."""

    parameters: dict[str, float]            # barrier_height (eV), half_separation (angstrom), reaction_energy (eV)
    sigma: dict[str, float]
    covariance_names: list[str]
    covariance: list[list[float]]
    window_ev: float
    rmse_ev: float
    max_error_ev: float
    n_points: int
    pinned: dict[str, str]

    def system(self, *, mass: float, temperature: float = 300.0, with_uncertainty: bool = True, context: dict | None = None,
               name: str = "calibrated double well") -> ScientificSystem:
        units = {"barrier_height": "eV", "half_separation": "angstrom", "reaction_energy": "eV"}
        params = {n: Quantity(v, units[n], self.sigma.get(n) if with_uncertainty else None, "calibration")
                  for n, v in self.parameters.items()}
        params["mass"] = Quantity(mass, "amu", source="input")
        params["temperature"] = Quantity(temperature, "K", source="input")
        for n, v in (context or {}).items():
            params[n] = Quantity(v[0], v[1], v[2] if len(v) > 2 else None, "input")
        structure = {}
        if with_uncertainty and self.covariance_names:
            structure["parameter_covariance"] = {"names": list(self.covariance_names), "matrix": self.covariance,
                                                 "minimum": {"barrier_height": 1e-3, "half_separation": 0.05}}
        return ScientificSystem(name, Scale.QUANTUM, DOUBLE_WELL_KIND, parameters=params, structure=structure)


def calibrate_double_well(x, energy_ev, *, window_ev: float | None = None, sigma_ev: float = 0.01) -> DoubleWellFit:
    """Fit the quartic double well to a 1D slice E(x) (eV). The window defaults to 0.6 eV above the barrier: the quartic cannot
    follow the real walls far from the wells, and the nuclear problem only needs the wells and the barrier."""
    x = np.asarray(x, dtype=float)
    e = np.asarray(energy_ev, dtype=float)
    e = e - e.min()
    wells = locate_wells(e)
    if wells is None:
        raise ValidationError("the slice has fewer than two wells: there is no double well to fit")
    i_left, i_top, i_right = wells
    barrier = float(e[i_top] - e[i_left])
    window = window_ev if window_ev is not None else float(e[i_top]) + 0.6
    mask = e <= window
    symmetric = bool(np.allclose(x, -x[::-1], atol=1e-9) and np.abs(e - e[::-1]).max() < _SYMMETRY_TOLERANCE)
    names = ["barrier_height", "half_separation"] + ([] if symmetric else ["reaction_energy"])
    lower = np.array([1e-3, 0.05] + ([] if symmetric else [-3.0]) + [-100.0])
    upper = np.array([20.0, 2.0] + ([] if symmetric else [3.0]) + [100.0])
    start = np.array([barrier, abs(x[i_left])] + ([] if symmetric else [float(e[i_right] - e[i_left])]) + [0.0])

    def model(theta):
        d_e = 0.0 if symmetric else theta[2]
        return double_well(x[mask], theta[0], theta[1], d_e) + theta[-1]

    solution = least_squares(lambda t: (model(t) - e[mask]) / sigma_ev, start, bounds=(lower, upper), x_scale="jac")
    theta = solution.x
    parameters = {n: float(v) for n, v in zip(names, theta[:-1])}
    if symmetric:
        parameters["reaction_energy"] = 0.0
    pinned = {n: ("lower" if abs(v - lo) < _BOUND_TOLERANCE * (hi - lo) else "upper")
              for n, v, lo, hi in zip(names, theta[:-1], lower[:-1], upper[:-1])
              if abs(v - lo) < _BOUND_TOLERANCE * (hi - lo) or abs(v - hi) < _BOUND_TOLERANCE * (hi - lo)}
    free = [i for i, n in enumerate(names) if n not in pinned] + [len(names)]
    jacobian = solution.jac[:, free]
    dof = max(int(mask.sum()) - len(free), 1)
    cov = max(2.0 * solution.cost / dof, 1.0) * np.linalg.pinv(jacobian.T @ jacobian, rcond=1e-12)
    cov_names = [names[i] for i in free[:-1]]
    covariance = cov[:-1, :-1]
    err = model(theta) - e[mask]
    return DoubleWellFit(
        parameters=parameters, sigma={n: float(np.sqrt(max(covariance[i, i], 0.0))) for i, n in enumerate(cov_names)},
        covariance_names=cov_names, covariance=covariance.tolist(), window_ev=float(window),
        rmse_ev=float(np.sqrt((err**2).mean())), max_error_ev=float(np.abs(err).max()), n_points=int(mask.sum()), pinned=pinned,
    )


# =====================================================================================================================
# experiment files
# =====================================================================================================================
def calibrated_experiment(calibration: Calibration, source: dict, *, n_samples: int = 100, seed: int = 0,
                          scan_distance: float | None = None) -> dict:
    """An experiment (as a dict, ready for YAML) that runs the same propagation as `source` on the calibrated model, with the fit
    uncertainty as an ensemble. `source` is the original experiment dict (with an `experiment:` key or without)."""
    spec = source.get("experiment", source)
    system = calibration.system(scan_distance=scan_distance if scan_distance is not None else _scan_distance(spec))
    parameters = {n: {"value": q.value, "unit": q.unit, **({"sigma": q.sigma} if q.sigma is not None else {})}
                  for n, q in system.parameters.items()}
    return {"experiment": {
        "id": f"{spec.get('id', 'experiment')}-calibrated",
        "phenomenon": spec.get("phenomenon", ""),
        "system": {
            "name": system.name, "scale": "electronic_structure", "kind": EVB2D_KIND,
            "structure": system.structure, "parameters": parameters,
        },
        "propagation": spec.get("propagation", ["electronic_structure"]),
        "ensemble": {"n_samples": n_samples, "seed": seed},
    }}


def write_experiment(spec: dict, path: str | Path, header: str = "") -> Path:
    """Write an experiment dict as YAML, with a comment header."""
    import yaml
    path = Path(path)
    comment = "".join(f"# {line}".rstrip() + "\n" for line in header.splitlines())
    path.write_text(comment + yaml.safe_dump(spec, sort_keys=False, default_flow_style=None, width=120), encoding="utf-8")
    return path


def _scan_distance(spec: dict) -> float | None:
    p = spec.get("system", {}).get("parameters", {}).get("scan_distance")
    return float(p["value"]) if p else None
