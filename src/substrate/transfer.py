"""Transfer: do model parameters fitted to one reference surface carry over to another molecule or another level of theory?

A calibration (see `calibration.py`) fits the cheap model to ONE reference. Whether the fitted values mean anything beyond that one
reference is a separate question, and the answer decides how much expensive quantum chemistry a new system needs. This module asks
it three ways, none of which is a statement about the fit's own uncertainty:

  1. cross-prediction: the parameters fitted to reference A, applied unchanged to reference B's grid, against B's real energies.
     The only freedom is a constant energy shift (the energy zero of each calculation is arbitrary). The diagonal is each
     reference's own calibration, so a row's off-diagonal entries are read against it.
  2. downstream answers: the rate and isotope effect the real chain gives for B, against the same chain run on A's parameters,
     at the heavy-atom distances where B's reference barrier has stated heights.
  3. few-shot learning curves: with only k of B's heavy-atom distances computed, how well does a fit predict B's other distances,
     starting from scratch and starting from A's values as a Gaussian prior?

Nothing here is a physical claim about the fitted numbers. They are effective parameters of a fixed functional form, and two sets
that differ greatly may still predict one surface equally well (the parameters are strongly correlated), which is why the
comparison is made on predictions and not parameter by parameter.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .calibration import (EVB2D_PARAMETERS, Calibration, FitSettings, _column_comparison, _fit, _is_symmetric, _reference,
                          calibrate_evb_2d, evb2d_energy)
from .errors import SubstrateError, ValidationError
from .ir import Quantity, ScientificSystem

WINDOWS = (0.5, 1.0, 1.5)


# =====================================================================================================================
# references
# =====================================================================================================================
@dataclass
class Reference:
    """A solved reference surface and its calibration."""

    name: str
    molecule: str                   # the template name recorded on the solved system, else "custom"
    method: str                     # "theory/basis"
    system: ScientificSystem        # the experiment's input system (unsolved), used to re-run the real chain at other distances
    x: np.ndarray
    r: np.ndarray
    e: np.ndarray                   # eV above the surface minimum
    calibration: Calibration

    @property
    def symmetric(self) -> bool:
        return _is_symmetric(self.x, self.e)

    @property
    def window_ev(self) -> float:
        """The energy window (eV above the surface minimum) this reference's calibration was fitted over."""
        return float(self.calibration.settings["window_ev"])

    def label(self) -> str:
        return f"{self.molecule} {self.method}"

    def system_at(self, distance: float, mass: float | None = None) -> ScientificSystem:
        """The real (quantum-chemistry) system for this reference at another heavy-atom distance, and optionally another mass."""
        changes = {"scan_distance": Quantity(float(distance), "angstrom")}
        if mass is not None:
            changes["particle_mass"] = Quantity(float(mass), "amu")
        return self.system.evolve(parameters={**self.system.parameters, **changes})


def make_reference(name: str, system: ScientificSystem, solved: ScientificSystem, settings: FitSettings | None = None, *,
                   cross_validate: bool = False) -> Reference:
    """Calibrate `solved` (the output of the quantum-chemistry engine for `system`) and wrap it with what the analyses need."""
    x, r, e = _reference(solved)
    molecule = solved.structure.get("molecule", {}).get("template", "custom")
    method = solved.structure.get("method", {})
    return Reference(name, molecule, f"{method.get('theory', '?')}/{method.get('basis', '?')}", system, x, r, e,
                     calibrate_evb_2d(solved, settings, cross_validate=cross_validate))


def fit_settings(experiment, base: FitSettings | None = None) -> FitSettings:
    """`base` with the fit window the experiment file declares (`calibration: {window_ev: ...}`), if any. A surface whose barrier top lies
    well above 1.5 eV (an asymmetric one: the metastable well and the barrier above it) must declare a window that includes it, because
    the fit otherwise never sees the barrier it is meant to reproduce. The choice is in the file, so it is visible and recorded."""
    base = base or FitSettings()
    window = experiment.notes.get("calibration", {}).get("window_ev")
    return base if window is None else replace(base, window_ev=float(window))


# =====================================================================================================================
# 1. cross-prediction
# =====================================================================================================================
@dataclass
class Prediction:
    """One calibration's parameters applied to one reference's grid."""

    source: str
    target: str
    shift_ev: float                             # the constant that best aligns the two energy zeros (fitted, not a parameter)
    rmse_ev: dict[str, float]                   # by window, over the target's points up to that energy
    max_error_ev: dict[str, float]
    columns: list = field(default_factory=list)  # ColumnComparison per target distance (target = reference, source = model)
    window_ev: float = 1.5                       # the energy window the shift was fitted over and the headline error is scored in

    @property
    def rmse_window(self) -> float:
        """The RMSE over the points inside `window_ev`."""
        return self.rmse_ev[f"{self.window_ev:g}"]

    @property
    def barrier_errors(self) -> list[float]:
        """Model minus reference barrier, at the target distances where both have a double well."""
        return [c.barrier_model - c.barrier_reference for c in self.columns if c.barrier_reference is not None and c.barrier_model is not None]

    @property
    def barrier_error_mean_abs(self) -> float:
        errors = self.barrier_errors
        return float(np.mean(np.abs(errors))) if errors else float("nan")

    @property
    def wells_wrong(self) -> int:
        """Target distances where the model has a different number of wells from the reference."""
        return sum(c.wells_reference != c.wells_model for c in self.columns)

    @property
    def barrier_missed(self) -> int:
        """Target distances where the reference has a double well and the model does not (or the reverse): no barrier to compare."""
        return sum((c.barrier_reference is None) != (c.barrier_model is None) for c in self.columns)


def predict_surface(calibration: Calibration, x, r) -> np.ndarray:
    """The calibrated model's energy on a (len(x), len(r)) grid, without the arbitrary energy zero."""
    X, R = np.meshgrid(np.asarray(x, dtype=float), np.asarray(r, dtype=float), indexing="ij")
    p = {**calibration.parameters, **calibration.fixed}
    return evb2d_energy(X, R, p, calibration.fixed["reference_distance"])


def _score(x, r, e, model, window_ev: float, shift: float | None = None) -> Prediction:
    """Compare model energies to reference energies `e` (both on the same grid) after the best constant shift."""
    inside = e <= window_ev
    if not inside.any():
        raise ValidationError("the reference has no points inside the comparison window")
    shift = float(np.mean(e[inside] - model[inside])) if shift is None else float(shift)
    err = model + shift - e
    windows = sorted({*WINDOWS, float(window_ev)})                  # the comparison window is always reported, whatever it is
    rmse = {f"{w:g}": float(np.sqrt((err[e <= w] ** 2).mean())) for w in windows if (e <= w).any()}
    worst = {f"{w:g}": float(np.abs(err[e <= w]).max()) for w in windows if (e <= w).any()}
    return Prediction("", "", shift, rmse, worst, _column_comparison(x, e, model + shift, r), float(window_ev))


def cross_predict(source: Calibration, target: Reference, window_ev: float | None = None) -> Prediction:
    """`source`'s fitted parameters, unchanged, against `target`'s real energies (best constant shift only). The window defaults to the one
    the target's own calibration was fitted over, so every source is scored on the range that matters for that target."""
    window = target.window_ev if window_ev is None else window_ev
    prediction = _score(target.x, target.r, target.e, predict_surface(source, target.x, target.r), window)
    prediction.source, prediction.target = source.target_name, target.name
    return prediction


def transfer_matrix(references: dict[str, Reference], window_ev: float | None = None) -> dict[tuple[str, str], Prediction]:
    """Every reference's calibration against every reference: {(source name, target name): Prediction}. The diagonal is each
    reference's own calibration, scored the same way as the others (so its shift is the best constant, not the fitted one). Each target is
    scored over its own fit window unless `window_ev` overrides it for all."""
    out = {}
    for source_name, source in references.items():
        for target_name, target in references.items():
            prediction = cross_predict(source.calibration, target, window_ev)
            prediction.source = source_name
            prediction.target = target_name
            out[(source_name, target_name)] = prediction
    return out


# =====================================================================================================================
# parameter heterogeneity
# =====================================================================================================================
def parameter_table(references: dict[str, Reference]) -> dict[str, dict[str, tuple[float | None, float | None]]]:
    """{parameter: {reference: (value, fit sigma or None)}}. A parameter pinned to a bound has no sigma. The energy offset is a row only
    when some reference is asymmetric; for a symmetric one it is fixed at zero and its entry is (None, None)."""
    names = [s.name for s in EVB2D_PARAMETERS]
    if any("diabatic_offset" in ref.calibration.parameters for ref in references.values()):
        names.append("diabatic_offset")
    return {n: {k: ((ref.calibration.parameters[n], ref.calibration.sigma.get(n)) if n in ref.calibration.parameters else (None, None))
                for k, ref in references.items()} for n in names}


def spread_ratio(values) -> float:
    """max / min of positive values: how many times the largest exceeds the smallest (NaN if the smallest is zero or about zero, in absolute terms or against the largest)."""
    v = np.asarray(list(values), dtype=float)
    return float(v.max() / v.min()) if len(v) > 1 and v.min() > 1e-6 and v.min() > 1e-6 * v.max() else float("nan")   # a value at ~0 (a bound) has no ratio


# =====================================================================================================================
# 2. downstream rates and isotope effects
# =====================================================================================================================
@dataclass
class RateRow:
    target: str
    source: str
    barrier_ev: float            # the target's reference barrier at the distance used
    distance: float
    k_real: float | None         # s^-1, the real chain's rate for the target
    k_model: float | None        # the same chain run on the source's parameters
    kie_real: float | None
    kie_model: float | None
    note: str = ""               # why a rate is missing, when it is (a refusal is information, not an error)

    @property
    def ratio(self) -> float | None:
        return None if not self.k_real or not self.k_model else self.k_model / self.k_real


def distance_at_barrier(target: Reference, barrier_ev: float) -> float | None:
    """The heavy-atom distance (rounded to 0.01 angstrom) where the target's reference barrier reaches `barrier_ev`, interpolated
    between its grid distances; None if the barrier never reaches it inside the scanned range."""
    barriers = []
    for column in _column_comparison(target.x, target.e, target.e, target.r):
        barriers.append(np.nan if column.barrier_reference is None else column.barrier_reference)
    barriers = np.array(barriers)
    ok = np.isfinite(barriers)
    if ok.sum() < 2 or not barriers[ok].min() <= barrier_ev <= barriers[ok].max():
        return None
    order = np.argsort(barriers[ok])
    return round(float(np.interp(barrier_ev, barriers[ok][order], target.r[ok][order])), 2)


def _chain_rates(pipeline, system_for_mass, route, masses) -> tuple[dict | None, str]:
    """k_f at each mass through the chain, or (None, why) when the chain refuses."""
    rates = {}
    try:
        for mass in masses:
            rates[mass] = pipeline.run(system_for_mass(mass), route).final.param("k_f")
    except SubstrateError as error:
        return None, f"{type(error).__name__}: {str(error)[:200]}"
    return rates, ""


def rate_comparison(sources: dict[str, Reference], targets: dict[str, Reference], pipeline, *, barriers=(0.15, 0.4, 0.8),
                    route=("electronic_structure", "quantum", "reaction"), proton: float = 1.007276, deuteron: float = 2.013553,
                    progress=None) -> list[RateRow]:
    """For each target and each stated barrier height: the real chain's rate and isotope effect at the distance where the
    target's reference barrier has that height, against the same chain run on each source's parameters at that distance.
    The model's scan window is set to the target's grid, so the two chains integrate over the same region."""
    route = list(route)
    rows: list[RateRow] = []
    for target_name, target in targets.items():
        for barrier in barriers:
            distance = distance_at_barrier(target, barrier)
            if distance is None:
                rows.append(RateRow(target_name, "-", barrier, float("nan"), None, None, None, None,
                                    "barrier height not reached inside the scanned distances"))
                continue
            if progress:
                progress(f"  real chain: {target_name}, barrier {barrier:g} eV (R = {distance:.2f})")
            real, why = _chain_rates(pipeline, lambda m: target.system_at(distance, m), route, (proton, deuteron))
            grid = {n: Quantity(v, "angstrom") for n, v in
                    (("x_extent", float(np.abs(target.x).max())), ("distance_min", float(target.r.min())),
                     ("distance_max", float(target.r.max())))}
            for source_name, source in sources.items():
                def modelled(mass, source=source):
                    system = source.calibration.system(scan_distance=distance, with_uncertainty=False,
                                                       context={"particle_mass": (mass, "amu"),
                                                                "heavy_atom_mass": (float(target.system.param("heavy_atom_mass")), "amu")})
                    return system.evolve(parameters={**system.parameters, **grid})
                model, model_why = _chain_rates(pipeline, modelled, route, (proton, deuteron))
                rows.append(RateRow(
                    target_name, source_name, barrier, distance,
                    real[proton] if real else None, model[proton] if model else None,
                    real[proton] / real[deuteron] if real else None, model[proton] / model[deuteron] if model else None,
                    "; ".join(w for w in (("real chain: " + why) if why else "", ("model chain: " + model_why) if model_why else "") if w)))
    return rows


# =====================================================================================================================
# 3. few-shot learning curves
# =====================================================================================================================
@dataclass
class CurvePoint:
    n_distances: int                    # target distances used in the fit (0 = the prior alone)
    distances: list[float]
    rmse_ev: float                      # over the held-out distances' points up to the window
    barrier_error_mean_abs: float       # over held-out distances where both have a double well; NaN if none
    wells_wrong: int                    # held-out distances with the wrong number of wells
    barrier_missed: int
    note: str = ""                      # why there is no score (the fit was refused)


def split_columns(n_distances: int) -> tuple[list[int], list[int]]:
    """Training candidates are the even-numbered distances, held-out tests the odd-numbered ones: the test set is the same for
    every number of training distances, always interior to the training range when both ends are used, and never trained on."""
    return list(range(0, n_distances, 2)), list(range(1, n_distances, 2))


def _training_subset(pool: list[int], k: int) -> list[int]:
    if k == 1:
        return [pool[len(pool) // 2]]
    return sorted({pool[int(i)] for i in np.round(np.linspace(0, len(pool) - 1, k))})


def prior_from(calibration: Calibration, relative_sigma: float, floor: float = 1e-3) -> tuple[tuple[str, float, float], ...]:
    """A Gaussian prior on every fitted parameter: the calibration's values, each with sigma = `relative_sigma` times its size.
    The same sigma is used whatever the data say, so it is a decision, not a result: sweep it (see `learning_curve`). An asymmetric
    calibration's energy offset is among its parameters and gets a prior like the rest; a symmetric one has no offset to carry."""
    return tuple((n, float(v), max(relative_sigma * abs(v), floor)) for n, v in calibration.parameters.items())


def learning_curve(target: Reference, counts=(1, 2, 3, 4, 6), *, source: Calibration | None = None,
                   relative_sigma: float | None = None, settings: FitSettings | None = None, starts: int = 6) -> list[CurvePoint]:
    """Fit the model to only some of the target's heavy-atom distances and score it on the others.

    With `source` and `relative_sigma` the fit is prior-regularised toward the source's values; without, it starts from scratch.
    The coupling is defined at the source's reference distance in both cases, so the two are fitted in the same variables.
    With a source, `counts` may include 0, the prior alone (scored with the best constant shift on the held-out points, which
    flatters it: it is told the energy zero)."""
    settings = settings or FitSettings()
    r_ref = source.fixed["reference_distance"] if source is not None else (
        settings.reference_distance if settings.reference_distance is not None else float(0.5 * (target.r.min() + target.r.max())))
    pool, test = split_columns(len(target.r))
    if len(test) < 3:
        raise ValidationError("the target needs at least 6 heavy-atom distances for a learning curve")
    symmetric = target.symmetric
    # a symmetric target has no free offset, so an asymmetric source's offset prior has nothing to act on and is left out
    prior = () if source is None or relative_sigma is None else tuple(e for e in prior_from(source, relative_sigma)
                                                                       if not (symmetric and e[0] == "diabatic_offset"))
    fit_settings = settings if source is None or relative_sigma is None else FitSettings(
        **{**settings.__dict__, "prior": prior, "reference_distance": r_ref})
    Xt, Rt = np.meshgrid(target.x, target.r[test], indexing="ij")
    e_test = target.e[:, test]
    points = []
    for k in counts:
        shift = None
        used: list[int] = []
        if k == 0:
            if source is None:
                raise ValidationError("a learning curve point with no target distances needs a source calibration")
            model = predict_surface(source, target.x, target.r[test])
        else:
            used = _training_subset(pool, k)
            try:
                names, best, _, _, _ = _fit(target.x, target.r, target.e, fit_settings, symmetric, r_ref, columns=used, starts=starts)
            except ValidationError as error:                 # too few points to fit at all: record it, do not invent a number
                points.append(CurvePoint(k, [float(target.r[j]) for j in used], float("nan"), float("nan"), 0, 0, str(error)))
                continue
            params = dict(zip(names, best.x[:-1]))
            params.setdefault("diabatic_offset", 0.0)
            model = evb2d_energy(Xt, Rt, params, r_ref)
            shift = float(best.x[-1])
        score = _score(target.x, target.r[test], e_test, model, settings.window_ev, shift)
        points.append(CurvePoint(k, [float(target.r[j]) for j in used], score.rmse_ev[f"{settings.window_ev:g}"],
                                 score.barrier_error_mean_abs, score.wells_wrong, score.barrier_missed))
    return points


# =====================================================================================================================
# text
# =====================================================================================================================
def format_matrix(matrix: dict[tuple[str, str], Prediction], names: list[str], metric: str = "rmse", window: str | None = None) -> str:
    """A source x target table (row = whose parameters, column = whose real energies).
    metric = 'rmse' (eV, over the target's points up to its fit window, or up to `window` eV if given), 'barrier' (mean |model - reference| barrier, eV, at the
    target's double-well distances) or 'wells' (target distances where the model has the wrong number of wells)."""
    cell = max(max(len(n) for n in names) + 1, 10)
    corner = "source \\ target"
    lines = [f"{corner:<18}" + "".join(f"{n:>{cell}}" for n in names)]
    for s in names:
        cells = []
        for t in names:
            p = matrix[(s, t)]
            if metric == "rmse":
                cells.append(f"{p.rmse_ev.get(window, float('nan')) if window else p.rmse_window:>{cell}.3f}")
            elif metric == "barrier":
                cells.append(f"{p.barrier_error_mean_abs:>{cell}.3f}")
            elif metric == "wells":
                cells.append(f"{p.wells_wrong:>{cell}d}")
            else:
                raise ValueError(f"unknown metric '{metric}' (rmse, barrier, wells)")
        lines.append(f"{s:<18}" + "".join(cells))
    return "\n".join(lines)


def baselines(reference: Reference, window_ev: float | None = None) -> dict[str, float]:
    """What knowing nothing about the surface scores, so a transfer error can be read against it.
    constant_rmse: predicting one constant energy (the best one) for every point up to the reference's fit window: the spread of the surface itself.
    mean_barrier: predicting no barrier at all, at the reference's double-well distances."""
    inside = reference.e <= (reference.window_ev if window_ev is None else window_ev)
    barriers = [c.barrier_reference for c in _column_comparison(reference.x, reference.e, reference.e, reference.r)
                if c.barrier_reference is not None]
    return {"constant_rmse": float(np.std(reference.e[inside])), "mean_barrier": float(np.mean(barriers)) if barriers else float("nan")}


def format_baselines(references: dict[str, Reference], window_ev: float | None = None) -> str:
    """The `baselines` of every reference, and the window each is scored over, as rows under a source x target table."""
    cell = max(max(len(n) for n in references) + 1, 10)
    rows = {n: baselines(r, window_ev) for n, r in references.items()}
    return "\n".join([f"{'constant guess':<18}" + "".join(f"{rows[n]['constant_rmse']:>{cell}.3f}" for n in references) + "   (rmse of the best constant)",
                      f"{'no-barrier guess':<18}" + "".join(f"{rows[n]['mean_barrier']:>{cell}.3f}" for n in references) + "   (mean barrier)",
                      f"{'fit window (eV)':<18}" + "".join(f"{(r.window_ev if window_ev is None else window_ev):>{cell}g}" for r in references.values()) + "   (rmse is scored over it)"])


def format_own_fits(references: dict[str, Reference], matrix: dict[tuple[str, str], Prediction]) -> str:
    """One row per reference: how well its own calibration reproduces it, and the diagnostics that say where it does not."""
    lines = [f"{'':<18}{'window':>7}{'rmse<=0.5':>10}{'<=1.0':>8}{'<=window':>9}{'worst':>8}{'barrier err':>13}{'wells':>7}{'pinned':>8}{'poor':>6}{'chi2':>7}{'starts':>8}"]
    for name, ref in references.items():
        own, cal = matrix[(name, name)], ref.calibration
        starts = f"{cal.starts_agreeing}/{cal.settings['starts']}"
        lines.append(f"{name:<18}{own.window_ev:>7g}{own.rmse_ev['0.5']:>10.4f}{own.rmse_ev['1']:>8.4f}{own.rmse_window:>9.4f}{own.max_error_ev[f'{own.window_ev:g}']:>8.3f}"
                     f"{own.barrier_error_mean_abs:>13.3f}{own.wells_wrong:>7}{len(cal.pinned):>8}{len(cal.poorly_determined):>6}"
                     f"{cal.reduced_chi2:>7.1f}{starts:>8}")
    lines.append("(rmse and worst over the reference points in each window, eV, window = the fit window above the surface minimum; barrier err = mean "
                 "|model - reference| over distances with a double well, eV, barrier measured from the donor-side well;")
    lines.append(" wells = distances with the wrong number of wells; pinned = parameters on a bound; poor = parameters whose fit sigma "
                 "exceeds half their value)")
    return "\n".join(lines)


def format_report(references: dict[str, Reference], matrix: dict[tuple[str, str], Prediction] | None = None) -> str:
    """The own fits and the three cross-prediction tables."""
    names = list(references)
    matrix = matrix or transfer_matrix(references)
    return "\n".join([
        "Each calibration against its own reference", "", format_own_fits(references, matrix), "",
        "Cross-prediction: row = whose parameters, column = whose real energies. Parameters unchanged; only the energy zero is shifted.",
        "", "rmse (eV) over the target's points up to its fit window above its minimum (see the last row)", "", format_matrix(matrix, names, "rmse"),
        format_baselines(references), "",
        "mean |barrier error| (eV) at the target's double-well distances", "", format_matrix(matrix, names, "barrier"), "",
        "target distances where the model has the wrong number of wells", "", format_matrix(matrix, names, "wells"),
    ])
