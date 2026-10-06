"""Transfer analysis against synthetic references, where the right answers are known exactly.

The references are the model engine's own surfaces with chosen parameters, so a calibration can recover its reference exactly and
every comparison has a closed-form expectation. Real quantum-chemistry references are in test_transfer_real.py.
"""
import math
from dataclasses import replace

import numpy as np
import pytest

from substrate import ClassicalBackend, Pipeline, Quantity, ValidationError, default_registry
from substrate.calibration import FitSettings, calibrate_evb_2d, evb2d_energy
from substrate.engines.electronic import EVBFlexibleProtonTransferEngine
from substrate.transfer import (
    Reference, _score, _training_subset, cross_predict, distance_at_barrier, format_matrix, learning_curve, make_reference,
    parameter_table, predict_surface, prior_from, rate_comparison, split_columns, spread_ratio, transfer_matrix,
)

from conftest import evb2d

BACKEND = ClassicalBackend()
GRID = dict(x_extent=(0.65, "angstrom"), distance_min=(2.3, "angstrom"), distance_max=(3.0, "angstrom"), n_x=(21, "1"), n_r=(21, "1"))        # the model engine needs n_r >= 21
SETTINGS = FitSettings(reference_distance=2.65)           # one reference distance for every fit, so parameters share variables


def reference(name, window=None, **overrides):
    """A synthetic reference: a symmetric model surface with the given parameters, calibrated (over `window` eV if given)."""
    overrides.setdefault("diabatic_offset", 0.0)
    system = evb2d(**overrides)
    for key, value in GRID.items():
        system.parameters[key] = Quantity(*value)
    solved = EVBFlexibleProtonTransferEngine().solve(system, BACKEND)
    return make_reference(name, system, solved, SETTINGS if window is None else replace(SETTINGS, window_ev=window))


@pytest.fixture(scope="module")
def base():
    return reference("base")


@pytest.fixture(scope="module")
def stiffer():                       # a modest change: 8% in the Morse width
    return reference("stiffer", morse_alpha=2.2 * 1.08)


@pytest.fixture(scope="module")
def weaker_coupling():               # a higher barrier everywhere
    return reference("weaker_coupling", coupling=0.45)


# -- scoring: closed-form checks on the comparison itself -----------------------------------------------------------------------
def test_a_constant_offset_between_the_two_energy_zeros_costs_nothing():
    rng = np.random.default_rng(0)
    e = np.sort(rng.uniform(0.0, 1.4, (9, 4)), axis=0)
    score = _score(np.linspace(-1, 1, 9), np.linspace(2.4, 2.7, 4), e, e - 0.3, 1.5)
    assert score.shift_ev == pytest.approx(0.3, abs=1e-12)
    assert score.rmse_ev["1.5"] == pytest.approx(0.0, abs=1e-12)


def test_the_error_is_the_rms_of_what_a_constant_cannot_remove():
    e = np.tile(np.linspace(0.0, 1.0, 8)[:, None], (1, 3))
    wiggle = 0.02 * np.where(np.arange(8) % 2 == 0, 1.0, -1.0)[:, None] * np.ones((1, 3))          # +-20 meV, zero mean
    score = _score(np.linspace(-1, 1, 8), np.linspace(2.4, 2.7, 3), e, e + wiggle + 5.0, 1.5)
    assert score.shift_ev == pytest.approx(-5.0, abs=1e-12)
    assert score.rmse_ev["1.5"] == pytest.approx(0.02, abs=1e-12)
    assert score.max_error_ev["1.5"] == pytest.approx(0.02, abs=1e-12)


def test_only_points_inside_the_window_are_scored():
    e = np.array([[0.0], [0.4], [3.0], [0.4], [0.0]])
    model = e.copy()
    model[2, 0] += 1.0                                              # a huge error, but at 3 eV: outside every window
    score = _score(np.linspace(-1, 1, 5), np.array([2.5]), e, model, 1.5)
    assert score.rmse_ev["1.5"] == pytest.approx(0.0, abs=1e-12)


# -- cross-prediction -----------------------------------------------------------------------------------------------------------
def test_a_calibration_predicts_its_own_reference(base):
    prediction = cross_predict(base.calibration, base)
    assert prediction.rmse_ev["1.5"] < 1e-5 and prediction.barrier_error_mean_abs < 1e-4 and prediction.wells_wrong == 0
    assert prediction.shift_ev == pytest.approx(base.calibration.shift_ev, abs=1e-4)             # the same energy zero the fit found


def test_the_prediction_error_grows_with_how_different_the_reference_is(base, stiffer):
    slightly = cross_predict(base.calibration, stiffer).rmse_ev["1.5"]
    much = cross_predict(base.calibration, reference("much", morse_alpha=2.2 * 1.4)).rmse_ev["1.5"]
    assert 1e-3 < slightly < much


def test_a_model_with_weaker_coupling_overestimates_barriers(base, weaker_coupling):
    errors = cross_predict(weaker_coupling.calibration, base).barrier_errors
    assert errors and all(error > 0.005 for error in errors)          # weaker coupling = a higher barrier at every double-well distance
    assert cross_predict(base.calibration, weaker_coupling).barrier_errors and all(
        error < -0.005 for error in cross_predict(base.calibration, weaker_coupling).barrier_errors)


def test_the_transfer_matrix_has_every_pair_and_a_best_diagonal(base, stiffer, weaker_coupling):
    refs = {"base": base, "stiffer": stiffer, "weaker": weaker_coupling}
    matrix = transfer_matrix(refs)
    assert set(matrix) == {(s, t) for s in refs for t in refs}
    for target in refs:                                              # no calibration predicts a reference better than its own
        own = matrix[(target, target)].rmse_ev["1.5"]
        assert all(matrix[(s, target)].rmse_ev["1.5"] >= own - 1e-9 for s in refs)
    table = format_matrix(matrix, list(refs))
    assert table.splitlines()[0].split()[-3:] == ["base", "stiffer", "weaker"] and len(table.splitlines()) == 4


def test_predict_surface_is_the_closed_form_model_without_the_energy_zero(base):
    x, r = base.x, base.r
    X, R = np.meshgrid(x, r, indexing="ij")
    p = {**base.calibration.parameters, **base.calibration.fixed}
    assert predict_surface(base.calibration, x, r) == pytest.approx(evb2d_energy(X, R, p, 2.65), abs=1e-12)


def test_the_parameter_table_lists_every_fitted_parameter_for_every_reference(base, stiffer):
    table = parameter_table({"base": base, "stiffer": stiffer})
    assert len(table) == 8 and set(table["morse_alpha"]) == {"base", "stiffer"}
    assert table["morse_alpha"]["stiffer"][0] / table["morse_alpha"]["base"][0] == pytest.approx(1.08, rel=0.02)
    assert spread_ratio([2.0, 3.0, 4.0]) == pytest.approx(2.0) and math.isnan(spread_ratio([1.0])) and math.isnan(spread_ratio([0.0, 1.0]))


# -- the distances where the downstream comparison is made -------------------------------------------------------------------------
def test_the_distance_for_a_stated_barrier_is_where_the_reference_has_that_barrier(base):
    for barrier in (0.1, 0.4):
        distance = distance_at_barrier(base, barrier)
        x = np.linspace(-0.65, 0.65, 801)
        column = evb2d_energy(x, distance, {**base.calibration.parameters, **base.calibration.fixed}, 2.65)
        from substrate.pes import locate_wells
        wells = locate_wells(column)
        assert wells and column[wells[1]] - column[wells[0]] == pytest.approx(barrier, abs=0.03)    # the grid is coarse, so not exact
    assert distance_at_barrier(base, 25.0) is None and distance_at_barrier(base, 1e-9) is None


# -- priors in the fit ---------------------------------------------------------------------------------------------------------------
def test_a_tight_prior_holds_a_parameter_where_the_prior_says(base):
    wrong = ("morse_depth", 7.0, 1e-4)
    cal = calibrate_evb_2d(base_target(), FitSettings(reference_distance=2.65, prior=(wrong,)), cross_validate=False)
    assert cal.parameters["morse_depth"] == pytest.approx(7.0, rel=1e-3)
    assert cal.rmse_ev["1.5"] > 0.01                                  # the data disagree with it, and the fit says so


def test_a_loose_prior_leaves_the_data_in_charge(base):
    loose = prior_from(base.calibration, relative_sigma=1e3)
    free = calibrate_evb_2d(base_target(), FitSettings(reference_distance=2.65), cross_validate=False)
    held = calibrate_evb_2d(base_target(), FitSettings(reference_distance=2.65, prior=tuple((n, 1.3 * v, s) for n, v, s in loose)),
                            cross_validate=False)
    for name, value in free.parameters.items():
        assert held.parameters[name] == pytest.approx(value, rel=5e-3)


def test_the_priors_penalty_is_not_counted_as_misfit(base):
    cal = calibrate_evb_2d(base_target(), FitSettings(reference_distance=2.65, prior=(("morse_depth", 7.0, 1e-3),)), cross_validate=False)
    x, r, e = base.x, base.r, base.e
    X, R = np.meshgrid(x, r, indexing="ij")
    inside = (e <= 1.5) & (x >= -1e-12)[:, None]
    residual = (evb2d_energy(X, R, {**cal.parameters, **cal.fixed}, 2.65) + cal.shift_ev - e)[inside] / 0.01
    assert cal.reduced_chi2 == pytest.approx((residual**2).sum() / (inside.sum() - cal.n_free), rel=1e-6)
    assert cal.settings["prior"] == [["morse_depth", 7.0, 1e-3]]


def test_a_prior_on_something_that_is_not_fitted_is_refused(base):
    with pytest.raises(ValidationError, match="not a free parameter"):
        calibrate_evb_2d(base_target(), FitSettings(prior=(("diabatic_offset", 0.0, 0.1),)), cross_validate=False)   # symmetric target
    with pytest.raises(ValidationError, match="not a free parameter"):
        calibrate_evb_2d(base_target(), FitSettings(prior=(("nonsense", 0.0, 0.1),)), cross_validate=False)
    with pytest.raises(ValidationError, match="must be positive"):
        calibrate_evb_2d(base_target(), FitSettings(prior=(("morse_depth", 4.6, 0.0),)), cross_validate=False)


def base_target():
    system = evb2d(diabatic_offset=0.0)
    for key, value in GRID.items():
        system.parameters[key] = Quantity(*value)
    return EVBFlexibleProtonTransferEngine().solve(system, BACKEND)


# -- learning curves --------------------------------------------------------------------------------------------------------------------
def test_training_and_test_distances_never_overlap_and_the_test_set_is_fixed():
    pool, test = split_columns(11)
    assert pool == [0, 2, 4, 6, 8, 10] and test == [1, 3, 5, 7, 9] and not set(pool) & set(test)
    assert _training_subset(pool, 1) == [6] and _training_subset(pool, 2) == [0, 10] and _training_subset(pool, 6) == pool
    assert all(len(_training_subset(pool, k)) == k for k in range(1, 7))


def test_a_prior_at_the_truth_predicts_unseen_distances_from_one_distance(base):
    curve = learning_curve(base, counts=(0, 1), source=base.calibration, relative_sigma=0.2, settings=SETTINGS)
    assert curve[0].n_distances == 0 and curve[0].distances == []
    assert curve[0].rmse_ev < 1e-4 and curve[1].rmse_ev < 5e-3 and curve[1].wells_wrong == 0


def test_a_wrong_prior_hurts_in_proportion_to_how_tightly_it_is_held(base):
    wrong = reference("wrong", morse_depth=4.6 * 1.6, morse_alpha=2.2 * 0.7).calibration
    errors = [learning_curve(base, counts=(6,), source=wrong, relative_sigma=sigma, settings=SETTINGS)[0].rmse_ev
              for sigma in (0.01, 0.1, 10.0)]
    assert errors[0] > 0.02 > 0.005 > errors[1] > errors[2] and errors[2] < 1e-4        # plenty of target data overrules a loose prior


def test_without_a_source_the_fit_starts_from_scratch_and_refusals_are_recorded_not_invented(base):
    curve = learning_curve(base, counts=(1, 6), settings=SETTINGS)
    assert curve[1].rmse_ev < 5e-3 and curve[1].n_distances == 6        # enough noise-free data: the model's own surface is recovered
    one = curve[0]
    assert math.isnan(one.rmse_ev) == bool(one.note) or one.note == ""     # a refused fit carries its reason; a fitted one carries a score
    with pytest.raises(ValidationError, match="needs a source"):
        learning_curve(base, counts=(0,), settings=SETTINGS)


def test_a_learning_curve_needs_enough_distances():
    solved = base_target()
    keep = [0, 5, 10, 15, 20]                                                      # five distances: too few to hold any out sensibly
    thin = solved.evolve(observables={**solved.observables, "surface_r": Quantity(solved.obs("surface_r")[keep], "angstrom"),
                                      "surface_energy": Quantity(solved.obs("surface_energy")[:, keep], "eV")})
    tiny = make_reference("tiny", evb2d(), thin, FitSettings(reference_distance=2.65))
    with pytest.raises(ValidationError, match="at least 6"):
        learning_curve(tiny, counts=(1,), settings=SETTINGS)


# -- downstream rates -----------------------------------------------------------------------------------------------------------------
def test_the_chain_on_a_calibration_reproduces_its_own_references_rates(base):
    rows = rate_comparison({"base": base}, {"base": base}, Pipeline(default_registry()), barriers=(0.25, 0.5))
    assert len(rows) == 2 and all(row.k_real and row.k_model for row in rows)
    for row in rows:
        assert 0.9 < row.ratio < 1.1 and row.kie_model == pytest.approx(row.kie_real, rel=0.1)
        assert row.barrier_ev in (0.25, 0.5) and 2.3 <= row.distance <= 3.0


def test_a_calibration_with_a_higher_barrier_gives_slower_rates_and_larger_isotope_effects(base, weaker_coupling):
    barrier_error = np.mean(cross_predict(weaker_coupling.calibration, base).barrier_errors)          # ~0.09 eV higher
    classical = math.exp(-barrier_error / 0.025852)                                                    # what a classical rate would lose
    rows = rate_comparison({"weaker": weaker_coupling}, {"base": base}, Pipeline(default_registry()), barriers=(0.2, 0.4, 0.8))
    assert len(rows) == 3
    for row in rows:
        assert row.ratio < 0.7 and row.kie_model > row.kie_real
        assert row.ratio > 3 * classical            # the chain is tunnelling-dominated here: far less sensitive to the barrier than TST


def test_a_barrier_the_target_never_reaches_is_reported_not_extrapolated(base):
    rows = rate_comparison({"base": base}, {"base": base}, Pipeline(default_registry()), barriers=(50.0,))
    assert rows[0].k_real is None and rows[0].k_model is None and "not reached" in rows[0].note


# -- the command line ---------------------------------------------------------------------------------------------------------------------
def test_the_transfer_command_tabulates_every_calibration_against_every_reference(tmp_path, capsys):
    import copy
    import json

    import yaml

    from conftest import EXPERIMENTS
    from substrate.cli import main

    base = yaml.safe_load((EXPERIMENTS / "proton_transfer_molecular.yaml").read_text())       # an asymmetric reference (offset 50 meV)
    paths = []
    for name, coupling in (("REF-A", 0.6), ("REF-B", 0.5)):
        spec = copy.deepcopy(base)
        spec["experiment"]["id"] = name
        spec["experiment"]["system"]["parameters"]["coupling"]["value"] = coupling
        paths.append(tmp_path / f"{name}.yaml")
        paths[-1].write_text(yaml.safe_dump(spec))
    saved = tmp_path / "transfer.json"
    assert main(["transfer", *map(str, paths), "--json", str(saved)]) == 0
    out = capsys.readouterr().out
    assert "REF-A" in out and "REF-B" in out and "Cross-prediction" in out and "2.650 angstrom" in out     # the mean of the mid-range
    data = json.loads(saved.read_text())
    assert set(data) == {"REF-A -> REF-A", "REF-A -> REF-B", "REF-B -> REF-A", "REF-B -> REF-B"}
    assert data["REF-A -> REF-A"]["rmse_ev"]["1.5"] < 1e-6 and data["REF-B -> REF-B"]["rmse_ev"]["1.5"] < 1e-6      # exact recovery
    assert 0.01 < data["REF-A -> REF-B"]["rmse_ev"]["1.5"] < 0.2                                                    # a 17% weaker coupling shows


def test_the_transfer_command_refuses_duplicate_ids_and_non_surfaces(tmp_path, capsys):
    from conftest import EXPERIMENTS
    from substrate.cli import main
    twice = str(EXPERIMENTS / "proton_transfer_molecular.yaml")
    assert main(["transfer", twice, twice]) == 1 and "two experiments are called" in capsys.readouterr().err
    assert main(["transfer", str(EXPERIMENTS / "proton_transfer.yaml")]) == 1                # a quantum-scale system has no 2D surface


# -- baselines ---------------------------------------------------------------------------------------------------------------------------
def test_the_no_model_baselines_are_what_a_constant_and_a_barrierless_guess_score(base):
    from substrate.transfer import baselines, format_baselines
    b = baselines(base)
    flat = _score(base.x, base.r, base.e, np.zeros_like(base.e), 1.5)                  # a model that predicts nothing but its energy zero
    assert b["constant_rmse"] == pytest.approx(flat.rmse_ev["1.5"], rel=1e-12)
    errors = [c.barrier_reference for c in flat.columns if c.barrier_reference is not None]       # the flat model has no barrier anywhere
    assert errors and b["mean_barrier"] == pytest.approx(np.mean(errors), rel=1e-12)
    assert all(c.barrier_model is None for c in flat.columns)
    text = format_baselines({"base": base})
    assert "constant guess" in text and "no-barrier guess" in text and f"{b['constant_rmse']:.3f}" in text and f"{b['mean_barrier']:.3f}" in text


# -- details that the synthetic comparisons above cannot see by themselves ---------------------------------------------------------------------
def test_the_model_chain_runs_on_the_targets_grid_and_masses_not_the_sources(base):
    """Each source was fitted on its own grid; the comparison must integrate over the target's, and with the target's heavy atom."""
    system = evb2d(diabatic_offset=0.0, heavy=14.0)
    for key, value in dict(x_extent=(0.8, "angstrom"), distance_min=(2.2, "angstrom"), distance_max=(3.1, "angstrom"),
                           n_x=(21, "1"), n_r=(21, "1")).items():
        system.parameters[key] = Quantity(*value)
    wide = make_reference("wide", system, EVBFlexibleProtonTransferEngine().solve(system, BACKEND), SETTINGS)

    class Spy:
        def __init__(self):
            self.inner, self.seen = Pipeline(default_registry()), []

        def run(self, system, route, **kw):
            self.seen.append(system)
            return self.inner.run(system, route, **kw)

    spy = Spy()
    rate_comparison({"base": base}, {"wide": wide}, spy, barriers=(0.4,))
    modelled = [s for s in spy.seen if s.name.endswith("(calibrated model)")]
    assert len(modelled) == 2 and {s.param("particle_mass") for s in modelled} == {1.007276, 2.013553}
    for s in modelled:
        assert (s.param("x_extent"), s.param("distance_min"), s.param("distance_max")) == (0.8, 2.2, 3.1)      # the target's grid
        assert s.param("heavy_atom_mass") == 14.0                                                              # the target's atom
    real = [s for s in spy.seen if not s.name.endswith("(calibrated model)")]
    assert len(real) == 2 and all(s.param("scan_distance") == real[0].param("scan_distance") for s in real)


def test_the_few_shot_score_is_the_fits_own_prediction_scored_independently(base):
    """The learning curve's number must equal what you get fitting the same columns by hand and scoring the held-out ones."""
    from substrate.calibration import _fit
    wrong = reference("wrong", morse_depth=4.6 * 1.3).calibration
    curve = learning_curve(base, counts=(3,), source=wrong, relative_sigma=0.3, settings=SETTINGS)[0]
    pool, test = split_columns(len(base.r))
    used = _training_subset(pool, 3)
    prior = SETTINGS.__class__(**{**SETTINGS.__dict__, "prior": prior_from(wrong, 0.3), "reference_distance": wrong.fixed["reference_distance"]})
    names, best, _, _, _ = _fit(base.x, base.r, base.e, prior, True, wrong.fixed["reference_distance"], columns=used, starts=6)
    X, R = np.meshgrid(base.x, base.r[test], indexing="ij")
    model = evb2d_energy(X, R, {**dict(zip(names, best.x[:-1])), "diabatic_offset": 0.0}, wrong.fixed["reference_distance"]) + best.x[-1]
    held_out = base.e[:, test]
    inside = held_out <= 1.5
    assert curve.rmse_ev == pytest.approx(float(np.sqrt(((model - held_out)[inside] ** 2).mean())), rel=1e-9)
    assert curve.distances == [float(base.r[j]) for j in used]


def test_the_prior_is_the_sources_values_with_a_relative_width_and_a_floor(base):
    prior = {name: (mean, sigma) for name, mean, sigma in prior_from(base.calibration, 0.25, floor=1e-3)}
    assert set(prior) == set(base.calibration.parameters)
    for name, value in base.calibration.parameters.items():
        assert prior[name][0] == value and prior[name][1] == pytest.approx(max(0.25 * abs(value), 1e-3))
    zero = prior_from(type("C", (), {"parameters": {"coupling_decay": 0.0}})(), 0.25, floor=1e-3)
    assert zero == (("coupling_decay", 0.0, 1e-3),)                       # a parameter at zero still gets a usable width


def test_a_prior_lets_a_fit_proceed_where_the_data_alone_could_not(base):
    thin = FitSettings(reference_distance=2.65, window_ev=0.12)            # so few points inside the window that nine parameters are not determined
    scratch = learning_curve(base, counts=(1,), settings=thin)[0]
    assert scratch.note and "too few" in scratch.note and math.isnan(scratch.rmse_ev)
    with_prior = learning_curve(base, counts=(1,), source=base.calibration, relative_sigma=0.2, settings=thin)[0]
    assert not with_prior.note and with_prior.rmse_ev < 0.05


# -- gaps found by mutation testing ----------------------------------------------------------------------------------------------------------
def test_the_worst_error_is_the_largest_absolute_error_in_each_window():
    e = np.tile(np.linspace(0.0, 1.2, 6)[:, None], (1, 2))                       # 0, .24, .48, .72, .96, 1.2
    error = np.array([0.1, -0.1, 0.0, 0.0, 0.0, 0.0])[:, None] * np.ones((1, 2))      # zero mean, so the shift leaves it alone
    score = _score(np.linspace(-1, 1, 6), np.array([2.4, 2.6]), e, e + error, 1.5)
    assert score.shift_ev == pytest.approx(0.0, abs=1e-12)
    assert score.max_error_ev["0.5"] == pytest.approx(0.1) and score.max_error_ev["1.5"] == pytest.approx(0.1)
    assert score.rmse_ev["1.5"] == pytest.approx(math.sqrt(2 * 0.01 / 6))
    assert score.rmse_ev["0.5"] == pytest.approx(math.sqrt(2 * 0.01 / 3))       # only the first three points are inside 0.5 eV


def test_barrier_errors_keep_their_sign_but_their_mean_is_of_the_magnitudes():
    from substrate.calibration import ColumnComparison
    from substrate.transfer import Prediction
    columns = [ColumnComparison(2.5, 2, 2, 0.30, 0.40, 0.3, 0.3), ColumnComparison(2.6, 2, 2, 0.50, 0.40, 0.3, 0.3),
               ColumnComparison(2.4, 1, 1, None, None, None, None)]
    p = Prediction("a", "b", 0.0, {}, {}, columns)
    assert p.barrier_errors == pytest.approx([0.1, -0.1]) and p.barrier_error_mean_abs == pytest.approx(0.1)
    assert p.wells_wrong == 0 and p.barrier_missed == 0
    bad = Prediction("a", "b", 0.0, {}, {}, columns + [ColumnComparison(2.7, 2, 1, 0.40, None, 0.3, None)])
    assert bad.wells_wrong == 1 and bad.barrier_missed == 1 and len(bad.barrier_errors) == 2
    assert math.isnan(Prediction("a", "b", 0.0, {}, {}, columns[2:]).barrier_error_mean_abs)              # no double well: no number


def test_a_model_with_no_barrier_anywhere_misses_every_barrier_the_reference_has(base):
    flat = _score(base.x, base.r, base.e, np.zeros_like(base.e), 1.5)
    n_double = sum(c.barrier_reference is not None for c in flat.columns)
    assert n_double >= 5 and flat.barrier_missed == n_double and flat.wells_wrong >= n_double
    assert cross_predict(base.calibration, base).barrier_missed == 0


def test_an_asymmetric_reference_is_predicted_through_its_fitted_offset():
    asym = reference("asym", diabatic_offset=0.05)
    assert asym.calibration.parameters["diabatic_offset"] == pytest.approx(0.05, abs=2e-3) and not asym.symmetric
    p = {**asym.calibration.parameters, **asym.calibration.fixed}
    X, R = np.meshgrid(asym.x, asym.r, indexing="ij")
    assert predict_surface(asym.calibration, asym.x, asym.r) == pytest.approx(evb2d_energy(X, R, p, 2.65), abs=1e-12)
    assert cross_predict(asym.calibration, asym).rmse_ev["1.5"] < 1e-4


def test_predictions_carry_the_names_of_their_source_and_target(base, stiffer):
    direct = cross_predict(base.calibration, stiffer)
    assert direct.source == base.calibration.target_name and direct.target == "stiffer"
    matrix = transfer_matrix({"base": base, "stiffer": stiffer})
    for (source, target), prediction in matrix.items():
        assert (prediction.source, prediction.target) == (source, target)
    assert matrix[("base", "stiffer")].rmse_ev["1.5"] == pytest.approx(direct.rmse_ev["1.5"])
    assert matrix[("base", "stiffer")].rmse_ev["1.5"] != pytest.approx(matrix[("stiffer", "base")].rmse_ev["1.5"], rel=1e-3)   # not symmetric


def test_the_order_of_the_distances_does_not_change_where_a_barrier_is_reached(base):
    import dataclasses
    reversed_ = dataclasses.replace(base, r=base.r[::-1], e=base.e[:, ::-1])
    for barrier in (0.1, 0.4):
        assert distance_at_barrier(reversed_, barrier) == distance_at_barrier(base, barrier)


def test_a_source_fitted_at_another_reference_distance_is_primed_in_its_own_variables(base):
    other = calibrate_evb_2d(base_target(), FitSettings(reference_distance=2.4), cross_validate=False)      # coupling defined at 2.4, not 2.65
    assert other.fixed["reference_distance"] == 2.4
    curve = learning_curve(base, counts=(1,), source=other, relative_sigma=0.1, settings=SETTINGS)[0]
    assert curve.rmse_ev < 5e-3                                                                           # the coupling was not misread


def test_a_source_without_a_prior_width_only_supplies_the_zero_shot_point(base):
    with_source = learning_curve(base, counts=(0, 3), source=base.calibration, relative_sigma=None, settings=SETTINGS)
    scratch = learning_curve(base, counts=(3,), settings=SETTINGS)
    assert with_source[0].n_distances == 0 and with_source[1].rmse_ev == pytest.approx(scratch[0].rmse_ev, rel=1e-9)


def test_a_reference_is_labelled_by_the_template_and_method_recorded_on_the_solved_surface(base):
    solved = base_target().evolve(structure={"molecule": {"template": "zundel_cation"}, "method": {"theory": "hf", "basis": "6-31g*"}})
    ref = make_reference("z", evb2d(), solved, SETTINGS)
    assert (ref.molecule, ref.method, ref.label()) == ("zundel_cation", "hf/6-31g*", "zundel_cation hf/6-31g*")
    assert (base.molecule, base.method) == ("custom", "?/?")


# -- a reference declares its own fit window -----------------------------------------------------------------------------------------------------------
def _raw(hints):
    return {"experiment": {"id": "x", "system": {"name": "n", "scale": "electronic_structure", "kind": "k"},
                           **({} if hints is None else {"calibration": hints})}}


def test_an_experiment_file_can_declare_its_fit_window_and_nothing_else():
    from substrate.experiment import parse_experiment
    from substrate.transfer import fit_settings
    assert parse_experiment(_raw(None)).notes == {}
    declared = parse_experiment(_raw({"window_ev": 2.5}))
    assert declared.notes == {"calibration": {"window_ev": 2.5}}
    tuned = fit_settings(declared, FitSettings(sigma_ev=0.02, starts=5))
    assert (tuned.window_ev, tuned.sigma_ev, tuned.starts) == (2.5, 0.02, 5)                      # only the window changes
    assert fit_settings(parse_experiment(_raw(None)), FitSettings()).window_ev == 1.5
    assert fit_settings(parse_experiment(_raw({})), FitSettings(window_ev=1.2)).window_ev == 1.2
    for bad in ({"window_ev": 0}, {"window_ev": -1}, {"window_ev": "wide"}, {"window_ev": True}, {"sigma": 1}, [2.5], "x"):
        with pytest.raises(ValidationError, match="takes only a positive"):
            parse_experiment(_raw(bad))


def test_each_target_is_scored_over_its_own_fit_window(base):
    wide = reference("wide", window=2.5, morse_alpha=2.2 * 1.1)
    assert (base.window_ev, wide.window_ev) == (1.5, 2.5)
    matrix = transfer_matrix({"base": base, "wide": wide})
    assert matrix[("base", "base")].window_ev == 1.5 and matrix[("base", "wide")].window_ev == 2.5
    p = matrix[("base", "wide")]
    assert p.rmse_window == p.rmse_ev["2.5"] != p.rmse_ev["1.5"]                                    # the headline number is the 2.5 eV one
    model = predict_surface(base.calibration, wide.x, wide.r)                                      # and it is the independent calculation
    inside = wide.e <= 2.5
    shift = (wide.e[inside] - model[inside]).mean()
    assert p.rmse_window == pytest.approx(np.sqrt(((model + shift - wide.e)[inside] ** 2).mean()), rel=1e-12)
    assert transfer_matrix({"base": base, "wide": wide}, 1.0)[("base", "wide")].window_ev == 1.0   # an explicit window overrides every target's own


def test_the_baselines_and_the_report_use_each_references_window(base):
    from substrate.transfer import baselines, format_baselines, format_own_fits
    wide = reference("wide", window=2.5)
    assert baselines(wide)["constant_rmse"] == pytest.approx(np.std(wide.e[wide.e <= 2.5]))
    assert baselines(wide)["constant_rmse"] > baselines(base)["constant_rmse"]                      # more of the wall is inside a wider window
    refs = {"base": base, "wide": wide}
    text = format_baselines(refs)
    assert "fit window (eV)" in text and text.splitlines()[-1].split()[3:5] == ["1.5", "2.5"]
    table = format_own_fits(refs, transfer_matrix(refs)).splitlines()
    assert table[0].split()[:2] == ["window", "rmse<=0.5"] and [row.split()[:2] for row in table[1:3]] == [["base", "1.5"], ["wide", "2.5"]]


# -- an asymmetric source's offset is part of its prior ----------------------------------------------------------------------------------------------------
def test_a_value_at_zero_has_no_spread_ratio():
    from substrate.transfer import spread_ratio
    assert math.isnan(spread_ratio([1e-19, 1.0, 2.0])) and spread_ratio([0.5, 1.0, 2.0]) == pytest.approx(4.0)       # a parameter on a bound at 0
    assert spread_ratio([5e-4, 1.0]) == pytest.approx(2000.0)                                                              # a small but real value still has a ratio
    assert math.isnan(spread_ratio([1e-5, 100.0]))                                                                         # 7 decades apart: the small one is zero for this purpose
    assert math.isnan(spread_ratio([1e-17, 6e-15]))                                                                        # several values all at ~0: no ratio either


def test_the_prior_of_an_asymmetric_source_includes_its_energy_offset():
    asym = reference("asym", diabatic_offset=0.05)
    prior = {name: (mean, sigma) for name, mean, sigma in prior_from(asym.calibration, 0.2)}
    assert prior["diabatic_offset"] == (pytest.approx(0.05, abs=1e-3), pytest.approx(max(0.2 * abs(asym.calibration.parameters["diabatic_offset"]), 1e-3)))
    assert "diabatic_offset" not in dict((n, 0) for n, _, _ in prior_from(reference("sym").calibration, 0.2))        # a symmetric source has none to carry


def test_the_offset_prior_is_applied_to_an_asymmetric_target_and_dropped_for_a_symmetric_one():
    import copy
    asym = reference("asym", diabatic_offset=0.05)
    shifted = copy.deepcopy(asym.calibration)
    shifted.parameters["diabatic_offset"] += 0.3                                                   # everything right except the offset
    tight = learning_curve(asym, counts=(1,), source=shifted, relative_sigma=0.01, settings=SETTINGS)[0]
    loose = learning_curve(asym, counts=(1,), source=shifted, relative_sigma=100.0, settings=SETTINGS)[0]
    assert tight.rmse_ev > 0.05 > 0.005 > loose.rmse_ev                                           # a tight prior holds the wrong offset; a loose one lets one distance correct it
    onto_symmetric = learning_curve(reference("sym2"), counts=(1,), source=asym.calibration, relative_sigma=0.3, settings=SETTINGS)[0]
    assert onto_symmetric.note == "" and onto_symmetric.rmse_ev < 1e-6                           # no free offset there, so no prior on it and no error


def test_the_parameter_table_has_an_offset_row_only_when_a_reference_needs_one(base):
    asym = reference("asym", diabatic_offset=0.05)
    assert "diabatic_offset" not in parameter_table({"base": base})
    table = parameter_table({"base": base, "asym": asym})
    assert len(table) == 9 and table["diabatic_offset"]["base"] == (None, None)                        # fixed at zero for a symmetric reference
    assert table["diabatic_offset"]["asym"][0] == pytest.approx(0.05, abs=2e-3) and table["diabatic_offset"]["asym"][1] is not None
