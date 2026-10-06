"""Calibration against synthetic references, where the true parameters are known.

Real quantum chemistry is the intended reference (see test_calibration_real.py); here the reference is the model engine's own
surface, so the right answer is known exactly and the machinery (recovery, honest uncertainty, bound detection, joint sampling,
the generated files) can be tested without PySCF.
"""
import json
import math

import numpy as np
import pytest

from substrate import ClassicalBackend, Pipeline, Quantity, Scale, ScientificSystem, SubstrateError, ValidationError, default_registry
from substrate.calibration import (
    Calibration, FitSettings, calibrate_double_well, calibrate_evb_2d, evb2d_energy, leave_one_distance_out,
)
from substrate.engines.electronic import EVBFlexibleProtonTransferEngine
from substrate.engines.quantum import double_well

from conftest import EXPERIMENTS, evb2d

BACKEND = ClassicalBackend()
TRUTH = dict(morse_depth=4.6, morse_alpha=2.2, morse_r_eq=0.96, coupling=0.6, coupling_decay=3.0, oo_depth=0.4,
             oo_alpha=2.5, oo_equilibrium=2.7)
TRUE_REFERENCE_DISTANCE = 2.5                      # where evb2d() defines its coupling


def synthetic(noise=0.0, seed=0, **overrides):
    """A solved model system (a reference surface) built from known parameters, optionally with noise."""
    system = evb2d(**overrides)
    for name, (value, unit) in dict(x_extent=(0.65, "angstrom"), distance_min=(2.3, "angstrom"), distance_max=(3.0, "angstrom"),
                                    n_x=(41, "1"), n_r=(21, "1")).items():
        system.parameters[name] = Quantity(value, unit)
    solved = EVBFlexibleProtonTransferEngine().solve(system, BACKEND)
    if noise:
        e = solved.obs("surface_energy")
        e = e + np.random.default_rng(seed).normal(0.0, noise, e.shape)
        solved = solved.evolve(observables={**solved.observables, "surface_energy": Quantity(e, "eV")})
    return solved


def coupling_at(r_ref, value=0.6, decay=3.0):
    """The true coupling re-expressed at another reference distance (the coupling is defined at `reference_distance`)."""
    return value * math.exp(-decay * (r_ref - TRUE_REFERENCE_DISTANCE))


# -- the model ------------------------------------------------------------------------------------------------------------
def test_the_closed_form_model_equals_the_engines_diagonalisation():
    solved = synthetic(oo_depth=0.55, coupling=0.9, diabatic_offset=0.03)
    x, r, e = solved.obs("surface_x"), solved.obs("surface_r"), solved.obs("surface_energy")
    X, R = np.meshgrid(x, r, indexing="ij")
    p = {**{k: v for k, v in TRUTH.items() if k not in ("oo_depth", "coupling")}, "oo_depth": 0.55, "coupling": 0.9,
         "diabatic_offset": 0.03}
    assert evb2d_energy(X, R, p, TRUE_REFERENCE_DISTANCE) == pytest.approx(e, abs=1e-10)


# -- recovery -----------------------------------------------------------------------------------------------------------------
def test_a_noise_free_reference_is_recovered_exactly():
    cal = calibrate_evb_2d(synthetic(diabatic_offset=0.0), cross_validate=False)
    r_ref = cal.fixed["reference_distance"]
    assert r_ref == pytest.approx(2.65)                                           # the middle of the distance range
    for name, truth in TRUTH.items():
        expected = coupling_at(r_ref) if name == "coupling" else truth
        assert cal.parameters[name] == pytest.approx(expected, rel=2e-3), name
    assert cal.rmse_ev["1.5"] < 1e-5 and cal.max_error_ev["1.5"] < 1e-4
    assert cal.pinned == {} and cal.starts_agreeing >= 8
    assert all(c.barrier_reference is None or abs(c.barrier_model - c.barrier_reference) < 1e-4 for c in cal.columns)


def test_the_reported_uncertainties_are_honest_on_noisy_data():
    # 5 meV of noise: the fitted values should sit within a few sigma of the truth, across independent noise realisations
    zs = []
    for seed in range(4):
        cal = calibrate_evb_2d(synthetic(noise=0.005, seed=seed, diabatic_offset=0.0), cross_validate=False)
        r_ref = cal.fixed["reference_distance"]
        for name, truth in TRUTH.items():
            expected = coupling_at(r_ref) if name == "coupling" else truth
            zs.append((cal.parameters[name] - expected) / cal.sigma[name])
    assert np.max(np.abs(zs)) < 4.0 and np.sqrt(np.mean(np.square(zs))) < 1.6          # neither overconfident nor wild


def test_a_noise_free_reference_does_not_collapse_the_reported_uncertainty():
    cal = calibrate_evb_2d(synthetic(diabatic_offset=0.0), cross_validate=False)
    assert cal.reduced_chi2 < 1e-6                                                 # the fit is essentially exact ...
    # ... yet the uncertainty is set by the assumed 10 meV tolerance, not by the (zero) misfit. Without that floor on the scale it
    # would shrink to ~1e-10 and claim the parameters are known perfectly.
    assert cal.sigma["morse_depth"] > 0.01 and cal.sigma["morse_alpha"] > 0.003 and cal.sigma["oo_equilibrium"] > 5e-4


def test_an_asymmetric_reference_gets_a_free_offset_and_a_symmetric_one_does_not():
    asymmetric = calibrate_evb_2d(synthetic(diabatic_offset=0.05), cross_validate=False)
    assert asymmetric.parameters["diabatic_offset"] == pytest.approx(0.05, abs=2e-4)
    assert "diabatic_offset" not in asymmetric.fixed and not asymmetric.settings["symmetric_reference"]
    symmetric = calibrate_evb_2d(synthetic(diabatic_offset=0.0), cross_validate=False)
    assert symmetric.fixed["diabatic_offset"] == 0.0 and symmetric.settings["symmetric_reference"]
    assert "diabatic_offset" not in symmetric.parameters


def test_settings_control_what_is_fitted():
    target = synthetic(diabatic_offset=0.0)
    wide = calibrate_evb_2d(target, FitSettings(window_ev=3.0), cross_validate=False)
    narrow = calibrate_evb_2d(target, FitSettings(window_ev=0.6), cross_validate=False)
    assert wide.n_points > narrow.n_points                                           # a wider window includes more points
    moved = calibrate_evb_2d(target, FitSettings(reference_distance=2.5), cross_validate=False)
    assert moved.fixed["reference_distance"] == 2.5 and moved.parameters["coupling"] == pytest.approx(0.6, rel=2e-3)


# -- bounds and identifiability ------------------------------------------------------------------------------------------------
def test_a_parameter_forced_onto_a_bound_is_reported_and_costs_fit_quality():
    target = synthetic(diabatic_offset=0.0)
    free = calibrate_evb_2d(target, cross_validate=False)
    tight = calibrate_evb_2d(target, FitSettings(bounds=(("oo_equilibrium", 2.0, 2.5),)), cross_validate=False)
    assert tight.pinned["oo_equilibrium"] == "upper" and "oo_equilibrium" not in tight.covariance_names
    assert "ON ITS UPPER BOUND" in tight.summary()
    assert tight.rmse_ev["1.5"] > 100 * max(free.rmse_ev["1.5"], 1e-4)                 # the bound is the reason the fit got worse
    with pytest.raises(ValidationError, match="unknown parameter"):
        calibrate_evb_2d(target, FitSettings(bounds=(("nonsense", 0.0, 1.0),)), cross_validate=False)


def test_the_target_must_be_a_solved_surface():
    with pytest.raises(ValidationError, match="not been solved"):
        calibrate_evb_2d(evb2d(), cross_validate=False)
    bad = synthetic()
    e = bad.obs("surface_energy").copy()
    e[3, 4] = np.nan
    with pytest.raises(ValidationError, match="finite"):
        calibrate_evb_2d(bad.evolve(observables={**bad.observables, "surface_energy": Quantity(e, "eV")}), cross_validate=False)


def test_leave_one_distance_out_is_an_out_of_sample_test():
    target = synthetic(diabatic_offset=0.0)
    cv = leave_one_distance_out(target, max_folds=100)
    assert len(cv["rows"]) == 19                                                    # 21 distances; the two ends are extrapolations
    assert cv["rmse_max"] < 1e-3                                                    # an exact model predicts an unseen distance exactly
    assert cv["barrier_error_max_abs"] < 1e-3
    assert len(leave_one_distance_out(target)["rows"]) == 12                        # by default the cost is capped, however dense the grid


# -- the calibrated systems --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def noisy_calibration():
    return calibrate_evb_2d(synthetic(noise=0.005, seed=3, diabatic_offset=0.0), cross_validate=False)


def test_the_calibrated_2d_system_reproduces_the_reference_surface(noisy_calibration):
    cal = noisy_calibration
    target = synthetic(diabatic_offset=0.0)
    model = EVBFlexibleProtonTransferEngine().solve(cal.system(scan_distance=2.8), BACKEND)
    x_t, r_t = target.obs("surface_x"), target.obs("surface_r")
    assert model.obs("surface_x") == pytest.approx(np.linspace(-0.65, 0.65, 41))
    assert model.obs("surface_r") == pytest.approx(np.linspace(2.3, 3.0, 21))
    e_model = model.obs("surface_energy") - model.obs("surface_energy").min()
    e_true = target.obs("surface_energy") - target.obs("surface_energy").min()
    inside = e_true <= 1.0
    assert np.sqrt(((e_model - e_true)[inside] ** 2).mean()) < 0.02                  # within a few times the 5 meV noise


def test_systems_carry_uncertainty_only_when_asked(noisy_calibration):
    cal = noisy_calibration
    with_u, without = cal.system(), cal.system(with_uncertainty=False)
    assert with_u.parameters["morse_depth"].sigma == pytest.approx(cal.sigma["morse_depth"]) and with_u.parameters["morse_depth"].source == "calibration"
    block = with_u.structure["parameter_covariance"]
    n = len(cal.covariance_names)
    assert block["names"] == cal.covariance_names and np.asarray(block["matrix"]).shape == (n, n)
    assert "diabatic_offset" in cal.covariance_names        # 5 meV of noise breaks the exact mirror symmetry, so the offset is fitted
    assert "diabatic_offset" not in block["minimum"]         # an offset may be negative: no floor
    assert block["minimum"]["oo_depth"] == pytest.approx(0.001) and block["minimum"]["coupling_decay"] == 0.0
    assert all(q.sigma is None for q in without.parameters.values()) and without.structure == {}


def test_the_1d_model_is_the_2d_model_cut_at_one_distance(noisy_calibration):
    cal = noisy_calibration
    from substrate.engines.electronic import EVBProtonTransferEngine
    system_2d = cal.system(scan_distance=2.8, with_uncertainty=False)
    system_2d.parameters["n_x"] = Quantity(241, "1")                                # the 1D engine's grid, so the slices coincide
    two_d = EVBFlexibleProtonTransferEngine().solve(system_2d, BACKEND)
    one_d = EVBProtonTransferEngine().solve(cal.system("electronic.evb_two_state", distance=2.8), BACKEND)
    assert one_d.obs("classical_barrier") == pytest.approx(two_d.obs("classical_barrier"), abs=1e-9)
    shift = two_d.obs("scan_energy") - one_d.obs("scan_energy")
    assert shift == pytest.approx(np.full_like(shift, shift[0]), abs=1e-9)           # the slices differ only by a constant
    assert one_d.obs("scan_coordinate").max() == pytest.approx(0.65, abs=1e-6)
    with pytest.raises(ValidationError, match="`distance`"):
        cal.system("electronic.evb_two_state")
    with pytest.raises(ValidationError, match="no calibrated model of kind"):
        cal.system("quantum.double_well_1d")


def test_context_is_copied_from_the_reference_and_can_be_overridden():
    target = synthetic(diabatic_offset=0.0, mass=2.013553)
    target.parameters["context.k_on"] = Quantity(1e7, "1/(M s)", source="input")
    cal = calibrate_evb_2d(target, cross_validate=False)
    system = cal.system()
    assert system.param("particle_mass", "amu") == 2.013553 and system.param("temperature", "K") == 300.0
    assert system.param("context.k_on", "1/(M s)") == 1e7                           # experiment-level context travels too
    assert cal.system(context={"particle_mass": (1.007276, "amu")}).param("particle_mass") == 1.007276


def test_a_calibration_survives_a_round_trip_through_json(noisy_calibration, tmp_path):
    path = tmp_path / "c.json"
    noisy_calibration.save(path)
    again = Calibration.load(path)
    assert json.loads(path.read_text())["target_name"] == noisy_calibration.target_name
    assert again.system(scan_distance=2.8).fingerprint() == noisy_calibration.system(scan_distance=2.8).fingerprint()
    assert again.summary() == noisy_calibration.summary()


# -- the analytic double well ------------------------------------------------------------------------------------------------
def test_the_quartic_double_well_is_recovered_from_a_slice():
    x = np.linspace(-0.8, 0.8, 161)
    for d_e in (0.0, 0.05):
        fit = calibrate_double_well(x, double_well(x, 0.5, 0.4, d_e) + 3.0)         # an arbitrary energy zero
        assert fit.parameters["barrier_height"] == pytest.approx(0.5, rel=1e-3)
        assert fit.parameters["half_separation"] == pytest.approx(0.4, rel=1e-3)
        assert fit.parameters["reaction_energy"] == pytest.approx(d_e, abs=2e-4) and fit.rmse_ev < 1e-4
    system = fit.system(mass=1.007276)
    assert system.kind == "quantum.double_well_1d" and system.param("barrier_height", "eV") == pytest.approx(0.5, rel=1e-3)
    assert system.structure["parameter_covariance"]["names"] == fit.covariance_names
    with pytest.raises(ValidationError, match="fewer than two wells"):
        calibrate_double_well(x, 0.5 * x**2)


# -- joint (correlated) parameter draws in the pipeline ---------------------------------------------------------------------------
def root_with_joint(floor=None):
    cov = [[0.04, 0.036], [0.036, 0.04]]                                            # sigma 0.2 each, correlation 0.9
    block = {"names": ["a", "b"], "matrix": cov, **({"minimum": floor} if floor else {})}
    return ScientificSystem(
        "root", Scale.QUANTUM, "quantum.double_well_1d",
        parameters={"a": Quantity(1.0, "1", 0.2), "b": Quantity(2.0, "1", 0.2), "c": Quantity(5.0, "1", 0.5), "d": Quantity(7.0, "1")},
        structure={"parameter_covariance": block},
    )


def test_joint_draws_reproduce_the_covariance_and_leave_other_parameters_independent():
    pipeline = Pipeline(default_registry())
    rng = np.random.default_rng(0)
    draws = [pipeline._draw(root_with_joint(), rng) for _ in range(4000)]
    ab = np.array([[d.parameters["a"].value, d.parameters["b"].value] for d in draws])
    assert ab.mean(axis=0) == pytest.approx([1.0, 2.0], abs=0.02)
    assert np.cov(ab.T) == pytest.approx(np.array([[0.04, 0.036], [0.036, 0.04]]), rel=0.1)
    c = np.array([d.parameters["c"].value for d in draws])
    assert c.std() == pytest.approx(0.5, rel=0.06) and abs(np.corrcoef(ab[:, 0], c)[0, 1]) < 0.06      # independent of the block
    assert {d.parameters["d"].value for d in draws} == {7.0}                                            # no sigma, never varied
    assert all(d.parameters["a"].sigma is None for d in draws)


def test_physical_floors_truncate_the_draw_instead_of_discarding_runs():
    pipeline = Pipeline(default_registry())
    root = root_with_joint({"a": 0.9, "b": 1.9})                                     # just below the means: ~30% of raw draws violate
    draws = [pipeline._draw(root, np.random.default_rng(i)) for i in range(300)]
    assert all(d.parameters["a"].value >= 0.9 and d.parameters["b"].value >= 1.9 for d in draws)
    impossible = root_with_joint({"a": 50.0})
    with pytest.raises(SubstrateError, match="above the stated minimums"):
        pipeline._draw(impossible, np.random.default_rng(0))


def test_a_malformed_covariance_is_refused():
    pipeline = Pipeline(default_registry())
    for matrix, names, message in (
        ([[1.0, 0.5], [0.4, 1.0]], ["a", "b"], "symmetric"),                          # not symmetric
        ([[1.0, 2.0], [2.0, 1.0]], ["a", "b"], "positive semi-definite"),            # |correlation| > 1
        ([[1.0]], ["nonexistent"], "scalar parameters"),
    ):
        root = root_with_joint()
        root.structure["parameter_covariance"] = {"names": names, "matrix": matrix}
        with pytest.raises(SubstrateError, match=message):
            pipeline._uncertain_names(root)


def test_ignoring_correlations_overstates_the_uncertainty_several_fold(noisy_calibration):
    pipeline = Pipeline(default_registry())
    joint = noisy_calibration.system(scan_distance=2.8)
    independent = noisy_calibration.system(scan_distance=2.8)
    independent.structure = {}                                                          # same sigmas, correlations discarded
    spread = lambda s: pipeline.run(s, ["electronic_structure"], n_samples=40, seed=1).trace[0].observables["classical_barrier"].sigma
    assert spread(independent) > 3 * spread(joint)                                      # ~5x here; ~15x on the real surface


def test_ensembles_over_a_calibration_do_not_lose_runs_to_unphysical_draws(noisy_calibration):
    system = noisy_calibration.system(scan_distance=2.8)
    block = system.structure["parameter_covariance"]
    i = block["names"].index("oo_depth")
    matrix = np.array(block["matrix"])
    matrix[i, :], matrix[:, i] = matrix[i, :] * 30.0, matrix[:, i] * 30.0               # make a negative depth likely
    matrix[i, i] = (noisy_calibration.parameters["oo_depth"] * 0.8) ** 2                # sigma = 80% of the value
    block["matrix"] = matrix.tolist()
    result = Pipeline(default_registry()).run(system, ["electronic_structure"], n_samples=60, seed=2)
    assert result.ensemble["n_failed"] == 0                                             # truncated at the floor, not dropped


# -- the command line --------------------------------------------------------------------------------------------------------------
def test_the_calibrate_command_writes_an_experiment_that_runs_and_reproduces_the_original(tmp_path, capsys):
    from substrate.cli import main
    from substrate import load_experiment
    out, saved = tmp_path / "calibrated.yaml", tmp_path / "calibration.json"
    source = EXPERIMENTS / "proton_transfer_molecular.yaml"
    assert main(["calibrate", str(source), "--out", str(out), "--json", str(saved), "--samples", "12"]) == 0
    assert "Calibration of the 2D valence-bond model" in capsys.readouterr().out and saved.exists()
    assert out.read_text().startswith("# Calibrated model of")
    exp = load_experiment(out)
    assert exp.system.kind == "electronic.evb_two_state_2d" and exp.n_samples == 12
    assert "parameter_covariance" in exp.system.structure and exp.propagation[-1] == Scale.REACTION
    original = load_experiment(source)
    pipeline = Pipeline(default_registry())
    rate = lambda e: pipeline.run(e.system, e.propagation).final.param("k_f")
    assert rate(exp) == pytest.approx(rate(original), rel=0.02)                          # the exact model was recovered
    assert main(["calibrate", str(EXPERIMENTS / "proton_transfer.yaml")]) == 1           # a quantum-scale system has no 2D surface
