"""Calibrating the model engines against REAL quantum chemistry (Hartree-Fock/6-31G* PySCF energies for the Zundel cation).

Skipped where PySCF is unavailable. The energies come from the persistent test cache, so only a first run computes anything.
These are the tests that matter most: they check that the cheap model reproduces the expensive calculation's downstream answers,
and that the module is honest about the part it cannot reproduce.
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from substrate import Pipeline, Quantity, Scale, ScientificSystem, default_registry, load_experiment
from substrate.calibration import calibrate_double_well, calibrate_evb_2d
from substrate.engines.qc_scan import QCScanEngine
from substrate.qc import PySCFProgram

from conftest import DEUTERON, EXPERIMENTS, PROTON, zundel_system

pytestmark = [
    pytest.mark.qc,
    pytest.mark.skipif(not PySCFProgram().available(), reason="PySCF is not installed (see scripts/setup_qc_env.sh)"),
]
ROUTE = ["electronic_structure", "quantum", "reaction"]
DISTANCES = (2.6, 2.7, 2.8, 2.9, 3.0)


@pytest.fixture(scope="module", autouse=True)
def persistent_cache():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SUBSTRATE_CACHE_DIR", str(Path(tempfile.gettempdir()) / "substrate-test-qc-cache"))
        yield


@pytest.fixture(scope="module")
def pipeline():
    return Pipeline(default_registry())


@pytest.fixture(scope="module")
def target():
    return QCScanEngine().solve(zundel_system("pyscf", scan_distance=2.8), None)


@pytest.fixture(scope="module")
def calibration(target):
    return calibrate_evb_2d(target)


def real_rate(pipeline, distance, mass=PROTON):
    return pipeline.run(zundel_system("pyscf", scan_distance=distance, mass=mass), ROUTE).final.param("k_f")


def model_rate(pipeline, calibration, distance, mass=PROTON):
    system = calibration.system(scan_distance=distance, with_uncertainty=False, context={"particle_mass": (mass, "amu")})
    return pipeline.run(system, ROUTE).final.param("k_f")


# -- the fit ---------------------------------------------------------------------------------------------------------------------
def test_the_fit_reproduces_real_hartree_fock_where_it_matters(calibration):
    assert calibration.rmse_ev["1.5"] < 0.015 and calibration.max_error_ev["1.5"] < 0.04       # eV, over the fitted window
    assert calibration.pinned == {} and calibration.reduced_chi2 < 3 and calibration.starts_agreeing >= 6
    for column in calibration.columns:
        assert column.wells_reference == column.wells_model                                       # one well vs two, at every distance
        if column.barrier_reference is not None:
            assert abs(column.barrier_model - column.barrier_reference) < 0.03                    # eV
            assert abs(column.well_x_model - column.well_x_reference) < 0.08                       # angstrom (grid spacing is 0.07)


def test_the_fit_predicts_distances_it_never_saw(calibration):
    cv = calibration.cross_validation
    assert cv["rmse_max"] < 0.02 and cv["barrier_error_mean_abs"] < 0.02 and cv["barrier_error_max_abs"] < 0.04
    assert cv["rmse_mean"] < 2 * calibration.rmse_ev["1.5"]                                     # out of sample is not much worse than in


def test_the_fit_reports_which_parameters_are_poorly_determined(calibration):
    summary = calibration.summary()
    assert "NOT the model's error" in summary and "leave-one-distance-out" in summary
    assert "oo_depth" in calibration.poorly_determined                                           # the data do not pin the O...O well depth


# -- downstream: does the cheap model give the expensive chain's answers? ---------------------------------------------------------
def test_the_calibrated_model_reproduces_real_rates_over_nine_orders_of_magnitude(pipeline, calibration):
    real = {d: real_rate(pipeline, d) for d in DISTANCES}
    model = {d: model_rate(pipeline, calibration, d) for d in DISTANCES}
    assert real[2.6] / real[3.0] > 1e8 and model[2.6] / model[3.0] > 1e8                          # a huge dynamic range ...
    for d in DISTANCES:
        assert 0.6 < model[d] / real[d] < 1.6, d                                                  # ... matched to within ~60%


def test_the_calibrated_model_reproduces_the_real_isotope_effect(pipeline, calibration):
    for d in DISTANCES:
        real = real_rate(pipeline, d) / real_rate(pipeline, d, DEUTERON)
        model = model_rate(pipeline, calibration, d) / model_rate(pipeline, calibration, d, DEUTERON)
        assert model == pytest.approx(real, rel=0.30), d


def test_calibration_improves_on_the_uncalibrated_defaults_by_orders_of_magnitude(pipeline, calibration):
    from conftest import evb2d
    default = evb2d()
    for name, value in dict(scan_distance=2.8, x_extent=0.7, distance_min=2.3, distance_max=3.0).items():
        default.parameters[name] = Quantity(value, "angstrom")
    default.parameters["n_x"], default.parameters["n_r"] = Quantity(41, "1"), Quantity(21, "1")
    real = real_rate(pipeline, 2.8)
    uncalibrated = pipeline.run(default, ROUTE).final.param("k_f")
    assert uncalibrated / real < 1e-3                                                             # the defaults were wrong by > 3 orders
    assert 0.6 < model_rate(pipeline, calibration, 2.8) / real < 1.6


def test_the_1d_model_gives_the_same_rate_as_the_2d_model_cut_at_that_distance(pipeline, calibration):
    one_d = pipeline.run(calibration.system("electronic.evb_two_state", distance=2.8), ROUTE).final.param("k_f")
    assert one_d == pytest.approx(model_rate(pipeline, calibration, 2.8), rel=0.05)


def test_the_calibrated_quartic_double_well_reproduces_real_rates(pipeline):
    for distance in (2.6, 2.8):
        scan = pipeline.run(zundel_system("pyscf", scan_distance=distance), ["electronic_structure"]).final
        fit = calibrate_double_well(scan.obs("scan_coordinate"), scan.obs("scan_energy"))
        rate = pipeline.run(fit.system(mass=PROTON, with_uncertainty=False), ["reaction"]).final.param("k_f")
        assert 0.8 < rate / real_rate(pipeline, distance) < 1.25, distance


# -- honesty: parameter uncertainty is not model error ---------------------------------------------------------------------------------
def test_the_fit_uncertainty_is_smaller_than_the_models_actual_error_and_correlations_matter(pipeline, calibration):
    joint = pipeline.run(calibration.system(scan_distance=2.8), ROUTE, n_samples=40, seed=1)
    independent_system = calibration.system(scan_distance=2.8)
    independent_system.structure = {}
    independent = pipeline.run(independent_system, ROUTE, n_samples=40, seed=1)
    relative = lambda r: r.final.parameters["k_f"].sigma / r.final.parameters["k_f"].value
    assert joint.ensemble["n_failed"] == 0
    assert relative(joint) < 0.15 and relative(independent) > 5 * relative(joint)                # joint draws respect the correlations
    error = abs(model_rate(pipeline, calibration, 2.8) / real_rate(pipeline, 2.8) - 1.0)           # the model's real deviation (~28%)
    assert error > 2 * relative(joint)             # the ensemble spread understates the error: it is parameter uncertainty only


# -- files ------------------------------------------------------------------------------------------------------------------------------
def test_the_committed_calibrated_experiment_runs_and_matches_a_fresh_calibration(pipeline, calibration):
    experiment = load_experiment(EXPERIMENTS / "zundel_calibrated.yaml")
    assert experiment.system.kind == "electronic.evb_two_state_2d" and "parameter_covariance" in experiment.system.structure
    nominal = pipeline.run(experiment.system, experiment.propagation).final.param("k_f")
    assert nominal == pytest.approx(model_rate(pipeline, calibration, 2.8), rel=0.01)


def test_the_calibrate_command_works_on_the_real_experiment(tmp_path, capsys):
    from substrate.cli import main
    out = tmp_path / "calibrated.yaml"
    assert main(["calibrate", str(EXPERIMENTS / "zundel_hf.yaml"), "--out", str(out), "--samples", "10"]) == 0
    text = capsys.readouterr().out
    assert "Zundel cation" in text and "leave-one-distance-out" in text and out.exists()
    assert load_experiment(out).n_samples == 10
