"""Transfer across molecules, against REAL quantum chemistry (Hartree-Fock energies from PySCF for the Zundel cation, the ammonium
dimer and the bifluoride anion, and the asymmetric water-ammonia ion). Skipped where PySCF is unavailable; the first run computes ~540
single points and later runs read them from the persistent test cache. (The count above is for the first four molecules; the methanol-water, ammonia-methylamine and fluoride-methanol ions add ~270 each.)

These tests record findings, not hopes: if a change to the model or the fit makes parameters carry across molecules, the
"does not transfer" tests fail, which is a prompt to look at why, not to loosen a threshold.
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from substrate import Pipeline, default_registry, load_experiment
from dataclasses import replace

from substrate.calibration import FitSettings, _is_symmetric, calibrate_evb_2d
from substrate.pes import prominent_minima
from substrate.qc import PySCFProgram
from substrate.transfer import cross_predict, fit_settings, learning_curve, make_reference, rate_comparison, transfer_matrix

from conftest import EXPERIMENTS

pytestmark = [
    pytest.mark.qc,
    pytest.mark.skipif(not PySCFProgram().available(), reason="PySCF is not installed (see scripts/setup_qc_env.sh)"),
]
NAMES = ["zundel_hf", "n2h7_hf", "fhf_hf"]
SETTINGS = FitSettings(reference_distance=2.7)


@pytest.fixture(scope="module", autouse=True)
def persistent_cache():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SUBSTRATE_CACHE_DIR", str(Path(tempfile.gettempdir()) / "substrate-test-qc-cache"))
        yield


@pytest.fixture(scope="module")
def pipeline():
    return Pipeline(default_registry())


@pytest.fixture(scope="module")
def solved(pipeline):
    out = {}
    for name in NAMES:
        experiment = load_experiment(EXPERIMENTS / "references" / f"{name}.yaml")
        out[name] = (experiment.system, pipeline.run(experiment.system, experiment.propagation).final)
    return out


@pytest.fixture(scope="module")
def references(solved):
    return {name: make_reference(name, system, target, SETTINGS) for name, (system, target) in solved.items()}


@pytest.fixture(scope="module")
def matrix(references):
    return transfer_matrix(references)


# -- the surfaces ---------------------------------------------------------------------------------------------------------------------
def test_each_template_is_recorded_on_the_solved_surface_with_its_atoms(solved):
    expected = {"zundel_hf": ("zundel_cation", 7, 1), "n2h7_hf": ("ammonium_dimer_cation", 9, 1), "fhf_hf": ("bifluoride_anion", 3, -1)}
    for name, (template, n_atoms, charge) in expected.items():
        molecule = solved[name][1].structure["molecule"]
        assert (molecule["template"], len(molecule["atoms"]), molecule["charge"]) == (template, n_atoms, charge)
        assert solved[name][1].structure["method"]["theory"] == "hf"


def test_every_surface_is_a_symmetric_proton_transfer_surface_with_barriers_that_grow_with_distance(references):
    for name, ref in references.items():
        assert ref.symmetric, name                                               # E(+x) = E(-x), to the engine's verification tolerance
        wells = [(c.distance, c.barrier_reference) for c in transfer_matrix({name: ref})[(name, name)].columns]
        barriers = [b for _, b in wells if b is not None]
        assert len(barriers) >= 6 and barriers == sorted(barriers), name        # one well when short, a barrier from some distance on
        assert barriers[-1] > 0.7 and barriers[0] < 0.2                          # eV: from nearly barrierless to a real barrier


# -- each calibration against its own reference --------------------------------------------------------------------------------------
def test_each_reference_is_fitted_to_within_a_few_tolerances(references, matrix):
    for name, ref in references.items():
        own = matrix[(name, name)]
        assert own.rmse_ev["1.5"] < 0.03 and own.max_error_ev["1.5"] < 0.08, name        # eV; the fit tolerance is 0.01
        assert own.wells_wrong == 0 and own.barrier_error_mean_abs < 0.05, name
        assert ref.calibration.pinned == {}, name


def test_the_parameters_that_have_a_physical_meaning_are_physical(references):
    for name, ref in references.items():
        p = ref.calibration.parameters
        assert 0.9 < p["morse_r_eq"] < 1.1 and 1.5 < p["morse_alpha"] < 3.5, name       # an X-H bond length (angstrom) and Morse width


# -- the finding: parameters do not carry across molecules --------------------------------------------------------------------------
def test_parameters_fitted_to_one_molecule_do_not_predict_another(matrix):
    for source in NAMES:
        for target in NAMES:
            if source == target:
                continue
            transferred, own = matrix[(source, target)], matrix[(target, target)]
            assert transferred.rmse_ev["1.5"] > 0.15 and transferred.rmse_ev["1.5"] > 10 * own.rmse_ev["1.5"], (source, target)
            assert transferred.barrier_error_mean_abs > 0.08, (source, target)          # eV: kT is 0.026, so rates are off by orders of magnitude


def test_the_rates_confirm_it_each_calibration_is_right_for_its_own_molecule_and_wrong_for_the_others(references, pipeline):
    rows = rate_comparison(references, references, pipeline, barriers=(0.4,))
    own = [r for r in rows if r.source == r.target]
    other = [r for r in rows if r.source != r.target and r.ratio is not None]
    assert len(own) == 3 and all(0.5 < r.ratio < 2.0 for r in own)
    assert len(other) >= 4 and all(not 0.2 < r.ratio < 5.0 for r in other)               # at least a factor of five away, every time


# -- few-shot learning ------------------------------------------------------------------------------------------------------------------
def test_given_enough_target_distances_a_prior_from_another_molecule_makes_no_difference(references):
    target, source = references["fhf_hf"], references["zundel_hf"].calibration
    scratch = learning_curve(target, counts=(6,), settings=SETTINGS)[0].rmse_ev
    primed = learning_curve(target, counts=(6,), source=source, relative_sigma=0.3, settings=SETTINGS)[0].rmse_ev
    assert scratch < 0.02 and primed == pytest.approx(scratch, abs=0.005)


# =====================================================================================================================
# the asymmetric reference: water-ammonia, [H2O...H...NH3]+
# =====================================================================================================================
@pytest.fixture(scope="module")
def wa(pipeline):
    experiment = load_experiment(EXPERIMENTS / "references" / "water_ammonia_hf.yaml")
    return experiment, pipeline.run(experiment.system, experiment.propagation).final


def test_the_water_ammonia_surface_is_asymmetric_and_its_metastable_well_appears_only_with_distance(wa):
    experiment, solved = wa
    molecule = solved.structure["molecule"]
    assert (molecule["template"], len(molecule["atoms"]), molecule["charge"]) == ("water_ammonia_cation", 8, 1)
    assert solved.structure["scan"] == {"mirror_symmetric": False} and solved.obs("calculations_run") + solved.obs("calculations_cached") >= 25 * 11
    x, r, e = solved.obs("surface_x"), solved.obs("surface_r"), solved.obs("surface_energy")
    e = e - e.min()
    assert not _is_symmetric(x, e) and x[np.unravel_index(e.argmin(), e.shape)[0]] > 0                       # the proton sits on the nitrogen
    j = int(np.argmin(abs(r - 3.0)))
    left, right = int(np.argmin(abs(x + 0.4))), int(np.argmin(abs(x - 0.4)))
    assert e[right, j] < 0.15 and e[left, j] > 1.0                                                           # eV: the oxygen side is much higher
    wells = [len(prominent_minima(e[:, k])) for k in range(len(r))]
    assert wells == [1] * 4 + [2] * 7                                                                          # one well up to 2.7 A, two from 2.8 A
    for k in range(len(r)):
        if wells[k] == 2:
            assert 0.9 < e[prominent_minima(e[:, k])[0], k] < 2.0, r[k]                                       # the O-side well sits 1-1.6 eV above the N-side one


def test_the_model_follows_the_asymmetric_surface_only_through_its_fitted_offset(wa):
    experiment, solved = wa
    free = calibrate_evb_2d(solved, fit_settings(experiment, FitSettings(reference_distance=2.7)), cross_validate=False)
    assert free.settings["window_ev"] == 2.5 and free.settings["symmetric_reference"] is False and "diabatic_offset" in free.parameters
    assert free.rmse_ev["2.5"] < 0.06 and free.max_error_ev["2.5"] < 0.25                                    # eV: four times the symmetric fits' error, but a fit
    assert all(c.wells_reference == c.wells_model for c in free.columns)
    assert all(abs(c.barrier_model - c.barrier_reference) < 0.15 for c in free.columns if c.barrier_reference is not None)
    assert free.parameters["diabatic_offset"] < -1.0                                                          # the offset is large, and an effective parameter
    zero = calibrate_evb_2d(solved, replace(fit_settings(experiment, FitSettings(reference_distance=2.7)), bounds=(("diabatic_offset", -1e-9, 1e-9),)),
                            cross_validate=False)
    assert zero.rmse_ev["2.5"] > 0.3 and zero.rmse_ev["2.5"] > 5 * free.rmse_ev["2.5"]                       # take the offset away and the fit is ten times worse
    assert sum(c.wells_reference != c.wells_model for c in zero.columns) >= 1                                 # and it loses a well somewhere


def test_the_declared_window_is_what_lets_the_fit_see_the_barrier(wa):
    experiment, solved = wa
    base = FitSettings(reference_distance=2.7)
    wide = calibrate_evb_2d(solved, fit_settings(experiment, base), cross_validate=False)
    narrow = calibrate_evb_2d(solved, base, cross_validate=False)                                              # the default 1.5 eV window
    error = lambda cal: abs(cal.columns[-1].barrier_model - cal.columns[-1].barrier_reference)                # the largest barrier, at 3.4 A
    assert narrow.settings["window_ev"] == 1.5 and error(narrow) > 0.15 > error(wide) and error(narrow) > 1.5 * error(wide)


def test_distances_never_seen_are_predicted_for_the_asymmetric_surface_too(wa):
    experiment, solved = wa
    cal = calibrate_evb_2d(solved, fit_settings(experiment, FitSettings(reference_distance=2.7)))
    cv = cal.cross_validation
    assert cv["rmse_mean"] < 0.06 and cv["barrier_error_mean_abs"] < 0.06                                     # leave-one-distance-out, eV


def test_symmetric_parameters_cannot_predict_the_asymmetric_surface_and_nor_the_reverse(wa, references):
    experiment, solved = wa
    target = make_reference("water_ammonia_hf", experiment.system, solved, fit_settings(experiment, FitSettings(reference_distance=2.7)))
    own = cross_predict(target.calibration, target).rmse_window
    for name in NAMES:
        forward, backward = cross_predict(references[name].calibration, target), cross_predict(target.calibration, references[name])
        assert forward.rmse_window > 0.3 and forward.rmse_window > 5 * own and forward.wells_wrong >= 2, name
        assert backward.rmse_window > 0.3 and backward.wells_wrong >= 2, name


# =====================================================================================================================
# a second, milder asymmetric reference: methanol-water, [H2O...H...HOCH3]+
# =====================================================================================================================
@pytest.fixture(scope="module")
def mw(pipeline):
    experiment = load_experiment(EXPERIMENTS / "references" / "methanol_water_hf.yaml")
    return experiment, pipeline.run(experiment.system, experiment.propagation).final


def test_the_methanol_water_surface_is_mildly_asymmetric_with_the_proton_on_the_methanol(mw):
    experiment, solved = mw
    molecule = solved.structure["molecule"]
    assert (molecule["template"], len(molecule["atoms"]), molecule["charge"]) == ("methanol_water_cation", 10, 1)
    assert solved.structure["scan"] == {"mirror_symmetric": False} and solved.obs("calculations_run") + solved.obs("calculations_cached") >= 23 * 11
    x, r, e = solved.obs("surface_x"), solved.obs("surface_r"), solved.obs("surface_energy")
    e = e - e.min()
    assert not _is_symmetric(x, e) and x[np.unravel_index(e.argmin(), e.shape)[0]] > 0                       # the proton sits on the methanol oxygen
    wells = [len(prominent_minima(e[:, k])) for k in range(len(r))]
    assert wells == [1] * 3 + [2] * 8                                                                          # one well up to 2.56 A, two from 2.64 A
    for k in range(len(r)):
        if wells[k] == 2:
            donor_side, acceptor_side = (e[i, k] for i in prominent_minima(e[:, k])[:2])
            assert 0.3 < donor_side - acceptor_side < 0.9, r[k]                                              # eV: about the proton-affinity difference, 0.65


def test_the_model_fits_the_mild_asymmetry_with_a_smaller_offset_than_water_ammonia_needs(mw):
    experiment, solved = mw
    settings = fit_settings(experiment, FitSettings(reference_distance=2.7))
    assert settings.window_ev == 2.5
    free = calibrate_evb_2d(solved, settings, cross_validate=False)
    assert free.rmse_ev["2.5"] < 0.07 and free.max_error_ev["2.5"] < 0.3
    assert all(c.wells_reference == c.wells_model for c in free.columns)
    assert all(abs(c.barrier_model - c.barrier_reference) < 0.06 for c in free.columns if c.barrier_reference is not None)
    assert -2.5 < free.parameters["diabatic_offset"] < -1.0                                                  # an effective offset, larger than the 0.65 eV proton-affinity gap
    zero = calibrate_evb_2d(solved, replace(settings, bounds=(("diabatic_offset", -1e-9, 1e-9),)), cross_validate=False)
    assert zero.rmse_ev["2.5"] > 3 * free.rmse_ev["2.5"]                                                      # without the offset the fit is several times worse ...
    assert sum(c.wells_reference != c.wells_model for c in zero.columns) >= 1                                 # ... and loses a well


def test_distances_never_seen_are_predicted_for_the_mild_asymmetry_too(mw):
    experiment, solved = mw
    cv = calibrate_evb_2d(solved, fit_settings(experiment, FitSettings(reference_distance=2.7))).cross_validation
    assert cv["rmse_mean"] < 0.07 and cv["barrier_error_mean_abs"] < 0.03                                    # leave-one-distance-out, eV


def test_transfer_degrades_with_the_chemical_distance_from_the_symmetric_ion(mw, wa, references):
    """Zundel -> methanol-water (one methyl, 0.65 eV of asymmetry) is a smaller change than Zundel -> water-ammonia (a different heavy atom,
    1.7 eV), and methanol-water -> water-ammonia is smaller than Zundel -> water-ammonia: the errors follow the chemistry."""
    settings = FitSettings(reference_distance=2.7)
    targets = {name: make_reference(name, experiment.system, solved, fit_settings(experiment, settings))
               for name, (experiment, solved) in (("methanol_water_hf", mw), ("water_ammonia_hf", wa))}
    zundel = references["zundel_hf"].calibration
    to_mw, to_wa = cross_predict(zundel, targets["methanol_water_hf"]), cross_predict(zundel, targets["water_ammonia_hf"])
    mw_to_wa = cross_predict(targets["methanol_water_hf"].calibration, targets["water_ammonia_hf"])
    assert 0.15 < to_mw.rmse_window < 0.35 and to_wa.rmse_window > 0.4 and mw_to_wa.rmse_window < to_wa.rmse_window
    assert to_mw.rmse_window < to_wa.rmse_window and to_mw.wells_wrong <= to_wa.wells_wrong
    assert to_mw.rmse_window > 3 * cross_predict(targets["methanol_water_hf"].calibration, targets["methanol_water_hf"]).rmse_window    # still a failure to transfer


# =====================================================================================================================
# a third asymmetric reference, the nitrogen counterpart: ammonia-methylamine, [H3N...H...H2NCH3]+
# =====================================================================================================================
@pytest.fixture(scope="module")
def am(pipeline):
    experiment = load_experiment(EXPERIMENTS / "references" / "ammonia_methylamine_hf.yaml")
    return experiment, pipeline.run(experiment.system, experiment.propagation).final


def test_the_ammonia_methylamine_surface_is_mildly_asymmetric_with_the_proton_on_the_methylamine(am):
    experiment, solved = am
    molecule = solved.structure["molecule"]
    assert (molecule["template"], len(molecule["atoms"]), molecule["charge"]) == ("ammonia_methylamine_cation", 12, 1)
    assert solved.structure["scan"] == {"mirror_symmetric": False} and solved.obs("calculations_run") + solved.obs("calculations_cached") >= 23 * 11
    x, r, e = solved.obs("surface_x"), solved.obs("surface_r"), solved.obs("surface_energy")
    e = e - e.min()
    assert not _is_symmetric(x, e) and x[np.unravel_index(e.argmin(), e.shape)[0]] > 0                       # the proton sits on the methylamine nitrogen
    wells = [len(prominent_minima(e[:, k])) for k in range(len(r))]
    assert wells == [1] * 2 + [2] * 9                                                                          # one well up to 2.59 A, two from 2.68 A
    for k in range(len(r)):
        if wells[k] == 2:
            donor_side, acceptor_side = (e[i, k] for i in prominent_minima(e[:, k])[:2])
            assert 0.2 < donor_side - acceptor_side < 0.5, r[k]                                              # eV: about the proton-affinity difference, 0.47


def test_the_model_fits_the_nitrogen_asymmetry_with_the_smallest_offset_of_the_three(am, mw, wa):
    experiment, solved = am
    settings = fit_settings(experiment, FitSettings(reference_distance=2.7))
    assert settings.window_ev == 2.5
    free = calibrate_evb_2d(solved, settings, cross_validate=False)
    assert free.rmse_ev["2.5"] < 0.055 and free.max_error_ev["2.5"] < 0.25
    assert all(c.wells_reference == c.wells_model for c in free.columns)
    assert all(abs(c.barrier_model - c.barrier_reference) < 0.1 for c in free.columns if c.barrier_reference is not None)
    zero = calibrate_evb_2d(solved, replace(settings, bounds=(("diabatic_offset", -1e-9, 1e-9),)), cross_validate=False)
    assert zero.rmse_ev["2.5"] > 2.5 * free.rmse_ev["2.5"] and sum(c.wells_reference != c.wells_model for c in zero.columns) >= 1
    offsets = {name: calibrate_evb_2d(sol, fit_settings(exp, FitSettings(reference_distance=2.7)), cross_validate=False).parameters["diabatic_offset"]
               for name, (exp, sol) in (("methylamine", am), ("methanol", mw), ("ammonia", wa))}
    assert offsets["ammonia"] < offsets["methanol"] < offsets["methylamine"] < -0.4                          # more asymmetry, bigger offset: in the order of the proton-affinity gaps


def test_distances_never_seen_are_predicted_for_the_nitrogen_asymmetry_too(am):
    experiment, solved = am
    cv = calibrate_evb_2d(solved, fit_settings(experiment, FitSettings(reference_distance=2.7))).cross_validation
    assert cv["rmse_mean"] < 0.06 and cv["barrier_error_mean_abs"] < 0.04                                    # leave-one-distance-out, eV


def test_the_symmetric_parent_is_the_nearest_neighbour_of_its_methyl_derivative(am, references, mw, wa):
    """N2H7+ -> its one-methyl derivative is a smaller change than any change of heavy atom, as Zundel -> methanol-water was: the
    ordering is not an accident of oxygen. It is still a failure to transfer (several times the derivative's own fit error)."""
    settings = FitSettings(reference_distance=2.7)
    target = make_reference("ammonia_methylamine_hf", am[0].system, am[1], fit_settings(am[0], settings))
    others = {"methanol_water_hf": mw, "water_ammonia_hf": wa}
    others = {name: make_reference(name, experiment.system, solved, fit_settings(experiment, settings)) for name, (experiment, solved) in others.items()}
    parent = references["n2h7_hf"]
    own = cross_predict(target.calibration, target).rmse_window
    forward, backward = cross_predict(parent.calibration, target), cross_predict(target.calibration, parent)
    assert 0.1 < forward.rmse_window < 0.25 and 0.1 < backward.rmse_window < 0.25
    assert forward.rmse_window > 3 * own                                                                      # still a failure to transfer
    for name, other in {"zundel_hf": references["zundel_hf"], **others}.items():
        assert cross_predict(other.calibration, target).rmse_window > 1.5 * forward.rmse_window, name         # every heavy-atom change is farther than the methyl
        assert cross_predict(target.calibration, other).rmse_window > 1.5 * backward.rmse_window, name


# =====================================================================================================================
# a fourth asymmetric reference, and the first anion: fluoride-methanol, [F...H...OCH3]-
# =====================================================================================================================
@pytest.fixture(scope="module")
def fm(pipeline):
    experiment = load_experiment(EXPERIMENTS / "references" / "fluoride_methanol_hf.yaml")
    return experiment, pipeline.run(experiment.system, experiment.propagation).final


def test_the_fluoride_methanol_surface_is_asymmetric_with_the_proton_on_the_oxygen(fm):
    experiment, solved = fm
    molecule = solved.structure["molecule"]
    assert (molecule["template"], len(molecule["atoms"]), molecule["charge"]) == ("fluoride_methanol_anion", 7, -1)
    assert solved.structure["method"]["basis"] == "6-31+g*"                                                   # an anion: diffuse functions
    assert solved.structure["scan"] == {"mirror_symmetric": False} and solved.obs("calculations_run") + solved.obs("calculations_cached") >= 23 * 11
    x, r, e = solved.obs("surface_x"), solved.obs("surface_r"), solved.obs("surface_energy")
    e = e - e.min()
    assert not _is_symmetric(x, e) and x[np.unravel_index(e.argmin(), e.shape)[0]] > 0                       # the proton sits on the oxygen
    wells = [len(prominent_minima(e[:, k])) for k in range(len(r))]
    assert wells == [1] * 4 + [2] * 7                                                                          # one well up to 2.54 A, two from 2.62 A
    for k in range(len(r)):
        if wells[k] == 2:
            donor_side, acceptor_side = (e[i, k] for i in prominent_minima(e[:, k])[:2])
            assert 0.4 < donor_side - acceptor_side < 0.9, r[k]                                              # eV: Hartree-Fock overstates the ~0.4 eV acidity gap


def test_the_model_fits_the_anion_and_the_offset_does_not_follow_the_acidity_gap(fm, am, mw):
    """The fitted offset was ordered by the proton-affinity gap across the three cations. For the anion, whose literature gap (~0.4 eV) is
    smaller than ammonia-methylamine's (0.47), it comes out twice as large and level with methanol-water's: the prediction that it would
    be smaller failed, so the offset is an effective parameter of the fit and not a proxy for the proton-affinity difference."""
    experiment, solved = fm
    settings = fit_settings(experiment, FitSettings(reference_distance=2.7))
    assert settings.window_ev == 2.5
    free = calibrate_evb_2d(solved, settings, cross_validate=False)
    assert free.rmse_ev["2.5"] < 0.06 and free.max_error_ev["2.5"] < 0.25
    assert sum(c.wells_reference != c.wells_model for c in free.columns) <= 1                                 # one distance of eleven has the wrong well count
    assert all(abs(c.barrier_model - c.barrier_reference) < 0.12 for c in free.columns if c.barrier_reference is not None)
    zero = calibrate_evb_2d(solved, replace(settings, bounds=(("diabatic_offset", -1e-9, 1e-9),)), cross_validate=False)
    assert zero.rmse_ev["2.5"] > 4 * free.rmse_ev["2.5"] and sum(c.wells_reference != c.wells_model for c in zero.columns) >= 2
    offset = lambda sys_, sol: calibrate_evb_2d(sol, fit_settings(sys_, FitSettings(reference_distance=2.7)), cross_validate=False).parameters["diabatic_offset"]
    assert -2.2 < free.parameters["diabatic_offset"] < -1.0
    assert free.parameters["diabatic_offset"] < offset(*am) - 0.4                                              # larger in magnitude than the smaller-gap cation's, by a clear margin


def test_distances_never_seen_are_predicted_for_the_anion_too(fm):
    experiment, solved = fm
    cv = calibrate_evb_2d(solved, fit_settings(experiment, FitSettings(reference_distance=2.7))).cross_validation
    assert cv["rmse_mean"] < 0.055 and cv["barrier_error_mean_abs"] < 0.05                                   # leave-one-distance-out, eV


def test_the_nearest_neighbour_shares_the_methyl_oxygen_not_the_symmetric_parent(fm, references, mw, am, wa):
    """FHF- is the symmetric ion this one was built from, but its nearest neighbour is methanol-water, whose acceptor is a methyl-bearing
    oxygen too: the cross-prediction error follows the shared fragment before it follows the parent. It is still a failure to transfer."""
    settings = FitSettings(reference_distance=2.7)
    target = make_reference("fluoride_methanol_hf", fm[0].system, fm[1], fit_settings(fm[0], settings))
    cations = {name: make_reference(name, experiment.system, solved, fit_settings(experiment, settings))
               for name, (experiment, solved) in (("methanol_water_hf", mw), ("ammonia_methylamine_hf", am), ("water_ammonia_hf", wa))}
    others = {"fhf_hf": references["fhf_hf"], "zundel_hf": references["zundel_hf"], "n2h7_hf": references["n2h7_hf"], **cations}
    forward = {name: cross_predict(other.calibration, target).rmse_window for name, other in others.items()}
    backward = {name: cross_predict(target.calibration, other).rmse_window for name, other in others.items()}
    assert min(forward, key=forward.get) == "methanol_water_hf" and min(backward, key=backward.get) == "methanol_water_hf"
    assert forward["methanol_water_hf"] < 0.8 * forward["fhf_hf"] and backward["methanol_water_hf"] < 0.8 * backward["fhf_hf"]    # nearer than the parent
    assert forward["methanol_water_hf"] > 3 * cross_predict(target.calibration, target).rmse_window                              # and still not a transfer
