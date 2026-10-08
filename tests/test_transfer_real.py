"""Transfer across molecules, against REAL quantum chemistry (Hartree-Fock energies from PySCF for the Zundel cation, the ammonium
dimer and the bifluoride anion, and the asymmetric water-ammonia ion). Skipped where PySCF is unavailable; the first run computes ~540
single points and later runs read them from the persistent test cache. (The count above is for the first four molecules; the methanol-water, ammonia-methylamine and fluoride-methanol ions add ~270 each, the chloride-HF ion ~300.)

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
from substrate.transfer import baselines, cross_predict, fit_settings, learning_curve, make_reference, rate_comparison, transfer_matrix

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


# =====================================================================================================================
# a fifth asymmetric reference, the strongest: chloride-HF, [Cl...H...F]-, which needs a Morse curve of its own for each bond
# =====================================================================================================================
@pytest.fixture(scope="module")
def cl(pipeline):
    experiment = load_experiment(EXPERIMENTS / "references" / "chloride_hf_hf.yaml")
    return experiment, pipeline.run(experiment.system, experiment.propagation).final


def _double_wells(x, r, e):
    """[(R, donor-side x, acceptor-side x, barrier top energy)] at every distance with two wells (x is the proton position, donor at -R/2)."""
    out = []
    for k, distance in enumerate(r):
        minima = prominent_minima(e[:, k])
        if len(minima) == 2:
            out.append((float(distance), float(x[minima[0]]), float(x[minima[1]]), float(e[minima[0]:minima[1] + 1, k].max())))
    return out


def test_the_chloride_hf_surface_is_strongly_asymmetric_with_the_proton_on_the_fluorine(cl):
    experiment, solved = cl
    molecule = solved.structure["molecule"]
    assert (molecule["template"], len(molecule["atoms"]), molecule["charge"]) == ("chloride_hf_anion", 3, -1)
    assert solved.structure["scan"] == {"mirror_symmetric": False} and solved.obs("calculations_run") + solved.obs("calculations_cached") >= 27 * 11
    x, r, e = solved.obs("surface_x"), solved.obs("surface_r"), solved.obs("surface_energy")
    e = e - e.min()
    assert not _is_symmetric(x, e) and x[np.unravel_index(e.argmin(), e.shape)[0]] > 0                       # the proton sits on the fluorine
    wells = [len(prominent_minima(e[:, k])) for k in range(len(r))]
    assert wells == [1] * 3 + [2] * 8                                                                          # one well up to 3.02 A, two from 3.13 A
    for k in range(len(r)):
        if wells[k] == 2:
            donor_side, acceptor_side = (e[i, k] for i in prominent_minima(e[:, k])[:2])
            assert 1.4 < donor_side - acceptor_side < 2.2, r[k]                                              # eV: far wider than any other reference's gap


def test_the_barrier_tops_need_a_window_beyond_the_older_rule_and_the_file_declares_it(cl):
    experiment, solved = cl
    x, r, e = solved.obs("surface_x"), solved.obs("surface_r"), solved.obs("surface_energy")
    tops = [top for *_, top in _double_wells(x, r, e - e.min())]
    assert 3.5 < max(tops) < 4.0 and min(tops) < 2.0                                                          # beyond 3.0 eV, so 4.0 is the smallest candidate reaching all
    settings = fit_settings(experiment, FitSettings(reference_distance=2.7))
    assert (settings.window_ev, settings.bonds) == (4.0, "separate")      # the file declares both


def test_the_two_bonds_have_different_lengths_which_one_shared_morse_curve_cannot_give(cl):
    """The model draws both diabatic states from ONE Morse curve. In the real wells the Cl-H bond is 1.25-1.37 A and the F-H bond 0.90-0.98 A
    (the grid step is 0.1 A), at every double-well distance, by at least 0.25 A."""
    experiment, solved = cl
    x, r, e = solved.obs("surface_x"), solved.obs("surface_r"), solved.obs("surface_energy")
    wells = _double_wells(x, r, e - e.min())
    cl_h = [distance / 2 + donor for distance, donor, _, _ in wells]
    f_h = [distance / 2 - acceptor for distance, _, acceptor, _ in wells]
    assert all(1.2 < a < 1.42 for a in cl_h) and all(0.85 < b < 1.03 for b in f_h)
    assert min(a - b for a, b in zip(cl_h, f_h)) > 0.25


def test_the_shared_morse_model_cannot_fit_it_where_every_other_asymmetric_reference_is_fitted(cl):
    experiment, solved = cl
    settings = replace(fit_settings(experiment, FitSettings(reference_distance=2.7)), bonds="shared")        # the file declares separate; this is the other
    free = calibrate_evb_2d(solved, settings, cross_validate=False)
    assert free.rmse_ev["4"] > 0.3 and free.max_error_ev["4"] > 1.0                                           # eV; the others are 0.04-0.05 and below 0.25
    assert sum(c.wells_reference != c.wells_model for c in free.columns) >= 6                                 # the fitted surface has (almost) no double well
    assert free.pinned.get("diabatic_offset") == "lower" and free.parameters["diabatic_offset"] < -9.9       # the offset runs to the (widened) bound: no meaning


def test_a_curve_of_its_own_for_each_bond_fits_it_as_well_as_the_other_asymmetric_references(cl):
    """The diagnosis, and the fix: the same surface, window and coupling, with separate Morse depth, width and equilibrium length for the
    chlorine and fluorine sides. 0.036 eV against 0.41 eV, every well count right, nothing on a bound, and bond lengths near those of the
    free diatomics (HCl 1.27 A, HF 0.92 A) although only the surface was fitted."""
    experiment, solved = cl
    settings = fit_settings(experiment, FitSettings(reference_distance=2.7))
    free = calibrate_evb_2d(solved, settings)
    assert free.bonds == "separate" and free.rmse_ev["4"] < 0.05 and free.max_error_ev["4"] < 0.2
    assert sum(c.wells_reference != c.wells_model for c in free.columns) == 0 and free.pinned == {}
    assert all(abs(c.barrier_model - c.barrier_reference) < 0.12 for c in free.columns if c.barrier_reference is not None)
    assert free.starts_agreeing >= 10                                                                         # of 12: the optimum is not a lucky start
    assert abs(free.parameters["morse_r_eq"] - 1.27) < 0.05 and abs(free.parameters["acceptor_morse_r_eq"] - 0.92) < 0.05
    assert free.parameters["acceptor_morse_alpha"] > free.parameters["morse_alpha"]                          # the stiffer, shorter fluorine bond
    assert -7.0 < free.parameters["diabatic_offset"] < -4.0 and free.sigma["diabatic_offset"] < 1.0           # interior, and well above the nearest asymmetric ion's
    assert free.cross_validation["rmse_mean"] < 0.05 and free.cross_validation["barrier_error_mean_abs"] < 0.06
    shared = calibrate_evb_2d(solved, replace(settings, bonds="shared"), cross_validate=False)
    assert shared.rmse_ev["4"] > 10 * free.rmse_ev["4"]


def test_the_other_asymmetric_references_gain_a_little_not_a_lot_from_a_curve_of_their_own(fm, mw, am, wa):
    """Not a new model for them: fitted with separate bonds they improve by 0.006-0.03 eV (leave-one-distance-out too), against 0.38 eV for the
    chloride-HF ion, and their two bond lengths come out within 0.02 A of each other."""
    for experiment, solved in (fm, mw, am, wa):
        settings = fit_settings(experiment, FitSettings(reference_distance=2.7))
        window = f"{settings.window_ev:g}"
        shared = calibrate_evb_2d(solved, settings, cross_validate=False)
        separate = calibrate_evb_2d(solved, replace(settings, bonds="separate"))
        assert 0.0 < shared.rmse_ev[window] - separate.rmse_ev[window] < 0.04, experiment.id
        assert separate.cross_validation["rmse_mean"] < 0.04 and abs(separate.parameters["morse_r_eq"] - separate.parameters["acceptor_morse_r_eq"]) < 0.03


def test_no_other_reference_transfers_either_way_and_the_bifluoride_is_not_its_nearest_neighbour(cl, references, mw, am, wa, fm):
    """Recorded prediction: the nearest hydrogen-bond neighbour would be the symmetric bifluoride (one fluoride swapped for chloride). It was
    not, in either direction. Into this surface the other ions score 0.68-1.0 eV (the best constant 1.06); this ion's parameters, now a good
    fit, are worse on theirs (1.6-2.4 eV against constants of 0.3-0.7): its large offset and wide bonds mean nothing to them."""
    settings = FitSettings(reference_distance=2.7)
    target = make_reference("chloride_hf_hf", cl[0].system, cl[1], fit_settings(cl[0], settings))
    others = {"fhf_hf": references["fhf_hf"], "zundel_hf": references["zundel_hf"], "n2h7_hf": references["n2h7_hf"]}
    others.update({name: make_reference(name, experiment.system, solved, fit_settings(experiment, settings))
                   for name, (experiment, solved) in (("methanol_water_hf", mw), ("ammonia_methylamine_hf", am), ("water_ammonia_hf", wa),
                                                     ("fluoride_methanol_hf", fm))})
    forward = {name: cross_predict(other.calibration, target).rmse_window for name, other in others.items()}
    backward = {name: cross_predict(target.calibration, other).rmse_window for name, other in others.items()}
    assert min(forward.values()) > 0.5                                                                         # eV over the 4 eV window; the other pairs' own fits are below 0.06
    assert min(forward, key=forward.get) != "fhf_hf" and forward["fhf_hf"] > forward["fluoride_methanol_hf"]
    assert min(backward.values()) > 1.0 and backward["fhf_hf"] == max(backward.values())                      # the symmetric parent is the worst of all, not the nearest
    assert all(backward[name] > 2 * baselines(other)["constant_rmse"] for name, other in others.items())      # worse than a constant on every one of them


# -- the chloride-HF ion at the other two methods, and its rates ------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cl_methods(pipeline):
    out = {}
    for tag in ("b3lyp", "mp2"):
        experiment = load_experiment(EXPERIMENTS / "references" / f"chloride_hf_{tag}.yaml")
        out[tag] = (experiment, pipeline.run(experiment.system, experiment.propagation).final)
    return out


def _window_rule(tops):
    """The rule every asymmetric reference's fit window follows: the smallest candidate that reaches every double-well barrier top."""
    return next(w for w in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0) if w >= max(tops))


def test_the_three_chloride_hf_files_share_the_window_the_rule_gives_for_the_hartree_fock_tops(cl, cl_methods):
    """One window per molecule, from its Hartree-Fock surface and kept for the other methods, as for every asymmetric reference: the three
    surfaces are fitted over the same range. The other two methods' own tops (2.79, 3.02 eV) would have given 3.0 and 3.5 eV."""
    def tops(solved):
        x, r, e = solved.obs("surface_x"), solved.obs("surface_r"), solved.obs("surface_energy")
        return [top for *_, top in _double_wells(x, r, e - e.min())]
    assert _window_rule(tops(cl[1])) == 4.0
    for tag, (experiment, solved) in (("hf", cl), *cl_methods.items()):
        settings = fit_settings(experiment, FitSettings(reference_distance=2.7))
        assert (settings.window_ev, settings.bonds) == (4.0, "separate"), tag
        assert max(tops(solved)) <= settings.window_ev, tag
    assert [_window_rule(tops(cl_methods[t][1])) for t in ("b3lyp", "mp2")] == [3.0, 3.5] and max(tops(cl_methods["b3lyp"][1])) == pytest.approx(2.79, abs=0.01)


def test_the_chloride_hf_surface_keeps_its_shape_at_b3lyp_and_mp2_with_a_smaller_gap(cl_methods):
    for tag, first_double, (gap_low, gap_high) in (("b3lyp", 3.35, (1.2, 1.6)), ("mp2", 3.24, (1.1, 1.55))):
        experiment, solved = cl_methods[tag]
        assert solved.structure["method"]["basis"] == "6-31+g*" and solved.structure["molecule"]["template"] == "chloride_hf_anion"
        x, r, e = solved.obs("surface_x"), solved.obs("surface_r"), solved.obs("surface_energy")
        e = e - e.min()
        assert not _is_symmetric(x, e) and x[np.unravel_index(e.argmin(), e.shape)[0]] > 0                  # the proton stays on the fluorine
        wells = [len(prominent_minima(e[:, k])) for k in range(len(r))]
        assert min(r[k] for k in range(len(r)) if wells[k] == 2) == pytest.approx(first_double, abs=0.01), tag   # later than Hartree-Fock's 3.13 A
        assert all(wells[k] == (2 if r[k] >= first_double - 0.01 else 1) for k in range(len(r))), tag
        gaps = [e[prominent_minima(e[:, k])[0], k] - e[prominent_minima(e[:, k])[1], k] for k in range(len(r)) if wells[k] == 2]
        assert gap_low < min(gaps) and max(gaps) < gap_high, tag                                              # smaller than Hartree-Fock's 1.60-1.97 eV


def test_separate_bonds_fit_the_other_two_methods_as_well_as_hartree_fock_and_the_shared_curve_fails_again(cl_methods):
    for tag in ("b3lyp", "mp2"):
        experiment, solved = cl_methods[tag]
        settings = fit_settings(experiment, FitSettings(reference_distance=2.7))
        window = f"{settings.window_ev:g}"
        free = calibrate_evb_2d(solved, settings)
        assert free.rmse_ev[window] < 0.036 and free.max_error_ev[window] < 0.15, tag                       # 0.031 and 0.094-0.126 eV
        assert sum(c.wells_reference != c.wells_model for c in free.columns) == 0, tag
        assert all(abs(c.barrier_model - c.barrier_reference) < 0.1 for c in free.columns if c.barrier_reference is not None), tag
        assert 1.2 < free.parameters["morse_r_eq"] < 1.4 and 0.9 < free.parameters["acceptor_morse_r_eq"] < 1.0, tag
        assert free.cross_validation["rmse_mean"] < 0.035, tag
        shared = calibrate_evb_2d(solved, replace(settings, bonds="shared"), cross_validate=False)
        assert shared.rmse_ev[window] > 8 * free.rmse_ev[window] and sum(c.wells_reference != c.wells_model for c in shared.columns) >= 5, tag     # 0.41-0.44 eV


def test_at_b3lyp_and_mp2_the_offset_sits_on_its_bound_and_widening_it_changes_nothing_that_matters(cl_methods):
    """Hartree-Fock puts the offset inside its +-10 eV bound (-5.4 eV). The other two put it on the bound, because it trades off against a
    coupling of 17-20 eV (B3LYP's is on its own bound, 20): with the bounds moved far out (-40 eV, coupling up to 80) the offset settles at
    -33 eV (coupling 40, a nearly singular fit) for B3LYP and -10.5 eV for MP2, the fit better by 0.002 and 0.0000 eV. So the offset is an
    effective parameter of a long valley, not a number to compare between ions, and the fit hardly cares."""
    for tag in ("b3lyp", "mp2"):
        experiment, solved = cl_methods[tag]
        settings = fit_settings(experiment, FitSettings(reference_distance=2.7))
        window = f"{settings.window_ev:g}"
        tight = calibrate_evb_2d(solved, settings, cross_validate=False)
        wide = calibrate_evb_2d(solved, replace(settings, bounds=(("diabatic_offset", -40.0, 10.0), ("coupling", 0.0, 80.0))), cross_validate=False)
        assert tight.pinned["diabatic_offset"] == "lower" and tight.parameters["diabatic_offset"] < -9.99, tag
        assert wide.parameters["diabatic_offset"] < -10.4 and wide.parameters["coupling"] > tight.parameters["coupling"] * 0.95, tag
        assert 0.0 <= tight.rmse_ev[window] - wide.rmse_ev[window] < 0.003, tag


def test_the_model_chain_runs_on_the_chloride_hf_ion_and_its_own_rate_is_within_a_factor_of_four(cl, pipeline):
    """Its grid (2.8-3.9 A) lies above the 2.7 A at which every study fit defines its coupling: the engine accepts a reference distance outside
    the scan range, so the model chain runs (it refused before). The calibration's own rate, at the distances where the barrier is 0.15, 0.4 and
    0.8 eV, is 0.26-0.43 of the real chain's."""
    settings = FitSettings(reference_distance=2.7)
    target = make_reference("chloride_hf_hf", cl[0].system, cl[1], fit_settings(cl[0], settings))
    rows = rate_comparison({"chloride_hf_hf": target}, {"chloride_hf_hf": target}, pipeline)
    assert len(rows) == 3 and all(row.ratio is not None for row in rows), [row.note for row in rows]
    assert all(0.2 < row.ratio < 1.0 for row in rows), [row.ratio for row in rows]
