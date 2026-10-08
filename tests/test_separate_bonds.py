"""The model with a Morse curve of its own for the acceptor bond ("separate bonds"), against closed forms and synthetic surfaces whose true
parameters are known. Real quantum chemistry (the chloride-HF ion, which the shared-curve model cannot fit) is in test_transfer_real.py.

The shared model draws both diabatic states from one Morse curve. When the two bonds differ (Cl-H 1.27 A, F-H 0.92 A) it cannot place both
wells; the separate model lets any of acceptor_morse_depth / _alpha / _r_eq replace the donor's value for the acceptor side.
"""
import json
from dataclasses import replace

import numpy as np
import pytest

from substrate import ClassicalBackend, Quantity, ValidationError
from substrate.calibration import (
    ACCEPTOR_PARAMETERS, BONDS, Calibration, FitSettings, calibrate_evb_2d, calibrated_experiment, evb2d_energy, leave_one_distance_out,
    parameter_specs,
)
from substrate.engines.electronic import EVBFlexibleProtonTransferEngine, EVBProtonTransferEngine
from substrate.experiment import parse_experiment
from substrate.transfer import (
    cross_predict, fit_settings, learning_curve, make_reference, parameter_table, predict_surface, prior_from,
)

from conftest import evb, evb2d

BACKEND = ClassicalBackend()
GRID = dict(x_extent=(0.65, "angstrom"), distance_min=(2.3, "angstrom"), distance_max=(3.0, "angstrom"), n_x=(41, "1"), n_r=(21, "1"))
DONOR = dict(morse_depth=4.6, morse_alpha=2.2, morse_r_eq=0.96)                        # evb2d's defaults
ACCEPTOR = dict(acceptor_morse_depth=(3.9, "eV"), acceptor_morse_alpha=(2.0, "1/angstrom"), acceptor_morse_r_eq=(1.05, "angstrom"))
OFFSET = -0.3
R_REF = 2.65                                                                            # the middle of the distance range 2.3-3.0
TRUTH = {**{n: v[0] for n, v in ACCEPTOR.items()}, **DONOR, "diabatic_offset": OFFSET, "coupling_decay": 3.0, "oo_depth": 0.4, "oo_alpha": 2.5,
         "oo_equilibrium": 2.7}


def solve_2d(**overrides):
    system = evb2d(**overrides)
    for key, value in GRID.items():
        system.parameters[key] = Quantity(*value)
    return system, EVBFlexibleProtonTransferEngine().solve(system, BACKEND)


# -- the model: closed forms ---------------------------------------------------------------------------------------------------
def test_an_acceptor_bond_parameter_left_out_takes_the_donors_value():
    """Each acceptor parameter on its own, against all three given: the two it leaves out must be the donor's."""
    equal = dict(acceptor_morse_depth=(4.6, "eV"), acceptor_morse_alpha=(2.2, "1/angstrom"), acceptor_morse_r_eq=(0.96, "angstrom"))
    shared = solve_2d()[1].obs("surface_energy")
    assert np.array_equal(solve_2d(**equal)[1].obs("surface_energy"), shared)                      # all three equal to the donor's: the shared model
    for name, changed in ACCEPTOR.items():
        alone = solve_2d(**{name: changed})[1].obs("surface_energy")
        spelled_out = solve_2d(**{**equal, name: changed})[1].obs("surface_energy")
        assert np.array_equal(alone, spelled_out), name
        assert np.abs(alone - shared).max() > 1e-3, name                                          # and it does change the surface


def test_exchanging_donor_and_acceptor_mirrors_the_surface_and_shifts_it_by_the_offset():
    """E'(-x) = E(x) - offset when the two bonds swap roles and the offset changes sign: an invariance the model must have, whatever the bonds."""
    forward = solve_2d(diabatic_offset=OFFSET, **ACCEPTOR)[1].obs("surface_energy")
    swapped = solve_2d(diabatic_offset=-OFFSET, morse_depth=ACCEPTOR["acceptor_morse_depth"], morse_alpha=ACCEPTOR["acceptor_morse_alpha"],
                       morse_r_eq=ACCEPTOR["acceptor_morse_r_eq"], acceptor_morse_depth=(4.6, "eV"), acceptor_morse_alpha=(2.2, "1/angstrom"),
                       acceptor_morse_r_eq=(0.96, "angstrom"))[1].obs("surface_energy")
    assert swapped[::-1] == pytest.approx(forward - OFFSET, abs=1e-10)


def test_with_no_coupling_each_well_sits_at_its_own_bond_length_and_its_own_energy():
    """Without coupling the ground state is min(V_A, V_B): the donor well at x = r_eq(donor) - R/2 at energy 0, the acceptor well at
    x = R/2 - r_eq(acceptor) at the offset. Cl...F at 3.5 A, bonds 1.29 and 0.93 A."""
    system = evb(R=3.5, depth=8.4, alpha=1.9, r_eq=1.29, coupling=1e-9, offset=-2.0)
    system.parameters.update({"acceptor_morse_depth": Quantity(7.2, "eV"), "acceptor_morse_alpha": Quantity(2.3, "1/angstrom"),
                              "acceptor_morse_r_eq": Quantity(0.93, "angstrom")})
    solved = EVBProtonTransferEngine().solve(system, BACKEND)
    x, e = solved.obs("scan_coordinate"), solved.obs("scan_energy")
    left, right = x < 0.2, x > 0.2
    i, j = np.flatnonzero(left)[e[left].argmin()], np.flatnonzero(right)[e[right].argmin()]
    assert x[i] == pytest.approx(1.29 - 1.75, abs=0.006) and e[i] == pytest.approx(0.0, abs=1e-3)
    assert x[j] == pytest.approx(1.75 - 0.93, abs=0.006) and e[j] == pytest.approx(-2.0, abs=1e-3)
    assert solved.obs("classical_reaction_energy") == pytest.approx(-2.0, abs=1e-3)               # product minus reactant


def test_the_closed_form_energy_equals_the_engines_diagonalisation_with_separate_bonds():
    system, solved = solve_2d(diabatic_offset=OFFSET, **ACCEPTOR)
    x, r, e = solved.obs("surface_x"), solved.obs("surface_r"), solved.obs("surface_energy")
    X, R = np.meshgrid(x, r, indexing="ij")
    p = {**DONOR, **{n: v[0] for n, v in ACCEPTOR.items()}, "coupling": 0.6, "coupling_decay": 3.0, "oo_depth": 0.4, "oo_alpha": 2.5,
         "oo_equilibrium": 2.7, "diabatic_offset": OFFSET}
    assert evb2d_energy(X, R, p, 2.5) == pytest.approx(e, abs=1e-10)
    del p["acceptor_morse_r_eq"]                                                                    # left out: the donor's, as in the engine
    assert evb2d_energy(X, R, p, 2.5) != pytest.approx(e, abs=1e-3)


@pytest.mark.parametrize("name, bad, message", [
    ("acceptor_morse_depth", (-1.0, "eV"), "must be positive"), ("acceptor_morse_alpha", (0.0, "1/angstrom"), "must be positive"),
    ("acceptor_morse_r_eq", (-0.9, "angstrom"), "must be positive"), ("acceptor_morse_r_eq", (0.9, "eV"), "expected 'angstrom'"),
    ("acceptor_morse_depth", (4.0, "angstrom"), "expected 'eV'")])
def test_an_unphysical_acceptor_bond_is_refused(name, bad, message):
    with pytest.raises(ValidationError, match=message):
        solve_2d(**{name: bad})
    one_d = evb()
    one_d.parameters[name] = Quantity(*bad)
    with pytest.raises(ValidationError, match=message):
        EVBProtonTransferEngine().solve(one_d, BACKEND)


# -- calibration: recovery and refusal -------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def target():
    return solve_2d(diabatic_offset=OFFSET, **ACCEPTOR)


@pytest.fixture(scope="module")
def separate(target):
    return calibrate_evb_2d(target[1], FitSettings(bonds="separate"))


@pytest.fixture(scope="module")
def shared(target):
    return calibrate_evb_2d(target[1], FitSettings(bonds="shared"), cross_validate=False)


def test_the_separate_model_recovers_its_own_parameters_from_a_noise_free_surface(separate):
    cal = separate
    assert cal.bonds == "separate" and cal.settings["bonds"] == "separate" and cal.fixed["reference_distance"] == pytest.approx(R_REF)
    for name, truth in TRUTH.items():
        assert cal.parameters[name] == pytest.approx(truth, rel=2e-3), name
    assert cal.parameters["coupling"] == pytest.approx(0.6 * np.exp(-3.0 * (R_REF - 2.5)), rel=2e-3)      # the coupling, defined at 2.65 A here
    assert cal.rmse_ev["1.5"] < 1e-5 and cal.pinned == {} and cal.n_free == 13                              # 12 parameters and an energy zero
    assert set(cal.parameters) == {s.name for s in parameter_specs(FitSettings(bonds="separate"), False)}
    assert cal.cross_validation["rmse_max"] < 1e-3 and cal.cross_validation["barrier_error_max_abs"] < 1e-3   # unseen distances are predicted exactly


def test_the_shared_model_cannot_fit_a_surface_whose_bonds_differ(separate, shared):
    assert shared.bonds == "shared" and not any(n in shared.parameters for n in (s.name for s in ACCEPTOR_PARAMETERS))
    assert shared.rmse_ev["1.5"] > 0.1 and shared.rmse_ev["1.5"] > 1e4 * separate.rmse_ev["1.5"]            # 0.17 eV against 1e-7


def test_separate_bonds_are_refused_for_a_symmetric_surface_and_an_unknown_model_is_refused(target):
    symmetric = solve_2d(diabatic_offset=0.0)[1]
    with pytest.raises(ValidationError, match="asymmetric surface"):
        calibrate_evb_2d(symmetric, FitSettings(bonds="separate"), cross_validate=False)
    with pytest.raises(ValidationError, match="bonds must be one of"):
        calibrate_evb_2d(target[1], FitSettings(bonds="both"), cross_validate=False)
    assert BONDS == ("shared", "separate") and calibrate_evb_2d(symmetric, cross_validate=False).bonds == "shared"


def test_the_free_parameters_are_the_shared_ones_then_the_acceptors_then_the_offset():
    names = lambda bonds, symmetric: [s.name for s in parameter_specs(FitSettings(bonds=bonds), symmetric)]
    assert names("shared", True) == ["morse_depth", "morse_alpha", "morse_r_eq", "coupling", "coupling_decay", "oo_depth", "oo_alpha", "oo_equilibrium"]
    assert names("shared", False) == names("shared", True) + ["diabatic_offset"]
    assert names("separate", False) == names("shared", True) + ["acceptor_morse_depth", "acceptor_morse_alpha", "acceptor_morse_r_eq", "diabatic_offset"]


def test_a_bound_can_be_given_for_an_acceptor_parameter_and_a_prior_on_one_is_refused_for_a_shared_fit(target):
    pinned = calibrate_evb_2d(target[1], FitSettings(bonds="separate", bounds=(("acceptor_morse_r_eq", 0.9, 1.0),)), cross_validate=False)
    assert pinned.pinned["acceptor_morse_r_eq"] == "upper" and pinned.rmse_ev["1.5"] > 1e-3                 # the true 1.05 is out of reach
    with pytest.raises(ValidationError, match="not a free parameter"):
        calibrate_evb_2d(target[1], FitSettings(prior=(("acceptor_morse_r_eq", 1.0, 0.1),)), cross_validate=False)


def test_the_summary_says_which_bond_model_was_fitted_and_lists_the_acceptor_parameters(separate, shared):
    assert "(separate bonds)" in separate.summary() and "acceptor_morse_r_eq" in separate.summary()
    assert "(shared bonds)" in shared.summary() and "acceptor_morse_r_eq" not in shared.summary()


def test_a_calibration_keeps_its_bond_model_through_json_and_an_older_file_means_shared(separate, shared, tmp_path):
    path = tmp_path / "c.json"
    separate.save(path)
    again = Calibration.load(path)
    assert again.bonds == "separate" and again.parameters == separate.parameters and again.summary() == separate.summary()
    old = json.loads(json.dumps(shared.to_dict()))
    del old["settings"]["bonds"]                                                                    # saved before the setting existed
    assert Calibration.from_dict(old).bonds == "shared"


# -- the calibrated systems --------------------------------------------------------------------------------------------------------
def test_the_calibrated_system_carries_the_acceptor_bond_and_reproduces_the_reference(target, separate):
    system = separate.system(scan_distance=2.8)
    for name, spec in ((s.name, s) for s in ACCEPTOR_PARAMETERS):
        assert system.parameters[name].value == pytest.approx(separate.parameters[name]) and system.parameters[name].unit == spec.unit
        assert system.parameters[name].sigma == pytest.approx(separate.sigma[name])
    block = system.structure["parameter_covariance"]
    assert {"acceptor_morse_depth", "acceptor_morse_alpha", "acceptor_morse_r_eq"} <= set(block["names"])
    assert (block["minimum"]["acceptor_morse_depth"], block["minimum"]["acceptor_morse_alpha"], block["minimum"]["acceptor_morse_r_eq"]) == (0.5, 0.3, 0.7)
    assert "diabatic_offset" not in block["minimum"]
    model = EVBFlexibleProtonTransferEngine().solve(system, BACKEND).obs("surface_energy")
    reference = target[1].obs("surface_energy")
    assert model - model.min() == pytest.approx(reference - reference.min(), abs=2e-4)
    assert "acceptor_morse_depth" not in calibrate_evb_2d(solve_2d()[1], cross_validate=False).system().parameters      # a shared calibration carries none


def test_the_one_dimensional_model_of_a_separate_calibration_is_its_two_dimensional_slice(separate):
    system_2d = separate.system(scan_distance=2.8, with_uncertainty=False)
    system_2d.parameters["n_x"] = Quantity(241, "1")
    two_d = EVBFlexibleProtonTransferEngine().solve(system_2d, BACKEND)
    one_d = EVBProtonTransferEngine().solve(separate.system("electronic.evb_two_state", distance=2.8), BACKEND)
    assert one_d.param("acceptor_morse_r_eq") == pytest.approx(separate.parameters["acceptor_morse_r_eq"])
    shift = two_d.obs("scan_energy") - one_d.obs("scan_energy")
    assert shift == pytest.approx(np.full_like(shift, shift[0]), abs=1e-9)
    assert one_d.obs("classical_barrier") == pytest.approx(two_d.obs("classical_barrier"), abs=1e-9)


def test_a_calibrated_experiment_file_carries_the_acceptor_bond(separate):
    spec = calibrated_experiment(separate, {"experiment": {"id": "t", "system": {"parameters": {}}}})
    parameters = spec["experiment"]["system"]["parameters"]
    assert parameters["acceptor_morse_r_eq"]["value"] == pytest.approx(separate.parameters["acceptor_morse_r_eq"])
    assert parameters["acceptor_morse_r_eq"]["unit"] == "angstrom" and "sigma" in parameters["acceptor_morse_r_eq"]


# -- an experiment file declares the bond model --------------------------------------------------------------------------------------
def _raw(hints):
    return {"experiment": {"id": "x", "system": {"name": "n", "scale": "electronic_structure", "kind": "k"}, "calibration": hints}}


def test_an_experiment_file_can_declare_the_bond_model_with_or_without_a_window():
    both = parse_experiment(_raw({"window_ev": 4.0, "bonds": "separate"}))
    assert both.notes == {"calibration": {"window_ev": 4.0, "bonds": "separate"}}
    tuned = fit_settings(both, FitSettings(sigma_ev=0.02))
    assert (tuned.window_ev, tuned.bonds, tuned.sigma_ev) == (4.0, "separate", 0.02)
    alone = parse_experiment(_raw({"bonds": "separate"}))
    assert fit_settings(alone, FitSettings(window_ev=1.2)).window_ev == 1.2 and fit_settings(alone).bonds == "separate"
    assert fit_settings(parse_experiment(_raw({"window_ev": 2.5}))).bonds == "shared"                  # a window alone leaves the bond model alone
    assert fit_settings(parse_experiment(_raw({"window_ev": 2.5})), FitSettings(bonds="separate")).bonds == "separate"
    assert fit_settings(parse_experiment(_raw({"bonds": "shared"})), FitSettings(bonds="separate")).bonds == "shared"          # a declaration overrides the base both ways
    window = parse_experiment(_raw({"window_ev": 4, "bonds": "separate"})).notes["calibration"]["window_ev"]
    assert window == 4.0 and isinstance(window, float)                                                # an integer in the file is read as a number of eV
    for bad in ({"bonds": "both"}, {"bonds": 1}, {"bonds": None}, {"bonds": True}, {"bonds": ["separate"]}, {"bonds": "Separate"}):
        with pytest.raises(ValidationError, match="takes only a positive `window_ev` and a `bonds` of shared or separate"):
            parse_experiment(_raw(bad))


# -- transfer ---------------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def references(target):
    settings = FitSettings(reference_distance=R_REF)
    symmetric_system, symmetric_solved = solve_2d(diabatic_offset=0.0)
    return {"unlike": make_reference("unlike", target[0], target[1], replace(settings, bonds="separate")),
            "like": make_reference("like", symmetric_system, symmetric_solved, settings)}


def test_the_parameter_table_has_acceptor_rows_only_when_a_reference_was_fitted_with_separate_bonds(references):
    table = parameter_table(references)
    names = list(table)
    assert names[-4:] == ["acceptor_morse_depth", "acceptor_morse_alpha", "acceptor_morse_r_eq", "diabatic_offset"]
    assert table["acceptor_morse_r_eq"]["unlike"][0] == pytest.approx(1.05, rel=2e-3) and table["acceptor_morse_r_eq"]["like"] == (None, None)
    assert not any(n.startswith("acceptor") for n in parameter_table({"like": references["like"]}))


def test_cross_prediction_uses_the_acceptor_bond_of_a_separate_source(references, target):
    unlike, like = references["unlike"], references["like"]
    assert cross_predict(unlike.calibration, unlike).rmse_window < 1e-5                                # the truth, applied to its own surface
    assert cross_predict(like.calibration, unlike).rmse_window > 0.05                                  # a shared-bond source cannot reproduce it
    x, r = np.asarray(unlike.x), np.asarray(unlike.r)
    X, R = np.meshgrid(x, r, indexing="ij")
    p = {**unlike.calibration.parameters, **unlike.calibration.fixed}
    assert predict_surface(unlike.calibration, x, r) == pytest.approx(evb2d_energy(X, R, p, R_REF), abs=1e-12)


def test_a_learning_curve_fits_the_targets_own_bond_model_whatever_the_settings_say(references):
    unlike, like = references["unlike"], references["like"]
    shared_settings = FitSettings(reference_distance=R_REF)                                           # says shared; the target is separate
    points = learning_curve(unlike, counts=(6,), settings=shared_settings)
    assert points[0].rmse_ev < 1e-3 and points[0].wells_wrong == 0                                    # a shared fit of this surface misses by 0.17 eV
    assert learning_curve(like, counts=(6,), settings=replace(shared_settings, bonds="separate"))[0].rmse_ev < 1e-3   # a symmetric target stays shared


def test_a_prior_entry_the_target_has_no_parameter_for_is_left_out_not_refused(references):
    unlike, like = references["unlike"], references["like"]
    settings = FitSettings(reference_distance=R_REF)
    assert {"acceptor_morse_depth", "acceptor_morse_alpha", "acceptor_morse_r_eq"} <= {n for n, _, _ in prior_from(unlike.calibration, 0.3)}
    onto_shared = learning_curve(like, counts=(0, 2), source=unlike.calibration, relative_sigma=0.3, settings=settings)
    assert [p.n_distances for p in onto_shared] == [0, 2] and all(np.isfinite(p.rmse_ev) for p in onto_shared)    # acceptor and offset entries dropped
    onto_separate = learning_curve(unlike, counts=(2,), source=like.calibration, relative_sigma=0.3, settings=settings)
    assert np.isfinite(onto_separate[0].rmse_ev)                                                       # a shared source has no acceptor entries to give


def test_the_own_fits_table_names_the_bond_model(references):
    from substrate.transfer import format_own_fits, transfer_matrix
    table = format_own_fits(references, transfer_matrix(references)).splitlines()
    assert table[0].split()[-1] == "bonds" and table[1].split()[-1] == "separate" and table[2].split()[-1] == "shared"


# -- the command line -----------------------------------------------------------------------------------------------------------
YAML = """\
experiment:
  id: {name}
  phenomenon: test
  system:
    name: {name}
    scale: electronic_structure
    kind: electronic.evb_two_state_2d
    parameters:
      morse_depth: {{value: 4.6, unit: eV}}
      morse_alpha: {{value: 2.2, unit: 1/angstrom}}
      morse_r_eq: {{value: 0.96, unit: angstrom}}
      coupling: {{value: 0.6, unit: eV}}
      coupling_decay: {{value: 3.0, unit: 1/angstrom}}
      reference_distance: {{value: 2.5, unit: angstrom}}
      oo_depth: {{value: 0.4, unit: eV}}
      oo_alpha: {{value: 2.5, unit: 1/angstrom}}
      oo_equilibrium: {{value: 2.7, unit: angstrom}}
      diabatic_offset: {{value: {offset}, unit: eV}}
{acceptor}      particle_mass: {{value: 1.007276, unit: amu}}
      heavy_atom_mass: {{value: 15.9949, unit: amu}}
      temperature: {{value: 300, unit: K}}
      x_extent: {{value: 0.65, unit: angstrom}}
      distance_min: {{value: 2.3, unit: angstrom}}
      distance_max: {{value: 3.0, unit: angstrom}}
      n_x: {{value: 41, unit: "1"}}
      n_r: {{value: 21, unit: "1"}}
  propagation: [electronic_structure]
{hint}"""
ACCEPTOR_YAML = ("      acceptor_morse_depth: {value: 3.9, unit: eV}\n      acceptor_morse_alpha: {value: 2.0, unit: 1/angstrom}\n"
                 "      acceptor_morse_r_eq: {value: 1.05, unit: angstrom}\n")


def _write(tmp_path, name, offset, acceptor=True, hint=""):
    path = tmp_path / f"{name}.yaml"
    path.write_text(YAML.format(name=name, offset=offset, acceptor=ACCEPTOR_YAML if acceptor else "", hint=hint), encoding="utf-8")
    return path


def test_the_calibrate_command_fits_separate_bonds_when_asked_and_refuses_a_symmetric_surface(tmp_path, capsys):
    from substrate import load_experiment
    from substrate.cli import main
    unlike, like, out = _write(tmp_path, "unlike", OFFSET), _write(tmp_path, "like", 0.0, acceptor=False), tmp_path / "calibrated.yaml"
    assert main(["calibrate", str(unlike), "--bonds", "separate", "--no-validation", "--out", str(out), "--samples", "4"]) == 0
    assert "(separate bonds)" in capsys.readouterr().out
    assert load_experiment(out).system.param("acceptor_morse_r_eq", "angstrom") == pytest.approx(1.05, rel=2e-3)
    assert main(["calibrate", str(unlike), "--no-validation"]) == 0 and "(shared bonds)" in capsys.readouterr().out       # the default is unchanged
    assert main(["calibrate", str(like), "--bonds", "separate", "--no-validation"]) == 1
    assert "asymmetric surface" in capsys.readouterr().err


def test_the_transfer_command_honours_the_declared_bond_model_even_when_the_window_is_overridden(tmp_path, capsys):
    from substrate.cli import main
    unlike = _write(tmp_path, "unlike", OFFSET, hint="  calibration: {window_ev: 2.0, bonds: separate}\n")
    like = _write(tmp_path, "like", 0.0)
    saved = tmp_path / "t.json"
    assert main(["transfer", str(unlike), str(like), "--reference-distance", "2.65", "--window", "1.2", "--json", str(saved)]) == 0
    report = capsys.readouterr().out
    rows = {}
    for line in report.splitlines():                                                                # the first table is the own-fits one
        if line.startswith(("unlike", "like")):
            rows.setdefault(line.split()[0], line.split())
    assert rows["unlike"][-1] == "separate" and rows["like"][-1] == "shared"                          # the file's hint survives --window
    assert rows["unlike"][1] == "1.2" and rows["like"][1] == "1.2"                                  # while the window is the override, for both (the file said 2.0)
    assert json.loads(saved.read_text())["unlike -> unlike"]["rmse_ev"]["1.2"] < 1e-5
    capsys.readouterr()
    assert main(["transfer", str(unlike), str(like), "--reference-distance", "2.65", "--window", "0.02"]) == 1      # the override is the fit window too:
    assert "too few reference points" in capsys.readouterr().err                                                  # 0.02 eV leaves too little to fit
