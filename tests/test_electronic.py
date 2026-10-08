import math

import numpy as np
import pytest

from substrate import ClassicalBackend, NoPathError, Quantity, Scale, ScientificSystem, ValidationError
from substrate.engines.electronic import EVBFlexibleProtonTransferEngine, EVBProtonTransferEngine
from substrate.engines.quantum import DoubleWellEngine, TabulatedPotentialEngine, double_well
from substrate.pes import locate_wells, prominent_minima
from substrate.translators.electronic_to_quantum import ElectronicToQuantum

from conftest import DEUTERON, evb, evb2d
from conftest import double_well as quartic_system


def solve(**kw):
    return EVBProtonTransferEngine().solve(evb(**kw), ClassicalBackend())


def morse(r, depth=4.6, alpha=2.2, r_eq=0.96):
    return depth * (1.0 - np.exp(-alpha * (r - r_eq))) ** 2


# -- the electronic engine ---------------------------------------------------------------------
def test_ground_state_matches_the_closed_form_two_level_energy():
    out = solve(R=2.5, coupling=0.6, offset=0.05)
    x = out.obs("scan_coordinate")
    v_a, v_b = morse(1.25 + x), morse(1.25 - x) + 0.05
    half_sum, root = 0.5 * (v_a + v_b), np.sqrt((0.5 * (v_a - v_b)) ** 2 + 0.6**2)
    assert out.obs("scan_energy") == pytest.approx(half_sum - root, abs=1e-12)
    assert out.obs("excited_state_energy") == pytest.approx(half_sum + root, abs=1e-12)


def test_symmetric_surface_is_symmetric_and_the_crossing_is_lowered_by_the_coupling():
    out = solve(offset=0.0)
    e = out.obs("scan_energy")
    assert e == pytest.approx(e[::-1], abs=1e-12)
    mid = len(e) // 2
    assert out.obs("scan_coordinate")[mid] == pytest.approx(0.0, abs=1e-12)
    assert e[mid] == pytest.approx(morse(1.25) - 0.6, abs=1e-12)           # V_A = V_B there, so E0 = V - Delta
    assert out.obs("reactant_weight")[mid] == pytest.approx(0.5, abs=1e-12)
    assert out.obs("electronic_gap")[mid] == pytest.approx(2 * 0.6, abs=1e-12)
    assert out.obs("electronic_gap_min") == pytest.approx(1.2, abs=1e-12)  # the gap is smallest at the crossing


def test_zero_coupling_collapses_to_the_lower_diabatic_envelope():
    out = solve(coupling=0.0)
    assert out.obs("scan_energy") == pytest.approx(np.minimum(out.obs("diabatic_a"), out.obs("diabatic_b")), abs=1e-12)


def test_well_depth_follows_second_order_perturbation_theory():
    # At the donor-bound geometry the other diabat is far above, so E0 ~ V_A - Delta^2 / (V_B - V_A)
    out = solve(offset=0.0)
    x = out.obs("scan_coordinate")
    i = int(np.argmin(np.abs(x - (0.96 - 1.25))))                          # r_A = r_eq
    v_a, v_b = out.obs("diabatic_a")[i], out.obs("diabatic_b")[i]
    assert out.obs("scan_energy")[i] == pytest.approx(v_a - 0.6**2 / (v_b - v_a), rel=0.1)
    assert out.obs("scan_energy")[i] < 0                                    # wells sit below the bare Morse minimum


def test_electronic_character_follows_the_proton():
    out = solve(offset=0.0)
    x, w = out.obs("scan_coordinate"), out.obs("reactant_weight")
    left, right = np.argmin(np.abs(x + 0.31)), np.argmin(np.abs(x - 0.31))
    assert w[left] > 0.9 and w[right] < 0.1 and np.all((w >= 0) & (w <= 1))


def test_barrier_grows_with_donor_acceptor_distance_and_shrinks_with_coupling():
    barrier = lambda **kw: solve(**kw).obs("classical_barrier")
    assert barrier(R=2.4) < barrier(R=2.5) < barrier(R=2.6) < barrier(R=2.7)
    assert barrier(coupling=0.8) < barrier(coupling=0.6) < barrier(coupling=0.4)


def test_asymmetry_comes_out_as_reaction_energy():
    assert 0.03 < solve(offset=0.05).obs("classical_reaction_energy") < 0.06   # ~ offset, reshaped slightly by mixing
    assert solve(offset=0.0).obs("classical_reaction_energy") == pytest.approx(0.0, abs=1e-9)


def test_short_donor_acceptor_distance_has_a_single_well_and_no_barrier():
    out = solve(R=2.0)
    assert out.obs("n_minima") == 1
    assert "classical_barrier" not in out.observables


def test_engine_rejects_bad_input():
    engine, backend = EVBProtonTransferEngine(), ClassicalBackend()
    with pytest.raises(ValidationError, match="non-negative"):
        engine.solve(evb(coupling=-0.1), backend)
    with pytest.raises(ValidationError, match="must exceed"):
        engine.solve(evb(R=1.1), backend)
    with pytest.raises(ValidationError, match="positive"):
        engine.solve(evb(depth=-1.0), backend)
    wrong_unit = evb()
    wrong_unit.parameters["coupling"].unit = "kcal/mol"
    with pytest.raises(ValidationError, match="expected 'eV'"):
        engine.solve(wrong_unit, backend)
    with pytest.raises(ValidationError, match="solves electronic_structure"):
        engine.solve(evb().evolve(kind="something.else"), backend)


def test_the_electronic_solve_goes_through_the_backend_seam():
    class Counting(ClassicalBackend):
        name = "counting-eigh"
        calls = []

        def symmetric_eigh(self, matrices):
            Counting.calls.append(matrices.shape)
            return super().symmetric_eigh(matrices)

    EVBProtonTransferEngine().solve(evb(), Counting())
    assert Counting.calls == [(241, 2, 2)]


# -- well finding ----------------------------------------------------------------------------------------
def test_locate_wells_ignores_noise_and_finds_the_true_pair():
    x = np.linspace(-1, 1, 801)
    v = double_well(x, 0.5, 0.4, 0.05) + np.random.default_rng(0).normal(0, 5e-4, x.size)
    assert len(prominent_minima(v)) == 2                                    # 0.5 meV noise is not a well
    i_left, i_top, i_right = locate_wells(v)
    assert x[i_left] == pytest.approx(-0.4, abs=0.02) and x[i_right] == pytest.approx(0.4, abs=0.02)
    assert abs(x[i_top]) < 0.05
    assert locate_wells(0.5 * x**2) is None


# -- the tabulated quantum engine -------------------------------------------------------------------------
def tabulated(x, v, mass=1.007276):
    params = {"mass": Quantity(mass, "amu"), "pes_x": Quantity(x, "angstrom"), "pes_energy": Quantity(v, "eV")}
    return ScientificSystem("table", Scale.QUANTUM, "quantum.tabulated_1d", parameters=params)


def test_tabulated_engine_reproduces_the_analytic_engine():
    x = np.linspace(-0.8, 0.8, 1500)                      # the grid the analytic engine uses for a = 0.4
    tab = TabulatedPotentialEngine().solve(tabulated(x, double_well(x, 0.5, 0.4, 0.05)), ClassicalBackend())
    ref = DoubleWellEngine().solve(quartic_system(), ClassicalBackend())
    for name in ("energy_levels", "levels_left", "levels_right", "transmission_left"):
        assert tab.obs(name) == pytest.approx(ref.obs(name), rel=1e-6)


def test_tabulated_engine_validates_its_table():
    engine, backend = TabulatedPotentialEngine(), ClassicalBackend()
    x = np.linspace(-0.8, 0.8, 400)
    v = double_well(x, 0.5, 0.4)
    with pytest.raises(ValidationError, match="strictly increasing"):
        engine.solve(tabulated(x[::-1], v), backend)
    with pytest.raises(ValidationError, match="equal-length"):
        engine.solve(tabulated(x, v[:-1]), backend)
    with pytest.raises(ValidationError, match="fewer than two wells"):
        engine.solve(tabulated(x, 0.5 * x**2), backend)


# -- electronic -> quantum translation -----------------------------------------------------------------------
def test_translation_hands_over_the_computed_surface_and_context(pipeline):
    r = pipeline.run(evb(), ["quantum"])
    scan, handed, solved = r.trace
    assert handed.kind == "quantum.tabulated_1d" and solved.scale == Scale.QUANTUM
    assert np.array_equal(handed.param("pes_x"), scan.obs("scan_coordinate"))
    assert np.array_equal(handed.param("pes_energy"), scan.obs("scan_energy"))
    assert handed.param("mass", "amu") == pytest.approx(1.007276) and handed.param("temperature", "K") == 300.0
    # the barrier the nuclear solver sees is the barrier the electronic solver computed
    assert solved.obs("barrier_top") - solved.obs("well_min_left") == pytest.approx(scan.obs("classical_barrier"), rel=2e-3)


def test_translator_requires_a_solved_system_and_a_particle_mass(pipeline):
    assert "not been solved" in ElectronicToQuantum().validate(evb())[0].message
    no_mass = evb()
    del no_mass.parameters["particle_mass"]
    with pytest.raises(ValidationError, match="particle_mass"):
        pipeline.run(no_mass, ["quantum"])


def test_single_well_surface_blocks_translation(pipeline):
    with pytest.raises(ValidationError, match="1 well"):
        pipeline.run(evb(R=2.0), ["quantum"])


def test_translator_warns_about_small_gaps_and_short_tables(pipeline):
    assert any("electronic gap" in w for w in pipeline.run(evb(coupling=0.1), ["quantum"]).warnings())
    assert any("scan edges" in w for w in pipeline.run(evb(scan_min_bond_length=0.8), ["quantum"]).warnings())
    assert not pipeline.run(evb(), ["quantum"]).warnings()


# -- the whole chain -------------------------------------------------------------------------------------------------
def test_five_stage_chain_with_unbroken_provenance(pipeline):
    r = pipeline.run(evb(), ["electronic_structure", "quantum", "reaction"])
    assert [s.scale for s in r.trace] == [
        Scale.ELECTRONIC_STRUCTURE, Scale.QUANTUM, Scale.QUANTUM, Scale.REACTION, Scale.REACTION,
    ]
    previous = r.root
    for system in r.trace:
        record = system.provenance[-1]
        assert record.input_fingerprint == previous.fingerprint() and record.output_fingerprint == system.fingerprint()
        previous = system
    assert len(r.final.provenance) == 5
    # naming only the destination is enough: the planner finds both translators
    assert [s.kind for s in pipeline.plan(evb(), ["reaction"])] == ["solve", "translate", "solve", "translate", "solve"]


def test_isotopes_share_one_electronic_surface_but_not_one_rate(pipeline):
    h = pipeline.run(evb(), ["reaction"])
    d = pipeline.run(evb(mass=DEUTERON), ["reaction"])
    assert np.array_equal(h.trace[1].param("pes_energy"), d.trace[1].param("pes_energy"))   # Born-Oppenheimer
    assert h.final.param("k_f") > 50 * d.final.param("k_f")


def test_contracting_the_heavy_atoms_speeds_the_reaction_until_the_reactant_state_vanishes(pipeline):
    k = {R: pipeline.run(evb(R=R), ["reaction"]).final.param("k_f") for R in (2.4, 2.5, 2.6, 2.7)}
    assert k[2.4] > k[2.5] > k[2.6] > k[2.7]
    assert k[2.4] / k[2.7] > 1e4                         # exponential sensitivity to a 0.3 angstrom change
    with pytest.raises(ValidationError, match="zero-point"):
        pipeline.run(evb(R=2.3), ["reaction"])           # barrier below the zero-point energy


def test_uncertainty_propagates_from_electronic_inputs_to_the_rate(pipeline):
    r = pipeline.run(evb(sigmas={"donor_acceptor_distance": 0.01, "coupling": 0.03}), ["reaction"], n_samples=30, seed=2)
    assert r.ensemble["varied"] == ["coupling", "donor_acceptor_distance"] and r.ensemble["n_ok"] == 30
    assert r.trace[0].observables["classical_barrier"].sigma > 0
    assert r.trace[1].parameters["pes_energy"].sigma.shape == (241,)       # an uncertainty band on the surface
    assert r.trace[1].parameters["mass"].sigma is None                     # unvaried inputs gain no spurious sigma
    k_f = r.trace[3].parameters["k_f"]
    assert 0 < k_f.sigma < 2 * k_f.value


def test_unreachable_scales_are_reported_cleanly(pipeline):
    with pytest.raises(NoPathError, match="reach"):
        pipeline.plan(evb(), ["molecular"])                # the 1D electronic kind has no 2D surface to hand to it


def test_electronic_example_experiment_runs_from_the_cli(capsys):
    from conftest import EXPERIMENTS
    from substrate.cli import main
    assert main(["run", str(EXPERIMENTS / "proton_transfer_electronic.yaml"), "--samples", "6"]) == 0
    out = capsys.readouterr().out
    assert "electronic_structure" in out and "classical_barrier" in out and "k_f" in out and "n_minima" in out


# -- the reference distance only places the coupling ------------------------------------------------------------------------------------
BACKEND = ClassicalBackend()


def test_the_reference_distance_may_lie_outside_the_scan_range_because_it_only_places_the_coupling():
    """Delta(R) = c exp(-k (R - R_ref)): a different R_ref with c rescaled by exp(k (R_ref' - R_ref)) is the same coupling at every R, so the
    surface and the slice are identical. R_ref = 2.1 A is outside the 2.3-3.0 A grid; the scan distance, which does cut the grid, must not be."""
    grid = dict(distance_min=(2.3, "angstrom"), distance_max=(3.0, "angstrom"), scan_distance=(2.8, "angstrom"), n_x=(41, "1"), n_r=(21, "1"))
    inside = EVBFlexibleProtonTransferEngine().solve(evb2d(reference_distance=2.5, coupling=0.6, **grid), BACKEND)
    outside = EVBFlexibleProtonTransferEngine().solve(
        evb2d(reference_distance=2.1, coupling=0.6 * math.exp(3.0 * (2.5 - 2.1)), **grid), BACKEND)
    assert outside.obs("surface_energy") == pytest.approx(inside.obs("surface_energy"), abs=1e-9)
    assert outside.obs("scan_energy") == pytest.approx(inside.obs("scan_energy"), abs=1e-9)
    assert outside.obs("classical_barrier") == pytest.approx(inside.obs("classical_barrier"), abs=1e-9)
    with pytest.raises(ValidationError, match="scan_distance inside them"):
        EVBFlexibleProtonTransferEngine().solve(evb2d(reference_distance=2.5, **{**grid, "scan_distance": (3.4, "angstrom")}), BACKEND)
    with pytest.raises(ValidationError, match="scan_distance inside them"):                      # no scan distance given: it defaults to the reference distance
        EVBFlexibleProtonTransferEngine().solve(evb2d(reference_distance=2.1, **{k: v for k, v in grid.items() if k != "scan_distance"}), BACKEND)
