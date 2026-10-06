import numpy as np
import pytest

from substrate import (
    AmbiguousPathError, ClassicalBackend, NoPathError, Quantity, Scale, ScientificSystem, ValidationError,
)
from substrate.engines.electronic import EVBFlexibleProtonTransferEngine, EVBProtonTransferEngine, evb_adiabats, morse
from substrate.engines.molecular import (
    StationaryPointEngine, signed_energy_quanta, vibrational_eigenvalues, vibrational_free_energy,
)
from substrate.engines.quantum import TabulatedPotentialEngine
from substrate.translators.electronic_to_molecular import ElectronicToMolecular
from substrate.translators.molecular_to_reaction import MolecularToReaction
from substrate.units import HBAR2_2AMU, H_EV_S, KB_EV, WAVENUMBER_PER_EV

from conftest import DEUTERON, PROTON, EXPERIMENTS, evb, evb2d

BACKEND = ClassicalBackend()
HEAVY = 15.9949
ROUTE_M = ["electronic_structure", "molecular", "reaction"]
ROUTE_Q = ["electronic_structure", "quantum", "reaction"]


def solve2d(**kw):
    return EVBFlexibleProtonTransferEngine().solve(evb2d(**kw), BACKEND)


def molecular(pipeline, **kw):
    return pipeline.run(evb2d(**kw), ["electronic_structure", "molecular"]).final


# =====================================================================================================
# the 2D electronic engine
# =====================================================================================================
def test_the_2d_surface_reduces_to_the_1d_engine_at_each_fixed_distance():
    # With no distance dependence in the coupling, column R of the surface is the 1D scan at that R plus V_OO(R).
    # This also pins down which axis is which.
    out = solve2d(coupling_decay=0.0)
    x, r, surface = out.obs("surface_x"), out.obs("surface_r"), out.obs("surface_energy")
    assert surface.shape == (len(x), len(r))
    for j in (0, 17, 45, 90):
        one_d = EVBProtonTransferEngine().solve(
            evb(R=r[j], offset=0.05, scan_min_bond_length=r[j] / 2 - 0.65, n_scan=len(x)), BACKEND
        )
        assert one_d.obs("scan_coordinate") == pytest.approx(x, abs=1e-12)
        assert surface[:, j] - morse(r[j], 0.4, 2.5, 2.7) == pytest.approx(one_d.obs("scan_energy"), abs=1e-12)


def test_the_2d_surface_matches_the_closed_form_with_distance_dependent_coupling():
    out = solve2d()
    x, r, surface = out.obs("surface_x"), out.obs("surface_r"), out.obs("surface_energy")
    for i, j in ((10, 5), (80, 40), (120, 75), (150, 88)):
        v_a, v_b = morse(r[j] / 2 + x[i], 4.6, 2.2, 0.96), morse(r[j] / 2 - x[i], 4.6, 2.2, 0.96) + 0.05
        delta = 0.6 * np.exp(-3.0 * (r[j] - 2.5))
        exact = 0.5 * (v_a + v_b) - np.sqrt((0.5 * (v_a - v_b)) ** 2 + delta**2) + morse(r[j], 0.4, 2.5, 2.7)
        assert surface[i, j] == pytest.approx(exact, abs=1e-12)


def test_the_symmetric_2d_surface_is_symmetric_in_the_proton_coordinate():
    s = solve2d(diabatic_offset=0.0).obs("surface_energy")
    assert s == pytest.approx(s[::-1, :], abs=1e-12)


def test_moving_the_scan_distance_cuts_the_same_surface_elsewhere():
    base, moved = solve2d(), solve2d(scan_distance=(2.64, "angstrom"))         # 2.64 is a grid distance: 2.2 + 44 * 0.01
    assert np.array_equal(moved.obs("surface_energy"), base.obs("surface_energy"))             # the surface is untouched
    assert moved.obs("scan_energy") == pytest.approx(base.obs("surface_energy")[:, 44], abs=1e-12)   # slice = a column of it
    assert base.obs("surface_r")[44] == pytest.approx(2.64, abs=1e-12)
    assert moved.obs("classical_barrier") > base.obs("classical_barrier")                      # farther apart: higher barrier
    with pytest.raises(ValidationError, match="scan_distance"):
        EVBFlexibleProtonTransferEngine().solve(evb2d(scan_distance=(9.0, "angstrom")), BACKEND)


def test_gap_observables_distinguish_the_rigid_slice_from_the_relaxed_path():
    out = solve2d(diabatic_offset=0.0)
    assert out.obs("electronic_gap_min") == pytest.approx(2 * 0.6, abs=1e-12)       # slice at R0: gap = 2 * Delta(R0)
    # along the relaxed path the TS sits near R = 2.4, where the coupling is larger, so the gap is larger
    assert out.obs("electronic_gap_min_relaxed") > out.obs("electronic_gap_min")


def test_the_2d_engine_rejects_bad_input():
    engine = EVBFlexibleProtonTransferEngine()
    with pytest.raises(ValidationError, match="non-negative"):
        engine.solve(evb2d(coupling_decay=-1.0), BACKEND)
    with pytest.raises(ValidationError, match="inside them"):
        engine.solve(evb2d(reference_distance=3.5), BACKEND)
    with pytest.raises(ValidationError, match="n_x >= 21"):
        engine.solve(evb2d(n_x=5), BACKEND)
    with pytest.raises(ValidationError, match="positive"):
        engine.solve(evb2d(oo_depth=-0.1), BACKEND)
    wrong = evb2d()
    wrong.parameters["oo_depth"].unit = "kcal/mol"
    with pytest.raises(ValidationError, match="expected 'eV'"):
        engine.solve(wrong, BACKEND)


# =====================================================================================================
# normal modes and thermochemistry, against independent calculations
# =====================================================================================================
def test_vibrational_eigenvalues_match_a_finite_difference_cartesian_hessian():
    # Unequal masses and a cross term in the surface: exercises the coordinate transform, the mass weighting and
    # the removal of translation, none of which the symmetric case can distinguish from a wrong answer.
    a, b, c, r0 = 12.0, 3.0, 2.0, 2.5
    masses = np.array([15.9949, 1.007276, 18.9984])

    def energy(z):
        x, r = z[1] - 0.5 * (z[0] + z[2]), z[2] - z[0]
        return 0.5 * a * x**2 + 0.5 * b * (r - r0) ** 2 + c * x * (r - r0)

    z0, h = np.array([0.0, 1.2, 2.5]), 1e-4
    hess = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            def shifted(si, sj):
                z = z0.copy()
                z[i] += si * h
                z[j] += sj * h
                return energy(z)
            hess[i, j] = (shifted(1, 1) - shifted(1, -1) - shifted(-1, 1) + shifted(-1, -1)) / (4 * h * h)
    weighted = hess / np.sqrt(np.outer(masses, masses))
    reference = np.sort(np.linalg.eigvalsh(weighted))
    assert abs(reference[0]) < 1e-5 and reference[1] > 1e-3                  # exactly one zero mode: translation
    expected = reference[1:]
    got = vibrational_eigenvalues(np.array([[a, c], [c, b]]), *masses[[0, 1, 2]])
    assert got == pytest.approx(expected, rel=1e-5)


def test_frequencies_follow_the_analytic_reduced_mass_when_the_heavy_atom_distance_is_stiff():
    # E(x, R) = E_slice(x) + k (R - R0)^2: x and R decouple, with 1/mu_x = 1/m_H + 1/(2 m) and 1/mu_R = 2/m
    r0, k_r = 2.5, 50.0
    scan = EVBProtonTransferEngine().solve(evb(R=r0, scan_min_bond_length=r0 / 2 - 0.65, n_scan=161), BACKEND)
    x = scan.obs("scan_coordinate")
    r = np.linspace(r0 - 0.3, r0 + 0.3, 41)
    table = scan.obs("scan_energy")[:, None] + k_r * (r[None, :] - r0) ** 2
    system = _molecular(x, r, table)
    out = StationaryPointEngine().solve(system, BACKEND)

    def slice_energy(xx):
        return evb_adiabats(BACKEND, r0 / 2 + xx, r0 / 2 - xx, 0.6, 0.05, 4.6, 2.2, 0.96)[0]

    def curvature(x_at, h=1e-3):
        return (slice_energy(x_at + h) - 2 * slice_energy(x_at) + slice_energy(x_at - h)) / h**2

    inv_mu_x = 1 / PROTON + 1 / (2 * HEAVY)
    to_cm = WAVENUMBER_PER_EV
    f_oh = np.sqrt(2 * HBAR2_2AMU * curvature(out.obs("geometry_reactant")[0]) * inv_mu_x) * to_cm
    f_oo = np.sqrt(2 * HBAR2_2AMU * 2 * k_r * (2 / HEAVY)) * to_cm
    assert out.obs("frequencies_reactant") == pytest.approx(sorted([f_oh, f_oo]), rel=5e-3)
    f_imag = np.sqrt(2 * HBAR2_2AMU * -curvature(out.obs("geometry_ts")[0]) * inv_mu_x) * to_cm
    assert out.obs("imaginary_frequency") == pytest.approx(f_imag, rel=5e-3)
    assert out.obs("frequencies_ts") == pytest.approx([f_oo], rel=5e-3)         # only the stiff mode survives at the TS
    assert out.obs("geometry_ts")[1] == pytest.approx(r0, abs=1e-6)             # and R cannot relax


def test_the_molecular_harmonic_zpe_agrees_with_the_anharmonic_quantum_levels_of_the_same_bond():
    # Two scales describing one vibration: harmonic normal mode vs the exact (anharmonic) nuclear levels.
    r0 = 2.5
    scan = EVBProtonTransferEngine().solve(evb(R=r0, scan_min_bond_length=r0 / 2 - 0.65, n_scan=161), BACKEND)
    x, slice_e = scan.obs("scan_coordinate"), scan.obs("scan_energy")
    r = np.linspace(r0 - 0.3, r0 + 0.3, 41)
    mol = StationaryPointEngine().solve(_molecular(x, r, slice_e[:, None] + 50.0 * (r[None, :] - r0) ** 2), BACKEND)
    mu_x = 1.0 / (1 / PROTON + 1 / (2 * HEAVY))
    quantum = TabulatedPotentialEngine().solve(
        ScientificSystem("q", Scale.QUANTUM, "quantum.tabulated_1d", parameters={
            "mass": Quantity(mu_x, "amu"), "pes_x": Quantity(x, "angstrom"), "pes_energy": Quantity(slice_e, "eV"),
        }), BACKEND)
    harmonic_zpe_x = 0.5 * mol.obs("frequencies_reactant")[-1] / WAVENUMBER_PER_EV
    assert quantum.obs("zpe_left") == pytest.approx(harmonic_zpe_x, rel=0.05)
    assert quantum.obs("zpe_left") < harmonic_zpe_x                             # anharmonicity lowers the true ZPE


def test_vibrational_free_energy_limits_and_closed_form():
    kt = 0.025852
    y = 0.4
    assert vibrational_free_energy(np.array([y]), kt) == pytest.approx(-kt * np.log(np.exp(-y / kt / 2) / (1 - np.exp(-y / kt))))
    assert vibrational_free_energy(np.array([0.3, 0.5]), 1e-4) == pytest.approx(0.5 * (0.3 + 0.5), rel=1e-12)   # T -> 0: ZPE
    assert vibrational_free_energy(np.array([0.01]), 1.0) == pytest.approx(np.log(0.01), abs=2e-3)               # classical limit
    assert signed_energy_quanta(np.array([-1.0, 4.0])) == pytest.approx(
        [-np.sqrt(2 * HBAR2_2AMU), np.sqrt(8 * HBAR2_2AMU)])


def test_rate_is_the_eyring_expression_recomputed_from_the_reported_frequencies(pipeline):
    o = molecular(pipeline).observables
    kt = KB_EV * 300.0

    def free_energy(point):
        quanta = o[f"frequencies_{point}"].value / WAVENUMBER_PER_EV
        return o[f"energy_{point}"].value + float(np.sum(0.5 * quanta + kt * np.log1p(-np.exp(-quanta / kt))))

    assert o["free_energy_reactant"].value == pytest.approx(free_energy("reactant"), abs=1e-12)
    dg = free_energy("ts") - free_energy("reactant")
    assert o["delta_g_forward"].value == pytest.approx(dg, abs=1e-12)
    assert o["rate_tst_forward"].value == pytest.approx(kt / H_EV_S * np.exp(-dg / kt), rel=1e-9)
    # ZPE-corrected barrier: classical barrier plus the change in zero-point energy
    assert o["barrier_zpe"].value == pytest.approx(
        o["barrier_classical"].value + o["zpe_ts"].value - o["zpe_reactant"].value, abs=1e-12)


def test_detailed_balance_holds_between_the_two_tst_rates(pipeline):
    o = molecular(pipeline).observables
    kt = KB_EV * 300.0
    ratio = o["rate_tst_forward"].value / o["rate_tst_reverse"].value
    assert ratio == pytest.approx(np.exp(-(o["free_energy_product"].value - o["free_energy_reactant"].value) / kt), rel=1e-9)


def test_stationary_points_have_the_right_character(pipeline):
    o = molecular(pipeline).observables
    assert len(o["frequencies_reactant"].value) == 2 and len(o["frequencies_product"].value) == 2
    assert len(o["frequencies_ts"].value) == 1 and o["imaginary_frequency"].value > 0     # first-order saddle point
    assert o["energy_ts"].value > max(o["energy_reactant"].value, o["energy_product"].value)
    assert o["geometry_reactant"].value[0] < o["geometry_ts"].value[0] < o["geometry_product"].value[0]
    assert 3000 < o["frequencies_reactant"].value[-1] < 4000                               # an O-H stretch
    assert o["frequencies_reactant"].value[0] < 800                                        # the O...O stretch


# =====================================================================================================
# the physics the molecular scale adds
# =====================================================================================================
def test_the_heavy_atoms_relax_and_that_lowers_the_barrier(pipeline):
    mol = molecular(pipeline)
    r_well, r_ts = mol.obs("geometry_reactant")[1], mol.obs("geometry_ts")[1]
    assert r_ts < r_well - 0.15                                                 # O...O compresses at the transition state
    rigid = solve2d(scan_distance=(float(r_well), "angstrom"))                                # the SAME surface, heavy atoms frozen at the well
    assert mol.obs("barrier_classical") < rigid.obs("classical_barrier") - 0.2


def test_isotope_effect_from_zero_point_energy_alone_is_semiclassical_in_size(pipeline):
    h = molecular(pipeline)
    d = molecular(pipeline, mass=DEUTERON)
    kie = h.obs("rate_tst_forward") / d.obs("rate_tst_forward")
    assert 4 < kie < 15                                                         # the textbook semiclassical range for O-H / O-D
    assert d.obs("barrier_zpe") > h.obs("barrier_zpe")
    assert d.obs("imaginary_frequency") < h.obs("imaginary_frequency")
    assert d.obs("energy_ts") == pytest.approx(h.obs("energy_ts"), abs=1e-9)    # Born-Oppenheimer: one surface for both


def test_at_the_default_frozen_distance_tunnelling_inflates_the_isotope_effect_beyond_zero_point(pipeline):
    tst = pipeline.run(evb2d(), ROUTE_M).final.param("k_f") / pipeline.run(evb2d(mass=DEUTERON), ROUTE_M).final.param("k_f")
    quantum = pipeline.run(evb2d(), ROUTE_Q).final.param("k_f") / pipeline.run(evb2d(mass=DEUTERON), ROUTE_Q).final.param("k_f")
    assert quantum > 5 * tst


def test_tst_rate_rises_with_temperature(pipeline):
    k = [molecular(pipeline, temperature=t).obs("rate_tst_forward") for t in (250, 300, 400, 600)]
    assert all(b > a for a, b in zip(k, k[1:]))


def test_the_barrier_disappears_in_steps_as_the_coupling_grows(pipeline):
    barrier = lambda c: molecular(pipeline, coupling=c)
    assert barrier(0.7).obs("barrier_classical") < barrier(0.6).obs("barrier_classical")
    # zero-point energy removes the barrier before the double well itself disappears ...
    with pytest.raises(ValidationError, match="free-energy barrier is not positive"):
        pipeline.run(evb2d(coupling=1.0), ROUTE_M)
    # ... and eventually the surface has no second well at all
    with pytest.raises(ValidationError, match="fewer than two wells"):
        pipeline.run(evb2d(coupling=1.2), ROUTE_M)


# =====================================================================================================
# engine validation
# =====================================================================================================
def _molecular(x, r, energy, mass_h=PROTON, heavy=HEAVY, temperature=300.0):
    return ScientificSystem("mol", Scale.MOLECULAR, "molecular.collinear_triatomic", parameters={
        "mass_donor": Quantity(heavy, "amu"), "mass_acceptor": Quantity(heavy, "amu"),
        "mass_hydrogen": Quantity(mass_h, "amu"), "temperature": Quantity(temperature, "K"),
        "surface_x": Quantity(x, "angstrom"), "surface_r": Quantity(r, "angstrom"),
        "surface_energy": Quantity(energy, "eV"),
    })


def test_molecular_engine_validates_its_inputs():
    engine = StationaryPointEngine()
    x, r = np.linspace(-0.6, 0.6, 61), np.linspace(2.2, 3.0, 41)
    single_well = 3.0 * x[:, None] ** 2 + 4.0 * (r[None, :] - 2.6) ** 2
    with pytest.raises(ValidationError, match="fewer than two wells"):
        engine.solve(_molecular(x, r, single_well), BACKEND)
    with pytest.raises(ValidationError, match="strictly increasing"):
        engine.solve(_molecular(x[::-1], r, single_well), BACKEND)
    with pytest.raises(ValidationError, match="surface_energy must be a"):
        engine.solve(_molecular(x, r, single_well[:-1]), BACKEND)
    with pytest.raises(ValidationError, match="positive"):
        engine.solve(_molecular(x, r, single_well, temperature=-1.0), BACKEND)
    with pytest.raises(ValidationError, match="solves molecular"):
        engine.solve(evb2d(), BACKEND)


# =====================================================================================================
# translators, routing and the whole chain
# =====================================================================================================
def test_the_planner_will_not_choose_between_tst_and_quantum_tunnelling(pipeline):
    with pytest.raises(AmbiguousPathError, match="molecular") as exc:
        pipeline.plan(evb2d(), ["reaction"])
    assert "quantum" in str(exc.value)
    assert [s.scale_out for s in pipeline.plan(evb2d(), ROUTE_M)][-1] == Scale.REACTION
    assert pipeline.plan(evb2d(), ROUTE_Q)[2].component.name == "quantum.tabulated_1d"


def test_kinds_keep_the_older_routes_unambiguous(pipeline):
    # the 1D electronic kind has no 2D surface, so it can only go through the quantum scale
    assert pipeline.plan(evb(), ["reaction"])[2].component.name == "quantum.tabulated_1d"
    with pytest.raises(NoPathError, match="you can reach") as exc:
        pipeline.plan(evb(), ["molecular"])
    assert "molecular" not in str(exc.value).split("you can reach:")[1]            # it genuinely cannot get there
    assert "quantum" in str(exc.value) and "reaction" in str(exc.value)


def test_both_routes_run_from_one_electronic_system_and_give_different_physics(pipeline):
    m = pipeline.run(evb2d(), ROUTE_M)
    q = pipeline.run(evb2d(), ROUTE_Q)
    assert [s.scale for s in m.trace] == [
        Scale.ELECTRONIC_STRUCTURE, Scale.MOLECULAR, Scale.MOLECULAR, Scale.REACTION, Scale.REACTION]
    assert m.trace[3].structure["derivation"]["route"].startswith("classical transition-state theory")
    assert "dominant_level" in q.trace[3].structure["derivation"]
    assert m.final.param("k_f") != pytest.approx(q.final.param("k_f"), rel=0.5)     # different approximations, not a check
    for result in (m, q):
        previous = result.root
        for system in result.trace:
            record = system.provenance[-1]
            assert record.input_fingerprint == previous.fingerprint() and record.output_fingerprint == system.fingerprint()
            previous = system


def test_reaction_network_carries_the_tst_rates_and_equilibrium(pipeline):
    r = pipeline.run(evb2d(), ROUTE_M)
    mol = r.trace[2]
    net = r.final
    assert net.param("k_f") == pytest.approx(mol.obs("rate_tst_forward")) and net.param("k_r") == pytest.approx(mol.obs("rate_tst_reverse"))
    kt = KB_EV * 300.0
    eq = net.param("K_eq")
    assert eq == pytest.approx(np.exp(-(mol.obs("free_energy_product") - mol.obs("free_energy_reactant")) / kt), rel=1e-9)
    assert net.obs("final_concentrations")[1] == pytest.approx(eq / (1 + eq), rel=2e-4)


def test_electronic_to_molecular_requires_a_solved_system_and_its_context(pipeline):
    assert "not been solved" in ElectronicToMolecular().validate(evb2d())[0].message
    missing = evb2d()
    del missing.parameters["heavy_atom_mass"]
    with pytest.raises(ValidationError, match="heavy_atom_mass"):
        pipeline.run(missing, ROUTE_M)


def test_electronic_to_molecular_warns_when_the_relaxed_path_has_a_small_gap(pipeline):
    r = pipeline.run(evb2d(coupling=0.15), ["electronic_structure", "molecular"])
    assert any("along the relaxed path" in w for w in r.warnings())
    assert not any("relaxed path" in w for w in pipeline.run(evb2d(), ["electronic_structure", "molecular"]).warnings())


def test_molecular_to_reaction_says_when_classical_tst_is_inadequate(pipeline):
    cold = pipeline.run(evb2d(), ROUTE_M).warnings()
    assert any("tunnelling is significant" in w for w in cold)                 # hbar*omega/kT ~ 12 at 300 K
    hot = pipeline.run(evb2d(temperature=2000.0), ROUTE_M).warnings()
    assert not any("tunnelling is significant" in w for w in hot)              # ~1.9 at 2000 K
    assert any("under 5 kT" in w for w in hot)                                  # but the barrier is now only a few kT
    unsolved = MolecularToReaction().validate(_molecular(np.zeros(3), np.zeros(3), np.zeros((3, 3))))
    assert "not been solved" in unsolved[0].message


def test_uncertainty_propagates_through_the_molecular_route(pipeline):
    r = pipeline.run(evb2d(sigmas={"coupling": 0.02, "oo_equilibrium": 0.02}), ROUTE_M, n_samples=24, seed=4)
    assert r.ensemble["n_ok"] == 24 and r.ensemble["varied"] == ["coupling", "oo_equilibrium"]
    assert r.trace[2].observables["barrier_classical"].sigma > 0
    k_f = r.final.parameters["k_f"]
    # the rate is exponentially sensitive to the O...O distance: shortening it by 0.04 angstrom lowers the barrier
    # by ~0.2 eV, so the spread is orders of magnitude and a symmetric "+/-" would be meaningless. The band is not.
    assert k_f.sigma > k_f.value
    lo, hi = k_f.band
    assert 0 < lo < hi and hi / lo > 100
    assert r.trace[1].parameters["mass_hydrogen"].sigma is None               # unvaried context gains no spurious sigma


def test_molecular_example_experiment_runs_from_the_cli(capsys):
    from substrate.cli import main
    assert main(["run", str(EXPERIMENTS / "proton_transfer_molecular.yaml"), "--samples", "4"]) == 0
    out = capsys.readouterr().out
    assert "molecular.stationary_points_tst" in out and "imaginary_frequency" in out and "rate_tst_forward" in out
