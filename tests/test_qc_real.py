"""The quantum-chemistry engine against real PySCF (through WSL on this machine); skipped where PySCF is unavailable.

These test the chemistry, using constraints that do not depend on this codebase agreeing with itself: the variational
principle, the Hartree-Fock limit, invariance under rigid motion, and the lowering of the energy by electron correlation.
Single-point energies are cached in a persistent directory, so only the first run pays for the calculations (~1 minute).
"""
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from substrate import QCError, ValidationError
from substrate.engines.qc_scan import QCScanEngine, build_geometry
from substrate.molecules import zundel_cation
from substrate.pes import prominent_minima
from substrate.qc import PySCFProgram, QCCache, QCJob, compute_cached

from conftest import DEUTERON, EXPERIMENTS, PROTON, zundel_system

program = PySCFProgram()
pytestmark = [
    pytest.mark.qc,
    pytest.mark.skipif(not program.available(), reason="PySCF is not installed (see scripts/setup_qc_env.sh)"),
]
HARTREE_EV = 27.211386245988


@pytest.fixture(scope="module", autouse=True)
def persistent_cache():
    path = Path(tempfile.gettempdir()) / "substrate-test-qc-cache"
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SUBSTRATE_CACHE_DIR", str(path))
        yield path


def energy(atoms, charge=0, spin=0, theory="hf", basis="cc-pvdz"):
    job = QCJob(tuple((s, float(x), float(y), float(z)) for s, x, y, z in atoms), charge, spin, theory, basis)
    return compute_cached(program, [job], QCCache())[0][0]


WATER = [("O", 0.0, 0.0, 0.1173), ("H", 0.0, 0.7572, -0.4692), ("H", 0.0, -0.7572, -0.4692)]
HELIUM = [("He", 0.0, 0.0, 0.0)]
H2 = [("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.7414)]


# -- the program, against physics ---------------------------------------------------------------------------------------
def test_energies_obey_the_variational_principle_and_the_hartree_fock_limit():
    h2 = energy([("H", 0, 0, 0), ("H", 0, 0, 0.7414)], basis="cc-pvtz").energy_hartree
    assert -1.133629 < h2 < -1.1325                      # a finite basis cannot beat the HF limit (-1.133629) and is close to it
    h_atom = energy([("H", 0, 0, 0)], spin=1, basis="cc-pvtz")                   # open shell: unrestricted HF
    assert -0.5 < h_atom.energy_hartree < -0.4995        # variational: never below the exact -0.5, and very close to it
    assert h_atom.converged and h_atom.homo_lumo_gap_hartree is not None


def test_the_energy_is_invariant_under_rigid_motion():
    c, s = math.cos(0.7), math.sin(0.7)
    moved = [(sym, c * x - s * y + 3.1, s * x + c * y - 2.2, z + 5.0) for sym, x, y, z in WATER]
    assert energy(moved).energy_hartree == pytest.approx(energy(WATER).energy_hartree, abs=1e-7)


def test_electron_correlation_lowers_the_energy_and_other_methods_differ():
    hf = energy(WATER).energy_hartree
    mp2 = energy(WATER, theory="mp2").energy_hartree
    pbe = energy(WATER, theory="dft:pbe")
    assert pbe.converged and abs(pbe.energy_hartree - hf) > 0.05                 # a different method is a different energy
    assert 0.1 < hf - mp2 < 0.3                                                  # ~0.2 hartree of correlation for water


def test_an_unknown_theory_is_an_error_not_a_silent_default():
    with pytest.raises(QCError, match="unknown theory"):
        energy(WATER, theory="magic")


# -- coupled cluster: CCSD and CCSD(T) ---------------------------------------------------------------------------------------------------
def test_coupled_cluster_is_exact_for_two_electrons_so_the_triples_vanish_and_the_exact_energy_is_a_bound():
    """CCSD is full CI for two electrons, so there are no triple excitations: CCSD(T) equals CCSD, and the energy is variational against
    the exact non-relativistic energy (He -2.903724, H2 -1.174476 hartree) while recovering most of the correlation beyond the HF limit."""
    for atoms, exact, hf_limit in ((HELIUM, -2.903724, -2.861680), (H2, -1.174476, -1.133629)):
        ccsd, triples = energy(atoms, theory="ccsd", basis="cc-pvtz"), energy(atoms, theory="ccsd(t)", basis="cc-pvtz")
        assert ccsd.converged and triples.converged
        assert triples.energy_hartree == pytest.approx(ccsd.energy_hartree, abs=1e-9)
        assert exact < triples.energy_hartree < hf_limit - 0.03


def test_the_triples_correction_is_negative_and_of_the_expected_size_for_water():
    hf, mp2 = energy(WATER).energy_hartree, energy(WATER, theory="mp2").energy_hartree
    ccsd, triples = energy(WATER, theory="ccsd"), energy(WATER, theory="ccsd(t)")
    assert ccsd.converged and triples.converged
    assert -0.006 < triples.energy_hartree - ccsd.energy_hartree < -0.0015          # about -0.003 hartree: a small, negative correction
    assert triples.energy_hartree < ccsd.energy_hartree < hf
    assert abs(triples.energy_hartree - mp2) < 0.03                               # MP2 and CCSD(T) give correlation energies within ~10% of each other


def test_coupled_cluster_refuses_an_open_shell_system_instead_of_approximating_it():
    with pytest.raises(QCError, match="closed-shell"):
        energy([("H", 0, 0, 0)], spin=1, theory="ccsd(t)", basis="cc-pvtz")


def test_recorded_bifluoride_coupled_cluster_energies_guard_against_a_change_in_the_program():
    """Recorded with PySCF 2.14.0, not an independent reference: all-electron CCSD and CCSD(T)/6-31+G* for FHF- at F...F 2.7 A, the proton at the centre."""
    fhf = [("F", 0, 0, -1.35), ("F", 0, 0, 1.35), ("H", 0, 0, 0.0)]
    assert energy(fhf, charge=-1, theory="ccsd", basis="6-31+g*").energy_hartree == pytest.approx(-199.87099744, abs=1e-6)
    assert energy(fhf, charge=-1, theory="ccsd(t)", basis="6-31+g*").energy_hartree == pytest.approx(-199.87929693, abs=1e-6)


# -- geometry relaxation ---------------------------------------------------------------------------------------------------------------------
def relax(atoms, charge=0, theory="hf", basis="6-31g*"):
    job = QCJob(tuple((sym, float(x), float(y), float(z)) for sym, x, y, z in atoms), charge, 0, theory, basis, "relax")
    return compute_cached(program, [job], QCCache())[0][0]


def _bond_and_angle(geometry):
    o, h1, h2 = (np.array(a[1:]) for a in geometry)
    v1, v2 = h1 - o, h2 - o
    return np.linalg.norm(v1), math.degrees(math.acos(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2))))


DISTORTED_WATER = [("O", 0.0, 0.0, 0.1173), ("H", 0.0, 0.92, -0.40), ("H", 0.0, -0.70, -0.52)]            # stretched on one side, bent on the other


def _water(r, theta_degrees):
    half = math.radians(theta_degrees) / 2.0
    return [("O", 0.0, 0.0, 0.0), ("H", 0.0, r * math.sin(half), -r * math.cos(half)), ("H", 0.0, -r * math.sin(half), -r * math.cos(half))]


def test_a_distorted_water_relaxes_to_the_known_hartree_fock_minimum():
    """HF/6-31G* water: O-H 0.947 angstrom, H-O-H 105.5 degrees. The check does not use a quoted total energy (those depend on whether the
    d functions are Cartesian or spherical, which differ between programs): the relaxed energy must equal the single-point energy at the known
    minimum, computed by the same program, and lie below it at geometries displaced from there in every direction."""
    result = relax(DISTORTED_WATER)
    assert result.converged and result.initial_energy_hartree > result.energy_hartree + 0.005          # it started well above the minimum
    at_minimum = energy(_water(0.947, 105.5), basis="6-31g*").energy_hartree
    assert result.energy_hartree == pytest.approx(at_minimum, abs=3e-5) and result.energy_hartree <= at_minimum + 1e-6
    for r, theta in ((0.967, 105.5), (0.927, 105.5), (0.947, 108.5), (0.947, 102.5)):
        assert energy(_water(r, theta), basis="6-31g*").energy_hartree > result.energy_hartree + 1e-4   # a minimum along the bond length and the angle
    bond, angle = _bond_and_angle(result.geometry)
    assert bond == pytest.approx(0.947, abs=0.004) and angle == pytest.approx(105.5, abs=0.6)
    assert [a[0] for a in result.geometry] == ["O", "H", "H"]                                           # the atoms come back in the order they went in


def test_relaxation_does_not_care_where_the_molecule_is_or_how_it_is_turned_and_a_minimum_stays_put():
    c, s = math.cos(0.9), math.sin(0.9)
    moved = [(sym, c * x - s * y + 2.0, s * x + c * y - 1.0, z + 3.0) for sym, x, y, z in DISTORTED_WATER]
    assert relax(moved).energy_hartree == pytest.approx(relax(DISTORTED_WATER).energy_hartree, abs=3e-6)
    minimum = relax(DISTORTED_WATER)
    again = relax(minimum.geometry)
    assert again.energy_hartree == pytest.approx(minimum.energy_hartree, abs=2e-6)
    assert again.initial_energy_hartree - again.energy_hartree < 1e-5                                   # nothing left to gain
    assert max(np.linalg.norm(np.array(a[1:]) - np.array(b[1:])) for a, b in zip(minimum.geometry, again.geometry)) < 0.01


def test_correlated_and_density_functional_relaxations_find_their_own_longer_bond_and_a_lower_energy_than_the_start():
    hf = _bond_and_angle(relax(DISTORTED_WATER).geometry)[0]
    for theory in ("mp2", "dft:b3lyp"):
        result = relax(DISTORTED_WATER, theory=theory)
        bond, angle = _bond_and_angle(result.geometry)
        assert result.converged and result.energy_hartree < result.initial_energy_hartree - 0.005, theory
        assert bond > hf + 0.005 and 0.95 < bond < 0.975 and 103.0 < angle < 106.5, theory             # correlation lengthens the bond, as it does in the literature


def test_a_relaxation_that_cannot_be_done_is_refused_instead_of_approximated():
    with pytest.raises(QCError, match="closed-shell Hartree-Fock, MP2 and DFT only"):
        relax(DISTORTED_WATER, theory="ccsd(t)")
    with pytest.raises(QCError, match="closed-shell"):
        compute_cached(program, [QCJob((("H", 0.0, 0.0, 0.0),), 0, 1, "hf", "6-31g", "relax")], QCCache())


# -- harmonic thermochemistry ------------------------------------------------------------------------------------------------------------------
KT_HARTREE = 3.166811563455608e-6 * 298.15                    # Boltzmann's constant in hartree per kelvin, times the temperature
HYDROGEN_MASS, FLUORINE_MASS, ELECTRON_MASSES_PER_AMU = 1.00782503223, 18.99840316273, 1822.888486209


def thermo(atoms, charge=0, theory="hf", basis="6-31g*"):
    job = QCJob(tuple((sym, float(x), float(y), float(z)) for sym, x, y, z in atoms), charge, 0, theory, basis, "thermo")
    return compute_cached(program, [job], QCCache())[0][0]


def test_an_atom_has_only_translation_so_its_enthalpy_is_five_halves_kt_above_its_energy():
    result = thermo(HELIUM)
    assert result.zero_point_hartree == 0.0
    assert result.enthalpy_298_hartree - result.energy_hartree == pytest.approx(2.5 * KT_HARTREE, abs=1e-8)       # 3/2 kT of translation + pV = kT


def test_the_zero_point_energy_of_a_diatomic_is_half_the_frequency_of_its_own_energy_curve():
    """An independent route: the force constant from a five-point finite difference of single-point energies along the bond, then
    omega = sqrt(k / mu). The linear molecule's rotation and translation then add 7/2 kT (3/2 + 1 + 1), and the vibrational part is negligible."""
    result = thermo([("H", 0.0, 0.0, 0.0), ("F", 0.0, 0.0, 1.05)])                                                    # started well off the minimum (0.912 angstrom)
    assert result.converged
    bond = float(np.linalg.norm(np.array(result.geometry[1][1:]) - np.array(result.geometry[0][1:])))
    step = 0.02                                                                                                         # angstrom

    def curve(offset):
        return energy([("H", 0.0, 0.0, 0.0), ("F", 0.0, 0.0, bond + offset)], basis="6-31g*").energy_hartree

    bohr = 0.529177210903
    k = (-curve(2 * step) + 16 * curve(step) - 30 * curve(0.0) + 16 * curve(-step) - curve(-2 * step)) / (12 * (step / bohr) ** 2)
    mu = HYDROGEN_MASS * FLUORINE_MASS / (HYDROGEN_MASS + FLUORINE_MASS) * ELECTRON_MASSES_PER_AMU
    omega = math.sqrt(k / mu)                                                                                           # hartree
    assert 0.018 < omega < 0.021                                                                                        # 3,950-4,600 cm-1: the right order for H-F
    assert result.zero_point_hartree == pytest.approx(omega / 2, rel=0.01)
    assert result.enthalpy_298_hartree - result.energy_hartree - result.zero_point_hartree == pytest.approx(3.5 * KT_HARTREE, abs=2e-6)


def test_thermochemistry_follows_the_method_and_a_density_functional_works_too():
    hf, b3lyp = thermo([("H", 0.0, 0.0, 0.0), ("F", 0.0, 0.0, 0.92)]), thermo([("H", 0.0, 0.0, 0.0), ("F", 0.0, 0.0, 0.92)], theory="dft:b3lyp")
    assert b3lyp.converged and b3lyp.zero_point_hartree < hf.zero_point_hartree          # B3LYP gives a softer H-F bond than Hartree-Fock
    assert b3lyp.energy_hartree < hf.energy_hartree


def test_a_saddle_point_has_no_thermochemistry_and_is_reported_as_not_converged():
    linear_water = [("O", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.96), ("H", 0.0, 0.0, -0.96)]       # the bending mode of a linear H-O-H is imaginary
    result = thermo(linear_water)
    assert not result.converged and result.enthalpy_298_hartree is None and result.zero_point_hartree is None


def test_thermochemistry_is_refused_where_there_is_no_analytic_hessian():
    for theory in ("mp2", "ccsd(t)"):
        with pytest.raises(QCError, match="closed-shell Hartree-Fock and DFT only"):
            thermo(HELIUM, theory=theory)
    with pytest.raises(QCError, match="closed-shell"):
        compute_cached(program, [QCJob((("H", 0.0, 0.0, 0.0),), 0, 1, "hf", "6-31g", "thermo")], QCCache())


# -- the Zundel cation: a real proton-transfer surface ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def surface():
    return QCScanEngine().solve(zundel_system("pyscf", scan_distance=2.8), None)


def test_the_real_surface_is_symmetric_and_the_gap_is_an_orbital_gap(surface):
    atoms = lambda x: build_geometry(zundel_cation(), x, 2.8)
    assert energy(atoms(0.4), charge=1, basis="6-31g*").energy_hartree == pytest.approx(
        energy(atoms(-0.4), charge=1, basis="6-31g*").energy_hartree, abs=1e-8)            # the exchange symmetry (a two-fold axis) of the staggered ion
    e = surface.obs("surface_energy")
    assert e.min() == pytest.approx(0.0, abs=1e-12) and np.all(e >= 0)
    assert surface.obs("classical_reaction_energy") == pytest.approx(0.0, abs=1e-6)       # symmetric: no reaction energy
    assert surface.obs("electronic_gap_min") > 5.0                                         # HF HOMO-LUMO gap, in eV


def test_the_proton_is_shared_at_short_distance_and_localised_at_long_distance(surface):
    x, r, e = surface.obs("surface_x"), surface.obs("surface_r"), surface.obs("surface_energy")
    short, long_ = e[:, 0], e[:, -1]                                                      # R = 2.3 and 3.0 angstrom
    assert abs(x[np.argmin(short)]) < 0.15 and len(prominent_minima(short)) == 1          # one well, proton in the middle
    assert len(prominent_minima(long_)) == 2                                              # two wells: the proton sits on one oxygen
    assert abs(x[np.argmin(long_)]) > 0.4


def test_the_barrier_grows_with_oxygen_oxygen_distance_as_the_model_engine_assumes():
    barriers = {}
    for distance in (2.6, 2.8, 3.0):
        out = QCScanEngine().solve(zundel_system("pyscf", scan_distance=distance), None)
        barriers[distance] = out.obs("classical_barrier")
    assert 0.08 < barriers[2.6] < 0.3 and 0.3 < barriers[2.8] < 0.7 and 0.7 < barriers[3.0] < 1.2     # eV, HF/6-31G*
    assert barriers[2.6] < barriers[2.8] < barriers[3.0]


def test_a_rerun_is_served_entirely_from_the_cache(surface):
    again = QCScanEngine().solve(zundel_system("pyscf", scan_distance=2.8), None)
    assert again.obs("calculations_run") == 0 and again.obs("surface_energy") == pytest.approx(surface.obs("surface_energy"), abs=0)


def test_the_provenance_records_which_program_and_method_made_the_numbers(surface):
    assert surface.observables["surface_energy"].source.endswith(":pyscf")
    assert "hf/6-31g*" in surface.dynamics
    assert any("rigid scan" in a for a in QCScanEngine.approximations)


# -- through the whole pipeline ------------------------------------------------------------------------------------------------
def test_the_quantum_route_gives_a_symmetric_rate_with_a_large_isotope_effect(pipeline):
    route = ["electronic_structure", "quantum", "reaction"]
    h = pipeline.run(zundel_system("pyscf", scan_distance=2.8, mass=PROTON), route).final
    d = pipeline.run(zundel_system("pyscf", scan_distance=2.8, mass=DEUTERON), route).final
    assert h.param("k_f") == pytest.approx(h.param("k_r"), rel=1e-3)                    # a symmetric ion: forward = reverse
    assert h.param("k_f") > 1e6 and h.param("k_f") > 50 * d.param("k_f")                # tunnelling-dominated isotope effect


def test_the_molecular_route_refuses_a_delocalised_proton_for_a_stated_reason(pipeline):
    # On the relaxed surface the heavy atoms can approach and the proton is shared: there is no proton-transfer reaction,
    # and the pipeline says so rather than returning a rate. (The frozen-distance quantum route above is the right model
    # when something, such as an enzyme active site, holds the donor and acceptor apart.)
    with pytest.raises(ValidationError, match="free-energy barrier is not positive|fewer than two wells"):
        pipeline.run(zundel_system("pyscf", scan_distance=2.8), ["electronic_structure", "molecular", "reaction"])


def test_real_chemistry_reaches_a_pathway_flux(pipeline):
    from conftest import with_biology
    system = with_biology(zundel_system("pyscf", scan_distance=2.8), k_on=1e10, k_release=1e12, enzyme_total=1e-9,
                          external_substrate=1e-1, transport_rate=1e4, drain_vmax=10.0, drain_km=1e-3)
    r = pipeline.run(system, ["electronic_structure", "quantum", "reaction", "biophysical", "biological"])
    assert len(r.trace) == 9 and r.final.obs("pathway_flux") > 0
    previous = r.root
    for s in r.trace:                                    # unbroken hash chain, from ab initio energies to a flux
        assert s.provenance[-1].input_fingerprint == previous.fingerprint()
        previous = s
    assert r.final.provenance[0].name == "electronic.qc_scan_2d"


def test_the_zundel_experiment_file_runs_from_the_cli(capsys):
    from substrate.cli import main
    assert main(["run", str(EXPERIMENTS / "zundel_hf.yaml")]) == 0
    out = capsys.readouterr().out
    assert "electronic.qc_scan_2d" in out and "calculations_cached" in out and "k_f" in out
