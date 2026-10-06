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
