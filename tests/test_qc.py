"""The quantum-chemistry engine.

Two layers. The plumbing (geometry construction, caching, symmetry, routing, failure handling) is tested against a fake
program whose energies are an analytic formula, so a mistake in the engine shows up as a mismatch with that formula and
the tests run in milliseconds. The chemistry is tested against real PySCF, through WSL on this machine, and skipped where
PySCF is not installed.
"""
import json
import math
import sys
import tempfile
from pathlib import Path, PureWindowsPath

import numpy as np
import pytest

from substrate import (
    AmbiguousPathError, Quantity, QCError, ScientificSystem, ValidationError, load_experiment,
)
from substrate.engines.electronic import EVBFlexibleProtonTransferEngine
from substrate.engines.qc_scan import QCScanEngine, build_geometry, validate_molecule
from substrate.molecules import zundel_cation
from substrate.qc import (
    PySCFProgram, QCCache, QCJob, QCProgram, QCResult, _to_wsl_path, _wsl_path, compute_cached, get_program, register_program,
)
from substrate.translators.electronic_to_quantum import ElectronicToQuantum

from conftest import DEUTERON, PROTON, evb2d, zundel_system

BACKEND = None          # the QC engine does not use the numerical backend

#: CODATA 2018. Written out here, NOT imported from the engine: if the fake and the engine shared one constant, a wrong
#: value would cancel between them and no test would notice.
HARTREE_EV = 27.211386245988


# =====================================================================================================
# a fake program: an analytic energy surface, so engine mistakes show up as mismatches with a formula
# =====================================================================================================
def _morse(r, depth, alpha, r_eq):
    return depth * (1.0 - math.exp(-alpha * (r - r_eq))) ** 2


class FakeEVB(QCProgram):
    """Energies from the two-state valence-bond formula (the same surface as the model 2D engine, written out independently
    here), computed from the atom positions the engine sends. A rigid-group penalty makes any flank atom that fails to move
    with its heavy atom cost energy, so mistakes in group handling cannot hide."""

    name = "fake"

    def __init__(self, offset=0.05, nonconverged_above=None, fail_between=None):
        self.offset, self.nonconverged_above, self.fail_between = offset, nonconverged_above, fail_between
        self.calls = 0
        atoms = zundel_cation()["atoms"]
        self.ref = {i: math.dist(atoms[i][1:], atoms[0 if i in (3, 4) else 1][1:]) for i in (3, 4, 5, 6)}

    def compute(self, jobs):
        out = []
        for job in jobs:
            self.calls += 1
            a = job.atoms
            z_d, z_a, z_h = a[0][3], a[1][3], a[2][3]
            r = z_a - z_d
            if self.fail_between is not None and self.fail_between[0] < r < self.fail_between[1]:
                raise QCError("the fake program was told to crash here")
            va = _morse(z_h - z_d, 4.6, 2.2, 0.96)
            vb = _morse(z_a - z_h, 4.6, 2.2, 0.96) + self.offset
            delta = 0.6 * math.exp(-3.0 * (r - 2.5))
            ground = 0.5 * (va + vb) - math.sqrt((0.5 * (va - vb)) ** 2 + delta**2) + _morse(r, 0.4, 2.5, 2.7)
            rigid = sum(10.0 * (math.dist(a[i][1:], a[0 if i in (3, 4) else 1][1:]) - self.ref[i]) ** 2 for i in (3, 4, 5, 6))
            ok = not (self.nonconverged_above is not None and r > self.nonconverged_above)
            out.append(QCResult((ground + rigid) / HARTREE_EV, ok, 0.3, "fake"))
        return out


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setenv("SUBSTRATE_CACHE_DIR", str(tmp_path))
    program = FakeEVB()
    register_program("fake", lambda: program)
    return program


def run(system):
    return QCScanEngine().solve(system, BACKEND)


# -- geometry -------------------------------------------------------------------------------------------------
def test_geometry_puts_the_atoms_where_the_coordinates_say_and_moves_groups_rigidly():
    molecule = zundel_cation()
    reference = molecule["atoms"]
    atoms = build_geometry(molecule, x=0.31, r=2.75)
    assert atoms[0][3] == pytest.approx(-1.375) and atoms[1][3] == pytest.approx(1.375)     # donor / acceptor at -R/2, +R/2
    assert atoms[2][3] == pytest.approx(0.31)                                                # the proton at x on the axis
    for group, heavy in (([3, 4], 0), ([5, 6], 1)):
        for i in group:
            # same offset from its own heavy atom as in the reference: the group travelled rigidly with it
            assert (atoms[i][1] - atoms[heavy][1], atoms[i][2] - atoms[heavy][2], atoms[i][3] - atoms[heavy][3]) == pytest.approx(
                (reference[i][1] - reference[heavy][1], reference[i][2] - reference[heavy][2], reference[i][3] - reference[heavy][3]))
    assert [a[0] for a in atoms] == [a[0] for a in reference]


def test_the_molecule_is_validated_before_any_energy_is_computed():
    good = zundel_cation()
    validate_molecule("m", good)
    cases = {
        "in no group": {**good, "donor_group": [3]},                       # atom 4 would not move with anything
        "both": {**good, "acceptor_group": [4, 5, 6]},                     # atom 4 claimed twice
        "refers to atom 9": {**good, "donor_group": [3, 9]},
        "z axis": {**good, "atoms": [["O", 0.1, 0.0, -1.2], *good["atoms"][1:]]},
        "lower z": {**good, "donor": 1, "acceptor": 0},
    }
    for message, bad in cases.items():
        with pytest.raises(ValidationError, match=message):
            validate_molecule("m", bad)


# -- the surface is the formula, in the right place -------------------------------------------------------------------
def test_the_surface_equals_the_analytic_energy_at_every_grid_point(fake):
    out = run(zundel_system("fake", n_x=21, n_r=9, mirror=False, scan_distance=2.6))
    x, r, surface = out.obs("surface_x"), out.obs("surface_r"), out.obs("surface_energy")
    assert surface.shape == (21, 9) and out.obs("scan_energy").shape == (21,)

    def exact(xi, rj):                                   # the formula, evaluated directly at (x, R)
        va, vb = _morse(rj / 2 + xi, 4.6, 2.2, 0.96), _morse(rj / 2 - xi, 4.6, 2.2, 0.96) + 0.05
        d = 0.6 * math.exp(-3.0 * (rj - 2.5))
        return 0.5 * (va + vb) - math.sqrt((0.5 * (va - vb)) ** 2 + d**2) + _morse(rj, 0.4, 2.5, 2.7)

    expected = np.array([[exact(xi, rj) for rj in r] for xi in x])
    reference = min(expected.min(), min(exact(xi, 2.6) for xi in x))
    assert surface == pytest.approx(expected - reference, abs=1e-9)
    assert out.obs("scan_energy") == pytest.approx([exact(xi, 2.6) - reference for xi in x], abs=1e-9)
    assert out.obs("surface_energy").min() >= 0 and out.obs("reference_energy") * HARTREE_EV == pytest.approx(reference, abs=1e-9)


def test_the_slice_is_a_column_of_the_surface_when_scan_distance_is_a_grid_distance(fake):
    out = run(zundel_system("fake", n_x=21, n_r=15, mirror=False, scan_distance=2.65))   # 2.3 + 7 * 0.05
    j = int(np.argmin(np.abs(out.obs("surface_r") - 2.65)))
    assert out.obs("scan_energy") == pytest.approx(out.obs("surface_energy")[:, j], abs=1e-9)
    assert out.obs("calculations_run") == 21 * 15                       # the slice reused the grid's calculations


def test_the_quantum_chemistry_engine_is_a_drop_in_for_the_model_engine(fake, pipeline, monkeypatch):
    """With a program that returns the model's energies, the whole molecular route must give the model engine's answers."""
    monkeypatch.setenv("SUBSTRATE_MAX_QC_JOBS", "20000")                    # the model engine's 161 x 91 grid is 14,813 jobs: far above the default limit
    system = zundel_system("fake", n_x=161, n_r=91, mirror=False, scan_distance=2.5, x_extent=0.65,
                           distance_min=2.2, distance_max=3.1)
    model = evb2d()
    model.parameters["particle_mass"], model.parameters["heavy_atom_mass"] = system.parameters["particle_mass"], system.parameters["heavy_atom_mass"]
    route = ["electronic_structure", "molecular"]
    real, ref = pipeline.run(system, route), pipeline.run(model, route)
    assert real.trace[0].obs("surface_energy") == pytest.approx(
        ref.trace[0].obs("surface_energy") - ref.trace[0].obs("surface_energy").min(), abs=1e-9)
    for name in ("barrier_classical", "barrier_zpe", "imaginary_frequency", "rate_tst_forward"):
        assert real.final.obs(name) == pytest.approx(ref.final.obs(name), rel=1e-6), name


# -- symmetry ---------------------------------------------------------------------------------------------------------
def test_mirror_symmetry_halves_the_work_and_gives_the_same_surface(tmp_path, monkeypatch):
    monkeypatch.setenv("SUBSTRATE_CACHE_DIR", str(tmp_path / "a"))
    symmetric = FakeEVB(offset=0.0)
    register_program("fake", lambda: symmetric)
    mirrored = run(zundel_system("fake", n_x=21, n_r=9, mirror=True))
    calls_mirrored = symmetric.calls
    monkeypatch.setenv("SUBSTRATE_CACHE_DIR", str(tmp_path / "b"))
    symmetric.calls = 0
    full = run(zundel_system("fake", n_x=21, n_r=9, mirror=False))
    assert mirrored.obs("surface_energy") == pytest.approx(full.obs("surface_energy"), abs=1e-12)
    assert calls_mirrored < 0.6 * symmetric.calls                          # roughly half the energies


def test_a_false_symmetry_claim_is_caught_not_trusted(fake):
    with pytest.raises(ValidationError, match="not symmetric under exchanging donor and acceptor"):
        run(zundel_system("fake", mirror=True))                            # the fake has a 0.05 eV offset: not symmetric


# -- cache ----------------------------------------------------------------------------------------------------------------
def test_a_rerun_computes_nothing_and_a_different_method_recomputes(fake):
    first = run(zundel_system("fake", n_x=21, n_r=9, mirror=False))
    n = fake.calls
    again = run(zundel_system("fake", n_x=21, n_r=9, mirror=False))
    assert fake.calls == n and again.obs("calculations_run") == 0 and again.obs("calculations_cached") == first.obs("calculations_run")
    assert again.obs("surface_energy") == pytest.approx(first.obs("surface_energy"), abs=0)
    run(zundel_system("fake", basis="cc-pvdz", n_x=21, n_r=9, mirror=False))     # a different basis is a different calculation
    assert fake.calls == 2 * n


def test_an_explicitly_passed_empty_cache_is_used_not_replaced_by_the_default(tmp_path, monkeypatch):
    # Regression: `cache or QCCache()` treated an empty cache (it defines __len__) as falsy and silently used the default
    # cache instead, so results landed in the user's real cache and stale hits could leak into a "fresh" one.
    monkeypatch.setenv("SUBSTRATE_CACHE_DIR", str(tmp_path / "default"))
    explicit = QCCache(tmp_path / "explicit.sqlite3")
    assert len(explicit) == 0 and not explicit                      # the trap: empty means falsy
    job = QCJob(tuple((s, float(x), float(y), float(z)) for s, x, y, z in zundel_cation()["atoms"]), 1, 0, "hf", "x")
    compute_cached(FakeEVB(), [job], explicit)
    assert len(explicit) == 1
    assert len(QCCache()) == 0                                       # the default cache was never touched


def test_cache_keys_and_storage(tmp_path):
    atoms = (("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.74))
    job = QCJob(atoms, 0, 0, "hf", "sto-3g")
    assert job.key("p") == QCJob(tuple((s, x, y, z + 1e-10) for s, x, y, z in atoms), 0, 0, "HF", "STO-3G").key("p")
    for different in (QCJob(atoms, 1, 0, "hf", "sto-3g"), QCJob(atoms, 0, 2, "hf", "sto-3g"),
                      QCJob(atoms, 0, 0, "mp2", "sto-3g"), QCJob(atoms, 0, 0, "hf", "6-31g")):
        assert different.key("p") != job.key("p")
    assert job.key("other-program") != job.key("p")
    cache = QCCache(tmp_path / "c.sqlite3")
    cache.put_many({"k": QCResult(-1.5, True, 0.7, "x")})
    assert cache.get_many(["k", "missing"]) == {"k": QCResult(-1.5, True, 0.7, "x")} and len(cache) == 1


def test_duplicate_requests_are_computed_once_and_unconverged_results_are_not_kept(tmp_path):
    program = FakeEVB()
    atoms = tuple(zundel_cation()["atoms"])
    job = QCJob(tuple((s, float(x), float(y), float(z)) for s, x, y, z in atoms), 1, 0, "hf", "x")
    cache = QCCache(tmp_path / "c.sqlite3")
    results, computed, cached = compute_cached(program, [job, job, job], cache)
    assert (computed, cached, program.calls, len(results)) == (1, 0, 1, 3)
    bad = FakeEVB(nonconverged_above=0.0)
    results, computed, _ = compute_cached(bad, [job], QCCache(tmp_path / "d.sqlite3"))
    assert not results[0].converged and len(QCCache(tmp_path / "d.sqlite3")) == 0


# -- failure handling ---------------------------------------------------------------------------------------------------------
def test_scf_non_convergence_is_a_validation_error(tmp_path, monkeypatch):
    monkeypatch.setenv("SUBSTRATE_CACHE_DIR", str(tmp_path))
    register_program("fake", lambda: FakeEVB(nonconverged_above=2.9))
    with pytest.raises(ValidationError, match="SCF did not converge"):
        run(zundel_system("fake", mirror=False))


def test_a_program_failure_is_not_swallowed_by_the_ensemble(tmp_path, monkeypatch, pipeline):
    monkeypatch.setenv("SUBSTRATE_CACHE_DIR", str(tmp_path))
    # the program crashes only for heavy-atom distances in a window the 3-point grid (2.3, 2.65, 3.0) never visits, so the
    # nominal run succeeds and only some slice distances drawn by the ensemble reach it
    register_program("fake", lambda: FakeEVB(fail_between=(2.615, 2.64)))
    system = zundel_system("fake", mirror=False, scan_distance=2.6, n_x=21, n_r=3)
    system.parameters["scan_distance"] = Quantity(2.6, "angstrom", sigma=0.04)
    with pytest.raises(QCError, match="told to crash"):                                  # not counted as an invalid draw
        pipeline.run(system, ["electronic_structure"], n_samples=20, seed=1)


def test_bad_inputs_are_refused(fake):
    for bad, message in (
        (zundel_system("fake", n_x=20), "odd"),
        (zundel_system("fake", n_x=5), "odd and >= 9"),
        (zundel_system("fake", scan_distance=3.5), "scan_distance inside them"),
    ):
        with pytest.raises(ValidationError, match=message):
            run(bad)
    missing = zundel_system("fake")
    del missing.structure["method"]
    with pytest.raises(ValidationError, match="needs a 'method' section"):
        run(missing)
    with pytest.raises(QCError, match="no quantum-chemistry program"):
        run(zundel_system("nonexistent"))
    with pytest.raises(ValidationError, match="solves electronic_structure"):
        QCScanEngine().solve(evb2d(), BACKEND)


def test_a_machine_without_pyscf_or_wsl_is_reported_helpfully_and_the_real_tests_would_skip(monkeypatch):
    import importlib.util
    import shutil
    from substrate import QCUnavailableError
    monkeypatch.delenv("SUBSTRATE_QC_MODE", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None if name == "pyscf" else object())
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: None)
    program = PySCFProgram()
    assert program.available() is False                      # this is what makes tests/test_qc_real.py skip
    with pytest.raises(QCUnavailableError, match="setup_qc_env.sh"):
        program.mode()


def test_program_discovery_and_paths():
    with pytest.raises(QCError, match="unknown SUBSTRATE_QC_MODE"):
        PySCFProgram(mode="cloud").mode()
    assert get_program("pyscf").name == "pyscf"


def test_a_windows_path_maps_into_wsl_the_same_way_on_every_platform():
    """The mapping is string work on a Windows path, so it is tested everywhere; resolving a real path is only meaningful on Windows."""
    assert _wsl_path(PureWindowsPath("C:/Users/me/proj/file.py")) == "/mnt/c/Users/me/proj/file.py"
    assert _wsl_path(PureWindowsPath(r"D:\data\x.py")) == "/mnt/d/data/x.py"
    for unmappable in ("//server/share/x.py", "/no/drive/x.py"):
        with pytest.raises(QCError, match="only drive-letter paths"):
            _wsl_path(PureWindowsPath(unmappable))


@pytest.mark.skipif(sys.platform != "win32", reason="WSL is reached from Windows; elsewhere a path has no drive letter to map")
def test_a_real_path_is_resolved_before_it_is_mapped():
    relative = Path("some") / "x.py"                                                 # no drive letter until it is resolved
    assert _to_wsl_path(relative) == _wsl_path(PureWindowsPath(Path.cwd() / relative))
    assert _to_wsl_path(relative).startswith("/mnt/") and _to_wsl_path(relative).endswith("/some/x.py")


# -- routing and the translators ----------------------------------------------------------------------------------------------
def test_the_qc_kind_routes_like_the_model_2d_kind(fake, pipeline):
    system = zundel_system("fake", mirror=False)
    with pytest.raises(AmbiguousPathError, match="2 distinct 2-hop routes"):
        pipeline.plan(system, ["reaction"])
    assert len(pipeline.plan(system, ["electronic_structure", "molecular", "reaction"])) == 5
    assert len(pipeline.plan(system, ["electronic_structure", "quantum", "reaction"])) == 5


def test_scan_resolution_rules_in_the_quantum_translator(fake, pipeline):
    fine = pipeline.run(zundel_system("fake", n_x=21, n_r=9, mirror=False, scan_distance=2.7), ["quantum"])
    assert not any("scan spacing" in w for w in fine.warnings())
    coarse = ElectronicToQuantum().validate(
        fine.trace[0].evolve(observables={**fine.trace[0].observables,
                                          "scan_coordinate": Quantity(np.linspace(-0.7, 0.7, 15), "angstrom"),
                                          "scan_energy": Quantity(fine.trace[0].obs("scan_energy")[3:18], "eV")}))
    assert any("scan spacing" in i.message for i in coarse)                                 # 0.1 angstrom: right at the limit
    too_few = ElectronicToQuantum().validate(
        fine.trace[0].evolve(observables={**fine.trace[0].observables,
                                          "scan_coordinate": Quantity(np.linspace(-0.7, 0.7, 9), "angstrom"),
                                          "scan_energy": Quantity(fine.trace[0].obs("scan_energy")[:9], "eV")}))
    assert "scan must be >= 15" in too_few[0].message


def test_experiment_files_can_carry_a_molecule(tmp_path):
    path = tmp_path / "e.yaml"
    path.write_text(
        "experiment:\n  system:\n    scale: electronic_structure\n    kind: electronic.qc_scan_2d\n"
        "    structure:\n      molecule: {atoms: [[H, 0, 0, 0]], donor: 0}\n      method: {theory: hf, basis: sto-3g}\n"
        "    parameters:\n      scan_distance: {value: 2.8, unit: angstrom}\n  propagation: [electronic_structure]\n")
    exp = load_experiment(path)
    assert exp.system.structure["method"]["basis"] == "sto-3g" and exp.system.parameters["scan_distance"].value == 2.8


# -- coupled cluster costs minutes a point, so its jobs travel in small chunks -----------------------------------------------------------
class _Counting(QCProgram):
    name = "counting"

    def __init__(self):
        self.sizes = []

    def compute(self, jobs):
        self.sizes.append(len(jobs))
        return [QCResult(-1.0, True) for _ in jobs]


def _hydrogen_molecules(theory, n, start=0):
    return [QCJob((("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.5 + 0.01 * i)), 0, 0, theory, "6-31g") for i in range(start, start + n)]


def test_coupled_cluster_jobs_are_sent_in_chunks_of_six_and_cheap_ones_in_chunks_of_forty(tmp_path):
    cache = QCCache(tmp_path / "c.sqlite3")
    for theory, n, expected in (("ccsd(t)", 14, [6, 6, 2]), ("CCSD", 7, [6, 1]), ("hf", 85, [40, 40, 5]), ("mp2", 41, [40, 1])):
        program = _Counting()
        results, run, cached = compute_cached(program, _hydrogen_molecules(theory, n), cache)
        assert (program.sizes, run, cached) == (expected, n, 0), theory
    program = _Counting()                                                       # one coupled-cluster job makes the whole request small-chunked
    compute_cached(program, _hydrogen_molecules("hf", 9, start=500) + _hydrogen_molecules("ccsd(t)", 1, start=500), cache)
    assert program.sizes == [6, 4] and max(program.sizes) <= 6
    assert _hydrogen_molecules("ccsd(t)", 1)[0].key("p") != _hydrogen_molecules("ccsd", 1)[0].key("p")        # the two never share a cached energy


# -- relaxation jobs: a second task that shares the cache without disturbing a single cached energy --------------------------------------------
WATER_ATOMS = (("O", 0.0, 0.0, 0.1173), ("H", 0.0, 0.7572, -0.4692), ("H", 0.0, -0.7572, -0.4692))


def test_a_single_point_keeps_the_cache_key_it_had_before_relaxation_jobs_existed():
    """The task is added to the hash only when it is not "energy", so every energy already on disk is still found. The recorded key was computed
    before the field existed: if this fails, the change has invalidated every user's cache."""
    job = QCJob(WATER_ATOMS, 0, 0, "hf", "6-31g*")
    assert job.task == "energy" and job.key("pyscf") == "845240efa9b2a4bc7f1483134b9c5891fdc532f8288be6583eed97d833b05df9"
    relax = QCJob(WATER_ATOMS, 0, 0, "hf", "6-31g*", "relax")
    assert relax.key("pyscf") != job.key("pyscf")                                           # a relaxation never answers for a single point, or the reverse
    assert relax.key("pyscf") == QCJob(WATER_ATOMS, 0, 0, "HF", "6-31G*", "relax").key("pyscf")
    thermo = QCJob(WATER_ATOMS, 0, 0, "hf", "6-31g*", "thermo")
    assert thermo.key("pyscf") not in (job.key("pyscf"), relax.key("pyscf"))                 # the three tasks never share an answer


def test_a_result_carries_a_relaxed_geometry_through_json_and_an_old_one_has_none():
    result = QCResult(-76.0, True, None, "p", (("O", 0.0, 0.0, 0.1), ("H", 0.0, 0.76, -0.47)), -75.9)
    again = QCResult.from_dict(json.loads(json.dumps(result.to_dict())))
    assert again == result and again.geometry[0] == ("O", 0.0, 0.0, 0.1) and again.initial_energy_hartree == -75.9
    old = QCResult.from_dict({"energy": -1.0, "converged": True, "homo_lumo_gap": 0.3, "program": "p"})        # a payload written before relaxation existed
    assert old.geometry is None and old.initial_energy_hartree is None and old.energy_hartree == -1.0
    assert "geometry" not in QCResult(-1.0, True).to_dict()                                  # a single point's payload is unchanged


def test_a_result_carries_its_thermochemistry_through_json_and_a_relaxation_has_none():
    geometry = (("He", 0.0, 0.0, 0.0),)
    result = QCResult(-2.8, True, None, "p", geometry, -2.7, 0.0213, -76.01)
    again = QCResult.from_dict(json.loads(json.dumps(result.to_dict())))
    assert again == result and again.zero_point_hartree == 0.0213 and again.enthalpy_298_hartree == -76.01
    relaxation = QCResult.from_dict(QCResult(-2.8, True, None, "p", geometry, -2.7).to_dict())
    assert relaxation.zero_point_hartree is None and relaxation.enthalpy_298_hartree is None
    assert "zero_point" not in QCResult(-2.8, True, None, "p", geometry, -2.7).to_dict()
    from substrate.qc import CC_CHUNK, CHUNK, chunk_size
    assert chunk_size([QCJob(WATER_ATOMS, 0, 0, "hf", "6-31g*", "thermo")]) == CC_CHUNK                 # a Hessian per job: small chunks, like a relaxation
    assert chunk_size([QCJob(WATER_ATOMS, 0, 0, "hf", "6-31g*")]) == CHUNK


def test_the_program_sends_the_task_to_the_worker_and_reads_back_geometry_and_thermochemistry():
    """PySCFProgram between the engine and the worker, with the worker replaced by a recording stub (no PySCF needed)."""
    class Stub(PySCFProgram):
        def __init__(self):
            super().__init__(mode="local")
            self.requests, self.version = [], "stub 1"

        def _run_local(self, requests):
            self.requests = requests
            return [{"id": 0, "energy": -76.0, "converged": True, "homo_lumo_gap": None, "initial_energy": -75.9,
                     "geometry": [["O", 0.0, 0.0, 0.1], ["H", 0.0, 0.76, -0.47]], "zero_point": 0.021, "enthalpy_298": -75.97},
                    {"id": 1, "energy": -1.0, "converged": True, "homo_lumo_gap": 0.5}]

    stub = Stub()
    thermo, single = stub.compute([QCJob(WATER_ATOMS, 0, 0, "hf", "6-31g*", "thermo"), QCJob(WATER_ATOMS, 0, 0, "hf", "6-31g*")])
    assert [r["task"] for r in stub.requests] == ["thermo", "energy"] and [r["id"] for r in stub.requests] == [0, 1]
    assert thermo.geometry == (("O", 0.0, 0.0, 0.1), ("H", 0.0, 0.76, -0.47)) and thermo.initial_energy_hartree == -75.9
    assert (thermo.zero_point_hartree, thermo.enthalpy_298_hartree, thermo.program) == (0.021, -75.97, "stub 1")
    assert single.geometry is None and single.zero_point_hartree is None and single.homo_lumo_gap_hartree == 0.5       # a single point is read as before


def test_relaxation_jobs_are_cached_with_their_geometry_and_travel_in_small_chunks(tmp_path):
    class Relaxer(QCProgram):
        name = "relaxer"

        def __init__(self):
            self.sizes = []

        def compute(self, jobs):
            self.sizes.append(len(jobs))
            return [QCResult(-1.0 - j.atoms[0][3], True, None, "r", tuple((s, x, y, z * 0.5) for s, x, y, z in j.atoms), -0.5) for j in jobs]

    cache, program = QCCache(tmp_path / "c.sqlite3"), Relaxer()
    jobs = [QCJob((("H", 0.0, 0.0, float(i)), ("H", 0.0, 0.0, i + 0.8)), 0, 0, "hf", "6-31g", "relax") for i in range(8)]
    first, run, cached = compute_cached(program, jobs, cache)
    assert (program.sizes, run, cached) == ([6, 2], 8, 0)                                    # six at a time, like coupled cluster
    again, run, cached = compute_cached(program, jobs, cache)
    assert (run, cached, program.sizes) == (0, 8, [6, 2]) and again == first                 # all from disk, geometry and starting energy included
    assert again[3].geometry == (("H", 0.0, 0.0, 1.5), ("H", 0.0, 0.0, 1.9)) and again[3].initial_energy_hartree == -0.5
