"""Resource limits and untrusted input: a request that is too large, or not a number, is refused before anything is allocated or computed.

Every limit is exercised at the place that enforces it, with the work that would follow replaced by something that fails the test if it runs.
"""
import numpy as np
import pytest

from substrate import ClassicalBackend, Pipeline, Quantity, ResourceLimitError, Scale, ScientificSystem, ValidationError, default_registry
from substrate.cli import main
from substrate.engines.biological import MetabolicNetworkEngine
from substrate.engines.electronic import EVBFlexibleProtonTransferEngine, EVBProtonTransferEngine
from substrate.engines.qc_scan import QCScanEngine
from substrate.engines.quantum import DoubleWellEngine
from substrate.engines.reaction import MassActionEngine
from substrate.experiment import parse_experiment
from substrate.limits import DEFAULTS, finite, limit, require, whole_number
from substrate.qc import QCJob, QCProgram, QCResult, check_request, compute_cached, register_program

from conftest import double_well, evb, evb2d, zundel_system

BACKEND = ClassicalBackend()
ALL_LIMITS = list(DEFAULTS)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in DEFAULTS:
        monkeypatch.delenv("SUBSTRATE_" + name.upper(), raising=False)


# -- the limits themselves -----------------------------------------------------------------------------------------------------
def test_the_documented_defaults_are_the_defaults():
    """These numbers are quoted in SECURITY.md and docs/architecture.md: changing one is a decision, so it has to change here too."""
    assert DEFAULTS == {"max_ensemble": 2_000, "max_grid_points": 1_000_000, "max_qc_jobs": 2_000, "max_expensive_qc_jobs": 200,
                        "max_atoms": 50, "max_references": 60, "max_spec_bytes": 1_000_000}


def test_each_limit_has_a_default_and_an_environment_variable_that_replaces_it(monkeypatch):
    for name, default in DEFAULTS.items():
        assert limit(name) == default
        monkeypatch.setenv("SUBSTRATE_" + name.upper(), "7")
        assert limit(name) == 7
    with pytest.raises(KeyError, match="unknown limit"):
        limit("max_everything")


@pytest.mark.parametrize("bad", ["0", "-5", "many", "1.5"])
def test_a_limit_that_is_not_a_positive_whole_number_is_an_error_that_names_the_variable(monkeypatch, bad):
    monkeypatch.setenv("SUBSTRATE_MAX_ENSEMBLE", bad)
    with pytest.raises(ResourceLimitError, match="SUBSTRATE_MAX_ENSEMBLE"):
        limit("max_ensemble")


def test_a_request_over_the_limit_is_refused_with_the_number_the_limit_and_the_way_to_change_it(monkeypatch):
    require("max_ensemble", 2000, "x")                                                  # at the limit is fine
    with pytest.raises(ResourceLimitError, match=r"x: 2,001 exceeds the limit of 2,000 \(set SUBSTRATE_MAX_ENSEMBLE to raise it\)"):
        require("max_ensemble", 2001, "x")
    monkeypatch.setenv("SUBSTRATE_MAX_ENSEMBLE", "5000")
    require("max_ensemble", 2001, "x")


def test_a_resource_limit_is_not_a_validation_error_so_an_ensemble_cannot_drop_it():
    assert not issubclass(ResourceLimitError, ValidationError)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 2.5, True, "7", None, [3]])
def test_a_count_that_is_not_a_whole_number_is_invalid_input(value):
    with pytest.raises(ValidationError, match="whole number"):
        whole_number("owner", "n", value)


def test_whole_numbers_pass_through_as_ints_and_the_bounds_apply():
    assert whole_number("o", "n", 5) == 5 and whole_number("o", "n", 5.0) == 5 and whole_number("o", "n", np.float64(8)) == 8
    assert isinstance(whole_number("o", "n", 5.0), int)
    with pytest.raises(ValidationError, match="at least 0"):
        whole_number("o", "n", -1, minimum=0)
    with pytest.raises(ResourceLimitError, match="o: n: 3,000"):
        whole_number("o", "n", 3000, maximum="max_ensemble")


def test_finite_refuses_nan_infinity_and_non_numbers_in_scalars_and_arrays():
    finite("o", "x", 1.0)
    finite("o", "x", [1.0, 2.0])
    for bad in (float("nan"), float("inf"), [1.0, float("nan")], np.array([0.0, -np.inf])):
        with pytest.raises(ValidationError, match="finite"):
            finite("o", "x", bad)
    with pytest.raises(ValidationError, match="not a number"):
        finite("o", "x", "hot")


# -- the engines refuse an oversized grid before building it --------------------------------------------------------------------
# These tests set a small limit and ask for a modestly larger size, so that if the guard under test were ever removed the work that follows is
# still small: a test of a guard must not depend on the guard to stay safe.
@pytest.fixture
def small(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_MAX_GRID_POINTS", "1000")


def test_the_one_dimensional_model_engine_refuses_a_huge_scan_and_a_fractional_one(small):
    with pytest.raises(ResourceLimitError, match="n_scan"):
        EVBProtonTransferEngine().solve(evb(n_scan=5000), BACKEND)
    with pytest.raises(ValidationError, match="n_scan"):
        EVBProtonTransferEngine().solve(evb(n_scan=float("nan")), BACKEND)
    with pytest.raises(ValidationError, match="n_scan"):
        EVBProtonTransferEngine().solve(evb(n_scan=120.5), BACKEND)


def test_the_two_dimensional_model_engine_limits_the_product_of_its_axes(small, monkeypatch):
    with pytest.raises(ResourceLimitError, match="n_x"):
        EVBFlexibleProtonTransferEngine().solve(evb2d(n_x=5000), BACKEND)
    with pytest.raises(ResourceLimitError, match=r"grid points \(n_x \* n_r\)"):
        EVBFlexibleProtonTransferEngine().solve(evb2d(n_x=500, n_r=500), BACKEND)           # each axis is within the limit, 250,000 points are not
    monkeypatch.setenv("SUBSTRATE_MAX_GRID_POINTS", "100")
    with pytest.raises(ResourceLimitError, match="grid points"):
        EVBFlexibleProtonTransferEngine().solve(evb2d(n_x=21, n_r=11), BACKEND)             # the default size, against a small limit


@pytest.mark.parametrize("axis", ["n_x", "n_r"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 20.5])
def test_the_two_dimensional_model_engine_refuses_an_axis_that_is_not_a_whole_number(axis, bad):
    with pytest.raises(ValidationError, match=axis):
        EVBFlexibleProtonTransferEngine().solve(evb2d(**{axis: bad}), BACKEND)


def test_the_schrodinger_engine_limits_its_grid_and_the_eigenvectors_it_would_hold(small, monkeypatch):
    with pytest.raises(ResourceLimitError, match="n_grid"):
        DoubleWellEngine().solve(double_well(n_grid=5000), BACKEND)
    with pytest.raises(ResourceLimitError, match="eigenvector entries"):
        DoubleWellEngine().solve(double_well(n_grid=800, n_levels=800), BACKEND)            # 800 points and 800 levels are each fine; 640,000 entries are not
    with pytest.raises(ValidationError, match="n_levels"):
        DoubleWellEngine().solve(double_well(n_grid=500, n_levels=float("inf")), BACKEND)
    for bad in (float("nan"), float("inf"), 1500.5):
        with pytest.raises(ValidationError, match="n_grid"):
            DoubleWellEngine().solve(double_well(n_grid=bad), BACKEND)
    monkeypatch.delenv("SUBSTRATE_MAX_GRID_POINTS")
    assert DoubleWellEngine().solve(double_well(), BACKEND).observables                    # the defaults are untouched


def test_the_reaction_engine_limits_its_time_axis(small):
    def system(n_time):
        return ScientificSystem(
            "net", Scale.REACTION, "reaction.network",
            parameters={"k_f": Quantity(3.0, "1/s"), "k_r": Quantity(1.0, "1/s"), "t_end": Quantity(1.0, "s"), "n_time": Quantity(n_time, "1")},
            state={"c": Quantity([1.0, 0.0], "M")},
            structure={"species": ["A", "B"],
                       "reactions": [{"id": "f", "reactants": {"A": 1}, "products": {"B": 1}, "rate_parameter": "k_f"},
                                     {"id": "r", "reactants": {"B": 1}, "products": {"A": 1}, "rate_parameter": "k_r"}]},
        )

    assert MassActionEngine().solve(system(200), BACKEND).observables                         # a normal size works
    with pytest.raises(ResourceLimitError, match="n_time"):
        MassActionEngine().solve(system(5000), BACKEND)


def test_the_biological_engine_limits_its_time_axis(small):
    structure = {
        "species": ["S"], "flux_reaction": "uptake",
        "reactions": [
            {"id": "transport", "law": "exchange", "stoich": {"S": 1}, "refs": {"a": "context.x0", "b": "S"}, "params": {"k": "context.k"}},
            {"id": "uptake", "law": "michaelis_menten", "stoich": {"S": -1}, "refs": {"x": "S"}, "params": {"vmax": "context.vmax", "km": "context.km"}},
        ],
    }
    params = {"context.x0": (1e-3, "M"), "context.k": (1.0, "1/s"), "context.vmax": (1e-4, "M/s"), "context.km": (1e-3, "M")}

    def system(n_time):
        parameters = {k: Quantity(v, u) for k, (v, u) in params.items()}
        parameters["n_time"] = Quantity(n_time, "1")
        return ScientificSystem("net", Scale.BIOLOGICAL, "biological.metabolic_network", parameters=parameters,
                                state={"c": Quantity([0.0], "M")}, structure=structure)

    assert MetabolicNetworkEngine().solve(system(200), BACKEND).observables                # a normal size works
    with pytest.raises(ResourceLimitError, match="n_time"):
        MetabolicNetworkEngine().solve(system(5000), BACKEND)


# -- the quantum-chemistry bridge ----------------------------------------------------------------------------------------------
class Counting(QCProgram):
    name = "counting"

    def __init__(self):
        self.calls = 0

    def compute(self, jobs):
        self.calls += len(jobs)
        return [QCResult(-1.0, True) for _ in jobs]


def jobs(n, task="energy", theory="hf", atoms=2):
    return [QCJob(tuple(("H", 0.0, 0.0, 0.8 * k + 0.001 * i) for k in range(atoms)), 0, 0, theory, "6-31g", task) for i in range(n)]


def test_a_request_for_too_many_jobs_is_refused_before_the_program_or_the_cache_is_touched(tmp_path, monkeypatch):
    program = Counting()
    with pytest.raises(ResourceLimitError, match="quantum-chemistry jobs in one request: 2,001"):
        compute_cached(program, jobs(2001), None)
    assert program.calls == 0
    results, ran, _ = compute_cached(program, jobs(2000), _cache(tmp_path))
    assert (len(results), ran) == (2000, 2000)
    monkeypatch.setenv("SUBSTRATE_MAX_QC_JOBS", "5")
    with pytest.raises(ResourceLimitError):
        compute_cached(program, jobs(6), _cache(tmp_path))


def _cache(tmp_path):
    from substrate.qc import QCCache
    return QCCache(tmp_path / "limits.sqlite3")


def test_the_expensive_jobs_have_a_smaller_limit_of_their_own(tmp_path, monkeypatch):
    program = Counting()
    for kind in ({"task": "relax"}, {"task": "thermo"}, {"theory": "ccsd(t)"}, {"theory": "ccsd"}):
        with pytest.raises(ResourceLimitError, match="coupled-cluster, relaxation and thermochemistry jobs"):
            compute_cached(program, jobs(201, **kind), _cache(tmp_path))
    assert program.calls == 0
    check_request(jobs(200, task="relax"))                                              # at the limit
    check_request(jobs(1000))                                                           # single points are not expensive jobs
    monkeypatch.setenv("SUBSTRATE_MAX_EXPENSIVE_QC_JOBS", "3")
    with pytest.raises(ResourceLimitError):
        check_request(jobs(4, task="thermo"))


def job_with(theory="hf", basis="6-31g"):
    return QCJob((("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.8)), 0, 0, theory, basis)


@pytest.mark.parametrize("basis", ["6-31G*", "6-31+G*", "aug-cc-pVTZ", "aug-cc-pV(T+d)Z", "def2-SVP", "sto-3g", "6-311++G(d,p)", "cc-pvdz", "ano_rcc"])
def test_real_basis_set_names_are_accepted(basis):
    check_request([job_with(basis=basis)])


BACKSLASH = chr(92)


@pytest.mark.parametrize("basis", ["/etc/passwd", "../secret", f"..{BACKSLASH}secret", f"C:{BACKSLASH}data{BACKSLASH}basis", "basis.dat", "my basis", "a;b", "a|b",
                                   "", "x" * 41, "a\nb", "$HOME"])
def test_a_basis_that_is_not_a_name_never_reaches_the_program(tmp_path, basis):
    """PySCF reads an unknown basis string as a file path, so anything path-like is refused before the program is called."""
    program = Counting()
    with pytest.raises(ValidationError, match="basis-set name"):
        compute_cached(program, [job_with(basis=basis)], _cache(tmp_path))
    assert program.calls == 0


@pytest.mark.parametrize("theory", ["hf", "HF", "mp2", "ccsd", "ccsd(t)", "dft:b3lyp", "dft:wb97x-d", "dft:m06-2x", "dft:b3lyp5", "dft:pbe0"])
def test_real_method_names_are_accepted(theory):
    check_request([job_with(theory=theory)])


@pytest.mark.parametrize("theory", ["dft:/etc/passwd", "hf; ls", "dft:a b", "", "hf\n", "dft:" + "x" * 80, "$(id)", f"dft:{BACKSLASH}{BACKSLASH}server{BACKSLASH}share"])
def test_a_theory_that_is_not_a_name_is_refused(theory):
    with pytest.raises(ValidationError, match="method name"):
        check_request([job_with(theory=theory)])


def test_a_job_with_too_many_atoms_is_refused(monkeypatch):
    check_request(jobs(1, atoms=50))
    with pytest.raises(ResourceLimitError, match="atoms in one quantum-chemistry job: 51"):
        check_request(jobs(1, atoms=51))
    monkeypatch.setenv("SUBSTRATE_MAX_ATOMS", "80")
    check_request(jobs(1, atoms=51))


def test_the_scan_engine_refuses_a_huge_grid_before_it_builds_a_single_geometry(tmp_path, monkeypatch):
    monkeypatch.setenv("SUBSTRATE_CACHE_DIR", str(tmp_path))
    program = Counting()
    register_program("counting", lambda: program)
    monkeypatch.setenv("SUBSTRATE_MAX_GRID_POINTS", "1000")
    with pytest.raises(ResourceLimitError, match="n_x"):
        QCScanEngine().solve(zundel_system("counting", n_x=5001, n_r=11), BACKEND)
    monkeypatch.setenv("SUBSTRATE_MAX_QC_JOBS", "100")
    with pytest.raises(ResourceLimitError, match="quantum-chemistry jobs in the scan: 133"):
        QCScanEngine().solve(zundel_system("counting", n_x=21, n_r=11), BACKEND)            # each axis is fine; 11 x (11 + 1) + 1 jobs (x >= 0 only, by mirror symmetry) are over the limit
    with pytest.raises(ValidationError, match="n_r"):
        QCScanEngine().solve(zundel_system("counting", n_x=21, n_r=float("nan")), BACKEND)
    assert program.calls == 0


# -- the pipeline and the command line ---------------------------------------------------------------------------------------
def test_an_oversized_or_malformed_ensemble_is_refused_before_the_chain_runs(monkeypatch):
    pipeline = Pipeline(default_registry())
    monkeypatch.setattr(Pipeline, "_execute", lambda *a, **k: pytest.fail("the chain ran"))
    with pytest.raises(ResourceLimitError, match="n_samples: 2,001"):
        pipeline.run(evb2d(), ["electronic_structure"], n_samples=2001)
    for bad, error in ((-1, ValidationError), (float("nan"), ValidationError), (2.5, ValidationError), (True, ValidationError)):
        with pytest.raises(error, match="n_samples"):
            pipeline.run(evb2d(), ["electronic_structure"], n_samples=bad)


def test_the_transfer_command_refuses_a_study_with_too_many_references(capsys):
    assert main(["transfer", *["experiments/zundel_hf.yaml"] * 61]) == 1
    assert "references in one transfer study" in capsys.readouterr().err


# -- experiment files are not trusted ----------------------------------------------------------------------------------------
GOOD = {"system": {"name": "w", "scale": "quantum", "kind": "quantum.double_well_1d",
                   "parameters": {"mass": {"value": 1.0, "unit": "amu", "sigma": 0.1}}}}


def spec(**changes):
    import copy
    raw = copy.deepcopy(GOOD)
    raw["system"]["parameters"]["mass"].update(changes)
    return raw


def test_a_good_experiment_parses():
    assert parse_experiment(GOOD).system.parameters["mass"].value == 1.0


@pytest.mark.parametrize("changes, match", [
    ({"value": float("nan")}, "must be finite"),
    ({"value": float("inf")}, "must be finite"),
    ({"value": [1.0, float("nan")]}, "must be finite"),
    ({"sigma": float("inf")}, "sigma of parameter 'mass' must be finite"),
    ({"sigma": -0.1}, "must not be negative"),
])
def test_a_non_finite_parameter_or_a_negative_sigma_is_refused_at_parse_time(changes, match):
    with pytest.raises(ValidationError, match=match):
        parse_experiment(spec(**changes))


def test_a_parameter_without_a_unit_and_a_spec_that_is_not_a_mapping_are_refused():
    raw = spec()
    del raw["system"]["parameters"]["mass"]["unit"]
    with pytest.raises(ValidationError, match="needs a value and a unit"):
        parse_experiment(raw)
    for bad in (None, [], "text", 7):
        with pytest.raises(ValidationError, match="must be a mapping"):
            parse_experiment(bad)


def test_the_ensemble_block_is_checked_like_any_other_count():
    for samples, error in ((10**6, ResourceLimitError), (-3, ValidationError), (1.5, ValidationError), (float("nan"), ValidationError)):
        raw = dict(GOOD, ensemble={"n_samples": samples})
        with pytest.raises(error, match="n_samples"):
            parse_experiment(raw)
    assert parse_experiment(dict(GOOD, ensemble={"n_samples": 500, "seed": 7})).n_samples == 500
    with pytest.raises(ValidationError, match="seed"):
        parse_experiment(dict(GOOD, ensemble={"seed": float("inf")}))


def test_the_experiment_file_cannot_raise_its_own_limits():
    raw = dict(GOOD, ensemble={"n_samples": 5000}, limits={"max_ensemble": 10**9})
    with pytest.raises(ResourceLimitError):
        parse_experiment(raw)
