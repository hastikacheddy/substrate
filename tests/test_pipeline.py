import json

import numpy as np
import pytest

from substrate import (
    AmbiguousPathError, ClassicalBackend, NoPathError, Pipeline, Registry, Scale, ScientificSystem, SubstrateError,
    Translator, ValidationError, default_registry, get_backend, load_experiment, register_backend,
)
from substrate.cli import main
from substrate.workflow import Workflow

from conftest import EXPERIMENTS, double_well


# -- planning ---------------------------------------------------------------------------
def test_plan_discovers_the_scale_chain(pipeline):
    steps = pipeline.plan(double_well(), ["quantum", "reaction"])
    assert [(s.kind, s.scale_out) for s in steps] == [
        ("solve", Scale.QUANTUM), ("translate", Scale.REACTION), ("solve", Scale.REACTION),
    ]
    # the starting scale may be omitted: it is implied by the root system
    implied = pipeline.plan(double_well(), ["reaction"])
    assert [(s.kind, s.scale_out) for s in implied] == [(s.kind, s.scale_out) for s in steps]
    assert [s.kind for s in pipeline.plan(double_well(), ["quantum"])] == ["solve"]


def test_unreachable_scale_names_what_is_reachable(pipeline):
    with pytest.raises(NoPathError, match="you can reach") as exc:
        pipeline.plan(double_well(), ["electronic_structure"])       # no translator goes back down a scale
    assert "reaction" in str(exc.value) and "biological" in str(exc.value)


def test_missing_engine_is_reported(pipeline):
    registry = Registry()
    with pytest.raises(SubstrateError, match="no engine registered for scale 'quantum'"):
        Pipeline(registry).plan(double_well(), ["quantum"])


class _Hop(Translator):
    def __init__(self, name, source, target, source_kinds=(), target_kind="any"):
        self.name, self.source, self.target = name, source, target
        self.source_kinds, self.target_kind = source_kinds, target_kind

    def translate(self, system):  # pragma: no cover - planning only
        raise NotImplementedError


def test_paths_respect_model_kinds():
    reg = Registry()
    reg.register_translator(_Hop("only-for-a", Scale.QUANTUM, Scale.REACTION, source_kinds=("kind.a",)))
    assert [t.name for t in reg.find_path(Scale.QUANTUM, Scale.REACTION, "kind.a")] == ["only-for-a"]
    with pytest.raises(NoPathError, match="you can reach: nothing"):
        reg.find_path(Scale.QUANTUM, Scale.REACTION, "kind.b")
    assert reg.find_path(Scale.QUANTUM, Scale.REACTION, None)       # unknown kind: nothing is excluded


def test_kinds_are_tracked_across_hops():
    reg = Registry()
    reg.register_translator(_Hop("q-m", Scale.QUANTUM, Scale.MOLECULAR, target_kind="mol.x"))
    reg.register_translator(_Hop("m-r", Scale.MOLECULAR, Scale.REACTION, source_kinds=("mol.y",)))
    with pytest.raises(NoPathError):                               # q-m produces mol.x, which m-r does not accept
        reg.find_path(Scale.QUANTUM, Scale.REACTION)


def test_equally_short_but_different_routes_are_ambiguous_not_guessed():
    reg = Registry()
    for name, mid in (("via-quantum", Scale.QUANTUM), ("via-molecular", Scale.MOLECULAR)):
        reg.register_translator(_Hop(f"e-{name}", Scale.ELECTRONIC_STRUCTURE, mid))
        reg.register_translator(_Hop(f"{name}-r", mid, Scale.REACTION))
    with pytest.raises(AmbiguousPathError, match="2 distinct 2-hop routes") as exc:
        reg.find_path(Scale.ELECTRONIC_STRUCTURE, Scale.REACTION)
    assert "quantum" in str(exc.value) and "molecular" in str(exc.value) and "propagation" in str(exc.value)
    # naming the intermediate scale resolves it
    assert [t.name for t in reg.find_path(Scale.ELECTRONIC_STRUCTURE, Scale.MOLECULAR)] == ["e-via-molecular"]


def test_find_path_prefers_fewest_hops_and_chains_when_needed():
    reg = Registry()
    chain = [_Hop("q-m", Scale.QUANTUM, Scale.MOLECULAR), _Hop("m-r", Scale.MOLECULAR, Scale.REACTION)]
    for t in chain:
        reg.register_translator(t)
    assert [t.name for t in reg.find_path(Scale.QUANTUM, Scale.REACTION)] == ["q-m", "m-r"]
    reg.register_translator(_Hop("q-r", Scale.QUANTUM, Scale.REACTION))
    assert [t.name for t in reg.find_path(Scale.QUANTUM, Scale.REACTION)] == ["q-r"]
    assert reg.find_path(Scale.QUANTUM, Scale.QUANTUM) == []
    with pytest.raises(NoPathError):
        reg.find_path(Scale.REACTION, Scale.QUANTUM)           # translators are directed


# -- execution, provenance, persistence -------------------------------------------------
def test_end_to_end_proton_transfer(pipeline):
    r = pipeline.run(double_well(), ["quantum", "reaction"])
    assert [s.scale for s in r.trace] == [Scale.QUANTUM, Scale.REACTION, Scale.REACTION]
    pops = r.final.obs("final_concentrations")
    assert pops.sum() == pytest.approx(1.0)
    assert r.final.obs("relaxation_time") == pytest.approx(1 / (r.final.param("k_f") + r.final.param("k_r")), rel=1e-5)


def test_run_does_not_mutate_the_input(pipeline):
    root = double_well()
    before = root.fingerprint()
    pipeline.run(root, ["reaction"])
    assert root.fingerprint() == before and not root.observables and not root.provenance


def test_provenance_forms_an_unbroken_hash_chain(pipeline):
    r = pipeline.run(double_well(), ["reaction"])
    assert [len(s.provenance) for s in r.trace] == [1, 2, 3]
    previous = r.root
    for system in r.trace:
        record = system.provenance[-1]
        assert record.input_fingerprint == previous.fingerprint()
        assert record.output_fingerprint == system.fingerprint()
        assert record.approximations and record.scale_out == system.scale.label
        previous = system
    assert r.trace[0].provenance[-1].backend == "classical"
    assert r.trace[1].provenance[-1].backend is None           # translators are backend-free


def test_fingerprint_is_deterministic_and_sensitive(pipeline):
    a = pipeline.run(double_well(), ["reaction"]).final.fingerprint()
    assert pipeline.run(double_well(), ["reaction"]).final.fingerprint() == a
    assert pipeline.run(double_well(barrier=0.51), ["reaction"]).final.fingerprint() != a


def test_json_round_trip_preserves_identity(pipeline):
    final = pipeline.run(double_well(), ["reaction"]).final
    again = ScientificSystem.from_json(final.to_json())
    assert again.fingerprint() == final.fingerprint()
    assert np.array_equal(again.obs("concentrations"), final.obs("concentrations"))
    assert [p.name for p in again.provenance] == [p.name for p in final.provenance]


def test_save_persists_every_stage(pipeline, tmp_path):
    r = pipeline.run(double_well(), ["reaction"], experiment_id="EXP-T")
    out = r.save(tmp_path / "run")
    names = sorted(p.name for p in out.iterdir())
    assert names == ["00_input.json", "01_quantum_solve.json", "02_reaction_translate.json",
                     "03_reaction_solve.json", "run.json"]
    manifest = json.loads((out / "run.json").read_text())
    assert manifest["experiment_id"] == "EXP-T" and len(manifest["trace"]) == 3
    assert manifest["trace"][-1] == r.final.fingerprint()


# -- backend seam ---------------------------------------------------------------------------
def test_custom_backend_is_used_and_recorded():
    @register_backend
    class Counting(ClassicalBackend):
        name = "counting"
        eig_calls = 0
        ode_calls = 0

        def lowest_eigenpairs(self, diag, offdiag, k):
            Counting.eig_calls += 1
            return super().lowest_eigenpairs(diag, offdiag, k)

        def integrate_ode(self, rhs, y0, t_eval):
            Counting.ode_calls += 1
            return super().integrate_ode(rhs, y0, t_eval)

    r = Pipeline(default_registry(), get_backend("counting")).run(double_well(), ["reaction"])
    assert Counting.eig_calls == 3 and Counting.ode_calls == 1     # full domain + two wells; one kinetics solve
    assert {s.provenance[-1].backend for s in (r.trace[0], r.trace[2])} == {"counting"}
    with pytest.raises(KeyError, match="no backend 'nope'"):
        get_backend("nope")


# -- uncertainty ensemble ---------------------------------------------------------------------
def test_ensemble_propagates_input_uncertainty_through_the_chain(pipeline):
    r = pipeline.run(double_well(sigma=0.02), ["reaction"], n_samples=30, seed=3)
    assert r.ensemble["n_ok"] == 30 and r.ensemble["varied"] == ["barrier_height"]
    k_f = r.trace[1].parameters["k_f"]
    assert 0 < k_f.sigma < k_f.value                            # tens of percent, from a 20 meV barrier sigma
    assert r.final.observables["relaxation_time"].sigma > 0
    assert r.trace[1].parameters["temperature"].sigma is None   # not varied, so no spurious uncertainty
    assert r.final.provenance[-1].step == "uncertainty"
    assert r.trace[0].parameters["barrier_height"].sigma == 0.02   # the user's input sigma is kept


def test_ensemble_is_reproducible_per_seed(pipeline):
    sigma = lambda seed: pipeline.run(double_well(sigma=0.02), ["reaction"], n_samples=12, seed=seed) \
        .trace[1].parameters["k_f"].sigma
    assert sigma(5) == pytest.approx(sigma(5), rel=1e-9)      # same draws; only last-bit float noise differs
    assert sigma(5) != pytest.approx(sigma(6), rel=1e-3)


def test_ensemble_counts_and_drops_samples_that_fail_validation(pipeline):
    # mean barrier sits just above the zero-point energy, so some draws land in the invalid regime
    r = pipeline.run(double_well(barrier=0.14, sigma=0.04), ["reaction"], n_samples=60, seed=1)
    assert r.ensemble["n_failed"] > 0
    assert r.ensemble["n_ok"] + r.ensemble["n_failed"] == 60
    assert any("failed validation" in w for w in r.warnings())


def test_ensemble_without_any_sigma_is_an_error(pipeline):
    with pytest.raises(SubstrateError, match="no root parameter has a sigma"):
        pipeline.run(double_well(), ["reaction"], n_samples=10)


# -- workflow ----------------------------------------------------------------------------------
def test_workflow_runs_a_diamond_in_dependency_order():
    order = []
    wf = Workflow()
    wf.add("a", lambda: order.append("a") or 1)
    wf.add("b", lambda a: order.append("b") or a + 10, ("a",))
    wf.add("c", lambda a: order.append("c") or a + 100, ("a",))
    wf.add("d", lambda b, c: order.append("d") or b + c, ("b", "c"))     # fan-in
    results = wf.run()
    assert results["d"] == 112 and order[0] == "a" and order[-1] == "d"
    with pytest.raises(ValueError, match="duplicate"):
        wf.add("a", lambda: 0)
    with pytest.raises(ValueError, match="unknown task"):
        wf.add("e", lambda x: x, ("nope",))


# -- experiment files and CLI --------------------------------------------------------------------
def test_example_experiment_file_loads_and_runs(pipeline):
    exp = load_experiment(EXPERIMENTS / "proton_transfer.yaml")
    assert exp.id == "EXP-PT-001" and exp.propagation == [Scale.QUANTUM, Scale.REACTION]
    assert exp.system.parameters["barrier_height"].sigma == 0.02
    r = pipeline.run(exp.system, exp.propagation, n_samples=10, seed=exp.seed)
    assert r.final.scale == Scale.REACTION


def test_incomplete_experiment_spec_is_a_validation_error():
    from substrate.experiment import parse_experiment
    with pytest.raises(ValidationError, match="missing"):
        parse_experiment({"experiment": {"phenomenon": "x"}})
    with pytest.raises(ValidationError, match="unknown scale"):
        parse_experiment({"experiment": {"system": {"scale": "nonsense", "kind": "k"}}})


def test_cli_runs_and_reports_errors(capsys, tmp_path):
    assert main(["run", str(EXPERIMENTS / "proton_transfer.yaml"), "--samples", "5", "--out", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    assert "SCALE PROPAGATION" in out and "k_f" in out and (tmp_path / "o" / "run.json").exists()

    bad = tmp_path / "bad.yaml"
    bad.write_text("experiment:\n  system: {scale: quantum, kind: quantum.double_well_1d}\n  propagation: [electronic_structure]\n")
    assert main(["run", str(bad)]) == 1
    assert "no translator chain" in capsys.readouterr().err


def test_ensemble_bands_survive_serialisation_and_change_identity(pipeline):
    plain = pipeline.run(double_well(sigma=0.02), ["reaction"], n_samples=12, seed=1).trace[1]
    k_f = plain.parameters["k_f"]
    assert k_f.band is not None and k_f.band[0] < k_f.band[1]
    again = ScientificSystem.from_json(plain.to_json())
    assert again.parameters["k_f"].band == pytest.approx(k_f.band) and again.fingerprint() == plain.fingerprint()
    assert plain.parameters["temperature"].band is None                  # arrays and unvaried quantities get none
    shifted = plain.evolve(parameters={**plain.parameters, "k_f": replace_band(k_f)})
    assert shifted.fingerprint() != plain.fingerprint()


def replace_band(q):
    from dataclasses import replace
    return replace(q, band=(q.band[0] * 1.5, q.band[1]))


# -- the ODE solver is shared by every thread of a process (the GUI serves several requests at once) -----------------------------------------
def test_concurrent_integrations_in_one_process_do_not_collide():
    """SciPy's LSODA wraps a Fortran routine that can solve one problem at a time per process: two threads integrating at once fail with
    "Integrator `lsoda` can be used to solve only one problem at a time". The GUI's server is multithreaded, so two browser tabs running an
    experiment together hit it. The backend serialises the integration; each thread must still get the right answer for its own problem."""
    import threading
    import time

    from substrate import ClassicalBackend
    backend, results, errors = ClassicalBackend(), {}, []

    def work(rate):
        try:
            def rhs(t, y):
                time.sleep(0.0002)                                        # a slow right-hand side makes the threads overlap
                return -rate * y
            t = np.linspace(0.0, 1.0, 50)
            results[rate] = backend.integrate_ode(rhs, np.array([1.0]), t)[:, 0]
        except Exception as error:                                         # recorded: a failure in a thread must fail the test, not vanish
            errors.append(repr(error))

    threads = [threading.Thread(target=work, args=(rate,)) for rate in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    for rate, y in results.items():
        assert y == pytest.approx(np.exp(-rate * np.linspace(0.0, 1.0, 50)), abs=1e-7), rate          # each thread solved its own equation
