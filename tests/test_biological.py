import numpy as np
import pytest

from substrate import (
    AmbiguousPathError, ClassicalBackend, Quantity, Scale, ScientificSystem, ValidationError,
)
from substrate.engines.biological import MetabolicNetworkEngine, _reversible_mm, pathway_structure
from substrate.engines.biophysical import EnzymeCycleEngine, _Cycle
from substrate.translators.biophysical_to_pathway import BiophysicalToPathway

from conftest import DEUTERON, EXPERIMENTS, enzyme, evb, evb2d, with_biology, with_context

BACKEND = ClassicalBackend()
ROUTE_M = ["electronic_structure", "molecular", "reaction", "biophysical", "biological"]


def network(structure, params, c0=None):
    """A biological system built directly. `params` maps parameter name -> (value, unit)."""
    n = len(structure["species"])
    return ScientificSystem(
        "net", Scale.BIOLOGICAL, "biological.metabolic_network",
        parameters={k: Quantity(v, u) for k, (v, u) in params.items()},
        state={"c": Quantity(c0 if c0 is not None else [0.0] * n, "M")},
        structure=structure,
    )


def solve(system):
    return MetabolicNetworkEngine().solve(system, BACKEND)


# -- a one-metabolite network with a closed-form steady state ------------------------------------------
def single_species(k, x0, vmax, km):
    structure = {
        "species": ["S"], "flux_reaction": "uptake",
        "reactions": [
            {"id": "transport", "law": "exchange", "stoich": {"S": 1}, "refs": {"a": "context.x0", "b": "S"},
             "params": {"k": "context.k"}},
            {"id": "uptake", "law": "michaelis_menten", "stoich": {"S": -1}, "refs": {"x": "S"},
             "params": {"vmax": "context.vmax", "km": "context.km"}},
        ],
    }
    params = {"context.x0": (x0, "M"), "context.k": (k, "1/s"), "context.vmax": (vmax, "M/s"), "context.km": (km, "M")}
    return network(structure, params)


def closed_form(k, x0, vmax, km):
    """k (x0 - S) = vmax S / (km + S)  is a quadratic in S: k S^2 + (k km + vmax - k x0) S - k x0 km = 0."""
    b = k * km + vmax - k * x0
    s = (-b + np.sqrt(b * b + 4 * k * k * x0 * km)) / (2 * k)
    return s, vmax * s / (km + s)


@pytest.mark.parametrize("k,x0,vmax,km", [(5.0, 1e-3, 1e-3, 1e-4), (0.1, 1e-2, 5e-3, 1e-3), (1e3, 1e-6, 1e-2, 1e-4), (2.0, 1.0, 1.0, 1.0)])
def test_steady_state_and_flux_match_the_closed_form(k, x0, vmax, km):
    out = solve(single_species(k, x0, vmax, km))
    s, j = closed_form(k, x0, vmax, km)
    assert out.obs("steady_S") == pytest.approx(s, rel=1e-8)
    assert out.obs("pathway_flux") == pytest.approx(j, rel=1e-8)
    assert out.obs("flux_transport") == pytest.approx(out.obs("flux_uptake"), rel=1e-8)     # a steady state conserves flux


def test_flux_control_matches_differentiating_the_closed_form_and_obeys_summation():
    k, x0, vmax, km = 5.0, 1e-3, 1e-3, 1e-4
    out = solve(single_species(k, x0, vmax, km))

    def log_flux(kk, vv):
        return np.log(closed_form(kk, x0, vv, km)[1])

    h = 1e-5
    c_transport = (log_flux(k * (1 + h), vmax) - log_flux(k * (1 - h), vmax)) / (np.log(1 + h) - np.log(1 - h))
    c_uptake = (log_flux(k, vmax * (1 + h)) - log_flux(k, vmax * (1 - h))) / (np.log(1 + h) - np.log(1 - h))
    # NB: scaling the *activity* of a reaction scales its whole rate; for exchange that is k, for MM it is vmax
    assert out.obs("flux_control_transport") == pytest.approx(c_transport, rel=1e-5)
    assert out.obs("flux_control_uptake") == pytest.approx(c_uptake, rel=1e-5)
    assert out.obs("flux_control_transport") + out.obs("flux_control_uptake") == pytest.approx(1.0, abs=1e-6)


def test_the_time_course_settles_onto_the_steady_state():
    out = solve(single_species(5.0, 1e-3, 1e-3, 1e-4))
    t, c = out.obs("time"), out.obs("concentrations")
    assert t[0] == 0 and c[0, 0] == 0.0                                  # starts empty
    assert np.all(np.diff(c[:, 0]) >= -1e-15)                            # fills monotonically
    assert c[-1, 0] == pytest.approx(out.obs("steady_S"), rel=1e-3)      # ten settling times: ~e^-10 left
    assert out.obs("max_eigenvalue") < 0
    assert out.obs("settling_time") == pytest.approx(1 / abs(out.obs("max_eigenvalue")), rel=1e-6)   # one metabolite: one mode


# -- thermodynamics at the pathway level -----------------------------------------------------------------
def equilibrium_network(x_end_over_x0):
    """X0 <-> S <-> P <-> X_end with a reversible enzyme in the middle; the enzyme's equilibrium constant is 10."""
    vf, ks, vr, kp = 4e-3, 1e-4, 4e-3 / 10 * (2e-4 / 1e-4), 2e-4
    keq = (vf / ks) / (vr / kp)
    structure = {
        "species": ["S", "P"], "flux_reaction": "enzyme",
        "reactions": [
            {"id": "transport", "law": "exchange", "stoich": {"S": 1}, "refs": {"a": "context.x0", "b": "S"}, "params": {"k": "context.k"}},
            {"id": "enzyme", "law": "reversible_mm", "stoich": {"S": -1, "P": 1}, "refs": {"s": "S", "p": "P"},
             "params": {"vf": "vf", "ks": "ks", "vr": "vr", "kp": "kp"}},
            {"id": "drain", "law": "exchange", "stoich": {"P": -1}, "refs": {"a": "P", "b": "context.x_end"}, "params": {"k": "context.k"}},
        ],
    }
    x0 = 1e-3
    params = {"context.x0": (x0, "M"), "context.x_end": (x_end_over_x0 * keq * x0, "M"), "context.k": (50.0, "1/s"),
              "vf": (vf, "M/s"), "ks": (ks, "M"), "vr": (vr, "M/s"), "kp": (kp, "M")}
    return network(structure, params), keq, x0


def test_a_pathway_at_the_enzymes_equilibrium_has_no_flux_and_reverses_beyond_it():
    system, keq, x0 = equilibrium_network(1.0)
    assert keq == pytest.approx(10.0)
    out = solve(system)
    assert abs(out.obs("pathway_flux")) < 1e-12                          # versus ~1e-3 M/s capacity
    assert out.obs("steady_S") == pytest.approx(x0, rel=1e-6)            # both pools equilibrate with their reservoirs
    assert out.obs("steady_P") == pytest.approx(keq * x0, rel=1e-6)
    assert not any(name.startswith("flux_control_") for name in out.observables)    # ln J is undefined at J = 0
    forward, backward = solve(equilibrium_network(0.5)[0]), solve(equilibrium_network(2.0)[0])
    assert forward.obs("pathway_flux") > 0 and backward.obs("pathway_flux") < 0
    for out in (forward, backward):                                      # control is defined, and sums to 1, in either direction
        controls = [out.obs(f"flux_control_{r}") for r in ("transport", "enzyme", "drain")]
        assert np.all(np.isfinite(controls)) and sum(controls) == pytest.approx(1.0, abs=1e-6)


# -- the rate law is the biophysical cycle, not an approximation of it ------------------------------------
def test_the_reversible_rate_law_reproduces_the_enzyme_cycles_exact_turnover():
    biophysical = EnzymeCycleEngine().solve(enzyme(k_f=1e5, k_r=1e4, k_on_product=3e6), BACKEND)
    o = biophysical.observables
    kcat_f, km_s, kcat_r, km_p = (o[n].value for n in ("kcat", "KM", "kcat_reverse", "KM_product"))
    assert kcat_f * km_p / (kcat_r * km_s) == pytest.approx(o["overall_equilibrium_constant"].value, rel=1e-9)   # Haldane
    rates = {n: biophysical.param(n) for n in ("k_f", "k_r", "context.k_on", "context.k_off", "context.k_release", "context.k_on_product")}
    cycle = _Cycle(biophysical.structure, rates)
    rng = np.random.default_rng(1)
    for _ in range(12):
        s, p = 10 ** rng.uniform(-7, -2), 10 ** rng.uniform(-7, -2)
        law = _reversible_mm({"s": s, "p": p}, {"vf": kcat_f, "ks": km_s, "vr": kcat_r, "kp": km_p})
        assert law == pytest.approx(cycle.turnover({"substrate": s, "product": p}), rel=1e-9)


# -- engine validation -------------------------------------------------------------------------------------
def test_engine_validation():
    base = lambda **p: single_species(5.0, 1e-3, 1e-3, 1e-4).evolve(**p)
    with pytest.raises(ValidationError, match="unknown rate law"):
        s = single_species(5.0, 1e-3, 1e-3, 1e-4)
        s.structure["reactions"][1]["law"] = "hill"
        solve(s)
    with pytest.raises(ValidationError, match="needs refs"):
        s = single_species(5.0, 1e-3, 1e-3, 1e-4)
        s.structure["reactions"][1]["refs"] = {"y": "S"}
        solve(s)
    with pytest.raises(ValidationError, match="unknown species"):
        s = single_species(5.0, 1e-3, 1e-3, 1e-4)
        s.structure["reactions"][1]["stoich"] = {"Q": -1}
        solve(s)
    with pytest.raises(ValidationError, match="flux_reaction"):
        s = single_species(5.0, 1e-3, 1e-3, 1e-4)
        s.structure["flux_reaction"] = "nope"
        solve(s)
    with pytest.raises(ValidationError, match="invalid value"):
        solve(single_species(5.0, 1e-3, 1e-3, 0.0))                       # K_M = 0 would divide by zero
    with pytest.raises(ValidationError, match="invalid value"):
        solve(single_species(-5.0, 1e-3, 1e-3, 1e-4))
    wrong = single_species(5.0, 1e-3, 1e-3, 1e-4)
    wrong.parameters["context.k"].unit = "M"
    with pytest.raises(ValidationError, match="expected '1/s'"):
        solve(wrong)
    with pytest.raises(ValidationError, match="one non-negative concentration per species"):
        wrong_shape = single_species(5.0, 1e-3, 1e-3, 1e-4)
        wrong_shape.state["c"] = Quantity([0.0, 0.0], "M")                  # two concentrations for one species
        solve(wrong_shape)
    with pytest.raises(ValidationError, match="solves biological"):
        MetabolicNetworkEngine().solve(evb(), BACKEND)


def test_an_irreversible_enzyme_feeding_a_drain_that_cannot_keep_up_has_no_steady_state():
    bp = EnzymeCycleEngine().solve(with_biology(enzyme()), BACKEND)                      # enzyme capacity > drain capacity
    out = BiophysicalToPathway().translate(bp)
    context = {k: q for k, q in bp.parameters.items() if k.startswith("context.")}
    with pytest.raises(ValidationError, match="no steady state reached"):
        solve(out.evolve(parameters={**out.parameters, **context}))


def test_a_network_that_never_settles_is_reported_not_returned():
    structure = {
        "species": ["S"], "flux_reaction": "influx",
        "reactions": [{"id": "influx", "law": "constant", "stoich": {"S": 1}, "refs": {}, "params": {"v0": "v0"}}],
    }
    with pytest.raises(ValidationError, match="no steady state reached"):
        solve(network(structure, {"v0": (1e-3, "M/s")}))                  # fills forever: nothing consumes S


def runaway_pathway(drain_vmax):
    """A pathway whose enzyme has the right Michaelis-Menten constants but a forward capacity ~10^6 times too large, with
    its reverse rate unchanged (so the enzyme is wildly out of equilibrium with its own thermodynamics)."""
    bp = EnzymeCycleEngine().solve(with_biology(enzyme(k_on_product=1e6), drain_vmax=drain_vmax), BACKEND)
    out = BiophysicalToPathway().translate(bp)
    context = {k: q for k, q in bp.parameters.items() if k.startswith("context.")}
    params = {**out.parameters, **context, "enzyme_vf": Quantity(bp.obs("kcat"), "M/s")}      # forgot to multiply by E_T
    return out.evolve(parameters=params)


def test_an_absurd_network_is_always_refused_within_a_bounded_amount_of_work_and_for_a_stated_reason(monkeypatch):
    # Without the guards these either stalled for minutes inside the integrator or "converged" to thousands of molar.
    # "Quickly" is asserted as bounded WORK (right-hand-side evaluations), not wall-clock seconds: a time limit on a machine
    # that is sometimes 2-3x slower is a flaky test, and the evaluation budget is the actual guarantee.
    from substrate.engines import biological
    calls = [0]
    original = biological._Network.rhs

    def counting(self, c, activity):
        calls[0] += 1
        return original(self, c, activity)

    monkeypatch.setattr(biological._Network, "rhs", counting)
    reasons = set()
    for drain_vmax in (1.6792e-3, 1.9e-3, 2.1e-3, 2.3e-3):
        calls[0] = 0
        system = runaway_pathway(drain_vmax)
        calls[0] = 0                                        # count only the refused solve, not building the system
        with pytest.raises(ValidationError) as exc:
            solve(system)
        assert calls[0] <= biological.INTEGRATION_BUDGET + 5_000          # the budget, plus the Newton polish and bookkeeping
        if "integration budget" in str(exc.value):
            reasons.add("budget")
        elif "above any physical limit" in str(exc.value):
            reasons.add("ceiling")
    assert reasons == {"budget", "ceiling"}                  # both guards are exercised, and nothing slipped through


def test_an_integrator_failure_is_a_refusal_not_a_crash():
    class GivesUp(ClassicalBackend):
        name = "gives-up"

        def integrate_ode(self, rhs, y0, t_eval):
            raise RuntimeError("ODE integration failed: Unexpected istate in LSODA.")

    with pytest.raises(ValidationError, match="the integrator failed"):
        MetabolicNetworkEngine().solve(single_species(5.0, 1e-3, 1e-3, 1e-4), GivesUp())


def test_an_unstable_steady_state_is_refused(monkeypatch):
    # None of the built-in laws can amplify a perturbation, so register one that does (dS/dt = +k S): S = 0 is a steady
    # state with a positive eigenvalue, and a control analysis there would describe a state the system cannot stay in.
    from substrate.engines import biological
    monkeypatch.setitem(biological._LAWS, "self_amplifying", (("x",), {"k": "1/s"}, lambda x, p: -p["k"] * x["x"]))
    structure = {
        "species": ["S"], "flux_reaction": "grow",
        "reactions": [{"id": "grow", "law": "self_amplifying", "stoich": {"S": -1}, "refs": {"x": "S"}, "params": {"k": "k"}}],
    }
    with pytest.raises(ValidationError, match="not stable"):
        solve(network(structure, {"k": (2.0, "1/s")}))


# -- the biophysical -> biological translator ------------------------------------------------------------------
def solved_enzyme(**kw):
    return EnzymeCycleEngine().solve(
        with_biology(enzyme(**{"k_on_product": 1e6, **kw}) if "k_on_product" not in kw else enzyme(**kw)), BACKEND)


def test_translation_maps_the_enzyme_kinetics_onto_the_pathway_rate_law():
    bp = EnzymeCycleEngine().solve(with_biology(enzyme(k_on_product=1e6)), BACKEND)
    out = BiophysicalToPathway().translate(bp)
    e_total = 1e-6
    assert out.param("enzyme_vf") == pytest.approx(bp.obs("kcat") * e_total)
    assert out.param("enzyme_ks") == pytest.approx(bp.obs("KM"))
    assert out.param("enzyme_vr") == pytest.approx(bp.obs("kcat_reverse") * e_total)
    assert out.param("enzyme_kp") == pytest.approx(bp.obs("KM_product"))
    assert out.structure["reactions"][1]["law"] == "reversible_mm" and out.structure["derivation"]["reversible"]
    assert out.structure == {**pathway_structure(True), "derivation": out.structure["derivation"]}


def test_an_enzyme_without_product_rebinding_becomes_irreversible_michaelis_menten():
    # an irreversible enzyme needs a drain that can keep up, or product piles up forever (see the next test's cousin)
    bp = EnzymeCycleEngine().solve(with_biology(enzyme(), drain_vmax=1.0), BACKEND)      # no k_on_product
    assert "kcat_reverse" not in bp.observables
    out = BiophysicalToPathway().translate(bp)
    assert out.structure["reactions"][1]["law"] == "michaelis_menten" and "enzyme_vr" not in out.parameters
    context = {k: q for k, q in bp.parameters.items() if k.startswith("context.")}     # what the pipeline carries across
    assert solve(out.evolve(parameters={**out.parameters, **context})).obs("pathway_flux") > 0


def test_translator_validation():
    translator = BiophysicalToPathway()
    bp = EnzymeCycleEngine().solve(with_biology(enzyme()), BACKEND)
    assert translator.validate(bp) == []
    assert "not been solved" in translator.validate(with_biology(enzyme()))[0].message
    missing = bp.evolve()
    del missing.parameters["context.drain_vmax"], missing.parameters["context.enzyme_total"]
    msg = translator.validate(missing)[0].message
    assert "context.drain_vmax" in msg and "context.enzyme_total" in msg
    negative = bp.evolve(parameters={**bp.parameters, "context.transport_rate": Quantity(-1.0, "1/s")})
    assert "must be positive" in translator.validate(negative)[0].message
    crowded = bp.evolve(parameters={**bp.parameters, "context.enzyme_total": Quantity(1.0, "M")})
    assert any("quasi-steady-state" in i.message for i in translator.validate(crowded))


# -- the whole chain ------------------------------------------------------------------------------------------------
def test_nine_stage_chain_from_electronic_structure_to_a_pathway(pipeline):
    r = pipeline.run(with_biology(evb()), ["biological"])
    assert [s.scale for s in r.trace] == [
        Scale.ELECTRONIC_STRUCTURE, Scale.QUANTUM, Scale.QUANTUM, Scale.REACTION, Scale.REACTION,
        Scale.BIOPHYSICAL, Scale.BIOPHYSICAL, Scale.BIOLOGICAL, Scale.BIOLOGICAL,
    ]
    previous = r.root
    for system in r.trace:
        record = system.provenance[-1]
        assert record.input_fingerprint == previous.fingerprint() and record.output_fingerprint == system.fingerprint()
        previous = system
    assert len(r.final.provenance) == 9
    for system in r.trace[1:]:                                           # context from every layer reaches every scale
        assert system.param("context.k_release", "1/s") == 1e4 and system.param("context.enzyme_total", "M") == 1e-6


def test_pathway_fluxes_are_balanced_bounded_and_controlled_in_total(pipeline):
    out = pipeline.run(with_biology(evb()), ["biological"]).final
    j = out.obs("pathway_flux")
    for reaction in ("transport", "enzyme", "drain"):
        assert out.obs(f"flux_{reaction}") == pytest.approx(j, rel=1e-6)    # one pathway, one flux
    capacity = out.param("enzyme_vf")
    assert 0 < j < capacity                                               # never faster than a saturated enzyme
    controls = [out.obs(f"flux_control_{r}") for r in ("transport", "enzyme", "drain")]
    assert sum(controls) == pytest.approx(1.0, abs=1e-6) and all(c > 0 for c in controls)
    assert out.obs("max_eigenvalue") < 0


def test_the_2d_electronic_kind_has_two_routes_to_a_pathway(pipeline):
    with pytest.raises(AmbiguousPathError, match="4-hop"):
        pipeline.plan(with_biology(evb2d()), ["biological"])
    assert len(pipeline.plan(with_biology(evb2d()), ROUTE_M)) == 9


# -- an isotope effect, followed from the molecule to the pathway flux -----------------------------------------------
ISOTOPE_BASE = dict(k_on=1e10, k_off=1e3, k_on_product=1e6, enzyme_total=1e-8, transport_rate=1e4, drain_vmax=10.0, drain_km=1e-3)


def propagate(pipeline, **regime):
    h, d = (pipeline.run(with_biology(evb2d(mass=m), **{**ISOTOPE_BASE, **regime}), ROUTE_M)
            for m in (1.007276, DEUTERON))
    return {
        "kie_kcat": h.trace[6].obs("kcat") / d.trace[6].obs("kcat"),
        "kie_flux": h.final.obs("pathway_flux") / d.final.obs("pathway_flux"),
        "c_enzyme": h.final.obs("flux_control_enzyme"),
        "saturation": h.final.obs("steady_S") / h.trace[6].obs("KM"),
    }


def test_the_isotope_effect_reaches_the_pathway_flux_only_if_every_layer_lets_it_through(pipeline):
    through = propagate(pipeline, external_substrate=1e-1, k_release=1e12)
    assert 8 < through["kie_kcat"] < 12                                    # the molecular scale's zero-point effect
    assert through["c_enzyme"] > 0.99 and through["saturation"] > 50       # the enzyme controls the flux and is saturated
    assert through["kie_flux"] == pytest.approx(through["kie_kcat"], rel=0.05)

    unsaturated = propagate(pipeline, external_substrate=1e-5, k_release=1e12)
    assert unsaturated["kie_kcat"] == pytest.approx(through["kie_kcat"], rel=0.01)   # same chemistry, same k_cat effect ...
    assert unsaturated["saturation"] < 0.1 and unsaturated["c_enzyme"] > 0.9         # ... the enzyme still controls the flux ...
    assert unsaturated["kie_flux"] < 1.3                                             # ... but the flux follows k_cat/K_M, not k_cat

    slow_release = propagate(pipeline, external_substrate=1e-1, k_release=1e3)
    assert slow_release["kie_kcat"] == pytest.approx(1.0, abs=0.02)        # release hides the chemistry inside the enzyme
    assert slow_release["kie_flux"] == pytest.approx(1.0, abs=0.02)

    transport_limited = propagate(pipeline, external_substrate=1e-1, k_release=1e12, transport_rate=1e-4, enzyme_total=1e-5)
    assert transport_limited["kie_kcat"] == pytest.approx(through["kie_kcat"], rel=0.01)   # the enzyme still has the effect ...
    assert transport_limited["c_enzyme"] < 0.01                                            # ... but does not control the pathway
    assert transport_limited["kie_flux"] == pytest.approx(1.0, abs=0.02)


# -- uncertainty and experiment files ------------------------------------------------------------------------------------
def test_biological_inputs_enter_the_uncertainty_ensemble(pipeline):
    root = with_biology(evb(), sigmas={"enzyme_total": 2e-7, "drain_vmax": 4e-4})
    r = pipeline.run(root, ["biological"], n_samples=16, seed=5)
    assert r.ensemble["n_ok"] == 16 and r.ensemble["varied"] == ["context.drain_vmax", "context.enzyme_total"]
    flux = r.final.observables["pathway_flux"]
    assert flux.sigma > 0 and flux.band[0] < flux.value < flux.band[1] and flux.sigma < flux.value
    assert r.final.observables["flux_control_enzyme"].sigma > 0           # control shifts as the enzyme is over/under-expressed


def test_the_pathway_example_experiment_runs_from_the_cli(capsys):
    from substrate.cli import main
    assert main(["run", str(EXPERIMENTS / "proton_transfer_pathway.yaml"), "--samples", "4"]) == 0
    out = capsys.readouterr().out
    assert "biological.metabolic_network" in out and "pathway_flux" in out and "flux_control_enzyme" in out
